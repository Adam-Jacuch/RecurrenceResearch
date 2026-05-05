import argparse
import gc
import numpy as np

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx

from jax.sharding import PartitionSpec as P

try:
    from jax.experimental.shard_map import shard_map
except ImportError:
    from jax import shard_map

from recurrentgemma import jax as rg_jax
from loader import DistributedTPULoader


# --- Hyperparameters ---
VOCAB_SIZE = 32000
DIM = 1132
DEPTH = 8
GLOBAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 76_500
SAVE_INTERVAL = 500
EVAL_INTERVAL = 100
VAL_DATASET_PATH = "/home/adam/datasets/fineweb-edu-val-4096"


def extract_logits(outputs):
    """
    RecurrentGemma / bridge output format can vary slightly depending on version.
    This keeps the training loop robust.
    """
    if isinstance(outputs, dict):
        return outputs["logits"]

    if hasattr(outputs, "logits"):
        return outputs.logits

    if isinstance(outputs, tuple):
        return outputs[0]

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Train Hawk Baseline on Trillium Europe")
    parser.add_argument("--reset", action="store_true", help="Start a fresh run")
    parser.add_argument("--run_id", type=str, required=True, help="Unique ID")
    parser.add_argument(
        "--path",
        type=str,
        default="/home/adam/datasets/fineweb-edu-mistral-4096",
    )
    args = parser.parse_args()

    GCS_CHECKPOINT_DIR = (
        f"gs://adam-axiom-storage-europe/checkpoints/hawk-baseline/{args.run_id}"
    )

    # 1. Hardware Mesh Setup
    device_count = jax.device_count()

    if GLOBAL_BATCH_SIZE % device_count != 0:
        raise ValueError(
            f"GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE} must be divisible by "
            f"device_count={device_count}"
        )

    LOCAL_BATCH_SIZE = GLOBAL_BATCH_SIZE // device_count

    print(f"JAX devices: {device_count}")
    print(
        f"Using data-parallel shard_map: "
        f"global batch={GLOBAL_BATCH_SIZE}, local batch={LOCAL_BATCH_SIZE}"
    )

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((device_count,)),
        ("data",),
    )

    replicated_sharding = jax.sharding.NamedSharding(
        mesh,
        P(),
    )

    dp_sharding = jax.sharding.NamedSharding(
        mesh,
        P("data", None),
    )

    # 2. Model Initialization
    with mesh:
        print(f"Initializing Hawk (DIM={DIM}, DEPTH={DEPTH})...")

        cfg = rg_jax.GriffinConfig(
            vocab_size=VOCAB_SIZE,
            width=DIM,
            mlp_expanded_width=DIM * 3,
            num_heads=1,
            block_types=(rg_jax.TemporalBlockType.RECURRENT,) * DEPTH,
            embeddings_scale_by_sqrt_dim=True,
            attention_window_size=2048,
            logits_soft_cap=30.0,
        )

        # Keep False. Linen checkpointing can hide submodules from the NNX bridge.
        linen_hawk = rg_jax.Griffin(cfg, gradient_checkpointing=False)

        dummy_input = jnp.zeros((1, 128), dtype=jnp.int32)
        dummy_segment_pos = jnp.arange(128, dtype=jnp.int32)[None, :]

        model = nnx.bridge.ToNNX(
            linen_hawk,
            rngs=nnx.Rngs(1),
        ).lazy_init(
            dummy_input,
            segment_pos=dummy_segment_pos,
            return_cache=False,
        )

        params = nnx.state(model, nnx.Param)
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

        print("-" * 30)
        print(f"Total Parameters: {total_params:,}")
        print(f"Model Size:       {total_params / 1e6:.2f} M")
        print("-" * 30)

        schedule = optax.cosine_decay_schedule(
            LEARNING_RATE,
            MAX_STEPS,
            alpha=0.1,
        )

        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=0.1,
            ),
        )

        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    graphdef, state = nnx.split((model, optimizer))

    # 3. Setup Orbax
    mngr = ocp.CheckpointManager(
        GCS_CHECKPOINT_DIR,
        item_names=("model", "optimizer", "step"),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=2,
            create=True,
        ),
    )

    # 4. Resume Logic
    start_step = 0

    if not args.reset and mngr.latest_step() is not None:
        latest = mngr.latest_step()
        print(f"Found existing checkpoint at step {latest}. Resuming...")

        empty_model_state = nnx.state(model)
        empty_opt_state = nnx.state(optimizer)

        restored = mngr.restore(
            latest,
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(empty_model_state),
                optimizer=ocp.args.StandardRestore(empty_opt_state),
                step=ocp.args.JsonRestore(),
            ),
        )

        start_step = int(restored["step"])

        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])

        graphdef, state = nnx.split((model, optimizer))

    # 5. Replicate model/optimizer state across cores.
    #
    # Hawk/RecurrentGemma's recurrent layer uses a TPU Mosaic/Pallas kernel.
    # Therefore the model call must happen inside shard_map so the Pallas kernel
    # sees local per-device tensors rather than automatically partitioned tensors.
    with mesh:
        def replicate_all(x):
            if isinstance(x, jax.Array):
                return jax.device_put(x, replicated_sharding)
            return x

        state = jax.tree_util.tree_map(replicate_all, state)

    # 6. Training Step
    def train_step_impl(state, batch_global):
        """
        Runs on each data-parallel shard.

        batch_global is logically [GLOBAL_BATCH_SIZE, seq_len], but inside
        shard_map each device sees only [LOCAL_BATCH_SIZE, seq_len].
        """
        inputs = batch_global[:, :-1]
        targets = batch_global[:, 1:]

        seq_len = inputs.shape[1]

        segment_pos = jnp.broadcast_to(
            jnp.arange(seq_len, dtype=jnp.int32)[None, :],
            inputs.shape,
        )

        model_, optimizer_ = nnx.merge(graphdef, state)

        def loss_fn(m):
            outputs = m(
                inputs,
                segment_pos=segment_pos,
                return_cache=False,
            )

            logits = extract_logits(outputs)

            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits,
                targets,
            ).mean()

            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model_)

        # Average gradients and loss across data-parallel replicas.
        grads = jax.tree_util.tree_map(
            lambda x: jax.lax.pmean(x, "data") if isinstance(x, jax.Array) else x,
            grads,
        )
        loss = jax.lax.pmean(loss, "data")

        optimizer_.update(model_, grads)

        return nnx.state((model_, optimizer_)), loss

    train_step = jax.jit(
        shard_map(
            train_step_impl,
            mesh=mesh,
            in_specs=(
                P(),              # replicated state
                P("data", None),  # shard batch over data axis
            ),
            out_specs=(
                P(),  # replicated updated state
                P(),  # replicated scalar loss
            ),
            check_rep=False,
        )
    )

    # 7. Validation Step
    def val_step_impl(state, batch_global):
        """
        Runs on each data-parallel shard.
        """
        inputs = batch_global[:, :-1]
        targets = batch_global[:, 1:]

        seq_len = inputs.shape[1]

        segment_pos = jnp.broadcast_to(
            jnp.arange(seq_len, dtype=jnp.int32)[None, :],
            inputs.shape,
        )

        model_, _ = nnx.merge(graphdef, state)

        outputs = model_(
            inputs,
            segment_pos=segment_pos,
            return_cache=False,
        )

        logits = extract_logits(outputs)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits,
            targets,
        ).mean()

        loss = jax.lax.pmean(loss, "data")

        return loss

    val_step = jax.jit(
        shard_map(
            val_step_impl,
            mesh=mesh,
            in_specs=(
                P(),              # replicated state
                P("data", None),  # shard batch over data axis
            ),
            out_specs=P(),        # replicated scalar loss
            check_rep=False,
        )
    )

    # 8. Data Setup
    loader = DistributedTPULoader(args.path, GLOBAL_BATCH_SIZE)
    val_loader = DistributedTPULoader(VAL_DATASET_PATH, GLOBAL_BATCH_SIZE)

    train_iterator = iter(loader)
    val_iterator = iter(val_loader)

    if start_step > 0:
        print(f"Fast-forwarding dataloader by {start_step} batches...")
        for _ in range(start_step):
            next(train_iterator)

    # 9. WandB
    wandb.init(
        project="reslm-neurips",
        id=args.run_id,
        resume="allow" if not args.reset else None,
        config={
            "model": "hawk",
            "depth": DEPTH,
            "dim": DIM,
            "vocab_size": VOCAB_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "device_count": device_count,
            "learning_rate": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "mlp_expanded_width": DIM * 3,
            "num_heads": 1,
            "block_type": "all_recurrent",
            "parallelism": "data_parallel_shard_map",
        },
    )

    wandb.define_metric("*", step_metric="step")

    print(f"🚀 Training Hawk {total_params / 1e6:.2f}M on FineWeb-Edu...")

    # 10. Training Loop
    with mesh:
        for step, batch in enumerate(train_iterator, start=start_step):
            if step >= MAX_STEPS:
                break

            batch_ids = batch["input_ids"].reshape(GLOBAL_BATCH_SIZE, -1)
            batch_ids = jax.device_put(batch_ids, dp_sharding)

            state, loss = train_step(state, batch_ids)

            if step % 10 == 0:
                train_loss = float(jax.device_get(loss))
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "step": step,
                    }
                )

            if step % EVAL_INTERVAL == 0:
                val_losses = []

                for _ in range(10):
                    try:
                        val_batch = next(val_iterator)
                    except StopIteration:
                        val_iterator = iter(val_loader)
                        val_batch = next(val_iterator)

                    val_batch_ids = val_batch["input_ids"].reshape(
                        GLOBAL_BATCH_SIZE,
                        -1,
                    )
                    val_batch_ids = jax.device_put(val_batch_ids, dp_sharding)

                    val_loss = val_step(state, val_batch_ids)
                    val_losses.append(float(jax.device_get(val_loss)))

                val_loss_mean = float(np.mean(val_losses))
                val_ppl = float(np.exp(val_loss_mean))

                train_loss = float(jax.device_get(loss))
                train_ppl = float(np.exp(train_loss))

                print(
                    f"Step {step} | "
                    f"Train Loss: {train_loss:.4f} "
                    f"(PPL: {train_ppl:.2f}) | "
                    f"Val Loss: {val_loss_mean:.4f} "
                    f"(PPL: {val_ppl:.2f})"
                )

                wandb.log(
                    {
                        "val/loss": val_loss_mean,
                        "val/ppl": val_ppl,
                        "train/ppl": train_ppl,
                        "step": step,
                    }
                )

            if step % SAVE_INTERVAL == 0 and step > start_step:
                print(f"Saving checkpoint to GCS at step {step}...")

                loss.block_until_ready()

                m_, o_ = nnx.merge(graphdef, state)

                # Copy checkpoint payload to host RAM before Orbax writes it.
                model_state_to_save = jax.device_get(nnx.state(m_))
                opt_state_to_save = jax.device_get(nnx.state(o_))

                del m_, o_
                gc.collect()

                mngr.save(
                    step,
                    args=ocp.args.Composite(
                        model=ocp.args.StandardSave(model_state_to_save),
                        optimizer=ocp.args.StandardSave(opt_state_to_save),
                        step=ocp.args.JsonSave(int(step)),
                    ),
                )

                mngr.wait_until_finished()

                del model_state_to_save, opt_state_to_save
                gc.collect()

                print(f"Checkpoint {step} committed.")

    print("Training complete.")

    mngr.wait_until_finished()
    wandb.finish()


if __name__ == "__main__":
    main()