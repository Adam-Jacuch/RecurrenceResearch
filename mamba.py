import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx

import gc

# External Mamba-2 JAX implementation
from mamba2_jax import Mamba2Config, Mamba2ForCausalLM
from loader import DistributedTPULoader  # Your provided loader

# --- Hyperparameters ---
VOCAB_SIZE = 32000
DIM = 1600
DEPTH = 8
GLOBAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 76_500
SAVE_INTERVAL = 500
EVAL_INTERVAL = 100
VAL_DATASET_PATH = "/home/adam/datasets/fineweb-edu-val-4096"  # Restored as a constant


def main():
    parser = argparse.ArgumentParser(description="Train Mamba-2 Baseline on Trillium")
    parser.add_argument("--reset", action="store_true", help="Start a fresh run")
    parser.add_argument("--run_id", type=str, required=True, help="Unique ID")
    parser.add_argument("--path", type=str, default="/home/adam/datasets/fineweb-edu-mistral-4096")
    parser.add_argument("--N", type=int, default=1, help="Recursive Depth (N=1 is single-pass)")
    args = parser.parse_args()

    # Configured for the Europe TPU!
    GCS_CHECKPOINT_DIR = f"gs://adam-axiom-storage/checkpoints/mamba2-baseline/{args.run_id}"

    # 1. Hardware Mesh Setup
    device_count = jax.device_count()
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape((device_count,)), ('data',))

    # Standard 2D Data Parallel Sharding: (batch, seq_len)
    dp_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data', None))

    # 2. Model Initialization
    with mesh:
        cfg = Mamba2Config(
            vocab_size=VOCAB_SIZE,
            hidden_size=DIM,
            num_hidden_layers=DEPTH,
            state_size=64,
            head_dim=64,
            expand=2,
            recursive_depth=args.N
        )
        model = Mamba2ForCausalLM(cfg, rngs=nnx.Rngs(1))

        schedule = optax.cosine_decay_schedule(LEARNING_RATE, MAX_STEPS, alpha=0.1)
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=0.1)
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    graphdef, state = nnx.split((model, optimizer))

    # 3. Setup Orbax
    mngr = ocp.CheckpointManager(
        GCS_CHECKPOINT_DIR,
        item_names=('model', 'optimizer', 'step'),
        options=ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    )

    # 4. Spot Preemption / Resume Logic
    start_step = 0
    if not args.reset and mngr.latest_step() is not None:
        print(f"Found existing checkpoint at step {mngr.latest_step()}. Resuming...")

        empty_model_state = nnx.state(model)
        empty_opt_state = nnx.state(optimizer)

        restored = mngr.restore(
            mngr.latest_step(),
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(empty_model_state),
                optimizer=ocp.args.StandardRestore(empty_opt_state),
                step=ocp.args.JsonRestore()
            )
        )

        start_step = restored['step']
        nnx.update(model, restored['model'])
        nnx.update(optimizer, restored['optimizer'])
        graphdef, state = nnx.split((model, optimizer))

    # 5. THE ULTIMATE SHARDING FIX (Replicating parameters across all cores)
    with mesh:
        replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        def replicate_all(x):
            if isinstance(x, jax.Array):
                return jax.device_put(x, replicated_sharding)
            return x

        state = jax.tree_util.tree_map(replicate_all, state)

    # 6. Training Logic
    @jax.jit
    def train_step(graphdef, state, batch_np):
        batch_global = batch_np.reshape(GLOBAL_BATCH_SIZE, -1)
        inputs_jax = jax.device_put(batch_global, dp_sharding)

        inputs = inputs_jax[:, :-1]
        targets = inputs_jax[:, 1:]

        model_, optimizer_ = nnx.merge(graphdef, state)

        def loss_fn(m):
            outputs = m(inputs, use_checkpointing=True)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs.logits

            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model_)
        optimizer_.update(model_, grads)

        return nnx.state((model_, optimizer_)), loss

    # --- Validation Step ---
    @jax.jit
    def val_step(graphdef, state, batch_np):
        batch_global = batch_np.reshape(GLOBAL_BATCH_SIZE, -1)
        inputs_jax = jax.device_put(batch_global, dp_sharding)

        inputs = inputs_jax[:, :-1]
        targets = inputs_jax[:, 1:]

        model_, _ = nnx.merge(graphdef, state)

        # Checkpointing is usually False for eval to speed it up,
        # but if you hit OOM issues during eval, you can flip this back to True.
        outputs = model_(inputs, use_checkpointing=False)

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs.logits

        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    # 7. Data Setup
    loader = DistributedTPULoader(args.path, GLOBAL_BATCH_SIZE)
    val_loader = DistributedTPULoader(VAL_DATASET_PATH, GLOBAL_BATCH_SIZE)

    train_iterator = iter(loader)

    if start_step > 0:
        print(f"Fast-forwarding dataloader by {start_step} batches...")
        for _ in range(start_step):
            next(train_iterator)

    # 8. Training Loop
    wandb.init(
        project="reslm-neurips",
        id=args.run_id,
        resume="allow" if not args.reset else None,
        config={"N": args.N, "depth": DEPTH, "dim": DIM}
    )
    wandb.define_metric("*", step_metric="step")

    print(f"🚀 Training Recursive Mamba-2 (N={args.N}) 180M on FineWeb-Edu...")
    with mesh:
        for step, batch in enumerate(train_iterator, start=start_step):
            if step >= MAX_STEPS: break

            state, loss = train_step(graphdef, state, batch["input_ids"])

            # --- Training Logging ---
            if step % 10 == 0:
                wandb.log({"train/loss": loss.item(), "step": step})

            # --- Validation Loop ---
            if step % EVAL_INTERVAL == 0:
                val_losses = []
                for _ in range(10):  # Evaluate over 10 batches to get a stable mean
                    val_batch = next(iter(val_loader))["input_ids"]
                    val_losses.append(val_step(graphdef, state, val_batch).item())

                val_loss_mean = np.mean(val_losses)
                val_ppl = np.exp(val_loss_mean)
                train_ppl = jnp.exp(loss)

                print(
                    f"Step {step} | Train Loss: {loss.item():.4f} (PPL: {train_ppl.item():.2f}) | Val Loss: {val_loss_mean:.4f} (PPL: {val_ppl:.2f})")

                wandb.log({
                    "val/loss": val_loss_mean,
                    "val/ppl": val_ppl,
                    "train/ppl": train_ppl.item(),
                    "step": step
                })

            # --- Checkpointing ---
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