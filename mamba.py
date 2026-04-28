import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx

# External Mamba-2 JAX implementation
from mamba2_jax import Mamba2Config, Mamba2ForCausalLM
from loader import DistributedTPULoader  # Your provided loader

# --- Hyperparameters ---
VOCAB_SIZE = 32000
DIM = 1620
DEPTH = 8
GLOBAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_STEPS = 76_500
SAVE_INTERVAL = 500
EVAL_INTERVAL = 100


def main():
    parser = argparse.ArgumentParser(description="Train Mamba-2 Baseline on Trillium")
    parser.add_argument("--reset", action="store_true", help="Start a fresh run")
    parser.add_argument("--run_id", type=str, required=True, help="Unique ID")
    parser.add_argument("--path", type=str, default="/home/adam/datasets/fineweb-edu-mistral-4096")
    args = parser.parse_args()

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
            expand=2
        )
        model = Mamba2ForCausalLM(cfg, rngs=nnx.Rngs(0))

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
        # FIX: Flatten the loader's 3D array into 2D before applying standard sharding
        batch_global = batch_np.reshape(GLOBAL_BATCH_SIZE, -1)
        inputs_jax = jax.device_put(batch_global, dp_sharding)

        inputs = inputs_jax[:, :-1]
        targets = inputs_jax[:, 1:]

        model_, optimizer_ = nnx.merge(graphdef, state)

        def loss_fn(m):
            # Mamba2ForCausalLM expects (B, L)
            logits = m(inputs).logits
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model_)
        optimizer_.update(model_, grads)

        return nnx.state((model_, optimizer_)), loss

    # 7. Data Setup
    loader = DistributedTPULoader(args.path, GLOBAL_BATCH_SIZE)
    train_iterator = iter(loader)

    if start_step > 0:
        print(f"Fast-forwarding dataloader by {start_step} batches...")
        for _ in range(start_step):
            next(train_iterator)

    # 8. Training Loop
    wandb.init(
        project="reslm-neurips",
        id=args.run_id,
        resume="allow" if not args.reset else None
    )
    wandb.define_metric("*", step_metric="step")

    print(f"🚀 Training Mamba-2 180M on FineWeb-Edu...")
    with mesh:
        for step, batch in enumerate(train_iterator, start=start_step):
            if step >= MAX_STEPS: break

            state, loss = train_step(graphdef, state, batch["input_ids"])

            if step % EVAL_INTERVAL == 0:
                ppl = jnp.exp(loss)
                wandb.log({"train/loss": loss.item(), "train/ppl": ppl.item(), "step": step})
                print(f"Step {step} | Loss: {loss.item():.4f} | PPL: {ppl.item():.2f}")

            if step % SAVE_INTERVAL == 0 and step > start_step:
                print(f"Saving checkpoint to GCS at step {step}...")
                m_, o_ = nnx.merge(graphdef, state)
                mngr.save(step, args=ocp.args.Composite(
                    model=ocp.args.StandardSave(nnx.state(m_)),
                    optimizer=ocp.args.StandardSave(nnx.state(o_)),
                    step=ocp.args.JsonSave(step)
                ))

    print("Training complete.")
    mngr.wait_until_finished()
    wandb.finish()


if __name__ == "__main__":
    main()