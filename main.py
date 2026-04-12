import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx
from functools import partial

# Import custom modules
from axiom import ax, tensor  # Pulled in for Axiom v2 semantic sharding
from loader import DistributedTPULoader
from model import Model  # Your Axiom ResLM model

# --- Configuration ---
GCS_CHECKPOINT_DIR = "gs://adam-axiom-storage/checkpoints/reslm-180m"
DATASET_PATH = "gs://adam-axiom-storage/datasets/fineweb-edu-mistral-4096"
VAL_DATASET_PATH = "gs://adam-axiom-storage/datasets/fineweb-edu-val-4096"  # For NeurIPS eval

# 180M Parameter Config
VOCAB_SIZE = 32000
DIM = 768
DEPTH = 12
N_STEPS = 4

GLOBAL_BATCH_SIZE = 64
LEARNING_RATE = 3e-4
MAX_STEPS = 100_000
EVAL_INTERVAL = 500
SAVE_INTERVAL = 500


def main():
    print(f"Initializing ResLM Training on {jax.device_count()} TPU cores...")

    # 1. Initialize W&B with Spot Preemption Resilience
    run = wandb.init(
        entity="adam-jacuch-stony-brook-university",
        project="reslm-neurips",
        config={
            "lr": LEARNING_RATE,
            "architecture": "ResLM (Axiom v2)",
            "dataset": "FineWeb-Edu 10BT",
            "batch_size": GLOBAL_BATCH_SIZE,
            "max_steps": MAX_STEPS
        },
        resume="allow"  # CRITICAL: Tells W&B to append to the same run if the TPU reboots
    )

    # 2. Define the Global Hardware Mesh for GSPMD
    device_count = jax.device_count()
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape((device_count,)), ('data',))
    ax_b = ax.b(GLOBAL_BATCH_SIZE).shard("data")

    # 3. Initialize Model & Optimizer
    model = Model(vocab=VOCAB_SIZE, dim=DIM, depth=DEPTH, N=N_STEPS, dropout=0.0)

    schedule = optax.cosine_decay_schedule(
        init_value=LEARNING_RATE,
        decay_steps=MAX_STEPS,
        alpha=0.1
    )
    tx = optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=1e-1)
    optimizer = nnx.Optimizer(model, tx)

    # 4. Setup Orbax Checkpoint Manager (Tracking 'step' for W&B syncing)
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(
        os.path.abspath(GCS_CHECKPOINT_DIR),
        item_names=('model', 'optimizer', 'step'),  # Added step tracking
        options=options
    )

    # 5. Resume from Spot Preemption
    start_step = 0
    if mngr.latest_step() is not None:
        print(f"Found existing checkpoint at step {mngr.latest_step()}. Resuming...")
        mngr.restore(
            mngr.latest_step(),
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(model),
                optimizer=ocp.args.StandardRestore(optimizer),
                step=ocp.args.JsonRestore()
            )
        )
        start_step = mngr.latest_step()

    # 6. Define the GSPMD Training Step (NO pmap!)
    @jax.jit
    def train_step(graphdef, state, opt_state, batch_np):
        # The loader yields (devices, local_batch, seq). Flatten to global for GSPMD.
        batch_global = batch_np.reshape(GLOBAL_BATCH_SIZE, -1)

        # Wrap in AxiomTensor and apply hardware constraints
        batch_tensor = tensor(batch_global, ax_b, ax.sq).apply_sharding()

        # Autoregressive shift
        inputs = batch_tensor[..., :-1]
        targets = batch_tensor[..., 1:]

        def loss_fn(state_):
            model_ = nnx.merge(graphdef, state_)

            # Forward pass inherently respects the semantic sharding
            logits = model_(inputs).apply_sharding()

            loss = optax.softmax_cross_entropy_with_integer_labels(logits.data, targets.data).mean()
            return loss, model_

        # Calculate & Apply gradients
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, model_), grads = grad_fn(state)
        opt_state_ = optimizer.update(opt_state, grads)

        new_graphdef, new_state = nnx.split(model_)
        return new_state, opt_state_, loss

    # 7. Define Validation Step (For NeurIPS generalization proof)
    @jax.jit
    def val_step(graphdef, state, batch_np):
        batch_global = batch_np.reshape(GLOBAL_BATCH_SIZE, -1)
        batch_tensor = tensor(batch_global, ax_b, ax.sq).apply_sharding()

        inputs = batch_tensor[..., :-1]
        targets = batch_tensor[..., 1:]

        model_ = nnx.merge(graphdef, state)
        logits = model_(inputs)
        return optax.softmax_cross_entropy_with_integer_labels(logits.data, targets.data).mean()

    # Extract initial state
    graphdef, state = nnx.split(model)
    opt_state = nnx.split(optimizer)[1]

    # 8. Initialize Dataloaders
    train_loader = DistributedTPULoader(DATASET_PATH, GLOBAL_BATCH_SIZE)
    # val_loader = DistributedTPULoader(VAL_DATASET_PATH, GLOBAL_BATCH_SIZE)

    # 9. The Training Loop (Executed within the Mesh context)
    print("Beginning training loop...")
    with mesh:
        for step, batch in enumerate(train_loader, start=start_step):
            if step >= MAX_STEPS:
                break

            state, opt_state, loss = train_step(graphdef, state, opt_state, batch["input_ids"])

            # Logging
            if step % 10 == 0:
                print(f"Step {step} | Train Loss: {loss.item():.4f}")
                run.log({"train/loss": loss.item(), "step": step})

            # Validation & Asynchronous checkpointing
            if step % SAVE_INTERVAL == 0 and step > start_step:
                print(f"Running evaluation & saving checkpoint to GCS at step {step}...")

                # --- Quick Val Loop (Uncomment when val_loader is ready) ---
                # val_losses = [val_step(graphdef, state, next(val_loader)["input_ids"]).item() for _ in range(10)]
                # val_loss_mean = np.mean(val_losses)
                # print(f"Step {step} | Val Loss: {val_loss_mean:.4f}")
                # run.log({"val/loss": val_loss_mean, "step": step})

                current_model = nnx.merge(graphdef, state)
                current_opt = nnx.merge(optimizer.graphdef, opt_state)

                mngr.save(
                    step,
                    args=ocp.args.Composite(
                        model=ocp.args.StandardSave(current_model),
                        optimizer=ocp.args.StandardSave(current_opt),
                        step=ocp.args.JsonSave(step)
                    )
                )

    print("Training complete.")
    mngr.wait_until_finished()
    run.finish()


if __name__ == "__main__":
    main()