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
N_STEPS = 2

GLOBAL_BATCH_SIZE = 32
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

    # 3. Initialize Model
    model = Model(vocab=VOCAB_SIZE, dim=DIM, depth=DEPTH, N=N_STEPS, dropout=0.0)

    # --- THE FIX: DUMMY FORWARD PASS TO MATERIALIZE AXIOM WEIGHTS ---
    print("Materializing Axiom dynamic parameters...")

    # We create a tiny fake batch to push through the network.
    # Axiom only uses the batch/seq length for execution, but calculates all
    # parameter shapes based on the feature dimensions (which are already known).
    dummy_batch = jnp.ones((GLOBAL_BATCH_SIZE, 16), dtype=jnp.int32)

    with mesh:
        dummy_tensor = tensor(dummy_batch, ax_b, ax.sq).apply_sharding()
        _ = model(dummy_tensor)

    # NOW initialize the optimizer! Because the model has weights now,
    # the optimizer will trace them perfectly.
    schedule = optax.cosine_decay_schedule(
        init_value=LEARNING_RATE,
        decay_steps=MAX_STEPS,
        alpha=0.1
    )
    tx = optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=1e-1)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Split them together
    graphdef, state = nnx.split((model, optimizer))

    # 4. Setup Orbax Checkpoint Manager
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(
        GCS_CHECKPOINT_DIR,
        item_names=('model', 'optimizer', 'step'),
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
        # CRITICAL: Re-split after restoring so the loop gets the updated state!
        graphdef, state = nnx.split((model, optimizer))

    # 6. Define the GSPMD Training Step (The Pure NNX Way)
    @jax.jit
    def train_step(graphdef, state, batch_np):
        batch_global = batch_np.reshape(GLOBAL_BATCH_SIZE, -1)
        batch_tensor = tensor(batch_global, ax_b, ax.sq).apply_sharding()

        inputs = batch_tensor[..., ax.sq[:-1]]
        targets = batch_tensor[..., ax.sq[1:]]

        # Reconstruct BOTH the model and optimizer inside the JIT boundary
        model_, optimizer_ = nnx.merge(graphdef, state)

        def loss_fn(m):
            logits = m(inputs, use_checkpointing=True).apply_sharding()
            return optax.softmax_cross_entropy_with_integer_labels(logits.data, targets.data).mean()

        # Calculate gradients using the model reference
        loss, grads = nnx.value_and_grad(loss_fn)(model_)

        # This naturally mutates BOTH optimizer_ and model_ in-place!
        optimizer_.update(model_, grads)

        # Re-pack the updated state together
        new_state = nnx.state((model_, optimizer_))
        return new_state, loss

    # 7. Define Validation Step
    @jax.jit
    def val_step(graphdef, state, batch_np):
        batch_global = batch_np.reshape(GLOBAL_BATCH_SIZE, -1)
        batch_tensor = tensor(batch_global, ax_b, ax.sq).apply_sharding()

        inputs = batch_tensor[..., ax.sq[:-1]]
        targets = batch_tensor[..., ax.sq[1:]]

        # We only need the model for validation
        model_, _ = nnx.merge(graphdef, state)
        logits = model_(inputs)
        return optax.softmax_cross_entropy_with_integer_labels(logits.data, targets.data).mean()

    # 8. Initialize Dataloaders
    train_loader = DistributedTPULoader(DATASET_PATH, GLOBAL_BATCH_SIZE)
    # val_loader = DistributedTPULoader(VAL_DATASET_PATH, GLOBAL_BATCH_SIZE)

    # 9. The Training Loop
    print("Beginning training loop...")
    with mesh:
        for step, batch in enumerate(train_loader, start=start_step):
            if step >= MAX_STEPS:
                break

            # Now we only pass the single unified state!
            state, loss = train_step(graphdef, state, batch["input_ids"])

            # Logging
            if step % 10 == 0:
                print(f"Step {step} | Train Loss: {loss.item():.4f}")
                run.log({"train/loss": loss.item(), "step": step})

            # Checkpointing
            if step % SAVE_INTERVAL == 0 and step > start_step:
                print(f"Running evaluation & saving checkpoint to GCS at step {step}...")

                # Merge back to save out
                current_model, current_opt = nnx.merge(graphdef, state)

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