#!/usr/bin/env python3

import os
import math
import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx

from axiom import ax, Module, init, tensor
from loader import DistributedTPULoader


# =============================================================================
# Hardcoded config
# =============================================================================

DATASET_PATH = "/home/adam/datasets/fineweb-edu-mistral-4096"
VAL_DATASET_PATH = "/home/adam/datasets/fineweb-edu-val-4096"

VOCAB_SIZE = 32000

# Transformer config.
#
# NOTE:
# Your ResLM used DIM=1316, but attention needs DIM % HEADS == 0.
# 1344 / 16 = 84, and this lands in roughly the same model-size neighborhood.
DIM = 1056
DEPTH = 8
HEADS = 8
HEAD_DIM = DIM // HEADS

MLP_EXPANSION = 4
DROPOUT = 0.0
QK_NORM = False

GLOBAL_BATCH_SIZE = 32

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-1
MAX_STEPS = 76_500
WARMUP_STEPS = 1_000
GRAD_CLIP = 1.0

EVAL_INTERVAL = 100
EVAL_BATCHES = 10
SAVE_INTERVAL = 500
LOG_INTERVAL = 10

USE_CHECKPOINTING = True

WANDB_ENTITY = "adam-jacuch-stony-brook-university"
WANDB_PROJECT = "reslm-neurips"


# =============================================================================
# Transformer model
# =============================================================================

ROPE_BASE = 10_000.0


def apply_rope_bhsd(x, base: float = ROPE_BASE):
    """
    Apply RoPE to a raw JAX tensor with layout:

        [batch, heads, seq, head_dim]

    With DIM=1056 and HEADS=8, HEAD_DIM=132, so the whole head dimension
    is rotated cleanly in pairs.
    """
    seq_len = x.shape[-2]
    head_dim = x.shape[-1]

    if head_dim % 2 != 0:
        raise ValueError(
            f"RoPE requires an even head_dim, got head_dim={head_dim}"
        )

    half = head_dim // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    inv_freq = 1.0 / (
        base ** (
            jnp.arange(0, half, dtype=jnp.float32) / float(half)
        )
    )

    positions = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = positions[:, None] * inv_freq[None, :]

    cos = jnp.cos(freqs).astype(x.dtype)
    sin = jnp.sin(freqs).astype(x.dtype)

    # Broadcast [seq, half] over [batch, heads, seq, half].
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    return jnp.concatenate([y1, y2], axis=-1)


class CausalSelfAttention(Module):
    def __init__(self) -> None:
        self.dim = DIM
        self.heads = HEADS
        self.head_dim = HEAD_DIM

    def __call__(self, x):
        head_block = ax.h(self.heads) & ax.dh(self.head_dim)

        # Bias-free QKV, standard Transformer style.
        q = x[..., ax.d.proj(out=head_block, use_bias=False)]
        k = x[..., (ax.sq >> ax.sk), ax.d.proj(out=head_block, use_bias=False)]
        v = x[..., (ax.sq >> ax.sk), ax.d.proj(out=head_block, use_bias=False)]

        # Unpack [h&dh] into explicit [h, dh].
        q = q[
            ax.b,
            ax.sq,
            ax.h(self.heads) & ax.dh,
            "->",
            ax.b,
            ax.h,
            ax.sq,
            ax.dh,
        ]

        k = k[
            ax.b,
            ax.sk,
            ax.h(self.heads) & ax.dh,
            "->",
            ax.b,
            ax.h,
            ax.sk,
            ax.dh,
        ]

        v = v[
            ax.b,
            ax.sk,
            ax.h(self.heads) & ax.dh,
            "->",
            ax.b,
            ax.h,
            ax.sk,
            ax.dh,
        ]

        # RoPE on q/k only, before attention.
        q = tensor(
            apply_rope_bhsd(q.data),
            ax.b,
            ax.h,
            ax.sq,
            ax.dh,
        )

        k = tensor(
            apply_rope_bhsd(k.data),
            ax.b,
            ax.h,
            ax.sk,
            ax.dh,
        )

        if QK_NORM:
            q = q[..., ax.dh.norm_rms()]
            k = k[..., ax.dh.norm_rms()]

        # Axiom native causal attention.
        ctx = q[..., ax.sq.attend(keys=k, values=v, dim=ax.dh, is_causal=True)]

        # Repack heads and project back to residual dimension.
        ctx = ctx[..., "->", ax.b, ax.sq, ax.h & ax.dh]
        out = ctx[..., (ax.h & ax.dh).proj(out=ax.d(DIM), use_bias=False)]

        if DROPOUT > 0.0:
            out = out[..., ax.d.dropout(DROPOUT)]

        return out


class FeedForward(Module):
    def __init__(self) -> None:
        self.dim = DIM
        self.expansion = MLP_EXPANSION

    def __call__(self, x):
        d = ax.d(DIM)

        # SwiGLU halves the projected axis, so project to 2 * expansion * dim.
        ff_in = ax.ff(DIM * MLP_EXPANSION * 2)

        h = x[..., d.proj(out=ff_in).swiglu()]

        # Small init on the residual projection, matching your ResLM style.
        out = h[..., ax.ff.proj(out=d, kernel_init=init.normal(1e-4))]

        if DROPOUT > 0.0:
            out = out[..., ax.d.dropout(DROPOUT)]

        return out


class TransformerBlock(Module):
    def __init__(self) -> None:
        self.attn = CausalSelfAttention()
        self.ff = FeedForward()

    def __call__(self, x):
        x = x + self.attn(x[..., ax.d.norm_rms()])
        x = x + self.ff(x[..., ax.d.norm_rms()])
        return x


class TransformerLM(Module):
    def __init__(self) -> None:
        self.layers = nnx.List(TransformerBlock() for _ in range(DEPTH))

    def __call__(self, x, use_checkpointing: bool = False):
        x, w = x.embed(
            vocab=ax.v(VOCAB_SIZE),
            out=ax.d(DIM),
            return_weight=True,
        )

        if DROPOUT > 0.0:
            x = x[..., ax.d.dropout(DROPOUT)]

        for layer in self.layers:
            x = nnx.remat(layer)(x) if use_checkpointing else layer(x)

        x = x[..., ax.d.norm_rms()]

        # Tied readout.
        return x[..., ax.d.proj(
            out=ax.v(VOCAB_SIZE),
            weight=w,
            use_bias=False,
        )]

# =============================================================================
# Helpers
# =============================================================================

def count_params(model) -> int:
    params = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size for x in leaves if hasattr(x, "size")))


def replicate_all_state_to_mesh(state, mesh):
    replicated_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )

    def replicate_one(x):
        if isinstance(x, jax.Array):
            return jax.device_put(x, replicated_sharding)
        return x

    return jax.tree_util.tree_map(replicate_one, state)


def make_restarting_iterator(loader):
    iterator = iter(loader)

    def get_next():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    return get_next


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train hardcoded Axiom Transformer LM")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if DIM % HEADS != 0:
        raise ValueError(f"DIM={DIM} must be divisible by HEADS={HEADS}")

    GCS_CHECKPOINT_DIR = f"gs://adam-axiom-storage-europe/checkpoints/axiom-transformer/{args.run_id}"

    print(f"Initializing hardcoded Axiom Transformer on {jax.device_count()} TPU cores...")
    print(f"Config: dim={DIM}, depth={DEPTH}, heads={HEADS}, head_dim={HEAD_DIM}")
    print(f"Checkpoint dir: {GCS_CHECKPOINT_DIR}")

    if args.reset:
        print("WARNING: --reset flag detected. Starting a fresh run.")

    run = wandb.init(
        id=args.run_id,
        name=args.run_id,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        resume="allow" if not args.reset else None,
        config={
            "architecture": "Axiom Transformer",
            "dataset": "FineWeb-Edu 10BT",
            "vocab_size": VOCAB_SIZE,
            "dim": DIM,
            "depth": DEPTH,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "mlp_expansion": MLP_EXPANSION,
            "dropout": DROPOUT,
            "qk_norm": QK_NORM,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_steps": MAX_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "grad_clip": GRAD_CLIP,
            "use_checkpointing": USE_CHECKPOINTING,
        },
    )

    wandb.define_metric("*", step_metric="step")

    device_count = jax.device_count()
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((device_count,)),
        ("data",),
    )

    ax_b = ax.b(GLOBAL_BATCH_SIZE).shard("data")

    with mesh:
        model = TransformerLM()

        print("Materializing Axiom implicit parameters...")

        dummy_batch = jnp.ones((GLOBAL_BATCH_SIZE, 16), dtype=jnp.int32)
        dummy_tensor = tensor(dummy_batch, ax_b, ax.sq).apply_sharding()

        _ = model(dummy_tensor, use_checkpointing=False)

        param_count = count_params(model)
        print(f"Parameter count: {param_count:,}")
        run.log({"model/params": param_count, "step": 0})

        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=LEARNING_RATE,
            transition_steps=WARMUP_STEPS,
        )

        cosine_schedule = optax.cosine_decay_schedule(
            init_value=LEARNING_RATE,
            decay_steps=max(MAX_STEPS - WARMUP_STEPS, 1),
            alpha=0.1,
        )

        schedule = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[WARMUP_STEPS],
        )

        tx = optax.chain(
            optax.clip_by_global_norm(GRAD_CLIP),
            optax.adamw(
                learning_rate=schedule,
                b1=0.9,
                b2=0.95,
                weight_decay=WEIGHT_DECAY,
            ),
        )

        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    graphdef, state = nnx.split((model, optimizer))

    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)

    mngr = ocp.CheckpointManager(
        GCS_CHECKPOINT_DIR,
        item_names=("model", "optimizer", "step"),
        options=options,
    )

    start_step = 0

    if not args.reset and mngr.latest_step() is not None:
        latest = mngr.latest_step()
        print(f"Found checkpoint at step {latest}. Restoring...")

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

        with mesh:
            state = replicate_all_state_to_mesh(state, mesh)

        print(f"Restored from checkpoint. start_step={start_step}")

    @jax.jit
    def train_step(graphdef, state, batch_np):
        batch_global = batch_np.reshape(GLOBAL_BATCH_SIZE, -1)
        batch_tensor = tensor(batch_global, ax_b, ax.sq).apply_sharding()

        inputs = batch_tensor[..., ax.sq[:-1]]
        targets = batch_tensor[..., ax.sq[1:]]

        model_, optimizer_ = nnx.merge(graphdef, state)

        def loss_fn(m):
            logits = m(inputs, use_checkpointing=USE_CHECKPOINTING).apply_sharding()

            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.data,
                targets.data,
            ).mean()

            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model_)

        optimizer_.update(model_, grads)

        new_state = nnx.state((model_, optimizer_))

        return new_state, loss

    @jax.jit
    def val_step(graphdef, state, batch_np):
        batch_global = batch_np.reshape(GLOBAL_BATCH_SIZE, -1)
        batch_tensor = tensor(batch_global, ax_b, ax.sq).apply_sharding()

        inputs = batch_tensor[..., ax.sq[:-1]]
        targets = batch_tensor[..., ax.sq[1:]]

        model_, _ = nnx.merge(graphdef, state)

        logits = model_(inputs, use_checkpointing=USE_CHECKPOINTING).apply_sharding()

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.data,
            targets.data,
        ).mean()

        return loss

    train_loader = DistributedTPULoader(DATASET_PATH, GLOBAL_BATCH_SIZE)
    val_loader = DistributedTPULoader(VAL_DATASET_PATH, GLOBAL_BATCH_SIZE)

    next_train_batch = make_restarting_iterator(train_loader)
    next_val_batch = make_restarting_iterator(val_loader)

    if start_step > 0:
        print(f"Fast-forwarding train loader by {start_step} batches...")
        for _ in range(start_step):
            _ = next_train_batch()

    print("Beginning training loop...")

    with mesh:
        for step in range(start_step, MAX_STEPS):
            batch = next_train_batch()["input_ids"]

            state, loss = train_step(graphdef, state, batch)

            if step % LOG_INTERVAL == 0:
                loss_f = float(loss.item())
                ppl = math.exp(min(loss_f, 20.0))

                print(f"Step {step} | Train Loss: {loss_f:.4f} | PPL: {ppl:.2f}")

                run.log(
                    {
                        "train/loss": loss_f,
                        "train/perplexity": ppl,
                        "step": step,
                    }
                )

            if step % EVAL_INTERVAL == 0 and step > start_step:
                val_losses = []

                for _ in range(EVAL_BATCHES):
                    val_batch = next_val_batch()["input_ids"]
                    val_loss = val_step(graphdef, state, val_batch)
                    val_losses.append(float(val_loss.item()))

                val_loss_mean = float(np.mean(val_losses))
                val_ppl = math.exp(min(val_loss_mean, 20.0))

                print(f"Step {step} | Val Loss: {val_loss_mean:.4f} | Val PPL: {val_ppl:.2f}")

                run.log(
                    {
                        "val/loss": val_loss_mean,
                        "val/perplexity": val_ppl,
                        "step": step,
                    }
                )

            if step % SAVE_INTERVAL == 0 and step > start_step:
                print(f"Saving checkpoint to GCS at step {step}...")

                current_model, current_opt = nnx.merge(graphdef, state)

                mngr.save(
                    step,
                    args=ocp.args.Composite(
                        model=ocp.args.StandardSave(nnx.state(current_model)),
                        optimizer=ocp.args.StandardSave(nnx.state(current_opt)),
                        step=ocp.args.JsonSave(step),
                    ),
                )

    print("Training complete.")

    mngr.wait_until_finished()
    run.finish()


if __name__ == "__main__":
    main()