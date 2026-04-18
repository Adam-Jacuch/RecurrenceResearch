import numpy as np
from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
import csv
import os
import gc

from model import Model
from axiom import ax, tensor, Module, init


# --- CONFIGURATION ---
@dataclass
class MQARConfig:
    vocab_size: int = 128
    seq_len: int = 128
    num_kv_pairs: int = 32
    num_queries: int = 16
    random_non_query_token: bool = True
    seed: int = 42

    sep_token: int = 1
    pad_token: int = 0


# --- PURE NUMPY DATASET ---
class MQARDataset:
    def __init__(self, config: MQARConfig, batch_size: int):
        self.c = config
        self.bs = batch_size
        self.rng = np.random.default_rng(self.c.seed)

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        c = self.c

        f_end = c.vocab_size // 3
        k_end = 2 * c.vocab_size // 3
        v_min = k_end
        key_space = k_end - f_end

        assert c.num_kv_pairs <= key_space, "Not enough unique keys in key band; increase vocab_size."

        keys = np.array([
            self.rng.choice(np.arange(f_end, k_end), size=c.num_kv_pairs, replace=False)
            for _ in range(self.bs)
        ])

        values = self.rng.integers(v_min, c.vocab_size, size=(self.bs, c.num_kv_pairs))

        if c.random_non_query_token:
            x = self.rng.integers(2, f_end, size=(self.bs, c.seq_len))
        else:
            x = np.full((self.bs, c.seq_len), c.pad_token, dtype=np.int32)

        y = np.full((self.bs, c.seq_len), -100, dtype=np.int32)

        required_len = (c.num_kv_pairs * 2) + 1 + (c.num_queries * 2)
        assert required_len <= c.seq_len

        sep_pos = c.seq_len - (c.num_queries * 2) - 1
        x[:, sep_pos] = c.sep_token

        max_even_slots = sep_pos // 2
        assert c.num_kv_pairs <= max_even_slots

        for b in range(self.bs):
            starts = self.rng.choice(np.arange(max_even_slots), size=c.num_kv_pairs, replace=False) * 2
            x[b, starts] = keys[b]
            x[b, starts + 1] = values[b]

        query_indices = self.rng.integers(0, c.num_kv_pairs, size=(self.bs, c.num_queries))
        for q in range(c.num_queries):
            kv_idx = query_indices[:, q]
            key_q = keys[np.arange(self.bs), kv_idx]
            val_q = values[np.arange(self.bs), kv_idx]

            key_pos = sep_pos + 1 + 2 * q

            x[:, key_pos] = key_q
            x[:, key_pos + 1] = val_q
            y[:, key_pos] = val_q

        return x, y

    def __iter__(self):
        while True:
            yield self.generate_batch()


def train_and_log(steps: int, model_name: str, model, config: MQARConfig, batch_size: int = 32):
    # 1. Setup Optax Optimizer
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=2e-3,
        warmup_steps=1000,
        decay_steps=steps,
        end_value=1e-5
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.0)
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # 2. Setup Orbax Checkpoint Manager
    checkpoint_dir = f"gs://adam-axiom-storage/checkpoints/mqar/{model_name}"
    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_names=('model', 'optimizer', 'step'),
        options=options
    )

    # 3. Resume Logic
    start_step = 0
    latest_step = mngr.latest_step()

    if latest_step is not None:
        print(f"🔄 Found checkpoint at step {latest_step} for {model_name}. Resuming...")

        empty_model_state = nnx.state(model)
        empty_opt_state = nnx.state(optimizer)

        restored = mngr.restore(
            latest_step,
            args=ocp.args.Composite(
                model=ocp.args.StandardRestore(empty_model_state),
                optimizer=ocp.args.StandardRestore(empty_opt_state),
                step=ocp.args.JsonRestore()
            )
        )

        nnx.update(model, restored['model'])
        nnx.update(optimizer, restored['optimizer'])
        start_step = restored['step']

    # 4. JIT Train Step
    @nnx.jit
    def train_step(model, optimizer, x, y):
        def loss_fn(m):
            x_tensor = tensor(x, ax.b, ax.sq)
            logits_tensor = m(x_tensor)
            logits = logits_tensor.data

            valid_mask = (y != -100)
            labels = jnp.where(valid_mask, y, 0)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            loss = jnp.where(valid_mask, loss, 0.0)
            mean_loss = jnp.sum(loss) / jnp.maximum(jnp.sum(valid_mask), 1.0)
            return mean_loss, logits

        (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)

        valid_mask = (y != -100)
        preds = jnp.argmax(logits, axis=-1)
        correct = (preds == y) & valid_mask
        acc = jnp.sum(correct) / jnp.maximum(jnp.sum(valid_mask), 1.0)
        return loss, acc

    # 5. Dataloader Fast-Forwarding
    dataset = MQARDataset(config, batch_size=batch_size)
    data_iter = iter(dataset)

    if start_step > 0:
        print(f"⏩ Fast-forwarding dataset by {start_step} batches to maintain sequence parity...")
        for _ in range(start_step):
            next(data_iter)

    acc_hist = []

    # 6. CSV Setup (Append if resuming, Write fresh if starting)
    os.makedirs("results", exist_ok=True)
    csv_file = f"results/{model_name}.csv"
    base_headers = ["step", "loss", "avg_acc"]

    if start_step == 0:
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(base_headers)

    print(f"\n🚀 Starting/Resuming Run: {model_name} from Step {start_step}")
    print(f"Task: Seq {config.seq_len} | KV {config.num_kv_pairs} | Vocab {config.vocab_size}")

    # 7. Training Loop
    SAVE_INTERVAL = 10_000

    for i in range(start_step, steps):
        x_np, y_np = next(data_iter)
        x_jax, y_jax = jnp.array(x_np), jnp.array(y_np)

        loss, acc = train_step(model, optimizer, x_jax, y_jax)

        acc_hist.append(float(acc))
        if len(acc_hist) > 20:
            acc_hist.pop(0)

        # Logging
        if i % 100 == 0 and i > start_step:
            avg_acc = sum(acc_hist) / len(acc_hist)
            row_data = [i, float(loss), float(avg_acc)]

            if i % 1000 == 0:
                print(f"Step {i:05d} | Loss: {loss:.4f} | Acc: {avg_acc * 100:.1f}%")

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)

        # Checkpointing
        if i % SAVE_INTERVAL == 0 and i > start_step:
            print(f"💾 Saving checkpoint to GCS at step {i}...")
            mngr.save(
                i,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(nnx.state(model)),
                    optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
                    step=ocp.args.JsonSave(i)
                )
            )

    # Final save at the very end
    if steps % SAVE_INTERVAL != 0:
        mngr.save(
            steps,
            args=ocp.args.Composite(
                model=ocp.args.StandardSave(nnx.state(model)),
                optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
                step=ocp.args.JsonSave(steps)
            )
        )

    mngr.wait_until_finished()
    print(f"✅ Finished {model_name}. Max Acc: {max(acc_hist) * 100:.1f}%\n")
    return


if __name__ == "__main__":
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')

    tasks = {
        "mqar_easy": MQARConfig(num_kv_pairs=64, seq_len=1024, num_queries=32, vocab_size=256),
        "mqar_med": MQARConfig(num_kv_pairs=128, seq_len=2048, num_queries=64, vocab_size=512),
        "mqar_hard": MQARConfig(num_kv_pairs=256, seq_len=4096, num_queries=128, vocab_size=1024)
    }

    model_configs = {
        "15M": [(576, 1), (512, 2), (464, 3)],
        "30M": [(832, 1), (736, 2), (672, 3)],
        "60M": [(1168, 1), (1040, 2), (940, 3)]
    }

    STEPS = 250_000

    for task_name, conf in tasks.items():
        for scale_name, configs in model_configs.items():
            for dim, n in configs:
                run_name = f"{task_name}_{scale_name}_N{n}_dim{dim}"

                model = Model(vocab=conf.vocab_size, dim=dim, depth=4, N=n, dropout=0.0)

                dummy_x = tensor(jnp.ones((1, conf.seq_len), dtype=jnp.int32), ax.b, ax.sq)
                _ = model(dummy_x, use_checkpointing=False)

                train_and_log(steps=STEPS, model_name=run_name, model=model, config=conf)

                del model
                dummy_x = None
                gc.collect()
                jax.clear_caches()