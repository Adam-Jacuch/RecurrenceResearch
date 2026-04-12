import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_from_disk
from flax.jax_utils import prefetch_to_device
from typing import Iterator, Dict


class DistributedTPULoader:
    def __init__(
            self,
            dataset_path: str,
            global_batch_size: int,
            seed: int = 42
    ):
        self.global_batch_size = global_batch_size
        self.num_devices = jax.device_count()

        if self.global_batch_size % self.num_devices != 0:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible "
                f"by the number of TPU devices ({self.num_devices})."
            )

        self.local_batch_size = self.global_batch_size // self.num_devices

        # 1. Load the pre-packed dataset straight from GCS
        print(f"Loading dataset from {dataset_path}...")
        self.dataset = load_from_disk(dataset_path)

        # 2. Set format to NumPy (Executes on CPU, avoids JAX dispatch overhead)
        self.dataset.set_format(type="numpy", columns=["input_ids"])

        self.total_samples = len(self.dataset)
        self.num_batches = self.total_samples // self.global_batch_size
        self.seed = seed
        self.epoch = 0

    def _get_iterator(self) -> Iterator[Dict[str, jax.Array]]:
        # Deterministic shuffling using NumPy's RNG for the CPU-side data shuffling
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = rng.permutation(self.total_samples)

        for i in range(self.num_batches):
            batch_indices = indices[i * self.global_batch_size: (i + 1) * self.global_batch_size]
            batch = self.dataset[batch_indices]

            # Extract the raw numpy array
            inputs = batch["input_ids"]

            # Reshape for TPU Data Parallelism: (num_devices, local_batch_size, sequence_length)
            # This is exactly what jax.pmap or nnx.vmap needs to map across cores.
            sharded_inputs = inputs.reshape(
                self.num_devices,
                self.local_batch_size,
                inputs.shape[-1]
            )

            yield {"input_ids": sharded_inputs}

        self.epoch += 1

    def __iter__(self):
        # 3. Wrap the generator in Flax's asynchronous prefetcher
        # This pushes the next batch to TPU HBM (High Bandwidth Memory)
        # while the current batch is processing, hiding CPU latency.
        return prefetch_to_device(self._get_iterator(), size=2)