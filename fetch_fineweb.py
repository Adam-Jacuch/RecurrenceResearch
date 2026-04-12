import os
import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Configuration ---
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SLICE = "sample-10BT"  # Adjust slice as needed (e.g., sample-10BT, sample-100BT)
GCS_DESTINATION = "gs://adam-axiom-storage/datasets/fineweb-edu-mistral-4096"

# Model specific config
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"  # Standard 32k vocab Mistral tokenizer
SEQUENCE_LENGTH = 4096
NUM_PROC = multiprocessing.cpu_count()


def main():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Fetching dataset: {DATASET_NAME} ({DATASET_SLICE})")
    # Streaming = False because we want to process it fully before uploading
    dataset = load_dataset(DATASET_NAME, name=DATASET_SLICE, split="train")

    # 1. Tokenization Phase
    # We only need the input_ids. We drop the raw text to save memory.
    def tokenize_fn(example):
        return tokenizer(example["text"], add_special_tokens=True)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=dataset.column_names,
        desc="Running tokenizer"
    )

    # 2. Packing Phase
    # Concatenate all tokens and chunk them into exact blocks of SEQUENCE_LENGTH.
    # This ensures your Axiom model never wastes compute on <PAD> tokens.
    def group_texts(examples):
        # Use a list comprehension to flatten in O(N) time
        concatenated_examples = {
            k: [item for sublist in examples[k] for item in sublist]
            for k in examples.keys()
        }

        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= SEQUENCE_LENGTH:
            total_length = (total_length // SEQUENCE_LENGTH) * SEQUENCE_LENGTH

        result = {
            k: [t[i: i + SEQUENCE_LENGTH] for i in range(0, total_length, SEQUENCE_LENGTH)]
            for k, t in concatenated_examples.items()
        }
        return result

    print(f"Packing into sequences of {SEQUENCE_LENGTH}...")
    packed_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=NUM_PROC,
        desc=f"Packing sequences"
    )

    # 3. Upload Phase
    print(f"Saving directly to GCS bucket: {GCS_DESTINATION}")
    # gcsfs handles the gs:// routing automatically
    packed_dataset.save_to_disk(
        GCS_DESTINATION,
        storage_options={"token": "google_default"}
    )

    print(f"Done! Dataset is ready for TPU consumption.")


if __name__ == "__main__":
    main()