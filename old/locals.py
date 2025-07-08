from datasets import load_dataset, Dataset
import os
from datasets import load_from_disk
from pathlib import Path

token_path = Path.home() / ".cache" / "huggingface" / "token_write"
with open(token_path, "r") as f:
    token = f.read().strip()


# Load your dataset (adjust as needed)
ds = load_dataset("abehandlerorg/blockeddocs", split="train")

# Filter to only rows where 'url' contains "/local"
filtered_ds = ds.filter(lambda x: "/local/" in x["url"])

# Save using Hugging Face's internal format
filtered_ds.save_to_disk("localdocs")


from datasets import DatasetDict
dataset = load_from_disk("localdocs")
dataset.push_to_hub("abehandlerorg/localblockeddocs", token=token)
