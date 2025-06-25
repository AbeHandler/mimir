from datasets import load_dataset, Dataset
import os

# Load your dataset (adjust as needed)
ds = load_dataset("abehandlerorg/blockeddocs", split="train")

# Filter to only rows where 'url' contains "/local"
filtered_ds = ds.filter(lambda x: "/local/" in x["url"])

# Save using Hugging Face's internal format
filtered_ds.save_to_disk("localdocs")
