"""
Standalone script to test loading gpt-oss model from local path
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/abe/dolma/scripts/R2/create/cpt/training/gpt-oss-20b_cptllama-2024-01-29-Y0_debug"

print(f"Checking if model directory exists: {model_path}")
print(f"Exists: {os.path.exists(model_path)}")
print(f"Is directory: {os.path.isdir(model_path)}")

if os.path.exists(model_path):
    files = os.listdir(model_path)
    print(f"Files in directory: {files}")

print(f"\nAttempting to load from: {model_path}")

# Try with local_files_only flag
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="balanced_low_0",
    trust_remote_code=True,
    local_files_only=True,
)
print(f"✓ Model loaded: {type(model).__name__}")
print(f"Model config: {model.config}")
