"""
Standalone script to test loading gpt-oss model from local path
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/abe/dolma/scripts/R2/create/cpt/training/gpt-oss-20b-unsloth-bnb-4bit_cptllama-2024-01-29-Y0_debug"

print(f"Loading model from: {model_path}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="balanced_low_0",
    trust_remote_code=True,
)
print(f"✓ Model loaded: {type(model).__name__}")
print(f"Model config: {model.config}")
