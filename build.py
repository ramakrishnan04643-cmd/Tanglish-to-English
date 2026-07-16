import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
LOCAL_DIR = "./model"  # Saves files inside your project folder

print(f"Downloading model directly to local folder: {LOCAL_DIR}...")
AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=LOCAL_DIR)
AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=LOCAL_DIR)
print("Download complete!")