#!/usr/bin/env python3
"""Quantize Qwen3-4B to GPTQ-Int4 for sm_70 (Volta) GPUs using llmcompressor"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import apply
import torch

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = "./Qwen3-4B-Instruct-2507-GPTQ-Int4-sm70"

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Loading model from {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# GPTQ recipe for sm_70 compatibility
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
)

# Calibration dataset
calibration_data = [
    "Hello, how are you today?",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, neural networks are computational systems.",
    "Please summarize the following text about climate change.",
    "What are the benefits of regular exercise for health?",
]

print("Quantizing model with GPTQ (this may take 20-60 minutes)...")
apply(
    model=model,
    tokenizer=tokenizer,
    dataset=calibration_data,
    recipe=recipe,
    output_dir=OUTPUT_DIR,
)

print(f"Done! Quantized model saved to {OUTPUT_DIR}")
