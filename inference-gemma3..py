#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference_gemma3.py

This script performs inference using a Gemma model, likely fine-tuned or loaded
via the Unsloth library. It's a conversion of a typical Jupyter Notebook
workflow (e.g., inference-gemma3.ipynb) into a standalone Python script.

Assumptions:
- You have installed the Unsloth library and its dependencies (e.g., PyTorch, Transformers).
  (e.g., pip install "unsloth[gemma] @ git+https://github.com/unslothai/unsloth.git")
- You have a CUDA-enabled GPU if using GPU acceleration.
- If you are using features from unsloth_zoo that involve GGUF conversion,
  it's assumed that unsloth_zoo is patched (as per unsloth_zoo.patch) and
  a local `llama.cpp` directory with `convert_hf_to_gguf.py` is present.
"""

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Configuration for model loading (adjust as needed)
MODEL_NAME = "unsloth/gemma-2b-it-bnb-4bit"  # Example: Gemma 2B instruction-tuned with 4-bit quantization
                                            # Replace with your specific Gemma model if different
MAX_SEQ_LENGTH = 2048  # Max sequence length for the model
DTYPE = None  # Autodetect. Can be torch.float16, torch.bfloat16, or torch.float32
LOAD_IN_4BIT = True  # Set to True to load in 4-bit for reduced memory, False for full precision (if model supports)

# Alpaca prompt template (common for instruction-tuned models)
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def load_model_and_tokenizer():
    """Loads the model and tokenizer using Unsloth."""
    print(f"Loading model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        # token="hf_YOUR_TOKEN_HERE", # Uncomment and add your Hugging Face token if needed for gated models
    )
    print("Model and tokenizer loaded.")

    print("Preparing model for inference...")
    FastLanguageModel.for_inference(model)
    print("Model prepared for inference.")
    return model, tokenizer

def generate_response(model, tokenizer, instruction):
    """Generates a response from the model given an instruction."""
    prompt = ALPACA_PROMPT_TEMPLATE.format(
        instruction,
        "",  # Output - leave this blank for generation
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nInstruction: {instruction}")
    print("Generating response...")
    print("### Response:")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    # Generate text. Adjust generation parameters as needed.
    # Common parameters: max_new_tokens, temperature, top_p, top_k, do_sample
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,  # Max number of new tokens to generate
        use_cache=True,
        # temperature=0.7,
        # top_p=0.9,
        # do_sample=True,
    )
    print("\n--- End of generation ---")

def main():
    """Main function to run the inference."""
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU. This will be very slow.")
        print("         Ensure PyTorch with CUDA support is installed and you have a compatible GPU.")

    model, tokenizer = load_model_and_tokenizer()

    # Example instructions to test the model
    instructions = [
        "What is the capital of France?",
        "Write a short story about a robot who dreams of becoming a chef.",
        "Explain the concept of black holes in simple terms.",
    ]

    for instruction in instructions:
        generate_response(model, tokenizer, instruction)
        print("-" * 50)

if __name__ == "__main__":
    main()