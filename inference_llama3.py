#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference_llama3.py

This script performs inference using a Llama 3 model, likely fine-tuned or loaded
via the Unsloth library. It's a conversion of a typical Jupyter Notebook
workflow (e.g., inference-llama3.ipynb) into a standalone Python script.

Assumptions:
- You have installed the Unsloth library and its dependencies.
  (e.g., pip install "unsloth[llama] @ git+https://github.com/unslothai/unsloth.git")
- You have a CUDA-enabled GPU if using GPU acceleration.
"""

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import argparse
import gc  # For garbage collection
import psutil  # For memory monitoring
import logging

# Configuration for model loading (adjust as needed)
DEFAULT_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"  # Example: Llama 3 8B Instruct with 4-bit quantization
                                                    # Replace with your specific Llama 3 model if different
DEFAULT_MAX_SEQ_LENGTH = 4096  # Max sequence length for Llama 3 models (can be up to 8192 for some)
DEFAULT_DTYPE = None  # Autodetect. Can be torch.float16, torch.bfloat16, or torch.float32
# LOAD_IN_4BIT is now controlled by --load_in_4bit flag, defaulting to True

# Llama 3 Instruct prompt template
LLAMA3_INSTRUCT_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""" # Note: The assistant part is left for the model to fill.

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_system_requirements():
    """Check if system meets minimum requirements."""
    if not torch.cuda.is_available():
        logging.warning("CUDA not available. Running on CPU will be very slow.")
        return False
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    if gpu_memory < 8:
        logging.warning(f"GPU has only {gpu_memory:.1f}GB memory. Models might not load.")
    
    return True

def cleanup_gpu():
    """Free up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_model_and_tokenizer(model_name, max_seq_length, dtype, load_in_4bit):
    """Loads the Llama 3 model and tokenizer using Unsloth."""
    try:
        cleanup_gpu()
        print(f"Loading model: {model_name}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # token="hf_YOUR_TOKEN_HERE", # Uncomment and add your Hugging Face token if needed
        )
        print("Model and tokenizer loaded.")

        print("Preparing model for inference...")
        FastLanguageModel.for_inference(model) # Prepare model for inference
        print("Model prepared for inference.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        raise

def generate_response(model, tokenizer, instruction, max_seq_length_for_tokenizer):
    """Generates a response from the model given an instruction."""
    
    # Apply the Llama 3 Instruct prompt template
    prompt = LLAMA3_INSTRUCT_PROMPT_TEMPLATE.format(instruction)

    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_seq_length_for_tokenizer).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nInstruction: {instruction}")
    print("Generating response...")
    print("<|start_header_id|>assistant<|end_header_id|>") # Manually print the start of assistant turn

    text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False) # Keep special tokens for Llama 3
    
    # Generate text. Adjust generation parameters as needed.
    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        streamer=text_streamer,
        max_new_tokens=512,  # Max number of new tokens to generate
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id, # Llama 3 specific EOS
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # Handle pad_token_id
        # Common generation parameters:
        # temperature=0.6,
        # top_p=0.9,
        # do_sample=True,
    )
    
    _ = model.generate(**generation_kwargs)
    print("<|eot_id|>") # Manually print the end of turn token
    print("\n--- End of generation ---")

def main():
    setup_logging()
    check_system_requirements()
    
    parser = argparse.ArgumentParser(description="Perform inference using a Llama 3 model with Unsloth.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Name or path of the Llama 3 model to load.")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help="Maximum sequence length for the model.")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True,
                        help="Load the model in 4-bit quantization. Use --no-load_in_4bit to disable.")
    parser.add_argument("--instruction", type=str, nargs='+',
                        help="Instruction(s) to provide to the model. If not provided, example instructions will be used.")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode, prompting for instructions.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU. This will be very slow.")
        print("         Ensure PyTorch with CUDA support is installed and you have a compatible GPU.")

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model_name, args.max_seq_length, DEFAULT_DTYPE, args.load_in_4bit
        )

        if args.interactive:
            print("Entering interactive mode. Type 'exit' or 'quit' to end.")
            while True:
                instruction = input("Instruction: ")
                if instruction.lower() in ["exit", "quit"]:
                    break
                generate_response(model, tokenizer, instruction, args.max_seq_length)
                print("-" * 70)
        elif args.instruction:
            for instruction in args.instruction:
                generate_response(model, tokenizer, instruction, args.max_seq_length)
                print("-" * 70)
        else:
            # Example instructions to test the model
            instructions = [
                "What are the main differences between Llama 2 and Llama 3?",
                "Write a Python function that calculates the factorial of a number.",
                "Explain the concept of a Large Language Model to a 5-year-old.",
            ]
            for instruction in instructions:
                generate_response(model, tokenizer, instruction, args.max_seq_length)
                print("-" * 70)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return 1
    finally:
        cleanup_gpu()

if __name__ == "__main__":
    exit(main())