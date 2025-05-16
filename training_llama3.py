#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_llama3.py

This script fine-tunes a Llama 3 model using the Unsloth library and TRL's SFTTrainer.
It's a conversion of a typical Jupyter Notebook workflow (e.g., training-llama3.ipynb)
into a standalone Python script.

Assumptions:
- You have installed Unsloth, TRL, Datasets, Accelerate, and their dependencies.
  (e.g., pip install "unsloth[llama] @ git+https://github.com/unslothai/unsloth.git")
  (e.g., pip install trl datasets accelerate bitsandbytes)
- You have a CUDA-enabled GPU for efficient training.
- Your dataset is prepared and accessible.
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset # To load datasets
import logging
import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TrainingConfig:
    base_model_name: str
    max_seq_length: int
    output_dir: str
    num_train_epochs: int
    learning_rate: float
    dataset_path: str
    batch_size: int
    gradient_accumulation: int
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.max_seq_length <= 8192, "max_seq_length must be ≤ 8192"
        assert self.num_train_epochs > 0, "num_train_epochs must be positive"
        assert os.path.exists(self.dataset_path), f"Dataset path {self.dataset_path} not found"
        # Add more validation as needed

class TrainingMonitor:
    def __init__(self):
        self.start_time = None
        self.training_steps = 0
    
    def log_memory_usage(self):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            logging.info(f"GPU Memory: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")

# Configuration (Adjust these parameters as needed)

# --- Model Configuration ---
BASE_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"  # Base Llama 3 model
MAX_SEQ_LENGTH = 2048  # Max sequence length. Llama 3 can handle up to 8192, but 2048/4096 is common for fine-tuning.
DTYPE = None
LOAD_IN_4BIT = True

# --- LoRA Configuration (if using LoRA) ---
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [ # Common for Llama 3
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_BIAS = "none"

# --- Dataset Configuration ---
DATASET_NAME_OR_PATH = "your_dataset_path_or_name" # Replace with your dataset
DATASET_TEXT_FIELD = "text" # The column in your dataset that contains the formatted text for training
                            # This 'text' field should ideally contain the full prompt and response,
                            # formatted according to Llama 3 Instruct chat template.

# --- Training Arguments ---
OUTPUT_DIR = "./llama3_finetuned_adapters"
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4 # Or 1e-4, 2e-5
OPTIMIZER = "adamw_8bit"
LR_SCHEDULER_TYPE = "linear" # or "cosine"
WARMUP_RATIO = 0.1
LOGGING_STEPS = 5
SAVE_STEPS = 20 # Save checkpoints more frequently for smaller datasets/epochs
SAVE_TOTAL_LIMIT = 2
FP16 = not torch.cuda.is_bf16_supported()
BF16 = torch.cuda.is_bf16_supported()

# Llama 3 Instruct prompt template for formatting data
# Ensure your dataset is pre-formatted or use a formatting function.
# The SFTTrainer can also apply chat templates if your data is structured as a list of messages.
LLAMA3_CHAT_TEMPLATE_EXAMPLE_INPUT_FIELD = "messages" # if your dataset has a 'messages' column

# Example of how a 'messages' field should look for SFTTrainer with Llama 3 template:
# [
#   {"role": "user", "content": "Hello, how are you?"},
#   {"role": "assistant", "content": "I'm doing great! How can I help you today?"}
# ]

def load_and_prepare_data(tokenizer):
    """
    Loads and prepares the dataset.
    This function needs to be adapted to your specific dataset format.
    If your data is not already in Llama 3 chat format (list of messages per example),
    you'll need to format it.
    """
    print(f"Loading dataset from: {DATASET_NAME_OR_PATH}")

    # --- Placeholder: Replace with your actual data loading and formatting ---
    if DATASET_NAME_OR_PATH == "your_dataset_path_or_name":
        print("WARNING: Using dummy dataset. Replace with your actual dataset loading.")
        # Dummy data in the "messages" format expected by SFTTrainer with apply_chat_template=True
        dummy_data = [
            {"messages": [
                {"role": "user", "content": "What is Unsloth?"},
                {"role": "assistant", "content": "Unsloth is a library for faster and more memory-efficient LLM fine-tuning."}
            ]},
            {"messages": [
                {"role": "user", "content": "Explain LoRA."},
                {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) is a technique to efficiently fine-tune large language models by adapting only a small subset of their parameters."}
            ]},
        ]
        from datasets import Dataset
        # The SFTTrainer will use tokenizer.apply_chat_template if dataset_text_field is not set
        # and the dataset has a column with message lists (e.g., "messages").
        # So, we set DATASET_TEXT_FIELD to None and ensure a 'messages' column.
        global DATASET_TEXT_FIELD
        DATASET_TEXT_FIELD = None # Let SFTTrainer handle chat templating
        dataset = Dataset.from_list(dummy_data)
        print(f"Dummy dataset created with 'messages' field. Example: {dataset[0]['messages']}")
    else:
        # Load your actual dataset here
        # If your dataset is already in a list of messages format (e.g., a column named 'messages'):
        # dataset = load_dataset(DATASET_NAME_OR_PATH, split="train")
        # DATASET_TEXT_FIELD = None # To use SFTTrainer's chat templating

        # If your dataset has 'instruction' and 'output' columns and you want to manually format:
        # dataset = load_dataset(DATASET_NAME_OR_PATH, split="train")
        # def format_llama3_chat(example):
        #     return {
        #         "text": tokenizer.apply_chat_template([
        #             {"role": "user", "content": example["instruction_column"]},
        #             {"role": "assistant", "content": example["output_column"]}
        #         ], tokenize=False, add_generation_prompt=False)
        #     }
        # dataset = dataset.map(format_llama3_chat)
        # DATASET_TEXT_FIELD = "text" # Now we have a 'text' field
        dataset = load_dataset(DATASET_NAME_OR_PATH, split="train") # Adjust as needed
        # Ensure your dataset is correctly formatted for Llama 3 or set DATASET_TEXT_FIELD = None
        # and provide a 'messages' column.
        pass
    # -------------------------------------------------------------------------

    return dataset

def main():
    logging.basicConfig(level=logging.INFO)
    
    config = TrainingConfig(
        base_model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        dataset_path=DATASET_NAME_OR_PATH,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation=GRADIENT_ACCUMULATION_STEPS
    )
    
    try:
        config.validate()
        monitor = TrainingMonitor()
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Training requires a GPU.")
            return

        print(f"Loading base model: {BASE_MODEL_NAME}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        print("Base model and tokenizer loaded.")

        # Llama 3 tokenizer might not have a pad_token set by default.
        # SFTTrainer needs it.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set tokenizer.pad_token to tokenizer.eos_token")

        if USE_LORA:
            print("Applying LoRA adaptations to the model...")
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                target_modules=LORA_TARGET_MODULES,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                bias=LORA_BIAS,
                use_gradient_checkpointing="unsloth",
                random_state=42,
                max_seq_length=MAX_SEQ_LENGTH,
            )
            print("LoRA applied. Trainable parameters:")
            model.print_trainable_parameters()

        dataset = load_and_prepare_data(tokenizer)

        print("Setting up SFTTrainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field=DATASET_TEXT_FIELD, # Set to None if dataset has 'messages' field for chat templating
            # formatting_func=your_formatting_func, # Alternatively, provide a formatting_func
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False, # Can be True for Llama 3 if data is suitable
            args=TrainingArguments(
                per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                warmup_ratio=WARMUP_RATIO,
                num_train_epochs=NUM_TRAIN_EPOCHS,
                learning_rate=LEARNING_RATE,
                fp16=FP16,
                bf16=BF16,
                logging_steps=LOGGING_STEPS,
                optim=OPTIMIZER,
                lr_scheduler_type=LR_SCHEDULER_TYPE,
                seed=42,
                output_dir=OUTPUT_DIR,
                save_strategy="steps",
                save_steps=SAVE_STEPS,
                save_total_limit=SAVE_TOTAL_LIMIT,
            ),
        )
        print("Trainer initialized.")

        print("Starting training...")
        trainer_stats = trainer.train()
        monitor.log_memory_usage()
        print("Training finished.")
        print(f"Trainer stats: {trainer_stats}")

        print(f"Saving LoRA adapters to {OUTPUT_DIR}...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Model adapters and tokenizer saved.")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return 1

if __name__ == "__main__":
    main()