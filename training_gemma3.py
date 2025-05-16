#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
training_gemma3.py

This script fine-tunes a Gemma model using the Unsloth library and TRL's SFTTrainer.
It's a conversion of a typical Jupyter Notebook workflow (e.g., training-gemma3.ipynb)
into a standalone Python script.

Assumptions:
- You have installed Unsloth, TRL, Datasets, Accelerate, and their dependencies.
  (e.g., pip install "unsloth[gemma] @ git+https://github.com/unslothai/unsloth.git")
  (e.g., pip install trl datasets accelerate bitsandbytes)
- You have a CUDA-enabled GPU for efficient training.
- Your dataset is prepared and accessible (e.g., a CSV file or Hugging Face dataset).
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset # To load datasets

# Configuration (Adjust these parameters as needed)

# --- Model Configuration ---
BASE_MODEL_NAME = "unsloth/gemma-2b-it-bnb-4bit"  # Base Gemma model to fine-tune
                                                 # Can be other Gemma variants like "unsloth/gemma-7b-bnb-4bit"
MAX_SEQ_LENGTH = 2048  # Max sequence length for the model and tokenizer
DTYPE = None  # Autodetect. Can be torch.float16, torch.bfloat16, or torch.float32
LOAD_IN_4BIT = True  # Load in 4-bit for memory efficiency

# --- LoRA Configuration (if using LoRA) ---
USE_LORA = True
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [ # Modules to apply LoRA to. Common for Gemma:
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_BIAS = "none" # "none", "all", or "lora_only"

# --- Dataset Configuration ---
# This is an example. You'll need to adapt this to your specific dataset.
# DATASET_NAME = "yahma/alpaca-cleaned" # Example Hugging Face dataset
DATASET_NAME_OR_PATH = "your_dataset_path_or_name" # Replace with your dataset (e.g., "data/conversations.csv")
DATASET_TEXT_FIELD = "text" # The column in your dataset that contains the formatted text for training
                            # This 'text' field should ideally contain the full prompt and response.

# --- Training Arguments ---
OUTPUT_DIR = "./gemma3_finetuned_adapters" # Directory to save LoRA adapters
NUM_TRAIN_EPOCHS = 1 # Number of training epochs
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Batch size per GPU
GRADIENT_ACCUMULATION_STEPS = 4 # Accumulate gradients for effective batch size
LEARNING_RATE = 2e-4
OPTIMIZER = "adamw_8bit" # Or "adamw_torch"
LR_SCHEDULER_TYPE = "linear"
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 50 # Save checkpoints every N steps
SAVE_TOTAL_LIMIT = 2 # Keep only the last N checkpoints
FP16 = not torch.cuda.is_bf16_supported() # Use fp16 if bf16 is not available
BF16 = torch.cuda.is_bf16_supported()

# Alpaca prompt template (if you need to format your data)
# Ensure your dataset is pre-formatted or use a formatting function.
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def load_and_prepare_data(tokenizer):
    """
    Loads and prepares the dataset.
    This function needs to be adapted to your specific dataset format.
    The goal is to have a dataset with a column (e.g., 'text')
    where each entry is a fully formatted prompt + response string.
    """
    print(f"Loading dataset from: {DATASET_NAME_OR_PATH}")
    # Example: Loading a CSV file
    # dataset = load_dataset("csv", data_files={"train": DATASET_NAME_OR_PATH})["train"]

    # Example: Loading from Hugging Face Hub
    # dataset = load_dataset(DATASET_NAME_OR_PATH, split="train")

    # --- Placeholder: Replace with your actual data loading and formatting ---
    # For demonstration, let's create a dummy dataset if DATASET_NAME_OR_PATH is not set
    if DATASET_NAME_OR_PATH == "your_dataset_path_or_name":
        print("WARNING: Using dummy dataset. Replace with your actual dataset loading.")
        data = [
            {"instruction": "What is Unsloth?", "output": "Unsloth is a library for faster and more memory-efficient LLM fine-tuning."},
            {"instruction": "Explain LoRA.", "output": "LoRA (Low-Rank Adaptation) is a technique to efficiently fine-tune large language models by adapting only a small subset of their parameters."},
        ]
        # Format the dummy data
        formatted_data = [
            ALPACA_PROMPT_TEMPLATE.format(item["instruction"], item["output"]) + tokenizer.eos_token
            for item in data
        ]
        from datasets import Dataset
        dataset = Dataset.from_dict({DATASET_TEXT_FIELD: formatted_data})
    else:
        # Load your actual dataset here
        # Ensure it has a column named DATASET_TEXT_FIELD containing the final text for training
        # Example: dataset = load_dataset("json", data_files="my_data.jsonl")["train"]
        # Or if your data is already formatted:
        dataset = load_dataset(DATASET_NAME_OR_PATH, split="train") # Adjust split if necessary
        # You might need a formatting function here if your dataset isn't pre-formatted
        # def formatting_prompts_func(examples):
        #    texts = []
        #    for instruction, output in zip(examples['instruction_column'], examples['output_column']):
        #        text = ALPACA_PROMPT_TEMPLATE.format(instruction, output) + tokenizer.eos_token
        #        texts.append(text)
        #    return { DATASET_TEXT_FIELD: texts, }
        # dataset = dataset.map(formatting_prompts_func, batched = True,)
        pass
    # -------------------------------------------------------------------------

    print(f"Dataset loaded. Example entry: {dataset[0][DATASET_TEXT_FIELD][:200]}...")
    return dataset

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Training requires a GPU.")
        return

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        # token="hf_YOUR_TOKEN_HERE", # Uncomment if model is gated
    )
    print("Base model and tokenizer loaded.")

    if USE_LORA:
        print("Applying LoRA adaptations to the model...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            use_gradient_checkpointing="unsloth", # Recommended by Unsloth
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
        dataset_text_field=DATASET_TEXT_FIELD, # Column with formatted text
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2, # Number of processes for dataset mapping
        packing=False,  # Can be True for more efficient training if data is suitable
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
    print("Training finished.")
    print(f"Trainer stats: {trainer_stats}")

    print(f"Saving LoRA adapters to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR) # Saves LoRA adapters
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model adapters and tokenizer saved.")

    # If you want to save the full merged model (takes more space)
    # print("Merging and saving full model (optional)...")
    # merged_model_dir = f"{OUTPUT_DIR}_merged"
    # model.save_pretrained_merged(merged_model_dir, tokenizer, save_method="merged_16bit") # or "merged_4bit"
    # print(f"Full merged model saved to {merged_model_dir}")

if __name__ == "__main__":
    main()