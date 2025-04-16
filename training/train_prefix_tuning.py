import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers.trainer_callback import EarlyStoppingCallback
from peft import PrefixTuningConfig, get_peft_model, TaskType
from peft.tuners.prefix_tuning import PrefixEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="Base model to use")
    parser.add_argument("--train_file", type=str, default="../data/formatted_train.jsonl", help="Path to training data")
    parser.add_argument("--val_file", type=str, default="../data/formatted_val.jsonl", help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="../models/llama3_prefix_tuned", help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_virtual_tokens", type=int, default=32, help="Number of virtual tokens for prefix tuning")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 for training")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    return parser.parse_args()

def load_dataset_from_jsonl(file_path):
    """Load dataset from JSONL file"""
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def preprocess_function(examples, tokenizer, max_length):
    """Preprocess the data by tokenizing"""
    # Format inputs for causal language modeling with instruction format
    inputs = []
    for instruction, inp, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Format: <instruction>\n<input>\n<output>
        text = f"{instruction}\n{inp}\n{output}"
        inputs.append(text)
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Create labels
    labels = model_inputs["input_ids"].clone()
    
    # Mask padding tokens from loss
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    
    return model_inputs

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading base model: {args.base_model_path}")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print(f"Loading model in {'8-bit' if args.load_in_8bit else 'full precision'}")
    model_kwargs = {
        "device_map": "auto",
    }
    
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        **model_kwargs
    )
    
    # Configure prefix tuning
    print(f"Setting up prefix tuning with {args.num_virtual_tokens} virtual tokens")
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        num_virtual_tokens=args.num_virtual_tokens,
        token_dim=base_model.config.hidden_size,
        num_transformer_submodules=1,  # For Llama, use 1 instead of default
        num_attention_heads=base_model.config.num_attention_heads,
        num_layers=base_model.config.num_hidden_layers,
        encoder_hidden_size=base_model.config.hidden_size,
        prefix_projection=True,
    )
    
    # Convert to PEFT model
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # Load datasets
    print(f"Loading datasets from {args.train_file} and {args.val_file}")
    try:
        train_dataset = load_dataset_from_jsonl(args.train_file)
        val_dataset = load_dataset_from_jsonl(args.val_file)
        print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
        
    # Preprocess datasets
    print("Preprocessing datasets")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Set up training arguments
    print("Configuring training")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=50,
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        fp16=args.use_fp16,
        report_to="tensorboard",
        gradient_accumulation_steps=4,  # To help with batch size limitations
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize trainer with early stopping
    print("Initializing trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    
    # Train
    print("Starting training")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    
    # Save tokenizer and training metadata
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metadata
    metadata = {
        "base_model": args.base_model_path,
        "tuning_type": "prefix",
        "num_virtual_tokens": args.num_virtual_tokens,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "training_time": trainer.state.log_history[-1].get("train_runtime", "N/A"),
        "date": trainer.state.log_history[-1].get("date", "N/A"),
    }
    
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
