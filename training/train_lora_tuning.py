from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import numpy as np
from transformers.trainer_callback import TrainerCallback
from datetime import datetime
import matplotlib.pyplot as plt
# Add at the top of your script
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load dataset
train_dataset = Dataset.from_json("../data/cuad_train.jsonl")
val_dataset = Dataset.from_json("../data/cuad_val.jsonl")

print(f"Original dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Reduce dataset size with stratified sampling to maintain class distribution
# Shuffle with a fixed seed for reproducibility
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)

# Limit to 10,000 for training and 1,000 for validation
train_dataset = train_dataset.select(range(min(20000, len(train_dataset))))
val_dataset = val_dataset.select(range(min(2000, len(val_dataset))))

print(f"Reduced dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Load tokenizer and model
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load model with memory-efficient settings
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")

# Format inputs for causal language modeling
def tokenize_fn(example):
    full_text = f"{example['instruction']}\n{example['input']}\nExplanation: {example['output']}"
    tokenized = tokenizer(full_text, padding="max_length", truncation=True, max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Process datasets
print("Processing training data...")
train_data = train_dataset.map(tokenize_fn)
print("Processing validation data...")
val_data = val_dataset.map(tokenize_fn)


# Create metrics callback to track and plot metrics
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.last_logged_step = -1
        
    def on_init_end(self, args, state, control, **kwargs):
        # Required implementation
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize tracking at training start
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.last_logged_step = -1
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        if "loss" in logs and state.global_step > self.last_logged_step:
            self.train_losses.append(logs["loss"])
            self.steps.append(state.global_step)
            self.last_logged_step = state.global_step
            
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            
    def plot_metrics(self, save_path="../models/training_metrics.png"):
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.train_losses, label="Training Loss")
        
        # Plot eval losses at their respective steps
        if self.eval_losses:
            # Handle case where we might have different numbers of eval and train steps
            if len(self.steps) >= len(self.eval_losses):
                # Evenly space eval points across the training steps
                eval_indices = np.linspace(0, len(self.steps)-1, len(self.eval_losses), dtype=int)
                eval_steps = [self.steps[i] for i in eval_indices]
                plt.plot(eval_steps, self.eval_losses, label="Validation Loss", marker="o")
            
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")

metrics_callback = MetricsCallback()

# Training arguments
args = TrainingArguments(
    output_dir="../models/llama3_lora_tuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=5e-4,
    report_to="none",
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[metrics_callback]
)

# Memory cleanup before training
torch.cuda.empty_cache()
print("Starting training...")
train_result = trainer.train()

# Print and save metrics
print("\nTraining complete! Final metrics:")
print(f"Training loss: {train_result.training_loss:.4f}")

# Run final evaluation
eval_result = trainer.evaluate()
print(f"Validation loss: {eval_result['eval_loss']:.4f}")

# Plot training metrics
metrics_callback.plot_metrics()

# Save the LoRA adapter
model.save_pretrained("../models/llama3_lora_tuned")
print("Model saved successfully!")

# Save training summary
with open("../models/training_summary.txt", "w") as f:
    f.write(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {model_id}\n")
    f.write(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")
    f.write(f"Final training loss: {train_result.training_loss:.4f}\n")
    f.write(f"Final validation loss: {eval_result['eval_loss']:.4f}\n")
    f.write(f"Total training steps: {trainer.state.global_step}\n")

print("Training summary saved to ../models/training_summary.txt")