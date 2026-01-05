from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# =========================
# Load dataset
# =========================
dataset = load_dataset("text", data_files={"train": "custom_data.txt"})

# =========================
# Load tokenizer & model
# =========================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# =========================
# Tokenization
# =========================
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=64
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# 🔍 SAFETY CHECK (VERY IMPORTANT)
if len(tokenized_dataset["train"]) == 0:
    raise ValueError("Tokenized dataset is empty. Check your custom_data.txt")

# =========================
# Data collator
# =========================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# =========================
# Training arguments
# =========================
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    report_to="none"
)

# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Save model
# =========================
model.save_pretrained("gpt2-finetuned")
tokenizer.save_pretrained("gpt2-finetuned")

print("✅ GPT-2 fine-tuning completed successfully")



