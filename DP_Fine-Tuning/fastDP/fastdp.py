import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from fastDP import PrivacyEngine
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
    

# Model Configs
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
train_batch_size = 2
eval_batch_size = 2
gradient_accumulation_steps = 8
num_epochs = 1
learning_rate = 2e-4
max_input_length = 512
max_target_length = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_epsilon = 8.0

# Load Dataset
dataset = load_dataset("rajpurkar/squad")

# Preprocess questions and context
def preprocess(example):
   example["input_text"] = "Question: " + example["question"] + " Context: " + example["context"] # Preprocessed inoputs include both questions and context
   example["target_text"] = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
   return example

dataset = dataset.map(preprocess)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Tokenize and pad examples
def tokenize(example):
   model_inputs = tokenizer(
      example["input_text"],
      max_length=max_input_length,
      truncation=True,
      padding="max_length"
   )
   labels = tokenizer(
      example["target_text"],
      max_length=max_target_length,
      truncation=True,
      padding="max_length"
   )
   # Replace padding token id with -100 to ignore in loss
   labels["input_ids"] = [(l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# Dataloaders
train_loader = DataLoader(tokenized_datasets["train"], batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=eval_batch_size)


# Intialize model
model = AutoModelForCausalLM.from_pretrained(
   model_name,
   dtype=torch.bfloat16,
   device_map="auto"
)
model.gradient_checkpointing_enable()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Differential privacy with FastDP
effective_batch_size = train_batch_size * gradient_accumulation_steps
privacy_engine = PrivacyEngine(
   model,
   batch_size=effective_batch_size,
   sample_size=len(tokenized_datasets["train"]),
   epochs=num_epochs,
   target_epsilon=target_epsilon,
   clipping_fn="automatic",
   clipping_mode="MixOpt",
   clipping_style="all-layer"
)
privacy_engine.attach(optimizer)


# Train
model.train()
for epoch in range(num_epochs):
   running_loss = 0.0
   for step, batch in enumerate(train_loader):
      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      labels = batch["labels"].to(device)

      outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss / gradient_accumulation_steps
      loss.backward()

      if (step + 1) % gradient_accumulation_steps == 0:
         optimizer.step()
         optimizer.zero_grad()

      running_loss += loss.item()
      if step % 50 == 0:
         print(f"Epoch {epoch+1}, Step {step}, Loss {running_loss / (step+1):.4f}")

# Save model and tokenizer
model.save_pretrained("./llama3-8b-instruct-squad-dp-model")
tokenizer.save_pretrained("./llama3-8b-instruct-squad-dp-tokenizer")