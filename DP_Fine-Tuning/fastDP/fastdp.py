import torch
from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   Trainer,
   TrainingArguments,
   default_data_collator
)
from datasets import load_dataset
# from peft import LoraConfig 
from fastDP import PrivacyEngine
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])

# Load Dataset
dataset = load_dataset("rajpurkar/squad")

def preprocess(example):
   example["input_text"] = "Question: " + example["question"] + " Context: " + example["context"] # Preprocessed inoputs include both questions and context
   example["target_text"] = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
   return example

dataset = dataset.map(preprocess)

# Initialize model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
   model_inputs = tokenizer(
      example["input_text"],
      max_length=512,
      truncation=True,
      padding="max_length"
   )
   labels = tokenizer(
      example["target_text"],
      max_length=128,
      truncation=True,
      padding="max_length"
   )
   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

model = AutoModelForCausalLM.from_pretrained(
   model_name,
   dtype=torch.bfloat16,
   device_map="auto"
)

# Training args
training_args = TrainingArguments(
   output_dir="./llama3-8b-instruct-squad-dp",
   per_device_train_batch_size=2,
   per_device_eval_batch_size=2,
   gradient_accumulation_steps=8,
   # evaluation_strategy="steps",
   # eval_steps=200,
   # save_steps=500,
   logging_steps=50,
   learning_rate=2e-4,
   num_train_epochs=1,
   weight_decay=0.01,
   warmup_steps=100,
   fp16=False,
   bf16=True,
   gradient_checkpointing=True,
   dataloader_drop_last=True,
   report_to="none"
)

# Trainer
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_datasets["train"],
   eval_dataset=tokenized_datasets["validation"],
   tokenizer=tokenizer,
   data_collator=default_data_collator
)

# Differential privacy with FastDP
privacy_engine = PrivacyEngine(
   model,
   batch_size=training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
   sample_size=len(tokenized_datasets["train"]),
   epochs=training_args.num_train_epochs,
   target_epsilon=8.0,  # DP budget (ε)
   clipping_fn="automatic",
   clipping_mode="MixOpt",
   clipping_style="all-layer"
)

privacy_engine.attach(trainer.optimizer)

trainer.train()
trainer.save_model("./llama3-8b-instruct-squad-dp")
tokenizer.save_pretrained("./llama3-8b-instruct-squad-dp")