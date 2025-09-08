import numpy as np
import torch
import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from fastDP import PrivacyEngine
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])
transformers.utils.logging.set_verbosity_debug()

class FastDPModel:
   def __init__(
      self,
      model_name,
      dataset_name,
      train_batch_size,
      eval_batch_size,
      gradient_accumulation_steps,
      num_epochs,
      learning_rate,
      max_input_length,
      max_target_length,
      target_epsilon,

      lora_r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      lora_target_modules=None,   # if None, good defaults for LLaMA/Mistral-family are used
      lora_bias="none",           # "none" | "lora_only" | "all"
   ):
      # Configs
      self.model_name = model_name
      self.dataset_name = dataset_name
      self.train_batch_size = train_batch_size
      self.eval_batch_size = eval_batch_size
      self.gradient_accumulation_steps = gradient_accumulation_steps
      self.num_epochs = num_epochs
      self.learning_rate = learning_rate
      self.max_input_length = max_input_length
      self.max_target_length = max_target_length
      self.target_epsilon = target_epsilon
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      # LoRA configs
      self.lora_r = lora_r
      self.lora_alpha = lora_alpha
      self.lora_dropout = lora_dropout
      self.lora_target_modules = lora_target_modules
      self.lora_bias = lora_bias

      # Setup
      self.dataset = None
      self.tokenizer = None
      self.train_loader = None
      self.val_loader = None
      self.model = None
      self.optimizer = None
      self.privacy_engine = None

   def preprocess_dataset(self, subsample_size, seed=42):
      dataset = load_dataset(self.dataset_name)

      if subsample_size is not None:
         # Shuffle first to avoid always taking the same top slice
         dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(subsample_size))
         dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(subsample_size // 10)) 

      def preprocess(example):
         example["input_text"] = "Question: " + example["question"] + " Context: " + example["context"]
         example["target_text"] = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
         return example

      dataset = dataset.map(preprocess)

      # Initialize tokenizer
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

      def tokenize(example):
         model_inputs = self.tokenizer(
            example["input_text"],
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length"
         )
         labels = self.tokenizer(
            example["target_text"],
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length"
         )
         labels["input_ids"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
         model_inputs["labels"] = labels["input_ids"]
         return model_inputs

      self.dataset = dataset.map(
         tokenize, batched=True, remove_columns=dataset["train"].column_names
      )

      self.train_loader = DataLoader(
         self.dataset["train"],
         batch_size=self.train_batch_size,
         shuffle=True,
         collate_fn=default_data_collator
      )
      self.val_loader = DataLoader(
         self.dataset["validation"],
         batch_size=self.eval_batch_size,
         collate_fn=default_data_collator
      )


   def init_model(self):
      self.model = AutoModelForCausalLM.from_pretrained(
         self.model_name,
         torch_dtype=torch.float32,
         device_map="auto"
      )
      self.model.gradient_checkpointing_enable()

      target_modules = self.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
      peft_config = LoraConfig(
         task_type=TaskType.CAUSAL_LM,
         r=self.lora_r,
         lora_alpha=self.lora_alpha,
         lora_dropout=self.lora_dropout,
         bias=self.lora_bias,
         target_modules=target_modules,
      )

      self.model = get_peft_model(self.model, peft_config)
      self.model.print_trainable_parameters()

      # Optimizer over only trainable (LoRA) params
      trainable_params = (p for p in self.model.parameters() if p.requires_grad)
      self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

      effective_batch_size = self.train_batch_size * self.gradient_accumulation_steps
      self.privacy_engine = PrivacyEngine(
         self.model,
         batch_size=effective_batch_size,
         sample_size=len(self.dataset["train"]),
         epochs=self.num_epochs,
         target_epsilon=self.target_epsilon,
         clipping_fn="automatic",
         clipping_mode="MixOpt",
         clipping_style="all-layer"
      )

      print("Attaching PrivacyEngine...")
      self.privacy_engine.attach(self.optimizer)
      print("PrivacyEngine attached.")

   def train(self):
      self.model.train()
      for epoch in range(self.num_epochs):
         running_loss = 0.0
         for step, batch in enumerate(self.train_loader):
               input_ids = batch["input_ids"].to(self.device)
               attention_mask = batch["attention_mask"].to(self.device)
               labels = batch["labels"].to(self.device)

               outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
               loss = outputs.loss / self.gradient_accumulation_steps
               loss.backward()

               if (step + 1) % self.gradient_accumulation_steps == 0:
                  self.optimizer.step()
                  self.optimizer.zero_grad()

               running_loss += loss.item()
               if step % 50 == 0:
                  print(f"Epoch {epoch+1}, Step {step}, Loss {running_loss / (step+1):.4f}")

      # Detach privacy engine
      try:
         self.privacy_engine.detach()
      except Exception:
         pass

      # Save LoRA adapters only
      adapter_dir = "./llama3-8b-instruct-squad-dp-lora"
      self.model.save_pretrained(adapter_dir)
      self.tokenizer.save_pretrained(adapter_dir)
     



if __name__ == "__main__":
   # Model Configs
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   dataset_name = "rajpurkar/squad"
   train_batch_size = 2
   eval_batch_size = 2
   gradient_accumulation_steps = 8
   num_epochs = 1
   learning_rate = 2e-4
   max_input_length = 512
   max_target_length = 512
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   target_epsilon = 8.0

   print(f"Model: {model_name}")
   print(f"On device: {device}")
   print(f"Target epsilon: {target_epsilon}")

   fastdp = FastDPModel(
         model_name=model_name,
         dataset_name=dataset_name,
         train_batch_size=train_batch_size,
         eval_batch_size=eval_batch_size,
         gradient_accumulation_steps=gradient_accumulation_steps,
         num_epochs=num_epochs,
         learning_rate=learning_rate,
         max_input_length=max_input_length,
         max_target_length=max_target_length,
         target_epsilon=target_epsilon,
   )

   fastdp.preprocess_dataset(subsample_size=5000, seed=101)
   fastdp.init_model()
   fastdp.train()
