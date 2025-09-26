import numpy as np
import torch
import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from fastDP import PrivacyEngine
from utils.model_utils import *
from utils.gpu_usage import *
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
   login(token=os.environ["HF_TOKEN"])
# transformers.utils.logging.set_verbosity_debug()

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
      train_size,

      lora_r=16,
      lora_alpha=16,
      lora_dropout=0.05,
      lora_target_modules=None,   # if None, good defaults for LLaMA
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
      self.train_size = train_size
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


   def preprocess_dataset(self, train_size, eval_size, seed=101):
      dataset = load_dataset(self.dataset_name)
      self.train_size = train_size

      dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(train_size))
      dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(eval_size))

      # Initialize tokenizer first
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

      def preprocess_and_tokenize_train(example):
         messages = [
            {"role": "system", "content": "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, in format Answer: {answer}"},
            {"role": "user", "content": f"Context: {example['context']} Question: {example['question']}"}
         ]
         input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
         target_text = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
         full_text = input_text + target_text
         
         tokenized = self.tokenizer(
               full_text,
               max_length=self.max_input_length + self.max_target_length,
               truncation=True,
               padding="max_length",
         )

         input_tokens = self.tokenizer(
               input_text,
               max_length=self.max_input_length,
               truncation=True,
               padding=False,
               add_special_tokens=False
         )
         input_length = len(input_tokens["input_ids"])

         labels = tokenized["input_ids"].copy()
         labels[:input_length] = [-100] * input_length  # mask input
         labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

         tokenized["labels"] = labels
         return tokenized
      
      def preprocess_and_tokenize_eval(example):
         messages = [
            {"role": "system", "content": "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, in format Answer: {answer}"},
            {"role": "user", "content": f"Context: {example['context']} Question: {example['question']}"}
         ]
         input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
         target_text = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""

         # Tokenize prompt only for inputs
         tokenized_inputs = self.tokenizer(
               input_text,
               max_length=self.max_input_length,
               truncation=True,
               padding="max_length"
         )

         # Tokenize full_text (with answer) just to build labels
         tokenized_full = self.tokenizer(
               input_text + target_text,
               max_length=self.max_input_length + self.max_target_length,
               truncation=True,
               padding="max_length"
         )

         # Mask out the input part, keep only the answer portion for labels
         labels = tokenized_full["input_ids"].copy()
         input_length = len(self.tokenizer(input_text, add_special_tokens=False)["input_ids"])
         labels[:input_length] = [-100] * input_length
         labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

         tokenized_inputs["labels"] = labels
         
         return tokenized_inputs

      train_dataset = dataset["train"].map(
         preprocess_and_tokenize_train,
         batched=False,
         remove_columns=dataset["train"].column_names,
      )

      eval_dataset = dataset["validation"].map(
         preprocess_and_tokenize_eval, 
         batched=False,
         remove_columns=dataset["validation"].column_names
      )

      self.train_loader = DataLoader(
         train_dataset,
         batch_size=self.train_batch_size,
         shuffle=True,
         collate_fn=default_data_collator,
      )
      
      self.val_loader = DataLoader(
         eval_dataset,
         batch_size=self.eval_batch_size,
         collate_fn=default_data_collator,
      )

   def init_model(self):
      self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="cuda:0")
      self.model = self.model.to(torch.float16)
      self.model.gradient_checkpointing_enable()

      target_modules = self.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
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
         sample_size=self.train_size,
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
               if step % 500 == 0:
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
   
   def evaluate(self):
      if self.val_loader is None:
         print("Validation loader not initialized. Run preprocess_dataset() first.")
         return
      model_device = next(self.model.parameters()).device
      print("Evaluating...")
      evaluate_model(
         self.model,
         self.val_loader,
         model_device,
         self.tokenizer,
         max_gen_length=60,
         show_samples=10
      )



if __name__ == "__main__":
   # Model Configs
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   dataset_name = "squad"
   train_batch_size = 1
   eval_batch_size = 1
   gradient_accumulation_steps = 8
   num_epochs = 5
   learning_rate = 2e-4
   max_input_length = 512
   max_target_length = 512
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   target_epsilon = 8.0
   train_size = 5000
   eval_size = 500


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
         train_size=train_size
   )

   # Start GPU utilization logging using utils
   gpu_util_thread, gpu_util_stop_event, gpu_util_data = start_gpu_utilization_logging(interval=1.0)

   fastdp.preprocess_dataset(train_size=train_size, eval_size=eval_size, seed=101)
   fastdp.init_model()
   fastdp.train()
   
   print(f"Model: {model_name}")
   print(f"On device: {device}")
   print(f"Number of epochs: {num_epochs}")
   print(f"Train batch size: {train_batch_size}")
   print(f"Eval batch size: {eval_batch_size}")
   print(f"Learning rate: {learning_rate}")
   print(f"Max input length: {max_input_length}")
   print(f"Max target length: {max_target_length}")
   print(f"Traing size: {train_size}")
   print(f"Eval size: {eval_size}")
   fastdp.evaluate()

   # Ouput GPU logging
   stop_gpu_utilization_logging(gpu_util_thread, gpu_util_stop_event)
   print_gpu_utilization_summary(gpu_util_data)