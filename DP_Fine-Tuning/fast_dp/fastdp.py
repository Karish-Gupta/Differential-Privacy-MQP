import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastDP import PrivacyEngine
from utils.model_utils import *
from utils.gpu_usage import *
from utils.preprocessing import preprocess_dataset
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
      # Initialize tokenizer
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

      # Call the external preprocessing function
      self.train_loader, self.val_loader = preprocess_dataset(
         tokenizer=self.tokenizer,
         dataset_name=self.dataset_name,
         train_size=train_size,
         eval_size=eval_size,
         max_input_length=self.max_input_length,
         max_target_length=self.max_target_length,
         train_batch_size=self.train_batch_size,
         eval_batch_size=self.eval_batch_size,
         seed=seed
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
         clipping_style="layer-wise",
         loss_reduction="mean",
         record_snr=True,
         accounting_mode="rdp"
      )

      print("Attaching PrivacyEngine...")
      self.privacy_engine.attach(self.optimizer)
      print("PrivacyEngine attached.")

      print(f"\nPrivacy Configuration:")
      print(f"  Target ε: {self.target_epsilon}")
      print(f"  Target δ: {self.train_size ** -1.1:.2e}")
      print(f"  Effective batch size: {effective_batch_size}")
      print(f"  Training samples: {self.train_size}")
      print(f"  Epochs: {self.num_epochs}")
      print(f"  Steps per epoch: {self.train_size // effective_batch_size}")


   def train(self):
      self.model.train()
      global_step = 0
      
      for epoch in range(self.num_epochs):
         running_loss = 0.0
         epoch_steps = 0
         
         for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
               input_ids=input_ids, 
               attention_mask=attention_mask, 
               labels=labels
            )
            loss = outputs.loss            
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
               self.optimizer.step()
               self.optimizer.zero_grad()
               global_step += 1
               epoch_steps += 1

            running_loss += loss.item() * self.gradient_accumulation_steps
            
            # More frequent logging
            if (step + 1) % 500 == 0:
               avg_loss = running_loss / (step + 1)
               print(f"Epoch {epoch+1}/{self.num_epochs}, Step {step+1}, "
                     f"Loss: {avg_loss:.4f}, Global Step: {global_step}")
         
         # Epoch summary
         epoch_loss = running_loss / len(self.train_loader)
         print(f"\n{'='*60}")
         print(f"Epoch {epoch+1}/{self.num_epochs} Complete")
         print(f"  Average Loss: {epoch_loss:.4f}")
         print(f"  Steps: {epoch_steps}")
         print(f"{'='*60}\n")

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
         max_gen_length=64,
         show_samples=10
      )



if __name__ == "__main__":
   # Model Configs
   model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
   dataset_name = "squad"
   train_batch_size = 4
   eval_batch_size = 4
   gradient_accumulation_steps = 32
   num_epochs = 5
   learning_rate = 5e-4
   max_input_length = 512
   max_target_length = 512
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   target_epsilon = 2.0
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
   print(f"Epsilon: {target_epsilon}")
   
   fastdp.evaluate()

   # Ouput GPU logging
   stop_gpu_utilization_logging(gpu_util_thread, gpu_util_stop_event)
   print_gpu_utilization_summary(gpu_util_data)