import numpy as np
import torch
import torch.distributed as dist
import functools
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastDP import PrivacyEngine
from utils.model_utils import *
from utils.gpu_usage import *
from utils.preprocessing import preprocess_dataset
from huggingface_hub import login
from accelerate import Accelerator
import os

# Import FSDP related modules
from torch.distributed.fsdp import (
   FullyShardedDataParallel as FSDP,
   CPUOffload,
   MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
   default_auto_wrap_policy,
   enable_wrap,
   wrap,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
   FullStateDictConfig,
   StateDictType,
)

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
      # gradient_accumulation_steps,
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
      
      # FSDP configs
      use_fsdp=True,
      fsdp_min_num_params=1e6,    # Min number of params for auto-wrapping (default: 1M)
      cpu_offload=False,          # Whether to offload parameters to CPU
      mixed_precision=True,       # Whether to use mixed precision
   ):
      # Configs
      self.model_name = model_name
      self.dataset_name = dataset_name
      self.train_batch_size = train_batch_size
      self.eval_batch_size = eval_batch_size
      # self.gradient_accumulation_steps = gradient_accumulation_steps
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
      
      # FSDP configs
      self.use_fsdp = use_fsdp
      self.fsdp_min_num_params = fsdp_min_num_params
      self.cpu_offload = cpu_offload
      self.mixed_precision = mixed_precision

      # Setup
      self.dataset = None
      self.tokenizer = None
      self.train_loader = None
      self.val_loader = None
      self.model = None
      self.optimizer = None
      self.privacy_engine = None
      
      # Initialize distributed environment if using FSDP
      if self.use_fsdp and not dist.is_initialized():
         dist.init_process_group(backend="nccl")
         self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
         self.rank = int(os.environ.get("RANK", 0))
         self.world_size = int(os.environ.get("WORLD_SIZE", 1))
         torch.cuda.set_device(self.local_rank)
         self.is_main_process = (self.rank == 0)
         self.accelerator = None
      else:
         # Fall back to Accelerator if not using FSDP
         self.accelerator = Accelerator(mixed_precision="fp16")
         self.is_main_process = self.accelerator.is_main_process



   def preprocess_dataset(self, train_size, eval_size, seed=101):
      # Initialize tokenizer
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

      # Set up distributed sampler parameters if using FSDP
      sampler_params = None
      if self.use_fsdp:
         sampler_params = {
            "rank": self.rank,
            "world_size": self.world_size,
            "shuffle": True
         }
         if self.is_main_process:
            print(f"Setting up distributed sampler with world_size={self.world_size}")

      # Call the external preprocessing function with distributed sampler parameters if needed
      self.train_loader, self.val_loader = preprocess_dataset(
         tokenizer=self.tokenizer,
         dataset_name=self.dataset_name,
         train_size=train_size,
         eval_size=eval_size,
         max_input_length=self.max_input_length,
         max_target_length=self.max_target_length,
         train_batch_size=self.train_batch_size,
         eval_batch_size=self.eval_batch_size,
         seed=seed,
         sampler_params=sampler_params
      )

   def init_model(self):
      self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
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
      if self.is_main_process:
         self.model.print_trainable_parameters()

      # Wrap model with FSDP if enabled
      if self.use_fsdp:
         # Define mixed precision policy if enabled
         mixed_precision_policy = None
         if self.mixed_precision:
            mixed_precision_policy = MixedPrecision(
               param_dtype=torch.float16,
               reduce_dtype=torch.float16,
               buffer_dtype=torch.float16,
            )
         
         # Define CPU offload if enabled
         cpu_offload_config = None
         if self.cpu_offload:
            cpu_offload_config = CPUOffload(offload_params=True)
         
         # Define auto wrap policy
         auto_wrap_policy = functools.partial(
            default_auto_wrap_policy,
            min_num_params=self.fsdp_min_num_params,
         )
         
         # Wrap the model with FSDP
         self.model = FSDP(
            self.model,
            fsdp_auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload_config,
            mixed_precision=mixed_precision_policy,
         )
         
         if self.is_main_process:
            print(f"Model wrapped with FSDP, world size: {self.world_size}")
      
      # Optimizer over only trainable (LoRA) params
      trainable_params = (p for p in self.model.parameters() if p.requires_grad)
      self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
      
      if self.accelerator:
         # Prepare model, optimizer, and loaders with accelerator
         self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model,
            self.optimizer, 
            self.train_loader, 
            self.val_loader
         )

      effective_batch_size = self.train_batch_size * (self.world_size if self.use_fsdp else 1)
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

      if self.is_main_process:
         print("Attaching PrivacyEngine...")
      self.privacy_engine.attach(self.optimizer)
      if self.is_main_process:
         print("PrivacyEngine attached.")


   def train(self):
      self.model.train()
      for epoch in range(self.num_epochs):
         running_loss = 0.0
         for step, batch in enumerate(self.train_loader):
            # Move batch to the appropriate device
            if self.use_fsdp:
               batch = {k: v.to(self.local_rank) for k, v in batch.items()}
            
            outputs = self.model(
                  input_ids=batch["input_ids"],
                  attention_mask=batch["attention_mask"],
                  labels=batch["labels"]
            )
            loss = outputs.loss
            
            if self.accelerator:
               self.accelerator.backward(loss)
            else:
               loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            if step % 500 == 0:
                  avg_loss = running_loss / (step + 1)
                  if self.is_main_process:  # only log once
                     print(f"Epoch {epoch+1}, Step {step}, Loss {avg_loss:.4f}")

      try:
         self.privacy_engine.detach()
      except Exception:
         pass

      if self.is_main_process:  # only main process saves
         adapter_dir = "./llama3-8b-instruct-squad-dp-lora"
         
         if self.use_fsdp:
            # FSDP state dict for saving
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
               state_dict = self.model.state_dict()
               if self.is_main_process:
                  # Need to extract the base model for saving with PEFT
                  unwrapped_model = self.model.module  # Get the model without FSDP wrapper
                  unwrapped_model.save_pretrained(adapter_dir)
                  self.tokenizer.save_pretrained(adapter_dir)
         else:
            # unwrap Accelerate wrapper when saving
            self.accelerator.unwrap_model(self.model).save_pretrained(adapter_dir)
            self.tokenizer.save_pretrained(adapter_dir)
         
         print(f"Model adapter saved to {adapter_dir}")

   
   def evaluate(self):
      if self.val_loader is None:
         print("Validation loader not initialized. Run preprocess_dataset() first.")
         return
      
      # Only evaluate on main process
      if not self.is_main_process:
         return
         
      model_device = next(self.model.parameters()).device
      print("Evaluating...")
      
      # For FSDP, need to consolidate model for evaluation
      if self.use_fsdp:
         with FSDP.summon_full_params(self.model):
            evaluate_model(
               self.model,
               self.val_loader,
               model_device,
               self.tokenizer,
               max_gen_length=64,
               show_samples=10
            )
      else:
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
   # gradient_accumulation_steps = 8
   num_epochs = 5
   learning_rate = 2e-4
   max_input_length = 512
   max_target_length = 512
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   target_epsilon = 2.0
   train_size = 5000
   eval_size = 500
   
   # FSDP Configs
   use_fsdp = True
   fsdp_min_num_params = 1e6  # Auto-wrap modules with >1M params
   cpu_offload = False
   mixed_precision = True


   fastdp = FastDPModel(
         model_name=model_name,
         dataset_name=dataset_name,
         train_batch_size=train_batch_size,
         eval_batch_size=eval_batch_size,
         # gradient_accumulation_steps=gradient_accumulation_steps,
         num_epochs=num_epochs,
         learning_rate=learning_rate,
         max_input_length=max_input_length,
         max_target_length=max_target_length,
         target_epsilon=target_epsilon,
         train_size=train_size,
         # FSDP parameters
         use_fsdp=use_fsdp,
         fsdp_min_num_params=fsdp_min_num_params,
         cpu_offload=cpu_offload,
         mixed_precision=mixed_precision
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