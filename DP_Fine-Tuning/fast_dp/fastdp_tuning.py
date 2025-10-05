"""
Complete Optuna Hyperparameter Tuning for Differential Privacy + LoRA Fine-tuning
Optimizes: Contains Accuracy and F1 Score with learning rate, LoRA config, batch size, clipping
"""

import optuna
from optuna.trial import Trial
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from fastDP import PrivacyEngine
import json
import os
from datetime import datetime
import re
import string
from utils.model_utils import *
from utils.gpu_usage import *
from utils.preprocessing import preprocess_dataset

class OptunaFastDPModel:
   """FastDP Model optimized for Optuna hyperparameter tuning"""
   
   def __init__(
      self,
      model_name,
      dataset_name,
      train_size,
      eval_size,
      num_epochs,
      target_epsilon,
      max_input_length=512,
      max_target_length=512,
   ):
      self.model_name = model_name
      self.dataset_name = dataset_name
      self.train_size = train_size
      self.eval_size = eval_size
      self.num_epochs = num_epochs
      self.target_epsilon = target_epsilon
      self.max_input_length = max_input_length
      self.max_target_length = max_target_length
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
      # Initialize tokenizer once
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
      
      # Preprocess dataset once
      self.train_loader = None
      self.val_loader = None
   
   def preprocess_dataset(self, train_batch_size, eval_batch_size, seed=101):
      """Preprocess dataset with given batch sizes"""
      
      self.train_loader, self.val_loader = preprocess_dataset(
         tokenizer=self.tokenizer,
         dataset_name=self.dataset_name,
         train_size=self.train_size,
         eval_size=self.eval_size,
         max_input_length=self.max_input_length,
         max_target_length=self.max_target_length,
         train_batch_size=train_batch_size,
         eval_batch_size=eval_batch_size,
         seed=seed
      )
   
   def create_model(self, trial_params):
      """Create model with trial hyperparameters"""
      model = AutoModelForCausalLM.from_pretrained(
         self.model_name, 
         device_map="cuda:0"
      )
      model = model.to(torch.float16)
      model.gradient_checkpointing_enable()
      
      # LoRA configuration from trial
      peft_config = LoraConfig(
         task_type=TaskType.CAUSAL_LM,
         r=trial_params['lora_r'],
         lora_alpha=trial_params['lora_alpha'],
         lora_dropout=trial_params['lora_dropout'],
         bias=trial_params['lora_bias'],
         target_modules=trial_params['target_modules'],
      )
      
      model = get_peft_model(model, peft_config)
      return model
   
   def train_and_evaluate(self, trial_params, trial_number):
      """Train model with given hyperparameters and return combined score"""
      
      print(f"\n{'='*70}")
      print(f"Trial {trial_number}: Testing hyperparameters")
      print(f"{'='*70}")
      for key, value in trial_params.items():
         print(f"  {key}: {value}")
      print(f"{'='*70}\n")
      
      # Create model
      model = self.create_model(trial_params)
      
      # Create optimizer
      trainable_params = [p for p in model.parameters() if p.requires_grad]
      optimizer = torch.optim.AdamW(trainable_params, lr=trial_params['learning_rate'])
      
      # Create privacy engine
      effective_batch_size = (trial_params['train_batch_size'] * 
                              trial_params['gradient_accumulation_steps'])
      
      privacy_engine = PrivacyEngine(
         model,
         batch_size=effective_batch_size,
         sample_size=self.train_size,
         epochs=self.num_epochs,
         target_epsilon=self.target_epsilon,
         clipping_fn=trial_params['clipping_fn'],
         clipping_mode=trial_params['clipping_mode'],
         clipping_style=trial_params['clipping_style'],
         loss_reduction="mean",
         record_snr=True,
         accounting_mode="rdp",
      )
      
      privacy_engine.attach(optimizer)
      
      # Training loop
      model.train()
      for epoch in range(self.num_epochs):
         running_loss = 0.0
         
         for step, batch in enumerate(self.train_loader):
               input_ids = batch["input_ids"].to(self.device)
               attention_mask = batch["attention_mask"].to(self.device)
               labels = batch["labels"].to(self.device)
               
               outputs = model(
                  input_ids=input_ids,
                  attention_mask=attention_mask,
                  labels=labels
               )
               loss = outputs.loss / trial_params['gradient_accumulation_steps']
               loss.backward()
               
               if (step + 1) % trial_params['gradient_accumulation_steps'] == 0:
                  optimizer.step()
                  optimizer.zero_grad()
               
               running_loss += loss.item() * trial_params['gradient_accumulation_steps']
         
         epoch_loss = running_loss / len(self.train_loader)
         print(f"  Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")
      
      # Evaluate - get both metrics
      contains_acc, f1_score = self.evaluate(model)
      
      # Combined score: weighted average prioritizing both metrics equally
      combined_score = 0.5 * contains_acc + 0.5 * f1_score
      
      print(f"\n  Results:")
      print(f"    Contains Accuracy: {contains_acc:.4f}")
      print(f"    F1 Score: {f1_score:.4f}")
      print(f"    Combined Score: {combined_score:.4f}")
      
      # Cleanup
      privacy_engine.detach()
      del model
      del optimizer
      del privacy_engine
      torch.cuda.empty_cache()
      
      return combined_score, contains_acc, f1_score
   
   def evaluate(self, model):
      """Evaluate model and return both contains accuracy and F1 score"""
      if self.val_loader is None:
         print("Validation loader not initialized.")
         return 0.0, 0.0
      
      model_device = next(model.parameters()).device
      print("Evaluating...")
            
      # Call your evaluation function
      results = evaluate_model(
         model,
         self.val_loader,
         model_device,
         self.tokenizer,
         max_gen_length=15,
         show_samples=0  # Don't show samples during optimization
      )
      
      # Extract metrics from results dictionary
      contains_acc = results.get("contains_accuracy", 0.0)
      f1_score = results.get("f1", 0.0)
      exact_match = results.get("exact_match_accuracy", 0.0)
      
      return contains_acc, f1_score


# ============================================================================
# OBJECTIVE FUNCTIONS
# ============================================================================

def objective(trial: Trial, base_model, fixed_params):
   """
   Optuna objective function to optimize Contains Accuracy and F1 Score
   
   Args:
      trial: Optuna trial object
      base_model: OptunaFastDPModel instance
      fixed_params: Dictionary of fixed hyperparameters
   
   Returns:
      Combined score (0.5 * contains_acc + 0.5 * f1_score)
   """
   
   # Hyperparameters to tune
   trial_params = {
      # Learning rate (log scale)
      'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True),
      
      # LoRA configuration
      'lora_r': trial.suggest_categorical('lora_r', [8, 16, 32, 64]),
      'lora_alpha': trial.suggest_categorical('lora_alpha', [16, 32, 64, 128]),
      'lora_dropout': trial.suggest_float('lora_dropout', 0.0, 0.1),
      'lora_bias': trial.suggest_categorical('lora_bias', ['none', 'lora_only']),
      
      # Target modules
      'target_modules': trial.suggest_categorical('target_modules', [
         ["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention only
         ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # All
      ]),
      
      # Batch size and accumulation
      'train_batch_size': trial.suggest_categorical('train_batch_size', [2, 4, 8]),
      'gradient_accumulation_steps': trial.suggest_categorical(
         'gradient_accumulation_steps', [16, 32, 64, 128]
      ),
      
      # Clipping configuration
      'clipping_fn': trial.suggest_categorical('clipping_fn', ['automatic', 'Abadi']),
      'clipping_mode': 'MixOpt',  # Fixed to fastest mode
      'clipping_style': trial.suggest_categorical(
         'clipping_style', ['all-layer', 'layer-wise']
      ),
      
      # Fixed evaluation batch size
      'eval_batch_size': fixed_params['eval_batch_size'],
   }
   
   # Preprocess dataset with trial batch sizes
   base_model.preprocess_dataset(
      train_batch_size=trial_params['train_batch_size'],
      eval_batch_size=trial_params['eval_batch_size']
   )
   
   # Train and evaluate
   try:
      combined_score, contains_acc, f1_score = base_model.train_and_evaluate(
         trial_params, trial.number
      )
      
      # Store individual metrics as user attributes for analysis
      trial.set_user_attr('contains_accuracy', contains_acc)
      trial.set_user_attr('f1_score', f1_score)
      
      # Report intermediate value for pruning
      trial.report(combined_score, base_model.num_epochs)
      
      # Check if trial should be pruned
      if trial.should_prune():
         raise optuna.TrialPruned()
      
      return combined_score
   
   except Exception as e:
      print(f"Trial {trial.number} failed with error: {e}")
      return 0.0  # Return worst score on failure



# ============================================================================
# STUDY RUNNERS
# ============================================================================

def run_optuna_study(
   model_name="meta-llama/Meta-Llama-3-8B-Instruct",
   dataset_name="squad",
   train_size=5000,
   eval_size=500,
   num_epochs=5,
   target_epsilon=2.0,
   n_trials=20,
   study_name="fastdp_optimization",
   storage=None,  # e.g., "sqlite:///optuna_study.db" for persistence
):
   """
   Run Optuna hyperparameter optimization study
   
   Args:
      model_name: HuggingFace model name
      dataset_name: Dataset to use
      train_size: Number of training samples
      eval_size: Number of evaluation samples
      num_epochs: Training epochs per trial
      target_epsilon: Privacy budget
      n_trials: Number of Optuna trials
      study_name: Name for the study
      storage: Database URL for persistent storage
   """
   
   print(f"\n{'='*70}")
   print(f"Starting Optuna Hyperparameter Optimization")
   print(f"{'='*70}")
   print(f"Model: {model_name}")
   print(f"Dataset: {dataset_name}")
   print(f"Training samples: {train_size}")
   print(f"Eval samples: {eval_size}")
   print(f"Epochs per trial: {num_epochs}")
   print(f"Target epsilon: {target_epsilon}")
   print(f"Number of trials: {n_trials}")
   print(f"{'='*70}\n")
   
   # Initialize base model
   base_model = OptunaFastDPModel(
      model_name=model_name,
      dataset_name=dataset_name,
      train_size=train_size,
      eval_size=eval_size,
      num_epochs=num_epochs,
      target_epsilon=target_epsilon,
   )
   
   # Fixed parameters
   fixed_params = {
      'eval_batch_size': 4,
   }
   
   # Create or load study
   study = optuna.create_study(
      study_name=study_name,
      direction='maximize',  # Maximize combined score
      storage=storage,
      load_if_exists=True,
      pruner=optuna.pruners.MedianPruner(
         n_startup_trials=5,
         n_warmup_steps=2,
         interval_steps=1
      ),
      sampler=optuna.samplers.TPESampler(seed=101)
   )
   
   # Run optimization
   study.optimize(
      lambda trial: objective(trial, base_model, fixed_params),
      n_trials=n_trials,
      timeout=None,
      show_progress_bar=True,
   )
   
   # Print results
   print(f"\n{'='*70}")
   print(f"Optimization Complete!")
   print(f"{'='*70}")
   print(f"\nBest trial:")
   trial = study.best_trial
   print(f"  Combined Score: {trial.value:.4f}")
   print(f"  Contains Accuracy: {trial.user_attrs.get('contains_accuracy', 'N/A'):.4f}")
   print(f"  F1 Score: {trial.user_attrs.get('f1_score', 'N/A'):.4f}")
   print(f"\nBest hyperparameters:")
   for key, value in trial.params.items():
      print(f"  {key}: {value}")
   
   # Save results
   results_dir = "optuna_results"
   os.makedirs(results_dir, exist_ok=True)
   
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   results_file = f"{results_dir}/{study_name}_{timestamp}.json"
   
   results = {
      'study_name': study_name,
      'timestamp': timestamp,
      'n_trials': len(study.trials),
      'best_trial': {
         'number': trial.number,
         'combined_score': trial.value,
         'contains_accuracy': trial.user_attrs.get('contains_accuracy', None),
         'f1_score': trial.user_attrs.get('f1_score', None),
         'params': trial.params,
      },
      'all_trials': [
         {
               'number': t.number,
               'combined_score': t.value,
               'contains_accuracy': t.user_attrs.get('contains_accuracy', None),
               'f1_score': t.user_attrs.get('f1_score', None),
               'params': t.params,
               'state': str(t.state),
         }
         for t in study.trials
      ]
   }
   
   with open(results_file, 'w') as f:
      json.dump(results, f, indent=2)
   
   print(f"\nResults saved to: {results_file}")
      
   return study

run_optuna_study()