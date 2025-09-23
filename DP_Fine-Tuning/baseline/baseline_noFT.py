import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from ..utils.model_utils import *
from ..utils.gpu_usage import *
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
# transformers.utils.logging.set_verbosity_debug()


class Baseline_no_fine_tuning:
    def __init__(
        self,
        model_name,
        dataset_name,
        eval_batch_size,
        max_input_length,
        max_target_length,
    ):
        # Configs
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.eval_batch_size = eval_batch_size
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup
        self.tokenizer = None
        self.val_loader = None
        self.model = None

    def preprocess_dataset(self, subsample_size, seed=101):
        dataset = load_dataset(self.dataset_name)

        if subsample_size is not None:
            dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(max(subsample_size // 10, 1)))

        # Initialize tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        system_prompt = "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, in format Answer: <answer>"
        
        def preprocess_and_tokenize_eval(example):
            # Prompt only (no gold answer appended to input_ids)
            input_text = system_prompt + " Context: " + example["context"] + " Question: " + example["question"] + " Answer: "
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


        eval_dataset = dataset["validation"].map(
            preprocess_and_tokenize_eval, 
            batched=False,
            remove_columns=dataset["validation"].column_names
        )
        
        self.val_loader = DataLoader(
            eval_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=default_data_collator,
        )

    def init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="cuda:0")
        self.model = self.model.to(torch.float16)

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
            max_gen_length=10,
            show_samples=10,
        )


if __name__ == "__main__":
    # Model Configs
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name = "squad"
    eval_batch_size = 1
    max_input_length = 512
    max_target_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_size = 5000
    eval_size = 500


    baseline_model_noFT = Baseline_no_fine_tuning(
        model_name=model_name,
        dataset_name=dataset_name,
        eval_batch_size=eval_batch_size,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    # Start GPU utilization logging using utils
    gpu_util_thread, gpu_util_stop_event, gpu_util_data = start_gpu_utilization_logging(interval=1.0)

    baseline_model_noFT.preprocess_dataset(train_size=train_size, eval_size=eval_size, seed=101)
    baseline_model_noFT.init_model()

    print(f"Model: {model_name}")
    print(f"On device: {device}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Max input length: {max_input_length}")
    print(f"Max target length: {max_target_length}")
    print(f"Traing size: {train_size}")
    print(f"Eval size: {eval_size}")
    baseline_model_noFT.evaluate()

    # Output GPU logging
    stop_gpu_utilization_logging(gpu_util_thread, gpu_util_stop_event)
    print_gpu_utilization_summary(gpu_util_data)
