import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from utils.model_utils import *
from utils.gpu_usage import *
from utils.preprocessing import collate_eval
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

    def preprocess_dataset(self, eval_size, seed=101):
        dataset = load_dataset(self.dataset_name)

        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(eval_size))

        # Initialize tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
        # Eval tokenizer
        def tokenize_eval(example):
            messages = [
                {"role": "system", "content": "You are a knowledgeable, concise, and direct AI assistant. Provide just the short answer and nothing else."
},
                {"role": "user", "content": f"Context: {example['context']} Question: {example['question']}"}
            ]

            # Get input text
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )

            # Tokenize input only
            tokenized = self.tokenizer(
                input_text, 
                max_length=max_input_length, 
                truncation=True, 
                padding="max_length", 
                add_special_tokens=False
            )

            # Store the target answer for evaluation metrics (but don't include in labels)
            target_text = example["answers"]["text"][0] if example["answers"]["text"] else ""
            tokenized["target_text"] = target_text

            return tokenized
        
        eval_dataset = dataset["validation"].map(
            tokenize_eval, 
            batched=False, 
            remove_columns=dataset["validation"].column_names
        )

        self.val_loader = DataLoader(
            eval_dataset, 
            batch_size=eval_batch_size, 
            collate_fn=collate_eval
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
            max_gen_length=64,
            show_samples=10,
        )


if __name__ == "__main__":
    # Model Configs
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name = "squad"
    eval_batch_size = 8
    max_input_length = 512
    max_target_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    baseline_model_noFT.preprocess_dataset(eval_size=eval_size, seed=101)
    baseline_model_noFT.init_model()

    print(f"Model: {model_name}")
    print(f"On device: {device}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Max input length: {max_input_length}")
    print(f"Max target length: {max_target_length}")
    print(f"Eval size: {eval_size}")
    baseline_model_noFT.evaluate()

    # Output GPU logging
    stop_gpu_utilization_logging(gpu_util_thread, gpu_util_stop_event)
    print_gpu_utilization_summary(gpu_util_data)
