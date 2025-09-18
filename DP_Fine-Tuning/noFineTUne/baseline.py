import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from utils import evaluate_model
from huggingface_hub import login
import os


print("Logging into Hugging Face Hub...")
login(token=os.environ["HF_TOKEN"])

class BaselineEval:
    def __init__(self, model_name, dataset_name, eval_batch_size, max_input_length, max_target_length):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.eval_batch_size = eval_batch_size
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.val_loader = None

    def preprocess_dataset(self, subsample_size=None, seed=101):
        dataset = load_dataset(self.dataset_name)

        if subsample_size:
            dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(subsample_size))

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        def preprocess_and_tokenize(example):
            title = example.get("title", "")
            input_text = f"Title: {title} Question: {example['question']} Context: {example['context']} Answer:"
            answer = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
            eos_token = self.tokenizer.eos_token or "<|eos|>"
            target_text = answer + eos_token

            # Build full sequence
            full_text = input_text + target_text
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_input_length + self.max_target_length,
                padding="max_length",
            )
            return tokenized

        dataset = dataset.map(
            preprocess_and_tokenize,
            batched=False,
            remove_columns=dataset["validation"].column_names,
        )

        self.val_loader = DataLoader(
            dataset["validation"],
            batch_size=self.eval_batch_size,
            collate_fn=default_data_collator,
        )

    def init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.model = self.model.to(torch.float16)
        self.model.eval()

    def evaluate(self):
        if self.val_loader is None:
            print("Run preprocess_dataset() first.")
            return
        model_device = next(self.model.parameters()).device
        evaluate_model(
            self.model,
            self.val_loader,
            model_device,
            self.tokenizer,
            max_gen_length=30,
            show_samples=10,
        )


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    dataset_name = "rajpurkar/squad"
    eval_batch_size = 1
    max_input_length = 512
    max_target_length = 512

    baseline_eval = BaselineEval(
        model_name=model_name,
        dataset_name=dataset_name,
        eval_batch_size=eval_batch_size,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    baseline_eval.preprocess_dataset(subsample_size=500, seed=101)
    baseline_eval.init_model()
    baseline_eval.evaluate()
