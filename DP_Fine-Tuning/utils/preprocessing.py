import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.utils.data import DataLoader

def collate_eval(batch):
    """
    Collate function that keeps the target_text as a list, while batching input_ids etc.
    """
    input_ids = torch.stack([torch.tensor(b["input_ids"]) for b in batch])
    attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch])

    target_texts = [b["target_text"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_text": target_texts,
    }



def preprocess_dataset(
   tokenizer,
   dataset_name,
   train_size,
   eval_size,
   max_input_length,
   max_target_length,
   train_batch_size,
   eval_batch_size,
   seed=101
):
   """
   Preprocess dataset for training/evaluation with chat template.
   Returns train_loader, val_loader.
   """
   dataset = load_dataset(dataset_name)
   dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(train_size))
   dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(eval_size))

   # Train tokenizer
   def tokenize_train(example):
      messages = [
         {"role": "system", "content": "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers."},
         {"role": "user", "content": f"Context: {example['context']} Question: {example['question']}"}
      ]

      input_text = tokenizer.apply_chat_template(
         messages,
         add_generation_prompt=True, 
         tokenize=False
      )

      target_text = example["answers"]["text"][0] if example["answers"]["text"] else ""
      messages_with_answer = messages + [{"role": "assistant", "content": target_text}]
      full_text = tokenizer.apply_chat_template(messages_with_answer, tokenize=False)

      # Tokenize input to get exact length for masking
      input_tokenized = tokenizer(
         input_text, 
         max_length=max_input_length, 
         truncation=True, 
         padding=False, 
         add_special_tokens=False
      )

      # Tokenize input and full conversation
      tokenized = tokenizer(
         full_text,
         max_length=max_input_length + max_target_length, 
         truncation=True, 
         padding="max_length", 
         add_special_tokens=False
      )

      # Create labels
      input_length = len(input_tokenized["input_ids"])

      labels = tokenized["input_ids"].copy()
      labels[:input_length] = [-100] * input_length
      labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

      tokenized["labels"] = labels
      return tokenized

   # Eval tokenizer
   def tokenize_eval(example):
      messages = [
         {"role": "system", "content": "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers."},
         {"role": "user", "content": f"Context: {example['context']} Question: {example['question']}"}
      ]

      # Get input text
      input_text = tokenizer.apply_chat_template(
         messages, 
         add_generation_prompt=True, 
         tokenize=False
      )

      # Tokenize input only
      tokenized = tokenizer(
         input_text, 
         max_length=max_input_length, 
         truncation=True, 
         padding="max_length", 
         add_special_tokens=False
      )

      # Store the target answer for evaluation metrics (but don't include in labels)
      target_text = example["answers"]["text"][0] if example["answers"]["text"] else ""
      tokenized["target_text"] = target_text

      # Tokenize target for BLEU/ROUGE metrics if needed
      target_tokenized = tokenizer(
         target_text, 
         max_length=max_target_length, 
         truncation=True, 
         padding="max_length", 
         add_special_tokens=False
      )
      tokenized["target_input_ids"] = target_tokenized["input_ids"]
      tokenized["target_attention_mask"] = target_tokenized["attention_mask"]

      return tokenized

   # Map datasets
   train_dataset = dataset["train"].map(
      tokenize_train, 
      batched=False, 
      remove_columns=dataset["train"].column_names
   )
   
   eval_dataset = dataset["validation"].map(
      tokenize_eval, 
      batched=False, 
      remove_columns=dataset["validation"].column_names
   )

   # DataLoaders
   train_loader = DataLoader(
      train_dataset, 
      batch_size=train_batch_size, 
      shuffle=True, 
      collate_fn=default_data_collator
   )

   val_loader = DataLoader(
      eval_dataset, 
      batch_size=eval_batch_size, 
      collate_fn=collate_eval
   )

   return train_loader, val_loader
