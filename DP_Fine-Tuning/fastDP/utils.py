import numpy as np
import torch
from tqdm import tqdm
import random
from sklearn.metrics import f1_score


def contains_token(pred, ref):
    """
    Returns True if the reference tokens appear in order inside prediction tokens.
    """
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()

    if not ref_tokens:
        return False

    # Sliding window check
    for i in range(len(pred_tokens) - len(ref_tokens) + 1):
        if pred_tokens[i:i+len(ref_tokens)] == ref_tokens:
            return True
    return False


# Eval function updated for different data collator
def evaluate_exact_match(model, val_loader, device, tokenizer, max_gen_length=50): 
    model.eval()
    em_scores = []

    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
            )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        decoded_labels = []
        for label_ids in labels:
            label_ids = [l for l in label_ids.tolist() if l != -100]
            text = tokenizer.decode(label_ids, skip_special_tokens=True)
            decoded_labels.append(text)

        for pred, ref in zip(preds, decoded_labels):
            pred_norm = pred.strip().lower()
            ref_norm = ref.strip().lower()
            em = 1 if contains_token(pred_norm, ref_norm) else 0
            em_scores.append(em)

    exact_match = np.mean(em_scores)
    print(f"Exact Match Accuracy: {exact_match:.4f}")
    return exact_match


# Token-level F1 score (macro)
def evaluate_f1(model, val_loader, device, tokenizer, max_gen_length=50):
    model.eval()
    all_preds = []
    all_refs = []

    for batch in tqdm(val_loader, desc="Evaluating F1"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
            )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        decoded_labels = []
        for label_ids in labels:
            label_ids = [l for l in label_ids.tolist() if l != -100]
            text = tokenizer.decode(label_ids, skip_special_tokens=True)
            decoded_labels.append(text)

        all_preds.extend(preds)
        all_refs.extend(decoded_labels)

    pred_tokens = [p.strip().lower().split() for p in all_preds]
    ref_tokens = [r.strip().lower().split() for r in all_refs]

    y_true, y_pred = [], []
    for ref, pred in zip(ref_tokens, pred_tokens):
        ref_set = set(ref)
        pred_set = set(pred)
        all_tokens = list(ref_set | pred_set)
        for token in all_tokens:
            y_true.append(token in ref_set)
            y_pred.append(token in pred_set)

    f1 = f1_score(y_true, y_pred)
    print(f"Token-level F1 Score: {f1:.4f}")
    return f1


def evaluate_model(model, val_loader, device, tokenizer, max_gen_length=50, show_samples=20, seed=42):
    model.eval()
    em_scores = []
    all_preds, all_refs = [], []

    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
            )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        decoded_labels = []
        for label_ids in labels:
            label_ids = [l for l in label_ids.tolist() if l != -100]
            text = tokenizer.decode(label_ids, skip_special_tokens=True)
            decoded_labels.append(text)

        # collect for F1
        all_preds.extend(preds)
        all_refs.extend(decoded_labels)

        # compute EM batch-wise
        for pred, ref in zip(preds, decoded_labels):
            pred_norm = pred.strip().lower()
            ref_norm = ref.strip().lower()
            em = 1 if contains_token(pred_norm, ref_norm) else 0
            em_scores.append(em)

    # --- Aggregate metrics ---
    exact_match = np.mean(em_scores)

    pred_tokens = [p.strip().lower().split() for p in all_preds]
    ref_tokens = [r.strip().lower().split() for r in all_refs]

    y_true, y_pred = [], []
    for ref, pred in zip(ref_tokens, pred_tokens):
        ref_set = set(ref)
        pred_set = set(pred)
        all_tokens = list(ref_set | pred_set)
        for token in all_tokens:
            y_true.append(token in ref_set)
            y_pred.append(token in pred_set)

    f1 = f1_score(y_true, y_pred)

    print(f"\nExact Match Accuracy: {exact_match:.4f}")
    print(f"Token-level F1 Score: {f1:.4f}")

    # --- Show sample predictions ---
    if show_samples > 0:
        print("\nSample predictions:\n")
        random.seed(seed)
        indices = random.sample(range(len(all_preds)), min(show_samples, len(all_preds)))
        for i in indices:
            print("=" * 80)
            print(f"Gold Answer: {all_refs[i]}")
            print(f"Predicted Answer: {all_preds[i]}")

    return {"exact_match": exact_match, "f1": f1}
