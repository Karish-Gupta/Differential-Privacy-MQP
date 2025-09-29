import numpy as np
import torch
from tqdm import tqdm
import random

def calculate_exact_match(preds, refs):
    """
    Returns exact match accuracy
    """
    return np.mean([1 if p.lower() == r.lower() else 0 for p, r in zip(preds, refs)])


def calculate_contains_acc(preds, refs):
    """
    Returns contains accuracy
    """
    return np.mean([1 if r.lower() in p.lower() or p.lower() in r.lower() else 0 for p, r in zip(preds, refs)])

def calculate_f1(preds, refs):
    """
    Returns f1 score
    """
    f1_scores = []
    for p, r in zip(preds, refs):
        ptoks, rtoks = p.lower().split(), r.lower().split()
        common = set(ptoks) & set(rtoks)
        num_common = sum(min(ptoks.count(t), rtoks.count(t)) for t in common)
        if num_common == 0:
            f1_scores.append(0.0)
        else:
            prec = num_common / len(ptoks)
            rec = num_common / len(rtoks)
            f1_scores.append(2 * prec * rec / (prec + rec))
    return np.mean(f1_scores)



def evaluate_model(model, val_loader, device, tokenizer, max_gen_length=50, show_samples=5, seed=101):
    model.eval()
    preds, refs = [], []

    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_texts = batch["target_text"]  # Ground truth answers from preprocessing

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        for i in range(input_ids.shape[0]):
            # Slice generated tokens after the prompt
            generated_ids = outputs[i, input_ids.shape[1]:]
            generated_ids = generated_ids[generated_ids != tokenizer.pad_token_id]
            if len(generated_ids) > 0 and generated_ids[-1] == tokenizer.eos_token_id:
                generated_ids = generated_ids[:-1]
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Get reference from target_text
            ref = target_texts[i].strip()
            
            # # Cut off everything before "Answer: "
            # if "Answer:" in pred:
            #     pred = pred.split("Answer:", 1)[1].strip()

            preds.append(pred)
            refs.append(ref)

    # Metrics
    exact_match = calculate_exact_match(preds, refs)
    contains_acc = calculate_contains_acc(preds, refs)
    f1 = calculate_f1(preds, refs)

    print(f"\nContains Accuracy: {contains_acc:.4f}")
    print(f"Exact Match Accuracy: {exact_match:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Sample outputs
    if show_samples > 0:
        print("\nSample predictions:\n")
        random.seed(seed)
        for i in random.sample(range(len(preds)), min(show_samples, len(preds))):
            print("=" * 80)
            print(f"Gold Answer: {refs[i]}")
            print(f"Predicted Answer: {preds[i]}")

    return {"contains_accuracy": contains_acc, "exact_match_accuracy": exact_match, "f1": f1}