import numpy as np
import torch
from tqdm import tqdm
import random
from sklearn.metrics import f1_score
import threading
import time
import csv
import subprocess

def exact_match(pred, ref):
    """
    Returns True if prediction exactly matches reference
    """
    return pred.strip().lower() == ref.strip().lower()

def contains_token(pred, ref):
    """
    Returns True if the reference tokens appear in order inside prediction tokens
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


def evaluate_model(model, val_loader, device, tokenizer, max_gen_length=50, show_samples=5, seed=101):
    model.eval()
    preds, refs = [], []

    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

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

            # Decode gold answer from labels
            label_ids = labels[i]
            answer_ids = [lid.item() for lid in label_ids if lid.item() != -100]
            ref = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            
            # Cut off everything before "Answer: "
            if "Answer:" in pred:
                pred = pred.split("Answer:", 1)[1].strip()

            preds.append(pred)
            refs.append(ref)

    # Metrics
    exact_match = np.mean([1 if p.lower() == r.lower() else 0 for p, r in zip(preds, refs)])
    contains_acc = np.mean([1 if r.lower() in p.lower() or p.lower() in r.lower() else 0 for p, r in zip(preds, refs)])

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
    f1 = np.mean(f1_scores)

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

    
def start_gpu_utilization_logging(logfile="gpu_utilization_debug.csv", interval=1.0, util_data=None, stop_event=None):
    """
    Start a background thread to log GPU utilization every `interval` seconds.
    Returns (thread, stop_event, util_data).
    """
    if stop_event is None:
        stop_event = threading.Event()
    if util_data is None:
        util_data = []
    def _gpu_utilization_logger():
        # Write CSV header
        with open(logfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu_index", "gpu_name", "utilization", "mem_used", "mem_total"])
        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
                )
                timestamp = time.time()
                lines = result.stdout.strip().split('\n')
                with open(logfile, "a", newline="") as f:
                    writer = csv.writer(f)
                    for line in lines:
                        idx, name, util, mem_used, mem_total = [x.strip() for x in line.split(',')]
                        writer.writerow([timestamp, idx, name, util, mem_used, mem_total])
                        util_data.append((float(idx), float(util)))
            except Exception:
                pass
            time.sleep(interval)
    thread = threading.Thread(target=_gpu_utilization_logger, daemon=True)
    thread.start()
    return thread, stop_event, util_data

def stop_gpu_utilization_logging(thread, stop_event):
    """
    Stop the background GPU utilization logging thread.
    """
    stop_event.set()
    if thread is not None:
        thread.join()

def print_gpu_utilization_summary(util_data):
    """
    Compute and print the average GPU utilization for each GPU and overall.
    """
    if not util_data:
        print("No GPU utilization data collected.")
        return
    from collections import defaultdict
    util_per_gpu = defaultdict(list)
    for idx, util in util_data:
        util_per_gpu[idx].append(util)
    print("\n[GPU UTILIZATION SUMMARY]")
    total_sum = 0
    total_count = 0
    for idx in sorted(util_per_gpu.keys()):
        vals = util_per_gpu[idx]
        avg = sum(vals) / len(vals)
        print(f"GPU {int(idx)}: Average Utilization {avg:.2f}% over {len(vals)} samples")
        total_sum += sum(vals)
        total_count += len(vals)
    if total_count > 0:
        overall_avg = total_sum / total_count
        print(f"Overall Average Utilization: {overall_avg:.2f}%")
    print()