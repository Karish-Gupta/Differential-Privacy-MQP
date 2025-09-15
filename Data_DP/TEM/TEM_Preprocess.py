import sys
import os
import urllib.request
import zipfile

from datasets import load_dataset
import csv
from pathlib import Path

def download_glove_if_missing(embedding_path):
        """Download GloVe embeddings if not present."""
        target_file = embedding_path
        if os.path.exists(target_file):
            print(f"[INFO] GloVe embeddings already available at {target_file}")
            return

        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        zip_path = os.path.join(os.path.dirname(target_file), "glove.6B.zip")

        if not os.path.exists(zip_path):
            url = "https://nlp.stanford.edu/data/glove.6B.zip"
            print(f"[INFO] Downloading GloVe embeddings from {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            print("[INFO] Download complete.")

        print("[INFO] Extracting GloVe embeddings...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(target_file))
        print("[INFO] Extraction complete.")

def make_csv():
    out_path = Path("squad_for_pypantera.csv")
    ds = load_dataset("rajpurkar/squad")

    # write train (or validation) — here I use train subset
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        for i, ex in enumerate(ds["train"]):
            question = ex.get("question","")
            context = ex.get("context","")
            text = f"Question: {question} Context: {context}"
            writer.writerow({"id": str(i), "text": text})

    print("Wrote:", out_path.resolve())

embedding_path = "./TEM/glove.6B.100d.txt"

download_glove_if_missing(embedding_path)
make_csv()