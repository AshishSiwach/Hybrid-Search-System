# upload_to_hf.py
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()  # reads from .env file

TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise ValueError("HF_TOKEN not set — add it to your .env file or environment")

REPO = "TripleH/querylens-data"

api = HfApi(token=TOKEN)

files = [
    ("ms_marco_passages.json", "data/ms_marco_passages.json"),
    ("ms_marco_queries.json",  "data/ms_marco_queries.json"),
    ("doc_embeddings.npy",     "data/doc_embeddings.npy"),
    ("faiss_hnsw.index",       "data/faiss_hnsw.index"),
    ("bm25_index.pkl",         "data/bm25_index.pkl"),
]

for remote_name, local_path in files:
    if not os.path.exists(local_path):
        print(f"MISSING: {local_path} — skipping")
        continue
    size = os.path.getsize(local_path) / (1024 * 1024)
    print(f"Uploading {remote_name} ({size:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_name,
        repo_id=REPO,
        repo_type="dataset",
        token=TOKEN,
    )
    print(f"Done: {remote_name}")

print("\nAll files uploaded.")