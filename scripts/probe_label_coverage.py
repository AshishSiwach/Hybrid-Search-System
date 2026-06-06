"""
probe_label_coverage.py — measure how sparse the MS MARCO labels are
relative to what the cross-encoder considers relevant.

The question this answers
-------------------------
After calibration, we have:
  - 10,000 (query, candidate) pairs
  - 215 of them labelled relevant (is_selected = 1)
  - F1 max only 0.29

Two possible explanations:
  (a) The cross-encoder is genuinely uncertain — labels are accurate,
      the model is just imperfect.
  (b) The labels are SPARSE — many passages are truly relevant but
      unlabelled, so the cross-encoder is being penalised for finding
      them.

This script measures (b) directly. At each score threshold T it asks:
"Of the candidates the model scored >= T, what fraction are labelled?"

If at high T the ratio is near 100%, labels are dense — most things the
model finds confidently are labelled. The model is the bottleneck.

If at high T the ratio is well below 100% (say 30-50%), labels are
sparse — many things the model finds confidently are NOT labelled.
The labels are the bottleneck.

Caching
-------
First run reuses the calibration's scoring logic — slow on CPU (~25-40
minutes). After that, raw (score, label) pairs are cached to
results/calibration_pairs.npz; subsequent runs are instant.

If you swap the cross-encoder or rebuild the corpus, delete the cache
file to force a fresh compute.

Usage
-----
    python scripts/probe_label_coverage.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Make `querylens` importable when this script is run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

from querylens.data_loader import MSMarcoLoader, load_bm25_index
from querylens.reranker    import CrossEncoderReranker
from querylens.retrievers  import BM25Retriever, DenseRetriever, HybridRetriever


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_PATH = RESULTS_DIR / "calibration_pairs.npz"

# Thresholds we'll probe — covers the interesting range of cross-encoder logits
PROBE_THRESHOLDS = [-2.0, 0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0]

CPU_BATCH = 16
GPU_BATCH = 64


# ── 1. Pipeline setup (mirrors calibrate_thresholds.py) ──────────────────────

def detect_device() -> Tuple[str, int]:
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA detected: %s", torch.cuda.get_device_name(0))
            return "cuda", GPU_BATCH
    except ImportError:
        pass
    logger.info("Running on CPU")
    return "cpu", CPU_BATCH


def load_config() -> dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def apply_device(config: dict, device: str, batch_size: int) -> dict:
    config["models"]["cross_encoder"]["device"]     = device
    config["models"]["cross_encoder"]["batch_size"] = batch_size
    config["models"]["dense_encoder"]["device"]     = device
    return config


def build_pipeline(config: dict):
    logger.info("Loading corpus...")
    loader = MSMarcoLoader(config)
    passages, _, _, _ = loader.load()

    bm25 = BM25Retriever(config)
    bm25_path = config["data"]["index_path"]
    if not Path(bm25_path).exists():
        raise FileNotFoundError("BM25 index missing — run scripts/main.py first.")
    bm25.index = load_bm25_index(bm25_path)
    bm25.tokenized_corpus = [p.lower().split() for p in passages]

    dense = DenseRetriever(config)
    faiss_path = config["data"]["faiss_index_path"]
    if not Path(faiss_path).exists():
        raise FileNotFoundError("FAISS index missing — run scripts/main.py first.")
    dense.load_faiss_index(faiss_path)

    hybrid   = HybridRetriever(bm25, dense, config)
    reranker = CrossEncoderReranker(config)
    return loader, passages, hybrid, reranker


# ── 2. Score collection (or load from cache) ─────────────────────────────────

def compute_pairs(loader, passages, hybrid, reranker, config) -> Tuple[np.ndarray, np.ndarray]:
    """Score every candidate of every test query. Identical logic to calibrate_thresholds."""
    max_q   = config["evaluation"]["test_queries"]
    queries = loader.get_test_queries(max_q)
    labels  = loader.build_relevance_labels()

    all_scores: List[float] = []
    all_relevance: List[int] = []

    for query_id, query_text in tqdm(queries, desc="Scoring candidates"):
        if query_id not in labels:
            continue
        relevant_set = {idx for idx, rel in labels[query_id].items() if rel > 0}
        if not relevant_set:
            continue

        candidate_indices, _ = hybrid.retrieve(query_text)
        if len(candidate_indices) == 0:
            continue

        pairs = [[query_text, passages[int(i)]] for i in candidate_indices]
        scores = reranker.model.predict(
            pairs,
            batch_size=reranker.batch_size,
            show_progress_bar=False,
        )

        for idx, score in zip(candidate_indices, scores):
            all_scores.append(float(score))
            all_relevance.append(1 if int(idx) in relevant_set else 0)

    return np.array(all_scores), np.array(all_relevance)


def load_or_compute_pairs(config) -> Tuple[np.ndarray, np.ndarray]:
    if CACHE_PATH.exists():
        logger.info("Loading cached pairs from %s", CACHE_PATH)
        d = np.load(CACHE_PATH)
        return d["scores"], d["labels"]

    logger.info("No cache found — computing pairs from scratch (slow on CPU)")
    device, batch_size = detect_device()
    config = apply_device(config, device, batch_size)
    loader, passages, hybrid, reranker = build_pipeline(config)

    scores, labels = compute_pairs(loader, passages, hybrid, reranker, config)

    np.savez(CACHE_PATH, scores=scores, labels=labels)
    logger.info("Cached %d pairs to %s — future runs will be instant", len(scores), CACHE_PATH)
    return scores, labels


# ── 3. The probe ─────────────────────────────────────────────────────────────

def probe(scores: np.ndarray, labels: np.ndarray) -> List[Dict]:
    """For each threshold, count above / labelled-above, return summary rows."""
    rows = []
    for t in PROBE_THRESHOLDS:
        above_mask     = scores >= t
        above          = int(above_mask.sum())
        labelled_above = int((above_mask & (labels == 1)).sum())
        ratio          = labelled_above / above if above > 0 else float("nan")
        rows.append({
            "threshold":      t,
            "above":          above,
            "labelled_above": labelled_above,
            "unlabelled_above": above - labelled_above,
            "ratio":          ratio,
        })
    return rows


def estimate_true_relevance_per_query(scores: np.ndarray, labels: np.ndarray, n_queries: int) -> Dict:
    """
    Rough per-query estimate at the F1-optimal-ish threshold (7.0).

    "above_per_query" — how many candidates the model deems strong per query
    "labelled_per_query" — average labels we have per query
    "inferred_true_per_query" — IF unlabelled high-scorers are mostly true
                                positives, this is the implied true count
    """
    T = 7.0
    above_mask  = scores >= T
    above       = int(above_mask.sum())
    labelled    = int((above_mask & (labels == 1)).sum())
    return {
        "threshold":              T,
        "above_per_query":        round(above / n_queries, 2),
        "labelled_per_query":     round(labelled / n_queries, 2),
        "ratio":                  round(labelled / above, 3) if above else None,
        "unlabelled_per_query":   round((above - labelled) / n_queries, 2),
    }


# ── 4. Reporting ─────────────────────────────────────────────────────────────

def print_report(rows: List[Dict], est: Dict, scores: np.ndarray, labels: np.ndarray, n_queries: int):
    W = 70
    print()
    print("═" * W)
    print("  LABEL COVERAGE PROBE")
    print("═" * W)
    print(f"  Total (query, candidate) pairs : {len(scores):>8,}")
    print(f"  Queries evaluated              : {n_queries:>8,}")
    print(f"  Labelled relevant (total)      : {int(labels.sum()):>8,}   ({labels.mean():.2%} of pairs)")
    print(f"  Labelled relevant per query    : {labels.sum() / n_queries:>8.2f}")
    print()
    print("  Score threshold sweep — 'how often does a high-scoring")
    print("  candidate also carry a relevance label?'")
    print()
    print(f"  {'≥ T':>6}  {'above':>8}  {'labelled':>10}  {'unlabelled':>12}  {'ratio':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*8}")
    for r in rows:
        ratio_str = f"{r['ratio']:.1%}" if not np.isnan(r['ratio']) else "  n/a"
        print(f"  {r['threshold']:>6.1f}  {r['above']:>8,}  {r['labelled_above']:>10,}  "
              f"{r['unlabelled_above']:>12,}  {ratio_str:>8}")
    print()
    print(f"  Per-query view at threshold {est['threshold']:.1f}:")
    print(f"    Model deems strong : {est['above_per_query']} candidates/query (avg)")
    print(f"    Labelled relevant  : {est['labelled_per_query']} candidates/query (avg)")
    print(f"    Unlabelled-strong  : {est['unlabelled_per_query']} candidates/query (avg)")
    print()

    # Interpretation
    print("  Interpretation:")
    high_thresh_ratio = next(r['ratio'] for r in rows if r['threshold'] == 9.0)
    mid_thresh_ratio  = next(r['ratio'] for r in rows if r['threshold'] == 7.0)
    if high_thresh_ratio > 0.7:
        print(f"    At score ≥ 9.0, {high_thresh_ratio:.0%} are labelled.")
        print(f"    Labels are dense at the high end — the model is the bottleneck.")
    elif high_thresh_ratio > 0.4:
        print(f"    At score ≥ 9.0, only {high_thresh_ratio:.0%} are labelled.")
        print(f"    Labels miss a meaningful fraction even at high scores —")
        print(f"    the cross-encoder is somewhat more confident than the labels reward.")
    else:
        print(f"    At score ≥ 9.0, only {high_thresh_ratio:.0%} are labelled.")
        print(f"    Labels are quite sparse even at the high end — the cross-encoder")
        print(f"    is significantly more confident than the labels reward.")
    print()
    print(f"    At score ≥ 7.0 (F1-optimal region), {mid_thresh_ratio:.0%} are labelled.")
    print(f"    Implication: about 1/{mid_thresh_ratio:.2f} ≈ {1/mid_thresh_ratio:.1f}× as many candidates")
    print(f"    look strong to the model as carry an explicit label.")
    print("═" * W)
    print()


def plot(rows: List[Dict], scores: np.ndarray, labels: np.ndarray, out_path: Path):
    """Two-panel: (left) stacked counts above each threshold; (right) ratio curve."""
    DARK = {"bg": "#0f0f0f", "panel": "#1a1a1a", "text": "#e5e5e5",
            "grid": "#2a2a2a", "labelled": "#2dc77a", "unlabelled": "#f87171",
            "ratio": "#facc15"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(DARK["bg"])
    fig.suptitle("QueryLens · Label Coverage Probe",
                 color=DARK["text"], fontsize=14, fontweight="bold", y=0.98)

    for ax in (ax1, ax2):
        ax.set_facecolor(DARK["panel"])
        for sp in ax.spines.values():
            sp.set_edgecolor(DARK["grid"])
        ax.tick_params(colors=DARK["text"], labelsize=9)
        ax.xaxis.label.set_color(DARK["text"])
        ax.yaxis.label.set_color(DARK["text"])
        ax.grid(color=DARK["grid"], linewidth=0.5, alpha=0.7)

    # Panel 1: stacked bars at each probe threshold
    ax1.set_title("Candidates above threshold: labelled vs unlabelled",
                  color=DARK["text"], fontsize=11, fontweight="bold", pad=10)
    thresholds = [r["threshold"] for r in rows]
    labelled   = [r["labelled_above"]   for r in rows]
    unlabelled = [r["unlabelled_above"] for r in rows]
    x = np.arange(len(thresholds))
    ax1.bar(x, labelled,   color=DARK["labelled"],   label="Labelled relevant", width=0.7)
    ax1.bar(x, unlabelled, color=DARK["unlabelled"], label="Unlabelled (high-scoring but no label)",
            width=0.7, bottom=labelled)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t:g}" for t in thresholds])
    ax1.set_xlabel("Score threshold")
    ax1.set_ylabel("Candidate count")
    ax1.legend(fontsize=9, facecolor=DARK["panel"], labelcolor=DARK["text"],
               framealpha=0.9, edgecolor=DARK["grid"])

    # Panel 2: ratio curve
    ax2.set_title("Label coverage ratio: labelled / above-threshold",
                  color=DARK["text"], fontsize=11, fontweight="bold", pad=10)
    ratios = [r["ratio"] for r in rows]
    ax2.plot(thresholds, ratios, color=DARK["ratio"], lw=2, marker="o", markersize=6)
    for r in rows:
        if not np.isnan(r["ratio"]):
            ax2.annotate(
                f"{r['ratio']:.0%}",
                (r['threshold'], r['ratio']),
                textcoords="offset points", xytext=(0, 8),
                ha="center", color=DARK["text"], fontsize=8,
            )
    ax2.set_xlabel("Score threshold")
    ax2.set_ylabel("Labelled / above (fraction)")
    ax2.set_ylim(0, max(0.5, max(r for r in ratios if not np.isnan(r)) * 1.2))
    ax2.axhline(1.0, color=DARK["grid"], linewidth=0.5)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved plot → %s", out_path)


# ── 5. Main ──────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    scores, labels = load_or_compute_pairs(config)

    n_queries = config["evaluation"]["test_queries"]
    rows = probe(scores, labels)
    est  = estimate_true_relevance_per_query(scores, labels, n_queries)

    print_report(rows, est, scores, labels, n_queries)

    # Save JSON summary
    out_json = RESULTS_DIR / "label_coverage.json"
    with open(out_json, "w") as f:
        json.dump({"rows": rows, "estimate_at_T7": est,
                   "n_pairs": int(len(scores)),
                   "n_labelled": int(labels.sum()),
                   "n_queries": int(n_queries)}, f, indent=2)
    logger.info("Saved JSON → %s", out_json)

    # Save plot
    plot(rows, scores, labels, RESULTS_DIR / "label_coverage.png")

    print(f"  Plot → results/label_coverage.png")
    print(f"  JSON → results/label_coverage.json")
    print(f"  Cache → results/calibration_pairs.npz (delete to force recompute)")
    print()


if __name__ == "__main__":
    main()
