"""
calibrate_thresholds.py — empirically calibrate Layer 2 confidence thresholds.

Runs the Hybrid + Reranker pipeline over your test queries, collects a
(cross-encoder score, is_relevant) pair for every candidate scored, then
derives thresholds from actual data instead of guesswork.

Outputs
-------
  results/threshold_calibration.png   — 4-panel visualization
  results/threshold_calibration.json  — recommended thresholds + stats

Usage
-----
  python calibrate_thresholds.py

Prerequisites: run main.py at least once so the FAISS and BM25 indexes exist.
"""

import json
import logging
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple

# Batch sizes per device — cross-encoder ms-marco-MiniLM-L-6-v2 is tiny (~80 MB),
# so a 6 GB GPU comfortably handles 64.  Bump GPU_BATCH if you have more VRAM.
CPU_BATCH = 16
GPU_BATCH = 64

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):          # silent fallback if tqdm not installed
        return it

from data_loader import MSMarcoLoader, load_embeddings, load_bm25_index
from retrievers  import BM25Retriever, DenseRetriever, HybridRetriever
from reranker    import CrossEncoderReranker
from decision    import DecisionLayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Pulled live from decision.py so the "current" lines on the plot always
# reflect what's actually deployed — no drift after you update the thresholds.
CURRENT = {
    "high":   DecisionLayer.HIGH_THRESHOLD,
    "medium": DecisionLayer.MEDIUM_THRESHOLD,
    "low":    DecisionLayer.LOW_THRESHOLD,
}


# ── 1. Pipeline loader ────────────────────────────────────────────────────────

def load_config() -> dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def detect_device() -> Tuple[str, int]:
    """
    Auto-detect CUDA. Returns (device_string, recommended_batch_size).

    Falls back to CPU silently if:
      - torch is missing (shouldn't happen — sentence-transformers depends on it)
      - CUDA is unavailable (no GPU, or torch built without CUDA support)
    """
    try:
        import torch
        if torch.cuda.is_available():
            name      = torch.cuda.get_device_name(0)
            vram_gb   = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_ver  = torch.version.cuda
            logger.info(
                "CUDA detected: %s  (%.1f GB VRAM, CUDA %s)",
                name, vram_gb, cuda_ver,
            )
            return "cuda", GPU_BATCH
    except ImportError:
        pass
    logger.info("No CUDA GPU available — running on CPU")
    return "cpu", CPU_BATCH


def apply_device(config: dict, device: str, batch_size: int) -> dict:
    """
    Override both encoders' device + batch_size in the in-memory config.
    Leaves configs/config.yaml on disk untouched.
    """
    config["models"]["cross_encoder"]["device"]     = device
    config["models"]["cross_encoder"]["batch_size"] = batch_size

    config["models"]["dense_encoder"]["device"]     = device
    # Dense encoder only runs once per query (200 calls total) — keep its
    # batch size at config default unless the user gave us a bigger one.
    config["models"]["dense_encoder"]["batch_size"] = max(
        batch_size,
        config["models"]["dense_encoder"].get("batch_size", 32),
    )
    return config


def build_pipeline(config: dict):
    logger.info("Loading corpus...")
    loader = MSMarcoLoader(config)
    passages, _, _, _ = loader.load()

    logger.info("Loading BM25 index...")
    bm25 = BM25Retriever(config)
    bm25_path = config["data"]["index_path"]
    if not Path(bm25_path).exists():
        raise FileNotFoundError("BM25 index missing — run main.py first.")
    bm25.index = load_bm25_index(bm25_path)
    bm25.tokenized_corpus = [p.lower().split() for p in passages]

    logger.info("Loading FAISS index...")
    dense = DenseRetriever(config)
    faiss_path = config["data"]["faiss_index_path"]
    if not Path(faiss_path).exists():
        raise FileNotFoundError("FAISS index missing — run main.py first.")
    dense.load_faiss_index(faiss_path)

    hybrid   = HybridRetriever(bm25, dense, config)
    reranker = CrossEncoderReranker(config)

    return loader, passages, hybrid, reranker


# ── 2. Score collection ───────────────────────────────────────────────────────

def collect_scores(
    loader:   MSMarcoLoader,
    passages: List[str],
    hybrid:   HybridRetriever,
    reranker: CrossEncoderReranker,
    config:   dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each test query:
      1. Retrieve top-50 candidates via hybrid retrieval (Layer 1).
      2. Score ALL 50 with the cross-encoder — not just the reranked top-10.
      3. Label each score 1 (relevant) or 0 (irrelevant) using ground truth.

    Returns flat arrays (scores, relevance) over all queries × candidates.
    """
    max_q   = config["evaluation"]["test_queries"]
    queries = loader.get_test_queries(max_q)
    labels  = loader.build_relevance_labels()

    all_scores:    List[float] = []
    all_relevance: List[int]   = []
    queries_used = 0

    for query_id, query_text in tqdm(queries, desc="Scoring candidates"):
        if query_id not in labels:
            continue
        relevant_set = {idx for idx, rel in labels[query_id].items() if rel > 0}
        if not relevant_set:
            continue

        candidate_indices, _ = hybrid.retrieve(query_text)
        if len(candidate_indices) == 0:
            continue

        # Score every candidate (bypass reranker's top_k truncation)
        pairs  = [[query_text, passages[int(i)]] for i in candidate_indices]
        scores = reranker.model.predict(
            pairs,
            batch_size=reranker.batch_size,
            show_progress_bar=False,
        )

        for idx, score in zip(candidate_indices, scores):
            all_scores.append(float(score))
            all_relevance.append(1 if int(idx) in relevant_set else 0)

        queries_used += 1

    logger.info(
        "Collected %d pairs from %d queries  |  %d relevant  (%.1f%%)",
        len(all_scores), queries_used,
        sum(all_relevance), 100 * np.mean(all_relevance),
    )
    return np.array(all_scores), np.array(all_relevance)


# ── 3. Threshold derivation ───────────────────────────────────────────────────

def precision_recall_f1_curve(
    scores:     np.ndarray,
    relevance:  np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision, recall, F1 at every threshold value."""
    n_relevant = relevance.sum()
    precision  = np.zeros(len(thresholds))
    recall     = np.zeros(len(thresholds))
    f1         = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        above = scores >= t
        tp = (above & (relevance == 1)).sum()
        fp = (above & (relevance == 0)).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / n_relevant if n_relevant > 0 else 0.0
        precision[i] = p
        recall[i]    = r
        f1[i]        = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return precision, recall, f1


def derive_thresholds(
    thresholds: np.ndarray,
    precision:  np.ndarray,
    recall:     np.ndarray,
    f1:         np.ndarray,
    scores:     np.ndarray,
    relevance:  np.ndarray,
) -> Dict:
    """
    HIGH   = lowest threshold where precision >= 80 %
             Rationale: above this we are confident enough to generate an answer.
    MEDIUM = threshold that maximises F1
             Rationale: best overall precision-recall tradeoff.
    LOW    = highest threshold that still achieves recall >= 80 %
             Rationale: below this we are probably missing too much.

    Each threshold is rounded to 2 decimal places.
    """
    # HIGH: smallest threshold with precision >= 0.80
    hi_mask = precision >= 0.80
    high    = float(thresholds[hi_mask].min()) if hi_mask.any() else float(thresholds[np.argmax(precision)])

    # MEDIUM: F1-optimal threshold
    medium  = float(thresholds[np.argmax(f1)])

    # LOW: highest threshold still at recall >= target (80 % preferred)
    low, low_recall_achieved = None, None
    for target in (0.80, 0.70, 0.60):
        lo_mask = recall >= target
        if lo_mask.any():
            low = float(thresholds[lo_mask].max())
            low_recall_achieved = target
            break
    if low is None:
        low = float(thresholds[np.argmax(recall)])
        low_recall_achieved = float(recall.max())

    return {
        "high":                round(high,   2),
        "medium":              round(medium, 2),
        "low":                 round(low,    2),
        "low_recall_target":   low_recall_achieved,
        "n_pairs":             int(len(scores)),
        "n_relevant":          int(relevance.sum()),
        "prevalence":          round(float(relevance.mean()), 4),
        "score_min":           round(float(scores.min()),  2),
        "score_max":           round(float(scores.max()),  2),
        "score_mean":          round(float(scores.mean()), 2),
        "score_median":        round(float(np.median(scores)), 2),
    }


# ── 4. Visualization ──────────────────────────────────────────────────────────

_DARK = {
    "bg":      "#0f0f0f",
    "panel":   "#1a1a1a",
    "text":    "#e5e5e5",
    "muted":   "#888888",
    "grid":    "#2a2a2a",
    "current": "#888888",
    "rec":     "#facc15",
    "rel":     "#2dc77a",
    "irrel":   "#f87171",
    "blue":    "#4a9eff",
    "purple":  "#9b6dff",
}


def _style(ax, title: str):
    ax.set_facecolor(_DARK["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(_DARK["grid"])
    ax.tick_params(colors=_DARK["text"], labelsize=9)
    ax.xaxis.label.set_color(_DARK["text"])
    ax.yaxis.label.set_color(_DARK["text"])
    ax.set_title(title, color=_DARK["text"], fontsize=11, fontweight="bold", pad=10)
    ax.grid(color=_DARK["grid"], linewidth=0.5, alpha=0.7)


def _vlines(ax, recommended: Dict):
    """Draw current (dashed grey) and recommended (solid yellow) vertical lines."""
    for val in CURRENT.values():
        ax.axvline(val, color=_DARK["current"], lw=1.0, ls="--", alpha=0.55)
    for val in (recommended["high"], recommended["medium"], recommended["low"]):
        ax.axvline(val, color=_DARK["rec"], lw=1.5, ls="-", alpha=0.85)


def plot_all(
    scores:      np.ndarray,
    relevance:   np.ndarray,
    thresholds:  np.ndarray,
    precision:   np.ndarray,
    recall:      np.ndarray,
    f1:          np.ndarray,
    recommended: Dict,
    output_path: Path,
):
    rel_scores   = scores[relevance == 1]
    irrel_scores = scores[relevance == 0]

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(_DARK["bg"])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── Panel 1: Score distribution histogram ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1, "Score Distribution: Relevant vs Irrelevant")

    bins = np.linspace(
        min(scores.min() - 0.5, -8),
        max(scores.max() + 0.5,  8),
        70,
    )
    ax1.hist(irrel_scores, bins=bins, density=True, alpha=0.55,
             color=_DARK["irrel"], label=f"Irrelevant  n={len(irrel_scores):,}")
    ax1.hist(rel_scores,   bins=bins, density=True, alpha=0.70,
             color=_DARK["rel"],   label=f"Relevant    n={len(rel_scores):,}")

    ymax = ax1.get_ylim()[1]
    _vlines(ax1, recommended)

    # Label the threshold lines directly on the panel
    for val, label in zip(
        (CURRENT["high"], CURRENT["medium"], CURRENT["low"]),
        ("HIGH", "MED", "LOW"),
    ):
        ax1.text(val, ymax * 0.97, f"cur\n{label}", ha="center", va="top",
                 color=_DARK["current"], fontsize=7)

    for val, label in zip(
        (recommended["high"], recommended["medium"], recommended["low"]),
        ("HIGH", "MED", "LOW"),
    ):
        ax1.text(val, ymax * 0.78, f"rec\n{label}", ha="center", va="top",
                 color=_DARK["rec"], fontsize=7, fontweight="bold")

    ax1.set_xlabel("Cross-Encoder Score (logit)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=9, facecolor=_DARK["panel"], labelcolor=_DARK["text"],
               framealpha=0.9, edgecolor=_DARK["grid"])

    # ── Panel 2: Precision / Recall / F1 curves ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2, "Precision / Recall / F1 vs Threshold")

    ax2.plot(thresholds, precision, color=_DARK["blue"],   lw=2, label="Precision")
    ax2.plot(thresholds, recall,    color=_DARK["rel"],    lw=2, label="Recall")
    ax2.plot(thresholds, f1,        color=_DARK["rec"],    lw=2, label="F1")

    _vlines(ax2, recommended)

    # Annotate the F1-optimal point
    best_f1_idx = np.argmax(f1)
    ax2.scatter(thresholds[best_f1_idx], f1[best_f1_idx], color=_DARK["rec"], zorder=6, s=60)
    ax2.annotate(
        f"F1={f1[best_f1_idx]:.2f}\n@ {thresholds[best_f1_idx]:.2f}",
        (thresholds[best_f1_idx], f1[best_f1_idx]),
        textcoords="offset points", xytext=(8, -20),
        fontsize=8, color=_DARK["rec"],
        arrowprops=dict(arrowstyle="->", color=_DARK["rec"], lw=0.8),
    )

    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9, facecolor=_DARK["panel"], labelcolor=_DARK["text"],
               framealpha=0.9, edgecolor=_DARK["grid"])

    # ── Panel 3: Precision-Recall curve ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3, "Precision–Recall Curve")

    # Sort by recall for a smooth curve
    order = np.argsort(recall)
    ax3.plot(recall[order], precision[order], color=_DARK["purple"], lw=2)
    ax3.fill_between(recall[order], precision[order], alpha=0.12, color=_DARK["purple"])

    # Mark operating points for each recommended threshold
    for val, label in zip(
        (recommended["high"], recommended["medium"], recommended["low"]),
        ("HIGH", "MEDIUM", "LOW"),
    ):
        t_idx = int(np.argmin(np.abs(thresholds - val)))
        ax3.scatter(recall[t_idx], precision[t_idx],
                    color=_DARK["rec"], zorder=5, s=70)
        ax3.annotate(
            label,
            (recall[t_idx], precision[t_idx]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color=_DARK["rec"],
        )

    # Baseline: random classifier
    prevalence = recommended["prevalence"]
    ax3.axhline(prevalence, color=_DARK["current"], lw=1, ls=":",
                label=f"Random ({prevalence:.1%})")

    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_xlim(-0.02, 1.05)
    ax3.set_ylim(-0.02, 1.05)
    ax3.legend(fontsize=9, facecolor=_DARK["panel"], labelcolor=_DARK["text"],
               framealpha=0.9, edgecolor=_DARK["grid"])

    # ── Panel 4: Summary table ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(_DARK["panel"])
    for sp in ax4.spines.values():
        sp.set_edgecolor(_DARK["grid"])
    ax4.axis("off")
    ax4.set_title("Recommended Thresholds", color=_DARK["text"],
                  fontsize=11, fontweight="bold", pad=10)

    headers = ["Level", "Current", "Recommended", "Δ"]
    rows = []
    for level in ("high", "medium", "low"):
        cur = CURRENT[level]
        rec = recommended[level]
        delta = rec - cur
        rows.append([level.upper(), f"{cur:.1f}", f"{rec:.2f}", f"{delta:+.2f}"])

    rows += [
        ["─" * 12, "─" * 8, "─" * 12, "─" * 6],
        ["Pairs",      "",  f"{recommended['n_pairs']:,}",    ""],
        ["Relevant",   "",  f"{recommended['n_relevant']:,}", ""],
        ["Prevalence", "",  f"{recommended['prevalence']:.1%}", ""],
        ["Score range","",  f"{recommended['score_min']:.1f} → {recommended['score_max']:.1f}", ""],
        ["Median",     "",  f"{recommended['score_median']:.2f}", ""],
    ]

    col_x = [0.02, 0.28, 0.54, 0.82]
    row_h = 0.083
    y0    = 0.92

    # Header row
    for j, h in enumerate(headers):
        ax4.text(col_x[j], y0, h, transform=ax4.transAxes,
                 color=_DARK["muted"], fontsize=9, fontweight="bold", va="top")

    for i, row in enumerate(rows):
        y = y0 - (i + 1) * row_h
        for j, cell in enumerate(row):
            color = _DARK["text"]
            if j == 0 and i < 3:
                color = _DARK["rec"]
            elif j == 3 and cell.startswith("+"):
                color = _DARK["rel"]
            elif j == 3 and cell.startswith("-"):
                color = _DARK["irrel"]
            elif i >= 4:
                color = _DARK["muted"]
            ax4.text(col_x[j], y, cell, transform=ax4.transAxes,
                     color=color, fontsize=9.5, va="top")

    # ── Title and legend note ─────────────────────────────────────────────────
    fig.suptitle(
        "QueryLens · Layer 2 Confidence Threshold Calibration",
        color=_DARK["text"], fontsize=14, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.005,
        "── ── Current thresholds (decision.py)     ─── Recommended thresholds (this run)",
        ha="center", color=_DARK["muted"], fontsize=9,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved plot → %s", output_path)


# ── 5. Console report ─────────────────────────────────────────────────────────

def print_report(recommended: Dict):
    W = 58
    print()
    print("═" * W)
    print("  LAYER 2 THRESHOLD CALIBRATION")
    print("═" * W)
    print(f"  {'Score-label pairs':<22}: {recommended['n_pairs']:>8,}")
    print(f"  {'Relevant passages':<22}: {recommended['n_relevant']:>8,}  ({recommended['prevalence']:.1%})")
    print(f"  {'Score range':<22}: {recommended['score_min']:>8.2f}  →  {recommended['score_max']:.2f}")
    print(f"  {'Median score':<22}: {recommended['score_median']:>8.2f}")
    print()
    print(f"  {'Level':<10}  {'Current':>9}  {'Recommended':>13}  {'Change':>8}")
    print(f"  {'─' * (W - 4)}")
    for level in ("high", "medium", "low"):
        cur   = CURRENT[level]
        rec   = recommended[level]
        delta = rec - cur
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
        print(f"  {level.upper():<10}  {cur:>9.1f}  {rec:>13.2f}  {arrow} {abs(delta):.2f}")
    print("═" * W)
    print()
    print("  Apply in decision.py:")
    print(f"    HIGH_THRESHOLD   = {recommended['high']}")
    print(f"    MEDIUM_THRESHOLD = {recommended['medium']}")
    print(f"    LOW_THRESHOLD    = {recommended['low']}")
    print()


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    config = load_config()

    # GPU-adaptive: override device + batch size based on what's available
    device, batch_size = detect_device()
    config = apply_device(config, device, batch_size)
    logger.info("Using device=%s  batch_size=%d", device, batch_size)

    loader, passages, hybrid, reranker = build_pipeline(config)

    scores, relevance = collect_scores(loader, passages, hybrid, reranker, config)

    if len(scores) == 0:
        logger.error("No scores collected — check that data files exist.")
        return
    if relevance.sum() == 0:
        logger.error("Zero relevant passages in candidates — check relevance labels.")
        return

    # Fine-grained threshold sweep over observed score range
    t_lo, t_hi = np.percentile(scores, 1), np.percentile(scores, 99)
    thresholds  = np.linspace(t_lo, t_hi, 600)

    precision, recall, f1 = precision_recall_f1_curve(scores, relevance, thresholds)
    recommended           = derive_thresholds(thresholds, precision, recall, f1, scores, relevance)

    print_report(recommended)

    # Save JSON (including run metadata so we know what produced these numbers)
    json_path = RESULTS_DIR / "threshold_calibration.json"
    with open(json_path, "w") as f:
        json.dump({
            "recommended": recommended,
            "current":     CURRENT,
            "run":         {"device": device, "batch_size": batch_size},
        }, f, indent=2)
    logger.info("Saved thresholds → %s", json_path)

    # Save plot
    plot_all(
        scores, relevance, thresholds,
        precision, recall, f1,
        recommended,
        output_path=RESULTS_DIR / "threshold_calibration.png",
    )

    print(f"  Visualization → results/threshold_calibration.png")
    print(f"  JSON output   → results/threshold_calibration.json")
    print()


if __name__ == "__main__":
    main()
