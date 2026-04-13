"""
Main experiment runner for hybrid search and reranking system.

Runs two evaluation passes automatically:
  1. General evaluation   — random sample of all MS MARCO queries
  2. Climate subsection   — climate/energy-tagged queries only

Run order:
  1. python dataset_builder.py   # build dataset once
  2. python main.py              # run experiments
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

from data_loader import (
    MSMarcoLoader,
    load_embeddings, save_embeddings,
    load_bm25_index, save_bm25_index,
)
from retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from reranker import CrossEncoderReranker, RetrievalPipeline
from evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ExperimentRunner:

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        logger.info("Loaded config from %s", config_path)
        Path(self.config["experiment"]["results_dir"]).mkdir(
            parents=True, exist_ok=True
        )

        self.passages:    List[str]  = []
        self.passage_ids: List[str]  = []
        self.metadata:    List[Dict] = []
        self._loader:     MSMarcoLoader = None

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------

    def setup_data(self) -> MSMarcoLoader:
        passages_path = self.config["data"]["ms_marco_passages_path"]
        queries_path  = self.config["data"]["ms_marco_queries_path"]

        if not Path(passages_path).exists() or not Path(queries_path).exists():
            raise FileNotFoundError(
                "MS MARCO dataset not found.\n"
                "Run: python dataset_builder.py"
            )

        loader = MSMarcoLoader(self.config)
        self.passages, self.passage_ids, self.metadata, _ = loader.load()
        self._loader = loader
        return loader

    # ------------------------------------------------------------------
    # Retriever construction
    # ------------------------------------------------------------------

    def build_retrievers(self) -> Dict:
        logger.info("Building retrieval systems...")

        bm25       = BM25Retriever(self.config)
        index_path = self.config["data"]["index_path"]
        if Path(index_path).exists():
            bm25.index = load_bm25_index(index_path)
            bm25.tokenized_corpus = [p.lower().split() for p in self.passages]
        else:
            bm25.build_index(self.passages)
            save_bm25_index(bm25.index, index_path)

        dense           = DenseRetriever(self.config)
        faiss_path      = self.config["data"]["faiss_index_path"]
        emb_path        = self.config["data"]["embeddings_path"]

        if Path(faiss_path).exists() and Path(emb_path).exists():
            logger.info("Loading cached FAISS index and embeddings")
            dense.load_faiss_index(faiss_path)
        else:
            embeddings = dense.encode_documents(self.passages)
            save_embeddings(embeddings, emb_path)
            dense.save_faiss_index(faiss_path)

        hybrid = HybridRetriever(bm25, dense, self.config)
        logger.info("Retrieval systems ready")
        return {"bm25": bm25, "dense": dense, "hybrid": hybrid}

    def _make_pipelines(self, retrievers: Dict, reranker) -> Dict:
        pipelines = {
            "BM25 Only":         RetrievalPipeline(
                retrievers["bm25"],   reranker, self.passages
            ),
            "Dense Only":        RetrievalPipeline(
                retrievers["dense"],  reranker, self.passages
            ),
            "Hybrid (RRF)":      RetrievalPipeline(
                retrievers["hybrid"], reranker, self.passages
            ),
            "Hybrid + Reranker": RetrievalPipeline(
                retrievers["hybrid"], reranker, self.passages
            ),
        }
        for method in ["BM25 Only", "Dense Only", "Hybrid (RRF)"]:
            orig = pipelines[method].search
            pipelines[method].search = (
                lambda q, _orig=orig: _orig(q, rerank=False)
            )
        return pipelines

    # ------------------------------------------------------------------
    # Main experiment runner
    # ------------------------------------------------------------------

    def run_experiments(self):
        logger.info("=" * 80)
        logger.info("Hybrid Search Ablation Study — MS MARCO Sample")
        logger.info("=" * 80)

        loader = self.setup_data()
        retrievers = self.build_retrievers()
        reranker   = CrossEncoderReranker(self.config)
        pipelines  = self._make_pipelines(retrievers, reranker)
        evaluator  = Evaluator(self.config)
        max_q      = self.config["evaluation"]["test_queries"]

        # ------------------------------------------------------------------
        # Pass 1 — General evaluation (all queries)
        # ------------------------------------------------------------------
        logger.info("--- Pass 1: General evaluation ---")
        general_queries = loader.get_test_queries(max_q, climate_only=False)
        general_labels  = loader.build_relevance_labels(climate_only=False)

        general_results = evaluator.run_ablation_study(
            pipelines, general_queries, general_labels
        )
        logger.info("General results: %s", general_results)

        # ------------------------------------------------------------------
        # Pass 2 — Climate domain subsection
        # ------------------------------------------------------------------
        n_climate = loader.climate_query_count()
        climate_results = None

        if n_climate >= 10:
            logger.info(
                "--- Pass 2: Climate domain analysis (%d queries) ---",
                n_climate,
            )
            climate_queries = loader.get_test_queries(
                max_q, climate_only=True
            )
            climate_labels  = loader.build_relevance_labels(
                climate_only=True
            )
            climate_results = evaluator.run_ablation_study(
                pipelines, climate_queries, climate_labels
            )
            logger.info("Climate results: %s", climate_results)
        else:
            logger.warning(
                "Only %d climate queries found — skipping domain subsection "
                "(need at least 10). Increase --max-queries in dataset_builder.py.",
                n_climate,
            )

        # ------------------------------------------------------------------
        # Save & visualise
        # ------------------------------------------------------------------
        self.save_results(general_results, climate_results)
        self.visualize_results(general_results, climate_results)
        self.print_results_table(general_results, "GENERAL")
        if climate_results:
            self.print_results_table(climate_results, "CLIMATE DOMAIN")

        logger.info("=" * 80)
        logger.info("Experiments complete!")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def save_results(self, general: Dict, climate: Optional[Dict] = None):
        out = {"general": general}
        if climate:
            out["climate_domain"] = climate
        path = Path(self.config["experiment"]["results_dir"]) / "results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Results saved to %s", path)

    def visualize_results(
        self,
        general: Dict,
        climate: Optional[Dict] = None,
    ):
        logger.info("Creating visualizations...")
        n_plots = 2 if climate else 1
        fig = plt.figure(figsize=(14 * n_plots, 10))
        gs  = gridspec.GridSpec(2, 2 * n_plots)

        self._plot_pair(fig, gs, general, "General (all queries)", col=0)
        if climate:
            self._plot_pair(
                fig, gs, climate, "Climate domain subsection", col=2
            )

        plt.tight_layout()
        fig_path = (
            Path(self.config["experiment"]["results_dir"])
            / "results_visualization.png"
        )
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        logger.info("Visualization saved to %s", fig_path)
        plt.show()

    def _plot_pair(self, fig, gs, results: Dict, title: str, col: int):
        df = pd.DataFrame(results).T

        ax1 = fig.add_subplot(gs[0, col:col+2])
        df.plot(kind="bar", ax=ax1, width=0.8)
        ax1.set_title(title, fontsize=13, fontweight="bold")
        ax1.set_xlabel("Method", fontsize=11)
        ax1.set_ylabel("Score", fontsize=11)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2 = fig.add_subplot(gs[1, col:col+2])
        sns.heatmap(
            df, annot=True, fmt=".3f", cmap="YlGnBu",
            ax=ax2, cbar_kws={"label": "Score"},
        )
        ax2.set_title(f"{title} — heatmap", fontsize=12)
        ax2.set_xlabel("Metric", fontsize=11)
        ax2.set_ylabel("Method", fontsize=11)

    def print_results_table(self, results: Dict, label: str = ""):
        print("\n" + "=" * 80)
        print(f"RESULTS — {label}")
        print("=" * 80)
        metrics = list(next(iter(results.values())).keys())
        print(f"{'Method':<25}", end="")
        for m in metrics:
            print(f"{m:>12}", end="")
        print()
        print("-" * 80)
        for method, scores in results.items():
            print(f"{method:<25}", end="")
            for m in metrics:
                print(f"{scores[m]:>12.4f}", end="")
            print()
        print("=" * 80 + "\n")


def main():
    runner = ExperimentRunner()
    runner.run_experiments()


if __name__ == "__main__":
    main()
