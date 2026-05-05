"""
main.py — Review Intelligence System

Orchestrates the complete pipeline:
  1. Read reviews from /data/reviews.txt
  2. Extract highlights + pain points from each review concurrently via Ollama
  3. Embed all highlights and all pain points
  4. Deduplicate each list using semantic similarity
  5. Write a detailed analysis report to /data/output/analysis.txt
  6. Return ranked, deduplicated results
"""

import asyncio
import datetime
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.deduplicator import deduplicate_with_analysis
from app.embedder import ReviewEmbedder
from app.extractor import extract_from_review

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REVIEWS_FILE_PATH = "/data/reviews.txt"
OUTPUT_DIR = "/data/output"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_app_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: loading ReviewEmbedder …")
    _app_state["embedder"] = ReviewEmbedder()
    logger.info("Startup: ReviewEmbedder ready.")
    yield
    logger.info("Shutdown: cleaning up.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Review Intelligence System", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Analysis report writer
# ---------------------------------------------------------------------------
def _write_analysis_report(
    *,
    reviews: list[str],
    extraction_results: list[dict],
    all_highlights: list[str],
    all_pain_points: list[str],
    highlights_analysis: dict,
    pain_points_analysis: dict,
    final_highlights: list[dict],
    final_pain_points: list[dict],
) -> None:
    """
    Write a human-readable analysis report to OUTPUT_DIR/analysis.txt.

    The report includes:
    - Per-review raw LLM extractions
    - Pre-deduplication item lists
    - Full pairwise cosine similarity matrices
    - Pairs that exceeded the threshold (and were merged)
    - Per-cluster member breakdown with mean-similarity scores
    - Final deduplicated results
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "analysis.txt")
    logger.info("Writing analysis report to %s …", report_path)

    lines: list[str] = []
    SEP = "─" * 68

    def section(title: str) -> None:
        lines.append("")
        lines.append(SEP)
        lines.append(f"  {title}")
        lines.append(SEP)

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    threshold = highlights_analysis["threshold"]

    lines.append("=" * 68)
    lines.append("  REVIEW INTELLIGENCE — ANALYSIS REPORT")
    lines.append("=" * 68)
    lines.append(f"  Generated  : {now}")
    lines.append(f"  Reviews    : {len(reviews)}")
    lines.append(f"  Threshold  : {threshold}")

    # ------------------------------------------------------------------
    # 1. Per-review LLM extractions
    # ------------------------------------------------------------------
    section("PER-REVIEW EXTRACTIONS  (raw LLM output)")
    for i, (review, extraction) in enumerate(zip(reviews, extraction_results), 1):
        excerpt = review[:80] + ("…" if len(review) > 80 else "")
        lines.append(f'\n  Review {i}  │ "{excerpt}"')
        hl = extraction.get("highlights", [])
        pp = extraction.get("pain_points", [])
        hl_str = ", ".join(f'"{x}"' for x in hl) if hl else "(none)"
        pp_str = ", ".join(f'"{x}"' for x in pp) if pp else "(none)"
        lines.append(f"    Highlights  ({len(hl)}): {hl_str}")
        lines.append(f"    Pain points ({len(pp)}): {pp_str}")

    # ------------------------------------------------------------------
    # 2. Per-category detailed analysis
    # ------------------------------------------------------------------
    for label, items, analysis, final in [
        ("HIGHLIGHTS", all_highlights, highlights_analysis, final_highlights),
        ("PAIN POINTS", all_pain_points, pain_points_analysis, final_pain_points),
    ]:
        # Pre-dedup list
        section(f"{label} — PRE-DEDUPLICATION  ({len(items)} items)")
        if not items:
            lines.append("  (none)")
        else:
            for i, item in enumerate(items):
                lines.append(f"  [{i:2d}] {item}")

        if len(items) <= 1:
            continue

        # Similarity matrix
        section(f"{label} — PAIRWISE COSINE SIMILARITY MATRIX")
        mat = analysis["similarity_matrix"]
        n = len(items)
        col_w = 7

        header = " " * 8 + "".join(f"[{i}]".rjust(col_w) for i in range(n))
        lines.append(header)
        for i in range(n):
            row = f"  [{i:2d}]  " + "".join(f"{mat[i][j]:.4f}".rjust(col_w) for j in range(n))
            lines.append(row)

        # Pairs above threshold
        above = [
            (i, j, mat[i][j])
            for i in range(n)
            for j in range(i + 1, n)
            if mat[i][j] > threshold
        ]
        lines.append("")
        lines.append(f"  Pairs above threshold ({threshold}) → merged into same cluster:")
        if above:
            for i, j, sim in sorted(above, key=lambda x: -x[2]):
                lines.append(
                    f'    [{i:2d}]↔[{j:2d}]  {sim:.4f}  "{items[i]}" ↔ "{items[j]}"'
                )
        else:
            lines.append("    (none — all items treated as distinct)")

        # Clusters
        section(f"{label} — CLUSTERS  (threshold={threshold})")
        sorted_clusters = sorted(analysis["clusters"], key=lambda c: -c["count"])
        for ci, cluster in enumerate(sorted_clusters, 1):
            lines.append(
                f'\n  Cluster {ci}  ×{cluster["count"]}  '
                f'representative: "{cluster["representative"]}"'
            )
            for m in cluster["members"]:
                is_rep = m["text"] == cluster["representative"]
                mean_str = (
                    f"  mean_sim={m['mean_sim_to_others']:.4f}"
                    if m["mean_sim_to_others"] is not None
                    else ""
                )
                rep_marker = "  ← chosen representative" if is_rep else ""
                lines.append(
                    f'    [{m["index"]:2d}] "{m["text"]}"{mean_str}{rep_marker}'
                )
            if cluster["intra_similarities"]:
                lines.append("    Intra-cluster pairwise similarities:")
                for pair in cluster["intra_similarities"]:
                    lines.append(
                        f'      [{pair["index_a"]:2d}]↔[{pair["index_b"]:2d}]  '
                        f'{pair["similarity"]:.4f}  '
                        f'"{pair["item_a"]}" ↔ "{pair["item_b"]}"'
                    )

    # ------------------------------------------------------------------
    # 3. Final results
    # ------------------------------------------------------------------
    section("FINAL RESULTS")
    lines.append("")
    lines.append("  HIGHLIGHTS:")
    for entry in final_highlights:
        lines.append(f'    ×{entry["count"]}  {entry["item"]}')
    if not final_highlights:
        lines.append("    (none)")

    lines.append("")
    lines.append("  PAIN POINTS:")
    for entry in final_pain_points:
        lines.append(f'    ×{entry["count"]}  {entry["item"]}')
    if not final_pain_points:
        lines.append("    (none)")

    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Analysis report written to %s", report_path)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    if "embedder" not in _app_state:
        raise HTTPException(status_code=503, detail="Embedder not yet initialised.")
    return {"status": "ok"}


@app.post("/process_file")
async def process_file() -> dict:
    """
    Read hotel reviews from /data/reviews.txt and return deduplicated,
    ranked highlights and pain points.

    Pipeline:
      read → extract (concurrent) → filter empty → embed → deduplicate
      → write analysis report → respond
    """
    embedder: ReviewEmbedder = _app_state["embedder"]

    # ------------------------------------------------------------------
    # Stage 1: read reviews
    # ------------------------------------------------------------------
    logger.info("Reading reviews from %s", REVIEWS_FILE_PATH)
    try:
        with open(REVIEWS_FILE_PATH, encoding="utf-8") as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        logger.error("Reviews file not found at %s", REVIEWS_FILE_PATH)
        raise HTTPException(
            status_code=500,
            detail=f"Reviews file not found: {REVIEWS_FILE_PATH}",
        )

    reviews = [line.strip() for line in raw_lines if line.strip()]
    logger.info("Found %d non-empty reviews.", len(reviews))

    if not reviews:
        logger.warning("Reviews file is empty — returning empty result.")
        return {"highlights": [], "pain_points": []}

    # ------------------------------------------------------------------
    # Stage 2: extract highlights and pain points concurrently
    # ------------------------------------------------------------------
    logger.info("Extracting via Ollama (%s) …", OLLAMA_HOST)
    extraction_results: list[dict] = await asyncio.gather(
        *[extract_from_review(review=r, ollama_host=OLLAMA_HOST) for r in reviews]
    )
    logger.info("Extraction complete for %d reviews.", len(extraction_results))

    # ------------------------------------------------------------------
    # Stage 3: flatten and filter empty strings
    # ------------------------------------------------------------------
    all_highlights: list[str] = []
    all_pain_points: list[str] = []

    for result in extraction_results:
        all_highlights.extend(
            item for item in result.get("highlights", []) if item.strip()
        )
        all_pain_points.extend(
            item for item in result.get("pain_points", []) if item.strip()
        )

    logger.info(
        "Flattened: %d highlights, %d pain points.",
        len(all_highlights),
        len(all_pain_points),
    )

    if not all_highlights and not all_pain_points:
        logger.warning("All extractions returned empty — returning empty result.")
        return {"highlights": [], "pain_points": []}

    # ------------------------------------------------------------------
    # Stage 4: embed
    # ------------------------------------------------------------------
    logger.info("Embedding highlights …")
    highlight_embeddings = embedder.embed(all_highlights)

    logger.info("Embedding pain points …")
    pain_point_embeddings = embedder.embed(all_pain_points)

    # ------------------------------------------------------------------
    # Stage 5: cross-validate — remove pain points that are semantically
    # close to highlights (catches items the model misclassified).
    # A pain point with cosine similarity ≥ threshold to ANY highlight is
    # almost certainly a positive item placed in the wrong category.
    # ------------------------------------------------------------------
    if all_highlights and all_pain_points:
        cross_sim = pain_point_embeddings @ highlight_embeddings.T  # (n_pp, n_hl)
        keep = cross_sim.max(axis=1) < 0.70
        removed = [p for p, k in zip(all_pain_points, keep) if not k]
        if removed:
            logger.info("Cross-validation removed misclassified pain points: %s", removed)
        all_pain_points = [p for p, k in zip(all_pain_points, keep) if k]
        pain_point_embeddings = pain_point_embeddings[keep]

    # ------------------------------------------------------------------
    # Stage 6: deduplicate (with analysis)
    # ------------------------------------------------------------------
    logger.info("Deduplicating highlights …")
    final_highlights, highlights_analysis = deduplicate_with_analysis(
        items=all_highlights,
        embeddings=highlight_embeddings,
    )

    logger.info("Deduplicating pain points …")
    final_pain_points, pain_points_analysis = deduplicate_with_analysis(
        items=all_pain_points,
        embeddings=pain_point_embeddings,
    )

    logger.info(
        "Done. %d unique highlights, %d unique pain points.",
        len(final_highlights),
        len(final_pain_points),
    )

    # ------------------------------------------------------------------
    # Stage 7: write analysis report
    # ------------------------------------------------------------------
    try:
        _write_analysis_report(
            reviews=reviews,
            extraction_results=extraction_results,
            all_highlights=all_highlights,
            all_pain_points=all_pain_points,
            highlights_analysis=highlights_analysis,
            pain_points_analysis=pain_points_analysis,
            final_highlights=final_highlights,
            final_pain_points=final_pain_points,
        )
    except Exception as exc:
        logger.error("Failed to write analysis report: %s", exc, exc_info=True)

    return {
        "highlights": final_highlights,
        "pain_points": final_pain_points,
    }
