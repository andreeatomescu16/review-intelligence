"""
main.py — Stage 3: full integration.

Orchestrates the complete pipeline:
  1. Read reviews from /data/reviews.txt
  2. Extract highlights + pain points from each review concurrently via Ollama
  3. Embed all highlights and all pain points
  4. Deduplicate each list using semantic similarity
  5. Return ranked, deduplicated results
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.deduplicator import deduplicate
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
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
# The embedder is initialised once at startup and shared across all requests.
# Storing it in a plain dict avoids global mutable module-level state while
# remaining accessible inside request handlers via the app.state mechanism.
_app_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and once at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Loads the embedding model into memory before the server starts accepting
    requests, so the first call to /process_file is not penalised by model
    loading latency.
    """
    logger.info("Startup: loading ReviewEmbedder …")
    _app_state["embedder"] = ReviewEmbedder()
    logger.info("Startup: ReviewEmbedder ready.")
    yield
    # Nothing to tear down for the embedder, but the yield is required.
    logger.info("Shutdown: cleaning up.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Review Intelligence System", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """Liveness probe — returns ok when the process is up and the embedder is loaded."""
    if "embedder" not in _app_state:
        raise HTTPException(status_code=503, detail="Embedder not yet initialised.")
    return {"status": "ok"}


@app.post("/process_file")
async def process_file() -> dict:
    """
    Read hotel reviews from /data/reviews.txt and return deduplicated,
    ranked highlights and pain points.

    Pipeline:
      read → extract (concurrent) → embed → deduplicate → respond

    Returns:
        JSON object with keys:
          - "highlights": list of {"item": str, "count": int} sorted by count desc
          - "pain_points": list of {"item": str, "count": int} sorted by count desc
    """
    embedder: ReviewEmbedder = _app_state["embedder"]

    # ------------------------------------------------------------------
    # Stage 1: read reviews
    # ------------------------------------------------------------------
    logger.info("Reading reviews from %s", REVIEWS_FILE_PATH)
    try:
        with open(REVIEWS_FILE_PATH, encoding="utf-8") as reviews_file:
            raw_lines = reviews_file.readlines()
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
    logger.info("Extracting highlights and pain points via Ollama (%s) …", OLLAMA_HOST)
    extraction_tasks = [
        extract_from_review(review=review, ollama_host=OLLAMA_HOST)
        for review in reviews
    ]
    # asyncio.gather runs all coroutines concurrently, cutting total latency
    # to roughly max(individual_latencies) instead of sum(individual_latencies).
    extraction_results: list[dict] = await asyncio.gather(*extraction_tasks)
    logger.info("Extraction complete for %d reviews.", len(extraction_results))

    # ------------------------------------------------------------------
    # Stage 3: flatten extracted items
    # ------------------------------------------------------------------
    all_highlights: list[str] = []
    all_pain_points: list[str] = []

    for result in extraction_results:
        all_highlights.extend(result.get("highlights", []))
        all_pain_points.extend(result.get("pain_points", []))

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
    # Stage 5: deduplicate
    # ------------------------------------------------------------------
    logger.info("Deduplicating highlights …")
    deduplicated_highlights = deduplicate(
        items=all_highlights,
        embeddings=highlight_embeddings,
    )

    logger.info("Deduplicating pain points …")
    deduplicated_pain_points = deduplicate(
        items=all_pain_points,
        embeddings=pain_point_embeddings,
    )

    logger.info(
        "Done. %d unique highlights, %d unique pain points.",
        len(deduplicated_highlights),
        len(deduplicated_pain_points),
    )

    return {
        "highlights": deduplicated_highlights,
        "pain_points": deduplicated_pain_points,
    }
