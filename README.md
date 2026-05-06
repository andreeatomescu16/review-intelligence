# Review Intelligence System

## Overview

The Review Intelligence System is a REST API that processes a batch of hotel
reviews to surface the most commonly mentioned positives and negatives. For
each review it calls a locally-running LLM (llama3.2:1b via Ollama) to extract
structured highlights and pain points, then uses dense semantic embeddings
(multi-qa-MiniLM-L6-cos-v1 via sentence-transformers) to cluster near-duplicate
phrases across reviews. The result is a ranked list of unique highlights and
pain points, each annotated with how many reviews mentioned that theme.

A full analysis report (similarity matrices, cluster breakdowns, per-review
extractions) is written to `/data/output/analysis.txt` on every run.

---

## Architecture

### Services

| Service | Image | Role |
|---|---|---|
| `ollama` | `ollama/ollama:latest` | LLM inference runtime; exposes REST API on port 11434 (internal only) |
| `ollama-pull` | `ollama/ollama:latest` | Init container; pulls `llama3.2:1b` once then exits |
| `app` | Custom (python:3.11-slim) | FastAPI backend; orchestrates the full pipeline on port 8000 |
| `frontend` | `nginx:alpine` | Serves the UI on port 3000; proxies `/api/` to the backend |

Startup order is enforced via Docker Compose dependency conditions:
`ollama` must pass its healthcheck → `ollama-pull` must exit 0 → `app` starts.
The `frontend` service depends on `app`. This means the API is guaranteed to
have a fully downloaded model before it accepts any traffic.

### Python modules

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI application; orchestrates the 7-stage pipeline; owns the embedder singleton via FastAPI lifespan |
| `app/extractor.py` | Async HTTP calls to Ollama; structured JSON extraction with `format: "json"`; retry with exponential backoff |
| `app/embedder.py` | Loads SentenceTransformer once at startup; encodes lists of strings into L2-normalised embedding vectors |
| `app/deduplicator.py` | Computes pairwise cosine similarity; Union-Find clustering; centroid-based representative selection |

### Processing pipeline

The `/process_file` endpoint executes seven stages:

1. **Ingest** — Read all reviews from `/data/reviews.txt`; strip whitespace and filter empty lines.
2. **Extraction** — For each review, call `llama3.2:1b` via Ollama to extract highlights and pain points as structured JSON. All requests are dispatched concurrently via `asyncio.gather()`.
3. **Flatten & filter** — Aggregate all extracted highlights into one list and all pain points into another; discard empty strings produced by the LLM.
4. **Embedding** — Embed all highlights and all pain points separately using the pre-loaded SentenceTransformer model.
5. **Cross-validation** — Remove pain points whose cosine similarity to any highlight exceeds 0.70. This catches items the LLM misclassified as negative when they are semantically positive (e.g. "very comfortable beds" appearing in the pain-points list).
6. **Deduplication** — Pairwise cosine similarity is computed for each list independently; items above the 0.8 threshold are merged into clusters via greedy Union-Find. Each cluster elects a centroid representative. A detailed analysis report is written to `/data/output/analysis.txt`.
7. **Response** — Deduplicated highlights and pain points are returned as JSON, each entry carrying a `count` field.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (Engine 20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2 plugin or standalone)

No other local dependencies are required — the LLM and embedding model are
both fetched/baked during the build and first run.

---

## Running the project

```bash
git clone <repo-url>
cd review-intelligence
docker compose up --build
```

The first run downloads `llama3.2:1b` (~1.3 GB) and bakes the embedding model
into the app image. This takes 2–5 minutes depending on your connection.

You will know the system is ready when you see:

```
ollama-pull-1  | Model ready.
ollama-pull-1 exited with code 0
app-1          | Startup: ReviewEmbedder ready.
app-1          | Uvicorn running on http://0.0.0.0:8000
```

**Option A — UI (recommended):**
Open [http://localhost:3000](http://localhost:3000) and click **Analyze Reviews**.

**Option B — curl:**
```bash
curl -X POST http://localhost:8000/process_file
```

Subsequent starts skip the model download (cached in the `ollama_models` Docker volume).

---

## Expected output

```json
{
  "highlights": [
    {"item": "Great breakfast and very clean rooms.", "count": 3},
    {"item": "Fantastic location and beautiful views from the room.", "count": 2},
    {"item": "Staff were friendly", "count": 2},
    {"item": "The beds were very comfortable", "count": 1}
  ],
  "pain_points": [
    {"item": "check-in took too long", "count": 3},
    {"item": "the Wi-Fi was very slow throughout our stay", "count": 1}
  ]
}
```

> **Note:** exact phrasing of representative items depends on which cluster
> member the model selected as centroid; counts reflect the number of reviews
> that mentioned each theme. LLM output is non-deterministic — results may
> vary slightly between runs.

A detailed breakdown is also written to `/data/output/analysis.txt` inside the
app container after every run.

---

## Design decisions

### Embedding model: `multi-qa-MiniLM-L6-cos-v1` over `all-MiniLM-L6-v2`

`multi-qa-MiniLM-L6-cos-v1` was fine-tuned explicitly on semantic
similarity and question-answer retrieval datasets, making its embedding space
better calibrated for detecting "same concept, different wording" across
paraphrases. `all-MiniLM-L6-v2` is trained on a broader but less focused
corpus, which produces good general-purpose embeddings but slightly weaker
discrimination on short, noun-heavy phrases like hotel review bullet points.
Both models share the same architecture (6 layers, 384-dimensional output,
~22M parameters, ~80 MB on disk), so there is no cost to choosing the more
task-appropriate model.

### Structured JSON output from Ollama

Ollama's `"format": "json"` parameter constrains the model's sampling to
produce syntactically valid JSON. Without this, small models (1B parameters)
frequently wrap their output in markdown code fences, add explanatory prose,
or produce subtly malformed JSON. Enforcing structure at the inference layer
eliminates an entire class of post-processing fragility and makes the retry
logic in `extractor.py` a genuine safety net rather than the primary parsing
strategy.

### Cross-validation before deduplication

Before deduplication, pain points are cross-validated against the highlights
list: any pain point whose cosine similarity to any highlight exceeds 0.70 is
removed. Small LLMs occasionally misclassify a positive attribute as a
complaint. This step catches those errors before they corrupt the pain-points
cluster. The 0.70 threshold was chosen to be comfortably below the
deduplication threshold (0.80), ensuring no legitimate pain point is removed
due to superficial lexical overlap with a highlight.

### Centroid-based representative selection

When multiple review phrases are merged into one cluster, the representative
phrase is the one with the highest mean cosine similarity to all other members
of the cluster — the phrase closest to the geometric centroid in embedding
space. This is preferable to selecting the first item encountered because the
first item is order-dependent and may be an outlier phrase that happened to
exceed the similarity threshold with just one neighbour. The centroid item is
the most canonical phrasing of the shared concept.

### Greedy Union-Find clustering

Union-Find with pairwise threshold comparison is deterministic (given a fixed
similarity matrix), requires no hyperparameters beyond the threshold, and
produces stable results as the item list grows. The O(n²) pairwise scan is
acceptable because n (the number of extracted items per request) is small in
practice — typically 5–30 items for a file of hotel reviews.

---

## Threshold sensitivity

Two thresholds are in use:

| Threshold | Purpose | Value |
|---|---|---|
| Cross-validation | Remove pain points semantically close to highlights | 0.70 |
| Deduplication | Merge near-duplicate items within the same category | 0.80 |

### Deduplication threshold behaviour

| Value | Observed behaviour | Risk |
|---|---|---|
| 0.75 | All three check-in pain point variants collapse into one cluster | Over-merging; distinct sub-complaints lost |
| **0.80** | Correct grouping on the example corpus | **(default)** |
| 0.85 | Semantically distant paraphrases may remain separate | Under-merging; duplicates survive |
| 0.95 | Near-identical phrasings only; most items remain separate | Deduplication largely ineffective |

The optimal threshold is domain and embedding-model dependent. For hotel review
analysis with `multi-qa-MiniLM-L6-cos-v1`, values in the 0.75–0.82 range tend
to produce the most intuitive results. The threshold can be adjusted by passing
`threshold` directly to `deduplicate_with_analysis()` in `main.py`.