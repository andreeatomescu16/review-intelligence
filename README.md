# Review Intelligence System

## Overview

The Review Intelligence System is a REST API that processes a batch of hotel
reviews to surface the most commonly mentioned positives and negatives. For
each review it calls a locally-running LLM (llama3.2:1b via Ollama) to extract
structured highlights and pain points, then uses dense semantic embeddings
(multi-qa-MiniLM-L6-cos-v1 via sentence-transformers) to cluster near-duplicate
phrases across reviews. The result is a ranked list of unique highlights and
pain points, each annotated with how many reviews mentioned that theme.

## Architecture

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI application; orchestrates the pipeline; owns the embedder singleton via FastAPI lifespan |
| `app/extractor.py` | Async HTTP calls to Ollama; structured JSON extraction; retry with exponential backoff |
| `app/embedder.py` | Loads SentenceTransformer once; encodes lists of strings into L2-normalised embedding vectors |
| `app/deduplicator.py` | Computes pairwise cosine similarity; Union-Find clustering; centroid-based representative selection |

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (Engine 20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2 plugin or standalone)

No other local dependencies are required — the LLM and embedding model are
both fetched/baked during the build and first run.

## Running the project

```bash
docker compose up --build
# wait for the model to be pulled (~1-2 min on first run)
curl -X POST http://localhost:8000/process_file
```

The `ollama-pull` init service will download `llama3.2:1b` on first run and
cache it in a Docker volume (`ollama_models`). Subsequent starts skip the
download. The app container will not start until the pull is complete.

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

> Note: exact phrasing of representative items depends on which cluster member
> the model selected as centroid; counts reflect the number of reviews that
> mentioned each theme.

## Design decisions

### Embedding model: multi-qa-MiniLM-L6-cos-v1 over all-MiniLM-L6-v2

`multi-qa-MiniLM-L6-cos-v1` was fine-tuned explicitly on semantic
similarity / question-answer retrieval datasets, making its embedding space
better calibrated for detecting "same concept, different wording" across
paraphrases. `all-MiniLM-L6-v2` is trained on a broader but less focused
corpus, which produces good general-purpose embeddings but slightly weaker
discrimination on short, noun-heavy phrases like hotel review bullet points.
Both models have the same architecture (6 layers, 384-dimensional output,
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

### Centroid-based representative selection

When multiple review phrases are merged into one cluster, the representative
phrase is the one with the highest mean cosine similarity to all other members
of the cluster — i.e. the phrase closest to the geometric centroid of the
cluster in embedding space. This is preferable to selecting the first item
encountered because the first item is order-dependent (it varies with the
order reviews appear in the file) and may be an outlier phrase that happened
to exceed the similarity threshold with one neighbour. The centroid item is
the most "canonical" phrasing of the shared concept.

### Greedy Union-Find clustering

Union-Find with pairwise threshold comparison is deterministic (given a fixed
similarity matrix), requires no hyperparameters beyond the threshold, and
produces stable results as the item list grows. It is equivalent to
single-linkage agglomerative clustering halted at a fixed distance, which is
appropriate here because we expect tight paraphrase clusters rather than
elongated chains. The O(n²) pairwise scan is acceptable because n (the number
of extracted items per request) is small in practice — typically 5–30 items
for a file of hotel reviews.

## Threshold sensitivity

The default similarity threshold is **0.8**.

At **0.75**, the three check-in pain point variants in the example reviews
("check-in took too long", "waiting time at check-in was frustrating",
"reception process was slow") all collapse into a single cluster, because their
embeddings are close enough in the `multi-qa-MiniLM-L6-cos-v1` space.

At **0.80** (the default), slightly more distant paraphrases may remain
separate — this gives a cleaner output when reviews contain genuinely distinct
sub-complaints that happen to share keywords.

At **0.85**, semantically distant paraphrases are more likely to remain in
separate clusters, which reduces over-merging at the cost of potentially
showing multiple entries for what a human reader would consider the same issue.

The optimal threshold is domain and embedding-model dependent. For hotel review
analysis with this embedding model, values in the 0.75–0.82 range tend to
produce the most intuitive results. The threshold can be adjusted by passing
`threshold` directly to `deduplicate()` in `main.py`.
