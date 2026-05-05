"""
deduplicator.py

Groups semantically equivalent strings (highlights or pain points) and returns
one representative per group together with the group's size (count).

Algorithm overview
------------------
1. Compute pairwise cosine similarity via dot product (embeddings are assumed
   L2-normalised — see embedder.py).
2. Greedy Union-Find clustering: iterate all pairs (i, j); if similarity
   exceeds the threshold, merge the two groups. O(n²) in the number of items
   but n is small (typically <50 per request).
3. Centroid representative selection: for each cluster, pick the member whose
   mean cosine similarity to all *other* members is highest — the item closest
   to the semantic centroid, independent of input order.

Threshold choice
----------------
0.75 is the empirically validated value for multi-qa-MiniLM-L6-cos-v1 on hotel
review phrasing. At this threshold "check-in took too long", "waiting time at
check-in was frustrating", and "check-in queue was unacceptably long" all
collapse into one cluster correctly. At 0.80 they do not.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.70


def _run_clustering(
    items: list[str],
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[dict], dict]:
    """
    Core clustering logic. Returns (results, analysis_dict).

    analysis_dict contains:
      - threshold: float used
      - items: original string list
      - similarity_matrix: n×n nested list, rounded to 4 dp
      - clusters: list of cluster detail dicts (members, representative,
                  intra-cluster similarities, mean similarities to others)
    """
    num_items = len(items)

    if num_items == 0:
        return [], {
            "threshold": threshold,
            "items": [],
            "similarity_matrix": [],
            "clusters": [],
        }

    # L2-normalised embeddings → dot product == cosine similarity
    similarity_matrix: np.ndarray = embeddings @ embeddings.T

    if num_items == 1:
        return (
            [{"item": items[0], "count": 1}],
            {
                "threshold": threshold,
                "items": items,
                "similarity_matrix": [[1.0]],
                "clusters": [
                    {
                        "members": [
                            {"index": 0, "text": items[0], "mean_sim_to_others": None}
                        ],
                        "representative": items[0],
                        "count": 1,
                        "intra_similarities": [],
                    }
                ],
            },
        )

    # --- Union-Find ---
    parent = list(range(num_items))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i in range(num_items):
        for j in range(i + 1, num_items):
            if similarity_matrix[i, j] > threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for idx in range(num_items):
        clusters.setdefault(find(idx), []).append(idx)

    logger.debug(
        "deduplicate: %d items → %d clusters (threshold=%.2f)",
        num_items,
        len(clusters),
        threshold,
    )

    # --- Per-cluster representative selection and analysis ---
    result: list[dict] = []
    cluster_analyses: list[dict] = []

    for member_indices in clusters.values():
        cluster_size = len(member_indices)

        if cluster_size == 1:
            representative = items[member_indices[0]]
            member_analysis = [
                {
                    "index": member_indices[0],
                    "text": items[member_indices[0]],
                    "mean_sim_to_others": None,
                }
            ]
            intra_sims: list[dict] = []
        else:
            cluster_emb = embeddings[member_indices]
            intra_sim = cluster_emb @ cluster_emb.T  # shape (k, k)
            np.fill_diagonal(intra_sim, 0.0)
            mean_sim = intra_sim.sum(axis=1) / (cluster_size - 1)
            best_local = int(np.argmax(mean_sim))
            representative = items[member_indices[best_local]]

            member_analysis = [
                {
                    "index": member_indices[k],
                    "text": items[member_indices[k]],
                    "mean_sim_to_others": round(float(mean_sim[k]), 4),
                }
                for k in range(cluster_size)
            ]

            # Use the global similarity_matrix (unmodified) for pair values
            intra_sims = [
                {
                    "index_a": member_indices[ki],
                    "index_b": member_indices[kj],
                    "item_a": items[member_indices[ki]],
                    "item_b": items[member_indices[kj]],
                    "similarity": round(
                        float(similarity_matrix[member_indices[ki], member_indices[kj]]),
                        4,
                    ),
                }
                for ki in range(cluster_size)
                for kj in range(ki + 1, cluster_size)
            ]

        result.append({"item": representative, "count": cluster_size})
        cluster_analyses.append(
            {
                "members": member_analysis,
                "representative": representative,
                "count": cluster_size,
                "intra_similarities": intra_sims,
            }
        )

    result.sort(key=lambda e: (-e["count"], e["item"]))

    analysis = {
        "threshold": threshold,
        "items": items,
        "similarity_matrix": [
            [round(float(v), 4) for v in row]
            for row in similarity_matrix.tolist()
        ],
        "clusters": cluster_analyses,
    }

    return result, analysis


def deduplicate(
    items: list[str],
    embeddings: np.ndarray,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[dict]:
    """Cluster and deduplicate items; return ranked results without analysis data."""
    result, _ = _run_clustering(items, embeddings, threshold)
    return result


def deduplicate_with_analysis(
    items: list[str],
    embeddings: np.ndarray,
    threshold: float = _DEFAULT_THRESHOLD,
) -> tuple[list[dict], dict]:
    """Cluster and deduplicate items; return (ranked results, full analysis dict)."""
    return _run_clustering(items, embeddings, threshold)
