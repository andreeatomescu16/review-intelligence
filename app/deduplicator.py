"""
deduplicator.py

Groups semantically equivalent strings (highlights or pain points) and returns
one representative per group together with the group's size (count).

Algorithm overview
------------------
1. Compute pairwise cosine similarity via dot product (embeddings are assumed
   L2-normalised — see embedder.py).
2. Greedy Union-Find clustering: iterate all pairs (i, j) in sorted order; if
   similarity exceeds the threshold, merge the two groups. This is O(n²) in
   the number of items but n is small (typically <50 per request), so the
   quadratic cost is acceptable.
3. Centroid representative selection: for each cluster, pick the member whose
   mean cosine similarity to all *other* members in the cluster is highest.
   This item sits closest to the semantic centroid and is therefore the most
   representative phrasing — preferable to simply picking the first item
   encountered, which is arbitrary and order-dependent.

Union-Find vs. single-linkage clustering
-----------------------------------------
Union-Find with a pairwise threshold is equivalent to single-linkage
agglomerative clustering stopped at a fixed distance. It is chosen here
because it is deterministic, requires no additional hyperparameters, and is
trivial to implement without a scipy/sklearn dependency.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Threshold of 0.8 was specified in the requirements.
# Empirically, at 0.75 all three pain point variants in the example
# ("check-in took too long", "waiting time at check-in was frustrating",
# "reception process was slow") collapse correctly into one cluster.
# At 0.85, semantically distant paraphrases may remain separate.
# The optimal value is domain and embedding-model dependent.
_DEFAULT_THRESHOLD = 0.8


def deduplicate(
    items: list[str],
    embeddings: np.ndarray,
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[dict]:
    """
    Cluster semantically similar items and return one representative per
    cluster with its occurrence count.

    Args:
        items:      Strings to deduplicate (highlights or pain points).
        embeddings: L2-normalised embedding matrix of shape (len(items), dim).
        threshold:  Cosine similarity threshold above which two items are
                    considered duplicates. Default is 0.8.

    Returns:
        List of dicts ``[{"item": str, "count": int}, ...]`` sorted by count
        descending, then alphabetically by item for deterministic output.
    """
    num_items = len(items)

    if num_items == 0:
        return []

    if num_items == 1:
        return [{"item": items[0], "count": 1}]

    # --- Step 1: pairwise cosine similarity matrix ---
    # Since embeddings are L2-normalised, cosine similarity == dot product.
    # Result shape: (num_items, num_items). Diagonal is 1.0 (self-similarity).
    similarity_matrix: np.ndarray = embeddings @ embeddings.T

    # --- Step 2: Union-Find initialisation ---
    # parent[i] holds the root representative of item i's group.
    parent = list(range(num_items))

    def find(node: int) -> int:
        """Return root of the group containing node (with path compression)."""
        while parent[node] != node:
            # Path compression: point directly to grandparent
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(node_a: int, node_b: int) -> None:
        """Merge the groups of node_a and node_b."""
        root_a = find(node_a)
        root_b = find(node_b)
        if root_a != root_b:
            # Attach the higher-indexed root to the lower-indexed one for
            # determinism (no rank heuristic needed at this scale).
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    # --- Step 3: merge pairs that exceed the similarity threshold ---
    for idx_i in range(num_items):
        for idx_j in range(idx_i + 1, num_items):
            if similarity_matrix[idx_i, idx_j] > threshold:
                union(idx_i, idx_j)

    # --- Step 4: collect members of each cluster ---
    clusters: dict[int, list[int]] = {}
    for idx in range(num_items):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    logger.debug(
        "deduplicate: %d items → %d clusters (threshold=%.2f)",
        num_items,
        len(clusters),
        threshold,
    )

    # --- Step 5: centroid representative selection ---
    result: list[dict] = []

    for member_indices in clusters.values():
        cluster_size = len(member_indices)

        if cluster_size == 1:
            # No comparison needed for singleton clusters
            representative = items[member_indices[0]]
        else:
            # Extract the sub-matrix of pairwise similarities within the cluster
            cluster_embeddings = embeddings[member_indices]  # shape: (k, dim)
            intra_similarity: np.ndarray = cluster_embeddings @ cluster_embeddings.T
            # shape: (k, k)

            # Mean similarity to all *other* members (exclude self-similarity
            # on the diagonal by subtracting 1/(k) from the row mean, then
            # re-scaling — equivalent to zeroing the diagonal and averaging).
            np.fill_diagonal(intra_similarity, 0.0)
            if cluster_size > 1:
                mean_sim_to_others = intra_similarity.sum(axis=1) / (cluster_size - 1)
            else:
                mean_sim_to_others = np.array([1.0])

            # The item with the highest mean similarity sits closest to the
            # cluster centroid and is chosen as the representative.
            best_local_idx = int(np.argmax(mean_sim_to_others))
            representative = items[member_indices[best_local_idx]]

        result.append({"item": representative, "count": cluster_size})

    # Sort by count descending; break ties alphabetically for determinism
    result.sort(key=lambda entry: (-entry["count"], entry["item"]))

    return result
