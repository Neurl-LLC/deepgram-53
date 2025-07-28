"""
evaluate.py

Simple information-retrieval (IR) metrics helpers you can extend.
Use these to evaluate the downstream quality of your semantic search:

- nDCG@k  : Normalized Discounted Cumulative Gain at cutoff k
- Recall@k: Fraction of relevant items retrieved within top-k
- MRR     : Mean Reciprocal Rank (1 / rank of first relevant)

Typical usage in this repo:
- The UI lets a user paste a set of "gold" relevant vector IDs for a query.
- After a search, we compute metrics using the IDs returned by Pinecone.

IMPORTANT
---------
These metrics require a notion of "relevance". In this demo we assume:
- Binary relevance (relevant or not).
- "Gold" comes from user input (IDs of relevant segments for the query).

In production you might:
- Store gold labels per query in a file or database,
- Or collect labels via thumbs-up/-down in the UI, then aggregate offline.
"""

from typing import List, Set
import math


def dcg(relevances: List[float]) -> float:
    """
    Discounted Cumulative Gain.
    Input:
      relevances: list of per-rank gains (e.g., 1.0 for relevant, 0.0 otherwise)
    """
    # DCG = sum_{i=1..n} (rel_i / log2(i + 1))
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(pred_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """
    Normalized DCG at cutoff k.

    Args:
      pred_ids: ranked list of IDs returned by the system
      gold_ids: set of relevant IDs (binary relevance)
      k      : cutoff (use top-k predicted IDs)

    Returns:
      nDCG in [0,1], where 1 is best.
    """
    if k <= 0:
        return 0.0
    pred_k = pred_ids[:k]
    gains = [1.0 if pid in gold_ids else 0.0 for pid in pred_k]
    # Ideal ranking is all relevant first
    ideal_gains = sorted(gains, reverse=True)
    idcg = dcg(ideal_gains)
    return 0.0 if idcg == 0 else dcg(gains) / idcg


def recall_at_k(pred_ids: List[str], gold_ids: Set[str], k: int) -> float:
    """
    Recall@k = (# of relevant items retrieved in top-k) / (# of all relevant)

    Returns:
      recall in [0,1]
    """
    if not gold_ids or k <= 0:
        return 0.0
    pred_k = set(pred_ids[:k])
    return len(pred_k & gold_ids) / len(gold_ids)


def mrr(pred_ids: List[str], gold_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank (single-query variant).

    Returns:
      1 / rank of the first relevant item, or 0 if none are found.
    """
    for idx, pid in enumerate(pred_ids, start=1):
        if pid in gold_ids:
            return 1.0 / idx
    return 0.0


if __name__ == "__main__":
    # Tiny example (run `python evaluate.py`)
    queries = {
        "refund escalation": {
            "pred": ["sess1:file1:3", "sess1:file2:10", "sess1:file1:5", "sess1:file3:2"],
            "gold": {"sess1:file1:5", "sess1:file2:10"},
        },
        "launch delay decision": {
            "pred": ["sess9:file7:1", "sess9:file7:2", "sess9:file7:3"],
            "gold": {"sess9:file7:2"},
        },
    }

    for q, data in queries.items():
        pred = data["pred"]
        gold = data["gold"]
        print(f"\nQuery: {q}")
        print(f"  nDCG@5:   {ndcg_at_k(pred, gold, 5):.3f}")
        print(f"  Recall@5: {recall_at_k(pred, gold, 5):.3f}")
        print(f"  MRR:      {mrr(pred, gold):.3f}")
