import os
import subprocess


def get_git_commit(repo_root: str | None = None) -> str:
    """Returns the current HEAD commit hash, or 'unknown' if unavailable."""
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def get_doc_id(hit) -> str:
    """Returns PMID string if available, falling back to Milvus entity ID."""
    pmid = getattr(hit.entity, "pmid", None)
    if pmid not in (None, ""):
        return str(pmid)
    return str(getattr(hit, "id", ""))


def precision_at_k(ranked_pmids: list[str], labels_for_query: dict[str, int], k: int) -> float:
    """Computes Precision@k against a {pmid: 0_or_1} relevance dictionary."""
    top_hits = ranked_pmids[:k]
    if not top_hits:
        return 0.0
    relevant_hits = sum(labels_for_query.get(pmid, 0) for pmid in top_hits)
    return relevant_hits / len(top_hits)


def top_k_overlap(ids_a: list[str], ids_b: list[str]) -> dict:
    set_a, set_b = set(ids_a), set(ids_b)
    denom = len(set_a | set_b)
    return {
        "shared_count": len(set_a & set_b),
        "jaccard": round(len(set_a & set_b) / denom, 4) if denom else 1.0,
    }


def kendall_tau_union(ranking_a: list[str], ranking_b: list[str]) -> float:
    """Computes Kendall's tau over the union of two ordered doc ID lists."""
    union_ids = list(dict.fromkeys(ranking_a + ranking_b))
    n = len(union_ids)
    if n < 2:
        return 1.0

    rank_a = {doc_id: idx for idx, doc_id in enumerate(ranking_a)}
    rank_b = {doc_id: idx for idx, doc_id in enumerate(ranking_b)}
    default_a, default_b = len(ranking_a), len(ranking_b)

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = rank_a.get(union_ids[i], default_a) - rank_a.get(union_ids[j], default_a)
            db = rank_b.get(union_ids[i], default_b) - rank_b.get(union_ids[j], default_b)
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    denom = concordant + discordant
    return round((concordant - discordant) / denom, 4) if denom else 1.0

