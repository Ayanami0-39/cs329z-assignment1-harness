from __future__ import annotations

import math
from dataclasses import dataclass

from cs329z_hw1.tokenizer import tokenize


@dataclass
class BM25Index:
    n: int
    avgdl: float
    lengths: list[int]
    postings: dict[str, list[tuple[int, int]]]


def build_bm25(docs: list[str]) -> BM25Index:
    lengths = []
    postings: dict[str, list[tuple[int, int]]] = {}
    for doc_id, text in enumerate(docs):
        tokens = tokenize(text)
        lengths.append(len(tokens))
        counts: dict[str, int] = {}
        for term in tokens:
            counts[term] = counts.get(term, 0) + 1
        for term, tf in counts.items():
            postings.setdefault(term, []).append((doc_id, tf))
    return BM25Index(
        n=len(docs),
        avgdl=(sum(lengths) / len(lengths)) if lengths else 0.0,
        lengths=lengths,
        postings=postings,
    )


def search_bm25(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    if k <= 0 or index.n == 0:
        return []
    scores: dict[int, float] = {}
    for term in set(tokenize(query)):
        entries = index.postings.get(term)
        if not entries:
            continue
        n = len(entries)
        idf = math.log((index.n - n + 0.5) / (n + 0.5) + 1)
        for doc_id, tf in entries:
            length = index.lengths[doc_id]
            norm = 1.5 * (0.25 + 0.75 * length / index.avgdl) if index.avgdl else 1.5
            contribution = idf * tf * 2.5 / (tf + norm)
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]
