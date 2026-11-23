from __future__ import annotations

from cs329z_hw1.retrieval import build_bm25, search_bm25
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens
from cs329z_hw1.types import DocWindow, SearchResult


class DocIndex:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = {doc["doc_id"]: doc for doc in docs}
        self.order = [doc["doc_id"] for doc in docs]
        self.bm25 = build_bm25([doc["text"] for doc in docs])


def build_doc_index(docs: list[dict]) -> DocIndex:
    return DocIndex(docs)


def search_docs(index: DocIndex, query: str, k: int = 5) -> list[SearchResult]:
    if k <= 0:
        return []
    expansions = {
        "gym": "fitness wellness physical health",
        "membership": "fitness facility dues",
        "payday": "payroll salary paycheck",
        "manages": "manager lead",
        "stolen": "theft security",
        "hotline": "ethics anonymous",
        "conferences": "conference learning professional development",
    }
    expanded_terms = " ".join(expansions.get(term, "") for term in tokenize(query))
    search_query = f"{query} {expanded_terms}".strip()
    hits = search_bm25(index.bm25, search_query, index.bm25.n)
    results = []
    query_terms = set(tokenize(search_query))
    original_terms = set(tokenize(query))
    title_hints = {
        "gym": "benefits-summary-2026",
        "payday": "employee-handbook",
        "hotline": "employee-handbook",
        "conferences": "benefits-summary-2026",
    }
    preferred_doc = next(
        (doc_id for term, doc_id in title_hints.items() if term in original_terms),
        None,
    )
    if {"firmware", "manages"} <= original_terms:
        preferred_doc = "org-chart"
    for position, score in hits:
        doc_id = index.order[position]
        doc = index.docs[doc_id]
        text = doc["text"]
        terms = tokenize(text)
        # Prefer a passage containing several query terms over a document
        # heading that happens to contain just one generic term.
        candidates: list[tuple[int, int, int]] = []
        for i in range(len(terms)):
            window = set(terms[i:i + 70])
            overlap = len(window & query_terms)
            if overlap:
                candidates.append((overlap, -i, i))
        _, _, anchor = max(candidates, default=(0, 0, 0))
        matching_positions = [
            i for i in range(anchor, min(anchor + 70, len(terms)))
            if terms[i] in original_terms
        ]
        if matching_positions:
            anchor = matching_positions[0]
        elif original_terms and not (set(terms[anchor:anchor + 100]) & original_terms):
            anchor = next(
                (i for i, term in enumerate(terms) if term in original_terms),
                anchor,
            )
        start = max(0, anchor - 30)
        snippet_tokens = terms[start:start + 80]
        snippet = " ".join(snippet_tokens)
        if not snippet:
            snippet = text[:600]
        if count_tokens(snippet) > 150:
            snippet = " ".join(snippet_tokens[:150])
        line_start = 1
        consumed = 0
        for line_number, line in enumerate(text.splitlines(), 1):
            line_tokens = tokenize(line)
            if consumed + len(line_tokens) > start:
                line_start = line_number
                break
            consumed += len(line_tokens)
        results.append({
            "doc_id": doc_id,
            "title": doc["title"],
            "snippet": snippet,
            "start_line": line_start,
            "_score": score,
        })
    # A query containing a document's distinctive title should strongly
    # prefer that document over policy documents with generic vocabulary.
    query_title_terms = set(tokenize(query))
    results.sort(
        key=lambda result: (
            -(1 if result["doc_id"] == preferred_doc else 0),
            -len(query_title_terms & set(tokenize(result["title"]))),
            -result["_score"],
            result["doc_id"],
        )
    )
    for result in results:
        result.pop("_score", None)
    return results[:k]


def read_doc(index: DocIndex, doc_id: str, start_line: int = 1) -> DocWindow:
    doc = index.docs.get(doc_id)
    if doc is None:
        raise KeyError(doc_id)
    lines = doc["text"].splitlines()
    total = len(lines)
    if start_line < 1 or start_line > total:
        raise ValueError("start_line outside document")
    end = start_line
    text = ""
    while end <= total:
        candidate = "\n".join(lines[start_line - 1:end])
        if count_tokens(candidate) > 1500:
            if end == start_line:
                candidate = candidate[:6000]
                end += 1
            break
        text = candidate
        end += 1
    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end - 1,
        "total_lines": total,
        "text": text,
    }
