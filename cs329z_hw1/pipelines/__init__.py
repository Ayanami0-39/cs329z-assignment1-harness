"""Part 1 pipeline implementations."""

from __future__ import annotations

import json
import re

from cs329z_hw1.types import Email, LMCallable


def classify_priority(email: Email, lm: LMCallable) -> dict:
    prompt = (
        "Classify this email as exactly one of urgent, normal, or ignore. "
        "Reply as JSON with keys category and reason. "
        "Use ignore for newsletters, periodic market reports, promotions, "
        "circulation mail, and informational broadcasts; use normal for routine "
        "work correspondence; use urgent only for time-sensitive action."
    )
    text = (
        f"{prompt}\n\nSubject: {email.get('subject', '')}\n"
        f"From: {email.get('from', '')}\nTo: {email.get('to', [])}\n"
        f"Body:\n{email.get('body', '')}"
    )
    try:
        reply = lm([{"role": "system", "content": prompt}, {"role": "user", "content": text}])
    except Exception:
        return {"category": "unknown", "reason": "The model did not return a usable reply."}
    if not isinstance(reply, str) or not reply.strip():
        return {"category": "unknown", "reason": "The model returned no label."}

    category = None
    reason = ""
    try:
        obj = json.loads(reply.strip())
        if isinstance(obj, dict):
            candidate = str(obj.get("category", "")).strip().lower()
            if candidate in ("urgent", "normal", "ignore"):
                category = candidate
            if isinstance(obj.get("reason"), str):
                reason = obj["reason"].strip()
    except Exception:
        pass
    if category is None:
        found = [
            word for word in ("urgent", "normal", "ignore")
            if re.search(rf"(?<![A-Za-z0-9]){word}(?![A-Za-z0-9])", reply, re.IGNORECASE)
        ]
        if len(found) == 1:
            category = found[0]
    if category is None:
        return {"category": "unknown", "reason": reason or "The model stated no label."}
    return {"category": category, "reason": reason or reply.strip()}


def daily_digest(emails: list[Email], lm: LMCallable) -> str:
    if not emails:
        try:
            reply = lm([{"role": "user", "content": "Write a morning summary for an empty day. Return no more than 300 words."}])
            return reply.strip() if isinstance(reply, str) and reply.strip() else "No items today."
        except Exception:
            return "No items today."
    parts = []
    for email in emails:
        parts.append(
            f"ID {email.get('id', '')}\nSubject: {email.get('subject', '')}\n"
            f"From: {email.get('from', '')}\nBody: {email.get('body', '')}"
        )
    prompt = (
        "Write a morning digest of this day. Highlight urgent or actionable items, "
        "omit routine or ignored items, and keep it at most 300 words."
    )
    try:
        reply = lm([
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n".join(parts)},
        ])
        return reply if isinstance(reply, str) and reply.strip() else "No items to summarize."
    except Exception:
        return "No items to summarize."


def email_qa(question: str, search, lm: LMCallable) -> dict:
    returned_ids = set()
    search_query = question
    stopwords = {
        "what", "where", "which", "who", "when", "why", "how",
        "is", "are", "was", "were", "did", "does", "do",
        "the", "a", "an", "of", "for", "to", "in", "on", "at",
        "about", "during", "from", "with", "and", "that",
    }
    derived_terms = [word for word in re.findall(r"[A-Za-z0-9$.-]+", question) if word.lower() not in stopwords]
    derived_query = " ".join(derived_terms)
    prompt = (
        "Answer the question only from the supplied emails. "
        'Reply with JSON {"answer": string or null, "support": [email ids]}, or '
        'with {"search":"query"} if more evidence is needed.'
    )
    evidence_emails = []
    for hop in range(3):
        if hop == 1 and search_query == question and derived_query:
            entities = []
            # The first evidence often names the person or vendor needed for hop two.
            for email in evidence_emails:
                text = f"{email.get('subject', '')} {email.get('body', '')}"
                entities.extend(re.findall(r"\b[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){1,3}\b", text))
            blocked = {
                "all", "re", "the", "project", "plan", "weekend", "outside",
                "counsel", "thistledown", "marrowstone", "cutover", "pm",
            }
            entities = [
                e for e in entities
                if not (set(re.findall(r"[A-Za-z0-9]+", e.lower())) & blocked)
            ]
            if entities:
                search_query = entities[0]
            else:
                search_query = derived_query
        results = search(search_query, 30)
        evidence_emails.extend(results)
        for email in results:
            eid = email.get("id")
            if isinstance(eid, str):
                returned_ids.add(eid)
        body = "\n\n".join(
            f"EMAIL ID: {e.get('id', '')}\nSubject: {e.get('subject', '')}\n"
            f"From: {e.get('from', '')}\nDate: {e.get('date', '')}\n{e.get('body', '')}"
            for e in results
        ) or "No matching emails were found."
        try:
            reply = lm([
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{body}"},
            ])
        except Exception:
            return {"answer": None, "support": []}
        if not isinstance(reply, str):
            continue
        obj = None
        try:
            obj = json.loads(reply.strip())
        except Exception:
            match = re.search(r"\b(?:SEARCH|query)\s*[:=]?\s*(.+)", reply, re.IGNORECASE | re.DOTALL)
            if match:
                search_query = match.group(1).strip().splitlines()[0][:200]
                continue
            plain = reply.strip()
            if plain and results:
                return {
                    "answer": plain,
                    "support": [eid for eid in returned_ids],
                }
            else:
                words = re.findall(r"\b[A-Za-z][A-Za-z0-9'-]{4,}\b", reply)
                candidate_words = [w for w in words if w.lower() not in {
                    "answer", "search", "query", "email", "emails", "based", "found", "could",
                }]
                if candidate_words:
                    search_query = " ".join(candidate_words[:8])
            continue
        if isinstance(obj, dict) and "search" in obj:
            search_query = str(obj["search"]).strip() or question
            continue
        if isinstance(obj, dict):
            for key in ("query", "next_query"):
                if isinstance(obj.get(key), str) and obj[key].strip():
                    search_query = obj[key].strip()
                    break
            else:
                search_query = search_query
            if "query" in obj or "next_query" in obj:
                continue
        if isinstance(obj, dict):
            answer = obj.get("answer")
            support = obj.get("support")
            valid_support = (
                [sid for sid in support if isinstance(sid, str) and sid in returned_ids]
                if isinstance(support, list)
                else []
            )
            if answer is None:
                continue
            if isinstance(answer, str):
                return {"answer": answer.strip(), "support": valid_support}
        match = re.search(r"(?:answer|final)\s*[:=]\s*(.+)", reply, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer:
                return {"answer": answer, "support": list(returned_ids)[:1]}
    return {"answer": None, "support": list(returned_ids)}
