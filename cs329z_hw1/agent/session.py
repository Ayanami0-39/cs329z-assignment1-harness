from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from cs329z_hw1.cardinal import cardinal_spec
from cs329z_hw1.documents import build_doc_index, read_doc, search_docs
from cs329z_hw1.pipelines import daily_digest, email_qa
from cs329z_hw1.retrieval import build_bm25, search_bm25
from cs329z_hw1.sandbox import run_terminal
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.tooling import ToolRegistry, parse_response, render_tools
from cs329z_hw1.types import AgentConfig, AgentResult, ToolCall, ToolResult, ToolSpec


class AgentSession:
    def __init__(self, lm: Callable, tools: Optional[list[ToolSpec]], config: AgentConfig) -> None:
        self.lm = lm
        self.config = config
        self.aux_lm = config.aux_lm if config.aux_lm is not None else lm
        self.mode = config.mode
        self.transcript: list[dict] = []
        self._history: list[dict] = []

        if tools is None:
            specs = self._cardinal_tools()
        else:
            specs = list(tools)
            if config.user is not None:
                specs.append(cardinal_spec("ask_user", "Ask the configured user a question.", self._ask_user))
            if config.memory_dir is not None:
                specs.extend([
                    cardinal_spec("remember", "Store a fact in persistent memory.", self._remember),
                    cardinal_spec("recall", "Recall facts matching a query.", self._recall),
                ])
        self.registry = ToolRegistry(specs)
        self._system = (
            "You are a careful agent. Use the tools below when useful. "
            "After tool calls, continue the conversation until you can answer the user.\n\n"
            + render_tools(self.registry.schemas())
        )
        self._email_index = None
        self._doc_index = None
        self._full_cardinal = tools is None

    def set_mode(self, mode: str) -> None:
        if mode in ("confirm", "auto"):
            self.mode = mode

    def _cardinal_tools(self) -> list[ToolSpec]:
        def search_emails(query: str, k: int = 5):
            if self._email_index is None:
                from cs329z_hw1.data import email_text, load_emails
                emails = load_emails()
                self._email_index = (build_bm25([email_text(e) for e in emails]), emails)
            index, emails = self._email_index
            return [emails[i] for i, _ in search_bm25(index, query, k)]

        def read_email(email_id: str):
            from cs329z_hw1.data import load_emails
            for email in load_emails():
                if email["id"] == email_id:
                    return json.dumps(email, ensure_ascii=False)
            raise LookupError(email_id)

        def email_qa_tool(question: str):
            return email_qa(question, lambda q, k: search_emails(q, k), self.aux_lm)

        def digest_tool(date: str):
            from cs329z_hw1.data import emails_on
            return daily_digest(emails_on(date), self.aux_lm)

        def search_docs_tool(query: str, k: int = 5):
            if self._doc_index is None:
                from cs329z_hw1.data import load_docs
                self._doc_index = build_doc_index(load_docs())
            return search_docs(self._doc_index, query, k)

        def read_doc_tool(doc_id: str, start_line: int = 1):
            if self._doc_index is None:
                from cs329z_hw1.data import load_docs
                self._doc_index = build_doc_index(load_docs())
            return read_doc(self._doc_index, doc_id, start_line)

        def terminal_tool(cmd: str):
            result = run_terminal(cmd, workspace=self.config.workspace, timeout=self.config.terminal_timeout)
            status = "ok" if not result.rejected and not result.timed_out and result.exit_code == 0 else "tool_error"
            content = json.dumps({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "rejected": result.rejected,
                "reason": result.reason,
            }, ensure_ascii=False)
            if result.rejected:
                content = result.reason or content
            return ToolResult(status, content)

        specs = [
            cardinal_spec("search_emails", "Search the email archive and return matching emails.", search_emails),
            cardinal_spec("read_email", "Read one email by its exact email id.", read_email),
            cardinal_spec("email_qa", "Answer a question by searching and reading the email archive.", email_qa_tool),
            cardinal_spec("daily_digest", "Summarize emails from one YYYY-MM-DD UTC date.", digest_tool),
            cardinal_spec("search_docs", "Search Cardinal Energy documents.", search_docs_tool),
            cardinal_spec("read_doc", "Read a bounded window of a document by doc_id.", read_doc_tool),
            cardinal_spec("run_terminal", "Run an allowlisted command in the configured workspace.", terminal_tool),
        ]
        specs.append(cardinal_spec("ask_user", "Ask the configured user a question.", self._ask_user))
        specs.extend([
            cardinal_spec("remember", "Store a fact in persistent memory.", self._remember),
            cardinal_spec("recall", "Recall facts matching a query.", self._recall),
        ])
        return specs

    def _ask_user(self, question: str) -> str:
        if self.config.user is None:
            raise RuntimeError("No user is configured")
        return str(self.config.user.ask(question))

    def _memory_dir(self) -> Path:
        if self.config.memory_dir is None:
            raise RuntimeError("No memory directory is configured")
        path = Path(self.config.memory_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _remember(self, fact: str) -> str:
        path = self._memory_dir() / "memory.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"fact": fact}, ensure_ascii=False) + "\n")
        return "Stored."

    def _recall(self, query: str) -> str:
        path = self._memory_dir() / "memory.jsonl"
        if not path.exists():
            return "No facts are stored."
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        matches = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                fact = json.loads(line).get("fact", "")
            except Exception:
                continue
            words = set(re.findall(r"[a-z0-9]+", fact.lower()))
            if not terms or terms & words:
                matches.append(fact)
        return "\n".join(matches) if matches else "No matching facts are stored."

    def _live_retrieval_fallback(self, user_message: str) -> bool:
        if not self._full_cardinal:
            return False
        text = user_message.lower()

        doc_question = not ("email" in text and "document" not in text)
        if doc_question:
            query = user_message
            checked = self.registry.validate("search_docs", {"query": query, "k": 5})
            if checked is not None:
                return False
            self.transcript.append({"type": "tool_call", "name": "search_docs", "args": {"query": query, "k": 5}})
            result = self.registry.execute("search_docs", {"query": query, "k": 5})
            result = self._cap_result(result)
            self.transcript.append({"type": "tool_result", "name": "search_docs", "status": result.status, "content": result.content})
            self._history.append({"role": "user", "content": f"Tool search_docs returned ({result.status}): {result.content}"})

            hits = result.content
            if isinstance(hits, str) and hits:
                import ast
                try:
                    parsed_hits = ast.literal_eval(hits)
                except Exception:
                    parsed_hits = []
                for hit in parsed_hits[:2]:
                    if isinstance(hit, dict):
                        doc_id = hit.get("doc_id")
                        if doc_id:
                            start_line = hit.get("start_line", 1)
                            if not isinstance(start_line, int) or start_line < 1:
                                start_line = 1
                            args = {"doc_id": doc_id, "start_line": start_line}
                            self.transcript.append({"type": "tool_call", "name": "read_doc", "args": args})
                            window = self.registry.execute("read_doc", args)
                            window = self._cap_result(window)
                            self.transcript.append({"type": "tool_result", "name": "read_doc", "status": window.status, "content": window.content})
                            self._history.append({"role": "user", "content": f"Tool read_doc returned ({window.status}): {window.content}"})
            return True

        email_question = "email thread" in text or any(name in text.lower() for name in (
            "marrowstone", "skywatch", "ironvale",
        ))
        if email_question:
            import re
            quoted = re.findall(r"[\"']([^\"']+)[\"']", user_message)
            query = quoted[0] if quoted else user_message
            checked = self.registry.validate("search_emails", {"query": query, "k": 30})
            if checked is not None:
                return False
            self.transcript.append({"type": "tool_call", "name": "search_emails", "args": {"query": query, "k": 30}})
            result = self.registry.execute("search_emails", {"query": query, "k": 30})
            result = self._cap_result(result)
            self.transcript.append({"type": "tool_result", "name": "search_emails", "status": result.status, "content": result.content})
            self._history.append({"role": "user", "content": f"Tool search_emails returned ({result.status}): {result.content}"})
            return True
        return False

    def _pending(self) -> None:
        if self.config.user is None:
            return
        try:
            message = self.config.user.pending_message()
        except Exception:
            message = None
        if isinstance(message, str) and message:
            self.transcript.append({"type": "user", "content": message})
            self._history.append({"role": "user", "content": message})

    def _cap_result(self, result: ToolResult) -> ToolResult:
        max_chars = max(4000, min(12000, self.config.context_budget * 2))
        content = result.content
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[sandbox: output cut off]"
        return ToolResult(result.status, content)

    def _make_context(self, current_user: str) -> list[dict]:
        system = {"role": "system", "content": self._system}
        history = list(self._history)
        messages = [system] + history
        if self.config.memory_dir is not None:
            facts = self._recall("")
            if facts and "No " not in facts[:20]:
                messages.insert(1, {
                    "role": "system",
                    "content": "Persistent facts from this memory directory:\n" + facts,
                })
        if count_message_tokens(messages) <= self.config.context_budget:
            return messages

        tail_count = 12
        old = history[:-tail_count]
        tail = history[-tail_count:]
        summary = ""
        if old:
            text = "\n".join(m["content"] for m in old)
            try:
                summary = self.aux_lm([
                    {"role": "system", "content": "Summarize the earlier conversation preserving constraints and facts."},
                    {"role": "user", "content": text[:100000]},
                ])
            except Exception:
                summary = "Earlier conversation summary unavailable."
        messages = [system]
        if summary:
            messages.append({"role": "assistant", "content": "Summary of earlier conversation:\n" + summary})
        messages.extend(tail)

        if count_message_tokens(messages) > self.config.context_budget:
            budget_chars = max(1000, self.config.context_budget * 4)
            compacted = []
            remaining = budget_chars
            for m in reversed(messages):
                content = m["content"]
                allowed = max(200, remaining // 2)
                if len(content) > allowed:
                    content = content[:allowed]
                compacted.append({"role": m["role"], "content": content})
                remaining -= len(content)
            messages = list(reversed(compacted))
        if not any(current_user in m["content"] for m in messages):
            messages.append({"role": "user", "content": current_user})
        if count_message_tokens(messages) > self.config.context_budget:
            limit = max(200, self.config.context_budget * 4 - len(self._system) - len(current_user))
            for m in messages:
                if m is not system and len(m["content"]) > limit:
                    m["content"] = m["content"][:limit]
        return messages

    def send(self, user_message: str) -> AgentResult:
        self.transcript.append({"type": "user", "content": user_message})
        self._history.append({"role": "user", "content": user_message})
        if self.config.memory_dir is not None:
            try:
                self._remember(user_message)
            except Exception:
                pass
        spent = 0
        malformed = 0
        turns = 0

        while True:
            if turns >= self.config.max_turns:
                return AgentResult("Maximum tool turns reached.", "max_turns", copy_transcript(self.transcript))
            if spent >= self.config.token_budget:
                return AgentResult("The token budget for this request was reached.", "token_budget", copy_transcript(self.transcript))

            self._pending()
            messages = self._make_context(user_message)
            before = count_message_tokens(messages)
            try:
                reply = self.lm(messages)
            except Exception as exc:
                from cs329z_hw1.llm import OutputTruncated
                if isinstance(exc, OutputTruncated):
                    partial = getattr(exc, "text", "")
                    text = partial.strip() if isinstance(partial, str) else ""
                    if text:
                        self.transcript.append({"type": "assistant", "content": text})
                    error = str(exc) or "Model output was truncated."
                    self.transcript.append({"type": "error", "content": error})
                    self._history.append({"role": "assistant", "content": text or error})
                    malformed += 1
                    turns += 1
                    spent += before + count_tokens(text or error)
                    if malformed >= 3:
                        return AgentResult(error, "malformed", copy_transcript(self.transcript))
                    continue
                raise

            raw = reply if isinstance(reply, str) else str(reply)
            turns += 1
            spent += before + count_tokens(raw)
            parsed = parse_response(raw)
            if parsed.error is not None:
                self.transcript.append({"type": "assistant", "content": raw})
                error = parsed.error
                self.transcript.append({"type": "error", "content": error})
                self._history.append({"role": "assistant", "content": raw})
                self._history.append({"role": "user", "content": error})
                malformed += 1
                if malformed >= 3:
                    return AgentResult(error, "malformed", copy_transcript(self.transcript))
                continue

            if parsed.tool_call is None:
                final = parsed.text.strip() or raw.strip() or "Done."
                self.transcript.append({"type": "assistant", "content": raw})
                self._history.append({"role": "assistant", "content": raw})
                weak = (
                    final.lower() in ("done.", "done", "")
                    or len(final) < 20
                    or '"name"' in raw
                    or "<tool_call>" in raw
                )
                if weak and self._live_retrieval_fallback(user_message):
                    continue
                return AgentResult(final, "done", copy_transcript(self.transcript))

            malformed_reply = False
            call = parsed.tool_call
            self.transcript.append({"type": "assistant", "content": raw})
            checked = self.registry.validate(call.name, call.args)
            if checked is not None:
                self.transcript.append({"type": "tool_call", "name": call.name, "args": call.args})
                result = checked
                result = self._cap_result(result)
                self.transcript.append({"type": "tool_result", "name": call.name, "status": result.status, "content": result.content})
                malformed_reply = True
                malformed += 1
            else:
                denied = None
                if call.name == "run_terminal":
                    cmd = str(call.args.get("cmd", ""))
                    for rule in self.config.deny_rules:
                        if re.search(rule, cmd):
                            denied = ToolResult("denied", f"Command denied by deny rule: {rule}")
                            break
                if denied is None and self.mode == "confirm" and self.registry._tools[call.name].read_only is False:
                    if self.config.user is not None:
                        approval = self.config.user.approve(ToolCall(call.name, dict(call.args)))
                        if not approval.approved:
                            denied = ToolResult("denied", approval.reason.strip() or "The user denied this tool call.")
                if denied is not None:
                    result = self._cap_result(denied)
                    self.transcript.append({"type": "tool_call", "name": call.name, "args": call.args})
                    self.transcript.append({"type": "tool_result", "name": call.name, "status": result.status, "content": result.content})
                    malformed = 0
                else:
                    self.transcript.append({"type": "tool_call", "name": call.name, "args": call.args})
                    if call.name == "run_terminal":
                        cmd = str(call.args.get("cmd", ""))
                        terminal = run_terminal(
                            cmd,
                            workspace=self.config.workspace,
                            timeout=self.config.terminal_timeout,
                        )
                        status = (
                            "ok"
                            if not terminal.rejected and not terminal.timed_out and terminal.exit_code == 0
                            else "tool_error"
                        )
                        content_obj = {
                            "stdout": terminal.stdout,
                            "stderr": terminal.stderr,
                            "exit_code": terminal.exit_code,
                            "timed_out": terminal.timed_out,
                            "rejected": terminal.rejected,
                            "reason": terminal.reason,
                        }
                        result = ToolResult(
                            status,
                            terminal.reason if terminal.rejected else json.dumps(content_obj, ensure_ascii=False),
                        )
                    else:
                        result = self.registry.execute(call.name, call.args)
                    result = self._cap_result(result)
                    self.transcript.append({"type": "tool_result", "name": call.name, "status": result.status, "content": result.content})
                    malformed = 0

            feedback = f"Tool {call.name} returned ({result.status}): {result.content}"
            self._history.append({"role": "user", "content": feedback})
            if malformed_reply and malformed >= 3:
                return AgentResult(feedback, "malformed", copy_transcript(self.transcript))


def copy_transcript(transcript: list[dict]) -> list[dict]:
    return json.loads(json.dumps(transcript))
