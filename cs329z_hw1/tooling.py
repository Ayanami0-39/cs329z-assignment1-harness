from __future__ import annotations

import base64
import json
import re

import jsonschema

from cs329z_hw1.types import ParsedResponse, ToolCall, ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._order = list(tools)

    def schemas(self) -> list[dict]:
        return [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in self._order
        ]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult("unknown_tool", f"Unknown tool: {name}")
        try:
            if not isinstance(args, dict):
                raise jsonschema.ValidationError("Arguments must be an object")
            jsonschema.validate(args, tool.parameters)
        except Exception as exc:
            return ToolResult("invalid_args", f"Invalid arguments for {name}: {exc}")
        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        checked = self.validate(name, args)
        if checked is not None:
            return checked
        tool = self._tools[name]
        try:
            value = tool.fn(**args)
            return ToolResult("ok", str(value))
        except Exception as exc:
            return ToolResult("tool_error", f"{type(exc).__name__}: {exc}")


def render_tools(schemas: list[dict]) -> str:
    lines = [
        "When you need to call a tool, put exactly one call at the end of your reply.",
        "Use this format: <tool_call:BASE64_JSON>, where BASE64_JSON encodes",
        '{"name":"tool_name","args":{"parameter":"value"}}.',
        "Plain text before the call is allowed. JSON argument types must be preserved.",
        "Do not use provider-specific DSML tags, XML invoke tags, or Markdown code fences.",
        "If you know the answer, give it directly instead of inventing another tool call.",
        "",
    ]
    for schema in schemas:
        name = schema["name"]
        params = schema.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])
        param_text = ", ".join(str(key) for key in properties)
        lines.append(f"- {name}: {schema['description']}")
        if param_text:
            lines.append(f"  parameters: {param_text}; required: {', '.join(required) or 'none'}")
    return "\n".join(lines)


_START = "<tool_call:"
_END = ">"


def format_tool_call(text: str, call: ToolCall) -> str:
    payload = json.dumps({"name": call.name, "args": call.args}, ensure_ascii=False, separators=(",", ":"))
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    prefix = text.strip()
    return f"{prefix}\n{_START}{encoded}{_END}" if prefix else f"{_START}{encoded}{_END}"


def parse_response(reply: str) -> ParsedResponse:
    if not isinstance(reply, str):
        return ParsedResponse("", None, "Model reply was not text.")

    plain_variant = re.search(r"<tool_call>(.*?)</tool_call>", reply, re.DOTALL)
    if plain_variant:
        before = reply[:plain_variant.start()].strip()
        after = reply[plain_variant.end():].strip()
        payload = plain_variant.group(1).strip()
        try:
            try:
                obj = json.loads(payload)
            except Exception:
                payload_padded = payload + "=" * (-len(payload) % 4)
                raw = base64.b64decode(payload_padded.encode("ascii"), validate=True)
                obj = json.loads(raw.decode("utf-8"))
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not isinstance(obj.get("args"), dict):
                raise ValueError("Invalid call object")
            return ParsedResponse((before + "\n" + after).strip(), ToolCall(obj["name"], obj["args"]), None)
        except Exception as exc:
            return ParsedResponse(before, None, f"Tool call was malformed: {exc}")

    # Accept a close-tag variant emitted by some providers:
    # <tool_call:BASE64</tool_call>
    close_variant = re.search(r"<tool_call:([A-Za-z0-9+/=]+)</tool_call>", reply)
    if close_variant:
        before = reply[:close_variant.start()].strip()
        after = reply[close_variant.end():].strip()
        try:
            encoded = close_variant.group(1)
            encoded += "=" * (-len(encoded) % 4)
            raw = base64.b64decode(encoded.encode("ascii"), validate=True)
            obj = json.loads(raw.decode("utf-8"))
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not isinstance(obj.get("args"), dict):
                raise ValueError("Invalid call object")
            return ParsedResponse((before + "\n" + after).strip(), ToolCall(obj["name"], obj["args"]), None)
        except Exception as exc:
            return ParsedResponse(before, None, f"Tool call was malformed: {exc}")

    # Some DeepSeek-compatible providers emit DSML tool tags instead of
    # following the text protocol. Accept the common invoke/parameter form.
    dsml = re.search(r"invoke\s+name=[\"']([^\"']+)[\"']", reply)
    if dsml:
        name = dsml.group(1)
        args = {}
        parameter_blocks = re.findall(
            r"DSML[^>]*\s+parameter\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</[^>]*DSML[^>]*>",
            reply,
            re.DOTALL,
        )
        if not parameter_blocks:
            parameter_blocks = re.findall(
                r"parameter\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</[^>]*>",
                reply,
                re.DOTALL,
            )
        if parameter_blocks:
            for key, value in parameter_blocks:
                value = value.strip()
                try:
                    parsed_value = json.loads(value)
                except Exception:
                    parsed_value = value
                if key == "args" and isinstance(parsed_value, dict):
                    args = parsed_value
                else:
                    args[key] = parsed_value
        else:
            arg_match = re.search(r"\{.*\}", reply, re.DOTALL)
            if arg_match:
                try:
                    args = json.loads(arg_match.group(0))
                except Exception:
                    pass
        before = reply[:dsml.start()].strip()
        after = reply[dsml.end():]
        return ParsedResponse((before + "\n" + after).strip(), ToolCall(name, args), None)

    starts = [m.start() for m in re.finditer(re.escape(_START), reply)]
    if not starts:
        return ParsedResponse(reply.strip(), None, None)

    complete = []
    for start in starts:
        end = reply.find(_END, start + len(_START))
        if end >= 0:
            complete.append((start, end + 1, reply[start + len(_START):end]))
    if len(complete) > 1:
        return ParsedResponse(reply.strip(), None, "More than one tool call was returned.")
    if not complete:
        start = starts[0]
        before = reply[:start].strip()
        encoded = reply[start + len(_START):].strip()
        if encoded:
            try:
                encoded = encoded.rstrip()
                encoded += "=" * (-len(encoded) % 4)
                raw = base64.b64decode(encoded.encode("ascii"), validate=True)
                obj = json.loads(raw.decode("utf-8"))
                if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("args"), dict):
                    return ParsedResponse(before, ToolCall(obj["name"], obj["args"]), None)
            except Exception:
                pass
        return ParsedResponse(reply[:starts[0]].strip(), None, "Tool call was cut off.")

    start, end, encoded = complete[0]
    tail = reply[end:]
    if _START in tail:
        return ParsedResponse(reply.strip(), None, "More than one tool call was returned.")
    try:
        encoded_padded = encoded + "=" * (-len(encoded) % 4)
        raw = base64.b64decode(encoded_padded.encode("ascii"), validate=True)
        obj = json.loads(raw.decode("utf-8"))
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not isinstance(obj.get("args"), dict):
            raise ValueError("Invalid call object")
        call = ToolCall(obj["name"], obj["args"])
    except Exception as exc:
        return ParsedResponse(reply[:start].strip(), None, f"Tool call was malformed: {exc}")
    text = (reply[:start] + tail).strip()
    return ParsedResponse(text, call, None)
