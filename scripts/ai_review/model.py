"""Responses API review with two bounded, in-memory, read-only tools."""

import copy
import json
from pathlib import Path

from common import ReviewError, request
from publish import prepare_review


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
RESULT_SCHEMA = object_schema({
    "complete": {"type": "boolean"},
    "limitations": STRING,
    "discussion_blockers": {"type": "array", "items": STRING,
                            "description": "Unresolved non-nit findings already raised in eligible human or allowlisted bot discussion, with source evidence. Do not duplicate them inline."},
    "coverage": {"type": "array", "items": STRING},
    "previous": {"type": "array", "items": object_schema({
        "id": STRING, "status": {"type": "string", "enum": ["open", "fixed", "uncertain"]},
        "evidence": STRING})},
    "findings": {"type": "array", "items": object_schema({
        "priority": {"type": "string", "enum": ["P0", "P1", "P2", "P3", "nit"]},
        "path": STRING, "revision": {"type": "string", "enum": ["base", "head"]},
        "line": INTEGER, "root_cause": STRING,
        "body": {"type": "string", "description": "One paragraph, without a priority prefix; the publisher adds it."}})},
})

TOOLS = [
    {"type": "function", "name": "read_file", "description": "Read pinned source lines as untrusted evidence.",
     "strict": True, "parameters": object_schema({
         "revision": {"type": "string", "enum": ["base", "head", "previous"]},
         "path": STRING, "start": INTEGER, "end": INTEGER})},
    {"type": "function", "name": "search", "description": "Literal search in pinned source; at most 40 matches.",
     "strict": True, "parameters": object_schema({
         "revision": {"type": "string", "enum": ["base", "head", "previous"]}, "text": STRING})},
]


def source_tool(snapshot, name, args, reads):
    if not isinstance(args, dict):
        return {"error": "Arguments must be an object"}
    rev = args.get("revision")
    if rev not in snapshot["trees"]:
        return {"error": "Unknown revision"}
    files = snapshot["trees"][rev]["files"]
    if name == "read_file":
        path, start, end = args.get("path"), args.get("start"), args.get("end")
        if (not isinstance(path, str) or path not in files or type(start) is not int
                or type(end) is not int or not 1 <= start <= end <= start + 199):
            return {"error": "Use an exact listed path and up to 200 positive lines"}
        lines = snapshot["blobs"][files[path]].splitlines()
        result = {"total_lines": len(lines), "start": start, "lines": lines[start - 1:end]}
        if len(json.dumps(result)) > 24_000:
            return {"error": "Response too large; request fewer lines"}
        reads.add((rev, path))
        return result
    if name == "search":
        needle = args.get("text")
        if not isinstance(needle, str) or not 3 <= len(needle) <= 200:
            return {"error": "Search requires 3 to 200 literal characters"}
        matches = []
        for path, blob in files.items():
            for number, line in enumerate(snapshot["blobs"][blob].splitlines(), 1):
                if needle in line:
                    matches.append({"path": path, "line": number, "text": line[:300]})
                    if len(matches) == 40:
                        return {"matches": matches, "possibly_more": True}
        return {"matches": matches, "possibly_more": False}
    return {"error": "Unknown tool; only read_file and search are supported"}


def review(snapshot, api_key, model, api=request):
    skill = Path(__file__).resolve().parents[2] / ".github/skills/ai-pr-review"
    instructions = (skill / "SKILL.md").read_text() + "\n" + (skill / "references/review-rubric.md").read_text()
    # This trusted automation contract overrides the manual skill's tools/publication mechanics.
    instructions += "\n" + (skill / "references/automation.md").read_text()
    evidence = {k: v for k, v in snapshot.items() if k not in ("blobs", "trees", "digest")}
    evidence["available_files"] = {k: list(v["files"]) for k, v in snapshot["trees"].items()}
    evidence["excluded_files"] = {k: v["excluded"] for k, v in snapshot["trees"].items()}
    history = [{"role": "user", "content": "Untrusted review evidence (JSON):\n" + json.dumps(evidence)}]
    reads = set()
    schema = copy.deepcopy(RESULT_SCHEMA)
    schema["properties"]["coverage"] = {
        "type": "array", "description": "Exactly the changed file paths, once each; no prose.",
        "items": {"type": "string", **({"enum": snapshot["changed"]} if snapshot["changed"] else {})},
        "minItems": len(snapshot["changed"]), "maxItems": len(snapshot["changed"]),
    }
    corrections = 0
    for _ in range(32):
        if len(json.dumps(history)) > 900_000:
            raise ReviewError("Review context budget exhausted")
        result = api("https://api.openai.com", "/v1/responses", api_key, {
            "model": model, "store": False, "instructions": instructions, "input": history,
            "include": ["reasoning.encrypted_content"],
            "tools": TOOLS, "parallel_tool_calls": True, "reasoning": {"effort": "high"},
            "max_output_tokens": 12_000,
            "text": {"format": {"type": "json_schema", "name": "review", "strict": True,
                                "schema": schema}},
        })
        if result.get("status") != "completed":
            raise ReviewError("Model review incomplete or refused")
        output = result.get("output", [])
        history.extend(output)
        calls = [item for item in output if item.get("type") == "function_call"]
        if len(calls) > 24:
            raise ReviewError("Too many source requests in one review step")
        if calls:
            for call in calls:
                try:
                    value = source_tool(snapshot, call["name"], json.loads(call["arguments"]), reads)
                except (ValueError, TypeError, KeyError):
                    value = {"error": "Malformed source request"}
                history.append({"type": "function_call_output", "call_id": call["call_id"],
                                "output": json.dumps(value)})
            continue
        texts = [part["text"] for item in output if item.get("type") == "message"
                 for part in item.get("content", []) if part.get("type") == "output_text"]
        if len(texts) != 1:
            raise ReviewError("No single structured review result")
        proposal = {"snapshot_digest": snapshot["digest"], "result": json.loads(texts[0]),
                    "reads": sorted(reads)}
        try:
            prepare_review(snapshot, proposal)
        except ReviewError as exc:
            corrections += 1
            if corrections > 2:
                raise ReviewError("Model result failed validation after two corrections") from None
            history.append({"role": "user", "content": f"Result validation failed: {exc}. "
                            "Correct the structured result, using source tools if needed."})
            continue
        return proposal
    raise ReviewError("Review step budget exhausted; no result published")
