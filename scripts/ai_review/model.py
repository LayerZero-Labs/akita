"""Responses API review with two bounded, in-memory, read-only tools."""

import copy
import json
from pathlib import Path
import time

from common import ReviewError, request
from publish import prepare_review


MAX_HISTORY_CHARS = 2_000_000
# Measured reviews take about 26 turns; this leaves room within MAX_HISTORY_CHARS.
MAX_TURNS = 48


def history_sizes(history):
    """Partition serialized history size without exposing or modifying evidence."""
    sizes = {"history_chars": len(json.dumps(history)), "evidence_chars": 0,
             "tool_results_chars": 0, "encrypted_reasoning_chars": 0}
    for item in history:
        if item.get("role") == "user":
            sizes["evidence_chars"] += len(json.dumps(item))
        elif item.get("type") == "function_call_output":
            sizes["tool_results_chars"] += len(json.dumps(item))
        elif item.get("type") == "reasoning" and isinstance(item.get("encrypted_content"), str):
            # Count escaped payload characters; field names and quotes are overhead.
            sizes["encrypted_reasoning_chars"] += len(json.dumps(item["encrypted_content"])) - 2
    sizes["other_chars"] = sizes["history_chars"] - sum(
        sizes[key] for key in ("evidence_chars", "tool_results_chars", "encrypted_reasoning_chars"))
    return sizes


def usage_counts(usage):
    """Log only nonnegative integer counters, never raw remote response fields."""
    counts = {}
    for name, path in (("input_tokens", ("input_tokens",)),
                       ("cached_input_tokens", ("input_tokens_details", "cached_tokens")),
                       ("output_tokens", ("output_tokens",)),
                       ("reasoning_tokens", ("output_tokens_details", "reasoning_tokens"))):
        value = usage
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        counts[name] = value if type(value) is int and value >= 0 else None
    return counts


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
RESULT_SCHEMA = object_schema({
    "complete": {"type": "boolean"},
    "limitations": STRING,
    "usefulness": object_schema({
        "motivation": {"type": "string", "enum": ["provided", "missing"]},
        "verdict": {"type": "string", "enum": ["beneficial", "unclear", "not_beneficial"]},
        "assessment": {"type": "string", "description": "Evaluate the PR's net repository benefit, separately from correctness or description alignment. Cite concrete evidence, costs, and any missing justification."},
    }),
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
    started = time.monotonic()
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
    for turn in range(1, MAX_TURNS + 1):
        sizes = history_sizes(history)
        print("AI review context: " + json.dumps({"turn": turn,
              "elapsed_seconds": round(time.monotonic() - started, 3), **sizes}), flush=True)
        if sizes["history_chars"] > MAX_HISTORY_CHARS:
            raise ReviewError(f"Local review context budget exhausted: {sizes['history_chars']} > "
                              f"{MAX_HISTORY_CHARS} characters")
        result = api("https://api.openai.com", "/v1/responses", api_key, {
            "model": model, "store": False, "instructions": instructions, "input": history,
            "include": ["reasoning.encrypted_content"],
            "tools": TOOLS, "parallel_tool_calls": True, "reasoning": {"effort": "high"},
            "max_output_tokens": 12_000,
            "text": {"format": {"type": "json_schema", "name": "review", "strict": True,
                                "schema": schema}},
        })
        print("AI review usage: " + json.dumps({"turn": turn,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              **usage_counts(result.get("usage"))}), flush=True)
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
    raise ReviewError(f"Review step budget exhausted ({MAX_TURNS} model turns); no result published")
