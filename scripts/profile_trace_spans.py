#!/usr/bin/env python3
"""Summarize selected spans from an Akita Chrome/Perfetto trace."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PATTERN = (
    r"eor|root_extension|extension_opening|SparseExtensionOpening|"
    r"dense_extension_reduction|fused_fold|tensor_packed_sparse|"
    r"onehot_tensor_extension"
)


def trace_events(trace: Any) -> list[dict[str, Any]]:
    events = trace.get("traceEvents", trace) if isinstance(trace, dict) else trace
    if not isinstance(events, list):
        raise ValueError("trace must be a Chrome trace object or event list")
    return events


def complete_events(trace: Any) -> list[dict[str, Any]]:
    """Return complete-duration events from either `X` or `B`/`E` traces."""

    events = trace_events(trace)
    complete = [
        event
        for event in events
        if event.get("ph") == "X"
        and isinstance(event.get("name"), str)
        and isinstance(event.get("dur"), (int, float))
    ]
    stacks: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        phase = event.get("ph")
        name = event.get("name")
        timestamp = event.get("ts")
        if not isinstance(name, str) or not isinstance(timestamp, (int, float)):
            continue
        key = (event.get("pid"), event.get("tid"))
        if phase == "B":
            stacks[key].append(event)
        elif phase == "E" and stacks[key]:
            start = stacks[key].pop()
            duration = timestamp - float(start["ts"])
            if duration >= 0:
                complete.append(
                    {
                        "name": start["name"],
                        "ts": start["ts"],
                        "dur": duration,
                        "pid": start.get("pid"),
                        "tid": start.get("tid"),
                        "cat": start.get("cat"),
                    }
                )
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Chrome/Perfetto JSON trace")
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="case-insensitive regex for span names",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="maximum aggregate rows to print",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=20,
        help="maximum individual matching events to print",
    )
    args = parser.parse_args()

    matcher = re.compile(args.pattern, re.IGNORECASE)
    trace = json.loads(args.trace.read_text())
    matches = [event for event in complete_events(trace) if matcher.search(event["name"])]

    aggregates: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_us": 0.0, "max_us": 0.0}
    )
    for event in matches:
        duration = float(event["dur"])
        row = aggregates[event["name"]]
        row["count"] += 1
        row["total_us"] += duration
        row["max_us"] = max(row["max_us"], duration)

    print(f"trace: {args.trace}")
    print(f"matched_events: {len(matches)}")
    print(f"matched_inclusive_ms: {sum(event['dur'] for event in matches) / 1000.0:.3f}")
    print()
    print("aggregate spans:")
    print(f"{'total_ms':>12} {'max_ms':>12} {'count':>8}  name")
    for name, row in sorted(
        aggregates.items(), key=lambda item: item[1]["total_us"], reverse=True
    )[: args.limit]:
        print(
            f"{row['total_us'] / 1000.0:12.3f} "
            f"{row['max_us'] / 1000.0:12.3f} "
            f"{int(row['count']):8d}  {name}"
        )

    if args.events:
        print()
        print("largest matching events:")
        print(f"{'dur_ms':>12} {'ts_us':>16}  name")
        for event in sorted(matches, key=lambda item: item["dur"], reverse=True)[
            : args.events
        ]:
            print(
                f"{event['dur'] / 1000.0:12.3f} "
                f"{event.get('ts', 0):16.0f}  {event['name']}"
            )


if __name__ == "__main__":
    main()
