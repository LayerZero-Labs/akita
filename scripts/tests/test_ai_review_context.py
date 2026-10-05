"""Context budget and count-only model diagnostics; no live API calls."""

import copy
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

from ai_review_support import proposal, seal, snapshot
from common import ReviewError
from model import review, usage_counts


class ContextTests(unittest.TestCase):
    def test_accumulated_history_accepts_new_boundary_and_stops_before_next_call_above_it(self):
        value = snapshot()
        padding = ""
        calls = []

        def api(origin, path, token, payload):
            calls.append(copy.deepcopy(payload))
            if len(calls) == 1:
                return {"status": "completed", "output": [
                    {"type": "reasoning", "encrypted_content": padding},
                    {"type": "function_call", "name": "read_file", "call_id": "read",
                     "arguments": json.dumps({"revision": "head", "path": "src/a.py", "start": 1, "end": 2})}]}
            self.assertEqual(payload["input"][1]["encrypted_content"], padding)
            return {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "output_text", "text": json.dumps(proposal(value)["result"])}]}]}

        # Measure fixed fixture overhead, then exercise the actual production cap.
        with redirect_stdout(io.StringIO()):
            review(value, "test-key", "test-model", api)
        overhead = len(json.dumps(calls[1]["input"]))
        for size in (919_510, 2_000_000, 2_000_001):
            with self.subTest(size=size):
                calls.clear()
                padding = "r" * (size - overhead)
                logs = io.StringIO()
                with redirect_stdout(logs):
                    if size > 2_000_000:
                        with self.assertRaisesRegex(ReviewError, "2000001 > 2000000 characters"):
                            review(value, "test-key", "test-model", api)
                        self.assertEqual(len(calls), 1)
                    else:
                        result = review(value, "test-key", "test-model", api)
                        self.assertTrue(result["result"]["complete"])
                        self.assertEqual(len(calls), 2)
                        self.assertEqual(len(json.dumps(calls[1]["input"])), size)
                context = [json.loads(line.split(": ", 1)[1]) for line in logs.getvalue().splitlines()
                           if line.startswith("AI review context:")]
                self.assertEqual(context[-1]["history_chars"], size)
                self.assertEqual(context[-1]["turn"], 2)

    def test_oversized_initial_history_stops_before_any_api_call(self):
        value = snapshot()
        value["description"] = "x" * 2_000_000
        seal(value)
        with redirect_stdout(io.StringIO()), patch("model.request") as api:
            with self.assertRaisesRegex(ReviewError, "Local review context budget exhausted"):
                review(value, "test-key", "test-model", api)
            api.assert_not_called()

    def test_diagnostics_are_numeric_preserve_history_and_report_usage_and_elapsed_time(self):
        value = snapshot()
        value["description"] = "PRIVATE-DESCRIPTION-CANARY"
        value["blobs"]["blob"] += "# PRIVATE-SOURCE-CANARY\n"
        seal(value)
        encrypted = "PRIVATE-REASONING-CANARY\n\"é"
        calls = []

        def api(origin, path, token, payload):
            calls.append(copy.deepcopy(payload))
            if len(calls) == 1:
                return {"status": "completed", "usage": {
                    "input_tokens": 100, "input_tokens_details": {"cached_tokens": 40},
                    "output_tokens": 30, "output_tokens_details": {"reasoning_tokens": 20},
                    "unexpected": "PRIVATE-USAGE-CANARY"}, "output": [
                        {"type": "reasoning", "encrypted_content": encrypted},
                        {"type": "function_call", "name": "read_file", "call_id": "PRIVATE-ID-CANARY",
                         "arguments": json.dumps({"revision": "head", "path": "src/a.py", "start": 1, "end": 3})}]}
            self.assertEqual(payload["input"][1]["encrypted_content"], encrypted)
            self.assertIn("PRIVATE-SOURCE-CANARY", payload["input"][-1]["output"])
            return {"status": "completed", "usage": {
                "input_tokens": "PRIVATE-USAGE-CANARY", "input_tokens_details": "PRIVATE-USAGE-CANARY",
                "output_tokens": -1, "output_tokens_details": {"reasoning_tokens": True}},
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": json.dumps(proposal(value)["result"])}]}]}

        logs = io.StringIO()
        with redirect_stdout(logs), patch("model.time.monotonic", side_effect=[100, 100, 101.25, 101.5, 103]), \
                patch("socket.socket", side_effect=AssertionError("Unexpected network access")):
            result = review(value, "PRIVATE-KEY-CANARY", "test-model", api)
        self.assertTrue(result["result"]["complete"])
        self.assertNotIn("PRIVATE-", logs.getvalue())
        entries = [json.loads(line.split(": ", 1)[1]) for line in logs.getvalue().splitlines()]
        self.assertEqual(len(entries), 4)
        for entry in entries:
            self.assertTrue(all(type(v) in (int, float) or v is None for v in entry.values()))
        first_context, first_usage, second_context, second_usage = entries
        self.assertEqual(first_context["encrypted_reasoning_chars"], 0)
        self.assertEqual(first_context["tool_results_chars"], 0)
        self.assertEqual(second_context["encrypted_reasoning_chars"], len(json.dumps(encrypted)) - 2)
        self.assertGreater(second_context["tool_results_chars"], 0)
        self.assertEqual(second_context["history_chars"], len(json.dumps(calls[1]["input"])))
        self.assertEqual(second_context["history_chars"], sum(second_context[k] for k in (
            "evidence_chars", "tool_results_chars", "encrypted_reasoning_chars", "other_chars")))
        self.assertEqual(second_context["elapsed_seconds"], 1.5)
        self.assertEqual(first_usage, {"turn": 1, "elapsed_seconds": 1.25, "input_tokens": 100,
                                      "cached_input_tokens": 40, "output_tokens": 30, "reasoning_tokens": 20})
        self.assertEqual(second_usage, {"turn": 2, "elapsed_seconds": 3, "input_tokens": None,
                                       "cached_input_tokens": None, "output_tokens": None, "reasoning_tokens": None})

    def test_missing_or_malformed_usage_cannot_inject_logs_or_break_diagnostics(self):
        for usage in (None, {}, "payload", [], {"input_tokens": False, "output_tokens": 1.5,
                                              "input_tokens_details": None, "output_tokens_details": []}):
            with self.subTest(usage=usage):
                self.assertEqual(usage_counts(usage), {"input_tokens": None, "cached_input_tokens": None,
                                                     "output_tokens": None, "reasoning_tokens": None})
        self.assertEqual(usage_counts({"input_tokens": 0})["input_tokens"], 0)

    def test_incomplete_response_still_reports_usage(self):
        logs = io.StringIO()
        with redirect_stdout(logs), self.assertRaisesRegex(ReviewError, "incomplete or refused"):
            review(snapshot(), "test-key", "test-model", lambda *args: {
                "status": "incomplete", "usage": {"output_tokens": 12_000}})
        self.assertIn('"output_tokens": 12000', logs.getvalue())


if __name__ == "__main__":
    unittest.main()
