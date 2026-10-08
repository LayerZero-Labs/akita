"""Usefulness assessment, motivation, and approval policy."""

import copy
import base64
import json
import unittest

from ai_review_support import (COMMAND, EVENT, FakeGitHub, HEAD, PR, proposal, seal, snapshot)
from collect import discussions, previous_state
from common import MARKER, REPOSITORY, ReviewError
from publish import publish, prepare_review, reconcile, review_text


class UsefulnessTests(unittest.TestCase):
    def test_usefulness_controls_visible_assessment_and_approval_without_inline_findings(self):
        for motivation in ("provided", "missing"):
            for verdict in ("beneficial", "unclear", "not_beneficial"):
                with self.subTest(motivation=motivation, verdict=verdict):
                    gh = FakeGitHub()
                    gh.labels.add("ai-approved")
                    value = snapshot(gh)
                    result = proposal(value)
                    result["result"]["findings"] = []
                    result["result"]["usefulness"].update(motivation=motivation, verdict=verdict)
                    publish(gh, EVENT, value, result)
                    body = gh.reviews[0]["body"]
                    self.assertIn("Usefulness:", body)
                    self.assertIn(review_text(result["result"]["usefulness"]["assessment"]), body)
                    self.assertEqual("Motivation is missing" in body, motivation == "missing")
                    self.assertEqual("Recommended for approval:" in body, verdict == "beneficial")
                    self.assertEqual("ai-approved" in gh.labels, verdict == "beneficial")
                    self.assertIn("ai-reviewed", gh.labels)
                    self.assertFalse(gh.inline)
                    self.assertEqual(previous_state(discussions(gh, 7), 7)["usefulness"],
                                     result["result"]["usefulness"])
                    writes = len(gh.writes)
                    publish(gh, EVENT, value, result)
                    self.assertEqual(len(gh.writes), writes)

    def test_missing_or_invalid_usefulness_cannot_be_published(self):
        for defect in ("omitted", "null", "extra", "verdict", "motivation", "blank", "oversized"):
            with self.subTest(defect=defect):
                gh = FakeGitHub()
                value = snapshot(gh)
                result = proposal(value)
                usefulness = result["result"]["usefulness"]
                if defect == "omitted":
                    del result["result"]["usefulness"]
                elif defect == "null":
                    result["result"]["usefulness"] = None
                elif defect == "extra":
                    usefulness["approve"] = True
                elif defect in ("verdict", "motivation"):
                    usefulness[defect] = "approve now"
                else:
                    usefulness["assessment"] = " " if defect == "blank" else "x" * 3001
                with self.assertRaises(ReviewError):
                    publish(gh, EVENT, value, result)
                self.assertFalse(gh.writes)

    def test_usefulness_text_is_escaped_and_preserves_inline_code(self):
        value = snapshot()
        result = proposal(value)
        result["result"]["usefulness"]["assessment"] = "Use `items[0]`; ![image](https://evil.test) <img src=x> @maintainer"
        body = prepare_review(value, result)["body"].split(" -->", 1)[1]
        self.assertIn("`items[0]`", body)
        self.assertNotIn("![image]", body)
        self.assertNotIn("<img", body)
        self.assertNotIn("@maintainer", body)

    def test_repeat_review_reassesses_usefulness_after_description_update(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        result["result"]["findings"] = []
        result["result"]["usefulness"].update(motivation="missing", verdict="unclear",
                                             assessment="Explain why the API break is necessary for the longer-term goal.")
        publish(gh, EVENT, value, result)
        self.assertNotIn("ai-approved", gh.labels)
        gh.pr["body"] = "## Motivation\nRemove unsafe API for the verifier hardening goal; migrate the only caller."
        event = copy.deepcopy(EVENT)
        event["comment"]["id"] = 13
        gh.comments.append({**COMMAND, "id": 13})
        fresh = snapshot(gh)
        fresh.update(request=13, description=gh.pr["body"], prior=previous_state(discussions(gh, 7), 7))
        seal(fresh)
        updated = proposal(fresh)
        updated["result"]["findings"] = []
        updated["result"]["usefulness"]["assessment"] = "The unsafe API removal serves verifier hardening and the only caller migrates."
        publish(gh, event, fresh, updated)
        self.assertIn("ai-approved", gh.labels)
        self.assertNotIn("Motivation is missing", gh.reviews[-1]["body"])
        self.assertEqual(previous_state(discussions(gh, 7), 7)["usefulness"], updated["result"]["usefulness"])

    def test_older_review_without_usefulness_remains_history_but_cannot_grant_approval(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        result["result"]["findings"] = []
        publish(gh, EVENT, value, result)
        state = previous_state(discussions(gh, 7), 7)
        del state["usefulness"]
        # A first-release (v1) state without usefulness, as the original workflow wrote it.
        del state["body_version"]
        gh.reviews[0]["body"] = (MARKER + "1 " + base64.b64encode(json.dumps(state).encode()).decode()
                                  + " -->\n\nRecommended for approval: no unresolved findings in this automated review.")
        self.assertEqual(previous_state(discussions(gh, 7), 7)["head"], HEAD)
        with self.assertRaisesRegex(ReviewError, "approval cleared"):
            reconcile(gh, {"repository": {"full_name": REPOSITORY}, "action": "edited", "pull_request": PR})
        self.assertNotIn("ai-approved", gh.labels)


if __name__ == "__main__":
    unittest.main()
