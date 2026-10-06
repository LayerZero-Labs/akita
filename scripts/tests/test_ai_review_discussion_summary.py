"""Visible discussion blockers and verification across review body versions."""

import base64
import json
import random
import unittest

from ai_review_support import EVENT, FakeGitHub, proposal, snapshot
from collect import discussions, previous_state
from common import MARKER, REPOSITORY, ReviewError
from publish import prepare_review, publish, reconcile, review_body, review_text, verify_review


class DiscussionSummaryTests(unittest.TestCase):
    def test_discussion_only_blockers_are_visible_without_duplicate_inline_findings(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        blockers = ["Comment 88: add the separate `ParentObservableKey.source_block_len` regression.",
                    "Comment 89: verify the `items[index]` boundary."]
        result["result"].update(findings=[], discussion_blockers=blockers)
        self.assertEqual(publish(gh, EVENT, value, result), "published")
        visible = gh.reviews[0]["body"].split(" -->", 1)[1]
        self.assertIn("Approval not recommended: unresolved feedback from existing discussion.\n\n"
                      "Unresolved feedback from existing discussion:\n- ", visible)
        for blocker in blockers:
            self.assertIn("- " + review_text(blocker), visible)
        self.assertIn("`ParentObservableKey.source_block_len`", visible)
        self.assertNotIn("Recommended for approval:", visible)
        self.assertEqual(gh.inline, [])
        self.assertEqual(gh.labels, {"documentation", "ai-reviewed"})
        self.assertEqual(publish(gh, EVENT, value, result), "already-published")
        self.assertEqual(len(gh.reviews), 1)
        state = previous_state(discussions(gh, 7), 7)
        self.assertEqual(state["discussion_blockers"], blockers)
        self.assertEqual(state["body_version"], 2)

    def test_blockers_are_escaped_and_cannot_inject_markdown_html_or_mentions(self):
        value = snapshot()
        result = proposal(value)
        result["result"]["discussion_blockers"] = [
            "Check `items[index]` and ``a`b``.\n\n![image](https://evil.test) <img src=x> @maintainer"]
        visible = prepare_review(value, result)["body"].split(" -->", 1)[1]
        self.assertIn("`items[index]`", visible)
        self.assertIn("``a`b``", visible)
        self.assertNotIn("![image]", visible)
        self.assertNotIn("<img", visible)
        self.assertNotIn("@maintainer", visible)
        self.assertNotIn("\n\n![", visible)

    def test_absent_blockers_do_not_add_a_false_warning_to_clean_or_nit_reviews(self):
        for nit in (False, True):
            with self.subTest(nit=nit):
                value = snapshot()
                result = proposal(value)
                if nit:
                    result["result"]["findings"][0]["priority"] = "nit"
                else:
                    result["result"]["findings"] = []
                body = prepare_review(value, result)["body"]
                self.assertIn("Recommended for approval:", body)
                self.assertNotIn("Approval not recommended:", body)

    def test_old_reviews_still_verify_reconcile_and_retry_without_rewriting(self):
        for blocked in (False, True):
            with self.subTest(blocked=blocked):
                gh = FakeGitHub()
                value = snapshot(gh)
                result = proposal(value)
                result["result"].update(findings=[], discussion_blockers=["Existing request"] if blocked else [])
                result["result"]["usefulness"]["assessment"] = "Useful change"
                payload = prepare_review(value, result)
                state = previous_state([{"id": 99, "own": True, "body": payload["body"]}], 7)
                del state["body_version"]
                # Literal old-format summary, independent of the current renderer.
                old_body = MARKER + "1 " + base64.b64encode(json.dumps(state).encode()).decode() + " -->"
                old_body += "\n\nUsefulness: benefit supported. Useful change"
                if not blocked:
                    old_body += "\n\nRecommended for approval: no unresolved findings in this automated review."
                payload["body"] = old_body
                gh.get("pulls/7/reviews", payload)
                self.assertEqual(verify_review(gh, 7, 99, old_body), state)
                gh.labels.add("ai-approved")
                reconcile(gh, {"repository": {"full_name": REPOSITORY}, "action": "edited",
                               "pull_request": gh.pr})
                self.assertEqual("ai-approved" in gh.labels, not blocked)
                self.assertIn("ai-reviewed", gh.labels)
                self.assertEqual(publish(gh, EVENT, value, result), "already-published")
                self.assertEqual(len(gh.reviews), 1)
                self.assertEqual(gh.reviews[0]["body"], old_body)

    def test_visible_summary_tampering_is_rejected(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        result["result"].update(findings=[], discussion_blockers=["Existing request"])
        publish(gh, EVENT, value, result)
        review = gh.reviews[0]
        review["body"] = review["body"].split("\n\nApproval not recommended:", 1)[0]
        with self.assertRaisesRegex(ReviewError, "could not be verified"):
            verify_review(gh, 7, review["id"], review["body"])

    def test_unsupported_body_versions_and_oversized_summaries_fail_closed(self):
        value = snapshot()
        payload = prepare_review(value, proposal(value))
        state = previous_state([{"id": 99, "own": True, "body": payload["body"]}], 7)
        for version in (0, 3, "2", None, True, 2.0):
            with self.subTest(version=version), self.assertRaisesRegex(ReviewError, "body version"):
                review_body({**state, "body_version": version})
        result = proposal(value)
        # Non-repeating text, so the compressed state does not shrink it away.
        rng = random.Random(5)
        result["result"]["discussion_blockers"] = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz ")
                                                           for _ in range(1900)) for _ in range(30)]
        with self.assertRaisesRegex(ReviewError, "comment size limit"):
            prepare_review(value, result)


if __name__ == "__main__":
    unittest.main()
