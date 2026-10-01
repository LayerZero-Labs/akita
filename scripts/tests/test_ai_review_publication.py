"""Review validation, rendering, coordinates, and finding history."""

import unittest

from ai_review_support import (AUTHOR, EVENT, FakeGitHub, proposal, seal, snapshot)
from collect import diff_lines, discussions, previous_state
from common import ReviewError
from publish import publish, prepare_review, review_text, validate


class PublicationTests(unittest.TestCase):
    def test_clean_and_nit_only_reviews_recommend_approval_once(self):
        for priority in (None, "nit", "P0", "P1", "P2", "P3"):
            with self.subTest(priority=priority):
                gh = FakeGitHub()
                value = snapshot(gh)
                result = proposal(value)
                if priority is None:
                    result["result"]["findings"] = []
                else:
                    result["result"]["findings"][0]["priority"] = priority
                self.assertEqual(publish(gh, EVENT, value, result), "published")
                payload = gh.writes[0][1]
                self.assertEqual("Recommended for approval:" in payload["body"], priority in (None, "nit"))
                self.assertEqual(payload["event"], "COMMENT")
                self.assertEqual(len(payload["comments"]), 0 if priority is None else 1)
                if priority == "nit":
                    self.assertIn("only optional nits remain", payload["body"])
                state = previous_state(discussions(gh, 7), 7)
                self.assertEqual(state["request"], value["request"])
                self.assertEqual(publish(gh, EVENT, value, result), "already-published")
                self.assertEqual(len(gh.writes), 2)
                self.assertIn("ai-reviewed", gh.labels)
                self.assertEqual("ai-approved" in gh.labels, priority in (None, "nit"))
                self.assertIn("documentation", gh.labels)

    def test_prior_findings_control_approval_even_when_no_new_findings_exist(self):
        for priority, status, recommend in (("P1", "open", False), ("P1", "fixed", True),
                                            ("P1", "uncertain", False), ("nit", "open", True),
                                            ("nit", "uncertain", False)):
            with self.subTest(priority=priority, status=status):
                value = snapshot()
                initial = proposal(value)
                initial["result"]["findings"][0]["priority"] = priority
                body = prepare_review(value, initial)["body"]
                value["prior"] = previous_state([{"id": 99, "own": True, "body": body}], 7)
                seal(value)
                result = proposal(value)
                result["result"]["findings"] = []
                result["result"]["previous"] = [{"id": value["prior"]["findings"][0]["id"],
                    "status": status, "evidence": "Checked current src/a.py:2 against the empty-input contract."}]
                payload = prepare_review(value, result)
                self.assertEqual("Recommended for approval:" in payload["body"], recommend)
                self.assertEqual(payload["comments"], [])

    def test_incomplete_reviews_and_unresolved_human_findings_do_not_recommend_approval(self):
        for complete, blockers in ((False, []), (True, ["Human comment 88: src/a.py:2 still fails on empty input."])):
            for nit in (False, True):
                with self.subTest(complete=complete, blockers=blockers, nit=nit):
                    value = snapshot()
                    result = proposal(value)
                    result["result"].update(complete=complete, discussion_blockers=blockers)
                    if nit:
                        result["result"]["findings"][0]["priority"] = "nit"
                    else:
                        result["result"]["findings"] = []
                    self.assertNotIn("Recommended for approval:", prepare_review(value, result)["body"])

    def test_inline_anchors_include_correct_sides_and_reject_off_diff_lines(self):
        patch_text = "--- a/x\n+++ b/x\n@@ -10,2 +10,2 @@\n same\n-old\n+new\n"
        self.assertEqual(diff_lines(patch_text), {"base": [10, 11], "head": [10, 11]})
        self.assertEqual(diff_lines("@@ -0,0 +1 @@\n+new\n"), {"base": [], "head": [1]})
        self.assertEqual(diff_lines("@@ -1 +0,0 @@\n-old\n"), {"base": [1], "head": []})
        value = snapshot()
        value["anchors"]["src/a.py"]["head"] = [1]
        seal(value)
        with self.assertRaises(ReviewError):
            prepare_review(value, proposal(value))

    def test_removed_lines_are_published_on_left_side(self):
        value = snapshot()
        result = proposal(value)
        result["reads"] = [["base", "src/a.py"]]
        result["result"]["findings"][0]["revision"] = "base"
        self.assertEqual(prepare_review(value, result)["comments"][0]["side"], "LEFT")

    def test_publishes_and_duplicate_event_is_noop(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        self.assertEqual(publish(gh, EVENT, value, result), "published")
        self.assertEqual(publish(gh, EVENT, value, result), "already-published")
        self.assertEqual(len(gh.writes), 2)
        self.assertEqual(gh.writes[0][0], "pulls/7/reviews")
        self.assertEqual(gh.writes[0][1]["comments"][0]["side"], "RIGHT")
        self.assertNotIn("###", gh.writes[0][1]["body"])

    def test_stale_head_base_discussion_description_and_revoked_access_stop_writes(self):
        for change in ("head", "base", "discussion", "description", "permission"):
            with self.subTest(change=change):
                gh = FakeGitHub()
                value = snapshot(gh)
                if change in ("head", "base"):
                    gh.pr[change]["sha"] = "c" * 40
                elif change == "discussion":
                    gh.comments.append({"id": 88, "body": "New evidence", "user": AUTHOR})
                elif change == "description":
                    gh.pr["body"] = "Updated scope"
                else:
                    gh.permission = "read"
                with self.assertRaises(ReviewError):
                    publish(gh, EVENT, value, proposal(value))
                self.assertFalse(gh.writes)

    def test_bad_output_never_becomes_a_github_operation(self):
        for field, bad in (("path", "https://evil.test"), ("path", "../../key"), ("line", 999),
                           ("line", True), ("priority", "approve"), ("revision", "previous")):
            value = snapshot()
            result = proposal(value)
            result["result"]["findings"][0][field] = bad
            with self.assertRaises(ReviewError):
                prepare_review(value, result)

    def test_digest_and_coverage_and_reads_are_required(self):
        for change in ("digest", "coverage", "reads"):
            value = snapshot()
            result = proposal(value)
            if change == "digest":
                value["description"] = "tamper"
            elif change == "coverage":
                result["result"]["coverage"] = []
            else:
                result["reads"] = []
            with self.assertRaises(ReviewError):
                validate(value, result)

    def test_model_text_cannot_embed_images_html_or_mentions(self):
        value = snapshot()
        result = proposal(value)
        result["result"]["findings"][0]["body"] = "doesn't ![click](https://evil.test) <img src=x> @maintainer"
        body = prepare_review(value, result)["comments"][0]["body"]
        self.assertNotIn("![click]", body)
        self.assertNotIn("<img", body)
        self.assertNotIn("@maintainer", body)
        self.assertIn("doesn't", body)

    def test_inline_code_renders_without_escaping_its_contents(self):
        self.assertEqual(review_text("use `items[index]` when `index < len(items)`"),
                         "use `items[index]` when `index < len(items)`")
        self.assertEqual(review_text("use ``a`b``"), "use ``a`b``")
        self.assertEqual(review_text("`<script>` and `@maintainer`"), "`<script>` and `@maintainer`")
        self.assertEqual(review_text("unfinished `code"), "unfinished \\`code")
        hostile = review_text("`safe` ![image](https://evil.test) <img src=x> @maintainer")
        self.assertTrue(hostile.startswith("`safe` "))
        self.assertNotIn("![image]", hostile)
        self.assertNotIn("<img", hostile)
        self.assertNotIn("@maintainer", hostile)

    def test_every_previous_finding_must_be_reassessed_and_can_regress(self):
        value = snapshot()
        body = prepare_review(value, proposal(value))["body"]
        state = previous_state([{"id": 99, "own": True, "body": body}], 7)
        value["prior"] = state
        seal(value)
        result = proposal(value)
        result["result"]["findings"] = []
        with self.assertRaises(ReviewError):
            validate(value, result)

        finding = state["findings"][0]
        for status in ("fixed", "open", "uncertain"):
            result["result"]["previous"] = [{"id": finding["id"], "status": status,
                                              "evidence": "src/a.py:2 still indexes an empty list."}]
            self.assertEqual(validate(value, result)[0]["status"], status)
        result["result"]["findings"] = proposal(snapshot())["result"]["findings"]
        with self.assertRaises(ReviewError):
            validate(value, result)

    def test_state_order_uses_time_across_comment_and_review_id_spaces(self):
        value = snapshot()
        older = prepare_review(value, proposal(value))["body"]
        value["request"] = 13
        seal(value)
        newer = prepare_review(value, proposal(value))["body"]
        state = previous_state([
            {"id": 9_000_000, "own": True, "body": older, "created_at": "2026-09-30T01:00:00Z"},
            {"id": 100, "own": True, "body": newer, "created_at": "2026-09-30T02:00:00Z"},
        ], 7)
        self.assertEqual(state["request"], 13)


if __name__ == "__main__":
    unittest.main()
