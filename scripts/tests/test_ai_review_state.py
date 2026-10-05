"""Review-state encoding, body versions, and a visible reason for every withheld approval."""

import base64
import copy
import itertools
import json
import random
import unittest
import zlib
from unittest.mock import patch

from ai_review_support import EVENT, PR, FakeGitHub, proposal, seal, snapshot
from collect import discussions, previous_state
import common
from common import MARKER, REPOSITORY, ReviewError, decode_state, encode_state
from publish import approval_blockers, prepare_review, publish, reconcile, summary

# The visible summary the first release (v1, no body_version) published for
# carried_over(); frozen from the publisher on main before body versions existed.
V1_SUMMARY = ("\n\nUsefulness: benefit supported. The src/a\\.py empty\\-batch fix restores the documented input "
              "contract without adding an abstraction\\.")
UPDATE = {"repository": {"full_name": REPOSITORY}, "action": "edited", "pull_request": PR}


def with_prior(value, findings):
    """Give the snapshot an earlier review state holding these (priority, root cause) findings."""
    value["prior"] = {"head": "c" * 40, "request": 11, "findings": [
        {"priority": priority, "path": "src/a.py", "revision": "head", "line": 2, "root_cause": cause,
         "body": "Earlier comment.", "id": f"{index:016x}", "status": "open", "evidence": "", "commit": "c" * 40}
        for index, (priority, cause) in enumerate(findings)]}
    seal(value)
    return value


def carried_over(value, status="open"):
    """A repeat review where one earlier P1 finding is reassessed."""
    with_prior(value, [("P1", "empty batch indexes zero")])
    result = proposal(value)
    result["result"]["findings"] = []
    result["result"]["previous"] = [{"id": "0" * 16, "status": status,
                                     "evidence": "src/a.py:2 still indexes `items[0]`."}]
    return result


def visible(body):
    return body.split(" -->", 1)[1]


def publish_legacy(gh):
    """Publish a review exactly as the first release did: v1 marker, no body_version."""
    value = snapshot(gh)
    result = carried_over(value)
    payload = prepare_review(value, result)
    state = decode_state(payload["body"])[1]
    del state["body_version"]
    body = f"{MARKER}1 {base64.b64encode(json.dumps(state).encode()).decode()} -->" + V1_SUMMARY
    gh.reviews.append({"id": 99, "user": {"login": "github-actions[bot]", "type": "Bot", "id": 1}, "body": body,
                       "submitted_at": "earlier", "state": "COMMENTED", "commit_id": payload["commit_id"]})
    return value, result, body


class VisibleBlockerTests(unittest.TestCase):
    def test_carried_over_blockers_are_named_in_the_visible_summary(self):
        for priority, status, blocking in (("P1", "open", True), ("P2", "uncertain", True),
                                           ("nit", "uncertain", True), ("P1", "fixed", False),
                                           ("nit", "open", False)):
            with self.subTest(priority=priority, status=status):
                value = with_prior(snapshot(), [(priority, "empty batch indexes zero")])
                result = proposal(value)
                result["result"]["findings"] = []
                result["result"]["previous"] = [{"id": "0" * 16, "status": status,
                                                 "evidence": "src/a.py:2 still indexes `items[0]`."}]
                payload = prepare_review(value, result)
                body = visible(payload["body"])
                self.assertEqual(payload["comments"], [])
                label = {"open": "still open", "uncertain": "uncertain"}.get(status)
                self.assertEqual(f"- [{priority}] src/a\\.py: empty batch indexes zero ({label}). "
                                 "src/a\\.py:2 still indexes `items[0]`\\." in body, blocking)
                self.assertEqual("earlier findings remain unresolved" in body, blocking)
                self.assertEqual("Recommended for approval:" in body, not blocking)

    def test_every_hidden_blocker_has_a_specific_visible_explanation(self):
        flags = ("incomplete", "discussion", "unclear", "prior_uncertain", "prior_open", "new_open")
        for combination in itertools.product((False, True), repeat=len(flags)):
            case = dict(zip(flags, combination))
            with self.subTest(**case):
                value = with_prior(snapshot(), [("P3", "uncertain root cause"), ("P2", "open root cause")])
                result = proposal(value)["result"]
                result["previous"] = [
                    {"id": "0" * 16, "status": "uncertain" if case["prior_uncertain"] else "fixed", "evidence": "a"},
                    {"id": f"{1:016x}", "status": "open" if case["prior_open"] else "fixed", "evidence": "b"}]
                if not case["new_open"]:
                    result["findings"] = []
                if case["incomplete"]:
                    result.update(complete=False, limitations="Generated header unreadable.")
                if case["discussion"]:
                    result["discussion_blockers"] = ["Comment 88 says src/a.py:2 crashes."]
                if case["unclear"]:
                    result["usefulness"]["verdict"] = "unclear"
                payload = prepare_review(value, {"snapshot_digest": value["digest"], "reads": [["head", "src/a.py"]],
                                                 "result": result})
                body = visible(payload["body"])
                reasons = approval_blockers(decode_state(payload["body"])[1])
                self.assertEqual(bool(reasons), any(case.values()))
                self.assertEqual("Recommended for approval:" in body, not reasons)
                if reasons:
                    self.assertIn(f"Approval not recommended: {'; '.join(reasons)}.", body)
                explanations = {
                    "coverage is incomplete": "Coverage incomplete: Generated header unreadable\\.",
                    "unresolved feedback from existing discussion": "- Comment 88 says src/a\\.py:2 crashes\\.",
                    "usefulness is not established": "Usefulness: benefit unclear.",
                }
                for reason in reasons:
                    if reason == "earlier findings remain unresolved":
                        self.assertTrue("uncertain root cause (uncertain)" in body
                                        or "open root cause (still open)" in body)
                    elif reason == "new P0-P3 findings in this review":
                        self.assertTrue(any(c["body"].startswith("[P2]") for c in payload["comments"]))
                    else:
                        self.assertIn(explanations[reason], body)

    def test_long_finding_history_is_listed_in_priority_order_and_stays_small(self):
        causes = [(("P3", "P2", "P1", "P0")[i % 4], f"cause {i:02} " + "x" * 290) for i in range(30)]
        value = with_prior(snapshot(), causes)
        result = proposal(value)
        result["result"]["findings"] = []
        result["result"]["previous"] = [{"id": f"{i:016x}", "status": "open", "evidence": "e" * 600}
                                        for i in range(30)]
        body = visible(prepare_review(value, result)["body"])
        listing = body.split("Earlier findings still blocking approval:", 1)[1]
        listed = [line for line in listing.splitlines() if line.startswith("- [P")]
        self.assertEqual([line[:6] for line in listed], ["- [P0]"] * 7 + ["- [P1]"] * 3)
        self.assertIn("\n- and 20 more.", listing)
        self.assertIn("e" * 200 + "\\.\\.\\.", listing)
        self.assertNotIn("e" * 201, listing)
        self.assertLess(len(listing.encode()), 6000)

    def test_listed_blockers_escape_model_text_and_survive_publication_read_back(self):
        gh = FakeGitHub()
        value = with_prior(snapshot(gh), [("P1", "crash @maintainer ![x](https://evil.test) <img src=x>")])
        result = proposal(value)
        result["result"]["findings"] = []
        result["result"].update(complete=False, limitations="<script>x</script> @maintainer")
        result["result"]["previous"] = [{"id": "0" * 16, "status": "open", "evidence": "@maintainer <b>x</b>"}]
        self.assertEqual(publish(gh, EVENT, value, result), "published")
        body = visible(gh.reviews[0]["body"])
        for hostile in ("@maintainer", "![x]", "<img", "<script>", "<b>"):
            self.assertNotIn(hostile, body)
        self.assertIn("Earlier findings still blocking approval:", body)
        self.assertEqual(publish(gh, EVENT, value, result), "already-published")
        self.assertEqual(gh.labels, {"documentation", "ai-reviewed"})

    def test_blank_limitations_on_incomplete_review_still_give_a_reason(self):
        for limitations in ("", "   \n\t"):
            value = snapshot()
            result = proposal(value)
            result["result"].update(complete=False, limitations=limitations, findings=[])
            self.assertIn("Coverage incomplete: no reason given", prepare_review(value, result)["body"])


class StateEncodingTests(unittest.TestCase):
    def test_new_reviews_use_compressed_v2_state_that_round_trips(self):
        value = snapshot()
        body = prepare_review(value, carried_over(value))["body"]
        self.assertTrue(body.startswith(MARKER + "2 "))
        marker, state = decode_state(body)
        self.assertEqual(body, marker + summary(state))
        self.assertEqual(state["body_version"], 2)
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(marker.split()[2]))), state)

    def test_first_release_reviews_render_and_verify_byte_for_byte(self):
        gh = FakeGitHub()
        value, result, body = publish_legacy(gh)
        state = decode_state(body)[1]
        self.assertNotIn("body_version", state)
        self.assertEqual(summary(state), V1_SUMMARY)
        reconcile(gh, UPDATE)
        self.assertEqual(gh.labels, {"documentation", "ai-reviewed"})
        self.assertEqual(publish(gh, EVENT, value, result), "already-published")
        self.assertEqual(len(gh.reviews), 1)

    def test_legacy_history_feeds_a_new_v2_review(self):
        gh = FakeGitHub()
        publish_legacy(gh)
        prior = previous_state(discussions(gh, 7), 7)
        self.assertEqual(prior["findings"][0]["id"], "0" * 16)
        event = copy.deepcopy(EVENT)
        event["comment"]["id"] = 13
        gh.comments.append({**gh.command, "id": 13})
        value = snapshot(gh)
        value.update(request=13, prior=prior)
        seal(value)
        result = proposal(value)
        result["result"]["findings"] = []
        result["result"]["previous"] = [{"id": "0" * 16, "status": "open", "evidence": "Still unchecked."}]
        self.assertEqual(publish(gh, event, value, result), "published")
        self.assertTrue(gh.reviews[-1]["body"].startswith(MARKER + "2 "))
        self.assertIn("Earlier findings still blocking approval:", gh.reviews[-1]["body"])

    def test_verification_does_not_depend_on_reproducing_the_compressed_bytes(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        result["result"]["findings"] = []
        publish(gh, EVENT, value, result)
        self.assertIn("ai-approved", gh.labels)
        # A later runner's zlib may compress differently; old reviews must still verify.
        real = zlib.compress
        with patch.object(common.zlib, "compress", side_effect=lambda data, level: real(data, 1)):
            self.assertNotEqual(encode_state({"x": "y" * 99}), f"{MARKER}2 "
                                f"{base64.b64encode(real(json.dumps({'x': 'y' * 99}).encode(), 9)).decode()} -->")
            reconcile(gh, UPDATE)
            self.assertEqual(publish(gh, EVENT, value, result), "already-published")
        self.assertIn("ai-approved", gh.labels)

    def test_unsupported_versions_and_malformed_state_fail_closed(self):
        value = snapshot()
        result = proposal(value)
        result["result"]["findings"] = []
        state = decode_state(prepare_review(value, result)["body"])[1]
        for version in (3, 0, "2", True):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ReviewError, "Unsupported review body version"):
                    summary({**state, "body_version": version})
        # Valid JSON once inflated, so only the size bound can reject it.
        bomb = base64.b64encode(zlib.compress(b'"' + b"a" * common.MAX_BYTES + b'"', 9)).decode()
        truncated = base64.b64encode(zlib.compress(json.dumps(state).encode())[:-4]).decode()
        plain = base64.b64encode(json.dumps(state).encode()).decode()
        for body in (f"{MARKER}3 {plain} -->", f"{MARKER}2 {bomb} -->", f"{MARKER}2 {truncated} -->",
                     f"{MARKER}2 {plain} -->", f"{MARKER}1 !!! -->", f"{MARKER}2 {plain}"):
            with self.subTest(body=body[:40]):
                with self.assertRaisesRegex(ReviewError, "Malformed review state"):
                    decode_state(body)

    def test_unsupported_body_version_clears_approval(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        result["result"]["findings"] = []
        publish(gh, EVENT, value, result)
        marker, state = decode_state(gh.reviews[0]["body"])
        state["body_version"] = 3
        gh.reviews[0]["body"] = encode_state(state) + gh.reviews[0]["body"][len(marker):]
        with self.assertRaisesRegex(ReviewError, "approval cleared"):
            reconcile(gh, UPDATE)
        self.assertEqual(gh.labels, {"documentation", "ai-reviewed"})

    def test_full_quota_of_findings_fits_the_comment_limit(self):
        # Three attempts per PR with at most 20 new findings each: 60 findings is the most a PR can hold.
        rng = random.Random(7)
        vocabulary = [f"{rng.choice('abcdefghijklmnop')}{w}" for w in
                      ("batch index kernel stream region claim round buffer witness commit fold shard "
                       "planner release acquire binding challenge transcript verifier prover").split()] * 3

        def prose(words):
            return " ".join(rng.choice(vocabulary) for _ in range(words))

        value = snapshot()
        value["prior"] = {"head": "c" * 40, "request": 11, "findings": [
            {"priority": "P2", "path": f"crates/akita-pcs/src/file_{i}.rs", "revision": "head", "line": 10 + i,
             "root_cause": prose(8), "body": prose(70), "id": f"{i:016x}", "status": "open", "evidence": "",
             "commit": "c" * 40} for i in range(60)]}
        seal(value)
        result = proposal(value)
        result["result"]["findings"] = []
        result["result"]["previous"] = [{"id": f"{i:016x}", "status": "open", "evidence": prose(35)}
                                        for i in range(60)]
        body = prepare_review(value, result)["body"]
        state = decode_state(body)[1]
        self.assertGreater(len(base64.b64encode(json.dumps(state).encode())), 60_000)  # v1 could not publish this.
        self.assertLessEqual(len(body.encode()), 60_000)


if __name__ == "__main__":
    unittest.main()
