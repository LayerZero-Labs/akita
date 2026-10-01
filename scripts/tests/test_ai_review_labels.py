"""Approval scope, lifecycle ordering, and label recovery."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from ai_review_support import (AUTHOR, COMMAND, EVENT, FakeGitHub, PR, proposal, seal, snapshot)
from collect import discussions, previous_state
from common import MARKER, REPOSITORY, ReviewError
from publish import publish, prepare_review, reconcile
import run as entrypoint


class LabelTests(unittest.TestCase):
    def test_delayed_and_replayed_updates_preserve_newer_approval(self):
        # The workflow lock permits either order; reconciliation must converge
        # whether it first runs before or after publication of the fresh review.
        for action in ("synchronize", "edited", "reopened"):
            for reconcile_first in (True, False):
                with self.subTest(action=action, reconcile_first=reconcile_first):
                    gh = FakeGitHub()
                    old = snapshot(gh)
                    clean = proposal(old)
                    clean["result"]["findings"] = []
                    publish(gh, EVENT, old, clean)
                    update = {"repository": {"full_name": REPOSITORY}, "action": action,
                              "pull_request": copy.deepcopy(gh.pr)}
                    if action == "synchronize":
                        gh.pr["head"]["sha"] = "c" * 40
                    elif action == "edited":
                        gh.pr["body"] = "Updated scope"
                    else:
                        gh.events.append({"id": 41, "event": "reopened"})
                    if reconcile_first:
                        reconcile(gh, update)
                        self.assertNotIn("ai-approved", gh.labels)
                    event = copy.deepcopy(EVENT)
                    event["comment"]["id"] = 13
                    gh.comments.append({**COMMAND, "id": 13})
                    fresh = snapshot(gh)
                    fresh.update(request=13, title=gh.pr["title"], description=gh.pr["body"],
                                 prior=previous_state(discussions(gh, 7), 7))
                    seal(fresh)
                    clean = proposal(fresh)
                    clean["result"]["findings"] = []
                    publish(gh, event, fresh, clean)
                    writes = len(gh.writes)
                    for _ in range(2):
                        reconcile(gh, update)
                    self.assertEqual(gh.labels, {"documentation", "ai-reviewed", "ai-approved"})
                    self.assertEqual(len(gh.writes), writes)
                    self.assertEqual(len(gh.reviews), 2)

    def test_reconciliation_recovers_labels_without_artifacts_or_model_key(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        clean = proposal(value)
        clean["result"]["findings"] = []
        publish(gh, EVENT, value, clean)
        gh.labels = {"documentation"}
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory, "event.json")
            event_path.write_text(json.dumps({"repository": {"full_name": REPOSITORY},
                                             "action": "edited", "pull_request": PR}))
            with patch.dict(os.environ, {"GH_TOKEN": "test", "GITHUB_RUN_ATTEMPT": "2",
                                         "GITHUB_EVENT_PATH": str(event_path)}, clear=True), \
                    patch.object(sys, "argv", ["run.py", "reconcile", "--directory", directory]), \
                    patch.object(entrypoint, "GitHub", return_value=gh), \
                    patch.object(entrypoint, "review") as model:
                entrypoint.main()
                model.assert_not_called()
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {"event.json"})
        self.assertEqual(gh.labels, {"documentation", "ai-reviewed", "ai-approved"})
        self.assertEqual(len(gh.reviews), 1)

    def test_reconciliation_clears_unverifiable_approval_without_falling_back(self):
        for defect in ("missing-inline", "malformed-state", "dismissed", "blocking"):
            with self.subTest(defect=defect):
                gh = FakeGitHub()
                value = snapshot(gh)
                result = proposal(value)
                result["result"]["findings"][0]["priority"] = "nit"
                publish(gh, EVENT, value, result)
                # A newer review must win over the original clean review.
                newer = copy.deepcopy(gh.reviews[-1])
                newer["id"] += 1
                gh.reviews.append(newer)
                gh.inline.append({**gh.inline[0], "id": 201, "pull_request_review_id": newer["id"]})
                if defect == "missing-inline":
                    gh.inline.pop()
                elif defect == "malformed-state":
                    newer["body"] = MARKER + "broken"
                elif defect == "dismissed":
                    newer["state"] = "DISMISSED"
                else:
                    result["result"]["findings"][0]["priority"] = "P2"
                    payload = prepare_review(value, result)
                    newer["body"] = payload["body"]
                    gh.inline[-1].update(payload["comments"][0])
                event = {"repository": {"full_name": REPOSITORY}, "action": "edited", "pull_request": PR}
                if defect == "blocking":
                    reconcile(gh, event)
                else:
                    with self.assertRaisesRegex(ReviewError, "approval cleared"):
                        reconcile(gh, event)
                self.assertEqual(gh.labels, {"documentation", "ai-reviewed"})

    def test_reconciliation_ignores_external_markers_and_does_not_invent_review_history(self):
        gh = FakeGitHub()
        payload = prepare_review(snapshot(gh), proposal(snapshot(gh)))
        gh.reviews.append({"id": 99, "body": payload["body"], "user": AUTHOR})
        gh.labels.add("ai-approved")
        reconcile(gh, {"repository": {"full_name": REPOSITORY}, "action": "edited", "pull_request": PR})
        self.assertEqual(gh.labels, {"documentation"})

    def test_reconciliation_rejects_ineligible_events_without_writes(self):
        for change in ("repository", "action", "number"):
            gh = FakeGitHub()
            event = {"repository": {"full_name": REPOSITORY}, "action": "edited", "pull_request": {"number": 7}}
            if change == "repository":
                event["repository"]["full_name"] = "attacker/akita"
            elif change == "action":
                event["action"] = "created"
            else:
                event["pull_request"]["number"] = "7/labels"
            with self.assertRaises(ReviewError):
                reconcile(gh, event)
            self.assertFalse(gh.writes)

    def test_reopen_then_replay_cannot_restore_approval_but_fresh_review_can(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        result["result"]["findings"] = []
        publish(gh, EVENT, value, result)
        gh.events.extend([{"id": 40, "event": "closed"}, {"id": 41, "event": "reopened"}])
        gh.labels.discard("ai-approved")
        publish(gh, EVENT, value, result)
        self.assertEqual(gh.labels, {"documentation", "ai-reviewed"})
        event = copy.deepcopy(EVENT)
        event["comment"]["id"] = 13
        gh.comments.append({**COMMAND, "id": 13})
        fresh = snapshot(gh)
        fresh["request"] = 13
        fresh["prior"] = previous_state(discussions(gh, 7), 7)
        seal(fresh)
        clean = proposal(fresh)
        clean["result"]["findings"] = []
        publish(gh, event, fresh, clean)
        self.assertIn("ai-approved", gh.labels)

    def test_reopening_after_collection_rejects_publication(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        gh.events.append({"id": 41, "event": "reopened"})
        with self.assertRaisesRegex(ReviewError, "stale"):
            publish(gh, EVENT, value, proposal(value))
        self.assertFalse(gh.writes)

    def test_concurrent_invalidation_before_label_add_cannot_leave_stale_approval(self):
        for change in ("head", "reopened", "closed", "body"):
            with self.subTest(change=change):
                gh = FakeGitHub()
                value = snapshot(gh)
                clean = proposal(value)
                clean["result"]["findings"] = []
                original = gh.get

                def update_pr_before_label_write(path, payload=None, method=None):
                    if path == "issues/7/labels" and payload:
                        # Return separate API objects so mutating the live PR
                        # cannot also change the publisher's earlier response.
                        if change == "reopened":
                            gh.events.append({"id": 41, "event": "reopened"})
                        elif change == "head":
                            gh.pr["head"]["sha"] = "c" * 40
                        elif change == "closed":
                            gh.pr["state"] = "closed"
                        else:
                            gh.pr["body"] = "New scope"
                        gh.labels.discard("ai-approved")
                    return copy.deepcopy(original(path, payload, method))

                with patch.object(gh, "get", side_effect=update_pr_before_label_write):
                    publish(gh, EVENT, value, clean)
                self.assertEqual(gh.labels, {"documentation", "ai-reviewed"})

    def test_failed_verification_still_blocks_labels_on_retry(self):
        for defect in ("missing", "duplicate", "body", "line", "side", "commit", "state", "summary"):
            with self.subTest(defect=defect):
                gh = FakeGitHub()
                value = snapshot(gh)
                result = proposal(value)
                result["result"]["findings"][0]["priority"] = "nit"
                original = gh.get

                def corrupt_publication(path, payload=None, method=None):
                    response = original(path, payload, method)
                    if path == "pulls/7/reviews" and payload:
                        if defect == "missing":
                            gh.inline.clear()
                        elif defect == "duplicate":
                            gh.inline.append(copy.deepcopy(gh.inline[0]))
                        elif defect in ("body", "line", "side"):
                            key, content = {"body": ("body", "Altered finding"),
                                            "line": ("original_line", 99), "side": ("side", "LEFT")}[defect]
                            gh.inline[0][key] = content
                        elif defect == "commit":
                            gh.reviews[-1]["commit_id"] = "c" * 40
                        elif defect == "state":
                            gh.reviews[-1]["state"] = "DISMISSED"
                        else:
                            gh.reviews[-1]["body"] += "\nUnexpected summary"
                    return response

                with patch.object(gh, "get", side_effect=corrupt_publication):
                    with self.assertRaisesRegex(ReviewError, "could not be verified"):
                        publish(gh, EVENT, value, result)
                for _ in range(2):
                    with self.assertRaisesRegex(ReviewError, "could not be verified"):
                        publish(gh, EVENT, value, result)
                self.assertEqual(len(gh.reviews), 1)
                self.assertEqual(gh.labels, {"documentation"})

    def test_replay_verifies_latest_review_even_when_request_is_older(self):
        gh = FakeGitHub()
        old = snapshot(gh)
        clean = proposal(old)
        clean["result"]["findings"] = []
        publish(gh, EVENT, old, clean)
        event = copy.deepcopy(EVENT)
        event["comment"]["id"] = 13
        gh.comments.append({**COMMAND, "id": 13})
        fresh = snapshot(gh)
        fresh["request"] = 13
        fresh["prior"] = previous_state(discussions(gh, 7), 7)
        seal(fresh)
        publish(gh, event, fresh, proposal(fresh))
        gh.labels.discard("ai-reviewed")
        gh.inline.clear()
        with self.assertRaisesRegex(ReviewError, "could not be verified"):
            publish(gh, EVENT, old, clean)
        self.assertEqual(gh.labels, {"documentation"})

    def test_blocking_or_incomplete_review_removes_only_approval_label(self):
        for complete, blockers in ((True, []), (False, []), (True, ["An unresolved human finding"])):
            gh = FakeGitHub()
            gh.labels.update({"ai-reviewed", "ai-approved"})
            value = snapshot(gh)
            result = proposal(value)
            result["result"].update(complete=complete, discussion_blockers=blockers)
            if not complete or blockers:
                result["result"]["findings"] = []
            publish(gh, EVENT, value, result)
            self.assertEqual(gh.labels, {"documentation", "ai-reviewed"})

    def test_label_failure_is_repaired_without_publishing_another_review(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        result["result"]["findings"] = []
        original = gh.get

        def fail_label_write(path, payload=None, method=None):
            if path == "issues/7/labels" and payload:
                raise ReviewError("Label write failed")
            return original(path, payload, method)

        with patch.object(gh, "get", side_effect=fail_label_write):
            with self.assertRaises(ReviewError):
                publish(gh, EVENT, value, result)
        self.assertEqual(len(gh.reviews), 1)
        self.assertEqual(gh.labels, {"documentation"})
        self.assertEqual(publish(gh, EVENT, value, result), "already-published")
        self.assertEqual(len(gh.reviews), 1)
        self.assertEqual(gh.labels, {"documentation", "ai-reviewed", "ai-approved"})

    def test_old_event_replay_uses_latest_review_and_cannot_restore_old_approval(self):
        gh = FakeGitHub()
        old = snapshot(gh)
        clean = proposal(old)
        clean["result"]["findings"] = []
        publish(gh, EVENT, old, clean)
        event = copy.deepcopy(EVENT)
        event["comment"]["id"] = 13
        gh.comments.append({**COMMAND, "id": 13})
        current = snapshot(gh)
        current["request"] = 13
        current["prior"] = previous_state(discussions(gh, 7), 7)
        seal(current)
        publish(gh, event, current, proposal(current))
        self.assertNotIn("ai-approved", gh.labels)
        publish(gh, EVENT, old, clean)
        self.assertNotIn("ai-approved", gh.labels)
        self.assertEqual(len(gh.reviews), 2)

    def test_changed_head_target_or_description_cannot_reapply_approval_on_retry(self):
        for change in ("head", "base_ref", "description"):
            gh = FakeGitHub()
            value = snapshot(gh)
            result = proposal(value)
            result["result"]["findings"] = []
            publish(gh, EVENT, value, result)
            if change == "description":
                gh.pr["body"] = "New scope"
            elif change == "base_ref":
                gh.pr["base"]["ref"] = "dev"
            else:
                gh.pr[change]["sha"] = "c" * 40
            publish(gh, EVENT, value, result)
            self.assertEqual(gh.labels, {"documentation", "ai-reviewed"})

    def test_target_branch_advance_preserves_approval_on_replay_and_reconciliation(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        result["result"]["findings"] = []
        publish(gh, EVENT, value, result)
        writes = len(gh.writes)
        gh.pr["base"]["sha"] = "c" * 40  # Another PR merges; no PR event fires.
        self.assertIn("ai-approved", gh.labels)
        publish(gh, EVENT, value, result)
        reconcile(gh, {"repository": {"full_name": REPOSITORY}, "action": "edited", "pull_request": PR})
        self.assertIn("ai-approved", gh.labels)
        self.assertEqual(len(gh.writes), writes)

    def test_target_branch_advance_during_label_write_does_not_revoke_approval(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        result = proposal(value)
        result["result"]["findings"] = []
        original = gh.get

        def advance_base_before_label_write(path, payload=None, method=None):
            if path == "issues/7/labels" and payload:
                gh.pr["base"]["sha"] = "c" * 40
            return copy.deepcopy(original(path, payload, method))

        with patch.object(gh, "get", side_effect=advance_base_before_label_write):
            publish(gh, EVENT, value, result)
        self.assertIn("ai-approved", gh.labels)

    def test_failed_review_verification_does_not_add_labels(self):
        gh = FakeGitHub()
        value = snapshot(gh)
        original = gh.get

        def corrupt_readback(path, payload=None, method=None):
            response = original(path, payload, method)
            if path.startswith("pulls/7/reviews/"):
                return {**response, "body": "Unexpected content"}
            return response

        with patch.object(gh, "get", side_effect=corrupt_readback):
            with self.assertRaises(ReviewError):
                publish(gh, EVENT, value, proposal(value))
        self.assertEqual(gh.labels, {"documentation"})


if __name__ == "__main__":
    unittest.main()
