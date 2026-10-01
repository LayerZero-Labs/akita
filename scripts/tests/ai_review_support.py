"""Shared in-memory GitHub fixture and review builders; no live API calls."""

import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai_review"))
from collect import discussions
from common import REPOSITORY, digest, reopen_epoch, revision

HEAD = "a" * 40
BASE = "b" * 40
AUTHOR = {"id": 42, "login": "maintainer", "type": "User"}
EVENT = {"action": "created", "repository": {"full_name": REPOSITORY},
         "issue": {"number": 7, "pull_request": {}}, "comment": {"id": 12, "user": AUTHOR},
         "sender": AUTHOR}
EVENT["issue"]["pull_request"] = {"url": "unused"}
PR = {"number": 7, "state": "open", "user": AUTHOR, "title": "Fix batch", "body": "Details",
      "head": {"sha": HEAD, "ref": "feature", "repo": {"full_name": REPOSITORY}},
      "base": {"sha": BASE, "ref": "main", "repo": {"full_name": REPOSITORY}}}
COMMAND = {"id": 12, "user": AUTHOR, "body": "/ai-review", "updated_at": "now",
           "issue_url": f"https://api.github.com/repos/{REPOSITORY}/issues/7"}


class FakeGitHub:
    def __init__(self):
        self.pr, self.command = copy.deepcopy(PR), copy.deepcopy(COMMAND)
        self.permission = "write"
        self.comments = [self.command]
        self.reviews, self.inline, self.writes = [], [], []
        self.labels = {"documentation"}
        self.events = []
        self.permissions_checked = []

    def get(self, path, payload=None, method=None):
        if method == "DELETE":
            if path != "issues/7/labels/ai-approved":
                raise AssertionError(path)
            self.writes.append((path, None))
            self.labels.discard("ai-approved")
            return [{"name": name} for name in self.labels]
        if payload is not None:
            self.writes.append((path, payload))
            if path == "issues/7/labels":
                self.labels.update(payload["labels"])
                return [{"name": name} for name in self.labels]
            if path == "issues/7/comments":
                posted = {"id": 1000 + len(self.comments), "body": payload["body"],
                          "user": {"login": "github-actions[bot]", "type": "Bot", "id": 1}}
                self.comments.append(posted)
                return posted
            posted = {"id": 99 + len(self.reviews), "user": {"login": "github-actions[bot]", "type": "Bot", "id": 1},
                      "body": payload["body"], "submitted_at": "later",
                      "state": "COMMENTED", "commit_id": payload["commit_id"]}
            self.reviews.append(posted)
            self.inline.extend({**c, "original_line": c["line"], "id": 200 + i,
                                "user": posted["user"], "pull_request_review_id": posted["id"]}
                               for i, c in enumerate(payload["comments"]))
            return posted
        if path.startswith("pulls/7/reviews/"):
            return next(r for r in self.reviews if r["id"] == int(path.split("/")[-1]))
        if path == "pulls/7":
            return self.pr
        if path.startswith("issues/comments/"):
            return next(c for c in self.comments if c["id"] == int(path.split("/")[-1]))
        raise AssertionError(path)

    def writer(self, user):
        self.permissions_checked.append(user.get("id"))
        return user.get("id") == 42 and self.permission in ("admin", "maintain", "write")

    def pages(self, path):
        if path == "issues/7/events":
            return self.events
        if path == "issues/7/labels":
            return [{"name": name} for name in self.labels]
        return {"issues/7/comments": self.comments, "pulls/7/reviews": self.reviews,
                "pulls/7/comments": self.inline, "pulls/7/reviews/99/comments": self.inline}[path]


def snapshot(github=None):
    github = github or FakeGitHub()
    value = {"repository": REPOSITORY, "number": 7, "request": 12,
             "reopen_epoch": reopen_epoch(github, 7),
             "revision": revision(github.pr), "merge_base": BASE,
             "title": "Fix batch", "description": "Details", "prior": None,
             "comments": discussions(github, 7), "changed": ["src/a.py"], "diff": "diff", "delta": "",
             "anchors": {"src/a.py": {"head": [1, 2], "base": [1, 2]}},
             "trees": {"head": {"files": {"src/a.py": "blob"}, "excluded": []},
                       "base": {"files": {"src/a.py": "blob"}, "excluded": []}},
             "blobs": {"blob": "def first(items):\n    return items[0]\n"}}
    seal(value)
    return value


def seal(value):
    value["digest"] = digest({k: v for k, v in value.items() if k != "digest"})


def proposal(value):
    return {"snapshot_digest": value["digest"], "reads": [["head", "src/a.py"]],
            "result": {"complete": True, "limitations": "", "discussion_blockers": [], "coverage": value["changed"],
                       "usefulness": {"motivation": "provided", "verdict": "beneficial",
                                      "assessment": "The src/a.py empty-batch fix restores the documented input contract without adding an abstraction."},
                       "previous": [], "findings": [{"priority": "P2", "path": "src/a.py",
                           "revision": "head", "line": 2, "root_cause": "empty batch indexes zero",
                           "body": "Empty inputs raise instead of returning None; validate first."}]}}
