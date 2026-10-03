"""Collect immutable git objects; never check out or execute PR content."""

import base64
import json
import os
import re
import subprocess

from common import MARKER, REPOSITORY, ReviewError, authorize, digest, reopen_epoch, revision, sha

# GitHub-issued identities, verified via /users/cursor[bot] and /apps/cursor.
# Keep this allowlist in trusted workflow code, never in PR-controlled config.
REVIEW_BOTS = {"cursor[bot]": (206951365, 1210556, "cursor")}


def git(*args):
    # No credential, hook, pager, textconv, external diff, or PR-owned config.
    env = {"PATH": os.environ["PATH"], "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0", "GIT_LITERAL_PATHSPECS": "1"}
    result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise ReviewError("Pinned git object could not be read or fetched")
    return result.stdout


def discussions(github, number):
    result = []
    permissions = {}
    for kind, endpoint in (("discussion", f"issues/{number}/comments"),
                           ("review", f"pulls/{number}/reviews"),
                           ("inline", f"pulls/{number}/comments")):
        for comment in github.pages(endpoint):
            user = comment.get("user") or {}
            # Bot evidence never becomes prior-review state or an instruction.
            own = (kind in ("discussion", "review") and user.get("login") == "github-actions[bot]"
                   and user.get("type") == "Bot" and comment.get("body", "").startswith(MARKER))
            uid = user.get("id")
            allowed = REVIEW_BOTS.get(user.get("login"))
            app = comment.get("performed_via_github_app")
            bot = (user.get("type") == "Bot" and allowed is not None and uid == allowed[0]
                   and (app is None or (app.get("id"), app.get("slug")) == allowed[1:]))
            # Inline/review endpoints omit app metadata; the immutable bot user ID
            # is still supplied by GitHub. Validate app identity when provided.
            if not own and not bot and uid not in permissions:
                permissions[uid] = github.writer(user)
            if not own and not bot and not permissions.get(uid):
                continue
            result.append({"kind": kind, "id": comment["id"], "user": user.get("login"),
                           "body": comment.get("body") or "", "own": own,
                           "created_at": comment.get("created_at", comment.get("submitted_at")),
                           "updated_at": comment.get("updated_at", comment.get("submitted_at")),
                           "path": comment.get("path"), "line": comment.get("line"),
                           "original_line": comment.get("original_line"),
                           "in_reply_to_id": comment.get("in_reply_to_id"),
                           "commit_id": comment.get("commit_id")})
    if len(json.dumps(result)) > 300_000:
        raise ReviewError("Discussion exceeds review budget")
    return result


def previous_state(comments, number):
    states = []
    for comment in comments:
        if not comment["own"]:
            continue
        match = re.match(re.escape(MARKER) + r"([A-Za-z0-9+/=]+) -->(?:\n|$)", comment["body"])
        if not match:
            raise ReviewError("Malformed previous review state")
        try:
            state = json.loads(base64.b64decode(match[1], validate=True))
            if state["repository"] != REPOSITORY or state["number"] != number:
                raise ValueError()
            sha(state["head"])
            if len(state["findings"]) > 100 or type(state["request"]) is not int:
                raise ValueError()
            for finding in state["findings"]:
                if not re.fullmatch(r"[0-9a-f]{16}", finding["id"]):
                    raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise ReviewError("Invalid previous review state") from None
        states.append((comment.get("created_at") or comment.get("updated_at") or "", comment["id"], state))
    return max(states, default=("", 0, None))[2]


def diff_lines(patch):
    """Map valid LEFT/RIGHT review coordinates from one file's unified diff."""
    result = {"base": [], "head": []}
    old = new = None
    for line in patch.splitlines():
        hunk = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if hunk:
            old, new = map(int, hunk.groups())
        elif old is not None and line.startswith(" "):
            result["base"].append(old)
            result["head"].append(new)
            old, new = old + 1, new + 1
        elif old is not None and line.startswith("-"):
            result["base"].append(old)
            old += 1
        elif new is not None and line.startswith("+"):
            result["head"].append(new)
            new += 1
    return result


def tree(commit, blobs):
    files, excluded = {}, []
    for entry in git("ls-tree", "-r", "-z", "-l", sha(commit)).split(b"\0"):
        if not entry:
            continue
        info, raw_path = entry.split(b"\t", 1)
        mode, kind, oid, size = info.split()
        path = raw_path.decode("utf-8", errors="strict")
        if (mode not in (b"100644", b"100755") or kind != b"blob"
                or int(size) > 250_000):
            excluded.append(path)
            continue
        key = oid.decode()
        if key not in blobs:
            raw = git("cat-file", "blob", key)
            try:
                text = raw.decode("utf-8")
                if "\x00" in text:
                    raise ValueError()
            except (UnicodeDecodeError, ValueError):
                excluded.append(path)
                continue
            blobs[key] = text
        files[path] = key
    return {"files": files, "excluded": excluded}


def since_previous(base, previous, head, changed):
    """Diff the previous reviewed head against the current one, in the PR's own files.

    A diff of the two commits alone would also carry every base-branch change the
    PR merged or was rebased onto since the previous review. That is not the PR's
    change and can exceed the review budget by itself. The PR's own files are the
    ones it changes now and the ones it changed at the previous head, so a
    reverted file still appears.
    """
    previous_base = git("merge-base", base, previous).decode().strip()
    before = git("diff", "--no-renames", "--name-only", "-z", previous_base, previous).decode().split("\0")[:-1]
    paths = sorted(set(changed) | set(before))
    if len(paths) > 300:
        raise ReviewError("Diff exceeds automated review budget; use a manual review")
    if not paths:
        return ""
    return git("diff", "--no-ext-diff", "--no-textconv", "--no-renames", previous, head, "--", *paths).decode()


def collect(github, event):
    pr = authorize(github, event)
    number = pr["number"]
    epoch = reopen_epoch(github, number)
    comments = discussions(github, number)
    prior = previous_state(comments, number)
    if any(c["own"] and previous_state([c], number)["request"] == event["comment"]["id"]
           for c in comments):
        return None  # Idempotent re-run of this exact command, even after newer reviews.
    revisions = revision(pr)
    for commit in {revisions["head"], revisions["base"]} | ({prior["head"]} if prior else set()):
        git("fetch", "--no-tags", "--no-recurse-submodules",
            f"https://github.com/{REPOSITORY}.git", sha(commit))
    merge_base = git("merge-base", revisions["base"], revisions["head"]).decode().strip()
    changed = git("diff", "--no-renames", "--name-only", "-z", merge_base, revisions["head"]).decode().split("\0")[:-1]
    diff = git("diff", "--no-ext-diff", "--no-textconv", "--no-renames", merge_base, revisions["head"]).decode()
    delta = since_previous(revisions["base"], prior["head"], revisions["head"], changed) if prior else ""
    if len(diff) + len(delta) > 300_000 or len(changed) > 150:
        raise ReviewError("Diff exceeds automated review budget; use a manual review")
    anchors = {path: diff_lines(git("diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                                    "--unified=3", merge_base, revisions["head"], "--", path).decode())
               for path in changed}
    blobs = {}
    trees = {name: tree(commit, blobs) for name, commit in
             {"head": revisions["head"], "base": merge_base,
              **({"previous": prior["head"]} if prior else {})}.items()}
    snapshot = {"repository": REPOSITORY, "number": number, "request": event["comment"]["id"],
                "reopen_epoch": epoch,
                "revision": revisions, "merge_base": merge_base, "title": pr["title"],
                "description": pr.get("body") or "", "comments": comments, "prior": prior,
                "diff": diff, "delta": delta, "changed": changed, "anchors": anchors,
                "trees": trees, "blobs": blobs}
    fresh = authorize(github, event)
    if (revision(fresh) != revisions or discussions(github, number) != comments
            or reopen_epoch(github, number) != epoch):
        raise ReviewError("PR changed during collection; post a new command")
    snapshot["digest"] = digest(snapshot)
    return snapshot
