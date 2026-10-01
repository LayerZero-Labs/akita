"""Validate model output and publish an idempotent review with inline findings."""

import base64
import html
import json
import re

from collect import discussions, previous_state
from common import MARKER, REPOSITORY, ReviewError, authorize, digest, revision


def bounded_text(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ReviewError("Invalid review text")
    return value


def review_text(value):
    """Keep matched CommonMark code spans; escape all other model Markdown."""
    value = value.replace("\n", " ").replace("\r", " ")
    runs = list(re.finditer(r"`+", value))
    pieces, cursor, index = [], 0, 0
    while index < len(runs):
        opening = runs[index]
        closing = next((j for j in range(index + 1, len(runs))
                        if runs[j].group() == opening.group()), None)
        if closing is None:
            index += 1
            continue
        pieces.append((False, value[cursor:opening.start()]))
        pieces.append((True, value[opening.start():runs[closing].end()]))
        cursor, index = runs[closing].end(), closing + 1
    pieces.append((False, value[cursor:]))
    output = []
    for is_code, piece in pieces:
        if not is_code:
            piece = html.escape(piece, quote=False).replace("@", "@\u200b")
            for char in "\\`*_{}[]()#+-.!|>~":
                piece = piece.replace(char, "\\" + char)
        output.append(piece)
    return "".join(output)


def validate(snapshot, proposal):
    unsigned = {k: v for k, v in snapshot.items() if k != "digest"}
    if digest(unsigned) != snapshot["digest"] or proposal["snapshot_digest"] != snapshot["digest"]:
        raise ReviewError("Review snapshot mismatch")
    result = proposal["result"]
    if set(result) != {"complete", "limitations", "discussion_blockers", "coverage", "previous", "findings"}:
        raise ReviewError("Unexpected review fields")
    if type(result["complete"]) is not bool or not isinstance(result["limitations"], str):
        raise ReviewError("Invalid completion status")
    if (not isinstance(result["coverage"], list)
            or any(not isinstance(x, str) for x in result["coverage"])
            or set(result["coverage"]) != set(snapshot["changed"])):
        raise ReviewError("Review did not account for every changed file")
    if len(result["limitations"]) > 4000 or len(result["findings"]) > 20:
        raise ReviewError("Review exceeds publication budget")
    if not isinstance(result["discussion_blockers"], list) or len(result["discussion_blockers"]) > 100:
        raise ReviewError("Invalid discussion blockers")
    for blocker in result["discussion_blockers"]:
        bounded_text(blocker, 2000)
    prior = {f["id"]: f for f in (snapshot["prior"] or {}).get("findings", [])}
    updates = result["previous"]
    if (not isinstance(updates, list) or len(updates) != len(prior)
            or {u["id"] for u in updates} != set(prior)):
        raise ReviewError("Review must reassess every previous finding")
    findings = []
    for update in updates:
        if set(update) != {"id", "status", "evidence"} or update["status"] not in ("open", "fixed", "uncertain"):
            raise ReviewError("Invalid previous finding status")
        bounded_text(update["evidence"], 2000)
        findings.append({**prior[update["id"]], "status": update["status"], "evidence": update["evidence"]})
    reads = {tuple(x) for x in proposal["reads"]}
    if result["complete"] and any(
        not any((rev, path) in reads for rev in ("base", "head"))
        for path in snapshot["changed"]
    ):
        raise ReviewError("Complete review requires reading every changed file")
    for finding in result["findings"]:
        if (set(finding) != {"priority", "path", "revision", "line", "root_cause", "body"}
                or finding["priority"] not in ("P0", "P1", "P2", "P3", "nit")
                or finding["revision"] not in ("base", "head")):
            raise ReviewError("Invalid finding fields")
        path, rev, line = finding["path"], finding["revision"], finding["line"]
        files = snapshot["trees"][rev]["files"]
        if (path not in snapshot["changed"] or path not in files or (rev, path) not in reads
                or type(line) is not int or not 1 <= line <= len(snapshot["blobs"][files[path]].splitlines())
                or line not in snapshot["anchors"].get(path, {}).get(rev, [])):
            raise ReviewError("Finding lacks a verified source location")
        bounded_text(finding["body"], 2000)
        root = bounded_text(finding["root_cause"], 300)
        fid = digest({"path": path, "root": root.casefold().strip()})[:16]
        if fid in {f["id"] for f in findings}:
            raise ReviewError("Duplicate finding; reassess existing finding instead")
        findings.append({**finding, "id": fid, "status": "open", "evidence": "",
                         "commit": snapshot["revision"]["head"] if rev == "head" else snapshot["merge_base"]})
    if len(findings) > 100:
        raise ReviewError("Finding history limit reached")
    return findings


def prepare_review(snapshot, proposal):
    findings = validate(snapshot, proposal)
    result = proposal["result"]
    state = {"repository": REPOSITORY, "number": snapshot["number"], "request": snapshot["request"],
             "head": snapshot["revision"]["head"], "findings": findings,
             "complete": result["complete"], "limitations": result["limitations"],
             "discussion_blockers": result["discussion_blockers"]}
    encoded = base64.b64encode(json.dumps(state).encode()).decode()
    body = f"{MARKER}{encoded} -->"
    remaining = [finding for finding in findings if finding["status"] != "fixed"]
    if (result["complete"] and not result["discussion_blockers"]
            and all(f["status"] == "open" and f["priority"] == "nit" for f in remaining)):
        reason = "only optional nits remain" if remaining else "no unresolved findings"
        body += f"\n\nRecommended for approval: {reason} in this automated review."
    comments = []
    previous = {f["id"] for f in (snapshot["prior"] or {}).get("findings", [])}
    for finding in findings:
        if finding["id"] not in previous:
            comments.append({"path": finding["path"], "line": finding["line"],
                             "side": "RIGHT" if finding["revision"] == "head" else "LEFT",
                             "body": f"[{finding['priority']}] {review_text(finding['body'])}\n\n"
                                     f"<!-- akita-ai-review-finding:{finding['id']} -->"})
    if len(body.encode()) > 60_000:
        raise ReviewError("Published review exceeds comment size limit")
    return {"commit_id": snapshot["revision"]["head"], "event": "COMMENT", "body": body, "comments": comments}


def publish(github, event, snapshot, proposal):
    payload = prepare_review(snapshot, proposal)
    pr = authorize(github, event)
    if snapshot["number"] != pr["number"] or snapshot["request"] != event["comment"]["id"]:
        raise ReviewError("Review does not belong to this request")
    comments = discussions(github, pr["number"])
    if any(c["own"] and previous_state([c], pr["number"])["request"] == snapshot["request"]
           for c in comments):
        return "already-published"
    if (revision(pr) != snapshot["revision"] or comments != snapshot["comments"]
            or pr["title"] != snapshot["title"] or (pr.get("body") or "") != snapshot["description"]):
        raise ReviewError("Review is stale; post a new /ai-review command")
    posted = github.get(f"pulls/{pr['number']}/reviews", payload)
    verified = github.get(f"pulls/{pr['number']}/reviews/{posted['id']}")
    # The review-specific endpoint returns legacy position-only comment objects.
    inline = [c for c in github.pages(f"pulls/{pr['number']}/comments")
              if c.get("pull_request_review_id") == posted["id"]]
    expected = {(c["path"], c["line"], c["side"], c["body"]) for c in payload["comments"]}
    actual = {(c.get("path"), c.get("original_line"), c.get("side"), c.get("body")) for c in inline}
    if (verified.get("body") != payload["body"] or verified.get("commit_id") != payload["commit_id"]
            or verified.get("state") != "COMMENTED" or actual != expected):
        raise ReviewError("Published review could not be verified; do not blindly retry")
    return "published"
