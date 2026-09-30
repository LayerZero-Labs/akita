"""Validate model output and publish an idempotent review with inline findings."""

import base64
import html
import json

from collect import discussions, previous_state
from common import MARKER, REPOSITORY, ReviewError, authorize, digest, revision


def bounded_text(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ReviewError("Invalid review text")
    return value


def plain(value):
    # Render model prose as text, not Markdown links, images, HTML or mentions.
    value = html.escape(value, quote=False).replace("@", "@\u200b")
    for char in "\\`*_{}[]()#+-.!|>~":
        value = value.replace(char, "\\" + char)
    return value.replace("\n", " ").replace("\r", " ")


def validate(snapshot, proposal):
    unsigned = {k: v for k, v in snapshot.items() if k != "digest"}
    if digest(unsigned) != snapshot["digest"] or proposal["snapshot_digest"] != snapshot["digest"]:
        raise ReviewError("Review snapshot mismatch")
    result = proposal["result"]
    if set(result) != {"complete", "limitations", "coverage", "previous", "findings"}:
        raise ReviewError("Unexpected review fields")
    if type(result["complete"]) is not bool or not isinstance(result["limitations"], str):
        raise ReviewError("Invalid completion status")
    if (not isinstance(result["coverage"], list)
            or any(not isinstance(x, str) for x in result["coverage"])
            or set(result["coverage"]) != set(snapshot["changed"])):
        raise ReviewError("Review did not account for every changed file")
    if len(result["limitations"]) > 4000 or len(result["findings"]) > 20:
        raise ReviewError("Review exceeds publication budget")
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
             "complete": result["complete"], "limitations": result["limitations"]}
    encoded = base64.b64encode(json.dumps(state).encode()).decode()
    body = f"{MARKER}{encoded} -->"
    comments = []
    previous = {f["id"] for f in (snapshot["prior"] or {}).get("findings", [])}
    for finding in findings:
        if finding["id"] not in previous:
            comments.append({"path": finding["path"], "line": finding["line"],
                             "side": "RIGHT" if finding["revision"] == "head" else "LEFT",
                             "body": f"[{finding['priority']}] {plain(finding['body'])}\n\n"
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
    inline = github.pages(f"pulls/{pr['number']}/reviews/{posted['id']}/comments")
    expected = {(c["path"], c["line"], c["side"], c["body"]) for c in payload["comments"]}
    actual = {(c["path"], c.get("original_line"), c["side"], c["body"]) for c in inline}
    if (verified.get("body") != payload["body"] or verified.get("commit_id") != payload["commit_id"]
            or verified.get("state") != "COMMENTED" or actual != expected):
        raise ReviewError("Published review could not be verified; do not blindly retry")
    return "published"
