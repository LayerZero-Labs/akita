"""Validate model output and publish one idempotent, immutable PR conversation comment."""

import base64
import html
import json
import re
from urllib.parse import quote

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
                or type(line) is not int or not 1 <= line <= len(snapshot["blobs"][files[path]].splitlines())):
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


def render(snapshot, proposal):
    findings = validate(snapshot, proposal)
    result = proposal["result"]
    state = {"repository": REPOSITORY, "number": snapshot["number"], "request": snapshot["request"],
             "head": snapshot["revision"]["head"], "findings": findings}
    encoded = base64.b64encode(json.dumps(state).encode()).decode()
    lines = [f"{MARKER}{encoded} -->", "### AI review", "",
             f"Reviewed `{snapshot['revision']['head']}`. Automated source review; no PR code was executed.", ""]
    previous = {f["id"] for f in (snapshot["prior"] or {}).get("findings", [])}
    for finding in findings:
        if finding["id"] in previous:
            lines.append(f"- Previous `{finding['id']}`: **{finding['status']}** — {plain(finding['evidence'])}")
        else:
            url = (f"https://github.com/{REPOSITORY}/blob/{finding['commit']}/"
                   f"{quote(finding['path'], safe='/')}#L{finding['line']}")
            lines += [f"- **[{finding['priority']}]** [{plain(finding['path'])}:{finding['line']}]({url}) "
                      f"(`{finding['id']}`): {plain(finding['body'])}"]
    if not result["complete"]:
        lines += ["", "**Incomplete review.** " + plain(result["limitations"] or "Coverage could not be established.")]
    elif not findings or all(f["status"] == "fixed" for f in findings):
        lines += ["No new findings; previous findings, if any, were assessed as fixed. This is not a merge approval."]
    if result["complete"] and result["limitations"]:
        lines += ["", "Limitations: " + plain(result["limitations"])]
    body = "\n".join(lines)
    if len(body.encode()) > 60_000:
        raise ReviewError("Published review exceeds comment size limit")
    return body


def publish(github, event, snapshot, proposal):
    body = render(snapshot, proposal)
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
    posted = github.get(f"issues/{pr['number']}/comments", {"body": body})
    verified = github.get(f"issues/comments/{posted['id']}")
    if verified.get("body") != body:
        raise ReviewError("Published comment could not be verified; do not blindly retry")
    return "published"
