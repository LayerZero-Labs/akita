"""Validate model output and publish an idempotent review with inline findings."""

import base64
import html
import json
import os
import re

from collect import discussions, previous_state
from common import MARKER, REPOSITORY, ReviewError, authorize, digest, reopen_epoch, revision


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
    if set(result) != {"complete", "limitations", "usefulness", "discussion_blockers", "coverage", "previous", "findings"}:
        raise ReviewError("Unexpected review fields")
    usefulness = result["usefulness"]
    if (not isinstance(usefulness, dict) or set(usefulness) != {"motivation", "verdict", "assessment"}
            or usefulness["motivation"] not in ("provided", "missing")
            or usefulness["verdict"] not in ("beneficial", "unclear", "not_beneficial")):
        raise ReviewError("Invalid usefulness assessment")
    bounded_text(usefulness["assessment"], 3000)
    if type(result["complete"]) is not bool or not isinstance(result["limitations"], str):
        raise ReviewError("Invalid completion status")
    if result["complete"] and not snapshot["changed"] and snapshot.get("excluded_artifacts"):
        raise ReviewError("Artifact-only changes require manual validation; review cannot be complete")
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


def approval_recommended(state):
    return (state["complete"] and not state["discussion_blockers"]
            and state.get("usefulness", {}).get("verdict") == "beneficial"
            and all(f["status"] == "fixed" or (f["status"] == "open" and f["priority"] == "nit")
                    for f in state["findings"]))


def review_body(state):
    encoded = base64.b64encode(json.dumps(state).encode()).decode()
    body = f"{MARKER}{encoded} -->"
    if state.get("excluded_artifact_count"):
        body += (f"\n\nAkita artifacts excluded: {state['excluded_artifact_count']} changed files under `artifacts/`. "
                 "Their contents were not reviewed; validate them separately with the artifact checks and CI.")
    if "usefulness" in state:
        usefulness = state["usefulness"]
        verdict = {"beneficial": "benefit supported", "unclear": "benefit unclear",
                   "not_beneficial": "benefit not supported"}[usefulness["verdict"]]
        body += f"\n\nUsefulness: {verdict}. {review_text(usefulness['assessment'])}"
        if usefulness["motivation"] == "missing":
            body += ("\n\nMotivation is missing or too vague in the PR description. "
                     "Please add a Motivation section explaining the problem and expected benefit.")
    if approval_recommended(state):
        remaining = any(finding["status"] != "fixed" for finding in state["findings"])
        reason = "only optional nits remain" if remaining else "no unresolved findings"
        body += f"\n\nRecommended for approval: {reason} in this automated review."
    if len(body.encode()) > 60_000:
        raise ReviewError("Published review exceeds comment size limit")
    return body


def approval_scope(revisions, title, description):
    """Approval follows PR content and target, not unrelated target-branch pushes."""
    return {"revision": {key: revisions[key] for key in ("head", "head_ref", "base_ref")},
            "title": title, "description": description}


def sync_labels(github, number, state):
    # Only the newest verified review may determine labels, including on retries.
    # These destinations and label names never come from model output.
    pr = github.get(f"pulls/{number}")
    scope = approval_scope(revision(pr), pr["title"], pr.get("body") or "")
    approved = (state is not None and state["head"] == pr["head"]["sha"]
                and pr["state"] == "open"
                and state.get("scope_digest") == digest(scope)
                and state.get("reopen_epoch") == reopen_epoch(github, number)
                and approval_recommended(state))
    endpoint = f"issues/{number}/labels"
    labels = {label["name"] for label in github.pages(endpoint)}
    if not approved and "ai-approved" in labels:
        github.get(f"{endpoint}/ai-approved", method="DELETE")
    wanted = ({"ai-reviewed"} if state is not None else set()) | ({"ai-approved"} if approved else set())
    if wanted - labels:
        github.get(endpoint, {"labels": sorted(wanted - labels)})
    if approved:
        # PR updates can race with this job even though label writers share a
        # lock. Recheck the same approval scope after the add-label request.
        fresh = github.get(f"pulls/{number}")
        if (fresh["state"] != "open"
                or approval_scope(revision(fresh), fresh["title"], fresh.get("body") or "") != scope
                or reopen_epoch(github, number) != state["reopen_epoch"]):
            current = {label["name"] for label in github.pages(endpoint)}
            if "ai-approved" in current:
                github.get(f"{endpoint}/ai-approved", method="DELETE")


def prepare_review(snapshot, proposal):
    findings = validate(snapshot, proposal)
    result = proposal["result"]
    state = {"repository": REPOSITORY, "number": snapshot["number"], "request": snapshot["request"],
             "head": snapshot["revision"]["head"], "findings": findings,
             "reopen_epoch": snapshot["reopen_epoch"],
             "scope_digest": digest(approval_scope(snapshot["revision"], snapshot["title"], snapshot["description"])),
             "complete": result["complete"], "limitations": result["limitations"],
             "usefulness": result["usefulness"],
             "excluded_artifact_count": len(snapshot.get("excluded_artifacts", [])),
             "discussion_blockers": result["discussion_blockers"]}
    comments = []
    previous = {f["id"] for f in (snapshot["prior"] or {}).get("findings", [])}
    for finding in findings:
        if finding["id"] not in previous:
            comments.append({"path": finding["path"], "line": finding["line"],
                             "side": "RIGHT" if finding["revision"] == "head" else "LEFT",
                             "body": f"[{finding['priority']}] {review_text(finding['body'])}\n\n"
                                     f"<!-- akita-ai-review-finding:{finding['id']} -->"})
    state["inline_digest"] = digest(sorted([c["path"], c["line"], c["side"], c["body"]] for c in comments))
    return {"commit_id": snapshot["revision"]["head"], "event": "COMMENT",
            "body": review_body(state), "comments": comments}


def verify_review(github, number, review_id, body):
    state = previous_state([{"id": review_id, "own": True, "body": body}], number)
    verified = github.get(f"pulls/{number}/reviews/{review_id}")
    # The review-specific comments endpoint returns legacy position-only data.
    # Replies do not belong to the original publication manifest.
    inline = [c for c in github.pages(f"pulls/{number}/comments")
              if c.get("pull_request_review_id") == review_id and not c.get("in_reply_to_id")]
    actual = sorted([c.get("path"), c.get("original_line"), c.get("side"), c.get("body")] for c in inline)
    if (body != review_body(state) or verified.get("body") != body or verified.get("commit_id") != state["head"]
            or verified.get("state") != "COMMENTED"
            or verified.get("user", {}).get("login") != "github-actions[bot]"
            or verified.get("user", {}).get("type") != "Bot"
            or digest(actual) != state.get("inline_digest")):
        raise ReviewError("Published review could not be verified; do not blindly retry")
    return state


def latest_verified_review(github, number):
    reviews = [r for r in github.pages(f"pulls/{number}/reviews")
               if r.get("user", {}).get("login") == "github-actions[bot]"
               and r.get("user", {}).get("type") == "Bot"
               and (r.get("body") or "").startswith(MARKER)]
    latest = max(reviews, key=lambda r: (r.get("submitted_at") or "", r["id"]), default=None)
    if latest is None:
        return None
    return verify_review(github, number, latest["id"], latest["body"])


def reconcile(github, event):
    """Reconcile live metadata under the same job lock as publication."""
    number = event.get("pull_request", {}).get("number")
    if (os.environ.get("GITHUB_REPOSITORY", REPOSITORY) != REPOSITORY
            or event.get("repository", {}).get("full_name") != REPOSITORY
            or event.get("action") not in ("synchronize", "edited", "reopened")
            or type(number) is not int or number <= 0):
        raise ReviewError("Not an eligible PR update event")
    # The event may be delayed/replayed; its old PR contents are not current.
    # No author command, artifact, or model invocation is needed for recovery.
    try:
        state = latest_verified_review(github, number)
    except (ReviewError, KeyError, TypeError, ValueError):
        sync_labels(github, number, None)
        raise ReviewError("Latest review could not be verified; approval cleared") from None
    sync_labels(github, number, state)


def publish(github, event, snapshot, proposal):
    payload = prepare_review(snapshot, proposal)
    pr = authorize(github, event)
    if snapshot["number"] != pr["number"] or snapshot["request"] != event["comment"]["id"]:
        raise ReviewError("Review does not belong to this request")
    comments = discussions(github, pr["number"])
    if any(c["own"] and previous_state([c], pr["number"])["request"] == snapshot["request"]
           for c in comments):
        state = latest_verified_review(github, pr["number"])
        if state is None:
            raise ReviewError("No submitted review available for label recovery")
        sync_labels(github, pr["number"], state)
        return "already-published"
    if (revision(pr) != snapshot["revision"] or comments != snapshot["comments"]
            or pr["title"] != snapshot["title"] or (pr.get("body") or "") != snapshot["description"]
            or reopen_epoch(github, pr["number"]) != snapshot["reopen_epoch"]):
        raise ReviewError("Review is stale; post a new /ai-review command")
    posted = github.get(f"pulls/{pr['number']}/reviews", payload)
    state = verify_review(github, pr["number"], posted["id"], payload["body"])
    sync_labels(github, pr["number"], state)
    return "published"
