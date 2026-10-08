"""Fixed-origin HTTP and bounded review data. Never log remote bodies or tokens."""

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import zlib

REPOSITORY = "LayerZero-Labs/akita"
WORKFLOW = ".github/workflows/ai-review.yml"
MAX_BYTES = 32 * 1024 * 1024
# Every review-state marker starts with this. v1 holds base64 JSON (the first
# release); v2 holds base64 zlib-compressed JSON, so the carried-over finding
# history fits GitHub's comment limit.
MARKER = "<!-- akita-ai-review:v"


class ReviewError(Exception):
    """Safe, locally generated error suitable for a public workflow log."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ReviewError("HTTP redirect refused")


def request(origin, path, token, payload=None, method=None):
    if origin not in ("https://api.github.com", "https://api.openai.com"):
        raise ReviewError("Unknown API origin")
    if not path.startswith("/") or path.startswith("//"):
        raise ReviewError("Invalid API path")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if origin == "https://api.github.com":
        headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    req = urllib.request.Request(origin + path, headers=headers, method=method,
                                 data=None if payload is None else json.dumps(payload).encode())
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=180) as response:
            data = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise ReviewError(f"API request failed (HTTP {exc.code})") from None
    except (urllib.error.URLError, TimeoutError):
        raise ReviewError("API request failed or timed out; no automatic write retry") from None
    if len(data) > MAX_BYTES:
        raise ReviewError("API response exceeds size limit")
    return json.loads(data)


class GitHub:
    def __init__(self, token):
        self.token = token

    def get(self, path, payload=None, method=None):
        return request("https://api.github.com", f"/repos/{REPOSITORY}/{path}", self.token, payload, method)

    def pages(self, path):
        items = []
        for page in range(1, 101):
            batch = self.get(f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
            if not isinstance(batch, list):
                raise ReviewError("Unexpected paginated response")
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise ReviewError("Pagination limit reached; refusing incomplete review")

    def writer(self, user):
        if user.get("type") != "User" or not re.fullmatch(r"[A-Za-z0-9-]+", user.get("login", "")):
            return False
        permission = self.get(f"collaborators/{user['login']}/permission")
        return (permission.get("user", {}).get("id") == user.get("id")
                and permission.get("permission") in ("admin", "maintain", "write"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ReviewError("Invalid commit SHA")
    return value


def encode_state(state):
    encoded = base64.b64encode(zlib.compress(json.dumps(state).encode(), 9)).decode()
    return f"{MARKER}2 {encoded} -->"


def decode_state(body):
    """Return (marker, state) from the start of a review body.

    Callers compare the marker as published rather than re-encoding the state,
    because zlib output can change between library versions.
    """
    match = re.match(re.escape(MARKER) + r"([12]) ([A-Za-z0-9+/=]+) -->(?:\n|$)", body)
    if not match:
        raise ReviewError("Malformed review state")
    try:
        data = base64.b64decode(match[2], validate=True)
        if match[1] == "2":
            inflater = zlib.decompressobj()
            data = inflater.decompress(data, MAX_BYTES)
            if inflater.unconsumed_tail or not inflater.eof:
                raise ValueError()
        return match[0].rstrip("\n"), json.loads(data)
    except (ValueError, zlib.error):
        raise ReviewError("Malformed review state") from None


def load_json(path):
    with open(path, "rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ReviewError("Review artifact exceeds size limit")
    return json.loads(data)


def save_json(path, value):
    data = json.dumps(value, ensure_ascii=True)
    if len(data.encode()) > MAX_BYTES:
        raise ReviewError("Review artifact exceeds size limit")
    with open(path, "w") as handle:
        handle.write(data)


def authorize(github, event):
    if (os.environ.get("GITHUB_REPOSITORY", REPOSITORY) != REPOSITORY
            or event.get("repository", {}).get("full_name") != REPOSITORY
            or event.get("action") != "created"
            or not event.get("issue", {}).get("pull_request")):
        raise ReviewError("Not an eligible PR comment event")
    number = event["issue"]["number"]
    comment_id = event.get("comment", {}).get("id")
    if type(number) is not int or type(comment_id) is not int:
        raise ReviewError("Invalid event identifiers")
    pr = github.get(f"pulls/{number}")
    comment = github.get(f"issues/comments/{comment_id}")
    if (comment.get("body", "") != "/ai-review"
            or comment.get("user", {}).get("type") != "User"
            or comment.get("issue_url") != f"https://api.github.com/repos/{REPOSITORY}/issues/{number}"
            or comment.get("user", {}).get("id") != pr.get("user", {}).get("id")
            or comment.get("user", {}).get("id") != event["comment"].get("user", {}).get("id")
            or event.get("sender", {}).get("id") != comment.get("user", {}).get("id")
            or pr.get("state") != "open"
            or (pr.get("head", {}).get("repo") or {}).get("full_name") != REPOSITORY
            or (pr.get("base", {}).get("repo") or {}).get("full_name") != REPOSITORY
            or not github.writer(comment["user"])):
        raise ReviewError("Review request is not authorized")
    return pr


def revision(pr):
    return {"head": sha(pr["head"]["sha"]), "base": sha(pr["base"]["sha"]),
            "base_ref": pr["base"]["ref"], "head_ref": pr["head"]["ref"]}


def reopen_epoch(github, number):
    """Use durable GitHub event IDs; identical code can still have been reopened."""
    ids = [event["id"] for event in github.pages(f"issues/{number}/events")
           if event.get("event") == "reopened"]
    if any(type(value) is not int or value <= 0 for value in ids):
        raise ReviewError("Invalid PR lifecycle event")
    return max(ids, default=0)
