"""Reserve review attempts durably before any job can access the model key."""

import re

from common import ReviewError, authorize

MAX_REVIEWS = 3
ATTEMPT_MARKER = "<!-- akita-ai-review-attempt:v1 request="


def reserve(github, event):
    # The workflow serializes this read/write transaction across the entire PR.
    number = authorize(github, event)["number"]
    requests = []
    for comment in github.pages(f"issues/{number}/comments"):
        user, body = comment.get("user", {}), comment.get("body", "") or ""
        if (user.get("login") != "github-actions[bot]" or user.get("type") != "Bot"
                or not body.startswith(ATTEMPT_MARKER)):
            continue
        match = re.match(re.escape(ATTEMPT_MARKER) + r"([1-9][0-9]*) -->(?:\n|$)", body)
        if not match:
            raise ReviewError("Invalid review attempt marker; refusing to reset quota")
        requests.append(int(match[1]))
    if event["comment"]["id"] in requests:
        print("This request already consumed a review attempt")
        return False
    if len(requests) >= MAX_REVIEWS:
        print(f"AI review limit reached ({MAX_REVIEWS} attempts per PR)")
        return False
    body = (f"{ATTEMPT_MARKER}{event['comment']['id']} -->\n\n"
            f"AI review attempt {len(requests) + 1}/{MAX_REVIEWS} reserved. "
            "Failed or incomplete attempts also count toward this PR's limit.")
    posted = github.get(f"issues/{number}/comments", {"body": body})
    if (posted.get("body") != body
            or posted.get("user", {}).get("login") != "github-actions[bot]"
            or posted.get("user", {}).get("type") != "Bot"):
        raise ReviewError("Could not verify review attempt reservation")
    return True
