"""Trusted workflow entry point; no PR-controlled arguments or executable content."""

import argparse
import os
import sys

from collect import collect
from common import GitHub, ReviewError, load_json, save_json
from model import review
from publish import publish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "review", "publish"))
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    root = args.directory
    os.makedirs(root, exist_ok=True)
    snapshot_path, result_path = f"{root}/snapshot.json", f"{root}/result.json"
    if args.stage == "collect":
        snapshot = collect(GitHub(os.environ["GH_TOKEN"]), load_json(os.environ["GITHUB_EVENT_PATH"]))
        if snapshot is not None:
            save_json(snapshot_path, snapshot)
            with open(os.environ["GITHUB_OUTPUT"], "a") as handle:
                handle.write("ready=true\n")
        print("Collection complete" if snapshot else "Request already reviewed")
    elif args.stage == "review":
        proposal = review(load_json(snapshot_path), os.environ["OPENAI_API_KEY"],
                          os.environ.get("AI_REVIEW_MODEL", "gpt-5.4"))
        save_json(result_path, proposal)
        print("Model review complete")
    else:
        status = publish(GitHub(os.environ["GH_TOKEN"]), load_json(os.environ["GITHUB_EVENT_PATH"]),
                         load_json(snapshot_path), load_json(result_path))
        print(status)


if __name__ == "__main__":
    try:
        main()
    except ReviewError as exc:
        print(f"AI review stopped: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        # Remote content must never get into a traceback in public Actions logs.
        print("AI review stopped: invalid data or unexpected failure", file=sys.stderr)
        sys.exit(1)
