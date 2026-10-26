#!/usr/bin/env bash
set -euo pipefail

# Every a16z/jolt revision Akita pins must be an ancestor of Jolt `main`.
# A commit that lives only on an unmerged branch can be garbage-collected
# once that branch is squash-merged or deleted, and the pin then stops
# resolving.
#
# Scans every tracked Cargo.toml (rev = "...") and Cargo.lock
# (?rev=...#<commit>) in the root, fuzz, and profile workspaces. A Jolt
# dependency pinned by branch or tag, or by an abbreviated rev, is rejected:
# only a full commit hash names an immutable revision.
#
# The ancestry check needs Jolt main's full commit graph, not a shallow
# history: a depth-limited clone would report an old pin as unrelated. A
# treeless partial clone fetches every commit and nothing else.

JOLT_URL="${JOLT_URL:-https://github.com/a16z/jolt}"
JOLT_MAIN_BRANCH="${JOLT_MAIN_BRANCH:-main}"

cd "$(git rev-parse --show-toplevel)"

revs_file="$(mktemp)"
jolt_dir="$(mktemp -d)"
trap 'rm -rf "$revs_file" "$jolt_dir"' EXIT

jolt_source='github\.com/a16z/jolt(\.git)?'
status=0

while IFS= read -r file; do
  case "$file" in
    *Cargo.toml)
      while IFS= read -r line; do
        number="${line%%:*}"
        if [[ "$line" =~ rev[[:space:]]*=[[:space:]]*\"([0-9a-f]{40})\" ]]; then
          printf '%s %s:%s\n' "${BASH_REMATCH[1]}" "$file" "$number" >>"$revs_file"
        else
          echo "error: $file:$number pins a16z/jolt without a full 40-hex rev" >&2
          status=1
        fi
      done < <(grep -nE "git[[:space:]]*=[[:space:]]*\"https://$jolt_source\"" "$file" || true)
      ;;
    *Cargo.lock)
      while IFS= read -r line; do
        number="${line%%:*}"
        if [[ "$line" =~ \?rev=[0-9a-f]+#([0-9a-f]{40})\" ]]; then
          printf '%s %s:%s\n' "${BASH_REMATCH[1]}" "$file" "$number" >>"$revs_file"
        else
          echo "error: $file:$number resolves a16z/jolt without a pinned rev" >&2
          status=1
        fi
      done < <(grep -nE "source = \"git\+https://$jolt_source[?#]" "$file" || true)
      ;;
  esac
done < <(git ls-files -- 'Cargo.toml' '*/Cargo.toml' 'Cargo.lock' '*/Cargo.lock')

if [[ ! -s "$revs_file" ]]; then
  echo "no a16z/jolt pins found"
  exit "$status"
fi

git clone --quiet --bare --filter=tree:0 --single-branch --no-tags \
  --branch "$JOLT_MAIN_BRANCH" "$JOLT_URL" "$jolt_dir"
main_head="$(git -C "$jolt_dir" rev-parse "refs/heads/$JOLT_MAIN_BRANCH")"

while read -r rev; do
  if git -C "$jolt_dir" merge-base --is-ancestor "$rev" "$main_head" 2>/dev/null; then
    echo "ok: $rev is on a16z/jolt $JOLT_MAIN_BRANCH"
  else
    echo "error: $rev is not an ancestor of a16z/jolt $JOLT_MAIN_BRANCH ($main_head); pinned at:" >&2
    grep "^$rev " "$revs_file" | cut -d' ' -f2- | sed 's/^/  /' >&2
    status=1
  fi
done < <(cut -d' ' -f1 "$revs_file" | sort -u)

exit "$status"
