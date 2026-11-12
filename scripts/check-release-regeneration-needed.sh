#!/usr/bin/env bash
# Decide whether the anchored releases must be regenerated.
#
# usage: scripts/check-release-regeneration-needed.sh <event-name> [<base-sha>]
#
# A release must be regenerated whenever its description, manifest, or the
# anchor changes, when a file a description names (a catalog or a universe
# slot file) changes, when the root Cargo.toml, Cargo.lock, .cargo/config.toml,
# or rust-toolchain.toml changes, or when any crate in the generator's
# dependency closure changes. The closure covers nearly all of Akita, so most
# changes trigger it. Pushes, scheduled runs, and manual runs always
# regenerate. Prints `needed=true|false`, also to $GITHUB_OUTPUT when set.
#
# The script fails closed: when the closure, the named inputs, or the changed
# paths cannot be computed, it exits with an error instead of skipping.
set -euo pipefail

event="${1:?usage: $0 <event-name> [<base-sha>]}"
base="${2:-}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

emit() {
  echo "needed=$1"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "needed=$1" >> "$GITHUB_OUTPUT"
  fi
}

if [ "$event" != "pull_request" ] || [ -z "$base" ]; then
  emit true
  exit 0
fi

# Every workspace crate the generator links, under any feature set. Command
# substitution in an assignment aborts under `set -e` when cargo fails.
tree="$(cargo tree -p akita-preprocess --all-features --edges normal,build --prefix none)"
closure_dirs=()
while IFS= read -r dir; do
  [ -n "$dir" ] && closure_dirs+=("${dir#"$repo_root"/}/")
done <<< "$(printf '%s\n' "$tree" | sed -n "s|.* (\(${repo_root}/[^)]*\)).*|\1|p" | sort -u)"
case " ${closure_dirs[*]} " in
  *" crates/akita-preprocess/ "*) ;;
  *)
    echo "error: the generator's dependency closure does not contain crates/akita-preprocess/" >&2
    exit 1
    ;;
esac

# Every file a description names. The generator rejects any such path that is
# not a normalized path relative to the repository root.
named_inputs="$(jq -r '.sets[] | .catalog, ((.program_rows // {}).universe_slots // [] | .[])' params/*/description.json)"

changed="$(git diff --name-only "$base"...HEAD)"

needed=false
while IFS= read -r path; do
  [ -n "$path" ] || continue
  case "$path" in
    params/* | artifacts/schedules/* | crates/akita-anchors/* | Cargo.toml | Cargo.lock \
      | .cargo/config.toml | rust-toolchain.toml | .github/workflows/preprocessing-releases.yml \
      | scripts/check-release-regeneration-needed.sh)
      needed=true
      ;;
  esac
  for dir in "${closure_dirs[@]}"; do
    if [[ "$path" == "$dir"* ]]; then
      needed=true
    fi
  done
  if grep -qxF -- "$path" <<< "$named_inputs"; then
    needed=true
  fi
  if [ "$needed" = true ]; then
    break
  fi
done <<< "$changed"

emit "$needed"
