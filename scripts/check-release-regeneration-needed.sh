#!/usr/bin/env bash
# Decide whether the anchored releases must be regenerated.
#
# usage: scripts/check-release-regeneration-needed.sh <event-name> [<base-sha>]
#
# A release must be regenerated whenever its description, manifest, or the
# anchor changes, when Cargo.lock or rust-toolchain.toml changes, or when any
# crate in the generator's dependency closure changes. The closure covers
# nearly all of Akita, so most changes trigger it. Pushes, scheduled runs, and
# manual runs always regenerate. Prints `needed=true|false`, also to
# $GITHUB_OUTPUT when set.
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

# Every workspace crate the generator links, under any feature set.
closure_dirs=()
while IFS= read -r dir; do
  closure_dirs+=("${dir#"$repo_root"/}/")
done < <(cargo tree -p akita-preprocess --all-features --edges normal,build --prefix none \
  | sed -n "s|.* (\(${repo_root}/[^)]*\)).*|\1|p" | sort -u)

needed=false
while IFS= read -r path; do
  case "$path" in
    params/* | artifacts/schedules/* | crates/akita-anchors/* | Cargo.lock | rust-toolchain.toml \
      | .github/workflows/preprocessing-releases.yml | scripts/check-release-regeneration-needed.sh)
      needed=true
      ;;
  esac
  for dir in "${closure_dirs[@]}"; do
    if [[ "$path" == "$dir"* ]]; then
      needed=true
    fi
  done
  if [ "$needed" = true ]; then
    break
  fi
done < <(git diff --name-only "$base"...HEAD)

emit "$needed"
