#!/usr/bin/env bash
# Complete experimental binary grouped opening; never part of default tests.
set -euo pipefail
cd "$(dirname "$0")/.."

log_carriers=22
shape=binary-and-prime-claims
dev_protocol=0
field=p128
all=0
output_dir=
reference_dir=
replay_catalogs=0
plan_only=0
check_catalogs=0
while (($#)); do
  case "$1" in
    --field) field="${2:?missing field}"; shift 2 ;;
    --all) all=1; shift ;;
    --output-dir) output_dir="${2:?missing output-dir}"; shift 2 ;;
    --reference-dir) reference_dir="${2:?missing reference-dir}"; shift 2 ;;
    --replay-catalogs) replay_catalogs=1; shift ;;
    --log-carriers) log_carriers="${2:?missing log-carriers}"; shift 2 ;;
    --shape) shape="${2:?missing shape}"; shift 2 ;;
    --dev-protocol) dev_protocol=1; shift ;;
    --plan-only) plan_only=1; shift ;;
    --check-catalogs) check_catalogs=1; shift ;;
    --help|-h)
      echo 'Usage: scripts/binary-source-full-size.sh [--field p128|p64] --shape SHAPE'
      echo '  --log-carriers 10|22|23 [--dev-protocol] [--plan-only]'
      echo 'Shapes: binary-claim, prime-claim (P64), binary-and-prime-claims,'
      echo '  binary-and-prime-claims-plus-ordinary.'
      echo 'P64: --all runs logs 22/23, all shapes and both modes; --shape all selects all shapes.'
      echo 'P64: --check-catalogs freshly plans every installed catalog and requires byte equality.'
      echo 'P64: --output-dir DIR retains fresh fixtures.'
      echo 'P64: --reference-dir DIR compares five artifacts per case.'
      echo 'P64: --replay-catalogs uses installed catalogs; references are never refreshed.'
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$field" in p128|p64) ;; *) echo "unknown field: $field" >&2; exit 2 ;; esac
if ((check_catalogs)); then
  if [[ "$field" != p64 ]] || ((replay_catalogs)); then
    echo '--check-catalogs requires p64 and forbids replayed catalogs' >&2; exit 2
  fi
  all=1
  plan_only=1
fi
if [[ "$field" = p64 ]]; then
  case "$log_carriers" in
    22|23) ;;
    *) echo 'P64 requires log-carriers 22 or 23' >&2; exit 2 ;;
  esac
  case "$shape" in
    all|binary-claim|prime-claim|binary-and-prime-claims|\
    binary-and-prime-claims-plus-ordinary) ;;
    *) echo "unknown shape: $shape" >&2; exit 2 ;;
  esac
  : "${CARGO_TARGET_DIR:?use the supplied CARGO_TARGET_DIR}"
  logs=("$log_carriers")
  shapes=("$shape")
  modes=(current)
  if ((dev_protocol)); then modes=(dev); fi
  if ((all)); then logs=(22 23); modes=(current dev); fi
  if ((all)) || [[ "$shape" = all ]]; then
    shapes=(
      binary-claim prime-claim binary-and-prime-claims binary-and-prime-claims-plus-ordinary
    )
  fi
  if [[ -z "$output_dir" ]]; then output_dir=$(mktemp -d "${TMPDIR:-/tmp}/akita-p64.XXXXXX"); fi
  mkdir -p "$output_dir"
  # Fresh planning must not silently consume a catalog from an environment cache.
  unset BINARY_SOURCE_CATALOG_CACHE
  echo "measurement_commit=$(git rev-parse HEAD)"
  echo "measurement_host=$(uname -sm)"
  echo "measurement_output_dir=$output_dir"
  git status --short
  for mode in "${modes[@]}"; do
    features=parallel,binary-source-measurements,transcript-keccak
    suffix=
    if [[ "$mode" = dev ]]; then features+=,dev-protocol; suffix=-dev; fi
    if ((check_catalogs)); then
      installed_files=(crates/akita-pcs/tests/fixtures/binary-source-p64"$suffix"/*.json)
      if ((${#installed_files[@]} != ${#logs[@]} * ${#shapes[@]})); then
        echo "installed catalog roster differs from planned cases: $mode" >&2; exit 1
      fi
    fi
    rtk cargo build --release -p akita-pcs --example binary_source_full_size \
      --no-default-features --features "$features"
    for log in "${logs[@]}"; do
      for selected_shape in "${shapes[@]}"; do
        fixture_dir="$output_dir/$mode-$log-$selected_shape"
        # Refuse any existing case, even if empty, to prevent accidental reuse.
        mkdir "$fixture_dir"
        if ((replay_catalogs)); then
          cp "crates/akita-pcs/tests/fixtures/binary-source-p64$suffix/$log-$selected_shape.json" \
            "$fixture_dir/native-catalog.json"
          phases=(prove verify)
        else
          phases=(plan prove verify)
        fi
        if ((plan_only)); then phases=(plan); fi
        for phase in "${phases[@]}"; do
          "$CARGO_TARGET_DIR/release/examples/binary_source_full_size" \
            --field p64 --log-carriers "$log" --shape "$selected_shape" \
            --fixture "$fixture_dir" --phase "$phase" > "$fixture_dir/$phase.log" 2>&1
        done
        if ((check_catalogs)); then
          installed="crates/akita-pcs/tests/fixtures/binary-source-p64$suffix"
          installed+="/$log-$selected_shape.json"
          # This gate never installs bytes. Preserve both files and the diff on failure.
          if ! cmp "$installed" "$fixture_dir/native-catalog.json"; then
            diff -u "$installed" "$fixture_dir/native-catalog.json" || true
            echo "stale installed catalog: $installed" >&2
            exit 1
          fi
          echo "fresh_catalog matched=$installed"
        fi
        if ((plan_only)); then continue; fi
        python3 -I - "$log" "$selected_shape" "$mode" "$fixture_dir" "$reference_dir" <<'PYP64'
import hashlib
from pathlib import Path
import sys

log, shape, mode, directory, reference = sys.argv[1:]
directory = Path(directory)
if "verification passed=true" not in (directory / "verify.log").read_text():
    raise SystemExit(f"unverified fixture: {directory}")
proof = (directory / "proof.bin").read_bytes()
print(f"proof_sha3_256={hashlib.sha3_256(proof).hexdigest()} bytes={len(proof)}")
if reference:
    baseline = Path(reference) / directory.name
    if baseline.resolve() == directory.resolve():
        raise SystemExit("independent reproduction requires distinct evidence roots")
    if "verification passed=true" not in (baseline / "verify.log").read_text():
        raise SystemExit(f"unverified reference: {baseline}")
    for name in (
        "native-catalog.json", "commitments.bin", "verifier.bin", "claims.json", "proof.bin"
    ):
        if (directory / name).read_bytes() != (baseline / name).read_bytes():
            raise SystemExit(f"independent reproduction mismatch: {directory / name}")
    print(f"independent_artifacts matched=5 case={directory.name}")
PYP64
      done
    done
  done
  exit
fi
if ((all || replay_catalogs)) || [[ -n "$output_dir" || -n "$reference_dir" ]]; then
  echo '--all, --output-dir, --reference-dir and --replay-catalogs require --field p64' >&2
  exit 2
fi
case "$log_carriers" in 10|22|23) ;; *) echo 'log-carriers must be 10, 22 or 23' >&2; exit 2 ;; esac
case "$shape" in
  binary-claim|binary-and-prime-claims|binary-and-prime-claims-plus-ordinary) ;;
  *) echo "unknown shape: $shape" >&2; exit 2 ;;
esac
: "${CARGO_TARGET_DIR:?use the supplied CARGO_TARGET_DIR}"
features=parallel,binary-source-measurements,transcript-blake2b,logging-transcript
if ((dev_protocol)); then features+=,dev-protocol; fi
echo "measurement_commit=$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'measurement_tree=dirty'
else
  echo 'measurement_tree=clean'
fi
echo "measurement_host=$(uname -sm)"
echo "measurement_load_before_build=$(uptime)"
rtk cargo build --release -p akita-pcs --example binary_source_full_size \
      --no-default-features --features "$features"
echo "measurement_load_before_run=$(uptime)"
fixture_dir=$(mktemp -d "$PWD/.binary-full-size.XXXXXX")
trap 'rm -rf "$fixture_dir"' EXIT
if ((plan_only)); then
  "$CARGO_TARGET_DIR/release/examples/binary_source_full_size" \
    --log-carriers "$log_carriers" --shape "$shape" --fixture "$fixture_dir" --phase plan
  exit
fi
"$CARGO_TARGET_DIR/release/examples/binary_source_full_size" \
  --log-carriers "$log_carriers" --shape "$shape" --fixture "$fixture_dir" --phase prove \
  | tee "$fixture_dir/prover.log"
"$CARGO_TARGET_DIR/release/examples/binary_source_full_size" \
  --log-carriers "$log_carriers" --shape "$shape" --fixture "$fixture_dir" --phase verify
echo "measurement_load_after_run=$(uptime)"
