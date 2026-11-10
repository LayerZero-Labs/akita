#!/usr/bin/env bash
# Complete experimental binary grouped opening; never part of default tests.
set -euo pipefail
cd "$(dirname "$0")/.."

log_carriers=22
shape=binary-and-prime-claims
dev_protocol=0
plan_only=0
while (($#)); do
  case "$1" in
    --log-carriers) log_carriers="${2:?missing log-carriers}"; shift 2 ;;
    --shape) shape="${2:?missing shape}"; shift 2 ;;
    --dev-protocol) dev_protocol=1; shift ;;
    --plan-only) plan_only=1; shift ;;
    --help|-h)
      echo 'Usage: scripts/binary-full-size.sh --log-carriers 10|22|23 --shape binary-claim|binary-and-prime-claims|binary-and-prime-claims-plus-ordinary [--dev-protocol] [--plan-only]'
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$log_carriers" in 10|22|23) ;; *) echo 'log-carriers must be 10, 22 or 23' >&2; exit 2 ;; esac
case "$shape" in binary-claim|binary-and-prime-claims|binary-and-prime-claims-plus-ordinary) ;; *) echo "unknown shape: $shape" >&2; exit 2 ;; esac
: "${CARGO_TARGET_DIR:?use the supplied CARGO_TARGET_DIR}"
features=parallel,binary-measurements,transcript-blake2b,logging-transcript
if ((dev_protocol)); then features+=,dev-protocol; fi
echo "measurement_commit=$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'measurement_tree=dirty'
else
  echo 'measurement_tree=clean'
fi
echo "measurement_host=$(uname -sm)"
echo "measurement_load_before_build=$(uptime)"
rtk cargo build --release -p akita-pcs --example binary_full_size --no-default-features --features "$features"
echo "measurement_load_before_run=$(uptime)"
fixture_dir=$(mktemp -d "$PWD/.binary-full-size.XXXXXX")
trap 'rm -rf "$fixture_dir"' EXIT
if ((plan_only)); then
  "$CARGO_TARGET_DIR/release/examples/binary_full_size" \
    --log-carriers "$log_carriers" --shape "$shape" --fixture "$fixture_dir" --phase plan
  exit
fi
"$CARGO_TARGET_DIR/release/examples/binary_full_size" \
  --log-carriers "$log_carriers" --shape "$shape" --fixture "$fixture_dir" --phase prove | tee "$fixture_dir/prover.log"
"$CARGO_TARGET_DIR/release/examples/binary_full_size" \
  --log-carriers "$log_carriers" --shape "$shape" --fixture "$fixture_dir" --phase verify
echo "measurement_load_after_run=$(uptime)"
