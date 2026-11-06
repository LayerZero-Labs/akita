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
      echo 'Usage: scripts/binary-source-full-size.sh --log-carriers 10|22|23 --shape binary-claim|binary-and-prime-claims|binary-and-prime-claims-plus-ordinary [--dev-protocol] [--plan-only]'
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$log_carriers" in 10|22|23) ;; *) echo 'log-carriers must be 10, 22 or 23' >&2; exit 2 ;; esac
case "$shape" in binary-claim|binary-and-prime-claims|binary-and-prime-claims-plus-ordinary) ;; *) echo "unknown shape: $shape" >&2; exit 2 ;; esac
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
rtk cargo build --release -p akita-pcs --example binary_source_full_size --no-default-features --features "$features"
echo "measurement_load_before_run=$(uptime)"
fixture_dir=$(mktemp -d "$PWD/.binary-full-size.XXXXXX")
trap 'rm -rf "$fixture_dir"' EXIT
if ((plan_only)); then
  "$CARGO_TARGET_DIR/release/examples/binary_source_full_size" \
    --log-carriers "$log_carriers" --shape "$shape" --fixture "$fixture_dir" --phase plan
  exit
fi
"$CARGO_TARGET_DIR/release/examples/binary_source_full_size" \
  --log-carriers "$log_carriers" --shape "$shape" --fixture "$fixture_dir" --phase prove | tee "$fixture_dir/prover.log"
python3 - "$log_carriers" "$shape" "$dev_protocol" "$fixture_dir/prover.log" <<'PY'
import pathlib
import re
import sys

log, shape, dev, output = sys.argv[1:]
protocol = "dev" if dev == "1" else "current"
# The dev protocol currently equals the current protocol: one pin per (log, shape).
rows = [line.split() for line in pathlib.Path("scripts/binary-source-full-size-digests.tsv").read_text().splitlines() if line and not line.startswith("#")]
expected = [row[2:] for row in rows if row[:2] == [log, shape]]
actual = re.findall(r"^proof bytes=(\d+) digest_algorithm=blake2b256 digest=([0-9a-f]{64})$", pathlib.Path(output).read_text(), re.MULTILINE)
if len(actual) != 1 or (expected and expected != [list(actual[0])]):
    raise SystemExit(f"proof digest mismatch: expected {expected}, observed {actual}")
print(f"proof_digest_pin matched={bool(expected)} protocol={protocol} log_carriers={log} shape={shape}")
PY
"$CARGO_TARGET_DIR/release/examples/binary_source_full_size" \
  --log-carriers "$log_carriers" --shape "$shape" --fixture "$fixture_dir" --phase verify
echo "measurement_load_after_run=$(uptime)"
