#!/usr/bin/env bash
# Cross-build x86_64 AVX2 tests and run them under Rosetta on Apple Silicon.
#
# Rosetta advertises AVX/AVX2 only when ROSETTA_ADVERTISE_AVX=1 (set by the runner).
# AVX-512 is not emulated; AVX-512-specific tests skip at runtime.
#
# Usage:
#   ./scripts/test-x86-rosetta.sh              # focused SIMD crates
#   ./scripts/test-x86-rosetta.sh --probe-only # print CPU feature detection only
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="$repo_root/scripts/rosetta-x86_64-runner.sh"
target="x86_64-apple-darwin"
probe_only=0

for arg in "$@"; do
  case "$arg" in
    --probe-only) probe_only=1 ;;
    -h | --help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "skip: host is $(uname -m); this script targets Apple Silicon + Rosetta" >&2
  exit 0
fi

if ! command -v arch >/dev/null 2>&1; then
  echo "error: \`arch\` not found" >&2
  exit 1
fi

chmod +x "$runner"

if ! rustup target list --installed | grep -qx "$target"; then
  echo "==> installing rust target $target"
  rustup target add "$target"
fi

export CARGO_TARGET_X86_64_APPLE_DARWIN_RUNNER="$runner"
export RUSTFLAGS="-C target-cpu=x86-64-v3"

probe_src="/tmp/akita_rosetta_probe_$$.rs"
probe_bin="/tmp/akita_rosetta_probe_$$"
trap 'rm -f "$probe_src" "$probe_bin"' EXIT

cat >"$probe_src" <<'EOF'
fn main() {
    println!("host build target: x86_64-apple-darwin (Rosetta runner)");
    println!("ROSETTA_ADVERTISE_AVX={}", std::env::var("ROSETTA_ADVERTISE_AVX").unwrap_or_else(|_| "<unset>".into()));
    println!("avx:     {}", std::is_x86_feature_detected!("avx"));
    println!("avx2:    {}", std::is_x86_feature_detected!("avx2"));
    println!("avx512f: {}", std::is_x86_feature_detected!("avx512f"));
    println!("avx512dq: {}", std::is_x86_feature_detected!("avx512dq"));
    println!("avx512bw: {}", std::is_x86_feature_detected!("avx512bw"));
    println!("bmi2:    {}", std::is_x86_feature_detected!("bmi2"));
}
EOF

echo "==> probing x86 CPU features under Rosetta"
rustc "$probe_src" -o "$probe_bin" --target "$target" -C target-cpu=x86-64-v3
probe_out="$("$runner" "$probe_bin")"
printf '%s\n' "$probe_out"

if [[ "$probe_only" -eq 1 ]]; then
  exit 0
fi

cd "$repo_root"

if ! grep -q 'avx2:    true' <<<"$probe_out"; then
  echo "error: Rosetta did not advertise AVX2; AVX tests would no-op" >&2
  echo "hint: ensure ROSETTA_ADVERTISE_AVX=1 in $runner" >&2
  exit 1
fi

echo "==> akita-algebra (ntt::avx)"
cargo test --target "$target" -p akita-algebra ntt::avx

echo "==> akita-field (packed)"
cargo test --target "$target" -p akita-field packed

echo "==> akita-prover (sparse_mul_acc SIMD parity)"
cargo test --target "$target" -p akita-prover sparse_mul_acc

echo "All Rosetta x86 AVX2 smoke tests passed."
