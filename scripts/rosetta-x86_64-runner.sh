#!/usr/bin/env bash
# Execute an x86_64 binary under Rosetta with AVX2 advertised to CPUID.
# Used as CARGO_TARGET_X86_64_APPLE_DARWIN_RUNNER for cross-target tests.
set -euo pipefail
export ROSETTA_ADVERTISE_AVX=1
exec arch -x86_64 "$@"
