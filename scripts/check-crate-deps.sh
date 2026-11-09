#!/usr/bin/env bash
set -euo pipefail

pkg="${1:?usage: scripts/check-crate-deps.sh <package> [forbidden-package ...]}"
shift

if ! cargo metadata --format-version 1 --no-deps | grep -q "\"name\":\"${pkg}\""; then
  echo "${pkg} not present yet; skipping dependency hygiene check"
  exit 0
fi

if [ "$#" -gt 0 ]; then
  forbidden=("$@")
else
  case "${pkg}" in
    akita-params)
      forbidden=(akita-types akita-prover akita-verifier akita-cpu-backend akita-pcs akita-setup akita-config akita-schedules akita-planner)
      ;;
    akita-sis-estimator)
      forbidden=(akita-types)
      ;;
    akita-verifier)
      forbidden=(akita-planner akita-prover akita-cpu-backend akita-pcs akita-metal jolt-metal akita-zk-prover akita-zk-verifier)
      ;;
    akita-prover)
      forbidden=(akita-planner akita-verifier akita-cpu-backend akita-setup akita-pcs akita-metal jolt-metal akita-zk-prover akita-zk-verifier)
      ;;
    akita-cpu-backend)
      forbidden=(akita-planner akita-verifier akita-setup akita-pcs akita-metal jolt-metal akita-zk-prover akita-zk-verifier)
      forbidden+=(akita-labinius-prover akita-labinius-verifier akita-labinius-pcs)
      ;;
    akita-config)
      forbidden=(akita-planner akita-prover akita-cpu-backend akita-verifier akita-pcs akita-metal jolt-metal akita-zk-prover akita-zk-verifier)
      ;;
    akita-schedules)
      forbidden=(akita-types akita-planner akita-config akita-prover akita-cpu-backend akita-verifier akita-setup akita-pcs)
      ;;
    akita-planner)
      # `akita-planner` is offline-only. It may use `akita-config` behind its
      # catalog-generation feature, but it must never pull in protocol-layer
      # prover/verifier/setup crates.
      forbidden=(akita-prover akita-cpu-backend akita-verifier akita-setup akita-pcs)
      ;;
    akita-metal)
      # Prover-only device kernels: never a verifier, planner or setup
      # dependency, and never the other way around (see the cases above).
      forbidden=(akita-verifier akita-planner akita-setup akita-pcs)
      ;;
    akita-setup)
      forbidden=(akita-verifier akita-pcs akita-metal jolt-metal akita-zk-prover akita-zk-verifier)
      ;;
    akita-zk-verifier)
      # Verifier-side zero-knowledge building blocks (#120): never a prover,
      # planner or setup dependency.
      forbidden=(akita-planner akita-prover akita-cpu-backend akita-setup akita-pcs akita-metal jolt-metal akita-zk-prover)
      ;;
    akita-labinius-verifier)
      forbidden=(akita-planner akita-prover akita-cpu-backend akita-setup akita-pcs akita-metal jolt-metal akita-zk-prover akita-labinius-prover)
      ;;
    akita-labinius-prover)
      forbidden=(akita-planner akita-verifier akita-cpu-backend akita-setup akita-pcs akita-metal jolt-metal akita-zk-prover akita-zk-verifier)
      ;;
    akita-labinius-pcs)
      # Offline artifact generation may opt into the planner; ordinary loading may not.
      forbidden=(akita-metal jolt-metal akita-zk-prover akita-zk-verifier)
      ;;
    akita-zk-prover)
      # Prover-side zero-knowledge building blocks (#120). Like akita-prover,
      # they reach CPU kernels only through backend traits.
      forbidden=(akita-planner akita-verifier akita-cpu-backend akita-setup akita-pcs akita-metal jolt-metal)
      ;;
    *)
      echo "no default forbidden dependency set for ${pkg}; pass forbidden packages explicitly" >&2
      exit 2
      ;;
  esac
fi

# Walk both the default-feature graph and the all-features graph so an
# opt-in feature can't sneak a forbidden crate into a downstream build.
# Verifier tests must also remain independent of the prover and PCS layers;
# otherwise test-only reverse edges can conceal a production layering error.
edge_kinds="normal"
if [ "${pkg}" = "akita-verifier" ] || [ "${pkg}" = "akita-zk-verifier" ] || [ "${pkg}" = "akita-labinius-verifier" ]; then
  edge_kinds="normal,dev"
fi
default_tree="$(cargo tree -p "${pkg}" --edges "${edge_kinds}")"
all_features_tree="$(cargo tree -p "${pkg}" --edges "${edge_kinds}" --all-features)"

for label in default all-features; do
  case "${label}" in
    default)      tree="${default_tree}" ;;
    all-features) tree="${all_features_tree}" ;;
  esac
  for candidate in "${forbidden[@]}"; do
    # The withdrawn standalone prover now calls the canonical CPU arithmetic.
    if [ "$#" -eq 0 ] && [ "$pkg" = "akita-labinius-prover" ] && [ "$candidate" = "akita-cpu-backend" ]; then
      continue
    fi
    if grep -qE "(^|[[:space:]])${candidate}([[:space:]]|$)" <<<"${tree}"; then
      echo "forbidden dependency found in ${pkg} (${label}): ${candidate}" >&2
      exit 1
    fi
  done
done

echo "${pkg} dependency hygiene check passed"

if [ "${pkg}" = "akita-prover" ]; then
  # Kernel contracts are backend-neutral under every feature combination.
  # Keep CPU storage, resource policy, routing, and portable witness artifacts
  # out of generic source as well as out of the production dependency graph.
  if rg --line-number --glob '*.rs' \
      '\b(CpuBackend|CpuProverConsumer|ComputeBackendSetup|LevelProveStacks|ProverConsumerFactory|SetupSourceFactory|NttCacheOwnerId|PortableCommitmentHandle|PreparedCommitmentResources)\b' \
      crates/akita-prover/src; then
    echo "CPU execution or witness storage leaked into generic prover contracts" >&2
    exit 1
  fi
  if rg --line-number --glob '*.rs' '\b(PackedSignedDigits|PackedNegativeBinary|read_portable|import_encoded)\b' crates/akita-prover/src; then
    echo "backend transfer readers or codecs leaked into generic prover code" >&2
    exit 1
  fi
fi

if [ "${pkg}" = "akita-labinius-pcs" ]; then
  # The union covers every runtime feature combination, including future features.
  runtime_features="$(cargo metadata --format-version 1 --no-deps | python3 -c '
import json
import sys
package = next(package for package in json.load(sys.stdin)["packages"]
               if package["name"] == "akita-labinius-pcs")
print(",".join(sorted(set(package["features"]) - {"labinius-catalog-gen"})))
')"
  for features in transcript-keccak labinius,transcript-keccak "$runtime_features"; do
    runtime_tree="$(cargo tree -p "$pkg" --edges normal --no-default-features --features "$features")"
    if grep -qE '(^|[[:space:]])akita-planner([[:space:]]|$)' <<<"$runtime_tree"; then
      echo "planner dependency found in runtime LaBinius PCS ($features)" >&2
      exit 1
    fi
  done
fi

if [ "${pkg}" = "akita-verifier" ]; then
  # Spongefish's `yolocrypto` feature exposes raw sponge state for prover-side
  # nonce search. Only prover features may enable it, so the verifier's
  # production graph must not, under default or all features. Dev edges are
  # excluded because verifier tests use the prover-side transcript.
  for feature_args in "" "--all-features"; do
    # shellcheck disable=SC2086
    spongefish_features="$(cargo tree -p "${pkg}" --edges normal,build,features --invert spongefish ${feature_args})"
    if grep -q 'spongefish feature "yolocrypto"' <<<"${spongefish_features}"; then
      echo "akita-verifier enables spongefish/yolocrypto (${feature_args:-default features})" >&2
      grep -B2 -A8 'yolocrypto' <<<"${spongefish_features}" >&2
      exit 1
    fi
  done
  echo "akita-verifier spongefish feature check passed"
fi
