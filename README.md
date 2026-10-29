# Akita PCS

Akita is a high-performance, modular lattice polynomial commitment scheme with transparent setup.
Its Module-SIS instances are priced at 169 bits of quantum Core-SVP cost under a lattice attack model, and its Fiat-Shamir challenges target a 128-bit rate per random-oracle query in the classical random-oracle model.
[Security status](book/src/intro.md#security-status) states what these targets cover.

Akita is the public scheme name for this implementation. The umbrella package is `akita-pcs`.
To try it, run `cargo run -p akita-pcs --release --example quickstart` and read the [quickstart](book/src/usage/quickstart.md).

Status: Akita is not yet a formally audited production release ([SECURITY.md](SECURITY.md)). It makes no backward-compatibility guarantees and does not provide zero knowledge.

The workspace is a family of focused `akita-*` crates under `crates/`. [`docs/crate-graph.md`](docs/crate-graph.md) has the full map and dependency graph. The main ownership boundaries are:

- `akita-error` owns the shared protocol error and reusable checked integer formulas for exact sizes, offsets, and ranges.
- Jolt's `jolt-field` package owns shared field arithmetic; `akita-serialization` and `akita-algebra` own Akita encoding, NTT, ring, and polynomial utilities.
- `akita-transcript`, `akita-challenges`, and `akita-sumcheck` own Fiat-Shamir transcripts, challenge sampling, and generic sumcheck machinery.
- `akita-params` owns parameter geometry, sizing, SIS tables, schedules, witness layout, dispatch, and grinding plans.
- `akita-types` owns proof, setup, and claim wire values, shared protocol math, and native transcript replay used by both roles.
- `akita-sis-estimator` owns offline scalar SIS attack-cost estimation and table certification.
- `akita-planner` is the `Cfg`-free offline schedule engine: candidate expansion, schedule-search DP, and external artifact emitter. Runtime crates never depend on it; its `catalog-gen` feature pulls in `akita-config` to emit artifacts.
- `akita-schedules` owns versioned schedule artifacts, semantic row validation, and validated owned runtime catalogs.
- `akita-config` owns concrete runtime config presets and the single `CommitmentConfig` policy trait. A config identifies the expected artifact family and policy but does not own rows.
- `akita-setup` owns requirement-backed setup construction and optional setup cache persistence.
- `akita-verifier` owns verifier replay without prover-only polynomial backends. It receives the validated schedule catalog explicitly.
- `akita-prover` owns backend-neutral protocol sequencing, proof assembly, and the opaque backend contracts.
- `akita-cpu-backend` owns the CPU sources, commitments, witness arithmetic, and caches behind those contracts.
- `akita-pcs` is the umbrella package: it owns the end-to-end `AkitaCommitmentScheme` orchestration, re-exports the common commit, prove, and verify types, and hosts examples, benches, and integration tests. (There is no separate `akita-scheme` crate.)

Verifier-only consumers should prefer the slim role crates directly:
`akita-verifier` for verification, `akita-types` for proof/setup/claim shapes,
and `akita-config` for concrete schedule/config policy. The umbrella
`akita-pcs` package is convenient for examples and end-to-end use, but it also
pulls in prover-facing APIs.

## Documentation

The [Akita Book](book/README.md) is the **canonical target** for narrative
documentation (how the scheme works, how to use it, and the foundations).
Integrators should start with the
[quickstart](book/src/usage/quickstart.md), the
[integration guide](book/src/usage/integration.md), and
[`book/src/how/architecture.md`](book/src/how/architecture.md). Recursion users
should also read [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md).
Build the book locally with `./scripts/serve-book.sh` (see
[`book/README.md`](book/README.md) for the toolchain). `AGENTS.md` is the
agent command runbook; `docs/` holds maintainer contracts (crate graph,
verifier contract, CI timing). `specs/` holds design records (lifecycle in
[`specs/PRUNING.md`](specs/PRUNING.md)). Documentation guardrails (CI + PR
comments) are in [`docs/documentation.md`](docs/documentation.md).

## External schedule artifacts

The runtime schedule contract is `TrustedScheduleCatalog<Cfg>`. A versioned
JSON artifact stores complete expanded rows. The application loads that
artifact as a config-bound trusted parameter for both proving and verification.
The public statement carries the 32 byte `OpeningScheduleSelection` row digest
beside the proof bytes. Neither can provide or replace schedule content.

Artifact loading checks the expected family, protocol epoch, config policy,
runtime challenge hooks, every expanded schedule invariant, and every committed
profile. It derives each row digest and rejects duplicate identities,
noncanonical JSON, and noncanonical digest order. Honest prover key selection and verifier digest
selection then use the same owned catalog.

An `AkitaCommitmentScheme<Cfg>` instance owns one validated catalog and uses it
for setup, commitment, proving, and verification. Direct verifier construction
takes a `TrustedScheduleCatalog<Cfg>` by value and prover orchestration borrows
one, so an integration can load an external trusted artifact without compiling
its rows into Akita.

Load artifact bytes from application-owned storage and supply the expected
chunking from trusted application configuration. This example uses a single
chunk:

```rust
use akita_config::proof_optimized::fp128;
use akita_params::ChunkedWitnessCfg;
use akita_pcs::AkitaCommitmentScheme;

let bytes = std::fs::read("parameters/fp128_onehot.aks")?;
let chunking = ChunkedWitnessCfg::default_non_chunked();
let scheme = AkitaCommitmentScheme::<fp128::OneHot>::from_schedule_artifact(&bytes, chunking)?;
```

Admission checks that the artifact matches this geometry. Subsequent operations
use the admitted `scheme.schedules().policy()`. See
[runtime witness chunking](book/src/how/configuration.md#runtime-witness-chunking)
for selecting a different chunk count and number of chunked folds.

Git tracks deterministic family artifacts under `artifacts/schedules/`.
Regenerate them after changing planner policy, candidate search, or artifact
structure:

```bash
scripts/generate-schedule-artifacts.sh
```

For a faster planner iteration loop, pass one or more family names, for example
`scripts/generate-schedule-artifacts.sh fp32_dense`. Run the unfiltered command
before committing planner changes.

The schedule-artifact drift job regenerates every family into a temporary
directory and rejects any byte difference from the tracked artifacts. It also
compares the regenerated catalog snapshot with `artifacts/schedule-catalog.tsv`.

## Lineage

Akita keeps the earlier implementation lineage explicit while giving the improved scheme its own name.
The [roadmap](book/src/roadmap/roadmap.md) lists what is implemented and what is planned.

## Contributing

Major features and architectural changes should start with a short spec.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [specs/TEMPLATE.md](specs/TEMPLATE.md) for the review workflow.

## Acknowledgements

The CRT/NTT and small-prime arithmetic design in this repository is informed by the Labrador/Greyhound C implementation family. In particular, the pseudo-Mersenne profile uses moduli of the form `q = 2^k - offset`. Akita provides a Rust-native architecture and APIs, while drawing algorithmic inspiration from those implementations.
