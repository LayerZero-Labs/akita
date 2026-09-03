# Akita PCS

Akita is a high-performance, modular lattice polynomial commitment scheme with transparent setup and post-quantum security.

Akita is the public scheme name for this implementation and the intended repository/package name is `akita-pcs`.
The codebase is being decomposed into a focused `akita-*` crate family rather than remaining a single monolithic package.

The current workspace exposes the main ownership boundaries under `crates/`:

- `akita-error` owns the shared protocol error and reusable checked integer formulas for exact sizes, offsets, and ranges.
- Jolt's `jolt-field` package owns shared field arithmetic; `akita-serialization` and `akita-algebra` own Akita encoding, NTT, ring, and polynomial utilities.
- `akita-transcript`, `akita-challenges`, and `akita-sumcheck` own Fiat-Shamir transcripts, challenge sampling, and generic sumcheck machinery.
- `akita-types` owns shared proof, setup, schedule, layout, SIS, and commitment data shapes used by both roles.
- `akita-planner` is the `Cfg`-free offline schedule engine: candidate expansion, schedule-search DP, and external artifact emitter. It sits *below* `akita-config`.
- `akita-schedules` owns versioned schedule artifacts, semantic row validation, and validated owned runtime catalogs.
- `akita-config` owns concrete runtime config presets and the single `CommitmentConfig` policy trait. A config identifies the expected artifact family and policy but does not own rows.
- `akita-setup` owns config-backed setup construction and optional setup cache persistence.
- `akita-verifier` owns verifier replay without prover-only polynomial backends. It receives the validated schedule catalog explicitly.
- `akita-prover` owns commitment, proving, setup expansion, recursive/ring-switch witness construction, and polynomial backends.
- `akita-pcs` is the umbrella package: it owns the end-to-end `AkitaCommitmentScheme` orchestration, re-exports the broad public surface, and hosts examples, benches, and integration tests. (There is no separate `akita-scheme` crate.)

Verifier-only consumers should prefer the slim role crates directly:
`akita-verifier` for verification, `akita-types` for proof/setup/claim shapes,
and `akita-config` for concrete schedule/config policy. The umbrella
`akita-pcs` package is convenient for examples and end-to-end use, but it also
pulls in prover-facing APIs.

## Documentation

The [Akita Book](book/README.md) is the **canonical target** for narrative
documentation (how the scheme works, how to use it, and the foundations). Most
chapters are still stubs that cite source paths and specs to fold; until prose
lands, integrators should read the [Akita Book](book/README.md) (start with
[`book/src/how/architecture.md`](book/src/how/architecture.md)),
[`book/src/usage/commitment-api.md`](book/src/usage/commitment-api.md),
and [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md).
Build the book locally with `./scripts/serve-book.sh` (see
[`book/README.md`](book/README.md) for the toolchain). `AGENTS.md` is the
agent command runbook; `docs/` holds maintainer contracts (crate graph,
verifier contract, CI timing). `specs/` holds design records (lifecycle in
[`specs/PRUNING.md`](specs/PRUNING.md)). Documentation guardrails (CI + PR
comments) are in [`docs/documentation.md`](docs/documentation.md).

## External schedule artifacts

The runtime schedule contract is `TrustedScheduleCatalog`. A versioned JSON
artifact stores complete expanded rows. The application loads that artifact as
a trusted parameter for both proving and verification. A proof contains only
the 32 byte `OpeningScheduleSelection` row digest. It cannot provide or replace
schedule content.

Artifact loading checks the expected family, protocol epoch, config policy,
runtime challenge hooks, every expanded schedule invariant, every committed
profile, and every row digest. Honest prover key selection and verifier digest
selection then use the same owned catalog.

`AkitaCommitmentScheme<Cfg>` instance owns one validated catalog and uses it
for setup, commitment, proving, and verification. Direct verifier and prover
orchestration accept `&TrustedScheduleCatalog`, so an integration can load an
external trusted artifact without compiling its rows into Akita.

Load artifact bytes from application-owned storage and construct the scheme:

```rust
let bytes = std::fs::read("parameters/fp128_onehot.aks")?;
let scheme = AkitaCommitmentScheme::<fp128::OneHot>::from_schedule_artifact(&bytes)?;
```

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
directory and rejects any byte difference from the tracked artifacts.

## Lineage

Akita keeps the earlier implementation lineage explicit while giving the improved scheme its own name.
This is also the line where planned protocol improvements over the original design live: faster verifier-oriented reductions via matrix-claim delegation, smaller large-field proofs via field-size lowering, and efficient zero-knowledge techniques under the Whiteout design direction.

## Contributing

Major features and architectural changes should start with a short spec.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [specs/TEMPLATE.md](specs/TEMPLATE.md) for the review workflow.

## Acknowledgements

The CRT/NTT and small-prime arithmetic design in this repository is informed by the Labrador/Greyhound C implementation family. In particular, the pseudo-Mersenne profile uses moduli of the form `q = 2^k - offset`. Akita provides a Rust-native architecture and APIs, while drawing algorithmic inspiration from those implementations.
