# Akita PCS

Akita is a high-performance, modular lattice polynomial commitment scheme with transparent setup and post-quantum security.

Akita is the public scheme name for this implementation and the intended repository/package name is `akita-pcs`.
The codebase is being decomposed into a focused `akita-*` crate family rather than remaining a single monolithic package.

The current workspace exposes the main ownership boundaries under `crates/`:

- `akita-field`, `akita-serialization`, and `akita-algebra` own foundational arithmetic, encoding, NTT, ring, and polynomial utilities.
- `akita-transcript`, `akita-challenges`, and `akita-sumcheck` own Fiat-Shamir transcripts, challenge sampling, and generic sumcheck machinery.
- `akita-types` owns shared proof, setup, schedule, layout, SIS, and commitment data shapes used by both roles.
- `akita-planner` is the `Cfg`-free schedule engine: generated table types, on-demand expansion, catalog identity validation, the schedule-search DP, and the offline table emitter. It sits *below* `akita-config`.
- `akita-schedules` stores the tracked generated schedule tables. Cargo features select which tables a build includes.
- `akita-config` owns concrete runtime config presets and the single `CommitmentConfig` policy trait. It depends on `akita-schedules` (`resolve_catalog_row_for_key` delegates to strict generated-catalog resolution).
- `akita-setup` owns config-backed setup construction and optional setup cache persistence.
- `akita-verifier` owns verifier replay without prover-only polynomial backends. It is directly `<Cfg>`-generic (depends on `akita-config`) and reaches generated schedule expansion transitively.
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

## Generated Schedules

Builds that use schedules read generated family modules from
`crates/akita-schedules/src/generated/`. Git tracks these deterministic planner
outputs. Ordinary builds, formatting, Clippy, tests, and profile jobs use them
as checked in. Regenerate them after changing planner policy, candidate search,
or generated catalog structure:

```bash
scripts/generate-schedule-tables.sh
```

The dedicated all-schedules drift job regenerates the complete catalog and
rejects any byte difference from the tracked files.

## Lineage

Akita keeps the earlier implementation lineage explicit while giving the improved scheme its own name.
This is also the line where planned protocol improvements over the original design live: faster verifier-oriented reductions via matrix-claim delegation, smaller large-field proofs via modulus switching and field-size lowering, and efficient zero-knowledge techniques under the Whiteout design direction.

## Contributing

Major features and architectural changes should start with a short spec.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [specs/TEMPLATE.md](specs/TEMPLATE.md) for the review workflow.

## Acknowledgements

The CRT/NTT and small-prime arithmetic design in this repository is informed by the Labrador/Greyhound C implementation family. In particular, the pseudo-Mersenne profile uses moduli of the form `q = 2^k - offset`. Akita provides a Rust-native architecture and APIs, while drawing algorithmic inspiration from those implementations.
