# Akita PCS

Akita is a high-performance, modular lattice polynomial commitment scheme with transparent setup and post-quantum security.

Akita is the public scheme name for this implementation and the intended repository/package name is `akita-pcs`.
The codebase is being decomposed into a focused `akita-*` crate family rather than remaining a single monolithic package.

The current workspace exposes the main ownership boundaries under `crates/`:

- `akita-field`, `akita-serialization`, and `akita-algebra` own foundational arithmetic, encoding, NTT, ring, and polynomial utilities.
- `akita-transcript`, `akita-challenges`, and `akita-sumcheck` own Fiat-Shamir transcripts, challenge sampling, and generic sumcheck machinery.
- `akita-types` owns shared proof, setup, schedule, layout, SIS, and commitment data shapes used by both roles.
- `akita-planner` is the `Cfg`-free schedule engine: generated table types, on-demand expansion, catalog identity validation, and the schedule-search DP. It sits *below* `akita-config`.
- `akita-schedules` owns feature-gated shipped schedule table data.
- `akita-config` owns concrete runtime config presets and the single `CommitmentConfig` policy trait. It depends on `akita-planner` and optionally `akita-schedules` (`runtime_schedule` delegates to `akita_planner::resolve_schedule`).
- `akita-setup` owns config-backed setup construction and optional setup cache persistence.
- `akita-verifier` owns verifier replay without prover-only polynomial backends. It is directly `<Cfg>`-generic (depends on `akita-config`) and reaches `akita-planner` transitively, so the schedule-search DP is verifier-reachable.
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
[`specs/single-point-opening-batch.md`](specs/single-point-opening-batch.md),
and [`profile/akita-recursion/README.md`](profile/akita-recursion/README.md).
Build the book locally with `./scripts/serve-book.sh` (see
[`book/README.md`](book/README.md) for the toolchain). `AGENTS.md` is the
agent command runbook; `docs/` holds maintainer contracts (crate graph,
verifier contract, CI timing). `specs/` holds design records (lifecycle in
[`specs/PRUNING.md`](specs/PRUNING.md)). Documentation guardrails (CI + PR
comments) are in [`docs/documentation.md`](docs/documentation.md).

## Lineage

Akita keeps the earlier implementation lineage explicit while giving the improved scheme its own name.
This is also the line where planned protocol improvements over the original design live: faster verifier-oriented reductions via matrix-claim delegation and tensor-structured challenges, smaller large-field proofs via modulus switching and field-size lowering, and efficient zero-knowledge techniques under the Whiteout design direction.

## Contributing

Major features and architectural changes should start with a short spec.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [specs/TEMPLATE.md](specs/TEMPLATE.md) for the review workflow.

## Acknowledgements

The CRT/NTT and small-prime arithmetic design in this repository is informed by the Labrador/Greyhound C implementation family. In particular, the pseudo-Mersenne profile uses moduli of the form `q = 2^k - offset`. Akita provides a Rust-native architecture and APIs, while drawing algorithmic inspiration from those implementations.
