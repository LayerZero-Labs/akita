# Akita PCS

Akita is a high-performance, modular lattice polynomial commitment scheme with transparent setup and post-quantum security.

Akita is the public scheme name for this implementation and the intended repository/package name is `akita-pcs`.
The codebase is being decomposed into a focused `akita-*` crate family rather than remaining a single monolithic package.

The current workspace exposes the main ownership boundaries under `crates/`:

- `akita-field`, `akita-serialization`, and `akita-algebra` own foundational arithmetic, encoding, NTT, ring, and polynomial utilities.
- `akita-transcript`, `akita-challenges`, and `akita-sumcheck` own Fiat-Shamir transcripts, challenge sampling, and generic sumcheck machinery.
- `akita-types` owns shared proof, setup, schedule, layout, and commitment data shapes used by both roles.
- `akita-config` owns concrete runtime config presets and config-backed schedule/SIS policy.
- `akita-verifier` owns verifier replay without depending on prover-only polynomial backends.
- `akita-prover` owns commitment, proving, setup expansion, recursive witness construction, and polynomial backends.
- `akita-planner` owns offline schedule search and proof-size/security planning.

## Lineage

Akita descends from Hachi and keeps that ancestry explicit, while giving the improved scheme its own name.
This is also the line where planned protocol improvements over the original design live: faster verifier-oriented reductions via matrix-claim delegation and tensor-structured challenges, smaller large-field proofs via modulus switching and field-size lowering, and efficient zero-knowledge techniques under the Whiteout design direction.

## Contributing

Major features and architectural changes should start with a short spec.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [specs/TEMPLATE.md](specs/TEMPLATE.md) for the review workflow.

## Acknowledgements

The CRT/NTT and small-prime arithmetic design in this repository is informed by the Labrador/Greyhound C implementation family. In particular, the pseudo-Mersenne profile uses moduli of the form `q = 2^k - offset`. Akita provides a Rust-native architecture and APIs, while drawing algorithmic inspiration from those implementations.
