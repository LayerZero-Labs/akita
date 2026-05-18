# Spec: Akita ZK Tail Rejection Prototypes

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-05-08 |
| Status | implementation review |
| PR | #70 |

## Summary

This PR adds an experimental `akita-zk` leaf crate for studying ring-native Ajtai opening proofs as a candidate replacement for Akita's current terminal tail opening.
The crate implements the relation, Fiat-Shamir transcript, compact response encoding, and several rejection-sampling policies needed to measure proof-size and prover-time tradeoffs on the observed production tail shape.
It deliberately does not integrate these proofs into `akita-scheme`, `akita-prover`, or `akita-verifier`.

## Intent

### Goal

Build a ring-native experimental harness for proving knowledge of a short Ajtai opening over `R_q = F_q[X] / (X^D + 1)` and comparing rejection policies on Akita's terminal folded witness shape.

The main new crate is `crates/akita-zk`.
It reuses `akita-algebra::CyclotomicRing`, `akita-challenges::SparseChallengeConfig`, `akita-transcript`, `akita-serialization`, and `akita-field`.
It introduces no new field implementation, ring implementation, challenge sampler, or transcript backend.

### Current State

The branch adds:

- `AjtaiRelation<F, D>` for public relations `A w = t` over Akita cyclotomic rings.
- `CompactRingVec` for canonical two's-complement response packing.
- Exact uniform-box rejection parameters and proof generation.
- Experimental Gaussian rejection parameters and proof generation.
- Measurement-only public-sign Gärtner rejection parameters and proof generation.
- Examples for `D=64`, `D=128`, and a unified `policy_comparison` harness.

The motivating production tail shape comes from the PR benchmark report:

- terminal witness length: `51,872` coefficients;
- digit width: `5` bits;
- target ring shape used by the comparison harness: `D=128`, `K=2`, `Prime32Offset99`.

### Invariants

- `akita-zk` remains a leaf prototype crate.
  No production commitment, proving, setup, or verifier path calls into it.
- All relation arithmetic is over `CyclotomicRing<F, D>`.
  The crate must not duplicate ring arithmetic already provided by `akita-algebra`.
- The public relation is `A w = t`.
  Provers must reject witnesses that either fail to open `t` or exceed the configured witness infinity bound.
- Fiat-Shamir challenges must bind the ring degree, matrix shape, challenge configuration, rejection parameters, matrix, commitment, and announcement.
  Distinct rejection policies use distinct transcript rejection labels.
- Compact proof verification must require the canonical response width `CompactRingVec::bits_for_bound(response_bound)`.
  Verifiers must reject over-wide encodings rather than accepting a larger proof under the same parameter set.
- Rejection parameters must rule out centered modular wrap.
  Box rejection checks `gamma + beta < q / 2`.
  Gaussian and Gärtner checks require `mask_bound + beta < q / 2`.
- The verifier side of all non-interactive proof checks uses only transcript, ring, integer, and field operations.
  Floating-point Gaussian and Gärtner calculations are prover-side and measurement-side only.
- The mathematical Gaussian rejection policy is the zero-knowledge target under an ideal discrete-Gaussian sampler.
  The current implementation is not a production sampler because it uses floating-point Box-Muller sampling and floating-point acceptance probabilities.
- The Gärtner implementation is not zero-knowledge.
  It sends the sign bit publicly and is exposed under `akita_zk::measurements` with `public_sign` in the type and function names.

### Non-Goals

- This PR does not replace Akita's current tail proof.
- This PR does not add a production discrete-Gaussian sampler.
- This PR does not claim production constant-time behavior for Gaussian or Gärtner proving.
- This PR does not implement BLISS or Lantern sign hiding for Gärtner.
- This PR does not add recursive folding changes, per-block rejection, or smaller terminal witness construction.
- This PR does not define final security parameters for a deployed tail proof.

## Evaluation

### Acceptance Criteria

- [ ] `akita-zk` builds as part of the workspace.
- [ ] Box, Gaussian, and public-sign Gärtner proof variants have honest prove/verify tests.
- [ ] Tampering with the response or public Gärtner sign fails verification.
- [ ] Compact response round-tripping is tested.
- [ ] Compact verifiers reject non-canonical response widths.
- [ ] The `policy_comparison` example reports analytical and measured proof sizes for the production tail shape.
- [ ] CI passes `cargo fmt`, clippy, docs, tests, CodeQL, Socket, and the one-hot benchmark workflow.
- [ ] PR review text states that Gärtner is measurement-only and non-ZK.

### Testing Strategy

Unit tests live inside `crates/akita-zk`.
They cover parameter derivation, compact encoding, honest verification, tamper rejection, interactive simulation, and public-sign Gärtner sign tampering.

The focused local checks are:

```bash
cargo fmt -q
cargo test -p akita-zk
cargo clippy -p akita-zk --all-targets --all-features -- -D warnings
```

The full PR checks are:

```bash
cargo clippy --all --all-targets --all-features -- -D warnings
cargo clippy --all --all-targets --no-default-features -- -D warnings
cargo test --workspace
```

### Performance

The comparison harness is:

```bash
cargo run --release -p akita-zk --example policy_comparison
HACHI_ZK_RUN_PROOFS=0 cargo run --release -p akita-zk --example policy_comparison
```

At the production tail shape, the current headline results are:

- exact box rejection is ZK and lands at a 27-bit response;
- Gaussian rejection is the main ZK proof-size target and reaches a 23-bit response at feasible but lower acceptance settings;
- public-sign Gärtner gives a non-ZK ceiling near a 22-bit response.

The Gärtner number should be read only as an upper bound on what a future sign-hiding design might buy before accounting for its extra proof cost.

## Design

### Architecture

`akita-zk` is intentionally separate from the production PCS crates.
The dependency direction is one-way: `akita-zk` consumes existing Akita primitives, while the rest of the workspace does not depend on `akita-zk`.

The proof shape is:

1. The prover samples a mask `y`.
2. The prover sends or commits to the announcement `a = A y`.
3. Fiat-Shamir samples a sparse ring challenge `c`.
4. The prover forms a shift `c w`.
5. The prover returns an accepted response `z`, according to the selected rejection policy.
6. The verifier checks the response bound and the relation equation.

For box and Gaussian variants, the verifier equation is:

```text
A z = a + c t
```

For the public-sign Gärtner measurement variant, the verifier equation is:

```text
A z = a + sign * c t
```

The compact proof representation stores the announcement as full field elements and the response as packed centered coefficients.
The compact encoding is intentionally not self-describing as a protocol object.
The verifier receives the public parameter set and checks that the compact response shape and bit width match those parameters exactly.

### Rejection Policies

Box rejection samples `y` uniformly from `[-gamma, gamma]` and accepts if `z = y + c w` lies in `[-gamma + beta, gamma - beta]`.
This gives an exact uniform accepted response distribution for the bounded shift model.

Gaussian rejection samples `y` from a rounded, tail-truncated Gaussian and accepts using the standard Lyubashevsky ratio.
The implementation is useful for sizing and acceptance experiments, but production use requires a constant-time discrete-Gaussian sampler and fixed or integer acceptance arithmetic.

Gärtner rejection implements the single-step `f_v / g_v` rule from Gärtner's iterative rejection paper.
Akita's current relation lacks the BLISS-style structure that hides the selected sign, so this PR makes the sign public and keeps the API in the measurement namespace.

### Alternatives Considered

One alternative was to wire the prototype directly into `akita-scheme`.
That was rejected because the rejection policies are still being evaluated and the Gaussian sampler is not production-grade.

Another alternative was to implement a full Gärtner or BLISS-shaped sign-hiding protocol immediately.
That was rejected because the proof-size upside at the current tail shape appears marginal relative to the additional proof machinery.

Per-block rejection remains a promising follow-up because it may reduce effective response width without introducing a sign-hiding sub-protocol.

## Documentation

This spec is the main reviewer guide for PR #70.
The PR description should link to it and keep the Gärtner non-ZK warning visible near the headline numbers.

The examples are executable documentation for the current experiments.
The crate-level docs should continue to state that this is an experimental prototype and that measurement-only code lives under `akita_zk::measurements`.

## Execution

Before merge, the PR should keep the hardening changes from this spec:

- public-sign Gärtner APIs use `public_sign` names;
- public-sign Gärtner is not re-exported from `akita_zk::protocols`;
- compact verifiers reject non-canonical widths;
- Gaussian and Gärtner no-wrap checks include the extra `beta` shift;
- unresolved automated review comments are fixed or explicitly addressed.

Future productionization should start from the Gaussian path or per-block rejection, not from the public-sign Gärtner harness.
If Gärtner is revisited, the first production requirement is sign hiding, not sampler optimization.

## References

- PR #70: `quang/zk-tail-rejection-prototypes`.
- `crates/akita-zk/examples/policy_comparison.rs`.
- `docs/proof-size-reduction-study.md`.
- `Gaertner_Iterative_Rejection_Sampling.pdf`.
- Lyubashevsky-style Fiat-Shamir-with-aborts rejection sampling.
- Lantern and Nguyen thesis notes for bimodal and sign-hiding background.
