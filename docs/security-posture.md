# Akita Security Posture

This document records the security boundaries that reviewers should keep in mind when changing Akita.
It is not a substitute for a third-party audit.

## Trust Boundaries

Akita has three main trust boundaries:

- Verifier-facing proof and setup bytes may be attacker controlled.
- Prover witnesses and intermediate polynomials may contain private application data.
- Build inputs, Git dependencies, and GitHub Actions influence the binary that users run.

Validated deserialization is the default for bytes that cross a trust boundary.
Unchecked deserialization is reserved for internal buffers whose producer and shape have already been validated in the same trust domain.

## Soundness-Critical Surfaces

Reviewers should treat these changes as security-sensitive:

- verifier acceptance logic,
- Fiat-Shamir domain labels, transcript order, or challenge derivation,
- canonical field, ring, proof, setup, and claim serialization,
- crate dependency edges into verifier-facing crates,
- configuration schedules that determine proof shape,
- unsafe field, algebra, NTT, or matrix kernels,
- dependency, toolchain, or CI changes.

## Unsafe Code Policy

Unsafe code is allowed only where it buys concrete performance or layout control that safe Rust cannot express cleanly.
Every unsafe block should have a local safety argument that names the invariant being relied on.
Verifier-facing crates should avoid unsafe code unless a spec explicitly justifies it.

## Resource Limits

Verifier-facing decoding must not allocate solely from attacker-provided lengths without an explicit bound.
When a proof shape already determines a length, prefer shape-derived allocation over self-described vector lengths.

## Current Assurance

Akita currently relies on strict Rust CI, crate-boundary checks, specs for large protocol changes, and targeted tests.
The hardening roadmap adds supply-chain checks, fuzzing, property tests, bounded untrusted decoding, and clearer unsafe and panic discipline.
