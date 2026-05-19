# Security Policy

Akita is a cryptographic proof-system implementation.
Security reports should be handled privately until the issue is understood and a fix is available.

## Reporting A Vulnerability

Please do not open public issues for suspected vulnerabilities.
Report security issues by emailing the maintainers listed in `Cargo.toml`, or by contacting the LayerZero Labs maintainers through the private channel used for Akita development.

Include:

- affected commit or release,
- whether the issue affects prover soundness, verifier correctness, serialization, transcript binding, dependency integrity, or private witness handling,
- a minimal reproducer or proof sketch when possible,
- whether the issue is public, embargoed, or already disclosed elsewhere.

## Scope

Security-sensitive surfaces include:

- verifier acceptance or rejection behavior,
- Fiat-Shamir transcript labels and challenge derivation,
- canonical serialization and deserialization of proofs, setup artifacts, and claims,
- crate-boundary violations that pull prover-only or planner-only code into verifier paths,
- dependency or CI supply-chain changes,
- unsafe code in field, algebra, and prover kernels,
- resource exhaustion from untrusted proof or setup bytes.

Performance regressions are security-relevant when they create practical denial-of-service risk for verifier or prover deployments.

## Current Status

Akita is not yet a formally audited production release.
Treat the implementation as security-critical research software unless a specific release states otherwise.
