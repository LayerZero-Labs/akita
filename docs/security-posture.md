# Akita Security Posture

This document records the security boundaries that reviewers should keep in mind when changing Akita.
It is not a substitute for a third-party audit.

## Trust Boundaries

Akita has four main trust boundaries:

- Proof bytes, commitments, claims, and opening points may be attacker controlled.
- Verifier preprocessing is recomputed by the verifier or authenticated by the deployment, and soundness assumes it was derived as claimed; see [Verifier Preprocessing](#verifier-preprocessing).
- Prover witnesses and intermediate polynomials may contain private application data.
- Build inputs, Git dependencies, and GitHub Actions influence the binary that users run.

Validated deserialization is the default for bytes that cross a trust boundary.
Unchecked deserialization is reserved for internal buffers whose producer and shape have already been validated in the same trust domain.

## Verifier Preprocessing

The verifier reads four preprocessed inputs besides the proof and the public statement:

- the expanded public matrix, a function of the setup seed;
- the setup-prefix commitment registry, a function of the seed and the schedule rows that offload setup;
- the optional terminal NTT cache, a function of the seed and one schedule row;
- the schedule catalog, which is planner output checked against the compiled SIS tables.

These inputs play the role of a verifying key.
Akita's setup is transparent, so none of them hides a secret.
That does not make arbitrary serialized preprocessing safe to accept.
Three separate claims stand behind a preprocessed object:

1. **Identity.** The setup seed names the intended public stream, and the schedule row digest names the intended row. Akita binds both into the transcript, so this claim follows from the protocol.
2. **Derivation.** The object is the stated function of the seed and the schedule. The matrix is the expansion of the seed, each setup-prefix commitment is the commitment to its prefix of the matrix, and the cache is the transform of its rows. Only computation establishes this claim: recomputing the object, or checking a proof of the derivation. Akita has no such proof today.
3. **Byte authentication.** The bytes this verifier loaded are the bytes for which the derivation was established. A release, a verifier program image, a pinned digest, or a signature can establish this claim. A checksum only detects accidental corruption.

Byte authentication does not replace derivation.
A pinned digest authenticates bytes and inherits the trust placed in whoever derived and checked them.
The trust anchor is the derivation, performed by the verifier or by a party that the deployment names.

A verifier therefore relies on a preprocessed object in one of two ways.
It recomputes the object and trusts no one for it.
Or it loads bytes that the deployment authenticates and trusts the party that derived them.
Recomputation is always allowed and never required.
Avoiding it is the reason preprocessing exists, and recomputing the setup-prefix commitments is the most expensive case.

Akita's soundness claim assumes that the derivation claim holds for every preprocessed object the verifier uses.
A verifier that loads preprocessing that neither it nor a party it trusts has derived is outside that claim, in the same way as a verifier that is given the wrong verifying key.
Akita does not authenticate preprocessing bytes itself.
It has no release manifest, digest pin, or signature for the catalog, the registry, or the cache.
A deployment that loads any of them without recomputing it must state the trust root that authenticated the bytes.

The schedule catalog is not a function of the seed, so its derivation claim has a different shape.
Catalog admission recomputes what can be recomputed, which is the SIS audit of each row.
Which catalog the operator approved is a byte authentication claim.

Decoding and derivation are separate checks:

- **Encoding validity** covers lengths, geometry, canonical field words, and residue ranges. Validated decoding enforces it and rejects malformed bytes with an error under the no-panic boundary below. Trusting an object's derivation does not relax its decoding.
- **Derivation** is the second claim above. The paths that skip re-derivation today still enforce encoding validity: decoding of the setup-prefix registry, a supplied terminal NTT cache, and the recursion benchmark's fixed-width setup decoder.

Unchecked deserialization skips encoding validity as well and reduces a noncanonical field word instead of rejecting it.
That is a stronger act: the caller vouches for every word of the bytes.
No production path decodes preprocessing this way today.
Such a path must be explicit at its call site and must not be reachable from proof bytes.

Akita also keeps cheap admission checks where an honest but faulty generator could produce data that verifies and is weaker than intended.
The SIS audit of catalog rows is the example.
A check is not needed when the fault it guards against makes every honest proof fail, because tests cover that.

What Akita checks today, per object:

- **Public matrix.** Validated decoding re-derives it from the seed, so the derivation claim is checked. Constructors and decoders named `trusted` or `unchecked` skip the re-derivation and rely on the caller.
- **Schedule catalog.** Admission audits each row. The origin of the catalog bytes is not checked.
- **Setup-prefix registry.** It is checked for encoding, for binding to the setup seed, and for slot structure. The commitments are not recomputed. The registry is structurally validated, not provenance validated.
- **Terminal NTT cache.** The verifier derives it from the setup when it prepares the cache itself. A supplied cache is checked for identity, shape, and residue ranges, and its contents are not recomputed.

[`specs/flat-public-matrix-and-exact-ntt-cache.md`](../specs/flat-public-matrix-and-exact-ntt-cache.md#setup-provenance-and-validation) is the design record for these claims.
It also fixes the intended proof of derivation, a certificate chain that ends at a prefix small enough to derive directly.
That mechanism is not implemented.

Review findings should be classified against this model.
A hostile proof that is accepted, panics, or forces unbounded work against correctly derived preprocessing is a defect.
Malformed preprocessing bytes that cause a panic or unbounded work are a defect.
A failure of a derivation or admission check that Akita does perform is a defect.
A scenario that requires well-formed preprocessing that was not derived as claimed, supplied through a path that assumes derivation, is outside the soundness claim.
It is a question about the deployment's trust root, or about whether a cheap admission check is worth adding.

## Verifier No-Panic Boundary

Verifier-facing execution must be panic-free for malformed public inputs.
Public proofs, setup artifacts, schedules, fold parameters, opening points, commitments, claim incidence summaries, direct witnesses, and transcript data must be rejected with `AkitaError` or `SerializationError` rather than `panic!`, assertions, unchecked indexing, arithmetic overflow, or unbounded allocation.

Verifier-reachable helpers may remain infallible only when a prior boundary has established the required invariant.
That validation should live at deserialization, setup validation, schedule selection, verifier API entry, or prepared-state construction.
Hot verifier arithmetic paths should not grow fallback evaluators or repeated defensive checks when a single boundary check can preserve the existing execution model.

## Soundness-Critical Surfaces

Reviewers should treat these changes as security-sensitive:

- verifier acceptance logic,
- verifier input validation or any verifier-reachable panic-shaped operation,
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
Validated proof-shape decoding must reject shape dimensions that exceed the generic verifier-facing allocation cap before allocating shape-controlled buffers.

## Current Assurance

Akita currently relies on strict Rust CI, crate-boundary checks, specs for large protocol changes, and targeted tests.
The hardening roadmap adds supply-chain checks, fuzzing, property tests, bounded untrusted decoding, and clearer unsafe and panic discipline.
