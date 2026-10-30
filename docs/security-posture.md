# Akita Security Posture

This document records the security boundaries that reviewers should keep in mind when changing Akita.
It is not a substitute for a third-party audit.

## Trust Boundaries

Akita has four main trust boundaries:

- Proof bytes, commitments, claims, and opening points may be attacker controlled.
- Verifier preprocessing comes from a pinned release that the binary's compiled trust anchor names, and soundness assumes the release was derived as claimed; see [Verifier Preprocessing](#verifier-preprocessing).
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

Akita authenticates preprocessing bytes with pinned releases ([`specs/pinned-preprocessing-releases.md`](../specs/pinned-preprocessing-releases.md)).
A release is the generator's deterministic output for a description under `params/`, named by one manifest whose hash is the release's `bundle_id`.
A binary trusts only the `(bundle_id, set name)` pairs of its compiled trust anchor (`akita-anchors` for Akita).
The loaders check the anchor before decoding the manifest and compare each set record with the compiled configuration; their per-object checks are listed below.
A required merge check regenerates every anchored release from scratch, so an anchored release is what the code at that commit produces.
The derivation claim for a production verifier therefore rests on the repository, the build, and the merge check, and `akita-preprocess check` lets anyone repeat it.

Production verifiers are built only from loader values: `PinnedVerifierSetup` and `PinnedTerminalCache` have private fields, no public constructor, and no deserializer.
Every path that builds a setup from loaded bytes without that check is an `unsafe fn` whose contract is that the caller guarantees derivation: `AkitaVerifierSetup::from_parts` and its decoder, `AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked`, the `borrow_trusted` views, `AkitaProverSetup::to_verifier_setup`, `AkitaVerifier::new` and `for_selection`, and the `AkitaCommitmentScheme` verifier helpers.
Here `unsafe` marks a soundness contract, not a memory-safety one.
Tests and benchmarks use these paths with setups they derived themselves.

The schedule catalog is not a function of the seed, so its derivation claim has a different shape.
Catalog admission recomputes what can be recomputed, which is the SIS audit of each row.
Which catalog the operator approved is a byte authentication claim.

Decoding and derivation are separate checks:

- **Encoding validity** covers lengths, geometry, canonical field words, and residue ranges. Validated decoding enforces it and rejects malformed bytes with an error under the no-panic boundary below. Trusting an object's derivation does not relax its decoding.
- **Derivation** is the second claim above. Validated decoding of the setup-prefix registry and a supplied terminal NTT cache checks encoding without recomputing their contents.

Trusted in-place setup and cache views can also skip field-word or residue-range checks.
Their explicit caller obligations are described under [Trusted in-place setup views](#trusted-in-place-setup-views) below: the deployment vouches for the words in its authenticated program image.
Unchecked deserialization can instead reduce a noncanonical field word rather than rejecting it.
These paths must be explicit at their call sites and must not be reachable from proof bytes.

Akita also keeps cheap admission checks where an honest but faulty generator could produce data that verifies and is weaker than intended.
The SIS audit of catalog rows is the example.
A check is not needed when the fault it guards against makes every honest proof fail, because tests cover that.

What the loader checks, per object:

- **Public matrix.** The loader derives the prefix its admitted rows need from the release's seed. Validated decoding also re-derives it. Only a guest takes an embedded prefix instead, under the contract of `PinnedMatrixPrefix::embedded_unchecked`.
- **Schedule catalog.** The catalog file matches its manifest hash and the set record's catalog digest, and admission audits each row. Program rows arrive with the program preprocessing whose digest authenticates them, and pass the same audit.
- **Setup-prefix registry.** Each payload matches its file hash and its recomputed prefix commitment digest matches the manifest, in registry mode; in seed-derived mode the loader derives the payload and compares the same digest. Protocol Stage 3 absorbs that digest in every proof.
- **Terminal NTT cache.** The verifier derives it from the setup when it prepares the cache itself. A release cache is checked against its manifest hash and then for identity, shape, and residue ranges. A guest-embedded view checks identity and shape; its contents rest on the contract of `PinnedTerminalCache::embedded_unchecked`.

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

### Trusted in-place setup views

A RISC-V guest verifier reads its setup from its own program image. Copying and
range-checking that matrix would dominate the guest's cost, so three paths view
trusted, `'static`, little-endian setup bytes in place. They are used only on
bytes the loader checked or on embedded guest bytes issued through the
guest-only `unsafe fn`s `PinnedMatrixPrefix::embedded_unchecked` and
`PinnedTerminalCache::embedded_unchecked`.

- `AkitaDeserialize::borrow_many_trusted`, and the `borrow_trusted` methods built
  on it, reinterpret field words as elements. They check length and alignment.
  They take a `TrustedBytes` token: canonical words are the caller's
  obligation, and constructing the token is the one `unsafe` step.
- `view_riscv64_scalar_q128_cache` reinterprets the residue rows of a prepared
  NTT cache. Residue ranges are a trusted-input assumption stated on
  `TrustedTerminalCache::View`.
- The bulk prime-field decoder reads into initialized element storage through
  one byte view. It then applies the single-element rule to every word.

Proof bytes never reach these paths.

## Resource Limits

Verifier-facing decoding must not allocate solely from attacker-provided lengths without an explicit bound.
When a proof shape already determines a length, prefer shape-derived allocation over self-described vector lengths.
Validated proof-shape decoding must reject shape dimensions that exceed the generic verifier-facing allocation cap before allocating shape-controlled buffers.

## Current Assurance

Akita currently relies on strict Rust CI, crate-boundary checks, specs for large protocol changes, and targeted tests.
The hardening roadmap adds supply-chain checks, fuzzing, property tests, bounded untrusted decoding, and clearer unsafe and panic discipline.
