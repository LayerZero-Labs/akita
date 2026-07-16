# Spec: Akita Transcript Hardening (Spongefish-aligned, Instance-Descriptor Preamble)

| Field       | Value                                       |
|-------------|---------------------------------------------|
| Author(s)   | @quangvdao                                      |
| Created     | 2026-05-18                                  |
| Status      | DRAFT — PR #90 review revisions applied 2026-05-18; OQ 6 (cutover ordering) remains. Ready for implementation after reviewer sign-off. |
| Branch      | `quang/akita-spongefish-transcript`         |
| PR          | #90                                         |

## Summary

PR #88 fixed one Fiat-Shamir soundness bug in Akita's terminal fold by absorbing `final_w` directly into the transcript instead of treating `next_w_commitment` as a substitute.
That fix was reactive; the diagnostic that surfaced it (`crates/akita-pcs/tests/transcript_trace.rs`) is one-shot test scaffolding, not a durable hardening layer.
The class of bug PR #88 closed (a value used in a soundness-critical computation without being bound into the transcript) is a general Fiat-Shamir failure mode, and is the same class as `mmaker`'s open regression test on Jolt (a16z/jolt#1536) and `quangvdao`'s own merged fix on Jolt (a16z/jolt#1358).

This spec proposes a **trimmed first-cut** transcript-hardening layer for Akita, in one PR off `main`, structured around **three active pillars** (P0, P2, P3).
Two further pillars discussed in earlier conversation (P1 prover/verifier trait split, P4 `Bound<T>` at module seams) are **explicitly deferred** to follow-up PRs and listed under §Deferred Follow-Ups; this PR enforces their soundness intent via runtime smell checks instead.

The single most consequential pillar is **P0: binding a canonical hash of the instance parameters into the transcript preamble**, via spongefish's `DomainSeparator.instance`.
This is the pattern Michele Orrù described off the cuff and is *stronger* than what `a16z/jolt#1455` (spec to port `jolt-transcript` to spongefish) proposes today.

The note already in `crates/akita-transcript/src/labels.rs:10-11` captures the long-term intent in one line; this spec formalizes the *first approximation* of that intent and the future-work hooks for the rest:

> NOTE: eventually we will switch to spongefish and drop the per-operation label,
> instead appending a full, exact description of the entire verifier as a beginning digest.

The full-verifier-program digest is the multi-week direction; this PR ships the instance-parameter digest (good first approximation of domain separation) and the diagnostic substrate that lets us evolve toward the full version later.

## Intent

### Goal

Build a first-approximation Fiat-Shamir hardening layer for Akita that delivers two soundness guarantees and one observability guarantee, in one PR off `main`:

1. **Instance domain separation (soundness).** Prover and verifier must reject when they're running different protocol instances. Caught by absorbing an `AkitaInstanceDescriptor` digest into the spongefish preamble. Distinguishes `D=32` from `D=64`, `Lagrange` from `Monomial`, schedule X from schedule Y, etc.
2. **Per-value binding hygiene (soundness, PR #88's bug class).** No value used in challenge derivation goes un-absorbed. Caught in this PR by a runtime smell check in `LoggingTranscript`; promoting to type-level enforcement (prover/verifier trait split) is deferred to a follow-up.
3. **Schedule observability.** Emit a flat, inspectable absorb/squeeze event stream for any test or example run, so future protocol changes can be reviewed by reading the schedule directly rather than re-deriving it from prover and verifier source.

### Key abstractions

- `AkitaInstanceDescriptor` (new, in `akita-types`) — canonical, deterministic, prefix-free serialization of every input that determines the verifier algorithm's behavior on this instance. Lifetime-tiered with four nested sub-structs: `AlgebraSection` (per-binary), `SetupSection` (per-`CommitmentConfig`-instantiation), `PlanSection` (per-proof), `CallSection` (per commit-and-open call). **Resolved field list in §Pillar P0.**
- `AkitaTranscript<Sponge>` (new, in `akita-transcript`) — thin wrapper over `spongefish::ProverState<Sponge>` (on the prover side) and `spongefish::VerifierState<'_, Sponge>` (on the verifier side). **Same trait surface on both sides** for this PR; the asymmetric prover/verifier trait split is deferred to a follow-up (see §Deferred Follow-Ups).
- Sponge backends are feature-selected within `akita-transcript`: `transcript-blake2b` selects `spongefish::instantiations::Blake2b512` (default), `transcript-keccak` selects `spongefish::instantiations::Keccak` when built with `--no-default-features`. Both implement `spongefish::DuplexSpongeInterface`. Cargo `--all-features` unifies both backend features, so all-features/default-unified builds resolve to Blake2b.
- The `spongefish::DomainSeparator` is constructed at transcript creation with a 64-byte protocol tag and the `AkitaInstanceDescriptor` canonical bytes as the instance encoding. The protocol tag is **parameterized per sponge backend** — `b"akita-pcs/transcript/v1/blake2b\0..."` under `transcript-blake2b`, `b"akita-pcs/transcript/v1/keccak\0..."` under `transcript-keccak` — defined as a `cfg`-gated `pub const PROTOCOL_TAG: &[u8; 64]` in `akita-transcript`. This keeps sponge-family identity inside spongefish's own discipline (no extra field in the descriptor) and makes cross-family transcripts byte-distinguishable at the very first absorb. P0's preamble lives inside spongefish's `DomainSeparator.instance`, not as a separate first-absorb. This is the spongefish-native place for it.
- `Label` (new, in `akita-transcript`) — a **zero-sized type in production and default test builds**, and a rich `{tag, file, line}` capture only when the `logging-transcript` feature is enabled. Plus a `label!("...")` macro that captures source location only when logging is enabled. Labels are **never absorbed into the production sponge**. Callsites pass a `Label` to every transcript method; the production build compiles it out entirely. See Pillar P2 for the design.
- `LoggingTranscript<Sponge>` (new, in `akita-transcript`, behind `feature = "logging-transcript"`) — wrapper over `AkitaTranscript` that records each labeled absorb/squeeze event into a thread-local buffer for inspection. Promoted from `crates/akita-pcs/tests/transcript_trace.rs`. Runs the five smell checks listed in §Smell Checks.

### Invariants

The implementation must preserve the following invariants. Where a check is mechanically enforced, the test or assertion is named.

1. **Preamble determines challenges.** For any two `(prover, verifier)` pair runs with different `AkitaInstanceDescriptor` bytes, the first challenge squeezed from the transcript differs. Tested by `transcript_preamble_separation` (new): randomize one field of the descriptor, prove on side A, verify with descriptor B, assert rejection at challenge #1.
2. **Prover/verifier event-stream equality.** For every exercised valid `(nv, incidence, basis)` case, the chronological sequence of `Absorb { label, bytes_digest, bytes_len }` and `Squeeze { label, len }` events recorded by `LoggingTranscript` on the prover and verifier sides is identical. Tested by hardening fixtures plus a differential prop-test, generalized from the historical `tests/transcript_trace.rs` diagnostic.
3. **No silent unbinding (smell-check enforced).** Every proof field that becomes an input to a challenge-dependent verifier subprotocol is absorbed before any dependent squeeze. Enforced in this PR by smell check #4 (`wire_value_before_squeeze_coverage`) inside `LoggingTranscript` at test time, NOT by the type system. Future work (P1, deferred) promotes this to compile-time via a prover/verifier trait split or a verifier proof-reader adapter.
4. **No back-compat preservation of today's transcript bytes.** Per the workspace's "Full Cutover, No Backward Compatibility" rule, this PR is free to change the transcript byte layout. Existing proofs do not need to verify after this PR. All in-tree end-to-end tests (`muldiv`, the akita-pcs e2e suite, examples/profile) must continue to pass against the new transcript.
5. **Labels never enter the production sponge.** Asserted by a unit test in `akita-transcript`: build an `AkitaTranscript` in production-build mode, exercise every method, dump the spongefish op log, assert no label-tagged bytes appear. The label parameter on every method is a ZST in release/non-logging builds and is compiled out.

### Non-Goals

1. **Migrating the proof types from structured (`AkitaBatchedProof`) to NARG byte tape.** Akita's proofs are deeply structured (per-level shapes, per-step variants, terminal vs intermediate). The structure is load-bearing for debugging, planner cost models, profile accounting, and serialization stability. We keep `AkitaBatchedProof` as the wire format. The transcript hardening operates *alongside* the structured proof, not by replacing it. (Note: this PR *does* adopt `spongefish` as the sponge / Fiat-Shamir construction; see §Goal. The NARG-as-proof migration is a separate, larger decision and is listed under §Deferred Follow-Ups.)
2. **Adding Poseidon as a transcript sponge.** Out of scope this PR. Blake2b and Keccak are both added (see §Goal, §Acceptance Criteria). Poseidon, if useful for in-circuit recursion later, lands as a follow-up.
3. **Hashing source code or the full verifier program.** A build-time digest of the verifier crate's source is paranoid in a way that costs more than it earns. The principled version of this — hashing a canonical serialization of the verifier algorithm authored in a typed IR (the bolt-lean direction) — is multi-week future work and is listed under §Deferred Follow-Ups.
4. **Waiting for `jolt-transcript` to ship its spongefish port** (a16z/jolt#1455). We adopt `spongefish` directly in `akita-transcript`. If `jolt-transcript`'s port lands later, our consumer integration (jolt-side) can rebase onto it; if it doesn't, Akita's own spongefish layer is sufficient.
5. **Prover/verifier trait split (P1 in earlier conversation).** Deferred; smell check #4 carries the invariant. See §Deferred Follow-Ups for the trigger conditions.
6. **`Bound<T>` at module seams (P4 in earlier conversation).** Deferred; most of its value depends on a working P1 or equivalent proof-reader boundary. See §Deferred Follow-Ups.

## Evaluation

### Acceptance Criteria

- [x] `akita-transcript` adds `spongefish` `0.7.x` as a workspace dependency.
- [x] Two new Cargo features added to `akita-transcript/Cargo.toml`: `transcript-blake2b` (default) and `transcript-keccak`, gating `spongefish::instantiations::Blake2b512` / `spongefish::instantiations::Keccak` respectively. Builds with no backend enabled fail; explicit Keccak uses `--no-default-features`; Cargo all-features/default-unified builds resolve to Blake2b.
- [x] New `AkitaInstanceDescriptor` type in `akita-types` with the four-section shape from §Pillar P0, including normalized batch incidence, effective post-fallback schedule digest, deterministic setup identity, and protocol feature mode. Canonical `AkitaSerialize` / `AkitaDeserialize` round-trip tests and prover/verifier event equality cover cross-side descriptor-byte equality.
- [x] The old Jolt-backed transcript backend and `Blake2bTranscript` / `KeccakTranscript` public aliases are removed. `AkitaTranscript<Sponge>` is the transcript implementation, backed by `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<'_, Sponge>`. The local generic `Transcript<F>` protocol trait remains as the shared trait surface for prover, verifier, and `LoggingTranscript`.
- [x] Spongefish absorbs Akita values through canonical Akita serialization or canonical field bytes wrapped in a local prefix-free `FramedBytes` `Encoding<[u8]>` adapter. Direct `Encoding` / `Decoding` impls for `akita-field` / `akita-serialization` types are intentionally not added in `akita-transcript`, because those would be foreign-trait impls over foreign crate types unless routed through local wrappers.
- [x] `AkitaTranscript` construction takes canonical `AkitaInstanceDescriptor` bytes plus a fixed protocol tag, builds the `spongefish::DomainSeparator` with `.instance(<descriptor bytes>)`, and returns the wrapped prover or verifier state. The byte-taking API avoids a crate cycle from `akita-transcript` back into `akita-types`.
- [x] `Label` ZST + `label!()` macro: every direct `AkitaTranscript` method takes `Label` as a leading argument. In `cfg(not(feature = "logging-transcript"))` builds, including ordinary unit tests, `Label` is a unit struct and the macro expands to it; the parameter is compiled out. Under `--features logging-transcript`, `Label` carries `{tag: &'static str, file: &'static str, line: u32}`.
- [x] A default-feature unit test asserts `size_of::<Label>() == 0`, and the `Label` value is never serialized, hashed, or retained by the production transcript path.
- [x] Invariant 5 (labels never enter the production sponge) is verified by a dedicated unit test.
- [x] `LoggingTranscript<Sponge>` is published as a real `pub` helper in `akita-transcript`, behind `feature = "logging-transcript"`, with a small CLI example `crates/akita-pcs/examples/transcript_schedule` that dumps the schedule. Cross-crate integration tests that need rich labels enable the feature explicitly.
- [x] The five smell checks listed in §Smell Checks below pass on the hardening tests that opt into `LoggingTranscript`.
- [x] A new `crates/akita-pcs/tests/transcript_hardening.rs` integration test exists and exercises: (a) preamble separation under perturbed descriptors, (b) prover/verifier event-stream equality, (c) the smell checks.
- [x] A differential property test (using `proptest`) under `crates/akita-pcs/tests/` fuzzes batch incidence and asserts event-stream equality + verify-success. The seed corpus covers smallest valid `nv`, mid-range, near-canonical `nv=20`, both Lagrange and Monomial basis modes, and non-uniform batch groupings `[1, 2]` / `[2, 1]`.
- [x] The previous `crates/akita-pcs/tests/transcript_trace.rs` diagnostic is replaced by `LoggingTranscript` hardening tests plus the `transcript_schedule` example; the stale local test file no longer exists on this branch.
- [x] `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`, and `cargo test` pass under `--features transcript-blake2b`, `--features transcript-blake2b,logging-transcript`, `--features transcript-keccak,logging-transcript --no-default-features`, and repository all-features CI.

### Smell Checks

Run by `LoggingTranscript`'s drop / verify-time check on every logging-feature transcript built in tests:

1. **Preamble is descriptor.** The constructed `spongefish::DomainSeparator` contains `.instance(<canonical descriptor bytes>)` (cross-checked by re-deserializing the instance encoding and comparing to a freshly-constructed `AkitaInstanceDescriptor`).
2. **No zero-byte absorbs.** Every `absorb_*` call has a non-empty bytes payload. (A zero-byte absorb is almost always a serialization bug.)
3. **Known labels only.** Every `Label.tag` is in `akita_transcript::labels::ALL_LABELS`. Catches accidental literal-string labels at callsites.
4. **Wire-value-before-squeeze coverage.** This is the smell check that carries the soundness invariant a prover/verifier trait split would have given us. For every `Squeeze` event in the verifier-side event log, every semantic wire-use event since the previous `Squeeze` must have a corresponding `Absorb` event recorded in between, under the same label, with the same canonical byte digest and length. Implementation: verifier replay records `Wire { label, bytes_digest, bytes_len }` when a structured proof field becomes an input to a challenge-dependent verifier subprotocol, not merely when the proof is deserialized. The terminal-fold case records the logical `w_hat` segment before sparse-seed sampling and records the final-witness remainder before ring-switch `alpha`/`tau1`; the terminal window does not squeeze `tau0`. The check fails if a `Wire` event is followed by `Squeeze` without an intervening matching `Absorb`. This is the smell check that would have caught PR #88's bug.
5. **Tracked wire coverage is complete.** The verifier-side logging harness uses a proof-field coverage manifest, or a `TrackedAkitaBatchedProof` view, to prove that every structured proof field that can feed a challenge-dependent subprotocol has an associated `Wire` instrumentation point. This prevents smell check #4 from passing vacuously when a new direct proof-field read is added without logging. Prover/verifier event-stream equality remains a paired integration invariant, not a smell check, because it does not catch symmetric omissions.

Each smell check is an ordinary assertion-shaped guard inside `LoggingTranscript` that fires only when the wrapper is active, including optimized test/profile runs.

### Testing Strategy

**New tests:**

- `crates/akita-types/src/instance_descriptor.rs::tests::canonical_encoding_roundtrip` — `AkitaSerialize` / `AkitaDeserialize` round-trip on the four-section descriptor.
- `crates/akita-types/src/instance_descriptor.rs::tests::cross_side_equality` — construct from prover-side inputs (setup + commit + open args) vs verifier-side inputs (setup + proof header); assert byte-identical.
- `crates/akita-transcript/src/lib.rs::tests::labels_never_in_production_sponge` — Invariant 5.
- `crates/akita-pcs/tests/transcript_hardening.rs::preamble_separation` — perturbing each descriptor field changes challenge #1.
- `crates/akita-pcs/tests/transcript_hardening.rs::event_stream_equality_small` — single fixture, full inspection.
- `crates/akita-pcs/tests/transcript_hardening.rs::smell_checks_pass_for_matched_wire_absorb` — exercises the positive wire-coverage path, with unit tests in `akita-transcript` covering the individual negative smell-check cases.
- `crates/akita-pcs/tests/transcript_hardening_proptest.rs` — `proptest`-driven fuzz over batch incidence, plus a deterministic seed corpus over `nv`, incidence, and basis, asserting event-stream equality and verify-success.
- `crates/akita-pcs/tests/transcript_hardening.rs::pr88_regression` — explicit replay of the PR #88 setup; smell check #4 must fail for the legacy shape that absorbs `next_w_commitment` but consumes cleartext `final_w`, and must also fail if `final_w` is mutated after a matching absorb. Locks in the bug class.

**Existing tests that must keep passing (with transcript-bytes updated):**

- The full `cargo test` workspace run.
- `AKITA_MODE=onehot AKITA_NUM_VARS=20 cargo run --release --example profile` smoke run.

**Tests that are replaced:**

- `crates/akita-pcs/tests/transcript_trace.rs` — replaced by durable `LoggingTranscript` hardening tests and the `transcript_schedule` example. All ad-hoc event-coalescing / parallel-test-mutex scaffolding moves into `LoggingTranscript` itself.

### Performance

The preamble adds *one* descriptor absorb per proof. The descriptor is on the order of a few hundred bytes (32 bytes for the prime modulus + several 32-byte digests + small per-call scalars). One Blake2b absorb of this size is sub-µs — well below noise on any benchmark we care about. No expected proof-size impact; no expected verify-time impact beyond noise.

Labels add **zero** bytes to the production sponge (Invariant 5) and **zero** runtime cost in production (the `Label` ZST is compiled out).

`AkitaTranscript<Sponge>` is a thin wrapper over the spongefish state; no expected runtime impact vs the previous transcript backend.

## Design

### Architecture

```
   AkitaInstanceDescriptor.canonical_bytes()
        │
        ▼
   spongefish::DomainSeparator
        .new(PROTOCOL_TAG)                         // cfg-gated 64-byte tag:
                                                   //   transcript-blake2b => b"akita-pcs/transcript/v1/blake2b\0..."
                                                   //   transcript-keccak  => b"akita-pcs/transcript/v1/keccak\0..."
        .session(...)                              // TBD: short fixed string, e.g. b"main"
        .instance(<descriptor canonical bytes>)    // P0 preamble lives here
        │
        ├──.to_prover() ──▶ spongefish::ProverState<Sponge>
        │                       wrapped as AkitaTranscript<Sponge>
        │
        └──.to_verifier(narg) ▶ spongefish::VerifierState<'_, Sponge>
                                wrapped as AkitaTranscript<Sponge>

   ┌─────────────────────────────────────────────────────────┐
   │ Both sides use the same trait surface:                   │
   │   AkitaTranscript<Sponge>                                │
   │     .absorb_field(Label, &F)                             │
   │     .absorb_bytes(Label, &[u8])                          │
   │     .absorb_serde(Label, &S)                             │
   │     .squeeze_scalar(Label) -> F                          │
   │     .squeeze_bytes(Label, len) -> Vec<u8>                │
   │                                                          │
   │ Label is ZST unless logging-transcript is enabled;       │
   │ rich {tag, file, line} only in logging builds.           │
   └─────────────────────────────────────────────────────────┘
        │
        ▼
   (logging-transcript feature only)
   LoggingTranscript<Sponge> ── records events ── runs 5 smell checks
                            ── proves event-stream equality
```

### Pillar P0 — Instance descriptor preamble

`AkitaInstanceDescriptor` is a new struct in `akita-types`, structured as four nested sub-structs grouped by **how often each section changes**. Reviewers can audit "is THIS field really per-call?" against the struct shape. Each sub-section's bytes are concatenated under a fixed schema; the whole thing is fed into `spongefish::DomainSeparator.instance(...)`.

Canonical serialization is deterministic across machines and across compilations of this codebase, so prover and verifier produce byte-identical preambles when running on this commit.

```rust
pub struct AkitaInstanceDescriptor {
    /// Bumps invalidate every prior transcript.
    pub version: u32,

    /// Static across the entire codebase for a given `CommitmentConfig`
    /// type-impl. Changes only when the algebraic substrate is reconfigured
    /// (new prime, new ring, new extension flavor). In practice this is
    /// effectively a compile-time constant per binary.
    pub algebra: AlgebraSection,

    /// Static across all proofs with the same deterministic setup identity.
    /// Changes when setup parameters change (different decomposition,
    /// different SIS family, different per-level matrix dimensions, different
    /// setup seed, different protocol feature mode, different planner config).
    /// In practice: changes per `CommitmentConfig` instantiation.
    pub setup: SetupSection,

    /// Per-proof. Changes whenever the final effective verifier schedule for
    /// this call differs. Defense in depth: the call section already determines
    /// this in principle, but binding the post-fallback effective schedule
    /// catches planner-version drift and schedule-rewrite drift.
    pub plan: PlanSection,

    /// Per-commit-and-open call. The visible interface arguments.
    pub call: CallSection,
}

pub struct AlgebraSection {
    /// Characteristic `p` of the base prime field `F_p`, big-endian.
    /// 32 bytes covers primes up to 256 bits, which spans every prime
    /// Akita might plausibly use: today Akita supports up to 128-bit
    /// primes; 256-bit would only enter if Akita adopted pairing-friendly
    /// scalar fields. TODO (transcript-hardening-v2): if Akita ever
    /// adopts a prime > 2^256, widen to `[u8; 64]` or
    /// length-prefix-encode.
    pub prime_modulus_be: [u8; 32],

    /// Cyclotomic index `D` defining `R = F_p[X] / Φ_D(X)`. TODO
    /// (transcript-hardening-v2): add a `ring_family` enum tag if Akita
    /// ever supports non-cyclotomic rings (NTRU, custom minimal polynomial,
    /// etc.). Today the family is implicit-cyclotomic.
    pub ring_dimension_d: u32,

    /// Extension degree `k_F = [F : F_p]` for the message field.
    /// `1` for base-field-native instances. Akita only supports
    /// extensions that embed as subfields of `R`, so `k_F` together
    /// with `(prime_modulus_be, ring_dimension_d)` uniquely determines
    /// the field up to isomorphism — no explicit irreducible polynomial
    /// is needed. TODO (transcript-hardening-v2): if Akita ever supports
    /// non-subfield extensions, add an explicit irreducible polynomial
    /// digest here.
    pub field_extension_degree: u8,

    /// Extension degree for the claim field (sumcheck output field).
    pub claim_extension_degree: u8,

    /// Extension degree for the challenge field (sumcheck challenge field).
    pub challenge_extension_degree: u8,
}

pub struct SetupSection {
    /// `log_basis`, `log_commit_bound`, `log_open_bound`.
    pub decomposition: DecompositionParams,

    /// Which SIS modulus family the security argument is sized against.
    pub sis_modulus_profile: SisModulusProfileId,

    /// Blake2b of canonical bytes of the verifier setup seed
    /// (`max_nuposition_bits`, `max_num_batched_polys`, `max_num_points`,
    /// `max_stride`, `public_matrix_seed`).
    pub setup_seed_digest: [u8; 32],

    /// Protocol-affecting compile-time feature mode. At minimum this records
    /// whether `zk` is enabled; non-protocol features such as `parallel` are not
    /// included because they do not change verifier transcript behavior.
    pub protocol_features: ProtocolFeatureSet,

    // NOTE (planner-refactor): the former `level_params_digest`
    // field has been dropped. The full per-level `LevelParams` are now bound
    // by `PlanSection::effective_schedule_digest` (which digests each step's
    // `LevelParams`, including the root-direct commit layout), so a separate
    // setup-level digest was redundant. `setup_seed_digest` still pins the
    // shared-matrix capacity, and `decomposition` / `sis_modulus_profile` are
    // bound above.
    //
    // TODO (transcript-hardening-v2): if Akita ever adds a per-deployment
    // salt for the transparent setup PRG (so different deployments have
    // different Ajtai matrices for the same params), add it to
    // `AkitaSetupSeed` or another canonical setup-identity field and bind it
    // through this section.
}

pub struct PlanSection {
    /// Blake2b of canonical bytes of the final effective verifier schedule for
    /// this proof, after any root-fold-to-root-direct fallback caused by the
    /// actual opening-point shape and extension-packing support predicates.
    /// Covers the ordered list of steps (fold | direct | terminal), the full
    /// per-level `LevelParams` (ring dimension, log_basis ladder, `{a,b,d}_key`
    /// dimensions/collision bounds, `fold_challenge_config`, block geometry, digit
    /// depths), the root-direct commit layout, and the terminal direct witness
    /// shape. This is a digest of behavior, not just the planner lookup key or
    /// `schedule_key` string.
    pub effective_schedule_digest: [u8; 32],
}

pub struct CallSection {
    /// Number of distinct opening points / point-local commitments.
    pub num_points: u32,

    /// Total number of committed polynomials addressed by the call.
    pub num_polys: u32,

    /// Total number of claimed openings addressed by the call.
    pub num_claims: u32,

    pub basis_mode: BasisMode,                       // Lagrange | Monomial

    /// Common opening-point arity `n = nuposition_bits`.
    /// Today `validate_batched_inputs` rejects mixed arities across points.
    pub opening_point_arity: u32,

    /// Blake2b of canonical bytes of the normalized
    /// `akita_types::proof::incidence::ClaimIncidenceSummary` (existing type
    /// at `crates/akita-types/src/proof/incidence.rs:94`): `nuposition_bits`,
    /// `num_polys_per_point`, `claim_to_point`, `claim_poly_indices`, and
    /// `public_rows: Vec<PublicOpeningRow>`. This distinguishes shapes like
    /// `[2, 1]` from `[1, 2]`, which have the same totals but different
    /// verifier branching and row-batching challenges.
    pub incidence_digest: [u8; 32],

    /// TODO (transcript-hardening-v2): if Akita ever lets the batch
    /// carry mixed point arities, this becomes `Vec<u32>` with per-point
    /// arities (and eventually per-poly arities if that model is added).
}
```

**Construction is symmetric for this PR**: the prover builds `AkitaInstanceDescriptor` from its setup + commit + open arguments and normalized incidence; the verifier builds it from its setup + public verify claims + the final effective schedule it will replay. The cross-side equality unit test in `akita-types` asserts both sides produce byte-identical descriptors from the same inputs.

**Ordering note.** `effective_schedule_digest` requires the effective schedule, but the schedule depends only on inputs known *before* the transcript exists: setup parameters, opening-point shape (from the public verify claims / commit-and-open arguments), and the extension-packing support predicate. Both prover and verifier therefore compute the effective schedule (and its digest) deterministically from those inputs, *before* constructing the `AkitaInstanceDescriptor` and the `spongefish::DomainSeparator`. There is no chicken-and-egg with the transcript.

The descriptor's canonical bytes are passed to `spongefish::DomainSeparator.instance(...)` at transcript construction; the rest of the protocol proceeds positionally.

### Pillar P2 — `AkitaTranscript`, `Label`, `LoggingTranscript`, and smell checks

#### P2.a `Label` ZST + `label!()` macro (production strips labels)

The production transcript bytes are **purely positional**; labels exist only as developer-side diagnostic ergonomics. The mechanism is a feature-gated `Label` type that vanishes in production:

```rust
// crates/akita-transcript/src/label.rs

#[cfg(feature = "logging-transcript")]
mod imp {
    #[derive(Debug, Clone, Copy)]
    pub struct Label {
        pub tag: &'static str,
        pub file: &'static str,
        pub line: u32,
    }
}

#[cfg(not(feature = "logging-transcript"))]
mod imp {
    /// Zero-sized in production and in ordinary tests. The compiler elides
    /// every label argument passed to transcript methods.
    #[derive(Debug, Clone, Copy)]
    pub struct Label;
}

pub use imp::Label;

#[cfg(feature = "logging-transcript")]
#[macro_export]
macro_rules! label {
    ($tag:literal) => {
        $crate::Label { tag: $tag, file: file!(), line: line!() }
    };
}

#[cfg(not(feature = "logging-transcript"))]
#[macro_export]
macro_rules! label {
    ($tag:literal) => { $crate::Label };
}
```

Callsites are uniform:

```rust
transcript.absorb_field(label!("witness_block"), &v);
let r = transcript.squeeze_scalar(label!("challenge_tau"));
```

- **Production and ordinary test builds:** `label!("...")` expands to `Label` (ZST). The label parameter is compiled away. The sponge sees only the bytes of `v`. Positional ordering plus the preamble are the entire soundness story. Ordinary tests exercise this production-shaped API so `size_of::<Label>() == 0` is covered without a special release harness.
- **`--features logging-transcript` build:** `label!(...)` captures `{tag, file, line}`. `LoggingTranscript` records each event as `(Label, EventKind, bytes_digest, bytes_len_or_squeeze_len)`. Smell checks and event-stream-equality tests have rich diagnostics; the schedule dump can be grepped by file/line/tag.

The macro is intentionally literal-only in this PR. If we later want shared constants, add a narrow `$tag:path` arm for a curated label registry; do not accept arbitrary expressions, because broad expressions make it easier to smuggle runtime label captures back into production callsites.

**Consequence for `crates/akita-transcript/src/labels.rs`:** today's `pub const ABSORB_*: &[u8; N]` strings become the literal `tag` arguments to `label!(...)`, used only by callsites and by smell check #3 (known-labels-only). They are no longer absorbed as bytes. The `NOTE` at lines 10-11 is removed; its long-term direction is preserved in §Deferred Follow-Ups as the bolt-lean / algorithm-digest item.

#### P2.b `AkitaTranscript<Sponge>` wrapper

A thin wrapper over `spongefish::ProverState<Sponge>` (prover side) and `spongefish::VerifierState<'_, Sponge>` (verifier side). For this PR, both sides expose the same local protocol trait surface, and the direct `AkitaTranscript` methods take `Label` arguments:

```rust
pub struct AkitaTranscript<F, Sponge = TranscriptSponge> { /* spongefish state */ }

impl<F, Sponge> AkitaTranscript<F, Sponge> {
    fn absorb_field(&mut self, label: Label, value: &F);
    fn absorb_bytes(&mut self, label: Label, bytes: &[u8]);
    fn absorb_serde<S: AkitaSerialize>(&mut self, label: Label, value: &S);
    fn squeeze_scalar(&mut self, label: Label) -> F;
    fn squeeze_bytes(&mut self, label: Label, len: usize) -> Vec<u8>;
}

pub trait Transcript<F> {
    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]);
    fn append_field(&mut self, label: &[u8], value: &F);
    fn append_serde<S: AkitaSerialize>(&mut self, label: &[u8], value: &S);
    fn challenge_scalar(&mut self, label: &[u8]) -> F;
    fn challenge_bytes(&mut self, label: &[u8], len: usize) -> Vec<u8>;
}
```

Implemented directly on the wrapped spongefish states and via the local `Transcript<F>` trait so prover, verifier, and `LoggingTranscript` can share the same protocol plumbing. The old `Blake2bTranscript` / `KeccakTranscript` aliases are not kept. The asymmetric `ProverTranscript` / `VerifierTranscript` split is deferred — see §Deferred Follow-Ups.

#### P2.c `LoggingTranscript<Sponge>` and smell checks

`LoggingTranscript<Sponge>` wraps an `AkitaTranscript` and records `(Label, EventKind, bytes_digest, bytes_len_or_squeeze_len)` events into a thread-local ring buffer. It is gated behind `feature = "logging-transcript"` and is `pub` so integration tests across crates can use it by opting into that feature explicitly.

In addition, when wrapping a verifier-side transcript, `LoggingTranscript` exposes a `record_wire_use(label, canonical_bytes)` method (or equivalently a `TrackedAkitaBatchedProof` view backed by that method). The verifier-side logging harness calls it when a structured proof field becomes an input to a challenge-dependent subprotocol, not merely when the proof is deserialized. This is what smell check #4 inspects. The instrumentation is logging-feature-only; production and ordinary test paths don't compile it.

The five smell checks (enumerated in §Smell Checks) run as ordinary assertion-shaped checks inside `LoggingTranscript`'s drop / verify-time check, active only when the wrapper is. They must fail even under optimized test/profile runs that disable `debug_assert!`.

### Pillar P3 — Differential prop-test

A `proptest` strategy generates random valid batch-incidence shapes, and a deterministic seed corpus covers the remaining expensive axes (`nv`, basis mode, near-canonical `nv=20`). Each case runs the prover and verifier with `LoggingTranscript` wrappers and asserts:

1. `verify` returns `Ok(())`.
2. Prover and verifier event streams are pointwise equal.
3. All five smell checks pass.

This is structurally the same as jolt#1455's `TranscriptConsistencyBlake2b` invariant, but applied one stack level up — at the protocol level rather than at the transcript primitive level.

### Alternatives Considered

1. **Snapshot test the transcript schedule** (rejected earlier in conversation, restated for the record). Snapshot tests lock in the current shape and become a tax on every legitimate protocol change. This spec replaces snapshotting with structural invariants (preamble determines challenges, event-stream equality, smell checks) that survive protocol changes.
2. **Adopt `spongefish` wholesale, NARG-as-proof.** Rejected for this PR. The structured `AkitaBatchedProof` is load-bearing for debug, profile accounting, planner cost models, and schedule-table generation. A NARG migration is a separate spec; see §Deferred Follow-Ups.
3. **Land the prover/verifier trait split (P1) in this PR.** Rejected. A superficial trait split is possible today, but it would not deliver the soundness property we want: the verifier still reads values from structured `AkitaBatchedProof` fields rather than from a transcript-owned byte source. Cleanly enforcing "you cannot use a wire value without absorbing it" at the type level requires either migrating to NARG (rejected, see (2)) or introducing a structured proof-reader adapter / tracked proof view that hides raw proof-field access. Smell check #4 in P2 enforces the same invariant at test time; promoting it to type-level enforcement is a clean follow-up once that adapter or NARG migration is in place.
4. **Land `Bound<T>` in this PR.** Rejected. Without P1 or an equivalent verifier proof-reader adapter, `Bound<T>` can at best prove "some explicit absorb happened"; it cannot prove that the exact wire value later handed to a challenge-dependent subprotocol was the value absorbed at the right transcript epoch. Once the proof-reader boundary lands, `Bound<T>` becomes the natural next defense in depth.
5. **Bind `CommitmentConfig` only, not the schedule plan.** Rejected. Akita's protocol shape branches on `AkitaSchedulePlan` (fold vs direct vs terminal at each level, ring dimension per level, log_basis ladder). The schedule must be bound for the preamble to deliver its claimed property.
6. **Keep per-call labels in the production sponge for redundant safety.** Rejected. Production labels add no soundness beyond positional order + preamble (with P0 in place), they bloat the transcript, and they re-introduce the bug class where a typo in a label string yields a different challenge stream silently. The label-strip-in-production design (P2.a) gives us label diagnostics in tests with zero runtime or transcript-byte cost in production.

## Open Questions

### Open Question 1 — Exact `AkitaInstanceDescriptor` field list — **RESOLVED 2026-05-18**

Resolved as the four-section shape in §Pillar P0. Three inline `TODO (transcript-hardening-v2)` markers flag the assumptions that may need revisiting:

- prime modulus byte width (`[u8; 32]` today, assumes ≤ 256-bit primes);
- ring family (cyclotomic-implicit today; would need a tag for NTRU / custom minimal polynomial);
- mixed opening-point arities (currently rejected; the common arity in `CallSection` suffices today).

Plus two more flagged in `SetupSection`:

- per-deployment setup PRG salt beyond the existing setup seed / public matrix seed (assumed absent today; if added, it must become part of canonical setup identity);
- non-subfield extensions (irreducible polynomial digest would be needed).

These TODOs are the catchment for future revisions.

### Open Question 2 — Adopt `spongefish` as a crate, or just its ideas? — **RESOLVED 2026-05-18: ADOPT**

We adopt `spongefish` as a direct dependency in `akita-transcript`. The decision is independent of whether `jolt-transcript` ships its own spongefish port (a16z/jolt#1455). Akita feeds canonical field bytes and canonical Akita serialization into spongefish through a local prefix-free `FramedBytes` adapter. Spongefish's arkworks codec feature is NOT enabled — Akita does not depend on `ark-ff` / `ark-serialize` at the transcript layer.

### Open Question 3 — Keep per-call labels? — **RESOLVED 2026-05-18: STRIP IN PRODUCTION, KEEP IN DEBUG**

Resolved via the `Label` ZST + `label!()` macro design in §Pillar P2.a. Production and ordinary test builds see a ZST label (Invariant 5). Logging-feature builds capture rich `{tag, file, line}` labels for diagnostics. Zero runtime cost in production.

### Open Question 4 — `Bound<T>` scope in this PR — **RESOLVED 2026-05-18: DEFERRED**

Originally resolved as "narrow scope this PR." Re-resolved on review: most of `Bound<T>`'s value depends on a working prover/verifier trait split (P1) or equivalent proof-reader adapter, which is itself deferred. Without that boundary, `Bound<T>` cannot prove that the exact structured proof value consumed later was the one absorbed at the right transcript epoch. **Deferred to a follow-up PR after P1 / proof-reader work lands.** See §Deferred Follow-Ups.

### Open Question 5 — Fate of `crates/akita-pcs/tests/transcript_trace.rs` — **RESOLVED 2026-05-18: REPLACE**

Replace the stale local diagnostic with durable `LoggingTranscript` hardening tests plus the `transcript_schedule` example. Ad-hoc event-coalescing and parallel-test-mutex scaffolding moves into `LoggingTranscript` itself.

### Open Question 6 — Cutover strategy — **RESOLVED 2026-05-18: ONE PR**

Akita has no back-compat guarantee, so we can do a hard cutover in one PR (replace `Transcript<F>` with `AkitaTranscript<Sponge>`, update all callsites in one pass). Consistent with the workspace's "Full Cutover, No Backward Compatibility" rule.

Counter: PR #88 was already large. A transcript cutover plus the descriptor work plus tests is also non-trivial. Worth confirming we want it in one PR rather than two (e.g., descriptor + spongefish wiring first; LoggingTranscript + smell checks + prop-test second).

My recommendation: one PR. The intermediate state (some callsites on old trait, some on new; descriptor exists but isn't enforced) is worse than either endpoint.

Resolved operationally by Quang's implementation request: one PR, sliced into granular commits and pushes.

## Deferred Follow-Ups

These are *named* future PRs with crisp triggers, so the scope cut in this PR is not silent.

1. **P1: Prover/verifier trait split.**
   *What:* Replace symmetric `AkitaTranscript<F>` with `ProverTranscript<F>` + `VerifierTranscript<F>`, or introduce an equivalent verifier proof-reader API, such that the verifier *cannot* obtain a proof field except through an absorb-as-side-effect operation. Promotes smell check #4 to a compile-time invariant.
   *Trigger:* Either (a) we migrate `AkitaBatchedProof` to a NARG byte tape (see #4 below), or (b) we land a structured proof-reader adapter / tracked proof view that gives `VerifierTranscript::prover_message::<T>(label)` a place to read from while preserving Akita's current structured proof benefits.
   *Blocked on:* a clean answer to "where does the verifier read wire bytes from?" — today the answer is "structured `AkitaBatchedProof` fields," which means a shallow trait split would not prevent raw proof-field reads.

2. **P4: `Bound<T>` at module seams.**
   *What:* A type-level marker tying a value digest to the current transcript epoch. Downstream APIs take `Bound<T>` instead of `T`. Catches the residual bug class "absorbed `x`, passed `x_other` to the next stage."
   *Trigger:* P1 / proof-reader work lands, and `Bound<T>` has private constructors that can only be produced by the transcript/proof-reader boundary for the exact value bytes consumed by the downstream verifier stage.

3. **Algorithm-as-bytes digest.**
   *What:* Keep the fixed human-readable protocol tag and additionally bind `algorithm_digest`, a derived hash of a canonical serialization of the Akita verifier algorithm itself (the bolt-lean direction). Provides domain separation not just between Akita instances but between protocol-shape changes.
   *Trigger:* All of the following exist: (a) a committed verifier-IR spec with a versioned dialect / opcode registry, (b) a canonical prefix-free byte serialization and hash definition, (c) Rust + Lean golden vectors for the same verifier fragment, (d) the Akita verifier is authored or generated in that IR (or a Rust equivalent shares the same dialect catalog), and (e) tests prove the digest is stable under irrelevant formatting changes and changes under protocol-shape changes.
   *Effort:* Multi-week project; will land in its own follow-up spec (tentatively `specs/verifier-program-ir.md`) when activated.

4. **NARG-as-proof migration.**
   *What:* Replace structured `AkitaBatchedProof` wire format with a spongefish NARG byte tape, or a structured-on-top-of-NARG variant. Enables P1 trivially, but risks losing planner accounting, per-step shape introspection, debug visibility, and serialization stability unless the migration spec replaces those explicitly.
   *Trigger:* At least one concrete pressure exists: (a) a structured proof-reader adapter prototype is rejected with documented complexity or unsoundness, (b) a downstream verifier/integration requires a NARG byte tape, or (c) quantified structured-proof overhead blocks a named target. The migration spec must say how planner accounting, per-step shape introspection, debug visibility, and serialization stability are preserved or replaced.

5. **Per-deployment setup PRG salt.**
   *What:* If Akita ever supports user-supplied salt for the transparent setup PRG so different deployments instantiate distinct Ajtai matrices for the same params, bind that salt in `SetupSection`.
   *Trigger:* Such a feature is proposed.

6. **Wider prime moduli or non-subfield extensions.**
   *What:* Widen `prime_modulus_be` past 32 bytes; add an irreducible polynomial digest to `AlgebraSection`.
   *Trigger:* Akita supports primes > 2^256 (unlikely) or extension fields that don't embed as ring subfields (more plausible long-term).

7. **Ring family tag.**
   *What:* Add `ring_family: RingFamily` enum to `AlgebraSection` if Akita ever supports non-cyclotomic rings (NTRU, custom minimal polynomial).
   *Trigger:* Such support is proposed.

## Documentation

- `crates/akita-transcript/README.md` (new) — minimum: trait surface, `Label` / `label!()` discipline, preamble construction, smell-check listing.
- `AGENTS.md` / `CLAUDE.md` — add a small "Transcript" section listing the three active pillars and pointing at this spec for detail.
- `crates/akita-pcs/examples/transcript_schedule` — new example that dumps a schedule for a chosen `(mode, nuposition_bits)` to stdout via `LoggingTranscript`.
- The `NOTE` in `crates/akita-transcript/src/labels.rs:10-11` is removed (its short-term content is now Pillar P2.a; its long-term content is Deferred Follow-Up #3).

## Execution

Suggested order:

1. **`AkitaInstanceDescriptor` in `akita-types`.** Define the four sub-structs (`AlgebraSection`, `SetupSection`, `PlanSection`, `CallSection`), `AkitaSerialize` / `AkitaDeserialize` impls with deterministic canonical encoding, round-trip test, and the cross-side equality unit test (prover and verifier construct identical bytes from their respective input sources).
2. **Spongefish wiring.** Add `spongefish = "0.7"` to workspace `Cargo.toml`. Add `transcript-blake2b` (default) + `transcript-keccak` features in `akita-transcript`, with Cargo all-features/default-unified builds resolving to Blake2b. Feed canonical Akita encodings into spongefish through the local prefix-free `FramedBytes` adapter.
3. **`Label` ZST + `label!()` macro.** Add `label.rs` with the cfg-gated `Label` type and macro. Add `ALL_LABELS: &[&str]` registry (smell check #3 consumes this).
4. **`AkitaTranscript<Sponge>` wrapper.** Thin wrapper over `ProverState<Sponge>` / `VerifierState<'_, Sponge>`. Constructor takes canonical `AkitaInstanceDescriptor` bytes + protocol tag, builds `spongefish::DomainSeparator` with `.instance(...)`, returns the wrapped state. Add the direct method surface (`absorb_field`, `absorb_bytes`, `absorb_serde`, `squeeze_scalar`, `squeeze_bytes`), each method taking a leading `Label`. Add Invariant 5 unit test.
5. **Cutover.** Replace today's `Blake2bTranscript` / `KeccakTranscript` types with `AkitaTranscript`, and keep only the local generic `Transcript<F>` protocol trait needed by prover/verifier/logging wrappers. Migrate all in-tree callsites in one pass (per workspace rule "Full Cutover, No Backward Compatibility"). Verify `cargo fmt && cargo clippy && cargo test` pass.
6. **`LoggingTranscript<Sponge>`.** Promote from `crates/akita-pcs/tests/transcript_trace.rs` into `akita-transcript` under `feature = "logging-transcript"`. Drop the local mutex / coalescing scaffolding. Add the `record_wire_use` instrumentation hook, or a `TrackedAkitaBatchedProof` view backed by that hook, for smell check #4.
7. **Smell checks.** Implement the five smell checks in `LoggingTranscript`. Each is an ordinary assertion-shaped guard active only when the wrapper is.
8. **Hardening tests.** Write `crates/akita-pcs/tests/transcript_hardening.rs` (preamble separation, event-stream equality, smell checks, PR #88 regression) and `transcript_hardening_proptest.rs` (proptest fuzz over batch incidence plus deterministic coverage of `nv` and basis).
9. **Replace `transcript_trace.rs`.** Use durable `LoggingTranscript` hardening tests plus the `transcript_schedule` example.
10. **`transcript_schedule` example.** New small CLI under `crates/akita-pcs/examples/`.
11. **Cross-feature CI.** Verify `cargo test` passes under `--features transcript-blake2b`, `--features transcript-blake2b,logging-transcript`, `--features transcript-keccak,logging-transcript --no-default-features`, and repository all-features CI.

## References

- `a16z/jolt#1455` — Spec to port `jolt-transcript` to spongefish (positional API, no per-call labels). Notable claim we partially endorse: "Jolt's protocol flow is deterministic, so positional order already provides domain separation that per-call labels would redundantly provide." Per the conversation that produced this spec, that claim is correct *only with* a preamble that binds the protocol-shape parameters; without one, positional-only does not protect against instance confusion. Our P0 + label-strip-in-production design is the corrected stance.
- `a16z/jolt#1536` — Michele Orrù's open regression test demonstrating that "opening points and the polynomial commitments are not part of the Fiat-Shamir transformation" in `jolt-openings`. Same bug class as Akita PR #88; smell check #4 here is designed to catch this class.
- `a16z/jolt#1358` (merged) — `quangvdao`'s "bind config parameters to Fiat-Shamir transcript". Closest prior art to this spec's Pillar P0.
- `a16z/jolt#1382` (open) — `quangvdao`'s extension of #1358 to more verifier-boundary bindings.
- [arkworks-rs/spongefish](https://github.com/arkworks-rs/spongefish) — duplex-sponge Fiat-Shamir crate aligned with `draft-irtf-cfrg-fiat-shamir-02`.
- [Plonky3 PR #1603](https://github.com/Plonky3/Plonky3/pull/1603) — `TranscriptBound<T>` type-level binding enforcement, analogous to this spec's deferred Pillar P4.
- `crates/akita-pcs/tests/transcript_trace.rs` (historical main-branch diagnostic) — one-shot diagnostic that surfaced the PR #88 bug.
- `crates/akita-transcript/src/labels.rs:10-11` — the existing NOTE this spec formalizes (short-term in Pillar P2.a, long-term in Deferred Follow-Up #3).
- `specs/terminal-fold-cutover.md` (in `akita/` worktree) — retrospective for PR #88, the bug that motivated this hardening layer.
