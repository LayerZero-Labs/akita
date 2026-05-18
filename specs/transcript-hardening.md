# Spec: Akita Transcript Hardening (Spongefish-aligned, Instance-Descriptor Preamble)

| Field       | Value                                       |
|-------------|---------------------------------------------|
| Author(s)   | @quangvdao + Cursor assistant (Claude Opus 4.7) |
| Created     | 2026-05-18                                  |
| Status      | DRAFT — scope and OQ 1, 2, 3, 4, 5 resolved 2026-05-18; OQ 6 (cutover ordering) remains. Ready for implementation. |
| Branch      | `quang/akita-spongefish-transcript`         |
| PR          | (not yet opened)                            |

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
- Sponge backends are feature-selected within `akita-transcript`: `transcript-blake2b` selects `spongefish::instantiations::Blake2b512` (default), `transcript-keccak` selects `spongefish::instantiations::Keccak`. Both implement `spongefish::DuplexSpongeInterface`.
- The `spongefish::DomainSeparator` is constructed at transcript creation with a fixed 64-byte protocol tag `b"akita-pcs/transcript/v1\0..."` and the `AkitaInstanceDescriptor` canonical bytes as the instance encoding. P0's preamble lives inside spongefish's `DomainSeparator.instance`, not as a separate first-absorb. This is the spongefish-native place for it.
- `Label` (new, in `akita-transcript`) — a **zero-sized type in production builds**, and a rich `{tag, file, line}` capture in `cfg(any(test, feature = "logging-transcript"))` builds. Plus a `label!("...")` macro that captures source location only when logging is enabled. Labels are **never absorbed into the production sponge**. Callsites pass a `Label` to every transcript method; the production build compiles it out entirely. See Pillar P2 for the design.
- `LoggingTranscript<Sponge>` (new, in `akita-transcript`, behind `cfg(any(test, feature = "logging-transcript"))`) — wrapper over `AkitaTranscript` that records each labeled absorb/squeeze event into a thread-local buffer for inspection. Promoted from `crates/akita-pcs/tests/transcript_trace.rs`. Runs the five smell checks listed in §Smell Checks.

### Invariants

The implementation must preserve the following invariants. Where a check is mechanically enforced, the test or assertion is named.

1. **Preamble determines challenges.** For any two `(prover, verifier)` pair runs with different `AkitaInstanceDescriptor` bytes, the first challenge squeezed from the transcript differs. Tested by `transcript_preamble_separation` (new): randomize one field of the descriptor, prove on side A, verify with descriptor B, assert rejection at challenge #1.
2. **Prover/verifier event-stream equality.** For any valid `(config, nv, num_polys, basis)`, the chronological sequence of `Absorb { label, bytes_len }` and `Squeeze { label, len }` events recorded by `LoggingTranscript` on the prover and verifier sides is identical. Tested by `transcript_event_stream_equality` (new differential prop-test, generalized from existing `tests/transcript_trace.rs`).
3. **No silent unbinding (smell-check enforced).** Every value the verifier reads from the proof and uses to derive a challenge is absorbed before any squeeze that depends on it. Enforced in this PR by smell check #4 (`wire_value_before_squeeze_coverage`) inside `LoggingTranscript` at test time, NOT by the type system. Future work (P1, deferred) promotes this to compile-time via a prover/verifier trait split.
4. **No back-compat preservation of today's transcript bytes.** Per the workspace's "Full Cutover, No Backward Compatibility" rule, this PR is free to change the transcript byte layout. Existing proofs do not need to verify after this PR. All in-tree end-to-end tests (`muldiv`, the akita-pcs e2e suite, examples/profile) must continue to pass against the new transcript.
5. **Labels never enter the production sponge.** Asserted by a unit test in `akita-transcript`: build an `AkitaTranscript` in production-build mode, exercise every method, dump the spongefish op log, assert no label-tagged bytes appear. The label parameter on every method is a ZST in release/non-logging builds and is compiled out.

### Non-Goals

1. **Migrating the proof types from structured (`AkitaBatchedProof`) to NARG byte tape.** Akita's proofs are deeply structured (per-level shapes, per-step variants, terminal vs intermediate). The structure is load-bearing for debugging, planner cost models, profile accounting, and serialization stability. We keep `AkitaBatchedProof` as the wire format. The transcript hardening operates *alongside* the structured proof, not by replacing it. (Note: this PR *does* adopt `spongefish` as the sponge / Fiat-Shamir construction; see §Goal. The NARG-as-proof migration is a separate, larger decision and is listed under §Deferred Follow-Ups.)
2. **Adding Poseidon as a transcript sponge.** Out of scope this PR. Blake2b and Keccak are both added (see §Goal, §Acceptance Criteria). Poseidon, if useful for in-circuit recursion later, lands as a follow-up.
3. **Hashing source code or the full verifier program.** A build-time digest of the verifier crate's source is paranoid in a way that costs more than it earns. The principled version of this — hashing a canonical serialization of the verifier algorithm authored in a typed IR (the bolt-lean direction) — is multi-week future work and is listed under §Deferred Follow-Ups.
4. **Waiting for `jolt-transcript` to ship its spongefish port** (a16z/jolt#1455). We adopt `spongefish` directly in `akita-transcript`. If `jolt-transcript`'s port lands later, our consumer integration (jolt-side) can rebase onto it; if it doesn't, Akita's own spongefish layer is sufficient.
5. **Prover/verifier trait split (P1 in earlier conversation).** Deferred; smell check #4 carries the invariant. See §Deferred Follow-Ups for the trigger conditions.
6. **`Bound<T>` at module seams (P4 in earlier conversation).** Deferred; most of its value depends on a working P1. See §Deferred Follow-Ups.

## Evaluation

### Acceptance Criteria

- [ ] `akita-transcript` adds `spongefish` `0.7.x` as a workspace dependency.
- [ ] Two new Cargo features added to `akita-transcript/Cargo.toml`: `transcript-blake2b` (default) and `transcript-keccak`, gating `spongefish::instantiations::Blake2b512` / `spongefish::instantiations::Keccak` respectively. Exactly one is enabled at a time.
- [ ] New `AkitaInstanceDescriptor` type in `akita-types` with the four-section shape from §Pillar P0, a canonical `AkitaSerialize` / `AkitaDeserialize` round-trip test, and a cross-side equality unit test (prover and verifier construct byte-identical descriptors from their respective sources).
- [ ] Today's symmetric `Transcript<F>` trait is replaced by `AkitaTranscript<Sponge>`, implemented as a thin wrapper over `spongefish::ProverState<Sponge>` / `spongefish::VerifierState<'_, Sponge>`. **Both sides use the same trait surface** for this PR. All in-tree callsites are migrated in one pass. No back-compat shim is kept.
- [ ] Local `spongefish::Encoding<[H::U]>` / `spongefish::Decoding<[H::U]>` impls are added for the `akita-field` / `akita-serialization` types we need (concrete prime field elements, extension elements, ring elements, flat vectors, digit blocks, sparse challenges). Each impl is exercised by a round-trip test.
- [ ] `AkitaTranscript` construction takes an `AkitaInstanceDescriptor` and a fixed protocol tag, builds the `spongefish::DomainSeparator` with `.instance(<descriptor bytes>)`, and returns the wrapped prover or verifier state.
- [ ] `Label` ZST + `label!()` macro: every `AkitaTranscript` method takes `Label` as a leading argument. In `cfg(not(any(test, feature = "logging-transcript")))` builds, `Label` is a unit struct and the macro expands to it; the parameter is compiled out. In test / logging builds, `Label` carries `{tag: &'static str, file: &'static str, line: u32}`.
- [ ] Invariant 5 (labels never enter the production sponge) is verified by a dedicated unit test.
- [ ] `LoggingTranscript<Sponge>` is published as a real `pub` helper in `akita-transcript`, behind `cfg(any(test, feature = "logging-transcript"))`, with a small CLI example `crates/akita-pcs/examples/transcript_schedule` that dumps the schedule for a chosen `(mode, num_vars)`.
- [ ] The five smell checks listed in §Smell Checks below all pass on every in-tree end-to-end test run that opts into `LoggingTranscript`.
- [ ] A new `crates/akita-pcs/tests/transcript_hardening.rs` integration test exists and exercises: (a) preamble separation under perturbed descriptors, (b) prover/verifier event-stream equality, (c) the smell checks.
- [ ] A differential property test (using `proptest`) under `crates/akita-pcs/tests/` fuzzes `(config, nv, num_polys, basis)` and asserts event-stream equality + verify-success. Seed corpus covers at minimum: smallest valid `nv`, one mid-range, one near-canonical (`nv=20`).
- [ ] `crates/akita-pcs/tests/transcript_trace.rs` is rewritten as a thin schedule-dump test that delegates to `LoggingTranscript`.
- [ ] `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`, and `cargo test` pass under both `--features transcript-blake2b` (default) and `--features transcript-keccak --no-default-features`.

### Smell Checks

Run by `LoggingTranscript`'s drop / verify-time check on every transcript built in tests:

1. **Preamble is descriptor.** The constructed `spongefish::DomainSeparator` contains `.instance(<canonical descriptor bytes>)` (cross-checked by re-deserializing the instance encoding and comparing to a freshly-constructed `AkitaInstanceDescriptor`).
2. **No zero-byte absorbs.** Every `absorb_*` call has a non-empty bytes payload. (A zero-byte absorb is almost always a serialization bug.)
3. **Known labels only.** Every `Label.tag` is in `akita_transcript::labels::ALL_LABELS`. Catches accidental literal-string labels at callsites.
4. **Wire-value-before-squeeze coverage.** This is the smell check that carries the soundness invariant a prover/verifier trait split would have given us. For every `Squeeze` event in the verifier-side event log, every wire-value-read event since the last `Squeeze` must have a corresponding `Absorb` event recorded in between (and the recorded bytes match the wire-value bytes). Implementation: `LoggingTranscript` records `Wire { label, bytes }` events whenever the verifier deserializes a value from the structured proof; the check correlates `Wire` events with subsequent `Absorb` events under the same label, and fails the test if a `Wire` event is followed by `Squeeze` without an intervening matching `Absorb`. This is the smell check that would have caught PR #88's bug.
5. **Prover and verifier event streams are identical** when both are run with the same descriptor. Asserted at the end of the integration test by zipping the two `LoggingTranscript` event vectors.

Each smell check is a `debug_assert!`-shaped guard inside `LoggingTranscript` that fires only when the wrapper is active.

### Testing Strategy

**New tests:**

- `crates/akita-types/src/instance_descriptor.rs::tests::canonical_encoding_roundtrip` — `AkitaSerialize` / `AkitaDeserialize` round-trip on the four-section descriptor.
- `crates/akita-types/src/instance_descriptor.rs::tests::cross_side_equality` — construct from prover-side inputs (setup + commit + open args) vs verifier-side inputs (setup + proof header); assert byte-identical.
- `crates/akita-transcript/src/lib.rs::tests::labels_never_in_production_sponge` — Invariant 5.
- `crates/akita-pcs/tests/transcript_hardening.rs::preamble_separation` — perturbing each descriptor field changes challenge #1.
- `crates/akita-pcs/tests/transcript_hardening.rs::event_stream_equality_small` — single fixture, full inspection.
- `crates/akita-pcs/tests/transcript_hardening.rs::smell_checks_pass` — exercises each numbered smell check above.
- `crates/akita-pcs/tests/transcript_hardening_proptest.rs` — `proptest`-driven fuzz over `(config, nv, num_polys, basis)`, asserting event-stream equality and verify-success.
- `crates/akita-pcs/tests/transcript_hardening.rs::pr88_regression` — explicit replay of the PR #88 setup; smell check #4 must fail if `final_w` is mutated post-absorb. Locks in the bug class.

**Existing tests that must keep passing (with transcript-bytes updated):**

- The full `cargo test` workspace run.
- `AKITA_MODE=onehot AKITA_NUM_VARS=20 cargo run --release --example profile` smoke run.

**Tests that are rewritten:**

- `crates/akita-pcs/tests/transcript_trace.rs` — becomes a thin schedule-dump test using `LoggingTranscript`. All ad-hoc event-coalescing / parallel-test-mutex scaffolding moves into `LoggingTranscript` itself.

### Performance

The preamble adds *one* descriptor absorb per proof. The descriptor is on the order of ~120 bytes (32 bytes for the prime modulus + small per-field tags + three 32-byte digests + small per-call scalars). One Blake2b absorb of 120 bytes is sub-µs — well below noise on any benchmark we care about. No expected proof-size impact; no expected verify-time impact beyond noise.

Labels add **zero** bytes to the production sponge (Invariant 5) and **zero** runtime cost in production (the `Label` ZST is compiled out).

`AkitaTranscript<Sponge>` is a thin wrapper over the spongefish state; no expected runtime impact vs the existing `Blake2bTranscript`.

## Design

### Architecture

```
   AkitaInstanceDescriptor.canonical_bytes()
        │
        ▼
   spongefish::DomainSeparator
        .new(b"akita-pcs/transcript/v1\0...")    // fixed 64-byte protocol tag
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
   │     .absorb_ring(Label, &R)                              │
   │     .squeeze_scalar(Label) -> F                          │
   │     .squeeze_bytes(Label, len) -> Vec<u8>                │
   │                                                          │
   │ Label is ZST in production (compiled away);              │
   │ rich {tag, file, line} in test / logging builds.         │
   └─────────────────────────────────────────────────────────┘
        │
        ▼
   (test / logging only)
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

    /// Static across all proofs with the same setup. Changes when setup
    /// parameters change (different decomposition, different SIS family,
    /// different per-level matrix dimensions, different planner config).
    /// In practice: changes per `CommitmentConfig` instantiation.
    pub setup: SetupSection,

    /// Per-proof. Changes whenever the planner-computed shape for this
    /// (num_vars, num_polys, num_claims, basis) tuple differs. Defense
    /// in depth: the call section already determines this in principle,
    /// but binding the actual planned shape catches planner-version drift.
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
    pub sis_modulus_family: SisModulusFamily,

    /// Blake2b of canonical bytes of `Vec<LevelParams>`. Each
    /// `LevelParams` captures (ring_dimension_per_level, log_basis,
    /// {a,b,d}_key dims and collision_inf, num_blocks, block_len).
    /// TODO (transcript-hardening-v2): if Akita ever adds a
    /// per-deployment salt for the transparent setup PRG (so different
    /// deployments have different Ajtai matrices for the same params),
    /// bind that salt here.
    pub level_params_digest: [u8; 32],
}

pub struct PlanSection {
    /// Blake2b of canonical bytes of the resolved `AkitaSchedulePlan`
    /// for this `(num_vars, num_polys, num_claims, basis)`. Covers
    /// the ordered list of steps (fold | direct | terminal) and the
    /// per-level choices the planner emits. Defense in depth against
    /// planner-version drift.
    pub schedule_plan_digest: [u8; 32],
}

pub struct CallSection {
    pub num_polys: u32,
    pub num_claims: u32,
    pub basis_mode: BasisMode,                       // Lagrange | Monomial
    pub opening_point_arity: u32,                    // n = num_vars
    /// TODO (transcript-hardening-v2): if Akita ever lets the batch
    /// carry per-poly opening points (instead of one shared across the
    /// batch), this becomes `Vec<u32>` with the per-poly arities. Today
    /// the batch shares one opening point so a single arity suffices.
}
```

**Construction is symmetric for this PR**: the prover builds `AkitaInstanceDescriptor` from its setup + commit + open arguments; the verifier builds it from its setup + the proof's shape header. The cross-side equality unit test in `akita-types` asserts both sides produce byte-identical descriptors from the same inputs.

The descriptor's canonical bytes are passed to `spongefish::DomainSeparator.instance(...)` at transcript construction; the rest of the protocol proceeds positionally.

### Pillar P2 — `AkitaTranscript`, `Label`, `LoggingTranscript`, and smell checks

#### P2.a `Label` ZST + `label!()` macro (production strips labels)

The production transcript bytes are **purely positional**; labels exist only as developer-side diagnostic ergonomics. The mechanism is a feature-gated `Label` type that vanishes in production:

```rust
// crates/akita-transcript/src/label.rs

#[cfg(any(test, feature = "logging-transcript"))]
mod imp {
    #[derive(Debug, Clone, Copy)]
    pub struct Label {
        pub tag: &'static str,
        pub file: &'static str,
        pub line: u32,
    }
}

#[cfg(not(any(test, feature = "logging-transcript")))]
mod imp {
    /// Zero-sized in production. The compiler elides every label argument
    /// passed to transcript methods.
    #[derive(Debug, Clone, Copy)]
    pub struct Label;
}

pub use imp::Label;

#[cfg(any(test, feature = "logging-transcript"))]
#[macro_export]
macro_rules! label {
    ($tag:literal) => {
        $crate::Label { tag: $tag, file: file!(), line: line!() }
    };
}

#[cfg(not(any(test, feature = "logging-transcript")))]
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

- **Production build:** `label!("...")` expands to `Label` (ZST). The label parameter is compiled away. The sponge sees only the bytes of `v`. Positional ordering plus the preamble are the entire soundness story.
- **Test / `--features logging-transcript` build:** `label!(...)` captures `{tag, file, line}`. `LoggingTranscript` records each event as `(Label, EventKind, bytes_or_len)`. Smell checks and event-stream-equality tests have rich diagnostics; the schedule dump can be grepped by file/line/tag.

**Consequence for `crates/akita-transcript/src/labels.rs`:** today's `pub const ABSORB_*: &[u8; N]` strings become the `tag` arguments to `label!(...)` (or a shared `pub const TAG_ABSORB_W: &str = "absorb_w";` list), used only by callsites and by smell check #3 (known-labels-only). They are no longer absorbed as bytes. The `NOTE` at lines 10-11 is removed; its long-term direction is preserved in §Deferred Follow-Ups as the bolt-lean / algorithm-digest item.

#### P2.b `AkitaTranscript<Sponge>` wrapper

A thin newtype over `spongefish::ProverState<Sponge>` (prover side) and `spongefish::VerifierState<'_, Sponge>` (verifier side). For this PR, both sides expose the same trait surface:

```rust
pub trait AkitaTranscript<F> {
    fn absorb_field(&mut self, label: Label, value: &F);
    fn absorb_bytes(&mut self, label: Label, bytes: &[u8]);
    fn absorb_ring<R: AkitaRing>(&mut self, label: Label, value: &R);
    fn squeeze_scalar(&mut self, label: Label) -> F;
    fn squeeze_bytes(&mut self, label: Label, len: usize) -> Vec<u8>;
}
```

Implemented directly on the wrapped spongefish states via the orphan rule (the trait is locally defined). The asymmetric `ProverTranscript` / `VerifierTranscript` split is deferred — see §Deferred Follow-Ups.

#### P2.c `LoggingTranscript<Sponge>` and smell checks

`LoggingTranscript<Sponge>` wraps an `AkitaTranscript` and records `(Label, EventKind, bytes_or_len)` events into a thread-local ring buffer. It is gated behind `cfg(any(test, feature = "logging-transcript"))` and is `pub` so integration tests across crates can use it.

In addition, when wrapping a verifier-side transcript, `LoggingTranscript` exposes a `record_wire_read(label, bytes)` method that integration test harnesses call whenever the verifier deserializes a value from `AkitaBatchedProof` and is about to use it. This is what smell check #4 inspects. The instrumentation is test-only; production paths don't compile it.

The five smell checks (enumerated in §Smell Checks) run as `debug_assert!`-shaped guards inside `LoggingTranscript`'s drop / verify-time check, active only when the wrapper is.

### Pillar P3 — Differential prop-test

A `proptest` strategy generates random `(config, nv, num_polys, basis)` tuples that are valid per the planner's constraints (skipping anything the planner would reject), runs the prover and verifier with `LoggingTranscript` wrappers, and asserts:

1. `verify` returns `Ok(())`.
2. Prover and verifier event streams are pointwise equal.
3. All five smell checks pass.

This is structurally the same as jolt#1455's `TranscriptConsistencyBlake2b` invariant, but applied one stack level up — at the protocol level rather than at the transcript primitive level.

### Alternatives Considered

1. **Snapshot test the transcript schedule** (rejected earlier in conversation, restated for the record). Snapshot tests lock in the current shape and become a tax on every legitimate protocol change. This spec replaces snapshotting with structural invariants (preamble determines challenges, event-stream equality, smell checks) that survive protocol changes.
2. **Adopt `spongefish` wholesale, NARG-as-proof.** Rejected for this PR. The structured `AkitaBatchedProof` is load-bearing for debug, profile accounting, planner cost models, and schedule-table generation. A NARG migration is a separate spec; see §Deferred Follow-Ups.
3. **Land the prover/verifier trait split (P1) in this PR.** Rejected. The verifier today reads values from structured `AkitaBatchedProof` fields rather than from a NARG byte tape; cleanly enforcing "you cannot use a wire value without absorbing it" at the type level requires either migrating to NARG (rejected, see (2)) or building a NARG-vs-structured adapter (non-trivial). Smell check #4 in P2 enforces the same invariant at test time; promoting it to type-level enforcement is a clean follow-up once the adapter or NARG migration is in place.
4. **Land `Bound<T>` in this PR.** Rejected. Without P1's trait split, `Bound<T>` is a marker without enforcement (anyone can construct it). Once P1 lands, `Bound<T>` becomes the natural next defense in depth.
5. **Bind `CommitmentConfig` only, not the schedule plan.** Rejected. Akita's protocol shape branches on `AkitaSchedulePlan` (fold vs direct vs terminal at each level, ring dimension per level, log_basis ladder). The schedule must be bound for the preamble to deliver its claimed property.
6. **Keep per-call labels in the production sponge for redundant safety.** Rejected. Production labels add no soundness beyond positional order + preamble (with P0 in place), they bloat the transcript, and they re-introduce the bug class where a typo in a label string yields a different challenge stream silently. The label-strip-in-production design (P2.a) gives us label diagnostics in tests with zero runtime or transcript-byte cost in production.

## Open Questions

### Open Question 1 — Exact `AkitaInstanceDescriptor` field list — **RESOLVED 2026-05-18**

Resolved as the four-section shape in §Pillar P0. Three inline `TODO (transcript-hardening-v2)` markers flag the assumptions that may need revisiting:

- prime modulus byte width (`[u8; 32]` today, assumes ≤ 256-bit primes);
- ring family (cyclotomic-implicit today; would need a tag for NTRU / custom minimal polynomial);
- per-poly opening-point arities (single shared arity today).

Plus two more flagged in `SetupSection`:

- per-deployment setup PRG salt (assumed absent today);
- non-subfield extensions (irreducible polynomial digest would be needed).

These TODOs are the catchment for future revisions.

### Open Question 2 — Adopt `spongefish` as a crate, or just its ideas? — **RESOLVED 2026-05-18: ADOPT**

We adopt `spongefish` as a direct dependency in `akita-transcript`. The decision is independent of whether `jolt-transcript` ships its own spongefish port (a16z/jolt#1455). Implement `spongefish::Encoding<[H::U]>` / `Decoding<[H::U]>` for our `akita-field` / `akita-serialization` types directly in `akita-transcript`. Spongefish's arkworks codec feature is NOT enabled — Akita does not depend on `ark-ff` / `ark-serialize` at the transcript layer.

### Open Question 3 — Keep per-call labels? — **RESOLVED 2026-05-18: STRIP IN PRODUCTION, KEEP IN DEBUG**

Resolved via the `Label` ZST + `label!()` macro design in §Pillar P2.a. Production sponge sees no labels (Invariant 5). Test / logging builds capture rich `{tag, file, line}` labels for diagnostics. Zero runtime cost in production.

### Open Question 4 — `Bound<T>` scope in this PR — **RESOLVED 2026-05-18: DEFERRED**

Originally resolved as "narrow scope this PR." Re-resolved on review: most of `Bound<T>`'s value depends on a working prover/verifier trait split (P1), which is itself deferred. Without P1, `Bound<T>` is a marker without enforcement. **Deferred to a follow-up PR after P1 lands.** See §Deferred Follow-Ups.

### Open Question 5 — Fate of `crates/akita-pcs/tests/transcript_trace.rs` — **RESOLVED 2026-05-18: REWRITE**

Rewrite as a thin schedule-dump test that delegates to `LoggingTranscript` and lives next to `transcript_hardening.rs`. Ad-hoc event-coalescing and parallel-test-mutex scaffolding moves into `LoggingTranscript` itself.

### Open Question 6 — Cutover strategy

Akita has no back-compat guarantee, so we can do a hard cutover in one PR (replace `Transcript<F>` with `AkitaTranscript<Sponge>`, update all callsites in one pass). Consistent with the workspace's "Full Cutover, No Backward Compatibility" rule.

Counter: PR #88 was already large. A transcript cutover plus the descriptor work plus tests is also non-trivial. Worth confirming we want it in one PR rather than two (e.g., descriptor + spongefish wiring first; LoggingTranscript + smell checks + prop-test second).

My recommendation: one PR. The intermediate state (some callsites on old trait, some on new; descriptor exists but isn't enforced) is worse than either endpoint.

**Status: needs confirmation.**

## Deferred Follow-Ups

These are *named* future PRs with crisp triggers, so the scope cut in this PR is not silent.

1. **P1: Prover/verifier trait split.**
   *What:* Replace symmetric `AkitaTranscript<F>` with `ProverTranscript<F>` + `VerifierTranscript<F>` such that the verifier *cannot* read a wire value without absorbing it as a side effect. Promotes smell check #4 to a compile-time invariant.
   *Trigger:* Either (a) we migrate `AkitaBatchedProof` to a NARG byte tape (see #4 below), or (b) we build a NARG-vs-structured adapter that gives `VerifierTranscript::prover_message::<T>(label)` a place to read from.
   *Blocked on:* a clean answer to "where does the verifier read wire bytes from?" — today the answer is "structured `AkitaBatchedProof` fields," which isn't compatible with spongefish's `narg_string` model.

2. **P4: `Bound<T>` at module seams.**
   *What:* A type-level marker that a value has been absorbed by the current transcript. Downstream APIs take `Bound<T>` instead of `T`. Catches the residual bug class "absorbed `x`, passed `x_other` to the next stage."
   *Trigger:* P1 lands.

3. **Algorithm-as-bytes digest.**
   *What:* Replace the fixed `b"akita-pcs/transcript/v1\0..."` protocol tag with a derived hash of a canonical serialization of the Akita verifier algorithm itself (the bolt-lean direction). Provides domain separation not just between Akita instances but between protocol-shape changes.
   *Trigger:* (a) bolt-lean defines a canonical serializer for the protocol dialect, (b) Akita authors the verifier in that dialect (or a Rust equivalent shares the dialect catalog), (c) we have a deterministic byte format both languages agree on.
   *Effort:* Multi-week project; will land in its own follow-up spec (tentatively `specs/verifier-program-ir.md`) when activated.

4. **NARG-as-proof migration.**
   *What:* Replace structured `AkitaBatchedProof` wire format with a spongefish NARG byte tape (or a structured-on-top-of-NARG variant). Enables P1 trivially. Loses the planner-accounting and per-step-shape introspection benefits of structured proofs.
   *Trigger:* We have a use case where the structured-proof costs outweigh the benefits (none today).

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
- `crates/akita-pcs/examples/transcript_schedule` — new example that dumps a schedule for a chosen `(mode, num_vars)` to stdout via `LoggingTranscript`.
- The `NOTE` in `crates/akita-transcript/src/labels.rs:10-11` is removed (its short-term content is now Pillar P2.a; its long-term content is Deferred Follow-Up #3).

## Execution

Suggested order (only Open Question 6 still needs confirmation; can proceed in parallel with steps 1–3):

1. **`AkitaInstanceDescriptor` in `akita-types`.** Define the four sub-structs (`AlgebraSection`, `SetupSection`, `PlanSection`, `CallSection`), `AkitaSerialize` / `AkitaDeserialize` impls with deterministic canonical encoding, round-trip test, and the cross-side equality unit test (prover and verifier construct identical bytes from their respective input sources).
2. **Spongefish wiring.** Add `spongefish = "0.7"` to workspace `Cargo.toml`. Add `transcript-blake2b` (default) + `transcript-keccak` features in `akita-transcript`. Implement `spongefish::Encoding<[H::U]>` / `Decoding<[H::U]>` for the akita-field / akita-serialization types we need.
3. **`Label` ZST + `label!()` macro.** Add `label.rs` with the cfg-gated `Label` type and macro. Add `ALL_LABELS: &[&str]` registry (smell check #3 consumes this).
4. **`AkitaTranscript<Sponge>` wrapper.** Thin newtype over `ProverState<Sponge>` / `VerifierState<'_, Sponge>`. Constructor takes `AkitaInstanceDescriptor` + protocol tag, builds `spongefish::DomainSeparator` with `.instance(...)`, returns the wrapped state. Add the trait surface (`absorb_field`, `absorb_bytes`, `absorb_ring`, `squeeze_scalar`, `squeeze_bytes`), each method taking a leading `Label`. Add Invariant 5 unit test.
5. **Cutover.** Replace today's `Blake2bTranscript` / `KeccakTranscript` types and the symmetric `Transcript<F>` trait. Migrate all in-tree callsites in one pass, passing `label!("...")` to every call (per workspace rule "Full Cutover, No Backward Compatibility"). Verify `cargo fmt && cargo clippy && cargo test` pass.
6. **`LoggingTranscript<Sponge>`.** Promote from `crates/akita-pcs/tests/transcript_trace.rs` into `akita-transcript` under `cfg(any(test, feature = "logging-transcript"))`. Drop the local mutex / coalescing scaffolding. Add the `record_wire_read` instrumentation hook for smell check #4.
7. **Smell checks.** Implement the five smell checks in `LoggingTranscript`. Each is a `debug_assert!`-shaped guard active only when the wrapper is.
8. **Hardening tests.** Write `crates/akita-pcs/tests/transcript_hardening.rs` (preamble separation, event-stream equality, smell checks, PR #88 regression) and `transcript_hardening_proptest.rs` (proptest fuzz over `(config, nv, num_polys, basis)`).
9. **Rewrite `transcript_trace.rs`.** Thin schedule-dump test using `LoggingTranscript`.
10. **`transcript_schedule` example.** New small CLI under `crates/akita-pcs/examples/`.
11. **Cross-feature CI.** Verify `cargo test` passes under both `--features transcript-blake2b` and `--features transcript-keccak --no-default-features`.

## References

- `a16z/jolt#1455` — Spec to port `jolt-transcript` to spongefish (positional API, no per-call labels). Notable claim we partially endorse: "Jolt's protocol flow is deterministic, so positional order already provides domain separation that per-call labels would redundantly provide." Per the conversation that produced this spec, that claim is correct *only with* a preamble that binds the protocol-shape parameters; without one, positional-only does not protect against instance confusion. Our P0 + label-strip-in-production design is the corrected stance.
- `a16z/jolt#1536` — Michele Orrù's open regression test demonstrating that "opening points and the polynomial commitments are not part of the Fiat-Shamir transformation" in `jolt-openings`. Same bug class as Akita PR #88; smell check #4 here is designed to catch this class.
- `a16z/jolt#1358` (merged) — `quangvdao`'s "bind config parameters to Fiat-Shamir transcript". Closest prior art to this spec's Pillar P0.
- `a16z/jolt#1382` (open) — `quangvdao`'s extension of #1358 to more verifier-boundary bindings.
- [arkworks-rs/spongefish](https://github.com/arkworks-rs/spongefish) — duplex-sponge Fiat-Shamir crate aligned with `draft-irtf-cfrg-fiat-shamir-02`.
- [Plonky3 PR #1603](https://github.com/Plonky3/Plonky3/pull/1603) — `TranscriptBound<T>` type-level binding enforcement, analogous to this spec's deferred Pillar P4.
- `crates/akita-pcs/tests/transcript_trace.rs` (current main) — one-shot diagnostic that surfaced the PR #88 bug.
- `crates/akita-transcript/src/labels.rs:10-11` — the existing NOTE this spec formalizes (short-term in Pillar P2.a, long-term in Deferred Follow-Up #3).
- `specs/terminal-fold-cutover.md` (in `akita/` worktree) — retrospective for PR #88, the bug that motivated this hardening layer.
