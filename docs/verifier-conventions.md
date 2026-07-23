# Akita verifier module map & conventions

Conventions for `crates/akita-verifier`, adopted in Phase 0 of the verifier
refactor ([`specs/akita-verifier-refactor.md`](../specs/akita-verifier-refactor.md))
and enforced thereafter. The verifier is a **linear replay** of the prover's
transcript; the code should read like that replay, top to bottom.

## The pipeline

`batched_verify` is the single public entry point. Everything it does is a
straight-line replay that mirrors the proof structure 1:1:

```
batched_verify
  1. validate_shape         proof/claims well-formed        (no-panic gate)
  2. bind_instance          schedule select + transcript instance descriptor
  3. replay_root            root fold             -> LevelState
  4. replay_suffix_levels   fold per level:       LevelState -> LevelState
       └─ replay_ring_switch    relation-matrix MLE eval at the challenge point
  5. check_terminal         quotient-free terminal witness / direct ring relations
```

All proofs are folded (the ZeroFold / root-direct path was removed in #311);
there is always a root fold, zero or more suffix folds, and a quotient-free
terminal.

Target module tree (one concern per file):

```
lib.rs                    public surface: batched_verify (+ pinned test exports)
protocol/
  orchestration.rs        batched_verify / verify / folded-root dispatch (thin)
  fold/
    engine.rs             per-fold replay engine
    trace_claim.rs        TraceWireAtRoleA + into_claim + remap_*
    eor.rs                extension-opening-reduction replay
  root.rs                 root-level replay
  suffix.rs               suffix-level replay
  terminal/               quotient-free terminal direct + NTT-prefix checks
  ring_switch/
    replay.rs             ring-switch replay
    evaluator.rs          relation-matrix MLE point evaluator
    tensor_challenges.rs  affine-interval challenge factors
  slice_mle/              r-tail / setup-contribution MLE eval (verifier-only)
stages/
  stage1.rs stage2.rs stage3.rs   sumcheck stage verifiers
```

## Rules

**Size budgets.** No function exceeds ~80 lines; no struct exceeds ~10 fields
without a builder or a borrowed context struct. A wide function is a pipeline
that hasn't been split yet; a wide struct is usually several concerns fused.

**Generic naming.** Base field is `F`, extension field is `E`. A
`Foo<F>` whose `F` is only ever instantiated with an extension field is
misnamed — rename it `Foo<E>`.

**No glob imports.** No `use super::*` / `use crate::...::*`. Import each symbol
explicitly so the dependency wall of a module is visible at its top.

**One error-wrap helper.** The repeated
`InvalidInput(format!("... level {i} failed: {err:?}"))` pattern lives in one
helper, not copy-pasted per call site.

**No wrapper slop.** Per [`AGENTS.md`](../AGENTS.md#single-source-of-truth-no-wrapper-slop)
(the #244 cutover): one canonical function per concept, called directly. No
thin pass-throughs, `_for_level` recomposers, or two-statement forwarding files.

**Minimal public surface.** The crate's downstream API is `batched_verify` plus
the small set of replay primitives pinned solely for `akita-pcs` integration
tests (documented in `lib.rs`). Everything else is crate-private. Reaching for a
new `pub` is a signal to move the consumer in or expose a narrower entry point.

**Test-support out of `src/`.** No `#[cfg(test)]` harness/fixture code ships in
`src/`. It moves to `tests/` or behind a `test-support` feature.

## Invariants (do not regress)

1. **Byte-exact prover/verifier consistency.** The verifier replays the prover's
   exact transcript (labels + absorb order) and produces identical accept/reject.
   Guarded by the `akita-pcs` scheme roundtrip tests, the
   [`mixed_d_rejections`](../crates/akita-verifier/tests/mixed_d_rejections.rs)
   rejection tests, and the `profile/akita-recursion` end-to-end path. (The
   legacy-vs-new differential harness that byte-compared transcripts through
   Phase 3 was a temporary migration net, retired with `akita-verifier-legacy` at
   the end of Phase 3 — see the refactor spec's Testing Strategy.)
2. **No-panic boundary.** See [`docs/verifier-contract.md`](verifier-contract.md).
   No refactor may introduce a verifier-reachable
   `unwrap`/`expect`/`unreachable!`/unchecked index/unbounded allocation.
3. **zkVM-guest buildability.** The crate keeps compiling and running inside the
   Jolt guest: syscall-free hot path (no time/thread/file/RNG), no dependency on
   `akita-prover`/`akita-setup`/`akita-pcs`. Guarded by the standalone/guest CI
   builds ([`portability.yml`](../.github/workflows/portability.yml),
   [`jolt-verifier-profile-smoke.yml`](../.github/workflows/jolt-verifier-profile-smoke.yml)).
4. **Behavior preservation over cleverness.** No stage rewrite lands until the
   `akita-pcs` roundtrip + `profile/akita-recursion` e2e + `mixed_d_rejections`
   coverage is green for the shapes it touches. Those catch accept/reject
   regressions but not a transcript-order divergence that still verifies; a
   soundness-sensitive rewrite that needs byte-exact protection should
   reintroduce a golden-transcript net first (the differential harness that
   provided this through Phase 3 is retired).
