**NEVER COMMIT THIS FILE.**

# EOR sumcheck crate-boundary worklog

## Goal & Scope
Refactor the extension-opening-reduction (EOR) sumcheck so concrete prover instance code lives in `akita-prover`, not `akita-sumcheck`, while keeping proof bytes and transcript bytes unchanged.
This worklog tracks the crate-boundary cutover work on branch `taghi/perf/eor-sc` and the corresponding spec rewrite at `specs/eor-sumcheck-prover-acceleration.md`.

## Starting State
- Branch: `taghi/perf/eor-sc`
- Worktree: `/Users/quang.dao/Documents/SNARKs/akita-pr-136-eor-sc`
- Related spec: `specs/eor-sumcheck-prover-acceleration.md`

## Plan
- Rewrite `specs/eor-sumcheck-prover-acceleration.md` to describe the intended final architecture (not a retrospective), including cutover checklist and guardrails.
- Move `crates/akita-sumcheck/src/extension_opening_reduction/` into `crates/akita-prover/src/protocol/extension_opening_reduction/`.
- Replace `ExtensionOpeningReductionSumcheck` with generic sumcheck drivers in prover and verifier code.
- Move EOR tests out of `akita-sumcheck` into `akita-prover` (or `akita-pcs` integration tests), and update benches accordingly.
- Run `cargo fmt`, `cargo clippy --all --all-targets -D warnings`, `cargo test`, and `cargo doc --all-features` as the verification gate.

## Decisions
- **[2026-06-02] Treat `specs/eor-sumcheck-prover-acceleration.md` as a living design spec, not a finished retrospective.**
  Rewrite the architecture section to match the intended crate boundary, even if it differs from earlier commits on the branch.
- **[2026-06-02] Put EOR shared tensor/output helpers in `akita-types`.**
  Prover-owned witness state moved to `akita-prover::protocol::extension_opening_reduction`.
  Pure tensor helpers such as tensor partial recomposition, tensor table materialization, tensor factor evaluation, and final output checks moved to `akita-types`.
  Reason: `akita-verifier` must not depend on `akita-prover`, and `akita-sumcheck` should stay protocol-independent.

## Deviations

## Tradeoffs

## Open Questions
- **[2026-06-02] Resolved: where should EOR verifier-shared helper functions live after the move?**
  Working assumption: put protocol-specific but prover and verifier shared EOR helpers in `akita-types`, keeping `akita-prover` as the only crate that owns witness-bearing prover state.
  Resolution: implemented in `akita-types`.

## Slice Retrospectives

### 2026-06-02 retrospective: spec rewrite and worklog bootstrap

**Bottom line:** created the worklog and rewrote `specs/eor-sumcheck-prover-acceleration.md` to describe the intended final crate boundary and an ordered cutover checklist.

- Risk: the spec now describes the target architecture, so implementation must be kept in lockstep with the checklist and acceptance criteria.
- Verification:
  - `git status --short` shows `WORKLOG-NEVER-COMMIT.md` is untracked.
  - Removed em-dash punctuation from the rewritten spec.

### 2026-06-02 retrospective: EOR crate-boundary cutover

**Bottom line:** EOR prover state and tests moved out of `akita-sumcheck`.
`akita-sumcheck` no longer contains EOR modules or exports; prover and verifier paths use generic sumcheck drivers with EOR helpers supplied by `akita-types`.

- Risk: `akita-types::extension_opening_reduction` includes both verifier-needed helpers and pure table-materialization helpers used by prover code.
  This keeps one implementation of the tensor algebra, but it means `akita-types` now has an optional `rayon` dependency behind its `parallel` feature.
- Non-issue checked: ZK EOR replay still compiles after replacing the old `ExtensionOpeningReductionSumcheck::verify_zk` wrapper with a local verifier helper that preserves the masked-claim recurrence.
- Verification:
  - `cargo fmt -q`
  - `cargo check --all-targets -p akita-sumcheck -p akita-types -p akita-prover -p akita-verifier`
  - `cargo check --all-targets -p akita-sumcheck -p akita-types -p akita-prover -p akita-verifier --features zk`
  - `cargo test -p akita-prover --test extension_opening_reduction` -> 22 passed
  - `cargo clippy --all-targets -p akita-sumcheck -p akita-types -p akita-prover -p akita-verifier -- -D warnings`
  - `cargo clippy --all-targets -p akita-sumcheck -p akita-types -p akita-prover -p akita-verifier --features zk -- -D warnings`

## Phase 2 plan (prover unification + verifier instance + type tightening + cleanup)
Approved by user 2026-06-02 ("do all of it"). Slices, each with its own retrospective:
1. Unify EOR prover into one struct (port fused fold into the batched `Dense/Dense` term + per-term cache, repoint recursive, delete the standalone dense struct, drop `Batched` from names). Byte-identical + A/B recursive-heavy modes.
2. `ExtensionOpeningReductionVerifier` implementing `SumcheckInstanceVerifier` (+ `ZkSumcheckFinalRelation`); collapse the two open-coded verifier helpers in `levels.rs` across root + recursive.
3. Tighten `ExtensionOpeningWitness`/`ExtensionOpeningFactor` so `Dense×Tensor` is unrepresentable (removes `sparse.rs` `unreachable!`s).
4. Clean the `akita-types` re-export pass-through in `mod.rs`; finish spec Phase-5 (`cargo doc`, byte-identical boxes).

## Phase 2 Decisions
- **[2026-06-02] Constructor naming after collapsing dense + batched provers into one `ExtensionOpeningReductionProver`.**
  Kept `new(terms, input_claim)` as the general (multi-term) constructor (matches the existing batched call sites in `root_extension.rs`) and added `from_dense_tables(witness_evals, factor_evals)` as the single-dense convenience used by `recursive.rs`.
  Reason: a single `new` cannot mean both `(terms, claim)` and `(witness, factor)`; `from_dense_tables` reads clearly at the recursive call site and keeps `new` for the general form.
- **[2026-06-02] Per-term fused-fold cache lives on the term, coeff-scaled.**
  `ExtensionOpeningReductionTerm` now carries `cached_accumulate: Option<(E,E)>`. `ingest_challenge` fuses fold+accumulate for the `Dense/Dense` term when `len >= 4` and caches `coeff * (constant, quadratic)`; `accumulate_into` consumes the cache or falls back to `accumulate_round`.
  Reason: this is exactly the standalone dense prover's strategy; coeff-scaling the cache keeps the round coefficients byte-identical to `accumulate_dense_round(.., coeff)`. Field addition is exact/commutative, so summing cached (dense) and freshly-accumulated (sparse) terms in any order is byte-identical.

## Follow-ups
- Run the full workspace acceptance gate (`cargo test`, `cargo doc -q --no-deps --all-features`) and the profile byte-identical proof gate.

