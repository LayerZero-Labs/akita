# Spec: Akita Verifier Refactor (minimal, auditable, rewritable)

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Author(s)     | sumchecker                                 |
| Created       | 2026-07-20                                 |
| Status        | active                                     |
| PR            |                                            |
| Supersedes    |                                            |
| Superseded-by |                                            |
| Book-chapter  | how/verification.md                        |

## Summary

`akita-verifier` is soundness-critical, byte-locked to the prover's Fiat–Shamir
transcript, and has accreted hard-to-audit orchestration ("AI spaghetti"):
200–280-line functions, a 24-field god struct built three different ways,
wrapper-around-a-boolean machinery, dead parameter plumbing, and test-only code
shipped in `src/`. The *math* is not the problem — it is already shared and
reasonably clean (it lives in `akita-types`/`akita-sumcheck`/`akita-algebra`).
The problem is the ~5,300 LOC of verifier-**owned** replay glue wrapped around
that math.

This spec defines how to rewrite that glue into a **minimal, linear, auditable
replay pipeline** — in-place, in the current monorepo, behind a frozen public
contract, with a **differential harness as the safety net** so the rewrite is
mechanically verifiable rather than risky. It is explicitly *not* a rewrite of
the shared protocol math, and it *defers* the planner/config boundary work to a
final phase with its own follow-up spec.

## Intent

### Goal

Rewrite `akita-verifier`'s verifier-owned orchestration into a small pipeline of
pure, single-concern replay stages that mirror the proof structure 1:1, keeping
`batched_verify` byte-for-byte behavior-identical and the crate as minimal as
possible (fewest lines, narrowest public surface, fewest dependencies).

Frozen contract (the migration boundary — **must not change**):

```rust
pub fn batched_verify<Cfg, T>(
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    setup: &AkitaVerifierSetup<Cfg::Field>,
    transcript: &mut T,
    claims: OpeningClaims<'_, Cfg::ExtField, &Commitment<Cfg::Field>>,
    basis: BasisMode,
) -> Result<(), AkitaError>
```
(`crates/akita-verifier/src/protocol/core/verify.rs:206`.) Both consumers —
`akita-pcs` (library) and the `profile/akita-recursion` zkVM guest — need *only*
this function.

> **Rebase note (this spec re-anchored onto current main, post-#311/#312).**
> Two protocol PRs landed after this initiative's original branch point.
> #311 ("folded-only proofs and quotient-free terminals") already: (a) dropped
> the `setup_contribution_mode` parameter from `batched_verify` (contract is now
> 5-arg — the old Phase-1 "dead-parameter-plumbing" cleanup is upstream);
> (b) **removed the ZeroFold / root-direct path entirely** (all proofs are now
> folded, with a quotient-free terminal); and (c) removed the `proof/` module
> and the `cleartext_witness_opening_matches` export. #312 ("unify and optimize
> Stage 1 digit range proofs") reworked `stages/stage1.rs` internals but left
> the `batched_verify` contract and proof shape unchanged. The target tree and
> proof-shape matrix below are updated accordingly; the concrete refactor
> targets in Design must be re-verified against the current source.

Target module tree (by concern; each file one concern):

```
lib.rs                    public surface: batched_verify (+ pinned test exports)
protocol/
  orchestration.rs        batched_verify / verify / folded-root dispatch (thin)
  fold/
    engine.rs             per-fold replay engine
    trace_claim.rs        TraceWireAtRoleA + into_claim + remap_* (split from fold.rs)
    eor.rs                extension-opening-reduction replay
  root.rs                 root-level replay
  suffix.rs               suffix-level replay
  terminal/               quotient-free terminal direct + NTT prefix checks
  ring_switch/
    replay.rs             ring-switch replay
    evaluator.rs          relation-matrix MLE point evaluator (eval_at_point decomposed)
    tensor_challenges.rs  affine-interval challenge factors
  slice_mle/              r-tail / setup-contribution MLE eval (verifier-only)
stages/
  stage1.rs stage2.rs stage3.rs   sumcheck stage verifiers
```

### Invariants

Preserve all of these; each names the mechanism that protects it.

1. **Byte-exact prover/verifier consistency.** The verifier must replay the
   prover's exact transcript (labels + absorb order) and produce identical
   accept/reject. Protected by: the `akita-pcs` scheme roundtrip tests +
   `profile/akita-recursion` end-to-end + `mixed_d_rejections`. (The differential
   harness protected this through Phases 0–3 and was retired at Phase-3 end — see
   Testing Strategy.)
2. **No-panic boundary.** Verifier-reachable code rejects malformed input with
   `AkitaError`/`SerializationError`, never a panic. Protected by:
   [`docs/verifier-contract.md`](../docs/verifier-contract.md) and
   `docs/verifier-panic-audit.md`. No refactor may introduce a new
   `unwrap`/`expect`/`unreachable!`/unchecked index on a verifier path.
3. **zkVM-guest buildability.** The crate must keep compiling and running inside
   the Jolt guest: **syscall-free hot path** (no time/thread/file/RNG),
   **no dependency on `akita-prover`/`akita-setup`/`akita-pcs`**, std+alloc
   allowed (guest target is `riscv64imac-zero-linux-musl`, *not* `no_std`).
   Protected by: a CI job that builds `akita-verifier` in isolation and for the
   guest target.
4. **Minimal public surface.** Downstream API stays `batched_verify` plus the
   two cross-check test exports that cannot move (see Non-Goals). Protected by:
   a surface test / review rule.
5. **Behavior preservation over cleverness.** No stage rewrite lands until the
   `akita-pcs` roundtrip + recursion e2e + `mixed_d_rejections` coverage is green
   for the shapes it touches. (Through Phase 3 this was the differential harness,
   now retired; a rewrite needing byte-exact transcript protection should
   reintroduce a golden-transcript net first.)

### Non-Goals

- **Not** rewriting the shared protocol math in `akita-types`/`akita-sumcheck`/
  `akita-challenges`/`akita-algebra`. The verifier *borrows* that math; it does
  not own it. Touch it only to extract already-duplicated glue (Phase 4).
- **Not** extracting `akita-verifier` to a separate repository now. Work stays in
  the monorepo. Extraction is a possible *future* reward once the boundary is
  clean and protocol churn slows (see Design → Alternatives).
- **Not** severing the `akita-config`/`akita-planner` coupling in this spec.
  That is Phase 5, deferred to a dedicated follow-up spec
  (`specs/akita-verifier-planner-severance.md`, to be written).
- **Not** pursuing true `no_std`/bare-metal. That is a workspace-wide effort
  (`rand_core[getrandom]` in `akita-field`/`akita-algebra`/`akita-types`) and is
  out of scope.

## Evaluation

### Acceptance Criteria

- [ ] `batched_verify` signature and behavior unchanged; all existing
      `akita-pcs` tests and `profile/akita-recursion` e2e pass throughout.
- [x] Differential harness existed, covered the proof-shape matrix
      ({fp32,fp64,fp128} × {dense,one-hot}), and reached full green parity
      (legacy vs. live: identical `Result` **and** identical recorded transcript
      bytes, 6/6). Retired at Phase-3 end (commit removing `akita-verifier-legacy`).
- [ ] CI builds `akita-verifier` standalone and for the guest target; a
      dependency-allowlist check fails on any new/heavy dependency.
- [~] Struct/glob sub-criteria met (no >10-field struct without a bundle/borrowed
      context; no production `use super::*`). The ~80-line function budget is
      **treated as a guide, not a hard rule**: a few single linear replay flows
      still exceed it by design (`ring_switch::eval_at_point` ~237,
      `suffix::prepare_fold_replay` ~231, `verify_recursive_fold` 102,
      `verify_fold` 92) — splitting them purely to hit 80 would add call
      indirection without a genuine logical boundary. Extractions landed only where
      they created a real boundary (e.g. `verify_recursive_fold` making the
      terminal/recursive dispatch symmetric).
- [ ] Extension-field generics are named `E`, base-field `F` (no
      `RelationMatrixEvaluator<F>`-style misnomers).
- [~] No `#[cfg(test)]` harness code remains in shipped `src/` — **won't-do
      (idiomatic)**, see Phase-3 item #3. A `#[cfg(test)]` internal characterization
      test is idiomatic Rust and never enters release/guest/dependency builds; a
      `tests/`/`test-support` move would expose the evaluator's ~28 private fields
      for no runtime benefit. The oversized 472-line `tests.rs` was instead split
      into three small `#[cfg(test)]` files (`fixtures`/`oracle`/`tests`) to address
      the file-size concern without any public-surface change.
- [~] Total verifier logic LOC decreases materially — **not met for the crate
      itself, by design.** `akita-verifier/src` is ~6,937 LOC vs 6,722 at the #312
      branch point: the de-glob (explicit imports), Phase-3 decomposition
      (function/struct boundaries + docs), the `verify_recursive_fold` extraction,
      and the `setup_contribution` test split all *added* lines in service of
      auditability. The one genuine reduction from this cycle — deleting the
      const-`D` `eval_position_at` (−29 LOC) — landed in `akita-types`, not the
      verifier crate. Item 3's expected src/ reduction did not materialize: the
      test module was kept in `src/` (idiomatic `#[cfg(test)]`) by owner decision,
      only reorganized. Conclusion: this criterion is superseded by the
      auditability goal for the verifier crate; the material repo-wide reduction
      came from retiring the temporary scaffolding (**−7,369 LOC**: 6,722 legacy
      crate + 647 differential test).
- [x] The temporary `akita-verifier-legacy` crate (the frozen legacy monolith)
      is removed at the end of Phase 3 (after full-matrix parity held).

### Testing Strategy

**Primary safety net — differential harness** (build this *first*, Phase 0):

- **Location:** `crates/akita-pcs/tests/verifier_differential.rs`. It lives in
  `akita-pcs` because that crate can reach `akita-prover`/`akita-setup` to
  *generate* proofs and both verifier entry points — placing it in
  `akita-verifier` would need an `akita-pcs` dev-dep, which is a cycle.
- **Dual entry points during migration:** freeze the pre-refactor monolith as a
  temporary sibling crate `akita-verifier-legacy` (a byte-for-byte copy at the
  Phase-0 branch point, `publish = false`, deleted at the end of Phase 3)
  exposing `akita_verifier_legacy::batched_verify`; the rewritten pipeline is
  `akita_verifier::batched_verify`. The harness runs both on identical inputs.
  (A sibling crate rather than a `legacy-verify` cargo feature: it keeps the
  frozen oracle fully isolated from the crate under refactor — no `#[path]` or
  glob gymnastics, no risk of the two implementations sharing a symbol — and the
  removal at Phase 3 is a clean crate delete.)
- **Byte-exact comparison:** drive both with a *recording* transcript (the crate
  already records via `absorb_and_record_bytes`) and assert the absorbed-byte
  logs are identical, not just that both return `Ok`. Transcript divergence is
  the failure mode a plain accept/reject check would miss.
- **Proof-shape matrix (accept cases):** root-direct (`ZeroFold`); 1-fold
  (`Terminal` root, no suffix); multi-fold with suffix; single-group root;
  multi-group root; terminal-witness variants (`FieldElements` vs
  `LogicalDigits`); × fields {`Fp32`, `Fp128`} × `BasisMode` × relevant
  `SetupContributionMode`.
- **Reject cases:** bit-flip proof bytes / commitment / claim value / opening
  point / schedule shape; assert legacy and new reject with the *same*
  `AkitaError` variant. This guards invariant #2 (no-panic) as well.
- **Retired** the `akita-verifier-legacy` crate + monolith after parity held
  across the full {fp32,fp64,fp128} matrix (6/6 green). Done at Phase-3 end.

**Secondary — characterization tests:** the core files (`fold`, `root`,
`suffix`, `ring_switch`) currently have *no* inline tests. Before Phase 3
restructuring, add a small checked-in golden corpus (serialized
proof/setup/claims + expected result + transcript-byte-log) for the highest-value
shapes, as a fast regression tripwire independent of the prover. Regenerating the
corpus is the intended signal that a *legitimate* protocol change occurred.

**Existing tests that must keep passing:** all `akita-pcs` scheme tests,
`crates/akita-pcs/tests/{ring_switch,stage1_roundtrip}.rs`,
`crates/akita-verifier/tests/mixed_d_rejections.rs`, and the recursion
host/artifact/guest builds.

### Performance

No verifier hot-path regression. The rewrite is structural; the shared math is
unchanged. Extract-method and de-duplication must not add per-round allocations
or defensive re-checks inside tight loops (invariant #2 rule 3). Verify against
the existing recursion guest cycle count / any verifier microbench before/after;
"no regression" is the bar. Proof size and security parameters are untouched.

## Design

### Architecture

The verifier is inherently a **linear replay** of the proof. The clean shape is a
pipeline of small pure stages, each threading the transcript, mirroring the proof
structure:

```
batched_verify
  1. validate_shape         proof/claims well-formed  (no-panic gate)
  2. bind_instance          schedule select + transcript instance descriptor
  3. replay_root            root fold          -> LevelState
  4. replay_suffix_levels   fold per level:    LevelState -> LevelState
       └─ replay_ring_switch    relation-matrix MLE eval at the challenge point
  5. check_terminal         terminal witness / direct openings
```

Rules that dissolve the current spaghetti:

- A small typed `LevelState` (opening point, opening, commitment, basis, w_len,
  carried setup-prefix) passed down the pipeline — **replaces the 24-field
  `PreparedFoldReplay` god struct** built at 3 sites.
- One function per proof step, ≲80 lines, one concern; math borrowed from
  `akita-types`, never re-implemented.
- Explicit imports; no `use super::*` (the `core.rs` 35-symbol glob wall goes
  away).
- One shared error-wrapping helper for the repeated
  `InvalidInput(format!("suffix verify level {i} failed: {err:?}"))` pattern.

**Concrete targets** (evidence gathered 2026-07-20; verify against source before
acting — code is under active churn):

*Mega-functions to break up:* `ring_switch::eval_at_point` (~204 lines, 5-deep
nesting); `core/fold::verify_fold` (~282 lines, incl. a 100-line trace-wire
selection); `suffix::prepare_fold_replay` (~212); `fold::into_claim` (~137);
`verify.rs::verify_folded_batched_proof` (~139); `root_fold::verify_root_inner`
(~182) / `verify_multi_group_root_inner` (~161).

*God structs / wide constructors:* `PreparedFoldReplay` (24 fields, 3 build
sites: `root_fold.rs:278`, `root_fold.rs:460`, `suffix.rs:323`);
`RelationMatrixGroupEvaluator` (15 fields, snapshots `LevelParams`);
`AkitaStage2Verifier` (14 fields, 13-arg ctor); `TraceWireAtRoleA` (3 variants
w/ ~7 shared fields).

*Intra-crate duplication:* `PreparedFoldReplay` construction (3×); per-group
opening-point loop (`root_fold` + `suffix` + `proof/direct`);
`prepare_relation_matrix_evaluator_inner` vs `_multi_group` near-dup;
opening/multiplier point-shape validation (3×); `ProductStageVerifier` vs
`PolynomialStageVerifier` near-identical.

*Thin wrappers to collapse:* `core/extension_opening_reduction.rs` (whole file, a
2-statement pass-through); `proof/mod.rs`; `ring_switch_verifier`/`_terminal` +
`RingSwitchVerifyCoreOutput::into_*`; `stage2::witness_eval`.

*Engine-in-a-file to split:* `verify.rs` mixes SIS root-direct recommitment
(lines 28–400) with top-level orchestration; `fold.rs` mixes orchestration +
trace-claim construction + EOR.

*Dead / test-only to remove or relocate:* `_setup_contribution_mode`
(`root_fold.rs:25`), `_terminal_final_w_len`, `_carried_setup_prefix`;
`validate_root_direct_recommitment_shape` (`#[cfg(test)]` wrapper); the entire
`slice_mle/setup_contribution` submodule is `#[cfg(test)]` inside `src/`.

**Component map & sizing** (baseline 2026-07-20, ~6,347 LOC; ~5,300 logic +
~1,000 test):

| Component | Files | ~LOC | Character |
|-----------|-------|------|-----------|
| Public surface / entry | `lib.rs`, `protocol/mod.rs` | 57 | clean; minor test-driven API inflation |
| Direct (zero-fold) opening | `proof/{mod,direct}.rs` | 197 | verifier-only; small |
| **Core fold reduction** | `protocol/core.rs` + `core/*` | **2,937 (46%)** | mega-functions, god struct, engine-in-a-file |
| **Ring-switch + MLE eval** | `protocol/ring_switch*` | **1,464 (23%)** | longest fn; near-dup builders; misnamed generics |
| Slice-MLE / setup-contrib | `protocol/slice_mle/**` | 550 | ~470 is `#[cfg(test)]` in `src/` |
| Sumcheck stages | `stages/*` | 1,142 | duplicate stage structs; 13-arg ctor |

**Dependency posture** (cleanest → most entangled). Narrow/foundational, leave
as-is: `serialization` (1 trait), `algebra` (stateless math), `transcript`
(1 trait + 2 helpers + shared labels), `sumcheck` (verifier-side traits only),
`field` (foundational trait tower), `challenges` (clean API, behavioral coupling
only). Deferred to Phase 5: `config` (re-runs the prover schedule planner via
`effective_batched_schedule` → `get_params_for_prove`, dragging in
`akita-planner`/`akita-schedules`; the `config → planner` edge is
*unconditional*) and `types` (~90 items, a de-facto shared-protocol-core
masquerading as a types crate).

**Prover duplication is small:** only ~3–6% (~150–300 LOC) is genuinely
extractable glue (per-group point loop, trace-table remap, point-shape
validation). Core math is already shared. This bounds Phase 4.

### Phased plan

Sequenced low-risk/high-clarity first; each phase ships independently and leaves
the crate green.

| Phase | Theme | Risk | Est. |
|-------|-------|------|------|
| 0 | Frozen contract + **differential harness** + conventions doc + CI guest/isolation build | none–low | 3–5 d |
| 1 | Dead code + wrappers + naming; move test-only code out of `src/`; shrink public surface | low | 2–4 d |
| 2 | Module re-org: split engine-in-a-file files; kill `use super::*` | low–med | 3–5 d |
| 3 | De-dup intra-crate + break up mega-functions into the pipeline; retire `legacy-verify` | med | 1.5–2.5 wk |
| 4 | Extract the ~3–4 shared prover/verifier glue helpers to their shared home | med (cross-crate) | 3–5 d |
| 5 | **Deferred** — type/config boundary + planner severance (own follow-up spec) | high (design) | 2–4 wk |

**Estimate:** Phases 0–3 (the minimal/auditable/rewritable core) ≈ **3.5–5
weeks**; +Phase 4 ≈ 3–5 days; Phase 5 planned separately.

Phase-0 conventions to adopt and enforce thereafter: size budgets (fn ≲80 lines,
struct ≲10 fields), `E`/`F` generic naming, no glob imports, one error-wrap
helper, test-support out of `src/`.

### Alternatives Considered

- **Extract to a separate `akita-verifier` repo now.** Rejected for now. The
  verifier is byte-locked to the prover through the transcript, `akita-types`
  shared math, and schedule derivation; git history shows these move in lockstep
  (frequent cross-cutting cutovers). A separate repo turns every such change into
  a cross-repo version-bump/pin/re-sync dance and invites stale-verifier
  transcript-drift bugs. The boundary discipline a repo split would force is
  instead obtained in-monorepo via CI (standalone + guest build), a dependency
  allowlist, and a public-surface test — the "extractable at any time"
  invariant. **Extract later** once (1) protocol cutovers are rare, (2) Phase 5
  boundaries are done, (3) the differential harness is stable; then extraction is
  a mechanical `git filter-repo` + path-dep→pinned-dep swap.
- **Ground-up blank-slate rewrite in a fresh crate/repo.** Rejected. A verifier
  is the most dangerous component to rewrite freehand (soundness-critical,
  byte-locked, thin test coverage). The strangler-fig + differential-harness
  approach gets a clean result *with* a correctness oracle at every step.
- **Golden-corpus-only (no live differential).** Kept as the *secondary* net,
  not primary: during active protocol churn a live legacy-vs-new differential
  auto-tracks legitimate changes, whereas a corpus needs manual regeneration.

## Documentation

- Update [`docs/verifier-contract.md`](../docs/verifier-contract.md) if any
  verifier-reachable boundary moves (it should not, in Phases 0–4).
- Fold durable content into `book/src/how/verification.md` once Phase 3 lands.
- Add a short "verifier module map + conventions" section (Phase 0 output) — link
  it from `AGENTS.md` so future contributors follow the pipeline shape.
- Write `specs/akita-verifier-planner-severance.md` for Phase 5 before starting
  it; update this spec's `Superseded-by`/cross-links as phases complete.
- Keep this spec's `Status` accurate (`proposed` → `active` → `implemented`).

## Execution

Task order:

1. **Phase 0 first, always.** No orchestration is touched before the differential
   harness is green on the legacy path and the CI guest/isolation build exists.
   The harness *is* the enabling deliverable.
2. Land Phases 1→2→3 as a series of small, individually-reviewable PRs, each
   keeping the harness green. Prefer many small diffs over big-bang.
3. Retire the `legacy-verify` feature and the monolith at the end of Phase 3;
   report before/after LOC.
4. Phase 4 coordinates with prover owners (touches `akita-prover`); confirm scope
   before starting.
5. Phase 5 is not started under this spec.

Risks to resolve first: thin inline coverage of core files (mitigated by the
harness + characterization corpus); ensuring the recording transcript captures
*every* absorb (a missed absorb hides a divergence).

## Phase 3 status & remaining work (as of 2026-07-21)

Phase 3 is largely complete and the crate is green throughout. Landed this cycle
(each its own differential-green commit): the `verify_fold` decomposition + the
`PreparedFoldReplay` god-struct split (24 → 6 cohesive fields), the per-group
opening-point loop de-dup (`prepare_group_opening_point`, sharing the per-group
target-length + point-variable extraction across the root and suffix loops), and
two thin-wrapper
collapses (`RingSwitchVerifyCoreOutput` + `into_intermediate`; the pointless
`stage2` `witness_eval` tracing span). The thin-wrapper collapse list is now
fully mined out (`proof/mod.rs` and `core/extension_opening_reduction.rs` were
removed upstream by #311).

Two type-boundary cleanups landed since (type-system tightening, behavior
preserved): `RelationMatrixEvaluator.flat_context` dropped its vestigial
`Option` — it is always constructed, so the seven `.ok_or(InvalidProof)` guards
became direct borrows — and `prepare_group_opening_point` now returns the
prepared point directly, rejecting a width mismatch with the canonical
`AkitaError::InvalidProof`. The `GroupOpeningPoint` enum that carried the
mismatch out so each caller could keep a distinct legacy error variant is gone;
with legacy retired, that distinction was incidental and made no production
contract.

Of the three items, all are now addressed: #1 and #2 landed (#2 scoped down);
#3 resolved by an explicit owner scoping decision (kept idiomatic, split for
readability). Details below.

1. **Close the fp32 differential cell, then retire legacy — DONE.** Added a
   self-contained fp32 one-hot fixture (`fp32_onehot_fixture`, option (b)):
   fp32's degree-4 extension means `E != F`, and every `OpeningFoldKernel` impl
   is base-field-only, so the `E`-valued opening is a plain Lagrange sum
   (`dense_lagrange_opening::<F, E>`) over the one-hot poly's densified evals
   (`evals[chunk·K + hot] = 1`, matching `OneHotPoly::direct_field_evals`),
   driven by `fp32::D128OneHot` (`schedules-fp32-d128-onehot`). The matrix
   reached 6/6 green across {fp32, fp64, fp128}, so full parity held and
   `akita-verifier-legacy` + `verifier_differential.rs` + the
   `verifier-differential` CI job + the differential wiring were deleted
   (−7,369 LOC repo-wide).

2. **Auditability breakups — DONE (scoped down).** Two changes landed, each
   re-verified against the `akita-pcs` scheme roundtrip suite +
   `profile/akita-recursion` e2e + `mixed_d_rejections` (the byte-exact
   differential net is gone; see the sequencing note):
   - **`eval_position_at` convergence + const-`D` deletion (genuine −29 LOC,
     mostly in `akita-types`).** Single-group `prepare_relation_matrix_evaluator_inner`
     was the last caller of the const-`D`
     `RingMultiplierOpeningPoint::eval_position_at`; the multi-group path already
     used the runtime-dimension `eval_position_at_dyn`. At the call site
     `alpha_pows.len() == D`, so the two are byte-identical (`as_ring_slice::<D>()`
     reinterprets the same flat coeffs via `#[repr(transparent)]`, and both eval
     helpers share one fold body). `inner` now uses `eval_position_at_dyn` and the
     const-`D` method is deleted from `akita-types`.
   - **`verify_recursive_fold` extraction (symmetric fold dispatch).** `verify_fold`
     already delegated the terminal payload arm to `verify_terminal_fold` but
     sprawled the ~85-line recursive arm inline, so the two payload variants sat at
     different altitudes. The recursive replay (bind next-level witness → inner
     ring-switch → stages 1/2/3) is now `verify_recursive_fold`, the sibling of
     `verify_terminal_fold`; `verify_fold` is a clean dispatcher: shared prefix
     (validate + derive stage-1 challenges + build relation instance) then a 2-arm
     match that delegates both payloads (the six recursive fields move into a
     `RecursiveFoldStages` bundle mirroring `PreparedFoldPayload::Recursive`). Pure
     extract-method — statement order, transcript absorbs, and dispatch are
     byte-identical.
   The originally-listed `ring_switch::eval_at_point` (~237) and
   `suffix::prepare_fold_replay` (~231) breakups were **deliberately dropped**
   (owner call): each is a single linear replay flow, and splitting it purely to
   hit the ~80-line budget would add call indirection (more jumping around) with
   no genuine logical boundary — a net-negative for auditability. The ~80-line
   budget is a guide, not a hard rule. `verify_recursive_fold` (unlike those) was
   worth extracting because it made the terminal/recursive dispatch *symmetric*, a
   real altitude fix rather than a line count.
   **Sequencing note:** these are soundness-sensitive, and they landed *after*
   legacy retirement removed the byte-exact differential net, so they relied on the
   `akita-pcs` e2e suite + `mixed_d_rejections` (which catch accept/reject
   regressions but not a transcript-order divergence that still verifies). Both
   changes are structure-only and preserve statement/absorb order, so the residual
   risk that class of net would have added is minimal here.

3. **Relocate test-only code out of shipped `src/` — RESOLVED (kept idiomatic
   `#[cfg(test)]`, split for readability).** As the item's own analysis showed,
   the tests hand-build `RelationMatrixEvaluator` / `FlatRelationContext` /
   `RelationMatrixGroupEvaluator` via full struct literals (~28 private fields)
   plus the `PreparedChallengeEvals::Flat` variant, so both a move to `tests/` and
   a `test-support` feature would expose the evaluator's *entire* internal
   representation — the feature only gates that exposure, it doesn't avoid it.
   **Owner decision:** a `#[cfg(test)]` unit test of crate internals is idiomatic
   Rust and is **never compiled into release / zkVM-guest / dependency builds**
   (the "shipped in `src/`" concern is source-tree footprint, not runtime), so the
   module stays `#[cfg(test)]` with no `test-support` machinery and no
   public-surface change. To address the file-size half of the concern, the
   472-line `tests.rs` was split by concern into three small `#[cfg(test)]` files
   under `slice_mle/setup_contribution/` — `fixtures.rs` (shape catalog + fixture
   builder + assertions), `oracle.rs` (the naive reference), `tests.rs` (the
   `#[test]` cases) — sharing visibility via `pub(super)`. This acceptance-criterion
   line ("no `#[cfg(test)]` harness in shipped `src/`") is therefore intentionally
   recorded as **won't-do (idiomatic)** rather than met.

## Known gaps & follow-ups (as of P0/P1 on #312)

Tracked so they are not silently lost. Each notes a fix path.

1. **Differential matrix: fp32 cell — CLOSED (harness now retired).** The
   harness covered fp128
   (dense Lagrange/Monomial, one-hot multi-fold) and fp64 dense (the `E != F`
   extension-field path), but not fp32. Root cause: `schedules-default` ships
   **no fp32 *dense* schedule** (only `fp32-d128-onehot`/`fp32-d256-onehot`), so
   a dense fp32 fixture cannot size its setup, and fp32's degree-4 extension
   means a one-hot fixture needs an extension-field one-hot opening (the shared
   `opening_from_poly` helper returns the base field only).
   *Fix (either):* (a) add a generated `schedules-fp32-*-full` table + feature
   and use a dense fp32 fixture; or (b) add a generic one-hot fixture driven by
   `fp32::D128OneHot`. **Closed via option (b)** (`fp32_onehot_fixture`; the
   harness is now retired) — the matrix reached 6/6 green across
   {fp32,fp64,fp128}. Correction to the earlier note: every
   `OpeningFoldKernel` impl is base-field-only, so the `E`-valued opening is *not*
   computed "via the fold kernel over the extension"; it is a plain
   `dense_lagrange_opening::<F, E>` over the one-hot poly's densified evals (the
   same way the fp64 dense fixture computes its `E`-opening). fp64 already
   exercises the `E != F` code path, so this is coverage breadth, not a
   correctness hole.

2. **Ring-switch verifier/terminal wrapper collapse — DONE.**
   The Phase-1 target (`ring_switch_verifier`/`_terminal` +
   `RingSwitchVerifyCoreOutput::into_*`) was mostly dissolved by #311 (terminal
   is now quotient-free/direct, so `ring_switch_verifier` takes the row layout
   directly and only `into_intermediate` remained). The residual
   (`RingSwitchVerifyCoreOutput` + `into_intermediate`) was collapsed once the
   M-eval hands-off constraint lifted: `ring_switch_verifier` now builds
   `RingSwitchVerifyOutput` directly (−22 LOC), preserving the basis-check /
   tau0-unwrap order. Remaining `ring_switch/{replay,evaluator}` file split is
   still open (see the `eval_at_point` breakup below).

3. **Pre-existing (not introduced here): `fold_protocol_epoch` test.** On #312,
   `crates/akita-pcs/tests/fold_protocol_epoch.rs` fails to compile under
   `--all-targets` **with the `logging-transcript` feature** — it reads
   `LevelParams::log_basis` (line ~161), a field renamed by #312. CI never hits
   it: the clippy job runs without `logging-transcript`, and the only job that
   built akita-pcs with `logging-transcript` (the differential job) compiled
   only `--test verifier_differential` — and that job is now removed with the
   harness, so *no* CI job compiles this test. *Fix:* update the stale field
   access (likely `log_basis_open`); belongs to whichever PR renamed the field.
   Untouched here (out of scope for the refactor).

4. **`use super::*` glob: `fold.rs` — now unblocked.** Phase 2 replaced the
   production `use super::*` wall with explicit imports in `verify.rs`,
   `root_fold.rs`, and `suffix.rs` (which also let `core.rs` shed the imports
   that existed only to feed those globs). `fold.rs`'s glob was originally held
   back because it drives stage-2 and relation evaluation and would have collided
   with the M-eval PR; with that PR cancelled (see the lifted constraint below),
   the `fold.rs` de-glob and the `core.rs` wall shed complete Phase 2.
   Test-module `#[cfg(test)] use super::*;` globs are left as idiomatic.

**Hands-off constraint — LIFTED (2026-07-20).** The originally-planned separate
PR reworking the verifier's **M (relation-matrix) evaluation logic** will not
land, so that surface — `stages/stage2`, `stages/stage3`, the setup-contribution
artifacts (`protocol/slice_mle/**`, `SetupContributionPlan`), and the
`ring_switch` evaluator (`eval_at_point`/`eval_flat_at_point`) — is now **in
scope** for this spec. This unblocks completing the `fold.rs` de-glob (Phase 2)
and the `eval_at_point` / stage-2 breakup (Phase 3). The frozen `batched_verify`
contract and the no-panic / zkVM-guest invariants still bind everywhere.

## References

- Existing: [`docs/verifier-contract.md`](../docs/verifier-contract.md),
  `docs/verifier-panic-audit.md`, [`docs/crate-graph.md`](../docs/crate-graph.md).
- Related specs: `core-protocol-naming-cleanup.md`,
  `protocol-core-eor-consolidation.md`, `akita-sumcheck-unification.md`,
  `shared-opening-claims-api.md`, `distributed-verifier-row-eval.md`.
- Consumers: `crates/akita-pcs/src/scheme/mod.rs:314`;
  `profile/akita-recursion/{host,guest,artifact}` (guest kernel
  `guest/src/lib.rs:105`; target notes in `profile/akita-recursion/README.md`).
- Entry point: `crates/akita-verifier/src/protocol/core/verify.rs:438`.
