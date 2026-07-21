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
   accept/reject. Protected by: the **differential harness** (Evaluation →
   Testing) and the existing `akita-pcs` scheme roundtrip tests +
   `profile/akita-recursion` end-to-end.
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
   differential harness is green for the shapes it touches.

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
- [ ] Differential harness exists, covers the proof-shape matrix (below), and is
      green: legacy vs. new produce identical `Result` **and** identical recorded
      transcript bytes on every fixture.
- [ ] CI builds `akita-verifier` standalone and for the guest target; a
      dependency-allowlist check fails on any new/heavy dependency.
- [ ] No function in `akita-verifier/src` exceeds ~80 lines; no struct exceeds
      ~10 fields without a builder/borrowed-context; no `use super::*` glob.
- [ ] Extension-field generics are named `E`, base-field `F` (no
      `RelationMatrixEvaluator<F>`-style misnomers).
- [ ] No `#[cfg(test)]` harness code remains in shipped `src/` (moved to `tests/`
      or a `test-support` feature).
- [ ] Total verifier logic LOC decreases materially (target: shrink the ~5,300
      logic LOC; report the before/after).
- [ ] The temporary `akita-verifier-legacy` crate (the frozen legacy monolith)
      is removed at the end of Phase 3.

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
- **Retire** the `akita-verifier-legacy` crate + monolith once parity holds
  across the matrix.

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
