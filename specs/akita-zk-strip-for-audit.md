# Spec: Strip ZK from Protocol Orchestration for Audit

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Quang Dao                      |
| Created       | 2026-06-25                     |
| Status        | in-progress (slices 0–3 landed) |
| PR            | [#218](https://github.com/LayerZero-Labs/akita/pull/218) |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  | roadmap/zero-knowledge.md      |

## Summary

The optional `zk` Cargo feature is **opt-in, prefix-only, and not actually
zero-knowledge today** — enabling it produces sound, verifying proofs, not ZK
proofs (the `sec:zk-joint-sigma` seam and suffix modulus switching are
unimplemented; the tail is discharged by a plain opening). See
[`book/src/roadmap/zero-knowledge.md`](../book/src/roadmap/zero-knowledge.md) and
[`specs/akita-zk-sumcheck-hiding-plain.md`](akita-zk-sumcheck-hiding-plain.md).

For the upcoming audit we want the transparent protocol to be the product under
review, with **zero ZK in the auditor's reading path** — the ~1,200
`#[cfg(feature = "zk")]` sites (826 positive + 295 `not(zk)` + Cargo defs)
fracture the hottest orchestration files (`wire.rs` 94, prover `fold.rs` 78,
`setup.rs` 64, `levels.rs` 52, verifier `fold.rs` 49). We do **not** want to
throw the ZK work away: it will be finished after the audit.

This spec removes all ZK from `main` into a preserved `zk-wip` branch + tag
("strip-to-branch"), keeps the one genuinely generic, pure-arithmetic helper
(LHL sizing math) always-on as `lhl_blinding`, and sequences the removal into
verifiable slices that **provably do not change transparent proof bytes,
transcript bytes, or verification behavior**.

### Implementation status (PR #218)

**Landed in [#218](https://github.com/LayerZero-Labs/akita/pull/218)** (mergeable
milestone; transparent path only):

| Slice | Scope | Status |
|-------|-------|--------|
| 0 | Preservation refs + golden tripwire | Done |
| 1 | `lhl_blinding` always-on | Done |
| 2 | ZK-only tests + `*_zk.rs` schedules | Done |
| 3 | Setup `zkB`/`zkD` matrices + drop `akita-r1cs` | Done |
| CI | Drop `all-features` / zk schedule-drift legs; transparent merge gate | Done (slice 5 partial) |

**Follow-up PR(s):**

| Slice | Scope | Status |
|-------|-------|--------|
| 4 | Core protocol orchestration (`fold`, masked sumcheck, `zk_hiding`, blinding hints, etc.) | Not started |
| 5 | Delete `zk` Cargo features; final CI grep gate | Partial (CI only) |

After #218 merges, `main` still carries ~650 `#[cfg(feature = "zk")]` sites in
prover/verifier/types/sumcheck (dead on the default build). `--features zk` does
not build. Auditors reviewing transparent code can use `main`; full "zero ZK in
tree" waits for slices 4–5.

Preservation: branch `zk-wip` and tag `zk-prefix-snapshot-2026-06` at `2c6b6b1f`.

## Intent

### Goal

Remove every `#[cfg(feature = "zk")]` / `#[cfg(not(feature = "zk"))]` site and
the `zk` Cargo feature itself from `main`, preserving the complete ZK state on a
`zk-wip` branch and an annotated tag, while promoting `akita-types/src/zk.rs`
(pure LHL capacity arithmetic) to an always-compiled `lhl_blinding` module.

Affected surfaces:

- **Proof types & wire** (`akita-types`): drop the cfg-gated `zk_hiding` field on
  `AkitaBatchedProof` ([`levels.rs:1117`](../crates/akita-types/src/proof/levels.rs)),
  the `sumcheck_proof` ↔ `sumcheck_proof_masked` / `next_w_eval` ↔
  `next_w_eval_masked` field-swaps in `levels.rs` / `wire.rs`, and the
  `zk_b_matrix` / `zk_d_matrix` fields + constructors + serde in
  [`setup.rs`](../crates/akita-types/src/proof/setup.rs).
- **Prover orchestration** (`akita-prover/src/protocol/core/{prove,fold,suffix,root_fold}.rs`,
  `core.rs`): delete `ZkHidingProverState`, `build_zk_hiding_context`, pad
  helpers, the `+ mask` arms, and the cfg-gated parameters threaded through the
  fold chain.
- **Verifier orchestration** (`akita-verifier/src/protocol/core/{verify,fold}.rs`,
  `core/zk.rs`, `slice_mle/zk_blinding.rs`): delete `verify_zk_hiding_commitment`,
  the `zk_hiding_cursor` / `ZkRelationAccumulator` threading, and `core/zk.rs`.
- **Commitment blinding** (`akita-prover/src/protocol/ring_switch/{commit,evals}.rs`,
  `masking.rs`, `zk_hiding_commit.rs`): delete B/D digit blinding and the hiding
  commitment.
- **Masked sumcheck** (`akita-sumcheck`): delete `SumcheckProofMasked`,
  `EqFactoredSumcheckProofMasked`, and the `prove_zk` / `verify_zk` drivers.
- **Deferred R1CS** (`akita-r1cs`): remove the crate from the workspace (zk-only;
  preserved on branch).
- **Schedules** (`akita-schedules`): delete the `*_zk.rs` generated tables and
  reconcile the `*_table()` selectors.
- **Cargo**: delete the `zk = [...]` feature from every crate
  ([13 crates](#feature-cascade-to-delete)).
- **Keep always-on**: `akita-types/src/zk.rs` → `lhl_blinding.rs` (pure math).

### Invariants

| # | Invariant | Why it holds | Protected by |
|---|-----------|--------------|--------------|
| I1 | Transparent (default-build) proof bytes are unchanged. | The default build never compiled any `#[cfg(feature="zk")]` line; every paired site keeps the existing `not(zk)` arm verbatim. | **New** golden-byte test (Slice 0); existing serde roundtrips `akita_e2e.rs` |
| I2 | Transparent transcript bytes/labels are unchanged. | The only zk absorb (`ABSORB_ZK_HIDING_COMMITMENT`) is itself zk-gated; the masked next-w-eval feeds the *same* single `append_serde(ABSORB_STAGE2_NEXT_W_EVAL, …)` whose value on the non-zk path is the genuine unmasked `w_eval` ([`fold.rs:873-877`](../crates/akita-prover/src/protocol/core/fold.rs)). Labels never enter the sponge (`labels.rs`, `sponge.rs` ignores the label arg). | `transcript_is_deterministic_for_identical_schedule`; sponge `labels_do_not_enter_production_sponge` |
| I3 | The masked sumcheck driver's extra absorb (`ABSORB_SUMCHECK_CLAIM` in `prove_zk`/`verify_zk`) must vanish with the whole driver — never be merged into the transparent driver. | The transparent `prove`/`verify` driver does not emit it; merging would corrupt the transparent transcript. | I2 transcript test; code review |
| I4 | Setup serialization for the transparent path is unchanged. | `zk_b_matrix`/`zk_d_matrix` are serialized strictly *after* shared fields, only under cfg; no `not(zk)` padding counterpart. | Setup serialize→deserialize roundtrip (`setup.rs` tests) |
| I5 | The non-zk path is the live production path (no orphaned stubs). | Stub sweep found zero `todo!`/`unimplemented!` in any `not(zk)` branch. | `cargo check` default build |
| I6 | `--features zk` is allowed to break partway through the strip (it is being removed); only the default build must pass after every slice. | The feature is the deletion target. | Slice gates below |

### Non-Goals

- **No `akita-zk-prefix` plugin crate, no dependency-inversion hooks.** Rejected
  (see [Alternatives](#alternatives-considered)): infeasible without first
  inverting three core abstractions, creates a Cargo cycle, and ossifies a
  boundary the roadmap says finished-ZK will discard.
- **No in-tree `protocol/zk/` submodule relocation.** It leaves cfg-gated
  parameters/fields/branches visible in `fold.rs`/`wire.rs`/`levels.rs` — does not
  meet "zero ZK in the auditor's path."
- **No change to transparent proof format, transcript, or planner output.**
- **No attempt to keep `--features zk` building on `main`** after the strip.
- **No fix of the planner↔prover `zk_hiding_witness_len` drift** — it is
  confirmed harmless (conservative `≥` headroom) and is deleted with both
  functions.
- **No completion of real ZK** (seam, suffix modulus switching, LNP22) — that is
  post-audit, off the `zk-wip` branch.

## Evaluation

### Acceptance Criteria

- [ ] `rg -n 'feature *= *"zk"' crates/ -g '*.rs'` returns **0** matches on `main`.
- [ ] No crate's `Cargo.toml` defines a `zk` feature; `akita-r1cs` is removed from
      `[workspace] members` and from all `optional`/`dep:` references.
- [x] `cargo check --workspace --no-default-features --features parallel,disk-persistence`
      succeeds (transparent merge gate; #218).
- [ ] `cargo clippy --all --all-targets -- -D warnings` and the `--no-default-features` variant both pass without any `zk` feature in existence.
- [x] The Slice-0 golden-byte digests (transparent folded proofs for fixed
      instances + fixed seeds) are **byte-identical** across slices 0–3:
      `fp128::D64Full` nv=15 → `7d860d0b…cfb0a` (90,007 bytes);
      `fp128::D64OneHot` nv=20 → `d0df7b19…c88c` (86,324 bytes).
- [x] `lhl_blinding` compiles in the default build, its unit tests pass, and it is
      no longer behind any `cfg`.
- [x] A `zk-wip` branch and an annotated tag (`zk-prefix-snapshot-2026-06`)
      exist at the pre-strip commit.
- [x] CI: the `test-all-features` / zk schedule-drift legs are removed; the
      `test` leg is the merge gate (#218).
- [ ] No `todo!`/`unimplemented!`/dead `let _ = (zk…)` shims remain from the strip.

### Testing Strategy

**Authoritative gate after every slice** (mirrors CI `test` job):

```bash
cargo nextest run --no-default-features --features parallel,disk-persistence
cargo check --workspace --no-default-features --features parallel,disk-persistence
cargo clippy --all --all-targets --no-default-features -- -D warnings
```

**Slice-0 tripwire** (added before any deletion): `crates/akita-pcs/tests/transparent_proof_golden.rs`
pins SHA-256 of serialized transparent proofs for two **folded** (not root-direct)
instances: `fp128::D64Full` nv=15 and `fp128::D64OneHot` nv=20. Pair with
transcript-determinism assertions in the same file.

**`--features zk` leg:** was run on slices 0–2 only. After slice 3 (`akita-r1cs`
removed) and slice 2 (zk schedules deleted), `cargo … --all-features` is expected
to fail until slice 4–5 finish. CI no longer runs this leg (#218).

**Schedule drift** (critical for Slice 2):

```bash
cargo test -p akita-config generated_schedule_tables_match_find_schedule
cargo test -p akita-config --no-default-features --test schedule_catalog_feature_off
```

**Tests deleted** (zk-only): `crates/akita-pcs/tests/zk.rs` (9 fns),
`crates/akita-pcs/tests/fold_linf_zk.rs` (3 fns), the two `zk_*` fns in
`single_poly_tensor_e2e.rs`, and the ~24 inline `#[cfg(feature="zk")]` test
modules. **Tests kept** (transparent): the 7 core e2e files (`akita_e2e.rs`,
`batched_aggregated_e2e.rs`, `heterogeneous_prove_e2e.rs`,
`recursive_setup_e2e.rs`, `single_poly_e2e.rs`, `tiered_e2e.rs`) and ~44
unconditional test files.

### Performance

No prover/verifier performance change on the transparent path (the stripped code
never executed in the default build). Expected secondary wins: faster
compilation and a smaller workspace (one fewer crate, ~1,200 fewer cfg sites).
Benchmarks are all transparent (`crates/akita-pcs/benches/*`) and unaffected; no
zk-specific bench exists.

## Design

### Architecture

ZK enters the codebase as a **compile-time bifurcation woven through the proof
type system and wire format**, not a localized plugin seam. The dependency
arrows already point inward (prover/verifier/setup → types; nothing depends on a
"zk crate" except the leaf `akita-r1cs`). The clean removal therefore deletes the
gated arms in place and keeps the `not(zk)` arms the default build already used.

Four entanglement classes, in ascending difficulty:

1. **`zk_hiding` envelope on `AkitaBatchedProof`** — *easy.* Already a leading,
   fully cfg-gated wire envelope with a zk-only transcript absorb. Delete the
   field, its 4 serde arms (`wire.rs`), and the prove/verify blocks
   ([`prove.rs:337-350`](../crates/akita-prover/src/protocol/core/prove.rs),
   [`verify.rs:632-641`](../crates/akita-verifier/src/protocol/core/verify.rs)).

2. **B/D commitment blinding** — *easy–moderate.* Additive-only and fully gated:
   `commit_w` adds `zk_b_digit_rows` to `u` only under cfg; the transparent build
   always committed the unblinded value. Delete the blinding blocks in
   `ring_switch/commit.rs`, the zk segments in `ring_switch/evals.rs`,
   `zk_hiding_commit.rs`, `masking.rs`, and the verifier recompute. **Cannot become
   an always-on hiding helper** — it changes committed bytes and needs the zk-only
   setup matrices + proof fields to discharge.

3. **Field-swapping in `levels.rs` / `wire.rs`** — *moderate, mechanical.* The
   masked types (`SumcheckProofMasked`, `EqFactoredSumcheckProofMasked`) are
   byte-identical newtypes over the transparent ones, so the wire bytes and shapes
   are identical across builds. Delete the `_masked` fields/arms (~24 doubled serde
   arms + cfg constructor params) and keep the transparent `sumcheck_proof` /
   `next_w_eval` fields unconditionally. **No transparent byte change.**

4. **Masked sumcheck through fold replay** — *the hard one.* Pads are sliced by a
   cursor from a single committed `hiding_witness` in `ZkHidingProverState`,
   threaded forward through the entire fold chain; the verifier mirrors the exact
   cursor offset arithmetic. This is real coupling: ~10 function signatures across
   prover + verifier carry cfg-gated positional params, plus the `+ mask` on
   `next_w_eval`. The drivers are already cleanly swapped via `cfg_if!`
   (`prove` vs `prove_zk`, `verify` vs `verify_zk`), so deletion = remove the zk
   driver wholesale (taking its extra `ABSORB_SUMCHECK_CLAIM` with it — invariant
   I3) and strip the gated params from the shared signatures.

#### What stays vs goes

| Keep on `main` (always-on) | Strip to `zk-wip` branch |
|---|---|
| `akita-types/src/zk.rs` → `lhl_blinding.rs` (pure LHL capacity math) | `ZkHidingProverState`, `build_zk_hiding_context`, pad helpers (`core.rs`) |
| (its self-contained unit tests) | `zk_hiding` field + `ZkHidingProof` (`levels.rs`/`containers.rs`) |
| | `sumcheck_proof_masked` / `next_w_eval_masked` fields (`levels.rs`/`wire.rs`) |
| | `zk_b_matrix`/`zk_d_matrix` + `derive_zk_*`/`validate_*zk*` (`setup.rs`) |
| | B/D blinding (`ring_switch/{commit,evals}.rs`, `masking.rs`, `zk_hiding_commit.rs`) |
| | `prove_zk`/`verify_zk` + `SumcheckProofMasked` (`akita-sumcheck`) |
| | `akita-r1cs` crate (deferred R1CS, zk-only) |
| | verifier `core/zk.rs`, `slice_mle/zk_blinding.rs` |
| | `*_zk.rs` generated schedule tables (`akita-schedules`) |
| | `zk = [...]` features in all 13 Cargo.toml |

**On keeping `lhl_blinding` always-on:** after the strip its consumers
(`schedule.rs`, `proof_size.rs`, `ring_relation.rs`, `proof_optimized.rs`,
`masking.rs`, …) are all deleted, so it has **no live caller on `main`**. That is
the intended "generic infra, standalone / cut off": it is pure, tested capacity
arithmetic that documents the hiding-mask sizing discipline and gives the
post-audit ZK work a stable anchor. Add a module doc note: *"LHL hiding-mask
capacity math, reserved for the optional hiding-commitment layer; not used by the
transparent protocol."* See [Open Questions](#open-questions) for the
alternative of sending it to the branch too.

#### Feature cascade to delete

Current `zk = [...]` graph (all to be removed):

```
akita-pcs/zk      -> akita-config/zk, akita-prover/zk, akita-setup/zk, akita-types/zk, akita-verifier/zk
akita-config/zk   -> akita-planner/zk, akita-types/zk, akita-schedules?/zk
akita-setup/zk    -> akita-config/zk, akita-prover/zk, akita-types/zk
akita-prover/zk   -> akita-config/zk, akita-sumcheck/zk, akita-types/zk, rand_core/getrandom
akita-verifier/zk -> dep:akita-r1cs, akita-r1cs/zk, akita-config/zk, akita-sumcheck/zk, akita-types/zk
akita-sumcheck/zk -> dep:akita-r1cs, akita-algebra/zk, akita-r1cs/zk, akita-transcript/zk
akita-types/zk    -> akita-sumcheck/zk
akita-planner/zk  -> akita-types/zk
akita-challenges/zk -> akita-transcript/zk
akita-schedules/zk  -> akita-planner/zk
akita-algebra/zk, akita-transcript/zk, akita-r1cs/zk -> []  (leaf stubs)
```

`zk` is in **no** crate's `default` features (verified: `akita-types` `default = []`,
`akita-pcs` `default = ["parallel","schedules-default"]`, etc.), so the default
build is already the transparent path.

### Execution — sequenced slices

Each slice is independently committable and gated by the transparent test run +
the Slice-0 golden digest. Slices 1–3 landed in [#218](https://github.com/LayerZero-Labs/akita/pull/218).
Slice 4 depends on 1–3; slice 5 finishes feature removal and CI grep gates.

**Slice 0 — Preserve + tripwire.** ✅ *Landed #218*
- `git branch zk-wip` at current `main`; `git tag -a zk-prefix-snapshot-2026-06`.
- Add the transparent golden-byte test + transcript-determinism test on `main`;
  commit the pinned digest constant. This is the tripwire for I1/I2.
- *Verify:* default test run green; record the digest.

**Slice 1 — Promote LHL math to always-on (`lhl_blinding`).** ✅ *Landed #218*
- `git mv crates/akita-types/src/zk.rs crates/akita-types/src/lhl_blinding.rs`.
- In `lib.rs`: replace `#[cfg(feature = "zk")] pub mod zk;` with un-gated
  `pub mod lhl_blinding;` and add the doc note.
- Repoint existing consumers from `crate::zk::…` to `crate::lhl_blinding::…`
  (they stay zk-gated for now — an always-on module is visible to gated code).
- *Verify:* default + `--features zk` both green (consumers still exist this slice).

**Slice 2 — Delete zk-only tests + reconcile generated schedules.** ✅ *Landed #218*
- Delete `crates/akita-pcs/tests/{zk.rs,fold_linf_zk.rs}`, the `zk_*` fns in
  `single_poly_tensor_e2e.rs`, the ~24 inline `#[cfg(feature="zk")]` test modules.
- Delete the `*_zk.rs` files under `crates/akita-schedules/src/generated/` and
  reconcile each `*_table()` in `generated/mod.rs` (drop the `#[cfg(zk)]` arm,
  un-gate the `not(zk)` arm, drop `pub mod *_zk;`).
- *Verify:* default green; `--features zk` may now fail to *find* deleted tables —
  acceptable; schedule-drift test green on the transparent leg.

**Slice 3 — Tear down setup blinding (`zk_b`/`zk_d`) + remove `akita-r1cs`.** ✅ *Landed #218*
- Delete `zk_b_matrix`/`zk_d_matrix` fields, `derive_zk_*`, `validate_*zk*`,
  `ZK_*_LABEL`, `max_zk_b_len`/`max_zk_d_len`, and their serde/`Valid` arms in
  `proof/setup.rs`; reconcile the two non-test construction sites
  (`api/setup.rs`, `akita-setup/src/lib.rs`). `AkitaCommitmentHint` blinding-digit
  arity cleanup deferred to slice 4 (still cfg-gated on the default build).
- Remove `akita-r1cs` from `[workspace] members` and from `dep:`/feature lines in
  `akita-verifier` and `akita-sumcheck`.
- *Verify:* default green + setup serde roundtrip; golden digest unchanged.

**Slice 4 — Tear down core orchestration (the heart).** ⏳ *Follow-up PR*
- Prover: delete `ZkHidingProverState`, `build_zk_hiding_context`, pad helpers
  (`core.rs`); the prove.rs construction/writeback; B/D blinding in
  `ring_switch/{commit,evals}.rs`; `masking.rs`; `zk_hiding_commit.rs`.
- Verifier: delete `verify_zk_hiding_commitment` (`core/zk.rs`), the
  `zk_hiding_cursor`/`ZkRelationAccumulator` threading, `slice_mle/zk_blinding.rs`.
- Types/sumcheck: drop the `_masked` field-swaps (`levels.rs`/`wire.rs`),
  `zk_hiding` field, `ZkHidingProof`, `SumcheckProofMasked`/`EqFactoredSumcheckProofMasked`,
  and the `prove_zk`/`verify_zk` drivers.
- **Reconcile the fold signature chain together** (`prepare_root`/`prove_root`/
  `prove_fold`/`prove_terminal_root_fold_with_params`/`bind_next_witness_for_ring_switch`):
  drop the `zk_hiding` param, un-gate surviving `not(zk)` params, collapse the
  `+ mask` arms to plain, and the `terminal_direct_witness_shape`
  PackedDigits→SegmentTyped selector.
- *Verify:* this is where I1/I2 are most at risk — gate on the golden digest +
  full transparent e2e. (Run `--features zk` once *before* this slice to confirm
  the pre-strip feature still worked; do not expect it after.)

**Slice 5 — Delete the `zk` feature + CI cleanup.** ⏳ *Partial (#218: CI only)*
- Remove the `test-all-features` job and the `zk` schedule-drift line from
  `.github/workflows/ci.yml`; keep `test` as the merge gate. ✅ *Done #218*
- Remove every `zk = [...]` line from the 13 Cargo.toml files and the leaf stubs.
- Grep gate: `rg 'feature *= *"zk"' crates/` → 0.
- *Verify:* full default CI green; `cargo check --workspace` errors out on any
  stray `--features zk` (proves the feature is gone).

#### Survives-complete (un-gate only, no completion work)

The tiered/recursive setup engine and the SegmentTyped terminal witness stack are
gated on **runtime** ring-dimension/`tier_split`, *orthogonal* to `zk`; the
`fp32`/`fp128` non-zk arms are the live path. `PackedDigits` becomes dead
production type after Slice 4 (still used by deleted tests) — leave it and flag a
follow-up. `schedule_terminal_direct_witness_shape` is already unconditional.

### Alternatives Considered

1. **`akita-zk-prefix` plugin crate with hook/trait dependency-inversion**
   (the prior council's Phase 2). **Rejected.** ZK is baked into the proof wire
   format (`AkitaBatchedProof.zk_hiding`) and the compute traits (`zk_b_digit_rows`
   are cfg-gated *methods* on `DigitRowsComputeBackend`, which a plugin cannot
   extend from outside Rust). A plugin needs `OperationCtx`/`RootCommitKernel`/proof
   types, so it must depend on `akita-prover`/`akita-types`; core calling into it
   creates a Cargo **cycle**. Making it work first requires inverting three core
   abstractions (backend kernel methods → standalone trait, setup matrices →
   optional extension, `zk_hiding` field → `Option`). Worse, the roadmap
   (`zero-knowledge.md:14-21`) says finished-ZK *replaces* the plain-opening prefix
   with the `sec:zk-joint-sigma` seam + suffix modulus switching + LNP22 — the very
   structures the hook would freeze. Net: high cost, ossifies the wrong boundary,
   discarded post-seam.

2. **In-tree `protocol/zk/` submodule relocation.** **Rejected as insufficient.**
   Moving the *bodies* (`ZkHidingProverState`, `build_zk_hiding_context`, masking)
   into a skippable `protocol/zk/` declutters `core.rs`, but the cfg-gated
   *parameters*, *struct fields*, and `+ mask` *branches* live inside the shared
   `prove_fold`/`PreparedFold` and **cannot move** — the auditor still reads ZK in
   `fold.rs`/`wire.rs`/`levels.rs`. A genuine partial win, but it does not meet
   "zero ZK in the auditor's path." Kept as the fallback if continuous
   `--features zk` buildability is later judged more important than a fully clean
   read.

3. **Snapshot / cfg-strip tarball.** Leave `main` as-is; a CI/script pass strips
   `#[cfg(feature="zk")]` to hand the auditor a generated ZK-free tree.
   **Rejected as primary** because the audited artifact would be a generated
   derivative that can drift from the repo of record; teams generally prefer the
   audited tree to be the real `main`. Viable as a stopgap if the strip cannot land
   before the audit window.

4. **Keep B/D blinding as an always-on "hiding commitment."** **Rejected.** The
   blinding changes committed bytes and is inert without the zk-only setup matrices
   + proof fields to discharge it; it is not a standalone hiding primitive. Only the
   LHL *sizing math* is genuinely generic.

## Documentation

- Update [`book/src/roadmap/zero-knowledge.md`](../book/src/roadmap/zero-knowledge.md):
  note that the ZK prefix is now off `main` on `zk-wip` (tag
  `zk-prefix-snapshot-2026-06`), to be reintroduced atop the finished seam.
- Update [`book/src/usage/feature-flags.md`](../book/src/usage/feature-flags.md):
  remove the `zk` feature row.
- Set `Status: superseded` (or add a banner) on the prefix-ZK design specs
  [`akita-zk-sumcheck-hiding-plain.md`](akita-zk-sumcheck-hiding-plain.md),
  [`akita-zk-commitment-hiding.md`](akita-zk-commitment-hiding.md),
  [`akita-zk-v-hiding.md`](akita-zk-v-hiding.md): mark them out of audit scope,
  retained as the design record for the `zk-wip` work.
- Add an audit-contract note: the transparent path is the product under review;
  `zk` is not built, `akita-r1cs` is absent, and `ProtocolFeatureSet.zk == false`.
- Update `AGENTS.md` / `Cargo.toml` comments referencing the `zk` feature.

## References

- Prior council (Cursor, 2026-06-24): the 3-phase plan this spec sharpens — Phase 1
  (lhl_blinding, kept), Phase 2 (`akita-zk-prefix` crate, rejected here).
- Roadmap: [`book/src/roadmap/zero-knowledge.md`](../book/src/roadmap/zero-knowledge.md)
  — seam + suffix modulus switching are the remaining real-ZK steps.
- Design record: [`specs/akita-zk-sumcheck-hiding-plain.md`](akita-zk-sumcheck-hiding-plain.md),
  [`specs/akita-zk-commitment-hiding.md`](akita-zk-commitment-hiding.md),
  [`specs/akita-zk-v-hiding.md`](akita-zk-v-hiding.md).
- Key anchors: `crates/akita-types/src/proof/{levels.rs:1117,wire.rs,setup.rs}`,
  `crates/akita-prover/src/protocol/core/{fold.rs:99,fold.rs:830,fold.rs:873,prove.rs:337}`,
  `crates/akita-verifier/src/protocol/core/{verify.rs:632,fold.rs,zk.rs}`,
  `crates/akita-sumcheck/src/{types.rs,drivers/standard.rs}`,
  `crates/akita-types/src/zk.rs` (→ `lhl_blinding.rs`).
- Verify commands: `cargo nextest run --no-default-features --features parallel,disk-persistence`;
  CI `test` job (`.github/workflows/ci.yml`).

## Open Questions

1. **`lhl_blinding` on `main` vs branch.** This spec keeps the pure LHL math
   always-on (honoring "keep generic infra standalone"). The alternative is to send
   `zk.rs` to `zk-wip` too, leaving `main` with *zero* zk-adjacent code — cleanest
   auditor view, but contradicts the stated "keep generic infra standalone."
   Recommended default: keep `lhl_blinding` with the doc note; revisit if the
   auditor flags unreferenced code.
2. **`PackedDigits` dead type** after Slice 4 — delete now (extra reconciliation)
   or leave as a flagged follow-up? Recommended: follow-up, to keep Slice 4 focused.
3. **`zk-wip` maintenance.** Since finished-ZK is largely a rewrite, treat the
   branch as a *frozen reference* (do not continuously rebase onto `main`). Confirm
   this is acceptable vs. periodic rebase.
