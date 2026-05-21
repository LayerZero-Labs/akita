# Spec: Terminal-Fold Cutover (Soundness Fix + v-Rows Drop)

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-05-17                     |
| Status      | implemented                    |
| PR          | #88                            |

## Summary

The recursive-fold terminal level previously committed to the next-level
witness on the transcript and then shipped a separate `final_witness`
*outside* the transcript-binding loop with no consistency check, letting
an adversary substitute any `final_witness` after the challenges had
been squeezed. This cutover (a) closes that soundness gap by absorbing
the cleartext `final_witness` directly into the transcript at the
terminal fold, and (b) drops the redundant D-block of the `M`-matrix
(the `v = D · w_hat` rows) from the terminal fold entirely, removing
`v` from `TerminalLevelProof`, shrinking the relation sumcheck, and
shrinking the shipped recursive witness. The PR also hard-splits the
proof shape into separate `AkitaLevelProof` / `TerminalLevelProof`
types and refits the planner and generated schedule tables to the new
terminal-level cost model.

## Intent

### Goal

Replace the soundness-broken transmit-then-commit terminal protocol
with a transmit-and-bind terminal protocol that:

1. Absorbs the cleartext final witness (`PackedDigits`) into the
   Fiat-Shamir transcript at the same point that previously held
   `next_w_commitment`, so all ring-switch and stage-2 challenges bind
   to the actual witness the verifier later consumes.
2. Drops `next_w_commitment`, `next_w_eval`, and the entire stage-1
   sumcheck from the terminal level (no longer needed: the cleartext
   witness is structurally range-checked by `PackedDigits` packing and
   the next-witness commitment is the witness itself).
3. Drops the D-block of the per-row `r` quotients from the terminal
   level under a new `MRowLayout::Terminal` mode, omits `v` from
   `TerminalLevelProof`, and runs the relation sumcheck without any
   `v`-rows.
4. Keeps `AkitaLevelProof` as the intermediate-only proof and adds a
   hard-split, non-shared `TerminalLevelProof` sibling, lifting the
   same split into `AkitaProofStep::{Intermediate, Terminal}` and
   `AkitaBatchedRootProof::{Fold, Terminal}`.
5. Refits the planner (dynamic + generated tables + baseline) so the
   schedule's recorded `next_w_len` for the last fold matches the
   prover's actual terminal-layout cleartext witness length.

Key abstractions touched: `AkitaBatchedProof`, `AkitaBatchedRootProof`,
`AkitaProofStep`, `AkitaLevelProof`, `TerminalLevelProof`,
`MRowLayout`, `QuadraticEquation::{new_prover,
new_recursive_multipoint_prover}`, `ring_switch_build_w`,
`compute_r_split_eq`, `generate_y`,
`ring_switch_finalize{_after_absorb, _with_gamma{,_after_absorb}}`,
`prove_terminal_fold_level_from_quadratic`,
`prove_terminal_root_fold_{from_quadratic, with_params}`,
`prove_terminal_recursive_fold_with_params`,
`derive_stage1_challenges`, `relation_claim_from_rows_extension`,
`schedule_plan_from_generated_entry`,
`w_ring_element_count_with_counts_for_layout`,
`w_ring_element_count_with_vector_counts_for_layout_bits`,
`finalize_terminal_direct_witness_shape`, `terminal_level_bytes` in
the baseline planner.

### Invariants

- **Witness binding.** At every fold level (root + intermediate +
  terminal), the verifier-recomputed Fiat-Shamir challenges must
  depend on the exact witness the verifier later consumes. For
  intermediate levels this remains the SIS commitment; for the
  terminal level it is the cleartext `PackedDigits` final witness.
  Protected by `tests/single_poly_e2e.rs`,
  `crates/akita-pcs/tests/akita_e2e.rs::*round_trip`,
  `crates/akita-scheme/src/tests.rs::verify_*`, and the
  `crates/akita-pcs/tests/transcript_trace.rs` audit fixture.
- **Terminal sumcheck shape.** The terminal relation sumcheck runs in
  relation-only mode (`batching_coeff = 0`, dummy `r_stage1`,
  `s_claim = 0`) with the D-block omitted from `m_evals_x`, so its
  rounds equal `col_bits + ring_bits` of the terminal-layout `w` only.
  Protected by `batched_onehot_roundtrip_matches_public_shape_context`
  and the `single_poly_e2e` round-trip assertion.
- **Schedule/runtime witness length agreement.** For every fold level,
  `runtime w.len() == planner-recorded next_w_len`. The terminal fold
  uses `MRowLayout::Terminal` on both sides. Protected by the
  `scheduled root next-w length did not match runtime witness` runtime
  guard and the
  `adaptive_{bounded,onehot}_plan_matches_runtime_next_w_len` tests in
  `crates/akita-config/src/schedule_policy.rs`.
- **Proof type separation.** `AkitaLevelProof` is intermediate-only,
  with `TerminalLevelProof` as a sibling type. Intermediate steps
  deserialize as `AkitaProofStep::Intermediate(AkitaLevelProof)`,
  terminal as `AkitaProofStep::Terminal(TerminalLevelProof)`; root deserializes as
  `AkitaBatchedRootProof::Fold` (multi-fold) or
  `AkitaBatchedRootProof::Terminal` (single-fold). Protected by
  `AkitaBatchedProof` shape derivation/`shape()` and the serialize +
  deserialize round-trips in `batched_onehot_roundtrip_matches_public_shape_context`.
- **No backward compatibility.** No deprecated aliases, no shipping
  the old `next_w_commitment` / `next_w_eval` / `v` fields. All call
  sites updated in one pass.

### Non-Goals

- Re-pricing the stage-1 sumcheck for intermediate levels (unchanged).
- Changing the SIS instance, security parameters, or the
  decomposition basis selection logic.
- Changing the recursive ring-switch protocol below the terminal fold.
- Optimizing the search-time cost model beyond the v-rows-drop
  awareness already added; the cost model's terminal estimate now
  uses `MRowLayout::Terminal` sizing, but its broader structure
  (tight-zpre, eq-compression, GKR tree, etc.) is unchanged.
- ZK D-blinding redesign. The terminal layout zeros the D-blinding
  digit segment of `w` (it has nowhere to live without the D-block)
  but the rest of the ZK hiding protocol is unchanged.

## Evaluation

### Acceptance Criteria

- [x] `cargo test --all` passes (default features = `parallel`).
- [x] `cargo test --all --features zk` passes.
- [x] `cargo test --all --no-default-features --features planner`
      passes.
- [x] `cargo clippy --all -- -D warnings` clean under default, `zk`,
      and `no-default-features+planner` configurations.
- [x] `cargo fmt -q` produces no diff.
- [x] `cargo run -p akita-planner --release --bin akita-planner --
      --validate` reports all baselines match.
- [x] `gen_schedule_tables` regeneration produces no diff against the
      checked-in tables (planner choices are stable under the cost-
      model improvement).
- [x] `AkitaBatchedProof::shape()` derives a shape that round-trips
      `serialize_uncompressed` /`deserialize_uncompressed`.
- [x] Terminal `TerminalLevelProof` no longer carries `v`,
      `next_w_commitment`, `next_w_eval`, or `stage1_sumcheck`.
- [x] Adversarial probe: tampering with `final_witness` (e.g.
      `batched_onehot_same_point_rejects_tampered_root_stage1_s_claim`)
      causes verification failure, because the tampered witness no
      longer matches the transcript-bound challenges.

### Testing Strategy

Existing tests that must continue passing:

- `crates/akita-pcs/tests/akita_e2e.rs`: all e2e fixtures
  (single-poly, batched same-point, fp32/fp64 static dense, adaptive
  envelope, tamper rejection).
- `crates/akita-pcs/tests/single_poly_e2e.rs`: nv-15 single-poly
  round trip.
- `crates/akita-pcs/tests/ring_switch.rs`: ring-switch direct tests
  with explicit `MRowLayout` plumbing.
- `crates/akita-pcs/tests/zk.rs`: ZK D-blinding hiding tests, updated
  to pass `MRowLayout::Intermediate` to `new_prover` after the
  argument-order change.
- `crates/akita-scheme/src/tests.rs`: 21 unit tests, including
  `batched_onehot_roundtrip_matches_public_shape_context` (proof
  shape derivation now uses `MRowLayout::Terminal` for the terminal
  level's final witness sizing).
- `crates/akita-config/src/schedule_policy.rs`:
  `adaptive_{bounded,onehot}_plan_matches_runtime_next_w_len` (the
  test's runtime helper now selects `MRowLayout::Terminal` for the
  last fold).

New / updated assertions added by this PR:

- `schedule_plan_from_generated_entry` computes the terminal fold's
  `runtime_next_w_len` under `MRowLayout::Terminal` (passing batched
  `(num_points, num_t_vectors, num_w_vectors, num_z_vectors)` when
  the terminal fold is also the root) and uses the same value for
  the trailing `DirectStep`'s witness shape, so the verifier's
  `final_witness.shape() == terminal_direct.witness_shape` check
  passes for batched-root-terminal schedules.
- `finalize_terminal_direct_witness_shape` in
  `akita-planner/src/schedule_params.rs` overrides the placeholder
  Direct-step shape with the true terminal shape (D-block dropped)
  after suffix recursion returns. Called from both
  `derive_optimal_suffix_schedule` (recursive folds; recursive vector
  counts) and `find_optimal_schedule` (root; batched vector counts).

Feature combinations covered: default (`parallel`), `zk`,
`no-default-features+planner`. Release-mode profiling
(`AKITA_MODE=onehot AKITA_NUM_VARS=20 cargo run --release --example
profile`) is unchanged in interface; the example continues to print
per-marker breakdowns.

### Performance

Proof-size impact (planner output, run from
`cargo run -p akita-planner --release --bin akita-planner --
--validate`):

- Baselines unchanged: `onehot nv=32 -> 91,445 B`, `full128 nv=25 ->
  156,917 B`, `full128 nv=32 -> 163,501 B`. These baselines already
  capture the soundness fix's smaller terminal level (no stage-1, no
  `next_w_commitment`, no `next_w_eval`, no `v`).
- Optimized universal planner: `onehot nv=32 -> 71,488 B` (-21.8 % vs
  D=64 baseline); `full nv=32 -> 74,720 B` (-54.3 %); see the
  `headline` table in `cmd_results` for the full spread.

Per-level wire change (terminal level only):

- Removed: `next_w_commitment` (commitment to the next-level
  witness), `next_w_eval`, the entire stage-1 sumcheck, and
  `v_coeffs = d_key.row_len() * D` of `v` payload.
- Added: `final_witness` is no longer ghost-shipped after the
  challenges; it now occupies the same transcript slot previously
  held by `next_w_commitment` and is `PackedDigits`-encoded (smaller
  per-element than the SIS commitment).
- Witness size: the terminal fold's `w` is built with
  `MRowLayout::Terminal`, dropping `nd` from the per-row `r`
  quotients (and dropping the corresponding D-blinding digit segment
  under `zk`). Concretely for the failing test's schedule
  (D64Onehot, nv=15, batched 2): root terminal `w` shrinks from
  87,808 to 65,792 field elements (-25.1 %).

Schedule-table regeneration produces no diff: the cost model
improvement is real but too small to flip any planner choices in the
generated envelopes; runtime witness sizing now matches the
intermediate-vs-terminal split because the terminal fold is sized
under `MRowLayout::Terminal`.

The planner's universal search now evaluates *both* the intermediate
and terminal suffix for every candidate fold (search.rs:
`best_from` and root loop in `run_universal_planner`) and picks the
cheaper of the two; this is required so the search doesn't undercount
the terminal-layout savings when the suffix is a single `Direct` step.

## Design

### Architecture

Affected crates and the change at each boundary:

- **akita-types**
  - `MRowLayout` is a public `Intermediate | Terminal` enum threaded
    through every site that has to know whether the D-block is part of
    the `M`-matrix or has been dropped.
  - `LevelParams::m_row_count_for(MRowLayout)` returns the
    layout-conditional D-block row count (`n_d` for Intermediate, 0
    for Terminal).
  - `w_ring_element_count_with_counts_for_layout` and
    `w_ring_element_count_with_vector_counts_for_layout_bits` compute
    the witness-ring count under either layout; the legacy
    layout-free helpers internally call the new helpers with
    `MRowLayout::Intermediate` so external callers see no behavior
    change.
  - `AkitaLevelProof` is now intermediate-only (still carries `v`,
    stage-1 sumcheck, `next_w_commitment`, `next_w_eval`); its
    terminal-only fields are extracted into a new sibling
    `TerminalLevelProof` (no `v`, no stage-1, no next-commitment;
    ships `final_witness` and an optional
    `extension_opening_reduction`).
  - `AkitaProofStep::{Intermediate, Terminal}` mirrors the level
    split; `AkitaBatchedRootProof::{Fold, Terminal}` mirrors it for
    the root; `AkitaBatchedProofShape::{Fold {root_shape,
    step_shapes}, Terminal(TerminalLevelProofShape)}` mirrors it for
    shapes.
  - `schedule_plan_from_generated_entry` and surrounding helpers
    learn about `MRowLayout::Terminal` and propagate the
    terminal-sized `runtime_next_w_len` to the trailing `DirectStep`.

- **akita-prover / `protocol/quadratic_equation.rs`**
  - `QuadraticEquation` carries an `m_row_layout: MRowLayout` field.
  - `new_prover` and `new_recursive_multipoint_prover` accept the
    layout as their last argument and gate the D-block contribution
    in `generate_y` (`y` for terminal is `[consistency | y_rings |
    commitment_rows | A-zeros]`, no D-block).
  - Under `zk`, terminal-layout construction zeros the D-blinding
    digits (no D-block means no D-blinding column segment in `w`).

- **akita-prover / `protocol/ring_switch.rs`**
  - `compute_r_split_eq` accepts `MRowLayout` and uses `n_d_active`
    for `num_rows`, `d_start`, `b_start`, `a_start`, and gates the
    D-block iteration.
  - `ring_switch_build_w` reads `quad_eq.m_row_layout()` and passes
    it to `compute_r_split_eq`; under `zk` it overrides the
    D-blinding segment to a zero-length `FlatDigitBlocks` for
    Terminal.
  - `build_w_coeffs` likewise consumes the layout-conditional
    D-blinding segment and packs `w` accordingly.
  - `ring_switch_finalize_*` variants accept `MRowLayout` and pass
    it through to `compute_r_split_eq` and `m_evals_x` construction.

- **akita-prover / `protocol/flow.rs`**
  - `prove_terminal_fold_level_from_quadratic`,
    `prove_terminal_root_fold_from_quadratic`,
    `prove_terminal_root_fold_with_params`, and
    `prove_terminal_recursive_fold_with_params` are new (or renamed)
    entry points; each absorbs the cleartext `final_witness`
    (`PackedDigits`) into the transcript via `ABSORB_SUMCHECK_W`
    before sampling any ring-switch challenges, runs stage-2 in
    relation-only mode, and emits a `TerminalLevelProof`.
  - The relation claim is built with `&[]` for the `v` argument at
    terminal levels (no D-block rows to sum).
  - Dispatch (`batched_prove` root loop): 1-fold schedule routes the
    root through `prove_terminal_root_fold_with_params`; multi-fold
    routes the root through `prove_root_fold_with_params` and uses
    the suffix policy in `prove_recursive_suffix_with_policy` to
    request a `Terminal` request at the last suffix level.

- **akita-verifier / `protocol/levels.rs`**
  - Splits the verifier path into `verify_intermediate_level` and
    `verify_terminal_level` (or equivalent inline branches), keyed
    off the proof variant. Terminal verification absorbs the
    cleartext `final_witness` into the transcript, re-derives the
    ring-switch challenges, and runs stage-2 in relation-only mode.
  - `derive_stage1_challenges` accepts `MRowLayout` and skips the
    `ABSORB_PROVER_V` absorb when the layout is Terminal (no `v` is
    transmitted).

- **akita-config / `schedule_policy.rs`**
  - `assert_plan_matches_runtime_w_sizes` (test helper) selects
    `MRowLayout::Terminal` for the last fold of every plan when
    computing the runtime expected `next_w_len`.

- **akita-planner / `search.rs`**
  - `LevelComputation` adds `terminal_next_w_len`.
    `compute_level_witness` returns both intermediate and terminal
    sizings (the latter drops `nd` from `m_row` and, under `zk`, the
    `nd_blind` segment from the blinding column count).
  - `best_from` and the root loop in `run_universal_planner` evaluate
    both suffix candidates and pick the cheaper. When the chosen
    suffix is terminal, the planner stores `terminal_next_w_len` as
    the fold's `next_w_len` (not `next_w_len`).
  - `terminal_level_bytes` (already present) prices the terminal
    fold as just `y` + the relation-only stage-2 sumcheck.

- **akita-planner / `baseline.rs`**
  - `terminal_level_bytes` already drops `v` and stage-1; the
    BASELINE_CASES expected totals reflect that.

### Alternatives Considered

- **Hash-the-tail (absorb a hash of `final_witness` instead of the
  witness itself).** Rejected: the verifier must hash the full
  `final_witness` anyway to check the hash, so the cost is at best
  identical to absorbing the witness directly, and the protocol is
  cleaner without the extra hash hop.
- **Keep `v` and prove a separate consistency check for the
  shipped-cleartext witness.** Rejected: ad-hoc consistency checks
  layered on top of the transcript-bound flow are exactly the kind of
  patch this cutover is removing. Dropping the D-block from the
  M-matrix at the terminal level is the right structural fix.
- **Keep `AkitaLevelProof` as a single struct with optional `v`,
  optional `stage1`, optional `next_w_commitment`.** Rejected as
  user-visible mess. The hard split into intermediate-only
  `AkitaLevelProof` and a new `TerminalLevelProof` sibling makes the
  proof variants checkable at the type level and forces the verifier
  to branch on the variant rather than on `Option` fields.
- **Refit only the dynamic planner (skip the generated-tables
  pipeline).** Rejected: schedules served from the generated tables
  must agree with the runtime witness sizing, otherwise the prover's
  guard rail (`scheduled root next-w length did not match runtime
  witness`) fires. The schedule.rs `runtime_next_w_len` override and
  the `terminal_witness_field_len` propagation are required.

## Documentation

- `AGENTS.md` / `CLAUDE.md`: no changes required; the canonical
  profiling command is unchanged.
- `crates/akita-pcs/tests/transcript_trace.rs` (audit fixture): now
  shows the cleartext witness absorbed via `ABSORB_SUMCHECK_W` at the
  terminal fold and no `next_w_commitment` / `next_w_eval` /
  stage-1 sumcheck absorbs at the terminal level.
- `specs/SPEC_REVIEW.md`: this spec follows the existing review
  workflow; no template changes.
- Per-crate `README` / docs (none currently exist beyond `AGENTS.md`):
  no changes needed.

## Execution

Done in the following order, all on `quang/akita-fix-tail`:

1. Keep `AkitaLevelProof` as the intermediate-only proof and add a
   new `TerminalLevelProof` sibling carrying the terminal-only fields;
   lift the split into `AkitaProofStep` and
   `AkitaBatchedRootProof`; update serialization + shape derivation;
   update all match sites in prover/verifier/tests.
2. Move the cleartext `final_witness` absorb to the same transcript
   slot previously held by `next_w_commitment` in every terminal
   fold path (root + recursive + 1-fold root). Drop the terminal
   stage-1 sumcheck, `next_w_commitment`, and `next_w_eval`.
3. Introduce `MRowLayout` and thread it through
   `QuadraticEquation::{new_prover, new_recursive_multipoint_prover}`,
   `compute_r_split_eq`, `ring_switch_build_w`, `build_w_coeffs`,
   `ring_switch_finalize_*`, `derive_stage1_challenges`,
   `generate_y`, and the relation-claim builder.
4. Update the planner: extend `LevelParams::m_row_count_for`,
   `w_ring_element_count_with_counts_for_layout`, and the schedule-
   plan builders so the terminal fold's `runtime_next_w_len` and the
   trailing `DirectStep`'s witness shape both use
   `MRowLayout::Terminal`. Add `finalize_terminal_direct_witness_shape`
   for the dynamic planner and `next_w_len_override` for `to_fold_step`.
   For the root-as-terminal case, preserve the batched vector counts
   (`(num_points, num_t_vectors, num_w_vectors, num_z_vectors)`).
5. Refit the universal planner (`search.rs`) to evaluate both
   intermediate and terminal suffixes per candidate fold and pick the
   cheaper. Add `terminal_next_w_len` to `LevelComputation`.
6. Update the `akita-config` runtime / planner test helper to
   compute the expected `next_w_len` under `MRowLayout::Terminal`
   for the last fold.
7. Regenerate the generated schedule tables under both default and
   `zk` features; confirm no diffs (planner choices are stable).
8. Run `cargo fmt -q`, `cargo clippy --all -- -D warnings`,
   `cargo test --all`, `cargo test --all --features zk`,
   `cargo test --all --no-default-features --features planner`,
   and `cargo run -p akita-planner --release --bin akita-planner --
   --validate`.

Risks resolved during implementation:

- *Witness-shape mismatch between planner and runtime* (
  `scheduled root next-w length did not match runtime witness`
  guard). Fixed by both the generated-tables override and the
  dynamic planner's `finalize_terminal_direct_witness_shape`.
- *Runtime `planner/runtime next_w_len mismatch` panic in
  `akita-config` adaptive tests*. Fixed by updating the test helper
  to use `MRowLayout::Terminal` for the last fold.
- *Custom-config tests in `akita-scheme/src/tests.rs` building a
  manual single-fold schedule with intermediate-sized `next_w_len`*.
  Fixed by switching those custom `get_params_for_prove`
  implementations to `w_ring_element_count_with_counts_for_layout` +
  `MRowLayout::Terminal`.
- *ZK D-blinding column in `w`*. Fixed by zeroing the D-blinding
  segment in `ring_switch_build_w` when the layout is Terminal.
- *Stale-cache StrReplace failures on `crates/akita-pcs/tests/zk.rs`*.
  Worked around by editing via a one-shot Python script.

## References

- PR #88 on the `quang/akita-fix-tail` branch.
- Commit `cd82e81b` (initial soundness fix + planner refit).
- `crates/akita-pcs/tests/transcript_trace.rs`: transcript flow audit
  fixture.
- `crates/akita-planner/src/baseline.rs`: baseline planner with
  `terminal_level_bytes`.
- `crates/akita-planner/src/search.rs`: universal planner with
  intermediate-vs-terminal suffix selection.
- `cargo run -p akita-planner --release --bin akita-planner --
  --validate`: baseline validation.
- `AKITA_MODE=onehot AKITA_NUM_VARS=20 cargo run --release --example
  profile`: end-to-end profiling.
