# Spec: Commitment Compression Cutover

Status: draft planning spec.

This spec maps the PR stack for removing the current tiered-commitment
implementation and replacing it with commitment compression.

The PR stack should have three semantic PRs:

1. Delete tiered commitment as implemented today.
2. Introduce commitment compression: compress `v` at every fold and compress
   next-level `u` at every non-penultimate fold.
3. Finish the terminal-tail cutover: remove the final recursive `u` and bind
   terminal `t` directly, with a terminal M-row layout that drops both D and
   COMMIT/B rows.

The design goal is proof-size reduction. The current tiered implementation was
primarily a setup-size / setup-scan optimization for the `B` matrix. It changes
the protocol shape, adds a `B' -> F` relation, and still sends a ring-shaped
commitment. That is too ad hoc to compose with a cleaner compression layer.

## Motivation

Today the public commitments sent over the wire are ring-native:

- `u`, the commitment to the next recursive witness, comes from the `B` side.
- `v`, the opening commitment, comes from the `D` side.

For `fp128, D = 64`, a single ring row is already about 1024 bytes. Even if a
commitment only needs a small number of scalar output coordinates for security,
encoding it as `RingCommitment<F, D>` gives a one-row floor.

The compression idea is to replace a large public ring-shaped commitment with a
small scalar image:

```text
raw commitment y = M x
digits c = Decompose(y)
compressed commitment z = C c
```

The verifier sees `z`, not `y`. The proof relation must enforce that the hidden
digits `c` really decompose the raw commitment `y`, and that `z = C c`.

The local small-instance sweep in
`LINF-SIS-SMALL-INSTANCE-SWEEP-NEVER-COMMIT.md` supports the basic intuition:
for tiny bounded witnesses over `fp128`, scalar unstructured SIS often reaches
the 128/138-bit range with a small output dimension, and recursive compression
tends to contract quickly before stabilizing. The note is not a production
parameter source, because it is scalar/unstructured while Akita's existing
commitments are module/ring-shaped and role-specific. The production stack
should call the same SIS sizing API used elsewhere, not bake in the note's
numbers.

## Current State

The current codebase has a tiered implementation, but it is opt-in and not part
of the benchmark/CI profile matrix.

- `CommitmentConfig::TIERED_COMMITMENT` defaults to `false`.
- The dedicated `fp128::D64OneHotTiered` config sets it to `true`.
- The planner carries a `tiered` discriminator and generated tiered schedule.
- The prover computes `u_final = F * decompose(u_concat)`, where `u_concat`
  comes from repeated smaller `B'` slices.
- The relation has extra tiered rows for the public `F` image and hidden
  `B_inner` consistency rows.
- The terminal segment path explicitly rejects tiered `u_concat` today.
- The profile mode exists, but profile CI excludes it.

The important mismatch is that tiered still uses `RingCommitment<F, D>` as the
public commitment shape. It can shrink the number of public rows, but it cannot
break the `D = 64` row-size floor. Commitment compression needs a public payload
with an arbitrary number of field coordinates.

There is parallel runtime ring-dimension work in flight:

- PR #227, `quang/runtime-ring-cutover`, is the collapsed incomplete attempt.
- PR #249, `quang/runtime-ring-full-cutover`, is the current full cutover.

That work is not a blocker for this stack, but it is directionally aligned. The
compression stack should not add new typed commitment wrappers while that work
is removing const-`D` from orchestration. The desired meeting point is an
untyped flat commitment payload, `FlatRingVec<F>` on current `main` or
`RingVec<F>` after PR #249, with all shape supplied by the schedule.

## Non-Goals

This stack does not try to preserve tiered compatibility.

This stack does not add migration shims between tiered and compression. Akita
does not promise backward compatibility here, and keeping both mechanisms would
make the relation, setup contribution, schedule tables, and terminal path harder
to reason about.

This stack does not initially optimize every possible commitment occurrence.
The safe order is to compress `v` first, then compress non-penultimate `u`, then
do the terminal-tail `t` optimization.

This stack should not fork the SIS/security logic from the in-flight direct
L-infinity estimator work. Compression sizing should go through a single
security-sizing API, with the current L-infinity to L2 table as the initial
backend and the direct L-infinity estimator as a later backend change.

## Commitment Objects

There are two commitments to discuss at each fold level.

### Opening commitment `v`

`v` is the public image for the current fold's opening witness. Compressing `v`
is conceptually straightforward:

```text
raw_v = D * e_hat
v_digits = Decompose(raw_v)
v_comp_1 = C_v_1 * v_digits
v_comp_2 = C_v_2 * Decompose(v_comp_1)   // optional second layer
public v = final v_comp
```

The hidden decomposition/intermediate data belongs to the current fold's
recursive witness and can be appended to the next witness produced by this fold.
The relation for the fold enforces the raw `D` image and the compression chain.

This should include the penultimate fold. The penultimate fold still has a real
`v` commitment for that fold, so the `v` proof-size win remains available.

### Next-witness commitment `u`

`u` is the public commitment carried into the next recursive level. The
compression suffix is appended to the next recursive witness, but it is not part
of the vector hit by the raw `B` commitment. This is the same structural idea as
the current tiered commitment: one relation computes a raw commitment over the
folded witness payload, and a second relation binds/compresses the hidden raw
commitment image.

This removes the apparent self-reference. The raw `B` image is defined only on
the base folded witness, not on the newly appended compression suffix.

The intended model should be:

```text
w_next = base_next_witness || compression_suffix || padding

raw_u = B * base_next_witness
compression_suffix encodes Decompose(raw_u) and optional intermediate images
public u = C_u(...compression_suffix...)
```

In other words, the public compressed `u` binds the suffix through the
compression matrix, while the relation binds the suffix back to the raw `B`
image of the base prefix. Sumcheck can enforce these consistency relations as
ordinary batched relation claims. They may be implemented as a new unstructured
sumcheck instance or folded into the existing ring-relation machinery, but the
schedule/layout must preserve the base-prefix/suffix distinction.

The implementation therefore needs a two-part next-witness layout:

- a semantic/base prefix that participates in the raw `B` commitment;
- a compression suffix that is checked by compression rows;
- schedule-visible padding to align the physical flat witness to ring elements.

## Stop Rule

Compression should stop at the penultimate fold level for `u`.

Definitions:

- A fold is terminal/penultimate when its immediate successor is the terminal
  `Direct` step.
- In code today, `ExecutionSchedule::level_schedule` already exposes this as
  `is_terminal`.
- In the planner, the dynamic program already distinguishes the branch whose
  suffix is `Direct` and computes a `next_witness_len_terminal` with today's
  `MRowLayout::WithoutDBlock`.

The PR2 policy should be:

| Fold level | Compress `v`? | Compress next `u`? | Send raw next `u`? |
| ---------- | ------------- | ------------------ | ------------------ |
| Non-penultimate fold | yes | yes | no |
| Penultimate fold | yes | no | yes, temporarily |
| Terminal `Direct` | no fold `v` | no next `u` | no |

The reason to stop `u` compression at the penultimate fold is that the next
witness is the terminal witness. We should not pay to construct and prove a
compressed `u` for a witness that we will immediately send in terminal form. In
PR2 the penultimate fold keeps the existing raw final `u` path so the
compression cutover and terminal-tail cutover are not mixed. In PR3 that final
raw `u` disappears completely.

### Recognizing the penultimate fold

Do not add a global schedule scan for this. The planner already learns the
penultimate fact at the transition where it matters.

A fold is penultimate exactly when the chosen suffix after that fold is
`Direct`. In the suffix dynamic program this is the "current step is `Fold`,
successor is `Direct`" branch. In the current planner this is also the branch
that uses `next_witness_len_terminal`, computed under today's
`MRowLayout::WithoutDBlock`.

Use that local branch fact directly:

```text
if successor_is_direct {
    // Penultimate fold.
    v_plan = compress_v(...)
    u_plan = None

    // PR2: keep pricing/proving the existing raw final u.
    // PR3: remove final u and use the no-D/no-COMMIT terminal layout.
} else {
    // Ordinary recursive fold.
    v_plan = compress_v(...)
    u_plan = compress_u(...)
}
```

At runtime, use the same fact that already exists on the expanded schedule.
`Schedule::level_schedule` sets `ExecutionSchedule::is_terminal` by checking
whether `steps[level + 1]` is `Direct`. Prover/verifier code should consume that
flag or the equivalent successor-is-`Direct` check, not recompute a separate
notion of penultimate.

This is O(1) at each planner transition and O(1) at each runtime fold. It adds
no extra pass over the schedule and no repeated traversal inside scoring.

## Terminal Tail Optimization

The final PR in the stack should implement the shelved terminal-tail
optimization:

- the final recursive witness should not be committed through `u`;
- the relevant terminal state should be the `t` segment of the terminal witness;
- verifier checks should bind that terminal `t` state directly instead of
  replaying a final `B` commitment.
- the terminal M-row layout should drop both the D/opening block and the
  COMMIT/B block.

This is separate from compression but depends on the same stopping rule. The
penultimate fold still compresses its own `v`; it simply does not produce a
compressed `u` for the terminal witness. PR2 keeps the raw final `u`; PR3
removes it and shrinks the terminal relation/tail accordingly.

There is already adjacent design material in `specs/tail-wire-encoding.md`.
That spec describes the terminal `t`-state direction and notes that tiered
terminal layouts must reject or route through the same `t` path. Removing
tiered first makes the terminal optimization cleaner.

Today's enum name `MRowLayout::WithoutDBlock` is not precise enough after PR3.
It currently means "drop D but keep COMMIT/B." The terminal-tail cutover should
replace it with a layout whose name and row offsets say what happens after the
full cutover, for example:

```rust
enum MRowLayout {
    WithDAndCommitBlocks,
    WithoutDAndCommitBlocks,
}
```

If PR2 needs the old shape temporarily, keep it only until PR3. Do not leave a
long-term ambiguous variant where "without D" secretly still includes B.

## Compression Layers

The first implementation should cap recursive compression at two layers.

This is not a security assumption. It is a planner/complexity cap:

- one layer gets below the ring-row floor;
- the second layer captures the common "commitment of compressed commitment"
  contraction;
- further layers should only be used if the planner can show a net byte win
  after accounting for the suffix witness growth and extra relation rows.

The local sweep suggests that some scalar settings reach their fixed point in
two layers, while others take three steps to fully stabilize. So "two" is a
reasonable first cap, not a theorem. The config should make the cap explicit,
for example `max_compression_layers = 2`, and the planner should be able to
evaluate `0..=cap` layers rather than hard-coding exactly two.

The default stopping condition inside the cap should be:

```text
choose the smallest total proof bytes among 0..=max_layers
```

not "always apply all layers." A second layer can be disabled for a level if
the added suffix/intermediate data costs more than the public bytes saved.

## Witness Layout

Compression witnesses should be appended at the end of the next recursive
witness, after the ring-switch quotient segment.

Current terminal/intermediate witness construction is effectively:

```text
z_hat || e_hat || t_hat || optional tiered u_concat || r_hat
```

The new shape should be:

```text
z_hat || e_hat || t_hat || r_hat || compression_suffix || padding
```

The suffix is unstructured and may not be a multiple of the current ring
dimension. The implementation must make padding explicit. Today
`RecursiveWitnessFlat` and `SuffixWitnessView::from_i8_digits` require the flat
digit length to be divisible by `D`; `ring_switch_finalize` assumes
`w.len() / D` ring elements. Therefore the planner must account for:

- logical compression suffix length;
- physical zero padding to the next ring element;
- transcript/proof descriptors that let prover and verifier agree on the
  logical suffix boundaries.

Padding must be included in the committed/evaluated physical witness shape but
must not become an unconstrained hiding place. It should either be constrained
to zero or excluded by construction and checked during witness assembly.

## Public Wire Shape

Compression cannot use `RingCommitment<F, D>` for the compressed public output.

At `fp128, D = 64`, one ring row is about 1024 bytes. A compressed commitment
targeting 192 to 512 bytes corresponds to roughly 12 to 32 field elements, not
an integral number of `D = 64` ring rows.

The stack needs a public commitment payload that can represent both:

- ring-native commitments during the transition and for uncompressed levels;
- scalar/flat compressed commitments with an arbitrary field-element length.

The clean direction is to cut public commitments over to the untyped flat vector
container already being introduced by the runtime ring-dimension work:

- on current `main`, this is closest to `FlatRingVec<F>`;
- after PR #249, this appears as `RingVec<F>`.

No enum is required if the schedule owns the semantic shape. A ring-native
commitment is just a flat coefficient vector whose expected length is
`num_rows * ring_dim`; a compressed scalar commitment is a flat coefficient
vector whose expected length is the compression output dimension. The verifier
must deserialize using schedule context and must reject shape mismatches without
panicking.

The transcript labels can keep the semantic names (`v`, `next_w_commitment`),
but the bytes absorbed must be the exact serialized compressed payload. This is
a wire change and must be tested with cross-config rejection.

## Planner Changes

The planner already has most of the information needed to recognize the
penultimate fold:

- `ExecutionSchedule` has `is_terminal`.
- `Schedule::level_schedule` sets it by checking whether the next step is
  `Direct`.
- `derive_candidate_level_params` already computes both
  `next_witness_len` and `next_witness_len_terminal`.
- The suffix dynamic program already has a branch where the next step is
  terminal direct.

Minimal planner additions:

1. Add a `CompressionPolicy` derived from config:
   - enabled/disabled;
   - max layers;
   - basis / digit policy;
   - target security bits from the existing security policy;
   - whether `u` compression is allowed on penultimate folds, fixed false for
     this stack.
2. Add a schedule-visible `CompressionPlan` for each fold:
   - `v_plan`;
   - `u_plan`;
   - hidden suffix logical length;
   - hidden suffix padded physical length;
   - public compressed field-element length;
   - security certificate or sizing trace sufficient for diagnostics.
3. Price candidate levels using the compression plan:
   - public bytes for compressed `v`;
   - public bytes for compressed `u`, except penultimate;
   - extra witness bytes/rows caused by suffix and padding;
   - extra setup contribution rows for compression matrices;
   - verifier work for compression relation rows.
4. When the DP branch suffix is terminal direct, price `u_plan = None`.
   In PR2, still price the existing raw final `u` bytes and rows. In PR3,
   drop those bytes and switch terminal sizing to the no-D/no-COMMIT layout.
   This must be keyed off the DP successor-is-`Direct` branch, not a later
   post-processing scan.

The planner should not perform an expensive global scan to identify
penultimate levels. The penultimate fact is already local in the DP transition
and in the expanded schedule.

## Security Sizing

Compression matrices should be sized through the same security API as the rest
of Akita. The first implementation can use the current L-infinity-to-L2 path
and L2 table; the open direct L-infinity estimator PR can replace the backend
without changing compression call sites.

The API should be role-aware. Compression introduces new SIS roles:

- `CompressVLayer(i)`
- `CompressULayer(i)`

Each role needs:

- input length `m`;
- input coefficient bound after decomposition;
- modulus/field;
- output dimension `n`;
- target security bits;
- scalar/unstructured vs module/ring shape.

The compression layer should be scalar/unstructured if we want arbitrary field
output dimensions. Reusing module/ring parameters would reintroduce the
ring-row floor.

The security certificate must cover every compression layer independently, just
as current certification covers `A`, `B`, and `D` roles. Do not let proof-size
pricing and security certification use different bounds.

## Setup and Relation Rows

The compression matrices are verifier-known setup matrices, but they are not
the current `A/B/D` ring matrices.

Implementation needs a new setup contribution path or an extension of the
existing setup contribution machinery for scalar/unstructured rows. This is
another reason to delete tiered first: the current setup contribution code has
special tiered `B'/F` handling that would otherwise compose poorly with scalar
compression rows.

Relation row layout should stop thinking of the public commitment block as
"ring rows only." It needs role-specific public outputs:

- compressed or raw `v`;
- compressed or raw `u`;
- hidden decomposed raw images;
- compression consistency rows.

The relation should keep one canonical offset/layout computation. Avoid adding
helper wrappers that reconstruct offsets differently for compression.

## PR Stack

Use one worktree per PR. The preferred stack has three PRs because each PR has a
clear semantic effect and leaves the code in a coherent state.

| PR | Branch | Worktree | Semantic change |
| -- | ------ | -------- | --------------- |
| 1 | `quang/remove-tiered-commitment` | `../akita-remove-tiered-commitment` | Delete tiered commitment; reorder M layout to `consistency \| A \| B \| D`; remove dead M public-block scaffolding; rename witness `num_public_rows` → `num_z_segments`. |
| 2 | `quang/commitment-compression` | `../akita-commitment-compression` | Add compressed commitments: compressed `v` on every fold, compressed `u` on every non-penultimate fold, raw final `u` preserved temporarily. |
| 3 | `quang/terminal-t-no-final-u` | `../akita-terminal-t-no-final-u` | Remove the final recursive `u`; bind terminal `t`; rename/update terminal M-row layout to drop both D and COMMIT/B. |

PR2 is the largest PR. It should still be one PR if possible because splitting
payload, planner, `v`, and `u` into separate PRs creates intermediate protocol
states that are less meaningful to review. Use internal milestones inside PR2
rather than separate PRs unless the diff becomes unreviewable.

Suggested internal milestones for PR2:

1. Payload and schedule substrate compile: flat commitment payloads,
   `CompressionPolicy`, `CompressionPlan`, descriptor binding, and proof-size
   estimates exist, but compression can still be disabled.
2. `v` compression works end-to-end on every fold, including the penultimate
   fold.
3. Non-penultimate `u` compression works end-to-end with the base-prefix /
   compression-suffix witness split.
4. Profile and CI coverage are enabled for the compressed verifier-optimized
   profile, while the penultimate raw `u` remains unchanged until PR3.

These milestones are review checkpoints, not merge points. PR2 should not merge
with only payload plumbing or only `v` compression unless we deliberately decide
to split the stack.

### PR1: Remove tiered commitment + M layout / naming cutover

Purpose: remove the old `B' -> F` tiered protocol so compression does not have
to compose with it, finish the y-ring trace-internalization cleanup so M-row
layout and witness naming are unambiguous before compression lands, and reorder
M rows so the `A` block immediately follows the consistency row.

PR1 is intentionally broader than tiered deletion alone. It also removes stale
M-matrix "public output row" scaffolding (openings already bind through the fused
trace term in stage-2 sumcheck, not through `M` rows) and renames the witness
`z_folded` width parameter from `num_public_rows` to `num_z_segments`.

**Locked M-row layout after PR1:** `consistency (1) | A | B | D`. There is no
public block in `M`. Public openings bind via the fused trace term in stage-2
sumcheck ([`specs/y-ring-trace-internalization.md`](y-ring-trace-internalization.md)).

**Terminal fold (`MRowLayout::WithoutDBlock`):** `consistency | A | B` (the
trailing `D` block is dropped). This aligns with PR3's direction of shrinking
terminal rows toward `consistency | A`.

**Descriptor version:** do not bump `AKITA_INSTANCE_DESCRIPTOR_VERSION`; row
order is an implicit protocol convention, not a per-instance descriptor field.

**Witness naming:** `num_z_segments` counts `z_folded` witness segments (planner
sets `1` for ordinary folds, `G` for grouped roots). It is not an M-row count.

Expected implementation surface:

- `akita-config`:
  - delete `TIERED_COMMITMENT`;
  - delete `fp128::D64OneHotTiered`;
  - remove tiered policy plumbing from `policy_of` and schedule selection.
- `akita-planner` / `akita-schedules`:
  - delete the tiered planner policy bit;
  - delete tiered schedule table selection;
  - delete generated `fp128_d64_onehot_tiered` schedules;
  - remove tiered generated-row fields such as `tier_split` / `n_f`;
  - planner walk uses `num_z_segments` for witness `z_folded` width.
- `akita-types`:
  - remove `LevelParams::tier_split` and `f_key`;
  - delete `effective_commit_rows()`, `b_inner_rows_per_group()`,
    `u_concat_ring_len_per_group()`; use `b_key.row_len()` at call sites;
  - simplify `m_row_count_for` and row-offset helpers to
    `consistency | A | B | D` (drop `num_public_outputs` parameter);
  - update `generate_y`, quotient row dispatch, `relation_claim_from_rows*`,
    setup-contribution `eq_tau1` slicing, and verifier hardcoded offsets to match;
  - rename witness-sizing `num_public_rows` → `num_z_segments` in schedule,
    tail, and terminal witness helpers;
  - delete `SetupContributionPlanInputs.num_public_rows`; `d_start = 1`;
  - delete `RingRelationSegmentLayout.offset_u` and tiered `u_len`;
  - remove tiered SIS norm helpers and descriptor bindings;
  - fix `layout/proof_size.rs` `r_count` to use the simplified `m_row_count_for`
    (drops the spurious extra row still priced today).
- `akita-prover`:
  - delete `tiered_commit_u_final`;
  - delete tiered `u_concat` witness emission from ring-switch coeff assembly;
  - delete tiered branches in ring relation quotient/setup contribution paths;
  - delete terminal rejection paths that exist only for tiered `u_concat`;
  - remove dead M public-row eval paths (`public_weights`, `NUM_PUBLIC_M_ROWS`,
    `num_public_m_rows` locals).
- `akita-verifier`:
  - remove tiered row-shape handling and cross-config tiered checks;
  - remove setup-contribution / row-eval `num_public_rows` threading;
  - simplify setup-contribution fixtures away from tiered and M-public layouts.
- Tests/benches/docs:
  - delete `tiered_e2e`;
  - remove `onehot_fp128_d64_tiered` profile mode;
  - archive `specs/tiered-commitment.md` to `specs/archive/2026-Q2/`;
  - update live specs that still describe `consistency | public | D | …` M layout
    or witness `num_public_rows` (for example `tail-wire-encoding.md`,
    `multi-group-batching.md`).

Acceptance tests:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`
- `./scripts/check-doc-guardrails.sh`
- targeted grep: no live `tiered`, `f_key`, `tier_split`, or `u_concat` code
  remains outside archived specs or historical notes.
- targeted grep: no `num_public_outputs`, `NUM_PUBLIC_M_ROWS`, or
  `num_public_m_rows` outside archive.
- targeted grep: no `num_public_rows` in crates/specs/book outside
  `trace_weight/` and archive (witness param is `num_z_segments`).

### PR2: Add commitment compression

Purpose: introduce the new proof-size mechanism while preserving the existing
terminal raw `u` behavior. After this PR, the steady-state recursive folds use
compressed public commitments, but the penultimate fold still sends the final
raw `u`.

Commitment policy:

| Fold level | Public `v` | Public next `u` |
| ---------- | ---------- | --------------- |
| Non-penultimate fold | compressed | compressed |
| Penultimate fold | compressed | raw, uncompressed |
| Terminal `Direct` | none | none |

Expected implementation surface:

- Public commitment payload:
  - cut orchestration/storage to untyped flat commitment vectors:
    `FlatRingVec<F>` on current `main`, or `RingVec<F>` if PR #249 has landed;
  - remove public orchestration dependencies on `RingCommitment<F, D>` where
    compression needs arbitrary scalar output length;
  - verifier deserialization must use schedule-owned expected lengths and
    reject malformed lengths without panicking.
- Planner and schedules:
  - add `CompressionPolicy` with `max_layers = 2`;
  - add per-fold `CompressionPlan { v_plan, u_plan, suffix_len, padded_len,
    public_len, sizing_certificate }`;
  - compute plans for `v` on every fold;
  - compute plans for `u` only when the successor is another fold;
  - keep penultimate raw `u` pricing in PR2;
  - account for compression suffix growth and padding in `next_w_len`;
  - include compression plans in generated schedule descriptors/digests.
- Security sizing:
  - add compression SIS roles, for example `CompressVLayer(i)` and
    `CompressULayer(i)`;
  - size scalar/unstructured matrices through the existing security API;
  - keep the call site compatible with the direct L-infinity table cutover.
- Witness layout:
  - append compression witness data after the ring-switch quotient segment:
    `z_hat || e_hat || t_hat || r_hat || compression_suffix || padding`;
  - `B` applies only to the base folded witness prefix;
  - compression rows bind the appended suffix to the hidden raw commitment
    images;
  - padding is schedule-visible and constrained/checked as zero.
- Prover:
  - compute raw `v = D * e_hat`, decompose it, and emit compressed public `v`;
  - compute raw non-penultimate `u = B * base_next_witness`, decompose it, and
    emit compressed public `u`;
  - append all hidden decompositions/intermediate compression witnesses to the
    next recursive witness suffix;
  - absorb compressed payload bytes under the existing semantic transcript
    labels.
- Verifier:
  - reconstruct expected payload lengths from the schedule;
  - verify compressed `v`/`u` relation claims through the chosen sumcheck
    placement;
  - preserve raw final `u` verification at the penultimate fold for PR2.
- Relation/sumcheck placement:
  - preferred implementation may use a new unstructured sumcheck instance for
    compression rows;
  - acceptable alternative is extending the ring-relation machinery, provided
    it keeps one canonical layout and does not treat scalar compression rows as
    fake ring rows.
- Proof-size accounting:
  - replace `v_bytes = n_d * D * field_bytes` with compressed `v` bytes when
    `v_plan` is present;
  - replace next-commit bytes with compressed `u` bytes for non-penultimate
    folds;
  - keep raw final `u` bytes for the penultimate fold in PR2;
  - charge compression suffix and extra sumcheck bytes.
- Tests/benches/docs:
  - add planner tests for penultimate behavior;
  - add tamper tests for compressed `v` and non-penultimate compressed `u`;
  - add malformed-length tests for flat commitment payloads;
  - add padding tests where compression suffix length is not divisible by the
    fold ring dimension;
  - add a compressed verifier-optimized profile/bench mode and include it in CI
    once stable.

Acceptance tests:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`
- `./scripts/check-doc-guardrails.sh`
- e2e proof with at least two fold levels, so both compressed non-penultimate
  `u` and raw penultimate `u` are exercised.
- cross-config rejection between compressed and uncompressed schedules.

### PR3: Terminal `t` cutover, no final `u`

Purpose: finish the tail reduction by deleting the final recursive `u` and
shrinking the terminal relation rows accordingly.

Expected implementation surface:

- M-row layout:
  - replace the terminal meaning of `MRowLayout::WithoutDBlock` with an explicit
    no-D/no-COMMIT layout, for example
    `MRowLayout::WithoutDAndCommitBlocks`;
  - if keeping a two-variant enum, rename `WithDBlock` to
    `WithDAndCommitBlocks`;
  - update `n_d_active_for`, `f_start`/commit-block start helpers,
    `a_start`, and `m_row_count_for` so the terminal layout is:
    `consistency | A`;
  - do not leave call sites locally subtracting B/COMMIT rows.
- Planner/schedules:
  - change `next_witness_len_terminal` to use the no-D/no-COMMIT layout;
  - update DP proof-size scoring to remove raw final `u` bytes;
  - regenerate schedules and descriptor digest pins.
- Tail witness:
  - terminal direct witness includes the terminal `t` state needed by the
    verifier;
  - `tail_segment_layout` computes `r_field_elems` from the no-D/no-COMMIT
    row count;
  - terminal witness shape and byte estimates include `t`, not final `u`.
- Prover:
  - penultimate fold no longer computes/absorbs/sends next-witness `u`;
  - terminal witness assembly exposes `t` in the terminal segment;
  - ring-switch finalize paths use the no-D/no-COMMIT layout for terminal
    quotient rows.
- Verifier:
  - penultimate replay does not expect `next_w_commitment`;
  - terminal verifier checks revealed witness maps to the terminal `t` state;
  - malformed proofs that include or omit the wrong terminal payload reject
    cleanly.
- Proof-size accounting:
  - remove raw final `u` bytes from penultimate fold pricing;
  - shrink terminal `r` tail by removing COMMIT/B rows;
  - update profile output labels so the win is attributable to terminal-tail
    cutover, not compression.
- Tests/benches/docs:
  - terminal-tail e2e proving no final `u` is serialized;
  - planner test that terminal row count excludes both D and COMMIT/B;
  - proof-size regression showing tail shrink;
  - update `specs/tail-wire-encoding.md`, the book, and profile docs.

Acceptance tests:

- `cargo fmt -q`
- `cargo clippy --all --message-format=short -q -- -D warnings`
- `cargo test`
- `./scripts/check-doc-guardrails.sh`
- targeted grep: no terminal path relies on `WithoutDBlock` semantics that keep
  a COMMIT/B block.

## Enablement Policy

The end-state should not be an obscure opt-in sibling like
`D64OneHotTiered`. Commitment compression is intended to define the
verifier-optimized fp128 profile.

Recommended rollout:

1. During PR2, keep compression behind an explicit config/profile until the
   compressed verifier-optimized profile is green.
2. At the end of PR2, compression should be on for that verifier-optimized
   profile, with raw final `u` still present.
3. At the end of PR3, remove the raw final `u` from that profile as part of
   terminal-tail cutover.
4. Do not keep tiered as an alternative production profile.

This answers the "verifier optimized environment" concern: the long-term knob
should be a profile/config policy, not a collection of ad hoc protocol variants.
The protocol change is still real, because breaking the 1 KB ring-row floor
requires a new public commitment shape and relation checks.

## CI and Bench Coverage

Current tiered status:

- tiered is not on by default;
- tiered has a profile mode;
- profile CI excludes the tiered mode;
- tiered e2e tests exist, but they are special-purpose coverage.

Compression end-state should be different. If this is the verifier-optimized
profile, CI and bench profiles should exercise it.

Minimum coverage:

- planner unit tests for `v` compression at penultimate folds;
- planner unit tests that `u` compression is absent at penultimate folds;
- proof-size tests showing public bytes and suffix bytes are both priced;
- malformed proof tests for wrong compressed payload length;
- tamper tests for compressed `v`;
- tamper tests for compressed non-penultimate `u`;
- padding tests for suffix lengths not divisible by `D`;
- cross-config rejection between compressed and uncompressed schedules;
- terminal-tail tests proving no final `u` is expected;
- profile benchmark mode for compressed fp128 D64 one-hot;
- profile CI includes the compressed verifier-optimized mode once stable.

Standard verification per PR:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
./scripts/check-doc-guardrails.sh   # when docs/specs/book are changed
```

## Open Implementation Choices

These are the points that still need an explicit implementation choice. The
`u` semantics above are not open: `B` applies only to the folded witness/base
prefix, and compression rows bind the appended suffix.

1. `u` compression relation placement.
   The protocol semantics are now clear: `B` is applied only to the folded
   witness/base prefix, and the appended compression suffix is bound by
   consistency relations. The remaining implementation choice is where those
   relations live: a new unstructured sumcheck instance, the existing
   ring-relation machinery, or a shared descriptor once the sumcheck engine
   stack is ready.

2. Public payload type.
   To get 192 to 512 byte commitments, compressed outputs must be serialized as
   field elements, not ring rows. The likely concrete type is the untyped
   `FlatRingVec<F>` / `RingVec<F>` commitment payload from the runtime ring
   cutover, with schedule-owned shape and no public `RingCommitment<F, D>` API
   at orchestration boundaries.

3. Root commitment scope.
   This spec focuses on recursive fold commitments. We need to decide whether
   the root user commitment is also compressed in the same stack, or whether
   root compression is a later PR after recursive compression lands.

4. Compression matrix setup.
   The compression matrices are scalar/unstructured. We need to decide whether
   to extend the current setup contribution machinery or add a separate scalar
   setup contribution path.

5. Default profile.
   I recommend eventual always-on compression for the verifier-optimized fp128
   profile, but staged opt-in during the PR stack. The exact config names and
   rollout point need a decision.

6. Compression depth cap.
   I recommend `max_compression_layers = 2` initially, with planner evaluation
   of `0..=2`. More than two should require measured proof-size wins and
   explicit schedule support.

7. ZK interaction.
   Current tiered tests are non-ZK-gated. Compression should either be made
   compatible with ZK blinding from the first behavior PR or explicitly rejected
   under `zk` until the hiding analysis is done.

8. Multi-group scope.
   Existing tiered multi-group support is out of scope. Compression should
   start with the same-point/single recursive commitment path unless we decide
   to pay the complexity cost for multi-group root batches immediately.

## Problematic Assumptions to Avoid

- Do not assume a ring commitment can be "small" below one ring row. It cannot.
- Do not append compression witnesses without schedule-visible padding and
  constraints.
- Do not apply the raw `B` commitment to the appended compression suffix. `B`
  remains a commitment to the folded witness/base prefix; compression rows bind
  the suffix.
- Do not let planner proof-size estimates and security certification use
  different bounds.
- Do not compress the final recursive `u`. PR2 keeps it raw temporarily; PR3
  removes it rather than compressing it.
- Do not keep tiered and compression live together unless there is a strong,
  measured reason. The relation and terminal path already show the bloat.
- Do not treat the local small-instance sweep as final production sizing. It is
  evidence for the direction and layer cap, not the source of truth.

## Expected Wins

The expected per-fold public-byte win is at least one ring row when both `u` and
`v` cross from ring-row encoding to scalar compressed encoding. In the current
fp128 D64 setting, one ring row is roughly 1 KB. If compressed outputs land in
the 192 to 512 byte range, each compressed commitment saves roughly 512 to 832
bytes before accounting for added suffix witness data.

The next-level witness grows by the hidden decompositions/intermediate images.
The local sweep suggests that this growth should be small relative to the public
bytes saved, but the production planner must price it exactly.

The penultimate fold should still save on `v`, while avoiding the bad trade of
compressing a final `u` that the terminal tail should not need.

## Implementation Surface

This is a broad protocol cutover, not a local optimization. Expected touched
areas:

- `akita-config`: config/profile policy and removal of tiered config.
- `akita-planner`: compression policy, per-level plans, DP pricing, generated
  schedule format.
- `akita-schedules`: delete tiered table, regenerate compressed profiles.
- `akita-types`: commitment payload type, proof serialization, proof-size
  accounting, layout offsets, SIS roles, setup contribution descriptors,
  terminal/tail descriptors.
- `akita-prover`: raw commitment computation, compression chain witnesses,
  suffix assembly/padding, transcript absorption, relation inputs.
- `akita-verifier`: payload shape validation, transcript absorption, relation
  claim reconstruction, terminal no-`u` behavior.
- `akita-pcs` tests/benches/profile examples: e2e, tamper, cross-config,
  profile modes.
- `book`, `docs`, and `specs`: replace tiered narrative with compression and
  terminal stopping rules.
- `profile/akita-recursion`: glue currently assumes `RingCommitment<F, D>` and
  will need payload-aware handling if compressed profiles are benchmarked there.

This surface is why the PR stack should remove tiered first. Keeping tiered live
while changing public commitment payloads, relation rows, suffix witness layout,
and terminal handling would multiply the number of cases.
