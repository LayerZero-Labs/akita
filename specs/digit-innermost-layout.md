# Spec: Digit Innermost Fold and Witness Layout

| Field | Value |
|---|---|
| Author(s) | Quang Dao; Codex assistant |
| Created | 2026-07-10 |
| Revised | 2026-07-15 |
| Status | active |
| PR | #294 |
| Supersedes | Root and recursive layout decisions in `setup-layout-repack.md`, `protocol-core-eor-consolidation.md`, and `distributed-verifier-row-eval.md` |
| Superseded by | |
| Book chapter | |

This spec follows the lifecycle in [`PRUNING.md`](PRUNING.md). It replaces the first version of this file. That version used a power of two fold block count, a compact position count, and a second virtual opening address. This version reverses those choices.

## Summary

Akita will use one digit innermost witness order at the root and at every recursive level.

For each commitment group, the number of positions in one source fold slice is a power of two. The number of live fold slices is exact and may be any positive integer. The final source slice may be partial. This gives the source index a direct binary split without adding internal zero gaps.

```text
source index = fold index * positions per fold + position

              high bits                       low bits
           [ fold index ]                 [ position ]
```

The decomposed witness keeps each digit next to the value it decomposes.

```text
z_hat[position][commit digit][fold digit]
e_hat[claim][live fold index][opening digit]
t_hat[claim][live fold index][A row][opening digit]
r_hat[relation row][quotient digit]
```

Digits remain tightly packed. Akita does not insert zero digits to make a digit count a power of two.

For distributed proving, each group divides its exact live fold prefix into contiguous shard ranges. Internal boundaries are aligned to a chosen power of two granule. A final residual range remains live and tight. Each group and shard unit contains `[z_hat | e_hat | t_hat]`. One shared `r_hat` tail follows all units.

## Current state

### Current main branch

The current `main` branch still has these properties.

* `BlockOrder::RowMajor` is used at the root.
* `BlockOrder::ColumnMajor` is used at recursive levels.
* `num_blocks` is a power of two.
* Recursive `block_len` may be a tight non power of two value.
* Multi chunk layouts require equal power of two block windows.
* `WitnessLayout` stores offsets and lengths in parallel vectors.
* `TraceChunkLayout` copies the same chunk geometry again.
* Tensor challenge factors are called left and right, and both lengths multiply to the full power of two block count.
* The tensor choice is stored on the top level `LevelParams`, so precommitted groups cannot choose it independently.
* Multi group plus multi chunk is rejected.

### PR #294 implementation status

Slices 1 through 3 are complete. They establish the final parameter geometry, the physical opening and fold order, and the canonical witness ranges.

Slice 4 is in progress. The branch stores exact live tensor prefixes, supports a partial final tensor row, and keeps tensor challenges in factored sparse form. Challenge sampling binds the group index and group local shape. Security sizing uses each group local shape and A width. Small signed sparse coefficients use add, subtract, and double fast paths.

One root grind nonce satisfies every group. The prover derives one batch contract and one probe order from the ordered group geometry. It previews each candidate across every group and changes the live transcript only after every group accepts. The verifier validates the same batch contract before replay.

Public tensor evaluation boundaries now call the full shape validator. A zero or non-power-of-two tensor low length returns `AkitaError` before indexing, division, or modulo.

The following work remains before Slice 4 is complete.

* Fix the D128 root proving failure where a predecomposed digit row exceeds the scheduled `log_basis` range.

The current challenge evaluator no longer stores one product evaluation for every live fold. Some verifier paths still visit every live fold and recompute factor evaluations. Slice 6 owns the succinct carry and trace evaluator that removes this remaining work. The current allocation change is not the final performance result.

The failing multi group terminal root tests assert a schedule shape that current `main` now rejects. They must be replaced with tests for grouped nonterminal folds and scalar direct or terminal folds. They are not an accepted later slice target.

### Integration with current `main`

PR #301 landed on `main` after this branch diverged. It makes setup prefix slots exact, moves recursive prefix materialization into the new setup and prover source modules, and fixes the schedule topology as follows.

* A grouped fold is nonterminal and recursive.
* A direct fold consumes one witness group and creates no outgoing setup claim.
* A terminal fold is scalar and direct.
* A recursive successor contains exactly one witness group and one setup prefix group.

This spec does not reopen that topology. Its canonical group and shard ranges apply to group bearing nonterminal folds. Direct and terminal consumers use the same range authority for their single witness group. The merge must keep the new setup prefix ownership from `main`, port the digit innermost address and exact live count rules into it, and delete the old `akita-setup/src/recursion.rs` path.

### Coordination with distributed prover work

PR #296 closed without landing on `main`. Any successor must consume the group and shard ownership ranges defined here. It must not add a second recursive witness hierarchy, a second range resolver, or a quotient per machine. Process placement and edge communication belong to the distributed prover. Coefficient order and semantic ownership belong here.

## Goals

This cutover has the following goals.

* Use one physical coefficient order at every level.
* Give the source fold index a direct low position and high fold index split.
* Keep all live witness and digit data tight.
* Support any positive live fold count.
* Support flat and tensor fold challenges over the exact live fold prefix.
* Support one or more groups and one or more shards with the same layout rule.
* Give setup, relation, trace, terminal, and recursive code one range authority.
* Keep verifier evaluation proportional to the compact factors when the weight is structured.
* Keep sparse challenge coefficients sparse through verifier evaluation.
* Delete old modes, copied geometry, and unused evaluators as each replacement lands.

## Non goals

This cutover does not add compatibility with old proof bytes or old commitment bytes.

This cutover does not add a second layout mode.

This cutover does not require unequal shard sumcheck work to be communication free. A prover may exchange the small edge state needed to complete a pair or a short fold chunk across a machine boundary.

This cutover does not require flat random challenges to have a sublinear evaluator. A flat challenge has one independent value per live fold.

This cutover does not redesign the mixed ring projection algebra. It keeps the projection work from PR #294 and makes it consume the final witness geometry.

## Terminology and geometry

For one commitment group `g`, define the following values.

```text
N_g       exact live source ring elements per claim
L_g       positions in one source fold slice
F_g       exact live fold slices per claim
F_cap_g   Boolean fold capacity
C_g       number of claims in the group

L_g       = 2^m_g
F_g       = ceil(N_g / L_g)
F_cap_g   = next_power_of_two(F_g)
```

`N_g`, `L_g`, and `F_g` are semantic values. `F_cap_g` and both bit counts are derived values.

The constructor checks:

```text
N_g > 0
L_g > 0
L_g is a power of two
F_g = ceil(N_g / L_g)
F_cap_g = next_power_of_two(F_g)
```

The live source condition is:

```text
0 <= f < F_g
0 <= p < L_g
f * L_g + p < N_g
```

The last live fold slice may contain a zero suffix.

```text
N = 13, L = 4, F = 4, F_cap = 4

f = 0: [  0   1   2   3 ]
f = 1: [  4   5   6   7 ]
f = 2: [  8   9  10  11 ]
f = 3: [ 12   .   .   . ]
```

If `F_g` is not a power of two, the remaining capacity blocks are absent from physical storage and are zero in the Boolean domain.

```text
N = 19, L = 4, F = 5, F_cap = 8

live folds:        0  1  2  3  4
capacity only:                    5  6  7
```

There is no virtual position stride. The physical source address and the multilinear address are both:

```text
address(f, p) = f * L_g + p
```

Because `L_g` is a power of two, the equality polynomial factors directly.

```text
eq(r, f * L_g + p)
    = eq(r_position, p) * eq(r_fold, f)
```

The variable order is:

```text
[ position bits | fold bits ]
     low bits         high bits
```

The high Boolean domain has `log2(F_cap_g)` variables. Evaluators consume only the live prefix `0 .. F_g`.

## Fold equation

Let `w_g[c, f, p]` be the source value for claim `c`, fold index `f`, and position `p`. Let `gamma_g[c, f]` be the sampled fold coefficient.

```text
z_g[p] = sum over c < C_g and f < F_g of
         gamma_g[c, f] * w_g[c, f, p]
```

For the partial last fold slice, `w_g[c, f, p]` is zero when `f * L_g + p >= N_g`.

The fold coefficient is a small ring challenge. The opening equality weight is not multiplied into `z_g` before decomposition. This preserves the fold infinity norm contract.

## Canonical parameter ownership

The cutover must reuse the current parameter model.

### `LevelParams`

`LevelParams` remains the full parameter object for the final group at a level. Replace its ambiguous fold geometry fields as follows.

```text
remove or rename                 final meaning

num_blocks                      live_fold_count = F
block_len                       fold_position_count = L
m_vars                          derive log2(L)
r_vars                          derive log2(F_cap)
new                             source_ring_len_per_claim = N
new                             shard_granule = S
fold_challenge_shape            group local Flat or Tensor { fold_low_len }
```

The implementation may keep the existing Rust field names during one compile checkpoint, but the final public names must state the meanings above. Do not keep deprecated aliases.

`LevelParams::with_decomp` must change its orientation. Given a power of two `L` and an exact source length `N`, it derives `F = ceil(N / L)`. It must not start from a power of two `F` and derive a tight `L`.

### `PrecommittedLevelParams`

`PrecommittedLevelParams` remains the group local parameter object for each precommitted root group. It must carry the same group local fold fields as `LevelParams`, including the challenge shape and shard granule.

Extend `LevelParamsLike` with the final group local accessors. Remove accessors for stored values that become derived.

### `OpeningClaimsLayout`

`OpeningClaimsLayout` remains the authority for commitment groups in transcript order, the claim count in each group, and the root relation processing order.

Do not copy either group order into `WitnessLayout`. `WitnessLayout::resolve` must call the existing batch order methods and record the group index on each resolved unit.

### `WitnessLayout`

`WitnessLayout` remains the one resolved physical range authority.

Replace `WitnessChunkLayout` with one unit record for the product of a group and a shard. This is a rename and expansion of the current struct, not an additional layer.

```rust
pub struct WitnessUnitLayout {
    pub group_index: usize,
    pub shard_index: usize,
    pub global_fold_start: usize,
    pub live_fold_count: usize,
    pub z_range: Range<usize>,
    pub e_range: Range<usize>,
    pub t_range: Range<usize>,
}

pub struct WitnessLayout {
    pub units: Vec<WitnessUnitLayout>,
    pub r_range: Range<usize>,
}
```

The final fields should be private with checked accessors. They are shown as public in the sketch only to make the data model clear.

Delete `WitnessChunkLengths`. A checked range already contains an offset and a length. Do not keep parallel `chunks` and `chunk_lengths` vectors.

Keep `ChunkedWitnessCfg` and `MultiChunkProfileId` only as planner policy. They choose the shard count and the number of active levels. They must not serve as resolved ownership geometry.

### Types from the first PR version

Delete these types and update their call sites.

```text
OpeningBlockLayout
OpeningBatchWitnessGroup
SemanticGroupId
MachineChunkId
WitnessOwnershipUnit
OpeningBatchWitnessLayout
```

The existing types listed above already own their information.

### Mixed setup projection

Keep `SetupProjectionGeometry` from PR #294 as the one resolved owner of mixed A, B, and D ring projection. It is separate because ring projection is not witness ownership.

It may store role dimensions, projection ratios, required setup footprint, round count, and checked verifier work. It must not copy group order, witness ranges, digit counts, or shard ranges.

## Physical witness order

For each group in relation order, emit each shard in shard order.

```text
group 0
  shard 0: [ z_00 | e_00 | t_00 ]
  shard 1: [ z_01 | e_01 | t_01 ]

group 1
  shard 0: [ z_10 | e_10 | t_10 ]
  shard 1: [ z_11 | e_11 | t_11 ]

shared:    [ r ]
```

For one unit, let `s_j` be its first global fold index and let `F_j` be its live fold count.

Its exact segment lengths are:

```text
Z_g  = L_g * delta_c_g * delta_f_g
E_gj = C_g * F_j * delta_o_g
T_gj = C_g * F_j * n_A_g * delta_o_g
```

Resolve each unit from one checked cursor.

```text
z_range = cursor .. cursor + Z_g
e_range = z_range.end .. z_range.end + E_gj
t_range = e_range.end .. e_range.end + T_gj
cursor  = t_range.end
```

After the final unit:

```text
r_range = cursor .. cursor + relation_rows * delta_r
```

No caller may recompute these bases from a uniform stride.

### `z_hat`

`z_hat` has axes:

```text
[position][commit digit][fold digit]
```

The fold digit is innermost.

```text
z_index(p, d_c, d_f)
    = z_start + d_f + delta_f * (d_c + delta_c * p)
```

Every shard for the group contains a complete copy of `z_hat_g`.

### `e_hat`

`e_hat` has axes:

```text
[claim][local live fold index][opening digit]
```

The opening digit is innermost.

```text
u = global_fold_index - s_j

e_index(c, u, d_o)
    = e_start + d_o + delta_o * (u + F_j * c)
```

### `t_hat`

`t_hat` has axes:

```text
[claim][local live fold index][A row][opening digit]
```

The opening digit is innermost.

```text
t_index(c, u, a, d_o)
    = t_start + d_o
      + delta_o * (a + n_A * (u + F_j * c))
```

### `r_hat`

`r_hat` has axes:

```text
[relation row][quotient digit]
```

The quotient digit is innermost.

```text
r_index(row, d_r)
    = r_start + d_r + delta_r * row
```

There is one `r_hat` tail for the complete relation. It is not copied per group or per machine.

### Tight digit counts

The digit counts are role local.

```text
delta_o   opening decomposition digits
delta_c   commitment decomposition digits
delta_f   folded value decomposition digits
delta_r   quotient decomposition digits
```

No digit count is rounded up for physical storage or multilinear addressing.

If a digit count happens to be a power of two, an evaluator may use a simpler internal path. Both paths must have the same physical meaning and must be tested against the same dense oracle.

## Shard ownership

Let `W` be the number of shards for the level. Let `S_g` be a power of two ownership granule for group `g`.

```text
S_g > 0
S_g is a power of two
W * S_g <= F_g
```

The planner may choose `S_g = 1`. This always gives exact balancing when `F_g >= W`.

Split the full granules first.

```text
A = floor(F_g / S_g)       full granules
R = F_g mod S_g            residual live folds
q = floor(A / W)
v = A mod W
```

The first `v` shards receive `q + 1` full granules. The remaining shards receive `q` full granules. The final shard also receives the residual `R`.

```text
F_j = full_granules_j * S_g
F_(W-1) += R
s_0 = 0
s_j = sum of F_i for i < j
```

This produces contiguous ranges. Every internal boundary is aligned to `S_g`. Only the final shard may have a non multiple of `S_g` count. The difference between the largest and smallest shard is at most `S_g` after the larger full granule counts are placed first.

The physical witness contains no shard padding.

The verifier benefit from `S_g` applies only to structured weights. If a local block index is written as:

```text
u = S_g * v + q
```

then a factored evaluator can use a low table of size `S_g` and a high table of size about `F_j / S_g`. Its rough state and evaluation cost is:

```text
delta * (S_g + ceil(F_j / S_g))
```

The balanced choice is near `sqrt(F_j)`. The planner must enumerate nearby powers of two and include load imbalance in the cost. It must not claim an `S_g` fold reduction for flat independent challenges.

## Opening point

Delete `BlockOrder`. Rename the fields of `RingOpeningPoint` so they state the axes.

```rust
pub struct RingOpeningPoint<F> {
    pub position_weights: Vec<F>,
    pub fold_weights: Vec<F>,
}
```

`position_weights` has exactly `L_g` entries.

`fold_weights` contains the exact live prefix of length `F_g`. The Boolean challenge still has `log2(F_cap_g)` high coordinates. Use a prefix equality evaluator instead of materializing and retaining the zero capacity suffix.

`ring_opening_point_from_field` takes the group fold geometry. It splits the point after `log2(L_g)` low coordinates. It rejects any other point length.

Delete the compact to virtual address conversion. Every equality weight uses the compact physical source address.

## Fold challenges

### Shape

Change the current challenge shape to carry the tensor low length.

```rust
pub enum ChallengeShape {
    Flat,
    Tensor { fold_low_len: usize },
}
```

The tensor low length `Q_g` must be a power of two.

```text
Q_g = fold_low_len
H_g = ceil(F_g / Q_g)
f   = Q_g * h + q
```

Only indices with `f < F_g` are live. `H_g` is exact and may be any positive integer. The final high row may use only a prefix of the low factor.

### Names and storage

Rename tensor left and right to fold high and fold low throughout code, transcript comments, tests, and docs.

Repurpose `TensorChallenges` as follows.

```text
fold_high                  C_g * H_g sparse challenges
fold_low                   C_g * Q_g sparse challenges
live_folds_per_claim       F_g
fold_low_len               Q_g
num_claims                 C_g
```

Derive `H_g`. Do not store both `H_g` and values from which it is derived.

`Challenges::num_blocks_per_claim()` returns `F_g`, not `H_g * Q_g`.

`Challenges::logical_len()` returns `C_g * F_g`.

Do not cache or materialize the `H_g * Q_g` products in the runtime challenge object.

### Transcript

The transcript binds the group index, `F_g`, challenge shape, and `Q_g` before challenge sampling.

For a tensor challenge, sample fold high first. Absorb its canonical digest. Then sample fold low. The existing sampling order may remain, but all labels and comments must use high and low names.

Flat sampling draws exactly `C_g * F_g` sparse challenges.

Tensor sampling draws exactly `C_g * (H_g + Q_g)` sparse challenges.

Each group chooses its own shape. A root batch may contain flat and tensor groups together.

### Sparse evaluation

Keep each sparse challenge in support form. For current production configurations, coefficients are small signed integers. The evaluator should use add, subtract, and double operations for coefficients in `{1, -1, 2, -2}` where that is faster than a general multiplication.

Do not form one dense ring product per live fold index for a tensor challenge.

## Succinct digit innermost evaluation

Tight digit storage creates carries because a digit stride need not be a power of two. This is an evaluator problem, not a reason to change physical storage.

For a digit innermost interval with base `b`, digit count `delta`, outer index `i`, and digit `d`, the address is:

```text
A(i, d) = b + delta * i + d
```

The verifier needs sums of this form.

```text
sum over live i and d of
    outer_weight(i) * gadget(d) * eq(r, A(i, d))
```

Keep one production carry kernel for this affine address family. It must support:

* an exact live prefix length;
* an arbitrary checked base offset;
* a tight digit count;
* a factored high and low outer weight;
* one partial final high row;
* Boolean challenges without inversion;
* checked work before allocation;
* no table whose size is the product of all axes.

Use this kernel for tensor fold weights, trace weights, and relation intervals. Delete specialized or test only compact equality evaluators after their useful oracle tests move to the shared kernel.

For tensor challenges, write `i = Q_g * h + q`. The carry induced by `delta * i + d` has at most `delta` digit states. The target cost is proportional to:

```text
delta * (H_g + Q_g)
```

plus small state and the sparse ring support work. The last partial row adds at most one extra low prefix evaluation.

For a flat challenge, reading the independent live challenge values costs at least `F_g`.

## Trace evaluation

Delete `TraceChunkLayout`. It copies ownership geometry from `WitnessLayout`.

Keep `TraceWeightLayout` only for trace specific information that is not already present in group parameters or `WitnessLayout`. Functions that evaluate trace columns must receive the canonical `WitnessLayout` and a group index.

For one shard, the `e_hat` trace interval is:

```text
sum over u < F_j and d < delta_o of
    fold_weight(s_j + u)
    * gadget(d)
    * eq(r, e_start + delta_o * (F_j * claim + u) + d)
```

The `t_hat` interval uses the same address rule with the A row inside the block axis.

The evaluator must handle:

* a nonzero global block start `s_j`;
* an exact local count `F_j`;
* a tight digit count;
* a nonzero physical segment offset;
* a structured fold equality weight;
* an optional tensor fold weight;
* a partial final tensor row.

It must not enumerate all live folds when the fold weight remains factored. It must not claim a succinct cost while calling an enumerative fallback.

## Relation and setup columns

All semantic setup columns use exact live folds and tight digits.

For group `g`:

```text
D role e column:
  [claim][global fold index][opening digit]

B role t column:
  [claim][global fold index][A row][opening digit]

A role z column:
  [position][commit digit]
```

The fold digit belongs to the witness side of the A relation through its gadget weight. It is not a separate A setup column.

The semantic setup matrices do not contain shard copies. A physical relation unit maps its local fold index `u` back to `s_j + u` before selecting a semantic setup column. The extended relation still has one physical occurrence for every copied `z_hat` and every partitioned `e_hat` and `t_hat` range.

For multi group roots, D columns concatenate group `e_hat` columns in relation order. Derive each group start by a checked prefix sum. Do not store a mutable `e_setup_col_offset` in a copied group descriptor.

`SetupProjectionGeometry` computes mixed role projection after these semantic columns are known. The prover and verifier use the same checked object for required footprint, round count, alpha powers, and work limits.

## Multi group roots

Each group owns:

* `N_g`, `L_g`, `F_g`, and derived `F_cap_g`;
* its digit counts;
* its fold challenge shape;
* its tensor low length when tensor mode is active;
* its shard granule;
* its `z_hat`, `e_hat`, and `t_hat` units.

The root relation is a direct sum of group fold relations plus the existing shared terms.

```text
per group:
  fold consistency
  A relation
  B relation

shared:
  one D relation over concat(e_hat_g)
  one r_hat tail
  one next witness commitment
```

`OpeningClaimsLayout` keeps transcript order. Its existing root group order gives relation order. Do not store a second order vector in the witness layout.

The current `reject_multi_group_multi_chunk` check is deleted only when emission, relation construction, setup weights, trace weights, terminal handling, and verifier replay all consume the group by shard product layout.

## Planner

The planner chooses these values independently for every group and active level.

```text
L_g   power of two positions per fold slice
S_g   power of two shard granule
Q_g   power of two tensor low length, when tensor mode is active
```

The planner derives `F_g` from `N_g` and `L_g`. It does not round `F_g` up for physical width formulas.

The cost model includes:

* physical `z_hat`, `e_hat`, `t_hat`, and `r_hat` ring elements;
* replicated `z_hat` for each shard;
* exact shard load counts;
* flat challenge sample and evaluation work;
* tensor sample work `C_g * (H_g + Q_g)`;
* digit carry evaluator work;
* trace evaluator work;
* mixed setup projection work;
* proof bytes and setup widths;
* the selected number of active distributed levels.

The planner must not force `S_g = Q_g`. If both are active, their common low alignment is the smaller power of two. A boundary that cuts a larger low row is handled as an edge interval.

The planner should enumerate nearby powers of two instead of using one fixed root rule. A square root granule is the natural balanced candidate, but it is not always best once shard count, replicated `z_hat`, tensor mode, and exact residual load are priced.

## Transcript and serialization

This is a breaking cutover. Do not add a version branch, old enum variant, conversion shim, or old byte decoder.

Bind all protocol affecting final fields in the schedule or instance descriptor. This includes:

* exact source length;
* position count;
* exact live fold count;
* shard count and active levels;
* group local shard granule;
* group local challenge shape;
* tensor low length;
* digit counts;
* relation row layout;
* mixed role ring dimensions.

Derived values such as `F_cap`, high tensor length, and bit counts need not be serialized twice. The verifier derives them and rejects overflow or inconsistency.

Proof labels and event order remain unchanged unless the high and low rename requires a label version change. If labels change, update prover, verifier, transcript tests, and the transcript hardening spec in the same slice.

## Verifier safety

Every verifier reachable constructor checks arithmetic before allocation.

Malformed geometry returns `AkitaError`. It must not panic, index unchecked, allocate from an unbounded proof value, or select a fallback layout.

Check work products before entering nested loops. A bound must match the actual loop product. Do not use an additive estimate for multiplicative work.

## Canonical dense oracle

One independent test oracle must build the exact physical witness and a dense globally zero padded multilinear table from the formulas in this spec.

The oracle must not call production index methods to compute expected indices.

Compare all of the following with the same cases.

* witness emission;
* relation columns;
* setup A, B, and D columns;
* trace columns;
* terminal bytes;
* recursive handoff;
* verifier relation evaluation;
* flat challenge evaluation;
* tensor challenge evaluation;
* mixed ring projection.

Cases must include:

* `F = 1`;
* a power of two `F`;
* a non power of two `F`;
* a partial last source block;
* digit counts 1, 3, 5, and a power of two;
* tensor `F` smaller than, equal to, and larger than `Q`;
* one partial final tensor row;
* one shard;
* two, four, and eight shards;
* a nonzero shard start;
* a residual shard;
* multiple groups with different `L`, `F`, digits, and challenge shapes;
* mixed A, B, and D ring dimensions;
* malformed geometry and work limit rejection.

## Acceptance criteria

* [ ] This file is the normative design record for PR #294.
* [ ] `BlockOrder`, row major versus column major dispatch, and all old layout branches are deleted.
* [ ] `OpeningBlockLayout` and its virtual address mapping are deleted.
* [ ] Root and recursive folds use power of two `L` and exact live `F`.
* [ ] `LevelParams` and `PrecommittedLevelParams` store the same group local fold meanings.
* [ ] Capacity and bit counts are derived instead of stored twice.
* [ ] `WitnessLayout` is the only physical range authority.
* [ ] `WitnessChunkLengths` and `TraceChunkLayout` are deleted.
* [ ] Every witness unit records a group, a shard, an exact global fold start, an exact live count, and checked `z`, `e`, and `t` ranges.
* [ ] Digits remain tight for `z_hat`, `e_hat`, `t_hat`, and `r_hat`.
* [ ] Tensor challenges use fold high and fold low names.
* [ ] Tensor high length may be non power of two and the final row may be partial.
* [ ] Each group chooses flat or tensor mode independently.
* [ ] Sparse factors remain sparse through verifier evaluation.
* [ ] The production carry kernel matches the dense oracle and does not allocate a Cartesian state table.
* [ ] Structured trace evaluation does not enumerate all live folds.
* [ ] Multi group plus multi shard uses the product layout and no longer takes the old rejection path.
* [ ] One shared `r_hat` tail follows all groups and shards.
* [ ] Setup, relation, trace, terminal, and recursive code consume the same group and shard ranges.
* [ ] Planner width formulas use exact `F`, and Boolean formulas use derived `F_cap`.
* [ ] Schedule and transcript descriptors bind every protocol affecting choice once.
* [ ] Malformed verifier inputs return `AkitaError` without panic.
* [ ] Generated schedules are regenerated with explained drift.
* [ ] Relevant dense oracle and end to end tests pass.
* [ ] `cargo fmt -q` passes.
* [ ] `cargo clippy --all --message-format=short -q -- -D warnings` passes.
* [ ] `cargo test` passes.
* [ ] `rtk cargo nextest run --profile ci --no-default-features --features parallel,disk-persistence` passes.
* [ ] `./scripts/check-doc-guardrails.sh` passes.

## Implementation plan

The slices below are compile checkpoints. Each slice deletes the path it replaces. No slice adds a compatibility layer.

### Slice 1: Parameter geometry and names

Files centered on this slice:

```text
crates/akita-types/src/layout/params.rs
crates/akita-types/src/layout/params/precommitted.rs
crates/akita-types/src/layout/digit_math.rs
crates/akita-planner/src/schedule_params.rs
crates/akita-challenges/src/tensor.rs
```

Steps:

1. Replace stored `m_vars` and `r_vars` with derived methods.
2. Add the exact source length and shard granule to both group parameter forms.
3. Make `with_decomp` derive `F` from `N` and power of two `L`.
4. Change challenge shape to `Tensor { fold_low_len }`.
5. Put challenge shape on each precommitted group.
6. Update descriptor bytes and schedule identity.
7. Update parameter validation and focused unit tests.

### Slice 2: Opening and fold cutover

Files centered on this slice:

```text
crates/akita-types/src/layout/opening_point.rs
crates/akita-prover/src/protocol/core/fold.rs
crates/akita-verifier/src/protocol/core/fold.rs
crates/akita-prover/src/protocol/core/root_fold.rs
crates/akita-verifier/src/protocol/core/root_fold.rs
crates/akita-prover/src/protocol/core/suffix.rs
crates/akita-verifier/src/protocol/core/suffix.rs
```

Steps:

1. Delete `BlockOrder` and all dispatch on it.
2. Rename opening point fields to position and fold weights.
3. Split opening variables at `log2(L)` and evaluate only the live `F` high prefix.
4. Rewrite fold kernels around `source = f * L + p` with a partial last fold slice.
5. Delete `OpeningBlockLayout` and all compact to virtual address conversion.
6. Update root, recursive, and terminal fold parity tests.

### Slice 3: Canonical witness ranges

Files centered on this slice:

```text
crates/akita-types/src/witness.rs
crates/akita-types/src/proof/ring_relation.rs
crates/akita-types/src/proof/tail_segments.rs
crates/akita-types/src/proof/witness_layout_contract.rs
```

Steps:

1. Replace `WitnessChunkLayout` with the group and shard unit record.
2. Delete `WitnessChunkLengths` and parallel vectors.
3. Resolve granule aligned exact shard ranges.
4. Resolve multi group by multi shard units in relation order.
5. Emit each unit as `[z_hat | e_hat | t_hat]` with one final `r_hat`.
6. Move the useful independent oracle tests out of the PR only contract module, then delete that module if it has no production role.
7. Delete `MultiGroupRingRelationSegmentLengths` once `WitnessLayout` computes the ranges directly.

### Slice 4: Challenge runtime and prover kernels

Files centered on this slice:

```text
crates/akita-challenges/src/fold_draw.rs
crates/akita-challenges/src/tensor.rs
crates/akita-prover/src/protocol/core/fold_kernels.rs
crates/akita-prover/src/protocol/ring_switch/coeffs.rs
crates/akita-verifier/src/stages/stage1.rs
```

Steps:

1. Rename left and right fields, labels, and helpers to fold high and fold low.
2. Store live `F` and low `Q`; derive high `H`.
3. Sample exact flat and tensor counts.
4. Support one partial final low row.
5. Remove tensor product materialization from runtime challenge paths.
6. Use sparse add, subtract, and double operations for small signed coefficients.
7. Draw each group with its own shape.

### Slice 5: Relation and setup cutover

Files centered on this slice:

```text
crates/akita-types/src/proof/relation_matrix_cols.rs
crates/akita-types/src/setup_contribution/geometry.rs
crates/akita-types/src/setup_contribution/plan
crates/akita-types/src/setup_contribution/setup_index_weight_evaluator.rs
crates/akita-prover/src/protocol/ring_relation.rs
crates/akita-verifier/src/protocol/ring_switch.rs
```

Steps:

1. Make every relation column index come from the canonical unit range and group parameters.
2. Use exact global fold indices for A, B, and D setup roles.
3. Derive D group offsets by a checked relation order prefix sum.
4. Keep `SetupProjectionGeometry` as the only mixed role projection owner.
5. Remove copied layout carriers and pass through aliases.
6. Compare direct and structured evaluation with the dense oracle.

### Slice 6: Carry and trace evaluator

Files centered on this slice:

```text
crates/akita-algebra/src/offset_eq.rs
crates/akita-types/src/trace_weight
crates/akita-verifier/src/protocol/ring_switch/tensor_challenges.rs
crates/akita-verifier/src/protocol/slice_mle
```

Steps:

1. Consolidate compact equality code into one production affine digit interval kernel.
2. Add exact prefix, base offset, high and low factor, and partial row support.
3. Delete `TraceChunkLayout`.
4. Route trace construction and evaluation through `WitnessLayout` units.
5. Implement distributed tensor subwindows instead of returning the current unsupported error.
6. Delete enumerative fallbacks that claim structured cost.
7. Add work cap boundary tests and dense oracle parity tests.

### Slice 7: Recursive and terminal consumers

Files centered on this slice:

```text
crates/akita-prover/src/backend/recursive/witness.rs
crates/akita-prover/src/backend/recursive/setup_prefix_source.rs
crates/akita-prover/src/backend/poly_helpers/decompose_fold_partitioned.rs
crates/akita-setup/src/recursive_prefixes.rs
crates/akita-config/src/setup_prefix_slots.rs
crates/akita-prover/src/protocol/ring_switch/finalize.rs
crates/akita-types/src/proof/terminal_witness.rs
```

Steps:

1. Make recursive witness construction consume canonical unit ranges.
2. Preserve tight digits and exact live counts at the next handoff.
3. Make setup prefix source and materialization consume the same ranges.
4. Keep direct and terminal folds scalar and use the canonical single group emission path.
5. Delete column major recursive helpers, duplicate terminal index formulas, and the old setup recursion module.
6. Test partial last fold slices across two recursive levels and scalar terminal consumers.

### Slice 8: Planner, schedules, docs, and deletion

Files centered on this slice:

```text
crates/akita-planner
crates/akita-config
crates/akita-schedules
book/src/how/recursion.md
book/src/how/proving/opening-points-block-order.md
book/src/how/verifying/matrix_evaluation.md
docs/block-order.md
specs/multi-group-batching.md
specs/distributed-verifier-row-eval.md
```

Steps:

1. Enumerate `L`, `S`, and `Q` independently.
2. Price exact physical widths, structured verifier work, and shard imbalance.
3. Regenerate every affected schedule table.
4. Delete `docs/block-order.md` and replace the book page with the final geometry.
5. Mark superseded layout sections in older live specs.
6. Rebase PR #296 onto the canonical ownership geometry and remove its conflicting layout types.
7. Run the full verification list and record performance changes.

## References

* [`specs/TEMPLATE.md`](TEMPLATE.md)
* [`specs/SPEC_REVIEW.md`](SPEC_REVIEW.md)
* [`specs/PRUNING.md`](PRUNING.md)
* [`specs/multi-group-batching.md`](multi-group-batching.md)
* [`specs/tensor-structured-folding-challenges.md`](tensor-structured-folding-challenges.md)
* [`specs/distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)
* [`book/src/how/recursion.md`](../book/src/how/recursion.md)
* `crates/akita-types/src/layout/params.rs`
* `crates/akita-types/src/layout/params/precommitted.rs`
* `crates/akita-types/src/witness.rs`
* `crates/akita-types/src/proof/ring_relation.rs`
* `crates/akita-challenges/src/tensor.rs`
* `crates/akita-types/src/trace_weight`

Authorship disclosure: Drafted by Codex assistant on behalf of Quang Dao with approval.
