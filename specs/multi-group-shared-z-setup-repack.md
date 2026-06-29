# Spec: Multi-Group Shared-z Setup Repacking

| Field         | Value                                                |
|---------------|------------------------------------------------------|
| Author(s)     |                                                      |
| Created       | 2026-06-28                                           |
| Status        | proposed                                             |
| PR            |                                                      |
| Supersedes    | `multi-group-batching.md` per-group root `z` model   |
| Book-chapter  | book/src/how/proving/root-fold-ring-switch.md        |

## Summary

This spec replaces the initial multi-group root model's "one `z_hat_g` per
commitment group" relation with one shared folded `z` over a union A-coordinate
system. The setup repacking rule aligns A entries across precommitted and final
commitment groups, then maps B/D/F role entries into the same generated setup
object with a deterministic allocator. The goal is to keep independently created
commitments compatible with a later grouped root proof while reducing the root
witness from `G` z segments to one shared z segment.

## Intent

### Goal

Build a grouped-root setup layout in which every group-local A matrix embeds
into one shared A union, so the root relation can prove:

```text
sum_g sum_i c_{g,i} * t_hat_{g,i} = A_union * z
```

with one folded digit-vector witness:

```text
z = sum_g sum_i c_{g,i} * pad_g(s_hat_{g,i})
```

instead of one `z_g` / `z_hat_g` segment per group.

### Invariants

- A alignment is coordinate-based, not area-based. If two groups both use
  A coordinate `(row, col)`, they must read the exact same setup coefficient.
- The shared `z` domain is the union of group-local `s_hat` coordinates.
  Smaller groups embed by zero-padding into that union.
- Different groups may use different decomposition bases for `s -> s_hat` and
  `t -> t_hat`, but each union coordinate must carry the recomposition weight
  for the group-local digit semantic that created it.
- The matrix relation uses one A union and one z segment. B, D, and F do not
  need semantic alignment with A; they only need deterministic, prover/verifier
  identical setup-index maps.
- Setup-index collisions across different roles are allowed, as in the current
  overlapping-prefix setup design. If several role cells map to the same setup
  index, setup-contribution weights add at that index.
- Within one logical role matrix, cells must map injectively. A single B row,
  D row, F row, or A row must never contain the same setup coefficient twice
  unless a future spec gives a separate soundness argument.
- The concrete setup stride and every role-cell map are verifier-visible layout
  data. They must be bound in the schedule/descriptor/setup identity before
  Fiat-Shamir replay uses them.
- All verifier-reachable malformed layout cases return `AkitaError` or
  `SerializationError`; no unchecked indexing, panic, or allocation from
  untrusted dimensions.

### Non-Goals

- No recursive multi-group root below level 0. The grouped root still emits one
  singleton recursive witness commitment for the suffix.
- No recursive setup-contribution support for `G > 1` unless a later phase
  explicitly generalizes the setup-product evaluator to the shared-z maps.
- No tiered multi-group support in the first implementation. F-role rules are
  specified so the allocator is not painted into a corner.
- No backward compatibility with old grouped proof bytes, descriptor bytes, or
  setup cache artifacts.
- No attempt to make staggered multi-group commitments cheaper than one scalar
  same-point batch when all polynomials are known up front.

## Design

### Terminology

`G`
: Number of commitment groups in the grouped root.

`g`
: Commitment group index in transcript order.

`C_cap`
: Configuration cap for the maximum A-grid column count. This is the user's
  proposed `MAX_COL`.

`C_setup`
: Concrete setup A-grid stride for one setup artifact or one supported setup
  family. It must satisfy `C_setup <= C_cap` and must be bound into setup
  identity. Use `C_setup`, not the loose cap, in setup-index formulas.

`A_g`
: Group-local A matrix with `R_A_g` rows and `C_A_g` columns.

`A_union`
: Shared A-coordinate grid containing all group-local A cells used by the
  grouped root.

`lambda`
: Flat generated setup ring index.

`coord`
: Two-dimensional setup coordinate `(row, col)` before conversion to `lambda`.

### MAX_COL Is a Cap, Not a Loose Flat Stride

The grouped A rule is:

```text
lambda_A(row, col) = row * C_setup + col
```

where:

```text
0 <= col < C_A_g <= C_setup <= C_cap
```

`C_cap` may be exposed by a config/preset as the maximum allowed column count,
but a concrete setup must choose and bind a tight `C_setup`. A loose global
stride wastes setup slots because the flat prefix length of a row-balanced shape
is:

```text
(R_active - 1) * C_setup + C_active
```

not `R_active * C_active`.

Therefore, the implementation must not blindly use `i * C_cap + j` against a
flat setup prefix unless `C_cap` is also the concrete tight stride for that
setup artifact. If a deployment needs one setup to support many future shapes,
the setup envelope must account for this stride cost explicitly.

### A-Union Coordinate System

Each group-local A embeds into the top-left corner of the A union:

```text
A_g(row, col) = S[row * C_setup + col]
```

for:

```text
0 <= row < R_A_g
0 <= col < C_A_g
```

The A-union active set is:

```text
U_A = { (row, col) | exists g such that row < R_A_g and col < C_A_g }
```

The corresponding setup-index set is:

```text
Lambda_A = { row * C_setup + col | (row, col) in U_A }
```

The set may be non-rectangular when one group has more rows and fewer columns
than another. The implementation must keep it as a set of coordinates, not
silently round it to the full rectangle unless the planner chooses to pay that
setup cost.

### Single-z Algebra

For each group and polynomial claim:

```text
t_{g,i} = A_g * s_hat_{g,i}
```

Embed `s_hat_{g,i}` into the A-union column domain:

```text
pad_g(s_hat_{g,i})[row, col] =
  s_hat_{g,i}[row, col] if (row, col) is active for group g
  0                    otherwise
```

Then define:

```text
z = sum_g sum_i c_{g,i} * pad_g(s_hat_{g,i})
```

Linearity gives:

```text
A_union * z
  = A_union * sum_g sum_i c_{g,i} * pad_g(s_hat_{g,i})
  = sum_g sum_i c_{g,i} * A_g * s_hat_{g,i}
  = sum_g sum_i c_{g,i} * t_{g,i}
```

The grouped root can therefore use one z segment whose width is the A-union
coordinate width, rather than `G` group-local z segments.

### Mixed Decomposition Bases

Groups may have different `log_basis` values if the M-matrix tracks the basis
per logical column family.

For `s -> s_hat`, the A-union coordinate stores a digit coordinate, not a raw
coefficient coordinate. If group `g` uses `log_basis = l_g`, the recomposition
weight associated with that group's active `s_hat` digit column is:

```text
2^(digit * l_g)
```

For `t -> t_hat`, each group's T segment uses that group's opening
decomposition basis:

```text
recompose_open_g(t_hat_g) = t_g
```

The root M-matrix must therefore support per-group/per-column recomposition
weights:

```text
T columns for group g use G_open(l_g)
Z/A-union columns contributed by group g use G_commit(l_g)
```

The current code often carries one `lp.log_basis` and derives one
`gadget_row_scalars` table for a whole level. This spec requires replacing that
assumption in grouped-root code paths with a layout-owned recomposition table.

### Setup Role Maps

Introduce a first-class setup layout map:

```text
SetupRoleMap(role, group, role_row, role_col) -> lambda
```

Required roles:

```text
A, B, D
```

Deferred but reserved role:

```text
F
```

The A role is fixed by the A-union rule. B/D/F are assigned by a deterministic
allocator over setup coordinates. Their logical matrix rows and columns remain
the rows and columns used by the Ajtai relation; the setup coordinates are only
physical storage coordinates.

The map must be derivable from schedule/config metadata. It must not depend on
prover hints or private witness data.

### Deterministic Physical Allocator

The allocator maintains a global set of used setup indices:

```text
Used = Lambda_A plus every B/D/F index already assigned
```

For a role matrix needing `N = rows * cols` logical cells, the allocator produces
`N` distinct setup indices. The default selection order is:

1. Reuse already-used setup indices first, in deterministic balanced-coordinate
   order, skipping any index already used within this same logical matrix.
2. Allocate new setup indices, also in deterministic balanced-coordinate order.
3. Assign the selected indices to the role matrix in logical row-major order.

Balanced-coordinate order is column-major over the active setup rows:

```text
(row = 0, col = 0), (row = 1, col = 0), ..., (row = R_active - 1, col = 0),
(row = 0, col = 1), (row = 1, col = 1), ...
```

with coordinates converted to:

```text
lambda = row * C_setup + col
```

This intentionally differs from smallest flat-prefix order. It avoids assigning
all extra coefficients to row 0 when a role needs more cells than the current A
footprint. For example, if the A grid has `R_active = 5` and a role needs ten
extra cells beyond the A footprint, the new cells are two extra columns across
each of the five active rows, not ten cells in the first row.

The planner objective is the smallest setup footprint compatible with this
balanced coordinate policy:

```text
N_setup = 1 + max(lambda used by any role map)
```

Because `N_setup` depends on `C_setup`, `R_active`, and the highest active
column, the planner should choose the smallest valid `C_setup` for the supported
shape family rather than a loose cap.

### Precommitted Group Layout

At standalone precommit time, the group freezes:

```text
CommitmentGroupLayout {
  key,
  m_vars,
  r_vars,
  log_basis,
  n_a,
  conservative_n_b,
  a_cols,
  a_setup_stride: C_setup,
  role_maps_digest,
}
```

The exact struct fields may differ, but the descriptor-bound metadata must let
the final planner and verifier reconstruct:

- the group-local A dimensions;
- the group-local B dimensions and conservative B row count;
- the group-local decomposition bases;
- the setup role maps used by the commitment;
- the setup stride used for A alignment.

If `C_setup` is setup-artifact-wide rather than group-local, the layout may store
a setup-layout version/digest instead of repeating the value.

### Final/Main Group Layout

When committing the final group with precommitted groups:

1. Reconstruct every precommitted setup role map.
2. Build `U_A` from all precommitted A dimensions and the final group's A
   dimensions.
3. Require the final group's A map to agree with every precommitted group on the
   intersection:
   ```text
   if row < R_A_g and col < C_A_g and row < R_A_final and col < C_A_final:
       lambda_A_g(row, col) == lambda_A_final(row, col)
   ```
4. Seed `Used` with every setup index used by every precommitted role map, not
   only their A maps.
5. Assign any final-group A cells by the A-union rule.
6. Assign final-group B/D/F cells with the deterministic allocator.

This rule lets the final group reuse coefficients already consumed by
precommitted B/D/F maps where that reuse is layout-compatible, while still
guaranteeing exact A alignment on the shared z relation.

### Root Witness Layout

The grouped root witness changes from:

```text
e_hat_0 || ... || e_hat_{G-1}
t_hat_0 || ... || t_hat_{G-1}
z_hat_0 || ... || z_hat_{G-1}
r_tail
```

to:

```text
e_hat_0 || ... || e_hat_{G-1}
t_hat_0 || ... || t_hat_{G-1}
z_shared
r_tail
```

where `z_shared` is over the A-union coordinate domain.

The root still keeps group-local:

- commitment rows `u_g`;
- public output rows;
- T segments;
- opening-side W/E segments;
- B rows.

Only the A/z side is merged.

### M-Row Layout

The non-tiered grouped root rows become:

```text
consistency
public rows for each group
D rows for concat(w_hat_g)
COMMIT/B rows for each group
A_union rows
```

The old multi-group model had one A block per group. This spec replaces those
with one A-union row block. A-union row `a` receives the sum of all group-local
A contributions that use row `a`, with inactive group columns contributing zero.

The row layout helpers must derive offsets from a grouped-root layout object.
Verifier-reachable code must not hardcode `num_segments = 1`, `num_z_vectors = G`,
or `A rows = sum_g n_a_g` for shared-z grouped roots.

### Setup Contribution

The direct setup-contribution evaluator must operate over `SetupRoleMap`, not
over role-local prefix rectangles.

For each setup index:

```text
omega_bar(lambda) =
  sum over all role cells mapped to lambda of role_cell_weight
```

This is the same addition rule as the existing overlapping-prefix setup, but the
pullback is now an explicit map rather than:

```text
role_row * role_width + role_col
```

The materialized direct evaluator is the correctness oracle. Recursive
setup-product offloading for `G > 1` remains out of scope until the succinct
weight evaluator can evaluate this mapped `omega_bar` without scanning the full
setup.

### Descriptor and Transcript Binding

The instance descriptor must bind:

- group count and group order;
- group sizes and opening routing;
- every frozen precommitted `CommitmentGroupLayout`;
- final grouped root schedule digest;
- setup seed/digest;
- setup layout version;
- `C_setup` and `C_cap`;
- A-union dimensions or digest;
- every role map digest, or enough canonical metadata to reconstruct the maps;
- per-group recomposition basis tables or the canonical data used to derive them;
- the fact that the root uses one shared z segment.

Changing any of these fields must change the transcript.

## Evaluation

### Acceptance Criteria

- [ ] A grouped root with two groups produces one shared z segment, not one
      z segment per group.
- [ ] If two groups overlap in A coordinate `(row, col)`, prover and verifier
      read the same setup index for both groups.
- [ ] Smaller groups zero-pad into the A-union z domain and verify against the
      same `A_union * z` relation.
- [ ] Groups with different `log_basis` values verify when their group-local
      recomposition weights are used in T and Z columns.
- [ ] Tampering with `C_setup`, A-union dimensions, role-map metadata, or a
      precommitted group layout changes descriptor bytes or causes verification
      rejection.
- [ ] Setup-contribution direct evaluation matches a slow materialized
      `omega_bar(lambda)` oracle over the explicit role maps.
- [ ] Setup envelope sizing reports the true `N_setup = 1 + max(lambda)`, not
      only the number of active logical cells.
- [ ] Unsupported recursive setup contribution with `G > 1` rejects clearly
      until the mapped weight evaluator is implemented.

### Testing Strategy

Unit tests:

- Build two A shapes where neither contains the other, such as `(rows=4, cols=8)`
  and `(rows=6, cols=5)`, and assert the A union is non-rectangular.
- Assert overlapping A coordinates map to identical setup indices.
- Assert B/D allocator output is injective per logical matrix.
- Assert balanced allocation distributes ten extra cells across five active rows
  as two extra columns per row.
- Assert the setup length uses `1 + max(lambda)` and catches loose `C_setup`
  inflation in a regression fixture.
- Assert descriptor bytes change when `C_setup`, role-map order, or group order
  changes.

Protocol tests:

- Two-group one-hot same-point proof with equal group shapes and one shared z.
- Two-group one-hot same-point proof with unequal A widths and zero-padding.
- Two-group proof where precommitted and final groups have different
  `log_basis` values, if the config permits the shape.
- Negative test: use the same proof with one role-map entry permuted and assert
  verification rejects.
- Negative test: present a per-group-z descriptor to the shared-z verifier and
  assert descriptor parsing or schedule validation rejects.

Setup-contribution tests:

- Compare direct mapped setup contribution against a naive sum over role cells.
- Compare materialized `omega_bar` against direct evaluation when A/B/D role
  cells collide at the same setup index.
- Keep recursive setup contribution rejected for `G > 1` until a later spec
  updates the succinct evaluator.

### Performance

Expected wins:

- root witness size drops by removing `G - 1` z segments;
- A rows drop from `sum_g n_a_g` to `rows(A_union)`;
- setup may grow relative to prefix repacking when A alignment forces a wider
  coordinate grid.

Expected costs:

- non-contiguous role maps require gather/scatter-aware matvec kernels or
  temporary packed role buffers;
- direct setup contribution must aggregate role-cell weights through the map;
- descriptor/schedule metadata grows by role-map or role-map-digest fields.

Benchmarks should report:

```text
group sizes
A-union rows
A-union active columns
C_setup
N_setup
z_shared rings
saved z rings versus per-group-z baseline
role-map materialization time
B/D/F gather time
```

## Alternatives Considered

### Keep one z per group

This is the initial `multi-group-batching.md` model. It is simpler because every
group relation stays local, but it pays one z segment and one A block per group.
It misses the main benefit of A alignment.

### Require identical group layouts

If every group has the same A shape and basis, sharing z is straightforward.
This is too restrictive for staggered commitments, where precommitted groups may
have smaller block lengths, different `K_g`, or different selected ranks.

### Use a loose global MAX_COL directly

Using:

```text
lambda = row * MAX_COL + col
```

with a loose cap makes row-balanced allocation expensive because holes become
real flat-prefix setup slots. The spec instead treats `MAX_COL` as a cap and
requires a concrete tight `C_setup` to be bound into setup identity.

### Fill the smallest flat-prefix holes first

This minimizes `max(lambda)` for a fixed loose stride, but it concentrates extra
cells in the first physical row and produces jagged role maps. The chosen
balanced-coordinate allocator better matches the intended row/column geometry
and makes the setup footprint tradeoff explicit.

## Documentation

If implemented, fold the durable protocol shape into
`book/src/how/proving/root-fold-ring-switch.md` and update
`book/src/how/configuration.md` for any new config surface around `MAX_COL` /
`C_setup`.

The existing `specs/multi-group-batching.md` should be updated or superseded for
the shared-z root model before implementation lands, because it currently states
that grouped roots have one `z_hat_g` per group.

## Execution

Suggested implementation slices:

1. Add explicit setup role-map types and digest/canonical serialization helpers.
2. Add A-union construction and balanced allocator tests.
3. Extend grouped root schedule/layout metadata to describe shared-z roots.
4. Change root witness sizing from `G` z segments to one A-union z segment.
5. Cut prover A/z relation generation to use `A_union` and zero-padded group
   contributions.
6. Cut verifier ring-switch replay to use the same A-union and per-group
   recomposition weights.
7. Rewrite direct setup contribution over `SetupRoleMap` and pin it against a
   naive materialized oracle.
8. Bind setup layout and role-map metadata in descriptors/transcripts.
9. Add end-to-end grouped proof tests and negative descriptor tampering tests.

## References

- `specs/multi-group-batching.md`
- `specs/setup-layout-repack.md`
- `crates/akita-types/src/setup_contribution.rs`
- `crates/akita-types/src/layout/flat_matrix.rs`
- `crates/akita-planner/src/group_batch.rs`
