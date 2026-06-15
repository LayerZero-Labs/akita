# Akita `commit()` Research

This note traces the current `commit()` implementation starting at
`AkitaCommitmentScheme::commit` in `crates/akita-pcs/src/scheme/mod.rs` and
following the prover-side code path through `crates/akita-prover`.

Scope:

- Main focus: large-field root commitments for dense and one-hot polynomial
  representations.
- Included because it shares the exact same outer machinery: `batched_commit`
  and recursive `commit_w`.
- Out of scope for this document: root tensor projection, small-field-specific
  protocol motivation, and verifier details beyond what is needed to understand
  the commitment object and hint.

## High-Level Shape

At the algebra level, a root commitment group is:

```text
for each polynomial p_i:
    s_i      = committed root witness, blocked into num_blocks blocks
    t_i[b]   = A * s_i[b]                         for each block b
    t_hat_i = decompose(t_i, num_digits_open)

u = B * concat_i(t_hat_i)
```

When the layout is tiered, the final line becomes:

```text
u_j     = B' * slice_j(concat_i(t_hat_i))          for j in 0..f  (f = tier_split)
u_hat   = decompose(u_0 || ... || u_{f-1})
u_final = F * u_hat
```

The returned commitment is `RingCommitment { u }`, where `u` is either the
single-tier `B` image or the tiered `F` image. The returned hint is
`AkitaCommitmentHint`, which stores all `t_hat_i` values and, currently, also
stores recomposed `t_i` rows for later prover use.

The core messiness today is that the A-side work has partially migrated to
operation-shaped backend methods (`dense_commit_rows`, `onehot_commit_rows`,
`sparse_ring_commit_rows`, `recursive_witness_commit_rows`), while the B/F-side
work still uses the lower-level `digit_rows` API directly from the generic
commitment function. Dense and one-hot also expose different input plans because
they avoid materializing the same logical vector in different ways.

## Entrypoint Trace

The public scheme method is deliberately thin:

```text
AkitaCommitmentScheme<D, Cfg>::commit(setup, backend, prepared, polys)
    -> akita_prover::commit::<Cfg, D, P, B>(
           polys,
           setup.expanded.as_ref(),
           backend,
           prepared,
       )
```

The scheme layer contributes:

- the static ring dimension `D`;
- the config type `Cfg`;
- the prover setup's expanded shared matrix;
- the compute backend and its prepared setup;
- the polynomial slice `polys`.

Everything interesting happens in `crates/akita-prover/src/api/commitment.rs`.

## `akita_prover::commit`

`akita_prover::commit` does five scoped things before actual matrix work.

1. It validates that the prepared backend state belongs to the provided expanded
   setup:

```text
backend.validate_prepared_setup::<D>(prepared, expanded)
```

For the CPU backend this compares the setup seed identity. It is intentionally
cheap and avoids re-hashing the matrix.

2. It validates the singleton commitment input with `prepare_commit_inputs`:

- `polys` must be nonempty.
- Every polynomial in the commitment group must have the same `num_vars`.
- The number of polynomials must not exceed
  `setup.seed.max_num_batched_polys`.
- `num_vars` must not exceed `setup.seed.max_num_vars`.
- The result is `OpeningBatch::same_point(num_vars, polys.len())`.

This is "singleton" only in the opening-point sense: one commitment group may
contain many polynomials.

3. It resolves root commitment parameters:

```text
let params = Cfg::get_params_for_batched_commitment(&opening_batch)?;
```

`CommitmentConfig::get_params_for_batched_commitment` resolves the runtime
schedule and returns the root step's `LevelParams`. If the root schedule is a
fold, it returns the fold step's params. If the root schedule is direct, it
returns the direct step's commit params.

The relevant fields of `LevelParams` for commit are:

- `ring_dimension`: must equal static `D`.
- `log_basis`: power-of-two gadget basis.
- `a_key`: inner Ajtai matrix `A`; `row_len()` is `n_a`, and
  `col_len()` must equal `block_len * num_digits_commit`.
- `b_key`: outer matrix `B`, or first-tier `B'` if tiered.
- `f_key`: optional second-tier matrix `F`; present iff the commit is tiered.
- `num_blocks`: number of root witness blocks.
- `block_len`: ring elements per root block.
- `num_digits_commit`: decomposition depth for the original committed witness
  before multiplying by `A`.
- `num_digits_open`: decomposition depth for `t_i = A*s_i`, before multiplying
  by `B`.
- `onehot_chunk_size`: expected one-hot chunk size for one-hot layouts.
- `tier_split`: number of first-tier `B'` slices when `f_key` is present.

4. It validates one-hot shape if the polynomial exposes one:

```text
validate_onehot_chunk_size_for_params(polys, &params)
```

This only rejects a mismatch when `params.onehot_chunk_size > 1` and the
polynomial reports `Some(actual)`. Dense polynomials report `None`.

5. It validates the resolved layout against the setup with
`validate_commit_level_params`:

```text
validate_commit_level_params::<F, D>(&params, expanded)
```

This checks `params.ring_dimension == D`, nonzero `num_blocks` / `block_len` /
digit depths, an i8-range `log_basis`, that `a_key.col_len()` equals
`block_len * num_digits_commit`, that `b_key.col_len()` is nonzero, and that the
`A` / `B` / `D` footprints fit in the prepared setup. (The root tensor-projection
transform decision runs between steps 4 and 5 but is out of scope for this
document.)

## `commit_with_validated_params`

This is the center of the current root commit implementation.

It first computes the flat `t_hat` size for one polynomial, measured in
`[i8; D]` digit planes:

```text
b_input_len_per_poly =
    params.num_blocks * params.a_key.row_len() * params.num_digits_open
```

Yes: `b_input_len_per_poly` is exactly the length of `t_hat_i` for one
polynomial `p_i`. It is not bytes, not field coefficients, and not the full
batch length. It is the number of ring digit planes produced after computing
the inner rows `t_i = A * s_i` and decomposing each row for opening.

Conceptually, for one polynomial:

```text
for each block:
    for each A row:
        for each opening digit:
            one [i8; D] digit plane
```

So:

```text
t_hat_i.flat_digits().len() == b_input_len_per_poly
total_b_input_len == polys.len() * b_input_len_per_poly
```

Then it allocates:

- `b_input_digits`: one flat `Vec<[i8; D]>` for all polynomials in the
  commitment group;
- `decomposed_inner_rows`: one `FlatDigitBlocks<D>` per polynomial;
- `recomposed_inner_rows`: one `Vec<Vec<CyclotomicRing<F, D>>>` per polynomial,
  grouped as `[poly][block][A row]`.

For every polynomial, in parallel when enabled, it calls:

```text
poly.commit_inner(
    backend,
    prepared,
    params.a_key.row_len(),
    params.block_len,
    params.num_blocks,
    params.num_digits_commit,
    params.num_digits_open,
    params.log_basis,
)
```

The return type is:

```text
CommitInnerWitness {
    recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    decomposed_inner_rows: FlatDigitBlocks<D>,
}
```

`validate_commit_inner_shape` then enforces:

- block count equals `params.num_blocks`;
- every recomposed block has exactly `n_a` rows;
- every decomposed block has exactly `n_a * num_digits_open` digit planes;
- total flat digit count matches `num_blocks * n_a * num_digits_open`;
- every decomposed row recomposes to the corresponding recomposed row via
  `CyclotomicRing::gadget_recompose_pow2_i8`.

The flat decomposed digits for that polynomial are copied into
`b_input_digits`. This makes `b_input_digits` the canonical input to the outer
commit matrix.

### Single-Tier Outer Commit

When `params.f_key` is absent:

```text
u = backend.digit_rows(prepared, params.b_key.row_len(), &b_input_digits, log_basis)
```

This computes a negacyclic digit matrix-vector product against the shared setup
matrix prefix. The row count is `params.b_key.row_len()`, but the column count
passed to the kernel is `b_input_digits.len()`, not `params.b_key.col_len()`.

That distinction matters for a future matrix API. `b_key.col_len()` is part of
the schedule/layout metadata and setup footprint validation, while the actual
logical input width at this call site is the flat length of all inner-opening
digits in the commitment group.

With the `zk` feature, the code also samples B-blinding digits, computes
`backend.zk_b_digit_rows`, and adds those rows into `u`.

The final single-tier commitment shape is:

```text
RingCommitment { u: Vec<CyclotomicRing<F, D>> }  // len == params.b_key.row_len()
```

### Tiered Outer Commit

When `params.f_key` is present, `commit_with_validated_params` calls
`tiered_commit_u_final`.

`tiered_commit_u_final` interprets `params.b_key` as the smaller first-tier
matrix `B'`. It requires:

```text
b_input_digits.len() % params.b_key.col_len() == 0
```

Then it does:

```text
for chunk in b_input_digits.chunks(params.b_key.col_len()):
    u_j = backend.digit_rows(prepared, params.b_key.row_len(), chunk, log_basis)

u_concat = u_0 || ... || u_{f-1}
u_hat    = decompose(u_concat, num_digits_open)
u_final  = backend.digit_rows(prepared, f_key.row_len(), &u_hat, log_basis)
```

The final sent commitment has length:

```text
params.effective_commit_rows() == f_key.row_len()
```

Tiered commits deliberately do not add ZK blinding at the F tier in the current
code. Comments say tiered proofs are exercised non-zk.

### Hint Construction

The hint stores:

- per-polynomial `decomposed_inner_rows`, i.e. all `t_hat_i`;
- optional recomposed rows, currently always supplied by this path;
- ZK B-blinding digit streams under `feature = "zk"`.

The prover later needs these hints for ring-relation and ring-switch work. The
commitment object itself only carries `u`.

## `batched_commit`

`AkitaCommitmentScheme::batched_commit` calls
`akita_prover::batched_commit`. It is structurally the same as `commit`, but
the input is:

```text
polys_per_commitment_group: &[&[P]]
```

Each slice is one commitment group. There is still one shared opening point
shape in the resulting `OpeningBatch`.

Important differences:

- `prepare_batched_commit_inputs` allows different natural arities inside the
  batch and uses the maximum arity as the padded root domain.
- It records the number of polynomials in each commitment group.
- It validates the total polynomial count against setup capacity.
- It resolves one shared root `LevelParams`.
- It calls `commit_with_validated_params` once per group.

So the current singleton `commit` path is really the one-group special case of
`batched_commit`.

## Other Commit API Variants

There are two explicit-params variants:

```text
commit_with_params(polys, expanded, backend, prepared, params)
batched_commit_with_params(groups, expanded, backend, prepared, params)
```

These skip config schedule resolution. The caller supplies `LevelParams`, then
the function validates setup/input/layout and calls `commit_with_validated_params`.

That means `commit_with_validated_params` is the real shared execution kernel
for root commitments. The config-driven `commit` and `batched_commit` wrappers
are responsible for deriving the right root layout first.

There is also an enum adapter:

```text
MultilinearPolynomial::{Dense, OneHot, Witness}
```

Its `AkitaPolyOps::commit_inner` implementation simply dispatches to the
contained representation. This adds another visible branch in call graphs, but
does not add another matrix algorithm by itself. The matrix algorithm is still
selected by the inner concrete type.

Finally, `commit_zk_hiding_witness` repeats the same dense inner-commit plus
single-tier B commit pattern for a ZK hiding witness:

```text
DensePoly::from_field_evals(...)
poly.commit_inner(...)
B * t_hat
add zk_b_digit_rows(...)
```

It is feature-specific and not the public root `commit()` path, but it is
another example of the same pipeline being open-coded around `commit_inner` and
`digit_rows`.

## Current Compute Backend Boundary

The current backend split lives in `crates/akita-prover/src/compute.rs`.

There are two layers relevant to commit:

```text
DigitRowsComputeBackend
    digit_rows(...)
    zk_b_digit_rows(...)
    zk_d_digit_rows(...)

CommitmentComputeBackend: DigitRowsComputeBackend
    dense_commit_rows(...)
    onehot_commit_rows(...)
    sparse_ring_commit_rows(...)
    recursive_witness_commit_rows(...)
```

The A-side commit work uses `CommitmentComputeBackend` plans. The B/F-side work
still uses `DigitRowsComputeBackend::digit_rows` directly.

Existing A-side plans:

- `DenseCommitRowsPlan`
  - `CachedDigits { digit_block_slices, log_basis }`
  - `CoeffBlocks { block_slices, num_digits_commit, log_basis }`
- `OneHotCommitRowsPlan`
  - `n_a`
  - `block_len`
  - `num_digits_commit`
  - `OneHotCommitBlocks::{SingleChunk, MultiChunk}`
- `RecursiveWitnessCommitRowsPlan`
  - used by recursive `commit_w`, not the initial user-facing root commit

Other backend plans exist, but the dense/one-hot root refactor does not need
them in scope yet.

This is already close to a matrix backend abstraction, but it is
operation-shaped rather than algebra-shaped. It says "compute dense commit
rows" rather than "multiply this matrix view by this vector view under this
digit representation."

## CPU Prepared Setup

`CpuBackend::prepare_expanded` builds:

- `ntt_shared`: an NTT-prepared view of the shared public setup matrix;
- `ntt_i8_capacity`: CRT/NTT capacity metadata;
- optional lazy prepared ZK B/D slots.

The shared setup matrix is physically one flat matrix. Different protocol
matrices (`A`, `B`, `D`, `F`) are read as prefixes or shaped views of that same
matrix. For example:

- one-hot A-side explicitly requests a ring view of
  `n_a x (block_len * num_digits_commit)`;
- B/F calls pass a row count and use the input length as the column count;
- validation checks that the requested footprint fits in the prepared setup.

This prefix-sharing is important for any future matrix abstraction. A clean API
probably needs an explicit matrix handle or shape, not just `row_len` plus
implicit `digits.len()`.

## Dense Commit Path

### Dense Representation

`DensePoly<F, D>` stores:

- `num_vars`;
- `coeffs: Vec<CyclotomicRing<F, D>>`;
- `small_i8_coeffs`, used by folding paths;
- `digit_cache: OnceLock<DenseDigitCache<D>>`.

`DensePoly::from_field_evals` packs field evaluations into ring coefficients.
The first `log2(D)` variables become coefficients inside a ring, and the
remaining variables index ring elements.

### Dense Inner Commit

The dense `AkitaPolyOps::commit_inner` implementation is:

```text
t = self.commit_rows(backend, prepared, n_a, block_len, num_digits_commit, log_basis)
t_hat = decompose_commit_rows(t, num_digits_open, log_basis)
return CommitInnerWitness { recomposed_inner_rows: t, decomposed_inner_rows: t_hat }
```

So dense commit does exactly:

```text
for each block b:
    s_b_hat = decompose(s_b, num_digits_commit)
    t_b     = A * s_b_hat
    t_b_hat = decompose(t_b, num_digits_open)
```

### Dense `commit_rows`

`DensePoly::commit_rows` has two input modes.

#### Cached Digit Planes

It first calls:

```text
self.digit_planes_for(num_digits_commit, log_basis)
```

This decomposes every ring coefficient into `num_digits_commit` balanced
`[i8; D]` planes and stores them in a `OnceLock` cache keyed by
`(num_digits_commit, log_basis)`.

If the cache matches the current request, `commit_rows` builds
`digit_block_slices`, one slice per root block, and calls:

```text
backend.dense_commit_rows(
    DenseCommitRowsPlan {
        n_a,
        input: CachedDigits { digit_block_slices, log_basis },
    }
)
```

The CPU backend handles this with:

```text
mat_vec_mul_ntt_dense_digits_i8_trusted(
    &prepared.ntt_shared,
    n_a,
    row_width,
    &digit_block_slices,
    log_basis,
)
```

This avoids decomposing during the matvec and avoids rescanning trusted cached
digits for bounds on each call. It also uses the dense variant that skips
full-plane zero checks.

#### Coefficient Blocks

If the dense digit cache is already initialized for a different
`(num_digits_commit, log_basis)` pair, the cache cannot be reused. The fallback
builds `block_slices: Vec<&[CyclotomicRing<F, D>]>` and calls:

```text
backend.dense_commit_rows(
    DenseCommitRowsPlan {
        n_a,
        input: CoeffBlocks {
            block_slices,
            num_digits_commit,
            log_basis,
        },
    }
)
```

The CPU backend computes:

```text
row_width = block_len * num_digits_commit
```

then chooses:

- `mat_vec_mul_ntt_i8_dense_single_row` if `n_a == 1`;
- `mat_vec_mul_ntt_i8_dense` otherwise.

These kernels decompose ring coefficients to i8 digits inside the matrix-vector
driver, tile matrix columns for cache locality, and multiply in the NTT domain.

### Dense Output Shape

For a well-formed root layout:

```text
t.len() == params.num_blocks
t[b].len() == n_a
t_hat.block_count() == params.num_blocks
t_hat.block_sizes()[b] == n_a * num_digits_open
```

The dense code itself derives `num_blocks` from `self.coeffs.len().div_ceil(block_len)`.
The generic commit wrapper then validates it against scheduled
`params.num_blocks`.

## One-Hot Commit Path

### One-Hot Representation

`OneHotPoly<F, D, I>` stores a sparse witness with at most one nonzero field
element per chunk of size `onehot_k`.

Fields:

- `num_vars`;
- `onehot_k`;
- `indices: Vec<Option<I>>`, where `None` means the whole chunk is zero and
  `Some(idx)` means the chunk-local hot position is `idx`;
- `total_ring_elems`;
- `block_cache: OnceLock<(usize, OneHotBlocks)>`.

Construction validates:

- `onehot_k != 0`;
- `onehot_k` and `D` are "nicely matched": one divides the other;
- total field elements are a power of two;
- total field elements are divisible by `D`;
- every hot index is in range.

`AkitaPolyOps::onehot_chunk_size` returns `Some(onehot_k)`, allowing the
config/layout validator to reject mismatched one-hot schedules.

### One-Hot Block Cache

`blocks_for(block_len)` lazily builds and caches the one-hot block table. It
rejects:

- zero or non-power-of-two `block_len`;
- `total_ring_elems % block_len != 0`;
- reuse of the same `OneHotPoly` with a different `block_len`.

The cache is one-layout-only. This is important because `commit`, `prove`, and
folding operations on the same polynomial are expected to share one root layout.

There are two block formats.

#### `SingleChunkEntry`

Used when:

```text
onehot_k >= D && onehot_k % D == 0
```

Here each one-hot chunk spans one or more ring elements, and each ring element
can contain at most one hot coefficient.

Each entry stores:

- `pos_in_block`: which ring element within the current block is nonzero;
- `coeff_idx`: which coefficient of that ring is hot.

Zero rings are omitted.

#### `MultiChunkEntry`

Used when:

```text
onehot_k < D && D % onehot_k == 0
```

Here one ring element contains multiple whole one-hot chunks, so one ring can
contain several hot coefficients.

Each entry stores:

- `pos_in_block`;
- `nonzero_coeffs: Vec<u16>`, the hot coefficient indices inside that ring.

Again, all-zero rings are omitted.

### One-Hot Inner Commit

The one-hot `AkitaPolyOps::commit_inner` implementation is:

```text
blocks = self.blocks_for(block_len)
t = backend.onehot_commit_rows(
    OneHotCommitRowsPlan {
        n_a,
        block_len,
        num_digits_commit,
        blocks: blocks.commit_plan_blocks(),
    }
)
t_hat = decompose(t, num_digits_open, log_basis)
return CommitInnerWitness { recomposed_inner_rows: t, decomposed_inner_rows: t_hat }
```

Conceptually, one-hot avoids materializing `s_hat`. Since the committed witness
has coefficients in `{0, 1}` and the one-hot commit bound is represented by the
first digit plane, the A-side multiplication can be implemented as sparse
column selection:

```text
t_b[row] += A[row][pos_in_block * num_digits_commit] * X^coeff_idx
```

For `MultiChunkEntry`, the same A column is shifted once per hot coefficient in
the ring.

### CPU One-Hot Commit Rows

The CPU backend converts the plan to an A matrix view:

```text
active_a_cols = block_len * num_digits_commit
a_view = shared_matrix.ring_view::<D>(n_a, active_a_cols)
```

Then it dispatches to:

```text
column_sweep_ajtai_onehot::<SingleChunkEntry, F, D>(...)
column_sweep_ajtai_onehot::<MultiChunkEntry, F, D>(...)
```

`column_sweep_ajtai_onehot` has three subpaths.

#### Safety Fallback

If any block's total shift-accumulation count exceeds
`MAX_WIDE_SHIFT_ACCUMULATIONS`, it uses:

```text
inner_ajtai_wide_onehot_safe
```

This flushes wide accumulators at entry boundaries before they overflow.

#### Small-Block Fast Path

If each thread gets at most `SWEEP_THRESHOLD` blocks, it runs each block
independently:

```text
inner_ajtai_wide_onehot
```

This loops over entries, over A rows, and over hot coefficient positions. It
loads the relevant A column, shifts it by `coeff_idx`, accumulates into a wide
ring accumulator, then reduces once at the end.

The math is:

```text
for entry in block:
    col = entry.pos_in_block * num_digits_commit
    for row in A rows:
        for coeff_idx in entry.coeffs():
            t[row] += A[row][col] * X^coeff_idx
```

#### Column-Sweep Path

For larger block batches, it runs `column_sweep_core`:

1. Partition blocks across Rayon threads.
2. Within each thread, process L2-sized block tiles.
3. Convert entries to flat tuples:

```text
(A column, local block index, coefficient index)
```

4. Sort by A column.
5. For each A row, load each touched A column once and scatter its shifted
   contribution into all local block accumulators.
6. Reduce wide accumulators to `CyclotomicRing<F, D>`.

This is a matrix-access optimization. Algebraically it is still `t_b = A*s_b`,
but the input vector is represented as sparse shifted monomials rather than
materialized digit planes.

### One-Hot Decomposition of `t`

After the A-side rows are computed, one-hot builds:

```text
FlatDigitBlocks::zeroed(vec![n_a * num_digits_open; t.len()])
```

Then it decomposes each nonzero `t_i` block:

```text
decompose_rows_i8_into(t_i, dst, num_digits_open, log_basis)
```

It skips decomposition for all-zero blocks because the destination block is
already zeroed.

The generic `validate_commit_inner_shape` still recomposes and verifies every
decomposed row against the recomposed `t`.

The scoped dense/one-hot A-side commit inputs are:

- dense coefficients;
- cached dense digit planes;
- one-hot entries.

## Recursive `commit_w` Comparison

`commit_w` in `crates/akita-prover/src/protocol/ring_switch/commit.rs` is not
the public root `commit()`, but it repeats the same structure:

```text
w_view.commit_inner(...)
validate_commit_inner_shape(...)
outer_input = inner.decomposed_inner_rows.flat_digits()
u = B * outer_input
or tiered u_final = F * decompose(blockdiag(B') * outer_input)
hint = singleton hint
```

The difference is that the polynomial representation is `SuffixWitness`, a
D-specific view over a flat recursive i8 witness. Its A-side plan is
`RecursiveWitnessCommitRowsPlan`.

This duplication is a useful target for refactoring. The outer B/F pipeline is
nearly identical between root `commit_with_validated_params` and recursive
`commit_w`; the main difference is whether there are multiple root polynomials
in one commitment group or a singleton recursive witness.

## Matrix Work Inventory

Current matrix/vector operations in and around `commit()`:

### A-Side Inner Commit

Purpose:

```text
t_b = A * s_b
```

Output shape:

```text
Vec<Vec<CyclotomicRing<F, D>>>  // [block][A row]
```

Current input variants:

- dense predecomposed digit blocks;
- dense coefficient blocks needing decomposition;
- one-hot single-chunk sparse entries;
- one-hot multi-chunk sparse entries;
- recursive witness i8 rows.

Current backend methods:

- `dense_commit_rows`
- `onehot_commit_rows`
- `recursive_witness_commit_rows`

### Decompose Inner Rows

Purpose:

```text
t_hat_b = decompose(t_b, num_digits_open, log_basis)
```

Output shape:

```text
FlatDigitBlocks<D>  // [block][A row][opening digit]
```

Current implementation location varies:

- dense has local `decompose_commit_rows`;
- one-hot inlines similar zeroed/decompose logic;
- sparse-ring has its own local helper;
- recursive witness has similar logic.

This is not matrix multiplication, but it is a repeated computing unit in the
commit pipeline.

### B-Side Outer Commit

Purpose:

```text
u = B * concat(t_hat_i)
```

Output shape:

```text
Vec<CyclotomicRing<F, D>>  // B rows
```

Current backend method:

- `digit_rows`

Current caller:

- root `commit_with_validated_params`;
- recursive `commit_w`;
- tiered `tiered_commit_u_final`.

### Tiered First-Tier B'

Purpose:

```text
u_j = B' * slice_j(concat(t_hat_i))
```

Output shape:

```text
Vec<CyclotomicRing<F, D>> per slice
```

Current backend method:

- `digit_rows`

The code loops over input chunks of width `params.b_key.col_len()`.

### Tiered F Commit

Purpose:

```text
u_final = F * decompose(u_0 || ... || u_{f-1})
```

Output shape:

```text
Vec<CyclotomicRing<F, D>>  // F rows
```

Current backend method:

- `digit_rows`

The same shared matrix prefix is used; the code passes `f_key.row_len()` and
the actual `u_hat.len()` as the logical width.

## Current Branches That Make `commit()` Feel Messy

The branches are real, but they are mixed across abstraction layers.

1. Public singleton vs batched input
   - `commit` and `batched_commit` differ mostly in validation and grouping.
   - Both end at `commit_with_validated_params`.

2. Dense vs one-hot vs recursive witness
   - Today this is owned by `AkitaPolyOps::commit_inner`.
   - Each representation builds a different compute plan for the same logical
     operation `A*s`.

3. Dense cached digits vs dense coefficient blocks
   - This is an optimization branch inside `DensePoly::commit_rows`.
   - Algebraically both are the same matrix-vector multiply.

4. One-hot single-chunk vs multi-chunk
   - This is a storage-layout branch inside `OneHotBlocks`.
   - Algebraically both are sparse shifted-monomial inputs to `A`.

5. One-hot small-block vs column-sweep vs safety fallback
   - This is a CPU kernel scheduling branch.
   - It should stay below a clean matrix backend interface.

6. Single-tier vs tiered outer commitment
   - This is protocol/layout level, not polynomial representation level.
   - It is currently implemented directly in `commit_with_validated_params`.

7. ZK vs non-ZK outer blinding
   - This is feature-gated and currently only part of the single-tier B path.

## Refactor Implications

A useful future `matrix_backend` design should probably separate four layers:

### 1. Protocol Commit Pipeline

Owns:

- input validation;
- opening batch to `LevelParams`;
- grouping many polynomials into one commitment group;
- deciding single-tier vs tiered layout;
- constructing `RingCommitment` and `AkitaCommitmentHint`.

This layer should not know whether dense uses cached digits, whether one-hot
uses column sweep, or whether CPU uses NTT tiling.

### 2. Witness-To-A-Input Adapter

Owns representation-specific views:

- dense coefficients;
- dense cached digit planes;
- one-hot sparse entries;
- recursive i8 rows.

It should expose a common "blocked vector" view of the A input. The view may be
dense, digit, sparse, shifted-monomial, or raw i8, but the protocol layer should
only see:

```text
number of blocks
A width
commit digit depth
how to multiply A by each block
```

### 3. Matrix Backend

Owns matrix multiplication:

```text
negacyclic_matvec(matrix, input) -> rows
cyclic_matvec(matrix, input) -> rows      // for ring-switch code, not root commit
```

The API should make matrix identity and shape explicit. Today many calls
implicitly read a prefix of the shared setup matrix based on row count and input
length. A more explicit backend would avoid hidden coupling like:

```text
backend.digit_rows(prepared, row_len, digits, log_basis)
```

where the column count is `digits.len()`.

### 4. Decomposition Unit

Owns:

```text
decompose_rows(t, num_digits_open, log_basis) -> FlatDigitBlocks
recompose/validate for debug or checked boundaries
```

This is repeated enough that it should be a first-class helper. It is not a
matrix multiply, but it sits between the A-side and B-side matrix multiplies and
is part of the conceptual commit pipeline.

## Suggested Vocabulary For A Future Design

These names are only research vocabulary for discussion; they are not an
implementation proposal yet.

```text
CommitMatrixRole
    AInner
    BOuter
    BOuterTierSlice
    FOuterTier

BlockedCommitInput
    DenseCoeffBlocks
    DenseDigitBlocks
    OneHotShiftedMonomials
    RecursiveI8Strided

CommitMatVecPlan
    matrix_role
    rows
    cols
    log_basis
    input

CommitMatVecOutput
    PerBlockRows(Vec<Vec<Ring>>)
    SingleRows(Vec<Ring>)
```

The important design point is that "dense", "one-hot", "sparse-ring", and
"recursive witness" are input representations for matrix multiplication, not
separate protocol concepts.

## Main Takeaways

- The conceptual root commit is already uniform:

```text
t_i = A * s_i
t_hat_i = decompose(t_i)
u = B * concat_i(t_hat_i)
```

- Dense differs because it starts from full ring coefficients and may cache
  their digit decomposition.
- One-hot differs because it never materializes `s_i`; it treats the input as
  sparse shifted monomials and accumulates selected A columns.
- The outer B/F work is common across dense, one-hot, and recursive witness
  commits, but it is still written at the low-level `digit_rows` layer.
- The clearest unification target is a protocol-level commit pipeline that
  invokes generic matrix-vector plans for A, B, and F, plus a shared row
  decomposition unit.
