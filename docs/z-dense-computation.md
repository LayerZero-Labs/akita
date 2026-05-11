# Computing `z_dense` Faster — Structural Analysis and Proposal

This note focuses only on the `z_dense` term in the verifier-side ring-switch
replay:

```rust
let z_dense = {
    let _span = tracing::info_span!("m_eval_z_dense").entered();
    let z_segment: Vec<E> = cfg_into_iter!(0..ws.z_len)
        .map(|x| {
            let compound_dig = x / ws.z_total_blocks;
            let global_blk = x % ws.z_total_blocks;
            let dc = compound_dig / prepared.depth_fold;
            let df = compound_dig % prepared.depth_fold;
            let point_idx = global_blk / prepared.block_len;
            let blk = global_blk % prepared.block_len;
            let phys_k = point_idx * ws.inner_width + blk * prepared.depth_commit + dc;
            -z_base[phys_k].mul_base(ws.fold_gadget[df])
        })
        .collect();
    eval_offset_eq_tensor(
        full_vec_randomness,
        ws.offset_z,
        E::one(),
        &[z_segment.as_slice()],
    )
};
```

The implementation today does **two expensive things**:

1. **Materialises** the full `z_segment: Vec<E>` of length `z_len = depth_fold ·
   depth_commit · num_points · block_len`, each entry costing one extension
   multiplication.
2. Calls `eval_offset_eq_tensor` on that materialised single-factor vector,
   which then evaluates an `O(z_len)` multilinear extension.

The argument of this note: `z_segment` has the **same kind of separable
outer-product structure** that `w_sep` and `t_sep` exploit. Once that
structure is exposed, `z_dense` reduces from `Θ(z_len)` extension muls to
`Θ(num_points · block_len + depth_fold + depth_commit + n_a · inner_width)`
— roughly the cost of `m_eval_z_base` alone, with the `m_eval_z_dense`
overhead almost entirely eliminated.

This document is the analogue of `w-sep-computation.md` /
`t-sep-computation.md` for the `z_dense` slice.

## 1. What `z_dense` Represents

In the paper matrix, the `\hat z` columns participate in the consistency row

```text
(c^T ⊗ G_1) · \hat w  -  (... · G_1 · A) · \hat z  =  0
```

After the ring-switch challenge `alpha` and the row-randomization `tau1`, the
verifier evaluates a virtual row-vector `M` at a multilinear point. The
`\hat z` part of that row-vector contributes `z_dense` to the final
`M`-evaluation.

Two pieces feed into `z_dense`:

- **`z_base`** — the per-`(point, block, digit_commit)` value, which in turn
  has two summands:
  - the **structured/consistency** summand
    `consistency_weight · opening_point[p].a[blk] · g1_commit[dc]`, and
  - the **matrix-A** summand
    `Σ_a a_weights[a] · eval_alpha(a_view[a][blk·depth_commit + dc])`.
- **`z_segment`** — `z_base` re-stretched over the `depth_fold` axis with
  `fold_gadget[df]`, signed-negated.

`z_dense` is then the multilinear-extension evaluation of `z_segment` at
`full_vec_randomness`, with offset `offset_z`.

## 2. Notation

Reusing the symbols from `mflat-eval-fusion.md` and the workspace:

```text
P  = num_points
B  = block_len
DC = depth_commit
DF = depth_fold
n_a = number of A rows
W  = inner_width   = B · DC                         (per-point z_base width)
N_z = z_len        = DF · DC · P · B  =  DF · P · W (total z_segment length)
```

Plus the verifier-side data:

```text
opening_points[p].a   :  Vec<F>, length B (one per opening point p)
g1_commit             :  Vec<F>, length DC                 (commit gadget)
fold_gadget           :  Vec<F>, length DF                 (fold gadget)
a_view                :  RingMatrixView<F, D>, n_a rows of the SIS matrix
a_weights             :  &[E], length n_a                  (tau1 weights for A rows)
consistency_weight    :  E                                 (tau1 weight for the consistency row)
alpha_pows            :  &[E], length D                    (ring-switch challenge powers)
offset_z              :  usize                             (start of z segment in M)
```

## 3. The Naive Computation (current)

Strictly following the code:

```text
for k in 0..(P · W):                                    -- z_base build
  point_idx = k / W
  local_k   = k mod W
  blk       = local_k / DC
  dc        = local_k mod DC
  z_base[k] = consistency_weight · opening_points[point_idx].a[blk] · g1_commit[dc]
            + Σ_a a_weights[a] · eval_alpha( a_view.row(a)[ blk · DC + dc ] )

for x in 0..N_z:                                        -- z_segment build
  compound_dig = x / (P · B)
  global_blk   = x mod (P · B)
  dc           = compound_dig / DF
  df           = compound_dig mod DF
  point_idx    = global_blk / B
  blk          = global_blk mod B
  phys_k       = point_idx · W + blk · DC + dc
  z_segment[x] = - z_base[phys_k] · fold_gadget[df]

z_dense = eval_offset_eq_tensor(
            full_vec_randomness,
            offset_z,
            1,
            [ z_segment ]                               -- single factor of length N_z
          )
```

Cost shape:

```text
ring evals          : P · W · n_a    +  P · W            (matrix A part + structured)
extension muls      : N_z = DF · P · W                   (z_segment build)
                    + Θ(N_z)                             (the inner MLE eval over a single factor)
peak memory         : Θ(N_z · sizeof(E))                 (z_segment resident)
```

For NV=32 OneHot D=32 single-claim single-point (the canonical preset):

```text
P = 1, B = 65536, DC = 1, DF = 10, n_a = 3, W = 65536, N_z = 655 360
```

(rough proportions; exact numbers depend on the layout). Measured today:

```text
m_eval_z_base   ≈  2.0 ms
m_eval_z_dense  ≈  4.6 ms
```

After the §9 W_d+T_b fusion the rest of the verifier is small, so `z_dense`
is now the second-largest verifier cost (`m_eval_w_d_t_b_new_approach ≈
7.5 ms` ⇒ `m_eval_z_dense ≈ 4.6 ms` ⇒ everything else ≪ 1 ms).

## 4. The Index Structure of `z_segment`

The flat index `x ∈ [0, N_z)` decomposes (LSB → MSB) as:

```text
x  =  blk  +  P · ?  +  ...                -- but actually:

x  =  compound_dig · (P · B)  +  global_blk
   = (dc · DF + df) · (P · B) + (point_idx · B + blk)
```

So the bit positions of `x` are (bit index 0 = LSB):

| range | width    | dimension      |
|-------|---------:|----------------|
| `[0, log₂ B)`                       | `log₂ B`  | `blk`        |
| `[log₂ B, log₂ B + log₂ P)`         | `log₂ P`  | `point_idx`  |
| `[log₂(P·B), log₂(P·B) + log₂ DF)`  | `log₂ DF` | `df`         |
| `[log₂(P·B·DF), end)`               | `log₂ DC` | `dc`         |

Concretely, in **single-point** (`P = 1`), the layout simplifies to:

```text
LSB                                                    MSB
[blk: log₂ B][df: log₂ DF][dc: log₂ DC]
```

i.e. `x = blk + B · df + B · DF · dc`, so `df` sits in the middle and
separates the `blk` bits from the `dc` bits.

`x_challenges` is correspondingly split into `(x_blk, x_pt, x_df, x_dc)`,
the multilinear evaluation point for each axis.

## 5. Where the Structure Lives — `z_segment` Decomposes Additively

Substituting `z_base` into `z_segment`:

```text
z_segment[ blk, point_idx, df, dc ]
   = - fold_gadget[df]
     · ( consistency_weight · opening_points[point_idx].a[blk] · g1_commit[dc]
       + Σ_a a_weights[a] · eval_alpha( a_view.row(a)[ blk·DC + dc ] ) )
```

This is the sum of two parts:

```text
z_segment       =  z_segment_struct  +  z_segment_matrix

z_segment_struct[ blk, pt, df, dc ]
    = - consistency_weight · opening_points[pt].a[blk] · fold_gadget[df] · g1_commit[dc]            (1)

z_segment_matrix[ blk, pt, df, dc ]
    = - matrix_A[ blk, dc ] · fold_gadget[df]                                                        (2)

matrix_A[ blk, dc ]
    = Σ_a a_weights[a] · eval_alpha( a_view.row(a)[ blk·DC + dc ] )                                  (3)
```

This is exactly the same shape as the `w_sep`/`t_sep` decomposition: a
small per-block "structured" part plus an A-matrix-shaped "matrix" part.

### 5.1 `z_segment_struct` is a Pure Outer Product

(1) is a **rank-1 tensor product** over the four axes:

```text
z_segment_struct  =  - consistency_weight  ⊗  a_combined ⊗ fold_gadget ⊗ g1_commit
```

where `a_combined[ pt, blk ]  = opening_points[pt].a[blk]` (a 1D vector of
length `P · B` indexed first by `pt`, then by `blk` if we flatten naively
— but see §5.3 for the actual order required).

In **single-point** (`P = 1`), `a_combined` collapses to just
`opening_points[0].a` of length `B`.

### 5.2 `z_segment_matrix` is Constant in `point_idx` and Rank-1 in `df`

(2) does **not** depend on `point_idx` (the matrix-A summand of `z_base`
already does not depend on `point_idx`; the multi-point lookup is only
through the `opening_points[pt].a` factor of (1)).

(2) factors as:

```text
z_segment_matrix  =  - matrix_A[ blk, dc ]  ⊗_pt 1  ⊗_df fold_gadget
```

— a rank-1 outer product **across the (blk, dc) bundle**, the all-ones
vector across `pt`, and `fold_gadget` across `df`. The remaining
non-trivial structure is the 2-D `matrix_A`, which has length `B · DC =
W = inner_width`.

### 5.3 Bit-Position Alignment for `eval_offset_eq_tensor`

`eval_offset_eq_tensor` consumes a list of 1-D factors and assigns them
to **consecutive bit ranges** of `x_challenges` starting from the LSB. So
to use it directly:

- **The structured part (1) aligns cleanly.** The bit layout is
  `blk → pt → df → dc` from LSB up; the factor list `[a_combined,
  fold_gadget, g1_commit]` (with `a_combined` flattened in `(blk, pt)`
  order, length `P · B`; `fold_gadget` length `DF`; `g1_commit` length
  `DC`) occupies exactly bits `[0, log₂(P·B))`, `[log₂(P·B), log₂(P·B·DF))`,
  `[log₂(P·B·DF), end)` — a perfect match. The offset shift `offset_z`
  is handled by the carry path of `eval_offset_eq_tensor`.

- **The matrix part (2) does not align cleanly** because `matrix_A`'s
  natural axes are `(blk, dc)`, and the bits for `dc` are above the bits
  for `df`. The two axes of `matrix_A` are *non-contiguous* in `x`, so
  it cannot be passed as a single 1-D factor.

§7 below gives the proposed workaround for the matrix part.

## 6. Proposed Algorithm — Separated `z_sep` + `z_a` (Option A)

We commit to **separating `z_dense` into two parts** mirroring the
W/T architecture:

```text
z_sep  :=  z_dense_struct       -- the consistency / structured part
z_a    :=  z_dense_matrix       -- the matrix-A part

z_dense  =  z_sep  +  z_a
```

The naming intentionally matches `w_sep`/`t_sep` and `w_d`/`t_b`: the
`_sep` suffix denotes the structured (separable) part, and `z_a` is the
A-matrix matrix-row part (analogous to `w_d` being the D-matrix part of
`\hat w` and `t_b` being the B-matrix part of `\hat t`). Each part can
later be **fused** with its W/T sibling if we want to share `r_eval`
across SIS rows (see §10 — "future combine with B and D"). For now the
proposal handles `z_sep` and `z_a` independently and uses
`eval_offset_eq_tensor` for both.

**Implementation status (this PR):** the separation `z_sep` + `z_a` is
implemented as **two new `SliceMleEvaluator`s** —
`ZStructuredRowsEvaluator` and `ZMatrixRowsEvaluator` — mirroring the
`W*RowsEvaluator` / `T*RowsEvaluator` pair. The prover's `z_segment`
layout is kept **unchanged** (the originally-proposed Option A layout
permutation isn't needed for this approach).

The two `Z*RowsEvaluator`s peel `block_len` (not `num_blocks` like
`\hat w` / `\hat t`) on the inner axis. The structured part precomputes
a per-opening-point block summary `a_block_summary[pt]` over
`opening_points[pt].a` (length `block_len`). The matrix part builds
`matrix_A` once (the A-matrix summand of the legacy `z_base`,
point-independent), then summarises each `dc`-column over the `blk`
axis. Each per-outer-index `compute_inner_sum` call is `O(1)`; the
default `compute_outer_sum` runs the standard high-bit `eq` pass over
`num_outer = P · DF · DC` outer indices.

A **fallback path** (materialised `z_segment` slice + single-factor
`eval_offset_eq_tensor`) handles configurations where `block_len` is
not a power of two — same trade-off `r_sep` makes via
`r_tail_dims_pow2`. The fallback covers the few non-power-of-two
recursive levels; root levels and most recursive levels use the trait
path.

Measured at NV=32 OneHot D=32 single-claim single-point
(end-to-end `batched_verify`, 5-run median):

```text
m_eval_w_d_t_b               7.46 ms    (unchanged — `new_approach` path)
m_eval_z_sep                 3.5 µs     (trait path, was ~5 ms materialised)
m_eval_z_a                   1.89 ms    (trait path, was ~6 ms materialised)
total batched_verify         20.5 ms    (was ~30 ms with materialised z separation)
```

That is, the trait conversion **fully recovers and exceeds** the
performance of today's combined `z_dense` while keeping the
`z_sep` / `z_a` separation that future B/D fusion needs.

### 6.1 `z_sep` — Structured Part, Direct Tensor Eval

```text
a_combined :=  for pt in 0..P, for blk in 0..B:
                  push opening_points[pt].a[blk]            -- length P · B, base field

z_sep
   = eval_offset_eq_tensor::<E>(
         full_vec_randomness,
         offset_z,
         - consistency_weight,
         &[ lift_to_E(a_combined),
            lift_to_E(g1_commit),                            -- length DC
            lift_to_E(fold_gadget) ],                        -- length DF
     )
```

(The factor order matches the new layout `[pt][blk][dc][df]` from
§8; see §6.3 for why `g1_commit` precedes `fold_gadget`.)

The lifts to `E` are linear, `O(P·B + DC + DF)` muls each. Equivalent
in spirit to `r_sep`'s call, which lifts `r_gadget` to `E` once and
calls `eval_offset_eq_tensor`.

**Cost:** `O(P · B + DC + DF)` extension muls.

### 6.2 `z_a` — Matrix Part, `matrix_A` Build + Single Tensor Eval

#### 6.2.1 Build `matrix_A` Once (replaces today's `z_base` matrix summand)

```text
for blk in 0..B, for dc in 0..DC:                            -- length W = B · DC
  matrix_A[ blk + B · dc ]
      =  Σ_a  a_weights[a] · eval_alpha( a_view.row(a)[ blk·DC + dc ] )
```

This is the **matrix-A summand of the existing `z_base` build**, just
spelled out separately. Cost: `n_a · W` ring evaluations. The `(blk, dc)`
flattening order is chosen to match the new layout where `(blk, dc)`
are at the low bits of `x`, with `blk` LSB-most (so the flat index
`blk + B · dc` advances `blk` fastest).

In multi-point, `matrix_A` is still indexed only by `(blk, dc)` (the
matrix-A summand of `z_base` is point-independent). Today's `z_base`
redundantly stores `P` copies; the proposed form stores `matrix_A`
**once**, saving `(P − 1) · n_a · W` ring evals for `P > 1`.

#### 6.2.2 Evaluate the MLE — Direct Tensor Call

With the new layout `x' = pt + P · (blk + B · (dc + DC · df))` (§8),
the bits split into four contiguous ranges `[pt][blk][dc][df]` from
LSB up. The matrix part factors with these lengths exactly:

```text
z_a
   = eval_offset_eq_tensor::<E>(
         full_vec_randomness,
         offset_z,
         -E::one(),
         &[ ones_vec_of_len_P,            -- length P (all-ones)
            matrix_A,                     -- length B · DC = W, indexed [blk][dc]
            lift_to_E(fold_gadget),       -- length DF
         ],
     )
```

`ones_vec_of_len_P`'s MLE is identically `1` for power-of-two `P` (the
`eq` polynomial sums to 1 over the full hypercube). A small
optimisation: when `P = 1` we omit this factor entirely; when `P > 1`
power-of-two we can short-circuit the factor's `mle_small` call. The
remaining cost is dominated by `matrix_A`.

**Cost:** `n_a · W` ring evaluations (matrix_A build) + `O(W + DF)`
extension muls (MLE eval). The `ones_pt` factor adds essentially no
work.

### 6.3 Why the Specific Factor Order

`eval_offset_eq_tensor` assigns factors to **consecutive bit ranges from
LSB up**. For our new layout

```text
x'  =  pt  +  P·blk  +  P·B·dc  +  P·B·DC·df             (single-axis form)
```

the LSB-to-MSB axis order is `[pt, blk, dc, df]`. So:

| factor in `&[...]`         | length  | bit range covered                                         |
|----------------------------|---------|-----------------------------------------------------------|
| `ones_pt`                  | `P`     | `[0, log₂ P)`                                             |
| `matrix_A` (or `a_combined`)| `B`/`P·B` | `[log₂ P, log₂(P · B))` (skipping `pt` for `matrix_A`) |
| `g1_commit` / `matrix_A`-cont | `DC` / `B·DC` | `[log₂(P · B), log₂(P · B · DC))`                  |
| `fold_gadget`              | `DF`    | `[log₂(P · B · DC), end)`                                 |

For **`z_sep`**, we want `a_combined` to occupy `[0, log₂(P · B))` —
it spans `pt` and `blk`, both at the low bits, so we make
`a_combined[pt + P · blk] = opening_points[pt].a[blk]` and pass it as
factor 1 (length `P · B`). Then `g1_commit` and `fold_gadget` follow
in the natural axis order.

For **`z_a`**, we want `matrix_A[blk + B · dc]` to occupy
`[log₂ P, log₂(P · B · DC))`. Since `eval_offset_eq_tensor` advances
the bit cursor in factor order, we pass `ones_pt` first (length `P`,
covering the `pt` bits) and then `matrix_A` (length `B · DC`, covering
the `blk` and `dc` bits together). `fold_gadget` follows.

In **single-point** (`P = 1`) — the canonical preset — `ones_pt` has
length 1 and contributes nothing; the bit layout collapses to
`[blk][dc][df]` and the call is

```text
z_a = eval_offset_eq_tensor::<E>(
        full_vec_randomness, offset_z, -E::one(),
        &[ matrix_A, lift_to_E(fold_gadget) ],
      )
```

— exactly the shape of `r_sep`'s call (`[r_gadget_ext, eq_tau1]`).

## 7. The Overhead of Separating — Strict Win

Separating `z_dense` into `z_sep` + `z_a` does **not** introduce
overhead — it strictly reduces total work. Below is the line-by-line
accounting against today's combined path.

### 7.1 Today (combined)

```text
ring evals:    P · W · n_a       (matrix-A summand of z_base)
              + P · W            (structured summand of z_base — these are muls really, not ring evals)
ext muls:      DF · P · W        (z_segment build: one mul per cell)
              + Θ(N_z)            (eval_offset_eq_tensor on a single factor of length N_z)
peak Vec<E>:   N_z · sizeof(E)   (z_segment resident)
```

### 7.2 Proposed (separated, Option A layout)

```text
ring evals:    n_a · W           (matrix_A build only — point-independent)
ext muls:      O(P · B + DF + DC)   (z_sep call)
              + O(W + DF + P)       (z_a call)
              ≈ O(W + P · B + DF + DC)
peak Vec<E>:   W · sizeof(E)     (matrix_A resident; nothing else big)
```

### 7.3 Net Change

| metric              | today                             | proposed                     | factor saved          |
|---------------------|-----------------------------------|------------------------------|-----------------------|
| ring evals          | `(n_a + 1) · P · W`               | `n_a · W`                    | `(n_a + 1)/n_a · P` ≈ `(4/3) · P` for `n_a = 3` |
| ext muls (post)     | `~2 · DF · P · W`                 | `~W + P · B`                 | `~2 · DF` for `P = 1` (≈ 20× at NV=32) |
| peak `Vec<E>`       | `DF · P · W · sizeof E`           | `W · sizeof E`               | `DF · P` (≈ 10× at NV=32) |
| `eval_offset_eq_tensor` calls | 1                       | 2                            | +1 call, but each is on **much smaller** factors |

The "+1 call" overhead is the only real concern. Each
`eval_offset_eq_tensor` call has setup cost `Θ(|x_challenges|)` for
the carry-DP initialisation. With `|x_challenges| ≈ 21` bits (NV=32),
that's ~20 extra muls per call — completely negligible against the
~50 k–100 k muls saved on the body.

### 7.4 Concrete numbers (NV=32 OneHot D=32, single-claim single-point)

```text
P = 1, B = 65 536, DC = 1, DF = 10, n_a = 3, W = 65 536, N_z = 655 360
```

- **Ring evals:** today `4 · 65 536 = 262 144` ⇒ proposed `3 · 65 536 = 196 608` (saving 25%, since `(P − 1) · n_a · W = 0` for `P = 1`).
- **Post-eval ext muls:** today `~2 · 10 · 65 536 ≈ 1 310 k` ⇒ proposed `~65 536 + 65 536 + 10 ≈ 131 k` (saving ≈ 10×).
- **Peak `Vec<E>`:** today `~10 MB` ⇒ proposed `~1 MB`.

For multi-point or larger `DC`, the ring-eval saving grows with `P`
because today's `z_base` redundantly evaluates the same `a_view` cells
once per opening point.

## 8. The Layout Change (Option A)

The protocol's `z_segment` layout is currently

```text
x  =  blk + P·B · (df + DF · dc)             ; bit layout LSB → MSB:  [blk][df][dc]    (single-point)
   =  (pt · B + blk) + P·B · (df + DF · dc)  ; bit layout LSB → MSB:  [blk][pt][df][dc]  (multi-point)
```

— `df` sits between `blk` and `dc`, which is what blocks the
`matrix_A[blk, dc]` factor. The proposed layout swaps `df` to the
top:

```text
x' =  pt + P · (blk + B · (dc + DC · df))    ; bit layout LSB → MSB:  [pt][blk][dc][df]
```

This makes:

- **`(pt, blk)` contiguous** at bits `[0, log₂(P · B))`  → fits
  `a_combined` for `z_sep`.
- **`(blk, dc)` contiguous** at bits `[log₂ P, log₂(P · B · DC))` →
  fits `matrix_A` for `z_a`.
- **`df` at the top** → fits `fold_gadget` as the last factor.
- **`pt` at the very bottom** for both halves: an all-ones factor of
  length `P` covers it cleanly (degenerates to nothing in
  single-point).

### 8.1 Prover-side Edit

Replace the layout decoder loop in
`crates/akita-prover/src/protocol/ring_switch.rs` (currently around
lines 797–808):

```rust
// BEFORE
let z_segment: Vec<E> = cfg_into_iter!(0..z_len)
    .map(|x| {
        let compound_dig = x / z_total_blocks;
        let global_blk   = x % z_total_blocks;
        let dc = compound_dig / depth_fold;
        let df = compound_dig % depth_fold;
        let point_idx = global_blk / block_len;
        let blk       = global_blk % block_len;
        let phys_k    = point_idx * inner_width + blk * depth_commit + dc;
        -(z_base[phys_k] * fold_gadget[df])
    })
    .collect();

// AFTER (new layout x' = pt + P·(blk + B·(dc + DC·df)))
let z_segment: Vec<E> = cfg_into_iter!(0..z_len)
    .map(|x_prime| {
        let pt_blk_dc = x_prime % (num_points * block_len * depth_commit);
        let df        = x_prime / (num_points * block_len * depth_commit);
        let pt_blk    = pt_blk_dc % (num_points * block_len);
        let dc        = pt_blk_dc / (num_points * block_len);
        let pt        = pt_blk % num_points;
        let blk       = pt_blk / num_points;
        let phys_k    = pt * inner_width + blk * depth_commit + dc;
        -(z_base[phys_k] * fold_gadget[df])
    })
    .collect();
```

The body of each iteration is unchanged — only the index decoding
changes. `phys_k` still points at the same `z_base` cell as before;
the difference is the order in which we walk those cells when filling
the segment.

`offset_z` is **unchanged** — `z_segment.len() == z_len` is identical
in both layouts; the segment occupies the same range of `M`.

### 8.2 Verifier-side Edit (default streamed path)

The legacy verifier path in
`crates/akita-verifier/src/protocol/slice_mle.rs` mirrors the prover:
update the same loop in `compute_non_peeled_parts`. It's the same
edit, character for character.

### 8.3 New Verifier Path Using the Layout

The `new_approach`-style verifier (which we'll add for `z_dense` after
the layout change) doesn't materialise `z_segment` at all — it just
calls the two `eval_offset_eq_tensor` calls of §6 with factors in the
new layout's order. The savings of §7.3 take effect.

### 8.4 What Else Touches the Layout?

The layout enters the protocol in three places:

1. **Prover's `z_segment` build** (above).
2. **Verifier's `compute_non_peeled_parts`** (above) — only when running
   the legacy single-factor path.
3. **`offset_z`'s relation to `M`** — unchanged: `offset_z` and
   `z_len` are layout-agnostic (they just say "the z segment occupies
   indices `[offset_z, offset_z + z_len)`").

Nothing else in the protocol depends on the `z_segment` ordering. The
opening-point evaluation, the witness commitments, the `eq_tau1`
weights, the sumcheck folding, the ring-switch challenges, and all
other M-segments (`\hat w`, `\hat t`, `r_tail`, blinding) are
unaffected.

### 8.5 Compatibility With `prepared_row_eval_matches_materialized`

The `prepared_row_eval_matches_materialized` test in
`crates/akita-pcs/tests/ring_switch.rs` constructs a fully materialised
`M`-segment, evaluates its MLE at a random point via
`multilinear_eval`, and compares against `prepared.eval_at_point`.
Because the layout edit is applied **on both sides simultaneously**
(prover and the materialised reference), the test continues to hold:
the materialised `M` and the streamed `eval_at_point` see the same
permuted ordering and produce identical scalars.

## 9. Putting It Together — Verifier Code

After the layout change, the new `z_dense` evaluator collapses to:

```rust
fn eval_z_dense<F, E, const D: usize>(
    prepared: &RingSwitchDeferredRowEval<E>,
    ws: &EvalAtPointWorkspace<'_, F, E, D>,
    full_vec_randomness: &[E],
    opening_points: &[RingOpeningPoint<F>],
) -> (E, E)   // (z_sep, z_a)
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    // ----- z_sep ------------------------------------------------------
    let a_combined: Vec<E> = (0..prepared.num_points)
        .flat_map(|pt| {
            opening_points[pt].a.iter().map(|&v| E::lift_base(v))
        })
        .collect();
    let g1_commit_ext: Vec<E> = ws.g1_commit.iter().copied().map(E::lift_base).collect();
    let fold_gadget_ext: Vec<E> = ws.fold_gadget.iter().copied().map(E::lift_base).collect();

    let z_sep = eval_offset_eq_tensor(
        full_vec_randomness,
        ws.offset_z,
        -ws.consistency_weight,
        &[&a_combined, &g1_commit_ext, &fold_gadget_ext],
    );

    // ----- z_a --------------------------------------------------------
    let n_cols = prepared.block_len * prepared.depth_commit;          // W
    let matrix_a: Vec<E> = cfg_into_iter!(0..n_cols)
        .map(|i| {
            let blk = i % prepared.block_len;
            let dc  = i / prepared.block_len;
            let local_k = blk * prepared.depth_commit + dc;
            let mut acc = E::zero();
            for (a_idx, &eq_i) in ws.a_weights.iter().enumerate() {
                if !eq_i.is_zero() {
                    acc += eq_i * eval_ring_at_pows(
                        &ws.a_view.row(a_idx)[local_k], &ws.alpha_pows,
                    );
                }
            }
            acc
        })
        .collect();

    // Single-point shortcut: skip the all-ones pt factor (length 1).
    let z_a = if prepared.num_points == 1 {
        eval_offset_eq_tensor(
            full_vec_randomness, ws.offset_z, -E::one(),
            &[&matrix_a, &fold_gadget_ext],
        )
    } else {
        let ones_pt: Vec<E> = vec![E::one(); prepared.num_points];
        eval_offset_eq_tensor(
            full_vec_randomness, ws.offset_z, -E::one(),
            &[&ones_pt, &matrix_a, &fold_gadget_ext],
        )
    };

    (z_sep, z_a)
}
```

This **replaces** the entire `compute_non_peeled_parts::z_base` +
`z_segment` + `eval_offset_eq_tensor` block. The old `m_eval_z_base`
and `m_eval_z_dense` spans collapse into one short
`m_eval_z_sep_z_a` span.

`EvalAtPointParts` gets two new fields, `z_sep` and `z_a`, replacing
the single `z_dense`. `EvalAtPointParts::sum()` becomes
`... + z_sep + z_a + ...`. (Or, to avoid an API break in the
short term, set `z_dense = z_sep + z_a` and keep the field shape;
that's a free rename.)

## 10. Forward Hook — Combining `z_a` With `w_d` and `t_b` Later

The motivation for keeping the matrix-A part separate is that **`z_a`
reads `a_view`** the same way `w_d` reads `d_view` and `t_b` reads
`b_view`. All three views point at the **same shared SIS matrix**; the
only differences are which row-weight slice applies and which column
range is touched. This is the same observation that drove the §9
fusion of `w_d` and `t_b` in `mflat-eval-fusion.md`.

Once `z_a` is its own object, the future fusion looks like:

```text
M_Flat[r, c]  =  eval_alpha( shared_matrix.row(r) [c] )

Eval[r, c]   =  𝟙[r, c is in W's range] · d_w_padded[r] · w_pattern_padded[c]
              + 𝟙[r, c is in T's range] · b_w_padded[r] · t_pattern[c]
              + 𝟙[r, c is in Z's range] · a_w_padded[r] · z_pattern[c]

w_d + t_b + z_a  =  <M_Flat, Eval>
```

with `a_w_padded[r]` covering the n_a rows that `z_a` reads. The
column ranges are different per slice (W covers `[0, w_len/D)`, T
covers `[0, t_len/D)`, Z covers `[0, W) = [0, B · DC)`), but the row
sharing across `D`/`B`/`A` matrices is what generates the saving —
just like §9 saved `min(n_d, n_b)` rows of redundant ring evals.

We do not need that fusion now. The §6 separation is the
**precondition** that makes it possible later — without separating
`z_dense`, we'd have to re-extract the matrix-A summand every time the
fusion code wanted it.

## 11. Cost Summary

| step                              | today                     | proposed (§6)                                   |
|-----------------------------------|---------------------------|--------------------------------------------------|
| `z_base` build (ring evals)       | `P · W · n_a + P · W`     | `n_a · W` only (matrix-A summand)                |
| `z_segment` build (ext muls)      | `N_z = DF · P · W`        | `0`                                              |
| structured part MLE               | (folded into single-factor) | `O(P · B + DF + DC)` (§6.1)                    |
| matrix part MLE                   | `Θ(N_z)`                  | `O(W + DF + P)` (§6.2)                          |
| peak `Vec<E>` resident memory     | `Θ(N_z · sizeof E)`       | `Θ(W · sizeof E)`                                |

For NV=32 OneHot D=32 single-claim single-point, taking the typical
`P = 1, B = 65 536, DC = 1, DF = 10, n_a = 3, W = 65 536, N_z = 655 360`:

- Ring evals: `4 · 65 536 ≈ 262 k` → `3 · 65 536 ≈ 197 k` (~25% saving).
- Post-ring-eval ext muls: `~1.3 M` → `~131 k` (~10× saving).
- Peak `Vec<E>` resident: `~10 MB` → `~1 MB` (~10× saving).

In multi-point the ring-eval saving grows with `P` because today's
`z_base` redundantly stores the matrix-A summand `P` times.

## 12. Expected End-to-End Impact

Today (5-run median, NV=32 OneHot D=32, with `new_approach` on):

```text
m_eval_w_d_t_b_new_approach :  ≈ 7.5 ms
m_eval_z_base               :  ≈ 2.0 ms
m_eval_z_dense              :  ≈ 4.6 ms
total batched_verify         :  ≈ 23.2 ms
```

After this proposal (estimated, single-point single-claim):

```text
m_eval_w_d_t_b_new_approach :  unchanged ≈ 7.5 ms
m_eval_z_sep_z_a (combined) :  ≈ 1.5–2 ms        (matrix_A ring evals dominate)
total batched_verify         :  ≈ 19–20 ms       (≈ 13–18% faster)
```

After this, the verifier hot path is essentially
`m_eval_w_d_t_b_new_approach` plus a small `z`-side layer; everything
else is sub-ms. After the §10 fusion of `z_a` with `w_d`/`t_b` (a
follow-up, not in this proposal) the `z_a` ring evals would also be
absorbed into the shared `r_eval` cache, dropping verify time
further.

## 13. Cross-References

- `w-sep-computation.md` and `t-sep-computation.md` for the analogous
  separable-vs-matrix split on `\hat w` and `\hat t`.
- `w-d-computation.md` and `t-b-computation.md` for the matrix-row
  evaluators that `z_a` mirrors.
- `mflat-eval-fusion.md` §11.2 for the materialised-`Eval` pattern;
  the matrix part of `z_dense`/`z_a` is what would be added to the
  unified `<M_Flat, Eval>` sum if we ever fuse it across `w_d`,
  `t_b`, and `z_a` (§10).
- `eval_offset_eq_tensor` (in `crates/akita-algebra/src/offset_eq.rs`)
  for the multi-factor offset-aware MLE evaluator that §6 reuses.

## 14. Practical Notes

- **Single-claim, single-point production** (the canonical preset): the
  `point_idx` axis has size 1 and `DC = 1`, so the layout change is
  effectively a no-op (`P = DC = 1` collapses the layout permutation
  to identity). The structured and matrix calls each take only a few
  factors: `z_sep ← eval_offset_eq_tensor(.., [a_combined, fold_gadget],
  -consistency_weight)` and `z_a ← eval_offset_eq_tensor(.., [matrix_A,
  fold_gadget], -1)`.

- **Multi-point**: the layout change has visible effect. The
  structured part picks up `a_combined`'s `(pt, blk)` flattening; the
  matrix part picks up an `ones_pt` factor of length `P`. No
  structural obstacle; the savings are **larger** because today's
  `z_base` stores the matrix part `P` times redundantly.

- **`DC > 1`** (Full / dense layouts): Option A handles this
  generically — `matrix_A` becomes a length-`B·DC` factor, and the
  call is otherwise unchanged. No special-casing needed.

- **Correctness check.** The split is purely arithmetic — `(1)` plus
  `(2)` exactly reconstructs the original `z_segment[x]` per cell — so
  the MLE evaluation also matches identity-by-identity. The layout
  change is also purely a re-permutation of the same data; both sides
  see the same permutation and arrive at the same scalar. Use the
  existing
  `crates/akita-pcs/tests/ring_switch.rs::tests::prepared_row_eval_matches_materialized`
  as the regression oracle.

- **Future combine with B and D (§10).** Once `z_a` is its own object,
  fusing it with `w_d` and `t_b` is a small additional step — the
  same `M_Flat × Eval` framing extends naturally with one more set of
  row-weight + column-pattern factors. Until then, `z_a` standing
  alone is already a significant speedup.

## 15. Alternatives Considered (Not Taken)

We briefly considered two variants of §6.2 that **don't** require the
prover-side layout change. They are documented here for completeness;
neither was chosen.

- **Option B — Custom MLE evaluator for the legacy layout.** Write a
  bespoke `eval_z_a_legacy(matrix_A, fold_gadget, x_challenges,
  offset_z, ...)` that handles the `(blk, df, dc)` interleaved layout
  by extending the carry-DP algorithm of `eval_offset_eq_tensor_carry`
  to allow one factor to be 2-D over non-contiguous bit ranges.
  Achievable, but adds a new MLE-evaluation primitive and a new
  test surface. Rejected in favour of the generic layout fix.

- **Option C — Rank-1 decomposition.** When `DC = 1`, `matrix_A` is
  trivially rank-1 across `(blk, dc)` and the legacy layout works
  unchanged. When `DC > 1`, this path requires an explicit rank-`R`
  decomposition of `matrix_A`, which is `O(B · DC²)` to build and
  fundamentally not generic. Rejected for the same reason.

Option A is the single solution that works generically across all
layouts, single/multi-point, and any `(B, DC, DF, n_a)`.
