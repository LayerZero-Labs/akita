# Computing `w_d + t_b + z_a` as a Single `M_Flat * Eval` Product

This note shows how the three matrix-row contributions on the verifier side —
`w_d` (the `D · \hat w = v` rows), `t_b` (the `B · \hat t = u` rows), and
`z_a` (the `A · \hat z` rows of the Z segment) — can be written together as a
single inner product

```text
w_d + t_b + z_a  =  <M_Flat, Eval>
```

where:

- `M_Flat` is a **flat** matrix of field elements — exactly one row per SIS
  matrix row, with every entry pre-evaluated at the ring-switch challenge
  `alpha`.
- `Eval` is a verifier-side weight matrix of the same shape.
- `<·, ·>` is the entry-wise (Frobenius) inner product
  `Σ_{r, c} M_Flat[r, c] · Eval[r, c]`.

The whole point of this framing is that `M_Flat` already contains all the
expensive `eval_alpha` work; the verifier only has to compute `Eval`, and the
final scalar `w_d + t_b + z_a` falls out as one inner product. All three
halves read rows of the same shared SIS matrix, so they share `M_Flat`
column-wise — only the column-only `Eval` patterns and the per-row weight
vectors differ between halves.

The user is free to read `M_Flat[r, c]` as a single field element regardless
of how the cyclotomic ring is internally represented — the assumption of this
note is **"each ring element of `M_Flat` is already evaluated at random
`alpha`, so every entry is a single field element"**.

This document is the "matrix–vector view" of the per-row sharing analysis in
`matrix-rows-fusion.md`. It does **not** propose new algebra: every quantity
below is already computable by the verifier today. The new content is
exclusively the **structure of `Eval`** that makes the matrix–vector
identity hold, and the per-cell decoding the verifier uses to populate it.

The relationship to the current implementation:

- `compute_matrix_rows_via_patterns` in
  `crates/akita-verifier/src/protocol/slice_mle.rs` is the single
  function that owns all three halves. It precomputes one `M_Flat`
  (a per-row `r_eval = eval_alpha(shared_matrix.row(r)[c])`) and three
  column-only patterns (`w_pattern_padded`, `t_pattern_per_group[g]`,
  `z_pattern_padded`), then folds them via one inner product per row.
- The previous separate evaluators `WMatrixRowsEvaluator`,
  `TMatrixRowsEvaluator`, and `ZMatrixRowsEvaluator` are gone; their
  `r_eval` / `matrix_a` builds were redundant since `d_view`, `b_view`,
  and `a_view` are all `setup.shared_matrix.ring_view::<D>(n_*, stride)`
  — i.e., views into the *same* backing store with different row
  counts. `M_Flat[r, c]` reads the same memory regardless of which view
  the verifier picks for row `r`.
- A non-pow-of-two `block_len` fallback inside the same function
  materialises the matrix-A summand of `z_segment` and calls
  single-factor `eval_offset_eq_tensor` for `z_a` only — keeping the
  function correct at every recursive level. Same trade-off `r_sep` /
  `r_dense` makes.

## 1. Notation

We reuse the notation from `w-d-computation.md`, `t-b-computation.md`, and
`matrix-rows-fusion.md`:

```text
B   = num_blocks                            (power of two)
C   = num_claims
L   = depth_open    = num_digits
n_a = number of A rows
n_d = number of D rows
n_b = number of B rows per commitment group
D   = ring dimension                        (CyclotomicRing<F, D>)
```

Plus the derived shorthands:

```text
P         = num_points                     opening points
DC        = depth_commit                   commit-side gadget length (Z half)
DF        = depth_fold                     fold-side gadget length (Z half)
B'        = block_len                      Z half's inner block size
                                           (NOT the same as B = num_blocks)

R         = max(n_d, n_b, n_a)             shared SIS row range
w_len / D = C · L · B                      ring-element width of W's column range
t_len / D = C · n_a · L · B                ring-element width of T's column range
z_range   = B' · DC                        ring-element width of Z's column range
N         = max(w_len/D, t_len/D, z_range) chosen ring-element width of M_Flat
```

`R` is the height of `M_Flat`. `N` is the width. Both are determined entirely
by the level's schedule parameters — they do not depend on the verifier's
randomness.

**Note on `z_range`.** At root levels the W and T column ranges dominate,
so `N = max(w_len/D, t_len/D)` and the Z column range fits inside that
width as a strict prefix. At several recursive levels `block_len` grows
faster than `num_blocks` shrinks, and `z_range > max(w_len/D, t_len/D)`;
`N` then captures the Z width too. The implementation pads
`w_pattern_padded` and `t_pattern_per_group[g]` with zeros over the
Z-only suffix so the inner product `<M_Flat, Eval>` keeps a single,
uniform shape across all levels.

The verifier-side data this note references:

```text
full_vec_randomness    = the multilinear evaluation point for the M-table
x_challenges           = full_vec_randomness                         (alias for clarity)
alpha                  = the ring-switch challenge
alpha_pows             = [1, alpha, alpha^2, ..., alpha^{D-1}]

offset_w               = start of the \hat w segment inside M
offset_t               = start of the \hat t segment inside M

eq_tau1                = the random linear combination over paper rows
d_start                = where D's row-weights live inside eq_tau1
b_start                = where B's row-weights live inside eq_tau1
d_weight[r]            = eq_tau1[d_start + r]
b_weight[r, g]         = eq_tau1[b_start + g · n_b + r]
claim_to_group[claim]  = (group_idx, claim_within_group)
```

All of these are already computed in
`EvalAtPointWorkspace::build` (`slice_mle.rs:778`).

## 2. Setting up `M_Flat`

From `ring_switch.rs` the views into the SIS matrix come from the same
backing store:

```rust
let stride = setup.seed.max_stride;
let d_view = setup.shared_matrix.ring_view::<D>(prepared.n_d, stride);
let b_view = setup.shared_matrix.ring_view::<D>(prepared.n_b, stride);
```

So for every shared row index `r ∈ [0, R)`, the rows `d_view.row(r)` and
`b_view.row(r)` (where they overlap) point at the same physical memory. The
two views differ only in the column **range** they look at — never in the
column values.

Define `M_Flat` as the row range of the shared SIS matrix, with every
ring-element entry collapsed to its evaluation at `alpha`:

```text
M_Flat[r, c]  :=  eval_alpha( shared_matrix.row(r) [c] )
              =   < shared_matrix.row(r)[c],  alpha_pows >          (1)
```

for `r ∈ [0, R)` and `c ∈ [0, N)`. Here

```text
eval_alpha(z_0 + z_1·X + ... + z_{D-1}·X^{D-1})
   =  z_0 + z_1·alpha + ... + z_{D-1}·alpha^{D-1}
```

is the standard "evaluate a cyclotomic ring element at the ring-switch
challenge `alpha`" reduction, performed via a dot product against
`alpha_pows` — exactly what `eval_ring_at_pows` in
`crates/akita-verifier/src/protocol/ring_switch.rs:377` does.

`M_Flat` is therefore a `R × N` matrix of field elements. It is "as big as
T's column range", because T's column range strictly contains W's column
range (Section 3).

For ZK or any layout that pads the column space differently, the `r_eval`
size in `matrix-rows-fusion.md` §2 already enforces the same bound; `M_Flat`
inherits that bound automatically.

`M_Flat` requires `R · N` ring evaluations to construct. This is the dominant
work of `t_b` and is shared with `w_d`; see Section 12 for the cost
accounting.

## 3. Two Reshapings of the Column Axis

`w_d` and `t_b` both read entries of `M_Flat` along the same `c` axis, but
they **reshape** that axis differently — using two different bijections
between a flat column index `c` and a `(claim, ..., block)` tuple. This
mismatch is the heart of `matrix-rows-fusion.md` §8 and is why a single
`(eq_low · eq_hi)` factorization cannot serve both.

### 3.1 The W reshaping (D matrix)

From `w-d-computation.md` §5 and §7, for `W`'s view:

```text
c_W( (claim_w, dig_w), b_w )
  =  claim_w · (B · L)
   + b_w     · L
   + dig_w
```

with the outer index

```text
q_W  =  dig_w · C + claim_w                                          q_W ∈ [0, C · L)
```

so `c_W` ranges over `[0, C · L · B) = [0, w_len / D)`. The decoding of `c`
into `(claim_w, b_w, dig_w)` is therefore:

```text
dig_w   =  c mod L
b_w     = (c / L) mod B
claim_w =  c / (B · L)
q_W     =  dig_w · C + claim_w
b_W(c)  =  b_w
```

This `c_W` is exactly what `WMatrixRowsEvaluator::compute_inner_sum` indexes
into via

```rust
lane_offset = claim_idx · per_claim_matrix_width + digit;
col         = block_idx · num_digits + lane_offset;
```

(`slice_mle.rs:477–490`).

### 3.2 The T reshaping (B matrix)

From `t-b-computation.md` §5 and §7, for `T`'s view:

```text
c_T( (claim_within_group, a_row, dig_t), b_t )
  =  claim_within_group · (n_a · L · B)
   + b_t                 · (n_a · L)
   + a_row               · L
   + dig_t
```

with the outer index

```text
q_T  =  claim_t + C · dig_t + C · L · a_row                          q_T ∈ [0, C · L · n_a)
```

so `c_T` ranges over `[0, C · n_a · L · B) = [0, t_len / D)`. The decoding
of `c` into `(claim_t, b_t, a_row_t, dig_t)` is:

```text
dig_t   =  c mod L
a_row_t = (c / L) mod n_a
b_t     = (c / (n_a · L)) mod B
claim_t =  c / (n_a · L · B)
q_T     =  claim_t + C · dig_t + C · L · a_row_t
b_T(c)  =  b_t
```

For multi-group settings the column index uses `claim_within_group` rather
than the flat `claim_t`; in single-group (production) settings these are
equal. The group dependence is carried by the **row weight** `b_weight[r, g]`,
not by the column index. See Section 11.

This `c_T` is exactly what `TMatrixRowsEvaluator::compute_inner_sum` indexes
into via

```rust
lane_offset = claim_idx_within_group * t_cols_per_claim
            + a_row_idx * num_digits
            + digit;
col         = block_idx * t_compound_per_block + lane_offset;
```

(`slice_mle.rs:668–684`).

### 3.3 W's range is a prefix of T's range

`matrix-rows-fusion.md` §2 already establishes:

```text
r_eval :  |--------- W slice (w_len/D cells) ---------|--- T-only ---|
          0                                         w_len/D       t_len/D
```

i.e. `c_W` always lands in `[0, w_len / D)` and `c_T` ranges over
`[0, t_len / D)`. Concretely `w_len / D = C · B · L` and
`t_len / D = C · n_a · B · L`, so the W-slice is the first
`(1 / n_a)`-fraction of the T-slice. The reshapings, however, are **not**
the same on shared cells: a given `c < w_len/D` is mapped to one
`(q_W, b_W)` by W's decode and to a generally different `(q_T, b_T)` by T's
decode. See §3.4.

### 3.4 The two reshapings disagree on shared cells

Take the cell `c = b_w · L + dig_w` (with `claim_w = 0`). Under the W decode:

```text
W:   (claim_w, b_w, dig_w)
```

Under the T decode the same `c` falls into:

```text
T:   claim_t = 0
     dig_t   = c mod L            =  dig_w
     a_row_t = (c / L) mod n_a    =  b_w mod n_a
     b_t     = (c / (n_a · L)) mod B  =  b_w / n_a
```

So `b_w ≠ b_t` in general (they differ by a factor of `n_a`), `q_W ≠ q_T`,
and consequently the low/high split bits `(low_idx, carry)` differ between
the two reshapings — even on the same cell. This is exactly the obstruction
identified in `matrix-rows-fusion.md` §5.

The structural consequence: `Eval[r, c]` must independently account for **W's
reshaping** and **T's reshaping** of the column axis, with separate
`eq_x(...)` factors for the two. They cannot be fused into a single `eq_x`.

## 4. Per-Row Formulas (Recap)

We restate the per-row decomposition from `matrix-rows-fusion.md` §3, with
the column-index reshapings of §3 written explicitly.

### 4.1 `W_row(r)`

From `w-d-computation.md` §5:

```text
W_row(r)
  =  Σ_{q_W ∈ [0, C·L)}  Σ_{b_w ∈ [0, B)}
       M_Flat[r, c_W(q_W, b_w)]
       · eq_x( offset_w + b_w + B · q_W )                              (2)
```

so

```text
w_d  =  Σ_{r=0}^{n_d - 1}  d_weight[r] · W_row(r)                       (3)
```

### 4.2 `T_row(r)`

From `t-b-computation.md` §5:

```text
T_row(r; group g)
  =  Σ_{q_T ∈ [0, C·L·n_a)}  Σ_{b_t ∈ [0, B)}
       M_Flat[r, c_T(q_T, b_t)]
       · eq_x( offset_t + b_t + B · q_T )                              (4)
```

with the per-row weight in the multi-group setting depending on the
claim's commitment group:

```text
t_b  =  Σ_{r=0}^{n_b - 1}  Σ_{q_T, b_t}
          b_weight[r, group(claim_T(q_T))]
          · M_Flat[r, c_T(q_T, b_t)]
          · eq_x( offset_t + b_t + B · q_T )                            (5)
```

In the single-group case `b_weight[r, g]` is just `b_weight[r]` and can be
pulled out of the inner sums:

```text
t_b  =  Σ_{r=0}^{n_b - 1}  b_weight[r] · T_row(r)
```

(parallel to (3)). The structural formulas below cover the general
multi-group case; single-group is a clean specialisation.

## 5. Loop Inversion: from Evaluator-Owned to Cell-Owned

The existing implementation iterates over `(q, b)` in two separate
evaluators (`W` and `T`), each of which recomputes `eval_alpha(matrix[r,c])`
on the fly per cell. The matrix–vector view inverts the iteration: iterate
over the **cells** `(r, c)` of `M_Flat`, and at each cell accumulate **the
weight that cell carries in the final answer**.

Concretely, swap the order of summation in (3) + (5):

```text
w_d + t_b
  =  Σ_{r=0}^{R-1}  Σ_{c=0}^{N-1}
       M_Flat[r, c]
       · ( W_part(r, c)  +  T_part(r, c) )                              (6)
```

where:

```text
W_part(r, c)
  =  𝟙[r < n_d] · 𝟙[c < w_len / D]
     · d_weight[r]
     · eq_x( offset_w + b_W(c) + B · q_W(c) )                           (7)

T_part(r, c)
  =  𝟙[r < n_b]
     · b_weight[ r, group(claim_T(c)) ]
     · eq_x( offset_t + b_T(c) + B · q_T(c) )                           (8)
```

The indicator functions reflect the SIS-row participation rules of
`matrix-rows-fusion.md` §10:

- `𝟙[r < n_d]` — row `r` participates in `D`'s rows (and hence in `w_d`).
- `𝟙[r < n_b]` — row `r` participates in `B`'s rows (and hence in `t_b`).
- `𝟙[c < w_len / D]` — only the W prefix of the column axis contributes to
  `w_d`; columns `c ≥ w_len/D` are T-only.

The map `c → (q_W(c), b_W(c))` is W's decode of §3.1; the map
`c → (q_T(c), b_T(c), claim_T(c))` is T's decode of §3.2.

## 6. The `Eval` Matrix

Define `Eval ∈ F^{R × N}` by

```text
Eval[r, c]  =  W_part(r, c)  +  T_part(r, c)                            (9)
```

with `W_part` and `T_part` exactly as in (7)–(8).

Then by (6):

```text
w_d + t_b  =  Σ_{r=0}^{R-1} Σ_{c=0}^{N-1}  M_Flat[r, c] · Eval[r, c]
           =  <M_Flat, Eval>                                            (10)
```

This is the identity the user is after. Everything in `Eval[r, c]` is
verifier-known:

| symbol            | source on the verifier                               |
|-------------------|------------------------------------------------------|
| `d_weight[r]`     | `eq_tau1[d_start + r]` (`slice_mle.rs:809`)          |
| `b_weight[r, g]`  | `eq_tau1[b_start + g · n_b + r]` (`slice_mle.rs:806`)|
| `claim_to_group`  | `prepared.claim_to_group`                             |
| `offset_w`, `offset_t` | derived in `EvalAtPointWorkspace::build`        |
| `eq_x(·)`         | pointwise eval of `EqPolynomial` at `full_vec_randomness` |

So the verifier can construct `Eval` from public data and randomness alone,
exactly as it constructs the inner-sum carries in
`{W,T}MatrixRowsEvaluator::compute_inner_sum` today.

## 7. Correctness Argument

Plug (7)–(8) into (10) and split the sum:

```text
<M_Flat, Eval>
  =  Σ_{r, c}  M_Flat[r, c] · W_part(r, c)
   + Σ_{r, c}  M_Flat[r, c] · T_part(r, c)
```

For the W-half: the indicators restrict the sum to `r < n_d` and
`c < w_len/D`. Within that range, replacing `c` by its W-decoded triple
`(claim_w, b_w, dig_w)` is a bijection onto `[0, C) × [0, B) × [0, L)`, so

```text
Σ_{r < n_d} d_weight[r] · Σ_{c < w_len/D}  M_Flat[r, c] · eq_x( offset_w + b_w(c) + B · q_W(c) )
  =  Σ_{r < n_d}  d_weight[r] · Σ_{q_W, b_w}  M_Flat[r, c_W(q_W, b_w)] · eq_x( offset_w + b_w + B · q_W )
  =  Σ_{r < n_d}  d_weight[r] · W_row(r)             [by (2)]
  =  w_d                                              [by (3)]
```

For the T-half: the indicator restricts the sum to `r < n_b`. Within that
range, replacing `c` by its T-decoded tuple `(claim_t, b_t, a_row_t, dig_t)`
is a bijection onto `[0, C) × [0, B) × [0, n_a) × [0, L)`, so

```text
Σ_{r < n_b}  Σ_{c}  b_weight[r, group(claim_T(c))] · M_Flat[r, c] · eq_x( offset_t + b_T(c) + B · q_T(c) )
  =  Σ_{r < n_b}  Σ_{q_T, b_t}  b_weight[r, group(claim_T(q_T))] · M_Flat[r, c_T(q_T, b_t)] · eq_x( offset_t + b_t + B · q_T )
  =  t_b                                              [by (5)]
```

Sum the two halves:

```text
<M_Flat, Eval> = w_d + t_b
```

which is (10). The cross-term obstruction of `matrix-rows-fusion.md` §5.3
does **not** appear because we have not tried to multiply a single
"inner sum" by a single `(eq_hi_w + eq_hi_t)` — instead we **add** two
independently-weighted equality contributions per cell.

## 8. Reusing the Low/High Equality Split

The naive way to compute `Eval[r, c]` evaluates `eq_x(·)` from scratch at
each `(r, c)`. That costs `Θ(R · N · |full_vec_randomness|)` field ops,
which is much worse than the current implementation.

The cheap way reuses the existing low/high equality split (the design at the
heart of `slice_mle.rs` and documented in `generic-offset-eq-block-design.md`).

Both `offset_w` and `offset_t` are aligned so that

```rust
debug_assert_eq!(w_offset_low, t_offset_low);
```

(`slice_mle.rs:1235`). Let `block_bits = log2(B)`. Then for both W's and T's
indices we can write

```text
eq_x( offset_* + b + B · q )
  =  eq_low( low_idx(b) )
   · eq_hi( offset_high_* + q + carry(b) )                              (11)
```

with

```text
offset_low      = offset_w & (B - 1)  = offset_t & (B - 1)
offset_high_w   = offset_w >> block_bits
offset_high_t   = offset_t >> block_bits
low_idx(b)      = (offset_low + b) mod B
carry(b)        = (offset_low + b) div B   ∈ {0, 1}
```

`eq_low` is a small table of size `B`; the verifier materialises it once
(`EqPolynomial::evals(full_vec_randomness[..block_bits])`) and reuses it for
**all four** of `w_sep`, `w_d`, `t_sep`, `t_b` — including our fused
`Eval`. `eq_hi` is never materialised; it is evaluated pointwise at the
few `(offset_high + q + carry)` indices that actually appear.

Applied to `Eval[r, c]`, this gives:

```text
W_part(r, c) =  𝟙[r < n_d, c < w_len/D]
              · d_weight[r]
              · eq_low( low_idx_W(c) )
              · eq_hi_w( offset_high_w + q_W(c) + carry_W(c) )         (12)

T_part(r, c) =  𝟙[r < n_b]
              · b_weight[ r, group(claim_T(c)) ]
              · eq_low( low_idx_T(c) )
              · eq_hi_t( offset_high_t + q_T(c) + carry_T(c) )         (13)
```

The two `eq_low(·)` factors above use the **same table** but generally **different
arguments**, because `b_W(c) ≠ b_T(c)`. Likewise the two `eq_hi` factors are
evaluated at different points (different `offset_high`, different `q`,
different `carry`). This is the algebraic price of the reshaping mismatch —
it shows up cleanly as two additive equality terms per cell, which is
exactly what `Eval[r, c]` already carries.

## 9. Fast Materialisation of `Eval` via Separable Structure

§8 expresses each cell of `Eval` as a product of `eq_low` (small, shared)
and `eq_hi` (evaluated on demand). Crucially, **both factors are functions
of `c` only**. The per-row factor (`d_weight[r]` / `b_weight[r, g]`) is
just a scalar that depends on `r`. So `Eval` is the sum of **column-only
patterns weighted by row scalars** — i.e. a low-rank outer-product
structure, not a generic `R × N` matrix that needs `R · N` independent
equality evaluations.

This section makes the structure explicit and gives a construction whose
total field-op cost (post-`M_Flat`) is

```text
O( R · N  +  num_q_T · |high_challenges| )
```

— the same order as the streamed production form of §10, but producing
one materialised `Eval` matrix that can be fed straight into a single
`<M_Flat, Eval>` inner product. The "naive `eq_x` per cell" approach of
§11.1 is `Θ(R · N · |full_vec_randomness|)`; this section is what fixes it
without giving up the one-`Eval` form.

### 9.1 The Column Patterns

Substitute (12)–(13) into the definition `Eval = W_part + T_part`. The
`eq_low(·)` and `eq_hi(·)` factors depend only on `c`; pull them out as
column-only "patterns":

```text
w_pattern[c]  :=  eq_low( low_idx_W(c) )
                · eq_hi_w( off_hi_w + q_W(c) + carry_W(c) )            for c ∈ [0, w_len/D)    (14)

t_pattern[c]  :=  eq_low( low_idx_T(c) )
                · eq_hi_t( off_hi_t + q_T(c) + carry_T(c) )            for c ∈ [0, N)          (15)
```

Then (12)–(13) become:

```text
W_part(r, c)  =  𝟙[r < n_d] · 𝟙[c < w_len/D]  · d_weight[r]            · w_pattern[c]
T_part(r, c)  =  𝟙[r < n_b]                       · b_weight[r, g(c)]  · t_pattern[c]
```

and

```text
Eval[r, c]
  =  𝟙[r < n_d, c < w_len/D] · d_weight[r]      · w_pattern[c]
   + 𝟙[r < n_b]                  · b_weight[r, g(c)] · t_pattern[c]                            (16)
```

where `g(c)` is shorthand for `group(claim_T(c)) = claim_to_group[claim_T(c)].0`.

### 9.2 `Eval` Is a Low-Rank Outer Product

**Single-group case** (production): `g(c) = 0` for all `c`, so
`b_weight[r, g(c)] = b_weight[r]` is independent of `c`. Define the
zero-padded W pattern:

```text
w_pattern_ext[c]  =  {  w_pattern[c]   if c < w_len/D
                     {  0              otherwise
```

Then `Eval` is literally a sum of two outer products:

```text
Eval  =  d_weight ⊗ w_pattern_ext  +  b_weight ⊗ t_pattern                                     (17)
```

where `(u ⊗ v)[r, c] := u[r] · v[c]`. Both summands are rank-1; `Eval`
itself is rank-2.

**Multi-group case**: define a group-restricted T pattern per commitment
group:

```text
t_pattern_g[c]  =  {  t_pattern[c]   if g(c) = g
                   {  0              otherwise
```

so

```text
b_weight[r, g(c)] · t_pattern[c]  =  Σ_{g = 0}^{num_groups − 1}  b_weight[·, g][r]  ·  t_pattern_g[c]
```

and (17) generalises to:

```text
Eval  =  d_weight ⊗ w_pattern_ext  +  Σ_{g = 0}^{num_groups − 1}  b_weight[·, g] ⊗ t_pattern_g  (18)
```

— a rank-`(1 + num_groups)` outer product. Production uses (17).

The structural punchline: `Eval` is fully described by **two short per-row
vectors** of length `R` (`d_weight`, `b_weight`) and **at most
`1 + num_groups` short per-column vectors** of length `N`
(`w_pattern_ext`, `t_pattern_g`). The `R × N` matrix is just the
rank-`(1 + num_groups)` reconstruction.

### 9.3 Precomputing the `eq_hi` Slices

`w_pattern` and `t_pattern` need `eq_hi` evaluated at consecutive integer
offsets around `off_hi_w` and `off_hi_t`. The `carry ∈ {0, 1}` slot of
(12)–(13) widens each range by 1:

```text
eq_hi_w_table[k]  :=  eq( high_challenges,  off_hi_w + k )    for k ∈ [0, num_q_W + 1)
eq_hi_t_table[k]  :=  eq( high_challenges,  off_hi_t + k )    for k ∈ [0, num_q_T + 1)
```

Lengths:

```text
|eq_hi_w_table|  =  num_q_W + 1  =  C · L + 1
|eq_hi_t_table|  =  num_q_T + 1  =  C · L · n_a + 1
```

Each entry is one multilinear equality evaluation at an integer point:

```text
eq( r, n )  =  Π_{j < |r|}  ( r_j   if  bit_j(n) = 1   else   1 − r_j )
```

costing `|high_challenges|` muls per entry. Total:

```text
cost( precompute eq_hi tables )
  =  ( num_q_W + num_q_T + 2 ) · |high_challenges|
  =  O( num_q_T · |high_challenges| )
```

This is the **same** `eq_hi` work the current `compute_outer_sum` passes
already do (see `SliceMleEvaluator::compute_outer_sum` in
`slice_mle.rs`), just reorganised into two upfront table builds rather
than spread across two outer-sum loops. There are no additional `eq`
evaluations.

At NV=32 OneHot D=32 single-claim
(`num_q_T = C · L · n_a = 1 · 64 · 3 = 192`,
`|high_challenges| ≈ 32 − log₂(2048) = 21`), that is `~260 · 21 ≈ 5500`
field muls — negligible compared to the rest of the verifier work.

The two tables can also be **shared with the streamed `compute_outer_sum`
paths** of §10 if both flavours ever coexist: the streamed form evaluates
the same `eq_hi` at the same indices, just on demand. Building the tables
once upfront eliminates the only redundant `eq` work between the two
algorithm flavours.

### 9.4 Building `w_pattern` and `t_pattern`

Each `w_pattern[c]` and `t_pattern[c]` is **two table lookups and one
multiplication**. Under W's reshaping (§3.1):

```text
dig_w        =  c mod L
b_w          =  (c / L) mod B
claim_w      =  c / (B · L)
q_W          =  dig_w · C + claim_w
sum          =  offset_low + b_w
low_idx_W    =  sum & (B − 1)
carry_W      =  sum >> block_bits                ∈ {0, 1}
w_pattern[c] =  eq_low[ low_idx_W ] · eq_hi_w_table[ q_W + carry_W ]
```

Under T's reshaping (§3.2):

```text
dig_t        =  c mod L
a_row_t      =  (c / L) mod n_a
b_t          =  (c / (n_a · L)) mod B
claim_t      =  c / (n_a · L · B)
q_T          =  claim_t + C · dig_t + C · L · a_row_t
sum          =  offset_low + b_t
low_idx_T    =  sum & (B − 1)
carry_T      =  sum >> block_bits
t_pattern[c] =  eq_low[ low_idx_T ] · eq_hi_t_table[ q_T + carry_T ]
```

Both are `O(1)` per cell:

```text
cost( build w_pattern )  =  w_len/D  =  C · L · B           muls
cost( build t_pattern )  =  N        =  C · L · n_a · B     muls
```

Total `O(N)` muls — a single linear pass over the column axis, parallel
over `c`. No randomness `r_j` is touched here; everything that depends on
`high_challenges` was pre-baked into the `eq_hi` tables in §9.3.

The two patterns can be built in one fused pass over `c ∈ [0, N)` since
they share the `c mod L` and `c & (B − 1)` arithmetic — though clarity
suggests keeping them separate.

### 9.5 Materialising `Eval` From the Patterns

Equation (16) reduces to **two table lookups and one fused multiply-add
per cell**:

```text
for r in 0..R:                                              [parallel over r]
  for c in 0..N:                                            [parallel over c]
    let mut v = E::zero()
    if r < n_d  and  c < w_len/D:
        v += d_weight[r] · w_pattern[c]
    if r < n_b:
        let g = group_of_c(c)        // single integer divide; constant over each group span
        v += b_weight[r, g] · t_pattern[c]
    Eval[r, c] = v
```

Cost:

```text
cost( materialise Eval )
  ≤  2 · R · N     muls
   + R · N         adds
```

This is the **only** stage that pays an `R × N` price for `Eval`. There
is no per-cell `eq_x` call, no high-bit-challenge loop, no `eq_low`
recomputation — all of that has been pre-baked into `w_pattern[c]` and
`t_pattern[c]`. The per-cell work is two `Vec<E>` lookups plus a fused
multiply-add — the minimum possible for a rank-`(1 + num_groups)`
outer-product reconstruction.

`group_of_c(c)` is a single integer divide:

```text
group_of_c(c)  =  claim_to_group[ c / (n_a · L · B) ].0
```

In single-group mode this collapses to a constant. In multi-group mode it
is piecewise constant — the same value for all `c` in the same
`claim_T(c)` block — so it can be hoisted outside the inner `c` loop with
a stride-`n_a · L · B` step.

### 9.6 Final Inner Product

`<M_Flat, Eval>` is now a literal element-wise inner product over a flat
`R × N` array:

```text
result = E::zero()
for r in 0..R:                                              [parallel over r]
  // Build M_Flat[r, ·] once (row-streamed; never holds the full matrix).
  let r_eval = (0..N).map(|c| eval_ring_at_pows(&shared_matrix.row(r)[c], &alpha_pows))
                     .collect::<Vec<E>>()
  for c in 0..N:
    result += r_eval[c] · Eval[r, c]
return result
```

`r_eval` is what `matrix-rows-fusion.md` §8 calls the per-row cache; it
IS `M_Flat[r, ·]`. The verifier never holds the full `R × N` `M_Flat` in
memory — only one row at a time.

### 9.7 Total Cost

Putting §9.3–§9.6 together:

```text
total field-op cost (post-M_Flat construction)
  =  cost( precompute eq_hi tables )            [ O( num_q_T · |high| ) ]
   + cost( build w_pattern, t_pattern )         [ O( N ) ]
   + cost( materialise Eval )                   [ O( R · N ) ]
   + cost( final inner product )                [ O( R · N ) ]
  =  O( R · N  +  num_q_T · |high_challenges| )
```

Ring-evaluation cost (to build `M_Flat`, the dominant overall cost) is
`O(R · N · D)` `mul_base` ops, exactly the same as the streamed form of
§10.

This is **asymptotically equal** to the existing streamed form, with a
small constant factor (`~3 · R · N` extra muls: 2 to build `Eval`, 1 to
fold). That overhead is dwarfed by the ring-evaluation cost
`R · N · D · mul_base`, and is trivially parallelisable along both axes.

### 9.8 Memory Trade-Off: Materialised vs Row-Streamed `Eval`

Two equally valid storage shapes for `Eval`, differing only in memory:

**Full materialisation** (`Eval` resident, `R · N · sizeof(E)` bytes):

```rust
let eval: Vec<Vec<E>> = (0..R).into_par_iter()
    .map(|r| build_eval_row(r, &w_pattern, &t_pattern, &d_weight, &b_weight))
    .collect();
let result = (0..R).into_par_iter()
    .map(|r| {
        let r_eval = build_m_flat_row(r);
        (0..N).map(|c| r_eval[c] * eval[r][c]).sum::<E>()
    })
    .sum::<E>();
```

`M_Flat` is still row-streamed (we never need it all at once), but `Eval`
is held in full. At NV=32 OneHot D=32 single-claim
(`R = 2, N ≈ 393 216, sizeof(E) = 16 B`) this is `~12.6 MB`. At 4 claims
it grows to `~50 MB`. Acceptable on most hosts, but not free.

**Row-streamed materialisation** (one `Eval` row resident at a time,
`N · sizeof(E)` bytes):

```rust
let result: E = (0..R).into_par_iter()
    .map(|r| {
        let r_eval = build_m_flat_row(r);
        let e_row  = build_eval_row(r, &w_pattern, &t_pattern, ...);   // one row only
        (0..N).map(|c| r_eval[c] * e_row[c]).sum::<E>()
    })
    .sum::<E>();
```

Same arithmetic, but `Eval[r, ·]` is computed and consumed inside one row
iteration, never held across rows. Peak resident memory is dominated by
`w_pattern` + `t_pattern` (length `N`, ~`6 MB` at NV=32 OneHot D=32
single-claim) rather than `R · N`.

The `<M_Flat, Eval>` identity holds in both shapes; only the storage
layout differs.

### 9.9 Relationship to the Streamed Form of §10

The streamed form of §10 — i.e. the current `WMatrixRowsEvaluator` /
`TMatrixRowsEvaluator` path — is mathematically the same expression as §9
with **one further optimisation step**: collapsing the `Eval`-side outer
product into per-`q` carry buffers before contracting with `M_Flat`.

Concretely: instead of materialising `Eval[r, c]` and then doing
`Σ_c M_Flat[r, c] · Eval[r, c]`, the streamed form rewrites the inner sum
as a `b`-grouped sum so that the
`eq_hi(off_hi + q + carry)` factor (the same for all `b` in one carry
class) can be hoisted out as one multiplication per `q`. This saves the
`2 · R · N` materialisation muls and replaces them with
`O(num_q_W + num_q_T)` outer-sum muls plus `~R · N` inner accumulations
into the carry buffers.

Both forms compute `<M_Flat, Eval>` exactly. The choice between them is
purely about loop structure and `Eval` storage; the algebra is identical.

## 10. Recovering the Existing Inner-Sum / Outer-Sum Algorithm

The materialised path of §9 builds `Eval` explicitly. The streamed path —
the existing `WMatrixRowsEvaluator` / `TMatrixRowsEvaluator` — instead
collapses the `Eval` side into per-`q` carry buffers and never holds the
`R × N` matrix in memory. This section derives the streamed form from
(10), (12)–(13) by hoisting the `eq_hi` factor out of the `b` axis. It is
the original form `matrix-rows-fusion.md` §8 describes.

### 10.1 Hoisting `eq_hi` out of `b` for `q_W` (W half)

Fix `r < n_d` and `q_W`. The cells with this `q_W` are
`c = c_W(q_W, b_w)` for `b_w ∈ [0, B)`. The W-half contribution from these
cells to `<M_Flat, Eval>` is

```text
d_weight[r] · Σ_{b_w}  M_Flat[r, c_W(q_W, b_w)] · eq_low( low_idx_W(b_w) ) · eq_hi_w( off_hi_w + q_W + carry_W(b_w) )
```

Group `b_w` by its carry slot (0 or 1) and pull the `eq_hi` factor out:

```text
=  d_weight[r] · [
       carry_w_0[r, q_W]  · eq_hi_w( off_hi_w + q_W     )
     + carry_w_1[r, q_W]  · eq_hi_w( off_hi_w + q_W + 1 )
   ]
```

with

```text
carry_w_c[r, q_W]
  =  Σ_{b_w : carry_W(b_w) = c}  M_Flat[r, c_W(q_W, b_w)] · eq_low( low_idx_W(b_w) )
```

This is exactly `summarize_strided_pow2_block_carries` applied to row `r`
of `M_Flat` with stride `L` and lane offset
`lane_W(q_W) = claim_w · B · L + dig_w` — i.e. the body of
`WMatrixRowsEvaluator::compute_inner_sum` (`slice_mle.rs:474–497`), only
reading `M_Flat[r, ·]` (pre-evaluated) instead of `d_view.row(r)` (evaluated
on the fly via `eval_ring_at_pows`).

### 10.2 Hoisting `eq_hi` out of `b` for `q_T` (T half)

Symmetrically, fixing `r < n_b` and `q_T`:

```text
[T-half contribution from this (r, q_T)]
  =  b_weight[r, group(claim_T(q_T))] · [
         carry_t_0[r, q_T]  · eq_hi_t( off_hi_t + q_T     )
       + carry_t_1[r, q_T]  · eq_hi_t( off_hi_t + q_T + 1 )
     ]
```

with

```text
carry_t_c[r, q_T]
  =  Σ_{b_t : carry_T(b_t) = c}  M_Flat[r, c_T(q_T, b_t)] · eq_low( low_idx_T(b_t) )
```

This is `summarize_strided_pow2_block_carries` with stride `n_a · L` and
lane offset
`lane_T(q_T) = claim_within_group · n_a · L · B + a_row · L + dig_t` — i.e.
the body of `TMatrixRowsEvaluator::compute_inner_sum`
(`slice_mle.rs:659–689`), again reading `M_Flat[r, ·]` instead of
`b_view.row(r)`.

### 10.3 Final outer sums (same as today)

After the per-row, per-`q` carry summaries are accumulated across rows into
two shared buffers,

```text
W_carry[q_W][c]   =  Σ_{r < n_d}  d_weight[r] · carry_w_c[r, q_W]
T_carry[q_T][c]   =  Σ_{r < n_b}  b_weight[r, group(claim_T(q_T))] · carry_t_c[r, q_T]
```

(precisely the buffers the current code already builds in
`{W,T}MatrixRowsEvaluator::evaluate` via the trait's default
`compute_outer_sum`), the final scalar is

```text
w_d  =  Σ_{q_W}  ( W_carry[q_W][0] · eq_hi_w(off_hi_w + q_W    )
                 + W_carry[q_W][1] · eq_hi_w(off_hi_w + q_W + 1) )

t_b  =  Σ_{q_T}  ( T_carry[q_T][0] · eq_hi_t(off_hi_t + q_T    )
                 + T_carry[q_T][1] · eq_hi_t(off_hi_t + q_T + 1) )

w_d + t_b  =  <M_Flat, Eval>
```

So the fused matrix–vector identity (10) decomposes — losslessly — back into
two separate `compute_outer_sum` passes (as observed in
`matrix-rows-fusion.md` §11). Fusion changes the **loop nest**; it does
**not** fuse the high-bit equality work.

## 11. The Verifier Algorithm, Step by Step

Putting §9 and §10 together yields three concrete algorithms for
`<M_Flat, Eval> = w_d + t_b`, in increasing optimisation order. All three
return the same scalar.

### 11.1 Naive (proof-of-correctness)

```text
build M_Flat:
  for r in 0..R:
    for c in 0..N:
      M_Flat[r, c] = eval_ring_at_pows( shared_matrix.row(r) [c], alpha_pows )

build Eval:
  for r in 0..R:
    for c in 0..N:
      v = 0
      if r < n_b:
        (claim_t, b_t, a_row_t, dig_t) = T_decode(c)
        q_T = claim_t + C · dig_t + C · L · a_row_t
        g   = claim_to_group[claim_t].0
        v  += b_weight[r, g] · eq_x( offset_t + b_t + B · q_T )
      if r < n_d  and  c < w_len/D:
        (claim_w, b_w, dig_w) = W_decode(c)
        q_W = dig_w · C + claim_w
        v  += d_weight[r] · eq_x( offset_w + b_w + B · q_W )
      Eval[r, c] = v

fold:
  acc = 0
  for r in 0..R:
    for c in 0..N:
      acc += M_Flat[r, c] · Eval[r, c]
  return acc
```

Cost: `R · N` ring evaluations to build `M_Flat`, `R · N · |x_challenges|`
field ops to build `Eval`, `R · N` muls to fold. The `Eval` build is the
bad one — `Θ(R · N · |x_challenges|)`. The next two algorithms each fix
it differently. §11.2 is the materialised-`Eval` form (uses §9). §11.3 is
the streamed-`Eval` form (uses §10).

### 11.2 Materialised `Eval` from `(eq_low, eq_hi)` patterns (uses §9)

This is the form the user wants: one `M_Flat`, one `Eval`, one inner
product. The trick is to never compute `eq_x` from scratch per cell — use
§9's pattern factorisation instead.

```text
precompute (small):
  eq_low                     : Vec<E> of length B
                               (= EqPolynomial::evals(full_vec_randomness[..block_bits]))

  eq_hi_w_table[k]           : eq( high_challenges,  off_hi_w + k )  for k ∈ [0, num_q_W + 1)
  eq_hi_t_table[k]           : eq( high_challenges,  off_hi_t + k )  for k ∈ [0, num_q_T + 1)

  w_pattern[c]  for c ∈ [0, w_len/D):
                let (dig_w, b_w, claim_w) = W_decode(c)
                let q_W      = dig_w · C + claim_w
                let sum      = offset_low + b_w
                let low_idx  = sum & (B − 1)
                let carry    = sum >> block_bits
                w_pattern[c] = eq_low[low_idx] · eq_hi_w_table[q_W + carry]

  t_pattern[c]  for c ∈ [0, N):
                let (dig_t, a_row_t, b_t, claim_t) = T_decode(c)
                let q_T      = claim_t + C · dig_t + C · L · a_row_t
                let sum      = offset_low + b_t
                let low_idx  = sum & (B − 1)
                let carry    = sum >> block_bits
                t_pattern[c] = eq_low[low_idx] · eq_hi_t_table[q_T + carry]

result = E::zero()

for r in 0..R:                                            [parallel over r]
  // Build M_Flat[r, ·] once.
  let r_eval = (0..N).into_par_iter()
                     .map(|c| eval_ring_at_pows(&shared_matrix.row(r)[c], &alpha_pows))
                     .collect::<Vec<E>>();

  // Build Eval[r, ·] inline, fold into one row dot product.
  // (Or hold the full Eval[r] in a Vec<E> first if you prefer two passes.)
  let row_contribution: E = (0..N).into_par_iter().map(|c| {
      let mut e = E::zero();
      if r < n_d && c < w_len_div_d {
          e += d_weight[r] · w_pattern[c];
      }
      if r < n_b {
          let g = claim_to_group[ c / (n_a · L · B) ].0;
          e += b_weight[r, g] · t_pattern[c];
      }
      r_eval[c] · e
  }).sum();

  result += row_contribution;

return result
```

Costs:

```text
precompute eq_hi tables : O( num_q_T · |high_challenges| )
build patterns          : O( N )
ring evals for M_Flat   : R · N        eval_ring_at_pows  (= R · N · D mul_base)
field muls for Eval+fold: O( R · N )    (≤ 3 · R · N)
```

Memory: peak `O( N · sizeof(E) )` for the two patterns and one row of
`r_eval`. The `R × N` `Eval` matrix never needs to be resident — it is
folded into `row_contribution` inside the row loop. If you do choose to
materialise it for inspection or testing, it costs `R · N · sizeof(E)`
bytes; the arithmetic is identical.

This is the "one M_Flat × one Eval = scalar" code shape, made fast by
factoring the `eq` work out of the cell loop and into two small
precomputed slices.

### 11.3 Streamed `Eval` (matches the existing `compute_outer_sum` paths)

The verifier never materialises the full `Eval` matrix; instead it folds
the carry summaries from §10 directly into `W_carry` / `T_carry` buffers
and runs `compute_outer_sum` once for each. This is the existing
production path:

```text
materialise:
  eq_low                   : Vec<F> of length B  (shared across w_sep, w_d, t_sep, t_b)
  W_carry[q_W][carry]      : zero of length 2 · C · L
  T_carry[q_T][carry]      : zero of length 2 · C · L · n_a

for r in 0..R:                                    [parallel]
  r_eval = build_M_Flat_row(r)                    [N ring evals; built in parallel over c]

  if r < n_d:
    dw = d_weight[r]
    for q_W in 0..(C · L):                        [parallel]
      lane_W = decode_lane_W(q_W)
      [carry0, carry1] =
        summarize_strided_pow2_block_carries_view(
          eq_low, offset_low, r_eval, num_blocks=B, block_stride=L, lane=lane_W
        )
      W_carry[q_W][0] += dw · carry0
      W_carry[q_W][1] += dw · carry1

  if r < n_b:
    for q_T in 0..(C · L · n_a):                  [parallel]
      (claim_t, a_row, dig_t) = decode_outer_T(q_T)
      (g, claim_in_g)         = claim_to_group[claim_t]
      bw = b_weight[r, g]
      lane_T = claim_in_g · (n_a · L · B) + a_row · L + dig_t
      [carry0, carry1] =
        summarize_strided_pow2_block_carries_view(
          eq_low, offset_low, r_eval, num_blocks=B, block_stride=n_a · L, lane=lane_T
        )
      T_carry[q_T][0] += bw · carry0
      T_carry[q_T][1] += bw · carry1

w_d = compute_outer_sum_high(W_carry, off_hi_w, high_challenges)
t_b = compute_outer_sum_high(T_carry, off_hi_t, high_challenges)
return w_d + t_b
```

The only difference from today is the introduction of `r_eval` as a per-row
cache — exactly the per-row pipeline of `matrix-rows-fusion.md` §8. The
`summarize_strided_pow2_block_carries_view` here is a thin wrapper that
takes a pre-evaluated row of field elements (a slice of `r_eval`) instead
of a row of cyclotomic ring elements; the arithmetic is otherwise identical
to `summarize_strided_pow2_block_carries` (`ring_switch.rs:377`).

Note `r_eval[c]` **is** `M_Flat[r, c]`. The "build `M_Flat`" step is fused
with the row iteration — the verifier never has to allocate the whole
`R × N` matrix, only one row at a time. The peak resident memory cost is
`N · sizeof(F)` rather than `R · N · sizeof(F)`.

### 11.4 Notes on parallelism

`r` is the outer parallel axis: rows are independent. Within a row, both the
`q_W` and `q_T` inner sweeps are also parallelisable (the existing
`{W,T}MatrixRowsEvaluator` already set `parallelize_outer() == true`). The
accumulation into `W_carry` / `T_carry` uses `Σ_r` over the row dimension; in
the per-row form this becomes a per-row `+=` that must be reduced across
the row-parallel threads (either lockless thread-local accumulators that
sum at the end, or a parallel reduce over a `Vec<[E; 2]>` of length
`R · num_q`). See `matrix-rows-fusion.md` §8 for the storage discussion.

## 12. Multi-Group Caveat

In single-group settings `b_weight[r, g] = b_weight[r]` and `claim_within_group =
claim_t`. Then `T_part(r, c)` factors cleanly:

```text
T_part(r, c)  =  𝟙[r < n_b] · b_weight[r] · eq_low(low_idx_T) · eq_hi_t(...)
```

so the row weight pulls out of the inner block sweep, and the `T` half of
the per-row pipeline reduces to one strided summary multiplied by a per-row
scalar — exactly the current `TMatrixRowsEvaluator` body with the matrix
swapped for `r_eval`.

In multi-group settings (`num_commitment_groups > 1`):

```text
b_weight[r, group(claim_t)] · M_Flat[r, c_T(...)] · eq_low(...) · eq_hi_t(...)
```

The `b_weight` factor varies along the `c` axis (specifically along the
`claim_t` portion of `c`), so we cannot pull it out of the inner block
sweep in general. Two clean options:

- **Slice `T_carry` per group.** Run one strided sweep per group, using only
  the slice of `claim_t` values whose `claim_to_group[claim_t].0 == g`,
  and weight that sweep by `b_weight[r, g]`. This matches today's
  `TMatrixRowsEvaluator` behaviour where each `compute_inner_sum` call
  picks the right group's row-weight window.
- **Per-`q_T` weight lookup.** Pull `b_weight[r, group(claim_T(q_T))]` into
  the outer `q_T` loop body. This is `O(1)` per `q_T` and is what
  `TMatrixRowsEvaluator::compute_inner_sum` does today (`slice_mle.rs:664–667`).

Either form is structurally fine; the `Eval[r, c]` definition (8) is
correct in both cases. The matrix–vector framing does not introduce any
new multi-group obligation beyond what `t_b` already required.

## 13. Boundary Cases: `n_d ≠ n_b`

The shared-row range is `r ∈ [0, min(n_d, n_b))`. Outside that range, only
one of the two indicators in (7)–(8) is non-zero:

| range of `r`             | W indicator | T indicator | what `M_Flat[r, ·]` is used for |
|--------------------------|-------------|-------------|----------------------------------|
| `r < min(n_d, n_b)`      | active      | active      | shared; full row of width `N` needed |
| `min(n_d, n_b) ≤ r < n_d`| active      | inactive    | only W prefix of width `w_len/D` needed |
| `min(n_d, n_b) ≤ r < n_b`| inactive    | active      | full row of width `N` needed |
| `r ≥ max(n_d, n_b)`      | inactive    | inactive    | row does not exist in `M_Flat` |

For rows where W is active but T is not, the verifier only needs
`M_Flat[r, c]` for `c < w_len/D` (a shorter row). For rows where T is
active but W is not, the verifier needs the full `c < N`. In production
`n_d = n_b`, so the first case is the relevant one and the row is always
full-width; the table above is the spec for the general case.

The indicators handle all four cases by construction, so the structural
identity (10) and both algorithms in §11.2 / §11.3 work unmodified — they
just skip the inactive half of `Eval[r, ·]` per row.

## 14. Cost Analysis

Let:

```text
R  = max(n_d, n_b)
m  = min(n_d, n_b)
N  = C · n_a · L · B
NW = C · L · B  =  N / n_a
```

### 14.1 Ring evaluations (the bottleneck)

| version                  | ring evals                                                | comment |
|--------------------------|------------------------------------------------------------|---------|
| baseline (separate eval) | `n_d · NW + n_b · N`                                       | `WMatrixRowsEvaluator` + `TMatrixRowsEvaluator` each call `eval_ring_at_pows` per cell touched |
| `M_Flat`-fused per row   | `m · N + (n_d - m) · NW + (n_b - m) · N`                    | one `eval_alpha` per cell per row, reused by both halves on shared rows |
| `M_Flat`-fused (n_d=n_b) | `n_d · N`                                                  | production: one row × N cells, both halves share |

For `n_d = n_b` (the production setup) the saving is exactly
`n_d · NW = n_d · C · L · B` ring evaluations, which is `D · n_d · C · L · B`
`mul_base` operations after the `eval_ring_at_pows` accounting in
`matrix-rows-fusion.md` §7.

This matches the saving promised in `matrix-rows-fusion.md` §6: **all of
W's per-row ring-evaluation budget vanishes** once `M_Flat[r, ·]` is built
once per row.

### 14.2 Field-element multiplications (post-evaluation)

**Streamed form (§11.3).** For each `r < n_d` we do `C · L` strided
block sweeps each touching `B` cells of `M_Flat[r, ·]`, multiplying by
`eq_low` and accumulating into two carry slots: `O(n_d · C · L · B)`
muls. Similarly for T: `O(n_b · C · L · n_a · B)` muls. These are the
same multiplication counts as the current `compute_inner_sum` paths;
nothing about the matrix–vector framing changes them.

The final `compute_outer_sum` passes are unchanged: each does
`O(num_q · |high_challenges|)` field ops for `eq_hi`, with
`num_q = C · L` (W) or `C · L · n_a` (T).

**Materialised form (§11.2).** The `eq_hi` work is upfront:
`O((num_q_W + num_q_T) · |high|)` muls in two table builds (§9.3). The
pattern build is `O(N)` muls (§9.4). The `Eval` materialisation is
`≤ 2 · R · N` muls (§9.5). The final fold is `R · N` muls (§9.6). Total:

```text
O( R · N  +  num_q_T · |high_challenges| )
```

— same big-O as the streamed form, with constants `~3 · R · N` muls
instead of `~R · N`. The extra constants are negligible against the
`R · N · D` `mul_base` cost to build `M_Flat`.

### 14.3 Memory

| object                   | size                                                | used by |
|--------------------------|-----------------------------------------------------|---------|
| `M_Flat` (all rows)      | `R · N · sizeof(F)`   — never needed in full        | conceptual |
| `r_eval` (one row)       | `N · sizeof(F)`        — one row at a time          | both forms |
| `eq_low`                 | `B · sizeof(E)`        — shared with `w_sep`, `t_sep` | both forms |
| `high_challenges`        | `(|x_challenges| - block_bits) · sizeof(E)`         | both forms |
| `eq_hi_w_table`          | `(num_q_W + 1) · sizeof(E)`                          | §11.2 |
| `eq_hi_t_table`          | `(num_q_T + 1) · sizeof(E)`                          | §11.2 |
| `w_pattern`              | `(w_len/D) · sizeof(E)`                              | §11.2 |
| `t_pattern`              | `N · sizeof(E)`                                       | §11.2 |
| `Eval` row (one)         | `N · sizeof(E)` (optional)                            | §11.2 row-streamed |
| `Eval` full              | `R · N · sizeof(E)` (optional)                        | §11.2 full materialisation |
| `W_carry`                | `2 · C · L · sizeof(E)`                              | §11.3 |
| `T_carry`                | `2 · C · L · n_a · sizeof(E)`                        | §11.3 |

`r_eval` is the per-row cache built at the top of each `r` iteration; it
is the analogue of the `Vec<E>` in `matrix-rows-fusion.md` §8.

At NV=32 OneHot D=32 single-claim (the numbers in
`matrix-rows-fusion.md` §7) `r_eval` is `~6 MB` per row in `Fp128` — fits
in L2/L3. Crucially, peak memory in the streamed form is **`r_eval`, not
`R · r_eval`**: we drop each row's cache before building the next. In the
materialised form (§11.2) the patterns add another `~12 MB` of resident
data (`w_pattern + t_pattern`), and full `Eval` materialisation adds a
further `R · N · sizeof(E) ≈ 12.6 MB` per claim — still manageable, but
the row-streamed materialisation variant of §9.8 avoids the latter.

## 15. Worked Example

Take the parameters from `t-b-computation.md` §8.6 and adapt them:

```text
n_a = 2, L = 2, C = 4, B = 4, n_d = n_b = 2
2 commitment groups, claims_per_group = 2
claim_to_group = [(0,0), (0,1), (1,0), (1,1)]
```

So:

```text
R          = 2
N          = t_len / D    = 4 · 2 · 2 · 4 = 64
w_len / D  = 4 · 2 · 4    = 32
num_q_W    = C · L         = 8
num_q_T    = C · L · n_a   = 16
```

Pick the cell `(r, c) = (1, 26)`. Decode it under both reshapings:

**W decode** (only valid because `c = 26 < w_len/D = 32`):

```text
dig_w   = 26 mod 2 = 0
b_w     = (26 / 2) mod 4 = 13 mod 4 = 1
claim_w =  26 / (B · L) = 26 / 8 = 3
q_W     = 0 · 4 + 3 = 3
```

`W_part(1, 26) = d_weight[1] · eq_x( offset_w + 1 + 4 · 3 ) = d_weight[1] · eq_x( offset_w + 13 )`.

**T decode** (valid for all `c < N`):

```text
dig_t   = 26 mod 2 = 0
a_row_t = (26 / 2) mod 2 = 13 mod 2 = 1
b_t     = (26 / 4) mod 4 = 6 mod 4 = 2
claim_t = 26 / 16 = 1
```

`(group_idx, claim_within_group) = claim_to_group[1] = (0, 1)`.

```text
q_T  =  1 + 4 · 0 + 4 · 2 · 1 = 9
T_part(1, 26)
   =  b_weight[1, 0] · eq_x( offset_t + 2 + 4 · 9 )
   =  b_weight[1, 0] · eq_x( offset_t + 38 )
```

So at this cell:

```text
Eval[1, 26] = d_weight[1] · eq_x(offset_w + 13)
            + b_weight[1, 0] · eq_x(offset_t + 38)
```

and its contribution to `<M_Flat, Eval>` is `M_Flat[1, 26] · Eval[1, 26]` —
two equality-polynomial factors weighting the same field element, one
W-shaped and one T-shaped. Neither factor is shared between them; both are
needed.

Cross-check against the original formulas: the same cell shows up in
`w_d`'s sum as one of the terms with `(q_W = 3, b_w = 1)` and in `t_b`'s
sum as one of the terms with `(q_T = 9, b_t = 2)`, weighted respectively
by `d_weight[1]` and `b_weight[1, 0]`. The matrix–vector form just collects
both into a single `Eval[r, c]` summand.

## 16. Practical Notes and Pitfalls

- **The two reshapings collide differently on each cell.** A given `c` has
  *different* `(q, b, low_idx, carry)` under W's decode and T's decode. So
  `Eval[r, c]` is necessarily a **sum** of two equality terms, not a
  product of one equality term with the cell.

- **Group dependence of `b_weight` lives along `c`, not along `r`.** In
  multi-group settings the T-part's row weight is *per-cell* (because the
  group depends on `claim_T(c)`). This is the same per-`q_T` lookup
  `TMatrixRowsEvaluator` already does — but it means that pulling
  `b_weight[r, g]` out as a per-row scalar only works in single-group
  configurations.

- **`Eval` is the sum of two equality tensors, not a single one.** This
  means the naive "call `eq_x` from scratch at every cell" approach costs
  `Θ(R · N · |x_challenges|)`. §9 fixes this without giving up the
  one-`Eval` form: separate the cell weight into `eq_low` (a small
  shared table) and `eq_hi` (two small precomputed slices indexed by
  `q + carry`), and `Eval[r, c]` reduces to **two table lookups and one
  multiply-add per cell**.

- **The materialised path (§9, §11.2) and the streamed path (§10, §11.3)
  compute the same scalar.** They differ in loop structure and memory
  layout. The materialised path keeps the "one M_Flat, one Eval" code
  shape; the streamed path collapses the `Eval` side into per-`q` carry
  buffers. Use whichever is more natural for the surrounding code — both
  are `O( R · N + num_q_T · |high| )` field ops on top of the unavoidable
  `R · N · D` `mul_base` for ring evaluations.

- **`M_Flat` is the bottleneck saving, not the inner-sum saving.** The
  matrix–vector framing eliminates W's redundant `eval_alpha` work by
  building each row of `M_Flat` once and letting both halves read from it
  — this is the `min(n_d, n_b)` ring-evaluation saving discussed in
  `matrix-rows-fusion.md` §6–7. It does **not** reduce the number of
  `eq_hi` evaluations: those scale with `num_q_W + num_q_T` regardless of
  which algorithm flavour is chosen.

- **`offset_low_bits` is the same for W and T**
  (`debug_assert_eq!(w_offset_low, t_offset_low)`). This is the reason
  `eq_low` is shared. If a future layout change broke that invariant, the
  W half and T half of `Eval` would each need their own `eq_low` table —
  doubling the small-table memory but otherwise leaving the matrix–vector
  identity intact.

- **`M_Flat` does not double-count anything.** Every cell `(r, c)` in
  `[0, R) × [0, N)` is read at most once in the final inner product, with
  `W_part(r, c)` and `T_part(r, c)` summed into a single weight. The cell
  is "used by W" exactly when `W_part(r, c) ≠ 0` (i.e. when both indicators
  fire), and "used by T" exactly when `T_part(r, c) ≠ 0`. Cells in W's
  range that fall outside W's reshaping support (none — W's reshaping is a
  bijection onto `[0, w_len/D)`) contribute zero by the indicator.

- **The `eq_hi` table is also a saving against the streamed form.** Today
  `compute_outer_sum` evaluates `eq_hi` on demand, recomputing
  `eq(high_challenges, off_hi + q + carry)` once per `q`. The materialised
  form (§11.2) computes the same values once into `eq_hi_w_table` /
  `eq_hi_t_table` upfront, then reads them by index. The total `eq` cost
  is the same; the materialised form just amortises it cleanly across
  both `Eval` halves.

## 17. What Changes vs. Today

The matrix–vector framing of (10) **is** the per-row pipeline of
`matrix-rows-fusion.md` §8, with the additional structural observation
that the inner contraction is literally an inner product of a
field-element matrix `M_Flat` against a verifier-computable weight matrix
`Eval`. The fusion can be implemented in two flavours, both correct and
both `O( R · N + num_q_T · |high| )` field ops on top of the
`R · N · D · mul_base` ring-evaluation cost.

### 17.1 Materialised form (§9, §11.2)

What changes:

1. The outer loop nest is inverted so that `r` is the outermost loop (per
   `matrix-rows-fusion.md` §8).
2. Within each row iteration, build `r_eval = M_Flat[r, ·]` once.
3. Precompute `eq_hi_w_table` and `eq_hi_t_table` upfront from
   `high_challenges` (small slices of size `num_q_* + 1`).
4. Build `w_pattern` and `t_pattern` once (linear pass over `c`).
5. Compute `Eval[r, ·]` from the patterns — either fully materialised
   (`R × N`) or row-streamed.
6. Fold via `<M_Flat, Eval>` — a single inner product.

What this gives you: the literal "one `M_Flat`, one `Eval`" code shape
the user asked about, at the same asymptotic cost as today's streamed
path. The `Eval` matrix is just the verifier-side weights laid out next
to the field-element rows of `M_Flat`.

### 17.2 Streamed form (§10, §11.3)

What changes:

1. Same outer loop inversion (`r` outermost), same per-row `r_eval`.
2. The W half (`q_W` sweep) and T half (`q_T` sweep) read from `r_eval`
   — no more `eval_ring_at_pows` calls during the block sweeps.
3. The two halves accumulate into `W_carry` / `T_carry` buffers as the
   current code already does.
4. Two unchanged `compute_outer_sum` passes finish the computation.

What this gives you: the smallest peak memory (no `Eval` materialised,
only `~B`-sized `eq_low` and two short carry buffers), at the cost of
splitting the verifier into two outer-sum calls instead of one inner
product.

### 17.3 What does **not** change in either form

- The `EqPolynomial::evals` precomputation of `eq_low` over `block_bits`.
- The `summarize_strided_pow2_block_carries` algebra (only its input
  changes from "row of cyclotomic ring" to "row of field elements"; the
  loop body is otherwise identical). Used by §11.3 only.
- The number of `eq_hi` evaluations
  (`num_q_W + num_q_T + 2 = O(num_q_T)`). Reorganised, not eliminated.
- The trait `SliceMleEvaluator` and its default `compute_outer_sum`.
- The verifier API surface (`eval_at_point_parts`, etc.).

The matrix–vector identity (10) is therefore not a redesign — it is the
clean way to **describe** the per-row sharing optimisation as a single
algebraic object, and to make explicit that the verifier's only new task
is to populate `Eval` correctly. The structure of `Eval` is captured in
full by (9), the column-pattern factorisation is captured by (14)–(16),
and the two concrete algorithms are captured by §11.2 (materialised) and
§11.3 (streamed).

## 18. Extension to `z_a` (the A-matrix half of the Z segment)

The same `<M_Flat, Eval>` framing extends cleanly to `z_a`, the
matrix-A summand of the `z` segment. Like `w_d` and `t_b`, it reads
rows of the same shared SIS matrix (just the first
`z_range = block_len · depth_commit` columns of those rows), and its
column-only pattern is constructed by the same peeled-block
decomposition.

### 18.1 Pattern derivation

The matrix-A summand of the legacy `z_segment` layout is

```text
z_seg_matrix[blk + B'·pt + B'·P·df + B'·P·DF·dc]
   = -fold_gadget[df] · matrix_A[blk · DC + dc]

matrix_A[c]  =  Σ_a a_weights[a] · M_Flat[a, c]    for c < z_range
```

(with `B' = block_len` so `idx = blk + B' · q` and
`q = pt + P·df + P·DF·dc`). For power-of-two `block_len` the eq factors
at the `log₂(B')` boundary:

```text
eq(full_vec, off_z + idx)
   = z_block_low_eq[(z_offset_low + blk) mod B']
   · eq_hi_z(z_offset_high + q + carry)
                                       carry = (z_offset_low + blk) / B'
```

so for `c = blk · DC + dc`,

```text
z_pattern[c]
   = z_block_low_eq[low_idx_z(blk)] · S_per_dc_per_carry[dc][carry_z(blk)]

S_per_dc_per_carry[dc][carry]
   = -Σ_{pt, df}  fold_gadget[df]
                · eq_hi_z(z_offset_high + (pt + P·df + P·DF·dc) + carry)
```

This makes `z_pattern[c]` an `O(1)` lookup per cell. The `S` table has
size `DC · 2`; building it costs `O(DC · 2 · P · DF)` muls plus `O(P ·
DF · DC)` `eq_eval_at_index` calls — utterly dwarfed by the per-row
SIS-matrix `eval_alpha` work.

### 18.2 Fusion into `m_eval[r, c]`

`a_w_padded[r] = a_weights[r]` for `r < n_a`, else zero. The fused
inner-product cell becomes

```text
m_eval[r, c] = d_w_padded[r] · w_pattern_padded[c]
            + Σ_g b_w_padded[r, g] · t_pattern_per_group[g][c]
            + a_w_padded[r] · z_pattern_padded[c]
```

with `z_pattern_padded[c] = 0` for `c ≥ z_range`. The outer loop runs
`r ∈ [0, R)` with `R = max(n_d, n_b, n_a)`. For rows where only Z is
active (`r ∈ [max(n_d, n_b), n_a)`) the inner loop shrinks to
`row_range = z_range` and skips the (zero-weighted) W/T terms entirely
— a small extra branch that pays back the `n_a − max(n_d, n_b)` rows
that would otherwise iterate `N` cells unnecessarily.

### 18.3 Non-pow-of-two `block_len` fallback

When `block_len` isn't a power of two the peeled-block decomposition
above doesn't apply. The implementation drops `z_pattern` (treats the
Z half as inactive in the fused inner product) and computes `z_a` via
dense materialisation of the matrix-A summand of `z_segment` followed
by single-factor `eval_offset_eq_tensor`. This is the same
materialise-and-MLE trade-off `r_dense` makes vs `r_sep`. It only
fires at the few non-pow-of-two recursive levels.

### 18.4 What this saves

`z_a`'s previous separate evaluator (`ZMatrixRowsEvaluator`) had to
build its own `matrix_a` table by reading `n_a` rows of the SIS matrix
at `alpha`. Several of those rows are *the same* rows W/T already
read for their own halves; with fusion, `r_eval` is shared and `z_a`
spends zero ring evaluations on rows `< max(n_d, n_b)`. End-to-end
verification at NV=32 OneHot D=32 dropped from ~20.5 ms (with
`m_eval_w_d_t_b` = 7.46 ms and a separate `m_eval_z_a` = 1.89 ms) to
~12 ms (with one fused `m_eval_w_d_t_b_z_a` ≈ 4.6 ms).
