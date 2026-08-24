# Fourth-Root Verifier for Hachi

**Status:** Design sketch — not yet implemented.

This document describes three protocol-level components that, taken together,
reduce the Hachi verifier's cost in the first folding round from O(√N) to
O(N^{1/4}) field operations (where N = 2^ℓ is the witness size):

1. **Tensor-structured folding challenges (T1):** Reduces the
   challenge-dependent verifier cost from O(√N) to O(N^{1/4}).
2. **Setup matrix MLE via claim-reduction sumcheck (T2):** Eliminates
   the dominant setup-dependent cost by deferring it to the next level.
3. **Tiered commitment design:** Shrinks the shared matrix by a factor f,
   enabling T2 at multiple recursive levels without cascading blowup.

## Background: Where the √N Cost Comes From

After the engineering optimizations described in `hachi-verifier-optimization.md`,
the level-0 verifier cost is ~65 ms for nv=36, dominated by:

| Subroutine            | Cost   | Asymptotic | Source                     |
|-----------------------|--------|------------|----------------------------|
| `compute_m_evals_x`   | 48 ms  | O(√N · λ)  | Setup matrix reads + c_α   |
| `sample_sparse_challenges` | 9.7 ms | O(√N)  | Deriving 2^r ring challenges |
| `stage2_m_eval`        | 6.5 ms | O(√N · λ)  | MLE evaluation of M-table  |

Both costs have the same root cause: the level-0 parameters split the
ℓ-α variables into m ≈ r ≈ (ℓ-α)/2, giving 2^r ≈ √(N/d) blocks, and
the verifier must process all of them.

The O(√N) cost decomposes into two independent components:

1. **Challenge-dependent cost:** Deriving and evaluating 2^r independent
   folding challenges c_i, computing c_α[i] = c_i(α) for each, and
   their contribution to the M-table entries.

2. **Setup-matrix-dependent cost:** Reading the expanded D/B/A matrices
   to build the M-table, then evaluating the M-table's MLE at the
   sumcheck challenge point.

We address (1) with tensor-structured challenges and (2) with a
claim-reduction sumcheck.

---

## Technique 1: Tensor-Structured Folding Challenges

### Idea

Instead of sampling 2^r independent challenges c_1, ..., c_{2^r} ∈ C,
decompose the block index set {0,1}^r = {0,1}^{r/2} × {0,1}^{r/2} and define:

```
c_{i‖j} = α_i · β_j,  where α ∈ C^{2^{r/2}}, β ∈ C^{2^{r/2}}
```

The verifier now derives and evaluates only 2 × 2^{r/2} = O(N^{1/4}) base
challenges. After ring-switching (evaluating at the ring-switch point),
c_alpha[i‖j] = c_alpha_left[i] · c_alpha_right[j], which has tensor
structure exploitable by the downstream MLE evaluations.

### Verifier savings

| Operation                  | Current          | With tensor      |
|----------------------------|------------------|------------------|
| Derive ring challenges     | 2^r              | 2 × 2^{r/2}      |
| Evaluate challenges at α   | 2^r              | 2 × 2^{r/2}      |
| c_α contribution to M-table | O(2^r)          | O(2^{r/2}) (tensor MLE) |

For N = 2^{30}, r = 10: current = 1024 challenges, tensor = 2 × 32 = 64.

### Extraction argument

The standard CWSS extraction (Lemma 6 in the Hachi paper) uses 2^r + 1
transcripts where the central challenge vector differs from each other in
exactly one coordinate. With tensor challenges, the extraction uses a
**2-level CWSS tree** (Definition 3 with μ = 2):

The protocol is modeled as a 5-round scheme with an empty intermediate
prover message:

| Round | Actor    | Content                    |
|-------|----------|----------------------------|
| 1     | Prover   | v = D · ŵ                  |
| 2     | Verifier | α ∈ C^{2^{r/2}}            |
| 3     | Prover   | ∅ (empty)                   |
| 4     | Verifier | β ∈ C^{2^{r/2}}            |
| 5     | Prover   | (ŵ, t̂, ẑ)                  |

CWSS parameters: ℓ₁ = ℓ₂ = 2^{r/2}, k₁ = k₂ = 2, S₁ = S₂ = C.
Tree size: K = (2^{r/2} + 1)² ≈ 2^r transcripts.

**Why flat CWSS (Option A) fails:** Viewing the tensor challenges as a
single vector in C^{2·2^{r/2}} with SS(C, 2·2^{r/2}, 2) structure does
not work. Varying α_p alone changes all 2^{r/2} challenges c_{p‖·}
simultaneously, giving only the β-folded aggregate w̄_p = Σ_q β_q · s_{p,q},
not individual s_{p,q}. The SS structure provides only transcripts differing
in one coordinate; extracting s_{p,q} requires the 4-transcript mixed
difference (two coordinates changed), which flat SS cannot provide.

**Why 2-level CWSS (Option B) works:** Take 4 transcripts for each (p,q):

```
tr(0,0): central (α^(0), β^(0))
tr(p,0): differs in α_p only
tr(0,q): differs in β_q only
tr(p,q): differs in both α_p and β_q
```

Expanding z = Σ_{p,q} α_p · β_q · s_{p,q} at each transcript and
taking the mixed second difference:

```
(z^{p,q} - z^{p,0}) - (z^{0,q} - z^{0,0})
  = (β_q' - β_q) · [(Σ_i α_i^{(p)} · s_{i,q}) - (Σ_i α_i · s_{i,q})]
  = (β_q' - β_q) · (α_p' - α_p) · s_{p,q}
```

This isolates s_{p,q} exactly. The cancellation is exact with zero
residual because the challenge structure c_{p,q} = α_p · β_q is bilinear.

**Fiat-Shamir compatibility:** In the non-interactive version, α = H₁(v)
and β = H₂(v, α). Since the round-3 prover message is empty, β depends
only on (v, α) — no prover action between α and β. This matches the
existing code: `QuadraticEquation::new_prover` absorbs v, then samples
challenges with nothing in between.

**This is a strict instance of the existing CWSS framework** (Definition 3
with μ = 2). No extension to the framework is needed. Lemma 3 handles
the composition directly.

### Norm blowup and MSIS security

The extracted denominator is now a ring product of two short differences:

```
c̄_{i,j} = (α_i' - α_i)(β_j' - β_j)
```

The convolution bound ‖a · b‖_∞ ≤ ‖a‖₁ · ‖b‖_∞ is valid in R_q =
Z_q[X]/(X^D + 1). The negacyclic reduction mod X^D + 1 introduces sign
flips but uses disjoint coefficient index sets, so |r_k| ≤ Σ|a_i|·|b̃_{k,i}|
with |b̃| ≤ ‖b‖_∞ everywhere. The same argument gives ‖a·b‖₁ ≤ ‖a‖₁·‖b‖₁.

**Per-row analysis of the 5-row verification system:**

| Row | Description | Challenge-dependent? | Tensor impact |
|-----|-------------|---------------------|---------------|
| 1 (D) | D·ŵ = v | No | Unchanged (MSIS norm 2b) |
| 2 (B) | B·t̂ = u | No | Unchanged (MSIS norm 2b) |
| 3 (eval) | b^T·G·ŵ = u_eval | No | Unchanged |
| 4 (fold) | (c^T⊗G₁)·ŵ = a^T·G·J·ẑ | **Yes** | ω̄ = (2ω)², β̄ = 4b^k |
| 5 (Ajtai) | (c⊗G_nA)·t̂ = A·J·ẑ | **Yes** | Same as row 4 |

Rows 1-3 contribute identically across all 4 transcripts; the stencil
produces zero for them. Rows 4-5 share the same extraction denominator
c̄ = (α_p'-α_p)(β_q'-β_q) and numerator structure.

**MSIS security impact (parametric in ω):** The MSIS norm ratio
(tensor vs. current) is exactly **4ω** = ‖c̄‖₁(tensor)/‖c̄‖₁(current)
× β̄(tensor)/β̄(current) = (2ω)²/(2ω) × 4b^k/(2b^k) = 4ω.

The field modulus is **q = 2^128 - 5823** (Q128_MODULUS in
`ntt/tables.rs`). The invertibility threshold is
q^{1/2}/√2 ≈ **2^{63.5}** — not ~46340 (which was the erroneous
value from confusing q with the 32-bit CRT-NTT sub-primes).

Production challenge families and their |C|:

| D | Config | ω | log₂|C| | Status |
|---|--------|---|---------|--------|
| 64 | SplitRing(hw=21, mag2=6) | 54 | ~128 | production |
| 128 | Uniform(w=31, ±1) | 31 | ~130 | production |
| 32 | Uniform(w=3, ±1) | 3 | ~15 | **test only** |
| 16 | Uniform(w=3, ±1) | 3 | ~12 | **test only** |

The D≤32 configs are test-only (`Fp128SmallD*`). Production challenge
families for D≤32 do not exist yet — they would need ω ≫ 3 to achieve
128-bit challenge-space security.

| Ring dim D | ω | Ratio = 4ω | Bits lost | ‖c̄‖_∞ | Invertibility margin |
|------------|---|------------|-----------|--------|---------------------|
| 64         | 54 | 216       | 7.8       | 216    | 216 ≪ 2^{63.5}     |
| 128        | 31 | 124       | 7.0       | 124    | 124 ≪ 2^{63.5}     |

For D=64 (production): ~8 bits lost against the 280+ bit MSIS security
floor. Invertibility has a factor-of-2^{55} margin. All well within
tolerance.

### Knowledge error

From Lemma 3 (CWSS → knowledge soundness), using the union-bound form
ε = Σ ℓᵢ · kᵢ / |Sᵢ|:

- Current (μ=1): ε = 2^r · 2 / |C|
- Tensor (μ=2): ε = 2 · (2^{r/2} · 2 / |C|) = 4 · 2^{r/2} / |C|

Both are negligible since |C| is exponential in λ.

*Note:* Lemma 3 as typeset in the paper states ε = Σ ℓᵢkᵢ / |Sᵢ|^{ℓᵢ}
(with the exponent). Under this formula, the current scheme has the
smaller error (|C|^{2^r} vs |C|^{2^{r/2}} in the denominator). Either
way, both bounds are astronomically small and the comparison is academic.

### Can we go to 3-level tensor (N^{1/6} verifier)?

c_{i‖j‖k} = α_i · β_j · γ_k with 3 × 2^{r/3} base challenges.

The extraction uses an 8-point stencil (2³ transcripts with
inclusion-exclusion signs), giving β̄ = 8b^k and
ω̄ = ‖δα·δβ·δγ‖₁ ≤ (2ω)³. The MSIS norm ratio vs. current is
16ω².

| Ring dim D | ω   | ‖triple‖_∞ | Ratio = 16ω² | Bits above current |
|------------|-----|-------------|-------------|-------------------|
| 64         | 54  | 11,664      | 46,656      | 15.5              |
| 128        | 31  | 3,844       | 15,376      | 13.9              |

**Invertibility is not a concern** with q = 2^{128} - 5823. The
threshold q^{1/2}/√2 ≈ 2^{63.5} dwarfs even the worst-case
‖triple product‖_∞ ≤ (2ω)² · 2 = 11,664 at D=64. The margin is
a factor of 2^{49.8}.

The real constraint on 3-level tensor is the **~16 bits of MSIS norm
degradation** at D=64 — still tolerable against the 280+ bit floor,
but with diminishing returns.

### Related work

- **Brakedown / Ligero:** The polynomial is shaped into a matrix and the
  row challenges have implicit tensor structure (Kronecker product of
  ℓ₀ individual 2-element vectors). Extraction uses a proximity gap
  argument (Diamond–Posen, Theorem 3.10 in eprint 2023/1784) rather
  than CWSS, but the structural idea is analogous.

- **Basilica (private communication):** Previously suggested tensor
  challenges for lattice-based folding; details not published.

---

## Technique 2: Setup Matrix MLE via Claim-Reduction Sumcheck

### Problem

The current verifier materializes the full M-evaluation table
(`compute_m_evals_x`): a vector of ~2^{16} ring elements (2^{22} field
elements at D=64) encoding the ring-switched verification matrix. It then
evaluates this table's MLE at the sumcheck challenge point
(`stage2_m_eval`). Together these account for ~55 ms — the dominant
verifier cost.

### M-table structure

The M-table is NOT simply "the raw setup matrix." Each entry is a
**hybrid** of setup-backed prefix-matrix reads and algebraic terms.

1. **Setup-dependent terms:** Reading ring elements from the PRG-expanded
   shared matrix and evaluating them at the ring-switch point α via
   `eval_ring_at_pows` (a D-to-1 linear map). These appear in all three
   data-bearing segments of `compute_m_evals_x`:
   - the w-segment (D-prefix rows),
   - the t-segment (B-prefix rows),
   - the z-segment (A-prefix rows).

2. **Purely algebraic terms:** Depending only on the opening-point
   weights, tensor/folding challenges, gadget scalars, and eq(τ₁,·)
   weights. These are additive decorations on top of the setup-backed
   segments, plus the r-tail.

The M-table column index does NOT correspond 1:1 to a shared-matrix
column. The w, t, and z segments read overlapping subrectangles of the
same shared matrix with different indexing schemes, so the claim
reduction must target the **setup-backed prefixes** rather than pretend
there is a single raw "M table" polynomial.

### Correct formulation: separate algebraic and setup-dependent parts

The verifier's stage-2 matrix oracle decomposes as:

```
m̃(r_x) = algebraic_MLE(r_x) + setup_MLE(r_x)
```

where:
- `algebraic_MLE(r_x)` depends only on the opening point, tensor
  challenges, gadget scalars, and eq(τ₁,·). The verifier computes this
  directly.
- `setup_MLE(r_x)` is the contribution coming from the shared matrix's
  D/B/A prefixes after evaluating the underlying ring elements at α.

If we expose the shared matrix as a 3D multilinear polynomial
`S[row_i][col_x][coeff_k]`, then the setup-dependent part is obtained by:

```
eval_ring_at_pows(S[i][x], α) = Σ_k pow_alpha(k) · S[i][x][k]
```

with an additional segment-remapping layer because the w/t/z slices use
different prefixes and column layouts.

### Enveloping matrix baseline

We now take as a protocol baseline that there is a single public
**enveloping matrix** `S_env` of size `max_rows × max_cols`, and that the
role matrices are literal row/column prefixes:

```text
D = S_env[0..n_d, 0..d_cols)
B = S_env[0..n_b, 0..b_cols)
A = S_env[0..n_a, 0..a_cols)
```

This matches the current setup architecture: one shared matrix with all
role-specific accesses implemented as prefixes of it.

Security-wise, this does **not** require any new "joint SIS" assumption.
If a cheating prover violates binding through one of the role-specific
equations, the reduction simply projects onto the smallest offending
prefix and invokes the ordinary SIS/MSIS argument for that prefix alone.
The enveloping matrix is therefore a sampling / notation convenience,
not a new hardness object.

### Recommended arrangement of the logical M table

The cleanest way to make the verifier-side MLEs "nice" is to stop
thinking of `M` as one monolithic flattened table and instead expose it
as a **virtual direct sum**:

```text
M_logic = D_part ⊕ B_part ⊕ (A_base ⊗ G_fold) ⊕ R_tail
```

with the following natural local coordinates:

- `D_part(block_idx, open_digit)` for the D-prefix contribution,
- `B_part(block_idx, a_slot, open_digit)` for the B-prefix contribution,
- `A_base(pos_in_block, commit_digit)` for the setup-backed A-prefix
  contribution,
- `R_tail(row_idx, level_idx)` as a purely algebraic tail.

The important structural point is that the current z-segment should be
viewed as

```text
z_segment(pos, commit_digit, fold_digit)
  = -A_base(pos, commit_digit) * fold_gadget(fold_digit),
```

so `fold_digit` is an **external algebraic tensor factor**, not part of
the setup-backed committed geometry.

This is much nicer than the current flattened
`w_segment || t_segment || z_segment || r_tail` view because:

1. all setup-backed pieces share the same envelope row variable,
2. the block axis stays explicit for tensor-structured challenges,
3. the A/fold factorization remains visible instead of being flattened
   away,
4. the pure `r_tail` never pollutes the committed MLE claim.

The recommended claim-reduction path under this arrangement is:

1. reduce first over the shared envelope row variable,
2. keep three slice-local setup claims (D/B/A),
3. batch those slice-local claims with a verifier challenge at the
   opening layer.

If we want "maximally nice" local variable ordering, the default should
be:

- D/B slices: block bits contiguous, then small digit/select bits,
- A slice: position-in-block bits, then commit-digit bits,
- if a single canonical storage order is required, use cheap
  segment-local permutations during claim reduction instead of
  interleaving unrelated coordinates into one global `x`.

This also composes naturally with the planned column-major witness
layout: if witness blocks become block-first / column-major, the D/B
slice geometry should mirror that same block-first ordering so the block
axis lines up across folding, tensor challenges, and setup-side claim
reduction.

### Protocol modification: batch range + relation inside stage 1

The design principle is:

- **yes:** batch the high-degree range-check and the low-degree relation
  check together inside the witness-domain stage 1;
- **no:** do not fuse that witness-domain stage with the new setup-side
  claim-reduction stage.

So the intended protocol shape is:

1. **Stage 1: batched witness-domain sumcheck.**
   Run a single batched sumcheck over `{0,1}^{num_u + num_l}` combining:

   - the existing range-check instance

     ```
     0 = Σ_z eq(τ₀, z) · Q(S(z)),
     ```

   - and the moved relation instance

     ```
     relation_claim = Σ_{x,y} w(x, y) · a(y) · m̃_τ₁(x).
     ```

   Since both instances have the same number of rounds, the existing
   front-loaded batched sumcheck machinery applies directly. The batched
   round polynomial degree is

   ```
   max(b/2 + 1, 2) = b/2 + 1,
   ```

   so the relation rides under the range-check's degree envelope instead
   of costing a second full witness-domain proof.

   At the common output point `r_stage1 = (r_x, r_y)`, the proof exposes:

   - `s_claim = w̃(r_stage1) · (w̃(r_stage1) + 1)` for the range-check,
   - `w_eval = w̃(r_stage1)` as the recursive witness MLE claim,
   - the relation-side scalar

     ```
     relation_oracle = w_eval · a(r_y) · m̃_τ₁(r_x).
     ```

   The important verifier subtlety is that stage 1's **round replay** and
   its **final batched output check** should be separated. The range-check
   contribution can be evaluated immediately, but the relation
   contribution depends on `m̃_τ₁(r_x)`, which is exactly what stage 2 is
   going to reduce. So the verifier should replay the batched sumcheck
   rounds first, then discharge the final combined output equality only
   after stage 2 has supplied the setup-side term. This is exactly the
   use-case supported by a "replay rounds now, inject external reduction
   later" batched-sumcheck API.

2. **Stage 2: claim-reduction / opening-alignment sumcheck.**
   Let

   ```
   λ := w_eval · a(r_y).
   ```

   The crucial point is: **do not divide by λ** to recover
   `m̃_τ₁(r_x)`. That division is not robust because λ may be zero. The
   safe formulation is to carry the known scalar λ into the next
   reduction and prove the scaled claim:

   ```
   λ · m̃_τ₁(r_x) = Σ_i eq(τ₁, i) · λ · M̃_α(i, r_x).
   ```

   After subtracting the verifier-computable algebraic contribution, this
   becomes a claim on the setup-backed prefixes of the shared matrix.
   There are two natural variants, with the first now preferred:

   - **Row-only reduction (preferred):** reduce over `i` first, yielding
     a scaled claim `λ · M̃_α(r_i, r_x)`, then handle the α /
     coefficient direction separately. This is especially attractive now
     that D/B/A all live inside one enveloping row prefix.
   - **Combined row+coefficient reduction:** reduce directly over
     `(i, k)`, yielding a scaled point claim on the preprocessed shared
     matrix commitment.

   The combined form is the cleanest statement:

   ```
   λ · setup_MLE(r_x)
     = Σ_{i,k} eq(τ₁, i) · pow_alpha(k) · λ · S(i, r_x, k),
   ```

   modulo the D/B/A prefix remapping. Because the w/t/z segments use
   different prefixes and coordinate maps, this is most naturally
   implemented as:

   - three setup claims (D, B, A) batched with a random coefficient
     **(preferred)**, or
   - one claim on a single remapped maximum-prefix polynomial
     (possible, but less clean).

### Proof size and verifier cost benefit

The current fused stage 2 is a full witness-domain sumcheck with:

- rounds = `num_u + num_l`,
- degree = 3,
- proof size = `3n` compressed field elements for `n = num_u + num_l`.

Under the batched-stage-1 design:

- stage 1 still has `n` witness-domain rounds, but its proof size stays
  controlled by the high-degree range-check:

  ```
  degree(stage1_batched) = max(b/2 + 1, 2) = b/2 + 1;
  ```

- so, compared with today's protocol, the moved relation check no longer
  costs an additional `3n` witness-domain sumcheck;
- the new claim-reduction stage is degree 2, but it runs only
  over the setup-side coordinates we still need to bind:
  `⌈log₂(m_row_count)⌉` rounds for row-only reduction, or
  `⌈log₂(m_row_count)⌉ + log₂(D)` rounds if the coefficient direction is
  reduced at the same time.

Since `m_row_count = n_d + n_b + 2 + n_a` is small, the new stage 2 is
much shorter than a full witness-domain sumcheck. This is the main proof
size and proving-time benefit.

The split happens along the right boundary:
stage 1 handles **all witness-domain work** in one batched proof, while
stage 2 handles **only setup-side reduction/alignment**.

### Preprocessing

Technique 2 requires committing (offline) to the shared matrix S
viewed as a multilinear polynomial over (row, column, coefficient)
indices. The polynomial has
log₂(max_rows) + log₂(max_cols) + log₂(D) variables.

The shared matrix dimensions are determined by the planner-chosen
schedule. The enveloping matrix has `max_rows × max_cols` ring
elements, where `max_rows = max(na, nb, nd)` and
`max_cols = max(a_cols, b_cols, d_cols)` with:
- `a_cols = m_eff × δ_commit` (A-commitment width)
- `b_cols = na × δ_open × 2^r` (B-commitment width)
- `d_cols = δ_open × 2^r` (D-commitment width)

Concrete shared matrix sizes (planner-optimized, onehot):

| Setting | D | max_rows | max_cols | Field elems | FlatMatrix | NTT cache | Total |
|---------|---|----------|----------|-------------|------------|-----------|-------|
| nv=32 | 32 | 3 | 399K | 2^{25.2} | 585 MB | 1.43 GB | 2.0 GB |
| nv=38 | 32 | 2 | 1.06M | 2^{26.0} | 1.02 GB | 2.54 GB | 3.6 GB |
| nv=44 | 64 | 2 | 8.52M | 2^{30.0} | 16.3 GB | 40.6 GB | 56.9 GB |

The NTT cache stores two CRT-NTT copies (negacyclic + cyclic) of the
shared matrix, each element occupying K × D × 4 bytes (K=5 CRT primes,
i32 Montgomery coefficients). The cache is ~2.5× the FlatMatrix
because (2 × K × 4) / 16 = 2.5.

At nv=44, the B-role dominates: `b_cols = 1 × 65 × 2^{17} = 8.5M`
ring elements, each 1024 bytes (D=64 × 16 B). The two rows produce a
16.3 GB FlatMatrix and a 40.6 GB NTT cache. This is the **existing**
shared matrix storage — the fourth-root verifier does not change it.

Committing to this matrix via Hachi PCS operates on ~2^{30} field
elements (for nv=44), producing a ~128 KB commitment. The one-time
setup cost scales with the matrix size (~0.9s for 2^{25} field
elements, ~30s for 2^{30}). The matrix polynomial is then batched
into the next folding level's witness for PCS opening.

**Prefix reuse for the commitment key is sound.** The MSIS binding
argument is not affected by the committed data being a deterministic
function of the commitment key: the adversary already knows the key
(transparent setup). The simplest approach uses a separate matrix label
in the XOF derivation, avoiding even the need to argue about
circularity. The architecture already supports this (`matrix.rs` line
95-105: `SHARED_MATRIX_LABEL` is a configurable domain separator).

### What feeds into the next Hachi layer

After the first folding round, the verifier holds two recursive objects:

1. **Witness MLE claim:** `w̃(r_stage1) = y'` — handled recursively as
   before.

2. **Setup-side claim:** operationally, the safest thing to carry is a
   **known-scalar multiple** of the setup evaluation, e.g.
   `λ · S̃(r_i, r_x, r_k) = y''` (or `λ · M̃_α(r_i, r_x)` if the
   coefficient reduction is deferred).

Conceptually this is still "the matrix claim." But for protocol
composition, it is better to keep the known scalar λ attached and absorb
it into later batching/opening coefficients, rather than divide it out.
That same scaled setup-side value is also what closes the delayed
relation part of the batched stage-1 output check.

Under the recommended arrangement above, this is best viewed not as a
single monolithic M-table opening, but as one **batched D/B/A setup
claim** against the common enveloping-matrix commitment.

Since all role matrices (A, B, D) are prefixes of the same shared
matrix, the setup-backed claims can still be batched down to one opening
against the preprocessed commitment.

### Setup opening: batched into next level

The claim-reduction sumcheck produces a point evaluation claim on the
shared matrix `S̃(r_i, r_x, r_k) = y'`. This claim must be verified by
opening S via Hachi PCS — the matrix polynomial is batched into the
next folding level's witness.

The matrix S at level L enters level L+1 **unfolded** as an additional
polynomial alongside the folded witness. This is the dominant cost of
Technique 2: the next level's effective witness grows by `|S_L|`.

| Setting | next_w (L1 input) | S_0 (matrix) | Combined | Growth | L1 capacity | Fits? |
|---------|------------------|-------------|----------|--------|-------------|-------|
| nv=32 | 40.1M | 38.3M | 78.5M | +96% | 67.1M | **No** |
| nv=38 | 453.8M | 68.2M | 522.0M | +15% | 536.9M | Yes (97%) |
| nv=44 | 2,835M | 1,090M | 3,926M | +38% | 4,295M | Yes (91%) |

For nv=32, the combined witness **overflows** L1's planned capacity.
The planner must be re-run with the matrix overhead accounted for.
For nv=38 and nv=44, L1 barely fits, but applying T2 at L1 as well
causes L2 overflow in both cases.

This cascading witness growth is the primary cost of Technique 2 and
must be factored into the planner's schedule optimization.

### Verifier cost after both techniques

Write `N' = N/D = 2^{ℓ−α}` for the witness length in ring elements. With a
symmetric (m, r) split, 2^r ≈ √N'.

| Component                  | Current        | After optimization  |
|----------------------------|----------------|---------------------|
| Derive folding challenges  | O(√N')         | O(N'^{1/4})         |
| Challenge evaluation at α  | O(√N' · D)     | O(N'^{1/4} · D)     |
| Setup matrix M-table       | O(√N' · D)     | eliminated (deferred) |
| M-table MLE evaluation     | O(√N' · D)     | eliminated (deferred) |
| Claim-reduction sumcheck   | —              | O(log m_row + log D) rounds |
| Setup opening (batched)    | —              | deferred to next level |
| **Total first-round verifier** | **O(√N' · D)** | **O(N'^{1/4} · D) + next-level overhead** |

For nv=30, D=64 baseline (r=12, m_row=5):
current ≈ (2 × m_row + 2) × 2^{12} × D ≈ 49K × D field ops →
optimized ≈ 4 × 2^6 × D = 256 × D field ops (96× reduction).

Note: the planner-optimized schedules use asymmetric splits and
varying D, so the concrete savings per setting are given in the
Verifier cost decomposition and Multi-Level Analysis sections below.

---

## Design Decisions

### 1. Norm cascade (T1)

Rows 1, 2, 3 are unaffected by tensor challenges (MSIS norm
2b = 32, log₂ = 5). Rows 4 and 5 are jointly affected, sharing the same
extraction denominator. The MSIS norm increase is exactly 4ω: ~8 bits at
D=64 (ω=54, production), ~7 bits at D=128 (ω=31, production). All well
within the 280+ bit MSIS security floor. Invertibility (Lemma 2) holds
with >2^{55} margin since q = 2^{128} - 5823.

### 2. CWSS extraction (T1)

The tensor scheme is a **strict instance** of Definition 3 with μ=2.
Flat CWSS (viewing tensor challenges as a single vector) fails because
SS structure only provides single-coordinate variations; the mixed
finite difference requires two-coordinate variations that only the
2-level tree provides. No extension to the CWSS framework is needed.

### 3. Stage composition (T2)

The protocol uses two stages:

1. **Batched witness-domain stage 1:** combines the range-check and the
   relation check in a single sumcheck under the range-check's degree
   envelope.
2. **Setup-side claim-reduction stage 2:** a low-degree sumcheck over
   setup coordinates only.

The relation's final output check depends on `m̃_τ₁(r_x)`, which stage 2
produces. So stage 1 replays its rounds first, and the combined output
equality is closed after stage 2 supplies the setup-side term. The key
algebraic invariant: carry the scaled quantity `λ · m̃_τ₁(r_x)` with
`λ = w_eval · a(r_y)` forward rather than dividing by λ (avoids the
zero-denominator corner case).

### 4. Preprocessing cost (T2)

The shared matrix size grows from 2^{25} field elements (nv=32, D=32)
to 2^{30} field elements (nv=44, D=64). The preprocessing commitment
takes ~0.9s at 2^{25} and ~30s at 2^{30}, with ~128 KB output. The
matrix polynomial is batched into the next folding level's witness
for PCS opening (see §Setup opening and §Multi-Level Analysis).

### 5. Enveloping matrix security / prefix reuse

Two distinct "prefix" facts:

1. **Role matrices as prefixes of one enveloping matrix:** sound. The
   security reduction projects onto the offending role-specific prefix,
   so binding reduces to SIS/MSIS on one prefix matrix, never on a
   fictitious "joint" matrix.

2. **Reusing the setup seed for the extra setup-matrix commitment key:**
   also sound. The cleanest implementation uses the same seed with a
   different matrix label (`b"matrix-commit"`) — a one-line change in
   the XOF derivation, zero extra storage.

### 6. Setup opening via batching (T2)

The setup claim is opened via Hachi PCS by batching the matrix
polynomial into the next folding level's witness. The matrix S from
level L enters level L+1 as an unfolded additional polynomial, growing
the next level's effective witness by |S_L|. For nv=32 this nearly
doubles the L1 witness; for nv=38/44 the growth is 15–38%. The
planner schedule must account for this overhead (see §Multi-Level
Analysis). The **tiered commitment design** (§Tiered Commitment Design)
addresses the cascading blowup when T2 is applied at multiple levels.

---

## Open Items

1. **Stage-1 proof payload.** The exact payload boundary between
   stage 1 and stage 2: stage 1 carries `s_claim` and `w_eval`,
   stage 2 becomes setup-only.

2. **Column-side reduction granularity.** With the enveloping matrix,
   row-first reduction is preferred. Whether the coefficient direction
   should be reduced in the same stage or one step later is TBD.

3. **Canonical slice-local coordinates.** The cleanest default:
   - D/B slices: block-first, then digit/select bits,
   - A slice: position-first, then commit-digit bits.

   If the witness adopts the column-major block layout from the proof-
   size work, the setup-side D/B slices should mirror that same block-
   first order.

4. **Scaled-claim / delayed-check plumbing.** The protocol carries
   `λ · setup_claim` forward rather than dividing by λ. The batched
   stage-1 verifier must support "replay rounds now, enforce final
   combined output later."

5. **Batching infrastructure for setup opening.** The matrix polynomial
   S must be batched into the next folding level's witness for PCS
   opening. This requires multi-polynomial batching at different opening
   points. The planner must account for the matrix carry when computing
   level capacities.

6. **Production challenge families for D ≤ 32.** T1 currently requires
   D ≥ 64 (production configs exist only for D=64 and D=128). Designing
   challenge families with large ω for D ≤ 32 would unlock T1 at
   deeper levels.

7. **Tiered commitment implementation.** The tiered design (§Tiered
   Commitment Design) is fully specified but not yet implemented. Key
   implementation tasks: per-chunk commitment loop, tier-3 meta-
   commitment, modified Stage 2 with 10 check groups, and planner
   integration for tiered schedules.

---

## Concrete Cost Analysis (Planner-Optimized Parameters)

All numbers below use the planner-optimized onehot schedules from
`hachi-planner` (tight z_pre, eq-compression, 4-ary GKR tree, header
stripping, multi-D rings, 128-bit SIS security).

### L0 parameters

| | nv=32 | nv=38 | nv=44 |
|-|-------|-------|-------|
| D | 32 | 32 | 64 |
| (m, r) | (16, 11) | (20, 13) | (21, 17) |
| (na, nb, nd) | (3, 2, 2) | (2, 2, 2) | (1, 2, 2) |
| m_row = nd+nb+2+na | 9 | 8 | 7 |
| sumcheck rounds | 26 | 29 | 32 |
| 2^r (blocks) | 2,048 | 8,192 | 131,072 |
| Planner total proof | 75,632 B | 78,896 B | 83,184 B |
| Baseline total proof | 99,805 B | 103,941 B | 106,533 B |

### Technique 1: Tensor challenge savings (level 0)

| | nv=32 | nv=38 | nv=44 |
|-|-------|-------|-------|
| Current challenges | 2,048 | 8,192 | 131,072 |
| Tensor challenges | 96 (32+64) | 192 (64+128) | 768 (256+512) |
| Challenge reduction | 21× | 43× | 171× |

Tensor challenges do not change proof size — the savings are
purely in verifier computation time.

### Technique 2: Claim-reduction sumcheck (level 0 proof-size impact)

Stage 2 changes from a degree-3 witness-domain sumcheck (rounds =
num_u + num_l) to a degree-2 claim-reduction over the setup-side
coordinates (rounds = ⌈log₂(m_row)⌉ + log₂(D)).

| Component | nv=32 | nv=38 | nv=44 |
|-----------|-------|-------|-------|
| Remove old stage 2 | −1,248 B | −1,392 B | −1,536 B |
| Add new stage 2 | +304 B | +272 B | +304 B |
| **Direct L0 proof Δ** | **−944 B** | **−1,120 B** | **−1,232 B** |

The direct sumcheck proof-size change at L0 is a modest saving. But
the **dominant** proof-size effect comes from the next-level witness
growth: the matrix S_0 is batched into L1's witness, growing it by
15–96% (see §Witness growth from Technique 2). The total proof-size
impact depends on the re-planned schedule for the heavier L1 witness.

### Verifier cost decomposition (level 0)

The verifier cost decomposes into two independent components with
different asymptotic behavior:

- **Challenge-dependent** (what Technique 1 reduces):
  derive + evaluate 2^r challenges at α. Cost: ~2 × 2^r × D.
- **Setup-dependent** (what Technique 2 reduces):
  materialize M-table from shared matrix + MLE evaluation.
  Cost: ~m_row × 2^r × (D + 1).

The setup-dependent cost dominates at 78–85% of total because of
the m_row factor:

| | nv=32 | nv=38 | nv=44 |
|-|-------|-------|-------|
| Challenge ops | 131K (18%) | 524K (20%) | 16.8M (22%) |
| Setup ops | 608K (82%) | 2.16M (80%) | 59.6M (78%) |
| **Total** | **739K** | **2.69M** | **76.4M** |

This decomposition determines which technique matters more:

| Scenario (L0 only) | nv=32 | nv=38 | nv=44 |
|---------------------|-------|-------|-------|
| Technique 1 only | 1.2× | 1.2× | 1.3× |
| Technique 2 only | 3.8× | 3.7× | 4.4× |
| Both techniques | 8.2× | 9.7× | 81.2× |

**Technique 1 alone is nearly useless** because it only reduces the
minority challenge-dependent cost while the dominant setup cost remains.
**Technique 2 is the primary driver** — it eliminates 78–85% of the
cost. Technique 1 becomes valuable only as a complement after
Technique 2 has already eliminated the setup cost, making the challenge
cost the new bottleneck.

Note: Technique 1 also has an MSIS security cost (see §Norm blowup
above): the 4ω norm ratio means ~8 bits lost at D=64, ~7 bits at
D=128. This may require slightly higher decomposition (`do`),
increasing proof size. The exact impact depends on the planner's SIS
security model and is not quantified here.

### Shared matrix storage (existing, not changed by these techniques)

The shared matrix is already allocated for the prover's mat-vec
operations. The B-role columns dominate at all three settings.

| | nv=32 | nv=38 | nv=44 |
|-|-------|-------|-------|
| max_rows × max_cols | 3 × 399K | 2 × 1.06M | 2 × 8.52M |
| Total field elements | 2^{25.2} | 2^{26.0} | 2^{30.0} |
| FlatMatrix | 585 MB | 1.02 GB | 16.3 GB |
| NTT cache (neg+cyc) | 1.43 GB | 2.54 GB | 40.6 GB |
| **Total setup storage** | **2.0 GB** | **3.6 GB** | **56.9 GB** |

The NTT cache stores two CRT-NTT copies (negacyclic + cyclic) of
the full shared matrix. Each CRT-NTT element is K × D × 4 bytes
(K = 5 CRT primes, i32 Montgomery coefficients), giving a 2.5×
expansion over the FlatMatrix.

The matrix polynomial is batched into the next folding level's
witness for PCS opening. This grows the next level's effective
witness by |S_L| (see §Witness growth from Technique 2).

---

## Multi-Level Analysis

The previous sections analyze optimizations at level 0 only. Since the
verifier must process all recursive levels, we need to understand
whether deeper levels become the bottleneck after L0 is optimized.

### L0 and L1 parameter comparison

| | nv=32 L0 | nv=32 L1 | nv=38 L0 | nv=38 L1 | nv=44 L0 | nv=44 L1 |
|-|----------|----------|----------|----------|----------|----------|
| D | 32 | 16 | 32 | 16 | 64 | 32 |
| r | 11 | 8 | 13 | 10 | 17 | 11 |
| m_row | 9 | 10 | 8 | 11 | 7 | 8 |
| 2^r | 2,048 | 256 | 8,192 | 1,024 | 131,072 | 2,048 |
| Challenge ops | 131K | 8.2K | 524K | 33K | 16.8M | 131K |
| Setup ops | 608K | 44K | 2.16M | 191K | 59.6M | 541K |
| **Total ops** | **739K** | **52K** | **2.69M** | **224K** | **76.4M** | **672K** |

### Does L1 become the bottleneck?

If both techniques are applied at L0 only, L0's cost drops to
~12K–197K field ops while L1 remains at 52K–672K. The result:

| Setting | L0 optimized | L1 unoptimized | L1/L0_opt | L1 is bottleneck? |
|---------|-------------|---------------|-----------|-------------------|
| nv=32 | 12K | 52K | **4.2×** | Yes |
| nv=38 | 25K | 224K | **9.1×** | Yes |
| nv=44 | 197K | 672K | **3.4×** | Yes |

**In every case, unoptimized L1 is 3–9× more expensive than
optimized L0.** The effect is most pronounced at nv=38, where L1
has r=10 (1024 blocks) with 11 M-table rows.

### Comprehensive scenario comparison

Total verifier field ops summed over all levels (7–8 depending on nv):

| Scenario | nv=32 | | nv=38 | | nv=44 | |
|----------|------:|------:|------:|------:|------:|------:|
| | ops | speedup | ops | speedup | ops | speedup |
| Baseline | 828K | 1.0× | 2.97M | 1.0× | 77.2M | 1.0× |
| T2 @ L0 | 219K | 3.8× | 806K | 3.7× | 17.5M | 4.4× |
| T2 @ L0+L1 | 176K | 4.7× | 614K | 4.8× | 17.0M | 4.5× |
| T1 @ L0 | 709K | 1.2× | 2.47M | 1.2× | 60.6M | 1.3× |
| T1 @ L0+L1 | 703K | 1.2× | 2.44M | 1.2× | 60.5M | 1.3× |
| T1+T2 @ L0 | 101K | 8.2× | 306K | 9.7× | 950K | 81× |
| **T1+T2 @ L0+L1** | **51K** | **16×** | **86K** | **35×** | **291K** | **265×** |
| T1+T2 @ all | 18K | 45× | 33K | 90× | 215K | 360× |

Key observations:

1. **T1 alone is nearly useless** (1.2–1.3×) because it addresses
   only the minority challenge-dependent cost.
2. **T2 alone gives 3.7–4.5×** by eliminating the dominant setup cost,
   but the challenge cost remains as the bottleneck.
3. **Both at L0** gives 8–81×; then L1 becomes the bottleneck.
4. **Both at L0+L1** captures most of the gain (16–265×, about 50–75%
   of the maximum possible speedup from optimizing all levels).
5. **Beyond L0+L1**, diminishing returns: optimizing all remaining
   levels adds another 2–3× on top of L0+L1.

### Technique 1 feasibility at L1

Technique 1 (tensor challenges) requires production challenge families
with large enough ω to maintain 128-bit challenge-space security.

| Level | D | Production config | ω | 4ω (norm ratio) | Feasible? |
|-------|---|-------------------|---|-----------------|-----------|
| nv=44 L0 | 64 | SplitRing(hw=21,mag2=6) | 54 | 216 (~8 bits) | Yes |
| nv=44 L1 | 32 | **None yet** | — | — | **No** |
| nv=32 L0 | 32 | **None yet** | — | — | **No** |
| nv=32 L1 | 16 | **None yet** | — | — | **No** |

Production challenge families for D ≤ 32 do not yet exist. Until they
are designed, Technique 1 can only be applied at levels with D ≥ 64
(currently only nv=44 L0). This means the practical near-term picture
is:

- **nv=44**: T1+T2 at L0 (81×), T2-only at L1 (→ ~265× total if T1
  is also eventually feasible at D=32).
- **nv=32, nv=38**: T2-only at L0 and L1 (3.8–4.8×), until D ≤ 32
  production challenge families enable T1.

### Witness growth from Technique 2 (batching)

The setup claim from each T2 level is resolved by batching the matrix
polynomial S into the **next** folding level's witness for PCS opening.
The matrix enters the next level **unfolded** — it was not committed
at the current level, so it doesn't benefit from folding compression.

**Digit decomposition asymmetry (the dominant cost).** The raw
field-element count of S is comparable to the folded witness. But S
has full-field (128-bit) coefficients, requiring `δ_commit_S =
ceil(128/lb) = 65` digits (at lb=2). The recursive witness has
`δ_commit_w = 1`. The `z_pre` term in the next witness is
`m_eff × δ_commit × δ_fold`, so S inflates `z_pre` by 65× per ring
element. This is the real cost of T2 — not the L1 capacity comparison.

**Split commitment design.** The combined [w ‖ S] polynomial uses a
split commitment: D-row joint, B-rows separate (with their own SIS
ranks), each sub-polynomial with its own optimal `(m, r)` split. The
SIS width constraint `na × δ_open × 2^r_S ≤ SIS_MAX_WIDTH[D][coll]`
caps `r_S` (e.g., `r_S ≤ 10` at D=16, `r_S ≤ 13` at D=32).

**L1 field-element capacity (necessary but misleading):** S must fit
in L1's polynomial evaluation table alongside the folded witness:

| | nv=32 | nv=38 | nv=44 |
|-|-------|-------|-------|
| L1 witness (no T2) | 40.1M | 453.8M | 2,835M |
| S_0 (L0 matrix) | 38.3M | 68.2M | 1,091M |
| L1 combined | 78.5M (+96%) | 522.0M (+15%) | 3,926M (+38%) |
| L1 planned capacity | 67.1M | 536.9M | 4,295M |
| Fits? | No | Yes (97%) | Yes (91%) |

This capacity check is necessary but **not sufficient**. Even when the
combined polynomial fits in L1's evaluation table, the downstream L2
input is dominated by `z_pre_S`, which dwarfs the baseline.

**Downstream L2 input (the real cost metric):** After committing [w ‖ S]
at L1 with the split design, the L2 input consists of `w_hat_w +
t_hat_w + z_pre_w + w_hat_S + t_hat_S + z_pre_S + r_ct`, where
`z_pre_S = m_eff_S × 65 × δ_fold_S` dominates:

| | nv=32 | nv=38 | nv=44 |
|-|-------|-------|-------|
| Baseline L2 input | 2.53M | 8.52M | 28.0M |
| T2@L0 L2 input | 34.9M | 63.6M | 183.0M |
| **Blowup ratio** | **13.8×** | **7.5×** | **6.5×** |
| z_pre_S fraction | 84% | 82% | 57% |
| L2 planned capacity | 4.2M | 8.4M | 67.1M |
| **L2 overflow** | **8.3×** | **7.6×** | **2.7×** |

**T2 at L0 alone causes L2 to overflow for all three settings.**
The z_pre_S term accounts for 57–84% of the inflated L2 input.

Split-design parameters at L1:

| | nv=32 | nv=38 | nv=44 |
|-|-------|-------|-------|
| L1 D | 16 | 16 | 32 |
| w: (m, r) | (14, 8) | (15, 10) | (16, 11) |
| S: r_S (SIS-constrained) | 10 | 10 | 13 |
| S: m_eff_S | 2,340 | 4,160 | 4,160 |
| δ_commit_S | 65 | 65 | 65 |
| z_pre_S (ring elems) | 1,825K | 3,245K | 3,245K |
| nb_S | 3 | 3 | 2 |
| split m_row | 13 | 13 | 10 |

**Cascading implication:** Even with T2 only at L0, the entire
downstream schedule (L2 onward) must be re-planned to accommodate
the 6.5–13.8× larger input. The planner needs a T2-aware mode that
accounts for the digit decomposition asymmetry at each T2 level.

### L1 shared matrix (for T2@L1)

If T2 is also applied at L1, the L1 shared matrix S_1 must be
carried to L2 as another full-field polynomial. S_1's max_cols is
dominated by the S polynomial's inner width `a_cols_S = m_eff_S ×
δ_commit_S`:

| Setting | L1 D | L1 max_rows × max_cols | S_1 field elems |
|---------|------|----------------------|----------------|
| nv=32 | 16 | 3 × 152K | 7.3M |
| nv=38 | 16 | 3 × 270K | 13.0M |
| nv=44 | 32 | 2 × 1.06M | 68.2M |

S_1 at L2 would also require `δ_commit = 65`, compounding the cascade.

### T2@L0+L1 cascade analysis

Applying T2 at **both** L0 and L1 means: (a) S_0 enters L1 (analyzed
above), inflating the L2 input by 6.5–13.8×; and (b) S_1 enters L2,
further inflating the L3 input. The compound effect is multiplicative
because the inflated L2 witness also gets digit-decomposed.

**L3 input blowup (T2@L0+L1 vs baseline):**

| | nv=32 | nv=38 | nv=44 |
|-|-------|-------|-------|
| Baseline L3 input | 604K | 1.14M | 2.01M |
| T2@L0+L1 L3 input | 14.4M | 19.0M | 63.8M |
| **Compound ratio** | **23.9×** | **16.6×** | **31.7×** |
| L3 planned capacity | 1.05M | 1.05M | 4.19M |
| **L3 overflow** | **13.8×** | **18.1×** | **15.2×** |

The compound blowup (16–32×) is far worse than T2@L0 alone (6.5–14×).
Three cascading effects compound:

1. The inflated L2 witness (from T2@L0) produces a larger `w_hat`,
   `t_hat`, `z_pre_w` at L2.
2. S_1 itself requires `δ_commit = 65` digit decomposition, producing
   a large `z_pre_S1` (347K–3.2M ring elements).
3. The combined L2 next witness enters L3 at 14–64M field elements,
   overflowing L3 capacity by 14–18×.

**Conclusion:** T2@L0+L1 is not viable with the baseline shared
matrix sizes. The **tiered commitment design** (§Tiered Commitment
Design below) resolves this by shrinking S by a factor f, bringing
the T2 ratios below 1× at manageable witness cost.

### Batching protocol design

The batching of S into the next level's commitment uses a **split
commitment** design that preserves preprocessing while sharing
challenges.

**Split commitment structure:** D-commitment is joint (computed at
proving time by concatenating `w_hat_w ‖ w_hat_S`), B-commitments
are separate (`B_w . t_hat_w = u_w` computed at proving time;
`B_S . t_hat_S = u_S` precomputed during preprocessing), and the
Ajtai binding covers the combined `z_pre`.

**Modified relation (6-block row structure):**

```
Block 1 (n_d rows):    D . [w_hat_w || w_hat_S] = v        -- JOINT, proving time
Block 2 (n_b_w rows):  B_w . t_hat_w = u_w                 -- w only, proving time
Block 3 (n_b_S rows):  B_S . t_hat_S = u_S                 -- S only, PREPROCESSED
Block 4 (1 row):       eval check for combined poly         -- JOINT, proving time
Block 5 (1 row):       fold check (shared challenges)       -- JOINT, proving time
Block 6 (n_a rows):    Ajtai binding for combined z_pre     -- JOINT (A_S.s_S precomputed)
```

m_row = `n_d + n_b_w + n_b_S + 2 + n_a` (increased by `n_b_S` over
baseline).

**Prefix packing for evaluation claims.** The polynomials w and S
may have different variable counts. Using prefix packing (Jolt paper):

1. Evaluation claims: `w(r_x) = y_w` and `S(r_S) = y_S` (from the
   claim-reduction sumcheck)
2. S has fewer variables; embed via prefix code with `r_S` as a
   slice of the full evaluation point
3. Sample packing challenges `r_pack` from transcript
4. Packed eval: `f_packed(r_pack, r_x) = w_sel . y_w + S_sel . y_S`
5. Single PCS opening proof at L+1 verifies the packed claim

**SIS constraints on r_S.** The outer width of the B_S commitment
is `na × δ_open × 2^r_S`. This must satisfy `outer_width ≤
SIS_MAX_WIDTH[D][collision]`. At D=16 with `collision = 3` (lb=2),
`SIS_MAX_WIDTH ≈ 133K`, giving `r_S ≤ 10` (with na=2, δ_open=65).
At D=32, `SIS_MAX_WIDTH ≈ 8.5M`, giving `r_S ≤ 13`.

**Structural lower bound.** The ratio `S_digits / next_w` has a
structural lower bound approximately independent of lb:

```
ratio >= max_rows × 128 / log₂(β)
```

where `β = challenge_l1_mass × 2^r × (b/2)`. Since
`δ_commit / δ_fold ≈ 128 / log₂(β)` (both have lb in the
denominator), the ratio cancels lb. Concrete bounds:

| Setting | max_rows | lb=2 | lb=4 |
|---------|----------|------|------|
| nv=32 (na=3) | 3 | 21× | 19× |
| nv=38 (na=2) | 2 | 13× | 11× |
| nv=44 (na=1) | 2 | 11× | 9× |

**Changing lb barely helps.** The ratio is fundamentally
`max_rows × 128 / (r + 7)`.

**Proof size vs prover time.** Despite the high ratio, the proof
size impact is small — extra sumcheck rounds = `ceil(log₂(1+ratio))
≈ 4–6`, adding < 1 KB total across all levels. The real cost is
**prover time**: the prover at L+1 handles a polynomial `(1+ratio)×`
larger.

**Streaming implementation.** Materializing `S_digits` at D=16 is
memory-prohibitive (2.5B+ i8 entries for nv=32). A streaming
`HachiPolyOps` variant should decompose S ring elements on-the-fly
during `decompose_fold` and `commit_inner`, using the shared matrix
(`HachiExpandedSetup`) already in memory. Memory cost: O(block_len)
per block, not O(|S| × δ).

**Planner modifications required:**

1. **T2-aware (m,r) optimization**: minimize total proof size
   including S_digits overhead. The optimal split shifts toward
   larger m, smaller r (to dilute the S/w ratio via z_pre).
2. **S_digits size accounting**: add `|S| × δ_commit_S × D_curr /
   D_next` ring elements to the next level's polynomial size.
3. **lb and D adjustments**: explore lb ∈ {2, 4, 8} at T2 levels;
   evaluate keeping D constant across the ring-switch boundary.
4. **Separate (m,r) for w and S**: the split design allows each
   polynomial its own optimal `(m_w, r_w)` and `(m_S, r_S)`.

### Recommendation

1. **T2 causes L2 overflow for all settings without tiering.** The
   digit decomposition asymmetry (`δ_commit_S = 65` vs `δ_commit_w = 1`)
   inflates the L2 input by 6.5–13.8× compared to baseline. Even
   nv=44 (the most favorable setting) overflows L2 by 2.7×. L1
   has enough field-element capacity in most cases, but the downstream
   `z_pre_S` blowup dominates.

2. **T2 at L0 only requires a T2-aware planner.** The planner must
   account for S's `δ_commit_S = 65` when computing L2+ capacities.
   The current baseline schedule is invalid for T2 because it
   under-provisions L2+ levels. The T2-aware planner should:
   - Add `S_digits` ring elements to L1's polynomial count
   - Use separate `(m, r)` splits for w and S (split design)
   - Re-optimize L2+ schedules for the 6.5–13.8× larger input
   - Possibly increase `lb` or `D` at L2+ to absorb the blowup

3. **T2@L0+L1 requires tiered commitment.** The L3 blowup with
   baseline matrix sizes (16–32× vs baseline, overflowing L3 by
   14–18×) is catastrophic. The tiered commitment design
   (§Tiered Commitment Design) resolves this by shrinking S, bringing
   T2 ratios below 1× across two levels.

4. **D=32 at L1 is far better for T2 than D=16.** SIS width tables
   at D=32 allow `r_S ≤ 13` (vs `r_S ≤ 10` at D=16), giving more
   flexibility for the S polynomial's `(m, r)` split. For nv=44
   (which already uses D=32 at L1), the blowup is 6.5× — the
   lowest across all settings.

5. **Technique 1 (tensor challenges) applies to L0 only where
   D ≥ 64** (i.e., nv=44). The MSIS norm penalty (~8 bits at D=64)
   may require slightly higher decomposition. T1 becomes valuable
   only after T2 eliminates the dominant setup MLE cost.

6. **The structural lower bound (`max_rows × 128 / log₂(β)`)
   limits how much lb or D changes can help.** The S_digits / next_w
   ratio is fundamentally 8–21× depending on `max_rows`. The only
   way to substantially reduce this is to minimize `max_rows`
   (requiring rank-1 commitments if security permits) or to shift
   the `(m, r)` split toward large m (diluting the ratio via z_pre).

7. **Proof size impact is small (< 1 KB).** The extra sumcheck
   rounds from the larger polynomial add only 4–6 rounds per level,
   totaling < 1 KB across all levels. The real cost is prover time
   at L1 (9–22× slower depending on the S/w ratio).

---

## Tiered Commitment Design

Technique 2's T2 cascade (`S_digits / next_w` ratios of 10–62×)
prevents applying T2 at multiple levels with baseline matrix sizes
(§Multi-Level Analysis). This section describes a **tiered commitment**
design that eliminates the cascade by trading witness growth for
shared-matrix shrinkage.

### Idea

Split the polynomial into k = f² chunks (shrink factor f), commit each
chunk with **shared** per-chunk matrices that are 1/f the column width
of the baseline, then absorb the per-chunk commitment vectors into the
witness. A small **tier-3 Hachi commitment** `(c_meta, v_meta, u_meta)`
binds the collection of per-chunk vectors. The proof contains
`(c, c_meta, v_meta, u_meta)` — independent of k.

With shrink factor f and baseline `(m, r)`:

- `r_chunk = r − log₂(f)`, fold challenges span `2^(r + log₂(f))` blocks
- `m_eff_chunk = ⌈N / 2^r_fold⌉` (1/f of baseline)
- Per-chunk D_chunk, B_chunk have 1/f the column width → **S shrinks by ≥ f**
- `w_hat + t_hat` grow by f (more blocks), but `z_pre` shrinks by ~f
- Per-chunk commitment vectors `v_j, u_j` are digit-decomposed and absorbed

### Modified Stage 2 relation

The Stage 2 relation at level L operates on the combined witness:

```
z = [ w_hat_1 .. w_hat_k          -- opening digits (all chunks)
    | t_hat_1 .. t_hat_k          -- redundancy digits (all chunks)
    | z_pre                        -- challenge-folded blocks
    | v_digits_1 .. v_digits_k    -- digit-decomposed per-chunk D-commitments
    | u_digits_1 .. u_digits_k    -- digit-decomposed per-chunk B-commitments
    | w_hat_meta                   -- tier-3 opening digits
    | t_hat_meta                   -- tier-3 redundancy digits
    | z_pre_meta                   -- tier-3 challenge-folded blocks
    | r_ct ]                       -- quotient row digits
```

**10 check groups:**

Original polynomial checks:

1. Per-chunk D-checks (k × nd_chunk rows): `D_chunk · w_hat_j = Recompose(v_digits_j)`
2. Per-chunk B-checks (k × nb_chunk rows): `B_chunk · t_hat_j = Recompose(u_digits_j)`
3. Eval check (1 row): links w_hat to Stage 1 evaluation claim
4. Fold check (1 row): `z_pre = Σ_j c_j · block_j`
5. Ajtai binding (na rows): `A · z_pre = c`

Tier-3 commitment checks:

6. D_meta check (nd_meta rows): `D_meta · w_hat_meta = v_meta`
7. B_meta check (nb_meta rows): `B_meta · t_hat_meta = u_meta`
8. Eval-like check (1 row): links w_hat_meta to the commitment collection
9. Fold check (meta) (1 row): `z_pre_meta = Σ_j c_meta_j · block_meta_j`
10. Ajtai binding (meta) (na_meta rows): `A_meta · z_pre_meta = c_meta`

**Structural properties:**

- Per-chunk rows (1, 2) have block-diagonal structure with **shared**
  D_chunk / B_chunk. MLE evaluation cost is O(|D_chunk|) + O(log k),
  independent of k.
- The T2 setup matrix S is ONE copy of D_chunk/B_chunk, not k copies.
  S_chunk = S_baseline / f in ring elements.
- The Recompose operation is structural and contributes nothing to the
  T2 setup matrix.

### Protocol flow (Fiat-Shamir)

```
1. Prover computes per-chunk commitments (v_j, u_j) for j=1..k
2. Prover computes tier-3 Ajtai binding: c_meta = A_meta · z_pre_meta
3. Prover computes original poly Ajtai binding: c = A · z_pre
4. Prover sends (c, c_meta) to transcript
5. Transcript derives evaluation point for original poly
6. Stage 1 sumcheck (original poly eval claim)
7. Transcript derives tier-3 folding challenges
8. Prover computes tier-3 opening: w_hat_meta, t_hat_meta
9. Prover sends (v_meta, u_meta) to transcript
10. Stage 2 sumcheck (combined 10-group relation)
```

Proof = `[c, c_meta, Stage1_msgs, v_meta, u_meta, Stage2_msgs]`

### Security argument (sketch)

Binding chain (two interlocking commitment levels):

- **Tier-3 binds the commitment collection:** `c_meta = A_meta · z_pre_meta`
  provides MSIS binding; fold and eval-like checks ensure consistency.
- **Per-chunk checks link collection to original polynomial:**
  `D_chunk · w_hat_j = Recompose(v_digits_j)` — the same `v_digits`/`u_digits`
  appear in both per-chunk and tier-3 checks, so consistency is automatic.
- **Original polynomial binding:** `c = A · z_pre`; eval and fold checks
  link w_hat, z_pre to the polynomial's blocks.
- **Composition:** Level L+1's standard Hachi commitment binds the entire
  next witness. Digit decomposition range constraints are enforced by MSIS
  extraction at the next level.

### Concrete numbers

All numbers computed by `scripts/t2_cascade_analysis.py` (`analyze_tiered_l0`),
using planner-optimized onehot schedules.

**nv=32 (D=32, T2 only):**

| f | k | S_red | S raw (GB) | Witness (M ring) | Growth | T2 ratio |
|---|---|-------|-----------|------------------|--------|----------|
| 1 | 1 | — | 1.14 | 1.3 | 1.00× | 62.1× |
| 2 | 4 | 4× | 0.25 | 1.2 | 0.95× | 14.5× |
| 4 | 16 | 9× | 0.13 | 1.8 | 1.44× | 4.8× |
| 8 | 64 | 36× | 0.03 | 2.3 | 1.82× | 0.9× |
| 16 | 256 | 72× | 0.02 | 4.5 | 3.56× | 0.2× |

S reduction exceeds the naive f× because SIS ranks decrease with
smaller chunk widths: at f=8, na drops from 3→1, shrinking B-column
width by 3× on top of the 8× from smaller r_chunk.

**nv=38 (D=32, T2 only):**

| f | k | S_red | S raw (GB) | Witness (M ring) | Growth | T2 ratio |
|---|---|-------|-----------|------------------|--------|----------|
| 1 | 1 | — | 2.03 | 14.2 | 1.00× | 9.8× |
| 2 | 4 | 2× | 1.02 | 10.0 | 0.71× | 6.9× |
| 4 | 16 | 4× | 0.51 | 9.8 | 0.69× | 3.5× |
| 8 | 64 | 8× | 0.25 | 14.7 | 1.03× | 1.2× |
| 16 | 256 | 16× | 0.13 | 26.6 | 1.88× | 0.3× |

At nv=38, z_pre dominates the baseline witness (89%). Tiering at
f=2–4 actually **shrinks** the total witness because z_pre's decrease
more than offsets w_hat+t_hat growth. The witness only exceeds baseline
at f≥8.

**nv=44 (D=64, T1+T2, l1_mass=216):**

| f | k | S_red | S raw (GB) | Witness (M ring) | Growth | T2 ratio | L1 needs |
|---|---|-------|-----------|------------------|--------|----------|----------|
| 1 | 1 | — | 32.50 | 46.4 | 1.00× | 23.9× | — |
| 2 | 4 | 2× | 16.25 | 48.8 | 1.05× | 11.4× | — |
| 4 | 16 | 4× | 8.12 | 76.0 | 1.64× | 3.6× | +1 bit |
| 8 | 64 | 8× | 4.06 | 140.3 | 3.02× | 1.0× | +2 bits |
| 16 | 256 | 32× | 1.02 | 274.8 | 5.92× | 0.1× | +3 bits |
| 32 | 1024 | 64× | 0.51 | 546.6 | 11.78× | 0.0× | +4 bits |
| 64 | 4096 | 128× | 0.25 | 1092.2 | 23.54× | 0.0× | +5 bits |

The z_pre buffer effect is strong: f=2 gives 2× S reduction for only
5% witness growth. At f≥16, nb/nd drop from 2→1, giving additional
S reduction beyond f (32× at f=16 instead of 16×).

### NTT cache trade-off

The NTT form is needed for efficient ring multiplication but can be
computed once on-the-fly at the start of each commitment operation.
Dropping the NTT cache halves persistent storage:

- **Raw only** (FlatMatrix): `S_ring × D × 16 bytes`
- **With NTT cache** (FlatMatrix + 2× CRT-NTT copies): `3.5× raw`

One-time NTT conversion cost at nv=44 is ~seconds, negligible when
amortized across the commitment operation. Row-by-row streaming is
possible.

For deployments where persistent disk is the bottleneck, dropping the
NTT cache is recommended.

### Combined T1+T2 at L0+L1 for nv ≥ 40

For large nv, the target is T1+T2 at L0 (81× verifier speedup) plus
T2 at L1 (265× total). This requires controlling the T2 cascade at
two levels.

| Scenario | L0 raw (GB) | Total raw (GB) | L0 witness | L1 T2 | L2 T2 | Viable? |
|----------|------------|---------------|-----------|-------|-------|---------|
| f=4, f_L1=1 | 8.1 | 10.2 | 1.6× | 3.6× | 38.2× | L1 marginal, L2 overflows |
| f=8, f_L1=1 | 4.1 | 5.1 | 3.0× | 1.0× | 19.7× | L1 OK, L2 overflows |
| **f=8, f_L1=4** | **4.1** | **4.3** | **3.0×** | **1.0×** | **1.2×** | **Both OK** |
| f=16, f_L1=4 | 1.0 | 1.1 | 5.9× | 0.1× | 0.5× | Both OK (generous) |
| f=4, f_L1=2 | 8.1 | 9.1 | 1.6× | 3.6× | 9.5× | L1 marginal, L2 overflows |

### Recommendations

**nv=32–38 (D=32, T2 only):**

- **Storage reduction only:** f=2–4. Gets storage from 1–2 GB to
  0.1–0.5 GB with no witness growth penalty (z_pre buffer effect).
- **T2@L0 enabling:** f=8. T2 ratio drops below 1× at modest witness
  cost (1.0–1.8×). Storage drops to 30–250 MB.

**nv ≥ 40 (D=64, T1+T2):**

- **Storage only (no T2):** f=2. Storage from 32.5 GB to 16.25 GB at
  5% witness cost.
- **T2@L0 only (81× speedup):** f=8. Storage ~4 GB, T2 ratio 1.0×.
- **T2@L0+L1 (265× speedup):** f=8 at L0 + f_L1=4 at L1. Total
  storage ~4.3 GB, both T2 ratios ≤ 1.2×. This is the sweet spot.
- **Aggressive storage + T2:** f=16 at L0 + f_L1=4 at L1. Total
  storage ~1.1 GB with generous T2 margins, at 5.9× witness growth.

---

## References

- Hachi paper: Sections 4.2–4.3, Appendix A (Lemmas 5–9)
- Diamond–Posen, "Succinct Arguments over Towers of Binary Fields"
  (eprint 2023/1784, EUROCRYPT 2025): Section 3.4, Theorem 3.10
- Brakedown (Golovnev et al., CRYPTO 2023): tensor product extraction
- Jolt/Spartan: claim-reduction sumcheck pattern for structured matrices
- FMN24 (Fenzi, Moghaddas, Nguyen): CWSS framework, Definition 3, Lemma 3
- LS18 (Lyubashevsky–Seiler): short invertible elements, Lemma 2
