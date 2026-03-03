# One-hot commitment cost + GPU-friendly PRG plan (Hachi §4.1 core)

This note consolidates our discussion about:

- **Commitment cost** (optimized, sparse-aware) for committing to a **one-hot-per-256** vector of length \(N = 256\cdot T\).
- How that cost depends on the **split** \(N = 2^M \cdot 2^R\) (square-root-ish vs skewed layouts).
- Why the cost is fundamentally \(\Theta(T)\) for Ajtai-style linear commitments, even for one-hot.
- A concrete **XChaCha20-based “virtual matrix”** strategy to derive public matrices \(A,B,\dots\) **locally** and **GPU-friendly**, avoiding materializing them.

The goal is to slot Hachi’s commitment core (paper §4.1) into a Jolt-style setting where the witness often contains large one-hot vectors.

---

## 1. What “commit” means in Hachi (§4.1) and in this repo

### 1.1 Paper definition (Eq. (13)–(14))

The paper commits to an \(\ell\)-variate multilinear coefficient table by reshaping it into \(2^r\) blocks of length \(2^m\) and doing:

- \(s_i := G^{-1}_{2^m}(f_i)\)  (digit decomposition)
- \(t_i := A s_i\)             (inner Ajtai)
- \(\hat t_i := G^{-1}_{n_A}(t_i)\)  (digit decomposition again)
- \(u := B[\hat t_1;\dots;\hat t_{2^r}]\) (outer Ajtai)

See the extracted paper lines:

```1166:1204:/Users/quang.dao/Documents/SNARKs/hachi/paper/hachi.pdf
4.1 Inner and Outer Commitment
...
s_i := G^{-1}_{2^m}(f_i) (13)
t_i := A s_i
t̂_i := G^{-1}_{n_A}(t_i)
u := B [ t̂_1 ; ... ; t̂_{2^r} ] ∈ R_q^{n_B} (14)
```

### 1.2 Repo implementation (WIP) matches the same structure

The current commitment core is `src/protocol/commitment/commit.rs`:

```63:93:/Users/quang.dao/Documents/SNARKs/hachi/src/protocol/commitment/commit.rs
let s_i = decompose_block(block, Cfg::DELTA, Cfg::LOG_BASIS);
let t_i = mat_vec_mul_unchecked(&setup.A, &s_i);
let t_hat_i = decompose_rows(&t_i, Cfg::DELTA, Cfg::LOG_BASIS);
...
let u = mat_vec_mul_unchecked(&setup.B, &t_hat_flat);
```

And the shape parameters are `M,R,N_A,N_B,LOG_BASIS,DELTA` in `CommitmentConfig`:

```9:23:/Users/quang.dao/Documents/SNARKs/hachi/src/protocol/commitment/config.rs
pub trait CommitmentConfig {
  const M: usize;     // block has 2^M entries
  const R: usize;     // number of blocks is 2^R
  const N_A: usize;   // inner rows
  const N_B: usize;   // outer rows
  const LOG_BASIS: u32;
  const DELTA: usize; // δ
}
```

with derived widths:

```63:71:/Users/quang.dao/Documents/SNARKs/hachi/src/protocol/commitment/config.rs
inner_width = (2^M) * DELTA
outer_width = N_A * DELTA * (2^R)
```

---

## 2. One-hot-per-256 model and the split knob

### 2.1 Witness structure

We consider a vector \(x \in F^N\) where:

- \(N = 256 \cdot T\),
- the vector is partitioned into \(T\) consecutive chunks of length 256,
- in each 256-chunk there is **exactly one** coefficient equal to 1, all others 0.

So the total number of ones is exactly:
\[
\#\{j : x_j = 1\} = T.
\]

### 2.2 Commitment reshaping is flexible: choose any \(M,R\)

Hachi commits to a table shaped as:

- number of blocks \(B := 2^R\),
- entries per block \(L := 2^M\),
- total length \(N = L \cdot B\).

For our setting:
\[
L\cdot B = 256T \quad\Longleftrightarrow\quad M + R = 8 + \log_2 T.
\]

This is where “square-root” vs “skewed” layouts live:

- **Skewed** (the naive “256 by T”): \(L=256\), \(B=T\).
- **Square-root-ish** (e.g. “\(16\sqrt{T}\) by \(16\sqrt{T}\)”): \(L=B=16\sqrt{T}\) (when feasible).

### 2.3 How many ones per *Hachi block*?

Each Hachi block holds \(L\) entries, which contains:
\[
g := L/256
\]
of the original 256-chunks, hence exactly:
\[
\boxed{g = L/256 = (256T/B)/256 = T/B}
\]
ones per block.

This identity is key: changing the split changes the distribution of ones across blocks, but the total number of ones stays \(T\).

---

## 3. Optimized commitment cost for one-hot (precise operation counts)

We measure cost at the **ring operation** level.

Notation:

- \(B := 2^R\) blocks
- \(L := 2^M = 256T/B\) entries per block
- \(g := T/B\) ones per block
- \(n_A := \texttt{N\_A}\), \(n_B := \texttt{N\_B}\)
- \(\delta := \texttt{DELTA}\), and \(b = 2^{\texttt{LOG\_BASIS}}\)
- ring degree \(D\) (so one ring element has \(D\) coefficients)

### 3.1 Inner decomposition \(s_i = G^{-1}(f_i)\)

For a one-hot coefficient equal to 1:

- in base-\(b\) gadget decomposition, only the **level-0 digit** is 1,
- higher digits are 0.

So \(s_i\) is **1-sparse per “1” entry**, and all zeros cost nothing *arithmetically* (but see PRG generation below).

### 3.2 Inner Ajtai: \(t_i := A s_i\) becomes “sum of columns”

Matrix \(A\) has shape \(n_A \times (\delta\cdot L)\). In the one-hot case, each block has \(g\) active columns (corresponding to level-0 digit of each “1” position).

For each block \(i\) and each inner row \(a \in [n_A]\):
\[
t_i[a] \;=\; \sum_{k=1}^{g} A[a, c_{i,k}]
\]
which uses **only additions**.

Precise counts:

- **Ring multiplications (inner)**:
\[
\boxed{\#\text{ring-mul(inner)} = 0}
\]

- **Ring additions (inner)**: per \((i,a)\) we add \(g\) terms, so \(g-1\) additions when \(g>0\).
Total:
\[
\boxed{\#\text{ring-add(inner)} = B \cdot n_A \cdot (g-1) = n_A\,(T - B)}
\]

This is the core “zeros don’t cost anything, multiply-by-one is identity” saving: it removes the \(L\) factor that a dense vector would incur.

### 3.3 Decompose \(t_i \mapsto \hat t_i := G^{-1}(t_i)\) is digit extraction, not ring arithmetic

This expands each \(t_i[a]\) into \(\delta\) ring elements by extracting base-\(b\) digits per coefficient.

In this repo, that’s implemented as coefficient-wise shift/mask in `gadget_decompose_pow2`:

```153:170:/Users/quang.dao/Documents/SNARKs/hachi/src/algebra/ring/cyclotomic.rs
let canonical = self.coeffs[i].to_canonical_u128();
let digit = (canonical >> shift) & mask;
F::from_canonical_u128_reduced(digit)
```

Exact coefficient-digit operations:
\[
\boxed{\#\text{digit-extract ops} = B \cdot n_A \cdot \delta \cdot D}
\]

### 3.4 Outer Ajtai: \(u := B[\hat t_1;\dots;\hat t_B]\) remains dense

The outer matrix \(B\) (paper’s notation) has shape \(n_B \times (n_A\delta B)\).

Even though the original witness was one-hot, \(\hat t_i\) is (heuristically) **dense**, because it is the digit decomposition of a sum of pseudorandom ring elements.

Therefore the outer commit is a standard dense matvec:

- **Ring multiplications (outer)**:
\[
\boxed{\#\text{ring-mul(outer)} = n_B \cdot (n_A\delta B)}
\]

- **Ring additions (outer)**:
to sum \(n_A\delta B\) products per output coordinate:
\[
\boxed{\#\text{ring-add(outer)} = n_B \cdot (n_A\delta B - 1)}
\]

### 3.5 Total arithmetic (optimized, one-hot-aware)

Combine:

- ring muls: \(\boxed{n_B n_A \delta B}\)
- ring adds: \(\boxed{n_A(T-B) + n_B(n_A\delta B - 1)}\)
- digit extraction: \(\boxed{B n_A \delta D}\)

**Critical observation:** the term \(n_A(T-B)\) is \(\Theta(T)\) unless \(B\) is almost as large as \(T\). So **commit time is fundamentally linear in \(T\)** for Ajtai-style commitments, even for one-hot, because you must incorporate \(T\) pseudorandom columns.

Changing the split \(L\times B\) mainly trades:

- smaller \(B\) → fewer outer ring muls (good),
- but inner ring adds still \(\approx n_A T\) (dominant at huge \(T\)).

---

## 4. What dominates at scale: PRG generation + memory bandwidth

Even if you count only ring additions, you still need to **obtain** the pseudorandom ring elements \(A[a, c_{i,k}]\) that you’re summing.

If you materialize \(A\), it is astronomically large. Therefore you must treat it as a **public function**:
\[
A[a,c] := \mathsf{PRG}(\text{seed}, \text{label}, a, c).
\]

So the “real” dominant costs for large \(T\) are:

- **PRG expansion throughput** (bytes/s or words/s),
- **writing/reading accumulators** (bandwidth),
- and only then NTT/ring multiplication (outer stage).

This is why “local PRG” matters.

---

## 5. Why Philox is attractive for GPUs but not a 128-bit-security answer

Philox/Random123 is engineered for HPC/Monte-Carlo randomness and GPU parallelism, but it is **explicitly not intended for cryptography**.

- Random123 docs: “They are not suitable for use in cryptography or security…”:
  - see `https://www.thesalmons.org/john/random123/releases/latest/docs/index.html`
- Recent cryptanalysis survey and attacks in an ML/GPU setting:
  - `https://eprint.iacr.org/2025/2161.pdf`

For Hachi’s commitment matrices, we want a conservative “indistinguishable from uniform” story, so we prefer a standard CSPRNG/PRF construction.

---

## 6. XChaCha20 “virtual matrix” design (local + GPU-friendly + crypto-shaped)

### 6.1 Why ChaCha/XChaCha

ChaCha20 is:

- **cryptographically standard** (widely deployed stream cipher),
- **random-access** in CTR form (block index = counter),
- **GPU-friendly** (only 32-bit add/xor/rotate; no tables).

XChaCha20 is convenient because it provides a **192-bit nonce**, so we can pack indices directly without a hash in most cases.

### 6.2 What we want: coefficient-level random access

We want to be able to compute:
\[
\text{coeff}(A; a, c, \text{coeff\_idx}, \text{prime\_idx})
\]
without generating unrelated matrix entries.

This is stricter than “entry-level local”; it enables GPU kernels that generate coefficients and immediately accumulate them (no intermediate storage).

### 6.3 Proposed indexing / domain separation

We define a “virtual matrix entry stream” for a given matrix family (A vs B), row/col, and (optional) CRT prime index.

**Key (256-bit)**:
- `key = public_matrix_seed` (32 bytes) from setup.

**Nonce (192-bit / 24 bytes)**:
Pack:

- `tag32`: a 32-bit domain tag for matrix label (e.g. `A=0x...`, `B=0x...`),
- `prime32`: CRT prime index (or 0 if not using CRT),
- `row64`: row index,
- `col64`: col index.

So:
\[
\text{nonce} = \text{LE32(tag)} \,\|\, \text{LE32(prime)} \,\|\, \text{LE64(row)} \,\|\, \text{LE64(col)}.
\]

**Counter (32-bit)**:
- `ctr = block_idx` where each ChaCha block yields 64 bytes.

This makes every \((\text{tag},\text{prime},\text{row},\text{col},\text{block\_idx})\) map to a unique 64-byte block.

### 6.4 Sampling coefficients mod a prime (CRT limb)

If we represent coefficients in CRT primes \(p_i\) (NTT-friendly 32-bit primes), then:

- generate 32-bit words from XChaCha stream,
- map them to \(\mathbb{Z}_{p_i}\).

Two options:

1. **Rejection sampling (strict uniform)**:
   - accept `x` if `x < 2^32 - (2^32 mod p_i)` and output `x mod p_i`.
   - for NTT primes close to \(2^{32}\), acceptance is \(\approx 1\) (rare resample).

2. **Fast reduction (tiny bias)**:
   - output `x mod p_i` directly.
   - fastest, but introduces slight bias unless \(p_i\mid 2^{32}\) (it doesn’t).

For cryptographic conservatism, prefer (1).

### 6.5 How this plugs into Hachi’s one-hot commit

Recall inner stage for one-hot:
\[
t_i[a] = \sum_{k=1}^{g} A[a, c_{i,k}]
\]
So we need to generate and add exactly **\(n_A \cdot T\)** ring elements’ worth of coefficients (across all ones):

- **PRG coefficient generation**: \(\Theta(n_A \cdot T \cdot D)\) coefficient limbs.
- **Accumulation**: \(\Theta(n_A \cdot (T-B) \cdot D)\) coefficient additions (ring-add count from §3.2 times \(D\)).

The XChaCha virtual matrix lets us do this with:

- no materialized \(A\),
- perfect locality: only the columns we touch,
- GPU kernels that fuse “generate → add” per coefficient.

---

## 7. GPU implementation strategies (RTX-class target)

### 7.1 Kernel shapes

Two common patterns:

1. **Thread-per-coefficient** (max parallelism):
   - each thread handles one `(a, block_i, coeff_idx[, crt_prime])`,
   - loops over the `g` selected columns and accumulates.
   - Pros: simple, coalesced writes.
   - Cons: inner loop over `g` may be large when \(B\) is small (but for one-hot, \(g=T/B\)).

2. **Thread-per-selected-column (streaming reduction)**:
   - each thread generates a PRG ring element for a selected column and writes into a temporary buffer,
   - then reduce-sum (warp/block reduction) into `t_i[a]`.
   - Pros: PRG generation is perfectly parallel.
   - Cons: needs extra memory traffic unless you do on-chip reduction.

In both cases, XChaCha’s CTR nature is GPU-friendly because each thread’s work is independent.

### 7.2 Fusing with CRT/NTT representation

If you store ring coefficients in an RNS/CRT basis for fast NTT multiplication, it is ideal to:

- generate ChaCha output directly into each CRT limb (prime index in nonce),
- keep accumulators in CRT form,
- only (optionally) reconstruct at boundaries.

This avoids expensive base conversions and keeps all arithmetic NTT-friendly.

### 7.3 Practical note: outer commitment cost

Outer commitment still needs \(\#\text{ring-mul(outer)} = n_B n_A\delta B\) ring multiplications.

If ring mul is NTT-based, it is compute-heavy but scales with \(B\), not \(T\).

At large \(T\), **inner PRG+adds dominate** unless you choose \(B\) so large that \(n_B n_A\delta B\) becomes comparable to \(n_A T\) (rare in one-hot settings).

---

## 8. Action items for integrating into this repo (design, not code yet)

This repo currently derives whole matrices using SHAKE and materializes them:

```31:46:/Users/quang.dao/Documents/SNARKs/hachi/src/protocol/commitment/utils/matrix.rs
let mut entry_rng = ShakeXofRng::new(seed, matrix_label, rows, cols, r, c);
CyclotomicRing::random(&mut entry_rng)
```

To support one-hot efficiently at scale, we likely want:

- A **non-materializing matrix API** (`VirtualMatrix`) with `entry(row,col)` computed on demand.
- A **ChaCha/XChaCha backend** with coefficient-level random access (nonce/counter packing in §6.3).
- A one-hot specialized inner commit path that sums only the \(T\) selected columns.

---

## 9. Bottom line

1. **One-hot removes inner ring multiplications** (inner Ajtai becomes a sum of columns), but it does **not** make commitment sublinear in \(T\): you still need \(\Theta(T)\) PRG expansion + coefficient additions.
2. The \(m,r\) split affects outer cost (\(\propto B\)) but does not remove the dominant \(\Theta(T)\) “touch \(T\) random columns” work.
3. The primary practical lever is therefore **PRG throughput and locality**.
4. **XChaCha20-CTR** is a strong fit for GPUs: cryptographically standard, random-access, and table-free ARX operations.

---

## 10. Concrete API proposal (Rust): “virtual matrices” + ChaCha backend

This section proposes a minimal set of abstractions so we can:

- stop materializing `A,B` as `Vec<Vec<CyclotomicRing<...>>>` in `setup()` (currently done in `commit.rs` via `derive_public_matrix`) and instead compute entries on demand, and
- add an optimized one-hot path that **only touches the columns needed**.

### 10.1 Design constraints from current code

Today, setup derives **full** matrices:

```31:46:/Users/quang.dao/Documents/SNARKs/hachi/src/protocol/commitment/utils/matrix.rs
pub(crate) fn derive_public_matrix<F: FieldCore + FieldSampling, const D: usize>(
    rows: usize,
    cols: usize,
    seed: &PublicMatrixSeed,
    matrix_label: &[u8],
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    (0..rows).map(|r| {
        (0..cols).map(|c| {
            let mut entry_rng = ShakeXofRng::new(seed, matrix_label, rows, cols, r, c);
            CyclotomicRing::random(&mut entry_rng)
        }).collect()
    }).collect()
}
```

and commitment multiplies by dense vectors:

```6:18:/Users/quang.dao/Documents/SNARKs/hachi/src/protocol/commitment/utils/linear.rs
pub(crate) fn mat_vec_mul_unchecked(...) -> Vec<CyclotomicRing<F, D>> {
  for row in mat {
    let mut acc = CyclotomicRing::<F, D>::zero();
    for (a, x) in row.iter().zip(vec.iter()) {
      acc += *a * *x;
    }
    out.push(acc);
  }
  out
}
```

This shape can’t exploit one-hot sparsity because it requires `row.iter()` (materialized) and always runs the `*a * *x` multiplication.

### 10.2 Proposed “matrix-like” trait (CPU + GPU-friendly)

We want a “read-only matrix oracle” that can return entries without storing them:

```rust
/// Read-only matrix oracle for commitment matrices A,B,D,...
/// `entry(r,c)` MUST be deterministic and domain-separated.
pub trait MatrixOracle<F, const D: usize> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn entry(&self, r: usize, c: usize) -> CyclotomicRing<F, D>;
}
```

Then change setup to store `A,B` as `Box<dyn MatrixOracle<...>>` (or a concrete generic type) rather than `Vec<Vec<_>>`.

### 10.3 Proposed PRG backend trait (so we can swap SHAKE ↔ XChaCha20)

At the bottom, `MatrixOracle::entry()` needs a deterministic PRF/XOF:

```rust
/// PRG/PRF for indexed expansion.
/// Produces a stream of 32-bit words for a given domain-separated (nonce,counter).
pub trait IndexedPrg32 {
    /// Fill `out_words` with pseudorandom words.
    fn fill_words(&self, nonce: [u8; 24], counter: u32, out_words: &mut [u32]);
}
```

The concrete implementation we want is **XChaCha20-CTR** with:

- `key = public_matrix_seed` (32 bytes),
- `nonce = pack(tag, prime_idx, row, col)` as in §6.3,
- `counter = block_idx` selecting which 64-byte ChaCha block.

### 10.4 `XChaChaVirtualMatrix`: matrix entries as PRG streams

Define:

```rust
pub struct XChaChaVirtualMatrix {
    pub seed: [u8; 32],
    pub tag32: u32,     // e.g. A=..., B=..., D=...
    pub prime32: u32,   // CRT limb index, or 0
    pub rows: usize,
    pub cols: usize,
}
```

and implement `MatrixOracle` by:

1. computing `nonce = LE32(tag32) || LE32(prime32) || LE64(row) || LE64(col)`,
2. generating enough 32-bit words via XChaCha blocks to cover `D` coefficients,
3. mapping each 32-bit word to a coefficient mod the CRT prime (rejection sampling preferred).

**Important:** this should ideally output coefficients **directly in CRT form** (for NTT), not via `FieldSampling`, so we don’t pay generic rejection-sampling for 128-bit primes.

### 10.5 Sparse-aware multiplication APIs (what we actually need for one-hot)

For one-hot, inner commit is “sum a few columns”:
\[
t_i[a] = \sum_{k=1}^{g} A[a, c_{i,k}]
\]

So we want an API that avoids all multiplications and avoids iterating `0` columns:

```rust
/// Compute y = A * x where x is {0,1}-sparse and provided as column indices.
///
/// `ones_cols` are the column indices where x[col] = 1 (all other columns are 0).
pub fn mat_one_hot_sum<F, const D: usize, M: MatrixOracle<F, D>>(
    a: &M,
    ones_cols: &[usize],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = vec![CyclotomicRing::<F, D>::zero(); a.rows()];
    for &col in ones_cols {
        for r in 0..a.rows() {
            out[r] += a.entry(r, col);
        }
    }
    out
}
```

This is the CPU reference version; the GPU version uses the same math but fuses PRG generation + coefficient addition.

For the outer commit, we still need dense matvec:
\[
u = B \cdot \hat t_{\text{flat}}
\]
but now `B.entry(r,c)` can be virtual and we can choose:

- CPU: NTT-based ring mul using a cached “row tiles” strategy,
- GPU: fused PRG→NTT→mul→accumulate kernels (future).

### 10.6 How `commit_ring_blocks()` changes conceptually

Today `commit_ring_blocks()` assumes materialized `setup.A`/`setup.B`.

With oracles, the flow becomes:

- build `ones_cols_i` for each block (indices of 1’s within the block’s level-0 digit plane),
- compute `t_i = sum_{col in ones_cols_i} A_col` (no ring mul),
- compute `t_hat_i = G^{-1}(t_i)` (digit extraction),
- compute `u = B * stack(t_hat)` (ring muls remain).

This preserves the paper definition; it only changes *how we compute* the same objects.

---

## 11. GPU implementation plan (incremental, low-risk)

### Phase 0 (doc + interfaces)
- Add the traits above and a CPU-only `XChaChaVirtualMatrix` implementation (no CUDA yet).
- Keep current SHAKE path as a fallback and for determinism tests.

### Phase 1 (CPU sparse inner commit)
- Implement `mat_one_hot_sum()` and a one-hot commit entry-point used by Jolt integration.
- Add correctness tests: compare sparse path vs dense path on small sizes.

### Phase 2 (GPU PRG + accumulation kernel)
- Port `XChaChaVirtualMatrix` entry generation to CUDA.
- Kernel: thread-per-coefficient accumulation into `t_i[a].coeff[k]` in CRT form.

### Phase 3 (GPU outer commitment)
- Only once inner is fast enough, consider GPU NTT + outer matvec acceleration.

---

## 12. Swapping Hachi’s “packing” for SuperNeo coefficient embedding (what changes)

This section addresses the specific “pick-and-choose” idea:

- keep **Hachi’s commitment core** (§4.1 inner/outer Ajtai + ring switching later),
- but replace the way we **embed a field witness vector** into ring elements with **SuperNeo’s coefficient embedding** (Def. 7 in SuperNeo §5),
- with the target regime: **128-bit base field**, **large ring degree** \(D\) (64–1024+), and “\(\kappa=1\)” / single-module flavor (i.e. \(n_A=n_B=1\) in the §4.1 notation).

### 12.1 What “embedding swap” means concretely for the committed object

Hachi’s commitment code path commits to a table of **ring elements**:
each entry is a `CyclotomicRing<F, D>` (i.e. \(D\) coefficients in the base field \(F\)).

If your starting object is a field vector \(x \in F^{256T}\) (one-hot-per-256), you must choose a packing:

- **SuperNeo coefficient embedding**: pack consecutive \(D\) field elements into the coefficient vector of one ring element.
  - Result: each ring element’s coefficients are in \(\{0,1\}\) and are **sparse**.
- **Dense packing / non-coefficient embedding (what we want to avoid for one-hot)**: any embedding that mixes coordinates (e.g. basis changes like Hachi’s \(\psi\)-style maps for extension fields) will generally turn a sparse field vector into a **dense** ring element, destroying “pay-per-bit”.

SuperNeo’s coefficient embedding definition (paper text) is:

```1220:1233:/Users/quang.dao/Documents/SNARKs/hachi/paper/superneo.pdf
Definition 7 (Coefficient Embedding).
... nF = d · nR.
... partition into d-sized sub-vectors ...
define the ring vector z := (z1, ... , z_{nR}) ∈ R_F^{nR}
```

In our power-of-two cyclotomic setting \(R_q = F[X]/(X^D+1)\), this is exactly: “take \(D\) field entries and interpret them as coefficients of one ring element”.

### 12.2 Sparsity profile of one-hot-per-256 after coefficient embedding

Let \(D\) be the ring degree, and pack \(D\) field entries per ring element.

- Total ring length is:
  \[
  n_R = \frac{256T}{D}.
  \]
- Each 256-chunk contains one 1. So each ring element spans \(D/256\) chunks, hence has about
  \[
  w \approx D/256
  \]
  ones among its \(D\) coefficients.

Examples:

- \(D=64\): \(w=0.25\) ⇒ each nonzero ring element is typically a **monomial** (exactly one 1 coefficient).
- \(D=256\): \(w=1\) ⇒ also monomial.
- \(D=1024\): \(w=4\) ⇒ each ring element has ~4 ones.

Crucially, the total number of ones across all coefficient slots remains:
\[
\sum_{j=1}^{n_R} \|\mathrm{cf}(z_j)\|_0 = T.
\]

### 12.3 Why SuperNeo gets “rot + add” (and when Hachi can too)

SuperNeo’s “pay-per-bit” observation is: if the multiplier \(b(X)\) has small/sparse coefficients, then ring multiplication can be implemented as a sum of coefficient rotations/shifts (no NTT):

```872:881:/Users/quang.dao/Documents/SNARKs/hachi/paper/superneo.pdf
When the b_i’s are small (such as bits), the cost to compute the ring operation is essentially
adding the rotations rot(a)_i for which b_i is non-zero.
```

This is a property of the **representation + algorithm**, not a property of the commitment protocol:

- In \(F[X]/(X^D+1)\), multiplying by \(X^k\) is a negacyclic rotation (+ sign flips).
- Multiplying by a sparse \(\{0,1\}\)-coefficient polynomial is “sum of a few rotations”.

Therefore:

- **If we embed one-hot via coefficient embedding**, then *any* place in Hachi where we need to multiply a public ring element by a witness-derived sparse ring element can use the same “rot + add” technique.
- **If we embed via a mixing map** that makes witness ring elements dense, then we lose this and must use generic ring multiplication (NTT/schoolbook).

### 12.4 Commitment-time impact when we keep Hachi §4.1 but use coefficient embedding

This breaks into two separable improvements.

#### Improvement A: gadget decomposition becomes extremely sparse (“pay-per-bit δ”)

Hachi §4.1 uses gadget decomposition \(G^{-1}\) with \(\delta=\lceil \log_b q\rceil\), because in the worst case coefficients are arbitrary mod \(q\).

But under coefficient embedding for a one-hot vector, coefficients are in \(\{0,1\}\). Then:

- only the **level-0 digit** is nonzero,
- higher digits are zero.

So the decomposed vector \(s_i = G^{-1}(f_i)\) has exactly “one nonzero digit per 1” across the whole witness.

**Consequences:**

- inner Ajtai \(t_i = A s_i\) becomes “sum a few columns” (as in §3), with cost proportional to the number of ones (=\(T\)).
- if we are allowed to change parameters, we can set \(\delta=1\) for bit witnesses (true pay-per-bit), shrinking:
  - the width of \(A\) by a factor \(\approx \delta\),
  - the width of \(B\) by the same factor,
  - and the opening witness size accordingly.

Even if we *keep* \(\delta\) at the 128-bit worst-case value (e.g. 32 for \(b=16\)), the extra digit planes are all-zero and can be skipped in a sparse-aware implementation.

#### Improvement B: per-term “multiply by digit” becomes trivial (rot+add or copy)

In sparse gadget form, the multipliers are digits in \(\{0,1\}\) (or very small), so:

- multiplying by 0 disappears,
- multiplying by 1 is identity,
- multiplying by \(X^k\) is rotate+signflip.

So the inner stage can be implemented with **no NTT** and essentially only ring additions + rotations, provided the matrix access is local (virtual matrix PRG).

#### About the “outer commit is negligible”

In the Hachi §4.1 core, outer ring multiplications scale like:
\[
\#\text{ring-mul(outer)} = n_B n_A \delta B.
\]

In your target parameter regime \(n_A=n_B=1\), this is \(\delta B\), which is tiny compared to the \(\Theta(T)\) inner accumulation at huge \(T\), for any reasonable \(B\ll T\). That’s the sense in which it’s “negligible” in the one-hot regime.

### 12.5 How it impacts Hachi’s later opening proof (ring switching + sum-check)

Swapping the embedding primarily changes the **distribution and bounds of witness coefficients**.

That propagates into Hachi’s opening proof in these places:

1. **Smallness / membership checks become much simpler.**
   - In ring switching, Hachi needs to prove the committed witness corresponds to coefficients in a small set / small interval (paper uses product polynomials over \([-b+1,\dots,b-1]\) after switching).
   - For coefficient-embedded bits, the set is just \(\{0,1\}\), so the “root-check” polynomial per coefficient can be as small as \(w(w-1)=0\) (degree 2) instead of degree \((2b-1)\).
   - This reduces the degree/sumcheck burden of the “smallness” sub-claim, and reduces the parameter pressure to choose a large \(b\).

2. **Norm growth under folding/challenges is better controlled.**
   - One of SuperNeo’s points is norm-preservation: small field entries stay small ring coefficients.
   - In Hachi, any step that forms random linear combinations (or uses sparse challenges) has easier norm bookkeeping when starting vectors are truly small in coefficient basis.

3. **Witness-table embedding sizes can drop if you let δ/τ adapt to bit-width.**
   - Hachi’s proof size and recursion boundary are sensitive to how large the “next committed object” is.
   - If bit-witness commitment uses \(\delta=1\) instead of \(\delta\approx 32\), then the committed tables, auxiliary openings, and any “digitized quotient” side witnesses shrink by that factor.

4. **What does NOT change (important):**
   - Hachi’s core “field-native verifier” story (ring switching + sum-check over \(\mathbb{F}_{q^k}\)) does not rely on coefficient embedding; it relies on automorphisms/trace and evaluation at \(\alpha\).
   - The outer Ajtai structure is still there if you keep §4.1 unchanged; coefficient embedding doesn’t remove it, it just makes the *inputs* to it cheaper to form.

### 12.6 Net effect summary (for the regime you care about)

If you keep Hachi’s commitment core and **only** swap the witness packing to SuperNeo coefficient embedding:

- **Commitment to one-hot becomes strictly faster** than any embedding that densifies the witness, because:
  - gadget digits become maximally sparse,
  - inner commit becomes “sum selected columns” (rot+add / copy),
  - and you can use local PRG to generate only touched matrix entries.
- **Opening proof later likely benefits** via:
  - lower-degree smallness checks (bits),
  - and possibly much smaller \(\delta\)/digitization parameters if you let them depend on bit-width.

But note the fundamental limit still holds: you must touch \(\Theta(T)\) public randomness (matrix columns) because the witness has \(T\) ones.

### 12.7 Parameter suggestions (128-bit field, large ring degree) for one-hot-per-256

This is a practical “what should we pick” note, assuming:

- base field \(F\) is ~128-bit prime,
- cyclotomic is power-of-two \(X^D+1\) (Hachi-style),
- witness is one-hot-per-256 in the *field* representation,
- and we pack via **coefficient embedding** into ring coefficients.

#### (A) Pick \(D\) to align with the 256-chunk structure (best case)

If you can choose \(D\) to be a multiple of 256 (e.g. \(D\in\{256,512,1024\}\)):

- each ring element spans exactly \(D/256\) of the original one-hot chunks,
- hence each ring element has exactly \(D/256\) ones in its coefficient vector (assuming the packing starts on a 256 boundary),
- and the per-ring-element multiplier is “sum of \(D/256\) monomials”.

Concrete:

- **\(D=256\)**: exactly **1** one per ring element ⇒ each nonzero ring element is a monomial \(X^k\).
  - multiply-by-witness is literally one rotation (plus sign) ⇒ maximally “rot+add”.
- **\(D=512\)**: **2** ones per ring element ⇒ sum of 2 rotations.
- **\(D=1024\)**: **4** ones per ring element ⇒ sum of 4 rotations.

This regime gives you the cleanest “pay-per-bit” implementation because the sparsity pattern is regular and stable.

#### (B) If \(D < 256\), coefficient embedding is still great (often even sparser)

If \(D\in\{64,128\}\):

- each 256-chunk covers multiple ring elements,
- and each ring element has either 0 or 1 one (typically a monomial).

This is excellent for rot+add *but* may be at odds with “large ring degree” goals elsewhere in Hachi (e.g. verification optimizations / sparse challenges / NTT amortization).

#### (C) If \(D\) is not a multiple of 256, nothing breaks, but bookkeeping gets messier

If \(D\) is not aligned to 256:

- the one-hot positions “wrap” across ring-element boundaries,
- each ring element still has about \(D/256\) ones, but the exact count varies by boundary effects,
- and your “which rotations to add” logic needs careful indexing.

This is still fine; it’s just less convenient (and makes it easier to make off-by-one bugs).

#### (D) “\(\delta=1\)” for bit/one-hot is the key pay-per-bit win (if you allow it)

With coefficient embedding, coefficients are in \(\{0,1\}\). For gadget decomposition in §4.1:

- the correct decomposition depth for bits is \(\delta=1\) (only the least significant digit plane is nonzero).
- keeping \(\delta \approx \lceil \log_b q\rceil\) (e.g. 32 for 128-bit with \(b=16\)) is a correctness-first choice, but wastes space/work unless the implementation is sparse-aware enough to skip the all-zero planes.

So, if you want “commitment to one-hot is as fast as possible”, the parameterization we want to *eventually* support is:

- \(n_A=n_B=1\) (your “\(\kappa=1\)” spirit),
- \(\delta=1\) for bit witnesses,
- and \(D\) a multiple of 256 (best: \(D=256\) or \(D=1024\)).

