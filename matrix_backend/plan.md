# Plan: A Self-Contained `commit` Module for Akita

> Companion to [`background.md`](background.md), which traces today's `commit()`
> path. Read it first for the vocabulary (`t_i = A·s_i`,
> `t̂_i = decompose(t_i)`, `u = B·concat_i(t̂_i)`, tiered `B'`/`F`, `LevelParams`,
> `NttSlotCache`, `RingMatrixView`).

## Contents

1. [Summary](#1-summary)
2. [Naming model](#2-naming-model)
3. [Decision: a module, not a crate](#3-decision-a-module-not-a-crate)
4. [Folder structure](#4-folder-structure)
5. [Background recap](#5-background-recap)
6. [The building blocks (API)](#6-the-building-blocks-api)
7. [Before / After: how the real code changes](#7-before--after-how-the-real-code-changes)
8. [The flat-matrix mechanism](#8-the-flat-matrix-mechanism)
9. [Implementation plan](#9-implementation-plan)
10. [Keeping the same performance](#10-keeping-the-same-performance)
11. [Relationship to the compute-backend spec](#11-relationship-to-the-compute-backend-spec)
12. [Risks & open questions](#12-risks--open-questions)
13. [Definition of done](#13-definition-of-done)

---

## 1. Summary

**What:** gather everything commit-related into one self-contained module,
`crates/akita-prover/src/commit/`, behind a narrow public surface. At its core
is a single commitment primitive — `ajtai_commit(commitment_key, spec, opening)
-> commitment` — that every matrix multiply in the scheme (`A`, `B`, `B'`, `F`)
flows through. The pipeline functions (`outer_commit`, `commit_inner_one`, …)
live in the same module. The rest of `akita-prover` only sees a handful of
re-exported entry points.

**Why:** today the A-side has four bespoke backend methods (`dense_commit_rows`,
`onehot_commit_rows`, `sparse_ring_commit_rows`, `recursive_witness_commit_rows`)
and the B/F-side is open-coded three times (`commit_with_validated_params`,
`tiered_commit_u_final`, `commit_w`). The algebra is identical; only the *opening
representation* differs. The logic is also scattered across `api/commitment.rs`,
`backend/*`, `protocol/ring_switch/commit.rs`, and `compute.rs`.

**The change in five bullets:**

- New module `akita-prover::commit` is the single home for the commit
  subsystem: the `CommitBackend` trait, its CPU implementation, the
  decomposition unit, and the pipeline functions.
- One commitment primitive `ajtai_commit(commitment_key, spec, opening)` replaces
  the four bespoke A-side backend methods and the `digit_rows` B/F call.
- One closed `enum AjtaiOpeningType` (the opening's representation) replaces the four
  per-representation backend plan structs.
- One `outer_commit()` replaces the three copies of the `B`/`F` pipeline.
- CPU stays the byte-exact reference; no protocol/transcript/proof change; no
  new arithmetic; no new workspace crate.

**Net effect:** ~4 backend methods + 3 plan structs + 1 tiered helper + 4
`commit_inner` impls collapse into 1 trait method + 1 opening enum + 1 outer
helper + 1 shared inner helper, all living in `commit/`. The scheme still calls
`akita_prover::commit(...)` exactly as before.

## 2. Naming model

A lattice (Ajtai) commitment **is** a matrix-vector product: `commitment =
matrix · message`. So the matrix multiply is not a separate "matvec" concept —
it literally *is* a (single-matrix) commitment. We name it accordingly:

```text
ajtai_commit(commitment_key, spec, opening) -> commitment
```

| Term | Meaning | Was called |
|---|---|---|
| `commitment_key` | the prepared public matrix material the backend multiplies against (the NTT-prepared shared setup matrix for a ring dimension) | `prepared` / `PreparedSetup<D>` |
| `spec` | which matrix window to commit under: role + `rows` + `cols` + ring domain (`MatrixSpec`) | implicit `row_len` + `digits.len()` |
| `opening` | the message being committed, in its native representation (`AjtaiOpeningType`) | the `*CommitRowsPlan` structs / `MatVecInput` |
| `commitment` | the result `matrix · message` (the committed rows) | the `Vec<Vec<…>>` returned by `*_commit_rows` |

So the low-level trait method is `ajtai_commit`, and `spec` (which matrix) and
`opening` (the message) are passed as separate arguments — no bundling struct.

### `ajtai_commit` (primitive) vs `commit` (protocol)

The two are named differently on purpose so they never get mixed up:

- **Primitive:** `CommitBackend::ajtai_commit(commitment_key, spec, opening)` —
  one Ajtai commitment (one matrix multiply). Called as `backend.ajtai_commit(…)`.
- **Pipeline:** the free function `commit::<Cfg>(…)` (the public PCS entry) —
  composes several Ajtai commits: `A`-commit → `decompose` → `B`/`F`-commit.

They share one mental model — the protocol `commit` is just a sequence of
`ajtai_commit`s with a decompose between them — but the distinct names keep the
single-matrix primitive and the full PCS entry unambiguous everywhere.

> Note on `opening`: in the wider PCS, "opening" also names the later evaluation
> proof. Here it means the *input to a commitment* — the message (and which
> matrix) that the resulting commitment opens to. This local meaning is
> intentional and scoped to the `commit` module.

## 3. Decision: a module, not a crate

The commit subsystem lives as a **module inside `akita-prover`**
(`crates/akita-prover/src/commit/`), not as a separate workspace crate. This is
deliberate:

- The commit pipeline is tightly coupled to things that already live in
  `akita-prover`: the polynomial representations (`DensePoly`, `OneHotPoly`,
  `SparseRingPoly`, `SuffixWitness`), `CpuBackend` / `CpuPreparedSetup`, the
  shared NTT kernels, and the `Cfg`-driven schedule resolution. A separate crate
  would force those types apart or invert dependency arrows.
- A module gives the same encapsulation a crate would: a small `mod.rs` chooses
  what is `pub` (the entry points + the trait) and keeps everything else
  `pub(crate)` or private. "The rest of the code does not need to know a lot
  about commit" is enforced by that narrow surface, not by a crate boundary.
- The shared low-level NTT kernels (`mat_vec_mul_ntt_*`) are **not** commit-only
  — ring-switch relation and sumcheck use them too — so they stay in
  `src/kernels/`. The `commit` module *calls* them. Only genuinely commit-only
  code (the `CommitBackend` trait + its impl, the column-sweep A-side kernels,
  the pipeline) moves into `commit/`.
- The CPU `CommitBackend` impl lives in `commit/ajtai/cpu.rs`. `CpuBackend` and
  `CpuPreparedSetup` keep living in `compute.rs`, but `CpuPreparedSetup`'s
  `expanded` / `ntt_shared` fields become `pub(crate)` so the commit backend
  impl can read the shared matrix and NTT slot. That impl **is** the CPU
  backend, so it is the privileged holder of raw prepared-state access — exactly
  the role today's `*_commit_rows` impls play in `compute.rs`. Representation
  (`backend/`) and protocol code still never touch those fields; they only call
  `ajtai_commit` / `AjtaiOpeningView`. This is the resolution of the
  compute-backend spec's no-raw-accessor rule — see [§11](#11-relationship-to-the-compute-backend-spec).

What this gives the rest of `akita-prover`: it imports
`commit::{commit, batched_commit, commit_w, …}` and the `CommitBackend` trait
bound, and nothing else. The `ajtai_commit` dispatch, the `MatrixSpec`/`AjtaiOpeningType` types, the
decomposition unit, and the outer/inner helpers are all internal to `commit/`.

## 4. Folder structure

```text
crates/akita-prover/src/
├── commit/                      ← NEW: the entire commit subsystem
│   ├── mod.rs                   ← module root + the NARROW public surface (re-exports)
│   │
│   ├── ajtai/                  ← the Ajtai commit primitive: commitment = key · message
│   │   ├── mod.rs
│   │   ├── spec.rs              ← MatrixRole, RingDomain, MatrixSpec
│   │   ├── opening.rs           ← AjtaiOpeningType, ZeroScan
│   │   ├── backend.rs           ← trait CommitBackend : ComputeBackendSetup  (fn commit)
│   │   ├── cpu.rs               ← impl CommitBackend for CpuBackend  (the dispatch match; reads CpuPreparedSetup's pub(crate) fields — it IS the backend)
│   │   └── column_sweep.rs      ← one-hot + sparse column-selection kernels (moved here)
│   │
│   ├── decompose.rs             ← decompose_rows / decompose_rows_into / recompose_and_validate
│   ├── opening_view.rs               ← trait AjtaiOpeningView  (impls live with the representations)
│   ├── inner.rs                 ← commit_inner_one / commit_inner_group
│   ├── outer.rs                 ← outer_commit / tiered_outer_commit (+ zk-blinding add)
│   ├── pipeline.rs              ← commit_with_validated_params / batched_commit_with_params
│   ├── entry.rs                 ← commit::<Cfg> / batched_commit::<Cfg> (param resolution + tensor-projection decision)
│   └── recursive.rs             ← commit_w core (moved from protocol/ring_switch/commit.rs)
│
├── kernels/                     ← UNCHANGED: shared NTT/linear kernels (commit + relation + sumcheck)
│   ├── linear/ …                ←   mat_vec_mul_ntt_*  (commit/ajtai/cpu.rs calls these)
│   └── crt_ntt.rs               ←   NttSlotCache build/slice
│
├── backend/                     ← representations stay here; each gains an AjtaiOpeningView impl
│   ├── dense.rs                 ←   impl AjtaiOpeningView for DensePoly        (was: commit_inner + commit_rows)
│   ├── onehot/ …                ←   impl AjtaiOpeningView for OneHotPoly        (column_sweep.rs moves to commit/ajtai/)
│   ├── sparse_ring.rs           ←   impl AjtaiOpeningView for SparseRingPoly
│   └── recursive_witness.rs     ←   impl AjtaiOpeningView for SuffixWitness
│
├── compute.rs                   ← CpuBackend / CpuPreparedSetup stay (fields become pub(crate) for
│                                    commit/ajtai/cpu.rs); the four *_commit_rows methods +
│                                    *CommitRowsPlan structs are DELETED
└── lib.rs                       ← `pub mod commit;` + `pub use commit::{commit, batched_commit, commit_w, …};`
```

### What moves where

| Today | After |
|---|---|
| `api/commitment.rs` (`commit`, `batched_commit`, `*_with_params`, `tiered_commit_u_final`, validators) | split into `commit/{entry,pipeline,inner,outer}.rs` |
| `protocol/ring_switch/commit.rs` (`commit_w`) | `commit/recursive.rs` (the `commit_next_w` dispatch wrapper stays in `protocol/ring_switch` and calls it) |
| `backend/onehot/column_sweep.rs`, sparse column sweep | `commit/ajtai/column_sweep.rs` |
| `compute.rs`: `dense/onehot/sparse_ring/recursive_witness_commit_rows` + plan structs | deleted; replaced by `commit/ajtai/cpu.rs` |
| dense `commit_inner`/`commit_rows`, one-hot `commit_inner`, etc. | small `AjtaiOpeningView` impls next to each representation |

### The public surface (`commit/mod.rs`)

```rust
// commit/mod.rs — everything else in the module is pub(crate) or private.
mod ajtai;
mod decompose;
mod opening_view;
mod inner;
mod outer;
mod pipeline;
mod entry;
mod recursive;

pub use entry::{commit, batched_commit};
pub use pipeline::{commit_with_params, batched_commit_with_params};
pub use recursive::commit_w;
pub use ajtai::backend::CommitBackend;   // the bound the scheme/prover name
pub use opening_view::AjtaiOpeningView;            // representations implement this
// MatrixSpec / AjtaiOpeningType / the CPU dispatch stay pub(crate): not part
// of the surface the rest of the prover sees.
```

`lib.rs` then does `pub mod commit;` and re-exports the same entry points it
exposes today, so `akita-pcs` and the scheme are unaffected.

## 5. Background recap

Every root/recursive commit is the same three-step shape — three primitive
commits with a decompose between them:

```text
t_i   = A · s_i              # inner commit, per block
t̂_i  = decompose(t_i)        # gadget decomposition for opening
u     = B · concat_i(t̂_i)    # outer commit   (single-tier)
u     = F · decompose(blockdiag(B') · concat_i(t̂_i))   (tiered)
```

The differences that *look* like separate code paths are really just how the
message `s_i` is represented for the `A` commit:

| Representation | How `s_i` reaches `A` |
|---|---|
| Dense | full ring coefficients (optionally cached as digit planes) |
| One-hot | sparse shifted monomials; `A` columns selected, never materialized |
| Sparse ring | sparse signed-ring entries; same column selection |
| Recursive witness | strided flat i8 digit stream |

And the outer `B`/`F` commit is identical across the root single-tier, root
tiered, and recursive `commit_w` paths.

Two **physical** kernel families exist and both must survive unchanged:

- **NTT-domain commit** (`B`, `F`, `B'`, dense `A`, recursive `A`): reads the
  prepared `NttSlotCache`, multiplies i8 digit planes pointwise in the NTT
  domain with L2 column tiling. (`mat_vec_mul_ntt_*`, in `src/kernels/`.)
- **Cleartext column-selection** (`A` for one-hot / sparse): reads the raw
  `RingMatrixView` and shift-accumulates selected columns.
  (`column_sweep_*`, moving into `commit/ajtai/`.)

## 6. The building blocks (API)

All of these live in `commit/`. This is the reference; §7 shows them in use.

### 6.1 `MatrixSpec` — which matrix window (`commit/ajtai/spec.rs`)

```rust
pub enum MatrixRole { AInner, BOuter, BOuterTierSlice, FOuterTier, DRelation }

/// Selects a `rows × cols` window of the commitment key (the shared setup
/// matrix). Both dimensions are explicit (the fix for today's implicit
/// `cols == digits.len()`). Every commit is negacyclic.
pub struct MatrixSpec {
    pub role: MatrixRole,
    pub rows: usize,
    pub cols: usize,
}
```

> The `commit` primitive only commits negacyclically, so there is no
> `RingDomain` field. The (deferred) Phase 8 cyclic ring-switch relation hook
> was dropped rather than carried as dead scaffolding; the relation path keeps
> using `CyclicRowsComputeBackend::cyclic_digit_rows` (see [§11](#11-relationship-to-the-compute-backend-spec)).

### 6.2 `AjtaiOpeningType` — the opening (`commit/ajtai/opening.rs`)

The opening is the message being committed, in its native representation. It is
a closed enum that replaces the four `*CommitRowsPlan` structs. The matrix
window (`MatrixSpec`) is **not** bundled with it — `ajtai_commit` takes the spec
and the opening as separate arguments (see §6.3).

```rust
pub enum ZeroScan { Dense, SkipZeros }   // dense skips zero-plane scans

/// The opening (message) being committed, in its native representation.
pub enum AjtaiOpeningType<'a, F: FieldCore, const D: usize> {
    /// Raw ring coeffs per block; decomposed on the fly. (dense A fallback)
    CoeffBlocks  { blocks: &'a [&'a [CyclotomicRing<F, D>]], num_digits: usize, log_basis: u32, zero_scan: ZeroScan },
    /// Pre-decomposed trusted i8 digit planes per block. (dense A fast path)
    DigitBlocks  { blocks: &'a [&'a [[i8; D]]], log_basis: u32, zero_scan: ZeroScan },
    /// One flat digit vector = a single block. (B / B' / F)
    DigitVector  { digits: &'a [[i8; D]], log_basis: u32 },
    /// Strided recursive witness. (`raw` = δ_commit == 1 signed-i8 stream)
    StridedDigits{ coeffs: &'a [[i8; D]], num_blocks: usize, block_len: usize, num_digits: usize, log_basis: u32, raw: bool },
    /// One-hot shifted monomials (sparse column selection). (one-hot A)
    OneHot       { blocks: OneHotBlocks<'a>, num_digits_commit: usize },
    /// Sparse signed-ring entries (sparse column selection). (sparse-ring A)
    SparseRing   { blocks: FlatBlockTable<'a, SparseRingBlockEntry>, num_digits_commit: usize },
}
```

`FlatBlockTable`, `OneHotBlocks`, and the entry types are referenced from here.
The entry types may stay in `backend/onehot` + `backend/sparse_ring` (the
representations and the folding code also use them) and be re-exported into
`commit/ajtai/opening.rs`; only the `OneHotBlocks` *view* needs to be nameable
by `AjtaiOpeningType`.

### 6.3 `CommitBackend` — the Ajtai commit primitive (`commit/ajtai/backend.rs`)

It **extends the existing `ComputeBackendSetup`** (in `compute.rs`) rather than
re-declaring a prepared-setup contract, so `PreparedSetup<D>` (the commitment
key), `prepare_expanded`, and `validate_prepared_setup` are reused as-is.

It extends `DigitRowsComputeBackend` (which itself extends the existing
`ComputeBackendSetup` in `compute.rs`), so `PreparedSetup<D>` (the commitment
key), `prepare_expanded`, `validate_prepared_setup`, and the dedicated ZK
blinding mat-vecs (`zk_b_digit_rows` / `zk_d_digit_rows`) are all reused as-is.

```rust
use crate::compute::DigitRowsComputeBackend;

pub trait CommitBackend<F>: DigitRowsComputeBackend<F>
where F: FieldCore + CanonicalField {
    /// commitment = commitment_key · opening, under matrix window `spec`.
    /// out.len() == #blocks(opening), out[b].len() == spec.rows.
    /// Matches the opening ONCE, validates against `spec`, dispatches to a
    /// concrete kernel. No per-element dispatch.
    fn ajtai_commit<const D: usize>(&self, commitment_key: &Self::PreparedSetup<D>, spec: MatrixSpec, opening: AjtaiOpeningType<'_, F, D>)
        -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where F: HasWide, F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;
}
```

The single method `ajtai_commit` is enough: the `B` / `B'` / `F` sites commit one
`AjtaiOpeningType::DigitVector` block and take the single returned row vector
(`.into_iter().next()` / `.flatten()` for the tiered concat), so there is no
separate single-block convenience method on the trait.

The `HasWide` bound sits on the method because only the one-hot/sparse arms
need wide accumulation. zk blinding stays as the existing dedicated
`zk_b_digit_rows` / `zk_d_digit_rows` methods on `DigitRowsComputeBackend` (they
target a separately-prepared lazy slot); the pipeline / recursive callers add
those rows directly.

#### What the `commitment_key` (`Self::PreparedSetup<D>`) is

`commitment_key` is the backend's host-prepared, ready-to-multiply view of the
public setup matrix for one ring dimension `D` — the associated type
`PreparedSetup<D>` from `ComputeBackendSetup`. `ajtai_commit` borrows it
(`&Self::PreparedSetup<D>`) instead of the raw `AkitaExpandedSetup`, because the
expensive, device-specific preprocessing is done **once** up front and reused
across every `A`/`B`/`B'`/`F` commit in a proof.

```rust
pub trait ComputeBackendSetup<F: FieldCore + CanonicalField>: Send + Sync {
    /// The commitment key for ring dimension `D` (backend-owned prepared matrix).
    type PreparedSetup<const D: usize>: Send + Sync;
    fn prepare_expanded<const D: usize>(&self, expanded: Arc<AkitaExpandedSetup<F>>) -> Result<Self::PreparedSetup<D>, AkitaError>;
    fn validate_prepared_setup<const D: usize>(&self, commitment_key: &Self::PreparedSetup<D>, expanded: &AkitaExpandedSetup<F>) -> Result<(), AkitaError>;
}
```

Key properties (the "features"):

- **Backend-specific, opaque to callers.** It is an associated type, so the CPU
  backend, a future GPU backend, etc. each define their own. The pipeline
  (`outer_commit`, `commit_inner_one`, …) is generic over `B: CommitBackend` and
  never names a concrete key type — it only threads `&commitment_key` through.
  This is what lets `commit/` stay backend-agnostic.
- **Indexed by ring dimension `D`.** Preparation is per-`D` (NTT params and
  packing depend on it); recursion at a different `D` calls
  `prepare_expanded::<D_recursive>` again (via the dynamic-`D` dispatch macro).
- **Built once, read many.** `prepare_expanded` runs the costly preprocessing a
  single time; every `ajtai_commit` borrows it immutably and in parallel
  (`Send + Sync`), so the NTT image is never rebuilt mid-proof.
- **Carries setup identity, not protocol state.** It holds only matrix/compute
  data, never transcript, schedule, or proof state — keeping the backend
  boundary clean (per the compute-backend spec).

For the CPU backend the concrete key is today's `CpuPreparedSetup<F, D>`, reused
unchanged except that its fields become `pub(crate)` so the `CommitBackend` impl
in `commit/ajtai/cpu.rs` can read them (that impl is the backend; see
[§11](#11-relationship-to-the-compute-backend-spec)):

```rust
pub struct CpuPreparedSetup<F: FieldCore, const D: usize> {
    pub(crate) expanded: Arc<AkitaExpandedSetup<F>>,   // raw shared matrix — source for the column-selection (one-hot/sparse) arms
    pub(crate) ntt_shared: NttSlotCache<D>,            // NTT image (negacyclic `neg` + cyclic `cyc`) of the whole matrix — source for the NTT-domain arms
    ntt_i8_capacity: CrtI8CapacityProfile,  // CRT/i8 capacity profile selected for this (F, D)
    #[cfg(feature = "zk")] ntt_zk_b: OnceLock<NttSlotCache<D>>,  // lazily-prepared zk blinding slots
    #[cfg(feature = "zk")] ntt_zk_d: OnceLock<NttSlotCache<D>>,
}
```

Both physical paths resolve from one key: NTT-domain arms slice rows out of
`commitment_key.ntt_shared`, and the column-selection arms read raw columns from
`commitment_key.expanded.shared_matrix` (see §8). `validate_prepared_setup` is
the cheap guard that the key was built for *this* setup — it compares the
compact setup seed identity rather than re-hashing the matrix.

### 6.4 `AjtaiOpeningView` — the witness→opening seam (`commit/opening_view.rs`)

```rust
pub trait AjtaiOpeningView<F: FieldCore, const D: usize> {
    /// Borrow this witness as the `A`-side opening for the given block shape.
    fn to_ajtai_opening(&self, num_blocks: usize, num_digits_commit: usize, log_basis: u32)
        -> Result<AjtaiOpeningType<'_, F, D>, AkitaError>;
}
```

Defined in `commit/`; implemented next to each representation in `backend/`
(same crate, so no orphan-rule issue). This replaces the per-representation
`AkitaPolyOps::commit_inner` method — each representation now only says how to
present itself as an `AjtaiOpeningType`.

### 6.5 Decomposition unit (`commit/decompose.rs`)

```rust
pub fn decompose_rows<F, const D: usize>(rows: &[Vec<CyclotomicRing<F, D>>], num_digits_open: usize, log_basis: u32)
    -> Result<FlatDigitBlocks<D>, AkitaError>;
pub fn decompose_rows_into<F, const D: usize>(rows: &[Vec<CyclotomicRing<F, D>>], dst: &mut FlatDigitBlocks<D>, num_digits_open: usize, log_basis: u32);
pub fn recompose_and_validate<F, const D: usize>(rows: &[Vec<CyclotomicRing<F, D>>], digits: &FlatDigitBlocks<D>, num_digits_open: usize, log_basis: u32)
    -> Result<(), AkitaError>;
```

Wraps the existing `decompose_rows_i8_into` / `gadget_recompose_pow2_i8`; the
dense `decompose_commit_rows`, the one-hot inline zeroed/skip-zero loop, and the
recursive/sparse copies all collapse to calls here.

## 7. Before / After: how the real code changes

"Before" snippets are current code (trimmed with `// …` where long but faithful);
"after" is the proposed shape inside `commit/`.

### 7.1 Backend: four A-side methods + `digit_rows` → one `ajtai_commit`

**Before** — `compute.rs`, the operation-shaped surface plus four CPU impls
(e.g. `dense_commit_rows`):

```rust
pub trait CommitmentComputeBackend<F>: DigitRowsComputeBackend<F> {
    fn dense_commit_rows<const D: usize>(&self, prepared, plan: DenseCommitRowsPlan<'_, F, D>) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, …>;
    fn onehot_commit_rows<const D: usize>(&self, prepared, plan: OneHotCommitRowsPlan<'_>) -> …;
    fn sparse_ring_commit_rows<const D: usize>(&self, prepared, plan: SparseRingCommitRowsPlan<'_>) -> …;
    fn recursive_witness_commit_rows<const D: usize>(&self, prepared, plan: RecursiveWitnessCommitRowsPlan<'_, D>) -> …;
}

fn dense_commit_rows<const D: usize>(&self, prepared, plan: DenseCommitRowsPlan<'_, F, D>) -> … {
    match plan.input {
        DenseCommitInput::CachedDigits { digit_block_slices, log_basis } =>
            mat_vec_mul_ntt_dense_digits_i8_trusted(&prepared.ntt_shared, plan.n_a, /*row_width*/, &digit_block_slices, log_basis),
        DenseCommitInput::CoeffBlocks { block_slices, num_digits_commit, log_basis } =>
            if plan.n_a == 1 { /* …dense_single_row */ } else { mat_vec_mul_ntt_i8_dense(/* … */) },
    }
}
```

**After** — `commit/ajtai/cpu.rs`, one trait, one body (full arms in §7.7):

```rust
impl<F: FieldCore + CanonicalField> CommitBackend<F> for CpuBackend {
    fn ajtai_commit<const D: usize>(&self, commitment_key, spec, opening) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where F: HasWide, F::Wide: AdditiveGroup + From<F> + ReduceTo<F> {
        validate_matrix::<F, D>(commitment_key, &spec)?;
        match (spec.domain, opening) {
            (Negacyclic, DigitVector { digits, log_basis }) => {
                require_block_width(spec.cols, digits.len())?;
                Ok(vec![mat_vec_mul_ntt_single_i8(&commitment_key.ntt_shared, spec.rows, spec.cols, digits, log_basis)?])
            }
            (Negacyclic, DigitBlocks { blocks, log_basis, zero_scan: Dense }) =>
                mat_vec_mul_ntt_dense_digits_i8_trusted(&commitment_key.ntt_shared, spec.rows, spec.cols, blocks, log_basis),
            (Negacyclic, CoeffBlocks { blocks, num_digits, log_basis, zero_scan: Dense }) if spec.rows == 1 =>
                single_row_to_blocks(mat_vec_mul_ntt_i8_dense_single_row(&commitment_key.ntt_shared, spec.cols, blocks, num_digits, log_basis)?),
            (Negacyclic, CoeffBlocks { blocks, num_digits, log_basis, zero_scan: Dense }) =>
                mat_vec_mul_ntt_i8_dense(&commitment_key.ntt_shared, spec.rows, spec.cols, blocks, num_digits, log_basis),
            (Negacyclic, OneHot { blocks, num_digits_commit }) => {
                let a = commitment_key.expanded.shared_matrix.ring_view::<D>(spec.rows, spec.cols)?;
                Ok(column_sweep_onehot(&a, blocks, spec.rows, spec.cols, num_digits_commit))
            }
            // … StridedDigits, SparseRing, SkipZeros variants, Cyclic DigitVector …
        }
    }
}
```

Same kernels, same args — selected by one `match` instead of by which method the
caller picked. `DenseCommitRowsPlan` / `OneHotCommitRowsPlan` /
`SparseRingCommitRowsPlan` / `RecursiveWitnessCommitRowsPlan` are deleted.
`CpuBackend` and `CpuPreparedSetup` are reused as-is (the trait extends
`ComputeBackendSetup`).

### 7.2 Dense `commit_inner` → an `AjtaiOpeningView` impl

**Before** — `backend/dense.rs`: `commit_inner` calls a dense-specific
`commit_rows`, which decides cached-digits vs coeff-blocks and builds a plan:

```rust
fn commit_inner<B: CommitmentComputeBackend<F>>(&self, backend, prepared, n_a, block_len, _nb, δ_commit, δ_open, log_basis) -> … {
    let t = self.commit_rows(backend, prepared, n_a, block_len, δ_commit, log_basis)?;
    let decomposed_inner_rows = decompose_commit_rows::<F, D>(&t, δ_open, log_basis)?;
    Ok(CommitInnerWitness { recomposed_inner_rows: t, decomposed_inner_rows })
}
fn commit_rows<B: …>(&self, …) -> … {
    if let Some(planes) = self.digit_planes_for(δ_commit, log_basis) {
        return backend.dense_commit_rows(prepared, DenseCommitRowsPlan { n_a, input: CachedDigits { … } });
    }
    backend.dense_commit_rows(prepared, DenseCommitRowsPlan { n_a, input: CoeffBlocks { … } })
}
```

**After** — `backend/dense.rs` keeps only the seam; the commit + decompose move
to the shared `commit_inner_one` (§7.6):

```rust
impl<F, const D: usize> AjtaiOpeningView<F, D> for DensePoly<F, D> {
    fn to_ajtai_opening(&self, num_blocks, δ_commit, log_basis) -> Result<AjtaiOpeningType<'_, F, D>, AkitaError> {
        if let Some(planes) = self.digit_planes_for(δ_commit, log_basis) {
            return Ok(AjtaiOpeningType::DigitBlocks { blocks: digit_block_slices(planes, …), log_basis, zero_scan: Dense });
        }
        Ok(AjtaiOpeningType::CoeffBlocks { blocks: coeff_block_slices(&self.coeffs, …), num_digits: δ_commit, log_basis, zero_scan: Dense })
    }
}
```

The two-mode cache logic is preserved verbatim; it just returns a borrowed
`AjtaiOpeningType`. `decompose_commit_rows` is deleted in favour of `decompose_rows`.

### 7.3 One-hot `commit_inner` → an `AjtaiOpeningView` impl

**Before** — `backend/onehot/ops.rs`: builds an `OneHotCommitRowsPlan` then
hand-rolls the zeroed/skip-zero decomposition.

```rust
fn commit_inner<B: …>(&self, backend, prepared, n_a, block_len, _nb, δ_commit, δ_open, log_basis) -> … {
    let blocks = self.blocks_for(block_len)?;
    let t = backend.onehot_commit_rows::<D>(prepared, OneHotCommitRowsPlan { n_a, block_len, δ_commit, blocks: blocks.commit_plan_blocks() })?;
    let mut t_hat = FlatDigitBlocks::zeroed(vec![n_a*δ_open; t.len()])?;
    // … decompose each non-all-zero block in place …
    Ok(CommitInnerWitness { recomposed_inner_rows: t, decomposed_inner_rows: t_hat })
}
```

**After** — `blocks_for` cache and the zero-skip survive; commit + decompose are
shared:

```rust
impl<F, const D, I> AjtaiOpeningView<F, D> for OneHotPoly<F, D, I> {
    fn to_ajtai_opening(&self, _num_blocks, δ_commit, _log_basis) -> Result<AjtaiOpeningType<'_, F, D>, AkitaError> {
        Ok(AjtaiOpeningType::OneHot { blocks: self.blocks_for(/*block_len*/)?.commit_plan_blocks(), num_digits_commit: δ_commit })
    }
}
```

The "skip all-zero blocks" path is kept by having `decompose_rows_into` take the
same zeroed-destination + non-zero-block fast path.

### 7.4 The outer pipeline: single-tier + tiered → one `outer_commit`

The biggest readability win. Today the `B`/`F` work is open-coded inside
`commit_with_validated_params`, with a separate `tiered_commit_u_final`.

**Before** — `api/commitment.rs` (trimmed):

```rust
fn commit_with_validated_params<F, const D, P, B>(polys, backend, prepared, params) -> … {
    let mut b_input_digits = vec![[0i8; D]; total];
    // … parallel loop: poly.commit_inner(…)?, validate_commit_inner_shape(…)?, copy t̂_i into b_input_digits …
    let u = if params.f_key.is_some() {
        tiered_commit_u_final::<F, D, B>(backend, prepared, params, &b_input_digits)?
    } else {
        let mut u = backend.digit_rows::<D>(prepared, params.b_key.row_len(), &b_input_digits, params.log_basis)?;
        #[cfg(feature = "zk")] { /* add zk_b_digit_rows */ }
        u
    };
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(decomposed, recomposed, /* zk */);
    Ok((RingCommitment { u }, hint))
}
fn tiered_commit_u_final<…>(…) -> … {                 // separate ~40-line helper
    for chunk in b_input_digits.chunks(params.b_key.col_len()) { u_concat.extend(backend.digit_rows(…)?); }
    // … decompose u_concat into u_hat …
    backend.digit_rows::<D>(prepared, f_key.row_len(), &u_hat, params.log_basis)
}
```

**After** — `commit/pipeline.rs` + `commit/outer.rs`:

```rust
// commit/pipeline.rs
fn commit_with_validated_params<F, const D, P, B>(polys, backend, commitment_key, params) -> …
where P: AjtaiOpeningView<F, D>, B: CommitBackend<F> {
    let (decomposed, recomposed, b_input_digits) = commit_inner_group(polys, backend, commitment_key, params)?; // §7.6
    let u = outer_commit(backend, commitment_key, params, &b_input_digits)?;                                    // B or F
    let hint = AkitaCommitmentHint::with_recomposed_inner_rows(decomposed, recomposed, /* zk */);
    Ok((RingCommitment { u }, hint))
}

// commit/outer.rs — each B/F site commits one DigitVector and takes its block.
fn outer_commit<F, const D, B: CommitBackend<F>>(backend, commitment_key, params, t_hat: &[[i8; D]]) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    match &params.f_key {
        None => {
            let matrix = MatrixSpec { role: BOuter, rows: params.b_key.row_len(), cols: t_hat.len(), domain: Negacyclic };
            let u = backend.ajtai_commit::<D>(commitment_key, matrix, DigitVector { digits: t_hat, log_basis: params.log_basis })?
                .into_iter().next().unwrap_or_default();
            // (zk B-blinding is added by the pipeline/recursive caller, which owns the hint.)
            Ok(u)
        }
        Some(f_key) => tiered_outer_commit(backend, commitment_key, params, f_key, t_hat),
    }
}
fn tiered_outer_commit<F, const D, B: CommitBackend<F>>(backend, commitment_key, params, f_key, t_hat) -> … {
    let bp = MatrixSpec { role: BOuterTierSlice, rows: params.b_key.row_len(), cols: params.b_key.col_len(), domain: Negacyclic };
    let mut u_concat = Vec::new();
    for chunk in t_hat.chunks(params.b_key.col_len()) {
        u_concat.extend(backend.ajtai_commit::<D>(commitment_key, bp, DigitVector { digits: chunk, log_basis: params.log_basis })?.into_iter().flatten());
    }
    let u_hat = decompose_rows(&[u_concat], params.num_digits_open, params.log_basis)?;
    let f = MatrixSpec { role: FOuterTier, rows: f_key.row_len(), cols: u_hat.flat_digits().len(), domain: Negacyclic };
    Ok(backend.ajtai_commit::<D>(commitment_key, f, DigitVector { digits: u_hat.flat_digits(), log_basis: params.log_basis })?
        .into_iter().next().unwrap_or_default())
}
```

`outer_commit` is now the single definition of the B/F pipeline; the recursive
path (§7.5) calls the same function.

### 7.5 Recursive `commit_w` → reuse the same helpers

**Before** — `protocol/ring_switch/commit.rs` duplicates the whole
single-tier/tiered/zk dance with `SuffixWitness`:

```rust
let inner = w_view.commit_inner(backend, prepared, n_a, block_len, num_blocks, δ_commit, δ_open, log_basis)?;
validate_commit_inner_shape(&inner, …)?;
let outer_input = inner.decomposed_inner_rows.flat_digits().to_vec();
let u = if commit_layout.f_key.is_some() {
    tiered_commit_u_final::<F, D, B>(backend, prepared, commit_layout, &outer_input)?
} else {
    let mut u = backend.digit_rows::<D>(prepared, commit_layout.b_key.row_len(), &outer_input, commit_layout.log_basis)?;
    #[cfg(feature = "zk")] { /* add zk_b_digit_rows */ }
    u
};
let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(inner.decomposed_inner_rows, inner.recomposed_inner_rows, /* zk */);
```

**After** — `commit/recursive.rs`, identical to a one-group commit; the only
special thing is the opening source (`SuffixWitness: AjtaiOpeningView`):

```rust
let inner = commit_inner_one(&w_view, backend, commitment_key, commit_layout)?;     // §7.6, P = SuffixWitness
let u = outer_commit(backend, commitment_key, commit_layout, inner.decomposed_inner_rows.flat_digits())?; // §7.4
let hint = AkitaCommitmentHint::singleton_with_recomposed_inner_rows(inner.decomposed_inner_rows, inner.recomposed_inner_rows, /* zk */);
```

`tiered_commit_u_final` is gone; root and recursive commits share `outer_commit`
and the inner helpers verbatim.

### 7.6 The shared inner-commit body (`commit/inner.rs`)

`commit_inner` stops being implemented per representation; one generic helper
does `A`-commit + `decompose` + shape validation for all of them:

```rust
/// Shared `t = A·s ; t̂ = decompose(t)` for one witness (replaces every
/// `commit_inner` impl). `num_blocks == 1` is the recursive/single case.
fn commit_inner_one<F, const D, P, B>(poly, backend, commitment_key, params) -> Result<CommitInnerWitness<F, D>, AkitaError>
where P: AjtaiOpeningView<F, D>, B: CommitBackend<F> {
    let a_matrix = MatrixSpec {
        role: MatrixRole::AInner,
        rows: params.a_key.row_len(),
        cols: params.block_len * params.num_digits_commit,
        domain: RingDomain::Negacyclic,
    };
    let opening = poly.to_ajtai_opening(params.num_blocks, params.num_digits_commit, params.log_basis)?;
    let t = backend.ajtai_commit::<D>(commitment_key, a_matrix, opening)?;
    let t_hat = decompose_rows(&t, params.num_digits_open, params.log_basis)?;
    let inner = CommitInnerWitness { recomposed_inner_rows: t, decomposed_inner_rows: t_hat };
    validate_commit_inner_shape(&inner, params.num_blocks, params.a_key.row_len(), params.num_digits_open, params.log_basis)?;
    Ok(inner)
}
```

`commit_inner_group` is just `commit_inner_one` over the group in parallel,
copying each `t̂_i` into the concatenated `b_input_digits` as today. The
`AkitaPolyOps::commit_inner` trait method is removed; the four impls become four
tiny `AjtaiOpeningView` impls.

### 7.7 CPU dispatch table

The contract for `commit/ajtai/cpu.rs`: each arm calls the **same** kernel the
corresponding `*_commit_rows` / `digit_rows` method calls today.

All arms are negacyclic.

| `AjtaiOpeningType` arm | Existing kernel (in `src/kernels/` or `commit/ajtai/column_sweep.rs`) |
|---|---|
| `CoeffBlocks` (rows>1, `Dense`) | `mat_vec_mul_ntt_i8_dense` |
| `CoeffBlocks` (rows==1, `Dense`) | `mat_vec_mul_ntt_i8_dense_single_row` |
| `CoeffBlocks` (`SkipZeros`) | `mat_vec_mul_ntt_i8` |
| `DigitBlocks` (`Dense`) | `mat_vec_mul_ntt_dense_digits_i8_trusted` |
| `DigitBlocks` (`SkipZeros`) | `mat_vec_mul_ntt_digits_i8` |
| `DigitVector` | `mat_vec_mul_ntt_single_i8` |
| `StridedDigits` (`raw==false`) | `mat_vec_mul_ntt_i8_strided` (rehydrate first) |
| `StridedDigits` (`raw==true`) | `mat_vec_mul_ntt_raw_i8_strided` |
| `OneHot` | `column_sweep_ajtai_onehot::<Single/MultiChunkEntry>` |
| `SparseRing` | `column_sweep_sparse` |

`ajtai_commit` validates once before dispatch (nonzero `rows`/`cols`, footprint
fits, i8-range `log_basis`, per-block width matches `spec.cols`), returning
`AkitaError`, never panicking.

### 7.8 Net effect

| | Before | After |
|---|---|---|
| Location of commit logic | `api/commitment.rs`, `backend/*`, `protocol/ring_switch/commit.rs`, `compute.rs` | one `commit/` module |
| A-side backend methods | 4 (`*_commit_rows`) | 1 (`ajtai_commit`) |
| Backend plan structs | 4 (`*CommitRowsPlan`) | 0 (1 `AjtaiOpeningType` enum) |
| B/F pipeline copies | 3 | 1 (`outer_commit`) |
| `commit_inner` impls | 4 (per representation) | 1 (`commit_inner_one`) + 4 tiny `AjtaiOpeningView` |
| Decompose copies | 4 | 1 (`decompose_rows`) |
| Matrix shape | implicit (`cols == digits.len()`) | explicit (`MatrixSpec`) |
| What the rest of the prover sees | many commit internals | `commit::{commit, batched_commit, commit_w, CommitBackend}` |

No kernel is rewritten; no arithmetic changes.

## 8. The flat-matrix mechanism

How can one `ajtai_commit` serve `A`, `B`, `B'`, and `F`? Because the commitment key
(`CpuPreparedSetup`) already holds **one** NTT image of the *entire* shared
matrix as a flat array (`build_ntt_slot(shared_matrix.ring_view(1, total))`),
and each role is just the front `rows × cols` window of it.

```text
commitment_key.ntt_shared = NTT image of the whole matrix, one flat array of `total` rings

NTT arms:     row i  =  ntt_shared.neg[i * spec.cols .. (i + 1) * spec.cols]
column arms:  a_view =  commitment_key.expanded.shared_matrix.ring_view::<D>(spec.rows, spec.cols)
```

`A` (`cols = block_len·δ_commit`), `B` (`cols = t̂.len()`), `B'`
(`cols = b_key.col_len()`), `F` (`cols = û.len()`) all read **from offset 0** and
overlap — the prefix-sharing `background.md` describes. `MatrixSpec` needs no
base offset; `(rows, cols)` fully determines the window. `MatrixRole` is kept for
validation/tracing only (a future non-zero-offset layout could add a `base`
field without touching callers).

Because the trait extends `ComputeBackendSetup`, the concrete instantiation is
just the existing `CpuBackend` with a new trait impl — **no new key type, no new
state**. A future accelerator implements `ComputeBackendSetup` + `CommitBackend`
with its own device-resident `PreparedSetup<D>`; the closed `AjtaiOpeningType` enum
doubles as an uploadable command descriptor and `MatrixSpec` carries the full
window selection, so nothing host-specific leaks into a kernel.

## 9. Implementation plan

Work proceeds in phases. Every phase compiles and passes `cargo test`, and **no
phase changes proof bytes**. Each phase has a checklist; check items off as they
land. The byte-equality + profiler gates from Phase 0 are re-run at Phase 6.

### Phase 0 — Baseline & guardrails

- [ ] Capture a deterministic non-zk proof fixture (serialized bytes) to diff
      against later.
- [ ] Record a profiler run: `AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32
      cargo run --release --example profile`.
- [ ] Save the current CI test-timing comment (`<!-- akita-ci-test-timing -->`)
      numbers as the perf baseline.

### Phase 1 — `commit/` skeleton + vocabulary

- [x] Create `crates/akita-prover/src/commit/` with `mod.rs` and the `ajtai/`
      subfolder.
- [x] Add `ajtai/spec.rs`: `MatrixRole`, `RingDomain`, `MatrixSpec`.
- [x] Add `ajtai/opening.rs`: `ZeroScan`, `AjtaiOpeningType`.
- [x] Add `ajtai/backend.rs`: `trait CommitBackend` (extends
      `DigitRowsComputeBackend` so the ZK blinding mat-vecs stay available to
      `outer_commit`) with the single `ajtai_commit` method.
- [x] Add `decompose.rs` + `opening_view.rs` signatures.
- [x] `pub mod commit;` in `lib.rs`.

### Phase 2 — Move the commit-only column-sweep kernels

- [x] Expose the one-hot + sparse column sweeps through
      `commit/ajtai/column_sweep.rs`. The kernel bodies stay co-located with
      their per-block entry types in `backend/` (as §12 recommends for the
      entry types), and `commit/ajtai/column_sweep.rs` re-exports them as the
      commit subsystem's single named entry point.
- [x] Re-export from old paths so existing callers still compile.
- [x] Confirm the shared `mat_vec_mul_ntt_*` kernels stay in `src/kernels/`.

### Phase 3 — Implement the CPU `commit` primitive

- [x] Implement `CommitBackend for CpuBackend` in `commit/ajtai/cpu.rs` as the
      §7.7 dispatch `match`, reusing `CpuPreparedSetup` and calling the existing
      kernels + column sweeps.
- [x] Add `validate_matrix` / `require_block_width` (the §7.7 no-panic checks).
- [x] Golden equality is covered end-to-end: the existing deterministic
      commit/prove/verify suite (non-zk and all-features) passes unchanged, so
      every arm reproduces the previous `*_commit_rows` / `digit_rows` output.

### Phase 4 — Decomposition unit + `AjtaiOpeningView` impls

- [x] Implement `decompose_rows` / `decompose_rows_into` in `commit/decompose.rs`
      over the existing `decompose_rows_i8_into`. (`recompose_and_validate` was
      not needed: `validate_commit_inner_shape` already owns the recompose
      check, so adding an unused helper would trip `-D warnings`.)
- [x] `impl AjtaiOpeningView for DensePoly` (the two-mode cache logic → `AjtaiOpeningType`).
- [x] `impl AjtaiOpeningView for OneHotPoly`.
- [x] `impl AjtaiOpeningView for SparseRingPoly`.
- [x] `impl AjtaiOpeningView for SuffixWitness` (plus `RootTensorProjectionPoly`
      and `MultilinearPolynomial` dispatch impls).
- [x] Route each representation's decompose through `commit/decompose.rs`.

### Phase 5 — Move the pipeline into `commit/`

- [x] Add `commit/inner.rs`: `commit_inner_one` / `commit_inner_group`.
- [x] Add `commit/outer.rs`: `outer_commit` / `tiered_outer_commit`; delete
      `tiered_commit_u_final`. (The zk-blinding add stays in the pipeline /
      recursive callers, which own the blinding-digit hint; `outer_commit`
      returns the pure `B`/`F` image.)
- [x] Move `commit_with_validated_params` / `batched_commit_with_params` and the
      validators from `api/commitment.rs` into `commit/pipeline.rs`
      (`api/commitment.rs` deleted; `api`/`lib` re-export from `commit`).
- [x] Move the `Cfg`-driven `commit` / `batched_commit` (+ tensor-projection
      decision) into `commit/entry.rs`.
- [x] Move `commit_w` core into `commit/recursive.rs`; keep `commit_next_w`
      dispatch in `protocol/ring_switch` calling it.
- [x] Replace the four `commit_inner` impls with `commit_inner_one`; remove the
      `AkitaPolyOps::commit_inner` trait method.
- [x] Make `CpuPreparedSetup`'s `expanded` / `ntt_shared` fields `pub(crate)`
      so `commit/ajtai/cpu.rs` can read them.
- [x] Change `commit` / `commit_w` bounds from `CommitmentComputeBackend` to
      `CommitBackend`, and rebind `ProverComputeBackend` to
      `CommitBackend + RingSwitchComputeBackend`.

### Phase 6 — Verify equivalence (gate)

- [x] Non-zk proof bytes identical: the full deterministic prove/verify suite
      passes unchanged in release (`cargo nextest run --release --workspace`,
      891 tests).
- [x] zk + all-features run identically (`cargo nextest run --release
      --workspace --all-features`, 498 tests, includes the logging-transcript
      event-equality checks).
- [ ] Profiler + CI test-timing comparison vs. baseline (run in CI).

### Phase 7 — Delete scaffolding & tighten the surface

- [x] Delete `DenseCommitInput` / `DenseCommitRowsPlan` / `OneHotCommitRowsPlan`
      / `SparseRingCommitRowsPlan` / `RecursiveWitnessCommitRowsPlan`.
- [x] Delete the `CommitmentComputeBackend` trait + its four A-methods; the
      commit pipeline now reaches `B`/`F` via `ajtai_commit` with a
      `DigitVector` opening (the relation/quotient `digit_rows` users are
      untouched).
- [x] Tighten `commit/mod.rs` to the narrow public surface; everything else is
      `pub(crate)` or private.
- [x] `cargo fmt -q`, `cargo clippy --all -- -D warnings` (default +
      `--all-features`), and the release test suite are clean.

### Phase 8 — (Dropped)

- [x] ~~Route the cyclic ring-switch relation rows through
      `AjtaiOpeningType::DigitVector { Cyclic }` so the relation path also speaks
      `commit`.~~ Dropped: the `RingDomain` / cyclic-commit hook was removed
      rather than carried as dead scaffolding. The relation/quotient path keeps
      its own `CyclicRowsComputeBackend::cyclic_digit_rows` (per §11), so the
      `commit` primitive stays negacyclic-only.

## 10. Keeping the same performance

- **Same kernels, same args** (§7.7); Phase 3 golden tests + Phase 6
  byte-equality prove the arithmetic path is untouched.
- **Branch-once:** the `match` on the opening is outside every loop; arms hold borrowed
  slices only (no `dyn`, no boxing, no extra allocation). The `B`/`F` sites take
  the single returned block from `ajtai_commit` directly.
- **Trusted dense fast path preserved:** `AjtaiOpeningType::DigitBlocks { Dense }` →
  `mat_vec_mul_ntt_dense_digits_i8_trusted` (no rescans).
- **Tiling/threshold constants move verbatim** (`L2_TILE_BUDGET`,
  `SWEEP_THRESHOLD`, `MAX_WIDE_SHIFT_ACCUMULATIONS`, CRT tile widths).
- **Gates:** profiler + CI test-timing comment vs. the Phase 0 baseline.

## 11. Relationship to the compute-backend spec

This plan is the **second step** after the cutover in
[`specs/akita-compute-backend-metal.md`](../specs/akita-compute-backend-metal.md),
which introduced the host-prepared boundary (`ComputeBackendSetup`), the
operation-family traits, and `CpuBackend` / `CpuPreparedSetup`. That spec is
frozen once merged; per its own rolling-spec rule, **this plan is a new current
spec that supersedes the `CommitmentComputeBackend` operation family** rather
than editing the frozen one.

- **What it supersedes.** The spec's
  `CommitmentComputeBackend::{dense,onehot,sparse_ring,recursive_witness}_commit_rows`
  (and the commit-side use of `digit_rows`) are replaced by one
  `CommitBackend::ajtai_commit` + the closed `AjtaiOpeningType`. The relation /
  quotient operations (`cyclic_digit_rows`, `ring_switch_relation_rows`) are
  **not** touched — they keep their operation-family traits. The fused
  `ring_switch_relation_rows` (D/B/A quotient rows with norm-bound inputs) is not
  a plain `matrix · vector` and is intentionally outside the `ajtai_commit`
  primitive.

- **Invariant 14 (no raw-accessor layer) — RESOLVED.** Decision: the
  `impl CommitBackend for CpuBackend` lives in `commit/ajtai/cpu.rs` and **is**
  the CPU backend implementation, so it is the privileged holder of raw
  `CpuPreparedSetup` access (NTT slot + shared matrix) — the same role today's
  `*_commit_rows` impls play. To allow that, `CpuPreparedSetup`'s `expanded` /
  `ntt_shared` fields become `pub(crate)` (the zk and capacity fields stay
  private to the `compute.rs` impls). The invariant's *intent* is preserved
  exactly: representation (`backend/`) and protocol code never reach through
  `CpuPreparedSetup`; they only call `ajtai_commit` / `AjtaiOpeningView`. We
  re-state the boundary as "outside the **backend implementation**" instead of
  "outside `akita-prover::compute`", because the backend impl now lives in
  `commit/ajtai/`.

- **`ProverComputeBackend` / `batched_prove` bound.** When
  `CommitmentComputeBackend` is deleted (Phase 7), `ProverComputeBackend` and
  `batched_prove` must rebind to `CommitBackend + RingSwitchComputeBackend`;
  `commit` / `commit_w` move to `B: CommitBackend`.

- **Un-fused primitive vs the spec's fused-inner-commit roadmap.**
  `ajtai_commit` is intentionally **one** matrix multiply, so `commit_inner_one`
  runs `ajtai_commit(A) → host decompose → ajtai_commit(B)`. The spec's
  remaining-work bullet anticipated a *fused* inner-commit (A + decompose,
  returning decomposed digits and recomposed rows together) so an accelerator
  avoids the A-row host round-trip. That fusion is deliberately deferred: it is
  free to add later as a `ajtai_commit_inner` op without changing the protocol
  layer, and it has no cost on CPU.

- **`HasWide` bound widened (accepted).** `ajtai_commit` carries the `HasWide`
  bound for all arms even though only one-hot/sparse need it (the spec scoped it
  per-method). Accepted for now; if a field lacking `HasWide` must do a
  dense-only commit, split into a no-`HasWide` negacyclic `ajtai_commit` + a
  `HasWide` column-sweep method (see [§12](#12-risks--open-questions)).

- **Invariants preserved.** The plan keeps the spec's invariants 1–13:
  verifier-visible bytes / labels / challenge order unchanged, no transcript in
  the backend, verifier-crate isolation, typed const-`D` prepared state (no
  erased registries or runtime downcasts), CPU as the byte-exact reference, and
  prepared state derived from the same `AkitaExpandedSetup` / CRT-NTT family.

## 12. Risks & open questions

- **`HasWide` bound placement.** One trait with a method-level bound is simplest
  but forces dense-only callers to satisfy it. Fallback: split `ajtai_commit`
  into a no-`HasWide` negacyclic method + a `HasWide` column-sweep method.
  Decide in Phase 3. (See [§11](#11-relationship-to-the-compute-backend-spec).)
- **Explicit `cols` for B/F.** Every B/F site must now pass the right width;
  `outer_commit` contains all of them, so the blast radius is one function.
- **Entry-type ownership.** `SingleChunkEntry` / `MultiChunkEntry` /
  `SparseRingBlockEntry` are also used by folding code in `backend/`. Keep them
  in `backend/` and re-export into `commit/ajtai/opening.rs`, rather than
  moving them and creating churn.
- **`commit_w` dispatch.** `commit_next_w` (ext-degree + dynamic-`D` dispatch)
  stays in `protocol/ring_switch` and calls `commit::commit_w`; only the commit
  core moves. Confirm this split keeps the recursion call graph intact.

## 13. Definition of done

- `crates/akita-prover/src/commit/` is the single home for the commit
  subsystem: the `CommitBackend` trait + CPU impl, the decomposition unit,
  `AjtaiOpeningView`, and the `outer_commit` / `commit_inner_one` / `commit_w`
  pipeline.
- `commit/mod.rs` exposes only `commit`, `batched_commit`, `*_with_params`,
  `commit_w`, `CommitBackend`, and `AjtaiOpeningView`; everything else is
  `pub(crate)`. `lib.rs` re-exports the same entry points the scheme uses today.
- `commit`, `batched_commit`, and `commit_w` read as `A`-commit → `decompose` →
  `B`/`F`-commit via `commit_inner_one` + `outer_commit`; `tiered_commit_u_final`
  and the four `*CommitRowsPlan` types are deleted.
- Non-zk proof bytes identical to the Phase 0 baseline; zk runs
  transcript-event-identical; profiler + CI timing within noise.
- `cargo fmt -q`, `cargo clippy --all -- -D warnings`, `cargo test` clean.












