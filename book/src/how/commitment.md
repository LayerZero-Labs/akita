# Setup and commitment

How public parameters are built and how a polynomial becomes an Ajtai
commitment, including the two backends (dense and one-hot) that compute the
commitment mat-vec. The commitment is what binds a polynomial to the prover
before the interactive reductions; opening it later proves a point evaluation
([recursion.md](./recursion.md), [verification.md](./verification.md)).

## Setup construction

The shared setup is one field-element vector, interpreted (packed tightly) as
the **A**, **B**, and **D** matrices at every fold level. Setup is generated
**once at capacity time** and reused across instances: it cannot depend on one
runtime schedule, so the setup-time generation ring dimension `gen_ring_dim` is
the **max `ring_dimension` across the config's schedule policy/catalog**
(`crates/akita-setup/src/lib.rs`, `setup_gen_ring_dim`). For the current
uniform-D presets this equals `Cfg::D`, and the verifier binds the same
`gen_ring_dim`, preserving transcript byte-parity.

Ownership is split across crates:

| Concern | Owner |
|---------|-------|
| Setup sizing policy (how large the setup must be) | `akita-config` |
| Setup artifact and matrix expansion | `akita-prover` (`AkitaProverSetup`) |
| Config-backed construction + optional persistence | `akita-setup` |

The prover setup artifact itself is **D-free**; the ring dimension is derived
from `Cfg` at construction (`crates/akita-setup/src/lib.rs`).

### Disk persistence

With the `disk-persistence` feature, a cache file stores the expanded setup
followed by the setup-prefix slots. Cache layout is versioned: **caches written
before setup-prefix persistence will fail to deserialize and must be
regenerated** (`crates/akita-setup/src/lib.rs`). When `Cfg` plans recursive
setup, the loader validates that the cached prefix registry covers every slot
the capacity needs (`lib.rs`, `validate_loaded_prefix_registry`).

The packed-overlapping-prefix layout and its verifier-side reuse are specified
in [`specs/setup-layout-repack.md`](../../../specs/setup-layout-repack.md).

Paper references: §3.8 `Setup`, §3.9 `sec:akita-setup` (packed shared setup).

## Ajtai commitment mechanics

The commitment is the two-tier Ajtai template. Each witness block is decomposed
into commit digits `s_hat`; the **inner commitment** is `t = A * s_hat`. The
outer commitment uses **B** and the opening relation uses **D**. Binding
reduces to Module-SIS: given the public matrices, a prover cannot find two
distinct decompositions that commit to the same value without breaking the
module-lattice hardness assumption ([lattices-sis.md](../foundations/lattices-sis.md),
[pcs-and-binding.md](../foundations/pcs-and-binding.md)).

The public API entry points live in `crates/akita-prover/src/api/commitment.rs`:

- `commit` commits one commitment group
  (`commitment.rs`); `batched_commit` is the batch path
  (`commitment.rs`).
- `prepare_batched_commit_inputs` validates the group: it must be nonempty, its
  padded `num_vars` and count must fit the setup capacity, and its natural
  polynomial arity selects that group's root layout
  (`commitment.rs`).

### SIS sizing

The single home for "given a width and a rounded-up coefficient bound at a
security floor, what is the minimum SIS-secure module rank, and what audited
commit-matrix parameters does it yield" is `crates/akita-types/src/sis/ajtai_key.rs`.
It consults the generated SIS-floor tables in the sibling
`generated_sis_table/` module. The key types are `SisSecurityPolicyId`
(`ajtai_key.rs`) and `SisModulusProfileId` (`ajtai_key.rs`); the
`Quantum128BitADPS16` policy is described in
[security.md](./security.md).

Paper references: §2.6 `sec:prelim-pcs` (two-tier Ajtai), §3.2
`sec:akita-layout` (commitment matrices, inner/outer commitments).

## The one-hot inner-Ajtai kernel

The inner commitment `t = A * s_hat` is the performance-critical mat-vec. The
one-hot backend does **not** materialize the decomposed vector `s_hat` and run a
dense mat-vec; instead it accumulates **only the nonzero contributions** using a
fused shift-accumulate into a `WideCyclotomicRing` (carry-free `i32` additions),
then reduces once at the end (`crates/akita-prover/src/backend/onehot/inner_ajtai.rs`):

```text
t[a] += A[a][entry.commit_col(num_digits)] * (X^{k_1} + X^{k_2} + ...)
```

The wide accumulator avoids a modular reduction per addition compared with a
direct field-ring accumulator. One-hot storage is described in
`crates/akita-prover/src/backend/onehot/mod.rs`.

Paper reference: App B.2.5 (one-hot commitment optimization).

## Polynomial backends: dense vs one-hot

Akita has two polynomial backends, selected per preset, that implement the four
prover operations (ring evaluation, per-block fold, decompose+fold, and the
inner-Ajtai commit):

- **Dense** (`crates/akita-prover/src/backend/dense/`): all ring coefficients
  materialized in memory. `DensePoly` uses balanced-digit decomposition,
  NTT-based matrix-vector multiply, and parallel block folds
  (`backend/dense/mod.rs`). This is the CRT+NTT digit mat-vec
  (`sec:akita-crt-matvec`).
- **One-hot** (`crates/akita-prover/src/backend/onehot/`): a sparse witness with
  at most one nonzero field element per chunk of size `onehot_k`. `OneHotPoly`
  iterates only the nonzero monomial positions, trading per-entry bookkeeping
  for dense-width work (`backend/onehot/mod.rs`).

For a given field, the one-hot backend is the usual production choice at
**fp128 D64**; **D128** remains a comparison / legacy profile. The preset table
and when to choose one-hot vs dense by field family are in
[quickstart.md](../usage/quickstart.md).

The packed `RingSubfieldFpExt8` multiplication kernels were an early SIMD
optimization for fp16 small-field presets. The fp16 field family and the
`RingSubfieldFpExt8` type have since been **fully removed** from the codebase
(no `Fp16` / `RingSubfieldFpExt8` paths remain in `crates/`); the historical
technique note is kept in
[`specs/simd-ring-subfield-fp8.md`](../../../specs/simd-ring-subfield-fp8.md)
and the broader packed-SIMD field-arithmetic story is listed for folding in
[optimizations.md](./optimizations.md) (chapter pending).

Paper reference: §3.2 `sec:akita-layout`, App B.2.5 (`sec:akita-crt-matvec`).
