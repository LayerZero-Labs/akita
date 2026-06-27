# Spec: Runtime Ring Cutover

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-06-24 |
| Revised       | 2026-06-27 (stack status, PR map, bounded generated mixed-D admission, D-free PCS API surface) |
| Status        | in progress (runtime-D cutover plus bounded generated mixed-D admission; DP search/catalog regen deferred) |
| PR            | stacked on `#227`–`#241` (see Stack status) |
| Supersedes    | partial supersession of `specs/akita-polyops-cutover.md` (storage half); coordinate PR order with `specs/protocol-field-geometry-cutover.md` (shared `PreparedFold` / `prove_suffix` surface) |
| Superseded-by | |
| Book-chapter  | book/src/how/architecture.md |

## Stack status (2026-06-27)

This PR is the full runtime ring-dimension cutover target for the current
bounded model: one active `ring_d` per fold, with later folds allowed to use a
smaller divisor of the setup envelope dimension. The hand-built mixed-D fixture
(`mixed_d_per_level_e2e`, `mixed_d_geometry_crosscheck`) remains the E2E
regression gate; generated schedule expansion now also admits compact rows whose
`ring_d` is a nonzero divisor of the policy envelope `D`.

Generated mixed-D **search** and production catalog regeneration do not exist
yet. That means generated tables can now represent mixed-D rows, but the offline
optimizer still does not choose them. Arbitrary unrelated per-fold Ds are out of
scope for this cutover.

### Immediate target (current stack)

- One active `ring_d` per fold; different folds may use different `D`s.
- Setup envelope may be larger than active fold `D` when active `D` divides the
  envelope dimension (fixture uses `D128Full` envelope, `D64` suffix views).
- Proof verifies from schedule-bound runtime `D`.
- `AkitaCommitmentScheme<Cfg>` and the normal PCS claim/verify surface are D-free;
  typed dispatch traits are explicitly named `TypedCommitmentProver<F, D>` /
  `TypedCommitmentVerifier<F, D>` and remain internal bridge points to const-generic kernels.

### Later target (explicitly deferred)

- DP search over `ring_d`, production generated mixed-D catalog regeneration,
  and generalized `D_max` envelope sizing beyond the current divisor-envelope model.
- Distinct per-block `d_a` / `d_b` / `d_d` inside one fold.
- Prefix-sized NTT cache optimization; proof/perf benchmark CI gates.

### PR map

| PR | Branch | Base | Waves / slice | Purpose | Status | Deferrals |
|----|--------|------|---------------|---------|--------|-----------|
| [#227](https://github.com/LayerZero-Labs/akita/pull/227) | `quang/runtime-ring-cutover` | `main` | 1–3 infra | `RingDimPlan`, shared setup geometry, NTT cache map, D-free prepared setup | Rebasing clean onto `main`; CI pending | Phase 4 planner; envelope `D_max` |
| [#228](https://github.com/LayerZero-Labs/akita/pull/228) | `quang/setup-footprint-stage3-unification` | `#227` | 3b | Stage-3 footprint unification with shared geometry | Restacked | — |
| [#230](https://github.com/LayerZero-Labs/akita/pull/230) | `quang/runtime-ring-prove-orchestration` | `#228` | 4–5 entry | Prove-time ring plan validation, NTT setup contract, suffix prefix fix | Restacked | — |
| [#231](https://github.com/LayerZero-Labs/akita/pull/231) | `quang/runtime-ring-stack-demote` | `#230` | 5 | Demote `ProverComputeStack` / `LevelProveStacks` off `const D` | Restacked | — |
| [#232](https://github.com/LayerZero-Labs/akita/pull/232) | `quang/runtime-ring-mixed-d-fixture` | (merged) | 0 | Wave 0 mixed-D fixture harness + oracle | Merged | — |
| [#233](https://github.com/LayerZero-Labs/akita/pull/233) | `quang/runtime-ring-backend-ring-d` | `#232` | 5b | Shared NTT cache via `Mutex` / `with_ntt_slot` | Restacked | — |
| [#234](https://github.com/LayerZero-Labs/akita/pull/234) | `quang/runtime-ring-uniform-suffix` | `#233` | 5c | Uniform suffix loop, prove-entry NTT warming | Restacked | — |
| [#235](https://github.com/LayerZero-Labs/akita/pull/235) | `quang/runtime-ring-delete-trait-lattice` | `#234` | 5d | Delete suffix trait lattice | Restacked | — |
| [#236](https://github.com/LayerZero-Labs/akita/pull/236) | `quang/runtime-ring-public-api` | `#235` | 5e | D-free `AkitaProverSetup`, `RingBuf` fold storage | Restacked | — |
| [#239](https://github.com/LayerZero-Labs/akita/pull/239) | `quang/runtime-ring-wave6-bulk` | `#236` | 6 | Fold storage demotion (bulk) | Restacked | — |
| [#240](https://github.com/LayerZero-Labs/akita/pull/240) | `quang/runtime-ring-wave7` | `#239` | 7 | Hint demotion | Restacked | — |
| [#241](https://github.com/LayerZero-Labs/akita/pull/241) | `quang/runtime-ring-wave7-mixed-d` | `#240` | 7 gate | PCS scheme demotion, mixed-D geometry gate, oracle refresh | Restacked; slow geometry test fixed | `CommitmentProver<F,D>` traits remain |
| [#243](https://github.com/LayerZero-Labs/akita/pull/243) | `quang/runtime-ring-dfree-pcs-api` | restacked `#241` tip | final cutover | D-free PCS API surface and claim carriers; typed dispatch traits renamed; bounded generated mixed-D admission; root-direct metadata errors fail loudly | In progress | DP search/catalog regen and generalized Dmax envelope sizing |

### Completed in code (current stack tip)

- `RingDimPlan`, `RingLevelContext`, shared `setup_geometry` / `setup_active_ring_elems_at`
- D-free prepared setup, NTT cache map, setup-prefix registry demotion
- Prove-time ring plan validation; uniform suffix loop (no stack-rebuild branch)
- Fold/hint storage demotion to `RingBuf` / flat storage
- Hand-built D128→D64 mixed-D fixture proves and verifies
- `AkitaCommitmentScheme<Cfg>` struct demotion plus D-free setup/prove/verify,
  flat commitment carriers, D-free public verifier trait, and explicitly typed
  internal prover/verifier dispatch traits.
- Generated compact schedule expansion accepts a level-local `ring_d` when it is
  nonzero and divides the policy envelope `D`; invalid root-direct commit metadata
  now fails schedule resolution instead of being silently dropped.

### Incomplete or deferred

- Generated planner mixed-D search and catalog regeneration; compact expansion
  admission exists, but the optimizer still emits uniform-D schedules today.
- `D_max` field-element envelope sizing; relaxing `gen_ring_dim == Cfg::D` at setup
- Generated schedule catalogs with multiple `D`s per family
- Root polynomial/backend capability traits remain const-generic at typed dispatch
  sites (`RootCommitPoly<F, D>`, `RootProvePoly<F, D>`, NTT/kernel paths).
- Per-role / per-block `d_a`, `d_b`, `d_d` execution inside one fold
- Prefix-sized NTT caches; full verifier no-panic audit on D-erased proof storage APIs
- Proof-byte golden pins and bench regression gates beyond smoke/profile

## Summary

Ring dimension `D` is baked into prover orchestration as a compile-time type
parameter (`ProverComputeStack<F, D>`, `CpuPreparedSetup<F, D>`, `prove_fold<..., D>`,
`RecursiveProveBackend` with a six-bound `ProveFlowBackendFor` supertrait lattice
per backend), while the schedule already stores a per-level
`LevelParams.ring_dimension`. When runtime `D` disagrees with preset `Cfg::D`,
suffix code re-prepares all four backend clusters and rebuilds stacks via
`dispatch_ring_dim_result!`.

This spec **demotes `D` from a storage and orchestration type parameter to a
runtime schedule value**. Bulk data lives in flat field buffers; hot kernels still
monomorphize at `const D` behind a single backend dispatch boundary. A
**`RingDimPlan`** is a derived view of validated per-level `ring_dimension` values
from the schedule; per-level **`RingLevelContext`** (ring dimension plus setup prefix
geometry) is computed at runtime via `context_at` from the live
`SetupRelationShape`, not stored inside the plan. **`PreparedSetup`** registers the
**minimum** NTT setup contract at **`prepare_setup`** (envelope `D` today; keyed
`(ring_d, num_ring_elements)`). Caches at different `ring_d` are physically distinct
transforms and are never shared (see NTT cache today); a uniform-`D` proof keeps exactly
one cache, identical to today.

A follow-on planner change (not fully scoped here) can emit **one optimal
mixed-D schedule per field family** instead of maintaining separate preset and
schedule-table families for each constant ring dimension (`fp128_d64` vs
`fp128_d128`, etc.).

## Background

### What works today

Several pieces already match the target shape:

| Piece | Location | Role |
|-------|----------|------|
| `FlatMatrix<F>` | `akita-types/src/layout/flat_matrix.rs` | D-free setup storage; `gen_ring_dim` = generation envelope |
| `RingMatrixView<'a,F,D>` | same | Zero-copy matrix view at runtime `D` |
| `FlatRingVec<F>` | `akita-types/src/proof/containers.rs` | D-erased proof wire (`ring_dim = 0` compact mode) |
| `RecursiveWitnessFlat` | `akita-prover/src/backend/recursive/witness.rs` | Flat `Vec<i8>` owner; `SuffixWitnessView` at `const D` |
| `LevelParams.ring_dimension` | `akita-types/src/layout/params.rs` | Per-level runtime ring degree in schedule |
| `GeneratedFoldStep.ring_d: u32` | `akita-planner/src/generated/mod.rs` | Per-step ring dim in compact table rows |
| `dispatch_ring_dim_result!` | `akita-types/src/dispatch.rs` | Runtime `usize` → `const D` bridge over `{32, 64, 128, 256}` |
| `AkitaSetupSeed` / `SetupMatrixEnvelope` | `akita-types/src/proof/setup.rs` | Seed carries `gen_ring_dim` + `max_setup_len`; envelope carries `max_setup_len` (+ zk lens) |
| `select_setup_prefix_slot` / `setup_prefix_level_params` | `akita-types/src/proof/setup_prefix.rs` | **Already D-free** (take `d_setup: usize`); already shared by prover and verifier |
| `SetupPrefixSlotId` (carries `d_setup`) | same | D lives in the slot id today |
| `SetupPrefixVerifierRegistry<F>` | `akita-types/src/proof/setup.rs` | **Already D-free** verifier registry |

`CyclotomicRing<F, D>` is `#[repr(transparent)]` over `[F; D]`. `Vec<CyclotomicRing<F, D>>`
and `Vec<F>` of length `N·D` are layout-identical; the refactor is about **where**
`const D` appears in types, not about changing ring arithmetic.

### Setup sizing today (normative grounding)

This subsection is normative reference for every "envelope" / "prefix" claim below.

**One buffer, prefix views.** The expanded setup is a single flat field buffer
`FlatMatrix<F> { data: Vec<F>, gen_ring_dim }`. The A/B/D/F role matrices are
**prefix/column views into this one buffer**, not separate allocations. Capacity is
therefore the **maximum single role footprint** across levels, not the sum
(`accumulate_matrix_envelope_for_level` takes `max(a_len, b_len, d_len, f_len)`).

**Units and the splitting identity.** The buffer holds `max_setup_len` ring elements
at `gen_ring_dim`, i.e. `max_setup_len * gen_ring_dim` field elements
(`AkitaSetupSeed::matrix_field_elements`). The load-bearing identity is:

```
FlatMatrix::total_ring_elements_at::<D>() = total_ring_elements * (gen_ring_dim / D)
    // requires gen_ring_dim % D == 0
```

The **field-element count is invariant**; viewing at smaller `D` splits each
generation-degree ring into `gen_ring_dim / D` smaller rings. This identity is the
entire mechanical basis for "commit the root at D=128, view the same bytes at D=64
for later folds."

**Sizing is per-config and schedule-derived.**
`proof_optimized_max_setup_matrix_size::<Cfg>(max_num_vars, max_num_batched_polys)`
is the sizing authority. It is **already per-config** — memoized on
`(TypeId::<Cfg>, max_num_vars, max_num_batched_polys)`. It does **not** budget for
other configs. It loops over every workload shape *this* config might prove
(`num_vars in 1..=max_num_vars` × a small poly-count set), calls
`Cfg::runtime_schedule(shape)` to get *this config's own* schedule per shape,
computes `matrix_envelope_for_schedule` of each, and takes the max
(`max_setup_len = max over shapes of envelope.max_setup_len`).

Consequences (these answer the "do we have to budget for everything?" question):

1. **No cross-config budgeting.** One preset → one envelope from that preset's
   schedules. A more precise per-preset policy is already what ships.
2. **Within-config workload budgeting is unavoidable.** One setup serves every
   `(num_vars, num_polys)` up to the declared maximum, so the envelope is a max over
   those shapes — that is "the largest witness this config supports," not "other
   configs."
3. **Hard dependence on the generated schedule.** The envelope is literally
   `max over shapes of matrix_envelope_for_schedule(Cfg::runtime_schedule(shape))`.
   Wrong/missing schedules → wrong envelope. Today this is safe because
   `gen_ring_dim == Cfg::D` is **enforced** at setup build and deserialize
   (`api/setup.rs`, `akita-setup/src/lib.rs`), so every footprint is in one unit
   (ring elements at one `D`).

**What mixed-D will require (Phase 4 contract, not Phase 1–3 work).** Once levels can
differ in `ring_d`, "ring elements" is no longer a comparable unit across levels.
Envelope accumulation must move to **field elements**:

```
footprint_field(level) = role_footprint_ring_elems_at(level, levelD) * levelD
max_field_len          = max over levels/roles/shapes of footprint_field
gen_ring_dim           = D_max = max ring_d used by any emitted step
max_setup_len          = max_field_len / D_max          // requires gen_ring_dim % levelD == 0 ∀ level
```

The Phase-3 mixed-D fixture needs **no** envelope change: it reuses a larger-D
preset's envelope (`D128Full` → `gen_ring_dim = 128`) and views at `D=64`
(`128 % 64 == 0`). Phases 1–3 keep single-D generation; they must only avoid
*assuming* `gen_ring_dim == Cfg::D` at fold time so that viewing the envelope at a
smaller `D` is legal.

### NTT cache today (normative grounding)

`NttSlotCache<const D: usize>` (`akita-prover/src/kernels/crt_ntt.rs`) is the
dominant prepared-setup allocation. Structure:

- Enum over prime family `Q32 | Q64 | Q128`, selected by **field modulus** (not by
  `D`) via `select_crt_ntt_params::<F, D>` — `K = 2 | 3 | 5` CRT primes.
- Each variant stores, **per ring element of the viewed matrix**, two CRT+NTT
  transforms: `neg: Vec<CyclotomicCrtNtt<i32, K, D>>` (negacyclic, for mat-vec) and
  `cyc: Vec<CyclotomicCrtNtt<i32, K, D>>` (cyclic, for quotients), plus
  `params: CrtNttParamSet<i32, K, D>` (twiddle/root tables).
- Built by `build_ntt_slot(ring_view::<D>(rows, cols))`, which maps each ring element
  through `CyclotomicCrtNtt::from_ring_pair_with_params`.

Cache length = `num_ring_elements` (at `D`). Each element is `K * D` i32 values, in
two copies (neg + cyc), so `cache_bytes ≈ num_ring_elements * K * D * 4 * 2` — for
fp128 (`K = 5`) roughly 5× the underlying field data (hence "much larger than the
plain setup vector").

**Why caches at different `D` cannot be shared.** Let the small/large dims be `D` and
`2D` (concretely 64 and 128). A dim-`2D` element `f = (a₀,…,a_{2D−1})` is *viewed* at
dim-`D` as two independent elements — the raw halves `p_lo = (a₀,…,a_{D−1})` and
`p_hi = (a_D,…,a_{2D−1})`, with `f = p_lo + Xᴰ·p_hi`. The dim-`D` cache must hold
`NTT±_D(p_lo)` and `NTT±_D(p_hi)` **separately**; the dim-`2D` cache holds
`NTT±_{2D}(f)`. Three facts, in increasing subtlety:

1. **Root nesting holds only for the cyclic transform.** The cyclic `N`-NTT (the `cyc`
   / quotient rep, ring `Xᴺ−1`) evaluates at the `N`-th roots of unity. Since
   `x⁶⁴ = 1 ⟹ x¹²⁸ = (x⁶⁴)² = 1`, the `D`-th roots are exactly the even powers of a
   primitive `2D`-th root: `{D-th roots} ⊂ {2D-th roots}`. The negacyclic `N`-NTT (the
   `neg` / matvec rep, ring `Xᴺ+1`) evaluates at the roots of `Xᴺ+1`: dim-`D` roots
   solve `xᴰ = −1` (order exactly `2D`), dim-`2D` roots solve `x^{2D} = −1` (order
   exactly `4D`) — **disjoint**. So the "roots of −1, same thing applies" intuition
   holds for `+1` (cyclic) but **fails for −1** (negacyclic): `α⁶⁴ = −1 ⟹ α¹²⁸ = +1 ≠ −1`.

2. **Even where the roots nest, the operands don't.** The exact radix-2
   (decimation-in-frequency) identity for the cyclic transform is
   ```
   NTT⁺_{2D}(f)[even] = NTT⁺_D(p_lo + p_hi)
   NTT⁺_{2D}(f)[odd]  = NTT⁻_D(p_lo − p_hi)
   ```
   Derivation: at an even point `ω²ᵏ` (so `ω²` is a primitive `D`-th root), `ω^{2kD}=1`
   yields `Σ(a_i + a_{D+i})(ω²)^{ki}`, the cyclic transform of the *sum*; at an odd
   point `ω^{2k+1}`, `ωᴰ=−1` yields `Σ(a_i − a_{D+i})ω^{(2k+1)i}`, the *negacyclic*
   transform of the *difference* on the `Xᴰ+1` roots. So the even sublattice of the
   `2D`-cyclic cache **is** a genuine `D`-cyclic transform — but of `p_lo+p_hi`, giving
   only one linear equation `NTT⁺_D(p_lo) + NTT⁺_D(p_hi)`. The other block lives in the
   negacyclic domain (different evaluation points) and cannot be combined to separate
   the halves. The transform of an interleaving is not the interleaving of the
   transforms.

3. **The salvageable part isn't worth it.** Deinterleaving **both** the `2D` cyc and
   neg caches yields four sum/difference relations across the two domains — exactly one
   Cooley–Tukey butterfly layer, which is invertible: one *could* recover the four
   `D`-caches with `O(D)` twiddle butterflies per element instead of an `O(D log D)`
   retransform. But (a) it only helps on the region the two caches *share*, which is
   normally empty — the `2D` cache is the full root envelope while a `D` cache is a
   small later prefix; (b) it saves build *compute*, not *memory* — both layouts must
   still be stored, and storage (~5× the field data for fp128) dominates; (c) it is
   domain-crossing and twiddle-heavy. So `build_ntt_slot` rebuilds from coefficients
   per `(D, view)`, and the cache keys on `ring_d` with no cross-`D` reuse.

**Size corollary.** The field-element count is invariant, so a *full-envelope* cache
holds the same total transformed-i32 count at any `D`
(`num_ring_elements_at_D · D · K = total_coeffs · K`). Smaller `D` does **not** shrink
the full-envelope cache; it only regroups the same coefficients. (Real memory savings
would come from caching a sub-envelope **prefix** sized to a proof's actual commit
footprint rather than the `max_num_vars` envelope — a deferred optimization, orthogonal
to `ring_d`; see NTT cache design.)

### What hurts today

**1. Dual authority for ring dimension**

| Source | Meaning |
|--------|---------|
| `CommitmentConfig::D` | Compile-time preset (e.g. `fp128::D64Full` → `D = 64`) |
| `LevelParams.ring_dimension` | Per-fold runtime value from schedule |
| `PlannerPolicy.ring_dimension` | Single D fixed for entire DP search |

Expansion rejects `ring_d != policy.ring_dimension`
(`akita-planner/src/generated/expand.rs`: `if ring_d == 0 || ring_d != policy.ring_dimension`).
Shipped tables never mix `ring_d` across steps. Suffix dispatch and the wide backend
trait bounds exist for a capability the planner cannot emit.

**2. Suffix orchestration tax**

When `level_d != Cfg::D`, `prove_suffix` takes the `else` arm of
`if level_d == D { … } else { dispatch_ring_dim_result!(level_d, |D_LEVEL| …) }`,
re-calls `prepare_expanded::<D_LEVEL>` on **all four** clusters (commit, opening,
tensor, ring), constructs a fresh `ProverComputeStack::<_, D_LEVEL>`, **drops the
setup-prefix registry** (`SetupPrefixProverRegistry::new()` empty workaround), and runs
`prove_fold` at `D_LEVEL` (`akita-prover/src/protocol/core/suffix.rs`). The verifier
suffix **always** dispatches via `dispatch_ring_dim_result!`, even when every level
uses the same `D` (`akita-verifier/src/protocol/core/suffix.rs`).

**3. Trait lattice tax**

`RecursiveProveBackend<F, P, E, D>` carries a **six-bound** supertrait lattice
(`akita-prover/src/compute/poly.rs`):

```
ProveFlowBackendFor<F, P, E, D>
+ ProveFlowBackendFor<F, RecursiveWitnessFlat, E, D>
+ ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 32>
+ ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 64>
+ ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 128>
+ ProveFlowBackendFor<F, RecursiveWitnessFlat, E, 256>
```

`SuffixRingSwitchProveBackend`, `SuffixWitnessOpeningProveBackendFor`,
`SuffixDispatchOpeningProveBackendFor`, `SuffixDispatchTensorProveBackendFor` (and the
root-tensor siblings) duplicate the `{32, 64, 128, 256}` fan-out.
`RECURSIVE_SUFFIX_RING_DIMENSIONS == &[32, 64, 128, 256]`.

**4. Prepared setup over-builds NTT**

`CpuBackend::prepare_expanded::<D>` converts the **entire** shared matrix at `D` into
`NttSlotCache<D>` (`compute/cpu.rs`):

```201:202:crates/akita-prover/src/compute/cpu.rs
        let total = expanded.shared_matrix.total_ring_elements_at::<D>()?;
        let ntt_shared = build_ntt_slot(expanded.shared_matrix.ring_view::<D>(1, total)?)?;
```

while setup sumcheck and recursive commit often need only a **prefix** at that
dimension (`setup_sumcheck.rs` already selects `setup_eval_len ≤ setup_len` when
offload is active). `CpuPreparedSetup<F, const D>` is `const D`-parameterized.

**5. Preset proliferation per ring dimension**

`fp128` ships separate `Cfg` types and schedule-table families per constant D
(`D32Full`, `D64Full`, `D128Full`, `D32OneHot`, `D64OneHot`, `D128OneHot`,
`D64OneHotTiered`, … in `crates/akita-config/src/proof_optimized/fp128.rs`). Runtime
ring cutover is a prerequisite for collapsing these into one field-family config with
schedule-driven `D`.

**6. Setup geometry is computed twice from parallel code paths**

The prover (`prepare_setup_sumcheck_terms` → `create_setup_contribution_inputs` →
`SetupContributionPlan::prepare`) and the verifier (`stage3.rs`) independently derive
the per-level setup prefix length. Nothing in the transcript binds the chosen
`setup_eval_len`, so a divergence between the two paths is a **silent soundness gap**,
not a caught error. Consolidating this into one shared function is part of this
cutover (see Normative contracts) and is soundness-load-bearing, not just cleanup.

## Intent

### Goal

Make ring dimension a **schedule-driven runtime parameter** end to end:

1. Different `(d_a, d_b, d_d)` per fold is first-class in prove, verify, and prepared
   state. `d_a` is the fold / ring-switch / inner-commitment ring (the legacy per-level
   `ring_d`); `d_b, d_d` are the outer- (`B`) and opening- (`D`) commitment rings.
   Uniform presets set `d_a == d_b == d_d` (byte-identical to today).
2. Suffix orchestration does not special-case cross-D folds (no stack rebuild).
3. Fold protocol storage (`PreparedFold`, `RingRelationInstance`) does not carry
   `const D` on the struct (Phase 3); in-memory owners use `RingBuf` / `RingSlice`.
4. NTT prepared caches are one full-envelope cache per **distinct** `ring_d` (keyed
   `(ring_d, num_ring_elements)`), with the **minimum envelope slot registered at
   `prepare_setup`**, never shared across `ring_d`. Additional slots may be built
   lazily at commit/prove if undersized (diagnostic warning; see Warm-cache policy).
   Prefix-sizing *within* a `ring_d` is a deferred optimization (see NTT cache design).
5. Infrastructure supports a future planner that optimizes the `(d_a, d_b, d_d)` triple
   per fold step within one field family. Until then, hand schedules pick the triple;
   the infrastructure must not assume `d_a == d_b == d_d`.

### Invariants

**Protocol correctness**

- Fold math, ring switch, stage 1/2/3 unchanged unless listed under Wire Changes.
- Verifier no-panic contract preserved (`docs/verifier-contract.md`).
- `RingDimPlan::dim_at(ℓ) == schedule[ℓ].params.ring_dimension` for every fold level;
  `dim_at(ℓ)` is the fold ring `d_a` (= `dims_at(ℓ).inner`).
- Per-role dims satisfy `d_d | d_b | d_a` at every level (validated in `from_schedule`).
- Flat buffer chunking of the committed witness at level `ℓ` uses `d_a`; the outer /
  opening commitment matvecs chunk at `d_b` / `d_d`. Malformed lengths return
  `InvalidProof` / `InvalidSetup`, never panic.

**Performance**

- Inner NTT / matvec / ring kernels remain `const D` monomorphizations (AVX/NEON
  unchanged).
- Zero-copy views at kernel boundaries (`RingSlice`, `RingMatrixView`); no
  `from_slice` / `to_vec::<D>()` in hot paths.
- No per-fold `prepare_expanded` or `ProverComputeStack` reconstruction in suffix.
- At most one 4-way `match` per backend call (not per fold orchestration step).

**Setup / NTT cache**

- One physical `FlatMatrix` per expanded setup (`gen_ring_dim` = capacity envelope).
- One NTT cache per **distinct** `ring_d`, keyed `(ring_d, num_ring_elements)` with
  `num_ring_elements = total_ring_elements_at::<ring_d>()` (the full envelope at that
  `ring_d`). Uniform-`D` ⇒ exactly one entry (today's behavior).
- Keys at different `ring_d` name physically distinct, non-overlapping transforms (see
  NTT cache today); never dedup or share across `ring_d`.
- The cache-hit accessor `NttSlotCacheAny::as_d::<D>()` is **fallible**: a stored
  variant whose `ring_d` ≠ the dispatched `D` returns `InvalidSetup`, never panics.
- The NTT cache is **independent of setup-prefix offload**: kernels take the full slot
  plus `(row_len, row_width)` and index a prefix; offload changes only the setup
  sumcheck's *direct* `ring_view`, never a cache key (see Setup-prefix offload).
- `setup_active_ring_elems_at(ℓ)` (the offload-decision count) is a pure function of
  `(schedule, level, setup envelope, relation shape)` — challenge-independent (see
  Normative contracts) and **identical on prover and verifier**. It **fails closed**
  (`InvalidSetup`) when `required > envelope total at ring_d`; it never silently caps.

**Descriptor / transcript**

- Per-fold `ring_dimension` is bound by the existing schedule digest in `PlanSection`
  (`LevelParams::append_descriptor_bytes` already pushes `ring_dimension`).
- `AlgebraSection` records the setup **envelope** degree (`gen_ring_dim`), not the
  per-fold suffix ladder. For all current presets `gen_ring_dim == Cfg::D`, so this is
  a **byte-level no-op today** (see Descriptor binding).
- Setup-prefix slot selection is bound by transcript absorption of `SetupPrefixSlotId`
  (`ABSORB_SETUP_PREFIX_SLOT`) when offloading is active at a level.

**Catalog / identity**

- `GeneratedScheduleCatalogIdentity.ring_dimensions` is already `&'static [usize]` and
  may list multiple distinct values once mixed-D tables ship; identity digest already
  supports this.

### Non-Goals

- Runtime-D-generic NTT butterflies (no dynamic `D` inside SIMD loops).
- Changing the `gen_ring_dim` / `max_setup_len` envelope **sizing policy** in Phases
  1–3 (the field-element accumulation described in "Setup sizing today" is Phase 4).
- **Planner search** over per-fold dims, including the `(d_a, d_b, d_d)` triple, in the
  first PR (infrastructure + hand-picked triples only; see Future: unified field-family
  planner). Per-block ring *geometry* is in scope; per-block *search* is not.
- **Per-block kernel execution.** The `B`/`D` commitment matvecs and the ring-switch
  weights honoring distinct `d_b, d_d`, and the non-uniform per-block fixture, are a
  tracked follow-up (PR-perblock-exec; see PR decomposition). This work lands the
  per-block *geometry* (the `CommitmentRingDims` triple, `d_d | d_b | d_a` validation,
  transcript binding) with an **interim construction guard `d_a == d_b == d_d`**, so the
  executed behavior is exactly per-level mixed-D — kernels still see one dim per level.
- Replacing `CyclotomicRing<F, D>` as stack/value type inside kernels.
- GPU / Metal backend design (`specs/akita-compute-backend-metal.md`).
- Merging `fp128_d64` and `fp128_d128` preset families in the same PR as runtime
  infrastructure (follow-on after planner work).

## Evaluation

### Acceptance Criteria

**Phase 1 — infrastructure (CI-hard)** — **landed in stack (#227–#228); criteria below are satisfied at tip unless noted.**

- [x] `setup_geometry_at` (shape-only, challenge-free; see Normative contracts) and
      `setup_active_ring_elems_at` in `akita-types`, with golden vectors.
- [x] `RingDimPlan`, `RingLevelContext` in `akita-types`.
- [x] `RingDimPlan::from_schedule` with validation catalog (see Normative contracts);
      takes `&AkitaSetupSeed` (carries `gen_ring_dim` + identity), not a new envelope
      type.
- [x] `NttCacheKey`, `NttSlotCacheAny` (+ fallible `as_d::<D>()`), `NttCacheMap`
      (`HashMap` keyed store) with lazy `ensure_ntt_slot`.
- [x] `CpuPreparedSetup<F>` (trait assoc type `PreparedSetup`) without `const D`;
      `prepare_expanded` builds an empty map; `prepare_setup` registers the minimum
      envelope contract (see Warm-cache policy). Lazy `ensure_ntt_slot` outside the
      contract logs a sizing diagnostic.
- [x] Single shared setup-geometry function consumed by **both** prover setup
      sumcheck and verifier stage 3 (replaces the two parallel derivations).
- [x] D-free `SetupPrefixRegistry` (replaces `SetupPrefixProverRegistry<F, D>`; the
      verifier registry is already D-free); delete the `if D == SETUP_OFFLOAD_D_SETUP`
      eligibility gates at both call sites (retain `SETUP_OFFLOAD_D_SETUP` as the
      `d_setup = 64` naming constant in `setup_prefix.rs`). See Phase-ordering note for the slot's
      commitment/hint.
- [x] `bind_transcript_instance_descriptor` without `const D`;
      `AlgebraSection::for_envelope` uses `gen_ring_dim`.

**Phase 2 — orchestration cutover (CI-hard)** — **mostly landed (#230–#236, #241 partial); trait-level `D` remains.**

- [x] `ProverComputeStack<F, B>`, `OperationCtx<F, B>` without `const D`.
- [x] Backend traits take `RingLevelContext` (or `NttCacheKey` where NTT-only);
      internal `dispatch_ring_dim!` only.
- [x] `prove_suffix` / `verify_suffix` / `commit_next_w`: uniform loop over
      `plan.context_at(level)`; **no** stack-rebuild branch.
- [x] `prove_fold` / `verify_fold` API takes `ring_d: usize` at boundary.
- [x] Delete `Suffix*ProveBackendFor`, `RECURSIVE_SUFFIX_RING_DIMENSIONS`,
      six-bound `RecursiveProveBackend` supertrait lattice.
- [x] `AkitaCommitmentScheme<Cfg>` without `const D` on the scheme struct (per-preset
      macro impls in `akita-pcs/src/scheme/impls.rs`). Public PCS methods and
      verifier trait use D-free flat commitments; typed prover/verifier dispatch
      traits are explicitly named `Typed...` because rustc cannot use `Cfg::D`
      in const-generic trait bounds on a single blanket impl.
- [x] Grep gate: no `dispatch_ring_dim_result!` in `protocol/core/suffix.rs` or
      verifier `protocol/core/suffix.rs`.

**Phase 2 exit (manual review)**

- Uniform-D shipped presets prove and verify with byte-identical descriptors to
  pre-cutover (no pinned-digest change expected since `gen_ring_dim == Cfg::D`; if any
  digest moves, investigate — it means a preset's envelope diverged from `Cfg::D`).
- Suffix cold path removed; perf neutral or better on profile preset
  `onehot_fp128_d64:32:1` (advisory, not CI gate).

**Phase 3 — fold storage cutover (CI-hard)** — **storage demotion landed (#236, #239–#240); wire-byte pins and bench gates deferred.**

- [x] `PreparedFold`, `RingRelationInstance`, verifier `PreparedFoldReplay` without
      `const D` on struct (use `RingBuf` / `RingSlice` internally).
- [x] `RingBuf<F>` in-memory alias over compact storage; `as_ring_slice::<D>()` API
      (same semantics as today's `FlatRingVec`); wire `FlatRingVec` encoding unchanged.
- [ ] No `to_vec::<D>()` / `from_vec::<D>()` on fold hot boundaries (grep audit).
      **Deferred follow-up:** `as_ring_slice_trusted::<D>()` remains on prover/verifier
      hot paths behind schedule validation; full no-panic audit not closed.
- [x] Hand-built mixed-D `Schedule` fixture proves and verifies (e.g. levels 0–1 at
      D=128, level 2+ at D=64) with transcript replay **before** deleting the legacy
      suffix cold-path reference implementation (`mixed_d_per_level_e2e`).

**Correctness / perf (CI-hard where noted)**

- [ ] Proof wire bytes unchanged (pinned roundtrip on representative proofs).
      **Deferred:** uniform-D byte identity not re-pinned across full stack merge.
- [x] Prover≡verifier setup-geometry cross-check on the mixed-D fixture (shape-level
      CI test + minimal-envelope integration; large envelope sweep is `#[ignore]`).
- [ ] `cargo bench -p akita-pcs --bench ring_ntt` and `--bench root_kernels`: no
      regression on `dense_root_matvec_full_nv25_d32` and CRT matvec baselines
      (manual / release bench; not CI today).

**Planner/catalog completion (deferred beyond this PR)**

- [x] `expand_to_level_params` accepts `ring_d != policy.ring_dimension` when
      `ring_d` is nonzero and divides the policy envelope dimension; root-direct
      compact commit metadata errors propagate instead of being silently dropped.
- [ ] DP searches `ring_d` per fold step.
- [ ] Envelope sizing accumulates in field elements with `gen_ring_dim = D_max` (see
      Setup sizing today). Relax the `gen_ring_dim == Cfg::D` enforcement in
      `api/setup.rs` / `akita-setup/src/lib.rs` to `gen_ring_dim % levelD == 0`.
- [ ] Catalog emits tables for unified field-family configs.

### Testing Strategy

**CI-hard**

- `RingDimPlan::from_schedule` on all shipped generated tables (uniform D today).
- `setup_geometry_at` / `setup_active_ring_elems_at` golden vectors per representative
  level shape (single-tier, tiered, with/without prefix offload), pinning
  `(level shape, ring_d, required, offload?) → (active, ntt)`. These must be derivable
  **without** challenges (regression guard against re-coupling to `eq_tau1`).
- `NttCacheKey` / warm-cache unit tests: uniform-D `prepare_setup` registers exactly one
  contract entry `(Cfg::D, total)`; lazy builds outside the contract are allowed but
  warned; mixed-D fixture registers every required `ring_d` on the contract at prepare.
  `num_ring_elements` and `D`.
- `NttSlotCacheAny::as_d::<D>()` returns the correct variant on match and
  `InvalidSetup` on `ring_d` mismatch (no panic).
- `RingBuf::as_ring_slice` / `FlatRingVec::as_ring_slice` roundtrip and alignment.
- Grep inventory for deleted symbols (`Suffix*ProveBackend`, suffix-level
  `dispatch_ring_dim_result!`, `RECURSIVE_SUFFIX_RING_DIMENSIONS`,
  `SetupPrefixProverRegistry<F,`, `if D == SETUP_OFFLOAD_D_SETUP` gates).
- Descriptor digest pins: uniform-D proofs byte-identical before/after Phase 2 (all
  shipped presets have `Cfg::D == gen_ring_dim`).
- Regression: PCS e2e, commitment contract, transcript hardening, fold-linf.

**Integration (Phase 3 gate)**

- Mixed-D hand schedule fixture (see below): prover + verifier + transcript replay.
  Build against legacy cross-D suffix path first, then re-run after cutover.
- Prover≡verifier setup-geometry agreement on the fixture.
- Optional: `scripts/check-doc-guardrails.sh` after book stub updates.

**Mixed-D fixture (normative sketch)**

- Preset `fp128::D128Full` for root commit and setup envelope (`gen_ring_dim = 128`).
- Hand-built `Schedule`: fold levels 0–1 use `ring_dimension = 128`, level 2+
  use `64`; `LevelParams` copied from shipped `D128Full` / `D64Full` tables with
  consistent `current_w_len` / `next_w_len` chain.
- Witness length divisible at each `D` transition; `128 % 64 == 0` so the envelope
  buffer splits cleanly with no envelope-sizing change.
- Expected NTT keys: exactly two — `(128, total_128)` and `(64, total_64)`. Offload
  does not change this (it touches only the setup sumcheck's direct read).

### Performance

- **Gate:** `ring_ntt.rs`, `root_kernels.rs` baselines; profile workloads in
  `book/src/usage/profiling.md`.
- **Expect:** neutral or faster on uniform-D proofs (suffix cold path removed).
- **Expect:** mixed-D proofs register one full-envelope NTT cache per distinct `ring_d` on
  the setup prepare contract; uniform-D registers exactly one (today's work, via
  `prepare_setup`). Offload adds no cache builds.
- **Memory:** `NttCacheMap` holds one entry per distinct `ring_d`: 1 for uniform-D, 2
  for the mixed-D fixture, ≤ 4 in principle.
- **Advisory (not CI):** profile preset `onehot_fp128_d64:32:1` prove time within
  5% of pre-cutover baseline after Phase 2.

## Design

### Target architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Schedule / wire (runtime)                                        │
│  RingDimPlan, LevelParams.ring_dimension, RingBuf, Schedule     │
└────────────────────────────┬─────────────────────────────────────┘
                             │ RingLevelContext per level
┌────────────────────────────▼─────────────────────────────────────┐
│  Prepared state (D-free)                                          │
│  FlatMatrix (gen_ring_dim), CpuPreparedSetup<F>, NttCacheMap     │
└────────────────────────────┬─────────────────────────────────────┘
                             │ dispatch_ring_dim!(ring_d, |D| …)
┌────────────────────────────▼─────────────────────────────────────┐
│  Kernels (const D)                                                │
│  NttSlotCache<D>, matvec, ring_switch, CyclotomicRing<F,D> ops     │
└──────────────────────────────────────────────────────────────────┘
```

### Setup geometry: count vs weights (the central correction)

The per-level setup product needs two things that the current code computes together
inside `SetupContributionPlan::prepare`, but which have very different dependencies:

| Quantity | Depends on | When available |
|----------|-----------|----------------|
| **`required` (lambda-axis ring rows)** = the row-layout footprint (`a_end`) | LevelParams + **relation shape** (`num_claims`, `num_polynomials`, `m_row_layout`) | Shape-only; challenge-independent |
| **weights** (`bar_omega`, eq slices) | LevelParams + relation + **`tau1` / `x_challenges`** | Only during that level's protocol |

`required` is what sizes the NTT prefix and the setup `ring_view`. Today it is
obtained by building the full plan (which materializes weight tables and needs
challenges) and reading `plan.required()`. That coupling is what makes any "compute
the NTT keys up front" design impossible.

**Fix:** factor the cheap count out of the expensive weight build.

```rust
/// Pure, challenge-free row-layout footprint for a setup level.
/// Same arithmetic as SetupContributionPlan's a_start/.../a_end derivation,
/// but stops before any eq/weight materialization and takes no challenges.
pub fn setup_geometry_at(
    level: usize,
    schedule: &Schedule,
    relation_shape: &SetupRelationShape,   // num_claims, num_polynomials, m_row_layout, tier dims
) -> Result<SetupGeometry, AkitaError>;     // { required: usize }

pub fn setup_active_ring_elems_at<F>(
    level: usize,
    schedule: &Schedule,
    expanded: &AkitaExpandedSetup<F>,
    relation_shape: &SetupRelationShape,
) -> Result<usize, AkitaError> {
    let ring_d   = schedule.fold_level(level)?.params.ring_dimension;
    let required = setup_geometry_at(level, schedule, relation_shape)?.required;
    let setup_len = expanded.shared_matrix().total_ring_elements_at_dyn(ring_d)?;
    // FAIL CLOSED — do NOT `min`. Today's setup sumcheck errors here
    // (setup_sumcheck.rs: `if required > setup_len { InvalidSetup }`), and a silent
    // cap would (a) read fewer setup rows than the product needs and (b) make the
    // `context_at` feasibility comparison vacuous (capped value is ≤ envelope by
    // construction). Preserve the guard.
    if required > setup_len {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected setup product".into(),
        ));
    }
    Ok(required) // == min(required, setup_len) after the guard, but overflow is rejected
}
```

`SetupContributionPlan::prepare` is refactored to call `setup_geometry_at` for its
`required`/endpoints rather than recomputing them, so the count has exactly one
implementation and the weight path layers on top. `SetupRelationShape` is the small
shape projection (no `eq_tau1`, no `RingCommitment`) extracted from the relation; in
the fold loop it is read from the live relation, and for `from_schedule` validation it
is derivable from the schedule's witness/claim chain.

This is the single shared function required by both prover and verifier (What hurts
#6). Both sides must call it; no parallel copy survives.

### Setup-prefix offload (decoupled from the NTT cache)

A correction to an earlier draft of this spec: **offload does not size or key the NTT
cache.** The commitment matvec/quotient kernels read the shared-matrix NTT cache
(full envelope per `ring_d`, indexing a prefix). The setup sumcheck — the only place
offload acts — reads the shared matrix **directly** (`ring_view::<D>(1, setup_eval_len)`
→ its own lifted table; it never touches the NTT cache). Offload only changes
`setup_eval_len` (the direct read length) and absorbs `slot.id`. There is therefore
**no `setup_ntt_ring_elems` quantity** and `RingLevelContext` carries no prover-only
NTT field:

```rust
pub const MAX_FOLD_LEVELS: usize = 16;            // > deepest shipped schedule; from_schedule fails closed above
pub const SUPPORTED_RING_DIMS: [usize; 4] = [32, 64, 128, 256];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingLevelContext {
    /// Per-role ring dims for this level (`d_d | d_b | d_a`).
    pub role_dims: CommitmentRingDims,
    /// The fold / ring-switch / inner-commitment ring. Invariant: `== role_dims.inner`.
    /// Retained as a field so existing `ctx.ring_d` reads stay valid (the legacy scalar).
    pub ring_d: usize,
    /// Shape-only count of setup (inner-side, `d_a`) ring rows the level's setup product
    /// touches (`SetupContributionPlan::required`). Drives the offload decision and the
    /// setup sumcheck's direct `ring_view`. Identical on prover and verifier.
    pub setup_active_ring_elems: usize,
}
```

The NTT cache key for a *role* at a level is
`(role_d, total_ring_elements_at::<role_d>())` for `role_d ∈ {d_a, d_b, d_d}` —
derivable from the dim and the seed alone, with no dependence on the relation shape, the
registry, or offload. A level therefore touches up to three keys (fold / ring-switch +
inner matvec at `d_a`, outer matvec at `d_b`, opening matvec at `d_d`); uniform presets
collapse to one. Distinct `ring_d` values across the plan must appear on the setup prepare
contract (see Warm-cache policy). In this PR the interim
`d_a == d_b == d_d` guard (Per-block ring dimension) means one key per level for every
schedule; per-level mixed-D still varies the key across levels.

**Offload decision at level ℓ** (normative; stays inside the setup sumcheck / stage 3,
exactly where it is today, now ungated):

1. `natural_field_len = ctx.setup_active_ring_elems * ring_d` (today's `required * D`).
2. `setup_prefix_level_params(level_params, n_prefix, ring_d)` — already takes
   `d_setup`; no generalization needed.
3. If params are `Some` **and** the side's prefix registry has a slot whose
   `SetupPrefixSlotId` (already carries `d_setup`) covers `natural_field_len`, offload
   is **active**: absorb `slot.id` (both sides, same order as today) and read the slot's
   prefix length.
4. Else read the full matrix as today.

**Each side runs this with its own already-existing registry** — the prover with the
session `SetupPrefixRegistry` (this spec's D-free successor to
`SetupPrefixProverRegistry`), the verifier with `SetupPrefixVerifierRegistry`. Both feed
the *same* shared `select_setup_prefix_slot` and look up the *same* `SetupPrefixSlotId`,
which is exactly what keeps `setup_eval_len` and the transcript absorption identical on
both sides. The earlier draft’s `context_at(prefix_registry: None)` on the verifier was
wrong: the offload decision does not live in `context_at` (which is registry-free and
symmetric) — it lives here, in the setup sumcheck (prover) and stage 3 (verifier), where
each side already holds its registry. Inputs to the decision are challenge-free
(`setup_active_ring_elems` is shape-only), so prover and verifier agree without a
transcript digest of `setup_eval_len`.

Because `select_setup_prefix_slot` already returns `None` when no matching slot exists,
deleting the `if D == SETUP_OFFLOAD_D_SETUP` gate is behavior-preserving for shipped
presets (setup construction still populates slots only at `d_setup = 64`; *which* slots
exist is a separate, out-of-scope question). The genuine remaining work is narrow:
**(a)** delete the `if D == SETUP_OFFLOAD_D_SETUP` gates at its two call sites
(`setup_sumcheck.rs`, `stage3.rs`); **(b)** demote the *prover*
registry/slot off `const D` (see Phase-ordering note). Retain
`SETUP_OFFLOAD_D_SETUP` (`d_setup = 64`) for setup-prefix slot construction in
`akita-setup` / `setup_prefix.rs`.

### `RingDimPlan` and `RingLevelContext`

Central runtime authority for per-fold ring geometry, derived once from the effective
`Schedule` at prove/verify entry. **`RingDimPlan` is a derived view**; it is not
separately digested (per-level `ring_dimension` is already bound in
`PlanSection::from_schedule` via `LevelParams::append_descriptor_bytes`).

```rust
/// Per-fold ring dimensions by role. Invariant: `opening | outer | inner`
/// (i.e. `d_d | d_b | d_a`), all ∈ `SUPPORTED_RING_DIMS`.
///   inner   = d_a — fold / ring-switch / inner-commitment ring (needs the challenge
///                   family; == legacy per-level `ring_d`)
///   outer   = d_b — outer-commitment (B) ring
///   opening = d_d — opening-commitment (D) ring
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentRingDims { pub inner: usize, pub outer: usize, pub opening: usize }
impl CommitmentRingDims {
    pub fn uniform(d: usize) -> Self { Self { inner: d, outer: d, opening: d } }
}

pub struct RingDimPlan {
    role_dims: [CommitmentRingDims; MAX_FOLD_LEVELS],   // validated per-level, per-role dims
    pub num_folds: usize,
}

impl RingDimPlan {
    pub fn from_schedule(
        schedule: &Schedule,
        seed: &AkitaSetupSeed,             // gen_ring_dim + identity already live here
    ) -> Result<Self, AkitaError>;

    pub fn dims_at(&self, level: usize) -> Result<CommitmentRingDims, AkitaError>;
    /// The fold ring `d_a` (= `dims_at(level).inner`); the legacy scalar accessor.
    pub fn dim_at(&self, level: usize) -> Result<usize, AkitaError>;
    /// Spans every distinct dim over ALL roles and ALL levels (drives warm-cache).
    pub fn unique_dims(&self) -> impl Iterator<Item = usize> + '_;

    /// Per-level geometry. A per-level RUNTIME call (needs the live relation shape).
    /// NOT precomputable before the fold loop — the relation shape at level ℓ is the
    /// output of folding ℓ-1. Identical signature and result on prover and verifier
    /// (the offload decision, which needs the registry, lives in the setup sumcheck,
    /// not here).
    pub fn context_at<F>(
        &self,
        level: usize,
        schedule: &Schedule,
        expanded: &AkitaExpandedSetup<F>,
        relation_shape: &SetupRelationShape,
    ) -> Result<RingLevelContext, AkitaError>;
}
```

`dim_at` returns `role_dims[level].inner` (= `d_a`) after bounds / support checks;
`dims_at` returns the full triple. `context_at` replaces today's
`validate_level_dispatch`: it checks `SUPPORTED_RING_DIMS`, the `d_d | d_b | d_a`
nesting, `schedule[level].params.ring_dimension == d_a`, then derives
`setup_active_ring_elems` (the inner / `d_a` setup-product count) via the shared
`setup_active_ring_elems_at`. The NTT cache keys for the level are
`(role_d, total_ring_elements_at::<role_d>())` per role, formed separately at the kernel
boundary — `context_at` does not compute them.

Throughout the rest of this spec the scalar `ring_d` / `dim_at(ℓ)` denotes the fold ring
`d_a`; the outer / opening dims `d_b, d_d` enter only the `B` / `D` commitment matvecs
and their NTT keys.

**Prove entry:** build `RingDimPlan::from_schedule(schedule, seed)` for validation; the verifier builds
the same plan from the same schedule + seed. Per-level contexts are computed **inside**
the loop (memoize per level if the same context is needed twice).

#### `RingDimPlan::from_schedule` validation

`from_schedule` returns `InvalidSetup` (never panic) on:

| Check | Rule |
|-------|------|
| Fold count | `schedule_num_fold_levels(schedule) ≤ MAX_FOLD_LEVELS` |
| Supported dims | every role dim `d_a, d_b, d_d ∈ SUPPORTED_RING_DIMS` |
| Role nesting | `d_d \| d_b \| d_a` at every level |
| Interim uniform guard (temporary) | `d_a == d_b == d_d` at every level — enforced until per-block kernel execution lands (PR-perblock-exec); relaxes to the `d_d \| d_b \| d_a` rule above |
| Envelope divisibility | `seed.gen_ring_dim % d_a == 0` at every level (implies `% d_b`, `% d_d` by nesting) |
| Schedule consistency | `dims_at(ℓ).inner == schedule[ℓ].params.ring_dimension` (= `d_a`) |
| Per-role witness chain | each committed object's length divisible by its role dim: witness / `d_a`, `t̂` / `d_b`, `ê` / `d_d`; terminal shape valid |
| Cross-level lengths | `next_w_len` consistent with digit layout at `d_a` and the next level's `d_a` when they differ |
| Root layout | `dims_at(0).inner` matches committed polynomial ring layout (validated at PCS commit entry) |

Active-prefix feasibility is a **runtime** check (it depends on the live relation
shape), not a `from_schedule` check: `setup_active_ring_elems_at` itself **fails closed**
with `InvalidSetup` when `required > total_ring_elements_at(ring_d)` — the same guard
today's setup sumcheck applies — and `context_at` surfaces that error. (Do **not** model
this as `min(required, setup_len)`: capping silently truncates an under-provisioned setup
and makes the comparison vacuous.) Offload-slot coverage is checked where offload is
decided (setup sumcheck / stage 3). Mixed-D schedules that bypass generated expansion (hand-built
fixtures, the mixed-D fixture) must still satisfy this catalog.

### Per-block ring dimension (A/B/D): correctness and soundness

Why a single fold may run three ring dimensions at once. The justification is the
ring-switch lift (Akita paper, `sec:prelim-ring-switch` and `rem:per-relation-ring-dim`),
summarized here so this spec is self-contained.

**Status in this PR.** The geometry below — the `CommitmentRingDims` triple,
`d_d | d_b | d_a` validation, and transcript binding — lands now; the kernels do **not**
yet consume distinct role dims. A construction-time guard enforces `d_a == d_b == d_d`
(uniform), so executed behavior is byte-identical to per-level mixed-D. Removing that
guard — wiring `B`/`D` matvecs and the ring-switch weights at distinct `d_b, d_d`, plus a
non-uniform fixture — is the per-block execution follow-up (PR-perblock-exec). The
correctness/soundness argument is recorded now so the follow-up inherits its contract.

**Model.** The root relation has four row families: outer-commitment `B_g t̂_g = u_g`,
opening-commitment `D ê = v`, folded-evaluation, and folded-commitment `… = A z`. Ring
switching lifts each family to `Z_q[X]` with its own modulus `X^{d_b}+1` and evaluates at
one shared challenge `α ∈ F_{q^k}`; the family contributes `w̃(x,y) · α̃^{(d_b)}(y) · m(x)`
to the combined sum-check, where `α̃^{(d_b)}` is the length-`d_b` prefix of the one power
ladder `{α^{2^j}}`. The committed witness MLE is flat over `Z_q`; the dimension enters
only as that public weight, so the commitment is agnostic to per-role dims.

**Which rows may diverge.** Only the folded rows multiply the witness by a fold challenge
`c ∈ C`, so only they require the challenge family and its operator-norm certification —
pinning `d_a` (and the inner matrix `A`, whose `A z` lives in `d_a`). `B_g t̂_g = u_g`
and `D ê = v` carry no challenge, so `d_b, d_d` are free of the `≥ 2^128` sparse-family
floor and chosen for Module-SIS, packing, and tail size; Akita nests them `d_d | d_b | d_a`.

**Completeness.** Each family's quotient `r^{(b)}` exists (Euclidean division by
`X^{d_b}+1`); the heterogeneous-arity sum-check (different `log d_b` degree-axes batched by
random coefficients) accepts the honest prover.

**Soundness** — three pieces:

1. *Per-block ring-switch error.* A false family identity survives random `α` with
   probability `≤ (2·d_b − 1)/q^k`; over the families the union bound gives
   `Σ_b (2·d_b − 1)/q^k`, negligible.
2. *Non-adaptive dimensions.* The per-level, per-role triple `(d_a, d_b, d_d)` is bound in
   the transcript (schedule digest via `LevelParams::append_descriptor_bytes`) **before**
   `α` and the fold challenges are drawn, so the union above is over a fixed family set,
   not one the prover chooses after seeing `α`.
3. *Binding transfers across views.* The shared digit objects `t̂` (folded in `A`-rows at
   `d_a`, committed by `B` at `d_b`) and `ê` (folded at `d_a`, committed by `D` at `d_d`)
   are read under two ring views of the *same* flat `Z_q` coefficients. The fold extractor
   (coordinate-wise special soundness in `α` and the fold challenges) recovers each witness
   sub-block as flat coefficients; the balanced-digit `ℓ_∞` bound is a function of those
   coefficients, hence view-independent, so the Module-SIS binding of `B` at `d_b` and `D`
   at `d_d` applies to the extracted object directly.

**Byte-compat.** Uniform presets set `d_a == d_b == d_d`, recovering exactly today's
single-ring behavior.

### Warm-cache policy

Because the NTT cache key is `(ring_d, total_ring_elements_at::<ring_d>())` — a function
of `ring_d` and the setup seed alone — the **minimum** slot for a uniform preset is known
at prepare time: `(envelope D, total)` for the setup's `gen_ring_dim`. Do **not** over-build
at prepare: register only what the setup contract promises (today exactly one envelope slot
per `prepare_setup`).

**Setup prepare (`prepare_setup`):**

```rust
// Minimum contract: envelope ring at compile-time D (uniform-D today).
let key = NttCacheKey::from_envelope(expanded, D)?;
prepared.register_setup_contract_ntt_slot(key)?;  // &mut prepared, once per prepare_setup
```

`prepare_expanded` returns an **empty** map. Host code that commits or proves should call
`prepare_setup`, not `prepare_expanded`, unless it will register the contract itself.

**Commit / prove (lazy extension):**

If a commit or prove path needs an NTT key **outside** the setup prepare contract, the
backend may still build it via `ensure_ntt_slot` (correctness fail-open), but **must emit a
diagnostic warning** (`tracing::warn!`, target `akita_prover::ntt_cache`) indicating the
setup envelope or prepare path was likely undersized. This should not happen on correctly
sized uniform-`D` workloads. Mixed-D hosts that genuinely need multiple dims must extend
`prepare_setup` to register every required key on the contract (still no speculative
over-build beyond the declared contract).

**`RingDimPlan` at prove/verify entry** validates schedule/seed geometry; it does **not**
warm caches. Warming is a prepare concern.

```rust
// Prove / verify entry (validation only):
let plan = RingDimPlan::from_schedule(schedule, seed)?;
// ... root_fold / suffix read ntt_slot via &prepared; no prove-time warm loop ...
```

**Traps (unchanged intent):**

- The suffix loop starts at level **1**; the **root fold at level 0** also reads the
  prepared-setup NTT cache. The setup contract must include every `ring_d` those paths
  read before the first fold (envelope `D` suffices for uniform-D today).
- Standalone `commit` before prove uses the same `prepare_setup` contract; it must not
  rely on prove-time warming.

For a uniform-`D` proof, `prepare_setup` registers exactly one cache `(Cfg::D, total)` —
byte-for-byte the work `prepare_expanded` does eagerly today, relocated behind
`register_setup_contract_ntt_slot`. A correctly sized mixed-D setup registers every distinct
`ring_d` on the contract at prepare time (for example `(128, total_128)` and `(64, total_64)`
for the mixed-D fixture); lazy builds outside the contract are a sizing bug, not the
steady-state path.

(`ensure_ntt_slot` remains idempotent. Prefix-sizing a cache below the full envelope is a
deferred optimization, orthogonal to this policy.)

### D-free `PreparedSetup` and NTT caches

**Today:**

```201:202:crates/akita-prover/src/compute/cpu.rs
        let total = expanded.shared_matrix.total_ring_elements_at::<D>()?;
        let ntt_shared = build_ntt_slot(expanded.shared_matrix.ring_view::<D>(1, total)?)?;
```

**Target:**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NttCacheKey {
    pub ring_d: usize,
    pub num_ring_elements: usize,
}

pub enum NttSlotCacheAny {
    D32(NttSlotCache<32>),
    D64(NttSlotCache<64>),
    D128(NttSlotCache<128>),
    D256(NttSlotCache<256>),
}

impl NttSlotCacheAny {
    /// Cache-HIT accessor. Fallible: a variant whose ring_d != D returns InvalidSetup.
    pub fn as_d<const D: usize>(&self) -> Result<&NttSlotCache<D>, AkitaError>;
}

pub struct CpuPreparedSetup<F> {
    expanded: Arc<AkitaExpandedSetup<F>>,
    shared_ntt: NttCacheMap,           // HashMap<NttCacheKey, NttSlotCacheAny>
    i8_capacity: CrtI8CapacityMap,     // per ring_d, from selected_crt_i8_capacity_profile
    #[cfg(feature = "zk")]
    zk_b_ntt: NttCacheMap,
    #[cfg(feature = "zk")]
    zk_d_ntt: NttCacheMap,
}

pub trait ComputeBackendSetup<F> {
    type PreparedSetup: Send + Sync;  // no const D
    fn prepare_expanded(expanded: Arc<AkitaExpandedSetup<F>>) -> Result<Self::PreparedSetup, ...>;
    /// Register `key` on the setup prepare contract (minimum envelope at `prepare_setup`).
    fn register_setup_contract_ntt_slot(prepared: &mut Self::PreparedSetup, key: NttCacheKey) -> Result<(), AkitaError>;
    /// Build the cache for `key` if absent. Outside the setup contract: allowed, but
    /// must log a sizing diagnostic (see Warm-cache policy).
    fn ensure_ntt_slot(prepared: &mut Self::PreparedSetup, key: NttCacheKey) -> Result<(), AkitaError>;
    /// Read a previously-built slot. `InvalidSetup` if the key was never built.
    fn ntt_slot<'a>(prepared: &'a Self::PreparedSetup, key: NttCacheKey)
        -> Result<&'a NttSlotCacheAny, AkitaError>;
}
```

`prepare_expanded` builds an **empty** map. `prepare_setup` registers the minimum envelope
contract. Additional keys may be built lazily via `ensure_ntt_slot` (with diagnostic if
outside the contract).

Cache-build path (inside `ensure_ntt_slot`):

```rust
dispatch_ring_dim!(key.ring_d, |D| {
    // key.num_ring_elements == total_ring_elements_at::<D>() in the baseline
    let view = expanded.shared_matrix().ring_view::<D>(1, key.num_ring_elements)?;
    NttSlotCacheAny::from(build_ntt_slot(view)?)
})
```

Cache-hit (consume) path at a kernel boundary — note the double dispatch (runtime
`ring_d` → `const D`, then the fallible enum extraction):

```rust
dispatch_ring_dim!(ctx.ring_d, |D| {
    let any = backend.ntt_slot(prepared, key)?;
    let slot: &NttSlotCache<D> = any.as_d::<D>()?;   // InvalidSetup if variant mismatches
    run_matvec::<F, D>(slot, ...)
})
```

ZK blinding matrices follow the same pattern in their own maps (`zk_b_ntt`, `zk_d_ntt`):
one full-envelope cache per distinct `ring_d`.

**`NttCacheMap`:** `HashMap<NttCacheKey, NttSlotCacheAny>`, populated by
`register_setup_contract_ntt_slot` at `prepare_setup` and, if needed,
`ensure_ntt_slot` (lazy, with diagnostic when outside the contract). Cardinality on the
contract: **1 for uniform-`D`**, more only when the setup explicitly registers them.

### Deferred optimization: prefix-sized NTT caches (mixed-D memory) — TRACKED FOLLOW-UP

The baseline keeps one **full-envelope** cache per distinct `ring_d`. For mixed-D this is
correct but memory-heavy: a proof that uses `D1` at the root and `D2 < D1` only for late,
small folds still carries a **second whole-matrix** transform at `D2`, even though the
`D2` folds touch only a small prefix. Worked example — root at `D1 = 128`, later folds at
`D2 = 64`, envelope `gen_ring_dim = 128` with `N` ring elements:

| Cache | Elements | Total transformed size |
|-------|----------|------------------------|
| `D1 = 128`, full envelope | `N` | `∝ N·128·K·2` |
| `D2 = 64`, full envelope (**baseline**) | `2N` | `∝ N·128·K·2` — same magnitude; the whole matrix again |
| `D2 = 64`, prefix-sized (**this optimization**) | `k` (touched prefix), `k ≪ 2N` | `∝ k·64·K·2` ≪ full |

Both full caches span the entire matrix (the field-element count is invariant) and cannot
share bytes (distinct transform domains — see NTT cache today), so the baseline pays ~one
extra whole-matrix transform per extra `ring_d`.

**The optimization.** Size each `ring_d`’s cache to the **maximum commit footprint across
the levels at that `ring_d`** (`max(row_len · row_width)`), not the full envelope:

```
num_ring_elements(d) = max over levels ℓ with dim_at(ℓ)==d of commit_footprint(ℓ)   // ≤ total_at_d
```

That footprint is a function of the schedule + opening-batch shape — challenge-free and
known up front — so prepare-time contract registration can size each entry precisely. No new key type is
needed (`NttCacheKey` already carries `num_ring_elements`; only the warmed value changes).

**Correctness requirement (the subtle part).** Every consumer at `ring_d` — root commit,
`commit_next_w`, matvec, quotient at each level — must fit inside the sized prefix, so the
warmed value must be the **max** request, not any single level’s. The existing
`validate_digit_row_request` check already fails closed if a request exceeds the cached
length, so an under-sizing bug surfaces as `InvalidSetup`, never as silent corruption.

**Why deferred (not in this PR).** It needs the commit-footprint projection and a proof
that the warmed bound covers every consumer at each `ring_d`; the cutover prioritizes
byte-identical uniform-`D` behavior (where this collapses to the single full-envelope
entry anyway, so there is nothing to gain there). It is orthogonal to both the `ring_d`
keying and the Phase-4 planner. **Track it as the first mixed-D memory follow-up** once
mixed-D schedules actually ship — that is when the second whole-matrix cache starts to
cost real memory.

### Flat ring storage

Unify in-memory owners; keep wire encoding unchanged:

| Type | Role |
|------|------|
| `RingBuf<F>` | In-memory owned `Vec<F>`; compact (no tagged `ring_dim`). **Wire name stays `FlatRingVec`.** |
| `RingSlice<'a,F,D>` | Borrowed `&[CyclotomicRing<F,D>]` via `repr(transparent)` |
| `RingMatrixView<'a,F,D>` | Setup matrix view (existing) |
| `DigitBuf` / `DigitRingView<'a,D>` | `Vec<i8>` + `&[[i8; D]]` (recursive witness pattern) |

API surface (match existing `FlatRingVec::as_ring_slice` ergonomics):

```rust
impl<F> RingBuf<F> {
    pub fn as_ring_slice<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError>;
    pub fn as_single_ring<const D: usize>(&self) -> Result<&CyclotomicRing<F, D>, AkitaError>;
}
```

`RingSlice<'a,F,D>` is a thin newtype wrapper when a named view type helps. Avoid
callback-only APIs (`with_rings`) on hot paths.

Migrate **Phase 3** owners: `RingRelationInstance` fields (`y`, `v`, row coeffs),
`PreparedFold`, verifier `PreparedFoldReplay`. `RingCommitment` and root
`DensePoly.coeffs` may keep `Cfg::D`-typed root layout through Phase 2 (root ring
fixed at commit time).

`CyclotomicRing<F, D>` remains for single-element algebra and as the view target; not
used as `Vec` element type in protocol storage.

### Phase-ordering note: D-free `SetupPrefixSlot` vs commitment/hint demotion

`SetupPrefixSlot<F, const D>` today embeds `commitment: RingCommitment<F, D>` and
`hint: AkitaCommitmentHint<F, D>` (`setup_prefix.rs`). Making the slot **fully** D-free
in Phase 1 would force demoting `RingCommitment` and `AkitaCommitmentHint`, which this
spec otherwise defers to Phase 3. Resolve by **decoupling keying from storage**:

- Phase 1: the **registry** keys on `SetupPrefixSlotId` (already carries `d_setup`) and
  the public path uses `SetupPrefixPublicCommitment<F> { rows: Vec<FlatRingVec<F>> }`
  (already D-free). The slot's prover-side `commitment`/`hint` may stay D-typed,
  reached by dispatching on `slot.id.d_setup`. The registry is "D-free at the id/keying
  level," which is what the offload decision and transcript binding need.
- Phase 3: demote `RingCommitment` / `AkitaCommitmentHint` (or store slot rows as
  `RingBuf`) and drop the residual `const D` from the slot.

This keeps Phase 1 landable without dragging Phase 3 commitment work forward.

### Backend façade: sole `match` on `ring_d`

**Delete dispatch from protocol orchestration:**

| File | Remove |
|------|--------|
| `akita-prover/src/protocol/core/suffix.rs` | `if level_d == D` / else rebuild stack |
| `akita-verifier/src/protocol/core/suffix.rs` | unconditional `dispatch_ring_dim_result!` |
| `akita-prover/src/protocol/ring_switch/commit.rs` | cross-D `dispatch_commit_w_with_layout_policy` |

**Keep / centralize dispatch:**

| Location | Role |
|----------|------|
| `akita-types/src/dispatch.rs` | `dispatch_ring_dim!(ring_d, \|D\| body)` (new infallible-body variant; or rename `dispatch_ring_dim_result!`) |
| `ComputeBackendSetup::ensure_ntt_slot` / `ntt_slot` | cache build + `as_d` extraction |
| `CommitmentComputeBackend::*` | matvec, digit_rows, commit_rows |
| `RingSwitchComputeBackend::*` | quotient / relation rows |
| `OpeningProveBackendFor` / `TensorBackendFor` | fold / tensor kernels |

Single `ProverComputeStack` (or hardware-tiered `LevelProveStacks` without per-D type
parameter). `OperationCtx::new` validates prepared state against expanded setup once;
per-call `ring_d` selects cache key and kernel monomorphization.

### Verifier path

Verifier has no `PreparedSetup` / NTT cache. Changes:

- Build `RingDimPlan` from schedule + seed (same as prover).
- `verify_fold(..., ctx: &RingLevelContext, ...)` without the suffix-level dispatch
  macro. `RingLevelContext` is identical to the prover's (no NTT field).
- `context_at` is the same call as the prover's; the verifier simply never warms an NTT
  cache. The offload active/inactive decision (for transcript absorption) happens in
  stage 3 via the shared `select_setup_prefix_slot`, exactly as on the prover.
- Flat proof decode: `proof.v().as_ring_slice::<D>()` where `D = ctx.ring_d`.
- `validate_level_dispatch` replaced by `RingDimPlan::context_at`.

### Descriptor binding (single authority)

| What | Authority | Notes |
|------|-----------|-------|
| Per-fold `ring_dimension` | `PlanSection` schedule digest | `LevelParams::append_descriptor_bytes` already pushes `ring_dimension` |
| Setup envelope degree | `AlgebraSection.ring_dimension_d` | Record `gen_ring_dim`. **No-op today**: `gen_ring_dim == Cfg::D` is enforced, so bytes are unchanged for every shipped preset |
| Field tower | `AlgebraSection` extension degrees | Unchanged |
| Setup-prefix offload | Transcript `ABSORB_SETUP_PREFIX_SLOT` | Absorbs `SetupPrefixSlotId` when offload active |
| `setup_*_ring_elems` | Derived only | Single shared function on both sides; **not** a separate digest field |

`bind_transcript_instance_descriptor` drops the `const D` type parameter. Add
`AlgebraSection::for_envelope::<F, E>(gen_ring_dim)` alongside the existing
`for_fields::<F, E, const D>`; both sides call `for_envelope` with
`expanded.shared_matrix().gen_ring_dim()`.

**Pinned digests:** because `Cfg::D == gen_ring_dim` for all current presets, the
`AlgebraSection` bytes are **unchanged** and `instance_descriptor/tests.rs` needs **no
re-pin**. If a pinned digest does move during this change, that is a signal that a
preset's envelope diverged from `Cfg::D` — investigate rather than blindly re-pin.
Document the forward-looking relabeling in `specs/transcript-hardening.md`.

### `CommitmentConfig` and PCS entry

| Today | After |
|-------|-------|
| `CommitmentConfig::D` | **Setup envelope default** (`gen_ring_dim`) and root-commit layout; not suffix authority |
| `AkitaCommitmentScheme<const D, Cfg>` | `AkitaCommitmentScheme<Cfg>` (**done**; normal public PCS surface is D-free, typed dispatch traits are `Typed...`) |
| `AkitaProverSetup<F, const D>` | `AkitaProverSetup<F>` (envelope degree read from `seed.gen_ring_dim`; relax the setup `gen_ring_dim != D` checks to compare against the seed itself) |
| `batched_prove` (D from scheme struct) | `batched_prove` builds `RingDimPlan` from resolved schedule + seed |
| `ring_challenge_config(d)` | called with `plan.dim_at(ℓ)` per fold |
| `bind_transcript_instance_descriptor<const D>` | envelope `gen_ring_dim`; no type-param `D` |

Root commit before schedule resolution may still use a config default `D` for API
ergonomics; the first fold level in the plan must match the committed polynomial
layout.

### Where the `match` lives (summary)

| Layer | 4-way match on ring dim? |
|-------|---------------------------|
| `prove_suffix` / `verify_suffix` | **No** |
| `prove_fold` / `verify_fold` orchestration | **No** (may call D-free backends that dispatch internally) |
| `PreparedFold`, `RingRelationInstance`, `RingBuf` | **No** |
| `PreparedSetup` storage | **No** (enum erasure only) |
| Each backend method entry | **Yes** (one branch; build + `as_d` consume) |
| NTT butterfly / matvec inner loop | **No** |

### Inventory: deleted symbols (target)

- `dispatch_ring_dim_result!` call sites in `suffix.rs`, verifier `suffix.rs`,
  `ring_switch/commit.rs` orchestration paths
- `SuffixRingSwitchProveBackend`, `SuffixDispatchOpeningProveBackendFor`,
  `SuffixDispatchTensorProveBackendFor`, `SuffixWitnessOpeningProveBackendFor`,
  root-tensor siblings
- `RECURSIVE_SUFFIX_RING_DIMENSIONS`
- `RecursiveProveBackend` six-bound `ProveFlowBackendFor` supertrait lattice
- `ProverComputeStack<F, const D, ...>`, `OperationCtx<F, B, const D>`
- `CpuPreparedSetup<F, const D>` as public type (replaced by D-free version)
- `PreparedFold<F, L, const D>`, `RingRelationInstance<F, const D>` (after Phase 3)
- `SetupPrefixProverRegistry<F, const D>` and the `if D == SETUP_OFFLOAD_D_SETUP`
  eligibility gates; the suffix cold-path empty `SetupPrefixProverRegistry::new()` workaround

Retain (already D-free / already shared — do not rewrite):

- `dispatch_ring_dim!` / `dispatch_ring_dim_result!` in `akita-types` for kernel entry
- `select_setup_prefix_slot`, `setup_prefix_level_params`, `SetupPrefixSlotId`
  (already take `d_setup`), `SetupPrefixVerifierRegistry<F>` (already D-free)
- `SETUP_OFFLOAD_D_SETUP` (`d_setup = 64` naming constant in `setup_prefix.rs`)
- `NttSlotCache<const D>`, `CyclotomicRing<F, D>`, all SIMD kernels
- `validate_level_dispatch` semantics (subsumed by `RingDimPlan::context_at`)

### Wire changes

**Default: no wire format change.** `AkitaLevelProof` already stores compact
`FlatRingVec`; per-level `D` is implied by the schedule digest in `PlanSection`.

**Descriptor:** `AlgebraSection.ring_dimension_d` semantics become envelope
`gen_ring_dim` (no byte change today; see Descriptor binding). No new `PlanSection`
field for `RingDimPlan`. Document the relabeling in `specs/transcript-hardening.md`.

### Public API cutover

Phased migration for PCS and compute surfaces (full cutover, no shims):

| Phase | `AkitaCommitmentScheme` | `CommitmentProver` / `Verifier` | `RingCommitment` / hints | `PreparedSetup` | Caller-visible `D` |
|-------|-------------------------|----------------------------------|--------------------------|-----------------|-------------------|
| 1 | `<const D, Cfg>` unchanged | `<F, D>` unchanged | `<F, D>` | D-free internal on `CpuBackend` | type param + schedule |
| 2 | `<Cfg>` (**shipped**) | typed dispatch only (`Typed...<F, D>`) | flat public commitments, typed internals | D-free | root dispatch: `Cfg::D`; prove: `RingDimPlan` |
| 3 | `<Cfg>` | public verifier `<F>`; no public prover trait | `RingBuf` / D-free where applicable | D-free | `RingDimPlan` only at PCS boundary |

**End-state integrator snippet (Phase 2+):**

```rust
type Scheme = AkitaCommitmentScheme<fp128::D64Full>;
let setup = Scheme::setup_prover(nv, batch)?;
let stack = UniformProverStack::uniform(&backend, &prepared, &setup.expanded)?;
let proof = Scheme::batched_prove(&setup, claims, &stack, transcript, ...)?;
```

Root polynomial traits (`RootProvePoly<F, D>` with `D = Cfg::D`) may remain through
Phase 2; suffix mixed-D does not require demoting root poly traits.

Custom backend implementors: see updated `docs/compute-backends.md` checklist
(`prepare_expanded` once, `ensure_ntt_slot(key)`, `RingLevelContext` on row kernels).

### Alternatives considered

| Alt | Verdict |
|-----|---------|
| A. Single `Cfg::D` only; delete mixed-D | Rejected: forecloses planner optimization |
| B. Flat storage only; keep suffix dispatch | Rejected: leaves orchestration tax |
| C. Runtime-D NTT without `const D` | Rejected: SIMD regression |
| D. 16 fixed `PreparedSetup` slots per proof | Rejected: duplicates caches |
| E. `enum PreparedSetup { D32(...), D64(...), ... }` per stack | Rejected: multiplies stacks; prefer keyed cache |
| F. Per-level prefix-keyed NTT cache (key per `(ring_d, level prefix)`) | Rejected: kernels index a prefix of one full slot, so per-level keys would build several overlapping caches and **regress** uniform-D from one cache to many. Adopted instead: one full-envelope cache per distinct `ring_d` on the setup prepare contract |
| G. Share NTT bytes across `ring_d` via Cooley–Tukey | Rejected: cyclic roots nest (`D`-th ⊂ `2D`-th) but the view splits into *raw halves* whose separate transforms are not sub-blocks of the `2D` transform — that holds `NTT(p_lo±p_hi)` in mixed cyclic/negacyclic domains; negacyclic roots don't nest at all. Recovery costs a domain-crossing butterfly over a usually-empty overlap and saves no memory (see NTT cache today) |

## Future: unified field-family planner

**Not in scope for Phases 1–3.** Documented here as motivation and direction.

### Today: constant D per preset family

`fp128` exposes separate `CommitmentConfig` impls and generated schedule modules per
ring dimension (`D32Full`, `D64Full`, `D128Full`, one-hot variants, tiered). Each
`PlannerPolicy` fixes `ring_dimension: Cfg::D`. The DP searches fold geometry
(`log_basis`, `m_vars`, `r_vars`, ranks) but not per-step `ring_d`. Users pick a preset
name that embeds D.

### Target: one schedule optimizer per field family

Once runtime ring infrastructure ships, the planner can treat **`ring_d` as a DP
decision variable** at each fold step:

```
GeneratedFoldStep {
    ring_d: u32,      // already stored; today always == policy.ring_dimension
    log_basis: u32,
    m_vars: u32,
    ...
}
```

**Relax** `expand_to_level_params` check `ring_d == policy.ring_dimension`.

**Extend** `find_schedule` / `schedule_params` to try `ring_d ∈ {32, 64, 128, 256}`
(or family-specific subset) per step, subject to:

- SIS floors at each `(family, ring_d, rank)` (`akita-types` generated tables)
- `ring_challenge_config(ring_d)` entropy validation
- **Setup envelope (field-element accumulation):** `gen_ring_dim = D_max` over all
  emitted steps; `max_setup_len = max_field_len / D_max`; `gen_ring_dim % ring_d == 0`
  at every step (see Setup sizing today). Relax the enforced `gen_ring_dim == Cfg::D`
  checks accordingly.
- Witness length divisibility: `current_w_len % ring_d == 0` at each transition
- Proof-size objective includes mixed-D costs (different `level_bytes` per D)

**Catalog simplification (potential):**

| Today | Future |
|-------|--------|
| `fp128_d64_full`, `fp128_d128_full`, … separate tables | One `fp128_full` table with mixed `ring_d` per step |
| User selects preset by embedded D | User selects field family + witness mode; planner picks D ladder |
| `CommitmentConfig::D` names the preset | `CommitmentConfig` names field + decomposition; `RingDimPlan` names D ladder |

**Open questions for planner PR (not resolved here):**

- Optimal `ring_d` transition rules (monotone decrease? arbitrary ladder?)
- How the `D_max` envelope and `max_setup_matrix_size` bound mixed-D schedules
- Whether root `D` must match the commitment API or can be schedule-only
- Catalog identity when `ring_dimensions` has multiple entries per family
- Interaction with tensor-projection geometry (`protocol-field-geometry-cutover.md`)
- Regenerating vs hand-tuning initial mixed-D tables for production presets

Runtime ring cutover **does not block** on these answers. Phases 1–3 must not assume
uniform `D`; Phase 4 planner work consumes the infrastructure.

## Documentation

- `book/src/how/architecture.md`: `RingDimPlan`, `RingBuf`, runtime `D` vs
  `gen_ring_dim`, setup-sizing model, NTT-cache-per-`(D, prefix)`, diagram of three
  layers.
- `book/src/how/proving/fold-path.md` (stub): schedule-driven ring dimension.
- `docs/compute-backends.md`: `PreparedSetup` without `const D`; `ensure_ntt_slot`.
- `docs/doc-blast-radius.json`: add regions for this spec.
- Cross-link from `protocol-field-geometry-cutover.md` (coordinate `PreparedFold`
  target shape and `prove_suffix` PR order; see Execution).

### Sequencing with `protocol-field-geometry-cutover.md`

Both specs touch `protocol/core/{fold,suffix,prove}.rs` and `compute/poly.rs`.

**Rule:** Land runtime ring **Phase 1** (`RingDimPlan`, keyed NTT, shared setup
geometry function, D-free prefix registry) before either spec rewrites fold
preparation.

**`PreparedFold` target:** D-free storage (`RingBuf` fields inside enum variants if
geometry adds `SingleField` / `TensorProjection` tails). Negotiate enum shape in the
geometry spec against this storage layout; do not land incompatible
`PreparedFold<F,L,D>` and `PreparedFold<F,L>` refactors in parallel.

**Geometry Phase 2** (fold prep split) should follow runtime ring **Phase 2** (suffix
loop without stack rebuild) or land in the same PR series with a shared owner for
`prove_suffix`.

## Execution

Phases 1–3 land as **one PR** on `quang/runtime-ring-cutover`, in eight waves (0–7).
The work is too coupled to split (shared geometry is soundness-load-bearing; the suffix
cutover depends on D-free prepared setup; the mixed-D fixture is the deletion gate).
Phase 4 (planner DP, field-element envelope sizing) is a **separate** PR, out of scope.

### Working agreement (read first)

- **Each wave must leave the workspace compiling and its tests green** before starting
  the next. Commit per wave (or per sub-step). The verify command for every wave is at
  minimum `cargo build --workspace` and `cargo build --workspace --features zk`, plus
  the per-wave tests listed.
- **Never delete a symbol until its replacement is green and in use.** Deletions are
  scheduled into the wave where the last caller is migrated (mostly Wave 5).
- **Behavior-preserving until Wave 6.** Through Wave 5, every shipped uniform-`D` proof
  must produce **byte-identical** proof wire and descriptor digests. If a digest moves,
  stop and find out why before re-pinning (see Descriptor binding).
- **The const `D` only ever disappears behind `dispatch_ring_dim!`.** Inside a
  `dispatch_ring_dim!(ring_d, |D| { … })` closure you still have a `const D`; that is
  where every NTT/matvec/ring kernel is monomorphized. Demoting a *type* off `const D`
  never means a kernel loses it.
- When in doubt about a count or a length, prefer the **shared function**
  (`setup_geometry_at` / `setup_active_ring_elems_at`) over re-deriving inline.
- **Performance contract (Wave 6+):** demoting `const D` off storage types must not add hot-path
  shape checks or allocations. Validate once at construction or verifier boundary; use trusted
  `RingBuf` borrows in prover loops. Full rules: [`docs/runtime-ring-perf-contract.md`](../docs/runtime-ring-perf-contract.md).

### PR decomposition (review scoping)

The waves below split into reviewable PRs stacked on `quang/zk-strip-4f-retire-golden`,
so no single diff carries both the additive infrastructure and the orchestration churn:

| PR | Waves | Character | Diff |
|----|-------|-----------|------|
| **PR-spec** (#215) | — | this design record | docs only |
| **PR-infra** | 1–3 | **additions only**: `RingDimPlan` / `CommitmentRingDims` / `RingLevelContext`, `setup_geometry_at`, NTT cache types, D-free `PreparedSetup` + prefix registry, descriptor binding, **per-block geometry + validation + interim `d_a==d_b==d_d` guard**. No behavior change; old `const D` paths untouched and still in use; CI-hard | medium, isolable |
| **PR-cutover** | 4–7 | the cutover: uniform suffix loop, delete the six-bound trait lattice and `const D` stacks, demote storage (`RingBuf`), mixed-D-**per-level** fixture gate | large, mostly mechanical deletion + signature change; resists further splitting — the `const D` lattice cannot be half-removed |
| **PR-perblock-exec** | follow-up | kernels honor distinct `d_b, d_d`; non-uniform per-block fixture; remove the uniform guard | small–medium |

Rationale: **all per-block work that ships in this effort lands in PR-infra** as types +
validation + a guard, reviewable in isolation; the risky churn is confined to PR-cutover,
which changes no math (behavior-preserving through Wave 5). PR-infra can merge and bake
before PR-cutover opens. This refines the "Phases 1–3 in one PR" note in the Summary:
prefer the split unless PR-infra turns out trivially small.

### Do NOT do in this PR

- No planner `ring_d` DP search; no relaxing `expand_to_level_params`’s
  `ring_d == policy.ring_dimension` check.
- No change to envelope **sizing/generation** policy: setup is still generated at one
  `gen_ring_dim == Cfg::D`. (Wave 5 only removes the now-redundant *type-level*
  `gen_ring_dim != D` comparison; it does not change how the buffer is sized — see W5e.)
- No prefix-sized NTT caches (caches stay full-envelope per `ring_d`) — tracked as a
  follow-up; see "Deferred optimization: prefix-sized NTT caches".
- No preset-family consolidation (`fp128_d64` + `fp128_d128`).
- No cross-`ring_d` NTT sharing.

### Wave 0 — Mixed-D fixture on the LEGACY path (de-risk + record the oracle)

The current suffix path already handles `level_d != Cfg::D` (the `else`/rebuild branch).
Exercise it first to (a) prove the hand-built fixture is well-formed and (b) record the
oracle the cutover must reproduce.

- Build the mixed-D fixture (see Mixed-D fixture sketch): `fp128::D128Full` setup
  (`gen_ring_dim = 128`), hand-built `Schedule` with levels 0–1 at `D=128`, level 2+ at
  `D=64`. If no test hook exists to feed a hand-built `Schedule` to prove/verify, add a
  **test-only** entry point as the first step.
- Prove + verify + transcript replay on **current** code. Snapshot the proof bytes and
  descriptor digest as a committed test oracle.
- **Done when:** the fixture proves and verifies on unmodified code and the oracle is
  committed. **Do not delete or change any production code in this wave.**
- **Gotcha:** the setup is built at `D=128` (so the setup-time `gen_ring_dim == Cfg::D`
  check passes); `D=64` appears only as a runtime *view* of the 128 envelope
  (`128 % 64 == 0`). No sizing change is needed or allowed.

### Wave 1 — Shared setup geometry + `RingDimPlan` (`akita-types`, additions only)

- `SetupRelationShape` (small projection: `num_claims`, `num_polynomials`,
  `m_row_layout`, tier dims — **no** `eq_tau1`, **no** `RingCommitment`).
- `setup_geometry_at(level, schedule, &SetupRelationShape) -> SetupGeometry { required }`
  — the shape-only row-layout footprint (`a_end`), challenge-free.
- `setup_active_ring_elems_at(...)` returns `required`, but **fails closed**
  (`InvalidSetup`) when `required > total_ring_elements_at(ring_d)` — preserve today's
  setup-sumcheck guard; do **not** silently `min`.
- Refactor `SetupContributionPlan::prepare` to obtain `required`/endpoints from
  `setup_geometry_at` (weights layer on top, unchanged).
- `RingDimPlan`, `RingLevelContext`, `RingDimPlan::from_schedule(schedule, &AkitaSetupSeed)`
  with the validation catalog; `dim_at`, `unique_dims`, `context_at`.
- **Done when:** new code compiles (unused is fine — wire one usage into a test);
  existing setup tests still pass.
- **Verify:** `cargo test -p akita-types`.
- **Gotcha (critical):** `setup_geometry_at` must reproduce
  `SetupContributionPlan::prepare().required()` **exactly**. Add a cross-check test that,
  for every shipped generated table’s level shapes, asserts the two agree. This test is
  the safety net for Waves 4–7 — write it before relying on the function.

### Wave 2 — NTT cache types (`akita-types` + `akita-prover/kernels`, additions only)

- `NttCacheKey { ring_d, num_ring_elements }`; `NttSlotCacheAny` (D32/D64/D128/D256) with
  `From<NttSlotCache<D>>` and fallible `as_d::<const D>() -> Result<&NttSlotCache<D>, _>`
  (returns `InvalidSetup` on `ring_d` mismatch, never panics); `NttCacheMap` type alias.
- **Done when:** compiles, unused; unit tests pass.
- **Verify:** `cargo test -p akita-prover ntt_slot_cache_any`.
- **Gotcha:** `as_d::<D>()` must compare the stored variant’s degree to the requested
  `D`; the `From` impls must map each `NttSlotCache<D>` to the matching variant. Test
  both the match and the mismatch branch.

### Wave 3 — D-free prepared setup + prefix registry + descriptor (`akita-prover`, `akita-types`, `akita-config`)

Three independently-committable sub-steps; keep each green.

- **3a — D-free `CpuPreparedSetup` + NTT map.** Change `ComputeBackendSetup::PreparedSetup`
  from a `<const D>` GAT to a plain associated type; `CpuPreparedSetup<F>` holds a
  `NttCacheMap` instead of `NttSlotCache<D>`. `prepare_expanded` builds an **empty** map.
  Add `ensure_ntt_slot(&mut, key)` / `ntt_slot(&, key)`.
  - **Ripple to watch:** every `B::PreparedSetup<D>` becomes `B::PreparedSetup`,
    including `OperationCtx`’s field — even though `OperationCtx` *keeps* its `const D`
    until Wave 5. Kernel read sites switch from `prepared.ntt_shared` to
    `ntt_slot(prepared, key)?.as_d::<D>()` (the `const D` is still in scope here, so
    `as_d::<D>()` resolves). To preserve behavior, each site warms/reads the key
    `(D, total_ring_elements_at::<D>())`; `validate_digit_row_request` keeps passing
    because the cached length is the full envelope.
- **3b — D-free `SetupPrefixRegistry` (keying) + ungate offload.** Replace
  `SetupPrefixProverRegistry<F, D>` with a registry keyed on `SetupPrefixSlotId`
  (slot `commitment`/`hint` stay D-typed, reached via `id.d_setup` dispatch — see
  Phase-ordering note). Delete the `if D == SETUP_OFFLOAD_D_SETUP` gates at both call
  sites (`setup_sumcheck.rs`, `stage3.rs`); keep `SETUP_OFFLOAD_D_SETUP` for slot construction.
  - **Gotcha:** ungating is behavior-preserving because `select_setup_prefix_slot`
    returns `None` when no matching slot exists, and setup construction still populates
    slots only at `d_setup = 64`. Do not change which slots are created.
- **3c — Descriptor.** Add `AlgebraSection::for_envelope::<F,E>(gen_ring_dim)`; switch
  `bind_transcript_instance_descriptor` off `const D` to call it with
  `expanded.shared_matrix().gen_ring_dim()`.
  - **Gotcha:** since `gen_ring_dim == Cfg::D` today, the bytes are identical and the
    pinned digests **must not move**. If they do, investigate (see Descriptor binding).
- **Verify:** `cargo test -p akita-prover -p akita-verifier -p akita-types`
  (with and without `--features zk`); descriptor digest tests unchanged.

### Wave 4 — One shared geometry path on both sides (`akita-prover`, `akita-verifier`)

- Make the prover setup sumcheck and the verifier stage 3 both call
  `setup_active_ring_elems_at` for the offload-decision count; delete the two parallel
  derivations. This closes soundness gap #6.
- **Done when:** existing prove/verify tests pass; add the **prover≡verifier geometry
  cross-check** test (both produce the same `setup_active_ring_elems` per level on the
  mixed-D fixture).
- **Gotcha:** both sides must construct the *same* `SetupRelationShape`. The cross-check
  is the guard; if it fails, the two shape projections disagree — fix the projection,
  do not special-case.

### Wave 5 — Orchestration cutover (`akita-prover`, `akita-verifier`, `akita-pcs`) — largest blast radius

Five sub-steps; keep each compiling.

- **5a — Plan plumbing.** Build `RingDimPlan::from_schedule` at prove and verify entry
  (validation only; no prove-time cache warming). `prepare_setup` registers the minimum
  envelope NTT contract; lazy `ensure_ntt_slot` outside the contract logs a sizing
  diagnostic. Keep the existing per-level dispatch for now (no behavior change yet).
- **5b — Demote `OperationCtx` / `ProverComputeStack` off `const D`.** They no longer
  carry `D`. Backend methods take `ring_d` (or `&RingLevelContext`) and dispatch
  internally at the kernel boundary (`dispatch_ring_dim!` + `as_d`). Migrate call sites
  method-by-method.
- **5c — Uniform suffix loops.** Rewrite `prove_suffix` / `verify_suffix` /
  `commit_next_w` as a single loop over `plan.context_at(level)`; **delete** the
  `if level_d == D { … } else { rebuild }` branch and the empty-registry workaround.
- **5d — Delete the dead lattice.** `Suffix*ProveBackendFor`, the root-tensor siblings,
  `RECURSIVE_SUFFIX_RING_DIMENSIONS`, and the six-bound `RecursiveProveBackend`
  supertrait set.
- **5e — Public API.** `AkitaCommitmentScheme<Cfg>`, `CommitmentProver<F>`,
  `batched_prove` off `const D`; `AkitaProverSetup<F>`. Relax the setup
  `if seed.gen_ring_dim != D` checks (`api/setup.rs`, `akita-setup/src/lib.rs`) to the
  seed-internal `shared_matrix.gen_ring_dim() == seed.gen_ring_dim`. **This is removing a
  now-meaningless type comparison, not a sizing change.**
- **Done when:** all uniform-`D` tests green and byte-identical; the **Wave-0 mixed-D
  fixture re-run on the new path reproduces the recorded oracle byte-for-byte**; grep
  gate clean (`! rg 'dispatch_ring_dim_result!' crates/akita-*/src/protocol/core/suffix.rs`).
- **Gotchas:** NTT contract is registered at `prepare_setup`, not in the prove loop.
  Offload stays in the
  setup sumcheck (ungated since 3b). The mixed-D fixture is your regression oracle for
  this whole wave — run it after every sub-step that touches the suffix.

### Wave 6 — Fold storage (`akita-types`, `akita-prover`, `akita-verifier`)

- `RingBuf<F>` + `as_ring_slice::<D>()` / `as_single_ring::<D>()`.
- Demote `const D` from `PreparedFold`, `RingRelationInstance`, `PreparedFoldReplay`
  (use `RingBuf` / `RingSlice`).
- Demote `RingCommitment` / `AkitaCommitmentHint` in prefix slots (or store slot rows as
  `RingBuf`) and drop the residual `const D` from `SetupPrefixSlot`.
- **Coordinate with `protocol-field-geometry-cutover.md`:** if `PreparedFold` becomes a
  tagged enum (`SingleField` / `TensorProjection`), land the D-free `RingBuf` fields
  inside the variants here rather than fighting a parallel geometry refactor.
- **Done when:** grep audit clean for `to_vec::<D>()` / `from_vec::<D>()` on fold hot
  boundaries; all tests green.

### Wave 7 — Final gate + cleanup

- [x] Re-run the mixed-D fixture end to end on the final tree (prove + verify + replay);
      byte-identical to the Wave-0 oracle (`mixed_d_per_level_e2e`).
- [x] Prover≡verifier setup-geometry cross-check (`mixed_d_geometry_crosscheck`).
- [x] PCS public API: `AkitaCommitmentScheme<Cfg>` without redundant struct `const D`
      (`akita-pcs/src/scheme/impls.rs` per-preset macro impls; D-free
      `setup_prover`, `setup_prover_recursion`, `setup_verifier`, `commit`,
      `batched_commit`, `commit_group`, `batched_prove`, and `batched_verify`;
      `ensure_root_ring_dim` on `batched_verify`). Normal public commitment and
      claim carriers are flat; typed bridge traits are explicitly named
      `TypedCommitmentProver<F, D>` / `TypedCommitmentVerifier<F, D>`.
- [x] `switch_at_fold == 0` rejected in `mixed_d_per_level_schedule` (Bugbot); unit test
      `mixed_d_schedule_rejects_switch_at_fold_zero`.
- [x] Bounded generated mixed-D admission: compact table rows may carry a smaller
      divisor `ring_d` than the policy envelope, and invalid root-direct compact
      commit metadata fails resolution.
- [ ] Planner/catalog completion (DP `ring_d` search, generalized `D_max` envelope,
      catalog regen; deferred beyond this bounded cutover).
- [ ] Grep inventory for all deleted symbols (see Inventory).
- [ ] `docs/doc-blast-radius.json` regions; optional book stubs; `docs/compute-backends.md`.

### Planner/catalog Completion (Deferred Beyond This PR)

1. DP search over per-step `ring_d`.
2. Field-element envelope sizing with `D_max`; relax enforced `gen_ring_dim == Cfg::D` in
   generation.
3. Regenerate catalogs (`akita-schedules`, `gen_schedule_tables.rs`); evaluate preset
   family consolidation.
4. Profile mixed-D vs best uniform-D on representative workloads.

### Module touch list

| Crate | Areas |
|-------|-------|
| `akita-types` | `dispatch.rs`, `layout/`, `schedule.rs`, `instance_descriptor`, `setup_contribution.rs`, `proof/setup.rs`, `proof/setup_prefix.rs`, `proof/ring_relation.rs`, `proof/containers.rs` |
| `akita-prover` | `compute/`, `kernels/crt_ntt.rs`, `backend/`, `protocol/core/`, `protocol/sumcheck/`, `protocol/ring_switch/`, `protocol/ring_relation.rs`, `api/` |
| `akita-verifier` | `protocol/core/{suffix,fold,verify,root_fold}.rs`, `protocol/ring_switch.rs`, `stages/stage1.rs`, `stages/stage3.rs`, `slice_mle/setup_contribution/` |
| `akita-challenges` | `fold_draw.rs` (`sample_folding_challenges` per-level `ring_d`) |
| `akita-pcs` | `scheme/mod.rs`, tests, benches, `examples/profile/workload.rs` |
| `akita-setup` | setup construction, recursion prefix slot population, `gen_ring_dim` checks |
| `akita-config` | `transcript_binding.rs`, `CommitmentConfig`, `proof_optimized.rs` (Phases 1–3; field-element envelope in Phase 4); `generated_families.rs` (Phase 4) |
| `akita-planner` | `expand.rs`, `schedule_params.rs`, `catalog_identity.rs` (Phase 4) |
| `akita-schedules` | generated table modules (Phase 4 regen) |
| `profile/akita-recursion` | guest glue `AkitaCommitmentScheme` types |

Add `runtime-ring-cutover` regions to `docs/doc-blast-radius.json`.

### Risks

| Risk | Mitigation |
|------|------------|
| Wrong setup geometry (`required`) | Single shared challenge-free function + golden vectors; fail closed on `ring_view` bounds |
| Prover/verifier geometry divergence | One shared function; prover≡verifier cross-check on mixed-D fixture (not just sampled golden vectors) |
| `NttSlotCacheAny` variant ≠ dispatched `D` | Fallible `as_d::<D>()` returns `InvalidSetup`; unit-tested both branches |
| Re-coupling geometry to challenges | Golden vectors assert the count is computable with no `tau1`/`x_challenges` |
| **Uniform-D cache regression** (per-level prefix keying → many overlapping caches) | One full-envelope cache per distinct `ring_d` on the setup contract; test asserts uniform-D `prepare_setup` registers **exactly one** entry |
| Offload mistakenly sizing the NTT cache | Offload affects only the setup sumcheck's *direct* `ring_view`; NTT cache is full-envelope per `ring_d` and independent (W3b) |
| **Wave-3 GAT→assoc-type ripple** breaks the build broadly | Do the `PreparedSetup<D>` → `PreparedSetup` migration in one sub-step (3a); `OperationCtx` keeps `const D` until Wave 5; kernel sites use `ntt_slot(key).as_d::<D>()` |
| `NttSlotCacheAny` variant ≠ dispatched `D` | Fallible `as_d::<D>()` returns `InvalidSetup`; unit-tested both branches |
| Phase-1 D-free slot vs Phase-3 commitment/hint | Key on `SetupPrefixSlotId`; slot commitment/hint stay D-typed until Wave 6 |
| Cache stampede on parallel prove | N/A today (setup contract built once at `prepare_setup`); if parallel prove shares prepared state, use `OnceLock`/`DashMap` per key |
| `AlgebraSection` semantic change | No-op for current presets (`gen_ring_dim == Cfg::D`); a moving digest is a red flag to investigate |
| Phase 3 + geometry cutover conflict | Sequencing rule above; single `PreparedFold` target |
| Mixed-D envelope sizing (Phase 4) | Field-element accumulation + `D_max`; SIS audit per `(ring_d, step)` |
| Mixed-D regression | Record the Wave-0 oracle on the legacy path; require byte-identical re-run after cutover (W5/W7) |

## References

- `crates/akita-types/src/layout/flat_matrix.rs`
- `crates/akita-types/src/proof/setup.rs` (`AkitaSetupSeed`, `SetupMatrixEnvelope`)
- `crates/akita-types/src/proof/containers.rs`
- `crates/akita-types/src/proof/setup_prefix.rs`
- `crates/akita-types/src/proof/ring_relation.rs`
- `crates/akita-types/src/setup_contribution.rs`
- `crates/akita-types/src/instance_descriptor/mod.rs`
- `crates/akita-prover/src/kernels/crt_ntt.rs` (`NttSlotCache`, `build_ntt_slot`)
- `crates/akita-prover/src/backend/recursive/witness.rs`
- `crates/akita-prover/src/protocol/core/suffix.rs`
- `crates/akita-prover/src/protocol/sumcheck/setup_sumcheck.rs`
- `crates/akita-prover/src/compute/poly.rs`
- `crates/akita-prover/src/compute/cpu.rs`
- `crates/akita-prover/src/api/setup.rs` (`gen_ring_dim == D` enforcement)
- `crates/akita-challenges/src/fold_draw.rs`
- `crates/akita-verifier/src/stages/stage3.rs`
- `crates/akita-planner/src/generated/expand.rs`
- `crates/akita-config/src/proof_optimized.rs` (`proof_optimized_max_setup_matrix_size`)
- `crates/akita-config/src/transcript_binding.rs`
- `crates/akita-config/src/generated_families.rs`
- `crates/akita-config/src/bin/gen_schedule_tables.rs`
- `crates/akita-config/src/proof_optimized/fp128.rs`
- `crates/akita-schedules/src/generated/`
- `profile/akita-recursion/`
- `specs/akita-polyops-cutover.md`
- `specs/protocol-field-geometry-cutover.md`
- `specs/schedule-catalog-ownership.md`
- `specs/transcript-hardening.md`
- `specs/fp16-small-field-support.md` (mixed-D deferred clause)
