# Spec: Runtime Ring Cutover

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-06-24 |
| Revised       | 2026-06-25 (review pass: setup-sizing + NTT-cache grounding, warm-cache redesign, phase-ordering fixes) |
| Status        | proposed |
| PR            | |
| Supersedes    | partial supersession of `specs/akita-polyops-cutover.md` (storage half); coordinate PR order with `specs/protocol-field-geometry-cutover.md` (shared `PreparedFold` / `prove_suffix` surface) |
| Superseded-by | |
| Book-chapter  | `how/architecture.md` (revise core-types / setup sections) |

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
**`FoldRingPlan`** records per-level `RingLevelContext` (ring dimension and the
setup prefix read at that level); **`PreparedSetup`** holds one NTT cache per
**distinct** ring dimension the proof uses (keyed `(ring_d, num_ring_elements)`),
warmed once at prove entry from `plan.unique_dims()`. Caches at different `ring_d`
are physically distinct transforms and are never shared (see NTT cache today); a
uniform-`D` proof keeps exactly one cache, identical to today.

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

1. Different `D` per fold is first-class in prove, verify, and prepared state.
2. Suffix orchestration does not special-case cross-D folds (no stack rebuild).
3. Fold protocol storage (`PreparedFold`, `RingRelationInstance`) does not carry
   `const D` on the struct (Phase 3); in-memory owners use `RingBuf` / `RingSlice`.
4. NTT prepared caches are one full-envelope cache per **distinct** `ring_d` (keyed
   `(ring_d, num_ring_elements)`), warmed at prove entry, never shared across `ring_d`.
   Prefix-sizing *within* a `ring_d` is a deferred optimization (see NTT cache design).
5. Infrastructure supports a future planner that optimizes `ring_d` per fold step
   within one field family.

### Invariants

**Protocol correctness**

- Fold math, ring switch, stage 1/2/3 unchanged unless listed under Wire Changes.
- Verifier no-panic contract preserved (`docs/verifier-contract.md`).
- `FoldRingPlan::dim_at(ℓ) == schedule[ℓ].params.ring_dimension` for every fold level.
- Flat buffer chunking at level `ℓ` uses `dim_at(ℓ)`; malformed lengths return
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
  Normative contracts) and **identical on prover and verifier**.

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
- **Full planner mixed-D DP search** in the first PR (infrastructure only; see
  Future: unified field-family planner).
- Replacing `CyclotomicRing<F, D>` as stack/value type inside kernels.
- GPU / Metal backend design (`specs/akita-compute-backend-metal.md`).
- Merging `fp128_d64` and `fp128_d128` preset families in the same PR as runtime
  infrastructure (follow-on after planner work).

## Evaluation

### Acceptance Criteria

**Phase 1 — infrastructure (CI-hard)**

- [ ] `setup_geometry_at` (shape-only, challenge-free; see Normative contracts) and
      `setup_active_ring_elems_at` in `akita-types`, with golden vectors.
- [ ] `FoldRingPlan`, `RingLevelContext` in `akita-types`.
- [ ] `FoldRingPlan::from_schedule` with validation catalog (see Normative contracts);
      takes `&AkitaSetupSeed` (carries `gen_ring_dim` + identity), not a new envelope
      type.
- [ ] `NttCacheKey`, `NttSlotCacheAny` (+ fallible `as_d::<D>()`), `NttCacheMap`
      (`HashMap` keyed store) with lazy `ensure_ntt_slot`.
- [ ] `CpuPreparedSetup<F>` (trait assoc type `PreparedSetup`) without `const D`;
      `prepare_expanded` builds an empty map; warm-cache at prove entry via
      `plan.unique_dims()` (one full-envelope entry per distinct `ring_d`; see
      Warm-cache policy).
- [ ] Single shared setup-geometry function consumed by **both** prover setup
      sumcheck and verifier stage 3 (replaces the two parallel derivations).
- [ ] D-free `SetupPrefixRegistry` (replaces `SetupPrefixProverRegistry<F, D>`; the
      verifier registry is already D-free); delete the `if D == SETUP_OFFLOAD_D_SETUP`
      eligibility gate and the constant. See Phase-ordering note for the slot's
      commitment/hint.
- [ ] `bind_transcript_instance_descriptor` without `const D`;
      `AlgebraSection::for_envelope` uses `gen_ring_dim`.

**Phase 2 — orchestration cutover (CI-hard)**

- [ ] `ProverComputeStack<F, B>`, `OperationCtx<F, B>` without `const D`.
- [ ] Backend traits take `RingLevelContext` (or `NttCacheKey` where NTT-only);
      internal `dispatch_ring_dim!` only.
- [ ] `prove_suffix` / `verify_suffix` / `commit_next_w`: uniform loop over
      `plan.context_at(level)`; **no** stack-rebuild branch.
- [ ] `prove_fold` / `verify_fold` API takes `ring_d: usize` at boundary.
- [ ] Delete `Suffix*ProveBackendFor`, `RECURSIVE_SUFFIX_RING_DIMENSIONS`,
      six-bound `RecursiveProveBackend` supertrait lattice.
- [ ] `AkitaCommitmentScheme<Cfg>`, `CommitmentProver<F>`, `batched_prove` without
      `const D` on scheme / trait (see Public API cutover).
- [ ] Grep gate: no `dispatch_ring_dim_result!` in `protocol/core/suffix.rs` or
      verifier `protocol/core/suffix.rs`.

**Phase 2 exit (manual review)**

- Uniform-D shipped presets prove and verify with byte-identical descriptors to
  pre-cutover (no pinned-digest change expected since `gen_ring_dim == Cfg::D`; if any
  digest moves, investigate — it means a preset's envelope diverged from `Cfg::D`).
- Suffix cold path removed; perf neutral or better on profile preset
  `onehot_fp128_d64:32:1` (advisory, not CI gate).

**Phase 3 — fold storage cutover (CI-hard)**

- [ ] `PreparedFold`, `RingRelationInstance`, verifier `PreparedFoldReplay` without
      `const D` on struct (use `RingBuf` / `RingSlice` internally).
- [ ] `RingBuf<F>` in-memory alias over compact storage; `as_ring_slice::<D>()` API
      (same semantics as today's `FlatRingVec`); wire `FlatRingVec` encoding unchanged.
- [ ] No `to_vec::<D>()` / `from_vec::<D>()` on fold hot boundaries (grep audit).
- [ ] Hand-built mixed-D `Schedule` fixture proves and verifies (e.g. levels 0–1 at
      D=128, level 2+ at D=64) with transcript replay **before** deleting the legacy
      suffix cold-path reference implementation.

**Correctness / perf (CI-hard where noted)**

- [ ] Proof wire bytes unchanged (pinned roundtrip on representative proofs).
- [ ] Prover≡verifier setup-geometry cross-check on the mixed-D fixture (both call the
      shared function and agree level-by-level).
- [ ] `cargo bench -p akita-pcs --bench ring_ntt` and `--bench root_kernels`: no
      regression on `dense_root_matvec_full_nv25_d32` and CRT matvec baselines
      (manual / release bench; not CI today).

**Planner (Phase 4, separate PR)**

- [ ] DP searches `ring_d` per fold step; `expand_to_level_params` accepts
      `ring_d != policy.ring_dimension` when policy allows.
- [ ] Envelope sizing accumulates in field elements with `gen_ring_dim = D_max` (see
      Setup sizing today). Relax the `gen_ring_dim == Cfg::D` enforcement in
      `api/setup.rs` / `akita-setup/src/lib.rs` to `gen_ring_dim % levelD == 0`.
- [ ] Catalog emits tables for unified field-family configs.

### Testing Strategy

**CI-hard**

- `FoldRingPlan::from_schedule` on all shipped generated tables (uniform D today).
- `setup_geometry_at` / `setup_active_ring_elems_at` golden vectors per representative
  level shape (single-tier, tiered, with/without prefix offload), pinning
  `(level shape, ring_d, required, offload?) → (active, ntt)`. These must be derivable
  **without** challenges (regression guard against re-coupling to `eq_tau1`).
- `NttCacheKey` / warm-cache unit tests: uniform-D warms exactly one entry
  `(Cfg::D, total)`; the mixed-D fixture warms exactly two; `cache_bytes()` scales with
  `num_ring_elements` and `D`.
- `NttSlotCacheAny::as_d::<D>()` returns the correct variant on match and
  `InvalidSetup` on `ring_d` mismatch (no panic).
- `RingBuf::as_ring_slice` / `FlatRingVec::as_ring_slice` roundtrip and alignment.
- Grep inventory for deleted symbols (`Suffix*ProveBackend`, suffix-level
  `dispatch_ring_dim_result!`, `RECURSIVE_SUFFIX_RING_DIMENSIONS`,
  `SETUP_OFFLOAD_D_SETUP`).
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
- **Expect:** mixed-D proofs build one full-envelope NTT cache per distinct `ring_d`,
  warmed at entry from `plan.unique_dims()`; uniform-D builds exactly one (today's work,
  relocated from `prepare_expanded`). Offload adds no cache builds.
- **Memory:** `NttCacheMap` holds one entry per distinct `ring_d`: 1 for uniform-D, 2
  for the mixed-D fixture, ≤ 4 in principle.
- **Advisory (not CI):** profile preset `onehot_fp128_d64:32:1` prove time within
  5% of pre-cutover baseline after Phase 2.

## Design

### Target architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Schedule / wire (runtime)                                        │
│  FoldRingPlan, LevelParams.ring_dimension, RingBuf, Schedule     │
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
    let required = setup_geometry_at(level, schedule, relation_shape)?.required;
    let setup_len = expanded.shared_matrix().total_ring_elements_at::</*ring_d*/>()?; // via dispatch
    Ok(required.min(setup_len))
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
    pub ring_d: usize,
    /// Shape-only count of setup ring rows the level's setup product touches
    /// (`SetupContributionPlan::required`). Drives the offload decision and the setup
    /// sumcheck's direct `ring_view`. Identical on prover and verifier.
    pub setup_active_ring_elems: usize,
}
```

The NTT cache key for a level is just `(ctx.ring_d, total_ring_elements_at::<ring_d>())`
— derivable from `ring_d` and the seed alone, with no dependence on the relation shape,
the registry, or offload. That is why warm-cache by `plan.unique_dims()` is feasible
(see Warm-cache policy).

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
**(a)** delete the `if D == SETUP_OFFLOAD_D_SETUP` gate at its two call sites
(`setup_sumcheck.rs`, `stage3.rs`) and the constant; **(b)** demote the *prover*
registry/slot off `const D` (see Phase-ordering note).

### `FoldRingPlan` and `RingLevelContext`

Central runtime authority for per-fold ring geometry, derived once from the effective
`Schedule` at prove/verify entry. **`FoldRingPlan` is a derived view**; it is not
separately digested (per-level `ring_dimension` is already bound in
`PlanSection::from_schedule` via `LevelParams::append_descriptor_bytes`).

```rust
pub struct FoldRingPlan {
    ring_dims: [usize; MAX_FOLD_LEVELS],   // validated per-level ring dims
    pub num_folds: usize,
}

impl FoldRingPlan {
    pub fn from_schedule(
        schedule: &Schedule,
        seed: &AkitaSetupSeed,             // gen_ring_dim + identity already live here
    ) -> Result<Self, AkitaError>;

    pub fn dim_at(&self, level: usize) -> Result<usize, AkitaError>;
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

`dim_at` returns `ring_dims[level]` after bounds / support checks. `context_at`
replaces today's `validate_level_dispatch`: it checks `SUPPORTED_RING_DIMS`,
`schedule[level].params.ring_dimension == ring_dims[level]`, then derives
`setup_active_ring_elems` via the shared `setup_active_ring_elems_at`. The NTT cache
key for the level is `(ctx.ring_d, total_ring_elements_at::<ring_d>())`, formed
separately at the kernel boundary — `context_at` does not compute it.

**Prove entry:** build `FoldRingPlan::from_schedule(schedule, seed)`; the verifier
builds the same plan from the same schedule + seed. Per-level contexts are computed
**inside** the loop (memoize per level if the same context is needed twice).

#### `FoldRingPlan::from_schedule` validation

`from_schedule` returns `InvalidSetup` (never panic) on:

| Check | Rule |
|-------|------|
| Fold count | `schedule_num_fold_levels(schedule) ≤ MAX_FOLD_LEVELS` |
| Supported D | every `params.ring_dimension ∈ SUPPORTED_RING_DIMS` |
| Envelope divisibility | `seed.gen_ring_dim % ring_d == 0` at every level |
| Schedule consistency | `ring_dims[ℓ] == schedule[ℓ].params.ring_dimension` |
| Witness chain | `current_w_len % ring_d == 0` at each level; terminal shape valid |
| Cross-level lengths | `next_w_len` consistent with digit layout at `ring_d` and the next level's `ring_d` when they differ |
| Root layout | `ring_dims[0]` matches committed polynomial ring layout (validated at PCS commit entry) |

Active-prefix feasibility (`setup_active_ring_elems ≤ envelope total at ring_d`) is a
**runtime** check inside `context_at` (it depends on the live relation shape), not a
`from_schedule` check; offload-slot coverage is checked where offload is decided (setup
sumcheck / stage 3). Mixed-D schedules that bypass generated expansion (hand-built
fixtures, the mixed-D fixture) must still satisfy this catalog.

### Warm-cache policy

Because the NTT cache key is `(ring_d, total_ring_elements_at::<ring_d>())` — a function
of `ring_d` and the seed alone, with no dependence on the relation shape, challenges, or
registry — the full set of keys **is** known before any fold runs: it is exactly
`plan.unique_dims()`, which spans the **whole** plan, root fold (level 0) included. So
warm the cache once at prove entry — **before the root fold, not just before the suffix
loop** — while you still hold `&mut` on the prepared setup, then run everything against
`&prepared` with **no interior mutability** in the hot path:

```rust
// Prove entry, after building the plan, BEFORE the root fold and the suffix loop:
for d in plan.unique_dims() {            // spans levels 0..num_folds, root included
    // total_ring_elements_at_dyn(d) is the runtime sibling of the const-generic
    // total_ring_elements_at::<D>() — pure arithmetic (total_ring_elements * gen_ring_dim / d
    // with the gen_ring_dim % d == 0 check); add it in Wave 2.
    let n = expanded.shared_matrix().total_ring_elements_at_dyn(d)?;
    prepared.ensure_ntt_slot(NttCacheKey { ring_d: d, num_ring_elements: n })?;  // &mut prepared
}
// freeze: hand &prepared to the root fold AND the suffix loop
root_fold(..., &prepared, plan.context_at(0, ...)?, ...)?;   // level 0, reads ntt_slot via &prepared
for level in 1..plan.num_folds {
    let ctx = plan.context_at(level, schedule, expanded, &relation_shape)?;
    let prepared_fold = prepare_suffix(stack, ..., &ctx, ...)?;   // reads ntt_slot via &prepared
    prove_fold(..., stack, &ctx, prepared_fold, ...)?;
}
```

**Root fold and the standalone commit path are not special-cased — they are covered, but
only if you warm by `unique_dims()` and place the warm before the root fold.** Two traps
to avoid (both flagged in review):

- The suffix loop starts at level **1**; the **root fold at level 0** runs earlier,
  outside `prove_suffix`, and also reads the prepared-setup NTT cache. Warming by
  `plan.unique_dims()` (which includes `dim_at(0)`) before the root fold covers it.
  Warming *inside* a `level >= 1` loop, or lazily on first suffix use, would leave the
  root fold without a built entry — do not do that.
- The standalone `commit` entry point (committing the root polynomial, which runs before
  any `FoldRingPlan` exists) must likewise `ensure_ntt_slot` its root-layout entry
  `(Cfg::D, total)` at its own entry, since `prepare_expanded` no longer builds it
  eagerly.

For a uniform-`D` proof this warms exactly one cache `(Cfg::D, total)` — byte-for-byte
the work `prepare_expanded` does eagerly today, just relocated and keyed (and the commit
path warms the same single entry). For the mixed-D fixture it warms exactly two:
`(128, total_128)` and `(64, total_64)`.

(`ensure_ntt_slot` is still idempotent and safe to call lazily if a future code path
prefers it; warming up front is chosen because it avoids any interior-mutability or
`OnceLock` machinery in the single-threaded hot loop and makes the cache contents a
pure function of the plan. Prefix-sizing a cache below the full envelope is a deferred
optimization, orthogonal to this policy.)

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
    /// Build the cache for `key` if absent. Called at prove entry with `&mut`
    /// (warm-cache); idempotent.
    fn ensure_ntt_slot(prepared: &mut Self::PreparedSetup, key: NttCacheKey) -> Result<(), AkitaError>;
    /// Read a previously-warmed slot. `InvalidSetup` if the key was never warmed.
    fn ntt_slot<'a>(prepared: &'a Self::PreparedSetup, key: NttCacheKey)
        -> Result<&'a NttSlotCacheAny, AkitaError>;
}
```

`prepare_expanded` builds an **empty** map (no eager NTT). The single warm-cache loop at
prove entry (see Warm-cache policy) populates one entry per distinct `ring_d`, each at
the full envelope `num_ring_elements = total_ring_elements_at::<ring_d>()`.

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
`ensure_ntt_slot` at prove entry (`&mut`). Cardinality = number of distinct `ring_d` in
the plan: **1 for uniform-`D`**, ≤ 4 in principle. Offload adds no entries (it never
touches the cache). A `HashMap` (rather than a fixed 4-slot table) is used only so the
key type stays forward-compatible with a future prefix-sized cache; today there is
exactly one entry per `ring_d`.

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

- Build `FoldRingPlan` from schedule + seed (same as prover).
- `verify_fold(..., ctx: &RingLevelContext, ...)` without the suffix-level dispatch
  macro. `RingLevelContext` is identical to the prover's (no NTT field).
- `context_at` is the same call as the prover's; the verifier simply never warms an NTT
  cache. The offload active/inactive decision (for transcript absorption) happens in
  stage 3 via the shared `select_setup_prefix_slot`, exactly as on the prover.
- Flat proof decode: `proof.v().as_ring_slice::<D>()` where `D = ctx.ring_d`.
- `validate_level_dispatch` replaced by `FoldRingPlan::context_at`.

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
| `AkitaCommitmentScheme<const D, Cfg>` | `AkitaCommitmentScheme<Cfg>` (Phase 2) |
| `AkitaProverSetup<F, const D>` | `AkitaProverSetup<F>` (envelope degree read from `seed.gen_ring_dim`; relax the setup `gen_ring_dim != D` checks to compare against the seed itself) |
| `batched_prove` (D from scheme struct) | `batched_prove` builds `FoldRingPlan` from resolved schedule + seed |
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
- `SetupPrefixProverRegistry<F, const D>`, `SETUP_OFFLOAD_D_SETUP` and its eligibility
  gate; the suffix cold-path empty `SetupPrefixProverRegistry::new()` workaround

Retain (already D-free / already shared — do not rewrite):

- `dispatch_ring_dim!` / `dispatch_ring_dim_result!` in `akita-types` for kernel entry
- `select_setup_prefix_slot`, `setup_prefix_level_params`, `SetupPrefixSlotId`
  (already take `d_setup`), `SetupPrefixVerifierRegistry<F>` (already D-free)
- `NttSlotCache<const D>`, `CyclotomicRing<F, D>`, all SIMD kernels
- `validate_level_dispatch` semantics (subsumed by `FoldRingPlan::context_at`)

### Wire changes

**Default: no wire format change.** `AkitaLevelProof` already stores compact
`FlatRingVec`; per-level `D` is implied by the schedule digest in `PlanSection`.

**Descriptor:** `AlgebraSection.ring_dimension_d` semantics become envelope
`gen_ring_dim` (no byte change today; see Descriptor binding). No new `PlanSection`
field for `FoldRingPlan`. Document the relabeling in `specs/transcript-hardening.md`.

### Public API cutover

Phased migration for PCS and compute surfaces (full cutover, no shims):

| Phase | `AkitaCommitmentScheme` | `CommitmentProver` / `Verifier` | `RingCommitment` / hints | `PreparedSetup` | Caller-visible `D` |
|-------|-------------------------|----------------------------------|--------------------------|-----------------|-------------------|
| 1 | `<const D, Cfg>` unchanged | `<F, D>` unchanged | `<F, D>` | D-free internal on `CpuBackend` | type param + schedule |
| 2 | `<Cfg>` | `<F>` | `<F, D>` root only | D-free | root: `Cfg::D`; prove: `FoldRingPlan` |
| 3 | `<Cfg>` | `<F>` | `RingBuf` / D-free where applicable | D-free | `FoldRingPlan` only at PCS boundary |

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
| F. Per-level prefix-keyed NTT cache (key per `(ring_d, level prefix)`) | Rejected: kernels index a prefix of one full slot, so per-level keys would build several overlapping caches and **regress** uniform-D from one cache to many. Adopted instead: one full-envelope cache per distinct `ring_d`, warmed at entry from `plan.unique_dims()` |
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
| `CommitmentConfig::D` names the preset | `CommitmentConfig` names field + decomposition; `FoldRingPlan` names D ladder |

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

- `book/src/how/architecture.md`: `FoldRingPlan`, `RingBuf`, runtime `D` vs
  `gen_ring_dim`, setup-sizing model, NTT-cache-per-`(D, prefix)`, diagram of three
  layers.
- `book/src/how/proving/fold-path.md` (stub): schedule-driven ring dimension.
- `docs/compute-backends.md`: `PreparedSetup` without `const D`; `ensure_ntt_slot`.
- `docs/doc-blast-radius.json`: add regions for this spec.
- Cross-link from `protocol-field-geometry-cutover.md` (coordinate `PreparedFold`
  target shape and `prove_suffix` PR order; see Execution).

### Sequencing with `protocol-field-geometry-cutover.md`

Both specs touch `protocol/core/{fold,suffix,prove}.rs` and `compute/poly.rs`.

**Rule:** Land runtime ring **Phase 1** (`FoldRingPlan`, keyed NTT, shared setup
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

### Do NOT do in this PR

- No planner `ring_d` DP search; no relaxing `expand_to_level_params`’s
  `ring_d == policy.ring_dimension` check.
- No change to envelope **sizing/generation** policy: setup is still generated at one
  `gen_ring_dim == Cfg::D`. (Wave 5 only removes the now-redundant *type-level*
  `gen_ring_dim != D` comparison; it does not change how the buffer is sized — see W5e.)
- No prefix-sized NTT caches (caches stay full-envelope per `ring_d`).
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

### Wave 1 — Shared setup geometry + `FoldRingPlan` (`akita-types`, additions only)

- `SetupRelationShape` (small projection: `num_claims`, `num_polynomials`,
  `m_row_layout`, tier dims — **no** `eq_tau1`, **no** `RingCommitment`).
- `setup_geometry_at(level, schedule, &SetupRelationShape) -> SetupGeometry { required }`
  — the shape-only row-layout footprint (`a_end`), challenge-free.
- `setup_active_ring_elems_at(...) = min(required, total_ring_elements_at::<ring_d>())`.
- Refactor `SetupContributionPlan::prepare` to obtain `required`/endpoints from
  `setup_geometry_at` (weights layer on top, unchanged).
- `FoldRingPlan`, `RingLevelContext`, `FoldRingPlan::from_schedule(schedule, &AkitaSetupSeed)`
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
  Phase-ordering note). Delete the `if D == SETUP_OFFLOAD_D_SETUP` gate at both call
  sites (`setup_sumcheck.rs`, `stage3.rs`) and the constant.
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

- **5a — Plan plumbing.** Build `FoldRingPlan::from_schedule` at prove and verify entry;
  warm the NTT cache via `plan.unique_dims()` (`&mut prepared`, then freeze) **before the
  root fold (level 0), not just before the suffix loop**; warm the root-layout entry in
  the standalone `commit` path too (see Warm-cache policy — both are review-flagged
  traps). Keep the existing per-level dispatch for now (no behavior change yet).
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
- **Gotchas:** warm-cache stays at entry (`&mut`), not in the loop. Offload stays in the
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

- Re-run the mixed-D fixture end to end on the final tree (prove + verify + replay);
  confirm byte-identical to the Wave-0 oracle.
- Confirm the prover≡verifier geometry cross-check passes.
- Grep inventory for all deleted symbols (see Inventory).
- `docs/doc-blast-radius.json` regions; optional book stubs; `docs/compute-backends.md`.

### Phase 4 — Planner (separate, follow-on PR; out of scope here)

1. DP search over per-step `ring_d`; relax `expand_to_level_params` policy check.
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
| **Uniform-D cache regression** (per-level prefix keying → many overlapping caches) | One full-envelope cache per distinct `ring_d`; warm at entry from `unique_dims()`; test asserts uniform-D warms **exactly one** entry |
| Offload mistakenly sizing the NTT cache | Offload affects only the setup sumcheck's *direct* `ring_view`; NTT cache is full-envelope per `ring_d` and independent (W3b) |
| **Wave-3 GAT→assoc-type ripple** breaks the build broadly | Do the `PreparedSetup<D>` → `PreparedSetup` migration in one sub-step (3a); `OperationCtx` keeps `const D` until Wave 5; kernel sites use `ntt_slot(key).as_d::<D>()` |
| `NttSlotCacheAny` variant ≠ dispatched `D` | Fallible `as_d::<D>()` returns `InvalidSetup`; unit-tested both branches |
| Phase-1 D-free slot vs Phase-3 commitment/hint | Key on `SetupPrefixSlotId`; slot commitment/hint stay D-typed until Wave 6 |
| Cache stampede on parallel prove | N/A today (warm at entry with `&mut`, then immutable loop); if parallel prove ships, use `OnceLock`/`DashMap` per key |
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
