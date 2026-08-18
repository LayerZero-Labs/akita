# Configuration and planning

How a preset turns into a concrete recursion schedule: the single
`CommitmentConfig` trait, the `LevelParams` it produces, and the planner that
selects (or searches for) the schedule and prices its proof size.

## CommitmentConfig and presets

The single user-facing trait that defines every per-config policy hook (algebra,
exact SIS profile, decomposition, layout, schedule, transcript bind, prove params), and
the `fp32` / `fp64` / `fp128` preset families built on it.

Both field roles live on the trait: `Field` carries committed witnesses, setup
matrices, and SIS, while `ExtField` carries public opening points, claimed
evaluations, and Fiat-Shamir challenges. The protocol geometry gates on
whether the two roles coincide (`EXT_DEGREE == 1`, all `fp128` presets) or
claims live in a proper extension (`EXT_DEGREE > 1`, `fp32` / `fp64`), never
on field bit-width. See
[Fold path and field geometry](./proving/fold-path.md).

**Implementation map**

- `crates/akita-config/src/lib.rs:54-120`.
- `crates/akita-config/src/proof_optimized/`.
- [`crates/akita-planner/README.md`](../../../crates/akita-planner/README.md) for the current planner/config boundary.

## Schedule and LevelParams

What a schedule fixes per level (decomposition depth and ring/ext degrees),
the `LevelParams` representation, and the invariants the verifier
re-derives rather than trusts.

**Implementation map**

- `crates/akita-types/src/layout/params.rs:41-97`.
- `crates/akita-types/src/schedule.rs` (`FoldStep`, `TerminalWitnessPlan`, `Schedule`).
- Paper §3.11 `sec:akita-planner` ("What the schedule fixes").
- Council architecture + newcomer reports (schedule invariants, level overload).

## The planner and proof size

The `Cfg`-free planner: catalog validation, on-demand compact→`LevelParams`
expansion, and the schedule-search DP fallback (verifier-reachable, so it must
reject malformed input, never panic). The feature-gated `akita-schedules` crate
owns shipped table data. The verifier-reachable proof-size formula.

**Implementation map**

- [`crates/akita-planner/README.md`](../../../crates/akita-planner/README.md) for the current planner overview, search model, generated tables, and supported features.
- `crates/akita-planner/src/` owns search and emission. Runtime catalog
  expansion and audit live in `crates/akita-schedules/src/`.
- `crates/akita-types/src/proof_size.rs` and `crates/akita-types/src/layout/proof_size.rs` (`level_proof_bytes`, planned witness sizing).
- Paper §3.11 `sec:akita-planner` (objective/constraints, the dynamic program, generated schedules).
- `crates/akita-planner/src/generated_families.rs`,
  `crates/akita-schedules/src/generated/`, and
  `crates/akita-schedules/src/resolve.rs` (`resolve_generated_catalog_row_for_key`).
- `book/src/usage/profiling.md` and `.github/workflows/profile-bench.yml`.

### Selective L2 candidates

The coefficient `L∞` route remains available at every fold. A production
preset may also enable the typed `L2` response model. Every shipped
fp32, fp64, and fp128 dense and one-hot family enables it. This includes each
generated multi-chunk and recursive companion.

The planner always retains the universal L infinity candidate. When typed
moments are available, it also evaluates the modeled L infinity depth. From
level 3 onward, an enabled family evaluates the same canonical block split with
an L2 A matrix for response bases 8 and above. The planner keeps an eligible L2
alternative only when it lowers the A rank. Basis 8 uses the same fused norm and
range-image leaf as larger bases, with its class-indexed source prepared lazily
because it has no product-stage prefix.

The planner estimates the squared norm of the actual recursive witness. It
applies the following rules.

* A balanced signed-digit root uses the deterministic maximum squared digit
  energy for every coefficient, summed over the digit planes its declared source
  bound needs. A bounded source stops short of the field width, so its final
  plane is charged only the range the bound leaves rather than a full
  `log_basis`. A one-hot root maximizes the physical squared energy over every
  hot position allowed by its policy-owned chunk contract. This follows the
  tensor projection kernel and includes coefficient collisions.
* The Z part uses the centered residues of a rounded normal variable. Its
  variance comes from the previous source energy and the challenge energy.
* The E, T, and R parts use the centered field digit moment for every live
  scalar. The final digit plane uses its actual remaining field width.
* Negative binary compression contributes one half unit of expected energy per
  coefficient.
* Extension tensor packing multiplies the logical energy by `(2K - 1) / K`,
  where `K` is the extension degree.

The planner rounds each source estimate upward while retaining seven leading
bits. This adds less than `1/64` relative error and keeps the suffix search
small. It then multiplies the source estimate by the challenge squared energy,
a 1.03 model envelope, and a `40/39` response allowance. If the model envelope
bounds the conditional mean, Markov's inequality gives at least `1/40`
acceptance probability on each independent attempt. The protocol permits 4096
attempts, so the resulting exhaustion bound for one response is below
`2^-149`.

The 1.03 factor covers approximations in the normal, field digit, challenge
covariance, and finite mixing models. It is an empirical completeness margin,
not a soundness claim. Response-model diagnostics measure exact source and
response energy in complete production proofs. The benchmark parser joins each
measurement to the planned fold in the same run, rejects a successful run whose
response exceeds its frozen cap, and records cap utilization and nonce attempts
for every L2 fold. Historical measurements are evidence, not compiled unit
test constants.

The field digit model is exact for uniform power of two residues, apart from
the negligible pseudo-Mersenne boundary. Recursive setup values can retain
correlation. This usually lowers their E, T, and R energy, so the model is
conservative. Separate component validation found at most 2.24 percent
unfavorable error.

The suffix comparison includes the norm proof, A payload, next witness, later
folds, and terminal response. A smaller A rank can reduce the next witness
enough to remove a fold, but the planner keeps `L∞` when the extra norm proof
costs more than the suffix saves.

This model affects completeness and schedule selection only. Once the planner
selects a route, its concrete cap is frozen into the generated schedule. The
prover rejection samples against that cap and the verifier enforces the same
value. The SIS calculation therefore still uses the public accepted cap and
does not trust the statistical model.

If the typed model is disabled, the geometry is ineligible, no Euclidean SIS
row exists, or the L2 route does not lower the A rank, the planner keeps the L
infinity candidate.
Runtime expansion never reruns the model. It only checks the frozen schedule.

A clear terminal L2 candidate has no recursive norm proof. The verifier checks
the decoded response norm directly. The planner may use the certified energy
to estimate a smaller Golomb payload for candidate comparison. The scheduled
Golomb byte cap and the payload grind remain unchanged.

Generated schedule identity includes the cap policy and the separate L2 table
digest. Runtime expansion derives the route, cap, proof shape, and A rank from
that identity. A mismatch between the preset policy and generated catalog is
an error.

Source type is not part of runtime schedule identity. Dense and one-hot
presets own different offline policies and generated catalogs, but equivalent
polynomial groups have the same runtime geometry. In particular, one-hot chunk
size is an input to `UnitOneHotFoldPolicy`; it is not serialized in a
commitment, proof, opening layout, or transcript.

The committed-source *bound* is different: it **is** part of runtime schedule
identity. See [Bounded committed sources](#bounded-committed-sources).

## Bounded committed sources

`DecompositionParams::log_commit_bound` is the declared bit width of the largest
centered coefficient a commitment must represent. It is the general parameter of
the committed-source class, not a per-preset constant with two legal values:

| `log_commit_bound` | Source | Example preset |
|---|---|---|
| `1` | unit one-hot | `fp128::OneHot` |
| `1 < B < field_bits` | bounded | `fp128::DenseBounded` (`B = 65`, see `DenseBounded::LOG_COMMIT_BOUND`) |
| `field_bits` | full field | `fp128::Dense` |

### The bound is a *signed* bit width

`B` denotes the centered range `[-2^(B-1), 2^(B-1) - 1]`: one sign bit plus
`B - 1` magnitude bits. The gadget decomposition never sees a raw residue in
`[0, q)` — it sees the centered representative, and balanced digits are
themselves signed — so every bound here is a bound on signed magnitude.

The practical consequence is an easy off-by-one. A `u64` workload reaches
`u64::MAX = 2^64 - 1`, so it needs `B = 65`, not `64`; declaring `64` would cover
only `[-2^63, 2^63 - 1]` and miss half of a uniform `u64` distribution. That is
why `fp128::DenseBounded` declares 65. `DenseBounded::MAX_CENTERED_MAGNITUDE`
states the same fact without the signed-bit-width indirection.

The two endpoints of the table above do **not** follow this reading:

* **Full field** (`B == field_bits`) means "any field element", not
  `[-2^(field_bits-1), 2^(field_bits-1) - 1]`. `num_digits_for_bound` routes it
  to `compute_num_digits_field_width`, plain `ceil(field_bits / log_basis)` with
  no sign correction, because `decompose_centering_threshold` shifts the
  threshold below `q/2` when `δ · log_basis == field_bits` so the longer negative
  reach covers what the shorter positive reach cannot.
* **Unit one-hot** (`B == 1`) is a depth selector, not an interval. Its source is
  structurally `{0, 1}`; the signed reading `[-1, 0]` would exclude a hot
  position.

So the signed convention describes exactly the interior, which is also the only
region where the declaration is enforced as an acceptance interval.

### Where the bound enters

Primarily the A-role digit depth
`num_digits_inner = ceil(B / log_basis_inner)`. From there it fixes the A input
width, the SIS rank that width demands, the shared setup matrix, and the level-1
witness the whole recursion suffix inherits. A caller who knows their witness is
`u64`-valued therefore pays for 65 bits of range instead of 128.

It has one further consumer, and the two must be read together:
`response_model::bounded_field_source_moment` charges a bounded source's final
digit plane only the range the bound leaves, rather than a full `log_basis`. That
is what turns the shallower depth into smaller L2 response caps down the
recursion suffix — and it is why the declared bound has to be *enforced* rather
than merely documented (see below).

The bound does **not** change the honest-fold sizing rule. A bounded source and
a full-field source both decompose into the same balanced digit alphabet and
share `BalancedSignedDigitFoldPolicy`; the folded response is sized from one
digit plane, whose norms do not depend on the bound. `UnitOneHotFoldPolicy` is
not the `B = 1` case of a bounded policy — its gains come from source sparsity,
which a dense source does not have.

### What a bounded commitment binds

A commitment sized from `B` is binding and complete only for polynomials inside
`akita_types::sis::accepted_committed_source_bounds`, the **intersection** of two
independent constraints:

1. **Representability** — the coefficient must be recoverable from
   `num_digits_inner` balanced digits
   (`checked_balanced_digit_representable_bounds`). Outside it the decomposition
   keeps only the scheduled digits and the commitment binds a truncation.
2. **Declaration** — the coefficient must lie inside `[-2^(B-1), 2^(B-1) - 1]`,
   the range the schedule was *priced* for by the source-moment model above.

The two differ because the depth rounds up: 13 base-`2^5` digits span 65 bits, so
they represent about `±2^64` while a `B = 64` schedule is priced for `±2^63`. The
gap is geometry-dependent and can be large — 256× at the `log_basis_inner = 9`
geometry the `nv = 24` row selects. Enforcing only representability would accept
coefficients the schedule never declared, which is exactly how a too-narrow
declaration ships silently; enforcing the intersection turns that into an input
error at `commit`.

`commit` compares each source's centered reach against that intersection and
returns `AkitaError::InvalidInput` naming both the reach the data needs and the
interval the schedule accepts. A full-field schedule constrains neither side and
skips the scan entirely, so unbounded presets pay nothing.

### Identity and per-group bounds

`log_commit_bound` lives in `DecompositionParams`, which is hashed into both
`policy_digest` and `identity_digest` and serialized into the instance
descriptor. Two families differing only in their bound therefore have distinct
catalog identities and cannot resolve each other's rows, and no new wire field
is needed.

A bounded family's generated module carries a banner naming its declared bound, so
a reader of `crates/akita-schedules/src/generated/` can see why its root rows have
a shallower `num_digits_inner` without decoding `CATALOG_IDENTITY`. The banner is
emitted only for the interior of the range: a full-field family decomposes over
the whole width, and a unit one-hot family already says so in its name.

A **precommitted** group is a different case. Its frozen
`CommittedGroupProfile` records geometry and matrices but neither the source class
nor the bound its producer declared — only the consequence, `num_digits_inner`
digits at `log_basis_inner`. Nothing downstream needs the label, because a grouped
row is keyed on exact descriptor equality and the wrong producer simply fails to
resolve. So that a reader can still tell the classes apart, each precommitted
descriptor in a generated row is preceded by a comment naming its source class:

```rust
precommitted_groups: &[
    // unit one-hot: admits {0, 1}, one hot position per 256 coefficients; 1 x base-2^3 digits, span 3 bits
    GeneratedRootPrecommittedGroup { descriptor: CommittedGroupProfile { .. }, .. },
    // balanced signed digit: 14 x base-2^5 digits, span 70 bits, representable envelope about +/-2^69; the producer's declared log_commit_bound may be tighter
    GeneratedRootPrecommittedGroup { descriptor: CommittedGroupProfile { .. }, .. },
],
```

The class comes from the honest fold policy the row was planned against, not from
guessing at the digit depth.

Note what each form does and does not claim. A unit one-hot source is
structurally constrained and its chunk size is part of its class, so the comment
states the **admitted set** exactly. A balanced-digit source's admitted set is its
producer's `log_commit_bound` intersected with the digit envelope, and the
producer's declaration is precisely what a frozen descriptor does not record — so
the comment names the **representable envelope** and says the declaration may be
tighter. `DenseBounded` is the live example: 14 base-`2^5` digits span 70 bits
while the producer enforces `[-2^64, 2^64 - 1]`, 32x tighter. Printing the
envelope as an acceptance claim would overstate the admitted set in exactly the
artifact these comments exist to make auditable.

The bound is per config, and a precommitted group freezes its own
`inner_commit_matrix` — so one grouped root can open groups with different
bounds. The `bounded_dense_precommit_with_onehot_final_group` end-to-end test
covers a `fp128::DenseBounded` precommit under a `fp128::OneHot` root. Only the
shared full-width opening geometry has to agree, which is why a bounded preset
keeps `log_open_bound` at the true field width: `t̂` and `ŵ` carry genuine field
elements.

### Choosing a bound

Adaptive presets select with
`SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload`, i.e.
setup-first. The bound's return is a smaller setup matrix and a smaller
prover-side witness, **not** a smaller proof.

Shipped `fp128` rows, bounded (`B = 65`, i.e. every `u64`) against full-width
(`B = 128`):

| `nv` | `num_digits_inner` | setup field elements | level-1 witness |
|---|---|---|---|
| 14 | 26 → 14 | 131 072 → 65 536 (−50%) | 439 232 → 302 336 (−31%) |
| 24 | 26 → 8 | 2 818 048 → 2 097 152 (−26%) | 15 614 976 → 10 590 592 (−32%) |
| 26 | 19 → 10 | 5 636 096 → 4 587 520 (−19%) | 31 922 560 → 26 603 264 (−17%) |

`log_basis_inner = 5` (which `nv = 14` selects) is the only basis in
`inner_basis_range` where covering `u64` rather than signed-64 costs a digit;
`nv = 24` (`lb = 9`) and `nv = 26` (`lb = 7`) need the same depth either way.

Estimated proof size is flat to slightly better across these sizes. At small
`nv` the setup-first objective can even trade a noticeably larger proof for a
smaller setup, so review proof size per `(nv, bound)` pair before adding a key
to a bounded catalog.

**Implementation map**

- `crates/akita-types/src/config.rs` (`DecompositionParams::log_commit_bound`,
  `validate`, `has_bounded_committed_source`).
- `crates/akita-types/src/sis/decomposition_digits.rs`
  (`accepted_committed_source_bounds`, `declared_committed_source_bounds`,
  `balanced_digit_representable_bounds`,
  `checked_balanced_digit_representable_bounds`, `num_digits_inner_for_bound`).
- `crates/akita-config/src/proof_optimized/fp128.rs` (`DenseBounded`).
- `crates/akita-prover/src/api/commitment.rs` (the producer-side guard) and
  `crates/akita-prover/src/compute/poly.rs`
  (`RootCommitSource::committed_centered_reach`).
