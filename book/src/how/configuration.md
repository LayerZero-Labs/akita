# Configuration and planning

This chapter explains how a preset becomes a concrete recursion schedule. It
covers the `CommitmentConfig` trait, the fold parameters stored in a schedule,
and the planner that searches for schedules and prices their proof size offline.

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

`CommitmentConfig` selects either a uniform schedule mode or bounded adaptive
A, B, and D domains. It does not carry a separate default ring dimension.
The resolved schedule owns every dimension used by setup preparation,
commitment, proving, and verification. Setup-prefix slots also record the
dimension selected by the fold that consumes the prefix.

**Implementation map**

- `crates/akita-config/src/lib.rs:54-120`.
- `crates/akita-config/src/proof_optimized/`.
- [`crates/akita-planner/README.md`](../../../crates/akita-planner/README.md) for the current planner/config boundary.

### Bounded committed sources

The `fp128::DenseBounded` preset is for dense polynomials whose centered
coefficients fit within a declared signed width of 65 bits. This range is
`[-2^64, 2^64 - 1]`, so it contains every `u64`. The narrower declaration lets
the generated schedule use the actual source range instead of charging every
coefficient for the full 128-bit field width. Commitment rejects a source that
does not fit the declared interval.

This preset uses the `schedules-fp128-dense-bounded` catalog. That catalog is
not part of `schedules-default`; applications must enable it explicitly.

## Schedule and fold parameters

A `FoldSchedule` stores one root `FoldParams`, zero or more recursive
`FoldParams`, and one `TerminalFoldParams`. Each nonterminal `FoldParams` points
to one `CommittedGroupParams`. That value owns the fold's groups, the shared D
matrix, payload mode, ring-relation mode, source encoding, and witness chunk
layout.

The groups are stored in protocol order. An incoming setup prefix comes first,
then ordinary precommitted groups, and the fold's own new group comes last.
Each `GroupOpenPhaseParams` holds a frozen `GroupCommitPhaseParams` and the
opening plan chosen by the consuming fold. A setup prefix is the group whose
`setup_natural_len` is set. There is no second prefix identity or separate
producer choice to keep in sync.

The fold owns one `open_matrix` for the complete D product. Each group owns its
opening basis and digit depth through `GroupOpeningPlan`. The verifier checks
this ownership, the group order, matrix dimensions, decomposition depths, and
witness lengths before it uses the schedule.

**Implementation map**

- `crates/akita-types/src/layout/params.rs` defines `CommittedGroupParams`.
- `crates/akita-types/src/schedule/profiles.rs` defines
  `GroupCommitPhaseParams`.
- `crates/akita-types/src/layout/params/precommitted.rs` defines
  `GroupOpenPhaseParams` and `GroupOpeningPlan`.
- `crates/akita-types/src/schedule.rs` defines `FoldParams`,
  `TerminalFoldParams`, and `FoldSchedule`.
- `crates/akita-schedules/src/resolve.rs` validates generated rows before the
  prover or verifier uses them.

## The planner and proof size

Normal planner search is `Cfg`-free. The optional `catalog-gen` feature enables
`akita-config`, so table-emission binaries can name concrete
`CommitmentConfig` presets. The feature-gated `akita-schedules` crate owns
shipped table data and runtime expansion into `FoldSchedule` and its committed
group parameters. Runtime proving and verification resolve an enabled
generated row and never run planner search. Shared proof-size formulas remain
verifier-reachable and reject malformed input with an error rather than
panicking.

**Implementation map**

- [`crates/akita-planner/README.md`](../../../crates/akita-planner/README.md) for the current planner overview, search model, generated tables, and supported features.
- `crates/akita-planner/src/` owns search and emission. Runtime catalog
  expansion and audit live in `crates/akita-schedules/src/`.
- `crates/akita-types/src/proof_size.rs` and `crates/akita-types/src/layout/proof_size.rs` (`level_proof_bytes`, planned witness sizing).
- `crates/akita-planner/src/generated_families.rs`,
  `crates/akita-schedules/src/generated/`, and
  `crates/akita-schedules/src/resolve.rs` (`resolve_generated_catalog_row_for_key`).
- `book/src/usage/profiling.md` and `.github/workflows/profile-bench.yml`.

### Recursive setup catalogs

Ordinary configuration catalogs use direct setup evaluation. The supported
`RecursiveCommitmentConfig<Cfg>` adapters select separate catalogs that may
carry a committed setup prefix into the next fold. Keeping the catalogs
separate prevents an ordinary verifier from accepting a recursive setup shape
under a direct configuration.

The planner prices the Stage 3 proof and the later prefix opening as part of the
complete suffix. Generated rows record every selected prefix edge. Runtime
expansion checks those edges and never reruns the search. See
[Setup offloading](./setup-offloading.md) for the selection rules, setup
artifacts, and recursive claim flow.

### Ring-relation cutover

Every nonterminal fold selects one `RingRelationMode`. `QuotientLift` adds
explicit quotient witness spans and proves exact lifted polynomial identities.
`ReducedEvaluation` proves the same native-ring equations after evaluation at
the transcript challenge, using signed-wrap residue kernels instead of quotient
witnesses. The choice is independent of raw versus compressed payload and of
the `L∞` versus `L2` response-security route.

The currently admitted reduced mode is deliberately narrower than its
algebraic definition. Levels 0 and 1 use quotient lifting. From level 2 onward,
the planner may choose one monotone reduced-evaluation suffix, but only for
`EvaluationTrace` openings with direct setup contribution. A reduced fold
cannot consume an incoming setup prefix, defer the A/B/D setup product to Stage
3, or switch back to quotient lifting later. These are product-scope admission
rules, not claims that the excluded combinations are mathematically
impossible. Runtime schedule validation checks them before proving or
transcript replay.

The suffix search prices both realizations using their exact witness layouts,
proof sizes, setup requirements, response models, and successor states. A
generated row records the selected mode at every nonterminal level. The mode
is part of the canonical plan descriptor and transcript preamble; it is not a
proof-supplied negotiation field and there is no verifier fallback.

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
  energy for every coefficient, summed over the digit planes its declared
  source bound needs. A bounded source stops short of the field width, so its
  final plane is charged only the range the bound leaves rather than a full
  `log_basis`. A one-hot root uses the canonical coefficient table: each
  policy-owned source chunk contributes at most one coefficient of magnitude
  one, distinct chunks occupy distinct coefficients, and the peak coefficient
  square is one.
* The Z part uses the centered residues of a rounded normal variable. Its
  variance comes from the previous source energy and the challenge energy.
* The E and T parts, plus the R quotient part when present, use the centered
  field digit moment for every live scalar. A reduced-evaluation fold has no R
  quotient span. The final digit plane uses its actual remaining field width.
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

### Subring coefficient packing candidates

At absolute fold levels 0 and 1, the planner searches the A ring dimension and
the challenge subring dimension together. For each pair it derives the packing
factor from `d_A = k h s`. It rejects unsupported challenge families and
invalid D divisibility before it constructs matrices.

The planner does not minimize `s`, `h`, or `d_A` directly. It keeps the same
complete-schedule objective as other candidates. Depending on the catalog
policy, the numeric prefix is either proof payload then total setup, or first
direct-setup capacity then proof payload then total setup. Exact numeric ties
prefer the smaller root output witness before the canonical descriptor breaks
the final tie. This allows a larger A ring to win when its lower module rank
reduces the complete setup or proof suffix.

Nonterminal folds at levels 0 and 1 require coefficient packing. A state with
no complete packing assignment is unsupported. Later nonterminal folds and the
terminal use evaluation trace. The terminal seed is priced separately, so a
zero EOR packing candidate cannot be mistaken for a terminal opening.

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
