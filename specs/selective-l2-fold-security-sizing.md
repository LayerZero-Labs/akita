# Spec: Selective L2 Fold Security Sizing

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-08-06 |
| Status        | active |
| PR            | [#369](https://github.com/LayerZero-Labs/akita/pull/369) |
| Supersedes    | The physical A role embedding factor in `archive/2026-Q3/weak-binding-norm-fix.md` |
| Superseded-by | |
| Book-chapter  | book/src/how/security.md |

## Summary

Akita currently sizes every committed A matrix from a coefficient L infinity
bound on the folded response. This remains the default. This change adds a
second candidate for selected later folds. That candidate proves a bound on the
squared L2 norm of the physical folded response and sizes only that level's A
matrix with the quantum ADPS16 Euclidean SIS table. A level that has no L2 proof
continues to use the current L infinity table.

This change also removes an unrelated factor of two from physical A role
security sizing in the small field profiles. The folding challenge and folded
response are already expressed as physical ring coefficients when the weak
binding argument uses them. Applying the Hachi logical to ring conversion at
that point converts neither value. It counts a conversion that has already
happened.

The purpose of the selective L2 route is to reduce the A rank at later folds
when the folded response has low total energy. Lowering A rank also shrinks the
T witness that the next fold must carry. The planner must therefore compare
complete suffixes, including any change in the number of folds, rather than
compare one level in isolation.

## Intent

### Goal

Add a sound optional L2 security route for selected nonterminal folds while
keeping the current L infinity route available at every fold.

### Per level rule

The planner constructs distinct candidates. It does not attach an L2 claim to
every level and it does not choose a norm after the proof has been produced.

| Candidate | Proof obligation | A role security table |
|-----------|------------------|-----------------------|
| L infinity | Existing digit range proof | Quantum ADPS16 coefficient L infinity table |
| L2 | Existing digit range proof and the new physical norm proof | Quantum ADPS16 Euclidean table |

Every planner state has an L infinity candidate. An L2 candidate exists only
when the planner policy supplies a public squared norm cap for that candidate.
If the planner selects the L infinity candidate, the proof contains no L2
claim and pays no L2 proof cost.

The terminal response has no Stage 1 proof. This change does not add an L2
route to the terminal response. A direct check on the clear terminal witness is
a separate change.

### Coordinate system

Let

```text
R = Z[X] / (X^D + 1)
```

with centered integer coefficients. All security bounds in this spec use the
physical coefficient vector obtained by flattening every ring row, chunk, and
coefficient position in the A role response.

For one raw fold challenge and response, define

```text
kappa_1 = maximum physical coefficient L1 norm of c
Z_inf   = accepted physical coefficient L infinity bound on z
S       = accepted complete physical squared L2 norm of z
```

The complete norm is

```text
S = sum over every physical A response coefficient of z_i^2.
```

It is not a per row norm and it is not a per matrix column norm.

The sparse challenge type already represents a polynomial in the physical
base field ring. Its `l1_norm()` already counts physical ring coefficients.
The folded response is centered and decomposed into the physical Z digit
planes before Stage 1 checks those planes.

### Security invariants

1. A level without an L2 proof must use L infinity A role sizing.
2. An L2 candidate must bind its public norm cap into the schedule and proof
   shape.
3. The verifier must prove the norm of the same physical Z coefficients that
   the A role Module SIS reduction uses.
4. The planner, prover, verifier, schedule audit, and proof size code must use
   one shared definition of the physical response domain.
5. The physical A role collision formula must not apply a Hachi embedding
   factor.
6. The Euclidean SIS estimator must receive the norm of the complete scalar
   collision vector exactly once.
7. The existing digit range proof remains present on an L2 candidate.
8. A proof must not move between L infinity and L2 schedules through a shape or
   transcript ambiguity.
9. Verifier reachable code must reject malformed norm claims and shapes with
   `AkitaError` or `SerializationError`. It must not panic.

### Non goals

This change does not add the following features.

* It does not replace L infinity sizing at every fold.
* It does not add an L2 check to the root or early folds.
* It does not use a Gaussian assumption as part of security.
* It does not use an uncertified operator norm cap of 17.
* It does not add operator norm rejection above D128 without a separate
  accepted support certificate.
* It does not force terminal rank three.
* It does not remove the digit range proof.
* It does not add zero knowledge machinery, four square slack, carry witnesses,
  or an inequality proof inside the field.

## Security argument

### The extracted A collision

The weak binding extractor compares two accepted openings. A raw challenge in
either opening has physical L1 norm at most `kappa_1`. Their difference has L1
norm at most `2 * kappa_1`.

Likewise, if each raw response has physical L infinity norm at most `Z_inf`, a
response difference has L infinity norm at most `2 * Z_inf`. If each raw
response has squared L2 norm at most `S`, a response difference has L2 norm at
most `2 * sqrt(S)`.

Clearing the two weak opening denominators produces two negacyclic products.
For physical ring coefficient vectors, Young's convolution inequalities give

```text
||a * b||_inf <= ||a||_1 * ||b||_inf
||a * b||_2   <= ||a||_1 * ||b||_2.
```

Each product has one challenge difference and one response difference. The
sum has two products. This gives the complete raw radius bounds

```text
C_inf     = 8 * kappa_1 * Z_inf
C_2_sq    = 64 * kappa_1^2 * S.
```

The three factors of two have separate sources. One comes from the challenge
difference. One comes from the response difference. One comes from adding the
two products that clear the two weak opening denominators.

The L infinity candidate rounds `C_inf` upward in the quantum ADPS16
coefficient L infinity table. The L2 candidate rounds `C_2_sq` upward in the
quantum ADPS16 Euclidean table.

Both formulas describe the same extracted A kernel vector. A selected schedule
needs one sound route. The planner may compare the ranks returned by the two
routes, but the selected route determines the proof obligation and table used
by the verifier's schedule audit.

### Why the Hachi factor is double counted today

The Hachi map `psi` packs logical extension field coordinates into a physical
base field ring. A norm conversion factor is needed only for an argument that
starts with a bound on the logical coordinates and then derives a bound on the
packed ring coefficients.

The current committed fold security path starts from physical values.

First, the fold challenge `c` is sampled as a `SparseChallenge`. It is already
a physical polynomial in the base field ring. Its L1 norm counts the nonzero
physical coefficients and their magnitudes. There is no extension field
challenge left to pack.

Second, the folded response `z` is stored as centered physical ring
coefficients in `z_folded_centered_per_chunk`. The prover decomposes those same
coefficients into the Z digit planes. Stage 1 checks the digit alphabet, and
the verifier's accepted `Z_inf` is the integer envelope obtained by
recomposing those physical digits.

Third, the A role Module SIS kernel vector uses the same physical ring
coordinates. The weak binding product inequalities act directly on the
physical challenge and physical response.

The current `ring_subfield_norm_bound` multiplication inside A role collision
sizing therefore has no input to convert. For fp32 and fp64 it changes

```text
8 * kappa_1 * Z_inf
```

into

```text
8 * kappa_1 * 2 * Z_inf.
```

That second expression treats `Z_inf` as if it were still a logical norm before
`psi`. It is not. Stage 1 already bounds the packed coefficients. The factor of
two is therefore counted after the conversion it was meant to cover.

This conclusion does not say that `psi` has no norm cost. A completeness or
honest sizing argument may begin with a logical extension field source. Such an
argument must apply the relevant logical to physical conversion once before it
chooses a physical digit depth or physical response cap. Once the argument has
reached `z_folded_centered_per_chunk` or the reconstructed Z digit planes, it
must not apply that conversion again.

The Hachi trace identity used for extension opening checks does not change this
conclusion. That identity connects a logical opening claim to a ring trace. It
does not rescale the physical A kernel vector produced by weak binding.

### The separate Euclidean width double count

The Euclidean estimator uses the scalar SIS dimensions

```text
n = rank * D
m = width * D.
```

For the L2 route, `C_2_sq` bounds the complete scalar collision vector across
all `width` input ring rows. The estimator must therefore use

```text
length_bound = sqrt(C_2_sq).
```

Using

```text
length_bound = sqrt(width * C_2_sq)
```

counts the rows twice. Their coefficients already appear in the sum that
defines `S` and hence in `C_2_sq`. The scalar column count `m` records the width
again for the lattice dimension, which is correct. It must not also enlarge the
norm.

### Security versus honest acceptance

The scheduled cap is a public acceptance condition. Security does not require
the honest folded witness to follow a Gaussian distribution. If the verifier
accepts only responses with `S <= S_max`, then the L2 collision formula holds
for every accepted proof.

The response distribution affects completeness and prover cost. It determines
how often the prover can find an allowed Fiat Shamir nonce whose response lies
below `S_max`. The planner uses exact source-energy calibrations where they are
available and a balanced-digit second-moment model elsewhere. End-to-end
measurements set the empirical multiplier. Neither input is an assumption in
the binding proof.

## Protocol design

### Schedule representation

The schedule needs one explicit A role security route for each committed
level. The canonical type should have the following meaning.

```text
InnerCommitSecurityRoute::Linf
InnerCommitSecurityRoute::L2 {
    response_l2_sq_cap,
    norm_subclaim_shape,
}
```

The exact Rust field names may change during implementation, but there must be
one canonical route value. A collection of independent booleans is not
acceptable because it permits impossible states.

The route and all L2 shape data are part of schedule identity, descriptor
bytes, proof shape derivation, serialization context, and schedule audit. The
proof stream remains headerless. The schedule tells the decoder whether norm
claims are present and how many values to read.

### Candidate eligibility

The planner always emits an L infinity candidate. It may emit an L2 candidate
only when a calibrated one-hot profile enables the response model and the
canonical derivation supplies `S_max` and the norm proof shape for that
physical response domain.

There is no global L2 cap and no hard coded rule that every level after a fixed
index must use L2. Different field profiles and witness geometries reach useful
energy ranges at different levels. Generated schedules record the exact levels
and concrete caps that select L2.

The model starts at level 3. The planner discards a candidate if it does not
lower the A rank or if its complete suffix is not smaller than an L infinity
suffix. The existing bounded nonce limit remains unchanged.

### Prover acceptance

An L infinity candidate uses the existing fold nonce rule.

An L2 candidate uses the same bounded nonce space. For each nonce, the prover
derives the fold challenge, computes the physical folded response, checks the
existing digit admission condition, and checks `S <= S_max`. The prover accepts
the first nonce that satisfies both conditions.

The verifier checks that the nonce lies within the scheduled attempt bound. It
does not trust the prover's search. It derives the selected challenge from the
nonce, verifies the physical norm proof, and checks the reconstructed integer
norm against `S_max`.

### Stage 1 norm claim

For an L2 candidate, Stage 1 receives the integer claim

```text
S = sum_x z(x)^2
```

over the canonical physical A response domain. The domain contains every live Z
coefficient exactly once and gives every padding address value zero.

The prover adds a degree two sumcheck term to the final Stage 1 substage. A
fresh transcript challenge batches it with the existing digit range term. The
range term remains equality factored. The norm term is an unweighted sum over
the physical response domain. The combined round message therefore uses the
general degree bound required by both terms.

At the final Stage 1 point, the norm term reduces to the square of one virtual
evaluation of physical `z`. Stage 2 proves that this virtual value is the
balanced basis recomposition of the committed Z digit plane evaluations at the
same physical point. The selector and basis powers come from `WitnessLayout`
and the selected level parameters. The prover cannot substitute a logical
prepacking response or omit a physical segment.

For the direct large field case, the expected wire change is one additional
extension field coefficient in each round of the final Stage 1 substage, plus
the integer norm claim and any final virtual evaluation needed by Stage 2. The
proof shape and proof size code must compute the exact count. No comment or
planner estimate may substitute for the serialized shape.

### Integer soundness in small fields

A field equation proves only a congruence. It proves an integer square sum only
when the public bounds rule out wraparound.

If the schedule derived worst case physical square sum is below the base field
modulus, the verifier may use one direct norm claim. A prover supplied claim
below the modulus is not enough by itself. The worst case value allowed by the
digit ranges must also be below the modulus.

Otherwise, write each physical response coefficient as

```text
z(x) = sum_j B^j * z_hat_j(x).
```

Then

```text
S = sum_j B^(2j) * I_jj
    + 2 * sum_{j < k} B^(j+k) * I_jk,

I_jk = sum_x z_hat_j(x) * z_hat_k(x).
```

The protocol partitions the physical address domain into public blocks. Each
block is short enough that the digit alphabet gives an absolute bound below
half the base field modulus for every `I_jk` subclaim. The verifier takes the
unique centered integer lift of each field value and reconstructs `S` with
checked integer arithmetic.

The block partition and limb pair list are schedule derived proof shape data.
The prover does not choose them. The verifier rejects an overflow, a missing
subclaim, a duplicate pair, an invalid block, or a reconstructed negative
square sum.

This construction certifies equality to the physical square sum. It does not
prove a field inequality and it does not use four square slack. The verifier
performs the final integer comparison `S <= S_max` after reconstruction.

### Stage 2 virtualization

Stage 2 already binds the final Stage 1 range image to the committed digit
witness. The L2 route adds one more virtual relation for the Z segments. It
uses the same physical address authority and digit source.

For each required virtual value, Stage 2 checks the linear recomposition

```text
z(r) = sum_j B^j * z_hat_j(r)
```

with the correct group, chunk, row, and coefficient selectors. The final norm
claim uses `z(r)^2`. Small field limb subclaims use the corresponding pairs of
digit plane evaluations.

The Stage 2 batching challenge is sampled after all Stage 1 claims have been
absorbed. This prevents a prover from choosing one false relation to cancel
another.

### Proof and transcript shape

The L2 route changes all of the following surfaces.

* `AkitaStage1Proof` and its shape need the norm claim and any small field
  subclaims selected by the schedule.
* The final Stage 1 round shape needs the additional coefficient count.
* Transcript labels must separate norm claims from range image claims.
* `FoldLevelProof` serialization remains schedule driven and headerless.
* `Valid` and deserialization checks must bound every allocation before use.
* `level_proof_bytes` and the profile reporter must count the exact serialized
  values.
* Logging transcript tests must show the same event order for prover and
  verifier.

## Planner and SIS estimator

### Quantum ADPS16 Euclidean table

The production L2 route uses the 128 bit quantum ADPS16 reduction cost model.
It does not use the older BDGL16 Euclidean profile.

The estimator work includes the following changes.

* Enable the ADPS16 quantum cost in the Euclidean path.
* Reject BDGL16 for production Euclidean table generation.
* Use `sqrt(C_2_sq)` as the scalar length bound.
* Generate a separate L2 table and digest. Do not overwrite the coefficient L
  infinity table.
* Store accepted cells and rejected successor evidence at the 128 bit boundary.
* Add golden tests for each field family and supported ring dimension.

The verifier never runs the estimator. It uses checked schedule parameters and
the generated table only.

The table generator emits a full audit CSV and hashes it into the generated
Rust table. The CSV is reproducible and is not checked in. Run the following
command to rebuild the Rust rows and the local audit file.

```sh
cargo run -p akita-sis-estimator --release --example euclidean_width_table -- --format rust-split
```

### Calibrated response model admission

The planner keeps its ordinary L infinity search at every state. A zero-length
source-energy calibration opts a family into the balanced-digit response model.
At level 3 and later, the planner evaluates the ordinary best block split under
both security routes for response bases 16 and above, and keeps the L2
alternative only when it lowers the A rank. Response basis 8 is excluded from
the L2 route because it caused deterministic stage-2 folded-oracle consistency
failures in two D64 production profiles. The underlying protocol cause has not
been generalized and fixed. Exact physical-geometry rows remain available for focused fixtures and
take precedence over the model only when the source basis, challenge ring
dimension, challenge energy, and the rest of the state key all match.
Production exact source-energy calibrations use a common 1.25 response-tail
multiplier.

For an uncalibrated state, the modeled cap is the balanced-digit second moment
`input_len * (B^2 + 2) / 12`, times the fixed challenge squared energy and a
1.75 empirical multiplier. End-to-end samples cover fp32, fp64, and fp128. The
recorded maxima retain at least 15 percent cap margin. This is a completeness
model, not a soundness assumption. The selected concrete cap is frozen into
the schedule and remains the verifier-enforced SIS input.

The existing suffix search prices the ordinary L infinity candidate and the
modeled L2 candidate. This comparison includes the different A rank, T width,
next witness length, later folds, and terminal response. The planner does not
keep extra predecessor splits merely to expose more modeled alternatives.

The final planner comparison includes the norm proof bytes, changed A payload,
changed T decomposition, changed next witness, all later folds, and the
terminal response.

### Reporting

For every shipped fp128, fp64, and fp32 profile affected by the change, the PR
must report the following values before and after the change.

* Total proof bytes.
* Number of recursive folds.
* The selected security route at every fold.
* A, B, and D ranks.
* Ring dimensions.
* Log bases and digit counts.
* The public L2 cap and observed physical norm on every L2 level.
* The norm proof bytes.
* The next witness length after every fold.
* Terminal response bytes.
* Nonce attempts and the observed failure rate for each L2 cap.

The report must separate three effects. It must show the old main result, the
result after removing only the physical Hachi double count, and the result after
adding selective L2 candidates. This prevents the PR from attributing an L
infinity correction to the new norm proof.

## Evaluation

### Acceptance criteria

* [x] A level with `InnerCommitSecurityRoute::Linf` serializes no L2 claim and
      uses only coefficient L infinity A role sizing.
* [x] A level with `InnerCommitSecurityRoute::L2` verifies the complete physical
      square sum and rejects `S > S_max`.
* [x] The prover and verifier derive the physical Z domain from one shared
      `WitnessLayout` authority.
* [x] Removing the Hachi factor changes physical A role collision sizing from
      `8 * kappa_1 * 2 * Z_inf` to `8 * kappa_1 * Z_inf` for fp32 and fp64.
* [x] Equal physical challenges and responses produce equal A role collision
      bounds across field profiles, independent of extension degree.
* [x] A separate test shows that a bound stated before `psi` still applies its
      logical to physical conversion once in honest sizing.
* [x] The Euclidean scalar mapping uses `sqrt(C_2_sq)` and never
      `sqrt(width * C_2_sq)` for a complete collision norm.
* [x] Production L2 table rows use quantum ADPS16 at 128 bits.
* [x] At each calibrated suffix state, the planner prices the ordinary L
      infinity candidate and at most one L2 alternative for its canonical
      split.
* [x] Root, early, and terminal levels carry no L2 proof.
* [x] Tampering with the norm, cap, route, subclaim, virtual evaluation, nonce,
      or proof shape causes verification to fail.
* [x] Small field tests cover positive and negative limb inner products,
      centered lifting boundaries, block boundaries, and integer overflow.
* [x] Headerless deserialization rejects oversized shapes before allocation.
* [x] Proof size accounting equals actual serialization for every supported
      field family.
* [x] Generated schedules pass audit and the report contains all required
      before and after values.
* [x] All repository CI gates and verifier no panic checks pass.

### Testing strategy

Unit tests cover the collision formulas, physical coordinate conversion, L2
table mapping, integer reconstruction, and proof shape arithmetic.

Protocol tests produce valid L infinity and L2 proofs from the same witness.
They then mutate each new transcript value and each schedule field. The
verifier must reject every mutation without panic.

Planner tests pin one case where the locally smaller A rank is not the cheapest
suffix. They also pin one case where the L2 candidate removes a fold and one
case where its proof cost makes the L infinity candidate win.

End to end tests cover fp128, fp64, and fp32. Small field tests must exercise a
configuration that uses more than one limb inner product block.

The final verification commands come from `AGENTS.md` and the current CI
workflow. Documentation changes also run `scripts/check-doc-guardrails.sh`.

### Performance

An L infinity selected schedule must have no prover, verifier, or proof size
cost from the L2 machinery.

An L2 selected large field level should add one field coefficient per final
Stage 1 round, plus its fixed claims. The implementation must report the actual
serialized count instead of assuming this estimate.

The prover may perform an extra physical square sum while testing a nonce. It
should compute that sum in the same pass that already materializes centered Z
coefficients. A second full ring switch per nonce is not acceptable.

Small field limb proofs may cost more. The planner must include those bytes and
must be free to reject every small field L2 candidate if none improves the
complete suffix.

## Design

### Architecture

The change crosses the following canonical owners.

* `akita-challenges` owns physical sparse challenge norms.
* `akita-types::sis` owns the L infinity and L2 collision formulas and generated
  table lookup.
* `akita-types::layout` owns the physical Z address domain and norm subclaim
  shape.
* `akita-types::proof` owns route derived proof shapes and wire values.
* `akita-prover` computes the physical norm, applies joint nonce admission,
  proves Stage 1, and supplies Stage 2 virtual relations.
* `akita-verifier` replays the same relations and performs the final integer cap
  check.
* `akita-planner` constructs separate L infinity and L2 candidates and prices
  complete suffixes.
* `akita-sis-estimator` generates the quantum ADPS16 Euclidean table.
* `akita-schedules` stores and audits the selected route and cap.
* The profile report accounts for proof bytes, ranks, fold count, norms, and
  nonce attempts.

The intended flow is

```text
physical centered z
        |
        +--> existing digit range proof --> L infinity candidate
        |
        +--> exact square sum proof ------> L2 candidate
                    |
                    +--> Stage 2 binds z to physical Z digit planes

selected candidate --> A rank --> T width --> next witness --> suffix planner
```

### Alternatives considered

#### Global L2 replacement

A global replacement charges norm proof bytes at levels where the tighter
security route gives no suffix benefit. It also removes the proven L infinity
fallback. This design keeps the routes as separate planner candidates.

#### Prover reported norm without a proof

The old diagnostic bound a reported norm into the transcript but did not prove
it. That was useful only for estimating planner changes. It cannot size a
production A matrix.

#### One global cap

Witness geometry and fold history differ by level and field profile. One cap
either fails honest proving or loses the expected rank reduction. Caps are
schedule values attached only to checked candidates.

#### Certified operator norm rejection

A smaller challenge multiplication operator norm improves the L2 formula.
The D64 continuation uses the `(31, 11)` shell with a certified true threshold
of 18 and a strict integer threshold of 19. The D128 continuation reuses the
production `(31, 0)` shell with a certified true threshold of 13 and a strict
threshold of 14. The transcript binds the family and both thresholds. The
prover and verifier replay the same rejection sequence. Each route has an exact
accepted support certificate that retains at least 128 bits. Other dimensions
continue to use the deterministic challenge L1 norm.

#### Direct terminal L2 check

The terminal response is already clear. The verifier decodes every centered
coefficient, computes its exact integer squared norm, and rejects a value above
the scheduled cap. This route needs no recursive norm proof. It uses certified
operator norm rejection and the same A role collision formula.

#### Four square inequality proof

The verifier only needs equality to the physical integer norm, followed by a
public integer comparison. Exact square sums or bounded limb inner products do
that directly. Four square slack and carry witnesses add proof state that this
change does not need.

#### Apply the Hachi factor conservatively

Conservative factors are sound only when they bound a real conversion or
operation. Here both values are already physical. Keeping the factor changes
security parameters according to extension degree even when the physical SIS
instance is identical. That is not a property of the extracted collision.

## Documentation

The implementation PR must update the following pages when the behavior lands.

* `book/src/how/security.md` must explain the two per level security routes and
  the physical coordinate rule.
* `book/src/how/proving/sumcheck-stages.md` must explain the optional Stage 1
  norm term and Stage 2 virtualization.
* `book/src/how/configuration.md` must explain how the planner selects a route
  and cap.
* `specs/archive/2026-Q3/weak-binding-norm-fix.md` must point to this correction
  for physical A role sizing.
* Generated schedule tables and their book presentation must be refreshed.

The implementation PR keeps this spec active. Mark it implemented when the PR
merges. It can then be archived because its durable security and protocol text
has been folded into the book.

## Execution

1. Land this spec and remove the stale global cutover design.
2. Remove the Hachi factor from physical A role L infinity sizing and add
   boundary tests.
3. Add the quantum ADPS16 Euclidean table and fix the complete norm mapping.
4. Add the schedule route, cap, descriptor binding, and exact proof size model.
5. Add prover norm measurement and joint nonce admission.
6. Add the large field Stage 1 norm term and Stage 2 virtual relation.
7. Add small field limb inner products only where the no wrap test requires
   them.
8. Add verifier replay and malformed proof tests.
9. Add the exact L2 candidate to the existing suffix comparison.
10. Regenerate schedules and produce the required three way report.
11. Run the full repository gates and update the owning book pages.

## References

* Akita paper, `sections/akita/9_core_security.tex`, especially the core weak
  binding lemma and radius to collision corollary.
* Akita paper, `sections/akita/3_preliminaries.tex`, for physical ring norm
  inequalities and the logical to ring conversion boundary.
* Hachi, Lemma 7, for weak binding after denominator clearing.
* `crates/akita-types/src/sis/norm_bound.rs`.
* `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`.
* `crates/akita-types/src/proof/stage1.rs`.
* `crates/akita-prover/src/protocol/sumcheck/digit_range/`.
* `crates/akita-prover/src/protocol/sumcheck/relation_range_image/`.
* `crates/akita-sis-estimator/src/euclidean.rs`.
* `specs/fold-linf-rejection.md` for the separate digit depth policy.
* `specs/sis-quantum128-scalar-n-table.md` for the production quantum security
  policy.
