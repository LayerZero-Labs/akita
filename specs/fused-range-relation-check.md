# Spec: Fused Range-Relation Fold Check

| Field | Value |
|-------|-------|
| Author(s) | Alberto Centelles |
| Created | 2026-08-28 |
| Status | proposed |
| PR | |
| Supersedes | |
| Superseded-by | |
| Book-chapter | |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this
document are to be interpreted as described in BCP 14 when, and only when,
they appear in all capitals.

Every nontrivial claim in this document carries one of four labels:

* **normative**: a protocol rule this specification defines;
* **proved**: a statement whose complete proof is restated in this document;
* **imported**: a statement taken unchanged from the Akita paper or from the
  shipping protocol's existing analysis, with citation;
* **OPEN**: a named gap, with what would settle it.

Paper citations refer to the Akita paper's numbering (Definition 8.4,
Lemma 8.5, Corollary 8.7, Theorem 8.13, Theorem 8.17, Theorem 8.21,
equations (78), (136), sections 2.2, 5).

## Summary

At every non-terminal fold level, the opening prover runs two sequential
sumchecks over the same committed digit table: the stage-1 digit-range check
(section 5.1; degree-4 inner messages at basis `b = 8`) and the stage-2
relation-and-binding check (section 5.3, equation (78); degree-3 messages).
Every round of both is an absorb-then-sample Fiat-Shamir barrier. This
sequential chain, not field arithmetic, is the binding constraint on the
prover's per-proof fixed cost and on its distributed critical path: each
barrier is a transcript rendezvous that no amount of parallel hardware
removes.

This specification defines an opt-in per-level check shape,
`FoldCheckShape::FusedRangeRelation`, that replaces both sumchecks by one
sumcheck of degree `b + 1` whose range term is the full vanishing polynomial
of the balanced digit alphabet. The virtual table `s = w(w + 1)`, the carried
claim, the inter-stage batching challenge `gamma`, and the stage boundary
disappear; the round count per level drops from `2 * mu'` plus range-tree
sub-stages to `mu'`. Over a full recursion this cuts the sequential chain from
304 rounds to 66-69 rounds per shape (with a companion 4-level schedule; see
Performance), at the cost of larger round messages and a larger total proof.
The shape is a rounds-for-bytes trade for schedules that price prover latency
and critical-path length over proof size; the split path remains the default
and is byte-frozen.

The document contains the normative definition of the fused check over the
effective oracle, the wire format, the transcript schedule, the validator and
knowledge-error accounting rules, the complete security argument (per-level
special soundness and the composition into Theorem 8.21), and the rejected
alternative designs, including the uncommitted-virtual-table repair and its
honest status.

## Intent

### Goal

Define, normatively and with a complete security argument, a per-level fold
check shape that proves the digit-range statement and the relation-and-binding
statement of one non-terminal fold level in a single degree-`(b + 1)` sumcheck
over the level's effective oracle, so that a schedule may trade proof bytes
for a halved sequential Fiat-Shamir chain without touching the split path.

The feature introduces:

* `FoldCheckShape` (`Split` default, `FusedRangeRelation`) on
  `CommittedGroupParams`, bound into the effective-schedule descriptor digest
  only when not `Split`;
* four transcript labels (`ak/c/frb`, `ak/c/frr`, `ak/c/fscr`, `ak/a/ffw`);
* a fused wire arm of `FoldLevelProof` (the split level minus its entire
  stage-1 payload, with `b + 1` stored coefficients per sumcheck round);
* fused arms of the proof-size model and of the configuration audit walk's
  field-error numerator.

### Scope

The fused shape is a per-level schedule option. It applies exactly to
non-terminal fold levels satisfying both of the following (**normative**):

1. the level's inner-commit security route is coefficient-Linf; and
2. the level's opening digit basis is `b in {4, 8}`.

Mixed schedules (some levels fused, some split) are valid; the flag is per
level and the security argument composes per level (Design, Security
Argument).

Exclusions, each with its reason:

* **Selective-L2 (Euclidean) levels are excluded.** The Euclidean route's
  physical norm binding (Lemma 8.15) is genuinely specific to the two-stage
  split: its challenge ordering interleaves the norm proof with the stage
  boundary that fusion removes. A fused Euclidean level would need a new
  binding argument or a separate small binding sumcheck. **OPEN**; the prover
  and validator MUST reject the fused flag on a selective-L2 level.
* **Bases `b >= 16` are excluded.** The fused check's degree is `b + 1` and
  its wire stores `b + 1` coefficients per round, so large bases are
  byte-catastrophic; schedules that want a fused tail re-base tail levels to
  `b = 8`. This is a proof-size policy, not a soundness boundary.
* **Basis `b = 2` is excluded** as a degenerate shape: its vanishing
  polynomial `X(X + 1)` coincides with the binariness form, so the range term
  duplicates the binariness term. The security argument below is valid for
  every even `b >= 2` with `q > b`; the `b in {4, 8}` rule is the enforced
  eligibility set.
* **The terminal level is untouched.** The terminal has no range or relation
  sumcheck (Proposition 8.10); there is nothing to fuse.
* **The QROM caveat is inherited unchanged.** The Fiat-Shamir analysis is in
  the classical ROM for split and fused alike (the paper's own caveat).
  **OPEN** upstream, not specific to this change.

### Invariants

1. **Split path byte-frozen.** A schedule with every level `Split` MUST
   produce descriptors, transcripts, and proofs byte-identical to the current
   protocol. The descriptor digest appends a shape tag only when the shape is
   not `Split`. Protected by the existing serialization round-trip and
   generated-schedule drift tests, plus a re-prove byte-equality check
   (Evaluation).
2. **Transcript symmetry.** Prover and verifier perform the same absorb and
   sample operations under the same labels in the same order at a fused level
   (Design, Transcript schedule). Protected by prover/verifier transcript-event
   equality tests.
3. **Wire-shape disjointness.** A fused level proof carries no stage-1 payload
   (no stages, no norm proof, no carried range-image evaluation); a split
   level proof always carries at least one stage-1 stage. The verifier MUST
   accept only the shape the schedule selects for the level: a fused proof
   against a split schedule, or a split proof against a fused schedule, MUST
   fail deterministically.
4. **No free prover message in any final check.** Every quantity consumed by
   the fused final check is either public or authenticated by the recursion
   (the claimed `w~(r_x)` under the level's outgoing commitment). This is the
   property that makes the removal of the carried-claim edge sound (Security
   Argument) and it MUST be preserved by any implementation or later revision.
5. **One accounting source.** The configuration audit walk, the proof-size
   model, and the planner MUST price a fused level from the same schedule
   fields (`check_shape`, basis, `mu'`); no component may reconstruct a
   competing per-level error or byte formula.
6. **Prover/verifier schedule agreement.** The shape is resolved identically
   on the prove and verify paths and bound into the shared effective-schedule
   descriptor digest before the first challenge, so a shape mismatch is a
   digest mismatch, not a late soundness failure.
7. **Eligibility enforcement.** Schedule validation MUST reject
   `FusedRangeRelation` on any level violating the Scope rules (route, basis,
   non-terminal).

### Non-Goals

* No change to the split check, its wire, its transcript, or its analysis.
* No change to fold challenge sampling, grinding, nonce accounting, opening
  payloads, ring switch, row batching, setup offloading (Stage 3), or the
  terminal.
* No Euclidean fused levels (Scope; OPEN).
* No committed virtual table: a variant that commits `s = w(w + 1)` is a
  witness enlargement, not a fusion, and is out of scope.
* No claim that the fused shape shrinks total proof bytes; it does not
  (Performance). Byte-optimal schedules keep the split shape.
* No default flip: the fused shape ships opt-in, default `Split`.

## Evaluation

### Acceptance Criteria

- [ ] Spec review per `specs/SPEC_REVIEW.md` passes with `spec-approved`
      (this PR contains no implementation).
- [ ] The security argument's per-level statement (Theorem F1), its
      composition (Theorem F2), and the ledger corollary (Corollary F1) are
      reviewed against the paper's Section 8 and found to import only the
      statements listed in Imported statements, unchanged.
- [ ] The knowledge-error accounting reproduces: changing subtotal
      `244 -> 271` at `b = 8`, `mu' = 27`; general-basis delta
      `(b/2 - 3) * mu' / |E|`; per-level tree factor
      `(2 + beta_j) * (b + 2)^mu'` versus the split's `2 * 24^mu'`.
- [ ] The follow-up implementation PR (separate) satisfies: dual-prover
      equality on identical witness and seed; tamper rejection for a modified
      round message, a modified next-witness evaluation, a wrong stored-width
      round, and each wire against the other shape's schedule; split path
      re-proves byte-identically with the feature compiled in.

### Testing Strategy

For the implementation PR that follows this spec:

* Existing tests that MUST remain green: all serialization round-trips,
  generated-schedule drift checks, prover/verifier transcript-event equality,
  end-to-end recursive and terminal prove/verify, and every split-path byte
  anchor.
* New unit tests: honest fused instance proves and verifies at `b = 4` and
  `b = 8` with the final oracle cross-checked from first principles; an
  out-of-range cell shifts the Boolean sum off the public claim; a
  non-binary cell on the restricted support is rejected through the
  `tau0`-anchored binariness term; fused wire round-trips through its shape
  descriptor with uniform `b + 1` stored coefficients per round; serialized
  fused body equals the fused proof-size model at both bases.
* New negative tests: fused flag on a selective-L2 level rejected at
  validation; fused flag at `b = 2` or `b >= 16` rejected at validation;
  cross-shape proofs rejected.
* Audit-walk test: a fused level's field-error numerator equals the split
  numerator minus the digit-range tree terms and the `3 * mu'` relation term,
  plus `(b + 1) * mu'` and the fresh-scalar unit (Design, Validator rules).

### Performance

Round counts (the point of the feature), from the planner model introduced
by the companion pricing PR (reproduce with
`cargo run --release -p akita-planner --features catalog-check --example
fused_frontier`; the `2^29` row is the same model evaluated at a `2^29`
production shape):

| Quantity | Split (shipping) | Fused | Direction |
|---|---|---|---|
| Rounds per level (root, `mu' = 27`) | `27 + 27 = 54` | `27` | `-50%` |
| Chain rounds, shipping 7-level schedule | `304` | `155` | `-49%` |
| Chain rounds, fused + 4-level schedule, `2^29` | `304` | `66` | `-78%` |
| Chain rounds, fused + 4-level schedule, `2^30` | `304` | `69` | `-77%` |

Proof-size effects (direction and magnitude; same command):

* Stored round coefficients: `b + 1 = 9` per fused round versus `4 + 3` per
  split round-pair; a standalone fused 7-level chain grows the round-message
  stream (about `17.0 KB -> 22.3 KB` at the root geometry) and therefore
  fails any no-growth byte constraint on its own.
* Packaged with a 4-level schedule, the round-message stream shrinks by about
  `5.1 KB`, but the total serialized proof grows because the shallower
  ladder's terminal payload dominates: `129,432 B` versus the shipping
  `76,084 B` at `2^30` (`+70%`); `120,952 B` at `2^28`; `108,232 B` at
  `2^24`; `120,808 B` at a `2^29` production shape.
* The trade is rounds for bytes. Fewer rounds are simultaneously fewer bytes
  hashed, fewer transcript barriers, and a shorter distributed critical path.

Knowledge-error effect: the changing per-level subtotal moves `244 -> 271` at
the root (`b = 8`, `mu' = 27`), an absolute shift of about `2^-123` at
`|E| ~ 2^128`; against the complete per-level numerator (at least 401 at the
root) the Fiat-Shamir budget shift is under `0.1` bits. The extraction tree
factor shrinks by about `2^33.5` per level, so admissibility (equation (83))
is preserved a fortiori. Verified by the audit walk's fused arm and the
Security Argument's Corollary F1.

Verifier cost: `-49%` transcript turns; degree-`(b + 1)` point evaluations
over fewer rounds net on the order of a few hundred extra field
multiplications, negligible against the effective-oracle MLE evaluation,
which is unchanged.

Prover cost is out of scope for this spec (no implementation); the follow-up
implementation PR owes measured attributions. One watch item is recorded
there: on any level whose round scans were compute-bound rather than
bandwidth-bound, the fused degree costs more than the saved table pass, and
the implementing PR must re-attribute with measurements.

## Design

### Architecture

The fused shape is a leaf-level branch inside the existing fold orchestration;
nothing moves between crates.

* `akita-types`: `FoldCheckShape` on `CommittedGroupParams`; descriptor-digest
  arm; fused wire arm of `FoldLevelProof`; fused proof-size arm.
* `akita-transcript`: the four fused labels.
* `akita-prover`: a fused range-relation sumcheck prover beside the two
  stages, reusing the compact digit source, the relation lane-weight tables,
  and the additional-terms machinery with the binariness anchor moved to
  `tau0`; `prove_fold` branches on the shape (one phase instead of two;
  `gamma` removed; `rho_bin`/`rho_rng` added).
* `akita-verifier`: a fused sumcheck verifier (round replay at degree
  `b + 1`, final check from the claimed evaluation and public data);
  `verify_fold` branches on the shape and enforces the fused wire shape.
* `akita-schedules` / `akita-planner`: schedule resolution carries the shape;
  the audit walk and proof-size model price it (Validator rules); the planner
  treats the shape as a schedule field, not a policy tier.

The recursion's interface is unchanged in both directions: the fused sumcheck
consumes the same anchors (`tau0`, `tau1`, `alpha`) and emits the same object
the split stage 2 emits, one evaluation claim `w~(r_x)` at one point, which
the next level (or Stage 3, where offloading is scheduled) consumes
identically.

### Normative definition

#### Objects and notation

Fix one non-terminal fold level flagged `FusedRangeRelation`.

```text
E          the challenge extension field F_{q^k}; q an odd prime, q > b
b          the level's opening digit basis, b in {4, 8}
A_b        = {-b/2, ..., b/2 - 1}, the balanced alphabet, |A_b| = b,
           embedded in F_q by centered reduction (injective since q > b)
Q(X)       = prod_{a in A_b} (X - a), the alphabet vanishing polynomial,
           deg Q = b; identically Q(X) = Q_sq(X * (X + 1)) where Q_sq is
           the shipping halved-image range polynomial of degree b/2
mu'        = col_bits + ring_bits, the level's opening variable count
w          the committed next-witness digit table {0,1}^mu' -> F_q; w~ its
           multilinear extension (MLE); "committed" means bound by the
           level's outgoing commitment payload, absorbed before every
           challenge named below
I          the schedule-derived negative-binary support, I subset {0,1}^mu';
           beta_j := 1 if I is nonempty, else 0
eq(t, x)   the equality MLE; eq_I(t, x) := sum_{y in I} eq(t, y) * eq(x, y)
           (the empty sum when I is empty)
tau0       the range anchor; tau1 the row-batching point; alpha the
           ring-switch challenge (sampled in the shipping order, after the
           commitment)
```

#### The effective oracle

The relation statement is proved against the **effective** batched oracle,
not the physical relation matrix alone (**normative**). Let `Rows` be the
padded logical row space of the level, of size `2^ceil(log2 n_j)`, containing:

1. the ordinary relation rows of the physical matrix `M(alpha)`;
2. the compression relation rows; and
3. the EvaluationTrace row at the last padded logical index `i_tr`, whose
   weight table is the public trace-weight MLE `T~` and whose public target is
   the level's incoming opening claim.

Each row `i` is a public table `R_i : {0,1}^mu' -> E` (multilinear weights
once `alpha` is fixed) with public target `v_i`. Define

```text
m_eff(x)  := sum_{i in Rows} eq(tau1, i) * R_i(x)
V_alpha   := sum_{i in Rows} eq(tau1, i) * v_i
```

The EvaluationTrace row is the constraint binding the committed next witness
to the opening being proved; its inclusion in the oracle and in the target is
REQUIRED. An implementation MAY carry the trace term as a separately weighted
oracle term rather than a physical matrix row, provided the sum it proves and
the target it checks equal `m_eff` and `V_alpha` above. The trace
coefficient MAY be zero at the last non-terminal fold; the definition and the
security argument are generic in the row set.

#### The fused check

After the outgoing commitment and the anchors `tau0`, `tau1`, `alpha` are
bound, the verifier samples fresh scalars `rho_bin` (iff the level carries a
compressed payload, equivalently iff `beta_j = 1`) and `rho_rng`, and both
parties run **one** sumcheck with `mu'` rounds for the claim (**normative**):

```text
(F)   V_alpha = sum_{x in {0,1}^mu'} [  w(x) * m_eff(x)
                + rho_rng * eq(tau0, x) * Q(w(x))
                + rho_bin * eq_I(tau0, x) * w(x) * (w(x) + 1) ]
```

with the convention `rho_bin := 0` when `beta_j = 0` (the third term is then
absent). The range and binariness rows contribute nothing to the public
target because an honest witness vanishes them pointwise; the input claim of
the sumcheck is exactly `V_alpha` (the batched relation claim plus the
incoming trace opening target).

Properties fixed by (F):

* **Degree.** The summand's polynomial extension has per-variable degree
  `max(2, 1 + b, 3) = b + 1` (**proved**; Security Argument, Lemma F4's
  degree hypothesis). Each round message is one mixed univariate of degree
  `b + 1`; the summand is not globally equality-factored and MUST NOT be
  wire-encoded as an equality-factored message.
* **Binariness anchor.** The binariness term is the split's term verbatim
  with its anchor moved from stage 1's challenge point to `tau0`
  (**normative**). Its argument requires only a fresh random anchor sampled
  after the commitment; `tau0` is that anchor for both the range and
  binariness terms.
* **Final check.** The sumcheck outputs the single claim `v' = w~(r_x)` at
  the final point `r_x`. The verifier computes the expected output claim from
  `v'` and public data alone:

```text
v' * m_eff~(r_x) + rho_rng * eq(tau0, r_x) * Q_sq(v' * (v' + 1))
                 + rho_bin * eq_I(tau0, r_x) * v' * (v' + 1)
```

  where `m_eff~(r_x)` is evaluated through the relation-matrix evaluator, the
  compression-weight oracle, and the trace weight (or through the offloaded
  setup claim where Stage 3 is scheduled, exactly as the split stage-2 final
  check consumes it). Note `Q_sq(v' * (v' + 1)) = Q(v')` is the evaluation of
  the composed polynomial `Q o W` at `r_x`, not the MLE of the table
  `Q o w`; the two are distinct objects and only the former appears here.
* **Output.** `v'` and `r_x` are exactly the object the recursion already
  carries; no carried range-image claim, no `gamma`, and no stage boundary
  exist at a fused level.

#### Transcript schedule

At a fused level the transcript operations are, in order (**normative**):

1. Outgoing commitment payload absorbed; current fold's grind nonce and fold
   challenges processed (unchanged from the split, before the opening
   sumchecks).
2. `tau0`, `tau1`, `alpha` sampled under their existing labels in the
   shipping order.
3. `rho_bin` sampled under label `ak/c/frb`, only when the level carries a
   compressed payload (`beta_j = 1`).
4. `rho_rng` sampled under label `ak/c/frr`.
5. `mu'` sumcheck rounds: each round the prover's message is absorbed under
   the shared sumcheck message absorption, then the round challenge is
   sampled under label `ak/c/fscr`.
6. The next-witness evaluation `v'` is absorbed under label `ak/a/ffw`
   before recursion continues.

The split labels (`gamma`, the compression-binary challenge, the range-image
absorption, the stage-2 next-witness absorption) MUST NOT appear at a fused
level; the fused labels MUST NOT appear at a split level. The four fused
labels are:

| Label | Operation |
|---|---|
| `ak/c/frb` | challenge `rho_bin` (binariness batching, anchor at `tau0`) |
| `ak/c/frr` | challenge `rho_rng` (range batching) |
| `ak/c/fscr` | fused sumcheck round challenge |
| `ak/a/ffw` | absorb the fused level's next-witness evaluation |

Ordering note: the two fresh scalars are sampled after all three anchors and
after the commitment; their order between themselves (`rho_bin` before
`rho_rng`) is fixed by this table for wire determinism and carries no
security weight (Security Argument, Lemma F3 requires only freshness).

#### Wire format

A fused level proof is the split level proof minus its entire stage-1 payload
(**normative**):

```text
FoldLevelProof (fused shape)
  opening payload               unchanged
  fold grind nonce              unchanged
  stage-1 payload               ABSENT: no stages, no carried range-image
                                evaluation, no norm proof
  fused sumcheck proof          mu' rounds, exactly b + 1 stored
                                coefficients per round, uniform width
  next-witness binding          unchanged (outer payload or terminal state)
  next-witness evaluation       unchanged (one E element)
  stage-3 sumcheck proof        unchanged (present iff offloading scheduled)
```

Decoding is headerless: the schedule's shape descriptor for the level keys
which fields are present. The two shapes are disjoint on the wire (a split
level always carries at least one stage-1 stage). Canonical-byte rules,
validation behavior, and compression modes are those of the existing level
proof encoding; no new primitive encodings are introduced. The verifier MUST
reject a fused level body whose sumcheck stores any round with a coefficient
count other than `b + 1`, and MUST reject any stage-1 payload bytes on a
fused level.

#### Schedule field and descriptor digest

`check_shape` is a field of `CommittedGroupParams` (**normative**):

* Default `Split`. A `Split` value contributes no bytes to the descriptor
  digest, so every existing descriptor is byte-identical.
* `FusedRangeRelation` appends the fixed tag `fused-range-relation-v1` to the
  level's descriptor bytes. The digest is transcript-bound before the first
  challenge, so an old verifier rejects a fused proof deterministically at
  the digest, and prover/verifier shape disagreement is a digest mismatch.
* The shape MUST be resolved identically on the prove and verify paths (both
  resolve schedules through the same configuration hook).

### Validator rules and knowledge-error accounting

The configuration audit walk prices a fused level's field-error numerator by
replacing the split's changing terms and keeping every other term identical
(**normative**; justified by Corollary F1 below):

```text
split level numerator (changing part):
    mu' * (tree stage sum) + (tree frontier sum)   digit-range tree
  + 3 * mu'                                        stage-2 relation check
  + 1                                              relation/range batching
  + 1                                              binary batching

fused level numerator (changing part):
    (b + 1) * mu'                                  fused sumcheck rounds
  + 1                                              rho_rng isolation
  + 1                                              rho_bin coordinate
```

that is, `(b + 1) * mu' + 1` replaces `tree + 3 * mu' + 1`, with the binary
unit unchanged on both sides (it is the binary batching scalar in the split
and the `rho_bin` coordinate of the fresh-scalar node in the fused shape).
All unchanged terms (ring switch `2 * d_max`, row batching
`ceil(log2 n_j)`, range anchor `mu'`, binary-support anchor `beta_j * mu'`
priced at the walk's `beta_j := 1` over-count, physical-norm term, Stage-3
term, receiver terms) are priced identically on both shapes.

At the reference geometry `b = 8`, `mu' = 27` (single-stage range tree of
inner degree 4): split changing subtotal
`27 + 5 * 27 + 3 * 27 + 1 = 244`; fused `27 + 9 * 27 + 1 = 271`; delta
`+27 = +mu'`. At general even `b` the delta is

```text
[(b + 1) - (b/2 + 1) - 3] * mu' / |E|  =  (b/2 - 3) * mu' / |E|
```

i.e. `+mu'` at `b = 8`, `-mu'` at `b = 4` (and `-2 * mu'` at the excluded
`b = 2`); the delta is not identically positive across bases
(**proved**; Corollary F1). This subtotal is not the complete per-level
numerator; against the complete numerator the budget shift is under `0.1`
bits per level at the root.

The honest transcript squeeze bound counts one `mu'`-round chain where the
split counts two (**normative**).

The planner MUST carry the shape in catalog rows and MUST price fused rows
with the fused proof-size arm; schedule validation MUST enforce the Scope
eligibility rules before prover or verifier execution.

### Security argument

This section is the normative restatement of the fused check's knowledge
soundness: per-level special soundness (Theorem F1), the per-level error
ledger (Corollary F1), and the composition into the paper's main theorem
(Theorem F2, replacing the two split-stage suffix extractors inside
Theorem 8.21's walk). Proofs are given in full; imported statements are
listed with citations at the end of the section.

Completeness first (**proved**): an honest witness has every cell in `A_b`
(admissibility forces the honest digit bound into the alphabet, Remark 8.6)
and every `I`-cell in `{-1, 0}`, so the range and binariness terms of (F)
vanish cell-wise and (F) reduces to the exact effective relation identity;
the honest sumcheck then passes every round and the final check exactly.

#### Lemmas

**Lemma F1 (alphabet is exactly the root set).** Over `E`, the root set of
`Q` is exactly the embedded `A_b`, all roots simple; for `u in E`,
`Q(u) = 0` iff `u in A_b`.

*Proof.* `Q = prod_{a in A_b} (X - a)` by definition. The `b` factors have
pairwise distinct roots because `q > b` makes the centered embedding
injective (integer differences of alphabet values have absolute value
`< b < q`), and a degree-`b` polynomial over a field has no roots beyond its
linear factors. (**proved**)

The identity `Q(X) = Q_sq(X * (X + 1))`, with `{a, -1 - a}` partitioning
`A_b`, is the split's degree-halving mechanism; it licenses evaluating `Q`
through the shipping halved-image polynomial in the final check, and nothing
else below uses it.

**Lemma F2 (anchor).** Let `u : {0,1}^m -> E` be any table fixed before an
anchor `t` is sampled uniformly from `E^m`, and `u~` its MLE. If `u` is not
identically zero, then `Pr[u~(t) = 0] <= m / |E|`. The same bound holds for
the restricted MLE `sum_{y in I} eq(t, y) * u(y)` when `u` is nonzero
somewhere on `I`. For `m = 0` both probabilities are `0`. (Applied below at
`t = tau0` with `m = mu'` and at `t = tau1` with `m = ceil(log2 n_j)`.)

*Proof.* The MLE of a nonzero table is a nonzero polynomial (it interpolates
the table on the cube), multilinear, hence of total degree at most `m`;
Schwartz-Zippel gives `<= m / |E|`. The restricted MLE interpolates the
table `u * 1_I` on the cube, so it is nonzero iff `u` restricted to `I` is
nonzero. At `m = 0` a nonzero constant never vanishes. (**proved**; the
bound is essentially tight: a one-cell table attains `2q - 1` of the bound's
`2q` field points at `m = 2`.)

**Lemma F3 (fresh-scalar isolation).** Fix `delta_0, delta_1, delta_2 in E`
before `(rho_rng, rho_bin)` is sampled uniformly and independently from
`E^2`.

(a) *Event form.* If `(delta_1, delta_2) != (0, 0)` then
`Pr[delta_0 + rho_rng * delta_1 + rho_bin * delta_2 = 0] <= 1 / |E|`. If
`delta_1 = delta_2 = 0` and `delta_0 != 0`, the probability is `0`.

(b) *Extraction form.* Given the identity
`delta_0 + rho * delta_1 + sigma * delta_2 = 0` at the three coordinate-wise
points `(rho, sigma)`, `(rho', sigma)`, `(rho, sigma')` with `rho' != rho`
and `sigma' != sigma`, one deterministically concludes
`delta_0 = delta_1 = delta_2 = 0`. When `beta_j = 0` (no `rho_bin`), two
points `rho' != rho` suffice for `delta_0 = delta_1 = 0`.

*Proof.* (a) If `delta_1 != 0`: for every fixed `rho_bin` exactly one
`rho_rng` satisfies the equation, so the satisfying count over `E^2` is
exactly `|E|` out of `|E|^2`; symmetrically when `delta_2 != 0` the count is
at most `|E|`. If both linear coefficients vanish the equation reads
`delta_0 = 0`. (b) The 3-by-3 coefficient matrix with rows `(1, rho, sigma)`,
`(1, rho', sigma)`, `(1, rho, sigma')` has determinant
`(rho' - rho) * (sigma' - sigma) != 0`, so the homogeneous system has only
the zero solution; the two-point case is the top-left 2-by-2 minor.
(**proved**)

**Lemma F4 (fused sumcheck tree extraction).** Let `G` be a `mu'`-variate
polynomial over `E` of per-variable degree at most `D`, and consider a
sumcheck transcript tree for claimed sum `c_0`: at each round node of depth
`t` with challenge prefix, the prover's message is a univariate `p_t` of
degree at most `D`; the verifier checks `p_t(0) + p_t(1) = c_{t-1}` and
passes `c_t := p_t(r_t)` to the child at challenge `r_t`; each round node
has `D + 1` children at pairwise distinct challenges; and every leaf at full
point `rho in E^mu'` satisfies the final condition `c_mu' = G(rho)`. If
every node's checks pass, then `c_0 = sum_{x in {0,1}^mu'} G(x)`. For
`mu' = 0` the tree is a single leaf and the conclusion is the final
condition itself.

*Proof.* For a prefix `rho_<t` define
`TS(rho_<t) := sum over tails in {0,1}^{mu' - t + 1} of G(rho_<t, tail)`,
and `S_t(X) := TS(rho_<t, X)` read as a univariate of degree at most `D`
(per-variable degree bound of `G`). By induction from the leaves, at every
node the incoming claim equals `TS` of its prefix. Base: at a leaf the
incoming claim is `c_mu' = G(rho) = TS(rho)` (empty tail). Step: at a
depth-`t` node, each child `i` gives, by the inductive hypothesis applied to
its subtree, `p_t(r_t^i) = TS(rho_<t, r_t^i) = S_t(r_t^i)`. The polynomials
`p_t` and `S_t` both have degree at most `D` and agree at `D + 1` distinct
points, hence are equal; the node's check then gives
`c_{t-1} = p_t(0) + p_t(1) = S_t(0) + S_t(1) = TS(rho_<t)`. At the root
`c_0 = TS(empty) = sum_x G(x)`. Distinctness of the `D + 1` children is
supplied by the transcript-tree framework, and `|E| >= D + 1` holds
at every production field size since `D = b + 1 <= 9` and `|E| = q^k` is
astronomically larger. (**proved**)

#### Theorem F1 (fused-level special soundness)

Fix a fused non-terminal level and a transcript prefix through the sampling
of `tau0`, `tau1`, `alpha` (commitment absorbed first). Suppose the
extractor holds:

* **(H1)** a *descendant-certified fused subtree*: a coordinate-wise tree
  whose root is the fresh-scalar node (the central draw
  `(rho_rng, rho_bin)`, plus one sibling differing only in the `rho_rng`
  coordinate, plus one sibling differing only in the `rho_bin` coordinate
  iff `beta_j = 1`), below each of which hangs a complete accepting sumcheck
  tree in the sense of Lemma F4 with `D = b + 1` (hence `b + 2` children per
  round node); and
* **(H2)** for every leaf, a descendant certificate authenticating the
  leaf's final claim: the recursion below the leaf yields a source table for
  the outgoing claim under the level's outgoing commitment, the leaf's
  transmitted `v'` equals `W(rho)` for the MLE `W` of that table, and all
  leaves' tables agree. (If two descendant-extracted tables under the same
  commitment differ, Corollary 8.7 already returns a role collision, and
  that collision is the theorem's conclusion.)

Then the deterministic extractor outputs, for the common table `w`:

* **(i) every effective relation row**:
  `sum_x R_i(x) * w(x) = v_i` for every `i in Rows`, including the
  EvaluationTrace binding `sum_x T~(x) * w(x) = v_{i_tr}`; **or** the event
  `E_row` occurred (probability at most `ceil(log2 n_j) / |E|` over `tau1`);
* **(ii) alphabet membership of every cell**: `w(x) in A_b` for every
  `x in {0,1}^mu'`; **or** the event `E_rng` occurred (probability at most
  `mu' / |E|` over `tau0`);
* **(iii) the binariness restriction**: `w(y) in {-1, 0}` for every
  `y in I`; **or** the event `E_bin` occurred (probability at most
  `mu' / |E|` over `tau0`; vacuous when `I` is empty).

The level's contributions to the interactive error ledger and to the
extraction tree are: `(b + 1) / |E|` error and `b + 2` children per sumcheck
round node (`mu'` nodes); `(1 + beta_j) / |E|` error and `2 + beta_j`
children at the fresh-scalar node under the paper's coordinate-wise
convention (the single-event probability is at most `1 / |E|` by
Lemma F3(a); the convention over-counts by `beta_j / |E|`); `mu' / |E|` and
`beta_j * mu' / |E|` for the two anchor events at the sampled `tau0`, no
branching; `ceil(log2 n_j) / |E|` for the row-batching event at the sampled
`tau1`, no branching.

*Proof.*

**Step A (sumcheck).** All leaves share one table `w` by (H2), so for each
fresh-scalar child with scalars `(rho, sigma)` one summand polynomial
`G_{rho,sigma}` is well defined (the polynomial extension of (F)'s summand
with those scalars; per-variable degree `b + 1`: the relation term is `2`,
the range term `1 + b`, the binariness term `3`). By (H2) each leaf's final
check equals `G_{rho,sigma}(rho_x)` evaluated correctly: the verifier
computed it from `v' = W(rho_x)` and public data, and every quantity in the
final-check formula is the true evaluation of the corresponding factor at
`rho_x`; in particular `Q_sq(v' * (v' + 1)) = Q(W(rho_x))`, the composed
polynomial's evaluation, exactly the range term's factor. Lemma F4 applied
below each fresh-scalar child yields that child's initial-claim identity

```text
delta_0 + rho * Rng + sigma * Bin = 0, where
delta_0 := sum_x w(x) * m_eff(x) - V_alpha
Rng     := sum_x eq(tau0, x) * Q(w(x))
Bin     := sum_x eq_I(tau0, x) * w(x) * (w(x) + 1)
```

The target `V_alpha` is the same public value at every child because it does
not depend on the fresh scalars (the range and binariness statements have
target zero).

**Step B (isolation).** Lemma F3(b) on the coordinate-wise children gives
`delta_0 = 0`, `Rng = 0`, and (if `beta_j = 1`) `Bin = 0`; when
`beta_j = 0` the summand has no binariness term and the two-point case
applies.

**Step C (row peel).** `delta_0 = 0` reads
`sum_i eq(tau1, i) * d_i = 0` with `d_i := sum_x R_i(x) * w(x) - v_i` (the
effective-oracle weight structure, covering ordinary, compression, and trace
rows in the padded index space). The map
`tau -> sum_i eq(tau, i) * d_i` is the MLE of the row-discrepancy table in
`ceil(log2 n_j)` variables, and that table is fixed when `tau1` is sampled
(it depends on `w`, the rows, and `alpha`, all absorbed or sampled before
`tau1` in the shipping order; if an implementation samples `alpha` after
`tau1`, the same event is charged with the two challenges' roles exchanged
inside the paper's existing row-batching and ring-switch terms, and nothing
else changes, since the fused check hands the peel the identical aggregate
identity the split stage 2 hands it). By Lemma F2, either every `d_i = 0`,
which is (i), or `E_row` occurred.

**Step D (anchors).** `Rng = 0` says the MLE of the table
`x -> Q(w(x))` vanishes at `tau0`; the table is fixed before `tau0` because
`w` is committed first. By Lemma F2, either `Q(w(x)) = 0` for all `x`,
which by Lemma F1 is (ii), or `E_rng` occurred. Identically, `Bin = 0` and
Lemma F2's restricted form give `w(y) * (w(y) + 1) = 0`, i.e.
`w(y) in {-1, 0}` (the quadratic `X * (X + 1)` has exactly these two roots
over a field), for every `y in I`, or `E_bin` occurred. Both anchor events
live at the same sampled `tau0` and are charged by a union bound; no
independence between them is used.

The ledger and tree counts restate the node parameters used: `mu'` scalar
round nodes at `k = b + 2` children (error `(k - 1) / |E|` each, priced by
the transcript-tree framework), one coordinate-wise fresh-scalar node with
`1 + beta_j` coordinates and `k = 2`, and three sibling-free Schwartz-Zippel
events. (**proved**, given the imported items I1 and I2 below.)

#### Corollary F1 (fused per-level error, replacing equation (136))

At a fused coefficient-Linf level `j`, the interactive knowledge error of
the level is

```text
eps_j_fused = kappa_j + (1 / |E|) * (   2 * d_max_j          ring switch
                                      + ceil(log2 n_j)       row batching
                                      + mu'_j                range anchor
                                      + beta_j * mu'_j       binary anchor
                                      + (b + 1) * mu'_j      fused sumcheck
                                      + 1 + beta_j           fresh-scalar node
                                      + nu_j )               = 0 on Linf
```

replacing the paper's (136), whose split-specific terms were: the
digit-range tree stage and frontier sums, the `3 * mu'` relation check, the
relation/range batching unit, and the binary batching unit. The Stage-3
addition (137) and everything outside the bracket are unchanged. The
changing-subtotal delta at general even `b` is `(b/2 - 3) * mu'_j / |E|`
(the split's tree contributes `(b/2 + 1) * mu'` at a single-stage basis
`b <= 8`); `244 -> 271` at `b = 8`, `mu' = 27`. The per-level extraction
tree factor of the opening sumchecks moves from the split's
`2 * ((b/2 + 2) * 4)^mu' = 2 * 24^mu'` at `b = 8` (range rounds at
`b/2 + 2 = 6` children, `gamma` node at 2, relation rounds at 4) to
`(2 + beta_j) * (b + 2)^mu' = 3 * 10^mu'`, smaller by about `2^33.5` at
`mu' = 27`, so the admissibility condition (83) is preserved a fortiori.
(**proved**, by Theorem F1's node accounting plus the imported unchanged
terms.)

#### Composition into Theorem 8.21

**Definition F1 (fused round list; replaces Definition 8.12 at fused
levels).** Identical to Definition 8.12 except for the round list below a
folding-challenge child, which reads: the ring-switch rounds and the fused
range-relation rounds (the fresh-scalar node and the `mu'` sumcheck rounds
of Theorem F1), followed by the recursive protocol that authenticates the
outgoing opening claim. The descendant-certified condition, the freedom of
later challenges across folding children, and the terminal tail clause
(Proposition 8.10 supplies the certificate directly) are verbatim.
(**normative**)

**Theorem 8.13-F (fold extraction over the fused suffix).** The statement of
Theorem 8.13 holds verbatim at a fused level, with the extracted
response-difference bound computed from the same certified alphabet `A_b`.

*Proof.* The paper's proof is reused with one substitution: where it invokes
the separate range, norm, and relation sumcheck extractors, apply Theorem F1
(whose hypotheses (H1), (H2) are exactly Definition F1's subtree plus
descendant certification) to obtain, per folding child, every effective
relation row, alphabet membership of every cell, and the binariness
restriction. There is no separate norm extractor to replace because
`nu_j = 0` on the coefficient route (the norm machinery, Lemma 8.15, is
Euclidean-only and out of fused scope). The ring-switch extractor
(Theorem 8.17's ring-switch half) then lifts the extracted per-row field
identities to the native mixed-dimension ring identities exactly as before;
it consumes per-row identities and does not depend on which sumcheck
produced them. The remainder of the paper's proof consumes only (a) exact
per-branch relations, (b) certified digit alphabets, and (c) unit challenge
differences, all supplied identically; the certified alphabet is the same
`A_b` (Lemma F1), so the recomposition bound (52) and the Corollary 8.16
coefficient-route collision radius are identical to the split's.
(**proved**, given Theorem F1 and imports I3, I4.)

**Collision-pricing conformance (Remark 8.6).** Remark 8.6 prices every
collision at the alphabet enforced by the level that scans a segment. The
fused check scans the same committed table over the same `mu'`-cube as the
split stage 1 and enforces the same alphabet `A_b` (via `Q`'s root set) and
the same `{-1, 0}` restriction on the same `I`. Segment coverage, the B/D
collision norms `b - 1`, the binariness-covered compression pricing, and
the Module-SIS inventory are therefore unchanged. (**proved**, from
Theorem F1 (ii), (iii).)

**The carried-claim edge and why its removal is sound.** In the split
suffix, the transcript-tree walk below a folding child has the shape:
stage-1 range rounds, then the `gamma` node with two children, then stage-2
rounds, then the recursion. The `gamma` edge exists because stage 1's final
check consumes the carried claim about the *uncommitted* virtual table
`s = w * (w + 1)`: that message is not authenticated by the recursion, so
the extractor must first run stage 2 leaf-to-root, use the `gamma` siblings
to isolate the carried-claim statement from the relation statement
(equation (78) is exactly this binding), and only then run stage 1's
backward induction with its final check anchored by the now-bound value.

In the fused suffix there is no such edge because there is no such message:
every quantity consumed by a fused final check is either public or
descendant-certified (Invariant 4). The final check reads `v'`
(descendant-certified, (H2)) and public functions of the transcript
(`m_eff~(r_x)`, `eq(tau0, r_x)`, `eq_I(tau0, r_x)`, `Q(v')`). The alphabet
statement's anchor therefore flows directly from the recursion through the
same certificate that anchors the relation statement, and the extraction
role of the `gamma` siblings (separating bundled statements) is taken over
by the fresh-scalar node's siblings (Lemma F3(b)), which sit before the
sumcheck rounds instead of between two sumchecks. The removal is not an
argument about the `gamma` edge at all: the fused walk never forms the
statement that edge isolated, because the uncommitted table, its MLE, and
its carried claim do not exist in the fused protocol. What must be, and is,
re-proved is that the alphabet statement obtained instead (`Rng = 0` at the
`tau0` anchor) certifies the same per-cell predicate: that is Theorem F1
(ii) via Lemma F1, at degree `b` instead of `b/2`, which is exactly the
degree floor priced under Alternatives Considered. (**proved**)

**Theorem F2 (Theorem 8.21 with the fused suffix).** Fix an admissible
schedule in which every non-terminal fold level is flagged split or fused,
a fused flag requiring the coefficient-Linf route at that level. Define the
interactive error with Corollary F1's `eps_j_fused` at fused levels
(Stage-3 term (137) unchanged) and the extraction tree with the fused
per-level factors above. Under the hypotheses of Theorem 8.21 otherwise
unchanged (Module-SIS hardness for the same inventory, setup distance,
admissibility (83) for the smaller tree), the Fiat-Shamir transform of the
assembled protocol is a knowledge-sound multilinear PCS with error at most

```text
(Q + 1) * eps_int_fused(sch) + Delta_setup + sum over the MSIS inventory of
Adv_MSIS
```

with extractor running time bounded as in the paper's (141)-(142).

*Proof.* By the paper's six steps, with the following per-step disposition.

* **Step 1 (idealize the setup): verbatim.** No reference to the stage
  structure. **Imported.**
* **Step 2 (Fiat-Shamir to an accepting transcript tree): parameter
  substitution only.** The round list at a fused level replaces the split's
  per-level entries (range-tree rounds, inter-stage frontier combinations,
  the `gamma` combination, relation rounds) by the fresh-scalar node
  (`1 + beta_j` coordinates, 2 values each) and `mu'` scalar rounds
  (`b + 2` values each). Both node types are instances the transcript-tree
  framework already prices: a coordinate-wise node with `l` coordinates
  contributes as the fold-challenge nodes do, and a scalar degree-`D`
  sumcheck challenge contributes `D + 1`. The interactive error of the
  listed protocol is Corollary F1's; the CWSS-to-Fiat-Shamir conversion
  (Theorem 3.13, equation (140)) gives the transcript tree except with
  probability `(Q + 1) * eps_int_fused`; admissibility (83) holds because
  the fused tree factors are strictly smaller. Condition on the tree.
  **Proved (substitution), given import I2.**
* **Step 3 (evaluation-trace terminal base case): verbatim.** The terminal
  runs no range, relation, or fused sumcheck (Proposition 8.10); its digits
  and norms are checked in clear and Lemma 8.9 converts its incoming
  recursive-witness group. Fusion touches nothing at the terminal, so the
  last non-terminal fold's outgoing object is descendant-certified exactly
  as before. **Imported.**
* **Step 4 (typed backward induction): rewritten at fused levels.** Suppose
  the opening emitted by non-terminal fold `h` has been authenticated by
  its recursive suffix. Fix one folding-challenge child at level `h`. Every
  branch of the later strong-check subtree ends in a claim under the same
  outgoing commitment, because that commitment is absorbed before `tau0`,
  `tau1`, `alpha`, the fresh scalars, and every sumcheck round (Transcript
  schedule); the induction hypothesis extracts a source table for each such
  claim; disagreement gives a Corollary 8.7 collision; otherwise the child
  is descendant-certified in the sense of Definition F1, which is (H2), and
  the child's fused subtree is (H1). Apply Theorem 8.13-F: inside each
  child, Theorem F1 and the ring-switch lift recover exact native-ring
  relations and certified digits (`A_b` everywhere, `{-1, 0}` on `I`), and
  hence certified response bounds via (52). Comparing those exact relations
  across folding siblings eliminates every unchanged coordinate without
  comparing any later challenge (the paper's argument, consuming only
  per-branch exactness). At split-flagged levels the paper's original
  Step 4 applies unchanged; mixed schedules compose because the induction
  is per level and each level's extractor consumes only the descendant
  certificate, which is flag-agnostic. The setup-offloading paragraph is
  verbatim with the substitution recorded under import I5. **Proved, given
  Theorem F1 and imports I3, I4, I5.**
* **Step 5 (recover the application openings): verbatim with one clause
  scoped.** The packing-level and evaluation-trace-level connections
  consume the extracted relations, not the sumcheck shape; the public-batch
  and group-combination arguments are untouched. The Euclidean paragraph
  (Lemma 8.15 and Corollary 8.16's Euclidean half) applies only at
  Euclidean-route levels, which are split-flagged by Scope, where the
  paper's own proof stands; at fused levels the coefficient half of
  Corollary 8.16 applies, fed by the same certified alphabet. The closing
  charge of residual failures to the Step-2 error now covers Theorem F1's
  three sibling-free events (`E_row`, `E_rng`, `E_bin`), which are
  fresh-public-coin Schwartz-Zippel events of exactly the kind that clause
  describes. **Imported plus Theorem F1.**
* **Step 6 (shared-view collisions to MSIS): verbatim.** The collision
  inventory and radii are unchanged (collision-pricing conformance above).
  **Imported.**

(**proved**, given the listed imports.)

#### Boundary cases

Each of the following was checked against the statements above; none
requires a hypothesis beyond those stated. (**proved**, case by case.)

* **Terminal level.** No fused check exists there (Step 3). At the last
  non-terminal fold, (H2) is supplied by Proposition 8.10 plus Lemma 8.9;
  the trace coefficient may be zero at that level, which Theorem F1
  tolerates since it is generic in the row set.
* **Empty binariness support (`beta_j = 0`).** `rho_bin` is not sampled,
  the summand has two terms, the fresh-scalar node has one coordinate and
  two children, Lemma F3's two-point case applies, and (iii) is vacuous.
* **`b = 4`.** The proof holds (every even `b >= 2` with `q > b`); the
  error delta versus the split is `-mu' / |E|`.
* **`mu' = 0`.** The sumcheck is the bare final check (Lemma F4's base
  case); the anchors contribute zero error (Lemma F2 at `m = 0`) and every
  `mu'`-proportional ledger term is zero.
* **Empty row set.** `delta_0 = -V_alpha` with `V_alpha = 0`; vacuous.
* **Repeated challenges.** Round-node child distinctness is supplied by the
  transcript-tree framework; the fresh-scalar node needs only its two
  coordinate-wise sibling inequalities; collisions across rounds or between
  `tau0`, `tau1`, `r_x`, or challenges of different levels are never
  compared by any step (the anchors are single-point Schwartz-Zippel
  events, not equality arguments).
* **Shared anchor.** Range and binariness both anchor at `tau0`; the two
  events are union-bounded at the same sample, costing
  `mu' + beta_j * mu'`, and no independence between them is used.
* **Offloaded final check.** Where Stage 3 is scheduled, the fused final
  check's `m_eff~(r_x)` evaluation is replaced by the offloaded setup
  claim, authenticated downstream by the setup-offloading soundness lemma,
  exactly as the split stage-2 final check consumes it; "public data" in
  (H2) then means public given the separately authenticated setup opening,
  and the extra error is (137)'s Stage-3 term, unchanged.
* **No circularity.** Theorem F1's hypotheses consume only descendant data
  (deeper levels, extracted first in Step 4's leaf-to-root order) and the
  level's own subtree; its conclusions feed only shallower levels. The
  table `w` is fixed by a commitment absorbed before every challenge the
  theorem reasons about, so every Schwartz-Zippel event is measurable at
  its sampling time.

#### Imported statements

Everything above that is not proved in place imports exactly the following,
each already load-bearing for the split protocol's own analysis and used
here unchanged:

| Import | Content | Source |
|---|---|---|
| I1 | Effective-oracle row structure: the trace binding is a logical row at the last padded index of the `tau1` row space, with a public weight table and the incoming opening claim as target; compression rows carry their own `eq(tau1, .)` weights; the row set may vary per level | shipping protocol (section 5.3, equation (78) context); restated normatively in this spec's Design, The effective oracle |
| I2 | CWSS framework and Fiat-Shamir conversion: per-node parameters compose to the interactive error, and the FS extractor obtains the transcript tree except with probability `(Q + 1) * eps_int` (equation (140)); node types priced: coordinate-wise nodes and scalar degree-`D` sumcheck rounds | Theorem 3.13 |
| I3 | Weak extraction and binding: Definition 8.4; Lemma 8.5 with its radii; Corollary 8.7 (role collision); Remark 8.6 (alphabet-to-collision-norm pricing); Theorem 8.13 (fold-coordinate extraction from exact per-branch relations); Corollary 8.16 (coefficient-route collision radius via the recomposition (52)) | paper, section 8 |
| I4 | Ring-switch extraction: the ring-switch step's special soundness and the lift of extracted field-level row identities to native mixed-dimension ring identities | Theorem 8.17 (ring-switch half) |
| I5 | Setup offloading: Stage 3 is a separate degree-2 sumcheck reading the same final point; receiver-side two-group handling; where the split analysis says stage 2 produces the recursive-witness opening, the fused sumcheck produces the identical object (final point and claim) | Lemma 8.20, Lemma 8.9 |
| I6 | Terminal and EOR base case | Proposition 8.10, Lemma 8.9, Definition 8.8 |

An exhaustive pass over the paper's Theorem 8.21 Steps 2-5 and supporting
statements for properties specific to the two-stage split found exactly
four, each closed above:

| Split-specific reliance | Disposition |
|---|---|
| Definition 8.12's round list names range and relation rounds separately | Definition F1 (rewritten round list). Closed. |
| Theorem 8.13's proof opening invokes separate range/norm/relation sumcheck extractors | Theorem 8.13-F (Theorem F1 substituted; the norm extractor is vacuous on the Linf route). Closed. |
| Lemma 8.20's proof says stage 2 already produces the recursive-witness opening | The fused sumcheck produces the identical object (the final point and claim), and Stage 3 reads the same point; no other sentence of Lemma 8.20 references the split. Closed (import I5). |
| Remark 8.19 says the opening obligation is included in the row-batching and relation-check terms | Literally true of the fused ledger as well: the trace row is a `tau1`-batched row inside Theorem F1 (i). Closed. |

The Euclidean-route stage-1/stage-2 physical binding (Lemma 8.15) is
genuinely split-specific and is exactly why the fused flag is restricted to
Linf levels; its fused analogue is **OPEN** and not claimed here.

#### Open items

| Item | Status | What settles it |
|---|---|---|
| Fused Euclidean levels (physical norm binding under fusion) | OPEN, excluded by Scope; the split analysis stands at Euclidean levels | a new binding lemma, or a separate small binding sumcheck on Euclidean levels |
| QROM Fiat-Shamir | OPEN upstream, split and fused alike (the paper's own caveat) | out of this spec's scope |
| Success-probability lemma for the rejected repair's break construction | OPEN; see Alternatives Considered (it gates nothing in this design) | a lemma on root existence in the induced univariates |

### Alternatives Considered

**A1. Random linear combination of the two stages at degree
`max(4, 3) = 4`.** Ill-typed (**proved**): an RLC requires both statements
to exist before the first shared round, but stage 2's statement contains
the equality weight and the carried claim at stage 1's *complete* challenge
point (visible in equation (78)'s dependence on that point). There is no
common statement to combine, so "fuse at degree 4" is not a protocol.

**A2. Keep the degree-`b/2` virtual table uncommitted and add an in-band
consistency term (the degree-5 repair).** Explicitly rejected. The tempting
repair keeps `s` at degree `b/2` and batches, in the same sumcheck, the
range check on `s` with a consistency zerocheck of
`s(x) - w(x) * (w(x) + 1)`, letting the prover send the final claim about
`s` as a free message. There is an explicit adversary construction against
it: for a committed `w` with one out-of-range cell, the adversary defines a
doctored table `s'` agreeing with `w * (w + 1)` off three points, plants a
root of the halved-image polynomial at the bad cell, solves two scalar
constraints at two adjustment points (one affine, one by taking a root of
an induced univariate of degree at most 4 over `E`), and runs the honest
sumcheck on the doctored summand; every round and the final check verify,
because an uncommitted table's final claim is a free message.

The honest status of this construction is **unproven, not a refutation**:
the adversary succeeds only when at least one induced univariate has a root
in `E`, and neither the uniformity nor the independence heuristics behind
"a random quartic has a root with constant probability" has been proved
(OPEN, above). This spec therefore treats the repair as **unproven**, not
as refuted: it MUST NOT be adopted without its own complete soundness
proof, which does not exist, and nothing in the fused design depends on the
break being unconditional. The structural point stands independently of the
break: sumcheck knowledge soundness requires every final-check input to be
checkable against a committed oracle or public data, and the repair
violates that (Invariant 4); the shipping stage 2 exists precisely to bind
the carried claim to the committed witness (equation (78)), the payment
that section 2.2's degree-halving substitution defers rather than removes.

**A3. A cheaper fused polynomial (degree below `b`).** Impossible within
this template (**proved**): in any single-sumcheck check of shape (F) in
which the alphabet is certified by a fixed public univariate applied to `w`
and the final claim is verified against `w~(r_x)` and public data alone,
completeness forces the univariate to vanish on all of `A_b` (every
alphabet value occurs in some honest witness), and the imported
collision-norm pricing (Remark 8.6, Corollary 8.16) forces its root set to
stay inside the priced alphabet, so the root set is exactly `A_b` and the
degree is at least `b`. The degree-halving of the split and fusion are
incompatible in this template; the design choice is binary (bytes or
rounds), the shipping code chose bytes, and this spec defines the other
branch as an option. Variants certifying the alphabet through other
committed structure fall outside the template and outside this spec's
scope.

**A4. Commit the virtual table.** Committing `s = w * (w + 1)` restores
degree `b/2 + 1` per round but enlarges the witness and the commitment
payload at every level; it is a witness enlargement, not a fusion, and does
not remove the second sumcheck's rounds. Out of scope.

**A5. Status quo (split only).** The split shape is byte-optimal and
remains the default; this spec adds an option and freezes the split path
(Invariant 1). A deployment that prices proof bytes over rounds should not
enable the fused shape.

## Documentation

* This spec is the normative reference for the fused shape until the
  implementation lands; its durable content then folds into
  `book/src/how/security.md` (knowledge-error ledger, Corollary F1) and the
  protocol chapter that owns the fold stages (check shapes, wire,
  transcript), after which the header's `Book-chapter` field is filled and
  the lifecycle advances per `specs/PRUNING.md`.
* `docs/doc-blast-radius.json`: the implementation PR must add the fused
  module paths to the security and proof-size blast radii.
* Related specs: `fold-linf-rejection.md` (grinding and honest sizing,
  unchanged by this spec), `selective-l2-fold-security-sizing.md` (the
  excluded Euclidean route), `heterogeneous-group-source-contracts.md`
  (group and terminal ownership, unchanged).

## Execution

Suggested implementation order (each gate independently checkable):

1. `akita-types`: `FoldCheckShape`, descriptor-digest arm, eligibility
   predicate, fused wire arm, fused proof-size arm; serialization tests.
2. `akita-transcript`: the four labels.
3. Prover: fused sumcheck module (dense reference path first) and the
   `prove_fold` branch; unit tests from Testing Strategy.
4. Verifier: fused round replay and final check; `verify_fold` branch;
   cross-shape and tamper rejection tests.
5. Schedules/planner: resolution carries the shape; audit-walk fused arm
   with the numerator test; planner rows.
6. Rollout: opt-in resolution override applied symmetrically on the prove
   and verify paths (both sides resolve through the same configuration
   hook), default off; the wire is versioned by the schedule digest, so no
   compatibility shims are needed.

Risks to resolve first: none protocol-level beyond the OPEN items (which
are scoped out, not deferred); the main implementation risk is accidental
transcript divergence between prover and verifier at the branch point,
covered by the transcript-event equality tests.

## References

* Akita paper, section 2.2 (degree-halving substitution), section 5
  (protocol; 5.1 range tree, 5.3 relation check, equation (78)), section 8
  (Definition 8.4, Lemma 8.5, Remark 8.6, Corollary 8.7, Definition 8.8,
  Lemma 8.9, Proposition 8.10, Definition 8.12, Theorem 8.13, Lemma 8.15,
  Corollary 8.16, Theorem 8.17, Lemma 8.18, Remark 8.19, Lemma 8.20,
  Theorem 8.21, equations (52), (78), (83), (126)-(129), (134), (136),
  (137), (140), (141)-(142)), Theorem 3.13.
* Companion pricing PR (planner model, `fused_frontier` example): the
  round/byte frontier rows quoted under Performance.
* `specs/fold-linf-rejection.md`, `specs/selective-l2-fold-security-sizing.md`,
  `specs/heterogeneous-group-source-contracts.md`.
