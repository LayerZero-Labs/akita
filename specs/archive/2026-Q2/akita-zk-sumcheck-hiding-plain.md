# Spec: Akita ZK Sumcheck Hiding Plain Opening

| Field     | Value                  |
| --------- | ---------------------- |
| Author(s) | Amirhossein Khajehpour |
| Created   | 2026-05-12             |
| Updated   | 2026-05-26             |
| Status    | implemented in branch  |
| PR        |                        |

## Summary

This branch implements a plain-opening version of Akita's sumcheck-hiding ZK
path. The proof carries a separate hiding-factor commitment `u_blind`, reveals
the `hiding_witness` and the B-side blinding digits for that commitment, checks
`u_blind` directly against the revealed payload, and verifies deferred R1CS rows
from the `akita-r1cs` crate directly against the revealed witness.

This is not the final zero-knowledge protocol. It fixes the transcript shape and
the relation inventory, but the hiding material remains public until a later
Spartan / LNP22 / tail-sigma proof replaces the plain opening.

Security boundary: this branch enforces
`u_blind = Commit(hiding_witness; r_B_h)` by recomputing the commitment from the
revealed `hiding_witness` and revealed `b_blinding_digits`. That is a plain
opening check, not a zero-knowledge opening proof. Absorbing `u_blind` before the
masked opening challenges fixes the intended commit-before-challenge transcript
shape, but the branch is still not a final zero-knowledge protocol because the
hiding material and commitment blinding digits remain public.

In folded `feature = "zk"` builds, the implementation currently hides:

- root gamma-combined `y_ring` messages, one ring mask per distinct root opening
  point;
- root extension-opening reduction partials and compressed reduction-sumcheck
  messages when tensor projection is active;
- recursive-level `y_ring` messages, one ring mask per recursive fold level;
- recursive extension-opening reduction partials and compressed
  reduction-sumcheck messages when the challenge field is a nontrivial
  extension;
- stage-1 eq-factored sumcheck round messages;
- stage-1 tree child claims between product layers;
- `AkitaStage1Proof::s_claim`;
- stage-2 compressed sumcheck round messages;
- recursive `next_w_eval` handoffs, serialized as
  `AkitaStage2Proof::next_w_eval_masked`.

External claimed openings remain public transcript inputs. The real witness
commitments and the hiding-factor commitment use separate Ajtai B-side blinding
digits; those short blinding digits are distinct from the full-field one-time
pads stored in `hiding_witness`.

## Non-Goals

- Do not claim full zero-knowledge from this branch.
- Do not hide external claimed opening values.
- Do not implement Spartan, LNP22, or the final joint tail sigma protocol.
- Do not remove the plain opening of `hiding_witness`.
- Do not change transparent proof shapes.
- Do not prove the hiding-factor commitment opening in zero knowledge. The
  current verifier enforces `u_blind = Commit(hiding_witness; r_B_h)` by direct
  recomputation from revealed data.
- Do not add a terminal direct-witness `next_w_eval_masked` field. Terminal
  direct-witness levels carry the direct witness payload and no successor
  next-witness evaluation handoff.

## Proof Objects

### Real Witness Commitment

The ordinary Akita commitment is still the commitment to the real witness
polynomial or folded witness. Under `feature = "zk"`, commitment generation adds
fresh B-side blinding digits to the wire-visible output:

```text
t       = A_msg * s
t_hat   = decompose(t)
u       = B_msg * t_hat + B_blind * r_B
```

The corresponding prover hint stores `b_blinding_digits` for the real
commitment. These are short Ajtai hiding digits, not sumcheck pads.

### Hiding-Factor Commitment

Folded ZK proofs additionally carry:

```text
ZkHidingProof {
    u_blind,
    hiding_witness,
    b_blinding_digits,
}
```

The prover constructs `u_blind` by interpreting `hiding_witness` as a padded
dense polynomial, committing to it with the original root fold parameters
(`root_step.params`) adjusted to the hiding-witness length, and adding
independent B-side blinding digits:

```text
t_h       = A_h_msg * hiding_witness
t_h_hat   = decompose(t_h)
u_blind   = B_h_msg * t_h_hat + B_h_blind * r_B_h
```

The verifier rejects empty `u_blind` / `hiding_witness`, reconstructs the padded
hiding-witness polynomial, recomputes the direct B-side commitment using the
revealed `b_blinding_digits`, compares the result to `u_blind`, and then absorbs
`u_blind` before root challenges. `b_blinding_digits` is not part of the
deferred R1CS inventory; it is consumed by this direct plain-opening commitment
check.

## Hiding Witness Layout

The implementation relies on a single cursor through `hiding_witness`, which is
stored as base-field coefficients. In this section an "extension scalar" means
one `L` element encoded as `<L as ExtField<F>>::EXT_DEGREE` consecutive
base-field slots. The prover allocates and consumes slots in this order:

```text
if root tensor projection is enabled:
  root_extension_opening_reduction:
    partial masks:
      num_claims * EXT_DEGREE extension scalars
    compressed reduction-sumcheck pads:
      (num_vars - log2(EXT_DEGREE)) * EXTENSION_OPENING_REDUCTION_DEGREE
      extension scalars

root_y_ring_masks:
  num_root_points * D base-field slots

if the schedule has at least one fold:
  root_level_pads:
    if the root fold is non-terminal:
      for each Stage-1 tree stage:
        for each eq-factored sumcheck round:
          EqFactoredUniPoly::coeffs_except_linear_term extension scalars
        stage child-claim masks, extension scalars
    Stage-2 compressed round pads, 3 extension scalars per round
  if the root fold is not terminal:
    root_next_w_eval_mask:
      1 extension scalar

  for each recursive fold level after the root:
    if EXT_DEGREE > 1:
      recursive_extension_opening_reduction:
        partial masks:
          EXT_DEGREE extension scalars
        compressed reduction-sumcheck pads:
          (current_opening_vars - log2(EXT_DEGREE))
          * EXTENSION_OPENING_REDUCTION_DEGREE extension scalars
    recursive_y_ring_mask:
      D base-field slots
    recursive_level_pads:
      same conditional Stage-1 / child-claim / Stage-2 layout as above
    if the recursive fold is not terminal:
      recursive_next_w_eval_mask:
        1 extension scalar
```

The round count for a level is `sumcheck_rounds(ring_dimension, next_w_len)`.
Stage 1 uses `stage1_tree_stage_shapes(rounds, b)`, where
`b = 1 << params.log_basis`. Stage 2 has degree bound `3`, so each compressed
round pad contains three stored extension coefficients. Extension-opening
reduction has degree bound `EXTENSION_OPENING_REDUCTION_DEGREE = 2`, so each
compressed reduction round contains two stored extension coefficients.

The prover allocates a scalar `next_w_eval` mask only for non-terminal fold
levels, where the level's output evaluation is used as the opening claim for a
successor recursive level. At a terminal direct-witness level, the stage-2 final
oracle uses the direct witness payload instead. Terminal proofs do not serialize
a `next_w_eval_masked` field and no terminal mask slot is allocated.

The verifier consumes the same cursor order for every slot it references, but it
finishes folded verification by requiring:

```text
hiding_witness.len() = consumed_cursor
```

No unreferenced terminal `next_w_eval` mask slot is permitted. Extra trailing
revealed hiding-witness slots are rejected.

## Root Flow

Root claimed opening values remain public:

```text
opening_i = P_i(x_j)
```

The prover absorbs the public claim incidence, commitments, opening points, and
opening values, samples gamma batching scalars, and forms one gamma-combined
`y_ring` per distinct opening point.

When root tensor projection is active, the root proof first carries masked
extension-opening reduction partials. The verifier records rows binding the
unmasked partials to the public opening values, absorbs the masked partials,
verifies the masked compressed reduction sumcheck, and later records an output
row tying the unmasked reduction final claim to the unmasked root `y_ring`
openings. When no root extension-opening reduction is present, the verifier uses
the direct root `y_ring` opening relation described below.

In ZK builds, each root `y_ring` is masked before it is serialized and absorbed:

```text
y_sent[j] = y_true[j] + g_y_root[j]
```

The verifier does not run the transparent trace equality directly on `y_sent`.
It records an R1CS row that unpacks the true `y_ring` from the revealed
`hiding_witness`:

```text
trace((y_sent[j] - g_y_root[j]) * sigma_{-1}(v_j))
  = d * sum_{claims i at point j} gamma_i * opening_i
```

In the extension-opening reduction case, the same recovered unmasked `y_ring`
opening is instead accumulated into the reduction-output row:

```text
final_claim_true =
  sum_rows factor(point(row)) * opening_from(y_sent[row] - g_y_root[row])
```

The serialized masked `y_ring` values are also used to build the masked
stage-2 wire relation claim, matching the prover's transcript-visible wire
data. The prover still constructs the root quadratic equation, ring-switch
witness, and true stage-2 round polynomials from the unmasked `y_ring` values;
`prove_zk(public_input_claim, ...)` absorbs only the transcript-visible masked
wire claim while retaining the true input claim for local round construction.

## Recursive Flow

For recursive levels over a nontrivial extension field, the proof carries a
masked extension-opening reduction before the level's `y_ring`. The verifier
records a partial-claim row against the current recursive opening, verifies the
masked compressed reduction sumcheck, and records an output row tying the
unmasked reduction final claim to the unmasked recursive `y_ring` opening.
Base-field recursive levels skip this reduction and record the direct `y_ring`
opening row.

Every non-terminal fold level serializes a masked next-witness evaluation in ZK
builds:

```text
next_w_eval_masked = Eval(next_w, r_stage2) + eta_w
```

The field is named `next_w_eval_masked`; the `next_w_eval()` accessor returns the
wire value, true in transparent builds and masked in ZK builds.

When a masked `next_w_eval` becomes the opening claim for the next recursive
level, the verifier carries the corresponding `hiding_witness` slot as
`opening_mask_index`. The next level also masks its ring-opening witness:

```text
y_sent = y_true + g_y
opening_sent = opening_true + eta_w
```

The recursive trace row is recorded as:

```text
trace((y_sent - g_y) * sigma_{-1}(opening_ring))
  = d * (opening_sent - eta_w)
```

Equivalently:

```text
trace(y_sent * sigma_{-1}(opening_ring)) - d * opening_sent
  =
trace(g_y * sigma_{-1}(opening_ring)) - d * eta_w
```

So recursive scalar handoffs are hidden in this implementation; they are not
public as in the older draft of this spec.

As in the root flow, the recursive prover uses the unmasked `y_ring` internally
for the quadratic-equation / ring-switch witness and for the true stage-2
relation. The masked `y_sent` value is the serialized wire value and is used to
form the masked stage-2 input claim.

## Stage-1 Flow

Stage 1 proves the range / norm relation using the staged eq-factored sumcheck
tree. In ZK builds, every stage uses `EqFactoredSumcheckProofMasked`, not the
transparent `EqFactoredSumcheckProof`.

For each eq-factored round, the prover computes the true inner polynomial
`q(X)`, adds a precommitted pad with the same stored shape, and absorbs only the
masked stored coefficients:

```text
q_tilde.stored = q.stored + rho.stored
stored indices = [0, 2, 3, ..., degree]
```

The linear coefficient is still omitted. For every eq-factored round, the
verifier advances the public masked scaled claim from the transcript-visible
stored coefficients and updates the hidden claim mask as a running symbolic
linear combination:

```text
eta_i = previous_coeff * eta_{i-1}
      + sum_j transition_coeff_j * rho_i,j
```

This is not an ordinary R1CS row and no verifier-local auxiliary is allocated
for the transition. The public transition is replayed directly from `q_tilde`;
the running mask LC is later used by final-oracle rows to unmask the final
scaled claim.

For staged range trees, product-stage child claims are also masked:

```text
child_claim_sent = child_claim_true + child_claim_mask
```

The verifier absorbs the masked child claims, derives the interstage batching
challenge, and carries the corresponding batched mask into the next stage.

The final stage-1 handoff is serialized in the existing field:

```text
AkitaStage1Proof::s_claim = s_claim_true + handoff_mask
```

`handoff_mask` is the accumulated mask returned by the final leaf eq-factored
sumcheck. It includes any incoming masked child-claim contribution and every
stored pad term in that leaf sumcheck's claim-mask recurrence:

```text
handoff_mask = eta_leaf_final
```

There is no separate hiding-witness slot allocated directly for
`AkitaStage1Proof::s_claim`; verifier code may name this value
`stage1_s_claim_mask`, but it is the accumulated handoff mask returned by the
final eq-factored verifier.

The verifier records exact R1CS rows for the stage-1 final oracle. For product
stages and polynomial/range evaluation, it allocates verifier-local auxiliary
variables in `akita_r1cs::ZkRelationAccumulator` and checks them during
`verify_all`.

## Stage-2 Flow

Stage 2 uses `SumcheckProofMasked`, not the transparent `SumcheckProof`. The
prover computes each true univariate round polynomial, adds a precommitted
degree-3 compressed pad, and absorbs the masked coefficient list with the linear
term omitted:

```text
g_tilde_i.stored = g_i.stored + rho_i.stored
stored indices = [0, 2, 3]
rho_i,1 = eta_{i-1} - 2 * rho_i,0 - rho_i,2 - rho_i,3
```

The transcript-visible input claim uses the masked stage-1 handoff and the
wire relation claim built from masked `y_ring` values:

```text
C_0_stage2_wire =
  batching_coeff * AkitaStage1Proof::s_claim + relation_claim_wire
```

The stage-2 prover keeps the true `s_claim` and true `relation_claim` for local
round-polynomial construction. In ZK builds, `prove_zk(public_input_claim, ...)`
absorbs the masked wire expression above as the transcript-visible input claim
without mutating the prover's true input claim.

The verifier derives the initial Stage-2 input mask symbolically:

```text
eta_{-1} = batching_coeff * handoff_mask + relation_claim_mask
```

The unmasked Stage-2 input is therefore:

```text
batching_coeff * s_claim_true + relation_claim_true
```

The implementation does not emit a separate handoff row for this expression:
`AkitaStage2Verifier::input_claim()` is already the transcript-visible masked
linear combination
`batching_coeff * AkitaStage1Proof::s_claim + relation_claim`, and the mask above
is threaded into the sumcheck as a running linear combination. Emitting a
handoff row here would only restate this construction rather than check an
independent witness relation.

That synthesized input mask is then used for Stage 2's first standard-sumcheck
round. Since compressed standard rounds omit the linear term, the masked chain
identity is enforced by reconstruction from the incoming masked claim:

```text
g_tilde_i,1 =
  C_tilde_{i-1} - 2 * g_tilde_i,0 - sum_{j >= 2} g_tilde_i,j
```

The corresponding mask transition is a public linear combination of the previous
claim mask and the stored round pads:

```text
eta_i =
  r_i * eta_{i-1}
  + (1 - 2 * r_i) * rho_i,0
  + sum_{j >= 2} (r_i^j - r_i) * rho_i,j
```

This transition is carried symbolically by the verifier; the current
plain-opening inventory does not emit one R1CS row per compressed standard
sumcheck round. The omitted linear term is defined from `C_tilde_{i-1}` and the
stored coefficients, so the usual round-chain identity
`g_tilde_i(0) + g_tilde_i(1) = C_tilde_{i-1}` has no independent residual to
check. Adding a row for this equality would only restate the decompression
definition unless the verifier also introduced explicit auxiliary wires for the
mask chain. The nontrivial standard-sumcheck relation recorded in R1CS is the
protocol-specific final oracle check after all round challenges have been
derived.

Row-count summary for compressed standard ZK sumchecks:

- No per-round R1CS rows are recorded for the sumcheck chain.
- Usually one final-oracle R1CS row is recorded per sumcheck.
- Caller-specific output rows may be recorded around the sumcheck when another
  protocol object must be tied to the unmasked reduced claim.

For Stage 2 this is roughly:

```text
1 final oracle relation
+ 0 per-round chain relations
```

Extension-opening reduction follows the same compressed-round pattern: it
propagates masks linearly through the rounds, returns one final unmasked claim
expression to the caller, and the caller records the later output check that
consumes that expression. Stage 1 is the exception only in message shape: its
eq-factored ZK sumchecks use `EqFactoredUniPoly` pads and the eq-factored
omitted-linear transition, then record their product/range final relations.

It consumes the compressed-round pad slots to build the linear mask for the
final masked claim:

```text
final_claim_true =
  final_claim_wire - eta_final
```

This final-claim unmasking is represented as a symbolic linear combination and
inlined into the final oracle check; it is not emitted as a standalone R1CS row.
Then `AkitaStage2Verifier::record_final_relation` records the final oracle as an
R1CS row. If the level output feeds another recursive level, it unpacks the true
witness evaluation from `next_w_eval_masked`:

```text
w_lc = next_w_eval_masked - eta_w
```

The recorded relation is:

```text
w_lc *
  [batching_coeff * eq(r_stage1, r_stage2) * (w_lc + 1)]
=
final_claim_true - alpha_eval(r_y) * row_eval(r_x) * w_lc
```

For terminal levels, `w_lc` is the direct witness evaluation computed by the
verifier from the final packed witness payload. Terminal proofs do not serialize
`next_w_eval_masked`; there is no terminal next-witness handoff field to bind.

## Relation Accumulator

`akita-r1cs` owns the deferred plain-opening relation system. Its
`ZkRelationAccumulator` stores ordinary R1CS rows and auxiliary-generation rows:

```text
<A, X> * <B, X> = <C, X>
aux = <A, X> * <B, X>
```

Hidden variables are indices into `ZkHidingProof::hiding_witness`. Auxiliary
variables are verifier-local values generated while checking the accumulator.

At the end of folded proof verification, the verifier runs:

```text
zk_relations.verify_all(&proof.zk_hiding.hiding_witness)
```

The current plain verifier checks every accumulated row it records. It does not
skip R1CS rows with auxiliary variables; those auxiliaries are synthesized during
`verify_all`. `verify_all` also lifts the revealed base-field hiding witness into
the verifier's challenge field before evaluating rows. The accumulator rejects
ordinary and auxiliary-generation rows with fewer than two effective private
variables, where private variables are hidden-witness slots or verifier-local
auxiliary slots after zero coefficients and duplicate-variable cancellations are
removed. The sumcheck crates use the accumulator only behind `feature = "zk"`;
transparent sumcheck drivers do not depend on the R1CS API.

For compressed standard sumchecks, `ZkRelationAccumulator` consumes the
compressed round-mask slots and returns the next claim mask as a linear
combination, but it intentionally records no per-round R1CS row. The compressed
message format already forces the masked round sum to equal the incoming masked
claim. The rows associated with such a sumcheck are instead the surrounding
semantic rows: an output row when the reduced claim is tied to another object,
and the final oracle row that checks the unmasked final claim against the
protocol oracle. Eq-factored Stage-1 sumchecks now follow the same running-mask
style for their claim-mask transition:
`masked_eq_factored_claim_mask` returns the next hidden claim mask as a linear
combination and records no per-round auxiliary row. Stage-1 still records the
semantic product/range final rows that tie the unmasked eq-factored final claims
to the protocol oracle.
Future work is to prove this same row inventory without revealing
`hiding_witness`.

## Transcript Rules

Transparent builds keep the existing proof shapes. Non-terminal folded
transparent and ZK builds both serialize the stage-2 next-witness evaluation
handoff in the proof; the ZK build serializes the masked wire value. Terminal
direct-witness proofs do not serialize a next-witness evaluation handoff.

In folded ZK builds:

- `u_blind` is absorbed under `ABSORB_ZK_HIDING_COMMITMENT` before root
  challenges that depend on hidden wire data.
- Root opening claims remain public and are absorbed as before.
- Root extension-opening reduction partials and compressed reduction-sumcheck
  messages are masked when that reduction is present.
- Root and recursive `y_ring` transcript messages are masked wire values.
- Recursive extension-opening reduction partials and compressed
  reduction-sumcheck messages are masked when the extension reduction is
  present.
- Stage-1 absorbs masked eq-factored round payloads and masked interstage child
  claims.
- `AkitaStage1Proof::s_claim` is absorbed under the existing label, but its wire
  value is masked.
- Stage-2 absorbs masked compressed round polynomials.
- `AkitaStage2Proof::next_w_eval_masked` is the ZK proof field for recursive
  handoffs, accessed through `next_w_eval()`.

For non-terminal fold levels, the verifier carries `next_w_eval()` and the
corresponding hidden mask slot into the successor `RecursiveVerifierState`. The
successor level's opening, y-ring, and stage-2 relations then bind that recursive
handoff. Terminal direct-witness levels have no serialized
`next_w_eval_masked` field.

The revealed `hiding_witness`, `b_blinding_digits`, and unmasked true values are
not absorbed before the masked messages they support are fixed.

## Implementation Notes

### `akita-types`

- `AkitaBatchedProof` has `#[cfg(feature = "zk")] zk_hiding:
  ZkHidingProof<F>` serialized before the root proof.
- `ZkHidingProof` contains `u_blind`, `hiding_witness`, and
  `b_blinding_digits`.
- `AkitaStage1StageProof` is feature-split:
  - non-ZK: `sumcheck_proof: EqFactoredSumcheckProof<F>`;
  - ZK: `sumcheck_proof_masked: EqFactoredSumcheckProofMasked<F>`.
- `AkitaStage2Proof` is feature-split:
  - non-ZK: `sumcheck_proof: SumcheckProof<F>` and `next_w_eval`;
  - ZK: `sumcheck_proof_masked: SumcheckProofMasked<F>` and
    `next_w_eval_masked`.
- `AkitaStage2Proof::next_w_eval()` returns the wire value for non-terminal
  handoffs in the active build. Terminal direct-witness proofs do not carry this
  handoff field.
- `AkitaLevelProof::y_ring` remains the proof field name; in ZK builds it
  carries the masked wire value.

### `akita-r1cs`

- Owns the plain-opening R1CS building blocks:
  `ZkR1csVariable`, `ZkR1csTerm`, `ZkR1csLinearCombination`, and
  `ZkRelationAccumulator`.
- Provides the masked standard-sumcheck and masked eq-factored running-mask
  helpers consumed by the ZK sumcheck verifier drivers.
- Keeps verifier-local auxiliary wires inside the accumulator. `verify_all`
  synthesizes those auxiliaries while checking rows against the revealed
  `hiding_witness`.

### `akita-sumcheck`

- `src/drivers.rs` is now only the driver module/export surface.
- `src/drivers/standard.rs` contains the transparent standard sumcheck
  `prove`/`verify` extension traits and, behind `feature = "zk"`, the masked
  standard sumcheck `prove_zk`/`verify_zk` extensions plus
  `ZkSumcheckFinalRelation`.
- `src/drivers/eq_factored.rs` contains the transparent eq-factored sumcheck
  `prove`/`verify` extension traits and, behind `feature = "zk"`, the masked
  eq-factored `prove_zk`/`verify_zk` extensions plus
  `ZkEqFactoredFinalRelation`.
- ZK-only proof payload types and driver extension exports are feature-gated.
  Transparent builds expose only the transparent proof types and driver traits.

### `akita-prover`

- `build_zk_hiding_context` samples root extension-reduction masks, root y-ring
  masks, per-level stage-1 pads, stage-1 child-claim masks, stage-2 pads,
  recursive extension-reduction masks, recursive y-ring masks, and non-terminal
  `next_w_eval` masks in the cursor order described above.
- `commit_zk_hiding_witness` commits the padded hiding witness as `u_blind`
  using the original root fold parameters (`root_step.params`) adjusted to the
  hiding-witness length, then adds independent B-side blinding digits.
- Root `y_rings` are masked before serialization; the quadratic-equation and
  ring-switch prover logic still uses the unmasked `y_rings`.
- Recursive `y_ring` values are masked before serialization; the
  quadratic-equation and true stage-2 relation still use the unmasked `y_ring`.
- Stage 1 calls `prove_zk` with precommitted eq-factored pads and child-claim
  masks, and stores the masked `s_claim`.
- Stage 2 calls `prove_zk` with the masked wire expression as the
  transcript-visible input claim plus precommitted compressed-univariate pads, and
  stores `next_w_eval + eta_w` for non-terminal handoffs in ZK builds.

### `akita-verifier`

- Folded verification rejects empty `u_blind` / `hiding_witness`, recomputes the
  hiding-factor commitment from revealed `hiding_witness` and
  `b_blinding_digits`, absorbs `u_blind`, and creates an
  `akita_r1cs::ZkRelationAccumulator`.
- The root verifier unpacks root extension-reduction and y-ring masks, then
  records root reduction/output or direct trace-pin R1CS rows.
- The recursive verifier unpacks the current opening mask, optional
  extension-reduction masks, and current y-ring mask before recording recursive
  reduction/output or direct trace-pin rows.
- Stage-1 and stage-2 verification consume only masked proof types in ZK builds.
- After each non-terminal fold-level Stage-2 verification, the verifier carries
  `stage2.next_w_eval()` and its mask into the next recursive opening state.
- Nonlinear final oracle checks are recorded as R1CS rows and checked by the
  plain verifier using revealed `hiding_witness` plus verifier-local
  auxiliaries.
- The verifier requires the revealed `hiding_witness` length to exactly equal the
  consumed cursor, so extra trailing slots are rejected.

## Acceptance Criteria

- Folded ZK proofs emit and transcript-bind a separate `u_blind`.
- Real commitments and hiding-factor commitments use dedicated B-side Ajtai
  blinding digits.
- Root and recursive `y_ring` proof fields are masked in ZK builds.
- Stage-1 and stage-2 sumcheck proof messages are masked by `hiding_witness`
  pads.
- Stage-1 tree child claims are masked.
- `AkitaStage1Proof::s_claim` is masked in ZK builds.
- Non-terminal `AkitaStage2Proof` values use `next_w_eval_masked` in ZK builds;
  recursive handoffs are interpreted as masked wire values. Terminal
  direct-witness proofs do not carry a next-witness handoff field.
- Recursive Stage-2 next-witness evaluation handoffs are carried with their mask
  into the successor recursive verifier state.
- The ZK proof path does not carry redundant transparent sumcheck proofs.
- The plain verifier recomputes and checks the `u_blind` hiding-factor
  commitment from revealed `hiding_witness` and `b_blinding_digits`.
- The plain verifier checks the currently recorded R1CS inventory against the
  revealed `hiding_witness`.
- The folded verifier rejects any extra trailing `hiding_witness` slots beyond
  the consumed cursor.
- Eq-factored Stage-1 mask propagation uses a running LC and does not allocate a
  verifier-local auxiliary for every round.
- The spec does not claim full zero-knowledge until the plain opening is
  replaced by a proof of the same relation inventory.

## Future Work

1. Prove the `u_blind` commitment opening instead of relying on the revealed
   payload.
2. Replace `verify_all(&hiding_witness)` with Spartan / LNP22 / tail-sigma
   verification over the same relation inventory.
3. Remove `hiding_witness` and Ajtai blinding material from the serialized proof
   once those relations are proven.
