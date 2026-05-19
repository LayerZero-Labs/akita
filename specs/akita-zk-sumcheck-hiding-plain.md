# Spec: Akita ZK Sumcheck Hiding Plain Opening

| Field     | Value                  |
| --------- | ---------------------- |
| Author(s) | Amirhossein Khajehpour |
| Created   | 2026-05-12             |
| Updated   | 2026-05-19             |
| Status    | implemented in branch  |
| PR        |                        |

## Summary

This branch implements a plain-opening version of Akita's sumcheck-hiding ZK
path. The proof carries a separate hiding-factor commitment `u_blind`, reveals
the committed `hiding_witness`, and verifies deferred R1CS rows from the
`akita-r1cs` crate directly against that revealed witness.

This is not the final zero-knowledge protocol. It fixes the transcript shape and
the relation inventory, but the hiding material remains public until a later
Spartan / LNP22 / tail-sigma proof replaces the plain opening.

In folded `feature = "zk"` builds, the implementation currently hides:

- root gamma-combined `y_ring` messages, one ring mask per distinct root opening
  point;
- recursive-level `y_ring` messages, one ring mask per recursive fold level;
- stage-1 eq-factored sumcheck round messages;
- stage-1 tree child claims between product layers;
- `AkitaStage1Proof::s_claim`;
- stage-2 full-univariate sumcheck round messages;
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

The verifier currently rejects empty `u_blind` / `hiding_witness` and absorbs
`u_blind` before root challenges. It does not yet recompute or prove the
`u_blind` commitment-opening equation; `b_blinding_digits` is serialized for the
plain payload but is not part of the current folded verifier's R1CS check.

## Hiding Witness Layout

The implementation relies on a single cursor through `hiding_witness`. The
prover allocates and consumes slots in this order:

```text
root_y_ring_masks:
  num_root_points * D field elements

if the schedule has at least one fold:
  root_level_pads:
    for each Stage-1 tree stage:
      for each eq-factored sumcheck round:
        EqFactoredUniPoly::coeffs_except_linear_term
      stage child-claim masks
    Stage-2 full round pads, 4 coefficients per round
  root_next_w_eval_mask:
    1 field element

  for each recursive fold level after the root:
    recursive_y_ring_mask:
      D field elements
    recursive_level_pads:
      same Stage-1 / child-claim / Stage-2 layout as above
    recursive_next_w_eval_mask:
      1 field element
```

The round count for a level is `sumcheck_rounds(ring_dimension, next_w_len)`.
Stage 1 uses `stage1_tree_stage_shapes(rounds, b)`, where
`b = 1 << params.log_basis`. Stage 2 has degree bound `3`, so each full-round pad
contains four coefficients.

The prover currently allocates a scalar `next_w_eval` mask for every fold level,
including the last fold. The verifier consumes that scalar only when the level's
output evaluation is used as the opening claim for another recursive level. At a
terminal direct-witness level, the stage-2 final relation uses the direct witness
instead, so the terminal `next_w_eval_masked` field and its final mask slot are
not referenced by the current R1CS inventory.

The verifier consumes the same cursor order for every slot it references, but it
finishes folded verification by requiring:

```text
hiding_witness.len() = consumed_cursor + 1
```

The `+ 1` is the single terminal `next_w_eval` mask slot described above. Extra
trailing revealed hiding-witness slots are rejected.

## Root Flow

Root claimed opening values remain public:

```text
opening_i = P_i(x_j)
```

The prover absorbs the public claim incidence, commitments, opening points, and
opening values, samples gamma batching scalars, and forms one gamma-combined
`y_ring` per distinct opening point.

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

The serialized masked `y_ring` values are also used to build the masked
stage-2 wire relation claim, matching the prover's transcript-visible wire
data. The prover still constructs the root quadratic equation, ring-switch
witness, and true stage-2 round polynomials from the unmasked `y_ring` values;
only the transcript-visible input claim is overridden to the masked wire claim.

## Recursive Flow

Every fold level serializes a masked next-witness evaluation in ZK builds:

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

The linear coefficient is still omitted. The verifier advances the masked
scaled-claim state with the same transition coefficients used by the transparent
eq-factored driver, while separately synthesizing a verifier-local auxiliary
wire for the next mask:

```text
eta_i = previous_coeff * eta_{i-1}
      + sum_j transition_coeff_j * rho_i,j
```

For every eq-factored round, the verifier also records a deferred R1CS row for
the masked transition itself. The `eta_i` on the left is the distinct auxiliary
wire above, not the same linear combination repeated syntactically:

```text
C_tilde_i - [previous_coeff * C_tilde_{i-1}
             + sum_j transition_coeff_j * q_tilde_i,j]
  =
eta_i - [previous_coeff * eta_{i-1}
         + sum_j transition_coeff_j * rho_i,j]
```

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

`handoff_mask` is derived from the final eq-factored round pad by evaluating the
stored terms at that round's challenge:

```text
handoff_mask = rho_leaf_final_round_stored(r_leaf_last)
```

There is no separate hiding-witness slot allocated directly for
`AkitaStage1Proof::s_claim`; verifier code may name this value
`stage1_s_claim_mask`, but it is the handoff mask derived from the final
eq-factored pad.

The verifier records exact R1CS rows for the stage-1 final oracle. For product
stages and polynomial/range evaluation, it allocates verifier-local auxiliary
variables in `akita_r1cs::ZkRelationAccumulator` and checks them during
`verify_all`.

## Stage-2 Flow

Stage 2 uses `SumcheckProofMasked`, not the transparent `SumcheckProof`. The
prover computes each true full univariate round polynomial, adds a precommitted
degree-3 pad, and absorbs the full masked coefficient list:

```text
g_tilde_i(X) = g_i(X) + rho_i(X)
rho_i(X) = rho_i,0 + rho_i,1 X + rho_i,2 X^2 + rho_i,3 X^3
```

The transcript-visible input claim uses the masked stage-1 handoff and the
wire relation claim built from masked `y_ring` values:

```text
C_0_stage2_wire =
  batching_coeff * AkitaStage1Proof::s_claim + relation_claim_wire
```

The stage-2 prover keeps the true `s_claim` and true `relation_claim` for local
round-polynomial construction. In ZK builds, `with_input_claim` changes only the
transcript-visible input claim to the masked wire expression above.

The verifier records deferred handoff relations that synthesize the true
Stage-1 handoff and the initial Stage-2 input mask:

```text
s_claim_true = AkitaStage1Proof::s_claim - handoff_mask
eta_{-1} = batching_coeff * handoff_mask + relation_claim_mask
```

It then checks that the unmasked stage-2 input equals:

```text
batching_coeff * s_claim_true + relation_claim_true
```

The implementation records this semantic handoff as one combined R1CS row after
synthesizing verifier-local auxiliaries, rather than as separate rows for each
displayed equality.

That synthesized input mask is then used for Stage 2's first standard-sumcheck
round. Round 0 must satisfy the same masked chain identity as later rounds:

```text
eta_0 = rho_0(r_0)
g_tilde_0(0) + g_tilde_0(1) - C_tilde_{-1}
  =
rho_0(0) + rho_0(1) - eta_{-1}
```

For every later round `i > 0`, the generic masked standard-sumcheck verifier
checks the masked chain identity:

```text
g_tilde_i(0) + g_tilde_i(1) - C_tilde_{i-1}
  =
rho_i(0) + rho_i(1) - eta_{i-1}
```

It also consumes the full-round pad slots to build the linear mask for the final
masked claim:

```text
final_claim_true =
  final_claim_wire - rho_last(r_last)
```

This final-claim unmasking is represented as a symbolic linear combination and
inlined into the final oracle check; it is not emitted as a standalone R1CS row.
Then `AkitaStage2Verifier::record_final_relation` records the final oracle as an
R1CS row. If the level output is recursive, it also unpacks the true witness
evaluation from `next_w_eval_masked`:

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
verifier from the final packed witness payload. In that case, the serialized
`next_w_eval_masked` field is not the source of the final oracle value.

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
`verify_all`. The sumcheck crates use the accumulator only behind
`feature = "zk"`; transparent sumcheck drivers do not depend on the R1CS API.
Future work is to prove this same row inventory without revealing
`hiding_witness`.

## Transcript Rules

Transparent builds keep the existing proof shapes. Folded transparent and ZK
builds both absorb the stage-2 next-witness evaluation handoff; the ZK build
absorbs the masked wire value.

In folded ZK builds:

- `u_blind` is absorbed under `ABSORB_ZK_HIDING_COMMITMENT` before root
  challenges that depend on hidden wire data.
- Root opening claims remain public and are absorbed as before.
- Root and recursive `y_ring` transcript messages are masked wire values.
- Stage-1 absorbs masked eq-factored round payloads and masked interstage child
  claims.
- `AkitaStage1Proof::s_claim` is absorbed under the existing label, but its wire
  value is masked.
- Stage-2 absorbs masked full round polynomials.
- `AkitaStage2Proof::next_w_eval_masked` is the ZK proof field for recursive
  handoffs, accessed through `next_w_eval()`.
- Each fold level absorbs the stage-2 next-witness evaluation wire value under
  `ABSORB_STAGE2_NEXT_W_EVAL` after the stage-2 sumcheck. In ZK builds this is
  `next_w_eval_masked`; in transparent builds it is the true `next_w_eval`.

This binds the recursive opening claim before the transcript is used for any
successor fold level.

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
- `AkitaStage2Proof::next_w_eval()` returns the wire value for the active build.
- `AkitaLevelProof::y_ring` remains the proof field name; in ZK builds it
  carries the masked wire value.

### `akita-r1cs`

- Owns the plain-opening R1CS building blocks:
  `ZkR1csVariable`, `ZkR1csTerm`, `ZkR1csLinearCombination`, and
  `ZkRelationAccumulator`.
- Provides the masked standard-sumcheck and masked eq-factored round relation
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

- `build_zk_hiding_context` samples root y-ring masks, per-level stage-1 pads,
  stage-1 child-claim masks, stage-2 pads, recursive y-ring masks, and per-level
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
- Stage 2 calls `prove_zk` with precommitted full-univariate pads, overrides the
  transcript-visible input claim to the masked wire expression, and stores
  `next_w_eval + eta_w` in ZK builds.
- After Stage 2, the prover absorbs the `next_w_eval()` wire value under
  `ABSORB_STAGE2_NEXT_W_EVAL` before recursion continues.

### `akita-verifier`

- Folded verification rejects empty `u_blind` / `hiding_witness`, absorbs
  `u_blind`, and creates an `akita_r1cs::ZkRelationAccumulator`.
- The root verifier unpacks root y-ring masks and records root trace-pin R1CS
  rows.
- The recursive verifier unpacks both the current opening mask and current
  y-ring mask before recording recursive trace-pin rows.
- Stage-1 and stage-2 verification consume only masked proof types in ZK builds.
- After each fold-level Stage-2 verification, the verifier absorbs
  `stage2.next_w_eval()` under `ABSORB_STAGE2_NEXT_W_EVAL`.
- Nonlinear final oracle checks are recorded as R1CS rows and checked by the
  plain verifier using revealed `hiding_witness` plus verifier-local
  auxiliaries.
- The verifier requires the revealed `hiding_witness` length to equal the
  consumed cursor plus the single terminal `next_w_eval` mask slot, so extra
  trailing slots are rejected.

## Acceptance Criteria

- Folded ZK proofs emit and transcript-bind a separate `u_blind`.
- Real commitments and hiding-factor commitments use dedicated B-side Ajtai
  blinding digits.
- Root and recursive `y_ring` proof fields are masked in ZK builds.
- Stage-1 and stage-2 sumcheck proof messages are masked by committed
  `hiding_witness` pads.
- Stage-1 tree child claims are masked.
- `AkitaStage1Proof::s_claim` is masked in ZK builds.
- `AkitaStage2Proof` uses `next_w_eval_masked` in ZK builds; recursive handoffs
  are interpreted as masked wire values.
- Stage-2 next-witness evaluation handoffs are absorbed under
  `ABSORB_STAGE2_NEXT_W_EVAL` before successor recursive challenges.
- The ZK proof path does not carry redundant transparent sumcheck proofs.
- The plain verifier checks the currently recorded R1CS inventory against the
  revealed `hiding_witness`.
- The folded verifier rejects extra trailing `hiding_witness` slots beyond the
  one terminal mask slot allocated by the prover layout.
- The spec does not claim full zero-knowledge until the plain opening is
  replaced by a proof of the same relation inventory.

## Future Work

1. Prove the `u_blind` commitment opening instead of relying on the revealed
   payload.
2. Replace `verify_all(&hiding_witness)` with Spartan / LNP22 / tail-sigma
   verification over the same relation inventory.
3. Remove `hiding_witness` and Ajtai blinding material from the serialized proof
   once those relations are proven.
4. Avoid allocating or serializing unused terminal `next_w_eval` masks, or make
   terminal handling explicitly prove/use the field.
