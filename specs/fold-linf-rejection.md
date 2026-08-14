# Spec: Folded-Witness ∞-Norm Grinding

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-06-10 |
| Status | implemented (#189), revised for the signed-sparse cutover |
| PR | [#189](https://github.com/LayerZero-Labs/akita/pull/189) |

## Summary

For a folded witness

```text
z = Σ_i c_i · s_i,
```

the next recursive level commits balanced base-`b` digits of `z`. The scheduled
digit count `K = num_digits_fold` must therefore cover the honest response while
remaining small enough to avoid unnecessary Ajtai columns and sum-check
variables.

Every supported fold challenge is a [`SparseChallengeConfig`]: its nonzero
positions and magnitudes are sampled without replacement, and every nonzero
coefficient receives an independent uniform sign. Conditioned on the positions
and magnitudes, each output coordinate of `z` is a Rademacher sum. Akita uses
that concentration bound to size the honest cap as

```text
cap = min(β_inf, t*),
```

where `β_inf` is the deterministic negacyclic product envelope and `t*` is the
sub-Gaussian tail threshold. The prover searches sequential Fiat-Shamir nonces
until the realized response fits the scheduled cap. The verifier replays the
accepted nonce and enforces the scheduled response representation; it does not
treat the nonce itself as evidence of the norm bound.

There is one sizing and grinding flow for every valid sparse configuration.

This honest coefficient-`L∞` sizing flow is separate from the A-role security
route. The ordinary A route prices the verifier-enforced balanced-digit
envelope in the coefficient-`L∞` SIS table. Eligible calibrated later folds may
instead carry the exact physical norm proof and Euclidean SIS route defined by
[`selective-l2-fold-security-sizing.md`](selective-l2-fold-security-sizing.md).

## Bounds

Let:

- `B = num_claims · num_live_blocks` be the number of folded blocks;
- `N` be the logical folded-coefficient count used for the union bound;
- `s_inf = ‖s‖_∞` be the honest witness coefficient bound;
- `c_l2_sq_max = count_pm1 + 4 · count_pm2`.

The last equality is exact because a sparse challenge contains `count_pm1`
coefficients of magnitude one and `count_pm2` coefficients of magnitude two.
Fresh independent signs make the per-coordinate variance proxy
`c_l2_sq_max · s_inf²` per block. Akita computes the conservative integer bound

```text
ln_term = ceil(ln(2 · N / (1 - p)))
t*²     = 2 · B · c_l2_sq_max · s_inf² · ln_term
t*      = ceil(sqrt(t*²))
```

with the offline per-group sizing convention `p = 1/8`. All arithmetic on this
path is checked integer arithmetic.

The deterministic envelope is

```text
β_inf = B · min(‖c‖_∞ · ‖s‖_1, ‖c‖_1 · ‖s‖_∞).
```

`fold_witness_linf_cap` returns `min(β_inf, t*)` together with `t*`.
The balanced-digit policy sizes directly from this cap. It does not discount
the cap before selecting a digit depth. The target `p` is an offline schedule-sizing
constant. Its protocol effect is bound through the generated schedule,
including its committed digit depths and response limits.

## Fiat-Shamir grinding

Each fold proof carries one `u32` nonce. For candidate nonces
`0, 1, ...`, the prover absorbs the nonce into the group-local fold-challenge
domain and samples the sparse challenge stream. The verifier performs the same
absorption and sampling for the proof's accepted nonce.

`FoldLinfProtocolBinding` binds only the protocol-wide nonce contract:

- the exclusive probe cap (`4096`);
- the nonce wire width (`4` bytes).

Nonce values outside the bound are rejected. Exhausting the bound returns a
prover error; it does not create an unbounded loop or a verifier panic.

Repeated nonce trials are accounted for by the adversary's total random-oracle
query budget in the Fiat-Shamir reduction. There is no separate 12-bit
soundness debit per fold. For a bad-challenge fraction `epsilon`, `q` nonce
queries increase success to at most `q * epsilon`, while also costing `q`
oracle queries. The same query factor applies if a prover varies other
pre-challenge randomness. See
[Polynomial commitments and binding](../book/src/foundations/pcs-and-binding.md#fiat-shamir-queries-and-fold-nonces).

For a multi-group fold, all groups share the same candidate nonce and must pass
together. The `p = 1/8` calculation is a marginal, per-group sizing guarantee.
It does not assert independence between groups or a joint expected-attempt
bound. The hard probe cap is the protocol-wide termination guarantee.

## Ownership

The current ownership boundaries are:

- `akita-challenges` samples fixed-weight signed sparse challenges and exposes
  `challenge_l2_sq_max`;
- `akita-types::sis` owns `β_inf`, the tail calculation, and balanced-digit
  sizing;
- the planner materializes `num_digits_fold` and response limits in generated
  schedules;
- the prover searches nonces using those scheduled limits;
- the verifier admits the nonce, replays the challenge stream, and checks the
  committed digits or terminal response against the scheduled representation.

Intermediate and terminal admission remain intentionally different. An
intermediate response is admitted by its balanced digit representation. A
terminal response is sent in clear and checked against the terminal group's
SIS-certified raw coefficient cap. See
[`heterogeneous-group-source-contracts.md`](heterogeneous-group-source-contracts.md)
for the current group and terminal ownership model.

## Security and correctness invariants

1. **One challenge law.** Every valid `SparseChallengeConfig` has fixed
   magnitudes and independent signs, so the same conditional Rademacher proof
   applies. `challenge_l2_sq_max` is the single variance source.
2. **One sizing source.** Planner and prover consume the same SIS primitives and
   scheduled digit depths. Security pricing does not reconstruct a competing
   cap.
3. **Transcript symmetry.** Prover and verifier absorb the same nonce before
   squeezing the same group-local challenge stream.
4. **Structural verification.** The verifier relies on the digit-range or
   terminal-response checks, not on an unverifiable claim that grinding found a
   small response.
5. **Bounded work.** Probing is sequential and capped. Malformed nonces and
   arithmetic overflow return `AkitaError` or `SerializationError`.
6. **Descriptor consistency.** The setup descriptor binds the nonce wire
   contract, while the effective plan binds the derived digit depths and
   response limits.

## Required regression coverage

- exact `challenge_l2_sq_max` values for sparse configurations;
- monotonicity, zero-input rejection, and overflow rejection in tail sizing;
- `min(β_inf, t*)` and universal digit-depth tests;
- generated-schedule drift checks;
- nonce wire round-trip and out-of-range rejection;
- prover/verifier transcript-event equality;
- end-to-end recursive and terminal prove/verify;
- tamper rejection for committed fold handles and terminal responses.

## Non-goals

- realized-norm evidence carried by the nonce;
- online lattice estimation;
- operator-norm rejection or calibrated empirical thresholds;
- using the honest tail cap directly for A-role MSIS collision pricing.

The ordinary coefficient-`L∞` A-role route continues to use the
verifier-enforced balanced-digit envelope.
