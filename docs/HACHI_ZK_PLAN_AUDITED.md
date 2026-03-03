# Making Hachi Zero-Knowledge (Audited Rewrite)

## Status

Design plan only. Not yet implemented.

Current code is knowledge-sound but not zero-knowledge. This note describes a
full-cutover path to ZK.

## Scope

This plan targets the current single-modulus Hachi flow:

`R_q relations -> ring switch at random alpha -> sumcheck over F`.

It does not cover two-field (`q`/`q'`) migration details for Jolt data proofs.
For that, see `docs/HACHI_FOR_JOLT.md` and
`docs/TWO_FIELD_OPENING_D_GT_1_CLEAN.md`.

## Notation

- `D`: ring degree (power of two)
- `b = 2^log_basis` (current default `log_basis = 4`, so `b = 16`)
- `y`: claimed opening value in the field
- `y_ring`: ring-level evaluation sent in the proof
- `v_open`: ring element derived from the inner opening point
- `trace(u)`: implemented as `D * ct(u)`

---

## 1. What the current implementation reveals

The proof currently serializes these components:

- `y_ring`
- `v` (the vector `D * w_hat` from `QuadraticEquation`)
- sumcheck round polynomials
- `sumcheck_aux.w` (raw witness coefficients)
- `w_commitment`

Leakage-focused view:

| Proof component | Current check | Leakage risk today |
|---|---|---|
| `y_ring` | Verifier checks `trace(y_ring * sigma_{-1}(v_open)) = D * y` | One linear constraint only; remaining coefficients leak extra linear info |
| `v` (`proof.v`) | Consumed by quadratic-equation and ring-switch checks | Deterministic linear image of decomposed witness data |
| Sumcheck round polynomials | Verified with degree bound and round consistency | Deterministic functions of witness tables; not masked yet |
| `sumcheck_aux.w` | Directly used by verifier to build expected output claim | Full raw witness reveal (`z || r` coefficients) |
| `w_commitment` | Absorbed into transcript | Currently not opened/verified against `sumcheck_aux.w` |

Important: there is no separate prover message called "final sumcheck claim".
The verifier derives the final claim from round polynomials and compares it to
the expected oracle evaluation.

---

## 2. Ring-switch boundary (precise statement)

Hachi's key architectural leverage is still valid:

- ring-level constraints are reduced by evaluating at random `alpha`
- sumcheck itself runs over field tables (`w_evals`, `m_evals_x`,
  `alpha_evals_y`)

But the boundary should be stated precisely:

- it is a computational-domain boundary (ring operations -> field operations)
- it is not a semantic boundary where ring-switch constraints disappear

Post-switch sumcheck still enforces ring-switch-derived consistency and range
constraints, just encoded as field relations.

---

## 3. ZK cutover design

### 3.1 Hide Ajtai-style commitments with MLWE noise (M0)

Current commitments are binding-oriented. To make them hiding, add noise terms:

```
u            = B * t_hat + E_u * r_u
v            = D * w_hat + E_v * r_v
w_commitment = ...       + E_w * r_w
```

Where `E_*` are public matrices and `r_*` are sampled from the chosen noise
distribution. The `r_*` values become witness components for later consistency
proofs.

Implementation note: once this is added, relation tests that currently assert
plain equalities (`D * w_hat = v`, `B * t_hat = u`) must be updated to include
the `E * r` terms.

### 3.2 Mask `y_ring` under the relation actually checked by code (M1)

Current verifier relation:

```
trace(y_ring * sigma_{-1}(v_open)) = D * y
```

So masks must satisfy:

```
trace(l_i * sigma_{-1}(v_open)) = 0
```

Protocol sketch (adapted from Greyhound-style constant-term masking):

1. Sample masks `l_1, ..., l_L` in the above kernel.
2. Commit to their decomposed form in the first round.
3. Verifier samples `alpha_1, ..., alpha_L`.
4. Prover sends `j_i = l_i + alpha_i * y_ring`.
5. Verifier checks
   `trace(j_i * sigma_{-1}(v_open)) = alpha_i * D * y`.

Soundness contribution is about `q^{-L}` (for challenge field size `q`), so
pick `L` from target security, not as a hard-coded constant.

### 3.3 Field-level masked sumcheck with carry-forward (M2)

Current fused verifier check is of the form:

```
batching_coeff * eq(tau0, r) * range_check(w(r), b)
+ w(r) * alpha_val(r_y) * m_val(r_x)
```

The prover/verifier enforce round polynomial degree bound `2 * b` (current
`b = 16`, so bound 32). Masking plan:

- per-round random mask polynomial `rho_i(X)` with degree `<= 2 * b`
- enforce zero-sum condition `rho_i(0) + rho_i(1) = 0`
- send `g_i_tilde = g_i + rho_i`

Carry-forward strategy:

- append mask parameters to witness state and enforce consistency constraints in
  the fused sumcheck relation
- avoid separate mask commitment/opening objects when consistency can be folded
  into the existing base-case check

Implementation constraints to respect in current code path:

- `w.len()` must remain divisible by `D`
- current compact path expects coefficients in `[-b/2, b/2 - 1]` for `Vec<i8>`
  conversion; either encode masks compatibly or refactor compact/range path
- changing witness width changes `num_u` and table shapes; `m_evals_x` sizing
  and relation dimensions must be updated together

### 3.4 Remove raw witness from proof via recursive opening (M3)

Replace `sumcheck_aux.w` with recursive PCS opening at the final sumcheck point.
The verifier should check masked final evaluations via the recursive opening,
not by receiving raw witness coefficients.

### 3.5 Bind `w_commitment` to the witness actually checked (M3 prerequisite)

Today, `w_commitment` is transcript-bound but not explicitly opened against
`sumcheck_aux.w`. The ZK cutover should close this gap:

- either verify a commitment opening that binds `w_commitment` to the witness
  used in final checks
- or remove/repurpose `w_commitment` until such a check exists

### 3.6 Prove well-formedness of new witness pieces

All added witness elements (`r_u`, `r_v`, `r_w`, `y_ring` masks, sumcheck
masks) must be tied back to commitments and algebraic constraints. Ring-level
and field-level pieces should be placed in the layer where they are natively
checked (pre-switch vs post-switch) to avoid accidental cross-domain leakage.

---

## 4. Security target and proof obligations (M4)

### 4.1 Target notion

Target after M0-M4: computational HVZK in ROM (Fiat-Shamir), plus existing
knowledge soundness/binding assumptions.

Current status: not ZK yet.

### 4.2 Obligation table

| Item | Current status | Target status |
|---|---|---|
| Commitment hiding (`u`, `v`, `w_commitment`) | Not hiding | MLWE-based computational hiding |
| `y_ring` privacy | Leaks extra linear info | Masked except for required opening relation |
| Sumcheck witness privacy | Raw `sumcheck_aux.w` is sent | Masked round flow + recursive final opening |
| End-to-end ZK argument | Not formalized in code/docs | Modular IOR-style composition with explicit simulator obligations |

### 4.3 Composition caveat

Using IOR composition (for example CFW-style frameworks) is reasonable, but
only after each local simulator obligation is explicit and discharged. In
particular, upstream simulated outputs must remain consistent with downstream
checks.

---

## 5. Parameter and overhead worksheet (scenario-level)

Treat numbers below as scenario estimates until M2/M3 representation choices
are implemented.

- `y_ring` masking reps:
  - choose `L` such that `q^{-L}` is below target soundness slack

- Sumcheck mask witness growth:
  - `extra_coeffs ~= num_rounds * dof_per_round`
  - with current fused degree bound `2b`, a zero-sum round mask has about `2b`
    degrees of freedom
  - example: `num_rounds ~= 36`, `b = 16` -> `extra_coeffs ~= 36 * 32 = 1152`
    field elements before packing/compression choices

- Proof-size and prover-time impact:
  - depends on mask encoding path, recursive opening format, and whether compact
    `i8` representation is retained
  - avoid treating rough multipliers as final until M2/M3 are benchmarked

- Base-case caveat:
  - if recursion terminates in direct lattice opening, base-case masking may
    still require rejection-sampling style costs and parameter feedback

---

## 6. Milestones with acceptance criteria

### M0 - Commitment hiding

Code changes:

- add `E * r` terms to commitment equations and witness structures

Acceptance checks:

- happy-path: proof verifies with hiding-enabled commitments
- failure-path: tampering commitment-noise consistency is rejected
- regression: updated equation tests reflect `E * r` terms

### M1 - `y_ring` masking

Code changes:

- implement kernel-mask sampling and challenge-response checks for `j_i`

Acceptance checks:

- happy-path masked flow verifies
- failure-path tampered `j_i` or wrong trace relation fails
- edge-case handling documented/tested when kernel sampling is ill-conditioned

### M2 - ZK sumcheck carry-forward

Code changes:

- add round masks with zero-sum constraint
- carry mask parameters in witness/relation state

Acceptance checks:

- round identities (`g(0)+g(1)=claim`) still hold
- degree bound enforcement remains valid at `2 * b`
- failure-path: one-coefficient mask corruption is detected
- shape invariants (`w` alignment, table sizes) enforced

### M3 - Recursive opening and witness privacy

Code changes:

- remove raw `sumcheck_aux.w` from proof format
- verify recursive opening at final sumcheck point
- bind `w_commitment` to checked witness data

Acceptance checks:

- happy-path recursive opening verifies
- failure-path mismatched witness opening fails
- serialization no longer includes raw witness coefficients

### M4 - Security write-up

Deliverables:

- per-layer simulator obligations
- assumption ledger (MLWE/MSIS/ROM dependencies)
- composition theorem mapping with explicit residual error accounting

Acceptance checks:

- every security claim in this document is tagged as implemented/proven/open

---

## 7. Open questions

1. **MLWE parameters.** Pick concrete dimensions/noise from estimator runs for
   the target security level and layout.

2. **Base-case recursion endpoint.** Decide whether recursion bottoms out into a
   direct lattice opening and, if so, account for its masking costs.

3. **Carry-forward representation.** Decide whether masks stay compatible with
   current compact `i8` path or require a new witness/range-check encoding.

4. **`w_commitment` binding design.** Choose and implement a concrete verifier
   check that binds committed and checked witness data.

5. **Simulator consistency for composition.** Ensure IOR_1 simulation choices
   remain indistinguishable under all downstream checks, not just local ones.

6. **Reference locking.** Pin exact artifact versions for CFW/Greyhound section
   references to avoid citation drift in future revisions.

---

## 8. Local reference map

- `src/protocol/proof.rs`
- `src/protocol/commitment_scheme.rs`
- `src/protocol/ring_switch.rs`
- `src/protocol/sumcheck/hachi_sumcheck.rs`
- `docs/HACHI_FOR_JOLT.md`
- `docs/ZK_AND_MODULUS_EMBEDDING.md`
- `paper/hachi.txt`
