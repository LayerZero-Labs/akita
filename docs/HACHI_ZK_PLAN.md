# Making Hachi Zero-Knowledge

## Status: Design Plan (not yet implemented)

## Scope

This plan targets the current single-modulus Hachi flow:

`R_q relations → ring switch at random α → sumcheck over F`

The approach draws on two sources:

- **Greyhound** (Nguyen–Seiler, CRYPTO 2024): lattice-level hiding and
  `y_ring` masking (§4.5).
- **ZK-IOPP** (Chiesa–Fenzi–Weissenberg 2026): carry-forward mask
  architecture and IOR composition security framework (§4, Theorem 4.5).

Both are adapted to exploit Hachi's ring-switch architecture.

## Notation

- `D`: ring degree (power of two)
- `b = 2^log_basis` (current default `log_basis = 4`, so `b = 16`)
- `δ = ⌈log_b(q)⌉` (decomposition depth)
- `y`: claimed opening value in the field
- `y_ring`: ring-level evaluation sent in the proof
- `v_open`: ring element derived from the inner opening point
- `ct(u)`: constant term of ring element `u`
- `trace(u) = D · ct(u)`

---

## 1. What Hachi leaks today

The current protocol is knowledge-sound but not zero-knowledge. The proof
serializes these prover-to-verifier messages:

| Message | What it reveals |
|---------|-----------------|
| `y_ring` | Ring evaluation of `f`. The verifier checks `ct(y_ring · σ_{-1}(v_open)) = y` (one linear constraint); the remaining D−1 coefficients leak extra linear equations about `f`. |
| `v` (`proof.v`) | `v = D · ŵ` — deterministic linear image of the decomposed witness. (`quadratic_equation.rs`, `compute_v`) |
| Sumcheck round polys | Each round polynomial encodes partial sums over the witness. (`hachi_sumcheck.rs`, `compute_round_univariate`) |
| `sumcheck_aux.w` | **Full raw witness** (`z` and `r` coefficients, concatenated). Temporary; will be replaced by recursive PCS opening. (`proof.rs`, `SumcheckAux`) |

Two messages are already safe under existing assumptions:

| Message | Why safe |
|---------|----------|
| `u` (commitment) | Ajtai commitment `u = B·t̂`. Binding under MSIS. Not currently hiding. |
| `w_commitment` | Ajtai commitment to the ring-switch witness `w`. Same situation. **Note:** `w_commitment` is transcript-bound but never opened against `sumcheck_aux.w` — this binding gap must be closed (see §3.5). |

---

## 2. Key structural advantage: the ring-switch boundary

Hachi differs from Greyhound in a critical way:

- **Greyhound** works entirely over R_q. Every constraint is a ring-level
  equation. ZK requires lattice-specific techniques (LNP22 coefficient
  masking, Gaussian masking) at every step. (JL projections appear only
  in the LaBRADOR subprotocol, not in Greyhound's own protocol.)

- **Hachi** has a ring-switch step (`ring_switch.rs`) that converts
  ring-level constraints into field-level constraints over F_{q^k}, then
  runs a standard sumcheck. After `build_w_coeffs`, no `CyclotomicRing`
  appears — everything is `Vec<F>`.

This means:
- **Lattice ZK** is needed only for pre-ring-switch messages: `u`, `v`,
  `y_ring`, `w_commitment`.
- **Field-level ZK** handles the sumcheck via standard masking.

**Precise boundary statement.** The ring-switch is a
*computational-domain* boundary (ring operations → field operations), not
a semantic boundary where ring-switch constraints disappear. Post-switch
sumcheck still enforces ring-switch-derived consistency and range
constraints, just encoded as field relations. The ZK design must account
for this: upstream simulated outputs must remain consistent with
downstream checks (see §6, M4).

---

## 3. The ZK recipe

### 3.1 Commitment hiding (Module-LWE)

**Problem.** Both `u = B·t̂` and `v = D·ŵ` are pure Ajtai (binding, not
hiding). Same for `w_commitment`.

**Fix.** Add an LWE noise term to each, following Greyhound §4.5:

```
u  = B·t̂  + E_u·r_u        (E_u new public matrix, r_u ← χ^μ)
v  = D·ŵ  + E_v·r_v        (E_v new public matrix, r_v ← χ^μ')
w_commitment similarly
```

Under Module-LWE, each becomes computationally indistinguishable from
random. The randomness vectors (`r_u`, `r_v`, etc.) become part of the
witness for later opening proofs.

**Parameters.** Neither Greyhound nor Hachi specifies concrete μ. Compute
via lattice-estimator for the given n_B, d, q, and noise distribution.
Greyhound uses uniform mod b (the decomposition base) as the noise
distribution.

**Implementation note.** Once added, relation tests that currently assert
plain equalities (`D · ŵ = v`, `B · t̂ = u`) must be updated to include
the `E · r` terms. The `E` matrices are derived from a public seed via
`derive_public_matrix` with new labels (e.g. `b"E_u"`).

### 3.2 Masking `y_ring` (adapted Greyhound technique)

**Problem.** `y_ring ∈ R_q` is sent in the clear. It has D coefficients
but only one is constrained by the trace relation:

```
ct(y_ring · σ_{-1}(v_open)) = y
```

where `v_open` is the publicly-known ring element from the inner opening
point. The other D−1 coefficients are extra linear equations about `f`.

**Difference from Greyhound.** Greyhound uses `ct(ȳ) = y`, so masks
satisfy `ct(l) = 0`. Hachi uses the trace pairing
`ct(y_ring · σ_{-1}(v_open)) = y`, so masks must satisfy
`ct(l · σ_{-1}(v_open)) = 0` (i.e., `l` lies in the kernel of the
linear form `u ↦ ct(u · σ_{-1}(v_open))`).

**Fix.** Follow the Greyhound §4.5 HVZK technique, adapted:

1. Sample L masks `l_1, ..., l_L ∈ R_q` satisfying
   `ct(l_i · σ_{-1}(v_open)) = 0`.
   (Pick D−1 random coefficients, solve for the last via the constraint.
   Requires at least one nonzero coefficient of `v_open` — see Open
   question 4.)
2. Commit to `l̂_i = G^{-1}(l_i)` in the first round
   (absorbed into the `v` commitment, already hiding via `E_v·r_v`).
3. Verifier sends challenges `α_1, ..., α_L ∈ Z_q`.
4. Prover reveals `j_i = l_i + α_i · y_ring`.
5. Verifier checks `ct(j_i · σ_{-1}(v_open)) = α_i · y`.

The `j_i` are one-time-padded on the kernel components — they reveal
nothing about `y_ring` beyond `ct(y_ring · σ_{-1}(v_open)) = y`.

**Soundness.** Extraction error additive `q^{-L}`. With `q ≈ 2^32`, set
`L = 4` for 128-bit security. The ZK property (hiding of kernel
components) is statistical.

### 3.3 Field-level ZK sumcheck

**Problem.** Sumcheck round polynomials are deterministic functions of the
witness.

**Fix.** Per-round masking using the Libra (XZZPS19) technique:

1. Prover samples random univariates `ρ_1, ..., ρ_n` of degree `2b = 32`
   (matching the fused round polynomial degree bound from
   `HachiSumcheckVerifier::degree_bound()`), each satisfying
   `ρ_i(0) + ρ_i(1) = 0`.
2. Send masked round polynomials: `g̃_i(X) = g_i(X) + ρ_i(X)`.
3. The claim telescopes correctly because `ρ_i(0) + ρ_i(1) = 0`.
4. At the end, the reduced claim involves `w(r*) + Σ_i ρ_i(r_i*)`.

**Mask representation: design decision required.** The plan's original
"carry-forward" approach (append mask coefficients to the witness vector
`w`, avoid separate commitment) has a fundamental tension with the fused
norm sumcheck:

- `w` entries are currently range-checked: `range_check_eval(w(x), b)`
  in `HachiSumcheckVerifier::expected_output_claim()` evaluates
  `∏_{j=-b/2}^{b/2-1} (w - j)`, which is zero iff `w ∈ {-b/2, ..., b/2-1}`.
- If mask coefficients are uniform field elements (good for ZK), they
  violate the range check.
- If mask coefficients are small balanced digits (pass range check), the
  per-coefficient masking is ≤ `b/2 = 8`, providing negligible hiding
  for round polynomial coefficients that are full field elements (~2^32).

**Resolution options:**

| Option | Approach | Pro | Con |
|--------|----------|-----|-----|
| A | Scope norm check to real witness only; keep masks in w but in a separate un-range-checked segment | Single commitment; carry-forward still works | Requires splitting `w_table` in `HachiSumcheckProver` into range-checked vs unrestricted |
| B | Separate mask polynomial with its own commitment/opening | Clean separation; masks are full-field | Separate mask commitment (~128B) + opening (~2-5KB per level) |
| C | Commit to a PRG seed; derive masks pseudorandomly | Minimal witness growth (~32 bytes seed) | Needs formal ROM argument; prover computes PRG expansion |

**Recommendation.** Option A preserves the carry-forward architecture with
a localized code change. The fused sumcheck becomes a three-component sum:

```
batching_coeff * eq(tau0, x) * range_check(w_real(x), b)
+ w(x) * alpha_val(y) * m_val(x)
+ mask_eval_term(x)
```

where `range_check` applies only to the real-witness portion of `w`.

**Implementation constraints (any option):**
- `w.len()` must remain divisible by `D` (enforced by `build_w_evals`
  in `ring_switch.rs`) — mask entries must be padded accordingly.
- `num_u = ceil(log2(w.len() / D))` determines `m_evals_x` table size;
  changing `w.len()` cascades to table shapes.
- The compact `Vec<i8>` path in `HachiSumcheckProver::new` assumes all
  entries fit in `i8`; mask entries may require bypassing this path.

**For Hachi's parameters** (~36 rounds, degree 32): each round mask has 32
degrees of freedom (33 coefficients minus 1 for zero-sum). Under
Option A, this adds ~1152 field elements to `w` (before D-padding).
Under Option B, these are committed separately.

### 3.4 Replace `sumcheck_aux.w` with recursive PCS opening

**Problem.** The proof currently includes the full raw witness `w`.

**Fix.** Replace with a recursive PCS opening at the sumcheck evaluation
point `r*`. The verifier receives a masked evaluation (the mask
contributions from §3.3 are already folded in), not the full witness.

This is already the intended design — `proof.rs` marks `SumcheckAux` as
"Temporary verifier auxiliary (will be removed with recursive PCS)."

### 3.5 Bind `w_commitment` to the checked witness

**Problem.** Today, `w_commitment` is absorbed into the Fiat-Shamir
transcript but never opened against `sumcheck_aux.w`. The verifier
derives the ring-switch challenge `α` from `w_commitment`, then
separately receives `w` in the clear. No check ties them together.

This is not currently exploitable (the raw `w` is sent), but becomes a
**soundness hole** if M3 removes `sumcheck_aux.w` before adding a
commitment opening check.

**Fix.** Before or concurrent with M3: verify a commitment opening that
binds `w_commitment` to the witness used in final checks. Alternatively,
if recursive opening replaces direct witness transmission, ensure the
opening proof is relative to `w_commitment`.

### 3.6 Proving well-formedness

After applying §3.1–3.4, the prover must demonstrate that all new witness
components (`r_u`, `r_v`, `l̂_i`, mask coefficients) are consistent with
the commitments and constraints. These are all linear relations over R_q
and can be folded into the existing stacked relation `M·z = y + (X^D+1)·r`.

Ring-level pieces (randomness `r_u`, `r_v`, mask decompositions `l̂_i`)
belong pre-ring-switch. Field-level mask coefficients from §3.3 belong
post-ring-switch, in the fused `HachiSumcheckProver`. Placing each piece
in the correct layer avoids cross-domain leakage.

---

## 4. What applies from prior work

### From Greyhound (Nguyen–Seiler)

| Technique | Applies? | Notes |
|-----------|----------|-------|
| E·r commitment hiding | **Yes** | Same Ajtai structure |
| Non-constant-coeff masking for ȳ | **Yes, adapted** | Mask kernel condition uses trace pairing `ct(l · σ_{-1}(v_open)) = 0`, not `ct(l) = 0` |
| LaBRADOR HVZK for constraints | **No** | Hachi uses field-level sumcheck after ring switch |
| Gaussian masking / rejection sampling | **Only at base case** | If recursion bottoms out into a direct lattice opening |
| JL projection masking | **No** | JL is used in LaBRADOR, not in Greyhound's or Hachi's own protocol |
| ABDLOP commitment redesign | **Not needed** | Incremental approach (E·r + masking) suffices |

### From ZK-IOPP (Chiesa–Fenzi–Weissenberg 2026)

| Technique | Applies? | Notes |
|-----------|----------|-------|
| Carry-forward mask accumulation | **Partially** | Concept applies; needs adaptation for range-checked witness (see §3.3) |
| IOR composition framework for ZK | **Yes** | Used for M4 security proof (Theorem 4.5) |
| ε-separation in sumcheck | **No** | Not needed: masks and witness in same field F_{q^k} |
| Low-degree masks (ℓ_zk = 2) | **No** | Hachi norm sumcheck is degree 32, not multilinear |
| ZK codes / ZK encodings | **No** | IOP-specific; replaced by Ajtai commitments |
| Interleaved folding / code switching | **No** | Hachi uses ring-switch, not code interleaving |
| Distance amplification (dispersers) | **No** | Hash-based optimization |
| Private zero-evaders / OOD sampling | **No** | Hachi uses ring-switch instead |

---

## 5. Overhead estimates

These are scenario-level estimates. Exact numbers depend on the mask
representation choice (§3.3) and recursive opening format (§3.4).

| Component | Current | With ZK | Notes |
|-----------|---------|---------|-------|
| Commitment computation | `A·s`, `B·t̂` | + `E·r` terms | ~1.1× prover time |
| `y_ring` handling | Sent in clear | L=4 masks committed + revealed | +4 ring elements (~0.5KB) |
| Sumcheck round polys | `g_i` | `g̃_i = g_i + ρ_i` (same degree, same size) | ~0 proof size increase |
| Witness vector `w` | baseline | +~1152 mask entries (Option A/C) | D-padding may increase this |
| Base-case opening | TBD | Gaussian masking if direct lattice opening | ~3× at base only |

**Proof-size and prover-time impact** depend on mask encoding path,
recursive opening format, and whether the compact `i8` representation
is retained. Avoid treating rough multipliers as final until M2/M3 are
benchmarked.

---

## 6. Security target and proof obligations

### 6.1 Target notion

- **Goal:** Zero-knowledge PCS — the opening proof reveals nothing about
  `f` beyond `f(x) = y`.
- **Achieved notion:** Computational HVZK in the ROM (Fiat-Shamir).
- **Not achieved:** Statistical ZK (commitments are only computationally
  hiding).

### 6.2 Proof strategy: IOR composition

Structure the ZK proof as modular IOR-HVZK lemmas composed via
[CFW26, Theorem 4.5].

**Step 1: Define each Hachi step as an IOR.**

| IOR | From | To |
|-----|------|----|
| IOR_1 (QuadraticEquation + y_ring masking) | PCS opening claim | Quadratic relation M·z = y+(X^D+1)·r |
| IOR_2 (RingSwitch + commitment hiding) | Ring-level quad eq | Field-level sumcheck instance |
| IOR_3 (ZK Sumcheck) | Field sumcheck instance | Reduced claim at r* |
| IOR_4 (Base case) | Reduced claim | Accept/reject |

**Step 2: State HVZK with distinguisher for each IOR.**

Following [CFW26, Definition 4.2], each IOR's simulator must produce
output indistinguishable to a "distinguisher" D modeling downstream
queries. Crucially, IOR_i must be HVZK not for arbitrary distinguishers
but for the **S_{i+1}-induced class D_{S_{i+1}}** — the class induced
by the next layer's simulator.

- **IOR_1 HVZK:** Simulator produces `(v, j_i)` indistinguishable to D
  that queries `w` through ring-switch and sumcheck.
  Security: MLWE hiding of `v` (§3.1) + trace-pairing masking (§3.2).

- **IOR_2 HVZK:** Simulator produces `w_commitment` indistinguishable
  to D that evaluates `w` through sumcheck.
  Security: MLWE hiding of `w_commitment` (§3.1).

- **IOR_3 HVZK:** Simulator samples masked round polynomials uniformly
  from the verification subspace T (transcripts satisfying sumcheck
  consistency). ZK: the affine map from masks to transcripts is
  surjective onto T (dimension count: degree-d mask with zero-sum
  constraint has d free coefficients, matching d non-constrained
  dimensions of the round polynomial).
  Security: statistical.

- **IOR_4 HVZK:** Base-case ZK (Gaussian masking if direct lattice
  opening, or recursive Hachi if iterated).

**Step 3: Compose via [CFW26, Theorem 4.5].**

End-to-end ZK error ≤ Σ_i ε_i (additive composition). Each ε_i is
IOR_i's simulation error. No "interaction verification" needed — the
framework handles it.

### 6.3 Assumptions per IOR

| IOR | Assumption |
|-----|------------|
| IOR_1 | Module-LWE (hiding) + trace-pairing masking (statistical on kernel) |
| IOR_2 | Module-LWE (w_commitment hiding) |
| IOR_3 | Statistical (uniform masks → uniform transcript) |
| IOR_4 | Depends on base-case design |

**Binding:** Module-SIS throughout.

### 6.4 Obligation: simulator consistency across layers

The IOR composition requires that each simulator's output is consistent
with downstream checks, not just locally indistinguishable. The key
instance: IOR_1's simulator (for `v`) must produce output that remains
indistinguishable under IOR_3's norm sumcheck. The correct argument uses
the compositional framework (D_{S_2} is efficient, so MLWE suffices) —
not a heuristic about random linear combinations.

---

## 7. Implementation milestones

### M0: Commitment hiding

**Code changes:**
- Add `E` matrices and randomness `r` to commitments (`u`, `v`,
  `w_commitment`).
- Derive `E` from public seed with new labels in commitment setup.
- Store `r` in `CommitWitness` / `HachiCommitmentHint`.
- Update relation tests to include `E · r` terms.

**Acceptance criteria:**
- Happy-path: proof verifies with hiding-enabled commitments.
- Failure-path: tampering commitment-noise consistency is rejected.
- Regression: existing equation tests updated for `E · r`.

### M1: y_ring masking

**Code changes:**
- Implement kernel-mask sampling
  (`ct(l · σ_{-1}(v_open)) = 0`) for masks.
- Add masking commitment to the first round.
- Add challenge-response protocol for `j_i = l_i + α_i · y_ring`.
- Add trace-pairing verification.
- Define new Fiat-Shamir transcript labels for mask commitment, mask
  challenges, and `j_i` reveals.

**Acceptance criteria:**
- Happy-path masked flow verifies.
- Failure-path: tampered `j_i` or wrong trace relation fails.
- Edge-case: `v_open = 0` handled (error or documented precondition).

### M2: ZK sumcheck

**Code changes (depend on §3.3 resolution):**
- Implement sum-of-univariates masking in `hachi_sumcheck.rs`:
  sample `ρ_i` with `ρ_i(0)+ρ_i(1)=0`, degree ≤ 2b = 32,
  add to round polynomials.
- If Option A: split `w_table` into range-checked and unrestricted
  segments; scope norm check to real witness only.
- If Option B: add separate mask commitment and opening.
- Update `build_w_coeffs` / `build_w_evals` for new witness shape;
  maintain D-divisibility invariant.

**Acceptance criteria:**
- Round identities (`g(0)+g(1)=claim`) still hold.
- Degree bound enforcement remains valid at `2b = 32`.
- Failure-path: single-coefficient mask corruption detected.
- Shape invariants (`w` alignment, table sizes) enforced.

### M3: Recursive opening and witness privacy

**Code changes:**
- Remove raw `sumcheck_aux.w` from proof format.
- Verify recursive opening at final sumcheck point.
- Bind `w_commitment` to checked witness data (§3.5 prerequisite).

**Acceptance criteria:**
- Happy-path recursive opening verifies.
- Failure-path: mismatched witness opening fails.
- Serialization no longer includes raw witness coefficients.

### M4: Security write-up

**Deliverables:**
- Per-IOR simulator obligations with explicit D_{S_{i+1}} verification.
- Assumption ledger (MLWE/MSIS/ROM dependencies).
- Composition theorem mapping with explicit residual error accounting.
- Concrete parameter sheet (μ, L, mask counts, SIS norm budget).

**Acceptance criteria:**
- Every security claim in this document tagged as
  implemented/proven/open.

---

## 8. Open questions

1. **MLWE dimension μ.** Neither Greyhound nor Hachi specifies concrete
   values. Compute via lattice-estimator.

2. **Base-case ZK.** If recursion bottoms out into a direct lattice
   opening (Greyhound/LaBRADOR style), Gaussian masking is needed there.
   Norm blowup (~13× with standard rejection, ~4-5× with iterative
   rejection sampling) feeds back into SIS parameters. Additionally, the
   base-case witness includes all accumulated mask entries; the Gaussian
   parameter σ must mask this enlarged witness, and the resulting SIS
   norm bound must remain secure.

3. **Mask representation choice.** §3.3 identifies a fundamental tension
   between carry-forward masking and the norm/range check. The
   resolution (Option A/B/C) determines the implementation shape of M2
   and affects overhead estimates. Decision needed before M2 starts.

4. **`y_ring` masking when `v_open` has a zero coordinate.** Sampling
   `l` in the kernel of `u ↦ ct(u · σ_{-1}(v_open))` requires solving
   a linear constraint, which needs at least one nonzero coordinate of
   `v_open`. For Lagrange basis weights,
   `v_open = CyclotomicRing::from_slice(&basis_weights(inner_point))`,
   which generically has all nonzero coordinates. But edge cases
   (opening at 0) need care.

5. **Mask degree vs fused sumcheck degree.** The masks `ρ_i` must have
   degree `2b = 32` (matching `degree_bound()` in
   `HachiSumcheckVerifier`). Under Option A (masks in `w` but
   un-range-checked), mask entries are free field elements with no norm
   constraint. Under Option B, this is a non-issue. Verify the degree
   count: `degree_bound() = 2 * b` (not `2b - 1`).

6. **Simulator consistency for IOR composition.** The IOR composition
   framework requires that IOR_1's simulator output (for `v`, `j_i`) is
   indistinguishable under IOR_3's norm sumcheck. The correct argument
   is compositional (MLWE hiding of `v` suffices because D_{S_2} is
   efficient), not heuristic. This needs to be written out as a formal
   proof sketch in the M4 security write-up.

7. **Transcript label allocation.** The current label set
   (`transcript/labels.rs`) has no ZK-related labels. M1 and M2 each
   introduce new Fiat-Shamir absorptions and challenges. Incorrect
   interleaving could break the simulation argument. Define the full
   ordering before writing code.

---

## 9. Local reference map

- `src/protocol/proof.rs` — `HachiProof`, `SumcheckAux`
- `src/protocol/quadratic_equation.rs` — `QuadraticEquation`, `compute_v`
- `src/protocol/commitment_scheme.rs` — `prove`, `verify`
- `src/protocol/ring_switch.rs` — `ring_switch_prover`, `build_w_coeffs`, `build_w_evals`
- `src/protocol/sumcheck/hachi_sumcheck.rs` — `HachiSumcheckProver`, `HachiSumcheckVerifier`
- `src/protocol/transcript/labels.rs` — Fiat-Shamir labels
- `paper/greyhound.pdf` — §4.5 (hiding + HVZK)
- `paper/zk-iopp.pdf` — §4 (IOR composition), Theorem 4.5
