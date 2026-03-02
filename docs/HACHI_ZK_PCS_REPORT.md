# Hachi as a Zero-Knowledge PCS: Design Report

## Executive Summary

Yes, there is a credible path to make Hachi a zero-knowledge PCS, and there are at least two viable routes:

1. **Incremental route (recommended first):** keep Hachi's current split/fold + ring-switch + sumcheck skeleton, and add a LNP22-style ZK layer (masking commitments, rejection-sampled responses, commit-and-prove simulation argument).
2. **Deeper redesign route:** replace Hachi's current commitment/opening core with an ABDLOP-like commitment layer and make ZK native across all recursive rounds.

In both cases, the Falcon+LaBRADOR "small modulus relation embedded into a larger proof modulus" technique is directly applicable to Hachi's ring-switching equations and is especially useful for cross-prime settings (small `q` commitment arithmetic, large `q'` sumcheck/verification arithmetic).

Expected cost to first working ZK prototype is significant but manageable:

- **Proof size:** likely from `55.1KB` baseline to roughly `70KB-105KB` for a practical first ZK version.
- **Prover time:** likely `2x-5x` baseline in early versions (with retries and added constraints).
- **Verifier time:** likely `1.1x-1.8x` baseline, depending on how much extra constraint/batching logic is moved to verifier.

This is not "drop-in." The key challenge is to add simulation privacy without destroying Hachi's verifier advantage.

---

## 1) What Hachi Has Today (and What Is Missing for ZK)

Hachi currently gives a knowledge-sound recursive opening protocol, but not a ZK claim.

- Commitment core in `paper/hachi.pdf` Section 4.1 is an inner/outer Ajtai-style commitment flow (`s_i`, `t_i`, `u`) and then weak-opening extraction language (`Lemma 7`).
- Security stack in Section 4.2/4.3 is CWSS/special-soundness extraction (`Lemma 8`, `Lemma 9`, `Lemma 10`, `Lemma 11`), then Fiat-Shamir for NIZK-style non-interactive proofs.
- There is **no explicit simulator-based ZK theorem** in the Hachi paper sections inspected.

In the repo, protocol-level opening prove/verify is still stubbed, so there is no complete end-to-end ZK-capable implementation yet.

### Concrete evidence

```1167:1262:paper/hachi.pdf
4.1 Inner and Outer Commitment
...
Lemma 7 (Weak Binding) ...
```

```1585:1600:paper/hachi.pdf
Fig. 3 ... prover does not send final message in the clear, but proves knowledge ...
Lemma 8 (CWSS of Figure 3) ...
```

```1750:1756:paper/hachi.pdf
... polynomial degree at most 2d-1 ...
Lemma 9 (Special soundness of Figure 4) ...
```

```1841:1845:paper/hachi.pdf
Lemma 10 (CWSS of Figure 5) ...
```

```1889:1920:paper/hachi.pdf
Lemma 11 ... special soundness ...
... final protocol satisfies coordinate-wise special soundness.
```

```13:24:src/protocol/prover/stub.rs
pub fn prove_opening_stub ... Err(... "not implemented yet")
```

```13:24:src/protocol/verifier/stub.rs
pub fn verify_opening_stub ... Err(... "not implemented yet")
```

```63:94:src/protocol/commitment/commit.rs
fn commit_ring_blocks ... deterministic decomposition + matrix multiplies
```

---

## 2) Why LNP22 and Falcon+LaBRADOR Are Exactly the Right Inputs

## LNP22 contributes the ZK machinery

LNP22 contributes the pieces Hachi currently lacks:

- **Mask-all-but-constant-coefficient** proof pattern (`g/h` masking) for ring-polynomial constraints.
- **Rejection sampling** to decouple revealed response distributions from secret witness.
- **Commit-and-prove simulatability** argument when commitments are one-shot and appended during proof.
- **Approximate range/no-wrap** proof pattern to lift modular equalities to integer statements.

### Concrete evidence

```1347:1401:paper/LNP22.pdf
... need to mask all but the constant coefficient ...
... instead of proving zero-knowledge, we show commit-and-prove ...
... simulator picks tg uniformly and h_j random with zero constant coefficient ...
```

```385:395:paper/LNP22.pdf
... proofs are modulo q ...
... approximate range proof to show no wraparound ...
... output z = R s + y with rejection sampling to hide s ...
```

## Falcon+LaBRADOR contributes the modulus-lift recipe

Falcon+LaBRADOR contributes the practical cross-modulus trick:

1. Lift mod-`q` relation to integer equation with explicit slack `q * v`.
2. Prove in larger modulus/ring.
3. Add no-wrap bounds so mod-`q'` validity implies integer validity.

### Concrete evidence

```1331:1345:paper/aggregate-falcon-labrador.pdf
6.1 Changing the Modulus ...
s_{i,1} + h_i s_{i,2} + q v_i - t_i = 0 in Z ...
... prove over larger q' + infinity-norm no-wrap checks ...
```

```1408:1497:paper/aggregate-falcon-labrador.pdf
6.2 ... approximate l_infinity smallness ...
... derive q' > (1024/15)(d+2)beta^2 N ...
```

---

## 3) Path A (Recommended): Incremental ZK Overlay on Current Hachi

This route minimizes risk and preserves Hachi's main engineering structure.

## A.1 Protocol idea

Keep:

- Section 4.1 commitment layout.
- Section 4.2 reduction to Eq. (20)-style stacked relation.
- Section 4.3 ring-switch + batched constraints + sumcheck recursion.

Add:

1. **Auxiliary masking commitments** for helper polynomials/vectors (`g`-style masks, optional projection masks).
2. **Masked constant-coefficient checks** for selected sensitive constraints.
3. **Rejection-sampled response objects** where witness-dependent values are revealed.
4. **Commit-and-prove simulator argument** for one-shot commitments per recursion layer.

## A.2 Where exactly to insert

- Insert masking in the `H_alpha/H_0` batching stage (digest Step C, Eq. (22)/(23)-style region).
- Add optional projection/no-wrap constraints as separate batched checks that reduce to same sumcheck pipe.
- Keep recursion shape identical: still reduce to opening of next witness table.

The digest already outlines where witness-table and quotient slack live:

```495:530:docs/HACHI_DIGEST.md
Step C ... ring switching ...
Mz = y + (X^d + 1) r ...
```

## A.3 Why this is the best first move

- Lowest disruption to existing architecture and parameter intuition.
- Fastest path to "first ZK version that works."
- Lets you isolate overhead per added ZK component and prune aggressively.

---

## 4) Path B: ABDLOP-First Redesign (Cleaner Long-Term)

Replace Hachi's commitment layer with an ABDLOP-style unified commitment that supports:

- Large witness lane (Ajtai-like "free dimension"),
- Small auxiliary lane (BDLOP append-friendly),
- Native masking commitments for all proof gadgets.

Then re-express Hachi Section 4 constraints as a set of linear/quadratic constant-coefficient relations in the LNP22 style.

## B.1 Upside

- Cleaner theorem story for ZK.
- Better composability for appended commitments and simulation.

## B.2 Downside

- Heavier rewrite of both protocol and code.
- Higher theorem burden before usable prototype.

Recommendation: do **Path A first**, reserve Path B as medium-term refactor once measurements identify dominant leakage/overhead bottlenecks.

---

## 5) Modulus Embedding: How Falcon+LaBRADOR Maps to Hachi

For same-prime Hachi (`q` with extension `F_{q^k}`), you often do not need extra modulus slack.

For cross-prime settings (`q -> q'`), use:

1. Cyclotomic quotient (already in Hachi): `(X^d+1) * r`.
2. Additional modulus quotient: `q * s`.

So the lifted polynomial identity becomes:

`lift_q(M) * lift_q(z) - lift_q(y) = (X^d + 1) * r + q * s` over integers.

This matches the digest's cross-prime section:

```891:899:docs/HACHI_DIGEST.md
Modulus switching / cross-prime sumcheck ...
no embedding F_q -> F_q' preserving field arithmetic, need integer lift.
```

```1028:1042:docs/HACHI_DIGEST.md
Step 2 generalized ...
add modulus quotient witness s ...
```

This is precisely Falcon's pattern in Hachi notation.

---

## 6) Overhead Model (Practical, Not Asymptotic Only)

Baseline from Hachi:

- Proof size ~= `55.1KB = 7.3 + 4.8 + 43`.
- Verify ~= `~185-227ms` first round.
- Prove currently dominated by first round and very high in prototype.

Useful back-of-envelope:

- Each extra ring-element commitment at `d=1024`, `q~2^32` is about `4KB`.
- Extra sumcheck rounds are cheap compared to full extra commitment/proof passes.
- Rejection sampling multiplies prover work by about `1 / p_accept`.

## Scenario estimates

### Optimistic (tight engineering)
- Proof size: `1.2x-1.35x` (`66KB-74KB`)
- Prover time: `1.6x-2.3x`
- Verifier time: `1.1x-1.3x`

### Expected (realistic first ZK release)
- Proof size: `1.45x-1.85x` (`80KB-102KB`)
- Prover time: `2.8x-5.2x`
- Verifier time: `1.2x-1.6x`

### Conservative (heavy no-wrap + retries + wide field penalty)
- Proof size: `1.9x-2.8x` (`105KB-154KB`)
- Prover time: `5.5x-12x`
- Verifier time: `1.4x-2.0x`

Interpretation: proof size is likely still usable for SNARK-oriented settings, but prover-time optimization becomes the dominant engineering challenge.

---

## 7) The "Spark" Ideas Worth Testing Early

1. **Hybrid privacy mode:** ZK only on the first 1-2 expensive rounds (where witness is largest and leakage risk highest), then cheaper non-ZK recursion on tiny reduced witnesses.
2. **Selective masking:** mask only constraints that expose high-information linear forms; leave low-risk constraints unchanged.
3. **Iterative rejection sampling for transcript responses:** adapt the iterative rejection idea (Gartner CRYPTO 2025) to reduce abort penalty in repeated response generation.
4. **Cross-prime ZK with canonical-lift discipline:** pick one lift convention globally and hard-code it into transcript domain separators to avoid ambiguity channels.
5. **"Verifier-preserving" design rule:** any ZK patch that touches verifier asymptotics gets rejected unless it preserves Hachi's square-root verifier profile.

---

## 8) Concrete Milestone Plan

## M0 - Baseline instrumentation
- Add phase-by-phase metrics: commitment bytes, sumcheck bytes, recursion handoff bytes, retry counts.
- Acceptance: reproduce paper-ish `55KB` breakdown on your benchmark config.

## M1 - Witness-table ZK plumbing
- Extend witness table to include auxiliary mask rows (and optional modulus slack rows).
- Keep verifier logic unchanged.

## M2 - Constraint integration
- Add masked constant-coefficient checks + optional no-wrap constraints into batched polynomial system.
- Acceptance: correctness tests pass, proof still recursively reducible.

## M3 - Rejection-sampling integration
- Add response sampling layer and measure accept rate.
- Acceptance: stable `p_accept` above chosen threshold and predictable prover multiplier.

## M4 - Security package
- Write composed security argument: extraction + simulation + Fiat-Shamir assumptions.
- Acceptance: internal proof note + parameter sheet + benchmark report.

---

## 9) Final Recommendation

Start with **Path A** and explicitly target:

- first ZK proof <= `95KB`,
- verifier <= `350ms`,
- prover <= `5x` current baseline.

If you hit those numbers, Hachi ZK is immediately interesting as a practical lattice PCS. If prover inflation is worse, pivot to selective/hybrid masking and iterative-rejection variants before attempting full ABDLOP redesign.

