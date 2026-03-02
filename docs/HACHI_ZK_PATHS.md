# Making Hachi Zero-Knowledge: Research Paths

## 1. What Hachi currently reveals

The protocol has several message flows (Figures 3–7 of the paper). Not all of them are safe for ZK.

### Messages that are already hiding

| Message | Protocol step | Why it's safe |
|---------|---------------|---------------|
| `u ∈ R_q^{n_B}` | Main commitment (Eq. 14) | Ajtai commitment, hiding under MLWE |
| `v ∈ R_q^{n_D}` | Aux commitment (Eq. 16) | Ajtai commitment, hiding under MLWE |
| `t` | Commitment to (z, r) (§4.3) | Ajtai commitment, hiding under MLWE |

### Messages that leak information about the witness

| Message | Protocol step | What it reveals |
|---------|---------------|-----------------|
| **Y ∈ R_q** | Extension-field embedding (§3.1) | A weighted sum of the packed coefficient blocks F_i at the evaluation point. Y is not determined by the output y alone — the trace Tr_H(Y·σ_{-1}(v)) = (d/k)·y "projects" Y, so different Y values are consistent with the same y. Sending Y reveals extra info about f. |
| **y_i ∈ F_{q^k}** (k-1 values) | Extension-field embedding (§3.2, Eq. 5) | Partial evaluations of slices of f. These are evaluations of f restricted to different settings of the first κ variables — they reveal f's structure beyond the single claimed evaluation f(x) = y. |
| **Sumcheck round polys g_j** | Sumcheck (§4.3, Figure 6) | Each g_j(X) is a low-degree univariate that encodes a partial sum over the witness table. The sequence of g_j reveals O(n·d) field elements' worth of information about the witness, where n = number of rounds and d = degree of g_j. |
| **ẽ_w(r\*)** | Final sumcheck evaluation | One evaluation of the committed witness MLE at the random point r\*. This is a random linear combination of witness coefficients. |

## 2. The key structural observation

**Hachi's ring switching transforms the problem into a sumcheck over F_{q^k}.** This is the crucial enabler for ZK.

After ring switching (§4.3), the verifier's checks become field equations at a random α ∈ F_{q^k}. The sumcheck runs entirely over F_{q^k}. This means:

- **Sumcheck ZK** is a field-level problem with well-understood solutions (Spartan, Libra, etc.)
- **Commitment ZK** is a lattice-level problem addressable by LNP22 techniques
- The two concerns are **decoupled** and can be handled independently

```
┌─────────────────────────────────────────────────────────────────┐
│                        HACHI PROTOCOL                           │
│                                                                 │
│  ┌─────────────────┐     ┌──────────────────┐                   │
│  │ Lattice layer    │     │ Field layer       │                  │
│  │ (R_q)            │     │ (F_{q^k})         │                  │
│  │                  │     │                   │                  │
│  │ • Ajtai commit   │────►│ • Ring switching   │                 │
│  │ • Short vectors  │     │ • Sumcheck         │                 │
│  │ • MSIS binding   │     │ • Range checks     │                 │
│  │                  │     │ • Final eval claim  │                │
│  └─────────────────┘     └──────────────────┘                   │
│         │                          │                            │
│         ▼                          ▼                            │
│  ZK approach:              ZK approach:                         │
│  LNP22 Gaussian            Standard field-level                 │
│  masking + rejection        ZK sumcheck                         │
│  sampling (at base case)    (masking polynomial)                │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Three paths to ZK

### Path A: ZK sumcheck with separate masking commitment

**Idea.** Use the standard ZK sumcheck construction from Spartan/Libra: add a random masking polynomial ρ to the sumcheck claim, so round polynomials are perfectly masked.

**Protocol changes.**

At each recursion level:

1. Prover samples a random multilinear ρ with the same variable count as the witness table ẽ_w
2. Prover commits to ρ using the same Ajtai-style commitment: `t_ρ = A · G⁻¹(ρ)`
3. Prover sends σ = Σ_{b ∈ {0,1}^n} ρ(b) to the verifier (one field element)
4. Run sumcheck on H + ρ instead of H. Each round polynomial g_j + ρ_j is statistically independent of the witness.
5. At the final point r\*, the verifier receives (ẽ_w + ρ)(r\*) — safe because ρ(r\*) is random
6. Prover opens ρ at r\* (recursive PCS opening), verifier subtracts to recover ẽ_w(r\*)

For Y and y_i: commit to these instead of sending in the clear, and include the trace relation and Eq. (5) as additional constraints in the sumcheck.

**Cost overhead.**

| Component | Current | With ZK | Overhead |
|-----------|---------|---------|----------|
| Commitments per level | 1 | 2 (witness + mask) | 2× |
| Sumcheck work per level | 1× | ~1.2× (evaluate mask alongside witness) | ~1.2× |
| Openings per level | 1 | 2 (witness + mask opening) | 2× |
| Proof size per level | baseline | +1 commitment + 1 opening + 1 field element | ~2× |
| Total prover time | baseline | roughly 2× | **~2×** |
| Total proof size | ~55KB | ~100–110KB | **~2×** |

**Pros:** Conceptually clean; uses well-understood field-level ZK sumcheck; no changes to the commitment scheme.

**Cons:** The masking polynomial is as large as the witness, so committing to it roughly doubles the work. The recursive opening of ρ at r\* doubles the recursion.

### Path B: ABDLOP-integrated masking (LNP22 style)

**Idea.** Replace Hachi's plain Ajtai commitment with an ABDLOP commitment (combined Ajtai + BDLOP). The BDLOP part holds masking values; ZK comes from the MLWE-based hiding of the commitment randomness, plus Gaussian masking for opening proofs.

**Modified commitment structure.**

Current:
```
u = A · s ∈ R_q^{n_A}     where s is short (gadget-decomposed witness)
```

Modified:
```
[t_A]   [A₁  A₂] [s₁]   [ 0 ]
[   ] = [      ] [  ] + [   ]
[t_B]   [0   B ] [s₂]   [ m ]
```

- s₁: main witness (short, in the "Ajtai" part) — same role as current s
- s₂: fresh commitment randomness (short) — new, provides hiding
- m: auxiliary values (arbitrary coefficients) — holds masking polynomials g for coefficient masking (LNP22 §1.3) and the "garbage terms" from quadratic proofs

**Where the ZK comes from.**

1. **Commitment hiding**: (A₂, A₂·s₂) is pseudorandom under MLWE, so (t_A, t_B) hides both s₁ and m.

2. **Sumcheck masking**: The coefficient-masking technique from LNP22 (Section 1.3, Eq. 7–8) applies directly. To prove `ct(f(s₁, m)) = 0`, the prover commits to a masking polynomial g (with ct(g) = 0 and random other coefficients) in the BDLOP part, then sends h = γ·f(s₁, m) + g. The non-constant coefficients are perfectly masked.

3. **Opening ZK**: When eventually opening the commitment, use Gaussian masking + rejection sampling. The prover computes z = c·s₂ + y where y ← D_σ, and rejection-samples so z ~ D_σ independent of s₂. The repetition rate is M ≈ 3 (using Rej1 from LNP22).

**The critical issue: integrating with sumcheck.**

Hachi doesn't open the commitment by directly revealing s₁. Instead, the sumcheck reduces everything to evaluation claims. So the LNP22 "opening proof" structure needs to be adapted to a sumcheck-based setting.

The natural integration: use the BDLOP part for the sumcheck masking polynomials. Instead of a separate commitment to ρ (as in Path A), the masking values live in the same ABDLOP commitment.

For a sumcheck with n rounds and max degree d:
- The masking needs n·(d+1) field elements (one degree-d univariate per round)
- These fit in the BDLOP part (m has ℓ ring elements, each with d coefficients)
- For Hachi's parameters: n ≈ 36 rounds, d ≤ 31 (from range check P_b with b=16), so ~1100 field elements — fits easily in the BDLOP part

**Cost overhead.**

| Component | Current | With ZK | Overhead |
|-----------|---------|---------|----------|
| Commitment size | n_A ring elements | n_A + n_B ring elements (BDLOP overhead) | +n_B elements |
| Commitment computation | A·s | A₁·s₁ + A₂·s₂ + BDLOP | ~1.5× |
| Sumcheck | baseline | Same number of rounds; masking is computed alongside | ~1.1× |
| Opening proof | current | Gaussian masking (σ ≈ 13·‖s₂‖, M ≈ 3) | ~3× at base case |
| Total prover time | baseline | ~1.3–1.5× (ABDLOP overhead distributed across levels) | **~1.5×** |
| Total proof size | ~55KB | +BDLOP elements + Gaussian-widened opening | **~1.4× (~75KB)** |

**Pros:** Lower overhead than Path A because the masking is integrated into the commitment (no separate masking commitment). Proof size increase is moderate.

**Cons:** Requires redesigning the commitment scheme. The Gaussian masking introduces a norm blowup (~13×) that feeds into MSIS parameter selection — may require slightly larger q or d. Also, "commit-and-prove simulatability" (not standard HVZK) — same security notion as LNP22, which is sufficient for applications but worth noting.

### Path C: Hybrid field-level sumcheck ZK + lattice-level base-case ZK

**Idea.** Observe that the ZK concern splits cleanly across Hachi's recursion:

- **During recursion** (levels 1 through L-1): the sumcheck runs over F_{q^k}. Use a lightweight field-level ZK technique — specifically, **sum-of-univariates masking** that requires NO extra commitments.
- **At the base case** (level L): the recursion bottoms out and Hachi hands off to Greyhound/LaBRADOR for the final opening. Apply LNP22-style Gaussian masking here.

This is the approach I think is most promising for Hachi specifically.

**The sum-of-univariates ZK sumcheck (from Libra).**

For a sumcheck over n variables with degree d in each variable:

1. Prover picks random univariates ρ₁,...,ρₙ of degree d, each satisfying ρᵢ(0) + ρᵢ(1) = 0
2. In round i, prover sends g̃ᵢ(X) = gᵢ(X) + ρᵢ(X) instead of gᵢ(X)
3. The ρᵢ term perfectly masks the original round polynomial
4. The verifier checks g̃ᵢ(0) + g̃ᵢ(1) = previous claim (unchanged, because ρᵢ(0)+ρᵢ(1) = 0)
5. At the end, the reduced claim involves ẽ_w(r\*) + Σᵢ ρᵢ(rᵢ\*). The prover reveals this sum.

The key point: the prover can reveal ẽ_w(r\*) + Σᵢ ρᵢ(rᵢ\*) without leaking ẽ_w(r\*), because the ρᵢ values are random. But the verifier needs to separate them to verify the PCS opening...

**Resolution: commit to the masking randomness, not the whole polynomial.** The prover commits to the n·(d+1) random coefficients of {ρᵢ} at the start (this is O(n·d) field elements, much smaller than the full witness). At the end, the prover opens this small commitment at the point (r₁\*,...,rₙ\*) to reveal Σᵢ ρᵢ(rᵢ\*).

For Hachi's parameters: n ≈ 36 rounds, d ≈ 31, so ~1100 field elements to commit. Each field element is in F_{q^k} with k=4 and q ≈ 2^32, so ~1100 × 16 bytes ≈ 17KB of masking randomness. This can be committed using a single small Ajtai commitment.

**Handling Y and y_i.**

- **Y**: Commit to Y instead of sending in the clear. Add the trace relation as a constraint in the sumcheck. Cost: one extra ring element in the commitment (+1 × 4KB).
- **y_i**: Commit to the k-1 partial evaluations. Add Eq. (5) as a constraint. Cost: (k-1) extension field elements (~48 bytes for k=4).

**The base-case opening.**

When Hachi's recursion bottoms out (witness small enough), it hands off to Greyhound/LaBRADOR. For ZK at this level, apply LNP22 techniques:

1. Use an ABDLOP commitment for the base-case witness
2. Gaussian masking + rejection sampling for the opening proof
3. Coefficient masking for the constant-term extraction (if using JL projection)

The base-case witness is small (paper §5.2: "next_round_witness size: 226"), so the Gaussian overhead is modest.

**Cost overhead.**

| Component | Current | With ZK | Overhead |
|-----------|---------|---------|----------|
| Per-level sumcheck | baseline | +n·d masking coefficients (~17KB committed) | tiny |
| Per-level proof elements | round polys g_i | masked round polys g̃_i (same size) | 0 |
| Extra commitments per level | 0 | 1 small (masking randomness) | +~1KB |
| Extra openings per level | 0 | 1 small (masking randomness) | +~1KB |
| Y and y_i | sent in clear (~5KB) | committed + proved in sumcheck | +~5KB commitment |
| Base-case opening | Greyhound/LaBRADOR | + Gaussian masking (~3× repetition) | ~3× at base only |
| **Total prover time** | baseline | **~1.2×** (masking generation + base-case overhead) | |
| **Total proof size** | ~55KB | **~65–70KB** (~20–30% increase) | |

**Pros:** Lowest overhead of the three paths. The sumcheck masking is nearly free (sum-of-univariates requires no large extra commitment). The lattice-specific ZK machinery (Gaussian masking) only kicks in at the base case, where the witness is small. The two concerns are cleanly separated.

**Cons:** Requires careful analysis of the masking randomness commitment. The base-case ZK (LNP22 style) still introduces a norm blowup, but only for the small base-case witness, so the impact on global parameters is limited.

## 4. The deeper research questions

### 4.1. Norm blowup from Gaussian masking and its feedback into parameters

In all paths, the Gaussian masking for the commitment opening introduces a norm blowup of ~13× (using Rej1 from LNP22, with σ = 13·‖v‖). This affects MSIS parameter selection: the extracted witness from the ZK protocol has norm ~13× larger than in the non-ZK case, so the MSIS instance must be harder (larger q or larger d).

**Can Gärtner's iterative rejection sampling help here?** If the challenge c in the base-case opening is sparse (which it is in Hachi — the challenges are drawn from a set with bounded ℓ₁ norm ω), then the iterative approach handles each nonzero coefficient of c independently, with a much larger per-step α = r/‖column of S‖ instead of α = r/‖Sc‖. This could reduce the required Gaussian width by a factor of ~√κ (where κ is the Hamming weight of c), which would reduce the norm blowup from ~13× to ~4–5×.

Whether this is applicable depends on the exact structure of Hachi's base-case opening protocol. If the base case uses a split-and-fold protocol (§4.1–4.2) where the prover sends z = Σ cᵢ·sᵢ, and each cᵢ is a short ring element, then the iterative approach applies directly.

### 4.2. Can sumcheck masking avoid any extra commitment entirely?

With the sum-of-univariates approach, the masking randomness is O(n·d) field elements. If the sumcheck degree d is small (e.g., d = 1 for the linear constraints H_α), then the masking randomness is just O(n) field elements — one per round. These could potentially be derived from a PRG seed committed at the start, with the seed being just λ bits.

If the verifier trusts that the prover used a committed PRG seed to generate the masking, then:
- Commitment: λ-bit seed (negligible cost)
- Verification: verifier re-derives the masking from the seed after opening (no extra PCS opening needed)
- Security: computational ZK (relies on PRG security)

This would make the ZK overhead essentially zero for the sumcheck portion. The only remaining cost is the base-case Gaussian masking.

**Open question:** Does this PRG-based masking compose securely with the lattice-based soundness argument? The concern is that the simulator needs to produce a valid-looking transcript without knowing the witness, but it needs to commit to the PRG seed before seeing the verifier's challenges. This should work in the ROM (Fiat-Shamir) because the simulator controls the random oracle.

### 4.3. What about the range check constraints (H_0)?

The range check H_0(τ_0) = Σ_{u,ℓ} eq(τ_0, (u,ℓ)) · P_b(ẽ_w(u,ℓ)) involves the polynomial P_b(T) = Π_{t=-(b-1)}^{b-1} (T-t) of degree 2b-1. For b = 16, this is degree 31.

In the sumcheck, the round polynomials for H_0 have degree up to 31 in each variable. The masking univariates ρᵢ must also have degree 31. This means each ρᵢ has 32 coefficients, and the verifier must check 32 evaluations per round (or equivalently, the prover sends 32 field elements per round instead of the usual 2 for degree-1 sumcheck).

**Key point:** the high degree from the range check dominates the proof size even in the non-ZK setting. The ZK masking adds one coefficient per round polynomial (32+1 = 33 instead of 32), which is negligible.

So the range check does NOT interact badly with ZK — the overhead from ZK masking is dwarfed by the existing range-check degree.

### 4.4. Can we get ZK "for free" from the commitment's hiding property?

The Ajtai commitment is computationally hiding under MLWE. One might hope that this implies ZK for the whole protocol, without any explicit masking.

This does NOT work, because the sumcheck round polynomials are deterministic functions of the witness (not of the commitment randomness). Even if the commitment hides the witness, the sumcheck transcript reveals partial sums that are not masked by the commitment randomness. A simulator that doesn't know the witness cannot produce valid-looking round polynomials.

The commitment's hiding property ensures that the verifier cannot recover the full witness from the commitment alone. But the sumcheck transcript, together with the commitment, may reveal more information than the commitment alone.

### 4.5. The security notion: ZK vs. commit-and-prove simulatability

LNP22 achieves "commit-and-prove simulatability" rather than standard HVZK (Section 3.2, pp. 20–21). The reason: each protocol run appends new commitments that leak information about the randomness s₂. So the commitment cannot be reused across runs.

For Hachi, this is not a problem: commitments are generated fresh for each opening proof. But it's worth being precise about the security notion:

- **Goal:** zero-knowledge PCS — the opening proof reveals nothing about f beyond f(x) = y
- **Achieved notion (with any of the three paths):** computational ZK in the ROM (Fiat-Shamir), assuming MLWE for commitment hiding + the masking technique for sumcheck ZK
- **Not achieved:** statistical ZK (the commitment is only computationally hiding)

## 5. Concrete instantiation sketch (Path C)

For the concrete parameters from the paper (ℓ = 30, d = 1024, k = 4, q ≈ 2^32):

### First recursion level

- **Witness table**: (μ + n·δ) × d ≈ 226 entries
- **Sumcheck**: ~36 rounds over F_{q^4}, degree up to 31
- **ZK masking**: 36 univariates of degree 31 → 36 × 32 = 1152 random field elements in F_{q^4}
- **Masking commitment**: commit to ~1152 × 16 bytes ≈ 18KB of randomness using one small Ajtai commitment

### Embedding step changes

- **Y**: committed instead of sent in clear. Adds one ring element to commitment (+4KB)
- **y_i**: committed instead of sent in clear. Adds k-1 = 3 extension field elements (~48 bytes)
- **Trace relation**: added as one linear constraint in the sumcheck (negligible overhead)

### Base-case opening

- **Witness size**: 226 ring elements (paper §5.2)
- **Gaussian masking**: σ = 13 · max(‖s‖), M ≈ 3 (repetition rate)
- **With iterative rejection sampling (if applicable)**: σ could be reduced to ~4·max(‖column of S‖), M ≈ 1 (essentially no aborts)
- **Extra proof elements**: the Gaussian-masked response z (one vector of ring elements with wider coefficients)

### Proof size estimate

```
Current (non-ZK):
  First-round sumcheck:           ~7.3KB
  Adaptation + Greyhound subproof: ~47.8KB
  Total:                           ~55.1KB

With ZK (Path C estimate):
  First-round sumcheck:            ~7.3KB  (same — masking doesn't change round poly count)
  Masking randomness commitment:   ~1KB    (small Ajtai commitment)
  Masking randomness opening:      ~1KB    (one small opening)
  Y commitment (was sent in clear): ~4KB   (one ring element commitment)
  y_i commitment:                  ~0.05KB (3 field elements)
  Adaptation + Greyhound subproof: ~47.8KB (same structure)
  Base-case Gaussian overhead:     ~5–10KB (wider opening proof)
  Total:                           ~66–71KB
```

**Overall overhead: ~20–30% proof size, ~20% prover time.**

## 6. Summary of the three paths

```
                       Path A              Path B              Path C
                   (ZK sumcheck +      (ABDLOP integrated)   (Hybrid: field ZK
                    separate mask)                             + lattice base)
─────────────────────────────────────────────────────────────────────────
Proof size          ~2×                 ~1.4×                 ~1.2–1.3×
Prover time         ~2×                 ~1.5×                 ~1.2×
Commitment changes  None                Full redesign         Small (base case)
Sumcheck changes    Mask with full ρ    BDLOP masking         Sum-of-univariates
Base-case changes   None (absorbed)     Gaussian masking      Gaussian masking
Complexity          Low                 High                  Medium
Security notion     Comp. ZK (ROM)      C&P simulatability    Comp. ZK (ROM)
```

## 7. The spark: what makes this feasible

The enabling insight is that **ring switching decouples the two ZK concerns**. Before ring switching, the protocol operates over R_q where ZK is hard (short vectors, lattice structure, no simple masking). After ring switching, the sumcheck runs over F_{q^k} where ZK is easy (standard field-level masking, well-understood techniques, no norm constraints on the mask).

This means Hachi inherently has a "ZK-friendly" structure that Greyhound and LaBRADOR lack: those protocols do their heavy lifting directly over R_q, requiring lattice-specific ZK techniques (JL projections, coefficient masking, Gaussian masking) at every step. Hachi only needs lattice-specific ZK at the base case.

The practical implication: making Hachi ZK should cost ~20–30% overhead (Path C), compared to the ~2× overhead that would be needed for a protocol that operates entirely over R_q.

## 8. Open questions and next steps

1. **Formalize the sum-of-univariates ZK sumcheck** in Hachi's specific constraint structure (H_α and H_0 batched). Verify that the masking degree matches the constraint degree correctly.

2. **Analyze the norm blowup at the base case.** How does the ~13× Gaussian width factor (or ~4× with iterative rejection sampling) feed back into MSIS parameters? Does it require changing q, d, or the decomposition base b?

3. **Can the PRG-seed approach (§4.2) eliminate the masking commitment entirely?** If so, the ZK overhead drops to essentially just the base-case Gaussian masking — perhaps <10% total overhead.

4. **What's the right base-case ZK protocol?** If Hachi hands off to Greyhound at the base, and Greyhound uses LaBRADOR internally, then the ZK base case is "make LaBRADOR ZK." The Falcon paper's PSS framework (predicate special soundness) would be relevant for the knowledge soundness analysis of the non-interactive (Fiat-Shamir) version.

5. **Interaction between ZK and the extension-field embedding (§3).** Committing to Y and y_i instead of sending them in the clear adds constraints to the sumcheck. Does this change the recursion depth or the witness size at subsequent levels?
