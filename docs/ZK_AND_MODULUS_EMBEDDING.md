# Zero Knowledge and Modulus Embedding in Lattice-Based Arguments

Notes from **LNP22** (*Lattice-Based Zero-Knowledge Proofs and Applications*, Lyubashevsky–Nguyen–Plançon, CRYPTO 2022), **AABKT24** (*Aggregating Falcon Signatures with LaBRADOR*, Aardal–Aranha–Boudgoust–Kolby–Takahashi, CRYPTO 2024), and **Gärtner25** (*Compact Lattice Signatures via Iterative Rejection Sampling*, Gärtner, CRYPTO 2025 Best Paper).

---

## Part 1: How Zero Knowledge Works

### The fundamental difficulty

In discrete-log ZK (Schnorr), the simulator picks `z` uniformly and computes the commitment backwards. In lattices, the secret `s` is *short*, so any response `z = cs + y` has a distribution that depends on `s`. You can't just sample `z` uniformly — it would be too large to be a valid response.

### Gaussian masking + rejection sampling

The fix is to sample the mask `y` from a discrete Gaussian D_σ, then **rejection-sample** the output `z = cs + y` so that its distribution is (statistically close to) D_σ regardless of `s`.

Protocol sketch:

1. Prover samples masking vector `y ← D_σ`, sends commitment to `y`
2. Verifier sends challenge `c` (a short polynomial in R_q)
3. Prover computes `z = cs + y`
4. Prover outputs `z` with probability `min(1, D_σ(z) / (M · D_{cs,σ}(z)))`, else aborts and restarts

After rejection sampling, `z ~ D_σ` independent of `s`. The simulator just samples `z ← D_σ` directly — no rewinding. The cost is a repetition rate M ≈ 3 (the prover aborts ~2/3 of the time).

**Three variants** (LNP22, Section 2.6, pp. 15–16):

| Variant | Source | Std dev needed | Repetition rate | What it leaks |
|---------|--------|----------------|-----------------|---------------|
| Rej1 (classical) | [Lyu12] | σ = 13‖v‖ | M ≈ 3 | Nothing |
| Rej2 (one-sided) | [LNS21a] | σ = 0.675·‖v‖ | M ≈ 3 | Sign of ⟨z, cs⟩ (1 bit) |
| Rej0 (bimodal) | [DDLL13] | Same as Rej1 | M instead of 2M | Nothing |

Rej2 is important: it gets a **>10x reduction** in σ by accepting only `z` with ⟨z, v⟩ ≥ 0. The cost is leaking one bit of the secret per run — acceptable when commitments are single-use.

### Iterative rejection sampling (Gärtner, CRYPTO 2025 Best Paper)

The fundamental limitation of all the above variants: the Gaussian width σ (and hence signature/proof size) must scale with `‖Sc‖` — the full secret-times-challenge product. The key ratio is `α = σ/‖v‖` where `v` is the largest vector the rejection sampling must handle. For BLISS-style bimodal sampling, the repetition rate is `exp(1/(2α²))`, so you need `α ≈ 1` to keep rejection probability reasonable, which forces σ to be quite large.

#### The core idea: more freedom in sign selection

In BLISS, the prover picks `z = y + Sc` or `z = y - Sc` with equal probability 1/2, then accepts or rejects. This is rigid: the sign choice is independent of the rejection condition.

Gärtner's insight: give the sampler **more freedom** by coupling the sign choice with the rejection decision. Instead of the BLISS procedure:

```
accept y+v with prob R(y+v)/2
accept y-v with prob R(y-v)/2
reject  otherwise
```

use more general functions f_v(y) and g_v(y):

```
R_v(y):
  (a, b) = (f_v(y), g_v(y))
  r ← U([0,1))
  if r < a:      return y - v
  if r < a + b:  return y + v
  return ⊥
```

The functions f_v and g_v must satisfy three properties (Lemma 1, p. 12):

1. `f_v(z) + g_v(z) ≤ 1` (valid probabilities)
2. `f_v(z) ≥ 0, g_v(z) ≥ 0` (non-negative)
3. `p(z+v)·f_v(z+v) + p(z-v)·g_v(z-v) = p(z)/M` (output follows target distribution)

where `p` is the target distribution and `M` is the repetition rate.

#### Construction of f_v and g_v

For norm-dependent distributions (like Gaussians), define the alternating sum:

```
S_v(y) = Σ_{k≥0} (-1)^k · ρ_r(y + kv) / ρ_r(y)
```

This sum converges extremely fast — 20 terms give error < 2⁻²⁵⁶ even in the worst case.

The key structural fact: when `⟨y,v⟩ ≥ 0`, the terms `p(y+kv)` are strictly decreasing in k, so consecutive pairs in the alternating sum are non-negative, guaranteeing `S_v(y) ≥ 0`. The functions are then:

```
f_v(y) = S_v(y)/M        if ⟨y,v⟩ ≥ ‖v‖²
         (1-S_v(-y))/M    if ⟨y,v⟩ < ‖v‖²

g_v(y) = (1-S_v(y))/M    if ⟨y,v⟩ ≥ -‖v‖²
         S_v(-y)/M        if ⟨y,v⟩ < -‖v‖²
```

The repetition rate is (Lemma 3, Corollary 1, p. 18):

```
M_α = (1 + 2α√(2π) · ρ(πα)) / (ρ_α(1) · (1 - ρ(2πα)))
```

where `α = r/‖v‖`.

#### Why this is so much better: the comparison

```
Rejection probability comparison (log₂ scale):

α       BLISS (bimodal)      Gärtner (new)
0.5     ~2⁰  (rejects ~50%)  ~2⁻¹ (rejects ~30%)
1.0     ~2⁻¹ (rejects ~40%)  ~2⁻⁵
1.5     ~2⁻² (rejects ~20%)  ~2⁻¹⁵
4.0     ~2⁻¹¹                ~2⁻¹⁰⁸ (negligible!)
```

For small α (≈ 0.5–1), the advantage is modest. For larger α, the advantage is **exponential**. The new method's rejection probability drops as `exp(-π²α²/2)` vs the old `exp(-1/(2α²))`.

#### The iterative construction: where it all pays off

The key trick is to **not** rejection-sample over `v = Sc` all at once. Instead, decompose `c` into its nonzero coefficients `c = Σᵢ cᵢ` (each cᵢ has a single nonzero entry), and build z iteratively:

```
z₀ = y
for i = 1, ..., κ:
    zᵢ = R_{vᵢ}(z_{i-1})     where vᵢ = S·cᵢ (a single column of S)
    if zᵢ = ⊥: abort and restart with new y
z = z_κ
```

Each step handles `vᵢ` which is just **one column of S** — much shorter than `Sc`. This means each step can use a much larger `α = r/‖vᵢ‖`, pushing into the regime where rejection is negligible.

The total rejection probability over κ steps is `1 - (1/M)^κ`. With the old bimodal method, making α larger to compensate for κ repetitions does **not** help — the gains and losses roughly cancel. With the new method, making α larger helps **exponentially more** than the linear cost of κ repetitions.

```
Non-iterative (BLISS):       α ≈ 0.7,  handles v = Sc,  ‖v‖ large
Iterative (Gärtner):         α ≈ 4–6,  handles v = column of S,  ‖v‖ ≈ ‖Sc‖/√κ
                             rejection per step < 2⁻¹⁰⁸
                             total rejection after κ steps ≈ κ · 2⁻¹⁰⁸ ≈ negligible
```

#### Concrete results

The paper instantiates this for NTWE-based signatures with ring degree n = 256:

| Scheme | Sig size | VK size | Sig + VK | vs ML-DSA |
|--------|----------|---------|----------|-----------|
| ML-DSA-65 | 3309 B | 1952 B | 5261 B | baseline |
| Gärtner (compact) | ~1100 B | ~1300 B | ~2400 B | < 50% |
| Falcon-512 | 666 B | 897 B | 1563 B | ~30% |
| Gärtner (no-abort, α≥4) | ~1800 B | ~1300 B | ~3100 B | < 60% |

The "compact" variant is comparable to Falcon in combined size. The "no-abort" variant (where rejection can be safely ignored since α ≥ 4 gives rejection prob < 2⁻¹⁰⁸) is still much smaller than ML-DSA.

#### Relevance to ZK arguments

The iterative rejection sampling technique is directly applicable to ZK lattice proofs, not just signatures. In any Fiat-Shamir-with-aborts protocol where the prover computes `z = y + cs` and rejection-samples:

- If the challenge `c` is sparse (few nonzero coefficients), you can rejection-sample one coefficient at a time
- Each step handles a much shorter perturbation vector, allowing a much narrower output distribution
- The result: **tighter masking** with the same or lower abort probability, which translates to smaller proofs

The constraint is that the challenge `c` must be decomposable into sparse components — which is exactly the case in many lattice proof systems where challenges are drawn from sets with bounded ℓ₁ norm or Hamming weight.

### Coefficient masking for constant-term proofs

The norm proof reduces to showing `ct(f(s₁, m)) = 0` where `ct(·)` extracts the constant coefficient. But proving a polynomial relation over R_q would reveal *all* coefficients of `f(s₁, m)`, not just the constant one.

**Fix** (from [ENS20], described in LNP22 Section 1.3, pp. 4–5): commit to a masking polynomial `g ∈ R_q` with `ct(g) = 0` and all other coefficients uniformly random. Given challenge γ ∈ Z_q, the prover sends:

```
h = γ · f(s₁, m) + g
```

The verifier checks `ct(h) = 0`.

- **ZK**: The non-constant coefficients of `f` are perfectly masked by `g`.
- **Soundness**: If `ct(f) ≠ 0`, then Pr_γ[γ · ct(f) + ct(g) = 0] ≤ 1/q₁ where q₁ is the smallest prime factor of q.
- **Amplification**: Repeat with λ independent gᵢ to get soundness error q₁^{-λ}.
- **Batching**: Multiple relations `ct(fᵢ) = 0` compress into one via random linear combination (LNP22 Eq. 8, p. 6):

```
h = Σᵢ γᵢ · fᵢ(s₁, m) + g
```

### Quadratic relations and the automorphism trick

The squared norm ‖s‖² appears as the constant coefficient of `σ_{-1}(s) · s`, where `σ_{-1}: X ↦ X⁻¹` is an automorphism of R_q = Z_q[X]/(X^d + 1) (LNP22 Lemma 2.4).

Given `z = cs + y`, the verifier computes (LNP22 Eq. 11, p. 7):

```
σ(z) · z − c²β² = (σ(s) · s − β²) · c² + (σ(s) · y + s · σ(y)) · c + σ(y) · y
```

This is quadratic in `c`. If the relation holds (i.e., σ(s) · s − β² = 0), the c² coefficient vanishes, leaving a linear equation. Three accepting transcripts on distinct challenges yield a Vandermonde system that forces the c² coefficient to zero.

**The catch**: this only works when σ(c) = c. The challenges must be *fixed* under the automorphism. Concretely (LNP22 p. 7):

```
c = c₀ + Σ_{i=1}^{d/2-1} cᵢ · (Xⁱ − X^{d−i})
```

which has d/2 free coefficients instead of d. This halves the challenge space but has small effect on proof size.

### The security notion: commit-and-prove simulatability

LNP22 does **not** prove standard HVZK. Instead it proves **commit-and-prove simulatability** (Section 3.2, pp. 20–21).

Why: each protocol run appends new BDLOP commitments (the gᵢ, garbage terms, etc.), which leak information about the commitment randomness s₂. So the commitment can't be reused across runs.

The simulator (Theorem 4.5) works via a 3-step hybrid:

- **S0**: Honest commitment, but sample z ← D_σ fresh (statistically close by rejection sampling lemma)
- **S1**: Replace commitment with random + structured offset (indistinguishable by Extended-MLWE)
- **S**: Sample commitment uniformly (identical distribution to S1)

This suffices for applications because commitments are auxiliary — generated fresh for each proof.

---

## Part 2: Embedding a Relation over a Smaller Modulus into a Larger One

### The core problem

You have a relation mod q (e.g., Falcon's `s₁ + h · s₂ = t mod q` with q = 12289), but your proof system works mod q' (LaBRADOR needs q' >> q for Johnson-Lindenstrauss projections). You need to prove the mod-q relation inside the mod-q' proof system.

### Two methods (LNP22, Section 1.3, p. 7)

**Method 1 — Composite modulus**: Set the commitment modulus to p·q and lift the equation by multiplying by p. Challenge differences must be invertible in both R_q and R_p. Simple but requires composite modulus, which increases the masking terms λ (since q₁ gets smaller).

**Method 2 — Explicit wrap-around witness**: This is what the Falcon paper actually uses.

### Method 2 in detail

#### Step 1: Lift to Z by introducing a wrap-around witness

Instead of proving `s₁ + h·s₂ = t mod q`, prove (Falcon paper Eq. 6, Section 6.1):

```
s₁ + h · s₂ + q · v − t = 0    over Z (no modular reduction)
```

where `v ∈ R` is the "carry" witness. If this holds over Z, it holds mod any modulus — including the original q.

#### Step 2: Prove the lifted equation mod q' and prevent q'-wrap-around

The lifted equation is proved inside LaBRADOR mod q'. But we need to guarantee it also holds over Z, not just mod q'. This requires proving that **all coefficients are small enough that no wrap-around mod q' occurred**.

#### Step 3: ℓ∞ smallness proof via JL projection (Falcon paper Section 6.2, pp. 21–23)

Two separate ℓ∞ checks are needed:

**Check A** — For the norm constraint. The four-square decomposition (see below) gives a constant-term equation in R_{q'}. For it to hold over Z, all input coefficients must satisfy:

```
‖(s₁ ‖ s₂ ‖ s₁' ‖ s₂' ‖ ε ‖ ε')‖_∞ < sqrt(q' / (2(2d+4)))
```

**Check B** — For the Falcon verification equation. Since `h · s₂` amplifies coefficients by a factor of dq:

```
‖s₁‖_∞ < q'/6
‖s₂‖_∞ < q'/(6dq)
‖v‖_∞  < q'/(6q)
```

Both are proved using LaBRADOR's built-in JL projection: sample a random {-1,0,1} matrix R, project the witness down to ~256 dimensions, check the projected norm. By JL, small projected norm implies small actual ℓ₂ norm with probability ≥ 1 − 2⁻¹²⁸. Small ℓ₂ implies small ℓ∞ (up to dimension factors).

**The approximate range proof** (LNP22 Section 5.1, Figure 9, pp. 38–40) introduces a **slack factor of ~189x** between the proven bound and the actual bound. This is why q' must be so much larger than q.

#### Step 4: Constraint on q' (Falcon paper Eq. 10, p. 23)

Combining the JL and completeness conditions:

```
q' > (1024/15) · (d + 2) · β² · N
```

where d = ring degree, β = Falcon signature norm bound, N = number of aggregated signatures.

Concrete numbers:

| Setting | Ring degree | Modulus requirement |
|---------|------------|---------------------|
| Falcon-512, N = 2²⁰ | d = 512 | q' > 2⁶⁰·¹² (61-bit modulus) |
| Falcon-1024, N = 2²⁰ | d = 1024 | q' > 2⁶²·¹⁶ (63-bit modulus) |

#### Step 5: Four-square decomposition for exact norm proofs

To prove ‖s‖² ≤ β² exactly (not approximately), both papers use Lagrange's four-square theorem. Find ε₀, ε₁, ε₂, ε₃ such that:

```
β² − ‖s₁‖² − ‖s₂‖² = ε₀² + ε₁² + ε₂² + ε₃²
```

Pack into a polynomial ε = ε₀ + ε₁X + ε₂X² + ε₃X³ and use the automorphism trick:

```
ct(σ_{-1}(s₁) · s₁ + σ_{-1}(s₂) · s₂ + σ_{-1}(ε) · ε) = β²
```

This is a constant-term relation in R_{q'}, provable via the coefficient-masking technique from Part 1.

#### Step 6: Subring optimization (Falcon paper Section 6.4)

Falcon uses degree d ∈ {512, 1024}, but LaBRADOR proof sizes scale with ring degree. Using a norm-preserving map φ: R^n_{q'} → S^{cn}_{q'} from [LNPS21], the witness is moved from degree-d ring R to degree-d/c subring S (doubling the module rank but halving the degree). This gives ≥ 2x reduction in proof size.

### Full pipeline diagram

```
┌───────────────────────────────────────────────────────────────┐
│  ORIGINAL RELATION (mod q, e.g. q = 12289)                   │
│  s₁ + h·s₂ ≡ t (mod q),   ‖(s₁,s₂)‖ ≤ β                   │
└───────────────────────────────┬───────────────────────────────┘
                                │
                    (1) introduce wrap-around witness v
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  LIFTED RELATION (over Z)                                     │
│  s₁ + h·s₂ + q·v = t                                         │
│  β² − ‖s₁‖² − ‖s₂‖² = ε₀² + ε₁² + ε₂² + ε₃²  (4-square)   │
└───────────────────────────────┬───────────────────────────────┘
                                │
                    (2) embed into R_{q'}, add σ_{-1} witnesses
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  PROOF SYSTEM RELATIONS (mod q')                              │
│  Linear:     s₁ + h·s₂ + q·v − t = 0           mod q'       │
│  Quadratic:  ct(s₁'·s₁ + s₂'·s₂ + ε'·ε − β²) = 0  mod q'   │
│  Conjugate:  ct(σ_{-1}(Xʲ)·a − b) = 0          mod q'       │
└───────────────────────────────┬───────────────────────────────┘
                                │
                    (3) approximate ℓ∞ proof (JL projection)
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  NO WRAP-AROUND GUARANTEE                                     │
│  All coefficients small enough ⟹ mod q' = over Z ⟹ mod q     │
│  Requires: q' > (1024/15)(d+2)β²N                            │
└───────────────────────────────┬───────────────────────────────┘
                                │
                    (4) move to subring (degree d → d/c)
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│  FINAL LaBRADOR PROOF (over subring S_{q'})                   │
│  ≥ 2x smaller proofs                                          │
└───────────────────────────────────────────────────────────────┘
```

---

## Remarks

1. **Slack from approximate range proof** (LNP22 Section 5.1): The JL projection introduces a ~189x slack between the proven and actual bound. This directly drives how large q' must be relative to q.

2. **Automorphism halves the challenge space** (LNP22 p. 7): Requiring σ(c) = c leaves only d/2 free coefficients. For small ring degrees (64 or 128 in LNP22), this is fine. For larger degrees it has even less impact.

3. **Four-square decomposition cost**: Lagrange's theorem is existential; computing the decomposition requires Rabin-Shallit or similar. One-time prover cost, not free but standard.
