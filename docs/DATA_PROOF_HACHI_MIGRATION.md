# Data Proof: Dory → Hachi Migration Analysis

This document analyzes the impact of replacing Dory (pairing-based PCS) with Hachi (lattice-based PCS) on the **Data Proof** protocol used in Zero's OpenZoneBlockTx.

Primary sources:

- `../Cyclone-Megadoc/protocol-spec/beacon/data_proof.md` — the Data Proof spec
- `../Cyclone-Megadoc/protocol-spec/beacon/da.md` — the Semi-AVID-PR / VID protocol
- `~/Downloads/Semi-AVID-PR and Jolt.pdf` — Saeid's note on shared-SRS optimization
- `docs/HACHI_KEY_DELEGATION.md` — Hachi two-level protocol with equations
- `docs/HACHI_FOR_JOLT.md` — Jolt integration spec

---

## 1. System Context: Where the Data Proof Sits

Zero (Cyclone) settles zone blocks via a three-phase protocol:

```
  PREPARE                    COMMIT                      OPEN
  ─────────                  ──────                      ────
  Sequencer runs VID         Sequencer submits           Sequencer submits
  with beacon validators     CommitZoneBlockTx           OpenZoneBlockTx
  ──────────────────         ──────────────────          ──────────────────
  Outputs:                   Includes:                   Includes:
  • Column commits (h_j)     • CoA                       • Jolt SNARK proof
  • CoA                      • Column commits             • DATA PROOF ◄── this document
  • RS-encoded chunks        • State root
```

The Data Proof bridges two independent commitment systems:

- **DA layer** (Prepare): column commitments \(h_j = \text{Pedersen}(\text{column}_j)\) via Semi-AVID-PR
- **Jolt ZKVM** (Open): polynomial commitment to the same data matrix \(U\) via Dory

It proves: "the data Jolt proved about is the same data the DA layer committed to."

### 1.1 Two data objects (critical distinction)

There are two different objects that are easy to conflate:

1. **DA block data \(U\)**: raw zone-block transaction bytes, transformed into field elements for VID and committed column-wise in Prepare/Commit.
2. **Jolt witness tables**: execution-trace polynomials committed in Stage 8 (one-hot address tables, increment tables, and optional advice tables).

The Data Proof is the bridge between these worlds. The "native small values" observation applies strongly to Jolt witness tables, but not automatically to raw DA block bytes.

### 1.2 Current DA packing path

Current VID docs use BLS12-381 field arithmetic for encoding. The documented packing rule is:

- each 32-byte field element stores 31 bytes of payload.

So the current DA path is:

`raw bytes -> BLS12-381 field elements -> matrix U -> RS encode -> Pedersen column commitments`.

---

## 2. Current Data Proof Protocol (Dory-Based)

### 2.1 Setup

- Data matrix \(U \in \mathbb{F}^{L \times k}\) encodes the zone block.
- DA column commitments: \(C_j = \sum_{i} U[i,j] \cdot g_i \in \mathbb{G}_1\) (Pedersen with key \(\vec{g}\)).
- Polynomial: \(f(X) = \sum_{i,j} U[i,j] \cdot X^{j + ki}\).
- Evaluation claim: \(f(x) = y\).

### 2.2 VMV Decomposition (PCS-Independent)

The evaluation factors as a vector-matrix-vector product:

\[
f(x) = \vec{b}^T U \vec{a}
\]

where \(\vec{a} = (1, x, \ldots, x^{k-1})\) and \(\vec{b} = (1, x^k, \ldots, x^{(L-1)k})\).

Defining \(\vec{w} = U\vec{a} \in \mathbb{F}^L\), the claim becomes: know \(\vec{w}\) with \(\langle \vec{w}, \vec{b} \rangle = y\) and \(\vec{w}\) consistent with \(U\).

### 2.3 BabyHyrax Baseline (Non-Succinct)

Prover sends \(\vec{w}\) in the clear (\(O(L)\) field elements). Verifier checks:

1. **Correctness**: \(\langle \vec{w}, \vec{b} \rangle \stackrel{?}{=} y\)
2. **Consistency**: \(\sum_i w_i \cdot g_i \stackrel{?}{=} \sum_j a_j \cdot C_j\)

### 2.4 Homomorphic Reduction (Key Trick)

Because Pedersen is **linear in the message**, the verifier computes locally:

\[
C' = \sum_{j \in [k]} a_j \cdot C_j = \sum_j a_j \left(\sum_i U[i,j] \cdot g_i\right) = \sum_i \left(\sum_j U[i,j] \cdot a_j\right) g_i = \text{Commit}(\vec{w}, \vec{g})
\]

This \(O(k)\) MSM gives the verifier a commitment to \(\vec{w}\) without receiving it. The problem reduces to: "prove you know \(\vec{w}\) that opens \(C'\) and satisfies \(\langle \vec{w}, \vec{b} \rangle = y\)."

### 2.5 Dory Engine (Succinct)

1. **Lift**: \(C'' = e(C', h) \in \mathbb{G}_T\) (pairing)
2. **Recursive folding**: \(\log L\) rounds, each halving vector size via challenges \(\alpha_i\)
3. **Key-folding**: precomputed \(\Delta\) values let verifier fold commitment keys in \(O(1)\) per round
4. **Tensor structure**: \(\vec{b}\)'s multiplicative structure lets verifier compute final folded scalar in \(O(\log L)\)
5. **Final check**: one multi-pairing in \(\mathbb{G}_T\)

### 2.6 Shared-SRS Optimization (Saeid's PDF)

If Dory generators \(g_i = \tau^i \cdot g\) come from the KZG SRS, then \(C_j = \hat{C}_j\) — the DA column commitment and the Dory sub-commitment are literally the same object. No extra consistency proof needed.

### 2.7 Audit Notes: Conventions and Trust Model

- **Row/column naming mismatch is mostly convention**: the Cyclone Data Proof spec is written in terms of **column commitments** \(C_j\), while Dory/Jolt implementation exposes **row commitments** as tier-1 hints. This is transpose-equivalent bookkeeping (with swapped VMV vectors), not a fundamental protocol contradiction.
- **The "lift" step is an abstraction, not a bug**: the spec writes \(C''=e(C',h)\). The implementation computes the pre-lift commitment by MSM over tier-1 commitments, then applies a pairing; same algebraic object, different decomposition.
- **CRS coupling caveat**: native Dory setup is transparent (random generators from public randomness). If generators are instead taken from a KZG powers-of-\(\tau\) SRS to force object identity, the system inherits KZG trusted-setup assumptions. No direct new attack is known from this audit, but transparency is no longer a valid claim in that mode.

---

## 3. The Gadget Decomposition Problem

### 3.1 Why Ajtai Is Not Linearly Homomorphic

Ajtai commitment: \(h_j = A \cdot G^{-1}(u_j)\), where \(G^{-1}\) is base-\(b\) digit decomposition.

The map \(u \mapsto G^{-1}(u)\) is **not linear** (carries break digit decomposition):

\[
\sum_j a_j \cdot G^{-1}(u_j) \neq G^{-1}\!\left(\sum_j a_j \cdot u_j\right)
\]

Therefore:

\[
\sum_j a_j \cdot h_j = A \cdot \underbrace{\left(\sum_j a_j \cdot G^{-1}(u_j)\right)}_{\neq\, G^{-1}(w)} \neq A \cdot G^{-1}(w)
\]

The verifier's linear combination produces something that is **not** a valid Ajtai commitment to \(w = Ua\).

**This breaks the Data Proof's core trick.** (Confirmed by `HACHI_KEY_DELEGATION.md` which states: "The earlier 'homomorphic batching' idea ... is NOT valid for Hachi commitments, because the commitment map includes digit decompositions and is not linear.")

### 3.2 Consequences for Semi-AVID-PR

The DA protocol's validator check (`da.md` lines 109–114) relies on:

\[
\text{Code.Encode}(\text{Commit}(u_1), \ldots, \text{Commit}(u_k))_i = \text{Commit}(\text{Code.Encode}(u_1, \ldots, u_k)_i)
\]

This is exactly linearity of the commitment map. With Ajtai + gadget decomposition, this identity fails. **Naively swapping Pedersen for Ajtai in the DA layer breaks Semi-AVID-PR's verification mechanism.**

### 3.3 Consequences for the Shared-SRS Trick

The Dory trick relied on a triple coincidence:

1. Pedersen linearity → homomorphic reduction works
2. Shared generators → column commitment = Dory sub-commitment (zero overhead)
3. Pairing lift → enters Dory IPA world

All three are lost with Hachi. There is no analog of "column commit = sub-commitment" with the two-layer Ajtai structure.

---

## 4. How Hachi Solves the Gadget Linearity Problem

Hachi's own opening protocol (Step B, `HACHI_KEY_DELEGATION.md` §1.3) never relies on homomorphic combination of Ajtai commitments. Instead, it uses a **auxiliary commitment + fold** mechanism.

### 4.1 The Pattern

Given committed blocks \(f_i\) with decompositions \(s_i = G^{-1}(f_i)\) and inner commitments \(t_i = A \cdot s_i\):

**Step 1: Prover sends fresh auxiliary commitment.**
- Compute partial evaluations: \(w_i = a^T \cdot G \cdot s_i = a^T \cdot f_i\)
- Freshly decompose and commit: \(\hat{w}_i = G^{-1}(w_i)\), \(v = D \cdot \hat{w}\)
- Send \(v\) (first prover message)

**Step 2: Verifier sends fold challenge.**
- Random sparse \(c = (c_1, \ldots, c_{2^r})\)

**Step 3: Prover folds on already-decomposed witnesses.**
- \(z = \sum_i c_i \cdot s_i\) (linear combination of vectors already in decomposed form)
- Send \(z\)

**Step 4: Verifier checks dual consistency through committed witnesses.**

The verifier does **not** receive all \(t_i\) explicitly. Instead, it checks a stacked relation over witnesses \((\hat{w}, \hat{t}, z)\):

- **Main-binding constraint**: \(B\hat{t} = u\) (binds \(\hat{t}\) to the public commitment \(u\)).
- **Eq. 19 (fold-inner consistency)**: \(A z = (c^T \otimes G_{n_A}) \cdot \hat{t}\).
- **Eq. 18 (fold-partials consistency)**: \((a^T G)\cdot z = (c^T \otimes G_1)\cdot \hat{w}\).

In the full protocol, \(z\) is represented via redecomposition \(z = J\hat{z}\), and these constraints are enforced inside Step C's ring-switch + sumcheck system.

### 4.2 Why This Works

The nonlinearity of \(G^{-1}\) is quarantined to two places where it doesn't interact with linear combinations:

- **Commitment time** (one-time): \(s_i = G^{-1}(f_i)\)
- **Auxiliary commitment time** (per reduction): \(\hat{w}_i = G^{-1}(w_i)\)

All subsequent operations use only:
- Linearity of \(A\) (matrix-vector multiply)
- Linearity of \(G\) (the forward gadget map, trivially a matrix)
- The identity \(G \cdot G^{-1}(x) = x\)

### 4.3 Soundness Sketch

If the prover commits to incorrect partial evaluations \(w'_i \neq a^T f_i\), then after seeing random \(c\), it must still produce witnesses satisfying:

1. \(B\hat{t}=u\) (bound to the committed polynomial),
2. \(A z = (c^T \otimes G_{n_A})\hat{t}\),
3. \((a^T G)z = (c^T \otimes G_1)\hat{w}\),
4. the opening equation.

These constraints force one folded witness \(z\) to satisfy two independently anchored linear views (main-commit anchor and auxiliary/evaluation anchor). A cheating assignment survives only with negligible probability over \(c\), assuming standard ROM/low-degree soundness used by Hachi.

---

## 5. Adapted Data Proof Architecture

### 5.1 Option A: Keep Pedersen for DA, Add Bridge Proof (Recommended)

```
┌──────────────────────────────────────────────────────┐
│  DA Layer (unchanged)                                 │
│  Column commits h_j = Pedersen(column_j) in G₁       │
│  Semi-AVID-PR linearity argument intact               │
└──────────────────────┬───────────────────────────────┘
                       │ h_1,...,h_k public
                       ▼
┌──────────────────────────────────────────────────────┐
│  BRIDGE PROOF (new component)                        │
│                                                      │
│  Prover: commit same U via Hachi (two-layer Ajtai)   │
│  Prove consistency using Hachi's Step B pattern:     │
│    1. Send auxiliary commitment v = D · ĝ(w)         │
│    2. Receive fold challenge c                        │
│    3. Send folded witness z = Σ c_i · s_i            │
│    4. Verifier checks stacked constraints            │
│       (Bĥt=u, Eq.18, Eq.19, opening equation)        │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│  HACHI OPENING PROOF                                 │
│  Step B: evaluation reduction                        │
│  Step C: ring-switch + sumcheck over F_{q^k}         │
└──────────────────────────────────────────────────────┘
```

**Trade-offs:**

- (+) DA layer untouched, Semi-AVID-PR works as-is
- (+) Bridge proof uses Hachi's own machinery (no new primitives)
- (-) Bridge proof adds prover cost (one auxiliary commitment + one fold)
- (-) Loses the "zero-overhead" consistency of the Dory shared-SRS trick
- (-) Requires an explicit DA↔Hachi consistency relation (not provided "for free" by Jolt Stage-8 batching)

### 5.2 Option B: Lattice-Native DA Without Gadget Decomposition

If data elements are bounded (e.g., bits, one-hot entries), use Ajtai **without** gadget decomposition:

\[
h_j = A \cdot u_j \quad\text{where } \|u_j\|_\infty \leq B
\]

Module-SIS binding holds for short pre-images. The commitment map \(u \mapsto A \cdot u\) IS linear, so Semi-AVID-PR works and the homomorphic reduction works.

**Trade-offs:**

- (+) Full linearity restored, both DA and Data Proof work natively
- (+) No bridge proof needed
- (-) Only works if data entries have bounded norm (must satisfy SIS security bound)
- (-) Loses binding for arbitrary field data with large entries
- (-) Needs careful parameter analysis: SIS norm bound vs data magnitude
- (-) RS encoding can amplify entry magnitude; bounded raw data does **not** automatically imply bounded encoded chunks

**Important clarification:** Option B may be plausible for some **Jolt witness commitments** (one-hot + bounded increments), but it is generally not plausible for current **raw DA block bytes** without a redesign of DA data representation.

### 5.3 Option C: Redesign DA Around Hachi's Commitment Structure

This is the "single commitment family" path: DA and Open both use Hachi, and Semi-AVID-PR's linearity-based check is replaced.

#### 5.3.1 What must be replaced

Current Semi-AVID-PR validator logic uses:
\[
\mathrm{Code.Encode}(h_1,\ldots,h_k)_i \stackrel{?}{=} \mathrm{Commit}(c_i)
\]
which depends on commitment linearity. With Hachi commitments, this check is invalid.

So Option C must replace that check with a proof-backed statement:
"the received chunk \(c_i\) is the correct RS encoding of data committed in Hachi."

#### 5.3.2 Option C1 (incremental): per-validator chunk-consistency proof

For each validator \(i\), the sequencer sends chunk \(c_i\) plus proof \(\pi_i\):

1. Commit original columns with Hachi: \((u_j)_{j\in[k]}\).
2. Validator derives a random evaluation point \(r\) (Fiat-Shamir).
3. Prover opens needed committed columns and chunk at \(r\) (batched opening).
4. Verifier checks RS relation at \(r\):
   \[
   c_i(r) \stackrel{?}{=} \sum_{j=1}^k \lambda_{i,j}\,u_j(r),
   \]
   where \(\lambda_{i,j}\) are public RS coefficients.

**Pros**: easiest way to remove linearity assumption.

**Cons**: proof load scales with validators/chunks unless heavily shared/batched.

#### 5.3.3 Option C2 (scalable target): one global encoding proof

Commit both source and encoded tables under Hachi:

- source columns \(U\),
- encoded columns \(E=\mathrm{Code.Encode}(U)\).

Publish one global proof \(\pi_{\mathrm{enc}}\) that enforces:
\[
\forall i,\quad E_{:,i} = \sum_{j=1}^k \lambda_{i,j} U_{:,j}.
\]

Implementation direction: encode this as a stacked linear relation and prove it with the same ring-switch + sumcheck machinery used by Hachi Step C.

Then validator-side chunk checks become:

- verify \(\pi_{\mathrm{enc}}\) once per block (or once on-chain),
- verify chunk opening against the committed \(E_{:,i}\).

**Pros**: amortizes proof cost; cleanest long-term Option C architecture.

**Cons**: substantial protocol redesign and engineering effort.

#### 5.3.4 Recommended Option C roadmap

1. **Spec freeze**: define exact matrix orientation and RS coefficient indexing (remove row/column ambiguity first).
2. **C1 prototype**: implement per-validator proof to de-risk correctness quickly.
3. **C2 upgrade**: replace C1 with one global encoding proof once relation encoding is stable.
4. **Integrate with Open path**: share transcript/challenge infrastructure with Hachi opening to avoid duplicated proof logic.

### 5.4 Do we still need byte packing? What if DA encoding uses a 128-bit field?

#### 5.4.1 Packing is still needed for byte-native blocks

If zone blocks remain byte-native, any field-based RS pipeline still needs a reversible bytes-to-field encoding step. So "packing" does not disappear; only its efficiency and algebraic target field change.

#### 5.4.2 Can we switch only the RS field to 128-bit under current Pedersen DA?

Not safely. Semi-AVID-PR's validator check relies on commitment linearity over the **same scalar field** as coding.

If RS is over \(F_p\) (128-bit) but Pedersen scalars are over \(F_r\) (e.g. BLS12-381 scalar field), then generally:
\[
\mathrm{Commit}_{F_r}\!\left(\sum_j \lambda_j u_j \bmod p\right)
\neq
\sum_j \iota(\lambda_j)\,\mathrm{Commit}_{F_r}(u_j),
\]
because \(\iota\) is only an encoding into \(F_r\), not a field isomorphism \(F_p \cong F_r\).

So a "field-size-only swap" breaks the linear-consistency check unless commitments are also moved to the same algebraic field/ring used by encoding (which is exactly the kind of redesign Option C entails).

#### 5.4.3 Packing efficiency trade-off at 128-bit

- Current BLS12-style rule: 31 bytes per field element.
- Simple 128-bit prime rule (byte-aligned, fixed): typically 15 bytes per element.

So naive 128-bit encoding can **increase** element count for the same raw bytes. A near-\(2^{128}\) prime (e.g. P13) allows more specialized packing strategies, but those add codec complexity and must preserve deterministic invertibility.

Concrete scale example for a 1 MiB block:

- 31-byte packing: \(\lceil 1{,}048{,}576 / 31 \rceil = 33{,}826\) field elements.
- 15-byte packing: \(\lceil 1{,}048{,}576 / 15 \rceil = 69{,}906\) field elements.

That is about \(2.07\times\) more elements under naive 15-byte packing.

For a near-\(2^{128}\) prime like P13 (\(2^{128}-8207\)), only 8,207 of the \(2^{128}\) 16-byte values are out of range (\(\approx 2^{-115}\) fraction). This suggests a specialized 16-byte codec may be practical, but it still requires explicit overflow handling and adversarially robust, deterministic decoding rules.

#### 5.4.4 Is 128-bit security sufficient?

For Option C-style unified Hachi commitments over a 128-bit modulus, the current repo analysis indicates "yes" for candidate parameters:

- `docs/HACHI_FOR_JOLT.md` Section 5.6 reports Module-SIS estimates above 128 bits for recommended P13 configurations, including conservative rows still around ~131 bits in the tight regime.

Additional notes:

- RS availability/correctness is information-theoretic and mainly requires \(q \gg n\) for committee-size \(n\) (128-bit is far above practical \(n\)).
- Any randomized polynomial-identity checks over a 128-bit field still need an explicit soundness budget (\(\approx \deg/q\)-style terms), but this is typically negligible at deployed degrees.

#### 5.4.5 Security checklist if DA moves to 128-bit encoding

If Option C moves DA encoding and commitments into a 128-bit algebraic domain, security is fine **only if** the codec is part of the proved relation:

1. **Canonical decoding**: bytes \(\leftrightarrow\) field elements must be injective and deterministic.
2. **Metadata binding**: codec ID, padding length, and any auxiliary channels must be included in the committed message and verified.
3. **No hidden side channels**: if using an overflow-assisted codec, the overflow bitmap/payload must be committed and proven consistent.
4. **Soundness accounting**: add codec-related checks to the same transcript/sumcheck soundness budget (do not treat codec validity as out-of-band).

### 5.5 Option C codec decision table (implementation)

To make Option C concrete, define:

- \(B\): block size in bytes,
- \(e\): payload bytes per field element (codec-dependent),
- \(w\): serialized bytes per field element (\(w=16\) for 128-bit field, \(w=32\) for current BLS-style),
- \(n\): committee size,
- \(S_0\): max share size (bytes).

Then:
\[
M=\left\lceil\frac{B}{e}\right\rceil,\quad
d_{\max}=\left\lfloor\frac{S_0}{w}\right\rfloor,\quad
k=\left\lceil\frac{M}{d_{\max}}\right\rceil,\quad
d=\mathrm{nextPow2}\!\left(\left\lceil\frac{M}{k}\right\rceil\right),\quad
r=\frac{k}{n}.
\]

#### 5.5.1 Codec options

| Codec | Domain | Payload \(e\) | Deterministic worst-case size | Engineering risk | Notes |
|---|---|---:|---|---|---|
| **Current baseline** | BLS-style | 31/elem (\(w=32\)) | Yes | Low | Existing VID docs/spec assumptions |
| **Fixed-15** | 128-bit field | 15/elem (\(w=16\)) | Yes | Low | Simple, robust, easy to audit |
| **Packed-16 + overflow channel** | 128-bit near-\(2^{128}\) prime | ~16/elem expected | **Only if overflow is explicit and bounded** | Medium/High | Higher average efficiency, but adversarial overflow handling is mandatory |

#### 5.5.2 Concrete comparison (1 MiB block, \(n=1024\), \(S_0=4096\))

| Codec | \(M\) source elems | \(d_{\max}\) | \(k\) | \(d\) | \(r=k/n\) |
|---|---:|---:|---:|---:|---:|
| 31/32 baseline | 33,826 | 128 | 265 | 128 | 0.2588 |
| 15/16 fixed | 69,906 | 256 | 274 | 256 | 0.2676 |
| 16/16 idealized | 65,536 | 256 | 256 | 256 | 0.2500 |

Interpretation:

- Moving from 31/32 to fixed 15/16 approximately doubles source element count \(M\), but because \(d_{\max}\) also doubles (16-byte elements), \(k\) only changes moderately in this sizing regime.
- "16/16 idealized" shows the upper bound if every 16-byte chunk maps directly.

#### 5.5.3 Overflow caveat for packed-16 codecs

For P13 (\(2^{128}-8207\)), random data hits out-of-range 16-byte words with probability \(\approx 2^{-115}\), so average overhead is negligible.

But protocol design must be **adversarially robust**, not average-case:

- a malicious block producer can choose bytes that maximize overflows,
- so overflow artifacts must be explicit and bounded,
- and a deterministic fallback (e.g., re-encode with fixed-15) should be specified when overflow budget is exceeded.

#### 5.5.4 Recommendation for Option C

1. **Phase 1 (safe default):** fixed-15 codec for consensus-critical deployment.
2. **Phase 2 (optional optimization):** packed-16 codec behind a strict overflow-cap + deterministic fallback rule.
3. In both phases, include `codec_id` and all codec metadata/artifacts in the committed statement and in the Option C encoding proof relation.

---

## 6. Additional Domain Mismatches

### 6.1 Scalar Field vs Ring Domain

| | Data Proof (Dory) | Hachi |
|---|---|---|
| \(\vec{w}\) lives in | \(\mathbb{F}^L\) (scalar field) | \(R_q^{2^r}\) (ring) |
| Inner product | \(\langle w, b \rangle = y\) in \(\mathbb{F}\) | \(b^T w\) in \(R_q\), then ring-switch to \(F_{q^k}\) |
| Evaluation point | \(x \in \mathbb{F}\) | \(x \in R_q^\ell\) or \(\mathbf{r} \in F_{q'}^n\) (two-field) |

For the packed regime \(d > 1\), `HACHI_FOR_JOLT.md` §4.3.2A identifies a typing issue: the partial evaluation \(w_i = a^T G s_i\) with \(a \in F_{q'}^{2^m}\) and \(G s_i \in R_q^{2^m}\) produces \(w_i \in F_{q'}[X]/(X^d+1)\), not \(R_q\). Resolution requires a Phase 0 inner evaluation sumcheck (\(\alpha = \log_2 d\) extra rounds).

### 6.2 Jolt CommitmentScheme Trait

Jolt's `CommitmentScheme` trait (`jolt-core/src/poly/commitment/commitment_scheme.rs`) assumes:

1. `combine_commitments(commitments, coeffs)` — homomorphic combination (broken for Ajtai)
2. `opening_point: &[F]` — single-field opening (Hachi needs two moduli \(q, q'\))
3. `Polynomial<F>` — field-native polynomial (Hachi works over rings)
4. Stage-8 batch opening relies on additive combination (`combine_hints` / `combine_commitments`) for internal RLC reductions; this is not the DA Data Proof itself, but it depends on the same algebraic property that breaks under naive Ajtai

Significant trait redesign needed to accommodate Hachi's algebraic structure.

---

## 7. Summary of Findings

| Property | Dory (current) | Hachi (proposed) | Status |
|---|---|---|---|
| Commitment linearity | Pedersen: fully linear | Ajtai + \(G^{-1}\): **not linear** | **Breaking change** |
| Homomorphic reduction | Verifier derives \(C'\) for free | Must use auxiliary commit + fold | Redesigned |
| DA layer compatibility | Semi-AVID-PR linearity intact | Breaks with naive Ajtai swap | Keep Pedersen or use Option B |
| Shared-SRS / shared-matrix | Column commit = Dory sub-commit | No analog exists | Bridge proof needed |
| Succinct opening engine | Pairing lift + IPA folding | Ring-switch + sumcheck | Full replacement |
| Verifier efficiency | \(O(\log L)\) via key-folding + tensor trick | \(O(\ell)\) via sumcheck | Comparable |
| Security | Pre-quantum | Post-quantum | Upgrade |
| Setup (native mode) | Transparent (public randomness) | Transparent (public seed) | Comparable |
| Setup (shared-SRS mode) | Inherits KZG trusted setup | N/A | Trust-model downgrade |

### What survives unchanged

- VMV decomposition \(f(x) = \vec{b}^T U \vec{a}\) (pure algebra)
- Prepare-Commit-Open lifecycle (protocol-level)
- Semi-AVID-PR structure (if Pedersen is kept for DA)

### What must be rebuilt

- The homomorphic reduction → auxiliary commitment + fold consistency pattern
- The Dory IPA engine → Hachi Step B + Step C
- The bridge between DA commitments and PCS commitment
- Jolt's `CommitmentScheme` trait boundary

---

## 8. Audit Outcomes (Updated)

### 8.1 Resolved points

1. **Row vs column mismatch**: resolved as a convention/transpose issue, not a protocol break.
2. **Data Proof vs Jolt Stage-8**: Stage-8 batching is not the DA bridge itself, but migration is entangled because both rely on additive homomorphism.
3. **"Lift" wording**: \(C''=e(C',h)\) is an acceptable abstraction of implementation flow.
4. **Eq. 19 verifier access**: direct check against hidden \(t_i\) is not possible from \(u\) alone; must be enforced via stacked witness constraints (\(B\hat t=u\), Eq. 18, Eq. 19) inside Step C.
5. **Shared-SRS trust model**: if Dory reuses KZG powers-of-\(\tau\), transparency is lost and trusted-setup assumptions are inherited.
6. **Data-model split**: "native-small-values" is accurate for many Jolt witness tables, but not for raw DA block bytes under current packing.
7. **128-bit RS under current Pedersen DA**: not a drop-in change; field mismatch breaks the linear-consistency invariant unless commitment algebra is redesigned too.

### 8.2 Remaining quantitative work

1. **Sparse-challenge soundness constants**: derive explicit end-to-end error term for the exact challenge distribution and fold depth used in production parameters.
2. **Option B bound under RS encoding**: compute concrete SIS margins after encoding (not just on raw witness entries).
3. **Option A bridge cost**: benchmark prover/verifier overhead relative to base Hachi opening.
4. **Option C execution plan**: evaluate C1 vs C2 with concrete proof-size and validator-throughput targets.
5. **Option C + 128-bit codec design**: choose a deterministic byte-to-field codec (fixed 15-byte vs near-\(2^{128}\) specialized), then quantify throughput/proof-size impact.
