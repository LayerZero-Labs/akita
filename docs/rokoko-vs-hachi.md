# RoKoko vs Hachi

A technical comparison of two lattice-based polynomial commitment schemes with
transparent setup and post-quantum security.

| | **RoKoko** (eprint 2026/575) | **Hachi** |
|---|---|---|
| Authors | Klooß, Lai, Nguyen, Osadnik, Tucci | — |
| Lineage | LaBRADOR → RPS → RnR → SALSAA → **RoKoko** | Independent design |
| Ring | \( R_q = \mathbb{Z}_q[X]/(X^{128}+1) \), \( q \approx 2^{50} \) | Cyclotomic rings, configurable degree |
| CRS | Tensor-structured random matrices (Ajtai) | Unstructured random matrices |
| Proof size | ~214 KB (2²⁸ elements) | Configurable via schedule |
| Prover | \( O(m) \) ring ops (linear) | Two-stage sumcheck |
| Verifier | \( O(\lambda^3 \log m / \log \lambda) \) ring ops (polylog) | Polylog |
| Implementation | Rust + AVX-512-IFMA, ~15K LOC | Rust + Rayon, modular crate |

**Paper library:** All referenced papers are in `~/Documents/Papers/`. The
RoKoko implementation is cloned at `../rokoko/`.

---

## Table of Contents

1. [Protocol Architecture](#1-protocol-architecture)
2. [Commitment Schemes](#2-commitment-schemes)
3. [Folding](#3-folding)
4. [Random Projections and the JL Lineage](#4-random-projections-and-the-jl-lineage)
5. [Sumcheck Integration](#5-sumcheck-integration)
6. [Ring Arithmetic and NTT](#6-ring-arithmetic-and-ntt)
7. [Norm Bounds and Slack](#7-norm-bounds-and-slack)
8. [Concrete Parameters](#8-concrete-parameters)
9. [Paper Lineage](#9-paper-lineage)

---

## 1. Protocol Architecture

### RoKoko: Static 7-Round Config Chain

RoKoko uses a **fixed sequence of 7 rounds** (`P` through `P_LAST`), each with a
static `Config` specifying witness dimensions and projection type. The config
chain is defined in `src/protocol/params.rs`:

```
Round 0 (P):   witness 1024×2,  rank 4,  projection Type0 (coarse)
Round 1 (P_1): witness  544×2,  rank 4,  projection Type1 (fine)
Round 2 (P_2): witness  296×2,  rank 2,  projection Type1
Round 3 (P_3): witness  168×2,  rank 2,  projection Type1
Round 4 (P_4): witness  104×2,  rank 2,  projection Skip
Round 5 (P_5): witness   72×2,  rank 2,  projection Skip
Round 6 (P_LAST): Simple round — send witness in the clear
```

The protocol driver is `src/protocol/parties/executor.rs`. Each round:
prover commits → verifier challenges → prover folds/projects → sumcheck.

### Hachi: Dynamic Schedule with Layout

Hachi uses `CommitmentConfig` + `LevelParams` to compute the recursion
schedule dynamically from the number of variables and configuration knobs.
The schedule is generated at runtime, not hardcoded.

The top-level API is `HachiCommitmentScheme` in
`src/protocol/commitment_scheme.rs` with `commit` / `prove` / `verify`.
Internally, the protocol runs a two-stage sumcheck:

- **Stage 1:** Range-check / norm verification
- **Stage 2:** Fused commitment + evaluation check

### Architectural Contrast

| Aspect | RoKoko | Hachi |
|--------|--------|-------|
| Schedule | Static config chain, 7 hardcoded rounds | Dynamic layout from `CommitmentConfig` |
| Round types | `Sumcheck` rounds + 1 `Simple` tail | Two-stage sumcheck per level |
| Witness packing | Binary-prefix packing (`paste_by_prefix`) | Dense polynomial representation |
| Proof structure | Flat: one proof per round | `HachiProof` → `HachiLevelProof` → `HachiProofTail` |

---

## 2. Commitment Schemes

### RoKoko: Recursive Ajtai Commitments

RoKoko uses a multi-layer Ajtai commitment where the CRS has **tensor
structure**. Each row of the commitment matrix is a structured row
`(a_0, a_1, ..., a_{2^μ - 1})` where the full matrix row is reconstructed via
tensor products.

**Basic commitment:** `Y = F · W mod q` where `F` is the CRS matrix of rank
`n` (module rank, typically 2–4).

**Recursive commitment** (`src/protocol/commitment.rs::recursive_commit`):
The basic commitment output `Y` is itself gadget-decomposed (`G⁻¹`) and
committed again, repeating for `depth` levels. Only the outermost commitment
(of rank independent of witness size) is inspected by the verifier.

```
Level 0:  com_0 = F_0 · W           (rank n_0, width m_w)
Level 1:  com_1 = F_1 · G⁻¹(com_0)  (rank n_1, width n_0 · ℓ)
Level 2:  com_2 = F_2 · G⁻¹(com_1)  (rank n_2, width n_1 · ℓ)
...
```

This compresses the commitment from `n_0` ring elements (linear in log of
witness size) down to a constant-rank outermost commitment.

### Hachi: Flat Commitments

Hachi uses a single-layer commitment scheme without recursive compression.
The commitment config (`src/protocol/commitment/`) specifies layouts and
schedules, but commitments are not recursively nested.

---

## 3. Folding

### RoKoko: Committed Folding with Superconstant ρ

The key innovation over prior work (KLNO25) is using **committed cross-terms**
to achieve a shrinking factor `ρ = λ` (superconstant) rather than `ρ = O(1)`.

**Folding** (`src/protocol/fold.rs`): Given witness `W = [w_1 | w_2 | ... | w_ρ]`
with `ρ` columns, and verifier challenges `c = (c_1, ..., c_ρ)`:

```
z = Σ cᵢ · wᵢ
```

The constraint `F · W = Y` becomes `F · z = Y · c`. But the sumcheck for
`F · z = Y · c` involves cross-terms between columns. In prior work (KLNO25,
ρ = O(1)), these cross-terms are sent in the clear — linear in `ρ`. RoKoko
instead **commits** to the cross-terms and proves consistency via sumcheck,
allowing `ρ = λ` without blowing up communication.

**Effect on recursion depth:** With `ρ = λ`, each round reduces the witness
width from `ρ` to 1 (a factor-`λ` compression), so the total number of rounds
is `O(log m / log λ)` instead of `O(log m)`.

### Hachi: Two-Stage Sumcheck Folding

Hachi's folding is integrated into its two-stage sumcheck:

- `src/protocol/sumcheck/hachi_stage1.rs` — range check
- `src/protocol/sumcheck/hachi_stage2.rs` — fused commitment/evaluation

The `two_round_prefix` module handles the initial rounds before the main
sumcheck loop kicks in.

---

## 4. Random Projections and the JL Lineage

This is the most nuanced area of comparison and the one with the richest
history across the lattice-based SNARK literature.

### The Problem

After folding, the extractor recovers a witness `W` satisfying the commitment
relation `F · W = Y`, but only guarantees `||W · diag(s)|| ≤ β'` for some
"slack" vector `s` with `||s|| ≤ ϱ`. The slack `ϱ` accumulates multiplicatively
across rounds. Random projections **eliminate** this slack by providing an
independent norm bound on `W`.

### Evolution of Approaches

#### Generation 1: Full Unstructured JL (LaBRADOR, CRYPTO 2023)

**Paper:** `LaBRADOR.pdf`

The projection matrix `J ∈ {-1,0,1}^{256 × m·φ}` is a full i.i.d. ternary
matrix (each entry 0 w.p. 1/2, ±1 w.p. 1/4 each). The prover sends
`p = J · cf(W) mod q ∈ Z_q^{256}` in the clear.

**JL guarantee (Lemma 4.1):**
```
E[||Π·w||] = √128 · ||w||
Pr[||Π·w|| < √30 · ||w||] ≤ 2⁻¹²⁸     (lower tail)
Pr[||Π·w|| > √337 · ||w||] ≤ 2⁻¹²⁸    (upper tail)
```

| Metric | Value |
|--------|-------|
| Projection dimensions | `{-1,0,1}^{256 × m·φ}` → 256 integers |
| Verifier cost | **O(m·φ)** — must generate full `J` |
| Proof overhead | ~1 KB constant (256 encoded integers + aggregated constraints) |
| Norm slack | `√(128/30) ≈ 2.07×` |
| Completeness | ~1/2 per attempt (retry with fresh `J`) |

**Fatal drawback:** Linear verifier time. The verifier must process `O(m·φ)`
entries of `J`, destroying succinctness.

#### Generation 2: Algebraic Norm-Check (RoK, Paper, SISsors, ASIACRYPT 2024)

**Paper:** `RoK_Paper_SISsors.pdf`

RPS takes a fundamentally different approach: **no JL at all**. The protocol
`Π_norm` proves `Trace(⟨w, w̄⟩) ≤ β²` algebraically via an inner-product
protocol `Π_ip`:

1. Encode `w ∈ R^m` as coefficients of `g_w(X) = Σ wⱼ Xʲ`
2. Form Laurent polynomial `L(X) = g_w(X) · ḡ_w(X⁻¹)`, constant coeff = `⟨w, w̄⟩`
3. Commit to the symmetric coefficients `(v₀, ..., v_{m-1})` via `h(X)`
4. Verify `L(ξ) = h(ξ) + h̄(ξ⁻¹) - v₀` at random `ξ` (Schwartz-Zippel)
5. Check `Trace(v₀) ≤ β²`

The full composition chain is:
```
(Π_norm → Π_batch → Π_{b-decomp} → Π_split → Π_fold)^μ → Π_finish
```

| Metric | Value |
|--------|-------|
| JL matrix | None |
| Prover cost | O(m log m) — polynomial multiplication bottleneck |
| Norm slack | Exact (no JL slack) |
| Proof size | 5–7 MB for 2³⁰ witnesses |

**Key insight:** Exact norm bounds without JL, but quasi-linear prover due to
polynomial multiplication.

#### Generation 3: Tensored JL (RoK and Roll, ASIACRYPT 2025)

**Paper:** `RoK_and_Roll.pdf`

The breakthrough. The projection matrix has **Kronecker/block-diagonal
structure**:

```
Ĵ = I_{m/m_rp} ⊗ J,    where J ∈ {-1,0,1}^{n_rp × m_rp}
```

The verifier only samples the small block `J` (size `n_rp × m_rp ≈ 256 × 256`),
not the full `256 × m·φ` matrix. Each block of `m_rp` witness elements gets
its own independent JL projection.

**Two-phase design:**

1. **Structured loop** (large `m`): tensored `I ⊗ J`, committed images,
   iterate fold → project → fold until `m = O(λ)`
2. **Unstructured loop** (small `m = O(λ)`): full JL matrix with
   **lift-and-batch** through the ring tower `R = R_ℓ ⊃ ... ⊃ R_0 = Z` to
   reduce communication from `Ω(λ²)` to `O(λ)` bits

| Metric | Tensored (structured loop) | Full (unstructured loop) |
|--------|---------------------------|--------------------------|
| Verifier | `O(n_rp · m_rp) = O(λ²)` — succinct | Used only when `m = O(λ)` |
| Structural preservation | Preserves vSIS tensor structure | Breaks tensor structure |
| JL concentration | Union bound over `m/m_rp` blocks | Single-block (tighter) |
| Communication | Committed image (not sent in clear) | Lift-and-batch through tower |

**Proof sizes** (Table 2 from paper):
```
λ=128:  4.07 MB (vs. 27.57 MB for RPS)  — 6.8× improvement
λ=256:  7.79 MB (vs. 112.13 MB for RPS) — 14.4× improvement
```

#### Generation 4: Sumcheck Norm-Check (SALSAA, eprint 2025/2124)

**Paper:** `SALSAA.pdf`

SALSAA replaces the quasi-linear polynomial-multiplication norm-check with a
**sumcheck-based** norm-check: express `||w||²_σ = Trace(⟨w, w̄⟩)` as a
sumcheck claim over the LDE, which the prover evaluates in **linear time**
via the dynamic-programming optimization.

Still uses tensored JL (`Π⊗RP` from RnR) for approximate norm + extraction
soundness. The sumcheck provides the exact norm verification.

| Metric | Value |
|--------|-------|
| Norm-check prover | **O(m)** — linear (down from O(m log m) in RPS) |
| Proof size | 979 KB for 2²⁸ elements (2–3× smaller than RnR) |
| Verify | 41 ms |

#### Generation 5: Full Synthesis (RoKoko, eprint 2026/575)

**Paper:** `eprint_2026_575.pdf`; **Implementation:** `../rokoko/`

RoKoko combines everything:
- **Tensored JL** (from RnR) for approximate norm + extraction
- **Sumcheck norm-check** (from SALSAA) for exact norm verification
- **Committed folding** with `ρ = λ` for superconstant shrinking
- **Recursive commitments** for constant-size outer commitment

Two projection types in the implementation:

**Type0 — Coarse** (`src/protocol/project.rs`): Applies `(I ⊗ J)` at the
ring-element level. Used when `m_w > m_rp` (witness still large). The
projected image is gadget-decomposed and committed.

**Type1 — Fine** (`src/protocol/project_2.rs`): Applies `J` at the
**coefficient level** (`cf(W)`), then uses trace + batching to lift back to
ring constraints. Used when `m_w` drops below `m_rp` but `φ · m_w` is still
large enough. More expensive per-element but works on smaller witnesses.

### Tensored vs Full JL: Complete Tradeoff Analysis

| Aspect | Full JL (LaBRADOR) | Tensored `I ⊗ J` (RoKoko) | Winner |
|--------|---------------------|---------------------------|--------|
| Verifier time | `O(m·φ)` — linear | `O(λ²)` — succinct | **Tensored** |
| Recursive composability | No (breaks tensor structure) | Yes | **Tensored** |
| JL concentration | Single block (tightest) | Union bound over `m/m_rp` blocks | Full (marginal) |
| Extraction simplicity | Direct inversion | Block-by-block + careful analysis | Full (marginal) |
| Proof overhead per use | ~1 KB (send image in clear) | Commit + decompose + sumcheck | Full |
| Proof size at scale | ~60 KB but **O(m) verify** | ~214 KB with **polylog verify** | **Tensored** |
| Small-witness regime | Better (compact, no commitment needed) | Overkill | **Full** |
| Completeness probability | ~1/2 (retry needed) | Same per-block | Tie |

**Key insight:** The tensored variant's only real drawback is that the
projection image `V = (I ⊗ J) · W` has `m · n_rp / m_rp` ring elements and
**cannot be sent in the clear** (not succinct). It must be committed, gadget-
decomposed, and proved correct — adding prover work and proof elements.

For large witnesses (`m >> λ`), the tensored variant is unambiguously superior
because succinctness matters. For the final rounds when `m = O(λ)`, both RoK
and Roll and RoKoko switch to the full/unstructured variant.

### Hachi's Approach

Hachi uses **ring switching** (`src/protocol/ring_switch.rs`) and
**opening-point reduction** (`src/protocol/opening_point.rs`) rather than
JL projections. The two-stage sumcheck handles norm verification inline
without a separate projection step.

---

## 5. Sumcheck Integration

### RoKoko: Sumcheck over F_{q²}

RoKoko's sumcheck operates over the **quadratic extension field** `F_{q²}`,
not directly over the ring `R_q` or base field `F_q`.

The pipeline:
1. All protocol constraints (folding, projection, commitment, norm) are
   expressed as sumcheck relations over `R_q`
2. A `RingToFieldCombiner` (`src/protocol/sumcheck_utils/ring_to_field_combiner.rs`)
   transforms ring claims into field claims over `F_{q²}`
3. The actual sumcheck round polynomials and challenges live in `F_{q²}`

**Constraint types** (from `src/protocol/sumchecks/builder.rs`):
- **Type0:** Folding correctness — `F · z = Y · c`
- **Type1:** Commitment consistency — `F · W = Y`
- **Type2:** Projection correctness — `(I ⊗ J) · W = G · Y_{proj}`
- **Type3:** Cross-term constraints for committed folding
- **Type4:** Recursive commitment constraints
- **Type5:** Norm bound — `Trace(⟨w, w̄⟩) ≤ β²`

The sumcheck runner (`src/protocol/sumchecks/runner.rs`) iterates:
extract univariate → hash to get challenge → partially evaluate all gadgets.

### Hachi: Two-Stage Sumcheck with Ring Switch

Hachi's sumcheck (`src/protocol/sumcheck/`) has a different structure:

- **Stage 1** (`hachi_stage1`): Range-check verification
- **Stage 2** (`hachi_stage2`): Fused commitment + evaluation check
- **Ring switch** (`ring_switch.rs`): Converts between rings for the sumcheck

The `two_round_prefix` module handles initial rounds before the main loop.

---

## 6. Ring Arithmetic and NTT

### RoKoko: Incomplete NTT over Degree-128 Cyclotomic

**Ring:** `R_q = Z_q[X]/(X^{128} + 1)` with `q = 1125899906839937 ≈ 2^{50}`.

**Four representations** (`src/common/ring_arithmetic.rs`):
1. `Coefficients` — standard polynomial form `(a₀, a₁, ..., a_{127})`
2. `EvenOddCoefficients` — interleaved even/odd coefficients
3. `IncompleteNTT` — 64 quadratic-extension-field elements
4. `HomogenizedFieldExtensions` — fully split for pointwise ops

The "incomplete" NTT is the key: since `X^{128} + 1` factors over `F_q` into
64 irreducible quadratics (because `q ≡ 1 mod 128` is not satisfied — the
ring "almost splits"), the NTT stops at degree-2 factors, giving 64 elements
of `F_{q²}` rather than 128 elements of `F_q`.

**Conjugation** (`conjugate_in_place`): Complex conjugation `σ: ζ ↦ ζ⁻¹` is
precomputed as a `ConjugationTransform` — a permutation + pointwise
conjugation in the incomplete-NTT domain.

**Backend:** The `incomplete-rexl` crate provides pure-Rust NTT with
AVX-512-IFMA acceleration (`fused_incomplete_ntt_mult`).

### Hachi: Configurable Rings with Rayon Parallelism

Hachi uses configurable fields and rings defined in `src/algebra/` with
NTT support via `src/algebra/domains/`. Parallelism comes from Rayon
(feature-gated under `parallel`), not SIMD intrinsics.

---

## 7. Norm Bounds and Slack

### Sources of Slack

In both protocols, folding introduces slack. When the verifier sends challenge
`c` and the prover folds `z = Σ cᵢ · wᵢ`, the extractor recovers
`(W, s)` with `||W · diag(s)|| ≤ β'` where `||s|| ≤ ϱ ≈ 2γ_C` (the
operator norm of the challenge distribution).

Without projection, `ϱ` would accumulate **multiplicatively** across all
rounds, destroying the extracted norm bound.

### How RoKoko Eliminates Slack

The random projection provides an **absolute** (un-slacked) norm bound:

```
α_rp · ||W|| ≤ ||(I ⊗ J) · W|| ≤ β_rp · ||W||
```

with `α_rp = √30`, `β_rp = √337` for `κ_rp = 2⁻¹²⁸` and `n_rp = 256`.

Since the committed image `Y = G⁻¹(V)` has a strict norm bound (`ϱ = 1` by
construction), extraction recovers:

```
||W|| ≤ cmp(β'_Y) / α_rp
```

No slack factor propagates. Each round's projection "resets" the extraction
norm to an absolute bound.

### The Remark 5 Optimization

For PCS applications where the polynomial evaluation claim can tolerate slack
(the committed polynomial need not have strictly small coefficients), RoKoko
can **skip the first Π^{proj-c}** (Section 8.3, Remark 5). This improves
prover time from `O(m·λ²)` to `O(m·λ)` but means the extracted witness from
the first round carries a slack factor `ϱ`.

### Hachi's Approach

Hachi handles norm control through its two-stage sumcheck design. Stage 1
performs range checks, and the ring-switch protocol
(`src/protocol/ring_switch.rs`) manages the algebraic transitions needed
to maintain norm guarantees.

---

## 8. Concrete Parameters

### RoKoko (from Table 1, eprint 2026/575)

Benchmarked on 2×48-core Xeon 8468 @ 2.1 GHz, AVX-512.

| Witness (#Z_q) | Commit | Prove | Verify | Proof Size |
|-----------------|--------|-------|--------|------------|
| 2²⁶ | 106 ms | 3.61 s | 7.3 ms | 176 KB |
| 2²⁸ | 451 ms | 13.14 s | 8.6 ms | 214 KB |
| 2³⁰ | 1.67 s | 50.86 s | 9.5 ms | 252 KB |
| 2³² | 6.42 s | 200.04 s | 10.5 ms | 291 KB |

### SALSAA (from Table 1, eprint 2025/2124)

Same hardware.

| Witness (#Z_q) | Commit | Prove | Verify | Proof Size |
|-----------------|--------|-------|--------|------------|
| 2²⁶ | 80 ms | 3.05 s | 34 ms | 808 KB |
| 2²⁸ | 348 ms | 10.61 s | 41 ms | 979 KB |
| 2³⁰ | 1.23 s | 39.7 s | 54 ms | 1123 KB |

### Key Comparisons

- **RoKoko vs SALSAA:** RoKoko has ~4× smaller proofs and ~5× faster
  verification, at the cost of ~25% slower proving (for 2²⁸).
- **RoKoko vs Greyhound:** Greyhound (NS24) achieves 53 KB proofs for 2²⁸
  elements but with 8.21 s prove and **1.15 s verify**. RoKoko trades 4×
  larger proof for 134× faster verification.
- **RoKoko vs LaBRADOR:** LaBRADOR achieves ~60 KB proofs but has
  **linear verification time** due to the full JL matrix.

---

## 9. Paper Lineage

The following diagram shows the evolution of techniques. Papers are in
`~/Documents/Papers/`.

```
                    LNS21 (PKC 2021)
                    Early random-masking for lattice shortness
                        │
                        ▼
                    LNP22 (CRYPTO 2022)
                    ℓ₂ shortness via polynomial products
                        │
                        ▼
    ┌───────────────────┴───────────────────┐
    │                                       │
    ▼                                       ▼
LaBRADOR (CRYPTO 2023)              BCS21 (eprint 2021/333)
Full unstructured JL                 Sumcheck for lattices
~60KB proofs, O(m) verify               │
    │                                    │
    ▼                                    │
RoK, Paper, SISsors (ASIACRYPT 2024)    │
Algebraic norm-check (no JL)             │
Π_norm, Π_ip, twisted trace             │
5–7 MB proofs                            │
    │                                    │
    ▼                                    │
RoK and Roll (ASIACRYPT 2025)            │
Tensored JL: I ⊗ J                      │
Lift-and-batch for unstructured rounds   │
4 MB proofs, O(λ) comm.                 │
    │                                    │
    ├────────────────────────────────────┘
    │
    ▼
SALSAA (eprint 2025/2124)
Sumcheck-based norm-check (linear time)
R1CS support, first lattice folding scheme
979 KB proofs
    │
    ▼
RoKoko (eprint 2026/575)
Committed folding (ρ = λ), recursive commitments
Type0 + Type1 projections, sumcheck-driven
214 KB proofs, 8.6 ms verify
```

### File Index

| File | Citation | Key contribution |
|------|----------|-----------------|
| `LNS21.pdf` | Lyubashevsky-Nguyen-Seiler, PKC 2021 | One-time commitments, random masking |
| `LNP22.pdf` | Lyubashevsky-Nguyen-Plançon, CRYPTO 2022 | ℓ₂ norm via polynomial-product inner products |
| `BCS21_Sumcheck_Arguments.pdf` | Bootle-Chiesa-Sotiraki, 2021 | Sumcheck for lattice-based proofs |
| `LaBRADOR.pdf` | Beullens-Seiler, CRYPTO 2023 | Full JL projection, ~60KB proofs |
| `Greyhound.pdf` | Nguyen-Seiler, CRYPTO 2024 | Fast lattice PCS, 53KB proofs |
| `RoK_Paper_SISsors.pdf` | KLNO, ASIACRYPT 2024 | Algebraic RoK toolkit, no JL |
| `RoK_and_Roll.pdf` | KLNO, ASIACRYPT 2025 | Tensored JL, lift-and-batch |
| `SALSAA.pdf` | KLOT, eprint 2025/2124 | Sumcheck norm-check, linear prover |
| `eprint_2026_575.pdf` | KLNOT, eprint 2026/575 | RoKoko — full synthesis |
| `Hachi.pdf` | — | Hachi PCS design |

### RoKoko Implementation Structure (`../rokoko/`)

```
src/
├── common/
│   ├── config.rs               — Global constants (degree=128, q≈2⁵⁰)
│   ├── ring_arithmetic.rs      — RingElement, 4 representations, NTT, conjugation
│   ├── projection_matrix.rs    — Bitmask-encoded {-1,0,1} matrices for SIMD
│   └── ...
├── protocol/
│   ├── params.rs               — Static 7-round config chain (P → P_LAST)
│   ├── config.rs               — Config enum (Sumcheck vs Simple)
│   ├── config_generator.rs     — Binary-prefix packing
│   ├── crs.rs                  — Tensor-structured CRS generation
│   ├── commitment.rs           — Basic + recursive Ajtai commitments
│   ├── fold.rs                 — z = Σ cᵢ · wᵢ
│   ├── open.rs                 — Evaluation/opening claims
│   ├── project.rs              — Type0: coarse (I ⊗ J) projection
│   ├── project_2.rs            — Type1: fine coefficient-level projection
│   ├── parties/
│   │   ├── executor.rs         — Top-level protocol driver
│   │   ├── prover.rs           — Recursive prover rounds
│   │   └── verifier.rs         — Recursive verifier rounds
│   ├── sumcheck_utils/
│   │   ├── linear_sumcheck.rs  — MLE evaluation
│   │   ├── selector_eq.rs      — Equality polynomial
│   │   ├── product_sumcheck.rs — Product gadget
│   │   ├── combiner.rs         — Random linear combination
│   │   └── ring_to_field_combiner.rs — R_q → F_{q²} transformation
│   └── sumchecks/
│       ├── builder.rs          — Type0–Type5 constraint construction
│       └── runner.rs           — Sumcheck main loop
└── incomplete-rexl/            — Pure-Rust NTT backend (AVX-512-IFMA)
```

---

## Fiat-Shamir

| | RoKoko | Hachi |
|--|--------|-------|
| Hash | BLAKE3 | BLAKE2b |
| Transcript | Custom XOF-based (`HashWrapper`) | `Blake2bTranscript` implementing `Transcript` trait |
| Challenge sampling | `sample_field_element_into`, `sample_ring_element_into` | Via `Transcript` trait methods |
