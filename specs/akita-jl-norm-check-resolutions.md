# Akita-JL norm-check: gap resolutions and corrected derivations (v5 companion)

*Rigorous follow-up to [`akita-jl-projection-protocol.md`](akita-jl-projection-protocol.md) (the protocol, v5) and [`grand-danois-jl-vs-exact-l2-norm-check.md`](grand-danois-jl-vs-exact-l2-norm-check.md) (the branch-decision doc). This file resolves the open items those docs flag, re-derives the load-bearing numbers from first principles, and corrects the specific errors in the prior design pass. Each section ends with a one-line verdict and the exact change it forces in the parent docs.*

| Field | Value |
|---|---|
| Author(s) | Quang Dao, Claude (Fable 5 + Opus 4.8) draft |
| Created | 2026-06-13 |
| Status | derivation + resolution note; gates the protocol spec's §5/§8 items |
| Grounds | akita main @ ca4c51fd (pre-#154); writeup `lattice-jolt/sections/akita/{2_preliminaries,3_basic_akita,c_operator_norm_certification}.tex`; papers GD/LaBRADOR/RoKoko/RnR/SALSAA/Hachi |

## 0. Corrections ledger (what the prior pass got wrong)

| # | Prior claim (where) | Status | Corrected statement | §|
|---|---|---|---|---|
| C1 | "Anchored A-role may remove `Γ̄`, with a clean `2T̄_s` path and a fallback" (protocol §2 v4 caveat) | **Retired framing** | The A-role shape remains `η_A = 2·Γ̄·β̄₂`. JL does not remove `Γ̄`. It replaces the loose deterministic input to `β̄₂` with a realized bound from the accepted JL image of the committed `w_next`, whose flat table contains `ẑ`. | §1 |
| C2 | Root strict-`2^{-128}` needs `n_J ≈ 410–420` (protocol §5e, decision doc) | **Wrong model** | That assumed a `Q=2^64` grinding budget akita does **not** declare. Under akita's shipped convention (per-level `2^{-128}`, per-draw 128-bit entropy floor, no explicit `Q`-amplification of statistical terms), the root needs **`n_J ≈ 285–295`**. `n_J ≈ 410` is only required if akita globally adopts explicit `2^64`-grind accounting — which would also force re-sizing the sumcheck/fold terms, not just JL. | §2 |
| C3 | "Basis unlock shrinks the tail; SIS optimum `lb*≈11–16`, JL unlocks a 3–4× tail win" (decision doc headline; protocol §4 lever 2) | **Wrong for bytes** | Measured: lifting `PROOF_OPTIMIZED_LOG_BASIS_MAX` 6→16 changes total proof size by **≤1% at nv=20 and 0% at nv≥28**. The `δ·lb ≈ field_bits` packing identity magnitude-locks every cleartext segment; the only residual `lb`-dependence is `n_a(lb)`, which *grows*. `lb*≈11–16` is an element/rank optimum that **does not translate to bytes**. The basis is **not a proof-size lever**; JL "unlocking" it buys ~nothing in bytes. | §3 |
| C4 | "Treat `J` as a fourth seed-derived setup role; offload its MLE via PR #138 setup prefix" (protocol §6.4b) | **Invalid** | `J` is **Fiat–Shamir-derived** (sampled from the transcript after `u_ℓ`), so it changes per proof and **cannot** be committed as a setup prefix the way the transparent setup matrix can. The verifier must evaluate `J̃₀` live, `O(n_J·m₀)` per consistency point. The offload as written is impossible; only a *closed-form generator* for `J₀` could help, and a random ±1 matrix has none. | §4 |
| C5 | "terminal `r̂` elision is the largest byte win found this pass" / "≈82 KB terminal floor" (my first pass) | **Already spec'd; reframed** | The `r̂`-drop is **PR #141** (approved, 5.25–6.15%), one of four levers in Quang's [`tail-wire-encoding.md`](tail-wire-encoding.md). The ≈82 KB is the *current fixed-width* terminal, not a floor — entropy coding (`z`: `bound→σ`) is actively shrinking it. JL and entropy coding monetize the **same `30–200×` bound-vs-realized gap** at committed vs cleartext levels (complementary). | §8 |

The corrections all push the same direction: **JL's value is protocol *simplicity*, no-wrap freedom on the image, and realized-norm A-role input, not a proof-size or basis win.** The byte case for Branch B is weaker than the decision doc claimed. The remaining case rests on deleting the four-square/carry/no-wrap machinery, avoiding the no-large-level fallback, and replacing the loose `β̄₂` input in `η_A = 2·Γ̄·β̄₂`. It does *not* rest on removing `Γ̄` or on a basis lever.

---

## 1. A-role accounting: JL supplies realized `β̄₂`; it does not remove `Γ̄`

### 1.1 What akita's A-role collision is today

The shipped A-role price is a collision in the cross-multiplied domain (`3_basic_akita.tex:1959–2174`; code `crates/akita-types/src/sis/norm_bound.rs:124–140`):

```
collision_linf = 8·ω·β_inf·ν
η_A = 2·Γ̄·β̄₂
```

The reason it is cross-multiplied is structural (`prop:committed-fold-price`, `3_basic_akita.tex:1968–2020`):

1. The extracted block is a fold-response quotient `s_i^{ext} = (z^{(ℓ,i)} − z^{(0)})/c̄_i`. Division by the ring unit `c̄_i` is not norm-preserving, so a bound on `‖c̄_i s_i^{ext}‖` does not transfer to `‖s_i^{ext}‖`.
2. The verifier's current range and one-hot checks bind the honest committed table, not the extracted quotient.
3. The collision is therefore formed by re-multiplying each opening's certified product by the other opening's challenge: `z_A = c̄'(c̄ s) − c̄(c̄' s')`, with `A z_A = 0` and `‖z_A‖₂ ≤ 2·Γ̄·β̄₂`.

The cross-mass `Γ̄` stays in the price. The term JL can improve is `β̄₂`.

### 1.2 The corrected JL path

The reveal path projects the next recursive witness after its commitment:

```
u′ = U·w_next
  → J sampled from the transcript
  → p = J·w_next
  → verifier checks ‖p‖₂ ≤ T_p over the integers
  → stage 2 checks projection consistency
```

The flat table is:

```
w_next = (ê, t̂, ẑ, r̂)
```

`w_next` is fixed before `J`, so modular JL can bound its realized Euclidean norm from the accepted image threshold. Since `ẑ` is a segment of `w_next`, `‖ẑ‖₂ ≤ ‖w_next‖₂`. This gives a realized bound for the response component that feeds `β̄₂`. The A-role price then remains:

```
η_A = 2·Γ̄·β̄₂
```

The change is the source of `β̄₂`: use the JL-derived realized bound instead of the current `√d·β_inf` path.

### 1.3 What this retires

This retires the earlier R-A/R-B split for the current path.
Do not claim that JL yields a clean `η_A = 2T̄_s`.
Do not frame the work as an extraction re-architecture that removes `Γ̄`.
Do not describe the operator-norm check as "belt and braces."

Slot 2 may still be useful for committed-image levels, but its accounting must be stated through the same `2·Γ̄·β̄₂` shape unless a separate proof actually changes the weak-binding lemma. That separate proof is not the gating path for the reveal prototype.

### 1.4 Verdict and parent-doc change

- **Keep the weak-binding shape:** `η_A = 2·Γ̄·β̄₂`.
- **JL changes `β̄₂`:** the accepted image of `w_next` gives a realized witness bound, and `ẑ` is contained in that witness.
- **Change protocol §2:** remove the R-A/R-B fork and the clean `2T̄_s` claim. The D4 gate is the slack map `T_p -> T_w -> β̄₂`, plus coordinate injectivity and row-count sizing.

---

## 2. Fiat–Shamir / grinding and the root `n_J` (correcting C2)

### 2.1 Akita's actual model (verified)

- **Target: `λ = 128`, per level** (`c_operator_norm_certification.tex:404–425`; `ε_level` at `3_basic_akita.tex:1716`). Rejection-sampling/ZK statistical slop is budgeted *separately* at `2^{-100}` (`2_preliminaries.tex:808`); the JL term is a **knowledge-error** term, so it sits against `2^{-128}`.
- **`Q` is symbolic.** The writeup uses the textbook ROM FS bound `(Q+1)·κ_CWSS` (`3_basic_akita.tex:1376`, Attema–Fehr–Klooß) with `Q` = RO-query count, and pins one contract: **per-draw min-entropy `≥ λ + log₂ Q` bits**, shipped as `MIN_FOLD_CHALLENGE_ENTROPY_BITS = 128` (PR #173, `crates/akita-challenges/src/config.rs:22`). The `2^{64}` budget I used previously appears *nowhere* in akita; it was the JL spec's own illustrative number.
- **Challenge spaces are sized to ≈ `2^{128}` exactly** (`|E| = q^4 = 2^{128}`; fold `|C| = 2^{129.6}` at d=128; `2^{131.5}` at d=64). The existing statistical terms (`ΛB/|C|`, sumcheck SZ `~/|E|`) evaluate to `2^{-116}…2^{-125}` — i.e. akita runs **right at the 128-bit edge with single-digit bits of slack and no explicit `Q`-amplification reserve.**

The #173 per-draw floor is the substantive subtlety: tensor folds reuse each factor across blocks, so a `64+64` split that passes a *product* rule leaves each factor brute-forceable; the floor enforces 128 bits **per draw**. This is the "grinding contract" akita actually ships — entropy floored per draw, `Q`-amplification left symbolic.

### 2.2 Sizing `n_J` under akita's convention

κ_JL is a **statistical net failure** (the modular-JL lower-tail miss), unlike the algebraic `1/|E|` terms. Its structure (protocol §5):

```
κ_JL(total) = (#blocks) · 2^{-c(n_J)},   #blocks = N_coeff / m₀  (= chunks × d slices, ring-granular)
```

where `N_coeff` = projected coefficients, `m₀` = block dimension (ring-chunk width), and `c(256) = 128` from LaBRADOR Lemma 4.2, extrapolating `c(n_J) ≈ n_J/2` (≈ ½ bit/row; **re-derive the exact constant — LaBRADOR's bound is a specific 256-row constant, not proven linear**).

**Matching akita's shipped convention** (drive κ_JL to the same `2^{-128}` the other per-level terms hit, no separate large-`Q` amplification):

```
(#blocks)·2^{-c(n_J)} ≤ 2^{-128}  ⟹  c(n_J) ≥ 128 + log₂(#blocks)  ⟹  n_J ≈ 256 + 2·log₂(#blocks)
```

Root numbers (nv=28-class projected object, `N_coeff ≈ 2^{25}`; whole-witness worst case `N_coeff ≈ 2^{30}`):

| `N_coeff` | `m₀` | #blocks | `c(n_J)` needed | **`n_J`** |
|---|---|---|---|---|
| `2^{25}` | `2^{12}` | `2^{13}` | 141 | ≈ **282** |
| `2^{25}` | `2^{15}` | `2^{10}` | 138 | ≈ **276** |
| `2^{30}` | `2^{12}` | `2^{18}` | 146 | ≈ **292** |

So **`n_J ≈ 285–295` at the root** (≈1.13× the LaBRADOR 256), not 410. The `n_J ≈ 410` figure requires `c(n_J) ≥ 128 + 64 + log₂(#blocks)` — i.e. an explicit `Q = 2^{64}` grind budget. **If akita ever adopts that, it must re-size the sumcheck and fold terms too** (their `(Q+1)/|E|` would be `2^{-52}`, not `2^{-128}`), so it is a *global* model change, not a JL-specific tax. Recorded as a conditional, not a requirement.

### 2.3 The principled reason to keep the range check at the root (new, clean argument)

There is a **qualitative** difference between the two root options, independent of the `n_J` count:

- **Stage-1 ∞-range check**: soundness is **algebraic** — digit-set membership is the vanishing of `Q_sq(t) = Π(t − k(k+1))` checked by sumcheck, error `~ deg/|E|`, the *same character* as every other akita sumcheck term, and **wrap-free** (membership is exact mod q; needs only `b/2 < q`). It introduces **no new failure mode** and no net/union bound.
- **JL** introduces a **new statistical failure mode** (the modular-JL net miss `κ_JL`), which is **grindable** (vary the transcript → fresh `Ĵ`; a fixed bad witness passes if any sampled `Ĵ` is bad) and carries the `(#blocks)` union. This is most expensive *exactly at the root* (largest `N_coeff`).

> **So the three-paradigm schedule is not just a cost heuristic — it is sound-design hygiene:** put the *algebraic* certificate (range check) where the *statistical* one (JL) is most expensive and most exposed to grinding (the root), and use JL only where its union bound is small and its O(N) verifier is affordable (mid/tail). The "exact-ℓ2 on z fails the no-wrap gate at nv=32 roots" point (decision doc) cuts the same way: the *range* part of stage-1 is wrap-free and stays; only the *exact-ℓ2 addition* needs the gate, and that addition is exactly what we'd drop at the root.

### 2.4 Verdict and parent-doc change

- **Change protocol §5e and decision-doc:** root `n_J ≈ 285–295` under akita's shipped convention; `≈410` only under an explicit `2^{64}` global grind model. Either way `>256`, so the root favors keeping the algebraic range check — now justified by the algebraic-vs-statistical hygiene argument, not only verifier cost.
- **Mid/tail levels:** `#blocks` is small (a late level has `N_coeff ≈ 2^{17}`, `m₀ = 2^{12}` → `2^5` blocks → `n_J ≈ 266`), so `n_J = 256–270` suffices; the union bound is a non-issue away from the root.

---

## 3. The basis is not a proof-size lever (correcting C3 — the biggest correction)

> **Note (2026-06-16): an earlier banner here tried to discredit this section by claiming the §3.2 DP "priced a `2^lb` per-fold range tax that JL removes." That banner was wrong and has been deleted.** The stage-1 inf-norm range check is a GKR product tree (`s = w(w+1)` degree-halving + `stage1_tree_stage_arities`), so it grows only **linearly** in `lb`, not as `2^lb` (`crates/akita-types/src/proof/stage1.rs:103-151`). Replacing a linear range check with a JL image of comparable per-level cost does not materially shift the fold-vs-cleartext crossover, so this section's conclusion stands: the basis is not a proof-size lever, and JL does not turn it into one. See `akita-jl-tail-projection-prototype.md` → "Motivation: why JL projection (and what it does not do)".

### 3.1 The packing identity (analytic)

Terminal cleartext bytes `= (Σ ring-element segments)·d·lb/8` (verified: `direct_witness_bytes` → `packed_digits_bytes`, `crates/akita-types/src/layout/proof_size.rs:27–41`; `bits = current_log_basis`). Per segment, with `δ_open = δ_full = ⌈field_bits/lb⌉`:

| segment | ring count | **bytes** ∝ | `lb`-dependence |
|---|---|---|---|
| ê | `nb·δ_open` | `nb·d·(δ_open·lb)` ≈ `nb·d·field_bits` | **invariant** |
| t̂ | `nb·n_a·δ_open` | `nb·n_a·d·(δ_open·lb)` ≈ `nb·n_a·d·field_bits` | **grows via `n_a(lb)` only** |
| ẑ | `inner_width·δ_fold` | `iw·d·(δ_fold·lb)` ≈ `iw·d·log₂(2β_z)` | **invariant** (1st order) |
| r̂ | `m_row·δ_full` | `m_row·d·(δ_full·lb)` ≈ `m_row·d·field_bits` | **invariant** (weak `n_a` via `m_row`) |

The mechanism: **`δ·lb ≈ magnitude-bits` for every segment** — decomposing a fixed-magnitude value into `δ = ⌈log_b M⌉` digits packed at `lb` bits costs `≈ log M` bits regardless of basis. So re-bucketing digits cannot shrink bytes. The *only* first-order `lb`-dependence is `n_a(lb)` (the SIS module rank), which **grows** because the collision bucket key `d·(2^lb)²` rises with `lb`: measured `d·B² = 1152, 6272, 28800, 123008, 8.3M` at `lb = 2,3,4,5,8`, pushing `n_a` up the table (`crates/akita-types/src/sis/ajtai_key.rs:113`). So bigger basis is byte-neutral-to-worse per segment.

### 3.2 The DP measurement (authoritative)

Running the real planner DP (`find_schedule` on `fp32_d128_onehot`) with `PROOF_OPTIMIZED_LOG_BASIS_MAX` lifted 6 → {8,10,12,16}:

| nv | cap | total bytes | #folds | terminal bytes | per-level lb |
|----|-----|------------:|-------:|---------------:|--------------|
| 20 | 6 (shipped) | 100,976 | 4 | 82,240 | [4,5,5,5] |
| 20 | 8 | **99,952** (−1.0%) | 5 | 64,512 | [8,8,8,8,8] |
| 20 | ≥10 | 99,952 | 5 | 64,512 | [8,…] (never >8) |
| 28 | 6 (shipped) | 116,288 | 7 | 82,000 | [2,2,4,5,5,5,5] |
| 28 | ≥8 | **116,288** (0%) | 7 | 82,000 | unchanged |
| 30 | 6 (shipped) | 132,624 | 11 | 81,040 | [2,2,2,2,2,2,4,5,5,5,5] |
| 30 | ≥8 | **132,624** (0%) | 11 | 81,040 | unchanged |

At nv=20 the DP picks `lb=8` (terminal shrinks 82,240 → 64,512) but adds a 5th fold whose transcript (~8.4 KB) eats ~90% of the gain → net **−1%**. At nv≥28 the DP keeps `lb ≤ 5` for *every* cap up to 16 → **byte-identical**. The DP never picks `lb > 8` even when allowed to 16.

### 3.3 Why the decision-doc analysis pointed the wrong way

The decision doc minimized `|t̂| + |ẑ|` in **ring elements** and found `lb* ≈ 11–16`. That is a genuine *element-count / SIS-rank* optimum, but it does **not** translate to bytes: terminal bytes carry an extra `·lb` factor on top of the element count, and `δ·lb` is magnitude-locked, so the element optimum (`t̂ ∝ n_a/lb`, falling then rising) becomes byte-monotone (`t̂ bytes ∝ n_a ∝ (a·lb+C_W)²`, rising). The "basis is throttled by the norm-check; true SIS optimum is 11–16" reading conflates element-rank with proof-bytes; **the planner minimizes bytes, which is why it empirically sits at `lb=2–5` even with the cap lifted — not because of the sumcheck-cost cap, but because there is no byte win past `lb≈8`.**

### 3.4 Verdict and parent-doc change

- **The basis cap is not throttling proof size.** Lifting it yields ≤1% (nv=20) to 0% (nv≥28). JL "unlocking the basis" is **not a proof-size lever** — delete that as a Branch-B argument.
- **Change decision-doc:** the "SIS rank scaling and the decomposition-basis optimum" section's *byte* conclusion is wrong; keep the element/rank analysis but add that it does **not** yield a byte win (DP-measured). The single quantified Branch-B basis argument is **retracted**.
- **Change protocol §4 lever 2 and §8.8:** the basis unlock is byte-neutral; the DP rerun is no longer "the decisive measurement to confirm a win" — it *was run* and shows no win. Remove `PROOF_OPTIMIZED_LOG_BASIS_MAX` lift from P2 unless justified by a *non-proof-size* objective (e.g. prover time, fewer rounds for recursion-depth reasons — measure separately).
- **What survives:** lifting `lb` may still help **prover time** (fewer digit planes to commit/sumcheck) and **recursion depth** (fewer levels for the verifier-circuit), which are different objectives — flag for separate measurement, do not claim a proof-size win.
- **Reinforced by [`tail-wire-encoding.md`](tail-wire-encoding.md):** once the tail is entropy-coded (the live plan), the wire carries true-entropy integers, not digit-planes, so the basis is not merely byte-neutral but **byte-irrelevant** at the tail — the digits never appear on the wire. The real tail win is entropy coding (`z`: `bound→σ`) + the r/u elisions, all basis-independent (§8).

---

## 4. Verifier cost of the consistency term, and the invalid setup-offload (correcting C4)

### 4.1 The `ω_JL` consistency cost

At the stage-2 final point the verifier evaluates, per projected region, `Σ_j ρ^j·J̃₀(j, r_in)` — the MLE of the `n_J × m₀` matrix `J₀` batched over image rows. `J₀` is a random ±1 (χ) matrix; its MLE has **no closed form**, so the honest cost is the number of nonzeros, `≈ n_J·m₀/2` field additions (±1-sparse dots against one precomputed eq-table), plus `O(ω·log D)` for the negacyclic-shift tensors on the `p̂`-side (the existing carry-automaton tool). At a mid level (`m₀ = 2^{12}`, `n_J ≈ 266`): `≈ 2^{19}` additions, no multiplications. At the root (`m₀` large for small union bound): this is the *other* reason JL is expensive at the root.

### 4.2 The setup-offload (§6.4b) is impossible as written

The protocol spec proposed treating `J` as "a fourth seed-derived setup role" and offloading `J̃` via the PR #138 setup-prefix mechanism (commit the setup table once, fold its evaluation into the setup product sumcheck). **This cannot work:** the setup matrix is *transparent and fixed across all proofs* (it can be committed once as a prefix); `J` is **Fiat–Shamir-derived from the per-proof transcript after `u_ℓ`** (this is exactly what makes the projection independent of `w_ℓ` — §1.2). A per-proof `J` cannot be a committed setup prefix. The only escape would be a `J₀` with a *closed-form MLE* (e.g. generated by a structured PRG whose evaluation has an arithmetic shortcut), which a random ±1 matrix does not have.

### 4.3 Verdict and parent-doc change

- **Change protocol §6.4b:** strike the "J as setup role / PR #138 offload." Replace with: the verifier evaluates `J̃₀` live at `O(n_J·m₀)` additions; the credit comes from **deleting stage-1** (its degree-`2^lb` range tree) and the **stage-2 degree drop 3→2**, not from offloading `J`. Note in-guest this is `O(n_J·m₀)` additions per consistency point — affordable at small `m₀`/late levels, a root-cost contributor (reinforcing §2.3).
- Open: whether a *structured-but-not-random* `J₀` (closed-form MLE, still JL-valid) exists is a research question; if so it would restore an offload path. Low priority.

---

## 5. The modular-JL precondition is not the binding constraint on `m₀` (clarifying the decision doc table)

The decision doc devotes a table to the precondition `‖block‖_2 ≤ q/125` (single) / `q/(125√337)` (nested) and computes `m_max = 12·Bound²/b²`. Re-checked: those `m_max` values are `2^{31}–2^{49}` coefficients — **astronomically larger than any block we would choose.** So the precondition is **never the binding constraint** on block size. The real constraint on `m₀` is the **verifier-cost / union-bound tradeoff** (§2.2, §4.1):

```
#blocks = N_coeff/m₀  (smaller m₀ → more blocks → larger κ_JL → larger n_J)
verifier cost ≈ n_J·m₀  (larger m₀ → more additions)
image fraction = n_J/m₀  (larger m₀ → smaller image overhead)
```

So larger `m₀` is better for **both** κ_JL **and** image overhead, worse **only** for verifier cost — and the product `(#blocks)·(verifier cost) = N_coeff·n_J` is fixed. The design choice is purely "how much verifier work can this level afford," with the precondition a non-issue.

**Ring-granular vs coefficient-granular headroom (clarifying the memory note's "d× headroom"):** a ring-granular block is one coefficient-slice across `m₀` ring elements (`m₀` coefficients, after permutation `M = I_D ⊗ J₀`); the union is over `D·(W/m₀) = N_coeff/m₀` blocks. Coefficient-granular with the same block dimension gives the same block count. The "d×" is **not** a union-bound advantage — it is that ring-granular `J₀` (constant-polynomial entries) is what makes `Ĵ` commute with the fold (T1) so the image can be *committed and folded*; coefficient-granular `J` does not commute and must be revealed or trace-embedded. The granularity choice is about T1/commutability, not headroom.

**Verdict / change:** decision-doc — demote the precondition table from "feasibility risk / block-size limit" to a footnote ("never binding; the constraint is verifier cost"); the `m₀` knob is governed by §2.2/§4.1.

---

## 6. Nested vs single projection: a per-level knob, priced

Because akita enforces `‖p‖` every level (§4 of the protocol — no cross-level compounding), the *per-level* slack is what enters the SIS rank, and the nested-vs-single choice is local:

| | compression/level | per-level slack | image rows | use when |
|---|---|---|---|---|
| single `I⊗J₀` | 256× (one stage) | `√(337/30) ≈ 3.35` | `n_J` | slack/rank dominates (the object is already small; tail) |
| nested `(I⊗J')(I⊗J)` | 256²× | `337/30 ≈ 11.23` | `n_J` (smaller image fraction) | image overhead dominates (large levels) |

The slack gap is `337/30 ÷ √(337/30) = √(337/30) ≈ 3.35×` in collision norm ≈ **1.7 bits ≈ ~0–1 module rank**. So: **nested at large levels** (where `n_J/m₀·δ_p` image overhead matters and the 3.35× slack costs ~1 rank we can afford on a big level), **single at small/tail levels** (where the rank is what we're shaving and the image is already cheap to reveal). This is a clean planner knob, not a global decision. (GD's error of using nested compression while quoting single-stage slack `3.35` is the §-errata C5 of the decision doc — do not repeat.)

---

## 7. ZK (brief; no blocker)

- **Committed Slot-2 `p̂`**: inherits hiding from the commitment — blinding columns extend to the `p̂` segment exactly as for `ê`; the fused consistency term gets a deferred mask in the #154 ZK accounting. No new leakage.
- **Revealed Slot-3 `p`**: leaks `n_J` linear functionals of `w_next` → **non-ZK unless masked**. Mask with `n_J` blinding evaluations (a small deferred-mask family) or restrict the reveal variant to non-ZK builds / terminal-adjacent levels where the witness is about to be sent in the clear anyway. Cursor accounting for the `p̂` segment and the fused-term masks is the §8.7 item; mechanical, no design risk.

---

## 8. Terminal economics — already a dedicated spec ([`tail-wire-encoding.md`](tail-wire-encoding.md)); reconcile, don't rediscover

**Correction to my own first pass.** The "terminal `r̂` elision" I flagged as a novel "largest byte win" is **already PR #141** (terminal r-drop + direct ring relations, *approved*, measured **5.25–6.15%**), and it is only one of *four* levers in Quang's [`tail-wire-encoding.md`](tail-wire-encoding.md) (created 2026-06-13). That spec is the authority on the tail; this section now just *reconciles the JL work with it* and **retracts** my "largest win I found" framing.

The four tail levers (all on the **cleartext** tail only — terminal / terminal-root / zero-fold), from that spec:
1. **r-drop + terminal-stage-2 drop** (PR #141): verifier recomputes the quotient from the cleartext witness; `r̂` and terminal stage-2 leave the wire. **5.25–6.15%.**
2. **u-drop / t-reveal** (B-role drop, mirror of the existing D-role drop): reveal `t` (already on the wire as `t̂`), check `t = A·ê` (A-rows), drop the `u = B·t̂` commitment and its B-rows. ~1 KB, ~free.
3. **z entropy coding** (Golomb–Rice keyed by the public `σ = isqrt(σ_inf²·T·ρ₂)`): the folded response `z` drops from `~log₂(2·bound)` to `~log₂(σ)+2.05` bits/coord.
4. **per-segment width recovery**: one-hot `ê` stops paying `z`/`t`'s global width and collapses to its near-binary entropy.

**This forces two corrections to §3 of *this* doc, in akita's favor:**

- **The ≈82 KB terminal is the *current fixed-width* number, not a floor.** Under [`tail-wire-encoding.md`](tail-wire-encoding.md) the terminal shrinks materially (lever 3 alone monetizes the same `30–200×` bound-vs-realized gap the JL work targets: `log₂(2β_z) → log₂(σ)+2.05` is ~4–6.6 bits/coord off the dominant `z` segment). My "≈82 KB floor, 60–80% of the proof, fixed-point-bound" describes the *pre-encoding* tail; the post-encoding tail is the live plan.

- **Entropy coding makes the basis even *more* irrelevant than §3 concluded — for a deeper reason.** §3's "δ·lb ≈ field_bits magnitude-locks the bytes" is the *fixed-width-packing* statement. Under entropy coding the wire carries **integers/entropy, not digit-planes** (the verifier re-decomposes), so a segment's bytes ≈ its **true entropy**, which is a property of the value distribution and **completely independent of the decomposition basis**. So: fixed-width ⇒ basis byte-neutral (§3, DP-measured); entropy-coded ⇒ basis byte-*irrelevant* (the digits aren't even on the wire). The basis lever is dead under *either* tail encoding. (Bonus: intermediate-level commitment bytes are `rank·D·field_bytes`, independent of digit count, and `rank = n_a/n_b` *grows* with basis — so basis is byte-neutral-to-worse at committed levels too.)

**The JL ↔ tail-encoding relationship (the unifying picture):** both monetize the **same `30–200×` bound-vs-realized norm gap**, at different levels:
- **JL** (this work) tightens the **committed-level SIS rank** by replacing the loose `β̄₂` input while keeping `η_A = 2·Γ̄·β̄₂`.
- **Entropy coding** (`tail-wire-encoding.md` lever 3) tightens the **cleartext tail bytes** — the realized-norm win for the *revealed* witness (`bound → σ`).
They are **complementary, not competing**: JL operates where the witness is committed (recursive levels), entropy coding where it is revealed (terminal). One concrete interaction: a **revealed Slot-3 JL image `p`** (protocol §1, tail levels) is itself a sub-Gaussian vector and should be **entropy-coded by the same Golomb–Rice codec** — fold it into [`tail-wire-encoding.md`](tail-wire-encoding.md)'s segment list as another `Gaussian{k}` segment. The B-role-drop soundness paragraph that spec defers (its S2 risk) and the D4 realized-norm lemma should be written with the same transcript-order discipline.

---

## 9. Consolidated revised recommendation

**The norm-check schedule, with every lever now priced:**

```
ROOT (largest level):     algebraic ∞-range check (stage 1)
                          — wrap-free, no new statistical failure mode, no grindable κ_JL,
                            no O(n_J·m₀) root verifier cost. JL here would need n_J≈290 AND
                            introduce the grindable net-miss exactly where #blocks is largest.
MID levels:               committed ring-granular Slot-2 JL (p̂ in v = D·(ê∥p̂)),
                          nested projection, n_J≈256–270, enforce ‖p̂‖ same-level
                          (micro-range on p̂ digits → exact-ℓ2-on-image upgrade);
                          A-role remains η_A = 2·Γ̄·β̄₂ with JL-derived β̄₂.
TAIL levels:              revealed dense Slot-3 JL (p in clear, single projection),
                          birth-certifies w_next, deletes stage 1 + Slot-2 at the next level.
TERMINAL:                 cleartext witness, ENTROPY-CODED per tail-wire-encoding.md
                          (r-drop PR#141, t-reveal/B-drop, z Golomb-Rice bound→σ, ê width recovery);
                          a revealed Slot-3 JL image is another Gaussian{k} segment there.
```

**Why this shape (the corrected case for Branch B):**
1. **Not** because of a basis/proof-size win — there is none (§3, C3).
2. Because it **deletes the four-square/carry/no-wrap machinery** and the **no-large-level fallback** (Branch A's `D_e ∝ N(b/2)²R < q` gate provably fails at nv=32 roots; the range check + JL schedule has no such gate).
3. Because the **realized `β̄₂` input** can drop the SIS rank while retaining the `Γ̄` factor. This is the real rank lever (§1).
4. Because **no-wrap freedom on the committed image** is achievable on fp32 (digit-range on the decomposed image is exact algebra mod q — Hachi's trick — unlike the exact-ℓ2 conjugated-inner-product check that forced RoKoko/SALSAA to `q≈2^50`).

**Gating items before any implementation (revised):**
1. **D4 realized-norm lemma** — prove that an accepted JL image of `w_next` implies the `β̄₂` bound used by the A-role rank calculation. This includes the map `T_p -> T_w -> β̄₂`, the coordinate-injectivity condition, and the row-count sizing.
2. **§2: pin akita's grinding model** — confirm `n_J ≈ 290` (shipped convention) vs `≈410` (explicit `2^{64}`); decide root paradigm. Likely "range check at root" regardless (§2.3).
3. **Re-derive `c(n_J)`** precisely (LaBRADOR's 256-row constant, not the `n_J/2` extrapolation).
4. ZK masks (§7); the tail-encoding work (§8) lives in its own spec [`tail-wire-encoding.md`](tail-wire-encoding.md) and is branch-independent — coordinate the JL Slot-3 revealed image into its segment list, and write the B-drop soundness paragraph with the same transcript-order discipline used by D4.

**Retracted / corrected items (do not re-introduce):** the basis-unlock proof-size win (§3, C3); `n_J≈410` as akita's requirement (§2, C2); the `J`-as-setup-role offload (§4, C4); the clean `2T̄_s` anchored-price framing (§1, C1); "terminal `r̂` elision is the largest win found" / "≈82 KB floor" (§8, C5 — it's PR #141, and the tail is being entropy-coded).
