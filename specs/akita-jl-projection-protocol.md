# Akita-JL v5: Norm Control by Random Projection (Full Cutover Protocol)

*Replaces the stage-1 ∞-norm range sumcheck and the planned exact-ℓ2-on-z certificate at selected non-root levels with transcript-sampled modular-JL projections, image norm checks, and projection consistency fused into stage 2. v5 corrects the A-role accounting: JL does not remove the operator-norm factor. The weak-binding shape remains `η_A = 2·Γ̄·β̄₂`. JL changes how `β̄₂` is bounded by certifying the realized norm of the already committed next witness `w_next`, whose flat table includes `ẑ`. The previous R-A/R-B anchored-extraction fork is retired.*

| Field     | Value |
|-----------|-------|
| Author(s) | Quang Dao, Claude (Fable 5) draft |
| Created   | 2026-06-12 (v2 same day; v1 history in git) |
| Status    | draft protocol spec (design + security sketch; no implementation) |
| Depends   | PR #154 (y-ring/trace fusion), #155 (collision_l2_sq tables), [`grand-danois-jl-vs-exact-l2-norm-check.md`](grand-danois-jl-vs-exact-l2-norm-check.md) (decision doc + GD errata), [`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) (superseded for the z-certificate if this ships) |

## 1. The window view and the placement theorem

Re-draw the protocol boundary: the natural unit is the **window between the commitment of one recursive witness and the commitment of the next**,

```
[ u_ℓ commits w_ℓ ]  →  (level-ℓ EOR, v, fold c, stage-2 sumchecks, claim (r₂,v′))  →  [ u_{ℓ+1} commits w_{ℓ+1} ]
```

A JL certificate for `w_ℓ` can be produced anywhere in this window (Π is independent of `w_ℓ` from the window's start, since `u_ℓ` is already absorbed). What the window does **not** leave free is the consistency mechanism, because of:

**(T1) Fold non-commutation.** `J·coeff(Σᵢ cᵢsᵢ) ≠ Σᵢ Rot(cᵢ)·J·coeff(sᵢ)` unless `J = J₀ ⊗ I_D` (constant-polynomial entries, "ring-granular"). So:

- **Slot 1 (v1 design, retired):** project the *next* witness right after its first commitment half, check consistency directly against its own stage-2 columns. Permits field-granular `J`, but needs a second commitment absorb `u′₂` and a root special case.
- **Slot 2 (GD slot, adopted):** project the *current* witness `w_ℓ` per fold block at level-ℓ start; commit the image digits `p̂` **jointly with `ê` in `v = D·(ê ∥ p̂)`**; check consistency *through the fold* at this same level (`Σᵢ cᵢ pᵢ = Ĵ z`). T1 then **forces ring-granular `Ĵ`** (entries constant polynomials; the coefficient-level action is `J₀ ⊗ I_D`, JL applied per coefficient slice with a `d×` union bound and `d×` modular-headroom). In exchange: no extra commitment (the `D`-row just widens, `p̂` is "more `ê`" — committed once in `v` for early binding and once in `w_next` for recursion, exactly the existing `ê` pattern). The A-role still keeps the `Γ̄` factor; the projection changes the realized norm input.
- **Slot 3 (reveal, sharpened in v4):** sample dense `J` immediately **after** `u′` absorbs `w_next`; project the **next** witness, `p = J·w_next`, and send `p` in the clear immediately; the consistency `Σ_j ρ^j·⟨J_j, w_next⟩ = ⟨ρ-powers, p⟩` joins the already-open stage-2 instance as one more γ-batched row against `w̃_next`; the verifier checks `‖p‖₂` over Z directly. This is **birth certification**: it certifies the same object — the witness in the gap between its commitment and its folding — that Slot 2 certifies one window later during **consumption**. No T1 constraint applies (nothing is folded after the projection), so field-granular dense `J` is fine; the verifier pays `O(N_next)` ±1-additions to evaluate `Σ_j ρ^j·J̃(j, r₂)` — affordable only at small levels. Wrap-free, covers `ê, t̂, ẑ, r̂` in one shot, and deletes stage 1 (and the Slot-2 projection) at the *next* level outright.

**Corollary (two slots per witness).** Every committed witness `w_ℓ` has exactly two certification opportunities: at **birth** (window ℓ−1, after its `u` — reveal, or a second commitment) or during **consumption** (window ℓ, in `v` — which forces ring-granularity via T1). The terminal witness is its own certificate (cleartext). The per-level paradigm schedule then reads off the cost laws: exact digit-range (basis-capped, no statistical loss) where JL's union bound and verifier additions are unaffordable — the largest level(s); committed Slot-2 JL at mid levels; Slot-3 reveal at tail levels.

**(T2) Conservation law** (unchanged): clear image ⇔ dense randomness ⇔ O(N) verifier; structured `I⊗J₀` ⇔ committed image of `n_J·(width/m₀)` entries certified downstream. Knob: `m₀` trades verifier additions (`n_J·m₀`) against image overhead (`(n_J/m₀)·δ_p` of the witness). At large levels use the **nested** projection (GD Lemma 3.2 — their printed constants are buggy, re-derive; window widens to `[30, 337]`) to push the image overhead to ~0.1% at `2·n_J·m₀` verifier additions.

## 2. The A-role price under JL

The A-role weak-binding price keeps the same shape as the current Euclidean accounting:

```
η_A = 2·Γ̄·β̄₂
```

`Γ̄` is the operator norm of the cross-multiplied fold challenges in `lem:batched-weak-binding`.
It is structural.
JL does not remove it.
The win is that JL supplies a tighter input for `β̄₂`.

Today stage 1 gives a deterministic infinity-norm envelope for the fold response.
The code path `committed_fold_collision_l2_sq` turns this into an L2 bucket by pricing `collision_linf = 8·ω·β_inf·ν` and then applying the L2 MSIS table.
That envelope is known to be 30 to 200 times above the realized `‖z‖₂` in calibration.

With Slot 3 reveal, the verifier samples `J` after `u′` has committed the next witness, receives `p = J·w_next`, checks `‖p‖₂ ≤ T_p` over the integers, and checks projection consistency in stage 2.
The flat witness is:

```
w_next = (ê, t̂, ẑ, r̂)
```

Since `ẑ` is a segment of `w_next`, `‖ẑ‖₂ ≤ ‖w_next‖₂`.
The modular-JL lower-tail argument turns the accepted image threshold into a schedule-fixed witness bound `T_w = f(T_p)`.
That gives a realized bound for the response term that feeds `β̄₂`.
The resulting A-role price is still `2·Γ̄·β̄₂`, but `β̄₂` is no longer the loose `√d·β_inf` envelope.

The previous R-A/R-B fork is retired for this path.
In particular, do not claim a clean `η_A = 2·T̄_s` price, and do not describe the main design as an extraction re-architecture that removes `Γ̄`.
The remaining D4 work is narrower: document the map `T_p -> T_w -> β̄₂`, choose the row count and regrind policy, and update `norm_bound.rs` and the planner tables to consume the realized bound.

Consequences:
1. The v1 "F2" family that projected only the recomposed fold response is still retired. The correct projected object for the reveal path is the committed `w_next`, because it contains `ẑ` and is fixed before `J`.
2. Operator-norm certification remains part of the A-role price.
3. The 30 to 200 times calibration gap is used for soundness pricing through `β̄₂` and for honest `T_p` sizing.

## 3. Protocol Π_JL-level (v2)

Post-#154 baseline; e-notation; single claim shown (batching orthogonal). `χ`: `P(0)=½, P(±1)=¼`. Schedule fixes per level: `n_J`, chunk width `m₀` (ring elements), nesting flag, image digit base/depth `(b_p, δ_p)`, per-block caps `T̄`, variant ∈ {committed, reveal}; all descriptor-bound.

1. *(K>1)* EOR sumcheck → `final_claim`; trace target with scaled weights (as in #154). Unchanged.
2. **V→P (FS): Ĵ from a transcript seed** — ring-granular: `J₀ ← χ^{n_J×m₀}` lifted entrywise to constant polynomials, block structure `I_{M/m₀} ⊗ J₀` per fold block (nested at large levels: second independent layer applied to the image before decomposition). Independence of `w_ℓ` holds because `u_ℓ` is in the transcript prefix.
3. **P: per fold block `i`, `pᵢ = Ĵ sᵢ ∈ R^{n_J·M/m₀}`** (sparse ±1 ring additions); regrind nonce if any image digit cap is exceeded; `p̂` = balanced base-`b_p` digits. **P→V: `v = D·(ê ∥ p̂)`** (the `D` role widens; one wire object as today).
4. V→P: fold challenges `c`; P computes `z = Σᵢ cᵢsᵢ` (regrind if `‖z‖₂` exceeds the honest bucket — completeness only).
5. P→V: `u′ = U·w_next`, `w_next = (ê, p̂, t̂, ẑ, r̂)` — `p̂` recurses exactly like `ê`.
6. V→P: `ρ ∈ E` (Vandermonde batch over image coordinates), `α`, `τ₁`. No `τ₀`, no stage 1.
7. **Stage 2, one degree-2 fused sumcheck** (γ after `u′`):

```
input:   V_α + γ·T_tr + γ²·0
oracle:  W̃(x,y)·[ α̃(y)·m_τ₁(x) + γ·ω_Tr(x,y) + γ²·ω_JL(x,y) ]
```

   where `ω_JL` encodes, per ring-coefficient slot, the folded consistency `Σᵢ cᵢ·(Ĝ_p p̂)ᵢ − Ĵ·Ĝ_z ẑ = 0` batched by powers of `ρ`:
   - on `p̂` columns: gadget tensor × shifted-Vandermonde terms, one per nonzero monomial of `cᵢ` (negacyclic shift = offset eq-tensor, the existing carry-automaton tool; `O(ω·log D)` per block at the final point);
   - on `ẑ` columns: gadget tensor × `Σ_{j<n_J} ρ^j·J̃₀(j, r_in)` — the eq-table once (`O(m₀)` mults), then ±1-sparse dots: `O(n_J·m₀)` **additions, zero multiplications**;
   - SZ in `ρ`: `(n_im−1)/|E|`. Degree 2; no new `M` rows beyond the widened `D`-row; `r̂` grows only by the `D`-row quotient widening.
8. Output claim `(r₂, v′ = W̃(r₂))` on `u′`. **Terminal:** `w_L` cleartext (includes the last `p̂`); verifier checks the relation rows and per-block `‖·‖₂ ≤ T̄(L)` over Z. This grounds the recursion.

**Reveal variant (small `N`, non-ZK):** per sharpened Slot 3 — `J` sampled after step 5 (`u′`), `p = J·w_next` in clear (~1 KB whole-witness; per-block grouping if per-block anchors are wanted), consistency one γ-batched row in step 7 against `w̃_next`, verifier checks `‖p‖₂` over Z (wrap-free), no `p̂`, no recursion of the image — and the *next* level runs no stage 1 and no Slot-2 projection. Slack 2.07 with median threshold + regrind (LaBRADOR's check-and-retry pattern).

## 4. Same-level image-norm enforcement and the (non-)compounding ladder

**The telescoping lemma (why enforcement is mandatory).** If `‖p‖` is left to the next level's certificate, the per-level certified/honest ratio obeys

```
slack(ℓ) ≈ [ √(Σ_h b_p^{2h})·‖p̂‖_honest / ‖p‖_honest ] · √(n_J/2)/√30 · slack(ℓ+1)
         ≈ √(12·δ_p) · √(n_J/2)/√30 · slack(ℓ+1)   ≈   14 · slack(ℓ+1),
```

i.e. `14^{#levels}` at the root — sound but useless. Even projecting recomposed values (no `√(12δ_p)`) still compounds at the bare JL slack/level. (Two different compoundings, kept distinct in v4: **(i)** full deferral of image-norm enforcement to the next level's certificate — the telescoping above; this is what GD's printed protocol does, with no compounding analysis at all, and at its corrected ≥11.23×/level the `q/(125√337)` modular-JL precondition budget of a 32-bit modulus is exhausted within ~2 levels. **(ii)** LaBRADOR Remark 5.2's milder effect: with `p` revealed and checked *every* level, witness norms reset and the per-level slack `√(128/30) ≈ 2.07` multiplies only into the *assumed MSIS bound* — ≈6.3 bits over 7 levels — which akita's schedule-fixed per-level `T̄` buckets absorb at sizing time, leaving the chain flat.) RoKoko runs an exact norm sumcheck each round for the same reason. **Conclusion: every level enforces `‖p‖₂ ≤ T_p` against a schedule-fixed bucket at the level that produces `p`.** The chain then becomes local: `‖s̄ᵢ‖₂ ≤ T_p,ᵢ/√30`, flat, no induction; the only cleartext norm check left is the terminal witness itself.

**Enforcement menu (per level, planner-selected like tiered/terminal modes):**

| mechanism | flat slack | machinery | where |
|---|---|---|---|
| reveal `p` in clear, verifier sums squares over Z | ≈ 2.1–2.3 (median+regrind) | none; wire = image bytes; leaks `n_im` functionals (non-ZK or masked) | image ≤ few KB: late/small levels; the root is excluded |
| committed `p̂` + micro range check on the `p̂` segment | ≈ 10–12 (max-vs-RMS of a range bound) | stage-1 code segment-restricted, tiny base `b_p` (caps only `b_p`, not the main basis) | first implementation at large levels |
| committed `p̂` + exact-ℓ2 on the image (§3.6 relocated) | ≈ 2.3–3.4 | four-square limbs + grouped-carry on `N = n_p·δ_p ≈ 2^16` cells — the no-wrap gate `D_e ∼ N·(b_p/2)²·R ≈ 2^21 ≪ q` **clears at every level** (vs provably failing on `z` at nv=32 roots); carries handle `Σp² > q` | large levels, refinement |

This menu is the prior-work pattern, verified (2026-06-13 deep-read): **no prior work runs an exact mod-q ℓ2 identity on a raw JL image** — the image concentrates mass (`E‖p‖₂² = (n_J/2)·‖w‖₂²`, *larger* than the witness), so the wrap wall is worse on `p` than on `w`. LaBRADOR reveals (row 1). RoKoko and RoK-and-Roll gadget-decompose the image *first*, commit digits, and certify the digits (rows 2–3): RoKoko's exact `trace⟨ŵ, ŵ̄⟩` sumcheck needs `r′·β̃²·f̂ < q/2` and is a stated reason for their `q ≈ 2^50` (SALSAA's `β′² < q/2` likewise) — structurally unavailable on fp32, which is why row 2's mechanism (digit-set membership is *exact algebra mod q*, wrap-free — Hachi's own trick) and row 3's carry-lifted variant are the small-field instantiations. RnR footnote 21's hazard ("the norm check is completely insecure in the presence of slack" — the ¼-fraction wraparound counterexample) does not bite here: every menu row runs on committed digit planes of `w_next`, never on relaxed openings.

**Pricing** (all Euclidean, #155 tables): **A-role realized-norm input** `η_A = 2·Γ̄·β̄₂`, where `β̄₂` comes from the accepted JL image of `w_next`; **B/D/u′ roles** remain digit-collision roles unless their own segment certificates are replaced. Basis cap lifts are not proof-size wins, but they may still help prover time or recursion depth. Buckets are schedule-fixed; prover regrind absorbs honest image-tail events; the 30 to 200 times `‖z‖₂` calibration informs both `β̄₂` sizing and honest `T_p` buckets.

**Tail ground truth (v4, from the planner tables, fp32_d128_onehot — *current fixed-width packing*):** the terminal cleartext witness is ≈82 KB ≈ **60–80% of the total proof** at every nv ≤ 30; `next_w_len` has a fixed point ≈1,020–1,035 rings (lb=5/n_A=8 steady state: ê 56 / t̂ 448 / ẑ 430–960 / r̂ 84–98); an intermediate level costs ≈6.4 KB (of which stage 1 ≈2.4 KB at lb=5), a terminal fold level ≈2 KB. The tail is **fixed-point-bound, not level-count-bound** — so round count does **not** dominate. **But the ≈82 KB is the *pre-entropy-coding* number**: [`tail-wire-encoding.md`](tail-wire-encoding.md) is the dedicated, active plan to shrink it (r-drop PR #141 = 5.25–6.15%; t-reveal/B-drop; `z` Golomb–Rice `bound→σ`; one-hot `ê` width recovery). JL/anchoring and that entropy coding monetize the **same `30–200×` bound-vs-realized gap** at committed vs cleartext levels — complementary (see [`resolutions §8`](akita-jl-norm-check-resolutions.md)).

**Tail levers** (re-ranked after the byte audit; all DP-gated, §8):
1. **Realized `β̄₂` shrink** — replacing the deterministic `√d·β_inf` envelope with the JL-derived realized bound can drop `n_a` at committed levels. `t̂` is `∝ n_a` at every level, so this is the rank lever that can reduce both recursive commitments and the terminal `t̂` segment. It composes with the [`tail-wire-encoding.md`](tail-wire-encoding.md) entropy coding.
2. **Basis unlock — NOT a proof-size lever (DP-measured; [`resolutions §3`](akita-jl-norm-check-resolutions.md)).** The `δ·lb ≈ field_bits` packing identity magnitude-locks *every* cleartext segment in bytes (ê, ẑ, r̂ basis-invariant; t̂ *grows* via `n_a(lb)`). Measured: lifting `PROOF_OPTIMIZED_LOG_BASIS_MAX` 6→16 changes total proof size by **−1% at nv=20 (eaten by an extra fold) and 0% at nv≥28** (the DP keeps `lb≤5` for every cap). The decision doc's `lb*≈11–16` is an *element/rank* optimum that does **not** translate to bytes; the earlier "~3–4× on the tail" was a ring-element count, retracted for bytes. Bigger basis may still help *prover time* / *recursion depth* (separate objectives — measure separately); it does **not** shrink the proof.
3. **Projection finisher — corrected claim.** A reveal-projection cannot replace the cleartext terminal: the Direct opening *is* the PCS base case, and something must still verify the evaluation claim (GD's §3.2.4 finisher composes with Greyhound as the base PCS; akita's base is Direct). What Slot-3 reveal actually buys at the last 1–2 committed levels: deletes stage 1 (−2.4 KB/level at lb=5) for +≈1 KB of revealed image, and unlocks levers 1–2 there.
4. **`r̂` quotient compression — GD's batched-quotient version is unsound as printed (§6.11); the terminal elision is already shipping.** GD's sound-ordering analogue (commit `w_next` without a batched quotient, send the single τ₁-batched quotient in the clear) is an `L`-valued ring element ≈2 KB/level against `r̂` ≈ 90 rings — a loss at committed levels. The win that *is* real is the **terminal** one (verifier recomputes the quotient from the cleartext witness, elide `r̂`): that is **PR #141** (approved, 5.25–6.15%), one lever of [`tail-wire-encoding.md`](tail-wire-encoding.md) — not a JL-branch item.
5. Stage-1 wire deletion at late levels nets against the revealed image (subsumed by lever 3).
6. Modulus switching composes iff each image is enforced before the switch (fused terms, no dense rows to transport).

## 5. Security sketch

**Per-level knowledge error**

```
ε = ΛB/|C| + [P + 2D + ⌈log n⌉ + 2·(deg-2 sumcheck) + (n_im−1) + 2]/|E| + (Q+1)·κ_JL,
κ_JL = (#chunks × d slices)·2^{−c(n_J)}      (statistical; Q = FS grinding budget)
```

**Extraction (now level-local in the norm chain).** (1) Level ℓ+1 supplies the unique opening of `u′` under the usual A/B/D binding arguments. (2) Stage-2 soundness forces the relation, the trace identity, and the JL consistency row. (3) Same-level enforcement bounds the accepted image: revealed images are checked over the integers, and committed images use a same-level image certificate. (4) Modular JL turns the accepted image threshold into a schedule-fixed witness bound for the projected object. For the Slot-3 reveal path, that object is `w_next`, and `ẑ` is one of its segments. (5) Weak binding still forms the A-role collision with `η_A = 2·Γ̄·β̄₂`; the JL witness bound supplies the realized `β̄₂` input. (6) Completeness uses honest norms plus regrind.

**The one argument that must be written in full before implementation:** the recursive tree extraction in the style of Greyhound relaxed soundness, adapted to akita's `thm:batched-root-cwss`. The writeup must cover the `t̂` bookkeeping, the uniqueness bootstrap order, the `(Q+1)` FS amplification of the statistical `κ_JL`, and the JL independence condition. For the Slot-3 reveal path, `J` is sampled after `u′`, so the projected object is the already committed `w_next`; the proof must show that an accepted `(p, J, w_next)` gives the `β̄₂` consumed by `norm_bound.rs`. It must not claim that JL removes `Γ̄`. The root FS budget remains as corrected in [`resolutions §2`](akita-jl-norm-check-resolutions.md): under akita's shipped convention, root `n_J` is about 285 to 295, not 410, and the root still favors the algebraic range check because JL adds a grindable statistical failure mode where the projected object is largest.

## 6. Fine points

1. **T1 is the granularity–slot law:** Slot 2 (image in `v`, consistency through the fold) ⇔ ring-granular `Ĵ`; field-granular `J` ⇔ Slot 1 (extra commitment, direct consistency). There is no field-granular-J-in-`v` option.
2. **Realized `β̄₂` is the prize**; JL tightens the norm input to the existing `2·Γ̄·β̄₂` A-role price. Do not claim that the operator-norm factor leaves the price.
3. **κ_JL union bound:** chunks × `d` slices; `n_J = 256` gives ~`2^{−105..−110}` at root scale. Either `n_J ≈ 256 + 2log₂(chunks·d)` or a documented statistical gap. (RoKoko's impl silently ignores this; don't copy.)
4. **Same-level enforcement of `‖p‖` is mandatory** (§4 telescoping lemma) — v1/v2 were wrong to drop it. On fp32 the exact-ℓ2 form needs grouped-carry (`Σp² > q` at dense levels), which is cheap at image scale; reveal is wrap-free.
4b. **Verifier cost: `J` CANNOT be offloaded as a setup role (correcting a v3 error; [`resolutions §4`](akita-jl-norm-check-resolutions.md)).** The earlier "commit `J` as a PR #138 setup prefix" is **invalid**: `J` is Fiat–Shamir-derived from the per-proof transcript (after `u_ℓ`) — that per-proof freshness is exactly what makes the projection independent of `w_ℓ` — so it is *not* a transparent fixed table and cannot be a setup prefix. The verifier evaluates `J̃₀` **live**: `O(n_J·m₀)` ±1-sparse additions (against one precomputed eq table, zero multiplications) + `O(ω·log D)` shifted tensors on the `p̂` side. The budget credit comes from **deleting stage 1** (degree-`2^lb` range tree) and the **stage-2 degree drop 3→2**, not from offloading `J`. `O(n_J·m₀)` is cheap at small `m₀`/late levels and a root-cost contributor (a further reason to keep the range check at the root, §5). A *structured-but-not-random* `J₀` with a closed-form MLE would restore an offload path — open research question, low priority.
5. **Image overhead vs verifier additions:** single-layer `(n_J/m₀)·δ_p` (≈25% at `m₀=2^12`, ≈4% at `2^15`); nested `(n_J/m₀)²·δ_p` (negligible) at `2·n_J·m₀` additions and `[30,337]` window. Nest at large levels, single at small, reveal at terminal-adjacent.
6. **Transcript binds the projection geometry** (seed, `n_J`, `m₀`, nesting, `b_p`, `δ_p`, `T̄` ladder, variant) — a proof under one geometry must not verify under another. Wire-before-squeeze: `v` (with `p̂`) before `c`; `ρ` after `u′`.
7. **ZK:** committed variant inherits hiding (blinding columns extend to `p̂`; deferred masks for the fused terms as in #154's ZK accounting). Reveal variant leaks `n_J` functionals — non-ZK or masked.
8. **Stage-1 removal reprices everything**, not just A: audit every `norm_bound.rs` consumer of "digits are range-checked". Stage 2 drops to degree 2; range tree, carried `s_claim`, `τ₀` all deleted; knowledge-error budget shrinks at every level.
9. **EOR/trace untouched**; `p̂` placement must respect the `z_first` column-alignment rules (it sits in the `ê`-side family; same offset/carry treatment).
10. **fp128 note:** Branch A's exact certificate is cheap at `q ≈ 2^128` (no wrap); the one-branch repo policy should be re-argued against fp128 numbers explicitly.
11. **GD's interaction order is confirmed broken — do not copy Fig. 1.** GD samples `(c, c′, ĉ)` in one message *before* `cm′`, then commits the `c′`-batched single quotient `r̂` — and `ẑ` — inside `cm′`; worse, the `cm′`↔`v`/`cm` consistency itself lives only inside the `c′`-batched single row, so *nothing* in `cm′` is bound pre-challenge. SZ over `c′`/`ĉ` gives nothing for data committed after the challenge, and the Thm 3.3 sketch embeds the flaw (its "alternative protocol" sends the witness in plaintext *after* receiving `c, c′, ĉ`). Sound orderings: (a) commit `w_next` first, then sample the batcher, then send the batched quotient in the clear (extra round, ~`d·k·log q` clear bits, quotient leaves the recursive witness); (b) per-row quotients committed pre-batch — which is exactly akita's shipped `r̂`-in-`w_next` + post-`u′` τ₁ structure. **Akita's current ordering is already the sound one**; the lesson is a checklist, not a fix: every object entering a batched/SZ-checked relation must be bound before the batching challenge is squeezed. For this spec that means: `p̂` before `c` (in `v`); `w_next` before `α, τ₁, ρ, γ` (in `u′`); the reveal-variant `p` in the clear before `ρ`; the projection geometry before the seed.
12. **Baseline and planner status (2026-06-13):** main is pre-#154 (#154 and #175 are open PRs); this spec's step list assumes #154 lands first. The shipped fp32 planner rows at nv=31/32 are **degenerate stubs** (single root fold + 1.2 GB Direct, the uncommittable-edge fallback); every "nv=32 root" calibration in this spec and the decision doc should be read against the real nv=28–30 rows (7–11 folds, 120–139 KB total, terminal cleartext ≈82 KB) until the planner handles nv ≥ 31. The nv=32 *gate-failure* analysis (no-wrap `D_e` vs `q`) is unaffected — it is parameter math, not a planner readout.

## 7. Migration

- **P0:** #154; #155 (done); keep op-norm work alive but no longer gating.
- **P1 (hybrid):** keep stage 1; add the Slot-2 projection and fused consistency, but keep the A-role shape `η_A = 2·Γ̄·β̄₂`. Image-norm enforcement comes free here because stage 1 still covers the whole witness including `p̂`, which is the micro-range mechanism. Smallest sound increment; works at every level including nv=32 roots.
- **P2 (full):** drop stage 1 except the `p̂`-segment micro-range (or upgrade to exact-ℓ2-on-image) at selected non-root levels. Keep stage 1 at the root, where the algebraic-vs-statistical hygiene argument ([`resolutions §2.3`](akita-jl-norm-check-resolutions.md)) is strongest. Use the reveal variant at small levels, plug the JL-derived `β̄₂` into A-role sizing, audit B/D/u′ assumptions, and regen the ladder and generated schedules. Do **not** lift `PROOF_OPTIMIZED_LOG_BASIS_MAX` for proof-size reasons. It is byte-neutral per [`resolutions §3`](akita-jl-norm-check-resolutions.md). Lift it only for measured prover-time or recursion-depth reasons.
- **P3:** exact-ℓ2-on-image upgrade (slack 12 → 2.3–3.4); projection finisher for the last 1–2 levels; `J`-as-setup-role offloading; nested tuning; ZK masks.

## 8. Measurements / open items before commitment

1. Write the §5 extraction in full (the gating item; everything else is engineering).
2. Re-derive the nested-projection constants (GD Lemma 3.1/3.2 are typo-laden) and fix the threshold policy (median+regrind 2.07 vs tail 3.35 single; /30 window nested).
3. In-guest cost: PRG trits + `n_J·m₀` additions + `ω·log D` shifted-tensor evals per level, swept over `m₀`, vs deleted stage-1 work.
4. Planner DP with JL realized-norm A-role pricing and explicit JL step costs. Quantify `n_a`, `t̂`, and proof-byte deltas against shipped tables.
5. `n_J` table per level for `2^{−128}` strict.
6. Witness-growth audit per level (image planes, single vs nested) through the DP.
7. ZK cursor accounting for the `p̂` segment and fused-term masks.
8. **Terminal fixed-point economics: RESOLVED ([`resolutions §3`](akita-jl-norm-check-resolutions.md)).** Basis unlock is **not a proof-size lever** (DP-measured: −1% nv=20, 0% nv≥28; `δ·lb≈field_bits` magnitude-locking + `n_a(lb)` growth). The ≈82 KB cleartext floor is real and `n_a`-sensitive (lever 1 attacks it). Still open: re-derive `c(n_J)` (LaBRADOR-256-constant, not `n_J/2`); fix the degenerate nv=31/32 planner rows.
9. **Root-level JL go/no-go: RESOLVED in direction ([`resolutions §2`](akita-jl-norm-check-resolutions.md)).** `n_J(root) ≈ 285–295` under akita's shipped convention (not 430); the algebraic-vs-statistical hygiene argument keeps stage 1 at the root regardless. Confirm akita's grinding model (symbolic `Q` vs explicit `2^{64}`) to finalize.
10. **Tail encoding is a dedicated spec** [`tail-wire-encoding.md`](tail-wire-encoding.md) (r-drop PR #141 = 5.25–6.15%, t-reveal/B-drop, `z` entropy coding, `ê` width recovery) — branch-independent and already the active tail plan. JL action: route a revealed Slot-3 image into its segment list as a `Gaussian{k}` segment; co-write the B-drop soundness paragraph with the D4 realized-norm lemma, since both need the same extraction-order discipline.
11. **D4 realized-norm lemma (THE gate):** prove that an accepted JL image of `w_next` implies the `β̄₂` bound used by the A-role rank calculation, including the map `T_p -> T_w -> β̄₂` and the coordinate injectivity condition.

## 9. Architecture surfaces

- `akita-challenges`: `χ` seed expansion, geometry binding.
- `akita-types`: `jl_weight/` sibling of `trace_weight/` (shifted-Vandermonde/offset closed forms, `J̃₀` inner factor); widened `D`-row layout; `p̂` segment in `layout/`; `sis` realized-norm repricing.
- `akita-prover`: image computation (ring-granular sparse accumulation, nested option), `p̂` decomposition into `v` and `w_next`; stage-2 third addend; stage-1 deletion (P2).
- `akita-verifier`: fused-term claims; `ω_JL(r)` evaluation; terminal block-norm checks; stage-1 deletion (P2).
- `akita-planner`/`akita-config`: `T̄` ladder, `(n_J, m₀, nesting)` schedule, basis range lift, regen.
- Writeup: new §3.6′ replacing the L2-certificate section; §3.12 states the D4 realized-norm proposition that feeds `β̄₂` while retaining `Γ̄`.
