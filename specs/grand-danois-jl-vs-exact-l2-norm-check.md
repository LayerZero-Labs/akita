# Norm-Check Branch, SIS-Tail Scaling, and the Decomposition-Basis Optimum

*Structured JL (Grand Danois) vs exact ∞+ℓ2 sumcheck certificate, plus the analytical model for the ideal decomposition basis.*

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao, Cursor agent draft (model: Claude Opus 4.8) |
| Created     | 2026-06-11 |
| Status      | decision doc + knowledge-dump note, open (jury still out on the branch) |
| PR          | none yet |

> **Reading note for future AIs / maintainers.** This file is a comprehensive design + knowledge dump, not just a yes/no decision. It contains (1) the two norm-check branches and how they differ, (2) a precise SIS-hardness scaling model derived from akita's own generated floor tables, (3) a closed-form for the proof-tail-optimal decomposition basis, and (4) the concrete small-field modular-JL numbers. The headline result lives in Design → "SIS rank scaling and the decomposition-basis optimum". A quick-reference table of the key numbers is at the very end.

## Summary

Akita proves that the fold response `z = Σ_i c_i · s_i` is short so the extractor can recover a norm-bounded weak opening.
Today this is an `‖z‖∞` digit range check, and the in-flight L2-MSIS cutover ([`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md)) adds an *exact* Euclidean certificate proving `Σ z[i]^2 ≤ B_l2` by sumcheck.
Grand Danois (eprint 2026/1196), the direct successor to Hachi (eprint 2026/156, akita's lineage), proves the same shortness a different way: a *structured Johnson-Lindenstrauss projection folded into the evaluation relation itself*, so norm and evaluation are one protocol.

This document does not pick a winner. **⚠️ Correction (2026-06-13, [`akita-jl-norm-check-resolutions.md`](akita-jl-norm-check-resolutions.md) §3): the "basis lever" argument below is RETRACTED for proof size.** The SIS-tail analysis ("SIS rank scaling and the decomposition-basis optimum") minimizes `|t̂| + |ẑ|` in **ring elements** and finds `lb* ≈ 11-16` — but this does **not** translate to proof **bytes**: the `δ·lb ≈ field_bits` packing identity magnitude-locks every cleartext segment, and the only residual `lb`-dependence is `n_a(lb)`, which *grows*. DP measurement (cap lifted 6→16) shows total proof size changes by **−1% at nv=20 and 0% at nv≥28**. So the basis is **not** a Branch-B proof-size argument; the case for Branch B rests on protocol simplicity (no four-square/carry/no-wrap), no large-level fallback, and the anchored A-role rank drop — not the basis. The analytical lb* below is retained as a correct *element/rank* optimum (relevant to prover time / recursion depth, not bytes).
It frames the two as two branches of the same norm-control family, states precisely where they differ, defines what we would have to measure to decide, and records one hard constraint: **we do not adopt vSIS**, so any JL adaptation here must run under akita's existing uniform-MSIS commitments and akita's own sumcheck verifier, not under Grand Danois's vSIS structured CRS.

## Intent

### Goal

Decide whether akita's recursive norm check should stay on the exact `∞ + ℓ2` sumcheck-certificate branch (current cutover) or move to a structured-JL-in-relation branch adapted from Grand Danois, by pinning the concrete slack, witness-growth, proof-size, verifier-cost, and small-field-feasibility trade between them.

### The two branches

Both branches prove the *same* security object: the realized shortness of the fold response `z` (the A-role weak-binding object, Lemma 7 in [`weak-binding-norm-fix.md`](weak-binding-norm-fix.md)), sound against an adversarial prover, not an honest-distribution estimate.
They differ only in *how* shortness is proved.

**Branch A — exact `∞ + ℓ2` sumcheck certificate (current, in-flight).**
The fold response is decomposed into balanced digits `z_hat`, range-checked for `‖z‖∞` (stage 1), and the L2 cutover adds a certificate proving the integer inequality `Σ z[i]^2 ≤ B_l2` exactly.
The certificate machinery is four-square slack `ell_hat`, a carry chain `carry_hat`, one degree-2 squared-sum sumcheck `Σ_x Z_α(x)^2 = V`, and a per-exponent no-wrap gate (`l2-msis-opnorm-folded-witness.md` lines 638-842).
Slack is ≈ 1 (the bound is exact) wherever a group size passes the no-wrap gate.
Two status caveats that matter for the decision:
- **The fallback is a design path, not yet a live behavior.** The spec does define a fallback (`l2-msis-opnorm-folded-witness.md` lines 828-831: "no certificate; price A-role at `L2_BOUND_SQUARED`"), but the realized certificate (S8-S10) is **unshipped**, so *today every level is priced at the deterministic `L2_BOUND_SQUARED` envelope* — the same loose `√d·β_inf` worst case the L2 cutover is meant to replace. Only #155 infra (L2 tables, `collision_l2_sq` rename, operator-norm rejection) and the S1/S4/S7 primitives have landed.
- **The no-wrap gate fails on the largest levels, not "rarely".** The gate is `D_e + … < q` with the binding term `D_e ∼ N·(b/2)²·R` (`N` = certified coefficients, `R = num_digits_fold`). At nv=32 the root fold response has `N ≈ block_len·d ≈ 2^30` while `q ≈ 2^32`, so the gate needs `(b/2)²·R \lesssim 4`, which **fails for every basis in range including `lb=2`**. So the largest levels at high `nv` cannot form the exact certificate at all and are stuck on the loose `β_inf` price — exactly the levels that dominate proof size and SIS rank, and exactly where Branch B (no gate, flat 3.35× slack) has the most to offer.

**Branch B — structured JL folded into the relation (Grand Danois, adapted, no vSIS).**
The verifier samples small structured projections, the prover commits the projection image `p = Ĵ·s` *in the same commitment as the opening witness*, the projection-consistency rows are folded into the *same* constraint matrix the evaluation sumcheck already proves, and the verifier accepts shortness from a norm check on the image.
There is no separate norm protocol, no four-square slack, no carry chain, no no-wrap gate, and no large-level fallback.
**Corrected slack picture (verified deep-read 2026-06-13, superseding the paper's claims):** the paper's headline `√(337/30) ≈ 3.35` (Thm 3.3) is the *single-stage* projection slack; the nested two-stage projection the protocol actually runs has window `[30, 337]`, hence per-round slack `337/30 ≈ 11.23`, and the paper mixes the two (Fig. 1's `√(337/30)·ω` bound on `p_i` is ~38× below the honest expected image norm and fails completeness outright). Moreover GD never enforces `‖p‖` at the level that produces it — enforcement is deferred down the recursion with no compounding analysis, which telescopes (see the protocol spec §4) — so an akita adaptation must add per-level image-norm enforcement that GD does not have. With enforcement, the achievable flat slack is ≈2.1–3.4 per the protocol spec's enforcement menu.

### Invariants (must hold for either branch)

- The certified object is the realized fold response `z` (or its committed digits), tied to the same `w_next` that is decomposed into the next recursive witness.
  A prover cannot certify one object and commit a different recursive witness.
- The norm proof is sound against adversarial accepting transcripts via CWSS, not an honest-distribution calibration.
  Calibration tables select thresholds and estimate prover abort rates; they never justify a bound.
- The verifier-reachable path is no-panic: malformed certificate, projection, challenge, digit, or shape is rejected with `AkitaError` / `SerializationError`.
- Whichever branch ships, exactly one branch ships.
  No dual norm-check tables, no parallel certificate-and-JL paths, no compatibility shim (full-cutover repo policy).
- Transcript binding distinguishes the active norm-check branch, so a proof under one branch cannot verify under the other.

### Non-Goals

- **No vSIS.**
  We do not adopt the vanishing-SIS structured CRS that Grand Danois uses for its polylog verifier (paper Def 2.2, lines 651-675).
  vSIS is a newer, less battle-tested assumption (the authors say so, lines 247-257), and akita keeps uniform-MSIS transparent setup.
  Consequently we also do not take Grand Danois's verifier-succinctness mechanism (per-row tensor MLE evaluation over `R_{q^k}`); akita keeps its own sumcheck verifier.
- No change to the operator-norm challenge family or the `num_digits_fold` `‖z‖∞` sizing; both branches inherit those from the existing cutover.
- No commitment to `F_{q^k}`-coefficient polynomials (Grand Danois drops this too, line 255).
- This doc decides the branch; it does not implement either.

## Evaluation

### Decision criteria (what resolves the jury)

The branch decision is settled when we have, for akita's production families (fp31/fp32 small-field and fp128, at representative `nv`):

- [ ] **SIS rank delta.**
      The A-role rank under Branch B's `√(337/30)` slack vs Branch A's exact `B_l2`, run through the same L2 estimator and planner (`sis-euclidean-estimator.md`, `collision_l2_sq` path).
      Branch B inflates the certified norm by 3.35×, hence the collision bound by 3.35× and the rank by the table ladder; quantify the resulting rank/byte cost per level.
- [ ] **Per-level witness growth.**
      Branch A adds `ell_hat` (4 slack limbs) + `carry_hat` (~`2R` carries).
      Branch B adds the projection witness `p̂` (~256 ring elements at the projected level, folded into the next recursive witness).
      Compare both against the existing `z_hat`/`w_hat`/`t_hat`/`r_hat` per-level witness.
- [ ] **End-to-end proof size.**
      Plug both into the planner proof-size model and compare total bytes at `nv ∈ {20, 28, 32}` for each family, including the round count (Branch B's nested projection may cut rounds; see below).
- [ ] **Small-field modular-JL block sizing (not a hard wall, see Design).**
      The modular-JL precondition is a *max projection-block size*, not an infeasibility: each block needs `‖block‖₂ ≤ q/(125√337)` (nested) or `q/125` (single). The committed digits give `‖block‖₂ = √m·b/√12`, so `m_max = 12·Bound²/b²`. At today's basis (`lb 2-3`) the whole root witness fits in one nested block on both fp31 and fp32; even at aggressive `lb=6` it only needs ≈10 nested blocks on fp31 (single fits in one). Confirm the block count and the resulting compression-ratio erosion, not feasibility.
- [ ] **Verifier cost (Jolt recursion).**
      Branch B's projection-consistency rows add structured-matrix evaluation work to the verifier sumcheck.
      Estimate the guest-cycle delta against the current ring-relation verifier, since the recursion target is the binding constraint, not laptop verify time.
- [ ] **Fallback behavior.**
      Confirm the level set where Branch A hits the deterministic no-wrap fallback (loose bound, lost tightening) and quantify what Branch B's flat 3.35 slack buys on exactly those levels. Expectation from the gate math: at high `nv` the root levels fail the gate for *all* bases, so they never certify under Branch A.
- [ ] **Decomposition-basis sweep (the SIS-tail argmin, see Design).**
      ~~Re-run the planner DP with the `lb` cap lifted past 6~~ — **DONE ([`resolutions §3`](akita-jl-norm-check-resolutions.md)): lifting the cap 6→16 changes total proof size by −1% (nv=20) / 0% (nv≥28); the DP never picks `lb>8` and keeps `lb≤5` at large nv.** The element-count `lb*≈11-16` is real but byte-neutral. There is no Branch-B basis *byte* win to quantify. (If a *prover-time* or *recursion-depth* objective is added later, re-run the DP against THAT objective — it was not measured here.)

### Testing Strategy

The doc itself ships no code, so it adds no tests.
The measurement work above should reuse:

- the existing L2 estimator and golden tables (`scripts/gen_sis_table.py`, `scripts/sis_golden/`, `sis-euclidean-estimator.md`);
- the planner proof-size model and drift guards (`akita-planner`, `akita-config`);
- the folded-witness L2 calibration instrumentation already in the cutover worktree (`l2-msis-opnorm-folded-witness.md` lines 289-525), extended to log `‖s‖₂` of the *projected* object for the modular-JL feasibility check.

### Performance

This is the decision being made, not an outcome to assert.
Expected directions to verify:

- Branch B trades ≈ 3.35× certified-norm slack (worse SIS rank) for a simpler protocol with no fallback and a stronger per-round witness-reduction lever (nested projection).
- Branch A keeps ≈ 1× slack (better rank where it certifies) but pays the carry/no-wrap machinery and degrades to a loose bound on the largest levels.
- The nested projection (Grand Danois Lemma 3.2) is the one lever that could change the *round count*. **⚠️ Correction (2026-06-13): round count does NOT dominate total proof size** — ground truth ([`resolutions §3`](akita-jl-norm-check-resolutions.md), planner tables) is that the **terminal cleartext witness is ≈82 KB = 60–80% of the proof and is fixed-point-bound, not round-count-bound** (each intermediate round is only ≈6.4 KB). Faster contraction reaches the same stopping-witness in fewer rounds, saving ≈6.4 KB/round eliminated but leaving the (current fixed-width) ≈82 KB terminal largely untouched. So nested projection saves `(rounds cut)×6.4 KB`, not a dominant fraction. The terminal itself is shrunk by [`tail-wire-encoding.md`](tail-wire-encoding.md) (r-drop PR #141 = 5.25–6.15%, t-reveal, `z` entropy coding `bound→σ`, `ê` width recovery) — branch-independent, and where the real tail bytes are recovered.

Profile/measure with the existing harness:

```bash
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile
```

## Design

### Where shortness is proved (the structural difference)

Branch A bolts the norm proof *alongside* the ring relation: the squared-sum sumcheck and carry claim are extra claims batched into stage 2, and the certified object is reconstructed from the committed digit planes.

Branch B folds the norm proof *into* the ring relation.
Grand Danois commits the projection image `p̂` together with the opening witness `ŵ` in one commitment (`D1` for `p̂`, `D2` for `ŵ`, paper lines 435-436), and adds projection-consistency rows `ĉ^{(i)} ⊗ (c⊤G)  …  −ĉ^{(i)} Ĵ Ĝ` directly to the constraint matrix `M` (eq. 3 / eq. 8, paper lines 358-429, 960-1032).
The same matrix-vector sumcheck akita already runs (`Mz = y + (X^d+1)r`, the Hachi identity, paper lines 1037-1048) then carries the norm proof, and shortness reduces to one verifier check `‖p_i‖₂ ≤ ~337·ω` (Theorem 3.3, paper lines 1200-1205).

This is the core appeal of Branch B: it **collapses akita's separate certificate into the relation**, eliminating the four-square solver, the carry chain, the no-wrap gate, and the large-level fallback.

### Slack and soundness basis

Branch A: exact integer equality `Σ z[i]^2 + Σ_h ell_h^2 = B_l2` proved by sumcheck under the no-wrap gate, so the verifier learns `Σ z[i]^2 ≤ B_l2` with slack ≈ 1 wherever a group size passes the gate, and the loose deterministic bound otherwise.

Branch B: modular JL.
For the nested projection `Π = (I⊗J')(I⊗J)` with independent `J, J' ∈ {−1,0,1}^{256×256j}` (Lemma 3.2, paper lines 906-927), the correct ratio is `‖Π̂'Π̂ w mod q‖₂ / ‖w‖₂ ∈ [30, 337]` (the paper prints it inverted) except with probability `2^{−128}(n/m + 256·n/m²)` (the paper's second term drops the 256 — `Π̂'` must be `I_{256n/m²} ⊗ Π'` to type-check).
A consistent verifier check is `‖p‖₂ ≤ 337·ω` (nested-threshold; needed for completeness since `E‖p‖₂² = 128²·‖w‖₂²` nested), and the extractor then concludes `‖weak opening‖₂ ≤ (337/30)·ω ≈ 11.23·ω` per round — **not** the paper's `√(337/30)·ω ≈ 3.35·ω`, which is the single-stage figure (Thm 3.3 checks the nested threshold and states the single-stage conclusion in the same paragraph). Single-stage projection: window `[√30, √337]`, threshold `√337·ω`, slack `3.35` — at 256× (not 256²×) compression per shot.
Both branches are adversarially sound (CWSS) *given per-level image enforcement*; the difference is exact-vs-flat-JL slack and the failure probability of the JL union bound (which at large levels needs `n_J > 256` or a documented gap — protocol spec §5(e)).

### The nested-projection reduction lever (the genuinely new bit)

Grand Danois's contribution (ii) (paper lines 22-24, 239-242, Lemma 3.2) is the part that is new over RoK-and-Roll and RoKoko: a *nested* projection with two small independent matrices gives ≈ `256^2` compression per round instead of `256`, i.e. more aggressive per-round witness reduction without a larger projection the verifier must manipulate.
Their estimate is 32-64× shrink per round, 5-6 rounds for `2^32`, ≈ 80-90 KB total (paper lines 1324-1330) — a 20-line back-of-envelope with no implementation; treat as aspiration, not data.
The trade the paper obscures (errata below): nesting buys the squared compression at the *squared* slack — `337/30 ≈ 11.23`/round nested vs `√(337/30) ≈ 3.35`/round single-stage — so the choice is per-level, not free.
This directly serves the "higher decomposition per level → fewer rounds → smaller tail" goal.
It is adoptable even as an enhancement to Branch A's recursion shape (it reduces what must be re-decomposed each round, independent of how shortness is certified). **But it is not an "outright win": round count does not dominate (the ≈82 KB terminal fixed point does, [`resolutions §3`](akita-jl-norm-check-resolutions.md)), so the saving is `(rounds cut)×≈6.4 KB`, modest.**

### Small-field modular-JL precondition (a block-size limit, not a feasibility wall)

An earlier draft flagged the modular-JL precondition as the decisive feasibility risk. On closer analysis with akita's real primes it is **not** a wall — it is a maximum projection-block size that the structured `I_{n/m}⊗J` projection satisfies by using enough blocks.

Exact constants. fp32 `q = 2^32 − 99 = 4.295×10^9`, fp31 `q = 2^31 − 19 = 2.147×10^9` (the "q ≈ 2^31" shorthand undersells fp32 by 2×). With `√337 = 18.358`:

| | single (`q/125`) | nested (`q/125√337`) |
|---|---|---|
| fp31 | `1.718×10^7` | `9.356×10^5` |
| fp32 | `3.436×10^7` | `1.872×10^6` |

The precondition bounds each projected block, on the committed short digits (balanced base-`b`, per-coordinate RMS `b/√12`), so `‖block‖₂ = √m · b/√12` for an `m`-coefficient block, and the **max projectable block size** is

```text
m_max = 12 · Bound² / b² .
```

`log₂ m_max` (RMS digits; worst-case digits subtract ≈ 1.6):

| lb (b) | nested fp31 | nested fp32 | single fp31 | single fp32 |
|---|---|---|---|---|
| 2 (4)  | 2^39.2 | 2^41.2 | 2^47.6 | 2^49.6 |
| 3 (8)  | 2^37.3 | 2^39.3 | 2^45.7 | 2^47.7 |
| 5 (32) | 2^33.3 | 2^35.3 | 2^41.7 | 2^43.7 |
| 6 (64) | 2^31.3 | 2^33.3 | 2^39.6 | 2^41.6 |

The akita root committed witness is `L·δ_commit = 2^nv·⌈field_bits/lb⌉` digit-coefficients (nv=32: `2^36` at `lb=2`, `2^34.6` at `lb=6`). Verdict:

- At the basis the DP actually uses (`lb 2-3` on the big levels), the **entire** root witness fits in **one** nested-projection block on both fields (`2^36 < 2^39.2`). The precondition is satisfied outright.
- Even at aggressive `lb=6` it only forces splitting: nested fp31 caps a block at `2^31.3`, so a `2^34.6` root splits into `≈ 10` blocks (the normal `I_{n/m}⊗J` mode; failure prob `2^{-128}·(n/m) ≈ 2^{-124.7}`); fp32 needs ≈ 2; **single projection fits the whole root in one block** (≈ 18× more headroom than nested).

So the precondition does not gate Branch B on small fields. Its only real consequence: on small fields at large basis the *nested* projection (256² compression, Lemma 3.2) must split into ~10 blocks, which inflates its projection output ~10× and erodes part of the nested compression advantage — there the *single* projection (256×, RoK-and-Roll style) is the comfortable choice. This does not affect fp128.

### SIS rank scaling and the decomposition-basis optimum

This is the question that actually decides the basis, independent of how shortness is proved. The planner's recursive-witness shape (`akita-types/src/layout/proof_size.rs`, `planned_w_ring_element_count`) is the sum of these ring-element counts:

```text
e_hat     = num_blocks · δ_open                       (ŵ opening digits)
t_hat     = num_blocks · n_a · δ_open                 (inner commitment t̂; n_a = a_key.row_len = A-matrix module rank)
z_pre     = inner_width · δ_fold                      (folded witness ẑ)
r_count   = m_row_count · ⌈field_bits/lb⌉             (ring-relation residual tail)
u_concat  = tiered second-tier digits (0 if single-tier)
(+ zk blinding columns under the `zk` feature)
```

The final tail is dominated by the two large terms `t_hat` and `z_pre` (the rest is ≤ 10-20%), so write `|t̂| = num_blocks·n_a·δ_open` and `|ẑ| = inner_width·δ_fold`. The parameters are chosen so these two roughly balance. Both depend on the decomposition basis `b = 2^lb`, so the basis that minimizes `|t̂| + |ẑ|` is what we want.

**The SIS security model (what sets `n_a`).** The Ajtai matrix `A` committing `W` ring columns at module rank `n`, ring degree `d`, modulus `q` is an `MSIS_{q,n,W,β}` instance with Euclidean collision bound `β = √(collision_l2_sq)`. The generated SIS-floor table (`akita-types/src/sis/generated_sis_table/`, produced by the pinned lattice-estimator for 128-bit security, BDGL16) encodes an exact law: for fixed `(family, d, rank)` the supported width is inversely proportional to the squared collision, and the per-rank capacity grows log-linearly in `√rank`:

```text
W · β²  ≤  A(rank),        A(rank) = e^{ c · √rank } .
```

Both facts are read directly off the table. Width ∝ 1/β²: e.g. (d=32) `593·2 = 296·4 = 148·8 = 1184`. Capacity exponent linear in `√rank`: (d=128, q32) `A = W·β² = {1.50×10^6, 5.58×10^8, 5.30×10^10}` at ranks `{1,2,3}`, so `ln A = {14.22, 20.14, 24.69} = 14.23·√rank`. So

```text
c ≈ 14.23   (d=128, q = 2^32),     c ≈ 7.1 (d=32, q=2^32),     i.e.  c ≈ 1.256·√d   for q32
```

and `c` scales as `√(d · log q)` across families (extract per `(family,d)` from the table). Inverting the law gives the closed-form module rank:

```text
n_a(W, β²)  ≈  ( ln(W · β²) / c )²            (q32: = ln(W·β²)² / (1.58·d) )
```

The rank is **quadratic in the log** of (width × squared-collision), and falls with `d` and `log q`. This is the precise "module rank depends on modulus, ring dim, and norm bound" dependency.

**How the basis enters each factor.**

```text
β²  = collision_l2_sq ∝ b²          (committed digits ≤ b/2 ⇒ realized ‖·‖ ∝ b; loose β_inf or JL-realized, the b-slope is the same)
W   = block_len · δ_commit ∝ 1/lb   (δ_commit = ⌈field_bits/lb⌉)
δ_open = ⌈field_bits / lb⌉ ∝ 1/lb
δ_fold = ⌈log_b(2·β_z)⌉ ≈ 1 + G/lb,   G = r_vars + log₂ω   (ω = challenge ℓ1 mass; the "ring-challenge blow-up")
```

So `ln(W·β²) = 2ln2·lb − ln(lb) + C_W`, with `C_W` the basis-independent part `ln(block_len·field_bits·d·κ)` (κ folds the `8·ω·…·ν` collision constants). `|ẑ| = inner_width·(1 + G/lb)` shrinks monotonically toward `inner_width`. `|t̂| = num_blocks·n_a·δ_open` has an **interior minimum**: because `n_a ∝ (ln β)² ∝ lb²` grows while `δ_open ∝ 1/lb` shrinks, the product behaves as `lb²·(1/lb) = lb` for large `lb` (rises) but as `C_W²/lb` for small `lb` (falls).

**The optimum.** Setting `d/dlb (|t̂| + |ẑ|) = 0`. The `|t̂|`-only optimum is where the marginal rank growth equals the marginal digit saving, `d ln(n_a)/dlb = 1/lb`, i.e. `2·L'/L = 1/lb` with `L = ln(W·β²)`. This solves to the condition that the collision term and the baseline contribute *equally* to `ln(W·β²)`:

```text
2 ln2 · lb*  ≈  C_W            ⟺            lb*  ≈  ¼ · log₂(W · β²)|_opt .
```

Adding the `|ẑ|` term pulls `lb*` up by an amount set by the t̂/ẑ balance:

```text
lb*  ≈  (1 / 2ln2) · √( C_W²  +  c² · G · inner_width / (num_blocks · field_bits) ) .
```

**Numeric verdict for fp32 (`d=128`, `c=14.23`).** Calibrating `C_W` against a real recursive level (nv=20, `block_len=256`, `r_vars=3`, `n_a=8` from the shipped table implies `ω ≈ 33`) gives `W·β² ∼ e^40 … e^66` across levels, hence

```text
lb*  ≈  ¼ · log₂(W·β²)  ≈  14 … 24 ,
```

far above the current search cap of `lb ≤ 6`. In other words **the SIS tail wants a basis 3-4× larger than today's cap**; the cap exists because the `‖z‖∞` range-check / squared-sum sumcheck cost grows as `2^lb`, not because the SIS tail prefers small bases. This is the analytical confirmation of the design intuition: the basis is throttled by the norm-check, and the true SIS optimum is much higher.

The genuine ceilings on `lb*` (once the sumcheck cost is removed) are:
- **Digit saturation.** `δ_open = ⌈field_bits/lb⌉` and `δ_fold → 1` bottom out; past `lb ≈ field_bits/2 … field_bits/3` (≈ 11-16 for fp32, larger for fp128) the basis only inflates `n_a` with no digit saving, so the true min sits where `δ_open` reaches 2-3 — i.e. `lb*` in the table above is in practice clamped to this knee.
- **The modular-JL block precondition** (previous subsection): on the biggest small-field levels the nested projection caps the per-shot block, nudging `lb` back down or onto the single projection.

**Decision consequence — CORRECTED (the proof-size half is retracted).** It remains true that Branch A's `‖z‖∞` range check is degree `2^lb` and its exact-ℓ2 no-wrap gate `D_e ∝ N·(b/2)²·R` tightens with `b` (and fails at nv=32 roots), while Branch B's JL check is basis-independent in *degree*. **But this does not buy a proof-size win**, because (DP-measured, [`resolutions §3`](akita-jl-norm-check-resolutions.md)) the proof bytes are basis-flat: lifting the basis shrinks `δ_open/δ_fold` *element* counts but the terminal cleartext packs at `lb` bits, so `δ·lb` is magnitude-locked, and `n_a(lb)` *grows*. The DP keeps `lb≤5` at large nv even with the cap lifted to 16. So "adopting JL unlocks the SIS-optimal basis which shrinks the tail" is **false in bytes** — the tail is not basis-shrinkable. JL's win is the *element-count / SIS-rank* reduction (which helps **prover time and recursion depth**, separate objectives to measure on their own), plus the anchored A-role rank drop and the no-fallback simplicity — **not** the proof byte count via basis.

### What the shipped DP picks today (empirical, for calibration)

The proof-optimized search range is `lb ∈ [2, 6]` (`PROOF_OPTIMIZED_LOG_BASIS_{MIN,MAX} = 2,6` in `crates/akita-config/src/proof_optimized.rs`). Given that freedom, the shipped tables (`crates/akita-planner/src/generated/fp32_d128_onehot.rs`) show:

- `log_basis` is **not** uniform. It is `2` on the large near-root levels and rises to `5` on the small recursive levels (nv=20 is `lb=5` throughout; nv=28/30/31 roots are `lb=2`; nv=32 root is `lb=3`). **Caveat (2026-06-13):** the shipped fp32 nv=31/32 rows are degenerate fallback stubs (nv=32: single root fold + 1.2 GB Direct terminal — the "uncommittable edge" path); empirical claims here are grounded on nv ≤ 30 (7-11 folds, 120-139 KB total, terminal cleartext ≈82 KB ≈ 60-80% of the proof).
- It **never** picks `6`, the max — consistent with the search being capped by sumcheck cost (modelled in the planner's proof-byte term) at a knee around `lb=5`, not by an SIS-tail preference for small bases.

Two facts make today's picture *not* a measurement of the SIS optimum:

1. **Pricing is loose.** With the realized certificate (S8-S10) unshipped, the A-role is priced at the deterministic `β_inf` worst case (`‖z‖∞ ≤ β_inf`, then `‖z‖₂ ≤ √d·β_inf`). The calibration tables (`l2-msis-opnorm-folded-witness.md` lines 347-425) show this is **30-200× over the realized `‖z‖₂`**. Loose pricing penalizes large `b` twice — `β_inf ∝ b` inflates both the A-role collision *and* `num_digits_fold` — so it biases the DP toward small bases. A realized-norm certificate (exact-ℓ2 *or* JL) removes most of that penalty and shifts the optimum up.
2. **The cap blocks the optimum.** `lb ≤ 6` is below the SIS-tail optimum (`≈ 11-16`) anyway, so even with realized pricing the current DP cannot express it.

So the empirical `lb = 2-5` is the loose-priced, sumcheck-capped optimum; the analytical `lb ≈ 11-16` is the realized-priced, uncapped SIS-tail optimum. The "basis sweep" decision-criterion item is exactly the experiment that turns the latter from a calibrated estimate into a measured number.

### Witness / proof-size sketch (to be made concrete by the planner)

- Branch A per certifying level: `+ ell_hat` (4 limbs) `+ carry_hat` (~`2R` carries) `+` masked `V` `+` one degree-2 sumcheck transcript `+` short carry linear claim.
- Branch B per projected level: `+ p̂` (~256 ring elements, recursively folded) `+` the projection-consistency rows inside the existing relation sumcheck (no new sumcheck instance) `+` one scalar verifier norm check.
- Net: Branch A has smaller per-level witness but a heavier, fallback-prone protocol; Branch B has larger per-level witness but a simpler protocol and a stronger round-count lever.

### Adapting Grand Danois without vSIS

Grand Danois uses vSIS only to make the *verifier* evaluate the (now structured) matrix MLE succinctly.
The JL norm-check construction itself (commit `p̂` with `ŵ`, fold projection rows into `M`, verifier checks `‖p‖₂`) does not require vSIS; it requires only that the verifier can evaluate the relevant rows of `M`, which akita's existing sumcheck verifier already does for uniform-MSIS matrices.
Under uniform MSIS, akita keeps its current per-row evaluation cost and simply gains the extra projection rows; it does not get Grand Danois's polylog verifier, which is acceptable because akita already has a working sumcheck verifier and we are explicitly not chasing the vSIS speedup.
The structured projection (`(I⊗J)` / nested `(I⊗J')(I⊗J)`) is what keeps the *projection rows* verifier-evaluable; that structure is independent of vSIS and is the only structure Branch B needs.

### The `c′` quotient row-compression: NOT a separable win — unsound as printed (confirmed 2026-06-13)

An earlier draft of this doc flagged GD's `c'⊤ M z = c'⊤ y` compression (eq. 11-12, paper lines 1055-1068) — collapse the matrix rows with a random combiner `c′` so the committed ring-switch quotient `r̂` shrinks from one polynomial per row to a single scalar polynomial — as a "separable low-risk win". **It is not. The optimization is a confirmed soundness bug in GD as printed.**

The protocol structure that would license `c′` row batching is Freivalds/Schwartz-Zippel: every object entering the batched relation must be **bound before `c′` is sampled**. GD's Fig. 1 (p. 20) does the opposite: the verifier sends `(c, c′, ĉ)` in one message, and *then* the prover commits `cm′` containing `ẑ` and the `c′`-dependent batched quotient `r̂` — with the `cm′`↔`v`/`cm` consistency itself enforced only *inside* the `c′`-batched single row. After seeing `c′`, a cheating prover needs only one short solution to one `R_q`-equation; in the CWSS extraction, branches over `c′` produce *different* commitments `cm′`, so the SZ subtraction argument collapses. The Thm 3.3 proof sketch (p. 19) embeds the flaw: its "alternative protocol" has the prover send the witness in plaintext *after* receiving `c, c′, ĉ`.

The sound orderings, and why neither is a free win for akita:

1. **Batch after commitment, quotient in the clear:** commit `w_next` without the batched quotient, sample the batcher (akita's `τ₁` is already sampled post-`u′` — akita's shipped ordering is the sound one), then send the batched quotient in plain. Akita's eq-weights live in the extension field `L`, so the clear quotient is an `L`-valued ring element ≈2 KB/level, vs `r̂` ≈ 90 rings ≈ 7-8% of a late-level witness: **a net loss at committed levels**. The narrow salvage is the terminal level only, where the verifier holds the cleartext witness and can recompute the quotient itself (elide `r̂`, ≈6-7 KB — protocol spec §8.10).
2. **Per-row quotients committed pre-batch:** exactly what akita already does (`r̂` = `m_row_count × ⌈field_bits/lb⌉` ring elements inside `w_next`, batched afterwards by `τ₁` in the sumcheck). No change, no saving.

Takeaway: akita already row-batches via `τ₁` at the sumcheck layer *and* commits the per-row quotient witness pre-batch — those are different layers, and GD's attempt to merge them (commit the post-batch quotient) is exactly the broken interaction order. The residual idea worth keeping is the terminal `r̂` elision above.

### GD errata ledger (verified against the PDF, 2026-06-13)

All confirmed by independent re-derivation; page numbers from eprint 2026/1196 (23 pp., 2026-06-08). **Use GD for its ideas (structured projection in-relation, nested lever, §3.2.4 finisher sketch); re-derive every constant and every interaction order; cite nothing numerically.**

| # | Item | Where | Status |
|---|---|---|---|
| 1 | `c′`/`ĉ` sampled before `cm′`; `ẑ`, batched `r̂`, and the `cm′`↔`v`/`cm` link all post-challenge | Fig. 1 p.20, §3.2.1 p.17 | **soundness bug** (previous subsection) |
| 2 | Thm 3.3 "alternative protocol" sends witness in plaintext *after* `c, c′, ĉ` | p.19 | proof embeds bug #1 |
| 3 | Ratio direction inverted in Lemmas 3.1/3.2 (`‖w‖/‖Πw‖` should be `‖Πw mod q‖/‖w‖`) | p.15 eqs. 6-7 | typo, magnitudes right |
| 4 | `Π̂′ = I_{n/m²}⊗Π′` doesn't type-check; must be `I_{256n/m²}⊗Π′`; failure prob second term `n/m²` → `256n/m²`; both-tails factor 2 also missing | p.15 | dimension bug |
| 5 | Headline slack `√(337/30) ≈ 3.35` is single-stage; nested protocol gives `337/30 ≈ 11.23`; Fig. 1's `√(337/30)·ω` bound on `p_i` fails completeness (~38× below honest) | p.19 vs p.20 | internal inconsistency |
| 6 | `p` never revealed; image norm deferred down recursion; no compounding analysis (no LaBRADOR-Rem-5.2 analogue); at 11.23×/level the `q/(125√337)` precondition budget dies in ~2 levels at q=2^32 | Fig. 1, Thm 3.3 | design gap, fatal for small q |
| 7 | Thm 3.3 failure prob `d·2^{−128}(Lȷ²+L)/ȷ⁴` substitutes `m = ȷ²` where the protocol has `m = 256ȷ` | p.19 | symbol-pushing error |
| 8 | ĉ-fold via geometric powers over N rows costs `(N−1)/q` per fold, not `1/q`; `k = 4` gives ≈2^−80, not 2^−128; need `k ≈ 7` | footnote 4 p.16 | soundness undercount |
| 9 | Challenge set `C` never defined (size, difference-invertibility, ring-SZ sampling-set property all load-bearing) | p.3, p.17 | unspecified |
| 10 | Relation `R` puts the norm slot on the canonical decomposition `ŝᵢ = G⁻¹(sᵢ)` — automatically short for *any* prover; the meaningful bound is on `sᵢ` | §3.2.3 p.19 | near-vacuous condition |
| 11 | "Binding … under the SIS assumption" — CRS is the vSIS tensor distribution; plain MSIS does not apply | p.19 vs Def 2.2 p.11 | misstated assumption |
| 12 | `κ`, `b′` undefined; `H`-target writes `t` for `cm`; PoK tuple omits `ẑ`; FS treatment is one sentence, no grinding analysis of the statistical JL failure | Fig. 1, §3.3 | gaps |

What survives for akita (vSIS-free): Lemmas 2.5/3.1/3.2 after corrections #3/#4 (the modular-JL window itself is LaBRADOR's, heuristic-flagged there too), the structured `I⊗J` projection shape, the ĉ row-fold idea (with #8's corrected soundness), and the §3.2.4 final-layer sketch (20 lines, no theorem — reveal-vs-commit of the final projection unstated; akita's sharpened Slot 3 in the protocol spec is the worked-out version).

### How the prior works certify a committed image (verified; the answer to "how do they avoid wrap?")

The wrap question for a committed image is harsher than for the witness itself, since `E‖p‖₂² = (n_J/2)·‖w‖₂²` — the image *concentrates* mass. Verified per-paper:

- **LaBRADOR**: never commits `p` — reveals it each level, checks `‖p‖₂ ≤ √128·β` on balanced representatives over Z (retry-until-median), slack 2.07. Wrap lives only inside the modular-JL Lemma 4.2 (`b ≤ q/125`, engineered for q ≈ 2^32). Remark 5.2's compounding is into the *assumed MSIS bound* (2.07/level), not the witness norms.
- **RoKoko** (committed image): gadget-decomposes the image *before* committing (`Y = G⁻¹(V)`); the *next* round's exact `trace⟨ŵ, ŵ̄⟩` sumcheck certifies the packed digit vector; needs `r′·β̃²·f̂ < q/2` — a stated driver of their **q ≈ 2^50**. On fp32 the same inequality caps certified column norms at ≈2^11: dead.
- **SALSAA**: `Π^norm` checkpoint each round (`t = ⟨wᵢ, w̄ᵢ⟩` in clear, `Trace(t) ≤ ν²`), needs `β′² < q/2`, also **q ≈ 2^50**. Same wall.
- **RoK-and-Roll**: image is a separate strict-norm relation branch whose fold is *delayed* so it extracts slack-free; modular-JL variant (Lemma 4/5) needs **no `q/125`-type precondition at all** (lower constant `1/(2√m)` instead of `√30` — a √m loss, fine at small image dims). Footnote 21 is the canonical "exact norm check is completely insecure under slack" counterexample.
- **Hachi** (the ancestor): no JL; digit-set membership via vanishing polynomial is *exact algebra mod q* — wrap-free at any q. This is the mechanism that survives on fp32 as the committed-image enforcer (micro-range on `p̂` digits), with the carry-lifted exact-ℓ2 as the tight upgrade (protocol spec §4 menu).

Consequence for the branch decision: Branch B on fp32 is viable **only** in the akita-adapted form (per-level enforcement via revealed image or decomposed-digit certification) — neither GD-as-printed (no enforcement) nor RoKoko/SALSAA-as-printed (q ≈ 2^50 exact check) transplants directly.

### Architecture (surfaces a Branch B cutover would touch)

If Branch B is chosen, the affected surfaces largely overlap the L2 cutover's, minus the carry/four-square machinery, plus a projection:

- `akita-challenges`: sample structured `J, J'`, transcript-stable, bound in the descriptor.
- `akita-prover` ring relation / ring switch: compute `p = Ĵ·s`, decompose to `p̂`, commit `p̂` with `ŵ`, emit projection-consistency rows into `M`.
- `akita-verifier`: evaluate the projection rows in the relation sumcheck, check `‖p‖₂ ≤ ~337·ω`, no-panic validate projection shapes.
- `akita-types::sis` / `akita-planner` / `akita-config`: A-role rank under the `√(337/30)` slack; no four-square, no carry, no no-wrap gate.
- transcript: bind the JL branch, projection structure, and norm threshold.

### Alternatives Considered

- **Keep Branch A as the only path.**
  Viable; it is the current direction and is exact where it certifies.
  Its weakness is the large-level no-wrap fallback and the carry/four-square complexity.
- **Adopt vSIS to get Grand Danois's full verifier-succinct scheme.**
  Rejected by constraint: vSIS is a new assumption we do not adopt.
- **Run both branches behind a flag.**
  Rejected: full-cutover repo policy, and dual norm models invite drift across security model, planner, prover, and verifier.
- **Take only the nested-projection lever (Lemma 3.2) as a witness-reduction enhancement, independent of the shortness branch.**
  Worth keeping live: the round-count lever may be the biggest win and is partially separable from the exact-vs-JL choice.

## Documentation

If a branch is chosen, update: the public security model docs (`book/src/how/security/*`), the norm-bounds page (`book/src/how/security/norm-bounds-weak-binding.md`), the L2 roadmap (`book/src/how/security/l2-msis-roadmap.md`), and either retire or rewrite the certificate sections of [`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md).
The paper note for eprint 2026/1196 is already indexed in the shared paper library.

## References

- Grand Danois, eprint 2026/1196 (`/Users/quang.dao/Documents/Papers/eprint-2026-1196.pdf`): structured-JL-in-relation, nested projection (Lemma 3.2, line 906), slack `√(337/30)` (Thm 3.3, line 1187 — **wrong as printed**, see errata ledger; nested is 337/30), modular-JL precondition (line 915), `c'` compression (lines 1055-1068 — **unsound as printed**, see the `c′` section), vSIS (Def 2.2, lines 651-675; explicitly not adopted). Do not cite GD constants without the errata ledger.
- Hachi, eprint 2026/156 (akita's lineage; the `Mz = y + (X^d+1)r` sumcheck identity).
- RoK and Roll, eprint 2025/1220 (structured projection); RoKoko, eprint 2026/575 (committed structured projection, sumcheck-exact ℓ2); LaBRADOR, eprint 2023/... (modular JL, Lemma 2.5 constants).
- [`l2-msis-opnorm-folded-witness.md`](l2-msis-opnorm-folded-witness.md) — current exact `∞+ℓ2` certificate cutover (Branch A).
- [`weak-binding-norm-fix.md`](weak-binding-norm-fix.md) — Lemma 7 weak-binding object (the shared security object).
- [`sis-euclidean-estimator.md`](sis-euclidean-estimator.md) — L2 estimator and table regen used for the rank-delta measurement.
- SIS scaling sources: `crates/akita-types/src/sis/generated_sis_table/` (the `W·β² = e^{c√rank}` law), `crates/akita-types/src/sis/ajtai_key.rs` (`min_secure_rank`, collision bucketing), `crates/akita-types/src/sis/norm_bound.rs` (collision derivation `8·ω·β_inf·ν`), `crates/akita-types/src/layout/proof_size.rs` (`planned_w_ring_element_count`, the `t̂`/`ẑ` shape), `crates/akita-config/src/proof_optimized.rs` (`PROOF_OPTIMIZED_LOG_BASIS_{MIN,MAX} = 2,6`).
- [`bounded-l1-sparse-challenge.md`](bounded-l1-sparse-challenge.md), [`tensor-structured-folding-challenges.md`](tensor-structured-folding-challenges.md) — challenge families feeding `ω`.
- Prior context: [JL approximate norm landscape](910e2d2a-7f6c-4cfd-8d08-02281972f52c), [RoKoko scheme and folding](5fc632a5-e57d-4c8f-813d-e62fc4bd7543).

## Appendix: quick-reference numbers (for future readers)

**Primes.** fp32 `q = 2^32 − 99 = 4.295×10^9`; fp31 `q = 2^31 − 19 = 2.147×10^9`; (the "q ≈ 2^31" shorthand in older docs undersells fp32 by 2×). fp32 shipped ring degree `d = 128`.

**SIS floor law** (`generated_sis_table/`, lattice-estimator, 128-bit, BDGL16, Euclidean):
```text
secure  ⟺  W · β²  ≤  A(rank) = e^{c·√rank}          β² = collision_l2_sq, W = ring-column width
n_a  ≈  ( ln(W·β²) / c )²                            (module rank, closed form)
c  ≈  1.256·√d   (q = 2^32)  ⇒  c ≈ 7.1 (d=32), 14.23 (d=128);   c ∝ √(d·log q) across families
```
Empirical anchors: width ∝ 1/β² exact (d=32: `593·2 = 296·4 = 148·8 = 1184`); `ln A = 14.23·√rank` (d=128/q32: `A = 1.50×10^6, 5.58×10^8, 5.30×10^10` at rank 1,2,3).

**Collision (A-role / fold response).** `collision_linf = 8·ω·β_inf·ν`, `β_inf = num_claims·2^r_vars·min(‖c‖∞‖s‖₁, ‖c‖₁‖s‖∞)`, dense `‖s‖∞ = b/2` ⇒ `β_inf ∝ b` ⇒ `β² = collision_l2_sq ∝ b²`. (`crates/akita-types/src/sis/norm_bound.rs`.)

**Digit counts.** `δ_open = δ_commit = ⌈field_bits/lb⌉`; `δ_fold ≈ 1 + (r_vars + log₂ω)/lb`. All ∝ 1/lb.

**Tail terms.** `|t̂| = num_blocks·n_a·δ_open`, `|ẑ| = inner_width·δ_fold`.

**Basis optimum.**
```text
lb*  ≈  ¼·log₂(W·β²)|_opt   (t̂-only)
lb*  ≈  (1/2ln2)·√( C_W² + c²·G·inner_width/(num_blocks·field_bits) )   (with ẑ; G = r_vars + log₂ω)
fp32 numeric:  lb* ≈ 14-24, clamped by digit saturation to ≈ 11-16.
```

**Basis caps (decreasing order of who binds first).**
| cap | value (fp32) | who/why | removed by JL? |
|---|---|---|---|
| sumcheck/range-check cost | `lb ≤ 6` (search), DP picks ≤ 5 | `‖z‖∞` check degree `2^lb` | yes (basis-independent JL) |
| exact-ℓ2 no-wrap gate | fails at nv=32 root for all `lb` | `D_e ∝ N·(b/2)²·R < q` | yes (no gate in JL) |
| SIS-tail optimum (ELEMENTS only) | `lb ≈ 11-16` | `n_a ∝ log²`, digits ∝ 1/lb | element/rank target; **NOT a byte win** — DP-flat, [`resolutions §3`](akita-jl-norm-check-resolutions.md) |
| digit saturation | `lb ≈ field_bits/2..3` | `δ_open, δ_fold → 1` | no (fundamental) |
| modular-JL nested precondition | block `m_max = 12·Bound²/b²` | per-block `‖·‖₂ ≤ q/(125√337)` | n/a (JL-internal; use single proj.) |

**Modular-JL precondition constants.** single `q/125`: fp31 `1.718×10^7`, fp32 `3.436×10^7`. nested `q/(125√337)`: fp31 `9.356×10^5`, fp32 `1.872×10^6`. Not a feasibility wall — sets max projection-block size; root witness fits in 1 block at `lb≤3`, ≈10 nested blocks at `lb=6` (fp31).
