# Making Hachi + Jolt Zero-Knowledge

This note describes how to make the combined **Jolt + Hachi** proof system zero-knowledge end-to-end with minimal blowup on proof size.
**Jolt** is the BN254 + Dory-PCS zkVM publicly available at [https://github.com/a16z/jolt](https://github.com/a16z/jolt) (Arun, Setty, Thaler; ePrint 2023/1217).
**Hachi** is a lattice-based multilinear polynomial commitment scheme with transparent setup and post-quantum security, introduced by Nguyen, O'Rourke, and Zhang (ePrint 2026/156).
This note works with a variant of Hachi that extends the published construction with several new ideas and optimizations under active development in this repository (deeper recursion, ring-switching inside the sumcheck, mega-polynomial layout for joint witness commitments, etc.); we do not spell those out here.
Replacing Jolt's pairing-based PCS with Hachi yields a lattice-native zkVM; the focus of this note is how to make that combined proof zero-knowledge.

Doing this is harder than it looks.
The proof spans **Jolt's 333 sumcheck rounds** (lookups, instruction execution, register/memory checking) and **Hachi's 200+ recursion sumcheck rounds**, with several Ajtai commitments in between.
Naïvely masking each round with a fresh lattice commitment would dominate proof size at 512 B per round.
Even masking only the final tail is intricate: the tail must simultaneously prove a bundle of *committed linear relations* (Ajtai openings, evaluation claims, batched sumcheck residuals, `V_ring` and `y_ring` bindings) **and** *committed quadratic relations* (the residual `Az · Bz = Cz` from Spartan, the residual `y_ring` trace check, plus the round-eval closure for any sumcheck round whose virtual polynomial is non-linear), all in zero-knowledge.
Each relation interacts with the sigma protocol's Gaussian masking and rejection sampling in subtle ways, and the linear/quadratic split drives nontrivial design choices throughout (e.g., LNP22's single-quadratic add-on at the tail; the batched-residual invariant for sumcheck pads).

Three high-level ideas carry the design (detailed in §1.0): one big pre-commitment `Com_pre` holding every sumcheck pad; Spartan fused into Hachi at the next-to-last folding level `L − 1`; and one joint Gaussian sigma protocol at the tail closing all linear residuals plus a small number of LNP22 single-quadratic add-ons (one per residual quadratic; the explicitly accounted-for one is Spartan's post-outer `Az · Bz = Cz`, §5.5; full audit pending in §11.12).
Together these keep the ZK proof within ±2 KB of the non-ZK proof (40 KB), plus 1–2 KB of commitment overhead.

**Scope of this note.**
The goal is an engineering-level spec for end-to-end ZK on Jolt + Hachi: the commit schedule, sumcheck pad layout, masking invariants, fused-Spartan placement, and joint tail sigma are pinned down to the level of precision needed to implement a prover and verifier, with concrete cost deltas in §9 and parameter sketches in §10.
It is *not* a formal soundness / ZK reduction (§11.10), and it deliberately ignores modulus switching (§11.11), flagged below as the single most important extension.

**Major missing link: interaction with modulus switching.**
Hachi's later recursion levels admit a modulus-switching gadget that drops the working modulus from `q ≈ 2^{128}` down to a small `q_lo` (e.g., 32-bit).
At the low-modulus regime MLWE becomes cheap (4 ternary ring elements per sample for 128-bit security; cf. the Lantern parameters in Appendix A), and rejection-free hiding techniques like Hint-MLWE ([https://eprint.iacr.org/2025/2239](https://eprint.iacr.org/2025/2239)) become competitive with Gaussian rejection sampling.
The design in this note works entirely at `q ≈ 2^{128}` and ignores this lever.
Integrating modulus switching with the ZK design is the single most important extension and could meaningfully shrink the tail sigma response.
Left as an open problem (§11.11).

The overall approach is **Path C** (hybrid field + lattice ZK).
Appendix A explains why the two alternatives (separate masking commitment; ABDLOP redesign) were rejected.

> **Status and audit caveat.** This note is directionally right but has *not* been fully audited end-to-end. High-level ideas (pre-committed sumcheck pads, fused Spartan at Hachi level `L − 1`, joint tail sigma with LNP22 single-quadratic) are the intended architecture; specific parameter counts, round numbers, and some of the more intricate soundness / hiding arguments may contain errors or loose ends. Readers should treat the headline structure as authoritative and the fine details as "best-effort first draft" — please push back on anything that doesn't cleanly check out, and corrections of specific claims are welcome.

> **Reading tips for external readers.**
>
> - References of the form `src/...`, `jolt-core/...`, `docs/...` point into the `hachi-pcs` and `jolt-hachi` source repositories and are mainly useful as cross-checks for contributors; external readers can safely ignore them.
> - References to other notes (`hachi-zk-unified-legacy.md`, `hachi-blindfold-walkthrough-and-lattice-generalization.md`, etc.) are local working drafts and can be omitted on a first read.
> - Paper references below include full titles and public URLs; local `.pdf` filenames refer to the author's working-papers folder and are *not* required to follow the note.

**Paper references.** Listed once here; subsequent mentions use the short handle in bold.

- **Hachi** — Ngoc Khanh Nguyen, George O'Rourke, Jiapeng Zhang. "Hachi: Efficient Lattice-Based Multilinear Polynomial Commitments over Extension Fields." ePrint 2026/156. [https://eprint.iacr.org/2026/156](https://eprint.iacr.org/2026/156).
- **LNP22** — Vadim Lyubashevsky, Ngoc Khanh Nguyen, Maxime Plançon. "Lattice-Based Zero-Knowledge Proofs and Applications: Shorter, Simpler, and More General." In *Advances in Cryptology – CRYPTO 2022*. [https://eprint.iacr.org/2022/284](https://eprint.iacr.org/2022/284) (local: `LNP22.pdf`).
- **Lantern / Nguyen thesis** — Ngoc Khanh Nguyen. "Lattice-Based Zero-Knowledge Proofs Under a Few Dozen Kilobytes." PhD thesis, ETH Zurich, 2022. [https://hdl.handle.net/20.500.11850/574844](https://hdl.handle.net/20.500.11850/574844) (local: `Nguyen_Thesis_LatticeZK_2022.pdf`).
- **LatticeFold+** — Dan Boneh, Binyi Chen. "LatticeFold+: Faster, Simpler, Shorter Lattice-Based Folding for Succinct Proof Systems." In *Advances in Cryptology – CRYPTO 2025*. [https://eprint.iacr.org/2025/247](https://eprint.iacr.org/2025/247) (local: `LatticeFold+.pdf`).
- **Symphony** — Binyi Chen. "Symphony: Scalable SNARKs in the Random Oracle Model from Lattice-Based High-Arity Folding." ePrint 2025/1905, rev. February 2026. [https://eprint.iacr.org/2025/1905](https://eprint.iacr.org/2025/1905) (local: `Symphony.pdf`).
- **LaBRADOR-Falcon** — Marius A. Aardal, Diego F. Aranha, Katharina Boudgoust, Sebastian Kolby, Akira Takahashi. "Aggregating Falcon Signatures with LaBRADOR." In *Advances in Cryptology – CRYPTO 2024*. [https://eprint.iacr.org/2024/311](https://eprint.iacr.org/2024/311) (local: `Aggregating_Falcon_Signatures_with_LaBRADOR.pdf`).
- **Gaertner (iterative rejection sampling)** — Joel Gärtner. "Compact Lattice Signatures via Iterative Rejection Sampling." In *Advances in Cryptology – CRYPTO 2025*. [https://eprint.iacr.org/2024/2052](https://eprint.iacr.org/2024/2052) (local: `Gaertner_Iterative_Rejection_Sampling.pdf`).
- **Libra** — Tiancheng Xie, Jiaheng Zhang, Yupeng Zhang, Charalampos Papamanthou, Dawn Song. "Libra: Succinct Zero-Knowledge Proofs with Optimal Prover Computation." In *Advances in Cryptology – CRYPTO 2019*. [https://eprint.iacr.org/2019/317](https://eprint.iacr.org/2019/317) (local: `Libra.pdf`).
- **VEGA** — Darya Kaviani, Srinath Setty. "Vega: Low-Latency Zero-Knowledge Proofs over Existing Credentials." ePrint 2025/2094. [https://eprint.iacr.org/2025/2094](https://eprint.iacr.org/2025/2094) (local: `VEGA_Kaviani_Setty_2025.pdf`).

(Local-only: earlier drafts of this note archived at `hachi-zk-unified-legacy.md`, `hachi-zk-paths.md`, `hachi-zk-pcs-report.md`, `hachi-zk-and-modulus-embedding.md`; code under `hachi-pcs` and `jolt-hachi` repositories.)

---

## 1. Overview

### 1.0 Reader's map and key ideas

This note describes how a single prover turns a **non-ZK Jolt + Hachi proof** (sumchecks + lattice PCS, with large amounts of intermediate data visible on the transcript) into a **ZK proof** without changing the underlying sumcheck machinery. Three high-level ideas carry all the weight:

1. **Mask every sumcheck round with a committed pad (not a fresh per-round commitment).**

Lattice commitments are 512 B each, so committing a random pad for every one of the 500 sumcheck rounds across Jolt + Hachi + Spartan would dominate proof size. Instead we pre-commit **all** pads inside one big Ajtai commitment `Com_pre` before any challenge is drawn. Each round sends the masked coefficient `m' = m + ρ` in cleartext; the round-sum identity is **not checked at round time** but recorded as a public linear residual. All residuals are batched and discharged once, at the tail, via a single Gaussian sigma protocol (§4.1, §4.3).
2. **Spartan is fused with Hachi at the next-to-last folding level (`L − 1`).**
Jolt's verifier algebra is arithmetized as a small R1CS whose aux witness (3,000 F_q elements) would otherwise balloon the tail. Placing its commitment `Com_aux1` mid-recursion, merged into Hachi's mega-polynomial at level `L − 1`, lets Spartan's inner sumcheck *fuse* with Hachi's stage-2 (shared variable space, one RLC-combined sumcheck), and lets level `L`'s fold shrink `Com_aux1` by a factor `D` before it reaches the tail. Spartan's outer sumcheck is short (14 rounds) and the residual quadratic identity `Az · Bz = Cz` on three scalars is discharged by LNP22's single-quadratic add-on at the tail (§5); any other residual quadratics that surface from non-linear round-eval closures are handled the same way (§5.5, §11.12).
3. **The tail is closed by one joint Gaussian sigma protocol.**
Everything — Ajtai openings, evaluation claims, batched sumcheck residuals, `V_ring` binding, `y_ring` residual pins, and the single LNP22 quadratic — folds into one combined response. Linear functionals stack for free (one extra field element each in `e_y`); the single quadratic costs one pre-committed `g_quad` ring element plus two extra scalars (§6).

What this buys us: total ZK tail is within ±2 KB of the non-ZK tail (40 KB), plus 1–2 KB of commitment overhead for `Com_aux1` and a few hundred extra pad slots inside `Com_pre`. No MLWE or PRG-security assumption enters; all hiding is statistical via LHL (for commitments) and perfect via OTP (for sumcheck rounds).

The rest of §1 gives the formal four-phase structure (§1.1), audits what leaks today (§1.2), and collects the design choices (§1.3). §2 specifies the two commitments and the `y_ring` fix. §3 covers Jolt's sumchecks and the R1CS. §4 covers Hachi's recursion and sumcheck masking. §5 covers the fused Spartan at level `L − 1` and the tail quadratic discharge. §6 is the joint tail sigma. §§7–10 cover verifier checks, simulator, cost, and concrete parameters. §11 is open questions.

### 1.1 Pipeline phases

Jolt and Hachi compose into a single joint protocol whose sumchecks are all masked by pre-committed pads.
Structurally, four phases, in execution order:

**Phase 1: `Com_pre`.**
One Ajtai commitment, issued before any challenge is drawn.
Contains Jolt's witness polynomials `{P_j}` via Hachi's mega-polynomial layout, plus a pad coefficient tuple per masked sumcheck round across the whole pipeline, plus one `y_ring` garbage polynomial `g^{(ℓ)}` per Hachi level (§2.4), plus one LNP22 garbage ring element for phase-4's residual-quadratic add-on (§5.5).

**Phase 2: Jolt sumchecks (masked).**
The seven batched Jolt sumchecks (§3.1) run with round polynomials masked by phase-1 pads.
Transcript size matches non-ZK Jolt exactly.
No algebraic verifier check is evaluated here; every round-sum and round-eval identity is recorded as a linear relation on the committed pads, deferred to phase 4.

**Phase 3: Hachi recursion with fused Spartan (masked).**
Hachi's per-level stage-1 and stage-2 sumchecks fold the `Com_pre` witness through `L` levels (§4).
Two of those levels do extra work.

At **level `L − 1`** (the "Spartan-fused" level), the prover issues a second Ajtai commitment `Com_aux1` carrying the R1CS aux variables that arithmetize Jolt's verifier algebra (`c_1_aux`, chain-boundary outputs, sum-of-products intermediates, PCS-binding scalars; §3.3).
`Com_aux1` is laid into Hachi's mega-polynomial at the level-`L−1` ring, so its MLE variables coincide with level `L − 1`'s Hachi witness variables.
Three sumchecks run at this level, all masked by phase-1 pads:

- **Spartan outer** (`\~12` rounds over R1CS rows): reduces `Σ_x eq(τ, x) · (Az·Bz − Cz)(x) = 0` to three row-level claims `(Az(r_x), Bz(r_x), Cz(r_x))`.
- **Hachi stage-1** (norm check, unchanged from non-ZK except for masking).
- **Fused [Hachi stage-2 + Spartan inner]** (`max(Hachi stage-2 rounds, log₂ #R1CS cols)` rounds): a single sumcheck that simultaneously closes Hachi's stage-2 fold and reduces Spartan's three row-level claims to one evaluation claim `v_Spartan = Z(r_y)` on the level-`L−1` extended witness (`Com_pre` portion ∪ `Com_aux1`).
Fusion works because `Com_aux1`'s MLE variables live in the same level-`L−1` variable space as Hachi's stage-2 sumcheck, allowing a single RLC-combined round polynomial per round.

At **level `L`** (the final fold), a normal Hachi fold runs on the level-`L−1` extended witness.
One `D`-fold shrinks `Com_aux1` from `\~few thousand` F_q at commit time to `\~100` F_q at the tail, so `Com_aux1` contributes negligibly to the final tail witness.

After phase 3 we have (i) a small direct-tail witness, (ii) a bundle of linear residuals (batched masked-sumcheck chain identities across all three clusters, Ajtai openings, evaluation claims, `y_ring` residuals, `V_ring` binding), and (iii) a **small number of residual quadratics**, of which the explicitly accounted-for one is the post-Spartan-outer identity `Az(r_x) · Bz(r_x) = Cz(r_x)` on three transcript scalars; auditing for additional quadratics that may surface from non-linear sumcheck round-eval closures (e.g., Hachi stage-2's `eq · w · (w + 1)`, stage-1's degree-`b/2` range check, the `y_ring` trace check) is open (§11.12).

**Phase 4: Joint tail sigma.**
One Gaussian-masking sigma protocol at the tail discharges every remaining relation simultaneously:

- Ajtai openings of `Com_pre` and `Com_aux1` on the folded tail witness.
- Hachi's tail evaluation claim and Spartan's closing claim `v_Spartan`.
- All linear residuals from the three masked-sumcheck clusters, one batching scalar per cluster.
- The `y_ring` well-formedness and residual pins (§2.4).
- `V_ring` linear binding.
- The residual quadratics (the explicitly accounted-for one is Spartan's `Az(r_x) · Bz(r_x) = Cz(r_x)`; possibly a few more, see §11.12), each handled by LNP22's single-quadratic add-on: per quadratic, one pre-committed garbage ring element from `Com_pre` lets the prover send 2 extra scalars in the sigma first message that jointly pin the quadratic when combined with the verifier's challenge (§5.5 / §5.6; Nguyen thesis §5.2.1).

Everything folds into a single combined response.
No per-round commitments, no secondary sumcheck phase.

ZK techniques inherit the same order:

- LHL-hiding Ajtai commitments protect everything committed (phases 1 and 3 mid-recursion commit).
- Committed-pad masking protects every sumcheck round (phases 2 and 3).
- Gaussian masking + rejection sampling + LNP22 single-quadratic protect the tail (phase 4).

### 1.2 What leaks today

Non-ZK messages audited from `src/protocol/commitment_scheme.rs` and `src/protocol/transcript/labels.rs`:


| #   | Message                      | Transcript label             | Origin                          | Status                                                                                                       |
| --- | ---------------------------- | ---------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 1   | `u` (main commitment)        | `ABSORB_COMMITMENT`          | `commit`                        | Statistically hiding by LHL                                                                                  |
| 2   | Evaluation claims            | `ABSORB_EVALUATION_CLAIMS`   | `prove_one_level`               | Public                                                                                                       |
| 3   | `y_ring`                     | `ABSORB_RING_SWITCH_MESSAGE` | `prove_one_level`               | **Leaks** D−1 non-constant coefficients beyond the public trace (LNP22 coefficient masking fixes this, §2.4) |
| 4   | `v = D · ŵ`                  | `ABSORB_PROVER_V`            | `QuadraticEquation::new_prover` | Statistically hiding by LHL                                                                                  |
| 5   | Stage-1 sumcheck round polys | `ABSORB_SUMCHECK_ROUND`      | `HachiStage1Prover`             | **Leaks** partial sums (masking fixes this)                                                                  |
| 6   | `s_claim`                    | `ABSORB_SUMCHECK_S_CLAIM`    | between stages                  | **Leaks** reduced claim (propagates through masking)                                                         |
| 7   | Stage-2 sumcheck round polys | `ABSORB_SUMCHECK_ROUND`      | `HachiStage2Prover`             | **Leaks** partial sums (masking fixes this)                                                                  |
| 8   | `next_w_commitment`          | `ABSORB_SUMCHECK_W`          | `commit_w`                      | Statistically hiding by LHL                                                                                  |
| 9   | `next_w_eval`                | level proof field            | intermediate levels             | **Leaks** MLE eval (propagates through masking)                                                              |
| 10  | Tail `PackedDigits`          | —                            | `HachiProofTail::new`           | **Full witness in clear** (sigma protocol fixes this)                                                        |


Rows 1, 2, 4, 8 are safe as-is. Rows 3, 5, 6, 7, 9, 10 require ZK treatment.

On top of that, Jolt adds seven more sumchecks (see §3.1) and a batched opening. Every Jolt round poly and every chain-handoff claim leaks intermediate Jolt state.

### 1.3 Design choices at a glance

Throughout this note:

- **Hiding** is statistical, via LHL (Ajtai commitments) and perfect OTP (committed pads).
No MLWE or PRG-security assumption is needed.
Appendix A explains why MLWE-based hiding is infeasible at Hachi's `q = 2^128 − 275`.
- **Soundness** reduces to Hachi's existing CWSS extraction plus an `n/q` loss per masked-sumcheck cluster (§4.3, §5.4); aggregate loss is negligible.
- **Minimum folding ring dimension** is `D = 32` (`src/protocol/dispatch.rs:114`; D=16 was retired in the 2026-Q1 planner update, `docs/proof-size-reduction-study.md:13-48`).
The tail sigma protocol promotes the witness to D=64 because D=32 has no production sparse-challenge family (`docs/fourth-root-verifier.md:170-179`).
- **Rejection-sampling variant** for the tail is deferred (Open Question §11.3).
Numerical anchors in §10 use Rej1 as default.
- **Spartan placement** is **fused into Hachi at level `L − 1`** (§5).
`Com_aux1` (R1CS aux variables for Jolt's verifier algebra) is committed mid-recursion, laid out inside Hachi's mega-polynomial at the level-`L−1` ring so that Spartan's inner sumcheck shares variables with Hachi's stage-2 and the two fuse via RLC into a single sumcheck.
Spartan's outer sumcheck runs separately (12 rounds, over R1CS rows) at the same level.
The subsequent level-`L` fold absorbs `Com_aux1` into the tail witness, shrinking it by a factor `D` so the final tail carries 100 F_q of `Com_aux1` content (vs. a few thousand at commit time).
Rejected alternatives:
  - **Upfront Spartan** (before Hachi) is infeasible: `Com_aux1` depends on challenges drawn during phase 2, which don't exist yet.
  - **Fully deferred Spartan at the tail** (earlier draft of this note) works but leaves `Com_aux1` at full width at the tail, roughly doubling the effective tail witness.
  - **Early Spartan** (at levels `<L − 1`) folds `Com_aux1` more aggressively but inflates every subsequent level's commit matrix. Level `L − 1` is the latest point that still gets at least one fold on `Com_aux1`.
- **Spartan-sumcheck masking technique** is **per-round committed pads**, matching Jolt's and Hachi's own sumcheck masking.
The NovaBlindFold alternative (random relaxed-R1CS pair, fold in the commitment, non-ZK Spartan on the folded instance) is covered in a local companion note; it requires one lattice commitment per sumcheck round (≥ 512 B each), which at 500 total rounds would cost 250+ KB in commitments alone, dominating the entire proof.
- **Verifier R1CS shape** is measured (§3.4): `L = 2·R + 1579` linear rows and `Q = 723` quadratic rows (constant across trace lengths and across multiple guest programs).
Quadratic rows come entirely from stage-boundary sum-of-products chains; their count is determined by Jolt's stage structure, not the trace length.
All 723 quadratic rows aggregate through Spartan outer into **one** residual quadratic identity `Az(r_x) · Bz(r_x) = Cz(r_x)` on three scalars.
- **Residual-quadratic discharge** is **LNP22 single-quadratic at the tail sigma** (§5.5, §5.6; Nguyen thesis §5.2.1).
The one residual quadratic from Spartan outer costs 1 pre-committed garbage ring element in `Com_pre` + 2 extra scalars in the sigma first message, all absorbed into the same Gaussian-masking sigma as the linear relations.
An alternative is a second, much smaller Spartan invocation on the residual quadratic alone (25 rounds on a 3-variable instance); offered as a fallback if the LNP22-over-Ajtai LHL argument for the public scalar `v = y^T R_2 y` turns up implementation obstacles (§11.6).
- `**y_ring` per-level leakage** is discharged by LNP22 coefficient masking (ENS20 Eq. 7) in the **batched-residual** form, matching §2.3 / §4.1 for sumcheck pads: one garbage ring element `g^{(ℓ)}` per level in `Com_pre`, sampled uniformly in `R_q` with no hyperplane constraint, and the masked value `y_ring · σ_{-1}(v) + g^{(ℓ)}` absorbed under the existing `ABSORB_RING_SWITCH_MESSAGE` label (§2.4).
The verifier does not perform the per-level trace check at round time; it records the public residual `δ^{(ℓ)} := ct(y_ring^{(ℓ)}) − opening^{(ℓ)}` and defers everything to the phase-4 tail sigma, which discharges D `F_q` well-formedness rows plus one residual-pin row `ct(g^{(ℓ)}) = δ^{(ℓ)}` on the extracted `g^{(ℓ)}` per level.
γ is dropped because the tail sigma's binding extractor on `Com_pre` nails down `g^{(ℓ)}` directly, subsuming LNP22's commit-and-reveal soundness argument — this does not depend on `q` being prime or on any sampling-time constraint on `g`.

---

## 2. Commitments: `Com_pre` and `Com_aux1`

The ZK protocol uses two commitments.
`Com_pre` is issued in phase 1, before any challenge is drawn, and carries the witness plus every pad and garbage slot needed downstream.
`Com_aux1` is issued mid-recursion at the boundary between Hachi levels `L − 2` and `L − 1`, once Jolt's phase-2 challenges have fixed the R1CS aux-variable values; it carries the R1CS aux witness.

### 2.1 What is committed, when

`**Com_pre` (phase 1, before any challenge is drawn).**
A single Ajtai commitment containing:

- Jolt's witness polynomials `{P_j}` via Hachi's mega-polynomial layout (`jolt-core/src/poly/commitment/hachi/*`, `docs/block-order.md`).
- Pad coefficients for every masked sumcheck round across the entire protocol: `{ρ_{cluster,i,k} : cluster ∈ {Jolt-stage, Hachi-level-ℓ-stage, Spartan-outer, Spartan-fused}, round i, degree index k ∈ [0, d]}`.
The total pad length is
  `L_pad = Σ_{cluster, i} (d_{cluster, i} + 1)`
  summed over the masked-sumcheck clusters of phases 2 and 3:
  - Jolt's seven sumcheck stages (phase 2, §3.1).
  - Every Hachi recursion level's stage-1 and stage-2 sumchecks (phase 3, §4.1).
  - Spartan outer at level `L − 1` (12 rounds, §5.3).
  - Fused `[Hachi stage-2 + Spartan inner]` at level `L − 1` (`max(stage-2 rounds, 14)` rounds; pads allocated once, since fusion shares rounds).
  Each coefficient is a fresh uniform element of `F_q`; no per-pad constraint is imposed at sampling time.
- One garbage polynomial `g^{(ℓ)} ∈ R_q` per Hachi level for LNP22 coefficient masking of the `y_ring` trace pin (§2.4), sampled uniformly in `R_q` with **no sampling-time constraint** (D independent uniform `F_q` coefficients).
The per-level constant-coefficient pin is carried as a residual `δ^{(ℓ)} := ct(y_ring^{(ℓ)}) − opening^{(ℓ)}` computed publicly from the transcript and discharged by one `F_q` linear row `ct(g^{(ℓ)}) = δ^{(ℓ)}` in the phase-4 tail sigma, alongside the D-row `R_q`-linear well-formedness of `y_ring^{(ℓ)}` — the same free-sample / batched-residual pattern as the sumcheck pads above.
- One ring element `g_{quad} ∈ R_q` for LNP22's single-quadratic add-on at the phase-4 tail sigma (§5.5). Sampled fresh at commit time; discharged as 1 garbage commitment slot + 2 scalars in the sigma first message to catch any cheat on the residual quadratic `Az(r_x) · Bz(r_x) = Cz(r_x)`.

All of `Com_pre` must bind into the transcript before any sumcheck challenge is drawn.
If pads were committed later, a malicious prover could pick them adaptively after seeing challenges and trivially cheat the tail residual check; binding upfront is structural.

`**Com_aux1` (mid-recursion, at the level-`L − 2` → level-`L − 1` boundary).**
A second Ajtai commitment, issued once Jolt's phase-2 challenges have fixed the R1CS aux-variable values but before level `L − 1` starts its sumchecks (§5.2).
Contains the R1CS aux witness that arithmetizes Jolt's verifier algebra:

- Per-round auxiliary variables `{c_1_aux_{s,i}}` (one per regular Jolt round) and `{c_0_aux_{s,i}}` (one per uni-skip round).
See §3.2 for why these are separated out.
- Chain-boundary outputs: `{next_claim_{s,i}}`, `{y_j}` (the batched evaluation outputs), `{initial_claim_vars}`, and any sum-of-products aux variables needed by chain-general outputs.
- PCS-binding scalars `V_ring` (one or more; see §11.1).

These values all depend on challenges drawn during phase 2, so the commitment is necessarily mid-protocol.
Total size: a few thousand full-field `F_q` elements.

**Layout of `Com_aux1`.**
`Com_aux1` is **merged into Hachi's mega-polynomial at level `L − 1`**: its F_q contents are gadget-decomposed at the level-`L − 1` log-basis `lb` into short-norm digits and appended as a new block of the level-`L − 1` Hachi witness.
Level `L − 1`'s Ajtai commit matrices extend column-wise to cover the new block, so `Com_aux1` uses the same `(A, B, D)` matrices as the main level-`L − 1` witness.
This alignment is what makes the `[Hachi stage-2 + Spartan inner]` sumcheck fusion of §5.3 possible.
The subsequent level-`L` fold then absorbs `Com_aux1` alongside the main witness, shrinking it by a factor `D` before the tail sigma.
See §5.2 for why this layout is structural rather than optional (closes earlier Open Question §11.5).

### 2.2 Two-tier Ajtai hiding via dedicated blinding

Every wire-visible Ajtai output must be explicitly blinded by a fresh short-norm randomness vector.
The hiding argument is LHL-over-`R_q`; the blinding entropy target is fixed by the product `D · κ` where `κ` is the module rank of the revealed output (not of any prover-internal intermediate).

#### 2.2.1 Ring, Ajtai matrix shape, binding

`R_q = Z_q[X] / (X^D + 1)`, `q = 2^128 − 275`.
Balanced-pow2 gadget digits live in `{−2^{lb−1}, …, 2^{lb−1} − 1}` (support size exactly `2^{lb}`, `lb ∈ {1, …, 6}` per `src/protocol/ring_switch.rs:1923`).
Support-separation `2M² < q` holds trivially (`M = 2^{lb−1}`, `2M² ≤ 2^{2·lb−1}`, and at `lb ≤ 6` we have `2M² ≤ 2^{11} ≪ 2^{128}`).
Each coefficient uniform on this support has `lb` bits of min-entropy.

Binding of a commitment `u = A · w ∈ R_q^κ` reduces to SIS at output dimension `D · κ`.
This is orthogonal to hiding and is governed by Hachi's existing SIS/collision analysis (`docs/fourth-root-verifier.md`).

**lhlCaveat: SIS parameters must be recalibrated once blinding is added.**  
Appending `m_B`, `m_D` blinding columns to `B`, `D` and matching short-norm slots to `t_hat`, `ŵ` changes the SIS instance in two ways:

1. **Witness dimension** grows: `n → n + m_B` for `B` and `n → n + m_D` for `D`. The SIS adversary has strictly more freedom to produce a short preimage (or a collision), because the columns of the extended matrix span a bigger module. In the worst case the adversary ignores the message columns entirely and collides using only the `m_·` blinding columns; binding then reduces to SIS against a uniform `R_q^{κ × m_·}` matrix, which is still hard as long as `m_· ≥`  the standard SIS sample-count floor (comfortably satisfied at `m_· ∈ [27, 260]` per §2.2.4).
2. **Extraction norm** grows slightly: the `σ`-protocol extractor produces `z_· = c · x_· + y_·` over the full extended vector, so the bound that feeds into SIS binding is evaluated on length `(n + m_·)` rather than `n`. Because blinding coefficients live on the same `lb`-bit support as message digits, `‖z_·‖_∞` does not grow; only the `ℓ_2` bound used for root-Hermite picks up a `√((n + m_·) / n)` factor. The worst case is at the smallest levels: at `max_num_vars = 9` (`w_ring ≈ 624` ring elements) with `m_· ≈ 68`, the factor is `√(692/624) ≈ 1.053` (≈ 0.074-bit shift in root-Hermite); at every larger level the factor is `< 1.01`.

Concretely this means the existing margins in `docs/fourth-root-verifier.md` (e.g., `B_extract < q^{2/3}`, root-Hermite factors) need a one-shot recomputation that adds the blinding columns to `n` and the blinding slots to the extractor's vector length. The change is expected to be within noise (sub-bit shift in root-Hermite, sub-percent shift in `B_extract`), but the recalibration has to be performed explicitly before any parameter set is published as secure; the per-level blinding-column counts in §2.2.4 are the inputs to that recomputation.

#### 2.2.2 What is on the wire per Hachi level

Each Hachi level commits via three Ajtai matrices `A, B, D` of module rank `n_a, n_b, n_d` respectively (`src/protocol/params.rs:110-141`).
The prover computes

```
t       = A · f_blocks                        (internal, never transmitted)
t_hat   = G^{-1}(t)                           (short-norm gadget digits, prover-held)
u       = B · t_hat                           ABSORB_COMMITMENT  (rank n_b)
ŵ       = G_1^{-1}(w)                         (short-norm gadget digits, prover-held)
v       = D · ŵ                               ABSORB_PROVER_V    (rank n_d)
```

See `src/protocol/commitment/commit.rs:331-385` for `u` and `src/protocol/quadratic_equation.rs:120-130, :314, :500, :696` for `v`.
`A`'s output `t` is the input to the gadget decomposition and never leaves the prover.
Only `u` and `v` appear in the transcript; `next_w_commitment` at level handoff (`ring_switch.rs:257, :436, :508`) becomes the next level's `u` and is blinded under that level's rules.

**Blinding is therefore added at `t_hat` and at `ŵ`, not at `w`.**
Concretely: extend the flat digit block `t_hat` by `m_B` fresh short-norm ring elements sampled uniformly on the balanced `lb`-bit support, and extend `ŵ` by `m_D` fresh short-norm ring elements sampled the same way.
The corresponding Ajtai matrices extend column-wise by the matching number of fresh uniform columns.
No relation in the protocol references the appended slots, so they carry through as pure blinding.

#### 2.2.3 LHL hiding bound and the entropy target

For a commitment `y = M · x` with `M ∈ R_q^{κ × (n+m)}` uniform and `x` short-norm, under support-separation the statistical distance is

```
Δ(y ; U(R_q^κ))  ≤  (1/2) · √( q^{D·κ} / 2^{H_∞(x)} ).
```

For `Δ ≤ 2^{-λ}` we need `H_∞(x) ≥ D · κ · log₂(q) + 2λ − 2`.
When `x = (x_msg, x_blind)` and we require hiding **independent of message content**, the inequality must be met by the blinding entropy alone:

```
H_∞(x_blind)  ≥  D · κ · log₂(q)  +  2λ  −  2.
```

For `u`, the LHL source is the fresh blinding vector, not the commitment output.
Writing the blinded commitment as

```
u = B_msg · t_hat + B_blind · r_B,
```

the public hash seed is `B_blind`, the source is `r_B`, and the extracted output is `B_blind · r_B ∈ R_q^{n_b}`.
For any fixed witness, `B_msg · t_hat` is only a fixed offset; adding a fixed offset preserves statistical distance from uniform.
Thus the LHL statement is applied to `(B_blind, B_blind · r_B)`, conditioned on public side information such as `B_msg`, `B_blind`, and the offset `B_msg · t_hat`.
Because `r_B` is sampled freshly and independently of this side information,

```
H_∞(r_B | B_msg, B_blind, B_msg · t_hat) = H_∞(r_B).
```

If `r_B` contains `m_B` ring elements, then it contains `m_B · D` independent coefficients.
Each coefficient is uniform on a balanced digit set of size `2^lb`, so every complete vector `r_B` has probability `2^{-m_B·D·lb}` and

```
H_∞(r_B) = -log₂(max_r Pr[r_B = r]) = m_B · D · lb.
```

The same accounting applies to `v`, replacing `r_B`, `B_blind`, and `n_b` by `r_D`, `D_blind`, and `n_d`.

Concretely at Hachi's `λ = 128`, `log₂(q) = 128`, entropy per blinding coefficient `= lb`:


| Blinding quantity                | Formula                                   | Expression at `λ = log₂(q) = 128` |
| -------------------------------- | ----------------------------------------- | --------------------------------- |
| Entropy target (bits)            | `D · κ · log₂(q) + 2λ − 2`                | `128·D·κ + 254`                   |
| Field-element count (F_q coeffs) | `(D · κ · log₂(q) + 2λ − 2) / lb`         | `(128·D·κ + 254) / lb`            |
| Ring-element count (R_q elts)    | `⌈(D · κ · log₂(q) + 2λ − 2) / (D · lb)⌉` | `⌈(128·κ + 254/D) / lb⌉`          |


The product `D · κ` is the fundamental quantity: it's the size in field elements of the revealed commitment output, which is what LHL must hide against.
`lb` only trades coefficient count against per-coefficient magnitude at fixed total entropy.

#### 2.2.4 Per-level blinding target for Hachi

Per level, `u` and `v` are the two revealed outputs, with module ranks `n_b` and `n_d`.
`t_hat` carries the blinding for `u`; `ŵ` carries the blinding for `v`.
Exact blinding cost per commitment:

```
  target_bits(u) = D · n_b · log₂(q) + 2λ − 2         ≈  128 · D · n_b  + 254
  target_bits(v) = D · n_d · log₂(q) + 2λ − 2         ≈  128 · D · n_d  + 254
  #Fq_coeffs(u)  = ⌈ target_bits(u) / lb ⌉
  #Fq_coeffs(v)  = ⌈ target_bits(v) / lb ⌉
  #Rq_elts(u)    = ⌈ #Fq_coeffs(u) / D ⌉
  #Rq_elts(v)    = ⌈ #Fq_coeffs(v) / D ⌉
```

**Exact enumeration of every `(D, n_b, n_d, lb)` combination that occurs at any fold level** of any generated schedule (`layout_num_claims = 1`, onehot backend, from `src/protocol/commitment/generated/fp128_d{32,64,128}_onehot.rs`):


| D   | `n_b` | `n_d` | `lb` | target u (bits) | target v (bits) | u F_q coeffs | v F_q coeffs | **total F_q coeffs** | u R_q elts | v R_q elts |
| --- | ----- | ----- | ---- | --------------- | --------------- | ------------ | ------------ | -------------------- | ---------- | ---------- |
| 32  | 1     | 1     | 2    | 4,350           | 4,350           | 2,175        | 2,175        | **4,350**            | 68         | 68         |
| 32  | 2     | 1     | 2    | 8,446           | 4,350           | 4,223        | 2,175        | **6,398**            | 132        | 68         |
| 32  | 2     | 2     | 2    | 8,446           | 8,446           | 4,223        | 4,223        | **8,446**            | 132        | 132        |
| 32  | 2     | 2     | 3    | 8,446           | 8,446           | 2,816        | 2,816        | **5,632**            | 88         | 88         |
| 32  | 2     | 2     | 4    | 8,446           | 8,446           | 2,112        | 2,112        | **4,224**            | 66         | 66         |
| 32  | 3     | 2     | 2    | 12,542          | 8,446           | 6,271        | 4,223        | **10,494**           | 196        | 132        |
| 32  | 3     | 3     | 2    | 12,542          | 12,542          | 6,271        | 6,271        | **12,542**           | 196        | 196        |
| 32  | 4     | 3     | 2    | 16,638          | 12,542          | 8,319        | 6,271        | **14,590**           | 260        | 196        |
| 64  | 1     | 1     | 2    | 8,446           | 8,446           | 4,223        | 4,223        | **8,446**            | 66         | 66         |
| 64  | 1     | 1     | 3    | 8,446           | 8,446           | 2,816        | 2,816        | **5,632**            | 44         | 44         |
| 64  | 1     | 1     | 4    | 8,446           | 8,446           | 2,112        | 2,112        | **4,224**            | 33         | 33         |
| 64  | 1     | 1     | 5    | 8,446           | 8,446           | 1,690        | 1,690        | **3,380**            | 27         | 27         |
| 64  | 2     | 2     | 2    | 16,638          | 16,638          | 8,319        | 8,319        | **16,638**           | 130        | 130        |
| 128 | 1     | 1     | 2    | 16,638          | 16,638          | 8,319        | 8,319        | **16,638**           | 65         | 65         |
| 128 | 1     | 1     | 3    | 16,638          | 16,638          | 5,546        | 5,546        | **11,092**           | 44         | 44         |
| 128 | 1     | 1     | 4    | 16,638          | 16,638          | 4,160        | 4,160        | **8,320**            | 33         | 33         |
| 128 | 1     | 1     | 5    | 16,638          | 16,638          | 3,328        | 3,328        | **6,656**            | 26         | 26         |


**Where each module-rank profile occurs as the root level (`L0`, `max_num_vars` range)** (`layout_num_claims = 1`):


| D   | `(n_b, n_d)` profile | Root level `max_num_vars` range | Also as intermediate?                           |
| --- | -------------------- | ------------------------------- | ----------------------------------------------- |
| 32  | (1, 1)               | 9–17                            | max_num_vars = 36                               |
| 32  | (2, 1)               | (none)                          | max_num_vars = 36                               |
| 32  | (2, 2)               | 18–36                           | max_num_vars ∈ 18–50 (most intermediate levels) |
| 32  | (3, 2)               | 37–38                           | (none)                                          |
| 32  | (3, 3)               | 39–48                           | (none)                                          |
| 32  | (4, 3)               | 49–50                           | (none)                                          |
| 64  | (1, 1)               | 10–36                           | max_num_vars ≥ 37 (all non-root levels)         |
| 64  | (2, 2)               | 37–50                           | (none)                                          |
| 128 | (1, 1)               | 11–50                           | max_num_vars = 40                               |


`layout_num_claims = 4` (batched layout) shifts some thresholds by ±1 but introduces no new `(n_b, n_d)` combinations.

`n_a` takes values in `{1, 2, 3, 4}` independently, but is invisible to the hiding argument: `A`'s output `t` never leaves the prover, so its module rank contributes nothing to the blinding target.

**Takeaways.**

- `(n_b, n_d)` at `D = 32` grows well beyond the (2, 2) ceiling for large `max_num_vars`. The peak is `(4, 3)` at `max_num_vars ∈ {49, 50}`, costing **14,590** F_q blinding coefficients (456 R_q elements) for that one root level.
- At `D = 64`, only two profiles occur: `(1, 1)` for `max_num_vars ≤ 36` and `(2, 2)` for `max_num_vars ≥ 37`. No `(1, 2)`-style asymmetric roots at the root level.
- At `D = 128`, the profile is universally `(1, 1)`; only `lb` varies with witness size.
- Smaller `lb` (fewer bits per coefficient) inflates F_q-coefficient counts proportionally but keeps R_q-element counts roughly stable (the total bits of entropy stay the same); verifier preferring larger `lb` reduces opening-response size, at the cost of slightly looser short-norm bounds.

#### 2.2.5 Blinding every wire-visible commitment

The same rule applies uniformly to every Ajtai output that enters the transcript:

- **Per Hachi level**: extend `t_hat` and `ŵ` as above. Every `ABSORB_COMMITMENT` / `ABSORB_PROVER_V` / `ABSORB_SUMCHECK_W` slot inherits its own fresh blinding vector sized by the per-level row of §2.2.4.
- `**Com_pre`**: root-level `u` / `v` blinding is sized by the root `(n_b, n_d)` profile at that `max_num_vars`. For a typical Jolt target at `max_num_vars ≈ 30` with `D = 32`, that's the `(2, 2)` row (≤ 8,446 F_q coeffs); at `max_num_vars ≈ 45`, that's the `(3, 3)` row (12,542 F_q coeffs); at `max_num_vars ≈ 50`, that's the `(4, 3)` row (14,590 F_q coeffs). Pads still live in the message portion of `w`; their role is OTP hiding of sumcheck messages (§4.2), no longer double duty as commitment-hiding entropy.
- `**Com_aux1` at level `L − 1**`: merged into Hachi's mega-polynomial layout (§2.1, §5.2), committed via level-`L − 1`'s existing `(A, B, D)` matrices. No dedicated standalone matrix needed; the level-`L − 1` blinding profile from the table above covers `Com_aux1`'s block as part of the extended witness. No extra blinding target beyond what level `L − 1` already pays for the main witness.
- **LNP22 quadratic garbage `g_{quad}`** (§5.5): one ring element pre-committed inside `Com_pre` at phase 1. Same LHL argument as every other `Com_pre` slot; no separate commitment, no separate blinding target.

#### 2.2.6 Opening response extension

The phase-5 sigma response (§6) and each per-level opening protocol extend to include blinding slots.
The masked opening `z = c · x + y` runs over the full vector `x = (x_msg, x_blind)`, giving response components `(z_msg, z_blind)`.
Verifier checks the commitment fold on the full concatenated vector.
Wire cost: `m_B + m_D` extra ring elements per level in its opening response, a few hundred bytes after Gaussian-rejection compression, tens of KB across the whole recursion.

#### 2.2.7 Remark on the old argument

Earlier drafts argued hiding emerged incidentally from pad entropy in `Com_pre` (thousands of uniform `F_q` pad coefficients giving `H_∞(ŵ) ≫ 8400` bits).
This was technically correct for `Com_pre` but fragile: it conflated OTP hiding of sumcheck messages with LHL hiding of Ajtai commitments, and it provided no hiding story for small follow-on commitments (`Com_aux1` mid-recursion).
Dedicated blinding on every wire-visible commitment fixes both: the hiding statement is now uniform, parameter-free, and independent of message content.

### 2.3 Pad coefficient layout

Per sumcheck round `i` at stage `s` in phase `p`, with round polynomial of degree `d_{p,s,i}`: commit `d_{p,s,i} + 1` pad coefficients `{ρ_{p,s,i,0}, ..., ρ_{p,s,i,d_{p,s,i}}}`, sampled independently uniformly in `F_q`.
**No per-pad constraint** (no `ρ_i(0) + ρ_i(1) = 0`, no chain relation) is imposed at sampling time; the required invariants are discharged at the tail as batched linear relations (§4.1, §4.3, §5.4).

This is a deviation from an earlier version of this doc which claimed the pads could be structurally constrained.
That turned out to be incompatible with upfront commitment: the chain invariant that would make round-by-round verifier checks pass (`ρ_i(0) + ρ_i(1) = ρ_{i-1}(r_{i-1})` for rounds i > 1) depends on challenges that aren't known at commit time.
The clean resolution is to commit freely and defer the chain constraints.

**Why not per-round commitments.**
The BlindFold / NovaBlindFold alternative (Pedersen-commit each round polynomial and defer verifier checks to a folding step) sends one commitment per round.
At 500 total rounds across phases 2, 3, and 4, and ≥ 512 B per lattice commitment, that costs 250+ KB in commitments alone — it dominates the rest of the proof.
Committed pads sidestep this: messages go in the clear (plus a mask), pads are batched into a single upfront commitment, and all residual checks fold into one sigma at the tail.
See the local companion note (`hachi-blindfold-walkthrough-and-lattice-generalization.md` in the author's working folder) for the per-round-commitment path and why it is research-only in the lattice setting.

Storage per round: `d_{p,s,i} + 1` field elements of `F_q`.
For a typical Hachi sumcheck with `d = 7` (range check with `b = 4`), that's 8 pad coefficients per round.
Total `L_pad` across the pipeline (Jolt's 333 rounds + Hachi's 150 rounds + Spartan's 20–25 rounds) is on the order of 4,000–5,000 field elements.
These slots live in the message portion of `Com_pre`'s witness; LHL hiding of `Com_pre` is a separate concern handled by the dedicated blinding of §2.2.

### 2.4 Discharging `y_ring` via LNP22 coefficient masking

**What the verifier actually needs.**
In the non-ZK path, `prove_one_level` absorbs `y_ring ∈ R_q` under `ABSORB_RING_SWITCH_MESSAGE` (`src/protocol/commitment_scheme.rs:3185`).
The only check against `y_ring` is the trace pin (`commitment_scheme.rs:3189-3193`):

```
trace(y_ring · σ_{-1}(v)) = D · opening,     i.e.     ct(y_ring · σ_{-1}(v)) = opening
```

where `trace(u) = D · u.coefficients()[0]` (`commitment_scheme.rs:3294-3297`).
By LNP22 Lemma 2.4 / Nguyen Thesis §1.1.3, this equals the `Z_q`-inner product `⟨y_ring, v⟩`.
It is a single-`F_q`-coefficient linear equation on the D-coefficient product `y_ring · σ_{-1}(v)`.
The remaining D−1 coefficients are pure leakage about the committed witness, since `y_ring_orig = Φ^{(ℓ)}(w^{(ℓ)})` is itself an `R_q`-linear functional of the current level's committed witness `w^{(ℓ)}` (`src/protocol/hachi_poly_ops/mod.rs:165-179`; the scalars in `Φ^{(ℓ)}` are public and determined by `tau0^{(ℓ)}, tau1^{(ℓ)}, alpha^{(ℓ)}, m_evals_x^{(ℓ)}, alpha_evals_y^{(ℓ)}`).

**Wrong fix (dropped).**
An earlier draft proposed committing `y_ring_orig` in `Com_pre` and discharging the trace pin as a linear stage-2 constraint.
That works but costs D `F_q` coefficients of commitment slot per level when a single ring element of garbage suffices, and it does not match the established lattice-ZK pattern for this exact shape.

**Adopted fix: ENS20 / LNP22 coefficient masking in batched-residual form (LNP22 §1.3 Eqs. (7)-(8), Nguyen Thesis §1.1.3; parallel to §2.3 / §4.1).**
The statement `ct(y_ring_orig · σ_{-1}(v)) = opening` is a `Z_q`-linear equation on the constant coefficient of a publicly-known `R_q`-linear functional of the committed witness.
This is exactly the shape LNP22 §1.3 handles.
We adopt the **free-garbage + batched-residual** variant, matching the per-round sumcheck-pad invariant of §2.3 / §4.1 / §4.3: commit the garbage freely, have the verifier defer every per-level check to the tail sigma, and let the phase-5 tail sigma discharge well-formedness and the constant-coefficient pin together as additional linear rows.
The transcript slot name stays `y_ring` and now carries the masked object that plays the role of LNP22's `h`.

1. **Garbage commitment (free, no sampling-time constraint).**

At phase-1 commit time, include one fresh polynomial `g^{(ℓ)} ∈ R_q` per Hachi level, sampled uniformly in `R_q` (D independent uniform `F_q` coefficients; **no hyperplane constraint**).
These `g^{(ℓ)}` live in the message portion of `Com_pre` (one ring element per level; negligible next to the pad budget).
This mirrors §2.3's free sumcheck pads, and for the same reason it is cleaner downstream: the invariant the verifier cares about is discharged once at the tail, not built into every sampled object.

1. **Mask-and-send (replaces sending `y_ring_orig` in clear).**

When the level reaches the point where `y_ring_orig` would be absorbed in the non-ZK path, the prover instead computes

```
y_ring^{(ℓ)}  :=  y_ring_orig^{(ℓ)} · σ_{-1}(v^{(ℓ)})  +  g^{(ℓ)}   ∈ R_q
```

and absorbs `y_ring^{(ℓ)}` under the existing `ABSORB_RING_SWITCH_MESSAGE` label.
`y_ring_orig^{(ℓ)}` is never absorbed.
Both `v^{(ℓ)}` (derived from the previous level's folded opening point via `reduce_inner_opening_to_ring_element`, `commitment_scheme.rs:3187`) and `g^{(ℓ)}` (pre-committed in `Com_pre`) are available at this point.

1. **Verifier's per-level action (defer, do not check).**

The verifier absorbs `y_ring^{(ℓ)}` but **does not perform the trace check at round time** — analogous to §4.1's rule that the verifier does not check `g̃_i(0) + g̃_i(1) = c_{i-1}` during a masked sumcheck round.
The verifier records the per-level residual

```
δ^{(ℓ)}  :=  ct(y_ring^{(ℓ)}) − opening^{(ℓ)}      ∈ Z_q
```

which is publicly derivable from the transcript (both `ct(y_ring^{(ℓ)})` and `opening^{(ℓ)}` are public by the time the level finishes).
Under the honest protocol `δ^{(ℓ)} = ct(g^{(ℓ)})` exactly; the tail sigma pins this equation on the extracted `g^{(ℓ)}`.

1. **Tail sigma discharge.**

The phase-5 tail sigma protocol (§6) proves, per level, the following linear rows, aggregated with the rest of the tail's linear functional via the standard batching challenge (alongside the §4.1 sumcheck residuals, the Spartan residuals of §5.4, and the PCS evaluation claim):

- `R_q`-linear well-formedness of `y_ring^{(ℓ)}` (D `F_q` rows per level): `y_ring^{(ℓ)} = Φ^{(ℓ)}(w^{(ℓ)}) · σ_{-1}(v^{(ℓ)}) + g^{(ℓ)}`, with public coefficients (`σ_{-1}(v^{(ℓ)})` and the `Φ^{(ℓ)}` scalars are all in the transcript by tail time) on committed slots (`w^{(ℓ)}` and `g^{(ℓ)}`).
- Residual pin (1 `F_q` row per level): `ct(g^{(ℓ)}) = δ^{(ℓ)}`, with RHS the public value recorded in step 3.

No new sumcheck, no new round; the tail sigma already discharges arbitrary linear relations of this shape.

**Embedding note.**
LNP22 §1.3 presents the scheme with `L_ver = ct(·)` (constant-coefficient extraction); §4.4 presents a σ_{-1}-trace variant that halves garbage when batching ≥ 2 relations into a single `g`.
Hachi's verifier extracts via `trace(u) = D · u.coefficients()[0] = D · ct(u)` (`commitment_scheme.rs:3294-3297`), which is the Galois trace from `R_q` down to `Z_q` for the power-of-2 cyclotomic `R_q = Z_q[X]/(X^D + 1)`.
Because `D = 2^k` is coprime to the prime `q = 2^{128} − 275`, `D · ct(x) = 0 ⟺ ct(x) = 0 ⟺ trace(x) = 0`: the constant-coefficient and Galois-trace choices coincide algebraically in Hachi's ring.
We work with `ct(·)` throughout this section.
The σ_{-1}-trace form would only matter if a future version batches ≥ 2 relations into a single `g`; revisit via LNP22 §4.4 if so.

**Hiding.**
All D coefficients of `g^{(ℓ)}` are uniform in `F_q`, so `y_ring^{(ℓ)} = y_ring_orig · σ_{-1}(v) + g^{(ℓ)}` is uniform in `R_q` conditioned on the transcript up to its absorption.
The simulator samples `y_ring^{(ℓ)}` uniformly in `R_q` (§8, Hybrid 3); the real and simulated distributions coincide.
No information about `y_ring_orig^{(ℓ)}` (and hence about the witness beyond the public `opening^{(ℓ)}`, which is already in the transcript) leaks.

**Soundness — why γ is dropped.**
LNP22 Eq. (7) writes `h = γ · f + g` with `γ` a Fiat-Shamir challenge drawn **after** `g` is committed, and the standard bound is `1/q_1` per shot where `q_1` is the smallest prime factor of `q`.
The role of `γ` is specific to LNP22's **commit-and-reveal** model: the verifier only ever sees `g` through the folded scalar `h`, and `γ` is what prevents the prover from adaptively choosing `ct(g)` to cancel a cheat.

Our setting is commit-and-**prove**: `g^{(ℓ)}` is bound inside `Com_pre` before any challenge is drawn, and the phase-5 tail sigma's special-soundness extractor (§6) recovers `g^{(ℓ)}` from two accepting transcripts via Ajtai binding of `Com_pre`.
The tail sigma then enforces `ct(g^{(ℓ)}) = δ^{(ℓ)}` on the **extracted** `g^{(ℓ)}` directly, as a single linear row in the aggregated functional.
Soundness of this row is `≤ negl(128)` via Schwartz-Zippel on the tail's batching challenge combined with the sigma's own special-soundness — **the same analysis as §4.3 for sumcheck residuals, applied verbatim**.
LNP22's `γ` and its `1/q_1` per-shot bound are subsumed by extraction: no adaptive `ct(g)` attack is possible once `g^{(ℓ)}` is bound in `Com_pre`, and this holds regardless of whether we frame the row as a structural `ct(g^{(ℓ)}) = 0` or a residual `ct(g^{(ℓ)}) = δ^{(ℓ)}`.
The argument does **not** rely on `q` being prime (though it happens to be) and does **not** rely on a sampling-time hyperplane constraint on `g`.

This is the same design choice as §2.3 / §4.1 for sumcheck pads: free sample at commit time, deferred batched pin at the tail, soundness discharged by the tail extractor.
For sumcheck it was forced (the per-round residual RHS depends on challenges not known at commit time, §2.3); for `y_ring` it is matched by choice for consistency and for a cleaner simulator (`y_ring^{(ℓ)}` uniform in `R_q` rather than on a `ct = opening` hyperplane).

**σ-invariance.**
The masking here uses only `σ_{-1}` inside a public factor; no ring-valued challenge is constrained to lie in a σ-invariant subring.
The `σ`-invariant-challenge subtlety (§11.6, relevant to *quadratic* LNP22 proofs) is orthogonal to this section.

**Cost per Hachi level.**


| Item                                          | Before (in clear) | After (masked)                                                                                           |
| --------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------- |
| Transcript ring elements absorbed as `y_ring` | D coeffs (leaky)  | D coeffs (fully masked, uniform in `R_q`)                                                                |
| `Com_pre` message overhead                    | 0                 | 1 ring element for `g^{(ℓ)}` (D free coefficients, no hyperplane constraint)                             |
| Extra verifier challenges                     | 0                 | 0 (γ dropped, see Soundness)                                                                             |
| Per-level immediate check                     | 1 trace check     | 0 (deferred to tail)                                                                                     |
| Extra tail-sigma rows                         | 0                 | D `F_q` rows (`y_ring` well-formedness) + 1 `F_q` row (residual pin `ct(g^{(ℓ)}) = δ^{(ℓ)}`, public RHS) |


Transcript size is unchanged; the added cost is one ring element of commitment per level plus `D + 1` linear rows bundled into the aggregated tail sigma linear functional.

---

## 3. Phase 2: Jolt Sumchecks (masked)

### 3.1 The seven Jolt sumcheck stages

Jolt runs seven batched sumchecks (`stage1_…` through `stage7_…` in `jolt-core/src/zkvm/prover.rs:727-738`), two of them preceded by a univariate-skip first round. Per-stage round counts, read directly from each instance's `num_rounds()` and folded through the batched `max_num_rounds` rule in `subprotocols/sumcheck.rs`:


| Stage | Contents                                                                                                                             | Regular rounds      | Uni-skip                                    |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------- | ------------------------------------------- |
| 1     | Spartan outer                                                                                                                        | `1 + log T`         | 1 round, `d = 27`, symmetric domain size 10 |
| 2     | RAM RW + Spartan product-virtual remainder + instruction-lookup claim-reduction + RAM raf + RAM output                               | `log K_ram + log T` | 1 round, `d = 6`, symmetric domain size 3   |
| 3     | Spartan shift + instruction-input + registers claim-reduction                                                                        | `log T`             | —                                           |
| 4     | Registers RW + RAM val-check                                                                                                         | `7 + log T`         | —                                           |
| 5     | Instruction-lookup read-raf + RAM ra-reduction + registers val-eval                                                                  | `128 + log T`       | —                                           |
| 6     | Bytecode read-raf + booleanity + RAM hamming-booleanity + RAM ra-virtual + lookups ra-virtual + inc-reduction (+ advice cycle phase) | `log K_bc + log T`  | —                                           |
| 7     | Hamming-weight claim-reduction (+ advice address phase)                                                                              | `log k_chunk`       | —                                           |


Constants:

- `7 = log₂(RISCV_REGISTER_COUNT + VIRTUAL_REGISTER_COUNT) = log₂ 128` (`common/src/constants.rs:2-5`).
- `128 = XLEN · 2` is the abstract instruction-lookup address space for RV64 (`jolt-core/src/zkvm/instruction_lookups/mod.rs:6`, `XLEN = 64` in `common/src/constants.rs:1`).
- `log k_chunk = 8` when `log T ≥ HACHI_ONEHOT_CHUNK_THRESHOLD_LOG_T`, else `4` (`jolt-core/src/poly/commitment/hachi/commitment_scheme.rs:746-752`).
- `log K_ram`, `log K_bc` are program-dependent; `ram_K ≥ compute_min_ram_K(program_io, trace_len)` (`jolt-core/src/zkvm/verifier.rs:302-307`).
- Uni-skip degree constants: `NUM_R1CS_CONSTRAINTS = 19` → `OUTER_UNIVARIATE_SKIP_DEGREE = 9`, `OUTER_FIRST_ROUND_POLY_NUM_COEFFS = 28` (`zkvm/r1cs/constraints.rs:413, 422`). `NUM_PRODUCT_VIRTUAL = 3` → `PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE = 2`, `PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS = 7` (`zkvm/r1cs/constraints.rs:532-538`).

**Total Jolt sumcheck rounds:**

`R_Jolt = 138 + 6·log T + log K_ram + log K_bc + log k_chunk`

= 2 uni-skip + `(1 + log T)` + `(log K_ram + log T)` + `log T` + `(7 + log T)` + `(128 + log T)` + `(log K_bc + log T)` + `log k_chunk`.

Realistic RV64 run with `log T = 24, log K_ram = 25, log K_bc = 18, log k_chunk = 8`: **333 rounds**. Stage 5 alone contributes `128 + log T ≈ 152` rounds and dominates the constant-term budget; an RV32 port would save 64 of those.

### 3.2 Masked round polynomials and `c_1_aux` compression

**Transcript format (per-round).** For every round polynomial `g_{s,i}(X) = Σ_{k=0}^{d} c_{s,i,k} · X^k` of degree `d = d_{s,i}`, the prover sends `d` cleartext field elements:

`(m'_{s,i,0}, m'_{s,i,2}, m'_{s,i,3}, ..., m'_{s,i,d})`  with  `m'_{s,i,k} = c_{s,i,k} + ρ_{s,i,k}`  for `k ≠ 1`.

The linear coefficient is **not transmitted**.
This matches non-ZK Jolt's `CompressedUniPoly` convention (`jolt-core/src/poly/unipoly.rs:25-27, 299-307`), which drops `c_1` and reconstructs it verifier-side from the sum constraint.
In the ZK path, `c_{s,i,1}` is never revealed: it becomes an auxiliary witness `c_1_aux_{s,i}` in `Com_aux1`, and the round-sum identity pins its value inside the R1CS (§3.3).

No per-round commitments.
The transcript absorbs `m'_{s,i,·}` and derives `r_{s,i}` exactly as in non-ZK Jolt.
Soundness of the honest-verifier sumcheck reduction is unchanged; what changes is that the verifier's algebraic round-sum and round-eval checks are deferred to the fused Spartan at Hachi level `L − 1` (§5).

**Hiding per round.** Conditioned on the transcript up to round `(s, i)`, each sent `m'_{s,i,k}` is a uniform-random element of `F_q` because `ρ_{s,i,k}` is uniform and independent of prior transcript content (pads were committed in `Com_pre` and never revealed). This is a perfect one-time pad on each sent coefficient. The omitted `c_1` is not sent at all, so nothing is revealed about it either.

**Uni-skip rounds: dropped-coefficient index.** For the two uni-skip first rounds, the symmetric evaluation domain makes odd power sums vanish (`power_sums[1] = 0`), so `c_1` is unconstrained by the sum row.
Dropping `c_1` there loses a degree of freedom.
Instead drop `c_0` (`power_sums[0] = |domain|` is always nonzero) and use `c_0_aux_{s,i}` in `Com_aux1`.
The per-stage choice of `k_0` (which coefficient is dropped) lives in `StageConfig`; the R1CS builder queries it to know which pad to skip allocating and which aux variable to allocate.

**Witness-layout accounting.** Let `R_Jolt = total_rounds` across Jolt's seven stages and `L_pad_Jolt = Σ_{s,i} d_{s,i}` (sum of per-round degrees, not d+1, because `pad_1` and `pad_0_uniskip` are dropped).
Then:

- `Com_pre` contains `L_pad_Jolt` pad slots (one per *sent* coefficient), plus pads for Hachi's and Spartan's sumchecks from phase 3 (including the level-`L − 1` Spartan outer and fused stage-2/inner rounds, §5.3).
- `Com_aux1` contains `R_Jolt` `c_1_aux` / `c_0_aux_uniskip` slots, plus chain-boundary aux, sum-of-products aux, and `V_ring` (§5.2).
- Net commitment-size delta vs. a naive `(d+1)`-per-round layout: zero on the pad side.
The `R_Jolt` slots "saved" by not padding the omitted coefficient are added back in `Com_aux1`.
- Transcript size matches non-ZK Jolt exactly: `d` field elements per regular round, `d_{uniskip}` per uni-skip round.

This is the **c_1 compression**.
It preserves the ZK wire-size identity with non-ZK Jolt at the cost of one extra aux row per round in the verifier R1CS.
The R1CS itself is a one-time piece of work closed by the fused Spartan at Hachi level `L − 1` (§3.3, §5).

### 3.3 Recording the R1CS for verifier's algebraic checks

The R1CS described below is **recorded during phase 2**, not materialized yet.
The prover accumulates the per-round constraints as each sumcheck round runs; the witness variables are R1CS aux slots that will live in `Com_aux1`, committed mid-recursion at the boundary between Hachi levels `L − 2` and `L − 1`.
Spartan proves the accumulated R1CS at level `L − 1`, fused with Hachi's stage-2 sumcheck (§5).

We keep BlindFold's PCS-agnostic DSL (`output_constraint.rs`, `layout.rs`, the `StageConfig`/`ClaimBindingConfig` types, `VerifierR1CSBuilder::build` in `jolt-core/src/subprotocols/blindfold/`).
What changes versus BlindFold:

- Witness layout: flat multilinear (no Hyrax grid).
- Variable swap: `c_{s,i,k} → m'_{s,i,k} − ρ_{s,i,k}` for `k ≠ 1` in regular rounds (and `k ≠ 0` in uni-skip rounds).
`c_1` stays as `c_1_aux` in `Com_aux1`.
- Chain-boundary outputs `{next_claim, y_j, V_ring, ...}` live in `Com_aux1` as witness variables.

Constraint families:

- **Round sum** (one row per regular round). Derived from the non-ZK identity `2·c_0 + c_1 + Σ_{k≥2} c_k = claim_prev` by the substitution above:
  ```
  (2·ρ_0 + ρ_2 + ρ_3 + ... + ρ_d + claim_prev − c_1_aux) · 1  =  M_baked · u
    where M_baked = 2·m'_0 + m'_2 + m'_3 + ... + m'_d   (from transcript).
  ```
  `claim_prev` is either the initial-claim variable of the chain, a baked constant, or the previous round's `next_claim` variable (`std::mem::replace` chaining trick at `r1cs.rs:564-567`). `c_1_aux` appears with coefficient `−1` on the A side; the row uniquely pins its value.
- **Round eval** (one row per regular round). Derived from `c_0 + γ·c_1 + γ²·c_2 + … + γ^d·c_d = next_claim`:
  ```
  (ρ_0 + γ²·ρ_2 + ... + γ^d·ρ_d + next_claim − γ·c_1_aux) · 1  =  E_baked · u
    where E_baked = m'_0 + γ²·m'_2 + ... + γ^d·m'_d.
  ```
  Powers of `γ` are baked public constants; `next_claim` lives in `Com_aux1`.
- **Uni-skip sum** (one row per uni-skip round). Verifier identity `Σ_k power_sums[k]·c_k = claim_prev` with `power_sums[k] = Σ_{t ∈ domain} t^k`. For symmetric domains `power_sums[1] = 0`, so `c_0` is the dropped-coefficient:
  ```
  (power_sums[2]·ρ_2 + ... + power_sums[d]·ρ_d + claim_prev − power_sums[0]·c_0_aux) · 1
    = M_baked_uniskip · u   where M_baked_uniskip = Σ_{k≥2} power_sums[k]·m'_k.
  ```
  Uni-skip has no separate eval row; its output feeds the first regular round's `claim_prev` through the chain-boundary machinery.
- **Chain-boundary linear output** (one row per linear chain boundary). `claim_last = Σ_j α_j · y_j` with `α_j` baked.
- **Chain-boundary general output** (sum-of-products rows). Same decomposition BlindFold uses today (`R1csConstraintVisitor` emits `max(1, L-1)` multiplication rows plus one sum row per term). Covers Spartan outer's `Az · Bz − Cz = eq(r,x) · …` closure and similar mixed forms.
- **Initial-input binding** (when a chain starts mid-pipeline with a claim that is itself a function of prior openings, Spartan inner's `ra·az + rb·bz + rc·cz`). Same DSL as the general output row; introduces one `InitialClaimVar` witness slot bound by the sum-of-products constraint.
- **PCS binding** (one linear row):
  ```
  V_ring  =  Σ_j γ^j · y_j
  ```
  `V_ring` is a single witness variable in `Com_aux1`; `γ^j` are baked.

This replaces BlindFold's per-eval `(extra_output_var, extra_blinding_var)` pair and its external Pedersen eval-commitment.
Hachi's tail sigma protocol (§6) proves `V_ring` is consistent with the opening of `Com_pre ∪ Com_aux1` via a free additional linear relation.

**Row counts.** `2·R_regular + R_uniskip` from round sum + eval (uni-skip has only a sum row), plus `O(stages)` from chain boundaries, plus 1 from PCS binding. With the round counts from §3.1, `R_regular = 136 + 6·log T + log K_ram + log K_bc + log k_chunk` and `R_uniskip = 2`, so the main body is

`272 + 12·log T + 2·log K_ram + 2·log K_bc + 2·log k_chunk + 2`  rows.

For the RV64 example: `272 + 288 + 50 + 36 + 16 + 2 = 664` rows, plus a few dozen for chain boundaries and PCS binding.

Spartan (run fused at Hachi level `L − 1`, §5) closes this R1CS.
Its output is a single claim `v_Spartan = Z(r_y)` on the multilinear encoding of `Z = (Com_pre portion ∪ Com_aux1 witnesses)`, which propagates through level `L`'s fold as one more linear row in the joint tail sigma's combined functional (§6).

### 3.4 Linearity structure of the verifier R1CS

A row `A·Z ⊙ B·Z = C·Z` is **linear** in the witness iff at least one of `A[row, :]` or `B[row, :]` has non-zero entries only in the constant column (column 0, carrying `Z[0] = 1`), so one side of the product reduces to a scalar constant.
Otherwise the row is **genuinely quadratic**.
This distinction drives §5.6's LNP22 alternative and ultimately the phase-4 design.

**Row-type taxonomy.**

- **Round sum** and **round eval** (one each per regular round), and **uni-skip sum** (one per uni-skip round) all have `B = 1`.
All linear.
- **Chain-boundary linear output** (`claim_last = Σ_j α_j · y_j`) and **PCS binding** (`V_ring = Σ_j γ^j · y_j`): `B = 1`. Linear.
- **Chain-boundary general output** (sum-of-products). `R1csConstraintVisitor` decomposes a product term `α · a_1 · a_2 · … · a_k` (where `α` is a baked challenge and `a_j` are openings) into a left-associative chain: `α · a_1 = aux_1` (linear — `α` is constant), then `aux_i · a_{i+1} = aux_{i+1}` for `i = 1, …, k−1` (quadratic — both factors are variables).
**Each k-factor term contributes 1 linear chain-start plus k−1 quadratic chain-steps.**
Batching the stage's constraints with an additional leading `α_j` preserves this linear-then-quadratic structure.
- **Initial-input binding** follows the same decomposition.

**Measurements (Jolt RV64IMAC with `feature = "zk"`).**
Instrumenting `VerifierR1CSBuilder::build` in `jolt-core/src/subprotocols/blindfold/r1cs.rs` with a per-row linearity classifier (A/B non-constant-column presence), then running the full Jolt ZK pipeline (`RV64IMACProver::prove`) on Fibonacci at `n ∈ {10, 50, 200, 1000}` and muldiv at the standard test input, the R1CS splits as:


| Guest / input | Total rounds `R` | Total rows | Linear rows | **Quadratic rows** |
| ------------- | ---------------- | ---------- | ----------- | ------------------ |
| fib `n=10`    | 228              | 2758       | 2035        | **723**            |
| fib `n=50`    | 234              | 2770       | 2047        | **723**            |
| fib `n=200`   | 240              | 2782       | 2059        | **723**            |
| fib `n=1000`  | 252              | 2806       | 2083        | **723**            |
| muldiv test   | 240              | 2782       | 2059        | **723**            |


Per-stage round counts at `n=1000`: `[uniskip₁=1, stage1=15, uniskip₂=1, stage2=27, stage3=14, stage4=21, stage5=142, stage6=27, stage7=4]`, sum `R = 252`.

**Fit.**
`L = 2·R + 1579` exactly across all five data points (the `2·R` is round sum + round eval; the constant `1579` is the linear contribution of all stage-boundary closures and PCS binding).
`Q = 723` is **constant** across trace lengths and across two different guest programs.
Total `C = L + Q = 2·R + 2302`.

**Scaling.**
`R` grows as `Θ(log T)` per stage: doubling `n` adds 1 round to each of the nine stages, so `R ≈ 9·log₂ T + O(1)`.
For a production trace length `T ≈ 2^{20}` we extrapolate `R ≈ 300–320`, giving `L ≈ 2200` and `Q = 723` (unchanged).
Quadratic density `Q / C ≈ 26%`, essentially constant.

**Why 723 and not 20.**
An earlier informal audit underestimated `Q` at 15–20 by missing that stage-boundary output constraints are **sums of many multi-factor product terms**, each 3–4 openings long.
Each k-factor term contributes `k−1` quadratic chain-step rows.
Jolt's stage closures (Spartan-outer's `Az·Bz − Cz`, per-stage batched openings, etc.) contain on the order of 200–250 such product terms total, yielding 700 quadratic rows.
This count is fixed by Jolt's stage structure and independent of trace length.

**Consequences for ZK.**

1. LNP22's batched-quadratic commit-and-prove (§5.6) handles **any number** of quadratic rows with **1 garbage commitment + 2 field elements**, via verifier-RLC-then-single-quadratic.

Whether `Q = 15` or `Q = 723` changes nothing in LNP22's cost profile. 723 is still small compared to the savings LNP22 offers against Spartan.
2. Spartan's cost (§5.2) depends on `log₂(C) + log₂(#vars) ≈ 12 + 14 = 26` sumcheck rounds regardless of the linear/quadratic split, so the split does not change Spartan's phase-4 overhead.
3. The `C = 2·R + 2302` and `Q = 723` numbers are the concrete inputs to the §9 wire-cost table and the §11.6 LNP22-vs-Spartan decision.

---

## 4. Phase 3: Hachi Recursion

Phase 3 runs Hachi's full recursion on the `Com_pre` witness.
Each recursion level has its own stage-1 and stage-2 sumchecks, masked with committed pads (§4.1).
Per-level `next_w_eval` is propagated through the recursion in masked form (§4.4).

Two levels do extra work:

- **Level `L − 1`** commits `Com_aux1` (Jolt R1CS aux variables, §2.1), runs Spartan outer as a separate masked sumcheck, and fuses Spartan inner with its own stage-2 sumcheck via RLC (§5.3). This reduces Jolt's full R1CS to one evaluation claim `v_Spartan = Z(r_y)` on the level-`L − 1` extended witness plus one residual quadratic `Az(r_x) · Bz(r_x) = Cz(r_x)` deferred to phase 4.
- **Level `L`** is an ordinary fold absorbing `Com_aux1` into the tail witness (§5.4).

The recursion terminates at a small tail witness, a pile of linear residual functionals, and exactly one residual quadratic, all discharged by the phase-4 joint tail sigma (§6) with its LNP22 single-quadratic add-on (§5.5).

### 4.1 Per-level sumcheck masking (committed pads, batched residual)

**Setup.** Consider one sumcheck of `F(X_1, …, X_n)` over `{0, 1}^n` with public initial claim `T = Σ_{x ∈ {0,1}^n} F(x)` and round degree `d`. (For Hachi's recursion, `F` is the stage-1 norm polynomial or stage-2 fused relation at one level.) The honest round polynomial in round `i` is

`F_i(X) = Σ_{x ∈ {0,1}^{n-i}} F(r_1, …, r_{i-1}, X, x)`

and satisfies the consistency chain

`F_1(0) + F_1(1) = T`,  `F_i(0) + F_i(1) = F_{i-1}(r_{i-1})` for `i > 1`.

**Committed pads.** Commit `d + 1` fully-free pad coefficients `{ρ_i[0], …, ρ_i[d]}` per round in `Com_pre` (for Hachi's recursion, these are level-local pad slots in that level's witness vector). No constraint is imposed at sampling time.

**Prover.** In round `i`, send the full masked round polynomial `g̃_i(X) = F_i(X) + ρ_i(X)` in the clear (all `d + 1` coefficients). The pads are never opened individually.

**Verifier.** Draws `r_i` from the transcript as usual and tracks the masked running claim

`c_0 = T`,  `c_i = g̃_i(r_i)` for `i ≥ 1`.

**The verifier does not check `g̃_i(0) + g̃_i(1) = c_{i-1}` during the sumcheck rounds.** Instead, after all `n` rounds, the verifier derives a transcript scalar `s` and checks a single batched identity at the tail:

`Σ_{i=1}^{n} s^i · Res_i  =  L_mask(ŵ)`                                    (∗)

where the residuals are

`Res_1 = g̃_1(0) + g̃_1(1) − T`,  `Res_i = g̃_i(0) + g̃_i(1) − g̃_{i-1}(r_{i-1})` for `i > 1`,

and `L_mask` is a linear functional of committed pad coefficients with public, transcript-derived weights. Expanding residuals in the honest case (`F_i(0) + F_i(1) − F_{i-1}(r_{i-1}) = 0`) gives

`Res_1 = ρ_1(0) + ρ_1(1) = 2·ρ_1[0] + Σ_{k≥1} ρ_1[k]`,
`Res_i = ρ_i(0) + ρ_i(1) − ρ_{i-1}(r_{i-1}) = 2·ρ_i[0] + Σ_{k≥1} ρ_i[k] − Σ_k ρ_{i-1}[k] · r_{i-1}^k`  for `i > 1`,

so `L_mask(ŵ) = Σ_i s^i · Res_i` is a specific, public linear combination of committed pad coefficients with weights derived from `(s, r_1, …, r_{n-1})`. An honest prover automatically satisfies (∗). The tail sigma protocol (§6) discharges (∗) as one of its linear relations.

**Cost.** Transmission per round: `d + 1` cleartext field elements (vs. `d + 1` non-ZK; no delta if Hachi's non-ZK transcript sends the full polynomial, `+1` per round if non-ZK uses CompressedUniPoly-style compression). Committed coefficients per round: `d + 1`, inside the per-level Ajtai commitment, well inside LHL-hiding margin. Wire cost of the batched residual check: 0 bytes (it folds into `e_y` at the tail, §6).

### 4.2 Hiding: perfect OTP per round, with simulator

**Claim (per-round OTP hiding).** For each round `i`, conditioned on the full transcript up to and including round `i − 1`, the marginal distribution of `g̃_i` is uniform over the full `(d+1)`-dimensional space of degree-`d` polynomials over `F_q`.

**Proof.** `g̃_i = F_i + ρ_i`. `F_i` is a deterministic function of `F` and the challenges `(r_1, …, r_{i-1})`, hence independent of `ρ_i`. The pad `ρ_i` has `d + 1` i.i.d. uniform coefficients sampled at commit time and embedded in an LHL-hiding Ajtai commitment (§2.2), so `ρ_i` is statistically uniform on `F_q^{d+1}` conditioned on the transcript. Adding the constant `F_i` preserves uniformity. ∎

**Simulator.** Given public `(T, r_y, v_final)` and oracle programming power:

1. For `i = 1, …, n`: sample `g̃_i` uniformly from the space of degree-`d` polynomials over `F_q`. Program the random oracle so that, after absorbing `g̃_i`, the next challenge equals the desired `r_i`.
2. Compute `c_n = g̃_n(r_n)` (the masked final claim propagated to the tail).
3. Run the tail sigma protocol's simulator (§6) with `c_n` as the first-message evaluation-claim input.

**Indistinguishability.** The honest `g̃_i` distribution is uniform on `F_q^{d+1}` by the claim above; the simulator's is the same. Identical distributions round by round. The only statistical gaps come from (a) LHL hiding of Ajtai commitments (≤ `2^{-128}` margin, §2.2) and (b) rejection-sampling closeness at the tail (`2^{-128}`, §6). No PRG or MLWE assumption enters.

### 4.3 Soundness: batched residual check

**Claim.** If the malicious prover's transcript passes verification (including the tail discharge of (∗)), then there exists an extractor that outputs a witness `ŵ*` (including pad coefficients `{ρ_i^*[k]}`) satisfying the full honest sumcheck reduction chain, with probability at least `ε − n/q` where `ε` is the prover's success probability and `n` is the number of rounds.

**Proof sketch.** Run the tail sigma protocol's special-soundness extractor (§6) on two accepting transcripts with the same first message but different challenges; this extracts committed witness coefficients `ŵ*`, including `{ρ_i^*[k]}`. Define `F_i^*(X) = g̃_i(X) − ρ_i^*(X)`. By the definition of `L_mask` and the extractor's correctness on (∗):

`Σ_i s^i · (ρ_i^*(0) + ρ_i^*(1) − ρ_{i-1}^*(r_{i-1}))  =  Σ_i s^i · (g̃_i(0) + g̃_i(1) − g̃_{i-1}(r_{i-1}))`.

Substituting `g̃_i = F_i^* + ρ_i^*` on the RHS gives

`Σ_i s^i · (F_i^*(0) + F_i^*(1) − F_{i-1}^*(r_{i-1}))  =  0`       (with `F_0^*(r_0) := T`).

`s` is drawn after all `g̃_i` and the pad commitments are fixed. By Schwartz-Zippel, either every term `F_i^*(0) + F_i^*(1) − F_{i-1}^*(r_{i-1})` is individually zero (i.e., the extracted `{F_i^*}` satisfies the honest sumcheck chain), or the bad event has probability at most `n/q`. The first case, combined with the tail's eval check, gives `F(r_1, …, r_n) = c_n − ρ_n^*(r_n)` by Schwartz-Zippel on the multilinear extension of `F^*`, closing the extraction. ∎

`n/q` loss is negligible for `q ≈ 2^128` and `n ≤ \~500`.

### 4.4 Propagation: `s_claim` and `next_w_eval` through recursion

The masked running claim `c_n = g̃_n(r_n) = F(r_1, …, r_n) + ρ_n(r_n)` includes only a single-round mask contribution, `ρ_n(r_n)`, because the chain relations batched in (∗) telescope all earlier contributions. (Specifically, if the honest chain holds extractor-side, the only un-cancelled mask term at round `n` is `ρ_n(r_n)`.)

This makes propagation across Hachi levels clean. Concretely:

- **Stage-1 reduced claim.** At one level, stage 1 produces `c_n^{(1)} = s_claim + ρ_n^{(1)}(r_n^{(1)})` where `ρ_n^{(1)}` is the last pad of stage 1. The verifier absorbs `c_n^{(1)}` (not `s_claim`) and feeds it into the stage-2 challenge derivation.
- **Stage-2 reduced claim.** Stage 2 produces `c_n^{(2)} = next_w_eval + ρ_n^{(2)}(r_n^{(2)})`. This masked value becomes the next level's evaluation claim, unchanged.
- **Recursion.** At each level `ℓ`, the masked reduced claim `next_w_eval_masked^{(ℓ)}` is what's published; the next level's witness now contains pads that include this level's pads (propagated via Hachi's ring switching, which transports committed slots across folds) plus one cross term accounting for `ρ_n^{(2, ℓ)}(r_n^{(2, ℓ)})`.

No level ever reveals an unmasked `s_claim` or `next_w_eval`.

At the tail, the cumulative mask contribution across all levels and stages, `σ_total = Σ_{ℓ, s} ρ_n^{(s, ℓ)}(r_n^{(s, ℓ)})`, is a specific linear functional of the tail witness coordinates (since all pads end up inside the tail witness through folding). This linear functional folds into the sigma protocol's `e_y` slot alongside the batched residual (∗) and the PCS evaluation claim. All three are one combined functional; no extra field element is sent for the mask discharge.

### 4.5 Ring dimension transition D=32 → D=64

Hachi's minimum production folding dim is D=32 (`src/protocol/dispatch.rs:114`).
D=32 has no production sparse-challenge family (`docs/fourth-root-verifier.md:170-179` shows only test-only Uniform(w=3) with `|C| ≈ 2^{15}`), so the sigma protocol of §6 cannot be instantiated at D=32 with 128-bit soundness; the norm blowup from a dense D=32 family would also be impractical (same failure mode as the retired D=16 attempt).

The fix is to promote the tail witness to D=64 before phase 4:

1. **Zero-padding for divisibility.** `commit_w` (`ring_switch.rs:498-589`, specifically the assertion at `ring_switch.rs:521-527`) requires `w.len() % D == 0`.

The D=32 tail `w_len` is a multiple of 32 but not always of 64; across the three 128-bit scenarios in `docs/proof-size-reduction-study.md`, tail `w_len` is 81,152 / 79,872 / 81,664, two of which need 32 zero-digits of padding.
Zero-pad up to the next multiple of 64.
2. **Non-monotone `d_at_level`.** Current production schedules are monotone-decreasing (e.g. 64→32 or flat 32-everywhere, `docs/proof-size-reduction-study.md:80-108`).
Going 32→64 requires `d_at_level` to support non-monotone transitions.
The `CommitmentConfig` trait supports this (overridable per-level), but no production config currently implements it.
3. **Dispatch table.** Already supports D=64 (`src/protocol/dispatch.rs:113-114`, `SUPPORTED_RING_DIMS = [32, 64, 128, 256, 512, 1024]`).
No new entries needed.
4. **Extra D=64 folding levels.** After the switch, fold additional levels at D=64 until `w_len` reaches `O(few thousand)` FE (working anchor: 4 levels bring `w_len` from 80K to 5K FE; exact count depends on the parameter planner, Open Question §11.2).
5. **Sigma protocol challenge.** The sigma protocol's challenge does not have to coincide with `stage1_challenge_config`; the D=64 SplitRing family is the natural choice, but a dedicated `tail_challenge_config` could be defined independently (e.g. D=128 Uniform(w=31) with `l1mass = 31` if deeper rings are preferred).

Whether this boundary needs a dedicated ring-switching *proof* or a bare re-commit is Open Question §11.4.

---

## 5. Fused Spartan at Hachi Level L − 1

Phase 3's recursion includes two special levels: level `L − 1`, where Spartan gets fused into Hachi's per-level sumcheck, and level `L`, which absorbs the Spartan-induced commitment `Com_aux1` into the tail witness.
Everything else in phase 3 (levels `1` through `L − 2`) is vanilla Hachi folding with masked sumchecks (§4).

Spartan's role here is to **reduce the R1CS for Jolt's verifier algebra** (2,900 constraints; 723 of them quadratic; §3.3, §3.4) to:

- a bundle of linear residuals, folded into phase 4's joint sigma alongside everything else linear; and
- exactly **one** residual quadratic identity on three transcript-visible scalars, discharged at phase 4 by LNP22's single-quadratic add-on (§5.5).

### 5.1 Why level `L − 1`, and not deferred to the tail

The earliest draft of this note ran Spartan at the Hachi tail, after all recursion finished (`Com_aux1` committed once, Spartan sumchecks run against it, reduced claim folded into the final sigma).
That layout is conceptually simple but has a proof-size problem: Jolt's verifier R1CS has on the order of a few thousand aux variables (full-field `F_q` elements), and a commitment `Com_aux1` holding them at the tail roughly **doubles the effective tail witness** going into the final sigma protocol.

(The non-ZK direct-tail is 80,000 gadget digits ≈ 2,500 F_q equivalent at `lb = 4`.
A fully-deferred `Com_aux1` of 3,000 full F_q elements doubles that.)

The fix is to run Spartan early enough that the subsequent Hachi folds **shrink `Com_aux1` along with the main witness**.
There are four positions on the table:


| Position                                 | Folds applied to `Com_aux1` | `Com_aux1` contribution at tail | Per-level commit-matrix bloat            |
| ---------------------------------------- | --------------------------- | ------------------------------- | ---------------------------------------- |
| **Upfront Spartan** (before Hachi)       | N/A — infeasible            | N/A                             | N/A                                      |
| Early Spartan (level `L − 2` or earlier) | ≥ 2                         | `≤ 3000 F_q / D²` ≈ `≤ 3 F_q`   | Every level pays 96K extra gadget digits |
| **Level `L − 1` (adopted)**              | 1                           | `3000 / 32 ≈ 94 F_q`            | One level pays 96K extra gadget digits   |
| Tail-deferred (earlier draft)            | 0                           | `\~3000 F_q` (doubles tail)     | None                                     |


Upfront is infeasible because `Com_aux1`'s contents depend on challenges drawn during phase 2.
Tail-deferred balloons the tail witness.
Early Spartan shrinks `Com_aux1` more but pays in commit-matrix bloat on every intermediate level.
Level `L − 1` is the latest point that still gets at least one fold on `Com_aux1` and the earliest point that commits `Com_aux1` after all necessary challenges exist.
One level of commit-matrix bloat on level `L − 1` only; rest of the recursion unchanged.

### 5.2 `Com_aux1` layout and content

At the boundary between level `L − 2` and level `L − 1` (immediately after the level-`L − 2` ring-switch, before level `L − 1`'s sumchecks begin), the prover issues a second Ajtai commitment `Com_aux1`.

**Contents** (identical to the earlier tail-`Com_aux1` list; only the commit time changes):

- **Jolt per-round aux.** `{c_1_aux_{s,i}}` (one per regular Jolt round, 330 values) and `{c_0_aux_{s,i}}` (one per uni-skip round, 2 values) — the coefficients dropped from transmission in phase 2 (§3.2).
- **Chain-boundary aux.** `{next_claim_{s,i}}`, `{y_j}` (batched evaluation outputs across stages), `{initial_claim_vars}` — one cluster per stage-junction, O(stages) total.
- **Sum-of-products aux.** Intermediate products from chain-general output constraints (Spartan-outer's `(Az) · (Bz) − Cz` closure and similar), plus any multiplication rows the `R1csConstraintVisitor` emits.
- **PCS-binding scalars.** `V_ring` (or a small vector of per-point `V_ring_j`; §11.1).

Total size: a few thousand full-field `F_q` elements.

**Layout.**
`Com_aux1` is **merged into Hachi's mega-polynomial at level `L − 1`**: its F_q contents are gadget-decomposed at the level-`L − 1` log-basis `lb` into `\~target × (128 / lb)` short-norm digits (e.g. 3,000 F_q at `lb = 4` gives 96K digits ≈ 100 ring elements at `D = 32`), and those digits occupy a fresh block appended to the level-`L − 1` Hachi witness.
Level `L − 1`'s Ajtai commit matrices extend column-wise to cover the new block, so `Com_aux1`'s commitment is computed alongside the main-witness commitment using the same level-`L − 1` matrix set (`A`, `B`, `D` of §2.2.2).

Two consequences.

- `Com_aux1` inherits the same LHL-hiding argument as the main witness (§2.2), with blinding columns sized by the level-`L − 1` `(n_b, n_d)` profile.
- Since `Com_aux1` lives in the level-`L − 1` variable space, its MLE variables coincide with a sub-layout of Hachi's level-`L − 1` witness MLE variables. **This is what makes Spartan-inner-into-Hachi-stage-2 fusion possible** (§5.3).

This closes the earlier open question (§11.5) in favor of the merged layout: for `Com_aux1` specifically, merging is not an optimization but a structural requirement for fusion.

### 5.3 The three sumchecks at level `L − 1`

Three sumchecks run at level `L − 1`, all masked by pads pre-committed in `Com_pre` (§2.3, §4.1).
They are:

**(a) Spartan outer** — independent sumcheck over R1CS row indices.

Reduces `Σ_x eq(τ, x) · (Az(x) · Bz(x) − Cz(x)) = 0` to three scalar claims `Az(r_x), Bz(r_x), Cz(r_x)`.
`log₂(C) ≈ 12` rounds (`C = L + Q ≈ 2,900`, §3.4), round degree 3 (eq × quadratic residual).
Variables are R1CS row indices, which have no alignment with Hachi's level-`L − 1` witness coordinates, so outer runs as its own sumcheck at the start of the level.
Round polys masked per §4.1; residuals accumulate into the Spartan cluster's batched identity with its own transcript scalar `s_outer`.

**(b) Hachi stage-1** — unchanged from the non-ZK path, except for committed-pad masking per §4.1.

**(c) Fused `[Hachi stage-2 + Spartan inner]` — one combined sumcheck via RLC.**

Hachi stage-2 closes Hachi's per-level fold (degree 2, `log₂(level-L−1 witness length)` rounds).
Spartan inner reduces the three row-level claims `(Az(r_x), Bz(r_x), Cz(r_x))` from outer to a single MLE evaluation `v_Spartan = Z(r_y)` on the R1CS witness `Z = (Com_pre portion of level-L−1 witness) ∪ (Com_aux1)` (degree 2, `log₂(#R1CS cols) = 14` rounds).

Because `Com_aux1` was merged into the level-`L − 1` mega-polynomial (§5.2), both sumchecks live in the **same variable space** (level-`L − 1` witness coordinates) and share the same round degree (2).
A transcript scalar `α_fused` (drawn after stage-1 finishes) combines them into one sumcheck whose round-`i` polynomial is

```
g̃_i^{fused}(X) = g̃_i^{stage-2}(X) + α_fused · g̃_i^{inner}(X)      (both masked via §4.1)
```

Round count is `max(stage-2 rounds, 14)`; the shorter sumcheck's variables are treated as fully reduced for the remaining rounds.

**Net round cost vs. running Spartan inner after Hachi stage-2 sequentially.**
Sequential: `stage-2 rounds + 14`. Fused: `max(stage-2 rounds, 14)`.
For typical level-`L − 1` witness sizes (where stage-2 rounds and 14 are comparable), the saving is `≈ 12–14` rounds.
Pads saved: `≈ 14 × (d + 1) ≈ 40–80` F_q slots in `Com_pre`.

**Level-`L − 1` outputs.**

- From outer: three transcript-visible scalars `Az(r_x), Bz(r_x), Cz(r_x)`, with the honest identity `Az(r_x) · Bz(r_x) = Cz(r_x)`. This is the **one residual quadratic**; discharged at phase 4 by §5.5.
- From stage-1: reduced claim `c_n^{stage-1}` (masked).
- From fused stage-2/inner: reduced claim `c_n^{fused} = next_w_eval + α_fused · v_Spartan + mask`, which threads into level `L` as the next-level Hachi claim.

### 5.4 Level `L` absorbs `Com_aux1`

Level `L` is an ordinary Hachi fold, no Spartan content.
It operates on the level-`L − 1` extended witness (main + `Com_aux1` block) and its own stage-1 and stage-2 sumchecks are masked by phase-1 pads.

The only thing special: the fused eval claim `c_n^{fused}` from level `L − 1` (which mixes `next_w_eval` and `v_Spartan` through `α_fused`) propagates as level `L`'s initial claim via standard Hachi claim-chaining.
Both components transform through the fold as linear functionals; by the time level `L` finishes, they sit as **linear** functionals of the final tail witness, combinable with every other phase-4 linear row.

After level `L`, the prover and verifier hold:

- Direct tail witness `ŵ_tail`: the main Hachi tail (80,000 gadget digits).
- `Com_aux1`'s folded contribution inside `ŵ_tail`: `3000 / 32 ≈ 94` F_q ≈ `3` D=32 ring elements. Negligible vs. the 2,500-F_q-equivalent main tail.
- Linear residuals from every masked-sumcheck cluster (Jolt phase 2, per-level Hachi in phase 3, Spartan outer at level `L − 1`, fused stage-2/inner at level `L − 1`, Hachi levels `L − 1` stage-1 and level `L`), each with its own cluster-batching scalar.
- Exactly one residual quadratic: `Az(r_x) · Bz(r_x) = Cz(r_x)`.

### 5.5 Residual-quadratic discharge: LNP22 single-quadratic

The Spartan post-outer residual quadratic is discharged at phase 4 by LNP22's single-quadratic add-on (Nguyen thesis §5.2.1, LNP22 Eqs. (8)-(9)).
The construction below specifies the single-quadratic case; if additional residual quadratics surface (§11.12), they are handled identically by replicating the add-on per relation (one fresh `g_{quad}` slot in `Com_pre` and 2 extra scalars in the sigma first message per quadratic) or batched via a verifier-supplied random linear combination.

**Form of the quadratic.**
Post-fold, `Az(r_x)`, `Bz(r_x)`, `Cz(r_x)` are each a specific linear functional of the tail witness pair `z_tail = (ŵ_tail)` with publicly-known coefficients.
So `Az(r_x) · Bz(r_x) − Cz(r_x) = z_tail^T R_2 z_tail + r_1^T z_tail + r_0` for public matrices/vectors `R_2, r_1, r_0` that fall out of Spartan outer's `r_x` challenge substituted into Jolt's fixed R1CS matrices, then transformed through level `L`'s fold map.

**What was pre-committed.**
One fresh ring element `g_{quad}` in `Com_pre` (§2.1), gadget-decomposed at the tail `lb`, contributing one extra Hachi-layout block of size 1 KB after blinding.

**Sigma first message (phase 4).**
In addition to the standard sigma messages `(t_tail, e_y)` of §6, the prover sends two extra scalars:

```
g_1 := 2 · y^T R_2 z_tail + r_1^T y        (opened via g_{quad}'s blinded commitment scalar)
v   := y^T R_2 y                           (sent in cleartext; hiding argument below)
```

where `y` is the sigma's Gaussian mask vector.

**Verifier check (phase 4).**
Alongside the linear-sigma check (§6), the verifier checks

```
z_tail^T · R_2 · z_tail  +  c · r_1^T z_tail  −  c² · r_0_public   ==   v  +  c · g_1
```

where `c` is the sigma challenge and `r_0_public := Az(r_x) · Bz(r_x) − Cz(r_x) |_{honest} = 0`.
Honest satisfaction is direct substitution `z_tail = c · w_tail + y`.

**Hiding of `v`.**
`v = y^T R_2 y` is a quadratic functional of the Gaussian mask `y`.
By the standard LNP22 argument (§5.2.1), `v` is statistically uniform on `F_q` mod a public affine shift conditioned on the transcript, provided `y`'s entropy exceeds the quadratic functional's output entropy budget.
At Hachi's sigma parameters (D=64, σ ≈ 5,616·β_tail'), `y`'s entropy is ample; the LHL argument is the same shape as §2.2's Ajtai-commitment hiding, applied to a different quadratic functional of the mask vector.

**Soundness of the quadratic add-on.**
If `Az(r_x) · Bz(r_x) ≠ Cz(r_x)`, the verifier's quadratic check fails except with probability `q_1^{−D/l}` where `q_1` is the smallest prime factor of `q` and `l` is the number of irreducible factors of `X^D + 1` mod `q`.
Hachi's design constrains `l ≤ 8` (at most 2, 4, or 8 factors, never fully-split; §11.6), so this is `≤ q^{−8} = 2^{−1024}`. Negligible.

**Cost summary.**


| Item                               | Cost                                                |
| ---------------------------------- | --------------------------------------------------- |
| `Com_pre` extension                | 1 ring element for `g_{quad}` (1 KB after blinding) |
| Sigma first message extension      | 2 scalars (32 B)                                    |
| Extra sumcheck rounds              | 0                                                   |
| Extra Ajtai commitments at phase 4 | 0                                                   |
| Soundness loss                     | ≤ 2^{−1024}                                         |


### 5.6 Fallback: small second Spartan on the residual

If the LHL argument for `v` turns up implementation obstacles in the Ajtai-only setting (e.g. a non-standard quadratic form from the fold map), a fallback is a **small second Spartan** invocation proving `Az(r_x) · Bz(r_x) = Cz(r_x)` alone.

The residual R1CS has 3 variables and 1 quadratic constraint: `log₂(3) ≈ 2` outer rounds + `log₂(3) ≈ 2` inner rounds, so roughly 4 masked sumcheck rounds with their own pads (20 F_q extra in `Com_pre`) and a 5-field-element additional residual into the tail sigma.
No LNP22 LHL argument needed; reuses the masked-sumcheck machinery of §4.1 verbatim.

Wire-cost delta vs. LNP22: 80 B (five scalars instead of two).
Available as a drop-in fallback.

This is the "second layer of Spartan" from earlier architectural sketches, now reduced to a 3-variable instance (not another run over the full R1CS).

### 5.7 Soundness

**Claim.** The fused-Spartan pipeline is sound provided:
(a) each masked-sumcheck cluster's batched residual is discharged by the tail sigma (§4.3);
(b) phase 4's sigma extracts the tail witness via special-soundness (§6);
(c) LNP22's quadratic add-on (§5.5, or §5.6's fallback) catches any cheat on the residual quadratic.

**Proof sketch.**
Run the sigma extractor on two accepting transcripts: recovers `ŵ_tail` (including `Com_aux1`'s folded contribution).
Invert Hachi's recursion levels `L, L − 1, …, 1` to recover `(Com_pre, Com_aux1)` in cleartext.

1. The Spartan-outer cluster's batched-residual discharge forces extracted outer round polynomials to satisfy the honest sumcheck chain (§4.3 applied to the outer cluster, with `12 / q` loss).
2. The fused `[stage-2 + inner]` cluster's batched-residual discharge forces both stage-2 and inner extracted round polys to satisfy their individual honest chains (two linearly-independent identities on the same transcript, batched under `α_fused` and the cluster scalar).
3. LNP22's quadratic add-on (§5.5) forces the extracted outer outputs to satisfy `Az(r_x) · Bz(r_x) = Cz(r_x)`, which combined with (1) gives honest R1CS satisfiability at `r_x`.
4. Spartan inner's honest chain + the extracted Z gives `v_Spartan = Z(r_y)` on the extracted witness; combined with Jolt's R1CS encoding of phase-2/3 verifier algebra, the full verifier check is satisfied on the extracted witness.

Net soundness loss: `Σ_clusters (n_cluster / q) + q_1^{−D/l}` ≈ `(500 / 2^{128}) + 2^{−1024}`. Negligible.

### 5.8 Hiding

Per-round OTP hiding (§4.2) applies to every masked sumcheck at level `L − 1` (outer, stage-1, fused stage-2/inner) identically to phase-2 Jolt rounds.
`Com_aux1`'s Ajtai commitment is LHL-hiding via dedicated blinding (§2.2), inheriting the level-`L − 1` `(n_b, n_d)` blinding profile.
`v = y^T R_2 y` is statistically uniform in `F_q` by the LHL argument in §5.5.
`g_{quad}`'s LHL hiding at `Com_pre` matches every other wire-visible Ajtai output.
No new security assumption enters.

### 5.9 Wire-cost summary

Cost delta vs. fully-deferred Spartan (which has been dropped as an option):


| Item                                    | Fully-deferred Spartan (rejected) | Fused Spartan at `L − 1` (adopted)                                                    |
| --------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------- |
| `Com_aux1` contribution to tail witness | 3,000 F_q (full width)            | 94 F_q (folded once)                                                                  |
| Spartan sumcheck rounds                 | 26 (outer 12 + inner 14)          | **12 (outer only; inner fuses into Hachi stage-2, saving 12–14 round-worth of pads)** |
| `Com_pre` pad budget for Spartan        | 25 · (d+1) ≈ 100 scalars          | 12 · (d+1) ≈ 50 scalars + 1 garbage ring (§5.5)                                       |
| Residual-quadratic discharge            | Absorbed by Spartan's own chain   | 1 garbage ring + 2 scalars at sigma (§5.5)                                            |
| Mid-recursion commit matrix bloat       | None                              | Level `L − 1` only: 96K extra gadget digits (100 ring elements at D=32)               |


Net proof-size saving vs. fully-deferred: roughly `3000 − 94 = 2906 F_q ≈ 45 KB` less tail witness, offset by 1 KB of `g_{quad}` in `Com_pre` and 1 KB of `Com_aux1` on-wire commitment scalar at level `L − 1`.
Ballpark net savings: 40 KB on the tail sigma response, 0 change on the transcript mid-section (round count goes down, not up, thanks to fusion).

---

## 6. Phase 4: Joint Tail Sigma Protocol

By this point `Com_aux1` has been folded into level `L − 1`'s witness (§5) and flowed through level `L`'s fold, so at the tail there is **one** witness — not two.
The tail sigma only has to open the tail commitment, discharge a pile of linear functionals on the tail witness, and discharge a small set of quadratic residuals (the explicitly accounted-for one is the Spartan outer `A · B = C` claim, §5.5; potentially a few more pending the §11.12 audit).

**State at tail entry.**

- Tail witness `w` with `‖w‖_∞ ≤ β_tail'` (Open Question §11.2), gadget decomposed `ŵ = G^{-1}(w)`, committed as `u_tail = M_tail · ŵ`.
- Two transcript-public scalars `a_pub, b_pub, c_pub` claimed to satisfy `a_pub · b_pub = c_pub` (the residual Spartan outer quadratic, §5.5). In the LNP22 add-on `a_pub`, `b_pub`, `c_pub` are each a linear functional of `ŵ` (with `g_quad` offset for one of them), which the sigma proof reveals masked.
- Collected linear functionals on `ŵ`:
  - Hachi-recursion evaluation claim `⟨w, eq(·, r_final)⟩ = v_masked` (§4.4).
  - Batched residuals `Σ_p s_p · L_mask^{(p)}(ŵ)` from masked-sumcheck clusters in phase 2 (Jolt), phase 3 (Hachi recursion), and phase 3's level-`L − 1` Spartan-outer and fused-stage-2 sumchecks.
  - PCS binding `V_ring = Σ_j γ^j · y_j`.
  - `y_ring` batched-residual rows (§2.4).
  - The single linear row that pins `g_quad`'s constant coefficient (LNP22 add-on, §5.5).

Every item except the one `a_pub · b_pub = c_pub` relation is **linear in `ŵ`**.
Linear items fold for free (one extra field element in `e_y` each).
The quadratic item costs LNP22's single-quadratic add-on: one extra garbage ring element `g_quad` (already in `Com_pre`, §2.1) and two extra transcript scalars.

**Protocol.**

1. Prover samples tail mask `y ← D_σ^{n_tail}` over `R_q` with `D = 64`.
2. Prover computes:
  - `t = M_tail · G^{-1}(y)` (mask commitment on the tail witness).
  - `e_y = L_joint(y)` for the sum of all linear functionals above evaluated on `y`.
  - LNP22 quadratic prelude: let `α = L_a(y)`, `β = L_b(y)` be the values of the linear functionals defining `a_pub` and `b_pub` on the mask, and `g_0 = ct(g_quad)` (committed). Define `h = α · β` and send `h` plus `g_quad`'s masked evaluation `μ = L_g(y) + g_0`.
3. Prover sends `(t, e_y, h, μ)`.
4. Transcript derives challenge `c` using `D = 64` `SparseChallengeConfig` (SplitRing, `weight = 21`, `maxmag = 6`, `l1mass = 54`, `|C| ≈ 2^{130}` per `docs/fourth-root-verifier.md:170-179`).
5. Prover computes `z = c · ŵ + y` and the LNP22 quadratic response `z_q = c · (a_val · b_val − c_val) + linear_combo(α, β, c)` per LNP22 Eq. (7)–(8) with `γ` dropped (soundness: §5.5).
6. Prover rejection-samples `(z, z_q)` (default Rej1; alternatives in Open Question §11.3).

On abort, restart from step 1.
7. Prover sends `(z, ẑ, z_q)` with `ẑ = G^{-1}(z)`.

**Verifier checks.**

   (a) `M_tail · ẑ == c · u_tail + t`.
   (b) `L_joint(z) == c · T_public + e_y`, where `T_public` is the transcript-computable target aggregating `v_masked`, `v_Spartan`, `Σ_p s_p · Res_p^{public}`, `V_ring`'s zero target, `y_ring` residual pins `δ^{(ℓ)}`, and the `g_quad` constant-coefficient pin.
   (c) LNP22 quadratic check: `L_a(z) · L_b(z) − c · L_c(z) == c^2 · 0 + c · linear_cross + h` up to the round-off defined by LNP22 Eq. (8).
   (d) `‖z‖_∞ ≤ B_z`.
   (e) `ẑ == G^{-1}(z)` (gadget well-formedness).

Check (b) holds honestly by linearity: every linear functional commutes with `z = c · w + y`.
Check (c) is LNP22's single-quadratic verification specialized to our commit-and-prove setting (see §5.5 for the `γ` drop and soundness argument).

**Multiple linear relations for free.** Adding another linear functional adds one field element to `e_y` and nothing else.
This is why §4.1's batched residuals, §3.3's `V_ring` binding, §5.3's `v_Spartan`, and §2.4's `y_ring` batched-residual rows are all absorbed without blowing up the proof.

**Gaussian width (Rej1 default).**

`σ = 13 · ‖c · w‖_2`, `‖c · w‖_2 ≤ l1mass · β_tail' · √D = 54 · β_tail' · 8 = 432 · β_tail'`,
`σ = 5,616 · β_tail'`.

For working-anchor `β_tail' ≈ 2^{12}`: `σ ≈ 2^{25}`, `B_z ≈ σ · √(D · n_tail) ≈ 2^{30.6}` (for `n_tail ≈ 78`).

**Norm blowup and MSIS binding.** Extracted tail-witness norm `B_extract = 2 · l1mass · B_z ≈ 2^{37.3}`.
MSIS hardness requires `B_extract < q^{2/3} ≈ 2^{85}` at `n_A = 2`.
Holds with 48-bit margin.

**Special soundness extracts the witness.** Given two accepting transcripts with the same first message but different challenges `c ≠ c'`, subtracting check (b) yields `L_joint(Δz) = Δc · T_public`.
The extracted `ŵ^* = (Δc)^{-1} · Δz` satisfies every linear relation simultaneously, including the joint Ajtai openings of all preceding commitments (`Com_pre` and `Com_aux1` via folding inversion).
For the quadratic, LNP22's two-transcript argument (Eq. (9)) extracts `a^*, b^*, c^`* with `a^* · b^* = c^* + ct(g_quad^*)` — and because the tail sigma's linear rows pin `ct(g_quad^*) = 0`, the extracted values satisfy `a^* · b^* = c^*` exactly.

---

## 7. Verifier's Checks (consolidated)

The verifier never reconstructs `Z` (the full R1CS witness).
End-to-end, across the four phases of §1.1:

1. **Phase 1.** Receives `Com_pre`; absorbs it into the transcript.
2. **Phase 2 (Jolt sumchecks).** Absorbs the 333 rounds' masked coefficients `{m'_{s,i,k}}` and draws challenges `{r_{s,i}}`.

Does *not* algebraically validate round sums or evals — all checks are recorded and deferred.
3. **Phase 3 (Hachi recursion with fused Spartan at level `L − 1`).**

- For every level `ℓ` the verifier absorbs masked stage-1 and stage-2 sumcheck messages `{g̃_{s,i}^{(ℓ)}}`, masked `next_w_eval^{(ℓ)}`, and the masked `y_ring^{(ℓ)}` (§2.4).
- At the boundary between levels `L − 2` and `L − 1` the verifier receives `Com_aux1` and absorbs it.
- At level `L − 1` the verifier additionally absorbs Spartan's outer sumcheck messages (independent run) and the fused `[Hachi stage-2 + Spartan inner]` sumcheck messages, deriving Spartan's `r_x, r_y` along the way.
- Level `L` is a standard fold absorbing `Com_aux1` into the tail.
- No per-round algebraic check anywhere; everything is recorded for the tail.

1. **Phase 4 (joint tail sigma).** A single sigma protocol proof closes:
  - Ajtai opening well-formedness for the tail witness: `M_tail · ẑ == c · u_tail + t`.
  - Hachi tail evaluation claim `v_masked` on `ŵ`.
  - Spartan closing claim `v_Spartan` linearized onto `ŵ` (after level `L`'s fold of `Com_aux1`).
  - Batched round-sum residual identities (∗) for all masked-sumcheck clusters: Jolt (phase 2), Hachi per-level (phase 3), and level-`L − 1` Spartan outer + fused stage-2/inner (phase 3).
  - PCS binding `V_ring = Σ_j γ^j · y_j` on `ŵ`.
  - Per-level `y_ring` residual pins `δ^{(ℓ)}` (§2.4).
  - LNP22 single-quadratic discharge of the residual `A · B = C` from Spartan outer (§5.5).
  - All linear rows fold into one combined `e_y`; the quadratic row costs two extra transcript scalars and one `g_quad` pin.

The verifier sees in the clear: `m'_{p,s,i,k}` (uniform by OTP in every phase), two Ajtai commitments (`Com_pre`, `Com_aux1`), per-level Hachi commitments (LHL-hiding), masked `y_ring^{(ℓ)}` per level, one joint sigma-protocol response.
It never sees: any pad coefficient, any `y_j`, any `next_claim`, `V_ring`, `s_claim`, `next_w_eval`, any sum-of-products aux, any `g_quad`, or any tail witness coordinate.

---

## 8. Simulator (end-to-end)

Given public `(Com_pre commitment, Com_aux1 commitment, initial claim T)` — none of which depend on the witness except through statistical-hiding commitments — the simulator produces a full accepting transcript.

**Hybrid argument.**

- **Hybrid 0 (real).** Honest execution.
- **Hybrid 1.** Replace `Com_pre` with a random commitment of matching dimension.
Indistinguishable by LHL statistical hiding (§2.2).
- **Hybrid 2.** Replace `Com_aux1` with a random commitment at the level-`L − 1` profile (§2.1).
Indistinguishable by the same LHL argument.
- **Hybrid 3.** Replace every absorbed `y_ring^{(ℓ)}` with a ring element drawn uniformly in `R_q` (all `D` coefficients independently uniform in `F_q`).
**Perfectly** indistinguishable by the LNP22 coefficient masking of §2.4 in batched-residual form: `g^{(ℓ)}` is committed uniformly in `R_q` (§2.1) with no sampling-time constraint, so `y_ring^{(ℓ)} = y_ring_orig^{(ℓ)} · σ_{-1}(v^{(ℓ)}) + g^{(ℓ)}` is uniform in `R_q` conditioned on the transcript up to its absorption.
The per-level residual `δ^{(ℓ)} := ct(y_ring^{(ℓ)}) − opening^{(ℓ)}` the simulator induces is a uniform `F_q` scalar, exactly matching the real distribution (where `δ^{(ℓ)} = ct(g^{(ℓ)})` with `g^{(ℓ)}` uniform).
- **Hybrid 4.** Replace every masked round polynomial with a uniformly-sampled degree-`d_{p,s,i}` polynomial, across every masked-sumcheck cluster: Jolt's 333 rounds (phase 2), every Hachi per-level sumcheck round (phase 3), and the level-`L − 1` Spartan outer + fused stage-2/inner rounds (phase 3 §5.3).
**Perfectly** indistinguishable by §4.2's per-round OTP claim, applied at every round.
- **Hybrid 5.** Replace `next_w_commitment^{(ℓ)}` and `v^{(ℓ)}` at every Hachi level with random.
Indistinguishable by LHL.
- **Hybrid 6 (simulated).** Compose the previous hybrids and back-compute the joint tail sigma protocol's first message `(t, e_y, h, μ)` (§6) from the verification equations given a freshly-sampled `z ← D_σ`.
The LNP22 quadratic add-on simulates cleanly: sample `z_q` from its honest distribution, then back-solve `h` from the quadratic check equation; statistical closeness follows LNP22's simulator analysis.
Statistically `2^{-128}`-close to Hybrid 5 by rejection-sampling closeness.

**Output.** A transcript indistinguishable from honest by a union of: LHL hiding margin (statistical, `2^{-128}` per commitment slot, many times over — LHL has orders of magnitude of margin), per-round OTP hiding (perfect), and rejection-sampling closeness at the joint tail sigma (statistical, `2^{-128}`).

**Security notion.** Statistical ZK in the random oracle model for the Fiat-Shamir compiled protocol.
No PRG or MLWE assumption enters.

---

## 9. Cost Analysis

**Per-Jolt-round wire cost (phase 2, §3.2):** matches non-ZK Jolt exactly.
`d` cleartext field elements per regular round, `d_{uniskip}` per uni-skip round.
Transcript size equal to non-ZK baseline.
Net pre-tail overhead on Jolt: 0 bytes on the wire.

**Per-Hachi-level wire cost (phase 3, §4.1):** `d + 1` cleartext coefficients per round (vs. `d` in CompressedUniPoly-compressed non-ZK, if applicable).
For a typical level with 15 stage-1 rounds + 10 stage-2 rounds at degrees 7 and 1–2, this is 25 extra field elements per level, 400 bytes.
Across 7–8 levels, a few KB.
For Hachi implementations that already send full round polys, the delta is 0.

**Per-Spartan-round wire cost (phase 3 §5.3):** Spartan outer runs as an independent masked sumcheck at level `L − 1` (degree 3, 14 rounds ≈ `log₂(2·R + 1579)`); Spartan inner is fused with Hachi stage-2 at level `L − 1`, so its round cost is absorbed into the fused sumcheck (degree bumps from 2 to 3). Net additional on-wire cost: 14 extra masked round polys at the outer (≈ 56 field elements ≈ 900 B) and the stage-2 polys grow by one coefficient each (≈ 10 field elements ≈ 160 B). Ballpark 1 KB extra vs. non-fused Hachi level `L − 1`.

**Commitment delta:**

- `Com_pre` grows by `L_pad_total` ≈ 4,000–5,000 field elements (pads across every masked sumcheck round in phases 2 and 3, including Spartan outer + fused inner at level `L − 1`), plus `L` garbage ring elements `g^{(ℓ)}` for `y_ring` masking (§2.4), plus one `g_quad` ring element for the LNP22 quadratic discharge (§5.5). This is a selector slot inside Hachi's mega-polynomial layout (no extra wire cost for the commitment itself).
- `Com_aux1` is a small commitment produced at the level-`L − 1` boundary (§5.2). It contains `R_Jolt` `c_1_aux` / `c_0_aux_uniskip` slots (300 at `T = 2^{20}`, 333 at `T = 2^{24}`; §3.1), plus 50 chain-boundary outputs, plus `V_ring` (1), totaling 350 field elements. Merged into Hachi's level-`L − 1` mega-polynomial, so its on-wire commitment scalar is bundled into level `L − 1`'s `u` commitment (≈ 1–2 KB total, not separately charged).
- The verifier R1CS size is `L + Q = 2·R + 2302` rows with `R ≈ 300` at `T = 2^{20}` (§3.4); Spartan's outer sumcheck runs for 14 rounds at level `L − 1` and no further sumcheck is needed for the quadratic (discharged by LNP22 add-on at the tail, §5.5).

**Tail delta (§6, Rej1, working anchor):**


| Component                                                             | Non-ZK | With ZK                                             | Delta            |
| --------------------------------------------------------------------- | ------ | --------------------------------------------------- | ---------------- |
| Extra folding levels (D=32 → D=64 plus 4 D=64 levels)                 | —      | 16–20 KB                                            | +16–20 KB        |
| Tail `PackedDigits` (saved)                                           | 40 KB  | removed                                             | −40 KB           |
| `t` (mask commitment, one witness only)                               | —      | `n_A` ring elements ≈ 1–2 KB                        | +1–2 KB          |
| `e_y` (combined eval + Spartan + mask + Vring + `y_ring` functionals) | —      | 1 field element ≈ 16 B                              | +16 B            |
| Response `ẑ`                                                          | —      | `n_tail · lb` coefficients, 25 bits each ≈ 10–16 KB | +10–16 KB        |
| LNP22 quadratic add-on (`h`, `μ`, `z_q`)                              | —      | 2 ring elements + 1 response ≈ 1–1.5 KB             | +1–1.5 KB        |
| Mask/eval/Spartan/Vring/y_ring discharge (linear functionals)         | —      | folded into `e_y`; 0 extra field elements           | 0 B              |
| **Net tail**                                                          | 40 KB  | 29–41 KB                                            | **−11 to +1 KB** |


The non-ZK baseline (40 KB) comes from the three 128-bit scenarios in `docs/proof-size-reduction-study.md` (tail sizes 40,576 / 39,936 / 40,832 B).
Under the working anchor, ZK is proof-size-neutral to slightly smaller at the tail — the single-quadratic LNP22 add-on is cheap, and merging `Com_aux1` into level `L − 1` avoids a separate aux-witness opening.

**Total proof-size comparison:**


| Scenario                                         | Proof delta vs. non-ZK | Prover time | Verifier time |
| ------------------------------------------------ | ---------------------- | ----------- | ------------- |
| Expected (Rej1, D=64 sigma, 4 extra D=64 levels) | −3 to +8 KB            | 1.2–1.4x    | 1.1–1.2x      |
| Conservative (wide Gaussian, extra padding)      | +5 to +18 KB           | 1.4–1.6x    | 1.2–1.3x      |


Exact numbers depend on Open Questions §11.2, §11.3.

---

## 10. Parameter Analysis

### 10.1 Parameters

From `docs/proof-size-reduction-study.md` (2026-Q1 update):

- `q = 2^128 − 275` (128-bit prime modulus).
- Last production folding level: `D = 32`, `w_len ≈ 80,000` FE, `lb = 4`, non-ZK tail ≈ 40 KB.
- D=32 challenge: no production family (only test-only Uniform(w=3, ±1) with `log₂|C| ≈ 15`).
- D=64 challenge: `l1mass = 54` (SplitRing, `weight = 21, maxmag = 6`), `log₂|C| ≈ 130`.
- D=128 challenge: `l1mass = 31` (Uniform(w=31, ±1)), `log₂|C| ≈ 130`.

### 10.2 Why the naive same-D sigma protocol fails at D=32

**Primary blocker:** no production challenge family at D=32 (see `docs/fourth-root-verifier.md:170-179`). A 128-bit-soundness family at D=32 would need `(2m)^{32} ≥ 2^{128}` with small `m`, forcing a dense challenge (weight 32, large magnitudes) with `l1mass` in the thousands.

**Historical backup:** the retired D=16 attempt failed by Gaussian-width blowup. The D=16 family had all 16 coefficients nonzero with magnitudes up to 128 (`l1mass = 2048`), giving `σ ≈ 106,496 · β_tail`, `B_z ≈ 23.6M · β_tail`, `B_extract ≈ 96.8G · β_tail`, and a sigma response of 309 KB (order of magnitude larger than the non-ZK tail). The same dense-challenge blowup would hit any 128-bit-soundness family at D=32.

**Conclusion.** D=64 is the smallest dimension where the sigma protocol can run with both a production challenge family and a tractable Gaussian width.

### 10.3 D=64 sigma with deeper recursion

Starting anchor: D=32 tail `w_len ≈ 80,000` FE. Per-level reduction at D=64 2×. After 4 extra D=64 folds: `w_len ≈ 5,000` FE, `n_w = 5000 / 64 ≈ 78` ring elements.

**Rej1:** `σ = 5,616 · β_tail'`, `B_z ≈ 396,600 · β_tail'`. For `β_tail' = 2^{12}`: `B_z ≈ 2^{30.6}`, `B_extract = 108 · B_z ≈ 2^{37.3}`.
MSIS margin `q^{2/3} / B_extract ≈ 2^{47.7}` ≈ 48 bits.
Response size: `78 · lb · 64 · 25 / 8 ≈ 15.6 KB`. `t_y ≈ 1–2 KB`. Total sigma subtotal 17–18 KB.

**Rej2 (one-sided, leaks 1 bit sign):** `σ ≈ 292 · β_tail'`, `B_z ≈ 20,600 · β_tail'`. For `β_tail' = 2^{12}`: `B_z ≈ 2^{26.9}`, `B_extract ≈ 2^{33.6}`. Response size 10 KB. Total tail 27–31 KB.

**Iterative RS (partial applicability):** `σ ≈ 1,694 · β_tail'`, `B_z ≈ 2^{28.8}`, `B_extract ≈ 2^{35.5}`. Response size 12 KB. Total tail 29–34 KB. Negligible abort probability.

### 10.4 Summary table


| Strategy                    | `σ / β_tail'` | `B_z` (β_tail'=2^12) | `B_extract` | MSIS margin | Sigma proof | Total tail     |
| --------------------------- | ------------- | -------------------- | ----------- | ----------- | ----------- | -------------- |
| D=32 naive                  | —             | —                    | —           | —           | —           | **infeasible** |
| D=64 + 4 levels (Rej1)      | 5,616         | 2^30.6               | 2^37.3      | 48 bits     | 18 KB       | 33–38 KB       |
| D=64 + 4 levels (Rej2)      | 292           | 2^26.9               | 2^33.6      | 51 bits     | 12 KB       | 27–31 KB       |
| D=64 + 4 levels (iterative) | 1,694         | 2^28.8               | 2^35.5      | 50 bits     | 14 KB       | 29–34 KB       |


**Winner:** D=64 with enough extra folding levels to drive `w_len` down to `O(few thousand)` FE.
Expected total tail is comparable to or smaller than non-ZK tail (40 KB).
Exact numbers wait on Open Question §11.2.

---

## 11. Open Questions

### 11.1 `V_ring` multiplicity in the Jolt integration

Does one aggregate `V_ring = Σ_j γ^j · y_j` suffice, or do we need per-stage `V_ring_s` because different Jolt stages close on disjoint `{r_j}` sets?
The non-ZK Jolt batch opener handles mixed points via `LazyOneHotSource`; mirroring that inside the R1CS may require one `V_ring` per distinct evaluation point, each as a separate `Com_aux1` witness slot.

### 11.2 Exact `β_tail'` and extra-level count

§10.3 uses `β_tail' ≈ 2^{12}` and 4 extra D=64 levels as a working anchor.
Precise values depend on the recursion schedule, norm growth from ring switching, and the challenge family at each level.
This needs a concrete computation from the parameter planner (`scripts/hachi_proof_planner.py`, `docs/planner-guide.md`).

### 11.3 Rejection-sampling variant

Not committed to a specific variant.
Rej1 is the default in §10; Rej2 and iterative RS at D=64 are compelling alternatives.
An open speculation is whether iterative RS could unlock sigma at D=32 once a suitable sparse-decomposable D=32 family is designed (would obviate the D=32 → D=64 transition entirely).

### 11.4 Ring-switching proof at the D=32 → D=64 boundary

`dispatch_ring_dim!` already supports both dimensions.
What's unresolved is whether a ring-switching *proof* is needed at this boundary (analogous to Hachi's existing ring switches) or a bare re-commit suffices.
If a proof is needed, it must itself be masked per §4.1.

### 11.5 `Com_aux1` layout — **merged layout adopted**

Resolved: `Com_aux1` is committed at the boundary between Hachi levels `L − 2` and `L − 1`, using level `L − 1`'s matrices `(A, B, D)`, with its witness appended to Hachi's mega-polynomial at that level (§2.1, §5.2).

Rationale.
(a) Folding through level `L` shrinks `Com_aux1`'s contribution to the tail by 2× per level, buying back most of the witness-size penalty of Jolt R1CS aux.
(b) `c_1_aux` and other R1CS aux variables are not gadget-decomposable to `lb = 4` in general (they live in `F_q`, not `[0, 2^{lb})`). The merged layout accommodates this because the fused Spartan inner sumcheck treats `Com_aux1` as a generic multilinear polynomial, not a short-norm Hachi witness; level `L`'s fold then renormalizes the combined witness under the target `lb` profile.
(c) Saves one full Ajtai commitment scalar on the wire (1–2 KB at D=32/D=64) vs. a separate `Com_aux1`.

Separate-commitment fallback is retained as a drop-in for debugging; see §5.6.

### 11.6 Spartan vs. LNP22 for the Jolt R1CS

With the fused-Spartan architecture (§5) adopted as the baseline, LNP22-over-Ajtai for the *entire* R1CS is not currently pursued. LNP22 is kept only in its minimal, single-quadratic role at the tail sigma (§5.5).

Why the full-LNP22 option is deferred:

1. **Implementation sequencing.** BlindFold's R1CS and Spartan scaffolding already exist in `jolt-hachi`; porting sumcheck + Ajtai commitments is a known shape (§11.8). LNP22-over-Ajtai at R1CS scale is greenfield.
2. **Fused Spartan already kills the sumcheck traffic overhead.** Level-`L − 1` fusion folds Spartan inner into Hachi stage-2, so the only additional sumcheck cost vs. plain Hachi is Spartan outer (14 rounds, <1 KB).
3. **LNP22 full-R1CS would need σ-invariance analysis.** Hachi's D=64 sparse challenge family is not σ-invariant; moving to a palindromic subfamily requires fresh parameter work.

LNP22 single-quadratic at the tail (§5.5) is the sweet spot: minimal implementation effort, no σ-invariance requirement (single relation, no batching), and completely avoids a second Spartan invocation for the one residual `A · B = C` claim.

### 11.7 Extension to batch openings

Hachi's `batch_prove` / `batch_verify` should extend to the batched Jolt opening without surprises, but the interaction between batch randomization and per-level masking needs verification.
In particular, whether one unified `e_y` at the tail sigma can carry batch-weighted eval claims alongside the pad-residual and `V_ring` bindings.

### 11.8 Retirement of the Pedersen stack

In the Jolt integration, BlindFold's R1CS compiler, `StageConfig` / `OutputClaimConstraint` / `layout.rs` survive unchanged; `relaxed_r1cs.rs`, `folding.rs`, `spartan.rs`, `protocol.rs`, `witness.rs`, and the `HyraxParams` grid logic in `r1cs.rs:437-485` are replaced by a Hachi-native Spartan prover that closes `v_Spartan = Z(r_y)` against the multilinear commitment.
Exact cut plan TBD.

### 11.9 Comparison with Longfellow's encrypt-then-prove

Longfellow (Google's ZK system) achieves ZK by masking sumcheck messages with a Ligero-committed OTP, then proving the masked relation in Ligero.
Hachi's committed-pad + Gaussian-tail approach is structurally different but pursues the same end.
A formal comparison of overhead would be valuable; see `hachi-longfellow-zk-technique.md`.

### 11.10 Formal security reduction

Write a complete security proof: (a) soundness via Hachi's existing CWSS/special-soundness extraction plus §4.3's and §5.4's `n/q` losses aggregated; (b) ZK via the hybrid argument of §8, formalized as a reduction to LHL hiding (§2.2), per-round OTP hiding (§4.2), and rejection-sampling closeness (§6).
Target notion: statistical ZK in the ROM for the Fiat-Shamir compiled protocol.

### 11.11 Integrate modulus switching with the ZK design

Hachi's later recursion levels can apply a modulus-switching gadget that drops the working modulus from `q ≈ 2^{128}` down to a small `q_lo` (e.g., 32-bit).
At the low-modulus regime two things change qualitatively:

1. **MLWE becomes cheap.**

The MLWE infeasibility table in Appendix A flips: at `q_lo ≈ 2^{32}`, 4 ternary ring elements per sample suffice for 128-bit security (cf. the Lantern parameters), so MLWE-based hiding (BDLOP, ABDLOP, Hint-MLWE) is back on the table.
2. **Rejection-free ZK becomes competitive.**
Hint-MLWE ([https://eprint.iacr.org/2025/2239](https://eprint.iacr.org/2025/2239)) and similar techniques achieve hiding without rejection sampling, eliminating the per-norm-check restart cost that drives the Gaussian-tail sigma's `σ` to the large values of §6 / §10.

The ZK design in this note works entirely at `q ≈ 2^{128}`, where Gaussian rejection sampling is the only option.
A unified design that performs modulus switching *before* the tail sigma — running the joint sigma at `q_lo` instead of `q` — would shrink the tail sigma response by roughly `log₂(q / q_lo) ≈ 96` bits per coefficient, a substantial proof-size win.
Technical challenges include:

- **Where to switch.**
The last few Hachi folding levels already operate at lower-norm regimes (cf. §4 and the planner schedules); the natural choice is to apply modulus switching at the boundary just before phase 4, but this interacts with `Com_aux1` placement at level `L − 1` (§5.2).
- **Per-pad correctness across the boundary.**
Phase-1 `Com_pre` is computed at `q`, but the phase-4 sigma response is at `q_lo`.
Either pads are committed twice (once at each modulus, with a binding proof), or `Com_pre` is laid out with `q_lo`-residues for the tail-relevant pads from the start.
- **Post-switch hiding regime.**
At `q_lo` the choice between Ajtai + LHL (still works), BDLOP/ABDLOP (uses cheap MLWE), and Hint-MLWE (rejection-free) becomes a genuine optimization rather than a forced choice.
- **LNP22 single-quadratic at small `q`.**
The LNP22 single-quadratic add-on (§5.5) has soundness `1/q_1`, where `q_1` is the smallest prime factor of the modulus.
At `q ≈ 2^{128}` prime, one shot is enough; at `q_lo ≈ 2^{32}` a small `q_1` may force λ-wise repetition (LNP22's original setting).
This needs to be re-derived for the chosen `q_lo`.
- **Ring-switch vs. modulus-switch interaction.**
§11.4 (ring-switch proof at `D = 32 → D = 64`) and modulus switching may share a single boundary; designing them jointly is open.

This is the single most important extension to the framework in this note.
Local working notes on the modulus-switching gadget itself: `hachi-zk-and-modulus-embedding.md`, `hachi-lowering-to-d64.md`.

### 11.12 Audit the full set of residual quadratics

§5.5 explicitly accounts for one residual quadratic (Spartan outer's `Az(r_x) · Bz(r_x) = Cz(r_x)`).
Other quadratics may surface from sumcheck round-eval closures whose virtual oracle is non-linear and is *not* itself reduced to a pure linear functional by the masking + batched-residual scheme.
Concrete candidates to audit:

- **Hachi stage-2's `eq · w · (w + 1)` term.**
The expected-output-claim formula at stage-2 (`src/protocol/sumcheck/hachi_stage2.rs`) contains a quadratic in `w_eval`. After fusion at level `L − 1` the closing point `(r_y, w_eval)` is folded into the tail witness, but the quadratic check `w_eval · (w_eval + 1)` against the `eq` factor produces a residual identity that must be discharged.
- **Hachi stage-1's degree-`b/2` range-check polynomial** (`range_check_eval_from_s` in `two_round_prefix.rs`).
A degree-`b/2 ≈ 2` polynomial in `s_claim` yields on the order of `b/2 − 1` residual quadratics at the tail (or one residual after a small extra Spartan-style batching).
- `**y_ring` trace check** (§2.4).
The batched-residual reformulation moves `ct(g^{(ℓ)}) = 0` to a linear residual, but the quadratic `ct(y_ring^{(ℓ)} · σ_{−1}(v^{(ℓ)})) = opening^{(ℓ)}` may surface as one residual quadratic per Hachi level if the `σ_{−1}`-twist isn't fully linearized by the fold.

For each, decide: handled by replicating the LNP22 single-quadratic add-on (per-relation `g_{quad}` slot, 2 extra sigma scalars), batched via verifier RLC into a single LNP22 instance, or absorbed by a small extra outer-Spartan over the residuals.
The total cost of the LNP22 quadratic discharge is linear in the residual-quadratic count, so the budget should stay sub-KB unless the count blows past 10–20.
This is the most important soundness item still to be audited end-to-end, and tightens §11.10's reduction.

---

## Appendix A. Why not Path A (separate masking commitment) or Path B (ABDLOP)?


| Axis               | Path A (separate mask) | Path B (ABDLOP redesign) | **Path C (adopted)**             |
| ------------------ | ---------------------- | ------------------------ | -------------------------------- |
| Proof size         | 2x                     | 1.4x                     | **1.0–1.2x**                     |
| Prover time        | 2x                     | 1.5x                     | **1.2x**                         |
| Commitment changes | None                   | Full redesign            | None (tail only)                 |
| Sumcheck changes   | Full masking poly `ρ`  | BDLOP masking            | Committed-pad sum-of-univariates |
| Engineering risk   | Low                    | High                     | Medium                           |


**Path A** commits to a full masking polynomial `ρ` of the same size as the witness, doubling both commitment work and proof size. The recursive opening of `ρ` doubles the recursion.

**Path B** replaces Hachi's Ajtai commitment with ABDLOP (Ajtai + BDLOP). ABDLOP's hiding relies on MLWE, which is infeasible at Hachi's modulus (see below). The BDLOP part would hold masking values and achieve hiding via MLWE, but:

**MLWE infeasibility at `q = 2^128 − 275`.**

MLWE security degrades as `log(q) / n` increases (`n = κ · D`). At `q ≈ 2^128`, the attacker gains a large advantage from lattice reduction: the LWE lattice has a short vector of length roughly `q^{1/(n+1)}`, found efficiently by BKZ unless `n` is very large. Empirical results (lattice estimator, primal_usvp, BDGL16 sieving):


| Config             | κ_MLWE | n = κ·D  | Error       | Security (bits) |
| ------------------ | ------ | -------- | ----------- | --------------- |
| Ternary secret     | 1      | 64       | ternary     | 3               |
| Ternary secret     | 2      | 128      | ternary     | 9               |
| Ternary secret     | 4      | 256      | ternary     | 22              |
| Ternary secret     | 8      | 512      | ternary     | 40              |
| Ternary secret     | 16     | 1024     | ternary     | 62              |
| Ternary secret     | 32     | 2048     | ternary     | 99              |
| **Ternary secret** | **77** | **4928** | **ternary** | **128**         |
| Gaussian σ=2^16    | 4      | 256      | Gaussian    | 27              |
| Gaussian σ=2^32    | 4      | 256      | Gaussian    | 38              |
| Gaussian σ=2^48    | 32     | 2048     | Gaussian    | 128             |


128-bit security requires either `κ_MLWE = 77` ternary (`n = 4928` ring coefficients, 39 KB per MLWE sample) or `σ ≈ 2^{48}` Gaussian at `n = 2048` (drowning signal in noise). Both are prohibitive.

**Consequence for Hachi.** ABDLOP-style commitments and Lantern's BDLOP-based proof framework cannot be used at Hachi's native modulus. Hachi's hiding is instead achieved unconditionally via LHL (§2.2), which works precisely because `q` is large (support-separation `2M² < q` is trivially met).

**Contrast with Lantern (`q ≈ 2^32, D = 64`).** At Lantern's parameters, MLWE `κ_MLWE = 4` ternary gives 128 bits of security; BDLOP adds 1 KB per sample. Lantern's `N_B = 19` output ring elements make LHL prohibitive there (61 KB extra input). Opposite ends of a structural trade-off:


|            | Hachi (q 2^128, `N_B = 1`)             | Lantern (q 2^32, `N_B = 19`)     |
| ---------- | -------------------------------------- | -------------------------------- |
| LHL cost   | 43 elements (already present)          | 387 elements (61 KB, infeasible) |
| MLWE cost  | 4,928 coefficients (39 KB, infeasible) | 256 coefficients (1 KB, cheap)   |
| **Winner** | **LHL**                                | **MLWE**                         |


---

## Appendix B. Building Blocks Reference

### B.1 Gaussian masking and rejection sampling

In a sigma protocol where the prover computes `z = c · s + y` with `s` short and `y ← D_σ`, the raw distribution of `z` depends on `s`. Rejection sampling fixes this: output `z` with probability

`min(1, D_σ(z) / (M · D_{c·s, σ}(z)))`

and abort otherwise. After rejection sampling, `z \~ D_σ` independent of `s`. The simulator samples `z ← D_σ` directly, with no rewinding.

- **Rej1** (Lyubashevsky 2012): `σ = 13 · ‖v‖`, repetition `M ≈ 3`, statistical distance `2^{-128}`. No leakage.
- **Rej2** (LNS21a, one-sided): `σ = 0.675 · ‖v‖`, repetition `M ≈ 3`, leaks 1 bit (sign of `⟨z, v⟩`). Acceptable when commitments are single-use.
- **Rej0** (bimodal): same width as Rej1, repetition `M` instead of `2M`. No leakage.

### B.2 Gärtner's iterative rejection sampling

When the challenge `c` is sparse with small coefficients, decompose `c = Σ_{i=1}^ω c_i` with each `c_i` having a single nonzero coefficient. Build `z` iteratively:

```
z_0 = y
for i = 1, ..., ω:
    v_i = S · c_i
    z_i = R_{v_i}(z_{i-1})
    if z_i = abort: restart from z_0 = fresh y
z = z_ω
```

Each step handles `v_i` with magnitude one column of `S`. For `α ≥ 4`, per-step abort probability `≤ 2^{-108}`, total over `ω` steps `ω · 2^{-108}`, negligible.

Width reduction: `σ ≈ 24 · ‖S‖_∞` instead of `13 · ‖S·c‖_2`, roughly `√ω` smaller.

**Applicability.** D=64 SplitRing(hw=21, mag2=6) partially applies: 21 terms with magnitudes ≤ 6 give `σ ≈ 24 · 6 · ‖w‖_2 = 144 · ‖w‖_2` per step, vs. Rej1's `702 · ‖w‖_2`. D=128 Uniform(w=31, ±1) is fully applicable (binary-magnitude coefficients).

D ≤ 32 is moot: no production challenge family at D=32, and any 128-bit-soundness family at D=32 would need dense coefficients that defeat Gärtner's sparse-decomposition premise.

### B.3 Coefficient masking for constant-term proofs (ENS20 / LNP22 §1.3)

To prove `ct(f(s)) = 0` without revealing other coefficients:

1. Commit to masking polynomial `g ∈ R_q` with `ct(g) = 0`, other coefficients uniform.
2. Given challenge `γ ∈ Z_q`, send `h = γ · f(s) + g`.
3. Verifier checks `ct(h) = 0`.

Non-constant coefficients are perfectly masked by `g`. Soundness: if `ct(f(s)) ≠ 0`, then `Pr[γ · ct(f(s)) + ct(g) = 0] ≤ 1/q_1` where `q_1` is the smallest prime factor of `q`.

Relevant if the tail sigma protocol needs to prove witness properties beyond the commitment relation (e.g., quadratic norm bounds via `ct(σ_{-1}(s) · s) = ‖s‖²`).

### B.4 Leftover Hash Lemma over `R_q`

See §2.2. Key identity: for `R_q = Z_q[X] / (X^D + 1)` with `X^D + 1` factoring into two irreducible halves, LHL over `R_q` applies when `2M² < q` (support-separation), and the hiding bound is `Δ ≤ (1/2) · √(q^{D·N_B} / 2^{H_∞(ŵ)})`. Hachi's `N_B = 1` design makes this trivially satisfied.

---

## Appendix C. Milestone Plan

### M0 — Parameter study (largely done)

§10.3 establishes that D=64 with 4 extra D=64 folding levels and Rej1 gives `B_extract ≈ 2^{37.3}` with 48 bits of MSIS margin under the working anchor (`β_tail' ≈ 2^{12}`, starting from the D=32 tail `w_len ≈ 80K` FE).
No tail-specific modulus needed. **Remaining:** pin down exact `β_tail'` and extra-level count from the parameter planner (Open Question §11.2); verify the D=32 → D=64 ring transition mechanics (§11.4).

### M1 — Hachi sumcheck masking

Implement committed-pad sum-of-univariates masking (§4.1) for both stage-1 and stage-2 Hachi sumchecks. Per level, allocate the `d + 1` pad slots in the witness layout so they ride inside the existing Ajtai commitment. Emit `g̃_i = F_i + ρ_i` in the clear. Record the per-level batched-residual functional `L_mask` and propagate the masked reduced claims through the recursion unchanged. Validate on the existing (non-ZK-tail) protocol by stubbing the tail with the current `PackedDigits` path.

**Acceptance:** masked sumcheck produces valid proofs; the batched residual functional is recorded and consistent with the pads embedded in the final tail witness.

### M2 — `y_ring` LNP22 coefficient masking (batched-residual form)

Per §2.4: allocate one garbage ring element `g^{(ℓ)}` per Hachi level in `Com_pre`, sampled uniformly in `R_q` (D free `F_q` coefficients, **no hyperplane constraint**).
In `prove_one_level`, replace the absorption of `y_ring_orig` under `ABSORB_RING_SWITCH_MESSAGE` with absorption of the masked value `y_ring^{(ℓ)} = y_ring_orig^{(ℓ)} · σ_{-1}(v^{(ℓ)}) + g^{(ℓ)}`.
Remove the per-level `ct(y_ring) = opening` check from the verifier at round time; the verifier instead records the public residual `δ^{(ℓ)} := ct(y_ring^{(ℓ)}) − opening^{(ℓ)}` for later discharge.
Add two families of linear rows per level to the tail-sigma linear functional: the `R_q`-linear well-formedness of `y_ring^{(ℓ)}` (D `F_q` rows) and the residual pin `ct(g^{(ℓ)}) = δ^{(ℓ)}` (1 `F_q` row, public RHS).

**Acceptance:** proofs verify; `y_ring_orig` no longer appears in the clear (`y_ring^{(ℓ)}` absorbed is statistically uniform in `R_q`); the tail sigma discharges the two row families end-to-end.

### M3 — Tail sigma protocol

Implement Gaussian-masking sigma protocol at D=64. Includes: promote tail witness from D=32 to D=64; fold 4 extra D=64 levels (exact count per §11.2); run sigma protocol with D=64 challenges. New tail proof structure: `(t, e_y, h, μ, ẑ, z_q)` replaces `PackedDigits`. Combined `e_y` carries Hachi-eval, Spartan-eval, batched residuals, `V_ring` binding, and `y_ring` residual pins; `(h, μ, z_q)` carry LNP22's single-quadratic discharge.

**Acceptance:** proofs verify; tail no longer reveals the witness; total ZK tail is proof-size-neutral or smaller than the 40 KB non-ZK tail under the working anchor.

### M4 — Jolt integration with fused Spartan at level `L − 1`

Implement the `Com_pre` + `Com_aux1` two-commit structure of §2.1 with all pads (Jolt + Hachi + Spartan) upfront in `Com_pre`.
Record phases 2 and 3 verifier-algebra relations during the masked sumchecks (§3.3).
At the boundary between Hachi levels `L − 2` and `L − 1`, commit `Com_aux1` with R1CS aux variables merged into Hachi's mega-polynomial layout.
At level `L − 1`, run Spartan outer as an independent masked sumcheck and fuse Spartan inner with Hachi stage-2 via RLC (§5.3). Level `L` folds `Com_aux1` into the tail witness; the residual quadratic is discharged by LNP22 at phase 4 (§5.5).
Validate end-to-end on a small RV64 program.

**Acceptance:** Jolt proofs verify under the ZK path; transcript size within 10% of non-ZK; no Jolt intermediate state revealed; `Com_aux1` opens jointly with the Hachi tail.

### M5 — Security argument

Write composed security argument: soundness (CWSS + `n/q` per cluster; §4.3, §5.4) + simulation (§8 hybrid) + Fiat-Shamir assumptions.
Include the LHL hiding argument (§2.2) and the MLWE infeasibility justification (Appendix A).

### M6 — Integration test

End-to-end ZK proofs on representative workloads. Measure proof size, prover time, verifier time. Compare against non-ZK baseline.