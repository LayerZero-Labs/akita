# Spec: L2 MSIS Cutover, Operator-Norm Challenges, and Folded-Witness L2 Certificates

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao, Cursor agent draft |
| Created     | 2026-06-04 |
| Status      | proposed, draft for iteration |
| PR          | [#155](https://github.com/LayerZero-Labs/akita/pull/155) (L2 MSIS tables), [#195](https://github.com/LayerZero-Labs/akita/pull/195) (certificate geometry + D64 op-norm cutover) |

## Summary

Akita currently prices committed-fold weak binding through coefficient `L∞`
collision buckets.
This spec cuts the protocol over to an `L2` / Euclidean MSIS security model and
adds the proof machinery needed to justify the tighter bound inside the
protocol.

The cutover has three linked parts.
First, folding challenges are sampled from operator-norm accepted distributions,
starting with the D=64 exact-shell family and rejection threshold `gamma(c) <= 16`.
Second, the folded witness `z = sum_i c_i * s_i` carries a certified Euclidean
norm bound, proved over the finite field as an exact integer statement.
Third, the SIS planner, generated tables, public security model, transcript
descriptor, proof sizing, and verifier checks all move to the new L2 norm model
in one pass.

This is intentionally a full cutover.
The branch does not keep parallel `collision_inf` and `collision_l2` pricing
paths, dual schedule tables, legacy aliases, or compatibility shims for the old
coefficient-`L∞` security model.

## Intent

### Goal

Replace Akita's committed-fold coefficient-`L∞` SIS pricing with certified
Euclidean folded-witness pricing, backed by operator-norm accepted challenges
and an in-protocol `||z||_2^2 <= B` certificate.

The primary protocol surfaces are:

- `akita-challenges`: operator-norm accepted ring-challenge families, including
  accepted-support accounting and transcript-stable rejection sampling.
- `akita-types::sis`: L2 MSIS security buckets, secure-rank lookup, folded
  witness L2 bound derivation, and digit/certificate sizing.
- `akita-types::proof`: proof shape changes for the folded-witness L2
  certificate (`B_l2`, trailing `ell_hat` / `carry_hat`, masked `V`, carry
  linear claim).
- `akita-prover`: computation of realized folded-witness square sums from
  `DecomposeFoldWitness.centered_coeffs` / committed `z_hat` digit planes,
  four-square slack and carry-limb construction, and integration into the fused
  stage-2 proof flow.
- `akita-verifier`: replay of the L2 certificate, no-panic validation of all
  certificate shapes, and consistency with the committed next witness.
- `akita-config` / `akita-planner`: schedule search, shipped-table selection,
  generated table representation, and proof-size accounting under the L2 MSIS
  model.

### Product scope (operator-norm rejection)

Per-level `op_norm_rejection` on `LevelParams` is ring-dimension-agnostic
infrastructure: the planner may enable it only when Γ collision pricing
strictly lowers audited A-rank vs ω pricing **and** the fold-level witness
scoring cost (`(1 + n_a)·δ_open·2^r + δ_commit·δ_fold·m_eff`, same as
`optimal_m_r_split`) is strictly lower with rejection on at that geometry.
**Production scope today is D=64 only.** The only shipped binding preset is
`ExactShell { count_mag1: 31, count_mag2: 11 }` with `T = 18` at ring degree 64.
D=32 uses `BoundedL1Norm`; D=128 and D=256 use `Uniform` sparse challenges
(`proof_optimized_ring_challenge_config`). For those families
`operator_norm_cap` equals L1 mass, so the flag stays false and the sampler
skips the rejection oracle. D=128/D=256 flat folds may still use fold-linf
tail-bound digit tightening; that is orthogonal to operator-norm rejection.
Extending rejection to other ring dimensions is deferred until a binding Γ
preset and a certified acceptance floor exist for that `(family, d)` pair.

### Invariants

- **Single security table, per-role norm derivation.** All SIS binding decisions
  price against one Euclidean (L2) MSIS floor table.
  The committed-fold A-role contributes its L2 collision bound directly.
  The B-role and D-role opening-digit collisions keep their natural coefficient
  `L∞` bound `2^lb - 1` (the difference of two balanced digits) and are converted
  into the same L2 table by the explicit inequality `||v||_2 <= sqrt(d)·||v||_inf`.
  No committed-fold rank, schedule, or proof-size path uses the old coefficient
  `L∞` collision-bucket *table* after the cutover.
  The only surviving `L∞` quantities are the per-digit difference feeding that
  conversion and the folded-witness `||z||_inf` bound that still sizes the digit
  count of the next recursive witness (`num_digits_fold`); neither prices the
  A-role binding rank.
- **Exact integer certificate via no-wrap limbs.** The verifier accepts the
  realized folded-witness L2 certificate only when every field equality it relies
  on is known to be an exact integer equality.
  Two structural realizations, selected by a public field-capacity gate, deliver
  this. The field-fitting realization, used when the worst-case square sum and
  slack fit the field, checks `Σ_x z_aug(x)^2 = B_l2` in one sumcheck. The
  grouped-carry realization, used otherwise (notably fp32 dense recursive levels,
  where the square sum exceeds `q`), groups the committed digits into no-wrap
  limbs and reconciles the wide integer with a carry chain.
  Both are gated by validated structural bounds, never by the realized value
  alone. Only when even single-digit grouping (`g = 1`) fails the gate does the
  level emit no certificate and price the A-role at the deterministic worst-case
  L2 bound.
  The deterministic bound is still Euclidean and still far tighter than the old
  `L∞` envelope; the certificate is a no-wrap-gated tightening, not a separate
  security model.
- **Folded witness is the certified object.** The certified vector is the
  centered integer folded witness represented by
  `DecomposeFoldWitness.centered_coeffs`, not an unrelated evaluation table or
  heuristic proxy.
- **Operator-norm challenge contract.** Every sampled accepted challenge
  satisfies the configured negacyclic convolution operator-norm cap.
  Prover and verifier must replay the same rejection-sampling transcript stream.
- **Accepted challenge entropy.** Each production challenge family has at least
  128 bits of accepted Fiat-Shamir support after rejection sampling.
  For D=64 exact shell, the target starting point is
  `ExactShell { count_mag1: 31, count_mag2: 11 }` with `gamma(c) <= 16`, whose
  raw support is about `2^130.152`.
- **No adversarial challenge bias hole.** The security proof must account for the
  accepted challenge distribution used by the extractor.
  Honest-pair experiments are calibration only and cannot justify the production
  bound by themselves.
- **Stage-2 consistency.** The L2 certificate must be tied to the same folded
  witness that is decomposed into the next recursive witness and used in the
  ring relation.
  A prover cannot certify one `z` and commit to a different `w`.
- **Verifier no-panic boundary.** Malformed verifier-facing challenges,
  certificates, schedule entries, proof shapes, digit decompositions, or
  overflow-prone dimensions are rejected with `AkitaError` or
  `SerializationError`, never by panicking.
- **Transcript binding.** The instance descriptor binds the active MSIS norm
  model, challenge family, operator-norm threshold, L2 bound policy, certificate
  shape, and schedule.
  A proof generated under the L2 model cannot verify under old L∞ parameters.
- **Generated schedule determinism.** Runtime DP fallback and shipped generated
  schedules use the same L2 bound formulas and security tables.
  Table-hit and table-miss schedule resolution must agree on every value that
  affects transcript, proof shape, and setup dimensions.
- **Single certificate for transparent and ZK.** The realized certificate is the
  same protocol in transparent and ZK builds: it never sends folded-witness inner
  products, and its only public scalar is `B_l2`.
  ZK builds toggle on the existing masking of the sumcheck messages, the claimed
  sum `V`, and the committed `ell_hat` / `carry_hat`; transparent builds run the
  identical claim structure without hiding witness.
  The mask accounting stays linear except for the single explicitly recorded
  quadratic relation (the squared-sum sumcheck).
- **Standalone code documentation.** Comments, docstrings, and implementations
  must make sense on their own, without the reader opening this spec.
  Do not tag source with slice identifiers (`S1`, `S8`, ...), do not cite "the
  spec", and do not describe a symbol only by its future spec consumer.
  Explain the math, the contract, and the symbol's role in codebase terms
  (concrete types, functions, and protocol objects).
  Slice and spec tracking belongs in commit messages and PR descriptions, not in
  the code.

### Non-Goals

- No long-term support for the coefficient-`L∞` committed-fold pricing model.
- No compatibility mode for proofs or schedules generated under the old model.
- No attempt to use empirical challenge or witness distributions as a security
  proof.
- No unrelated tensor-challenge, setup-offloading, or terminal-proof refactor.
- No generic user-facing arbitrary rejection predicate API in the first cut.
  Production challenge families are explicit policy variants with audited
  support and norm facts.
- No weakening of the existing digit range checks.
  The L2 certificate supplements the digit/ring-relation proof; it does not
  replace the checks needed to bind decomposition and recursive witness layout.

## Evaluation

### Acceptance Criteria

- [x] *(#155 partial)* Specs and `norm_bound.rs` agree on Euclidean MSIS lookup,
      Lemma 7 on fold response `z` (`8·ω·β_inf·ν` → `collision_l2_sq_for_linf_envelope`),
      and `β_inf = fold_witness_beta`. Full public security-doc cutover completes with
      S6+ certificate wording.
- [x] *(#155, S5b)* `akita_types::sis` exposes `committed_fold_collision_l2_sq` /
      `rounded_up_collision_norm_s`, derived `d·B²` table keys (`COEFF_LINF_BUCKETS`,
      `collision_l2_sq_for_linf_envelope`), and pow2-ladder fallback; `collision_inf` is
      removed from production call sites (`collision_l2_sq` on `AjtaiKeyParams`).
- [x] *(#155, S3 infra)* Exact-shell operator-norm rejection sampling,
      `operator_norm_cap`, per-level `op_norm_rejection` on `LevelParams`, and
      descriptor binding are implemented. Production D=64 ships `(31, 11)` with
      `T = 18`; the planner enables rejection sampling only on fold levels where
      Γ pricing strictly lowers audited A-rank vs ω pricing.
- [x] *(#195, geometry)* `fold_l2_certificate` exposes realization selection,
      grouped-digit layout, no-wrap gate, and carry cell budgets; `B_l2_pub` is
      computed by [`fold_witness_l2_pub_bound_sq`] from public level inputs only
      (not from realized `Z_SQUARED`). Geometry uses the **raw** bound; MSIS ladder
      rounding is deferred to S11 schedule materialization (see **Realized-certificate
      tier**). Prover/verifier certificate replay remains open (S8–S10).
- [ ] The D=64 accepted family has a rigorous support lower bound of at least
      128 bits, not just a Monte Carlo estimate. *(#195 ships rational floor
      `117/500` at `T = 18`; vendored checked cert in CI remains open.)*
- [ ] The prover derives the grouped `z_hat` limbs from the actual
      `DecomposeFoldWitness.centered_coeffs` used for ring-switch witness
      construction, and its reconstructed `Z_SQUARED` matches a direct integer
      reference in tests.
- [ ] The realized certificate adds four-square slack `ell_0..ell_3` so the bound
      becomes the equality `Σ_i z[i]^2 + Σ_h ell_h^2 = B_l2`, and commits
      `ell_hat` (and, in the grouped-carry realization, `carry_hat`) as trailing
      segments of `w_next`. No folded-witness inner product is ever sent.
- [ ] The field-fitting realization proves `Σ_x z_aug(x)^2 = B_l2` in one degree-2
      sumcheck, with `z_aug = z || ell_0..ell_3` recomposed from the committed
      digits.
- [ ] The grouped-carry realization squeezes one challenge `alpha`, proves the
      single sumcheck `Σ_x Z_alpha(x)^2 = V` for the `alpha`-weighted limb
      recomposition `Z_alpha`, and reconciles
      `V = Σ_e alpha^e T_e + Σ_e alpha^e (B·h_{e+1} - h_e)` against the public
      bound digits `T_e` and the committed carries via one short linear claim.
- [ ] The verifier accepts a realized level only when the public no-wrap gate
      holds for every convolution exponent
      (`D_e + H'_e + (B-1) + B·H'_{e+1} < q`, with `H'_e` the committed carry
      cell run's realizable budget), and the folded carry soundness argument in
      **Grouped-Carry Soundness** is implemented (polynomial identity in `alpha`,
      carry layout `delta_carry(e)`, boundary `h_0 = h_{E+1} = 0`).
      A test pins which levels choose field-fitting, grouped-carry, or
      deterministic-fallback tiers.
- [ ] The certified statement is over the committed `z_hat` / `ell_hat` /
      `carry_hat` digit planes, and a test ties every limb, slack, and carry
      evaluation to the committed `w_next` segment via gadget recomposition (a
      tampered `z_hat`, `ell_hat`, or `carry_hat` fails the check).
- [x] *(#155, S5b)* B-role and D-role collisions use `collision_l2_sq_for_linf_envelope`
      on `2^lb − 1` (`rounded_up_collision_norm_t/w`). Dedicated table-conversion test
      remains a follow-up; pricing path is wired.
- [ ] The squared-sum sumcheck reduces to limb evaluations `a^{<j>}(rho)`
      (equivalently the single `Z_alpha(rho)`), and a linear virtualization step
      ties those plus the carry evaluation `carry_hat(rho_c)` to the existing
      `w_next` opening (batched or explicitly justified adjacent), without
      duplicating witness scans more than necessary.
- [ ] On certifying levels the proof shape accounts for the masked claimed sum
      `V`, the carry linear-claim transcript, and any stage-message changes (no
      partial-sum payload exists). A test pins the per-level message shape, and
      the descriptor binds it.
- [ ] The committed `z_hat` / `ell_hat` / `carry_hat` segments and `B_l2` are
      transcript-bound before `alpha` or any squared-sum / carry challenge is
      squeezed (wire-before-squeeze), and a logging-transcript test enforces the
      ordering.
- [ ] Proof shape, proof-size formula, shape deserialization, and compressed
      proof validation account for the certificate payload (`B_l2`, the
      `ell_hat` / `carry_hat` witness growth, the masked `V`, and the carry
      linear-claim layout, pinned by a serialization test).
- [x] *(#155, S5b; #195 partial S11)* Runtime DP, `expand_to_level_params`, and
      shipped generated schedule tables size A-role ranks from `collision_l2_sq`;
      `num_digits_fold` still uses `β_inf`. Until S11 wires realization tiers, **all**
      fold levels price A-role via Lemma 7 on `β_inf` (conservative on would-be
      deterministic-fallback levels). `folded_witness_l2_bound_squared` is implemented
      for S11 fallback pricing only. Certificate `B_l2_pub` on the wire waits for S6.
- [x] *(#155, S5b)* ZK and non-ZK shipped schedule tables are separate DP optima under
      different proof-byte accounting; drift guards run per feature set (see
      [`sis-euclidean-estimator.md`](sis-euclidean-estimator.md) ZK vs non-ZK section).
- [ ] `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`,
      and `cargo test` pass on the cutover branch *(CI gate for merge)*.
- [ ] End-to-end prover/verifier tests fail if any one of the committed folded
      witness, L2 certificate, next-witness commitment, or ring-relation rows is
      tampered independently.

### Testing Strategy

Unit tests:

- Challenge tests for D=64 exact-shell acceptance:
  deterministic rejection replay, support-domain bytes, accepted `gamma(c) <= T`,
  and stable transcript behavior across prover and verifier.
- Exact or interval-certified tests for the challenge support lower bound.
  These tests may validate a checked certificate artifact instead of enumerating
  the full challenge space in CI.
- SIS tests pinning L2 secure-rank lookup against generated tables, including
  bucket rounding and unsupported-bucket rejection.
- Folded-witness tests that compare `centered_coeffs` square sums with a direct
  negacyclic integer reference for dense, one-hot, recursive, and tensor-shaped
  folds.
- Grouped-carry tests covering field-fitting selection, single-digit grouping,
  last short groups, negative `C_e` / carries, the integer carry recurrence
  against a direct `Σ z[i]^2` reference, no-wrap-gate fallback, and a regression
  that mismatched `delta_carry(e)` or out-of-range carry cells are rejected.

Protocol tests:

- Stage-2 prover/verifier round-trip with the L2 term active.
- Transparent and ZK proof paths, if the ZK feature remains enabled on this
  branch.
- Root, recursive intermediate, and terminal paths.
  Terminal paths may use a different bound shape, but must be explicit.
- Multipoint and same-point batching, so `num_claims` routing cannot certify the
  wrong folded witness segment.
- Serialization/deserialization shape tests for the updated proof objects.

Drift guards:

- Generated schedule tables match runtime DP fallback.
- Proof-size model matches serialized proof bytes.
- Instance descriptor bytes change intentionally and are pinned by tests.
- Grep-style tests or review checks confirm no committed-fold production path
  still uses `rounded_up_collision_norm_s` or `fold_witness_beta` as an L∞
  security price.

### Performance

Expected direction:

- SIS A-rank should drop relative to the corrected L∞ committed-fold reprice
  when the L2 bound is substantially below the coordinate worst case.
- The L2 certificate payload is `B_l2`, the masked `V`, the squared-sum and carry
  sumcheck transcripts, and the `ell_hat` / `carry_hat` witness growth (no
  partial sums are sent). The carry payload shrinks as the group size `g` grows
  and `R` shrinks; it is empty in the field-fitting realization.
- Net proof size should improve only if the rank and recursive schedule savings
  exceed the certificate overhead.
- fp32 dense recursive levels keep the deterministic `L2_BOUND_SQUARED` (no
  certificate), so their gain comes only from the L2 reprice, not from the
  realized tightening.

Benchmarks and profile commands:

```bash
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile
```

#### Folded-witness L2 calibration (2026-06-04)

Local profile instrumentation in the `akita` worktree logs exact integer
`||z||_2`, `||z||_∞`, and per-coordinate RMS from
`DecomposeFoldWitness.centered_coeffs` at the `validate_decompose_fold`
boundary.
Sample 0 runs full prove plus verify; extra samples reuse the same commitment
and cloned hint with distinct transcripts.

```bash
AKITA_ALLOW_DEBUG_PROFILE=1 AKITA_PROFILE_TRACE=0 \
  AKITA_PROFILE_LOG=akita_prover::protocol::ring_relation=info \
  AKITA_PROFILE_SPAN_CLOSES=0 AKITA_PROFILE_ANSI=0 \
  AKITA_MODE=<mode> AKITA_NUM_VARS=<nv> \
  AKITA_PROFILE_Z_L2_SAMPLES=<N> \
  cargo run -q -p akita-pcs --example profile
```

**Methodology.**

- Certified object: `DecomposeFoldWitness.centered_coeffs` for
  `z = sum_i c_i * s_i`.
- Fold width: `B = num_claims * num_blocks` (typically `num_claims = 1` at
  root).
- Per-coordinate RMS: `z_rms = ||z||_2 / sqrt(coeffs)`.
- Current production D=64 exact shell is `(31, 11)`, so
  `rho2 = 31 + 4 * 11 = 75`.
  Operator-norm rejection is enabled per fold level only when Γ pricing lowers
  audited A-rank; the sampler enforces `gamma(c) <= 18` on those levels.
- Formula comparisons in historical calibration notes use candidate `Gamma = 16`.
- Backsolved source second moment (calibration only):
  `mu2_implied = z_rms^2 / (rho2 * B)`.
  This is not a direct measurement of `sum_i ||s_i||_2^2`.
- Exploratory triangle reference (calibration only, **not** production sizing):
  `det_rms = Gamma * B * sqrt(mu2_implied)`.
- Honest second-moment reference (fitted from the same sample):
  `honest_rms = sqrt(rho2 * B * mu2_implied)`.
- Old coordinate envelope ratio:
  `||z||_2 / linf_l2_envelope`, where `linf_l2_envelope` is the existing
  `sqrt(coeffs) * beta_linf` planner proxy.

**Caveats.**

- All numbers below are honest-pair calibration, not a security proof.
- Dense profile runs may panic afterward on a pre-existing proof-size overcount
  assertion (`planned` vs `actual` bytes).
  Norm logs are emitted before that panic.
- Op-norm rejection on fixed-energy exact shell does not change `rho2 = 78`;
  it would tighten `Gamma` and tail behavior, not coefficient energy.
- These runs use the *current* `(30, 12)` shell (`rho2 = 78`), not the proposed
  production `(31, 11)` shell (`rho2 = 75`). The backsolved second moments should
  be re-measured under `(31, 11)` before they feed planner code.
- Next instrumentation: log `sum_i ||s_i||_2` and `sum_i ||s_i||_2^2`
  directly at the decompose-fold source boundary before turning heuristics into
  planner code.

##### One-hot root level (terminal fold)

| mode | nv | B | samples | rows | coeffs | `beta_linf` | old L∞ env | `||z||_2` mean | `||z||_2` range | `z_rms` | `mu2_implied` | old env ratio | obs/det |
|------|----|---|---------|------|--------|-------------|------------|---------------|-----------------|---------|---------------|---------------|---------|
| onehot_fp32_d64 | 16 | 4 | 100 | 256 | 16384 | 216 | 27648 | 186.716 | 183.728-190.683 | 1.458719 | 0.006820 | 0.00675 | 0.2760 |
| onehot_fp32_d64 | 18 | 8 | 20 | 512 | 32768 | 432 | 78200 | 373.633 | 369.816-376.215 | 2.064052 | 0.006828 | 0.00478 | 0.1952 |
| onehot_fp32_d64 | 20 | 16 | 50 | 1024 | 65536 | 864 | 221184 | 747.919 | 743.923-753.662 | 2.921560 | 0.006839 | 0.00338 | 0.1380 |
| onehot_fp128_d64 | 16 | 4 | 30 | 256 | 16384 | 216 | 27648 | 141.151 | 139.764-142.373 | 1.102739 | 0.003898 | 0.00511 | 0.2760 |

One-hot observations:

- `mu2_implied` is stable near `0.00682` for `fp32_d64` across
  `nv = 16, 18, 20` despite `B` growing `4 -> 8 -> 16`.
- `||z||_2` scales like `sqrt(B)` at fixed density, as expected from the
  second-moment model.
- `fp128_d64` one-hot has lower `mu2_implied` (`0.003898`) and lower absolute
  `||z||_2`, consistent with sparser effective witness density at the same
  terminal shape.
- Old L∞ envelope ratios stay near `0.005` to `0.007`, roughly
  `150x` to `200x` pessimistic vs observed `||z||_2`.

##### One-hot `fp32_d64`, `nv = 20` (multi-level, 50 samples)

At `nv = 20`, one-hot proofs emit five folded-witness levels.
Root stays sparse; recursive levels inherit dense digit statistics.

| rows | coeffs | B | `||z||_2` mean | `z_rms` | `mu2_implied` | obs/det |
|------|--------|---|---------------|---------|---------------|---------|
| 1024 | 65536 | 16 | 747.919 | 2.921560 | 0.006839 | 0.1380 |
| 663 | 42432 | 8 | 22177.698 | 107.663801 | 18.577085 | 0.1952 |
| 406 | 25984 | 8 | 34368.493 | 213.209959 | 72.852217 | 0.1952 |
| 309 | 19776 | 8 | 33650.917 | 239.291721 | 91.769170 | 0.1952 |
| 273 | 17472 | 8 | 33319.664 | 252.074724 | 101.833811 | 0.1952 |

Recursive levels show the same rising-`mu2` pattern as dense proofs once the
witness is no longer one-hot sparse.

##### Dense `fp32_d64`, `nv = 16` (20 samples, four levels)

| rows | coeffs | B | `beta_linf` | old L∞ env | `||z||_2` mean | `||z||_2` range | `z_rms` | `mu2_implied` | old env ratio | obs/det |
|------|--------|---|-------------|-----------|---------------|-----------------|---------|---------------|---------------|---------|
| 896 | 57344 | 8 | 6912 | 1,655,189 | 51315.581 | 50828-51643 | 214.292 | 73.592 | 0.03100 | 0.1952 |
| 493 | 31552 | 8 | 6912 | 1,227,770 | 42388.828 | 42044-42809 | 238.637 | 91.265 | 0.03453 | 0.1952 |
| 342 | 21888 | 8 | 6912 | 1,022,602 | 35967.097 | 35609-36324 | 243.110 | 94.718 | 0.03517 | 0.1952 |
| 285 | 18240 | 8 | 6912 | 933,504 | 34104.167 | 33832-34340 | 252.520 | 102.191 | 0.03653 | 0.1952 |

Dense observations at `nv = 16`:

- Absolute `||z||_2` is orders of magnitude above one-hot, but old L∞ envelope
  ratios remain `0.031` to `0.037` (roughly `30x` to `35x` pessimistic).
- `mu2_implied` rises from `73.6` to `102.2` deeper in the recursion tree.
- At fixed `B = 8`, `obs/det` is constant `0.1952` across all four levels.

##### Dense `fp32_d64`, `nv = 18` (6 samples, seven levels)

| rows | coeffs | B | `||z||_2` mean | `z_rms` | `mu2_implied` | obs/det |
|------|--------|---|---------------|---------|---------------|---------|
| 2048 | 131072 | 32 | 21260.877 | 58.725432 | 1.382041 | 0.0976 |
| 765 | 48960 | 32 | 15170.494 | 68.561295 | 1.891533 | 0.0974 |
| 675 | 43200 | 16 | 19657.419 | 94.576800 | 7.168233 | 0.1380 |
| 668 | 42752 | 8 | 26645.005 | 128.865773 | 26.613366 | 0.1952 |
| 408 | 26112 | 8 | 36070.090 | 223.216945 | 79.849858 | 0.1952 |
| 310 | 19840 | 8 | 34040.113 | 241.668560 | 93.600131 | 0.1952 |
| 273 | 17472 | 8 | 33432.820 | 252.930786 | 102.526111 | 0.1952 |

Root levels at `B = 32` have much lower `mu2_implied` (`~1.4`) than recursive
levels at `B = 8` (`26` to `102`).

##### Dense `fp128_d64`, `nv = 16` (8 samples, six levels)

| rows | coeffs | B | `||z||_2` mean | `z_rms` | `mu2_implied` | obs/det |
|------|--------|---|---------------|---------|---------------|---------|
| 1024 | 65536 | 32 | 59251.337 | 231.450537 | 21.462808 | 0.0976 |
| 749 | 47936 | 8 | 21843.826 | 99.769506 | 15.952119 | 0.1952 |
| 658 | 42112 | 16 | 30208.971 | 147.208537 | 17.364938 | 0.1380 |
| 467 | 29888 | 8 | 30271.810 | 175.101536 | 49.136914 | 0.1952 |
| 361 | 23104 | 8 | 28590.044 | 188.092394 | 56.698332 | 0.1952 |
| 321 | 20544 | 8 | 27758.986 | 193.669440 | 60.109594 | 0.1952 |

`fp128` dense levels span `mu2_implied` from `15.95` to `60.11`, lower than
the corresponding `fp32` dense band at similar recursive depths.

##### Formula comparison

Two candidate models were checked against the same logs.

**Exploratory triangle bound (calibration only, not production sizing)**

```text
||z||_2 <= Gamma * sum_i ||s_i||_2
```

This decoupled `Gamma · B · ‖s‖_2` envelope is **not** Lemma 7 and must not
appear in planner or MSIS table code. With `mu2_implied` backsolved from each
sample, the fitted exploratory RMS is
`det_rms = Gamma * B * sqrt(mu2_implied)`.
The observed ratio `z_rms / det_rms` depends primarily on `B`:

| B | observed / deterministic (`Gamma = 16`, `rho2 = 78`) |
|---|------------------------------------------------------|
| 4 | 0.2760 |
| 8 | 0.1952 |
| 16 | 0.1380 |
| 32 | 0.0975 to 0.0976 |

This matches the closed form

```text
observed / deterministic ~= sqrt(rho2) / (Gamma * sqrt(B))
```

so the triangle bound is still pessimistic by about `3.6x` at `B = 4`,
`5.1x` at `B = 8`, `7.2x` at `B = 16`, and `10.2x` at `B = 32`.
It is much tighter than the old L∞ envelope (roughly `30x` to `200x` loose on
these runs), but not tight enough to replace a certified Euclidean bound.

**Honest second-moment bound**

```text
E ||z||_2^2 <= rho2 * sum_i ||s_i||_2^2
```

Equivalently, with per-coordinate source second moment `mu2`:

```text
E[z_rms^2] <= rho2 * B * mu2
```

When `mu2_implied` is computed from the same sample, `honest_rms` matches
`z_rms` by construction.
The substantive content is therefore in `mu2_implied` itself:

- One-hot terminal levels: `mu2_implied ~ 0.00682` (`fp32`) or
  `~ 0.00390` (`fp128`).
- Dense recursive levels: `mu2_implied` varies by level and field, typically
  rising from tens to about `102` in the deepest `fp32` levels logged here.

**Interpretation for planner work.**

- Do not price Euclidean security from `sqrt(coeffs) * beta_linf`; the old
  envelope is far too loose even when `||z||_2` is large in absolute terms.
- `Gamma * sum ||s_i||_2` is a usable worst-case skeleton, but honest data
  sit at roughly `sqrt(rho2) / (Gamma * sqrt(B))` of that skeleton.
- The second-moment model tracks honest scaling when `sum_i ||s_i||_2^2` is
  taken at the actual source blocks for each level.
  One-hot and dense levels need different policies.
- Direct source-block L2 sum logging is the gating step before any of this
  becomes planner input.

#### D32 Bounded-L1 Calibration

The L2 cutover should also revisit whether D32 should switch back into the
production set.
Current D32 uses `SparseChallengeConfig::BoundedL1Norm`, the fixed
`D = 32`, `M = 8`, `B = 121` sampler that draws a uniform 128-bit rank into a
retained subset of the bounded ball.
Exact dynamic programming over that retained production distribution gives:

```text
E ||c||_1              = 114.123123661
E ||c||_2^2            = 591.468541687
per-coordinate E[c_i^2] = 18.483391928
sqrt(E ||c||_2^2)      = 24.320126268
```

A local Monte Carlo over 300,000 production-distributed single challenges gave
the negacyclic convolution operator norm:

```text
gamma(c) mean = 44.57
gamma(c) p50  = 43.83
gamma(c) p90  = 52.99
gamma(c) p95  = 56.09
gamma(c) p99  = 62.26
```

This is far below the coefficient worst-case `||c||_1 = 121`, but much larger
than the D64 exact-shell threshold currently being studied.
The numbers are calibration only.
They suggest D32 is worth repricing under the L2/operator-norm model, but they
do not by themselves justify adding D32 back to production schedules.

## Design

### Notation

| Symbol | Meaning |
|--------|---------|
| `s` | Committed witness (A-role, Ajtai) |
| `z` | Fold response / folded witness: `z = Σ_i c_i · s_i` |
| `ẑ` | Decomposed digits of `z` at the next level |
| `z^{<j>}` | Grouped folded-witness limb assembled from a contiguous group of `ẑ` digit planes |
| `β_inf` | Deterministic `‖z‖_inf` envelope ([`fold_witness_beta`]) |
| `ν` | `ring_subfield_norm_bound` |
| `b` | Original gadget basis `2^lb` |
| `B` (`b_grp`) | Grouped basis `b^g`, where `g` is the number of original `ẑ` digits per grouped limb |
| `R` | Grouped-limb count `ceil(K / g)`, `K = num_digits_fold` |
| `ell_h` | Lagrange four-square slack integers, `Σ_h ell_h^2 = B_l2 − Z_SQUARED`; committed as `ell_hat` |
| `a^{<j>}` | Augmented limb family over the size-`(N+4)` domain (folded witness for `x < N`, slack at the tail) |
| `C_e` | Un-reduced base-`B` coefficient of the squared norm, `Σ_{r+s=e} <a^{<r>}, a^{<s>}>` |
| `T_e` | Public base-`B` digit of the bound, `B_l2 = Σ_e B^e T_e` |
| `h_e` | Signed carry, `C_e + h_e − T_e − B·h_{e+1} = 0`, `h_0 = h_{E+1} = 0`; committed as `carry_hat` |
| `res_e` | Carry residual `C_e + h_e − T_e − B·h_{e+1}`; vanishing for all `e` implies the bound |
| `H_e` | Tight carry magnitude budget from the structural recurrence (not witnessed) |
| `H'_e` | Realizable carry budget `(b/2)(2^{δ_carry(e)}−1)` from the committed cell layout |
| `δ_carry(e)` | Balanced digit cells committed for `h_e` (base-2 weights); `H'_e >= H_e` |
| `α` | Challenge-field randomizer folding the limbs and carries into one sumcheck |
| `Z_α` | `α`-weighted recomposition `Σ_j α^j a^{<j>}`, the squared-sum sumcheck's polynomial |

Do not confuse `z` with `ŵ` (D-role opening witness).

### Security object (Lemma 7, L2 instantiation)

Weak binding prices the **fold response** `z`, not committed `s`:

```text
‖z_A‖_2  ≤  8 · op_norm(c) · ‖z‖_2 · ν
```

with `op_norm(c)` realized as `ω = ‖c‖_1` for sizing today.

### Layer 1 — L2 MSIS table lookup (shipped)

Convert the Lemma-7 collision into the Euclidean SIS floor per ring row:

```text
collision_A_inf = 8 · ω · β_inf · ν,
collision_l2_sq   = lookup_L2(d, collision_A_inf).
```

`lookup_L2` is `collision_l2_sq_for_linf_envelope` in `ajtai_key.rs`: prefer the
tabulated derived key `K = d · B²` with `B = ceil_coeff_linf_bucket(linf)` (same
coefficient-`L∞` ladder as pre-cutover main), else round `d · linf²` up to the next
generated power-of-two bucket.

`committed_fold_collision_l2_sq` / `rounded_up_collision_norm_s` in
`crates/akita-types/src/sis/norm_bound.rs` call this path.
B/D roles pass digit collisions `2^lb − 1` through the same helper.

### Layer 2 — Deterministic `‖z‖_2` envelope (shipped, pre-certificate)

No realized `‖z‖_2` certificate in production yet. The planner bounds
`‖z‖_inf` by `β_inf` and converts to L2 for the estimator:

```text
β_inf = fold_witness_beta(...)
      = num_claims · 2^r_vars ·
        min(||c||_inf · ‖s‖_1, ‖c‖_1 · ‖s‖_inf),

‖z‖_inf  ≤  β_inf,
‖z‖_2   ≤  √d · β_inf   (hence collision_l2_sq = lookup_L2(d, 8ωβ_inf ν)).
```

`β_inf` is shared with `num_digits_fold` and prover
`validate_decompose_fold` (`DecomposeFoldWitness.centered_inf_norm`).

Operator-norm rejection (`gamma(c) <= Gamma`) is a separate challenge-family
contract. No Lemma-7 factor (`8`, `ν`, fold arity, or `β_inf`) may be dropped
or replaced by a decoupled `Gamma · B · ‖s‖_2` triangle bound.

B-role and D-role collisions stay at `2^lb − 1` with the same
`‖v‖_2^2 ≤ d · ‖v‖_inf^2` conversion.

### Realized-certificate tier (default, transparent and ZK)

The certificate certifies the **fold response** `z`, not committed `s`.
It proves the integer inequality

```text
Z_SQUARED = sum_{row, coeff} z[row][coeff]^2 <= B_l2_pub
```

without revealing `Z_SQUARED` or any folded-witness inner product, in both
transparent and ZK builds.
`B_l2_pub` is a **public per-level parameter** fixed from challenge family and
witness class before proving. It is **not** chosen from the realized
`Z_SQUARED` at prove time.

**Public derivability (hard contract).** Every party (prover, verifier, planner)
must be able to recompute the same scalar from **layout-visible inputs only**:
`challenge_l2_sq_per_block` (from the bound challenge family), fold block count
`num_claims · 2^{r_vars}`, folded row count
`inner_width = block_len · δ_commit`, and witness-class norms
(`FoldWitnessNorms`).
The prover may abort if honest `Z_SQUARED` exceeds `B_l2_pub`; it must not pick a
witness-dependent bucket at prove time.
On the wire, `B_l2_pub` is either recomputed from those public fields or read
from bytes pinned at schedule time; it is never inferred from witness data.

```text
B_l2_pub_raw = challenge_l2_sq_per_block · num_fold_blocks · inner_width
               · witness.l1_norm · witness.infinity_norm
             = rho2 · B · W · ||s||_2^2_row_max
```

where `rho2 = ||c||_2^2` on the exact-shell family (e.g. `75` for production
`(31,11)`), `B = num_claims · 2^{r_vars}` fold blocks,
`W = inner_width = block_len · δ_commit` folded response rows, and
`||s||_2^2_row_max` is the witness-class envelope `‖s‖_1 · ‖s‖_∞` for one
ring row (`FoldWitnessNorms` in code; implemented as
[`fold_witness_l2_pub_bound_sq`]).
Do not multiply by `N_z = W · d` here: `FoldWitnessNorms` already bounds the
full `d`-coefficient ring-row norm.

**Rounding policy.** The geometry module (`fold_l2_certificate`, no-wrap gate,
field-fitting eligibility) uses **`B_l2_pub_raw`** as a single exact `u128`.
Optional MSIS ladder rounding (`ceil_ladder` / [`fold_witness_l2_pub_collision_bucket`])
is a **schedule-time** decision (S11): if used, the rounded value is materialized
into `LevelParams` (or the generated schedule entry) when the planner expands a
level, alongside `collision_l2_sq` and `n_a`. The verifier reads the pinned scalar
from layout or replays the same public formula; it must **not** call SIS table
lookup (`ceil_supported_collision`) at verify time. Whatever scalar is pinned
must be used consistently for the no-wrap gate, public bound digits `T_e`, and
the certificate equality target (gate and wire must agree).

The prover proves `Z_SQUARED <= B_l2_pub`; the certificate slack uses
`B_l2_pub − Z_SQUARED`. Lemma-7 A-role rank on **certifying** levels still uses
`8 · Γ · …` on `β_inf`, not `Z_SQUARED` or `B_l2_pub`.

The deterministic ceiling (no-wrap gate fallback when no certificate is emitted)
is

```text
Z_SQUARED <= L2_BOUND_SQUARED
```

derived from the same `β_inf` / digit-range contract as `num_digits_fold`:
each coefficient of `z` is bounded by `balanced_digit_max(lb, num_digits_fold)`,
so

```text
L2_BOUND_SQUARED = coeffs · balanced_digit_max(lb, num_digits_fold)^2
```

This bounds **`z`**, never `‖s‖_2` or any per-block `s_l2_max` surrogate.
Implemented as [`folded_witness_l2_bound_squared`] in `fold_l2_certificate.rs`.

**A-role pricing by realization tier (S11).**

| Tier | A-role `collision_l2_sq` source |
|------|--------------------------------|
| Field-fitting or grouped-carry (certificate emitted) | Lemma 7 on `β_inf` (unchanged; certificate tightens the proved bound, not the rank formula) |
| Deterministic fallback (no certificate) | `L2_BOUND_SQUARED` converted through the L2 MSIS ladder (replaces Lemma 7 on those levels) |

Until S11 wires `select_l2_certificate_realization` into the planner, **all**
fold levels use Lemma 7 plus `collision_l2_sq_for_linf_envelope` on
`8 · mass · β_inf · ν` (Layer 1–2 above). That is conservative on levels that
would later fall back to `L2_BOUND_SQUARED`. No `Gamma · B · ‖s‖_2` triangle
bound is used for security sizing.

### Grouped-Carry L2 Certificate

The realized certificate proves `Z_SQUARED <= B_l2` directly from the committed
digit planes of `z`, without forming a single wrapped field sum and without
revealing any inner product.
It is the default on every certifying level in both transparent and ZK builds.

First convert the inequality to an equality with Lagrange four-square slack:

```text
sum_i z[i]^2 + sum_{h=0}^{3} ell_h^2 = B_l2.
```

The slack integers `ell_0..ell_3` are committed as balanced base-`b` digit planes
`ell_hat` inside `w_next`, exactly like `z_hat`.
Because `ell_h^2 >= 0`, the equality implies `sum_i z[i]^2 <= B_l2`.

Let `K = num_digits_fold`, `b = 2^lb`, and write the committed fold digits as

```text
z[i] = sum_{d=0}^{K-1} b^d · z_hat_d[i],
```

each `z_hat_d[i]` a balanced digit already range-checked by stage 1.
Choose a deterministic group size `g >= 1`, set the grouped basis `B = b^g`,
`R = ceil(K / g)`, and `g_j = min(g, K - jg)` for the last short group.
The grouped limb is a fixed linear view of the committed digits:

```text
z^{<j>}[i] = sum_{t=0}^{g_j-1} b^t · z_hat_{jg+t}[i],   so   z[i] = sum_{j=0}^{R-1} B^j · z^{<j>}[i].
```

Apply the same grouping to each slack integer `ell_h`, and append the four slack
values as four extra coordinates at the tail of the certified domain (length
`N + 4`, `N = coeffs`), so one augmented limb family covers witness and slack:

```text
a^{<j>}(x)     = z^{<j>}[x]      for x < N,
a^{<j>}(N + h) = ell_h^{<j>}     for 0 <= h < 4.
```

This notation is intentionally tied to `z`.
Do not call the grouped limb `u` in the paper or code, because `u` is too easy to
confuse with outer commitment notation.
Suggested code names are `FoldNormGrouping`, `group_digits`, `group_log_basis`,
`grouped_fold_limb`, and `carry_limbs`.

The squared norm is then a polynomial in the grouped basis `B`:

```text
sum_i z[i]^2 + sum_h ell_h^2 = sum_e B^e · C_e,
    C_e = sum_{r+s=e} <a^{<r>}, a^{<s>}>   (ordered pairs, so cross terms count twice).
```

The `C_e` are the un-reduced base-`B` coefficients of the wide squared norm.
Each is a small signed integer (bounded by `D_e`; see the gate below) and **none
is ever revealed**.

**One squared-sum sumcheck.**
Squeeze one challenge `alpha` from the challenge (extension) field after `z_hat`,
`ell_hat`, `carry_hat`, and `B_l2` are transcript-bound.
Define the `alpha`-weighted recomposition, a single polynomial that is a public
linear view of the committed digits:

```text
Z_alpha(x) = sum_{j=0}^{R-1} alpha^j · a^{<j>}(x).
```

Since `sum_x Z_alpha(x)^2 = sum_{r,s} alpha^{r+s} <a^{<r>}, a^{<s>}> = sum_e alpha^e C_e`,
the only quadratic obligation is the single sumcheck

```text
sum_x Z_alpha(x)^2 = V.
```

This one degree-2 instance forms the full ordered-pair convolution implicitly, so
no pairwise inner product is ever materialized or sent: the proof carries the
masked `V` and round polynomials, not `O(R^2)` (or upper-triangular `O(R^2/2)`)
pair claims. The prover forms each honest `C_e` from the upper triangle (`r <= s`,
cross terms doubled) as a constant-factor speedup with no proof impact.

This has the same shape as the field-fitting sumcheck `sum_x z_aug(x)^2 = B_l2`;
the differences are the digit weights (`alpha^j · b^t` instead of `b^d`) and that
the claimed sum `V` is private and tied to the carries below instead of equal to
the public `B_l2`.

**Carry reconciliation, folded by the same `alpha`.**
Let `T_e` be the public base-`B` digits of the bound (`B_l2 = sum_e B^e T_e`) and
let `h_e` be signed carries with `h_0 = 0`, `h_{E+1} = 0`, satisfying

```text
C_e + h_e - T_e - B · h_{e+1} = 0   for every e,
```

whose telescoping gives `sum_e B^e C_e = B_l2`.
The carries are committed as balanced base-`b` digit planes `carry_hat` inside
`w_next`.
Folding all carry equations by the same `alpha` reuses the sumcheck's claimed sum:

```text
V = sum_e alpha^e C_e = sum_e alpha^e T_e + sum_e alpha^e (B · h_{e+1} - h_e).
```

The first right-hand term is a public scalar; the second is a fixed public-linear
view of the committed carries, discharged by one short `ceil(log2 E)`-round linear
claim that reduces to `carry_hat(rho_c)` (batchable into the stage-2 opening).
The individual `C_e` and `h_e` are never sent; only the masked `V` and the masked
limb / carry evaluations appear, so the certificate's sole public scalar is
`B_l2`.

**Field-fitting realization.**
When the whole squared sum and the bound fit the field,

```text
coeffs · digit_abs_max(lb, K)^2 + 4 · B_l2 < q,
```

no grouping or carries are needed: the prover proves `sum_x z_aug(x)^2 = B_l2`
directly with `z_aug = z || ell_0..ell_3` and the public claimed sum `B_l2`.
This is the degenerate single-coefficient instance of the certificate and shares
its machinery (one degree-2 sumcheck plus the `ell_hat` virtualization); the
grouped-carry realization adds only `alpha`, `carry_hat`, and the folded carry
claim.

### Group Selection And No-Wrap Gate

Soundness requires every coefficient `C_e` and every carry `h_e` to be exactly
recoverable as integers, so that each field carry equation is an integer
equation.

Structural coefficient bounds (every balanced digit lies in `[-b/2, b/2 - 1]`,
the same per-coefficient bound `‖s‖_inf = b/2` the scheme already relies on for
committed digits):

```text
A_j     = (b/2) · (b^{g_j} - 1) / (b - 1)        (folded-witness limb j)
A^ell_j = the same bound for the slack limbs
D_e     = sum_{r+s=e} [ N · A_r · A_s + 4 · A^ell_r · A^ell_s ]   (ordered pairs)
```

The honest carry recurrence `h_0 = 0`, `h_{e+1} = (C_e + h_e - T_e) / B` gives the
smallest sound magnitude budget `H_0 = 0`,
`H_{e+1} = ceil( (D_e + H_e + (B - 1)) / B )`.
`H_e` is in general **not** a power of `B` (it is a ceiling of a ratio), so the
carry is never range-checked against `H_e` directly.
Each carry is committed as a balanced-digit cell run inside `carry_hat` with public
base-2 recomposition weights, `h_e = sum_k 2^k · carry_hat[e][k]`, where every cell
is an ordinary balanced digit (`|cell| <= b/2`).
The realizable, power-of-two-granular budget is then

```text
H'_e = (b/2) · (2^{delta_carry(e)} - 1),
```

and `delta_carry(e)` is the smallest cell count with `H'_e >= H_e` (completeness).
Because the cells are ordinary balanced digits, their per-cell bound is exactly the
one the scheme already enforces on `z_hat`; the carry segment adds no new per-cell
range rule, only a public cell count and weight vector in the carry virtualization.
Any committed carry obeys `|h_e| <= H'_e` automatically when the layout matches
`delta_carry(e)`; see **Grouped-Carry Soundness** for why that is enough.

Then `|C_e| <= D_e` and `|h_e| <= H'_e`, so the carry residual
`res_e = C_e + h_e - T_e - B · h_{e+1}` obeys
`|res_e| <= D_e + H'_e + (B - 1) + B · H'_{e+1}`.
The level accepts the grouped-carry certificate only if, for every exponent `e`,

```text
D_e + H'_e + (B - 1) + B · H'_{e+1} < q,
```

where `q` is the characteristic of the accumulation field.
This gate is a **structural eligibility** check: it certifies that any residual
consistent with the public digit and carry layouts cannot wrap modulo `q` as a
single per-exponent integer.
It does **not** by itself prove all `res_e = 0`; that implication comes from the
folded polynomial identity checked at random `alpha` (next section).

Base-2 carry weights keep `H'_e` within a factor 2 of the tight `H_e`.
Reusing the gadget weights `b^k` for carries would round each budget up to a
base-`b` boundary (up to a factor `b`), inflating the dominant `B · H'_{e+1}` term
by roughly `b` and costing about `lb` bits of certifying headroom.
The carry segment is tiny (about `2R - 1` carries, each `delta_carry` cells, versus
`N >= 10^6` witness cells), so the finer base-2 weighting is essentially free in
witness size.

The gate is structural and public.
It is checked from level parameters before any carry value is trusted as an
integer; it is not enough that the honest realized values happen to be small.

The level selects its realization deterministically from public parameters:

```text
if coeffs · digit_abs_max(lb, K)^2 + 4 · B_l2 < q:
    field-fitting realization (no grouping, no carries)
else:
    choose the largest g in 1..max(K-1, 1) such that the per-exponent gate holds for all e;
    if some g works: grouped-carry realization with that g
    else:            no certificate; price A-role at L2_BOUND_SQUARED
```

The largest valid `g` minimizes `R`, hence the carry count `E` and the carry
witness `carry_hat`.
The gate's binding terms are `D_e` and `B·H'_{e+1}`, both of order
`R · N · (b/2)^2` at the middle exponent, so single-digit grouping (`g = 1`)
certifies up to roughly `N ~ q / (R · b^2)` coefficients.
With proof-optimized `lb = 3`, `q ≈ 2^31`, `D = 64`, that is on the order of
`10^4`–`10^5` D64 rings per level (decreasing as `R = num_digits_fold` grows), so
typical fp31 / fp32 dense recursive levels certify rather than fall back; only the
largest levels hit the fallback.

Extension fields do **not** widen the gate.
`F_{q^k}` has characteristic `q`, so every base-embedded coefficient still reduces
modulo `q`. A larger base prime widens the gate; an extension over the same base
prime does not.

### Grouped-Carry Soundness

This section states what the grouped-carry proof obligates, how carries are
constrained, and how soundness combines the no-wrap gate with a folded polynomial
identity.

#### Certified statement

The level proves, on the committed fold-response digits,

```text
sum_i z[i]^2 + sum_h ell_h^2 <= B_l2.
```

Four-square slack turns the inequality into
`sum_i z[i]^2 + sum_h ell_h^2 = sum_e B^e C_e`.
When every per-exponent residual vanishes, the carry chain identifies
`sum_e B^e C_e` with `sum_e B^e T_e = B_l2`.

#### Honest carries are unique

Fix `C_e` from the committed `z_hat` and `ell_hat`, and public `T_e` from `B_l2`
in base `B`.
With `h_0 = 0`, the recurrence

```text
h_{e+1} = (C_e + h_e - T_e) / B
```

defines the honest carry sequence whenever each division is exact in `Z`.
For an honest prover, exactness follows from the four-square equality.
There is no freedom in honest `h_e` beyond the witness.

#### Obligations checked in the proof (malicious prover)

The transcript never sends `C_e` or individual `h_e`.
It checks:

1. **Squared-sum sumcheck:** `sum_x Z_alpha(x)^2 = V`, with `Z_alpha` the
   `alpha`-weighted recomposition of committed grouped limbs in `z_hat` and
   `ell_hat`.
2. **Folded carry claim:** in the challenge field,
   `V = sum_e alpha^e T_e + sum_e alpha^e (B·h_{e+1} - h_e)`, with each `h_e`
   reassembled from committed `carry_hat` cells.
3. **Virtualization:** evaluations from (1) and (2) are public linear functionals
   of the same committed `w_next` opened at stage 2.
4. **Boundary carries:** `h_0 = 0` and the terminal carry `h_{E+1} = 0` (see
   indexing below).
5. **Carry layout:** for each `e`, exactly `delta_carry(e)` balanced cells with
   public base-`2` weights; per-cell magnitudes bounded like `z_hat`.
6. **No-wrap gate:** `D_e + H'_e + (B-1) + B·H'_{e+1} < q` for every `e`, from
   public parameters before reading carry values.

Define

```text
res_e = C_e + h_e - T_e - B · h_{e+1}.
```

Items (1) and (2) imply, in the accumulation field,

```text
P(alpha) = sum_e alpha^e res_e = 0.
```

#### Folded polynomial identity (not per-equation mod-`q` lifting)

The verifier does **not** check `res_e ≡ 0 (mod q)` separately for each `e`.
Soundness treats `P(alpha) = sum_e res_e alpha^e` as a polynomial of degree at
most `E`, where `E = 2R - 2` is the maximal convolution exponent.

If some `res_e` is nonzero in `Z`, then `P` is a nonzero polynomial over the
challenge field (coefficients embed in `F_{q^k}`).
For `alpha` uniform in a large enough challenge domain, `P(alpha) = 0` with
probability at most about `E / |S|` (Schwartz–Zippel / polynomial identity test).

`alpha` is squeezed only after `z_hat`, `ell_hat`, `carry_hat`, and `B_l2` are
bound, so the prover cannot adapt carries to a known `alpha`.

The argument also needs standard sumcheck soundness for
`sum_x Z_alpha(x)^2 = V` and linear-claim soundness for the folded carry
relation on the opened `carry_hat`.
Then, except with negligible probability over `alpha` and the sumcheck coins,
`res_e = 0` for all `e`.

#### Role of the no-wrap gate

The gate is **not** a substitute for the polynomial identity test.
It certifies **level eligibility**: structurally bounded `C_e` and committed
`h_e` are so small that field arithmetic cannot confuse distinct integers when
forming the bounded intermediates behind the sumcheck and carry virtualization.

Under validated digit bounds, `|C_e| <= D_e` and `|h_e| <= H'_e`, so

```text
|res_e| <= D_e + H'_e + (B-1) + B·H'_{e+1} < q.
```

So a **single** per-exponent residual, if checked mod `q` against these bounds,
could not be a nonzero wrap.
Implementation paths that rebuild bounded carry terms in the field cannot conflate
`r` with `r + q` while the gate holds.

The gate is evaluated on the realizable budget `H'_e`, never on tight `H_e` alone.
That is conservative for soundness (stricter gate) at the cost of completeness on
borderline levels.

#### Realizable magnitude `H'_e` (why not range-check `H_e` directly)

`H_e` from the recurrence is generally not a power of `B` and is not witnessed.
Each `h_e` is committed as `delta_carry(e)` balanced cells with base-`2` weights:

```text
h_e = sum_{k=0}^{delta_carry(e)-1} 2^k · carry_hat[e][k],
|carry_hat[e][k]| <= b/2.
```

Hence any committed carry satisfies

```text
|h_e| <= H'_e = (b/2) · (2^{delta_carry(e)} - 1).
```

`delta_carry(e)` is the smallest count with `H'_e >= H_e`.
Base-`2` weights keep `H'_e` within a factor `2` of tight `H_e`; base-`b` carry
weights would inflate budgets by up to a factor `b`.

**Soundness:** there is no separate `|h_e| <= H_e` gadget.
The verifier enforces public `delta_carry(e)`, the virtualization weights, and
per-cell digit bounds.
A prover cannot encode `|h_e| > H'_e` without wrong cell count or out-of-range
digits.

**Completeness:** if `delta_carry(e)` is too small, honest carries may not fit
(prover failure).
The layout must match `fold_l2_certificate::carry_cell_layout` (or equivalent).

**Why `H'_e` is safe despite not equaling `H_e`:** the no-wrap gate uses `H'_e`,
not `H_e`, in the residual bound, so enlarging the representable envelope only
tightens eligibility.
Soundness does not require `H'_e = H_e`; it requires the committed layout to
declare a representable envelope that the gate accepts.

#### Worked example (fp32-style recursive dense level)

The numbers below match `select_l2_certificate_realization` /
`fold_l2_certificate::carry_cell_layout` for a representative certifying level
(similar to the `recursive_dense_level_certifies_on_fp32` unit test).

**Public geometry**

| Parameter | Value |
|-----------|------:|
| Fold coefficient count `N` | 57,344 |
| Fold digits `K = num_digits_fold` | 5 |
| Log basis `lb` (so `b = 2^lb`) | 3 (`b = 8`) |
| Field characteristic `q` | `2^32 - 99` |
| Selected group size `g` | 2 (largest `g` passing the no-wrap gate) |
| Grouped limb count `R = ceil(K/g)` | 3 |
| Grouped radix `B = b^g` | 64 |
| Last-group width | 1 (limb max bounds `A = [36, 36, 4]`) |

With `lb = 3`, each balanced cell satisfies `|cell| <= b/2 = 4`.

**Carry budgets across indices**

`H_e` is the tight structural budget from the recurrence (ignoring public
`T_e`, which only makes the recurrence more conservative).
`δ = delta_carry(e)` is the smallest cell count with `H'_e >= H_e`.
All values are exact integers from the reference formulas in
`crates/akita-types/src/sis/fold_l2_certificate.rs`.

| Carry `h_e` | `D_e` (structural) | `H_e` tight | `δ` | `H'_e` realizable | `H'_e / H_e` |
|-------------|-------------------:|------------:|----:|------------------:|-------------:|
| `h_0` | 74,323,008 | 0 | 0 | 0 | — |
| `h_1` | 148,646,016 | 1,161,297 | 19 | 2,097,148 | 1.806 |
| `h_2` | 90,839,232 | 2,340,740 | 20 | 4,194,300 | 1.792 |
| `h_3` | 16,516,224 | 1,455,938 | 19 | 2,097,148 | 1.440 |
| `h_4` | 917,568 | 280,816 | 17 | 524,284 | 1.867 |

Boundary carries `h_0` and `h_5` (`h_{E+1}` with `E = 4`) are zero; only
`h_1..h_4` need committed cells.

**Zoom in on `h_1` (first nontrivial carry)**

1. **Structural coefficient bound:** `D_1 = 148,646,016` (convolution at exponent 1
   from `N · A_r · A_s` terms with grouped limbs).

2. **Tight budget from the recurrence** (with `h_0 = 0`):
   `H_1 = ceil((D_0 + H_0 + (B-1)) / B) = 1,161,297`.
   This value is not a power of `B` and is not witnessed directly.

3. **Cell layout:** `δ_carry(1) = 19` because
   `H'_1 = (b/2)·(2^19 − 1) = 4 · 524,287 = 2,097,148` is the first
   representable envelope with `H'_1 >= H_1`.
   Nineteen balanced cells per carry index is tiny next to the `N` witness cells.

4. **What the verifier enforces for `h_1`:**
   - exactly 19 cells in the `carry_hat` segment for index `1`;
   - each cell in `[-4, 3]` (balanced base `8`);
   - hence `|h_1| <= H'_1 = 2,097,148` automatically, without a separate
     `|h_1| <= H_1` check.

5. **No-wrap gate at exponent `e = 1`:**
   using realizable budgets `H'_1` and `H'_2`:

   ```text
   D_1 + H'_1 + (B-1) + B·H'_2
     = 148,646,016 + 2,097,148 + 63 + 64·4,194,300
     = 419,178,427
     < q = 4,294,967,197.
   ```

   So any single-exponent residual `res_1` consistent with the structural
   digit and carry layouts cannot be a nonzero wrap mod `q`.
   Soundness still requires the folded identity `P(alpha) = 0` to force
   `res_1 = 0` exactly, not merely mod `q`.

6. **Honest recurrence** (integer division, not shown with real `C_1`, `T_1`):
   with committed `h_1` and `h_2` reassembled from cells,

   ```text
   h_2 = (C_1 + h_1 - T_1) / B.
   ```

   The tight budget `H_1 = 1,161,297` is what the honest prover needs;
   the committed layout allows up to `H'_1 = 2,097,148`, a factor-1.8 envelope
   that keeps base-2 cells within a factor `2` of tight `H_e` and makes the
   no-wrap gate stricter than it would be with `H_e` alone.

**Takeaway:** non-power-of-two `H_e` is handled by rounding up to a representable
`H'_e` via `delta_carry(e)`, not by witnessing `H_e`.
Soundness uses `H'_e` in the gate; completeness needs `H'_e >= H_e` for the
honest carry.

#### Exponent indexing and telescoping

Let `E = 2R - 2`.
Carry indices run `e = 0, 1, …, E`.
Boundary conditions: `h_0 = 0` and `h_{E+1} = 0`.
Proof shape, virtualization, and verifier replay must share this indexing.

Telescoping with `res_e = 0` for all `e` gives `sum_e B^e C_e = sum_e B^e T_e`.

#### What is not claimed

- Carries are not unique among all integer tuples satisfying the folded identity;
  uniqueness holds for the honest witness.
- Individual `C_e` are not sent; magnitudes are structural (`D_e`) and tied to
  committed digits through `Z_alpha`.
- `H_e` is not directly witnessed; only `H'_e` and `delta_carry(e)` appear on the
  wire.

### Sumcheck And Virtualization

The certified statement is about the recomposed integer folded witness, but the
committed object is `w_next`, whose `z_hat`, `ell_hat`, and `carry_hat` segments
hold the base-`b` digit planes.
The certificate has three algebraic parts, all tied back to that one commitment:

1. the single degree-2 squared-sum sumcheck `sum_x Z_alpha(x)^2 = V`;
2. the folded carry claim
   `V = sum_e alpha^e T_e + sum_e alpha^e (B·h_{e+1} - h_e)`; and
3. a linear virtualization tying every evaluation the first two produce to the
   committed digit planes of `w_next`.

The squared-sum sumcheck reduces to the single evaluation `Z_alpha(rho)`.
This is a public linear view of `z_hat` and `ell_hat`: with the digit-major
layout and segment offsets used by `emit_z_folded_block_inner` and the
`fold_gadget = gadget_row_scalars(depth_fold, log_basis)` family, fold digit
`df = jg + t` inside group `j` carries the public weight `alpha^j · b^t`, and the
slack digits carry the analogous scalar gadget weights, with zero outside the
`z_hat` / `ell_hat` segments.

The carry claim reduces (via one short `ceil(log2 E)`-round linear sumcheck, or by
batching into the stage-2 point) to `carry_hat(rho_c)`, a public linear view of
the `carry_hat` segment: reindexing `sum_e alpha^e (B·h_{e+1} - h_e)` puts weight
`B · alpha^{e-1} - alpha^e` on carry `h_e`, and each carry expands as
`h_e = sum_k 2^k · carry_hat[e][k]`, so cell `(e, k)` carries the public weight
`(B · alpha^{e-1} - alpha^e) · 2^k`.

Both `Z_alpha(rho)` and `carry_hat(rho_c)` are discharged onto the existing
`w_next` opening.
They may be merged into the stage-2 opening point when the final `w_next(rho')`
claim can be shared, or kept as adjacent structured linear claims at separately
derived points; the proof shape and descriptor bind the choice.
Either way the certified object is the same committed `w_next`, so a prover cannot
certify one set of digit planes and commit a different recursive witness.

### Stage Placement And Batching

The certificate's only nonlinear part is the one squared-sum sumcheck; everything
else is linear in `w_next`.
Certifying levels therefore add, relative to a deterministic level:

- `ell_hat` and (grouped-carry only) `carry_hat` trailing segments of `w_next`,
- the masked claimed sum `V` (the field-fitting realization uses the public
  `B_l2` instead),
- the squared-sum sumcheck transcript fused into stage 1, and
- the carry linear claim plus limb / carry virtualization, batched into stage 2.

No grouped partial-sum payload is sent.
Each stage derives an unambiguous batching vector from its transcript point.
The descriptor binds the realization (field-fitting, grouped-carry, or
deterministic), the group size `g`, `R`, the carry exponent count `E`, the bound
digits `T_e` layout, and whether the carry / virtualization claims are merged into
the stage-2 point or carried adjacent.
The existing single `CHALLENGE_SUMCHECK_BATCH` scalar is kept only where the
transcript derives a full coefficient vector for every active claim at that stage;
the grouped-carry realization additionally squeezes `alpha` before the squared-sum
sumcheck.

### Footguns For Implementation

- Wire before squeeze.
  `ell_hat` and `carry_hat` are committed through `next_w_commitment`, and `B_l2`
  is transcript-bound, before `alpha` or any squared-sum / carry challenge is
  squeezed.
- The no-wrap gate is structural eligibility, not the full soundness argument.
  Check `D_e + H'_e + (B-1) + B·H'_{e+1} < q` for every `e` from public parameters
  (with the realizable carry budget `H'_e`) before the level may use grouped-carry.
  Per-exponent mod-`q` lifting applies to structurally bounded single residuals;
  vanishing of all `res_e` comes from the folded polynomial identity at random
  `alpha` (see **Grouped-Carry Soundness**).
- Carries are signed, and bounded by the committed cell run layout, not by tight
  `H_e` directly.
  `C_e` can be negative, so `carry_hat` is a balanced decomposition with public
  base-2 recomposition weights.
  Size `delta_carry(e)` as the smallest cell count with
  `H'_e = (b/2)(2^{delta_carry} - 1) >= H_e`, and evaluate the no-wrap gate
  with `H'_e` (never with `H_e`).
  The verifier must pin `delta_carry(e)` in proof shape and reject carry segments
  with the wrong cell count; per-cell magnitudes use the existing `z_hat` digit
  bound, which implies `|h_e| <= H'_e` but not `|h_e| <= H_e`.
  Do not reuse the `b^k` gadget weights for carries (that wastes ~`lb` bits of
  headroom).
- Boundary carries are fixed.
  Enforce `h_0 = 0` and `h_{E+1} = 0`; the terminal zero telescopes the chain to
  `B_l2`.
- Slack can exceed `u64`.
  On small fields `B_l2 - Z_SQUARED` can exceed `2^64`, so the four-square solver
  needs a `u128` target path; the verifier must reject (not assume) any slack that
  does not fit the encoded budget.
- No inner products are sent.
  The certificate never serializes `P_{r,s}` or `Z_SQUARED`; only `B_l2`, the
  committed digit growth, the masked `V`, and the sumcheck / carry transcripts
  appear. Do not reintroduce a public-partial-sum path for transparent builds.
- Reuse the committed `z_hat` layout.
  Limbs, slack, and carries must read the same `num_digits_fold` and physical
  segment layout as `build_w_coeffs`; a second decomposition is allowed only if
  the protocol also proves it recomposes to the committed planes.
- Terminal levels need an explicit policy.
  They may use the same certificate if `z_hat` is committed in `w_next`, or must
  fall back to deterministic pricing.

### Operator-Norm Rejection Sampling

For D=64 the initial production candidate is:

```text
SparseChallengeConfig::ExactShell {
    count_mag1: 31,
    count_mag2: 11,
}
operator_norm_threshold = 16
```

The raw support is:

```text
binom(64, 42) · binom(42, 31) · 2^42 ~= 2^130.152255.
```

To retain 128 accepted challenge bits, the proof must establish:

```text
Pr[gamma(c) <= 16] >= 2^(128 - 130.152255...) ~= 0.225.
```

The implementation must not rely on machine floating-point FFTs to enforce
acceptance.
The reference predicate is the exact negacyclic convolution operator norm.
Let `M_D(c)` be the integer matrix for multiplication by `c` in
`Z[X] / (X^D + 1)`.
Equivalently, for the negacyclic roots
`zeta_j = exp((2j + 1) * pi * i / D)`, `0 <= j < D`,

```text
gamma_D(c)^2
  = lambda_max(M_D(c)^T M_D(c))
  = max_j |sum_k c_k * zeta_j^k|^2.
```

The exact acceptance check is therefore:

```text
accept(c) iff T^2 * I - M_D(c)^T M_D(c) is positive semidefinite over Q.
```

This check is dimension-generic for `D in {32, 64, 128}` and gives the
specification truth value.
It may be implemented by rational interval arithmetic over the DFT expression,
a fraction-free LDL/Sturm certificate for the PSD condition, or an offline
accepted-support certificate whose verifier checks the same predicate.

The production path should use the DFT diagonalization with fixed-point integer
intervals.
Precompute signed `Q`-bit tables

```text
C[j][k] ~= 2^Q * cos((2j + 1) * pi * k / D)
S[j][k] ~= 2^Q * sin((2j + 1) * pi * k / D)
```

with certified componentwise entry error at most `eps_root` in scaled integer
units.
For a candidate challenge, compute integer accumulators

```text
R_j = sum_k c_k * C[j][k]
I_j = sum_k c_k * S[j][k]
A_j = R_j^2 + I_j^2.
```

Let `L1 = ||c||_1` and `r = L1 * eps_root`.
The true scaled real and imaginary parts obey

```text
real(2^Q * c(zeta_j)) in [R_j - r, R_j + r],
imag(2^Q * c(zeta_j)) in [I_j - r, I_j + r].
```

Thus a conservative integer upper bound is

```text
upper_j =
  A_j
  + 2 * (|R_j| + |I_j|) * r
  + 2 * r^2.
```

The sampler accepts only when

```text
upper_j <= T^2 * 2^(2Q)
```

for every `j`.
If any frequency falls in the narrow interval band where the upper bound
rejects but the centered fixed-point value is close to the threshold, the
implementation may either reject the candidate or call the exact predicate.
Always accepting from the lower bound is forbidden unless the exact predicate
has been run.
If the implementation rejects the boundary band without exact fallback, the
accepted-support proof must be for this stricter fixed-point predicate, not for
the mathematical predicate `gamma_D(c) <= T`.

This fast path is deterministic, integer-only, and works without changing the
transcript stream.
`D = 64` with `T = 16` is the first production target, but the same table
format should cover `D = 32` and `D = 128`.
Only one representative of each conjugate frequency pair needs to be checked,
though tests should compare the reduced loop against the full `0..D` formula.
Use `i128` accumulators for the current coefficient ranges and `D <= 128`;
the implementation must validate the worst-case accumulator and square bounds
at construction time.
With, for example, `Q = 48`, `D <= 128`, and `L1 <= 256`, the table error term
is far below one unit in the unscaled operator norm, so the exact fallback
should trigger only for candidates extremely close to the threshold.

The sampler must consume transcript randomness in a stable way.
Rejected candidates are not allowed to create prover/verifier divergence.
The challenge domain separator must include the exact shell parameters and
operator-norm threshold.

### Folded-Witness ∞-Norm Rejection (digit-count tightening)

This is a stacked follow-on to #155, not part of the L2 rank pricing.
Operator-norm rejection (above) and the L2 certificate price the **A-role rank**;
this section instead tightens the **digit count** `num_digits_fold` of the next
recursive witness `z_hat`, which is sized by the `‖z‖_inf` envelope `β_inf` and is
orthogonal to the A-role collision bound.
The analysis is specific to the D=64 exact-shell family.

**What sizes the digit count today.**
`z` enters the next level only through its balanced base-`b` digit planes `z_hat`,
and the plane count `K = num_digits_fold` is what stage 1 turns into the structural
per-coordinate bound `balanced_digit_max(lb, K) = (b/2)·(b^K − 1)/(b − 1)`
([`decomposition_digits.rs::num_digits_fold`]). `K` is chosen so

```text
balanced_digit_max(lb, K) >= β_inf
      = fold_witness_beta(...)
      = num_claims · 2^r_vars · min(‖c‖_inf · ‖s‖_1, ‖c‖_1 · ‖s‖_inf)
```

i.e. `β_inf = T_p · ω · σ_inf` in the worst case (`T_p = num_claims · 2^r_vars`,
`ω = ‖c‖_1`, `σ_inf = ‖s‖_inf`). This worst case assumes all `T_p · ω`
challenge-coefficient products align in sign at one output coordinate, which the
honest fold never attains. The prover already aborts when the realized
`‖z‖_inf` exceeds `β_inf` ([`ring_relation.rs::validate_decompose_fold`]).

**Rejection-sampled tightening.**
Replace the abort with a transcript-bound grind: re-derive the fold challenge from
an incremented nonce, re-fold, and accept the first `z` with `‖z‖_inf <= t` for a
threshold `t < β_inf`. A level then commits the smallest `K` with
`balanced_digit_max(lb, K) >= t`, which crosses a base-`b` digit boundary wherever
`t` and `β_inf` straddle a power of `b`.

**Why it terminates in poly time (D=64).**
The D=64 exact shell `(count_mag1, count_mag2)` places `count_mag1` coordinates of
magnitude 1 and `count_mag2` of magnitude 2 on a uniform support, each nonzero
coordinate carrying an *independent uniform sign* (`sample_exact_shell_challenge`,
`XofCursor::next_sign`). For production `(31, 11)`, `ω = ‖c‖_1 = 53` and the
energy `rho2 = ‖c‖_2^2 = 31 + 4·11 = 75` are fixed (every member meets
`rho2 <= ‖c‖_inf · ‖c‖_1 = 108`).

Fix an output coordinate `r` of `z = sum_{(l,i)} c_{l,i} * s_{l,i}` (the fold of
`T_p` blocks). Expanding the negacyclic products,

```text
z_r = sum_{(l,i)} sum_{a in supp(c_{l,i})} eps_{l,i,a} · m_{l,i,a} · (± s_{l,i, r⊖a}),
```

a signed sum of the independent signs `eps_{l,i,a}` with weights of magnitude
`m_{l,i,a} · |s| <= m_{l,i,a} · σ_inf` (`m = |c| in {1, 2}`). Conditioned on every
support and magnitude pattern, `z_r` is a zero-mean Rademacher sum with variance
proxy

```text
V_r = sum_{(l,i)} sum_a m_{l,i,a}^2 · s_{l,i, r⊖a}^2
    <= σ_inf^2 · sum_{(l,i)} ‖c_{l,i}‖_2^2
    <= T_p · rho2 · σ_inf^2 =: V.
```

Hoeffding for Rademacher sums gives `Pr[|z_r| > t | support, magnitudes] <=
2·exp(−t^2 / 2V)` for every conditioning, hence unconditionally, and a union bound
over the `N = coeffs` coordinates yields the tail

```text
Pr[‖z‖_inf > t] <= 2·N·exp(−t^2 / 2V).        (T)
```

Let `p = Pr[gamma(c) <= Gamma]` be the operator-norm acceptance probability
(`p = 1` when the cap does not bind; production `(31, 11)` with `T = 18` enables
per-level rejection only where Γ pricing lowers A-rank). For challenges from the accepted distribution, Bayes
against (T) on the unconditioned event gives

```text
Pr[‖z‖_inf > t | all T_p blocks accepted] <= Pr[‖z‖_inf > t] / p^{T_p}
                                          <= (2N / p^{T_p}) · exp(−t^2 / 2V),
```

so taking

```text
t >= t* = sqrt( 2V · ln(4N / p^{T_p}) )
        = sqrt( 2 · T_p · rho2 · σ_inf^2 · (ln 4N + T_p · ln(1/p)) )
```

makes the conditional miss probability at most 1/2. Each accepted challenge then
yields `‖z‖_inf <= t*` with probability at least 1/2, so the grind re-folds at
most twice in expectation (poly time). At `p = 1` this is just
`t* = sqrt(2V · ln 4N)`; a binding cap adds the loose `T_p · ln(1/p)` term (the
price of the Bayes step), which keeps termination in `O(1)` expected re-folds for
any constant `p`.

**Soundness is unchanged.**
The verifier never reads the accepting nonce as evidence that `‖z‖_inf <= t*`. It
reads the bound off the committed digits: the stage-1 range check already forces
`|z_r| <= balanced_digit_max(lb, K)`, and the level published `K` before the fold.
The CWSS extractor inspects only accepting transcripts and never how `c` was
sampled, so narrowing the challenge support by the two rejection layers changes
nothing it recovers; the grind's bias on the challenge distribution is absorbed
into the standard Fiat-Shamir knowledge error `(Q+1)·κ` for the `Q` random-oracle
queries an adversary makes, the only contract being that the accepted support
retains `λ + log2 Q` bits (Accepted-challenge entropy invariant). Lowering `K`
tightens the `‖z‖_inf` the verifier structurally enforces without touching
binding.

**How much is gained (D=64), and the gap to honest folds.**
At `p = 1`,

```text
t* / β_inf = sqrt(2 · rho2 · ln 4N) / (ω · sqrt(T_p)),
```

independent of `σ_inf` and growing only as `sqrt(ln N)`. For `(rho2, ω) =
(78, 54)` and `N ≈ 2^16`, this is `≈ 0.41, 0.29, 0.20, 0.14` at fold widths
`T_p = 4, 8, 16, 32`, so the rigorous threshold sits inside one base-`b` digit of
the worst case at the wider folds. It is loose against the measured folds in the
calibration tables above by roughly another order of magnitude, because `V` charges
every source coordinate at the worst case `σ_inf` while honest source blocks are
far smaller in mean square (`mu2_implied`). Production thresholds are calibrated
against the realized response (the `z_rms` / `mu2_implied` tables) and are
correspondingly tighter; (T) is only what guarantees termination.

**Transcript and planner consequences (later slices).**
- The grind nonce is bound before the challenge is squeezed (wire-before-squeeze),
  exactly like the op-norm rejection stream, so the verifier replays it
  deterministically and re-derives the same challenge.
- The threshold policy `t(level, family)` feeds `β_inf` / `num_digits_fold`, must
  be planner-visible (it changes the schedule DP where a lowered `K` crosses a
  `2^lb` boundary), and is bound in the instance descriptor.
- `validate_decompose_fold` becomes the grind loop with a capped attempt count;
  exceeding the cap is a prover-only error (not verifier-reachable).

### SIS Tables And Planner

The current SIS tables are keyed by a rounded coefficient `L∞` collision bucket.
The cutover needs generated L2 MSIS tables keyed by the Euclidean bound used in
the security model.

Required changes:

- Rename table fields away from `collision_inf`.
- Generate audited L2 bucket ladders and secure-rank floors.
- Update `AjtaiKeyParams` to carry the L2 bound name and descriptor bytes.
- Update `LevelParams`, schedule entries, generated-family tables, and runtime
  DP fallback to use L2 bounds.
- Delete the committed-fold L∞ *rank* derivation after all call sites move.
  Retain the folded-witness L∞ bound `fold_witness_beta` and the prover's
  `beta_linf_fold_bound` / `validate_decompose_fold` abort: they no longer price
  the A-role rank, but they still size the digit count of the next recursive
  witness `z_hat` (`num_digits_fold` depends on `||z||_inf`).
- Keep the B-role and D-role opening-digit collisions at their natural
  coefficient bound `2^lb - 1` (the difference of two balanced digits), and
  convert each into the unified L2 table by `||v||_2 <= sqrt(d) · ||v||_inf`.
  The generated tables and public docs must state this conversion explicitly so
  a single L2 MSIS floor covers all three roles.

### Public Security Model Documentation

The public repo documentation must change together with the code.
At minimum:

- Replace the current `MSIS_{q,d}(n,m,eta)` definition using
  `||z||_inf <= eta` with the Euclidean norm used by the implementation.
- State the accepted challenge distribution and its min-entropy after rejection.
- Define `gamma(c)` as the negacyclic convolution operator norm.
- State that A-role sizing uses the existing Lemma 7 bound converted by
  `||v||_2^2 <= d · ||v||_inf^2` (no alternate weak-binding derivation).
- State the grouped-carry folded-witness certificate, the per-exponent no-wrap
  gate as structural eligibility, the folded polynomial identity
  `P(alpha) = sum_e alpha^e res_e = 0` at random `alpha`, and the representable
  carry layout (`H'_e`, `delta_carry(e)`) that enforces `|h_e| <= H'_e` without
  witnessing tight `H_e` directly.
- Explain the full cutover and remove superseded L∞ **estimator** language while
  keeping the Lemma 7 collision formula.

## Architecture

### Data Flow

```text
sample accepted c_i with gamma(c_i) <= Gamma
        |
        v
decompose_fold: centered_coeffs for z = sum_i c_i * s_i
        |
        +--> z_hat digit planes
        |        |
        |        +--> select realization + FoldNormGrouping from public params
        |        |
        |        +--> four-square slack ell -> ell_hat ; carries h -> carry_hat
        |        |
        |        +--> squeeze alpha ; squared-sum sumcheck sum_x Z_alpha(x)^2 = V
        |        |
        |        +--> carry claim V = sum_e alpha^e T_e + <c(alpha), carry_hat>
        |                 |
        |                 v
        |        virtualization onto committed z_hat/ell_hat/carry_hat in w_next
        |
        +--> z_folded_rings (z_hat), w_hat, t_hat, r_hat, ell_hat, carry_hat
                 |
                 v
build_w_coeffs (w_next) -> commit next w -> stage 1 -> stage 2 (+ L2 sumchecks)
```

### Affected Files

Likely primary files:

- `crates/akita-challenges/src/config.rs`
- `crates/akita-challenges/src/sampler/`
- `crates/akita-types/src/sis/norm_bound.rs`
- `crates/akita-types/src/sis/ajtai_key.rs`
- `crates/akita-types/src/sis/generated_sis_table/`
- `crates/akita-types/src/sis/decomposition_digits.rs`
- `crates/akita-types/src/layout/params.rs`
- `crates/akita-types/src/proof/levels.rs`
- `crates/akita-types/src/proof/shapes.rs`
- `crates/akita-types/src/proof_size.rs`
- `crates/akita-planner/src/schedule_params.rs`
- `crates/akita-config/src/proof_optimized.rs`
- `crates/akita-prover/src/lib.rs`
- `crates/akita-prover/src/protocol/ring_relation.rs`
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` (`emit_z_folded_block_inner`
  is the `z_hat` digit-plane source for grouped limbs)
- `crates/akita-types/src/proof/ring_relation.rs` (`ring_column_z_first`, segment
  layout: expose the committed `z_hat` offset for grouped virtualization)
- `crates/akita-prover/src/protocol/sumcheck/akita_stage2/`
- `crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs`
  (`compute_setup_contribution` / `fold_gadget`: the structured-matrix evaluation
  the `G'` virtualization reuses)
- `crates/akita-verifier/src/stages/stage2.rs`
- `crates/akita-verifier/src/protocol/levels.rs`
- `scripts/gen_sis_table.py`
- `specs/weak-binding-norm-fix.md`

## Alternatives Considered

### Keep L∞ SIS And Add L2 As An Optimization Hint

Rejected.
If security still uses L∞, the L2 proof does not improve SIS ranks.
It only adds proof bytes.

### Use Honest Average L2 Norms Without A Certificate

Rejected.
The extractor needs a bound on accepted adversarial transcripts, not on an
honest simulation distribution.
Experiments are useful for selecting thresholds and estimating prover abort
rates, but they are not a security argument.

### Use Only Operator-Norm Challenge Rejection

Rejected as incomplete.
`gamma(c) <= Gamma` bounds multiplication as an operator, but the proof still
needs the folded response to be short under the new MSIS norm.
The verifier must either see or certify the relevant folded-witness L2 bound.

### Keep Both L∞ And L2 Schedule Tables

Rejected.
This repo makes no backward-compatibility guarantee, and dual schedule tables
would invite drift between the security model, planner, prover, and verifier.
The cutover should replace the old path.

### Publish Grouped Partial Sums In The Clear (transparent-only)

Rejected as the default.
Publishing the `P_{r,s}` partial sums avoids slack and carries, but it leaks the
full digit-group Gram matrix of the folded witness, so it cannot be the ZK path
and would force two certificate protocols.
The grouped-carry certificate sends no inner products, costs only `ell_hat` /
`carry_hat` plus one short linear claim more than the field-fitting case, and is a
single design for transparent and ZK builds, so it is preferred even though the
public-partial-sum variant is marginally simpler on large fields.

## Execution

The work decomposes into 13 slices across six tracks (challenge family, L2 SIS,
proof shape, prover, verifier, planner/transcript/tests).
Four slices are independent and can start immediately; the rest serialize behind
the L2 norm/table API (S4, S5) and the proof-shape change (S6).

Status on `main` + PR #155 + PR #195:

- **S1, S4, S5a, S5b**: done (#155).
- **S3** *(#195)*: production D=64 `(31, 11)` / `T = 18`, per-level
  `op_norm_rejection`, certified acceptance floor `117/500`, sampler + fold_draw +
  verifier stage-1 replay; shipped `fp128_d64_*` tables regen. S2 vendored cert
  in CI still open.
- **S7** *(#195)*: `four_squares_u128` path for slack above `2^64`.
- **S8 geometry** *(#195, types-only)*: `fold_l2_certificate.rs` (realization
  selection, no-wrap gate, carry layout); not wired to prover/verifier.
- **S11 partial** *(#195)*: op-norm rejection in DP/expand + schedule regen;
  certificate-tier `B_l2_pub` pinning and deterministic-fallback rank pricing
  wait for S6/S11.
- **S6, S8 prover, S9, S10, S13**: not started on #195.

### Decisions To Lock (gating)

These are the former Phase 0 items.
Each gates specific slices, noted in parentheses.

- L2 MSIS definition and estimator input convention, including the B/D
  `||v||_2 <= sqrt(d)·||v||_inf` conversion into the single L2 table. (S4, S5)
- D=64 exact-shell operator-norm acceptance lower bound, and the fallback if it
  lands below `0.225` (larger shell or higher `T`). (S2)
- Production D=64 shell and threshold: `(31, 11)`, `T = 18`. (S2, S3).
  Per-level `op_norm_rejection` on `LevelParams` enables the rejection oracle
  only when Γ collision pricing strictly lowers audited A-rank.
- Public `B_l2_pub` from challenge second moment and witness-class envelope (not
  `‖s‖_2` surrogates or realized `Z_SQUARED`); raw bound for geometry, optional
  ladder pin at schedule time (S6, S11). Deterministic-fallback A-role pricing
  from `L2_BOUND_SQUARED` waits for S11. (S4, S8, S11)
- Certificate placement: whether the limb / carry virtualization claims are
  merged into the stage-2 point or kept adjacent at a separately derived point. (S9)
- The per-exponent no-wrap gate `D_e + H'_e + (B-1) + B·H'_{e+1} < q` for every
  convolution exponent (`H'_e` from the committed carry cell run), the realization
  selection (field-fitting vs grouped-carry), and the per-level fallback to the
  deterministic bound when no group size satisfies the gate. (S8, S10)
- The canonical Euclidean MSIS estimator and table-generation command: see
  [`sis-euclidean-estimator.md`](sis-euclidean-estimator.md) (S5a). (S5b consumes its output.)

### Slice Dependency Graph

```text
WAVE 0  (independent, start now, parallel)
  S1  op-norm predicate gamma_D(c) <= T     [akita-challenges, pure]   DONE
  S7  four-square slack helper (default path)  [pure algorithm]       DONE
  S4  L2 norm primitives (Lemma 7 + l2_sq_from_linf)  [akita-types::sis]  DONE
  S2  D=64 support lower bound >= 128 bits   [research / certificate]

WAVE 1
  S5a lattice-estimator pin + gen_sis_table   (spec: sis-euclidean-estimator.md)  DONE
  S5b L2 SIS tables + collision_l2_sq rename  (S4, S5a)                         #155
  S3  threshold + transcript rejection       (S1)   DONE #195 (S2 cert in CI open)
  S6  proof shape / serialization / size     (parameterize B_l2 early)

WAVE 2
  S8  prover certificate assembly            (S4, S6, S7)
  S11 planner + shipped tables + drift       (S4, S5, S6)
  S12 transcript instance-descriptor bind    (S3, S5, S6)

WAVE 3
  S9  squared-sum sumcheck + carry claim + virtualization  (S6, S8)
  S10 verifier replay + no-panic             (S6, S9)

WAVE 4
  S13 e2e tamper tests + ZK parity           (all)
```

### Slices

Each slice lists its crate/files, deliverable, and dependencies.
Per-slice test obligations are in the Testing Strategy section.

**S1 — Exact operator-norm acceptance predicate.** *(independent, DONE)*
`crates/akita-challenges/src/sampler/op_norm.rs`.
Implements `gamma_D(c) <= T` as the integer DFT predicate: a certified `pi`
enclosure (Machin series in `i128`) feeds interval-Taylor cos/sin tables at scale
`2^q` carrying a sound `eps_root`; the predicate forms integer accumulators
`R_k, I_k` and accepts only when the conservative upper bound
`R_k^2 + I_k^2 + 2(|R_k|+|I_k|)r + 2r^2 <= T^2 2^{2q}` (with `r = ||c||_1 eps_root`)
holds for every reduced frequency, rejects when a lower bound already exceeds the
threshold, and reports the boundary band as indeterminate (treated as reject).
Dimension-generic for `D in {4, 32, 64, 128}`; worst-case `i128` accumulator and
square bounds validated at construction. No floating point on the decision path.

**S2 — D=64 accepted-support lower bound.** *(independent, research)*
Establish a rigorous `>= 128`-bit accepted-support lower bound for shell
`(31, 11)` at `T = 16` (`Pr[gamma(c) <= 16] >= 0.225`), as a checked certificate
artifact rather than full enumeration.
Decide the fallback (larger shell or higher `T`) if it lands short.
Gates the production policy in S3.

**S4 — L2 norm primitives.** *(independent, DONE)*
`crates/akita-types/src/sis/norm_bound.rs`.
Adds `committed_fold_collision_l2_sq` / `rounded_up_collision_norm_s` (Lemma 7 on
fold response `z` via `β_inf = fold_witness_beta`, then `collision_l2_sq_for_linf_envelope`),
and the B/D `collision_l2_sq_for_linf_envelope` on `2^lb − 1`
(`||v||_2^2 <= d·||v||_inf^2`). Squared
domain keeps every value an exact `u128` integer (`sqrt(D)` is irrational for
`D ∈ {32, 128}`); the real square root is taken only at bucket/slack selection
(S8). `fold_witness_beta` prices both `num_digits_fold` and the A-role collision
through `β_inf`. Realized `Z_SQUARED` certificates are implemented in S8, not
here.

**S7 — Four-square slack helper.** *(DONE; #155 + #195 `u128` path)*
`crates/akita-types/src/sis/four_square.rs`.
Pure helper computing `ell_0..ell_3` with `sum ell_j^2 = B_l2 - Z_SQUARED`. A
Rabin–Shallit-style prime hunt is the fast path; a theorem-backed finite
two-squares-residual fallback makes the solver total. Integer-only decision path
(no floating point).
The four-square slack is committed as `ell_hat` on every realized level (not just
ZK), so the certificate proves an equality; small-field levels need a `u128`
target path because `B_l2 - Z_SQUARED` can exceed `2^64` *(#195)*.

**S3 — Threshold + transcript-stable rejection sampling.** *(DONE in #195)*
`crates/akita-challenges/src/config.rs`, `sampler/exact_shell.rs`, `sampler/mod.rs`,
`fold_draw.rs`; `LevelParams.op_norm_rejection`; planner
`choose_op_norm_rejection_for_a_role`; prover `fold_grind` + verifier stage-1 replay.
Production D=64 ships `(31, 11)` with `T = 18` and rational acceptance floor
`117/500`. Per-level rejection runs only when Γ pricing strictly lowers audited
A-rank. Vendored S2 checked certificate in CI remains a follow-up.

**S8a — Certificate geometry (types-only).** *(DONE in #195)*
`crates/akita-types/src/sis/fold_l2_certificate.rs`.
Realization selection, grouped-digit layout, structural no-wrap gate, carry cell
budgets, and `folded_witness_l2_bound_squared`. Uses raw `B_l2_pub` from
[`fold_witness_l2_pub_bound_sq`]. No prover/verifier consumers yet.

**S5a — Euclidean SIS table regen (lattice-estimator).** *(done in #155)*
[`specs/sis-euclidean-estimator.md`](sis-euclidean-estimator.md).
Vendor the open lattice-estimator reliability PR branch as `third_party/lattice-estimator`,
harden `scripts/gen_sis_table.py`, and check in Akita-local golden CSV under
`scripts/sis_golden/`. Repoint to `malb/lattice-estimator` after upstream merge. No Rust
estimator crate.

**S5b — L2 SIS tables + key rename.** *(S4, S5a; done in #155)*
`crates/akita-types/src/sis/{ajtai_key,generated_sis_table}.rs`.
Regenerate and stitch two key families: power-of-two buckets
(`2^MIN_LOG_BUCKET .. 2^MAX_LOG_BUCKET`) plus derived `K = d · B²` for
`COEFF_LINF_BUCKETS`; rename `collision_inf` to `collision_l2_sq` (`u128`) across
`AjtaiKeyParams`, `min_secure_rank`, `ceil_supported_collision`, and descriptor bytes;
route A/B/D norm-bound pricing through `collision_l2_sq_for_linf_envelope` (derived key
default, pow2 fallback). Remove the old committed-fold L∞ rank-pricing paths. Regen remains
Sage + `scripts/{gen_sis_table,stitch_generated_sis_table}.py` against the pinned submodule.

**S6 — Proof shape, serialization, proof size.** *(parameterizable early)*
`crates/akita-types/src/proof/{levels,shapes}.rs`, `proof_size.rs`,
`proof/ring_relation.rs`.
Add the realization marker (field-fitting / grouped-carry / deterministic), the
`FoldNormGrouping` descriptor (`group_digits`, `group_count`, last-group width,
carry exponent count `E`, per-exponent carry cell counts `delta_carry`), the
`B_l2` scalar, trailing `ell_hat` / `carry_hat` offsets, the masked `V` claim,
shape validation, serialization tests, and the proof-size formula. No partial-sum
payload.
Parameterized on `B_l2`'s type, so it does not wait on S5's values.

**S8 — Prover certificate assembly.** *(S4, S6, S7, S8a)*
`crates/akita-prover/src/protocol/ring_relation.rs`, `ring_switch/coeffs.rs`.
From `DecomposeFoldWitness.centered_coeffs`, compute `Z_SQUARED`, the four-square
slack `ell` (and `ell_hat`), select the realization and the largest group size
`g` satisfying the per-exponent no-wrap gate (evaluated with the realizable carry
budget `H'_e`), derive the carries `h` and decompose each into `delta_carry(e)`
balanced cells with base-2 recomposition weights (`carry_hat`), and fall back to
deterministic `L2_BOUND_SQUARED` when no `g` passes. Append `ell_hat` /
`carry_hat` to `w_next` and transcript-bind them and `B_l2` before squeezing
`alpha` (wire-before-squeeze).

**S11 — Planner + generated tables + drift guards.** *(S4, S5, S6; partial #195)*
`crates/akita-planner`, `crates/akita-config`.
Update the runtime DP, regenerate shipped schedules, update proof-size formulas,
run the generated-table drift guards, and retarget profile modes if the secure
family set changes. *(#195: op-norm rejection in DP/expand + D64 schedule regen;
certificate `B_l2_pub` pin, realization tier on `LevelParams`, and
deterministic-fallback rank from `L2_BOUND_SQUARED` wait for S6.)*

**S12 — Transcript instance-descriptor binding.** *(S3, S5, S6)*
`crates/akita-config` transcript binding.
Bind the active MSIS norm model, challenge family + operator-norm threshold, L2
bound policy, certificate shape, the number and order of stage-2 claims, and the
schedule.
Pin the descriptor bytes with a test; a proof under L2 must not verify under old
L∞ parameters.

**S9 — Squared-sum sumcheck + carry claim + virtualization.** *(S6, S8)*
`crates/akita-prover/src/protocol/sumcheck/akita_stage2/`, verifier
`slice_mle/setup_contribution.rs`.
Run the single degree-2 sumcheck `sum_x Z_alpha(x)^2 = V`, the folded carry claim
`V = sum_e alpha^e T_e + <c(alpha), carry_hat>` (a short linear sumcheck), then
discharge `Z_alpha(rho)` and `carry_hat(rho_c)` through linear virtualization onto
the committed `w_next` opening. Reuse `fold_gadget` / `compute_setup_contribution`
structured evaluation for the `z_hat` / `ell_hat` / `carry_hat` segments.
Certifying levels bind the active claim vector and whether the claims are merged
into stage 2 or carried adjacent.

**S10 — Verifier replay + no-panic.** *(S6, S9)*
`crates/akita-verifier/src/stages/stage2.rs`, `protocol/levels.rs`.
Recompute the realization and no-wrap gate from public params, replay the
squared-sum sumcheck and carry claim, check `h_0 = h_{E+1} = 0`, validate `B_l2`,
`ell_hat`, and `carry_hat` lengths / digit bounds / offsets, confirm every
evaluation is anchored to the committed `w_next`, and reject every malformed
challenge / certificate / shape with `AkitaError` / `SerializationError`.

**S13 — End-to-end + ZK parity.** *(all)*
End-to-end prover/verifier tests that fail under independent tampering of the
committed folded witness, `ell_hat`, `carry_hat`, `B_l2`, the next-witness
commitment, and the ring-relation rows. Because the certificate is one design for
both builds, the ZK path runs the same claims with masking on; a test asserts no
inner-product payload is serialized under `feature = "zk"`.

## Open Questions

1. Resolved: [`specs/sis-euclidean-estimator.md`](sis-euclidean-estimator.md) defines the
   canonical offline regen path: general fixes in `malb/lattice-estimator`, pinned submodule,
   `scripts/gen_sis_table.py`, and Akita-local golden checks (no in-repo Rust estimator).
2. Resolved: `B_l2_pub` is a fixed public per-level parameter from
   `challenge_l2_sq_per_block · num_fold_blocks · witness.l1_norm · witness.infinity_norm`,
   rounded up on the L2 MSIS ladder (`fold_witness_l2_pub_collision_bucket` in code).
   The prover may abort if realized `Z_SQUARED` exceeds `B_l2_pub` (same as any other
   public bound); it must not pick a tighter witness-dependent bucket at prove time.
3. Resolved: the default realized certificate is the grouped-carry design.
   It commits `ell_hat` (four-square slack) and `carry_hat` (carry limbs) in
   `w_next`, proves one squared-sum sumcheck `sum_x Z_alpha(x)^2 = V`, and folds
   the carry chain into one linear claim. It sends no inner products, so it is the
   single design for transparent and ZK builds.
4. Resolved (virtualization worked out in "Sumcheck And Virtualization"): the
   squared-sum sumcheck reduces to `Z_alpha(rho)` and the carry claim to
   `carry_hat(rho_c)`, both discharged onto the committed `w_next` via the
   existing `z_hat` / `ell_hat` / `carry_hat` digit layout and `fold_gadget`
   weights.
   The residual implementation choice is whether those linear claims are merged
   into the existing stage-2 point or kept adjacent at a separately derived point.
5. For tensor challenges, should `Gamma` be derived from factor operator norms,
   the expanded product challenge, or a separate accepted tensor-product policy?
6. Resolved: the B/D roles keep their coefficient `L∞` digit-collision bound
   `2^lb - 1` and convert into the unified L2 table via
   `||v||_2 <= sqrt(d)·||v||_inf` (see Invariants and SIS Tables And Planner).
7. Resolved: small-field (31/32-bit) realized L2 certificates use the
   grouped-carry realization with the per-exponent no-wrap gate
   `D_e + H'_e + (B-1) + B·H'_{e+1} < q` (`H'_e` the committed carry cell run's
   realizable budget, base-2 weights), certifying rather than falling back on
   typical fp31 / fp32 dense recursive levels.

## References

- `specs/sis-euclidean-estimator.md` (S5a: offline estimator + table regen)
- `specs/weak-binding-norm-fix.md`
- `specs/bounded-l1-sparse-challenge.md`
- `specs/tensor-structured-folding-challenges.md`
- `crates/akita-types/src/sis/norm_bound.rs`
- `crates/akita-prover/src/protocol/ring_relation.rs`
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` (`build_w_coeffs`)
- `crates/akita-prover/src/protocol/sumcheck/akita_stage2/mod.rs`
- `crates/akita-verifier/src/protocol/slice_mle/setup_contribution.rs`
