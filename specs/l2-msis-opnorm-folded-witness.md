# Spec: L2 MSIS Cutover, Operator-Norm Challenges, and Folded-Witness L2 Certificates

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao, Cursor agent draft |
| Created     | 2026-06-04 |
| Status      | proposed, draft for iteration |
| PR          | https://github.com/LayerZero-Labs/akita/pull/148 |

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
      Lemma 7 on fold response `z` (`8·ω·β_inf·ν` → `l2_sq_from_linf`), and
      `β_inf = fold_witness_beta`. Full public security-doc cutover completes with
      S6+ certificate wording.
- [x] *(#155, S5b)* `akita_types::sis` exposes `committed_fold_collision_l2_sq` /
      `rounded_up_collision_norm_s`; `collision_inf` is removed from production
      call sites (`collision_l2_sq` on `AjtaiKeyParams`).
- [x] *(#155, S3 infra)* Exact-shell operator-norm rejection sampling,
      `operator_norm_cap`, and descriptor binding are implemented. Production
      D=64 keeps `(30, 12)` with `T = 54` (no rejection) until S2 certifies
      `(31, 11), T = 16`.
- [ ] The D=64 accepted family has a rigorous support lower bound of at least
      128 bits, not just a Monte Carlo estimate.
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
      cell run's realizable budget), so each carry residual that is zero modulo
      `q` is zero as an integer. A test pins which levels choose field-fitting,
      grouped-carry, or deterministic-fallback tiers.
- [ ] The certified statement is over the committed `z_hat` / `ell_hat` /
      `carry_hat` digit planes, and a test ties every limb, slack, and carry
      evaluation to the committed `w_next` segment via gadget recomposition (a
      tampered `z_hat`, `ell_hat`, or `carry_hat` fails the check).
- [x] *(#155, S5b)* B-role and D-role collisions use `l2_sq_from_linf` on `2^lb − 1`
      (`rounded_up_collision_norm_t/w`). Dedicated table-conversion test remains a
      follow-up; pricing path is wired.
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
- [x] *(#155, S5b)* Runtime DP, `expand_to_level_params`, and shipped generated
      schedule tables size A-role ranks from `collision_l2_sq`; `num_digits_fold`
      still uses `β_inf`. Certificate-tier `B_l2` sizing waits for S6.
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
  against a direct `Σ z[i]^2` reference, and no-wrap-gate fallback.

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
AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=32 cargo run --release --example profile
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
- Current production D=64 exact shell is `(30, 12)`, so
  `rho2 = 30 + 4 * 12 = 78`.
  Proposed cutover shell `(31, 11)` would give `rho2 = 75`.
- Formula comparisons use candidate `Gamma = 16`.
  The profile sampler does **not** yet enforce operator-norm rejection.
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
| `h_e` | Signed carry, `C_e + h_e − T_e − B·h_{e+1} = 0`, `h_0 = h_E = 0`; committed as `carry_hat` |
| `δ_carry(e)` | Balanced digit cells committed for `h_e` (base-2 weights); realizable budget `H'_e = (b/2)(2^{δ_carry}−1) ≥ H_e` |
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
collision_l2_sq   = ceil_bucket(d · collision_A_inf^2).
```

`committed_fold_collision_l2_sq` / `rounded_up_collision_norm_s` in
`crates/akita-types/src/sis/norm_bound.rs` implement this path.

### Layer 2 — Deterministic `‖z‖_2` envelope (shipped, pre-certificate)

No realized `‖z‖_2` certificate in production yet. The planner bounds
`‖z‖_inf` by `β_inf` and converts to L2 for the estimator:

```text
β_inf = fold_witness_beta(...)
      = num_claims · 2^r_vars ·
        min(||c||_inf · ‖s‖_1, ‖c‖_1 · ‖s‖_inf),

‖z‖_inf  ≤  β_inf,
‖z‖_2   ≤  √d · β_inf   (hence collision_l2_sq = ceil(d · (8ωβ_inf ν)²)).
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
Z_SQUARED = sum_{row, coeff} z[row][coeff]^2 <= B_l2
```

without revealing `Z_SQUARED` or any folded-witness inner product, in both
transparent and ZK builds.
When a certificate is emitted, the prover chooses a bucket `B_l2` from the L2
MSIS ladder with `Z_SQUARED <= B_l2`. The deterministic ceiling (no-wrap gate
fallback when no certificate is emitted) is

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

Until this path ships, A-role table lookup uses only Lemma 7 plus
`l2_sq_from_linf` on `8 · ω · β_inf · ν` (Layer 1–2 above). No
`Gamma · B · ‖s‖_2` triangle bound is used for security sizing.

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
let `h_e` be signed carries with `h_0 = 0`, `h_E = 0`, satisfying

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
coeffs · balanced_digit_max(lb, K)^2 + 4 · B_l2 < q,
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
one the scheme already enforces on `z_hat`; the carry segment adds no new range
obligation, only a different public weight vector in the carry virtualization.

Then `|C_e| <= D_e` and `|h_e| <= H'_e`, so the carry residual
`res_e = C_e + h_e - T_e - B · h_{e+1}` obeys
`|res_e| <= D_e + H'_e + (B - 1) + B · H'_{e+1}`.
The level accepts the grouped-carry certificate only if, for every exponent `e`,

```text
D_e + H'_e + (B - 1) + B · H'_{e+1} < q,
```

where `q` is the characteristic of the accumulation field.
Under this gate the `alpha`-folded carry claim, which forces `res_e ≡ 0 (mod q)`,
forces `res_e = 0` as an integer, because the only multiple of `q` with absolute
value below `q` is `0`.

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
if coeffs · balanced_digit_max(lb, K)^2 + 4 · B_l2 < q:
    field-fitting realization (no grouping, no carries)
else:
    choose the largest g in 1..K-1 such that the per-exponent gate holds for all e;
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
- The no-wrap gate is on the carry residual, not the realized values.
  Check `D_e + H'_e + (B-1) + B·H'_{e+1} < q` for every `e` from public parameters
  (with the realizable carry budget `H'_e`) before trusting any carry equation as
  an integer equation.
- Carries are signed, and bounded by the committed cell run, not by `H_e`.
  `C_e` can be negative, so `carry_hat` is a balanced decomposition with public
  base-2 recomposition weights. Size `delta_carry(e)` as the smallest cell count
  with `H'_e = (b/2)(2^{delta_carry} - 1) >= H_e`, and evaluate the no-wrap gate
  with `H'_e` (never with `H_e`). The cells are ordinary balanced digits, so they
  need no range machinery beyond the existing `z_hat` digit bound; do not reuse the
  `b^k` gadget weights for carries (that wastes ~`lb` bits of headroom).
- Boundary carries are fixed.
  Enforce `h_0 = 0` and `h_E = 0`; the closing `h_E = 0` is what forces the total
  to match `B_l2`.
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
  gate, and why each carry residual that is zero modulo `q` is an exact integer
  zero (certificate tier).
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
- `crates/akita-types/src/sis/generated_sis_table.rs`
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

Status: S1 (`crates/akita-challenges/src/sampler/op_norm.rs`), S7
(`crates/akita-types/src/sis/four_square.rs`), and the S4 L2 norm primitives
(`crates/akita-types/src/sis/norm_bound.rs`, squared-domain) are implemented as
pure, not-yet-wired building blocks on `main`.

Implementation on branch `quang/s3-s5-sis-estimator-spec` (PR #155) and later slices:

- **S5a** ([`sis-euclidean-estimator.md`](sis-euclidean-estimator.md)): upstream
  lattice-estimator reliability fixes, vendored LE PR branch submodule, hardened
  `scripts/gen_sis_table.py`, and Akita golden. *(Done in #155.)*
- **S5b** (same #155): L2 table regen + stitch, `collision_l2_sq` rename, wire A-role
  and B/D L2 pricing from `norm_bound.rs`. *(In progress.)*
- **S3**: operator-norm threshold + transcript rejection (blocked on **S2** for the
  production D=64 shell/threshold; see below).
- **S6, S8–S13**: proof shape, certificate, planner schedules, e2e (unchanged).

### Decisions To Lock (gating)

These are the former Phase 0 items.
Each gates specific slices, noted in parentheses.

- L2 MSIS definition and estimator input convention, including the B/D
  `||v||_2 <= sqrt(d)·||v||_inf` conversion into the single L2 table. (S4, S5)
- D=64 exact-shell operator-norm acceptance lower bound, and the fallback if it
  lands below `0.225` (larger shell or higher `T`). (S2)
- Production D=64 shell and threshold; starting candidate `(31, 11)`, `T = 16`. (S2, S3).
  **Frozen on `main` until S2 certifies:** keep `ExactShell { count_mag1: 30, count_mag2: 12 }`
  with no `operator_norm_threshold` field. Do not land `(31, 11), T = 16` in
  `proof_optimized` presets until the S2 accepted-support lower bound is a checked artifact.
- Certificate `Z_SQUARED` ceiling from `β_inf` and `balanced_digit_max` per
  level (not `‖s‖_2` surrogates). (S4, S8)
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
  S3  threshold + transcript rejection       (S1, S2 for production policy)
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
fold response `z` via `β_inf = fold_witness_beta`, then `l2_sq_from_linf`), and
the B/D `l2_sq_from_linf` (`||v||_2^2 <= d·||v||_inf^2`) conversion. Squared
domain keeps every value an exact `u128` integer (`sqrt(D)` is irrational for
`D ∈ {32, 128}`); the real square root is taken only at bucket/slack selection
(S8). `fold_witness_beta` prices both `num_digits_fold` and the A-role collision
through `β_inf`. Realized `Z_SQUARED` certificates are implemented in S8, not
here.

**S7 — Four-square slack helper.** *(independent, DONE; on the default path)*
`crates/akita-types/src/sis/four_square.rs`.
Pure helper computing `ell_0..ell_3` with `sum ell_j^2 = B_l2 - Z_SQUARED`. A
Rabin–Shallit-style prime hunt is the fast path; a theorem-backed finite
two-squares-residual fallback makes the solver total. Integer-only decision path
(no floating point).
The four-square slack is committed as `ell_hat` on every realized level (not just
ZK), so the certificate proves an equality; small-field levels need a `u128`
target path because `B_l2 - Z_SQUARED` can exceed `2^64`.

**S3 — Threshold + transcript-stable rejection sampling.** *(S1; production shell after S2)*
`crates/akita-challenges/src/config.rs`, `sampler/exact_shell.rs`, `sampler/mod.rs`.
Add `operator_norm_threshold` to `ExactShell`, reject-and-resample with stable
XOF consumption (no prover/verifier divergence) calling the S1 predicate, and
bind shell parameters + threshold into `domain_separator_bytes`.
Tests and non-production presets may use `(31, 11), T = 16` before S2 lands.
**Do not** change `proof_optimized` D=64 production presets until S2 certifies the
accepted-support lower bound.

**S5a — Euclidean SIS table regen (lattice-estimator).** *(done in #155)*
[`specs/sis-euclidean-estimator.md`](sis-euclidean-estimator.md).
Vendor the open lattice-estimator reliability PR branch as `third_party/lattice-estimator`,
harden `scripts/gen_sis_table.py`, and check in Akita-local golden CSV under
`scripts/sis_golden/`. Repoint to `malb/lattice-estimator` after upstream merge. No Rust
estimator crate.

**S5b — L2 SIS tables + key rename.** *(S4, S5a; in #155)*
`crates/akita-types/src/sis/{ajtai_key,generated_sis_table}.rs`.
Regenerate L2 bucket ladders (`2^MIN_LOG_BUCKET .. 2^MAX_LOG_BUCKET`) + secure-rank floors;
rename `collision_inf` to `collision_l2_sq` (`u64`, power-of-two ladder) across
`AjtaiKeyParams`, `min_secure_rank`, `ceil_supported_collision`, and descriptor bytes; wire
A-role Lemma-7 conversion (`8·ω·beta_inf·ν` via `l2_sq_from_linf`) and B/D
`l2_sq_from_linf` pricing from S4.
Remove the old committed-fold L∞ rank-pricing paths. Regen remains Sage +
`scripts/gen_sis_table.py` against the pinned submodule.

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

**S8 — Prover certificate assembly.** *(S4, S6, S7)*
`crates/akita-prover/src/protocol/ring_relation.rs`, `ring_switch/coeffs.rs`.
From `DecomposeFoldWitness.centered_coeffs`, compute `Z_SQUARED`, the four-square
slack `ell` (and `ell_hat`), select the realization and the largest group size
`g` satisfying the per-exponent no-wrap gate (evaluated with the realizable carry
budget `H'_e`), derive the carries `h` and decompose each into `delta_carry(e)`
balanced cells with base-2 recomposition weights (`carry_hat`), and fall back to
deterministic `L2_BOUND_SQUARED` when no `g` passes. Append `ell_hat` /
`carry_hat` to `w_next` and transcript-bind them and `B_l2` before squeezing
`alpha` (wire-before-squeeze).

**S11 — Planner + generated tables + drift guards.** *(S4, S5, S6)*
`crates/akita-planner`, `crates/akita-config`.
Update the runtime DP, regenerate shipped schedules, update proof-size formulas,
run the generated-table drift guards, and retarget profile modes if the secure
family set changes.

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
squared-sum sumcheck and carry claim, check `h_0 = h_E = 0`, validate `B_l2`,
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
2. Should the certified bucket `B_l2` be a fixed worst-case-per-level value, or
   may the prover abort against a tighter `B_l2` with a separately proved
   acceptance probability?
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
