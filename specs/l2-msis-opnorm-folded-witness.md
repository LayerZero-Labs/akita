# Spec: L2 MSIS Cutover, Operator-Norm Challenges, and Folded-Witness L2 Certificates

| Field       | Value |
|-------------|-------|
| Author(s)   | Quang Dao, Cursor agent draft |
| Created     | 2026-06-04 |
| Status      | proposed, draft for iteration |
| PR          | |

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
Third, the SIS planner, generated tables, paper definition, transcript
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
  certificate and any four-square slack witness.
- `akita-prover`: computation of realized folded-witness square sums from
  `DecomposeFoldWitness.centered_coeffs`, construction of the slack
  certificate, and integration into the fused stage-2 proof flow.
- `akita-verifier`: replay of the L2 certificate, no-panic validation of all
  certificate shapes, and consistency with the committed next witness.
- `akita-config` / `akita-planner`: schedule search, shipped-table selection,
  generated table representation, and proof-size accounting under the L2 MSIS
  model.
- `lattice-jolt` paper text: Module-SIS definition, weak-binding theorem,
  challenge distribution, and folded-witness norm-check sections.

### Invariants

- **Single security norm.** All committed-fold A-role binding decisions use the
  L2 MSIS bound.
  No committed-fold rank, schedule, or proof-size path uses the old coefficient
  `L∞` collision bucket after the cutover.
- **Exact integer certificate.** The verifier accepts the folded-witness L2
  check only when the field equality is known to be an exact integer equality.
  The implementation must prove, by validated bounds, that no relevant square
  sum or slack term wraps modulo the base or extension field.
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
- **ZK parity.** If ZK builds are active, the L2 certificate and stage-2
  batching must have masked-proof relations analogous to the transparent path.
  The mask accounting must remain linear except for the explicitly recorded
  quadratic relations.

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

- [ ] The paper's Module-SIS definition and Akita weak-binding theorem use the
      same Euclidean norm that the Rust planner and verifier enforce.
- [ ] `akita_types::sis` exposes one committed-fold A-role L2 collision or
      witness-bound API, and the old committed-fold `collision_inf` API is
      removed from all production call sites.
- [ ] D=64 operator-norm accepted exact-shell challenges are implemented with
      transcript-deterministic rejection sampling and pinned domain bytes.
- [ ] The D=64 accepted family has a rigorous support lower bound of at least
      128 bits, not just a Monte Carlo estimate.
- [ ] The prover computes `sum centered_coeff^2` from the actual
      `DecomposeFoldWitness.centered_coeffs` used for ring-switch witness
      construction.
- [ ] The proof carries a four-square or equivalent certificate for
      `L2_BOUND_SQUARED - sum centered_coeff^2`.
- [ ] The verifier checks the L2 equality, all certificate digit bounds, and all
      no-overflow preconditions before accepting the proof.
- [ ] The L2 certificate is batched into the stage-2 proof flow, or an explicitly
      justified adjacent sumcheck, without duplicating witness scans more than
      necessary.
- [ ] Proof shape, proof-size formula, shape deserialization, and compressed
      proof validation account for the new certificate payload.
- [ ] Runtime planner fallback, generated table expansion, and shipped generated
      tables all size ranks and folded-witness digits under the L2 model.
- [ ] `cargo fmt -q`, `cargo clippy --all --message-format=short -q -- -D warnings`,
      and `cargo test` pass on the cutover branch.
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
- Four-square certificate tests covering zero slack, maximal allowed slack,
  malformed digit encodings, and field-wrap rejection.

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
- Proof size may grow locally from the L2 certificate payload, especially if the
  four-square slack witness is decomposed with many digits.
- Net proof size should improve only if the rank and recursive schedule savings
  exceed the certificate overhead.

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
- Deterministic triangle reference:
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

**Deterministic triangle bound**

```text
||z||_2 <= Gamma * sum_i ||s_i||_2
```

With `mu2_implied` backsolved from each sample, the fitted deterministic RMS is
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

### Current Model

Today the folded witness bound is:

```text
beta_inf =
  num_claims · 2^r_vars ·
  min(||c||_inf · ||s||_1, ||c||_1 · ||s||_inf).
```

The committed A-role collision is then priced as:

```text
collision_A_inf = 8 · ||c||_1 · beta_inf · nu.
```

That model is implemented in `crates/akita-types/src/sis/norm_bound.rs`,
threaded through `LevelParams::num_digits_fold`, and checked on the prover side
by `validate_decompose_fold` against
`DecomposeFoldWitness.centered_inf_norm`.

The L2 cutover replaces this coordinate envelope with a certified Euclidean
bound on the actual folded response.

### L2 Bound Model

Let a level fold `B = num_claims · 2^r_vars` block responses.
For each accepted challenge `c_i`, let `gamma(c_i)` be the negacyclic
convolution operator norm.
If every accepted challenge satisfies `gamma(c_i) <= Gamma`, then:

```text
||sum_i c_i * s_i||_2
  <= sum_i ||c_i * s_i||_2
  <= Gamma · sum_i ||s_i||_2.
```

A deterministic per-level bound is therefore:

```text
beta_l2 = Gamma · B · s_l2_max.
```

For a vector of `W` folded ring rows, the conservative square bound is:

```text
L2_BOUND_SQUARED = W · beta_l2^2.
```

This bound is safe but may be loose.
The certificate proves the realized square sum:

```text
Z_SQUARED = sum_{row, coeff} z[row][coeff]^2.
```

and accepts if:

```text
Z_SQUARED <= L2_BOUND_SQUARED.
```

For exact-shell D=64 `(31, 11)`, each challenge also has fixed coefficient
energy:

```text
||c||_2^2 = 31 + 4 · 11 = 75.
```

For one-hot or monomial witness blocks, multiplication by the selected monomial
turns the row into a signed shift of `c`, so the sharper per-product fact is:

```text
||c * s||_2^2 = 75.
```

The planner may use this sharper one-hot bound only when the protocol actually
certifies the corresponding one-hot witness contract at that level.
Dense balanced digit levels should start with the operator-norm bound above.

### Four-Square Slack Certificate

Sumcheck proves equalities, while the target is a bound.
Use Lagrange's four-square theorem to convert the inequality into an equality:

```text
Z_SQUARED + a0^2 + a1^2 + a2^2 + a3^2 = L2_BOUND_SQUARED.
```

The proof carries decomposed integer witnesses for `a0, a1, a2, a3`.
The verifier checks:

1. each `a_j` is represented by valid bounded digits,
2. each square is computed over a range that cannot wrap the field,
3. the equality above holds in the field,
4. the global no-wrap precondition proves the field equality is the same as the
   integer equality.

Open design choice:

- **Pre-embed certificate.** Prove the square sum directly over
  `DecomposeFoldWitness.centered_coeffs` before those coefficients are emitted
  into the recursive `w` table.
  This is the cleanest security statement.
- **Stage-2 selected-entry certificate.** Reuse the existing `w_evals_compact`
  table and add a third fused stage-2 term over the `z` segment.
  This may share scans but makes the exact relation to `centered_coeffs` more
  subtle because `w` contains decomposed digit planes plus other segments.

The default implementation direction is pre-embed semantics with stage-2
batching only if the `z` segment identity is explicit and audited.
Correctness of the certified object is more important than saving one witness
scan.

### Stage-2 Integration

Stage 2 currently fuses:

```text
batching_coeff · s_claim + relation_claim
```

where `s_claim` carries the stage-1 digit range check and `relation_claim`
checks the ring-switch relation on the same witness table.

The L2 certificate should become a sibling claim:

```text
rho_range · s_claim
  + rho_relation · relation_claim
  + rho_l2 · l2_claim.
```

Implementation may keep the existing single `CHALLENGE_SUMCHECK_BATCH` scalar
for the old two-term fold only if the transcript derives an unambiguous vector
of batching coefficients for all active claims.
The descriptor and proof shape must bind the number and order of stage-2
claims.

If the L2 proof is not literally over the same MLE oracle as current stage 2,
the protocol should still place it in the same stage-2 proof phase and derive
its batching challenge from the same transcript point.
Do not hide a separate proof system behind stage-2 naming.

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

The implementation must not rely on floating-point FFTs to enforce acceptance.
Acceptable production paths include:

- exact integer or rational-interval verification of `gamma(c)^2 <= 256`,
- a checked certificate for the accepted support subset,
- or a deterministic integer transform specialized to D=64 whose error bounds
  are proven and tested.

The sampler must consume transcript randomness in a stable way.
Rejected candidates are not allowed to create prover/verifier divergence.
The challenge domain separator must include the exact shell parameters and
operator-norm threshold.

### SIS Tables And Planner

The current SIS tables are keyed by a rounded coefficient `L∞` collision bucket.
The cutover needs generated L2 MSIS tables keyed by the Euclidean bound used in
the paper theorem.

Required changes:

- Rename table fields away from `collision_inf`.
- Generate audited L2 bucket ladders and secure-rank floors.
- Update `AjtaiKeyParams` to carry the L2 bound name and descriptor bytes.
- Update `LevelParams`, schedule entries, generated-family tables, and runtime
  DP fallback to use L2 bounds.
- Delete committed-fold L∞ rank derivation after all call sites move.
- Keep B-role and D-role opening-witness bounds explicit.
  If those roles remain coefficient bounded, the paper and code must state how
  their Euclidean MSIS contribution is derived.

### Paper Cutover

The paper must change together with the code.
At minimum:

- Replace the current `MSIS_{q,d}(n,m,eta)` definition using
  `||z||_inf <= eta` with the Euclidean norm used by the implementation.
- State the accepted challenge distribution and its min-entropy after rejection.
- Define `gamma(c)` as the negacyclic convolution operator norm.
- Prove the folded response L2 bound from `gamma(c)` and the committed witness
  norm.
- State the four-square slack certificate and why the finite-field equality is
  exact over the integers.
- Re-derive the weak-binding theorem with L2 parameters and no hidden
  coefficient-`L∞` bucket.
- Explain the full cutover and remove superseded L∞ schedule/pricing language.

## Architecture

### Data Flow

```text
sample accepted c_i with gamma(c_i) <= Gamma
        |
        v
decompose_fold: centered_coeffs for z = sum_i c_i * s_i
        |
        +--> compute Z_SQUARED = sum centered_coeff^2
        |        |
        |        +--> four-square slack witness
        |
        +--> z_folded_rings, w_hat, relation rows
                 |
                 v
ring_switch_build_w -> commit next w -> stage 1 -> stage 2
                                      \-> L2 certificate phase / term
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
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage2/`
- `crates/akita-verifier/src/stages/stage2.rs`
- `crates/akita-verifier/src/protocol/levels.rs`
- `scripts/gen_sis_table.py`
- `specs/weak-binding-norm-fix.md`
- paper files under `Documents/Research/lattice-jolt/sections/akita/`

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
would invite drift between paper, planner, prover, and verifier.
The cutover should replace the old path.

## Execution

### Phase 0: Proof And Parameter Decisions

- Finalize the L2 MSIS definition and estimator input convention.
- Prove or certify the D=64 exact-shell operator-norm acceptance lower bound.
- Decide the production D=64 shell and threshold.
  Starting candidate: `(31, 11)`, `T = 16`.
- Decide the L2 bound policy for dense, one-hot, tensor, and terminal levels.
- Decide whether the first implementation certifies over pre-embed
  `centered_coeffs` or over an explicit `z` segment in `w`.

### Phase 1: Challenge Family

- Add the accepted operator-norm challenge variant.
- Implement transcript-stable rejection sampling.
- Add exact or certified acceptance-support validation.
- Bind shell parameters and threshold in domain separator bytes.
- Update proof-optimized D=64 policy after the support proof is in place.

### Phase 2: L2 SIS Primitives

- Replace committed-fold L∞ bound APIs with L2 bound APIs.
- Generate L2 SIS tables.
- Update `AjtaiKeyParams`, descriptors, and schedule layout derivation.
- Remove old committed-fold L∞ paths.

### Phase 3: L2 Certificate

- Add square-sum computation to `DecomposeFoldWitness`.
- Add four-square slack witness generation.
- Add proof object fields and shape validation.
- Add verifier checks and no-wrap validation.
- Integrate with stage-2 batching or a clearly adjacent stage-2 certificate
  proof.

### Phase 4: Planner And Generated Tables

- Update runtime DP.
- Regenerate shipped schedules.
- Update proof-size formulas.
- Run generated-table drift guards.
- Retarget profile modes if the secure family set changes.

### Phase 5: Paper And Docs

- Update the Akita paper sections in lockstep with code.
- Update specs that mention committed-fold L∞ pricing.
- Add a short profile-guide note explaining the new challenge rejection cost and
  certificate overhead.

## Open Questions

1. What exact Euclidean MSIS estimator and table-generation command should be
   canonical for the repo?
2. Should the certified bound be worst-case deterministic per level, or can the
   prover abort against a tighter threshold with a separately proved acceptance
   probability?
3. Is the four-square certificate best represented as four full integers per
   level, per segment, or as decomposed MLE columns?
4. Can the pre-embed `centered_coeffs` certificate be fused into the existing
   stage-2 sumcheck without making the certified object ambiguous?
5. For tensor challenges, should `Gamma` be derived from factor operator norms,
   the expanded product challenge, or a separate accepted tensor-product policy?
6. How much of the old B/D role coefficient-bound story must change to keep the
   paper's MSIS norm uniform?

## References

- `specs/weak-binding-norm-fix.md`
- `specs/bounded-l1-sparse-challenge.md`
- `specs/tensor-structured-folding-challenges.md`
- `crates/akita-types/src/sis/norm_bound.rs`
- `crates/akita-prover/src/protocol/ring_relation.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage2/mod.rs`
- `D64-EXACT-SHELL-OPNORM-RESEARCH-PROMPT-NEVER-COMMIT.md`
- `A-ROLE-PRICING-HANDOFF-NEVER-COMMIT.md`
- `Documents/Research/paper-note/notes/akita-labrador-greyhound-proofsize-leopard-2026-06-03.md`
- `Documents/Research/lattice-jolt/sections/akita/2_preliminaries.tex`
