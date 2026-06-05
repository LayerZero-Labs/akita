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
  certificate and any four-square slack witness.
- `akita-prover`: computation of realized folded-witness square sums from
  `DecomposeFoldWitness.centered_coeffs`, construction of the slack
  certificate, and integration into the fused stage-2 proof flow.
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
- **Exact integer certificate, gated on field capacity.** The verifier accepts
  the realized folded-witness L2 certificate only when the field equality is
  known to be an exact integer equality.
  The implementation must prove, by validated bounds, that the structural
  worst-case square sum admitted by the digit range check (about
  `coeffs · beta_linf^2`) and the slack squares fit the working field (base or
  chosen extension) without wrapping, not merely that the realized value fits.
  When the bound cannot fit the field (notably fp32 at the dense recursive
  levels, where the realized square sum already exceeds `q`), the level emits no
  realized certificate and prices the A-role at the deterministic worst-case L2
  bound instead.
  The deterministic bound is still Euclidean and still far tighter than the old
  `L∞` envelope; the certificate is a field-capacity-gated tightening, not a
  separate security model.
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

- [ ] The public repo specs and security docs use the same Euclidean Module-SIS
      norm, operator-norm challenge distribution, and folded-witness bound that
      the Rust planner and verifier enforce.
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
      `B_l2 - sum centered_coeff^2`, with each slack witness bounded by `< 2^32`
      and the four slack witnesses together occupying at most one ring element.
- [ ] The verifier checks the L2 equality, all certificate digit bounds, and all
      no-overflow preconditions before accepting the proof.
- [ ] The realized certificate is emitted only when the structural no-wrap gate
      `coeffs·balanced_digit_max(lb, num_digits_fold)^2 + 4·B_l2 < q_eff` holds;
      fp64 and fp128 emit it, fp32 dense recursive levels fall back to the
      deterministic `L2_BOUND_SQUARED`, and a test pins which levels take which
      tier.
- [ ] The certified squared-sum statement is over the recomposed `z` and `ell_j`,
      and a test ties it to the committed decompositions `z_hat` and `ell_hat_j`
      via gadget recomposition (a tampered `z_hat` or `ell_hat_j` fails the check).
- [ ] B-role and D-role collisions are converted into the unified L2 table by
      `||v||_2 <= sqrt(d)·||v||_inf`, and a test pins the conversion against the
      generated tables.
- [ ] Sumcheck 1 (`sum_x z_aug(x)^2 = B_l2`) runs in the stage-1 phase and
      sumcheck 2 (`z_aug = G' · w_next`) in the stage-2 phase, with sumcheck 2's
      `w_next(rho')` output joining the recursive-witness opening (batched or
      explicitly justified adjacent), without duplicating witness scans more than
      necessary.
- [ ] On certifying levels the stage-1 message drops the eq-factored linear-term
      omission (about one extra field element per round); non-certifying levels
      keep the eq-factored message. A test pins the per-level stage-1 message
      shape, and the descriptor binds it.
- [ ] `ell_hat` is committed and transcript-bound before the sumcheck-1 challenges
      are squeezed (wire-before-squeeze), and a logging-transcript test enforces
      the ordering.
- [ ] Proof shape, proof-size formula, shape deserialization, and compressed
      proof validation account for the new certificate payload (canonical byte
      layout of `ell_hat_j` pinned by a serialization test).
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
- The L2 certificate payload is at most one ring element per certified level
  (four slack witnesses, each `< 2^32`, balanced-digit decomposed), so the local
  proof-size growth is small and bounded.
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

Here `s_l2_max` is the per-block committed-witness Euclidean bound, the L2
analogue of the existing `FoldWitnessNorms` (`||s||_inf`, `||s||_1`):

```text
s_l2_max = sqrt(D) · (b/2)   dense balanced digits (||s||_inf = b/2),
s_l2_max = 1                 a one-hot block (a single unit coefficient).
```

It should be derived once and shared by the planner and the prover abort, the
way `fold_witness_beta` and the prover's `beta_linf_fold_bound` share
`ring_product_infinity_norm_bound` today.

For a vector of `W` folded ring rows, the conservative square bound is:

```text
L2_BOUND_SQUARED = W · beta_l2^2.
```

This deterministic bound is safe but loose (the calibration below puts the
triangle bound roughly 13x to 100x above the realized square sum).
It is the bound used directly when no realized certificate is emitted (the
field-capacity fallback in the Invariants).

When a certificate is emitted, the prover instead commits to a tighter chosen
bucket `B_l2` from the L2 MSIS ladder with

```text
Z_SQUARED <= B_l2 <= L2_BOUND_SQUARED,
Z_SQUARED = sum_{row, coeff} z[row][coeff]^2,
```

and proves `Z_SQUARED <= B_l2`, so the A-role is sized against `B_l2` rather than
the loose deterministic `L2_BOUND_SQUARED`.
The two tiers are the same security model: deterministic `L2_BOUND_SQUARED` is
always sound from the operator-norm contract plus the digit-range check, and
`B_l2` is the certificate-tightened version used only where the field can hold
it.

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
Z_SQUARED + ell_0^2 + ell_1^2 + ell_2^2 + ell_3^2 = B_l2.
```

The proof carries decomposed integer witnesses for `ell_0, ell_1, ell_2, ell_3`.
The verifier checks:

1. each `ell_j` is represented by valid bounded digits,
2. each square is computed over a range that cannot wrap the field,
3. the equality above holds in the field,
4. the global no-wrap precondition proves the field equality is the same as the
   integer equality.

**Slack witness size.**
Each `ell_j` is bounded by `sqrt(B_l2)`.
With the realized square sums logged in the calibration (`Z_SQUARED` around
`2^32` even at the densest levels), `ell_j` is about 16 bits in practice; the
implementation pins a generous ceiling of `ell_j < 2^32`.
Decomposed into balanced base-`2^lb` digits, each `ell_j` needs
`delta_ell = num_digits_for_bound(32, field_bits, lb)` digits, and the four are
packed contiguously (slack-major) into the `ell_hat` segment, `4·delta_ell`
field coefficients total. That is at most one ring element: `4·delta_ell <= D`
for `D = 64` at any `lb`, and for `D = 32` once `lb >= 4` (at most two
otherwise). `ell_hat` is appended to `w_next` and the next-level hypercube is
rounded up to a power of two on the *augmented* witness, so the certificate's
committed payload is at most one extra ring element per certified level and is
partly (often fully) absorbed by the existing power-of-two round-up.

**Field-capacity gate.**
Soundness needs the *structural* worst-case square sum to fit the field, not just
the realized value: the sumcheck only proves `<z, z> ≡ Z_SQUARED (mod q)`, so the
verifier must rule out wrap using only what it can prove about `z`.
The verifier never sees `z`; it sees that each committed digit plane of `z_hat`
passes the stage-1 range check, so the strongest per-coefficient bound it can
assert is `|z[i]| <= balanced_digit_max(lb, num_digits_fold)`.
That bound is `≈ beta_linf` but rounded up to the next whole digit (the digit
count is sized to cover `[-beta_linf, beta_linf]` plus one bit; see
`decomposition_digits.rs::num_digits_fold`), so it can exceed `beta_linf` by up to
a factor `2^lb`. A level emits the realized certificate only when

```text
coeffs · balanced_digit_max(lb, num_digits_fold)^2  +  4 · B_l2  <  q_eff,
```

over the `coeffs = W · D` certified coefficients, where `q_eff` is the modulus of
the field the square sum is accumulated in, the first term dominates, and
`4 · B_l2` covers the four range-checked slack squares (each `ell_j^2 <= B_l2`).
Below we write the first term as `coeffs · beta_linf^2` for the crossover
estimates; the true gate is larger by the digit-rounding factor above.

Production moduli are `q_fp32 = 2^32 − 99 ≈ 2^32`, `q_fp64 = 2^64 − 59 ≈ 2^64`,
`q_fp128 ≈ 2^128`.
Dropping the sub-dominant slack term, a level fits iff

```text
beta_linf  <  sqrt(q_eff) / sqrt(coeffs)   (coeffs = W · D).
```

Calibration crossovers (D=64):

- fp32 (`sqrt(q) ≈ 2^16`): a one-hot root at `nv = 16` (`coeffs ≈ 2^14`,
  `beta_linf = 216`) gives `coeffs·beta_linf^2 ≈ 2^29.5 < 2^32`, so it *fits*;
  the `nv = 20` one-hot root (`beta_linf = 864`) is `≈ 2^35.5` and a dense level
  (`beta_linf = 6912`) is `≈ 2^41.3`, both well over `2^32`. So fp32 emits the
  certificate only at small one-hot roots and falls back everywhere else.
- fp64 (`sqrt(q) ≈ 2^32`): every profiled D=64 level is `≤ 2^43`, leaving about
  20 bits of headroom. fp64 overflows only when `coeffs·beta_linf^2 ≥ 2^64`, i.e.
  `beta_linf ≳ 2^32 / sqrt(coeffs)` (about `4·10^6` at `coeffs ≈ 10^6`). That
  happens only at deep recursive levels with a large gadget basis and/or large
  fold arity, so fp64 effectively always certifies at the sizes we run; the
  per-level gate guards the rare large-level case.
- fp128: never binds in practice.

A level skips the certificate and prices at the deterministic `L2_BOUND_SQUARED`
(see Invariants) exactly when the gate fails; the decision is per (field, level)
on the computed structural sum, not per field globally.

**Is the gate conservative?**
Yes relative to *realized* norms, and no relative to *what the verifier can
prove*. The realized `sum z[i]^2` is far below the gate: the calibration logs a
realized `Z_SQUARED ≈ 2^32` even at the densest levels, while the structural gate
for that same dense D=64 level is `≈ 2^41` (and larger once the digit-rounding
factor is included). `beta_linf` is a worst-case `L∞` fold bound and honest
coefficients sit well under it. But the verifier cannot use the realized value,
because the whole point of the certificate is that it does not trust the prover's
`z` to be small. An adversarial, range-check-passing prover may
drive `<z, z>` up to the structural bound, so that bound is exactly the no-wrap
floor, not an engineering margin. The gate is therefore not expected to match
experimental data; it bounds the worst case while `B_l2` prices the realized
(tight) value. The gate and the bucket are independent: passing the gate only
decides whether the equality is exact, and `B_l2` decides how tight the A-role
pricing is.

This also means there is a real tightening lever, separate from changing fields:

- Extension fields do **not** help. `F_{q^k}` has characteristic `q`, so a sum of
  base-field-embedded squares still reduces mod `q` coordinate-wise and wraps at
  the same point. A genuinely larger base prime raises the no-wrap ceiling; see
  follow-up below for staying on 31/32-bit fields.
- A tighter verifier-enforced per-coefficient bound shrinks the gate directly.
  Constraining the top fold digit so the asserted bound is `beta_linf` rather than
  `balanced_digit_max ≈ 2^lb · beta_linf` recovers up to a factor `2^{2·lb}` in
  the gate (`lb` is typically 5–11, i.e. 10–22 bits), which would let more fp32
  levels certify instead of falling back. This is an optional optimization, not
  required for soundness.

Absent those, fp32 falls back at all but the smallest one-hot roots, and
fp64/fp128 effectively always certify.

**Follow-up (deferred): realized certificates on 31/32-bit fields.**

The initial cutover does not implement this. When the structural gate above
fails, levels on `q ≈ 2^31` or `2^32` price the A-role at the deterministic
`L2_BOUND_SQUARED` instead of a tighter realized `B_l2`. A later slice may allow
the same base field to support the realized certificate on dense recursive levels
by accumulating the squared norm without forming a single wrapped field sum over
recomposed coordinates (for example, wide accumulation from the committed digit
planes already bound by stage 1). Whether that can be made sound and cheap enough
for production is open; it is listed here only so the fallback tier is not read
as a permanent limitation of small fields.

### Recomposed Witness vs Committed Decomposition (two linked sumchecks)

The L2 statement is about the *recomposed* integer folded witness, but the object
the protocol commits to at the next level is its *decomposition*.
Notation:

- `z`: the centered integer folded witness `z = sum_i c_i · s_i`, exactly
  `DecomposeFoldWitness.centered_coeffs`. The object whose Euclidean norm we
  certify.
- `z_hat`: the balanced base-`2^lb` digit planes of `z`, emitted by
  `emit_z_folded_block_inner` (`ring_switch/coeffs.rs`) into the committed
  recursive witness, with `z[coeff] = sum_d 2^{lb·d} · z_hat^{(d)}[coeff]`.
- `ell_0..ell_3`: the four Lagrange slack integers; `ell_hat_j` their balanced-digit
  planes, with `ell_j = sum_d 2^{lb·d} · ell_hat_j^{(d)}`.
- `z_aug`: the augmented vector `z || ell` (the four slack entries appended last,
  padded to a power of two), so `<z_aug, z_aug> = sum_i z[i]^2 + sum_j ell_j^2`.
- `w_next`: the full committed recursive witness, in the existing adaptive
  segment order (`build_w_coeffs`, keyed by `ring_column_z_first(lp)`):
  - `m_vars >= r_vars`: `z_hat | w_hat | t_hat | (zk) | r_hat | ell_hat`,
  - else: `w_hat | t_hat | (zk) | z_hat | r_hat | ell_hat`,
  where `ell_hat = ell_hat_0 || … || ell_hat_3` is the new certificate segment appended
  after `r_hat`, packed slack-major (`4·delta_ell` coefficients), with the
  next-level power-of-two hypercube round-up computed on the augmented `w_next`.

The certificate is two sumchecks, the second discharging the first's output into
the witness opening the protocol already sends.

**Sumcheck 1 (norm certification over `z_aug`).**
Prove the sum-of-squares equality

```text
sum_x z_aug(x)^2 = B_l2
```

with `z_aug` the virtual multilinear extension over `n_aug = ceil(log2(coeffs + 4))`
variables; the four-square identity makes it exact.
Each round is degree 2, and the sumcheck reduces to one evaluation claim

```text
z_aug(rho) = v.
```

(The verifier first checks the structural no-wrap gate so this is an integer
identity, not just a mod-`q` identity.)

**Sumcheck 2 (virtualization onto the committed witness).**
`z_aug` is not committed; only `w_next` is.
They are related by a fixed, public, structured linear map

```text
z_aug = G' · w_next,
```

where `G'`:

- on the `z_hat` block, applies the per-coefficient gadget recomposition
  `z[coeff] = sum_d 2^{lb·d} z_hat^{(d)}[coeff]`, using the same `fold_gadget =
  gadget_row_scalars(depth_fold, log_basis)` weights and the same
  `(dc, df, point, block)` index map the relation check already uses for the
  `A · ẑ` term in `compute_setup_contribution`;
- on the `ell_hat` block, applies the scalar gadget `ell_j = sum_d 2^{lb·d} ell_hat_j^{(d)}`;
- is zero on `w_hat`, `t_hat`, `r_hat`, the zk blinding segments, and padding.

Writing `M(rho, y) = sum_x eq(rho, x) G'(x, y)`,

```text
z_aug(rho) = (G' · w_next)(rho) = sum_y M(rho, y) · w_next(y),
```

so sumcheck 2 is a product sumcheck over `y` that reduces to a single

```text
w_next(rho') = v',
```

which is exactly the evaluation claim the protocol already sends for the next
recursive witness; it merges into that opening with no new commitment.
The verifier evaluates `M(rho, rho')` succinctly because `G'` is the same
structured family as the setup/relation matrix: a tensor of `eq` factors over the
block/point/digit indices times the geometric-series gadget weights
`sum_d 2^{lb·d}(…)`, computed by the same machinery as `compute_setup_contribution`
(`fold_gadget`, `eq_eval_at_index`, segment offsets `offset_z`, plus a new
`offset_ell` and the scalar gadget for `ell_hat`).
This is the existing relation-matrix sumcheck shape (`relation_claim =
sum_i eq(tau1, i) y_alpha[i]` reducing to `w(r) · structured(r)`), with sumcheck 1
producing the `z_aug(rho)` claim that sumcheck 2 discharges.

**Why this satisfies stage-2 consistency.**
The certified `z_aug` reduces to `w_next(rho')`, the *same* committed witness that
carries `z_hat` (and now `ell_hat_j`).
A prover cannot certify one `z` and commit a different `w`, because both flow
through the single `w_next(rho')` opening.

The remaining choice is placement only: whether sumcheck 2's `w_next(rho')` claim
is batched into the existing stage-2 point (one combined opening) or kept as an
adjacent claim at the same transcript point.
The certified relation is identical either way.

### Stage Placement And Batching

The two certificate sumchecks map onto the existing two stages by their algebraic
shape, not by convenience:

- **Stage 1** runs the norm sumcheck `0 = sum_z eq(tau0, z) · Q(w(z)(w(z)+1))`
  (`akita_stage1`) over the eq-factored (`GruenSplitEq`) path. Sumcheck 1
  (`sum_x z_aug(x)^2 = B_l2`) is the same kind of nonlinear check on committed
  digit values, so it batches here.
- **Stage 2** fuses the carried `s_claim` with `relation_claim`, the ring-switch
  relation (a matrix-row check reducing to a `w` opening). Sumcheck 2
  (`z_aug = G' · w_next`) is exactly that shape, so it batches here, and its
  `w_next(rho')` output joins the recursive-witness opening.

**Eq-factored cost in stage 1.**
The stage-1 eq-factored path sends each round's `q` with its linear coefficient
omitted, recovered from the single global `eq(tau0,·)` factor.
The `z_aug^2` summand has no `eq` factor, so batching it forecloses that omission:
certifying rounds revert to the plain batched-sumcheck message, about one extra
field element per round.
This is paid only on levels that emit a certificate; fallback or no-cert levels
keep the eq-factored stage-1 message unchanged.
The descriptor and proof shape must gate the stage-1 message format (and the
per-stage batching vectors) on whether the level certifies.

**Batching.**
Each stage derives an unambiguous batching vector from its transcript point:
stage 1 over `{norm_claim, l2_claim}`, stage 2 over
`{s_claim, relation_claim, virtualization_claim}`.
The existing single `CHALLENGE_SUMCHECK_BATCH` scalar is kept only if the
transcript derives a full coefficient vector for all active claims at that stage.
The descriptor must bind the number and order of claims at each stage, and must
not hide a separate proof system behind stage naming.

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
        +--> Z_SQUARED = sum z^2 ; four-square slack ell_0..ell_3 ; z_aug = z || ell
        |        |
        |        +--> sumcheck 1: sum_x z_aug(x)^2 = B_l2  -->  z_aug(rho) = v
        |                 |
        |                 v
        |        sumcheck 2: z_aug(rho) = (G' * w_next)(rho)  -->  w_next(rho') = v'
        |
        +--> z_folded_rings (z_hat), w_hat, t_hat, r_hat, ell_hat = ell_hat_j
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
- `crates/akita-prover/src/protocol/ring_switch/coeffs.rs` (`build_w_coeffs`:
  append the `ell_hat = ell_hat_j` segment; `emit_z_folded_block_inner` is the `z_hat`
  recomposition source)
- `crates/akita-types/src/proof/ring_relation.rs` (`ring_column_z_first`, segment
  layout: add `offset_ell`)
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

## Execution

The work decomposes into 13 slices across six tracks (challenge family, L2 SIS,
proof shape, prover, verifier, planner/transcript/tests).
Four slices are independent and can start immediately; the rest serialize behind
the L2 norm/table API (S4, S5) and the proof-shape change (S6).

Status: S1 (`crates/akita-challenges/src/sampler/op_norm.rs`), S7
(`crates/akita-types/src/sis/four_square.rs`), and the S4 L2 norm primitives
(`crates/akita-types/src/sis/norm_bound.rs`, squared-domain) are implemented as
pure, not-yet-wired building blocks on `main`.

Follow-up implementation is split across branch `quang/s3-s5-sis-estimator-spec`
(spec-first) and later PRs:

- **S5a** ([`sis-euclidean-estimator.md`](sis-euclidean-estimator.md)): Rust offline
  estimator + `gen_sis_table` (spec in flight; implementation after spec approval).
- **S5b**: L2 table regen, `collision_l2_sq` rename, wire A-role pricing (blocked on S5a).
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
- L2 bound policy for dense, one-hot, tensor, and terminal levels (the
  `s_l2_max` source per level). (S4)
- Certificate placement: whether sumcheck 2's `w_next(rho')` claim is batched
  into the stage-2 point or kept adjacent at the same point (the certified
  relation is identical either way). (S9)
- The structural field-capacity gate
  `coeffs · balanced_digit_max(lb, num_digits_fold)^2 + 4·B_l2 < q_eff` and the
  per-(field, level) fallback to the deterministic bound (binds often on fp32,
  rarely on fp64, never on fp128). (S8, S10)
- The canonical Euclidean MSIS estimator and table-generation command: see
  [`sis-euclidean-estimator.md`](sis-euclidean-estimator.md) (S5a). (S5b consumes its output.)

### Slice Dependency Graph

```text
WAVE 0  (independent, start now, parallel)
  S1  op-norm predicate gamma_D(c) <= T     [akita-challenges, pure]   DONE
  S7  four-square decomposition helper      [pure algorithm]           DONE
  S4  L2 norm primitives (s_l2_max, ...)    [akita-types::sis, pure]   DONE
  S2  D=64 support lower bound >= 128 bits   [research / certificate]

WAVE 1
  S5a Rust Euclidean SIS estimator + gen      (spec: sis-euclidean-estimator.md)
  S5b L2 SIS tables + collision_l2_sq rename  (S4, S5a)
  S3  threshold + transcript rejection       (S1, S2 for production policy)
  S6  proof shape / serialization / size     (parameterize B_l2 early)

WAVE 2
  S8  prover certificate assembly            (S4, S6, S7)
  S11 planner + shipped tables + drift       (S4, S5, S6)
  S12 transcript instance-descriptor bind    (S3, S5, S6)

WAVE 3
  S9  two sumchecks (z_aug^2 = B_l2 ; G'·w)  (S6, S8)
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
Adds the squared-domain `s_l2_max_squared` (`D·(b/2)^2` dense, `1` one-hot),
`beta_l2_squared = (Gamma·B·s_l2_max)^2`, `l2_bound_squared = W·beta_l2^2`, and
the B/D `l2_sq_from_linf` (`||v||_2^2 <= d·||v||_inf^2`) conversion. Squared
domain keeps every value an exact `u128` integer (`sqrt(D)` is irrational for
`D ∈ {32, 128}`); the real square root is taken only at bucket/slack selection
(S8). These are pure and not yet wired into rank pricing (first consumers: S5,
S8, S11).
Retain `fold_witness_beta` for digit sizing (`num_digits_fold`); it no longer
prices the A-role rank.

**S7 — Four-square decomposition helper.** *(independent, DONE)*
`crates/akita-types/src/sis/four_square.rs`.
Pure helper computing `ell_0..ell_3` with `sum ell_j^2 = B_l2 - Z_SQUARED`, each
`ell_j < 2^32`. A Rabin–Shallit-style prime hunt is the fast path; a
theorem-backed finite two-squares-residual fallback makes the solver total for
every `u64` target. Integer-only decision path (no floating point).
No protocol dependency; consumes only the target integer (first consumer: S8).

**S3 — Threshold + transcript-stable rejection sampling.** *(S1; production shell after S2)*
`crates/akita-challenges/src/config.rs`, `sampler/exact_shell.rs`, `sampler/mod.rs`.
Add `operator_norm_threshold` to `ExactShell`, reject-and-resample with stable
XOF consumption (no prover/verifier divergence) calling the S1 predicate, and
bind shell parameters + threshold into `domain_separator_bytes`.
Tests and non-production presets may use `(31, 11), T = 16` before S2 lands.
**Do not** change `proof_optimized` D=64 production presets until S2 certifies the
accepted-support lower bound.

**S5a — Rust Euclidean SIS estimator.** *(spec-approved before code)*
[`specs/sis-euclidean-estimator.md`](sis-euclidean-estimator.md),
future `crates/akita-sis-estimator/`.
Offline crate reproducing lattice-estimator `SIS.lattice(..., norm=2, BDGL16)` on the
`cost_euclidean` path; `gen_sis_table` binary emits regenerated rows. Golden tests pin
equivalence. This slice does not change protocol code on its own.

**S5b — L2 SIS tables + key rename.** *(S4, S5a)*
`crates/akita-types/src/sis/{ajtai_key,generated_sis_table}.rs`.
Regenerate L2 bucket ladders (`2^MIN_LOG_BUCKET .. 2^MAX_LOG_BUCKET`) + secure-rank floors;
rename `collision_inf` to `collision_l2_sq` across `AjtaiKeyParams`, `min_secure_rank`,
`ceil_supported_collision`, and descriptor bytes; wire A-role `8·Γ·ν·‖z‖₂` pricing from S4.
Remove the old committed-fold L∞ rank-pricing paths. Deprecate Sage-only regen as the
canonical path once the Rust binary is checked in.

**S6 — Proof shape, serialization, proof size.** *(parameterizable early)*
`crates/akita-types/src/proof/{levels,shapes}.rs`, `proof_size.rs`,
`proof/ring_relation.rs`.
Add the `ell_hat = ell_hat_j` segment and `offset_ell` to the segment layout
(`ring_column_z_first`), the certificate payload fields and chosen `B_l2` /
gate-tier marker, shape validation, the canonical `ell_hat_j` byte-layout test,
and the proof-size formula.
Parameterized on `B_l2`'s type, so it does not wait on S5's values.

**S8 — Prover certificate assembly.** *(S4, S6, S7)*
`crates/akita-prover/src/protocol/ring_relation.rs`, `ring_switch/coeffs.rs`.
Compute `Z_SQUARED` from `DecomposeFoldWitness.centered_coeffs`, assemble
`z_aug = z || ell`, decompose `ell_hat_j` slack-major, append `ell_hat` as the
last segment in `build_w_coeffs` (so the next-level power-of-two hypercube is
sized on the augmented witness), and apply the field-capacity gate
`coeffs · balanced_digit_max(lb, num_digits_fold)^2 + 4·B_l2 < q_eff` with the
per-(field, level) deterministic-bound fallback.
The four-square solver runs before sumcheck 1, and `ell_hat` is committed and
transcript-bound before the sumcheck-1 challenges are squeezed
(wire-before-squeeze); `z_aug` entries embed as base-field elements, exact only
under the no-wrap gate (all entries `< sqrt(q)`).

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

**S9 — Two sumchecks.** *(S6, S8)*
`crates/akita-prover/src/protocol/sumcheck/akita_stage2/`, verifier
`slice_mle/setup_contribution.rs`.
Sumcheck 1 (`sum_x z_aug(x)^2 = B_l2`) in the stage-1 phase; sumcheck 2
(`z_aug = G' · w_next`) in the stage-2 phase, reusing `fold_gadget` /
`compute_setup_contribution` for the succinct `G'` evaluation and landing
`w_next(rho')` in the existing stage-2 opening.
Certifying levels lose the eq-factored linear-term omission (about one extra
field element per round); non-certifying levels keep the current eq-factored
message.

**S10 — Verifier replay + no-panic.** *(S6, S9)*
`crates/akita-verifier/src/stages/stage2.rs`, `protocol/levels.rs`.
Replay the L2 equality, check all certificate digit bounds, verify the structural
no-wrap gate before treating the field equality as an integer equality, and
reject every malformed challenge/certificate/shape with `AkitaError` /
`SerializationError`.

**S13 — End-to-end + ZK parity.** *(all)*
End-to-end prover/verifier tests that fail under independent tampering of the
committed folded witness, the L2 certificate, the next-witness commitment, and
the ring-relation rows; ZK-path parity if the feature stays enabled.

## Open Questions

1. Resolved: [`specs/sis-euclidean-estimator.md`](sis-euclidean-estimator.md) defines the
   Rust `akita-sis-estimator` crate and `gen_sis_table` binary as the canonical offline
   regen path, with golden parity to lattice-estimator `SIS.lattice(..., norm=2, BDGL16)`.
2. Should the certified bucket `B_l2` be a fixed worst-case-per-level value, or
   may the prover abort against a tighter `B_l2` with a separately proved
   acceptance probability?
3. Resolved (magnitude and placement): the four slack witnesses `ell_0..ell_3` are
   bounded (about 16 bits realized, `< 2^32` pinned) and decompose into at most
   one ring element total, and `ell_hat_j` is committed as the `ell_hat` segment
   appended last in `build_w_coeffs`. What remains is the canonical byte layout
   test for that segment (see Acceptance Criteria).
4. Resolved (virtualization worked out in "Recomposed Witness vs Committed
   Decomposition"): two linked sumchecks. Sumcheck 1 certifies
   `sum_x z_aug(x)^2 = B_l2` over `z_aug = z || ell`, sumcheck 2 discharges
   `z_aug(rho) = (G' · w_next)(rho)` onto the committed `w_next` via the gadget
   recomposition `G'`, reusing `compute_setup_contribution`/`fold_gadget` for the
   succinct `G'` evaluation. The only residual choice is whether sumcheck 2's
   `w_next(rho')` claim is batched into the stage-2 point or kept adjacent at the
   same point (the certified relation is identical either way).
5. For tensor challenges, should `Gamma` be derived from factor operator norms,
   the expanded product challenge, or a separate accepted tensor-product policy?
6. Resolved: the B/D roles keep their coefficient `L∞` digit-collision bound
   `2^lb - 1` and convert into the unified L2 table via
   `||v||_2 <= sqrt(d)·||v||_inf` (see Invariants and SIS Tables And Planner).
7. Deferred: small-field (31/32-bit) realized L2 certificates without widening
   the base prime; see "Follow-up (deferred)" under the field-capacity gate.

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
