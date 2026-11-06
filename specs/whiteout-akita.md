# Whiteout zero knowledge on Akita

| Field  | Value |
| ------ | ----- |
| Status | implemented behind the `zk` features; no preset selects it by default |
| Scope  | Akita's half of Whiteout: one-shot fold responses, hiding commitments, the masked prefix of an opening, the residual commitment, and the response proof |

jolt-zk owns the pads, the masking rows, the Σ-protocol and the response
statement. Jolt composes the full proof. This note covers what Akita
implements and why it is sound and zero knowledge.

"The paper" is the Whiteout paper; `wo:…` names its theorem labels and
"paper §n" its sections. The headline instance is Jolt at T = 2^24 over
`F = Prime128OffsetA7F7` (q = 2^128 − 2^32 + 22537).

**Indexing.** The paper's fold j is Akita's level ℓ = j − 1; the root is
level 0. The paper's witness w_j is the output witness of level j − 1. This
note uses levels except when it quotes the paper. The cut is defined once:

> `cut_level = k − 1 = 1` (`akita_params::zk::ZK_CUT_LEVEL`).

It is validated schedule data, and k = 2 is the only supported value.

## Decision

A zero-knowledge opening runs Akita levels 0..=`cut_level` masked and
randomized and stops after Stage 2 of `cut_level`. The cut level's output
witness w_k, committed by the residual commitment u_a, is opened by
jolt-zk's Σ-protocol. The Σ-protocol's response vector e is then proved
with an ordinary clear Akita opening, the response proof.

- **One-shot fold responses (§2).** At levels ≤ `cut_level` a fold sends no
  nonce, draws its challenges at counter 0, and probes its response once. An
  out-of-range response aborts the whole proof with
  `AkitaError::FoldOverflow { level }`, and the caller reruns with fresh
  randomness. Digit ranges are sized so that the abort has probability at
  most 2^−136 in total.
- **Hiding randomness (§3).** Every B slice and D image made at a masked
  level, the root commitment, every precommitted group a masked level
  folds, and the residual commitment get fresh randomness columns. One
  helper sizes them, and the randomness digits join the witness whose
  relation already checks that image.
- **Masked levels (§4).** Each stage check is written once, generic over
  jolt-transcript's `ClaimDomain`, and runs in the clear value domain, the
  masked rows domain and the pad-layout shape domain. The verifier records
  the masked checks and lowers them into residual rows once the auxiliary
  values arrive. The cut hands off a typed `ResidualOpening` and
  `ResidualClaim`.
- **Residual commitment (§5).** u_a is Module-SIS binding at the Σ
  collision radius R_Σ ≈ 2^47.82, is committed over `R_128`, and is sent
  compressed as `cm(u_a)`.
- **Response proof (§6).** `akita_pcs::zk::AkitaResponseProver` implements
  jolt-zk's `ResponseProver`: a product sumcheck, a radix-16 digit range, a
  fused degree-3 sumcheck with a binary constraint for compression digits,
  and an Akita opening.
- **Publication (§7).** `ZkCommitmentConfig<Cfg>` selects zero knowledge;
  zero-knowledge schedules come from their own families and catalogs.

Clear configurations produce the same setup, commitment and proof bytes
with or without the `zk` features (§1.5).

Two alternatives were rejected. Retrying a fold response under a
zero-knowledge schedule makes the accepted counter depend on the witness
(§2.6). An ℓ2 route at a masked level sends the response energy in the clear
(§2.5).

## 1. Mode selection and binding

### 1.1 Features

`zk` (on every crate that has zero-knowledge code) compiles those modules
and selects nothing; selection is by configuration type (§1.2). `zk` holds
no sampler, so a verifier that enables it never compiles prover randomness:
akita-pcs's jolt-zk dependency carries `whiteout` only. `zk-prover` (implies
`zk`) adds the samplers: jolt-crypto's `sample`, `rand_core/getrandom` and
the `OsRng` hiding source on akita-cpu-backend, and jolt-zk's
`whiteout-prover` on akita-pcs. Without it a schedule that declares hiding
randomness fails with `UnsupportedSchedule`. CI fails if
`cargo tree -p akita-pcs --features zk -e features,normal` reaches
`whiteout-prover`, jolt-dit or a `sample` feature.

`test-support` on akita-pcs implies `zk-prover` and adds test hooks only:
`zk::UncheckedResponseProver` (§9) and `BackendCompression` (§5.4).
`zk-diagnostics` adds the dudect harness (§6.6).

### 1.2 Configuration

`CommitmentConfig` has an associated
`const ZK_SCHEDULE_FAMILY_NAME: Option<&'static str>` (default `None`) and
`const IS_ZK: bool`, which is true only for `ZkCommitmentConfig`.
`ZkCommitmentConfig<Cfg>` (akita-config) delegates every policy item to
`Cfg`, sets `IS_ZK = true`, and takes its family from
`Cfg::ZK_SCHEDULE_FAMILY_NAME`, resolved in a `const` that fails to compile
for a `Cfg` without one. The recursive adapter forwards
`RecursiveScheduleConfig::RECURSIVE_ZK_SCHEDULE_FAMILY_NAME`. Presets with a
zero-knowledge family:

- `fp128::Dense` → `fp128_dense_zk`;
- `RecursiveCommitmentConfig<fp128::OneHot>` → `fp128_onehot_recursive_zk`.

The headline Jolt instance is
`ZkCommitmentConfig<RecursiveCommitmentConfig<fp128::OneHot>>`.

Both clear entry points, `batched_prove` and `batched_verify` (and so the
scheme's `batched_prove` and `verify_standalone`), call
`require_clear_opening`, which refuses a zero-knowledge configuration. A
zero-knowledge configuration is opened only through the masked prefix
(§4) and the Σ-protocol.

### 1.3 Descriptor mode bit

`SetupSection::from_parts` takes the `ProtocolFeatureSet` from the
configuration (`zk = Cfg::IS_ZK`, in `transcript_instance_descriptor`).
The descriptor decodes against the expected feature set, passed as its
deserialization context, and rejects any other set. The bit never reads a
Cargo feature. Clear descriptors keep `zk = false`, so clear bytes do not
change. A clear verifier rejects a zero-knowledge proof at the descriptor,
and a zero-knowledge verifier rejects a clear one.

### 1.4 Schedule binding and catalog admission

- **Plan digest.** `PlanSection.effective_schedule_digest` covers the whole
  zero-knowledge schedule: the prefix rows, the cut (`FoldSchedule.zk`, a
  `ZkOpeningSchedule`), every randomness width, the one-shot targets and
  the residual's parameters. The cut is serialized and appended to the
  schedule's descriptor bytes (tag 0x5a, then the cut) only when present,
  so clear artifacts, row digests and descriptors are unchanged. The
  grinding-plan digest binds each fold-response run's `nonce_bits`, so the
  one-shot runs (§2.1) are bound as well. The verifier derives both digests
  from the admitted row, so a proof made for one mode fails Fiat–Shamir in
  the other even before the descriptor bit is read.
- **Catalogs.** `akita_schedules::ScheduleAdmission::{Clear,
  ZeroKnowledge(root source)}` is part of `ValidatedScheduleCatalog`
  (`try_new_admitting`, `from_artifact_bytes_admitting`, `admission()`). A
  clear catalog rejects every zero-knowledge row, and a zero-knowledge
  catalog rejects every clear row. `TrustedScheduleCatalog::new` refuses a
  catalog whose admission is not `admission_of::<Cfg>()`.
  `TrustedScheduleCatalog::try_from_rows` admits rows as `Cfg` does.
- **Row admission.** `ZkOpeningSchedule::admit` (§4.1) runs in release at
  catalog admission and again where the backend first samples hiding
  randomness. It returns `AdmittedZkSchedule`, the only input from which
  `ResidualGeometry` and the Σ shape are built (§4.7).

### 1.5 Clear invariance

For any clear `Cfg`, a build with `zk` produces the same setup,
commitment and proof bytes as a build without it. Every zero-knowledge
change is inert at zero randomness width, `nonce_bits` from the clear plan,
and `zk = false`. `golden_proof_digests` pins proof digests for several
catalog rows and runs on every PR.

## 2. One-shot fold responses

A clear fold squeezes a seed, sends a 12-bit LEB128 nonce found by
`first_jointly_accepted_nonce`, and draws its challenges from the fork at
that nonce. The honest prover moves to the next nonce whenever the
response probe (`admit_fold_response`) rejects. The probe reads the
witness, so the accepted nonce is a function of the witness. A masked level
cannot publish it.

### 2.1 Encoding: a run with `nonce_bits = 0`

- **Plan.** At each level ≤ `cut_level` the plan has the ordinary
  `GrindingRun` at site `FoldResponse { level }` with `nonce_bits = 0`.
  `FoldResponseSearch` derives `OneShot` from it (clear runs derive
  `Grind`). `append_canonical_bytes` already binds `nonce_bits`, so the
  plan digest distinguishes the modes without changing the policy bytes
  that every plan digest absorbs.
- **Validation.** `GrindingRun::validate` admits `nonce_bits = 0` at a
  `FoldResponse` site only for a plan built from a zero-knowledge schedule.
- **Transcript.** `commit_fold_response` and `read_fold_response` branch on
  the run's `nonce_bits`, which is public plan data. At 0 they send and
  receive nothing (`receive_nonce(0)` would read a byte, so it is never
  called), and the fork's counter is 0. The order is: site, seed squeeze,
  then the challenges, after any proof of work the plan schedules. No
  witness-dependent value precedes the challenges.
- **Prover.** `replay_multi_group_fold_decompose_witnesses` has a one-probe
  branch selected by `nonce_bits = 0`. If the probe rejects at counter 0,
  for any group or chunk, the prover returns
  `AkitaError::FoldOverflow { level }`. It never tries counter 1.
- **Rerun.** jolt maps `FoldOverflow` to `ZkError::FoldOverflow`, and the
  caller reruns the whole proof. The rerun resamples every pad, every
  commitment's randomness, and the Σ mask. Reusing a pad after an abort
  would publish two maskings of one decoded value. Paper hybrid H1 charges
  the abort event 2δ_fold.

### 2.2 Digit sizing

`HonestFoldSizingQuery` carries a `failure_target: FoldFailureTarget`:

```rust
pub enum FoldFailureTarget {
    /// Sizing for a prover that retries: each policy's own per-attempt constant.
    PerAttempt,
    /// One shot: failure probability ≤ 2^−security_bits / masked_folds for this fold.
    OneShot { security_bits: u32, masked_folds: u32 },
}
```

Each existing policy reads the target; there is no new policy variant.

- **`BalancedSignedDigitFoldPolicy`.** `PerAttempt` keeps the per-attempt
  rate 1/8. `OneShot` replaces the union term `ln(2N/(1−p))` with

  ```text
  L = ⌈(⌈log₂ 2N⌉ + security_bits + ⌈log₂ masked_folds⌉) · 71/100⌉
  ```

  in checked integer arithmetic. Since `ln x ≤ ⌈log₂ x⌉·ln 2` and
  `ln 2 < 0.71`, L upper-bounds `ln(2N) + (security_bits + log₂ masked_folds)·ln 2`.
  N counts every physical response coefficient over all witness windows,
  not the per-window count the per-attempt rule uses. The cap is
  `t* = ⌈√(2σ²L)⌉` with `σ² = blocks · η² · ‖w‖∞²`, capped by the
  deterministic envelope. The tail is evaluated at (L+1)², which covers
  both ends of the balanced range (Lemma `wo:lem:fold-overflow`).
- **`UnitOneHotFoldPolicy`** (the one-hot root Jolt uses). `PerAttempt`
  keeps 7/8. `OneShot` puts `2^−security_bits / masked_folds` in the same
  Chernoff solve, with `2^security_bits` built exactly from its exponent
  bits. When the solve does not reach the target within the policy's λ
  grid, the policy falls back to the deterministic cap, at which δ = 0.
- **Modeled caps.** A masked level never takes one. The planner drops the
  modeled ℓ∞ cap there (the recursive and setup-prefix candidates feed a
  per-attempt rate of 1/40, which is never a one-shot bound), and it emits
  no ℓ2 route (§2.5).

Every replacement over-approximates the lemma's tail, so the digit range is
at least the paper's and δ_fold only shrinks.

The extra digits are applied before `rounded_up_role_a_inf_norm`, so the A
matrix is re-priced at the enlarged range, as Lemma
`wo:lem:randomized-base` requires for each schedule. The schedule audit
recomputes the A bound from `num_digits_fold`. At k = 2 the re-priced A
ranks for levels 0 and 1 equal the clear ranks, as paper §6.2 states.

### 2.3 The failure budget

`security_bits = 136` and `masked_folds = (cut_level + 1) · W_ℓ`, where W_ℓ
counts the groups that carry witness data at level ℓ (§3.4). Each masked
level gets an equal share of 2^−136, split equally among its witness groups,
so the union over all witness responses is

```text
Σ_ℓ W_ℓ · 2^−136 / ((cut_level + 1) · W_ℓ) = 2^−136 = δ_fold.
```

A single-group level has `masked_folds = 2`, so each such fold fails with
probability at most 2^−137. Theorem `wo:thm:zk` charges 2δ_fold ≤ 2^−135.
The paper's 2^−950 figure is per fold.

At the headline level-1 shape the one-shot range meets
`2N·exp(−reach²/2σ²) ≤ 2^−137`, checked in exact rational arithmetic with
`ln 2 < 6932/10000`. The headline root's one-hot thresholds are 564 per
attempt and 1469 one-shot; both fit its four base-8 digits (positive reach
1755).

The planner prices masked fold responses at the clear nonce width. That
overcounts by at most two bytes per level and affects ranking only.

### 2.4 Challenge families and clear rows

The paper assumes the unfiltered d = 64 family (η² = 71) at every fold up to
the cut. Akita's masked levels use:

| Row | Level 0 | Level 1 |
|---|---|---|
| headline, `RecursiveCommitmentConfig<fp128::OneHot>` 36:1 | d_a = 512, packing in the 64-subring, (31 ±1, 10 ±2), η² = 71 | d_a = 512, evaluation trace, (19 ±1), η² = 19 |
| `fp128::Dense` 14:1 | d_a = 512, packing in the 64-subring, (31 ±1, 10 ±2), η² = 71 | d_a = 256, packing in the 128-subring, (31 ±1), η² = 31 |

All are unfiltered: the operator-norm rejection exists only on ℓ2 routes,
which admission forbids at masked levels. So the lemma's 1/p factor is 1,
and the one-shot sizing uses each family's own η².

**Clear rows are not one-shot rows.** Clear rows are sized per attempt, and
most are one digit short of the one-shot target at a masked level. Among
single-group fp128 rows with more than two levels:

- one-hot, non-recursive: 15:4, 18:1, 20:1, 20:2, 28:1, 30:1, 32:1, 40:1,
  44:1 and 50:1 are short at level 0; 20:4, 32:4 and 36:1 at level 1; only
  16:2 passes;
- dense: every row (14:1, 15:2, 16:1, 16:2, 17:4, 24:1, 26:1, 28:1, 30:1,
  32:1) is short at level 0.

So zero-knowledge rows always come from the zero-knowledge planner
(`find_zk_schedule`), never from a clear row.

### 2.5 ℓ2 routes are excluded at masked levels

An ℓ2 route sends the response energy ‖z‖₂² as a clear 16-byte integer
(`akita-types/src/l2_sites.rs`), which the verifier compares with
`response_l2_sq_cap` (`akita-verifier/src/stages/physical_l2_norm.rs`). At
a masked level z is the next witness, which paper §4.1 never reveals. Its
energy is a function of the witness, and no hybrid of `wo:thm:zk`
simulates it. **An ℓ2 route at a level ≤ `cut_level` is not zero knowledge,
independent of any retry.** The ℓ2 cap is also sized for a retrying prover
(§2.6.3): one-shot rejection at that cap has a constant, witness-dependent
probability, not δ_fold.

So `ZkOpeningSchedule::admit` rejects `InnerCommitSecurityRoute::L2` at
every level ≤ `cut_level`, which also excludes every operator-norm
challenge family there, and the zero-knowledge planner never emits one. A
masked channel refuses the plain `exchange` the ℓ2 integer uses, so a
schedule that slipped through would fail closed. Admitting ℓ2 needs a check
that never reveals ‖z‖₂² (for example a range proof on cap − ‖z‖₂² inside
the masked transcript), the one-shot ℓ2 sizing of §2.6.3, and the re-pricing
of §2.6.5. That is an open question (§10).

### 2.6 Public retry under operator-norm families (design, not built)

The sparse challenge stream (Akita #255) draws each fold group's challenges
from one squeezed message. Under the two operator-norm families a message
can run out of accepted candidates, and the honest prover then moves to the
next nonce. This section shows that a retry on run-out alone is zero
knowledge, sizes it, and specifies it, for use once §2.5's exclusion is
lifted. Nothing here is built.

Operator-norm families occur only on ℓ2 routes (the family's rejection is set
exactly when the level has a `response_l2_sq_cap`), and ℓ2 fold grinding
requires exactly one scalar group. So every operator-norm level has G = 1.

#### 2.6.1 The retry trigger

Order of one fold, for prover and verifier alike:

1. The fold-response nonce n, a LEB128 prover message bounded by
   `nonce_bits`.
2. The fold proof of work at `FoldChallengeWork`: the least grind nonce
   below 2^(g+7).
3. The group's public payload is absorbed, one fixed-length message is
   squeezed and decoded. The decoder returns `None` exactly when fewer than
   W of the B candidates have their rank in range and pass the certified
   operator-norm predicate. The verifier maps `None` to `InvalidProof`.

The honest loop moves to the next nonce on (E), the message running out, or
on (R), the response probe rejecting. (R) is either (R∞), a coefficient
outside the digit range, or (R₂), ‖z‖₂² above the cap.

- The trigger is mixed: one loop retries on E ∨ R∞ ∨ R₂, and both R events
  read the witness.
- E alone is public. Whether message m runs out depends only on m, through
  its ranks and the predicate table. The message is a squeeze of the sponge
  at σ‖n‖w, where σ is the state after the public prefix and w is the least
  grind nonce with a public property. So E(n) = E(σ, n, H) is a
  deterministic function of σ, n and the oracle H.
- The verifier sees the accepted n and w. Rejected candidates leave no
  bytes: previews run on detached sponges.

The nonce precedes the challenges and the response is computed after them,
so the nonce is a challenge-candidate counter. It carries witness
information only because the prover skips on R. **The protocol needs no
change:** in zero-knowledge mode the prover skips only on E and turns R into
`FoldOverflow`. No byte, domain or verifier rule changes.

#### 2.6.2 The public retry and its zero-knowledge lemma

With G = 1 and N = 2^`nonce_bits`:

```text
PublicRetry(σ, H):
  for n in 0..N:
      w ← least grind nonce for σ‖n;  m ← squeeze;  C ← Dec(m)   (public)
      if C = ⊥: continue                                       (E)
      probe the response with C once
      if R∞ or R₂: abort FoldOverflow                          (never retried)
      return (n, w, C)
  abort PoolExhausted                                          (public)
```

Write n*(σ, H) for the counter it returns.

**Notation.** S is the challenge shell: a coefficients ±1 and b coefficients
±2, with η² = a + 4b and κ₁ = a + 2b. A ⊆ S is the accepted set: rank in
range and ‖c‖_op < Γ = 18 under the strict predicate. Let p := |A|/|S|. The
certificates (`akita-challenges`) give:

| Family | \|A\| | \|S\| | p |
|---|---|---|---|
| d = 64, (31, 11) | ≥ 2^128.062 | 2^130.152 | ≥ 2^−2.090 ≈ 0.2349 |
| d = 128, (31, 0) | ≥ 2^128.563 | 2^129.621 | ≥ 2^−1.058 ≈ 0.4802 |

The rates 0.1305 and 0.3694 quoted for #255 are weaker lower bounds and
remain valid if substituted.

**Lemma 2.6a (the public retry is simulatable).** Let Π' be Whiteout in which
each masked fold ℓ runs `PublicRetry` instead of the counter-0 probe of
§2.1, and in which ‖z‖₂² is not revealed (§2.5). Let:

- δ'_ℓ = δ'_ℓ,∞ + δ'_ℓ,2 bound the probability of R∞ ∪ R₂ at n*, given the
  history (Lemma 2.6b and §2.6.3);
- ΔQ bound the oracle queries the retries add to the honest algorithms;
- h_σ lower-bound the min-entropy of σ given everything the distinguisher
  has seen or queried before the fold.

Then Theorem `wo:thm:zk` holds for Π' with error

```text
2δ'_fold + δ_LHL + T_att·ε/M + (T_att·(Q_tot + ΔQ) + T_att²)/2^ν + ε_fresh,
δ'_fold = Σ_{ℓ ≤ cut_level} δ'_ℓ,   ε_fresh = Σ_{ℓ ≤ cut_level} N·Q_tot/2^{h_σ}.
```

The simulator runs the lines of `PublicRetry` marked public on its simulated
σ.

*Proof sketch.* Follow the hybrids of `wo:thm:zk`.

- **Determinism.** (n*, w*, C) = F(σ, H) for a deterministic F the verifier
  could evaluate. The only witness-dependent action at the fold is the
  probe, and it affects the view only through the abort.
- **Freshness** (new; implicit in the paper's H1 at N = 1). Let Fresh be the
  event that no point σ‖n with n < N was queried before the fold, by the
  distinguisher or by an earlier honest step.
  - σ absorbs commitments and masked messages that carry fresh pads and
    commitment randomness, so given the distinguisher's view it has
    min-entropy at least h_σ, and Pr[¬Fresh] ≤ N·Q_tot/2^{h_σ}.
  - On Fresh, the oracle answers at σ‖n are uniform and independent of the
    fold inputs x and of the history. Messages at distinct n are
    independent, and whether a message runs out does not depend on its
    accepted values.
  - So, given x and the run-outs at n < n*, the W challenges at n* are
    i.i.d. uniform on A.
  - Without this step, a distinguisher that chose the witness after querying
    H could correlate x with the challenges.
- **H1.** Replace the prover by P', which at n* continues with a fixed public
  vector instead of aborting on R. E is decided before the probe, so P'
  reaches the same n*. On Fresh the views differ only on R at n*, which has
  probability at most δ'_ℓ. The same event bounds the failure of
  ct(Q') = 0 in H6.
- **H2.** H2 programs H only at the Σ inputs x_i, which come after the cut
  and lie in their own domain. The retry queries are honest queries, counted
  in Q_tot + ΔQ. No fold point is programmed, so F is evaluated on the same
  oracle in every hybrid.
- **H3 to H9.** Each step replaces a value of the prefix. (n*, w*, C) are F
  of already-replaced values and of H, so they are the same public
  post-processing in neighbouring hybrids, and no step's distance grows.
  H7's premise still holds, with n* and w* among the earlier public values.
- **Final hybrid.** The simulator evaluates F on its simulated σ, including
  the `PoolExhausted` outcome, which is the same public function. So the
  simulator outputs ⊥ with the same probability.

Summing the losses proves the bound. □

*The forbidden retry.* Retrying on R makes n* depend on the witness. The
accepted C is then conditioned on the witness, and P' would reach a
different n*, which breaks H1. `PublicRetry` keeps R one-shot: one probe,
then an abort and a whole-proof rerun with fresh pads and randomness (§2.1).

**Lemma 2.6b (ℓ∞ overflow under A).** Fix the fold's inputs x_1..x_ℓ, where
ℓ = W is the pair count, with coefficients of absolute value at most β. Let
the challenges be i.i.d. uniform on A. A response with n coefficients,
represented on [−L₋, L₊], leaves that range with probability at most
δ'_∞ = min(δ_H, δ_B), summing over both ends L ∈ {L₋, L₊}:

```text
δ_H = p^{−ℓ} · n · Σ_L exp(−(L+1)² / (2ℓη²β²))
δ_B = n · Σ_L exp(−(L+1)² / (2ℓη²β²/p + (2/3)·κ₁β·(L+1)))
```

*Proof.*

- **δ_H.** Apply `wo:lem:fold-overflow` to the uniform law on S, which has
  independent uniform signs (the rank reads position bits only, and the low
  w bits are free signs). Then condition on all ℓ challenges lying in A, an
  event of probability p^ℓ.
- **δ_B.** Fix one response coefficient Z = Σ_i Z_i, where Z_i depends only
  on c_i, so the Z_i are independent. A is closed under negation, because
  the shell and the operator norm are, so E Z_i = 0. |Z_i| ≤ κ₁β, and
  E_A[Z_i²] ≤ E_S[Z_i²]/p ≤ η²β²/p. Bernstein's inequality, the integrality
  of Z, and a union bound over the coefficients and both ends finish the
  proof. □

*Cost.* δ_H adds ℓ·log₂(1/p) bits to the target exponent: 2.09 bits per
challenge at d = 64 and 1.06 at d = 128. L grows like the square root of the
exponent, so this is cheap for small W. δ_B inflates the variance by 1/p,
which raises L by about ½·log₂(1/p) (+1.05 bits at d = 64, +0.53 at
d = 128) while the linear term stays small. It wins for large W and costs at
most one more radix-16 digit.

If G > 1 ever carried an operator-norm family, #255's interleaved loop would
probe group g before decoding group g+1. `PublicRetry` must then decode
every group before any probe; otherwise δ'_ℓ needs a union over every
counter at which a probe ran.

#### 2.6.3 A one-shot ℓ2 cap

`response_l2_sq_cap` is the planner's mean energy times an envelope and a
mean multiplier, reasoned for about 4096 attempts. One-shot rejection at
that cap has constant, witness-dependent probability, so a masked level must
never use it, exactly as §2.2 forbids the modeled ℓ∞ cap. Both candidate
one-shot caps use public certified per-input bounds ‖x_i‖₂ ≤ X_i, from the
certified digit alphabets and the randomness columns.

- **Worst case.** ‖Σ_i c_i x_i‖₂ ≤ Σ_i ‖c_i‖_op·‖x_i‖₂ < Γ·Σ_i X_i. Set
  cap = ⌈Γ·Σ X_i⌉². Then δ'_2 = 0.
- **Tail (McDiarmid).** ‖z‖₂ is a function of the independent c_i, and
  replacing c_i changes it by at most 2Γ X_i. Also
  E‖z‖₂ ≤ √(Σ E_A‖c_i x_i‖²) ≤ η·√(Σ X_i²/p). Hence

  ```text
  Pr[‖z‖₂ ≥ η√(ΣX_i²/p) + t] ≤ exp(−t²/(2Γ²ΣX_i²)).
  ```

  At δ'_2 = 2^−137, t = Γ·√(2·137·ln 2)·√(ΣX_i²) ≈ 248·√(ΣX_i²).

**Choice: worst case.** It removes δ'_2 and needs no independence or
freshness premise. With equal X_i its radius 18·√ℓ·X is below the tail
radius (8.66/√p + 248)·X ≈ 266·X whenever ℓ ≤ 218. ℓ2 routes have G = 1, and
their W is the live-block count times the claim count; whether a production
ℓ2 level exceeds 218 is not known. If one does, admission rejects it rather
than switching bounds, which keeps one sizing rule. The worst-case cap
exceeds the modeled cap by about Γ·√ℓ/(η·√m) for envelope multiplier m, so
A must be re-priced (§2.6.5).

#### 2.6.4 Error budget

- **δ'_fold.** The one-shot target stays ≤ 2^−136/2 per masked fold: δ'_∞
  from Lemma 2.6b's digits and δ'_2 = 0 at the worst-case cap. So
  2δ'_fold ≤ 2^−135, as in §2.3. The accepted challenge law has no bias
  term: on Fresh, given that the message does not run out, it is exactly
  uniform on A, the law both lemmas use.
- **T_att** is unchanged: it counts Σ attempts.
- **ΔQ.** Per masked fold and candidate: the grind search (< 2^(g+7)) plus
  the squeeze (≤ 2^8 permutations); counting permutations over-counts oracle
  queries. With N = 2^12, two masked folds and g ≤ 10:

  ```text
  ΔQ ≤ 2·2^12·(2^17 + 2^8) < 2^30.1,    T_att·ΔQ/2^ν ≈ 2^−218.5.
  ```
- **ε_fresh.** With h_σ ≥ 264, the 2λ + 8 bits of one commitment slice
  (paper §6.2), at Q_tot = 2^100:

  ```text
  ε_fresh ≤ 2·2^12·2^100/2^264 = 2^−151.
  ```
- **Pool exhaustion** is completeness only, not a zero-knowledge term: the
  simulator reproduces it exactly. The pool is sized from measured rates
  lowered for sizing, about 2^−161 per message at those rates. At the weak
  certified rate 0.1305 (d = 64), an independent review estimated run-out
  at about 2^−50.8 per message for W = 1, 2^−44 for W = 16, and ≈ 1 for
  W ≥ 256; these figures were not recomputed. The retry count and
  `PoolExhausted` are therefore bounded only heuristically, and #255's "at
  most 2^−32" does not hold at that rate for large W.

The total stays below 2^−128 at Q_tot = 2^100.

#### 2.6.5 Soundness and re-pricing

The verifier is #255's, unchanged: it reads the nonce below
2^`nonce_bits`, checks the fold proof of work, rejects a message that runs
out, and rejects ‖z‖₂² above the cap (or the hidden replacement check of
§2.5). The prover's search policy is not part of the protocol, so a cheating
prover gains no accepting branch over clear Akita at that site. Relative to
`nonce_bits = 0` it has up to 2^`nonce_bits` candidates, each priced by the
fold proof of work (W·2^128 ≤ 2^K·2^g), so the query-counted bound of
`wo:lem:randomized-base` is unchanged. Unfiltered masked levels keep
`nonce_bits = 0`.

Two parameters are re-priced by the lemma's own rule ("re-checked for each
schedule"):

- **ℓ∞.** The enlarged digit ranges re-price A's coefficientwise collision
  radius, as in §2.2.
- **ℓ2.** A's Module-SIS instance is priced at an ℓ2 collision radius
  derived from the cap: two accepted responses differ by at most 2√cap.
  The worst-case cap enlarges that radius, so the audit must re-run the ℓ2
  estimate at the new cap and raise A's rank if needed. The price is
  `akita_schedules::selective_l2_inner_matrix`, which turns
  `response_l2_sq_cap` into the squared collision bound
  (`role_a_collision_l2_sq_for_response_bound`) and looks it up with
  `sis_l2_table_key_for_collision_sq`; the catalog audit recomputes it.

#### 2.6.6 Implementation sketch

- `FoldResponseSearch` gains `PublicRetry`, and
  `ZkOpeningSchedule::fold_response_search(level)` becomes its single owner:
  `OneShot` at unfiltered masked levels, `PublicRetry` at masked
  operator-norm levels, `Grind` elsewhere (`nonce_bits` alone cannot tell
  `PublicRetry` from `Grind`).
- The prover's search splits into `draw(n) -> Option<PublicFoldCandidate>`,
  which takes no witness handle, so witness independence holds by
  construction, and `probe(&PublicFoldCandidate)`.
- `MaskedFoldSizing` gains the operator-norm rejection, p (one owner in
  akita-challenges) and the worst-case ℓ2 cap; the modeled ℓ2 cap is `None`
  at masked levels. `PoolExhausted` becomes a typed error, not charged to
  δ_fold. Clear digests do not change.
- Tests: a run-out at counter 0 returns n = 1 with one probe; two in-range
  witnesses give equal n*, w* and bytes; a rejecting probe never leads to
  another draw; the budget check gains min(δ_H, δ_B), the cap, ΔQ and
  ε_fresh.

## 3. Hiding randomness

### 3.1 What is sampled

"Made at level ℓ" means the level whose prover computes the image. "Output of
level ℓ" is the witness committed by level ℓ's `NextWitnessPayload` (the
paper's w_{ℓ+1}).

| Commitment | Image | Randomness | Joins | Alphabet |
|---|---|---|---|---|
| root, made at commit time | each B slice | `r_root,s` | output of level 0, beside the root's T digits | open basis of level 0 |
| D image made at level ℓ ≤ `cut_level` (`OpeningPayload`) | one per level | `r'_ℓ` | output of level ℓ | open basis of level ℓ |
| output of level ℓ < `cut_level` (`NextWitnessPayload`) | each B slice | `r_ℓ+1,s` | output of level ℓ+1, beside its T digits | open basis of level ℓ+1 |
| precommitted group folded at a masked level | each B slice | per group | output of level 0, in the group's span | open basis b = 8 |
| output of `cut_level`, the residual u_a | one slice | `r_a` | the Σ vector s, not a further witness (§4.8) | radix 16 |

Level ℓ's relation, over its output witness, checks the B rows of its input
commitment and its own D rows. Each randomness vector therefore sits where
its image is already checked. A setup prefix keeps zero randomness: it is
public setup data.

### 3.2 Sizing, one owner

`akita_params::zk::hiding_randomness_columns(image, alphabet)` is the only
caller of jolt-crypto's `lattice::required_randomness_columns`. `image` is
the outer map's `{ rows, ring_dim }`, per slice for B and per level for D;
`alphabet` is the open basis of the witness that holds the randomness.
`b_randomness_columns(group)` and `d_randomness_columns(params)` are its
two wrappers. The planner calls them when it builds a zero-knowledge
schedule, and `ZkOpeningSchedule::admit` recomputes every width in release
and rejects a mismatch. At d_b = 64, rank 1 and log_basis 3, one slice
needs 2,819 coefficients, 45 `R_64` columns.

The width rule is the paper's: the randomness's entropy must be at least the
image's bit length plus 2λ + 8, which Lemma `wo:lem:stat-hiding` turns into
statistical hiding of the image. Randomness is sampled with jolt-crypto's
`sample_box_digits` from the prover's CSPRNG; the box has unit differences
since b ≤ 64 < q^{1/4}/2.

**Norms.** The randomness digits use the open-basis alphabet, the one
alphabet Akita's range check proves. So s_inf does not change, and the
randomness is range-checked because it lies inside `live_coeff_len`. What
changes is length: the next level's block count and coefficient count grow.
The planner sizes t* and the one-shot term on the widened layout and
re-prices the B and D ranks at the widened input width. The paper reports
rank 1 kept at k = 2.

### 3.3 Geometry and code paths

`CommitmentSliceGeometry` and `OpenCommitMatrixParams` carry
`randomness_columns`; `physical_input_width` is message plus randomness
width, and every width-equality site and matrix view reads it. Widening
re-indexes every row after row 0 of the flat matrix, so a zero width leaves
every view byte-identical. `WitnessLayout` has a `randomness` tail section
after the compression spans (B randomness per group in relation order,
slice-major, then the D randomness), so existing offsets keep their values.
The nonzero paths (`commit_outer_slices`, `compute_relation_d_rows`, the
quotient recomputation, `scatter_et`, the verifier's
`SetupContributionPlan`) are inert at width 0.

`commit` on a zero-knowledge row samples the root's randomness from the
backend's hiding source and keeps it in the commitment handle. Hiding
commits use the constant-time digit kernel (`digit_rows_secret`) and wipe
their input buffers.

### 3.4 Groups at masked levels

A masked level folds one or more groups in canonical order: an optional
setup prefix, the root's precommitted groups, and the level's own group. All
of their responses share the level's one probe, and all of their D segments
lie in the level's one D image. Recursive levels have no precommitted
groups, and the root has no setup prefix.

| Group | Source | Commitment | Opening evaluation | Response z_g | B randomness | Fold sizing |
|---|---|---|---|---|---|---|
| own, level 0 | witness | `commit` on a zero-knowledge row, hiding | Jolt's masked claim | next witness | `b_randomness_columns` | root source contract |
| own, level ≥ 1 | witness | `NextWitnessPayload`, hiding | masked `w_eval` | next witness | `b_randomness_columns` | universal balanced, bounded digits |
| precommitted (root only) | witness (untrusted advice) or statement-fixed (trusted advice, a committed program) | caller's `commit` with `GroupContext::explicit`, before the opening | Jolt's masked claim | next witness | `b_randomness_columns`, always | universal balanced, bounded digits |
| setup prefix (level ≥ 1) | public setup slot | setup-derived, public | the slot's MLE at a public point | next witness; a function of public data | 0 | universal balanced, bounded digits |

W_ℓ is the number of groups at level ℓ that carry witness data, every group
but a setup prefix (`zk::witness_group_count`): W_0 = 1 + the number of
precommitted groups, and W_ℓ = 1 above the root.

**Precommitted groups are hiding.** Each one's B slice images depend on its
source, so each slice gets `b_randomness_columns(group)` columns, the same
helper and alphabet as the root. The width is a field of the frozen profile
(`GroupCommitPhaseParams::randomness_columns`), fixed by the caller when it
picks the profile from a zero-knowledge row of the precommit's shape. The
commit path is the existing one: `CpuBackend::commit` with
`GroupContext::explicit(&profile)` samples fresh randomness whenever the
profile's width is nonzero and keeps it in the handle. An explicit profile
does not name the opening that consumes it, so admission fixes the alphabet:
a hiding profile is admissible only at a masked level, every group there
opens at b = 8, and the randomness is drawn from `MASKED_LOG_BASIS`. At the
opening the handle's randomness is placed in level 0's output witness in the
group's `RandomnessOwner::Outer` span.

**Statement-fixed groups are hiding too.** The paper lets a commitment fixed
in preprocessing (trusted advice, a committed program) carry no randomness,
with zero knowledge relative to it (paper §3.1). Akita cannot see
provenance: "precommitted" is a layout notion. Admission therefore requires
the full width for every precommitted group at a masked level. Jolt makes
those commitments hiding and keeps their randomness with the prover's
preprocessing, so zero-knowledge preprocessing is produced separately from
clear preprocessing: a hiding profile comes from a zero-knowledge row, a
clear catalog rejects it, and a clear profile fails zero-knowledge
admission. One preprocessed commitment never serves both modes. Exempting a
group would need a typed per-group mode bound in the row, which does not
exist.

**The setup prefix is public.** Its commitment is setup-derived and known to
the verifier. Its evaluation claim is the slot's MLE at a point drawn from
public challenges (Stage 3 of the previous level, §4.6). Its response
z_S = Σ_b c_b · ŝ_b is a function of the setup and the level's public fold
challenges, so the simulator can compute it and whether it overflows. Its D
segment lies in the level's D image, which r′ hides as a whole
(`d_randomness_columns` depends only on the D matrix shape). In the next
witness z_S is covered by that witness's commitment randomness; at the cut
it sits in w_k as a public segment (§4.7). No message of a masked level
depends on witness data through the setup prefix.

**Sizing.** Each group's one-shot range is sized for
`fold_failure_target(W_ℓ)` = `OneShot { security_bits: 136, masked_folds:
(cut_level + 1) · W_ℓ }` (§2.3). A setup prefix is sized for the same
per-response target but is not counted in W_ℓ: its overflow is a public
event, so it costs completeness (at most 2^−137 per masked level), not zero
knowledge. Single-group levels keep `masked_folds = 2`. The sizing inputs
(`MaskedFoldSizing::for_group`) are per group:

- the root's own group: the root source contract;
- a recursive own group: the universal balanced policy over
  `FoldWitnessNorms::bounded(log_basis_inner, d_A)`;
- a precommitted group or a setup prefix: the universal balanced policy
  over the group's own inner digit alphabet, claim count, blocks, inner
  width, challenge family and open basis.

A row does not record a precommitted group's producer contract, and the
universal bound needs none. Every committed digit lies in the balanced
alphabet of its inner basis, so ‖w‖∞ ≤ 2^(log_basis_inner − 1) for every
producer, and the sub-Gaussian tail of the response holds for every witness
with that bound. For a balanced-digit producer this equals or exceeds the
contract's sizing; for a one-hot producer it is looser than the Chernoff
solve. It only widens digit ranges, so δ_fold only shrinks. The frozen A
bound of a precommitted profile must cover its one-shot digits at the larger
`masked_folds`; the planner skips a profile whose bound or randomness width
does not fit.

**Zero knowledge, per case.**

- *Precommitted witness group.* Its B images are hidden by Lemma
  `wo:lem:stat-hiding` exactly as the root's: same width, fresh uniform
  digits per commit from the alphabet the next range check proves. Its
  response, D segment and evaluation are covered by the hybrids that cover
  the root: the response sits in the next witness, the D segment in the
  hidden D image, and the evaluation is one of Jolt's masked claims. Its
  overflow counts in δ_fold. A commitment reused across proofs is still one
  commitment: its randomness enters each proof only inside a witness that is
  itself committed with fresh randomness.
- *Statement-fixed group.* Hidden as above, which is stronger than the paper
  requires.
- *Setup prefix.* Every value it contributes is public or inside a hidden
  image, and its overflow is simulatable.
- *Several root groups.* The hybrids apply per image; δ_fold is the union
  bound of §2.3.

The paper adds the setup group to the next fold and lists offloading claims
among public messages, but does not say that the setup group's response
joins a masked witness. It is a function of public data and is covered as
above.

**Admission and planning.** Per masked level, `admit` requires every group
to open at b = 8, every group to have at least `MaskedFoldSizing::for_group`'s
digits for `fold_failure_target(W_ℓ)`, a setup prefix to have zero B
randomness, and every other group exactly `b_randomness_columns`. The
planner's `LevelMasking::Masked` carries the level's one-shot target; a
masked level below the cut may consume a setup prefix, sized for that target
with no modeled cap; a masked root folds precommitted groups sized as above
and skips a frozen profile whose randomness width is wrong.

### 3.5 Exporting hiding randomness

A commitment reused across proofs (Jolt's trusted advice and committed
programs) must be openable by any process that holds the data, not only the
one that committed it. In `akita-cpu-backend/src/hiding.rs`, re-exported by
akita-pcs:

- `CommitmentHandle::hiding_randomness() -> Result<HidingRandomness>`
  exports the B randomness; a clear commitment is an `InvalidInput`.
- `HidingRandomness` is the secret: `Zeroizing` digits, a `Debug` that
  prints only the length, and a wire form `to_bytes()` / `from_bytes()`
  (`b"AKHR"`, version 1, a little-endian `u64` count, one byte per digit)
  that rejects another magic or version, a count that disagrees with the
  payload, and an empty payload.
- `CpuBackend::recommit_hiding(family, source, context, randomness,
  expected)` runs `commit`'s path with a one-shot replaying `HidingRng` in
  place of the CSPRNG. The replay fails closed on a length other than the
  profile's `randomness_coefficient_len`, a digit outside its alphabet, a
  second draw, or a clear profile. The result is returned only when its
  coefficients equal `expected`'s, so the call both checks C against
  (source, r) and rebuilds a handle that opens it.

Committing other data under an exported r would leave the hiding lemma,
which assumes one commitment per draw: the B·r terms cancel in a
difference. The API never yields a second commitment under r, because
`recommit_hiding`, its only consumer, accepts only the expected commitment.
The wire form carries no copy of C, and `import_commitment` still refuses
hiding handles.

## 4. Masked levels and the cut

### 4.1 The cut and admission

`cut_level = 1` (k = 2). `ZkOpeningSchedule { prefix, cut }` holds the root
plus the recursive folds up to `cut_level` and the cut parameters.
`ZkOpeningSchedule::admit(schedule, root_source, claim_extension_degree)`
enforces paper Def. `wo:def:source-cutoff` in release, each failure a typed
error:

- `E::DEGREE = 1` (else `Unsupported`): Jolt has no extension-field claims;
- the cut lies before the last nonterminal fold, and no level up to and
  including the one after the cut is a terminal;
- every level ≤ `cut_level` uses the ℓ∞ route, the quotient relation, open
  basis b = 8 for every group, and one witness chunk;
- every masked level satisfies §3.4's per-group digits and randomness and
  carries exactly the D randomness width of §3.2;
- the level after the cut has no setup prefix, so the cut level runs no
  Stage 3;
- the cut successor (level `cut_level + 1`, never folded) describes the
  residual commitment: d_A = d_B, one outer slice, payload mode `Raw`, open
  basis `RESIDUAL_RANDOMNESS_LOG_BASIS = 4`, exactly
  `b_randomness_columns(own)` B randomness columns, zero D randomness, one
  inner digit plane, and both matrices binding at the Σ radius (§5.1);
- levels after `cut_level + 1` carry no randomness in any group.

A trusted clear catalog rejects any row with a cut or with randomness, with
distinct errors. The planner's masked path skips every candidate admission
would refuse.

The masked levels exclude the reduced-evaluation relation because the
masked stage checks (§4.3) are written for the quotient relation, and with
the cut fixed at 1 no test reaches a reduced-evaluation masked level. The
rows of §4.2 assume b = 8 and E = F; admission enforces both.

### 4.2 Transcript of a masked level

| Message | Clear | Zero knowledge |
|---|---|---|
| `RootCommitment`, `RootPoint` | `public_all` | unchanged: a hiding commitment, a public point |
| `Openings` (level-0 statement claims) | `public_all` | `bind_derived_base` per claim; the verifier holds them as Jolt affine forms (§4.7) |
| `OpeningPayload` (D image) | public | `exchange_public_all`, hidden by r′ |
| fold seed, challenges | nonce, then the fork | the fork at counter 0 (§2.1) |
| `NextWitnessPayload` | public | `exchange_public_all`, hidden by r; at `cut_level` it is `cm(u_a)` (§5.4) |
| ring switch α, τ₀, τ₁ | public | public |
| Stage-1 rounds, `stage1_range_image` | private | masked |
| Stage-2 rounds, `stage2_w_eval` | private | masked |
| Stage 3 (level 0 only): claim, rounds, prefix evaluation | private | public, checked in the clear (§4.6) |
| level-1 scalar openings: predecessor `w_eval`, setup-prefix evaluation | `public_all` | `w_eval` through `bind_derived_base`; the setup-prefix evaluation stays public |

Per-element `bind_derived_base` absorbs the same sponge bytes as one
`public_all`, because spongefish's `public_messages` is a loop of
`public_message`. Only jolt-transcript's event log changes: one `Public`
record becomes several.

**Public payloads on a masked channel.** jolt-transcript's `Channel` has an
explicit public send, `exchange_public`/`exchange_public_all`, which on a
clear transcript writes the same bytes as `exchange`. jolt-zk's
`MaskedProver` and `MaskedVerifier` refuse `exchange`/`exchange_all` with
`TranscriptError::ClearMessageOnMaskedChannel` (in Akita,
`AkitaError::UnsupportedSchedule`), so a private value can never leave a
masked channel unmasked by accident. The messages public in both modes use
the explicit send on both sides: the D image, the next-witness payload,
Stage 3's claim, rounds and prefix evaluation. The terminal's messages
(`NextWitnessInnerState`, `TerminalEFields`) and the ℓ2 integer stay plain
`exchange`s: no masked level sends them (§2.5), and a masked channel refuses
them. Clear proofs are byte-identical.

The other open questions about a masked level have short answers:

- `response_l2_sq` is sent only on ℓ2-route levels, which are after the cut.
- `NextWitnessInnerState` is sent only by the last level before the
  terminal, which is after the cut.
- Stage-3 claims are public (§4.6).

### 4.3 One formula per check

jolt-transcript's `private` module owns `ClaimDomain`, next to
`PrivateChannel` and `PadSlots`:

```rust
pub trait ClaimDomain<F> {
    type Value: Clone;
    type Error;
    fn constant(&self, value: F) -> Self::Value;
    fn add(&self, lhs: &Self::Value, rhs: &Self::Value) -> Self::Value;
    fn scale(&self, value: &Self::Value, scale: F) -> Self::Value;
    fn define_auxiliary(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    /// A new auxiliary slot holding x·y; x is a single masked or auxiliary slot.
    fn product(&mut self, x: Self::Value, y: Self::Value) -> Result<Self::Value, Self::Error>;
    /// x·y = target, with no new slot.
    fn product_into(&mut self, x: Self::Value, y: Self::Value, target: Self::Value)
        -> Result<(), Self::Error>;
    /// value = 0.
    fn check_zero(&mut self, value: Self::Value) -> Result<(), Self::Error>;
}
```

| Domain | `check_zero` | `product_into` |
|---|---|---|
| clear value domain (jolt-transcript) | compare, typed error | compare x·y with target |
| jolt-zk `RowsDomain` | push an affine row | push a product row with z = target |
| jolt-zk `OperandShapeDomain` | no-op | record x's pad as an operand, no slot |
| jolt-zk `ValuesDomain` (prover auxiliary values) | as the value domain | as the value domain |

The Stage-1 leaf check and the Stage-2 oracle check are written once in
akita-types (`check_range_leaf`, `check_stage2_oracle`), generic over
`D: ClaimDomain<F>`. The clear verifier uses the value domain, so there is
no parallel masked formula. At b = 8 on the ℓ∞ route each masked level
lowers to:

| Check | Clear formula | Lowering | Products | Aux |
|---|---|---|---|---|
| Stage-1 leaf | claim = Σ_{k≤4} c_k·ri^k | `u = product(ri, ri)`; `product_into(u, c₄u + c₃ri + c₂, claim − c₁ri − c₀)` | 2 | 1 |
| Stage-2 oracle | claim = w·(B(w+1) + R) | `product_into(w, B(w+1) + R, claim)` | 1 | 0 |

In the Stage-2 row, B collects both w(w+1) sites (the virtual term
`batch·eq·w(w+1)` and the binary term `bb·binW·w(w+1)`), and R collects every
w × public term (relation weight, packing, trace and the setup claim). Both
are public. That is 3 products, 1 auxiliary slot and 3 operand pads (ri, u,
w_eval) per masked level; at k = 2, 6 products, 2 auxiliary slots, 6
operand pads and one affine cut row. These are the paper's counts (paper §6.1).

**Recorded, then lowered.** The rows domain needs the masked auxiliary
values, which jolt sends after the cut, so the masked verifier cannot push
rows during its replay. The fold verifier is written once over a
`ClaimReplay` trait (`akita-verifier/src/claims.rs`). The clear instance
checks each claim on the spot. The masked instance (`MaskedReplay`) holds
each claim as an `AffineRow` and records each level's two checks as data:
the leaf, the Stage-2 weights and the claim forms. `MaskedPrefix::lower`
then runs the one formula in any domain: `rows` (`RowsDomain`), `shape`
(`OperandShapeDomain`) and `auxiliary_values` (`ValuesDomain`). The three
domains allocate the same slots because they share one lowering routine.
`MaskedPrefix` is `#[must_use]`: its stage checks hold only once its rows
are checked.

**The packing-sum check needs no row.** The clear verifier compares the
per-group authenticated sum with the batched trace target. Both are formed
from the same openings with public coefficients, so the masked replay
compares the two forms coefficient by coefficient. That is a public check;
it holds for every honest layout, and no change to the transcript can make
it fail.

**The prover.** The prover keeps every clear self-check by running the same
formulas in the value domain on decoded values. It lowers the leaf check over
its own decoded values (`ProveLevelOutput::range_leaf`) in `ValuesDomain`
to compute Akita's auxiliary values, which jolt sends after its own, in
slot order. The Stage-2 check takes no auxiliary slot.

**Product stages and extension claims fail closed.** Admission rejects
both. The masked replay also refuses each: product stages run their rounds
through `verify_single`, whose rounds arrive masked and fail with
`MaskedRoundInClearReplay`, and extension claims fail with
`UnsupportedSchedule`.

### 4.4 Masked sumchecks and channels

- **Rounds.** `verify_single` folds masked coefficients into a claim value
  and drops `PadSlots`, so it cannot serve masked mode. jolt-sumcheck's
  `receive_masked_single(instance, context)` returns the masked rounds and
  draws each challenge through the context, so Akita's per-round
  `grinded_ext_challenge` works unchanged. Akita lowers them with jolt-zk's
  `masking::sumcheck_rows` (`RoundFormat::Compressed` for Stage 2,
  `RoundFormat::EqFactored` for Stage 1) inside the domain of §4.3.
- **Channels.** `ProverGrinding<'a, H, C>` and `VerifierGrinding<'a, 'p, H, C>`
  are generic over `C: Channel<Sponge = H> + PrivateChannel`, defaulting to
  the clear transcript. In masked mode C is the prover adapter over
  jolt-zk's `CommittedPads::send` or jolt-zk's `MaskedVerifier`. Jolt lends
  `&mut CommittedPads` to the opening for levels 0..=`cut_level`. A masked
  channel does not lend its transcript; it offers typed proof-of-work
  operations only (`MaskedProver::{grind, send_nonce, narg_len}`,
  `MaskedVerifier::{check_grind, receive_nonce, remaining}`). Its terminal
  byte sends return `UnsupportedSchedule`. Proof of work, nonces and context
  records go to the clear channel underneath, so they take no pad and cost
  the same bytes in both modes.
- **Errors.** Under `zk`, a masked channel's error that crosses the sumcheck
  driver (`SumcheckError::PrivateChannel`) maps through `From<ZkError>`. A
  transcript or tape failure is `InvalidProof`; a pad schedule or message
  field that disagrees with the schedule is a setup error.
- **Stopping at the cut.** Both roles check `finish_through(cut_level)`: the
  replay consumed every grinding-plan entry of levels 0..=`cut_level` and
  nothing after. The schedule's plan and descriptor still cover the levels
  after the cut, which run nowhere.
- **One backend.** `prove_masked_prefix` proves every masked level on one
  backend and bypasses the registry's dyn executors, which stay typed on the
  clear transcript.

### 4.5 The counting replay

The pad layout is a function of the public schedule, and jolt's
`pad_layout` counts Jolt's stages and the Akita prefix in one replay. A
replay over zero bytes fails the proof-of-work predicates, so the count uses
a third `GrindingReplay` implementation, `CountingGrinding`. It consumes
proof-of-work plan entries without reading a nonce or checking a predicate,
forks every fold response at counter 0, and runs over zero bytes or a reader
that never fails, never over proof bytes (a proof can end early or fail to
decode). Its state is the channel of `CommittedPadsView::counting()`. It
produces only a pad count and an operand list, never a verdict, so the
masked verifier stays the single owner of the schedule.
`count_masked_prefix` is public because jolt's `count_masked` calls it; it
reads no proof of work and gives no verdict.

### 4.6 Masked Stage 3 and several root groups

**Several root groups.** `MaskedOpeningClaims` holds one affine form per
claim of every group, in group order (the order of
`OpeningClaims::flat_evaluations`). The prover rejects forms that do not
decode to the claims. The caller binds the forms to the transcript with its
masked claim messages, sent before Akita runs; `MaskedReplay::bind_openings`
passes placeholders to the masked `bind_derived`, which binds nothing, and
binds only a public prefix evaluation. The verifier batches each form
through `ClaimReplay`, so every group's claim enters the level-0 rows.
Nothing new is sent in the clear.

**Stage 3 at masked level 0.** A setup prefix at level 1 means level 0 runs
Stage 3. Every value it sends is public:

| Value | Send | Why it is witness-independent |
|---|---|---|
| setup-prefix slot id | public bytes | schedule and setup data |
| input claim σ_S | `exchange_public_all` | σ_S = Σ_p S(p)·ω(p): S is the public setup, and ω is a function of the ring-switch challenge α and of τ₁ and r_x, Fiat–Shamir challenges that are public in the masked transcript |
| round polynomials | public rounds (`RoundPrivacy::Public`) | round i is the partial sum of S·ω over the unbound variables, with the bound ones at earlier Stage-3 challenges: a function of S, ω and public challenges |
| input-claim binding | `bind_derived` → public | re-absorbs σ_S, already sent |
| setup-prefix evaluation | `exchange_public_all` | the slot's MLE at the Stage-3 challenge point |

The verifier folds the rounds in the clear and checks
`output = prefix_eval · weight · α̃` through `GrindingReplay::check_public`:
the clear and masked verifiers reject a mismatch, and the counting replay
checks nothing. The Stage-2 relation weight
`… + common_alpha_evaluation() * setup_claim` multiplies w only by σ_S, a
public value, so the level-0 oracle row stays one product. Public rounds
write what a clear transcript's private exchanges write, so clear proofs do
not change.

### 4.7 The hand-off

**Inbound.** In zero-knowledge mode the root takes its statement claims as
affine forms, `MaskedOpeningClaims<F> = Vec<AffineRow<F>>`, built by jolt
from its masked stage-8 claims. The `GroupBatchStatement` supplies only
commitments, points and selection; its evaluations are not read. The prover
checks in release that each form decodes under the pads to its own claim.

**The cut row.** One value, `ResidualClaim`, carries the hand-off: the
point, the masked w̃_k(r₂), and the payload `cm(u_a)` as this verifier
received it. The cut row `ResidualClaim::WITNESS − w̃_k(r₂) = 0` is built
from it, and the residual back end opens that commitment at that point, so
the row, the point and the commitment cannot be wired apart. The point has
one owner, `akita_types::residual_point`, which both roles call: it converts
the cut level's Stage-2 point r₂ once from `LOW_TO_HIGH` to `HIGH_TO_LOW`
through jolt-poly's `match_endianness`, and when w_k is longer than 2^|r₂|
it pads with leading zeros to ⌈log₂ |w_k|⌉ variables. A dense evaluation of
the exported w_k, written from the MLE's definition with the first
coordinate as the most significant bit, equals the decoded w_eval, and the
reversed point does not.

**Outbound.** `prove_masked_prefix` returns Akita's auxiliary values and
`ResidualOpening`: the cut level and point, the decoded w̃_k(point), raw u_a
and its payload `cm(u_a)`, w_k's digits with their alphabet (the cut
level's open basis), t̂_a with its gadget basis, and r_a with its alphabet.
The backend kernel `ResidualExportKernel::export_residual` builds it from
the cut successor's handles before the proof scope finishes: on the CPU
backend, the export packet's logical digits, the gadget decomposition of its
inner rows, and the retained outer randomness. The residual's secret
temporaries, t̂ included, are wiped.

**Public geometry.** The residual's public facts are functions of public data
that both roles need, so they are not fields of `ResidualOpening`.
`akita_params::zk::ResidualGeometry` owns them: w_k's alphabet, the residual
commitment's profile (t̂_a's gadget basis is its outer digits), r_a's
alphabet, and the public spans of w_k (the response and inner-digit ranges of
the cut level's setup-prefix group, §3.4). `akita_pcs::zk::ResidualShape`
builds the Σ-protocol's residual segments and blocks from it:

- w_k padded to whole Akita blocks and cut into runs of native elements,
  public runs marked `SegmentMask::Public`; one t̂ segment per Akita block;
  r_a. An element that straddles a public span's end stays Gaussian, so no
  zero cells are added.
- N, cut from the setup stream: one block per Akita block,
  `A·w_b − G·t̂_b = 0`, and `[B | B′](t̂_a, r_a) = u_a` with u_a a compressed
  target whose payload is `cm(u_a)` (§5.4). The residual must take w_k in one
  inner digit plane of an alphabet that holds w_k's, so `G⁻¹(w_k)` is w_k;
  every admitted row does, and otherwise `UnsupportedSchedule`.
- The evaluation claim, after checking that w_k's padded span keeps
  ⌈log₂|w_k|⌉ variables.

Both take `AdmittedZkSchedule`, so the Σ setup cannot be built from an
unadmitted schedule.

### 4.8 Hiding the residual commitment

u_a, the cut level's commitment to w_k, must be hiding: the paper commits to
w_k "in the same way" as every earlier witness, and "the full outer image
must be hidden" so that the response proof's binary decompositions of u_a
are simulatable.

**Where r_a comes from.** The cut level's successor commitment already takes
its randomness width from the successor's own-group profile and its
alphabet from the successor's open basis. The backend samples r_a there and
keeps it in the commitment material, as for every masked successor. In
zero-knowledge mode the after-cut level is never folded, so its parameters
describe the residual commitment (A_a, [B_a | B′_a]) and nothing else, and
its "consuming opening" is the Σ-protocol. Its open basis is therefore
defined as the Σ alphabet of r_a, `RESIDUAL_RANDOMNESS_LOG_BASIS = 4`
(radix 16).

**Column count.** `r_a` has `b_randomness_columns(after_cut.own_group())`
columns: the §3.2 helper at rows = outer rank, ring_dim = d_B, alphabet
radix 16. The count is per outer slice, and the residual has one slice
because the Σ-protocol's single [B_a | B′_a] block assumes one. On the
checked-in rows (rank 10 over `R_128`, §5.2) r_a is 321 columns.

**Zero knowledge.** u_a's raw image is hidden by Lemma `wo:lem:stat-hiding`
at the same width rule (entropy ≥ image bits + 2λ + 8).

**Paper delta.** The paper puts the randomness of the commitment to w_j into
w_{j+1}, in the alphabet of the witness that contains it. At the cut no
later Akita witness exists, so r_a joins the Σ vector s as its own segment,
in the Σ alphabet, and the Σ segment alphabet check certifies it instead of
a range sumcheck. The entropy rule is unchanged, so the hiding argument is
the paper's. Binding of [B_a | B′_a] is the paper's weak-opening argument at
the Gaussian radius, which covers every segment of s. r_a is a separate
segment after t̂_a and is not part of w_k, so `Witness(0) = w̃_k(r₂)` ranges
over w_k's index space only.

## 5. The residual commitment

### 5.1 Binding at the Σ collision radius

The Σ-protocol extracts s̄ from three accepting responses and binds it
through each commitment matrix M it opens at radius 4Γ_Δ·B_M (Thm
`wo:thm:knowledge`; the binding definition in the paper's security
section). At the cut those matrices are A_a and [B_a | B′_a], and the pad
commitment's A_p, which jolt prices. They must be Module-SIS hard at that
radius, not at Akita's honest fold bounds.

Pricing them at the honest bounds is not enough. With [B_a | B′_a] priced at
ℓ∞ 15 and A_a at about 2^19.7, the Σ-protocol certifies t̂_a and r_a only
inside its Gaussian box, about 2^38.9. A cheating prover could then take a
short w′ for a false evaluation, set t̂′ = G⁻¹(A·w′), and solve the
inhomogeneous SIS instance [B | B′](t̂′, r′) = u_a for r′ with
‖r′‖∞ ≲ 2^32 by lattice reduction, which opens u_a to a false w′.

**Norm: coefficient ℓ∞.** ‖(c − c′)v‖∞ ≤ ‖c − c′‖₁·‖v‖∞, so Γ_Δ = 2κ₁ with
κ₁ = max ‖c‖₁ over the Σ challenge family. The paper sizes these three
matrices coefficientwise. The Euclidean alternative is worse here: B_M would
be the ℓ2 norm of the response over the matrix's whole input, up to √n times
the box, and at [B | B′]'s widths of 10^5–10^6 coefficients its radius
passes 2^56, beyond Akita's Euclidean table (squared collisions up to 2^84).

**Radius.** R_Σ = 4Γ_Δ·B(D) = 8κ₁·B(D), with:

- κ₁ = 31 + 2·11 = 53 for the d = 64 selective-ℓ2 family the Σ-protocol
  samples (`D64_SELECTIVE_L2_CHALLENGE_CONFIG`), so Γ_Δ = 106;
- B(D) = 8(16^D − 1)/15, the magnitude of the D-digit balanced radix-16
  certified box (its negative end; the positive end is smaller), the
  largest response coefficient on any segment of s;
- D = 10, the Σ regime. Every Gaussian segment of one Σ instance has the
  same digit count, and public segments have fewer, so B_M = B(D) for both
  residual matrices.

At D = 10: B = 586,406,201,480 ≈ 2^39.09 and
R_Σ = 248,636,229,427,520 ≈ 2^47.82. jolt-zk owns the formula and D
(`jolt_zk::whiteout::BINDING_RESPONSE_DIGITS`,
`jolt_zk::whiteout::binding_collision_radius()`), and `SigmaParams`
rejects a table whose response digits exceed `BINDING_RESPONSE_DIGITS`, so
no Σ instance is certified at a box larger than the commitments were priced
for. Akita reads the radius and never restates it.

**No column restriction.** The residual takes w_k in one inner digit plane,
which `ResidualShape::new` and `admit` both require. So G⁻¹(w_k) = w_k, and
the Σ input of A_a is A_a's whole matrix of width `positions_per_block`, the
matrix Akita's own level prices.

**The price.** `akita_params::sis::sigma_binding` is a generated ℓ∞ width
table keyed by (modulus, d, R_Σ): the largest secure width per module rank
under the same ADPS16 quantum model and generator as Akita's ℓ∞ table
(`akita-sis-estimator/examples/sigma_binding_width_table.rs`). It has its
own digest, so Akita's main ℓ∞ table, and with it every clear schedule's key
and bytes, is unchanged. `sigma_binding_min_rank(profile, d, width)` is the
lookup.

- `admit`, at level `cut_level + 1`, requires one inner digit plane, A_a's
  output rank at least `sigma_binding_min_rank(d_A, positions_per_block)`,
  and [B | B′]'s at least `sigma_binding_min_rank(d_B, B width with r_a)`;
  each failure is an `InvalidSetup` naming the matrix.
- The planner's after-cut search raises both ranks to those minima: A_a
  first, then [B | B′] inside the r_a fixed point
  (`price_with_hiding_randomness`), since r_a's width follows [B | B′]'s
  rank.
- `FoldSchedule::sis_occurrences` lists both residual matrices at R_Σ for a
  zero-knowledge schedule, so `estimate_schedule_security`
  (`catalog_security --check`) estimates them directly. Every checked-in Σ
  occurrence estimates at 128.26 bits or more.

**No terminal at or after the cut.** `LevelMasking::allows_terminal` drops
terminal candidates at a masked level and at the level after the cut. With a
Σ-bound residual a terminal at level 1 would otherwise be cheaper, and only
`admit` would catch it.

**The other matrices the Σ-protocol touches** are jolt's. A_p commits to
part of s (the pads), so it is priced at the same radius through
`sigma_binding_min_rank`. A_D commits range-checked radix-16 digits, not s,
and the knowledge extractor uses an A_D collision on digit vectors, so it
needs Module-SIS only at the digit-difference bound. jolt prices it with
`akita_schedules::digit_commitment_min_rank(policy, d, width, log_basis)`,
the planner's own D-role digit-commitment price exposed as a stable entry
point.

### 5.2 The residual over `R_128`

Module-SIS hardness at R_Σ fixes the image dimension n = rank·d, not d: the
Σ-binding table gives rank 19 at d = 64 (n = 1,216), 10 at 128, 5 at 256
and 3 at 512 (n = 1,280, 1,280, 1,536) for the residual widths. A matrix's
setup, and the NTT-domain products of N·z in both Σ roles and of the
level-2 commitment, scale as n × (input coefficients) / d, so they fall by
about d/64 at fixed n. The paper chooses larger rings for these matrices
for the same reason.

The planner gives the level after the cut the ring dimension
`akita_params::zk::RESIDUAL_RING_DIMENSION = 128` (d_A = d_B = d_D),
exempt from the monotone dimension ceiling the cut level sets. In
`opening_work_domain`, a level whose `LevelMasking` is `AfterCut` searches
`{uniform(128)}` instead of `dimension_candidates(policy, level, ceiling)`.
The rule reads only the zero-knowledge search's cut, which no clear search
sets, and no `PlannerPolicy` field, so no policy digest, clear schedule,
catalog or descriptor changes. Levels after the residual keep the ceiling
from the residual's dimensions down. `admit` is unchanged: it never compared
a level's ring with its predecessor's and prices whatever residual a row
holds, so a d = 64 residual stays admissible.

**Why lifting the ceiling there is correct.** The ceiling is a rule of the
planner's search domain, not of the protocol:

1. *No check reads it.* `RingDimensionSchedule` enters only the planner
   (`dimension_candidates`, `initial_dimension_ceiling`, the memo key) and
   the policy digest. `FoldSchedule::validate_structure`, catalog admission,
   `admit`, the prover and the verifier never compare a level's ring
   dimensions with its predecessor's.
2. *The interface between levels is dimension-generic.* A level's output is a
   flat coefficient vector. The predecessor's Stage-2 geometry takes the
   successor's ring dimension as a parameter
   (`validate_stage2_successor_capacity`,
   `RelationAddressGeometry::new_with_coefficient_block`;
   `witness_commitment_domain_len` pads the live length to whole
   `R_{d_succ}` elements with zeros). Nothing assumes d_succ ≤ d_pred, and
   the successor's commitment is computed in its own ring.
3. *The after-cut level is never folded* (§4.4, "Stopping at the cut"). Its
   parameters describe the residual commitment (A_a, [B_a | B′_a], r_a)
   only; its D matrix, fold challenge family and every later level are
   planned but never executed.
4. *The Σ-protocol is native over `R_{64m}`.* Restriction of scalars along
   `R_64 ↪ R_128`, X ↦ Y², makes every `R_128`-linear map `R_64`-linear,
   preserves coefficient norms, and turns multiplication by a challenge c
   into multiplication by c(Y²), which has the same ℓ1 norm. So
   ‖(c − c′)v‖∞ ≤ 2κ₁‖v‖∞ holds verbatim, Γ_Δ = 106 and R_Σ ≈ 2^47.82 are
   unchanged, and the binding reduction applies to the `R_128` matrices as
   stated. `ResidualShape::new` accepts any d_A = d_B that is a multiple
   of 64.
5. *Hardness is priced at the row's own d.* The planner asks the Σ-binding
   table at d = 128, `admit` re-checks it, and `sis_occurrences` lists both
   matrices for `catalog_security --check`.

The ceiling's other role, bounding the audited search, is unaffected: the
exception adds one fixed tuple at one level.

**Rows** (default artifact set; rank × width in `R_128` columns; every SIS
occurrence ≥ 128.26 bits):

| row | cut level dims (A, B, D) | A_a | [B_a \| B′_a] | r_a | setup fields | modeled bytes |
|---|---|---|---|---|---|---|
| dense 14:1 | (256, 64, 64) | 9 × 1,024 | 10 × 1,473 | 321 | 1,885,440 | 91,973 |
| one-hot 31:1 + 14:1 | (128, 64, 64) | 10 × 2,048 | 10 × 4,161 | 321 | 5,326,080 | 98,293 |
| one-hot 36:1 (headline) | (128, 64, 64) | 10 × 2,048 | 10 × 10,881 | 321 | 16,777,216 | 103,365 |

(The dev-protocol set: 87,941, 94,261 and 98,331 modeled bytes.) Every
row's cut level has a role ring below 128, so every row uses the exception.
Compared with a d = 64 residual at its binding ranks (19 × `R_64` on both
matrices, 610 r_a columns), residual work (`rank × width × d` over both
matrices) is 0.37×, 0.54× and 0.63×, and the headline's w_k falls from
10.75M to 8.40M coefficients because the planner re-selects level 1. The
modeled bytes count raw u_a (1,280 coefficients); the proof sends
`cm(u_a)` (§5.4). The single-group statement's e has ν = 23; the
multi-group statement and the headline have ν = 27.

d = 256 would cut residual work further (rank 5), but it needs the
presets' B search domain, today `{64, 128}`, to grow, and that domain is a
policy-digest input shared with clear presets. It needs a
zero-knowledge-only domain for the after-cut level and a confirmation that
the monotone ceiling has no correctness role beyond bounding the search.

### 5.3 The cut level is never offloaded

The priced search drops the offloaded edge into the cut level
(`CutOffload::Direct`): the cut level never consumes a setup prefix outside
the test-support entry `find_zk_schedule_offloading_at_cut_for_test`.

- *Why.* A setup prefix at the cut level puts the parent's response and
  inner digits into w_k. w_k is the residual commitment's input, so it
  becomes part of the Σ-protocol's private vector s and of the response
  vector e. The planner's objective (setup first, then proof bytes and work)
  prices none of that.
- *What happens without it.* The `R_128` residual makes the offload win on
  setup from 35 variables. The 36:1 row then has w_k = 36.86M coefficients
  and e with ν = 29, which jolt-zk's `MAX_RESPONSE_NUM_VARS = 27` refuses:
  the row cannot be proved, though nothing unsound is accepted.
- *Enforcement.* `the_priced_search_never_offloads_into_the_cut_level` pins
  the direct cut at 36 and 37 variables, and
  `every_checked_in_zk_row_fits_the_response_bound` checks every checked-in
  zero-knowledge row's e against the bound through jolt-zk's own response
  layout. The bound counts the pads, so it depends on the statement; it is a
  test over the checked-in rows, not a plan-time rejection.
- *Masked Stage 3 at the cut stays supported.* The forced entry removes the
  direct edge into the cut level and nothing else (the offloaded edge must
  still lower the first direct setup, which it does at 31:1), and its rows
  pass the same `admit`, Σ radius included. The multi-group test instance
  plans its 31:1 + 14:1 row with it, so the masked Stage 3 of §4.6 is proved
  end to end. Production rows do not use it.

### 5.4 The compressed residual commitment

The cut level sends `cm(u_a)` instead of raw u_a, and the response proof
checks u_a through its compression witness ξ_a in e, as the paper's Item 3
does.

**Design.**

- *What is sent.* At the cut level's `NextWitnessPayload` site the masked
  prefix sends `cm(u_a)` under the response proof's chain (§6.3). The site,
  its position (after the commitment to w_k, before the ring switch) and
  its class (public, `exchange_public_all`) are unchanged; only the length
  changes. Every other fold, clear or masked, sends its successor payload
  raw.
- *Whose maps.* K₁ and K₂ are the offset-0 prefix of the response setup's
  shared matrix, the matrix the response verifier's chain rows read. The
  masked prefix takes the compression as jolt-zk's `ImageCompression`
  (`AkitaResponseProver` on the prover side, `CompressionShape` for the
  verifier's payload length), so the image the prefix compresses and the rows
  the response proof checks have one owner.
- *Akita's payload mode stays `Raw`.* Akita's own `Compressed` mode puts the
  chain witness into the successor fold's witness and proves it there. The
  after-cut level is never folded, so that mode would have no checker.
  Admission keeps requiring `Raw` (§4.1); what the cut level sends is
  Whiteout's compression, checked by the response proof. No catalog row,
  plan or descriptor changes.
- *Σ statement.* `ResidualShape::blocks` takes u_a as a compressed target,
  `BlockTarget::Compressed { payload: cm(u_a), image }`, with the image on
  the prover's side only. e gains u_a's chain witness after the response,
  V and θ (the order of `SigmaSetup::response_layout`: compressed targets of
  N in block order, residual blocks before the pad block, then t_D). Item 3
  reads u_a as `G₁ξ₁` (`ImageCompression::reconstruct`) in place of a public
  target, Item 2 gains u_a's chain rows, and `I_bin` covers ξ_a. The
  Σ-protocol absorbs every compressed target's payload first, so `cm(u_a)`
  is absorbed a second time there, which adds no byte.

**Knowledge soundness.** This is the paper's protocol at this point, so the
paper's argument applies. The steps that depend on the change:

1. In the tree of `Π^rnd_{≤k}` (Lemma `wo:lem:randomized-base`), the
   commitment to w_k is fixed before fold k's challenges. The transcript now
   fixes `cm(u_a)`, not u_a. Each response-proof leaf extracts an e whose
   ξ_a is binary (the ζ·1_{I_bin} constraint, §6.3) and satisfies the chain
   rows. By Lemma `wo:lem:compressed-binding`, applied per chunk, all leaves
   below one `cm(u_a)` reconstruct the same u_a, or a chunk's K₁ or K₂ has a
   norm-one kernel vector. From there Thm `wo:thm:knowledge` proceeds
   unchanged: Lemma `wo:lem:weak-unique` gives one s̄ across the ρ-leaves,
   and the cutoff certificate opens that u_a.
2. Item 3's equation is `N z = f + c·G₁ξ₁` on u_a's rows, still linear in e
   since c is public. Its weight on ξ_a is the transposed −c weight on u_a's
   rows pulled back through `ImageCompression::reconstruct_transpose`
   (`G₁^T`), the path the pad image already takes; no new weight family.
3. The chain maps are Akita-priced (compression SIS at ℓ∞ = 1). R_Σ, the
   ranks of A_a and [B_a | B′_a], and the Σ box are unchanged: ξ_a is in e,
   not in s, and is certified at {−1, 0} by the binary constraint, not by
   the box.
4. Error terms: chain rows join the α round, whose error stays the lift's
   126/q (§6.5). If |e| crosses a power of two, ν grows by one and the
   ledger by 5/q.

**Statistical hiding.** Unchanged from the paper: `cm(u_a)` and ξ_a are
deterministic functions of u_a, and hybrid H8 replaces u_a, which r_a hides
(§4.8), with a uniform image; the simulator compresses the uniform image.

**Constant time.** u_a is compressed three times: by the masked prefix, by
`ResponseWitness::new`, and by `BackendCompression::check` inside the
response prover's statement check, which also reconstructs it. All run the
constant-time compression path of §6.6, and the reconstruction lifts its
{−1, 0} digits with `lift_ct`. The hiding argument holds for the view the
paper models; a timing channel on u_a's compression is outside that view,
so the path must not have one.

**As built.** `prove_masked_prefix` takes `compression: &dyn
ImageCompression<F>` and sends each fold's successor payload through
`SuccessorPayload` (`RawPayload` for every clear fold and the root; the
prefix's `CutPayload` compresses at `cut_level` and keeps the payload for
`ResidualOpening`). `verify_masked_prefix` and `count_masked_prefix` take
the compression and read only its `payload_len`; `ClaimReplay::successor_payload_len`
returns it at the cut, after checking the image length against
`ResidualGeometry::commitment_len`. `ResidualClaim::commitment` is
`cm(u_a)`. The root sends its successor raw: `prove_masked_prefix` refuses
a cut at level 0, and admission pins `ZK_CUT_LEVEL = 1`.
`BackendCompression` is the response prover's compression on any CPU
backend, public under `test-support` only.

## 6. The response proof

### 6.1 Statement and transcript

`akita_pcs::zk::AkitaResponseProver<Cfg>` implements jolt-zk's
`ResponseProver<F, 256>` for the response statement: Items 1–5 of the paper,
with compressed u_a, t_p and t_D. e is jolt's `ResponseWitness.digits`,
zero-padded to 2^ν; every digit is in [−8, 7]. The proof follows paper §4.3:

1. Commit to e: an ordinary Akita commitment without randomness, as paper
   §4.3 specifies.
2. Draw α. Send `t_×`, run the degree-2 product sumcheck, then send `u_×`
   and `w_×`.
3. Draw η and form the row batch `⟨A, e⟩ = b_η` from Items 2, 3 and 5, the
   three product claims and the chain rows. Draw τ_pad for the padding term
   (§6.4).
4. Run Akita's digit-range reduction at radix 16 (the Stage-1 tree with
   arities [2], then the leaf, as a standalone reduction). It yields the
   range image at τ.
5. Draw γ, then ζ, and run the degree-3 fused sumcheck over
   `A·E + B·E(E + 1)`, with `B_j = (γ + ζ·1_{I_bin}(j))·eq(τ, j)`. Send v and
   check `A(r)v + B(r)v(v+1)`.
6. Open E(r) = v with `batched_prove` under the response schedule
   (`fp128::ResponseDigits`, family `fp128_response_digits`): a dense
   bounded-digit family that bounds committed digits to [−8, 7], with
   ordinary clear grinding. The planner chooses the A basis within (3, 4) and
   pins the root opening basis to 3. The family is planned direct-only:
   recursive setup planning's envelope-first selection kept a one-fold row
   2.3× larger at ν = 24.

The new kernels implement the jolt-sumcheck prover trait, and the new
messages use spongefish codecs under their own site family
(`SITE_FAMILY_RESPONSE`).

### 6.2 Weights

The verifier never builds the statement matrix H. `ResponseStatement::weight_at`
(jolt-zk) evaluates `A(r)` from factored rows: it factors eq(r, ·) at the
native ring dimension D; every response segment starts and ends on a native
element, so z reads as one scale per native element times one common ring,
and N, L and the lifted forms apply with one column combination per row
(`PreparedMatrix::combine_columns`). Item 2 reads V's digit columns per
64-coefficient quarter of eq(r_lo, ·). The chain rows are Akita's
`ReducedCompressionRelationWeights::evaluate_at_point`. `b_η` comes from
`ResponseStatement::constant` and the payloads. The prover keeps the dense
A, built by `ResponseStatement::weigh` (the transposed rows applied to
eq(η, ·)) and Akita's `accumulate_dense` for the chain rows.

The row maps and their transposes live in jolt-zk, not Akita, although the
paper gives Akita the structured weights. Evaluating them in Akita from
accessors would publish every lifted-relation internal and mirror its layout
maps. jolt-zk exposes `rows` (forward), `weigh` (transpose) and `weight_at`
over one row order, and Akita owns the batch order.

The dense alternative was measured first: at |e| = 89.7M (ν = 27, 10
Gaussian digits) a dense `A(r)` took 3.8 s of verifier time, three orders
of magnitude over budget. The paper's other option, offloading the setup
contraction σ_N = Σ_p S(p)·ω_N(p) into the response opening's Stage 3, is
not built: the factored scan, parallelized over blocks and column chunks,
takes about 10–14 ms on 12 idle performance cores at the headline N shapes
(about 44M field entries), under the ≈15 ms budget. The offload would add
about 2.9 KB of proof, grow the setup from 0.70M to 4.19M field elements,
and need a precommitted 2^22 setup-prefix group at fold 0.

### 6.3 Compression

The verifier holds only compressed images: in Item 3 the targets u_a and t_p,
and in Item 2 t_D, become their gadget reconstructions Σ_i 2^i·ξ_i
(negative-binary digits), affine in e. The ξ digits join e, the binary
constraint joins the fused sumcheck on `I_bin`, and Item 2 gains the chain
equations of each image. The images are already hidden, so compressing them
changes no hiding argument.

- **Chunks.** The paper uses one two-map chain per image. Akita's clear
  chain plans stop at an 8 KiB source (`MAX_COMPRESSION_INPUT_BYTES`), which
  still bounds every clear B/D image. Whiteout's images instead chunk at
  `CompressionChainPlan::max_certified_source_coefficients`, the largest
  source whose first map stays inside its rank-one SIS cell under the
  169-bit policy: at q128 the ladder is d = 32 then 16, the cell is 60,460
  `R_32` columns, and the bound is 15,115 coefficients
  (15,115 · 128 / 32 = 60,460). The second map's input is the first map's
  fixed output, so it does not bound the source. `for_certified_source`
  derives the same two maps under the same certification without the 8 KiB
  cap, under its own policy id, `NegativeBinaryTwoMapSisBoundV1` (tag 5),
  which no descriptor admits. An image takes the fewest chunks, balanced to
  within one coefficient, so `cm(t)` is 16·⌈len/15,115⌉ coefficients; t_p
  and u_a take one chunk.
- **Chain rows.** The paper's verifier checks `t = G₁ξ₁`, `K₁ξ₁ = G₂ξ₂` and
  `K₂ξ₂ = cm(t)`. Here `t = G₁ξ₁` is applied coefficient by coefficient
  inside jolt-zk's Items 2 and 3, and each chain equation is one row whose
  negacyclically reduced residual is evaluated at the product challenge α,
  as Akita's `ReducedEvaluation` relation evaluates its own compression
  maps. The rows come from one owner,
  `ReducedCompressionRelationWeights::for_standalone_chains`. A nonzero
  residual has degree at most d − 1 ≤ 31, so it vanishes at α with
  probability at most 31/q; α is drawn after e and every `cm` are fixed. No
  union over maps is needed: a cheat survives only if every violated row
  vanishes, which is at most as likely as one of them vanishing.
- **Padding digits.** Each ξ occupies its map's padded span. The honest
  prover writes zero digits past the real ones, and nothing enforces it:
  padding digits stay inside `I_bin`, so they are in {−1, 0}, and inside the
  map's input width, which Akita's SIS sizing covers. They change `cm`, not
  t, so binding is untouched. Over A7F7 every chain map's digits fill its
  ring rows, so the padding is empty. K₁ and K₂ are the shared matrix's
  offset-0 prefix, and `AkitaResponseProver::setup` refuses a setup too
  narrow to cover them.
- **The binary constraint is load-bearing.** ζ is drawn right after γ, and
  `I_bin` is every compression-witness segment of e. The prover chooses
  `cm`, so without ζ it could rewrite a ξ₁ coefficient's bits (0, −1) at
  (b, b + 1) as (−2, 0): `G₁ξ₁` is kept, and recomputing ξ₂ = Bin(K₁ξ₁) and
  cm = K₂ξ₂ keeps both chain rows, so Items 2 and 3, the chain rows and the
  [−8, 7] range all pass. The range cannot stand in for it: Lemma
  `wo:lem:compressed-binding` needs {−1, 0} digits, and Akita certifies
  compression SIS only at ℓ∞ = 1. `rejects_prover_chosen_compressions`
  sends that forgery, and the verifier rejects at the fused sumcheck; with ζ
  zeroed it accepts.
- **The statement's compression check is canonical.** jolt-zk's
  `check_witness` asks the response prover whether a witness opens a
  payload; Akita answers by recompressing the reconstructed image, so only
  the canonical witness passes. The verifier enforces the weaker chain
  equations and binary digits. Completeness is unchanged; soundness is the
  verifier's.
- **The Σ-protocol's seed.** The paper's attempt fork absorbs t_D; the code
  absorbs `cm(t_D)`, which is also what the prover sends.

### 6.4 Padding term

The paper batches the padding equations e_j = 0 for j ∈ [|e|, 2^ν) as rows of
H by eq(η, ·). Here, after η, the verifier draws τ_pad ∈ F^ν, and A gains
`1[j ≥ |e|]·eq(τ_pad, j)` (`PaddingTerm`; it hides nothing), evaluated as
one restricted equality over the interval [|e|, 2^ν)
(`NegativeBinarySupport`). η then ranges over the real rows only, and the
term is one interval evaluation instead of an eq(η, ·) window at an
unaligned offset.

A nonzero padding entry makes Σ_{j ≥ |e|} eq(τ_pad, j)·e_j a nonzero
multilinear polynomial in τ_pad. Together with the η batch it is one nonzero
polynomial in disjoint variables: for |e| ≥ 1 neither part can be a nonzero
constant that the other cancels, so no batching scalar is needed. The cost
is ν/q and ν more challenges. Tested end to end by committing a −1 past |e|,
which is in the alphabet, a root of E(E + 1), and outside `I_bin`.

### 6.5 Soundness ledger

The paper's 128-bit headline covers zero knowledge, rejection sampling and
fold overflow, but states no total for the response proof's challenge
rounds. Over F_q with q ≈ 2^128, at ν = ⌈log₂|e|⌉, m batched rows and n_×
product pairs, each round's error is the probability that a false statement
survives it. Computational binding of the Akita commitment and Akita's own
opening error are separate.

| Round | Check | Degree / structure | Error |
|---|---|---|---|
| α | Item 4's product lift `Σμ_i X_iY_i − H_e = θ·(X^64 + 1)` at α, with the chain rows | the lift's residual has degree ≤ 126 (products of degree < 64); a chain row's ≤ 31 | 126/q, no union: every violated row must vanish |
| product sumcheck | `Σ_i U_iW_i = t_×` | degree 2, ⌈log₂ n_×⌉ rounds | 2⌈log₂ n_×⌉/q |
| η | row batch `⟨A, e⟩ = b_η` | multilinear in η | ⌈log₂ m⌉/q |
| τ_pad | padding term (§6.4), with the η batch | one multilinear polynomial in disjoint variables | ν/q |
| τ and Stage 1 | digit range at radix 16 | Akita's Stage-1 reduction, with its grinding | Akita's Stage-1 error at log_basis 4 |
| γ, ζ | range image and binary constraint into B | linear in each | 2/q |
| binary constraint | `Σ_j eq(τ, j)·1_{I_bin}(j)·e_j(e_j + 1) = 0` at τ | multilinear in τ | ν/q |
| fused sumcheck | `A·E + B·E(E + 1)` | degree 3, ν rounds | 3ν/q |

At the headline (ν = 27, ⌈log₂ m⌉ ≤ 27, ⌈log₂ n_×⌉ ≤ 10) the explicit
rows sum to at most

```text
(126 + 20 + 27 + 27 + 2 + 27 + 81)/q = 310/q ≈ 2^−119.7,
```

before Stage 1. The α round alone is 126/q ≈ 2^−121. **The response proof is
below 128 bits over a single F_q.** The options to reach the headline are to
draw α (and the sumcheck challenges) from an extension field, to repeat the
α round, or to state the headline as about 2^−119.7 for this component. The
product and fused sumchecks also carry no proof of work: the paper counts
2a + 4ν over q without grinding, so its term holds as stated at q ≈ 2^128;
the reused digit-range reduction keeps Stage 1's grinding runs.

The ledger is priced at ν ≤ 27. jolt-zk's `MAX_RESPONSE_NUM_VARS = 27` holds
that bound, and `SigmaSetup::response_layout`, which both roles build e's
layout with, refuses a longer e (`WhiteoutError::ResponseTooLarge`). The other
two inputs (⌈log₂ m⌉ ≤ 27, ⌈log₂ n_×⌉ ≤ 10) are not enforced in code.

### 6.6 Constant time

Timing is outside the paper's model, so these are implementation
requirements. Each Σ attempt compresses its t_D with Akita's executor before
the accept bit, rejected attempts included, so the executor must be
constant time in its image. The path is `AkitaResponseProver::compress` →
`zk::compression::compress` → `CpuBackend::compress_relation_image` →
`execute_compression_chains` → per map `PackedNegativeBinary::from_coefficients`,
`expand_rows` and `compression_negacyclic_rows` (`ReducedEvaluation`;
Whiteout never runs `QuotientLift`). An audit found three leaks, all fixed
with outputs unchanged:

- the packing branched on `canonical == 0` and on each bit; it now uses
  masks (`overflowing_sub`, a nonzero mask, an unconditional `|=` of the
  bit) and tests the accumulated noncanonical flag after the loop.
  `expand_rows` branched per digit and now writes −bit;
- the map products skipped all-zero digit planes (`CHECK_ZERO = true`), a
  data-dependent change of work. Low bit planes of small magnitudes are
  zero, and at d = 16 a plane of a uniform image is zero with probability
  2^−16;
- the reconstruction `to_ring` used jolt-algebra's `centered_mixed_radix`
  and `digits_to_field`, whose `Fp128::from_i128` branches on the sign.

The compression maps now run under `I8CommitPolicy::ConstantTime`
(`mat_vec_mul_ntt_digits_i8_ct`): no zero-plane skip and jolt-algebra's
`to_ring_ct` (masked Garner, sign-free lift). Clear-mode commits share the
executor and take the same path; on `compression_execution_bench` the
change is about neutral (digitization 25.4–26.1 µs → 12.6–12.8 µs, products
249–254 µs → 259–270 µs), so it is not gated to zero knowledge. Read and
found data-independent: the chunk, stage and chain executors (grouping keys
are plan shapes), the shape checks, `validate_bytes`,
`validate_digit_rows_for_log_basis` (a branch-free min/max fold),
`DigitMontLut::get` on the compression path (a masked index into a
two-entry table), and the bit-to-`i8` conversion.

Hiding commits (§3.3) use the same constant-time digit kernel.

**dudect.** `examples/compression_dudect.rs` (feature `zk-diagnostics`)
times `ImageCompression::compress` on the DIT pool, fixed vs uniform images
at 320 coefficients (t_D) and 512, and `reconstruct` on every length:

```text
cargo run --release -p akita-pcs --no-default-features \
    --features parallel,zk-diagnostics --example compression_dudect -- 1000000 0
```

On an Apple M4 Max under load (load average 10–100), every after-fix test
passes (|t| < 4.5, max 2.52 at 10^6 measurements); before the fix the
fixed-zero class failed at |t| = 12,685 in a crop and the fixed-uniform
class at 331 raw. With the packing and zero-skip fixes but the old
`to_ring`, crops still reached |t| = 10 at 2·10^5, so the constant-time
reconstruction is needed too. At u_a's length a lower-power run (2·10^4)
passed with compress max |t| 2.59 and reconstruct max 1.80.

**The response prover.** It reads e only at public indices and lifts its
digits with `jolt_algebra::lift_ct`: `AkitaResponseProver::prove`,
`FusedKernel` (the streamed rounds' chunks and the held tables' folds),
`SplitB` (indexed by `I_bin`'s public positions and the round index), and
jolt-zk's `PlaneLayout::recompose`. e's digits and every table derived from
them are wiped when freed: `Zeroizing` digits, `SecretTable` for field
tables and the streamed chunks' scratch, `SignedBlockPoly`'s drop for the
committed `i8` copy, `SignedScratch`'s for the signed-block reader's staged
rows, and jolt-zk's `SecretVec` for decoded parts. The signed-block reads
lift by `lift_ct`, signed readers skip no zero, and every comparison on e's
extrema has the same outcome for every in-alphabet e. Open items are in
§10.

## 7. Publication

`akita_pcs::zk` re-exports `ZkCommitmentConfig`, `prove_masked_prefix`,
`ResidualOpening`, `MaskedPrefix`, `ResidualClaim`, `ResidualShape` and
`AkitaResponseProver`. The planner's `find_zk_schedule` plans every row of a
zero-knowledge family, precommit producers included. Checked-in families,
in both artifact sets and catalog snapshots:

- `fp128_dense_zk`: 14:1, the hiding precommit producer and the
  single-group statement;
- `fp128_onehot_recursive_zk`: the 36:1 headline, and 31:1 beside a 14:1
  hiding precommit (as Jolt commits untrusted advice);
- `fp128_response_digits`, clear: one row per end-to-end statement's e
  (ν = 23 and 27).

`schedule-planner-cost.tsv` carries the zero-knowledge rows, so the
schedule-artifact drift job covers them. CI `test-zk` runs the workspace
tests with `zk`, the end-to-end gating tests with akita-pcs's
`test-support` and `logging`, and the feature-tree check of §1.1;
`test-dev-protocol` also runs `tests/masked_prefix.rs`.

## 8. Security summary, mapped to the paper

| Choice | Paper item |
|---|---|
| `nonce_bits = 0` runs; one probe; whole-proof rerun with all randomness fresh (§2.1) | paper §4.1 "Fold responses"; Lemma `wo:lem:fold-overflow`; H1 |
| One-shot targets on every policy, 2^−136 split over (cut_level + 1)·W_ℓ responses; modeled caps off before the cut; A re-priced (§2.2–2.3) | Lemma `wo:lem:randomized-base`; Thm `wo:thm:zk` (2δ_fold) |
| ℓ2 routes rejected at masked levels; a public retry only after a hidden energy check (§2.5–2.6) | Thm `wo:thm:zk` H1 with freshness, H2 with ΔQ; Lemma `wo:lem:fold-overflow` in tuple or Bernstein form; Lemma `wo:lem:randomized-base` re-pricing |
| Randomness per slice, one sizing helper, open alphabet, in the next witness; every precommitted group hiding; setup prefix public (§3) | paper §4.1 "Commitment randomness"; Lemma `wo:lem:stat-hiding`; H9; paper §3.1 advice (stricter) |
| Setup-prefix overflow is a completeness term only (§3.4) | Thm `wo:thm:zk` (union over witness responses) |
| Stage 3 public; no setup prefix at the cut (§4.1, §4.6) | paper §3.1; Def. `wo:def:source-cutoff` |
| One `ClaimDomain` formula per check; rows never dropped (§4.3) | Lemma `wo:lem:residual-correct` |
| Mode bit from config; zero-knowledge schedule in the plan digest (§1.3–1.4) | the transcript binds statement and schedule (Def. `wo:def:algebraic-verifier`) |
| Residual hiding with r_a in the Σ alphabet (§4.8) | Lemma `wo:lem:stat-hiding`; paper §4.1 "The residual commitment" |
| Residual binding at R_Σ, over `R_128` (§5.1–5.2) | Thm `wo:thm:knowledge`; the paper's coefficientwise sizing |
| `cm(u_a)` with ξ_a in e (§5.4) | Item 3; Lemma `wo:lem:compressed-binding` |
| e committed without randomness; clear Akita opening (§6.1) | paper §4.3 and §5.3 |
| Binary constraint, chain rows at α, padding term (§6.3–6.4) | Lemma `wo:lem:compressed-binding`; the ledger of §6.5 |

## 9. Verification

Clear invariance: `golden_proof_digests` and the byte-identity dump hold
(§1.5). Each item below is a permanent test unless marked manual.

**One-shot folds and admission.**

- A zero-knowledge plan is the clear plan of the same folds except for a
  `FoldResponse` run with `nonce_bits = 0` at every level ≤ `cut_level`. The
  prover's argument string does not grow at those sites, and an inserted
  nonce is rejected.
- A backend that rejects the response at a masked level makes the prover
  return `FoldOverflow { level }` after exactly one probe there.
- The exact-rational one-shot check of §2.3
  (`one_shot_headline_balanced_fold_meets_the_per_fold_target`). The
  balanced cap is pinned as the least integer meeting the closed form, with
  one and two witness windows, and the headline root's one-hot thresholds
  are frozen (564, 1469). The re-priced A ranks at the masked levels equal
  the clear ranks (`one_shot_sizing_keeps_the_clear_a_ranks_at_the_masked_levels`).
- `ZkOpeningSchedule::admit` rejects each violation of §4.1 and §3.4 with its
  own error: an ℓ2 route at a masked level; a precommitted group with no or
  wrong randomness; a setup prefix with randomness; short fold digits in a
  preceding group; a group opened at a basis other than 8; randomness after
  the cut; a setup prefix after the cut; a compressed residual payload; a
  residual priced below R_Σ (`admission_prices_the_residual_at_the_sigma_radius`
  rejects a rank-1 [B_a | B′_a] and an A_a one rank short, each by name, and
  two inner digit planes).

**Hiding.**

- Two hiding commitments to one polynomial differ, and two explicit commits
  of one source under a hiding precommitted profile differ.
- Flipping one randomness digit breaks that level's relation.
- `headline_slice_needs_forty_five_columns` and
  `residual_commitment_needs_ninety_seven_columns` (rank 3 over `R_256`) pin
  the sizing helper.
- `exported_hiding_randomness_rebuilds_exactly_its_commitment`.

**Masked prefix** (`tests/masked_prefix.rs`, `stage_checks/tests.rs`).

- On honest decoded values every emitted row holds; shifting any decoded
  message by +1, with the auxiliary values recomputed consistently, breaks
  at least one row; every pad slot has a nonzero coefficient in some row or
  lowered round; the shape domain counts the lowering's products and
  auxiliary slots.
- `CountingGrinding` counts the same pads and operands as an honest masked
  run, on zero bytes and on random bytes; `finish_through` requires exactly
  the levels up to the cut.
- `multi_group_masked_prefix_with_a_setup_prefix` runs the §3.4 row (a dense
  precommit beside a one-hot root at 31 variables, a level-1 setup prefix)
  through the masked prover and verifier, including Stage 3 at level 0.
- `ResidualOpening` satisfies `N·s = t` through jolt-crypto's `MatrixView`
  and `PreparedMatrix::apply_centered` on Akita's setup stream
  (`A·G⁻¹(w_k) = G·t̂_a` and `[B | B′](t̂_a, r_a) = u_a`); two openings of one
  statement send different u_a and r_a; the dense w̃_k(point) equals w_eval.
- `clear_entry_points_refuse_a_zero_knowledge_configuration` and
  `the_descriptor_binds_the_configuration_mode`.
- The dynamic-backend fault tests cover Stage-1 round panics and Stage-2
  oracle misses.

**End to end** (`tests/zk_end_to_end.rs`). One transcript: t_p compressed by
the response prover, the masked claims, the masked prefix, the auxiliary
block, `prove_sigma` over the prefix's rows and `ResidualShape`, and
`AkitaResponseProver` from the checked-in response catalog. Both statement
rows verify: the single-group 14:1 row and the 31:1 multi-group row with a
hiding precommit and a level-1 setup prefix. The pads are committed by
jolt-zk's `LatticePadCommitment`, and the same pads mask the messages
through `CommittedPads::uncommitted`; the verifier's view is
`CommittedPadsView::counting()`, checked to end at the layout's count.

- `every_received_byte_is_in_one_public_class` (public-site allowlist,
  gating): the verifier's event log tiles the proof, and each message gets
  one class from an allowlist keyed by site. A masked class without a
  `TranscriptOp::Masked` record, or any other class with one, fails. The
  positive control, the same verifier with the statement claims read
  through the clear transcript, is flagged.
- `shifting_any_slot_breaks_a_row` (coverage, gating), on both rows.
- `pads_are_used_once_and_never_reused` and
  `a_fold_overflow_rerun_reuses_nothing` (pad freshness, gating): the
  `Masked` records take pads 0..`num_pads` in order, once each; two proofs
  of one witness from different seeds share no pad, masked value, randomized
  image element, r_a, r_p, Σ nonce or root commitment. Randomness vectors are
  compared whole, since single digits of a small alphabet collide by chance.
  The forced overflow uses a test backend's rejected probe at each masked
  level; the rerun shares nothing with the aborted attempt except the root
  commitment, which is the statement.
- `a_consistent_cheat_fails_at_every_layer`: on the single-group row, one
  case per masked class (claim, Stage-1 round, range image, Stage-2 round,
  w_eval), pads shifted so one message decodes one higher and the auxiliary
  products recomputed. The Σ prover refuses; with `prove_sigma_unchecked`
  the response prover's statement check refuses; with
  `UncheckedResponseProver` as well, a forged response proof over Akita's
  real rows reaches `verify_sigma`, which rejects it. With the residual rows
  dropped on both sides the forgery verifies, so the test pins the rows, not
  only Fiat–Shamir binding.
- `item_3_reads_u_a_from_its_compression_witness`: a prefix that sends an
  honest `cm(u_a + e₀)` fails Item 3 at every layer, and the same unchecked
  path over the honest payload verifies.
  `a_forged_chain_on_u_a_is_rejected_at_the_fused_sumcheck`: a moved chunk
  payload with ξ_a built from it fails only that chunk's K₂ row, and the
  verifier rejects at the fused sumcheck.
- `every_checked_in_zk_row_fits_the_response_bound` (§5.3).

**Response proof** (`src/zk/tests.rs`). `AkitaResponseProver` accepts every
statement jolt's checking prover accepts and rejects each Item 1–5 violation
that `check_witness` rejects, on synthetic `SigmaSetup` statements. On small
shapes the factored `A(r)`, the dense A at r, and H contracted column by
column through `check_witness`'s maps agree; jolt-zk checks `weight_at`
against `weigh` at D = 64 and 256. `rejects_prover_chosen_compressions` and
the padding test of §6.4.

**Planner and parameters.**
`checked_in_zk_rows_commit_the_residual_at_its_sigma_binding_ranks`,
`zk_search_never_ends_at_a_masked_or_residual_level`,
`the_priced_search_never_offloads_into_the_cut_level`,
`cells_match_the_certified_csv_and_the_current_radius`,
`the_binding_rank_grows_with_width_and_covers_no_empty_matrix`, and
`certified_source_bound_is_the_first_maps_sis_width`.

**Manual diagnostics.**

- `examples/zk_distinguisher.rs`: over N proofs for each of two witnesses
  with equal public input, a per-byte χ² homogeneity test over the whole
  proof (about 75 KB), Bonferroni-corrected at α = 0.01. Two injected leaks
  calibrate its power: a masked slot with pad 0, and a slot whose pad has
  only 64 random bits. Both are flagged from N = 16 proofs per witness (5/5
  trials; 2/2 at N = 64) and neither at N = 8; honest runs are never
  flagged. It detects gross leaks only.
- `examples/compression_dudect.rs` (§6.6).
- `examples/zk_response.rs`: proof bytes, prover time and verifier time of
  the response proof on jolt-zk's synthetic instance.

## 10. Open issues

- **Response-proof soundness is about 2^−119.7** (§6.5), below the 128-bit
  headline over a single F_q. The fix (extension-field challenges, a
  repeated α round, or a restated headline) is undecided, as is whether the
  product and fused sumchecks should carry proof of work.
- **Constant time, Akita side.**
  - x86_64 is not measured; the NEON/AVX forward and inverse NTTs and
    pointwise dots are not audited line by line, and dudect covers the NEON
    path only empirically. The guarantee needs DIT on cores that time
    integer operations by operand; the attempts run on a DIT pool.
  - `QuotientLift` compression is not constant time (zero-plane skip and
    `to_ring_cyclic`); no secret image uses it.
  - The executor's intermediate buffers (packed bytes, expanded rows, map
    images) are not wiped. `PackedSignedDigits` sizes its bit width from the
    extrema of e and keeps an unwiped packed copy, and the compression
    state's packed bits drop unwiped.
  - Not audited, because they are shared kernels that take the same digit
    rows as clear proofs: the response commitment's matvec
    (`I8CommitPolicy::Dense`, reconstructed by `to_ring`, not `to_ring_ct`);
    its scalar digit-to-Montgomery fallback `DigitMontLut::get`, which
    indexes a table by the digit (reached with `i16` primes, on x86_64
    without AVX2 or at D < 64, and on targets with neither NEON nor AVX2;
    not on the M4, where the response commit uses `i32` NEON at every
    level); the opening's decompose-fold accumulators; and the digit-range
    session, whose `CompactDigitSource` groups digits by value class, a
    memory access that depends on e. These are requirements on those
    kernels.
  - The commitment kernels of the masked levels run on secret digits once
    per proof; beyond the constant-time digit kernel for hiding commits, they
    are not audited.
- **Constant time, Σ-protocol side (jolt-zk).** After the public blind, the
  Σ-protocol's `N·y` product still fails dudect for constant non-negative
  masks (0, +1, +B) at |t| 4.5–11 in crops (raw ≤ 3.2); the fixed class is
  about 0.05–0.08% slower. The cause is not localized. Jolt keeps Whiteout
  behind an explicit opt-in feature while this is open.
- **ℓ2 routes at masked levels** stay excluded (§2.5). Enabling them needs a
  hidden energy check and the §2.6 design; the pool-exhaustion bound under
  operator-norm families is heuristic, and whether any production ℓ2 level
  has W > 218 is unknown.
- **Q_tot.** The paper defines Q_tot, the oracle queries of the
  distinguisher and the honest algorithms together, but gives no value. At
  Q_tot = 2^100 its term is about 2^−148.6. The evaluation's assumed value
  is a question for the paper authors.
- **Setup cache.** With disk persistence, `akita_setup::load_prover_setup`
  reads the public matrix from the cache file and runs
  `AkitaExpandedSetup::check` on it, but does not re-derive it from the
  seed. A tampered cache changes K₁, K₂ and every other public matrix for
  any prover or verifier on that machine. The response setup's own check is
  about size, not content. The fix belongs to akita-setup.
- **Public spans of w_k.** `ResidualGeometry` marks the setup prefix's
  response and inner-digit ranges public on the argument of §3.4. No test
  recomputes them from public data, and a wrong span would leak through
  z = c·s. The e (outer-digit) range of that group is left Gaussian.
- **Pads bound to their commitment.** jolt-zk builds `CommittedPads` and
  `CommittedPadsView` only in a `ZkBackend`. Akita's end-to-end harness
  commits its pads and masks with `CommittedPads::uncommitted` over the same
  pads, so only the harness ties them together; jolt's backend closes this
  for real proofs. The harness's pad-commitment rank (2) and A_D rank (1)
  are harness choices; jolt sizes them (§5.1).
- **Fold-overflow lemma wording.** "Conditioned on a public event of
  probability at least p" is correct, but the example "such as an
  operator-norm filter" misleads: with a per-challenge filter, a fold of ℓ
  challenges conditions on an event of probability p^ℓ (Lemma 2.6b). The
  paper's masked folds are unfiltered, so no number changes.
- **Pads instead of randomness below the cut.** Plausibly sound if the
  image is revealed only through ξ digits inside masked or Gaussian-hidden
  witnesses, the compressed form is sent after the pad commitment, and its
  coefficients take pads. The root is an exception (it is committed before
  any pad exists and may be reused). It would shrink about 2.8k randomness
  coefficients per slice to a few pads per commitment. Unconfirmed with the
  paper authors; not built.
- **Paper items still open:** d_A = d_B for the residual, the box envelope,
  and ⊥ after T_att.
- **Not exercised.** Product stages, extension-field claims and k > 2;
  admission rejects all three. The headline 36:1 row is planned and
  admitted but not proved here; its sizes are modeled. The one-hot Chernoff
  solve reaching 2^−137 within its λ grid is checked only on the headline
  shapes; elsewhere the deterministic fallback applies with δ = 0.
