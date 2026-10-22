# Spec: Whiteout on Akita `dev` (RP4: M6e, W2, W1, W3, W5)

| Field     | Value |
| --------- | ----- |
| Author(s) | @markosg04 |
| Created   | 2026-10-21 |
| Status    | proposed (revised after protocol and engineering review) |
| PR        | one per slice of §8, into `dev` |
| Parent    | jolt `specs/shared-primitive-layer.md` (RP4), `specs/whiteout-interfaces.md` (§2, §5, Open questions) |
| Siblings  | jolt `specs/whiteout-sigma.md` (W4, register A1–A23), `specs/whiteout-masking.md` (M6b/M6c) |

This note is a design only. Paper labels refer to
`akita-paper/sections/whiteout/` as read on 2026-10-21. Akita references are to
this repository at `87b4ca046` (`whiteout/dev`); jolt references are to jolt
`whiteout/dev` at `591fc9d65`, written `jolt:`. The headline instance is Jolt
at T = 2^24 with `F = Prime128OffsetA7F7` (q = 2^128 − 2^32 + 22537).

**Indexing.** The paper's fold j is Akita's **level ℓ = j − 1**; the root is
level 0. The paper's witness w_j is the output witness of level j − 1. This
note uses levels everywhere except when it quotes the paper. The cut is
defined once:

> `cut_level = k − 1 = 1`.

It is validated schedule data, and k = 2 is the only supported value.

## Decision

RP4 makes Akita levels 0..=`cut_level` run masked and randomized. The opening
stops after Stage 2 of `cut_level`. The Σ-protocol's response is then proved
with an ordinary Akita proof. In outline:

- **W2.** At levels ≤ `cut_level`, a ZK schedule carries a fold-response run
  with `nonce_bits = 0`. That run sends no nonce, draws its challenges from
  the fork at counter 0, and probes the response once. An out-of-range
  response returns `AkitaError::FoldOverflow { level }`. The digit ranges come
  from the existing fold policies, each given a one-shot target probability.
  Clear schedules and their plan digests do not change.
- **W1.** Every B slice and every D image made at levels ≤ `cut_level` gets
  fresh randomness columns. It also covers the root commitment, made by
  `commit_hiding`. One Akita helper sizes the columns, and the randomness
  digits join the witness whose relation checks that image. Clear schedules
  have zero randomness columns.
- **W3.** The stage checks at levels ≤ `cut_level` are written once over
  `jolt_transcript::ClaimDomain`. They run in the value domain (clear), the
  rows domain (masked verifier) and the shape domain (pad layout). Masked
  sumcheck rounds use a context-aware masked receive in jolt-sumcheck. The cut
  hands W6 a typed `ResidualOpening`.
- **W5.** Akita implements `jolt_zk::whiteout::ResponseProver` with raw images.
  The last PR publishes the ZK config type and its schedule family.
  Compression of u_a, t_p and t_D (W5c) lands in RP4, after W5a (§5.4).
- **M6e.** M6e deletes `akita-zk-prover` and `akita-zk-verifier` and updates
  the dependency guards.

The jolt-side prerequisites are fixups to tickets that are still open: M2a and
M6b (§7). The Akita-side prerequisites are fixups to M2b and M3b, plus
carried `main` seams (§8).

## 1. Mode, selection, and binding

- **Feature.** `zk` on `akita-params`, `akita-types`, `akita-prover`,
  `akita-verifier`, `akita-pcs`, `akita-planner` and `akita-config`, off by
  default. It compiles the new modules and the `jolt-zk` dependency (with
  `whiteout`, and with `whiteout-prover` on prover crates only). It selects
  nothing (branches.md rules 2–3).
- **Config hooks** (carried seam S10). `CommitmentConfig` gains
  `fn zk_schedule_family_name() -> Option<&'static str>`, implemented per
  preset; the default is `None`. `ZkCommitmentConfig<Cfg>` accepts only a
  `Cfg` whose hook returns `Some`, and it resolves that family. The adapter
  composes with recursion as `ZkCommitmentConfig<RecursiveCommitmentConfig<Cfg>>`.
  - The recursive adapter forwards the hook to its own
    `RECURSIVE_SCHEDULE_FAMILY_NAME` plus `_zk`, for example
    `fp128_onehot_recursive_zk`.
  - The headline Jolt instance uses exactly that composition. Level 0
    offloads to level 1, so recursive setup planning is required.
- **Descriptor bit** (carried seam S9). `ProtocolFeatureSet::current()` is
  `const { zk: false }`, `SetupSection::from_parts` hard-codes it, and `Valid`
  rejects every other value (`akita-types/src/instance_descriptor/mod.rs:145-149,189,413-421`).
  The seam makes three changes:
  - `from_parts` takes the `ProtocolFeatureSet` from the config:
    `zk = Cfg::IS_ZK`, where `IS_ZK` is false for every preset and true only
    for `ZkCommitmentConfig`;
  - `Valid` compares the decoded set against the expected set, which the
    caller passes in from the config;
  - `current()` is deleted.

  The bit never reads the Cargo feature. On `main` every config is clear, so
  bytes and acceptance do not change. A clear verifier rejects a ZK proof at
  the descriptor, and a ZK verifier rejects a clear one.
- **Schedule binding.** `PlanSection.effective_schedule_digest` covers the
  whole ZK effective schedule: the prefix rows, `ResidualCutParams` (cut
  level, residual matrices, slice count, `r_a` width, payload mode), every
  randomness width, the one-shot targets, and the response schedule.
  `SetupSection` binds the mode bit.
- **Artifacts.** ZK schedule rows are new files in the ZK family; existing
  rows are never modified (rule 6). Until the last PR they exist only as
  test rows from `akita_planner::find_schedule_for_test_zk`, admitted by
  `TrustedScheduleCatalog::try_new_with_test_zk` (test-support + zk).
- **Visibility.** Before the last PR, `ZkCommitmentConfig` and the ZK
  entry points are `#[cfg(feature = "test-support")] pub`. Nothing in a
  default or `zk`-only build can select ZK.

**Clear-mode invariant (C0).** With any clear `Cfg`, `dev` with or without
`zk` produces the same setup, commitment and proof bytes as `main` at the
merge base. PR 1 adds a golden proof-digest test for several catalog rows, so
C0 is checked mechanically from then on (§8).

## 2. W2: one-shot fold responses

### 2.1 Today

Each fold squeezes a seed. It then sends a 12-bit LEB128 nonce, found by
`first_jointly_accepted_nonce` (`akita-prover/src/protocol/fold_grind.rs:199-212`),
and draws its challenges from `Fork::new(seed, nonce)`
(`akita-types/src/transcript_grinding/replay.rs:309-407`, verifier
`:507-523`). Acceptance (`admit_fold_response`,
`akita-cpu-backend/src/opaque/fold.rs:409-452`) reads the witness.

The digit ranges are sized for a prover that retries, with three different
per-attempt targets:

- 1/8 for the universal balanced policy (`akita-params/src/sis/fold_linf_cap.rs:14-15`);
- 7/8 for the one-hot Chernoff policy (`honest_fold_policy.rs:206`);
- 1/40 for the modeled cap that recursive and setup-prefix candidates feed
  in (`akita-planner/src/schedule_params/candidate/recursive.rs:229-236`;
  `candidate.rs:172-177` takes `modeled.min(universal)`).

### 2.2 Encoding: a run with `nonce_bits = 0`

- **Plan.** A ZK schedule's plan has, at each level ≤ `cut_level`, the
  ordinary `GrindingRun` at site `FoldResponse { level }` with
  `nonce_bits = 0`. `append_canonical_bytes`
  (`akita-params/src/transcript_grinding.rs:532-538`) already binds
  `nonce_bits`, so the plan digest distinguishes the two modes without
  changing the policy bytes (`active_grinding_policy_bytes`) that every
  plan digest absorbs. Clear plans and their digests are unchanged (C0).
- **Validation.** `GrindingRun::validate` admits `nonce_bits = 0` at a
  `FoldResponse` site only on `dev`, and only for a plan built from a ZK
  schedule. `main` keeps rejecting it.
- **Transcript** (carried seam S1). `commit_fold_response` and
  `read_fold_response` branch on the run's `nonce_bits`, which is public plan
  data. At 0 they send and receive nothing: `receive_nonce(0)` reads a byte
  today, so it must not be called. The fork is `Fork::new(seed, 0)`. The order
  is therefore: site, seed squeeze, challenges. No witness-dependent public
  value precedes them.
- **Prover** (carried seam S11). `replay_multi_group_fold_decompose_witnesses`
  gets a one-probe branch, selected by `nonce_bits = 0`. If the probe rejects at
  counter 0, the prover returns `AkitaError::FoldOverflow { level }`, a new
  retryable variant that also covers rejection of any chunk. It never tries
  counter 1.
- **Errors.** W6 maps `FoldOverflow` to `ZkError::FoldOverflow`. The caller
  reruns the whole proof. It resamples every pad, g, r_p, every commitment's
  randomness and the Σ mask, because reusing a pad after an abort would
  publish two maskings of one decoded value. Paper H1 charges this event
  2δ_fold.

### 2.3 Digit sizing

The existing policies get a target parameter (carried seam S13), so there is
no new policy variant. `HonestFoldSizingQuery` gains a
`failure_target: FoldFailureTarget` field:

```rust
pub enum FoldFailureTarget {
    /// Today's sizing for a prover that retries: each policy's own per-attempt constant.
    PerAttempt,
    /// One shot: failure probability ≤ 2^−(security_bits) / masked_folds for this fold.
    OneShot { security_bits: u32, masked_folds: u32 },
}
```

Each policy reads the target:

- **`BalancedSignedDigitFoldPolicy`.** `PerAttempt` keeps 1/8. `OneShot`
  replaces the union term `ln(2N/(1−p))` with
  `L = ⌈(⌈log₂ 2N⌉ + security_bits + ⌈log₂ masked_folds⌉) · 71/100⌉`, in
  checked integer arithmetic (`fold_linf_cap.rs`). This upper-bounds
  `ln(2N) + (security_bits + log₂ masked_folds)·ln 2`, since
  `ln x ≤ ⌈log₂ x⌉ · ln 2` and `ln 2 < 0.71`. N counts every physical
  response coefficient (all witness windows), not the per-window count the
  per-attempt rule uses. The cap is `t* = ⌈√(2σ²L)⌉` with
  `σ² = blocks · η² · ‖w‖∞²`, capped by the deterministic envelope, and
  (L+1)² covers both balanced ends (Lemma `wo:lem:fold-overflow`).
- **`UnitOneHotFoldPolicy`.** This is the one-hot root that Jolt fp128 uses.
  `PerAttempt` keeps 7/8. `OneShot` replaces 7/8 with
  `2^−security_bits / masked_folds` in the same Chernoff solve
  (`honest_fold_policy.rs:195-215`); `2^security_bits` is built exactly from
  its exponent bits. The policy falls back to the deterministic cap when the
  solve does not reach the target. At that cap δ = 0.
- **Modeled caps.** A masked level never takes one: the planner's
  `LevelMasking::Masked` drops the modeled ℓ∞ cap (a 1/40 per-attempt cap is
  never a one-shot bound) and never emits an ℓ2 route, which carries a
  per-attempt ℓ2 cap.
- **Planner.** `akita_planner::find_schedule_for_test_zk` (test-support + zk)
  plans masked levels with the `OneShot` target, hiding columns at their
  rank fixed point, and the ℓ∞ route, and stores the cut in the schedule.
- **Admission.** `ZkOpeningSchedule::admit(schedule, root_source,
  claim_extension_degree)` is the one check, run in release at catalog
  admission (`TrustedScheduleCatalog::try_new_with_test_zk`, which takes the
  root source and claim degree from the configuration) and again where the
  backend first samples hiding randomness (commit). It requires
  `E::DEGREE = 1`, a cut before the last nonterminal fold, no setup prefix
  at the level after the cut (no Stage 3 at the cut), and d_A = d_B for the
  residual commitment. At every level ≤ `cut_level` it requires the ℓ∞
  route, the quotient relation, the open basis b = 8 for every group, one
  witness chunk, and the exact D randomness width of §3.2; for every group
  the level folds it requires at least the `OneShot` policy's fold digits
  and the B randomness width of §3.5 (exact for witness groups, zero for a
  setup prefix). Later levels carry no randomness in any group. The
  planner's masked path skips every candidate admission would refuse. A
  trusted catalog rejects any row with a cut, or with randomness, with
  distinct errors.

The parameters are `security_bits = 136` and
`masked_folds = (cut_level + 1) · W_ℓ`, where W_ℓ counts the groups that
carry witness data at level ℓ (§3.5). A single-group level has
`masked_folds = 2`, so δ_fold ≤ 2^−136 in total by construction.

**Challenge families at the masked levels.** The paper assumes the unfiltered
d = 64 family (η² = 71) at folds ≤ the cut. In Akita:

| Row | Level 0 | Level 1 |
|---|---|---|
| headline shape, `RecursiveCommitmentConfig<fp128::OneHot>` 36:1 | d_a = 512, packing in the 64-subring, (31 ±1, 10 ±2), η² = 71 | d_a = 512, evaluation trace, (19 ±1), η² = 19 |
| test ZK row, `fp128::Dense` 14:1 from the planner | d_a = 512, packing in the 64-subring, (31 ±1, 10 ±2), η² = 71 | d_a = 256, packing in the 128-subring, (31 ±1), η² = 31 |

Neither level-1 family is the paper's. All are unfiltered: the operator-norm
rejection exists only on ℓ2 routes (§2.5.0), and admission forbids those at
masked levels, so the lemma's 1/p factor stays 1 and the one-shot sizing uses
each family's own η².

**F1 finding.** Clear rows are sized per attempt, and most are one digit short
of the one-shot target at a masked level. Among single-group fp128 rows with
more than two levels:

- one-hot (non-recursive): 15:4, 18:1, 20:1, 20:2, 28:1, 30:1, 32:1, 40:1,
  44:1 and 50:1 are short at level 0; 20:4, 32:4 and 36:1 at level 1; only
  16:2 passes;
- dense: every row (14:1, 15:2, 16:1, 16:2, 17:4, 24:1, 26:1, 28:1, 30:1,
  32:1) is short at level 0;
- the recursive headline shape 36:1 passes at both levels. Its root one-hot
  thresholds, 564 per attempt and 1469 one-shot, both fit the scheduled four
  base-8 digits (positive reach 1755).

So ZK rows come from the planner, never from a clear row.

**Paper deltas.**

- *Cut storage and binding.* The paper's k is a protocol parameter. Akita
  stores the cut in the schedule (`FoldSchedule.zk`, a `ZkOpeningSchedule`).
  It is serialized and appended to the schedule's descriptor bytes (tag 0x5a,
  then the cut) only when present, so clear artifacts, row digests and
  descriptors are unchanged. The effective-schedule digest in `PlanSection`
  binds the cut, and the grinding-plan digest binds its `nonce_bits = 0`
  runs; until S9's `SetupSection` mode bit lands, these digests are the only
  thing that separates the modes (no proof tag). Argument: the verifier
  derives both from the admitted row, so a proof for one mode fails
  Fiat–Shamir in the other.
- *One-shot sizing.* The union term is bounded with `⌈log₂⌉·0.71` instead of
  `ln`, over all physical coefficients, so the digit range is at least the
  paper's. Argument: every replacement over-approximates the lemma's tail, so
  δ_fold only shrinks.
- *Masked levels are ℓ∞ only, with no modeled cap.* The paper's masked folds
  are ℓ∞ (§6.2) but the code would otherwise allow an ℓ2 route, whose
  ‖z‖₂² is sent in the clear and whose cap and operator-norm family are
  sized for a retrying prover: the rejection rate there is a
  witness-dependent constant, not δ_fold. Admission and the planner exclude
  it until one-shot ℓ2 sizing and a hidden energy check exist (Q8).
- *Preceding groups at masked levels.* Superseded by §3.5: every group a
  masked level folds is one-shot sized, precommitted groups are hiding, and a
  setup prefix stays public.
- *Quotient relation at masked levels.* Admission and the planner exclude
  reduced evaluation at levels ≤ the cut: the masked stage checks of §4.4 are
  written for the quotient relation, and with the cut fixed at 1 no test
  reaches a reduced-evaluation masked level (W1's cut-2 test is gone).
  Allowing it needs those checks and a test at a cut that reaches it.
- *Base 8 and base-field claims at masked levels.* §4.2's rows assume b = 8
  and E = F; admission now enforces both instead of assuming them.
- *Challenge families.* The masked levels use the families tabled above, not
  the paper's d = 64 family everywhere; the sizing uses each family's η².
- *Error terms.* The theorem's terms are unchanged. δ_fold is now
  ≤ 2^−137 per masked fold by construction (checked for the headline level-1
  shape in exact rational arithmetic), so 2δ_fold ≤ 2^−135. The planner
  prices masked fold responses at the clear nonce width, an overcount of at
  most two bytes per level that affects ranking only.

The extra digits are applied before `rounded_up_role_a_inf_norm`, so A is
re-priced at the enlarged range (Lemma `wo:lem:randomized-base`). The schedule
audit recomputes the A bound from `num_digits_fold`
(`akita-schedules/src/audit.rs:318-350`). The paper's 2^−950 figure is per
fold (Q5, answered).

### 2.4 Acceptance

- **W2-a.** A ZK plan is the clear plan of the same folds except for a
  `FoldResponse` run with `nonce_bits = 0` at every level ≤ `cut_level`. The
  prover's argument string does not grow at those sites, and a nonce inserted
  there is rejected. Clear plan and proof digests are unchanged (golden
  test).
- **W2-b.** On the planner's test ZK row, a backend that rejects the
  response at a masked level makes `prove` return `FoldOverflow { level }`
  after exactly one probe there. Without the rejection, the same proof, which
  also carries the W1 hiding randomness, verifies.
- **W2-c.** At the headline level-1 shape, the one-shot digit range meets
  `2N·exp(−reach²/2σ²) ≤ 2^−137`, checked in exact rational arithmetic with
  `ln 2 < 6932/10000`. The balanced cap is pinned as the least integer
  meeting the closed form above, with one and with two witness windows, and
  the headline root's one-hot thresholds are frozen (564, 1469). No test
  asserts the total ZK bound: it would be arithmetic on the paper's
  constants, and the paper gives no value for Q_tot (Q6).
- **W2-d.** At k = 2, the re-priced A ranks for levels 0 and 1 equal the clear
  ranks, as paper §6.2 states. If they differ, the test fails.

### 2.5 One-shot responses under #255's operator-norm families

Akita PR #255 (`origin/feat/sparse-challenge-stream` at `038024d5d`, protocol
v9, not merged into `whiteout/dev`) draws each fold group's challenges from
one squeezed message. For the two operator-norm families, a message can run
out of accepted candidates, and the honest prover then moves to the next
fold-response nonce. This section answers three questions:

- whether that retry can run inside a ZK schedule;
- what it costs;
- what W2 must change for it.

References are to `038024d5d` unless prefixed with `w2:`, which means
`~/akita-wo-w2` on `wo/akita-w2`. **[V]** marks what was read in code or the
paper; **[I]** marks inference.

#### 2.5.0 Scope, and a blocker before the retry

**[V]** Operator-norm families occur only on L2 security routes. The
family's rejection is set exactly when the level has a `response_l2_sq_cap`
(`akita-params/src/schedule.rs:204-215`;
`layout/params/precommitted.rs:516-521`). L2 fold grinding requires exactly
one scalar group (`fold_grind.rs:439-443`). So every operator-norm level has
G = 1. #255's single-group loop already decodes the message before probing
the response, and the retry question reduces to "skip on run-out, abort on
response rejection".

**[V] Blocker.** An L2 route sends the response energy ‖z‖₂² as a clear
16-byte integer:

- the prover exchanges it in `l2_prefix` (`akita-types/src/l2_sites.rs:39-50`);
- the verifier compares it with the cap
  (`akita-verifier/src/stages/physical_l2_norm.rs:163`).

At a masked level, z is the next witness (paper §4.1, "the fold responses are
never revealed"). Its energy is a function of the witness, and no hybrid of
`wo:thm:zk` simulates it. **An L2 route at a level ≤ `cut_level` is
therefore not zero knowledge, independent of any retry.** Admitting one
needs two pieces:

- the L2 check rewritten so that ‖z‖₂² is never revealed, for example a
  range proof on cap − ‖z‖₂² inside the masked transcript;
- the ℓ2 sizing of §2.5.3.

That redesign is outside this note (Q8).

**Can ZK planning emit one?**

- The headline masked levels are ℓ∞ with the unfiltered family. This is a
  paper claim (§6.2, "folds before fold 5"), not checked in code.
- The ZK planner does not yet exist (A26), and which catalog levels use L2 is
  undecoded (A25). So nothing yet rules an L2 masked level out.
- **Admission rule, until Q8 and the ℓ2 sizing land:** `ZkOpeningSchedule::new`
  rejects `InnerCommitSecurityRoute::L2` at every level ≤ `cut_level`. This
  also excludes every operator-norm family there.

The rest of this section is the argument and design for the retry, for use
once that blocker is removed. It is not needed for RP4 as scoped.

#### 2.5.1 The retry trigger under #255

Order of one fold, prover and verifier alike **[V]**:

1. Fold-response nonce n: sent as a LEB128 prover message
   (`replay.rs:378`; verifier `replay.rs:566-572`, bounded by `nonce_bits`).
2. Fold proof of work at `FoldChallengeWork`: the least grind nonce below
   2^(g+7) (`replay.rs:394-402`, preview `:451-452`; verifier `:578-583`).
3. The group's public payload is absorbed, one message of fixed length is
   squeezed, and it is decoded (`fold_draw.rs:157-200`). The decoder returns
   `None` exactly when fewer than W of the B candidates pass "rank in range
   and certified operator-norm predicate" (`stream.rs:233-249`, `:268-297`).
   The verifier maps `None` to `InvalidProof` (`fold_draw.rs:333`).

The honest loop `first_jointly_accepted_nonce` (`fold_grind.rs:191-204`)
moves to the next nonce on either of two events:

- (E) the message runs out (`:240-247`);
- (R) the response probe rejects (`:287`).

`admit_fold_response` rejects (R) in two cases:

- (R∞) a coefficient leaves the digit range
  (`akita-cpu-backend/src/opaque/fold.rs:415-420`);
- (R₂) ‖z‖₂² exceeds `response_l2_sq_cap` (`fold.rs:421-448`). The cap is
  set on every L2 route (`fold_grind.rs:434-446`).

What this means for zero knowledge:

- **The trigger is mixed [V].** One loop retries on E ∨ R∞ ∨ R₂. Both R
  events read the witness.
- **E alone is public [V].** Whether message m runs out depends only on m,
  through its ranks and the predicate table (`stream.rs:14-24`). The message
  is a squeeze of the sponge at state σ‖n‖w:
  - σ is the state after the public prefix, which the verifier also
    computes;
  - w is the least grind nonce with a public property.

  So E(n) = E(σ, n, H) is a deterministic function of σ, n and the oracle
  H.
- **What the verifier sees [V].** It sees the accepted n and the grind nonce
  w. Rejected candidates leave no bytes: previews run on detached sponges
  (`replay.rs:411-454`).

**The protocol needs no change [I].** The nonce precedes the challenges, and
the response is computed after them, so the nonce is a challenge-candidate
counter. It carries witness information only because the prover skips on R.
The change is on the prover side: in ZK mode, skip only on E, and turn R
into `FoldOverflow`. No byte, domain or verifier rule changes.

#### 2.5.2 The public retry search, and the zero-knowledge lemma

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
±2, with η² = a + 4b and κ₁ = a + 2b. A ⊆ S is the predicate-accepted set:
the challenges whose rank is in range and that pass the strict operator-norm
predicate, with ‖c‖_op < Γ = 18. Let p := |A|/|S|. The certificates give:

| Family | \|A\| | \|S\| | p |
|---|---|---|---|
| D64 (31, 11) | ≥ 2^128.062 | 2^130.152 | ≥ 2^−2.090 ≈ 0.2349 |
| D128 (31, 0) | ≥ 2^128.563 | 2^129.621 | ≥ 2^−1.058 ≈ 0.4802 |

Sources: `akita-challenges/src/config.rs:20-23`, `stream.rs:60-71` **[V]**.
The rates 0.1305 and 0.3694 quoted in #255's spec are weaker lower bounds,
and remain valid if substituted.

**Lemma 2.5a (public retry is simulatable).** Let Π' be Whiteout in which
each masked fold ℓ runs `PublicRetry` instead of §2.2's counter-0 probe, and
in which ‖z‖₂² is not revealed (§2.5.0). Let:

- δ'_ℓ = δ'_ℓ,∞ + δ'_ℓ,2 bound the probability of R∞ ∪ R₂ at n*, given the
  history (Lemma 2.5b and §2.5.3);
- ΔQ bound the oracle queries that the retries add to the honest
  algorithms;
- h_σ be a lower bound on the min-entropy of σ given everything the
  distinguisher has seen or queried before the fold.

Then Theorem `wo:thm:zk` holds for Π' with error

```text
2δ'_fold + δ_LHL + T_att·ε/M + (T_att·(Q_tot + ΔQ) + T_att²)/2^ν + ε_fresh,
δ'_fold = Σ_{ℓ ≤ cut_level} δ'_ℓ,   ε_fresh = Σ_{ℓ ≤ cut_level} N·Q_tot/2^{h_σ}.
```

The simulator runs the lines of `PublicRetry` marked public on its simulated
σ.

*Proof sketch.* Follow the hybrids of `wo:thm:zk`.

- **Determinism.** The values (n*, w*, C) are F(σ, H) for a deterministic F
  that the verifier could evaluate. The only witness-dependent action at the
  fold is the probe, and it affects the view only through the abort.
- **Freshness (new, also implicit in the paper's H1 at N = 1).** Let Fresh
  be the event that none of the points σ‖n, for n < N, was queried before
  the fold, by the distinguisher or by an earlier honest step.
  - σ absorbs commitments and masked messages that carry fresh pads and
    commitment randomness. Given the distinguisher's view, σ is therefore
    unpredictable with min-entropy at least h_σ, and Pr[¬Fresh] ≤
    N·Q_tot/2^{h_σ}.
  - On Fresh, the oracle answers at σ‖n are uniform and independent of the
    fold inputs x and of the history. Messages at distinct n are independent,
    and whether a message runs out does not depend on its accepted values
    (`stream.rs:19-22`; spec `sparse-challenge-stream.md:233-238`).
  - So, given x and the run-outs at n < n*, the W challenges at n* are i.i.d.
    uniform on A.
  - Without this step, a distinguisher that chose the witness after querying
    H could correlate x with the challenges.
- **H1.** Replace the real prover by P', which at n* continues with a fixed
  public vector instead of aborting on R. Because E is decided before the
  probe, P' reaches the same n*. On Fresh, the views differ only on R at n*,
  which has probability at most δ'_ℓ. The same event bounds the failure of
  ct(Q') = 0 in H6.
- **H2.** H2 programs H only at the Σ inputs x_i, which come after the cut
  and lie in their own domain. The retry queries are honest queries, counted
  in Q_tot + ΔQ. No fold point is programmed, so F is evaluated on the same
  oracle in every hybrid.
- **H3 to H9.** Each step replaces a value of the prefix. (n*, w*, C) are F
  of already-replaced values and of H, so they are the same public
  post-processing in neighbouring hybrids, and no step's distance grows. H7's
  premise still holds, with n* and w* among the earlier public values.
- **Final hybrid.** The simulator evaluates F on its simulated σ, including
  the `PoolExhausted` outcome. That outcome is the same public function, so
  the simulator outputs ⊥ with the same probability.

Summing the losses proves the bound. □

**Contrast with the forbidden retry.** Retrying on R makes n* depend on the
witness. The accepted C is then also conditioned on the witness, and P'
would reach a different n*, which breaks H1. `PublicRetry` keeps R one-shot:
one probe, then an abort and a whole-proof rerun with fresh pads and
randomness (§2.2).

**Lemma 2.5b (ℓ∞ overflow under A).** Fix the fold's inputs x_1..x_ℓ, where
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
  independent uniform signs: the rank reads position bits only, and the low
  w bits are free signs (`stream.rs:7-12`). Then condition on the event "all
  ℓ challenges lie in A", which has probability p^ℓ.
- **δ_B.** Fix one response coefficient, Z = Σ_i Z_i, where Z_i depends only
  on c_i, so the Z_i are independent.
  - A is closed under negation, because the shell and the operator norm are,
    so E Z_i = 0.
  - |Z_i| ≤ κ₁β.
  - E_A[Z_i²] ≤ E_S[Z_i²]/p ≤ η²β²/p.
  - Bernstein's inequality, the integrality of Z, and a union bound over the
    coefficients and both ends finish the proof. □

**Cost [I].** δ_H adds ℓ·log₂(1/p) bits to the target exponent: 2.09 bits
per challenge for D64 and 1.06 for D128. Since L grows like the square root
of the exponent, this is cheap for small W. δ_B inflates the variance by
1/p, which raises L by about ½·log₂(1/p): +1.05 bits for D64 and +0.53 for
D128. This holds while the linear term stays small. It wins for large W and
costs at most one more radix-16 digit.

If G > 1 ever carried an operator-norm family (today `fold_grind.rs:439-443`
forbids it), #255's interleaved loop would probe group g before decoding
group g+1. `PublicRetry` must then decode every group before any probe;
otherwise δ'_ℓ needs a union over every counter at which a probe ran.

#### 2.5.3 Sizing the ℓ2 cap for one shot

**[V]** The cap `response_l2_sq_cap` is modeled for a retrying prover. It is
the planner's mean energy times an envelope and a mean multiplier
(`akita-planner/src/response_model.rs:186-195`; the
`response_model.rs:37` comment reasons about 4096 attempts). One-shot
rejection at that cap has constant probability, and that probability depends
on the witness. Masked levels must therefore never use the modeled ℓ2 cap,
exactly as §2.3 forbids the modeled ℓ∞ cap.

There are two candidate one-shot caps. Both are computed from public,
certified per-input bounds ‖x_i‖₂ ≤ X_i, taken from the certified digit
alphabets and the randomness columns, not from a modeled mean.

- **Worst case.** ‖Σ_i c_i x_i‖₂ ≤ Σ_i ‖c_i‖_op·‖x_i‖₂ < Γ·Σ_i X_i. Set
  cap = ⌈Γ·Σ X_i⌉². Then δ'_2 = 0.
- **One-shot tail (McDiarmid).** ‖z‖₂ is a function of independent c_i, and
  replacing c_i changes it by at most 2Γ X_i. Also E‖z‖₂ ≤ √(Σ E_A‖c_i x_i‖²)
  ≤ η·√(Σ X_i²/p). Hence

  ```text
  Pr[‖z‖₂ ≥ η√(ΣX_i²/p) + t] ≤ exp(−t²/(2Γ²ΣX_i²)).
  ```

  At δ'_2 = 2^−137, t = Γ·√(2·137·ln 2)·√(ΣX_i²) ≈ 248·√(ΣX_i²).

**Choice: worst case [I].**

- It removes δ'_2 entirely, and it needs no independence or freshness
  premise.
- With equal X_i, its radius 18·√ℓ·X is below the tail radius
  (8.66/√p + 248)·X ≈ 266·X whenever ℓ ≤ 218. L2 routes have G = 1, and
  their W is the live-block count times the claim count. Whether production
  L2 levels exceed 218 is unverified.
- If one does, the admission check rejects it rather than switching bounds.
  This keeps one sizing rule.

The worst-case cap is larger than the modeled cap by about Γ·√ℓ/(η·√m), for
envelope multiplier m. It must be re-priced (§2.5.5).

#### 2.5.4 Error budget

- **δ'_fold.** The `OneShot` target stays ≤ 2^−136/2 per masked fold:
  - δ'_∞ is reached with Lemma 2.5b's digits;
  - δ'_2 = 0 at the worst-case cap.

  So 2δ'_fold ≤ 2^−135, as before. The accepted challenge law has no bias
  term: on Fresh and given that the message does not run out, it is exactly
  uniform on A, which is the law both lemmas use.
- **T_att.** Unchanged: it counts Σ attempts.
- **ΔQ.** Per masked fold and candidate, ΔQ counts the grind search
  (< 2^(g+7)) plus the squeeze (≤ 2^8 permutations). Counting permutations
  over-counts oracle queries. With N = 2^12, two masked folds and g ≤ 10:

  ```text
  ΔQ ≤ 2·2^12·(2^17 + 2^8) < 2^30.1
  T_att·ΔQ/2^ν ≈ 2^−218.5
  ```
- **ε_fresh.** With h_σ ≥ 264, the 2λ + 8 bits of one commitment slice
  (paper §6.2), at Q_tot = 2^100 (Q6):

  ```text
  ε_fresh ≤ 2·2^12·2^100/2^264 = 2^−151
  ```
- **Pool exhaustion.** This is completeness only and not a ZK term, because
  the simulator reproduces it exactly.
  - The pool is sized from measured rates lowered for sizing
    (`stream.rs:44-56`), giving about 2^−161 per message at those rates.
  - At the weak certified rate 0.1305 for D64, a message runs out with
    probability about 2^−50.8 at W = 1, 2^−44 at W = 16, and ≈ 1 at
    W ≥ 256. These figures are from the independent review, not recomputed.
  - So the retry count and `PoolExhausted` are bounded only heuristically.
    #255's "at most 2^−32" claim does not hold at that rate for large W.

The total stays below 2^−128, with Q_tot from Q6. W2-c adds ΔQ, ε_fresh, and
Lemma 2.5b's δ'_∞ for operator-norm levels.

#### 2.5.5 Soundness

The verifier is #255's, unchanged **[V]**:

- it reads the nonce below 2^`nonce_bits`;
- it checks the fold proof of work;
- it rejects a message that runs out (`fold_draw.rs:333`);
- it rejects ‖z‖₂² above the cap (`physical_l2_norm.rs:163`), or the
  replacement check of Q8.

The prover's search policy is not part of the protocol. A cheating prover
gains no accepting branch relative to clear Akita at that site. Relative to
W2's `nonce_bits = 0`, it has up to 2^`nonce_bits` candidates, each priced by
the fold proof of work (W·2^128 ≤ 2^K·2^g). So the query-counted bound of
`wo:lem:randomized-base` is unchanged **[I]**. Unfiltered masked levels keep
`nonce_bits = 0`.

Two parameters must be re-priced, both by the lemma's own rule ("re-checked
for each schedule"):

- **ℓ∞.** The enlarged digit ranges re-price A's coefficientwise collision
  radius, as in §2.3.
- **ℓ2 (new).** On an L2 route, A's Module-SIS instance is priced at an ℓ2
  collision radius derived from the cap. Two accepted responses differ by
  at most 2√cap. The worst-case cap of §2.5.3 enlarges that radius, so the
  schedule audit must re-run the L2 SIS estimate for A at the new cap and
  raise A's rank if needed. Paper §6.2 has no such row, because its masked
  levels are ℓ∞. Which audit function prices the L2 radius was not located
  (A31).

#### 2.5.6 Implementation (after #255 and Q8)

- **`FoldResponseSearch`** (`w2:akita-params/src/transcript_grinding.rs:377-385`)
  gains `PublicRetry`: try counters in order, skipping one only when its
  message runs out; probe once; a rejection is `FoldOverflow`.
  - `ZkOpeningSchedule::fold_response_search(level)` is the single owner:
    - `OneShot` at masked levels with an unfiltered family;
    - `PublicRetry` at masked operator-norm levels;
    - `Grind` elsewhere.
  - The constructor checks that `OneShot` has `nonce_bits = 0` and
    `PublicRetry` has `FOLD_RESPONSE_NONCE_BITS`. The existing derivation from
    `nonce_bits` alone (`w2:…:497-508`) cannot tell `PublicRetry` from
    `Grind`.
- **Prover** (`w2:fold_grind.rs:215-231`, `accepted_fold_response`). The
  closure is split into two:
  - `draw(n) -> Option<PublicFoldCandidate>`: preview, work and decode. It
    takes no opening or witness handle, so witness independence holds by
    construction.
  - `probe(&PublicFoldCandidate) -> Option<T>`.

  `PublicRetry` skips on `draw = None` and returns `FoldOverflow { level }`
  on `probe = None`. `Grind`'s behavior and bytes are unchanged. The terminal
  path is untouched, since masked levels are nonterminal (`w2:zk.rs:194-215`).
- **Sizing.** `MaskedFoldSizing` (`w2:zk.rs:194-246`) carries only
  `SparseChallengeConfig`. It gains:
  - the `OperatorNormRejection`;
  - p, from the certificate exponents, with one owner in `akita-challenges`;
  - the worst-case ℓ2 cap from §2.5.3.

  ZK planning sets the modeled ℓ2 cap to `None` at masked levels and uses
  the worst-case cap instead.
- **Admission (now).** Reject L2 routes at levels ≤ `cut_level` (§2.5.0).
  Lift the rejection only together with Q8's hidden L2 check, the sizing
  above, and the ℓ2 re-pricing of §2.5.5.
- **Wire.** No new message or domain. A masked operator-norm level sends the
  12-bit LEB128 nonce (1–2 bytes). ZK plan digests change; clear digests do
  not (C0).
- **Errors.** `PoolExhausted` becomes a typed variant in place of #255's
  `InvalidInput("fold grind exceeded …")`. W6 may rerun on it. It is not
  charged to δ_fold.
- **Port of W2.** #255 removes `Fork::new(seed, nonce)`. The one-shot branch
  (`w2:replay.rs:443-445`, `:578-583`) becomes: no nonce message, then the
  fold proof of work, then the squeeze. `FoldPreview::new` must not absorb a
  counter at `nonce_bits = 0`.

#### 2.5.7 Tests

- **W2-e, admission (now).** `ZkOpeningSchedule::new` rejects a test schedule
  with an L2 route at level 1.
- **W2-f, a retry happens.** #255's `FixedDraw` double
  (`fold_grind.rs:480`): counter 0 runs out and counter 1 decodes. Assert
  `PublicRetry` returns n = 1, and calls `probe` exactly once, at n = 1.
  Integration test, after Q8: a test-only pool override with run-out ≈ 1/2,
  then prove, verify, and read n = 1 from the event log.
- **W2-g, witness independence.** On one fixed preview sponge state, run
  `PublicRetry` with probe closures for two different in-range witnesses.
  Assert equal n*, w* and message bytes. A third witness that overflows ℓ∞
  and a fourth that exceeds the ℓ2 cap each return `FoldOverflow`, with the
  same number of `draw` calls.
- **W2-h, overflow still aborts.** A successful draw at 0 followed by a
  rejecting probe gives `FoldOverflow`, one probe, and no `draw(1)`. Run-out
  at 0, success at 1 and a rejecting probe give `FoldOverflow`, not n = 2.
- **W2-i, budget.** W2-c's exact-rational check gains:
  - min(δ_H, δ_B) at D64, p = 2^−2.090;
  - the worst-case ℓ2 cap compared with Γ·ΣX_i;
  - the ΔQ and ε_fresh terms.

## 3. W1: outer hiding randomness per slice

### 3.1 What is sampled and where it goes

In the table, "made at level ℓ" means the level whose prover computes the
image, and "output of level ℓ" is the witness committed by level ℓ's
`NextWitnessPayload` (the paper's w_{ℓ+1}).

| Commitment | Image | Randomness | Joins | Alphabet |
|---|---|---|---|---|
| root (Jolt's packed polynomial), made at commit time | each B slice | `r_root,s` | output of level 0, beside the root's T digits | open basis of level 0 |
| D image made at level ℓ ≤ `cut_level` (`OpeningPayload`) | one per level | `r'_ℓ` | output of level ℓ | open basis of level ℓ |
| output of level ℓ < `cut_level` (`NextWitnessPayload`) | each B slice | `r_ℓ+1,s` | output of level ℓ+1, beside its T digits | open basis of level ℓ+1 |
| output of `cut_level`, the residual commitment u_a | one slice | `r_a` | **s**, not a further witness | radix 16 (W4 §1.5) |

Level ℓ's relation, over its output witness, checks the B rows of its input
commitment and its own D rows (`akita-params/src/layout/relation_rows.rs:139`).
Each randomness vector therefore sits where its image is already checked.

The setup-prefix slot keeps zero randomness: it is public setup data. A
precommitted group folded at a masked level carries randomness like the root
(§3.5): Akita cannot tell a per-proof group (untrusted advice, `FieldInc`)
from a statement-fixed one (trusted advice, a committed program), so it hides
all of them.

### 3.2 Sizing, one owner

`akita_params::zk::hiding_randomness_columns(image: ImageShape, alphabet: DigitAlphabet)`
is the only caller of
`jolt_crypto::lattice::required_randomness_columns(image_coeffs, 0, ring_dim, log_basis, 8)`.

- `ImageShape` is `{ rows, ring_dim }` of the outer map, per slice for B and
  per level for D.
- `DigitAlphabet` is the open basis of the witness that holds the randomness.

No other site passes literal arguments. The helper is called by the planner
when it builds a ZK schedule, and by `ZkOpeningSchedule::admit` in release, which
recomputes the width for every slice and D image before the cut and rejects
any mismatch. At d_b = 64, rank 1 and log_basis 3, one slice needs 2,819
coefficients, which is 45 R_64 columns.

Randomness is sampled with `jolt_crypto::lattice::sample_box_digits` from the
prover's CSPRNG. The box has unit differences, since b ≤ 64 < q^{1/4}/2.

**Norms.** The randomness digits use the open-basis alphabet, which is the one
alphabet Akita's range check proves (`ring_switch.rs:123,198`;
`stage1.rs:238-239`). So s_inf does not change, and the randomness is
range-checked because it lies inside `live_coeff_len`. What changes is length:
the block count B and the coefficient count N of the next level grow. The
planner sizes t* and the one-shot term on the widened layout, and re-prices the
B and D ranks at the widened input width. The paper reports rank 1 kept at
k = 2; W1-d tests that.

### 3.3 Geometry, layout, and code paths

- **Width.** `CommitmentSliceGeometry` (`akita-params/src/commitment_slicing.rs:178`,
  `try_new` at `:202`) gains `randomness_columns` (carried seam S2).
  `physical_input_width` becomes message width plus randomness width. Every
  width-equality site reads it:
  - `layout/params/commitment.rs:107`, `precommitted.rs:417`, `builder.rs:112`;
  - `akita-schedules/src/audit.rs:295`, `setup_contribution/plan/prepare.rs:444`;
  - `schedule/profiles.rs:259`, `setup_envelope.rs:540`, `setup_prefix_slots.rs:518`;
  - `akita-planner/src/schedule_params/candidate.rs:114`, `relation_weights/compiler.rs:416`.

  `commitment_polynomial_count` (`commitment.rs:133`) divides by the message
  width. `OpenCommitMatrixParams` gains the same field.
- **Views.** Widening re-indexes every row after row 0
  (`flat_matrix.rs:94-106`), so B and D are cut at the widened width, and
  setup sizing (`setup_envelope.rs:44,84,115`, `setup_projection.rs:37`) reads
  it. A zero width leaves every view byte-identical.
- **Layout.** `WitnessLayout` (`akita-params/src/witness.rs:97`) gains a
  `randomness` tail section after the compression spans (carried seam S3), so
  existing offsets keep their values. It holds the B randomness per group, in
  relation order and slice-major, then the D randomness. In clear schedules its
  length is 0. The section must be counted in both `tail::resolve` sinks, in
  `grouped_witness_body_coefficients`, `chunk_shape` and
  `validate_internal_ranges`, and in the classification in
  `finalize.rs:287-332`.
- **Nonzero-randomness code paths** (carried seam S12, inert at width 0):
  - `commit_outer_slices` (`arithmetic/commitment_stage.rs:88`) appends each
    slice's randomness;
  - `compute_relation_d_rows` (`d_rows.rs:31`) appends r′;
  - the quotient recomputation (`relation_quotient.rs:625-740`) uses the
    widened maps;
  - `scatter_et` adds B′ and D′ setup columns;
  - the verifier's `SetupContributionPlan` (`physical_b.rs`,
    `prepare.rs:61-106`, `segments.rs`) adds them as column offsets of the one
    widened view.

  The randomness is carried in `RingRelationGroupWitness` and
  `RingRelationWitness`, and `emit_witness_tail` writes it.
- **`commit_hiding`** lives in the `zk` module of `akita-pcs`, as
  `impl HidingCommitment for AkitaCommitmentScheme<ZkCommitmentConfig<_>>`. It
  returns the root randomness in the opening hint (interfaces §3), and it is
  not a `main` API.
- **Kernels.** Akita's commitment kernels run on secret digits in ZK mode.
  RP4 does not make them constant time (A24).

### 3.4 Acceptance

- **W1-a.** At width 0, C0 holds (golden test).
- **W1-b.** A clear-protocol proof whose schedule has nonzero test randomness
  widths verifies, using the test-only constructor. Two hiding commitments to
  one polynomial differ.
- **W1-c.** Flipping one randomness digit in a witness breaks that level's
  relation (prover-side test path).
- **W1-d.** For every ZK schedule, `ZkOpeningSchedule::admit` accepts the planner's
  widths and ranks. A frozen-shape test pins the headline: rank 1 at levels 0
  and 1, and 45 columns per slice.

### 3.5 Groups at masked levels (W2b)

A masked level folds one or more groups (`CommittedGroupParams::groups`, in
canonical order): an optional setup prefix, the root's precommitted groups,
and the level's own group. All of their responses share the level's one
probe, and all of their D segments lie in the level's one D image. This
section decides each kind. Recursive levels have no precommitted groups, and
the root has no setup prefix.

| Group | Source | Commitment | Opening evaluation | Response z_g | B randomness | Fold sizing |
|---|---|---|---|---|---|---|
| own, level 0 | witness | `commit` on a ZK row, hiding (W1) | Jolt's masked claim | next witness | `b_randomness_columns` | root source contract |
| own, level ≥ 1 | witness | `NextWitnessPayload`, hiding (W1) | masked `w_eval` | next witness | `b_randomness_columns` | universal balanced, bounded digits |
| precommitted (root only) | witness (untrusted advice, `FieldInc`) or statement-fixed (trusted advice, program) | caller's `commit` with `GroupContext::explicit`, before the opening | Jolt's masked claim | next witness | `b_randomness_columns`, always | universal balanced, bounded digits, no contract |
| setup prefix (level ≥ 1; level 1 in the headline) | public setup slot | setup-derived, public | public: the slot's MLE at a public point (§4.2) | next witness; a function of public data | 0 | universal balanced, bounded digits |

Let W_ℓ be the number of groups at level ℓ that carry witness data: every
group but a setup prefix (`zk::witness_group_count`). W_0 = 1 + the number of
precommitted groups, and W_ℓ = 1 above the root.

**Hiding (W1 extended).**

- *Precommitted groups.* Each one's B slice images depend on its source, so
  each slice gets `b_randomness_columns(group)` columns, the same helper and
  alphabet as the root. The width is a field of the frozen profile
  (`GroupCommitPhaseParams::randomness_columns`), so the caller fixes it when
  it picks the profile, from a ZK row of the precommit's shape. The commit
  path is the existing one: `CpuBackend::commit` with
  `GroupContext::explicit(&profile)` samples fresh randomness from the
  backend's hiding source whenever the profile's width is nonzero, and keeps
  it in the commitment handle. An explicit profile does not say which opening
  consumes it, so the alphabet is fixed by admission instead: a hiding
  profile is admissible only at a masked level, and every group there opens
  at b = 8, so the randomness is drawn from `MASKED_LOG_BASIS`
  (`akita-cpu-backend/src/commitment/api.rs`, `resolve_commit_params`). At the
  opening, the handle's randomness is placed in level 0's output witness in
  the group's `RandomnessOwner::Outer` span, which W1 already lays out per
  group in relation order. W6's `HidingCommitment` for a precommitted group
  is therefore this call on a hiding profile; no new Akita API is needed.
- *Statement-fixed precommitted groups.* The paper lets a commitment fixed
  in preprocessing (trusted advice, a committed program) carry no randomness,
  with zero knowledge relative to it (paper §3.1). Akita cannot see
  provenance: "precommitted" is a layout notion. Admission therefore
  requires the full width for every precommitted group at a masked level.
  Jolt makes those preprocessing commitments hiding too, and keeps their
  randomness with the prover's preprocessing. W6 therefore produces ZK
  preprocessing separately from clear preprocessing: a hiding profile comes
  from a ZK row, a clear catalog rejects it, and a clear profile fails ZK
  admission, so one preprocessed commitment never serves both modes. Admitting an exempt group
  would need a typed per-group mode bound in the row; it is not built.
- *Setup prefix.* Zero randomness. Its commitment is setup-derived and known
  to the verifier.

**Public data in the setup prefix.** The prefix polynomial is a slot of the
public setup. Its evaluation claim is the slot's MLE at a point drawn from
public challenges (Stage 3 of the previous level, §4.3). Its response
z_S = Σ_b c_b · ŝ_b is a function of the setup and of the level's public fold
challenges, so it, and whether it overflows its digit range, can be
computed by the simulator. Its D segment lies in the level's D image, which
r′ hides as a whole (`d_randomness_columns` depends only on the D matrix
shape). In the next witness z_S is covered by that witness's commitment
randomness; at the cut it sits in w_k as a public segment (§4.6). No message
of a masked level depends on witness data through the setup prefix.

**Sizing and the union bound.** Each group's one-shot digit range is sized
for `fold_failure_target(W_ℓ)`:
`OneShot { security_bits: 136, masked_folds: (cut_level + 1) · W_ℓ }`. Each
masked level gets an equal share of 2^−136, split equally among its witness
groups, so the union over all witness responses is
Σ_ℓ W_ℓ · 2^−136 / ((cut_level + 1) · W_ℓ) = 2^−136 = δ_fold. A setup prefix
is sized for the same per-response target, but it is not counted in W_ℓ: its
overflow is a public event, so it costs completeness (at most 2^−137 per
masked level) but not zero knowledge. Single-group levels keep
`masked_folds = 2`, so every existing ZK row is unchanged.

The sizing inputs (`MaskedFoldSizing::for_group`) are per group:

- the root's own group: the root source contract, as before;
- a recursive own group: the universal balanced policy over
  `FoldWitnessNorms::bounded(log_basis_inner, d_A)`, as before;
- a precommitted group or a setup prefix: the universal balanced policy over
  the group's own inner digit alphabet, with its claim count, blocks, inner
  width, challenge family and open basis.

A schedule row does not record a precommitted group's producer contract
(`akita-prover/src/opening.rs` checks only the final group's). The universal
bound needs none: every committed digit lies in the balanced alphabet of its
inner basis, so ‖w‖∞ ≤ 2^(log_basis_inner − 1) for every producer, and the
sub-Gaussian tail of the response holds for every witness with that bound.
For a balanced-digit producer this equals or exceeds the contract's own
sizing (its reach per plane is at most the alphabet's); for a one-hot
producer it is looser than the Chernoff solve. Jolt's precommitted groups are
dense, so the cost is nil there. The planner sizes masked precommits the same
way, so admission and planning agree.

**Several root groups.** Each root group is a witness group with its own B
randomness and its own one-shot sizing; the one D image and its r′ cover all
of them; every group opens at b = 8, the alphabet its randomness is drawn
from. The frozen A bound of a precommitted profile must cover its one-shot
digits at the larger `masked_folds`; the planner skips a profile whose bound
or randomness width does not fit.

**Zero-knowledge argument, per case.**

- *Precommitted witness group.* Its B images are hidden by Lemma
  `wo:lem:stat-hiding` exactly as the root's: the same width, fresh uniform
  digits per commit from the alphabet the next range check proves. Its
  response, D segment and evaluation are handled by the hybrids that already
  cover the root: the response sits in the next witness, the D segment in
  the hidden D image, and the evaluation is one of Jolt's masked claims. Its
  overflow event counts in δ_fold. A commitment reused across proofs is still
  one commitment: its randomness enters each proof only inside a witness that
  is itself committed with fresh randomness.
- *Statement-fixed precommitted group.* Hidden as above, which is stronger
  than the paper requires.
- *Setup prefix.* Every value it contributes is public or inside a hidden
  image (above); its overflow is simulatable.
- *Several root groups.* The hybrids apply per image; δ_fold is the union
  bound above.

**Admission** (`ZkOpeningSchedule::admit`), per masked level: every group
opens at b = 8; every group has at least `MaskedFoldSizing::for_group`'s
digits for `fold_failure_target(W_ℓ)`; a setup prefix has zero B randomness
and every other group exactly `b_randomness_columns`. It still rejects a
setup prefix at the level after the cut. At unmasked levels no group carries
randomness.

**Planner.** `LevelMasking::Masked` carries the level's one-shot target. A
masked level may consume a setup prefix (`allows_setup_prefix` is false only
after the cut), whose search sizes the prefix for that target with no
modeled cap. A masked root folds precommitted groups, sized as above, and
skips a frozen profile whose randomness width is not
`b_randomness_columns`. `find_schedule_for_test_zk` takes the precommitted
source contracts, which it still needs for equivalence classes and the
clear planner.

**Headline.** For `RecursiveCommitmentConfig<fp128::OneHot>` at 36:1 the ZK
planner offloads level 0's setup to a level-1 setup prefix (7 digits, no
randomness) and the row passes admission.

**Paper deltas.**

- *Multi-group union bound.* The paper's δ_fold sums over one response per
  masked fold. Akita splits each fold's share among its witness groups and
  counts a setup prefix's public overflow as a completeness term only.
  Argument: union bound over witness responses; a public event is
  simulatable.
- *Precommitted groups are always hiding.* Stricter than paper §3.1, which
  exempts preprocessing commitments; Akita cannot see provenance. Closes jolt
  D24.
- *Contract-free sizing for preceding groups.* The universal bound replaces
  the producer contract, which no row records. It only widens digit ranges,
  so δ_fold only shrinks.
- *The setup group's response at a masked level.* The paper adds the setup
  group to the next fold (`04_protocol.tex:493-505`) and lists offloading
  claims among public messages (`03_setting.tex:48-51`) but does not say that
  the setup group's response joins a masked witness. It is a function of
  public data and is covered as above.

**Tests (W2b).** Admission accepts the planner's row with a precommitted
group and a level-1 setup prefix, and rejects each violation: a precommitted
group with no or wrong randomness, a setup prefix with randomness, short fold
digits in a preceding group, a preceding group opened at a basis other than
8, randomness on any group after the cut, and a setup prefix after the cut.
An e2e ZK proof over that row verifies and rejects a wrong opening. Two
explicit commits of one source under a hiding precommitted profile differ.

## 4. W3: the masked levels and the cut

### 4.1 The cut

`cut_level = 1` (k = 2). `ZkOpeningSchedule::admit` enforces Def.
`wo:def:source-cutoff` in release:

- every level ≤ `cut_level` uses `InnerCommitSecurityRoute::Linf`, so no norm
  is checked in the clear;
- `cut_level` has no Stage 3 (level 0 may offload to level 1, as in the
  headline);
- the successor plan at `cut_level` is the residual commitment, never
  `NextWitnessPlan::TerminalT`;
- `E::DEGREE == 1`, else `AkitaError::Unsupported`;
- the residual commitment has d_A = d_B (A7).

There is no planning for k ∈ {3, 4}.

`ZkOpeningSchedule { prefix, cut: ResidualCutParams }`. `prefix` is the root
plus `recursive_folds[..cut_level]`, with no terminal. `ResidualCutParams` is:

- `cut_level`;
- inner A_a and outer [B_a | B′_a], 3 × R_256 each;
- one slice;
- `r_a` columns from the §3.2 helper (97 R_256);
- payload mode `Raw` (u_a is 768 coefficients, above the 512-coefficient
  chain cap; compression is W5c's contract, §5.4).

Akita's SIS tables price the matrices at the Gaussian radius (rule 3).

### 4.2 Transcript of a masked level

| Message | Clear | ZK |
|---|---|---|
| `RootCommitment`, `RootPoint` | `public_all` | unchanged (hiding commitment, public point) |
| `Openings` (level 0 statement claims) | `public_all` | `bind_derived_base` per claim. The verifier holds them as Jolt affine forms (§4.6) |
| `OpeningPayload` (D image or its compressed form) | `exchange_private_base` | `exchange_all`, public, hidden by r′ (M3b fixup) |
| fold seed, challenges | nonce + fork | fork(seed, 0) (W2) |
| `NextWitnessPayload` (at `cut_level`: u_a) | public | public, hidden by r |
| ring switch α, τ₀, τ₁ | public | public |
| stage-1 rounds, `stage1_range_image` | `exchange_private` | masked |
| stage-2 rounds, `stage2_w_eval` | `exchange_private` | masked |
| Stage 3 (level 0 only): claim, rounds, prefix eval | `exchange_private` | public and verified in the clear (M3b fixup) |
| level-1 scalar openings: predecessor `w_eval` and setup-prefix eval | `public_all` | `w_eval` through `bind_derived_base`; the setup-prefix eval stays `public_all` |

These rows assume the Jolt fp128 shape: E = F, b = 8 (no stage-1 product
stages), and the Linf route.

**Why no new jolt method.** `bind_derived_all` is not needed. Spongefish's
`public_messages` is a loop of `public_message`
(`spongefish/src/narg_prover.rs:242-246` at `ef97413`; `:212-216` at
`d2d190b`), so per-element `bind_derived_base` absorbs the same sponge
bytes as one `public_all`. What changes is jolt-transcript's event log: one
`Public` record becomes several. A frozen event-log fixture, if any, is
regenerated in the M3b fixup (A29).

### 4.3 Answers to the open questions

- **Extension-field products.** Jolt has none (E = F). An extension-field ZK
  schedule is rejected with `Unsupported`.
- **`response_l2_sq`.** It falls after the cut: it is sent only on L2-route
  levels (`akita-types/src/l2_sites.rs:26-46`), and ZK schedules forbid L2 at
  or before `cut_level`.
- **Stage-3 claims.** They are public. σ_S = Σ_p S(p)·ω(p) depends only on the
  public setup and on τ₁, r_x and α. The non-deferred verifier computes it
  from those inputs (`stage2.rs:270-272`), and the prefix evaluation is the MLE
  of a public slot. The stage-2 relation weight
  `… + common_alpha_evaluation() * setup_claim`
  (`akita-verifier/src/relation/evaluation.rs:88-90`) then multiplies w only by
  a public value.
- **`NextWitnessInnerState`.** It falls after the cut: only the last level
  before the terminal sends it (`prove/fold/mod.rs:546-576`). Scalar openings
  are handled as in §4.2.

### 4.4 One formula per check: `ClaimDomain`

`ClaimDomain` moves from jolt-zk to jolt-transcript's `private` module (fixup
to M3a for the trait and the value domain, and to M6b for jolt-zk's domains; §7). There it sits next to `PrivateChannel` and `PadSlots`, and it
gains an error type and two operations:

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
    /// x·y = target, with no new slot: the product lands on an existing affine value.
    fn product_into(&mut self, x: Self::Value, y: Self::Value, target: Self::Value)
        -> Result<(), Self::Error>;
    /// value = 0.
    fn check_zero(&mut self, value: Self::Value) -> Result<(), Self::Error>;
}
```

The domains behave as follows:

| Domain | `check_zero` | `product_into` |
|---|---|---|
| clear value domain (in jolt-transcript) | compare, typed error | compare x·y with target |
| jolt-zk `RowsDomain` | push an affine row | push a product row with z = target |
| `OperandShapeDomain` | no-op | record x's pad as an operand, no slot |
| `ValuesDomain` (prover auxiliary values) | as the value domain | as the value domain |

The stage-1 leaf check and the stage-2 oracle check at every level are written
once, generic over `D: ClaimDomain<F>`. This is a fixup to M2b, where the stage
verifiers already moved onto jolt-sumcheck. The clear verifier uses the value
domain, so there is no parallel masked formula and no Akita copy of the
trait. The checks live in akita-types so the prover can run them in
`ValuesDomain`. Akita's normal dependencies reach only jolt-transcript, which
it already uses. jolt-zk is a dev-dependency for the masked-domain tests
until the `zk` feature lands; the extracted pin must contain M6b.

At b = 8 with the Linf route, the paper's lowering gives these rows per
masked level:

| Check | Clear formula | Lowering | Products | Aux |
|---|---|---|---|---|
| stage-1 leaf (`stage1.rs:186-190`) | claim = Σ_{k≤4} c_k·ri^k | `u = product(ri, ri)`; `product_into(u, c₄u + c₃ri + c₂, claim − c₁ri − c₀)` | 2 | 1 |
| stage-2 oracle (`stage2.rs:248-335`) | claim = w·(B(w+1) + R) | `product_into(w, B(w+1) + R, claim)` | 1 | 0 |

In the stage-2 row, B collects both w(w+1) sites: the virtual term
`batch·eq·w(w+1)` (`stage2.rs:329`) and the binary term `bb·binW·w(w+1)`
(`:457-475`). R collects every w × public term: relation weight, packing,
trace and the setup claim. Both B and R are public.

That is 3 products and 1 auxiliary slot per masked level, so 6 and 2 at
k = 2. These are the paper's counts (§6.1), and Q3 is answered.

The cut row is `check_zero(Witness(0) − w_eval)`, with
`ResidualVar::Witness(0)` at the stage-2 point r₂ of `cut_level`. r₂ is
converted once with `match_endianness` (`LOW_TO_HIGH` → `HIGH_TO_LOW`; A18, I4).

### 4.5 Masked sumchecks, channels, and the verifier

- **Rounds.**
  - `verify_single` folds masked coefficients into a claim value and drops
    `PadSlots`, so it cannot serve masked mode.
  - jolt-sumcheck gains `receive_masked_single(instance, context: &mut impl VerifierRoundContext)`
    (fixup to M2a). It returns `MaskedRounds { rounds: Vec<MaskedRound>, challenges }`
    and draws each challenge through the context, so Akita's per-round
    `grinded_ext_challenge` works unchanged.
  - Akita lowers the result with `jolt_zk::masking::sumcheck_rows`, with
    `RoundFormat::Compressed` for stage 2 and `RoundFormat::EqFactored`
    (fixup to M6b) for stage 1, inside the `RowsDomain` used for §4.4's
    checks. The returned running claim is the `claim` in those formulas.
- **Channels.**
  - `GrindingReplay::State` becomes generic: `ProverGrinding<'a, H, C>` and
    `VerifierGrinding<'a, 'p, H, C>` over `C: Channel<Sponge = H> + PrivateChannel`
    with `C::Error: Into<AkitaError>`. This is a fixup to M2b. `C` defaults
    to the clear transcript so existing call sites stay unchanged.
  - In masked mode, C is the prover adapter over `CommittedPads::send`
    (fixup to M6b) or jolt-zk's `MaskedVerifier`.
  - Jolt's prover lends `&mut CommittedPads` to the opening for levels
    0..=`cut_level`.
- **Structure.** The masked verifier is today's stage verifiers instantiated
  with the rows domain and the masked rounds. No separate recorded-check list
  exists. It returns
  `MaskedPrefix<F> { relation_rows, cut: CutState }` (`impl ResidualRows<F>`),
  where the rows are the ones the domain pushed.
- **Prover.** The prover keeps every clear self-check, by running the same
  formulas in the value domain on decoded values. It computes Akita's
  auxiliary values in `ValuesDomain`. W6 sends them after Jolt's auxiliary
  values, in slot order.
- **Counting (N8).** A replay over zero bytes fails the proof-of-work
  predicates (`grinded_ext_challenge`, ring switch). So the pad layout uses a
  third `GrindingReplay` implementation, `CountingGrinding`, whose grind
  queries return without checking a predicate and whose state is
  `CommittedPadsView::counting()`'s channel. It is a separate type that
  produces only a pad count and an operand list, never a verdict. The masked
  verifier code therefore stays the single owner of the schedule, and
  `pad_layout` (W6) runs Jolt's stages and the Akita prefix through it.

### 4.6 The hand-off and `ResidualOpening`

In ZK mode the Akita root takes its statement claims as affine forms,
`MaskedOpeningClaims<F> = Vec<AffineRow<F>>`, built by W6 from Jolt's masked
stage-8 claims. This replaces Dory's stage-8 row
(`jolt:crates/jolt-verifier/src/verifier.rs:219-245`).

At the cut the prover returns
`ResidualOpening { w_k, t_hat_a, r_a, points, public_segments, blocks }`.
That is W4's `Witness` (W4 §4):

- `w_k`: `i64` digits in Akita's flat order, which at d_a = 256 is ring by ring
  and equals W4's native order;
- `points = [r₂]`, `HIGH_TO_LOW`;
- `public_segments`: whole R_256 elements, namely the setup-group response and
  fully public inner digits. They are aligned to 256-coefficient boundaries,
  padding with zero cells (A3, A27);
- `blocks`: `ResidualBlock`s for A_a with gadget t̂, and [B_a|B′_a] on
  (t̂, r_a), cut from Akita's setup stream.

The verifier receives raw u_a.

### 4.7 Acceptance

- **W3-a.** On random witnesses at test shapes, the decoded transcript and pads
  satisfy every emitted row.
- **W3-b, coverage gate.** Every pad slot of the Akita prefix has a nonzero
  coefficient in some row or in a lowered round.
- **W3-c.** A test-only probe shifts each decoded message by +1 and recomputes
  the auxiliary values consistently; at least one row breaks.
- **W3-d.** `CountingGrinding` counts the same pads and operands as an honest
  masked run, on zero bytes and on random bytes.
- **W3-e.**
  - `ResidualOpening` decodes to s with `N·s = t` under jolt-zk's
    `SigmaSetup`, for Akita's own u_a.
  - A dense `w̃_k(r₂)` equals the decoded `w_eval`, which pins A4/A18.
  - Akita's commitment kernels agree with `TwoLevelLayout` semantics.
- **W3-f.** C0 (golden test) after the M3b and M2b fixups and the S-seams.
- **W3-g.** `ZkOpeningSchedule::admit` rejects each §4.1 violation.
- **After W2b.** W2b (§3.5) rebases onto W3c and makes two more shapes
  admissible at masked levels: multi-group masked roots, whose
  `MaskedOpeningClaims` then hold one form per claim of every group in group
  order, and Stage 3 at masked level 0. W3c's masked path accepts the first
  unchanged, because the root binds and batches its openings through
  `ClaimReplay` per claim. For the second, Stage 3's rounds must go through
  the clear sumcheck path, since its claim, rounds and prefix evaluation are
  public (§4.2). Today the driver sends them as private messages, and a
  masked replay fails closed with `MaskedRoundInClearReplay`.

### 4.8 W3c as built: deviations from the paper and from §4.4–4.6

W3c (`wo/akita-w3c`) runs levels 0..=`cut_level` masked and stops at the cut.
Each departure below states its argument.

- **Checks are recorded, then lowered (§4.5 "no separate recorded-check
  list").** The rows domain needs the masked auxiliary values, which W6
  sends after the cut (§7, step 6), so the masked verifier cannot push rows
  during its replay. The fold verifier is written once over a `ClaimReplay`
  trait (`akita-verifier/src/claims.rs`). The clear instance checks each
  claim on the spot; the masked instance (`MaskedReplay`) holds each claim as
  an `AffineRow` and records each level's two stage checks as data: the
  leaf, the Stage-2 weights, and the claim forms. `MaskedPrefix::lower` then
  runs the one formula in `akita-types` (`check_range_leaf`,
  `check_stage2_oracle`) in any `ClaimDomain`: `RowsDomain` (`rows`),
  `OperandShapeDomain` (`shape`) and `ValuesDomain` (`auxiliary_values`). No
  formula is restated, and the three domains allocate the same slots
  because they share one lowering routine. The paper's verifier holds the
  same data.
- **The packing-sum check needs no row.** The clear verifier compares the
  per-group authenticated sum with the batched trace target. Both are formed
  from the same openings with public coefficients. The masked replay
  therefore compares the two forms coefficient by coefficient, which is a
  public check. It is a tautology for honest layouts, and no mutation of
  the transcript can make it fail.
- **The cut row and the hand-off.** One value, `ResidualClaim`, carries the
  hand-off: the point, the masked w̃_k(r₂), and u_a as this verifier received
  it. The cut row `ResidualClaim::WITNESS − w̃_k(r₂) = 0` is built from it,
  and the residual back end opens that commitment at that point, so the row,
  the point and the commitment cannot be wired apart. The point has one
  owner, `akita_types::residual_point`, which both roles call. It converts r₂
  once from `LOW_TO_HIGH` to `HIGH_TO_LOW` through jolt-poly's
  `match_endianness`. When w_k is longer than 2^|r₂| it pads with leading
  zeros to ⌈log₂ |w_k|⌉ variables. A dense evaluation of the exported w_k,
  written from the MLE's definition with the first coordinate as the most
  significant bit, equals the decoded w_eval, and the reversed point does
  not. This pins A18.
- **Counts.** At b = 8 on the ℓ∞ route, each masked level lowers into three
  product pairs (`u = ri²`, the leaf row, the Stage-2 row), one auxiliary
  slot and three operand pads (ri, u, w_eval). The test row at k = 2 has 6
  products, 2 auxiliary slots, 6 operand pads and one affine row, the cut
  row. These are the paper's counts (§6.1).
- **The prover's auxiliary values.** The prover lowers the leaf check over
  its own decoded values (`ProveLevelOutput::range_leaf`), with
  `check_range_leaf` in `ValuesDomain`, and runs the same check in the clear
  domain as a release self-check. The test compares the result with
  `MaskedPrefix::auxiliary_values` under the pads. The Stage-2 check takes
  no auxiliary slot, so the prover does not lower it.
- **Statement claims.** The masked root takes `MaskedOpeningClaims` (one
  `AffineRow` per claim). The `GroupBatchStatement` supplies only the
  commitments, points and selection, and its evaluations are not read. The
  prover checks in release builds that each form decodes, under the pads, to
  its own claim.
- **One backend.** `prove_masked_prefix` proves every masked level on one
  backend and bypasses the registry's dyn executors. Those executors stay
  typed on the clear transcript. Routing masked levels across backends
  needs the executor trait to become generic over the channel.
- **Counting replay.** `CountingGrinding` consumes proof-of-work plan
  entries without reading a nonce, and forks every fold response at counter
  0. W6's `ShapeTranscript` makes the same choice for Jolt's grinding. The
  replay runs over zero bytes, or a reader that never fails, long enough for
  the prefix's messages; never over proof bytes. The layout is a function
  of the public schedule, and a proof can end early or fail to decode. The
  test schedule grinds at both masked levels, so the masked channels' nonces
  and the counting replay's nonce skip both run.
- **Stopping at the cut.** Both roles check
  `finish_through(cut_level)`: the replay consumed every grinding-plan entry
  of levels 0..=`cut_level` and nothing after. The schedule's plan and
  descriptor still cover the clear levels after the cut.
- **`ResidualOpening`.** It carries:
  - the cut level and the point;
  - the decoded w̃_k(point);
  - raw u_a;
  - w_k's digits with their alphabet (the cut level's open basis);
  - t̂_a with its gadget basis;
  - r_a with its alphabet.

  It is built by the `ResidualExportKernel::export_residual` backend kernel
  (§4.8.1). `public_segments` and `blocks` are not built (§4.8.1).
- **Composition with Jolt's rows.** `MaskedPrefix::rows` numbers auxiliary
  slots from `layout.auxiliary.start`, as `RowsDomain::new` does. Appending
  it after Jolt's `JoltResidualRows` in one relation needs an offset
  constructor for `RowsDomain` (W6).
- **Channel transcript access (M6b fixup).** The masked channels do not
  lend their transcript. They offer typed proof-of-work operations only:
  `MaskedProver::{grind, send_nonce, narg_len}` and
  `MaskedVerifier::{check_grind, receive_nonce, remaining}`. Akita's
  `ProverGrindingChannel` and `VerifierGrindingChannel` are these operations.
  Their terminal byte sends (`send_public_bytes`,
  `send_bounded_public_bytes`) return `UnsupportedSchedule` on a masked
  channel, and public field messages go through `Channel::exchange_all`.
- **Error classification.** Under `zk`, a masked channel's error that
  crosses the sumcheck driver (`SumcheckError::PrivateChannel`) maps through
  `From<ZkError>`. A truncated or malformed masked round is therefore
  `AkitaError::InvalidProof`, not `Internal`.
  `CommittedPads::uncommitted`, behind jolt-zk's `test-support` feature, builds
  pads with no commitment for Akita's tests. Until W6, only a `ZkBackend`
  builds `CommittedPads`.
- **Not exercised.** The tests cover k = 2 at b = 8 on `fp128::Dense`
  14:1. They do not exercise product stages, extension-field claims or
  Stage 3 at a masked level, and admission rejects all three. The masked
  replay also fails closed on each. Product stages and Stage 3 run their
  rounds through `verify_single`, whose rounds arrive masked over a masked
  channel and fail with `MaskedRoundInClearReplay` (§4.2 makes Stage 3's
  claim and prefix evaluation public, but the sumcheck driver still sends its
  rounds as private messages). Extension claims fail with
  `UnsupportedSchedule`.

#### 4.8.1 Hiding the residual commitment

u_a, the cut level's commitment to its output witness w_k, has no hiding
randomness today: `admit` gives every level after the cut zero B
randomness, and the cut successor is the after-cut level's own group. The
paper commits to w_k "in the same way" as every earlier witness
(04_protocol.tex:81). That is the gap closed here. The design composes with
§3.5 (precommitted groups, W2b): §3.5 changes only levels ≤ `cut_level`, and
this note changes only the cut successor.

**Where r_a comes from.** The successor commitment that the cut level
makes already takes its randomness width from the successor's own-group
profile (`randomness_columns`). It takes its alphabet from the successor's
open basis (`akita-cpu-backend/src/commitment/plan.rs:225`,
`with_randomness_log_basis(params.own_group().opening.log_basis_open)`). The
backend samples r_a there and keeps it in the commitment material, as W1
does for every masked successor. No new sampling path is needed. In ZK mode
the after-cut level is never folded: its parameters describe the residual
commitment (A_a, [B_a | B′_a]) and nothing else. Its "consuming opening" is
the Σ-protocol, so its open basis is defined as the Σ alphabet of r_a.

**Column count.** `r_a` = `b_randomness_columns(after_cut.own_group())`. This
is the existing §3.2 helper, `hiding_randomness_columns(rows = outer rank,
ring_dim = d_B, modulus, alphabet = after_cut.open().log_basis)`, with the
after-cut open basis fixed at `RESIDUAL_RANDOMNESS_LOG_BASIS = 4` (radix 16,
the alphabet W4 sized for). Over R_256 this gives 97 columns at rank 3 (the
existing `residual_commitment_needs_ninety_seven_columns` test) and 161 at
rank 5 (W4 §1.5's table, which uses the same formula and slack). The count
is per outer slice. The cut successor is required to have one slice (A28),
because W4's single [B_a | B′_a] block assumes one.

**`admit` change** (only these lines; the §3.5 per-group loop is
untouched). At level `cut_level + 1`:

- the own group carries exactly `b_randomness_columns(own)` B columns;
  previously it carried zero;
- the own group opens at `RESIDUAL_RANDOMNESS_LOG_BASIS`;
- its outer slice count is 1;
- the level's D randomness stays 0, because the level is never folded;
- there is still no preceding group (no Stage 3 at the cut, as before).

Levels above `cut_level + 1` keep zero randomness everywhere. The order
relative to the §3.5 merge is mechanical: both edit the
"later levels carry no randomness" branch of `admit`. Whichever lands
second splits that branch into `cut_level + 1` (this rule) and above (zero).
The ZK planner must then fix the after-cut open basis at 16 and recompute
the after-cut level's witness geometry with the r_a span. The clear planner
and clear rows do not change (C0).

**Where r_a enters `ResidualOpening`.** The full hand-off becomes:

```text
ResidualOpening {
    cut_level, point: [r₂] (HIGH_TO_LOW), witness_eval, residual_commitment: u_a (raw),
    w_k: Vec<i64>,            // the cut level's output witness, Akita flat order
    w_k_log_basis: u32,       // its digit alphabet (b = 8 at masked levels)
    t_hat_a: Vec<i64>,        // gadget digits of A_a·w_k, at the after-cut inner basis
    r_a: Zeroizing<Vec<i8>>,  // b_randomness_columns(after_cut)·d_B digits
    r_a_log_basis: u32,       // = after_cut.open().log_basis
    public_segments,          // §4.6: setup-prefix responses at the cut level (§3.5)
    blocks,                   // §4.6: A_a with gadget, [B_a | B′_a] cut from the setup stream
}
```

W6 builds the Σ segment table from the declared alphabets: `Witness` at
`w_k_log_basis`, `InnerDigits` at the inner basis, and `ResidualRandomness` at
`r_a_log_basis`. The alphabet has one owner, the schedule's after-cut open
basis, and Σ reads it. Still missing, in order:

1. A backend kernel, `export_residual(session, witness, material)`, that
   returns w_k's logical digits, t̂_a and r_a from the cut successor's
   handles before the proof scope finishes. On the CPU backend these are
   the export packet's `LogicalDigits`, the gadget decomposition of its
   `InnerRows`, and the retained outer randomness.
2. `blocks`: views of A_a and [B_a | B′_a] over the setup stream, where B′_a
   is the outer matrix's randomness-column part.
3. `public_segments`: the setup-prefix response spans of the cut level's
   witness layout, aligned to R_256 boundaries (A27).

**Effect on W3-e.**

- *N·s = t* (as designed). With r_a sampled and exported, the outer rows
  [B_a | B′_a](t̂_a, r_a) = u_a hold for Akita's own u_a. W3-e can then
  check `N·s = t` under jolt-zk's `SigmaSetup`, built from `blocks`. Without
  this change the check would pass with r_a = 0 and prove nothing about
  hiding. The test should also check that two proofs of one statement give
  different u_a.
- *MLE order (A18).* Unchanged. r_a is a separate Σ segment after t̂_a and
  is not part of w_k, so `Witness(0) = w̃_k(r₂)` ranges over w_k's index
  space only. The dense-evaluation test still needs `export_residual`'s
  w_k. It pins Akita's flat order against the `LOW_TO_HIGH` → `HIGH_TO_LOW`
  conversion that `CutState` and `ResidualOpening` already apply.

**Zero knowledge.** u_a's raw image is hidden by Lemma `wo:lem:stat-hiding`
at the same width rule (entropy ≥ image bits + 2λ + 8). That makes the
response proof's binary decompositions of u_a simulatable, as the paper
requires ("the full outer image must be hidden", 04_protocol.tex:68-72).

**As built.**
- The `admit` conditions are implemented, each with its own typed error and
  a negative test. One condition is added beyond the design: the cut
  successor's payload is `Raw` ("sends the residual commitment compressed").
  §4.1 already fixes the payload mode as Raw, and the Σ-protocol opens the
  outer image itself.
- The ZK planner fixes the after-cut open basis at 16, the slice count at
  1 and the payload at Raw. It sizes r_a through the same priced-randomness
  path as the masked levels (`LevelMasking::hides_outer`). The test row's
  residual ring dimension is 64.
- `export_residual` is implemented for the CPU backend.
- Tests:
  - W3-e `N·s = t`, through jolt-crypto's `MatrixView` and
    `PreparedMatrix::apply_centered` on Akita's setup stream: the inner rows
    `A·G⁻¹(w_k) = G·t̂_a` and the outer rows
    `[B | B′](t̂_a, r_a) = u_a` hold for Akita's own u_a;
  - two openings of one statement send different u_a and r_a;
  - the dense w̃_k(point) equals w_eval (A18).
- `public_segments` and `blocks` remain for after the precommit merge (§3.5
  decides the setup-prefix spans) and W6, which builds the
  `ResidualBlock` views from the same setup stream the test reads.

**Paper delta.** The paper puts the randomness of the commitment to w_j
into w_{j+1}, "in the alphabet of the witness that later contains it". At
the cut no later Akita witness exists. r_a joins the Σ vector s as its own
segment, in the Σ alphabet (radix 16), and the Σ segment alphabet check
certifies it (W4 §2.1, release check 1) instead of a range sumcheck. The
entropy rule is unchanged, so the hiding argument is the paper's. The
binding argument for [B_a | B′_a] is the paper's weak-opening argument at
the Gaussian radius, which already covers every segment of s.

## 5. W5: `ResponseProver` for Akita

### 5.1 Scope

`akita_pcs::zk::AkitaResponseProver<Cfg>` implements `ResponseProver<F, 256>`
for jolt-zk's statement as implemented: items 1–5, raw t_p and t_D, and raw
u_a in N's targets. Ownership follows W4 §3.1.

e is jolt's `ResponseWitness.digits`, zero-padded to 2^ν. Every digit is in
[−8, 7].

### 5.2 Transcript

The response proof follows the paper's §4.3 order:

1. Commit to e, an ordinary Akita commitment without randomness.
2. Draw α. Send `t_×`, run the degree-2 product sumcheck, then send `u_×`
   and `w_×`.
3. Draw η and form `A_j` and `b_η` from items 2, 3 and 5, the three product
   claims, and the padding rows.
4. Run Akita's digit-range reduction at b = 16: the stage-1 tree with arities
   [2], then the leaf, reused as a standalone reduction. It yields s_rng at τ.
5. Draw γ and run the degree-3 fused sumcheck. Send v and check
   `A(r)v + B(r)v(v+1)`. With no compression there is no binary constraint
   until W5c.
6. Open E(r) = v with `batched_prove` under the response schedule. That
   schedule is a dense bounded-digit family at log_basis 4, with recursive
   setup planning and ordinary clear grinding.

The new kernels implement the jolt-sumcheck prover trait, and the new messages
use spongefish codecs.

### 5.3 Weights

`A(r)` is evaluated from typed descriptions read through `ResponseStatement`'s
accessors, as W4 §3.2 and paper §4.3 describe: slot selections, radix powers,
the eq weights of `Witness(j)` claims, and negacyclic convolution by c, by the
matrices and by L. The verifier never builds `H`. Its setup part,
σ_N = Σ_p S(p)·ω_N(p), uses the prefix views of A_a, [B_a|B′_a], A_p and L,
and A_D.

- **W5a** computes σ_N directly: the verifier scans those views.
- **W5b** offloads σ_N into the response opening root's Stage 3. It is sent as
  a public claim, batched by a fresh λ, with a `ResponseRelation` projection
  in `active_setup_projection_geometry`. W5b is an optimization PR. It lands
  in RP4 if the direct scan exceeds the verifier budget measured in W5-e, and
  is deferred otherwise (A30).

### 5.4 Compression (W5c)

Compression of u_a, t_p and t_D lands in RP4, together with its jolt-side
hook. The images are already masked, so compressing them changes no hiding
argument. Raw images cost about 25 KB at the headline, against the paper's
3 × 128 B. W5a ships them raw; W5c then establishes the following:

- The verifier holds only compressed forms. In item 3, the targets t (u_a and
  t_p) and, in item 2, t_D become their gadget reconstructions Σ_i 2^i·ξ_i
  (negative-binary digits), affine in e. The ξ digits join e, the binary
  constraint joins the fused sumcheck on `I_bin`, and item 2 gains the chain
  equations of each image.
- Images above 512 coefficients are split into ⌈len/512⌉ chunks, each
  compressed by the existing two-map chain (Q4).
- jolt-zk sends compressed t_p and t_D through a `ResponseProver` hook, so
  I1 holds. The Σ oracle input becomes (σ, cm(t_D), n).

### 5.5 The last PR (W5d): ZK mode on, and the tests

The last PR:

- moves `ZkCommitmentConfig`, the ZK entry points and `AkitaResponseProver`
  from `test-support` to public;
- generates the ZK family artifacts and catalog rows;
- sets `IS_ZK` for `ZkCommitmentConfig`;
- adds the CI job `test-zk` (workspace tests with `--features zk`).

The tests run on an Akita-side end-to-end harness: a synthetic statement
with masked claims, jolt-zk's whiteout Σ-protocol over Akita's `ResidualRows`,
and `AkitaResponseProver`. Three of them gate the PR:

- **HT1, public-site allowlist.** The verifier's event log puts every received
  byte in exactly one class: masked slot, randomized commitment, public
  challenge-derived data (Stage 3 and points), Σ message, or response proof.
  No `exchange_private` reaches a clear channel. The positive control is a
  build where one site is routed clear, which the test must flag.
- **Coverage.** W3-b over the full Jolt-shaped synthetic prefix.
- **HT3, pad uniqueness and freshness.**
  - Every pad index is consumed exactly once, and the cursor ends at
    `num_pads`.
  - Two proofs of one witness with different RNG seeds share no pad, masked
    value, commitment image, randomness digit or Σ nonce.
  - A `FoldOverflow` rerun, forced via the test schedule, reuses nothing
    from the aborted attempt.

The remaining tests:

- **HT4, consistent cheat.** Shift one decoded message m_i, then continue the
  prover honestly from the shifted transcript: the later messages are
  consistent with the shifted value, and the pads and challenges are
  recomputed. The prover's per-row check must fail. With that check bypassed
  through the test path, the response prover's statement check must fail, and
  so must the verifier. One case runs for each masked site class.
- **HT5, clear invariance.** The golden proof digests hold.
- **HT2, two-witness distinguisher (diagnostic, not a gate).** Over N proofs
  for each of two witnesses with equal public input, a per-byte χ²
  homogeneity test over all proof bytes, Bonferroni-corrected, finds no
  difference. It has two injected-leak positive controls: one masked slot with
  pad 0, and one slot whose pad has only 64 random bits. The test must detect
  both at the same N, which calibrates its power. It is a documented manual or
  scheduled diagnostic (`examples/zk_distinguisher.rs`), because CI cost rules
  it out as a per-PR test.

### 5.6 Acceptance (W5a, W5b)

- **W5-a.** `AkitaResponseProver` accepts every statement jolt's
  `CheckingResponseProver` accepts. It rejects each item-1–5 violation that
  `check_witness` rejects, using synthetic `SigmaSetup` statements, so W5a
  does not depend on W3.
- **W5-b.** On small shapes, the verifier's `A(r)` and `B(r)` equal a
  contraction of `H` computed row by row through `check_witness`'s own maps.
- **W5-c (W5b).** The combined Stage 3 verifies, and a tampered σ_N is
  rejected.
- **W5-e.** Proof bytes, prover time and verifier time are reported against
  the paper's model at the headline, including the raw-image surcharge.

### 5.8 Performance plan (W5-e follow-up)

W5a's measured verifier is 3.8 s at the synthetic headline (|e| ≈ 2^27), about
300× the paper's no-offload model; the prover is 12 s, about 10× the paper's
+1.15 s. Sampled shares (verifier): 51% `apply_transposed` over CRT limbs for
Nᵀη; 28% single-threaded `multilinear_eval` of dense A over 2^27; 6% the
constant b computed as `rows(zeros)`; 5.5% `eq_table(η)` over 2^26 for the
padding rows; 5% building A plane by plane.

- **P1, factored A(r) (no layout change).** In segment k, entry
  o_k + p·C_k + 256j + u has weight radix^p·w_k[256j+u]; with C_k and o_k
  multiples of 256, eq(r,·) splits as eq(r_lo8,u)·eq(r_hi,·), so the column
  weights cost O(|e|/256) and the Nᵀ term is one σ_N scan plus one ring
  product per row on F-coefficients. Padding leaves the η batch: A gains
  δ·1[j ≥ len]·eq(τ_pad, j), evaluated by an O(ν) bit-DP (adds ν/q to the
  error; paper delta). b comes from the targets, with no matrix application.
  Estimate ≈ 60–65 ms synthetic, ≈ +12 ms at the paper's geometry.
- **P2, W5b.** After P1, about 80% of the verifier is the σ_N scan; W5b moves
  it into Stage 3 (§5.3). Stage 3's random column point needs ω̃ evaluated
  succinctly, which needs A15 segment alignment (C_k = 2^{c_k}, aligned o_k)
  or an O(P·ν) shifted-eq evaluator. Estimate ≈ 12–15 ms total, response
  overhead ≈ 1–3 ms.
- **Prover.** Top costs: the fused degree-3 sumcheck over three dense 2^27
  tables (32%), the same transposed pass (24%), the digit range (13%).
  Fixes: keep e as i8 in early rounds, split-eq (Gruen) for B, build A from
  ω × radix planes, a direct F ring transpose.
- **Ownership.** jolt-zk gains `ResponseStatement::weight_at` (factored A(r)
  with the σ_N scan inside), `ResponseStatement::constant`, an F-coefficient
  view of `PreparedMatrix`, and for W5b a `SetupContraction` accessor; `weigh`
  stays for the prover and the small-shape W5-b test.

## 6. M6e

M6e changes the following:

- **Deletes `akita-zk-prover` and `akita-zk-verifier` entirely.** That covers
  their crates, the workspace members, `Cargo.lock`, and their
  `scripts/check-crate-deps.sh` calls (`.github/workflows/ci.yml:86-87`;
  `AGENTS.md:34-35`). Their contents moved byte for byte to jolt-zk in M6d,
  and they have no dependents.
- **Removes them from every `forbidden=(…)` list** in
  `scripts/check-crate-deps.sh`.
- **Adds a feature check for `akita-verifier`.** It must resolve `jolt-zk`
  without `whiteout-prover` (`cargo tree -e features -i jolt-zk`).

The M1c/M5c re-export shims are not M6e. They shipped to `main` in RP2, so
deleting them is a `main` change, done after jolt-akita stops naming those
paths.

## 7. jolt prerequisites and W6 needs

### Prerequisites, as fixups to open jolt tickets

RP1–RP3 are not merged upstream, so these are `fixup!` commits:

| Change | Ticket |
|---|---|
| `ProverRoundContext` exposes `type Channel: Channel + PrivateChannel` and `channel_mut()` in place of the concrete `transcript_mut()`; channel errors reach the caller as `SumcheckError::PrivateChannel` | M2a |
| `receive_masked_single(instance, &mut impl VerifierRoundContext) -> MaskedRounds` (§4.5) | M2a |
| prover-side masked `PrivateChannel` adapter over `CommittedPads::send`, with a release check that the message field is the pad field | M6b |
| `RoundFormat::EqFactored { equality_point }` in `sumcheck_rows` | M6b |
| `ClaimDomain` in jolt-transcript `private`, with `Error`, `product_into`, `check_zero`, and the clear value domain | M3a (it must be at RP2's pin, because the M2b fixup uses it) |
| jolt-zk's `RowsDomain`, `OperandShapeDomain` and `ValuesDomain` implement the moved trait; jolt-zk's own copy is deleted | M6b |

There are no prerequisites for `bind_derived_all` (§4.2) or compression (W5c).
`ZkError::FoldOverflow` lands in W6, with its first producer mapping.

### W6 (RP5) needs from RP4

- `ZkCommitmentConfig<RecursiveCommitmentConfig<OneHot>>` and its ZK family.
- `commit_hiding`, behind `HidingCommitment`.
- The ZK opening prover. It takes `&mut CommittedPads` and
  `MaskedOpeningClaims`, and returns `ResidualOpening` and the auxiliary values.
- The masked verifier, returning `MaskedPrefix: ResidualRows`.
- `CountingGrinding`, so `pad_layout` counts Jolt and Akita in one replay.
- `AkitaResponseProver`.
- The `AkitaError::FoldOverflow` → `ZkError::FoldOverflow` mapping. The jolt
  prover returns it, and the caller reruns.
- This transcript order:
  1. preamble, hiding commitment;
  2. t_p, compressed (W5c);
  3. Jolt stages 1–7 masked;
  4. stage-8 claims to Akita;
  5. levels 0..=`cut_level` masked;
  6. the auxiliary block (Jolt's, then Akita's);
  7. ρ, σ, t_D, n;
  8. the response proof.

## 8. PR slicing for RP4

Fixups to tickets RP2 has not yet extracted:

- **M3b fixup** (formerly S4): `OpeningPayload` and Stage 3 go to
  `exchange_all`, and `w_eval` and the statement claims go to
  `bind_derived_base`.
- **M2b fixup** (formerly S5): channel-generic grinding, plus stage-1/2
  checks generic over `ClaimDomain` with the clear value domain.

Both are byte-identical (C0) and land before RP2's extraction.

**Carried `main` seams.** Each is opened against `main` with an
`Extension seam:` comment linking akita#120 and carried into `dev` first:

| Seam | What |
|---|---|
| S1 | run-level `nonce_bits = 0` handling in `commit_fold_response` and `read_fold_response` |
| S2 | `randomness_columns` in `CommitmentSliceGeometry` and `OpenCommitMatrixParams` |
| S3 | zero-length randomness tail section |
| S6 | suffix loop `SuffixEnd::{Terminal, Cut}` |
| S9 | `ProtocolFeatureSet` from config: `from_parts` parameter, `Valid` against the expected set, `IS_ZK` |
| S10 | `CommitmentConfig::zk_schedule_family_name`, per preset |
| S11 | one-probe branch in `replay_multi_group_fold_decompose_witnesses` |
| S12 | nonzero-randomness paths: `commit_outer_slices`, `d_rows`, quotient, `scatter_et`, `SetupContributionPlan` |
| S13 | `HonestFoldSizingQuery::failure_target`, read by each policy |

S7, the `ResponseRelation` projection, is dev-only, in W5b. S8, split
compression, lands with W5c.

| # | PR | Seams | Depends on | Test bar |
|---|---|---|---|---|
| 1 | **M6e**, plus the golden proof-digest test (C0) for a few catalog rows | — | jolt pin with the §7 fixups | builds with and without `zk`; dependency check; golden digests |
| 2 | **W2**, plus the test-only ZK planner path and catalog admission | S1, S11, S13 | 1 | W2-a…d |
| 3 | **W1** (`commit_hiding` in the `zk` module) | S2, S3, S12 | 1 | W1-a…d |
| 4 | **W3**: the cut, `ZkOpeningSchedule::admit`, the masked prover and verifier, `CountingGrinding`, `ResidualOpening` (raw u_a) | S6, S9, S10 | 2, 3, M2b/M3b fixups | W3-a…g |
| 5 | **W5a**: `AkitaResponseProver`, direct σ_N | — | 1 | W5-a, W5-b |
| 6 | **W5b**: σ_N offload (conditional, §5.3) | S7 | 5 | W5-c |
| 7 | **W5d (last)**: publish, artifacts, `IS_ZK`, CI `test-zk` | — | 4, 5 (6 if landed) | HT1, coverage, HT3 (gates); HT4, HT5; W5-e |

PRs 2, 3 and 5 are independent once PR 1 has landed. No PR before 7 makes ZK
selectable.

## 9. Soundness and zero knowledge, mapped to the paper

| Choice | Paper item |
|---|---|
| `nonce_bits = 0` runs; one probe; whole-proof rerun with all randomness fresh | §4.1 "Fold responses"; Lemma `wo:lem:fold-overflow`; H1 |
| L2 routes rejected at masked levels (‖z‖₂² is sent in the clear). After Q8: `PublicRetry` (skip only on pool run-out; ℓ∞ and ℓ2 overflow abort), worst-case ℓ2 cap, ℓ2 SIS re-pricing | Thm `wo:thm:zk` H1 (with freshness), H2 (ΔQ in Q_tot); Lemma `wo:lem:fold-overflow` in tuple or Bernstein form; Lemma `wo:lem:randomized-base` re-pricing (§2.5, Lemmas 2.5a, 2.5b) |
| One-shot targets on every policy (2^−136/2 per fold); modeled cap off before the cut; A re-priced | Lemma `wo:lem:randomized-base`; Thm `wo:thm:zk` (2δ_fold) |
| Randomness per slice, sized by one helper, open alphabet, in the next witness; every precommitted group at a masked level hiding, the setup prefix public (§3.5) | §4.1 "Commitment randomness"; Lemma `wo:lem:stat-hiding`; H9; §3.1 advice (stricter) |
| One-shot sizing per group, `masked_folds = (cut_level + 1) · W_ℓ`; setup-prefix overflow public | Lemma `wo:lem:fold-overflow`; Thm `wo:thm:zk` (2δ_fold, union over witness responses) |
| Residual 3 × R_256, d_A = d_B | §4.1 "The residual commitment"; A7 |
| Stage 3 public; no setup claim from `cut_level` | §3.1; Def. `wo:def:source-cutoff` |
| One `ClaimDomain` formula per check; rows never dropped | Lemma `wo:lem:residual-correct` |
| Mode bit from config in `SetupSection`; ZK schedule in the plan digest | the transcript binds the statement and schedule (Def. `wo:def:algebraic-verifier`) |
| e committed without randomness; clear Akita opening | §4.3; §5.3 |

## 10. Ambiguity register

The numbering continues the W4 note's A1–A23.

- **A24, constant-time Akita kernels.** The commitment kernels run on secret
  digits once per proof. Timing is outside the paper's model, and RP4 does
  not audit these kernels.
- **A25, Jolt catalog levels.** The survey did not decode which levels of the
  current Jolt catalog use the L2 route. The ZK planner makes its own
  schedules, so this affects only the comparison with the paper.
- **A26, small test shapes.** The ZK planner finds an admissible
  `fp128::Dense` 14:1 row; other test shapes are unverified.
- **A27, setup-group alignment.** Aligning the setup group to R_256 boundaries
  adds zero cells, in an unmeasured count.
- **A28, residual slicing.** This note fixes one slice.
- **A29, event-log fixtures.** Moving to per-element `bind_derived_base`
  changes jolt-transcript's event-log grouping but not the sponge bytes.
  Whether Akita freezes an event log anywhere is unverified; the M3b fixup
  regenerates any it finds.
- **A30, W5b.** Whether the direct σ_N scan fits the verifier budget is
  measured in W5-e.
- **A31, certified operator-norm rates and the L2 radius.** The rates are
  resolved: p ≥ 2^−2.090 (D64) and p ≥ 2^−1.058 (D128), from the certificate
  exponents (`akita-challenges/src/config.rs:20-23`, `stream.rs:60-71`).
  Still open: which audit function prices A's L2 collision radius from
  `response_l2_sq_cap`, needed for the ℓ2 re-pricing of §2.5.5, and whether
  any production L2 level has W > 218 (§2.5.3).
- A2, A3 and A12 from W4 are resolved by §4.3, §4.6 and §5.4.

## 11. Questions for the paper authors

- **Q1, pads instead of randomness, for levels below the cut.** The reviewer
  thinks this is plausibly sound. It would need three conditions: the image
  is revealed only through ξ digits inside masked or Gaussian-hidden
  witnesses, the compressed form is sent after the pad commitment, and its 8
  coefficients take pads. The root is an exception. Its commitment is made at
  commit time, before any pad exists, and it may be reused across proofs, so
  it keeps randomness. Does the paper agree? W1 would shrink from about 2.8k
  coefficients per slice to 8 pads per commitment.
- **Q2.** A7 (d_A = d_B), A9 (the box envelope) and A14 (⊥ after T_att) remain
  open from W4. A8 (L) looks fine to the reviewer.
- **Q4, compression above 8 KiB.** The cost script's `FCHAIN_MAX_IMAGE = 884`
  disagrees with Akita's 512-coefficient cap. Should the model count the
  split chain (§5.4)?
- **Q6, Q_tot.** `05_security.tex:316` defines Q_tot, which bounds the
  distinguisher's and the honest algorithms' oracle queries together, but
  §6.2 gives no value. At Q_tot = 2^100 the term is about 2^−148.6. What
  value should the evaluation assume?
- **Q7, wording of `wo:lem:fold-overflow`.** "Conditioned on a public event
  of probability at least p" is correct. Only the example "such as an
  operator-norm filter" misleads: with a per-challenge filter, a fold of ℓ
  challenges conditions on an event of probability p^ℓ. Suggest
  rewording the example. The paper's masked folds are unfiltered, so
  no number changes.
- **Q8, L2 routes at masked levels.** Akita's L2 route sends ‖z‖₂² in the
  clear (`l2_sites.rs:39-50`, checked at `physical_l2_norm.rs:163`), and at a
  masked level z is the next witness. Is the intended design to never use
  L2 before the cut, as §6.2's parameters do and §2.5.0's admission rule
  enforces? Or should the L2 check become a hidden range proof on
  cap − ‖z‖₂²? Only the second makes #255's operator-norm retry
  (§2.5) reachable.
- **Answered.** Q3: 3 products and 1 auxiliary slot per masked level (§4.4).
  Q5: 2^−950 is per fold.

## Not verified by this note

- Akita facts come from reading `87b4ca046` and from the review. No branch of
  this design was built.
- Headline numbers come from the paper and the W4 note: 45 columns per slice,
  97 for r_a, ranks unchanged at k = 2. The ZK planner has not been run. W1-d
  and W2-d test these numbers.
- That the one-hot Chernoff solve reaches 2^−137 within the policy's λ grid
  (`-12..=2`) is unverified. If it does not, the deterministic cap applies,
  with δ = 0 at a larger digit count.
