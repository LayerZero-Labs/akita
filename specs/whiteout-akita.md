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
  rows are never modified (rule 6). Until the last PR they exist only as a
  test-only constructor (`ZkOpeningSchedule::for_tests`, under
  `test-support`).
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
  replaces `ln(2N/(1−p))` with
  `⌈ln(2N) + (security_bits + ⌈log₂ masked_folds⌉)·ln 2⌉`, computed in
  checked integer arithmetic, so (L+1)² covers both balanced ends (Lemma
  `wo:lem:fold-overflow`).
- **`UnitOneHotFoldPolicy`.** This is the one-hot root that Jolt fp128 uses.
  `PerAttempt` keeps 7/8. `OneShot` replaces 7/8 with
  `2^−security_bits / masked_folds` in the same Chernoff solve
  (`honest_fold_policy.rs:195-215`). The policy falls back to the
  deterministic cap when the solve does not reach the target. At that cap
  δ = 0.
- **Modeled cap.** It is not used at levels ≤ `cut_level`: ZK planning sets
  `modeled_linf_cap = None` there, so `candidate.rs:172-177` takes the policy's
  digits. A 1/40 per-attempt cap is never a one-shot bound.
- **Admission.** `ZkOpeningSchedule`'s constructor rejects a row whose
  `num_digits_fold` at any level ≤ `cut_level` is below the `OneShot` policy's
  digits. Until a planner path emits ZK rows, that check is what pins the
  one-shot sizing.

The parameters are `security_bits = 136` and `masked_folds = cut_level + 1 = 2`,
so δ_fold ≤ 2^−136 in total. With the paper's other terms (§6.2), W2-c
computes the total zero-knowledge bound below 2^−128. The paper gives no
value for Q_tot (Q6), so the test asserts only the terms it can compute:

```text
δ_LHL + T_att·ε/M + 2δ_fold + (T_att·Q_tot + T_att²)/2^ν
```

The extra digits are applied before `rounded_up_role_a_inf_norm`, so A is
re-priced at the enlarged range (Lemma `wo:lem:randomized-base`). The schedule
audit recomputes the A bound from `num_digits_fold`
(`akita-schedules/src/audit.rs:318-350`). Folds at or before the cut use
the unfiltered family (d = 64, η² = 71), so the lemma's 1/p factor is 1. The
paper's 2^−950 figure is per fold (Q5, answered).

### 2.4 Acceptance

- **W2-a.** A ZK plan has a `FoldResponse` run with `nonce_bits = 0` at every
  level ≤ `cut_level`, and the event log has no `Nonce` event at those sites.
  Clear plan digests are unchanged (golden test).
- **W2-b.** On the test-only ZK schedule, a forced too-small digit count makes
  `prove` return `FoldOverflow { level }` after one probe. With a correct
  schedule and a fresh RNG, the proof verifies.
- **W2-c.** For the headline shapes, each per-fold one-shot bound is
  ≤ 2^−136/2, checked against the lemma's closed form in exact rational
  arithmetic (one-hot: against the Chernoff bound at the chosen λ). The total
  ZK bound above is reported and asserted ≤ 2^−128.
- **W2-d.** At k = 2, the re-priced A ranks for levels 0 and 1 equal the clear
  ranks, as paper §6.2 states. If they differ, the test fails.

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

Precommitted groups keep zero randomness: the committed program, trusted
advice, and the setup-prefix slot. Zero knowledge holds relative to them, as in
paper §3.1, "Advice and committed programs".

### 3.2 Sizing, one owner

`akita_params::zk::hiding_randomness_columns(image: ImageShape, alphabet: DigitAlphabet)`
is the only caller of
`jolt_crypto::lattice::required_randomness_columns(image_coeffs, 0, ring_dim, log_basis, 8)`.

- `ImageShape` is `{ rows, ring_dim }` of the outer map, per slice for B and
  per level for D.
- `DigitAlphabet` is the open basis of the witness that holds the randomness.

No other site passes literal arguments. The helper is called by the planner
when it builds a ZK schedule, and by `ZkSchedule::validate` in release, which
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
- **W1-d.** For every ZK schedule, `ZkSchedule::validate` accepts the planner's
  widths and ranks. A frozen-shape test pins the headline: rank 1 at levels 0
  and 1, and 45 columns per slice.

## 4. W3: the masked levels and the cut

### 4.1 The cut

`cut_level = 1` (k = 2). `ZkSchedule::validate` enforces Def.
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
- **W3-g.** `ZkSchedule::validate` rejects each §4.1 violation.

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
| 2 | **W2**, plus the test-only `ZkOpeningSchedule::for_tests` constructor | S1, S11, S13 | 1 | W2-a…d |
| 3 | **W1** (`commit_hiding` in the `zk` module) | S2, S3, S12 | 1 | W1-a…d |
| 4 | **W3**: the cut, `ZkSchedule::validate`, the masked prover and verifier, `CountingGrinding`, `ResidualOpening` (raw u_a) | S6, S9, S10 | 2, 3, M2b/M3b fixups | W3-a…g |
| 5 | **W5a**: `AkitaResponseProver`, direct σ_N | — | 1 | W5-a, W5-b |
| 6 | **W5b**: σ_N offload (conditional, §5.3) | S7 | 5 | W5-c |
| 7 | **W5d (last)**: publish, artifacts, `IS_ZK`, CI `test-zk` | — | 4, 5 (6 if landed) | HT1, coverage, HT3 (gates); HT4, HT5; W5-e |

PRs 2, 3 and 5 are independent once PR 1 has landed. No PR before 7 makes ZK
selectable.

## 9. Soundness and zero knowledge, mapped to the paper

| Choice | Paper item |
|---|---|
| `nonce_bits = 0` runs; one probe; whole-proof rerun with all randomness fresh | §4.1 "Fold responses"; Lemma `wo:lem:fold-overflow`; H1 |
| One-shot targets on every policy (2^−136/2 per fold); modeled cap off before the cut; A re-priced | Lemma `wo:lem:randomized-base`; Thm `wo:thm:zk` (2δ_fold) |
| Randomness per slice, sized by one helper, open alphabet, in the next witness; precommitted groups excluded | §4.1 "Commitment randomness"; Lemma `wo:lem:stat-hiding`; H9; §3.1 advice |
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
- **A26, small test shapes.** Whether the ZK planner finds feasible schedules
  at test sizes is unverified. `for_tests` (PR 2) covers tests either way.
- **A27, setup-group alignment.** Aligning the setup group to R_256 boundaries
  adds zero cells, in an unmeasured count.
- **A28, residual slicing.** This note fixes one slice.
- **A29, event-log fixtures.** Moving to per-element `bind_derived_base`
  changes jolt-transcript's event-log grouping but not the sponge bytes.
  Whether Akita freezes an event log anywhere is unverified; the M3b fixup
  regenerates any it finds.
- **A30, W5b.** Whether the direct σ_N scan fits the verifier budget is
  measured in W5-e.
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
