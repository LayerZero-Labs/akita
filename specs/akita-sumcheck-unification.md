# Spec: Akita Sumcheck Unification

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-06-08                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Akita's sumchecks are hard to read, hard to audit, and hard to extend.
The per-stage math is hand-inlined separately on the prover and verifier, the prover fuses several sumchecks over one witness inside a bespoke inner loop, and every new sumcheck (the y-ring trace term, the L2 certificate sumchecks, the setup-offloading product sumcheck) is added by editing a monolithic stage driver.
The result is a cartesian product of modes (`two_round_prefix` x prefix-x x prefix-y x sparse x compact/full x fuse-next-round) interleaved with fold logic, three different batching implementations, and a real correctness/perf hazard: adding the trace term disables the `two_round_prefix` prover optimization wholesale (`crates/akita-prover/src/protocol/sumcheck/akita_stage2/lifecycle.rs` in the y-ring worktree).

This spec unifies Akita's sumchecks behind one declarative descriptor that both the prover and verifier consume, modeled on the idealized modular Jolt design.
A sumcheck instance becomes a sum-of-products expression over typed sources (openings, challenges, publics).
The verifier evaluates the descriptor as math; the prover runs the same descriptor through a generic fused kernel, with bespoke hot-loop kernels admitted only as fast paths that must produce byte-identical round messages.
A per-level protocol plan derives the ordered list of sumcheck instances, carried-opening claims, and transcript schedule from `LevelParams` plus feature gates, so the prover and verifier agree on protocol shape by construction.

## Intent

### Goal

Overhaul `akita-sumcheck` into the lower-layer home for all generic sumcheck machinery (the declarative descriptor algebra, the boolean-hypercube engine, the regular and eq-factored proof formats, batching, accumulators, the prover fast-path trait, and the proof sink), and add a new `akita-protocol` crate that holds only protocol descriptions (the concrete per-stage formulas, the per-level protocol plan, gating, transcript schedule, and identifier types) calling into `akita-sumcheck` for every building block, so each sumcheck's exact equation and message schedule is defined once and consumed by both the prover and verifier.

### Invariants

The implementation must preserve the following properties.
Each names the test, benchmark, or protocol relation that protects it.

- Single source of truth.
  The round-polynomial summand and the verifier's expected-output equation for a sumcheck are the *same* descriptor object, parametrized only by how a `Source` is resolved (prover resolves an opening to a witness view, verifier resolves it to a claimed value).
  Protected by: descriptor construction lives in `akita-protocol`; prover and verifier both import it; there is no second hand-written copy of any stage equation.

- Prover/verifier transcript equality.
  The Fiat-Shamir message ordering (claim absorption, per-round append, challenge squeeze, output-claim absorption) is derived from the shared protocol plan and is identical on both sides.
  Protected by: existing `logging-transcript` event-stream equality tests (`crates/akita-transcript`, `LoggingTranscript`), extended to the stage-2 pilot.

- Byte-identical fast paths.
  Every prover compute optimization (the generic kernel, the compact-integer scan, Gruen split-eq, `two_round_prefix`, streaming) computes the identical round polynomials for a given instance.
  A fast path is an alternative computation of the same declared message, never a different message.
  Protected by: a new equivalence test that runs each fast path and the generic Tier-A kernel on representative instances and asserts equal round polynomials every round (new test, `akita-prover` or `akita-sumcheck`).

- Proof-shape stability for the pilot.
  The serialized stage-2 proof produced by the unified pilot is byte-identical to current `main` on the deterministic fixtures.
  Protected by: existing serialization fixtures (`crates/akita-pcs/tests/algebra/serialization.rs`) and a byte-equality check against a captured baseline proof.

- Verifier no-panic boundary.
  The descriptor evaluator and protocol-plan derivation are verifier-reachable and must reject malformed input with `AkitaError`, never panic, per the AGENTS.md no-panic contract.
  Protected by: `try_evaluate`-style fallible evaluation; existing verifier malformed-input tests extended to descriptor evaluation.

- Performance parity.
  Stage-2 prover wall time does not regress versus `main` once the compact fast path is wired through the descriptor.
  Protected by: the canonical profile command (Performance section) and the existing field/PCS benches.

- Soundness preservation.
  No change to the mathematical content of any existing protocol relation; the descriptor expresses the *current* stage-2 equation, not a new one.
  Protected by: existing end-to-end prove/verify tests for stage 2 (`crates/akita-prover/tests`, `crates/akita-verifier`), which must remain green unchanged.

- Instance kind drives batchability and proof format.
  A regular instance is arbitrarily batchable and serializes in the regular compressed format.
  An eq-factored instance is not batchable: its proof-size optimization (sending the inner `q` with its linear term omitted) is valid only when the instance is proven standalone.
  When an eq-factored instance is batched with any other instance it must fall back to the regular compressed format (no proof-size win); the prover still uses Gruen split-eq for computation.
  Protected by: the protocol plan selects the proof format from the batching decision; a new test that a standalone eq-factored instance serializes in eq-factored format and a batched one in regular format.

- Boolean hypercube only.
  Every Akita sumcheck round is over the boolean hypercube; univariate skip is not supported.
  Protected by: the engine exposes no non-boolean round domain and no centered-integer path; this is a deliberate simplification versus the Jolt reference template.

### Non-Goals

- Not changing the soundness, equations, or proof semantics of any existing Akita protocol.
  This is a structural refactor; the pilot reproduces stage 2 exactly.
- Not implementing y-ring trace internalization, L2 certificate sumchecks, or setup-claim offloading in this spec.
  The protocol-plan abstraction is fully designed to *absorb* them, but they remain separate PRs.
- Not implementing the polyops/witness-source cutover.
  This spec consumes whatever witness/polynomial sources exist; the descriptor's openings resolve through them.
- Not migrating stage 1 (range tree, eq-factored), the extension-opening reduction, or the setup product sumcheck in the pilot.
  The spec defines the full-unification target for all of them; only stage 2 is implemented first.
- Not introducing a GPU backend.
  The two-tier kernel contract is backend-agnostic, but only the CPU backend is in scope.
- Not adding univariate skip or centered-integer-domain sumchecks.
  All Akita sumchecks remain over the boolean hypercube; the Spartan-outer uniskip pattern from the Jolt reference template is explicitly not mirrored, which removes the centered-Lagrange round machinery from scope.

## Evaluation

### Acceptance Criteria

- [ ] `akita-sumcheck` is overhauled to own the generic descriptor algebra (`Source`, `Term`, `Expr`, `SumcheckInstanceDescriptor`, `InstanceKind`, `ClaimSlot`, generic over identifier types), both proof formats (regular compressed and eq-factored), the Tier-A kernel, the `SumcheckFastPath` trait, and the proof sink; it names no protocol-specific identifier or equation.
- [ ] New crate `akita-protocol` exists holding only protocol description: the concrete `AkitaOpeningId`/`AkitaPublicId`/`AkitaChallengeId` identifier types, per-stage formula constructors, and the per-level protocol plan (`LevelProtocolPlan`, `StagePlan`, `BatchingScheme`, `plan_level`), composed from `akita-sumcheck` building blocks.
- [ ] `akita-protocol` depends only on `akita-sumcheck`, `akita-types`, `akita-field`, `akita-challenges`; it names no `CommitmentConfig`-style preset type beyond what the plan inputs require.
- [ ] Instance-kind/format handling: a standalone eq-factored instance serializes in eq-factored format; an eq-factored instance batched with others serializes in the regular compressed format (with Gruen split-eq still used for compute); a regular instance is arbitrarily batchable. Covered by a format-selection test.
- [ ] All sumcheck rounds are over the boolean hypercube: the engine has no non-boolean round domain and no centered-integer path.
- [ ] Stage 2 verifier `expected_output_claim` is computed by evaluating the shared descriptor (no hand-inlined equation in `akita-verifier`).
- [ ] Stage 2 prover builds the shared descriptor, runs the generic Tier-A fused kernel by default, and uses the compact-integer scan as a registered fast path.
- [ ] Equivalence test: compact fast path round polynomials equal generic kernel round polynomials on representative stage-2 instances, every round.
- [ ] Proof bytes for stage 2 are identical to `main` on deterministic fixtures.
- [ ] `logging-transcript` event-stream equality holds for the unified stage-2 prover and verifier.
- [ ] The protocol plan derives the stage-2 instance list and transcript schedule from `LevelParams` + gates, and the same function is called by prover and verifier.
- [ ] A written forward-design section shows how the y-ring trace term, L2 `{Range,L2}`/`{S,Relation,Virtualization}` claim lists with mode gating, and setup-offloading Stage-3 eligibility map onto the plan, including central gamma-power batching allocation.
- [ ] Stage-2 prover wall time within noise of `main` on the canonical profile workload.

### Testing Strategy

Existing tests that must remain green unchanged:

- Stage-2 end-to-end prove/verify (`crates/akita-prover/tests`, `crates/akita-verifier` stage tests).
- Serialization fixtures (`crates/akita-pcs/tests/algebra/serialization.rs`).
- Transcript tests under the `logging-transcript` feature.

New tests:

- Descriptor evaluation: `Expr::try_evaluate` returns the correct value and rejects malformed source references with `AkitaError`.
- Fast-path equivalence (byte/field-identical round polynomials) for the compact-integer stage-2 scan versus the generic kernel.
- Protocol-plan determinism: `plan_level` is a pure function of `(LevelParams, gates)` and prover/verifier obtain identical plans.
- Format selection: a standalone eq-factored instance round-trips in eq-factored format; the same instance placed in a batch with another instance round-trips in the regular compressed format, and both verify.
- Proof byte-equality versus a captured `main` baseline on the deterministic stage-2 fixtures.

Feature combinations: run with and without `parallel`; run the transcript tests with `logging-transcript`; the proof byte-equality check runs in release on the deterministic fixture config.

### Performance

The hot paths are the stage-2 fused witness scan and (later) the stage-1 eq-factored scan.
Baseline and treatment are measured with the canonical profile command from AGENTS.md:

```bash
AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=32 cargo run --release --example profile
```

Expected direction: no regression in stage-2 prover time.
The generic Tier-A kernel may be slower than the bespoke scan in isolation; parity is achieved because the compact scan remains the selected fast path for the stage-2 instance shape.
Proof size: unchanged for the pilot (byte-identical proofs).
If a later stage cannot reach parity through a fast path, that is a blocking finding to surface, not a silent regression.

## Design

### Architecture

Four layers, mirroring idealized modular Jolt (`/Users/quang.dao/Documents/SNARKs/jolt-audit-prep/crates`: `jolt-sumcheck`, `jolt-claims`, `jolt-witness`, `jolt-backends`, `jolt-verifier`), adapted to Akita's lattice specifics.
The generic descriptor algebra and engine live in the lower `akita-sumcheck` layer, and `akita-protocol` holds only the protocol descriptions that compose them.

```text
akita-field / akita-challenges / akita-types
        |
akita-sumcheck   (OVERHAULED lower layer: descriptor algebra (Expr/Term/Source/descriptor),
                  boolean-hypercube engine, regular + eq-factored proof formats, batching,
                  accumulators, compact fold, fast-path trait, proof sink)
        |
akita-protocol   (NEW, pure protocol description: identifier types, concrete per-stage formulas,
                  per-level protocol plan, gating, transcript schedule)
        |        \
akita-prover      akita-verifier
 (Tier-A generic   (evaluate descriptor -> expected_output_claim; discharge openings)
  kernel + Tier-B
  fast paths)
```

`akita-sumcheck`'s own `lib.rs:3-5` already states this direction ("owns only protocol-independent sumcheck machinery. Akita-specific stage provers, verifier instances, and two-round-prefix skip proofs stay in the PCS protocol crate until their role-specific APIs are split"); this spec finishes that split into `akita-protocol`.

#### 1. Descriptor algebra (`akita-sumcheck`)

A sumcheck instance's round-polynomial summand is a sum-of-products over typed sources.
This is the generic algebra (analogous to Jolt's `Expr`, `crates/jolt-claims/src/claims.rs:41`, and `SumcheckRegularBatchInstance`, `crates/jolt-backends/src/sumcheck/request.rs:120`); it lives in `akita-sumcheck` and is generic over the identifier types, which are defined in `akita-protocol`.

```rust
// akita-sumcheck::descriptor (generic over identifier types O, P, C)
pub enum Source<O, P, C> {
    Opening(O),    // an MLE source: prover resolves to a witness view; verifier to a claimed value
    Challenge(C),  // a Fiat-Shamir scalar
    Public(P),     // a public weight: relation row, trace weight, range coefficient, eq point
}

pub struct Term<F, O, P, C> { pub coefficient: F, pub factors: Vec<Source<O, P, C>> } // coeff * product
pub struct Expr<F, O, P, C> { pub terms: Vec<Term<F, O, P, C>> }                       // sum of terms

pub enum InstanceKind {
    Regular,      // batchable; regular compressed proof format
    EqFactored,   // not batchable; eq-factored proof format only when standalone
}

pub struct SumcheckInstanceDescriptor<F, O, P, C> {
    pub label: &'static str,
    pub num_rounds: usize,
    pub degree: usize,
    pub kind: InstanceKind,
    pub input_claim: ClaimSlot,   // chained input (split-sumcheck handoff)
    pub output_claim: ClaimSlot,  // chained output
    pub poly: Expr<F, O, P, C>,   // the summand g(x) for this instance
    pub views: Vec<ViewRequirement>, // witness/polynomial sources needed by the prover
}
```

There is no domain field: all rounds are over the boolean hypercube (univariate skip is out of scope), so the engine never needs a non-boolean round domain.

The identifier types are generic so the same descriptor serves both sides.
The verifier instantiates evaluation with `resolve_opening: O -> F` (claimed values); the prover instantiates the kernel with `resolve_opening: O -> PolynomialView` (witness tables).
`akita-protocol` defines the concrete `AkitaOpeningId`, `AkitaPublicId`, `AkitaChallengeId` enums and constructs concrete `Expr` values; `akita-sumcheck` never names a protocol-specific identifier.

Akita-specific additions beyond a plain `Expr`:

- An eq factor must be recognizable so the prover can apply the Gruen split-eq optimization.
  The descriptor marks the eq source (for example a dedicated `Source::Eq(point_id)` or an opening descriptor tagged `is_eq`), so a fast path can specialize it while the generic kernel treats it as an ordinary factor.
- Sources carry an encoding hint (field vs compact integer) so the compact-integer fast path can recognize a compact witness source; the generic kernel ignores the hint and reads the field MLE.

#### 2. Instance kinds and proof formats (`akita-sumcheck`)

Three orthogonal axes must not be conflated; the current code blurs them and that is a source of the spaghetti.

- Protocol / proof-format axis: regular compressed format versus eq-factored format.
  The eq-factored format is the proof-size optimization: for a round polynomial that factors as `s(X) = l(X) * q(X)` with `l` the linear eq factor, the prover sends `q` with its linear term omitted (`EqFactoredUniPoly`, `crates/akita-sumcheck/src/types.rs`), and the verifier reconstructs `s` from `l`.
  This format is only valid for a standalone eq-factored instance, because batching linearly combines round polynomials and the combined polynomial no longer shares a single eq factor.

- Batching axis: which instances are combined into one front-loaded batched sumcheck.
  Regular instances are arbitrarily batchable (`prove_batched_sumcheck`, `crates/akita-sumcheck/src/batched_sumcheck.rs`, which today only consumes the regular `SumcheckInstanceProver`).
  An eq-factored instance cannot be batched while keeping its format.
  When the protocol plan places an eq-factored instance in a batch, the engine emits it in the regular compressed format and drops the proof-size win.

- Prover-compute axis: Gruen split-eq, the compact-integer scan, `two_round_prefix`, and streaming.
  These are alternative prover computations of the identical declared round polynomials.
  They are protocol-invariant and never change the proof bytes.

Crucially, Gruen split-eq sits on the compute axis, while the eq-factored format sits on the proof-format axis.
"Disabling the eq-factored optimization when batched" means dropping the proof-format win, not dropping Gruen split-eq: the prover keeps computing the eq factor with split-eq, and only the wire encoding changes.

The protocol plan owns the batching decision, so it owns whether each eq-factored instance keeps or loses its format.
The engine therefore takes the instance kind and the batching grouping as inputs and selects the wire format deterministically; the verifier derives the same selection from the same plan.

#### 3. Verifier = evaluate the descriptor (`akita-verifier`)

The verifier already has a clean shape (`crates/akita-verifier/src/stages/stage2.rs`); this replaces the hand-inlined equation with descriptor evaluation:

```rust
fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
    self.descriptor.poly.try_evaluate(
        |opening| self.resolve_opening(opening, challenges),  // witness eval / claimed value
        |challenge| self.resolve_challenge(challenge),        // batching coeff, gamma powers
        |public| self.resolve_public(public, challenges),     // relation row, eq, range coeff
    )
}
```

`try_evaluate` is fallible and panic-free.
The sum-to-eval reduction is the generic engine; the eval-to-commitment discharge stays in the existing opening layer.
This is the same separation Jolt enforces (`crates/jolt-sumcheck/src/lib.rs:33`: the `EvaluationClaim` must be discharged against the oracle by the caller).

Improvement over Jolt: Jolt keeps the prover request shape (`jolt-prover`) and the verifier formula (`jolt-claims`) as two artifacts cross-checked by a test (`stage1_request_matches_verifier_spartan_outer_shape`).
Akita unifies them into one descriptor consumed by both sides, so they cannot drift and no shape-match test is needed; only the fast-path-vs-generic equivalence test remains.

#### 4. Two-tier prover contract (`akita-prover` + `akita-sumcheck`)

Tier A is the generic fused kernel.
It folds all batched instances once per challenge and computes each instance's round message by walking the declared `Expr` over witness views, evaluating each factor's view at the round's hint points and combining per the product structure.
Promote the front-loaded batching driver (`crates/akita-sumcheck/src/batched_sumcheck.rs`) and the accumulator helpers (`accum.rs`, `compact_fold.rs`) into this kernel.
This is the always-correct reference implementation.

Tier B is bespoke kernels for hot stages, admitted only through a fast-path registry:

```rust
trait SumcheckFastPath<F, O, P, C> {
    fn matches(&self, instance: &SumcheckInstanceDescriptor<F, O, P, C>) -> bool;
    fn evaluate_round(&mut self, round: usize, previous_claim: F) -> UniPoly<F>;
    fn bind(&mut self, round: usize, r: F);
}
```

The driver selects a matching fast path or falls back to Tier A.
The hard contract: for any instance a fast path claims to match, its per-round polynomials equal the Tier-A kernel's, enforced by the equivalence test.

Prover compute optimizations are exactly the Tier-B fast paths and are protocol-invariant.
This is the corrected model for `two_round_prefix`: it is not a protocol change (the verifier sees identical round polynomials, domain, and proof bytes; the optimization is reconstructed prover-side and never serialized).
It is a fast-path computation of the same declared messages, in the same category as the compact-integer scan and Gruen split-eq.
Consequence: a fast path that does not yet compose with an added summand (the current trace-disables-`two_round_prefix` situation) falls back to Tier A, which is slower but produces identical bytes, and is a perf item to generalize, never a protocol fork.

The per-round loop and transcript interaction live behind a proof-sink abstraction (the Akita analog of Jolt's `Stage6ProofSink`, `crates/jolt-prover/src/stages/stage6/prove.rs:715`): absorb input claims, per round append the round polynomial and squeeze the challenge, finish by absorbing output claims.
Clear and ZK (committed-round) sinks share the loop.

#### 5. Per-level protocol plan (`akita-protocol`)

The level's protocol is runtime-derived, not fixed.
A pure function maps level parameters and feature gates to an ordered schedule:

```rust
pub struct LevelProtocolPlan<F, O, P, C> {
    pub stages: Vec<StagePlan<F, O, P, C>>,
    pub carried_openings: CarriedOpeningPlan,   // singleton or list (offloading)
    pub transcript_schedule: TranscriptSchedule,
}
pub struct StagePlan<F, O, P, C> {
    pub instances: Vec<SumcheckInstanceDescriptor<F, O, P, C>>, // batched together
    pub batching: BatchingScheme,  // gamma-power index per regular instance, allocated centrally
}
pub fn plan_level(params: &LevelParams, gates: &ProtocolGates)
    -> Result<LevelProtocolPlan<..>, AkitaError>;
```

Both prover and verifier call `plan_level` with the same inputs and obtain the same schedule, so Fiat-Shamir ordering, batching, and per-instance proof format are identical by construction.
`ProtocolGates` carries the feature switches: trace on/off, L2 certificate mode, setup-offload eligibility, ZK.

The plan also resolves the instance-kind/proof-format interaction from the previous section: a `StagePlan` with a single eq-factored instance keeps the eq-factored format; any `StagePlan` that batches an eq-factored instance with others demotes it to the regular format (the prover keeps Gruen split-eq compute).
`BatchingScheme` therefore allocates gamma powers only over instances that are actually batched together, and a standalone eq-factored stage carries no batching coefficient.

How the in-flight features map onto the plan (forward design, not implemented here):

- y-ring trace internalization.
  The trace term is one extra `Term` (`gamma^2 * W * TraceWeight`) appended to the stage-2 instance's `Expr`, with `TraceWeight` a `Source::Public`.
  No new instance, no new rounds.
  The gate `trace=true` selects the extended `Expr`; the gamma-power index for the trace term comes from the central `BatchingScheme`, not a hardcoded `gamma^2`.

- L2 certificate sumchecks.
  `certificate_mode` (`Deterministic` vs `Realized`, `crates/akita-types/src/sis/l2_certificate.rs` in the PR-158 worktree) gates the instance lists: stage 1 `{Range}` vs `{Range, L2}`, stage 2 `{S, Relation}` vs `{S, Relation, Virtualization}`.
  Each is a `SumcheckInstanceDescriptor` in the relevant `StagePlan`; the realized stage-1 format (regular fused-root vs eq-factored) is exactly the `InstanceKind`-plus-batching choice from section 2, selected by the plan and distinct from prover optimizations.

- Setup-claim offloading.
  Eligibility (`D == D_setup`, `N_prefix >= N_min`, presence of a next recursive fold; `STACK.md`) gates whether a Stage-3 setup product-sumcheck instance is appended and whether `carried_openings` is a singleton or a list.
  The relation-shift variant (`STACK.md` Slice 03B/04) is a different `plan_level` output for the same level, chosen by gates.

The plan is the single place gamma-power batching is allocated, so trace, L2, and offloading cannot collide on the same power.

#### 6. Crate boundaries

- `akita-sumcheck` (existing, overhauled into the generic lower layer): the descriptor algebra (`Source`/`Term`/`Expr`/`SumcheckInstanceDescriptor`/`InstanceKind`, generic over identifier types), the boolean-hypercube engine (round loop, batched kernel; promote `batched_sumcheck.rs`), both proof wire formats (regular compressed and eq-factored), accumulators (`accum.rs`), compact fold (`compact_fold.rs`), the `SumcheckFastPath` trait, and the proof-sink abstraction.
  Names no protocol-specific identifier or equation.
  The current `traits.rs` `SumcheckInstance{Prover,Verifier}` / `EqFactored*` pairs are folded into the descriptor + engine model rather than kept as parallel hand-rolled traits.
- `akita-protocol` (new): pure protocol description only. The concrete `AkitaOpeningId` / `AkitaPublicId` / `AkitaChallengeId` identifier types, the per-stage formula constructors that build concrete `Expr` values, `plan_level`, gating, transcript schedule, and claim-slot chaining.
  Calls into `akita-sumcheck` for every struct/trait/helper; holds no generic engine code.
  Depended on by `akita-prover` and `akita-verifier`. Verifier-reachable, so panic-free.
- `akita-types` (existing, slimmed): pure types and shared helpers (layouts, proof structs, SIS tables); stage equations move out to `akita-protocol`.
- `akita-prover`: Tier-A kernel instantiation plus Tier-B fast paths (compact scan, Gruen split-eq, `two_round_prefix`).
- `akita-verifier`: descriptor evaluation for `expected_output_claim`, opening discharge.

Mapping of current pieces to new homes:

| Current | New home |
|---|---|
| `akita-sumcheck/traits.rs`, `drivers/`, `types.rs` (proof formats), `batched_sumcheck.rs` | overhauled `akita-sumcheck`: descriptor algebra + engine + both proof formats + Tier-A kernel |
| `akita_stage2/{mod,lifecycle,round_flow,dense_terms}.rs` fused scan | stage-2 descriptor (`akita-protocol`) + compact fast path (`akita-prover`) |
| `akita-verifier/stages/stage2.rs` `expected_output_claim` equation | stage-2 descriptor evaluation against the shared descriptor |
| stage-1 tree interstage batching, stage-2 coeff-addition fusion | `BatchingScheme` in the protocol plan (`akita-protocol`) |
| eq-factored proof-size optimization (`EqFactoredUniPoly`) | eq-factored wire format in `akita-sumcheck`, selected by the plan's batching decision |
| `two_round_prefix/` | Tier-B fast path (`akita-prover`), protocol-invariant |

### Alternatives Considered

- Keep editing monolithic stage drivers.
  Rejected: each new sumcheck multiplies the mode cartesian product and risks protocol/optimization coupling (the trace-disables-prefix hazard).

- One fully-generic kernel for all sumchecks (no Tier-B).
  Rejected: the hottest Akita scans (compact-integer, Gruen split-eq) would regress.
  Idealized Jolt itself keeps bespoke kernels for its heaviest sumcheck (Spartan outer) while keeping the declaration canonical; we follow that two-tier approach.

- Verifier-only unification (shared formulas, leave prover fusion as-is).
  Rejected by direction: the user wants full unification end-to-end, and the prover fusion is the larger maintenance pain.

- Separate prover request and verifier formula artifacts with a cross-check test (Jolt's current shape).
  Rejected in favor of a single shared descriptor, which removes drift and the cross-check test entirely.

- Macro-generated stage code.
  Rejected: macros hide the math; a data descriptor is auditable and serializable.

- Put the descriptor algebra in `akita-protocol` (my first draft).
  Rejected per direction: the descriptor algebra, engine, proof formats, and fast-path trait are generic, protocol-independent machinery and belong in the lower `akita-sumcheck` layer.
  `akita-protocol` should be purely protocol description that composes those lower-layer building blocks, which also matches `akita-sumcheck`'s existing stated intent (`lib.rs:3-5`).

## Documentation

- Update `AGENTS.md` crate list to add `akita-protocol`, re-describe `akita-sumcheck` as the overhauled generic lower layer (descriptor algebra + engine + proof formats), and refocus `akita-types`.
- Crate-level docs for `akita-sumcheck` describing the descriptor algebra, the two proof formats, and the instance-kind/batching rules; crate-level docs for `akita-protocol` describing the per-stage formulas and the protocol plan, with the stage-2 instance as the worked example.
- Cross-link this spec from `specs/` and reference it from the y-ring, L2, and setup-offloading specs as the shared abstraction they target.
- A short "anatomy of an Akita sumcheck" section in the crate docs: the artifacts to define for a new sumcheck and where each lives.

## Execution

Stage-2 pilot, end-to-end on both sides:

1. Overhaul `akita-sumcheck` into the generic lower layer: add the descriptor algebra (`Source`/`Term`/`Expr`/`SumcheckInstanceDescriptor`/`InstanceKind`, generic over identifier types), fold the existing `traits.rs`/`drivers/` pairs and the two proof formats (`types.rs`) into the descriptor + engine model, promote `batched_sumcheck.rs` to the Tier-A kernel, and add the `SumcheckFastPath` trait and proof-sink abstraction.
2. Scaffold `akita-protocol` with the concrete identifier types (`AkitaOpeningId`/`AkitaPublicId`/`AkitaChallengeId`) and a stub `plan_level` that returns the current stage-2 schedule by composing `akita-sumcheck` building blocks.
3. Define the stage-2 descriptor: a regular instance over sources `[eq, W, alpha, m]` with products for the virtual term (`batching_coeff * eq * W * (W+1)`) and the relation term (`W * alpha * m`), degree 3 over the boolean hypercube, input claim `batching_coeff * s_claim + relation_claim`.
4. Verifier: replace `expected_output_claim` with descriptor evaluation; keep opening discharge.
5. Prover: build the descriptor, run the Tier-A generic kernel; add the compact-integer scan as a `SumcheckFastPath`; wire the proof sink.
6. Add the fast-path equivalence test and the proof byte-equality fixture.
7. Confirm transcript equality and perf parity.

Migration order after the pilot (separate PRs): stage-1 (eq-factored, range tree, which exercises the standalone-eq-factored vs batched-demotion path), the extension-opening reduction, the setup product sumcheck.

Stacking with other work:

- This spec is intended as the base of the stack: it is the shared abstraction the y-ring, L2, and setup-offloading work target, and they become declarative additions on top.
- The final order (this strictly first vs y-ring landing first and rebasing) is still open, but the protocol plan is designed so y-ring can land either before or after by adding the trace term as an `Expr` addend gated by `trace=true`.

Risks to resolve first:

- Perf parity of the compact fast path through the descriptor indirection.
  Mitigation: the fast path owns the hot loop; the descriptor only selects it.
- The eq-factor and compact-encoding source tagging must be expressive enough for the Gruen and compact optimizations.
  Mitigation: prototype the stage-2 fast path against the descriptor before migrating other stages.
- Verifier no-panic in `try_evaluate` and `plan_level`.
  Mitigation: fallible APIs and malformed-input tests from the start.

## References

- Idealized modular Jolt (worktree `/Users/quang.dao/Documents/SNARKs/jolt-audit-prep`, branch `refactor/audit-prep`):
  - `crates/jolt-sumcheck/src/{lib,verifier,batched_verifier,claim}.rs` (generic engine).
  - `crates/jolt-claims/src/claims.rs` (`Expr`/`Term`/`Source`), `protocols/jolt/formulas/spartan.rs` (per-sumcheck formulas).
  - `crates/jolt-backends/src/sumcheck/request.rs` (`SumcheckRegularBatch*`), `cpu/sumcheck/kernels/regular_batch.rs` (generic fused kernel + shape-matched fast paths).
  - `crates/jolt-witness/src/{provider,polynomial}.rs` (witness/polynomial source).
  - `crates/jolt-prover/src/stages/stage1/{request,prove}.rs` (declared instance, split-sumcheck, proof sink).
- Akita current sumchecks:
  - `crates/akita-sumcheck/src/{lib,traits,batched_sumcheck,accum,compact_fold,types}.rs`, `drivers/` (regular + eq-factored).
  - `crates/akita-prover/src/protocol/sumcheck/akita_stage2/{mod,lifecycle,round_flow,dense_terms}.rs`.
  - `crates/akita-verifier/src/stages/stage2.rs`.
- Related specs and worktrees:
  - `specs/y-ring-trace-internalization.md` (worktree `akita-y-ring-trace-internalization`): fused trace term.
  - `specs/l2-folded-witness-sumchecks.md`, `crates/akita-types/src/sis/l2_certificate.rs` (worktree `akita-pr-158-l2-certificate-sumchecks`): claim lists and mode gating.
  - `specs/akita-polyops-cutover.md` (worktree `akita-polyops-cutover-spec`): witness/polynomial source boundary.
  - `STACK.md`, `specs/setup-layout-repack.md`: setup-claim offloading and conditional protocol shape.
- Profiling: `AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=32 cargo run --release --example profile`.
