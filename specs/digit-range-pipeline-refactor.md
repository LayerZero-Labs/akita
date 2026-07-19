# Digit Range Pipeline Refactor and Mixed-Dimension Sumcheck: Implementation Handoff

| Field | Value |
|---|---|
| Author(s) | Quang Dao (direction); Codex planning synthesis |
| Created | 2026-07-18 |
| Revised | 2026-07-19; concurrent Opus/Fable review reconciled; per-level setup prefixes, verifier-time selection objective, staged 7a/7b cutover, gated `Batched`, envelope-level `setup_contribution_eval`, shared-emitter ownership, current #311 stack, and bounded additive/cutover PR delivery model recorded |
| Status | F1 implementation may proceed on #311; #309 is deferred to the pre-C5 semantic-basis integration gate |
| First PR branch | `quang/plan-digit-range-pipeline` |
| Stack base | PR #311 at `bc959ef34572aee143ba0114094b0b4212b4e111` |
| Historical audit base | `main` at `f5c180a49a83f5ce3e8b683a34208166ffed2f66` |
| Related | [`digit-innermost-layout.md`](digit-innermost-layout.md), [`runtime-ring-cutover.md`](runtime-ring-cutover.md), [`transcript-hardening.md`](transcript-hardening.md), [`packed-sumcheck.md`](packed-sumcheck.md), [`akita-sumcheck-unification.md`](akita-sumcheck-unification.md) |

## Read this first

Four problems are entangled in the current pipeline:

| Problem | Current consequence | Target |
|---|---|---|
| Two Stage 1 implementations | LB2/LB3 use compact kernels; LB4/LB5/LB6 fall into an eager field-valued tree | One streaming range-product prover driven by one topology plan |
| Too many layout-shaped kernels | Dense, sparse-y, x-prefix, y-prefix, round-2, and two-round paths repeat the same pair/fold mechanics | One flat pair-scan and fold toolkit; each semantic subprotocol retains its own equation |
| Public x/y geometry | Mixed A/B/D dimensions fall into a `ring_bits == 0` sentinel and dense relation weights | One flat LSB-first `WitnessDomain`; role-local dimensions stay inside semantic providers |
| Offloading adds a third stage | The relation point is fixed too late, so setup certification and witness carrying require a later setup sumcheck | In recursive-offload mode, fuse relation checking into the final Stage 1 range leaf and move setup/range-image reduction to Stage 2 |

The target has one shared range-product implementation and two schedule-bound final-leaf
compositions. The split is deliberate: pushing relation work into Stage 1 helps recursive
offloading, but would make the direct prover fold `digit_witness` twice and add a scalar
for no proof-size benefit.

```text
FoldCheckTopology::DirectSetup
  Stage 1: optimized equality-factored range tree, including range-only leaf
  Stage 2: one standard relation-and-range-image-consistency sumcheck
           -> next_witness_eval

FoldCheckTopology::RecursiveSetupOffload
  Stage 1: same optimized range-product prefix
           + final range leaf and relation in one standard sumcheck
           -> range_image_eval and digit_witness_eval at one joint point
  Stage 2: RecursiveClaimReductionPlan
           Separate    setup contribution proof + range-image consistency proof
           Batched     one lifted setup-and-range-image proof
           -> next_witness_eval + setup_prefix_eval
```

The important abstraction boundary is deliberately narrower than a generic sumcheck
engine. All range-product layers use the same eq-factored implementation. Direct mode
uses the range-only leaf composer and one standard Stage 2 relation/consistency proof.
Recursive offload uses the standard joint leaf and either a separate eq-factored
consistency proof plus standard setup proof, or one standard batch. They share address
iteration, exact-prefix access, bounded accumulation, folding, and parallel reduction—not
a trait that pretends their equations
or transcript dependencies are the same.

### Names and responsibilities

| Name | Owns | Must not own |
|---|---|---|
| `DigitRangePlan` | supported basis, roots, product topology, range-layer shape, transcript child order | relation placement, hot tables, or witness storage |
| `WitnessDomain` | one checked flat Boolean address space, live prefix, bind order, transcript-point mapping | an x/y split or role-specific relation algebra |
| `ValidatedDigitSource` | checked compact digits and `RangeImageClass` access | protocol topology |
| `range_image(digit)` | the exact map `digit * (digit + 1)` used to pair balanced roots | a table called merely `S`, "square", or "norm" |
| `ExactPrefixTable<[E; LANES]>` | the current derived lane records plus one default record | several tree levels or a `LaneTable` wrapper |
| `scan_adjacent_pairs` / reduction functions | checked adjacent-pair traversal and serial/parallel reduction mechanics | a range, relation, setup, or reduction equation, or a `PairScan` facade type |
| `RelationWeight` | one semantic flat public polynomial with dense and factorized implementations | witness folding or transcript choreography |
| `non_setup_relation_weight_eval` | full evaluation of every relation/trace term except the setup contribution | a common-base lane fragment |
| `setup_contribution_eval` | full flat setup portion at the active relation point | setup weight alone or a high-lane partial |
| `DirectRelationRangeImagePlan` | direct Stage 2 relation plus range-image consistency equation | recursive setup certification |
| `RangeRelationLeafPlan` | recursive final-leaf relation-term shape and batching choreography; joint degree is derived from `DigitRangePlan` | duplicate domain/range/degree state or setup certification |
| `RangeCheckPoint` | direct Stage 1 range-only output point anchoring Stage 2 consistency | recursive relation evaluation |
| `RangeRelationPoint` | the shared Stage 1 output point for `range_image_eval` and `digit_witness_eval` | a setup-prefix opening point |
| `SetupCoefficientDomain` | the setup prefix's own flat coefficient address space | witness-domain coordinates |
| `SetupContributionPlan` | exact setup slot/domain, role mapping, and the canonical full setup-weight polynomial at a relation point | a common-base-only claim |
| `FoldCheckPlan` | complete direct-versus-offloaded topology, domains, proof shape, and degree schedule | hot tables or runtime heuristics |
| `RecursiveClaimReductionPlan` | schedule-bound `Separate` or `Batched` offloaded Stage 2 shape | a per-round branch or wire tag |
| `SetupWitnessBatchGeometry` | projections of the fresh Stage 2 point to setup and witness native domains | the Stage 1 joint point |
| `DigitRangeRelationProof` | optimized range-product layers plus the final joint leaf | Stage 2 range-image consistency |
| `RangeImageConsistencyProof` | binds the independent range-image MLE to `digit_witness` and reduces both claims to `next_witness_eval` | setup certification |
| `SetupContributionProof` | proves the full flat setup contribution against an exact committed setup prefix | a high-lane partial claim |

Use these names in implementation and review. In particular, use `pair_scan/`, not
`engine/`: the latter suggests a semantic unification this design explicitly rejects.

Mandatory production renames at Packet 7:

| Legacy/vague name | Target name |
|---|---|
| `S`, `s_table`, `padded_s`, `s_claim` | `range_image`, `range_image_table`, `range_image_eval` |
| public/local `W` fields | `digit_witness`; keep `W` only in equations |
| `next_w_*` | `next_witness_*` |
| `v` proof field | `scaled_fold_witness`; keep `v` only in equations |
| `AkitaStage1Proof` | `DigitRangeProof` or `DigitRangeRelationProof`, selected by topology |
| direct `AkitaStage2Proof` | `DirectRelationRangeImageProof` |
| `SetupSumcheckProof` / `stage3_sumcheck_proof` | `SetupContributionProof` or `BatchedSetupAndRangeImageProof` inside `RecursiveClaimReductionProof` |
| `BatchedStage3Geometry` | `SetupWitnessBatchGeometry` |
| setup claim `C`, `claimed_setup`, high-lane setup claim | full `setup_contribution_eval` |
| a setup matrix/prefix called `S` | `SetupCoefficient`, `setup_prefix`, `SetupCoefficientDomain` |

Do not land aliases for these names. Serializer field order is migrated atomically from
the schedule shape; internal code calls the one canonical concept directly.

Several target names are reused, not new: `akita-types` already owns production
`SetupContributionPlan`, `SetupContributionGroupInputs`, `SetupProjectionGeometry`, and
`BatchedStage3Geometry`, all consumed by the current Stage 3. This plan redefines their
ownership rather than introducing them. Packet 7 must record the semantic delta for each
reused type in the PR — in particular the carried setup claim's change from its current
meaning to the full flat `setup_contribution_eval` — so a reviewer can diff meanings, not
just names.

### Settled decisions

These are not implementation choices to reopen without new benchmark or correctness
evidence:

1. There is one flat physical Boolean address domain for `digit_witness`. Mixed role
   dimensions do not create ragged witness axes or per-segment bind schedules. The
   independently committed setup prefix has its own flat `SetupCoefficientDomain`.
2. The range product tree stays. Recursive offload changes only its final leaf to a
   standard range-and-relation sumcheck whose first round walks the compact witness once
   for both terms. Direct setup keeps the optimized equality-factored range-only leaf and
   current one-pass Stage 2 relation/range-image composition.
3. The high-basis baseline is one-round streaming from compact classes, with fixed
   2/4/8-lane kernels selected by table data. Do not create one module per basis.
4. Two-round deferral is a measured LB4 experiment. LB5/LB6 may receive an on-the-fly
   prototype only after the one-round design lands; no full LB5/LB6 bivariate LUT survives.
   The production tree remains the checked arity-2/arity-4 topology; binary-only variants
   add substages and child claims without reducing round coefficients.
5. Exact-prefix skipping for derived Stage 1 tables always carries their nonzero default
   and analytically accounts for the omitted equality-weight suffix.
6. `g = min(d_a, d_b, d_d)` is an internal factorization of a flat relation polynomial,
   not proof metadata or a public x/y split.
7. Stage 2 always binds the virtual range-image table to the digit witness. Direct setup
   combines that binding with relation checking; recursive offload combines or sequences
   it with setup certification. There is no Stage 3 in the target protocol.
8. Every current live digit address receives the common balanced range proof. This series
   adds no compressed-commitment or negative-binary proof field, coefficient, or transcript
   slot.
9. Packets before the two-stage cutover remain byte-identical. Against the mandatory
   post-#311 baseline, the cutover preserves the non-terminal direct proof shape and
   atomically changes only recursive-offload proof/transcript shape. The terminal fold is
   already quotient-free and sumcheck-free under #311 and is outside `FoldCheckPlan`.
   Every admitted recursive target must be no larger than its legacy proof in complete
   serialized bytes and must reduce measured verifier work; bytes are a no-regression
   bound, verifier time is the selection objective (setup offloading exists to remove the
   verifier's setup scan, not to save bytes). The descriptor
   selects the shape without a serialized tag or legacy decoder.

The remaining sections are a reference handoff: they define the math, ownership,
deletions, packet order, test oracles, and quantitative stop gates.

### Reader map

| Implementer | Read first | Owned packets |
|---|---|---|
| Range prover / LB4-LB6 kernels | `Stage 1 prover design`, `Exact live prefixes`, performance gates | 3-5, then joint-leaf work in 7 |
| Protocol/proof types | `Two-stage range, relation, and setup design`, degree ledger, transcript order | 2, 6-7 |
| Mixed dimensions/setup | `Mixed ring dimensions`, relation finalization/opening geometry | 6, 8-9 |
| Verifier/security | `Stage 1 verifier design`, transcript order, correctness/security matrix | 1, 7, 11 |
| Planner/serialization | proof-size accounting, planner integration, setup eligibility | 1-2, 7, 9 |
| Upstream digit pipeline | `Upstream prover digit production` | 10 |

Owners may investigate later packets in test/bench branches, but a production PR may not
merge across an open predecessor gate. The central stack ledger below, not branch age or
implementation convenience, determines merge order.

## Non-negotiable outcomes

Replace the current digit range-check implementation with one compact, flat-addressed,
streaming prover architecture. The implementation must do all of the following:

1. Make `DigitRangePlan` the sole authority for roots, product topology, child order, and
   leaf inner degree. Make one composed `FoldCheckPlan` the sole authority for final-leaf
   composition, complete non-terminal proof shape, prover/verifier choreography, and proof-size
   calculation.
2. Delete the split between the compact base-4/base-8 prover and the eager
   base-16/base-32/base-64 tree prover.
3. Never materialize a padded field-valued
   `range_image = digit_witness(digit_witness + 1)` table, all quartic leaves, or all
   product-tree levels.
4. Prove one tree substage at a time from the original compact digits, materialize only
   the current folded child lanes, and drop that state before the next substage.
5. Optimize log bases 4, 5, and 6 (`b = 16, 32, 64`) with measured fixed-lane kernels,
   while retaining one readable implementation rather than a module family per basis.
6. Replace the relation path's public x/y geometry and its `ring_bits == 0` mixed-dimension
   sentinel with one canonical flat Boolean address domain.
7. For recursive setup offload, move the complete linear relation, including trace, into
   the final Stage 1 range subproof. Preserve the product tree before that final subproof;
   do not replace LB4/LB5/LB6 with the paper's direct high-degree range-image polynomial.
   Preserve direct setup's one-pass Stage 2 relation/consistency composition.
8. In recursive-offload mode, replace current Stage 2 with range-image consistency plus
   the current Stage 3 setup product, either as separate proofs or a checked
   independent-domain batch selected by `RecursiveClaimReductionPlan`. Delete Stage 3 as
   a protocol stage and name.
9. Use the common nested ring dimension only as an internal factorization fast path.
   It must not become proof metadata or a second protocol domain.
10. Generalize relation-weight, setup-weight, Stage 1 relation-point, and Stage 2 opening
    boundaries so mixed role dimensions work in both direct and recursive setup modes.
    Recursive offload fails
    closed when its exact committed setup slot or next-level opening route is unavailable.
11. Make the two-stage cutover improve the recursive-offload verifier without regressing
    proof bytes. A balanced
    batched cell is expected to save one extension-field element per common round, but the
    planner must compare complete legacy, separate-target, and batched-target serialized
    sizes for unequal domains. Packet 1 derives them and Packet 7 rejects any target that
    exceeds the matching current recursive shape in complete bytes. Byte parity is
    admissible — when `lambda >= mu` the `Separate` round count exactly ties the legacy
    shape, and a tie must not disable offloading — but every admitted target must reduce
    measured verifier work relative to its legacy recursive baseline.
12. Delete the old architecture as each replacement lands. No compatibility wrappers,
    dormant alternate engines, or `_for_level` pass-through helpers may survive.

This is an implementation handoff, not a menu. The packet order, deletion rules, tests,
and performance gates below are mandatory. If a packet cannot meet both the cleanup and
performance gates, stop and report the evidence instead of retaining a second path.

## Scope and authority

### In scope

- One range-product tree plus direct and recursive-offload final-leaf composers, prover first
  and verifier second.
- Direct Stage 2 relation/range-image consistency; offloaded Stage 1 joint leaf and Stage 2
  range-image/setup claim reduction.
- Flat sumcheck addressing, exact live prefixes, and mixed A/B/D ring dimensions.
- Elimination of the current Stage 3 proof/type/transcript layer.
- Prover-side digit production and fold-grind allocation cleanup after the range/sumcheck
  core has landed.
- Dedicated microbenchmarks, allocation/RSS measurement, proof/transcript differential
  tests, malformed-proof tests, and documentation.

### Explicit non-goals

- Do not implement compressed commitments.
- Do not implement the paper's fused negative-binary range check. This plan reserves the
  exact support/provider seam in range-image consistency without adding a production term.
- Do not change the range proof to use different polynomials for inner, outer, and opening
  digit segments. The current certificate uses one dominating opening basis.
- Do not implement a direct degree-16 or degree-32 range polynomial for log bases 5 or 6.
- Do not add unsafe SIMD. [`packed-sumcheck.md`](packed-sumcheck.md) remains an orthogonal,
  downstream scalar-to-packed optimization.
- Do not create an `akita-protocol` crate, a general `Source`/`Term`/`Expr` descriptor
  algebra, or a trait-object protocol engine.
- Do not fuse Stage 1 and Stage 2 into one proof. In recursive offload, their challenge
  dependency is what makes setup certification possible: Stage 2 starts only after the
  Stage 1 relation point exists.
- Do not redesign the commitment/setup cache system except where mixed-dimension schedule
  eligibility must be modeled and validated.

### Authority order

When sources disagree, use this order:

1. The active tranche base defined below: #311 for F1 through O4c, then #309's
   semantic-basis contract from I5 onward.
2. This handoff for the target implementation and packet order.
3. The Akita paper as mathematical context.
4. Every other unmerged branch as prior art only.

The paper was reviewed through the `paper-note` writing entry `akita`, especially the
basic Akita setting, fold check, implementation details, and verifier-offloading sections.
The useful facts are:

- balanced bases are `4, 8, 16, 32, 64`;
- the paper writes `S(W) = W(W + 1)` and
  `Q_b(S) = product_{k=0}^{b/2-1}(S - k(k + 1))`; this handoff calls the same
  objects `range_image(W)` and `range_image_polynomial`;
- bases above 8 use a short product tree of quartic leaves;
- a future negative-binary layer is still admitted by the common balanced range proof and
  is later narrowed with the same `W(W + 1)` identity on restricted support;
- relation row families may use native ring dimensions and their own
  `(alpha^d + 1)` quotient denominators;
- under verifier offloading (2026-07-19 revision), the relation is fused into the final
  level of the range product tree, all earlier product levels keep their compact
  equality-factored messages, and both witness claims leave Stage 1 at one point;
- setup-prefix provisioning is per level: each offloaded level opens the least committed
  slot covering its own footprint and discharges it at the immediately following fold.

The paper's compressed commitment and negative-binary paths must not be smuggled into this
implementation series. Open PR #295 contains compressed-commitment substrate, but it does
not expand this project's scope.

### Open-PR audit and branch coordination

This audit was refreshed on 2026-07-19 from public pull refs and each direct PR page; the
conflict analysis below uses current #311 head `bc959ef3`. The
open set was #277, #282, #295, #307, #308, #309, #310, and #311. Pull refs alone are not
state evidence; the direct pages reported #277/#282/#295/#307 as draft and
#308/#309/#310/#311 as open.

| PR | Relevant change | Overlap decision |
|---|---|---|
| [#311](https://github.com/LayerZero-Labs/akita/pull/311) | Quotient-free direct terminal relations; removes terminal Stage 1, relation sumcheck, outgoing binding, and numeric terminal stage structure | **Hard prerequisite.** Build on it. `FoldCheckPlan` governs non-terminal `FoldLevelProof` only. Do not touch its direct terminal checker, terminal transcript, or `TerminalLevelProof` payload. |
| [#309](https://github.com/LayerZero-Labs/akita/pull/309) | Splits `log_basis`/digit depths into semantic inner, outer, and opening roles | **Deferred prerequisite for C5, not F1.** F1-C3/O4 consume the already-computed checked range basis and never read `LevelParams`; after #309 the same boundary is sourced from `log_basis_open`. Integrate #309 before flat relation/setup providers need all three role bases. Its reviewed head has 41 conflict paths with #311, mostly schedule regeneration and shared level plumbing, so importing it into F1 would destroy the first PR's bounded diff without changing Stage 1 semantics. |
| [#310](https://github.com/LayerZero-Labs/akita/pull/310) | Distributed multi-chunk recursive setup-offloading schedules and e2e coverage | Land or rebase before Packet 7's distributed rollout. The reviewed heads have only one textual conflict with #311 and one with #309. Preserve its schedules and convert its Stage-3 assertions to semantic Stage-2 claim reduction; do not clone its e2e fixture. |
| [#308](https://github.com/LayerZero-Labs/akita/pull/308) | Planner-backed K16 preset | No algorithmic overlap. Let it land before schedule regeneration; never hand-copy or overwrite its preset rows. |
| [#307](https://github.com/LayerZero-Labs/akita/pull/307) | Shared Jolt field implementation | Broad mechanical overlap but little protocol overlap. New kernels use only repository field/sumcheck traits and add no field implementation. If it lands first, rebaseline before Packet 1; otherwise its branch rebases over this work. |
| [#295](https://github.com/LayerZero-Labs/akita/pull/295) | Compressed-commitment spec and substrate | Intentional non-goal and future integration boundary. Do not edit `layout/compression`, compressed wire, or compression planner modules. Proof-container changes use the canonical post-#311 representation; whichever cutover lands second must adapt directly, with no compatibility wrapper. |
| [#282](https://github.com/LayerZero-Labs/akita/pull/282) | Per-mode profile CI jobs | Keep benchmark capture out of its workflow/mode files. Use existing profile modes and add the dedicated range microbenchmark only after #282's disposition. |
| [#277](https://github.com/LayerZero-Labs/akita/pull/277) | Consolidates PCS integration tests into one binary | No protocol overlap, but it changes every integration-test path. Resolve it before adding e2e files; add tests only at the resulting canonical location and never maintain old/new forwarding test modules. |

`git merge-tree --write-tree` on the audited heads found 41 conflict records between
#311/#309, one between #311/#310, one between #309/#310, 75 between #311/#295, and 70
between #311/#307. Those counts measure textual integration pressure, not semantic scope:
#295 and #307 are broad draft cutovers, while #309 changes the basis vocabulary that the
later relation/setup work must consume. This is why #311 alone is F1's hard base, #309 is
the explicit I5 gate before C5, #310 is a narrow capability integration, and #295/#307 are
boundary constraints rather than code to import here.

The key interaction is #311. It makes this project smaller: the terminal fold already
reveals a segment-typed witness and performs direct ring/trace checks, with no range proof,
relation sumcheck, Stage 2, or Stage 3. Therefore this handoff neither introduces
`TerminalRelationProof` nor claims a terminal degree-3-to-degree-2 saving. Its proof-size
comparison covers non-terminal folds only.

The F1 branch is stacked directly on the audited #311 head and may implement the locked
surface below without #309. The fixed integration policy is:

1. Keep #311 as the hard base. Its audited head is `bc959ef3`; if it advances, repeat the
   direct-page, changed-file, and terminal-contract audit before rebasing this specification.
2. F1 through O4c accept the checked range basis already carried by the ring-switch
   output. `DigitRangePlan` must not read `LevelParams`, so #309 later changes only the
   upstream source of that value from the uniform basis to `log_basis_open`.
3. Before C5, rebase and land #309 at audited head `d4100f3f` (or its refreshed head),
   resolving terminal conflicts in favor of #311 and recording the resulting I5 base.
   From C5 onward, relation/setup code consumes semantic `log_basis_inner`,
   `log_basis_outer`, and `log_basis_open` directly; do not restore a largest-basis rule.
4. Prefer landing/rebasing #310 and the small #308 preset before Packet 7 schedule
   regeneration. If #310 is still pending, core Packets 0-6 may proceed, but Packet 7 may
   not claim distributed recursive coverage or add a duplicate fixture.
5. Resolve #277 before adding integration-test files and #282 before editing profile-mode
   CI. These are path/ownership gates, not reasons to block kernel development.
6. Do not merge or wholesale cherry-pick `origin/quang/relation-weight-kernel-cutover`.
   The accepted prior-art snapshot is `e87295b7` and its
   `specs/sumcheck-kernel-cutover.md`. Port only its flat relation polynomial,
   pair-stream, fused fold-scan, and isolated initial-round batching ideas against the
   then-current layout; this handoff supersedes that snapshot as an implementation
   authority.
7. Do not wait for or merge `origin/refactor/universal-digit-fast-layout`. After the flat
   range/relation architecture lands, port the relevant verifier arithmetic from
   `d2bfda96` (initial fast accumulation) and `57fb0025` (prefix-scan/carry-bucket
   summary), using `8a86908f` as the canonicalized form, in an isolated final slice.
8. Do not begin Stage 1/2 packed SIMD work until the scalar architecture, mixed dimensions,
   and all scalar gates in this document pass.

### Intended implementation diff surface

The implementation PR is deliberately a non-terminal fold-check cutover, not a general
PCS rewrite. The following table is normative. A touched production file outside these
responsibilities requires an explicit amendment to this section in the same PR; nearby
cleanup is not sufficient justification.

| Surface | Intended ownership |
|---|---|
| `akita-sumcheck` | Only checked pair iteration, default-aware exact-prefix folding, and ordinary fused fold/next-scan mechanics. No Akita roots, relation terms, transcript labels, or stage enums. |
| `akita-prover::protocol::sumcheck` | Replace current Stage-1 tree, Stage-2 relation/range binding, and Stage-3 setup/carry modules with `digit_range`, `direct_relation`, `claim_reduction`, and `relation_weight`. |
| `akita-prover::protocol::core::fold` | Non-terminal orchestration only: construct one `FoldCheckPlan`, call the selected concrete composer directly, and emit the post-#311 `NextWitnessBinding`. No terminal-witness construction or terminal transcript changes. |
| `akita-types::layout` / setup contribution | Canonical `DigitRangePlan`, `WitnessDomain`, typed points, semantic relation events/providers, `SetupCoefficientDomain`, and one `SetupContributionPlan`. |
| `akita-types::proof` / proof sizing | Change only the non-terminal `FoldLevelProof` fold-check payload, its shape/wire/validation, and its exact byte formula. Preserve #311's `TerminalLevelProof`, `TerminalLevelProofShape`, and terminal wire bytes. |
| `akita-verifier` | Replay the two non-terminal topologies and mixed providers; preserve #311's `terminal_direct.rs`, `terminal_ntt.rs`, and terminal section of suffix verification unchanged. |
| planner/config/schedules | Add authenticated non-terminal topology and reduction selection, exact complete-size scoring, eligibility, and regenerated artifacts. Preserve unrelated presets and do not hand-edit generated rows. |
| tests/benches/docs | Differential range/relation/setup tests, the resulting canonical integration-test location, a dedicated range microbenchmark, and documentation required by the cutover. |

Explicitly outside the diff surface:

- #311 direct-terminal algebra, terminal proof shape, terminal transcript order, terminal
  schedule topology, and `tail-wire-encoding` semantics;
- #295 compression planners, compression layouts, compressed proof wire, and compression
  replay;
- field arithmetic, packed-field backends, NTT/CRT kernels, and dependency replacement
  from #307;
- #282 workflow partitioning and existing profile-mode selection;
- #308's preset constants except mechanically regenerated derived artifacts;
- commitment algorithms, setup persistence, and witness chunk topology. Packet 10 may
  replace internal temporary-to-destination emission while preserving the exact existing
  layout/wire; recursive claim reduction may add only its required opening-route metadata.

At each packet review, run `git diff --name-only IMPLEMENTATION_BASE...HEAD` and classify
every path against this table. Generated schedules are reviewed as generated output, not
as authority. No packet may acquire a second implementation merely to reduce merge
conflicts with an open PR.

### Single sources of truth and the no-wrapper rule

The repository policy applies literally. This design does not authorize facade layers,
aliases, or duplicated formulas:

| Concept | Sole production authority | Forbidden duplication |
|---|---|---|
| Balanced range map and roots | `akita-types` digit-range arithmetic used directly by plan, prover, and verifier | local `S` functions, verifier copies, or basis-specific root builders |
| Range topology/degree/child order | one checked `DigitRangePlan` constructor | `_for_level` helpers, verifier reconstruction, per-LB plan types, or copied shape tables |
| Complete non-terminal topology | one checked `FoldCheckPlan` constructor | independent prover/verifier topology inference, runtime fallback, or serialized variant tag |
| Relation polynomial | one semantic relation-event emitter and one `RelationWeight` plan | direct/offload emitters, x/y adapters retained as production APIs, or trace copies |
| Setup polynomial | one `SetupContributionPlan` over one `SetupCoefficientDomain` | high-lane partial claims, separate direct/recursive weight builders, or slot coercion |
| Proof shape and bytes | `FoldCheckPlan`-derived non-terminal proof shape plus serialization-parity tests | free-standing degree/round formulas in planner, prover, and verifier |
| Mechanical scan/fold | the small `akita-sumcheck` functions, called directly | `Engine`, `Source`, `Composition`, forwarding traits, or stage-specific wrappers |

The three semantic prover entry points contain real logic:
`RangeOnlyLeafProver`, `RangeRelationLeafProver`, and
`DirectRelationRangeImageProver`. Orchestration pattern-matches `FoldCheckTopology` and
calls the selected entry point directly. Do not add `prove_stage1_for_level`,
`prove_fused_first_round`, compatibility constructors, pass-through re-exports, or a
generic `FusedKernel`. A type method may assemble its fields into the canonical call, but
it may not own another copy of the equation, degree, point order, or shape.

## Reconciliation with the concurrent plan

A concurrent planning pass reached the same high-level priorities and presented them more
clearly. Its strongest ideas are incorporated here rather than maintained as a second
implementation authority:

- lead with the three concrete problems and a good-path/bad-path contrast;
- include a line-counted file inventory so "cleanup" has a measurable ownership closure;
- treat proof/transcript byte identity as a per-slice gate;
- encode basis specialization mostly in validated table data, with bespoke code only when
  a complete benchmark justifies it;
- make the dependency graph, risk register, and unresolved experiments visible;
- use `WitnessDomain` as the public name for the generalized address contract.

Several attractive parts of that plan need a narrower or different design. The following
resolutions are final for this handoff:

| Concurrent proposal | What is strong about it | Resolution here |
|---|---|---|
| One generic eq-factored "composed-witness" engine for Stage 1, the product tree, and Stage 2 | Correctly notices extensive duplicated pair iteration, reduction, and folding | Share only direct `scan_adjacent_pairs`, exact-prefix, accumulator, fold, and parallel-reduction functions. Current Stage 1 product layers are eq-factored while current Stage 2 is standard; the target adds a standard joint Stage 1 leaf and either eq-factored or standard Stage 2 reduction. Their equations, degree accounting, drivers, and transcript choreography remain explicit. |
| `RoundWitnessSource::Value` plus `fold(&mut self, r)` | Tries to hide compact-versus-field state from callers | A Rust associated type cannot change from compact `i16` to field `E` after a challenge. Do not add an enum branch to every corner to rescue the abstraction. Stage-owned, statically typed compact and field phases call the same mechanical scan functions. |
| `is_zero_pair` and implicit-zero suffix skipping | Essential for avoiding padded witness scans | Sound for the digit witness and every linear relation term that contains it, but not for derived range tables. Most quartic leaves and randomized product batches have nonzero values at padded `range_image = 0`. Use `ExactPrefixTable` with a per-lane default and add the exact omitted `GruenSplitEq` suffix mass. |
| Build compact per-class leaf tuples | Uses the small `RangeImageClass` alphabet and removes repeated Horner evaluation | Adopt as the source of fixed-lane tables, but do not describe the current tree as `b/2` quartic leaf tables: it has `b/8` quartic leaves. Materialize only the current 2/4/8-lane substage state, never the full forest. |
| Extend two-round prefix tables to LB4/LB5/LB6 | Correctly identifies first-round compactness as an optimization lever | Make the one-round streaming kernel the baseline. Four-class key spaces grow from 4,096 at LB4 to 65,536 at LB5 and 1,048,576 at LB6 before field-coefficient payloads. Benchmark a bounded LB4 table; for LB5/LB6 permit only an on-the-fly experiment after the baseline wins. |
| Model mixed dimensions as ragged `WitnessDomain` axes and choose common-prefix versus segment-concatenated binding later | Removes hard-coded x/y names and recognizes the common low dimension | Do not leave the protocol domain unresolved. Bind the existing flat physical address LSB-first. Compile role-local row-family algebra into a semantic flat `RelationWeight`; use `g = min(d_a,d_b,d_d)` only to factor that polynomial internally as `A_g(y) M(x)` plus checked fringes. |
| Select a per-segment alpha vector in the relation prover | Recognizes that one global `alpha_evals_y` is invalid for mixed role dimensions | A per-pair segment branch is not the semantic contract and does not solve folding or verifier evaluation. Define row-family events at native dimensions, then give dense and common-base providers the same flat-polynomial semantics in the final Stage 1 subproof. |
| Reserve `binary_support: Option<RestrictedEqWeights>` in a production composition | Anticipates the paper's later fused negative-binary path | Do not add inactive production state or transcript ambiguity. Reserve only an internal additive-provider seam in Stage 2 range-image consistency reduction. A future restricted term must be the single MLE of the pointwise restricted equality table, not the product of two MLEs. |
| Schedule-frequency table and one profile capture | Establishes that LB4/LB5/LB6 deserve first-class attention | Keep the conclusion, not the numbers as performance evidence. Counts must define whether they refer to inner, outer, opening, or certified range basis after Packet 0; configuration frequency is not runtime weight. The captured run reached verifier output rather than isolating prover phases, so Packet 1's benchmark work remains mandatory. |
| Keep the current two-stage placement while cleaning Stage 2 | Preserves the direct prover's efficient single signed-witness scan | Retain that placement for `DirectSetup`, but it cannot reduce recursive-offload proof size. Add the paper-motivated `RecursiveSetupOffload` composer, move relation checking into its final Stage 1 leaf, and delete numeric Stage 3. |
| Describe mixed dimensions primarily as a layout abstraction | Correctly finds that the core Boolean sumcheck is already one-dimensional | Insufficient for setup offload: setup coefficients form an independently committed flat domain, and role-native algebra must compile to one semantic relation/setup weight. Add `SetupCoefficientDomain`, typed points, full setup-contribution semantics, and exact opening routes. |

The concurrent plan's prose and workstream structure were stronger than the original
draft, and this handoff adopts that presentation style. Its weakest points were the two
generic traits, under-specified nonzero derived tails, a per-segment-alpha mixed design,
and lack of a paper-driven offloading proof-shape cutover. Conversely, the first
reconciliation draft overcorrected by forcing relation-in-Stage-1 onto direct proofs; the
prover audit showed that this would add a signed-witness fold and scalar for no benefit.

The final result is intentionally less generic but more precise: one flat witness domain,
one streaming range-product implementation, two small schedule-bound final-leaf composers,
one separate setup-coefficient domain, a small shared mechanics layer, and basis-specific
data feeding fixed-lane kernels.

## Terminology

- `LB`: base-2 logarithm of the balanced digit basis.
- `b = 2^LB`: digit basis.
- Current balanced digit range: `[-b/2, b/2 - 1]`.
- `digit_witness` / `W`: the committed balanced-digit witness polynomial. Use the full
  `digit_witness` name in code; reserve `W` for equations and short local algebra.
- `range_image(digit) = digit(digit + 1)`: the degree-halving map. This replaces every vague
  production identifier named `S`, `s_table`, `norm`, or `squared`.
- `RangeImageClass`: the collision class of a balanced digit under `range_image`:

  ```text
  range_image_class(w) = w       when w >= 0
                       = -w - 1  when w < 0

  range_image(w)
    = range_image_class(w) * (range_image_class(w) + 1),
      0 <= range_image_class(w) < b/2.
  ```

- `range_image_polynomial(p) = product_k (p - k(k + 1))`: the vanishing polynomial
  over range-image values. The paper calls this `Q_sq`; production code does not.
- `range_image_eval`: an MLE evaluation of the pointwise Boolean table
  `range_image(digit_witness[z])`. It is generally **not** equal to
  `digit_witness_eval * (digit_witness_eval + 1)` away from Boolean points.

- `N`: full Boolean domain length for a sumcheck table.
- `L`: exact live prefix length, `0 < L <= N`.
- `g`: common nested role dimension,
  `g = gcd(d_a, d_b, d_d) = min(d_a, d_b, d_d)` for the supported power-of-two tuples.
- **Protocol domain**: the sole flat physical coefficient address, LSB-first.
- **Local view**: a checked internal `(column, coefficient)` or common-base lane view used
  by a kernel. A local view is never a wire-level coordinate system.

### Challenge and transcript vocabulary

Use semantic names and labels in production; reserve Greek letters for equations:

| Production name | Meaning |
|---|---|
| `range_product_batch_challenge` | batches child claims between range-product layers |
| `direct_range_binding_challenge` | binds the direct range-image claim into the Stage 2 relation proof |
| `range_relation_batch_challenge` | batches the recursive final range leaf with the linear relation claim |
| `range_image_binding_challenge` | merges recursive `range_image_eval` and `digit_witness_eval` for consistency |
| `setup_range_binding_batch_challenge` | batches native setup and range-image consistency claims in recursive `Batched` only |

Do not reuse one label merely because two challenges are both colloquially called
`gamma`, `theta`, or `batch_challenge` in local algebra.
Follow [`transcript-hardening.md`](transcript-hardening.md): semantic labels and
`LoggingTranscript` frames are diagnostic in production builds, while positional order
plus the absorbed instance/protocol descriptor carries Fiat-Shamir domain separation.
Therefore Packet 7 must bind the version/topology/shape descriptor explicitly; a pretty
label string is not a security boundary.

## Current-state audit

The line counts and call paths in this section describe planning base `f5c180a4` so the
spaghetti/deletion baseline remains reproducible. They are not the implementation base.
Before Packet 1, refresh the inventory on #311, remove every terminal path already deleted
by #311 from this project's ownership closure, and ratify new F1 line counts without
weakening the percentage/absolute deletion gates. Refresh the relation/setup inventory
again at I5 after #309 lands.

### End-to-end call path

The relevant prover path is:

```text
batched_prove
  -> prove
  -> prove_root
  -> prepare_fold_inner / finish_prepared_fold
  -> RingRelationProver::new
  -> ring_switch_build_w
  -> commit_w
  -> ring_switch_finalize
  -> prove_stage1
  -> prove_stage2
  -> optional prove_stage3 / terminal suffix
```

Primary files:

- `crates/akita-prover/src/protocol/core/{prove,root_fold,fold}.rs`
- `crates/akita-prover/src/protocol/ring_relation.rs`
- `crates/akita-prover/src/protocol/fold_grind.rs`
- `crates/akita-prover/src/protocol/ring_switch/{coeffs,finalize}.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage1_tree.rs`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage1/`
- `crates/akita-prover/src/protocol/sumcheck/akita_stage2/`
- `crates/akita-verifier/src/stages/{stage1,stage2,stage3}.rs`
- `crates/akita-verifier/src/protocol/{ring_switch,core/fold}.rs`

The snapshot's production surface is large enough that file moves alone are not a cleanup
metric. These line counts exclude test modules unless stated otherwise and define the
initial ownership closure Packet 1 must reproduce mechanically:

| Area | Production lines | Main responsibility today |
|---|---:|---|
| `akita_stage1_tree.rs` | 696 | large-basis dispatch, eager tree construction, product and leaf substages |
| `akita_stage1/` | 2,433 | compact/dense Stage 1 lifecycle plus sparse, prefix, and round-2 kernels |
| `akita_stage2/` | 2,939 | fused Stage 2 lifecycle plus dense, x/y-prefix, and round-2 kernels |
| `two_round_prefix/` | 1,836 | shared LUT/reconstruction mechanics and stage-specific wrappers |
| verifier `stages/stage1.rs` | 354 | product/leaf proof replay |
| verifier `stages/stage2.rs` | 441 | fused relation replay and final claim |
| types `proof/stage1.rs` | 200 | topology, coefficients, point reorder, and shape helpers |

Packet 1 must record symbol-level ownership as well as lines. The end-state test is not a
fixed percentage reduction: it is deletion of duplicate semantic authorities and layout
branches, with any retained line justified by one invariant-bearing responsibility.

### Stage 1: two implementations and an eager forest

There are two prover types named `AkitaStage1Prover`:

- the exported wrapper/tree implementation in `akita_stage1_tree.rs`;
- the older compact implementation in `akita_stage1/mod.rs`.

For `b <= 8`, the wrapper delegates to the compact backend. For `b > 8`, it calls
`padded_s_table`, expands every compact digit into a padded field-valued
paired-digit-product table (the legacy code calls it `S`), builds
all quartic leaf tables, builds every product layer, and retains nested
`Vec<Vec<Vec<E>>>` state while proving the substages.

Current product shapes are:

| LB | Basis | Product stages | Leaf stage |
|---:|---:|---|---|
| 2 | 4 | none | degree 2 |
| 3 | 8 | none | degree 4 |
| 4 | 16 | one parent, arity 2 | two quartic leaves |
| 5 | 32 | one parent, arity 4 | four quartic leaves |
| 6 | 64 | one parent, arity 2; two parents, arity 4 | eight quartic leaves |

Approximate digit-range-owned field-table peaks, excluding split-equality state,
round-polynomial temporaries, and allocator metadata, are:

| LB | Current peak |
|---:|---:|
| 4 | `3N` (paired-digit table plus two leaves) |
| 5 | `5N` (paired-digit table plus four leaves) |
| 6 | `11N` (paired-digit table, eight leaves, two intermediate parents) |

The compact backend contains good ideas—compact first rounds, small lookup tables,
implicit prefixes, fused fold/next-round scans, and unreduced accumulation—but its
implementation is organized around `ring_bits`, x/y phases, sparse-y, x-prefix, and a
large shared two-round-prefix module. Extending those branches basis by basis would grow
the cartesian product rather than fix it.

### Stage 1 handoff and verifier issues

`prove_stage1` currently:

- calls the verifier-reachable `reorder_stage1_coords`, whose shared helper uses
  `assert_eq!`;
- passes six raw geometry values;
- `mem::take`s the compact witness;
- calls `prove_recover_w` only to restore the same `Arc<[i8]>` for Stage 2.

The verifier reconstructs tree topology separately, clones tau and child-claim vectors,
and duplicates transcript absorption. Proof-size code separately calls the loose tree
shape helpers. `validate_stage1_tree_basis` accepts every power-of-two basis at least 4,
even though current configuration and i8 witness bounds stop at 64.

### Stage 2: correct fusion, layout-driven spaghetti

Stage 2 currently scans the compact digit witness, combines the linear ring relation/trace
terms with the virtual range-image binding, and moves to field storage only after
challenges. Its fusion is locally efficient and remains the direct-setup composer. It
fixes the relation point too late for the paper's setup-offloading placement, so Packet 7
moves the linear half to Stage 1 only for recursive offload and leaves range-image
consistency plus setup certification in Stage 2.

Its implementation is nevertheless a cartesian product of:

- compact versus field witness;
- prefix-y versus prefix-x versus dense scans;
- relation factor versus trace side table;
- one-round versus two-round-prefix logic;
- fused fold/next-round versus separate fold;
- uniform x/y geometry versus flattened `ring_bits == 0` fallback.

`RingSwitchOutput` exposes an eight-field raw bundle whose relation-weight field has two
different meanings: a per-column `M(x)` in uniform mode and a full flat vector in mixed
mode. `prove_stage2` then takes roughly a dozen raw arguments and clones the optional trace
table.

### Mixed dimensions: partial algebra, incomplete sumcheck consumers

Current main already has `CommitmentRingDims`, role-local relation-weight construction,
row-family quotient denominators, and `SetupProjectionGeometry`. But:

- `ring_switch_finalize` uses x/y only when all role dimensions are uniform;
- nonuniform roles set `ring_bits = 0` and eagerly materialize the flat relation weights;
- Stage 2 verifier logic still splits challenges using one `ring_bits`;
- trace remapping is tied to a uniform ring dimension and can allocate a remapped table;
- Stage 3 slices the Stage 2 point using `d_a` bits in places where the common base `g`
  is required;
- recursive setup-prefix storage is fixed to `SETUP_OFFLOAD_D_SETUP = 64`;
- planner-generated levels still use uniform role dimensions.

The existing `mixed_d_per_level` tests vary a homogeneous `D` between levels. They do not
test different A/B/D dimensions inside one level.

### Upstream digit production

`ring_switch_build_w`, fold-grind, and decomposition still perform avoidable copying:

- nested centered coefficient vectors;
- temporary digit planes copied into the final witness;
- recomposition followed by decomposition;
- per-chunk sparse challenge windows;
- field conversion before a grind nonce is accepted;
- large terminal clones retained only for later encoding on the planning-base code.

This is secondary to the Stage 1 allocation problem but is part of the prover digit
pipeline and receives a later packet below. Packet 1 removes any terminal clone already
resolved by #311 from this project's ownership rather than re-editing that path.

## Target architecture

### Layer boundaries

```text
Level/schedule metadata
  -> FoldCheckPlan
       -> DigitRangePlan + WitnessDomain + relation/setup plans
       -> prover: compact digit source + flat pair/fold kernels
       -> verifier: shared semantic plans + closed-form evaluators
       -> proof sizing/serialization: shared shapes

Row-family algebra at native d_A/d_B/d_D
  -> checked flat-address semantic emitter
       -> dense exact-prefix oracle
       -> common-base/factorized prover provider
       -> prepared verifier evaluator

DirectSetup topology
  -> Stage 1 range-only leaf
  -> Stage 2 relation + range-image consistency
  -> direct full setup replay at Stage 2 point

RecursiveSetupOffload topology
  -> Stage 1 joint leaf fixes RangeRelationPoint
  -> full setup contribution closes deferred Stage 1 relation
  -> Stage 2 separate or batched setup/range-image reduction
       -> NextWitnessPoint + SetupOpeningPoint
```

`akita-types` owns validated semantic plans and layouts. `akita-sumcheck` owns only
protocol-independent Boolean sumcheck drivers, split equality, polynomial proof formats,
and field fold/accumulator mechanics. Akita-specific range and relation algebra stays in
the Akita prover/verifier crates. No new crate is needed.

### `DigitRangePlan`: one range topology authority

Add a field-independent, validated plan under `akita-types`, with an API equivalent to:

```rust
pub struct DigitRangePlan {
    log_basis: u8,
    stages: FixedRangeStages,
}

pub enum DigitRangeStage {
    Product {
        arity: u8,
        parent_count: u8,
        child_range: Range<u8>,
    },
    Leaf {
        degree: u8,
        leaf_count: u8,
    },
}
```

F1 constructs this plan from the checked concrete basis already carried by the
ring-switch output (`b in {4,8,16,32,64}`) and derives `log_basis`; it does not accept or
inspect `LevelParams`. I5/#309 changes the producer of that concrete basis to
`log_basis_open` without changing the `DigitRangePlan` API.

The exact storage type may use arrays plus counts rather than a new small-vector
dependency. `basis()` and `half_basis()` are checked derived accessors; do not store
independently constructible copies of those values. The constructor is private to the
defining module and no external struct literal may bypass it. The contract is fixed:

- accept exactly `LB in 2..=6`;
- derive the integer roots `k(k + 1)` and quartic leaf grouping;
- expose the five explicit production topologies in the table above;
- derive ordered subproof degree, parent count, child count, and claim order;
- derive its range-layer shapes for a supplied Boolean round count;
- validate a proof's subproof/child shape without allocation;
- provide narrow semantic operations such as batching child claims and leaf coefficients.

The plan does not own field-valued hot lookup tables. Store small leaf polynomial
coefficients as checked integer arrays and lift them at the prover/verifier field boundary.

After migration, delete or make private and absorb:

- `validate_stage1_tree_basis`;
- `stage1_tree_product_stage_arities`;
- `stage1_tree_stage_shapes`;
- `stage1_stage_count`;
- generic public `combine_polys`, `linear_combination`, and range-only `eval_poly` exports;
- unused base-field-only `absorb_interstage_claims`.

Prover, verifier, serialization shape validation, and tests call the same plan for range
topology. Complete proof sizing calls `FoldCheckPlan`; no call site may reconstruct
arities from `b.trailing_zeros()`.

### `FoldCheckPlan`: one complete proof-shape authority

Compose range topology with the scheduled relation placement and final-leaf shape:

```rust
pub struct FoldCheckPlan {
    digit_range: DigitRangePlan,
    witness_domain: WitnessDomain,
    topology: FoldCheckTopology,
}

pub enum FoldCheckTopology {
    DirectSetup(DirectRelationRangeImagePlan),
    RecursiveSetupOffload {
        relation_leaf: RangeRelationLeafPlan,
        claim_reduction: RecursiveClaimReductionPlan,
    },
}

pub enum RecursiveClaimReductionPlan {
    Separate(SeparateReductionShape),
    Batched(SetupWitnessBatchGeometry),
}
```

This is the final form only if E8 is admitted. Through C7, the recursive topology owns
`SeparateReductionShape` directly and there is no one-variant
`RecursiveClaimReductionPlan`; E8 introduces the enum only when `Batched` becomes a real
second production shape.

The descriptor authenticates this choice before proof messages. Headerless serialization
uses the authenticated shape and carries no enum tag. Prover, verifier, level/root proof
shape, proof-size calculation, transcript schedule, and allocation caps all consume this
same plan. `DirectSetup` and `RecursiveSetupOffload` are two small semantic composers over
one range-product implementation and one provider toolkit, not two range engines.
Nested composer plans are privately constructed checked views. They never store another
`DigitRangePlan`, `WitnessDomain`, round count, or independently supplied degree; all such
values are derived from the parent.

### `WitnessDomain`: one flat address contract

Add one checked domain/layout authority, conceptually:

```rust
pub struct FlatBooleanDomain {
    live_len: usize,
    domain_len: NonZeroPowerOfTwo,
    num_vars: usize,
    variable_order: LsbFirst,
}

pub struct WitnessDomain {
    flat: FlatBooleanDomain,
    segments: Vec<WitnessSegment>,
    transcript_point_map: TranscriptPointMap,
}

pub struct WitnessSegment {
    flat_range: Range<usize>,
    local_ring_dim: Option<usize>,
    row_family: Option<RelationRowFamily>,
}
```

Requirements:

- `domain_len = checked_next_power_of_two(live_len)`;
- the protocol address is the raw physical coefficient index in `0..domain_len`;
- addresses `live_len..domain_len` are exact public zeroes for the wire witness;
- segments are sorted, nonoverlapping, within `live_len`, and use checked local embeddings;
- production bind order is fixed to raw address bits `0, 1, ...` (LSB first);
- `transcript_point_map` explicitly maps existing challenge draw slots to those physical
  coordinates and reproduces the current homogeneous order; no caller slices or reorders
  a point by convention;
- local ring dimensions belong to segments/row families, not to the protocol domain;
- a segment carries only the identity of its row family; every piece of row-family
  algebra (weights, denominators, exponent patterns) stays in the relation plans, so
  `WitnessDomain` never grows relation semantics through the back door;
- current uniform layouts are the special case, not a separate public API.

Additive range obligations are a future design constraint, not a production field in this
series. The current constructor defines every live compact-witness address as common
balanced range; Stage 1 and Stage 2 therefore refer to the same unmasked pointwise
`range_image(digit_witness)` table. A future compressed negative-binary segment also has the common
obligation and may add a restricted-negative-binary obligation. Do not accept a live
field/unchecked address in this digit-witness domain until a separate feature adds explicit
obligation metadata and implements the support-masked Stage 1/Stage 2 construction
described below. This matches the paper and avoids the stale design error of removing
negative-binary digits from the common range proof.

Stage 1 receives only the compact witness, `DigitRangePlan`, `FlatBooleanDomain`, and
the ordered equality point. It must not receive `live_x_cols`, `col_bits`, or `ring_bits`.

Migration is atomic at the full Stage 2 boundary. Packets 2-5 add only the
`FlatBooleanDomain` plus a checked transcript-slot-to-physical-coordinate map local to
Stage 1. They do not add segmented `WitnessDomain`, replace `RingSwitchOutput`, or
duplicate the public x/y handoff. Packet 6 introduces the complete `WitnessDomain`,
replaces that raw handoff, and deletes the public x/y constructors in the same packet. No
adapter or dual geometry may survive a packet boundary.

### Exact live prefixes and nonzero derived tails

Introduce one internal checked table mechanic, equivalent to:

```rust
pub struct ExactPrefixTable<T> {
    domain_len: NonZeroPowerOfTwo,
    explicit: Vec<T>,
    default: T,
}
```

Its semantics are `explicit[i]` for `i < explicit.len()` and `default` otherwise. Adjacent
affine fold:

- handles one odd boundary pair against `default`;
- materializes `ceil(explicit.len() / 2)` outputs;
- halves `domain_len`;
- keeps `default` unchanged because folding `(default, default)` is constant;
- returns errors on invalid sizes instead of panicking.

This is a kernel mechanic, not a vague protocol-wide `TailPolicy`.

For every linear relation or witness-reduction term, the witness default is zero, so the
omitted all-default suffix contributes zero. For derived range terms:

- `range_image` defaults to zero;
- a quartic leaf defaults to its polynomial evaluated at paired-digit value zero;
- only the leaf containing the zero root necessarily defaults to zero;
- a product node defaults to the product of its child defaults;
- an interstage-randomized leaf or parent batch generally has a nonzero default.

Therefore a truncated scan must analytically add the fully implicit suffix. Add a checked
`SplitEqSuffixMass` helper that consumes the actual remaining `GruenSplitEq` tables. If
the first omitted pair index is `P`, current loops index
`j = j_high * num_first + j_low`; compute:

```text
h0 = P / num_first
l0 = P % num_first

mass_from(P) =
    e_second[h0] * sum(e_first[l0..])
  + sum(e_second[h0 + 1..]) * sum(e_first[..])
```

with the aligned-boundary and end cases checked explicitly. Cache prefix/suffix sums for
the current round. `SplitEqSuffixMass` operates only on
`GruenSplitEq::remaining_eq_tables()`; it must not include `current_scalar` or the current
linear equality factor, which are owned by the eq-factored driver. The remaining tables
each sum to one by the current `GruenSplitEq` invariant, so `1 - prefix_mass` is
algebraically valid, but prefer the explicit formula above because its indexing and
boundary cases are auditable. Multiply the omitted equality mass by the constant local
round polynomial produced by the derived lane defaults exactly once.

Property-test `ExactPrefixTable` and suffix mass against fully padded materialization for
every short live length, odd/even boundaries, random nonzero defaults, and every bind.

## Stage 1 prover design

### Canonical compact source

The prover keeps the original `Arc<[i8]>` for the complete Stage 1/Stage 2 lifetime. Before
any Stage 1 transcript mutation, construct a `ValidatedDigitSource`. Its constructor scans
the immutable live prefix once and returns `AkitaError` for an out-of-range digit. All
subsequent class access is infallible under that invariant. At the Stage 1 boundary:

1. Validate every live digit against the plan's balanced interval.
2. Convert it to a `RangeImageClass` byte using `k = w` for nonnegative `w`
   and `k = -w - 1` otherwise.
3. Treat every wire-padding address as class zero without storing it.
4. Build at most one compact `RangeImageClass` buffer if repeated class conversion is measured to
   dominate; fuse its construction with validation. Otherwise compute the class while
   scanning the already-validated original digits. Retain the faster measured choice, not
   both.

Do not allocate a field-valued paired-digit-product table. Do not consume and return the `Arc` through a
`prove_recover_w` ownership dance. The range prover borrows or cheaply clones the shared
`Arc`. Direct Stage 1 outputs its range proof and `range_image_eval`. Recursive-offload
Stage 1 outputs `DigitRangeRelationProof`, one `RangeRelationPoint`,
`range_image_eval`, and `digit_witness_eval`.

### Tiny node-by-class tables

For each substage, derive small field-valued tables from the plan:

```text
node_value[class][lane]
```

where a lane is one immediate range-product child of the current substage. The largest
class count is 32 and the largest product-stage lane count is 8. A class-indexed table may
produce only range-derived values: `digit` and `-digit-1` share one `RangeImageClass` but
have different signed witness values. The recursive joint leaf therefore gets
`range_image` from the class table and loads the signed `digit_witness` lane directly from
`ValidatedDigitSource`.

Build only the current substage's table. The table's default lane record is the same
`node_value[0]`, because padding means `digit_witness = 0`, hence
`RangeImageClass = 0`. Interstage batching
weights are sampled before constructing the next substage's local round tables, so combine
parent/leaf coefficients once rather than carrying a dynamic weight multiply in the inner
loop.

### First implementation: one-round streaming

Every high-basis product substage follows this lifecycle:

1. **Round 0 compact scan.** Iterate logical adjacent address pairs. Load two classes,
   using class zero at the boundary/padded endpoint. Look up all immediate child lane
   values and accumulate exactly the current eq-factored round polynomial.
2. **Tail accounting.** Add fully implicit pair contributions with
   `SplitEqSuffixMass`; do not scan or allocate the padded suffix.
3. **Challenge 0.** After the transcript supplies `r_0`, rescan the compact class pairs
   and materialize only their folded child lane records.
4. **Address-major state.** Store
   `values[folded_address * LANES + lane]`, logically `Vec<[E; LANES]>`, with the
   lane-default record carried separately by `ExactPrefixTable`.
5. **Later rounds.** For every round, scan adjacent explicit lane records, add the fully
   implicit suffix with the current `SplitEqSuffixMass` and lane defaults, and then fold
   the lane records in place. Where profitable, fuse the fold with the next round's
   explicit-prefix scan while the records are cache-hot.
6. **End of substage.** Read the final child claims in canonical plan order, absorb them,
   derive interstage batching weights, and free the lane buffer before constructing the
   next substage.
7. **Next substage.** Rescan the original compact classes at its new equality point. Never
   retain the previous tree layer as a shortcut.

Packets 4-5 land the range-only leaf behind its byte-identical oracle. Packet 7 retains it
for direct setup and adds the recursive-offload **range-and-relation leaf composer**
without changing the preceding product layers:

1. Combine all quartic leaf coefficient arrays with the interstage weights.
2. Sample `range_relation_batch_challenge` only after the leaf input claim and
   `linear_relation_claim` are transcript-bound.
3. Compute round 0 directly from compact digit pairs. Accumulate the anchored leaf
   polynomial over the virtual range-image table and the linear relation/trace term over
   the digit witness in the same pass.
4. After `r_0`, materialize exactly two folded lanes at `N/2`:
   `range_image` and `digit_witness`. Fold the `RelationWeight` provider through
   its own closed representation.
5. Use one standard round kernel thereafter. The range term has degree
   `leaf_degree + 1` because of its equality factor; the relation term has degree 2.
   Account analytically for the nonzero implicit range suffix; the padded linear relation
   suffix is zero.
6. Return `range_image_eval` and `digit_witness_eval` at the same
   `RangeRelationPoint`.

The standard joint leaf cannot blindly reuse the equality-factored suffix rule. For an
implicit padded pair, `digit_witness = 0`, so the relation contribution is zero, while the
range contribution is:

```text
current_eq_scalar
  * current_eq_linear(t)
  * remaining_eq_suffix_mass
  * LeafBatch(range_image(0)).
```

The joint-leaf caller may reuse the remaining-table calculation from
`SplitEqSuffixMass`, but it must restore the current equality scalar and current linear
factor exactly once. Differentially test both points `t=0,1`, odd live prefixes, and
nonzero randomized `LeafBatch(0)`.

The earlier product layers keep the current topology, `EqFactoredUniPoly` normalization,
`range_product_batch_challenge` samples, and child-claim order. Only the recursive-offload final leaf
changes wire format at Packet 7: LB2 uses a degree-3 standard message; LB3 and every
quartic tree leaf use a
degree-5 standard message. This one-coefficient-per-round increase is smaller than a
separate relation sumcheck and preserves the high-basis product tree that the paper's
direct combined polynomial would discard.

Expected peak current-substage table occupancy is:

| LB | Current | Mandatory one-round target | Later two-round target |
|---:|---:|---:|---:|
| 4 | `3N` | `N` (two lanes at `N/2`) | `N/2` |
| 5 | `5N` | `2N` (four lanes at `N/2`) | `N` |
| 6 | `11N` | `4N` (eight lanes at `N/2`) | `2N` |

Small lookup tables, split-equality tables, and proof output must be reported separately
from these digit-range table counts.
The final joint subproof occupies `N` field elements for its two folded witness lanes.
Report relation-provider state separately and together: the production joint-leaf peak is
`N + provider_state`, normally `N + N/g + fringes`. Add a combined peak gate; LB4 is the
sensitive cell because its range-only target is already `N`.

### Two-round deferral: experiment, not baseline

Organize the one-round lifecycle so a later measured two-round specialization can replace
the initial kernel of equality-factored product layers and the direct range-only leaf
without changing their proof API. Do not add a runtime
`defer_rounds` field, public knob, or dormant two-round implementation in the baseline
packet. The first implementation materializes after one challenge.

A product-layer two-round kernel scans logical quartets, constructs a bivariate local polynomial
`H(X,Y)`, reconstructs the ordinary wire messages

```text
q_0(X) = (1 - tau_1) H(X, 0) + tau_1 H(X, 1)
q_1(Y) = H(r_0, Y),
```

and materializes at `N/4` only after `r_1`. This is algebraically independent of x/y
storage and can remain proof-byte-identical. It is not automatically faster:

This derivation does not automatically apply to the recursive standard joint leaf. A
joint-leaf two-round optimization must separately derive the current equality factors,
signed witness lane, and relation-provider bivariate contribution, then pass a complete
joint-leaf benchmark. It is not a Packet 5 specialization by inheritance.

`H` contains only the remaining-variable equality tables and the local batched product.
It excludes `current_scalar` and the current/next linear equality factors, which remain
owned by the eq-factored driver. Fully implicit quartets are added analytically with
equality mass; partial quartets load missing corners from the per-lane default.

| Basis | Classes | Four-class keys |
|---:|---:|---:|
| 16 | 8 | 4,096 |
| 32 | 16 | 65,536 |
| 64 | 32 | 1,048,576 |

A degree-4 bivariate table needs up to 25 field coefficients per key. Therefore:

- benchmark a generated/static two-round table only for LB4;
- benchmark on-the-fly bivariate aggregation plus compact rescan for LB5/LB6 only after
  the one-round path lands;
- reject full four-class LUTs for LB5/LB6;
- delete any two-round experiment that fails its per-basis end-to-end gate.

### Fixed-lane kernel, not basis-specific module families

Use one fixed-lane product kernel instantiated through a small explicit dispatch:

| LB | Substage | Lanes | Degree |
|---:|---|---:|---:|
| 4 | root | 2 quartic children | 2 |
| 5 | root | 4 quartic children | 4 |
| 6 | root | 2 degree-16 children | 2 |
| 6 | second product | 8 quartic children, two batched parents | 4 |
| 4/5/6 | leaf | 1 `range_image` lane | 4 |

The specialization is a match arm, constants, and a fixed array width. Do not create
`lb4.rs`, `lb5.rs`, `lb6.rs`, a trait per arity, or wrapper constructors.

All high-basis pair tables are field-valued in the first implementation. Exhaustive
integer analysis of the current topology gives:

| Quantity | b=16 | b=32 | b=64 |
|---|---:|---:|---:|
| maximum `range_image` | 56 | 240 | 992 |
| maximum quartic-leaf endpoint | 23 bits | 32 bits | 40 bits |
| maximum one-round root coefficient | 44 bits | 123 bits | 303 bits |
| maximum two-round root coefficient | 46 bits | 127 bits | 305 bits |

In particular, a four-leaf LB6 node reaches roughly 158 bits and LB6 root coefficients
exceed 300 bits. Consequences:

- never build LB5/LB6 parent or pair LUTs in `i64`/`i128`;
- never use a giant generic integer parent LUT;
- only LB4 may later benchmark a narrow unsigned/unreduced table;
- the randomized final leaf batch is field-valued even though individual leaf
  coefficients fit small integers.

Deferred reduction must have a documented bound. `HasUnreducedOps` does not currently
expose a universal accumulator capacity. Either add checked associated term limits to the
specific accumulator interface or reduce canonical bounded chunks before combining them.
Honor `DELAYED_PRODUCT_SUM_IS_EXACT`; never infer safety from the domain size or a release
build's lack of overflow.

### Per-basis required work

#### LB4 / basis 16

- Eight `RangeImageClass` rows.
- Root product: two child lanes, degree 2.
- First correct version: field-valued class-pair table.
- After parity, benchmark the precisely bounded narrow-coefficient/unreduced table.
- Benchmark two-round deferral because the 4,096-key space is plausible.
- Keep the existing two-stage proof topology. A direct degree-8 proof is a separate
  protocol experiment and may not replace this path without independent proof-byte and
  verifier pricing.

#### LB5 / basis 32

- Sixteen `RangeImageClass` rows.
- Root product: four child lanes, degree 4.
- Use a field pair table or direct fixed-lane evaluation, whichever wins the complete
  substage benchmark including table construction.
- Do not use the technically-fitting but fragile signed-`i128` coefficient path.
- Do not retain a two-round four-class LUT.

#### LB6 / basis 64

- Thirty-two `RangeImageClass` rows.
- Root substage: two child lanes, degree 2; child values are field values.
- Second product substage: eight leaf lanes grouped into two weighted arity-4 parents.
- Build challenge-dependent combined round data only after the interstage weights exist.
- Final leaf substage: one field-valued batched quartic over `range_image`.
- Benchmark class-pair field tables against direct field evaluation; include construction
  and cache misses, not just lookup throughput.
- Reject all fixed-width integer parent arithmetic and all full two-round LUTs.

#### LB2/LB3 regression path

Move the good compact backend mechanics into the same architecture. LB2 and LB3 retain
their direct degree-2/degree-4 proof shapes, compact initial rounds, and any measured
initial-round batching. They must not remain delegated to a second prover type.

### Why the production tree is not binary-only

Binary-only looks simpler at the individual multiplication, but it is not simpler at the
protocol boundary. Every additional tree level is a transcript-dependent sumcheck
substage: it adds a complete witness scan/fold lifecycle, child absorption, and a fresh
interstage challenge. Adjacent levels cannot be fused because the lower anchor and batch
weights do not exist until the upper proof finishes.

Let `L = log_basis`, `n = WitnessDomain::num_vars()`, `N = 2^n`, and `|E|` be one
serialized extension-field element. There are `Q = 2^(L-3)` quartic leaves for `L >= 3`.
If only internal nodes become binary while quartic leaves remain, the exact comparison is:

| LB | Planned substages | Binary/quartic substages | Planned child claims | Binary child claims | Proof delta |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 1 | 0 | 0 | 0 |
| 3 | 1 | 1 | 0 | 0 | 0 |
| 4 | 2 | 2 | 2 | 2 | 0 |
| 5 | 2 | 3 | 4 | 6 | `2|E|` |
| 6 | 3 | 4 | 10 | 14 | `4|E|` |

The binary/quartic child-claim count is `2(Q - 1) = 2^(L-2) - 2`. Its round messages do
not shrink: both topologies serialize `2(L - 1)` range coefficients per witness round,
namely 2, 4, 6, 8, and 10 for LB2 through LB6. Replacing one arity-4 layer by two binary
layers changes `4` coefficients into `2 + 2`, while adding a full substage at LB5/LB6.
Its one-round field-state peaks are still `N`, `2N`, and `4N` for LB4/LB5/LB6, but total
memory traffic and compact rescans increase.

A fully binary tree with quadratic leaves is strictly worse:

| LB | Substages | Child claims | Delta from plan | One-round peak |
|---:|---:|---:|---:|---:|
| 2 | 1 | 0 | 0 | -- |
| 3 | 2 | 2 | `2|E|` | `N` |
| 4 | 3 | 6 | `4|E|` | `2N` |
| 5 | 4 | 14 | `10|E|` | `4N` |
| 6 | 5 | 30 | `20|E|` | `8N` |

Here the child count is `2^(L-1) - 2`. The lower recursive joint-leaf degree would fall
from 5 to 3, but the added degree-2 product levels conserve the complete recursive round
message at `2L - 1` coefficients. Thus it buys no round-size reduction, adds child
scalars, doubles the planned high-basis field-state peak, and requires another soundness
ledger.

Ignoring topology-invariant envelope and Stage-2 fields, the exact element counts are:

```text
direct digit-range elements
  = 2n(L - 1) + child_claims(topology) + 1

recursive digit-range/relation elements
  = n(2L - 1) + child_claims(topology) + 2.
```

The final direct scalar is `range_image_eval`; the two recursive scalars are
`range_image_eval` and `digit_witness_eval`. #311's sumcheck-free terminal is unaffected.

Keep the planned compressed topology:

```text
LB2  quadratic leaf
LB3  quartic leaf
LB4  binary root -> quartic leaves
LB5  arity-4 root -> quartic leaves
LB6  binary root -> arity-4 layer -> quartic leaves.
```

There is still only one product-layer implementation. It accepts the checked private
arity `2 | 4` from `DigitRangePlan` and contains small private match arms for coefficient
accumulation; it is not split into binary/quaternary prover types or wrapper functions.
A disposable benchmark may test a two-round binary kernel—its four-class bidegree `(2,2)`
has nine coefficients rather than 25 for arity 4—but no binary topology flag, proof
variant, or losing implementation may land. In particular, LB5's 65,536 keys can already
mean 589,824 extension elements and LB6's roughly 9.4 million-element table is rejected.

### Stage 1 module end state

The exact filenames may adjust to the crate's conventions, but the responsibility split
must look like:

```text
crates/akita-prover/src/protocol/sumcheck/digit_range/
  mod.rs                  orchestration and product-to-leaf choreography
  validated_digits.rs     checked compact digit and RangeImageClass access
  tables.rs               plan-derived per-class fixed-lane data
  product_stage.rs        one fixed-lane product-stage implementation
  range_leaf.rs           direct equality-factored range-only leaf composer
  relation_leaf.rs        recursive standard range-image-and-relation composer
  initial_rounds.rs       optional measured round batching
  tests.rs

crates/akita-prover/src/protocol/sumcheck/direct_relation/
  mod.rs                  direct Stage 2 relation/range-image composition
  compact_round.rs        one-pass compact signed-digit scan
  field_round.rs          field fold and optional fused-next scan
  tests.rs

crates/akita-prover/src/protocol/sumcheck/claim_reduction/
  mod.rs                  recursive Stage 2 plan dispatch and transcript choreography
  witness.rs              equality-factored range-image consistency reduction
  setup_product.rs        recursive setup product over its native domain
  batched.rs              independent-domain lift and degree-3 batch
  tests.rs

crates/akita-prover/src/protocol/sumcheck/relation_weight/
  mod.rs                  prover-side provider state and scan integration
  providers.rs            closed dense/common-base/sparse fold forms
  tests.rs

crates/akita-sumcheck/src/
  pair_scan.rs            checked flat pair/chunk iteration and reduction mechanics
  exact_prefix.rs         explicit prefix plus default-aware adjacent fold
  fold_scan.rs            field fold and optional fused-next-scan mechanics
```

The three shared files contain no Akita range roots, relation terms, transcript accesses,
stage phase enum, or provider algebra. They are ordinary functions over typed slices and
closures, not `RoundWitnessSource`/`RoundComposition` protocols. If a primitive cannot be
specified without Akita semantics, keep it in `digit_range`, `claim_reduction`, or
`relation_weight`.

No production module may exceed 500 lines and no hot kernel may exceed 160 lines without a
line-by-line review exception recorded in the implementation PR. `digit_range/mod.rs` is
expected to approach the cap because it owns orchestration for five topologies; its
exception is pre-authorized provided the overage is choreography, not equations or a
second copy of any kernel. Do not game the cap with forwarding helpers or artificial
splits. The final names should
say `digit_range`, not preserve `akita_stage1_tree` as a misleading compatibility shell.

Delete in the same series:

- `padded_s_table`;
- `build_leaf_tables`;
- `pointwise_product`;
- `ProductStageLayer` and `build_product_stage_layers`;
- `Stage1Witness::{Compact,PaddedS}`;
- `prove_recover_w`;
- nested `Vec<Vec<Vec<E>>>` tree storage;
- per-table fold loops;
- the duplicate `AkitaStage1Prover` type;
- Stage 1-specific `x_prefix`, `sparse_y`, and old raw geometry constructors;
- Stage 1 portions of the monolithic `two_round_prefix/common.rs` after their measured
  replacement lands.

Do not perform a half-way public proof rename before the protocol cutover. Internal
modules may adopt semantic names after parity, but Packet 7 creates `DigitRangeProof`,
`DigitRangeRelationProof`, and `range_image_eval` directly and deletes the numeric-stage
types. Never introduce `final_s_eval` as another transitional name.

## Stage 1 verifier design

The verifier consumes `DigitRangePlan` and a checked ordered point. It does not reconstruct
roots, tree arities, or child counts.

Required changes:

- validate supported LB, subproof count, every child count, degree, and round count from
  the plan before allocating or replaying;
- borrow tau and proof child-claim slices instead of cloning them;
- stream extension-field child absorption through one semantic helper shared with the
  prover transcript choreography;
- batch child claims and leaf coefficients with fixed small arrays;
- use the plan's field-lifted leaf polynomials and the standard univariate evaluator;
- dispatch once from `FoldCheckPlan`: direct setup replays the equality-factored range-only
  leaf; recursive offload replays a standard final leaf of enforced degree 3 or 5 and
  reads exactly `range_image_eval` followed by `digit_witness_eval` at one checked point;
- return a consuming `DeferredRangeRelationCheck` for recursive offload and refuse to
  accept the level until it is closed with the full setup contribution;
- remove `reorder_stage1_coords`; point ordering comes from `WitnessDomain` and every
  length/permutation error returns `AkitaError`;
- place explicit caps on proof substage count, child claims, rounds, and any temporary
  allocation before reading attacker-controlled lengths.

The divergent verifier combined-kernel branch is not a substitute for this cleanup: it
optimizes a structured affine digit evaluator in relation replay. Port its algebraic
prefix-scan/carry-bucket technique only after this plan's canonical domain lands, using the
dense evaluator as a differential oracle.

## Two-stage range, relation, and setup design

### Direct setup keeps the prover-optimized placement

When the verifier evaluates setup locally, fixing the relation point in Stage 1 provides
no offloading benefit. Keep the efficient existing placement, expressed with the cleaned
plans and providers:

```text
Stage 1:
  range_claim
    = sum_z Eq(range_anchor,z) * RangePolynomial(RangeImage(z))
  -> range_image_eval at range_check_point

Stage 2:
  linear_relation_claim
    + direct_range_binding_challenge * range_image_eval
  = sum_z DigitWitness(z) * [
        RelationWeight(z)
      + direct_range_binding_challenge
          * Eq(range_check_point,z) * (DigitWitness(z) + 1)
    ]
  -> next_witness_eval at next_witness_point.
```

The Stage 2 proof is standard degree 3 and scans the signed digit witness once. Its final
relation weight includes the locally computed full setup contribution. Preserve the
current direct round-element/scalar count and use the old prover/transcript as the
differential oracle through Packet 6. Packet 7 may change semantic framing under an
authenticated protocol version, but it must not increase intermediate direct serialized size, scan
count, or allocation count on non-terminal folds. #311 terminal folds are outside this
composer and remain unchanged.

### Split the current fused Stage 2 by statement, not by layout

The current fused prover contains two semantic terms hidden inside many layout branches:

```text
LinearRelationTerm
  = DigitWitness(z) * RelationWeight(z)

RangeImageConsistencyTerm(anchor)
  = Eq(anchor,z) * DigitWitness(z) * (DigitWitness(z) + 1).
```

Make those ownership boundaries explicit without adding a general expression engine:

- `DirectRelationRangeImagePlan` owns their one-pass standard degree-3 composition;
- `RangeRelationLeafPlan` owns `LinearRelationTerm` plus the anchored range leaf in the
  recursive Stage 1 standard proof;
- `RangeImageConsistencyProof` owns only the equality-factored consistency term in
  recursive `Separate`;
- `BatchedSetupAndRangeImageProof` owns the standard-lifted consistency term in recursive
  `Batched`;
- `RelationWeight` is the one canonical public polynomial used by direct and recursive
  placement; `range_image(digit)` is the one canonical pointwise map.

Share pair traversal, compact signed-digit access, accumulation, and folding. Do not share
a stage driver or transcript callback trait. This makes the placement switch auditable:
the linear term moves; the consistency term is not duplicated or omitted; trace remains a
single additive component of `RelationWeight`.

### Paper placement, tree-preserving refinement

The verifier-offloading section of the paper makes the essential dependency clear: the
relation point must be fixed in Stage 1 so the setup weight at that point can be certified
in Stage 2. As of the 2026-07-19 revision the paper specifies exactly the placement this
handoff implements — the relation fused into the final level of the range product tree,
every earlier product level keeping its compact equality-factored messages. Apply this
dependency only to `FoldCheckTopology::RecursiveSetupOffload`.

Do not implement a full-width single range-and-relation sumcheck as production. Treating the independent
range-image MLE as the input would give standard degrees 9/17/33 for LB4/LB5/LB6 after
the equality factor; substituting `DigitWitness(DigitWitness+1)` directly would instead
give 17/33/65. Either discards the exact tree this refactor is optimizing. Preserve every
product layer and fuse the relation only into the **final range leaf**. This refinement
has the same two-stage dependency and same-point output required by the paper:

- all earlier product layers remain equality-factored degree 2 or 4;
- the final leaf is one standard degree-3 message for LB2 or degree-5 message for LB3-LB6;
- `range_image_eval` and `digit_witness_eval` are evaluations of two independent MLEs at
  the same fresh `RangeRelationPoint`;
- Stage 2 alone proves that the Boolean range-image table is the pointwise image of the
  digit witness.

The independence in the third bullet is soundness-critical. Away from Boolean points,
`range_image_eval` is generally not
`digit_witness_eval * (digit_witness_eval + 1)`.

### Stage 1 final range-and-relation equation

Let `leaf_input_claim` and `leaf_anchor` be the claim and point produced by the preceding
product layer. For LB2/LB3, they are the initial zero range claim and `tau_range`, and
the joint leaf is the entire recursive Stage 1: there are no preceding product layers.
Let `LeafBatch` be the plan-derived quadratic or randomized quartic in the virtual
range-image table. Let `linear_relation_claim` already include trace's claim coefficient,
and let `RelationWeight` include all matching linear relation and trace providers exactly
once. After those values are transcript-bound, sample `range_relation_batch_challenge` and prove:

```text
leaf_input_claim
  + range_relation_batch_challenge * linear_relation_claim
= sum_z [
     Eq(leaf_anchor, z) * LeafBatch(RangeImage(z))
   + range_relation_batch_challenge * DigitWitness(z) * RelationWeight(z)
   ].
```

Here `RangeImage[z] = range_image(DigitWitness[z])` only on Boolean vertices. The prover
folds `RangeImage` and `DigitWitness` as independent multilinear tables after round 0.

#### Mandatory fused first-round kernel

For `RecursiveSetupOffload`, “one joint leaf” means one sumcheck and one compact-witness
traversal, not two accumulators driven by two scans. Write the first coordinate as `T`,
the remaining address as `u`, and `delta value_u = value(1,u) - value(0,u)`. Round zero is:

```text
joint_round_0(T)
  = eq(leaf_anchor[0], T)
      * sum_u Eq(leaf_anchor[1..], u)
          * LeafBatch(
              RangeImage(0,u) + T * delta RangeImage_u
            )
    + range_relation_batch_challenge
      * sum_u
          (DigitWitness(0,u) + T * delta DigitWitness_u)
          * (RelationWeight(0,u) + T * delta RelationWeight_u).
```

The compact loop loads each signed digit pair exactly once, derives both range-image
values, asks the one canonical relation provider for its two endpoint weights, and updates
the fixed range and relation coefficient accumulators. The resulting degree is exactly 3
for LB2 and 5 for LB3-LB6: `max(leaf_inner_degree + 1, 2)`.

After sampling the first challenge, one second compact traversal materializes both
independent folds in the same address loop:

```text
folded_range_image[u]
  = RangeImage(0,u) + r_0 * delta RangeImage_u

folded_digit_witness[u]
  = DigitWitness(0,u) + r_0 * delta DigitWitness_u.
```

Never compute the first value as
`folded_digit_witness * (folded_digit_witness + 1)`: the pointwise identity holds before
multilinear folding, not after it. The two folded tables use `N/2 + N/2 = N` field
elements. The relation provider folds its own checked representation. Later joint rounds
may zip the two aligned folded tables in one address loop, but they remain semantically
independent tables and one combined claim with one challenge sequence.

The implicit padded suffix contributes only the anchored range term:

```text
current_eq_scalar
  * eq(leaf_anchor[current_round], T)
  * remaining_eq_suffix_mass
  * LeafBatch(range_image(0)).
```

The relation suffix is zero because the padded digit witness is zero. A “fused first
round followed by two independent sumchecks” is invalid: once the claims are batched,
every later round must use the same challenge and prove the same combined claim.

`RangeRelationLeafProver` owns this compact-to-folded state transition and implements the
standard sumcheck interface directly. Do not add `prove_fused_first_round`, `FusedKernel`,
or callback/source traits. Its checked plan is a view of `FoldCheckPlan` and stores no
second domain, range plan, round count, or degree.

For `DirectSetup`, the answer is deliberately different. The final Stage-1 leaf remains
equality-factored because every batched quadratic/quartic leaf shares the one
`Eq(leaf_anchor,z)` factor—not merely because there is “one item” in a batch. Its inner
degree remains 2 for LB2 and 4 for LB3-LB6. It cannot fuse with direct Stage 2 without
changing the protocol: `RangeCheckPoint`, `range_image_eval`, and
`direct_range_binding_challenge` do not exist until Stage 1 is complete, and Stage 2 uses
an independent challenge point. Caching a field witness during Stage 1 only substitutes a
large write and later field scan for the second compact scan; it is not the baseline.

Required kernel gates are coefficient-by-coefficient equality with a separately
materialized dense oracle, one logical digit-pair visit in recursive round-zero
accumulation, one post-challenge visit producing both folds, all LB2-LB6 degree bounds,
odd live prefixes, nonzero `LeafBatch(range_image(0))`, transcript equality, and peak
state `N + relation_provider_state`.

At `range_relation_point`, the verifier's deferred final equality is:

```text
final_stage1_claim
= Eq(leaf_anchor, range_relation_point)
     * LeafBatch(range_image_eval)
 + range_relation_batch_challenge
     * digit_witness_eval
     * (non_setup_relation_weight_eval + setup_contribution_eval).
```

`non_setup_relation_weight_eval` includes matrix-consistency rows, structured rows,
quotient rows, evaluation trace, and every other non-setup term. This equation is for
recursive offload: Stage 2 supplies
`setup_contribution_eval` as a transcript-bound claim and proves it against the
precommitted setup prefix. Do not force
trace or other nonfactorable terms into the setup/common-base factor.

Use one narrow semantic plan:

```rust
pub struct RangeRelationLeafPlan {
    relation_terms: FixedRelationTerms,
}
```

This is not a general expression graph. The hot implementation has exactly the anchored
range-image leaf and the linear digit-witness relation term. It is selected only by the
recursive-offload topology; direct setup uses `DirectRelationRangeImagePlan`.
It is a checked child of `FoldCheckPlan`, not an independent public constructor: it borrows
the parent's `WitnessDomain` and `DigitRangePlan`, consumes runtime `RangeLeafInput`, and
derives standard degree as `digit_range.leaf_inner_degree() + 1`. Do not store duplicate
domain, range plan, or degree fields.

Make delayed verifier closure explicit and linear in the type system:

```rust
pub struct DeferredRangeRelationCheck<E> { /* validated final-leaf state */ }

impl<E> DeferredRangeRelationCheck<E> {
    pub fn close_with_setup_contribution(
        self,
        setup_contribution_eval: E,
    ) -> Result<RangeRelationOutput<E>, AkitaError>;
}
```

`verify_digit_range_relation` returns this value after replaying the recursive final leaf.
The caller absorbs the full claimed setup contribution and consumes the deferred check
before any Stage 2 challenge.
Consuming `self` prevents an omitted or double-applied setup contribution.

### Recursive Stage 2 range-image consistency reduction

After absorbing `range_image_eval`, `digit_witness_eval`, and
`setup_contribution_eval`, sample `range_image_binding_challenge`. Define:

```text
range_image_consistency_claim
  = range_image_binding_challenge * digit_witness_eval + range_image_eval.
```

Stage 2 proves over the same flat `WitnessDomain`:

```text
range_image_consistency_claim
= sum_z Eq(range_relation_point, z) * [
     range_image_binding_challenge * DigitWitness(z)
   + DigitWitness(z) * (DigitWitness(z) + 1)
   ].
```

This has one common equality factor and an inner degree-2 composition. A separate
recursive proof uses `EqFactoredSumcheckProof` and ends at `next_witness_eval` at
`next_witness_point`. Its final sumcheck check is:

```text
Eq(range_relation_point, next_witness_point) * [
  range_image_binding_challenge * next_witness_eval
  + next_witness_eval * (next_witness_eval + 1)
].
```

Stage 2 matches the witness representation once at round dispatch and calls statically
typed compact or field scans; it never branches on representation inside the pair loop.
The old relation x/y-prefix/dense dispatch tree is deleted, not repurposed as claim
reduction.

### Recursive mode: schedule-selected setup and range-image consistency reduction

The setup prefix has its own `SetupCoefficientDomain`: raw flat field-coefficient address
`j`, exact live prefix, next-power-of-two domain, and LSB-first binds. It is not a segment,
projection, or alternative view of `WitnessDomain`. Overlapping A/B/D setup views add at
the same physical coefficient.

After Stage 1 fixes `range_relation_point`, construct the public setup weight and prove:

```text
setup_contribution_eval
  = sum_j SetupCoefficient(j)
          * SetupRelationWeight(j; range_relation_point, tau_relation, alpha).
```

This setup product has standard degree 2. There are two sound Stage 2 realizations. Both
prove the same two semantic claims and carry the same two final openings; only their
round-polynomial schedule differs.

`SeparateReductionShape` owns `setup_rounds`, `range_image_rounds`, and the checked
independent opening route. `SetupWitnessBatchGeometry` owns the corresponding batched
round counts, lifts, and route. The validated level descriptor selects the enum before
proving starts. It is proof shape,
not a prover heuristic: serialize no variant tag, do not switch after transcript sampling,
and do not allow the verifier to infer a mode from malformed vector lengths.

#### `Separate`

Run two semantic subproofs in a fixed transcript order:

1. `SetupContributionProof`, a standard degree-2 setup-product sumcheck over
   `SetupCoefficientDomain`, ending at `setup_prefix_eval`; then
2. `RangeImageConsistencyProof`, the equality-factored inner-degree-2 range-image consistency reduction over
   `WitnessDomain`, ending at `next_witness_eval`.

For native setup and witness round counts `lambda` and `mu`, this sends
`2*lambda + 2*mu` round-polynomial field elements. The resulting setup and witness
opening points are independent. This plan is admissible only when the next-level opening
router can carry both typed points. Do not coerce one point into a suffix of the other.

#### `Batched`

Batch the setup product with the standard degree-3 form of range-image consistency reduction in one proof.
This is the right shape when the two native domains have sufficiently similar round
counts. Both reductions carry the same two openings to the next level — `Batched`'s two
points are suffix projections of one fresh point while `Separate`'s are independent — so
the distinction between them is proof bytes, not routing capability; any opening router
that supports `Batched`'s correlated points supports `Separate`'s independent ones.

The batched prover and verifier are an upgrade of existing machinery, not a fresh build:
the current `AkitaStage3Prover` already batches the degree-2 setup product with a
carried-witness term over unequal native round counts through `BatchedStage3Geometry`.
The target shape raises the witness term from the linear carried claim to the quadratic
range-image consistency composition, moving the batch degree from 2 to 3. Implement it by
adapting that geometry (renamed `SetupWitnessBatchGeometry`), and record the semantic
delta of the witness term in the Packet 7 PR.

Let `n_setup` and `n_witness` be the native round counts and
`n = max(n_setup, n_witness)`. For term `i`, define
`delta_i = n - n_i` and lift it over `delta_i` independent leading coordinates with scale
`lift_scale_i = 2^(-delta_i)`. In an inactive leading round the term emits the constant
polynomial `current_claim / 2`; in active rounds its native polynomial is scaled by
`lift_scale_i` exactly once. The Boolean sum of each lift equals its native claim.

After both native claims and domain descriptors are bound, sample
`setup_range_binding_batch_challenge` and prove:

```text
setup_contribution_eval
  + setup_range_binding_batch_challenge * range_image_consistency_claim
= sum_u [
     Lift_setup(SetupCoefficient * SetupRelationWeight)(u)
   + setup_range_binding_batch_challenge * Lift_witness(RangeImageConsistency)(u)
   ].
```

The batched degree is 3. `SetupWitnessBatchGeometry` owns the two native round counts,
inactive-prefix deltas, lift scales, and checked suffix projections from the fresh batched
point to `setup_opening_point` and `next_witness_point`.

The final verifier equality is:

```text
final_stage2_claim
= lift_scale_setup
     * setup_prefix_eval
     * setup_relation_weight_eval(setup_opening_point, range_relation_point)
 + setup_range_binding_batch_challenge
     * lift_scale_witness
     * Eq(range_relation_point, next_witness_point)
     * [range_image_binding_challenge * next_witness_eval
        + next_witness_eval * (next_witness_eval + 1)].
```

The lifted function algebraically contains `lift_scale_i`, so each active round message is
`lift_scale_i * native_round_polynomial` and the final value is
`lift_scale_i * native_mle_eval`. Implement this with one persistent multiplier; do not
pre-scale the native table or claim and multiply by the geometry scale again. Inactive
rounds update the current claim with `/2`; they do not mutate the native table.

#### Exact selector and recursive boundary

The round-polynomial field-element counts are:

```text
separate = 2 * (lambda + mu)
batched  = 3 * max(lambda, mu)
```

These formulas are only a round-message prefilter. The planner must compare three complete
serialized sizes for the exact level:

```text
legacy_recursive_bytes
target_separate_bytes
target_batched_bytes
```

As a round-only cross-check, let `R_b` be all equality-factored range-tree messages,
`mu` the witness rounds, `lambda` the setup rounds, and `M=max(lambda,mu)`:

```text
legacy recursive = R_b + 3*mu + 2*M
target separate  = R_b + 1*mu + 2*lambda + 2*mu
                 = R_b + 3*mu + 2*lambda
target batched   = R_b + 1*mu + 3*M
```

The `1*mu` term is the standard joint leaf's one-coefficient increase over the range-only
leaf. This explains the regimes: separate improves round messages only when the setup
domain is shorter than the padded legacy Stage 3 domain; batched improves when
`M < 2*mu`. Neither statement accounts for scalars or envelopes.

Include the old/new final leaf, all setup/range-image rounds, every scalar, envelopes,
opening metadata, extension encoding, and descriptor impact. Among admissible targets,
choose `Batched` only when its complete size is strictly smaller than `Separate`; equality
defaults to `Separate`. A target recursive topology is admissible only when its complete
size is no larger than `legacy_recursive_bytes`; byte parity is admissible because the
selection objective is verifier work, and `Separate` exactly ties the legacy round count
whenever `lambda >= mu`. Otherwise the planner keeps
`FoldCheckTopology::DirectSetup` for that candidate and reports why setup offloading did
not survive the size gate. The choice is encoded in the schedule, never made at runtime.

The direct-versus-recursive selection is priced on the complete objective, not proof bytes
alone: (a) the verifier-side saving from replacing the setup scan with the setup-product
replay, and (b) the next-level cost of carrying the setup-prefix opening, whose
decomposition is `ceil(log_b(q))` digit planes deep because the prefix holds full-field
coefficients. Item (b) enters only the direct-versus-recursive comparison — the legacy
recursive shape carries the same opening, so legacy-versus-target comparisons are
unaffected. Offload eligibility is independent of the level's folding-challenge structure
(flat or tensor), but the planner records the challenge structure of offloaded levels:
the full verifier-cost saving assumes the challenge-dependent scan is also reduced by
tensor challenges, and a flat-challenge offloaded level saves only the setup-dependent
term.

If separate-point routing is not implemented, `Separate` is temporarily inadmissible
rather than silently replaced with a larger proof. Packet 1 must price complete
serialization before Packet 7 freezes this rule.

The next recursive boundary carries exactly:

- `(next_witness_commitment, next_witness_point, next_witness_eval)`; and
- `(setup_prefix_commitment, setup_opening_point, setup_prefix_eval)`.

If the next level cannot route both openings, or the exact setup slot/domain does not
match, recursive offload is inadmissible and the planner must choose direct mode.

Setup-prefix provisioning is deliberately per level. Each offloaded level selects the
least committed slot covering its own active footprint, and its setup opening is
discharged at the immediately following fold through the opening router. There is no
shared largest-prefix commitment and no cross-level accumulation of setup claims: the
active footprint shrinks with the recursion, so per-level slots keep later offloads
profitable, whereas one shared largest prefix would force every later discharge to open
the largest object and break the setup-versus-witness balance at every level but the
first. At most one outstanding setup opening exists at any recursive boundary. Chained
offloading is well-defined under this rule: a level that receives a carried setup opening
discharges it inside its own root relation, and that discharge is orthogonal to whether
the level itself emits a new setup claim.

### Proof and level-envelope types

Replace numeric stage fields with semantic proof types at the atomic cutover:

```rust
pub struct RangeProductLayerProof<E> {
    pub sumcheck: EqFactoredSumcheckProof<E>,
    pub child_claims: Vec<E>,
}

pub struct RangeOnlyLeafProof<E> {
    pub sumcheck: EqFactoredSumcheckProof<E>,
    pub range_image_eval: E,
}

pub struct DigitRangeProof<E> {
    pub product_layers: Vec<RangeProductLayerProof<E>>,
    pub final_leaf: RangeOnlyLeafProof<E>,
}

pub struct RangeRelationLeafProof<E> {
    pub sumcheck: SumcheckProof<E>,
    pub range_image_eval: E,
    pub digit_witness_eval: E,
}

pub struct DigitRangeRelationProof<E> {
    pub product_layers: Vec<RangeProductLayerProof<E>>,
    pub final_leaf: RangeRelationLeafProof<E>,
}

pub struct DirectRelationRangeImageProof<E> {
    pub sumcheck: SumcheckProof<E>,
    pub next_witness_eval: E,
}

pub struct RangeImageConsistencyProof<E> {
    pub sumcheck: EqFactoredSumcheckProof<E>,
    pub next_witness_eval: E,
}

pub struct SetupContributionProof<E> {
    pub sumcheck: SumcheckProof<E>,
    pub setup_prefix_eval: E,
}

pub struct BatchedSetupAndRangeImageProof<E> {
    pub sumcheck: SumcheckProof<E>,
    pub setup_prefix_eval: E,
    pub next_witness_eval: E,
}

pub enum RecursiveClaimReductionProof<E> {
    Separate {
        setup: SetupContributionProof<E>,
        range_image: RangeImageConsistencyProof<E>,
    },
    Batched(BatchedSetupAndRangeImageProof<E>),
}

pub enum FoldCheckProof<E> {
    DirectSetup {
        digit_range: DigitRangeProof<E>,
        relation_and_range_image: DirectRelationRangeImageProof<E>,
    },
    RecursiveSetupOffload {
        digit_range_relation: DigitRangeRelationProof<E>,
        setup_contribution_eval: E,
        claim_reduction: RecursiveClaimReductionProof<E>,
    },
}

pub struct FoldLevelProof<F: FieldCore, E: FieldCore> {
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<E>>,
    pub scaled_fold_witness: RingVec<F>,
    pub fold_grind_nonce: u32,
    pub next_witness_binding: NextWitnessBinding<F>,
    pub fold_check: FoldCheckProof<E>,
}
```

The sketches name ownership; use the repository's actual generic and container types.
They show the final target after optional ledger PR E8. Packet 7b stores
`SetupContributionProof` and `RangeImageConsistencyProof` directly in the recursive
variant; it must not add a one-variant `RecursiveClaimReductionProof`, an empty `Batched`
placeholder, or a forwarding accessor. E8 introduces the two-variant enum and
`BatchedSetupAndRangeImageProof` only if the batched gate passes, while preserving the
existing direct and separate encodings.
`setup_contribution_eval` closes Stage 1 and is absorbed before any Stage 2 challenge, so
it lives exactly once in the `RecursiveSetupOffload` envelope — matching the wire's
common prefix below — rather than duplicated across the two reduction variants.
Modify #311's existing `FoldLevelProof` directly; do not wrap it in an
`IntermediateFoldCheckProof`. Preserve `NextWitnessBinding::{OuterCommitment,
TerminalInnerState}` as the one schedule-shaped outgoing-state authority. The outer
commitment leaves the old Stage-2 payload and becomes the common non-terminal envelope
field; `TerminalInnerState` still serializes no duplicate commitment. `v` becomes
`scaled_fold_witness` outside equations, and no proof type or accessor contains `stage3`
in its name. Delete pass-through and cross-variant accessors; pattern-match the semantic
proof or return `AkitaError`.

#311's existing `TerminalLevelProof` is not replaced, extended, or wrapped. It remains
exactly the optional extension-opening reduction, grind nonce, and clear segment-typed
witness. Its direct ring and trace relations have no fold-check proof object.

Headerless serialization follows transcript-use order exactly; do not rely on Rust enum
layout or add custom per-type ordering:

```text
Intermediate envelope prefix:
  extension-opening reduction when scheduled
  scaled_fold_witness
  fold_grind_nonce
  next_witness_binding
    OuterCommitment: serialized commitment
    TerminalInnerState: no payload

DirectSetup fold check:
  each product layer: sumcheck, then child_claims
  range-only leaf: sumcheck, then range_image_eval
  direct relation/range-image sumcheck, then next_witness_eval

RecursiveSetupOffload common prefix:
  each product layer: sumcheck, then child_claims
  joint leaf sumcheck, range_image_eval, digit_witness_eval
  setup_contribution_eval

Separate suffix:
  setup contribution sumcheck, setup_prefix_eval
  range-image consistency sumcheck, next_witness_eval

Batched suffix:
  batched setup/range-image sumcheck, setup_prefix_eval, next_witness_eval
```

Proof structs, shape descriptors, serializers, deserializers, size formulas, and transcript
read order all mirror this sequence. Reject extra or missing fields from the
schedule-selected shape before allocation. The #311 terminal wire order and size remain
unchanged and are not derived from `FoldCheckPlan`.

### Normative transcript order

The authenticated `FoldCheckPlan` selects the variant; serialize no tag. Bind the protocol
version, topology, native round counts, coordinate order, lift convention, role
dimensions, and exact setup slot/commitment identity before proof messages.
The level envelope must already have absorbed the schedule-selected
`next_witness_binding`: either the exact outer commitment or #311's already-bound terminal
inner state. Every later `next_witness_eval` targets that exact state and opening route.

Direct setup keeps its existing algebraic order: finish and absorb the range proof,
sample the direct range-binding challenge at the existing transcript location, run the
standard relation/range-image proof, absorb `next_witness_eval`, and check the direct final
equation with locally evaluated setup. Packet 7 may add semantic framing under a new
version, but no new challenge or scalar enters this topology.

Recursive-offload common order is:

1. Complete every equality-factored range product layer and absorb child claims.
2. Bind the final leaf input claim, anchor, relation claim, and provider descriptor; sample
   `range_relation_batch_challenge` and run the standard joint leaf.
3. Absorb `range_image_eval`, then `digit_witness_eval`.
4. Absorb the full `setup_contribution_eval`; consume
   `DeferredRangeRelationCheck` and close Stage 1.

Then use the selected Stage 2 frame:

- `Separate`: run `SetupContributionProof` under setup-specific labels, absorb
  `setup_prefix_eval`, and close that frame. Only then sample
  `range_image_binding_challenge`, run `RangeImageConsistencyProof` under range-image
  labels, absorb `next_witness_eval`, and close that frame. There is no setup/range batch
  challenge.
- `Batched`: sample `range_image_binding_challenge` and derive the native consistency
  claim. After both native claims and descriptors are fixed, sample
  `setup_range_binding_batch_challenge`, run the standard combined proof, absorb
  `setup_prefix_eval`, then `next_witness_eval`, and check the combined final equation.

Use semantic framing labels for every claim, round, point, and final value; do not rely
only on generic sumcheck labels. Never reuse a range-tree interlayer challenge,
`range_relation_batch_challenge`, `range_image_binding_challenge`, or
`setup_range_binding_batch_challenge`.

### Proof-size accounting

For a witness-domain round, the optimized range tree sends the following equality-factored
range coefficients before any relation/setup work:

| LB | Range-tree elements/round | Interlayer child-claim scalars | Direct total/round (`+3`) | Legacy recursive common-round (`+5`) | Target batched common-round (`+4`) |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0 | 5 | 7 | 6 |
| 3 | 4 | 0 | 7 | 9 | 8 |
| 4 | 6 | 2 | 9 | 11 | 10 |
| 5 | 8 | 4 | 11 | 13 | 12 |
| 6 | 10 | 10 | 13 | 15 | 14 |

The table is a common-round sanity check, not a serializer. Product layers can have
different native rounds, and the separate/batched setup domain may differ from the witness
domain.

For the existing common-round recursive shape, the current implementation sends:

```text
eq-factored final range leaf: leaf_degree
current fused relation/range binding: 3
current setup/witness Stage 3: 2
total: leaf_degree + 5 extension-field elements.
```

The target batched shape sends:

```text
standard final range-and-relation leaf: leaf_degree + 1
combined setup/witness Stage 2: 3
total: leaf_degree + 4 extension-field elements.
```

Thus a balanced batched cell is expected to save one extension-field element per common
round, conditional on Packet 1 confirming the legacy accounting. For
unequal native domains, use the exact formulas above: `2*(lambda+mu)` for separate and
`3*max(lambda,mu)` for batched, then add the final-leaf, scalar, and envelope fields and
compare both complete targets with the complete legacy proof.

Direct setup retains the existing algebraic placement:

```text
range-only final leaf: leaf_degree
relation/range-image Stage 2: 3
total: leaf_degree + 3 per witness-domain round
```

Scalar accounting is likewise explicit:

| Topology | Legacy witness/setup scalars | Target scalars |
|---|---:|---:|
| Direct setup | 2 (`range_image_eval`, `next_witness_eval`) | 2, same meanings |
| Recursive offload | 5 (legacy range eval, Stage 2 eval, setup claim/prefix eval, carried eval) | 5 (`range_image_eval`, `digit_witness_eval`, `setup_contribution_eval`, `setup_prefix_eval`, `next_witness_eval`) |

The recursive saving comes from round messages and stage placement, not scalar deletion.

The cutover adds no intermediate-direct scalar, round coefficient, or proof field. It may regenerate
challenge values if the authenticated protocol version and semantic transcript framing
change, but intermediate direct serialized size must remain exactly equal and the prover
must retain one signed-witness scan/fold. Recursive mode must be no larger in complete
serialized bytes and must improve measured verifier time. Packet 1 derives exact formulas for unequal domains, every scalar,
both #311 `NextWitnessBinding` variants, and extension encoding; Packet 7 checks formulas against actual
serialization before the cutover lands.

### Degree ledger

`FoldCheckPlan` enforces these degrees; the verifier never infers them from received vector
lengths:

| Subproof | Format | Enforced degree |
|---|---|---:|
| Earlier range-product layer | equality-factored | inner 2 or 4 from `DigitRangePlan` |
| Direct range-only leaf | equality-factored | inner 2 for LB2; inner 4 for LB3-LB6 |
| Direct relation/range-image consistency | standard | 3 |
| Recursive range/relation leaf | standard | 3 for LB2; 5 for LB3-LB6 |
| Separate range-image consistency | equality-factored | inner 2 |
| Separate setup contribution | standard | 2 |
| Batched setup/range-image reduction | standard | 3 |

Rederive every soundness/security budget that prices sumcheck degree from this ledger for
both topologies and both recursive reduction plans. Do not copy the paper's literal
direct-polynomial degree or the current Stage 3 budget.

### Terminal folds after #311

#311 has already made the stronger cut: terminal folds expose the segment-typed witness
and perform reduced A-ring and trace checks directly. They have no digit-range proof,
relation sumcheck, Stage 2, Stage 3, outgoing commitment, or `next_witness_eval`.

This project treats that terminal contract as immutable. `FoldCheckPlan` cannot be
constructed for `TerminalLevelProof`; planner sizing calls #311's terminal-size authority
directly; prover and verifier dispatch to #311's terminal path without an empty fold-check
placeholder. Recursive setup offloading remains impossible at a terminal because there is
no next level to carry the setup-prefix opening. Tests assert byte-for-byte terminal proof
and transcript equality against the post-#311 baseline.

### Future negative-binary seam

A future negative-binary term belongs in the active Stage 2 range-image-consistency
composer (direct combined or recursive reduction), because that is where the pointwise
range-image relation is bound. Its weight must be the single MLE of the Boolean table
`eq(range_image_anchor,z) * I_binary(z)`, not the product of equality and support MLEs off
the cube. This series adds no field, coefficient, claim slot, transcript challenge, or
inactive branch for it.

## Mixed ring dimensions and relation-weight providers

### Flat protocol, common-base internal fast path

The only relation address in either topology is the raw physical coefficient index `z`,
LSB-first. The range proof and both Stage 2 composers use that same `WitnessDomain`;
recursive setup certification uses the separate `SetupCoefficientDomain`. For a
validated nested role tuple define:

```text
g = gcd(d_a, d_b, d_d) = min(d_a, d_b, d_d)
k = log2(g)
z = g * x + y,  0 <= y < g.
```

The `(x, y)` notation in this section is private kernel algebra. Do not serialize `g`, add
another `ring_bits`, expose a common-base x/y domain, or split the semantic relation claim.

Compile a native row-family coefficient spread to canonical events:

```text
Event {
    physical_start: p,
    length: q * g,
    alpha_exp_start: e,
    scalar: c,
}
```

For a factorized event, the compiler proves that `p = g*p0` and `e = g*e0`. For local
`t = g*h + y`:

```text
c * alpha^(e + t)
  = alpha^y * [c * (alpha^g)^(e0 + h)].
```

Accumulate the high-lane contribution with `+=`:

```text
M[p0 + h] += c * (alpha^g)^(e0 + h)
R(g*x + y) = A_g(y) * M(x)
A_g = [1, alpha, ..., alpha^(g-1)].
```

Do not use alpha inverses. `alpha = 0` is a valid challenge and must work.

This exact factorization covers current E/T/Z/R relation construction:

- A/E consistency and setup-D contributions occupying one physical span add into the
  same high-lane accumulator even when their exponent patterns differ;
- T/B is analogous;
- Z and quotient R contributions have exponent start zero and row-family denominators are
  scalar amplitudes;
- gadget, tau, row, setup-ring, and challenge factors are scalar amplitudes;
- role exponent resets compile as additive periodic high-lane patterns.

Use a closed provider representation equivalent to:

```rust
enum LinearWeightProvider<E> {
    CommonBase(CommonBaseRelationWeight<E>),
    DenseExactPrefix(ExactPrefixTable<E>),
    SparseRuns(SparseWeightRuns<E>),
}

struct CommonBaseRelationWeight<E> {
    base_dim: usize,
    low_factor: Vec<E>,       // A_g
    high_lanes: HighLaneWeights<E>,
}

enum HighLaneWeights<E> {
    Dense(Vec<E>),            // coalesced M, length at most N/g
    Spans(Vec<FactorizedSpan<E>>),
    Sparse(SparseWeightRuns<E>),
}

enum ExponentPattern<E> {
    Linear { phase: usize },
    Periodic { period: usize, phase: usize },
    PrescaledDense(Vec<E>),
    PrescaledSparse(SparseWeightRuns<E>),
}
```

The exact representation may coalesce all compatible spans into one dense `M` of length
`N/g`; this is already a major win. `HighLaneWeights` selects one storage form—it must not
retain both dense `M` and the source spans. Keep sparse spans only when the end-to-end
benchmark beats coalescing. `FactorizedSpan` must support multiple additive patterns over
the same physical interval and partial A-width columns; do not model a span as exactly one
`column_weight * alpha_local` product. Preserve the current semantic emitter's overlap
invariant or explicitly coalesce overlaps before assigning storage.

Factorization is legal only when a checked compiler proves:

1. `g` is a nonzero power of two and divides every native role dimension;
2. physical start, exponent start, and factorized length are `g`-aligned;
3. the raw-to-opening address mapping preserves coefficient order;
4. live raw prefix and padded capacity are `g`-aligned for the factorized portion;
5. the local address map preserves the low `k` raw address bits;
6. a partial native interval is masked exactly and no periodic pattern leaks into an
   inactive subcolumn.

If any precondition fails, split checked aligned spans and fringes or use the same pair
scan with a dense/sparse flat provider. Never repair misalignment with inverse powers,
silent padding, or a different challenge domain.

Every physical contribution is owned exactly once. Spans are compiler temporaries and are
dropped after coalescing into `HighLaneWeights::Dense`, or they remain the sole
`HighLaneWeights::Spans` storage. Unaligned dense/sparse fringes are separate additive
providers. The final evaluator computes
`A_g(r_low) * M(r_high) + fringe(r)` exactly once; it never replays source spans after
coalescing or folds a fringe into both terms.

### Why the common base, not `d_a`

Factoring at `d_a` is wrong in mixed mode. A D-role setup contribution resets its alpha
exponent on each D subcolumn, while an A consistency contribution may continue across the
A-local coefficient width. They do not share one length-`d_a` low factor. At `g`, every
native exponent has the form `g*h + y`, so all role families share exactly `alpha^y` and
retain their different high-lane periodic patterns.

Separate challenge domains per role are also wrong: they would produce incompatible
relation points and downstream witness openings in either topology.

### Fold behavior and expected gain

During the first `k` raw binds, fold `A_g` while scanning contiguous common-base witness
lanes. After those binds, the provider is the scalar `A_g(r_low)` times the folded lane
table `M`. Arbitrary trace or future support weights need not share the alpha factor; keep
them as additive dense/sparse providers in the same scan rather than forcing the combined
weight into one factorization.

The current mixed fallback stores `N` field weights and spreads a native event across
`D` coefficients. Common-base compilation stores at most `N/g` lane weights and expands
the event across `D/g` lanes. For supported `g = 32, 64, 128`, this cuts relation-table
footprint and coefficient-spread construction by 32-128x. A `128/64/32` tuple still gets
a 32x reduction while the protocol remains flat. Direct setup folds this provider in its
one Stage 2 signed-witness scan. Recursive offload folds it in the joint Stage 1 leaf and
scans `digit_witness` again for Stage 2 consistency. Report provider construction,
direct-composer, recursive final-leaf, recursive claim-reduction, and combined gains
separately.

### One semantic relation emitter

Refactor `compute_relation_weight_evals_inner` into a checked semantic emitter over
row-family events and physical flat addresses. The emitter is the single source for:

- a dense exact-live vector used as the scalar correctness oracle;
- the `CommonBase`/sparse prover compiler;
- direct evaluation at a verifier point;
- setup-contribution attribution tests.

Do not retain `compute_relation_matrix_col_evals` and
`compute_relation_weight_evals` as two public semantic builders. A compact provider and a
dense vector are representations of the same emitted polynomial.

The verifier receives the same `WitnessDomain`, role-family metadata, and semantic
event rules, but evaluates the prepared polynomial without materializing prover tables.
It must not infer local coordinates from one global compile-time `D`.

Crate placement follows the diff-surface table: the semantic event emitter, the
common-base compiler, and the prepared verifier evaluator live in `akita-types` and are
consumed by both prover and verifier. The prover's `relation_weight/` module holds only
prover-side fold state and scan integration over those shared plans. A second emitter or
evaluator in `akita-verifier` is exactly the "two public semantic builders" violation
this section deletes; do not reintroduce it across a crate boundary.

## Relation finalization and recursive Stage 2 opening geometry

### Three points with three different meanings

Use descriptive point types and never slice a raw `Vec<E>` at a call site:

```rust
pub struct WitnessDomainPoint(/* checked point in WitnessDomain */);
pub struct RangeCheckPoint(WitnessDomainPoint);
pub struct RangeRelationPoint(WitnessDomainPoint);
pub struct SetupOpeningPoint(/* point in SetupCoefficientDomain */);
pub struct NextWitnessPoint(WitnessDomainPoint);

pub struct CommonBaseWitnessPoint { /* private factorized view of WitnessDomainPoint */ }
pub struct SetupWitnessBatchGeometry { /* checked Stage 2 lift and suffix maps */ }
```

`RangeCheckPoint` is sampled by the direct range-only Stage 1 leaf and is the only accepted
anchor for `DirectRelationRangeImagePlan`. `RangeRelationPoint` is sampled by the recursive
final Stage 1 leaf. A checked private
method may view its underlying `WitnessDomainPoint` at common base `g` as the low
`log2(g)` raw coordinates plus the remaining lane coordinates. Direct mode applies the
same view to `NextWitnessPoint`. The view is used only to evaluate a factorized
`RelationWeight`:

```text
RelationWeight(range_relation_point)
  = MLE(A_g, coefficient_point) * MLE(M, lane_point)
    + fringe_eval.
```

`SetupOpeningPoint` and `NextWitnessPoint` are sampled in recursive Stage 2. Under
`Separate` they are independent native points. Under `Batched`,
`SetupWitnessBatchGeometry` projects the fresh padded point to two checked suffixes and
owns both inactive-coordinate lift scales. None of these Stage 2 points is a slice or
reinterpretation of `RangeRelationPoint`.

Delete `Stage2PointProjection`, `BatchedStage3Geometry`, `AkitaStage3Prover`, and every
caller-owned `d_a.trailing_zeros()` slice. The replacement is semantic point ownership,
not a renamed generic slicer.

### One full setup-contribution scalar

The setup contribution crossing the Stage 1/Stage 2 boundary is exactly:

```text
setup_contribution_eval
  = sum_j SetupCoefficient(j)
          * SetupRelationWeight(j; range_relation_point, tau_relation, alpha).
```

It is the complete flat setup contribution expected by `RelationWeight` at the Stage 1
point. It is not a high-lane partial `C`, and Stage 1 must not multiply it by
`A_g(r_low)` again. Common-base factorization is an internal way to build or evaluate
`SetupRelationWeight`; it cannot change the scalar's public meaning.

Both topologies call one canonical setup-weight builder over a checked
`WitnessDomainPoint`:

- direct mode calls it at `next_witness_point` and evaluates the setup coefficient table
  locally as part of the Stage 2 final relation-weight evaluation;
- recursive mode calls it at `range_relation_point`, accepts the resulting full scalar as
  a claim, then proves the identical flat dot product against the exact precommitted setup
  prefix.

For the same arbitrary witness-domain point, differential tests compare dense evaluation,
factorized evaluation, local setup replay, and recursive setup-proof replay of the full
scalar. Do not compare only a common-base lane fragment.

For a qualifying setup-weight span, write the Stage 1 relation point as
`range_relation_point = (r_low, r_lane)` and a setup address as `j = g*J + y`. A valid
internal factorization has the form:

```text
SetupRelationWeight(g*J + y; range_relation_point)
  = A_g(r_low) * A_g(y) * Omega(J; r_lane),

setup_contribution_eval
  = A_g(r_low)
      * sum_{J,y} SetupCoefficient(g*J + y) * A_g(y) * Omega(J; r_lane).
```

The recursive setup-product weight includes the fixed witness-side scalar
`A_g(r_low)`. At a fresh setup opening point `(rho_setup_low, rho_setup_lane)`, its
final evaluation is:

```text
A_g(r_low) * A_g(rho_setup_low) * MLE(Omega, rho_setup_lane).
```

These are two different factors over two different points; each appears once.
Nonfactorable trace and fringe terms stay wholly inside `non_setup_relation_weight_eval`.

### Role-expanded setup addresses

Current setup equality slices can evaluate one witness-column address and repeat it over
projected base sublanes. Mixed geometry instead needs one checked physical setup address
for every native role sublane.

Add one typed address helper shared by relation-event emission, direct setup replay,
`SetupContributionPlan`, setup-weight construction, and recursive proof preparation. For
native role dimension `D` and `u in 0..D/g`:

```text
physical_base_lane
  = witness_col * (d_a/g)
  + native_role_subcol * (D/g)
  + u.
```

The helper receives a typed native setup column already mapped to
`(witness_col, native_role_subcol)`. It checks divisibility, `u`, role-subcolumn bounds,
all arithmetic, and final domain membership. Prover and verifier may not reconstruct this
map independently from matrix-column order. D/B columns carry their A-witness
role-subcolumn mapping; A uses subcolumn zero; Z sums fold-digit equality separately at
every A sublane.

Overlapping A/B/D views contribute with `+=` to the same raw `SetupCoefficient` address.
The setup domain therefore needs its own validated semantic names:

```rust
pub struct SetupCoefficientDomain(FlatBooleanDomain);
pub struct SetupCoefficientIndex(usize);
pub struct SetupRelationWeight<E>(/* prepared public polynomial */);
```

Do not name the setup matrix, setup prefix, or setup polynomial `S`; that would recreate
the ambiguity removed from the range path.

### Future independently committed oracle constraint

Do not add a generic production `OracleAddressEmbedding` enum in this series. The Stage 1
range and relation terms and the Stage 2 range-image consistency reduction use the same physical
`digit_witness` domain. Setup certification uses only `SetupCoefficientDomain`; the
recursive opening route uses the named typed points above.

The future compression handoff must nevertheless preserve this soundness rule: equal
padded round counts do not justify challenge reuse. Its coordinate map must be injective,
in range, order-explicit, and separate active from inactive coordinates. A fixed-coordinate
embedding owns the equality selector for the actual fixed bit values. A repeated-table
lift owns its audited `2^{-inactive}` sum coefficient exactly once. A contiguous interval
is not a Boolean subcube unless the map proves it. A common-base view is valid only when
the canonical low `log2(g)` coordinates map identically and the offset is `g`-aligned.
Until that future feature implements and discharges those obligations, reject any new
independently committed oracle from either stage.

### Recursive setup-offload eligibility

Production recursive setup-prefix commitments are currently fixed to
`SETUP_OFFLOAD_D_SETUP = 64`. `SetupCoefficientDomain` stores its flat domain and exact
slot identity; it does not store an independently constructible base dimension. Its
`checked_common_base_view(role_dims)` derives `g` from bound role dimensions and validates
all alignment.

First mixed-role release policy:

- Stage 1 relation evaluation and direct setup replay support every validated nested role
  tuple.
- Recursive setup offload is eligible only when an exact slot exists and all current slot
  ID, `d_setup`, natural-length coverage, padded-domain, commitment-parameter, and setup
  envelope checks pass. `slot.d_setup == derived_common_base` is necessary, not
  sufficient.
- With current production slots, this means `g == 64`, for example `128/64/64`.
- The planner selects `FoldCheckTopology::DirectSetup` when no recursive slot qualifies and
  emits a diagnostic. A prover/verifier given either scheduled recursive variant with
  a missing or mismatched slot returns `InvalidSetup`; it never dynamically downgrades the
  mode or changes proof shape.
- C7's direct `SeparateReductionShape` additionally requires a next-level opening route
  for two independent points. If E8 lands, `RecursiveClaimReductionPlan::Batched`
  requires validated lift/suffix geometry. Both require the complete setup and witness
  opening route promised by the level envelope.
- A `128/64/32` recursive setup path requires a separately generated, committed, and
  audited D32 setup-prefix slot. Parameterize the APIs now, but do not pretend the D64
  artifact serves D32.

## Planner and proof-shape integration

### Round counts and descriptors

Replace `sumcheck_rounds(level_d, next_w_len)` with a D-free checked function derived from
the canonical Boolean domain:

```text
rounds = ceil_log2(live_len), with the repository's checked nonempty convention.
```

Remove comments/formulas that define the count as `col_bits + ring_bits`. Packets before
the atomic two-stage protocol cutover preserve homogeneous descriptor bytes, proof bytes,
transcript events, challenge draw order, and challenge-to-physical-address mapping. The
cutover intentionally changes intermediate proof shape as specified above; thereafter the
new schedule descriptor and semantic transcript are the sole oracle. The flat-domain
representation work by itself is not permission to change the proof language.

Before accepting a mixed proof, bind its role dimensions, flat domain semantics, segment
layout, row-family layout, common-base projection, `FoldCheckTopology`, recursive reduction
shape, and projection version in a single unambiguous schedule/instance descriptor. If
the current descriptor cannot express that identity without changing homogeneous bytes,
add a mixed-only version/capability in
Packet 8 or stop for a separate protocol spec. A verifier must never accept two address or
projection interpretations under one digest.

### Mixed-dimension schedule gate

The planner may emit `d_a != d_b` or `d_a != d_d` only after:

- native quotient construction and denominators pass row-family tests;
- direct Stage 2 relation/consistency and recursive Stage 1 relation plus Stage 2 reduction
  pass uniform and mixed tests;
- trace weights use the flat domain without eager remap;
- direct setup replay, recursive setup proof, and witness carry use distinct typed points;
- recursive setup mode follows the exact eligibility rule above;
- multi-group and multi-chunk layouts pass;
- serialized proof-size formulas match actual proofs;
- prepared setup-cache cost is included in candidate scoring.

First measured production candidate should be fp128 with `d_a = 128` and
`d_b = d_d = 64`, compared against homogeneous D64. Do not jump to D256 or enable a broad
tuple catalog until the D128 lift improves the full objective.

The planner preview must compare homogeneous, unrestricted mixed, and cache-capped mixed
schedules. Count unique prepared cache keys across the schedule and attribute total byte
delta to nonterminal folds, terminal tail, setup-product proof, and other bytes. Retain the
existing conservative cap direction:

```text
mixed_prepared_cache_bytes <= min(
    baseline_prepared_cache_bytes * 5 / 4,
    baseline_prepared_cache_bytes + 256 MiB
).
```

Stop mixed-D rollout if no candidate survives the cache cap or if the apparent win is
primarily a terminal-tail artifact rather than recurring fold savings.

Planner CPU cost may price measured basis/domain/role work, but it must not serialize or
select a CPU kernel implementation. LUT versus direct evaluation is a local build/runtime
kernel choice whose output is identical.

## Upstream prover digit production

This workstream starts only after the range/sumcheck core is stable, so it does not obscure
the dominant Stage 1 win.

### Destination-oriented decomposition

Change balanced decomposers to write directly into final validated witness segments:

```rust
decompose_*_into(source, params, destination_segment)
```

Requirements:

- z/e/t/r digits write once into canonical digit-innermost storage;
- use a flat checked centered table rather than `Vec<Vec<Vec<i32>>>`;
- do not return temporary plane vectors only to copy them;
- avoid recomposition followed by decomposition when the digit representation is already
  available;
- group order uses checked ranges/index maps, not reorder-and-concatenate copies.

Add const-specialized decomposition loops for LB4/LB5/LB6 only after one generic
destination-oriented kernel is correct:

- hoist mask, half-basis, and shifts;
- select the ordinary signed-`i128` versus rare overflow path once per row/chunk;
- benchmark coefficient-major versus final plane/digit-major emission;
- prefer branchless normalization only when measured;
- rely on compiler vectorization before considering explicit SIMD.

### Fold-grind workspace

Introduce one reusable `FoldProbeWorkspace` per group:

1. draw challenges into reusable storage;
2. traverse each source coefficient once;
3. accumulate the global response and every chunk response together;
4. keep candidates in centered integer rows;
5. check digit bounds, infinity norm, and Golomb admissibility before field conversion;
6. convert only an accepted nonce to the final fold witness;
7. reuse all scratch allocations on rejected nonces.

Preserve implicit zero planes for one-hot inputs until final physical emission. Dense
inputs must reuse cached digit planes when available rather than decompose canonical field
coefficients again for every nonce.

### Ring relation construction

Split the long `RingRelationProver::new` only at real artifact boundaries with distinct
invariants and ownership:

```text
prepare_opening_digits
  -> sample_fold_response
  -> build_relation_instance
  -> build_recursive_witness_inputs
```

These are phases, not forwarding wrappers. If a proposed type does not eliminate a clone,
centralize validation, or establish an invariant, do not add it.

## Delivery model: central hub, bounded PRs, and atomic cutovers

This document is the central implementation hub. It owns the target architecture, packet
contracts, dependency order, PR ledger, benchmark gates, and accepted deviations. A child
PR may refine implementation detail, but it must update the affected ledger row and this
document in the same diff when it changes a dependency, invariant, cutover boundary, or
performance gate. Do not create a second branch-local design spec.

The work must not live as one long-running mega-PR. Deliver it as short tranches. Each
tranche has zero or more additive/behavior-preserving foundation PRs followed by exactly
one atomic production cutover PR. Merge a completed tranche before opening the next
production tranche; rebase the next branch on the new `main` instead of carrying the full
historical stack indefinitely. Investigations and benchmarks may run in parallel, but
their production branches do not merge out of ledger order.

### What “additive” permits

An additive foundation PR may merge independently only when every added artifact is
already useful and exercised without creating a second protocol implementation:

- test-only dense oracles, differential fixtures, and fixed benchmark scenarios;
- checked plan/domain types that immediately replace duplicated validation or sizing
  authority while preserving current proof bytes and transcript events;
- arithmetic, pair-scan, fold, and provider primitives called directly by the current
  production path with byte-identical behavior;
- private candidate kernels confined to `#[cfg(test)]` or benches until a winner is
  selected.

“Additive” does **not** permit an unused production prover/verifier, a feature-flagged
second engine, ignored proof fields, a dormant schedule variant, a compatibility decoder,
or an adapter whose only job is to keep old and new APIs alive together. If a foundation
cannot stand on its own under those rules, keep it as a stacked draft and merge it only
back-to-back with its cutover.

### Atomic cutover rule

The closing PR of each tranche must, in one diff:

1. select the new canonical entry point from every affected production caller;
2. update the prover, verifier, authenticated plan, proof shape, sizing, serialization,
   transcript, schedules, and generated artifacts that the feature actually changes;
3. delete the replaced implementation, names, wrappers, decoders, and runtime selector;
4. prove old/new parity or the declared protocol delta against the tranche baseline; and
5. update this ledger with the final PR, head, measurements, and any approved deviation.

No follow-up “cleanup PR” may be required to make a cutover single-source-of-truth. Later
optimization PRs may improve the new canonical implementation, but they may not finish a
deletion owed by the cutover that activated it.

### Planned PR stack

Branch names are recommended names, not API. `Base` is the required semantic predecessor;
once a tranche merges, descendants rebase onto the resulting `main`. Every implementation
PR fills its `PR / status` and evidence cells before it becomes ready for review.

| ID | Recommended branch | Base | Packets | Kind | Bounded responsibility | PR / status |
|---|---|---|---:|---|---|---|
| B0 | #311 | `main` | 0a | prerequisite cutover | Land/use #311 and ratify its terminal contract as F1's exact base | current audited head `bc959ef3` |
| F1 | `quang/plan-digit-range-pipeline` | B0 | hub, 1-3 | additive foundation + atomic internal cutover | Land this hub, baselines/oracles, canonical range plan/domain, and the single `DigitRangeProver` cutover as one coherent first PR | **implementation active** |
| C3 | `quang/digit-range-02-streaming-cutover` | F1 | 4 | atomic compute cutover | Make the generic streaming prover canonical for LB2-LB6 and delete eager forest/padded-table production paths | planned |
| O4a | `quang/digit-range-03-lb4-kernel` | C3 | 5 | bounded optimization | Select and install the LB4 winner; delete candidates and knobs | planned |
| O4b | `quang/digit-range-04-lb5-kernel` | O4a | 5 | bounded optimization | Select and install the LB5 winner; delete candidates and knobs | planned |
| O4c | `quang/digit-range-05-lb6-kernel` | O4b | 5 | bounded optimization | Select and install the LB6 winner; delete candidates and knobs | planned |
| I5 | refreshed #309 integration branch | O4c + #309 | 0b | prerequisite integration | Rebase/land #309 over the current stack, resolve #311 terminal conflicts, and ratify the three semantic bases before relation/setup work | planned before C5 |
| C5 | `quang/digit-range-06-flat-relations` | I5 | 6 | atomic compute/API cutover | Move current direct relation/setup evaluation to the flat semantic providers; delete public x/y and sentinel paths with unchanged wire | planned |
| C6 | `quang/digit-range-07-direct-fold-check` | C5 | 7a | atomic direct cutover | Install the semantic direct fold-check container, keep direct bytes unchanged, unschedule recursive offload, and delete legacy recursive Stage 2/3 | planned |
| C7 | `quang/digit-range-08-recursive-two-stage` | C6 and #310 integration | 7b | atomic protocol cutover | Add the fused recursive Stage 1 leaf and `Separate` Stage 2 reduction across proof/prover/verifier/wire/sizing/schedules | planned |
| E8 | `quang/digit-range-09-batched-reduction` | C7 | optional target | atomic capability cutover | Add and emit `Batched` across plan/proof/prover/verifier/wire/sizing only if complete-size and differential gates beat `Separate`; otherwise add no production shape | planned / optional |
| M9 | `quang/digit-range-10-mixed-dimensions` | C7 or E8 | 8 | atomic compute/capability cutover | Extend canonical providers to mixed role dimensions under the fixed two-stage proof language and remove the mixed-plus-setup rejection | planned |
| M10 | `quang/digit-range-11-mixed-planner` | M9 | 9 | capability enablement | Price and schedule eligible mixed/offloaded candidates; regenerate only approved rows | planned |
| O11 | `quang/digit-range-12-prover-cleanup` | M10 | 10 | bounded optimization | Destination-oriented digit emission, nonce/fold workspace reuse, and invariant-bearing constructor split | planned |
| O12 | `quang/digit-range-13-verifier-kernels` | O11 | 11 | bounded optimization | Port only measured structured verifier kernels onto the canonical providers | planned |
| D13 | `quang/digit-range-14-docs-packing-handoff` | O12 | 12 | closure | Final audit, book/spec/profile synchronization, scalar packing handoff; no deferred cutover deletion | planned |

O4a/O4b/O4c are written linearly to minimize hot-file conflicts, but their candidate
experiments may run as sibling branches from C3. Only the measured winner for each basis
is rebased into the listed production order. E8 is optional: if `Batched` does not beat
`Separate`, record the stop result in this hub and skip directly to M9.

### F1: locked first PR contract

This is a settled delivery decision. The current planning PR becomes the first
implementation PR; do not merge it as a documentation-only waypoint. Its coherent claim
is:

> establish the canonical range architecture, prove that it preserves the current
> protocol, and cut every Stage 1 caller over to one production `DigitRangeProver`.

F1 combines the hub with Packets 1-3 because the pieces are mutually justifying: the
baseline/oracles make the refactor reviewable, `DigitRangePlan` removes duplicate shape
authority, and the architecture cutover ensures the new plan is exercised rather than
dormant. Splitting before the F1 cutover would either merge unused substrate or leave the
repository with two prover owners.

#### Base and branch rule

F1's exact base is the current #311 head recorded as B0. It consumes the checked concrete
range basis already carried by `RingSwitchOutput` and passes it directly to
`DigitRangePlan`; neither the plan nor the Stage 1 prover reads `LevelParams`. When I5
lands #309, the ring-switch producer supplies that same concrete value from
`log_basis_open`, so no compatibility wrapper or F1 API change is permitted. Record the
literal B0 and F1 head SHAs in this ledger before implementation review begins.

Recommended final PR title:

```text
refactor(prover): establish one digit-range architecture
```

#### Exact positive change surface

The final `git diff B0...F1` may touch only the following ownership regions. A necessary
path outside this manifest requires a hub amendment explaining the invariant it owns;
“the compiler needed it” is not sufficient.

| Surface | Allowed paths | Required final responsibility |
|---|---|---|
| Central hub and lifecycle | `specs/digit-range-pipeline-refactor.md`, `specs/akita-sumcheck-unification.md`, `specs/packed-sumcheck.md` | Keep this stack authoritative; record F1 base/head, benchmark method, status, and any deviation |
| Benchmark and counters | `crates/akita-pcs/Cargo.toml`, new `crates/akita-pcs/benches/digit_range.rs`, narrowly scoped Stage 1 tracing/counter sites | Pin LB2-LB6, live-prefix, digit-distribution, serial/parallel, allocation, and whole-Stage-1 baselines without changing production decisions |
| Range shape/domain authority | `crates/akita-types/src/proof/stage1.rs`, Stage-1-only exports in `crates/akita-types/src/proof/mod.rs` and `crates/akita-types/src/lib.rs`, the Stage-1 regions of `proof/{levels,shapes,wire}.rs` and `proof_size.rs`, and existing Stage-1 shape consumers in `crates/akita-types/src/schedule_tests.rs` and `crates/akita-pcs/src/scheme/tests/mod.rs` | One checked `DigitRangePlan` and one checked Stage-1 view of `WitnessDomain`; prover, verifier, shape validation, sizing, and their existing assertions consume them directly. These two test paths are named explicitly because deleting the old free topology helpers necessarily cuts their existing assertions over to the plan; they gain no production responsibility. |
| Canonical prover | new `crates/akita-prover/src/protocol/sumcheck/digit_range/`, `protocol/sumcheck/mod.rs`, the public export in `crates/akita-prover/src/lib.rs`, Stage-1-owned pieces of `two_round_prefix/{common,stage1}.rs`, the Stage-1 import seam in `protocol/core.rs`, and the Stage-1 boundary in `protocol/core/fold.rs` | One production `DigitRangeProver` owns construction, transcript choreography, claims, folding, and all LB2-LB6 dispatch |
| Removed prover surface | `protocol/sumcheck/akita_stage1_tree.rs` and `protocol/sumcheck/akita_stage1/` | Migrate invariant-bearing code into `digit_range/`, then delete both old prover owners and every pass-through export |
| Verifier parity | `crates/akita-verifier/src/stages/stage1.rs` and only the Stage-1 replay region of `crates/akita-verifier/src/protocol/core/fold.rs` | Consume `DigitRangePlan`/checked points, preserve equations and transcript order, and reject malformed shapes without panic |
| Differential tests and test-only oracles | `crates/akita-pcs/tests/stage1_roundtrip.rs`, narrowly scoped transcript-hardening tests, `digit_range/` unit tests, Stage-1-owned imports/assertions in existing `akita_stage2/tests.rs` and `two_round_prefix/tests.rs`, and test/bench-only dense range or relation helpers | Exhaust plans and malformed inputs; compare proof bytes, events, challenges, claims, points, and round polynomials against the pre-F1 oracle |

The new module should be organized by invariant-bearing state, not by basis or old
backend. Acceptable internal seams are plan validation, active representation state,
round production/folding, prefix optimization, and tests. Do not create `lb2.rs`,
`lb4.rs`, `compact_backend.rs`, `tree_backend.rs`, an `Engine` trait, or a facade that
forwards to the two old provers.

F1 may temporarily retain both **active representations** required by the current
protocol — compact digits for LB2/LB3 and padded range-image state for LB4/LB5/LB6 — but
they must be private states inside one prover and must not own separate plan, transcript,
proof-shape, or round-loop implementations. Both are exercised production states, not a
dormant alternate engine. C3 owns replacing the padded high-basis state with streaming;
F1 must neither optimize nor generalize it.

Within the new Stage 1 code, production identifiers use `digit_witness`, `range_image`,
and `range_image_eval`; do not introduce a new `S`-named table, claim, or helper. The
legacy `AkitaStage1Proof` type and any field name whose mechanical rename would pull
Stage 2/wire migration into this PR remain unchanged until their scheduled semantic
cutover. Do not add a compatibility alias for either vocabulary.

#### Mandatory deletion and single-owner gate

F1 is not ready until all of the following are true:

- exactly one non-test `struct DigitRangeProver` owns Stage 1 range proving;
- no non-test `struct AkitaStage1Prover` remains;
- `pub mod akita_stage1`, `pub mod akita_stage1_tree`, and their re-exports are gone;
- `prove_recover_w`, `take_w_evals_compact`, and the
  `mem::take -> prove -> restore` witness handoff are gone; the existing `Arc<[i8]>` is
  shared directly;
- range topology, arity, degree, child count/order, and round count are derived only from
  `DigitRangePlan` in prover, verifier, sizing, and shape validation;
- the old free topology helpers are private implementation details or deleted; no
  `_for_level`, forwarding constructor, compatibility reader, or runtime engine flag is
  added;
- every retained source file over the repository line limit is split only at an
  invariant-bearing boundary and passes the repository line guardrail without a blanket
  exception.

#### Explicitly forbidden in F1

F1 must not change:

- any serialized byte, proof field order/count, transcript label/order, challenge, claim,
  or final point for a fixed input and transcript seed;
- the high-basis padded-table/forest algorithm beyond moving and descriptively naming its
  private state; C3 owns the streaming replacement and its memory/speed claims;
- Stage 2 relation algebra, Stage 3 setup offload, relation placement, setup contribution,
  `FoldCheckPlan`, or recursive proof topology;
- public x/y geometry, mixed-role dimensions, setup slots, planner selection, generated
  schedules, commitment compression, terminal proof/checking, digit emission, fold grind,
  or verifier performance kernels;
- `two_round_prefix/stage2.rs` or production Stage 2/Stage 3 modules, except that test-only
  baseline code may observe their outputs without changing their behavior.

#### F1 ready-to-merge gate

Against the literal B0 SHA, F1 must demonstrate:

- byte-for-byte proof and logging-transcript identity for LB2-LB6 across direct and
  currently scheduled recursive non-terminal levels;
- identical Stage 1 round messages, challenges, interstage claims, final point, and legacy
  range-image claim;
- exhaustive `DigitRangePlan` shape tests and verifier rejection of malformed LB, count,
  degree, point width/order, and child shape without panic;
- each fixed Stage 1 benchmark cell within the ratified parity interval, no allocation or
  peak-memory increase, and no end-to-end primary workload above `1.02x`;
- a complete `B0...F1` ownership diff showing only the manifest above and an `rg` deletion
  report for the forbidden old owners/wrappers;
- the current repository-wide preflight, all three feature-matrix Clippy commands, focused
  `akita-types`/`akita-prover`/`akita-verifier` Stage 1 tests,
  `akita-pcs --test stage1_roundtrip`, transcript-hardening coverage, and the CI-profile
  nextest suite, each polled to a real exit code.

If preserving the old high-basis implementation inside one owner cannot meet parity or
requires a wrapper around either legacy prover, F1 stops. Do not pull C3 streaming work
backward merely to make the first PR pass.

### Per-PR review and merge protocol

Every PR in the stack must state:

- its ledger ID, exact base SHA, head SHA, and immediate predecessor PR;
- the cumulative tranche baseline used for correctness and performance comparison;
- the one production authority it adds, replaces, or optimizes;
- whether proof bytes/transcript are identical or intentionally cut over;
- deleted symbols and the `rg`/guardrail evidence that no second path remains;
- tests, benchmark cells, allocation results, and unresolved gates;
- the next ledger ID, without claiming that later work is already implemented.

Review the immediate `base...head` diff for boundedness and the tranche-base cumulative
diff for architectural coherence. Merge foundation PRs only when independently valid;
then merge the cutover promptly after its gates pass. If another Akita PR changes a base,
provider, schedule, or proof shape, update this hub first, rebase the earliest affected
unmerged PR, and rebuild all descendants. Never merge around the conflict with a wrapper.

## Execution map and risk register

The packet order is linear even where investigations may run in parallel. The PR ledger
above is the delivery authority; the packet graph below is the technical dependency map.
Before each cutover, its foundation packet's deletions, byte-identity gates, and benchmark
report must be complete. Packet 7 is deliberately split into two bounded cutovers:

```text
0a #311 terminal baseline (F1 base)
  -> 1 baselines and oracles
  -> 2 canonical range plans and checked points (wire-preserving)
  -> 3 one range-product architecture
  -> 4 streaming high-basis reference
  -> 5 measured per-basis winners
  -> 0b integrate #309 semantic inner/outer/open bases
  -> 6 flat relation-provider cleanup under the current wire
  -> 7a direct semantic cutover and recursive capability removal
  -> 7b atomic recursive two-stage protocol cutover
  -> 8 mixed common-base relation and setup providers
  -> 9 mixed-candidate planner rollout and cost model
  -> 10 upstream digit-emission cleanup
  -> 11 verifier performance port
  -> 12 documentation and packing handoff
```

Packet 4 may prototype Packet 5 candidates; Packet 6 may prototype the joint leaf and
Stage 2 proofs in benchmarks/tests; Packet 8's dense setup oracle may be prepared earlier.
Later production code must not merge early. This avoids benchmarking a moving semantic
target or carrying dual protocol shapes across packet boundaries.

| Risk | Failure mode | Required prevention / stop condition |
|---|---|---|
| Derived-table padding mistaken for zero | Valid proofs drift because omitted Stage 1 suffixes have nonzero leaf/product values | Differentially test every round against the dense oracle; require per-lane defaults and exact split-equality suffix mass before deleting it |
| Delayed-reduction overflow | Release-only proof corruption | Establish an accumulator-specific term bound and chunk reduction rule; reject a specialization whose bound cannot be proved |
| A pre-cutover cleanup changes the wire | Existing schedules drift before the versioned migration | Compare proof bytes, logging-transcript events, round messages, challenges, claims, and final points after every Packet 2-6 semantic move |
| Two-stage cutover is only partial | Prover, verifier, sizing, or transcript still interprets numeric Stage 3 fields | Packet 7a deletes the legacy recursive protocol while unscheduling the capability; Packet 7b adds the complete new recursive proof, plan, prover, verifier, serializer, transcript, sizing, and schedules atomically |
| Mixed-role point slicing is wrong | Direct and recursively proved full setup contributions disagree | Use typed points; test dense/factorized provider equality and direct/recursive full-scalar equality for every supported tuple |
| Generic abstraction hides algebra | Degree or batching coefficients are applied twice or omitted | No semantic engine traits or expression graph; Stage 1 and Stage 2 equations remain in their owning modules and are reviewed independently |
| Microbenchmark win regresses the prover | LUT construction, cache misses, allocations, or parallel overhead erase a hot-loop gain | Gate on whole-substage and whole-prover ratios as well as the kernel; delete the losing path and its knob |
| Future-feature scope creep | Inactive compressed/binary fields create ambiguous states | Add no production field or transcript slot; document only the provider seam and degree constraint |

The only deliberately unresolved items are benchmark experiments, not architecture:

| Experiment | Candidate set | Decision rule |
|---|---|---|
| LB4 class pair | direct field evaluation, field table, bounded narrow table | retain one complete-substage winner |
| LB4 initial two rounds | one-round baseline, bounded 4,096-key deferral | retain only if it improves the whole Stage 1 and code-size gates |
| LB5 product stage | direct fixed four-lane evaluation, field table | no narrow integer or full two-round table |
| LB6 product/leaf stages | direct versus field table at each 2/8-lane boundary | choose separately per boundary, then retain one static plan |
| Fused fold-next scan | separate fold/scan versus fused traversal | retain per stage only when end-to-end and allocation results improve |

## Implementation packets

Packets are technical acceptance scopes; the central ledger above maps them to bounded
PRs and is normative for delivery. A temporary oracle may exist under tests while its
replacement is differentially validated; it must not become a second production path.
Packets 0-6 are homogeneous transcript/wire preserving. Packet 7a is a byte-identical
direct cutover with recursive offloading deliberately unscheduled; Packet 7b is the
authenticated recursive protocol cutover. Packets 8-12 must be byte-identical to the new
protocol for a fixed `FoldCheckPlan`.

### Packet 0a/0b: establish tranche-specific external bases

**Goal:** make the implementation base unambiguous.

**Actions:**

- 0a: start F1 from the audited #311 head and record the exact post-cutover terminal
  proof/transcript oracle;
- 0a: pass the checked concrete range basis from `RingSwitchOutput` into
  `DigitRangePlan`, with no `LevelParams` dependency;
- 0b, after O4c and before C5: rebase and land the current
  `origin/refactor/multi-group-digit-decompose` series, resolving every terminal conflict
  in favor of #311's no-sumcheck contract;
- 0b: confirm `log_basis_open` dominates inner/outer bases, confirm the ring-switch range
  basis is sourced from `log_basis_open`, regenerate schedule artifacts, and ratify the
  exact I5 base.

**Stop:** do not implement F1 before #311. Do not begin C5 relation/setup work against a
legacy single `log_basis`; I5/#309 must land first.

### Packet 1: baselines, spans, and dense oracles (F1 foundation)

**Goal:** make every later optimization measurable and differentially checkable.

**May add:**

- `crates/akita-pcs/benches/digit_range.rs` and Cargo bench entry;
- prover tracing spans/counters for Stage 1 setup, compact scan, materialization, product
  substages, leaf substage, fold, and finalization;
- test-only dense fully padded range-tree oracle;
- test-only dense flat relation-weight materializer and evaluator;
- allocation and peak-field-element counters owned by the range stage.

**Must record:**

- current proof bytes and logging-transcript events;
- post-#311 terminal proof bytes/events as an immutable negative-diff oracle;
- current round polynomial, challenge, fold state, child claim, final point, and legacy
  range-image claim;
- complete actual/formula byte counts for direct, legacy recursive, target separate, and
  target batched shapes over equal and unequal native round counts;
- a reviewed soundness ledger for both topologies: exact claimed polynomials, batching
  challenge order, degree/round contribution to sumcheck error, independence of the two
  Stage 1 MLE evaluations, full setup-claim binding, and the extraction/opening handoff;
- per-basis `3N/5N/11N` table allocation evidence or corrected measured counts;
- whole-stage, substage, relation construction, verifier, and e2e timing.

The baseline is the post-Packet-0 main commit, with exact commit, Cargo features,
`rust-toolchain`, target triple/ISA, machine, core/thread policy, and command lines checked
into the benchmark report. Packet 1 also checks in:

- an exact Stage 1 and Stage 2 production ownership manifest down to files/functions;
- the `git diff --name-only` implementation-base classifier from `Intended implementation
  diff surface`, with every currently expected production path assigned once;
- a documented line-count script/method so moved code remains in the ownership closure;
- the exact benchmark cells used for geometric means and per-basis gates;
- a benchmark-only single-threaded counting allocator method for allocated bytes/count;
- a standalone-process max-RSS method using normalized `getrusage(RUSAGE_SELF).ru_maxrss`
  (or a checked equivalent on the target OS);
- an `llvm-size`/documented release artifact target and feature set for text-size tracking.

The soundness ledger is a pre-implementation gate for Packet 7, not documentation to add
after code. The numeric speed, deletion, and text-size targets below are the starting stop targets.
Packet 1 ratifies them against evidence before Packet 2 starts, or amends this spec with the
baseline report and rationale. They may not be silently relaxed in an implementation PR.

Benchmark setup and witness generation stay outside timed closures.

**Exit:** checked-in benchmarks run all primary LB4/LB5/LB6 and Stage 2 cases, ownership and
measurement methods are reproducible, numeric targets are ratified, and the old
implementation is captured as a test oracle. No protocol behavior changes.

### Packet 2: canonical range plans and checked points (F1 foundation, wire-preserving)

**Goal:** one range-topology/domain authority with byte-identical behavior.

**Touch:**

- `crates/akita-types/src/proof/stage1.rs` or its replacement;
- `crates/akita-types/src/{lib.rs,proof/mod.rs,proof/wire.rs}`;
- `crates/akita-types/src/proof/{levels,shapes}.rs`;
- `crates/akita-types/src/{proof_size,layout/proof_size}.rs`;
- `crates/akita-prover/src/protocol/core/fold.rs`;
- `crates/akita-prover/src/protocol/sumcheck/{mod.rs,two_round_prefix/stage1.rs}`;
- `crates/akita-prover/src/lib.rs`;
- `crates/akita-verifier/src/protocol/core/fold.rs`;
- `crates/akita-pcs/tests/stage1_roundtrip.rs`;
- Stage 1 prover/verifier construction and shape tests.

The `levels`, `shapes`, `wire`, `core::fold`, and verifier files are touched only in their
non-terminal regions. Packet 2 must show an empty semantic diff for #311
`TerminalLevelProof`, `TerminalLevelProofShape`, terminal serialization, and direct
terminal replay.

**Actions:**

- add `DigitRangePlan` and the checked Stage 1-facing flat domain/point object;
- restrict LB to 2..=6;
- route range-layer sizing, prover, verifier, and serialization validation through
  `DigitRangePlan`; complete `FoldCheckPlan` lands in Packet 7;
- replace panic-based coordinate reorder with checked point construction;
- differential-test the new ordered-point constructor against the old uniform mapping:
  physical x-major/ring-minor storage receives `[ring coordinates, column coordinates]`
  exactly as before; this packet adds no transcript absorption or point-order wire change;
- remove duplicate topology helpers and generic leaked range helpers;
- keep current proof bytes and transcript events identical.

**Exit:** every supported plan shape is exhaustively asserted; malformed basis/shape/point
returns errors; cleanup benchmark is within 3% and memory does not increase.

### Packet 3: one range-product architecture and module cleanup (F1 atomic cutover)

**Goal:** move compact LB2/LB3 mechanics and tree choreography into one `digit_range`
module before changing high-basis storage.

**Actions:**

- introduce one `DigitRangeProver` taking compact witness, plan, domain, and point;
- remove the take/prove-recover/restore ownership pattern;
- split old two-round-prefix common code by actual Stage 1 versus Stage 2 responsibility;
- share only flat pair/fold mechanics, not Stage 1/Stage 2 equations;
- rename/move modules in a mechanical commit;
- delete the duplicate prover type and pass-through exports.

**Exit:** one range-product prover and one `DigitRangePlan` constructor remain;
proof/transcript bytes are
identical; LB2/LB3/LB4/LB5/LB6 each stay within 3% before the high-basis optimization.

### Packet 4: streaming high-basis reference kernel

**Goal:** remove the eager forest while preserving exact wire behavior.

**Actions:**

- add checked `RangeImageClass` access;
- add field-valued node-by-class lookup construction;
- implement round-0 compact pair scan;
- implement `ExactPrefixTable` with per-lane nonzero defaults;
- implement actual split-equality suffix mass;
- materialize address-major lane state at `N/2`;
- free every substage before rescanning compact classes for the next one;
- implement the final quartic-from-range-image leaf stage;
- differentially compare every round/fold/claim against the padded oracle;
- delete the padded field-valued range-image table, retained leaf forest, retained product
  layers, and nested table production code.

**Exit:** proof and transcript bytes are identical; table peaks meet LB4 `N`, LB5 `2N`,
and LB6 `4N`, plus separately reported fixed state; the unspecialized reference path is no
slower than 1.02x on each Packet-1 primary basis cell using the ratified ratio rule. If it
cannot meet that gate, Packets 4 and 5 become one atomic, non-mergeable packet; do not
merge a slow reference path while waiting for specialization. Final improvement gates
remain owned by Packet 5.

### Packet 5: measured LB4/LB5/LB6 specialization

**Goal:** retain only per-basis optimizations that improve the complete prover.

Implement and gate in order:

1. LB4 two-lane field pair table versus direct evaluation.
2. LB4 bounded narrow/unreduced table experiment.
3. LB4 two-round defer experiment.
4. LB5 four-lane field table versus direct evaluation.
5. LB6 two-lane root, eight-lane second product, and quartic leaf table/direct choices.
6. Fused fold-next-scan where it improves the whole substage.

Every experiment includes construction time, allocations, cache behavior, serial and
parallel prove time, verifier time, and e2e effect. Delete the losing implementation; do
not retain a hidden knob or planner-visible kernel mode.

Do not rename the public proof object in this packet. Packet 7 creates the final semantic
`DigitRangeProof` and `DigitRangeRelationProof` types atomically with the wire/shape
cutover; no `final_s_eval` transition is allowed.

**Exit:** final scalar Stage 1 meets all performance gates below. LB2/LB3 do not regress.

### Packet 6: canonical flat relation and setup providers under the current wire

**Goal:** remove layout-shaped relation spaghetti and establish full-flat semantics before
moving any relation work between stages.

**Touch:** ring-switch relation construction; current Stage 2 prover/verifier; trace
providers; current setup replay/offload preparation; shared domain types.

**Actions:**

- replace the raw `RingSwitchOutput` geometry bundle with `WitnessDomain`, compact digits,
  semantic provider plans, ordered points, range plan, and alpha;
- reproduce the exact homogeneous challenge-to-physical-address map;
- refactor row-family construction into one flat semantic event emitter;
- add dense exact-live `RelationWeight` and `SetupRelationWeight` oracles;
- preserve and re-express the existing homogeneous factorized relation provider behind the
  new flat semantic API; for qualifying uniform layouts it remains the Packet 7 production
  provider with at most `N/g` state, while dense exact-live storage is test/oracle fallback
  only;
- expose one provider fold/evaluation contract usable by the direct Stage 2 composer and
  recursive Stage 1 joint leaf without either owning x/y layout vocabulary;
- add `SetupCoefficientDomain` and define the full `setup_contribution_eval` even though
  current consumers still use their legacy stage placement;
- make trace an additive address-mapped provider/view rather than allocating a remapped
  table, and prove it is included exactly once;
- share only compact/field pair-scan, bounded accumulation, fold, and parallel reduction;
- delete public `ring_bits == 0`, global x/y constructors, `x_prefix`, `y_prefix`,
  `sparse_y`, duplicate dense/prefix folds, and trace clones after parity.

Keep current Stage 1/2/3 proof types and transcript schedule in this packet. Target joint
leaf and claim-reduction kernels may exist only in tests/benches.

**Exit:** descriptor bytes, proofs, transcript events, challenges, claims, and points are
byte-identical; dense providers match current round polynomials at every round; public
constructors contain no raw x/y geometry; qualifying homogeneous production provider
state remains at most `N/g`; uniform performance is at most 1.02x baseline.

### Packet 7: bounded direct and recursive protocol cutovers

**Goal:** install the clean two-stage protocol through two independently atomic PRs,
without maintaining old and new recursive engines together.

Packet 7a is the direct semantic cutover. It lands the direct `FoldCheckPlan`, direct proof
container/composer, serializer, verifier, sizing, and schedules with byte-identical
`DirectSetup` behavior. In the same PR it disables recursive-offload schedules and deletes
the legacy recursive Stage 2/Stage 3 path. Recursive joint-leaf and claim-reduction code
may still exist in tests/benches, but not as an unused production engine. The temporary
capability gap is explicit and reviewable: after 7a, every non-terminal production level
is direct and every production proof has one canonical interpretation.

Packet 7b is the recursive protocol cutover. It extends the authenticated plan and proof
shape with `RecursiveSetupOffload` and lands the joint Stage 1 leaf, Stage 2 reduction,
prover, verifier, serializer, sizing, setup-route validation, planner selection, and
regenerated recursive schedules in the same PR. There is no legacy recursive decoder or
fallback to revive. #310's distributed recursive schedules must already be integrated or
7b cannot claim distributed coverage.

`Separate` is the complete Packet 7b scope. `Batched` does not land dormant in 7b: optional
ledger PR E8 adds its plan variant, proof shape, prover, verifier, sizing, validation, and
schedule emission atomically only after it beats `Separate` on complete-size and
differential gates. Existing direct and separate encodings remain unchanged because the
authenticated schedule selects a headerless shape; E8 is a new schedule capability, not
a compatibility migration. If the gate fails, no production `Batched` variant or code is
added.

**7a actions:**

- add the direct-only `FoldCheckPlan` and direct semantic proof container;
- migrate `next_witness_binding` into the common non-terminal envelope, preserving both
  #311 variants and exact bytes;
- route direct prover, verifier, serializer, deserializer, proof sizing, and schedules to
  the new direct authority;
- remove recursive-offload schedule rows and reject recursive plans at validation;
- delete the legacy recursive Stage 2/Stage 3 prover, verifier, proof fields, accessors,
  sizing branches, and decoders;
- preserve #311 terminal proof shape, transcript, direct checks, and bytes exactly.

**7b actions:**

- extend 7a's `FoldCheckPlan` and authenticate topology, domains, role dimensions,
  coordinate/lift convention, and exact setup slot before proof messages;
- retain direct setup's range-only Stage 1 and standard degree-3 Stage 2
  relation/range-image composer, now over the canonical flat providers;
- add the recursive standard joint final leaf with degrees 3/5, two independent folded
  lanes, signed-digit loads outside `RangeImageClass`, and the standard-leaf suffix rule;
- implement consuming `DeferredRangeRelationCheck` and close it with the full setup
  contribution before any Stage 2 challenge;
- implement `RangeImageConsistencyProof` and `SetupContributionProof`, stored directly in
  the recursive proof; validate opening-route eligibility before transcript mutation;
- extend proof types, shape descriptors, proof-size formulas, serializer/deserializer,
  transcript labels/version, prover, verifier, and schedules together without changing
  7a direct bytes;
- update the planner schema/emitter with `FoldCheckTopology`, with `Separate` as the only
  recursive reduction and no one-variant reduction enum; implement the complete-size
  selector and route/slot eligibility needed by current homogeneous candidates, and
  regenerate every affected homogeneous schedule artifact in this same packet;
- when #310 is present, update its one canonical distributed-setup-offload e2e fixture and
  schedule assertions from numeric Stage 3 to semantic recursive Stage 2; do not add a
  parallel digit-range-specific copy of that fixture;
- preserve 7a's common `next_witness_binding` envelope and both #311 payload cases; use
  `scaled_fold_witness` outside math;
- leave #311 `TerminalLevelProof`, direct terminal ring/trace checks, transcript order, and
  wire bytes unchanged; assert that `FoldCheckPlan` rejects terminal construction;
- assert that 7a already deleted `AkitaStage3Prover`, `SetupSumcheckProof`,
  `stage3_sumcheck_proof`, `BatchedStage3Geometry`, numeric-stage
  modules/fields/accessors, panicking cross-variant accessors, and legacy decoders/wrappers;
- enforce every degree and received count from `FoldCheckPlan` before allocation.

The 7b cutover is accepted only if actual serialization matches all direct/separate
formulas, non-terminal direct proof size is unchanged, #311 terminal bytes/events are
unchanged, and every scheduled recursive-offload target is no larger than its legacy
counterpart in complete bytes with measured verifier time improved.
Compare complete old `Stage1+Stage2+Stage3` with complete new `Stage1+Stage2` for the same
non-terminal fold; round-only arithmetic is insufficient.

**7a exit:** direct proof bytes/events are unchanged, recursive schedules are absent, no
production `stage3` identifier or legacy recursive proof reader remains, and no dormant
recursive production composer is present.

**7b exit:** direct proving time, witness scan count, allocation count, and bytes do not
regress beyond the cutover gate; recursive proof bytes do not increase and recursive
verifier time decreases against the legacy recursive baseline; prover/verifier logging
transcripts match the new semantic oracle exactly.

### Packet 8: generalize common-base relation and setup providers to mixed dimensions

**Goal:** extend Packet 6's homogeneous factorized provider to mixed role dimensions
without changing the Packet 7 proof language for a fixed plan.

**Actions:**

- implement the checked semantic-event-to-common-base compiler with additive
  linear/periodic patterns, exact partial intervals, and dense/sparse fringes;
- fold the low factor and high-lane table under canonical raw LSB-first binds;
- implement the shared role-expanded setup coefficient address helper;
- make dense, factorized direct replay, and recursively proved full setup contributions
  agree for every validated nested tuple, including `128/64/32`;
- derive `g` through `SetupCoefficientDomain::checked_common_base_view(role_dims)`; never
  store it as independently constructible proof metadata;
- validate exact slot, natural/padded length, commitment parameters, and opening route;
- test `Separate` independent points and `Batched` suffix/lift geometry on unequal domains;
- remove mixed-plus-setup rejection only after full-scalar parity passes.

**Exit:** factorized and dense providers produce identical messages, folds, final
values, proof bytes, and transcript events; qualifying provider state is at most `N/g`
plus explicit fringes; no sentinel, raw slice, or high-lane public setup claim remains.

### Packet 9: mixed-candidate planner rollout and cost model

**Goal:** extend Packet 7's topology selector to mixed candidates and admit them only when
the complete system benefits.

**Actions:**

- extend domain-derived round/size formulas and serialization parity to mixed layouts;
- compare complete direct, target separate, and target batched bytes for mixed candidates,
  using Packet 1's legacy baseline data for attribution; apply route/slot eligibility
  before scoring;
- score direct-versus-recursive on the complete objective from the selector section:
  measured verifier-time saving of the removed setup scan against the
  `ceil(log_b(q))`-deep carried-opening cost at the next level, with proof bytes as the
  no-regression constraint;
- add cache-aware preview and proof-byte attribution;
- enable fp128 `128/64/64` first if it survives all gates;
- emit `DirectSetup` with a diagnostic when recursive offload or mixed geometry is
  ineligible; never downgrade inside prover/verifier;
- regenerate mixed-enabled schedule tables only after the selected profile is approved;
  do not alter Packet 7's homogeneous topology schema.

**Exit:** homogeneous non-terminal direct levels retain their proof size and measured
behavior, terminal levels retain #311's sumcheck-free direct shape; at
least one mixed/offloaded schedule is end-to-end proven and verified, or the preview
records a stop decision and emits none.

### Packet 10: digit emission and fold-grind cleanup

**Goal:** remove upstream copies/allocations and tune LB4/LB5/LB6 decomposition.

**Actions:** implement destination-oriented decomposition, flat centered storage, reusable
grind workspace, delayed accepted-nonce field conversion, and measured const-specialized
loops. Split `RingRelationProver::new` only at invariant-bearing artifact boundaries.

**Exit:** exact emitted witness bytes are unchanged; rejected nonce attempts reuse storage;
each retained specialization beats the generic direct-emission kernel and no allocation
count grows.

### Packet 11: verifier performance port

**Goal:** optimize structured verifier relation evaluation after canonical semantics land.

**Actions:** rederive/port useful prefix-scan and carry-bucket algebra from the divergent
combined-kernel branch; adapt it to both relation placements and mixed common-base views;
keep the dense evaluator as oracle; preserve the consuming deferred-check boundary and
remove obsolete branch vocabulary.

**Exit:** verifier output is identical for direct/offloaded and uniform/mixed layouts,
malformed inputs stay panic-free, and every primary verifier benchmark is at most 1.02x
baseline with a material win on targeted structured cases.

### Packet 12: docs and downstream packing handoff

**Goal:** make code, book, active specs, and profiles agree.

Packet 12 is a closure audit, not a place to finish old-code deletion. If a superseded
production path from C3, C5, C6, or C7 still exists, reopen that cutover gate instead of
deferring the cleanup here.

Update at least:

- `book/src/how/proving/sumcheck-stages.md`;
- `book/src/foundations/eq-factored-sumcheck.md`;
- `book/src/foundations/multilinear-sumcheck.md` and `book/src/how/security.md` for the
  reviewed degree/soundness ledger;
- `book/src/how/architecture.md`;
- `book/src/how/verification.md` and `docs/verifier-contract.md` if boundaries change;
- `docs/soundness-audit.md` for the new statement/transcript/opening invariants;
- profiling documentation and exact commands;
- spec lifecycle metadata.

This planning branch already marks
[`akita-sumcheck-unification.md`](akita-sumcheck-unification.md) superseded and blocks
Stage 1/2 work in [`packed-sumcheck.md`](packed-sumcheck.md). Keep that lifecycle metadata
correct. Preserve the old spec's diagnosis, separation of proof
format/batching/compute, Boolean-only invariant, and byte-identical fast-path rule;
explicitly reject its general descriptor algebra and new crate. Port the smaller
`e87295b7` kernel-cutover concepts into this architecture rather than importing the stale
branch spec as a second authority.

After scalar completion, [`packed-sumcheck.md`](packed-sumcheck.md) may use the contiguous
address-major lane buffers and accumulator bounds. Scalar/NoPacking remains the exact
oracle.

## Benchmark program

### Range product and final-leaf-composer benchmark

Add `crates/akita-pcs/benches/digit_range.rs`. The path is fixed here so Packet 1's
ratified measurement method stays stable; moving the bench invalidates the recorded
baseline commands. Report these phases separately:

- plan and lookup-table setup;
- compact class conversion/source scan;
- initial compact round;
- every product substage;
- direct range-only leaf;
- recursive standard range-and-relation leaf;
- range-image lane, signed-witness lane, and relation-provider work inside that leaf;
- materialization;
- later pair scan;
- fold and fused fold-next-scan;
- whole direct Stage 1 and recursive Stage 1 prove;
- verifier replay.

Primary matrix:

| Axis | Values |
|---|---|
| Field | fp128 primary; fp64/Ext2 and fp32/Ext4 smoke |
| LB | 4, 5, 6; LB2/LB3 regression |
| Domain | `2^18` repeatable; `2^22` manual profile; cross-field `2^16/2^18` |
| Live ratio | 100%, 75%, exact schedule-derived partial prefix |
| Digits | uniform balanced, zero-heavy, alternating negative/positive endpoints |
| Threads | one Rayon thread, production parallel pool |

Measure:

- median, confidence interval, and ns/live digit;
- per-substage wall time;
- peak range-owned field elements and bytes;
- total allocated bytes and allocation count;
- compact-to-field conversion count;
- full-table scan/fold count;
- standalone process max RSS;
- verifier time;
- proof bytes and `LoggingTranscript` event count.

Use a fixed toolchain, machine, ISA, thread count, and thermal/background policy. Use at
least 20-30 Criterion samples for primary microbenchmarks and store raw artifacts/allocation
reports. Hard-gate deterministic bytes, events, field-element peaks, and microkernels;
report noisy e2e timing rather than creating flaky CI thresholds. Run baseline/candidate
microbenchmarks in interleaved paired batches; a `<=` timing gate uses the upper bound of
the Packet-1-ratified 95% paired ratio confidence interval, not a favorable single median.

### Relation placement, Stage 2, and mixed-dimension benchmark

Matrix:

- uniform `64/64/64`, `128/128/128`;
- mixed `128/64/64` and `128/64/32` under every eligible topology;
- direct relation/range-image proof, recursive `Separate`, and recursive `Batched`;
- setup/witness native round ratios `lambda:mu` of `1:2`, `1:1`, and `2:1`;
- trace absent/present;
- common range present/terminal absent;
- one and multiple witness groups/chunks;
- dense oracle, common-base provider, and retained sparse provider;
- initial round batching on/off for measurement only;
- exact partial live intervals and unaligned fringe fallback tests.

Report relation construction separately from the direct Stage 2 scan and recursive Stage
1 joint leaf. Report recursive consistency, setup, and batch work separately. Confirm the
expected `N/g` provider storage and construction reduction rather than hiding it in
whole-prover noise. Always include complete old `Stage1+Stage2+Stage3` versus new
`Stage1+Stage2` timing, allocations, and proof bytes.

### Decomposition/fold-grind benchmark

Vary:

- LB4/LB5/LB6;
- ring dimensions 32/64/128/256;
- ordinary and overflow coefficient paths;
- random canonical, near-modulus, negative centered, and digit-boundary inputs;
- generic direct destination versus retained specialization;
- first accepted nonce and multiple rejected nonce attempts;
- singleton, multi-group, and multi-chunk witnesses.

### E2E profiles

At minimum:

- fp128 D64 one-hot nv32 np1 direct and recursive; record the actual bases exercised by the
  generated schedule and add pinned deterministic synthetic schedules/workloads for each
  of LB4, LB5, and LB6 rather than relying on generated choices remaining stable;
- fp128 batched nv30 np4;
- fp32 and fp64 D128 nv28 representatives;
- first approved mixed fp128 schedule;
- multi-group and multi-chunk cases.

Record range-product, final-leaf-composer, relation-provider, and recursive reduction shares
of total time before interpreting an e2e percentage.

## Ratified performance and cleanup gates

Every ratio is against the Packet 1 post-Packet-0 baseline with identical features,
toolchain, machine, ISA, inputs, and threads. Packet 1 must ratify the confidence-interval
procedure and numeric targets before implementation begins.

### Correctness/cleanup parity gate

For Packets 0-6, and for compute-only Packets 8-12 relative to Packet 7's new oracle:

- serialized proof bytes identical;
- logging transcript events identical;
- paired timing ratio remains within the ratified +/-3% parity interval;
- no peak-memory or allocation-count increase;
- malformed inputs remain error-returning and panic-free.

Packet 7 uses its own intentional-cutover gate below; old/new bytes need not be identical
there.

### Final scalar range-product gate (Packet 5)

All must hold on the fixed primary machine. The geometric mean cell set is every fp128
`2^18` primary combination of LB4/LB5/LB6, the three live-prefix cases, the three digit
distributions, and both thread modes declared above. The LB6 primary target is fp128,
`2^22`, 100% live, uniform balanced digits, production parallel pool.

- geometric mean LB4/LB5/LB6 prove time `<= 0.75x` baseline;
- every LB4/LB5/LB6 case `<= 0.90x` baseline;
- LB6 primary target `<= 0.70x` baseline;
- verifier `<= 1.02x` baseline;
- no LB2/LB3 case `> 1.02x` baseline;
- mandatory peak current-substage tables, excluding separately reported fixed LUT/split-eq
  state: LB4 `<= 1.05N`, LB5 `<= 2.05N`, LB6 `<= 4.05N`;
- aspirational retained two-round kernels: LB4 `<= 0.55N`, LB5 `<= 1.05N`, LB6
  `<= 2.05N`;
- old proof bytes remain identical;
- no primary e2e workload `> 1.02x`; representative nv32 improves at least 5% when the
  baseline shows Stage 1 is at least 10% of proving time. These e2e thresholds are manual
  fixed-machine release gates, not noisy CI pass/fail checks.
- total range-owned peak bytes, including lookup tables and split-equality state, decrease
  from baseline; allocation count and total allocated bytes do not grow in any primary
  LB4/LB5/LB6 cell.

### Atomic protocol-cutover gates (Packets 7a and 7b)

Packet 7a must first prove direct byte/transcript identity, preserve #311 terminal
behavior, remove every legacy recursive Stage 2/3 production path, and emit no recursive
schedule. Packet 7b then applies all recursive requirements below against Packet 1's
legacy recursive baseline; it may not rely on a compatibility decoder or fallback path.

All of the following are hard requirements:

- intermediate `DirectSetup` serialized size is exactly unchanged; no scalar, round, or
  envelope field is added there;
- #311 `TerminalLevelProof` serialization and transcript events are byte-for-byte
  unchanged, and no terminal fold-check shape can be constructed;
- each scheduled recursive target is no larger than its actual legacy proof in complete
  serialized bytes — a round-only formula is insufficient, and byte parity is admissible
  only with the verifier-time improvement below;
- each scheduled recursive target's measured verifier time improves on its legacy
  recursive baseline;
- in 7b, formula sizes equal actual serialization for direct and separate shapes; E8 must
  add the same gate for batched before that shape exists in production;
- direct complete fold-check prove time is at most `1.02x` its Packet 6 baseline in every
  primary cell, with identical signed-witness scan/fold count and no allocation growth;
- recursive complete fold-check prove time is at most `1.02x` baseline in every primary
  cell and has geometric mean at most `0.95x`; Packet 1 may ratify different numeric values
  only with checked-in evidence before implementation starts;
- recursive joint-leaf folded witness state is `N` field elements; total joint-leaf state,
  including provider, is at most `N + N/g + explicit_fringes + 5%` for qualifying
  common-base cells;
- verifier time is at most `1.02x` baseline;
- semantic transcript events exactly match the new versioned oracle, with setup and
  range-image frames disjoint;
- no legacy decoder, Stage 3 wrapper, or runtime topology heuristic remains.

### Code deletion gate

At the end of Packet 5, measured with Packet 1's checked-in Stage 1 ownership manifest and
line-count script:

- non-test Stage 1 production code falls by at least 35% and at least 1,250 lines from the
  measured `akita_stage1_tree.rs` + `akita_stage1/*.rs` + Stage 1 prefix share baseline;
- a production module above 500 lines or hot kernel above 160 lines triggers mandatory
  line-by-line review and written exception; do not game the trigger with forwarding
  helpers or artificial file splitting;
- `rg` finds no production `Vec<Vec<Vec<E>>>`, padded field-valued range-image builder,
  all-level range-tree
  builder, duplicate shape formula, or second Stage 1 prover;
- at the flat cutover, `rg` finds no public Stage 1/Stage 2 constructor with
  `live_x_cols`, `col_bits`, or `ring_bits`;
- digit-range release/monomorphized text on Packet 1's fixed artifact/features grows at
  most 10% under the documented `llvm-size`/symbol method;
- any specialization that fails its basis gate is removed, not left dormant.

At the end of Packet 7b, measured with Packet 1's combined Stage 2/Stage 3 ownership closure
(including moved provider/pair-scan code and stage-owned prefix functions):

- non-test relation/range-image/setup-reduction production code falls by at least 30% and
  at least 900 lines relative to the old Stage 2 plus Stage 3 closure;
- `x_prefix`, `y_prefix`, `sparse_y`, and the old dual-geometry dispatch modules are gone;
- public final-leaf-composer/reduction constructors do not exceed the two scheduled
  topologies and one recursive reduction shape; E8 may raise the latter to two only when
  its batched gate passes;
- moving code to a generic/provider module does not remove it from the measured ownership
  closure;
- allocation count and total allocated bytes do not grow on any direct uniform primary
  cell; recursive peaks meet the joint-leaf/provider gate above.

Code deletion, memory, and speed are coequal. A fast dual architecture fails. A clean but
slower architecture fails.

## Correctness and security test matrix

### Range polynomial and class tests

- Exhaust every valid digit for every LB2..LB6.
- Check symmetry pairs `digit` and `-digit-1` map to the same `RangeImageClass`.
- Check endpoints `-b/2` and `b/2-1`.
- Reject just-below and just-above digits at the internal boundary.
- Compare integer leaf roots/coefficients against an independent slow polynomial builder.
- Verify every production plan topology and child order.

### Range product and recursive joint-leaf differential tests

For old padded oracle versus streaming implementation, compare after every operation:

- initial input claim;
- ordinary round polynomial coefficients/message;
- sampled challenge;
- explicit folded lane records;
- implicit lane defaults;
- equality suffix contribution;
- child claims and ordering;
- interstage batched claim and next point;
- final `range_image_eval` and point on the direct range-only path;
- serialized proof;
- transcript events.

Cover all-zero witness values and full, random, odd, and every short positive non-power
live length; separately verify that `live_len = 0` is rejected. Use poisoned backing
storage after the live prefix; randomized batches whose leaf/product default is nonzero;
serial and parallel execution; fp128, fp64/Ext2, fp32/Ext4.

For the recursive joint leaf, compare a dense standard-sumcheck oracle after every round:

- leaf input claim/anchor and relation claim are fixed before
  `range_relation_batch_challenge`;
- the class-derived range-image lane and signed digit lane are distinct sources;
- round polynomial has planned degree 3 for LB2 and 5 for LB3-LB6;
- standard implicit-tail contribution includes current equality scalar/linear factor once;
- relation provider is folded once and trace appears exactly once;
- `range_image_eval` and `digit_witness_eval` are at the same point but are never equated
  off-cube;
- dense and streaming messages, challenges, folds, points, and semantic transcript events
  agree.

### Direct relation and flat/mixed provider tests

- Uniform flat adapter versus the current x/y implementation at every round.
- Dense semantic relation weights versus common-base provider for alpha 0, 1, and random.
- All validated nested triples, including `128/64/32`.
- Multi-group, multi-chunk, overlaps, partial active intervals, and partial A-width columns.
- Factorized fold state and round messages versus dense at every bind.
- Trace provider absent/present and exact-address mapping without remap allocation.
- Range support adjacent to unchecked/field support; no obligation bleeding across segments.
- A pure pair-kernel test sums two caller-supplied range bindings with distinct coefficients
  and supports; no future transcript slot, proof field, or production feature is created.
- Initial round windows split at segment/range/live boundaries.
- Reject or correctly fall back on unaligned spans and local address maps that do not
  preserve the common low coordinates.

For intermediate `FoldCheckTopology::DirectSetup`, compare the cleaned standard degree-3
composer with the legacy proof round by round and assert unchanged scalar count, serialized size,
signed-witness scans, and direct full setup evaluation.
Its equality anchor must be a checked `RangeCheckPoint`; wrong-length/raw points and a
`RangeRelationPoint` from the other topology are rejected by construction or validation.

### Recursive Stage 2 consistency, setup, and geometry tests

- Direct setup replay equals the setup part of the factorized relation evaluator.
- Direct local replay equals the recursively proved full setup contribution at the same
  arbitrary witness-domain point.
- Perturb every expanded role sublane independently to catch repeated column-equality bugs.
- `RangeImageConsistencyProof` matches a dense equality-factored oracle round by round.
- `Separate` produces independent `SetupOpeningPoint` and `NextWitnessPoint`; reject it
  when the opening router cannot carry both.
- `Batched` matches independent native proofs for equal and unequal domains; inactive
  leading rounds emit half the current claim; no active message or final evaluation
  double-applies its lift scale.
- Setup/range batching challenge is absent in `Separate` and sampled only after both native
  claims in `Batched`.
- Setup and range-image transcript frames are disjoint in `Separate`: mutating any absorb
  inside one frame changes no challenge drawn before the other frame's first absorb, and
  the frames' semantic labels never interleave.
- Witness carry uses the full raw witness-domain point and is invariant under setup-only
  factorization refactors.
- `128/64/64` recursive mode uses D64 slot when present.
- For a scheduled direct candidate, `128/64/32` needs no recursive slot; a scheduled
  recursive proof rejects missing D32 or coerced D64 slots without changing mode.

### Verifier no-panic and allocation tests

Reject with `AkitaError` or serialization error:

- invalid LB and inconsistent basis descriptor;
- zero/overflow live lengths and non-power domain lengths;
- malformed segment ranges, overlaps, role dimensions, and local address maps;
- wrong point length/order/projection version;
- wrong subproof/child/round counts;
- oversized coefficient/claim vectors;
- unsupported provider, projection, or setup modes;
- missing recursive setup slot or dimension mismatch;
- extra/missing fields for the authenticated headerless `FoldCheckPlan` shape;
- received polynomial lengths or degrees that disagree with the plan;
- terminal proofs containing range/claim-reduction payloads or setup offload.

Fuzz/proptest malformed proof bytes and validated-then-mutated plans. No verifier-reachable
`assert!`, `panic!`, `unwrap`, unchecked index, or unbounded allocation is allowed.
Compile/API tests should make `DeferredRangeRelationCheck` consumable exactly once; a
runtime mutation of `setup_contribution_eval` must fail the final Stage 1 equality.

### Planner/proof-size tests

- Formula bytes equal actual serialized bytes for homogeneous and mixed levels.
- Non-terminal direct target bytes equal the non-terminal direct post-#311 baseline;
  terminal bytes equal the post-#311 direct-terminal baseline; each recursive target is
  no larger than its matching legacy baseline in complete bytes, and its recorded
  verifier-time delta is negative, before it is schedulable.
- Complete-size selection agrees with serialization even when its winner differs from the
  round-only `2(lambda+mu)` versus `3 max(lambda,mu)` comparison.
- Round count depends only on the canonical Boolean domain.
- Homogeneous schedules remain the special case of mixed metadata.
- Cache keys are counted once per schedule and rejected candidates report attribution.
- Planner cannot enable mixed roles or recursive offload until the selected Stage 1
  relation placement, Stage 2 reduction, trace, chunk, route, slot, and descriptor
  capabilities are all present.

## Rejected designs

The implementation must explicitly reject:

- preserving dual compact-small/eager-large Stage 1 engines;
- preserving dual flat and global x/y public Stage 2 APIs;
- `ring_bits == 0` as a mixed-layout sentinel;
- common-base `g` as a wire x/y split;
- `d_a` as the mixed low factor;
- per-role challenge domains or separate role sumchecks;
- alpha-inverse offset tricks;
- assuming all derived Stage 1 tails are zero;
- materializing a padded field-valued range-image table, all leaves, or all product layers;
- giant/fixed-width LB5/LB6 integer parent LUTs;
- full two-round LB5/LB6 four-class LUTs;
- binary-only product topology as a production option: it preserves round coefficients
  while adding substages, child claims, scans, and at least equal memory traffic;
- one module, trait, or helper family per LB;
- a general protocol expression graph/new crate/mandatory slow Tier-A engine;
- forcing trace into the common alpha factorization;
- repeating one witness-column equality value across mixed role sublanes;
- caller-owned relation/setup/witness point slices;
- batching independently committed oracles only because padded round counts match;
- keeping recursive-offload relation checking in Stage 2 and retaining a numeric Stage 3;
- forcing the recursive Stage 1 relation placement onto direct setup despite its extra
  signed-witness fold and scalar;
- implementing the two final-leaf composers as two complete range-product engines;
- removing future negative-binary digits from the common balanced range proof;
- direct degree-16/32 LB5/LB6 range polynomials;
- direct degree-8 LB4 or alternative LB6 tree in the production series without a separate
  Pareto protocol spec and planner/security repricing;
- encoding CPU kernel choice in schedules/proofs;
- cherry-picking stale mixed/relation/verifier branches wholesale;
- unsafe SIMD before scalar cleanup and packing-ready accumulator bounds;
- any pass-through alias or `_for_level` wrapper forbidden by the repository's
  single-source policy.

## Validation commands

`AGENTS.md` is the single authority for the current repository-wide preflight, feature
matrix, and CI-fidelity selectors; do not preserve a stale duplicate command list here.
Every implementation PR runs that current preflight plus focused tests and benchmarks for
its ownership surface, and polls every live command to a real exit code. F1's additional
focused requirements are fixed in its ready-to-merge gate above. Before the complete
series merges, run the current CI-profile nextest suite and every path-specific workflow
triggered by the cumulative diff.

Follow the repository dependency-cache policy before the first `lake`/Lean command if a
future cross-repository validation adds one; no Lean validation is currently required.

For a Markdown-only F1 head, `git diff --check` and
`./scripts/check-doc-guardrails.sh` are sufficient. Once its first Rust implementation
commit lands, the full F1 gate applies; the prior documentation-only validation is not
evidence for the implementation head.

## Definition of done

This work is complete only when:

- one `DigitRangePlan` defines every supported range topology and one `FoldCheckPlan`
  defines every complete wire shape;
- one compact flat-addressed range prover handles LB2..LB6;
- LB4/LB5/LB6 meet the scalar speed and memory gates;
- the eager padded field-valued range-image table and retained product forests are gone;
- one shared mechanical pair/fold toolkit supports the direct composer and recursive
  consistency/setup reducers without hiding their equations;
- common-base factorization meets dense round-by-round parity and `N/g` storage;
- direct setup retains one signed-witness relation/range-image scan; recursive offload uses
  the joint Stage 1 leaf with one fused round-zero witness traversal plus selected Stage 2
  reduction;
- trace appears exactly once at the scheduled relation placement without sharing false
  factorization;
- direct mixed setup replay and recursively proved full setup contribution agree; typed
  Stage 1 relation and Stage 2 setup/witness points replace caller-owned slices;
- recursive setup offload fails closed outside available `d_setup` slots;
- planner enables only measured, cache-aware mixed schedules;
- verifier malformed-input tests satisfy the no-panic contract;
- old x/y/prefix/tree wrappers and duplicate shape formulas are deleted;
- code deletion, proof size, transcript, allocation, microbenchmark, e2e, and documentation
  gates all pass;
- numeric Stage 3 code, fields, shapes, accessors, and legacy decoder are gone;
- #311 terminal proof bytes, transcript, direct ring/trace checks, and no-sumcheck topology
  are unchanged;
- compressed commitments and the fused negative-binary feature remain unimplemented but
  can land as new domains/providers/supports without another coordinate rewrite.

The implementation team should optimize in this order: eliminate the eager range forest,
consolidate the range prover, establish flat relation/setup semantics under the old wire,
perform the atomic two-stage cutover, prove the common-base mixed providers, enable only
winning schedules, then tune upstream digit emission and verifier arithmetic. Do not
trade this order for another round of local patches inside the current spaghetti.
