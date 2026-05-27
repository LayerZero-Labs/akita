# Setup Claim Offloading Stack

This document is the durable workflow for the setup-claim-offloading series.
The stack is semantic, not path-generated: each branch is a real review branch
whose diff should compile and make sense against its parent.

The important update is that this is no longer a single linear ladder. The
layout cutover is the shared foundation, but most of setup offloading should be
split into parallel lanes with explicit integration points.

## Invariants

- `main` is the base for the spec PR and for the first implementation branch.
- Stack branches are source-of-truth branches, not disposable materializations
  from a larger integration branch.
- A scratch integration branch may exist for smoke testing the full tip, but it
  must not drive PR branch generation by path restoration.
- Each implementation PR must be locally coherent and green for its focused
  checks.
- Force-pushes are allowed while PRs are drafts, but only with
  `--force-with-lease`.
- After changing a parent branch, rebase affected children and inspect
  `git range-diff` before pushing.
- Generated schedule tables, lockfiles, and benchmark updates should be
  isolated when they would obscure protocol-review diffs.
- Do not introduce compatibility shims for old setup layouts. Akita makes no
  backward-compatibility guarantees.

## Dependency DAG

```text
00 docs/spec
   |
01 packed setup layout full cutover
   |
   +--> 02A materialized setup inner-product oracle
   |        |
   |        +--> 03A succinct omega_S evaluator
   |        |
   |        +--> 03B setup product-sumcheck skeleton
   |
   +--> 02B setup prefix commitment artifacts
   |
   +--> 02C recursive carried-opening batching
   |
   +--> 02D proof-size/gating policy scaffolding

03A + 03B + 02B + 02C + 02D
   |
04 setup-offloading integration
   |
05 tables, benchmarks, cleanup
```

`02C` is intentionally independent of the packed setup layout at the math
level. It may be developed in parallel from `main`, but its review PR should be
rebased onto `01` before integration so proof structs and transcript labels do
not fork.

## Review Branches

Branch names below are collaboration-neutral suggestions. The live spec PR may
still sit on an author-prefixed branch, but new implementation branches should
not encode one person's ownership.

| ID | Suggested branch | Base | Scope | Exit criterion |
|----|--------|------|-------|----------------|
| 00 | `setup-layout-repack` | `main` | Spec-only PR. Preserve the packed setup-layout design and this stack plan. | Diff contains only `STACK.md` and `specs/setup-layout-repack.md`. |
| 01 | `setup-layout-repack-impl` | `main` after 00 lands | Full layout cutover: remove `max_stride`, introduce packed base A/B/D role views, split ZK B/D blinding out of base `S`, update all existing setup-matrix consumers, and keep direct verification/proving working. | No `setup.seed.max_stride` users remain; direct prover/verifier tests pass; no setup-prefix commitments or setup offloading proof objects. |
| 02A | `setup-inner-product-oracle` | 01 | Express the direct setup contribution as `<S_{<=N}, omega_S>` with a materialized weight oracle and equivalence tests against the packed direct scan. | Materialized `<S, omega_S>` matches current packed setup contribution for root, recursive, batched, and A/J fixtures. |
| 02B | `setup-prefix-ladder` | 01 | Add preprocessing metadata, prefix-slot policies, and commitment-hint storage for selected power-of-two prefixes of the flat setup coefficient vector. No offloading proof yet. | Setup metadata names available prefix slots; runtime can select the tightest eligible prover-ready slot; below-threshold or unavailable shapes fall back to direct scan or produce a configured missing-slot error. |
| 02C | `recursive-carried-openings` | 01 for review; can prototype from `main` | Replace singleton recursive carry state with a carried-opening claim list and root-style incidence at the recursive boundary. Use a common padded field domain first. | Existing singleton recursion is the size-one incidence case; a witness-plus-dummy-setup carried batch verifies end to end. |
| 02D | `setup-offload-gating` | 01 | Add the policy surface for `D_setup`, `N_min`, eligible levels, selected prefix length, and direct fallback. | The prover/verifier agree on eligibility and selected prefix without changing proof semantics yet. |
| 03A | `setup-weight-evaluator` | 02A | Implement succinct verifier evaluation of `omega_S(rho_lambda, rho_y)`, including root digit-fast A/J carry-DP and recursive block-fast D/B/A views. | Evaluator matches materialized `omega_S` at random points without scanning `S`. |
| 03B | `setup-product-sumcheck` | 02A | Add the setup product-sumcheck skeleton over the selected prefix. First implement it as a post-Stage-2 Stage 3 closed against a local/materialized setup opening oracle. | Product-sumcheck checks `<S_{<=N}, omega_S>` against the materialized oracle and binds `sigma_S`, `rho`, `alpha`, `tau_1`, and `r_x`; no extra witness claim is left unresolved. |
| 04 | `setup-claim-offloading` | integration of 02B, 02C, 02D, 03A, 03B | Replace the direct setup scan with delegated setup claims when eligible. Carry `(rho_lambda, rho_y, s_rho)` into the next recursive fold and batch it with the folded-witness opening. | Eligible root/L1 proofs verify without local setup scan; ineligible or terminal-without-next-fold cases use direct fallback. |
| 05 | `setup-offload-tables-tests` | 04 | Regenerate tables if needed, add broad benchmarks/regression tests, and remove temporary comparison or oracle helpers. | End-to-end proof size and verifier-time breakdowns reflect setup offloading under representative batched workloads. |

## Parallel Work Slices

### Slice 01: Packed Layout Foundation

This slice is the unavoidable merge spine. It removes the old physical setup
contract and all current consumers must move together. Do not split this into
half-cutovers that leave both `max_stride` and `max_setup_len` as live protocol
paths.

Owned files will likely include setup seed serialization, descriptor binding,
setup generation/cache validation, role-view helpers, prover commitment paths,
verifier direct commitment paths, ring-switch quotient kernels, and
`compute_setup_contribution`.

### Slice 02A/03A: Weight Algebra and Evaluators

This lane is mostly verifier math and tests. It should not need prefix
commitments or recursive proof changes. Start with materialized `omega_S`, then
replace the verifier side with the succinct evaluator.

The key deliverables are:

- alpha lives in the weight, never in committed setup;
- overlapping A/B/D raw coordinates add their weights;
- root A remains digit-fast because one-hot is root-only and requires it;
- root A/J uses the row-aware carry-DP evaluator;
- recursive D/B/A can use block-fast views when recursive setup offload is
  enabled.

### Slice 02B: Prefix Commitment Artifacts

This lane is preprocessing and metadata. It should not know the product
sumcheck internals. Its job is to make selected committed prefixes available.

A prefix slot is keyed by the setup identity and the commitment shape:

```text
(setup digest/layout tag, D_setup, N_prefix, commitment params)
```

For verifier metadata, a populated slot stores the selected prefix commitment
and its public shape. For prover metadata, the same slot must also store the
commitment hint material needed to later batch the setup opening claim:

```text
RingCommitment + AkitaCommitmentHint
```

The hint includes the decomposed inner rows, i.e. the `t_hat` material produced
by the inner Ajtai commitment, and any recomposed rows or ZK blinding digit
streams required by the active feature set. Without this prover-side material,
the offloading path would have to recompute the setup-prefix commitment witness
right when it is trying to batch the carried `S_{<=N_setup}` opening.

The general runtime selection rule is still:

```text
N_prefix = 2^ceil(log2(N_active^F))
delegate iff N_prefix >= N_min
N_setup = N_prefix
```

Initial constants:

```text
D_setup = 32
N_min = 2^23 field coefficients
```

If the proof level has `D != D_setup`, delegation is rejected at that level and
the verifier uses the direct setup computation.

The prefix artifact policy should support at least two populated-slot modes:

- `FullLadder`: generate every power-of-two prefix between `N_min` and
  `N_max`. This is the durable deployment policy when one setup artifact should
  serve many batching shapes.
- `SelectedSlots`: generate only an explicit list of prefix lengths. This is
  the right policy for CI, benchmark fixtures, and single-config deployments
  where the active root shape is known in advance.

Missing slots should be handled by policy, not by hidden recomputation:

- `StrictError`: fail with a setup/policy error if delegation was requested but
  the selected prefix slot is absent. This is the production-preprocessed mode.
- `GenerateAndPersist`: prover-side convenience for benches and local
  development; compute the missing slot, persist it, and bind the resulting
  commitment normally. The verifier still consumes explicit metadata.
- `DirectFallback`: skip setup offloading for that shape and use the direct
  setup scan. This is useful while offloading remains an optimization.

Avoid protocol-level panics for missing slots. A benchmark harness may choose
to panic on missing preprocessing, but prover/verifier library code should
return an ordinary `AkitaError`.

### Slice 02C: Recursive Carried-Opening Batching

This lane is the protocol-boundary refactor. The current recursive path is a
singleton carry:

```text
(commitment, opening_point, opening)
```

The target is a list of carried opening claims:

```text
(commitment, point, value, basis, natural_len, padded_len, kind)
```

Use root-style incidence at the recursive boundary. For the first cut, batch
claims in a common padded power-of-two field domain. This avoids heterogeneous
MLE arity in the initial implementation and keeps the verifier shape simple.

The setup-offload integration later appends:

```text
(S_{<=N_setup} commitment, (rho_lambda, rho_y), s_rho, ...)
```

to the usual folded-witness claim.

### Slice 03B/04: Product Sumcheck and Integration

This lane wires the actual delegated setup claim. It should initially depend on
the materialized oracle, then swap to the succinct `omega_S` evaluator and the
prefix commitment opening carry.

The first implementation should use the cleaner post-Stage-2 placement:

1. Run the existing Stage 2 so it still reduces the Stage-1 norm claim and the
   relation claim to one folded-witness opening claim.
2. Let the Stage-2 final row evaluation use
   `m_local(r_x) + sigma_S`.
3. After Stage 2 fixes `r_x`, run a setup product sumcheck proving
   `sigma_S = <S_{<=N_setup}, omega_S(tau_1, alpha, r_x)>`.
4. Locally evaluate `omega_S(rho_lambda, rho_y)`.
5. Carry the remaining setup-prefix opening claim into the next recursive fold.
6. Batch that carried setup claim with the folded-witness opening.

The no-new-stage optimization is to shift the relation matrix work back before
the setup product sumcheck and use Stage 2 for the setup product. That is valid
only if the witness claim produced by the shifted relation work is also closed:
either carry both witness openings into the next recursive incidence batch, or
add an explicit witness-claim reduction that combines the relation witness
claim and the norm witness claim into one later witness opening. Without that
extra reduction/carry, a witness claim is left unresolved.

If there is no subsequent recursive fold, setup offloading is disabled for that
level in the first implementation.

## Why Not a Jolt-Style Splitter

The Jolt refactor-audit stack uses pathspec ownership to regenerate disposable
PR branches from one source branch. That is a good fit when slices are mostly
crate- or directory-shaped.

Akita setup offloading cuts through shared protocol invariants:

- setup seed serialization and descriptor identity;
- setup sizing policy;
- `FlatMatrix` role views;
- prover paths that consume A/B/D setup rows;
- verifier direct-recommit paths that consume A/B setup rows;
- fused setup contribution replay;
- ZK blinding paths, which deliberately use a separate small setup seed/domain
  instead of the base setup matrix;
- recursive proof state and carried opening claims;
- generated schedule/table policy.

Those changes often touch the same files in different semantic phases. A
path-restoration splitter would make invalid intermediate branches too easy.
Use manual semantic branches, with optional helper scripts only for bookkeeping.

## Useful Local Commands

Create the next branch in the stack:

```bash
git switch <parent-branch>
git pull --ff-only
git switch -c <next-branch>
```

Rebase one child after changing its parent:

```bash
git switch <child-branch>
git fetch layerzero main
git rebase <new-parent-branch>
git range-diff <old-parent-branch>..<old-child-branch> <new-parent-branch>..HEAD
git push --force-with-lease
```

Check stack ancestry:

```bash
git log --oneline --decorate --graph main..setup-claim-offloading
```

Focused checks for implementation branches:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
```

Use narrower package tests while developing, but the branch should document any
checks that were skipped before it is made ready for review.

## PR Discipline

- Open each PR as a draft until its dependencies are stable.
- Set PR bases to the narrowest available dependency branch. Parallel lanes may
  target the common foundation branch rather than each other.
- The PR title should describe the project change directly.
- PR bodies should state dependencies, focused scope, non-goals, and checks.
- Do not mix implementation and generated-table churn unless the generated
  files are the point of that branch.
- Do not add backward-compatibility shims for old setup layouts unless a
  reviewer explicitly asks for a temporary comparison harness.

## Current Review Frontier

PR 00 is the spec-only cleanup in `specs/setup-layout-repack.md`.
