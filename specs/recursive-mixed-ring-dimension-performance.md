# Recursive Mixed Ring-Dimension Performance

| Field | Value |
|---|---|
| Status | Experiment complete; planner response proposed |
| Date | 2026-07-28 |
| Branch base | `af770e1296` |
| Workloads | PR #331 recursive multi-group and recursive multi-group W8R2 |
| Related design | `specs/mixed-ring-dimension-per-level.md` |

## Purpose

This document records why the first recursive mixed-ring profile is slower
than the production all-D64 schedule and translates those findings into
planner requirements.

It is intentionally separate from the general mixed-D planner design. The
general design explains how dimensions should become planner choices. This
document explains what the recursive experiment taught us about the cost model
and which mistakes the planner must avoid.

## Profiles compared

Both workloads open two precommitted 16-variable singleton groups and two
32-variable final-group polynomials (`nv=32`, `np=4`).

The production control uses D64 for every matrix and level. The experimental
profile uses:

```text
L0 final and precommitted groups: 256/128/128
L1:                               128/64/64
L2+:                              64/64/64
L0 setup-prefix source/commit:    A128/B64
```

The W8R2 workload activates an eight-way distributed witness partition in the
recursive suffix. It does not split the multi-group root: each precommitted
root group has only four live blocks and cannot support eight chunks.

The mixed profile is a synthetic, test-support schedule. It is not selected by
the production planner.

## Measurement summary

Each result is the mean of two retained release runs after one discarded
warmup on an Apple M4 Max, with 16 prove and 16 verify Rayon threads.

| Metric | Plain D64 | Plain mixed | Delta | W8R2 D64 | W8R2 mixed | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Commit | 1.5994 s | 2.0683 s | +29.3% | 1.6094 s | 2.0631 s | +28.2% |
| Prove | 2.1767 s | 3.4511 s | +58.5% | 6.4145 s | 6.7657 s | +5.5% |
| Verify | 14.462 ms | 33.892 ms | +134.3% | 20.538 ms | 41.801 ms | +103.5% |
| Proof bytes | 97,826 | 100,039 | +2.3% | 107,757 | 100,353 | -6.9% |
| Setup vector | 1,720 MiB | 1,024 MiB | -40.5% | 4,128 MiB | 1,032 MiB | -75.0% |
| Prover NTT cache | 4,300 MiB | 7,680 MiB | +78.6% | 10,320 MiB | 7,740 MiB | -25.0% |
| Peak RSS | 8.073 GiB | 11.949 GiB | +48.0% | 18.299 GiB | 12.161 GiB | -33.5% |

The mixed profile is therefore not a recursive latency win in its current
form. W8R2 does obtain a smaller proof and substantially lower setup/RSS
footprints, but verification is still about twice as slow.

## Root causes

### 1. Mixed root projections create more verifier work

The root changes from uniform D64 to `256/128/128`. A is twice the common
relation dimension, so each A-native source ring projects to two D128 relation lanes. The B
and D portions of the relation require corresponding projected subcolumns.

The setup-prefix geometry changes as follows:

| Root setup-prefix geometry | D64 | Mixed |
|---|---:|---:|
| Natural field length | 28,180,480 | 45,088,768 |
| Padded field length | 33,554,432 | 67,108,864 |
| Padded source ring slots | 524,288 | 524,288 |
| Source ring dimension | 64 | 128 |

Although the number of padded source rings is unchanged, the mixed projection
has:

- twice the flat field domain;
- an additional coefficient-axis sumcheck round;
- extra B/D subcolumns induced by the A-to-role projection ratio;
- more setup-index weight evaluation work.

This explains why proof bytes are a poor proxy for verifier work here. The
root Stage-3 proof grows only from 880 to 912 bytes, while the verifier must do
materially more local work to validate it.

The verifier NTT cache is 1.375 MiB in every measured column. The slowdown is
therefore not verifier cache construction; it is the mixed setup-product and
relation geometry.

### 2. The synthetic transition contracts the witness poorly

The most important recursive boundaries are:

| Boundary | D64 | Mixed | Delta |
|---|---:|---:|---:|
| Root output witness | 51,360,128 | 84,773,120 | +65.1% |
| L1 output witness | 13,819,904 | 39,540,864 | +186.1% |

The mixed L1 and early D64 suffix also use 43 outer/opening digits where the
production D64 continuation uses 26. This corresponds to a basis-3
decomposition rather than the control's basis-5 continuation.

Fewer total folds do not compensate for entering the suffix with a witness
almost three times larger and substantially deeper digit streams.

### 3. Larger-ring kernels are not free

The mixed root commits its A matrices at D256 and B matrices at D128 for the
final group and both precommitted groups. These operations use fewer matrix
rows, but each large-ring NTT is more expensive.

For the plain workload, preparing exact D256, D128, and D64 slots increases
the prover NTT cache from 4.3 GiB to 7.68 GiB. Reducing the root A rank from
five to two does not recover the cost of those larger-ring operations.

W8R2 already requires a much larger D64 prepared setup. Its mixed profile
therefore saves cache and RSS, which is why memory and latency do not move in
the same direction.

### 4. The control and experiment were selected differently

The all-D64 schedule is a production catalog entry selected as one complete
recursive schedule.

The mixed profile is assembled in stages:

1. plan a D256 multi-group root;
2. rebuild its B and D matrices at D128;
3. plan one D128 continuation;
4. rebuild its B and D matrices at D64;
5. attach the recursive setup prefix;
6. plan a D64 suffix from that exact boundary.

Every resulting matrix and boundary is validated, but this is not a global
search. A locally acceptable L1 may force an expensive later suffix. The
synthetic result must not be interpreted as the best schedule for the
requested dimension tuple.

## Correctness defects exposed by the experiment

Uniform D64 had hidden two independent-dimension assumptions:

1. `active_setup_field_len` omitted B/D projection subcolumns. For the mixed root
   it planned exactly half of the required prefix.
2. `commit_setup_prefix` used the prefix source dimension for its B
   commitment. A D128 source with a D64 B matrix serialized rows at twice the
   intended width.

Both canonical paths now use the same projection and independent role
dispatch that Stage 3 enforces. Validation still rejects missing, undersized,
or dimension-mismatched prefix slots.

## What is inherent and what is not

The following costs are inherent to this exact schedule:

- the `256/128/128` root has two A-native lanes over the D128 relation base;
- its setup prefix has a 67,108,864-field padded domain;
- committing root/precommit matrices invokes D256 and D128 kernels;
- crossing three dimensions requires prepared setup support for each one.

The following costs are not inherent to mixed dimensions generally:

- choosing basis 3 and 43 digits at L1/L2;
- producing an 84.8M root witness and 39.5M L1 witness;
- selecting the root, transition, and suffix independently;
- using the specific `256/128/128 → 128/64/64 → 64` tuple;
- optimizing proof bytes without an explicit verifier-work objective.

A planner-native search can improve the latter group, but it cannot assume
that larger D is automatically faster.

## Planner requirements derived from this result

### Preserve existing behavior

Mixed-D planning must be opt-in. When the policy domain is exactly the uniform
setup-generation tuple:

- `PlannerPolicy::ring_dimension` remains the only candidate;
- existing scalar-D comparison policies remain unchanged;
- generated schedule rows and catalog identities remain unchanged;
- recursive D64 setup-offload behavior remains available;
- runtime resolution remains catalog-only.

The current synthetic profile modes must remain available as explicit
benchmark controls after planner-native mixed schedules exist.

### Search complete continuations

The planner must not select a root or L1 solely on its local matrix size. For
each admitted dimension tuple and block/basis choice it must derive the exact
outgoing witness and price the complete continuation.

At minimum, state identity must include:

```text
level
current witness field length
current decomposition basis
incoming setup-prefix geometry
candidate A/B/D dimensions
active chunk policy
```

Candidates with different outgoing witness lengths or bases cannot be merged
merely because they have the same local proof-byte estimate.

### Keep dimension and basis search coupled

The observed 26-to-43 digit regression demonstrates that dimension selection
cannot be followed by an independent greedy basis choice. Every candidate
must include both:

```text
(d_a, d_b, d_d, log_basis, block split)
```

and its score must include the continuation derived from that exact payload.

### Model verifier work explicitly if verifier speed is an objective

The general mixed-D design currently proposes:

```text
(physical setup field elements, proof bytes, deterministic tie-break)
```

That policy can legitimately choose a verifier-slower schedule, because proof
bytes do not price:

- setup-projection evaluation terms;
- native projection subcolumns;
- sumcheck round count;
- equality-window and setup-index weight work;
- per-role ring arithmetic.

If the product objective is verification speed, introduce a deterministic,
platform-independent verifier-work component. A candidate starting point is:

```text
VerifierWork {
    setup_projection_terms,
    setup_sumcheck_rounds,
    relation_range_terms,
    ring_switch_terms_by_dimension,
    opening_terms_by_dimension,
}
```

The exact coefficients must be specified and versioned; wall-clock timings
must not enter generated catalog selection.

Until that model exists, planner output should expose verifier-work components
for analysis but retain the approved setup/proof-byte ordering.

### Retain Pareto alternatives

Setup footprint is a maximum across matrices and levels, not an additive
local cost. The planner must retain alternatives that differ in:

- maximum physical setup footprint;
- exact proof bytes;
- verifier-work components;
- outgoing witness length;
- decomposition basis;
- dimension tuple.

Prematurely dropping a locally larger candidate can lose the globally best
schedule after a later transition.

## Planner implementation status on `feat/planner-per-matrix-d`

The first planner-native cut is an opt-in offline scalar search. The canonical
`find_schedule` entry point reads the catalog-bound dimension domain directly
from `PlannerPolicy`.

Implemented:

- `PlannerPolicy::ring_dimension_candidates` carries strictly sorted, unique
  `(d_a, d_b, d_d)` tuples. Policy validation checks native role-projection
  geometry and requires
  every role dimension to divide `PlannerPolicy::ring_dimension`, the setup
  generation dimension.
- `find_schedule` searches that policy-bound domain. An exact uniform
  setup-generation singleton retains the historical proof-payload objective.
- Root and recursive candidates derive A/B/D SIS keys at their selected role
  dimensions. B and D physical widths include `d_a / d_role` projection
  subcolumns; candidates are built directly rather than retargeted afterward.
- The mixed search enumerates every admitted tuple and valid block split only
  at L0 and L1. Tuples are component-wise non-increasing, L2 through the
  terminal are uniform D64. Rank-one dimension pruning remains disabled until
  an equivalence key is proved against the unpruned traversal.
- Mixed-boundary suffix states retain the required `(setup, proof)`
  alternatives per exact first `CommittedGroupParams`, because the parent
  proof formula sees that first step. Once dimensions freeze at L2, candidate
  split derivation reuses the existing uniform-D64 planner path.
- Setup scoring uses exact physical base-field elements and converts once to
  ring elements at the setup-generation dimension. A canonical
  `akita-types` schedule helper now exposes the physical envelope.
- Recursive setup planning is rejected by the mixed entry point. Grouped,
  setup-offloaded, and existing multi-chunk catalog generation continue
  through the unchanged singleton-D planner.
- Regenerating all shipped schedule tables after these changes produces no
  generated-file diff.

Still pending:

- a catalog-bound mixed-D policy and selection-policy identity;
- generated-row expansion/replay for planner-selected mixed dimensions;
- setup/config separation of generation D from admitted candidate domains;
- planner-native mixed multi-group roots and recursive setup prefixes;
- a versioned verifier-work objective, if latency rather than setup footprint
  is the product metric;
- retirement of synthetic profile builders after an adaptive catalog fully
  replaces their coverage.

Therefore this cut makes direct scalar mixed-D schedules searchable and
testable, but it is not yet a shipped adaptive runtime family and does not
claim recursive mixed-D planner completion.

## Required experiments

Before declaring a recursive mixed-D default:

1. Build a synthetic all-D64 control through the same test-support path and
   benchmark it in the same binary as mixed candidates.
2. Collect verifier spans by level and stage, especially:
   - root setup-plan preparation;
   - setup-index weight evaluation;
   - Stage-3 sumcheck replay;
   - relation-range verification;
   - ring-switch verification.
3. Hold `d_a/d_b/d_d` fixed and vary L1 basis to compare 26- and 43-digit
   continuations.
4. Compare at least:
   - `128/128/128 → 128/64/64 → 64`;
   - `256/128/128 → 128/64/64 → 64`;
   - `256/128/128 → 128/128/64 → 64`;
   - all D64.
5. Run both plain recursive and W8R2 workloads.
6. Record setup, commit, prove, verify, proof bytes, fold/tail split, exact
   boundary lengths, setup projection terms, NTT cache, and peak RSS.

## Acceptance criteria for planner-native recursive mixed D

Planner-native recursive mixed-D support is ready only when:

1. Scalar-D policies reproduce their previous schedules exactly.
2. Dimension search is enabled only by an explicit policy/catalog identity.
3. The planner derives matrices directly at their selected dimensions; it
   does not post-retarget a uniform schedule.
4. Root and recursive choices price their complete exact continuation.
5. Setup-prefix source, A, B, and consuming dimensions are independently
   validated and included in schedule identity.
6. An exhaustive small-domain oracle agrees with optimized search.
7. Generated rows replay to the same dimensions, boundaries, proof bytes, and
   setup envelope as offline planning.
8. Plain recursive and W8R2 mixed schedules prove and verify.
9. Existing D64 recursive catalogs and benchmark modes remain usable.
10. Benchmark reporting presents verifier work separately from proof bytes.

## Current conclusion

The first recursive mixed-D tuple saves setup memory in important cases but is
not a latency improvement. Its verifier regression is primarily explained by
the larger mixed root setup projection; its prover regression is amplified by
poor early witness contraction and 43-digit decompositions.

The correct response is not to disable mixed dimensions. It is to make
dimension, basis, split, setup-prefix, and continuation selection one planner
problem while retaining the current D64 policy as the default and control.
