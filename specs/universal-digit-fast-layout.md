# Spec: Universal Digit-Fast Witness Layout

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao; Cursor assistant (GPT-5.6 Sol) |
| Created       | 2026-07-10 |
| Status        | active |
| PR            | #294 |
| Supersedes    | Layout decisions in `setup-layout-repack.md`, `protocol-core-eor-consolidation.md`, and `distributed-verifier-row-eval.md` |
| Superseded-by | |
| Book-chapter  | |

This spec follows the lifecycle in [`PRUNING.md`](PRUNING.md).
It records the target contract for PR #294 while implementation and review are still in progress.
It supersedes only the root versus recursive layout decisions in the listed specs.
It does not supersede their unrelated setup offload, extension opening, or distributed verifier decisions.

## Summary

Akita must use one digit-fast physical witness layout at every fold level.
The old recursive layout transposed each witness so that blocks were the fastest axis.
That transpose conflicts with distributed proving because each machine must own a contiguous range of complete logical blocks.
The new layout stores every block contiguously and keeps decomposition digits inside their semantic column.
The multilinear opening domain uses a derived virtual stride so that non-power-of-two physical block lengths still admit an exact position and block factorization.
The virtual stride moves only structural zero positions inside the already padded opening domain.
It does not add witness columns, setup columns, commitment bytes, proof bytes, or distributed payload.

## Intent

### Goal

Replace the root digit-fast and recursive block-fast split with one checked layout contract shared by the planner, prover, verifier, setup contribution code, trace code, terminal code, and distributed proving code.

The primary layout authorities are:

* `OpeningBatchWitnessLayout` for semantic groups, machine ownership units, physical segment ranges, and exact `e`, `t`, `z`, and `r` indices.
* `OpeningBlockLayout` for the mapping between compact physical storage and the zero-padded multilinear opening domain.
* `SetupProjectionGeometry` for mixed setup ring dimensions, Stage 3 footprint sizing, round counts, and verifier work accounting.

These types must own the relevant validation.
Callers must not rebuild their formulas.

### Invariants

#### One physical order

Root and recursive witnesses use the same physical axis order.
There is no `BlockOrder` enum.
There is no column-major recursive mode.
There is no compatibility path for the removed layout.

#### Contiguous distributed ownership

A machine ownership unit contains one contiguous `[z | e | t]` range.
The machine owns a contiguous interval of complete logical blocks.
The shared `r` tail follows all ownership units.
The physical layout does not transpose a machine's blocks.

#### Physical and opening addresses are distinct

Physical storage uses compact addresses.
The multilinear opening domain uses a power-of-two position stride.
Every prover and verifier path must use the correct address space.
No path may apply equality weights directly to a compact physical address unless both address spaces coincide.

#### Exact prover and verifier agreement

The prover and verifier must derive identical:

* semantic group order;
* relation processing order;
* ownership unit order;
* physical segment ranges;
* virtual opening addresses;
* opening point split;
* setup role projection;
* Stage 3 round count;
* verifier work bound.

Any disagreement must produce an `AkitaError`.
It must not select another evaluator or layout.

#### Compact physical payloads

Structural opening zeros are not physical witness values.
They do not appear in:

* Ajtai commitment inputs;
* setup A, B, or D columns;
* `z`, `e`, `t`, or `r` wire segments;
* terminal witness bytes;
* proof-size accounting;
* distributed communication.

#### Small integer `z`

The block axis is compressed into `z` before the relation opens `z`.
The compressed values remain small integer combinations protected by the fold infinity-norm contract.
Opening field weights must not be inserted into `z`.

#### Transcript and wire rules

The proof object shape and transcript label sequence do not change solely because of this layout cutover.
The physical coefficient order does change.
Proofs, commitments, cached witnesses, and setup-derived artifacts made under the old order are not compatible.
Akita makes no backward compatibility guarantee for these bytes.

#### Verifier safety

Verifier-reachable layout code must:

* use checked arithmetic;
* reject malformed geometry before allocation;
* enforce work and allocation caps;
* reject invalid opening point lengths;
* avoid panic, unchecked indexing, and unbounded materialization.

#### One source of truth

Every physical or virtual index comes from the canonical layout objects.
Setup weights, relation columns, trace weights, terminal emission, and verifier replay must not keep independent copies of the same formulas.
Small pass-through helpers and aliases are not acceptable.

### Non-Goals

This spec does not add a second layout mode.
This spec does not preserve old proof or commitment bytes.
This spec does not add physical padding between blocks.
This spec does not make `z` block-indexed.
This spec does not apply field opening weights before the fold norm check.
This spec does not redesign setup prefix commitments or recursive setup offload.
This spec does not enable the product of multiple semantic groups and multiple machine chunks unless the proof protocol gains explicit support for that product.
This spec does not change the security estimator or SIS parameter tables except where schedule geometry must be regenerated after the layout cutover.

## Evaluation

### Acceptance Criteria

- [ ] `specs/universal-digit-fast-layout.md` is the normative design record for PR #294.
- [ ] `BlockOrder` and every recursive column-major path are removed.
- [ ] Root and recursive physical witness emission use the same digit-fast formulas.
- [ ] `OpeningBatchWitnessLayout` fields cannot be constructed or mutated without validation.
- [ ] Semantic groups and machine ownership chunks use distinct typed identifiers.
- [ ] `OpeningBatchWitnessLayout` is the only authority for physical `e`, `t`, `z`, and `r` indices.
- [ ] `OpeningBlockLayout` is the only authority for physical to opening address conversion.
- [ ] Compact physical storage and virtual opening tables agree for every valid schedule.
- [ ] The prover and verifier require an opening position table of exactly `P` entries.
- [ ] Non-power-of-two live block lengths pass direct opening, recursive fold, relation, trace, setup, and terminal parity tests.
- [ ] The compressed `z` relation uses shared position weights and retains its infinity-norm contract.
- [ ] The setup A, B, and D role views use digit-fast semantic columns at every level.
- [ ] Mixed role dimensions use one checked common base in both prover and verifier Stage 3.
- [ ] Stage 3 uses one source of truth for required footprint, ring bits, rounds, prefix sizing, alpha powers, and work accounting.
- [ ] The exact A-role verifier work product is checked before evaluation.
- [ ] Direct and structured setup weight evaluations match an independent dense oracle.
- [ ] Multi-group root proofs pass with transcript order distinct from relation processing order.
- [ ] Two-unit and eight-unit distributed layouts pass cross-layer parity tests.
- [ ] Malformed group order, chunk geometry, opening length, role dimensions, and work bounds return `AkitaError`.
- [ ] Transcript tamper and wrong-point tests still reject.
- [ ] Proof serialization shape does not grow solely because of virtual opening zeros.
- [ ] Planner proof-size estimates match runtime witness lengths.
- [ ] Generated schedule tables show no unexplained drift.
- [ ] `cargo fmt -q` passes.
- [ ] `cargo clippy --all --message-format=short -q -- -D warnings` passes.
- [ ] `cargo test` passes.
- [ ] `./scripts/check-doc-guardrails.sh` passes.
- [ ] The profiling smoke tests record any setup, prover, verifier, memory, or proof-size change.
- [ ] A final thermo-nuclear code-quality review reports no blocking duplication, wrapper slop, dead layout path, or oversized mixed-responsibility module.

### Testing Strategy

#### Layout unit tests

`OpeningBlockLayout` tests must cover:

* `B` equal to 1, 2, and 8;
* `L` equal to 1;
* power-of-two `L`;
* non-power-of-two `L`, including 3 and 1184;
* overflow;
* zero block length;
* non-power-of-two block count;
* out-of-range physical and opening coordinates.

For every valid case, the tests must check:

\[
\operatorname{physical}(b,p)=bL+p
\]

\[
\operatorname{opening}(b,p)=bP+p
\]

\[
P=\operatorname{nextPowerOfTwo}(L)
\]

\[
\operatorname{nextPowerOfTwo}(BL)=BP
\]

The tests must show a concrete non-power-of-two case where:

\[
\chi_\rho(bL+p)
\ne
\chi_{\rho_{\mathrm{position}}}(p)
\chi_{\rho_{\mathrm{block}}}(b)
\]

The same test must show:

\[
\chi_\rho(bP+p)
=
\chi_{\rho_{\mathrm{position}}}(p)
\chi_{\rho_{\mathrm{block}}}(b)
\]

#### Cross-layer physical tests

One independent oracle must compare:

* physical witness emission;
* `OpeningBatchWitnessLayout` indices;
* dense relation matrix columns;
* setup weight columns;
* trace columns;
* terminal witness bytes;
* verifier relation replay.

The oracle must derive formulas directly from the spec.
It must not call the production index methods to compute its expected values.

The cases must include:

* all dimensions greater than one;
* one semantic group;
* multiple semantic groups;
* two ownership units;
* eight ownership units;
* non-power-of-two live position counts;
* replicated `z`;
* a shared `r` tail.

#### Recursive opening tests

The recursive witness test must compare:

\[
\sum_{b=0}^{B-1}
\sum_{p=0}^{L-1}
w[bL+p]a[p]c[b]
\]

against the prover's blockwise fold and the verifier's opening claim.

The dense virtual table must place each live value at `bP+p`.
Every structural position from `bP+L` through `(b+1)P-1` must be zero.

#### Setup contribution tests

Direct setup weight materialization and structured evaluation must match for:

* D;
* B;
* A;
* uniform role dimensions;
* nested mixed role dimensions;
* nonzero D column offsets;
* non-power-of-two live lengths;
* one ownership unit;
* two ownership units;
* eight ownership units;
* replicated `z`;
* multiple semantic groups where supported.

The A-role work bound test must check the exact product at the cap and one term above the cap.
The test must not allocate the over-cap domain.

#### End-to-end tests

The following proof paths must pass:

* root singleton;
* root multi-group;
* recursive singleton;
* recursive setup contribution;
* recursive mixed role dimensions;
* terminal fold;
* extension opening reduction;
* transcript logging;
* tampered proof rejection;
* wrong opening point rejection.

The mixed role test must prove and verify with `SetupContributionMode::Recursive`.
It must assert that at least one Stage 3 setup sumcheck is present.

#### CI and feature coverage

Run:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
rtk cargo nextest run --profile ci --no-default-features --features parallel,disk-persistence
./scripts/check-doc-guardrails.sh
```

Run the transcript tests with `logging-transcript`.
Run the portable x86 verifier check.
Run schedule table drift checks.
Run fuzz target compilation.

### Performance

The virtual opening domain must have the same cardinality as the old globally padded domain.

For power-of-two `B`:

\[
BP=\operatorname{nextPowerOfTwo}(BL)
\]

This identity means the layout cutover must not increase:

* opening challenge count;
* Stage 2 witness table domain;
* Stage 3 witness table domain;
* proof object shape;
* serialized proof length;
* physical witness length;
* Ajtai setup width;
* distributed witness payload.

The dense table implementation may move structural zeros from one global suffix into per-block gaps.
It must not allocate a larger table.

The setup index evaluator must enforce `MAX_COMPACT_STRIDE_TERMS`.
The guard must count the exact loop product.
It must not use an additive approximation for nested loops.

Before merge, profile:

```bash
cargo bench -p akita-pcs --bench setup_index_weight
cargo run -p akita-pcs --release --example profile --features profile-ci
```

Record setup time, prover time, verifier time, peak memory, witness ring elements, and proof bytes.
Any regression above 10 percent in a shipped profile needs an explanation and a follow-up decision.

## Design

### Architecture

The layout has two related but distinct descriptions.

```text
Semantic witness layout

group
  ownership unit 0: [ z_0 | e_0 | t_0 ]
  ownership unit 1: [ z_1 | e_1 | t_1 ]
  ...
shared tail:         [ r ]
```

```text
Opening domain

physical block 0: [ live L values ]
opening block 0:  [ live L values | structural zeros to P ]

physical block 1: [ live L values ]
opening block 1:  [ live L values | structural zeros to P ]
```

`OpeningBatchWitnessLayout` owns the semantic witness layout.
`OpeningBlockLayout` owns the second mapping.
Neither type replaces the other.

The first type answers what a physical column means.
The second type answers where that physical column lives in the multilinear opening domain.

### Terminology

#### Semantic group

A semantic group corresponds to one commitment group in an opening batch.
It has its own claim count, logical relation block count, live position count, digit depths, and setup E offset.

#### Machine chunk

A machine chunk is a distributed ownership coordinate.
It is not a commitment group.
It owns a contiguous interval of logical blocks.

#### Ownership unit

An ownership unit is one physical `[z | e | t]` stride for one semantic group and one machine chunk.

#### Relation order

Relation order determines physical witness emission and relation processing.
For grouped roots, the final group may appear before precommitted groups.

#### Transcript order

Transcript order determines the public commitment and claim order absorbed by Fiat-Shamir.
It can differ from relation order.
The layout stores both orders explicitly.

#### Physical block

A physical block is one compact interval of `L` live ring elements used by the recursive fold and distributed prover.

#### Opening block

An opening block is the corresponding interval of `P` positions in the multilinear domain.
Only its first `L` positions are live.

### Physical witness order

For one ownership unit, physical segments appear as:

```text
[ z | e | t ]
```

All ownership units appear in relation order.
The shared `r` tail appears once after the final unit.

Let:

* `c` be a claim index;
* `q` be a local block index inside an ownership unit;
* `Q` be the number of blocks in that unit;
* `a` be an A output row;
* `p` be a live position;
* `d_o` be an opening digit;
* `d_c` be a commitment digit;
* `d_f` be a fold digit;
* `d_r` be a quotient digit;
* `delta_o` be the opening decomposition depth;
* `delta_c` be the commitment decomposition depth;
* `delta_f` be the fold decomposition depth;
* `delta_r` be the quotient decomposition depth;
* `n_A` be the number of A rows.

#### e segment

The semantic axes are:

```text
(claim, block, opening digit)
```

The opening digit is the innermost axis.

\[
j_e(c,q,d_o)
=
o_e+d_o+\delta_o(q+Qc)
\]

#### t segment

The semantic axes are:

```text
(claim, block, A row, opening digit)
```

The opening digit is the innermost axis.

\[
j_t(c,q,a,d_o)
=
o_t+d_o+\delta_o(a+n_A(q+Qc))
\]

#### z segment

The semantic axes are:

```text
(position, commitment digit, fold digit)
```

The fold digit is the innermost axis.

\[
j_z(p,d_c,d_f)
=
o_z+d_f+\delta_f(d_c+\delta_c p)
\]

Every ownership unit for a semantic group contains a complete copy of that group's `z` segment.
The block axis is absent because the fold compressed it before decomposition.

#### r segment

The semantic axes are:

```text
(relation row, quotient digit)
```

The quotient digit is the innermost axis.

\[
j_r(u,d_r)
=
o_r+d_r+\delta_r u
\]

The `r` segment is shared across all groups and ownership units.

### Physical and opening address contract

Let:

* `B` be the number of opening blocks;
* `L` be the live physical block length;
* `P` be `next_power_of_two(L)`;
* `b` be an opening block index;
* `p` be a live position.

Physical storage uses:

\[
\operatorname{physical}(b,p)=bL+p
\]

The multilinear opening domain uses:

\[
\operatorname{opening}(b,p)=bP+p
\]

The positions:

\[
bP+L,\ldots,(b+1)P-1
\]

are structural zeros.

The layout constructor requires:

\[
B>0
\]

\[
B\text{ is a power of two}
\]

\[
L>0
\]

\[
\operatorname{nextPowerOfTwo}(BL)=BP
\]

### Why the old recursive layout did not have the blocker

The old recursive physical order used:

\[
\operatorname{old}(p,b)=pB+b
\]

Because `B` is a power of two, the binary block bits and position bits occupy disjoint ranges.
The equality polynomial factors exactly:

\[
\chi_\rho(pB+b)
=
\chi_{\rho_{\mathrm{block}}}(b)
\chi_{\rho_{\mathrm{position}}}(p)
\]

The old transpose existed partly to preserve this factorization.
It also made each position's blocks contiguous rather than each block's positions contiguous.
That is the wrong ownership property for distributed proving.

The first compact row-major attempt used:

\[
\operatorname{physical}(b,p)=bL+p
\]

When `L` is not a power of two, carries mix the block and position bits.
The equality polynomial does not factor:

\[
\chi_\rho(bL+p)
\ne
\chi_{\rho_{\mathrm{block}}}(b)
\chi_{\rho_{\mathrm{position}}}(p)
\]

The `z` witness has already removed `b`.
No later evaluator can recover a block-dependent opening weight.

The virtual stride restores the factorization without restoring the transpose:

\[
\chi_\rho(bP+p)
=
\chi_{\rho_{\mathrm{block}}}(b)
\chi_{\rho_{\mathrm{position}}}(p)
\]

### Opening point order

The opening point uses position variables first and block variables second.

For little-endian multilinear indices:

```text
[ position bits | block bits ]
```

The position basis table has length `P`.
The block basis table has length `B`.
Physical folds consume only the first `L` position weights.

The prover and verifier must require exact lengths.
Accepting a position table of length `L` when `L < P` is invalid.

### z compression and the fold norm

Let `w[b,p]` be a small integer witness value before block compression.
Let `c_b` be the small fold coefficient for block `b`.

The compressed value is:

\[
z[p]=\sum_b c_b w[b,p]
\]

The relation opens:

\[
\sum_p a[p]z[p]
\]

Under the virtual opening map:

\[
\sum_p a[p]\sum_b c_b w[b,p]
=
\sum_{b,p} c_b a[p]w[b,p]
\]

The block factor is already part of `c_b`.
The remaining position factor is shared across blocks.
This is why the position-only `z` wire is sufficient.

Applying `eq(rho,bL+p)` would require a different position weight for each block.
That cannot be represented after `z` has removed the block axis.

The implementation must never solve this by multiplying field opening weights into `z`.
That would make `z` field-valued and invalidate the small integer fold bound.

### Setup role ordering

The shared setup object supplies overlapping A, B, and D prefix views.
The role matrices are not stored as disjoint concatenated matrices.

Let:

* `u` be a setup row;
* `B_g` be a semantic group's full logical block count;
* `b` be a global logical block;
* `c` be a claim;
* `e_g` be the group's D-role column offset;
* `W_A`, `W_B`, and `W_D` be physical role row widths.

#### D columns

The D semantic axes are:

```text
(claim, block, opening digit)
```

\[
K_D(u,c,b,d_o)
=
W_Du+e_g+\delta_o(B_gc+b)+d_o
\]

#### B columns

The B semantic axes are:

```text
(claim, block, A row, opening digit)
```

\[
K_B(u,c,b,a,d_o)
=
W_Bu+n_A\delta_o(B_gc+b)+\delta_o a+d_o
\]

#### A columns

The A semantic axes are:

```text
(position, commitment digit)
```

\[
K_A(u,p,d_c)
=
W_Au+\delta_c p+d_c
\]

A has no commit gadget multiplier in the setup contribution.
Its witness-side `z` term uses the fold gadget.

The setup role views use live physical widths.
They do not include the virtual positions from `L` through `P-1`.

### Mixed setup ring dimensions

Let the role ring dimensions be:

\[
d_A,\quad d_B,\quad d_D
\]

Stage 3 uses:

\[
d_0=\min(d_A,d_B,d_D)
\]

Every role dimension must be a power-of-two multiple of `d_0`.

The projection ratio for role `R` is:

\[
R_R=d_R/d_0
\]

`SetupProjectionGeometry` owns:

* `d_0`;
* each role ratio;
* each projected footprint;
* the maximum required setup index;
* setup index domain length;
* ring bits;
* Stage 3 round count;
* alpha power length;
* natural field length;
* verifier evaluation work.

The prover and verifier must consume the same geometry object.
Neither side may derive rounds or required length independently from `d_A`.

For one role with native footprint `F_R`, its projected base-ring width is:

\[
F_RR_R
\]

The setup index footprint is:

\[
\operatorname{required}
=
\max(F_AR_A,F_BR_B,F_DR_D)
\]

The Stage 3 round count is:

\[
\log_2(d_0)
+
\log_2(\operatorname{nextPowerOfTwo}(\operatorname{required}))
\]

### Exact verifier work accounting

Verifier work checks must match the actual nested loops.

For the A role, the term count includes:

* live z columns;
* A rows;
* projection lanes;
* ownership units;
* fold digits.

The checked product is:

\[
\operatorname{AWork}
=
z_{\mathrm{cols}}\cdot n_A\cdot R_A\cdot U\cdot\delta_f
\]

The budget guard must reject work above `MAX_COMPACT_STRIDE_TERMS`.
It must run before evaluation or allocation.

### Multi-group and machine chunk rules

Semantic groups and machine chunks are separate axes.
The layout must not use one integer to represent both.

For grouped roots:

* transcript order follows public commitment and claim order;
* relation order follows the relation's processing order;
* every group has its own semantic dimensions;
* D-role E offsets follow relation order;
* physical ownership units follow relation order.

For distributed chunks:

* every chunk owns the same number of logical blocks;
* chunk block windows are disjoint;
* `e` and `t` use local block indices inside the chunk;
* `z` is replicated in every chunk;
* `r` is shared once.

The current proof protocol rejects multiple semantic groups combined with multiple machine chunks.
This rejection is a protocol boundary, not a reason to merge the two axes in the type system.

### Structured evaluation

All equality evaluations use opening addresses.
They do not use physical addresses directly.

The implementation may use:

* direct per-index equality evaluation;
* an exact compact stride evaluator;
* an exact sparse pair carry evaluator;
* tensor factors that are consumed without materialization.

Every implementation must match the same dense oracle.
There is no correctness fallback between alternate layout meanings.

An optimized evaluator must:

* use additions and multiplications only;
* support Boolean challenges without inversion;
* handle non-power-of-two live lengths exactly;
* check arithmetic overflow;
* enforce a verifier work cap;
* avoid a Cartesian state allocation.

If a new compact evaluator has no production caller, it must be removed rather than kept as test-only future machinery.

### Trace evaluation

Trace construction and evaluation use the same physical and opening mappings as the relation matrix.
Trace code must not maintain a second block-fast formula.

The extension trace path may retain its field-specific algebra.
Its column addresses still come from the canonical opening map.

### Terminal witness

Terminal emission uses the same physical `e`, `t`, `z`, and `r` indices as intermediate folds.
The terminal wire does not include virtual zeros.
Terminal proof-size accounting uses physical lengths.

### Planner and proof-size effects

The planner stores the live block length `L`.
It derives the opening position stride `P`.
It must not replace `L` with `P` in:

* witness-size formulas;
* setup A width;
* terminal payload size;
* distributed communication size;
* commitment matvec dimensions.

The planner uses `P` only for:

* opening point arity;
* multilinear table domain;
* recursive sumcheck round counts that depend on opening bits.

Generated schedule tables must be regenerated if any recorded round count or witness length changes.
Any drift must be explained by this contract.

### Serialization and transcript

The layout cutover removes a public enum and changes coefficient order.
It is a breaking change.

No compatibility tag or version branch is added.
The prover and verifier use the schedule and proof shape from the same code version.

The transcript keeps the same labels and event order unless another spec explicitly changes them.
The absorbed values can change because witness coefficients and derived claims change.

Tests must compare event labels and counts.
They must not require old digest bytes.

### Validation boundaries

The canonical constructors validate:

* nonempty semantic groups;
* unique and complete group orders;
* nonzero digit depths;
* power-of-two opening block count;
* nonzero live block length;
* divisibility of logical blocks across machine chunks;
* supported group and chunk products;
* checked physical ranges;
* checked virtual ranges;
* witness capacity;
* nested setup role dimensions;
* exact opening point lengths;
* verifier work bounds.

Resolved layout fields must be private.
Tests must not construct invalid resolved layouts with struct literals.

### Current implementation status and blockers

> **PR #294 is not merge-ready.**

The core physical and virtual layout cutover is present.
`BlockOrder` has been removed.
The compact physical and virtual opening maps are implemented.
The core virtual-stride tests passed before the latest mixed-role Stage 3 edits.

The current blockers are:

1. The mixed-role recursive Stage 3 test fails during root relation preparation.
2. The mixed-role test proves with `SetupContributionMode::Direct` while claiming to exercise recursive Stage 3.
3. CI formatting is not clean.
4. CI clippy rejects dead code and a test helper with too many arguments.
5. Documentation CI rejects the dead `active_setup_ring_slots` function.
6. Test, fuzz, portable verifier, and schedule drift jobs are red.
7. Resolved witness layout fields remain public.
8. Setup contribution carriers still duplicate layout geometry.
9. Legacy test-only structured evaluators remain.
10. Compact equality evaluators have no production caller.
11. Several files exceed the maintainability target or mix unrelated responsibilities.
12. Live book, docs, and specs still describe `BlockOrder`, block-fast recursive witnesses, and the removed `WitnessLayout`.
13. Performance benchmarks have not been run.

No acceptance criterion may be marked complete until these blockers are resolved and the full verification list passes.

### Alternatives Considered

#### Keep the old recursive transpose

This preserves a clean binary factorization.
It breaks the distributed ownership goal because positions, rather than complete blocks, are contiguous.
It also preserves two physical layout meanings.
This option is rejected.

#### Use compact physical addresses as MLE addresses

This uses `bL+p` for both storage and opening.
It is correct only when `L` is a power of two.
It is incompatible with the position-only `z` wire for general schedules.
This option is rejected.

#### Physically pad every block to P

This restores a rectangular tensor.
It enlarges physical setup and witness paths unless every consumer treats the zeros implicitly.
The virtual map obtains the same opening semantics without physical columns.
This option is rejected.

#### Keep z indexed by block

This would support arbitrary block-dependent opening weights.
It multiplies the `z` wire and relation width by the block count.
It changes proof size and distributed communication.
This option is rejected.

#### Apply opening field weights before z decomposition

This removes the need for a block axis after compression.
It makes `z` field-valued and breaks the fold infinity-norm contract.
This option is rejected.

#### Add a compact fast path and retain the old path

This creates two protocol meanings and a fallback boundary.
It is incompatible with the repository's full-cutover policy.
This option is rejected.

## Documentation

The implementation PR must update:

* `book/src/how/recursion.md`;
* `book/src/how/verifying/matrix_evaluation.md`;
* `book/src/how/verifying/distributed-relation-verifier.md`;
* `docs/block-order.md`;
* `specs/setup-layout-repack.md`;
* `specs/protocol-core-eor-consolidation.md`;
* `specs/distributed-verifier-row-eval.md`;
* `specs/multi-group-batching.md` where it names old layout types;
* `AGENTS.md` only if the verifier contract summary needs a new pointer.

`docs/block-order.md` should be deleted or replaced with a short pointer to the book and this spec.
Live specs that remain otherwise useful should state which layout sections this spec supersedes.
They should not be marked wholly superseded when they still own unrelated design work.

The durable explanation should be folded into:

* `book/src/how/recursion.md`;
* `book/src/how/verifying/matrix_evaluation.md`;
* `book/src/how/verifying/distributed-relation-verifier.md`.

After the book owns the stable contract, this spec should move to the appropriate quarterly archive.

## Execution

### Slice 1: Restore a green baseline

- [ ] Run `cargo fmt -q`.
- [ ] Remove `active_setup_ring_slots`.
- [ ] Fix clippy findings without adding allow attributes for avoidable design problems.
- [ ] Fix the mixed-role test mode.
- [ ] Diagnose and fix the root relation preparation failure.
- [ ] Run the mixed-role recursive Stage 3 test.
- [ ] Run affected crate checks.
- [ ] Commit and push the repair.

### Slice 2: Enforce canonical layout ownership

- [ ] Make resolved layout fields private.
- [ ] Add narrow accessors and typed iterators that expose real concepts.
- [ ] Remove direct struct construction from tests.
- [ ] Replace duplicated setup contribution geometry with a shared layout reference and group identity.
- [ ] Remove repeated E offset and segment traversal formulas.
- [ ] Run cross-layer oracle tests.
- [ ] Commit and push the repair.

### Slice 3: Remove dead paths and split modules

- [ ] Delete legacy segment-length APIs used only by tests.
- [ ] Delete test-only structured evaluator implementations after replacing their useful tests.
- [ ] Either use the compact equality evaluators in production or delete them.
- [ ] Remove forwarding re-exports.
- [ ] Split witness layout from chunk profile policy.
- [ ] Split compact equality algorithms from offset interval helpers.
- [ ] Split setup contribution tests by contract.
- [ ] Move witness emitters out of `tail_segments.rs`.
- [ ] Commit and push the cleanup.

### Slice 4: Complete documentation

- [ ] Update book chapters.
- [ ] Remove stale `BlockOrder` prose.
- [ ] Mark old spec sections as superseded by this spec.
- [ ] Run documentation guardrails.
- [ ] Commit and push the documentation.

### Slice 5: Final verification

- [ ] Run full tests.
- [ ] Run strict clippy.
- [ ] Run nextest CI profile.
- [ ] Run portable verifier checks.
- [ ] Run schedule drift checks.
- [ ] Build fuzz targets.
- [ ] Run profiling smoke tests.
- [ ] Record proof-size and performance results in the PR.
- [ ] Run a final correctness review.
- [ ] Run a final thermo-nuclear code-quality review.
- [ ] Update this spec's status and acceptance criteria only after every blocking check passes.

## References

* [`specs/TEMPLATE.md`](TEMPLATE.md)
* [`specs/SPEC_REVIEW.md`](SPEC_REVIEW.md)
* [`specs/PRUNING.md`](PRUNING.md)
* [`docs/documentation.md`](../docs/documentation.md)
* [`specs/setup-layout-repack.md`](setup-layout-repack.md)
* [`specs/protocol-core-eor-consolidation.md`](protocol-core-eor-consolidation.md)
* [`specs/distributed-verifier-row-eval.md`](distributed-verifier-row-eval.md)
* [`specs/multi-group-batching.md`](multi-group-batching.md)
* [`book/src/how/recursion.md`](../book/src/how/recursion.md)
* [`book/src/how/verifying/matrix_evaluation.md`](../book/src/how/verifying/matrix_evaluation.md)
* [`book/src/how/verifying/distributed-relation-verifier.md`](../book/src/how/verifying/distributed-relation-verifier.md)
* `crates/akita-types/src/witness.rs`
* `crates/akita-types/src/proof/witness_layout_contract.rs`
* `crates/akita-types/src/setup_contribution/geometry.rs`
* `crates/akita-prover/src/protocol/ring_switch/evals.rs`
* `crates/akita-pcs/tests/recursive_setup_e2e.rs`

Authorship disclosure: Drafted by Cursor assistant (model: GPT-5.6 Sol) on behalf of Quang Dao with approval.
