# Folded-only terminal direct ring relations

| Field | Value |
| --- | --- |
| Branch | `quang/terminal-direct-ring-relations` |
| Status | implemented |
| Compatibility | hard protocol and wire cutover |

## Contract

Every supported proof schedule has exactly this topology:

```text
root Fold | one or more suffix Folds | terminal Direct
```

The terminal `Direct` step is a data-free planner marker plus the public
segment-typed witness shape and byte budget. It has no commitment parameters.
Inputs for which the planner cannot produce at least two shrinking folds return
`AkitaError::UnsupportedSchedule`; the prover emits no proof.

Consequently:

- there is no zero-fold or root-direct proof;
- there is no one-fold root-terminal proof;
- every terminal has a predecessor that binds its canonical inner `t` state;
- the terminal relation has no outer `u`, B block, D block, quotient, or
  relation sumcheck.

The runtime and generated schedule validators enforce the topology before any
witness is interpreted. Generated catalogs omit unsupported keys rather than
encoding a fallback row.

## Structural proof representation

`AkitaBatchedProof` represents the topology directly:

```text
root: FoldLevelProof
recursive_folds: Vec<FoldLevelProof>
terminal: TerminalLevelProof
```

The proof stream carries no root-shape or per-step enum. Headerless decoding
derives the exact root, recursive, and terminal shapes from the validated
schedule. A supported proof contains at least the root and one suffix fold.

An intermediate edge uses one of two bindings:

- `OuterCommitment` for an ordinary recursive edge;
- `TerminalInnerState` for the final edge into the terminal.

A terminal fold has no outgoing witness-binding policy. This absence is
represented by `None` in the execution schedule, not by a third enum variant.

## Terminal statement

The predecessor absorbs the terminal witness's canonical `t` bytes under the
next-witness-binding transcript label before sampling dependent challenges.
The terminal rebinds the same bytes as its current public state.

After extension-opening reduction, when present, the terminal verifier checks
the revealed segment-typed witness directly in `F[X] / (X^D + 1)`:

```text
A * w_terminal == t_terminal
witness_trace_eval == trace_eval_target
```

The only terminal relation layout is
`RelationMatrixRowLayout::WithoutCommitmentBlocks`, whose physical rows are:

```text
consistency | A
```

The removed `WithoutDBlock = consistency | A | B` layout was meaningful only
for a root terminal whose external `u` had no predecessor. Since that topology
is unsupported, retaining the layout would create an uninhabited security
state and duplicate row logic.

Extension-opening reduction remains independent. Revealing terminal `z`, `e`,
and `t` does not reveal the pre-reduction polynomial table, so its partials and
sumcheck remain and supply the target consumed by the direct trace check.

## Transcript ordering

The terminal sequence is fixed:

```text
predecessor absorbs canonical terminal t as its outgoing binding
terminal absorbs the same t as current state
terminal absorbs e
terminal samples the sparse challenge and fold challenges
terminal absorbs z
terminal performs the reduced A relation and trace checks
```

The instance descriptor binds the complete folded topology, every expanded
fold `LevelParams`, witness shapes, and total byte budget. Prover and verifier
must reject any schedule/proof disagreement before replay.

## Single sources of truth

- `Schedule::validate_structure` owns runtime topology validation.
- `GeneratedScheduleTableEntry::validate` owns compact-table topology.
- `Schedule::{root_fold, root_fold_mut, num_fold_levels}` owns root/fold access.
- `ExecutionSchedule::relation_matrix_row_layout` selects the terminal layout.
- `LevelParams` row-offset helpers own physical relation ranges.
- the direct terminal checker owns reduced A-relation and trace semantics.
- the existing extension-opening verifier owns EOR semantics.

Callers must use these primitives directly. Do not add root/terminal wrappers,
shape aliases, fallback schedulers, or a second row-layout implementation.

## Security and verifier safety

- Reject malformed topology, witness lengths, shapes, role dimensions, and
  commitment encodings with `AkitaError`; verifier-reachable code must not
  panic.
- The final predecessor-to-terminal `t` binding is load-bearing. Omitting the B
  check is sound only because the accepted public state changes from `u` to the
  predecessor-bound `t`.
- The prover must not pad or enlarge a polynomial merely to manufacture a
  supported schedule. Unsupported degenerate inputs fail during planning.
- Commitment parameters require a nonzero D width because every supported root
  now executes a fold.

## Validation requirements

- generated catalog regeneration and drift agreement, including agreement on
  `UnsupportedSchedule` for omitted keys;
- direct-terminal versus former quotient semantics across supported role
  dimensions;
- transcript tamper tests for terminal `t`, `e`, `z`, the opening target, and
  schedule topology;
- malformed-proof no-panic tests;
- build, formatting, clippy, feature-matrix tests, and documentation guardrails;
- Perfetto-backed `onehot_fp128_d64`, `nv=32` profiling with no verification
  regression relative to the pre-cutover trace.
