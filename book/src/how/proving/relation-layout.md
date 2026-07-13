# Relation layout

`RelationLayout` is the canonical, checked description of a ring relation. It
is compiled from authenticated level parameters, opening layout, and (when
present) compression description. The resulting layout is derived protocol
state; it is not itself serialized or independently authenticated. It answers
two different questions:

1. Which logical coefficient vectors exist, and where are they addressed?
2. Which matrix-row families consume those vectors, in what ring, and with
   which right-hand side and quotient?

The distinction matters. A paper diagram can write one vector `w` and one
matrix `M`, but the implementation must simultaneously support several opening
groups, an optional `D` block, different native ring dimensions, physical
witness chunking, and commitment compression.

## The two logical axes

The coefficient axis is a flat, contiguous arena. The drawing is schematic:
segment lengths are parameter-dependent.

```text
                    group-major body                         base quotient tail
  ┌───────────────────────────────┬────────────────────────┬──────────────────┐
  │ current                       │ precommitted[0] ...     │ q(family) ...    │
  │ Z_current | E_current | T_current | Z_0 | E_0 | T_0 ... │                  │
  └───────────────────────────────┴────────────────────────┴──────────────────┘
                                                                       │
                              optional compression extension           ▼
                              ┌─────────────────────┬──────────────────────┐
                              │ all Xi inputs       │ all Xi quotients     │
                              │ Xi(layer,source)... │ qXi(layer,source)... │
                              └─────────────────────┴──────────────────────┘
```

Every box is a `RelationSegment` with a stable `RelationSegmentId` and a
half-open `CoeffSpan`. Code should resolve stable IDs through `RelationLayout`,
never reconstruct offsets from parameter formulas.

Compression uses layer-major order. Within each layer, sources occur as
current, precommitted groups in increasing index order, then opening; sources
not configured at that layer are omitted. All `Xi` inputs are allocated before
all compression quotients, rather than interleaving `Xi` and `qXi` per map.

A `BaseQuotient { row: family_id }` names one coefficient segment for an entire
base family. For a family with `m` native rows, quotient decomposition depth
`L`, and native ring dimension `d`, that segment contains `m × L × d` field
coefficients: one `L × d` quotient representation per native row.

The row axis is a schedule of typed `RelationRowFamily` values:

```text
 row 0                                                                        N
  │                                                                            │
  ▼                                                                            ▼
  ┌─────────────┬────────┬────────┬─────────┬─────────┬─────┬──────────────┐───────┬─────┐
  │ consistency │ A(cur) │ B(cur) │ A(pre0) │ B(pre0) │ D ? │ compression* │ trace │ pad │
  └─────────────┴────────┴────────┴─────────┴─────────┴─────┴──────────────┘───────┴─────┘
  └──────────────── quotient-bearing matrix families ────────────────────┘   ▲
                                                                               │
                                                    field-level, not a family ─┘
```

`RelationRowPlan::families()` contains only the live matrix families. Each
family records its row span, stable ID, input segments, right-hand side,
quotient segment, and native ring dimension. `trace_row()` is the next row, but
is deliberately outside the family list and has no ring-switch quotient. The
trace row is included before the domain is padded to the checked power-of-two
`padded_row_count()`.

The exact live-row order is: consistency; `A` then `B` for every group in
current-first order; optional `D`; compression families in the same
layer-major, current/precommitted/opening order described above; then the trace
row and padding. Absent compression sources contribute no row.

## Family meanings

| Family | Logical inputs | Ordinary right-hand side | Notes |
|---|---|---|---|
| consistency | every group's `Z` and `E` | zero | one row |
| `A(group)` | that group's `Z` | zero | one family per group |
| `B(group)` | that group's `T` | group commitment, or zero when augmented | compression moves its RHS payload into the chain |
| `D` | every group's `E` | opening, or zero when augmented | optional; absent from the terminal D-free base layout |
| compression | a compact `Xi`, optionally its successor | zero except for the chain's last family, which carries `TerminalPayload` | one family per compression map |

The code and paper do not use identical row-family names. The code's
`Consistency` family is the paper's fold-evaluation row involving `Z/E`. The
code's `A(group)` family is the paper's fold-consistency (or `A`) family,
associated algebraically with `Z/T`. These are terminology differences, not
extra equations. Use the typed `RelationRowInputs` edges and the matrix builder
as the authority when tracing an implementation dependency.

Groups use **current-first order**: `Current`, then `Precommitted { index: 0 }`,
`Precommitted { index: 1 }`, and so on. This stable protocol order may differ
from a paper's source-vector order. It is normative for the derived relation,
witness execution, and proof interpretation; it is not a claim about a
standalone serialization order.

Each family has its own `native_ring_dim`. There is no general “relation ring
dimension.” For example, an `A` family follows its `A` key, `B` follows its `B`
key, and `D` follows its `D` key; compression maps may use smaller rings. The
current fused matrix/quotient kernel still requires one uniform dimension and
explicitly rejects both mixed-dimension families and compression rows. That is
an execution limitation, not a limitation of `RelationLayout`; per-family
execution is the intended consumer of the richer plan.

## Logical addressing versus physical witness chunks

`RelationSegment` and `WitnessLayout` describe different coordinate systems.
For one group split into two chunks, `Z` is repeated while `E/T` are
partitioned:

```text
 single-group logical base                  physical two-chunk witness
 [ Z | E | T | base-q ]       ───────────▶  [ Z | E/2 | T/2 ] [ Z | E/2 | T/2 | r ]

 Xi inputs and compression quotients: outside WitnessLayout
```

For multiple groups, configured multi-chunk mode is rejected. Instead, each
group produces one full chunk in current-first order:

```text
 multi-group logical base                       physical group chunks
 [ Zcur | Ecur | Tcur | Zpre0 | Epre0 | Tpre0   [ Zcur  | Ecur  | Tcur ]
   | base-q ]                       ───────────▶  [ Zpre0 | Epre0 | Tpre0 | r ]

 Xi inputs and compression quotients: outside WitnessLayout
```

The logical `BaseQuotient` segments above contain field coefficients. Their
physical projection is the shared base-quotient `r` tail measured in ring
elements. Compression quotients are logical segments only and never enter that
tail.

`WitnessLayout` has exactly two projection modes:

- With one group, configured multiple chunks repeat the complete `Z` segment
  in every chunk, partition `E` and `T` evenly across chunks, and place the
  shared base `r` tail only in the last chunk.
- With multiple groups, configured multi-chunk mode is rejected. The projection
  emits one chunk per group in current-first order; each chunk contains that
  group's complete `Z/E/T`, and only the last group chunk carries the shared
  base `r` tail.

Consequently, physical chunk offsets are not coefficient-arena offsets.
Compression inputs and compression quotients remain compact logical carriers
outside `WitnessLayout`.

## Compression and negative-binary support

Compression augments an existing `B` or `D` family with a gadget input. The
payload that was formerly that family's commitment/opening RHS moves to the
last compression family in the chain. The augmented base family and every
intermediate compression family have zero RHS; only the last family has
`TerminalPayload`. Each map adds a compression row family and a compression
quotient segment; a successor edge records both the next segment and the gadget
basis used to interpret it.

`RelationLayout` includes in negative-binary support exactly those `Xi` input
segments whose authenticated map alphabet is `NegativeBinary`. It excludes all
quotient segments, including the compression quotients. The selected spans are
normalized into sorted disjoint runs and exposed through
`negative_binary_support()`. The digit range-check must augment its ordinary
equality support with precisely these runs. Negative binary is not restricted
to a particular layer: every authenticated negative-binary map contributes its
`Xi` span and therefore to the corresponding security accounting.

## Reading the paper and implementation together

The paper is the best source for the algebraic relation and its security
argument. Its compact `M w = h` presentation intentionally suppresses several
implementation coordinates. When the pictures differ, use these rules:

- the paper's compact witness `[z | e | t | r | u1 | v1 | ...]` is an
  algebraic presentation, not coordinate equality with the implementation;
  the normative logical code order is group-major `Z/E/T`, all base quotient
  segments, all `Xi` inputs, then all compression quotient segments;
- code `Consistency` means the paper's fold-evaluation `Z/E` row, while code
  `A(group)` means the paper's fold-consistency/`A` family associated with
  `Z/T`;
- the code's current-first multi-group order is normative for the derived
  relation, witness execution, and proof interpretation;
- `D` is optional at terminal D-free boundaries;
- ring dimension belongs to each row family, not the whole relation;
- the field-level trace row lies outside quotient-bearing matrix families;
- logical coefficient spans are not physical witness chunks;
- compression rows and negative-binary support extend the relation even though
  the present uniform fused kernel cannot execute them yet.

The canonical definitions live in
`crates/akita-types/src/layout/relation.rs`. Protocol code should consume
`RelationLayout` and `RelationRowPlan` directly rather than introduce another
layout formula or pass-through helper.
