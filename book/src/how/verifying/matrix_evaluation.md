# Matrix evaluation at a point

The verifier evaluates the relation matrix multilinear extension without
materializing the matrix. Its column geometry comes from the same
`WitnessLayout` that emitted the witness.

Every nonterminal fold proves that one flat witness satisfies a public system
of ring relations. The verifier sees the public setup, the proof, and one
random evaluation of that witness. This chapter explains the rows, columns,
and compact evaluation used to check that system at one point.

## The semantic relations

For one commitment group, four ordinary row families explain the source
relations:

| Family | Purpose | Main witness segment |
|---|---|---|
| consistency | Tie the folded response to the opening digits | `z_hat`, `e_hat` |
| A or inner | Check the inner commitment image | `z_hat`, `t_hat` |
| B or outer | Check the outer commitment image | `t_hat` |
| D or opening | Check the opening commitment image | `e_hat` |

Each commitment group has its own consistency, A, and B rows. The consuming
level owns one D family after it combines all group opening digits. A group may
have its own opening point, polynomial count, live block count, and A and B
ring dimensions. The D ring dimension belongs to the consuming level.

These four families are the best way to understand the mathematics. They are
not always the complete production row list.

## Compressed production rows

Production schedules normally use compressed commitment payloads. In that
mode, the ordinary B and D targets are private intermediate values. Two
compression maps reduce each group B image through F rows and reduce the
level-wide D image through H rows. Only the terminal F and H payloads are
public.

The physical row order is:

```text
for each group in authenticated relation order:
    consistency
    A rows
    B rows
level-wide D rows
for each compression map:
    one F row per group
    one level-wide H row
```

In compressed mode, the public right hand side is zero for the ordinary B and
D rows and for every nonterminal compression row. The terminal F and H rows
contain the fixed public payloads. Quotient-lift relation mode gives every
physical row a quotient witness at that row's native ring dimension. Reduced
evaluation keeps the same rows and right hand sides but omits every ordinary
and compression quotient range.

`RelationRhsLayout::row_families` is the row-order authority. The code does not
maintain a second verifier-specific row list.

## Canonical walk

`WitnessLayout` is the only owner of witness coefficient ranges. It orders
chunk and group units as follows:

```text
chunk 0:
    group in relation order: [Z | E | T]
chunk 1:
    group in relation order: [Z | E | T]
...
optional shared ordinary quotient rows
compression digits, alignment ranges, and optional compression quotient rows
```

For a group with `B_g` exact live blocks and `C` chunks, chunk `c` owns

```math
\left[
\left\lfloor\frac{cB_g}{C}\right\rfloor,
\left\lfloor\frac{(c+1)B_g}{C}\right\rfloor
\right).
```

The ranges cover the exact live prefix. They are not padded to equal length.
If there are more chunks than blocks, some ranges are empty.

The three ordinary segments have different ownership rules:

- `Z` is replicated once per chunk. An empty chunk keeps its `Z` range, and
  the honest prover writes zero there.
- `E` is partitioned by exact block ownership. It contains the opening digits
  for claims and blocks owned by that unit.
- `T` is partitioned by the same block ownership. It contains the inner
  commitment images used by the B relation.

The ordinary quotient rows, when present, and the compression suffix are
shared once after all chunk and group units. In reduced-evaluation mode the
compression suffix contains F/H digits and alignment only.

Each unit carries an exact global block range. Relation, setup, and trace
evaluators consume these checked ranges. They do not reconstruct offsets from
a second chunk layout description. Multi-group and multi-chunk layouts are the
ordinary product of the same two indices.

## Exact block weights

For a group with exact live block count `F`, the fold challenge supplies `F`
independent sparse coefficients. The exact count is transcript bound and is
validated before any indexing. Sparse challenge values use the ring add,
subtract, and double fast paths where applicable.

## What the verifier evaluates

Let `tau1` select a row and let `x` select a flat witness coefficient. The
verifier needs the multilinear extension

```math
\widetilde M(\tau_1,x)
=
\sum_{i,j}
eq(\tau_1,i)eq(x,j)M_{i,j}.
```

The ring switch challenge `alpha` evaluates each native ring row. The schedule
selects one of two coefficient functionals. Quotient lifting factors the common
low coefficient coordinates from `x` and applies powers of `alpha`. Reduced
evaluation contracts the exact physical equality window against a terminal
residue kernel, which already includes those coefficient coordinates.

In quotient-lift mode, the final result has three ordinary parts:

```math
\widetilde M
=
\widetilde M_{\mathrm{structured}}
+
\widetilde M_{\mathrm{setup}}
+
\widetilde M_{\mathrm{quotient}}.
```

The compressed F and H contribution is prepared separately and added by the
Stage 2 verifier. This keeps the ordinary A, B, and D setup geometry unchanged.

In reduced-evaluation mode there is no quotient term and no outer common-alpha
factor:

```math
\widetilde M
=
\widetilde M_{\mathrm{structured,reduced}}
+
\widetilde M_{\mathrm{setup,reduced}}.
```

The separately prepared compressed F/H contribution uses the same reduced
functional and is added once.

### Structured witness terms

The structured term covers the non-setup coefficients of the consistency, A,
B, and D rows. Its inputs include:

- sparse fold challenges evaluated against the mode-selected coefficient
  functional;
- opening point weights for source positions and live blocks;
- gadget weights for the A, B, and D decompositions;
- exact group, claim, chunk, and block ranges from `WitnessLayout`; and
- row weights from the `tau1` equality polynomial.

The evaluator stores compact affine descriptions of these axes. It does not
materialize a matrix row or a dense witness-sized weight vector.

### Setup term

The setup term covers `A * Z`, `B * T`, and `D * E`. One
`SetupContributionPlan` owns the setup address geometry for all three roles.
It supports two ways to obtain the same value:

- Direct mode scans the required public setup prefix during Stage 2.
- Deferred mode uses a claimed setup value in Stage 2 and checks that value in
  Stage 3.

The mode changes where the setup inner product is checked. It does not change
the relation polynomial.

Reduced evaluation is admitted only with direct setup contribution. The plan
prepares one terminal residue functional for each distinct checked native
coefficient window, then reuses the existing fused A/B/D traversal with those
weights. It does not create one kernel per setup lane or scan the three roles
separately.

### Quotient term

In quotient-lift mode, each physical row has quotient digits for division by
`X^D + 1` at its native ring dimension. The verifier evaluates those explicit
digits and multiplies by the row weight and the evaluated denominator.
Compression quotient rows are handled by the separate compression evaluator
and are not counted twice. Reduced evaluation has no quotient term; the signed
wrap contribution is already present in the terminal residue functional.

## Setup roles and mixed rings

The A, B, and D setup contributions use the same group and chunk ranges. D
group offsets follow checked relation-group prefix sums.
`SetupProjectionGeometry` owns mixed-ring projection, so verifier evaluation
does not maintain a parallel setup-column layout.

The active
[`role-native-projected-digit-layout`](../../../../specs/role-native-projected-digit-layout.md)
spec defines the E and T verifier cutover. Its target physical order is:

```text
[semantic value][role subcolumn][role digit][native coefficient]
```

The setup matrix and relation witness use the same subcolumn and digit axes.
In quotient-lift mode, a role subcolumn `s` of dimension `r` has weight
`alpha^(s * r)`. In reduced mode, the checked physical window selects the
corresponding entries of the terminal residue functional. Both modes apply the
role gadget power on the digit axis.

When the projection ratio is one, the verifier does not allocate projection
powers or multiply by one. It uses the unprojected contiguous equality window
directly. Mixed groups use exact coefficient ranges and the shared minimum
relation block. There are no carrier spans or per-role padding to scan.

The A, B, and D roles may use different ring dimensions. The verifier chooses
their greatest common coefficient block as the low coefficient boundary. A
role of dimension `d_R` then has `d_R / d_0` relation lanes.

`RelationAddressGeometry` owns this split. `SetupProjectionGeometry` owns the
matching setup projection. The verifier never pads a smaller role to a larger
carrier ring. It applies the appropriate `alpha` power to each native role lane
and uses the same flat witness address for direct and deferred setup checks.

## Preparation and evaluation

Ring-switch preparation validates all public geometry before it creates a
`RelationMatrixEvaluator`. The prepared object keeps only succinct state:

- sparse challenges evaluated at `alpha`;
- source-position opening evaluations;
- expanded row equality weights;
- checked relation address geometry; and
- the shared `WitnessLayout` and group metadata needed to build setup tensors.

At the final Stage 2 point, `evaluate_relation_at_point` dispatches once on the
authenticated relation mode. The quotient branch obtains the direct or
deferred setup term, evaluates the quotient tail, and applies the common low
coefficient factor. The reduced branch rejects deferred setup, evaluates the
structured and fused direct-setup terms with terminal residue kernels, omits
the quotient tail, and returns the already complete flat MLE.

The verifier never materializes dense relation weights. Dense verifier tables
exist only as test oracles. The prover's reduced mode does use one ephemeral
dense Stage-2 weight table; it is not a verifier allocation or proof field.

## Safety contract

Before evaluation, the verifier checks the opening dimensions, group-local
layout, unit ranges, setup geometry, and work bounds. Malformed proof data
returns `AkitaError`. Verifier-reachable evaluation does not panic or allocate
from an unchecked proof-controlled dimension.

All row counts, native dimensions, address products, and allocation sizes are
also checked before use.

Direct setup evaluation is linear in the public setup prefix because those
coefficients are arbitrary and must be read. Structured terms are linear in
their explicit challenges and any live quotient digits, with logarithmic
equality contractions over repeated affine address axes. Reduced preparation
uses `O(d)` auxiliary extension-field state for native ring dimension `d` and
the same single setup scan. The verifier never scales with the prover's
materialized relation table.

## Code map

- Row families and public right hand sides:
  `crates/akita-types/src/proof/relation.rs` and
  `crates/akita-types/src/proof/relation_layout.rs`.
- Witness ranges: `crates/akita-types/src/witness.rs`.
- Relation address geometry:
  `crates/akita-types/src/proof/relation_address.rs`.
- Residue and terminal coefficient functionals:
  `crates/akita-algebra/src/ring/residue.rs` and
  `crates/akita-types/src/proof/coefficient_functional.rs`.
- Verifier preparation:
  `crates/akita-verifier/src/protocol/ring_switch.rs`.
- Final point evaluation:
  `crates/akita-verifier/src/protocol/ring_switch/relation_evaluation.rs`.
- Setup contribution:
  `crates/akita-types/src/setup_contribution/`.

The next chapter shows how Stage 2 combines this relation value with the range
image and evaluation trace.
