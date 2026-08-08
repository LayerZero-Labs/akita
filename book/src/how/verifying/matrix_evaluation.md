# Relation matrix and witness layout

Every nonterminal fold proves that one flat witness satisfies a public system
of ring relations. The prover handles the full witness. The verifier sees only
the public setup, the proof, and one random evaluation of that witness. It must
therefore evaluate the relation matrix at one random row and column point
without constructing the matrix.

This chapter explains the matrix, its rows, its columns, and the compact
evaluation used by the verifier. The next chapters explain how Stage 2 combines
this value with the range and opening checks.

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
contain the fixed public payloads. The quotient witness still covers every
physical row at that row's native ring dimension.

`RelationRhsLayout::row_families` is the row-order authority. The code does not
maintain a second verifier-specific row list.

## The flat witness

`WitnessLayout` is the only owner of witness coefficient ranges. The outer
physical order is chunk first, then group:

```text
chunk 0:
    group in relation order: [Z | E | T]
chunk 1:
    group in relation order: [Z | E | T]
...
shared ordinary quotient rows
compression digits, alignment ranges, and compression quotient rows
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

The ordinary quotient rows and the compression suffix are shared once after
all chunk and group units.

## What the verifier evaluates

Let `tau1` select a row and let `x` select a flat witness coefficient. The
verifier needs the multilinear extension

```math
\widetilde M(\tau_1,x)
=
\sum_{i,j}
eq(\tau_1,i)eq(x,j)M_{i,j}.
```

The ring switch challenge `alpha` evaluates each native ring row. The verifier
factors the common low coefficient coordinates from `x`, applies the powers of
`alpha` for those coordinates, and evaluates the remaining relation lane
address with a bounded equality window.

The final result has three ordinary parts:

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

### Structured witness terms

The structured term covers the non-setup coefficients of the consistency, A,
B, and D rows. Its inputs include:

- sparse fold challenges evaluated at powers of `alpha`;
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

### Quotient term

Each physical row has quotient digits for division by `X^D + 1` at its native
ring dimension. The verifier evaluates those explicit digits and multiplies by
the row weight and the evaluated denominator. Compression quotient rows are
handled by the separate compression evaluator and are not counted twice.

## Mixed ring dimensions

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

At the final Stage 2 point, `eval_flat_at_point` prepares the common relation
point, evaluates the structured terms, obtains the direct or deferred setup
term, evaluates the quotient tail, and applies the common low coefficient
factor.

The dense matrix and dense relation weights exist only as test oracles.

## Safety and cost

All row counts, native dimensions, unit ranges, address products, and work
bounds are checked before allocation or indexing. Malformed proof or setup data
returns `AkitaError`.

Direct setup evaluation is linear in the public setup prefix because those
coefficients are arbitrary and must be read. Structured terms are linear in
their explicit challenges and quotient digits, with logarithmic equality
contractions over repeated affine address axes. The verifier never scales with
the prover's materialized relation table.

## Code map

- Row families and public right hand sides:
  `crates/akita-types/src/proof/relation.rs` and
  `crates/akita-types/src/proof/relation_layout.rs`.
- Witness ranges: `crates/akita-types/src/witness.rs`.
- Relation address geometry:
  `crates/akita-types/src/proof/relation_address.rs`.
- Verifier preparation:
  `crates/akita-verifier/src/protocol/ring_switch.rs`.
- Final point evaluation:
  `crates/akita-verifier/src/protocol/ring_switch/relation_evaluation.rs`.
- Setup contribution:
  `crates/akita-types/src/setup_contribution/`.

The next chapter shows how Stage 2 combines this relation value with the range
image and evaluation trace.
