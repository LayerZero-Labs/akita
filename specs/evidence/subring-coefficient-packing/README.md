# Catalog revision evidence

These immutable TSV files compare the complete generated schedule catalog at
the reviewed PR base `5a4f72bce3920ecb753751187cd2eaab3f915b8b` with the
coefficient packing branch.

- `base.tsv` is the 68-row selective L2 baseline snapshot. It was produced at
  the exact base commit with a reporting-only backport of the snapshot schema
  and the base revision's canonical proof, setup, EOR, digest, and
  first-direct-capacity functions.
- `head.tsv` is the 71-row merged snapshot. It removes the unsupported fp128
  dense nv44 stress row and adds three bounded-dense scalar rows plus one
  grouped row with a bounded-dense precommit.
- `comparison.tsv` is the complete logical-key union. It reports exact lookup
  and schedule digests, first-direct padded capacity, total setup fields, proof
  bytes, fold counts, successor witness lengths, per-level EOR bytes, opening
  methods, packing geometry, and A security routes. The base and head snapshots
  both include the padded first-direct capacity reconstructed by their own exact
  materialized schedules. Missing values are comparison drift, not wildcards.

Snapshots normally use family plus final and precommitted polynomial layouts as
the cross-revision logical row key. When two current rows share those layouts,
the non-family producer contract is appended to disambiguate them. This keeps
the legacy one-hot row matched to the base while recording the new
`balanced(bound=65)` precommit row as an addition. Exact lookup-key digests
remain separate columns because this PR intentionally changes the
commitment-profile version.

Generate a snapshot at the revision being measured with:

```text
scripts/generate-schedule-tables.sh \
  --catalog-snapshot path/to/snapshot.tsv
```

Compare a baseline snapshot while regenerating the current revision with:

```text
scripts/generate-schedule-tables.sh \
  --catalog-baseline path/to/base.tsv \
  --catalog-report path/to/comparison.tsv
```

The comparison reports removed baseline rows alongside additions and changes.
This repository permits intentional catalog-breaking changes, so revision
policy is reviewed from the checked evidence. Same-head generated-table drift
remains an automatic failure.

## Resolved fp32 nv20 planner objective

The two fp32 dense nv20 rows have a sharp choice between setup size and proof
size. Adaptive direct schedules now use the same objective as recursive
schedules: first direct setup capacity, then proof bytes, total setup, and the
canonical descriptor. No amortized or weighted objective was added.

| Row | First-direct capacity | Setup fields | Proof bytes | Fold levels |
| --- | ---: | ---: | ---: | ---: |
| No precommit | 131,072 | 458,752 | 64,896 | 6 |
| One precommit | 262,144 | 655,360 | 66,912 | 6 |

The previous total-setup-first policy selected three-level schedules with
2,549,124 and 3,222,244 byte terminal-heavy proofs. Those rows were not proof
accounting errors. They were the result of treating total setup as the primary
coordinate. The unified objective compares the padded capacity needed at the
first direct edge instead. In both rows the six-level and former three-level
choices tie on that primary coordinate, so proof bytes decide the comparison.

The total setup fields increase because later folds add matrices. That is a
secondary coordinate by design. A host that cannot materialize the selected
setup can set the existing explicit setup field budget; the planner does not
guess an expected proof count.
