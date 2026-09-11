# Catalog revision evidence

These immutable TSV files compare the complete generated schedule catalog at
the reviewed PR base `5a4f72bce3920ecb753751187cd2eaab3f915b8b` with the
coefficient packing branch.

- `base.tsv` is the 68-row selective L2 baseline snapshot. It was produced at
  the exact base commit with a reporting-only backport of the snapshot schema
  and the base revision's canonical proof, setup, EOR, digest, and
  first-direct-capacity functions.
- `head.tsv` is the 83-row merged snapshot. It removes the unsupported fp128
  dense nv44 stress row; adds three bounded-dense rows and one grouped row with
  a bounded-dense precommit; and includes the additional grouped nv34 and
  fp32/fp64 scalar coverage rows.
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

## Current fp32 nv20 adaptive objective

The two fp32 dense nv20 rows have a sharp choice between setup size and proof
size. Adaptive direct schedules retain the first-direct-first V2 objective:
first-direct setup capacity, proof bytes, total setup, root output-witness
length, then the canonical descriptor. Recursive schedules use a separate
power-of-two-bucketed setup objective. No amortized or weighted objective is
used.

| Row | First-direct capacity | Setup fields | Proof bytes | Fold levels |
| --- | ---: | ---: | ---: | ---: |
| No precommit | 131,072 | 458,752 | 62,447 | 6 |
| One precommit | 262,144 | 524,288 | 63,254 | 6 |

The first-direct-primary objective retains the smaller first-direct capacity,
then selects proof bytes before exact total setup. A host can apply the existing
explicit setup field budget as an admission limit; the planner does not guess
an expected proof count or convert setup and proof into a weighted score.
