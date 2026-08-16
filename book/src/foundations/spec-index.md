# Spec index

> **Status:** stub. Part of the initial Akita Book scaffold.

A status-tagged index into `specs/`, so readers can find the design record behind
a chapter and tell active design from historical record. Each entry: spec, one
line, status (`active` / `implemented` / `superseded` / `archived`), and the book
chapter it feeds. Keep this in sync with `specs/PRUNING.md` and the archive index.

The active design frontier (keep as live specs):
`flat-public-matrix-and-exact-ntt-cache`, `role-native-projected-digit-layout`,
`setup-layout-repack`,
`setup-offloading-planner`, `eor-streamed-prover`, `packed-sumcheck`,
`planner-incidence-generalization`,
`akita-field-refactor`, `akita-compute-backend-metal` (Metal tail),
`large-digit-ntt-infrastructure`.

The approved SIS security-policy frontier is
`sis-quantum128-scalar-n-table`: a scalar, role-driven table using one ADPS16
quantum LGSA policy at a 128-bit target.

Recent archived records include
[`subring-coefficient-packing`](../../../specs/archive/2026-Q3/subring-coefficient-packing.md),
whose opening geometry, transcript order, planner policy, and security contract
now live in the ring-switch, configuration, profiling, and security chapters;
[`commitment-compression-cutover`](../../../specs/archive/2026-Q3/commitment-compression-cutover.md),
and
[`relation-range-image-sumcheck`](../../../specs/archive/2026-Q3/relation-range-image-sumcheck.md).
Their durable compression and Stage 2 descriptions now live in the Akita fold
realization and sumcheck stages chapters. Other recent archived records include
[`group-local-opening-points`](../../../specs/archive/2026-Q3/group-local-opening-points.md),
whose durable claim ownership and protocol dataflow now live in the architecture,
verification, commitment API, and extension-opening chapters. The
[`PR 375 prover optimization record`](../../../specs/archive/2026-Q3/pr375-prover-streaming-and-onehot-unification.md)
now lives in the same archive. Its durable source ownership, CPU resource, and
NTT lifecycle rules live in the optimization and commitment API chapters. The
[`profile-bench-coverage-matrix`](../../../specs/archive/2026-Q3/profile-bench-coverage-matrix.md),
whose current benchmark contract now lives in the profiling chapter, is also
archived.

## Sources to fold in

- `specs/PRUNING.md` (process + classification), `specs/archive/README.md`
- Council specs-audit report (full classification table)
