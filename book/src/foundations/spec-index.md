# Spec index

> **Status:** stub. Part of the initial Akita Book scaffold.

A status-tagged index into `specs/`, so readers can find the design record behind
a chapter and tell active design from historical record. Each entry: spec, one
line, status (`active` / `implemented` / `superseded` / `archived`), and the book
chapter it feeds. Keep this in sync with `specs/PRUNING.md` and the archive index.

The active design frontier (keep as live specs): `setup-layout-repack`, `eor-streamed-prover`, `packed-sumcheck`,
`planner-incidence-generalization`, `single-point-opening-batch`, `akita-field-refactor`,
`akita-compute-backend-metal` (Metal tail), `crt-ntt-prime-profiles`,
`transcript-immediate-fixes`.

The approved SIS security-policy frontier is
`sis-quantum128-scalar-n-table`: a scalar, role-driven table using one ADPS16
quantum LGSA policy at a 128-bit target.

## Sources to fold in

- `specs/PRUNING.md` (process + classification), `specs/archive/README.md`
- Council specs-audit report (full classification table)
