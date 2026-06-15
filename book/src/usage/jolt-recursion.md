# Jolt recursion

> **Status:** stub. Part of the initial Akita Book scaffold.

The standalone `profile/akita-recursion/` sub-workspace (excluded from the main
workspace; Rust 1.95 + RISC-V): the artifact → host → guest flow, the
`AkitaJoltInputs` blob, cycle accounting (Jolt guest pins **`fp128::D32OneHot`**
for cycle benchmarks; D32 costs more zkVM cycles than D64 at equal `num_vars`),
the trusted-benchmark vs production-validation distinction, and the nv=32
full-prove trace-length limit. Link the sub-workspace README rather than
duplicating its cycle tables.

## Sources to fold in

- `profile/akita-recursion/README.md` (canonical runbook)
- `profile/akita-recursion/glue/src/lib.rs`
- `specs/akita-crate-followup-jolt-integration.md`
