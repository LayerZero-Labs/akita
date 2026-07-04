# 128-bit L-infinity SIS table reference

This directory preserves the temporary 128-bit coefficient-`L∞` Rust split table
used by PR #255 for benchmark comparison. It is a reference artifact only.

Production lookup is wired to the checked-in 138-bit table under
`crates/akita-types/src/sis/generated_sis_table/`. The files here are not
compiled by any crate and must not be imported by runtime code.

Source commit:

```text
6397365d bench(sis): wire 128-bit linf tables
```

Generation command:

```bash
cargo run -p akita-sis-estimator --release --features parallel \
  --example infinity_width_table -- \
  --format rust-split --target-bits 128 --profile local-minimum --progress-every 500
```
