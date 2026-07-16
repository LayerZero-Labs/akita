# Troubleshooting

> **Status:** stub. Part of the initial Akita Book scaffold.

Common failure modes and fixes (Jolt-book parallel): the `--release` profile
guard (`AKITA_ALLOW_DEBUG_PROFILE`), eq-table OOM at high `num_vars`, setup-cache
invalidation, `RAYON_NUM_THREADS`, and recursion-guest panics
(`JOLT_BACKTRACE=full`).

## Sources to fold in

- `crates/akita-pcs/examples/profile/main.rs` (guards, knobs)
- `profile/akita-recursion/README.md` (guest debugging)
- Council usage report (gaps & concerns)
