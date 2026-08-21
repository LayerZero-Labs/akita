# Feature flags

Cargo features on `akita-pcs` and downstream crates.
This repo makes **no backward-compatibility guarantee** for feature combinations;
integrators should pin versions and read release notes.

## Default-on

| Feature | Enables |
|---------|---------|
| `parallel` | Rayon thread pools across `akita-field`, `akita-algebra`, `akita-prover`, `akita-setup`, `akita-sumcheck`, `akita-verifier` |
| `schedules-default` | The default generated schedule catalog bundles on `akita-config` |
| `transcript-blake2b` | The default Spongefish transcript backend using Blake2b and SHA3 support |

To build without Rayon while keeping the other defaults, use:

```bash
cargo build --no-default-features \
  --features schedules-default,transcript-blake2b
```

Using `--no-default-features` alone also removes the default schedule catalogs
and transcript backend.

## Opt-in

| Feature | Enables |
|---------|---------|
| `transcript-keccak` | The alternative Spongefish Keccak transcript backend. Enable exactly one production transcript backend. |
| `disk-persistence` | Disk-backed setup cache paths (`akita-setup/disk-persistence`) |
| `logging-transcript` | `LoggingTranscript` schedule events and wire-before-squeeze smell checks in transcript tests |
| `response-model-diagnostics` | Extra response and source energy measurements for model calibration. This can scan complete witnesses and must not be enabled for performance measurements. |
| `profile-ci` | Compatibility union of schedule features needed by the CI profile benchmark matrix |
| `profile-ci-*` | Narrow schedule and mode groups used by individual CI profile benchmark jobs (see [Profiling](./profiling.md)) |
| `profile-bench-selected` | Internal mode-registry marker enabled by each narrow profile benchmark group; do not enable it alone |
| `schedules-fp128-dense-bounded` | Generated catalog for the bounded dense preset `fp128::DenseBounded` (committed-source bound 65 signed bits inside the 128-bit field, i.e. every `u64`). Not in `schedules-default`; see [Bounded committed sources](../how/configuration.md#bounded-committed-sources) |

Per-crate feature tables live in each `crates/*/Cargo.toml`.
Schedule catalog features (`schedules-fp128-onehot`, etc.) are documented in
[Configuration and planning](../how/configuration.md).
