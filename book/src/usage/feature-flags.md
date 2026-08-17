# Feature flags

Cargo features on `akita-pcs` and downstream crates.
This repo makes **no backward-compatibility guarantee** for feature combinations;
integrators should pin versions and read release notes.

## Default-on

| Feature | Enables |
|---------|---------|
| `parallel` | Rayon thread pools across `akita-field`, `akita-algebra`, `akita-prover`, `akita-setup`, `akita-sumcheck`, `akita-verifier` |
| `schedules-default` | Dev/CI schedule catalog bundles on `akita-config` |

Disable parallel locally: `cargo build --no-default-features` (or add only the features you need).

## Opt-in

| Feature | Enables |
|---------|---------|
| `disk-persistence` | Disk-backed setup cache paths (`akita-setup/disk-persistence`) |
| `logging-transcript` | `LoggingTranscript` schedule events and wire-before-squeeze smell checks in transcript tests |
| `response-model-diagnostics` | Extra response and source energy measurements for model calibration. This can scan complete witnesses and must not be enabled for performance measurements. |
| `profile-ci` | Compatibility union of schedule features needed by the CI profile benchmark matrix |
| `profile-ci-*` | Narrow schedule and mode groups used by individual CI profile benchmark jobs (see [Profiling](./profiling.md)) |
| `profile-bench-selected` | Internal mode-registry marker enabled by each narrow profile benchmark group; do not enable it alone |
| `schedules-fp128-dense64` | Generated catalog for the bounded dense preset `fp128::Dense64` (committed-source bound 64 inside the 128-bit field). Not in `schedules-default`; see [Bounded committed sources](../how/configuration.md#bounded-committed-sources) |

Per-crate feature tables live in each `crates/*/Cargo.toml`.
Schedule catalog features (`schedules-fp128-onehot`, etc.) are documented in
[Configuration and planning](../how/configuration.md).
