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
| `zk` | Zero-knowledge proving path; pulls `akita-r1cs` into sumcheck and verifier |
| `profile-ci` | Schedule features needed for the CI profile-bench matrix (see [Profiling](./profiling.md)) |

Per-crate feature tables live in each `crates/*/Cargo.toml`.
Schedule catalog features (`schedules-fp128-d64-onehot`, etc.) are documented in
[Configuration and planning](../how/configuration.md).
