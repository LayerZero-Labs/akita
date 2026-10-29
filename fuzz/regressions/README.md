# Regression inputs

Minimal inputs that once made a target fail, replayed by every campaign's
startup baseline: `akita-fuzz-dev seeds` (and therefore `prepare`) copies each
input under `<target>/` into that target's seeds.

| Input | What it checks |
|---|---|
| `public_deserialize/committed-group-noncanonical-coefficient-bound.bin` | A `CommittedGroup` whose stored coefficient bound differs from the resolved key bound is rejected, not normalized. |

Schedule-artifact regressions are a shipped artifact with one field changed.
They are generated with the seeds (`ARTIFACT_REGRESSIONS` in
`devtool/src/main.rs`) instead of stored, because each artifact is hundreds
of KiB:

| Seed | What it checks |
|---|---|
| `regression-terminal-log-basis-zero` | A zero terminal `log_basis` is rejected with an error, not a panic. |
| `regression-oversized-num-digits-fold` | An enormous `num_digits_fold` is rejected promptly instead of iterating the declared digit count. |

Keep an input that still fails under `pending/<target>/` until the fix lands:
as a seed it would fail every campaign's baseline. Replay one with

```bash
cargo run --release -p akita-fuzz-dev -- replay <target> regressions/<target>/<file>
```

or with the instrumented binary:
`AKITA_FUZZ_TARGET=<target> dist/bin/fuzz_all regressions/<target>/<file>`.
