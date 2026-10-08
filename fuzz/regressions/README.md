# Regression inputs

Minimal inputs that once made a target fail, named after what they exercise
and stored under the target that replays them. `akita-fuzz-dev seeds` (and
therefore `prepare`) copies every input under `<target>/` into that target's
seeds, so each campaign's startup baseline replays them before fuzzing.

| Input | What it checks |
|---|---|
| `public_deserialize/committed-group-noncanonical-coefficient-bound.bin` | A `CommittedGroup` whose stored coefficient bound differs from the resolved key bound is rejected, not normalized. |
| `schedule_artifact/terminal-log-basis-zero.bin` | A schedule artifact with a zero terminal `log_basis` is rejected with an error, not a panic. |
| `schedule_artifact/oversized-num-digits-fold.bin` | Admitting an artifact with an enormous `num_digits_fold` returns promptly instead of iterating the declared digit count. |

Keep an input that still fails under `pending/<target>/` until the fix lands:
as a seed it would fail every campaign's baseline. Replay one with

```bash
cargo run --release -p akita-fuzz-dev -- replay <target> regressions/<target>/<file>
```

or with the instrumented binary:
`AKITA_FUZZ_TARGET=<target> dist/bin/fuzz_all regressions/<target>/<file>`.
