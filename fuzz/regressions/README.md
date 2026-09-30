# Regression inputs

Minimal inputs for findings in `../FINDINGS.md`, named `<finding>-<what>.bin`
under the target that found them. Inputs under `<target>/` are for fixed
findings: `akita-fuzz-dev seeds` (and therefore `prepare`) copies them into that
target's seeds, so the startup baseline replays them on every run. Keep inputs
for unfixed findings under `pending/<target>/` instead; as seeds they would fail
the baseline on every run. Replay one with

```bash
cargo run --release -p akita-fuzz-dev -- replay <target> regressions/<target>/<file>
```

or with the instrumented binary:
`AKITA_FUZZ_TARGET=<target> dist/bin/fuzz_all regressions/<target>/<file>`.
After a fix, move the input from `pending/<target>/` to `<target>/`.
