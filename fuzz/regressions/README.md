# Regression inputs

Minimal inputs for findings in `../FINDINGS.md`, named `<finding>-<what>.bin`
under the target that found them. They are not seeds: an unfixed finding would
fail the campaign's startup baseline on every run. Replay one with

```bash
cargo run --release -p akita-fuzz-dev -- replay <target> regressions/<target>/<file>
```

or with the instrumented binary:
`AKITA_FUZZ_TARGET=<target> dist/bin/fuzz_all regressions/<target>/<file>`.
After a fix, move the input into the target's seeds so it stays covered.
