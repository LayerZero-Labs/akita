# CI test timing

Design spec: [`specs/ci-test-timing.md`](../specs/ci-test-timing.md).

Every PR gets an upserted timing comment (marker `<!-- akita-ci-test-timing -->`) showing per-pass wall time vs a main baseline, critical-path wall time when passes run in parallel, and per-test outliers from the nextest JUnit output.

## How CI runs tests

- **Non-zk** and **all-features** nextest passes run in parallel matrix jobs (`slice:index/total` via `matrix.shard` / `strategy.job-total`; 1-based index, not `strategy.job-index`).
- [Swatinem/rust-cache](https://github.com/Swatinem/rust-cache) is enabled per pass (`cache: false` on `setup-rust-toolchain` so the explicit shared-key step owns `target/`).
- The `test-timing` job merges shard JUnit and uploads artifact `ci-test-timing-data` containing `summary.json` and the rendered comment/report.

## Local repro

```bash
python3 scripts/doc_blast_radius.py --base origin/main --head HEAD   # blast-radius only
```

For timing artifact layout and renderer trust boundary, see the spec linked above.
