# Binary-source catalog fixtures

These native catalogs are experimental inputs outside production artifacts.
The example checks identity, native geometry and suffix admission on every load.
The freshness gate independently plans every installed case and requires byte equality:

```bash
scripts/binary-source-full-size.sh --field p64 --check-catalogs --output-dir /tmp/p64-fresh
```

Set an absolute `CARGO_TARGET_DIR` and use a fresh output directory. The gate disables
catalog caching, refuses existing case directories and never installs files. A mismatch
prints the catalog diff and fails, retaining both inputs for review. It also rejects an
unexpected installed roster. Admission and replay alone do not establish freshness.

For independent proof reproduction, run `scripts/binary-source-full-size.sh --field p64 --all`
in two fresh output directories, with `--reference-dir` pointing to the first on the second
run. Require verification and equality of catalog, commitment, verifier setup, claims and
proof bytes before installing reviewed JSON. Run the freshness gate after installation.
The CI replay and tamper controls run in both protocol modes. These checks do not certify
production security, and no historical proof digest baseline is required.
