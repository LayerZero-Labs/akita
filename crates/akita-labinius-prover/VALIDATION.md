# Clear binary opening implementation and validation

Base and unchanged HEAD: `3d69096eadb129c097e4fc2e4f2d289e82e0ea1a` (`origin/dev`).
Changes are uncommitted. Nothing was pushed, no branch was switched, and no PR
was opened. No reference implementation was opened or read.

The two new crates implement the complete standalone clear opening behind the
off-by-default `labinius` feature. Existing files contain only workspace,
lockfile, dependency-guard, CI, and crate-index registration edits. No frozen
crate, schedule artifact, or Akita proof-path file changed.

## Files and line counts

Counts are final physical file lines; existing-file edits are also shown.

| File | Lines | Change |
| --- | ---: | --- |
| `.github/workflows/ci.yml` | 541 | +21 |
| `AGENTS.md` | 127 | +2 |
| `Cargo.toml` | 111 | +2 |
| `Cargo.lock` | 2523 | +26 |
| `scripts/check-crate-deps.sh` | 110 | +8 / -1 |
| `docs/crate-graph.md` | 248 | +18 |
| `book/src/how/architecture.md` | 180 | +5 |
| `crates/akita-labinius-verifier/Cargo.toml` | 42 | new |
| `crates/akita-labinius-verifier/src/channel.rs` | 91 | new |
| `crates/akita-labinius-verifier/src/codec.rs` | 139 | new |
| `crates/akita-labinius-verifier/src/commitment.rs` | 46 | new |
| `crates/akita-labinius-verifier/src/endpoint.rs` | 262 | new |
| `crates/akita-labinius-verifier/src/frontend.rs` | 165 | new |
| `crates/akita-labinius-verifier/src/lib.rs` | 143 | new |
| `crates/akita-labinius-verifier/src/profile.rs` | 264 | new |
| `crates/akita-labinius-verifier/src/source.rs` | 147 | new |
| `crates/akita-labinius-prover/Cargo.toml` | 54 | new |
| `crates/akita-labinius-prover/src/lib.rs` | 114 | new |
| `crates/akita-labinius-prover/tests/admission.rs` | 279 | new |
| `crates/akita-labinius-prover/tests/common/mod.rs` | 200 | new |
| `crates/akita-labinius-prover/tests/oracles.rs` | 273 | new |
| `crates/akita-labinius-prover/tests/roundtrips.rs` | 49 | new |
| `crates/akita-labinius-prover/tests/tamper.rs` | 326 | new |
| `specs/labinius-clear-opening.md` | 190 | new |
| `crates/akita-labinius-prover/VALIDATION.md` | 187 | new validation report |

## Public API

The verifier crate exports `BinaryClearSetup<F,D,M>`,
`BinaryClearCommitment<F,D,M>` (public column-major `images`),
`BinaryEvaluationClaim { point, value }`, and `ClearChannel`.

- `BinaryClearSetup::new(matrix, n_a, m, columns, lower, upper, lambda_fold,
  profile, coefficient_prime, ring_degree) -> Result<Self, AkitaError>` admits
  geometry, challenge budget, and no-wrap, with read-only setup accessors and
  `identity_bytes::<H>()`.
- `commit_binary_clear::<H,F,D,M>(setup, source)` returns the commitment.
- `prove_binary_clear(setup, source, commitment, point, claim, channel)` and
  `verify_binary_clear(setup, commitment, point, claim, channel)` leave the
  caller's channel unfinished. Both return `Result<(), AkitaError>`.
- `prove_binary_clear_bytes(...) -> Result<Vec<u8>, AkitaError>` and
  `verify_binary_clear_bytes(..., proof) -> Result<(), AkitaError>` create
  backend channels. The verifier wrapper enforces EOF.
- `frontend::{prove_frontend,verify_frontend}` return the mandatory
  `BinaryEvaluationClaim` source-opening obligation. This boundary is ready
  for the later mixed-root consumer without changing the frontend.
- Shared pure functions expose binary equality weights, scalar conversion,
  challenge conversion, source-column packing, matrix action, left expansion,
  integer fold, response packing/parity, and endpoint verification.
- `codec` exposes fixed-width binary exchange and canonical interval offsets;
  `channel` contains all transcript types, transport operations, fold-draw
  adapters, channel constructors/finishers, and matrix digest binding.

All transport use is confined to `channel.rs`. The common sumcheck transition
and the column expansion equation each have one canonical implementation.
Every new Rust file is below 700 lines; the largest is 326 lines.

## Commands and literal final results

Cargo test/check/Clippy invocations used the repository-required `rtk` prefix;
commands below show the actual prefix where used. Cargo compilation commands
ran sequentially. All final commands exited 0.

| Command | Literal result |
| --- | --- |
| `cargo fmt --all --check` | no output; exit 0 |
| `taplo fmt --check` | `found files total=37 excluded=0`; no formatting errors; exit 0 |
| `rtk cargo check -p akita-labinius-verifier -p akita-labinius-prover --no-default-features --features labinius,transcript-blake2b` | `cargo check (61 crates compiled)`; `Finished dev profile [unoptimized + debuginfo] target(s) in 5.19s`; exit 0 |
| `rtk cargo test -p akita-labinius-verifier -p akita-labinius-prover --no-default-features --features labinius,transcript-blake2b` | `cargo test: 12 passed (8 suites, 5.31s)`; exit 0 |
| `rtk cargo test -p akita-labinius-verifier -p akita-labinius-prover --no-default-features --features labinius,transcript-keccak` | `cargo test: 12 passed (8 suites, 5.28s)`; exit 0 |
| `rtk cargo clippy -p akita-labinius-verifier -p akita-labinius-prover --all-targets --no-default-features --features labinius,transcript-blake2b -- -D warnings` | `cargo clippy: No issues found`; exit 0 |
| `rtk cargo clippy -p akita-labinius-verifier -p akita-labinius-prover --all-targets --no-default-features --features transcript-blake2b -- -D warnings` | `cargo clippy: No issues found`; exit 0 |
| `rtk cargo clippy -p akita-labinius-verifier -p akita-labinius-prover --all-targets --no-default-features --features labinius,transcript-keccak -- -D warnings` | `cargo clippy: No issues found`; exit 0 |
| `scripts/check-crate-deps.sh akita-labinius-verifier` | `akita-labinius-verifier dependency hygiene check passed`; exit 0 |
| `scripts/check-crate-deps.sh akita-labinius-prover` | `akita-labinius-prover dependency hygiene check passed`; exit 0 |
| `scripts/check-crate-deps.sh akita-verifier` | `akita-verifier dependency hygiene check passed`; exit 0 |
| `scripts/check-crate-deps.sh akita-prover` | `akita-prover dependency hygiene check passed`; exit 0 |
| `scripts/check-rust-file-lines.sh --no-baseline` | `Rust file line-cap check passed: scanned 931 tracked Rust files; cap=1500; baseline_entries=0; generated_skipped=0.`; exit 0 |
| `scripts/test-rust-file-lines.sh` | `Rust file line-cap self-tests passed.`; exit 0 |
| `python3 -m unittest discover -s scripts/tests -p 'test_*.py'` | `Ran 179 tests in 5.602s`; `OK`; exit 0 |
| `scripts/check-shared-field-identity.sh` | matching root/fuzz/recursion field and polynomial identity at Jolt revision `c7c2d08d5bb0c7c80501f29f723e595af7fcff3c`; exit 0 |
| `scripts/check-external-schedule-artifacts.sh` | `external schedule artifact source guards passed`; exit 0 |
| `typos` | no output; exit 0 |
| `./scripts/check-doc-guardrails.sh` | `All documentation guardrails passed.`; exit 0 |
| `cargo machete --with-metadata` | `cargo-machete didn't find any unused dependencies in this directory. Good job!`; exit 0 |
| `git diff --check` | no output; exit 0 |
| `! { git diff --name-only origin/dev -- crates; git ls-files --others --exclude-standard -- crates; } \| grep -v '^crates/akita-labinius-'` | no output; exit 0 |
| `! cargo tree -p akita-labinius-prover -e features --no-default-features --features transcript-blake2b \| grep -E 'labinius-(binary\|trinomial\|challenges\|sis)'` | no output; exit 0 |
| `! cargo tree -p akita-labinius-verifier -e features --no-default-features --features transcript-blake2b \| grep -E 'labinius-(binary\|trinomial\|challenges\|sis)'` | no output; exit 0 |

`rustfmt --edition 2021` on each owned Rust file and the worker's corresponding
`--check` invocations returned exit 0 with no output. Read-only inspection used
`cat`, `sed`, `rg`, `git status`, `git diff`, `git rev-parse`, and `wc -l`.
One early lookup of nonexistent transcript/ring paths reported those paths
missing; the public APIs were then located in their actual modules. No private
reference implementation was involved.

The documentation build emitted the existing mdbook Mermaid/KaTeX preprocessor
version warnings and still exited 0. The Python script fixtures also printed
mock AI-review status messages while returning `OK`.

### Development failures and repairs

- Initial formatting checks reported diffs in unfinished new files. Formatting
  and the final repository-wide check passed. The independent review also
  observed an initial exit 1 and a final exit 0.
- The first shared-field identity attempt exited 101 because new workspace
  registration required a lockfile update and that script invokes locked Cargo
  metadata. The targeted Cargo check generated only the two new lock entries;
  the repeated identity check passed.
- Three intermediate Blake2b test builds exited 101, respectively reporting
  `E0599` for `F::zero`, `E0599` for `F::from_u64`, and `E0597` for an inferred
  test-closure borrow lifetime. The fixture now uses a default coefficient and
  explicit borrowed closure types. All-targets Clippy and both test suites pass.
- An earlier completed Blake2b suite printed
  `cargo test: 11 passed (8 suites, 5.33s)`. After the requested multi-round
  tamper coverage was added, the final suite printed the 12-test result above.
  An earlier Keccak suite printed
  `cargo test: 12 passed (8 suites, 5.36s)` before the final common-round refactor.
- Initial `cargo machete --with-metadata` exited 1 with
  `akita-labinius-prover: akita-transcript` unused. The manifest requires this
  direct dependency to forward exactly the requested backend features, while
  all Rust transcript operations live in the verifier channel module. A
  narrowly documented cargo-machete metadata exception covers that required
  feature-forwarding dependency. No test or check was removed or weakened.

## Coverage and independent review

The 12 tests include both hosts, both coefficient primes, D162 and D648, matrix
ranks 1 and 2, D324, and additional asymmetric packed geometry. They compare
commitments to schoolbook ring action; frontend partials, all rounds and terminal
values to dense definitions; transparent tensor weights to coefficient MLEs;
column expansion to direct MLEs; integer folding to wide convolution/reduction;
parity to F162 products; and embedding/packing to componentwise action.

Tampering covers all live partials, every atom of a four-round sumcheck,
terminal values, all column expansions, each coefficient with parity-preserving
+2, a +1 with nonzero prime image, statement/setup/profile changes, host/prime/
degree changes, malformed high bits, truncation/trailing bytes, out-of-range
unsigned offsets, and every scalar response row. Admission failures and
out-of-interval prover failure without retry are tested. A forced zero
transparent weight still rejects an invalid original-source commitment.

The independent reviewer authored no implementation. Its own targeted opt-in
all-targets Clippy compilation passed, along with formatting, line caps, 179
Python tests, shared-field identity, spelling, documentation, and diff checks.
Final verdict: `VERDICT: approve`, with no unresolved findings. The review's
zero-interval assertion and multi-round tamper coverage concerns were resolved.

## Duplication to resolve and API assumptions

Duplication to resolve: **none**. No private foundation API was copied.

- `validate_packing` is private. The public `pack_scalar_components` already
  performs that validation; setup admission separately checks the required
  degree/sign geometry, so no private helper had to be reimplemented.
- `field_digest` is exported but accepts only fields, not a geometry prefix.
  Matrix binding composes it with a framed fresh-channel header as specified
  in the protocol record. Every such use remains inside `channel.rs`.
- The challenge multiplication bound API returns `u64`; the occurrence bound
  constructor accepts `u128`. Admission explicitly widens it.
- The brief calls D648/m2/C8 asymmetric, but `k*m = 4*2 = 8 = C`. Requested
  examples remain; additional D648/m2/C4 and D324 tests exercise asymmetry.
- For a zero-width accepted interval `[0,0]`, the wire codec uses one zero byte,
  consistent with the minimum nonempty fixed-byte encoding. The spec and
  endpoint tests explicitly pin that convention.

Nothing remains open within this task. Production SIS admission, outer/root
integration, projection/composition security, and zero knowledge remain the
explicit non-goals of the clear differential oracle.
