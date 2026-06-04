# Spec: Consolidate `akita-scheme` into `akita-pcs`

| Field     | Value                        |
|-----------|------------------------------|
| Author(s) | @omibo                       |
| Created   | 2026-06-04                   |
| Status    | proposed                     |
| PR        | TBD                          |

## Summary

Today the end-to-end PCS lives in two workspace crates:

- `akita-scheme` — a thin (~700-line) crate that owns the `AkitaCommitmentScheme`
  type, its `CommitmentProver` / `CommitmentVerifier` impls, the `D`-dispatch
  helpers (`dispatch_prove_level`, `dispatch_prove_terminal_level`), and the
  `#[cfg(test)]` orchestration test suite under `src/tests/`.
- `akita-pcs` — the umbrella crate that re-exports the public Akita surface and
  owns all integration tests (`tests/`), benches (`benches/`), and the `profile`
  example. `akita-pcs` depends on `akita-scheme` and re-exports
  `AkitaCommitmentScheme` from it.

The two crates are 1:1 coupled: `akita-scheme` is consumed **only** by
`akita-pcs`, and the sole `akita_scheme::` import in the whole tree is the
single re-export line in `akita-pcs/src/lib.rs`. The split therefore buys no
dependency-slimness or reuse — both crates sit at the top of the graph and pull
the full prover + verifier + setup + config surface.

This spec proposes collapsing the two into **one** crate. It recommends keeping
the published umbrella name `akita-pcs` (folding `akita-scheme`'s code into it
and deleting `akita-scheme`), and documents the alternative of keeping the name
`akita-scheme` as the user originally suggested. The project makes **no**
backward-compatibility guarantees, so renames and crate removals are acceptable.

## Background: why they are split today

The split originated in the crate-decomposition refactor
([`specs/akita-pcs-crate-decomposition.md`](akita-pcs-crate-decomposition.md)),
which defined:

- `akita-scheme`: "end-to-end PCS orchestration, exposing `AkitaCommitmentScheme`
  and wiring `akita-config`, `akita-setup`, `akita-prover`, and `akita-verifier`
  together."
- `akita-pcs`: "the aggregate package … re-exporting the public Akita surface."

The decomposition's load-bearing invariant is **verifier slimness**: a
verifier-only consumer (e.g. Jolt) depends on `akita-verifier` + `akita-types` +
`akita-config`, never on the umbrella. That invariant is enforced between
`akita-verifier` and `akita-prover` and guarded by
`scripts/check-crate-deps.sh`. It does **not** depend on the `scheme` vs `pcs`
boundary: both are above the verifier/prover split, and neither is reachable
from a verifier-only build.

In other words, the `scheme` / `pcs` boundary is purely cosmetic ("orchestration
logic" vs "aggregation + harness"). It is not part of the dependency-direction
contract that protects Jolt integration.

## Intent

### Goal

Reduce the workspace from two top-of-graph crates to one, eliminating a crate
boundary that has exactly one internal consumer and provides no
dependency-isolation value. The single crate owns the `AkitaCommitmentScheme`
orchestration **and** the public re-exports, integration tests, benches, and
the profile example.

### Recommendation

**Consolidate, keeping the name `akita-pcs`. Delete `akita-scheme`.**

Rationale for keeping `akita-pcs` as the surviving name (vs. the user's
suggested `akita-scheme`):

1. `akita-pcs` is the published, public-facing package name: `documentation =
   "https://docs.rs/akita-pcs"`, `repository`, README, and `keywords`/`categories`
   metadata all describe `akita-pcs` as *the* scheme package.
2. The only external (out-of-workspace) consumer,
   `profile/akita-recursion/artifact`, depends on `akita-pcs` with specific
   features. Keeping the name leaves that manifest untouched.
3. Docs, specs, CI, the dependency diagram, and `AGENTS.md` already treat
   `akita-pcs` as the umbrella. The decomposition spec explicitly names
   `akita-pcs` as the repository/scheme name.
4. The merge then becomes a one-directional fold (move ~700 lines of scheme
   into pcs and delete scheme), instead of a rename that touches the external
   artifact, docs.rs metadata, README, AGENTS, and multiple specs.

The `AkitaCommitmentScheme` *type* name does not change under either option;
only the crate package name is in question.

### Non-Goals

1. No protocol, transcript, serialization, schedule, or field/ring arithmetic
   behavior changes. This is a pure packaging move: proof bytes, Fiat-Shamir
   streams, and all existing tests must be byte-for-byte unchanged.
2. No changes to the prover/verifier/config/setup/planner crate boundaries.
   The verifier-slimness contract and `scripts/check-crate-deps.sh` rules for
   `akita-verifier`, `akita-prover`, `akita-config`, `akita-planner`, and
   `akita-setup` are untouched.
3. No new public API beyond what `akita-pcs` already re-exports plus the
   now-owned `AkitaCommitmentScheme`.
4. No migration of the unrelated `profile/akita-recursion/` sub-workspace beyond
   what the chosen naming option requires (none, if we keep `akita-pcs`).

## Why consolidation is safe (current coupling)

| Fact | Evidence |
|------|----------|
| `akita-scheme` is consumed only by `akita-pcs` | Only Cargo.toml dependency edge to `akita-scheme` is in `akita-pcs/Cargo.toml`. |
| Only one `akita_scheme::` import exists | `akita-pcs/src/lib.rs:66` — `pub use akita_scheme::AkitaCommitmentScheme;`. |
| `akita-pcs` deps ⊇ `akita-scheme` deps | `akita-pcs` already depends on `akita-config`, `akita-setup`, `akita-prover`, `akita-verifier`, `akita-types`, `akita-transcript`, `akita-serialization`, `akita-field`, `akita-challenges`. |
| Scheme tests' dev-dep already present in pcs | Both crates declare `akita-config` `test-support` as a dev-dependency. |
| Nothing imports `akita_scheme` outside the workspace | `profile/akita-recursion/*` depends on `akita-pcs`, not `akita-scheme`. |

Because `akita-pcs`'s production and dev dependency sets are already supersets
of `akita-scheme`'s, folding the scheme source in requires no new dependencies —
only the removal of the `akita-scheme` edge.

## Design

### Target layout (recommended: keep `akita-pcs`)

Move `akita-scheme/src/lib.rs` into a new private module in `akita-pcs`, and move
the scheme's `src/tests/` suite alongside it as the module's unit tests.

```text
crates/akita-pcs/
  src/
    lib.rs            # umbrella re-exports; now also `mod scheme; pub use scheme::AkitaCommitmentScheme;`
    scheme.rs         # moved from akita-scheme/src/lib.rs (AkitaCommitmentScheme + dispatch helpers)
    scheme/
      tests/          # moved from akita-scheme/src/tests/ (mod.rs, single.rs, batched.rs,
                      #   onehot.rs, layout.rs, fp32_ring_subfield.rs)
  tests/              # unchanged integration tests
  benches/            # unchanged
  examples/           # unchanged
  Cargo.toml          # drop akita-scheme dep + feature forwards
```

(`scheme.rs` + `scheme/tests/` may instead be `scheme/mod.rs` + `scheme/tests/`;
choice is cosmetic. The existing `#[cfg(test)] mod tests;` line in the moved file
keeps working with the latter.)

### `akita-pcs/src/lib.rs` changes

- Replace `pub use akita_scheme::AkitaCommitmentScheme;` with `mod scheme;` and
  `pub use scheme::AkitaCommitmentScheme;`.
- All other re-exports are unchanged.
- The moved code currently relies on `#![warn(missing_docs)]`/`#![warn(unreachable_pub)]`
  being *absent* in `akita-scheme`. `akita-pcs` sets both. The
  `AkitaCommitmentScheme` struct is already documented; its internal `fn`s are
  private (no `missing_docs` impact), and trait-impl methods don't require docs.
  Verify with clippy; add `#[allow(unreachable_pub)]` or doc comments only if the
  lint actually fires on moved items.

### `akita-pcs/Cargo.toml` changes

- Remove the `akita-scheme` dependency line.
- Remove `"akita-scheme/parallel"` from the `parallel` feature and
  `"akita-scheme/zk"` from the `zk` feature. The equivalent direct forwards
  (`akita-prover/parallel`, `akita-verifier/parallel`, `akita-setup/parallel`,
  `akita-config/zk`, etc.) are already present, so feature behavior is preserved.
- Keep `[package.metadata.cargo-machete] ignored = ["akita-setup"]`; re-check
  machete after the move (scheme used `akita-setup` directly; pcs uses it through
  setup re-export — confirm it is still a real, used dependency and adjust the
  ignore list if machete complains).

### Workspace + tooling changes

- `Cargo.toml` (root): remove `"crates/akita-scheme"` from `members`.
- `scripts/check-crate-deps.sh`: remove the `akita-scheme)` case.
- `.github/workflows/ci.yml`: remove the `scripts/check-crate-deps.sh akita-scheme`
  line.
- Delete the `crates/akita-scheme/` directory.

### Documentation changes

- `AGENTS.md`: remove the `akita-scheme` bullet; update the `akita-pcs` bullet to
  state it now owns `AkitaCommitmentScheme` orchestration in addition to
  re-exports/examples/benches/integration tests.
- `docs/crate-graph.md`: remove the `Scheme` node and its edges; re-point the
  `Pcs --> Scheme` edge's downstream edges (`Config`, `Prover`, `Setup`,
  `Verifier`, etc.) directly onto `Pcs` (most already exist).
- Specs that reference `akita-scheme` as a live crate
  (`specs/akita-compute-backend-metal.md`, `specs/planner-config-consolidation.md`,
  `specs/security-hardening.md`, `docs/soundness-audit.md`,
  `profile/akita-recursion/README.md`): these are historical/contextual. Update
  only the operational references that would otherwise break (e.g.
  `cargo test -p akita-scheme --lib` → `cargo test -p akita-pcs --lib`,
  `scripts/check-crate-deps.sh akita-scheme`). Leave historical narrative intact;
  do not rewrite closed specs.

### Behavior preservation

- `AkitaCommitmentScheme` and all its trait impls move verbatim; no signature or
  body changes.
- The scheme unit tests move verbatim and run as `akita-pcs` lib tests.
- Integration tests, benches, and the profile example already import via
  `akita_pcs::…`; the re-export of `AkitaCommitmentScheme` is preserved, so they
  need no changes.

## Alternative: keep the name `akita-scheme` (user's original framing)

If you prefer the surviving crate to be named `akita-scheme`, the mechanical work
is the mirror image and strictly larger:

- Move `akita-pcs`'s `lib.rs` re-exports, `tests/`, `benches/`, `examples/`, and
  the relevant `Cargo.toml` sections into `akita-scheme`.
- Rename the `akita-scheme` package metadata to claim the public role
  (`documentation`, `repository`, `keywords`, `categories`, `readme`, the bench
  `[[bench]]`/`[[example]]` tables).
- Update `profile/akita-recursion/artifact/Cargo.toml` to depend on
  `akita-scheme` (with the `parallel` feature) instead of `akita-pcs`, plus its
  `src/main.rs` imports.
- Update `docs.rs`/README/AGENTS/CI/`check-crate-deps.sh`/specs from `akita-pcs`
  to `akita-scheme` wherever they name the public package.
- Delete `crates/akita-pcs/`.

This is viable (no backward-compat constraint) but touches the external artifact
and all public-name metadata for no functional gain over the recommended
direction. **Recommendation stands: keep `akita-pcs`.**

A third option — **keep them separate** — is also on the table but not
recommended: the boundary has one consumer, adds no isolation, and forces the
extra `scheme/{parallel,zk}` feature forwarding and an extra CI dep-check for
purely cosmetic reasons.

## Evaluation

### Acceptance Criteria

- [ ] Workspace has no `akita-scheme` member; `crates/akita-scheme/` is deleted.
- [ ] `AkitaCommitmentScheme` is defined in `akita-pcs` and still re-exported as
      `akita_pcs::AkitaCommitmentScheme`.
- [ ] The former scheme unit tests run as `akita-pcs` lib tests and pass.
- [ ] `akita-pcs/Cargo.toml` no longer references `akita-scheme` (dep or feature
      forwards); `parallel` and `zk` feature behavior is unchanged.
- [ ] `scripts/check-crate-deps.sh akita-scheme` is removed from CI; all remaining
      dep-hygiene checks still pass.
- [ ] `docs/crate-graph.md` and `AGENTS.md` reflect the single-crate layout.
- [ ] `cargo fmt -q` passes.
- [ ] `cargo clippy --all --all-targets --all-features --message-format=short -q -- -D warnings`
      passes (in particular, no new `missing_docs`/`unreachable_pub` violations
      on the moved orchestration code).
- [ ] `cargo test` passes at the workspace root.
- [ ] `cargo test -p akita-pcs --no-default-features` and
      `cargo test -p akita-pcs --features zk` pass.
- [ ] `profile/akita-recursion/artifact` still builds against `akita-pcs`
      unchanged (recommended option).
- [ ] `AKITA_MODE=onehot_fp128_d128 AKITA_NUM_VARS=25 cargo run --release --example profile`
      produces the same proof bytes / level counts as before the move.

### Testing strategy

- Rely on the existing scheme unit tests + `akita-pcs` integration tests
  (`akita_e2e.rs`, `single_poly_e2e.rs`, `multipoint_batched_e2e.rs`,
  `batched_aggregated_e2e.rs`, `zk.rs`, `ring_switch.rs`, `setup.rs`) as the
  regression net. No new tests are required for a pure packaging move.
- Confirm `cargo machete` is clean for `akita-pcs` after the dependency edit.

### Risks

- **Lint deltas on moved code** (low): `akita-pcs` enables `missing_docs` /
  `unreachable_pub` that `akita-scheme` did not. Mitigated by a clippy pass;
  fixes are doc comments or visibility tweaks, not logic changes.
- **`cargo-machete` false positives** (low): the dependency set changes; re-run
  machete and adjust the `ignored` list if needed.
- **Stale spec/doc references** (cosmetic): handled by updating only operational
  command references and leaving historical narrative.

## Execution outline

1. Create `crates/akita-pcs/src/scheme/mod.rs` from `akita-scheme/src/lib.rs`
   and move `akita-scheme/src/tests/` to `crates/akita-pcs/src/scheme/tests/`.
2. Wire `mod scheme; pub use scheme::AkitaCommitmentScheme;` into
   `akita-pcs/src/lib.rs`; drop the `akita_scheme` re-export.
3. Edit `akita-pcs/Cargo.toml`: remove the `akita-scheme` dependency and feature
   forwards.
4. Remove `crates/akita-scheme` from root `Cargo.toml` members; delete the
   directory.
5. Update `scripts/check-crate-deps.sh` and `.github/workflows/ci.yml`.
6. Update `AGENTS.md` and `docs/crate-graph.md`; fix operational `-p akita-scheme`
   command references in docs/specs.
7. Run `cargo fmt`, `cargo clippy … -D warnings`, `cargo machete`, `cargo test`,
   `cargo test -p akita-pcs --no-default-features`, `cargo test -p akita-pcs
   --features zk`, and the profile example; confirm proof-byte parity.

## References

- Aggregate crate: [`crates/akita-pcs/src/lib.rs`](../crates/akita-pcs/src/lib.rs)
- Orchestration crate: [`crates/akita-scheme/src/lib.rs`](../crates/akita-scheme/src/lib.rs)
- Crate-decomposition origin: [`specs/akita-pcs-crate-decomposition.md`](akita-pcs-crate-decomposition.md)
- Dependency diagram: [`docs/crate-graph.md`](../docs/crate-graph.md)
- Dependency hygiene guard: [`scripts/check-crate-deps.sh`](../scripts/check-crate-deps.sh)
