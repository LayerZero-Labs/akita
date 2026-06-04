# Spec: w-to-e Notation Cutover

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-06-04                     |
| Status      | implemented |
| PR          | #150                           |

## Summary

Align the Akita implementation with the paper's opening-side notation.
Greyhound/Hachi $\hat{\mathbf{w}}$ for per-block opening digits becomes
$\hat{\mathbf{e}}$ ($e_i := \langle \mathbf{a}, \mathbf{f}_i\rangle$,
$\hat e_i := \mathbf{G}_{b_1,1}^{-1}(e_i)$).
This is a behavior-preserving identifier rename across the opening pipeline
(`e_folded`, `e_hat`, terminal transcript segments) plus one transcript-label
correction: `ABSORB_SUMCHECK_W` misstates what is absorbed at every level.

The cutover deliberately does **not** touch the full next-level witness
$\mathbf{w}$, its packed size, its commitment, or its MLE evaluation.
It introduces no proof-struct field renames, no serialization changes, and no
generated-schedule-table changes, so serialized proofs and transcript sponge
bytes are byte-identical before and after.

Older specs (for example `core-protocol-naming-cleanup.md`, status
implemented) are historical.
They explicitly kept `w_hat` and `ABSORB_SUMCHECK_W`; this document supersedes
those naming choices for opening notation only.
Do not edit historical spec files; treat them as pre-cutover records.

Paper alignment for §§3--5 landed in `lattice-jolt` (opening digits
$\hat{\mathbf{e}}$, removal of $z_{\mathsf{pre}}$ prose).

## Intent

### Goal

One full cutover PR that renames the **opening-side** `w` cluster (the
per-block opening digits $\hat{\mathbf{e}}$ and the pre-digit folded openings
$e_i$) to `e`, and renames the ring-switch witness-binding transcript label so
it no longer claims to absorb a cleartext witness coefficient vector before
sumcheck.

Keep `w` everywhere the object is the **full next-level recursive witness**
$\mathbf{w}$ (hypercube / packed column, its commitment $\mathbf{u}'$, its size,
and its MLE), not the partially evaluated opening blocks.

### Notation glossary (paper ↔ Rust target)

| Paper | Current Rust (representative) | Target Rust |
|-------|------------------------------|-------------|
| $e_i \in R_q$ (pre-digit opening) | `w_folded` | `e_folded` |
| $\hat e_i$, $\hat{\mathbf{e}}$ | `w_hat` | `e_hat` |
| $v = \mathbf{D}\hat{\mathbf{e}}$ | `RingRelationInstance::v`, `ABSORB_PROVER_V` | keep `v` / `ABSORB_PROVER_V` |
| next-level witness binding slot | `ABSORB_SUMCHECK_W` | `ABSORB_NEXT_LEVEL_WITNESS_BINDING` (see below) |
| $\mathbf{u}' = \mathrm{Commit}(\mathbf{w})$ | `next_w_commitment` | **keep `next_w_commitment`** |
| $\mathbf{w} = (\hat{\mathbf{e}}, \hat{\mathbf{t}}, \hat{\mathbf{z}}, \hat{\mathbf{r}}, \ldots)$ | `build_w_coeffs`, `w_ring_element_count`, `num_w_vectors` | **keep `w`** |
| $\mathsf{mle}_{\mathbf{w}}$ | sumcheck `w` table, stage-2 `next_w_eval` | **keep `w`** |
| Terminal cleartext $\mathbf{w}^{(t)}$ | `final_witness` field | **keep field name**; relabel absorb (below) |

**Do not conflate:**

- **$v$** (opening commitment, $\mathbf{D}\hat{\mathbf{e}}$) with **$\mathbf{u}'$**
  (next-level witness commitment).
- **`e_folded`** (ring openings before digits) with **`w_ring_element_count`**
  (size of full packed $\mathbf{w}$) or **`num_w_vectors`** (count of root
  relation $\mathbf{w}$ vectors that *sizes* $\mathbf{w}$).

### Transcript label: `ABSORB_SUMCHECK_W` → `ABSORB_NEXT_LEVEL_WITNESS_BINDING`

**Problem.**
The current name and comment ("Absorb the `w` coefficient vector before
sumcheck", `crates/akita-transcript/src/labels.rs:79`) describe the wrong
object at every level.
At intermediate folds this slot absorbs **`next_w_commitment`**, the Ajtai
commitment **$\mathbf{u}'$** to the next-level witness polynomial $\mathbf{w}$,
not a cleartext coefficient vector
(`crates/akita-verifier/src/protocol/ring_switch.rs:175-176`,
`crates/akita-prover/src/protocol/ring_switch/finalize.rs:197`).
At the **terminal** fold the *same wire position* absorbs cleartext
`final_witness` (packed $\mathbf{w}$), not a commitment
(`crates/akita-types/src/proof/levels.rs:345-368`,
`crates/akita-verifier/src/protocol/levels.rs:254`).
Opening **$v = \mathbf{D}\hat{\mathbf{e}}$** is already `ABSORB_PROVER_V` and is
unaffected.

**Decision: one payload-agnostic label.**
Replace the single constant `ABSORB_SUMCHECK_W` with a single constant
`ABSORB_NEXT_LEVEL_WITNESS_BINDING`.
This is a pure constant rename: every current `ABSORB_SUMCHECK_W` reference
(intermediate and terminal) becomes `ABSORB_NEXT_LEVEL_WITNESS_BINDING`.

Rationale for a single label rather than splitting intermediate vs terminal:

- It is the smallest coherent change that fixes the misleading name; the new
  name is honest for both payloads (commitment $\mathbf{u}'$ at intermediate
  levels, cleartext $\mathbf{w}$ at the terminal).
- Fiat--Shamir is positional, so the same constant occupies the same wire slot
  at every level; no prover/verifier ordering or domain-separation change.
- It introduces no new wire label and no `logging-transcript` schedule-event
  divergence, so the only `logging-transcript` change is the mechanical
  constant rename in expectations. A split label (a separate terminal tag)
  was considered and rejected: it adds a new byte tag, a new allowlist entry,
  and a terminal-vs-intermediate site partition that a mechanical executor can
  get wrong, for only a marginal documentation gain.

**Wire label bytes:** keep `b"ak/a/w"` unchanged.
Fiat--Shamir is positional; the label text is diagnostic only and does not
enter production sponge bytes (`AGENTS.md` transcript hardening P2).

**Updated doc comment** (replaces the misleading one at `labels.rs:79`):

```rust
/// Binds the next-level witness at this fold step. Intermediate folds absorb
/// the Ajtai commitment `u'` to the next-level witness `w` (`next_w_commitment`);
/// the terminal fold absorbs the cleartext `final_witness` (packed `w`) in the
/// same wire position. Diagnostic label only; sponge bytes are positional.
pub const ABSORB_NEXT_LEVEL_WITNESS_BINDING: &[u8] = b"ak/a/w";
```

**Explicitly not renamed:** the proof field `next_w_commitment`, the stage-2
field `next_w_eval`, and the `final_witness` proof field.
`next_w_commitment` accurately names a commitment to the full next-level
witness $\mathbf{w}$, consistent with keeping `next_w_eval`; renaming only the
absorb label is sufficient to remove the misstatement, and keeping these field
names guarantees zero serialization-byte change.

### Rename table (opening cluster)

Mechanical renames (longest compounds first).
Rust identifiers (consts, fns, fields, types) are compiler-checked: a missed
reference fails the build. String literals (tracing span names) and
doc/comment mentions are **not** compiler-checked and need explicit grep.

| Current | Target | Compiler-checked? |
|---------|--------|-------------------|
| `terminal_w_hat_bytes_from_blocks` | `terminal_e_hat_bytes_from_blocks` | yes |
| `absorb_terminal_w_hat` | `absorb_terminal_e_hat` | yes |
| `ABSORB_TERMINAL_W_HAT` | `ABSORB_TERMINAL_E_HAT` (keep `b"ak/a/twh"`) | yes |
| `w_hat_digit_offset`, `w_hat_digit_count` (struct fields + locals) | `e_hat_digit_offset`, `e_hat_digit_count` | yes |
| `TerminalWitnessTranscriptParts::w_hat` | `::e_hat` | yes |
| `RingRelationWitness::w_hat` | `::e_hat` | yes |
| `w_folded_by_poly`, `w_folded_by_claim` | `e_folded_by_poly`, `e_folded_by_claim` | yes |
| `w_folded` (locals) | `e_folded` | yes |
| `pre_folded_by_poly`, `pre_folded_by_claim` | `pre_folded_e_by_poly`, `pre_folded_e_by_claim` | yes |
| `w_hat` (locals, e.g. in `ring_relation.rs`) | `e_hat` | yes |
| `w_hat_count` (locals, e.g. `schedule.rs`) | `e_hat_count` | yes |
| `ABSORB_SUMCHECK_W` | `ABSORB_NEXT_LEVEL_WITNESS_BINDING` (keep `b"ak/a/w"`) | yes |
| tracing span `"decompose_batched_w_hat"` | `"decompose_batched_e_hat"` | **no (string)** |
| tests `WHatDigit` | `EHatDigit` | yes |

**Explicitly keep (full witness `w`, or unrelated):**

- `build_w_coeffs`, `w_ring_element_count`, `planned_w_ring_element_count`
  (size/build of full packed $\mathbf{w}$).
- `num_w_vectors` (count of root relation $\mathbf{w}$ vectors; a public
  `AkitaScheduleLookupKey` field and the ABI-stable `GeneratedScheduleKey`
  field; the multiplicand of `w_ring_element_count` at
  `crates/akita-types/src/schedule.rs:261-264`).
- `next_w_commitment`, `next_w_eval` (commitment/MLE of full $\mathbf{w}$).
- `final_witness` proof field (terminal cleartext $\mathbf{w}$).
- `ABSORB_TERMINAL_W_REMAINDER` (remainder of $\mathbf{w}$ outside logical
  $\hat e$).
- `t_hat`, `r_hat`, `z_folded_rings`, `commit_w`, `logical_w`, `expected_w_len`.
- **Do not touch** `extension_opening_reduction::final_witness_and_factor_evals`
  and its callers; that `final_witness` is the EOR folded-witness eval, not the
  terminal cleartext witness, and is out of scope.

### Invariants

- **Behavior preserving.** No algorithm, numeric, or serialized proof layout change.
- **Proof bytes unchanged.** No proof-struct field is renamed; witness segment
  order and packed digit semantics are unchanged.
- **Transcript sponge unchanged.** Absorb order and payload bytes unchanged;
  only Rust identifier names, the diagnostic label *constant* names, and label
  doc comments change. Wire byte tags (`b"ak/a/w"`, `b"ak/a/twh"`) are unchanged.
- **Generated tables unchanged.** No schedule-key field is renamed, so
  `crates/akita-planner/src/generated/*.rs` are byte-identical and regeneration
  produces an empty diff.
- **Full cutover.** No `w_hat` / `w_folded` aliases left in `crates/`; no
  `ABSORB_SUMCHECK_W` / `ABSORB_TERMINAL_W_HAT` left in `crates/`.
- **Historical specs untouched.** Amend only this file and code/docs; do not
  rewrite `core-protocol-naming-cleanup.md`.

### Non-Goals

- Renaming the full-witness cluster (`num_w_vectors`, `next_w_commitment`,
  `next_w_eval`, `build_w_coeffs`, `w_ring_element_count`, `final_witness`).
- Renaming `t_hat`, `r_hat`, or `z_folded_rings` → `z_hat`.
- Renaming sumcheck MLE symbols or `GruenSplitEq` / `split_eq`.
- Changing `ABSORB_PROVER_V` bytes or the opening commitment $v$.
- Changing `MRowLayout` `d_key` / `const D` matrix-vs-ring vocabulary (Tier-2 doc follow-up).
- Splitting the witness-binding label into intermediate/terminal tags (rejected above).
- Paper edits (tracked in `lattice-jolt`, separate commit).

## Evaluation

### Acceptance Criteria

- [x] `rg 'w_hat' crates/` and `rg 'w_folded' crates/` return no matches (substring
      grep, so compound identifiers like `w_hat_digit_end` cannot slip through).
- [x] `rg 'ABSORB_SUMCHECK_W' crates/` and `rg 'ABSORB_TERMINAL_W_HAT' crates/`
      return no matches.
- [x] `rg 'decompose_batched_w_hat' crates/` returns no matches (tracing span string).
- [x] `rg '\bnum_w_vectors\b' crates/`, `rg '\bnext_w_commitment\b' crates/`,
      and `rg '\bnext_w_eval\b' crates/` still return matches (kept on purpose).
- [x] `cargo build --workspace` and `cargo build --workspace --all-features` succeed.
- [x] `cargo nextest run --profile ci-non-zk` and `--profile ci-all-features` green.
- [x] `cargo test -p akita-pcs --features logging-transcript --test transcript_hardening` green.
- [x] Generated schedule tables unchanged (no `num_w_vectors` / planner field renames).
- [x] Byte-identical serialized proofs for dense + onehot (`nv=15`, fixtures in
      `/tmp/akita-w-to-e-baseline/`).
- [x] `specs/w-to-e-notation.md` status updated to `implemented`.

### Testing Strategy

- `akita-types` `terminal_witness` unit tests (layout + plane-major bytes)
  must stay green after the `*_w_hat_* → *_e_hat_*` field renames.
- PCS transcript hardening + proptest (`logging-transcript`): the
  `ABSORB_SUMCHECK_W` / `ABSORB_TERMINAL_W_HAT` constant references in
  `crates/akita-pcs/tests/transcript_hardening.rs` and
  `crates/akita-transcript/src/logging.rs` rename with the constants (compiler
  enforces it); event-stream equality and wire-before-squeeze smell checks are
  unaffected because wire bytes and positions are unchanged.
- `regen_diff` (`crates/akita-config/tests/regen_diff.rs`) must show no diff,
  confirming the schedule keys (and thus `num_w_vectors`) are untouched.
- Byte-identical proof fixtures: serialize one dense and one onehot proof on
  `main` (or pre-change `HEAD`) and after the cutover; assert equality.

## Design

### Affected crates and surfaces

- `akita-types`: `src/proof/terminal_witness.rs` (layout fields + helper +
  tests), `src/proof/mod.rs` and `src/lib.rs` (re-exports of
  `terminal_*_bytes_from_blocks`), `src/proof/ring_relation.rs`,
  `src/schedule.rs` (only the `w_hat_count` *local*; the `num_w_vectors`
  *field* stays).
- `akita-transcript`: `src/labels.rs` (constant + allowlist array + doc comment),
  `src/logging.rs` (test references).
- `akita-prover`: `src/protocol/ring_relation.rs`,
  `src/protocol/ring_relation_witness.rs`, `src/protocol/ring_switch*.rs`,
  <!-- `src/protocol/flow/*` (label references only; `final_witness`/`logical_w` -->
  identifiers stay), kernels referencing `w_hat`/`w_folded`.
- `akita-verifier`: `src/protocol/levels.rs`, `src/protocol/levels/recursive.rs`,
  `src/protocol/ring_switch.rs` (label references + doc comments).
- `akita-pcs`: `tests/*` referencing the renamed symbols.
- `scripts/tail_dimension_model.py`: any `w_hat`/`w_folded` comment or symbol.

### Weak-model execution protocol

This is a behavior-preserving rename. Execute it conservatively:

1. Rename **one identifier at a time, across all crates and tests at once**.
   Never leave a half-renamed identifier; the build will break.
2. After each slice, run `cargo build --workspace --all-features` and only
   advance when it is green.
3. Rust consts/fns/fields/types are compiler-checked; rely on the build to
   find every reference. For the two non-compiler-checked items (the tracing
   span string `"decompose_batched_w_hat"` and doc/comment text), grep
   explicitly.
4. Do **not** rename anything in the "Explicitly keep" list. In particular, do
   not touch `num_w_vectors`, `next_w_commitment`, `next_w_eval`, the
   `final_witness` field, or `extension_opening_reduction::final_witness_*`.
5. Make no logic edits. If a rename appears to require a logic change, stop and
   report; that is a sign the symbol is not a pure opening-side rename.

### Implementation order (slices)

**Slice 0 — Baseline fixtures (no code change).**
Capture byte-identical proof fixtures for one dense and one onehot config on
the current `HEAD` (serialize and save the bytes to a scratch location outside
the repo). These are the before-images for the final byte-identical check.
Verify: fixtures saved; `cargo build --workspace` green.

**Slice 1 — `w_hat` opening-digit cluster.**
Rename, atomically across all crates/tests:
`terminal_w_hat_bytes_from_blocks → terminal_e_hat_bytes_from_blocks`,
`absorb_terminal_w_hat → absorb_terminal_e_hat`,
`w_hat_digit_offset → e_hat_digit_offset`,
`w_hat_digit_count → e_hat_digit_count`,
`TerminalWitnessTranscriptParts::w_hat → ::e_hat`,
`RingRelationWitness::w_hat → ::e_hat`,
remaining `w_hat` locals → `e_hat`, `w_hat_count` → `e_hat_count`,
tests `WHatDigit → EHatDigit`.
Also re-export sites in `akita-types/src/proof/mod.rs` and `src/lib.rs`.
Verify: `rg 'w_hat' crates/` empty; build green.

**Slice 2 — `w_folded` opening-fold cluster.**
Rename `w_folded_by_poly → e_folded_by_poly`,
`w_folded_by_claim → e_folded_by_claim`,
`pre_folded_by_poly → pre_folded_e_by_poly`,
`pre_folded_by_claim → pre_folded_e_by_claim`,
remaining `w_folded` locals → `e_folded`.
Verify: `rg 'w_folded' crates/` empty; build green.

**Slice 3 — tracing span string.**
Change the `tracing::info_span!("decompose_batched_w_hat")` literal in
`crates/akita-prover/src/protocol/ring_relation.rs` to
`"decompose_batched_e_hat"`.
Verify: `rg 'decompose_batched_w_hat' crates/` empty; build green.

**Slice 4 — witness-binding label.**
Rename the constant `ABSORB_SUMCHECK_W → ABSORB_NEXT_LEVEL_WITNESS_BINDING` in
`crates/akita-transcript/src/labels.rs` (definition, the allowlist array, and
the doc comment per the Transcript-label section). The compiler updates every
reference (prover, verifier, tests). Also rename
`ABSORB_TERMINAL_W_HAT → ABSORB_TERMINAL_E_HAT` (keep `b"ak/a/twh"`).
Update the verifier/prover doc comments that mention `ABSORB_SUMCHECK_W`.
Verify: `rg 'ABSORB_SUMCHECK_W' crates/` and `rg 'ABSORB_TERMINAL_W_HAT' crates/`
empty; build green;
`cargo test -p akita-pcs --features logging-transcript --test transcript_hardening` green.

**Slice 5 — docs, script, comments, final verification.**
Sweep remaining `w_hat`/`w_folded` mentions in comments and
`scripts/tail_dimension_model.py`. Run the full acceptance-criteria gate:
both nextest profiles, `regen_diff`, and the byte-identical fixture comparison
against Slice 0. Update this spec's Status to `implemented` with the commit.

### Worktree

The branch `quang/w-to-e-notation` already exists and is the implementation
worktree (this PR's branch). Implement directly on it; no new worktree needed.

## Documentation

- Paper: `lattice-jolt` §§3--5 (merged separately).
- This spec is the authoritative post-cutover naming index.
- Optional: short module doc glossary in `akita-types/src/proof/mod.rs`.

## References

- `lattice-jolt/sections/akita/3_basic_akita.tex` (opening $\hat{\mathbf{e}}$).
- `specs/core-protocol-naming-cleanup.md` (historical; kept `w_hat`).
- `specs/terminal-fold-cutover.md` (terminal cleartext absorb at the
  ex-`ABSORB_SUMCHECK_W` slot).
- `crates/akita-transcript/src/labels.rs:79-81` (label + misleading comment).
- `crates/akita-types/src/schedule.rs:80-81,261-264` (`num_w_vectors` semantics).
