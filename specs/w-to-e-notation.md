# Spec: w-to-e Notation Cutover

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao                      |
| Created     | 2026-06-04                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Align the Akita implementation with the paper's opening-side notation:
Greyhound/Hachi $\hat{\mathbf{w}}$ for per-block opening digits becomes
$\hat{\mathbf{e}}$ ($e_i := \langle \mathbf{a}, \mathbf{f}_i\rangle$,
$\hat e_i := \mathbf{G}_{b_1,1}^{-1}(e_i)$).
This is a behavior-preserving identifier rename across the opening
pipeline (`e_folded`, `e_hat`, schedule counts, terminal transcript
segments) plus one deliberate transcript-label correction:
`ABSORB_SUMCHECK_W` misstates what is absorbed at non-terminal levels.

Older specs (for example `core-protocol-naming-cleanup.md`, status
implemented) are historical.
They explicitly kept `w_hat` and `ABSORB_SUMCHECK_W`; this document
supersedes those naming choices for opening notation only.
Do not edit historical spec files; treat them as pre-cutover records.

Paper alignment for §§3--5 landed in `lattice-jolt` (opening digits
$\hat{\mathbf{e}}$, removal of $z_{\mathsf{pre}}$ prose).

## Intent

### Goal

One full cutover PR that renames the **opening-side** `w` cluster to `e`
and renames the ring-switch **next-witness binding** transcript label so
it no longer claims to absorb the witness polynomial `w`.

Keep `w` only where the object is the **full next-level recursive witness**
(hypercube / packed column), not partially evaluated opening blocks.

### Notation glossary (paper ↔ Rust target)

| Paper | Current Rust (representative) | Target Rust |
|-------|------------------------------|-------------|
| $e_i \in R_q$ (pre-digit opening) | `w_folded` | `e_folded` |
| $\hat e_i$, $\hat{\mathbf{e}}$ | `w_hat` | `e_hat` |
| $v = \mathbf{D}\hat{\mathbf{e}}$ | `RingRelationInstance::v`, `ABSORB_PROVER_V` | keep `v` / `ABSORB_PROVER_V` |
| $\mathbf{u}' = \mathrm{Commit}(\mathbf{w})$ | `next_w_commitment`, `ABSORB_SUMCHECK_W` | `next_witness_commitment`, `ABSORB_NEXT_WITNESS_COMMITMENT` |
| $\mathbf{w} = (\hat{\mathbf{e}}, \hat{\mathbf{t}}, \hat{\mathbf{z}}, \hat{\mathbf{r}}, \ldots)$ | `build_w_coeffs`, `w_ring_element_count` | **keep `w`** |
| $\mathsf{mle}_{\mathbf{w}}$ | sumcheck `w` table, stage-2 `next_w_eval` | **keep `w`** |
| Terminal cleartext $\mathbf{w}^{(t)}$ | `final_witness` at terminal slot | keep field name; relabel absorb (below) |

**Do not conflate:**

- **$v$** (opening commitment, $\mathbf{D}\hat{\mathbf{e}}$) with **$\mathbf{u}'$**
  (next-level witness commitment).
- **`e_folded`** (ring openings before digits) with **`w_ring_element_count`**
  (size of full packed $\mathbf{w}$).

### Transcript label: `ABSORB_SUMCHECK_W` → `ABSORB_NEXT_WITNESS_COMMITMENT`

**Problem.** The current name and comment (“absorb the `w` coefficient vector
before sumcheck”) describe the wrong object.
At intermediate folds the prover absorbs **`next_w_commitment`**, the Ajtai
commitment **$\mathbf{u}'$** to the next-level witness polynomial
$\mathbf{w}$, not cleartext $\mathbf{w}$ and not opening **$v$**.
Opening **$v = \mathbf{D}\hat{\mathbf{e}}$** is already
`ABSORB_PROVER_V`.

**Proposed Rust identifier:** `ABSORB_NEXT_WITNESS_COMMITMENT`.

**Wire label bytes:** keep `b"ak/a/w"` (Fiat--Shamir is positional;
label text is diagnostic only).

**Terminal fold (same schedule slot, different payload).**
At the terminal level this position absorbs cleartext `final_witness`
(packed $\mathbf{w}$), not a commitment.
The implementation PR should either:

1. **Preferred:** split diagnostic labels while preserving absorb order:
   - intermediate: `ABSORB_NEXT_WITNESS_COMMITMENT` (`b"ak/a/w"`),
   - terminal: `ABSORB_FINAL_WITNESS_CLEARTEXT` (new diagnostic tag, e.g.
     `b"ak/a/fwc"`, **only** if we accept a `logging-transcript` schedule
     update; sponge bytes unchanged either way), or
2. **Minimal:** one shared label `ABSORB_NEXT_LEVEL_WITNESS_BINDING` documented
   as “commitment $\mathbf{u}'$ or terminal cleartext, depending on level”.

This spec recommends **(1)** for clarity; **(2)** is acceptable if we want
a single grep target.

**Related proof field rename:** `next_w_commitment` → `next_witness_commitment`
(on `Stage2Proof` / level proofs), in the same cutover.

### Rename table (opening cluster)

Mechanical renames (longest compounds first):

| Current | Target |
|---------|--------|
| `terminal_w_hat_bytes_from_blocks` | `terminal_e_hat_bytes_from_blocks` |
| `absorb_terminal_w_hat` | `absorb_terminal_e_hat` |
| `ABSORB_TERMINAL_W_HAT` | `ABSORB_TERMINAL_E_HAT` (keep `b"ak/a/twh"`) |
| `w_hat_digit_offset`, `w_hat_digit_count`, … | `e_hat_digit_*` |
| `TerminalWitnessTranscriptParts::w_hat` | `::e_hat` |
| `RingRelationWitness::w_hat` | `::e_hat` |
| `w_folded_by_poly`, `w_folded_by_claim` | `e_folded_by_poly`, `e_folded_by_claim` |
| `w_folded` | `e_folded` |
| `pre_folded_by_poly`, `pre_folded_by_claim` | `pre_folded_e_by_poly`, `pre_folded_e_by_claim` |
| `w_hat_count` (locals) | `e_hat_count` |
| `num_w_vectors` | `num_e_vectors` |
| `offset_w` | `offset_e` |
| tracing `decompose_batched_w_hat` | `decompose_batched_e_hat` |
| tests `WHatDigit` | `EHatDigit` |

**Explicitly keep (full witness `w`):**

- `build_w_coeffs`, `w_ring_element_count`, `planned_w_ring_element_count`
- `ABSORB_SUMCHECK_W` wire position semantics only until renamed as above
- `ABSORB_TERMINAL_W_REMAINDER` (remainder of $\mathbf{w}$ outside logical $\hat e$)
- `ABSORB_SUMCHECK_W` in sumcheck module sense: N/A (different code path)
- `t_hat`, `r_hat`, `z_folded_rings`, `commit_w`, `logical_w`, `expected_w_len`

### Invariants

- **Behavior preserving.** No algorithm, numeric, or serialized proof layout change.
- **Proof bytes unchanged.** Witness segment order and packed digit semantics unchanged.
- **Transcript sponge unchanged.** Absorb order and payload bytes unchanged; only
  Rust identifier names and diagnostic label strings may change.
- **Full cutover.** No `w_hat` / `w_folded` aliases left in `crates/`.
- **Historical specs untouched.** Amend only this file and code/docs; do not rewrite
  `core-protocol-naming-cleanup.md`.

### Non-Goals

- Renaming `t_hat`, `r_hat`, or `z_folded_rings` → `z_hat`.
- Renaming sumcheck MLE symbols or `GruenSplitEq` / `split_eq`.
- Changing `ABSORB_PROVER_V` bytes or the opening commitment $v$.
- Changing `MRowLayout` `d_key` / `const D` matrix-vs-ring vocabulary (Tier-2 doc follow-up).
- Paper edits (tracked in `lattice-jolt`, separate commit).

## Evaluation

### Acceptance Criteria

- [ ] `rg '\bw_hat\b' crates/` and `rg '\bw_folded\b' crates/` return no matches.
- [ ] `rg 'ABSORB_SUMCHECK_W' crates/` return no matches (replaced per Transcript label section).
- [ ] `cargo nextest run --profile ci-non-zk` and `ci-all-features` green.
- [ ] `cargo test -p akita-pcs --features logging-transcript --test transcript_hardening` green.
- [ ] Byte-identical serialized proofs for at least one dense and one onehot fixture before/after.
- [ ] `specs/w-to-e-notation.md` status updated to `implemented` with PR link.

### Testing Strategy

- `akita-types` `terminal_witness` unit tests (layout + plane-major bytes).
- PCS transcript hardening + proptest (`logging-transcript`).
- `generated_tables` if `num_e_vectors` touches planner keys (regen if needed).

## Design

### Implementation order

1. `akita-types` (`terminal_witness.rs`, `schedule.rs`, `proof_size.rs`, `ring_relation.rs` layout, re-exports).
2. `akita-prover` + `akita-verifier` (same commit: `ring_relation*`, `ring_switch`, kernels).
3. `akita-transcript` labels + proof type `next_witness_commitment`.
4. `akita-planner` generated tables if `num_e_vectors` renames schedule keys.
5. Tests, examples, scripts (`tail_dimension_model.py`), comments.
6. Do **not** modify other `specs/*.md` files.

### Worktree

```bash
cd /Users/quang.dao/Documents/SNARKs/akita
git fetch layerzero main
git worktree add ../akita-w-to-e-notation -b quang/w-to-e-notation main
```

## Documentation

- Paper: `lattice-jolt` §§3--5 (merged separately).
- This spec is the authoritative post-cutover naming index.
- Optional: short module doc glossary in `akita-types/src/proof/mod.rs`.

## References

- `lattice-jolt/sections/akita/3_basic_akita.tex` (opening $\hat{\mathbf{e}}$).
- `specs/core-protocol-naming-cleanup.md` (historical; kept `w_hat`).
- `specs/terminal-fold-cutover.md` (terminal cleartext absorb at ex-`ABSORB_SUMCHECK_W` slot).
