# Recursion and proof shape

> **Status:** stub. Part of the initial Akita Book scaffold.

How folds chain into a recursion, where the recursion terminates, and what the
serialized proof looks like as a result.

## Intermediate vs terminal levels

What distinguishes an intermediate fold level from the terminal level (witness
size reduction, the D-block treatment, the tight recursive witness layout), and
the Fiat-Shamir soundness lesson from the terminal-fold cutover.

**Sources to fold in**

- `crates/akita-types/src/layout/params.rs:26-33` (`RelationMatrixRowLayout`).
- `crates/akita-prover/src/protocol/flow/recursive.rs`, `flow/inputs.rs` (`batched_prove`).
- Paper §3.6 `sec:akita-minor-opts` (terminal fold, tight recursive witness layout), §3.8 ("Witness size reduction and termination").
- `specs/terminal-fold-cutover.md` (Fiat-Shamir soundness lesson).

## Proof anatomy

The serialized structure: `AkitaBatchedProof` / `AkitaBatchedRootProof` /
`AkitaLevelProof` / `AkitaProofStep`, the `BlockOrder` root-vs-recursive split,
and where singleton openings sit as the 1×1 special case.

**Sources to fold in**

- `crates/akita-types/src/proof/levels.rs:749-853`, `proof/batch.rs`, `proof/opening_batch.rs`.
- `crates/akita-types/src/sis/decomposition_digits.rs` (`decomp_depths`).
