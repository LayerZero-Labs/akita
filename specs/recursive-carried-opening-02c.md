# Recursive Carried-Opening 02C Direction

## Goal

Complete slice 02C by making the recursive boundary a true carried-opening
batch. The ordinary folded-witness opening remains claim 0. A setup-prefix
opening may be appended as another carried claim only when there is a subsequent
recursive fold that can consume it.

This is not just proof metadata. If the setup-prefix claim is checked by the
recursive fold relation, it changes the recursive incidence shape and therefore
the recursive level layout that prover and verifier must both bind.

## Rules

- Do not patch an extra setup-prefix claim into a recursive suffix that was
  already planned for singleton carried openings.
- Do not enable a setup-prefix carry at a terminal boundary in the first cut.
  `specs/setup-layout-repack.md` explicitly disables setup offloading when no
  subsequent recursive fold consumes the carried batch.
- The carried batch must have a single root-style incidence summary at the
  recursive boundary.
- The carried batch must use one common padded power-of-two domain. Smaller
  natural domains are zero-padded into that common domain.
- Source commitments are carried once. Claims reference them by `source_idx`.
  Source 0 is the ordinary folded-witness commitment.
- Prover state may hold source witnesses and hints. Proof-visible state must
  hold only source commitments and claim metadata.
- The transcript must bind source commitments once, then claim metadata,
  including `source_idx`.

## Implementation Direction

1. Add a carried-incidence helper on recursive prover/verifier state. It should
   produce the recursive batch incidence from the current carried claims instead
   of assuming the singleton witness claim.
2. Select recursive fold parameters from that carried incidence before proving a
   recursive level. The selector must see `num_points`, `num_claims`, and the
   current witness length implied by the carried batch.
3. Bind the effective recursive carried incidence/schedule in the transcript
   descriptor. Prover and verifier must derive the same shape before replay.
4. Wire setup-prefix carry insertion before recursive suffix selection, not as
   an after-the-fact mutation of root raw output.
5. Add the exit test only after the planned carried shape is canonical:
   witness-plus-dummy-setup carried batch verifies through at least one
   non-terminal recursive fold, and singleton recursive proofs still verify.

## Implementation Layer (rebased onto current main)

This slice is reimplemented on top of current `main`, which refactored the exact
surface 02C touches. Build on these structures rather than the pre-refactor flow:

- Recursive prover state is `RecursiveProverState` (`akita-prover/src/protocol/flow.rs`),
  carrying a single `opening` at `sumcheck_challenges`. Generalize this into the
  carried-opening batch (claim 0 = folded witness) here, not in a revived monolith.
- The recursive fold is driven by `prove_recursive_suffix` → `prepare_fold_data` →
  `PreparedRecursiveFold` (`flow/recursive.rs`). The carried-incidence helper and
  batch construction must live in `prepare_fold_data`, and fold-param selection must
  use `next_level_params` (post-#170), not a `commit_w_for_next` closure.
- The verifier recursive replay generalizes the singleton
  `current_state.opening{,_point,_mask}`/`basis` (+ `zk_eor_final`) into the same
  carried batch the prover binds.
- Setup-prefix carry rides on the existing `akita-types::proof::setup_prefix` module
  (`SetupPrefixProverRegistry`, `setup_prefix_level_params`, `select_setup_prefix_slot`,
  `SETUP_OFFLOAD_D_SETUP`) and the stage-3 setup-product sumcheck (`SetupSumcheckProof`,
  #147). Do not add a parallel setup-prefix path.
- Schedule selection adds `find_recursive_carried_suffix_schedule` in the planner,
  coexisting with #157 tiered sizing: fold-digit depth in `schedule.rs` must follow the
  carried incidence (`num_claims`) while preserving the tiered `u_concat` term.
- Tests live in `akita-pcs` (`crates/akita-pcs/src/scheme/tests/`); `akita-scheme` no
  longer exists (#152).

## Non-Direction

- Do not introduce a local fallback evaluator or unchecked dense path.
- Do not widen verifier matrix bounds to accept a proof whose descriptor-bound
  params were selected for a smaller carried batch.
- Do not use a terminal-only shortcut as the acceptance test.
- Do not treat serializing an unused dummy commitment as full 02C completion.
- Do not duplicate setup-prefix commitment or stage-3 setup-product machinery that
  #138/#147 already provide on main.
- Do not reintroduce the pre-#171 monolithic `prove_recursive_fold_with_params` or the
  pre-#170 commit closures.
