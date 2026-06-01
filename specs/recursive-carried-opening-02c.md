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

## Non-Direction

- Do not introduce a local fallback evaluator or unchecked dense path.
- Do not widen verifier matrix bounds to accept a proof whose descriptor-bound
  params were selected for a smaller carried batch.
- Do not use a terminal-only shortcut as the acceptance test.
- Do not treat serializing an unused dummy commitment as full 02C completion.
