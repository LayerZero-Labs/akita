# Spec: Recompute the last terminal block

| Field         | Value                          |
|---------------|--------------------------------|
| Author(s)     | Quang Dao                      |
| Created       | 2026-10-29                     |
| Status        | active                         |
| PR            |                                |
| Supersedes    |                                |
| Superseded-by |                                |
| Book-chapter  | how/verifying/terminal.md      |

## Summary

The terminal response sends the last live block of the opening partials `e`
and of the inner images `t`, although the two terminal ring relations
determine both once the rest of the response is known. This change omits
those blocks and sends a 32-byte digest of each in their place. The verifier
solves each relation for the omitted block, re-checks the relation, and
compares the digest. Proofs shrink by roughly 4 KB, about 6%, at the cost of
two digests per terminal group.

The change first landed on `dev` behind the `dev-protocol` feature in
[#177](https://github.com/LayerZero-Labs/akita/pull/177). This spec makes it
the current protocol on `main`.

## Intent

### Goal

Make every terminal response omit the last live block of `e` and `t`, bind
each omitted block by a digest at the position where the block was absorbed,
and have the verifier recover and check both blocks.

### Invariants

- **Soundness.** Relative to the protocol that sends every block, the
  soundness error grows by at most `(C(Q,2) + 2Q) / 2^256` for `Q` digest
  queries, in the classical random-oracle model. The Book proves this in
  `how/verifying/terminal.md` ("Why omitting the block is sound").
- **Transcript order.** The `t` digest is absorbed by the predecessor fold
  directly after the sent `t` fields, before its ring-switch and sumcheck
  challenges. The `e` digest is absorbed directly after the sent `e` fields,
  before the terminal fold-response nonce and the sparse challenges.
  `transcript_hardening` checks that prover and verifier emit the same event
  sequence and that mutating either digest rejects.
- **Verifier acceptance.** The verifier inverts the last fold challenge only to
  propose a candidate block. It accepts only if the sparse product of the
  candidate re-satisfies the relation and its digest matches. A non-unit
  challenge is rejected. Unit tests in
  `crates/akita-verifier/src/terminal/direct.rs` cover both.
- **No panic.** Recovery follows the
  [verifier no-panic contract](../docs/verifier-contract.md).
- **Sizing.** `terminal_sent_blocks` and `TERMINAL_BLOCK_DIGEST_BYTES` in
  `akita-params` are the single source for the prover, the verifier, the
  schedule audit, and proof-size pricing. `akita-pcs` checks real proof bytes
  against the byte model.

### Non-Goals

- Choosing the digest hash. `field_digest` uses a fresh instance of the
  transcript sponge, so it follows the transcript backend. The transcript
  stack starting at [#253](https://github.com/LayerZero-Labs/akita/pull/253)
  owns any change to that hash.
- A claim against a prover with quantum access to the digest oracle.
- Omitting more than one block per segment.

## Evaluation

### Acceptance Criteria

- [ ] Prover and verifier never send or expect the last live `e` or `t`
      block; no feature flag selects the old layout.
- [ ] `artifacts/schedules/` and `artifacts/schedule-catalog.tsv` are
      regenerated, and no catalog row grows.
- [ ] Every CI suite passes, including both transcript backends.
- [ ] `transcript_hardening` mutates each digest on its own and the verifier
      rejects.

### Testing Strategy

The existing end-to-end, soundness, and transcript-hardening suites exercise
the new layout on every catalog shape. The CI transcript matrix runs them
under Blake2b and Keccak. New coverage: the last-block recovery unit tests and
the cyclotomic inverse tests from #177, and the per-digest mutation assertion
in `crates/akita-pcs/tests/transcript_hardening.rs`.

### Performance

Proof size falls on every catalog row; see the PR for the catalog
comparison. The verifier adds one ring inversion and `1 + n_A` dense ring
products at the terminal. The prover hashes two blocks.

## Design

### Architecture

- `akita-transcript`: `field_digest`, `send_field_digest`,
  `receive_field_digest`.
- `akita-algebra`: `CyclotomicRing::inverse`.
- `akita-params`: `terminal_sent_blocks`, `TERMINAL_BLOCK_DIGEST_BYTES`, and
  the terminal byte bound.
- `akita-types`: the `NextWitnessInnerStateDigest` and `TerminalEDigest`
  proof sites.
- `akita-prover`: sends both digests.
- `akita-verifier`: receives both digests and recovers the blocks in
  `terminal/direct.rs`.

### Alternatives Considered

- **Keep the full response.** It costs `(1 + n_A) D` field elements per
  terminal group for no soundness benefit.
- **Keep both layouts behind a flag.** `dev` did this while the change was
  under review. On `main` it would thread a mode through the prover, verifier,
  and planner, and it would need a second schedule set.
- **Omit the block with no digest.** That is unsound: recovering the block
  makes both relations hold by construction.

## Documentation

`book/src/how/verifying/terminal.md` owns the protocol and its soundness
argument. `book/src/how/transcript.md` and `book/src/how/recursion.md` state
where the digests sit.

## References

- [#177](https://github.com/LayerZero-Labs/akita/pull/177): the `dev`
  implementation behind `dev-protocol`.
