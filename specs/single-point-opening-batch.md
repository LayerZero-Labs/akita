# Single-point opening batches

Status: landed in PR #186 (`refactor/collapse-ext-field-remove-multipoint`).

## Summary

Batched `prove` / `verify` take **one shared opening point** for every claim in
the call. Multipoint incidence (different evaluation points within one batch) is
removed.

Public API shape:

```text
(shared_point, Vec<CommittedOpenings>)   // verifier
(shared_point, Vec<CommittedPolynomials>) // prover
```

`OpeningBatch` in `crates/akita-types/src/proof/opening_batch.rs` is the
normalized batch descriptor (routing, gamma row, transcript binding).

## Supported call shapes

| Shape | Example | Folded prove/verify |
| --- | --- | --- |
| Singleton | 1 poly, 1 commitment, 1 point | Yes |
| Same-point multi-poly | 1 commitment bundling `N` polys at one point | Yes (primary production path) |
| Multi-commitment, same point | `N` commitments, 1 poly each, one point | Not yet (future PR) |
| Multipoint | Different points in one batch | **Removed** |

## Caller guidance

- To open polynomials at **different points**, run separate `prove` / `verify`
  calls (or re-commit under a new batch at a new point).
- For **multiple polynomials at one point**, use one `CommittedPolynomials` /
  `CommittedOpenings` entry whose `polynomials` / `openings` vectors list every
  slot (`batched_commit`, `OpeningBatch::new`).

## Wire / transcript impact

- `CallSection` and setup seed no longer carry `num_points` / `max_num_points`.
- Transcript batch-shape absorption uses `append_opening_batch_shape_to_transcript`
  (per-slot commitment group, poly index, natural arity, kind tag) instead of the
  old multipoint incidence encoding.
- Proofs and descriptors from before this cutover are not cross-verifiable.

## Out of scope (this cutover)

- Folded recursion with multiple commitment objects at one shared point
  (planner key, `w_len`, witness segment layout).
- Historical specs that still mention `incidence.rs` or multipoint e2e tests.
