# Spongefish main-parity report

Baseline: `origin/main` at `c0cb822f28b7b9efe85b1924b029d36e13cdf516`.
Candidate: `feat/spongefish`.

## Result

The native Spongefish proof path meets the requested parity target for the
primary fp128 one-hot nv=36 workload:

| Build | Proof bytes | Multi-verifier median | Single-verifier median |
| --- | ---: | ---: | ---: |
| Pinned main | 69,776 | 11.760 ms | 54.639 ms |
| Native Spongefish candidate | 69,756 | 11.672 ms | 54.554 ms |

These are five alternating warm runs on the same host with one polynomial,
Blake2b, release mode, 16 prover/multi-verifier threads, and one single-verifier
thread. The candidate is 20 bytes smaller; timing differences are below 1% and
favor the candidate at the median.

## Schedule parity

The regression came from changing the planner objective from main's packed
aggregate nonce bits to the sum of per-message native LEB128 maxima. The branch
also removed parent-bit-alignment-aware frontier dominance. Those changes
selected different valid schedules in 45 of 99 rows across 14 catalog families.

The implementation restores the main objective everywhere that participates in
selection:

- `PackedProofCost` carries payload bytes, semantic nonce bits, and query count;
- nonce bits are rounded once after aggregation;
- suffix dominance accounts for every possible parent bit remainder;
- optimized and unpruned searches prepend semantic nonce bits;
- runtime materialization recomputes the same packed estimate;
- the exact adapted-schedule fixture is restored.

`scripts/generate-schedule-artifacts.sh` completed all 16 families in 1,064
seconds. Its fresh outputs compare byte-for-byte equal to `origin/main`, and the
catalog-dependent regression test passes. No generated row was copied,
restored, hand-edited, or special-cased.

Native LEB128 bytes remain available as a diagnostic and parser bound but do
not influence schedule selection. The frozen packed-bit estimate is deliberate
accounting debt, not a verifier safety bound.

## Proof-size cause and fix

Exact instrumentation showed:

- actual native nonce bytes are 335, already smaller than main's 400-byte
  packed nonce stream;
- the native terminal response is also slightly smaller than main;
- the complete 1,224-byte nonterminal excess was 306 redundant native
  sumcheck length atoms of four bytes each.

Sumcheck coefficient counts are public fixed-shape data. Native proving now
pads lower-degree honest polynomials with high zero coefficients to the public
degree bound, and verification receives exactly that many canonical atoms.
Removing the redundant proof-controlled lengths preserves bounded allocation
and eliminates alternate parses. Challenge changes move Golomb-compressed tail
size slightly, yielding the final 69,756-byte proof.

Nonce encoding is stack-backed canonical unsigned LEB128. The profiler records
actual committed nonce bytes separately from the schedule-derived maximum.

## Verification performance

The measurable overhead was cumulative:

- 64-byte reduced field draws consumed more sponge output than necessary;
- context records were hashed before hundreds of fixed-grammar operations;
- fold public payloads used one absorption call per byte;
- terminal replay decoded `z`, converted and re-encoded all coefficients for a
  byte comparison, then decoded the same canonical payload again;
- nonce preview encoding allocated small vectors.

The final path uses exact canonical rejection sampling at the field's byte
width, a descriptor-bound positional grammar, batched fold-payload absorption,
one canonical terminal decode, and stack-backed nonce encoding. Diagnostic
context records remain available under `logging-transcript` but are not hashed
in production. The fixed public grammar and native protocol epoch provide the
domain boundary; the sole variable payload retains an explicit bounded length.

Exact sampling is uniform rather than statistically close. Keccak additionally
absorbs the accepted retry count so distinct squeeze histories cannot
reconverge after a later absorb.

## Preserved security properties

- Spongefish remains the sole production transcript and proof transport.
- Every proof message is absorbed before its dependent challenge.
- Proof-of-work and fold-response grinding retain bounded search, nonce ranges,
  predicates, plan ordering, and verifier checks.
- Fields and nonces use canonical decoding; sumcheck counts come only from the
  validated public shape.
- Terminal allocation is bounded before receipt, Golomb decoding is canonical,
  and direct norm/relation/trace checks remain mandatory.
- Verification requires complete grinding-plan consumption and Spongefish EOF.
- Malformed proof paths retain the verifier no-panic contract.

The authoritative design is [`spongefish-integration.md`](spongefish-integration.md).
