# Spongefish migration: main-branch parity report

Status: implementation handoff; no code changes are part of this report.

Comparison baseline: `origin/main` at `c0cb822f28b7b9efe85b1924b029d36e13cdf516`.
Reviewed branch: `feat/spongefish` at `8291491880d90d60513a052a30e07dac3ea4d520`.
Review date: 2026-09-19. These are the locally available refs; this review did
not fetch a newer remote head. Historical source references below mean files
at the pinned baseline, including files subsequently removed on the branch.

## Objective and implementation priorities

Complete the migration to Spongefish while preserving main's schedule tables,
proof-size behavior, and performance. Changing the proof/transcript format is
allowed. Grinding, bounded response search, supported fields and setup modes,
and verifier soundness must be preserved.

Schedule parity must result from algorithmic parity. Correct the planner and
generation code so running the generator produces the same schedules as main.
Do not copy, check out, restore, or hand-edit main's generated schedule files
as the fix. The pinned main artifacts are comparison references only; candidate
artifacts must be freshly produced by the corrected candidate code.

For the next implementation, this report takes precedence over the permissive
performance and proof-growth targets in `spongefish-integration.md` and the
current native nonce pricing requirements in `transcript-grinding.md`. Rewrite
those specifications around the final implemented design when doing the work;
do not append a history of successive designs.

Use main's planner objective and schedule-selection behavior even if it is an
imperfect estimate of a new wire format. Record estimation discrepancies for a
later fix. This permission concerns optimization/accounting, not verifier
acceptance: do not weaken nonce ranges, work predicates, response norms,
canonical decoding, allocation bounds, or security-query limits. An estimate
used to rank candidates can remain imperfect; a bound enforcing soundness or
safe decoding cannot silently become an underestimate.

## Findings

Main already uses Spongefish prover/verifier states and the same selected
Blake2b512/Keccak sponge implementations. The branch expands Spongefish's role
to authoritative proof serialization, receipt, absorption, and EOF. It also
changes nonce transport, framing, challenge sampling, and planner costs. Those
additional changes, rather than a replacement of the hash primitive, explain
the differences under investigation.

The directly identified schedule-selection changes are:

1. Main prices all nonce bits as one packed stream, rounded once to bytes.
   The branch prices each inline nonce separately. The reviewed branch uses
   a worst-case LEB128 length per nonce.
2. Main's dynamic-programming frontier compares suffixes across possible
   parent bit alignments. The branch removed that alignment-sensitive
   comparison and compares additive byte totals instead.
3. Both the optimized search and its unpruned reference, plus runtime schedule
   estimates, were changed to use the new cost. Regeneration selected different
   valid schedules and propagated changed producer profiles into grouped rows.

Copying main's `.aks` files is not an acceptable fix, and changing one final
byte-count formula is insufficient for reproducible parity. Restore main's cost composition,
dominance comparisons, and runtime estimator together, then require generation
to reproduce the pinned main artifacts exactly.

## Exact schedule inventory

There are 99 rows across 16 catalog files. Comparing row contents independently
of ordering finds 45 changed rows in 14 families. No family changed its row
count, and the top-level artifact metadata (excluding `rows`) is identical to
main in all 16 files. Several arrays also changed order.

The inventory groups rows by their ordered root `(num_vars, num_polynomials)`
tuples and compares full row multisets within each group. This preserves
duplicates: `fp128_onehot.aks` has 30 rows but only 29 distinct coarse tuples.
Implementers must use complete profile identities for lookup and byte-for-byte
artifact comparison for the final gate; do not collapse rows into a dictionary
keyed only by variable and polynomial counts.

| Catalog in `artifacts/schedules/` | Rows | Changed contents |
|---|---:|---:|
| `fp128_dense.aks` | 11 | 7 |
| `fp128_dense_bounded.aks` | 3 | 1 |
| `fp128_dense_multi_chunk.aks` | 1 | 1 |
| `fp128_dense_recursive.aks` | 5 | 1 |
| `fp128_onehot.aks` | 30 | 12 |
| `fp128_onehot_multi_chunk.aks` | 4 | 0 |
| `fp128_onehot_multi_chunk_w2r2.aks` | 3 | 3 |
| `fp128_onehot_multi_chunk_w4r2.aks` | 1 | 0 |
| `fp128_onehot_recursive.aks` | 3 | 2 |
| `fp128_onehot_recursive_multi_chunk_w8r2.aks` | 2 | 2 |
| `fp32_dense.aks` | 5 | 3 |
| `fp32_dense_recursive.aks` | 6 | 2 |
| `fp32_onehot.aks` | 8 | 2 |
| `fp64_dense.aks` | 8 | 6 |
| `fp64_dense_recursive.aks` | 5 | 1 |
| `fp64_onehot.aks` | 4 | 2 |
| Total | 99 | 45 |

In the ordinary fp128 one-hot family, changed coarse root keys are:

```text
[(14,1), (16,1)]
[(14,1), (20,2)]
[(44,1)]
[(16,1), (16,1), (34,2)]
[(30,4)]
[(14,2)]
[(32,1)]
[(16,2)]
[(14,2), (20,1)]
[(50,1)]
[(14,1), (14,1), (14,1), (20,4)]
[(32,4)]
```

The canonical single-polynomial nv=36 row is exactly equal to main. The
canonical nv=32 row is different at the reviewed head. Earlier nv=32 benchmark
claims about an unchanged schedule apply to an earlier branch revision, not
this head.

For nv=32, the first recursive fold illustrates the cascade:

| Selected property | Main | Reviewed branch |
|---|---:|---:|
| Challenge subring dimension | 256 | 64 |
| Sparse support `(count_pm1, count_pm2)` | (23, 0) | (31, 10) |
| Outer slice count | 1 | 8 |
| Inner matrix output rank | 1 | 2 |
| Open matrix input width | 17,716 | 4,429 |
| Output witness length | 3,366,336 | 3,729,152 |

Later input widths, block counts, and response caps consequently differ. Across
other rows the changes include ring dimensions, decomposition digits, opening
methods, relation/payload modes, fold counts, and terminal layouts. These are
actual selected parameter changes, not simply updated nonce metadata.

## Planner changes to reverse

| Location | Main behavior | Branch behavior / required action |
|---|---|---|
| `crates/akita-planner/src/schedule_params.rs` | `PackedProofCost` carries payload bytes, nonce bits, query count; rounds the combined bits once | Extra native byte accumulator and native prepend path. Restore main's composition and checked overflow behavior |
| Same file, frontier comparisons | Compares suffix costs at every relevant parent remainder via alignment ordering | Parent remainder ignored. Restore main's dominance rules, including ties |
| `crates/akita-planner/src/schedule_params/suffix_dp.rs` | Prepend semantic nonce bits | Uses native byte cost. Restore main objective |
| `crates/akita-planner/src/test/unpruned_search/candidate.rs` | Reference search uses packed-bit cost | Also migrated to native bytes. Restore independently checked reference behavior |
| `crates/akita-schedules/src/runtime.rs` | Expanded estimate and materialization use `ceil(total_nonce_bits / 8)` | Uses `GrindingPlan::native_nonce_bytes()` and checks cached native bytes. Restore main's planner-estimate contract |
| `crates/akita-types/src/transcript_grinding.rs` | Carries semantic nonce widths and query counts | Adds per-nonce wire bound to accumulated cost. Do not let this bound change the frozen objective |
| `crates/akita-types/src/transcript_grinding_plan.rs` | Candidate cost exposes semantic bits and queries | Adds native byte cost. Keep security derivation; remove its influence on selection |

The reviewed branch prices an emitted nonce with semantic width `b` as
`ceil(b / 7)` bytes. Main prices a complete plan as `ceil(sum(b) / 8)`.
These are different objectives even when honest counters usually fit one byte.
The former is a maximum over accepted counters, not the measured proof size.

The diff does not show a new candidate-generation algorithm or changes to the
SIS tables causing this drift. Altered selection is consistent with the changed
objective/frontier rules; exact regeneration after restoration is the required
causal confirmation. Do not compensate by editing individual parameter rows.

The branch also relaxed a regression fixture in
`crates/akita-planner/src/test/adapted_schedule.rs` from exactly one packing
choice to a nonempty domain after a selected schedule changed. Restore the
original fixture expectation when the corrected generator reproduces main's
tables, preserving the
original regression coverage.

## Proof-size changes and measurement limits

Historical nv=36 measurements from this task used `onehot_fp128`, one
polynomial, direct setup, Blake2b, release builds, 16 prover/multi-verifier
threads and one single-verifier thread. The schedule is equal across these
versions. No benchmark was rerun for this documentation-only review.

| Version | Actual proof bytes | Multi verify median | Single verify median |
|---|---:|---:|---:|
| Pinned main | 69,776 | 12.234 ms | 56.222 ms |
| Chunked native, four-byte nonces | 71,932 | 13.197 ms | 56.720 ms |
| Reviewed compact-native branch | 70,925 | 13.008 ms | 56.717 ms |

The reviewed proof is 1,149 bytes larger than main. Verification is about
6.3% slower multi-threaded and 0.9% slower single-threaded on these separate
five-run samples. These observations are evidence of an unresolved parity
gap, not a statistically controlled attribution to individual code changes.

Main's nonce stream is 400 bytes for this case. The intermediate native stream
was exactly 1,328 bytes (332 four-byte messages). The reviewed branch reports
664 bytes as a **maximum**, not an observed nonce byte count. Its profiler does
not currently measure actual compact nonce bytes. Therefore the earlier claim
that all remaining proof growth is terminal compression is not established.
Instrument actual nonce and payload byte contributions before assigning the
remaining delta.

Additional concrete differences include:

- Main serializes a packed nonce prefix and structured protocol payloads;
  the branch interleaves native nonce messages with protocol messages.
- Main's terminal response representation includes a length prefix in its
  structured serialization. The native terminal path emits bounded z bytes
  through a native `u32` length atom. Audit every length/count/tag and omitted
  field, rather than subtracting nonce sizes from total bytes and assuming
  everything else matches.
- The branch changes protocol/session framing, descriptor grinding revision,
  per-message context, nonce encoding, and challenge sampling. Different
  challenges change response coefficients and therefore Golomb lengths even
  with identical schedules and source inputs.

Keeping LEB128 does not guarantee main's size for all proofs. For a 32-bit
nonce it can require five bytes; main's packed representation contributes
exactly 32 bits. The honest first solution often compresses well,
but accepted proofs need not use the first solution. Do not claim worst-case
parity from a single honest fixture.

## Performance changes to investigate

`crates/akita-transcript/src/native.rs` adds a 96-byte public context before
logical message/challenge groups. Public records add hashing work but no proof
bytes. Main uses positional, length-framed absorption through its transcript
adapter. Rust method names and labels are not substitutes for cryptographic
framing; compact records or binding static context once require an explicit
argument that the public schedule determines the complete operation grammar.

Main requests `2 * F::NUM_BYTES` for a scalar and consumes whole 32-byte squeeze
blocks, truncating excess. The native branch requests 64 bytes per base-field
coordinate and reduces them, with a 192-bit statistical-distance budget and a
global query cap. For fp128, this doubles the requested scalar-challenge bytes.
Main's sampling behavior is a concrete parity candidate, but copying its byte
width alone does not establish the branch's existing sampling certificate.
Review the actual field conversion and security argument before replacing it;
document any accounting deficiency instead of falsely retaining the stronger
certificate. A demonstrated soundness failure cannot be deferred as sizing debt.

Native field receipt decodes and re-encodes individual fields; the scalar codec
allocates encoding vectors. `NativeNonce::encode` also allocates a small vector
on every encode, including candidate previews and verifier receipt. Stack-backed
encodings and batched validated receipt are concrete optimization candidates.
Preserve byte equivalence between preview, prover, and verifier.

The byte-payload path already batches receipt in 1024/64-byte chunks. The fold
root functions still loop over payload bytes, and the terminal verifier decodes
z, reconstructs a terminal response, and compares its canonical re-encoding.
Measure allocations and time in these paths. Remove duplicate work only with
equivalent canonicality and response checks. The Stage-2 arithmetic diff itself
is only a signature/import adjustment, so it is not evidence of a changed
arithmetic algorithm explaining the measured regression.

## Implementation sequence

1. Pin the main revision above for all comparisons. Correct the scheduling
   algorithm to match main's planner objective, alignment-sensitive frontier
   behavior, reference-search costs, runtime estimates, and deterministic
   selection/ordering. Generate all 16 catalog families from that corrected
   code into a fresh output directory, without using existing candidate or main
   artifacts as replacement outputs. Compare the generated files against pinned
   main byte for byte, including order and headers. Investigate any difference
   in the algorithm or generation inputs; do not patch the output files or add
   row-specific exceptions. Publish candidate artifacts only through the normal
   generator after this comparison passes. Keep transcript identity changes
   separate from schedule-generation inputs that must retain main's behavior.
2. Add measurement of actual proof-byte contributions and allocation/time
   profiles using existing harnesses. Separate measured wire bytes from the
   frozen planner estimate. Do not label a maximum as actual serialized bytes.
3. Design and validate the nonce transport needed for strict size parity.
   Use main's semantic bit widths and packed-stream accounting. If achieving
   its bit density requires packing across messages, resolve that within the
   Spongefish integration before declaring the migration complete.
4. Optimize context absorption, scalar encoding, payload receipt, and terminal
   canonicality work, prioritizing measured costs. Review main's challenge
   consumption as a candidate for parity, subject to the soundness constraints
   above. Preserve all grinding and norm-rejection functionality.
5. Update specifications and measurement tools to the final contract; record
   frozen estimate discrepancies as deferred work. Run repository preflight,
   CI feature graphs, native mutation tests on both backends, schedule drift
   tests, and representative supported protocol modes.
6. Benchmark pinned main and candidate with interleaved runs on the same idle
   machine. Use nv=36 one-hot as the primary case. Include other fields, dense,
   grouped, and recursive/offloaded cases to detect mode-specific regressions.
   Compare actual bytes over deterministic seeds and distributions over many
   seeds, plus prover and both verifier timings. Do not mix compile time or
   the previously observed host-wide slowdown into measured samples.

### Native transport design constraint

The pinned Spongefish byte API serializes exactly the bytes returned by
`Encoding<[u8]>` and its verifier absorbs the decoded message's re-encoding.
Its native cursor is byte-oriented. Main's packed nonce prefix is replayed at
later protocol positions, while inline nonces arrive where they are absorbed.
Cross-message bit packing therefore is not a drop-in codec replacement.

Evaluate a small, reviewed extension to Spongefish or a canonical packing
boundary that preserves the intended transcript schedule. Do not absorb
future nonces early without analyzing adaptive grinding and challenge
dependencies. Do not introduce a separate production transcript engine or
count only an external compressed wrapper while reporting the native stream
as smaller. A reversible outer encoding is a distinct architecture with decode
costs that must be explicitly assessed if chosen.

This is the unresolved engineering constraint for literal proof-size parity.
Neither restoring planner estimates nor keeping LEB128 solves it by itself.
Byte-identical proofs are not required, but the final format must meet the
requested size target. Changed random coins mean identical schedules alone
cannot guarantee identical compressed lengths on every seed.

## Deferred accounting work

Preserve main's selection objective during this migration. Record the gap
between its estimate and actual native wire bytes as optimization debt,
including per-message alignment, length tags, canonical variable payloads,
and the choice of maximum versus expected nonce size. Main's packed nonce
formula is appropriate for main's packed transport; it is not itself evidence
of a main-branch bug merely because inline transport costs more.

The branch's profiling helper currently asserts that actual bytes do not
exceed the planner estimate and allows a fixed 3,072-byte overcount. Reassess
those diagnostic assumptions when freezing main's cost model. Keep hard
proof-input bounds independently derived from validated public shapes; do not
use an intentionally approximate objective as a parser allocation bound.

Future accounting improvements may change optimal tables. They belong in a
separate, explicitly approved schedule update after this parity migration.

## Acceptance criteria

- A fresh run of the corrected candidate generator produces all 16 catalog
  files equal to pinned main byte for byte. Existing generated outputs cannot
  supply or mask the result. A second run produces no drift. nv=32 and nv=36
  canonical rows receive explicit checks. Copying/restoring main's artifacts,
  editing generated rows, or special-casing known catalog keys fails this gate.
- Main's planner comparisons and their regression tests are restored; no
  per-row hand tuning compensates for a changed objective.
- Actual proof size meets main's target; a smaller estimate does not count.
  Report the exact seed-by-seed results and any distributional limitation.
- Repeated paired benchmarks show no reproducible slowdown outside measured
  run-to-run variability, for proving and both verifier thread configurations.
  The existing ~6.3% multi-verifier gap does not satisfy this requirement.
- Spongefish remains authoritative for the migrated transcript and proof
  transport, with one accepted verification path and complete EOF/plan checks.
- Strict canonical parsing, bounded allocation, nonce ranges, grinding work,
  response rejection, field-sampling justification, and query accounting remain
  enforced. Known sizing debt is documented without weakening these contracts.
- Deleted or relaxed tests from the migration are audited for lost protocol
  coverage, including the changed planner fixture; passing remaining tests
  alone does not establish equivalence to main.

## Review method and scope

This report is based on read-only diffs of the pinned branches, source inspection
of planner/transcript/prover/verifier paths, JSON catalog comparisons, and the
earlier benchmark observations recorded in this task. It is a focused parity
review, not a complete cryptographic audit of all 182 changed files. No code,
catalog, dependency, or existing specification was changed for this report.
