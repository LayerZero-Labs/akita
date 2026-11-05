# Spec: structured Stage 2 weight oracle for the binary root

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-11-04 |
| Status | active |
| PR | |
| Supersedes | None |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

## Summary

The binary root proves its fused relation and range-image sumcheck (Stage 2)
over a padded digit table of `2^n` addresses. Before this change the prover
first materialized the complete public weight table `q` of `2^n` field
elements from the structured weight families
([`binary-source-opening.md`](binary-source-opening.md)), then handed it to
the ordinary dense Stage 2 prover. Both steps scan the full domain several
times, and much of that work is serial. This note records how Stage 2 reads
`q`, which families factor across which rounds, and a structured oracle that
serves the first rounds from the families' factored form and builds a dense
table only once the domain has shrunk by `2^8`. The witness stays dense. The
round messages, the transcript and every proof byte stay identical.

## Baseline profile

Measured on `f92e959ee` at `--log-carriers 22 --shape binary-and-prime-claims`, with the
#317 catalog cache, on the shared 64 GB machine. The breakdown uses temporary
spans that were not committed.

| Span | Wall ms |
| --- | ---: |
| `prove_complete_opening` | 1,669.6 |
| `binary_root_relation_materialize` | 223.3 |
| `binary_root_stage2` | 327.4 |
| Peak RSS (prove) | 4,191,240,192 bytes |

Inside `binary_root_relation_materialize`, the family visits take:
authentication 61 ms (about 614 ms of CPU: weighing setup columns dominates),
prime planes 72 ms (serial), direct A 30 ms (serial), image 18.5 ms, parity
rows 18 ms and explicit terms 0.06 ms. Allocating and zeroing the 512 MB table
takes about 30 ms.

Inside `binary_root_stage2` (330 ms), `begin_root_relation` takes 239 ms. It
copies the digits serially (43 ms), checks them serially (17 ms), packs them
(3 ms) and runs `RelationRangeImageProver::new` (129 ms). Of the 129 ms, 112 ms
is the serial conversion of the `2^n` compact digits into field elements when
the prover enters its lane phase in round 0. The 32 sumcheck rounds take
91 ms; round 0 alone takes 17.4 ms.

The 2^22 `five-prime-claims` shape spends 584 ms materializing and 592 ms in Stage 2.

## How Stage 2 reads the weight

`RelationRangeImageProver` binds address bits least significant first. Round
`i` pairs addresses `2j` and `2j+1` of the partially bound tables. Its message
is the sum of three parts.

1. The norm part `gamma * eq(stage1_point, x) * w(w+1)`, computed through
   `GruenSplitEq` with linear-term recovery. It reads only the witness.
2. The relation part `sum_x w(x) q(x)`. It is stored as `RoundMessage
   { at_one: sum w1 q1, quadratic: sum dw dq, cubic }`, and turned into a
   polynomial with `into_polynomial(previous_claim)`.
3. The sparse negative-binary terms (`AdditionalRelationTerms`), which keep
   their own addresses and weights.

Only part 2 reads `q`: through `(q(2j), q(2j+1))` pairs while computing the
message, and through the fold `q <- q0 + r (q1 - q0)` after each challenge. At
the end, `expected_final_claim` reads the fully bound weight. The binary root
called `from_dense_relation` with zero coefficient bits, so the prover entered
its `Lane` phase immediately: it converted the whole witness to field elements
and handed the dense `q` to `LaneProduct`.

Field arithmetic is exact. Any computation that produces the same field
elements for every `(q0, q1)` pair and every fold produces the same round
polynomials, and hence the same bytes. Only how `q` is stored changes.

## Which families factor, and over which rounds

A root segment is a list of lanes. Lane `(element, digit)` holds `stride`
consecutive addresses starting at `start + (element * digits + digit) * stride`.
Its first `degree` addresses are natural coefficients; the rest are padding.
Every stride is a power of two, at least 256 (648 pads to 1024 and 162 to 256),
and every segment start is a multiple of its stride. The low `log2(stride)`
address bits are therefore the coefficient offset, and the high bits select
the lane.

**Recomposed families** give every digit lane of an element the same profile,
scaled by the digit's power of eight: `q(lane(e, d), t) = 8^d g_e(t)`.

| Family | Segment | Digits | Profile `g_e` |
| --- | --- | ---: | --- |
| Direct A | response (0) | 4 | `h_e = sum_i adj(rho_i, A_ij)` |
| Image | image (1) | 8 | `-sum_k c_k * shift_k(extended rho)` |
| Prime response rows | response (0) | 4 | `±relation * row_weight * alpha^c` at `4c + row % 4` |
| Prime plane columns | prime plane | 43 | `reading * column_weight * omega_c - relation * K_c` |

**Per-address families** have no cross-lane structure.

- Parity rows weigh only the digit-0 lane of a response element.
- Digit authentication (B over the image segment, D over the opening
  segments) runs 64 compact digits per setup column straight through the
  natural coefficients, across lane boundaries. Each address has its own weight.
- Explicit terms (the word-claim terminal, the ordinary mixed group) are
  address lists.

**Binding order.** LSB-first binding agrees with the recomposed structure. After
`k <= log2(stride)` rounds, the bound table is still `8^d g~_e(r_1..r_k, t')`
on each lane, with the folded profile shared by every digit. With a block
width of `2^T`, `T = min(n, 8)` (the smallest stride), no pair in the first
`T` rounds crosses a block, and every block sits inside one lane. The order
works against the families in two places only.

1. The sharing across digits ends once the offset bits are bound. On
   stride-1024 lanes it could survive two more rounds, but by then the table
   has `2^(n-8)` entries and is cheap to keep dense.
2. Authentication columns have no low-bit factorization at all, so the oracle
   stores them per address. Their cost is set by their support, not by the
   binding order.

Neither forces a different order, so the design keeps the existing transcript
order and is byte-identical by construction.

## Design

### Block profiles (`akita-types`)

`RootWeightProfiles<E>` in `native_root_relation/profiles.rs` stores the
weight as one record per aligned block of `2^T` addresses (`2^(n-T)` records)
and a pool of rows. A pool row is `2^(T - bound)` field elements and starts
`2^T` wide; row 0 is all zeros.

```text
q(block s, offset o) = scale_s * pool[shared_s][o] + pool[own_s][o]
```

- `shared_s` points at the profile row of a recomposed element. The `stride /
  2^T` blocks of lane `(e, d)` point at consecutive profile rows of element
  `e`, with `scale_s = 8^d`. Families on the same element accumulate into the
  same rows (direct A and the prime response rows share the response
  elements).
- `own_s` points at a private row that collects per-address weights:
  parity, authentication and explicit terms. It is allocated when a block is
  first touched.

`NativeRootWeights::profiles(bits)` builds the representation. Recomposed
families emit `(element, offset, value)` once per element instead of once per
digit lane. The per-address families and the explicit terms reuse their
existing `visit`. Every family keeps one construction of its weights:
`evaluate` (verifier), `visit` (materialization and tests) and the new
recomposed emission all read the same profile code.

Binding is linear and block-local. `bind(r)` folds every pool row in place
(one `fold_evals_in_place` over the whole pool, because rows are contiguous
and of even width) and keeps the block records. `pair(left)` reads the two
weights of one pair. After `T` rounds each row has one entry, and
`take_dense()` returns the `2^(n-T)` folded weights
`scale_s * pool[shared_s] + pool[own_s]`. These are exactly the dense table
folded `T` times.

### Stage 2 seam (`akita-cpu-backend`)

The ordinary Stage 2 prover gains a weight oracle trait next to
`DenseRelationWeights`:

```rust
pub(crate) trait StructuredRelationWeights<E: Field>: Send + Sync {
    fn len(&self) -> usize;
    fn pair(&self, left: usize) -> (E, E);
    fn bind(&mut self, challenge: E);
    fn take_dense(&mut self) -> Vec<E>;
}
```

`RelationWeightOracle::Structured` and `CoefficientRelation::Structured` carry
it. Coefficient rounds call the existing generic compact and folded kernels
with `|left| oracle.pair(left)`, the same kernels the reduced dense path calls
with `|left| (dense[left], dense[left + 1])`. Ingest binds the oracle where the
dense path binds its table. At the lane transition, `take_dense` supplies the
lane weights with scale one. `CpuStage2SessionHandle::from_structured_relation`
shares its constructor with `from_dense_relation` and takes the coefficient bit
count. The dense path keeps zero coefficient bits and is unchanged for every
ordinary caller.

### Binary root wiring

- The dev-only `OpaqueStage2Kernel::begin_root_relation` takes the
  `NativeRootWeights` instead of a dense vector.
- The CPU backend builds the profiles with `T = min(n, 8)` and starts Stage 2
  with `T` coefficient bits.
  - The first `T` rounds run on the compact witness and the profiles.
  - Round 0 reads compact digits.
  - Ingest folds the witness through the pair lookup table in parallel.
  - The serial `2^n` conversion of digits into field elements disappears.
- `binary_root_relation_materialize` is removed from proving.
  - The fault-injection test backend materializes, perturbs and rebuilds an
    explicit oracle.
  - `CompiledNativeRoot` (the ordinary native root) keeps the dense path.
- The root witness reuses the packed digit storage: an aligned prefix is
  cloned, the digits are repacked only when the domain needs padding, and the
  range check reads the cached digit bounds.

## Cost model

Let `N = 2^n`, `B = 2^T`, `S` the number of shared profile entries and `O` the
number of own entries.

| Step | Dense today | Structured |
| --- | --- | --- |
| Weight build | zero `N`; visit `sum_family digits * naturals` | zero `S + O`; visit `naturals / digits` for recomposed families, `naturals` for per-address families |
| Witness setup | serial copy, check and E conversion of `N` | packed clone (repack only when padding); cached bounds; no conversion |
| Rounds `0..T` | `N` pairs and folds of `N` dense weights | `N` pairs, folds of `S + O` pool entries, compact round 0 |
| Rounds `T..n` | dense on `N / B` | dense on `N / B` |
| Peak weight memory | `N` (q) plus `N` (E witness at round 0) | `S + O` plus `N / 2` (folded witness after round 0) |

At 2^22 `binary-and-prime-claims` (`n = 25`, block 256):

- Shared entries: 4,096 response elements × 1,024, plus 768 image elements ×
  1,024, plus 256 prime elements × 256. That is about 5.05M.
- Own entries: B authentication on the image lanes (3 touched blocks per lane:
  4.72M), D authentication (2.88M), and parity on the digit-0 response lanes
  (3.15M). That is about 10.7M.
- The pool is therefore about 15.8M entries (253 MB), against 33.5M (512 MB)
  for the dense table.

## Expected gains

These are estimates from the cost model and the profile above; measurements
follow below. All four shapes have `n = 25` at 2^22 except `five-prime-claims` (`n = 26`).

| Shape | Weights today → structured (ms) | Stage 2 today → structured (ms) | Complete proving |
| --- | --- | --- | --- |
| `binary-claim` | ~150 → ~70 | ~320 → ~80 | −300 to −350 ms |
| `binary-and-prime-claims` | 223 → ~95 | 327 → ~80 | −350 to −400 ms (1.67 s → ~1.3 s) |
| `binary-and-prime-claims-plus-ordinary` | ~260 → ~130 | ~330 → ~90 | −350 to −400 ms |
| `five-prime-claims` | 584 → ~170 | 592 → ~160 | −800 to −850 ms |

- The prime-plane gain is largest because those lanes have 43 digits.
- Authentication remains the floor of the weight build. Its setup columns have
  no shared structure, and weighing them costs about 600 ms of CPU at 2^22.
- The 0.4–0.8 s target for complete proving is not reachable from these two
  spans alone. The remaining spans are owned by other work: projections,
  relation compile, the D commitment and the root commitment. At 2^22
  `binary-and-prime-claims` they sum to about 1.1 s.

## Acceptance

- Setup, commitment and proof bytes and the transcript are unchanged.
  `scripts/binary-source-full-size-digests.tsv` and `compatibility.rs` are
  untouched; the full-size pins match.
- The ordinary root bytes (`ordinary_grouped_root_bytes`) are unchanged.
- A differential test runs Stage 2 on the dense table and on the profiles for
  fixtures with several owners, prime planes and the mixed ordinary group. It
  asserts identical round polynomials and final claims.
- Peak RSS does not grow by more than 5%.

## Measurements

### On `dev` with the B/D setup offload

The base is `dev` at `ff2297aae`, which moves the B and D setup columns into
the typed Stage 3 prefix. The baseline executable is `ff2297aae` (SHA-256
`bd317aab58ffe3901163b1d0c68a48c9776026570e8f36ccd03ab2c04a6918ac`); the
structured executable is `537beafa2` (SHA-256
`88bd6f0c7419bcf3c15b13fa863effe080bf8f9fc418f06619acf36d6e44bd53`), the merge
of that base into this work, whose code is that of this note's final commit.
Both were built with
`--features parallel,binary-source-measurements,transcript-blake2b,logging-transcript`.
The procedure is the one below: three interleaved runs per configuration in
isolated processes, one catalog in `BINARY_SOURCE_CATALOG_CACHE`, one discarded
warm-up per shape, and each run started only once no other full-size run or
build was active. All 35 runs, warm-ups included, reproduced their digest pins
and verified. The figures are medians in milliseconds. The one-minute load
ranged over 4.91–15.77, higher than in the earlier campaign; differences of
tens of milliseconds are noise. Logs are
`$CARGO_TARGET_DIR/logs/m-{base,new}-<log>-<shape>-<run>.txt`.

| N | Shape | Weights + Stage 2, base → structured | › profile build | Complete proving, base → structured | Prover peak RSS bytes, base → structured |
| --- | --- | --- | ---: | --- | --- |
| 2^22 | `binary-claim` | 421.6 → 178.5 | 97.2 | 1,583.4 → 1,494.0 (−89) | 4,445,175,808 → 2,996,060,160 (−33%) |
| 2^22 | `binary-and-prime-claims` | 588.5 → 250.9 | 152.6 | 2,391.3 → 2,105.5 (−286) | 5,238,603,776 → 5,181,784,064 (−1%) |
| 2^22 | `binary-and-prime-claims-plus-ordinary` | 583.3 → 271.3 | 168.8 | 2,560.5 → 2,226.9 (−334) | 5,584,814,080 → 5,495,062,528 (−2%) |
| 2^22 | `five-prime-claims` | 1,167.8 → 395.0 | 279.2 | 3,471.4 → 2,818.0 (−653) | 9,370,451,968 → 8,814,641,152 (−6%) |
| 2^23 | `binary-and-prime-claims` | 1,027.1 → 345.7 | 208.6 | 3,513.2 → 2,900.1 (−613) | 8,610,463,744 → 5,448,220,672 (−37%) |

The live-heap peaks of the counting allocator, median GB, show where the
process peak now sits:

| N | Shape | Stage 2 peak, base → structured | Stage 3 peak, both |
| --- | --- | --- | ---: |
| 2^22 | `binary-claim` | 2.86 → 1.52 | 1.77 |
| 2^22 | `binary-and-prime-claims` | 3.58 → 2.28 | 3.30 |
| 2^22 | `binary-and-prime-claims-plus-ordinary` | 3.66 → 2.71 | 3.37 |
| 2^22 | `five-prime-claims` | 6.56 → 3.81 | 6.00 |
| 2^23 | `binary-and-prime-claims` | 5.67 → 2.94 | 3.50 |

What changes on this base:

- **Stage 2 is unaffected by the offload.** The prover's Stage 2 relation
  still carries the B and D authentication families; only the verifier drops
  them and reads their contribution through the typed prefix. Weights plus
  Stage 2 fall by 53–66%, as before.
- **Memory.** Stage 3 now builds a dense weight over the typed A/B/D prefix.
  For the 2^22 shapes with prime planes it sets the process peak, so the
  structured Stage 2 no longer lowers peak RSS there. At 2^22 `binary-claim` and 2^23
  `binary-and-prime-claims`, Stage 3 stays below the old Stage 2 peak and peak RSS falls by
  a third. The 5% ceiling holds everywhere.
- **Stage 3 timing.** Stage 3 runs the same code in both executables, but took
  40–90 ms longer in the structured runs for every configuration except
  `binary-and-prime-claims-plus-ordinary`. This has not been investigated. One possible cause
  is page faults: the baseline's larger Stage 2 leaves freed pages that Stage 3
  can reuse. It accounts for part of the gap between the Stage 2 saving and
  the complete-proving saving, most visibly at `binary-claim`, together with load noise.

### Before the B/D setup offload

The base was `f92e959ee`, before the B/D setup offload. The baseline
executable is `f92e959ee`; the structured executable is `2e59907c9`, whose
code was this work's code at the time. Both were built
with `--features parallel,binary-source-measurements,transcript-blake2b,logging-transcript`.
Each configuration proved three times in isolated processes, interleaving
the two executables, with `BINARY_SOURCE_CATALOG_CACHE` set to one catalog.
A discarded warm-up run filled the catalog for each shape first. All 33
measured runs reproduced their digest pins and verified. The figures are
medians in milliseconds. The one-minute load ranged over 7.37–12.69: the host
was shared, so differences of a few tens of milliseconds are noise. Logs are
`$CARGO_TARGET_DIR/logs/m-{base,new2}-<log>-<shape>-<run>.txt`.

| N | Shape | Weights + Stage 2, base → structured | › profile build | Complete proving, base → structured | Prover peak RSS bytes, base → structured |
| --- | --- | --- | ---: | --- | --- |
| 2^22 | `binary-claim` | 451.7 → 175.0 | 97.3 | 1,524.1 → 1,273.1 (−251) | 4,194,680,832 → 2,581,151,744 (−38%) |
| 2^22 | `binary-and-prime-claims` | 555.6 → 227.3 | 140.4 | 1,746.6 → 1,416.0 (−331) | 4,195,729,408 → 2,646,753,280 (−37%) |
| 2^22 | `binary-and-prime-claims-plus-ordinary` | 572.5 → 270.9 | 169.2 | 2,041.4 → 1,753.5 (−288) | 4,753,358,848 → 3,263,168,512 (−31%) |
| 2^22 | `five-prime-claims` | 1,194.2 → 415.7 | 287.2 | 2,686.3 → 1,866.9 (−819) | 6,945,423,360 → 2,792,046,592 (−60%) |
| 2^23 | `binary-and-prime-claims` | 993.5 → 334.3 | 200.3 | 3,148.9 → 2,497.3 (−652) | 8,059,813,888 → 4,623,138,816 (−43%) |

"Weights + Stage 2" is `binary_root_relation_materialize` plus
`binary_root_stage2` for the baseline. The structured executable has no
materialization span; its `binary_root_stage2` includes the profile build
(`binary_root_weight_profiles`).

What these measurements showed:

- **Stage 2 rounds.** Excluding the profile build, the root's Stage 2
  took about 80–130 ms at 2^22, against 310–590 ms. The baseline's root
  `RelationRangeImageProver::new` took about 123 ms at 2^22 `binary-and-prime-claims`
  building its dense field-element state; the structured one takes
  under 0.1 ms, because round 0 reads the packed digits.
- **Profile build.** The profile build is the larger remaining part, and the
  authentication family dominates it: 52 ms (`binary-claim`), 76 ms (`binary-and-prime-claims`) and
  170 ms (`five-prime-claims`) at 2^22. Its setup columns have no shared structure, so
  its own rows are as dense as the table over the authenticated region. The
  build is bound by weighing the setup entries, which runs in parallel
  batches; streaming into the profiles is not the bottleneck. A synthetic
  microbenchmark of the row pool's growth showed that reserving it up front
  saves nothing.
- **Memory.** Peak RSS falls by 31–60%, so the 5% ceiling holds with room.
  The dense table and the dense field-element witness of round 0 no longer
  exist together.
- **Against the estimates.** Complete proving gains match the estimates for
  `binary-and-prime-claims` and `five-prime-claims`. The weight build is slower than estimated:
  authentication costs what it did in the dense build, and the cost model
  undercounted it.
- **The 0.4–0.8 s target.** Complete proving at 2^22 `binary-and-prime-claims` was about
  1.42 s on that base (about 2.11 s on the current base, under heavier load
  and with B and D in Stage 3). Even with no weight build and no Stage 2 at all, it would be about
  1.19 s. The rest belongs to the projections, relation compile, the D
  commitment, the root commitment, Stage 3 and the ordinary suffix folds.

Further gains in this span need cheaper authentication weights, for example
reusing setup entries the prover has already derived for the commitment.
That sits in relation compile, which this change does not touch.
Fusing the witness fold into the next coefficient round, as the lane rounds
do, would save at most a few tens of milliseconds.
