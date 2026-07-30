# Spec: Commitment compression cutover

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-05-01 |
| Status | active |
| PR | |
| Supersedes | |
| Superseded-by | |
| Book-chapter | |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in BCP 14
(RFC 2119 and RFC 8174) when, and only when, they appear in all capitals.

## Decision and scope

Akita will productionize commitment compression in two checkpoints.

The **parallel-safe implementation checkpoint** makes the existing diagnostic
path use a checked, source-independent plan, compact negative-binary witnesses,
and a reusable bounded-memory executor. It still computes compression only as a
shadow operation. Raw B- and D-side images remain the public protocol payloads,
and the diagnostic discards its terminal payloads and witnesses.

The later **atomic protocol cutover** replaces raw public images, binds
compression plans and witnesses in the relation and transcript, and updates the
wire, verifier, planner, setup contribution, and recursive flows together.
That work MUST wait for the heterogeneous group-source contracts described in
`specs/heterogeneous-group-source-contracts.md` to merge and stabilize.

There is no compatibility mode. After the atomic cutover, Akita will have one
compressed protocol, not raw and compressed siblings.

This specification has three kinds of statements:

- **Observed current behavior** describes base
  `6f7fde8658bc77fde8c5d1b0fda732068f11e6e7`.
- **Checkpoint requirements** govern the parallel-safe implementation that may
  land before the group-source cutover.
- **Cutover requirements** are deliberately deferred and do not authorize an
  early protocol change.

## Audited baseline and provenance

### Observed current behavior

At the audited base:

1. `compression-diagnostics` invokes a standalone ladder after live B and D
   images have been computed.
2. The diagnostic copies every complete source, decomposes it into bit-major
   `i8` negative-binary digits, batches equal `(field, d, width, rank)` shapes,
   executes a real negacyclic NTT matrix-vector product, reports metrics, and
   discards the output.
3. The ladder accepts nonempty complete images through 16 KiB. Inputs through
   8 KiB use two maps and inputs over 8 KiB through 16 KiB use three.
4. Every map is rank one. The terminal is 128 bytes: `d = 8`, `16`, or `32`
   over q128, q64, or q32 respectively.
5. The compression cache is keyed separately from the general NTT cache and
   stores an exact negacyclic-only setup prefix. First construction is
   single-flight.
6. Raw B/D images remain in the proof and transcript. Compression has no wire,
   verifier, relation, setup-contribution, proof-size, or schedule effect.

The diagnostic currently owns plan geometry in `akita-planner` and execution
orchestration in `akita-prover::diagnostics`. That split is acceptable for an
experiment but not for a persistent witness: it duplicates derivation across
planning, digitization, and execution and materializes an `i8` digit buffer for
every right-hand side in an equal-shape batch.

### Historical design record

The earliest `specs/compressed-commitments.md` design proposed an exactly
two-layer, 160-byte payload with small terminal dimensions and rank five. Later
commits changed the target to one rank-one 128-byte ring element and allowed
two or three maps. The historical branch
`quangvdao/feat/compressed-commitments`, including checkpoint
`d557356d` and tip `76428c4a`, contains useful arithmetic, relation, planner,
wire, and tamper-test ideas but does not build against the audited base.

That branch is not an implementation base. Its stage order, relation layout,
proof containers, schedule types, source ownership, terminal handling, and
multi-group assumptions predate:

- the folded-only, quotient-free terminal protocol from PR #311;
- the fully restacked PR #336/#335/#322/#334 state;
- mixed per-matrix ring dimensions and current prepared-setup behavior; and
- the pending group-local source, descriptor, schedule-key, and claims
  ownership cutover.

The 160-byte design is therefore historical, not an alternative checkpoint
encoding. The live 128-byte ladder is smaller, uses the current native
compression dispatch, is already backed by exact checked-in SIS cells, and is
exercised by real diagnostic kernels. Changing its terminal target or map
count requires new estimator and runtime evidence and a spec revision.

## Terminology

A **source image** is one complete flat field-coefficient image produced by a
current B or D matrix.

A **compression map plan** is a checked, source-independent description of one
negative-binary recommitment:

```text
y_(i+1) = C_i * negbin(y_i).
```

A **compression chain plan** contains one source coefficient count and an
ordered list of checked map plans ending at the terminal target.

A **stage witness** is the negative-binary digit vector for one map input.

A **terminal payload** is the final flat field-coefficient image of one
validated chain and one source identity. The checkpoint payload carries no
group ordering or protocol role.

An **equal-shape batch** is an execution batch whose maps have identical field
profile, ring dimension, input width, and output rank. It is only a shared
matrix scan; it does not merge commitment identities.

## Compression equations and digit contract

### Negative-binary decomposition

For canonical field coefficient `y_j`, define:

```text
y_j = -sum_k bit_k(q - y_j) * 2^k mod q
digit_(k,j) = -bit_k(q - y_j)
digit_(k,j) in {-1, 0}.
```

Digits are ordered bit-major:

```text
linear_index(k, j) = k * source_coefficients + j.
```

Zero padding follows the last real digit through the end of the final ring row.
The digit width is the canonical field modulus bit width, not a caller-provided
hint.

Recomposition alone does **not** prove that a malicious witness is
negative-binary. Many unrestricted field vectors recompose to the same image.
At the checkpoint, the checked constructor and executor MUST reject any digit
outside `{-1, 0}`. At the atomic protocol cutover, every persisted compression
digit MUST also be covered by a verifier-enforced condition equivalent to:

```text
w * (w + 1) = 0.
```

Pricing a map at coefficient infinity bound one without that complete
verifier-enforced support is unsound.

### B-side stages

For one B-side source `u = B * t_hat`:

```text
xi_F,1 = negbin(u)
u_1    = F_1 * xi_F,1
xi_F,i = negbin(u_(i-1))       for 2 <= i <= L_F
u_i    = F_i * xi_F,i
payload_F = u_(L_F).
```

Each group remains an independent B-side source. Equal-shape execution MAY
batch maps, but the chain, terminal payload, and later certificate identity
remain independent.

### D-side stages

For the D-side opening source `v = D * e_hat`:

```text
xi_H,1 = negbin(v)
v_1    = H_1 * xi_H,1
xi_H,i = negbin(v_(i-1))       for 2 <= i <= L_H
v_i    = H_i * xi_H,i
payload_H = v_(L_H).
```

The terminal folded role has no D image and therefore no H chain. A B
commitment entering a terminal fold still has its own F chain at the eventual
cutover; the terminal fold does not create a new outgoing B commitment.

### Canonical stage geometry

For map `i`, with input image coefficient count `L_i`, field bit width `b`,
ring dimension `d_i`, input width `m_i`, and output rank `n_i`:

```text
real_digits_i   = L_i * b
capacity_i      = m_i * d_i
m_i             = ceil(real_digits_i / d_i)
output_coeffs_i = n_i * d_i
L_(i+1)         = output_coeffs_i.
```

All multiplication, addition, division-rounding, byte conversion, and integer
conversion MUST be checked before allocation. The current ladder requires
`n_i = 1`.

## Parameter and security contract

### Current terminal and ladder

The checkpoint MUST preserve this ladder:

| Source bytes | q128 dimensions | q64 dimensions | q32 dimensions | Maps |
|---:|---|---|---|---:|
| 1 through 8 KiB | 16, 8 | 32, 16 | 64, 32 | 2 |
| over 8 through 16 KiB | 32, 16, 8 | 64, 32, 16 | 128, 64, 32 | 3 |

The first image is 256 bytes for the two-map ladder and 512 bytes for the
three-map ladder. Every terminal is exactly 128 bytes:

| Profile | Field bytes | Terminal `d` | Rank | Coefficients | Bytes |
|---|---:|---:|---:|---:|---:|
| q128 | 16 | 8 | 1 | 8 | 128 |
| q64 | 8 | 16 | 1 | 16 | 128 |
| q32 | 4 | 32 | 1 | 32 | 128 |

The historical exactly-two-layer/160-byte shape MUST NOT be reintroduced
without a separate measured security, setup, wire, and verifier comparison.

### Production security floor

The repository-wide SIS policy at the audited base is
`Quantum128BitADPS16`. Compression MUST use that same policy identifier and the
same `compression_sis_cell` / `min_compression_secure_rank` authority used by
plan validation. It MUST NOT introduce a verifier-only formula, planner-only
rank rule, or second table.

Every map is a separate SIS instance and MUST independently satisfy the
quantum-128 floor at its exact profile, ring dimension, coefficient bound, and
input width. No multi-target or batch discount applies. The exact checked-in
rank-one surface is:

| Profile | `d` | Maximum certified width |
|---|---:|---:|
| q128 | 8 / 16 / 32 | 508 / 7,077 / at least 4,096 |
| q64 | 16 / 32 / 64 | 254 / 3,538 / at least 2,048 |
| q32 | 32 / 64 / 128 | 127 / 1,769 / at least 1,024 |

This matches repository policy because `DEFAULT_SIS_SECURITY_POLICY` is the
same quantum-128 ADPS16 policy at the audited base. Older prose referring to a
138-bit classical production floor is stale.

## Complete-image bound and slicing

The checkpoint accepts complete images of at most 16 KiB and rejects larger
ones. It MUST NOT silently slice, truncate, concatenate, or reinterpret an
image.

The generated and offline schedule census in the evidence report MUST list
every live B and D source byte size at the exact audited head. If a shipped
source exceeds 16 KiB, the checkpoint is incomplete: either increase the
audited complete-image ladder with new SIS and memory evidence or defer that
source explicitly. Do not implement slicing in this PR.

Slicing is a protocol decision because it changes source identity, setup reuse,
relation rows, ordering, and security reduction. Any later slicing plan MUST
bind exact boundaries and prove the security of its repeated/shared matrix
structure. Independent certification of slices does not by itself certify a
concatenated commitment.

## Parallel-safe plan and payload types

### Canonical ownership

`akita-types` MUST own source-independent checked map and chain plan types in a
compression-specific module. These types MUST NOT depend on:

- `LevelParams`;
- schedule keys or generated rows;
- group or source descriptors;
- public claims;
- F/H role names; or
- protocol proof containers.

One canonical constructor MUST validate modulus profile, field bytes and bits,
source size, map count, terminal target, dimension progression, rank-one
geometry, exact digit capacity and padding, and the exact compression SIS cell.
Callers MUST receive an already validated plan rather than revalidate selected
fields independently.

The diagnostic planner MUST consume this authority or disappear. It MUST NOT
remain as a second geometry implementation.

### Packed stage witness

Persistent stage digits MUST use one bit per real digit, where bit one denotes
`-1` and bit zero denotes `0`. The representation MUST record or derive the
exact real digit count from its checked map plan and MUST:

- reject byte-length mismatch;
- reject nonzero padding bits;
- round-trip q128, q64, and q32 values in canonical bit-major order;
- reconstruct the exact map input image;
- expand only into the typed ring rows required by one bounded kernel batch.

An unpacked `i8` buffer MAY exist only as bounded scratch. It MUST NOT be
retained across stages or returned as the persistent witness.

The storage accounting for a source with `L` coefficients and `b` modulus bits
is:

```text
packed bytes = ceil(L * b / 8)
i8 bytes     = L * b.
```

Thus packing reduces persistent digit storage by exactly 8×, excluding a
partial final byte.

### Intermediate images

The executor MUST retain every stage's packed digits and the terminal payload.
It MUST NOT also retain nonterminal images. For `i > 1`, recomposing
`xi_i` canonically recovers `u_(i-1)` or `v_(i-1)`. Holding both values would
duplicate witness state and create two representations to reconcile.

The executor MAY keep one current image buffer while advancing a chain and
MUST reuse or replace that buffer between stages.

### One-source terminal payload

The checkpoint terminal type MUST represent exactly one flat payload tied to
one validated chain plan. Construction MUST reject a coefficient or byte
length different from the plan's terminal size.

It MUST NOT define group ordering, F/H role ordering, fold/root payload
containers, commitment wrappers, or public serialization. PR A owns the future
source descriptors and ordered group boundary needed to define those objects.

## Bounded-memory execution

The reusable executor MUST live outside diagnostic orchestration and accept
checked source-independent plans. It returns:

- packed stage witnesses;
- one checked terminal payload per input identity; and
- measured digitization, preparation/kernel, retained-witness, terminal, and
  peak-scratch facts.

Execution MUST:

1. partition work by exact equal shape;
2. preserve each input's independent identity and result ordering;
3. process a bounded number of right-hand sides per kernel call;
4. expand packed digits just in time;
5. reuse current-image and scratch buffers where the backend permits;
6. use only the existing compression cache namespace and exact prefix;
7. reject a setup prefix shorter than `rank * input_width`; and
8. return errors for malformed plans or backend output shapes.

The bound MUST be expressed in bytes or derived from an explicit RHS batch
limit and checked geometry. Peak scratch MUST not grow with the total number of
equal-shape sources.

F and H are distinct logical matrices at the protocol layer. At the checkpoint,
the source-independent executor MAY read the same physical seed-expanded setup
prefix for equal shapes, matching current diagnostic behavior. This physical
reuse does not assign protocol identity. The atomic cutover MUST decide how
descriptor identity binds logical F/H views while preserving the repository's
shared-prefix setup model.

The diagnostic module becomes a thin adapter: collect current B/D images, call
the canonical executor, emit metrics, and discard its result. The adapter MUST
NOT duplicate digitization, plan geometry, batching, or kernel dispatch.

## Strict checkpoint boundary

Before PR A merges, all of these requirements are invariant:

- Raw B/D images remain public protocol payloads.
- Transcript absorption order, labels, and bytes remain unchanged.
- Schedule, descriptor, `LevelParams`, group-source, and generated-row schemas
  remain unchanged.
- Public commitment and claims APIs remain unchanged.
- Canonical relation layout and proof containers remain unchanged.
- Setup-contribution planning and verifier behavior remain unchanged.
- Compression does not trigger generated schedule regeneration.
- The feature-enabled diagnostic computes real terminal payloads and packed
  witnesses through the production-quality core, reports them, and discards
  them.

A small module declaration in a shared crate root MAY be changed to expose the
new source-independent module. No other PR A-owned surface may change.

## Deferred atomic protocol cutover

The cutover MUST wait until these PR A contracts are stable:

1. descriptor-bearing committed groups;
2. final group requests;
3. ordered schedule keys;
4. group-local parameters and security sizing;
5. generated/runtime key encoding; and
6. prove/verify claims ownership.

After those contracts merge into this branch with an ordinary merge commit,
the cutover must update as one coherent protocol change:

- source/chain identity and mandatory-versus-threshold policy;
- ordered B-group and D terminal payload containers;
- witness and canonical relation layout, including full negative-binary
  support;
- transcript absorption and cross-protocol rejection;
- exact wire serialization and malformed-input validation;
- schedule generation, runtime expansion, proof sizing, and setup contribution;
- direct, recursive, multi-group, and terminal proving;
- verifier evaluation and no-panic rejection;
- generated rows and setup envelopes; and
- deletion of raw commitment payloads and the diagnostic adapter.

Compression is expected to be mandatory after cutover. If measurements justify
a protocol-owned threshold instead, the threshold MUST be authenticated by the
schedule and descriptor and applied identically by prover and verifier. A
runtime-only heuristic or public opt-out is forbidden.

Verifier-facing lengths, dimensions, ranks, supports, and arithmetic MUST be
validated before allocation or indexing. Malformed input MUST return
`AkitaError` or `SerializationError`; verifier-reachable code MUST NOT panic,
use unchecked indexing, or allocate from an unvalidated length.

## Acceptance evidence

### Correctness and rejection tests

The checkpoint MUST include:

- q128/q64/q32 negative-binary round trips;
- exact bit-major ordering and zero-padding tests;
- packed-to-typed-row conversion for every reachable dimension;
- schoolbook equivalence for every reachable ladder map;
- sequential and batched equality;
- mixed-shape partitioning without identity merging;
- malformed profile, field width, source length, map count, dimension,
  progression, rank, target, overflow, unsupported SIS cell, and width tests;
- nonbinary expanded-digit rejection;
- packed byte-length and padding rejection;
- terminal payload length rejection;
- undersized setup-prefix rejection;
- compression cache namespace, key, exact-prefix, and single-flight tests; and
- an end-to-end diagnostic proof that verifies under the unchanged protocol.

### Schedule census

At the exact implementation head, produce a reproducible census of all live
generated and explicitly supported offline schedule shapes. For every fold and
group, report:

- profile and schedule family/key;
- B or D role;
- ring dimension and output rank;
- source coefficient and byte count; and
- selected two- or three-map compression chain, or the exact rejection.

The report MUST state its Git SHA and whether ignored generated tables were
freshly regenerated. It MUST identify the maximum B and D image separately and
state whether any source requires slicing.

### Timing and memory

Release measurements MUST report, separately:

- negative-binary digitization and packing;
- cold preparation/cache construction;
- cached matrix multiplication;
- terminal bytes;
- retained packed witness bytes;
- equivalent retained `i8` bytes;
- maximum expanded RHS bytes per kernel call;
- current-image bytes; and
- measured or conservatively computed peak scratch.

Reports MUST name the exact head SHA, hardware, build profile, feature set,
thread count, source shapes, sample count, and warm/cold methodology. Kernel
time that includes cold preparation MUST be labeled as such.

The checkpoint has no protocol performance claim. These measurements decide
whether the later cutover keeps mandatory compression, selects a
protocol-owned threshold, changes the RHS bound, or revisits the ladder.

## Implementation slices

### B1 — canonical plan and security core

- Add source-independent checked map and chain plans in `akita-types`.
- Use the existing compression SIS authority.
- Delete or migrate duplicate diagnostic geometry.
- Exercise the canonical plan immediately from diagnostics and tests.

### B2 — compact witness and one-source payload

- Extract canonical bit-major negative-binary decomposition.
- Persist packed digits with exact padding checks.
- Add the checked one-source terminal payload.
- Keep all new artifacts on the live diagnostic execution path.

### B3 — reusable execution

- Move chain execution outside diagnostic orchestration.
- Partition exact shapes and bound RHS scratch.
- Reuse the existing compression cache.
- Return witnesses, terminal payloads, and metrics.
- Reduce diagnostics to collection, reporting, and discard.

### B4 — evidence and hardening

- Complete the correctness, rejection, cache, census, timing, and memory
  evidence above.
- Run repository preflight before expensive compilation.
- Validate scoped feature-on and feature-off builds and tests.

Stop after B4. Do not merge PR A or cross the protocol boundary in this work.

## Historical references

- `quangvdao/feat/compressed-commitments:specs/compressed-commitments.md`
  records the superseded full-cutover design and its evolution from 160 to
  128 bytes.
- `quangvdao/feat/compressed-commitments:specs/archive/2026-Q3/commitment-compression-cutover.md`
  preserves the earlier optional/variable-depth cutover draft.
- `d557356d` is the non-building handoff checkpoint.
- `76428c4a` is the historical branch tip with a shadow payload replay harness.
- `b241d3f0` reset the current repository to the diagnostic-first rollout.
- `4e5f0d5c` through `a7c2a0c0` are the current SIS, kernel, diagnostic,
  cache, decomposition, oracle, and cleanup commits.
- PR #311 and the restacked #336/#335/#322/#334 state define the current
  folded-only protocol and invalidate the historical integration topology.
