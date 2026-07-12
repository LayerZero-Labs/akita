# Spec: Compressed Commitments

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-11 |
| Status        | proposed |
| PR            | |
| Supersedes    | |
| Superseded-by | |
| Book-chapter  | |

## Summary

Akita replaces every transmitted Ajtai commitment with a generated compression
chain of depth two or three. Each map recommits a scheduled gadget decomposition
of the preceding image and carries its own authenticated alphabet. Any map may
use the negative-binary alphabet `{-1, 0}`. The production planner additionally
permits the opening-base alphabet on the first map and requires negative binary
thereafter. The binary certificate is local to the scheduled binary digits and
fuses into the existing Stage-2 sumcheck that carries the digit-range claim,
without raising its individual degree.
The payload shape is a planner output, subject to the same standalone SIS
security checks as every preceding map. The shipped planner configurations
select one rank-one terminal ring element with `d = 8`, `16`, and `32` over
q128, q64, and q32, respectively, and therefore serialize to 128 bytes. The
protocol does not enforce that byte count independently of the generated plan.
The planner selects two or three maps, and every map is priced as a standalone
MSIS instance.

This is the protocol encoding, not an optional mode. It includes
native mixed ring dimensions down to `d = 8`, prefix-shared setup matrices,
and schedule-frozen compression metadata for standalone and multi-group
commitments. B/D block-axis slicing has a specified composition boundary but is
not part of the implementation covered here.

## Intent

### Goal

Make generated compression the only wire representation of root commitments,
opening commitments, and nonterminal recursive next-witness commitments, with
one schedule descriptor controlling planning, setup, proof layout, prover,
verifier, serialization, transcript binding, and security certification.

### Protocol objects

For one B-side commitment image `u = B t_hat`, define

```text
u = G_{b_F} xi_F,1
u_1 = F_1 xi_F,1 = G_2 xi_F,2
u_j = F_j xi_F,j = G_2 xi_F,j+1       for 1 <= j < L_F
u_L = F_L xi_F,L.
```

For one D-side opening image `v = D e_hat`, define analogously

```text
v = G_{b_H} xi_H,1
v_1 = H_1 xi_H,1 = G_2 xi_H,2
v_j = H_j xi_H,j = G_2 xi_H,j+1       for 1 <= j < L_H
v_L = H_L xi_H,L.
```

Only `u_L` and `v_L` are transmitted. All preceding images and digit vectors
are relation witnesses. `G_b` is the signed gadget map for base `b`. Every
`F_j` and `H_j` has its own ring dimension, module rank, input width, and
standalone SIS certificate.

The chain depth is explicit in the generated schedule and is restricted to two
or three maps. “Layer” refers
to F/H recommitments, not the original B/D commitment and not B/D input slicing.

### Coverage

Compression applies independently to:

1. every standalone or multi-group root `u` when that commitment is created;
2. every fold opening `v`;
3. every recursive next-witness `u` except the terminal one.

Different commitment identities are never concatenated merely to share a
compression output. In particular, precommitted groups may be created at
different times and retain independent frozen F plans and payloads. Their
relations may later batch at the same opening point.

The last recursive transition binds the transparent terminal state as
`t = A w_terminal` in the next-state transcript slot. It therefore creates no
outgoing `u`, B row, or F chain. The terminal level directly checks the revealed
witness against this exact `t` state and has no D row or H chain.

The F chain of the commitment entering the terminal step must still be checked.
If its compression digits are part of the cleartext terminal witness, decoding
enforces their generic range and directly validates `{-1,0}` on the scheduled
binary span; it does not run a vacuous binary sumcheck. The segment-typed
terminal layout must therefore name and length the compression segment
explicitly.

### Per-map alphabet contract

Every `F_j` and `H_j` descriptor contains an authenticated alphabet tag. The
relation layout, digit spans, binary support, range proof, and security
certificate are all derived from that same tag; none may infer an alphabet from
the map index. Any map may be negative binary. The production search permits an
opening-base alphabet only on the first map and uses negative binary for every
later map:

- If it uses the opening-base alphabet and commitment/opening schedules are
  co-generated, `b_F = b_H = b_1`.
- A standalone conservative commitment whose first map uses the opening-base
  alphabet freezes `b_F = 4`, admitting
  `{-2, -1, 0, 1}`.
- A later opening of that commitment requires `b_1 >= b_F`; a smaller base is a
  completeness failure and is rejected during schedule validation.
- The frozen B and F1 ranks are validated against the actual later `b_1`.
  Conservative generation prices them for the largest permitted later base,
  following the existing conservative-B principle.
- A negative-binary map is independent of the opening base. Its digit depth is
  the field bit width, its exact collision bound is one, and its complete input
  digit span participates in the local binary certificate.
- An opening-base map is priced at the collision bound certified by the generic
  range proof, namely `b_range - 1` for the level-wide balanced range alphabet.
  Its honest gadget base `b_cmp` determines recomposition and completeness but
  does not tighten adversarial digits by itself. In particular, it is unsound
  to price the map at `b_cmp - 1` when `b_cmp < b_range` unless a separate
  verifier obligation enforces that narrower complete span.
- A negative-binary map is priced at bound one if and only if its complete digit
  span is included in the verifier-enforced binary support. Missing, partial,
  overlapping, or out-of-range spans invalidate the descriptor.
- Later production maps are negative binary and are not sized conservatively
  over opening bases.

The planner normally chooses equal F/H dimensions at each compression layer,
but equality and dimension monotonicity are planner preferences, not semantic
verifier invariants. An initial prover backend may return an explicit unsupported
configuration error for nonnested layouts it cannot execute; the verifier must
not reject a valid descriptor solely because it violates a performance
preference.

### Local negative-binary certificate

Every digit belonging to a map whose scheduled alphabet is negative binary
remains in the generic digit-range proof. This keeps the main
range pipeline uniform, even though its larger alphabet alone does not imply
binary digits. A second, local condition narrows exactly the scheduled binary
positions.

Let `W : {0,1}^m -> F` be the virtualized digit table and let `I_bin` be the
schedule-derived support containing every F/H input digit whose map alphabet is
negative binary. At the point
`r_virt` output by the final stage-1 digit-range sumcheck, enforce

```text
sum_x 1_{I_bin}(x) eq(r_virt, x) W(x)(W(x) + 1) = 0.          (BIN)
```

The underlying table is identically zero exactly when every supported digit is
in `{-1,0}`. If it is nonzero, evaluation at random `r_virt` fails to detect it
with probability at most `m/|E|`. After the generic virtualization claim is
fixed, the transcript samples the ordinary Stage-2 batching challenge `gamma`
and, when `I_bin` is nonempty, a distinct fresh `rho_bin`. Stage 2 batches (BIN)
with the carried generic-range claim using the pointwise Boolean weight

```text
omega_bin(x) = gamma eq(r_virt, x)
             + rho_bin 1_{I_bin}(x) eq(r_virt, x).
```

The implementation must not multiply two separately represented multilinear
polynomials inside the sumcheck. Instead it constructs the single multilinear
extension of the pointwise Boolean table:

```text
omega_tilde(X)
  = gamma eq(r_virt, X) + rho_bin eq_I_bin(r_virt, X),

eq_I_bin(r, X)
  = MLE_X [ x |-> 1_{I_bin}(x) eq(r, x) ].
```

The resulting term remains `omega_tilde(X) W(X)(W(X)+1)`: individual degree
two in `W` and degree one in every weight coordinate, exactly as before.
`I_bin` is a short union of schedule-known intervals or subcubes. The prover
stores only the nonzero portions. The verifier evaluates `eq_I_bin` by affine
interval equality contractions; it never allocates a dense support table.
For each nonterminal level with nonempty support, the security ledger charges
both the `m/|E|` restricted-table anchor error and the `1/|E|` fresh-batching
error. These are in addition to, not hidden inside, the unchanged degree-three
Stage-2 sumcheck error.

### Invariants

1. **Mandatory encoding.** No supported config, test, CI mode, profile, or
   benchmark sends a raw B/D commitment or exposes a compression opt-out.
2. **Depth two or three.** Any other depth is malformed. Every
   descriptor length, witness span, relation row, and transcript event is
   derived from the frozen F/H map lists.
3. **Independent identities.** One public compressed payload corresponds to one
   commitment identity. Multi-group roots preserve group boundaries.
4. **Dual range proof.** Every input digit whose map alphabet is negative
   binary participates in both the generic
   range proof and (BIN).
5. **Fresh batching challenge.** `rho_bin` is sampled only after the generic
   virtualization claim and `r_virt` are transcript-bound.
6. **No degree regression.** The batched range sumcheck has the same round
   degree sequence as the uncompressed protocol.
7. **Standalone security.** Every B, D, F_j, and H_j instance is priced
   independently at no less than `DEFAULT_SIS_SECURITY_BITS = 138`. No
   multi-target discount is used.
8. **Exact binary-map norm.** Every negative-binary map uses coefficient bound one and
   is accepted only when the verifier enforces (BIN) for the corresponding
   input positions.
9. **Native dimensions.** F/H dimensions are powers of two accepted by the
   field tier's compression dispatch table. Production execution supports
   `d >= 8`; `d = 1,2,4` are rejected rather than routed through an implicit
   scalar or schoolbook fallback.
10. **Canonical row order.** Relation rows are ordered
    `consistency | A | B groups | D | (F_j groups | H_j)_{j=1..max(L_F,L_H)} |
    evaluation trace`. Omitted roles contribute zero rows without reordering
    surviving roles.
11. **One quotient per native ring row.** Every F/H row uses a quotient in its
    own `d >= 8` ring.
12. **Shared setup prefix.** Every logical A/B/D/F/H view begins at flat setup
    coefficient zero. No per-role offset or additional setup domain label is
    introduced.
13. **One direct scan.** Direct verification aggregates all logical weights for
    a shared setup coefficient and scans the maximum active prefix once.
14. **Frozen descriptors.** All commitment-time choices needed later—depth,
    per-map alphabet/base, F shapes, key certificates, digit widths, layout spans, and
    payload length—are explicit in generated schedules, commitment hints, and
    transcript-bound descriptors.
15. **No-panic verification.** Malformed dimensions, ranks, depths,
    widths, supports, payload lengths, or descriptor arithmetic return
    `AkitaError`/`SerializationError`; verifier-reachable code does not panic,
    index unchecked, or allocate from an unvalidated length.

### Non-goals

- Zero knowledge. Future ZK randomness belongs in the zeroth/original
  commitment layer; compression then commits to an already hiding image.
- A raw-commitment fallback or proof-time depth/alphabet negotiation.
- A multi-target SIS security claim.
- B/D block-axis slicing. The interface and equations are fixed below, but the
  planner, descriptors, prover, and verifier implemented here are unsliced.

## Parameterization and performance model

### Payload

Payload coefficient count, native dimension, and rank are planner outputs. A
candidate is admissible only if the terminal map passes schedule validation and
its standalone SIS certificate clears the configured security floors. Payload
bytes are then the consequence

```text
terminal rank * terminal ring dimension * canonical field-element bytes.
```

The shipped planner configurations select the following shapes:

| field | coefficient bytes | terminal `d` | terminal rank | coefficients | bytes |
|-------|-------------------|-------|------------|--------------|-------|
| q128  | 16 | 8  | 1 | 8  | 128 |
| q64   | 8  | 16 | 1 | 16 | 128 |
| q32   | 4  | 32 | 1 | 32 | 128 |

### Natural q128 anchor

The main planner report should show the common 4 KiB incoming native image,
while sweeps cover the 1–8 KiB supported range. For q128 and a 128-byte terminal
payload, the following matrix-optimal three-map shapes are reference points:

| `b_1` | F1 `(d_1,n_1)` | F1 image | compact F1/F2/F3 | envelope | total |
|--------------------|-----------------|----------|------------|------------|
| 4  | (32,1) | 512 B | 256/64/32 KiB | 256 KiB | 352 KiB |
| 8  | (32,1) | 512 B | 172/64/32 KiB | 172 KiB | 268 KiB |
| 16 | (32,1) | 512 B | 128/64/32 KiB | 128 KiB | 224 KiB |
| 32 | (64,1) | 1 KiB | 104/128/32 KiB | 128 KiB | 264 KiB |
| 64 | (64,1) | 1 KiB | 88/128/32 KiB | 128 KiB | 248 KiB |

These figures are regression anchors, not hand-authored production parameters.
The checked-in/generated planner output must reproduce them (within documented
estimator changes). The protocol-wide floor is 138 classical bits, and every
compression map must also report at least 128 bits under the ADPS quantum
Core-SVP model.

For the 4 KiB q128 anchor, the selected maps have the following estimates:

| `b_1` | F1 classical/quantum | F2 classical/quantum | F3 classical/quantum |
|-------|----------------------|----------------------|----------------------|
| 4  | 288 / 261 | 321 / 291 | 169 / 153 |
| 8  | 205 / 186 | 321 / 291 | 169 / 153 |
| 16 | 153 / 139 | 321 / 291 | 169 / 153 |
| 32 | 287 / 260 | 274 / 249 | 169 / 153 |
| 64 | 228 / 207 | 274 / 249 | 169 / 153 |

The base-16 first map is the narrowest margin in the displayed first layers;
the terminal map is not the binding security constraint for this three-map
frontier.

### SIS-security authority and PR boundary

This PR does not change the production SIS security model or generated SIS
tables. The current checked-in 138-bit classical table remains the sole planner
and verifier authority, and this PR generates compression schedules from the
rows it already provides. Quantum estimates in this document are diagnostic
exploration, not an additional acceptance floor.

A separate security PR owns any decision to require 128-bit quantum security,
the accuracy and versioning of the selected attack model, additional table
coverage, and table regeneration. It must compare the complete schedule and
performance consequences before changing production policy. That investigation
is not a dependency of compression.

The compression implementation uses the existing `SisTableKey` and
`min_secure_rank` authority unchanged. A raw coefficient bound is rounded up by
the existing `sis_table_key_for_linf_bound` authority, so a negative-binary map
with actual bound one is certified conservatively by the shipped bound-two
bucket. A candidate is valid only when the existing lookup covers its field
family, dimension, rounded bucket, and width at the required rank. The planner
must select among those candidates; it must not synthesize a rank or add a
local security exception. This PR may regenerate schedule catalogs, but not SIS
tables.

The search must also evaluate two-map candidates. At minimum these include an
opening-base first map followed by the terminal map, and a negative-binary
first map from the native image to a 256-byte rank-one intermediate followed by
the terminal map. A 1 KiB binary-first q128 chain has a 128 KiB setup envelope;
the corresponding 2/4/8 KiB envelopes are 256/512/1024 KiB. These candidates
trade fewer witness segments and relation rows against larger first-map scans.

For a map from `L` input field coefficients to `n` output ring elements of
dimension `d`, compact ring storage is

```text
n * (L / d) ring elements = n * L field coefficients,
```

not the `n*d` by `L` dense scalar matrix. The compact setup footprint is thus a
factor `d` smaller than a dense scalar encoding of the same map. All envelope
comparisons must use flat field-coefficient footprint, not an unqualified count
of ring elements whose sizes differ by role.

Persistent setup allocation and direct-verifier scanning are governed by the
maximum active compact view, not the sum of matrix sizes, because every view
reuses the same prefix.

### B/D block-axis slicing interface

The compression chain consumes a checked flat source-image vector. In this
implementation that vector is the ordinary unsliced B or D image. A later
slicing implementation may construct the same input as follows without changing
any F/H chain equation.

For one commitment identity, let the source digit vector have canonical
logical coordinates `[block][source_row][digit]`, with `block` the only sliced
axis. Commitment identities remain outside this vector and are never joined by
slicing. For a
power-of-two slice count `f` dividing the block count `B`, slice `s` owns the
half-open block interval

```text
[s * B/f, (s + 1) * B/f),                 0 <= s < f.
```

Every slice retains all source rows and digit planes. It is therefore
a union of one contiguous block interval in each digit plane, not necessarily
one contiguous interval of the physical witness. The canonical restricted
digit vector is obtained by iterating slice-local block, then source row, then
digit in the same order as the unsliced layout. No padding,
gaps, duplicate coordinates, or reordered coordinates are allowed.

One narrower source matrix is reused for every slice. For B, compute

```text
u_s = B_slice * t_hat|_s,                  0 <= s < f_B,
u_source = u_0 || ... || u_(f_B-1).
```

For D, compute `v_s` and `v_source` analogously with an independently selected
`f_D`. The first F or H map decomposes the coefficient vector of `u_source` or
`v_source` in exactly this slice-major order. Subsequent compression maps are
unchanged. The public payload remains one schedule-sized image for each commitment
identity, not one payload per slice. At a multi-group root, each B-side
commitment applies this construction independently under its own frozen plan;
the identities are not concatenated before compression. The D-side construction
applies once to the opening identity defined by the opening schedule.

A slicing descriptor must authenticate the source role, original block count,
slice count, blocks per slice, matrix shape, per-slice image coefficient count,
total concatenated source-image coefficient count, and all derived witness and
relation spans. Validation uses checked arithmetic and rejects a non-power-of-two
slice count, nondivisibility, zero or oversized dimensions, inconsistent totals,
and any partition that is not an exact cover. Binary support is derived only
from F/H digit spans after the sliced source images have been concatenated.

Security certification prices the reused source matrix at the width of one
slice and prices the first compression map at the full concatenated source-image
width. Because reuse induces a structured repeated-matrix relation, the slicing
implementation requires a binding reduction for that exact structure; security
must not be inferred by treating the slice applications as independent random
matrices. Setup evaluation folds all slice weights onto the single stored source
view, while relation rows retain one residual per slice in increasing slice
order.

No slice count or slice selector appears in the schedules implemented by this
work, and no diagnostic slice estimate may affect a proof. Slicing is a
follow-up implementation with separate security review and regression tests.

## Design

### First-class compression plan

Do not extend `CommitmentRingDims` with F/H fields. That type remains the
A/B/D fold-geometry contract. Replace parallel source-key and compression
metadata with a minimal input spec and one canonical validated plan,
approximately:

```rust
struct CompressionMapSpec {
    key: AjtaiKeyParams,
    alphabet: CompressionAlphabet,
}

struct CompressionChainSpec {
    source: CompressionSourceId,
    maps: Vec<CompressionMapSpec>, // checked length 2..=3
}

struct ValidatedCompressionCatalog {
    // private, checked compiled fields
}
```

`CompressionAlphabet` is either `NegativeBinary` or
`OpeningBase { log_basis }`. The latter authenticates the map's honest
recomposition base `b_cmp`; it is not the range-check base. The compiler derives
the canonical field bit width from the active field, uses exactly that many
negative-binary digits, and uses the ordinary full-field digit-count primitive
for an opening-base map. Every active opening-base map requires
`0 < log2(b_cmp) <= log2(b_range) < 128`: the level-wide range proof must cover
every digit admitted by the map's recomposition base. In a co-generated level,
the current F1 and opening H1 maps additionally require `b_cmp = b_range`; a
frozen precommitted F1 may use its authenticated smaller base. Standalone
commitment creation fixes an opening-base F1 to base 4; a negative-binary F1
remains base-independent. The standalone compiler receives the authenticated
maximum permitted later opening base from the configuration policy and requires
it to cover base 4. The ordinary standalone `LevelParams.log_basis` may remain
the minimum configured base and is not reused as this conservative maximum.
SIS collision pricing for an opening-base input uses
`b_range - 1`, where `b_range` is the level-wide authenticated range-proof base;
it never substitutes `b_cmp - 1`. Negative-binary pricing uses raw bound one
only for the complete compiler-derived input span.
An opening-base alphabet is valid only on the first map of a chain; every later
map is negative binary. The first map may also be negative binary.

One `validate_and_compile` call consumes every chain spec for the level and
returns the sole `ValidatedCompressionCatalog`; there is no separately
constructible per-chain validated plan. The compiler rejects missing,
duplicate, or out-of-range sources. A co-generated level requires the current
F chain, every frozen precommitted F chain, and the opening H chain in canonical
order; standalone creation requires exactly its new current F chain. This
validates the catalog's geometry,
dispatch, and SIS certificates, but does not by itself authorize protocol
relation execution: it may drive honest standalone chain computation, but only
the semantic-layout compiler may certify that every bound-one map input is
present in the verifier-enforced binary support of a proof.

Catalog compilation authenticates whether its context is a co-generated level
or standalone commitment creation; this is protocol meaning, not an execution
strategy flag. The standalone context carries the authenticated maximum later
opening base used by the existing conservative-B policy; the compiled catalog
retains that effective range/security base. It does not infer context from the legacy
`RelationMatrixRowLayout`: `WithoutDBlock` currently conflates standalone and
terminal paths. A terminal proof creates no new catalog. Its semantic layout
must consume the already-frozen incoming commitment catalog and reject any
attempt to compile a new current B/F or opening D/H chain.

`CompressionSourceId` resolves the existing B/D role identity and key; it does
not copy source dimensions or key metadata. Its stable variants distinguish the
current B source, each authenticated precommitted B source, and the opening D
source; an opening source is invalid when the D block is absent.
`AjtaiKeyParams` is the sole owner
of row length, column length, `SisTableKey`, and ring dimension. Digit depth,
coefficient lengths, witness spans, and binary support are derived facts, never
free plan fields or serialized vectors.

The checked constructor proves, with checked multiplication before allocation,

```text
d_i              = key_i.sis_table_key.ring_dimension
input_coeffs_1   = source_output_coeffs * digit_depth_1
input_coeffs_i   = output_coeffs_(i-1) * digit_depth_i       for i > 1
input_coeffs_i   = key_i.col_len * d_i
output_coeffs_i  = key_i.row_len * d_i
payload_coeffs   = output_coeffs_L.
```

One canonical digit-math primitive derives each digit depth and gadget scalar
from the authenticated alphabet, field, and base. No caller supplies both an
alphabet and its derived depth. The semantic-layout compiler consumes the
catalog's private compiled maps directly; do not add integer-index forwarding
getters merely to reveal those fields. Only that compiler may turn the compiled
facts into spans and support.

Names may change during implementation, but these authorities and equalities
are normative and descriptor-bound. There is one B-side chain per commitment identity and one
D-side chain per opening schedule. `PrecommittedGroupParams` must freeze the
F chain in addition to its current geometry and conservative B
rank. `ExecutionSchedule`/`LevelParams` must carry the current H chain and the F
chain for the recursive commitment being created.

Use one checked constructor and validation routine called by planner output,
setup generation, deserialization boundaries, prover, and verifier. Do not add
thin `_for_level` wrappers or separate “certified” versus “executed” bounds.

### Small-ring execution and SIS tables

Compression execution and certification must:

1. use the existing SIS rows and canonical bound-bucketing authority; in
   particular, certify an actual bound-one negative-binary layer against the
   existing bound-two bucket;
2. generate schedules only from candidates covered by those existing rows,
   even when arithmetic execution supports a smaller dimension;
3. add a compression execution dispatch independent of A-role sparse-challenge
   support and the current field-specific B/D minima;
4. extend the cached CRT/NTT path through `d = 8`; a production descriptor is
   invalid when its dimension lacks the compression, NTT, or SIS-table
   capability for the active field preset;
5. exercise mixed dimensions and the smallest enabled dimension in every field
   preset. Supporting `d = 1,2,4` later requires an explicit dispatch, cache,
   quotient, SIS-table, and performance-policy extension.

### Compression dispatch and cached NTT contract

Do not broaden `RingRole::{Inner,Outer,Opening}` to make small F/H dimensions
look like B/D support. Add a purpose-specific compression dispatch slot and
derive all validators and runtime-to-const-generic arms from the same policy
table. The initial production policy is:

| field tier | A-role | B/D roles | compression roles | CRT/NTT cache |
|------------|--------|-----------|-------------------|---------------|
| q128 | `64,128` | `32..256` | `8,16,32,64` | `8..512` |
| q64  | `64..256` | `32..256` | `16,32,64,128` | `16..1024` |
| q32  | `64..256` | `64..256` | `32,64,128,256` | `32..2048` |

Every range is the listed powers of two. The compression table expresses
execution capability, not planner preference: an apparently attractive map is
still rejected unless the canonical SIS lookup can round its actual bound to a
shipped `(field family, d, coefficient bucket)` row whose rank clears the
security floor. In particular, the current tables begin at `d = 32`, so the
initial generated schedules cannot select `d = 8` or `d = 16` even though the
arithmetic path supports them. The verifier validates the descriptor against
both execution capability and SIS coverage rather than inferring a dimension
from the field width.

The tier minima are intentional. Supporting `d=8` for q64 or q32, or `d=16`
for q32, would add planner states, SIS rows, dispatch branches, and heavy
const-generic kernel instantiations without serving a shipped schedule. Such a
dimension is added only after a measured planner candidate justifies the code
size and compile-time cost. Arithmetic roots existing in the CRT primes are not
by themselves a reason to expose a protocol arm.

The tier maxima `64,128,256` are equally intentional. Compression does not
reuse the larger NTT capability merely because another role requires it. A map
above the active compression maximum is malformed even when its dimension is
present in the field tier's NTT or A/B/D dispatch tables.

Heavy compression dispatch occurs at one canonical compute-backend boundary.
Planner, chain, hint, and protocol code pass a checked runtime plan to that
boundary; they do not each expand the runtime-to-const-generic macro. This
limits each admitted `(field,d)` kernel to the necessary backend/cache
instantiations instead of multiplying it across protocol call sites. The
implementation report records clean release build time and text/binary size
before and after the new arms; unexplained duplicate kernel monomorphizations
must be removed before adding another dimension.

Concretely, `dispatch/mod.rs` gains
`ProtocolDispatchSlot::Compression`, while `dispatch/policy.rs` gains its arm
list for every tier. `ProtocolDispatchSlot::Ntt` gains the corresponding tier
minimum (`8,16,32`), and `NttSlotCacheAny` gains `D8`.
`Envelope` remains independent. The NTT minimum is derived from the NTT policy
rather than the B/D minimum. Compression schedule validation checks, in this
order:

1. nonzero power-of-two `d` and membership in the field-tier compression slot;
2. `d | gen_ring_dim` for the shared flat setup;
3. membership in the field-tier NTT slot for the selected backend path;
4. exact agreement between plan dimension and `SisTableKey.ring_dimension`;
5. an audited SIS row at the authenticated coefficient bound and adequate rank;
6. checked divisibility of input/output/setup/quotient field lengths by `d`.

No caller may treat generic power-of-two validity as protocol authorization.
Generic A/B/D schedule validation enforces structural invariants only: nonzero
powers of two, the A-challenge capability, a protocol-wide B/D minimum of 32,
and `d_d | d_b | d_a`. Exact execution authorization comes from the
purpose-and-field-tier dispatch policy. Compression and NTT do not share or
broaden an A/B/D dimension list.

The existing CRT auxiliary primes already support the new smaller transforms,
so no new primes are required: a primitive root for the larger power-of-two
domain yields the needed smaller domain. Nevertheless, each tier's newly
enabled minimum (`d=8,16,32`) must receive explicit twiddle, round-trip,
negacyclic product, cyclic product, and CRT-capacity tests. The capacity profile
and chunking logic remain authoritative; a smaller ring creates more ring
columns for a fixed flat width and must not bypass safe-accumulation chunking.

### NTT cache planning

`NttSlotCache<D>` already stores both negacyclic and cyclic transforms of the
same flat setup prefix. Retain that representation. A cache entry is identified
by

```text
(ring_d, num_prefix_ring_elements).
```

No role, map identity, or matrix shape belongs in the key: all logical setup
views begin at coefficient zero, and the cached object is a flat transformed
prefix. At a fixed `d`, equal-length views are literally the same cache entry,
and a longer warmed prefix serves every shorter view by slicing.

The schedule compiler derives one `PreparedNttPlan` (name illustrative) for
the complete authenticated schedule catalog accepted by one backend-prepared
proving context. For each active compression dimension, it takes the maximum
flat prefix across every F/H map, layer, and commitment identity in that
catalog:

```text
compression_envelope[d]
  = max_{F/H map K with d_K=d} (required_field_coeffs(K) / d).
```

Every participating footprint is validated as divisible by `d` before this
plan is constructed; this is exact division, not padding. Structural padding
used by a higher-level matrix layout is not part of the cached setup prefix.

`AkitaProverSetup` remains the backend-independent expanded setup artifact; it
does not store CPU/accelerator NTT state. The scheme derives the checked
`PreparedNttPlan` from its validated schedule catalog and passes it to the one
canonical `ComputeBackendSetup::prepare_setup` boundary. That boundary
registers and eagerly materializes all planned keys in the backend-prepared
context before any commitment or transcript work. Compression must not rely on the current
`with_shared_ntt::<D>()` helper, which always requests the full setup envelope.
It obtains the schedule-planned key and borrows the corresponding prefix slot
through the canonical backend cache API. Profile output reports each
`(d,prefix)` entry and counts both its cyclic and negacyclic storage.

The backend-prepared contract coalesces compression and existing role-cache
requirements before construction. For each `d`, it owns one slot whose length
is the larger of `compression_envelope[d]` and any A/B/D cache prefix already
required at that dimension. Thus a longer existing role cache serves every
compression layer by slicing; otherwise prover setup stores exactly the
compression envelope. Commit and opening clusters never construct, extend, or
rebuild these slots. Lazy construction is permitted only as a diagnostic test
fallback and is a production/profile failure.

`PreparedNttPlan` is the sole key-selection authority. Given `(d, required
prefix)`, it returns the containing canonical slot and checked slice length.
Callers do not search the cache map, choose between equal and longer keys, or
construct `NttCacheKey` directly. A backend context prepared without the
validated catalog plan cannot enter a compression proving path.

F-chain execution uses the commit operation cluster; H-chain execution uses the
opening cluster. Recursive next-F maps use the active level's commit cluster.
The ring-switch cluster continues to own the existing relation quotient path;
compression quotients are produced by compression-chain execution as described
next, so they are not recomputed through the A-role quotient API.

### Multi-map compression fusion

Cache sharing and execution fusion are distinct. Every map at the same `d`
already shares one transformed setup prefix. A fused runtime matvec is allowed
only when all input digit vectors are simultaneously fixed. In the ordinary
fold schedule, H is computed before the fold challenge while the next F input
depends on that challenge. Those F/H maps share the prepared cache but cross a
transcript/data-dependency barrier, so their negacyclic chain images cannot be
evaluated together. Likewise, adjacent layers of one chain are sequential
because the next decomposition depends on the preceding image.

That barrier does not force the F quotient to be completed at commitment time.
The F chain needs only `u_neg` to derive its next image and payload. At opening
time, every F digit vector is available from its authenticated hint and its
negacyclic RHS is already determined by the successor gadget image or terminal
payload. Its missing cyclic product may therefore be completed in the same
matrix scan that computes an equal-shape H map. For one aligned layer, the
opening bucket requests

```text
F item: cyclic only; known u_neg
H item: cyclic and negacyclic
```

and derives both quotients after the scan. This does not move a transcript
event or defer any value needed to construct the F commitment.

Independent maps in the same operation window can be fused. The primary case
is several group/commitment maps with the same dimension and generated shape.
Expose one canonical batch backend operation whose items request negacyclic,
cyclic, or both outputs; a batch of length one is the ordinary path. Its first
optimized bucket requires equal

```text
(field tier, d, row count, column count).
```

The authenticated digit bound determines safe accumulator subchunking inside
the bucket, not semantic fusion eligibility. Items with different bounds may
share setup traffic while retaining their own checked capacity limits.

For a bucket of `R` right-hand sides requesting both domains, the kernel
column-tiles once. For each setup entry it loads the cached
cyclic/negacyclic pair once, constructs the paired digit transforms for each
right-hand side, and updates `R` pairs of row accumulators. With matrix shape
`n` by `L`, separate execution reads roughly `2 R n L` transformed setup
elements, whereas the batch reads `2 n L`. Thus two simultaneously available
maps can remove about half of setup-cache traffic.

For the common one-F/one-H case across the commitment/opening boundary, eager
paired execution reads `4 n L` setup elements in total. Deferred F completion
reads `n L` negacyclic entries during commitment, then one `2 n L` paired slot
scan during opening for F-cyclic plus H-cyclic/negacyclic, reducing total setup
traffic to `3 n L`, or 25%. With `G` aligned F maps completed beside one H map,
the corresponding traffic changes from `2(G+1)nL` to `(G+2)nL` before any
additional same-window F batching.

Neither form removes digit transforms, pointwise multiply-accumulates, or
inverse transforms required by the requested domains. In particular, deferring
F cyclic completion loses the shared coefficient reduction of constructing an
F cyclic/negacyclic transform pair at once. The backend chooses eager paired or
deferred completion from a measured internal execution policy; this is not a
descriptor or transcript choice. Accumulator memory grows as `O(R n)` and the
batch size must be capped by the same L2-derived tiling policy used by the
current fused A/B/D kernel.

The common implementation owns CRT safe-width chunking for the whole bucket,
with per-item subchunks where bounds differ. Exact-shape grouping is the
production fast path. For unequal widths at the same `d`, a later flat-prefix
fusion may maintain each item's own row/column counters while scanning the
shared setup prefix, then run tails separately. A naive common-column-prefix
shortcut is invalid for rank greater than one because each shape has its own
row-major column stride; unequal-width fusion is deferred, not forbidden.

The planner may use exact fusion eligibility as a final tie-break between
otherwise equivalent candidates, but it must not weaken SIS security or enlarge
the setup envelope merely to align shapes. Profiles report bucket sizes, setup
elements loaded, digit transforms, pointwise products, and tail work. The
batched and independent kernels are compared on the common q128/q64/q32
1-KiB-image cases; fusion stays enabled only when it improves the applicable
prover benchmark without violating the 5% gate elsewhere.

Verifier fusion is broader because it has no secret-input dependency: all
same-prefix setup weights, including equal-shape F/H maps, are accumulated into
the single combined setup-prefix scan already required below. Direct and
offloaded verification must use that combined scan rather than evaluating one
setup MLE per compression map.

### Native compression quotient

For one compression map `K` at dimension `d`, let `xi` be its packed input
digits. Compute the negacyclic and cyclic matrix products together:

```text
u_neg = K * xi              in F[X]/(X^d + 1),
u_cyc = K * xi              in F[X]/(X^d - 1),
r_K   = (u_cyc - u_neg) / 2.
```

Coefficientwise division by two is valid for every supported odd prime. These
objects satisfy the canonical polynomial lift

```text
K xi = u_neg + (X^d + 1) r_K             in F[X].
```

For a nonterminal row, honest recomposition gives
`u_neg = G xi_next`; for a terminal row it gives `u_neg = payload`. Thus the
same `r_K` is the quotient for

```text
K xi - G xi_next = (X^d + 1) r_K
```

or the terminal equation. Gadget recomposition itself uses field scalars and
does not create an additional ring-product quotient.

Add one canonical batched compression matvec kernel that consumes the
schedule-sized `NttSlotCache<d>` prefix and returns the requested cyclic and/or
negacyclic image for each batch item. The canonical quotient derivation then
returns `(u_neg,r_K)`, accepting a previously determined `u_neg` for a deferred
F completion. When both domains are requested, the kernel must fuse their
accumulations over the same column tiles and construct the input's two digit
transforms together; invoking
`mat_vec_mul_ntt_single_i8` and its cyclic sibling as two independent full
passes is the correctness oracle, not the performance implementation. The
negative-binary alphabet uses the exact signed digits `{-1,0}` with the base-2
kernel bound; an opening-base map uses its authenticated log basis. Both paths
remain subject to the existing CRT safe-width chunking.

Chain execution already needs `u_neg` to derive the next digit vector. Eager
execution also derives and decomposes `r_K` immediately. Deferred F execution
retains the authenticated digit segments already required by the relation; its
known negacyclic RHS is reconstructed from the successor segment or terminal
payload rather than storing an NTT-domain image. At opening, cyclic completion
derives and decomposes `r_K` before relation-witness assembly. Do not persist
transformed digit vectors merely to bridge the phases: their memory cost
defeats the cache-traffic saving. The descriptor fixes one logical quotient and
both execution strategies call the same canonical quotient derivation.

For distributed execution, each worker computes its additive `u_neg` and
`r_K` contributions from its column shard. Reducing `u_neg` produces the global
intermediate image and reducing `r_K` produces the global quotient. It is
unnecessary to communicate `u_cyc` separately because the quotient operation
is linear. Both reductions use the map's native row shape.

`RingRole::{Inner,Outer,Opening}` need not be overloaded with compression
semantics. Prefer one generalized canonical matrix-role descriptor used by
setup/relation code, or a distinct compression dispatch slot, over pretending
F/H are B/D roles. Preserve the repository rule that security sizing and the
verifier-enforced norm read the same primitive bound.

### Witness layout

Define one canonical **semantic relation-witness layout** with schedule-visible
compression segments. Conceptually, its identity-ordered segments include

```text
xi_F,1 | xi_H,1 | ... | xi_F,L_F | xi_H,L_H
```

with F segments repeated per root/recursive commitment identity as required by
the canonical row order. The semantic layout owns segment identities, lengths,
native dimensions, relation roles, and canonical virtual order. Binary support
is derived from every F/H-input segment whose map alphabet is negative binary;
it is never serialized as an
attacker-controlled bitmap.

The compression-local arena uses flat field coefficients as its address unit.
It first allocates every `xi` segment in layer-major relation order—current F,
authenticated precommitted F order, then H at each live layer—and then allocates
the decomposed F/H quotient segments in that same family order. Missing layers
allocate no placeholder. These starts remain local until the global semantic
compiler performs one checked translation before padding. A `xi` span itself
has no native-dimension tag: its current map and predecessor gadget views may
interpret the same coefficients under different dimensions, and compilation
checks divisibility for every such view. Quotient spans never enter binary
support.

Do not append global compression spans to today's flat multi-chunk
`WitnessLayout`. Distributed proving applies machine-ownership policies to the
semantic layout after it is constructed. In the single-machine case this
composition yields the ordinary physical order directly. In the distributed
case F/H digit segments are column-partitioned across machine-local witnesses,
as specified below.

`AkitaCommitmentHint` and prover-private opening hints must retain the original
decompositions plus every F-chain digit vector and any
recomposition material required to open later. Intermediate images need not be
stored when they can be recomposed from the digit vectors. Hints may grow
substantially; implicit reconstruction from a current planner default is
forbidden because a standalone commitment freezes its own plan.

The F-chain witness belongs to the proof that later opens that commitment. The
commit operation computes the raw B image, every scheduled decomposition, and
the terminal payload, then stores the F digits in the commitment hint. It must not append a
new outgoing commitment's own F digits to the witness being committed, which
would create a self-reference. When that commitment is opened, its stored F
digits enter the relation witness. The opening-local H chain is computed from
the current D image and enters that same relation witness.

### Distributed proving integration

The public compressed commitment remains one global protocol object. Its digit
witness may be distributed. For the B/F chain, machine `j` first computes a
partial raw B image `u_j`; the workers reduce `u = sum_j u_j`. They then execute
one canonical compression chain distributively:

1. machine `j` derives its scheduled column shard `xi_F1,j` from the canonical
   global `u`;
2. it computes `u_1,j = F_1,j xi_F1,j`, where `F_1,j` is the corresponding
   column restriction;
3. the short reduction `u_1 = sum_j u_1,j` fixes the intermediate image;
4. for each later layer `ell`, machine `j` derives its scheduled
   `xi_Fell,j` shard from `u_{ell-1}` and contributes
   `u_ell,j = F_ell,j xi_Fell,j`;
5. reducing the final contributions gives the one transmitted payload.

The H chain is identical after reducing the partial D images. No compression
digit vector crosses the worker boundary. Matrix multiplication and witness
storage are both distributed, while the descriptor, security instance, and
public payload remain exactly the standalone global chain.

It is incorrect to compress each `u_j` independently and add the resulting
payloads. Digit decomposition is not additive, and this would replace the
certified terminal instance by the wider repeated-column map

```text
[F_L | ... | F_L] [xi_FL,0; ...; xi_FL,W-1].
```

Standalone certification of `F_L` does not price that witness width. Sending
one independently certified payload per worker is sound but multiplies wire
size by `W` and is not supported.

Each F/H digit segment is partitioned independently in its own native ring-
column axis. Equal F/H dimensions are unnecessary, and the column count need
not divide the machine count. For `L` columns and `W` machines, machine `j`
owns

```text
[floor(j L / W), floor((j + 1) L / W)).
```

Missing final slots up to `ceil(L/W)` are structural zeros. They are not
serialized digits, setup columns, range-check entries, or quotient inputs. The
real negative-binary F/H shards define the distributed binary support.

The semantic layout exposes a distribution policy per segment:

```text
PerMachineFull       folded responses and local quotient contributions
BlockPartitioned     e/t segments selected by block ownership
ColumnPartitioned    F/H compression digits
```

The distributed layout applies these policies; compression code never assigns
worker offsets independently. For a compression map `K` with successor gadget
map `G`, worker `j` computes its partial negacyclic image and product quotient

```text
u_neg,j  = neg(K_j xi_j)
r_prod,j = (cyc(K_j xi_j) - u_neg,j) / 2
K_j xi_j = u_neg,j + (X^d + 1) r_prod,j.
```

The successor digits are derived from the reduced global image, so in general
`u_neg,j != G_j xi_next,j`. Define the local residual
`delta_j = u_neg,j - G_j xi_next,j`, with the schedule-designated RHS owner
also subtracting the public terminal image. A worker's extended relation equals
`delta_j`; it is not required to vanish. Reduction makes `sum_j delta_j = 0`,
so the coordinated sumcheck proves the global rows:

```text
sum_j (B_j t_j - G_b,j xi_F1,j)                 = 0
sum_j (F_ell,j xi_Fell,j - G_2,j xi_F(ell+1),j) = 0   for ell < L
sum_j  F_L,j xi_FL,j                            = u_pub.
```

The H rows are analogous. One schedule-derived machine owns the public RHS as
an additive convention only; this does not make its local relation valid on its
own. Every machine digit-decomposes and carries `r_prod,j` in the canonical
quotient span at that row family's native dimension. Machines contribute round
polynomials for their local residuals, and the coordinator sums those
polynomials before each challenge. No worker-to-worker quotient reduction is
needed, and no test may assert that an individual compression row vanishes.
The payload itself appears once in the proof and transcript.

Independent commitment identities remain independent under distribution.
Multi-group payloads are never concatenated into one F chain. Their canonical
hints expose semantic F segments that can be repartitioned for the active
machine count without changing the commitment descriptor. Recursive F shards
remain worker-local while the next level is distributed and are gathered or
recomputed only at the explicit W-to-1 cutover.

The companion distributed-prover design is maintained on the public branch
[`refactor/machine-major-distributed-prover`](https://github.com/quangvdao/akita/tree/refactor/machine-major-distributed-prover),
with the normative composition in
[`specs/machine-major-distributed-prover.md`](https://github.com/quangvdao/akita/blob/refactor/machine-major-distributed-prover/specs/machine-major-distributed-prover.md).

#### Cross-PR ownership and preferred order

This compression work owns `ValidatedCompressionCatalog`, semantic relation-row and
relation-witness layouts, mixed-ring relation providers, F/H hints, and security
certification. The distributed work owns machine input/output geometry,
distribution policies applied to semantic segments, local additive relation
contributions, process orchestration, and W-to-1 cutover. Setup, trace, range,
and sum-check code consume their composition rather than growing separate
compression and distributed layout authorities.

Preferred implementation order:

1. land compression descriptor, semantic layout, small/mixed-ring quotient, and
   canonical relation-provider foundations;
2. rebase the distributed implementation onto those authorities and add
   machine sharding;
3. finish compression's recursive/multi-group integration on the composed layout;
4. add the distributed process runtime only after the composed single-host
   prover and structured verifier pass end to end.

This ordering avoids freezing a `z/e/t/r`-only distributed public type and
avoids extending the flat multi-chunk layout with global F/H tails that
would immediately need to be removed.

### Relation rows and quotient construction

This is a prerequisite refactor, not additive row plumbing. Current mixed-ring
support is largely type/schedule scaffolding: `ring_switch_build_w` rejects
`d_d != d_a`, `RingRelationInstance::ensure_ring_dim` requires one dimension,
`compute_multi_group_relation_quotient` returns one uniform quotient vector,
`emit_r_decomposition_tail` assumes one ring for every row, and the current
relation sumcheck has one ring-coordinate axis and one `alpha_compact`. These
uniform assumptions must be removed before compression rows are wired into the
proof.

Replace the two-case `RelationMatrixRowLayout` with a checked schedule-derived
layout whose native order is:

```text
consistency
A_current
B_current
(A_precommitted_i, B_precommitted_i) for i in authenticated precommitted order
D
(Fell_current, Fell_precommitted_0, ..., Fell_precommitted_(G-2)
 Hell) for ell = 1..max(L_F,L_H)
evaluation trace
```

At recursive scalar levels there is only the current group. This current-first
order preserves the existing multi-group A/B relation order and therefore the
existing proof bytes when Slice 3 migrates those rows. `CompressionSourceId`
distinguishes `CurrentOuter`, `PrecommittedOuter(i)`, and `Opening`; no compiler
infers identity from an untyped vector position. Compression layers use that
same current-first order, omit absent layers, and place H after the live F rows
at each layer. The evaluation trace remains last.

Within each B or D family, source rows retain the existing unsliced block-fast
order. The first compression map consumes the decomposition of that image.

Do not embed compression rows in the existing uniform `m_compact` / powers-of-
alpha vector. They are small native-ring inner-product relations with one sparse
operand. Introduce one canonical relation-weight provider consumed by both the
prover quotient builder and verifier/setup evaluator:

- existing consistency/A/B/D providers may use their current dense/structured
  formulas;
- F/H providers expose only nonzero witness spans and compact setup weights;
- trace remains its existing succinct provider.

The provider reports native ring dimension, row span, witness spans, setup view,
and quotient requirement. This avoids materializing a single dense
`relation_matrix_col_evals` over all compression columns and avoids duplicating
role formulas between direct and recursive setup modes.

Every F/H native relation `y = K x` contributes its own negacyclic quotient at
that role's ring dimension. There is no scalar-row exception or quotient-free
compression path. Quotient witness spans and row counts are part of the
descriptor. Checked conversions project a shared flat coefficient view into a
role ring; divisibility must be validated before projection.

The public RHS is zero on B, D, and every nonterminal chain row. Only the
terminal F/H rows contain the transmitted compressed payloads. `RelationRhsLayout`, relation-claim
assembly, prepared replay state, root verification, suffix verification, and
the zero-fold path must all adopt this meaning; none may continue interpreting
`Commitment` as public B rows.

### Setup prefix and contribution evaluation

`AkitaSetupSeed` continues to expand one flat setup vector. Logical matrix views
are shape descriptors over the prefix:

```text
flat coefficient i for any A/B/D/F/H view  --> setup[i]
```

There is no cursor and no disjoint offset. Equal shapes are identical views;
shorter shapes reuse the longer view's prefix. The generation-ring dimension is
a storage/expansion unit only. Footprints are compared in flat coefficients,
then converted once with checked divisibility.

Generalize `SetupContributionPlan` and `SetupIndexWeightEvaluator` from fixed
A/B/D fields to the canonical relation providers. For a direct verification,
the evaluator computes

```text
weight[i] = sum_role weight_role[i]
```

over every active logical view containing `i`, then performs one scan over
`0..max_role_footprint`. Sliced views contribute interval/block selector factors;
F/H views contribute their sparse witness MLE factors. Recursive setup-product
mode commits to the same combined weight table and must agree byte-for-byte on
the final claim.

Use the smallest active native ring as the common scan coordinate, projecting
larger native roles onto it. Since all supported dimensions are powers of two,
this does not require planner-enforced monotonicity. The smallest production
compression coordinate is `d=8`; dimensions `1,2,4` have no implicit scalar or
schoolbook path. The recursively committed setup prefix may remain packed at
its existing storage dimension; its logical claim must be the same flattened
prefix claim.

Current envelope code comparing `row_len * col_len` ring elements must be
replaced with flat-coefficient footprint comparisons so mixed dimensions are
priced correctly.

### Proof objects and transcript

Replace `Commitment`'s semantic assumption of native B/D ring rows with a
checked flat payload whose coefficient count comes from the chain plan. The
wire object need not carry a ring dimension; the transcript and deserializer
receive it from the bound schedule descriptor. Root proof containers hold one
terminal F payload per commitment identity; fold proofs hold the terminal H
opening payload and, when nonterminal, the next terminal F payload.

The payload encoding is exactly the terminal output vector in ring-row-major
coefficient order `(row_0.c_0, ..., row_0.c_(d-1), row_1.c_0, ...)`. Each
coefficient uses the field's canonical fixed-width `AkitaSerialize` encoding.
The payload contains no vector
length, ring dimension, rank, alphabet, chain depth, or other header; those
values come from the already-bound descriptor. Consequently its encoded length
is exactly the coefficient count fixed by the validated terminal-map plan times
the canonical field-element width. A containing proof derives the number and order of
payloads from the schedule rather than serializing an attacker-controlled
count. Deserialization validates the descriptor and expected coefficient count
before reading or allocating, rejects truncation and trailing coefficients, and
constructs the native ring view only after checked divisibility.

Descriptor bytes must bind, in stable order:

- both chain depths and every layer's base/digit depth;
- every F/H ring dimension, rank, input width, output width, and SIS table key;
- compression witness spans, row spans, and quotient spans;
- expected public payload coefficient counts;
- terminal role omissions and the binary-support derivation version.

Use dedicated transcript labels for compressed payloads and `rho_bin`. Absorb
the schedule descriptor before any payload interpreted by it. Transcript tests
must pin the order and prove that changing any plan field changes the transcript.

### Sumcheck integration

The protocol has two independent proof obligations:

1. the digit-range chain, including (BIN);
2. the linear relation chain, including all compression rows.

Compression does not reorder them. (BIN) remains attached to digit
virtualization at `r_virt`, while every F/H equation joins the ordinary linear
relation sumcheck. When verifier offloading is enabled, its setup-product and
carried-claim machinery consumes the same relation claim and combined setup
weights; the offloading protocol alone determines how those messages are split
into stages. Compression introduces no second stage model and does not change
the protocol organization when offloading is disabled.

In the current implementation both compression obligations enter the fused
`akita_stage2` sumcheck. `akita_stage1` remains unchanged: it proves the generic
digit-range polynomial and returns `(s_claim, r_virt)`. Let

```text
S(w)        = w(w + 1),
p_base(y_A,x_A) = a_dA(y_A) m_base(x_A),
p_cmp(z)    = sum_K p_K(z), one sparse provider per native row family K,
eq_bin(X)   = MLE_X [z |-> 1_I_bin(z) eq(r_virt, z)].
```

The existing rows keep their current `d_A` factorization and `a_dA` table.
Provider `K` interprets only its own semantic span at native dimension `d_K`
and uses its own

```text
a_dK = (1, alpha, ..., alpha^(d_K-1)).
```

It then places those Boolean weights at the corresponding addresses `z` of the
canonical committed witness. The shared Stage-2 coordinate is only an address
space; it is not a claim that every row has the same native `(y,x)` axes or the
same `a` length. Stage 2 proves exactly

```text
gamma s_claim + relation_claim + trace_claim
 = sum_z (gamma eq(r_virt,z) + rho_bin eq_bin(z)) S(W(z))
   + sum_z W(z) (p_base(z) + sum_K p_K(z))
   + trace_oracle.
```

The added binary claim is zero, so it does not change the input claim. The
coefficient of that claim is `rho_bin`, not `gamma*rho_bin`; otherwise the
binary obligation would disappear whenever `gamma = 0`. Both added terms have
individual degree at most three, so Stage 2 retains its current round count,
degree bound, and proof bytes.

#### Compression rows in the relation matrix

For each commitment identity, the scheduled rows and their nonzero witness
support are:

| row family | nonzero witness spans | public right-hand side |
|------------|-----------------------|------------------------|
| B | source B digits, `xi_F,1`, B quotient | zero |
| D | source D digits, `xi_H,1`, D quotient | zero |
| `F_j`, `j < L_F` | `xi_F,j`, `xi_F,j+1`, F_j quotient | zero |
| `F_L` | `xi_F,L`, F_L quotient | terminal F payload |
| `H_j`, `j < L_H` | `xi_H,j`, `xi_H,j+1`, H_j quotient | zero |
| `H_L` | `xi_H,L`, H_L quotient | terminal H payload |

The B and D rows are the existing source rows augmented by the first gadget
recomposition term; they are not duplicated. Each nonterminal F/H row is
`K_j xi_j = G_j xi_j+1` before quotient lifting. Each terminal row is
`K_L xi_L = payload`. The quotient span represents the unique lift through
`X^d+1` at that row family's native dimension. Signs in the column provider
are derived once from the canonical identity
`M w = y + (X^d+1) r`; quotient construction, prover weights, and verifier
evaluation must consume that same provider.

The existing B/D setup-matrix contributions and their native `a_dB`/`a_dD`
factorizations remain on the current relation path. Only the new
`-G xi_F,1` or `-G xi_H,1` support is supplied sparsely at the corresponding
B/D native dimension. Later F/H rows are entirely compression providers. This
keeps the existing ring-relation implementation intact while allowing every
new row family to select an independently smaller ring.

These families occupy the row order already fixed above:

```text
consistency | A | B groups | D |
(F_j groups | H_j) for j = 1..max(L_F,L_H) | evaluation trace.
```

An absent group/layer contributes zero semantic rows and cannot reorder the
surviving families; padding is derived only after the live schedule rows.
`tau1` weights are taken at these authenticated row offsets. Only the last row
of each present chain reads a public payload.

Do not append compression columns to the dense
`relation_matrix_col_evals_compact: Vec<E>`. Compile their row-batched weights
into a compact object whose runs refer to checked semantic witness spans:

```rust
struct SparseRelationWeights<E> {
    // Sorted, disjoint live Stage-2-coordinate runs; no padded zeros.
    runs: Vec<SparseWeightRun<E>>,
}

struct CompressionRelationProvider {
    row_span: RowSpan,
    input_span: WitnessSpan,
    successor_span: Option<WitnessSpan>,
    quotient_span: Option<WitnessSpan>,
    setup_view: SetupView,
    native_ring_dim: usize,
}
```

The exact Rust names may differ, but the provider is the sole authority for
quotient construction, row-batched prover weights, direct verification, and
offloaded setup weights. Construction validates sorted nonoverlapping spans,
row bounds, native-ring divisibility, quotient presence, and the terminal RHS
before allocating weights. Overlapping runs from different row families are
summed into one sparse accumulator so the witness is not rescanned per map.

At construction, each provider emits weights only for the flattened Stage-2
cells touched by its input, successor, and quotient spans. This full-coordinate
representation decouples native compression rings from the existing relation
factorization. For flat coefficient offset `ell` within a native `d_K` span,
the provider uses

```text
x_K = floor(ell / d_K),
y_K = ell mod d_K,
coefficient weight = alpha^y_K * native column weight at x_K,
```

then maps `ell` through the checked semantic layout to the committed-witness
address. It never pads `a_dK` to length `d_A` and never interprets the span with
`y_A`. In every round, sibling
sparse weights `p_0,p_1` are paired with witness values `w_0,w_1`; the prover
adds the coefficients of `p(t)w(t)` and binds
`p' = p_0 + r(p_1-p_0)`. This costs `O(s_cmp,r)`, with
`s_cmp,r+1 <= ceil(s_cmp,r/2)`. No operation is proportional to the padded
witness width merely because compression is enabled. A provider may retain its
native `a_dK(y_K)m_K(x_K)` factorization while constructing round zero when that
is cheaper, but the folded sparse state has the same full-address semantics.

The compact relation contribution is computed as a separate coefficient array
and added to the existing relation round polynomial. It must be included in all
current Stage-2 execution paths: compact and full dense terms, prefix-y,
prefix-x, the fused next-prefix cache, `round2_prefix`, and the two-round
bivariate skip proof. Disabling the two-round prefix when compression is
present is not an acceptable implementation because it converts a sparse
protocol addition into a global prover regression.

#### Sparse binary-support prover

The descriptor derives `I_bin` as sorted, disjoint complete digit spans in the
flattened Stage-2 Boolean order. A checked constructor rejects overlap,
out-of-range endpoints, partial map spans, padding, and any negative-binary map
whose complete input span is absent. It initializes only

```text
p_0[z] = rho_bin eq(r_virt,z),  z in I_bin,
```

using interval equality recurrences; it does not construct a length-`2^m`
indicator or equality table. The ordinary `GruenSplitEq` state continues to
represent `gamma eq(r_virt,z)`. A `RestrictedEqState` stores sorted active
`(index, weight)` runs for `p_r`. In each ordinary round it:

1. visits only sibling pairs touched by a nonzero `p_r` entry;
2. reads the corresponding `w_0,w_1` from the already-live witness table;
3. adds the coefficients of
   `p_r(t) * w(t) * (w(t)+1)` to the round polynomial; and
4. binds `p_(r+1)[i] = p_r[2i] + r(p_r[2i+1]-p_r[2i])`, merging siblings and
   dropping zero runs.

Thus memory is `O(|I_bin|)`, the first-round work is `O(|I_bin|)`, and active
support never grows. This state is orthogonal to `GruenSplitEq`: changing the
global split-equality kernels to branch on membership across the entire witness
would violate the support-proportional requirement.

The two-round prefix path needs a sparse bivariate contribution built from the
same support runs. It reconstructs the binary portions of rounds zero and one,
then binds the sparse state at both challenges during the existing handoff.
Likewise, every cached/fused next-round path must cache the sum of base,
compression-relation, and restricted-binary coefficients. The implementation
may factor this through one canonical `round_poly`/`bind` sparse-state API; it
must not add separate copies of the protocol formula to each optimized kernel.

#### Verifier and transcript

After Stage 1 has transcript-bound `s_claim` and `r_virt`, the transcript draws
`gamma` and then, iff the descriptor-derived `I_bin` is nonempty, `rho_bin`
under distinct labels. The descriptor and therefore this branch were absorbed
earlier. A terminal cleartext fold, which validates its scheduled binary spans
directly, draws neither challenge and does not create a vacuous restricted
state. Verifier offloading does not alter this ordering; it only changes how
the already-fixed relation/setup claim is discharged.

At the final Stage-2 point `r_2`, the verifier evaluates

```text
eq_aug = gamma eq(r_virt,r_2)
       + rho_bin sum_(z in I_bin) eq(r_virt,z) eq(r_2,z)
```

and returns `eq_aug W(r_2)(W(r_2)+1)` for the virtual oracle. The restricted
sum uses the canonical `eval_offset_eq_interval` machinery (extended once if a
two-point factor API is needed) over the descriptor's interval union; it never
materializes `I_bin`. Complexity is `O(k_bin * m)` for `k_bin` spans and `m`
Stage-2 variables, with power-of-two subcubes contracted in constant work per
boundary node.

For the relation oracle, `RelationMatrixEvaluator` computes

```text
p_relation(r_2) = a(r_y) m_base(r_x)
                + sum_provider provider.eval_tau_weighted(r_2, tau1, alpha).
```

Each compression provider evaluates its input setup view, gadget-successor
span, and quotient span directly from their offsets and native dimensions.
The direct setup path folds all active logical setup weights onto one shared
prefix scan; the offloaded path commits to the identical combined weights.
Non-setup verifier work is proportional to the number of maps, native rows,
and support boundaries; setup-dependent work is one scan of the maximum active
flat prefix, never a sum of role footprints and never the global witness length.
`AkitaStage2Verifier::new` validates all counts and points before evaluation,
and every interval arithmetic failure returns `AkitaError` rather than indexing
or allocating unchecked.

### Conservative and multi-group commitments

Standalone commit generation freezes:

```text
group geometry
A/B keys and d_A/d_B
b_F = 4
largest permitted later opening base used for B/F1 security
all F map plans and SIS certificates
F digit widths and public payload shape
```

At opening time, an opening-base-first plan checks `b_1 >= 4` and re-evaluates
the frozen B/F1 collision bounds for actual `b_1`; a binary-first plan has no
compression-base compatibility check. Neither path silently replans or changes
payload length. The D/H chain is generated from the combined opening schedule
under the same per-map alphabet rules.

Multi-group relation batching is heterogeneous: each F plan evaluates its own
native ring rows and setup prefix view, then the shared relation challenge
batches the resulting claims. Compatibility means the claims can be batched at
the same evaluation point, not that their commitments or matrices must be
concatenated.

### Planner search and fixed point

Compression cannot be appended after the existing schedule search. Its digit
segments enlarge the recursive witness, its native relation rows enlarge the
quotient tail, and both can change later fold shapes and sumcheck rounds. Every
candidate must therefore solve the compression plan inside the same recurrence
that derives its successor witness and proof cost.

The planner policy specifies allowed native dimensions, ranks, depth range, and
an objective or cap for terminal payload bytes. Payload size is never a
standalone verifier rule: schedule validation recomputes the terminal shape,
exact wire length, and SIS certificate, and accepts precisely when those facts
are mutually consistent and meet the configured security floor.
Bind these choices into the policy digest and generated catalog identity. Candidate
generation must include both depths and both permitted first-map alphabets;
later maps are negative binary. Candidates with the same terminal payload are
compared on setup envelope, verifier scan, logical matrix footprint, witness
growth, and prover work.

Catalog selection is a deterministic function of authenticated public geometry
and the versioned policy. It is independent of the sampled setup coefficients,
and the proof or prover hint cannot carry a free alternative schedule
descriptor. A verifier derives the unique catalog entry before interpreting
schedule-dependent bytes. If a future API permits a choice among entries, its
security theorem and estimator must first price the resulting finite-union
loss; arbitrary post-setup schedule choice is forbidden.

Candidate scoring is lexicographic:

1. completeness and the production security policy encoded by the checked-in
   SIS table (currently 138-bit standalone classical security); quantum cost is
   reported diagnostically unless a separate security PR changes policy;
2. minimum global compact setup prefix;
3. minimum total persistent prepared-cache bytes, summed over active native
   dimensions after catalog-wide envelope coalescing;
4. minimum sum of active per-level direct scans;
5. smaller recursive witness and prover work;
6. smaller remaining proof bytes.

The suffix dynamic program needs a
Pareto state or equivalent structured score: the global prefix is a maximum
across levels, transformed NTT-cache storage is a sum across distinct active
dimensions, verifier work is closer to a sum, and none can be faithfully
converted into proof bytes.

### Shared-prefix security statement

All logical maps reuse correlated prefixes of one seed-expanded matrix, as A,
B, and D already do. The paper's first-differing-relation extraction lemma
already covers fixed heterogeneous prefix overlaps: it identifies the first
failed layer and reduces it to that layer's standalone MSIS instance without
claiming that the matrices are independent. The implementation audit must
cross-check that its exact map order and prefix views instantiate that lemma
and that deterministic catalog selection above makes the schedule fixed and
setup-independent. No additional independence assumption is permitted.

## Implementation authorities and dependency graph

```text
generated schedule catalog + field/security policy
    -> ValidatedCompressionCatalog
        -> CompiledCompressionSemantics
        -> SemanticRelationLayout
            -> RelationRowPlan
            -> derived binary support
        -> PreparedNttPlan
        -> prover-only CompressionExecutionBatch
```

`ValidatedCompressionCatalog` is the sole authority for map order, native
dimension, rank, alphabet, SIS certificate, terminal omissions, and frozen
group choices. It has private fields and one checked constructor in
`akita-types`; no weaker raw or "certified" constructor exists.

`CompiledCompressionSemantics` is a non-executable, compression-local
projection: flat-coefficient F/H input and quotient spans, layer-major local
F/H row spans, typed B/D augmentation intents, and the ordered input-segment
identities whose authenticated maps are negative binary. It stores no local or
global support runs: the global compiler translates those identities once and
derives the normalized support. It owns no absolute global witness or
`tau1` row offset, padding, trace placement, physical chunk ownership, copied
B/D source span, or second B/D quotient. A witness span has no intrinsic native
ring dimension: the consuming relation-row view supplies the dimension, and a
successor span may be interpreted under both adjacent row dimensions after
checked divisibility. Quotient depth comes from the active proof level, not a
standalone catalog's conservative maximum opening base.

`SemanticRelationLayout` is the sole authority for stable row identities,
semantic witness addresses, quotient spans, omissions, and binary support.
`RelationRowPlan` is its public-data projection consumed by prover, verifier,
direct setup evaluation, and verifier offloading. Consumers refer to semantic
identities rather than recomputing offsets. `PreparedNttPlan` is a
performance-only projection and cannot change protocol meaning.
`CompressionExecutionBatch` is ephemeral prover state; batching and fusion may
change it freely without changing the catalog, transcript, or proof.

Hints are data, never authority. They carry a catalog identity plus secret
digits and recomposition data, but do not repeat dimensions, alphabets, row
offsets, support, or SIS parameters. Their identity and lengths are validated
against the catalog before allocation or use.

Implementation rules:

- **Enforce:** checked private plan/layout types; migrate existing A/B/D
  consumers to `RelationRowPlan` before adding F/H; one sparse folding engine
  providing round polynomials, binding, and two-round grids; and real traits
  only at backend or serialization boundaries.
- **Encourage:** pure compiled projections, semantic IDs, test-only dense
  oracles, and one internal multi-RHS arithmetic engine shared by single and
  batched calls.
- **Discourage:** single-use indirection, role-specific cache vocabulary, and
  helper types that merely rename an existing checked value.
- **Forbid:** thin `_for_level`, `_checked`, or `_certified` forwarding
  wrappers; copied gadget, row, support, or key formulas; serialized derived
  offsets/support; role/map identities in NTT cache keys; strategy flags in
  protocol types; caller-selected cache keys or lazy warming; separate single
  and batched kernels; production raw/dense/partial compression modes; and
  verifier-reachable panics or unchecked indexing.

A preparatory `pub(crate)` authority may land dormant behind tests, but no
parallel public authority may land. When a new authority replaces an existing
one, all consumers migrate and the old authority and temporary adapters are
deleted in the same implementation slice.

## Architecture and code map

The following is the expected blast radius on current `origin/main`. Exact file
splits may change, but ownership must not drift across crates.

| Concern | Current anchors | Required direction |
|---------|-----------------|--------------------|
| Dimensions/roles | `akita-types/src/layout/ring_dims.rs` | retain A/B/D dims; add checked compression map dimensions beginning at d=8 |
| Level/schedule metadata | `akita-types/src/layout/params.rs`, `schedule.rs` | first-class depth-two/three F/H plans with per-map alphabets; freeze group plans |
| SIS sizing | `akita-types/src/sis/`, `akita-sis-estimator/` | use the existing canonical lookup and conservative coefficient-bound bucketing; a separate security PR owns model changes and table regeneration |
| Dispatch | `akita-types/src/dispatch/{mod,policy}.rs` | compression slot/path independent of fold-challenge minima |
| NTT cache | `akita-types/src/ntt_cache.rs`, `akita-prover/src/kernels/crt_ntt.rs`, backend prepared-setup contract | add D8, compile catalog-wide per-dimension envelopes, and cache each cyclic/negacyclic pair once |
| Compression kernels | `akita-prover/src/kernels/linear/fused_quotients.rs`, CRT/NTT helpers, compute backends | refactor the existing A/B/D tiler into one internal multi-RHS engine; compression uses the same paired transforms, safe-width chunking, and quotient primitive |
| Setup envelope | planner `matrix_envelope.rs` | flat-coefficient maximum over all active views |
| Flat setup views | `akita-types`/`akita-pcs` matrix and setup modules | all roles start at coefficient zero; no cursor |
| Witness | `akita-types/src/witness.rs`, prover hints | checked compression spans and binary support derivation |
| Relation prover | `akita-prover/src/protocol/ring_relation*` | native-role providers, sparse F/H logic, per-role quotients |
| Ring switch | `akita-types/src/proof/relation_matrix_cols.rs`, prover finalize | stop requiring one dense uniform compression-column vector |
| Range proof | prover `sumcheck/akita_stage1` | unchanged generic range claim and `r_virt` output |
| Fused Stage 2 | prover `sumcheck/akita_stage2/{lifecycle,round_flow,dense_terms,x_prefix,y_prefix,round2_prefix,two_round_prefix}`, verifier `stages/stage2.rs` | sparse restricted-equality state plus sparse compression-relation weights in every optimized path |
| Relation verifier | `akita-verifier/src/protocol/ring_switch.rs`, `stages/stage2.rs` | validate and evaluate all F/H equations and native quotients |
| Setup offload | `setup_contribution`, verifier offloading path | consume generalized providers and the same shared-prefix claim without changing the compression protocol |
| Proof schema | `akita-types/src/proof/{commitment,levels,shapes,hints}.rs` | compressed payloads and exact size checks |
| Commit/open APIs | `akita-pcs/src/scheme`, prover compute/ring switch | mandatory chains, frozen conservative plans, no fallback |
| Profiles/tests | `akita-pcs/examples/profile`, scheme tests | payload/setup/scan reporting and full preset coverage |

### Concrete small-ring arithmetic implementation surface

The small-ring work is a vertical slice; adding `8` to one generic dimension
list is neither necessary nor sufficient. These authorities must agree:

- `akita-types/src/layout/ring_dims.rs` checks only A/B/D structural invariants,
  including the protocol-wide B/D minimum `32`; it owns no broad supported-dims
  list. Compression and NTT support live in their purpose-specific dispatch
  policies.
- `akita-types/src/dispatch/{mod,policy}.rs` adds the `Compression` slot and
  derives compression and NTT arms per field tier. `ntt_min_ring_d` is derived
  from the NTT slot itself, not from the B/D role policy. Synchronization tests
  exhaustively compare each role, compression, NTT, and envelope runtime
  dispatch against its policy predicate without another hand-maintained union.
- `akita-prover/src/kernels/crt_ntt.rs` adds `NttSlotCacheAny::D8`. Its cache
  constructor builds both transforms for exactly the requested flat prefix and
  retains the existing CRT-capacity partitioning. The cyclic and negacyclic
  tables are one logical cache slot and are accounted together.
- `akita-prover/src/compute/{backend,cpu,delegating_cpu,stack}.rs` exposes one
  batched compression operation taking a plan-resolved cache view and checked
  same-shape buckets. CPU and delegated backends dispatch the same dimension
  arms exactly once at the outer backend boundary; a delegating backend forwards
  the already selected operation without redispatch. A one-item batch is the
  canonical single-map path. The operation does not call
  `with_shared_ntt`, because that helper requests the full setup envelope, and
  it does not compose two public backend calls that each scan and transform the
  digits independently.
- `akita-prover/src/kernels/linear/fused_quotients.rs` owns one internal
  multi-RHS engine used by the existing A/B/D flow and compression. It shares
  setup loads across eligible right-hand sides and shares digit loading,
  tiling, safe-width chunk boundaries, and CRT reconstruction between the
  requested cyclic and negacyclic outputs. The returned canonical object is a
  checked batch of requested domain images; the quotient derivation consumes
  those images or a descriptor-derived known negacyclic RHS. Existing quotient
  helpers are consolidated behind one arithmetic primitive; compression must
  not add a second CRT tiler, quotient formula, or separate single-item kernel.
- This PR changes no generated SIS data. Planner construction and verifier
  validation call the existing `sis_table_key_for_linf_bound` and minimum-rank
  authority, including its conservative rounding of actual bound one to the
  shipped bound-two bucket. Generated schedule catalogs contain only candidates
  accepted by that authority; neither side substitutes a local estimate.
- The schedule compiler computes one compression envelope across every layer
  and identity in the authenticated catalog for each active `d`, coalesces it
  with any longer existing role-cache requirement, and passes the checked plan
  to backend preparation. The backend eagerly builds the resulting slots;
  `AkitaProverSetup` itself remains backend-state-free. Commitment execution
  consumes F slices; opening execution consumes F/H slices; recursive levels
  consume only slices authorized by their authenticated schedule. An unplanned
  lazy cache build is a profile/test diagnostic failure.

The arithmetic test matrix has three layers. Kernel tests compare the fused
result with direct schoolbook cyclic and negacyclic multiplication for signed
digits and opening-base digits. Backend tests compare each allowed runtime arm
with the const-generic kernel and verify exact-prefix cache reuse. Protocol
tests exercise mixed dimensions in one proof, reject a dimension accepted by a
different purpose or tier, and reject missing SIS or NTT capability before
allocation.

### Concrete Stage-2 implementation surface

The first implementation should follow this cut line; moving a responsibility
requires preserving the same single source of truth.

- `akita-types/src/witness.rs` and the schedule/layout modules construct checked
  semantic F/H input, successor, quotient, row, and binary-support spans. They
  expose no physical offset computed independently by prover or verifier.
- `akita-types/src/proof/relation_matrix_cols.rs` stops returning one monolithic
  compression-extended vector. Its canonical result contains the existing base
  weights plus `SparseRelationWeights` compiled from the relation providers.
  Dense expansion exists only under tests.
- `akita-prover/src/protocol/ring_relation*` uses those same providers to build
  each native quotient and terminal RHS. It must not independently reconstruct
  gadget signs or row offsets.
- `akita-prover/src/protocol/core/fold.rs` absorbs `s_claim`, samples `gamma`,
  conditionally samples `rho_bin`, and passes the transcript-bound support and
  both sparse states into Stage 2. Terminal cleartext flow performs direct span
  validation and skips these challenges as specified above.
- `akita-prover/src/protocol/sumcheck/akita_stage2/lifecycle.rs` validates and
  owns `RestrictedEqState` and `SparseRelationState`. The existing
  `GruenSplitEq::with_initial_scalar(stage1_point, gamma)` remains the generic
  component; the Stage-2 input claim remains
  `gamma*s_claim + relation_claim + trace_claim`.
- `dense_terms.rs`, `x_prefix.rs`, and `y_prefix.rs` compute the existing base
  polynomials and add the two sparse states through one canonical coefficient
  API. `round_flow.rs` binds both states on every challenge and includes them in
  every cached next-round polynomial.
- `round2_prefix.rs` and `sumcheck/two_round_prefix/stage2.rs` add the sparse
  bivariate grids and perform the same two-challenge handoff as the base terms.
  They may call state methods but may not reimplement support derivation or row
  formulas.
- `akita-algebra/src/offset_eq.rs` owns any required two-point restricted
  interval contraction. Prover initialization, verifier final evaluation, and
  tests call this one primitive.
- `akita-verifier/src/protocol/core/fold.rs` mirrors the challenge schedule and
  rejects descriptor/support disagreement before constructing Stage 2.
  `akita-verifier/src/stages/stage2.rs` stores `rho_bin` and the checked support,
  uses `eq_aug` in `expected_output_claim`, and keeps degree bound three.
- `akita-verifier/src/protocol/ring_switch.rs` and setup-contribution modules
  evaluate the same compression providers in direct and offloaded modes. The
  former returns the sparse compression contribution at the final full Stage-2
  point; the latter folds identical setup weights into the shared prefix.
- Transcript-label modules add exactly one conditional binary-batching label;
  fixtures pin absent/present support, direct/offloaded verification,
  recursive/multi-group schedules, and terminal omission.
- Stage-2 unit tests compare every optimized path with a dense full-coordinate
  oracle. Cross-crate tests tamper each row/support/payload role. Benchmark code
  implements the two controls and counters specified below without exposing a
  production opt-out.

## Evaluation

### Acceptance criteria

- [ ] Every shipped q128/q64/q32 schedule contains an explicit depth-two or
  depth-three F/H chain where its B/D commitment exists; terminal omissions are
  explicit.
- [ ] Every public payload is independently deserialized with the
  schedule-derived exact coefficient count; shipped q128/q64/q32 fixtures select
  the displayed rank-one shapes and consequently encode to 128 bytes.
- [ ] This PR's diff contains no generated SIS-table changes. Every generated
  compression schedule uses the unchanged canonical lookup, including its
  conservative coefficient-bound bucketing, and rejects unsupported
  family/dimension/bucket/width combinations.
- [ ] Profiles report standalone classical and diagnostic quantum estimates for
  every B/D/F/H key. Only the security policy encoded by the checked-in table
  is an acceptance floor until the dedicated security review changes it.
- [ ] Opening-base-first conservative commitments freeze base four, reject
  later `b_1 < 4`, and validate frozen B/F1 sizing against actual later bases
  without replanning. Binary-first commitments have no such base dependency.
- [ ] Every negative-binary F/H map uses bound one only when its scheduled digit spans are
  covered by the verifier's binary obligation.
- [ ] A test exhaustively checks `w(w+1)=0` accepts only `{-1,0}` in every
  shipped field.
- [ ] Dense-reference tests show `omega_tilde` equals the pointwise weighted
  table for singleton, interval, union, empty, and boundary supports; the
  sumcheck degree sequence is unchanged.
- [ ] Tampering an F/H digit, intermediate image, final payload, binary support,
  alphabet/base, dimension, rank, or depth is rejected.
- [ ] Multi-group tests compress groups independently, allow heterogeneous F
  shapes, and reject swapped payloads or descriptors.
- [ ] W2/W4/W8 distributed-chain tests reduce one canonical raw image,
  intermediate image, terminal payload, and summed product quotient and match
  the corresponding single-machine values. Machine-local quotient witness
  segments and proof encodings need not be byte-identical to W1.
- [ ] Distributed semantic-layout tests shard F/H columns without requiring
  equal dimensions or exact divisibility, and derive binary support from only
  the real negative-binary F/H shards. Summed round polynomials and the lifted
  global relation vanish; individual worker residuals are not required to.
- [ ] A negative fixture rejects independently compressing partial machine
  images as if standalone terminal-map certification covered the repeated-column map.
- [ ] Compact matrix footprint tests verify `n*L` field coefficients and prefix
  reuse; no code path sums role footprints for allocation.
- [ ] Backend preparation receives the catalog-derived `PreparedNttPlan` and
  eagerly materializes exactly one cyclic/negacyclic cache slot per active
  dimension, sized to the maximum of the catalog-wide compression envelope and
  any existing role-cache requirement. `AkitaProverSetup` stores no backend NTT
  state. Commit/open execution has zero cache builds, and every shorter F/H view
  is served through the plan's checked slice resolver.
- [ ] Direct and recursive setup-contribution modes produce identical combined
  claims for mixed F/H dimensions.
- [ ] Relation-row layout tests pin the canonical order and terminal omissions.
- [ ] Every F/H relation includes the correct native-ring quotient at its own
  dimension. Dense arithmetic reference tests cover every compression-dispatch
  dimension, including the tier minima d=8/16/32; dimensions below the active
  tier minimum are rejected.
- [ ] Eligible equal-shape compression maps use the batched multi-RHS kernel;
  its outputs match independent cyclic/negacyclic reference calls, profiles
  account for shared setup loads, and transcript-dependent F/H or adjacent
  chain maps are never fused across their dependency barrier.
- [ ] Eager paired quotient construction and deferred F-cyclic completion
  produce identical quotient digits and proof bytes. The q128/q64/q32 1-KiB
  F/H benchmarks select the faster internal strategy and report setup traffic,
  coefficient reductions, pointwise products, and peak accumulator memory.
- [ ] No raw-commitment or compression opt-out flag exists in configs,
  generated catalogs, public APIs, proof enums, profiles, or tests.
- [ ] The final recursive `u` is absent from the wire and transcript.
- [ ] Malformed-proof fuzz/property tests cover overflow, unsupported dims,
  invalid divisibility, oversized ranks/supports, bad payload lengths, and bad
  chain metadata without verifier panic or unbounded allocation.
- [ ] Proof-size accounting and profile output separately report terminal F/H payloads,
  compression witnesses, relation proof, setup prefix, and live setup scan.
- [ ] Direct verification and verifier-offloaded verification accept the same
  F/H relation and setup claims; enabling offloading changes only the offloading
  message schedule, not compression descriptors, payloads, or equations.
- [ ] Paired release benchmarks satisfy the 5% hard prover and verifier limits
  for both Stage 2 and end-to-end workloads; reports include the 2% target,
  confidence intervals, and support-scan counters.
- [ ] Every optimized Stage-2 path, including the two-round prefix and cached
  fused-next-round paths, matches one dense reference with binary and relation
  supports at the first/last index, across pair boundaries, overlapping after

### Testing strategy

Add small deterministic unit tests at each canonical primitive, then cross-crate
end-to-end tests. Arithmetic tests compare compact/sparse implementations with a
dense scalar reference. Transcript fixtures cover scalar, recursive,
multi-group, disk-backed, direct-setup, and recursive-setup paths. Each malformed
descriptor field gets a verifier-boundary rejection test.

The mandatory gate is:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
rtk cargo nextest run --profile ci --no-default-features --features parallel,disk-persistence
./scripts/check-doc-guardrails.sh
```

Run relevant nondefault feature/profile matrices already exercised by CI; the
protocol is incomplete if a test or benchmark needs a raw-commitment fallback.

### Performance and reports

Performance is a release gate, not an advisory. For each shipped field, the
representative 1/4/8-KiB workloads, W1 and supported distributed worker counts,
and both direct and verifier-offloaded setup modes, measure paired runs on the
same pinned machine, feature set, thread count, schedule catalog, and public
input. Planner/setup generation and cold filesystem I/O are reported separately.
Prover time means commitment/opening computation plus proof generation;
verifier time means checked deserialization, schedule validation, and proof
verification.

For both prover and verifier, the median enabled/control ratio must be at most
`1.05`; `1.02` is the optimization target. Use at least ten warmups and thirty
paired samples, with each reported sample batched long enough to exceed timer
noise. Report the median, p95, and a bootstrap 95% confidence interval. A noisy
CI does not relax the median gate; the benchmark must instead increase its
batch duration.

Two controls are required:

1. a Stage-2 microbenchmark over the same expanded witness and relation layout,
   comparing the complete sparse binary/relation additions with a
   benchmark-only arithmetic control that omits those zero/additive terms; and
2. an end-to-end benchmark at the same application input and security target,
   comparing the planner-selected compressed schedule with a benchmark-only
   source-commitment control.

These controls are harness internals, not protocol configurations: they cannot
serialize proofs, enter generated catalogs, or create a raw-commitment fallback.
The Stage-2 gate isolates implementation quality; the end-to-end gate prevents
the planner from hiding excessive commitment or verification work behind a
fast sparse sumcheck. A shipped planner fixture that exceeds 5% on either side
is rejected or replanned. Results between 2% and 5% require an explicit profile
showing the remaining dominant kernels.

Instrumentation must also report, per Stage-2 round, base witness cells,
compression-relation cells, binary-support cells, and sparse-state entries
folded. Tests assert that the last two counters are bounded by their live
supports and never by the padded witness length. Add dense-reference benchmark
variants only for correctness and diagnosis; they are excluded from production
measurements.

Add a reproducible arbitrary-chain sweep to `akita-sis-estimator` or the planner. For
incoming native images of 1, 2, 4, and 8 KiB and every shipped field/base, report:

- every F/H map's `(d,n)`, digit width, and standalone classical/quantum security;
- intermediate-image bytes and the planner-selected terminal payload;
- compact matrix field coefficients and bytes;
- maximum flat persistent prefix, summed prepared NTT-cache bytes by dimension,
  and per-level live scan;
- estimated direct-verifier field additions/multiplications;
- diagnostic hypothetical B/D slicing envelopes, clearly marked as non-protocol estimates.

The q128 4 KiB table above and the shipped q128/q64/q32 payload fixtures are
checked regressions. The verifier expectation is at most a few tens of
thousands of field multiplications for typical F/H views; if measurement
materially exceeds that, optimize the shared-prefix/sparse evaluator or let the
planner select a different admissible chain.

## Execution plan

Implementation proceeds only after this proposed spec is approved.

0. **Reuse existing regression authorities.** Confirm coverage from schedule
   drift tests, exact proof-size-versus-serialization tests, transcript/proof
   fixtures, SIS goldens, and merge-base profile benchmarks. Add only missing
   semantic or dense-oracle coverage needed to make later slices reviewable.
   Temporary pre-cutover snapshots must be labeled and deleted in Slice 8;
   durable tests assert protocol invariants rather than duplicating whole stale
   artifacts.
1. **Arithmetic capabilities.** Add tier-specific compression/NTT execution
   dispatch through d=8 and synchronization tests. Generate compression
   schedules using only the existing canonical SIS keys and conservative bound
   buckets. No generated SIS table changes here; a later security-model PR may
   add coverage and regenerate the affected schedules after separate review.
2. **Compile compression-local semantics.** Add the private minimal input spec,
   `validate_and_compile`, flat F/H input and quotient spans, local row spans,
   B/D augmentation intents, and derived negative-binary segment provenance as dormant
   `pub(crate)` authorities with malformed-input tests. Do not assign global
   witness offsets or absolute A/B/D/trace row offsets in this slice.
3. **Compile and migrate the global semantic relation.** Move—not copy—the
   existing z/e/t/r resolver and A/B/D relation construction into the sole
   `SemanticRelationLayout` and embedded `RelationRowPlan`; translate the
   negative-binary segment identities into the one normalized global support,
   compose the checked
   compression-local semantics, migrate setup contribution and every existing
   consumer, then delete the uniform-layout/offset authorities. Existing proof
   bytes remain unchanged.
4. **Unify arithmetic and preparation.** Refactor `fused_quotients` into the
   shared multi-RHS core, consolidate quotient derivation, and add
   `PreparedNttPlan` to the canonical backend preparation boundary. Migrate
   A/B/D first, prove zero lazy builds, then exercise dormant F/H maps.
5. **Unify sparse folding.** Introduce one restricted-equality/sparse-relation
   engine with round polynomial, bind, and two-round-grid operations. Migrate
   every optimized Stage-2 path with empty sparse state; proof bytes remain
   unchanged and each path matches the dense oracle.
6. **Compile schedules and hints.** Compile the deterministic catalog supplied
   by the dedicated schedule-generation PR into envelopes, summed cache-memory
   reports, frozen group choices, and validated F/H hint data. Planner/type code
   may change here, but generated table/catalog files do not. Exercise full
   chain arithmetic internally without exposing an alternate public encoding.
7. **Internal compressed proof harness.** Wire F/H relation providers, native
   product quotients, derived binary support, direct/offloaded evaluation, and
   prover/verifier sparse folding under `cfg(test)`. Establish dense-oracle and
   tamper equivalence before changing the public wire.
8. **Atomic local protocol cutover.** In one review stack, update commitment and
   proof schemas, transcript, commit/open, verifier, recursive and terminal
   flows, multi-group handling, and disk persistence. Delete raw B/D public
   semantics and every temporary adapter in the same stack; a partially wired
   production W1 mode must not land.
9. **Distributed follow-up.** After the semantic authorities merge, compose
   machine ownership and aggregate-only compression residuals on the companion
   branch. Test summed round polynomials and global lifted relations, never
   worker-local vanishing.
10. **Production gates and docs.** Run full CI/profile sweeps, enforce the 5%
    gates, update proof/setup/cache reports, fold durable exposition into the
    book, and advance/archive specs according to `specs/PRUNING.md`.

Every slice must compile, pass its focused and guardrail tests, and preserve one
canonical function per concept. A dormant internal authority is acceptable; a
second public planner, layout, kernel, cache selector, or wire mode is not.
Temporary adapters are removed in the same slice that migrates their final
consumer.

## Alternatives considered

### Global L2/JL certificate

Greyhound/Labrador/Rokoko-style global norm certificates do not identify the
small subset of compression digits whose alphabet must be tightened. A new local
certificate would still be required. Akita already has a per-digit range proof,
following Hachi, so the necessary locality is native to the protocol and the
binary subset adds cleanly to its optimized sumcheck.

### Standalone binary sumcheck

Correct but unnecessary. Fusing the support-weight table with the existing
digit-range virtualization reuses its quadratic factor and adds neither a new
sumcheck nor a degree. It also keeps the binary positions in the uniform generic
range pipeline.

### One compression layer or fixed depth

Rejected. Even with an exact negative-binary input bound, a direct q128 map to
128 bytes has approximately 116, 94, and 76 classical bits for 1, 2, and 4 KiB
inputs. After one generic compression map, a 512-byte intermediate maps to 128
bytes at only 141 classical bits and approximately 128 quantum bits, while a
1 KiB intermediate remains below target. The natural default therefore inserts
a 256-byte image and uses

```text
1–4 KiB -> 512 B or 1 KiB -> 256 B -> 128 B.
```

The last arrow has approximately 169 classical and 153 quantum bits in the ADPS
model. Chain depth remains planner-generated because smaller inputs and other
fields may reach the same target with a different secure sequence. The protocol
does not negotiate or alter depth after the schedule is frozen.

### Scalar-only terminal maps

Rejected. They throw away compact ring storage and make terminal setup matrices
unnecessarily large. The shipped rank-one native-ring choices give a 128-byte
endpoint at `d = 8,16,32` over q128/q64/q32 and materially reduce setup/verifier work.

### Disjoint setup offsets

Rejected. Seed-expanded matrices are pseudorandom prefix views; disjoint cursors
sum allocation and verifier scans without a security benefit under the existing
shared-setup proof model.

### Concatenate multi-group commitments

Rejected. Groups are created and frozen independently, sometimes at different
times. Concatenation changes commitment identity and later-opening semantics.
Batch their relation claims, not their transmitted commitments.

## Documentation

After implementation stabilizes, fold the protocol explanation into
`book/src/how/optimizations.md`, the security contract into
`book/src/how/security.md`, sumcheck integration into
`book/src/how/proving/sumcheck-stages.md`, and planner tradeoffs into
`book/src/how/configuration.md`. Update `book/src/how/recursion.md` for terminal
re-anchoring and `book/src/how/verification.md` for descriptor/no-panic checks.
Until then this spec is the implementation source of truth.

## References

- Paper source and parameter discussion: private `paper-note` entry “lattice
  jolt akita”, Akita Sections 2 and 4.
- Setup prefix and offload: `specs/setup-prefix-ladder.md`,
  `specs/setup-product-sumcheck.md`, `specs/batched-stage3-setup-opening.md`.
- Terminal contract: `specs/terminal-fold-cutover.md`.
- Mixed-role baseline: `book/src/how/architecture.md` and
  `crates/akita-types/src/layout/ring_dims.rs`.
