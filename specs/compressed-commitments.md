# Spec: Compressed Commitments

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-11 |
| Status        | proposed |
| PR            | |
| Supersedes    | `specs/archive/2026-Q3/commitment-compression-cutover.md` |
| Superseded-by | |
| Book-chapter  | |

## Summary

Akita will replace every transmitted Ajtai commitment with a generated
compression chain of depth two or three. Each map recommits a scheduled gadget
decomposition of the preceding image. A map may use the opening-base alphabet
or the negative-binary alphabet `{-1, 0}`; production search permits the
opening-base alphabet only for the first map and uses negative binary
thereafter. The binary certificate is local to the scheduled binary digits and fuses
into the existing digit-range sumcheck without raising its individual degree.
The canonical payload is one rank-one terminal ring element: `d = 8`, `16`,
and `32` over q128, q64, and q32, respectively. It serializes to 128 bytes in
every field. The planner selects two or three maps, and every map is priced as
a standalone MSIS instance.

This is a mandatory protocol cutover, not an optional encoding. It includes
native mixed ring dimensions down to `d = 1`, prefix-shared setup matrices,
and schedule-frozen compression metadata for standalone and multi-group
commitments. B/D block-axis slicing is deferred from this PR. The spec also
records the intended sumcheck-stage reorder:
the complete relation sumcheck moves into the last part of stage 1, while the
setup product and carried-claim reduction become stage 2. The digit-range chain,
including the extra binary obligation, remains independent of that move.

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
or three maps in this cutover. “Layer” refers
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

The terminal recursive `u` is removed by terminal re-anchoring. It is not sent
raw and is not run through a redundant compression chain. The terminal relation
layout therefore omits both the B commitment output and D opening rows when the
cleartext terminal witness contract makes them unnecessary.

The F chain of the commitment entering the terminal step must still be checked.
If its compression digits are part of the cleartext terminal witness, decoding
enforces their generic range and directly validates `{-1,0}` on the scheduled
binary span; it does not run a vacuous binary sumcheck. The segment-typed
terminal layout must therefore name and length the compression segment
explicitly.

### Per-map alphabet contract

The first compression map may use either the opening base or negative binary:

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
- A negative-binary first map is independent of the later opening base. Its
  digit depth is the field bit width, its exact collision bound is one, and its
  support participates in the local binary certificate.
- Every later map is negative binary and is not sized conservatively over
  opening bases.

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

This polynomial vanishes exactly at `W(x) in {-1, 0}`. After the generic
virtualization claim is fixed, the transcript samples a fresh `rho_bin` and
batches (BIN) with the existing quadratic relation using the pointwise Boolean
weight

```text
omega_bin(x) = eq(r_virt, x) (1 + rho_bin 1_{I_bin}(x)).
```

The implementation must not multiply two separately represented multilinear
polynomials inside the sumcheck. Instead it constructs the single multilinear
extension of the pointwise Boolean table:

```text
omega_tilde(X)
  = eq(r_virt, X) + rho_bin eq_I_bin(r_virt, X),

eq_I_bin(r, X)
  = MLE_X [ x |-> 1_{I_bin}(x) eq(r, x) ].
```

The resulting term remains `omega_tilde(X) W(X)(W(X)+1)`: individual degree
two in `W` and degree one in every weight coordinate, exactly as before.
`I_bin` is a short union of schedule-known intervals or subcubes. The prover
stores only the nonzero portions. The verifier evaluates `eq_I_bin` by affine
interval equality contractions; it never allocates a dense support table.

### Invariants

1. **Mandatory encoding.** No supported config, test, CI mode, profile, or
   benchmark sends a raw B/D commitment or exposes a compression opt-out.
2. **Depth two or three.** Any other depth is malformed in this cutover. Every
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
9. **Native dimensions.** F/H dimensions are powers of two and may be
   `1, 2, 4, 8, 16` or any larger supported dimension. Scalar-only compression
   is explicitly rejected as an architecture.
10. **Canonical row order.** Relation rows are ordered
    `consistency | A | B groups | D | (F_j groups | H_j)_{j=1..max(L_F,L_H)} |
    evaluation trace`. Omitted roles contribute zero rows without reordering
    surviving roles.
11. **One quotient per native ring row.** Each nonscalar row uses a quotient in
    its own ring. Scalar `d = 1` rows need no negacyclic quotient.
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
- Backward-compatible proof or schedule decoding.
- A raw-commitment fallback or proof-time depth/alphabet negotiation.
- A multi-target SIS security claim.
- B/D block-axis slicing or restoration of the old “tiered commitment” feature.
  Slicing is a possible follow-up after distributed and multi-chunk schedules
  provide evidence that it reduces real setup envelopes.
- Mandating a fixed list of payload sizes. They are planner outcomes; only the
  shipped default and security floor are normative.

## Parameterization and performance model

### Default payload

The production planner targets 128 bytes across the shipped field widths by
halving field width while doubling the terminal ring dimension:

| field | coefficient bytes | terminal `d` | terminal rank | coefficients | bytes |
|-------|-------------------|-------|------------|--------------|-------|
| q128  | 16 | 8  | 1 | 8  | 128 |
| q64   | 8  | 16 | 1 | 16 | 128 |
| q32   | 4  | 32 | 1 | 32 | 128 |

The planner may select, for example, 192- or 256-byte payloads when the smaller
setup matrix and faster direct verifier justify the larger wire image. Such
alternatives must be produced by the same estimator and descriptor path.

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
estimator changes). The protocol-wide floor remains 138 classical bits until a
separate security-policy cutover, but the shipped 128-byte default must also
report at least 128 bits under the ADPS quantum Core-SVP model.

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

### Deferred B/D slicing

This PR keeps the existing unsliced block-fast B/D source geometry. It adds no
slice count, slice selector, or sliced witness layout to public descriptors.
The compression-chain input is expressed only as a checked flat source-image
coefficient count, so a later slicing implementation can feed the same
interface without changing chain semantics.

Planner reports may estimate hypothetical slice counts `2/4/8` for diagnostics,
but those estimates cannot alter schedules or proofs. A follow-up slicing PR
requires measurements from multi-chunk and distributed schedules showing that
the unsliced B/D view, rather than a compression map or another role, determines
the live setup envelope.

## Design

### First-class compression plan

Do not extend `CommitmentRingDims` with F/H fields. That type remains the
A/B/D fold-geometry contract. Replace parallel source-key and compression
metadata with one canonical compressed-commitment plan, approximately:

```rust
struct AjtaiMapPlan {
    key: AjtaiKeyParams,
    alphabet: CompressionAlphabet,
    digit_depth: usize,
    ring_dim: usize,
    input_coeffs: usize,
    output_coeffs: usize,
}

struct CompressionChainPlan {
    maps: Vec<AjtaiMapPlan>,
    digit_spans: Vec<WitnessSpan>,
    binary_support: Vec<WitnessSpan>,
}

struct CompressedCommitmentPlan {
    source: AjtaiMapPlan,
    chain: CompressionChainPlan,
}
```

Names may change during implementation, but these facts must remain explicit
and descriptor-bound. There is one B-side chain per commitment identity and one
D-side chain per opening schedule. `PrecommittedGroupParams` must freeze the
F chain in addition to its current geometry and conservative B
rank. `ExecutionSchedule`/`LevelParams` must carry the current H chain and the F
chain for the recursive commitment being created.

Use checked constructors and one validation routine called by planner output,
setup generation, deserialization boundaries, prover, and verifier. Do not add
thin `_for_level` wrappers or separate “certified” versus “executed” bounds.

### Small-ring execution and SIS tables

Current runtime roles and protocol dispatch bottom out at d=16, while production
SIS tables cover only d=32/64/128/256 and coefficient-bound buckets begin at
two. The cutover must:

1. extend the SIS estimator and generated production tables to exact bound one
   and dimensions `1,2,4,8,16`;
2. keep exact requested bounds in certification—do not round one up to two for
   any negative-binary layer;
3. add a compression execution dispatch independent of A-role sparse-challenge
   support and the current field-specific B/D minima;
4. provide a correct small-ring multiplication path when NTT/packed kernels do
   not support the dimension, with optimized specializations optional;
5. keep d=1 valid, while exercising mixed non-scalar defaults in every field.

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
worker offsets independently. Local compression rows sum to the global rows:

```text
sum_j (B_j t_j - G_b,j xi_F1,j)                 = 0
sum_j (F_ell,j xi_Fell,j - G_2,j xi_F(ell+1),j) = 0   for ell < L
sum_j  F_L,j xi_FL,j                            = u_pub.
```

The H rows are analogous. One schedule-derived machine owns the public RHS for
local quotient construction; the payload itself appears once in the proof and
transcript. Every machine carries a complete-shaped local quotient contribution
for each non-scalar row family at that family's native dimension.

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

This compression work owns `CompressionChainPlan`, semantic relation-row and
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
3. finish compression's recursive/multi-group cutover on the composed layout;
4. add the distributed process runtime only after the composed single-host
   prover and structured verifier pass end to end.

This ordering avoids freezing a `z/e/t/r`-only distributed public type and
avoids extending the old flat multi-chunk layout with global F/H tails that
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
A
B_1 ... B_G
D
(Fell_1 ... Fell_G
 Hell) for ell = 1..max(L_F,L_H)
evaluation trace
```

At recursive scalar levels `G=1`. At a multi-group root the frozen group chains
appear in group schedule order, followed by the newly committed group. The
evaluation trace remains last.

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

Every non-scalar native relation `y = K x` contributes its own negacyclic
quotient at that role's ring dimension. Quotient witness spans and row counts are
part of the descriptor. Checked conversions project a shared flat coefficient
view into a role ring; divisibility must be validated before projection.

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
this does not require planner-enforced monotonicity. If d=1 is active the scan
is simply a flat field-coefficient scan. The recursively committed setup prefix
may remain packed at its existing storage dimension; its logical claim must be
the same flattened prefix claim.

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

Descriptor bytes must bind, in stable order:

- both chain depths and every layer's base/digit depth;
- every F/H ring dimension, rank, input width, output width, and SIS table key;
- compression witness spans, row spans, and quotient spans;
- expected public payload coefficient counts;
- terminal role omissions and the binary-support derivation version.

Use new transcript labels for compressed payloads and `rho_bin`; do not reuse
the abandoned PR's optional-compression labels. Absorb the schedule descriptor
before any payload interpreted by it. Transcript tests must pin the order and
prove that changing any plan field changes the transcript.

### Sumcheck stage ownership

The target protocol has two conceptual chains:

1. the digit-range chain, including (BIN);
2. the linear relation chain, including all compression rows.

They remain decoupled. The relation rows move together into the final substage
of stage 1, after the large-basis digit-range substages. The current stage-3
setup product and carried stage-2 claim then become stage 2. Moving relation
ownership must not move (BIN): it remains attached to digit virtualization at
`r_virt`.

Implementation may land compression against the current three-stage internals
only as an explicitly temporary, transcript-breaking stack dependency. The
final PR cannot retain two public stage models. Proof types, shape accounting,
profile reporting, transcript labels, and book terminology must describe the
optimized two-stage contract.

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

The planner policy may specify allowed native dimensions, the mandatory
terminal-byte target, and the allowed depth range, but has no enable field.
Bind these choices into the policy digest and generated catalog identity. Candidate
generation must include both depths and both permitted first-map alphabets;
later maps are negative binary. Candidates with the same terminal payload are
compared on setup envelope, verifier scan, logical matrix footprint, witness
growth, and prover work.

Candidate scoring is lexicographic:

1. completeness and at least 138-bit standalone classical security, plus the
   selected preset's quantum floor (128 bits for the shipped default);
2. the 128-byte production target;
3. minimum global compact setup prefix;
4. minimum sum of active per-level direct scans;
5. smaller recursive witness and prover work;
6. smaller remaining proof bytes.

The current suffix dynamic program minimizes bytes alone. This cutover needs a
Pareto state or equivalent structured score: the global prefix is a maximum
across levels, verifier work is closer to a sum, and neither can be faithfully
converted into proof bytes.

### Shared-prefix security statement

All logical maps reuse correlated prefixes of one seed-expanded matrix, as A,
B, and D already do. The security writeup and code audit must extend the
existing first-differing-relation extraction argument to F/H views, including
heterogeneous prefix overlaps. It must identify the first failed
layer and reduce it to that layer's standalone MSIS instance; it must not claim
the matrices are independent. This proof obligation is required before the
shared-prefix implementation is approved.

## Architecture and code map

The following is the expected blast radius on current `origin/main`. Exact file
splits may change, but ownership must not drift across crates.

| Concern | Current anchors | Required direction |
|---------|-----------------|--------------------|
| Dimensions/roles | `akita-types/src/layout/ring_dims.rs` | retain A/B/D dims; add checked compression map dims and small-ring support |
| Level/schedule metadata | `akita-types/src/layout/params.rs`, `schedule.rs` | first-class depth-two/three F/H plans with per-map alphabets; freeze group plans |
| SIS sizing | `akita-types/src/sis/`, `akita-sis-estimator/` | d=1..16, exact bound 1, standalone 138-bit generated tables |
| Dispatch | `akita-types/src/dispatch/policy.rs` | compression slot/path independent of fold-challenge minima |
| Setup envelope | planner `matrix_envelope.rs` | flat-coefficient maximum over all active views |
| Flat setup views | `akita-types`/`akita-pcs` matrix and setup modules | all roles start at coefficient zero; no cursor |
| Witness | `akita-types/src/witness.rs`, prover hints | checked compression spans and binary support derivation |
| Relation prover | `akita-prover/src/protocol/ring_relation*` | native-role providers, sparse F/H logic, per-role quotients |
| Ring switch | `akita-types/src/proof/relation_matrix_cols.rs`, prover finalize | stop requiring one dense uniform compression-column vector |
| Range proof | prover `sumcheck/akita_stage1`, verifier stage 1 | fused `omega_tilde`, fresh `rho_bin`, succinct interval evaluator |
| Relation verifier | `akita-verifier/src/stages/stage2.rs` during transition | validate all F/H equations and native quotients |
| Setup offload | `setup_contribution`, verifier `stages/stage3.rs` | generalized providers and one shared-prefix scan/claim |
| Proof schema | `akita-types/src/proof/{commitment,levels,shapes,hints}.rs` | compressed payloads, two-stage ownership, exact size checks |
| Commit/open APIs | `akita-pcs/src/scheme`, prover compute/ring switch | mandatory chains, frozen conservative plans, no fallback |
| Profiles/tests | `akita-pcs/examples/profile`, scheme tests | payload/setup/scan reporting and full preset coverage |

## Evaluation

### Acceptance criteria

- [ ] Every shipped q128/q64/q32 schedule contains an explicit depth-two or
  depth-three F/H chain where its B/D commitment exists; terminal omissions are
  explicit.
- [ ] Default public payload is 128 bytes for q128/q64/q32 and is independently
  deserialized with the schedule-derived exact coefficient count.
- [ ] Generated SIS tables include d=1,2,4,8,16 and exact coefficient bound one;
  every B/D/F/H key reports standalone classical and quantum costs, clears the
  138-bit classical floor, and every shipped-default compression map clears
  128 quantum bits under the selected model.
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
  intermediate image, and payload and match the single-machine F/H chain
  byte-for-byte.
- [ ] Distributed semantic-layout tests shard F/H columns without requiring
  equal dimensions or exact divisibility, and derive binary support from only
  the real negative-binary F/H shards.
- [ ] A negative fixture rejects independently compressing partial machine
  images as if standalone terminal-map certification covered the repeated-column map.
- [ ] Compact matrix footprint tests verify `n*L` field coefficients and prefix
  reuse; no code path sums role footprints for allocation.
- [ ] Direct and recursive setup-contribution modes produce identical combined
  claims for mixed F/H dimensions.
- [ ] Relation-row layout tests pin the canonical order and terminal omissions.
- [ ] Every nonscalar F/H relation includes the correct native-ring quotient;
  scalar rows omit it. Dense arithmetic reference tests cover d=1,2,4,8,16.
- [ ] No raw commitment/compression policy/tiered flag survives in configs,
  generated catalogs, public APIs, proof enums, profiles, or tests.
- [ ] The final recursive `u` is absent from the wire and transcript.
- [ ] Malformed-proof fuzz/property tests cover overflow, unsupported dims,
  invalid divisibility, oversized ranks/supports, bad payload lengths, and bad
  chain metadata without verifier panic or unbounded allocation.
- [ ] Proof-size accounting and profile output separately report terminal F/H payloads,
  compression witnesses, relation proof, setup prefix, and live setup scan.
- [ ] The final protocol exposes the optimized two-stage sumcheck ownership;
  relation rows and (BIN) remain in their specified independent chains.

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
cutover is incomplete if a test or benchmark needs a raw-commitment fallback.

### Performance and reports

Add a reproducible arbitrary-chain sweep to `akita-sis-estimator` or the planner. For
incoming native images of 1, 2, 4, and 8 KiB and every shipped field/base, report:

- every F/H map's `(d,n)`, digit width, and standalone classical/quantum security;
- first-layer and terminal payload bytes;
- compact matrix field coefficients and bytes;
- maximum persistent prefix and per-level live scan;
- estimated direct-verifier field additions/multiplications;
- diagnostic hypothetical B/D slicing envelopes, clearly marked as non-protocol estimates.

The q128 4 KiB table above and universal 128-byte defaults are checked
regressions. The verifier expectation is at most a few tens of thousands of
field multiplications for typical F/H views; if measurement materially exceeds
that, first optimize the shared-prefix/sparse evaluator before enlarging the
default payload. A larger payload is a reviewed planner change, not an automatic
fallback.

## Execution plan

Implementation proceeds only after this proposed spec is approved.

1. **Descriptor and arithmetic foundations.** Add compression-chain plan
   types, canonical semantic relation-row and relation-witness layouts,
   validation, descriptor bytes, small-ring arithmetic dispatch, exact
   bound-one SIS estimation, and regenerated tables. No proof behavior changes.
2. **Planner and setup envelope.** Generate mandatory F/H plans, conservative
   frozen group metadata, flat-coefficient envelopes, and
   parameter reports. Lock the q128/q64/q32 fixtures.
3. **Commitment and hints.** Compute the generated root compression chain at commitment
   time, persist explicit hints, and cut public commitment payloads to the new
   checked flat encoding.
4. **Witness and relations.** Materialize the semantic layout for W1, add native
   quotient construction and sparse F/H relation
   providers. Establish dense-reference equivalence before verifier wiring;
   machine distribution composes with these authorities on the public companion
   branch rather than extending the old flat chunk layout.
5. **Local binary range.** Add derived `I_bin`, fresh `rho_bin`, fused
   `omega_tilde`, succinct prover/verifier evaluators, and unchanged-degree
   tests.
6. **Verifier and setup contribution.** Generalize the setup plan/evaluator,
   validate every chain and payload, and prove direct/recursive setup-mode
   equivalence with one prefix scan.
7. **Recursive, terminal, and multi-group cutover.** Compress every required
   F/H identity, remove final recursive `u`, and cover heterogeneous frozen
   group plans plus disk persistence.
8. **Stage reorder and schema cleanup.** Move relation to the final stage-1
   substage, rename setup/carried reduction to stage 2, delete stage-3 public
   schema and all raw/optional/old-tiered artifacts.
9. **Production gates and docs.** Run full CI/profile sweeps, update proof-size
   and verifier reports, fold durable protocol exposition into the book, and
   advance/archive specs according to `specs/PRUNING.md`.

Each implementation slice must preserve one canonical function per concept. Temporary adapters
must be removed in the same stack; no forwarding wrappers or parallel legacy
and compressed planners remain at completion.

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
unnecessarily large. Rank-one native rings give the universal 128-byte endpoint
at `d = 8,16,32` over q128/q64/q32 and materially reduce setup/verifier work.

### Disjoint setup offsets

Rejected. Seed-expanded matrices are pseudorandom prefix views; disjoint cursors
sum allocation and verifier scans without a security benefit under the existing
shared-setup proof model.

### Concatenate multi-group commitments

Rejected. Groups are created and frozen independently, sometimes at different
times. Concatenation changes commitment identity and later-opening semantics.
Batch their relation claims, not their transmitted commitments.

### Restore “tiered commitment”

Rejected. The old feature mixed setup-width slicing, an extra recommitment, an
optional flag, and a ring-sized wire output. The compression cutover restores
none of it. A future block-axis-slicing proposal must be justified independently
from measurements and must feed the same mandatory compression-chain interface.

## Documentation

After implementation stabilizes, fold the protocol explanation into
`book/src/how/optimizations.md`, the security contract into
`book/src/how/security.md`, the stage ownership into
`book/src/how/proving/sumcheck-stages.md`, and planner tradeoffs into
`book/src/how/configuration.md`. Update `book/src/how/recursion.md` for terminal
re-anchoring and `book/src/how/verification.md` for descriptor/no-panic checks.
Until then this spec is the implementation source of truth.

The superseded cutover draft is archived as historical design. It must not be
used for optionality, variable depth, scalar-only execution, or setup offsets.

## References

- Paper source and parameter discussion: private `paper-note` entry “lattice
  jolt akita”, Akita Sections 2 and 4.
- Superseded local draft: `specs/archive/2026-Q3/commitment-compression-cutover.md`.
- Abandoned implementation: GitHub PR #260 / branch
  `quang/commitment-compression` (conceptual archaeology only).
- Setup prefix and offload: `specs/setup-prefix-ladder.md`,
  `specs/setup-product-sumcheck.md`, `specs/batched-stage3-setup-opening.md`.
- Stage-reorder rationale: `specs/setup-layout-repack.md`.
- Terminal contract: `specs/terminal-fold-cutover.md`.
- Mixed-role baseline: `book/src/how/architecture.md` and
  `crates/akita-types/src/layout/ring_dims.rs`.
