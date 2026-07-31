# Spec: Role Native Projected Digit Layout

| Field | Value |
|---|---|
| Author(s) | Quang Dao; Codex assistant |
| Created | 2026-07-31 |
| Revised | 2026-07-31 |
| Status | active |
| PR | #337 |
| Supersedes | The coefficient-level E and T projection order left implicit by `digit-innermost-layout.md`, `setup-layout-repack.md`, `mixed-ring-dimension-per-level.md`, and `relation-range-image-sumcheck.md` |
| Superseded by | |
| Book chapter | how/proving/opening-points-layout.md; how/verifying/matrix_evaluation.md |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**,
**SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **NOT RECOMMENDED**, **MAY**, and
**OPTIONAL** in this document are to be interpreted as described in BCP 14
when, and only when, they appear in all capitals, as described in RFC 2119 and
RFC 8174.

## Authority and scope

This spec is the normative source for the coefficient-level order of projected
E and T digits. It also owns the formulas that map those digits into B and D
matrix columns and into the outgoing A carrier.

The related specs keep their existing authority:

- [`digit-innermost-layout.md`](digit-innermost-layout.md) owns group order,
  chunk ownership, witness segment order, and exact segment ranges.
- [`setup-layout-repack.md`](setup-layout-repack.md) owns the overlapping packed
  setup prefix and the A, B, and D role views of that prefix.
- [`mixed-ring-dimension-per-level.md`](mixed-ring-dimension-per-level.md) owns
  valid ring dimension tuples, setup envelope requirements, and planner policy.
- [`relation-range-image-sumcheck.md`](relation-range-image-sumcheck.md) owns
  Stage 2 relation, evaluation trace, and range-image sum-check structure.

If another live spec gives a different coefficient order for projected E or T
digits, this spec takes precedence.

## Summary

E and T are semantic values in the A ring. B and D commitments consume those
values through smaller native rings when `d_b < d_a` or `d_d < d_a`.

The implementation **MUST** split an A ring into native role subrings before it
applies the role gadget decomposition. It **MUST** store each native subring's
digits together.

The canonical order is:

```text
[semantic value][role subcolumn][digit][native ring coefficient]
```

The role specific orders are:

```text
E: [claim][block][D subcolumn][opening digit][D coefficient]
T: [claim][block][A row][B subcolumn][outer digit][B coefficient]
```

The implementation **MUST NOT** decompose an A ring into A-wide digit planes
and then split each digit plane into smaller role rings. It **MUST NOT**
transpose native role digits into digit-major carrier planes during witness
emission.

## Goals

The cutover has these goals:

- One projection and decomposition rule for both E and T.
- One coefficient order for producers, commitment kernels, hints, witnesses,
  relation evaluation, setup evaluation, and quotient computation.
- Contiguous live role data followed by carrier padding within each semantic
  value.
- No transpose between native role storage and the outgoing witness.
- The same proof bytes and hot loops as the current uniform layout when
  `a = b = d = C`.
- Structured mixed-dimension evaluation with no dense Cartesian weight table.

## Non-goals

This cutover does not change source block ownership, chunk ownership, witness
segment order, matrix ranks, SIS bounds, or planner objectives.

This cutover does not preserve mixed-dimension commitment bytes, proof bytes,
hint bytes, or setup-prefix artifacts from the old order.

This cutover does not add a layout mode or a compatibility decoder.

## Ring notation

Let

\[
R_m = F[X]/(X^m + 1).
\]

For one commitment group, define:

```text
a       A ring dimension
b       B ring dimension
d       D ring dimension
C       outgoing batch carrier dimension
H       number of claims in this group
B       number of live source blocks in this group
n_A     number of A matrix rows in this group
delta_B number of outer digits used by T
delta_D number of opening digits used by E
beta_B  outer gadget basis
beta_D  opening gadget basis
```

The validated geometry **MUST** satisfy:

\[
b\mid a,\qquad d\mid a,\qquad a\mid C.
\]

All supported dimensions are powers of two, so these divisibility conditions
also give exact subcolumn counts:

\[
q_B=\frac{a}{b},\qquad q_D=\frac{a}{d},\qquad
Q_B=\frac{C}{b},\qquad Q_D=\frac{C}{d}.
\]

Here `q_B` and `q_D` count live role subcolumns. `Q_B` and `Q_D` count role
subcolumns in the outgoing carrier.

## Canonical split and decomposition

Let

\[
y(X)=\sum_{k=0}^{a-1}y_kX^k\in R_a
\]

and let `r` be either `b` or `d`. Define `q = a / r`. The native role
subring at index `s` is:

\[
y_s(X)=\sum_{\kappa=0}^{r-1}y_{sr+\kappa}X^\kappa\in R_r,
\qquad 0\le s<q.
\]

The split is exact:

\[
y(X)=\sum_{s=0}^{q-1}X^{sr}y_s(X).
\]

Let the role use digit count `delta` and gadget basis `beta`. The balanced
coefficient decomposition of each native role subring is:

\[
y_s(X)=\sum_{o=0}^{\delta-1}\beta^o\widehat y_{s,o}(X),
\qquad \widehat y_{s,o}\in R_r.
\]

Therefore:

\[
\boxed{
y(X)=\sum_{s=0}^{q-1}\sum_{o=0}^{\delta-1}
X^{sr}\beta^o\widehat y_{s,o}(X)
}
\]

The decomposition is coefficientwise. Splitting before decomposition gives the
same signed digits as decomposing the full A ring and then splitting its digit
planes. The two methods differ only in physical order. The implementation
**MUST** split before decomposition.

The projected decomposition and recomposition operations **MUST** satisfy:

\[
\operatorname{recompose}_r(\operatorname{decompose}_r(y))=y
\]

for every valid A ring value and for both native roles. The decomposition
**MUST** retain the existing balanced digit range and tie-breaking rule. This
cutover changes storage order, not digit values.

## Canonical flat order

Let `i` be the semantic value index within a setup role or witness unit. Let
`s` be a role subcolumn, `o` a role digit, and `kappa` a coefficient in the
native role ring.

The native role digit stream **MUST** use:

\[
\boxed{
\operatorname{native\_offset}(i,s,o,\kappa)
=(((i q+s)\delta+o)r+\kappa)
}
\]

The native role matrix column **MUST** use:

\[
\boxed{
j_R(i,s,o)=((i q+s)\delta+o)
}
\]

For one semantic value, the order is:

```text
s = 0: digit 0, digit 1, ..., digit delta - 1
s = 1: digit 0, digit 1, ..., digit delta - 1
...
s = q - 1: digit 0, digit 1, ..., digit delta - 1
```

The implementation **MUST NOT** use the following old order for projected E or
T storage:

```text
digit 0: subcolumn 0, subcolumn 1, ...
digit 1: subcolumn 0, subcolumn 1, ...
```

## Carrier order and padding

`WitnessLayout` expresses E and T segment lengths in A carrier ring elements.
This spec defines the coefficient order inside those ranges.

Let `range_start` be the start of an E or T range in carrier ring elements.
The coefficient address in the outgoing witness **MUST** be:

\[
\boxed{
x_R(i,s,o,\kappa)
=C\cdot\operatorname{range\_start}
+(((iQ+s)\delta+o)r+\kappa)
}
\]

The live subcolumns are `0 <= s < q`. The carrier padding subcolumns are
`q <= s < Q`.

Every coefficient in a padding subcolumn **MUST** be zero. The implementation
**MUST** place all live subcolumns before all padding subcolumns for each
semantic value. It **MUST NOT** place carrier padding between live digits.

The coefficient count for one semantic value remains:

\[
Q\delta r=C\delta.
\]

The cutover therefore does not change E or T segment lengths.

### Uniform specialization

When `a = r = C`, both ratios are one. The address reduces to:

\[
x_R(i,0,o,\kappa)
=C\cdot\operatorname{range\_start}+((i\delta+o)C+\kappa).
\]

This is the existing uniform byte order. A conforming implementation **MUST**
preserve it exactly.

## E layout

Each semantic opening value is an A ring:

\[
e_{c,f}(X)\in R_a,
\]

where `c` is a claim and `f` is a live source block.

The index ranges are `0 <= c < H` and `0 <= f < B`.

The native D decomposition is:

\[
e_{c,f}(X)=
\sum_{s=0}^{q_D-1}\sum_{o=0}^{\delta_D-1}
X^{sd}\beta_D^o\widehat e_{c,f,s,o}(X),
\qquad \widehat e_{c,f,s,o}\in R_d.
\]

For semantic setup storage, define:

\[
i_E(c,f)=f+B c,
\]

where `B` is the group's exact live block count. The physical D column is:

\[
\boxed{
j_D(c,f,s,o)=((i_E(c,f)q_D+s)\delta_D+o)
}
\]

For one witness unit, let `u` be the local block index and `F_u` the unit's
block count. The local semantic index is:

\[
i_E^{unit}(c,u)=u+F_u c.
\]

The E coefficient address is:

\[
\boxed{
x_E(c,u,s,o,\kappa)=
C\cdot e_{start}
+(((i_E^{unit}(c,u)Q_D+s)\delta_D+o)d+\kappa)
}
\]

The prover currently receives each E value as a flat A-width coefficient
block and already views it as `q_D` native D rings before decomposition. The
cutover **MUST** preserve that native D order through D commitment, E storage,
and witness emission. Witness emission **MUST NOT** rebuild one A-width plane
per opening digit.

## T layout

Each semantic inner commitment row is an A ring:

\[
t_{c,f,i}(X)\in R_a,
\]

where `i` is an A matrix row.

The index ranges are `0 <= c < H`, `0 <= f < B`, and `0 <= i < n_A`.

The native B decomposition is:

\[
t_{c,f,i}(X)=
\sum_{s=0}^{q_B-1}\sum_{o=0}^{\delta_B-1}
X^{sb}\beta_B^o\widehat t_{c,f,i,s,o}(X),
\qquad \widehat t_{c,f,i,s,o}\in R_b.
\]

For semantic setup storage, define:

\[
i_T(c,f,i)=i+n_A(f+B c).
\]

The physical B column is:

\[
\boxed{
j_B(c,f,i,s,o)=((i_T(c,f,i)q_B+s)\delta_B+o)
}
\]

For one witness unit, define:

\[
i_T^{unit}(c,u,i)=i+n_A(u+F_u c).
\]

The T coefficient address is:

\[
\boxed{
x_T(c,u,i,s,o,\kappa)=
C\cdot t_{start}
+(((i_T^{unit}(c,u,i)Q_B+s)\delta_B+o)b+\kappa)
}
\]

The term `outer digit` **MUST** refer to `num_digits_outer` and
`log_basis_outer`. Code and documentation **MUST NOT** call a T digit an
opening digit or call its count `depth_open`.

## B and D commitment relations

Let `D_{j,k}(X)` be an entry in the batch-shared D matrix over `R_d`. Let
`J_D(g,c,f,s,o)` be the global D column defined in the multi-group section.
The D commitment relation is:

\[
V_j(X)=\sum_{g,c,f,s,o}
D_{j,J_D(g,c,f,s,o)}(X)\widehat e_{g,c,f,s,o}(X).
\]

Its ring relation is:

\[
\sum_{g,c,f,s,o}
D_{j,J_D(g,c,f,s,o)}(X)\widehat e_{g,c,f,s,o}(X)-V_j(X)
=(X^d+1)r_j^D(X).
\]

Each group owns its B matrix. Let `B^{(g)}_{j,k}(X)` be a group B matrix entry
in `R_{b_g}`. The group B commitment relation is:

\[
U_{g,j}(X)=\sum_{c,f,i,s,o}
B^{(g)}_{j,j_{B,g}(c,f,i,s,o)}(X)
\widehat t_{g,c,f,i,s,o}(X).
\]

Its ring relation is:

\[
\sum_{c,f,i,s,o}
B^{(g)}_{j,j_{B,g}(c,f,i,s,o)}(X)
\widehat t_{g,c,f,i,s,o}(X)-U_{g,j}(X)
=(X^{b_g}+1)r_{g,j}^B(X).
\]

Changing a role from the old order to `j_R` above is a column permutation. It
does not change matrix width, matrix rank, digit bounds, SIS bounds, or setup
capacity. Current D storage already follows the required physical order. B
storage does not. The B cutover changes mixed-dimension commitment bytes.

The B and D matrix multiplication kernels **MUST** consume the native role
digit stream directly. They **MUST NOT** receive an A-wide digit stream and
derive their column order by raw rechunking.

## Consistency and A relations

Every consistency relation use of E **MUST** recompose the semantic A ring by:

\[
\operatorname{recompose}_D(\widehat e)(X)=
\sum_{s,o}X^{sd}\beta_D^o\widehat e_{s,o}(X).
\]

Every A relation use of T **MUST** recompose the semantic A ring by:

\[
\operatorname{recompose}_B(\widehat t)(X)=
\sum_{s,o}X^{sb}\beta_B^o\widehat t_{s,o}(X).
\]

The existing folded Z terms are unchanged. In abstract form, the affected
relations remain:

\[
\sum_{c,f}\rho_{c,f}\operatorname{recompose}_D
(\widehat e_{c,f})(X)-Z_{cons}(X)
=(X^a+1)r^{cons}(X),
\]

and

\[
\sum_{c,f}\rho_{c,f}\operatorname{recompose}_B
(\widehat t_{c,f,i})(X)-Z_{A,i}(X)
=(X^a+1)r_i^A(X).
\]

At the relation point `alpha`, the projection and gadget weight is:

\[
\boxed{
\alpha^{sr}\beta^o
}
\]

The layout changes which equality address receives that weight. It does not
change the projection algebra.

## Relation equality tensors

For a role `R` with native dimension `r`, define the projected equality term:

\[
\operatorname{peq}_R(i,s,o,\kappa)=
\alpha^{sr}\operatorname{eq}
(\tau,x_R(i,s,o,\kappa)).
\]

The complete digit weight is:

\[
\beta^o\operatorname{peq}_R(i,s,o,\kappa).
\]

Relation construction and setup contribution **MUST** treat this as the tensor
product of:

```text
semantic equality and challenge factors
subcolumn projection powers alpha^(s * r)
digit gadget powers beta^o
native ring coefficient evaluation
```

The subcolumn and digit axes **MUST** use the same order on the setup side and
the witness side. A structured evaluator **MUST NOT** transpose either axis or
materialize a table whose size is the product of all logical factors.

When `q = 1`, the evaluator **MUST NOT** allocate an alpha-power vector,
evaluate a power-sequence MLE, or multiply by `alpha^0`. When `Q = 1`, it
**MUST** borrow the unprojected contiguous equality window. When `q = 1 < Q`,
it **MUST** address the live carrier spans directly while still omitting the
unit projection factor.

## Evaluation trace

The evaluation trace consumes E coefficients from the outgoing witness. The
new E order does not change the trace functional.

Let claim `h` belong to group `g(h)`. Let its D dimension be `d_h`, its opening
basis be `beta_{D,h}`, and its normalized claim coefficient be `theta_h`. Let
`mu_{h,f}` be its block-opening weight at group-global block `f`, and let
`I_h` be its source-inner trace vector of length `a_h`.

Let a common coefficient block have size `c_0`, and let
`l_{D,h} = d_h / c_0`. Write a native D coefficient as
`v c_0 + kappa`, where `0 <= v < l_{D,h}` and `0 <= kappa < c_0`. The
coefficient weight is:

\[
\Theta_{h,f,s,o,v,\kappa}=
\theta_h\mu_{h,f}\beta_{D,h}^o
I_h[sd_h+vc_0+\kappa].
\]

The implementation **MUST** contract the coefficient axis through partial
evaluations of the form:

\[
P_{h,s,v}(r_\kappa)=
\sum_{\kappa=0}^{c_0-1}
\operatorname{eq}(r_\kappa,\kappa)
I_h[sd_h+vc_0+\kappa].
\]

It **MUST NOT** recover the old digit-major layout by transposing E or by
materializing a carrier-expanded trace table. The coefficient partials for one
claim **MUST** cost `O(a_h)` and **MUST** be reused across its blocks and opening
digits. Preparing live support **MUST** cost
`O(H_g B_g q_{D,g} delta_{D,g})` for group `g`. Both costs **MUST** be
independent of the carrier padding count `Q_{D,g} - q_{D,g}`.

## Multi-group and multi-chunk rules

For a witness unit, `u` is local to the unit. Setup columns use the global block
index:

\[
f=\operatorname{global\_block\_start}+u.
\]

The semantic setup matrices **MUST NOT** contain chunk copies. Each witness
unit **MUST** map its local block to the one global setup column defined above.

Let `H_g` be the number of claims in group `g`. The number of semantic role
values in that group is:

\[
N_{D,g}=H_gB_g,
\qquad
N_{B,g}=H_gB_gn_{A,g}.
\]

The batch-shared D matrix concatenates group columns. Its physical group width
and prefix are:

\[
W_{D,g}=N_{D,g}q_{D,g}\delta_{D,g},
\qquad
p_{D,g}=\sum_{h<g}W_{D,h}.
\]

The global D matrix column **MUST** be:

\[
\boxed{J_D(g,c,f,s,o)=p_{D,g}+j_{D,g}(c,f,s,o)}
\]

The D prefix sum **MUST** follow authenticated relation group order.

Each group owns a separate B matrix. Its physical width is:

\[
W_{B,g}=N_{B,g}q_{B,g}\delta_{B,g}.
\]

The B column **MUST** be the group-local `j_{B,g}` defined above. It **MUST**
start at zero for each group. The implementation **MUST NOT** add a cross-group
B column prefix.

Chunk count **MUST NOT** appear in any semantic count, physical matrix width,
or group prefix.

For a group with `a < C`, only the first `q_R = a / r` role subcolumns are
live. The remaining `Q_R - q_R` subcolumns are zero carrier padding. The
implementation **MUST NOT** extend B or D setup width to cover those zero
padding subcolumns.

Group order and chunk order remain the order defined by `WitnessLayout`.

## Storage and type contracts

### T commitment hints

An `AkitaCommitmentHint` T digit stream **MUST** store native B digit planes.
Its digit stride **MUST** equal `d_b`, not `d_a`.

For each logical source block, its plane count **MUST** be:

\[
n_A q_B\delta_B.
\]

The flat order within that block **MUST** be:

```text
[A row][B subcolumn][outer digit]
```

The hint **MUST NOT** store an A-wide T digit representation as a second source
of truth.

### E digit storage

An E digit stream **MUST** store native D digit planes. Its digit stride
**MUST** equal `d_d`.

For each semantic E value, its plane count **MUST** be:

\[
q_D\delta_D.
\]

The flat order for that value **MUST** be:

```text
[D subcolumn][opening digit]
```

Block metadata **MUST** preserve semantic E value boundaries. It **MUST NOT**
treat each D subcolumn as an unrelated semantic block.

### Inner commitment output

The inner A commitment operation produces recomposed A rows. The role
transition that knows both `d_a` and `d_b` owns T decomposition.

The output type for the inner A operation **MUST** store only the recomposed A
rows and their A dimension. It **MUST NOT** require a decomposed digit stride
to equal the A dimension.

The codebase **MUST** have one canonical A-to-B projected decomposition
operation. Dense, one-hot, sparse, recursive, and setup-prefix paths **MUST**
call that operation. They **MUST NOT** keep backend-specific copies of the
layout rule.

### Witness addresses

E and T are no longer addressable as one carrier ring index per digit in the
mixed case. An address API **MUST** return a checked coefficient offset or a
checked contiguous native-role span.

`e_index` and `t_index` style APIs that omit the role subcolumn **MUST NOT** be
used as the physical mixed-dimension address authority.

## Required implementation behavior

The implementation **MUST** meet these requirements:

1. The projected decomposer scans each semantic A row and emits native role
   digits directly in canonical order.
2. The B and D commitment kernels consume that output without a transpose.
3. T recomposition rebuilds each B subring from its contiguous digit run and
   writes it directly into the correct A coefficient range.
4. E recomposition follows the same rule with D subrings.
5. Witness emission copies each live native role run directly and writes zero
   carrier padding only after the live run.
6. Relation, setup, trace, quotient, recursive, and terminal consumers use the
   same canonical address formulas.
7. The old digit-major projected order and all conversion helpers that exist
   only to preserve it are deleted in the same cutover.

The implementation **MUST NOT** add a second runtime path selected by a broad
mixed-versus-uniform dispatch boundary. It **MAY** use local compile-time
specialization and local `q_R = 1` or `Q_R = 1` branches inside the canonical
operations.

## Complexity and performance requirements

For `n` semantic A values and role digit count `delta`, projected decomposition
and recomposition **MUST** use:

\[
O(n a\delta)
\]

coefficient work. They **MUST NOT** add an `O(q)` copy of the full A digit
stream.

Witness emission **MUST** use work proportional to the live bytes plus the
required zero padding. It **MUST NOT** allocate one temporary A carrier plane
per digit.

B and D matrix scans **MUST** remain row contiguous in native role rings.

The structured setup and relation evaluators **MUST** keep the subcolumn and
digit factors as compact tensor axes. They **MUST NOT** allocate a dense table
over groups, chunks, blocks, rows, subcolumns, digits, and coefficients.

When `q_R = 1`, the canonical implementation **MUST** avoid:

- multiplication by one for projection weights;
- allocation of projection power vectors;
- layout conversion;

When `Q_R = q_R`, it **MUST** avoid carrier padding loops. When
`a = b = d = C`, it **MUST** avoid any regression from the current uniform
commitment and verifier kernels.

### Performance acceptance protocol

Performance comparisons **MUST** use the PR merge base as the baseline. Each
comparison **MUST** build both revisions with the same Rust toolchain, release
profile, feature set, and target CPU. It **MUST** alternate baseline and PR
measurements on one machine after at least one unmeasured warmup for each
revision.

The focused comparison **MUST** cover `setup_index_weight` and
`relation_evaluator` with these shapes:

- uniform `a = b = d = C`;
- B mixed and D uniform;
- D mixed and B uniform;
- B and D both mixed;
- `a < C` carrier padding;
- multiple groups and multiple chunks.

It **MUST** also cover the uniform root commitment kernel and the end-to-end
profile cases affected by the cutover. Each focused case **MUST** use at least
ten measured samples and report the median.

A case is a performance regression when the PR median is more than three
percent slower than the baseline median. The cutover **MUST NOT** have a
regression in any uniform or mixed focused case. Production mixed paths
**MUST NOT** contain a layout conversion, a non-unit live stride, or a dense
projected weight table.

The three-sample CI profile comment is supporting evidence, not the only
acceptance measurement. If it reports a slowdown greater than three percent,
the implementation **MUST** run the focused comparison for that component and
record the result in the PR. A reported regression **MUST NOT** be dismissed as
noise without that comparison.

## Serialization and protocol identity

This is a breaking protocol cutover.

Mixed-dimension B column labels and mixed-dimension E and T witness addresses
change. Current D commitment storage already has the required subcolumn then
digit order, so this cutover does not permute D commitment columns. The
implementation **MUST** update every protocol epoch, fixture, setup-prefix
artifact identity, and serialized hint expectation that depends on changed
bytes.

The implementation **MUST NOT** accept old mixed-dimension hints or proof bytes.

The schedule and instance descriptor already bind the role dimensions and
digit bases. If the current protocol identity does not distinguish the old and
new physical order, the cutover **MUST** change that identity once. It
**MUST NOT** serialize a permanent layout mode.

## Safety requirements

Every verifier-reachable address calculation **MUST** check multiplication,
addition, range length, and role divisibility before indexing or allocation.

Malformed dimensions, block ownership, digit counts, or storage lengths
**MUST** return `AkitaError`. Verifier-reachable code **MUST NOT** panic, use
unchecked indexing, or allocate from an unchecked proof-controlled product.

## Independent oracle

The test suite **MUST** contain an independent dense oracle. The oracle **MUST**
compute expected addresses from the boxed formulas in this spec. It **MUST NOT**
call production address, tensor, or column helpers.

For each case, the oracle **MUST** compare:

- native E and T digit streams;
- B and D commitment inputs;
- outgoing witness bytes;
- B and D setup columns;
- consistency, A, B, and D relation evaluations;
- direct and structured setup contributions;
- evaluation trace weights;
- quotient rows;
- recursive handoff and terminal consumption.

Tests **MUST** include:

- uniform `a = b = d = C`;
- mixed B only;
- mixed D only;
- both B and D mixed;
- `a < C` in a multi-group carrier;
- multiple claims and multiple live blocks;
- multiple chunks with a nonzero global block start;
- digit counts 1, 3, and a power of two;
- direct and setup-offloaded verification;
- malformed stride, ratio, padding, and block metadata.

The uniform tests **MUST** assert byte equality against the pre-cutover uniform
fixtures. Mixed tests **MUST** assert the new order explicitly rather than only
checking prover and verifier agreement.

## Acceptance criteria

- [ ] This spec is the normative coefficient-level E and T layout contract.
- [ ] E and T both split into native role rings before decomposition.
- [ ] T hints store B-native digits with stride `d_b`.
- [ ] E storage uses one semantic value boundary with D subcolumns inside it.
- [ ] B and D physical columns use `[semantic][subcolumn][digit]`.
- [ ] Outgoing E and T use `[semantic][carrier subcolumn][digit][coefficient]`.
- [ ] Live data precedes carrier padding for every semantic value.
- [ ] No projected digit transpose remains in production code.
- [ ] One canonical projected decomposition operation serves every source type.
- [ ] One canonical inverse recomposition operation serves relation code.
- [ ] Direct and structured setup evaluation use the same projected equality
      tensors.
- [ ] Evaluation trace coefficient partials cost `O(a_h)` per claim and do not
      scan carrier padding.
- [ ] `q_R = 1` paths allocate no projection powers and perform no projection
      multiplication by one.
- [ ] Uniform commitment bytes and witness bytes remain unchanged.
- [ ] Uniform and mixed performance passes the defined acceptance protocol.
- [ ] Mixed commitment, witness, relation, setup, trace, quotient, recursive,
      and terminal paths match the independent dense oracle.
- [ ] Old mixed layout helpers, fixtures, and compatibility paths are deleted.
- [ ] Protocol identity and setup-prefix artifacts are updated once for the
      breaking cutover.
- [ ] Verifier-reachable malformed inputs return `AkitaError` without panic.
- [ ] No mixed benchmark regression is caused by layout conversion, a non-unit
      live stride, or a dense projected weight table.
- [ ] Repository formatting, documentation guardrails, Clippy feature graphs,
      and focused tests pass.

## Implementation surface

The cutover is expected to change these areas:

```text
crates/akita-prover/src/kernels/linear/decompose.rs
crates/akita-prover/src/compute/hint_recompose.rs
crates/akita-prover/src/api/commitment.rs
crates/akita-prover/src/api/setup_prefix.rs
crates/akita-prover/src/protocol/ring_relation.rs
crates/akita-prover/src/protocol/ring_relation/relation_quotient.rs
crates/akita-prover/src/protocol/ring_switch/commit.rs
crates/akita-prover/src/protocol/ring_switch/coeffs.rs
crates/akita-prover/src/protocol/ring_switch/relation_weights.rs
crates/akita-types/src/proof/hints.rs
crates/akita-types/src/proof/tail_segments.rs
crates/akita-types/src/witness.rs
crates/akita-types/src/setup_contribution/plan/
crates/akita-types/src/trace_weight/
crates/akita-verifier/src/protocol/ring_switch/
```

This list is a review guide. It is not permission to add forwarding wrappers or
parallel layout authorities.

## Alternatives considered

### Keep T digit first and change only E

This preserves the current T hint and makes B commitment migration smaller. It
leaves two projection rules and keeps carrier padding between live T digits.
It is rejected.

### Decompose A first, then transpose once

This preserves the current A-wide decomposition kernel but adds a full digit
transpose before B or D consumption. It adds memory traffic and a second
physical order. It is rejected.

### Keep one A carrier plane per digit

This keeps the old `e_index` and `t_index` model. In mixed multi-group layouts,
it places padding between live digit runs and prevents setup and witness axes
from sharing one contiguous tensor. It is rejected.

### Add separate uniform and mixed implementations

This can preserve current uniform code without changing its local types. It
creates two protocol implementations that can drift. It is rejected. The
canonical implementation **MAY** still specialize local `q_R = 1` and `Q_R = 1`
inner loops.

## Documentation updates

The implementation PR **MUST** keep these documents aligned:

- [`digit-innermost-layout.md`](digit-innermost-layout.md), for logical witness
  ownership and segment order;
- [`setup-layout-repack.md`](setup-layout-repack.md), for semantic and physical
  role columns;
- [`mixed-ring-dimension-per-level.md`](mixed-ring-dimension-per-level.md), for
  projection geometry and performance claims;
- [`relation-range-image-sumcheck.md`](relation-range-image-sumcheck.md), for
  Stage 2 relation and evaluation trace structure;
- [`../book/src/how/proving/opening-points-layout.md`](../book/src/how/proving/opening-points-layout.md),
  for the narrative witness order;
- [`../book/src/how/verifying/matrix_evaluation.md`](../book/src/how/verifying/matrix_evaluation.md),
  for projected relation and setup evaluation.

After implementation is stable and the book contains the durable contract,
this spec **SHOULD** be marked implemented and then archived under the normal
spec lifecycle.

## References

- [BCP 14 and RFC 2119](https://www.rfc-editor.org/rfc/rfc2119)
- [RFC 8174](https://www.rfc-editor.org/rfc/rfc8174)
- [`digit-innermost-layout.md`](digit-innermost-layout.md)
- [`setup-layout-repack.md`](setup-layout-repack.md)
- [`mixed-ring-dimension-per-level.md`](mixed-ring-dimension-per-level.md)
