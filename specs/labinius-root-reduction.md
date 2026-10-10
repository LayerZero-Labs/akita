# LaBinius root reduction

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY in this document
have the BCP 14 meanings defined by RFC 2119 and RFC 8174 when capitalized.

## Statement and output

The opt-in `root` modules compose the [field-switch frontend](labinius-clear-opening.md),
[admitted setup](labinius-setup-contract.md), [lowered root relation](labinius-lowered-root.md)
and [root sumchecks](labinius-root-sumcheck.md). They reduce an opening of
one commitment, to a binary claim, a prime claim or both, to two or three
evaluation claims over a challenge field. They do not implement a
polynomial commitment scheme, recursion, setup offloading or zero knowledge.

The statement consists of an immutable `AdmittedRootSetup<D,M>`, a binding
commitment to the image digit table Y, and a `RootStatement<H,E>` holding at
least one of two claims about the committed binary source:

- a **binary claim**: a host-field point `r` and a claimed host-field value
  `t`, the multilinear evaluation of the committed words over H;
- a **prime claim** `PrimeClaim<E>`: a linear functional of the committed
  bits, read as zero and one in the challenge field E, defined
  [below](#prime-claim).

`RootStatement::mode` names the three opening modes: `Binary`, `Prime` and
`Both`. A statement with neither claim is `InvalidInput`. The mode is part of
the statement and is [bound before any claim](#grinding-plan). The host
field H MUST implement the sealed `SwitchField` contract; it fixes the source
word type in every mode. All binary frontend operations
use B = F162. The profile uses D = 648, the minus trinomial, packing degree
k = 4 and commitment prime q = 33568993. The image and response tables hold
stored base-16 digits; there is no digit-base parameter.

The lowered relation runs over a field pair. The base field F is a prime
field (`F: Field + CanonicalEncoding`) whose characteristic P passes
`LabiniusRootShape::derive_encoding`
([root admission](labinius-root-admission.md#proof-prime-admission-and-clear-integer-ranges)).
The committed tables hold small integers of F. The challenge field E
(`E: ExtField<F>`) holds every Fiat-Shamir field challenge, every sumcheck
round message, every evaluation point and value, the message `y_Y` and all
public weights. Integer no-wrap and unit-difference admission are statements
about P, never about `|E|`. E is admitted exactly when every field-challenge
site of the reduction has a proof-of-work target within Akita's cap
([root admission](labinius-root-admission.md#challenge-field-admission)).
Neither field is a parameter of the setup or of the clear commitment, and
neither needs a root of unity. Two pairs are exercised:
`F = E = Prime128Offset275`, and `F = Prime64Offset59` with
`E = jolt_field::Ext2<Prime64Offset59>`. `F = E = Prime64Offset59` is rejected.

`LoweredRootLayout::new::<F,E,_,_>(admitted.setup(), admitted.shape())` admits
the pair and determines all dimensions. Write `d=162`, `m=layout.m()`,
`C=layout.columns()`, `n_A=layout.n_a()`, `nu=layout.witness_log_len()`,
`mu=layout.image_log_len()` and `pi=layout.prime_log_len()`. The layout fixes
the response digit table W of length `2^nu`, the image digit table Y of
length `2^mu` and the prime left opening uP of length `2^pi`, including
padding. Their multilinear extensions use little-endian index bits.

The output `RootEvaluationClaims<E>` contains

```text
response_point = rho, response_value = w_eval,  W~(rho) = w_eval;
image_point = rho', image_value = y_eval,       Y~(rho') = y_eval;
prime = Some((rho'', p_eval)),                  uP~(rho'') = p_eval.
```

The field `prime` is `Some` exactly when the statement has a prime claim.
The verifier MUST discharge every claim through its oracle before accepting.
A successful frontend alone supplies a `BinaryEvaluationClaim`, not an
opening authenticated against Y.

### Prime claim

Write `src[j][col]` for packed source ring element j of column col, an
integer polynomial of degree below D
([lowered relation](labinius-lowered-root.md#prime-row)).
Source cell `col*scalar_rows + j*k + c` is scalar component c of that ring
element, and bit s of the cell sits at coefficient `k*s + c` with sign
`sigma(k*s + c)`: negated for odd s when `k > 1`. This is the packing of the
clear commitment, not a new one. A prime claim is

```text
v = sum_col eq(r_col, col) * sum_j eq(r_ring, j) * <omega, src[j][col]>,
```

with `r_col` in `E^(log2 C)`, `r_ring` in `E^(log2 m)`, `v` in E, and `omega`
a public vector of D weights in E, one per coefficient of a ring element.
The coefficients of `src` are reduced into F and so into E. The weights cover
all D positions, including those no source bit occupies when the embedded
cell has fewer than `D/k` bits, so a consumer can test every position.

`PrimeClaim::cell_evaluation(layout, point, bit_weights, value)` builds the
common case, an evaluation over cells:

```text
value = sum_cell eq(point, cell) * sum_(s < D/k) bit_weights[s] * bit_s(cell);
omega[k*s + c] = sigma(k*s) * bit_weights[s] * eq(r_c, c).
```

`point` has one coordinate per bit of the cell index, low bit first, and is
split as `r_c = point[..log2 k]`, `r_ring = point[log2 k..log2 scalar_rows]`,
`r_col = point[log2 scalar_rows..]`. This is the split of rows and columns
the binary opening uses for its left expansion. The factor `sigma` cancels
the packing sign. `PrimeClaim::validate` rejects a claim whose points or
weights do not have the layout's dimensions.

The prime left opening is the table

```text
uP[col] = sum_j eq(r_ring, j) * src[j][col]    in E[Y]/(Phi_D), one per column;
uP[t + P_c*col] = coefficient t of uP[col]     for t < D, and 0 for D <= t < P_c,
```

with `P_c = next_power_of_two(D)`: coefficient innermost, column outermost,
`2^pi = P_c*C` entries. The prover's `lowered::prime_left_opening` computes
it from the source words. Its entries are arbitrary elements of E; no
alphabet or range is checked on them. Entries of a coefficient tail carry no
public weight and are not constrained.

## Oracle contract

`RootProverOracle<E>` and `RootVerifierOracle<E>` define the prover and
verifier contracts. The channel entry points `prove_root_reduction` and
`verify_root_reduction` return the output claims after discharge. Their
`_bytes` forms create the root session; the prover returns proof bytes and
claims, and the verifier returns claims after EOF validation.

Every trait method is generic over the reduction's channel
`S: ClearChannel`, takes it as its last argument `channel: &mut S` and
returns `Result<(), AkitaError>`. With those two parts left out, the methods
are:

```text
trait RootVerifierOracle<E: Field> {
    fn bind_image(&mut self, layout: &LoweredRootLayout);
    fn bind_prime_opening(&mut self, layout: &LoweredRootLayout);
    fn bind_response(&mut self, layout: &LoweredRootLayout);
    fn discharge(&mut self, claims: &RootEvaluationClaims<E>);
}
trait RootProverOracle<E: Field> {
    fn bind_image(&mut self, layout: &LoweredRootLayout);
    fn commit_prime_opening(&mut self, layout: &LoweredRootLayout, table: &[E]);
    fn commit_response(&mut self, layout: &LoweredRootLayout, digits: &[u8]);
    fn discharge(&mut self, claims: &RootEvaluationClaims<E>);
}
```

The prover and verifier oracles take the same channel used by the reduction.
Their commitments and opening messages MAY have any scheme-specific encoding.
They MUST bind the same table layout and field pair as the reduction. The
image and response tables hold stored digits, which are small integers of F.
The prime left opening holds elements of E. Every claim is an evaluation of
a table's multilinear extension at a point of E.
There are four oracle call sites, in this order. The second is made exactly
when the statement has a prime claim; an oracle that serves no prime claim
is never asked for it:

| Call | When | Required guarantee |
| --- | --- | --- |
| Bind image commitment | Once during statement binding, before the frontend | Absorb data that binds every entry of the image digit table Y and its owner before any frontend challenge. An opening MUST refer to this table. |
| Bind prime left opening | With a prime claim: once, after the binary left expansion U when there is one, before the fold-response nonce and the fold challenges | The prover receives the exact length-`2^pi` table of E elements and commits it. The verifier receives and absorbs the same commitment. It MUST bind every entry, including the coefficient tails, before any fold challenge. |
| Bind response commitment | Once after the fold challenges, before alpha, xi, gamma or eta | The prover receives the exact length-`2^nu` digit byte table and commits it. The verifier receives and absorbs the same commitment. It MUST bind every entry, including padding. |
| Discharge evaluation claims | Once, last, after every evaluation message | Authenticate every output claim against those same commitments. A verifier error MUST reject. All opening challenges MUST follow w_eval, y_eval and p_eval. |

The prime left opening MUST be bound before the fold challenges. The prime
row compares it with a fold of the response, and a table chosen after the
challenges could be adjusted to them
([lowered relation](labinius-lowered-root.md#prime-opening-and-the-unit-condition)).

The image digit table is a function of the clear commitment:
`akita_labinius_prover::lowered::encode_image(layout, commitment)` is the one
function that builds it, for a commitment made before the opening and for the
reduction. The prover recomputes it from the clear commitment it is given, so
the prover's image oracle MUST have bound exactly that table. Neither binding
checks a digit alphabet; the reduction does, for both tables.

Let `epsilon_bind` be the oracle's total binding failure probability and
`epsilon_eval` its total evaluation-soundness error for these claims,
including any shared opening reduction. These are supplied by the selected
commitment scheme; this protocol assigns them no numerical value. A commitment
that permits a different table for each claim does not satisfy this contract.

The public transparent oracles `TransparentRootProverOracle<F,E>` and
`TransparentRootVerifierOracle<F,E>` are differential oracles: **sends the
whole table; not succinct; no hiding**. Image binding absorbs the padded Y
table as public bytes, one per stored digit, in table order. Response binding
sends exactly one byte per W entry. Both fix those bytes without checking
their alphabet, so the combined sumchecks enforce that condition. Prime
binding sends the uP table as one message, every entry as its canonical base
coordinates (`exchange_transparent_prime_opening`); a non-canonical
coordinate rejects. Discharge evaluates the tables in chunks and compares
with the output claims; without a prime claim the prime table is not
consulted. The image table is public input, so this absorption contributes
no proof bytes.
The transparent response message contributes `2^nu` proof bytes and the
transparent prime message `f*2^pi`; discharge contributes none. These oracles
MUST NOT be described as a succinct PCS.

## Transcript and wire grammar

The session domain and the first absorbed public message are
`akita/labinius/root-reduction/v1`. The public domain message is length-prefixed
with a u64 little-endian byte count. Setup identity is exactly
`admitted.identity_bytes::<H>()`; it binds the profile, certified rank,
derivation inputs, seed and nested matrix-view identity. It names neither
field, so the field pair follows as one public message:

1. the characteristic P of F as sixteen little-endian bytes;
2. the extension degree `e = E::DEGREE` as four little-endian bytes;
3. the multiplication table of the coordinate basis `b_0, ..., b_(e-1)` of E
   over F, where `b_i = E::from_base_slice` of the i-th unit vector: for each
   pair `i <= j` in lexicographic order, the e base coordinates of `b_i * b_j`,
   each a canonical little-endian element of F.

The table determines E as an F-algebra together with the basis in which
challenges are sampled and messages are encoded, so two extensions of one
degree with different defining polynomials have different bytes. The message
has 36 bytes for `E = F = Prime128Offset275` and 68 bytes for
`Ext2<Prime64Offset59>` over `Prime64Offset59`.
The next public message is `RootGrindingPlan::canonical_bytes`, the
[grinding plan](#grinding-plan) of this statement's opening mode. It depends
on the setup, the fields and the mode alone, and it is what binds the mode:
the three modes have three different plans, and no separate mode byte is
absorbed. The image commitment follows it, then the binary claim when the
statement has one, then the prime claim when it has one.
Host point coordinates and value use exactly the coordinate-word encoding of
`bind_statement`, with no new representation. The prime claim is one public
message: the domain `akita/labinius/root-prime-claim/v1`, length-prefixed
like the session domain, then `r_col`, `r_ring`, `omega` and `v` in that
order, every element as its canonical base coordinates. The layout fixes all
three counts, so none is encoded. All proof counts are statement-derived; no
proof-supplied shape or length exists.

Let `f=E::DEGREE*F::NUM_BYTES`, `h=H::ROWS`, `a=H::BATCH_BITS`,
`w_H=size_of::<H::Source>()`, `n=r.len()`, and
`b_KA=ceil(e_KA/8)`, `b_Q=ceil(e_Q/8)`, `b_K=ceil(e_K/8)` from
`LabiniusRootEncoding`.
F128 has `(h,a,w_H)=(128,7,16)`; F192 has `(192,8,8)`.
An element of E is encoded as its `E::DEGREE` base coordinates in order, each
a fixed-width canonical little-endian element of F; a non-canonical coordinate
MUST reject. These are the bytes of Akita's `ExtensionAtom<F, E>`, and sumcheck
round messages use `akita_sumcheck` over the same pair. A challenge in E is
`E::DEGREE` base-field challenges, by `akita_transcript::ext_challenge`. Both
exercised pairs have `f=16`. B elements are 21 little-endian bytes with the six
unused high bits zero. KA, Q and K use unsigned offsets
`value+2^(e-1)` in `ceil(e/8)` little-endian bytes. An offset at least `2^e`
MUST reject; these are precisely the ranges enforced by `LoweredPublic::new`,
not an additional honest-witness range.

In the table, P is a proof message, V a verifier challenge, and pub an absorbed
public message. Public messages and challenges occupy zero proof bytes. The
second column gives the modes a step belongs to: all three, those with a
binary claim (`Binary`, `Both`), or those with a prime claim (`Prime`,
`Both`). A step outside the statement's mode is absent: nothing is sent,
drawn or absorbed for it, and nothing is zero-filled.

| Step | Modes | Actor and value | Encoding and proof length | Site or label |
| --- | --- | --- | --- | --- |
| 1 | all | pub: domain, admitted identity, field pair, grinding plan, image commitment; with a binary claim r, t; with a prime claim `r_col`, `r_ring`, omega, v | Length-prefixed domain; canonical admitted identity; the field-pair message above; the plan's canonical bytes; oracle binding; low-coordinate-first host words; the prime-claim message above | `akita/labinius/root-reduction/v1`, statement binding |
| 2 | binary | P: partials; V: batching point; P/V: frontend rounds; P: terminal source value | `h*w_H` partial bytes; a uniform B coordinates; n rounds of two B coefficients; one B terminal; total `h*w_H+(2n+1)*21`; no nonce | Existing `prove_frontend` / `verify_frontend` order; plan sites `FrontendBatch` and `FrontendRound`, zero work |
| 3 | binary | P: left expansion U | C canonical B elements; `21*C` bytes; `verify_left_expansion` MUST pass | Left expansion message |
| 4 | prime | Oracle: prime left opening commitment | Scheme-dependent; transparent oracle sends `f*2^pi` bytes | Prime binding call |
| 5 | all | P: fold-response nonce; V: fold challenges | One canonical LEB128 `NonceAtom` below 4096, one or two bytes; then C challenges from the admitted binary sampler; profile, count and label absorbed before its root draw | `akita/labinius/root-fold/v1` |
| 6 | all | Oracle: response commitment | Scheme-dependent; transparent oracle sends `2^nu` bytes | Response binding call |
| 7 | all, then binary | P: KA; with a binary claim Q, K | `n_A*D*b_KA` offset bytes, row then coefficient; then, with a binary claim, `(d-1)*b_Q` and `d*b_K` offset bytes; no length prefix | Clear lowered witness messages, `exchange_root_auxiliary` |
| 8 | all; xi binary; eta prime | P: nonce, V: alpha; P: nonce, V: xi; P: nonce, V: gamma; P: nonce, V: eta | In that order, each present site's proof-of-work nonce and then one fresh E draw | Plan sites `Alpha`, `Xi`, `Gamma`, `PrimeRow`; `LRRD`, details 2, 3, 4, 8 |
| 9 | all; y_P prime | P: y_Y; with a prime claim y_P | One E element each, f bytes; define `s=c_pub-y_Y`, plus `eta*y_P` with a prime claim | Image and prime-row weighted-sum messages |
| 10 | all | P: nonce, V: tau; P: nonce, V: beta | One nonce for the whole point, then nu fresh E coordinates, lowest index bit first; one nonce, then one fresh E beta | Plan sites `EqualityPoint` and `Batch`, invocation 0; `LRRD`, detail 5 / group i (tau_i), detail 6 (beta) |
| 11 | all | pub: combined instance header; P/V: response rounds | Header binds kind combined, invocation 0 and nu, followed by public absorption of beta*s; nu rounds, each 17 E coefficients, then that round's nonce, then its challenge; `17*nu*f` bytes and nu nonces | Root sumcheck instance domain; plan site `Round`, invocation 0; LRSC diagnostics |
| 12 | all | P: w_eval | One E element, f bytes; MUST satisfy the combined terminal with structured `kw` | Response evaluation message |
| 13 | all | P: nonce, V: tau'; P: nonce, V: beta' | One nonce for the whole point, then mu fresh E coordinates, lowest index bit first; one nonce, then one fresh E beta' | Plan sites `EqualityPoint` and `Batch`, invocation 1; `LRRD`, detail 5 / group i (tau'_i), detail 6 (beta') |
| 14 | all | pub: combined instance header; P/V: image rounds | Header binds kind combined, invocation 1 and mu, followed by public absorption of beta'*y_Y; mu rounds, each 17 E coefficients, then that round's nonce, then its challenge; `17*mu*f` bytes and mu nonces | Root sumcheck instance domain; plan site `Round`, invocation 1; LRSC diagnostics |
| 15 | all | P: y_eval | One E element, f bytes; MUST satisfy the combined terminal with structured `ky` | Image evaluation message |
| 16 | prime | P: nonce, V: theta | One nonce, then one fresh E theta | Plan site `Batch`, invocation 2; `LRRD`, detail 6 |
| 17 | prime | pub: product instance header; P/V: prime rounds | Header binds kind product, invocation 2 and pi, followed by public absorption of `v+theta*y_P`; pi rounds, each two E coefficients, then that round's nonce, then its challenge; `2*pi*f` bytes and pi nonces | Root sumcheck instance domain; plan site `Round`, invocation 2; LRSC diagnostics |
| 18 | prime | P: p_eval | One E element, f bytes; MUST satisfy the product terminal with structured `kv+theta*kp` | Prime evaluation message |
| 19 | all | Oracle: discharge every claim | Scheme-dependent; transparent oracle emits nothing | Final oracle call |

Step 5 is the fold-response nonce of
[the clear opening](labinius-clear-opening.md#fold-response-nonce), with the
same search, message and constants. The prover MUST compute `fold_integer` for
each candidate and accept the first whose every coefficient lies in the
admitted interval; it then calls `encode_witness`. The verifier applies no
acceptance test to the nonce beyond its domain: the alphabet term of the
response instance, over three stored digits per coefficient, is what confines
the committed response to that interval.
The same response determines KA, Q and K. The verifier builds `LoweredPublic`
only after receiving these messages and drawing the challenges of step 8.
Passing a commitment that mismatches the source is a caller error; the prover
returns `InvalidProof` when a commitment-row remainder is then not divisible
by q.
The response terminal MUST equal
`combined_terminal(tau,rho,beta,w_eval,kw)` with
`kw=witness_weight_mle(layout,public,setup,rho)`. The image terminal MUST
equal `combined_terminal(tau',rho',beta',y_eval,ky)` with
`ky=image_weight_mle(layout,public,rho')`. Neither check may be skipped at a
zero public weight. The two instances have the same type: the alphabet term
on every table entry plus the table's linear term of the lowered relation.
The message y_Y splits the relation's constant between them, `s=c_pub-y_Y`
for the response and y_Y for the image.

With a prime claim the lowered relation has one more row, the
[prime row](labinius-lowered-root.md#prime-row), batched by the fresh scalar
eta. Its response side is part of the response weights `K_W`, and its
`uP` side is the message `y_P=<uP,K_P>`, which splits the row between the
response instance and the product instance exactly as y_Y splits the
commitment rows between the response and image instances: the response
claim becomes `s=c_pub-y_Y+eta*y_P` (`LoweredPublic::response_claim`). The
product instance proves both linear claims on uP at once,

```text
<uP, K_v + theta*K_P> = v + theta*y_P,
K_v[t + P_c*col] = eq(r_col,col)*omega[t],   t < D,
```

for a theta drawn after v, y_P and uP are fixed. Its terminal MUST equal
`product_terminal(p_eval, kv+theta*kp)` with
`kv=claim.value_weight_mle(layout,rho'')` and
`kp=prime_row_weight_mle(layout,public,rho'')`. The prover computes uP
itself, from the source and `r_ring`; it is not a caller input. A prime
claim whose value is not the functional of the source is a caller error: the
product sumcheck refuses the false sum and proving returns `InvalidInput`.

A nonce in steps 8, 10, 11, 13, 14, 16 and 17 is present exactly when its
site's target is nonzero, which holds for every site in E whenever
`|E| < 2^128`, as in both exercised pairs. Its encoding, predicate and checks are those of
the [grinding plan](#grinding-plan).

Field challenge sites use family `0x4c525244` (`LRRD`): the detail is the
site's tag in the plan, the invocation is the instance, and the group is the
coordinate index within an equality point. Sumcheck sites use family
`0x4c525343` (`LRSC`). A nonce and its predicate draw are recorded under the
site of the challenge they protect: group 0 of the `LRRD` site, or the
round's `LRSC` challenge site. These site assignments identify diagnostics;
the absorbed session, statement, plan and instance headers provide
cryptographic domain separation.

The versioned sumcheck instance header is a cryptographic public binding.
Diagnostic `ProtocolSiteId` records alone absorb nothing; changing an invocation
number only in those records separates nothing. The canonical binding concatenates
`akita/labinius/root-sumcheck-instance/v1`, the kind byte (0 combined, 1
product), invocation as u32 little-endian and dimension as u64 little-endian,
in one public call.
See [root sumchecks](labinius-root-sumcheck.md).
Both prover entry points and verifier replay MUST absorb that header before
rounds, including for zero-round instances. A round message MUST precede its
nonce, and the nonce its challenge. The public statement and witness messages
already fix each input claim and public weight before its sumcheck.

Byte entry points MUST reject truncation, non-canonical atoms and trailing
bytes. Proof failures return `AkitaError::InvalidProof`; invalid public setup
or geometry returns the applicable input/setup error. Statement-sized
allocation and all size arithmetic MUST follow the verifier no-panic contract.

## Grinding plan

Every field challenge of the reduction is a site of a public plan and is
priced by Akita's rule
([transcript grinding](transcript-grinding.md#security-model)). A site whose
conditional bad fraction is at most `L/|K|`, for its challenge set K, has the
target

```text
g = grind_bits_for_loss(L, |K|) = the least g with L * 2^128 <= |K| * 2^g,
```

on the exact cardinality: `|K| = P^e` for a site in E
(`ChallengeFieldOrder::from_field`) and `|K| = 2^162` for a site in B
(`ChallengeFieldOrder::from_full_capacity`). A target above
`MAX_GRINDING_BITS = 25` is an error, so the plan exists exactly for an
admitted field pair. The target arithmetic, the cap, the nonce slack, the
predicate transition, the nonce codec and the loss-factor functions are
Akita's and are called, not restated. Akita's `GrindingSite` enumerates
Akita's own sites and its replay types own an Akita channel, so the
reduction has its own site type and cursor over those shared primitives.

`RootGrindingPlan::new::<H, F, E>(shape, encoding, mode)` derives the plan
of an opening mode: an ordered list of runs, each a site, its loss factor L,
its target g, its nonce width and its number of consecutive visits. With
`a = H::BATCH_BITS`, `n = ceil(log2(num_cells))` and b the number of binary
claims, 0 or 1, the runs in replay order are:

| Site | Tag, invocation | Modes | Set | Visits | Loss factor L | Akita loss function |
| --- | --- | --- | --- | ---: | --- | --- |
| `FrontendBatch` | 0, 0 | binary | B | 1 | a | `multilinear_point_loss_factor(a)` |
| `FrontendRound` | 1, 0 | binary | B | n | 2 | `polynomial_identity_loss_factor(2)` |
| `Alpha` | 2, 0 | all | E | 1 | `D-1` | `polynomial_identity_loss_factor(D-1)` |
| `Xi` | 3, 0 | binary | E | 1 | `2d-2` | `polynomial_identity_loss_factor(2d-2)` |
| `Gamma` | 4, 0 | all | E | 1 | `max(n_A+b-1, 1)` | `powers_batch_loss_factor(n_A+b)` |
| `PrimeRow` | 8, 0 | prime | E | 1 | 1 | `powers_batch_loss_factor(2)` |
| `EqualityPoint` | 5, 0 | all | E | 1 | nu | `multilinear_point_loss_factor(nu)` |
| `Batch` | 6, 0 | all | E | 1 | 1 | `powers_batch_loss_factor(2)` |
| `Round` | 7, 0 | all | E | nu | 17 | `polynomial_identity_loss_factor(17)` |
| `EqualityPoint` | 5, 1 | all | E | 1 | mu | `multilinear_point_loss_factor(mu)` |
| `Batch` | 6, 1 | all | E | 1 | 1 | `powers_batch_loss_factor(2)` |
| `Round` | 7, 1 | all | E | mu | 17 | `polynomial_identity_loss_factor(17)` |
| `Batch` | 6, 2 | prime | E | 1 | 1 | `powers_batch_loss_factor(2)` |
| `Round` | 7, 2 | prime | E | pi | 2 | `polynomial_identity_loss_factor(2)` |

The loss factors are the numerators of the
[soundness ledger](#soundness-ledger); 17 is the degree bound of a combined
round and 2 that of a product round. `PrimeRow` is the scalar eta of the
prime row and `Batch` of invocation 2 the scalar theta of the product
instance. The product instance has no equality point. An equality point is
one site: one nonce protects the whole point and
its coordinates are drawn after it at distinct diagnostic sites, which is
Akita's treatment of a site that draws several independently labelled values
under one predicate. A run that is never visited is not recorded.

The plan is a function of the mode, and the three plans differ: the plan of
a mode with a binary claim contains the `FrontendBatch` run and no other
plan does, and the plan of a mode with a prime claim contains the `PrimeRow`
run and no other plan does. Their canonical bytes therefore differ, and
absorbing them binds the mode.

A site with `g > 0` has nonce width `g + 7` bits
(`GRINDING_NONCE_SLACK_BITS`) and carries one inline nonce per visit,
directly before the protected challenge. A site with `g = 0` has width zero,
emits no proof bytes and makes no state transition; it stays in the plan so
that every field challenge is accounted for. The two frontend sites are of
this kind for both host fields: `L * 2^128 <= 2^162` whenever `L <= 2^34`.
The frontend draws its B challenges itself and has no nonce slot, so plan
derivation returns `InvalidSetup` if a frontend site had a nonzero target,
and replay starts at the first site in E.

For a visit with `g > 0` the transition is Akita's proof-of-work transition:

1. The prover calls `search_grinding_nonce`: for each candidate in
   `[0, 2^(g+7))` in increasing order it clones the public duplex state,
   absorbs the candidate as a canonical unsigned LEB128 `NonceAtom` and
   squeezes 32 predicate bytes. The first candidate whose first g predicate
   bits, read low bit first, are zero is selected. Exhaustion returns
   `InvalidInput`.
2. `commit_grinding_nonce` writes the winner once as a proof message and
   draws the predicate on the live state. The prover requires that it equals
   the preview.
3. The verifier calls `receive_grinding_nonce`. It MUST reject with
   `InvalidProof` a missing or non-canonical nonce, a nonce of `2^(g+7)` or
   more, and a predicate with a nonzero bit among its first g.
4. The protected challenge is drawn afterwards, as a separate draw.

The plan's canonical bytes are, in order:

1. the domain `akita/labinius/root-grinding-plan/v1`;
2. the security target 128 as two little-endian bytes, then the nonce slack
   7 and the target cap 25, one byte each;
3. the run count as four little-endian bytes;
4. for each run: the tag as one byte, the invocation as four little-endian
   bytes, L as eight, g and the nonce width as one byte each, and the visit
   count as eight.

Akita binds a digest of its plan and policy in the descriptor before any
proof message. The reduction has no descriptor and does the equivalent
directly: `bind_root_statement` absorbs these bytes as the public message
after the field pair, before the image binding and before any proof message
or challenge. For a given mode the sites and loss factors are also a
function of data bound earlier, since the admitted identity fixes the shape
and H and the field-pair message fixes `|E|`; the bytes add the mode and the
policy constants, which no earlier message names.

Each role owns a monotone cursor over the plan. Every protected draw names
its site, which MUST be the next planned visit, and both roles MUST find the
plan completely consumed before the oracle discharges the claims. A
violation is `InvalidInput`: it is a defect of the calling code and cannot
be caused by proof bytes. A channel without a scheduled plan rejects every
round challenge. Standalone sumchecks schedule their own round-site plan
with `RootGrindingPlan::sumcheck_rounds::<F, E>`; the reduction uses
`RootGrindingPlan::new`.

The plan covers the reduction's own field challenges. In Akita's query
kinds, its sites are `ProofOfWork` entries. The fold-response nonce of step
5 is the `FoldResponse` kind: a 12-bit domain, at most 4096 trials and no
predicate. The fold challenge vector is the `FoldChallengeGroup` kind: zero
work bits, with its rate supplied by the size of the support family S.
Neither is a field challenge and neither is in the plan. Challenges drawn by
the commitment oracle belong to the selected scheme, which MUST price them
itself.

## Soundness ledger

The ledger has the form of Akita's transcript-grinding accounting
([transcript grinding](transcript-grinding.md#fold-work-decision-and-total-security-accounting)).
Each field challenge is a site i of the [grinding plan](#grinding-plan), with
challenge set `K_i`, loss factor `L_i` and target `g_i`. For a challenge
uniform conditional on the preceding transcript, fixed tables and
binding/evaluation-sound oracles, the conditional bad fraction of site i is
at most `L_i/|K_i|`; the table below derives each `L_i`. A random-oracle
address of site i yields a challenge only when its nonce clears the
predicate, an independent `2^-g_i` event, so one address has weight
`2^-g_i * L_i/|K_i|`. For a prover that touches `q_i` distinct addresses of
site i, the verifier's own completions included,

```text
E_field = sum_i q_i * 2^-g_i * L_i / |K_i|.
```

B and E are distinct fields: `|B|=2^162` and `|E|=P^e` for extension degree
e. This applies Akita's accounting to these sites. It is not a random-oracle
security theorem for the reduction, which remains an open obligation below;
the plan stores the local algebraic loss of each site and no extraction tree
factor.

| Site | Set | Fixed before it | Bad event and derivation | L |
| --- | --- | --- | --- | --- |
| `FrontendBatch`, a coordinates | B | Setup, image commitment, host statement and partials | An incorrect fixed partial vector has a nonzero multilinear discrepancy; total degree at most a | a |
| `FrontendRound`, each `z_i` | B | Partials, batching point and previous rounds; this round's two coefficients | A false degree-two product-sumcheck transition matches at a random point | 2 |
| `Alpha` | E | Y, W commitment, U, uP, folds, KA, Q and K | A false remainder row, a commitment row or the prime row, has degree at most `D-1`; choose one false row, with no factor for their number | `D-1` |
| `Xi` | E | Same fixed witness messages, and alpha | A false parity residual modulo P has degree at most `2d-2=322` | `2d-2` |
| `Gamma` | E | All row polynomials and their alpha/xi evaluations | A nonzero combination of the n_A commitment-row evaluations and, with a binary claim, the parity evaluation is a polynomial in gamma of degree at most `n_A+b-1` | `max(n_A+b-1,1)` |
| `PrimeRow`: eta | E | The same, gamma, and uP | With G the gamma-batched residual and R the prime-row residual at alpha, `G+eta*R=0`; if R is nonzero at most one eta solves it | 1 |
| `EqualityPoint` 0: tau, nu coordinates | E | W, Y, public weights, c_pub, y_Y and, with a prime claim, eta and y_P, hence s | An invalid response digit makes `Z(tau)=sum_x eq(tau,x) A(W(x))`, `A(w)=prod_(a<16)(w-a)`, a nonzero multilinear polynomial of total degree at most nu | nu |
| `Batch` 0: beta | E | W, Y, weights, s and tau | `Z(tau)+beta*(L-s)=0`, `L=<W,K_W>`; unless both coefficients vanish, at most one beta solves it | 1 |
| `Round` 0, each `rho_i` | E | Tables, weights, tau, beta, s, instance header and current round message | Degree at most 17 per round | 17 |
| `EqualityPoint` 1: tau', mu coordinates | E | Y, public weights and y_Y, all fixed before tau | The same alphabet polynomial over the image digits, total degree at most mu | mu |
| `Batch` 1: beta' | E | Y, weights, y_Y and tau' | `Z'(tau')+beta'*(L'-y_Y)=0`, `L'=<Y,K_Y>`; at most one beta' solves it | 1 |
| `Round` 1, each `rho'_i` | E | Tables, weights, tau', beta', y_Y, completed response instance, w_eval, header and current round message | Degree at most 17 per round | 17 |
| `Batch` 2: theta | E | uP, v, y_P and the weights `K_v`, `K_P` | `(<uP,K_v>-v)+theta*(<uP,K_P>-y_P)=0`; unless both differences vanish, at most one theta solves it | 1 |
| `Round` 2, each `rho''_i` | E | uP, weights, theta, the claim, both completed combined instances, header and current round message | Degree at most 2 per round | 2 |

Two challenge families are not field sites. The fold challenge vector is
drawn from the finite support family S, embedded in B by parity, after the
setup, the image commitment, the claims, all of U and, with a prime claim,
the commitment to uP are fixed. A nonzero
B-linear discrepancy has at most one family member per conditioned coordinate
because parity is injective. This is Akita's `FoldChallengeGroup`: zero work
bits, each of its C coordinate addresses has weight `1/|S|`, and a complete
vector contributes `C/|S|`; extraction across fold challenges and its loss
remain open. The prime row uses the same vector. Isolating one column of uP
from two transcripts divides by a difference of two family members modulo P,
and proof-prime admission makes every such difference a unit, so any two
distinct members serve and the prime claim adds no term to this entry
([lowered relation](labinius-lowered-root.md#prime-opening-and-the-unit-condition)).
The oracle's challenges come after every commitment and every
output claim and contribute the selected scheme's
`epsilon_bind+epsilon_eval`.

A row that is false modulo P, as a polynomial with coefficients in the prime
field F, is a nonzero polynomial over E of the same degree, so a uniform
element of E is a root with probability at most degree over `|E|`.
The matrix relation is reduced modulo Phi_D before evaluation, so each
remainder residual has degree at most D-1, including the carry term.
The prime row is reduced the same way and has coefficients in E from the
start: uP is a table of E elements, and nothing about it is lifted to the
integers.
The unreduced parity residual still has degree 322: Phi times Q has that
maximum degree. Gamma combines n_A A rows and, with a binary claim, puts its
next power `gamma^n_A` on the parity row. The characteristic MUST exceed 17 so
combined interpolation at the distinct nodes `0..=17` is defined and the
alphabet polynomial vanishes on exactly sixteen integers; every admitted
proof prime does.

The fold-response nonce lets a prover obtain up to 4096 fold-challenge vectors
for one transcript prefix. This is Akita's `FoldResponse` entry. `C/|S|` is
the bound for one vector. Each candidate nonce is a random-oracle query and
is charged to the query budget through the addresses it touches, as in
[`fold-linf-rejection.md`](fold-linf-rejection.md); there is no separate
12-bit debit and the fold budget `lambda_fold` is unchanged.

Beta and beta' MUST be fresh, uniform conditional on the preceding
transcript. The claims s and y_Y MUST be fixed before them: for nonzero beta
an adaptive choice `s=L+Z(tau)/beta` would satisfy the equation for an invalid
alphabet. The `nu/|E|+1/|E|` bound is conservative: if L=s and Z is nonzero,
no beta works. The response instance establishes `<W,K_W>=c_pub-y_Y` and the
image instance `<Y,K_Y>=y_Y`, for the one y_Y fixed before either point, so
together they carry the relation `<W,K_W>+<Y,K_Y>=c_pub`.

With a prime claim, theta MUST be fresh in the same sense, and v and y_P
MUST be fixed before it. The response instance then establishes
`<W,K_W>=c_pub-y_Y+eta*y_P` and the product instance `<uP,K_v>=v` and
`<uP,K_P>=y_P`, so the three instances carry the value claim and

```text
<W,K_W> + <Y,K_Y> - eta*<uP,K_P> = c_pub,
```

which is `G+eta*R=0` in the notation of the table. The message y_P is sent
after eta; that is harmless, because the product instance ties it to the uP
and the `K_P` fixed before eta.

With `q_a` marking the fold coordinate addresses a prover touches, define

```text
E_fold = sum_a q_a / |S|;
E_accounted = E_field + E_fold + epsilon_bind + epsilon_eval.
```

The plan enforces `L_i * 2^128 <= |K_i| * 2^g_i` at every site, and fold
admission gives `|S| >= C * 2^lambda_fold`. With `lambda_fold >= 128` every
address therefore has weight at most `2^-128`, and for a prover that touches
at most Q addresses in total

```text
E_field + E_fold <= Q * 2^-128.
```

This is Akita's 128-bit per-address rate. An aggregate target needs a
declared query budget, exactly as in Akita. There is no error for the exact
host reconstruction, left expansion, range checks, integer no-wrap
implication or canonical encoding.

At `(log_num_cells,log_fold_width,lambda_fold)=(22,8,128)`, the profile gives
`m=4096`, `C=256`, `n_A=2`, `D=648`, `n=22`, `nu=24`, `mu=22`, `pi=18`. The
bounded-weight-46 family has

```text
|S| = sum_(j=0)^46 binomial(162,j)
    = 107268495235892251993000716928106195079952;
-log2(1/|S|) = 136.300278 bits per coordinate address;
-log2(C/|S|) = 128.300278 bits per vector.
```

The sites at this geometry follow. Weight bits are
`-log2(2^-g * L/|K|)`. The two exercised pairs,
`E = F` with `P = 2^128 - 275` and `E = Ext2` over `P = 2^64 - 59`, have
`|E| = 2^128 - 275` and `|E| = (2^64 - 59)^2`; both are below `2^128` by a
relative `2^-57` or less, so their targets are equal and their weight bits
agree at the displayed precision.

| Site | Modes | Set | Visits | L | g | Nonce width, bits | Weight bits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FrontendBatch`, F128 | binary | B | 1 | 7 | 0 | 0 | 159.193 |
| `FrontendBatch`, F192 | binary | B | 1 | 8 | 0 | 0 | 159.000 |
| `FrontendRound` | binary | B | 22 | 2 | 0 | 0 | 161.000 |
| `Alpha` | all | E | 1 | 647 | 10 | 17 | 128.662 |
| `Xi` | binary | E | 1 | 322 | 9 | 16 | 128.669 |
| `Gamma`, with a binary claim | binary | E | 1 | 2 | 2 | 9 | 129.000 |
| `Gamma`, without | `Prime` | E | 1 | 1 | 1 | 8 | 129.000 |
| `PrimeRow` | prime | E | 1 | 1 | 1 | 8 | 129.000 |
| `EqualityPoint` 0 | all | E | 1 | 24 | 5 | 12 | 128.415 |
| `Batch` 0 | all | E | 1 | 1 | 1 | 8 | 129.000 |
| `Round` 0 | all | E | 24 | 17 | 5 | 12 | 128.913 |
| `EqualityPoint` 1 | all | E | 1 | 22 | 5 | 12 | 128.541 |
| `Batch` 1 | all | E | 1 | 1 | 1 | 8 | 129.000 |
| `Round` 1 | all | E | 22 | 17 | 5 | 12 | 128.913 |
| `Batch` 2 | prime | E | 1 | 1 | 1 | 8 | 129.000 |
| `Round` 2 | prime | E | 18 | 2 | 2 | 9 | 129.000 |

There are 53 visits with work in mode `Binary`, 72 in `Prime` and 73 in
`Both`. Every site in E needs at least one bit
because `|E| < 2^128`: even `L = 1` fails `2^128 <= |E|`. The largest weight
is `2^-128.415`, at the response equality point, in every mode.

The per-address rate is the quantity Akita prices and the one this ledger
states. It is not the failure probability of one interactive run, which
proof of work does not change: with fresh verifier challenges one run fails
with probability at most the sum of the bad fractions over its visits. In
mode `Binary` that is
`[D-1 + 322 + n_A + (nu + 1 + 17*nu) + (mu + 1 + 17*mu)]/|E| = 1801/|E|`
for the sites in E, plus `(a+2n)/2^162` for the frontend and `C/|S|` for
the fold. A prime claim adds `1 + 1 + 2*pi = 38` to the numerator, for eta,
theta and the product rounds; without a binary claim xi leaves it and gamma
counts 1, and the frontend term is absent. The second sum takes one address
per visit and weights each by `2^-g`:

| Mode | Sum of bad fractions over E | One run | Weighted sum, one address per visit |
| --- | ---: | ---: | ---: |
| `Binary` | `1801/|E|` | `2^-117.185` | `28.64 * 2^-128 = 2^-123.160` |
| `Prime` | `1516/|E|` | `2^-117.434` | `38.01 * 2^-128 = 2^-122.752` |
| `Both` | `1839/|E|` | `2^-117.155` | `38.64 * 2^-128 = 2^-122.728` |

The 64-bit field as its own challenge field would need 65 to
74 bits per site; the layout rejects it.

The KA term has degree below D, so it does not change the alpha bound. The
derivation bias, an additive term of the SIS reduction, is owned by
[root admission](labinius-root-admission.md#derivation-bias). Neither closes
the composition obligations below.

## What acceptance proves, and open obligations

The conditional theorem intended by this composition is: if the verifier
accepts and the oracles are binding and evaluation-sound, then the fixed
committed W and Y both have the required alphabet and satisfy the lowered
polynomial relations of the statement's mode, with the received KA and, with
a binary claim, Q and K in their enforced ranges,
except with the probability the [ledger](#soundness-ledger) assigns to the
sites in E, plus `epsilon_bind+epsilon_eval`. For one interactive run that
probability is the sum of their bad fractions; for a prover charged by
random-oracle address it is their part of `E_field`.
By the [lowered-root specification](labinius-lowered-root.md), decoding W then
gives a response in the accepted interval, decoding Y gives an integer image
table T with every coefficient in the enforced digit interval, the commitment
relation `A z - sum_i c_i T_i = q KA` modulo P, and, with a binary claim,
the F162 parity endpoint.
The admitted parity inequality promotes the checked parity identity modulo P to
an integer identity before reducing modulo two. The admitted commitment
inequality promotes the commitment relation the same way: every term is now
bounded, so it holds over the integers and therefore modulo q, in each
accepting transcript by itself
([lowered relation](labinius-lowered-root.md#source-binding-and-projection)).
The image digits need not be those of the canonical residues; any table in the
alphabet that satisfies the relation is accepted, and only its residues modulo
q enter the clear endpoint. Alphabet-valid padding has zero public relation
weight and is permitted.

With a prime claim, acceptance further gives, for the fixed committed uP,
the value identity `v = sum_col eq(r_col,col)*<omega,uP[col]>` and the prime
row `sum_j eq(r_ring,j)*z[j] = sum_col iota(Ch_col)*uP[col]` in
`E[Y]/(Phi_D)`, between the decoded response z reduced modulo P and uP. The
row is an identity modulo P only; uP is not range-checked and nothing about
it is promoted to the integers. Across two transcripts that share uP and
differ in one fold challenge, the row isolates that column of uP as the
functional `sum_j eq(r_ring,j)*(.)` of the extracted rational source reduced
modulo P; that step divides by a fold-challenge difference modulo P and is
the one place the reduction uses the unit condition of proof-prime admission
([lowered relation](labinius-lowered-root.md#prime-opening-and-the-unit-condition)).
The prime claim is therefore an evaluation of the extracted rational source
reduced modulo P. The binary claim concerns the same source reduced modulo
two. That the two readings describe one bit string, that is, that the
source is integral with every coefficient zero or its packing sign, is not
proved by this reduction. It is the consumer's obligation, which the general
weight vector omega makes expressible as prime claims and which the reduction
does not discharge
([exact bits](labinius-lowered-root.md#exact-bits-are-the-consumers-obligation)).

This theorem is conditional on the polynomial identity and sumcheck arguments
in the cited specifications. It does not assert knowledge of the original
binary source from one transcript. In particular, the following remain open
and MUST NOT be presented as established results:

- Knowledge extraction across fold challenges, the necessary repetitions or
  rewinding, and the resulting extraction loss.
- The SIS reduction at the admitted rank, including source comparison and
  consumer semantics for the extracted mod-two projection.
- The frontend's complete soundness statement connecting host claims to that
  extracted source, for both sealed host fields. The arithmetic batching and
  sumcheck terms above are only its local reductions.
- For a prime claim, the exact-bit check that identifies the source reduced
  modulo P with the binary source, and any consumer protocol that performs
  it.
- Fiat–Shamir security in the random-oracle model for this multi-round protocol,
  including the seed expander, indexed fold substreams and oracle composition.
- The binding and evaluation soundness of the polynomial commitment scheme
  behind the oracle. [The PCS composition](labinius-pcs.md) instantiates the
  oracle with a nested Akita opening and inherits Akita's guarantees; it
  proves none of them.
- Any setup-contribution opening replacing the materialized matrix scan.

The root transcript closes admitted-identity absorption and the order of
statement, witness and challenge binding. It does not close those remaining
obligations merely by defining an encoding or passing differential tests.

## Completeness and verifier cost

For valid setup and source input with an honest oracle, the algebraic protocol
is complete whenever some nonce in the search domain gives an integer fold
inside the admitted interval. The probability source is the admitted binary
challenge distribution, not a sumcheck event. The interval is narrower than
the deterministic envelope `C*Gamma` (23552 against `[-2184, 1911]` at the
sample geometry), so a candidate can be rejected. Akita's cap derivation
bounds by 7/8 the probability that one candidate's response exceeds the cap
1266, under the sign model stated in
[root admission](labinius-root-admission.md#fold-response), and the interval
contains `[-1266, 1266]`. If the 4096 candidates were independent at that bound, all
would fail with probability at most `(7/8)^4096 < 2^-789`. The prover returns
`InvalidInput` in that event. In a measurement over 40 seeds at the sample
geometry every proof used nonce 0. Allocation failure and malformed caller
input are execution errors rather than probabilistic completeness losses.

A proof-of-work search can also fail. A visit with target g tries at most
`2^(g+7)` candidates. Modelling each as accepted independently with
probability `2^-g`, the search is exhausted with probability
`(1-2^-g)^(2^(g+7)) < exp(-128)`, and the prover then returns `InvalidInput`.
A proof in mode `Binary` has `7+nu+mu` visits in E, one fewer without a
binary claim and `2+pi` more with a prime claim: 53, 72 and 73 at the sample
geometry for `Binary`, `Prime` and `Both`. Its searches try
`sum 2^g = 3080`, 2642 and 3156 candidates in expectation there.

Let `h=H::ROWS`, `dc=4` the response digit slots and `P_c=next_power_of_two(D)`.
Excluding the oracle, the verifier performs these operations; counts describe
work in E and omit hashing and decoding. The omitted hashing includes one
predicate evaluation per visit with a nonzero target: one nonce absorption
and one 32-byte squeeze.

| Work | Operation count or bound |
| --- | --- |
| Frontend reconstruction and rounds | h host-source contributions, a B batching coordinates, n degree-two evaluations; tensor terminal `O(n*h)` B multiplications and `O(n*h^2)` binary XORs |
| Left expansion | Equality expansion and C B multiply-adds; `O(C)` B operations |
| Public parity and challenge lifts | `k*m*d` binary-row Horner steps; C lifted challenges and two degree-d Horner evaluations; `(2d-1)` Q/K Horner steps |
| Public constant | `n_A*D` cached-H Horner steps, factored parity-offset contributions over d+k*m, `n_A*D` KA Horner steps, and for the image offset one O(D) shift recurrence with C sparse challenge dot products; no matrix traversal |
| Response replay and terminal | nu degree-17 evaluations, an O(nu) equality evaluation and 16 alphabet factors |
| Image replay and terminal | mu degree-17 evaluations, an O(mu) equality evaluation and 16 alphabet factors |
| Response structured weight | One matrix pass with `n_A*m*D` multiply-adds into `n_A*D` coefficients, one O(D) shift recurrence and row-factor contractions; unchanged parity contractions over dc, d and k*m entries |
| Image structured weight | One O(D) shift recurrence, then C sparse challenge dot products of weight at most w, `C*n_A` entry factors and the digit gadget over eight slots |
| Prime row, response side | With a prime claim: D powers of alpha and the offset sum for the constant; at the terminal one equality evaluation over `log2 m` coordinates and D scalar equality evaluations over the coefficient block |
| Prime replay and terminal | With a prime claim: pi degree-2 evaluations; the value weight by D scalar equality evaluations and one over `log2 C` coordinates; the row weight by one O(D) shift recurrence and C sparse challenge dot products |

For the scalar `eq_eval_at_index` loops, the parity terminal costs
`O(dc*log(dc)+d*log(P_c/k)+k*m*log(k*m))` operations in E after public binary
rows are cached. The matrix terminal uses O(n_A*D+m) field workspace rather
than a padded setup-weight table. Equality expansions and the shift
recurrence take O(D) field work and workspace. Neither structured terminal scans
a committed table. Setup preparation derives the cached
H rows once; online public construction scans the scalar rows but no matrix.
The response terminal still scans the materialized setup once, so this slice
makes no sublinear setup-verifier claim. Transparent oracle discharge adds
MLE work linear in `2^nu+2^mu`, and in `2^pi` with a prime claim, outside
these reduction counts.

## Reduction wire size

The reduction-owned byte count, excluding oracle bytes, is the sum of a
part every mode has and one part per claim:

```text
common:  nonce + work + n_A*D*ceil(e_KA/8) + f*[3 + 17*nu + 17*mu]
binary:  h*w_H + (2n+1)*21 + 21*C + (d-1)*ceil(e_Q/8) + d*ceil(e_K/8)
prime:   f*[2 + 2*pi]
```

The binary part is the frontend, U, Q and K. The prime part is y_P, the
product round bodies and p_eval. Every term is fixed by the statement
except the two nonce terms. `nonce` is
the LEB128 length of the accepted fold-response nonce: one byte below 128 and
two bytes otherwise. `work` is the total LEB128 length of the proof-of-work
nonces, one per visit of a [plan](#grinding-plan) site with a nonzero target;
a nonce of width `g+7` bits occupies between one and `ceil((g+7)/7)` bytes.
`root_reduction_wire_size::<H, F, E>(shape, mode)` counts the fold-response
nonce at two bytes and `work` at `RootGrindingPlan::nonce_max_bytes` of the
mode's plan, the sum of those maxima, so it returns the largest proof, and a
proof is shorter by the nonce bytes it does not use. The function needs only
the shape, the mode, the field pair and H; it requires no materialized matrix
and returns an error for a
field pair the shape does not admit. Field-element sizes come from the
pair's encoding, `f=E::DEGREE*F::NUM_BYTES`, not from a profile. All
products, sums, widths and powers of two MUST use checked arithmetic.
Absorbed statement bytes, headers and challenges contribute zero proof bytes.

At the sample geometry `(e_KA,e_Q,e_K)=(36,35,35)`, so KA, Q and K all use
five bytes per integer. Both exercised pairs have `f=16` and the same plan
targets, so their sizes agree. The nonce maxima are three bytes each for
alpha and xi and two bytes for every other visit:

| Message region | Modes | Bytes |
| --- | --- | ---: |
| Frontend partials, F128 | binary | 2048 |
| Frontend partials, F192 | binary | 1536 |
| Frontend round bodies and source terminal | binary | 945 |
| U | binary | 5376 |
| Fold-response nonce | all | 1 or 2 |
| KA | all | 6480 |
| Q | binary | 805 |
| K | binary | 810 |
| Proof-of-work nonces, `Binary` | | 53 to 108 |
| Proof-of-work nonces, `Prime` | | 72 to 145 |
| Proof-of-work nonces, `Both` | | 73 to 148 |
| y_Y | all | 16 |
| y_P | prime | 16 |
| Response round bodies | all | 6528 |
| w_eval | all | 16 |
| Image round bodies | all | 5984 |
| y_eval | all | 16 |
| Prime round bodies | prime | 576 |
| p_eval | prime | 16 |

| Mode | Largest, F128 | Largest, F192 | Smallest, F128 |
| --- | ---: | ---: | ---: |
| `Binary` | 29134 | 28622 | 29078 |
| `Prime` | 19795 | 19795 | 19721 |
| `Both` | 29782 | 29270 | 29706 |

A prime claim costs 648 bytes at most on top of a binary one: 608 bytes of
field elements and at most 40 bytes for 20 nonces. Mode `Prime` has no
message that depends on H, so its size is the same for both host fields. In
sixteen measured F128 proofs per mode, eight per field pair, the proofs took
29079 to 29083 bytes in mode `Binary`, 19722 to 19726 in `Prime` and 29707
to 29710 in `Both`: a nonce below 128 is one byte, which a target of at most
5 bits almost always finds.
The transparent oracle adds 16777216 response bytes at this geometry and,
with a prime claim, 4194304 bytes for uP. The
4194304 image digits are absorbed public data, not proof bytes. The byte
wrappers add no framing or final marker. EOF is the proof boundary.
