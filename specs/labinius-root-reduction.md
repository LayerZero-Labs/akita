# LaBinius root reduction

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY in this document
have the BCP 14 meanings defined by RFC 2119 and RFC 8174 when capitalized.

## Statement and output

The opt-in `root` modules compose the [field-switch frontend](labinius-clear-opening.md),
[admitted setup](labinius-setup-contract.md), [lowered root relation](labinius-lowered-root.md)
and [root sumchecks](labinius-root-sumcheck.md). They reduce a binary opening to
two coefficient-field evaluation claims. They do not implement a polynomial
commitment scheme, recursion, setup offloading or zero knowledge.

The statement consists of an immutable `AdmittedRootSetup<F,D,M>`, a digit base
`b` in `{1,2,4}`, a binding commitment to the image table Y, a host-field point
`r` and a claimed host-field value `t` of the committed binary source. The
host field H MUST implement the sealed `SwitchField` contract. F is the odd
coefficient prime field; all binary frontend operations use B = F162 instead.
Both admitted profiles use D = 648, the minus trinomial,
packing degree k = 4, and P128 coefficient prime.

`LoweredRootLayout::new(admitted.setup(), admitted.shape(), base)` determines
all dimensions. Write `d=162`, `m=layout.m()`, `C=layout.columns()`,
`n_A=layout.n_a()`, `nu=layout.witness_log_len()` and
`mu=layout.image_log_len()`. The layout fixes response table W of length
`2^nu` and image table Y of length `2^mu`, including padding. Their
multilinear extensions use little-endian index bits.

The output `RootEvaluationClaims` contains

```text
response_point = rho, response_value = w_eval,  W~(rho) = w_eval;
image_point = rho', image_value = y_eval,       Y~(rho') = y_eval.
```

The verifier MUST discharge both claims through its oracle before accepting.
A successful frontend alone supplies a `BinaryEvaluationClaim`, not an
opening authenticated against Y.

## Oracle contract

`RootProverOracle<F>` and `RootVerifierOracle<F>` define the prover and
verifier contracts. The channel entry points `prove_root_reduction` and
`verify_root_reduction` return the output claims after discharge. Their
`_bytes` forms create the root session; the prover returns proof bytes and
claims, and the verifier returns claims after EOF validation.

The prover and verifier oracles take the same channel used by the reduction.
Their commitments and opening messages MAY have any scheme-specific encoding.
They MUST bind the same table layout and coefficient field as the reduction.
There are exactly three oracle call sites, in this order:

| Call | When | Required guarantee |
| --- | --- | --- |
| Bind image commitment | Once during statement binding, before the frontend | Absorb data that binds every entry of Y and its owner before any frontend challenge. An opening MUST refer to this table. |
| Bind response commitment | Once after the fold challenges, before alpha, xi or gamma | The prover receives the exact length-`2^nu` digit byte table and commits it. The verifier receives and absorbs the same commitment. It MUST bind every entry, including padding. |
| Discharge evaluation claims | Once, last, after both evaluation messages | Authenticate both output claims against those same commitments. A verifier error MUST reject. All opening challenges MUST follow w_eval and y_eval. |

Let `epsilon_bind` be the oracle's total binding failure probability and
`epsilon_eval` its total evaluation-soundness error for these two claims,
including any shared opening reduction. These are supplied by the selected
commitment scheme; this protocol assigns them no numerical value. A commitment
that permits a different table for each claim does not satisfy this contract.

The public transparent oracles are differential oracles: **sends the whole
table; not succinct; no hiding**. Image binding absorbs every canonical
coefficient of the padded Y table as public bytes in table order. Response
binding sends exactly one byte per W entry. It fixes those bytes without
checking their alphabet, so the combined sumcheck enforces that condition.
Discharge evaluates the response from digit bytes in chunks and the image with
`akita_algebra::poly::multilinear_eval`, then compares with the output claims.
The image table is public input, so this absorption contributes no proof bytes.
The transparent response message contributes `2^nu` proof bytes; discharge
contributes none. These oracles MUST NOT be described as a succinct PCS.

## Transcript and wire grammar

The session domain and the first absorbed public message are
`akita/labinius/root-reduction/v1`. The public domain message is length-prefixed
with a u64 little-endian byte count. Setup identity is exactly
`admitted.identity_bytes::<H>()`; it binds the profile, certified rank,
derivation inputs, seed and nested matrix-view identity. The base tag is one
byte, 0, 1 or 2 for b = 1, 2 or 4. Host point coordinates and value use exactly
the coordinate-word encoding of `bind_statement`, with no new representation.
All proof counts are statement-derived; no proof-supplied shape or length exists.

Let `f=F::NUM_BYTES`, `h=H::ROWS`, `a=H::BATCH_BITS`,
`w_H=size_of::<H::Source>()`, `n=r.len()`, and
`q=ceil(e_Q/8)`, `k_Q=ceil(e_K/8)` from `LabiniusRootEncoding`.
F128 has `(h,a,w_H)=(128,7,16)`; F192 has `(192,8,8)`.
Canonical F coefficients are fixed-width little-endian and MUST reject a
non-canonical encoding. B elements are 21 little-endian bytes with the six
unused high bits zero. Q, K and tag-1 KA use unsigned offsets
`value+2^(e-1)` in `ceil(e/8)` little-endian bytes. An offset at least `2^e`
MUST reject; these are precisely the ranges enforced by `LoweredPublic::new`,
not an additional honest-witness range.

In the table, P is a proof message, V a verifier challenge, and pub an absorbed
public message. Public messages and challenges occupy zero proof bytes.

| Step | Actor and value | Encoding and proof length | Site or label |
| --- | --- | --- | --- |
| 1 | pub: domain, admitted identity, base tag, image commitment, r, t | Length-prefixed domain; canonical admitted identity; tag; oracle binding; low-coordinate-first host words | `akita/labinius/root-reduction/v1`, statement binding |
| 2 | P: partials; V: batching point; P/V: frontend rounds; P: terminal source value | `h*w_H` partial bytes; a uniform B coordinates; n rounds of two B coefficients; one B terminal; total `h*w_H+(2n+1)*21` | Existing `prove_frontend` / `verify_frontend` order |
| 3 | P: left expansion U | C canonical B elements; `21*C` bytes; `verify_left_expansion` MUST pass | Left expansion message |
| 4 | V: fold challenges | C challenges from the admitted binary sampler; profile, count and label absorbed before its root draw | `akita/labinius/root-fold/v1` |
| 5 | Oracle: response commitment | Scheme-dependent; transparent oracle sends `2^nu` bytes | Response binding call |
| 6 | P: QA, optional KA, Q, K | `n_A*(D-1)*f` canonical F bytes; tag 1 adds `n_A*D*ceil(e_KA/8)` offset bytes; then `(d-1)*q` and `d*k_Q` offset bytes | Clear lowered witness messages; exact KA grammar in [small-modulus spec](labinius-small-modulus-root.md#exact-auxiliary-message-grammar-and-matrix-identity) |
| 7 | V: alpha, xi, gamma | Three fresh F draws at distinct sites, in that order | `LRRD`, details 0 (alpha), 1 (xi), 2 (gamma) |
| 8 | P: y_Y | One canonical F element, f bytes; define `s=c_pub-y_Y` | Image weighted-sum message |
| 9 | V: tau, beta | nu fresh F coordinates, lowest index bit first, then one fresh F beta | `LRRD`, detail 3 / round i (tau_i), detail 4 (beta) |
| 10 | pub: combined instance header; P/V: combined rounds | Header binds kind, invocation 0, nu and base, followed by public absorption of beta*s; nu rounds sending `2^b+1` F coefficients each; `nu*(2^b+1)*f` bytes | Root sumcheck instance domain; LRSC diagnostics |
| 11 | P: w_eval | One canonical F element, f bytes; MUST satisfy combined terminal with structured `kw` | Response evaluation message |
| 12 | pub: product instance header; P/V: product rounds | Header binds kind, invocation 1 and mu, followed by public absorption of y_Y; mu rounds sending two F coefficients each; `2*mu*f` bytes; initial claim y_Y | Root sumcheck instance domain; LRSC diagnostics |
| 13 | P: y_eval | One canonical F element, f bytes; MUST satisfy product terminal with structured `ky` | Image evaluation message |
| 14 | Oracle: discharge both claims | Scheme-dependent; transparent oracle emits nothing | Final oracle call |

The prover MUST compute `fold_integer` and reject without retry if any
coefficient leaves the admitted interval; it then calls `encode_witness`.
The same response determines QA, Q and K, and tag-1 KA. The verifier builds `LoweredPublic`
only after receiving these messages and drawing alpha, xi and gamma. The
combined terminal MUST equal
`combined_terminal(base,tau,rho,beta,w_eval,kw)` with
`kw=witness_weight_mle(layout,public,setup,rho)`. The product terminal MUST
equal `product_terminal(y_eval,ky)` with
`ky=image_weight_mle(layout,public,rho')`. Neither check may be skipped at a
zero public weight.

Coefficient challenge sites use family `0x4c525244` (`LRRD`), with the
detail/round assignments above. Sumcheck sites use family `0x4c525343`
(`LRSC`). These site assignments identify diagnostics; the absorbed session,
statement and instance headers provide cryptographic domain separation.

The versioned sumcheck instance header is a cryptographic public binding.
Diagnostic `ProtocolSiteId` records alone absorb nothing; changing an invocation
number only in those records separates nothing. The canonical binding concatenates
`akita/labinius/root-sumcheck-instance/v1`, the kind byte (0 combined,
1 product), invocation as u32 little-endian, dimension as u64 little-endian,
and one base byte (1, 2 or 4 for combined; 0 for product), in one public call.
See [root sumchecks](labinius-root-sumcheck.md).
Both prover entry points and verifier replay MUST absorb that header before
rounds, including for zero-round instances. Round messages MUST precede their
challenges. The public statement and witness messages already fix each input
claim and public weight before its sumcheck.

Byte entry points MUST reject truncation, non-canonical atoms and trailing
bytes. Proof failures return `AkitaError::InvalidProof`; invalid public setup
or geometry returns the applicable input/setup error. Statement-sized
allocation and all size arithmetic MUST follow the verifier no-panic contract.

## Soundness ledger

The following bounds concern the interactive protocol with fresh challenges
uniform conditional on the preceding transcript, fixed tables and
binding/evaluation-sound oracles. For the Fiat–Shamir implementation, this is an
interactive accounting ledger, not a random-oracle security theorem. B and F
are distinct fields: `|B|=2^162` and `|F|=P`. Fold challenges come from the
finite support family S, embedded in B by parity, rather than uniformly from B.

| Challenge | Fixed before it | Bad event and derivation | Error term |
| --- | --- | --- | --- |
| Frontend batching point, a B coordinates | Setup, image commitment, host statement and partials | An incorrect fixed partial vector has a nonzero multilinear discrepancy; total degree at most a | `a/|B|` |
| Frontend round `z_i` in B | Partials, batching point and previous rounds; this round's two coefficients | A false degree-two product-sumcheck transition matches at a random point | `2/|B|` per round, `2n/|B|` total |
| Fold challenge vector | Setup, image commitment, binary claim and all U | Conditional fold comparison: a nonzero B-linear discrepancy has at most one family member per conditioned coordinate because parity is injective; union over C coordinates | `C/|S|`; extraction and its loss remain open |
| alpha in F | Y, W commitment, U, folds and QA/Q/K (plus tag-1 KA) | A false unreduced A row has degree at most `2D-2`; choose one false row, with no n_A factor | `(2D-2)/P` |
| xi in F | Same fixed witness messages, and alpha | A false parity residual modulo P has degree at most `2d-2=322` | `322/P` |
| gamma in F | All row polynomials and their alpha/xi evaluations | A nonzero combination of n_A A evaluations and the parity evaluation has degree at most n_A | `n_A/P` |
| tau, nu F coordinates | W, Y, public weights, c_pub and y_Y, hence s | An invalid digit makes `Z(tau)=sum_x eq(tau,x) P_b(W(x))` a nonzero multilinear polynomial of total degree at most nu | `nu/P` |
| beta in F | W, Y, weights, s and tau | `Z(tau)+beta*(L-s)=0`, `L=<W,K_W>`; unless both coefficients vanish, at most one beta solves it | `1/P` |
| Combined round `rho_i` in F | Tables, weights, tau, beta, s, instance header and current round message | Degree at most `2^b+1` per round | `(2^b+1)/P` per round; `nu*(2^b+1)/P` total |
| Product round `rho'_i` in F | Tables, weights, y_Y, completed combined instance, w_eval, header and current round message | Degree at most two per round | `2/P` per round; `2mu/P` total |
| Oracle challenges | Both commitments and both output claims | Binding and evaluation errors of the selected oracle | `epsilon_bind+epsilon_eval` |

The degree bounds follow from multiplication of degree-D-minus-one A, response
and image polynomials, and degree-d-minus-one parity polynomials. A monic
trinomial times a length-D-minus-one quotient has the same maximum degree
`2D-2`; Phi times Q has degree 322. Gamma combines n_A A rows with its final
power `gamma^n_A` on the parity row. Characteristic MUST exceed `2^b+1` so
combined interpolation at distinct nodes `0..=2^b+1` is defined.

Beta MUST be fresh, uniform conditional on the preceding transcript. The
claim s MUST be fixed before beta: for nonzero beta an adaptive choice
`s=L+Z(tau)/beta` would satisfy the equation for an invalid alphabet. The
`nu/P+1/P` bound is conservative: if L=s and Z is nonzero, no beta works.

Define

```text
E_lowered = [2D-2 + 322 + n_A + nu + 1 + nu*(2^b+1) + 2mu] / P;
E_frontend = (a+2n) / 2^162;
E_accounted = E_frontend + C/|S| + E_lowered
              + epsilon_bind + epsilon_eval.
```

`E_lowered` is the conditional composition error for the lowered relation.
`E_accounted` also records the frontend arithmetic and conditional fold terms;
it does not close the source-extraction or Fiat–Shamir obligations below.
There is no error for the exact host reconstruction, left expansion, range
checks, integer no-wrap implication or canonical encoding.

At `(log_num_cells,log_fold_width,lambda_fold)=(22,8,128)`, tag 0 gives
`m=4096`, `C=256`, `n_A=1`, `D=648`, `n=22`, `mu=18`,
`P=340282366920938463463374607427473266697`. The bounded-weight-46 family has

```text
|S| = sum_(j=0)^46 binomial(162,j)
    = 107268495235892251993000716928106195079952;
-log2(C/|S|) = 128.300278 bits.
```

The following bits are `-log2(error)` before oracle errors. F128 uses a=7 and
F192 uses a=8; their accounted values agree at the displayed precision.

| b | nu | Numerator of E_lowered | Lowered bits | Accounted bits, F128 | Accounted bits, F192 |
| --- | --- | --- | --- | --- | --- |
| 1 | 26 | 1758 | 117.220281 | 117.219614 | 117.219614 |
| 2 | 25 | 1804 | 117.183016 | 117.182367 | 117.182367 |
| 4 | 24 | 2086 | 116.973477 | 116.972915 | 116.972915 |

The 128-bit fold allocation and certified SIS profile do not imply a 128-bit
bound for this union: polynomial identity tests over the 128-bit coefficient
prime determine the displayed arithmetic bound.

For tag 1, the extra KA term has degree below D, so the alpha bound is
unchanged. Rank 3 changes the gamma term to `3/P`; at the sample geometry
mu grows from 18 to 20 while nu stays fixed. The numerator of E_lowered
therefore grows by six. Exact auxiliary bytes follow the admitted shape. The additional derivation
bias and conditional cancellation/extraction argument are owned by
[the small-modulus specification](labinius-small-modulus-root.md#soundness-ledger-delta).
They do not close the composition obligations below.

## What acceptance proves, and open obligations

The conditional theorem intended by this composition is: if the verifier
accepts and the two oracles are binding and evaluation-sound, then except with
probability at most `E_lowered+epsilon_bind+epsilon_eval`, the fixed committed
W has the required alphabet and the fixed committed W and Y satisfy the
lowered polynomial relations, with the received Q/K (and tag-1 KA) in their
enforced ranges.
By the [lowered-root specification](labinius-lowered-root.md), decoding W then
gives the clear A endpoint for tag 0 and the lifted A relation with the
selected-base carry envelope for tag 1; both give the F162 parity endpoint.
The tag-1 clear endpoint uses the narrower one-bit envelope as recorded in the
small-modulus specification, so arbitrary accepting lifted witnesses need not
satisfy that narrower envelope. The admitted no-wrap inequality promotes the checked parity identity modulo P to
an integer identity before reducing modulo two. Alphabet-valid padding has
zero public relation weight and is permitted.

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
- Fiat–Shamir security in the random-oracle model for this multi-round protocol,
  including the seed expander, indexed fold substreams and oracle composition.
- A concrete binding and evaluation-sound polynomial commitment scheme and
  any setup-contribution opening replacing the materialized matrix scan.

The root transcript closes admitted-identity absorption and the order of
statement, witness and challenge binding. It does not close those remaining
obligations merely by defining an encoding or passing differential tests.

## Completeness and verifier cost

For valid setup and source input with an honest oracle, the algebraic protocol
is complete whenever the integer fold lies in the admitted interval. The
honest prover returns an error without retry otherwise. Its probability source
is the admitted binary challenge distribution, not a sumcheck event. Both
admitted profiles additionally enforce the deterministic envelope
`C*Gamma_inf <= min(upper,-lower)`; with binary source coefficients this makes
the out-of-interval probability zero. Allocation failure and malformed caller
input are execution errors rather than probabilistic completeness losses.

Let `h=H::ROWS`, `dc=e_v/b`, `P_c=next_power_of_two(D)`, and
`A_pad=next_power_of_two(n_A*m*D)`. Excluding the oracle, the verifier performs
these operations; counts describe field work and omit hashing and decoding:

| Work | Operation count or bound |
| --- | --- |
| Frontend reconstruction and rounds | h host-source contributions, a B batching coordinates, n degree-two evaluations; tensor terminal `O(n*h)` B multiplications and `O(n*h^2)` binary XORs |
| Left expansion | Equality expansion and C B multiply-adds; `O(C)` B operations |
| Public A contractions and quotients | `n_A*m*D` Horner steps, `n_A*(D-1)` quotient Horner steps, and `n_A*m` row-batching multiply-adds |
| Public parity and challenge lifts | `k*m*d` binary-row Horner steps; C embedded degree-D and two degree-d Horner evaluations; `(2d-1)` Q/K Horner steps |
| Public constant | `m*D` coefficient-weight/offset contributions, n_A quotient contributions, powers of lengths D, d, n_A+1 and dc |
| Combined replay and terminal | nu degree-`2^b+1` evaluations, an O(nu) equality evaluation and `2^b` alphabet factors |
| Product replay and terminal | mu degree-two evaluations and one multiplication |
| Response structured weight | Canonical setup-weight preparation/materialization on A_pad entries and `n_A*m*D` matrix multiply-adds; parity contractions over dc, d and k*m entries |
| Image structured weight | D coefficient contractions and C*n_A image-entry contractions |

For the current scalar `eq_eval_at_index` loops, the parity part of the
response structured evaluator costs
`O(dc*log(dc)+d*log(P_c/k)+k*m*log(k*m))` F operations after the public binary
rows are cached. The A part costs `O(A_pad+n_A*m*D)` F work and O(A_pad)
workspace. The image structured evaluator costs
`O(D*log(P_c)+C*n_A*(mu-log(P_c)))` F operations and constant extra workspace.
Neither structured evaluation scans the response table of `2^nu` entries.
Public construction still scans the materialized setup and scalar rows; this
slice makes no sublinear setup-verifier claim. Transparent oracle discharge
adds direct MLE work linear in `2^nu+2^mu`, outside these reduction counts.

## Exact reduction wire size

The exact reduction-owned byte count, excluding oracle bytes, is

```text
h*w_H + (2n+1)*21 + 21*C
+ f*n_A*(D-1) + (d-1)*ceil(e_Q/8) + d*ceil(e_K/8)
+ [tag 1 only: n_A*D*ceil(e_KA/8)]
+ f*[3 + nu*(2^b+1) + 2mu].
```

This formula needs only shape, base and H; it requires no materialized matrix.
All products, sums, widths and powers of two MUST use checked arithmetic.
Absorbed statement bytes, headers and challenges contribute zero proof bytes.
For tag 0 at the first real geometry, Q/K byte widths are both five for every base:
`(e_Q,e_K)=(39,38),(40,38),(40,40)` for b = 1, 2, 4 respectively.

| Message region | b=1 | b=2 | b=4 |
| --- | ---: | ---: | ---: |
| Frontend partials, F128 | 2048 | 2048 | 2048 |
| Frontend partials, F192 | 1536 | 1536 | 1536 |
| Frontend round bodies and source terminal | 945 | 945 | 945 |
| U | 5376 | 5376 | 5376 |
| QA | 10352 | 10352 | 10352 |
| Q | 805 | 805 | 805 |
| K | 810 | 810 | 810 |
| y_Y | 16 | 16 | 16 |
| Combined round bodies | 1248 | 2000 | 6528 |
| w_eval | 16 | 16 | 16 |
| Product round bodies | 576 | 576 | 576 |
| y_eval | 16 | 16 | 16 |
| **Total F128** | **22208** | **22960** | **27488** |
| **Total F192** | **21696** | **22448** | **26976** |

The transparent oracle adds respectively 67108864, 33554432 and 16777216
response bytes at this profile. The 262144 image coefficients are absorbed
public data, not proof bytes. The byte wrappers add no framing or final marker.
EOF is the proof boundary.
