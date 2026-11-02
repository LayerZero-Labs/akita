# Spec: binary source grouped opening

| Field | Value |
| --- | --- |
| Author(s) | Akita contributors |
| Created | 2026-10-28 |
| Status | implemented experimental path; security admission pending |
| PR | Base root: [#290](https://github.com/LayerZero-Labs/akita/pull/290); field switch and exact-divisibility projections on the current branch |
| Supersedes | None |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

## Scope and selection

A binary source is a commitment-group kind in the canonical grouped PCS. The
caller commits, supplies an optional binary word claim over GF(2^128) and/or
prime evaluation claims, and uses the
ordinary grouped opening entry points inside the caller's transcript. A carrier-level
field switch produces an internal binary terminal before the root checks
the source relation and repacks its bounded coefficients into the ordinary
negacyclic witness. The existing grouped recursion and terminal then finish
the same proof.

The implementation is available through the default-off `labinius-trinomial`
feature. Selecting a binary source handle and a word or prime claim selects the
extension at runtime. Enabling the feature alone does not select it. Ordinary
groups retain their existing encoding and protocol path.

This is an experimental arithmetic and protocol fixture, with no security
claim. It uses `P = 2^128 - 275`, base-field challenges (`E = F`), and the
Lagrange basis. It admits one binary group first, with at most one ordinary
group second. A binary group may contain several polynomials. It requires an ordinary successor fold whose predecessor setup
contribution is evaluated directly. It does not add a production binary
planner profile or certify a new schedule.

## Commitment and word-claim interfaces

The [binary commitment manifest](../crates/akita-types/src/binary_source/commitment_manifest.rs)
freezes a rank-three A matrix over the degree-648 minus trinomial ring modulo
`q = 14,253,409`, and an uncompressed rank-two B matrix over the degree-64
negacyclic ring modulo P. Its ordinary carrier profile identifies only the
suffix schedule. The actual A/B setup identities are separate descriptor
fields. Typed setup labels distinguish the q-valued A stream, the P-valued
image commitment and the P-valued opening commitment.

The relevant new declarations are:

```rust
impl BinaryCommittedGroupManifest {
    pub fn commitment_profiles(
        index: &BinaryIndexMap,
        polynomials: usize,
        setup_group: u64,
    ) -> Result<BinaryCommitmentProfiles, AkitaError>;

    pub fn new(
        descriptor: BinarySourceDescriptor,
        setup_group: u64,
        carrier_profile: &GroupCommitPhaseParams,
    ) -> Result<Self, AkitaError>;
}

impl<F, E> CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + 'static,
    E: Field + 'static,
{
    pub fn import_binary_source(
        &self,
        manifest: BinaryCommittedGroupManifest,
        carrier_profile: GroupCommitPhaseParams,
        words: Vec<Vec<u128>>,
    ) -> Result<SourceHandle<F, E>, AkitaError>;
}

impl BinaryWordClaim {
    pub fn new(
        descriptor: &BinarySourceDescriptor,
        polynomial: [u8; 32],
        point: Vec<[u8; 16]>,
        value: [u8; 16],
    ) -> Result<Self, AkitaError>;
}

impl<'a, E: Field, F: Field + CanonicalEncoding>
    PolynomialGroupClaims<'a, E, CommittedGroup<F>>
{
    pub fn new_binary(
        commitment: CommittedGroup<F>,
        claim: BinaryWordClaim,
    ) -> Result<Self, AkitaError>;

    pub fn new_binary_planes(
        commitment: CommittedGroup<F>,
        word: Option<BinaryWordClaim>,
        primes: Vec<PrimePlaneClaim>,
    ) -> Result<Self, AkitaError>;
}
```

Each inner `words` vector supplies one polynomial's logical words. The CPU
backend applies the descriptor's signed index map, forms `T = A s mod q`,
chooses centered representatives, and stores T as three-byte offset words.
The ordinary B commitment binds T's eight balanced base-eight digits. T remains
a private hint. The public output is a `CommittedGroup` carrying the ordinary
payload and the typed binary manifest; its binary wire format carries both.
The canonical backend `commit` operation uses `GroupContext::explicit` with
the manifest's carrier profile.

`BinaryWordClaim` identifies one polynomial and claims its logical word
multilinear extension (MLE) at `point`. Coordinates and value use the unique
16-byte, low-degree-first polynomial-basis encoding of
`H = GF(2^128) = F2[X]/(X^128 + X^7 + X^2 + X + 1)`. No wider word field is
admitted. The constructor checks polynomial ownership and the descriptor's
complete point dimension. The descriptor fixes the public coordinate split
between intra-carrier word slots and carrier addresses; callers cannot change
that split independently of the commitment contract.

The opening accepts at most one word claim per polynomial identifier.
`OpeningClaims::from_groups` rejects a second binary claim on the same
polynomial with `AkitaError::InvalidInput`, including claims in separate groups.
Applications reduce multiple points before invoking the opening. Akita provides
no application-side H reduction helper.

The canonical word-claim bytes bind the polynomial identifier, complete point
and value at order zero. `BinaryTerminalClaim` is produced inside the opening
at order two; no caller-facing grouped opening accepts one directly. Its point
and value use canonical 21-byte GF(2^162) encodings with zero unused high bits.
The root authenticates the produced whole-carrier terminal through its existing
parity and value relations.

The result enters `OpeningClaims::from_groups`, the existing selection APIs,
`AkitaCommitmentScheme::batched_prove_in_session` and
`AkitaVerifier::batched_verify_in_session`. There is one proof type, one
caller transcript and one grinding-plan cursor. No standalone binary PCS,
nested session seed or separate root transcript is invoked. The caller can
continue using its live transcript after verification and checks exhaustion
at its chosen proof boundary.

## Opening order

The following table records the restricted protocol order. The order numbers
identify the state transitions of the binary source opening contract.

| Order | Implemented action | Fixed before the next challenge |
| --- | --- | --- |
| 0 | Bind caller context, setup, selected schedule, group kinds, commitments, binary manifest, the optional word claim, every unique prime plane's polynomial, canonical P point, tensor-component tag and value, and any ordinary point/value. | The complete public statement. |
| 2 | If a word claim is selected, send 128 H carrier partials; draw seven B row-selector coordinates; run n degree-two B sumcheck rounds; send the whole-carrier terminal value. | All partials precede the row selector; each round message precedes its challenge; the terminal precedes U and the column challenges. |
| 3 | Commit the selected U table and every EP plane in one ordinary D payload. U uses one digit; EP coefficients use 43 centered balanced digits. | The complete D payload precedes all column challenges. |
| 4 | Send the ordinary response nonce and draw the indexed binary column challenges; the mixed ordinary group uses complete preimages rather than a column-fold response. | The accepted bounded response Z. |
| 5 | Commit the outgoing witness containing Z/T digits for every polynomial, selected U/EP views and selected K/K_val; a mixed opening also includes ordinary commitment preimages. | Every witness segment, including Z and T, precedes projection rows. |
| 6 | Draw seven exactly uniform q-valued rows J, then send the seven signed integer values y in order. Check each encoding, bound and divisibility by q. | The outgoing commitment precedes J; all seven y precede output batching. |
| 7 | Draw the shared ring-switch challenge, range-image point and independent relation-batching weights, whose first seven entries are the output scalars mu; form rho = sum_l mu_l J_l and evaluate direct setup and public weights. | All tested tables, commitments and projection values. |
| 8 | Run the ordinary Stage 1 range-image sumcheck with the base-eight digit alphabet. | The range-image terminal precedes Stage 2 batching. |
| 9 | Run one fused Stage 2 for sum_l mu_l (<J_l, RA> - y_l) = 0, B/D authentication, selected parity/value and prime response/reading equations, padding, and range-image binding; include ordinary relations when selected. | One final opening of the same outgoing witness. |
| 10 | Omit root Stage 3: direct setup evaluation at order 7 replaces its adjoint proof and prefix claims. | The authenticated root opening passed to the suffix. |
| 11 | Continue ordinary grouped recursion and terminal verification. | Return the live caller transcript and, through the readings API, authenticated readings with the conditional-agreement contract; the caller checks exact proof exhaustion. |

Order 1 is absent: application reductions happen before this opening. There
is no committed quotient or quotient fallback. Direct public weights replace
an equality tree or a new bilinear wiring compiler.

## Carrier-level field switch

Let the descriptor's word width be a power of two `ell <= 128`, and let
`w_per = 128/ell`. The padded source has `N = 2^n` carriers in canonical index
order. The logical word claim point splits as
`r' = (r_intra, r_addr)`, with `log2(w_per)` low coordinates selecting a word
slot and n remaining coordinates selecting a carrier. Logical words occupy
the low ell bits of H. The descriptor's
[`claim_variable_split`](../crates/akita-types/src/binary_source/index.rs)
publishes this split.

For carrier payload bit `s = w*ell + u`, define the public H coefficient and
F2-linear map

```text
c_s = eq_H(r_intra, w) X^u
L(x) = sum_(s=0..127) bit_s(x) c_s.
```

The logical word MLE value is therefore
`v = sum_i eq_H(r_addr,i) L(carrier_i)`. When `ell = 128`, `r_intra` is empty
and L is the identity. Narrow words change only this public map; the source
terminal still evaluates complete carriers.

Write `e_i = eq_H(r_addr,i)`, and define the ordinary carrier partials

```text
q[k] = XOR_(i : bit_k(e_i)=1) carrier_i,  k = 0..127.
```

Because L commutes with XOR, the verifier checks the bound word value by
`v = sum_(k=0..127) X^k L(q[k])`. This is implemented by
[`BinaryWordClaim::reconstructed_value`](../crates/akita-types/src/binary_source/word_claim.rs).
It does not apply L to the authenticated source terminal.

The order-two transcript sequence is exact:

1. Exchange `q[0]` through `q[127]`, each as 16 canonical H bytes. All 2,048
   bytes enter the caller's transcript before any row-selector coordinate.
2. Check `v = sum_k X^k L(q[k])`. Failure is `AkitaError::InvalidProof` on
   verification.
3. Draw seven independent uniform B coordinates `lambda[0..7]`, where
   `B = GF(2^162)`. A challenge uses 162 bits from a transcript challenge block,
   masking unused high bits; there is no odd-prime reduction or modulo bias.
4. Start the sumcheck at `a = sum_k eq_B(lambda,k) phi(q[k])`, where phi is
   F2-linear low-coordinate inclusion from H to B, not a field homomorphism.
   The relation is
   `a = sum_i W(i) phi(carrier_i)`, with
   `W(i) = sum_k eq_B(lambda,k) bit_k(e_i)`.
5. For each carrier-address axis `j = 0..n`, exchange the quadratic
   coefficients `c0` and `c2` as 21 bytes each, then draw uniform `z[j]` in B.
   The linear coefficient is `c1 = a + c2`, since characteristic two gives
   `g(0)+g(1) = c1+c2 = a`. Update `a = c0 + c1*z[j] + c2*z[j]^2`.
6. Exchange the mandatory 21-byte whole-carrier source value t. Check
   `a = W_tilde(z) t` using the transparent weight evaluator. Retain t as the
   internal root obligation even if `W_tilde(z) = 0`. Never divide by the
   transparent weight and never omit or replace the terminal when it is zero.

The canonical prover and verifier use the existing field-switch arithmetic
and packed kernels from `akita-algebra`. They exchange messages through the
ordinary proof stream inside the caller's transcript. They do not invoke the
standalone driver or depend on its crates.

### Terminal coordinate mapping

For `C = 2^c` commitment columns, the descriptor maps carrier index i to column
`b = i mod C` in the low bits and scalar row `r = floor(i/C)` above them.
Point coordinate zero controls the low table-index bit in both the arithmetic
field switch and the root equality tensors. The n sumcheck coordinates split
as

```text
rho_col = z[..c]
rho_row = z[c..]
terminal.value = t = sum_i eq_B(z,i) phi(carrier_i).
```

The seven lambda coordinates batch partial rows only. They contribute no
source-address coordinate. The root's
[`binary_row_partials`](../crates/akita-cpu-backend/src/binary_root/evaluation.rs)
contracts over scalar rows inside each column, and
[`relation_weights`](../crates/akita-types/src/binary_source/evaluation_weights.rs)
uses the same separate row and column equality tensors. Thus its final
column contraction authenticates precisely the terminal above, including the
published permutation between canonical carriers and column-contiguous
source packing.

### Frontend soundness and proof bytes

For N carriers, the frontend's interactive algebraic error is at most
`(7 + 2n)/2^162`: `7/2^162` for the degree-seven row-selector test and
`2n/2^162` for the n degree-two product-sumcheck rounds. This is the frontend
term, separate from the root authentication, setup, binding and hash terms.
The public F2-linear map L introduces no additional random test. Any H
multi-point reduction belongs to the application and its error ledger; this
opening performs none and charges no H-reduction term. These algebraic bounds
do not certify a production schedule or an overall Fiat-Shamir security level.

Order two adds exactly
`128*16 + 21*(2n+1) = 2048 + 21*(2n+1)` proof bytes: 128 H partials, two B
coefficients per round and one B source terminal. Challenges and the public
word claim add no proof bytes. For the fixture `n = 3`, the frontend adds
2,195 bytes to both binary-only and mixed openings. Subsequent transcript
challenges change, so downstream variable-length encodings can also affect
the measured total proof size.

## Relations and authenticated repacking

All identities below use integer coefficient lifts before reduction modulo P.
Let `Psi = X^648 - X^324 + 1`, `Phi = Z^162 + Z^81 + 1`, and let
`rem` denote the reduced remainder for its subscript. The root enforces:

```text
RA_i = rem_Psi(sum_j A_ij z_j - sum_b iota(c_b) T_(b,i))
y_l = <J_l, RA>,  y_l = 0 (mod q),  l = 0..6
```

When a word claim is present, the root also enforces:

```text
rem_Phi(sum_r Br V0_r - sum_b U_b c_b) = 2 K
rem_Phi(sum_b E_b U_b) - v = 2 K_val
```

Here iota packs the scalar challenge into the first component of the source
ring's four-component module view. For word claims, Br and E_b are direct binary equality
weights from the terminal point, V0 is the low response-digit view, and c is
the binary fold challenge. K has nine balanced base-eight digits per
coefficient and K_val has six. These carry tables are committed at order 5 and
never transmitted in the clear. A word opening has five base binary segments:
response, image, U, K and K_val, with EP appended when prime planes are selected.
A prime-only opening has response, image and EP segments. U has no Booleanity
predicate; admission uses its full malicious digit interval `[-4, 3]`.

### Exact divisibility projections

After fixing the accepted response nonce and committing the outgoing witness,
[BinaryProjectionRows](../crates/akita-types/src/binary_source/projection_rows.rs)
binds the domain label `binary exact divisibility rows` and draws seven rows
of `a = rank * n_poly * 648` coefficients each, in projection, polynomial,
rank-row and natural coefficient order. One set of seven rows covers the entire
group, including prime-only openings; the seven messages remain 84 bytes
independently of the polynomial count. Each 32-byte transcript challenge block supplies ten
nonoverlapping little-endian three-byte candidates; its final two bytes are
discarded. Candidates `0 <= x < q` are accepted unchanged; candidates
`q <= x <= 2^24-1` are rejected. Each accepted entry is exactly uniform in
`{0, ..., q-1}`, with no modular reduction or bias. After 256 rejected
candidates for one entry, the sampler returns a typed error rather than
looping indefinitely. J is outside the response retry loop: each attempted
response has no access to J.

The prover next binds `binary exact divisibility values` and sends
`y[0]` through `y[6]`, each in exactly twelve bytes, for 84 proof bytes total.
The first 95 bits encode the unsigned magnitude in little-endian order; bit
95 is the sign. Zero has its sign bit clear. The verifier rejects any wrong
length, negative zero, magnitude exceeding `U_J`, or value not divisible by q.
The bounds below ensure that every honest y fits the unique encoding.

Only after all seven messages enter the transcript does order 7 draw the
ring-switch scalar alpha, the range-image point tau0, then the independent
relation-batching coordinates tau1. Its first seven entries are
`mu[0]` through `mu[6]`. Stage 2 authenticates

```text
sum_l mu_l (<J_l, RA> - y_l) = 0  over P
rho[i][t] = sum_l mu_l J_l[i][t]
```

The direct A evaluator takes this explicit rho. The public linear target is
`sum_l mu_l y_l`, lifted with its signed integer interpretation into P. The
remaining relation coordinates authenticate parity, value, B/D, padding and
any ordinary rows. No quotient table, quotient bound or quotient relation
participates in this path.

If the seven individual projection identities hold over P, then they hold
over the integers: `|<J_l, RA> - y_l| <= 2 U_J < P`. Their checked divisibility
therefore implies seven zero dot products against `RA mod q`. For any fixed
nonzero residual modulo the prime q, one exactly uniform row annihilates it
with probability `1/q`; seven independent rows give `q^-7`. The outgoing
commitment and accepted response must precede J for this argument, and every
y must precede mu for the independent output-batching argument.

[BinaryRootPublic](../crates/akita-types/src/binary_source/root_public.rs)
constructs the shared linear relation, including direct authentication of T
against the public B payload and the U view against D. Stage 1 range-checks
the outgoing digit table, and Stage 2 binds the range image and linear
identities to its one witness opening. The
[repacking plan](../crates/akita-types/src/binary_source/root_plan.rs) gives
every native coefficient its ordinary witness address and authenticates zero
padding. This coefficient repacking is not a ring homomorphism.

The mixed specialization samples only the binary column challenges at order 4.
It fixes complete ordinary preimages at order 5 instead of folding an ordinary
response; this costs more witness space and restricts the admitted source.
A mixed opening adds the ordinary source S, ordinary A-image T digits, and
both ordinary commitment-compression bit tables to the same outgoing witness.
Direct Stage 2 weights enforce A, B, both compression maps and the ordinary
evaluation. The compression bits satisfy their existing `{-1, 0}` predicate;
this predicate is separate from the binary U contract. The supported ordinary
profile has one polynomial, one inner digit, base-eight outer digits and
exactly two 128-bit compression maps. Every native ring degree must be at most
648, as budgeted by the root grinding plan. Its source coefficients must fit one
balanced base-eight digit. The current mixed fixture uses only zero and one.
Unsupported geometries produce typed errors.

## Checked fixture bounds

The fixture uses 128-bit words, eight padded carriers, `M = 4` scalar rows,
`C = 2` columns and `m = 1` degree-648 source row. The challenge operator bound
is `Gamma = 92`. The matrix bound is `B_A = 7,126,704`, the response bound is
`B_Z = 2,340`, and the enforced image magnitude is `B_T = 9,586,980`.

The admission implementations are
[BinaryProjectionBounds](../crates/akita-types/src/binary_source/projection.rs) and
[BinaryEvaluationBounds](../crates/akita-types/src/binary_source/evaluation_bounds.rs).
Both use checked integer arithmetic and reject equality at a strict no-wrap
threshold. The projection bounds are

```text
L_R = 972 m B_A B_Z + C Gamma B_T
a = rank * n_poly * 648
U_J = (q-1) a L_R
2 U_J < P,  U_J < 2^95.
```

The 972 factor is the maximum absolute row sum of the degree-648 integer
multiplication kernel modulo Psi. Reducing product monomials uses
`X^648 = X^324 - 1` and `X^972 = -1`; counting the contributing coefficient
pairs gives maximum weight 972. The centered setup interval gives
`B_A = (q-1)/2 = 7,126,704`. Four balanced base-eight response digits have
interval `[-4*(1+8+64+512), 3*(1+8+64+512)] = [-2340, 1755]`, hence
`B_Z = 2340`. Eight image digits have interval
`[-4*(8^8-1)/7, 3*(8^8-1)/7] = [-9,586,980, 7,190,235]`, hence
`B_T = 9,586,980`; substituting the smaller honest centered interval here
would be unsound. The binary challenge has weight 46 and coefficient
multiplication operator bound `Gamma = 2*46 = 92`; the factor two accounts
for half-degree monomial multiplication in the trinomial basis.

For `P = 340282366920938463463374607431768211181` and rank three
and one polynomial (`a = 1944`), the fixture and upper envelopes are:

| Geometry `(m, C)` | `L_R` | `U_J` | `2 U_J` |
| --- | ---: | ---: | ---: |
| `(1, 2)` fixture | 16,211,309,718,240 | 449,193,104,205,686,897,172,480 | 898,386,208,411,373,794,344,960 |
| `(4096, 256)` | 66,394,525,036,769,280 | 1,839,701,006,388,918,996,628,930,560 | 3,679,402,012,777,837,993,257,861,120 |
| `(8192, 256)` upper envelope | 132,788,824,280,985,600 | 3,679,395,756,376,625,911,509,811,200 | 7,358,791,512,753,251,823,019,622,400 |

The existing parity/value carry admission requires power-of-two `M = 4m`
and C. Its positive carry reaches restrict these to `m <= 8192` and
`C <= 256`. Every power-of-two pair within these limits passes both carry checks. Since
`L_R` is increasing in m and C, the last row dominates every admitted
rank-three single-polynomial geometry. Its `U_J` is below
`2^95 = 39,614,081,257,132,168,796,771,975,168`, and its strict lift margin is
`P - 2 U_J = 340282366913579671950621355608748588781`.
The constructor checks both inequalities independently for every geometry,
including other ranks and polynomial counts. At this upper envelope its
constructor-driven boundary test admits 10 polynomials and rejects 11. The
10-polynomial bound is `U_J = 36,793,957,563,766,259,115,098,112,000`;
the 11-polynomial bound is `40,473,353,320,142,885,026,607,923,200`,
which exceeds `2^95`. Both satisfy the weaker field-lift inequality. The 95-bit encoding
limit is a separate requirement: `2 U_J < P` alone does not establish it.

The fixture's unchanged parity/value carry bounds are:

| Quantity | Fixture result |
| --- | ---: |
| Honest K bound; enforced BK | 4,440; 76,695,844 |
| K no-wrap bound | 153,400,568 |
| K margin | 340282366920938463463374607431614810613 |
| Honest K_val bound; enforced BK_val | 486; 149,796 |
| Value bound `3 × 162 × C × 4 + 1 + 2 BK_val` | 303,481 |
| Value margin | 340282366920938463463374607431767907700 |

The honest carry bounds fit both positive and negative digit reaches. These
integer inequalities justify the selected exact lifts; they do not establish
SIS security, whole-schedule grinding security or application agreement.

### Projection error ledger

| Term | Bound and scope |
| --- | --- |
| Divisibility testing | `q^-7 = 2^-166.353625727542493448...`, approximately `8.3669807259e-51`, per attempt for a fixed nonzero joined residual modulo q. Repeated attempts require the corresponding grinding/Fiat-Shamir accounting. |
| Independent output batching | At most `1/|E| = 1/P`, approximately `2.9387358771e-39`, for a nonzero vector of seven projection-identity defects. The fixture uses `E = F`; this term is separate from `q^-7`. |
| Exact lift | Zero additional error when `2 U_J < P` and both envelopes are enforced. |
| Sampler exhaustion | Completeness abort at most `7 a (1-q/2^24)^256`. For rank three and one polynomial this is `13608 (1-q/2^24)^256`, approximately `3.4031e-207` or `2^-685.872...`; accepted samples remain unbiased. |

These root terms are additional to the frontend term `(7 + 2n)/2^162`,
ordinary sumcheck and batching terms, commitment binding, setup security,
grinding and Fiat-Shamir terms. This ledger does not claim an overall
166-bit security level: independent output batching alone has a roughly
128-bit error bound in this base-field fixture.

## Evidence and remaining interfaces

The canonical fixture is
[binary_source_end_to_end.rs](../crates/akita-pcs/tests/binary_source_end_to_end.rs).
The measured complete proof sizes for its 128-bit-word fixtures are:

| Protocol | Opening | Before merge | Merged |
| --- | --- | ---: | ---: |
| Current | Binary-only | 77,936 | 77,936 |
| Current | Mixed binary/ordinary | 92,101 | 92,101 |
| `dev-protocol` | Binary-only | 72,877 | 72,877 |
| `dev-protocol` | Mixed binary/ordinary | 87,119 | 87,119 |

The before column was remeasured at projection commit `b3c084284`. The merged
column was measured after combining projections and prime planes. Full proof
digests, not just lengths, match in both protocol modes when no plane is selected.
The 8-bit-word fixture measures 77,937 bytes in the current protocol and 72,898
bytes with `dev-protocol`.

The projection messages themselves add exactly 84 bytes. Removing the
committed quotient shrinks the natural outgoing witness by `rank*n_poly*648*8`
digits in the small fixture, or 15,552 digits per polynomial at rank three. Complete totals also depend on the
selected suffix geometry and later variable-length encodings after the changed
transcript challenges. The field-switch contribution remains exactly 2,195
bytes for every three-round fixture.

The fixture exercises the caller's prefix and
post-verification squeeze, commitment serialization, kind consistency,
statement mutations, malformed proof bytes, truncation and trailing bytes.
A test-only backend alters T, Z, U, K, K_val or padding before outgoing
commitment. It uses an incorrect public oracle to complete the adversarial
proof; verification must reject using the independently compiled correct oracle.
Each rejection runs behind a panic-catching boundary.

The suite also proves against a commitment to a different source
while retaining the original source witness and independently evaluated claim.
A second authentication test changes D before the challenges while retaining
the original U witness. The test backend completes both adversarial proofs;
the verifier must reject them even though prover and verifier share the
relation compiler. Removing B or D authentication locally makes the respective
rejection test fail.

A direct first-challenge probe invokes the canonical order-zero binder used by
both drivers, on prover and verifier channels. It changes only the manifest,
polynomial identifier, each intra-carrier or carrier-address point coordinate,
H value, binary commitment, or ordinary commitment, point coordinate or
evaluation, and checks that both roles produce the same changed challenge.
Independent wire replay in the frontend differential test checks that all
partials enter the transcript before lambda. Catalogs are loaded from external
schedule artifacts, including the separate dev-protocol catalog.
The structural source check scans the two root orchestration files for
standalone binary driver calls and nested channel creation. It does not inspect
transitive helper calls. Test outcomes belong in the accompanying validation
report.

The field-switch acceptance tests cover the frontend's additional obligations:

- `field_switch_differential_logical_words_and_carrier_terminals` independently
  evaluates logical word MLEs and the carrier terminal for random sources of
  widths 1, 8, 32 and 128. Its four index shapes include odd carrier-address
  dimensions. Independent wire replay also checks the partials-before-lambda
  binding and terminal coordinate split. A local negative control moved the
  lambda draws before partial absorption on both sides; this test failed at
  the independently replayed terminal point. The mutation was reverted.
- `field_switch_rejects_each_partial_and_round_message` alters each of 128
  partials, both messages of every sumcheck round, and the terminal bytes.
- `field_switch_claim_mutations_reject` alters the word value, each point
  coordinate (including every intra-carrier coordinate), and an admitted
  polynomial identifier.
- `field_switch_rejects_false_source_word` changes a source word after fixing
  its honest claim. `field_switch_binary_root_rejects_a_foreign_source_commitment`
  additionally tests authentication against a different committed source.
- `field_switch_zero_multiplier_authenticates_terminal` scripts a zero public
  weight, changes the retained source terminal, and checks rejection by the
  existing root value authentication without division.

The projection acceptance tests cover the root's additional obligations:

- `projection_two_polynomials_and_prime_plane_rejects_y_mutations` opens two
  committed polynomials with a prime plane and no word claim, then rejects
  mutations of each of the seven shared projection messages without panic.
- `projection_polynomial_count_admission_starts_rejecting_at_upper_envelope`
  discovers the largest admitted polynomial count through the constructor and
  checks both sides of the ten/eleven boundary.

- `projection_binary_group_opens_once_inside_the_callers_transcript` and the
  mixed fixture mutate every y by +1, +q, sign flip, an excessive magnitude,
  negative zero, deletion and insertion of a byte, and truncation. Verification
  and exact exhaustion run behind the same panic-catching boundary.
- `projection_challenges_bind_commitment_and_every_output` checks that changing
  the outgoing commitment changes J, and changing any one of the seven y
  values changes every mu on both channel roles.
- `projection_orchestrators_fix_witness_then_rows_then_outputs_then_batches`
  checks the fixation order in both actual drivers, including paired
  prover/verifier reorderings.
- `projection_sampler_rejects_q_and_byte_range_top` scripts q-1, q and
  `2^24-1` candidates and checks the accepted row and challenge-block count.
  `projection_sampler_exhaustion_rejects_without_looping_forever` checks the
  per-entry rejection cap.
- `projection_encoding_rejects_noncanonical_bounds_and_lengths` independently
  checks signed encoding, negative zero, out-of-range values, nonmultiples of
  q, and twelve-byte length enforcement.

The ordinary regression fixture is
[ordinary_grouped_root_bytes.rs](../crates/akita-pcs/tests/ordinary_grouped_root_bytes.rs),
which pins a 70,541-byte ordinary proof. Its feature-disabled and
feature-enabled executions are the compatibility gate when the extension is
unselected. Existing pinned fixtures and goldens remain part of that gate.


Both paired local ordering mutations were exercised before committing: moving
mu ahead of the y messages in both drivers, and moving J ahead of the outgoing
commitment in both drivers. Each made
`projection_orchestrators_fix_witness_then_rows_then_outputs_then_batches`
fail with exit code 101. Both mutations were reverted; the restored end-to-end
suite passes in both protocol modes.

The field switch produces `BinaryTerminalClaim` for the existing root. The
direct A-term function
[`binary_direct_a_weights`](../crates/akita-types/src/binary_source/root_plan.rs)
takes the explicit `rho[i][t]` projection weights over all polynomial/rank rows
and is the replacement point for a typed Stage 3 adjoint. Optional EP planes
authenticate prime claims as described below; no agreement proof is discharged
by a binary terminal alone.

## Prime evaluation planes

The default-off feature also exposes
`PolynomialGroupClaims::new_binary_planes(commitment, word, primes)`, where
`word: Option<BinaryWordClaim>` and `primes: Vec<PrimePlaneClaim>`. The existing
`new_binary` constructor retains its word-only behavior. A group must contain
at least one word or prime claim. Every prime claim contains:

```rust
pub struct PrimePlaneClaim {
    pub polynomial: usize,
    pub point: Vec<u128>,
    pub tensor_component: u8,
    pub value: u128,
}
```

The polynomial index names its owner in the committed descriptor. Coordinates
and value must be canonical residues strictly less than
`P = 340282366920938463463374607431768211181`. Their transcript encoding is
16-byte little endian, with fixed-width counts and owner indices. The point has
`n + 8` coordinates: eight low scalar-coefficient bits, then the low carrier
column bits, then scalar-row bits. Natural scalar coefficients occupy indices
0 through 161; indices 162 through 255 are virtual zero. Signed ring packing is
undone before this reading, so an honest payload reads as its unsigned bits.

The tensor component is a caller-owned decomposition tag from 0 through 3.
It is bound and counts toward plane identity; it does not impose an extra row
mask or change equality weights. One separable point reads all scalar rows in
one plane. This API does not infer arbitrary nonseparable coefficient weights
from a tag and supplies no application reduction or fusion helper.

`PrimePlaneManifest` is the single admission boundary. It permits at most five
unique `(polynomial, point, tensor_component)` planes in the entire binary
group, hence at most five distinct points. Different polynomials never share a
plane. Exact duplicate requests share the first occurrence and do not change
order-zero bytes, D, witness size, or proof bytes. Conflicting values for the
same functional are invalid. A sixth plane, including one distinguished only
by a component tag or owner, returns `AkitaError::InvalidInput`; it cannot
increase a proof-controlled allocation. Input request lists are also limited
to 4096 entries before deduplication.

For each unique plane, split its point into coefficient, column and row
coordinates. Let `omega_s = eq(rho_coeff,s)`, `h_b = eq(rho_col,b)` and
`e_r = eq(rho_row,r)`, all in P. The prover constructs

```text
EP_b(Z) = sum_r e_r src_(r,b)(Z) mod P
sum_r e_r V_r(Z) = rem_Phi(sum_b c_b(Z) EP_b(Z)) mod P
v_P = sum_b h_b sum_(s=0..161) omega_s EP_b[s] mod P.
```

The response V uses **all four balanced response digits**, with the signed
packing undone. Reading its low digit here would authenticate a different
functional. Each plane uses the challenges of its own polynomial. The first
identity is tested after reduced scalar multiplication and ring switching;
the scalar reading is a separate linear equation. Each receives an independent
row-batching coefficient at order 7 and joins the existing fused Stage 2.

At order 0, the canonical manifest and all unique prime claims are public
messages before any challenge. At order 3, D commits the compact natural-digit
concatenation of U, if selected, followed by all EP planes in first-occurrence
order. Within EP the order is plane, column, digit, natural coefficient. There
is no inter-plane degree-64 padding: only the final D chunk is padded. D is
fixed before the order-4 nonce and indexed column challenges. At order 5, those
same EP digits occupy the outgoing witness with 256-coefficient scalar strides.
D authentication addresses this exact compact-to-padded map. Order 9 proves
the equations above in the same Stage 2 as the ordinary root obligations.
No new proof messages or challenges are inserted after the suffix.

A prime-only opening skips order 2 and omits U, K and K_val completely from
both D and the outgoing witness. It still authenticates A through the same seven exact projections; no KA
segment remains for any polynomial.
An opening with no prime claims preserves the projection word-only transcript
atoms, witness addresses and bytes.

### Representation bounds and error accounting

An EP coefficient is first reduced modulo P and centered in
`[-170141183460469231731687303715884105590,
  170141183460469231731687303715884105590]`. Forty-three balanced base-eight
digits have negative reach `388894133623929672529570979922020813092` and positive
reach `291670600217947254397178234941515609819`, so both centered endpoints fit.
Forty-two digits have positive reach only
`36458825027243406799647279367689451227`, which is insufficient. Forty-three
digits do **not** represent every canonical nonnegative residue: `P-1` exceeds
the positive reach. Centering is required for completeness.

EP identities are equations in P, so they introduce no integer no-wrap
inequality. Their reduced polynomial degree is at most 161. In the interactive
algebraic analysis, an erroneous set of prime polynomial equations survives
a uniform common alpha with probability at most `161/P`; independently
batching the surviving equations contributes at most `1/P`. These bounds do
not multiply by the number of planes: fix any nonzero residual polynomial,
then use independent batching weights. The aggregate root already charges the
conservative degree loss 648 and independent row-batching loss 1; the prime
rows fit those existing bounds. The Stage-2 degree remains three. A widened
domain increases its round count and the associated existing sumcheck loss.
At the computed `2^26` root domain, 26 degree-three Stage-2 rounds contribute
at most `78/P` by the interactive sumcheck union bound.
These are algebraic statements, not a whole-protocol Fiat-Shamir or SIS
certificate.

### Authenticated output

```rust
pub fn batched_verify_readings_in_session(
    &self,
    session: &mut VerifierSession<'_>,
    statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
    basis: BasisMode,
    session_id: [u8; 32],
    group_id: [u8; 32],
) -> Result<BinaryOpening<Option<BinaryWordClaim>, Vec<PrimePlaneClaim>>, AkitaError>;
```

This method constructs the existing conditional-agreement result only after
the complete grouped opening and suffix verify. It returns unique authenticated
prime claims in manifest order and the optional word reading. The terminal
prime digest binds the commitment payload, source descriptor and canonical prime claims using a
domain-separated encoding. Session and group identifiers name context already
authenticated by the caller for this exact committed source. The method adds no transcript messages, preserves
the caller's continuation, and does not check agreement. Agreement remains the
application's proof-token contract in `agreement.rs`; Akita adds neither a
checker nor an application helper. The caller checks final proof exhaustion.

### Planner and admission inputs

`akita_planner::prime_planes::price_binary_root_planes` is behind the planner's
default-off `labinius-trinomial` feature. It uses the verifier's canonical
witness plan, including every unique plane, optional word views, and optional
ordinary preimages. It reports compact D digits/columns, physical segment
widths, the next power-of-two domain, ordinary ring-element count, and an
existing-table D rank lookup. It neither selects an extension nor changes
schedule artifacts.

```rust
pub fn price_binary_root_planes(
    manifest: &BinaryCommittedGroupManifest,
    word: bool,
    prime: &PrimePlaneManifest,
    destination_ring_dimension: usize,
    ordinary: Option<GroupCommitPhaseParams>,
) -> Result<BinaryPrimePlanePrice, AkitaError>;
```

At C=256, one plane contributes `43*162*256 = 1,783,296` natural digits and
`43*256*256 = 2,818,048` physical cells. Five planes contribute 8,916,480 natural
digits and 14,090,240 physical cells. The compact degree-64 D widths are:

| Planes | Prime-only D columns | With U D columns |
| --- | ---: | ---: |
| 1 | 27,864 | 28,512 |
| 2 | 55,728 | 56,376 |
| 5 | 139,320 | 139,968 |

For the C=2 fixture, five planes add 69,660 natural digits and 110,080 physical
cells; D has 1,089 prime-only columns or 1,094 with U. The canonical plan
contains no KA segment. The following complete totals were measured through
`price_binary_root_planes` at rank three with one polynomial and no ordinary
preimages; each row gives prime-only and word-plus-prime physical cells.

| Geometry | Planes | Prime-only cells | With word cells | Minimum domain | Ordinary degree-64 elements |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=2^22, C=256 | 1 | 25,886,720 | 25,956,096 | 33,554,432 | 524,288 |
| N=2^22, C=256 | 2 | 28,704,768 | 28,774,144 | 33,554,432 | 524,288 |
| N=2^22, C=256 | 5 | 37,158,912 | 37,228,288 | 67,108,864 | 1,048,576 |
| N=2^23, C=256 | 1 | 42,663,936 | 42,733,312 | 67,108,864 | 1,048,576 |
| N=2^23, C=256 | 2 | 45,481,984 | 45,551,360 | 67,108,864 | 1,048,576 |
| N=2^23, C=256 | 5 | 53,936,128 | 54,005,504 | 67,108,864 | 1,048,576 |

The small N=8, C=2 word-plus-prime fixture has 79,616, 101,632 and
167,680 cells for one, two and five planes respectively; their domains are
131,072, 131,072 and 262,144. Optional ordinary preimages add their canonical
segment widths before rounding the domain. At the two large geometries, Z
occupies 16,777,216 and 33,554,432 cells respectively, T occupies 6,291,456,
and optional U/K/K_val occupy 65,536/2,304/1,536. These measured segment
widths account for each complete total above.

Balanced digit differences have coefficient bound 7. The existing degree-64,
bound-7 q128 scalar SIS row permits 76,306 columns at rank one and 106,021,444
at rank two. The five-plane D exceeds rank-one coverage and fits the existing
rank-two row; `Q128Offset275` uses that row through `row_owner`. This lookup
covers only the declared D width under that table's model. No new estimator or
full admission certificate has been run for the widened ordinary suffix or
the experimental q24 root. No production certification is asserted.

Several polynomial owners also widen the indexed challenge family. With `n_poly`
owners and C columns its extraction-address input is `I = n_poly*C`, not just C.
The current bounded-weight support set has
`|S46| = sum_(i=0..46) binom(162,i) =
107268495235892251993000716928106195079952`, or 136.300278309 bits.
At C=256, the elementary `I/|S46|` term is `2^-128.300278309` for one
owner, `2^-127.300278309` for two, and `2^-125.978350214` for five.
Consequently the inequality `I/|S46| <= 2^-128` already fails for two
owners at that geometry. The experimental multi-polynomial tests do not
constitute production admission or change the sampler. Prime planes on one
owner do not introduce extra column challenges; the plane cap does not grant
each owner an independent security budget.

## Measured proof sizes

The eight-carrier, C=2 test fixture uses the existing ordinary suffix catalog.
Each plane in these measurements has a distinct prime point. Sizes include the
whole grouped opening inside the caller's transcript.

| Reading | Planes | Current protocol bytes | `dev-protocol` bytes |
| --- | ---: | ---: | ---: |
| Prime only | 1 | 75,782 | 70,698 |
| Prime only | 2 | 75,761 | 70,707 |
| Prime only | 5 | 75,760 | 70,682 |
| Word and prime | 1 | 77,938 | 72,840 |
| Word and prime | 2 | 77,937 | 72,831 |
| Word and prime | 5 | 77,897 | 72,910 |
| Word and prime, mixed ordinary | 1 | 92,129 | 87,019 |
| Word and prime, mixed ordinary | 2 | 92,120 | 87,101 |
| Word and prime, mixed ordinary | 5 | 92,110 | 87,072 |

With no prime claims, the base and updated word-only proofs are byte-identical:

| Fixture | Current protocol bytes | `dev-protocol` bytes |
| --- | ---: | ---: |
| Binary word, width 128 | 77,936 | 72,877 |
| Binary word, width 8 | 77,937 | 72,898 |
| Binary word and ordinary group | 92,101 | 87,119 |

The compatibility fixture pins full proof digests captured from base commit
`b3c084284`, re-executed in both protocol modes before the merge measurement.
Duplicate requests also produce identical proof bytes. The new two-polynomial,
one-plane prime-only mutation fixture measures 75,869 bytes in the current
protocol and 70,855 bytes in the dev protocol.
