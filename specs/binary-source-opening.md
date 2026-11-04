# Spec: binary source grouped opening

| Field | Value |
| --- | --- |
| Author(s) | Akita contributors |
| Created | 2026-10-28 |
| Status | implemented experimental path; security admission pending |
| PR | Base root: [#290](https://github.com/LayerZero-Labs/akita/pull/290); field switch, exact-divisibility projections, prime planes and typed Stage 3 |
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
group second. A binary group may contain several polynomials. It requires an ordinary successor fold that consumes the typed A setup-prefix
claim as an ordinary committed group. It does not add a production binary
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
`AkitaCommitmentScheme::batched_prove_embedded` and
`AkitaVerifier::batched_verify_embedded`. There is one proof type, one
caller transcript and one grinding-plan cursor. No standalone binary PCS,
nested session seed or separate root transcript is invoked. The caller can
continue using its live transcript after verification and checks exhaustion
at its chosen proof boundary with the returned `PendingEof`.

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
| 7 | Draw the shared ring-switch challenge, range-image point and independent relation-batching weights, whose first seven entries are the output scalars mu; form rho = sum_l mu_l J_l and compile the non-A public weights. | All tested tables, commitments and projection values. |
| 8 | Run the ordinary Stage 1 range-image sumcheck with the base-eight digit alphabet. | The range-image terminal precedes Stage 2 batching. |
| 9 | Run one fused Stage 2 for sum_l mu_l (<J_l, RA> - y_l) = 0, B/D authentication, selected parity/value and prime response/reading equations, and range-image binding; include ordinary relations when selected. No row reads a padding address, and the verifier's B/D weights move to order 10. | One final opening of the same outgoing witness. |
| 10 | Bind the typed prefix identity; send one setup product claim for the A, B and D rows of the whole group; run one degree-two Stage 3 sumcheck; send one prefix evaluation. | Each round message precedes its challenge; the prefix claim joins the next ordinary grouped fold. |
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
remaining relation coordinates authenticate parity, value, B/D, any ordinary
rows and their `{-1, 0}` bits, and the prime planes, in that order. The vector
has `7 + 2·[word] + 2·b_rows + O + 2·planes` entries, where `O` counts the
mixed shape's ordinary linear rows and negative-bit addresses
([`BinaryRootPublic::batch_count`](../crates/akita-types/src/binary_source/root_public.rs)).
No quotient table, quotient bound or quotient relation participates in this
path.

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
every native coefficient its ordinary witness address. This coefficient
repacking is not a ring homomorphism.

The root compiles no row on a *padding address*: a stride tail of a segment
lane or an address at or beyond the last segment. Every root row, the typed A
product and every source reading reads natural addresses only, so the decoded
root equations and readings are the same for any two tables that agree on the
natural cells. The honest prover writes zero padding. A prover that commits
another value there must still pass the Stage 1 range check, the successor's
authentication of the table it committed and the physical-norm checks; these
read every committed cell and are unchanged. `BinaryRootPublic::new` checks
only that the natural end lies within the logical witness length. The
argument, the support lemma and the rejected alternatives are in
[binary-source-zero-padding.md](binary-source-zero-padding.md).

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

## Typed A setup product

The setup preparer exposes these opt-in methods (the generic CPU field bounds
are the same as ordinary prefix preparation):

```rust
pub fn prepare_binary_a_prefixes(
    &self,
    manifest: &BinaryCommittedGroupManifest,
    typed: &BinaryTypedPrefix,
    first: &SetupPrefixSlotId,
    remaining: &[SetupPrefixSlotId],
) -> Result<SetupPrefixProverRegistry<F, CommitmentHandle<F, E>>, AkitaError>;

pub fn prepare_binary_a_verifier_setup(
    &self,
    ordinary: &AkitaVerifierSetup<F>,
    manifest: &BinaryCommittedGroupManifest,
    typed: &BinaryTypedPrefix,
    first: &SetupPrefixSlotId,
    supplied: &SetupPrefixVerifierSlot<F>,
) -> Result<AkitaVerifierSetup<F>, AkitaError>;

impl<E: Field> BinaryAProduct<E> {
    pub fn new(
        plan: &BinaryRootWitnessPlan,
        rho: &[[E; 648]],
        stage2_point: &[E],
        prefix_len: usize,
    ) -> Result<Self, AkitaError>;
    pub fn factors(&self) -> Result<(Vec<E>, Vec<E>), AkitaError>;
    pub fn evaluate(&self, point: &[E]) -> Result<E, AkitaError>;
}
```

The typed prefix adds the B and D authentication setup to the same table.
[Typed prefix layout](#typed-prefix-layout) defines its layout and identity;
[binary-source-setup-offload.md](binary-source-setup-offload.md) argues that
the offload is sound:

```rust
impl BinaryTypedPrefix {
    pub fn new(
        manifest: &BinaryCommittedGroupManifest, word: bool, planes: usize,
        destination_ring_dimension: usize,
    ) -> Result<Self, AkitaError>;
    pub fn validate_prefix(&self, prefix_len: usize) -> Result<(), AkitaError>;
    pub fn identity(
        &self, manifest: &BinaryCommittedGroupManifest, slot: &SetupPrefixSlotId,
    ) -> Result<Vec<u8>, AkitaError>;
    pub fn setup_coefficients<F: Field>(
        &self, seed: &[u8; 32], manifest: &BinaryCommittedGroupManifest, prefix_len: usize,
    ) -> Result<Vec<F>, AkitaError>;
    pub fn prover_weights<E: Field>(
        &self, a: &BinaryAProduct<E>, root_point: &[E], alpha: E,
        batches: [&[E]; 2], prefix_len: usize,
    ) -> Result<Vec<E>, AkitaError>;
    pub fn digit_weight<E: Field>(
        &self, point: &[E], root_point: &[E], alpha: E, batches: [&[E]; 2],
    ) -> Result<E, AkitaError>;
}
```

The coefficient compiler and stream contracts are:

```rust
pub fn binary_a_setup_coefficients<F: Field>(
    matrix: &[Vec<[i128; 648]>], manifest: &BinaryCommittedGroupManifest, prefix_len: usize,
) -> Result<Vec<F>, AkitaError>;
pub fn binary_direct_a_weights<E: Field>(
    plan: &BinaryRootWitnessPlan, matrix: Option<&[Vec<[i128; 648]>]>,
    challenges: &Challenges, rho: &[[E; 648]],
) -> Result<NativeRootWeights<E>, AkitaError>;
```

The existing relation methods now expose the same rho boundary:

```rust
impl BinaryRootWitnessPlan {
    pub fn relation_weights<E: Field>(
        &self, claim: Option<&BinaryTerminalClaim>, matrix: Option<&[Vec<[i128; 648]>]>,
        challenges: &Challenges, alpha: E, rho: &[[E; 648]],
        terminal_batches: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError>;
}
impl BinaryRootPublic {
    pub fn relation<E: Field>(
        &self, challenges: &Challenges, alpha: E, batches: &[E],
        rho: &[[E; 648]], projected_values: &[i128; 7],
        commitment: &[E], d_payload: &[E], ordinary_claim: Option<(&[E], E, &[E])>,
        matrix: Option<&[Vec<[i128; 648]>]>,
    ) -> Result<(NativeRootWeights<E>, Vec<(usize, E)>), AkitaError>;
}
```

Proving passes `Some(matrix)`; verification passes `None`. The canonical direct A compiler retains the image weights in both cases.

The behavior-preserving ordinary seams provide the following signatures:

```rust
pub struct FactoredStage3Request<'a, F: Field, E: Field, ProofSession> {
    pub session: &'a ProofSession,
    pub level: u32,
    pub coefficients: Arc<[F]>,
    pub index_factor: Vec<E>,
    pub coefficient_factor: Vec<E>,
}
// OpaqueStage3Kernel method; unsupported backends return UnsupportedSchedule.
fn begin_factored_stage3(
    &self, request: FactoredStage3Request<'_, F, E, Self::ProofSessionHandle>,
) -> Result<(E, Self::Stage3SessionHandle), AkitaError>;
// CpuBackend method.
pub fn prepare_setup_prefix_from_coefficients(
    &self, id: &SetupPrefixSlotId, coefficients: Vec<F>,
) -> Result<PreparedSetupPrefix<F, CommitmentHandle<F, E>>, AkitaError>;
// AkitaVerifierSetup methods.
pub fn with_prefix_context(
    self, id: SetupPrefixSlotId, context: Vec<u8>,
) -> Result<Self, AkitaError>;
pub fn prefix_context(&self, id: &SetupPrefixSlotId) -> Option<&[u8]>;
// Explicit planner selection; ordinary search remains unchanged.
pub fn find_schedule_with_root_setup_mode(
    key: &ScheduleLookupKey,
    final_source_contract: CommittedSourceContract,
    precommitted_source_contracts: &[CommittedSourceContract],
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<FoldChallengeDistribution, AkitaError>,
    root_setup_mode: SetupContributionMode,
    root_main_constraint: Option<&CommittedGroupParams>,
) -> Result<PlannedFoldSchedule, AkitaError>;
```

Prefix contexts require an existing slot and 1..=262,144 bytes. This is a
resource cap, not a soundness inequality. With `test-support`,
`BinaryAReadGuard::forbid() -> BinaryAReadGuard` makes the canonical A accessor
return an error until the guard is dropped; it adds no proof message.

The scoped prover registry is passed to the canonical grouped backend route;
the prepared verifier setup is passed to the existing verifier constructor.

The q-valued A stream is separate from every ordinary P-valued setup stream.
The existing deterministic sampler rejects u32 values at or above
`4,290,276,109 = 301 q`, then reduces modulo q and centers in
`[-7,126,704, 7,126,704]`. Exactly 4,691,187 of the 2^32 candidates are rejected.
The centered coefficients embed in P without changing the source distribution.
The A tensor orders coefficients by source row j, padded rank row i, and
padded coefficient s. The coefficient stride is 1,024 and rank three pads to
four, so the tensor holds `1,024 * next_power_of_two(rank) * m` cells. The B/D
pieces fill aligned subcubes of the tensor's zero padding first and then
follow the tensor; all remaining prefix capacity is zero
([Typed prefix layout](#typed-prefix-layout)). The domain must be a power of
two at least `BinaryTypedPrefix::domain_len`.

Setup preparation commits this table with the ordinary setup-prefix executor,
checks the supplied commitment against the deterministic typed table once, and
retains an installation identity containing the stream/layout label, complete
binary manifest, the typed layout and the scheduled slot. Online verification
requires that identity. A different typed table or the ordinary stream's table
fails preparation;
a different manifest cannot reuse the prepared verifier view. Scoped contexts
are transient: serialization rejects before writing bytes, so a persisted
ordinary setup must have its typed prefix prepared again after loading. The
recursion glue blob has no field for a context, and its encoder rejects a
scoped setup the same way. That glue does not yet carry binary statements.
The first typed slot cannot be reused as an outgoing ordinary setup slot later
in the same schedule. Ordinary prefix caches remain unchanged.

Write `Psi = Y^648 - Y^324 + 1`. Projection rows cover the joined residual
in polynomial-major order, so
`rho[p][i][t] = sum_l mu_l J_l[p][i][t]`. The same A matrix is used by every
polynomial. Let R be the power-of-two source-row count per polynomial. The
response segment starts at address zero and orders cells by polynomial,
source row, four response digits and a 1,024-coefficient stride. Split its
Stage 2 point into coefficient coordinates `r_t`, two digit coordinates `r_d`,
`log2(R)` row coordinates `r_j`, and the remaining owner/padding coordinates
`r_p`. Define

```text
d = sum_(a=0..3) 8^a eq(r_d,a)
bar_rho[i][t] = sum_p eq(r_p,p) rho[p][i][t]
g_i(k) = <bar_rho[i], rem_Psi(Y^k)>
h_i(s) = d * sum_(t=0..647) eq(r_t,t) g_i(s+t)
W_A(r) = sum_(j,i,s) A[i][j][s] eq(r_j,j) h_i(s).
```

The owner contraction uses the entire remaining Stage 2 point, including
coordinates selecting the response segment against later witness segments.
Only live owners, rows, rank entries and coefficients contribute; all typed
prefix padding is zero. This yields one rectangular public weight on the
shared A prefix, regardless of polynomial count, word-claim presence or prime
plane count. There is one product claim, one degree-two sumcheck and one
A-prefix evaluation for the entire group, also when an ordinary group joins.
No quotient segment or `-q rho` term remains.

The signed reduction kernel is Hankel. The existing trinomial adjoint computes
each h_i with two exact products of length `2D-1 = 1,295`, without roots of
unity. Multiplying a full length-1,295 kernel by a length-648 reversed input
would have length `3D-2 = 1,942`; it is not a single length-1,295 product.
The implementation splits the products and retains their required windows.
Its existing arithmetic count is 154,872 field multiplications per adjoint,
versus 419,904 for the dense 648-by-648 product (464,616 versus 1,259,712 for
rank three). These counts do not include all surrounding field operations.
The verifier's inputs are equality tables `v[t] = eq(r_t, t)`, for which
`TrinomialAdjoint::apply_equality` computes the same `h_i` without
convolutions. The set bits of D = 648 split `0..D` into aligned blocks of
512, 128 and 8; on each block the correlation with `g_i` folds one equality
coordinate per level, `X_(l+1)(t) = X_l(t) + r[l] (X_l(t + 2^l) - X_l(t))`.
That costs 18,505 multiplications at D = 648.

Order 10 uses the same Stage 3 session, sumcheck transport and verification
as ordinary setup offloading:

1. Absorb the canonical typed prefix identity as a public message.
2. Send the claimed setup product in P.
3. For each of `k = log2(prefix_len)` rounds, send the two ordinary quadratic
   sumcheck coefficients, apply the existing scheduled grinding step, and draw
   the round challenge in P.
4. Send the prefix evaluation at the resulting point. Check the terminal
   product against the public weight: the factored A weight plus
   `BinaryTypedPrefix::digit_weight`.
5. Combine the setup product with the Stage 2 linear weight without the A and
   B/D setup weights, and the same witness evaluation.

The resulting prefix point/value enters the next ordinary grouped recursion,
using the authenticated prepared commitment. The ordinary suffix consumes it;
the terminal accepts no unconsumed prefix claim. There is no second proof or
nested transcript. Order 10 contributes `32k + 32` field-message bytes, plus
any grinding nonces prescribed by the selected schedule. Its interactive
algebraic error is at most `2k/P`. The fixture uses k = 18, giving
`36/(2^128-275)`, approximately `1.058e-37` (2^-122.83). This is added to the
existing root and suffix algebraic terms and the commitment-binding and hash
errors, including authentication of the typed prefix commitment. No numerical
binding-security claim is made for the experimental fixture. No new integer no-wrap inequality is needed. This
experimental path still makes no production SIS or Fiat-Shamir admission claim.

Online verification constructs rank-many adjoints, evaluates the padded
coefficient factor, and evaluates the row equality product in logarithmic work
without expanding the source-row domain. It never materializes or reads A.
It evaluates the B/D weight from the segment geometry and never reads a B/D
setup entry.
Independent long-division differential tests include two-polynomial groups and
odd row cutoffs, even though the public source descriptor currently
requires power-of-two padded source geometry. A test-only accessor guard makes
any root A materialization fail during complete successful verification.

### Typed prefix layout

This subsection defines the typed prefix table `S` beyond the A tensor, its
Stage 3 weight and its identity. `BinaryTypedPrefix::new` computes it from the
manifest, whether the word is opened, the prime plane count and the
successor's ring dimension, through the root plan of those inputs. All
arithmetic is exact. `K` is the residue kernel at `alpha` on length-64
vectors, `r` is the root's Stage 2 point and `beta_B`, `beta_D` are the B and
D authentication row batches.

**Digit segments.** Role B has one segment, root segment 1 (the image). Role D
has the segments of `plan.opening_segments()` in plan order: U when the word is
opened, then, when there are prime planes, one EP segment that holds every
plane, with `elements = planes * columns`. `SegmentLanes` gives each
segment's `start`, `stride`, degree `n`, `digits` and `elements`. The segment
has `L = elements * digits` lanes. Lane `l` holds `n` digits, at root
addresses `start + l * stride + c` for `c < n`. These digits are the role's
compact digits `offset + l * n + c`, where `offset` is the sum of `L * n` over
the role's earlier segments. Role column `c` is the role's compact digits
`[64c, 64c + 64)`. A role with `T` compact digits has `ceil(T / 64)` columns.
Construction fails with `InvalidSetup` if a segment has `L = 0` or `n = 0`,
if its stride is not a power of two, or if its start is not a multiple of its
stride.

**Period copies.** A segment's period is `P = 64 / gcd(n, 64)` lanes, so its
`P * n` digits fill whole columns. Period `q` covers lanes
`[qP, min(qP + P, L))`, and the segment has `ceil(L / P)` periods. Each period
stores its own copy of

```text
J = ceil((offset mod 64 + min(P, L) * n) / 64)
```

columns. Copy column `j < J` of period `q` is role column
`floor(offset / 64) + q * P * n / 64 + j`. Write `u = 64j + t - (offset mod 64)`
for slot `t < 64` of that copy column. The slot carries a digit when
`0 <= u < P * n` and lane `l = qP + floor(u / n)` is below `L`. That digit is
at root address `start + l * stride + (u mod n)`. Every other slot carries no
digit.

There are `rows = manifest.b_rows()` authentication rows. For row `row`, the
64 prefix cells of a copy column hold the coefficients of
`setup_entry(seed, label, row, column)`. The label is `manifest.b_label()` for
B, and the same label with role `WitnessCommitment` for D. A copy column whose
role column is at or beyond the role's column count holds zeros. A column
that two periods or two segments share is stored once in each copy.

The Stage 3 weight on the same 64 cells is `beta_role[row] * K(e)`. Here
`e[t] = eq(r, a)` when slot `t` carries the digit at root address `a`, and
`e[t] = 0` otherwise.

**Pieces.** `binary_runs(x)` lists the set bits of `x` from highest to
lowest. The run of bit `i` is `(start, i)`, where `start` is the sum of the
higher set bits. Pieces are enumerated in this order: segment (B, then D in
plan order), row, a run `(q0, b)` of `binary_runs(ceil(L / P))`, and a run
`(first, e)` of `binary_runs(J)`. Each piece holds periods `[q0, q0 + 2^b)` by
copy columns `[first, first + 2^e)` of one row, which is `2^(b + e + 6)`
cells. Cell `(q, j, t)` of a piece has local index

```text
base + ((q - q0) * 2^e + (j - first)) * 64 + t.
```

The pieces are then sorted stably by decreasing size.

**Regions.** A region maps a local index to a prefix address. A hole region
has fixed bits with values, free bits in increasing order and a bound `top`.
The address of local index `x` has every fixed bit at its value, bit `i` of
`x` at the `i`-th free bit, and every bit at or above `top` equal to zero.
Holes exist only when the A natural length `m * row_stride` is a power of
two, where `row_stride = 1024 * R` and `R = next_power_of_two(rank)`. Then
`top = log2(m * row_stride)`. An A tensor cell is
`j * row_stride + i * 1024 + s`. The holes come from two families, in this
order:

1. padded ranks `[rank, R)` with coefficients `[0, 1024)`;
2. live ranks `[0, rank)` with coefficients `[648, 1024)`.

In each family, the rank range and the coefficient range are each cut into
aligned runs. Starting at the range's start `x`, take the largest `w` such
that `2^w` divides `x` and `[x, x + 2^w)` ends within the range. Record the
run `(x, w)`, then continue from `x + 2^w`. Rank runs are the outer loop and
coefficient runs the inner loop. Each pair of a rank run `(x_r, w_r)` and a
coefficient run `(x_c, w_c)` with `w_c >= 6` forms one hole:

- free bits `0 .. w_c`, then `10 .. 10 + w_r`, then `10 + log2(R) .. top`;
- fixed bits `w_c .. 10`, set from `x_c`, and bits `10 + i` for
  `w_r <= i < log2(R)`, set from bit `i` of `x_r`.

The holes are sorted stably by decreasing free-bit count. The last region is
the tail. It has no fixed bit and `top = 64`, and it maps every local index to
itself.

**Placement.** Pieces are placed in their sorted order. A piece goes to the
first hole, in region order, whose unused capacity
`2^(free-bit count) - cursor` is at least the piece's size. Its base is that
hole's cursor, and the cursor advances by the size. Otherwise the piece goes
to the tail. The tail cursor starts at the A natural length rounded up to a
multiple of the first tail piece's size. The piece's base is the tail cursor,
which then advances by the size. Sizes are non-increasing powers of two, so
every base is a multiple of its piece's size. `natural_len` is the final tail
cursor, or the A natural length when no piece reaches the tail. `domain_len`
is the smallest power of two at least `natural_len`. Every cell outside the A
tensor and the pieces is zero, in both `S` and the Stage 3 weight.

**Identity.** `BinaryTypedPrefix::identity` is the ASCII string
`akita/binary-source/typed-prefix/a-then-digit-pieces/v3`, then
`manifest.canonical_bytes()`, then a sequence of little-endian `u64` words,
then the slot's compressed serialization. The words have this width on every
target. They are, in order:

1. the A row count `m`, the A row stride and `rows`;
2. the segment count, then per segment `role` (0 for B, 1 for D), `start`,
   `stride`, `n`, `digits`, `elements` and `offset`;
3. the region count, then per region in region order `top`, the fixed-bit
   count and each fixed bit's index and value (0 or 1). A hole follows with
   its free-bit count and free bits. The tail follows with the word
   `2^64 - 1`;
4. the piece count, then per piece in `(region, base)` order its segment
   index, `row`, `q0`, `b`, `first`, `e`, region index and `base`;
5. `natural_len`.

The test `typed_prefix_identity_bytes_are_pinned` pins these bytes for one
geometry.

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
Complete proof sizes for both protocol modes are recorded in the measured
proof-size tables below. The selected typed-root suffix differs from the
previous direct-root schedule.

The projection messages themselves add exactly 84 bytes. Removing the
committed quotient shrinks the natural outgoing witness by `rank*n_poly*648*8`
digits in the small fixture, or 15,552 digits per polynomial at rank three. Complete totals also depend on the
selected suffix geometry and later variable-length encodings after the changed
transcript challenges. The field-switch contribution remains exactly 2,195
bytes for every three-round fixture.

The fixture exercises the caller's prefix and
post-verification squeeze, commitment serialization, kind consistency,
statement mutations, malformed proof bytes, truncation and trailing bytes.
A test-only backend alters T, Z, U, K or K_val before outgoing commitment. It
uses an incorrect public oracle to complete the adversarial proof; verification
must reject using the independently compiled correct oracle. The same backend
also raises one in-range padding digit. That change violates no root row, so
the proof must verify against the unchanged readings. Each verification runs
behind a panic-catching boundary.

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

Typed Stage 3 acceptance coverage is explicit:

| Mutation or property | Test |
| --- | --- |
| Every quadratic round coefficient, setup product claim, and prefix evaluation | `stage3_binary_group_opens_once_inside_the_callers_transcript`; `stage3_binary_and_ordinary_groups_share_one_opening` |
| A different typed prefix table; the ordinary stream's prefix commitment | `stage3_binary_group_opens_once_inside_the_callers_transcript`, through `reject_foreign_prefixes` |
| Same-geometry manifest swap; missing typed installation identity | `stage3_prefix_context_rejects_manifest_swap_and_missing_context_before_proof` |
| No online A read, including an accessor negative control | Both successful `stage3_binary_*` openings |
| Rank-one/rank-three differential, one/two polynomials, rows 1/2/3/5, independent signed reduction | `stage3_a_product_matches_independent_signed_remainder_and_response_digits` |
| Zero equality coordinates, odd live-row cutoffs, larger padded prefixes | `stage3_a_product_terminal_matches_dense_factors_with_zero_coordinates_and_odd_cutoff` |
| Malformed rho, point, and prefix geometry reject without panic | `stage3_a_product_rejects_malformed_rho_point_and_prefix_without_panic` |

The full opening mutation helper catches verifier panics as test failures;
all proof-field mutations require rejection. Setup-source substitution tests
exercise the trusted installation boundary before online verification.

The ordinary regression fixture is
[ordinary_grouped_root_bytes.rs](../crates/akita-pcs/tests/ordinary_grouped_root_bytes.rs),
which pins a 65,567-byte ordinary proof. Its feature-disabled and
feature-enabled executions are the compatibility gate when the extension is
unselected. Existing pinned fixtures and goldens remain part of that gate.


Both paired local ordering mutations were exercised before committing: moving
mu ahead of the y messages in both drivers, and moving J ahead of the outgoing
commitment in both drivers. Each made
`projection_orchestrators_fix_witness_then_rows_then_outputs_then_batches`
fail with exit code 101. Both mutations were reverted; the restored end-to-end
suite passes in both protocol modes.

The field switch produces the optional `BinaryTerminalClaim` for the existing
root. The canonical A-term compiler
[`binary_direct_a_weights`](../crates/akita-types/src/binary_source/root_projection_weights.rs)
takes explicit projection weights over all polynomial/rank rows. Typed Stage 3
authenticates its setup contribution without an online A read. Optional EP
planes authenticate prime claims as described below; no agreement proof is
discharged by a binary terminal alone.

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
pub fn batched_verify_readings_embedded(
    &self,
    state: &mut VerifierChannel<'_>,
    statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
    basis: BasisMode,
    setup_policy: SetupPolicy<'_, '_, Cfg::ExtField>,
    session_id: [u8; 32],
    group_id: [u8; 32],
) -> Result<
    (PendingEof, BinaryOpening<Option<BinaryWordClaim>, Vec<PrimePlaneClaim>>),
    AkitaError,
>;
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

The eight-carrier, C=2 fixtures retain their existing source and suffix geometry.
The following complete grouped proofs were measured from the 29 end-to-end
tests after the merge of protocol v8 (Akita on `jolt-transcript` inside the
caller's transcript), the deletion of the root's padding rows and the B/D
setup offload. The dev protocol currently equals the current protocol, so
both modes produce these bytes. Each plane in these measurements has a
distinct prime point. Every binary group uses one Stage 3 inside the caller's
transcript, which also covers the B and D authentication rows.

| Reading | Planes | Proof bytes |
| --- | ---: | ---: |
| Prime only | 1 | 78,804 |
| Prime only | 2 | 78,771 |
| Prime only | 5 | 78,782 |
| Word and prime | 1 | 81,011 |
| Word and prime | 2 | 80,973 |
| Word and prime | 5 | 81,025 |
| Prime only, mixed ordinary | 1 | 78,881 |
| Prime only, mixed ordinary | 2 | 78,888 |
| Prime only, mixed ordinary | 5 | 78,948 |
| Word and prime, mixed ordinary | 1 | 81,149 |
| Word and prime, mixed ordinary | 2 | 81,081 |
| Word and prime, mixed ordinary | 5 | 81,068 |

The word-only fixtures use the same typed Stage 3 root without prime planes:

| Fixture | Proof bytes |
| --- | ---: |
| Binary word, width 128 | 80,963 |
| Binary word, width 8 | 80,945 |
| Binary word and ordinary group | 81,132 |

The separate ordinary regression is 65,567 bytes, with the extension feature
both disabled and enabled.
The compatibility fixture pins these three binary word-only proof lengths and
their full digests. Before the offload, after the v8 merge, they were 81,002,
81,030 and 81,092 bytes. The fixture also pins the 2,518 typed-prefix identity
bytes and their digest. The identity bytes did not change in the v8 merge; the
digest did, because v8 hashes instance descriptors with SHA3-256 rather than
Blake2b-256. Duplicate prime requests also produce identical proof bytes.

Several-polynomial measurements use two source polynomials, still sharing one
projection set, one typed prefix, and one Stage 3:

| Two-polynomial reading | Planes | Proof bytes |
| --- | ---: | ---: |
| Word only | 0 | 80,232 |
| Prime only, projection mutation fixture | 1 | 78,032 |
| Prime only | 2 | 78,025 |
| Word and prime | 2 | 80,228 |

## Full-size measurements

The full-size measurement target is experimental and makes no security claim.
The benchmark geometries are N=2^22 and N=2^23 carriers, C=256, rank three,
and the proof field `2^128-275`. The smoke geometry uses N=2^10 and C=2 to
exercise the same opening without the fixed C=256 image-table cost. The host
inspected for this work is an Apple M4 Max with 16 physical cores and
68,719,476,736 bytes of RAM (64 GiB).
For the mixed smoke shape, the ordinary group retains 2^18 coefficients,
matching the existing mixed opening fixture;
the two benchmark sizes use N ordinary coefficients.
Binary carriers use a fixed integer mixer; the ordinary fixture alternates
zero and one.

The driver selects the default-off `binary-source-measurements` feature, which
enables `labinius-trinomial` and the offline plane planner. It accepts:

```sh
scripts/binary-source-full-size.sh --log-carriers 10 --shape word-prime-ordinary
scripts/binary-source-full-size.sh --log-carriers 22 --shape word-prime
scripts/binary-source-full-size.sh --log-carriers 23 --shape word-prime --dev-protocol
scripts/binary-source-full-size.sh --log-carriers 22 --shape prime5 --plan-only
```

The other shapes are `word` and `word-prime-ordinary`; `prime5` has five prime
planes and no word claim. `--plan-only` performs layout reconciliation without
committing or proving and must not be reported as an opening measurement.
The driver prints the commit and `uptime` load averages. Proving and verification
run in separate processes, so their process peak RSS values are independent;
each includes that process's cold preparation. Trace spans report inclusive
wall times, which must not be summed when spans overlap. The verifier must
pass with the online A accessor forbidden, match the caller's transcript
continuation, and consume the complete proof.

The log-10 smoke run uses ordinary catalog admission. It plans its schedules
with `ValidatedScheduleCatalog::try_new`, writes them as an ordinary catalog
artifact, and loads that artifact for verification with ordinary prover setup.
Explicit native-root admission (`ValidatedScheduleCatalog::try_new_native_root`)
and explicit setup-prefix preparation run only at N=2^22 and N=2^23. The
prover and verifier kernels are shared, so the smoke run checks the opening
but not those two full-size harness paths. Its proof is 82,308 bytes and is
pinned. After the v8 merge and before the B/D setup offload it was 82,298
bytes. Before the v8 merge, with the offload, it was 87,366 bytes in the
current protocol and 82,285 bytes in the dev protocol.

### Proof digest pins

[`scripts/binary-source-full-size-digests.tsv`](../scripts/binary-source-full-size-digests.tsv)
pins the proof size and Blake2b-256 digest of every full-size and smoke shape.
The driver fails when a proof differs from its row. The dev protocol now equals
the current protocol, so the file has one row per `(log_carriers, shape)` and
`--dev-protocol` checks the same row. The `--dev-protocol` smoke reproduces the
82,308-byte pin.

The current pins were re-captured from driver runs at `237eb5a13`. That commit
merges the v8 port of the prover-speed pass into the B/D setup offload. Before
re-pinning, each new proof was compared with the v8 pin it replaces, which has
no offload. The offload had to change it the way it changed the pre-v8
dev-protocol proofs. For every full-size shape, the per-stage difference
between the two v8 proofs equals the per-stage difference that the offload
made before v8. The comparison excludes one-atom nonces, whose varint width
depends on their value, and the Rice-coded terminal payload. Compared directly
with the offload's pre-v8 dev-protocol proof, each new proof has the same stage
grammar plus v8's one fold-challenge nonce per fold level. The field switch,
projections, Stage 1–3 and physical L2 categories equal the pre-v8 offload
partition below in bytes and atoms. The lengths and their changes:

| N | Shape | Pre-v8 dev before → with offload | Change | v8 before → with offload | Change |
| --- | --- | --- | ---: | --- | ---: |
| 2^10 | word-prime-ordinary | 82,308 → 82,285 | −23 | 82,298 → 82,308 | +10 |
| 2^22 | word | 82,739 → 82,786 | +47 | 82,773 → 82,764 | −9 |
| 2^22 | word-prime | 82,788 → 84,423 | +1,635 | 82,743 → 84,392 | +1,649 |
| 2^22 | word-prime-ordinary | 82,747 → 84,394 | +1,647 | 82,762 → 84,406 | +1,644 |
| 2^22 | prime5 | 79,860 → 89,212 | +9,352 | 79,899 → 89,195 | +9,296 |
| 2^23 | word | 84,544 → 84,580 | +36 | 84,583 → 84,590 | +7 |
| 2^23 | word-prime | 84,554 → 84,576 | +22 | 84,594 → 84,590 | −4 |
| 2^23 | word-prime-ordinary | 84,598 → 84,535 | −63 | 84,569 → 84,553 | −16 |
| 2^23 | prime5 | 81,534 → 89,248 | +7,714 | 81,562 → 89,212 | +7,650 |

Where the prefix domain grows, both changes are the added Stage 3 rounds and
the larger successor schedule. Elsewhere both are a few tens of bytes of
value-dependent payload. The smoke run's offload adds no stage, so its
comparison is direct: the new proof differs from the pre-offload v8 proof
only in value-dependent widths.

The v8 pins before the offload were re-captured from driver runs after merging
`dev` at protocol v8.
That merge ports the transcript to protocol v8 and adds one fold-challenge
work nonce to each fold level after the binary root. Grinding nonces and the
Rice-coded terminal payload depend on challenge values, so every digest and
most lengths change. The pre-v8 dev-protocol proofs already omitted the last
live e and t blocks. They are the comparable baseline. The pre-v8 columns are
the pins after the padding-row deletion and before the offload:

| N | Shape | Pre-v8 current bytes | Pre-v8 dev bytes | v8 bytes before the offload |
| --- | --- | ---: | ---: | ---: |
| 2^10 | word-prime-ordinary | 87,337 | 82,308 | 82,298 |
| 2^22 | word | 87,804 | 82,739 | 82,773 |
| 2^22 | word-prime | 87,805 | 82,788 | 82,743 |
| 2^22 | word-prime-ordinary | 87,786 | 82,747 | 82,762 |
| 2^22 | prime5 | 84,879 | 79,860 | 79,899 |
| 2^23 | word | 89,641 | 84,544 | 84,583 |
| 2^23 | word-prime | 89,603 | 84,554 | 84,594 |
| 2^23 | word-prime-ordinary | 89,638 | 84,598 | 84,569 |
| 2^23 | prime5 | 86,595 | 81,534 | 81,562 |

At every full-size row, the field switch, projections, Stage 1–3 and physical
L2 categories have the same bytes and atoms as the pre-v8 dev-protocol
partition. Only the remaining payload, framing and work category changes.

The v8 pins also differ from those of the structured-weight branch below this
one (for example 82,743 rather than 82,752 bytes at N=2^22 `word-prime`). The
padding-row deletion changes the root's batching challenges, so every later
challenge, nonce and the terminal payload change. The message grammar does
not: at the smoke size both branches emit the same 955 message ranges in the
same order. Six ring-switch and fold-challenge nonces differ by one byte, and
the terminal payload is 21,188 rather than 21,176 bytes. At full size the six
named categories are equal on both branches.

Before the v8 merge, all eighteen pre-v8 rows were regenerated after the
padding-row deletion, which changed every transcript-dependent digest. The
chronology of the pins superseded by that deletion, and what they
established, is in the appendix.

### Results after the v8 merge

These rows measure `237eb5a13`: the B/D setup offload, the prover-speed pass
and protocol v8 together. One executable (SHA-256
`71f17c7eb0c73c19298578523ee2d7f9afd88e1bc897f67b16cb780c077d468a`)
served two campaigns, both with the catalog cache. Each shape proved once and
`word-prime` three times, and every proof verified three times in isolated
processes. All 25 proofs reproduce their pins. All 49 verifications report
`verification passed=true`, match the caller's transcript continuation and
keep online A reads forbidden.

The first campaign supplied the pins. It ran at one-minute loads of 28 to 58
on the 16-core host, and its wall times are not reported. The second, below,
ran at loads of 9.76 to 15.14. Each cell is the median of three, with the
peak RSS of that median run. The logs are
`$CARGO_TARGET_DIR/318-logs/{m,q}-<log>-<shape>.{prove<i>,verify<i>}.txt`.
The last column is the offload's pre-v8 current-protocol median, measured at
loads of 5.69 to 7.49.

| N | Shape | Proof bytes | Atoms | Verification (ms) | Verifier peak RSS bytes | One-minute load | Pre-v8 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 2^22 | word | 82,764 | 25,591 | 14.9 | 76,709,888 | 9.91 | 14.9 |
| 2^22 | word-prime | 84,392 | 25,701 | 16.5 | 156,352,512 | 10.25 | 16.1 |
| 2^22 | word-prime-ordinary | 84,406 | 25,717 | 178.0 | 644,055,040 | 9.76 | 124.8 |
| 2^22 | prime5 | 89,195 | 26,128 | 15.4 | 129,482,752 | 9.86 | 15.2 |
| 2^23 | word | 84,590 | 25,755 | 16.7 | 156,811,264 | 12.28 | 16.2 |
| 2^23 | word-prime | 84,590 | 25,756 | 18.3 | 159,465,472 | 11.17 | 17.5 |
| 2^23 | word-prime-ordinary | 84,553 | 25,718 | 351.5 | 1,048,330,240 | 13.77 | 215.2 |
| 2^23 | prime5 | 89,212 | 26,147 | 20.9 | 136,249,344 | 15.13–15.14 | 16.5 |

The binary-only shapes verify in 14.9–20.9 ms, as before v8. The mixed shape
is slower: its ordinary-root flat terms (`binary_root_weights_ordinary`)
took 141.7 ms at N=2^22 and 299.4 ms at N=2^23, against 85.9 and 160.7 ms
before v8. The v8 merge without the offload was not timed, so this rise is
not attributed.

Complete `word-prime` proving, with the prover-speed pass's v8 medians
before the offload:

| N | Proving (ms) | Three runs (ms) | Prover peak RSS bytes | One-minute loads | Before the offload (ms) |
| --- | ---: | --- | ---: | --- | ---: |
| 2^22 | 2,018.4 | 1,972.1, 2,217.5, 2,018.4 | 5,221,335,040 | 10.76, 11.47, 10.71 | 1,783.7 |
| 2^23 | 3,500.0 | 3,203.5, 3,500.0, 3,587.5 | 8,595,046,400 | 9.86, 11.18, 10.17 | 3,012.9 |

The process CPU time of the median runs rose from 11.7 to 15.6 seconds at
N=2^22 and from 20.7 to 23.4 seconds at N=2^23. At N=2^22 the prefix domain
grows from 2^24 to 2^25, and at both sizes the prover also builds the
dense `16 * 2^k'`-byte Stage 3 weight.

### Verification after the verifier speedups and the v8 merge

These rows compare two builds at N=2^22. The base is the offload after the
v8 merge, `237eb5a13` (executable SHA-256
`71f17c7eb0c73c19298578523ee2d7f9afd88e1bc897f67b16cb780c077d468a`, the
executable of the section above). The final build is the verifier speedups
after the v8 merge, `841a32355` (executable SHA-256
`acec1a1edf447f829c200380ca7380fe328ee7e90892831e4e3d66ee96c90815`). Both
come from the measurement feature set. The final executable proved each
shape once, and every proof reproduced its pin. Both executables then
verified that fixture three times in isolated processes, alternating base and
final, with Rayon's default thread pool and the catalog cache. All 24
verifications reported `verification passed=true`, matched the caller's
transcript continuation and kept online A reads forbidden. Each cell is the
median of three, with the peak RSS of that median run. The one-minute load
before each run was between 9.10 and 9.48. The logs are
`$CARGO_TARGET_DIR/320-logs/22-<shape>.{prove,r4.v<i>,r6.v<i>}.txt`.

| Shape | Base (ms) | Base RSS bytes | Final (ms) | Final RSS bytes | Final `binary_root_verify` (ms) | Loads |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| word | 15.2 | 76,087,296 | 9.9 | 74,219,520 | 3.9 | 9.48 |
| word-prime | 17.2 | 157,401,088 | 11.4 | 157,040,640 | 4.7 | 9.10–9.28 |
| word-prime-ordinary | 190.9 | 644,890,624 | 17.1 | 170,377,216 | 10.2 | 9.17 |
| prime5 | 15.3 | 127,778,816 | 11.3 | 126,992,384 | 4.3 | 9.44 |

The mixed shape's ordinary-root weights (`binary_root_weights_ordinary`)
take 2.0 ms in the final median run, against 149.4 ms in the base. The
binary-only shapes gain 4.0–5.8 ms, close to the pre-v8 gains below.

### Verification after the verifier speedups

These measurements predate the v8 merge.

Six byte-preserving commits after the B/D setup offload make the binary root
verifier cheaper. None changes setup, commitment or proof bytes, the
transcript, or the digest pins. They compare two builds:

- **Base** is commit `1bed82e3c`, whose source equals the offload commit
  `7ca9e6656` apart from documentation (executable SHA-256
  `a2cb4f7ab91231a8043e346525022d17b690182a2483335aa05bb04ca685dcfb`).
- **Final** is commit `4fcb1e43f` (executable SHA-256
  `adf284b81a8e11b745c00b2bba8820cc37fc28b01bb3bbeba964e0237b3a5fdb`).

Both executables come from the measurement feature set (`parallel`,
`binary-source-measurements`, `transcript-blake2b`, `logging-transcript`).
The base executable proved each configuration once. Both executables then
verified that fixture three times in isolated processes, alternating base and
final, with Rayon's default thread pool (parallel) and with
`RAYON_NUM_THREADS=1` (single-threaded). Every one of the 96 runs reported
`verification passed=true`, matching caller transcript continuation and
forbidden online A reads. Each cell is the median of three runs, with the
peak RSS of the median run. Other jobs shared the host: the one-minute load
before each run was between 11.33 and 12.75 and is listed per configuration.

Parallel verification:

| N | Shape | Base ns | Base RSS bytes | Final ns | Final RSS bytes | Base loads | Final loads |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2^22 | word | 15,142,208 | 75,726,848 | 10,073,500 | 73,875,456 | 12.75, 12.75, 12.75 | 12.75, 12.75, 12.75 |
| 2^22 | word-prime | 16,765,875 | 156,794,880 | 11,340,417 | 156,041,216 | 12.75, 12.53, 12.53 | 12.75, 12.53, 12.53 |
| 2^22 | word-prime-ordinary | 135,130,833 | 642,465,792 | 18,496,416 | 169,033,728 | 12.53, 12.53, 12.65 | 12.53, 12.53, 12.65 |
| 2^22 | prime5 | 15,454,333 | 126,779,392 | 11,070,084 | 126,599,168 | 12.65, 12.65, 12.65 | 12.65, 12.65, 12.65 |
| 2^23 | word | 16,360,000 | 155,762,688 | 10,521,500 | 155,336,704 | 12.65, 12.65, 12.12 | 12.65, 12.12, 12.12 |
| 2^23 | word-prime | 17,831,459 | 156,680,192 | 12,408,375 | 156,696,576 | 12.12, 12.12, 12.12 | 12.12, 12.12, 12.12 |
| 2^23 | word-prime-ordinary | 226,495,292 | 1,047,445,504 | 19,017,292 | 178,913,280 | 12.12, 11.71, 11.71 | 11.71, 11.71, 11.71 |
| 2^23 | prime5 | 17,002,667 | 133,906,432 | 12,002,583 | 134,529,024 | 11.71, 11.33, 11.33 | 11.71, 11.33, 11.33 |

Single-threaded verification:

| N | Shape | Base ns | Base RSS bytes | Final ns | Final RSS bytes | Base loads | Final loads |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2^22 | word | 24,055,375 | 72,368,128 | 18,791,041 | 71,663,616 | 12.75, 12.75, 12.75 | 12.75, 12.75, 12.75 |
| 2^22 | word-prime | 31,414,834 | 153,567,232 | 25,839,667 | 152,993,792 | 12.75, 12.53, 12.53 | 12.53, 12.53, 12.53 |
| 2^22 | word-prime-ordinary | 212,846,833 | 642,318,336 | 35,422,959 | 166,313,984 | 12.53, 12.53, 12.65 | 12.53, 12.53, 12.65 |
| 2^22 | prime5 | 29,563,167 | 124,616,704 | 25,417,667 | 125,337,600 | 12.65, 12.65, 12.65 | 12.65, 12.65, 12.65 |
| 2^23 | word | 30,198,208 | 153,518,080 | 24,942,750 | 153,337,856 | 12.65, 12.12, 12.12 | 12.65, 12.12, 12.12 |
| 2^23 | word-prime | 32,319,875 | 154,828,800 | 26,771,375 | 154,206,208 | 12.12, 12.12, 12.12 | 12.12, 12.12, 12.12 |
| 2^23 | word-prime-ordinary | 356,886,167 | 1,044,185,088 | 38,534,125 | 175,882,240 | 11.71, 11.71, 11.71 | 11.71, 11.71, 11.71 |
| 2^23 | prime5 | 32,119,250 | 131,989,504 | 27,139,667 | 132,005,888 | 11.71, 11.33, 11.33 | 11.71, 11.33, 11.33 |

The mixed shape gains most: 7.3× parallel and 6.0× single-threaded at
N=2^22, 11.9× and 9.3× at N=2^23. Its verifier peak RSS falls from 642 MB to
169 MB and from 1,047 MB to 179 MB, because the ordinary root no longer builds
one flat term per address. The binary-only shapes gain 4.4–5.8 ms parallel
and 4.1–5.6 ms single-threaded; their RSS is unchanged within 2 MB.

The inclusive spans of the single-threaded median runs locate the change,
base → final, in milliseconds. The ordinary suffix's Stage 2 verification
(`stage2_verifier`, which contains `relation_setup_scan`) is untouched and
is now the largest single cost.

| Span | 2^22 word | 2^22 word-prime | 2^22 word-prime-ordinary | 2^22 prime5 | 2^23 word | 2^23 word-prime | 2^23 word-prime-ordinary | 2^23 prime5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `verify_complete_opening` | 24.1 → 18.8 | 31.4 → 25.8 | 212.8 → 35.4 | 29.6 → 25.4 | 30.2 → 24.9 | 32.3 → 26.8 | 356.9 → 38.5 | 32.1 → 27.1 |
| `binary_root_verify` | 9.2 → 3.8 | 10.4 → 4.6 | 191.9 → 14.3 | 8.5 → 4.3 | 9.8 → 4.4 | 11.3 → 5.7 | 335.7 → 17.4 | 10.1 → 5.2 |
| `binary_root_relation_compile` | 3.9 → 1.4 | 4.3 → 1.7 | 112.7 → 5.8 | 1.7 → 1.8 | 4.0 → 1.5 | 4.5 → 2.0 | 203.4 → 9.2 | 2.2 → 2.1 |
| `binary_root_weights_ordinary` | absent | absent | 94.8 → 3.9 | absent | absent | absent | 174.9 → 7.1 | absent |
| `derive_public_matrix_prefix` | absent | absent | 2.4 → 2.4 | absent | absent | absent | 4.8 → 4.8 | absent |
| `binary_root_tau1_draw` | 0.0 → 0.0 | 0.0 → 0.0 | 3.3 → 3.3 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | 1.8 → 1.8 | 0.0 → 0.0 |
| `binary_root_relation_evaluate` | 2.8 → 1.3 | 3.4 → 1.7 | 72.4 → 3.2 | 4.3 → 1.4 | 3.3 → 1.8 | 4.1 → 2.5 | 127.4 → 4.6 | 5.3 → 1.9 |
| `binary_root_a_stage3` | 1.8 → 0.4 | 2.0 → 0.5 | 1.9 → 0.5 | 1.9 → 0.5 | 1.8 → 0.3 | 1.9 → 0.5 | 1.9 → 0.5 | 1.9 → 0.5 |
| `stage2_verifier` | 13.0 → 13.2 | 19.1 → 19.3 | 19.0 → 19.1 | 19.1 → 19.1 | 18.5 → 18.7 | 19.1 → 19.2 | 19.2 → 19.2 | 19.8 → 20.0 |
| `relation_setup_scan` | 10.0 → 10.1 | 17.0 → 17.1 | 16.8 → 17.0 | 15.8 → 15.9 | 16.4 → 16.6 | 16.9 → 17.0 | 17.1 → 17.0 | 16.5 → 16.8 |

In the parallel median runs `binary_root_verify` takes 3.8–5.5 ms for the
binary-only shapes and 11.9 and 12.4 ms for the mixed shape; `stage2_verifier`
takes 4.3–5.3 ms and `relation_setup_scan` 2.7–3.7 ms.

Each commit was measured against its parent, median of three, on the
configurations it targets; the commit messages hold the same figures. The
first commit's before figures are the base medians above, because its own
runs measured only the commit (peak RSS 169 MB and 179 MB). The runs were
under different loads than the table above, so the steps do not sum to the
base-to-final difference.

| Commit | Change | Configuration | Parallel ms | Single-threaded ms | Load |
| --- | --- | --- | --- | --- | --- |
| `85902a75e` | Ordinary root rows as translated patterns with residue kernels, evaluated in closed form | 2^22 / 2^23 mixed | 135.1 → 26.5 / 226.5 → 29.3 | 212.8 → 43.4 / 356.9 → 47.9 | 8.8–16.7 |
| `b66cfdb7f` | Ordinary root residue kernels derived in parallel; superseded by `4fcb1e43f` | 2^22 / 2^23 mixed | 26.5 → 25.9 / 29.3 → 27.5 | 43.4 → 43.4 / 47.9 → 47.2 | 8.8–16.7 |
| `86cc8a002` | Terminal partial columns as one structured family | 2^22 word / 2^23 word | 15.9 → 13.5 / 23.1 → 18.4 | 23.8 → 21.5 / 31.1 → 28.8 | 10.5–13.4 |
| `9fbde6581` | Trinomial adjoint of an equality table in closed form | 2^22 word-prime / 2^23 word-prime | 14.3 → 11.7 / 17.3 → 14.3 | 28.7 → 25.7 / 30.1 → 27.0 | 21.4–23.1 |
| `2c31ad17d` | Prime plane work shared across one polynomial's planes | 2^22 prime5 / 2^23 prime5 | 12.6 → 10.7 / 14.0 → 11.4 | 26.3 → 25.0 / 28.3 → 26.4 | 6.7–6.9 |
| `4fcb1e43f` | Ordinary root residue kernels read through low windows | 2^22 / 2^23 mixed | 20.2 → 17.7 / 20.2 → 18.3 | 37.5 → 34.6 / 42.3 → 38.0 | 7.4–7.5 |

The figures before these speedups, with the B/D offload comparison, are in
the appendix. The logs are in the measurement target directory under
`r6/logs/<log>-<shape>.{base,final}.t{all,1}.v<i>.txt`.

### Results after the v8 merge, before the offload

These runs predate the offload. All eight full-size runs and the smoke run
reproduced the v8 pins before the offload, in the second table above. Each
reported `verification passed=true`, matched the caller's transcript
continuation and kept online A reads forbidden. The logs are
`$CARGO_TARGET_DIR/v8-logs/316-<log>-<shape>.txt`, measured after the
padding-row deletion; their proofs supplied those pins. Timings were not
measured.

| N | Shape | Proof bytes | Atoms | Prover peak RSS bytes | Verifier peak RSS bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 2^22 | word | 82,773 | 25,597 | 5,123,981,312 | 75,563,008 |
| 2^22 | word-prime | 82,743 | 25,567 | 5,139,398,656 | 76,136,448 |
| 2^22 | word-prime-ordinary | 82,762 | 25,589 | 5,200,101,376 | 562,003,968 |
| 2^22 | prime5 | 79,899 | 25,440 | 7,694,123,008 | 79,511,552 |
| 2^23 | word | 84,583 | 25,749 | 9,741,746,176 | 156,172,288 |
| 2^23 | word-prime | 84,594 | 25,760 | 9,986,932,736 | 157,810,688 |
| 2^23 | word-prime-ordinary | 84,569 | 25,733 | 9,939,501,056 | 1,039,564,800 |
| 2^23 | prime5 | 81,562 | 25,588 | 9,576,710,144 | 160,710,656 |

The partition's first six categories equal the pre-v8 tables below. The
remaining payload, framing and work category is, in bytes/atoms:

| N | word | word-prime | word-prime-ordinary | prime5 |
| --- | ---: | ---: | ---: | ---: |
| 2^22 | 56,319/23,580 | 56,289/23,550 | 56,308/23,572 | 56,324/23,587 |
| 2^23 | 56,263/23,556 | 56,274/23,567 | 56,249/23,540 | 56,277/23,570 |

Each run selects six recursive folds and eight fold levels, and reports
`canonical_plan_matches=true`. Its root output and six recursive output lengths
equal the corresponding pre-v8 row below.

### Pre-v8 current-protocol results

These results and all their timings predate the v8 merge. Their proofs
reproduce the retired pre-v8 pins, not the current ones.

These rows measure the B/D setup offload without the prover-speed pass.
That pass was measured separately, on a tree without the offload; its medians
are in the prover span profile below.

All rows come from one campaign on the source committed as `7ca9e6656`. The
logged HEAD is `db34922aa`; the logged dirty tree held that commit's source
changes, uncommitted, and documentation edits. One executable (SHA-256
`252c55385c76e2bee28b13c30bf45f1151310d133215f1cd82eb3c8016d50eeb`) served
every configuration. Each configuration proved once, supplying its row of the
digest TSV, then verified three times in isolated processes from the same
fixture. The fixture setup deserialization row comes from the median
verification run; every other phase row and the prover peak RSS come from the
single proving run. All three verifications of every configuration reported
`verification passed=true`, matching caller transcript continuation and
forbidden online A reads. The logs are
`$CARGO_TARGET_DIR/bd-offload/<log>-<shape>.{prove,verify1,verify2,verify3}.txt`.
Pre-v8 verification time and verifier peak RSS after the verifier speedups
are in the section of that name above; the values this campaign measured, and the figures before the offload, are in the
appendix. The `uptime` load triples of these runs are:

| N | Shape | Before proving | After proving | After the third verification |
| --- | --- | --- | --- | --- |
| 2^22 | word | 5.76/6.74/6.58 | 6.19/6.71/6.58 | 6.19/6.71/6.58 |
| 2^22 | word-prime | 6.19/6.71/6.58 | 7.49/6.94/6.67 | 7.49/6.94/6.67 |
| 2^22 | word-prime-ordinary | 6.13/6.65/6.58 | 7.30/6.88/6.66 | 7.30/6.88/6.66 |
| 2^22 | prime5 | 7.49/6.94/6.67 | 6.23/6.68/6.59 | 6.13/6.65/6.58 |
| 2^23 | word | 7.30/6.88/6.66 | 5.86/6.56/6.56 | 5.86/6.56/6.56 |
| 2^23 | word-prime | 5.86/6.56/6.56 | 5.69/6.37/6.48 | 5.69/6.37/6.48 |
| 2^23 | word-prime-ordinary | 6.39/6.31/6.44 | 5.82/6.16/6.38 | 5.82/6.16/6.38 |
| 2^23 | prime5 | 5.69/6.37/6.48 | 6.39/6.31/6.44 | 6.39/6.31/6.44 |

| N | Shape | Proof bytes | Atoms | Prover peak RSS bytes |
| --- | --- | ---: | ---: | ---: |
| 2^22 | word | 87,801 | 25,828 | 5,180,358,656 |
| 2^22 | word-prime | 89,464 | 25,976 | 6,055,002,112 |
| 2^22 | word-prime-ordinary | 89,463 | 25,976 | 6,232,817,664 |
| 2^22 | prime5 | 94,242 | 26,374 | 10,003,873,792 |
| 2^23 | word | 89,626 | 25,992 | 10,183,802,880 |
| 2^23 | word-prime | 89,658 | 26,024 | 10,190,553,088 |
| 2^23 | word-prime-ordinary | 89,661 | 26,026 | 10,211,393,536 |
| 2^23 | prime5 | 94,261 | 26,392 | 10,593,697,792 |

The complete proof partition below is bytes/atoms per category across all fold
levels. The other category includes the remaining payload, framing and work
families. The categories sum exactly to each proof's size and atom count.

| N | Shape | Field switch | Projections | Stage 1 | Stage 2 | Stage 3 | Physical L2 | Other |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | word | 2,993/173 | 84/7 | 11,043/858 | 7,217/587 | 824/74 | 4,293/318 | 61,347/23,811 |
| 2^22 | word-prime | 2,993/173 | 84/7 | 14,357/1,142 | 7,119/579 | 857/77 | 2,754/204 | 61,300/23,794 |
| 2^22 | word-prime-ordinary | 2,993/173 | 84/7 | 14,357/1,142 | 7,119/579 | 857/77 | 2,754/204 | 61,299/23,794 |
| 2^22 | prime5 | 0/0 | 84/7 | 20,891/1,646 | 7,168/583 | 890/80 | 1,377/102 | 63,832/23,956 |
| 2^23 | word | 3,035/175 | 84/7 | 14,422/1,147 | 7,168/583 | 857/77 | 2,754/204 | 61,306/23,799 |
| 2^23 | word-prime | 3,035/175 | 84/7 | 14,422/1,147 | 7,168/583 | 857/77 | 2,754/204 | 61,338/23,831 |
| 2^23 | word-prime-ordinary | 3,035/175 | 84/7 | 14,422/1,147 | 7,168/583 | 857/77 | 2,754/204 | 61,341/23,833 |
| 2^23 | prime5 | 0/0 | 84/7 | 20,891/1,646 | 7,168/583 | 890/80 | 1,377/102 | 63,851/23,974 |

Where the typed prefix grows to 2^25 at N=2^22 (`word-prime` and
`word-prime-ordinary`), the successor schedule changes the way it does at
N=2^23, whose prefix was already 2^25: Stage 1 grows from 11,043 to 14,357
bytes, Physical L2 shrinks from 4,293 to 2,754 bytes, and Stage 3 grows from
824 to 857 bytes (one round). At 2^26 (`prime5` at both sizes) Stage 1 reaches
20,891 bytes. N=2^22 `word` and the N=2^23 shapes other than `prime5` keep
their partitions except for value-dependent coding in the other category.

Each final current-protocol run selected six ordinary recursive folds and
eight total fold levels. The root's exact outgoing physical lengths and
commitment domains match the canonical geometry table below, with
`canonical_plan_matches=true`. The `native_transition` line of each log
reports the typed prefix's natural cells, which equal the values in the
setup offload note. The first suffix uses A degree 512. The six successive
recursive output lengths below identify the measured chain; the sixth is the
terminal input. Full frozen profiles, block geometry and other matrix
dimensions remain in the logs.

| N | Shape | Successive recursive output cells |
| --- | --- | --- |
| 2^22 | word | 21,206,528 → 2,633,536 → 953,408 → 280,448 → 128,896 → 93,696 |
| 2^22 | word-prime | 17,269,248 → 1,757,184 → 552,960 → 245,504 → 121,856 → 93,696 |
| 2^22 | word-prime-ordinary | 17,445,376 → 1,771,520 → 565,248 → 245,504 → 121,856 → 93,696 |
| 2^22 | prime5 | 19,017,216 → 1,752,448 → 579,712 → 207,104 → 124,672 → 93,696 |
| 2^23 | word | 17,777,152 → 1,785,856 → 565,248 → 245,504 → 121,856 → 93,696 |
| 2^23 | word-prime | 17,869,312 → 1,800,192 → 565,248 → 245,504 → 121,856 → 93,696 |
| 2^23 | word-prime-ordinary | 18,182,656 → 1,814,528 → 565,248 → 245,504 → 121,856 → 93,696 |
| 2^23 | prime5 | 19,377,664 → 1,765,120 → 579,712 → 207,104 → 124,672 → 93,696 |

The phase tables report inclusive wall times in nanoseconds, under the load
triples above.

N=2^22:

| Phase | word ns | word-prime ns | word-prime-ordinary ns | prime5 ns |
| --- | ---: | ---: | ---: | ---: |
| Catalog planning | 26,377,241,416 | 28,476,150,041 | 10,488,192,167 | 29,101,533,000 |
| Cold prover setup | 41,027,459 | 81,349,250 | 81,609,875 | 168,867,875 |
| Independent fixture evaluation | 17,466,084 | 1,673,474,542 | 1,668,774,000 | 8,242,829,292 |
| Binary commitment | 1,658,082,500 | 1,658,549,875 | 1,656,235,958 | 1,654,998,208 |
| Ordinary commitment | absent | absent | 12,199,375 | absent |
| Cold prefix preparation | 209,160,833 | 470,878,917 | 469,966,167 | 374,030,375 |
| Cold verifier setup | 190,201,166 | 471,201,750 | 444,318,375 | 342,559,959 |
| Fixture serialization | 36,084,125 | 94,831,250 | 92,006,625 | 97,901,334 |
| Fixture setup deserialization | 67,061,667 | 169,304,125 | 169,284,834 | 132,578,625 |
| Parity row convolution | 9,774,666 | 10,652,542 | 9,926,000 | absent |
| Complete proving | 11,321,687,750 | 12,357,436,292 | 12,534,301,083 | 16,359,650,083 |

N=2^23:

| Phase | word ns | word-prime ns | word-prime-ordinary ns | prime5 ns |
| --- | ---: | ---: | ---: | ---: |
| Catalog planning | 31,336,903,750 | 31,482,820,625 | 11,295,199,625 | 31,778,615,875 |
| Cold prover setup | 83,035,417 | 82,813,792 | 79,797,625 | 166,126,208 |
| Independent fixture evaluation | 34,367,083 | 3,331,080,459 | 3,332,365,000 | 16,487,547,750 |
| Binary commitment | 1,873,371,916 | 1,876,317,791 | 1,867,254,958 | 1,870,184,917 |
| Ordinary commitment | absent | absent | 16,369,917 | absent |
| Cold prefix preparation | 579,331,542 | 575,989,167 | 583,435,667 | 506,762,375 |
| Cold verifier setup | 548,335,375 | 544,259,958 | 550,007,291 | 475,372,292 |
| Fixture serialization | 127,670,334 | 121,204,084 | 94,683,834 | 84,029,250 |
| Fixture setup deserialization | 169,508,125 | 169,406,583 | 169,441,375 | 140,536,833 |
| Parity row convolution | 19,365,292 | 19,173,750 | 20,151,834 | absent |
| Complete proving | 21,856,220,334 | 22,819,519,709 | 23,138,842,542 | 26,659,122,166 |

### Prover span profile

Six byte-preserving prover commits after the span commit `f5e9ad551`, ending
at `06b8435eb`, cut complete proving by a factor of 6.3 to 7.7 across the
shapes. Each change either reorders an exact field sum or removes work whose
output was unused.
No layout, schedule, transcript or proof byte changed. All 48 runs below
reproduced their digest pins.

The tables below predate the v8 merge. After merging the v8 port of the
padding-row deletion (`53b25df19`, measured at `85d0d17b3`) and before the
offload, `word-prime` was proved three more times per size with the catalog
cache. Every run
reproduced its v8 pin. The one-minute loads were higher than in the pre-v8
sweep, so the small increase is within the host's noise:

| N | Complete proving after the v8 merge, before the offload (ms) | Pre-v8 final (ms) | Prover peak RSS bytes | One-minute load |
| --- | ---: | ---: | ---: | --- |
| 2^22 | 1,783.7 | 1,737.8 | 4,193,042,432 | 9.48–11.76 |
| 2^23 | 3,012.9 | 2,927.9 | 8,056,356,864 | 8.90–10.26 |

Those logs are `$CARGO_TARGET_DIR/v8-logs/317-q-<log>-word-prime-<run>.txt`.

The driver records each prover span's call count, inclusive wall time and
process CPU time (`getrusage` user plus system). It also records live and
peak heap through a counting global allocator. Prover peak RSS is the
driver's isolated-process figure.

Each configuration proved three times in isolated processes, interleaving the
baseline executable (`f5e9ad551`) with the final executable (`06b8435eb`).
All runs set `BINARY_SOURCE_CATALOG_CACHE` to one catalog that was planned once,
so planning is excluded from both executables. This also lowers peak RSS
relative to the earlier single runs, which planned inside the proving process.
These sweep executables chose a cache entry by its filename. The driver now
stores the protocol, size, shape, schedule family and planner-policy digests
in each entry and re-prices the binary root from the cached rows. It rejects
an entry that disagrees with the request, and prints the admitted
`catalog_digest` in every proving run.
The tables give medians in milliseconds, except where a row says otherwise.
A row marked with `›` is a sub-span of the nearest row above it with one mark
fewer; inclusive times overlap and must not be summed.
Logs are `$CARGO_TARGET_DIR/r3/m-{c1,fin}-<log>-<shape>-<run>.txt`, and
each starts with its `uptime` load triple. One-minute loads before and after
the runs of each executable stayed within these ranges:

| N | Shape | Baseline runs | Final runs |
| --- | --- | --- | --- |
| 2^22 | word | 3.24–6.08 | 5.15–6.08 |
| 2^22 | word-prime | 5.94–8.67 | 6.40–8.67 |
| 2^22 | word-prime-ordinary | 5.20–7.52 | 5.20–5.96 |
| 2^22 | prime5 | 4.97–10.74 | 4.97–10.74 |
| 2^23 | word | 4.95–6.98 | 4.95–6.98 |
| 2^23 | word-prime | 4.51–6.47 | 4.51–5.88 |
| 2^23 | word-prime-ordinary | 4.24–6.64 | 4.24–6.64 |
| 2^23 | prime5 | 4.97–7.17 | 4.97–7.17 |

Final executable, N=2^22:

| Span | word | word-prime | word-prime-ordinary | prime5 |
| --- | ---: | ---: | ---: | ---: |
| Binary commitment | 368.2 | 379.9 | 316.2 | 331.2 |
| › pack words | 26.1 | 26.6 | 23.9 | 25.0 |
| › A matrix derivation | 86.6 | 86.5 | 81.7 | 84.3 |
| › image hint (A·s) | 103.5 | 107.2 | 90.0 | 95.1 |
| › B payload | 145.7 | 147.2 | 115.6 | 122.5 |
| Complete proving | 1,628.7 | 1,737.8 | 1,837.1 | 2,534.3 |
| › binary root level | 1,353.8 | 1,479.2 | 1,581.7 | 2,276.6 |
| ›› field switch | 57.9 | 50.7 | 45.1 | 0.0 |
| ›› D commitment | 22.2 | 100.9 | 98.2 | 357.3 |
| ››› D payload | 6.1 | 61.4 | 57.5 | 269.7 |
| ›› response build | 121.1 | 124.9 | 109.8 | 107.6 |
| ››› response fold | 71.2 | 73.4 | 57.6 | 62.5 |
| ››› successor table | absent | absent | absent | absent |
| ››› repack | 12.1 | 12.3 | 15.1 | 22.8 |
| ›› projections | 256.7 | 247.2 | 214.8 | 225.6 |
| ››› residual rows | 191.0 | 177.9 | 158.9 | 164.0 |
| ›› relation compile | 168.6 | 140.3 | 321.2 | 138.8 |
| ›› Stage 1 | 18.9 | 15.8 | 17.0 | 20.5 |
| ›› weight materialization | 149.1 | 227.8 | 228.2 | 584.3 |
| ›› Stage 2 | 327.7 | 323.7 | 326.4 | 591.5 |
| ›› Stage 3 | 124.6 | 126.0 | 120.7 | 148.9 |
| Complete proving CPU | 10,890 | 12,510 | 12,447 | 18,017 |
| Proving heap peak MB, maximum | 2,855 | 2,859 | 2,940 | 4,766 |
| Prover peak RSS bytes | 4,196,433,920 | 4,188,766,208 | 4,743,806,976 | 6,940,606,464 |

Final executable, N=2^23:

| Span | word | word-prime | word-prime-ordinary | prime5 |
| --- | ---: | ---: | ---: | ---: |
| Binary commitment | 577.0 | 536.7 | 536.0 | 531.7 |
| › pack words | 55.1 | 50.5 | 49.8 | 49.7 |
| › A matrix derivation | 175.0 | 169.5 | 169.5 | 169.6 |
| › image hint (A·s) | 207.3 | 189.4 | 187.9 | 184.8 |
| › B payload | 135.8 | 122.5 | 122.8 | 121.1 |
| Complete proving | 2,901.9 | 2,927.9 | 3,369.4 | 3,726.5 |
| › binary root level | 2,473.9 | 2,549.4 | 2,982.8 | 3,342.1 |
| ›› field switch | 112.7 | 95.8 | 95.2 | 0.0 |
| ›› D commitment | 41.1 | 118.1 | 135.4 | 405.9 |
| ››› D payload | 6.3 | 57.2 | 55.8 | 266.8 |
| ›› response build | 207.4 | 200.5 | 208.1 | 186.2 |
| ››› response fold | 121.4 | 117.4 | 117.8 | 126.0 |
| ››› successor table | absent | absent | absent | absent |
| ››› repack | 22.7 | 22.4 | 25.9 | 23.7 |
| ›› projections | 470.4 | 427.8 | 440.5 | 437.8 |
| ››› residual rows | 337.6 | 314.3 | 314.8 | 313.9 |
| ›› relation compile | 293.8 | 282.3 | 617.8 | 279.8 |
| ›› Stage 1 | 26.1 | 22.4 | 28.1 | 27.0 |
| ›› weight materialization | 242.3 | 372.5 | 399.0 | 928.3 |
| ›› Stage 2 | 627.9 | 612.3 | 623.5 | 634.3 |
| ›› Stage 3 | 254.4 | 243.8 | 249.8 | 249.0 |
| Complete proving CPU | 19,784 | 20,892 | 21,623 | 25,919 |
| Proving heap peak MB, maximum | 5,667 | 5,670 | 5,820 | 5,682 |
| Prover peak RSS bytes | 8,075,493,376 | 8,051,064,832 | 8,649,621,504 | 8,057,487,360 |

The baseline tables are in the appendix. The commits and their measured
effects on N=2^22 `word-prime` are listed below. Each commit's executable
proved three times in isolated processes with the catalog cache, and each
figure is the median of those runs. A commit's before figure is the median of
the previous executable, so `3cf7db7bf` starts from the span baseline
`f5e9ad551`, whose runs are the baseline tables' word-prime 2^22 runs.

The logs are `$CARGO_TARGET_DIR/r3/m-c<k>-22-word-prime-<run>.txt`. Executable
`c1` is `f5e9ad551`, and `c2` through `c7` are the six commits in table order.
Each log records the `uptime` load triple before and after its run. The last
column gives the range of one-minute loads in those records for the commit's
own runs; the baseline runs ranged over 5.94–8.67.

| Commit | Change | Span before → after (ms) | Complete proving before → after (ms) | One-minute load |
| --- | --- | --- | --- | --- |
| `3cf7db7bf` | Direct A weights by exact integer adjoints in a smooth auxiliary field | relation compile 7,171.4 → 135.7 | 12,286.0 → 5,359.2 | 9.32–11.88 |
| `e8950959c` | Skip the residual rows the response build never reads | response build 1,594.5 → 306.4 | 5,359.2 → 3,908.9 | 5.80–6.13 |
| `5c5068822` | Transform the quotient operands in parallel | residual rows 1,244.8 → 161.1 | 3,908.9 → 2,908.3 | 6.56–8.92 |
| `337760b99` | Commit column-aligned B and D digit pieces in parallel | B payload 1,454.2 → 128.6; D payload 657.1 → 59.8 | 2,908.3 → 2,285.5 | 5.62–7.03 |
| `8321a72b3` | Weigh the structured root families in parallel batches | weight materialization 636.3 → 215.9 | 2,285.5 → 1,817.9 | 5.23–5.62 |
| `06b8435eb` | Repack the root witness from natural digits | response build 308.3 → 108.4 | 1,817.9 → 1,640.9 | 6.70–7.61 |

These logs, not the commit messages, are the source. The commit messages
transcribe some of these figures and loads, but not uniformly: `3cf7db7bf`
records neither, and `337760b99` gives a D-payload before figure of 665.6 ms,
which is the `3cf7db7bf` executable's median. The `5c5068822` runs that
precede `337760b99` give 657.1 ms. An earlier version of this table started
`3cf7db7bf` from an earlier profiling run (relation compile 7,496.0 ms,
complete proving 13,153.3 ms) whose log was not retained.

Loads differed from commit to commit, between 5.23 and 11.88. The per-commit
columns are therefore not an exact decomposition of the end-to-end gain; the
interleaved tables are the comparison.

The remaining N=2^22 `word-prime` cost splits into three kinds.

- **Representation cost.** Stage 2 (about 324 ms) runs the ordinary
  range-image sumcheck over a dense 2^25-cell table. Weight materialization
  (about 228 ms) fills that table from the structured families. The response
  lanes use a 1,024-cell stride for 648 coefficients, so 376 of every 1,024
  lane cells are padding. A structured Stage 2 oracle would remove most of both spans.
  That needs a prover seam in the ordinary Stage 2 session.
- **New protocol work.** The new protocol work has no counterpart in an
  ordinary opening. It consists of the projections (about 247 ms, mostly
  residual rows), relation compilation (about 140 ms; 321 ms with the mixed
  ordinary group), Stage 3 (about 126 ms), the D commitment (about 101 ms) and
  the field switch (about 51 ms). Stage 3 and the binary commitment's A matrix
  derivation (about 87 ms) include deriving setup entries from the seed.
- **Ordinary recursion.** The six ordinary suffix folds take about 250 ms. An
  ordinary 2^24-variable opening of a comparable 15.6M-cell level-0 witness
  proves in about 200 ms on the same host.

The kernels shared with the earlier binary prototype in `akita-labinius-prover`
are not the bottleneck. The response fold uses its bit-sliced carry-save
kernel, and the image hint uses its prepared limb commitment. Scaled to this
geometry, each spends about 20–30% more CPU time than that crate's own
single-threaded benchmark.

#### Structured Stage 2

The binary root's Stage 2 now reads its public weight as block profiles
instead of a dense table. This removes the weight materialization span, and
round 0 runs on the packed digits. The design and full measurements are in
[`binary-source-stage2-oracle.md`](binary-source-stage2-oracle.md).

On the current base, `dev` at `ff2297aae` with the B/D setup offload, the
baseline executable is `ff2297aae` and the structured executable is
`537beafa2`. Each figure is the median of three interleaved runs with the
catalog cache; all 35 runs, warm-ups included, reproduced their pins and
verified. One-minute loads were 4.91–15.77.

| N | Shape | Weight materialization + Stage 2 (ms) | Complete proving (ms) | Prover peak RSS bytes |
| --- | --- | --- | --- | --- |
| 2^22 | `word` | 421.6 → 178.5 | 1,583.4 → 1,494.0 | 4,445,175,808 → 2,996,060,160 |
| 2^22 | `word-prime` | 588.5 → 250.9 | 2,391.3 → 2,105.5 | 5,238,603,776 → 5,181,784,064 |
| 2^22 | `word-prime-ordinary` | 583.3 → 271.3 | 2,560.5 → 2,226.9 | 5,584,814,080 → 5,495,062,528 |
| 2^22 | `prime5` | 1,167.8 → 395.0 | 3,471.4 → 2,818.0 | 9,370,451,968 → 8,814,641,152 |
| 2^23 | `word-prime` | 1,027.1 → 345.7 | 3,513.2 → 2,900.1 | 8,610,463,744 → 5,448,220,672 |

Stage 2's own live-heap peak falls by 26–48%. For the 2^22 shapes with prime
planes, Stage 3 now sets the process peak: its dense weight over the typed
A/B/D prefix is unchanged by this work, so peak RSS barely moves there. At
2^22 `word` and 2^23 `word-prime`, Stage 3 stays below the old Stage 2 peak
and peak RSS falls by a third.

The earlier measurement, on base `f92e959ee` before the B/D setup offload,
used the structured executable `2e59907c9` and 33 runs at one-minute loads of
7.37–12.69:

| N | Shape | Weight materialization + Stage 2 (ms) | Complete proving (ms) | Prover peak RSS bytes |
| --- | --- | --- | --- | --- |
| 2^22 | `word` | 451.7 → 175.0 | 1,524.1 → 1,273.1 | 4,194,680,832 → 2,581,151,744 |
| 2^22 | `word-prime` | 555.6 → 227.3 | 1,746.6 → 1,416.0 | 4,195,729,408 → 2,646,753,280 |
| 2^22 | `word-prime-ordinary` | 572.5 → 270.9 | 2,041.4 → 1,753.5 | 4,753,358,848 → 3,263,168,512 |
| 2^22 | `prime5` | 1,194.2 → 415.7 | 2,686.3 → 1,866.9 | 6,945,423,360 → 2,792,046,592 |
| 2^23 | `word-prime` | 993.5 → 334.3 | 3,148.9 → 2,497.3 | 8,059,813,888 → 4,623,138,816 |

The representation cost described above is now mostly the authentication
family's weight build, which has no shared structure.

#### Stage 3 without the dense weight

After the structured Stage 2, Stage 3 sets the prover's live-heap peak for the
shapes with prime planes. Its weight is still a dense table over the typed
prefix: `BinaryTypedPrefix::prover_weights` fills `16 * 2^k'` bytes, where
`2^k'` is the prefix domain. This subsection profiles that Stage 3 and plans
how to remove the dense weight without changing a byte.

**Profile.** The executable is `fab398c16` plus the Stage 3 phase spans of
`57d148ab3` and a few temporary spans inside the typed-prefix builders. Each
figure is the median of three runs at N=2^22 with the catalog cache. All 16
runs, warm-ups included, reproduced their digest pins and verified.
One-minute loads were 5.39–14.58.

| Shape | `2^k'` | Complete proving (ms) | Stage 2 (ms) | Stage 3 (ms) | Prover peak RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `word` | 2^24 | 1,301.3 | 150.4 | 278.8 | 2,990,473,216 |
| `word-prime` | 2^25 | 1,704.4 | 192.7 | 445.8 | 5,188,812,800 |
| `word-prime-ordinary` | 2^25 | 1,900.6 | 226.0 | 429.7 | 5,500,960,768 |
| `prime5` | 2^26 | 2,600.7 | 348.8 | 757.3 | 8,804,007,936 |

The Stage 3 span tree, in milliseconds, with the live heap in GB:

| Shape | Layout | Source | Weights | Begin (coefficient pass) | Rounds | Heap at entry → peak |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `word` | 1.6 | 130.6 | 18.1 | 66.2 (8.4) | 62.7 | 0.96 → 1.77 |
| `word-prime` | 1.6 | 166.5 | 27.3 | 131.6 (13.2) | 117.8 | 1.69 → 3.30 |
| `word-prime-ordinary` | 1.6 | 156.9 | 27.5 | 128.4 (12.9) | 114.9 | 1.76 → 3.37 |
| `prime5` | 1.6 | 234.0 | 53.6 | 256.1 (23.8) | 212.1 | 2.78 → 6.00 |

- *Layout* builds `BinaryTypedPrefix` and `BinaryAProduct`.
- *Source* derives the typed setup table with
  `BinaryTypedPrefix::setup_coefficients` and copies it into an `Arc<[F]>`.
  At `word-prime`, a temporary span puts the serial A derivation inside it at
  124 ms per call and the B/D copies into the A padding at 17 ms. The copy
  briefly holds the table twice: the source span's own peak is the entry heap
  plus `32 * 2^k'` bytes.
- *Weights* is `prover_weights`: one more `16 * 2^k'` bytes.
- *Begin* is `begin_factored_stage3`. The parallel coefficient pass computes
  the input claim. The rest of the span is `combine_index_weights`, which
  copies the dense weight serially into a third table of `16 * 2^k'` bytes,
  because a one-coefficient view scales the weight by one. That copy is the
  Stage 3 peak: source, weight and combined weight, `48 * 2^k'` bytes above
  the entry heap.
- *Rounds* runs the `k'` degree-two rounds; the first fold halves the table.

The typed table that *Source* derives was already derived and committed by
the cold prefix preparation of the same process (`cold_prefix_preparation`,
415 ms at `word-prime`). The prover derives it three times per process, so
the serial A derivation costs about 370 ms of the `word-prime` process.

**Where the weight comes from.** The prefix holds the A tensor and the B/D
digit pieces. Every other cell is zero in the setup table and in the weight.

- *A tensor.* A cell is `row * row_stride + rank * 1024 + c`, with
  `row_stride = 2^ceil(log2(rank)) * 1024`. The weight is
  `I[row] * C[rank * 1024 + c]`: the root-point equality of the row times one
  coefficient factor. It is a tensor product. `C` is zero for `c >= 648` and
  on the padded ranks, so it is sparse: at rank 3 only `3 * 704` of every
  4,096 cells lie in a live 64-cell chunk.
- *B/D pieces.* Copy column `j` of period `q` of a digit segment is
  `beta_row * K(e_{q,j})`, where `K` is the residue kernel at `alpha` and
  `e_{q,j}[t] = eq(r, address)`. The address is
  `(f + qP + p) * stride + c` for lane `p` of the period and coefficient `c`,
  so the equality factors as
  `coeff_eq[c] * lane_eq[(f + p) mod P] * eq(r_hi, floor(f / P) + s_p + q)`,
  with `s_p = ((f mod P) + p) >> log2(P)` either 0 or 1. Hence
  `e_{q,j} = G_0(q) E_0[j] + G_1(q) E_1[j]`, where
  `G_s(q) = eq(r_hi, floor(f / P) + s + q)` and `E_s[j]` does not depend on
  `q`. The weight is periodic in `q` up to two scalars. A partial last period
  uses its own `E_s`. `K` is linear, so every copy is a sum of two scaled rows
  `K(E_s[j])`, and a segment needs at most `4 * columns` rows.

Every A chunk starts on a multiple of 64. Every B/D copy is an aligned
64-cell block, since every region keeps the six slot bits free and first.
So the weight is, per live 64-cell block, `scale_0 * row_0 + scale_1 * row_1`
over a pool of a few thousand rows.

**How the rounds bind.** `SetupProductSumcheck` with one coefficient binds
the least significant address bit first. Its round is
`(sum s0 w0, sum (s0 dw + ds w0), sum ds dw)` over the cell pairs. Rounds 0–5
bind the six bits inside a 64-cell block. A live block stays live and a zero
block stays zero. Binding folds each pool row in place, and the block records
do not change. After six rounds each block is one cell, and the remaining
table has `2^(k'-6)` entries, 8 MB at `2^k' = 2^25`.

**Plan**, largest gain first:

1. *A structured seam, behaviour-preserving.* Add
   `SetupProductSumcheck::structured`, which takes the setup table and a
   `StructuredSetupWeights` oracle. It runs rounds 0–5 over the live blocks
   only. Per block and pool row it accumulates `sum s0 r0`, `sum s1 r1` and
   `sum ds dr` unreduced. It scales them by the block's scales and derives the
   linear coefficient as `sum s1 w1 - constant - quadratic`. All of these
   are exact field sums. After six binds it scatters the block values into
   the dense index state and continues on the ordinary rounds.
   `OpaqueStage3Kernel::begin_typed_stage3` exposes the seam behind
   `labinius-trinomial`. Ordinary callers keep `begin_factored_stage3`
   unchanged. A differential test runs both on compiled roots.
2. *Switch the binary root to it.* This removes the weights span (18–54 ms),
   the combine copy (58–232 ms) and two `16 * 2^k'` tables, 0.5–2.1 GB of
   live heap. Rounds 0–5, about half of the rounds span, read only the live
   blocks.
3. *Reuse the prepared prefix table.* The prepared prefix handle keeps the
   committed typed table. Reading it instead of re-deriving it removes most
   of the source span (131–234 ms) and its transient second copy.
4. *Other regressions after the B/D setup offload*: page faults in Stage 3
   after a smaller Stage 2, the serial A derivation in the cold preparation,
   and the B/D copies into the A padding.

**Results.** Items 1–3 landed in `90e6de4d7`, `c1a6ab234` and `4f1719168`.
Item 4 landed as `75000a98e`: the A derivation, its i128 lift and its prefix
scatter now run in parallel. Each matrix entry already had its own setup
stream. With the structured weight and the reused table, Stage 3 allocates no
prefix-sized table before its rounds, so the page faults after Stage 2 are
gone with them. Setup, commitment and proof bytes are unchanged.

The table compares `fab398c16` plus spans (the profile executable above) with
`75000a98e`. Each figure is the median of three interleaved runs after a
warm-up, with the catalog cache. All 40 runs reproduced their digest pins and
verified. One-minute loads were 5.77–11.70 at N=2^22 and 6.99–8.80 at
N=2^23. Times are in milliseconds.

| N | Shape | Complete proving | Stage 3 | Stage 3 heap peak (GB) | Prover peak RSS bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 2^22 | `word` | 1,375.8 → 1,052.7 | 294.3 → 39.9 | 1.77 → 1.18 | 2,985,230,336 → 2,221,113,344 |
| 2^22 | `word-prime` | 1,806.3 → 1,279.3 | 490.4 → 47.7 | 3.30 → 1.95 | 5,177,180,160 → 3,106,586,624 |
| 2^22 | `word-prime-ordinary` | 2,095.4 → 1,536.8 | 496.8 → 51.2 | 3.37 → 2.03 | 5,491,441,664 → 3,690,053,632 |
| 2^22 | `prime5` | 2,818.5 → 2,108.9 | 863.4 → 73.6 | 6.00 → 3.24 | 8,806,416,384 → 5,414,109,184 |
| 2^23 | `word-prime` | 2,627.6 → 1,950.5 | 580.6 → 61.1 | 3.50 → 2.26 | 5,435,490,304 → 3,981,492,224 |

Outside complete proving, the A derivation also shortens the cold prefix
preparation and the cold verifier setup. At N=2^22 `word-prime` they go from
430.9 to 339.2 ms and from 453.0 to 328.5 ms. At `word` they nearly halve,
from 204.7 to 112.0 ms and from 188.5 to 97.2 ms.

**What remains.**

- Stage 3 is now its rounds: 40–74 ms at N=2^22. They read the committed
  table in place, which costs 6–9 ms over a freshly written copy. Rounds 6
  and later run on the dense index state of `2^(k'-6)` entries.
- Stage 3 no longer sets the live-heap peak of complete proving. The peak is
  now Stage 2's `RelationRangeImageProver::fold_round`: 1.52, 2.28 and
  3.81 GB at N=2^22 for `word`, `word-prime` and `prime5`. At `word-prime`
  and `prime5` the cold verifier setup peaks slightly higher, at 2.34 and
  3.93 GB.
- The cold prefix preparation and verifier setup still take 250–350 ms at
  N=2^22 for the prime shapes. They already use about twelve cores. They
  have no inner spans, so the split between deriving and committing the
  typed prefix is not yet measured. The B/D copies into the A padding
  (17 ms) derive one setup entry per copy. Deriving each entry once would
  save part of that, but only in the cold path.

### Second prover pass

A second byte-preserving pass targets the binary root spans outside weight
materialization and Stage 2, and profiles the ordinary suffix folds. Its
profile executable adds sub-spans to `f92e959ee`: the A matrix derived in
proving (`binary_root_a_matrix`), the Stage 3 A coefficients, the three phases
of the residual rows, the direct A adjoint, and the field switch's source
extraction, partial evaluations and rounds. The profile is three isolated
proving runs at N=2^22 `word-prime` with `BINARY_SOURCE_CATALOG_CACHE`. Every
run reproduced its pin. The host was shared and the one-minute load was
27.2–37.5, so these spans run 20–40% above the post-merge figures above. The
third run was disturbed (complete proving 8.5 s) and is excluded by the
median. Logs are `$CARGO_TARGET_DIR/p2/logs/s<run>-22-word-prime.txt`.

| Span (ms, median of 3) | Wall | Process CPU |
| --- | ---: | ---: |
| Complete proving | 2,431.9 | 12,185.5 |
| › binary root level | 2,013.7 | 9,668.0 |
| ›› projections | 390.1 | 2,481.8 |
| ››› response fold (replayed) | 116.5 | 840.2 |
| ››› residual rows | 289.6 | 1,713.4 |
| ›››› response and challenge operands | 80.7 | 409.1 |
| ›››› A row operands (3 calls) | 43.2 | 46.2 |
| ›››› transform products (3 calls) | 168.9 | 1,231.8 |
| ›› A matrix derivation before relation compile | 97.6 | 87.6 |
| ›› relation compile | 257.5 | 1,912.1 |
| ››› direct A adjoint | 250.1 | 1,904.7 |
| ›› D commitment | 143.8 | 1,011.2 |
| ››› D payload | 98.7 | 813.4 |
| ››› prime digits | 30.7 | 177.8 |
| ››› row partials | 16.4 | 16.4 |
| ›› field switch | 64.2 | 251.6 |
| ››› source extraction | 18.1 | 18.0 |
| ››› partial evaluations | 17.8 | 100.7 |
| ››› rounds | 28.9 | 133.2 |
| ›› Stage 3 | 131.5 | 245.1 |
| ››› A coefficients | 106.4 | 106.4 |
| ››› coefficient and index passes | 17.2 | 116.5 |
| › ordinary suffix (complete minus root) | 418.2 | 2,517.5 |

The response fold span runs twice, once in the response build and once in the
projection replay; the row gives half the two-call total.

What these spans spend their time on:

- **Residual rows.** Each row multiplies the bounded response by every A
  entry in the 128-bit auxiliary field, through 4,096 response, 768 image,
  256 challenge and 12,288 A transforms of length 1,296. The response, A and
  image operands are converted serially. The quotients are computed but never
  read: the projections read only the remainders.
- **Duplicate setup derivation.** The commitment derives A from the seed and
  keeps it in the private source handle. Proving derives it again before the
  relation compile (98 ms, serial) and a third time inside the Stage 3 A
  coefficients (106 ms, serial, including filling the 2^24-cell table).
- **Response fold.** The projections replay the fold that the accepted
  response build already computed.
- **Direct A adjoint.** Each of 4,096 source rows runs three forward and two
  inverse auxiliary transforms of length 1,944. The output needs only the
  middle D coefficients of each linear product, which a length-1,296 cyclic
  transform already gives without wrap-around.
- **D payload.** Each setup column multiplies a 64-coefficient setup entry by
  64 balanced digits with full field products.

Candidate changes, with the expected gain at N=2^22 `word-prime` scaled to the
post-merge figures:

| Candidate | Spans removed or reduced | Expected gain (ms) |
| --- | --- | ---: |
| Residual remainders over limb primes by CRT, without quotients | residual rows | 150–170 |
| Reuse the commitment's A in the relation compile and Stage 3 | A derivation, Stage 3 A coefficients | 140–170 |
| Reuse the accepted response in the projections | replayed response fold | 60–75 |
| Length-1,296 middle products in the direct A adjoint | relation compile | 30–50 |
| Small-digit accumulation in the B and D digit commitments | D payload; B payload in the commitment | 30–40 |
| Parallel A derivation and field-switch source extraction | binary commitment, field switch | 10–20 in proving |

The ordinary suffix folds take about 420 ms in this profile (about 260 ms at
the post-merge load). Their largest spans are Stage 2 (130 ms over six
folds), `RingRelationProver::prepare` (119 ms, 86 ms in the first fold),
the fold grinding (92 ms, of which the sampling is 61 ms), Stage 1 (76 ms)
and the six successor commitments (about 43 ms). The first suffix fold,
with a 21.3M-cell input, takes about 60% of the suffix. None of these spans
lives in binary-source code.

The direct A adjoint cannot use length 1,296: the auxiliary field's smooth
subgroup has order 17,496 = 2^3 * 3^7, so no transform length has four factors
of two. The change below uses 1,458 = 2 * 3^6, the smallest admissible length
of at least 2D-1.

Second-pass changes, in commit order. Each row is an interleaved comparison
against the previous commit: median of 3, `BINARY_SOURCE_CATALOG_CACHE`, every
pin reproduced. Rows were measured at different loads, so read each row's
before and after together, not down a column.

| Change | Shape | Span: before → after (ms) | Complete proving (ms) | Load |
| --- | --- | --- | ---: | --- |
| Residual remainders over limb primes by CRT | `word-prime` | residual rows 217.1 → 28.0; projections 281.7 → 104.2 | 2,164.6 → 1,720.9 | 35.4–44.0 |
| Reuse the commitment's A | `word-prime` | A derivation 78.8 → 0; Stage 3 A coefficients 111.0 → 10.5 | 1,842.8 → 1,632.6 | 8.0–13.0 |
| Reuse the accepted response in the projections | `word-prime` | projections 85.7 → 25.8 | 1,422.5 → 1,349.2 | 9.2–11.2 |
| Length-1,458 middle products in the integer adjoint | `word-prime` | direct A 209.0 → 157.9 (CPU 1,938 → 1,469) | within noise | 13.4–15.3 |
| Small-digit accumulation in digit commitments | `word-prime` | D payload 70.4 → 29.9; commitment B payload 160.7 → 63.7 | 1,498.6 → 1,436.0 | 11.8–16.0 |
| Parallel ordinary root weights, allocation-free recomposition | `word-prime-ordinary` | ordinary weights 217.8 → 74.4; relation compile 366.9 → 211.7 | 1,871.4 → 1,686.3 | 9.8–11.0 |
| Parallel field-switch source gather | `word-prime` | field-switch source 17.0 → 6.0 | 1,314.8 → 1,304.0 | 7.4–10.5 |
| Parallel q setup matrix derivation | `word-prime` | commitment A 82.9 → 19.1; cold typed-prefix preparation 140.0 → 75.1 | not in proving | 7.4–10.5 |

Final comparison of `f92e959ee` against the last commit, interleaved, median
of 3 isolated proving runs per side with `BINARY_SOURCE_CATALOG_CACHE`. All
thirty runs reproduced their pins. Logs are
`$CARGO_TARGET_DIR/p2/logs/ab-base-c7-{B,A}<run>-<log>-<shape>.txt` and, for
the repeated N=2^23 comparison, `r23-{B,A}<run>-23-word-prime.txt`.

| Shape | Complete proving (ms) | Binary root level (ms) | Prover peak RSS (GB) | Load |
| --- | ---: | ---: | ---: | --- |
| `word`, N=2^22 | 1,564.0 → 1,140.1 | 1,293.0 → 858.2 | 4.19 → 3.95 | 9.7–11.9 |
| `word-prime`, N=2^22 | 1,828.4 → 1,308.2 | 1,545.7 → 1,016.5 | 4.20 → 3.96 | 11.9–17.7 |
| `word-prime-ordinary`, N=2^22 | 2,009.6 → 1,421.0 | 1,726.2 → 1,129.7 | 4.76 → 4.31 | 15.1–16.7 |
| `prime5`, N=2^22 | 2,706.8 → 2,158.2 | 2,420.1 → 1,867.0 | 6.97 → 6.64 | 11.0–15.6 |
| `word-prime`, N=2^23 | 3,685.5 → 2,495.5 | 3,127.9 → 2,000.7 | 8.08 → 7.54 | 14.3–26.9 |

The first N=2^23 comparison gave 3,228.0 → 2,716.7 ms at load 11.6–16.9. Two
of its three final-side runs were disturbed: their untouched spans, Stage 2,
the range-image fold and the suffix relation preparation, ran up to twice their
base time. The repeated comparison in the table has no such outlier.

Root spans at N=2^22 `word-prime`, from the same final runs:

| Span (ms) | `f92e959ee` | Final |
| --- | ---: | ---: |
| Projections | 265.1 | 27.8 |
| Relation compile, A derivation excluded | 159.0 | 127.6 |
| Weight materialization (out of scope) | 242.8 | 245.5 |
| Stage 2 (out of scope) | 335.6 | 336.6 |
| Stage 3 | 127.6 | 27.0 |
| D commitment | 101.6 | 62.0 |
| Field switch | 53.2 | 36.2 |
| Ordinary suffix (complete minus root) | 282.3 | 291.8 |

At `f92e959ee` the A derivation before the relation compile had no span. It is
part of the root level, not of the relation compile row. Peak RSS falls in
every shape. This is consistent with proving no longer building its own copies
of A or the residual transforms; the attribution was not measured.

Setup derivation is now outside proving. The source commitment derives A once
(19 ms at N=2^22 after the parallel derivation), and proving reads that copy
through the projection witness. Typed-prefix preparation derives A again for
its own prefix commitment. Moving the commitment's derivation into setup
preparation would save those 19 ms per commitment, and the preparation already
pays for the derivation. It would also make the prepared setup hold A,
3 * 4,096 * 648 coefficients (about 127 MB as `i128`), for the lifetime of
the setup instead of the source handle. This pass does not make that move.

What remains at N=2^22 `word-prime`, about 1.3 s against the 0.4–0.8 s target:

- **Out of scope.** Weight materialization (245 ms) and Stage 2 (337 ms)
  together are about 45% of complete proving. The target is out of reach
  without them.
- **Ordinary suffix, about 290 ms.** It is unchanged and none of it is
  binary-source code. The breakdown above still holds. The first fold takes
  about 60%. The largest candidates are ordinary-protocol work, not
  evaluated here: the first fold's `RingRelationProver::prepare` (86 ms), the
  fold-grinding sampler (61 ms), and Stage 2 over the six folds (130 ms).
- **Direct A adjoint, 121 ms (1,445 ms CPU).** 4,096 columns times five
  128-bit transforms. Caching A's transforms would remove three of the five,
  but it costs 3 * 4,096 * 1,458 auxiliary elements (about 290 MB) of
  resident memory, which is outside the RSS budget.
- **Response fold, 67 ms (843 ms CPU).** This is already a bit-sliced
  carry-save kernel and runs in parallel across rows.
- **D commitment, 62 ms.** The D payload (25 ms) costs about 5.9 µs of CPU
  per setup entry. Its SHAKE256 setup streams probably dominate it; that is
  inferred from the per-entry cost, not profiled. The prime digits take 20 ms and the row partials
  take 17 ms serially. At `prime5` the D commitment is 243 ms, of which the D
  payload is 140 ms and the prime digits 101 ms. The prime-plane weight visit
  takes 362 ms serially there but is part of weight materialization.
- **Smaller serial spans.** Evaluation witness (20 ms), repacking (12 ms),
  witness import (11 ms) and response digits (9 ms).

#### Second pass on the Stage 3 weight base

This pass was merged (`a35eed1be`) onto `0ff6a96f7`. That base is `dev` at
`36a4e9bd5`, with the B/D setup offload (#318), the closed-form verifier
weights (#320) and structured Stage 2 (#321), plus a Stage 3 that runs on
block-structured weights and reads the committed prefix table. The base
supersedes two of this pass's changes:

- Stage 3 builds no A coefficients. It reads the table that prefix
  preparation committed, so the reused commitment A now serves only the
  relation compile.
- The closed-form weight families replace the ordinary root's term
  enumeration, so this pass's parallel ordinary weighing is gone. The base's
  families are faster: the relation compile at `word-prime-ordinary` is
  below.

Both sides parallelized the q setup matrix and the typed A scatter. The merge
keeps the base's q matrix and this pass's scatter, which fills every prefix
cell in one parallel pass without first zeroing the prefix serially. The
public digit commitments keep the small-digit accumulation and draw each
setup entry through the base's batched `fill_p`. Setup, commitment and proof
bytes are unchanged.

Base `0ff6a96f7` and head `a35eed1be` were proved alternately in isolated
processes with `BINARY_SOURCE_CATALOG_CACHE`. Each round ran the base, this
head and the third-pass head `8f575025c`, after one warm-up round. Each figure
is the median of three. All 60 runs, warm-ups included, reproduced their pins
and verified. 2^23 ran last. Verification is the online
`verify_complete_opening` span; it excludes setup deserialization. The host
was shared and loaded, so only figures from the same rounds are compared.

| N | Shape | Complete proving (ms) | Root level (ms) | Verification (ms) | Prover peak RSS (bytes) | Loads |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2^22 | `word` | 1,194.0 → 1,055.3 | 918.7 → 713.6 | 10.1 → 10.2 | 2,221,703,168 → 2,039,644,160 | 9.29–11.99 |
| 2^22 | `word-prime` | 2,269.4 → 1,769.5 | 1,612.5 → 1,080.9 | 14.3 → 17.9 | 3,095,003,136 → 3,036,069,888 | 14.50–22.84 |
| 2^22 | `word-prime-ordinary` | 1,769.0 → 1,350.3 | 1,251.5 → 897.7 | 17.9 → 17.0 | 3,297,558,528 → 3,141,369,856 | 15.46–19.54 |
| 2^22 | `prime5` | 2,270.8 → 1,746.0 | 1,559.5 → 1,028.5 | 11.2 → 11.2 | 5,414,666,240 → 5,118,984,192 | 10.29–16.40 |
| 2^23 | `word-prime` | 2,439.7 → 1,617.9 | 1,997.7 → 1,211.3 | 13.8 → 12.8 | 3,984,883,712 → 3,805,167,616 | 13.99–20.40 |

The gain comes from the spans this pass targeted. At 2^22 `word-prime` the
relation compile falls from 255.5 to 161.3 ms and the D commitment from 151.0
to 86.4 ms. At `prime5` the D commitment falls from 415.0 to 216.3 ms. At
`word-prime-ordinary` the relation compile falls from 167.5 to 140.3 ms.
Stage 2 and Stage 3 run the base's code in both executables. Stage 3 CPU
time agrees within 2.5%. Stage 2 uses 2.5–6.3% more CPU time in this head at
every shape. The cause is not measured. Verification time is within noise.

### Third prover pass

A third byte-preserving pass starts from `7137782e9`, which merges the second
prover pass (`e7b6c127a`) and structured Stage 2 (`4d536c84e`) on
`f92e959ee`. The profile executable adds one field to the driver's span
records: the names of each span's enclosing spans, so the span tree below is
measured rather than inferred. Spans opened on worker threads have no parent.

The merged base was proved three times at N=2^22 `word-prime` in isolated
processes with `BINARY_SOURCE_CATALOG_CACHE`. Every run reproduced its pin. An
earlier set of three runs at a one-minute load of 17.9–19.1 took 1,732, 2,943
and 2,587 ms with unchanged CPU time (7.8–8.0 s); it is disturbed and not used.
The runs below had one-minute loads of 10.9–12.2. Logs are
`$CARGO_TARGET_DIR/p3/logs/c0-22-word-prime-{4,5,6}.txt`.

| Span (ms, median of 3) | Wall | Process CPU |
| --- | ---: | ---: |
| Complete proving | 1,000.8 | 8,415.6 |
| › binary root level | 720.1 | 5,779.1 |
| ›› Stage 2 | 258.4 | 1,981.4 |
| ››› weight profiles | 161.1 | 874.9 |
| ›››› authentication (2 calls) | 94.6 | 657.5 |
| ›››› parity rows | 23.2 | 92.2 |
| ›››› prime planes | 18.0 | 22.1 |
| ›››› direct A | 13.3 | 13.3 |
| ›››› image | 11.0 | 85.6 |
| ›› relation compile | 134.8 | 1,431.6 |
| ››› direct A adjoint | 128.0 | 1,424.9 |
| ›› response build | 117.8 | 897.3 |
| ››› response fold | 66.8 | 787.3 |
| ››› evaluation witness | 20.1 | 54.0 |
| ››› repack | 11.8 | 11.8 |
| ››› witness import | 10.0 | 36.1 |
| ››› response digits | 8.5 | 8.5 |
| ›› D commitment | 65.2 | 512.6 |
| ››› D payload | 27.3 | 328.9 |
| ››› prime digits | 20.9 | 160.3 |
| ››› row partials | 17.0 | 17.0 |
| ›› Stage 3 | 39.5 | 258.8 |
| ›› field switch | 37.9 | 252.0 |
| ›› projections | 28.0 | 117.9 |
| ›› Stage 1 | 19.1 | 182.7 |
| ›› outgoing commitment | 18.0 | 161.6 |
| › ordinary suffix (complete minus root) | 280.7 | 2,636.5 |
| ›› Stage 2 sumchecks (6) | 97.8 | 889.1 |
| ›› `RingRelationProver::prepare` (6) | 74.1 | 682.2 |
| ››› fold grinding (6) | 58.1 | 559.0 |
| ›››› sampling (6) | 37.9 | 388.2 |
| ››› opening rows (6) | 15.8 | 125.8 |
| ›› Stage 1 sumchecks (6) | 51.5 | 499.7 |
| ›› successor commitments (6) | 26.7 | 221.6 |
| ›› coefficient-packing partials | 19.6 | 192.2 |
| Prover peak RSS bytes | 2,386,247,680 | |

The Stage 2 sumcheck spans (`RelationRangeImageProver::*`) are shared by the
root and the suffix and are reported under the span that encloses them. The
suffix successor commitments are `commit_w_level` minus the root's outgoing
commitment.

Five spans run serially or nearly so: the prime-plane and direct-A profile
emission, the row partials, the repack and the response digits. Together they
take 69 ms. The authentication profile build weighs its columns in parallel,
but streams every weight into the profile rows serially, one address at a
time; it runs at 7 cores of 16.

Candidate changes, with the expected gain at N=2^22 `word-prime`:

| Candidate | Spans reduced | Expected gain (ms) |
| --- | --- | ---: |
| Direct A adjoint as trace-form ring products over limb primes | direct A adjoint | 90–110 |
| Parallel authentication rows, one residue kernel per setup column | authentication profile | 50–70 |
| Parallel profile emission for the parity, prime and direct-A families | weight profiles | 25–40 |
| Parallel D row partials; profile the D payload's SHAKE streams against its arithmetic | D commitment | 15–25 |
| Parallel repack, response digits and evaluation witness | response build | 15–25 |
| Ordinary fold-grinding sampler and the first fold's preparation | `RingRelationProver::prepare` | 10–30 |
| Response fold | response fold | 0–20 |

The direct A adjoint is the largest. It computes
`h_j = sum_i adj(rho_i, A_ij)` for 4,096 columns through five 128-bit
transforms of length 1,458 per column, about 350 µs of CPU each. Multiplication
by `A` in `Z[Y]/(Phi_1944)` is self-adjoint under the trace form, so
`adj(rho, A) = G (A * G^-1 rho)`, where `G` is the trace form's Gram matrix in
the power basis. `Tr(Y^n)` vanishes unless 324 divides `n`, so `G` has two
nonzero entries per row, and `G^-1 rho` is computed once per functional. The
products `A_ij * w` then run in the 648-point trinomial limb domains that the
residual rows already use, and are recovered exactly by Chinese remaindering.
No transform of A is kept.

The ordinary candidates are ordinary-protocol code. Any change there keeps
the fold-grinding acceptance rule and nonce order, and lands in its own
commit.

#### Third-pass changes

Every change below preserves setup, commitment and proof bytes; each one
reproduced every pin it was measured on. None touches ordinary-protocol code.
Each commit message records its median of three interleaved runs against its
parent at 2^22 `word-prime`, with the one-minute loads.

| Commit | Change | Span: before → after (ms) |
| --- | --- | --- |
| `ca3ccb522` | Direct A adjoint as trace-form ring products | direct A adjoint 129.5 → 33.9 |
| `866a1caa5` | Authentication profile rows filled by lane in parallel | authentication emission 70.3 → 35.3 |
| `dea319a3d` | Recomposed profiles filled one element per task | direct A 13.5 → 4.0, image 10.3 → 7.8, prime 18.0 → 2.6 |
| `84d3091dc` | Parity row weights filled by response lane | parity rows 20.5 → 7.8 |
| `eaf31d0f6` | D row partials one column per task | row partials 22.9 → 2.3 |
| `fb09842a1` | Parallel repack and response digits | repack 11.5 → 3.1, response digits 8.1 → 1.5 |
| `cec527e88` | Evaluation-witness columns reduced in parallel | evaluation witness 18.5 → 11.3 |

The direct A adjoint uses three limb primes, 67,091,329, 134,217,649 and
268,433,353. Each canonical `w = G'^-1 rho` is cut into three unsigned 43-bit
limbs, which are reduced and transformed once per functional. Every column then
transforms `A_ij` once per prime, accumulates the pointwise products of all its
rows and limbs, and inverts once per limb. Garner reconstruction is exact
because `2 * 3 * rank * 648 * bound * 2^43` stays below the primes' product.
The path applies only when the bound is at most half the smallest prime;
other inputs keep the auxiliary transforms.

`ProfileBuilder` gained two parallel entry points. `add_lanes` gives every
lane's blocks their own rows in address order and hands runs of 32 lanes to
parallel tasks, which serve the authentication and parity families.
`add_shared` gives every element its shared rows and lets one task add each
element's whole profile, which serves the direct A, image and prime families.
Authentication also runs its residue kernel once per setup column on the
batched combination of the setup rows, since the kernel is linear.

The D payload was profiled before any change to it. For 8,192 columns at rank 2
on one thread, sampling the setup coefficients from their SHAKE streams takes
20.6 ms of the 73.3 ms commitment, so the integer negacyclic digit products are
about 72% and SHAKE about 28%. Rewriting the products as 16-bit limbs in i32
lanes brought the commitment to 64.7 ms, a 12% gain. That is worth about 12 ms
at `prime5` and 2.5 ms at `word-prime`, so the rewrite was not kept.

#### Third-pass results

Base `7137782e9` and head `cec527e88` were proved alternately, three times
each per configuration, in isolated processes with a cached catalog. All 30
runs reproduced their pins. 2^23 ran last.

| N | Shape | Complete proving (ms) | Root level (ms) | Process CPU (ms) | Peak RSS (bytes) | Loads |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2^22 | `word` | 745.1 → 595.1 | 501.7 → 343.7 | 7,540 → 6,219 | 2,290,221,056 → 2,608,889,856 | 5.17–6.52 |
| 2^22 | `word-prime` | 822.3 → 636.1 | 584.0 → 398.8 | 8,566 → 7,143 | 2,370,994,176 → 2,704,932,864 | 6.52–8.27 |
| 2^22 | `word-prime-ordinary` | 993.3 → 756.4 | 748.2 → 512.2 | 9,471 → 8,084 | 2,875,932,672 → 2,932,752,384 | 7.31–8.08 |
| 2^22 | `prime5` | 1,094.6 → 822.4 | 846.3 → 577.7 | 11,498 → 9,979 | 2,538,864,640 → 2,839,494,656 | 4.06–7.54 |
| 2^23 | `word-prime` | 1,392.5 → 1,221.6 | 1,013.6 → 784.4 | 13,946 → 11,436 | 4,158,078,976 → 5,080,317,952 | 4.55–9.19 |

The base column is faster than the opening profile above because the machine
was less loaded; only interleaved pairs are compared.

Root spans at 2^22 `word-prime`, from the same runs:

| Span (ms, median of 3) | Base | Head |
| --- | ---: | ---: |
| Stage 2 | 205.0 | 126.5 |
| › weight profiles | 129.8 | 54.3 |
| ›› authentication | 70.1 | 35.2 |
| ›› parity rows | 20.5 | 7.5 |
| ›› prime planes | 17.5 | 1.7 |
| ›› direct A | 13.1 | 3.9 |
| ›› image | 8.3 | 5.7 |
| relation compile | 104.8 | 34.1 |
| › direct A adjoint | 98.2 | 27.1 |
| response build | 104.1 | 83.4 |
| › response fold | 56.3 | 57.8 |
| › evaluation witness | 18.4 | 11.3 |
| › repack | 11.4 | 3.1 |
| › response digits | 8.4 | 1.4 |
| D commitment | 54.8 | 39.3 |
| › prime digits | 16.3 | 16.1 |
| › row partials | 16.1 | 1.8 |
| Stage 3 | 31.8 | 32.6 |
| field switch | 31.1 | 31.9 |
| projections | 24.2 | 24.3 |

The ordinary suffix is unchanged: its Stage 2 sumchecks take 85.5 ms, its
`RingRelationProver::prepare` 58.3 ms, of which fold grinding is 45.9 ms, and
its Stage 1 sumchecks 48.0 ms.

Peak RSS rises 13–14% at `word`, `word-prime` and `prime5`, 2% at
`word-prime-ordinary` and 22% at 2^23. Live heap does not: the prove-phase
live-heap peak is identical in base and head, 1,516, 1,565, 1,996 and 2,009 MB
for the four 2^22 shapes and 2,961 MB at 2^23. The difference is freed memory
that the system allocator keeps resident in its large-block cache. With
`MallocLargeCache=0`, peak RSS at `word-prime` falls from 1,687,420,928 to
1,663,991,808 bytes at 2^22 and from 3,142,221,824 to 3,101,229,056 bytes at
2^23. In the base, the authentication emission allocated and freed a batch
of column weights every 4,096 columns. That churn happened to evict the
cache, so resident memory dropped from 2,048 to 1,430 MB during the emission.
The lane fill allocates nothing large, so freed buffers of earlier stages stay
cached and later stages map fresh pages. The prover keeps no more live data
than before; the RSS budget is exceeded only by allocator residency.

What remains at 2^22 `word-prime` is about 400 ms in the root and 237 ms in
the suffix. No root span above 10 ms is still serial. The largest are:

- the response fold at 58 ms, an integer fold of every carrier that already
  runs at 14 cores of 16;
- the authentication emission at 35 ms, which reads every setup column's
  entries from per-entry SHAKE streams;
- the Stage 2 sumcheck rounds after the profiles, at about 72 ms;
- the D commitment at 39 ms: payload 21 ms and prime digits 16 ms.

At `prime5` the D commitment is 179 ms, of which the payload is 100 ms and the
prime digits 78 ms; they already run at 15 and 10 cores. The suffix's six ordinary
folds run at 12–14 cores. A further gain there needs a different algorithm,
not more threads, and the fold-grinding acceptance rule and nonce order fix
their sampling work.

A protocol change could buy the following; none is made here:

- The D commitment, and the B commitment at commit time, multiply every
  setup entry, a 64-coefficient polynomial over `P = 2^128 - 275`, by its
  digit chunk as integers. Arithmetic is about 72% of the D payload. A commitment modulus with a
  64-point negacyclic transform, or a limb-prime path like the direct A
  adjoint's, would cut most of that share: at most about 70 ms of the
  `prime5` payload and 15 ms at `word-prime`.
- Setup entries come from one SHAKE stream each. A setup derived from fewer,
  longer streams would cut the SHAKE share of the D payload, about 28%, and
  part of the authentication emission.
- The suffix costs about 40 ms per ordinary fold on average over six folds.
  A schedule with one fewer fold would save about that much, at the cost of
  larger proofs per level.

### Pre-v8 dev-protocol results

These results and all their timings predate the v8 merge. Their proofs
reproduce the retired pre-v8 pins, not the current ones.

The dev-protocol rows come from the same tree and campaign, with one
dev-protocol executable (SHA-256
`bc1103d79e2bea9235cc5cad763c36222e20a7840eb0502dff81679a354ec595`) and the
same procedure. Every run verified with matching caller transcript
continuation and forbidden online A reads. The logs are
`$CARGO_TARGET_DIR/bd-offload/<log>-<shape>-dev.{prove,verify1,verify2,verify3}.txt`.
The recursive output chains equal the current-protocol chains above. The
verification rows and verifier peak RSS below predate the verifier speedups
and were not remeasured. The dev protocol runs the same binary root
verifier, so similar gains are expected but unmeasured. The `uptime` load
triples are:

| N | Shape | Before proving | After proving | After the third verification |
| --- | --- | --- | --- | --- |
| 2^22 | word | 5.82/6.16/6.38 | 5.37/6.00/6.31 | 5.37/6.00/6.31 |
| 2^22 | word-prime | 5.37/6.00/6.31 | 6.27/6.11/6.33 | 6.27/6.11/6.33 |
| 2^22 | word-prime-ordinary | 7.06/6.38/6.41 | 6.72/6.37/6.41 | 6.74/6.38/6.41 |
| 2^22 | prime5 | 6.27/6.11/6.33 | 7.06/6.38/6.41 | 7.06/6.38/6.41 |
| 2^23 | word | 6.74/6.38/6.41 | 13.67/8.47/7.19 | 13.67/8.47/7.19 |
| 2^23 | word-prime | 13.67/8.47/7.19 | 10.41/8.46/7.27 | 10.41/8.46/7.27 |
| 2^23 | word-prime-ordinary | 15.33/9.75/7.82 | 10.66/9.28/7.74 | 10.66/9.28/7.74 |
| 2^23 | prime5 | 10.41/8.46/7.27 | 15.33/9.75/7.82 | 15.33/9.75/7.82 |

| N | Shape | Proof bytes | Atoms | Prover peak RSS bytes | Verifier peak RSS bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 2^22 | word | 82,786 | 25,610 | 5,189,074,944 | 75,251,712 |
| 2^22 | word-prime | 84,423 | 25,735 | 6,067,929,088 | 155,107,328 |
| 2^22 | word-prime-ordinary | 84,394 | 25,706 | 6,231,687,168 | 643,334,144 |
| 2^22 | prime5 | 89,212 | 26,143 | 10,006,872,064 | 128,319,488 |
| 2^23 | word | 84,580 | 25,746 | 10,179,739,648 | 155,353,088 |
| 2^23 | word-prime | 84,576 | 25,742 | 10,190,028,800 | 157,106,176 |
| 2^23 | word-prime-ordinary | 84,535 | 25,700 | 10,358,407,168 | 1,047,003,136 |
| 2^23 | prime5 | 89,248 | 26,179 | 10,594,435,072 | 133,496,832 |

| N | Shape | Field switch | Projections | Stage 1 | Stage 2 | Stage 3 | Physical L2 | Other |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | word | 2,993/173 | 84/7 | 11,043/858 | 7,217/587 | 824/74 | 4,293/318 | 56,332/23,593 |
| 2^22 | word-prime | 2,993/173 | 84/7 | 14,357/1,142 | 7,119/579 | 857/77 | 2,754/204 | 56,259/23,553 |
| 2^22 | word-prime-ordinary | 2,993/173 | 84/7 | 14,357/1,142 | 7,119/579 | 857/77 | 2,754/204 | 56,230/23,524 |
| 2^22 | prime5 | 0/0 | 84/7 | 20,891/1,646 | 7,168/583 | 890/80 | 1,377/102 | 58,802/23,725 |
| 2^23 | word | 3,035/175 | 84/7 | 14,422/1,147 | 7,168/583 | 857/77 | 2,754/204 | 56,260/23,553 |
| 2^23 | word-prime | 3,035/175 | 84/7 | 14,422/1,147 | 7,168/583 | 857/77 | 2,754/204 | 56,256/23,549 |
| 2^23 | word-prime-ordinary | 3,035/175 | 84/7 | 14,422/1,147 | 7,168/583 | 857/77 | 2,754/204 | 56,215/23,507 |
| 2^23 | prime5 | 0/0 | 84/7 | 20,891/1,646 | 7,168/583 | 890/80 | 1,377/102 | 58,838/23,761 |

N=2^22:

| Phase | word ns | word-prime ns | word-prime-ordinary ns | prime5 ns |
| --- | ---: | ---: | ---: | ---: |
| Catalog planning | 26,444,312,458 | 28,269,842,667 | 10,192,390,708 | 29,025,299,917 |
| Cold prover setup | 42,911,125 | 85,848,291 | 73,045,084 | 164,313,584 |
| Independent fixture evaluation | 17,362,666 | 1,664,382,125 | 1,624,482,625 | 8,243,331,917 |
| Binary commitment | 1,655,460,167 | 1,652,722,625 | 1,638,416,791 | 1,652,331,333 |
| Ordinary commitment | absent | absent | 12,349,167 | absent |
| Cold prefix preparation | 208,787,583 | 463,395,042 | 472,135,209 | 377,798,500 |
| Cold verifier setup | 193,442,125 | 429,606,917 | 466,694,875 | 349,797,625 |
| Fixture serialization | 50,071,583 | 110,349,291 | 96,665,917 | 59,144,583 |
| Fixture setup deserialization | 67,360,417 | 169,222,750 | 166,987,791 | 132,926,291 |
| Parity row convolution | 11,354,000 | 10,744,041 | 10,296,334 | absent |
| Complete proving | 11,301,030,375 | 12,329,756,500 | 12,677,745,292 | 16,257,100,542 |
| Complete verification | 14,931,209 | 16,311,917 | 122,121,209 | 14,824,334 |

N=2^23:

| Phase | word ns | word-prime ns | word-prime-ordinary ns | prime5 ns |
| --- | ---: | ---: | ---: | ---: |
| Catalog planning | 31,553,362,042 | 32,214,784,167 | 11,380,303,292 | 31,845,983,042 |
| Cold prover setup | 75,469,459 | 86,793,708 | 101,245,292 | 150,365,041 |
| Independent fixture evaluation | 33,506,209 | 3,389,147,625 | 3,386,491,792 | 16,302,570,833 |
| Binary commitment | 1,813,536,791 | 1,909,552,875 | 1,896,395,125 | 1,828,071,417 |
| Ordinary commitment | absent | absent | 16,915,375 | absent |
| Cold prefix preparation | 515,129,375 | 599,981,708 | 599,618,917 | 480,262,750 |
| Cold verifier setup | 507,194,291 | 571,697,333 | 566,880,416 | 443,222,917 |
| Fixture serialization | 142,648,417 | 96,069,250 | 122,213,833 | 75,522,042 |
| Fixture setup deserialization | 172,382,000 | 172,799,917 | 171,407,667 | 139,297,125 |
| Parity row convolution | 19,338,458 | 19,834,958 | 19,906,125 | absent |
| Complete proving | 21,678,343,250 | 23,120,246,417 | 23,485,572,750 | 26,473,936,250 |
| Complete verification | 16,207,833 | 19,281,709 | 218,095,959 | 16,759,625 |

### Canonical layout and native schedule reconciliation

The current-protocol planner and completed full-size openings reconcile the
following canonical repacking counts with `price_binary_root_planes`. Natural digits exclude
virtual coefficient tails; physical cells
include those tails but exclude final power-of-two padding. These counts
agree with the design's section 12.1; its five-plane row additionally includes
a word claim, unlike the `prime5` harness shape.

| N | Shape | Natural digits | Physical cells | Minimum domain | Padding cells |
| --- | --- | ---: | ---: | ---: | ---: |
| 2^22 | word | 14,642,046 | 23,138,048 | 33,554,432 | 18,912,386 |
| 2^22 | word-prime | 16,425,342 | 25,956,096 | 33,554,432 | 17,129,090 |
| 2^22 | word-prime-ordinary | 21,393,790 | 30,924,544 | 33,554,432 | 12,160,642 |
| 2^22 | prime5 | 23,514,624 | 37,158,912 | 67,108,864 | 43,594,240 |
| 2^23 | word | 25,258,878 | 39,915,264 | 67,108,864 | 41,849,986 |
| 2^23 | word-prime | 27,042,174 | 42,733,312 | 67,108,864 | 40,066,690 |
| 2^23 | word-prime-ordinary | 36,172,158 | 51,863,296 | 67,108,864 | 30,936,706 |
| 2^23 | prime5 | 34,131,456 | 53,936,128 | 67,108,864 | 32,977,408 |

The ordinary group contributes N source cells and 704,512 image digit cells.
Its two compression segments contribute 65,536/4,096 cells at N=2^22 and
32,768/4,096 at N=2^23 under the selected ordinary profiles. These are minimum
binary witness domains; a selected ordinary carrier schedule may reserve a
larger domain and must report it separately.

At source commit `4fc40d2db14cdc670161f9c310a20bf9e994c148`, the benchmark
harness admits an explicit native-root transition using the canonical physical
witness length and typed prefix length. The ordinary suffix planner starts
from those exact lengths. Catalog admission validates that geometry, and the
binary verifier consumes the same authenticated successor input length. This
replaces an ordinary-carrier capacity search, which found no carrier at
N=2^22 (see the appendix). Smoke runs keep their original ordinary carrier
schedules, so their pinned proof bytes are unchanged. Both measured
`word-prime` schedules have six recursive folds and eight total fold levels.
Their first successor uses degree 512 and the later ordinary folds use degree
64. Full selected profiles are printed in each measurement log.

The mixed ordinary source materializer also accepts the canonical B slice
geometry rather than requiring one block per slice. Its T digits remain in
global block/A-row/subcolumn/digit/coefficient order. Public B weights select
canonical block ranges and use local block indices inside each slice. Frozen
profile validation still authenticates the physical B width; the complete T
length comes from the canonical logical B width. A cheap regression checks
four blocks with one or two slices, degree-128 A and degree-64 B, complete
A/B/F/H identities, subcolumn ordering and malformed-width rejection.

The typed A tensor corrects section 12.3's proposed domains. Its
independent coefficient and rank axes require `m*1024*4`, rather than rounding
the flat live count `m*648*3`. The A tensor alone needs **16,777,216 (2^24)**
and **33,554,432 (2^25)** cells, twice the proposed 2^23 and 2^24. The regression
`full_size_typed_prefix_requires_padded_rank_and_coefficient_axes` checks that
the canonical product accepts these sizes and rejects the smaller domains
without allocating the full prefix. `BinaryAPrefixLen` is the only owner of
the A tensor's sizing. `BinaryTypedPrefix` places the B/D setup around it and
owns the prefix length: 2^24, 2^25 or 2^26 at N=2^22 and 2^25 or 2^26 at
N=2^23, depending on the shape ([Typed prefix layout](#typed-prefix-layout)
and the setup offload note, §4). Setup coefficients, the
Stage 3 weights, `price_binary_root_planes` and the harness's native
transition all read it.

### Reconciliation with the design's cost budgets

Section 12.3's serialized `word-prime` estimates were 89,033 and 89,525 bytes.
The measured current-protocol proofs before the v8 merge are 89,464 and
89,658 bytes, differences of **+431** and **+133** bytes. The estimates used
the smaller proposed A-prefix domains and planner-estimated ordinary suffix
bytes; the measured proofs use the corrected rank/coefficient padding, the
typed prefix that also holds the B/D setup, and the exact native-root
transition. Their similar total sizes do not validate the original suffix
estimate or its individual message assumptions. Section 12.1's five-plane row
also includes a word claim, whereas the harness's `prime5` shape does not.

The pre-v8 dev-protocol `word-prime` proofs were 84,423 and 84,576 bytes:
differences of **-4,610** and **-4,949** bytes from those design estimates, and
5,041 and 5,082 bytes below the corresponding pre-v8 current-protocol proofs.
That dev protocol omitted the last live e and t blocks through terminal
recomputation. This is a grammar distinction, not an optimization that
preserves proof bytes. The v8 protocol includes it, so the v8 proofs are
comparable with the pre-v8 dev-protocol proofs, not the pre-v8 current ones.
After the v8 merge, the `word-prime` proofs are 84,392 and 84,590 bytes.

The timings in this subsection predate the v8 merge.
Section 12.4's prover targets of 0.4–0.8 seconds at N=2^22 and 0.8–1.6 seconds
at N=2^23 are not achieved. In the offload campaign, without the prover-speed
pass, final `word-prime` complete proving took 12,357,436,292 ns and
22,819,519,709 ns under the load triples above. The prover-speed pass,
measured without the offload, brought it to 1,737,754,208 ns and
2,927,935,667 ns, the prover span profile medians: 2.2 and 1.8 times the
upper ends of the targets.
Section 12.5's verifier targets are 5–10 ms at N=2^22 and 6–13 ms at
N=2^23. After the verifier speedups, parallel `word-prime` verification takes
12,408,375 ns at N=2^23, within its target, and 11,340,417 ns at N=2^22,
1.3 ms above it, under one-minute loads of 12.12–12.75. Single-threaded it
takes 25,839,667 ns and 26,771,375 ns, so neither target is met there. The binary
root takes 4.3 and 5.5 ms parallel (`binary_root_verify`). The ordinary
suffix's Stage 2 verification takes 5.3 and 5.1 ms parallel and 19.3 and
19.2 ms single-threaded, most of it the direct setup scan.

Dev-protocol `word-prime` proving took 12,329,756,500 ns and 23,120,246,417 ns
in the offload campaign, and 1,674,145,125 ns and 2,910,122,000 ns in the
prover-speed medians; verification took 16,311,917 ns and 19,281,709 ns
before the verifier speedups, under its recorded loads.

### Verifier and kernel cost relative to the design

The timings in this subsection predate the v8 merge.
The verifier draws one batching coefficient per root row and none per padding
cell. At N=2^22 `word-prime` the median τ₁ draw takes 2,250 ns, against
861,793,834 ns for the 17,129,090 per-cell padding challenges it drew before
the deletion (appendix). The mixed shape adds one row per digit of the two
ordinary compression segments (`w(w+1)=0`): 69,632 coefficients at N=2^22 and
36,864 at N=2^23, which take 3.3 and 1.8 ms single-threaded
(`binary_root_tau1_draw`) and are the largest binary-root cost that a
byte-preserving change cannot remove. The design's section 8.2 values 856,064 and 1,712,128
count additions in a proposed shared-subset direct-weight kernel, not the
complete verifier. The canonical A accessor remains absent from online
verification.

Every public root weight is a structured family. One construction
serves both sides: the verifier evaluates the family's multilinear extension
at the Stage 2 point, and the prover streams the same field elements into its
dense table. No family stores one term per address or scans the outgoing
domain. A test-only copy of the previous flat construction checks the
evaluation and the dense table on nineteen small shapes. Sixteen use the
end-to-end geometry of 8 padded carriers and 2 columns, which has one response
quad. Without the mixed ordinary group they cover the word claim with zero,
one, two or five prime planes, prime-only claims with two or three planes, word
widths 128 and 8, one and two owners, and extra zero rings. With the mixed
group they cover zero, one, two or five planes beside the word claim, and one,
two or five prime-only planes. Three more use 32 carriers and 2 columns, or 64
carriers and 4 columns, which have four response quads. No tested shape
reaches a production-size geometry.

- **Image.** One coefficient adjoint per owner and A row is applied to the
  low equality block, followed by each column's challenge support of at most
  46 entries. Because the input is an equality table, the adjoint runs in
  closed form (`TrinomialAdjoint::apply_equality`); the typed A product uses
  the same call.
- **Prime planes.** Four signed power sums of the low block and one
  shifted-remainder kernel per 162-coefficient low-block slot serve every
  plane. Planes of one polynomial read the same response elements and the same
  challenge supports, so their relation rows combine before the high factors
  and the support dot products (signed additions of kernel entries) are taken
  once per polynomial, column and slot.
- **Parity rows.** Rows are answered by Four Russians subset sums.
- **Terminal partial columns.** The partial-column rows of one owner form one
  family (`PartialColumnWeights`). Per column the verifier adds the bits and
  the ±1 challenge coefficients into one multiplier and reads one
  shifted-remainder pattern per low offset, instead of one explicit term per
  column coefficient.
- **Ordinary root.** Each ordinary identity repeats one public pattern (an A
  row in every block, a B row in every slice, a compression map, a
  recomposition at every bit), kept once as `TranslatedLanes`. For the A rows,
  B rows and maps the lane is a residue kernel of a setup ring, so the
  verifier evaluates it from the ring and the low block through split
  windows (`ResidueKernelPoint`) without expanding the kernel. The scalar
  source evaluation is one offset-equality sum.
- **B and D.** The setup-dependent B/D weights move to the typed prefix and
  Stage 3. The verifier evaluates their terminal weight from the segment
  geometry and the piece placement, with no setup read
  ([binary-source-setup-offload.md](binary-source-setup-offload.md), §7). The
  B/D claims stay in Stage 2.

Before the offload, the verifier derived 2 × (62,208 + 28,512) = 181,440
SHAKE256 ring entries at N=2^22 `word-prime` and evaluated one residue kernel
per setup column. After the byte-preserving speedups that span
(`root_weights_evaluate_authentication`) still took 28.0 ms of 43.2 ms; at
N=2^22 `prime5` it took 58.3 of 72.3 ms. The offload removes it (comparison
table in the appendix). The mixed shape's ordinary root still derives its
setup matrix during verification. In the single-threaded median runs that
derivation (`derive_public_matrix_prefix`) takes 2.4 ms at N=2^22 and 4.8 ms
at N=2^23, of 3.9 and 7.1 ms for `binary_root_weights_ordinary`; parallel it
takes 0.6 and 0.9 ms. Before the verifier speedups that span was dominated by
building one flat term per ordinary coefficient weight: 85,915,708 ns at
N=2^22 and 160,702,291 ns at N=2^23 in the offload campaign.

The parity rows follow sections 5.3 and 8.2 on the verifier side. By the Hankel
symmetry of the shifted remainders, row `r` contributes the subset sum of
`P_k = SRE(((-1)^c low[4c+k])_c, alpha)` over its set bits, where
`k = r mod 4`. The verifier builds `4 × 21 × 256` tables over 8-bit chunks.
The final chunk holds only bits 160 and 161, so building the tables takes
21,168 additions. Each row then costs 21 lookup additions, and each response
quad one multiplication by its high equality factor. The prover still needs
every row's 162-coefficient kernel. It keeps one `shifted_remainder_evaluations`
call per row: 497 field multiplication operators (10 for exponentiation to 81,
one for the modulus evaluation, 162 for Horner evaluation and 324 for the shift
recurrence), plus 162 for the batch scaling. The same call executes 407 binary
field addition/subtraction operators, excluding 81 unary sign negations.
Materializing the dense table then adds each of the row's 162 weights into its
address, so a complete parity row costs 569 additions/subtractions. The same
subset tables would cost the prover 21 vector additions of 162 entries per row.
The prover kernel columns below count kernel generation and batch scaling only,
excluding dense accumulation; the last prover column adds it. All counts
exclude equality-weight generation, field conversion, column kernels, carry
kernels and unary negations.

| N | Row weights M | Verifier multiplications | Verifier additions | Prover kernel multiplications | Prover kernel additions/subtractions | Prover additions/subtractions with accumulation | Proposed subset additions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | 16,384 | 6,085 | 370,956 | 10,797,056 | 6,668,288 | 9,322,496 | 856,064 |
| 2^23 | 32,768 | 10,181 | 719,116 | 21,594,112 | 13,336,576 | 18,644,992 | 1,712,128 |

These count executed Rust arithmetic operators, including multiplication by
zero or a sign; they are not measured machine instructions. The verifier
columns include its four shifted-remainder calls (1,988 multiplications and
1,628 additions). The proposed subset column does not include its public
coefficient-kernel work, so it is not an end-to-end operation total.

The design's rough 59,049-multiplication adjoint estimate also does not describe
the current kernels. As detailed in the typed A section above, the general
two-convolution adjoint uses 154,872 leaf field multiplications per rank row,
or 464,616 for rank three. The verifier only applies the adjoint to equality
tables, once per owner and A row for the image and once per rank row for the
A product, and uses the closed form: 18,505 multiplications per application
at D = 648, or 55,515 for rank three. Surrounding contractions add further
operations. The prover expands each column by at most 46 signed shifts of the
extended row.

### Memory

Before the v8 merge, the N=2^23 `word-prime` prover peak in the offload
campaign was 10,190,553,088 bytes in the current protocol and 10,190,028,800
bytes in the dev protocol. The largest measured peak, N=2^23 dev `prime5`, was
10,594,435,072 bytes. Against the tables before the B/D setup offload
(appendix), the prover peak rose in every shape. Where the prefix domain does
not grow, it rose by 0.14 GB (N=2^22 `word`) to 0.69 GB (N=2^23
`word-prime-ordinary`, from 9,517,465,600 to 10,211,393,536 bytes). The
offload adds a dense `16 * 2^k'`-byte Stage 3 weight in every shape. Removing
Stage 3's redundant full-length index table left the process peak nearly
unchanged. Whole-process peak RSS does not say which phase sets the peak, so
that attribution is unmeasured. The setup offload note, §8, separates prefix
growth, dense weights and backend state. Before the v8 merge, the largest
verifier peak after the verifier speedups was 178,913,280 bytes (N=2^23
`word-prime-ordinary`, parallel), down from 1,047,445,504 bytes before them.

The prover-speed pass was measured without the offload. Its N=2^23
`word-prime` medians were 8,051,064,832 bytes in the current protocol and
8,057,176,064 bytes in the dev protocol, against 8,462,876,672 bytes for the
baseline executable under the same conditions. After the v8 merge and before
the offload, the median was 8,056,356,864 bytes; the v8 merge's single run
without the prover-speed pass, which planned its catalog in the proving
process, peaked at 9,986,932,736 bytes. With the offload after the v8 merge, the
median was 8,595,046,400 bytes. The largest peak after the v8 merge was
10,705,600,512 bytes, at N=2^23 `prime5` in the first campaign, which
planned its catalog in the proving process. All are below one sixth of the
68,719,476,736-byte host. Peak RSS includes cold preparation in each isolated
process. These values establish memory fit for this fixture; they do
not certify another source count, shape or parameter choice.

The packed importer keeps binary sources within the importer's unchanged cap.
It retains column-contiguous `u128` carriers using the published
`kernel_carrier_index` permutation. Storage is exactly **67,108,864 bytes** at
N=2^22 and **134,217,728 bytes** at N=2^23 per polynomial, within the unchanged
134,217,728-byte budget. Immutable source, A matrix, images and image digits
are shared through `Arc` when root preparation clones its private ownership.

A commitment now uses `PreparedLimbCommitMatrix` and
`commit_binary_clear_limb_prepared`; response folding uses the existing
bit-sliced carry-save kernel. A checked sparse-challenge adapter preserves
support order, deterministic signs and nonce selection. Prime planes use the
existing shared eight-row subset-sum kernel through an explicit weighted-bit
arithmetic boundary, without expanding signed source rings. Differential
regressions compare image, response, residual, prime-plane and binary partial
values with the previous arithmetic for multiple polynomials, narrow word
widths and padding.

## Verifier protocol-change candidates

The byte-preserving speedups leave three costs that need a change to the
proof or the transcript. None is implemented. Expected gains are upper
bounds read from the single-threaded and parallel median spans of the final
verification runs. They omit the cost of whatever replaces the removed work.

### Structured negative-bit batch

**Cost.** The mixed shape batches one τ₁ coefficient per digit of the two
ordinary compression segments, 69,632 at N=2^22 and 36,864 at N=2^23. Each is
an independent grinded extension challenge, so `binary_root_tau1_draw` takes
3.3 and 1.8 ms in both thread modes. That is 9% and 5% of single-threaded
mixed verification and 18% and 9% of parallel.

**Change.** Draw the negative-bit range as an equality tensor
`eq(τ_neg, i)` over `ceil(log2(rows))` = 17 or 16 fresh coordinates, or as
powers of one challenge, instead of one coefficient per row. The rest of τ₁
is unchanged. The field-message count stays the same, but every later
challenge moves, so the proof bytes, the transcript state and the digest pins
change. Serialized grinding nonces are variable-length, so the proof length
can change by a few bytes, and repricing the batch can change the grinding
schedule.
The negative-bit weight is already evaluated through its support's
multilinear extension (`negative_binary_support_mle`, 0.02 ms); with tensor
coefficients it has a closed form.

**Expected gain.** About 3.3 ms at N=2^22 and 1.8 ms at N=2^23 for the mixed
shape, in both thread modes. Other shapes have no negative-bit rows.

**Soundness question.** Independent coefficients let a nonzero row residual
survive the batch with probability `1/P`. Tensor coefficients give a nonzero
multilinear polynomial in `τ_neg`, so the error is at most `17/P` or `16/P`;
powers give at most `(rows - 1)/P`. The question is whether the root's
soundness argument uses the independence of τ₁ anywhere else, for example
where the negative-bit range is combined with the other ranges and with the
Stage 2 batching challenge, and whether the grinding schedule for
`GrindingSite::Tau1Point` depends on the number of drawn coefficients.

### Ordinary root setup rows through a setup prefix

**Cost.** The mixed shape's ordinary root derives its setup matrix during
verification (`derive_public_matrix_prefix`: 2.4 and 4.8 ms single-threaded,
0.6 and 0.9 ms parallel). It evaluates the A, B and compression-map rows as
residue-kernel lanes. `binary_root_weights_ordinary` takes 3.9 and 7.1 ms
single-threaded and 2.1 and 3.1 ms parallel. The ordinary share of the
relation evaluation, the mixed span minus the `word-prime` span, is about 1.5
and 2.1 ms single-threaded and 1.1 and 1.4 ms parallel.

**Change.** Move the setup-dependent ordinary rows into a Stage 3 setup
product claim, as the B/D setup offload did for the binary rows
([binary-source-setup-offload.md](binary-source-setup-offload.md)). The
ordinary root reads the canonical ordinary setup prefix with each role's own
dimensions, so the claim would be against that prefix's prepared commitment.
It would be either a second Stage 3 session or a batch with the typed prefix
claim. A second session adds `32k + 32` field-message bytes for a prefix of
`2^k` cells, plus scheduled grinding nonces. The Stage 2 linear weight loses
those rows, and the transcript changes.

**Expected gain.** At most 5.4 ms at N=2^22 and 9.2 ms at N=2^23
single-threaded, and 3.2 and 4.5 ms parallel, less the added Stage 3 replay.
The binary root's Stage 3 replay (`binary_root_a_stage3`) takes 0.3–0.5 ms
for comparison.

**Soundness question.** The B/D offload argument carries over: the product
sumcheck adds `2k/P`, and the prefix commitment's binding is used. Open
questions are whether the ordinary prefix identity is authenticated at the
root the way the typed prefix is, and whether two prefix claims of different
lengths can share one Stage 3 session without a new batching term.

### Suffix setup scans

This cost belongs to the ordinary protocol on `main`, not to the binary
source extension.

**Cost.** Each of the six ordinary suffix folds evaluates its setup
contribution directly from the expanded setup (`SetupContributionMode::Direct`,
span `relation_setup_scan`). Together they take 10.1–17.1 ms single-threaded
and 2.7–3.7 ms parallel. This is the largest remaining verifier cost: 54–67%
of single-threaded binary-only verification.

**Change.** Select the recursive setup-contribution mode
(`SetupContributionMode::Recursive`) for some suffix folds. That moves each
selected fold's setup contribution into a Stage 3 session against a setup
prefix. It is a schedule change in the planner, and it changes proof bytes
and the transcript. Each moved fold adds `32k + 32` field-message bytes and
its scheduled grinding nonces. A byte-preserving speedup of the scan kernel
on `main` is a separate option and was not examined here.

**Expected gain.** Up to 17 ms single-threaded and 3.7 ms parallel if every
fold moves, less one Stage 3 replay per moved fold. The proof grows with each
moved fold, so the choice trades proof bytes against verifier time.

**Soundness question.** None beyond the ordinary protocol's existing
setup-prefix argument. Each moved fold adds `2k/P` of sumcheck error and
relies on the binding of its prefix commitment. The planner's security
accounting has to price those terms.

## Appendix: full-size measurement history

This appendix keeps the superseded and intermediate evidence behind the
full-size results above. None of it is a current result, all of it predates
the v8 merge, and it does not repeat the pre-v8 final tables. The proof digest pins section above states which source each
pin was captured from.

### Original source-storage limit

At base commit `8f515ebbcb3015cf9988ad6de4004c96075e90cc`, neither benchmark
size can reach commitment through the existing importer. Its source storage
is `[i128;648]` per source ring element, bounded to 134,217,728 bytes. The
required allocations are calculated below, excluding all other buffers:

| N | Source ring elements | Source bytes | A-commitment kernel multiplications | Source-fold multiplications per nonce |
| --- | ---: | ---: | ---: | ---: |
| 2^22 | 1,048,576 | 10,871,635,968 | 1,320,903,770,112 | 440,301,256,704 |
| 2^23 | 2,097,152 | 21,743,271,936 | 2,641,807,540,224 | 880,602,513,408 |

The source bytes are `(N/4)*648*16`. The A-commitment and source-fold counts are
respectively `(N/4)*3*648^2` and `(N/4)*648^2`, from the unconditional
schoolbook loops in
[`signed_trinomial_product`](../crates/akita-algebra/src/ring/trinomial/signed.rs)
and their callers in the
[source importer](../crates/akita-cpu-backend/src/opaque/binary_source.rs) and
[response builder](../crates/akita-cpu-backend/src/binary_root.rs).
That version also deep-cloned the source storage during root preparation. These are calculated
allocation and operation counts, not measured RSS or timing results.
They exclude subsequent residual construction and other opening arithmetic.
The packed importer described under Memory (`da4f701d2`) replaced this
representation before the first executable full-size baseline. No full-size
proof digest is claimed for the original expanded representation, because its
importer rejects both benchmark geometries before commitment.

### Initial smoke runs and the superseded carrier search

The initial current-protocol smoke runs were taken on 2026-11-02 with the
harness source subsequently
identified by commit `e6f7ddbd98f0b17231267c04a9c121839bcfb6a8`. The mixed
driver started before that commit and logged base HEAD `8f515ebbcb3015cf9988ad6de4004c96075e90cc`
with a dirty tree; the prime-only run logged HEAD `e6f7ddbd9`, with specification
edits still uncommitted. Both openings verified, matched the
caller's transcript continuation, and passed with the online A accessor
forbidden. Every proof byte was accounted for exactly once. These N=2^10,
C=2 results validate the harness; they are not full-size benchmark results.

| Shape | Proof bytes | Atoms, including framing and work | Prover peak RSS bytes | Verifier peak RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| word-prime-ordinary | 87,341 | 25,756 | 1,546,911,744 | 404,439,040 |
| prime5 | 84,842 | 25,598 | 2,051,637,248 | 437,993,472 |

Both selected six recursive folds and eight total fold levels, with a
scheduled witness domain of 8,388,608 cells. Their scheduled live lengths
were 5,452,288 and 4,276,544 respectively. The Blake2b-256 proof digests were:

```text
word-prime-ordinary 878fe23f6701a2936ca09fa887c91449f4feedeb528e0652d3ab9374b61d83df
prime5              1c0c7581aef2a052798fc0505003047713a9e87c068ea4ec554ac0ecc504b79c
```

The mixed smoke with HEAD `13b668f6eb40787f01b5ad250bcca76766063866` and
uncommitted specification edits reproduced that exact digest after
the reporting and capacity-check corrections. Its byte/atom partition across
all fold levels is below. Stage rows include their tagged sumcheck messages,
tagged work nonces and late claims; the remaining payload, framing and work
families are reported separately. The driver also prints each family,
invocation and level, avoiding ambiguity between protocol stages and site
stage numbers.

| Category | Bytes | Atoms |
| --- | ---: | ---: |
| Binary field switch | 2,489 | 149 |
| Integer projections | 84 | 7 |
| Stage 1 tagged families | 11,302 | 907 |
| Stage 2 tagged families | 6,776 | 551 |
| Stage 3 tagged families | 692 | 62 |
| Physical L2 tagged families | 4,212 | 312 |
| Other payload, framing and work families | 61,786 | 23,768 |
| Total | 87,341 | 25,756 |

The following are inclusive tracing-span wall times in nanoseconds. `uptime`
load averages immediately before/after the mixed run were
**50.12/27.48/19.77 → 100.45/46.82/27.57**; before/after the prime-only run
they were **72.09/52.27/31.41 → 16.04/35.55/29.48**. These noisy timings
are not optimization comparisons. Peak RSS includes each process's cold
preparation and planner work.

| Phase | word-prime-ordinary ns | prime5 ns |
| --- | ---: | ---: |
| Catalog planning | 52,348,544,375 | 237,928,101,292 |
| Cold prover setup | 63,737,583 | 25,723,792 |
| Independent fixture evaluation | 696,458 | 1,895,917 |
| Binary commitment | 691,570,375 | 549,641,416 |
| Ordinary commitment | 8,499,167 | absent |
| Cold prefix preparation | 85,656,708 | 23,382,750 |
| Cold verifier setup | 38,130,041 | 23,553,750 |
| Complete proving | 3,837,995,792 | 1,729,289,334 |
| Complete verification | 699,767,625 | 546,214,958 |

Native-root admission has superseded the following search. Before it, the
N=2^22 `word-prime` command exited with an error during the bounded ordinary
carrier search, before source import. At carrier log 30 the selected ordinary
row had 24,725,376 live witness cells and a 4,194,304-cell prefix, below the
binary requirements of 25,956,096 physical cells and a 16,777,216-cell prefix.
Its planning span took 211,509,359,125 ns at starting `uptime` load averages
**13.80/31.64/28.40**; process peak RSS was 848,642,048 bytes. This is a
failed planning attempt, not a prover memory measurement or a proof digest.
The later harness capacity check distinguishes physical live cells from the
padded commitment domain; this row fails both the old and corrected checks.

### Initial current-protocol baselines

The full-size results below use the packed-source and native-admission source
subsequently identified by `4fc40d2db14cdc670161f9c310a20bf9e994c148`.
The first N=2^22 driver logged HEAD `da4f701d2` with native-admission changes
uncommitted; N=2^23 logged clean HEAD `4fc40d2db`. The saved current-protocol
baseline executable has SHA-256
`7398ec122a80fbf811ea4d3e861b9f4af9595bb5b94823650c39ad70fee84d83`.
Later word and five-plane measurements explicitly record that source commit
and executable digest. These are baselines before the subsequent parity,
Stage-2 lifetime and adjoint optimizations.

The original `word-prime` logs are
`$CARGO_TARGET_DIR/n22-native-baseline.txt` and
`$CARGO_TARGET_DIR/n23-native-baseline.txt`. The saved-executable shape runs are
`$CARGO_TARGET_DIR/baseline-current-<log>-<shape>.txt`.
Successful rows below committed, proved and verified one complete opening
inside the caller's transcript, matched its continuation, consumed every proof
byte, and kept online A reads forbidden.

| N | Shape | Proof bytes | Proof atoms | Prover peak RSS bytes | Verifier peak RSS bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 2^22 | word | 87,811 | 25,835 | 6,685,179,904 | 1,473,527,808 |
| 2^22 | word-prime | 87,786 | 25,813 | 7,068,237,824 | 2,120,646,656 |
| 2^22 | prime5 | 84,936 | 25,678 | 13,015,891,968 | 6,235,439,104 |
| 2^23 | word | 89,586 | 25,952 | 13,085,933,568 | 2,469,560,320 |
| 2^23 | word-prime | 89,661 | 26,024 | 13,687,668,736 | 3,689,857,024 |
| 2^23 | prime5 | 86,548 | 25,774 | 16,770,433,024 | 8,435,122,176 |

The N=2^23 `word-prime` baseline prover peak is below one fifth of the host's
68,719,476,736-byte memory. RSS includes the complete prover process and cold
preparation. The initial N=2^22 mixed baseline failed on the one-block-per-B-slice
restriction discussed below; it produced no proof or verifier measurement.
Its planning and error-path spans are not successful opening timings.

Their proof bytes, atoms and stage partitions equal the final rows. Their
digests are the `word-prime`, `word` and `prime5` pins described above.

The measured inclusive phase times below are nanoseconds. None was taken below
one-minute load 4. Each row records the complete `uptime` load triple before
and after its run, so these shared-host timings are not unloaded benchmarks.

| N | Shape | Complete proving ns | Complete verification ns | Load before | Load after |
| --- | --- | ---: | ---: | --- | --- |
| 2^22 | word | 14,690,556,625 | 2,156,176,333 | 8.69/10.52/10.14 | 11.84/11.44/10.53 |
| 2^22 | word-prime | 16,569,918,917 | 2,493,462,334 | 12.12/9.48/8.98 | 11.40/9.69/9.09 |
| 2^22 | prime5 | 21,659,233,542 | 5,741,997,125 | 28.27/14.65/11.77 | 17.14/14.18/11.83 |
| 2^23 | word | 28,153,493,500 | 3,515,415,083 | 17.14/14.18/11.83 | 11.88/13.36/11.74 |
| 2^23 | word-prime | 30,060,170,166 | 4,098,428,834 | 16.20/12.52/10.34 | 13.03/12.29/10.45 |
| 2^23 | prime5 | 33,363,737,625 | 5,881,257,792 | 11.88/13.36/11.74 | 10.82/12.61/11.64 |

The detailed `word-prime` preparation phases use the same loads as their rows
above: N=2^22 **12.12/9.48/8.98 → 11.40/9.69/9.09**, N=2^23
**16.20/12.52/10.34 → 13.03/12.29/10.45**.

| Phase | N=2^22 ns | N=2^23 ns |
| --- | ---: | ---: |
| Catalog planning | 27,140,578,375 | 32,006,987,833 |
| Cold prover setup | 49,687,917 | 97,725,333 |
| Independent fixture evaluation | 1,684,364,459 | 3,368,307,542 |
| Binary commitment | 1,717,501,666 | 2,003,490,625 |
| Cold prefix preparation | 179,300,667 | 671,433,750 |
| Cold verifier setup | 160,960,292 | 627,335,917 |

### First successful full-size mixed openings

After the canonical B-slice repair, both mixed benchmark sizes completed with
`verification passed=true`, matching the caller's transcript continuation and
keeping online A reads forbidden. The logs are
`$CARGO_TARGET_DIR/before-adjoint-current-22-word-prime-ordinary.txt` and
`$CARGO_TARGET_DIR/before-adjoint-current-23-word-prime-ordinary.txt`.
They record source `2b8854e14_plus_canonical_b_slice_fix` and executable SHA-256
`1012001f9a34a1471927da3adbaabf813a5deecf32d9433146fb8ecb487e5263`.
This version includes the parity and buffer-lifetime changes and precedes the
adjoint scratch optimization. There is no successful earlier mixed run from
which to infer their timing effect. Commit `e74bfada5` pins the two mixed
digests, completing pins for all eight current-protocol full-size shapes.

| N | Mixed proof bytes | Atoms | Prover peak RSS bytes | Verifier peak RSS bytes | Scheduled live cells | Domain cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | 87,790 | 25,815 | 7,450,640,384 | 2,402,238,464 | 30,924,544 | 33,554,432 |
| 2^23 | 89,664 | 26,029 | 13,254,672,384 | 4,207,935,488 | 51,863,296 | 67,108,864 |

Both schedules have six recursive folds and eight total fold levels.

These inclusive tracing times use nanoseconds. The `uptime` loads were
**8.20/9.20/10.17 → 7.51/8.94/10.05** at N=2^22 and
**7.51/8.94/10.05 → 9.47/9.23/10.09** at N=2^23.

| Mixed phase | N=2^22 ns | N=2^23 ns |
| --- | ---: | ---: |
| Catalog planning | 8,101,997,834 | 11,266,501,292 |
| Cold prover setup | 45,033,375 | 86,524,166 |
| Independent fixture evaluation | 1,690,290,459 | 3,325,695,292 |
| Binary commitment | 1,756,466,041 | 1,939,607,958 |
| Ordinary commitment | 12,613,625 | 12,838,417 |
| Cold prefix preparation | 172,515,209 | 587,664,958 |
| Cold verifier setup | 154,191,459 | 574,272,792 |
| Parity row convolution | 9,940,500 | 20,286,834 |
| Complete proving | 14,815,040,000 | 27,541,188,625 |
| Complete verification | 2,396,975,334 | 4,034,210,958 |

Their stage partitions equal the final `word-prime-ordinary` rows.

### Optimizations after the baseline

A ten-second sample of the N=2^22 `word-prime` prover is retained at
`$CARGO_TARGET_DIR/n22-root-baseline.sample.txt`. The sampled window contained
2,993 relation-construction samples and 803 samples in the dense parity
row-product path.
Those counts describe that window and call graph, not percentages of the
complete opening. An earlier sample exposed per-field unbuffered fixture I/O;
the driver now uses buffered readers and writers. That change affects fixture
transport, not proof encoding or transcript messages.

Parity construction reuses the existing parallel set-bit integer accumulation
kernel from `akita-labinius-prover`. It unpacks each response ring's low digits
once, sums unreduced scalar products, then calls the canonical trinomial
remainder once. The previous path performed a dense checked product and
reduction separately for every scalar row. The following are source-level
integer operation counts, excluding challenges, partial-column products and
carry encoding; they are independent of host load.

| N | Removed row-product multiplications | Previous convolution additions | Previous remainder additions | Previous row-result additions | Compact scalar scratch bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2^22 | 429,981,696 | 429,981,696 | 6,619,136 | 2,654,208 | 2,654,208 |
| 2^23 | 859,963,392 | 859,963,392 | 13,238,272 | 5,308,416 | 5,308,416 |

The new row-product multiplication count is zero. Coefficient accumulation
uses exactly `162 * sum(popcount(row_weights))` additions, plus 323 additions
per nonempty worker-total merge; one final remainder uses 404 additions or
subtractions. Constant-size bound checks are separate from that product work.
The unreduced absolute coefficient bounds are 10,616,832 and 21,233,664:
`(N/256)*162*4`. They cover every worker subset and fit i64. Generic wider
inputs retain checked i128 arithmetic. Differential tests compare the unreduced
sum, its remainder, compact and wide signed inputs, and native carry digits
with the previous dense construction.

The root prover and verifier now release large temporary public-oracle data
before later stages. The payload calculations below describe lifetimes, not
measured RSS savings. A and the Stage-2 terms are unnecessary after the
Stage-2 relation has consumed them; the relation-batching challenge vector
is likewise released after its last use.

| N | Relation-batching challenge bytes | Typed A payload bytes | Stage-2 term payload bytes | Total payload bytes released earlier |
| --- | ---: | ---: | ---: | ---: |
| 2^22 | 274,065,680 | 127,401,984 | 1,912,111,104 | 2,313,578,768 |
| 2^23 | 641,067,280 | 254,803,968 | 3,545,530,368 | 4,441,401,616 |

The intermediate current-protocol `word-prime` runs in
`$CARGO_TARGET_DIR/n22-parity-lifetime.txt` and
`$CARGO_TARGET_DIR/n23-parity-lifetime.txt` both report
`proof_digest_pin matched=True` and `verification passed=true`. Their exact
proof bytes and digests match the initial full-size pins. The drivers logged
dirty HEADs `f6c1b1b45` and `082dd6278`, respectively; these are combined-change
measurements, not isolated timing attribution to one optimization.

| N | Version | Complete proving ns | Complete verification ns | Prover peak RSS bytes | Verifier peak RSS bytes | Load before | Load after |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2^22 | Initial baseline | 16,569,918,917 | 2,493,462,334 | 7,068,237,824 | 2,120,646,656 | 12.12/9.48/8.98 | 11.40/9.69/9.09 |
| 2^22 | Parity and lifetime changes | 14,333,478,208 | 2,388,861,791 | 7,058,669,568 | 2,120,138,752 | 11.14/12.76/11.63 | 10.10/12.40/11.57 |
| 2^22 | Final with adjoint scratch reuse | 16,972,422,083 | 2,700,073,541 | 7,080,706,048 | 2,117,910,528 | 16.57/14.78/11.73 | 42.73/21.98/14.56 |
| 2^23 | Initial baseline | 30,060,170,166 | 4,098,428,834 | 13,687,668,736 | 3,689,857,024 | 16.20/12.52/10.34 | 13.03/12.29/10.45 |
| 2^23 | Parity and lifetime changes | 27,749,908,709 | 4,097,043,459 | 13,663,404,032 | 3,689,889,792 | 9.66/9.59/10.36 | 7.70/9.03/10.09 |
| 2^23 | Final with adjoint scratch reuse | 26,545,813,375 | 4,037,486,416 | 12,891,324,416 | 3,691,986,944 | 10.21/12.17/13.19 | 10.82/12.03/13.06 |

The new parity row-convolution span was 9,396,583 ns and 19,887,417 ns,
respectively, under those same load triples. The baseline has no corresponding
isolated span. Earlier-release payload totals do not predict these small RSS
differences because another phase can still determine the process peak.

Commit `ae173d73c540ad06de2d50d51ca97b1cbca866a0` reuses one checked arena
through each recursive Karatsuba convolution in the trinomial adjoint.
The arithmetic, leaf split threshold and multiplication count are unchanged.
The following counts apply to one degree-648 `TrinomialAdjoint::apply`, which
performs two convolutions. Cumulative field cells count every allocation's
requested length; they are not simultaneously retained cells or peak RSS.

| Per adjoint application | Before | After |
| --- | ---: | ---: |
| Buffer allocations | 3,646 | 8 |
| Cumulative allocated field cells | 112,617 | 16,021 |
| Leaf field multiplications | 154,872 | 154,872 |

Each convolution uses 5,096 scratch field cells, or 81,536 bytes for this
128-bit field. For one polynomial at rank three, the prover's projected A
functional makes `3*(m+C)` adjoint applications: 13,056 at N=2^22 and 25,344
at N=2^23. This removes exactly **47,497,728** and **92,201,472** buffer
allocations from that root construction. Its leaf multiplication totals remain
2,022,008,832 and 3,925,075,968. The online typed Stage-3 contraction only needs
one adjoint per rank row and still makes no A accessor calls.

### Historical dev-protocol baseline

The dev-protocol baseline was rebuilt from historical source
`4fc40d2db14cdc670161f9c310a20bf9e994c148`, with the canonical B-slice
materializer repair and only two added tracing spans: `fixture_serialization`
and `fixture_setup_deserialization`. It retains the historical unbuffered I/O,
dense parity construction and recursive adjoint allocations. The repair patch
SHA-256 is
`4f7f0ab7b3c43ee919375f443307144742cded0404ea1688ac3c988483899612`;
the saved executable SHA-256 is
`83318968292ba2c8400eb879a963cc919249716c51ae07cc73f3c892beccee97`.
It was built with the supplied target directory and Cargo shim for
`aarch64-apple-darwin`, with `dev-protocol` selected.

All eight logs, `$CARGO_TARGET_DIR/baseline-dev-<log>-<shape>.txt`, report
successful verification, matching caller transcript continuation and forbidden
online A reads. Commit `1b80cd6a9` pins all eight historical dev-protocol
proof digests before optimized dev-protocol measurements. This is a
retrospective replay of pre-optimization source: the dev baseline was rebuilt
and pinned after the optimization commits, unlike the initial current-protocol
`word-prime` pins captured before those optimizations. The saved baseline
and optimized executables are run with the driver's same `--phase prove` and
`--phase verify` arguments and a common temporary `--fixture` directory.
This avoids repeated Cargo-lock waits between measurements. Their source and
executable digests and `uptime` loads are logged separately. The driver's
`--dev-protocol` smoke is a separate build/run entry-point check.

| N | Shape | Baseline proof bytes | Atoms | Prover peak RSS bytes | Verifier peak RSS bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| 2^22 | word | 82,784 | 25,609 | 6,717,030,400 | 1,474,199,552 |
| 2^22 | word-prime | 82,731 | 25,557 | 7,070,449,664 | 2,121,285,632 |
| 2^22 | word-prime-ordinary | 82,743 | 25,569 | 7,744,585,728 | 2,402,058,240 |
| 2^22 | prime5 | 79,851 | 25,393 | 12,682,412,032 | 6,235,619,328 |
| 2^23 | word | 84,578 | 25,746 | 13,090,422,784 | 2,471,755,776 |
| 2^23 | word-prime | 84,574 | 25,739 | 13,677,051,904 | 3,692,511,232 |
| 2^23 | word-prime-ordinary | 84,621 | 25,787 | 13,799,866,368 | 4,208,295,936 |
| 2^23 | prime5 | 81,554 | 25,581 | 16,754,196,480 | 8,435,105,792 |

The baseline's canonical outgoing geometry and six-level recursive output
chains match the corresponding current-protocol rows above. Dev-protocol
proofs have a different terminal grammar; their byte counts are not evidence
that current-protocol proofs can omit the same messages.
Its stage partition equals the final dev-protocol partition.

These load triples apply to all historical dev-protocol phase times below;
no row was measured below one-minute load 4.

| N | Shape | Load before | Load after |
| --- | --- | --- | --- |
| 2^22 | word | 16.64/11.36/10.71 | 26.12/14.75/12.00 |
| 2^22 | word-prime | 26.12/14.75/12.00 | 29.11/18.34/13.58 |
| 2^22 | word-prime-ordinary | 29.11/18.34/13.58 | 32.18/20.71/14.68 |
| 2^22 | prime5 | 32.18/20.71/14.68 | 23.10/20.69/15.23 |
| 2^23 | word | 23.10/20.69/15.23 | 25.71/22.03/16.30 |
| 2^23 | word-prime | 25.71/22.03/16.30 | 19.51/20.99/16.54 |
| 2^23 | word-prime-ordinary | 19.51/20.99/16.54 | 28.63/23.29/17.77 |
| 2^23 | prime5 | 28.63/23.29/17.77 | 19.76/21.65/17.84 |

Historical dev-protocol N=2^22, inclusive nanoseconds, with the load triples above:

| Phase | word ns | word-prime ns | word-prime-ordinary ns | prime5 ns |
| --- | ---: | ---: | ---: | ---: |
| Catalog planning | 30,920,451,042 | 32,485,773,875 | 9,311,988,791 | 31,613,850,625 |
| Cold prover setup | 57,612,167 | 66,512,792 | 67,569,500 | 94,900,958 |
| Independent fixture evaluation | 19,537,042 | 1,849,675,042 | 1,863,183,875 | 9,182,538,250 |
| Binary commitment | 1,809,069,666 | 2,043,854,292 | 1,961,923,584 | 2,028,501,500 |
| Ordinary commitment | absent | absent | 24,582,125 | absent |
| Cold prefix preparation | 199,941,666 | 253,185,084 | 249,568,083 | 263,713,334 |
| Cold verifier setup | 185,257,041 | 222,610,041 | 208,754,833 | 207,131,292 |
| Fixture serialization | 6,343,925,167 | 6,107,946,875 | 6,260,954,041 | 6,150,190,041 |
| Fixture setup deserialization | 89,677,083 | 87,560,250 | 89,923,458 | 92,268,083 |
| Complete proving | 19,643,377,583 | 19,518,184,125 | 17,057,029,209 | 23,907,995,375 |
| Complete verification | 2,417,749,334 | 2,698,914,042 | 2,869,937,041 | 5,818,740,292 |

Historical dev-protocol N=2^23, inclusive nanoseconds, with the load triples above:

| Phase | word ns | word-prime ns | word-prime-ordinary ns | prime5 ns |
| --- | ---: | ---: | ---: | ---: |
| Catalog planning | 36,788,665,000 | 36,754,778,959 | 13,405,380,042 | 36,218,120,041 |
| Cold prover setup | 139,934,000 | 128,158,708 | 182,121,166 | 126,810,875 |
| Independent fixture evaluation | 41,927,667 | 3,702,988,500 | 3,689,426,792 | 17,796,588,791 |
| Binary commitment | 2,348,539,625 | 2,297,066,167 | 2,290,839,541 | 2,334,813,333 |
| Ordinary commitment | absent | absent | 25,619,250 | absent |
| Cold prefix preparation | 948,339,083 | 736,382,500 | 858,827,250 | 890,161,041 |
| Cold verifier setup | 885,988,958 | 762,290,792 | 753,013,958 | 816,168,958 |
| Fixture serialization | 15,382,593,792 | 14,740,160,750 | 15,237,461,834 | 14,805,404,416 |
| Fixture setup deserialization | 241,358,542 | 234,842,833 | 227,985,250 | 233,246,875 |
| Complete proving | 33,873,689,833 | 35,893,664,792 | 34,586,278,334 | 38,174,043,667 |
| Complete verification | 4,120,492,125 | 4,785,251,541 | 4,827,243,125 | 6,817,642,875 |

### Historical-to-final dev-protocol comparison

Every row below preserves its historical proof's exact bytes and digest.
The final columns are the `b64fdbb4c` results, before the structured root
weights; the structured root weight history below supersedes their
verification times.
The timings compare the complete combined changes; they do not isolate parity,
allocation lifetime, adjoint scratch, buffering, compiler or load effects.
Assess the times with the baseline load table in this appendix and the final
dev-protocol load table above.
Peak RSS is an independent process measurement, not the sum of buffer sizes.

| N | Shape | Baseline prove ns | Final prove ns | Baseline verify ns | Final verify ns | Baseline prover RSS bytes | Final prover RSS bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | word | 19,643,377,583 | 13,180,117,792 | 2,417,749,334 | 2,077,195,042 | 6,717,030,400 | 5,611,520,000 |
| 2^22 | word-prime | 19,518,184,125 | 14,305,025,750 | 2,698,914,042 | 2,490,934,167 | 7,070,449,664 | 7,067,271,168 |
| 2^22 | word-prime-ordinary | 17,057,029,209 | 14,260,011,083 | 2,869,937,041 | 2,416,851,959 | 7,744,585,728 | 7,479,492,608 |
| 2^22 | prime5 | 23,907,995,375 | 19,831,039,583 | 5,818,740,292 | 5,496,839,916 | 12,682,412,032 | 12,482,592,768 |
| 2^23 | word | 33,873,689,833 | 25,438,106,167 | 4,120,492,125 | 3,525,101,458 | 13,090,422,784 | 10,916,085,760 |
| 2^23 | word-prime | 35,893,664,792 | 26,886,707,000 | 4,785,251,541 | 4,093,525,875 | 13,677,051,904 | 13,666,058,240 |
| 2^23 | word-prime-ordinary | 34,586,278,334 | 26,642,876,458 | 4,827,243,125 | 4,112,921,833 | 13,799,866,368 | 12,827,443,200 |
| 2^23 | prime5 | 38,174,043,667 | 30,962,749,791 | 6,817,642,875 | 5,993,753,084 | 16,754,196,480 | 14,592,933,888 |

The fixture I/O spans provide the following before/after evidence for buffered
transport, under those same baseline and final load triples. Fixture
serialization and setup deserialization are outside the complete opening
spans. These are phase measurements from combined-change runs, not an isolated
I/O microbenchmark or a protocol speedup.

| N | Shape | Unbuffered serialization ns | Buffered serialization ns | Historical setup deserialization ns | Final setup deserialization ns |
| --- | --- | ---: | ---: | ---: | ---: |
| 2^22 | word | 6,343,925,167 | 43,618,084 | 89,677,083 | 71,212,208 |
| 2^22 | word-prime | 6,107,946,875 | 39,096,625 | 87,560,250 | 73,428,584 |
| 2^22 | word-prime-ordinary | 6,260,954,041 | 36,684,959 | 89,923,458 | 74,661,083 |
| 2^22 | prime5 | 6,150,190,041 | 46,458,083 | 92,268,083 | 78,432,291 |
| 2^23 | word | 15,382,593,792 | 152,262,000 | 241,358,542 | 181,717,083 |
| 2^23 | word-prime | 14,740,160,750 | 128,314,709 | 234,842,833 | 192,590,583 |
| 2^23 | word-prime-ordinary | 15,237,461,834 | 105,148,417 | 227,985,250 | 187,963,167 |
| 2^23 | prime5 | 14,805,404,416 | 208,287,000 | 233,246,875 | 194,434,917 |

### Structured root weight history

Before the structured root weight families, the final tables above reported
the following verification times and verifier peak RSS from `b64fdbb4c`, under
the load triples of their phase tables.

| Protocol | N | Shape | Complete verification ns | Verifier peak RSS bytes |
| --- | --- | --- | ---: | ---: |
| Current | 2^22 | word | 2,614,753,208 | 1,472,561,152 |
| Current | 2^22 | word-prime | 2,700,073,541 | 2,117,910,528 |
| Current | 2^22 | word-prime-ordinary | 2,596,214,334 | 2,403,041,280 |
| Current | 2^22 | prime5 | 6,010,423,708 | 6,235,340,800 |
| Current | 2^23 | word | 3,518,153,167 | 2,470,723,584 |
| Current | 2^23 | word-prime | 4,037,486,416 | 3,691,986,944 |
| Current | 2^23 | word-prime-ordinary | 4,093,851,125 | 4,208,427,008 |
| Current | 2^23 | prime5 | 6,389,273,750 | 8,434,941,952 |
| Dev | 2^22 | word | 2,077,195,042 | 1,473,511,424 |
| Dev | 2^22 | word-prime | 2,490,934,167 | 2,119,794,688 |
| Dev | 2^22 | word-prime-ordinary | 2,416,851,959 | 2,403,155,968 |
| Dev | 2^22 | prime5 | 5,496,839,916 | 6,234,652,672 |
| Dev | 2^23 | word | 3,525,101,458 | 2,470,641,664 |
| Dev | 2^23 | word-prime | 4,093,525,875 | 3,690,250,240 |
| Dev | 2^23 | word-prime-ordinary | 4,112,921,833 | 4,208,427,008 |
| Dev | 2^23 | prime5 | 5,993,753,084 | 8,434,696,192 |

The families replaced flat term lists built from the outgoing domain. At
N=2^22 `word-prime` on the current protocol, with one-minute load between 4 and 5,
the 2.27 s verification included 1,171 ms of root relation construction. That
construction comprised the image term (445 ms), padding (85 ms), parity rows
(66 ms), prime planes (127 ms), B authentication (283 ms), D authentication
(127 ms) and merging (29 ms). Flat evaluation took a further 323 ms and the
padding challenge draw 735 ms. After the last family, the image, prime and
parity evaluations take 1.9, 0.5 and 0.6 ms. Padding evaluation takes about
58 ms, and B/D evaluation about 370 ms, of which about 238 ms derives setup
entries.

Each commit below was measured on one saved N=2^22 `word-prime`
current-protocol fixture. Each entry is the median of three isolated
verifications, with the one-minute load range of those runs; the earlier
value is the previous entry's.

| Commit | Change | Complete verification ns | One-minute load |
| --- | --- | ---: | --- |
| `230439aa6` | Baseline with family tracing spans | 2,343,248,167 | 6-7 |
| `70a741f0d` | Padding weights stream runs from the repacking geometry | 2,133,204,875 | 6-7 |
| `71ec0dbfc` | Image term through one adjoint per A row and sparse challenge dots | 1,742,762,125 | 8-9 |
| `d10056024` | Prime plane weights in factored form | 1,487,466,208 | 9-10 |
| `1af4caa2b` | Parity rows through Four Russians subset sums | 1,422,413,500 | 10-12 |
| `4c203ebb1` | B/D authentication through one residue kernel per setup column | 1,204,842,084 | 10-11 |

The later commits `fe7d0b526` and `8d395668a` change only documentation and a
test.

### Before padding-row deletion

The padding-row deletion (`8536dfa71`) removed the root's per-cell padding
rows and their batching scalars; everything else is unchanged. Before it, the
final tables reported these values. Proof bytes, complete proving and prover
peak RSS come from `b64fdbb4c`; complete verification and verifier peak RSS
come from `8d395668a`, under the load triples recorded with them.

| Protocol | N | Shape | Proof bytes | Complete proving ns | Prover peak RSS bytes | Complete verification ns | Verifier peak RSS bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Current | 2^22 | word | 87,811 | 15,183,593,500 | 5,647,679,488 | 1,328,603,792 | 682,721,280 |
| Current | 2^22 | word-prime | 87,786 | 16,972,422,083 | 7,080,706,048 | 1,341,655,250 | 625,704,960 |
| Current | 2^22 | word-prime-ordinary | 87,790 | 15,325,837,834 | 7,487,815,680 | 1,350,053,125 | 949,370,880 |
| Current | 2^22 | prime5 | 84,936 | 23,093,974,750 | 11,841,126,400 | 2,936,744,000 | 1,475,411,968 |
| Current | 2^23 | word | 89,586 | 25,241,062,250 | 10,936,434,688 | 2,337,992,542 | 1,497,088,000 |
| Current | 2^23 | word-prime | 89,661 | 26,545,813,375 | 12,891,324,416 | 2,328,043,666 | 1,440,219,136 |
| Current | 2^23 | word-prime-ordinary | 89,664 | 26,455,607,667 | 12,852,248,576 | 2,340,802,584 | 2,032,025,600 |
| Current | 2^23 | prime5 | 86,548 | 31,498,905,792 | 14,589,149,184 | 2,404,516,125 | 1,211,678,720 |
| Dev | 2^22 | word | 82,784 | 13,180,117,792 | 5,611,520,000 | 1,195,380,917 | 681,279,488 |
| Dev | 2^22 | word-prime | 82,731 | 14,305,025,750 | 7,067,271,168 | 1,215,787,167 | 623,312,896 |
| Dev | 2^22 | word-prime-ordinary | 82,743 | 14,260,011,083 | 7,479,492,608 | 1,224,120,833 | 948,371,456 |
| Dev | 2^22 | prime5 | 79,851 | 19,831,039,583 | 12,482,592,768 | 2,939,910,916 | 1,475,575,808 |
| Dev | 2^23 | word | 84,578 | 25,438,106,167 | 10,916,085,760 | 2,339,421,625 | 1,494,253,568 |
| Dev | 2^23 | word-prime | 84,574 | 26,886,707,000 | 13,666,058,240 | 2,364,934,333 | 1,439,842,304 |
| Dev | 2^23 | word-prime-ordinary | 84,621 | 26,642,876,458 | 12,827,443,200 | 2,334,033,583 | 2,027,765,760 |
| Dev | 2^23 | prime5 | 81,554 | 30,962,749,791 | 14,592,933,888 | 2,626,463,584 | 1,211,662,336 |

Before the deletion, the digest table held these pins. Each superseded pin was captured at a different point in the history above. A pin establishes byte
equality only between the source it was captured from and later sources:

- Packed source storage (`da4f701d2`) predates every full-size baseline. The
  earlier expanded importer rejected both benchmark sizes before commitment,
  so no pin establishes byte equality with the pre-packed representation.
- The current-protocol `word-prime` pins (`325b72ca6` at N=2^22, `f6c1b1b45`
  at N=2^23) come from the native-admission source `4fc40d2db`. They predate
  the parity (`082dd6278`), buffer-lifetime (`b17e9a2e2`), fixture-buffering
  (`2b8854e14`) and adjoint-scratch (`ae173d73c`) changes. The `word` and
  `prime5` pins (`56aed7ed4`) come from the saved `4fc40d2db` baseline
  executable.
- The current-protocol `word-prime-ordinary` pins (`e74bfada5`) follow the
  B-slice repair. They were captured with that repair applied before it was
  committed as `b64fdbb4c`, on source that already included the parity,
  buffer-lifetime and buffering changes but not the adjoint-scratch change.
- The dev-protocol full-size pins (`1b80cd6a9`) were reconstructed after all
  of those optimizations. They were built in a fresh worktree from historical
  source `4fc40d2db`, plus the B-slice repair and two tracing spans.
- The current-protocol smoke pin (`325b72ca6`) reproduces the first smoke run.
  The dev-protocol smoke pin was added in `05df5f614`.

The `b64fdbb4c` final runs reproduced all sixteen full-size pins of that time. Together those pins
support the claim that the parity, buffer-lifetime, buffering and adjoint
changes preserve proof bytes. They do not show that the packed representation
preserves the bytes of the expanded one.

### Before the B/D setup offload

Before the B/D setup offload, the final tables reported these values from code
commit `8536dfa71`, after the padding-row deletion and before the
byte-preserving verifier speedups `38755e7e7`..`db34922aa`. Each
configuration proved once and verified three times; verification is the
median, with the peak RSS of that run. The one-minute loads recorded around
those runs ranged from 9.48 to 22.11 (current) and 7.08 to 15.46 (dev). The other phase rows came from
`b64fdbb4c`. The digest table held these proof sizes; the dev-protocol smoke
pin was 82,308 bytes and the current-protocol smoke pin 87,337 bytes.

| Protocol | N | Shape | Proof bytes | Complete proving ns | Prover peak RSS bytes | Complete verification ns | Verifier peak RSS bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Current | 2^22 | word | 87,804 | 11,492,767,791 | 5,041,061,888 | 254,508,416 | 75,104,256 |
| Current | 2^22 | word-prime | 87,805 | 12,724,620,834 | 5,068,324,864 | 366,722,875 | 75,759,616 |
| Current | 2^22 | word-prime-ordinary | 87,786 | 12,590,064,167 | 5,178,376,192 | 615,104,291 | 560,644,096 |
| Current | 2^22 | prime5 | 84,879 | 16,216,057,708 | 7,606,075,392 | 783,566,834 | 80,904,192 |
| Current | 2^23 | word | 89,641 | 22,705,387,458 | 9,918,758,912 | 266,389,958 | 158,236,672 |
| Current | 2^23 | word-prime | 89,603 | 25,467,671,042 | 9,698,836,480 | 387,217,417 | 158,466,048 |
| Current | 2^23 | word-prime-ordinary | 89,638 | 26,341,418,583 | 9,517,465,600 | 964,557,042 | 1,041,711,104 |
| Current | 2^23 | prime5 | 86,595 | 27,111,337,334 | 9,195,012,096 | 791,450,834 | 158,810,112 |
| Dev | 2^22 | word | 82,739 | 12,256,222,875 | 5,061,033,984 | 261,353,584 | 74,596,352 |
| Dev | 2^22 | word-prime | 82,788 | 14,150,406,375 | 5,083,938,816 | 363,747,209 | 75,382,784 |
| Dev | 2^22 | word-prime-ordinary | 82,747 | 12,806,476,042 | 5,180,211,200 | 625,042,541 | 562,757,632 |
| Dev | 2^22 | prime5 | 79,860 | 15,958,563,125 | 7,618,150,400 | 781,013,083 | 79,659,008 |
| Dev | 2^23 | word | 84,544 | 22,128,615,583 | 9,911,517,184 | 252,647,125 | 155,254,784 |
| Dev | 2^23 | word-prime | 84,554 | 23,296,890,708 | 9,917,906,944 | 364,523,333 | 157,040,640 |
| Dev | 2^23 | word-prime-ordinary | 84,598 | 23,485,063,167 | 9,922,002,944 | 815,302,375 | 1,039,532,032 |
| Dev | 2^23 | prime5 | 81,534 | 26,446,996,167 | 9,554,968,576 | 779,827,333 | 156,106,752 |

The prefix domain was 2^24 at N=2^22 and 2^25 at N=2^23 for every shape. The
recursive output chains of N=2^22 `word-prime`, `word-prime-ordinary` and
`prime5`, and of N=2^23 `prime5`, were:

| N | Shape | Successive recursive output cells |
| --- | --- | --- |
| 2^22 | word-prime | 21,294,592 → 2,652,800 → 953,408 → 280,448 → 128,896 → 93,696 |
| 2^22 | word-prime-ordinary | 21,514,752 → 2,672,064 → 953,408 → 280,448 → 128,896 → 93,696 |
| 2^22 | prime5 | 21,778,944 → 2,691,328 → 953,408 → 280,448 → 128,896 → 93,696 |
| 2^23 | prime5 | 18,256,384 → 1,814,528 → 565,248 → 245,504 → 121,856 → 93,696 |

The other chains are unchanged by the offload. The end-to-end fixture sizes
before the offload were, in the current and dev protocols: prime only
83,824/78,781, 83,836/78,765 and 83,806/78,817 bytes for one, two and five
planes; word and prime 86,024/80,991, 86,060/80,963 and 86,029/80,973; the
word-only fixtures 86,070/80,987 (width 128), 86,046/81,013 (width 8) and
86,155/81,075 (with the ordinary group).

### B/D setup offload: verification before and after

This comparison preceded the byte-preserving verifier speedups after the
offload; its offload column is the base of the pre-v8 tables after the
verifier speedups.

The B/D setup offload
([binary-source-setup-offload.md](binary-source-setup-offload.md)) moves the
B and D authentication weights into the typed prefix and Stage 3. Online
verification no longer derives any B/D setup entry. The table compares three
builds on the same current-protocol fixtures:

- **Base** is commit `798c0fb06`, the parent of this work (executable SHA-256
  `16d8741916517c4f47b7617ce8f3a59a974cab4db3d2a2d519df9ba2fc2da0a0`).
- **Speedups** is commit `db34922aa`, which adds four byte-preserving
  verifier speedups (one XOF read per B/D setup entry, parallel B/D columns,
  reused ordinary-root residue kernels and parallel explicit-term sums) and a
  feature-gating fix
  (executable SHA-256
  `0d55a560e7101c976541b69e2d19e9c5247862cfd55d4619091cea932166577d`).
- **Offload** is commit `7ca9e6656`, the source of the pre-v8
  current-protocol results.

Base and speedups verified the same proofs, three times each, alternating per
configuration; proof bytes are identical between them. The offload builds its
own proofs. Each cell is the median of three isolated verification
processes, with the peak RSS of that median run. The logs are
`$CARGO_TARGET_DIR/bd-offload/{base,speedups}/<log>-<shape>.verify<i>.txt`
and `$CARGO_TARGET_DIR/bd-offload/<log>-<shape>.verify<i>.txt`.

| N | Shape | Base ms | Base RSS bytes | Speedups ms | Speedups RSS bytes | Offload ms | Offload RSS bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^22 | word | 249.3 | 74,022,912 | 33.7 | 75,939,840 | 14.9 | 75,284,480 |
| 2^22 | word-prime | 353.0 | 75,268,096 | 43.2 | 77,643,776 | 16.1 | 157,270,016 |
| 2^22 | word-prime-ordinary | 601.2 | 562,708,480 | 150.5 | 566,575,104 | 124.8 | 644,710,400 |
| 2^22 | prime5 | 764.5 | 78,872,576 | 72.3 | 84,705,280 | 15.2 | 126,566,400 |
| 2^23 | word | 253.1 | 154,927,104 | 34.2 | 157,614,080 | 16.2 | 156,794,880 |
| 2^23 | word-prime | 364.1 | 158,023,680 | 51.4 | 159,121,408 | 17.5 | 157,450,240 |
| 2^23 | word-prime-ordinary | 803.9 | 1,038,237,696 | 241.5 | 1,048,707,072 | 215.2 | 1,047,756,800 |
| 2^23 | prime5 | 785.3 | 157,761,536 | 84.1 | 163,905,536 | 16.5 | 133,218,304 |

The one-minute load before each of the three verification runs:

| N | Shape | Base loads | Speedups loads | Offload loads |
| --- | --- | --- | --- | --- |
| 2^22 | word | 4.73, 4.73, 4.73 | 4.73, 4.73, 4.73 | 6.19, 6.19, 6.19 |
| 2^22 | word-prime | 4.73, 4.73, 4.73 | 4.73, 4.73, 4.73 | 7.49, 7.49, 7.49 |
| 2^22 | word-prime-ordinary | 4.59, 4.59, 4.59 | 4.59, 4.59, 4.59 | 7.30, 7.30, 7.30 |
| 2^22 | prime5 | 4.59, 4.73, 4.73 | 4.59, 4.59, 4.59 | 6.23, 6.13, 6.13 |
| 2^23 | word | 5.85, 5.85, 5.85 | 5.85, 5.85, 5.85 | 5.86, 5.86, 5.86 |
| 2^23 | word-prime | 3.58, 3.77, 3.77 | 3.77, 3.77, 3.77 | 5.69, 5.69, 5.69 |
| 2^23 | word-prime-ordinary | 3.87, 3.87, 3.87 | 3.80, 3.80, 3.80 | 5.82, 5.82, 5.82 |
| 2^23 | prime5 | 3.82, 3.72, 3.72 | 3.82, 3.82, 3.82 | 6.39, 6.39, 6.39 |

With the speedups alone, B/D authentication (`root_weights_evaluate_authentication`)
was still 28.0 of 43.2 ms at N=2^22 `word-prime` and 58.3 of 72.3 ms at
N=2^22 `prime5`. After the offload that span is gone, and the binary-only
shapes verify in 14.9–17.5 ms at both sizes. The mixed shape's remaining
time is the ordinary root's flat-term construction
(`binary_root_weights_ordinary`: 85.9 ms at N=2^22 and 160.7 ms at N=2^23),
which the offload does not touch.

Verifier peak RSS follows the selected prefix, not the B/D work. Where the
prefix domain grows (N=2^22 `word-prime` and `word-prime-ordinary` from 2^24
to 2^25, `prime5` to 2^26), the successor folds and their prepared setup grow
with it. Against the speedups build, RSS rises by 42 MB (`prime5`) to 80 MB
(`word-prime`) at N=2^22. Where it does not grow (N=2^22 `word`, and N=2^23
except `prime5`), RSS is within 2 MB of the speedups build. N=2^23 `prime5` drops by
31 MB under its different successor schedule.

### Verification before the verifier speedups

The pre-v8 current-protocol campaign on `7ca9e6656` reported these complete
verification medians and verifier peak RSS values, under the load triples of
that campaign. The parallel base column of the pre-v8 table after the
verifier speedups remeasures the same source.

| N | Shape | Complete verification ns | Verifier peak RSS bytes |
| --- | --- | ---: | ---: |
| 2^22 | word | 14,900,416 | 75,284,480 |
| 2^22 | word-prime | 16,091,291 | 157,270,016 |
| 2^22 | word-prime-ordinary | 124,834,791 | 644,710,400 |
| 2^22 | prime5 | 15,235,792 | 126,566,400 |
| 2^23 | word | 16,205,875 | 156,794,880 |
| 2^23 | word-prime | 17,545,167 | 157,450,240 |
| 2^23 | word-prime-ordinary | 215,159,375 | 1,047,756,800 |
| 2^23 | prime5 | 16,455,833 | 133,218,304 |

### Before the prover-speed pass

The prover span profile replaced these prover figures in the final tables.
Binary commitment comes from the `b64fdbb4c` phase runs. Complete proving and
prover peak RSS come from the single `8536dfa71` proving runs, which planned
their catalogs inside the proving process. Each was measured under the load
triples recorded with it.

| Protocol | N | Shape | Binary commitment ns | Complete proving ns | Prover peak RSS bytes |
| --- | --- | --- | ---: | ---: | ---: |
| Current | 2^22 | word | 1,931,572,875 | 11,492,767,791 | 5,041,061,888 |
| Current | 2^22 | word-prime | 1,863,130,792 | 12,724,620,834 | 5,068,324,864 |
| Current | 2^22 | word-prime-ordinary | 1,955,571,458 | 12,590,064,167 | 5,178,376,192 |
| Current | 2^22 | prime5 | 1,897,854,000 | 16,216,057,708 | 7,606,075,392 |
| Current | 2^23 | word | 2,057,973,375 | 22,705,387,458 | 9,918,758,912 |
| Current | 2^23 | word-prime | 2,007,653,125 | 25,467,671,042 | 9,698,836,480 |
| Current | 2^23 | word-prime-ordinary | 1,989,260,875 | 26,341,418,583 | 9,517,465,600 |
| Current | 2^23 | prime5 | 2,007,191,500 | 27,111,337,334 | 9,195,012,096 |
| Dev | 2^22 | word | 1,780,184,042 | 12,256,222,875 | 5,061,033,984 |
| Dev | 2^22 | word-prime | 1,771,492,417 | 14,150,406,375 | 5,083,938,816 |
| Dev | 2^22 | word-prime-ordinary | 1,771,982,542 | 12,806,476,042 | 5,180,211,200 |
| Dev | 2^22 | prime5 | 1,778,575,292 | 15,958,563,125 | 7,618,150,400 |
| Dev | 2^23 | word | 2,025,972,583 | 22,128,615,583 | 9,911,517,184 |
| Dev | 2^23 | word-prime | 2,017,258,875 | 23,296,890,708 | 9,917,906,944 |
| Dev | 2^23 | word-prime-ordinary | 2,033,678,958 | 23,485,063,167 | 9,922,002,944 |
| Dev | 2^23 | prime5 | 2,011,634,083 | 26,446,996,167 | 9,554,968,576 |

The baseline built residual rows at two call sites: inside the response build
and again in the projection replay. The collector sums spans by name, so the
baseline residual-rows figure is the total over both call sites. The baseline
tables therefore list it at the binary root level, beside the response build
and the projections, and it exceeds the projections span. In the final
executable only the projections build residual rows, so the final tables nest
the row under projections.

The baseline executable `f5e9ad551` is the branch base `798c0fb06` with only
the spans and the CPU and heap metrics added, built with the catalog cache.
Its interleaved medians, from the same sweep as the final prover span tables,
were:

Baseline executable, N=2^22:

| Span | word | word-prime | word-prime-ordinary | prime5 |
| --- | ---: | ---: | ---: | ---: |
| Binary commitment | 1,698.8 | 1,700.7 | 1,627.9 | 1,698.3 |
| › pack words | 26.8 | 26.7 | 24.1 | 27.2 |
| › A matrix derivation | 86.3 | 87.6 | 84.1 | 86.6 |
| › image hint (A·s) | 104.2 | 104.7 | 89.1 | 104.9 |
| › B payload | 1,474.8 | 1,475.4 | 1,425.1 | 1,474.0 |
| Complete proving | 11,384.1 | 12,286.0 | 12,220.0 | 16,031.1 |
| › binary root level | 11,102.1 | 12,004.1 | 11,969.9 | 15,753.8 |
| ›› field switch | 56.6 | 52.5 | 47.9 | 0.0 |
| ›› D commitment | 35.9 | 708.6 | 699.7 | 3,385.9 |
| ››› D payload | 15.2 | 665.4 | 650.1 | 3,292.1 |
| ›› response build | 1,609.2 | 1,584.5 | 1,557.9 | 1,633.9 |
| ››› response fold | 1,357.6 | 1,312.3 | 1,282.5 | 1,341.2 |
| ››› successor table | 83.2 | 84.6 | 79.8 | 85.7 |
| ››› repack | 139.5 | 157.1 | 164.4 | 192.8 |
| ›› projections | 1,341.9 | 1,345.2 | 1,274.2 | 1,339.6 |
| ›› residual rows, aggregate of response build and projections | 2,563.4 | 2,521.0 | 2,448.1 | 2,538.1 |
| ›› relation compile | 7,016.0 | 7,171.4 | 7,218.1 | 7,086.1 |
| ›› Stage 1 | 16.8 | 15.4 | 20.3 | 20.8 |
| ›› weight materialization | 459.6 | 634.2 | 665.4 | 1,381.3 |
| ›› Stage 2 | 312.5 | 316.7 | 287.7 | 580.3 |
| ›› Stage 3 | 125.9 | 124.6 | 126.8 | 164.7 |
| Complete proving CPU | 16,809 | 18,122 | 18,213 | 23,249 |
| Proving heap peak MB, maximum | 2,880 | 2,859 | 2,953 | 4,791 |
| Prover peak RSS bytes | 4,467,998,720 | 4,470,521,856 | 4,759,044,096 | 7,138,295,808 |

Baseline executable, N=2^23:

| Span | word | word-prime | word-prime-ordinary | prime5 |
| --- | ---: | ---: | ---: | ---: |
| Binary commitment | 1,910.3 | 1,847.3 | 1,854.7 | 1,899.9 |
| › pack words | 54.1 | 49.7 | 49.7 | 50.8 |
| › A matrix derivation | 173.4 | 167.6 | 171.4 | 169.9 |
| › image hint (A·s) | 203.0 | 189.5 | 190.9 | 213.3 |
| › B payload | 1,474.2 | 1,437.3 | 1,437.0 | 1,458.2 |
| Complete proving | 22,239.0 | 22,483.7 | 22,958.9 | 26,069.9 |
| › binary root level | 21,815.6 | 22,101.2 | 22,574.2 | 25,682.1 |
| ›› field switch | 114.1 | 101.3 | 97.8 | 0.0 |
| ›› D commitment | 61.3 | 737.3 | 745.1 | 3,400.9 |
| ››› D payload | 15.2 | 658.9 | 656.9 | 3,265.3 |
| ›› response build | 3,089.3 | 2,975.1 | 3,014.8 | 3,014.8 |
| ››› response fold | 2,643.3 | 2,543.5 | 2,536.4 | 2,579.7 |
| ››› successor table | 149.2 | 138.7 | 139.4 | 138.2 |
| ››› repack | 249.4 | 247.5 | 289.8 | 276.7 |
| ›› projections | 2,618.7 | 2,527.5 | 2,528.0 | 2,541.4 |
| ›› residual rows, aggregate of response build and projections | 4,966.8 | 4,837.2 | 4,820.0 | 4,892.1 |
| ›› relation compile | 14,299.7 | 13,992.5 | 14,358.0 | 14,005.4 |
| ›› Stage 1 | 22.4 | 23.4 | 26.7 | 29.0 |
| ›› weight materialization | 563.6 | 815.0 | 826.4 | 1,729.4 |
| ›› Stage 2 | 560.6 | 586.3 | 551.2 | 598.2 |
| ›› Stage 3 | 243.1 | 236.0 | 233.4 | 236.7 |
| Complete proving CPU | 32,066 | 32,623 | 33,333 | 37,020 |
| Proving heap peak MB, maximum | 5,718 | 5,721 | 5,833 | 5,733 |
| Prover peak RSS bytes | 8,476,114,944 | 8,462,876,672 | 8,947,924,992 | 8,454,848,512 |
