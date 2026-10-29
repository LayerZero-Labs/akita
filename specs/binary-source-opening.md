# Spec: binary source grouped opening

| Field | Value |
| --- | --- |
| Author(s) | Akita contributors |
| Created | 2026-10-28 |
| Status | implemented experimental path; security admission pending |
| PR | Pending |
| Supersedes | None |
| Superseded-by | |
| Book-chapter | book/src/how/architecture.md |

## Scope and selection

A binary source is a commitment-group kind in the canonical grouped PCS. The
caller commits, supplies one binary terminal claim, and uses the ordinary
grouped opening entry points inside the caller's transcript. The root checks
the source relation and repacks its bounded coefficients into the ordinary
negacyclic witness. The existing grouped recursion and terminal then finish
the same proof.

The implementation is available through the default-off `labinius-trinomial`
feature. Selecting a binary source handle and a binary claim selects the
extension at runtime. Enabling the feature alone does not select it. Ordinary
groups retain their existing encoding and protocol path.

This is an experimental arithmetic and protocol fixture, with no security
claim. It uses `P = 2^128 - 275`, base-field challenges (`E = F`), and the
Lagrange basis. It admits one binary group first, with at most one ordinary
group second. It requires an ordinary successor fold whose predecessor setup
contribution is evaluated directly. It does not add a production binary
planner profile or certify a new schedule.

## Commitment and terminal interfaces

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

impl BinaryTerminalClaim {
    pub fn new(
        descriptor: &BinarySourceDescriptor,
        polynomial: [u8; 32],
        rho_row: Vec<[u8; 21]>,
        rho_col: Vec<[u8; 21]>,
        value: [u8; 21],
    ) -> Result<Self, AkitaError>;
}

impl<'a, E: Field, F: Field + CanonicalEncoding>
    PolynomialGroupClaims<'a, E, CommittedGroup<F>>
{
    pub fn new_binary(
        commitment: CommittedGroup<F>,
        claim: BinaryTerminalClaim,
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

`BinaryTerminalClaim` is the interface that a future field switch will
produce. `rho_row` selects scalar carriers within a column, `rho_col` selects
columns, and `value` is the binary terminal v. Every coordinate and v has the
canonical 21-byte polynomial-basis encoding of `GF(2^162)`; the unused high
bits are zero. The constructor checks ownership, point dimensions and
canonical encodings. The terminal claim's canonical bytes bind every field
before the opening's first challenge. Tests construct honest terminals
directly from source words; no field switch is implemented here.

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
| 0 | Bind caller context, setup, selected schedule, group kinds, commitments, binary manifest, `rho_row`, `rho_col`, v, and any ordinary point/value. | The complete public statement. |
| 3 | Commit the U table in the ordinary D payload, with one balanced digit per coefficient. | U precedes all column challenges. |
| 4 | Send the ordinary response nonce and draw the indexed binary column challenges; the mixed ordinary group uses complete preimages rather than a column-fold response. | The accepted bounded response Z. |
| 5 | Commit the outgoing witness containing Z/T digits, the U view, K, K_val and KA; a mixed opening also includes ordinary commitment preimages. | Every witness segment precedes relation batching. |
| 7 | Draw the shared ring-switch challenge and row-batching weights, together with the range-image point; evaluate direct setup and public weights. | All tested tables and commitments. |
| 8 | Run the ordinary Stage 1 range-image sumcheck with the base-eight digit alphabet. | The range-image terminal precedes Stage 2 batching. |
| 9 | Run one fused Stage 2 for the A quotient, B/D authentication, parity, value, padding, and range-image binding; include ordinary relations when selected. | One final opening of the same outgoing witness. |
| 10 | Omit root Stage 3: direct setup evaluation at order 7 replaces its adjoint proof and prefix claims. | The authenticated root opening passed to the suffix. |
| 11 | Continue ordinary grouped recursion and terminal verification. | Return the live caller transcript; the caller checks exact proof exhaustion. |

Orders 1, 2 and 6 are absent: there are no application reductions, field-switch
partials or frontend rounds, or projections. The committed-quotient fallback
replaces the projection protocol. Direct public weights replace an equality
tree or a new bilinear wiring compiler.

## Relations and authenticated repacking

All identities below use integer coefficient lifts before reduction modulo P.
Let `Phi = Z^162 + Z^81 + 1`; `rem_Phi` denotes its reduced remainder.
The root enforces:

```text
RA = q KA
rem_Phi(sum_r Br V0_r - sum_b U_b c_b) = 2 K
rem_Phi(sum_b E_b U_b) - v = 2 K_val
```

Here Br and E_b are direct binary equality weights from the terminal point,
V0 is the low response-digit view, and c is the binary fold challenge. K has
nine balanced base-eight digits per coefficient and K_val has six. KA's depth
comes from its checked honest-carry bound. Their tables are committed at
order 5 and are never transmitted in the clear. U has no Booleanity predicate;
admission uses its full malicious digit interval `[-4, 3]`.

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
[BinaryQuotientBounds](../crates/akita-types/src/binary_source/quotient.rs) and
[BinaryEvaluationBounds](../crates/akita-types/src/binary_source/evaluation_bounds.rs).
Both use checked integer arithmetic and reject equality at a strict no-wrap
threshold. For `P = 340282366920938463463374607431768211181`:

| Quantity | Fixture result |
| --- | ---: |
| Honest KA bound `ceil(B_A (972 m B_Z + C Gamma) / q)` | 1,137,332 |
| KA digits; enforced BKA | 8; 9,586,980 |
| `LR = 972 m B_A B_Z + C Gamma B_T` | 16,211,309,718,240 |
| `LR + q BKA` | 152,858,456,733,060 |
| Quotient margin `P - (LR + q BKA)` | 340282366920938463463374454573311478121 |
| Honest K bound; enforced BK | 4,440; 76,695,844 |
| K no-wrap bound | 153,400,568 |
| K margin | 340282366920938463463374607431614810613 |
| Honest K_val bound; enforced BK_val | 486; 149,796 |
| Value bound `3 × 162 × C × 4 + 1 + 2 BK_val` | 303,481 |
| Value margin | 340282366920938463463374607431767907700 |

The honest carry bounds fit both positive and negative digit reaches. These
integer inequalities justify the selected exact lifts; they do not establish
SIS security, whole-schedule grinding security or application agreement.

## Evidence and remaining interfaces

The canonical fixture is
[binary_source_end_to_end.rs](../crates/akita-pcs/tests/binary_source_end_to_end.rs).
With the current protocol, its measured proof sizes are 75,702 bytes for the
binary-only opening and 89,811 bytes for the mixed opening. With `dev-protocol`
enabled, the fixture selects the existing dev
suffix catalog and produces 70,610-byte and 84,791-byte proofs respectively.
It exercises the caller's prefix and
post-verification squeeze, commitment serialization, kind consistency,
statement mutations, malformed proof bytes, truncation and trailing bytes.
A test-only backend alters T, Z, U, K, K_val, KA or padding before outgoing
commitment. It uses an incorrect public oracle to complete the adversarial
proof; verification must reject using the independently compiled correct oracle.
Each rejection runs behind a panic-catching boundary.
The source check rejects standalone binary drivers and nested channel creation
on this path. Test outcomes belong in the accompanying validation report.

The ordinary regression fixture is
[ordinary_grouped_root_bytes.rs](../crates/akita-pcs/tests/ordinary_grouped_root_bytes.rs),
which pins a 70,541-byte ordinary proof. Its feature-disabled and
feature-enabled executions are the compatibility gate when the extension is
unselected. Existing pinned fixtures and goldens remain part of that gate.

The future field switch produces `BinaryTerminalClaim` without changing its
shape. The direct A-term function
[`binary_direct_a_weights`](../crates/akita-types/src/binary_source/root_plan.rs)
is the replacement point for a typed Stage 3 adjoint. Projections and prime
EP planes are not implemented by this opening. The source descriptor's
existing prime-plane contract remains separate; no prime claim or agreement
proof is discharged by a binary terminal alone.
