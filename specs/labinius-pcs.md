# LaBinius polynomial commitment

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

This specification uses the uppercase requirement words of BCP 14
([RFC 2119](https://www.rfc-editor.org/rfc/rfc2119),
[RFC 8174](https://www.rfc-editor.org/rfc/rfc8174)).

## Statement and composition

The opt-in `akita-labinius-pcs` crate opens one commitment to a binary claim,
a prime claim, or both. It composes
[the root reduction](labinius-root-reduction.md) with one nested Akita grouped
opening. The commitment is succinct and non-hiding. Source words use a sealed
`SwitchField` host H, either F128 or F192.

`RootStatement<H,E>` is the opening statement. Its optional binary claim is
a host-field point and value. Its optional `PrimeClaim<E>` is a linear
functional of the committed bits, interpreted over the challenge field E.
At least one claim MUST be present. `RootStatement::mode` selects `Binary`,
`Prime`, or `Both`; the mode is part of the reduction's grinding plan and is
bound before any claim or proof message. The prime claim's dimensions,
packing signs, coefficient weights, and `PrimeClaim::cell_evaluation` are
specified in [the root reduction](labinius-root-reduction.md#prime-claim).

There is one commit path and one opening byte format per selected mode:

- **Commit.** Compute the clear root commitment with
  `commit_binary_clear_prepared`, encode its image digit table Y with
  `encode_image`, and commit Y as one dense Lagrange-basis polynomial under
  `P::Digits`. The public commitment is its `CommittedGroup<F>`.
- **Open.** Run the reduction with an oracle backed by Akita. With a prime
  claim, commit the prime left opening uP under `P::Elements` before the fold
  challenges. Commit the response digit table W after those challenges.
  Authenticate all returned evaluations with one grouped Akita proof.
- **Verify.** Replay the same reduction and grouped opening, and require exact
  consumption of the proof byte string.

| Mode | Precommitted groups, in order | Final group | Grouped opening order |
| --- | --- | --- | --- |
| `Binary` | Y, digits | W, digits | `[Y, W]` |
| `Prime` | Y, digits; uP, elements | W, digits | `[Y, uP, W]` |
| `Both` | Y, digits; uP, elements | W, digits | `[Y, uP, W]` |

Binary openings select the existing two-group row and preserve their image
commitment and proof bytes when using binary setup sizing. Installing an
element catalog does not select a prime mode. Prime and combined openings select the same three-group row;
the outer reduction and its mode binding distinguish their statements.
No committed table is transmitted, and there is no second nested proof.

## Proof-field families and configurations

`family::FieldFamily` fixes the committed-table base field F, the challenge
field E, and two Akita configurations over that pair.

| Family | Base F | Challenge E | Digit configuration and catalog | Element configuration and catalog |
| --- | --- | --- | --- | --- |
| `Family64` | `Prime64Offset59` | `Ext2<Prime64Offset59>` | `Digits64`, `labinius_fp64_digits_b4` | `Elements64`, `labinius_fp64_elements` |
| `Family128` | `Prime128Offset275` | `Prime128Offset275` | `Digits128`, `labinius_fp128_digits_b4` | `Elements128`, `labinius_fp128_elements` |

`Digits64` mirrors `proof_optimized::fp64::Dense`; `Digits128` mirrors
`proof_optimized::fp128::DenseBounded`. Their decomposition has
`log_commit_bound=5`, the smallest signed width containing the stored alphabet
`[0,15]`, and `log_open_bound=Some(64)` or `Some(128)`. The commit bound is a
representability condition. The reduction's combined sumchecks enforce the
unsigned alphabet on Y and W at every Boolean position, including padding.

The element configurations mirror `proof_optimized::fp64::Dense` and
`proof_optimized::fp128::Dense`. They have `log_commit_bound` equal to the
base-field width and `log_open_bound=None`: uP contains full-width field
elements and has no range or alphabet check. Each configuration takes its
ring-dimension schedule, decomposition basis, basis search ranges, challenge
policy, and source class from its preset. Digit and element configurations
within one family use the same field pair and SIS modulus profile:
`Q64Offset59` or `Q128Offset275`.

The associated-type bounds of `FieldFamily` cover the setup, CPU commitment,
grouped prover, and grouped verifier. The composition is generic over the
family rather than duplicated for the two shipped pairs.

## One base-field polynomial for the prime table

The reduction supplies uP as `layout.prime_len()` elements of E. Coefficients
are the lowest axis, columns the highest, with padding as in
[the lowered relation](labinius-lowered-root.md#prime-row). Its evaluation
claim is `(point, value)` in `RootEvaluationClaims::prime`.

`flatten_prime_table` commits this as exactly one polynomial over F:

- On `Family128`, E equals F. The table is unchanged and has
  `layout.prime_log_len()` variables.
- On `Family64`, the table contains the base coordinates of each E element:
  entry `2*i+k` is coordinate k of uP[i], in the canonical order of `ExtField::base_coefficient`.
  The extra coordinate variable is first and lowest.

In `jolt_field::Ext2`, the coordinates have basis `(1, omega)`, where
omega has base coordinates `[0,1]`, constructed with `ExtField::from_base_fn`.
If the coordinate tables are T0 and T1,
then `T(r)=T0(r)+omega*T1(r)`. The nested polynomial satisfies
`T'(x,r)=(1-x)*T0(r)+x*T1(r)`. Thus `prime_opening_claim` converts the
reduction claim as follows:

```text
Family128: nested_point = point; nested_value = value.
Family64:  nested_point = [omega/(1+omega), point...];
           nested_value = value/(1+omega).
```

All points use lowest-variable-first order. Prover and verifier call the same
conversion. An extension degree other than one or two is `InvalidSetup`;
a malformed evaluation point is `InvalidProof`. The conversion adds no
message to the reduction and does not split the coordinates into separate
polynomials.

At geometry `(22,8,128)`, Y has `2^22` digits, W has `2^24` digits, and uP
has `2^18` challenge-field elements. The nested element table has `2^18`
base elements on `Family128` or `2^19` on `Family64`.

## Shipped schedule catalogs and setup sizing

`family::tables::DigitTables` derives `image_log_len`, `response_log_len`, and
`element_log_len` from the root shape, base characteristic, and extension
degree. These are the padded lengths used by `LoweredRootLayout` and by the
one-polynomial conversion.

`shipped::SUPPORTED_GEOMETRIES` fixes profile `D648Q25BoundedW46`, fold-width
log 8, fold security 128, and cell-count logs 16, 18, 20, 22, and 24. Each
digit catalog contains the scalar image and response rows, the two-group
response rows, and the three-group response rows. Each element catalog
contains the scalar producer rows for the prime table. A grouped lookup key
contains the exact producer profiles, in precommitted order, rather than
only the table lengths. Identical keys are stored once.

`shipped_catalog::<P>(geometry, artifact_root)` returns the table dimensions,
the trusted digit catalog, and the trusted element catalog. It loads their
`.aks` files through `TrustedScheduleCatalog::from_artifact_bytes`, the same
admission boundary as ordinary Akita catalogs. Unsupported geometries return
`ShippedCatalogError::UnsupportedGeometry` before file access. I/O and
admission failures remain distinct. Applications MUST distribute both
catalogs when using this loader.

`DigitTables::setup_requirements::<P>(&digits, elements)` selects sizing:
`None` sizes the binary rows for two groups; `Some(&elements)` sizes for three
and unions the digit and element requirements. Binary sizing excludes
three-group rows from its catalog scan, preserving the binary setup bytes.
The three-group setup also covers binary openings.

The optional `labinius-catalog-gen` feature enables the planner and
`gen_labinius_schedule_artifacts`; runtime loading with only `labinius` runs
no planner. Regenerate with:

```sh
cargo run --release -p akita-labinius-pcs --no-default-features \
  --features labinius,labinius-catalog-gen,transcript-blake2b \
  --bin gen_labinius_schedule_artifacts -- --output-dir artifacts
```

The generator emits the two digit catalogs and two element catalogs into
`schedules-labinius/`. Every precommitted group is priced with its producer's
`committed_source_contract`: digit for Y, element for uP. Its outputs are only
`.aks` files; it writes no catalog snapshot. `--check` compares generated
outputs byte-for-byte with the tracked files. The CI job **LaBinius schedule
artifact drift** checks the whole directory.

A workspace selection of `akita-params/dev-protocol` uses
`schedules-labinius-dev/` instead. Add that explicit feature to the command
to regenerate the dev set. Each set holds four catalogs. Enabling `labinius`
never selects the dev protocol, and neither main-protocol artifact directory
is modified by this generator.

## Public API and caller obligations

`Prover<P>::new` and `Verifier<P>::new` take a `SupportedGeometry`, a
`TrustedScheduleCatalog<P::Digits>`, an optional
`TrustedScheduleCatalog<P::Elements>`, and the nested Akita setup. `None`
supports binary openings; `Some` additionally supports prime and combined
openings. Constructors do not load artifacts or run a planner. The caller
trusts the catalogs and setup.

Before opening or reading proof bytes, each constructor checks the setup
with `Valid::check`, derives the root setup from the nested descriptor's
`setup_seed`, admits the field pair and layout, resolves the binary row and
any selected three-group row, and checks required prefix-slot coverage and
row capacity. The prover uses `ensure_prover_schedule_fits_setup`; the
verifier requires `AkitaVerifier` to admit each selected row's digest. The
prover additionally prepares the root matrix in its limb transform domain
and creates one CPU backend for all table configurations.

`Prover::commit<H>(source)` requires the geometry's source length and
`H::Source: Sync`. It returns `Committed<P>` with the public image commitment,
the clear commitment, and a private retained image-table handle.
`Prover::open<H>(source, committed, &statement)` returns the complete proof
bytes. `Verifier::verify<H>(commitment, &statement, proof)` succeeds only
after every nested claim is verified and the outer EOF check passes.

The caller MUST supply claims about the same source and use the same admitted
geometry, family, catalogs, and setup for both sides. A prime claim MUST have
the layout's point and weight dimensions; its weights describe the intended
functional, including positions where the source has no bit. The commitment
does not encode which application-level functional the caller intended.

`root()` and `oracle(commitment, mode)` are public for callers that own an
outer channel. Such callers MUST run the reduction over that root and the
family's field pair, choose the oracle mode matching `RootStatement::mode`,
and enforce their own EOF. A missing element catalog when a prime mode is
requested is `InvalidSetup`. The outer transcript binds the setup descriptor
but not prefix-registry payloads, so the caller MUST install prefix
commitments with the required setup provenance.

Without `labinius` the crate exports nothing and activates no LaBinius
dependency feature. No feature of this crate forwards `dev-protocol`.

## Transcript order and binding

Define `LP(x)=u64_le(byte_length(x)) || x`. Before `bind_image`, the reduction
absorbs its domain label, root identity including matrix seed, field-pair
identity, and canonical `RootGrindingPlan` including the opening mode. After
image binding it absorbs the selected binary and prime claims. Each field challenge pays the
plan's proof of work.

Oracle calls MUST occur once in the following order. Every call first enters
a terminal failed state; only a successful nonterminal call advances to its
next state. An error, repeated call, prime call in binary mode, or out-of-order
call leaves it failed. Every later call returns `InvalidProof` without
accessing the channel.

| Call | Timing and operation | Bytes or check |
| --- | --- | --- |
| `bind_image` | Before the frontend; check layout, first precommitted profile, and commitment | `public`: `LP("akita/labinius/pcs/v1")`, `LP(canonical compressed AkitaSetupDescriptor)`, `LP(digit schedule family name)`, `LP(canonical compressed Y CommittedGroup)` |
| `commit_prime_opening` / `bind_prime_opening` | Only with a prime claim; after image binding and before fold challenges | One `message`: canonical uP `CommittedGroup`, exactly Kp bytes, with the three-group row's second precommitted profile |
| `commit_response` / `bind_response` | After fold challenges and before alpha, xi, gamma | One `message`: canonical W `CommittedGroup`, exactly Kw bytes, with the selected row's final profile |
| `discharge` | After all returned evaluations; form `[Y,W]` or `[Y,uP,W]` | `public`: `LP(transcript_instance_descriptor::<F,P::Digits>(..., Lagrange))`, then all point coordinates followed by values, canonical compressed E elements, in group order |
| Session derivation | After descriptor and claims | `public`: `LP("akita/labinius/pcs-opening-session/v1")`; one outer `challenge_block`; session string is that label followed by the 32-byte block |
| Grouped opening | Akita `batched_prove` / `batched_verify`, Lagrange basis | Two `message`s: u64 little-endian nested proof length, then nested proof bytes |

The descriptor covers the selected row, setup, decomposition, opening layout,
basis, and nested grinding plan. Since it has no points or values,
`discharge` absorbs every claim separately from the actual grouped statement,
including the converted prime claim when present. The prime commitment is
already in the outer transcript as its canonical message. The grouped proof
also binds all commitments and claims on its own transcript.

The session block therefore depends on the complete root statement, selected
mode, all reduction messages and nonces, every commitment, the instance
descriptor, and every nested claim. Akita binds this session into its nested
challenges. The nested proof length and bytes are outer messages and are
covered by EOF. There is no added domain label for the prime commitment;
its call site, mode, exact size, and trusted profile identify it.

The reduction consumes its grinding plan before `discharge`. The session block
has no bad set because no acceptance check compares it with a field value,
so it has no reduction proof of work. Akita prices its own field challenges
under the nested plan.

## Proof layout, size, and allocation bounds

Both inline commitments have no length prefix. Their sizes Kp and Kw are
fixed by the trusted profiles. `commitment_size` uses the serializer's fixed
header and the terminal coefficient count from
`CompressionChainPlan::for_complete_source`, with checked arithmetic.
`exchange_commitment` allocates and reads exactly that size, requires exact
decoding, checks the expected profile, and requires canonical re-encoding
identical to the received bytes. The prover checks the same size before
sending. These framing failures are `InvalidProof`.

The nested proof length MUST be nonzero and at most
`expanded_schedule_proof_bound` of the resolved row and digit policy. A
planner estimate is not a parser bound. Both sides enforce this bound;
reservations are fallible and size formulas use `akita_error::checked`.
Nested verification retains its native error. No proof-supplied row or
unbounded allocation is accepted.

Let R be the reduction byte count, bounded by
`root_reduction_wire_size::<H,F,E>(shape, mode)`, and A the nested proof size.
The complete proof has:

```text
Binary: R + Kw + 8 + A.
Prime or Both: R + Kp + Kw + 8 + A.
```

For prime modes, Kp bytes occur before the fold challenges; Kw bytes occur
at the response call site. Reduction bytes surround those commitments. The
8-byte length and A-byte grouped proof are last. Public absorptions and the
session block add no proof bytes. A varies with Akita's Golomb-Rice terminal
payload and grinding nonces. R varies only through the canonical LEB128
nonces: the fold-response nonce and every visited nonzero proof-of-work site.
All other reduction message sizes are fixed by the statement and mode
([reduction wire size](labinius-root-reduction.md#reduction-wire-size)).

## Prover memory

All committed tables use `DensePoly::<F>::from_field_evals`. Y and W cost one
F element per digit, in addition to the reduction's digit bytes and dense
source mirrors. At `(22,8,128)`, those field tables cost 64 MiB and 256 MiB
on `Family128`, or 32 MiB and 128 MiB on `Family64`. The flattened prime table
costs 4 MiB on either family. Y is retained with `Committed`, while uP and W
are retained for one opening. Reduction and Akita workspaces add to this.
Succinctness refers to proof size, not prover memory or a sublinear scan of
the materialized root setup.

## Tests

`tests/binary_opening.rs` and `tests/prime_opening.rs` run end to end at the
smallest shipped geometry on both families. They commit, serialize, open,
verify, and reject table-driven changes to statements and proof sections.
Prime claims use random cell points and bit weights, with their true values
computed from source bits. Prime tamper rows cover the value, weights,
each point coordinate, commitments, reduction sections, nested framing,
nested proof, mode changes, truncation, and trailing bytes. Every row requires
`AkitaError::InvalidProof`.

`tests/common/mod.rs` supplies the shared fixtures and forwarding recorder.
It locates proof sections from the bytes exchanged by the oracle rather than
reimplementing the reduction wire grammar. The reduction's own tests cover
its individual proof-of-work nonces. The binary test pins one complete proof
digest per shipped family. `tests/prime_conversion.rs` compares the flattened
polynomial's evaluation with a direct extension-valued evaluation, including
random, zero, one, and basis-coordinate tables and Boolean endpoints.
Ignored `(22,8)` samples report setup, commit, open and verify timings, and
proof-size parts; prime samples give the prime commitment and three-group
nested proof their own lines.

## Conditional guarantees and exclusions

Acceptance authenticates every returned evaluation against the same Y and W,
and the same committed uP when present, conditional on ordinary Akita's
weak-binding and evaluation soundness for the grouped opening. One proof
opens the tables bound before the challenges they depend on. Together with
[the root reduction](labinius-root-reduction.md),
[the lowered relation](labinius-lowered-root.md), and
[the root sumchecks](labinius-root-sumcheck.md), this supplies the required
commitment oracle. Root admission follows
[the setup contract](labinius-setup-contract.md). Akita's assumptions and setup
contract are in [the verifier contract](../docs/verifier-contract.md) and
[security](../book/src/how/security.md). No new numerical oracle error is
assigned here.

Verification does not recompute the clear commitment or establish that an
arbitrary supplied commitment came from a binary source. The composition is
non-hiding and supplies neither a source-extraction theorem nor a completed
Fiat-Shamir composition theorem. It adds no SIS admission beyond the root
and Akita setup contracts. End-to-end tests do not discharge the reduction's
remaining extraction, SIS, frontend-soundness, or random-oracle obligations.
