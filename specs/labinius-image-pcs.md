# LaBinius image-table PCS adapter

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Statement and scope

The opt-in `akita-labinius-pcs` crate commits the clear image of a binary source
under a seed-derived admitted D648/P128 root setup, then opens one evaluation
of the committed image table with ordinary Akita. `ImageConfig` is the existing
full-field `fp128::Dense` configuration and `F` is its coefficient field.
The verifier's public statement is an admitted-root identity, an Akita setup,
one `CommittedGroup<F>`, a coefficient-field point and a claimed value.

This specification uses the uppercase requirement words of BCP 14
([RFC 2119](https://www.rfc-editor.org/rfc/rfc2119),
[RFC 8174](https://www.rfc-editor.org/rfc/rfc8174)).

`ImageProver::commit` MUST call `commit_binary_clear_prepared` with the supplied
prepared matrix, admitted setup and source. The existing kernel checks that
the prepared matrix belongs to that setup. The adapter MUST derive the image
geometry through `LoweredRootLayout`, then call `flatten_image` and import the
result as one dense Lagrange-basis polynomial. Commitment uses the scalar
context with no precommitted groups.

For image entry `e=col*n_A+i` and live coefficient `t<648`, the canonical layout
is `Y[t+1024*e]=images[e].coefficients()[t]`. Each ring's coefficient tail and
final domain padding are zero in an honestly constructed table. The image
domain is the next power of two of `1024*C*n_A`; its log comes from the layout.
The chosen digit base `Bits2` only reaches that existing layout owner. All three
digit bases yield identical image geometry. Multilinear coordinates bind index
bits from lowest to highest, as in the lowered-root relation.

An accepted opening authenticates the claimed evaluation of the table bound
by the public Akita commitment, conditional on Akita's binding and evaluation
soundness. Verification does not recompute the clear image or establish that
an arbitrary supplied commitment was created from a binary source. The image-only
adapter proves no binary source opening and provides no extraction, SIS claim,
or complete Fiat–Shamir composition theorem. The crate also exposes
[the root PCS](labinius-pcs.md), which composes this commitment with the root
reduction and a grouped Akita opening.

## Public API and trusted inputs

`ImageProver::new` takes a `TrustedScheduleCatalog<ImageConfig>` and
`akita_cpu_backend::AkitaProverSetup<F>`. `ImageVerifier::new` takes the same
catalog type and `akita_types::AkitaVerifierSetup<F>`. Constructors prepare
existing CPU or verifier state; proof parsing performs no artifact decoding or
planner search. The catalog's provenance is the caller's responsibility.

`ImageCommitOutput` exposes the `CommittedGroup<F>`, the CPU
`CommitmentHandle<F,F>` and the `BinaryClearCommitment` image.
`ImageEvaluation` borrows a point and carries one coefficient-field value.
`open_on_channel` and `verify_on_channel` borrow an already-active `ClearChannel`.
They MUST leave session termination and EOF to the caller.

API note: `ImageProver::commit<H: SwitchField>` additionally requires
`where H::Source: Sync`. This is the existing prepared kernel's unconditional
bound, including with `parallel` disabled. Both supported source types satisfy
it. No change to `SwitchField` or the existing kernel is needed.

The scalar row MUST match the commitment's complete frozen profile, image log
and polynomial count. A missing row, different profile or image length, invalid
commitment payload, wrong point dimension, or unsupported setup MUST reject
before any parent channel operation or proof byte is read. Verifier setup
preflight MUST check coverage of every slot returned by
`required_setup_prefix_slot_ids_for_schedule` for the selected scalar row,
even when catalog setup sizing conditionally omits prefix enumeration. The
prover MUST validate retained handle ownership,
producer contract, shape and public commitment against the existing backend
admission boundary before deriving the nested session. A rejected statement
does not require the caller to continue using a partially advanced channel.

## Transcript order

Define `LP(x)=u64_le(byte_length(x)) || x`. Public absorption uses the canonical
bytes below. It contributes no proof bytes. Every integer has fixed width and
little-endian encoding. Field coordinates use canonical fixed-width
little-endian encoding. The field modulus identity is the existing canonical
big-endian byte encoding supplied by `field_modulus_be_bytes`.

Both roles MUST run the same `bind_image_statement` function for steps 1–5.

| Step | Operation | Bytes, in order |
| --- | --- | --- |
| 1 | `public` | `LP("akita/labinius/image-pcs/v1")` |
| 2 | `public` | `LP(admitted.identity_bytes::<H>())`, `LP(canonical compressed AkitaSetupDescriptor)`, `LP(ImageConfig::schedule_family_name())`, field modulus bit count as u32, `LP(field modulus identity)`, image log as u32, image length as u64 |
| 3 | `public` | `LP("Y")`, image length as u64, `LP(canonical compressed CommittedGroup)`, point coordinates, claimed value |
| 4 | Scalar row resolution and `public` | Resolve from the trusted catalog and validate the opening-claims layout; absorb `LP(transcript_instance_descriptor::<F,ImageConfig>(..., Lagrange))` |
| 5 | `public`, then one challenge | `LP("akita/labinius/image-akita-opening-session/v1")`, then exactly one `challenge_block` |
| 6 | Nested opening and `message` | Run existing Akita batched prove/verify under the derived session; send/read u64 proof length followed by exactly that many inner proof bytes |
| 7 | Return | The caller continues or checks its own enclosing EOF |

The image verifier resolves the row, validates input and checks setup-prefix
coverage before public absorption, leaving the parent channel untouched on
preflight rejection. Step 4 binds the fully resolved row,
opening layout, basis and grinding plan through the existing descriptor owner.
The commitment is public input and MUST NOT be copied into the proof.

The parent binds the setup descriptor, but not setup-prefix registry payloads,
as in native Akita. The caller must establish prefix commitment provenance when
installing the setup; coverage preflight checks that required slots are present.

The inner Akita session is the raw step-5 domain followed by the 32-byte seed.
It depends on every preceding public statement byte and the caller's prior
transcript state. A continued parent transcript also depends on the inner
proof bytes because the frame and payload pass through `message`.
A second invocation derives its seed from that progressed state.

## Framing, bound and reusable transport

`session::NestedOpeningSession` owns step 5 and step 6. Its generic constructor
takes a resolved trusted row and a configuration, so a later grouped statement
can reuse it with a different domain after binding its complete statement and
instance. The proof framing is independent of the number of claim groups.

The verifier MUST convert the received u64 length to `usize`, reject zero or
any length larger than `expanded_schedule_proof_bound` for the resolved row,
and then allocate the payload buffer with a fallible reservation. The bound
uses the row's exact scalar or grouped lookup key, expanded schedule and
configuration policy. A planner estimate MUST NOT be used as a parser bound.
The prover applies the same bound before sending the proof.

The inner Akita verifier consumes the exact framed payload and enforces its
own completion and EOF. The adapter MUST NOT check enclosing EOF: a caller
that ends its protocol MUST do that itself to reject trailing bytes.
Framing failures return `AkitaError::InvalidProof`. Native Akita replay errors
are preserved; the existing verifier can classify some payload-dependent
arithmetic or decoding failures as `InvalidInput`, rather than `InvalidProof`.
Catalog and setup failures retain their own error variants.

## Feature isolation and regression evidence

The crate root is gated by `labinius`; with that feature off the crate exports
nothing and activates no LaBinius dependency feature. `labinius` forwards only
to the two existing LaBinius crates. Transcript choices and `parallel` forward
to their corresponding dependencies, and no feature forwards `dev-protocol`.
No existing schedule artifact or protocol implementation changes.

Test support constructs caller-owned scalar catalogs with the planner and
admits them through `ValidatedScheduleCatalog` and `TrustedScheduleCatalog`.
The planner is a dev-dependency; the dependency guard walks normal edges for
this crate and forbids it in the runtime graph, including with all features.

Regression tests cover both host fields, image logs 10 and 11, all three digit
bases, canonical padding, direct scalar Akita commitment and proof agreement,
the expanded proof bound, parent-prefix binding, post-proof challenges and
progressed invocations. The first `(22,8,128)` profile's image log 18 is checked
by canonical shape derivation without performing that image action.
Malformed framing, inner message regions, commitment bytes and right-sized
random payloads are tested under `catch_unwind`. For long proofs, truncation
coverage samples 32 evenly spaced payload prefixes plus every framing prefix
and payload endpoints; it does not enumerate every long-proof prefix.
