# LaBinius root polynomial commitment

Status: implemented
Book-chapter: book/src/how/architecture.md
Tracking: https://github.com/LayerZero-Labs/akita/issues/45

## Statement and composition

The opt-in `akita-labinius-pcs` crate supplies a succinct, non-hiding
polynomial commitment for a binary multilinear polynomial represented by
switch-field source words. The public statement consists of an admitted root
setup, one Akita setup, a padded image commitment Y, a host-field point r and
a claimed host-field value t. The host is either F128 or F192, as defined by
`SwitchField`; the coefficient field F is the ordinary fp128 Akita field.

The prover computes Y with the canonical image commitment path in
[the image adapter](labinius-image-pcs.md). To open `f(r)=t`, it runs
[the root reduction](labinius-root-reduction.md) with an Akita-backed oracle.
The oracle commits the reduction's complete padded response digit table W and
authenticates its two returned evaluations with one grouped Akita opening.
Y is the precommitted first group and W is the final group. Their points and
table sizes may differ. Verification replays the same composition and checks
standalone EOF. The response table itself does not appear in the proof.

This specification uses the uppercase requirement words of BCP 14
([RFC 2119](https://www.rfc-editor.org/rfc/rfc2119),
[RFC 8174](https://www.rfc-editor.org/rfc/rfc8174)).

## Public API and trusted inputs

`RootPcsProver<C>` owns the admitted root, prepared matrix, both trusted catalogs,
one Akita prover setup and one CPU backend. `RootPcsVerifier<C>` owns the root,
both catalogs and prepared Akita verification state. `C: DigitConfig` fixes
the digit base. Both constructors take the root, scalar image catalog, digit
catalog and their corresponding Akita setup. No constructor loads schedule
artifacts or runs a planner; catalogs are trusted by the caller.

`commit<H>` takes switch-field source words and returns `ImageCommitOutput`.
`open<H>` takes those words, that retained output, a host point and value, and
returns standalone proof bytes. Both require `H::Source: Sync` to use the
prepared image kernel. `verify<H>` takes the public `CommittedGroup<F>`, host
point and value, and proof bytes. It succeeds only after inner verification
and outer EOF. The public `oracle` methods expose single-use prover and
verifier oracles for callers that own a root reduction channel.

Both catalogs MUST be provisioned on one setup. Constructors compute
`SetupRequirements::from_catalog` at the setup's capacity bounds and `union`
the requirements, then require the setup matrix to cover that union. They
also resolve the exact selected image and grouped rows and check their setup
admission and coverage of every required setup-prefix slot before any parent
channel operation or proof byte is read. Both constructors check the selected
scalar image row and the grouped digit row that is replayed. Slot requirements
come from `required_setup_prefix_slot_ids_for_schedule`, independently of the
conditional prefix enumeration in `SetupRequirements::from_catalog`. Missing
rows, prefix slots and insufficient setups return errors. The grouped key MUST
contain the exact scalar Y profile, as
created by `PrecommittedProducer::try_new` with
`ImageConfig::committed_source_contract()` during offline planning. A matching
table length alone is insufficient. The planner is only a test dependency.

The root prover reuses `ImageProver::commit` on its backend. Retained handles
from other backend owners MUST reject before transcript operations. Existing
image-only channel entry points remain available with their original format.
Enabling `labinius` makes the extension available; selecting these types and
catalogs selects the extension. Without that feature the crate exports nothing.
No feature enables or forwards `dev-protocol`.

## Digit configurations and storage

| Configuration | Base b | Signed source bits | Opening bound | Stable family |
| --- | ---: | ---: | ---: | --- |
| `Digits1` | 1 | 2 | 128 | `labinius_digits_b1` |
| `Digits2` | 2 | 3 | 128 | `labinius_digits_b2` |
| `Digits4` | 4 | 5 | 128 | `labinius_digits_b4` |

Each configuration follows `DenseBounded`: dense source, balanced signed
decomposition, the same field, SIS modulus profile, ring challenge hook and
basis search ranges. `log_commit_bound=b+1` denotes a signed interval that
contains every honest unsigned digit in `[0,2^b)`. `log_open_bound=Some(128)`
allows opening witnesses to contain arbitrary coefficient-field elements.

An accepted grouped opening gives, under Akita's weak-binding and evaluation
soundness, consistent coefficient-field tables for Y and W. The root combined
sumcheck constrains that W table to the unsigned digit alphabet `[0,2^b)` at
every Boolean position, including padding. The `DenseBounded` commit bound is
an honest-prover representability condition; it does not enforce the alphabet.
It admits honest digits while Akita's conditional guarantees provide consistent
tables for the root reduction.

W is imported through `DensePoly::<F>::from_field_evals`. Its field allocation
costs 16 bytes per digit, in addition to the reduction's original digit bytes
and the dense source's small signed-byte mirror. There is no public byte-table
source. At the first root profile with two-bit digits, W has `2^25` entries,
so the field table alone costs 512 MiB. Reduction and Akita workspaces add to
this cost. Succinctness describes transmitted proof size, not prover memory
or a sublinear scan of the materialized public root setup.

## Oracle transcript

Define `LP(x)=u64_le(byte_length(x)) || x`. The reduction binds the admitted
root identity and base before calling the image oracle, and binds the host
point and value in the same statement phase. It already fixes both evaluation
claims before calling discharge. Oracle methods MUST run exactly once, in the
following order. On entry to any call the oracle becomes terminally failed;
only a successful image or response operation advances to the next live state.
An error, repeated call or out-of-order call leaves it failed. Every later call
returns `AkitaError::InvalidProof` without touching the parent channel.

| Call and timing | Operation | Bytes or guarantee |
| --- | --- | --- |
| `bind_image`, before frontend challenges | Validate Y against the scalar row; `public` | `LP("akita/labinius/root-pcs/v1")`, `LP(canonical compressed AkitaSetupDescriptor)`, `LP(image family)`, `LP(digit family)`, `LP(canonical compressed Y CommittedGroup)` |
| `commit_response` / `bind_response`, after fold challenges and before alpha, xi, gamma | Commit W on the same backend with `GroupContext::scheduler_with_precommitted_groups`; `message` | u64 little-endian canonical W byte length, then the canonical W `CommittedGroup` bytes; its profile MUST equal the grouped row's final-group profile |
| `discharge`, after w_eval and y_eval | Build `[Y,W]` claims; `public` | `LP(transcript_instance_descriptor::<F,C>(..., Lagrange))`, then Y point coordinates and value followed by W point coordinates and value, each in canonical fixed-width little-endian F encoding |
| Nested session derivation | `public`, then one challenge | `LP("akita/labinius/root-akita-opening-session/v1")`, then one parent `challenge_block`; session is domain concatenated with that 32-byte seed |
| Grouped opening transport | Existing Akita batched prove / verify; `message` | u64 little-endian inner proof length, then exactly that many inner proof bytes |

The parent binds the Akita setup descriptor, but does not bind the payloads of
the setup-prefix registry. This is inherited from native Akita; callers remain
responsible for installing prefix commitments with the required setup provenance.
Prefix-slot coverage is checked during setup preflight.

The Akita instance descriptor covers the selected row, setup, decomposition,
layout, basis and grinding plan. It does not contain actual points or values,
so discharge MUST bind them explicitly again in the parent. The inner Akita
proof also binds its commitments and evaluation statement. The response
commitment is absorbed before all coefficient challenges. Nested session
challenges follow both evaluation messages. All oracle proof bytes are
absorbed by the parent channel, so later challenges depend on the inner proof.

## Framing and allocation bounds

W's announced byte length MUST equal the exact canonical size implied by the
trusted grouped final profile before allocating its buffer. There is no public
profile-only serialized-size function. The adapter asks
`CommittedGroup::compressed_size` on an empty-payload sizing value for the
fixed header, then adds the checked product of the terminal coefficient count
and F's compressed field size. The count comes from
`CompressionChainPlan::for_complete_source` and the profile's outer slice,
rank and dimension, exactly as in the canonical commitment validator. This
uses no hard-coded header size. The actual commitment MUST decode exactly,
have the trusted profile, and reserialize to identical bytes. Any failure at
this response framing or parsing boundary is `AkitaError::InvalidProof`.

The inner proof length MUST be nonzero and at most
`expanded_schedule_proof_bound` for the already-resolved row and digit policy.
`NestedOpeningSession` owns this framing and bound. A planner estimate is not
a parser bound. Buffer reservations are fallible; arithmetic uses
`akita_error::checked`. Native inner verifier errors are preserved. The outer
byte entry point rejects truncation and trailing bytes. No proof-supplied row,
planner search, artifact decoding, or unbounded proof length is accepted.

If R is `root_reduction_wire_size`, K the exact serialized W size, and A the
actual grouped Akita proof size, the complete standalone proof size is
`R + 8 + K + 8 + A`. Public binding bytes and session challenges contribute
zero proof bytes. There is one grouped proof, with no separate Y proof.

## Conditional guarantees and exclusions

Acceptance authenticates both returned coefficient-field evaluations against
consistent Y and W coefficient-field tables, conditional on ordinary Akita's
weak-binding and evaluation soundness. Together with the assumptions and local
soundness arguments in
[the root reduction](labinius-root-reduction.md),
[the lowered relation](labinius-lowered-root.md), and
[the root sumchecks](labinius-root-sumcheck.md), this supplies the concrete
commitment oracle required by that reduction. Root admission follows [the setup contract](labinius-setup-contract.md).
Akita's assumptions and setup contract are described in
[the verifier contract](../docs/verifier-contract.md)
and [security](../book/src/how/security.md). No new numerical oracle error is
assigned here.

This composition is non-hiding. It claims neither a source-extraction theorem
nor a completed Fiat–Shamir composition theorem. It supplies no additional SIS
admission beyond the existing root and Akita setup contracts. In particular,
passing end-to-end tests does not close the root reduction's stated extraction,
SIS reduction, frontend soundness or random-oracle obligations.
