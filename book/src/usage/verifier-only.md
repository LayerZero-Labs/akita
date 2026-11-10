# Verifier only integration

Akita provides a separate verifier package for hosts that want the smallest
trusted dependency path. This package contains the public types and replay
logic needed to check an approved proof schedule. Polynomial backends,
commitment hints, setup generation, and planner search remain in the proving
packages.

This boundary is useful for a zkVM guest, a network service, or any process that
only receives proofs.

## Dependencies

Pin every crate to the same Akita revision. A verifier consumer needs these
packages:

```toml
[dependencies]
akita-verifier = { git = "https://github.com/LayerZero-Labs/akita", rev = "<commit>" }
akita-config = { git = "https://github.com/LayerZero-Labs/akita", rev = "<commit>", default-features = false }
akita-types = { git = "https://github.com/LayerZero-Labs/akita", rev = "<commit>", default-features = false }
akita-transcript = { git = "https://github.com/LayerZero-Labs/akita", rev = "<commit>", default-features = false }
akita-serialization = { git = "https://github.com/LayerZero-Labs/akita", rev = "<commit>" }
akita-params = { git = "https://github.com/LayerZero-Labs/akita", rev = "<commit>", default-features = false }
akita-error = { git = "https://github.com/LayerZero-Labs/akita", rev = "<commit>" }
```

`akita-params` supplies `BasisMode` and `ChunkedWitnessCfg`. `akita-error`
supplies `AkitaError`.

The default `akita-verifier` features enable the Keccak transcript backend.
Schedule rows are external data, not features. The host loads approved artifact
bytes, constructs a `TrustedScheduleCatalog<Cfg>`, and supplies that config-bound
catalog to every verification call.

The verifier package stays small when the consumer depends only on the crates
above. The repository checks in CI that `akita-verifier` has no dependency on
`akita-pcs`, `akita-prover`, `akita-cpu-backend`, or `akita-planner`.

## Inputs to verification

The verifier receives these public values:

- An `AkitaVerifierSetup` for the configuration and supported schedule.
- The approved external schedule artifact or its validated catalog.
- A schedule selection produced with the proof.
- Ordered commitments, points, and claimed values.
- Spongefish argument bytes.
- The application transcript domain and basis mode.

The host should place them in one versioned public artifact or authenticate the
setup separately. The [proof artifacts guide](./proof-artifacts.md) explains the
bundle.

## Keep proof bytes opaque

Do not decode the proof into a host-side Rust object. Pass the argument
bytes to verification; the approved schedule supplies all message counts and
allocation bounds.

Commitments and verifier setup use their own canonical decoders and validation
rules. Decode every public object before constructing the statement.

## Build the public statement

Each group owns one point and one value for each committed polynomial.

```rust
let claims = OpeningClaims::from_groups(vec![
    PolynomialGroupClaims::new(
        opening_point,
        claimed_values,
        &committed_group,
    )?,
])?;

let statement = GroupBatchStatement::new(
    schedule_selection,
    claims,
)?;
```

For a multi group proof, preserve the exact order used by the prover. The final
group is last, and every earlier group is a precommitted group.

## Verify directly

Build one `AkitaVerifier` from the verifier setup and the trusted catalog, then
pass the application session label to its verification method.

```rust
let verifier = akita_verifier::AkitaVerifier::new(verifier_setup, catalog)?;
verifier.batched_verify(
    &proof_bytes,
    TRANSCRIPT_DOMAIN,
    statement,
    BasisMode::Lagrange,
    akita_verifier::SetupPolicy::Check(None),
)?;
```

Construction fixes the catalog rows this verifier admits: those whose opening
key fits the setup descriptor and whose direct matrix uses fit the setup's
public matrix. It prepares the terminal matrix transforms of those rows once.
Verification never mutates the verifier, so one instance can serve many proofs
and threads. A verifier that checks a single proof, such as a zkVM guest, uses
`AkitaVerifier::for_selection` to admit and prepare only that proof's row.

The verifier resolves the explicit schedule selection in the supplied catalog.
It does not run the planner. It then binds the instance descriptor, validates
the public claims, and replays every fold through terminal verification.

The repository's [quickstart example](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-pcs/examples/quickstart.rs)
uses this verifier on the proof bytes that it produces. Cargo
compiles that call as part of the normal example checks.

## Verify inside a host transcript

A host that keeps its own transcript calls `batched_verify_embedded` in place
of `batched_verify`. It passes its own `akita_transcript::VerifierChannel` and
no session label.

```rust
let pending = verifier.batched_verify_embedded(
    &mut state,
    statement,
    BasisMode::Lagrange,
    akita_verifier::SetupPolicy::Check(None),
)?;
// ... the host replays its remaining messages ...
pending.check_eof(state)?;
```

A successful return is not acceptance. The call returns a `PendingEof`, and
only `PendingEof::check_eof` on the same state accepts the proof. Two rules
follow:

- Call `check_eof` after the host's last message, with the state that Akita
  verified on.
- If the call uses a setup accumulator, the accumulator is settled when
  `batched_verify_embedded` returns. Discard it if `check_eof` then fails.

[Embed Akita in a host transcript](./proof-artifacts.md#embed-akita-in-a-host-transcript)
shows the prover side and lists the host's duties.

## Supply verifier setup

The proving environment normally creates verifier setup before packaging or
serving proofs.

For a first integration, use:

```rust
let verifier_setup = scheme.setup_verifier(&prover_setup)?;
```

For a fixed production schedule, use
`setup_verifier_for_schedule`. It retains only the public matrix prefix that
the selected verifier path reads directly, plus the complete setup prefix
commitment registry.

The verifier package consumes this artifact. It does not need setup generation
code. A host can authenticate the package by digest, distribute it with the
application, or regenerate and validate it before placing it in the verifier
environment.

## Shrink a single-proof verifier

A verifier inside a zkVM guest pays for every decoded byte and every
arithmetic step. Akita has three ways to reduce what such a verifier carries
and prepares. None of them changes proof bytes or the transcript. Each one
replaces work that the verifier would do with bytes that the verifier is told
to trust, so each one needs the deployment to bind those bytes to a trusted
source. [What the verifier trusts](../how/verification.md#what-the-verifier-trusts)
states the model.

| Technique | The guest skips | The guest still checks | The caller must guarantee |
| --- | --- | --- | --- |
| Selected-row catalog view | Decoding of schedule rows it does not use | Format, protocol epoch, configuration binding, every carried row | Where the view bytes came from |
| In-place setup view | The copy of the public matrix and the range check of each coefficient | The setup descriptor and the setup-prefix registry | The bytes are canonical and derived from the setup seed |
| Trusted terminal cache | The terminal matrix transform | Setup identity, schedule row identity, geometry, lengths | The cache was derived from the named setup |

### Carry only the selected schedule rows

A schedule catalog has one row for each supported proof shape. A guest that
checks one proof needs one row. The host calls
`ValidatedScheduleCatalog::to_verifier_view` with the schedule selections it
wants to keep. The result is a byte string that holds those rows in full and
every other row only as its 32-byte digest. The view therefore keeps the
transcript identity of the complete catalog.

The guest loads the view with `TrustedScheduleCatalog::from_verifier_view`:

```rust
// Host, with the complete validated catalog.
let view: Vec<u8> = catalog.to_verifier_view(&[selection])?;

// Guest.
let catalog = TrustedScheduleCatalog::<Cfg>::from_verifier_view(&view, chunking)?;
```

Loading checks the format and protocol epoch, the configuration binding, the
invariants of every carried row, and that each carried row is in the carried
digest list. A view from another epoch is rejected with `InvalidSetup`. A row
that the view omits does not resolve, so a proof that names it is rejected. A
view serves verification with an existing setup. It cannot size a new setup.
The digest list is as trusted as the view bytes, so the caller owns their
provenance.

### View the setup in place

The usual setup decoder copies the public matrix and checks that every field
element is canonical. `AkitaVerifierSetup::borrow_trusted` does neither. It
reads the matrix where it lies in the input bytes.

The input is a `TrustedBytes` value over `'static` bytes of an uncompressed
serialized verifier setup. The only way to make one is the `unsafe`
constructor `TrustedBytes::new`. The caller thereby states, at one place in
the program, that every field word in those bytes is canonical. A guest's own
program image is the intended source.

The view still validates the setup descriptor and the setup-prefix registry.
It takes the matrix as derived from the setup seed without deriving it again.
The first matrix coefficient must be aligned for the field type;
`AkitaExpandedSetup::coefficient_offset` returns its byte offset so that the
host can place the bytes correctly.

### Install a prepared terminal matrix

`AkitaVerifier::for_selection` admits one schedule row and prepares only that
row's terminal matrix. Its last argument can supply the prepared matrix:

```rust
// Host.
let cache: Vec<u8> = akita_verifier::build_riscv64_terminal_ntt_cache(
    &verifier_setup,
    schedule,
    selection.row_digest,
)?;

// Guest.
let verifier = AkitaVerifier::for_selection(
    verifier_setup,
    catalog,
    selection,
    Some(TrustedTerminalCache::Decode(&cache)),
)?;
```

`TrustedTerminalCache::Decode` copies the residues and rejects any that are
out of range. `TrustedTerminalCache::View` reads `'static` bytes in place
without a copy or a range pass. Pass `None` to let the verifier compute the
transform itself.
[Per-level replay](../how/verification.md#per-level-replay) describes what the
verifier checks about a cache and what it cannot check. The recursion guest in
`profile/akita-recursion/guest/src/integration.rs` uses this path with a cache
that is bound to the guest program.

## Handle rejection as a normal result

The verifier treats every input as possibly malformed. Proof bytes,
commitments, claims, and opening points may also be adversarial, and the
verifier's soundness does not depend on who produced them. The verifier setup
and the schedule catalog are different: the host supplies them, as described
[above](#supply-verifier-setup), and soundness assumes they were derived as
claimed. The host either derives them itself or authenticates bytes that a
party it trusts derived. Their bytes are still decoded defensively. In every
case the verifier returns `AkitaError` or `SerializationError` when decoding
or proof replay fails.

Match on the error at the host boundary and reject the request. Do not retry
verification with another configuration or reconstructed schedule. A valid
proof names one configuration, one catalog row, and one ordered public
statement.

The verifier code follows a strict no panic contract for malformed public
input. CI checks the dependency boundary, and the test suite exercises shape,
schedule, transcript, serialization, and proof rejection paths. The
[verification chapter](../how/verification.md) describes the full contract and
the checks applied at each stage.

## Verify the package boundary

Run these checks when changing a verifier integration:

```bash
scripts/check-crate-deps.sh akita-verifier
cargo clippy -p akita-verifier --all-targets --release \
  --no-default-features \
  --features transcript-keccak \
  -- -D warnings
```

The first command confirms that the verifier crate has not gained prover or
planner dependencies. The second checks the narrow feature graph used by a
minimal verifier build.
