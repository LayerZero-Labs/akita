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
```

The default `akita-verifier` features enable the Blake2b transcript backend.
Schedule rows are external data, not features. The host loads approved artifact
bytes, constructs a `TrustedScheduleCatalog`, and supplies that catalog to every
verification call.

The verifier package stays small when the consumer depends only on the crates
above. The repository checks in CI that `akita-verifier` has no dependency on
`akita-pcs`, `akita-prover`, `akita-planner`, or `akita-setup`.

## Inputs to verification

The verifier receives these public values:

- An `AkitaVerifierSetup` for the configuration and supported schedule.
- The approved external schedule artifact or its validated catalog.
- A schedule selection produced with the proof.
- Ordered commitments, points, and claimed values.
- An expected proof shape.
- Compressed proof bytes.
- The application transcript domain and basis mode.

The host should place them in one versioned public artifact or authenticate the
setup separately. The [proof artifacts guide](./proof-artifacts.md) explains the
bundle.

## Decode with the expected shape

```rust
let proof = AkitaBatchedProof::<F, E>::deserialize_compressed(
    &mut std::io::Cursor::new(&proof_bytes),
    &expected_shape,
)?;
```

The expected shape must come from the selected row in the host's approved
catalog. It gives the decoder concrete bounds before it allocates nested proof
objects.

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

Create a verifier side transcript with the application session label, then call
the top level verifier entry point.

```rust
let mut transcript =
    AkitaTranscript::<F>::unbound_verifier(TRANSCRIPT_DOMAIN);

akita_verifier::batched_verify::<Config, _>(
    &proof,
    &verifier_setup,
    &catalog,
    &mut transcript,
    statement,
    BasisMode::Lagrange,
)?;
```

The verifier resolves the explicit schedule selection in the supplied catalog.
It does not run the planner. It then binds the instance descriptor, validates
the public claims, and replays every fold through terminal verification.

The repository's [quickstart example](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-pcs/examples/quickstart.rs)
uses this direct verifier call after producing and decoding a proof. Cargo
compiles that call as part of the normal example checks.

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

## Handle rejection as a normal result

All verifier facing proof, setup, schedule, claim, and transcript data is
untrusted public input. The verifier returns `AkitaError` or
`SerializationError` when decoding or proof replay fails.

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
  --features transcript-blake2b \
  -- -D warnings
```

The first command confirms that the verifier crate has not gained prover or
planner dependencies. The second checks the narrow feature graph used by a
minimal verifier build.
