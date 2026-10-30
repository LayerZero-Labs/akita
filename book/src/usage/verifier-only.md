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

The default `akita-verifier` features enable the Keccak transcript backend.
Schedule rows are external data, not features. The host loads approved artifact
bytes, constructs a `TrustedScheduleCatalog<Cfg>`, and supplies that config-bound
catalog to every verification call.

The verifier package stays small when the consumer depends only on the crates
above. The repository checks in CI that `akita-verifier` has no dependency on
`akita-pcs`, `akita-prover`, `akita-planner`, or `akita-setup`.

## Inputs to verification

The verifier receives these public values:

- A `PinnedVerifierSetup` issued by the release loader for an anchored
  parameter set, which carries the setup and the catalog.
- A schedule selection produced with the proof.
- Ordered commitments, points, and claimed values.
- Spongefish argument bytes.
- The application transcript domain and basis mode.

The setup comes from a pinned release, not from the proof bundle. The [proof
artifacts guide](./proof-artifacts.md) explains the rest of the bundle.

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

Load the verifier setup from a release, build one `AkitaVerifier` from it, and
pass the application session label to its verification method.

```rust
let verifier = akita_verifier::AkitaVerifier::from_pinned(pinned_setup)?;
verifier.batched_verify(
    &proof_bytes,
    TRANSCRIPT_DOMAIN,
    statement,
    BasisMode::Lagrange,
)?;
```

A full load admits every catalog row of its parameter set: those whose
opening key fits the set limits. The loader derives the public matrix those
rows read and checks that the setup admits exactly them. Construction prepares
the terminal matrix transforms of those rows once. Verification never mutates
the verifier, so one instance can serve many proofs and threads. A verifier
that checks a single proof uses a single-row load, which admits and prepares
only that proof's row.

The verifier resolves the explicit schedule selection in the pinned catalog.
It does not run the planner. It then binds the instance descriptor, validates
the public claims, and replays every fold through terminal verification.

## Supply verifier setup

A verifier trusts only the `(bundle_id, set name)` pairs of its compiled trust
anchor. Akita's anchor is `akita_anchors::AKITA_ANCHOR`; an integration that
publishes its own release compiles its own anchor as a constant. The set name
comes from the integration's own configuration, never from transported bytes.

```rust
use akita_anchors::{AKITA_ANCHOR, AKITA_V1_MANIFEST};
use akita_config::preprocessing::PrefixCommitmentSource;
use akita_types::preprocessing::{open_anchored_manifest, DirectoryReleaseFiles, ReleaseKind};

let release = open_anchored_manifest(
    &AKITA_ANCHOR,
    ReleaseKind::Akita,
    AKITA_V1_MANIFEST,
    &["fp128-dense-recursive"],
)?;
let files = DirectoryReleaseFiles::new(release_dir);
let pinned_setup = akita_verifier::preprocessing::load_verifier_setup::<Cfg>(
    &release,
    &files,
    "fp128-dense-recursive",
    None, // every admitted row; a single-row verifier passes its selection
    None, // no program rows
    PrefixCommitmentSource::Registry,
)?;
```

The loader authenticates every byte it uses against the anchored manifest;
[pinned releases](../how/verification.md#pinned-releases) lists its checks.
Release files are therefore untrusted transport: a directory, the
`akita-setup` release cache, embedded bytes, or downloaded release assets all
work.

A host that prefers not to read registry files may derive the commitments from
the seed instead, with `PrefixCommitmentSource::SeedDerived` and the CPU
reference deriver from `akita-cpu-backend`. The loader compares each derived
commitment with the manifest, so both modes issue the same verifier.
Seed-derived mode costs one commitment derivation per profile used, so hosts
that load often should use registry mode, and a guest must not use it.

A deployment with its own seed, catalogs, or limits writes a description
under its own `params/`, generates its release with `akita-preprocess`, and
passes its own anchor to the same loader.

## Handle rejection as a normal result

The verifier treats every input as possibly malformed. Proof bytes,
commitments, claims, and opening points may also be adversarial, and the
verifier's soundness does not depend on who produced them. The verifier setup
and the schedule catalog are different: the loader authenticates them against
the compiled anchor, as described [above](#supply-verifier-setup), and
soundness assumes the anchored release was generated honestly. Their bytes are
still decoded defensively. In every case the verifier returns `AkitaError` or
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
  --features transcript-keccak \
  -- -D warnings
```

The first command confirms that the verifier crate has not gained prover or
planner dependencies. The second checks the narrow feature graph used by a
minimal verifier build.
