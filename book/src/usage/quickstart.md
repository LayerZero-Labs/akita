# Your first proof

The quickest way to understand Akita is to run one complete commitment and
opening proof. The repository includes a checked example that uses the real
production API from setup through verification.

Run this command from the repository root:

```bash
cargo run -p akita-pcs --release --example quickstart
```

The first release build compiles the complete proving stack. Later runs reuse
those build results. A successful run ends with output of this form:

```text
Akita proof verified (... bytes)
```

The complete source is
[`crates/akita-pcs/examples/quickstart.rs`](https://github.com/LayerZero-Labs/akita/blob/main/crates/akita-pcs/examples/quickstart.rs).
The normal Cargo checks compile this example with the public API. The chapter
therefore stays tied to code that works.

## The statement being proved

The example uses `fp128::Dense`, Akita's direct configuration for arbitrary
values in its 128 bit field. It creates a table with $2^{14}$ field elements.
That table represents a multilinear polynomial with 14 variables.

The example then chooses a point with 14 coordinates and evaluates the
multilinear polynomial there. This value is the public claim. In a larger proof
system, the host protocol usually produces the table, point, and claimed value.

```rust
type Config = fp128::Dense;
type F = fp128::Field;

const NUM_VARS: usize = 14;

let polynomial = DensePoly::from_field_evals(NUM_VARS, &evaluations)?;
let evaluation = evaluate_multilinear(&evaluations, &point);
```

The helper that computes `evaluation` is independent of the Akita prover. It
exists to give the verifier a concrete public value to check.

## Load schedules and build reusable setup

The configuration determines the expected schedule family and policy. The
application loads approved artifact bytes from its own storage; Akita validates
them before setup or proof work begins. `setup_prover` then sizes public matrix
data from the exact rows in that catalog.

```rust
let artifact_bytes = std::fs::read("parameters/fp128_dense.aks")?;
let scheme = AkitaCommitmentScheme::<Config>::from_schedule_artifact(
    &artifact_bytes,
)?;
let setup = scheme.setup_prover(NUM_VARS, 1)?;
let backend = CpuBackend::DEFAULT;
let prepared = backend.prepare_setup(&setup)?;
let stack = UniformProverStack::uniform(
    &backend,
    &prepared,
    setup.expanded.as_ref(),
)?;
```

The prepared backend holds reproducible compute state such as transformed
matrix prefixes. Applications should reuse it across commitments and proofs.
Setup is public. Akita does not require a secret trapdoor or a trusted setup
ceremony.

## Commit to the polynomial

One call commits to one group of polynomials. This example has one polynomial
and no earlier groups.

```rust
let commit_output = scheme.commit(
    &setup,
    std::slice::from_ref(&polynomial),
    &stack,
    GroupContext::scheduler_without_precommitted_groups(),
)?;
```

The call returns two values:

- `committed_group` is public. The verifier receives it.
- `hint` is private prover data. The prover keeps it with the polynomial.

The group context tells Akita which catalog row to use for the commitment. A
later chapter explains how earlier commitment groups change this context.

## Assemble the opening claim

An opening claim joins the point, claimed value, and commitment. The prover also
supplies the original polynomial and its private hint.

```rust
let prover_claims = OpeningClaims::from_groups(vec![
    PolynomialGroupClaims::new(
        point.clone(),
        vec![evaluation],
        commit_output.committed_group.clone(),
    )?,
])?;

let polynomial_group = [&polynomial];
let prover_data = SelectedProverOpeningData::from_committed_claims::<Config>(
    prover_claims,
    vec![commit_output.hint],
    vec![&polynomial_group],
    scheme.schedules(),
)?;
let selection = prover_data.selection();
```

`SelectedProverOpeningData` checks that the public claims, commitment profiles,
private hints, and polynomial groups have the same order and shape. It also
selects the exact trusted catalog row for the complete batch.

## Produce the proof

The prover starts a transcript with an application specific domain. This domain
separates the proof from every other protocol that may use the same transcript
construction.

```rust
const TRANSCRIPT_DOMAIN: &[u8] = b"akita/book/quickstart/v1";

let mut prover_transcript = AkitaTranscript::<F>::unbound_prover(TRANSCRIPT_DOMAIN);
let proof = scheme.batched_prove(
    &setup,
    prover_data,
    &stack,
    &mut prover_transcript,
    BasisMode::Lagrange,
)?;
```

`BasisMode::Lagrange` means that the committed table contains values on the
Boolean cube. This is the standard representation for multilinear extensions
in proof systems.

## Encode and decode the proof

Applications send bytes, not Rust objects. The example therefore performs a
real compressed serialization round trip before verification.

```rust
let proof_shape = proof.shape();
let mut proof_bytes = Vec::new();
proof.serialize_compressed(&mut proof_bytes)?;

let decoded_proof = AkitaBatchedProof::<F, F>::deserialize_compressed(
    &mut std::io::Cursor::new(&proof_bytes),
    &proof_shape,
)?;
```

The shape gives the decoder explicit limits and structure. A deployment should
derive or authenticate that shape from its supported configuration and public
statement before allocating for an incoming proof.

## Verify with fresh public state

The verifier needs public setup, the commitment, the point, the claimed value,
and the selected schedule row. It does not receive the polynomial or commitment
hint.

```rust
let verifier_setup = scheme.setup_verifier(&setup)?;
let verifier_claims = OpeningClaims::from_groups(vec![
    PolynomialGroupClaims::new(
        point,
        vec![evaluation],
        &commit_output.committed_group,
    )?,
])?;
let statement = GroupBatchStatement::new(selection, verifier_claims)?;

let mut verifier_transcript =
    AkitaTranscript::<F>::unbound_verifier(TRANSCRIPT_DOMAIN);
scheme.batched_verify(
    &decoded_proof,
    &verifier_setup,
    &mut verifier_transcript,
    statement,
    BasisMode::Lagrange,
)?;
```

The prover and verifier each create a fresh transcript for their side of the
protocol. Akita binds the complete public statement before deriving proof
challenges, so a change to the group order, point, value, commitment,
configuration, or schedule causes verification to fail.

## What to change next

The example fixes its choices so that the lifecycle stays easy to follow. A
real integration will choose them from the host protocol.

- Use [Choosing a configuration](./configuration.md) to select the field and
  polynomial representation.
- Use [Integrating the PCS](./integration.md) for several polynomials, several
  opening points, or earlier commitment groups.
- Use [Verifier only integration](./verifier-only.md) when verification must
  compile without the prover backend.
- Use [Integrating with a proof system](./integrations.md) to connect Akita to
  a host protocol or recursive verifier.
- Use [Profiling](./profiling.md) to measure a production size workload.
