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
Akita proof verified
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
let backend = std::sync::Arc::new(CpuBackend::<Config>::new(
    setup.expanded.clone(),
    scheme.schedules(),
)?);
```

The prepared backend holds reproducible compute state such as transformed
matrix prefixes. Applications should reuse it across commitments and proofs.
Setup is public. Akita does not require a secret trapdoor or a trusted setup
ceremony.

## Commit to the polynomial

One call commits to one group of polynomials. This example has one polynomial
and no earlier groups.

```rust
let source = backend.import_source(vec![polynomial])?;
let commit_output = backend.commit(
    &source,
    GroupContext::scheduler_without_precommitted_groups(),
)?;
```

The call returns two values:

- `committed_group` is public. The verifier receives it.
- `private_handle` retains the exact immutable source and commitment parameters.

The group context tells Akita which catalog row to use for the commitment. A
later chapter explains how earlier commitment groups change this context.

## Assemble the opening claim

An opening claim joins the point, claimed value, and commitment. The prover also
supplies the reusable commitment handle. It cannot substitute another polynomial.

```rust
let prover_claims = OpeningClaims::from_groups(vec![
    PolynomialGroupClaims::new(
        point.clone(),
        vec![evaluation],
        commit_output.committed_group.clone(),
    )?,
])?;

let prover_data = SelectedProverOpeningData::from_committed_claims::<Config>(
    prover_claims,
    vec![commit_output.private_handle.clone()],
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

let proof = scheme.batched_prove(
    &setup,
    prover_data,
    &backend,
    TRANSCRIPT_DOMAIN,
    BasisMode::Lagrange,
)?;
```

`BasisMode::Lagrange` means that the committed table contains values on the
Boolean cube. This is the standard representation for multilinear extensions
in proof systems.

## Transport the proof

The proof is already the canonical Spongefish argument byte string. Store or
send it directly; the public schedule bounds every message the verifier reads.

```rust
let proof_bytes: Vec<u8> = proof;
```

Akita resolves the authenticated schedule before reading proof messages and
rejects truncation, noncanonical atoms, and trailing bytes.

## Verify with fresh public state

The verifier needs public setup, the commitment, the point, the claimed value,
and the selected schedule row. It does not receive the polynomial or private
prover state.

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

scheme.batched_verify(
    &proof_bytes,
    &verifier_setup,
    TRANSCRIPT_DOMAIN,
    statement,
    BasisMode::Lagrange,
)?;
```

Akita constructs fresh native prover and verifier states and binds the complete
public statement before deriving proof
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
