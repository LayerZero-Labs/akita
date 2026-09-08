# Integrating the PCS

Akita gives a host proof system a clear boundary between private proving work
and public verification. The prover owns the large polynomial tables and the
compute state used to open them. The verifier receives commitments, claimed
values, and one compact proof.

This boundary is designed for real integrations. A host can prepare setup once,
reuse commitments across later claims, batch several polynomial groups into one
opening, and carry the verifier into a package that has no prover backend or
runtime planner.

## The integration flow

A complete host follows this order:

1. Choose a configuration for the host field and polynomial representation.
2. Load and validate the approved external schedule artifact.
3. Build prover setup for the largest supported workload from that catalog.
4. Prepare the compute backend and keep it warm across proofs.
5. Commit to each polynomial group and retain its private hint.
6. Assemble the ordered opening claims for one proof.
7. Let the catalog select the exact trusted row for that batch.
8. Produce and encode the proof.
9. Send the catalog identity, public statement, proof, and verifier setup.
10. Restore the same approved catalog, decode under its expected shape, and verify.

The [first proof](./quickstart.md) runs this lifecycle with one polynomial. The
chapters below explain how to turn that example into an application boundary.

## Public and private data

Akita keeps the data split explicit.

| Data | Owner | Purpose |
| --- | --- | --- |
| Original polynomials | Prover | Source data for commitment and opening |
| Commitment hints | Prover | Private data that connects each polynomial group to its commitment |
| Prepared compute state | Prover | Reusable transforms and backend resources |
| Trusted schedule catalog | Both | Approved external parameters used by setup, proving, and verification |
| Verifier setup | Public | Public matrix data and setup prefix commitments needed by verification |
| Commitments | Public | Values that fix the committed polynomial groups |
| Opening points and values | Public | The claims being proved |
| Schedule selection | Public | Identity of the approved catalog row used by the batch |
| Proof bytes | Public | Encoded evidence checked by the verifier |

The verifier never needs the original polynomial or the private commitment
hint. The prover never chooses unchecked cryptographic parameters. Both sides
use the same public configuration and trusted catalog identity.

## Four contracts to preserve

An integration stays simple when it preserves four contracts.

### Group contract

One commitment call creates one homogeneous group. Every polynomial in that
group has the same number of variables. Every polynomial in one opening claim
group shares one opening point.

### Order contract

Group order is part of the public statement. The prover must keep commitments,
hints, polynomials, points, and claimed values in the same order. Akita checks
this shape before proving and binds it into the transcript.

### Setup contract

The prover setup and verifier setup must describe the same public matrix stream
and setup prefix commitments. Prepared CPU caches may differ because they are
local compute state, not protocol identity.

### Revision contract

The prover and verifier must use the same Akita protocol revision. Pin every
Akita crate together and revalidate proof exchange when upgrading.

## Read the focused guides

- [Commitment groups and opening claims](./commitment-api.md) explains single
  groups, earlier commitments, and ordered batched claims.
- [Setup and prepared compute state](./setup-runtime.md) explains reusable
  public setup, verifier setup, backend caches, and persistence.
- [Proof encoding and transcripts](./proof-artifacts.md) explains the public
  artifact, schedule selection, transcript domains, and decoding.
- [Verifier only integration](./verifier-only.md) gives the direct verifier
  dependency path and call.

The protocol chapters under [How it works](../how/how-it-works.md) explain why
these contracts are sufficient. Application code can use them without
reimplementing the protocol internals.

If Akita will sit inside a larger proof system, continue to [Integrating with a
proof system](./integrations.md). That chapter explains the adapter around this
PCS lifecycle: field conversion, statement ownership, artifact transport, and
verifier placement.
