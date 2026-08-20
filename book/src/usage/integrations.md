# Integrating with a proof system

Akita is a standalone polynomial commitment scheme. It can serve any proof
system that works with polynomial evaluation tables and can state the values it
wants to open. The host system does not need to adopt Akita's internal ring
arithmetic or proof planner. It needs a small adapter that translates its own
polynomials and claims into Akita's public interface.

This chapter describes that adapter. The [Jolt recursion](./jolt-recursion.md)
chapter then shows a complete integration in which a zkVM executes the Akita
verifier.

## What the adapter owns

The host system owns the meaning of the committed data. It may be a virtual
machine trace, a lookup table, a memory-checking polynomial, or another value
used by the host protocol. Akita sees an ordered collection of polynomial
tables and opening claims.

The adapter joins these two views.

| Host concern | Adapter decision | Akita input |
| --- | --- | --- |
| Which values are committed | Preserve the host's canonical order | Dense or one hot polynomial sources |
| Which field holds those values | Select the matching Akita configuration | `CommitmentConfig` |
| Which claims belong together | Group polynomials by arity and opening point | `OpeningClaims` |
| Which public statement is proved | Preserve commitments, points, values, and group order | `GroupBatchStatement` |
| Which protocol session owns the proof | Choose a stable application label | `AkitaTranscript` |
| What crosses the prover boundary | Define a versioned public bundle | Setup, schedule selection, claims, and proof |

This is the main design rule: keep the host meaning outside Akita, but make the
translation into Akita explicit and deterministic. The prover and verifier must
construct the same ordered statement from the same public host data.

## Start from the host's polynomial model

Akita's current API opens multilinear polynomials. A multilinear polynomial in
$n$ variables is represented by its $2^n$ values on Boolean inputs. Many proof
systems already store trace columns and lookup tables in this form.

For each host polynomial, record:

1. Its field.
2. Its number of variables.
3. Its table representation.
4. The point where the host protocol opens it.
5. The position of its claimed value in the host transcript.

Use `DensePolynomial` for a general evaluation table. Use `OneHotPoly` when the
host already has one selected value in each fixed size chunk. Keeping one hot
data structured avoids building a large dense table only to recover the same
sparsity inside the prover.

Polynomials with the same arity can share a commitment group. Polynomials in
one opening group also share one opening point. If the host opens different
groups at different points, preserve those groups and their order. Akita can
batch them into one proof.

## Select the field boundary once

Choose one Akita configuration whose base field matches the values supplied by
the host. Keep conversion code at the adapter boundary. A field element should
enter Akita through one canonical conversion and leave through one canonical
encoding.

Do not convert values by copying their in-memory bytes. Rust field types can
use different internal representations even when they describe the same prime
field. Convert through canonical integers or an agreed canonical byte encoding,
then test the boundary with zero, one, the largest canonical value, and random
values.

The [configuration guide](./configuration.md) explains the shipped dense and
one hot configurations. Akita chooses the ring dimensions and fold schedule
after the adapter supplies the field and claim shape.

## Build one public statement

The host should define a single versioned Akita artifact. It contains, or
securely identifies, everything the verifier needs:

- the Akita revision and configuration;
- the verifier setup;
- the exact generated schedule selection;
- the ordered commitments;
- the ordered opening points and claimed values;
- the expected proof shape;
- the proof bytes;
- the transcript domain used by this host protocol.

This bundle is more than a transport format. It is the complete public claim.
The verifier should not infer missing group order, try several configurations,
or choose a schedule after receiving the proof.

Use a transcript label owned by the host, such as
`my-system/trace-openings/v1`. A versioned label separates this proof from every
other Akita use and gives the host a clean protocol upgrade boundary.

The [proof encoding and transcripts](./proof-artifacts.md) chapter gives the
exact Akita objects and decoding flow.

## Place the verifier where it provides the most value

Akita's verifier is a separate Rust package. A host can place it in several
environments:

| Verifier placement | Typical use |
| --- | --- |
| In the host process | Check locally generated proofs and network requests |
| In a dedicated service | Keep proving and verification deployments separate |
| In a zkVM guest | Prove that the Akita verifier accepted a proof |
| In another recursive circuit | Compress or aggregate Akita verification with other claims |

The direct `akita-verifier` path has no polynomial backend, setup generator, or
planner search. This makes it the right boundary for small verifier targets.
The host decodes the public bundle, reconstructs the typed statement, starts a
fresh verifier transcript, and calls `akita_verifier::batched_verify`.

When the verifier itself runs inside another proof system, begin with native
verification of the same bundle. Then move the exact accepted bundle into the
guest or circuit. This gives the integration a simple debugging order:

```text
host creates Akita proof
        ↓
native Akita verifier accepts it
        ↓
guest or circuit decodes the same public bundle
        ↓
guest or circuit runs the Akita verifier
        ↓
outer proof system proves that execution
```

The native check separates an Akita statement error from a guest runtime or
outer prover error.

## Plan setup and upgrades as deployment inputs

The proving service prepares the large reusable compute state. The verifier
receives only the public setup needed by its selected schedule. A deployment
can package this setup with the verifier or authenticate it by an application
owned identifier.

Pin every Akita crate to one revision. Akita's protocol and proof encoding can
advance together, so treat an upgrade as a new host artifact version. Rebuild
setup packages, regenerate proof fixtures, and run the complete host-to-verifier
path before deploying the new version.

## Validate the integration at its real scale

A useful integration test covers more than one successful proof. It should:

1. Compare host and Akita field conversions in both directions.
2. Compare each claimed value with an independent host evaluation.
3. Round trip the complete public artifact through serialization.
4. Accept a valid proof through the final verifier entry point.
5. Reject changed commitments, points, values, schedules, and proof bytes.
6. Measure proving time, peak memory, verifier time, and proof size at the
   host's intended polynomial sizes.

For a zkVM or recursive circuit, also measure input decoding separately from
Akita verification. A large verifier setup can make transport and decoding the
dominant cost even when the verifier kernel is fast.

The repository's Jolt harness follows this pattern. It produces a real Akita
proof, checks it natively, transports one bounded verifier bundle into a RISC-V
guest, runs the direct verifier there, and measures each phase.
