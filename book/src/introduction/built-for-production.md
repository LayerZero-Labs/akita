# Built for production

Akita treats deployment engineering as part of the cryptographic protocol. A
proof system can adopt the PCS only when its parameters, proof format,
verifier, optimized arithmetic, and operational tools agree on one exact
design.

Akita is the first implementation in this line built around that complete
requirement.

## From research result to deployable primitive

Related public implementations such as
[LaBRADOR](https://github.com/lazer-crypto/labrador),
[LaZer](https://github.com/lazer-crypto/lazer),
[RoKoko](https://github.com/lattice-arguments/rokoko), and
[Jindo](https://github.com/SNUCP/jindo) made important lattice proof techniques
reproducible. Their code is organized around research experiments, benchmark
configurations, or narrow machine targets.

Akita takes the next step. It provides a standalone PCS that another proof
system can integrate as a maintained dependency. The planner chooses
parameters, and normal builds consume generated artifacts. Separate prover and
verifier paths share canonical serialization. Portable arithmetic, profiling,
and end-to-end host integration complete the deployment system.

Production readiness comes from agreement across that complete system.

## External schedule artifacts are fixed before proving

An Akita fold uses concrete ring dimensions, digit decompositions, challenge
spaces, and response bounds. Those choices determine security, proof shape, and
the checks performed by the verifier. A proof must not be allowed to define
them.

The planner searches this space offline. It prices complete proof schedules,
checks their security requirements, and emits reviewed `.aks` artifacts. The
host loads the selected artifact into one validated catalog shared by the
configured prover and verifier. Neither
runs the planner during normal proving or verification.

The public opening statement carries a 32-byte digest that names one row in
that catalog. This digest is not a schedule and is not part of the proof
payload. The verifier accepts it only when it matches a row in its own enabled
catalog. It then reads the dimensions, bounds, and expected proof shape from
that local row. An unknown digest is rejected, so a prover cannot supply an
unchecked schedule.

This separation keeps verification small and deterministic. Parameter changes
also appear as ordinary generated source changes that code review and
continuous integration can inspect.

The [Configuration and planning](../how/configuration.md) chapter explains the
catalog keys and validation path. The [Security model](../how/security.md)
explains how the repository prices Module-SIS and response bounds.

## The verifier is an independent boundary

Verifier-only consumers can depend on `akita-verifier`, `akita-types`, and
`akita-config` without pulling in prover polynomial backends or planner search.
This crate boundary keeps proving convenience code away from the code that
accepts or rejects a proof.

Every verifier facing artifact is untrusted. The verifier checks lengths,
shapes, schedule identity, setup identity, and canonical field encodings before
using them. Malformed input returns `AkitaError` or `SerializationError`. It
must not trigger a panic or an unchecked allocation.

This rule is enforced as a repository contract rather than an informal coding
preference. The [Verification](../how/verification.md) chapter describes the
complete boundary and its rejection rules.

## Proof identity is explicit

Akita binds the configuration, setup identity, opening layout, selected
schedule, commitments, points, and claimed evaluations into the transcript.
The prover and verifier derive challenges only after they have committed to the
same public statement.

Proofs and setup artifacts use canonical encodings with checked decoding. The
verifier derives the expected proof shape from the public statement and
selected schedule before decoding. Applications pin an exact Akita revision
when they exchange proof artifacts because protocol evolution can change those
bytes.

The [Transcript and instance binding](../how/transcript.md) chapter follows the
absorption order and challenge schedule. The Usage section explains the
artifact lifecycle expected from an integrating application.

## Optimized arithmetic remains portable

Large lattice operations need optimized ring arithmetic. Akita provides scalar
implementations together with AVX2, AVX-512, and NEON paths. Runtime dispatch
selects an implementation supported by the host CPU. Differential tests compare
optimized outputs with portable reference paths.

The implementation also distinguishes arithmetic representations by their
exact bounds. Some AVX-512 kernels use 50-bit residues in 64-bit lanes, while
ordinary i32 transforms retain their own dispatch rules. These are backend
choices. They do not change the proof statement or public setup identity.

Portable reference paths make correctness testable on ordinary development
machines. Optimized paths allow the same protocol to use wide instructions on
production hosts. The [Optimizations](../how/optimizations.md) and [NTT, CRT,
and fast ring arithmetic](../foundations/ntt-crt.md) chapters explain these
layers.

## The prover controls memory as well as time

The committed polynomial is often the largest object in the application.
The current dense API receives a materialized table. When the application
passes an owned field buffer, Akita reuses that buffer instead of making
another complete copy.

The generated fold schedule replaces the source with progressively smaller
intermediate data. At large supported polynomial sizes, this additional
working state grows more slowly than the source table. The dense digit cache is
capped at 512 MiB. Large ring relation and quotient transforms operate on
bounded chunks instead of retaining complete transformed matrices.

The current 128-bit dense benchmark makes this concrete. Its $2^{28}$ field
elements occupy a 4.0 GiB source table. The first fold reduces that table to
55,332,544 field elements, about one fifth as many. The complete benchmark
process peaks at about 5.1 GiB while it constructs setup, commits, proves, and
verifies.

One-hot sources store their hot positions instead of a full table. Setup
matrices come from a deterministic public stream and can be materialized only
to the capacity required by a deployment. Prepared NTT entries are reusable
compute state rather than part of public setup identity.

The prover exposes explicit release points for hosts that need to free shared
matrix state before a recursive suffix. These controls let an integration make
memory policy visible and measurable instead of relying on hidden global
caches.

A fully streamed dense source could remove the need to store the original table
inside Akita. That source path is not part of the current production API.

## Performance claims come from complete proofs

The profile harness measures setup, commitment, proof generation,
serialization, and verification for complete opening statements. It reports
proof bytes, peak resident memory, selected schedule geometry, and separate
phase timings.

Representative dense, one-hot, and grouped profiles across the supported field
sizes currently produce Akita proof payloads of roughly 65 to 80 KB. The
profile records the exact statement and configuration behind each result. This
keeps a proof size attached to the workload that produced it.

Continuous integration compares a pull request with its merge base on the same
runner. It also checks external schedule artifact drift, portable and optimized
arithmetic, feature combinations, verifier dependencies, documentation links,
and malformed input rejection.

The [Profiling](../usage/profiling.md) chapter explains how to reproduce and
interpret these measurements.

## Jolt exercises the complete boundary

Jolt is Akita's first major integration. It is a zkVM that proves correct
execution of 64-bit RISC-V programs. Its memory checks create large one-hot
tables, which makes it a natural host for Akita's sparse commitment path.

The integration also exercises more than proving speed. Akita proof bytes must
cross a host and guest boundary. The guest must decode hostile bytes, construct
the exact public statement, resolve a shipped schedule, and verify within its
memory and execution limits. The host must preserve setup identity and
transcript inputs across the same boundary.

These requirements shaped the separate verifier package, canonical artifact
format, setup narrowing, cache release controls, and integration harness. They
are general PCS boundaries, not Jolt specific shortcuts. Other proof systems
can use the same interfaces.

The [Jolt recursion](../usage/jolt-recursion.md) chapter covers the current host
and guest path.

## What the production claim means

Akita combines the protocol and system work required for another proof system
to adopt it:

- generated parameters tied to the verifier's checks;
- a standalone verifier with a strict hostile input boundary;
- canonical proof and setup artifacts;
- portable and optimized arithmetic with differential tests;
- sparse and streaming prover paths with explicit memory policy;
- complete profiling and integration harnesses.

These are not surrounding conveniences. They determine whether the same
cryptographic statement survives planning, proving, serialization,
verification, optimization, and host integration. Akita treats that agreement
as part of the primitive.
