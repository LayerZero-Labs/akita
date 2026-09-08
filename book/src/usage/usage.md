# Using Akita

Akita gives a proof system a complete polynomial commitment interface. An
application commits to a large table, claims the value of that table's
multilinear extension at a chosen point, and asks Akita to prove the claim. The
verifier checks the proof without receiving the table.

Akita is built for use as infrastructure. The repository provides generated
security parameters, optimized prover code, a compact proof format, and a
separate verifier package. A host proof system can use the complete Rust API or
carry only the verifier into a smaller trusted environment.

## The complete lifecycle

An Akita integration performs four operations.

| Operation | What the application supplies | What Akita returns |
| --- | --- | --- |
| Setup | A configuration and maximum workload size | Public setup data and reusable prover state |
| Commit | One or more polynomial tables | A public commitment and private prover data |
| Prove | The committed tables, opening points, and claimed values | One batched opening proof |
| Verify | Public setup, commitments, claims, and proof bytes | Success or a structured error |

The shortest complete path looks like this:

```text
choose a configuration
        ↓
build and prepare setup
        ↓
commit to polynomial groups
        ↓
state the opening points and values
        ↓
prove and encode
        ↓
decode and verify
```

The [first proof](./quickstart.md) chapter runs this full path with one dense
polynomial. It uses a real external schedule artifact and the same public API used by
larger integrations.

## What Akita commits to

The current API commits to multilinear polynomials represented by their values
on the Boolean cube. A polynomial with $n$ variables has $2^n$ entries. Akita
supports arbitrary dense tables and a compact one hot representation for tables
with one selected entry in each fixed size chunk.

Applications may commit to several polynomials at once. Polynomials with the
same number of variables and the same opening point form one commitment group.
A proof may contain several ordered groups, and each group may use its own
point. Akita binds the group order, commitments, points, claimed values,
configuration, and selected proof schedule into the transcript.

This group model lets a host combine earlier commitments with a final group in
one proof. It also lets Akita preserve structured inputs instead of expanding
every table into a dense allocation.

## Choose your path

| If you want to | Read |
| --- | --- |
| Run one complete proof | [Your first proof](./quickstart.md) |
| Pick a field and input representation | [Choosing a configuration](./configuration.md) |
| Integrate commitment, proving, and verification | [Integrating the PCS](./integration.md) |
| Carry only verification into another environment | [Verifier only integration](./verifier-only.md) |
| Choose Cargo features | [Feature flags](./feature-flags.md) |
| Measure time, memory, and proof size | [Profiling](./profiling.md) |
| Diagnose a failed run | [Troubleshooting](./troubleshooting.md) |
| Add Akita to another proof system | [Integrating with a proof system](./integrations.md) |
| Verify Akita inside a zkVM | [Jolt recursion](./jolt-recursion.md) |

Readers who want to understand the protocol itself can continue to
[How it works](../how/how-it-works.md). Readers who need the mathematical ideas
first can start with [Foundations](../foundations/foundations.md).

## Which crate to use

The repository separates proving from verification on purpose.

| Crate | Role |
| --- | --- |
| `akita-pcs` | Complete setup, commitment, proving, and verification interface |
| `akita-config` | Production configurations and trusted artifact validation |
| `akita-schedules` | External artifact decoding, row audit, and owned catalog lookup |
| `akita-prover` | Polynomial representations, prepared compute state, and prover kernels |
| `akita-verifier` | Verification without prover polynomial backends or planner search |
| `akita-types` | Proofs, commitments, claims, schedules, and setup types shared across the boundary |
| `akita-transcript` | The transcript that derives proof challenges from the public statement and prior messages |

An application that proves and verifies can begin with `akita-pcs`. A small
verifier should depend directly on `akita-verifier`, `akita-types`, and
`akita-config`. Pin every Akita crate to the same revision because a proof and
its verifier must use the same protocol format.

## Akita chooses the proof plan

The application chooses the field, data shape, and group layout. Akita then
resolves an approved row from the catalog supplied by the application. That row fixes the
ring dimensions and all later fold parameters.

This division keeps the application interface small. It also keeps parameter
search out of the verifier. The verifier uses the exact catalog bound to its
preprocessing or setup package and replays the statement-selected approved row.

The [configuration guide](./configuration.md) explains the choices an
application should make and the choices Akita deliberately makes for it.
