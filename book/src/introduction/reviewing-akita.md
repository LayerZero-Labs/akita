# Reviewing and auditing Akita

Akita is designed so that a reviewer can follow a public claim from the Book to
the code that enforces it. The main boundaries are visible in crate
dependencies, external schedule artifacts, transcript inputs, checked decoding,
and verifier rejection paths.

This chapter maps the starting points for a security argument and a code audit.

## Start with the accepted statement

An Akita proof does not stand alone. The verifier accepts a statement that
includes ordered commitment groups, opening points, claimed evaluations, a
configuration, public setup, and a generated schedule selection.

Begin an audit by writing down that complete statement. Check which values come
from the host application and which values the verifier derives. Then confirm
that the transcript binds every public value that can change the meaning of the
proof.

The [Commitment API](../usage/commitment-api.md) explains the public object
lifecycle. [Transcript and instance binding](../how/transcript.md) gives the
protocol order. [Verification](../how/verification.md) follows the checks made
before and during replay.

## Treat every verifier input as hostile

Proof bytes, setup artifacts, schedules, commitments, points, evaluations, and
length fields may all come from an attacker. Verifier reachable code must
reject malformed data with `AkitaError` or `SerializationError`.

The no panic contract prohibits verifier reachable `panic!`, `assert!`,
`unwrap`, unchecked indexing, and unbounded allocation unless an earlier
boundary has established the required invariant. Checked arithmetic is part of
the same rule because an overflow can otherwise turn a small encoded shape into
the wrong allocation or loop bound.

Review the boundary in this order:

1. Canonical decoding and size limits.
2. Configuration and generated schedule admission.
3. Setup capacity and setup identity.
4. Opening claim shape and group order.
5. Transcript binding before each challenge.
6. Response bounds and relation checks.
7. Terminal consistency checks.

This order follows the path from attacker controlled bytes to the final
cryptographic decision.

## Crate boundaries carry security meaning

Akita uses separate crates so that verifier code does not depend on the planner
or prover implementation.

| Crate | Review role |
| --- | --- |
| `akita-error` | Shared error type and checked integer formulas for sizes, offsets, and ranges |
| external `jolt-field` and `akita-algebra` | Field, ring, transform, and polynomial arithmetic |
| `akita-serialization` | Canonical encoding and checked decoding |
| `akita-transcript` and `akita-challenges` | Instance binding and challenge derivation |
| `akita-sumcheck` | Shared sumcheck proof types and drivers |
| `akita-params` | Parameter geometry, sizing, SIS tables, schedules, and witness layout |
| `akita-sis-estimator` | Offline lattice estimator that generates the SIS tables |
| `akita-types` | Public proof, setup, schedule, commitment, and claim shapes |
| `akita-planner` | Offline schedule search and cost evaluation |
| `akita-schedules` | External schedule artifact decoding, admission, and lookup |
| `akita-config` | Concrete policy and schedule admission |
| `akita-setup` | Public setup construction and setup artifacts |
| `akita-prover` | Protocol sequencing and proof assembly |
| `akita-cpu-backend` | Commitment and witness arithmetic, source polynomials, and prepared setup |
| `akita-verifier` | Proof replay and rejection of invalid statements |
| `akita-pcs` | End to end orchestration and broad public exports |

Verifier only applications should depend directly on `akita-verifier`,
`akita-types`, and `akita-config`. If verifier replay begins to require a
prover polynomial backend or planner search, the dependency change itself is a
security review event.

The [Architecture overview](../how/architecture.md) contains the complete crate
graph and core type map.

## Check one source of truth for each concept

Security and sizing logic must use the same primitive that the verifier
enforces. A planner estimate, certificate check, and runtime verifier must not
carry separate versions of one bound.

The repository follows one canonical function per concept. Generic checked
integer formulas live in `akita_error::checked`. Schedule validation belongs to
the configuration and schedule admission path. Transcript labels and order
belong to the transcript implementation. Serialization validity belongs to the
type that owns the encoding.

When a review finds two helpers that appear to compute the same security value,
the right question is which one the verifier trusts. Thin wrappers and copied
formulas make that answer harder to establish and should not become alternate
sources of truth.

## Schedule artifacts need review too

Normal builds load approved external schedule artifacts. The planner is an
offline tool, but its output selects the dimensions, challenge rules, bounds,
and opening methods that the verifier accepts.

A schedule review should establish:

- the lookup key describes the complete ordered opening layout;
- the selected row belongs to the artifact family bound to the configuration;
- every fold satisfies the accepted Module-SIS and response policies;
- setup capacity covers each direct matrix use;
- setup offloading metadata matches the carried setup commitment;
- the terminal parameters cover the remaining relation.

Generation scripts and continuous integration byte-compare committed `.aks`
artifacts with fresh planner output. An external schedule artifact is still
protocol source. Review changes to it with the same care as handwritten
verifier code.

The [Configuration and planning](../how/configuration.md), [Setup
offloading](../how/setup-offloading.md), and [Security
model](../how/security.md) chapters divide this review into its protocol parts.

## Follow transcript identity end to end

The transcript turns public objects and prover messages into challenges. If the
prover and verifier absorb different statements, they are no longer executing
the same protocol.

Check that both sides bind the same configuration, setup identity, schedule,
claim layout, commitments, points, and evaluations in the same order. Then
check that each prover message is absorbed before the challenge that depends on
it. Group order and point coordinates are protocol data, not presentation
details.

Logging transcript tests record the schedule of absorb and squeeze events.
Hardening tests mutate public values and confirm that verification rejects the
changed statement. These tests provide useful evidence, but the review should
also establish why every meaning changing value appears in the transcript.

## Review optimized arithmetic against a reference

Akita uses scalar, AVX2, AVX-512, and NEON arithmetic. Unsafe code and vector
kernels deserve their own pass because a platform specific error can change a
commitment or verifier relation without changing high level Rust code.

For each optimized kernel, identify:

1. The scalar or mathematical reference operation.
2. The accepted input range and intermediate bound.
3. The runtime feature check that selects the kernel.
4. Differential tests over boundary values and random inputs.
5. Exact reduction before values cross into canonical serialization or
   transcript code.

The [Optimizations](../how/optimizations.md) chapter maps the main kernels. The
[NTT, CRT, and fast ring arithmetic](../foundations/ntt-crt.md) chapter explains
the representations and exact reconstruction bounds.

## Fuzzing

Fuzzing is a test method that generates many random or mutated inputs and
watches the code under test for a crash, a hang, or a wrong result. Akita
keeps a standalone fuzzing campaign in the `fuzz/` directory. It uses
`cargo-fuzz` and libFuzzer, and it is built to run for days and to resume
after an interruption.

The targets are registered in `fuzz/campaign/targets.toml` in three kinds:

| Kind | What the targets exercise | Examples |
| --- | --- | --- |
| Primitive | Field, ring and NTT arithmetic, decomposition, multilinear helpers, sum-check, transcript, fold-challenge sampling | `field_arith`, `ring_ntt`, `fold_challenges` |
| Boundary | Decoders and the public prover and verifier entry points | `public_deserialize`, `schedule_artifact`, `verifier_boundary` |
| End to end | Complete commit, prove, and verify runs on catalog rows | `pcs_dense`, `pcs_recursive`, `pcs_reject` |

`fuzz/COVERAGE.md` lists every subsystem with its target and marks which of
four properties the target checks:

- **Robustness**: no crash or hang on arbitrary input at a public entry
  point.
- **Completeness**: valid supported statements prove and verify.
- **Soundness testing**: false statements and invalid proofs are rejected.
- **Resource behavior**: bounded requests stay within limits.

The `Fuzz` workflow in `.github/workflows/fuzz.yml` runs the targets marked
for CI for 60 seconds each. `fuzz/README.md` explains how to run the whole
campaign and how to reproduce and minimize a finding.

A reviewer should read the limits that `fuzz/COVERAGE.md` states for itself:

- A passing target covers the inputs it sampled. It does not cover every
  admissible table, point, or catalog row.
- The rejection target mostly tests binding. A changed statement changes the
  Fiat-Shamir challenges, so the proof fails at its first check even if a
  later algebraic check were wrong.
- No target is a cheating prover that builds a consistent transcript for a
  false statement.
- Production-sized rows are not fuzzed, because one case would take minutes
  and gigabytes under the sanitizer.
- CPU-dependent kernels are exercised only on the machine that runs the
  campaign.

Fuzzing is therefore evidence for robustness and for the
[verifier no-panic contract](../how/verification.md#the-verifier-no-panic-contract)
on the inputs it reached. It does not replace the soundness argument in the
[security model](../how/security.md).

## Understand what each repository source proves

The repository has several forms of documentation, and they have different
jobs.

| Source | What it establishes |
| --- | --- |
| Current code | Runtime behavior and enforced API boundaries |
| External schedule artifacts | Concrete schedules and parameters admitted by normal builds |
| Live specifications | Accepted designs that still contain details not folded into the Book |
| The Book | The maintained explanation of current behavior |
| Tests | Evidence for selected valid cases, rejection paths, and cross implementation agreement |
| Fuzzing campaign | Evidence that sampled inputs cause no crash, hang, or wrong acceptance |
| Archived specifications | Design history, not current requirements |

When these sources disagree, record a documentation or implementation defect.
Do not combine parts from different versions into an unstated protocol. The
[Spec index](../foundations/spec-index.md) identifies live and historical
design records.

## Suggested audit paths

Different reviews can start at different boundaries:

| Review goal | Start with | Continue with |
| --- | --- | --- |
| Binding and security parameters | [Polynomial commitments and binding](../foundations/pcs-and-binding.md) | [Security model](../how/security.md) and generated schedules |
| Transcript soundness | [Transcript and instance binding](../how/transcript.md) | Transcript labels, logging tests, and proof replay |
| Malformed input safety | [Verification](../how/verification.md) | Serialization, checked allocation, and rejection tests |
| Setup correctness | [Setup and commitment](../how/commitment.md) | Setup capacity, persistence, and setup offloading |
| Arithmetic correctness | [NTT, CRT, and fast ring arithmetic](../foundations/ntt-crt.md) | Scalar references, vector kernels, and differential tests |
| Host integration | [Usage](../usage/usage.md) | Verifier only integration and the host adapter |

A complete audit eventually crosses all of these paths. The table identifies a
clear first boundary for each kind of question.
