# Spec: Circuit-Oriented BN254 Akita and On-Chain Lattice Jolt

| Field         | Value                    |
|---------------|--------------------------|
| Author(s)     | Quang Dao, Codex         |
| Created       | 2026-09-03               |
| Status        | proposed                 |
| PR            |                          |
| Supersedes    |                          |
| Superseded-by |                          |
| Book-chapter  |                          |

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** in this document are to be interpreted as described in
[BCP 14](https://www.rfc-editor.org/info/bcp14) when, and only when, they appear
in all capitals.

## Summary

This specification proposes an on-chain wrapping pipeline for lattice Jolt. The
application proof remains a fast Jolt proof over the 128-bit Solinas field with
Akita. A second Jolt instance over the BN254 scalar field proves that a RISC-V
program accepted that application proof. A circuit-oriented BN254 Akita profile
makes the second Jolt verifier almost entirely native BN254 field arithmetic.
Finally, a generated circuit proves that the BN254 Jolt verifier accepted, and
Ethereum verifies only that short outer proof. Groth16 is the first baseline;
Spartan plus HyperKZG and PLONK are later backends for the same verifier IR.

The central engineering claim is deliberately narrower than a performance
claim: this route removes the non-native elliptic-curve, extension-field, and
pairing checks that dominate wrapping a Dory verifier. It replaces them with
field arithmetic, range checks, standard-hash gadgets, sparse-challenge
sampling, and a finite setup-MLE base case. That is a materially simpler circuit
surface. Whether the extra field-switch Jolt proof is faster than the current
Dory wrapper is an empirical question and a required decision gate, not an
assumption of this design.

## Intent

### Goal

Build one fixed-profile pipeline that proves, on Ethereum, that a lattice-Jolt
application proof was accepted, while keeping the 128-bit application prover
and using a circuit-generated BN254 Akita verifier rather than hand-maintained
wrapper-specific verifier code.

### Decisions

The first implementation SHOULD make the following choices:

1. The application proof uses the current 128-bit lattice-Jolt path.
2. Field switching is a Jolt proof of a RISC-V verifier program, not a direct
   non-native-field translation into Groth16 or PLONK.
3. The outer Jolt proof uses a purpose-built Akita profile with BN254 `Fr` as
   both its commitment field and its degree-one challenge/evaluation field.
4. The BN254 Akita profile uses uniform power-of-two cyclotomic rings, starting
   with `D = 64`, and a fixed challenge shell with an exact unit-difference
   certificate.
5. The profile uses coordinatewise response bounds and digit range proofs. It
   has no operator-norm rejection sampler and no variable-length terminal
   codec.
6. Direct terminal NTT/CRT work is replaced by a batched matrix-product
   sumcheck whose base case is a small digit MLE and a fixed setup MLE.
7. Schedule, setup seed, transcript, sampler, proof shape, program class, and
   public-input layout are hard-coded into a small number of circuit classes.
8. BLAKE3 is the leading standard-hash circuit target for both transcripts
   verified inside the wrapper: Jolt's outer transcript and Akita's nested
   transcript. SHA-256 is the conservative library-backed fallback until the
   BLAKE3 gadget and transcript composition pass audit. The two transcripts
   remain separate, domain-separated states connected by the existing
   statement-challenge bridge. Poseidon is out of scope. Keccak remains
   appropriate for Solidity or an outer PLONK transcript, but is not the
   initial in-circuit proof transcript.
9. One typed semantic verifier emits both concrete checks and a circuit IR.
   Groth16, Spartan plus HyperKZG, and PLONK are compiler backends, not
   separately rewritten verifiers.
10. Groth16 is the first on-chain target because fixed circuit classes make a
    circuit-specific setup tolerable and its verifier cost is largely
    independent of circuit size. Spartan plus HyperKZG follows as the closest
    reuse of Jolt's current wrapper machinery. PLONK follows when a universal
    SRS or easier circuit upgrades are more important.

### End-to-end statement

Let `S` be the public application statement. The intended proof stack is:

```mermaid
flowchart TD
    A[Application execution] --> P0["P0: Jolt over fp128 + Akita fp128"]
    P0 --> G["RISC-V guest: verify_fp128_jolt_akita(S, P0)"]
    G --> P1["P1: Jolt over BN254 Fr + Akita BN254"]
    P1 --> V["Generated semantic verifier / circuit IR"]
    V --> P2["P2: Groth16 first; Spartan+HyperKZG and PLONK next"]
    P2 --> E["Solidity verifier using BN254 precompiles"]
```

The on-chain contract accepts only if `P2` proves that the fixed BN254 Jolt
verifier accepts `P1`, whose proved RISC-V execution accepts `P0` for `S`.

The pipeline SHOULD also support a bootstrap mode in which the application is
run directly under BN254 Jolt. That omits `P0` and the field-switch guest, and
is useful for validating the BN254 Akita verifier compiler before paying the
cost of two Jolt proofs. It is not the intended high-performance application
mode.

### Security boundary

The application and field-switch proofs can retain Akita's lattice assumptions,
but every proposed final wrapper uses a classical elliptic-curve commitment or
pairing assumption. The on-chain statement is therefore **not post-quantum**,
even though it certifies a post-quantum proof stack. `P1` remains useful as a
future input to a post-quantum on-chain verifier or another transparent wrapper.

The public statement MUST bind at least:

- the application program or bytecode digest;
- the public input/output digest;
- the fp128 Jolt and Akita profile identities;
- the BN254 Jolt and Akita profile identities;
- the verifier-guest ELF/program digest;
- the circuit ID and compiler/IR version; and
- the exact outer proof system and verifying key.

No proof-provided schedule, sampler policy, transcript policy, setup geometry,
or relation list may alter the circuit after key generation.

### Invariants

1. **One acceptance program.** The optimized native verifier, scalar reference
   verifier, and circuit emitter MUST execute the same semantic verifier
   program. Backend-specific kernels MAY implement one named primitive, but
   differential tests MUST show that they implement the same operation.
2. **One fixed profile identity.** The circuit ID MUST commit to every value
   that changes the accepted language, including field modulus, ring
   dimensions, challenge shell, transcript, schedule, setup seed, digit widths,
   public inputs, compiler revision, and terminal relation.
3. **Degree-one BN254 challenges.** The circuit profile MUST use BN254 `Fr` for
   proof scalars, sumcheck challenges, opening points, and claimed evaluations.
   It MUST NOT introduce an extension field merely to reuse a small-field
   profile.
4. **Exact sparse challenge law.** The prover and every verifier backend MUST
   derive identical sparse challenges from the transcript. Indices MUST be
   distinct and in range. The first profile uses the injective enumerative
   decoder below; a profile that retains partial Fisher-Yates MUST also prove
   its bounded rejection rule.
5. **Pairwise-unit challenges.** The selected shell MUST have an exact proof
   that every distinct challenge pair has an invertible difference in the
   selected fully split cyclotomic ring.
6. **No operator-norm rejection.** The circuit profile MUST NOT replay the
   fixed-point trigonometric operator-norm predicate. Security and response
   sizing MUST use deterministic shell bounds, starting with the exact `L1`
   bound.
7. **Range, not encoding, proves smallness.** Every terminal and recursive
   digit consumed by the circuit MUST have a fixed signed representation, a
   canonical reconstruction into `Fr`, and an explicit range constraint.
8. **Finite setup closure.** Setup offloading MUST terminate at a fixed,
   circuit-key-bound setup MLE evaluation. It MUST NOT create an unauthenticated
   setup value or an infinite chain of opening claims.
9. **No direct terminal NTT.** The circuit terminal MUST NOT invoke NTT, CRT,
   floating point, architecture-specific prepared caches, Golomb-Rice decode,
   or sparse-ring convolution. Its operations are field arithmetic, Boolean or
   small-range constraints, standard hashing, and fixed-table MLE evaluation.
10. **Transcript parity.** Concrete and symbolic executions MUST emit identical
    ordered event traces and challenges for both the outer Jolt transcript and
    the nested Akita transcript, including the exact bridge challenge, on every
    accepted fixture.
11. **Fail closed.** Malformed proofs, schedules, field encodings, digits,
    sampler traces, setup claims, and circuit inputs MUST reject. The native
    verifier retains Akita's no-panic contract.
12. **No backward-compatibility path.** This profile introduces a new proof,
    schedule, setup, transcript, and circuit identity. It does not decode or
    accept an earlier Akita proof under the new verifier.

### Non-Goals

- Direct verification of an Akita proof in EVM bytecode.
- A post-quantum final on-chain proof.
- Preserving current proof bytes, schedules, setup artifacts, or terminal
  payloads.
- Minimizing unwrapped Akita proof size.
- Minimizing ordinary host verifier latency for the BN254 circuit profile.
- Supporting arbitrary ring dimensions, arbitrary schedules, or arbitrary
  Jolt programs in one circuit.
- Poseidon, MiMC, or another algebraic hash as the circuit transcript.
- Claiming lower end-to-end cost than the Dory wrapper before measurement.

## Evaluation

### Acceptance Criteria

- [ ] `jolt-field::Fr` supports the degree-one `ExtField<Fr>` and
  `MulBaseUnreduced<Fr>` contracts required by Akita, with differential tests
  against arkworks BN254 `Fr`.
- [ ] BN254 elements use strict canonical 32-byte encodings. Decoders reject
  noncanonical values, and the Solinas-only, BN254-only, and combined feature
  graphs build independently.
- [ ] A canonical fixed-width integer representation supports modulus
  comparison, subtraction, small shifts, low-bit extraction, centered
  representatives, balanced digit peeling, and exact reconstruction for every
  BN254 decomposition basis selected by the planner.
- [ ] Akita has a BN254 SIS modulus identity that is not truncated to `u128`,
  and generated security tables cover every matrix role, width, dimension, and
  response bound used by the fixed circuit schedule.
- [ ] A scalar-reference D64 backend completes setup, commit, prove, and verify
  for dense and one-hot fixtures, and rejects proof mutations, before any
  optimized BN254 NTT backend becomes trusted.
- [ ] The native BN254 radix-2 backend is differentially tested against the
  scalar reference for every ring operation used by setup, proving, and
  verification.
- [ ] An exact checker proves the selected D64 shell's support floor and
  pairwise-unit-difference property over BN254 `Fr`.
- [ ] The fixed sparse sampler has concrete, scalar-reference, symbolic, and
  gnark parity tests for challenge bytes, enumerative indices, magnitudes, and
  signs. The Fisher-Yates fallback also tests swaps and rejection behavior.
- [ ] The fixed Jolt and Akita transcripts use the selected standard-hash
  primitive with separate domain states, and parity tests cover every absorb,
  squeeze, field encoding, and the Jolt-to-Akita bridge challenge.
- [ ] The circuit profile contains no operator-norm predicate and uses only
  verifier-enforced coordinatewise digit bounds.
- [ ] A circuit terminal replaces the current direct terminal NTT path and
  passes a standalone algebraic soundness review.
- [ ] Setup offloading can reach the circuit terminal, and the final fixed
  setup MLE is recomputed in the circuit rather than trusted from the witness.
- [ ] One deterministic, versioned IR is emitted from the canonical semantic
  verifier. Re-emitting the same profile produces an identical IR digest.
- [ ] The concrete optimized verifier, scalar verifier, IR interpreter, gnark
  solver, Groth16 proof, Spartan-plus-HyperKZG proof, and PLONK proof agree on
  an accepted corpus and reject every single-field mutation in the adversarial
  corpus.
- [ ] The compiler reports operation counts by source label and protocol layer:
  field multiplication, Boolean/range constraint, Jolt standard-hash
  compression, Akita standard-hash compression, sampler swap, setup-MLE
  multiplication, and public input.
- [ ] The bootstrap direct-BN254 Jolt proof is wrapped end to end and accepted
  by generated Solidity contracts for Groth16 and at least one reusable-SRS
  backend.
- [ ] The fp128 verifier guest produces `P1` under BN254 Jolt/Akita, and the
  generated circuit and Solidity contract accept the resulting `P2`.
- [ ] The final benchmark report records `P0` prover time and size, verifier
  guest cycles, `P1` prover time and size, circuit constraints, setup/proving
  key size, `P2` prover time and memory, proof bytes, Solidity bytecode size,
  verification gas, and public-input gas.
- [ ] The final report compares those measurements with the then-current Dory
  wrapper at commit-pinned revisions and states which route wins on engineering
  complexity, prover cost, proof size, and gas.

### Testing Strategy

The implementation requires four layers of tests.

**Protocol tests** cover the BN254 field identity, SIS tables, shell support,
unit-difference certificate, decomposition bounds, terminal relation, and
schedule audit. These are exact integer or field tests and do not depend on a
circuit backend.

**Twin-execution tests** run the same semantic verifier with concrete and IR
backends. They compare accepted/rejected status, all public assertions, the
ordered transcript-event trace, derived challenges, and the final public
statement. Tests MUST include malformed lengths, noncanonical BN254 elements,
wrong schedule/circuit IDs, out-of-range digits, duplicate shuffle positions,
biased or over-cap sampler draws, missing setup values, and unused IR inputs.

**Compiler tests** interpret the IR, solve generated R1CS and SCS constraints,
and create and verify Groth16, Spartan-plus-HyperKZG, and PLONK proofs. Every
serialized proof field and public input receives a tamper test. A compiler
audit MUST show that every private input reaches an assertion and that every
assertion reaches the final constraint system.

**End-to-end tests** prove a small direct-BN254 Jolt program first, then prove
the fp128 verifier guest. The generated Solidity contracts run in an EVM test
harness with gas snapshots and negative tests for the statement digest,
circuit ID, proof, and verifying key.

### Performance Objectives and Decision Gates

This profile optimizes a circuit-weighted verifier objective, not proof bytes or
native host time. The planner SHOULD minimize a measured cost vector rather
than one synthetic scalar until backend measurements justify weights:

```text
(Jolt standard-hash compressions,
 Akita standard-hash compressions,
 Boolean/range constraints,
 nonconstant field multiplications,
 Fisher-Yates/RAM constraints,
 residual setup-MLE width,
 public inputs,
 IR nodes,
 proving-key bytes)
```

Fixed proof bytes, extra sumcheck messages, and a slower native verifier are
acceptable when they reduce circuit cost. Hard limits still apply to the RISC-V
guest trace, wrapper prover memory, setup size, Solidity bytecode size, and EVM
gas.

The work MUST stop for design review at these gates:

1. **Hash gate.** Count every Jolt and Akita transcript compression, including
   their bridge, in one representative fixed schedule. Compare audited BLAKE3
   and SHA-256 lowerings on the complete schedule. If standard hashing
   dominates the circuit, evaluate batching, challenge coalescing with a proved
   joint bound, or a standard-hash lookup table before tuning field
   multiplications.
2. **Terminal gate.** Compare the proposed terminal sumcheck plus fixed setup
   MLE with a direct constant-matrix MLE. Keep the smaller complete circuit,
   not the design with fewer native operations.
3. **Guest gate.** Measure the fp128 verifier guest before a full `P1` proof.
   The historical fixed-D64 `nv=20` profile was `65,283,025` cycles after
   zero-copy input, while the current adaptive `nv=32` profile may approach or
   exceed the existing four-billion-cycle harness limit. Field switching is
   not viable until a production-sized guest trace and memory estimate exist.
4. **Wrapper gate.** Compare direct-BN254 Jolt wrapping with the two-proof
   field-switch path. This separates circuit/compiler cost from guest-recursion
   cost.
5. **Dory gate.** Compare against a commit-pinned Dory wrapper. Structural
   simplicity alone does not justify a materially worse production prover.

## Design

### Audited Implementation Baseline

This plan is based on a code audit of Akita `main` at
`26bdbac796a2fcc8092a2fa3be9ffdc1721a380d`, the quotient-free work in
[#466](https://github.com/LayerZero-Labs/akita/pull/466) at
`1d2800432a81755e51af3edd30360a160a4b811b`, and the external-catalog work in
[#428](https://github.com/LayerZero-Labs/akita/pull/428) at
`c02ed79283d424ac5aabaaa23cceb582750eec2b`. The PR revisions are dependencies,
not assertions about current `main`; implementation PRs MUST record the exact
rebased revisions they use.

The audit supports the mathematical premise but rejects the idea that BN254 is
only a configuration change. The generic cyclotomic-ring layer does not
fundamentally require a pseudo-Mersenne field, and the instance descriptor
already has room for a 32-byte modulus. The SIS estimator already performs its
core arithmetic with arbitrary-precision integers. The hard work lies at
concrete boundaries that still assume a 128-bit field: Jolt capability traits,
canonical scalar encoding, balanced decomposition, SIS-profile dispatch, and
the optimized CRT/NTT backend.

Akita currently pins Jolt revision
`72dc6451628d8b1dd794147a1f1cc40be0d77963`. That revision contains a canonical
32-byte BN254 `Fr` implementation, field arithmetic, serialization, and
Montgomery accumulators. It does not implement all Akita capability traits.
Because both the traits and `Fr` are owned by Jolt crates, Rust's orphan rules
require the missing implementations to land on the Jolt side.

### Two Profiles, Not One Compromise

The final pipeline has two different verifier workloads and SHOULD treat them
as separate profiles:

| Profile | Proof it opens | Primary objective |
|---|---|---|
| `Fp128RecursionGuest` | Application proof `P0` inside RISC-V/Jolt | Minimize guest trace, input decode, and guest memory |
| `Bn254CircuitV1` | Field-switch proof `P1` inside an outer wrapper | Minimize circuit operations and standard-hash calls |

The existing fp128 application profile can bootstrap `Fp128RecursionGuest`, but
the recursion profile MAY choose more setup offloading or a different fixed
schedule if that substantially reduces guest cost. `Bn254CircuitV1` does not
optimize ordinary native verification at all.

The implementation SHOULD begin with a test-only `Bn254D64Fixture` that exists
solely to establish scalar correctness and backend parity. It MUST NOT expose a
public, stable BN254 config while the challenge law, transcript, schedule, and
terminal relation can still change. `Bn254CircuitV1` becomes public only after
those choices and its external catalog are frozen and bound into the profile
identity.

### BN254 Field, Decomposition, and SIS Work

The BN254 scalar modulus is:

```text
decimal = 21888242871839275222246405745257275088548364400416034343698204186575808495617
hex     = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
LE u64  = [43e1f593f0000001, 2833e84879b97091,
           b85045b68181585d, 30644e72e131a029]
```

The first implementation has four field cutovers.

1. **Jolt capability traits.** `jolt-field::Fr` needs Akita's degree-one
   `ExtField<Fr>`, `MulBaseUnreduced<Fr>`, `Unreduced`,
   `WithCommitAccumulator`, `Fold`, and `WithPacking` contracts. The initial
   implementation SHOULD use degree-one identities, `Fr` for product and wide
   values, immediate reduction, the canonical fold formula, and
   `NoPacking<Fr>`. Delayed reduction and packing are later optimizations.
2. **Canonical encoding.** Akita needs strict 32-byte little-endian `Fr`
   encoding and decoding, including `FpExtEncoding<Fr>`. A decoder MUST reject
   a 256-bit integer greater than or equal to the modulus. Feature graphs for
   the Solinas field, BN254, and both fields together MUST compile.
3. **Wide balanced decomposition.** There are 84 non-test
   `to_u128_checked()` call sites across 25 files at the audited baseline. The
   central blocker is
   `crates/akita-algebra/src/ring/cyclotomic/decomposition.rs`. The hot path
   SHOULD use a fixed `[u64; 4]` canonical integer, not `BigUint`, for compare,
   subtract, shifts of at most eight bits, low-bit extraction, the
   `q`-representative, balanced digit peeling, and exact reconstruction. The
   current `u128` path MAY remain as a specialized fast path. Small security
   bounds that are genuinely below `u128` MUST remain small rather than being
   widened indiscriminately.
4. **Exact SIS profile identity.** `SisModulusProfileId::modulus()` and config
   validation currently round-trip through `u128`. The BN254 profile MUST own
   the exact `[u8; 32]` modulus, `field_bits = 254`, a stable profile tag, and a
   generated-table identity. Config validation and instance descriptors SHOULD
   use a generic `field_modulus_be_bytes::<F>()`. Cache names and digests MUST
   include the full modulus identity rather than a 32-hex-digit `q` label.

The larger modulus roughly doubles the digit count at the same decomposition
basis. For a 128-bit versus 254-bit representative, the planner starts from:

| `log2(basis)` | fp128 digits | BN254 digits |
|---:|---:|---:|
| 3 | 43 | 85 |
| 4 | 32 | 64 |
| 5 | 26 | 51 |
| 6 | 22 | 43 |
| 7 | 19 | 37 |
| 8 | 16 | 32 |

The circuit planner MUST search through basis eight. It MUST NOT inherit the
current basis-three-through-six search merely because those values are good
for native fp128 execution.

The generated SIS tables currently cover Q32, Q64, and Q128 profiles. BN254
needs independent exact rows for every matrix role, width, ring dimension, and
response bound selected by the fixed schedule. Table generation SHOULD use the
existing arbitrary-precision estimator, emit both `Linf` and selected `L2`
rows, and bind the generator revision, inputs, and digest. A wrong field modulus
MUST fail profile dispatch; it MUST NOT reuse Q128 rows.

The recommended first geometry is uniform `D = 64`. BN254 `Fr` satisfies
`r = 1 mod 2^28`, so `X^D + 1` splits completely for every power-of-two
`D <= 2^27`. Complete splitting is not itself a blocker: the exact D64
evaluation-kernel certificate at commit
[`743fbeb830423d4430c043b6a220f9b47876511e`](https://github.com/LayerZero-Labs/akita/tree/743fbeb830423d4430c043b6a220f9b47876511e/scripts/bn254_challenge_units)
proves that the kernel contains no nonzero integer vector of squared norm at
most `336`.

### Native Correctness and Ring Backends

The existing optimized ring path is not a BN254 backend. In particular,
`crates/akita-types/src/ntt_cache.rs` selects auxiliary Q32, Q64, or Q128 CRT
machinery. BN254 should not be forced through that enum merely to reuse the
native fp128 architecture.

Bring-up SHOULD use two explicit backends:

1. `ScalarReferenceBackend<F>` performs schoolbook field and polynomial
   arithmetic without NTT or auxiliary CRT. It is the correctness oracle and
   enables small end-to-end D64 fixtures before optimization.
2. `Bn254Radix2Backend` performs native cyclic and negacyclic radix-2 NTTs in
   `Fr`. Complete splitting and the field's two-adicity make this the natural
   optimized path.

The scalar backend MUST complete setup, commit, prove, and verify for both a
dense and a one-hot fixture before the optimized backend is accepted. Every
relevant ring operation and resulting proof MUST then be checked
differentially. This ordering isolates field and protocol correctness from NTT
layout, twiddle, and cache bugs.

### Circuit-Oriented D64 Challenge Shell

The initial shell SHOULD be `(count_pm1, count_pm2) = (24, 13)` unless the
complete response/SIS planner finds a cheaper fixed schedule. Its exact
properties are:

```text
weight          = 37
L1              = 24 + 2*13 = 50
L2^2            = 24 + 4*13 = 76
support          = C(64,37) * C(37,24) * 2^37
log2(support)    = 128.28475074122736
difference L2^2 <= 4*76 = 304 < 336
```

Thus it clears the 128-bit single-draw support floor, and the D64 certificate
proves that every distinct pair has a unit difference. It also improves on the
current raw D64 `(31,10)` profile in both sampled weight (`37` instead of `41`)
and deterministic `L1` bound (`50` instead of `51`). The alternative `(21,15)`
shell uses only 36 positions and has 128.3311 support bits, but raises `L1` to
51 and `L2^2` to 81. The planner SHOULD compare these two exact candidates;
the baseline prefers `(24,13)` because `L1` propagates into every downstream
response bound while one Fisher-Yates step is local.

No operator-norm rejection is needed or allowed. The shell is sampled once,
and the response schedule uses the worst-case shell bounds. This removes the
current fixed-point roots, trigonometric table, retry loop, and verifier replay
from the circuit. It may require more or wider response digits than an
operator-rejected profile; that trade is measured in range constraints.

### D128 and Larger-Ring Optimization Path

D64 is the bring-up profile, not a claim that it is the final circuit optimum.
The selected exact zero-collision D128 candidate uses a fixed 72-position
Galois mask and the shell `(count_pm1, count_pm2) = (38, 5)`:

```text
ambient dimension = 128
allowed positions = 72
weight            = 43
L1                = 38 + 2*5 = 48
L2^2              = 38 + 4*5 = 58
log2(support)      ~= 129.51
difference L2^2   <= 4*58 = 232
```

The exact certificate at
[`ff4481b2e59f3c9d09016ac97045ef65ea4af490`](https://github.com/LayerZero-Labs/akita/tree/ff4481b2e59f3c9d09016ac97045ef65ea4af490/scripts/bn254_challenge_units)
covers this mask and shell. The same supported challenges embed into D256 and
larger power-of-two rings. This family improves the deterministic `L1` bound
from 50 to 48, but pays for a larger ring and a 72-position decoder. The
planner SHOULD compare its complete constraint cost with D64 after the D64
backend is correct; the smaller response bound alone does not determine the
winner.

A challenge-family descriptor MUST bind the ambient dimension, allowed-position
mask, signed-magnitude counts, decoder ID, entropy width, deterministic `L1`
and `L2` bounds, exact-certificate digest, and rejection policy. The circuit
profiles described here set rejection to `None`. This descriptor is part of
the schedule and instance identity, so D64 and D128 proofs cannot be
cross-interpreted.

### Sparse Challenge Decoding and Fisher-Yates Fallback

The current sampler draws distinct positions with a partial Fisher-Yates
shuffle and derives signs from indexed SHAKE256. Replaying that algorithm is
possible, but a fixed circuit profile can avoid both the shuffle and a
hash-based XOF while preserving an exact 128-bit strong challenge set.

The first profile SHOULD take 128 transcript bits as an integer `u` and decode
them injectively into the `(24,13)` shell:

```text
signs = u mod 2^37
q     = floor(u / 2^37)                 # 0 <= q < 2^91
M     = C(37,24) = 3,562,467,300
q     = position_rank * M + magnitude_rank

0 <= position_rank  < C(64,37) = 846,636,978,475,316,672
0 <= magnitude_rank < C(37,24)
```

Lexicographic combinatorial unranking maps `position_rank` to 37 distinct
positions among 64 and maps `magnitude_rank` to the 24 magnitude-one slots
among those 37 positions. The remaining 13 slots have magnitude two, and the
37 low bits select signs. This map is injective because

```text
C(64,37) * C(37,24)
    = 3,016,116,550,789,119,501,144,825,600
    > 2^91.
```

It therefore emits exactly `2^128` equiprobable challenges from a subset of
the certified shell. Distinctness is structural, and the full-shell D64
certificate proves the unit-difference property for the subset. The circuit
needs constant quotient/remainder reconstruction, approximately 101 bounded
rank-comparison steps, and 37 sign bits; it needs no sampler rehash, random RAM,
or rejection failure. The compiler MUST use one specified combinadic ordering
and publish reference vectors at the first, last, and boundary ranks.

This decoder is entropy-optimal at the transcript boundary: it consumes
exactly 128 bits to choose exactly `2^128` challenges, with no failed draws or
many-to-one shuffle order. It is also the expected circuit winner for D64. A
straight lexicographic lowering performs at most 64 position-rank comparisons
and 37 magnitude-rank comparisons, plus one constant quotient/remainder and
37 sign bits. By contrast, faithful partial Fisher-Yates needs 37 bounded
draws, mutable permutation access, conditional swaps, and hundreds of XOF
bytes. This is a structural advantage, not merely a faster hash choice.

Combinatorial unranking is not claimed to be globally constraint-minimal.
PLONK lookup tables, chunked arithmetic decoding, or a deliberately structured
`2^128` subset of the certified shell may lower the same mapping more cheaply.
Any replacement MUST remain injective, retain the deterministic response
bounds and unit-difference certificate, and beat combinadic unranking in the
complete backend. Changing only the unranking algorithm while preserving the
same canonical rank order does not change the challenge distribution.

The planner SHOULD still benchmark the current partial Fisher-Yates law. For a
shell of weight `w = 37`, step `i` draws `j_i` uniformly from `[i, 63]`, swaps
`perm[i]` and `perm[j_i]`, and emits `perm[i]`. A faithful symbolic lowering
constrains:

1. exact counter/XOF expansion under the profile's selected standard hash;
2. fixed-width unsigned candidate words;
3. the exact rejection threshold `floor(2^b / (64-i)) * (64-i)`;
4. selection of the first accepted candidate within a fixed public attempt
   cap;
5. quotient/remainder reconstruction of `j_i - i`;
6. the 64-entry conditional swap; and
7. sign bits and the fixed split between 24 magnitude-one and 13
   magnitude-two positions.

The attempt cap MUST be chosen from a proved union bound over all 37 draws.
Cap exhaustion is a public sampler failure, not modulo reduction. For example,
with 32-bit candidates and five attempts per draw, the exact union bound over
moduli 28 through 64 is below `2^-129.15`; this consumes 740 candidate bytes
before hash-expander framing. A 16-bit decoder needs 13 attempts to push the
same union bound below `2^-132.93` and consumes 962 candidate bytes. Even with
BLAKE3, producing that many constrained XOF bytes requires many compression
calls. This is why enumerative decoding is the baseline.

The first Groth16 Fisher-Yates backend can use a dense 64-entry array with
one-hot or bitwise index selection. Its swap cost is only thousands of simple
constraints; the hash expansion is expected to dominate. The PLONK backend MAY
lower the same semantic swaps through a lookup/RAM argument. Both backends MUST
match the concrete partial Fisher-Yates output exactly.

Witnessing 37 positions and proving only range/distinctness is not enough to
bind them to the transcript. A permutation/grand-product proof still needs an
exact transcript decoder. A hash-seeded affine permutation or small ad hoc PRP
is not an acceptable shortcut: its support and pairwise-unit argument define a
different challenge family.

### Transcript Choice

The circuit transcript and the Solidity wrapper transcript solve different
problems. Solidity has a cheap `KECCAK256` opcode and SHA-256 precompile. A
Groth16 or PLONK circuit must constrain every bit operation; it receives no
benefit from those EVM operations.

`P2` verifies the complete `P1` verifier, not only its Akita opening proof.
The circuit therefore constrains two Fiat-Shamir states: Jolt's outer
transcript and Akita's nested transcript. Current Jolt native batching first
absorbs the Akita setup and opening statement into the Jolt transcript,
squeezes one Jolt field challenge, and absorbs that challenge into a fresh
Akita transcript. The Akita proof bytes are later absorbed back into Jolt. The
first implementation SHOULD preserve this composition and its domain
separation; replacing it with one shared state would change the protocol and
requires a separate security analysis.

The preferred circuit target SHOULD be BLAKE3 for both states. It uses 32-bit
ARX operations like BLAKE2s, but only seven compression rounds, and its root
XOF can derive a 32-byte next state and a 16-byte field challenge from one
64-byte output block. Jolt PR
[#1837](https://github.com/a16z/jolt/pull/1837) already contains a streaming
keyed-BLAKE3 transcript and a bit-R1CS compression gadget. At the pinned PR
head `89f73577af3ea3e709ddb2557be62d1afb462675`, one variable-IV compression
plus feed-forward is asserted to use 15,792 R1CS constraints; the fixed-IV
case uses 15,408. This is direct evidence that BLAKE3 can be much cheaper than
the current standard-hash alternatives when the Boolean lowering is designed
for it.

A small gnark prototype at commit
[`fd5c2443d59970eb1c3e4202fb8f10a23ef60632`](https://github.com/Consensys/gnark/tree/fd5c2443d59970eb1c3e4202fb8f10a23ef60632)
implemented the same seven-round compression through gnark's `uints.U32`
API. It compiled the following fixed-length digest circuits, each with a
32-byte public-output equality:

| Input bytes | BLAKE3 R1CS | SHA-256 R1CS | BLAKE3 SCS | SHA-256 SCS |
|---:|---:|---:|---:|---:|
| 32 | 76,496 | 165,986 | 229,266 | 497,472 |
| 64 | 76,560 | 200,663 | 229,506 | 601,761 |
| 740 | 192,440 | 554,068 | 580,912 | 1,662,424 |

Thus the unoptimized gnark BLAKE3 port used about 54% fewer constraints at 32
bytes and 62% fewer at 64 bytes; over 740 bytes it used about 65% fewer. The
prototype is cost evidence, not an audited gadget or a committed artifact. It
also understates the cost of a Fisher-Yates counter expander, which must
generate 740 output bytes rather than merely hash a 740-byte message.

SHA-256 remains the fallback because gnark ships a maintained gadget and its
standardization and security margin are conservative. It was also cheaper than
legacy Keccak-256 in the earlier direct BN254 benchmark. At the same gnark
commit
[`fd5c2443d59970eb1c3e4202fb8f10a23ef60632`](https://github.com/Consensys/gnark/tree/fd5c2443d59970eb1c3e4202fb8f10a23ef60632),
compiling a variable-byte digest circuit with a 32-byte output equality gave:

| Input bytes | SHA-256 R1CS | Keccak-256 R1CS | SHA-256 SCS | Keccak-256 SCS |
|---:|---:|---:|---:|---:|
| 0 | 165,326 | 237,566 | 495,712 | 741,543 |
| 32 | 165,922 | 237,630 | 497,312 | 741,735 |
| 64 | 200,599 | 237,694 | 601,601 | 741,927 |
| 96 | 201,196 | 237,758 | 603,197 | 742,119 |
| 128 | 235,871 | 237,822 | 707,486 | 742,311 |

These counts are selection evidence, not a final `P1` verifier estimate. The
final benchmark MUST count the actual Jolt and Akita absorb/squeeze schedules,
including the bridge, and MUST pin the gnark version. The compiler SHOULD batch
pending transcript bytes into fixed blocks, avoid hashing static profile
material repeatedly, and derive multiple jointly analyzed values from one
digest only when the proof gives a joint soundness bound for doing so.

Adopting BLAKE3 requires an audited gadget, exact Rust/IR/Go vectors, and a
security review of the keyed rolling transcript rather than only the
compression function. BLAKE3's specification targets 128-bit security; the
profile MUST state the classical and quantum transcript-security claims
explicitly and MUST NOT infer a 256-bit margin merely from the output length.
Current gnark exposes SHA-2 and SHA-3 standard gadgets but no BLAKE3 package.
Keccak SHOULD remain the Solidity-facing default for generated PLONK verifier
transcripts. `P1`, including the nested Akita proof, remains private witness
data to `P2`; the contract replays neither the Jolt nor Akita transcript.

Public application bytes SHOULD be reduced to a small fixed digest before they
become Groth16/PLONK public inputs. If the contract must bind raw calldata, it
can compute SHA-256 through precompile `0x02` and pass two canonical 128-bit
limbs. Otherwise the contract API may accept an already authenticated digest.
The circuit MUST recompute or otherwise soundly inherit the same digest; merely
accepting prover-chosen digest limbs is insufficient.

### Circuit Terminal

The current terminal decodes a variable-length Golomb-Rice payload, checks an
optional L2 norm, multiplies sparse challenges by ring shifts, and performs an
exact CRT/NTT matrix-vector product. Those are sensible native operations and
poor circuit operations.

The circuit profile SHOULD expose a new fixed-width terminal proof. Suppose the
remaining relation includes `m` public A rows and a revealed small vector `z`.
The verifier first batches rows with a fresh field challenge `rho`:

```text
A_rho[j] = sum_i rho^i A[i,j]
t_rho    = sum_i rho^i t[i]
```

It then verifies the matrix product through a multilinear inner-product
sumcheck:

```text
sum_j A_rho[j] * z[j] = t_rho.
```

At the final sumcheck point `r`, the circuit checks
`A_rho_tilde(r) * z_tilde(r)`. It computes `z_tilde(r)` from the fixed-width
terminal digits and computes `A_rho_tilde(r)` from setup constants pinned by
the circuit key. Consistency rows can be included in the same batched relation
when the degree bound permits, or in a second fixed relation sumcheck.

Every coordinate of `z` uses a fixed signed-bit representation and a canonical
field reconstruction. Coordinatewise range checks replace both Golomb-Rice
canonicality and the optional terminal L2 scan. Proof size can increase because
the terminal witness is private input to the short outer proof.

This construction removes NTT and CRT, but it does not make setup work vanish.
The final setup MLE still has a dynamic point. In R1CS, multiplying an equality
weight by a hard-coded setup coefficient is linear, while constructing all
equality weights costs field multiplications. The setup-offloading planner MUST
therefore minimize the residual fixed MLE width and stop at a hard maximum.

### Maximum Setup Offloading With a Finite Base Case

Existing Akita can offload setup contributions only on nonterminal edges. A
later fold authenticates the carried setup-prefix opening, and the terminal has
no successor, so it always evaluates setup directly. Current Jolt integration
additionally rejects provisioned schedules containing any recursive
setup-prefix contribution in `crates/jolt-akita/src/schedule_registry.rs`.

The circuit profile requires both restrictions to change:

1. Jolt's shape guard and schedule registry must admit the exact fixed
   setup-prefix topology.
2. A `CircuitTerminal` must consume the last incoming setup-prefix claim and
   close it against a small fixed setup MLE.

The planner SHOULD offload every earlier contribution for which carrying the
opening reduces the complete circuit cost. It then emits enough ordinary folds
to reduce the witness and setup prefix below fixed terminal caps. The terminal
recomputes the remaining setup MLE from circuit constants. No setup evaluation
is accepted solely because the prover supplied it.

An alternative is to evaluate the complete constant setup MLE directly in the
circuit. The terminal gate compares that full circuit with recursive
offloading. Because constant linear combinations can be cheaper in R1CS than
their native operation count suggests, “maximum” means the measured
circuit-cost optimum, not mechanically selecting every possible Stage 3 edge.

### Symbolic Verifier and Compiler

The compiler SHOULD use a typed, explicit verifier IR rather than making Rust
field operators or equality comparisons perform hidden recording side effects.
The minimum value domains are:

- `FieldVar` for BN254 `Fr`;
- `BoolVar`;
- fixed-size `ByteVar` arrays;
- `SmallSignedVar<bits>`; and
- public and private input handles with stable source labels.

The semantic verifier uses explicit operations such as `add`, `mul`,
`assert_equal`, `range_check_signed`, `standard_hash_compress`, `sample_shell`,
and `evaluate_fixed_mle`. Assertions are never inferred from `PartialEq`, and
IR state is passed explicitly rather than through thread-local storage.

```mermaid
flowchart LR
    S["Canonical semantic verifier"] --> R["Scalar reference backend"]
    S --> N["Optimized native backend"]
    S --> I["Typed SSA/DAG IR backend"]
    I --> G["gnark R1CS / Groth16"]
    I --> H["R1CS / Spartan + HyperKZG"]
    I --> P["gnark SCS / PLONK"]
    R --> T["Twin and mutation tests"]
    N --> T
    G --> T
    H --> T
    P --> T
```

High-level primitives are allowed only when they have one documented semantic
contract and independent reference/compiled implementations. For example, the
native backend may lower `evaluate_fixed_mle` to a cache-friendly kernel while
the circuit backend emits equality weights. Neither backend may change the
claimed polynomial.

The IR MUST be deterministic, serializable, versioned, source-labelled, and
hashable. It MUST record every assertion explicitly and expose cost counters.
The circuit generator MUST reject unused witness nodes, unconsumed proof
fields, backend operations without a lowering, and profile-dependent control
flow not fixed at compile time.

Jolt PR
[#1322](https://github.com/a16z/jolt/pull/1322) demonstrated a Rust symbolic
verifier to JSON IR to gnark/Groth16 pipeline. It also used a separate
`TranspilableVerifier`, thread-local value tunneling, and `PartialEq` assertion
recording. This design SHOULD reuse the proven Rust-to-gnark path and Jolt's
current `jolt-claims` symbolic relation definitions, but SHOULD NOT reproduce
those hidden-state boundaries. One canonical protocol program and explicit IR
builder better match Akita's single-source-of-truth rule.

### Fixed Circuit Classes

Groth16 circuits are shape-specific, and the user-visible verifying key must not
depend on proof data. The first release SHOULD define one or a few named classes
such as:

```text
LATTICE_JOLT_WRAP_V1_SMALL
LATTICE_JOLT_WRAP_V1_MEDIUM
LATTICE_JOLT_WRAP_V1_LARGE
```

Each class fixes maximum trace variables, advice/program layout, Akita schedule,
setup seed, proof shape, transcript schedule, terminal caps, and public-input
count. Smaller statements pad canonically. A class has one circuit ID, IR
digest, backend-specific keys, Solidity contract, and benchmark record.

The initial public Akita component SHOULD use BN254 `Fr` as both field types,
degree-one extensions, uniform D64 rings, `EvaluationTrace`, reduced evaluation
where algebraically permitted, the certified D64 shell, no operator rejection,
one fixed standard transcript, and one fixed external catalog. Setup uses a
fixed seed and a finite offloading plan. The terminal is the fixed
matrix-product/MLE sumcheck described above. Coefficient packing, response
compression, adaptive schedules, and native-verifier terminal NTTs are disabled
unless their complete circuit lowering reduces constraints.

Schedule search SHOULD rank candidates first by standard-hash compressions,
then Boolean and range constraints, setup-MLE multiplications, remaining field
multiplications, proof bytes, and finally native prover time. This ordering is
the default only until measured backend weights replace it.

Key generation MUST NOT record one unverified reference run to discover the
accepted transcript schedule. The schedule is generated from the public
profile, and a reference proof is only a test fixture. Ceremony and key
rotation policy are separate deployment workstreams.

### Jolt Integration

Current Jolt already has a native BN254 `Fr` backend, a generic prover/verifier
surface, symbolic sumcheck relations in `jolt-claims`, and an Akita adapter.
However, `crates/jolt-akita/src/adapters.rs` currently fixes `AkitaField` to
Akita's fp128 field, delegates only fp128 schedules, and contains fp128-specific
one-hot commitment kernels. The BN254 cutover therefore requires a field-generic
adapter boundary and either generic one-hot kernels or a BN254 implementation.

Jolt and Akita also have distinct transcript implementations today. In
`crates/jolt-akita/src/native_batching.rs`, the adapter binds setup and opening
claims to Jolt's transcript, squeezes a Jolt field element, and absorbs it into
a fresh Akita transcript through `bridge_jolt_statement_challenge`. The proof
stores that bridge encoding and is then appended to Jolt's transcript. The
symbolic verifier MUST reproduce this nesting. The initial circuit profile
SHOULD use the same standard primitive for both states to share one audited
bit-gadget lowering, while retaining the separate domains and bridge.

The final two-proof path is:

1. Produce `P0` with fp128 lattice Jolt and Akita.
2. Run a fixed RISC-V guest that strictly decodes `P0`, reconstructs the exact
   public statement, verifies Jolt and Akita, and returns success.
3. Prove that guest execution with Jolt over BN254 `Fr`, using
   `Bn254CircuitV1` Akita for its polynomial openings, producing `P1`.
4. Run the generated semantic BN254 Jolt/Akita verifier on `P1` to produce the
   circuit witness.
5. Prove the circuit with Groth16, Spartan plus HyperKZG, or PLONK, producing
   `P2`.
6. Verify `P2` in Solidity.

Inside step 2, fp128 arithmetic is ordinary RISC-V word computation proved by
BN254 Jolt; it is not emulated again in the final Groth16/PLONK circuit. Jolt
inlines for fp128 multiplication/reduction, standard hashes, and verifier input
decode MAY reduce the guest trace, provided the RISC-V semantics and Jolt
constraints are reviewed.

The existing `profile/akita-recursion/` harness is the starting measurement
tool. Its trusted expanded-matrix input path is not a production trust boundary.
A production guest MUST either derive setup from a pinned seed, authenticate a
prepared setup artifact, or bind setup through the proof/circuit profile.

### Wrapper Outputs

The same IR SHOULD compile through gnark's R1CS and SCS builders. The first
Groth16 artifact is the production candidate because a BN254 Groth16 verifier
uses a constant number of pairing checks plus work linear in the small public
input vector. Ethereum's EIP-1108 prices BN254 addition at 150 gas,
multiplication at 6,000 gas, and a `k`-pair pairing check at
`45,000 + 34,000*k`, before call, memory, calldata, and contract overhead.
Actual gas MUST come from the generated contract.

Spartan plus HyperKZG is the second backend to evaluate because Jolt's current
wrapper work already supplies a relevant R1CS-to-on-chain path. Akita's verifier
relation is native BN254 scalar-field arithmetic, so it does not inherit the
Dory path's deferred non-native curve and pairing subproof. The backend can
commit to and open the same emitted R1CS relation, but it MUST consume the same
public statement and IR digest as Groth16. Reuse of wrapper infrastructure does
not justify a separately maintained verifier program.

PLONK is retained because a universal SRS and easier circuit-class changes may
outweigh its larger proof and verifier. gnark currently exports Solidity
verifiers for both BN254 Groth16 and BN254 PLONK. The generated code and exact
gnark revision are part of the deployed artifact and require independent EVM
tests.

Public inputs SHOULD remain a small fixed vector of canonical BN254 elements.
Every additional Groth16 public input adds a verifying-key scalar multiplication,
so large statements are hashed into fixed limbs.

### Why This Is Simpler Than Wrapping Dory

The current Dory wrapper effort in Jolt PR
[#1837](https://github.com/a16z/jolt/pull/1837), head
`56343ddbea18b5021b6971b7c2c3f17d1a67726f`, is a sophisticated single-layer
Spartan/HyperKZG construction. Its field-arithmetic R1CS is only 5,254
constraints, but the deferred Dory final check requires a 201,575-row,
149-column limb table for G1/G2 multi-scalar multiplication, a four-pair Miller
loop, final exponentiation, subgroup checks, GT norm-one checks, canonical
extension-field limbs, and guarded incomplete curve additions. The reported
default proof is 7,392 payload bytes and 7,533 bincode bytes. On the reported
M4 Mac mini run with ten threads, the online wrapper prover takes roughly
16.5--16.9 seconds on an idle machine, and its op-count EVM model is about 4.94
million gas. The PR does not yet include a Solidity verifier.

The proposed Akita circuit has no inner curve points, pairings, Miller loop,
final exponentiation, subgroup checks, or Fq/Fq2/Fq12 canonicality. Its hard
parts are standard hashes, small integer ranges, exact sparse-challenge
decoding, and setup/witness MLEs. This is a much smaller semantic attack
surface.

The comparison is not one-sided. The Dory work is single-layer and already has
concrete measurements. The Akita field-switch path adds an entire Jolt proof of
the fp128 verifier, and current Akita-in-Jolt traces can be extremely large.
The justified expectation is **easier circuit engineering and audit**, not yet
lower prover time or gas.

### Alternatives Considered

**Direct EVM Akita verification.** Rejected for the initial route. BN254
`ADDMOD`/`MULMOD` and `KECCAK256` help, but EVM has no native ring fold,
sumcheck, NTT, sparse sampler, or range-proof verifier. A direct implementation
would expose a large bespoke bytecode verifier and likely cost far more gas than
one outer SNARK.

**Directly arithmetize the fp128 verifier in Groth16/PLONK.** Rejected as the
main design because every fp128 field operation becomes non-native arithmetic.
It remains a useful small benchmark to determine whether the extra Jolt proof
is actually justified.

**Run the application directly over BN254.** Retained as bootstrap mode. It
removes the field-switch proof but makes the complete application prover pay
for 254-bit field arithmetic.

**Reuse the current native terminal.** Rejected for the circuit profile because
Golomb decode, CRT/NTT, prepared caches, and optional L2/operator-norm work are
not the desired circuit language.

**Poseidon transcript.** Explicitly out of scope. It would be cheaper in field
constraints but changes the requested standard-hash trust and compatibility
surface.

**Keccak as the in-circuit Jolt/Akita transcript.** Not selected initially. It
aligns with the EVM opcode but is substantially more expensive than both
BLAKE3 and SHA-256 in the current BN254 evidence. The contract replays neither
proof transcript, so EVM opcode alignment provides little benefit.

**Hand-write separate wrapper verifiers.** Rejected. Groth16, Spartan plus
HyperKZG, and PLONK must consume one semantic verifier IR. The maintenance and
soundness risk of protocol-specific rewrites is exactly what the compiler is
intended to remove.

## Execution

### Critical Path

The merge-order critical path is:

```text
#466 and #428 stabilization
  -> Jolt Fr capability traits
  -> canonical 256-bit decomposition
  -> exact BN254 SIS profile and tables
  -> scalar-reference D64 end to end
  -> native Fr radix-2 NTT
  -> exact sampler and transcript
  -> fixed Bn254CircuitV1 catalog and schedule
  -> circuit terminal and finite setup closure
  -> semantic verifier IR
  -> Jolt integration
  -> Groth16, Spartan+HyperKZG, and PLONK wrappers
```

The first critical risk is not the fully split ring. It is whether the 254-bit
decomposition and scalar backend reproduce every Akita invariant without an
implicit `u128` assumption. The later critical risk is compiler completeness:
the terminal, transcript, and every proof field must reach an explicit circuit
assertion.

The sequence below uses PR-sized units. Review branches MAY be stacked, and the
Jolt capability change can proceed while Akita's base PRs stabilize. A later
unit MUST NOT publish a stable profile or generated key before all earlier
language-defining units are merged.

### Ordered Change Series

1. **PR 0 — Stabilize the base stack.** Rebase or merge the quotient-free work
   from #466 and the external-catalog work from #428. Record exact Akita and
   Jolt revisions in the implementation worklog. Resolve schedule and catalog
   interfaces before adding BN254-specific APIs.

   **Acceptance gate:** `main` has one canonical schedule/catalog path and all
   existing fp128 tests pass unchanged.

2. **PR 1 — Implement Jolt `Fr` capability traits.** In Jolt, implement
   `ExtField<Fr>`, `MulBaseUnreduced<Fr>`, `Unreduced`,
   `WithCommitAccumulator`, `Fold`, and `WithPacking`. Begin with degree-one
   identities, immediate reductions, and `NoPacking<Fr>`.

   **Acceptance gate:** trait-law and randomized differential tests against
   arkworks cover canonical encoding, base multiplication, accumulator
   reduction, and folding.

3. **PR 2 — Add the Akita `bn254` feature and strict encoding.** Wire
   `jolt-field::Fr` through field identity and `FpExtEncoding`. Use canonical
   32-byte little-endian values and reject noncanonical inputs. Keep feature
   dependencies narrow.

   **Acceptance gate:** Solinas-only, BN254-only, and combined feature graphs
   build and test; malformed BN254 encodings reject without panic.

4. **PR 3 — Introduce canonical 256-bit field integers.** Add one fixed-width
   integer abstraction and route modulus comparison, centered
   representatives, balanced decomposition, reconstruction, and source-bound
   calculations through it. Retain the existing `u128` specialization where
   appropriate. Test decomposition bases one through eight and values around
   zero, `q/2`, and `q - 1`.

   **Acceptance gate:** every selected digit sequence reconstructs exactly,
   bounds are checked before allocation, endpoint tests do not panic, and no
   BN254 verifier path calls `to_u128_checked()` on a field representative.

5. **PR 4 — Add exact BN254 profile dispatch.** Give the profile its full
   modulus bytes, 254-bit width, stable tag, descriptor encoding, table ID,
   cache naming, and proof-size accounting. Support D64 only at this stage.

   **Acceptance gate:** a one-bit modulus change fails config and table
   selection; BN254 cannot dispatch to a Q128 table or cache artifact.

6. **PR 5 — Generate and verify BN254 SIS tables.** Add an exact BN254
   estimator constructor and stable label. Generate the `Linf` rows and
   selected `L2` rows needed by the finite D64 fixture, with provenance and
   digests. Exercise one-below and one-above boundary values.

   **Acceptance gate:** every scheduled matrix query resolves to a generated
   BN254 row, deliberately missing or undersized rows fail closed, and an
   independent report reproduces the selected security values.

7. **PR 6 — Remove accidental pseudo-Mersenne bounds.** Audit generic ring,
   prover, and verifier APIs. Retain `PseudoMersenne` only on kernels that
   actually use Solinas reduction; replace broad convenience bounds with the
   narrow field capabilities each operation requires.

   **Acceptance gate:** the generic BN254 scalar verifier type-checks without
   claiming pseudo-Mersenne structure, and existing fp128 optimized kernels
   retain their specialized bounds.

8. **PR 7 — Add the scalar reference backend.** Implement
   `ScalarReferenceBackend<F>` using schoolbook field and polynomial
   arithmetic. Add the test-only `Bn254D64Fixture`. Complete setup, commit,
   prove, and verify for dense and one-hot inputs, plus mutation negatives.

   **Acceptance gate:** this is the decisive foundation milestone. No native
   BN254 NTT or public BN254 config proceeds until both fixtures pass under
   sanitizers or equivalent checked test builds.

9. **PR 8 — Add the native BN254 radix-2 backend.** Implement native cyclic
   and negacyclic `Fr` NTTs, twiddle generation, cache identities, and inverse
   normalization. Do not route BN254 through the Q128 auxiliary-CRT enum.

   **Acceptance gate:** forward/inverse transforms, multiplication, folding,
   setup, proof generation, proof bytes, and verification agree with the scalar
   backend for every relevant D64 operation.

10. **PR 9 — Bind certified challenge-family descriptors.** Define the
    descriptor fields for ambient dimension, allowed mask, signed-magnitude
    counts, entropy width, decoder, deterministic bounds, certificate digest,
    and rejection policy. Import reproducible D64 `(24,13)` and D128
    72-position `(38,5)` certificates.

    **Acceptance gate:** descriptor mutations change the profile identity and
    fail verification; exact support and unit-difference checks replay from
    committed artifacts.

11. **PR 10 — Implement the 128-bit enumerative decoder.** Specify one
    combinadic order and implement the injective D64 mapping. Retain partial
    Fisher-Yates under a separate legacy decoder ID; do not silently change its
    distribution.

    **Acceptance gate:** native and independent reference implementations agree
    at first, last, and internal rank boundaries; injectivity, position
    distinctness, magnitude counts, signs, and no-rejection behavior are tested.

12. **PR 11 — Freeze the circuit transcript.** Implement keyed rolling BLAKE3
    as the leading candidate and SHA-256 as the maintained fallback. Specify
    domain tags, framing, absorb and squeeze order, field reduction, proof
    encoding, and the Jolt-to-Akita bridge. Give each transcript a distinct
    profile identity.

    **Acceptance gate:** logging, optimized native, scalar reference, IR, and
    independent gadget vectors agree on every event and challenge. A hash
    audit selects the production transcript before `Bn254CircuitV1` freezes.

13. **PR 12 — Materialize `Bn254CircuitV1` and its external catalog.** Freeze
    D64, evaluation trace, reduction modes, source classes, digit bases, setup
    seed, challenge descriptor, transcript, and a small fixed schedule family.
    Instrument the complete verifier and search candidates through basis eight
    using the circuit-cost order in this spec.

    **Acceptance gate:** schedule generation is deterministic; the profile
    contains no coefficient packing, response compression, operator rejection,
    adaptive branch, or native-terminal optimization unless a complete circuit
    benchmark proves a constraint reduction.

14. **PR 13 — Add the circuit terminal and setup closure.** Introduce explicit
    `DirectNtt` and `MatrixProductSumcheck` terminal modes. Implement the
    batched relation, small signed-digit MLE, fixed setup MLE, range checks, and
    finite setup-prefix closure. Compare recursive offloading with direct
    constant setup evaluation.

    **Acceptance gate:** the `Bn254CircuitV1` path reaches no NTT, auxiliary
    CRT, Golomb-Rice codec, operator-norm check, floating point, prepared native
    cache, or unauthenticated setup value.

15. **PR 14 — Build the semantic verifier IR.** Define explicit field, Boolean,
    byte, range, transcript, sampler, fold, sumcheck, MLE, and terminal
    operations. Port the verifier slice by slice: input and range validation,
    transcripts, sumchecks, sampler, folding, terminal, then the full program.
    Serialize and hash the deterministic IR.

    **Acceptance gate:** scalar, optimized, and interpreted executions agree on
    the accepted corpus and mutation corpus; every input is consumed, every
    assertion is explicit, and every operation has a compiler lowering.

16. **PR 15 — Generalize `jolt-akita`.** Remove the adapter's fp128 hard-coding
    and expose one semantic integration path with concrete
    `Fp128RecursionGuest` and `Bn254CircuitV1` bundles. First prove and verify a
    direct BN254 Jolt program. Then freeze the strict fp128-verifier guest and
    produce `P1` for `P0`.

    **Acceptance gate:** mutations at both proof layers reject, the bridge and
    nested transcript match the standalone verifiers, and guest cycles and
    memory satisfy the predeclared go/no-go threshold.

17. **PR 16 — Emit and benchmark wrapper backends.** Compile the same IR/R1CS
    to Groth16 first, Spartan plus HyperKZG second, and PLONK third. Pin tool
    revisions, setup artifacts, circuit IDs, keys, public-input layouts, and
    generated Solidity. Measure constraints, prover time and memory, key size,
    proof bytes, bytecode size, calldata, and verification gas.

    **Acceptance gate:** direct BN254 and `P0 -> P1 -> P2` fixtures verify in an
    EVM harness; every public input and proof field has a mutation test; the
    report compares all three backends with the commit-pinned Dory wrapper.

### Parallel Evidence Track

Cost instrumentation SHOULD begin once PR 2 provides stable field encoding and
MUST finish before PR 12 freezes the catalog. It records Jolt and Akita
transcript messages separately, including the bridge, plus digit widths,
sampler work, setup coefficients, terminal coordinates, and field operations.
The track also reproduces BLAKE3, SHA-256, and Keccak gadget benchmarks at
pinned revisions, measures a direct fp128-verifier circuit as a control, and
records `profile/akita-recursion` guest cycles and memory at small and
production-representative sizes.

After PR 16, an independent audit MUST cover the challenge certificates,
transcript and sampler, terminal identity, setup closure, compiler completeness,
statement binding, and generated contracts. Deployment proceeds only after the
full benchmark is reproduced on a fixed machine and EVM fork. The valid final
decision is Groth16, Spartan plus HyperKZG, PLONK, more than one backend, or
stop; completing the implementation does not predetermine deployment.

## Risks and Open Questions

1. **The 254-bit port can fail before circuit work begins.** Balanced
   decomposition, centered representatives, profile dispatch, and source
   bounds cross many `u128` assumptions. The scalar-reference milestone is a
   correctness gate, not optional scaffolding.
2. **Guest recursion may dominate everything.** The extra `P1` proof is the
   largest performance risk. A generic RISC-V fp128 verifier may be too large
   even if the final circuit is excellent.
3. **Standard hashes may dominate the circuit.** `P2` constrains the complete
   Jolt transcript as well as Akita's nested transcript. Even BLAKE3 can
   produce millions of constraints when replayed hundreds of times.
   Transcript scheduling is likely more important than field multiplication
   tuning.
4. **Setup offloading may not be cheapest in R1CS.** Constant setup
   coefficients enter linear combinations cheaply. The planner must measure
   the complete circuit rather than inherit the native setup-scan objective.
5. **BN254 SIS work is substantial.** Power-of-two ring compatibility avoids a
   non-cyclotomic redesign, but it does not avoid wide-modulus plumbing, table
   generation, decomposition retuning, or security review.
6. **The native NTT can disagree with the protocol oracle.** Complete splitting
   makes an `Fr` NTT available, but does not prove twiddle order, negacyclic
   twisting, inverse normalization, or cache identity. Differential tests must
   remain permanent.
7. **Fixed circuit classes create key operations.** Groth16 needs a setup per
   class, and every backend needs versioned verifying keys and deployment
   policy.
8. **Compiler soundness becomes a security boundary.** A missed assertion,
   unused proof input, or mismatched hash encoding can accept false statements
   even when the Rust verifier is correct.
9. **The final proof is classical.** The wrapper should be described as
   on-chain verification of a lattice proof, not a post-quantum on-chain proof.
10. **Jolt and Akita move quickly.** Commit-pinned evidence and generated
   artifacts are mandatory; branch-head assumptions are not durable.

## Documentation

This proposed design remains in `specs/` until implementation stabilizes. It
must not be described as current behavior in the Akita Book. When the BN254
profile and compiler ship, durable material should be split between:

- `book/src/foundations/rings-and-fields.md` for BN254 and D64 rings;
- `book/src/how/configuration.md` for the fixed circuit profile;
- `book/src/how/transcript.md` for the standard-hash transcript and sampler;
- `book/src/how/verification.md` for the semantic verifier and circuit
  terminal;
- `book/src/how/setup-offloading.md` for terminal setup closure; and
- a new usage chapter for Jolt wrapping and Solidity deployment.

The spec should then be marked implemented, folded, and archived under the
normal lifecycle policy.

## References

- Akita terminal implementation:
  `crates/akita-verifier/src/protocol/core/terminal_direct.rs` and
  `crates/akita-verifier/src/protocol/core/terminal_ntt.rs`.
- Akita setup offloading explanation:
  `book/src/how/setup-offloading.md`.
- Akita transcript and sparse sampler:
  `crates/akita-transcript/src/sponge.rs` and
  `crates/akita-challenges/src/sampler/position_sample.rs`.
- Jolt transcript and Akita bridge:
  `crates/jolt-transcript/src/lib.rs`,
  `crates/jolt-akita/src/native_batching.rs`, and
  `crates/jolt-akita/src/adapters.rs` in the Jolt repository.
- Akita verifier-in-Jolt measurements:
  `profile/akita-recursion/README.md`.
- Exact BN254 D64 challenge certificate:
  [commit `743fbeb830423d4430c043b6a220f9b47876511e`](https://github.com/LayerZero-Labs/akita/tree/743fbeb830423d4430c043b6a220f9b47876511e/scripts/bn254_challenge_units).
- Exact BN254 D128 S72 challenge certificate and larger-ring embedding:
  [commit `ff4481b2e59f3c9d09016ac97045ef65ea4af490`](https://github.com/LayerZero-Labs/akita/tree/ff4481b2e59f3c9d09016ac97045ef65ea4af490/scripts/bn254_challenge_units).
- Jolt symbolic verifier/Groth16 transpiler:
  [a16z/jolt PR #1322](https://github.com/a16z/jolt/pull/1322).
- Current Jolt/Dory wrapper:
  [a16z/jolt PR #1837](https://github.com/a16z/jolt/pull/1837).
- BLAKE3 specification and security rationale:
  [BLAKE3 specification source](https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.tex).
- gnark Groth16 BN254 Solidity exporter:
  [BN254 verifier source](https://github.com/Consensys/gnark/blob/master/backend/groth16/bn254/verify.go).
- gnark PLONK BN254 Solidity verifier:
  [BN254 Solidity source](https://github.com/Consensys/gnark/blob/master/backend/plonk/bn254/solidity.go).
- gnark SHA-2 and SHA-3 gadgets:
  [`std/hash`](https://github.com/Consensys/gnark/tree/master/std/hash).
- Ethereum BN254 precompile pricing:
  [EIP-1108](https://eips.ethereum.org/EIPS/eip-1108).
- Ethereum SHA-256 and other precompiled contracts:
  [Ethereum Yellow Paper](https://ethereum.github.io/yellowpaper/paper.pdf).
