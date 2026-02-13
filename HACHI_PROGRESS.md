## Hachi PCS implementation progress

This file is the **single source of truth** for implementation status and near-term priorities.

### Goals (project-level)

- **Production-ready implementation**: correctness, security, maintainability, and performance are first-class goals.
- **Standalone codebase**: implementation and comments should stand on their own; external acknowledgements live in `README.md`.
- **Constant-time cryptographic core**: arithmetic and protocol-critical paths must be constant-time with respect to secret data.
- **No shortcuts / no fallback design**: avoid temporary or degraded code paths in the core implementation.

### Non-negotiable requirements

- **Constant-time discipline**
  - No secret-dependent branches or memory access patterns in cryptographic hot paths.
  - No secret-indexed table lookups; table access patterns must be independent of secret data.
  - Keep data representations and reductions explicit and auditable for timing behavior.
  - Add targeted tests/reviews for constant-time-sensitive code as features land.
- **Code quality bar**
  - Clear naming, explicit invariants, small cohesive modules, and API docs for public interfaces.
  - No placeholder crypto logic in mainline code (no "temporary" arithmetic shortcuts).
  - Tests are required for correctness-critical arithmetic before dependent protocol code is built.
  - No section-banner comments (e.g., `// ---- Section ----`, `// === ... ===`). Let the code and doc-comments speak for themselves.
- **Standalone implementation policy**
  - Do not mention external inspirations/ports in core code comments.
  - Keep terminology and structure internally coherent and project-native.
  - Keep external attribution limited to dedicated docs (for now: `README.md` acknowledgements).
- **Git discipline**
  - Do not commit or push without explicit user approval.

### Implementation workflow (cautious + approval-driven)

- Before each major subsystem, present implementation options with trade-offs.
- Seek explicit approval before proceeding with a selected option.
- Pause at milestone boundaries for review and feedback before continuing.
- Prefer slow, verifiable progress over rapid, high-risk changes.
- Ask for user input frequently when requirements are ambiguous or involve design trade-offs.

### Definition of Done (all crypto-critical work)

- **Security / constant-time**
  - Secret-independent control flow and memory access in cryptographic paths.
  - Constant-time review notes included for non-trivial arithmetic/ring changes.
- **Correctness**
  - Unit tests for edge cases and algebraic identities.
  - Cross-check vectors/reference checks added where practical.
- **Code quality**
  - Clear naming, explicit invariants, and no placeholder logic in core paths.
  - Public interfaces documented sufficiently for safe usage.
- **Performance**
  - Hot-path performance impact evaluated (benchmark or measured rationale).
- **Tooling + CI**
  - `cargo fmt --all --check` passes.
  - `cargo clippy --all --all-targets --all-features` passes.
  - `cargo test` (or targeted suite for touched modules) passes.
- **Process**
  - Implementation options reviewed with user before major subsystem changes.
  - Milestone update recorded in this file.

### Scope (current)

- **Implemented so far (Phase 0 + Phase 1 functional core)**: prime fields (32/64/128-bit representations), extension fields, cyclotomic `R_q = Z_q[X]/(X^d + 1)`, CRT+NTT representation, backend/domain layering, ring automorphisms, and functional gadget decomposition.
- **Phase 2+ protocol status**: interface scaffold plus ring-native §4.1 commitment core are present (`Transcript`, Blake2b/Keccak backends, phase-grounded labels, `RingCommitmentScheme`, config layer, and setup/commit implementation). Open-check prover/verifier paths remain stubbed.
- **Deferred future phase**: integration into Jolt (replacement of Dory with Hachi) is intentionally out of current execution scope; cross-repo analysis is design input only.

### Critical review snapshot (2026-02-13)

- **Phase 1 functional milestone appears complete**
  - Ring/gadget components listed in Phase 1 are implemented and currently checked off.
  - Conversion and arithmetic paths in coefficient and CRT+NTT domains are exercised by passing tests.
- **Not yet "production-ready" despite functional completion**
  - Constant-time hardening follow-ups remain in inversion and final CRT projection paths (see `CONSTANT_TIME_NOTES.md`).
  - Current ring multiplication in coefficient form remains `O(D^2)` schoolbook (`src/algebra/ring/cyclotomic.rs`), with CRT+NTT available as the faster domain path.
- **Tooling/quality gate status (current branch snapshot)**
  - `cargo test` passes, including protocol transcript/label/commitment contract tests and new ring-commitment core/config/stub tests.
  - `cargo fmt --all --check` passes.
  - `cargo clippy --all --all-targets --all-features` passes.
- **Phase 2 scaffold + commitment core landed; proof-system work still pending**
  - `src/protocol/*` now provides transcript + commitment abstraction boundaries with `Transcript` naming.
  - Two transcript backends are wired (`Blake2bTranscript`, `KeccakTranscript`) with deterministic replay/order/reset tests.
  - Hachi-native labels are now calibrated to paper-stage phases (§4.1, §4.2, §4.3, §4.5).
  - Commitment absorption is label-directed at call sites (`AppendToTranscript` no longer hardcodes commitment labels).
  - Ring-native commitment setup/commit flow for §4.1 is implemented in `src/protocol/commitment/commit.rs` behind `RingCommitmentScheme`.
  - Prover/verifier split folders are wired with explicit stubs (`src/protocol/prover/stub.rs`, `src/protocol/verifier/stub.rs`) for future open-check implementation.
- **Conclusion**
  - Treat **Phase 1 as functionally complete**.
  - Treat **Phase 2 as active/in-progress** (commitment core implemented; prove/verify and later reductions still open).
  - Remaining strict CT follow-ups stay tracked in `CONSTANT_TIME_NOTES.md`.

### Status board

#### Phase 0 — Algebra

- [x] Prime field `Fp32` (u32 storage; u64 mul) implementing `FieldCore + CanonicalField` (`src/algebra/fields/fp32.rs`)
- [x] Prime field `Fp64` (u64 storage; u128 mul) implementing `FieldCore + CanonicalField` (`src/algebra/fields/fp64.rs`)
- [x] Prime field `Fp128` (u128 storage; 256-bit intermediate) implementing `FieldCore + CanonicalField` (`src/algebra/fields/fp128.rs`, `src/algebra/fields/u256.rs`)
- [x] Branchless constant-time `add_raw`, `sub_raw`, `neg` for all field types
- [x] Division-free fixed-iteration reduction for `Fp32/Fp64` multiplication paths
- [x] Rejection-sampled `FieldSampling::sample()` for all field types (no modular bias)
- [x] Pow2Offset pseudo-Mersenne registry + aliases (`q = 2^k - offset`, bounded `k <= 128`, `q % 8 == 5`) (`src/algebra/fields/pseudo_mersenne.rs`)
- [x] Constant-time review notes for current algebra/ring paths (`CONSTANT_TIME_NOTES.md`)
- [x] Deterministic parameter presets
  - [x] `q = 2^32 - 99` constants scaffold (`src/algebra/ntt/tables.rs`)
  - [x] `Pow2Offset` presets selected for 64/128-bit path:
    - `q = 2^64 - 59` (`POW2_OFFSET_MODULUS_64`)
    - `q = 2^128 - 275` (`POW2_OFFSET_MODULUS_128`)
    - source: `src/algebra/fields/pseudo_mersenne.rs`
- [x] `Module` implementations:
  - [x] `VectorModule<F, N>` (fixed-length vectors; `Module` via scalar*vector mul) (`src/algebra/module.rs`)
  - [x] `PolyModule<F, D>` removed from current scope (not needed for near-term Hachi milestones)
- [ ] Extension fields:
  - [x] `Fp2<F, NR>` quadratic extension (`src/algebra/fields/ext.rs`)
  - [x] `Fp4<F, NR, XI0, XI1>` tower extension (`src/algebra/fields/ext.rs`)
- [x] Serialization for algebra types (`HachiSerialize` / `HachiDeserialize`) (+ `u128/i128` primitives in `src/primitives/serialization.rs`)
- [x] NTT small-prime arithmetic: Montgomery-like `fpmul`, Barrett-like `fpred`, branchless `csubq`/`caddq`/`center` (`src/algebra/ntt/prime.rs`)
- [x] CRT limb arithmetic: `LimbQ`, `QData` (`src/algebra/ntt/crt.rs`)
- [x] Tests (49 total in `tests/algebra.rs`):
  - [x] field arithmetic, identities, distributivity (Fp32/Fp64/Fp128)
  - [x] zero inversion returns None
  - [x] serialization round-trips (all field types, extensions, Poly, VectorModule)
  - [x] Fp2 conjugate, norm, distributivity
  - [x] U256 wide multiply and bit access
  - [x] LimbQ round-trip, add/sub inverse, QData consistency
  - [x] NTT normalize range, fpmul commutativity
  - [x] Poly add/sub/neg
  - [x] Cyclotomic ring identities and serialization (D=4, D=64)
  - [x] NTT forward/inverse round-trips (single prime and all Q32 primes)
  - [x] Cyclotomic CRT+NTT full round-trip (`from_ring` -> `to_ring`)
  - [x] Scalar backend path equivalence (`*_with_backend` vs default path)
  - [x] Pow2Offset profile invariants (`q = 2^k - offset`, `q % 8 == 5`)
  - [x] `FieldSampling::sample()` output bound checks
  - [x] Checked deserialization rejects non-canonical field encodings
  - [x] Galois automorphism checks (`sigma` composition + multiplicativity)
  - [x] Functional gadget decompose/recompose round-trip checks
  - [x] Sparse `+/-1` challenge support checks (`hamming_weight = omega`)
- [x] Dedicated Pow2Offset primality regression tests (`tests/primality.rs`)
  - [x] Miller-Rabin probable-prime checks for all registered Pow2Offset moduli
  - [x] Composite sanity rejection checks

#### Phase 1 — Ring + gadgets (functional core)

- [x] Cyclotomic ring `Rq<F, D>` with `X^D = -1` (`src/algebra/ring/cyclotomic.rs`)
- [x] CRT+NTT-domain ring representation + CRT conversion (`src/algebra/ring/crt_ntt_repr.rs`)
- [x] Backend/domain layering for ring execution (`src/algebra/backend/*`, `src/algebra/domains/*`)
- [x] Galois automorphisms `sigma_i: X ↦ X^i` (odd `i`)
- [x] Functional gadget decomposition/recomposition (`G^{-1}` / `G` behavior) for base-`2^d` digits, without materializing dense gadget matrices
- [x] sparse short challenges (paper: `||c||_1 ≤ ω`, sparse ±1)

#### Phase 2+ — Protocol (later)

- [x] Protocol module scaffold (`src/protocol/*`) and top-level re-exports
- [x] Transcript interface (`Transcript`) plus Blake2b/Keccak implementations
- [x] Hachi-native transcript label schedule aligned to paper phases (§4.1/§4.2/§4.3/§4.5)
- [x] Commitment trait surface + streaming trait surface + contract tests
- [x] Label-directed transcript absorption for commitments (`AppendToTranscript` takes label at call site)
- [x] ring-native commitment core (`RingCommitmentScheme`, `commit.rs`, config wiring) for §4.1 setup/commit
- [x] protocol prover/verifier folder split with explicit stubs (`prover/stub.rs`, `verifier/stub.rs`)
- [x] ring-commitment tests (`ring_commitment_core`, `ring_commitment_config`, `prover_verifier_stub_contract`)
- [ ] commitment open-check prove/verify implementation (currently stubs)
- [ ] evaluation → linear relation (paper §4.2)
- [ ] ring-switching + sumcheck (paper §4.3, Fig. 4–7)
- [ ] recursion / “stop condition” + optional Greyhound composition (§4.5)

#### Phase 3 — Integration into Jolt (deferred; not active now)

- [ ] Define compatibility boundary document (what must match Jolt/Dory behavior vs what can remain Hachi-native)
- [ ] Provide Jolt-facing transcript adapter design (`Jolt` transcript pattern ↔ Hachi transcript object)
- [ ] Provide Jolt-facing PCS shim design (`CommitmentScheme`/`StreamingCommitmentScheme` mapping)
- [ ] Add transcript/commitment compatibility tests for integration-readiness (without wiring into Jolt yet)

### Conventions

- **Correctness first**: lock arithmetic with tests before touching protocol code.
- **Security first**: enforce constant-time behavior for secret-dependent operations.
- **Lean deps**: avoid heavyweight crypto crates until there is a clear need.
- **Explicit parameter sets**: each field/ring preset lives in code with a clear name and rationale.

### Module layout

```
src/algebra/
├── backend/        Backend execution traits + scalar backend
├── domains/        Domain-level aliases (coefficient / CRT+NTT)
├── fields/         Prime fields, pseudo-mersenne registry, u256, and extensions
├── ntt/            NTT kernels (butterfly), prime kernels (prime), CRT helpers (crt), presets (tables)
├── module.rs       VectorModule
├── poly.rs         Poly container
└── ring/           Cyclotomic ring and CRT+NTT representation
```

### References

- Hachi paper: `paper/hachi.pdf`
- Core traits: `src/primitives/arithmetic.rs`, `src/primitives/serialization.rs`

