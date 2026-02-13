## Hachi PCS implementation progress

This file is the **single source of truth** for implementation status and near-term priorities.

### Goals (project-level)

- **Production-ready implementation**: correctness, security, maintainability, and performance are first-class goals.
- **Standalone codebase**: implementation and comments should stand on their own, with no references to where ideas/code came from.
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
- **Standalone implementation policy**
  - Do not mention external inspirations/ports in code comments or public docs.
  - Keep terminology and structure internally coherent and project-native.

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

- **Phase 0 (algebra)**: prime fields (32/64/128-bit representations), modules over them, and extension fields over them.
- Later: `R_q = Z_q[X]/(X^d + 1)` cyclotomic ring arithmetic, gadget decompositions, commitments, ring-switching, sumcheck, recursive PCS.

### Status board

#### Phase 0 — Algebra

- [x] Prime field `Fp32` (u32 storage; u64 mul) implementing `Field` (`src/algebra/fields/fp32.rs`)
- [x] Prime field `Fp64` (u64 storage; u128 mul) implementing `Field` (`src/algebra/fields/fp64.rs`)
- [x] Prime field `Fp128` (u128 storage; 256-bit intermediate) implementing `Field` (`src/algebra/fields/fp128.rs`, `src/algebra/fields/u256.rs`)
- [x] Branchless constant-time `add_raw`, `sub_raw`, `neg` for all field types
- [x] Rejection-sampled `random()` for all field types (no modular bias)
- [ ] Deterministic parameter presets
  - [x] `q = 2^32 - 99` constants scaffold (`src/algebra/ntt/tables.rs`)
  - [ ] 64-bit and 128-bit example prime presets
- [ ] `Module` implementations:
  - [x] `VectorModule<F, N>` (fixed-length vectors; `Module` via scalar*vector mul) (`src/algebra/module.rs`)
  - [ ] `PolyModule<F, D>` (polynomials as a module over base field)
- [ ] Extension fields:
  - [x] `Fp2<F, NR>` quadratic extension (`src/algebra/fields/ext.rs`)
  - [x] `Fp4<F, NR, XI0, XI1>` tower extension (`src/algebra/fields/ext.rs`)
- [x] Serialization for algebra types (`HachiSerialize` / `HachiDeserialize`) (+ `u128/i128` primitives in `src/primitives/serialization.rs`)
- [x] NTT small-prime arithmetic: Montgomery-like `fpmul`, Barrett-like `fpred`, branchless `csubq`/`caddq`/`center` (`src/algebra/ntt/prime.rs`)
- [x] CRT limb arithmetic: `LimbQ`, `QData` (`src/algebra/ntt/crt.rs`)
- [x] Tests (24 total in `tests/algebra.rs`):
  - [x] field arithmetic, identities, distributivity (Fp32/Fp64/Fp128)
  - [x] zero inversion returns None
  - [x] serialization round-trips (all field types, extensions, VectorModule)
  - [x] Fp2 conjugate, norm, distributivity
  - [x] U256 wide multiply and bit access
  - [x] LimbQ round-trip, add/sub inverse, QData consistency
  - [x] NTT normalize range, fpmul commutativity
  - [x] Poly add/sub/neg

#### Phase 1 — Ring + gadgets (next)

- [ ] Cyclotomic ring `Rq<F, D>` with `X^D = -1`
- [ ] Galois automorphisms `sigma_i: X ↦ X^i` (odd `i`)
- [ ] gadget matrices `G_{b,n}` + decomposition `G^{-1}` for base-`b` digits
- [ ] sparse short challenges (paper: `||c||_1 ≤ ω`, sparse ±1)

#### Phase 2+ — Protocol (later)

- [ ] inner/outer commitments (paper §4.1)
- [ ] evaluation → linear relation (paper §4.2)
- [ ] ring-switching + sumcheck (paper §4.3, Fig. 4–7)
- [ ] recursion / “stop condition” + optional Greyhound composition (§4.5)

### Conventions

- **Correctness first**: lock arithmetic with tests before touching protocol code.
- **Security first**: enforce constant-time behavior for secret-dependent operations.
- **Lean deps**: avoid heavyweight crypto crates until there is a clear need.
- **Explicit parameter sets**: each field/ring preset lives in code with a clear name and rationale.

### Module layout

```
src/algebra/
├── fields/         Prime fields (fp32, fp64, fp128, u256) and extensions (ext)
├── ntt/            NTT small-prime kernels (prime), CRT helpers (crt), presets (tables)
├── module.rs       VectorModule
└── poly.rs         Poly container
```

### References

- Hachi paper: `paper/hachi.pdf`
- Core traits: `src/primitives/arithmetic.rs`, `src/primitives/serialization.rs`

