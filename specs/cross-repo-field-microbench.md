# Spec: Cross-Repo Field Arithmetic Microbenchmarks (Akita vs Plonky3)

| Field       | Value                                             |
|-------------|---------------------------------------------------|
| Author(s)   | Quang Dao                                         |
| Created     | 2026-06-01                                        |
| Status      | proposed                                          |
| PR          | https://github.com/LayerZero-Labs/akita/pull/142  |

## Summary

This spec defines a reusable cross-repo field-arithmetic microbenchmark that compares Akita's small-field arithmetic against Plonky3's, to fill a microbench section in an upcoming Akita performance paper.
The comparison fixes a common base-field bit width (31-bit) and a common extension-field shape, then measures scalar and SIMD-packed arithmetic for the operations that dominate prover inner loops.
It is deliberately not a strict apples-to-apples modulus comparison: the two systems use different prime moduli and different internal representations (Akita uses canonical Solinas / pseudo-Mersenne primes, Plonky3 uses Montgomery 31-bit primes), so the comparison is "same bit width, same extension degree, best available SIMD per architecture".

The central framing is security-equivalent extension degree, not equal extension degree.
Plonky3 is hash-based: achieving true 128-bit security over a 31-bit base field in practice drives the working extension up to degree 5.
The sharp reason is that the soundness error of the random-evaluation arguments has the form `O(instance_size / |ext_field|)`, so the bit-security is roughly `log2(|ext_field|) - log2(instance_size)`.
For a 31-bit base, a degree-4 extension is about `2^124` and a degree-5 extension is about `2^155`, so a large instance of size `2^22` leaves only about `124 - 22 = 102` bits at degree 4 (well under 128), while degree 5 gives about `155 - 22 = 133` bits.
This field-size term comes from sampling challenges in the extension field (Schwartz-Zippel / random-linear-combination / DEEP-style quotienting and FRI / Reed-Solomon proximity gaps, BCIKS20), so it shrinks only by enlarging the field; it is not the FRI query / proof-of-work component, so grinding cannot cheaply buy the lost bits back.
Akita is lattice-based: it reaches the same 128-bit target at degree 4, because its random-challenge soundness comes from sumcheck, whose error is bounded by `O(degree * num_rounds / |ext_field|)` with `num_rounds = log2(instance_size)` (LFKN92; Thaler, *Proofs, Arguments, and Zero-Knowledge*, Prop. 2.9), so the field-size loss scales with `log2(N) ~ 22`, not with `N`; combined with extraction soundness from SIS / lattice hardness, the degree-4 extension does not have to absorb an instance-size-scaled random-evaluation or query-soundness term the way Plonky3's degree does.
The fair security-equivalent comparison is therefore Akita degree-4 against Plonky3 degree-5, with Plonky3 degree-4 also measured as an additional (commonly cited, but security-insufficient for the 31-bit base) reference point.

Data must cover architecture-specific SIMD: NEON on an aarch64 workstation (Apple-silicon class), and both AVX2 and AVX-512 on an x86_64 server (recent AMD/Intel class with native AVX-512).
Scalar paths are also captured where Plonky3 exposes them, but the headline numbers presented in the paper are the best available SIMD per architecture.

The study spans two phases.
Phase 1 is measurement: the cross-repo 31-bit comparison above, plus Akita-only 128-bit field arithmetic as an additional point of comparison (Plonky3 has no 128-bit field, so this contextualizes the cost of Akita's large-field path against the small-field-plus-extension path).
Phase 2 is investigation: once all numbers are in hand, study further efficiency improvements to the Akita field/kernel code, informed by the Plonky3 kernels where their techniques transfer to Akita's canonical Solinas representation.

## Intent

### Goal

Produce a single reusable Criterion-based microbenchmark, hosted in the existing `akita-pcs` `field_arith` suite, that emits directly comparable Akita and Plonky3 rows for 31-bit base fields and their degree-4 and degree-5 extensions, plus Akita-only 128-bit base-field rows, runnable on aarch64/NEON, x86/AVX2, and x86/AVX-512, and captured into a machine-readable table for the paper; then, from that data, identify and benchmark-gate concrete efficiency improvements to Akita's field arithmetic, drawing on Plonky3 kernel techniques where they apply.

### Security-Equivalence Framing

The motivation for benchmarking both Plonky3 degree-4 and degree-5 is the security model, and the bench must be presented with this framing:

| Setting | 128-bit-secure working field over a 31-bit base | Why |
|---------|-------------------------------------------------|-----|
| Plonky3 (hash-based) | base + degree-5 extension | random-evaluation soundness error is `O(instance_size / |ext_field|)`, i.e. about `31*d - log2(N)` bits; at `N = 2^22`, degree-4 (`~2^124`) gives only `~102` bits, degree-5 (`~2^155`) gives `~133` bits. The field-size term is the RS proximity-gap component (BCIKS20); the lost bits scale with instance size and are not the FRI query/PoW term, so grinding cannot cheaply recover them |
| Akita (lattice-based) | base + degree-4 extension | extraction soundness from SIS/lattice hardness; the only field-size-dependent term is the sumcheck error `O(degree * log2(N) / |ext_field|)` (LFKN92; Thaler, Prop. 2.9), which scales with the number of rounds `log2(N)`, not with `N`, so there is no `O(N / |ext_field|)` term and the extension field does not need the extra degree |

The precise soundness bit-accounting (the constant in `O(instance_size / |ext_field|)`, per-component losses, FRI query count, proximity gaps) belongs to the paper's security section, not to this bench spec.
Akita's own challenge-grinding policy (proof-of-work on Fiat-Shamir nonces) would tune the query/PoW component, not the field-size term, and is specified separately in PR 102 (`Add transcript grinding specification`, spec-only, not yet implemented) with its merged prerequisite PR 104 (`Harden transcript replay, setup identity, and recursion inputs`); it does not change the degree choice benchmarked here.
This spec records only the resulting extension-degree choice and benchmarks exactly those fields, so the paper can state "at equal 128-bit security, Akita's degree-4 arithmetic costs X while Plonky3's degree-5 arithmetic costs Y on the same hardware and SIMD width".

### Field Matrix

Base field, 31-bit:

| Bench label | Library | Type | Modulus / representation |
|-------------|---------|------|--------------------------|
| `mersenne31` | Akita | `Fp32<{2^31-1}>` | `2^31 - 1`, Solinas |
| `prime31_offset19` | Akita | `Prime31Offset19` | `2^31 - 19`, Solinas |
| `p3_mersenne31` | Plonky3 | `Mersenne31` | `2^31 - 1` (exact-modulus anchor vs Akita `mersenne31`) |
| `p3_baby_bear` | Plonky3 | `BabyBear` | `2^31 - 2^27 + 1`, Montgomery |
| `p3_koala_bear` | Plonky3 | `KoalaBear` | `2^31 - 2^24 + 1`, Montgomery |

Degree-4 extension (Akita security-equivalent, Plonky3 reference):

| Bench label | Library | Type |
|-------------|---------|------|
| `mersenne31_{tower,power,ring_subfield}_fp4` | Akita | `TowerBasisFp4` / `PowerBasisFp4` / `RingSubfieldFp4` over `Fp32<{2^31-1}>` |
| `prime31_offset19_{tower,power,ring_subfield}_fp4` | Akita | same three bases over `Prime31Offset19` (already present) |
| `p3_baby_bear_ext4` | Plonky3 | `BinomialExtensionField<BabyBear, 4>` |
| `p3_koala_bear_ext4` | Plonky3 | `BinomialExtensionField<KoalaBear, 4>` |

Degree-5 extension (Plonky3 security-equivalent over a 31-bit base; no Akita analog):

| Bench label | Library | Type |
|-------------|---------|------|
| `p3_baby_bear_ext5` | Plonky3 | `BinomialExtensionField<BabyBear, 5>` |
| `p3_koala_bear_ext5` | Plonky3 | `QuinticTrinomialExtensionField<KoalaBear>` |

Mersenne31's Plonky3 extension is complex-based rather than a plain binomial, so `p3_mersenne31_ext4` / `ext5` are included only if the Plonky3 0.5.3 API exposes a clean degree-4/degree-5 extension over Mersenne31; otherwise Mersenne31 serves as the exact-modulus base-field anchor only, and BabyBear/KoalaBear carry the extension comparison.

128-bit base field (Akita only, no Plonky3 counterpart):

| Bench label | Library | Type | Modulus / representation |
|-------------|---------|------|--------------------------|
| `prime128_offset275` | Akita | `Prime128Offset275` | `2^128 - 275`, Solinas |

This row already exists in `base.rs` (`F128`) and is complemented by the `field_arith/kernel/fp128_accumulator` pattern in `kernel.rs`; Phase 1 promotes it to an explicit, captured comparison point and ensures it is measured on all three SIMD targets.
It is presented as a contrast, not a head-to-head: it shows the per-op cost of operating directly in a 128-bit field versus the 31-bit-base-plus-extension approach that both Akita (degree-4) and Plonky3 (degree-5) use, so the paper can argue about where each representation pays.

### Operations

Each (library, field, arch, SIMD) cell measures the same operation set, matching the existing `field_arith` core:

- Scalar latency chains: `add`, `sub`, `neg`, `double`, `add_neg`, `double_add`, `mul`, `mul_add`, `square`, `mul_self`, `inverse`.
- Packed (SIMD) latency chains: the same set, normalized to ns/lane.
- Scalar and packed throughput streams: `add`, `sub`, `mul`, `square`, `inverse`.

### Invariants

- Identical workload across libraries: same operation set, same iteration counts (driven by the shared `ArithmeticBenchParams` and `AKITA_BENCH_*` env knobs), same RNG-seeded random inputs, same Criterion group/ID naming, so Akita and Plonky3 rows land in the same Criterion groups and diff directly.
- Per-lane normalization: packed rows report ns/lane (existing `field_arith` convention via `PF::WIDTH`), so different SIMD widths between libraries and architectures remain comparable.
- SIMD width is selected once per build via target features and applies to both libraries simultaneously (both Akita and Plonky3 gate their packed backends on the same `target_feature`s), so within one build cell the comparison uses the same vector width.
- No change to Akita field arithmetic, representations, proof formats, transcript bytes, schedules, or public APIs. This is additive bench-only code plus one new spec.
- Correctness of the benchmarked Akita types is already covered by `crates/akita-field` scalar/packed parity tests; correctness of Plonky3 types is upstream. The bench asserts nothing about cryptographic soundness; the security framing is documentation.
- Verifier no-panic contract is untouched by Phase 1: no verifier-reachable code changes.
- Phase 2 constraint: any Akita field/kernel change must preserve canonical field representation (values stay in `[0, P)`), keep packed-vs-scalar lane parity, leave proof formats / transcript bytes / schedules / public APIs unchanged, and uphold the verifier no-panic contract. Every Phase 2 change must be gated by a measurable win on the Phase 1 benches (latency and/or throughput, per arch), mirroring the benchmark-gated discipline in `specs/fp31-field-optimization-retrospective.md`. Phase 2 borrows Plonky3 techniques only where they translate cleanly to Akita's Solinas representation; it does not adopt Montgomery representation (consistent with the prior fp31 decision).

### Non-Goals

- No Plonky2. It has no 31- or 32-bit field; its only relevance would be a 64-bit Goldilocks anchor, which is out of scope for this 31-bit study.
- No end-to-end proof-time, proof-size, or prover-throughput comparison. This is field arithmetic only.
- Phase 2 does not commit to landing any specific optimization. It is scoped to investigate and benchmark candidate improvements; each candidate ships only if it is a benchmark-gated win under the Phase 2 invariants. A candidate that does not clear the bar is recorded as rejected (with the measured reason), not merged.
- No 32-bit cross-repo row: Plonky3 has no 32-bit prime, so Akita's `prime32_offset99` has no counterpart and is not part of the cross-repo matrix (it remains in the Akita-only `field_arith` rows).
- No runtime CPU feature dispatch and no CI regression gate. These are paper-data runs, executed manually per architecture.
- No new sub-workspace. Part of the Plonky3 0.5.3 graph is already in the workspace `Cargo.lock` (transitively via `spongefish`: `p3-field`, `p3-koala-bear`, `p3-monty-31`, plus the poseidon/challenger/dft/matrix/util crates); `p3-mersenne-31` and `p3-baby-bear` are *not* yet in the lock and are added here at the same 0.5.3 version and source, so no new advisory surface is expected. Combined with `field_arith/comparison.rs` already carrying a foreign field dev-dep (`ark-bn254`), the bench lives in-crate.

## Evaluation

### Acceptance Criteria

- [ ] `cargo bench -p akita-pcs --bench field_arith --no-run` builds on aarch64 (local) with the new Plonky3 rows.
- [ ] The same builds clean on the x86_64 server under both the AVX2 and AVX-512 target-feature configurations.
- [ ] `cargo clippy -p akita-pcs --benches -- -D warnings` and `cargo fmt --check` are clean.
- [ ] `cargo deny check` passes with the added direct Plonky3 dev-deps. Two crates (`p3-mersenne-31`, `p3-baby-bear`) are new lockfile entries at 0.5.3; the rest of the 0.5.3 graph is already present transitively, so no new advisories/licenses are expected.
- [ ] Akita `mersenne31` degree-4 rows (`tower`/`power`/`ring_subfield`) are added to `ext4.rs` so the exact-modulus base anchor has an extension counterpart (closes the gap noted in `specs/avx-simd-port.md`).
- [ ] Full data captured on all three SIMD cells: NEON (aarch64 workstation), AVX2 (x86_64 server), AVX-512 (x86_64 server), saved as named Criterion baselines.
- [ ] The Akita 128-bit base-field row (`prime128_offset275`) is captured on all three SIMD targets alongside the 31-bit rows.
- [ ] A machine-readable summary (CSV + a generated markdown table) pivots median ns/op (scalar) and ns/lane (packed) by (library, field, extension degree, operation, arch, SIMD), suitable for direct inclusion in the paper.
- [ ] Phase 2: a written investigation note enumerates candidate Akita field/kernel improvements informed by Plonky3 kernels, each with a before/after measurement on the Phase 1 benches and a keep/reject decision under the Phase 2 invariants. Landed candidates (if any) ship as separate benchmark-gated commits; rejected candidates are recorded with their measured reason.

### Testing Strategy

- The benchmark itself is the artifact; correctness of benchmarked arithmetic is covered by existing `crates/akita-field` parity tests and upstream Plonky3 tests.
- `--no-run` build checks gate compilation on every target before measurement runs.
- Determinism: fixed RNG seeds per case (mirroring the existing `field_arith` seed convention) so re-runs are stable modulo machine noise.
- Measurement hygiene: pin a single core and a fixed CPU governor on the x86_64 server, run the aarch64 workstation on stable power with background load minimized, and take Criterion medians with adequate warmup.

### Performance

This spec produces data; it does not gate on a regression threshold.
Expected qualitative result to verify against (not assert): on a 31-bit base, Akita's degree-4 packed multiply (ns/lane) should be compared against Plonky3's degree-5 packed multiply at equal security, and separately against Plonky3's degree-4 as a reference.
Base-field `mul` ns/lane for Akita `mersenne31` vs Plonky3 `p3_mersenne31` is the cleanest single signal (identical modulus), and should be within the same order of magnitude on each SIMD target; any large gap is itself a paper-worthy finding to explain.

## Design

### Architecture

All changes are additive and confined to the `akita-pcs` bench suite plus a data-collection script.

Files:

| File | Change |
|------|--------|
| `crates/akita-pcs/Cargo.toml` | add dev-deps `p3-field`, `p3-mersenne-31`, `p3-baby-bear`, `p3-koala-bear` at `= "0.5.3"`; `p3-field`/`p3-koala-bear` already resolve via the lock, `p3-mersenne-31`/`p3-baby-bear` are new same-version entries |
| `crates/akita-pcs/benches/field_arith/plonky3.rs` | new module: a Plonky3-generic bench core mirroring `arithmetic.rs`, emitting identical Criterion group/ID strings, plus `bench_p3_base_matrix`, `bench_p3_ext4_matrix`, `bench_p3_ext5_matrix` |
| `crates/akita-pcs/benches/field_arith/mod.rs` | `pub(crate) mod plonky3;` and re-export the three new entry points |
| `crates/akita-pcs/benches/field_arith.rs` | add the three new functions to `criterion_group!` |
| `crates/akita-pcs/benches/field_arith/ext4.rs` | add Akita `mersenne31_*_fp4` rows (Akita degree-4 only) |
| `crates/akita-pcs/benches/field_arith/plonky3.rs` (cont.) | the Plonky3-only `ext5` family lives here as `bench_p3_ext5_matrix` (listed above); there is no separate `ext5.rs` because Akita has no degree-5 analog |
| `scripts/field_microbench_collect.py` | parse Criterion `estimates.json` across saved baselines into the paper CSV + markdown table |
| `bench-data/field-microbench.{csv,md}` | new committed output artifacts: machine-readable CSV plus the paper-ready markdown pivot, emitted by the collect script |

Reusability: the existing `bench_arithmetic_case` core is bound to Akita's `FieldCore` / `PackedField` traits and cannot accept Plonky3 types directly (orphan rule plus distinct traits).
The new `plonky3.rs` therefore carries two parallel generic cores that run the same operation set and write to the same Criterion group/ID format strings (`field_arith/{family}/latency_chain/{label}_w{WIDTH}` and `.../throughput_stream/...`):
- `bench_p3_base_case<F: Field>` uses `F::Packing` (`PackedField`, real SIMD width).
- `bench_p3_ext_case<Base, EF: ExtensionField<Base>>` uses `<EF as ExtensionField<Base>>::ExtensionPacking` (`PackedFieldExtension`, base-lane SIMD width). For extensions, `<EF as Field>::Packing = Self` (WIDTH 1) and must not be used for packed rows.
It reuses `ArithmeticBenchParams`, `data.rs`, and every `AKITA_BENCH_*` env knob, so the only duplication is the trait-bound loop bodies.
The composite chains (`mul_self`, `add_neg`, `double_add`) are written as the identical arithmetic expressions on both sides, not as a same-named library method, so the per-op workload (operation kind and count) is the same across Akita and Plonky3.
A later consolidation behind a single `trait FieldBenchOps` (with newtype wrappers for the Plonky3 types) is possible but out of scope; the parallel core is the minimal low-risk first cut.

Plonky3 0.5.3 API points pinned for implementation:

- ring/field traits: `PrimeCharacteristicRing` for `ZERO`/`ONE`/`square`/`double`, `Field` for inversion (`try_inverse`/`inverse`).
- random sampling: `p3-field` depends on `rand 0.10`, which does not interoperate with the bench harness `rand 0.8` `StdRng`. Sample base elements via `F::from_u64(rng.next_u64())` and extension elements via `EF::from_basis_coefficients_fn(|_| Base::from_u64(rng.next_u64()))`.
- base packed type: `F::Packing` with `PackedField::from_fn` / `WIDTH`.
- extension packed type: `<EF as ExtensionField<Base>>::ExtensionPacking` with `PackedFieldExtension::from_ext_slice`; packed `inverse` is omitted (no `PackedField::inverse`); scalar `inverse` uses `Field::inverse`.
- Mersenne31 extension constructor (complex-based): confirm whether degree-4/degree-5 extensions over `Mersenne31` are exposed cleanly; if not, restrict extension rows to BabyBear/KoalaBear.

### SIMD Build Configurations

A single `RUSTFLAGS` setting per build selects the packed backend for both libraries.

| Arch | Machine | SIMD | Build flags |
|------|---------|------|-------------|
| aarch64 | aarch64 workstation | NEON | `RUSTFLAGS="-Ctarget-cpu=native"` (NEON is baseline on aarch64) |
| x86_64 | x86_64 server | AVX2 (no AVX-512) | `RUSTFLAGS="-Ctarget-cpu=x86-64-v3"` |
| x86_64 | x86_64 server | AVX-512 | `RUSTFLAGS="-Ctarget-cpu=native"` (host with avx512f/dq/bw/vl) or `-Ctarget-cpu=x86-64-v4` |

This matches the flag convention already used in `specs/avx-simd-port.md` (`x86-64-v3` for AVX2, `native` for AVX-512).
Both Akita's `packed_{neon,avx2,avx512}` backends and Plonky3's per-arch packed backends key off the same target features, so each cell compares equal vector widths.

### Data Collection and Artifact Format

- Per machine/SIMD, save a named Criterion baseline: `--save-baseline neon`, `avx2`, `avx512`.
- Aggregate with `scripts/field_microbench_collect.py`, reading `target/criterion/**/<baseline>/estimates.json`, producing:
  - `bench-data/field-microbench.csv` with columns `library, field, ext_degree, basis, op, kind(scalar|packed), arch, simd, width, ns_per_op_or_lane, lower, upper`.
  - `bench-data/field-microbench.md`: the paper-ready pivot table (security-equivalent rows highlighted: Akita d4 vs Plonky3 d5).
- `critcmp` may be used for quick interactive diffs, but the committed artifact is the CSV + generated markdown so the paper has a stable source.

### Phase 2: Kernel-Level Efficiency Investigation (Akita)

After Phase 1 data is captured on all three SIMD targets, use the per-op, per-arch numbers to locate Akita's weakest cells relative to Plonky3 (largest ns/lane gaps on the security-relevant ops: packed `mul`, `mul_add`, `square`, and the degree-4 extension multiply), then study the corresponding Plonky3 kernels for transferable techniques.

Candidate areas to examine (the data decides which are worth pursuing; this list is a starting map, not a commitment):

- Packed base-field multiply and reduction on each backend (`crates/akita-field/src/fields/packed_{neon,avx2,avx512}.rs`) versus Plonky3's `monty-31` / `mersenne-31` packed multiply and reduction.
- Extension-field multiply kernels (`crates/akita-field/src/fields/packed_ext.rs`, `ext/{tower,power,ring_subfield}_fp4.rs`) versus Plonky3's `BinomialExtensionField` multiply, especially Karatsuba-style schedules and interleaving of base multiplies with reduction.
- AVX-512 opportunities already flagged but not taken in `specs/avx-simd-port.md` (for example IFMA52 for narrow primes, packed-type alignment), re-evaluated against the fresh AVX-512 numbers from the x86_64 server.
- Inversion batching patterns if inversion shows up as a hot cell.

Each candidate follows the same loop: measure baseline on the Phase 1 benches, implement under the Phase 2 invariants (canonical Solinas representation, parity, no format/transcript/API change, no Montgomery cutover), re-measure on the affected arch(es), and keep only on a clear win. Findings (kept and rejected, with numbers) are written up so the paper and a follow-up retrospective spec can cite them; this mirrors the structure of `specs/fp31-field-optimization-retrospective.md`.

### Alternatives Considered

- New excluded sub-workspace (`bench/field-cross-repo/`), mirroring `profile/akita-recursion`. Rejected for this scope: Plonky3 0.5.3 is already in the main lock, `field_arith` already hosts a foreign field dev-dep, and an in-crate module reuses the existing harness and naming for free. The sub-workspace would only be justified to isolate a genuinely new heavy graph (e.g. Plonky2), which is out of scope.
- Strict equal-modulus comparison only (Mersenne31 vs Mersenne31). Kept as the anchor row, but rejected as the whole study: the paper needs the production Plonky3 fields (BabyBear, KoalaBear) and the security-equivalent degree mapping, which equal-modulus alone cannot express.
- Equal extension degree (d4 vs d4) as the headline. Rejected as the headline because it understates Plonky3's true 128-bit cost over a 31-bit base; d4-vs-d4 is retained only as a reference row, with d4(Akita)-vs-d5(Plonky3) as the security-equivalent headline.
- Including Plonky2. Rejected: no 31/32-bit field.

## Documentation

- This spec is the design artifact.
- The generated `bench-data/field-microbench.md` table feeds the paper's microbench section directly.
- A short "Running the cross-repo field microbench" subsection should be added to the bench docs (or `docs/`), listing the three SIMD commands and the x86_64-server setup prerequisites, so the runs are reproducible for paper revisions.

## Execution

### Machine Prerequisites

- aarch64 workstation: Rust 1.95 toolchain; NEON is baseline.
- x86_64 server: ensure a Rust 1.95 toolchain is installed (`rustup toolchain install 1.95.0`) so the repo's `rust-version = 1.95` resolves, and check out this branch there. Ensure the cargo bin directory is on PATH for non-interactive shells (use absolute paths or source the environment in the run script).

### Task Checklist

1. Add the four Plonky3 dev-deps to `crates/akita-pcs/Cargo.toml`; run `cargo deny check`.
2. Implement `plonky3.rs` (generic core + base/ext4/ext5 matrices), pinning the 0.5.3 API points above.
3. Add Akita `mersenne31_*_fp4` rows to `ext4.rs`; wire the Plonky3-only `ext5` family inside `plonky3.rs` (`bench_p3_ext5_matrix`).
4. Wire `mod.rs` and the `criterion_group!` in `field_arith.rs`.
5. `--no-run` build check on aarch64; fix lints/format.
6. Push branch; set up the x86_64 server (1.95 toolchain + checkout); `--no-run` build check there under both AVX2 and AVX-512 flags.
7. Run measurement cells and save baselines:

```bash
# aarch64 workstation / NEON
RUSTFLAGS="-Ctarget-cpu=native" \
  cargo bench -p akita-pcs --bench field_arith --release -- --save-baseline neon \
  'field_arith/(base|ext4|ext5)/'

# x86_64 server, AVX2
RUSTFLAGS="-Ctarget-cpu=x86-64-v3" \
  cargo bench -p akita-pcs --bench field_arith --release -- --save-baseline avx2 \
  'field_arith/(base|ext4|ext5)/'

# x86_64 server, AVX-512
RUSTFLAGS="-Ctarget-cpu=native" \
  cargo bench -p akita-pcs --bench field_arith --release -- --save-baseline avx512 \
  'field_arith/(base|ext4|ext5)/'
```

8. Targeted sanity diffs (exact-modulus anchor and security-equivalent pair):

```bash
# exact-modulus base anchor
cargo bench -p akita-pcs --bench field_arith -- 'field_arith/base/.*(mersenne31|p3_mersenne31)'
# security-equivalent extension headline: Akita d4 vs Plonky3 d5
cargo bench -p akita-pcs --bench field_arith -- 'field_arith/(ext4|ext5)/'
```

9. Run `scripts/field_microbench_collect.py` against the three baselines; commit `bench-data/field-microbench.{csv,md}`. The `field_arith/base/` filter already includes the Akita `prime128_offset275` row, so the 128-bit numbers are captured by the same three runs; confirm they appear in the summary.

Phase 2 (after all Phase 1 data is committed):

10. From the summary, rank Akita's largest ns/lane gaps versus Plonky3 on packed `mul`, `mul_add`, `square`, and degree-4 extension multiply, per arch.
11. For each candidate (see the Phase 2 design map), read the corresponding Plonky3 kernel, implement under the Phase 2 invariants, re-measure on the affected arch(es), and keep only on a clear benchmark win.
12. Write up kept and rejected candidates with numbers (a follow-up retrospective spec in the style of `specs/fp31-field-optimization-retrospective.md`); land wins as separate benchmark-gated commits.

### Risks To Resolve First

- Plonky3 0.5.3 trait/method names for ring ops, inversion, random sampling, and the `Packing` associated type. Resolve by reading the pinned `p3-field` 0.5.3 source before writing the loop bodies.
- Mersenne31 extension availability at degree 4/5 (complex-based). Decide anchor-only vs full extension early.
- Plonky3 packed backends must be compile-time `target_feature` / `target-cpu` gated with no runtime CPU dispatch, or the "equal vector width per cell" invariant breaks. Plonky3 0.5.3 follows the same `target-cpu=native` convention as Akita (per the upstream README), but confirm there is no runtime feature detection in the `Packing` path before trusting per-cell width parity.
- x86_64-server toolchain/PATH and CPU pinning for stable numbers.

## References

- Existing Akita field microbench suite: `crates/akita-pcs/benches/field_arith/` (`arithmetic.rs`, `base.rs`, `ext2.rs`, `ext4.rs`, `params.rs`, `comparison.rs`).
- `specs/fp31-field-optimization-retrospective.md`: Akita 31-bit optimization, Plonky3 Monty31/Mersenne31 references, recorded packed-mul numbers.
- `specs/avx-simd-port.md`: AVX2/AVX-512/NEON packed backends, target-cpu flag convention, and the noted Mersenne31 ext4 bench gap this spec closes.
- `specs/general-field-support.md`, `specs/extension-claim-incidence-cutover.md`: Akita extension-field representations (`Fp2`, tower/power/ring-subfield `Fp4`).
- Plonky3 0.5.3: `p3-field`, `p3-mersenne-31`, `p3-baby-bear`, `p3-koala-bear`. `p3-field`, `p3-koala-bear`, `p3-monty-31` (plus poseidon/challenger/dft/matrix/util) are already in `Cargo.lock` via `spongefish`; `p3-mersenne-31` and `p3-baby-bear` are added at the same 0.5.3 version.
- Sumcheck soundness `<= v*d / |F|` over `v` rounds: Lund-Fortnow-Karloff-Nisan, "Algebraic Methods for Interactive Proof Systems" (LFKN, 1992); Thaler, *Proofs, Arguments, and Zero-Knowledge*, Prop. 2.9.
- FRI / Reed-Solomon proximity-gap soundness and its field-size dependence (distinct from the query/PoW term): Ben-Sasson, Carmon, Ishai, Kopparty, Saraf, "Proximity Gaps for Reed-Solomon Codes" (BCIKS, 2020), ePrint 2020/654.
- Plonky3 fields and ~128-bit extensions (Mersenne31 complex extension; BabyBear/KoalaBear quartic and quintic; soundness depends on field size, query count, and extension degree): Plonky3 README; Polygon/Plonky3 audit (Least Authority, 2024).
- Akita transcript grinding: PR 102 (`Add transcript grinding specification`, spec-only, not yet implemented); merged prerequisite PR 104 (`Harden transcript replay, setup identity, and recursion inputs`).
- Test fleet: an aarch64 workstation (Apple-silicon class, NEON) and an x86_64 server (recent AMD/Intel class with native AVX-512, exercised at both AVX2 and AVX-512).
