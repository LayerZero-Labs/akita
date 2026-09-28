# Spec: Metal Compute Backend Track

| Field | Value |
|---|---|
| Author(s) | Quang Dao |
| Created | 2026-08-19 |
| Status | active |
| Supersedes | The historical CPU cutover record in `archive/2026-Q3/akita-compute-backend-metal-cutover.md` |
| Book-chapter | book/src/roadmap/compute-backends.md |

## Summary

The CPU compute-backend cutover is complete. The remaining work is the Metal
backend track. This specification records only that current work. The detailed
CPU migration history remains in the archived cutover record.

Metal is an optional prover implementation. It must not change the PCS
protocol, verifier behavior, transcript order, schedule selection, proof
serialization, or security sizing. The host and protocol layers remain the
owners of those decisions.

## Scope

The track covers:

- a `crates/akita-metal` implementation with explicit capability reporting;
- safe device, buffer, and pipeline ownership;
- typed preparation from the canonical expanded setup and selected schedule;
- one deterministic dispatch smoke test before production kernels;
- dense ring and NTT kernels followed by field, MLE, and sum-check kernels;
- deterministic CPU and Metal differential tests for each migrated operation;
- a documented Jolt opening adapter after the core backend boundary is stable.

The CPU backend remains the reference implementation. Unsupported hardware must
continue to use the CPU path without compiling or loading Metal-only code.

## Invariants

1. The backend does not sample transcript challenges or choose protocol order.
2. The backend receives typed prepared state and does not expose device storage
   through protocol-facing setup or proof types.
3. The verifier has no dependency on the Metal or prover backend crates.
4. A Metal result is keyed by an existing protocol operation, not a backend
   invented semantic identifier.
5. Unsupported devices return a typed error or use the CPU backend. They do
   not panic or silently change the schedule.
6. Every migrated operation has one backend boundary. Compatibility shims and
   parallel old and new APIs are not introduced.

## Design

### Crate and dependencies

`crates/akita-metal` is a prover-only crate. It depends on `jolt-metal` (a16z/jolt,
pinned to the same revision as `jolt-field`) for the device runtime, the error
model and the MSL field arithmetic, and adds Akita's ring and commitment kernels
on top. `scripts/check-crate-deps.sh` keeps both crates out of the verifier,
prover, CPU backend, config, planner and setup graphs, and keeps the verifier,
planner, setup and PCS crates out of `akita-metal`. The CPU backend is a
development dependency only, as the test and benchmark reference.

On targets without Metal the crate compiles against `jolt-metal`'s uninhabited
backend, and `AkitaMetal::new` returns an error of class `Unavailable`.

### Kernel principles

These follow `jolt-metal`'s specification (`specs/jolt-metal-field.md` in
a16z/jolt), adapted to Akita:

1. **One arithmetic.** Field arithmetic and accumulators come from `jolt-metal`'s
   headers; Akita's small-prime Montgomery arithmetic lives once, in
   `shaders/akita/mont.h`, and mirrors `akita_algebra::ntt::prime`. Kernels never
   redefine either.
2. **Generic kernels, spelled instances.** Kernels are MSL templates over the
   field type and shape parameters (ring degree, digit type, tile sizes).
   Moduli, primes and tables come from the Rust types and parameter sets
   (`NttPrime`, `NttTwiddles`, `jolt_metal::MslType`); no modulus or prime is
   written in MSL. `jolt-metal` instantiates templates over one type, so each
   kernel module lists its own explicit instantiations, and the library compiles
   every one at setup.
3. **Bit-exact results.** Every kernel's output equals the CPU path it replaces,
   whatever the grid and threadgroup shape. Where the CPU's intermediate words
   are part of a contract (the standalone transforms), the kernels reproduce them
   word for word; elsewhere intermediate residues only need to be congruent, and
   outputs are canonical field elements.
4. **Checked shapes, classified errors.** Shapes are validated before anything is
   encoded and rejected with `AkitaMetalError::Shape` (class `Setup`), so the
   caller can take the CPU path. The crate denies arithmetic side effects,
   indexing and panics in library code and uses `akita_error::checked`.
5. **Tuning knobs are template parameters** with a single definition in Rust and
   a default valid everywhere. Configuration is a pure function of the shape.

Balanced digit buffers cross the host/device boundary as `DeviceDigitPlanes<T>`.
The type owns both private device storage and the `log_basis` used to validate
that storage. Caller-owned slices are range-checked before upload, and device
decomposition returns the same typed state. Matvec consumes that state directly;
it cannot receive an independently supplied basis that would price CRT capacity
for a smaller range than the stored digits.

### Testing

Each kernel has a differential test against the CPU function it replaces, for
the fp128 and fp64 presets, on seeded random inputs plus edge inputs at every
range, centering and carry boundary. Each shader header's operations are also
checked one by one against the CPU (`tests/mont.rs`). Hand-made mutants of the
shader arithmetic must be killed by the tests; the kill rate and the argument
for any surviving (equivalent) mutant are recorded in the pull request.

GPU tests serialize on a file lock and abort after two minutes, since a GPU hang
can stall the machine. Test files are `#![cfg(target_os = "macos")]`; hosted CI
runs clippy and the host-side checks, and the local GPU results are recorded in
each pull request.

### Performance claims

Benchmarks compare device GPU time and end-to-end call wall time against the CPU
backend on every core (wall time) on the same inputs, after checking that all
outputs match one shared CPU oracle. Matvec reports full-profile and
planner-selected limb/profile variants on those inputs. Device matrix
preparation is reported in a separate benchmark group, so transform/upload cost
is not folded into steady-state multiplication. Each kernel pull request states
the hardware, the command, the work per element and the bounding limit, measured
with `jolt-metal`'s `limits` benchmark on the same machine.

### Kernels

| Kernel | Status | CPU reference |
|---|---|---|
| Negacyclic NTT, forward and inverse, `D` in 64..1024, any CRT profile of 30-bit primes | done | `akita_algebra::ntt::butterfly` |
| Garner CRT reconstruction into the field (fp128 over Q128, fp64 over Q64) | done | `CyclotomicCrtNtt::to_ring` |
| NTT matvec over i8 and i16 digit planes, with CRT segments past capacity | done | `mat_vec_mul_ntt_digits_i8`, `PreparedNttCache::mat_vec_i16` |
| Balanced digit decomposition into i8 and i16 planes | done | `decompose_rows_i8_into`, `CyclotomicRing::balanced_decompose_pow2_i16_into` |

The matvec's digit transforms need only residues congruent to the CPU's, so
they use Harvey's lazy radix-4 butterflies with Shoup products and
per-lane twiddles held in registers. On the dense inner commitment
(`D = 1024`, 16-bit digits) the digit transforms take about two thirds of the
time and are bound by modular-multiply throughput: on an M4 GPU a dependent
Shoup product runs at about 117 G/s (Montgomery 82 G/s, 32-bit `mulhi`
234 G/s, fp32 FMA 832 G/s), and the transforms reach about 64 G butterflies/s.
Floating-point modular arithmetic does not help: exact fp32 products need
primes below `2^12` (two and a half times the transforms), and two-product
FMA reduction for primes near `2^22` costs about as much as Shoup's with half
again as many primes. Further gains need less transform work, not cheaper
arithmetic.

Limb-split matrices provide it. `DeviceNttMatrix::from_rings` prepares the
matrix on the device from its coefficient form, splitting each centered
entry into `L` balanced limbs of `ceil(129 / L)` bits. Each limb's product
with a digit plane is small enough for a prefix of the CRT primes, the digit
transforms are shared by the limbs, and only the multiply-accumulate repeats;
the CRT step recombines `sum_l 2^(w l) t_l` in the field. `plan_matvec` picks
the cheapest `(primes, limbs)` that fits one CRT segment. On the dense nv26
inner commitment this is three primes and three limbs (96.6 to 62.9 ms on an
M4), and on the one-hot nv32 outer commitment two primes and four limbs
(4.6 to 4.0 ms).

## Acceptance criteria

- The workspace builds without Metal dependencies on unsupported targets.
- Device discovery and one deterministic dispatch have focused tests.
- Migrated kernels have CPU differential tests for supported field and ring
  profiles.
- Backend setup rejects mismatched setup metadata or schedule artifacts.
- The verifier and serialized proof remain unchanged for a CPU reference run.
- Any performance claim includes the command, target, hardware, and baseline.

## References

- `book/src/roadmap/compute-backends.md`
- `crates/akita-prover/src/backend/`
- `crates/akita-cpu-backend/src/arithmetic/`
- `crates/akita-algebra/src/ntt/`
- `crates/akita-prover/src/kernels/`
