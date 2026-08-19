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
- `crates/akita-prover/src/compute/`
- `crates/akita-algebra/src/ntt/`
- `crates/akita-prover/src/kernels/`
