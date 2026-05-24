//! Jolt guest program that deserializes a serialized Akita verifier input
//! bundle (from [`akita_recursion_glue::AkitaJoltInputs`]) and runs the
//! Akita batched verifier inside the Jolt RISC-V emulator.
//!
//! Three cycle-tracking markers wrap the per-phase work so the host driver
//! can attribute total cycles to:
//!
//! - `deserialize_input`: blob -> typed `AkitaJoltInputs<F, D>`.
//! - `transcript_init`:   construct the `AkitaTranscript`.
//! - `akita_verify`:      timer-free `akita-config` verifier adapter.
//!
//! Return code:
//!
//! - `0` — verification succeeded.
//! - `1` — decode failure.
//! - `2` — verifier rejected the proof.

use akita_config::proof_optimized::fp128;
use akita_config::{batched_verify_with_config, CommitmentConfig};
use akita_recursion_glue::AkitaJoltInputs;
use akita_transcript::AkitaTranscript;
use akita_types::{BasisMode, CommittedOpenings, VerifierClaims};

use jolt::{end_cycle_tracking, start_cycle_tracking};

type F = fp128::Field;
const D: usize = 32;
type Cfg = fp128::D32OneHot;

const _: () = {
    // Hard-fail at compile time if the guest monomorphization drifts away from
    // the config and host artifact generator (`../artifact/src/main.rs`).
    assert!(D == <Cfg as CommitmentConfig>::D);
};

// Memory limits sized for the Akita verifier with `D=32 OneHot`. The
// verifier-input blob is ≈ 4 MiB at nv=20 but grows to ≈ 576 MiB at
// nv=32 (dominated by the expanded verifier setup matrix). We give:
//   - `max_input_size` = 768 MiB so the nv=32 blob fits with headroom.
//                        Keep this literal equal to
//                        `akita_recursion_glue::MAX_JOLT_BLOB_BYTES`.
//   - `heap_size`      = 1 GiB so the decoded verifier setup + transient
//                        verifier-internal allocations fit alongside the
//                        raw input.
//   - `stack_size`     = 16 MiB for sumcheck recursion + extension-field
//                        arithmetic frames.
//
// `backtrace = "off"` strips DWARF symbols + `.eh_frame` and skips
// `-Cforce-frame-pointers=yes`. Removes ~3-8 % of cycles in the verifier
// path (no frame-pointer save/restore around every Rust function call).
// Re-enable `backtrace = "dwarf"` temporarily to symbolicate a guest
// panic; the `host` driver already plumbs `JOLT_BACKTRACE=full`.
#[jolt::provable(
    backtrace = "off",
    stack_size = 16777216,
    heap_size = 1610612736,
    max_input_size = 805306368,
    max_output_size = 1024,
    max_trace_length = 4294967296
)]
fn akita_verify(input: &[u8]) -> u32 {
    // `&[u8]` (rather than `Vec<u8>`) so the postcard-decoded input is a
    // zero-copy borrow into the guest's input region — no heap
    // allocation, no megabyte-scale memcpy on entry. The Jolt macro
    // emits `postcard::take_from_bytes::<&[u8]>(input_slice)`, which
    // postcard implements as a borrowed `Bytes` slice.
    start_cycle_tracking("deserialize_input");
    #[cfg(any(
        feature = "trusted-benchmark-artifact",
        akita_trusted_benchmark_artifact
    ))]
    let decoded_result = AkitaJoltInputs::<F, D>::read_trusted_host_artifact_bytes(input);
    #[cfg(not(any(
        feature = "trusted-benchmark-artifact",
        akita_trusted_benchmark_artifact
    )))]
    let decoded_result = AkitaJoltInputs::<F, D>::read_from_bytes(input);

    let decoded = match decoded_result {
        Ok(decoded) => decoded,
        Err(_) => {
            end_cycle_tracking("deserialize_input");
            return 1;
        }
    };
    end_cycle_tracking("deserialize_input");

    start_cycle_tracking("transcript_init");
    let mut transcript = AkitaTranscript::<F>::unbound_verifier(&decoded.transcript_domain);
    end_cycle_tracking("transcript_init");

    let openings = [decoded.opening];
    let opening_groups = [&openings[..]];

    let claims: VerifierClaims<F, _> = vec![(
        &decoded.opening_point[..],
        CommittedOpenings {
            openings: opening_groups[0],
            commitment: &decoded.commitment,
        },
    )];

    start_cycle_tracking("akita_verify");
    let result = batched_verify_with_config::<F, _, D, Cfg>(
        &decoded.proof,
        &decoded.verifier_setup,
        &mut transcript,
        claims,
        BasisMode::Lagrange,
    );
    end_cycle_tracking("akita_verify");

    match result {
        Ok(()) => 0,
        Err(_) => 2,
    }
}
