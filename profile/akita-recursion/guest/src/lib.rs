//! Jolt guest program that deserializes a serialized Akita verifier input
//! bundle (from [`akita_recursion_glue::AkitaJoltInputs`]) and runs the
//! Akita batched verifier inside the Jolt RISC-V emulator.
//!
//! Cycle-tracking markers wrap the per-phase work so the host driver
//! can attribute total cycles to:
//!
//! - `deserialize_input`: blob -> typed `AkitaJoltInputs<F, D>`.
//! - `install_terminal_cache`: install the optional fp128 terminal cache.
//! - `transcript_init`:   construct the `AkitaTranscript`.
//! - `akita_verify`:      `akita_verifier::batched_verify` (the kernel
//!   that `akita-pcs::AkitaCommitmentScheme::batched_verify` wraps; we call it directly to
//!   avoid `std::time::Instant::now()`, which traps on the Jolt RISC-V
//!   emulator).
//!
//! Return code:
//!
//! - `0` — verification succeeded.
//! - `1` — decode failure.
//! - `2` — verifier rejected the proof.

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::RecursiveCommitmentConfig;
use akita_error::AkitaError;
use akita_recursion_glue::{AkitaJoltCase, AkitaJoltInputs};
use akita_transcript::AkitaTranscript;
use akita_types::BasisMode;
use akita_verifier::batched_verify;

use jolt::{end_cycle_tracking, start_cycle_tracking};

include!(concat!(env!("OUT_DIR"), "/prepared_verifier_cache.rs"));

fn verification_status(result: Result<(), AkitaError>) -> u32 {
    match result {
        Ok(()) => 0,
        Err(_) => 2,
    }
}

// Memory limits sized for the adaptive fp128 OneHot verifier. Blob size
// scales with `nv` (≈2.6 MiB at nv=20; tens-to-hundreds of MiB at nv=32).
// We give:
//   - `max_input_size` = 768 MiB so large nv blobs fit with headroom.
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
macro_rules! decode_artifact {
    (strict, $input:expr, $field:ty, $cfg:ty, $d:expr) => {
        AkitaJoltInputs::<$field, $d, <$cfg as akita_config::CommitmentConfig>::ExtField>::read_from_bytes::<$cfg>($input)
    };
    (trusted_generic, $input:expr, $field:ty, $cfg:ty, $d:expr) => {{
        #[cfg(any(
            feature = "trusted-benchmark-artifact",
            akita_trusted_benchmark_artifact
        ))]
        {
            AkitaJoltInputs::<
                $field,
                $d,
                <$cfg as akita_config::CommitmentConfig>::ExtField,
            >::read_trusted_host_artifact_bytes::<$cfg>($input)
        }
        #[cfg(not(any(
            feature = "trusted-benchmark-artifact",
            akita_trusted_benchmark_artifact
        )))]
        {
            AkitaJoltInputs::<
                $field,
                $d,
                <$cfg as akita_config::CommitmentConfig>::ExtField,
            >::read_from_bytes::<$cfg>($input)
        }
    }};
    (trusted_fp128, $input:expr, $field:ty, $cfg:ty, $d:expr) => {{
        #[cfg(any(
            feature = "trusted-benchmark-artifact",
            akita_trusted_benchmark_artifact
        ))]
        {
            AkitaJoltInputs::<$field, $d>::read_trusted_fp128_host_artifact_bytes::<$cfg>($input)
        }
        #[cfg(not(any(
            feature = "trusted-benchmark-artifact",
            akita_trusted_benchmark_artifact
        )))]
        {
            AkitaJoltInputs::<$field, $d>::read_from_bytes::<$cfg>($input)
        }
    }};
}

macro_rules! install_program_cache {
    (none, $decoded:expr) => {};
    (fp128, $decoded:expr) => {
        if let Some(cache) = PROGRAM_BOUND_VERIFIER_CACHE {
            start_cycle_tracking("install_terminal_cache");
            if $decoded
                .verifier_setup
                .install_trusted_prepared_verifier_ntt_cache(
                    cache,
                    $decoded.schedule_selection.row_digest,
                )
                .is_err()
            {
                end_cycle_tracking("install_terminal_cache");
                return 1;
            }
            end_cycle_tracking("install_terminal_cache");
        }
    };
}

macro_rules! define_akita_guest {
    ($name:ident, $case:expr, $field:ty, $cfg:ty, $d:expr, $decode:ident, $cache:ident) => {
        #[jolt::provable(
            backtrace = "off",
            stack_size = 16777216,
            heap_size = 1610612736,
            max_input_size = 805306368,
            max_output_size = 1024,
            max_trace_length = 4294967296
        )]
        fn $name(input: &[u8]) -> u32 {
            start_cycle_tracking("deserialize_input");
            let decoded = match decode_artifact!($decode, input, $field, $cfg, $d) {
                Ok(decoded) if decoded.case == $case => decoded,
                Ok(_) | Err(_) => {
                    end_cycle_tracking("deserialize_input");
                    return 1;
                }
            };
            end_cycle_tracking("deserialize_input");

            install_program_cache!($cache, decoded);

            start_cycle_tracking("transcript_init");
            let mut transcript =
                AkitaTranscript::<$field>::unbound_verifier(&decoded.transcript_domain);
            end_cycle_tracking("transcript_init");

            start_cycle_tracking("akita_verify");
            let statement = match decoded.verifier_statement() {
                Ok(statement) => statement,
                Err(_) => {
                    end_cycle_tracking("akita_verify");
                    return 1;
                }
            };
            let result = batched_verify::<$cfg, _>(
                &decoded.proof,
                &decoded.verifier_setup,
                &mut transcript,
                statement,
                BasisMode::Lagrange,
            );
            end_cycle_tracking("akita_verify");
            verification_status(result)
        }
    };
}

define_akita_guest!(
    akita_verify_fp32,
    AkitaJoltCase::OneHotFp32,
    fp32::Field,
    fp32::OneHot,
    2048,
    trusted_generic,
    none
);
define_akita_guest!(
    akita_verify_fp64,
    AkitaJoltCase::OneHotFp64,
    fp64::Field,
    fp64::OneHot,
    512,
    trusted_generic,
    none
);
define_akita_guest!(
    akita_verify_fp128_direct,
    AkitaJoltCase::OneHotFp128Direct,
    fp128::Field,
    fp128::OneHot,
    512,
    trusted_fp128,
    fp128
);
define_akita_guest!(
    akita_verify_fp128_recursive,
    AkitaJoltCase::OneHotFp128Recursive,
    fp128::Field,
    RecursiveCommitmentConfig<fp128::OneHot>,
    512,
    trusted_fp128,
    fp128
);
define_akita_guest!(
    akita_verify,
    AkitaJoltCase::OneHotFp128MultiGroupRecursive,
    fp128::Field,
    RecursiveCommitmentConfig<fp128::OneHot>,
    512,
    trusted_fp128,
    fp128
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_rejection_returns_documented_status() {
        assert_eq!(verification_status(Err(AkitaError::InvalidProof)), 2);
    }
}
