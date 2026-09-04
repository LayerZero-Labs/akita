//! Akita-owned execution boundary for Jolt recursion guests.

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_recursion_glue::{AkitaJoltCase, AkitaJoltInputs};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, SerializationError, Valid};
use akita_transcript::AkitaTranscript;
use akita_types::{BasisMode, FpExtEncoding};
use akita_verifier::batched_verify;
use jolt::{end_cycle_tracking, start_cycle_tracking};
use jolt_field::{CanonicalBytes, CanonicalEncoding, ExtField, Field, PseudoMersenne};

include!(concat!(env!("OUT_DIR"), "/prepared_verifier_cache.rs"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum GuestStatus {
    Success = 0,
    InputRejected = 1,
    ProofRejected = 2,
}

impl GuestStatus {
    const fn code(self) -> u32 {
        self as u32
    }

    fn for_verification(result: Result<(), AkitaError>) -> Self {
        match result {
            Ok(()) => Self::Success,
            Err(_) => Self::ProofRejected,
        }
    }
}

type ArtifactDecoder<Cfg, const D: usize> = fn(
    &[u8],
    &TrustedScheduleCatalog,
) -> Result<
    AkitaJoltInputs<<Cfg as CommitmentConfig>::Field, D, <Cfg as CommitmentConfig>::ExtField>,
    SerializationError,
>;

pub(crate) fn execute<Cfg, const D: usize>(
    input: &[u8],
    expected_case: AkitaJoltCase,
    decode: ArtifactDecoder<Cfg, D>,
) -> u32
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field
        + CanonicalEncoding
        + CanonicalBytes
        + PseudoMersenne
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + Valid,
{
    start_cycle_tracking("deserialize_input");
    let (schedules, input) = match akita_recursion_glue::split_schedule_catalog::<Cfg>(input) {
        Ok(decoded) => decoded,
        Err(_) => {
            end_cycle_tracking("deserialize_input");
            return GuestStatus::InputRejected.code();
        }
    };
    let decoded = match decode(input, &schedules) {
        Ok(decoded) if decoded.case == expected_case => decoded,
        Ok(_) | Err(_) => {
            end_cycle_tracking("deserialize_input");
            return GuestStatus::InputRejected.code();
        }
    };
    end_cycle_tracking("deserialize_input");

    if let Some(cache) = PROGRAM_BOUND_VERIFIER_CACHE {
        start_cycle_tracking("install_terminal_cache");
        let installed = decoded
            .verifier_setup
            .install_trusted_prepared_verifier_ntt_cache(
                cache,
                decoded.schedule_selection.row_digest,
            )
            .is_ok();
        end_cycle_tracking("install_terminal_cache");
        if !installed {
            return GuestStatus::InputRejected.code();
        }
    }

    start_cycle_tracking("transcript_init");
    let mut transcript =
        AkitaTranscript::<Cfg::Field>::unbound_verifier(&decoded.transcript_domain);
    end_cycle_tracking("transcript_init");

    start_cycle_tracking("akita_verify");
    let statement = match decoded.verifier_statement() {
        Ok(statement) => statement,
        Err(_) => {
            end_cycle_tracking("akita_verify");
            return GuestStatus::InputRejected.code();
        }
    };
    let result = batched_verify::<Cfg, _>(
        &decoded.proof,
        &decoded.verifier_setup,
        &schedules,
        &mut transcript,
        statement,
        BasisMode::Lagrange,
    );
    end_cycle_tracking("akita_verify");
    GuestStatus::for_verification(result).code()
}

macro_rules! declare_akita_guest {
    ($name:ident, $case:expr, $cfg:ty, $d:expr) => {
        // The limits cover the largest cataloged fp128 input without deriving
        // any allocation size from untrusted bytes. Backtraces stay disabled
        // in benchmark programs because frame pointers add measured cycles;
        // use `backtrace = "dwarf"` only for a diagnostic build.
        #[jolt::provable(
            backtrace = "off",
            stack_size = 16777216,
            heap_size = 1610612736,
            max_input_size = 805306368,
            max_output_size = 1024,
            max_trace_length = 4294967296
        )]
        fn $name(input: &[u8]) -> u32 {
            type GuestInputs = akita_recursion_glue::AkitaJoltInputs<
                <$cfg as akita_config::CommitmentConfig>::Field,
                $d,
                <$cfg as akita_config::CommitmentConfig>::ExtField,
            >;
            #[cfg(any(
                feature = "trusted-benchmark-artifact",
                akita_trusted_benchmark_artifact
            ))]
            let decode = GuestInputs::read_trusted_benchmark_artifact_bytes::<$cfg>;
            #[cfg(not(any(
                feature = "trusted-benchmark-artifact",
                akita_trusted_benchmark_artifact
            )))]
            let decode = GuestInputs::read_from_bytes::<$cfg>;
            crate::integration::execute::<$cfg, $d>(input, $case, decode)
        }
    };
}

pub(crate) use declare_akita_guest;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_preserve_guest_contract() {
        assert_eq!(GuestStatus::Success.code(), 0);
        assert_eq!(GuestStatus::InputRejected.code(), 1);
        assert_eq!(GuestStatus::ProofRejected.code(), 2);
        assert_eq!(GuestStatus::for_verification(Ok(())).code(), 0);
        assert_eq!(
            GuestStatus::for_verification(Err(AkitaError::InvalidProof)).code(),
            2
        );
    }
}
