//! Root and suffix fold verifier replay for Akita proofs.
//!
//! This module owns the shared per-fold replay engine plus path-specific prep
//! in `verify`, `root_fold`, and `suffix`. Schedule/config dispatch stays with
//! the scheme crate until the verifier-facing config boundary is extracted.

mod verify;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_transcript::labels::ABSORB_TERMINAL_E_HAT;
use akita_transcript::Transcript;
use akita_types::{SegmentTypedWitness, TerminalWitnessTranscriptParts};

mod fold;
mod root_fold;
mod suffix;
mod terminal_direct;
mod terminal_ntt;

pub use verify::batched_verify;

pub(in crate::protocol::core) type SetupPrefixOpening<E> = (Vec<E>, E);
pub(in crate::protocol::core) type FoldVerifyOutput<E> = (Vec<E>, Option<SetupPrefixOpening<E>>);

pub(in crate::protocol::core) use fold::{
    verify_fold, verify_fold_eor, FoldEorReplay, PreparedFoldPayload, PreparedFoldReplay,
    PreparedNextWitness, RelationReplayInputs, TracePreparation,
};

fn prepare_terminal_witness_replay<F, T>(
    transcript: &mut T,
    final_witness: &SegmentTypedWitness<F>,
    final_w_len: usize,
) -> Result<TerminalWitnessTranscriptParts, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if final_witness.num_elems() != final_w_len {
        return Err(AkitaError::InvalidProof);
    }
    let parts = final_witness.terminal_transcript_parts()?;
    transcript.absorb_and_record_bytes(ABSORB_TERMINAL_E_HAT, &parts.e_folded);
    Ok(parts)
}
