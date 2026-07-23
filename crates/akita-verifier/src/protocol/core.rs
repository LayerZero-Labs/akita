//! Root and suffix fold verifier replay for Akita proofs.
//!
//! This module owns the shared per-fold replay engine plus path-specific prep
//! in `verify`, `root_fold`, and `suffix`. Schedule/config dispatch stays with
//! the scheme crate until the verifier-facing config boundary is extracted.

mod verify;
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_transcript::labels::ABSORB_TERMINAL_E_HAT;
use akita_transcript::Transcript;
use akita_types::{
    prepare_opening_point, BasisMode, FpExtEncoding, LevelParamsLike, PointVariableSelection,
    PreparedOpeningPoint, SegmentTypedWitness, TerminalWitnessTranscriptParts,
};

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

/// Prepare one group's opening point from a shared protocol point.
///
/// Shared by the multi-group root and suffix per-group loops. Each caller
/// dispatches `D` (root per-group, suffix once around the loop — same `D`, same
/// result), supplies its own `basis`, and absorbs the returned padded point.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if the group's point-variable count does
/// not match the expected width. Also errors if the opening-point length
/// overflows or a selected point-variable index is out of range for
/// `source_point`.
pub(in crate::protocol::core) fn prepare_group_opening_point<F, E, const D: usize>(
    group_lp: &dyn LevelParamsLike,
    point_vars: &PointVariableSelection,
    source_point: &[E],
    basis: BasisMode,
    alpha_bits: usize,
) -> Result<PreparedOpeningPoint<F, E>, AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FpExtEncoding<F>,
{
    let target_len = alpha_bits
        .checked_add(group_lp.position_index_bits())
        .and_then(|n| n.checked_add(group_lp.block_index_bits()))
        .ok_or_else(|| {
            AkitaError::InvalidSetup("group opening point length overflow".to_string())
        })?;
    let actual_len = point_vars.num_vars();
    if actual_len != target_len {
        return Err(AkitaError::InvalidProof);
    }
    let group_point = point_vars
        .indices()
        .iter()
        .map(|&idx| {
            source_point
                .get(idx)
                .copied()
                .ok_or(AkitaError::InvalidProof)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prepared = prepare_opening_point::<F, E, D>(
        &group_point,
        basis,
        group_lp.num_positions_per_block(),
        group_lp.num_live_blocks(),
        alpha_bits,
    )?;
    Ok(prepared)
}
