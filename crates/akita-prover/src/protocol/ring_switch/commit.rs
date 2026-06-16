use super::*;
use crate::commit::CommitBackend;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::AdditiveGroup;

pub use crate::commit::commit_w;

/// Result of committing the next logical recursive witness.
pub struct NextWitnessCommitment<F: FieldCore> {
    /// Physical witness representation when extension packing changes the logical witness.
    pub witness: Option<RecursiveWitnessFlat>,
    /// Commitment to the physical next-level witness.
    pub commitment: FlatRingVec<F>,
    /// Prover hint for `commitment`.
    pub hint: RecursiveCommitmentHintCache<F>,
}

/// Dispatch a recursive `w` commitment to the selected ring dimension under
/// config `Cfg`.
///
/// The prover crate owns typed backend preparation and `commit_w` execution;
/// the recursive layout is derived from `Cfg`.
///
/// # Errors
///
/// Returns an error if layout selection, backend preparation, commitment, or
/// D-erased hint conversion fails.
#[inline(never)]
fn dispatch_commit_w_with_layout_policy<Cfg, B>(
    backend: &B,
    commit_params: LevelParams,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessCommitment<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + HasWide,
    <Cfg::Field as HasWide>::Wide: AdditiveGroup + From<Cfg::Field> + ReduceTo<Cfg::Field>,
    B: CommitBackend<Cfg::Field>,
{
    let commit_d = commit_params.ring_dimension;
    dispatch_ring_dim_result!(commit_d, |D_COMMIT| {
        let prepared_commit = backend.prepare_expanded::<D_COMMIT>(expanded.clone())?;
        if <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE == 1 {
            let (wc, wh) = commit_w::<Cfg::Field, B, { D_COMMIT }>(
                logical_w,
                expanded.as_ref(),
                backend,
                &prepared_commit,
                &commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: None,
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        } else {
            // The tensor pack is length-preserving (it redistributes the same
            // digit count), so the committed witness fits the schedule's
            // recursive commit params directly — no per-length re-derivation.
            let committed_w =
                tensor_pack_recursive_witness::<Cfg::Field, Cfg::ExtField, { D_COMMIT }>(
                    logical_w,
                )?;
            let (wc, wh) = commit_w::<Cfg::Field, B, { D_COMMIT }>(
                &committed_w,
                expanded.as_ref(),
                backend,
                &prepared_commit,
                &commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: Some(committed_w),
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        }
    })
}

/// Commit the next recursive witness under config `Cfg`.
///
/// The same-D fast path reuses the caller's prepared backend context. Cross-D
/// commitments prepare a typed backend context for the target ring dimension.
/// The recursive commitment layout is derived from `Cfg::decomposition()` and
/// `Cfg::ring_subfield_embedding_norm_bound()`.
///
/// # Errors
///
/// Returns an error if layout selection, commitment, backend preparation, or
/// D-erased hint conversion fails.
#[inline(never)]
pub fn commit_next_w<Cfg, B, const D: usize>(
    commit_params: &LevelParams,
    expanded: &std::sync::Arc<AkitaExpandedSetup<Cfg::Field>>,
    backend: &B,
    prepared: &B::PreparedSetup<D>,
    logical_w: &RecursiveWitnessFlat,
) -> Result<NextWitnessCommitment<Cfg::Field>, AkitaError>
where
    Cfg: CommitmentConfig,
    Cfg::Field: FieldCore + CanonicalField + RandomSampling + HasWide,
    <Cfg::Field as HasWide>::Wide: AdditiveGroup + From<Cfg::Field> + ReduceTo<Cfg::Field>,
    B: CommitBackend<Cfg::Field>,
{
    if commit_params.ring_dimension == D {
        if <Cfg::ExtField as ExtField<Cfg::Field>>::EXT_DEGREE == 1 {
            let (wc, wh) = commit_w::<Cfg::Field, B, D>(
                logical_w,
                expanded.as_ref(),
                backend,
                prepared,
                commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: None,
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        } else {
            // The tensor pack is length-preserving, so the committed witness
            // fits the schedule's recursive commit params directly.
            let committed_w =
                tensor_pack_recursive_witness::<Cfg::Field, Cfg::ExtField, D>(logical_w)?;
            let (wc, wh) = commit_w::<Cfg::Field, B, D>(
                &committed_w,
                expanded.as_ref(),
                backend,
                prepared,
                commit_params,
            )?;
            Ok(NextWitnessCommitment {
                witness: Some(committed_w),
                commitment: FlatRingVec::from_commitment(&wc),
                hint: RecursiveCommitmentHintCache::from_typed(wh)?,
            })
        }
    } else {
        dispatch_commit_w_with_layout_policy::<Cfg, B>(
            backend,
            commit_params.clone(),
            expanded,
            logical_w,
        )
    }
}
