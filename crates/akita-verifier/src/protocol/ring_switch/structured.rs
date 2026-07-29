use super::RelationMatrixGroupEvaluator;
use akita_field::parallel::*;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt, MulBase};

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_group_et_from_eq_slices<F, E>(
    group: &RelationMatrixGroupEvaluator<E>,
    consistency_weight: E,
    a_row_weights: &[E],
    g_open_ext: &[E],
    g_t_commit_ext: &[E],
    e_eq_slice: &[E],
    t_eq_slice: &[E],
) -> Result<(E, E), AkitaError>
where
    F: FieldCore + FromPrimitiveInt,
    E: FieldCore + MulBase<F>,
{
    let e_stride = group.depth_open;
    let t_stride = group
        .n_a
        .checked_mul(group.depth_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("T fold stride overflow".into()))?;
    let block_claims = group
        .num_claims
        .checked_mul(group.num_live_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("structured block count overflow".into()))?;
    let expected_e = block_claims
        .checked_mul(e_stride)
        .ok_or_else(|| AkitaError::InvalidSetup("structured E width overflow".into()))?;
    let expected_t = block_claims
        .checked_mul(t_stride)
        .ok_or_else(|| AkitaError::InvalidSetup("structured T width overflow".into()))?;
    if e_eq_slice.len() != expected_e
        || t_eq_slice.len() != expected_t
        || g_open_ext.len() != group.depth_open
        || g_t_commit_ext.len() != group.depth_commit
        || a_row_weights.len() != group.n_a
    {
        return Err(AkitaError::InvalidProof);
    }
    let challenge_factors = (0..group.num_claims)
        .map(|claim| {
            group
                .c_alphas
                .affine_factors::<F>(claim, group.num_live_blocks)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let t_weights = a_row_weights
        .iter()
        .flat_map(|&row_weight| {
            g_t_commit_ext
                .iter()
                .map(move |&gadget| row_weight * gadget)
        })
        .collect::<Vec<_>>();
    let _span = tracing::info_span!(
        "structured_et_from_setup_slices",
        block_claims,
        e_columns = expected_e,
        t_columns = expected_t
    )
    .entered();
    cfg_fold_reduce!(
        0..block_claims,
        || Ok((E::zero(), E::zero())),
        |acc: Result<(E, E), AkitaError>, block_claim| {
            let (mut e_acc, mut t_acc) = acc?;
            let claim = block_claim / group.num_live_blocks;
            let block = block_claim % group.num_live_blocks;
            let challenge = challenge_factors
                .get(claim)
                .and_then(|factors| factors.low.get(block))
                .copied()
                .ok_or(AkitaError::InvalidProof)?;
            let e_start = block_claim
                .checked_mul(e_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let e_end = e_start
                .checked_add(e_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let e_block = e_eq_slice
                .get(e_start..e_end)
                .ok_or(AkitaError::InvalidProof)?;
            let mut e_weight = E::zero();
            for (&eq, &gadget) in e_block.iter().zip(g_open_ext) {
                e_weight += eq * gadget;
            }
            e_acc += challenge * consistency_weight * e_weight;

            let t_start = block_claim
                .checked_mul(t_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let t_end = t_start
                .checked_add(t_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let t_block = t_eq_slice
                .get(t_start..t_end)
                .ok_or(AkitaError::InvalidProof)?;
            let mut t_weight = E::zero();
            for (&eq, &weight) in t_block.iter().zip(&t_weights) {
                t_weight += eq * weight;
            }
            t_acc += challenge * t_weight;
            Ok((e_acc, t_acc))
        },
        |lhs: Result<(E, E), AkitaError>, rhs: Result<(E, E), AkitaError>| {
            let (lhs_e, lhs_t) = lhs?;
            let (rhs_e, rhs_t) = rhs?;
            Ok((lhs_e + rhs_e, lhs_t + rhs_t))
        }
    )
}
