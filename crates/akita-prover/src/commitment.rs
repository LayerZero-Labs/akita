//! Prover-owned commitment kernels.

use crate::crt_ntt::NttSlotCache;
use crate::dense::DensePoly;
use crate::linear::mat_vec_mul_ntt_single_i8;
use crate::{HachiPolyOps, HachiProverSetup};
use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::{CanonicalField, FieldCore, HachiError};
use akita_types::{
    checked_total_claims, checked_total_groups, DirectWitnessProof, FlatDigitBlocks,
    HachiCommitmentHint, HachiVerifierSetup, LevelParams, MultiPointBatchShape, RingCommitment,
};

/// Commit a group of polynomials using already-selected level parameters.
///
/// Root config/schedule policy chooses `params`; this function owns only the
/// prover-side matrix work for the supplied concrete layout.
///
/// # Errors
///
/// Returns an error if an inner witness commitment or hint allocation fails.
pub fn commit_with_params<F, const D: usize, P>(
    polys: &[P],
    setup: &HachiProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(RingCommitment<F, D>, HachiCommitmentHint<F, D>), HachiError>
where
    F: FieldCore + CanonicalField,
    P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let t_hat_flat_len_per_poly =
        params.num_blocks * params.a_key.row_len() * params.num_digits_open;
    let mut t_hat_flat = vec![[0i8; D]; polys.len() * t_hat_flat_len_per_poly];
    let mut t_hat_vec: Vec<FlatDigitBlocks<D>> = (0..polys.len())
        .map(|_| FlatDigitBlocks::new(Vec::new(), Vec::new()))
        .collect::<Result<_, _>>()?;
    let mut t_vec: Vec<Vec<Vec<CyclotomicRing<F, D>>>> = vec![Vec::new(); polys.len()];
    cfg_chunks_mut!(t_hat_flat, t_hat_flat_len_per_poly)
        .zip(cfg_iter!(polys))
        .zip(cfg_iter_mut!(t_hat_vec))
        .zip(cfg_iter_mut!(t_vec))
        .try_for_each(|(((dst, poly), t_hat), t)| -> Result<(), HachiError> {
            let inner = poly.commit_inner_witness(
                &setup.expanded.shared_matrix,
                &setup.ntt_shared,
                params.a_key.row_len(),
                params.block_len,
                params.num_digits_commit,
                params.num_digits_open,
                params.log_basis,
                setup.expanded.seed.max_stride,
            )?;
            dst.copy_from_slice(inner.t_hat.flat_digits());
            *t_hat = inner.t_hat;
            *t = inner.t;
            Ok(())
        })?;
    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        &setup.ntt_shared,
        params.b_key.row_len(),
        setup.expanded.seed.max_stride,
        &t_hat_flat,
    );
    Ok((
        RingCommitment { u },
        HachiCommitmentHint::with_t(t_hat_vec, t_vec),
    ))
}

/// Commit multiple polynomial groups with one already-selected root layout.
///
/// Root config/schedule policy chooses `params`; this function owns the
/// repeated prover-side commitment work for each supplied group.
///
/// # Errors
///
/// Returns an error if any group commitment fails.
#[allow(clippy::type_complexity)]
pub fn batched_commit_with_params<F, const D: usize, P>(
    poly_groups: &[&[P]],
    setup: &HachiProverSetup<F, D>,
    params: &LevelParams,
) -> Result<(Vec<RingCommitment<F, D>>, Vec<HachiCommitmentHint<F, D>>), HachiError>
where
    F: FieldCore + CanonicalField,
    P: HachiPolyOps<F, D, CommitCache = NttSlotCache<D>>,
{
    let mut commitments = Vec::with_capacity(poly_groups.len());
    let mut hints = Vec::with_capacity(poly_groups.len());
    for group in poly_groups {
        let (commitment, hint) = commit_with_params::<F, D, P>(group, setup, params)?;
        commitments.push(commitment);
        hints.push(hint);
    }
    Ok((commitments, hints))
}

/// Recompute root-direct commitments from direct field-element witnesses.
///
/// Root verification supplies the config-selected direct commitment layout;
/// this function owns the prover-side reconstruction and grouped recommit.
///
/// # Errors
///
/// Returns an error if the direct witness payloads are malformed, the batch
/// shape is inconsistent, or a recomputed commitment differs from the expected
/// commitment.
pub fn verify_root_direct_commitments_with_params<F, const D: usize>(
    witnesses: &[DirectWitnessProof<F>],
    setup: &HachiVerifierSetup<F>,
    flat_commitments: &[RingCommitment<F, D>],
    batch_shape: &MultiPointBatchShape,
    params: &LevelParams,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField,
{
    if flat_commitments.len() != batch_shape.claim_group_sizes.len() {
        return Err(HachiError::InvalidProof);
    }
    let total_groups = checked_total_groups(
        &batch_shape.point_group_sizes,
        "root_direct_commitment_check",
    )?;
    if total_groups != batch_shape.claim_group_sizes.len() {
        return Err(HachiError::InvalidProof);
    }
    let total_claims = checked_total_claims(
        &batch_shape.claim_group_sizes,
        "root_direct_commitment_check",
    )?;
    if total_claims != witnesses.len() {
        return Err(HachiError::InvalidProof);
    }

    let total = setup.expanded.shared_matrix.total_ring_elements_at::<D>();
    let verifier_ntt =
        crate::crt_ntt::build_ntt_slot(setup.expanded.shared_matrix.ring_view::<D>(1, total))
            .map_err(|_| HachiError::InvalidProof)?;
    let temp_setup = HachiProverSetup {
        expanded: setup.expanded.clone(),
        ntt_shared: verifier_ntt,
    };

    let mut claim_offset = 0usize;
    let mut poly_groups = Vec::with_capacity(batch_shape.claim_group_sizes.len());
    for &group_size in &batch_shape.claim_group_sizes {
        let group_witnesses = &witnesses[claim_offset..claim_offset + group_size];
        let group_polys = group_witnesses
            .iter()
            .map(|witness| {
                let field_witness = witness
                    .as_field_elements()
                    .ok_or(HachiError::InvalidProof)?
                    .coeffs();
                let coeff_len = field_witness.len();
                if !coeff_len.is_power_of_two() {
                    return Err(HachiError::InvalidProof);
                }
                let num_vars = coeff_len.trailing_zeros() as usize;
                DensePoly::<F, D>::from_field_evals(num_vars, field_witness)
                    .map_err(|_| HachiError::InvalidProof)
            })
            .collect::<Result<Vec<_>, _>>()?;

        poly_groups.push(group_polys);
        claim_offset += group_size;
    }
    let poly_group_refs = poly_groups
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<&[DensePoly<F, D>]>>();

    let mut expected_commitments = Vec::with_capacity(poly_group_refs.len());
    for group in poly_group_refs {
        let (commitment, _) =
            commit_with_params::<F, D, DensePoly<F, D>>(group, &temp_setup, params)
                .map_err(|_| HachiError::InvalidProof)?;
        expected_commitments.push(commitment);
    }

    if expected_commitments != flat_commitments {
        return Err(HachiError::InvalidProof);
    }

    Ok(())
}
