use crate::protocol::batched::{
    direct_decomposed_inner_rows, field_evals_to_rings, mat_vec_mul_i8_plain, zk_b_blinding_rows,
};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_r1cs::{zk_masked_linear_value_lc, ZkR1csLinearCombination};
use akita_types::{
    recover_ring_subfield_inner_product, AkitaVerifierSetup, LevelParams, RingSubfieldEncoding,
    ZkHidingProof,
};

pub(super) fn zk_recovered_y_ring_lc<F, E, const D: usize>(
    y_ring: &CyclotomicRing<F, D>,
    y_masks: &[ZkR1csLinearCombination<E>],
    inner_reduction: &CyclotomicRing<F, D>,
) -> Result<ZkR1csLinearCombination<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: RingSubfieldEncoding<F>,
{
    if y_masks.len() != D {
        return Err(AkitaError::InvalidProof);
    }
    let masked_opening = recover_ring_subfield_inner_product::<F, E, D>(y_ring, inner_reduction)?;
    let mut mask_coeffs = Vec::with_capacity(D);
    for coeff_idx in 0..D {
        let mut basis_y = CyclotomicRing::<F, D>::zero();
        basis_y.coeffs[coeff_idx] = F::one();
        mask_coeffs.push(recover_ring_subfield_inner_product::<F, E, D>(
            &basis_y,
            inner_reduction,
        )?);
    }
    zk_masked_linear_value_lc(masked_opening, y_masks, &mask_coeffs)
}

pub(super) fn verify_zk_hiding_commitment<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    root_params: &LevelParams,
    proof: &ZkHidingProof<F>,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if D == 0 || proof.u_blind.is_empty() || proof.hiding_witness.is_empty() {
        return Err(AkitaError::InvalidProof);
    }

    let num_ring = proof
        .hiding_witness
        .len()
        .div_ceil(D)
        .max(1)
        .checked_next_power_of_two()
        .ok_or(AkitaError::InvalidProof)?;
    let eval_len = num_ring
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("ZK hiding witness length overflow".to_string()))?;
    let mut evals = vec![F::zero(); eval_len];
    let live_evals = evals
        .get_mut(..proof.hiding_witness.len())
        .ok_or(AkitaError::InvalidProof)?;
    live_evals.copy_from_slice(&proof.hiding_witness);

    let hiding_params = root_params.with_decomp(
        num_ring.trailing_zeros() as usize,
        0,
        root_params.num_digits_commit,
        root_params.num_digits_open,
        num_ring,
    )?;
    let witness_rings = field_evals_to_rings::<F, D>(&evals)?;
    let b_input_digits = direct_decomposed_inner_rows(&witness_rings, setup, &hiding_params)?;
    let shared_matrix = setup.expanded.shared_matrix();
    let b_required = hiding_params
        .b_key
        .row_len()
        .checked_mul(b_input_digits.len())
        .ok_or_else(|| AkitaError::InvalidSetup("ZK hiding B footprint overflow".to_string()))?;
    if b_required > shared_matrix.total_ring_elements_at::<D>()? {
        return Err(AkitaError::InvalidSetup(
            "ZK hiding commitment exceeds shared matrix length".to_string(),
        ));
    }

    let b_matrix =
        shared_matrix.ring_view::<D>(hiding_params.b_key.row_len(), b_input_digits.len())?;
    let b_rows: Vec<_> = b_matrix.rows().collect();
    let mut expected_u_blind_rings = mat_vec_mul_i8_plain::<F, D>(&b_rows, &b_input_digits);
    let blinding_rows =
        zk_b_blinding_rows::<F, D>(setup, &hiding_params, &proof.b_blinding_digits)?;
    for (row, blinding) in expected_u_blind_rings.iter_mut().zip(blinding_rows) {
        *row += blinding;
    }
    let expected_len = expected_u_blind_rings
        .len()
        .checked_mul(D)
        .ok_or(AkitaError::InvalidProof)?;
    if proof.u_blind.len() != expected_len {
        return Err(AkitaError::InvalidProof);
    }
    let expected_u_blind = expected_u_blind_rings
        .iter()
        .flat_map(|ring| ring.coeffs.iter().copied())
        .collect::<Vec<_>>();
    if proof.u_blind.as_slice() != expected_u_blind.as_slice() {
        return Err(AkitaError::InvalidProof);
    }

    Ok(())
}
