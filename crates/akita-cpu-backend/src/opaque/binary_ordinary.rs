//! Ordinary full preimages in the explicitly selected mixed native root.

use super::owned::CommitmentHandle;
use super::owned_kind::SourceGroupKind;
use super::CpuBackend;
use akita_error::{checked, AkitaError};
use akita_params::{dispatch_for_field, GroupCommitPhaseParams};
use jolt_field::{CanonicalEncoding, Field};

/// Materialize S, T, F and H in the public native plan's natural digit order.
///
/// This path admits each polynomial with one inner gadget plane and coefficients in
/// the shared base-eight alphabet. The real ordinary commitment still uses its
/// frozen A/B matrices and complete F/H compression chain. Retained polynomial
/// quotient images are unnecessary: the native rows reduce each product in its
/// own negacyclic ring before applying the common evaluation point.
pub(super) fn prepare_ordinary_source<F, E>(
    backend: &CpuBackend<F, E>,
    handle: &CommitmentHandle<F, E>,
    profile: &GroupCommitPhaseParams,
) -> Result<Vec<Vec<i8>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    profile.validate_frozen_precommit(F::MODULUS_BITS)?;
    let modulus = akita_params::field_modulus::<F>()?;
    if handle.owner != backend.owner_id()
        || &handle.committed.parameters != profile
        || !matches!(
            handle.committed.source.group_kind(),
            SourceGroupKind::Ordinary
        )
        || profile.inner.digits.num_digits != 1
        || profile.inner.digits.log_basis < 3
        || profile.outer.digits.log_basis != 3
        || profile.outer.matrix.ring_dimension() != 64
        || !profile
            .inner
            .matrix
            .sis_modulus_profile()
            .matches_modulus(modulus)
        || !profile
            .outer
            .matrix
            .sis_modulus_profile()
            .matches_modulus(modulus)
    {
        return Err(AkitaError::InvalidInput(
            "ordinary preimage does not match the admitted mixed-root geometry".into(),
        ));
    }
    let source_len = checked::pow2(profile.group.num_vars()).ok_or(AkitaError::InvalidProof)?;
    let a_degree = profile.inner.matrix.ring_dimension();
    if checked::product([profile.blocks.live_ring_elements_per_claim, a_degree]) != Some(source_len)
        || handle.committed.metadata.num_vars() != profile.group.num_vars()
        || handle.committed.metadata.num_polynomials() != profile.group.num_polynomials()
    {
        return Err(AkitaError::InvalidInput(
            "ordinary source length differs from its frozen profile".into(),
        ));
    }
    let polynomials = handle.committed.source.dense_polynomials()?;
    if polynomials.len() != profile.group.num_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let per_polynomial = checked::product([
        profile.blocks.live_blocks,
        profile.blocks.positions_per_block,
        a_degree,
    ])
    .ok_or(AkitaError::InvalidProof)?;
    let source_capacity =
        checked::product([per_polynomial, polynomials.len()]).ok_or(AkitaError::InvalidProof)?;
    let alphabet = [-4i8, -3, -2, -1, 0, 1, 2, 3];
    let fields = alphabet.map(|digit| F::from_i64(i64::from(digit)));
    let mut source = Vec::new();
    source
        .try_reserve_exact(source_capacity)
        .map_err(|_| AkitaError::InvalidProof)?;
    for polynomial in polynomials {
        let coefficients = polynomial
            .field_coeffs()
            .get(..source_len)
            .ok_or(AkitaError::InvalidProof)?;
        for coefficient in coefficients {
            let position = fields
                .iter()
                .position(|field| field == coefficient)
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "ordinary mixed-root source exceeds the base-eight alphabet".into(),
                    )
                })?;
            source.push(*alphabet.get(position).ok_or(AkitaError::InvalidProof)?);
        }
        let padding = per_polynomial
            .checked_sub(source_len)
            .ok_or(AkitaError::InvalidProof)?;
        let padded_len = checked::sum([source.len(), padding]).ok_or(AkitaError::InvalidProof)?;
        source.resize(padded_len, 0);
    }

    let retained = handle.committed.ordinary_retained()?;
    let images = retained.inner_rows();
    let image_elements = checked::product([
        profile.blocks.live_blocks,
        profile.inner.matrix.output_rank(),
    ])
    .ok_or(AkitaError::InvalidProof)?;
    let expected_coefficients =
        checked::product([image_elements, a_degree]).ok_or(AkitaError::InvalidProof)?;
    if images.len() != profile.group.num_polynomials()
        || retained.ring_dim() != a_degree
        || images
            .iter()
            .any(|image| image.coeff_len() != expected_coefficients)
    {
        return Err(AkitaError::InvalidInput(
            "ordinary inner image differs from its frozen profile".into(),
        ));
    }
    let t = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        a_degree,
        |D_A| {
            let mut block_rows = Vec::new();
            for image in images {
                block_rows.extend(
                    image
                        .as_ring_slice::<D_A>()?
                        .chunks_exact(profile.inner.matrix.output_rank()),
                );
            }
            crate::kernels::linear::decompose_commit_blocks_into::<F, D_A, 64>(
                &block_rows,
                profile.outer.digits.num_digits,
                profile.outer.digits.log_basis,
            )
            .map(|digits| digits.into_parts().0)
        }
    )?;
    let expected_t = checked::product([
        profile.group.num_polynomials(),
        profile.blocks.live_blocks,
        profile.inner.matrix.output_rank(),
        profile.outer.digits.num_digits,
        a_degree,
    ])
    .ok_or(AkitaError::InvalidProof)?;
    if t.len() != expected_t {
        return Err(AkitaError::InvalidProof);
    }

    // Canonical standalone commitments compress the complete stacked B image
    // through one F/H chain, even when B uses several physical slices.
    let compression = profile.outer_compression_plan()?;
    let witness = retained.outer_compression_witness(&compression)?;
    let [f_stage, h_stage] = witness.stages() else {
        return Err(AkitaError::InvalidProof);
    };
    let f = f_stage
        .expand_rows::<32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let h = h_stage
        .expand_rows::<16>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if f.len() != f_stage.map().padded_digit_count()
        || h.len() != h_stage.map().padded_digit_count()
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(vec![source, t, f, h])
}
