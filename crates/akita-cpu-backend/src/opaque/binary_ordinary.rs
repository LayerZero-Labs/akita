//! Ordinary full preimages in the explicitly selected mixed native root.

use super::owned::CommitmentHandle;
use super::owned_kind::SourceGroupKind;
use super::CpuBackend;
use akita_error::{checked, AkitaError};
use akita_params::dispatch_for_field;
use akita_types::binary::OrdinaryNativeRootPlan;
use jolt_field::{CanonicalEncoding, Field};

impl<F, E> CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    /// Materialize the selected ordinary plan from the actual committed handle.
    ///
    /// Includes every source digit, the A image and both compression
    /// preimages. The caller must authenticate all segments through the
    /// shared root relation.
    pub(crate) fn ordinary_root_preimage(
        &self,
        handle: &CommitmentHandle<F, E>,
        plan: &OrdinaryNativeRootPlan,
    ) -> Result<Vec<Vec<i8>>, AkitaError> {
        let profile = plan.profile();
        profile.validate_frozen_precommit(F::MODULUS_BITS)?;
        let modulus = akita_params::field_modulus::<F>()?;
        if handle.owner != self.owner_id()
            || &handle.committed.parameters != profile
            || !matches!(
                handle.committed.source.group_kind(),
                SourceGroupKind::Ordinary
            )
            || profile.outer.digits.log_basis != 3
            || profile.outer.matrix.ring_dimension()
                != akita_types::binary::MIXED_ROOT_ORDINARY_B_DIMENSION
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
        if checked::product([profile.blocks.live_ring_elements_per_claim, a_degree])
            != Some(source_len)
            || handle.committed.metadata.num_vars() != profile.group.num_vars()
            || handle.committed.metadata.num_polynomials() != profile.group.num_polynomials()
        {
            return Err(AkitaError::InvalidInput(
                "ordinary source length differs from its frozen profile".into(),
            ));
        }
        let polynomials = handle.committed.source.dense_polynomials()?;
        let retained = handle.committed.ordinary_retained()?;
        let inner_rows = retained.inner_rows()?;
        if polynomials.len() != profile.group.num_polynomials()
            || inner_rows.len() != polynomials.len()
            || retained.ring_dim() != a_degree
        {
            return Err(AkitaError::InvalidProof);
        }
        let mut source = Vec::new();
        for polynomial in &polynomials {
            let coefficients = polynomial
                .field_coeffs()
                .get(..source_len)
                .ok_or(AkitaError::InvalidProof)?;
            let digits = source_digits(plan, coefficients)?;
            source
                .try_reserve_exact(digits.len())
                .map_err(|_| AkitaError::InvalidProof)?;
            source.extend(digits);
        }
        let image_elements = checked::product([
            profile.blocks.live_blocks,
            profile.inner.matrix.output_rank(),
        ])
        .ok_or(AkitaError::InvalidProof)?;
        // T keeps global block/A-row/subcolumn/digit/coefficient order. B slices
        // select canonical contiguous block ranges from that table; a slice need
        // not contain exactly one block.
        let mut t = Vec::new();
        for image in inner_rows.iter() {
            if checked::product([image_elements, a_degree]) != Some(image.coeff_len()) {
                return Err(AkitaError::InvalidInput(
                    "ordinary inner image differs from its frozen profile".into(),
                ));
            }
            let digits = dispatch_for_field!(
                ProtocolDispatchSlot::Role(RingRole::Inner),
                F,
                a_degree,
                |D_A| {
                    let rings = image.as_ring_slice::<D_A>()?;
                    let block_rows = rings
                        .chunks_exact(profile.inner.matrix.output_rank())
                        .collect::<Vec<_>>();
                    crate::kernels::linear::decompose_commit_blocks_into::<F, D_A, 64>(
                        &block_rows,
                        profile.outer.digits.num_digits,
                        profile.outer.digits.log_basis,
                    )
                    .map(|digits| digits.into_parts().0)
                }
            )?;
            let expected_t =
                checked::product([image_elements, profile.outer.digits.num_digits, a_degree])
                    .ok_or(AkitaError::InvalidProof)?;
            if digits.len() != expected_t {
                return Err(AkitaError::InvalidProof);
            }
            t.try_reserve_exact(digits.len())
                .map_err(|_| AkitaError::InvalidProof)?;
            t.extend(digits);
        }
        // Canonical standalone commitments compress the complete stacked B image
        // through one F/H chain, even when B uses several physical slices.
        let compression = profile.outer_compression_plan()?;
        let witness = retained.outer_compression_witness(&compression)?;
        let [f_stage, h_stage] = witness.stages() else {
            return Err(AkitaError::InvalidProof);
        };
        let (f, h) = match compression.field_bits() {
            128 => (
                f_stage
                    .expand_rows::<32>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
                h_stage
                    .expand_rows::<16>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
            ),
            64 => (
                f_stage
                    .expand_rows::<64>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
                h_stage
                    .expand_rows::<32>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
            ),
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "unsupported mixed compression field".into(),
                ))
            }
        };
        if f.len() != f_stage.map().padded_digit_count()
            || h.len() != h_stage.map().padded_digit_count()
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(vec![source, t, f, h])
    }
}

/// Balanced base-eight source digits. The plan admits one digit per
/// coefficient, so every value must lie in `[-4, 3]`.
fn source_digits<F: Field + CanonicalEncoding>(
    plan: &OrdinaryNativeRootPlan,
    coefficients: &[F],
) -> Result<Vec<i8>, AkitaError> {
    use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
    let profile = plan.profile();
    let degree = profile.inner.matrix.ring_dimension();
    let digits = profile.inner.digits.num_digits;
    let digit_count =
        checked::product([coefficients.len(), digits]).ok_or(AkitaError::InvalidProof)?;
    if !coefficients.len().is_multiple_of(degree) {
        return Err(AkitaError::InvalidProof);
    }
    let mut source = Vec::new();
    source
        .try_reserve_exact(digit_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    let modulus = akita_params::field_modulus::<F>()?;
    let params = BalancedDecomposePow2Params::<F>::new(digits, 3);
    let ring_digits = checked::product([degree, digits]).ok_or(AkitaError::InvalidProof)?;
    for ring in coefficients.chunks_exact(degree) {
        for coefficient in ring {
            let value = coefficient
                .to_u128_checked()
                .ok_or(AkitaError::InvalidProof)?;
            if value >= 4 && modulus.checked_sub(value).is_none_or(|x| x > 4) {
                return Err(AkitaError::InvalidInput(
                    "ordinary coefficient exceeds one balanced digit".into(),
                ));
            }
        }
        let start = source.len();
        let end = checked::sum([start, ring_digits]).ok_or(AkitaError::InvalidProof)?;
        source.resize(end, 0);
        akita_algebra::balanced_decompose_coefficients_pow2_i8_into(
            ring,
            source.get_mut(start..end).ok_or(AkitaError::InvalidProof)?,
            &params,
        );
    }
    Ok(source)
}

#[cfg(test)]
#[path = "binary_ordinary_slice_tests.rs"]
mod slice_tests;
