//! Ordinary full preimages in the explicitly selected mixed native root.

use super::owned::CommitmentHandle;
use super::owned_kind::SourceGroupKind;
use super::CpuBackend;
use akita_error::AkitaError;
use jolt_algebra::checked;
use akita_params::dispatch_for_field;
use akita_types::binary_source::OrdinaryNativeRootPlan;
use jolt_algebra::{CanonicalEncoding, Field};

impl<F, E> CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    /// Materialize the selected ordinary plan from the actual committed handle.
    ///
    /// Includes every source digit, A image, both compression preimages, and
    /// (for an explicitly signed plan) the negative offset bits. The caller
    /// must authenticate all segments through the shared root relation.
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
        if checked::product([profile.blocks.live_ring_elements_per_claim, a_degree])
            != Some(source_len)
            || handle.committed.metadata.num_vars() != profile.group.num_vars()
            || handle.committed.metadata.num_polynomials() != profile.group.num_polynomials()
        {
            return Err(AkitaError::InvalidInput(
                "ordinary source length differs from its frozen profile".into(),
            ));
        }
        if let Some(width) = plan.source_width() {
            if handle.committed.producer_contract != width.producer_contract()? {
                return Err(AkitaError::InvalidInput(
                    "ordinary source width differs from its producer contract".into(),
                ));
            }
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
        let mut bits = Vec::new();
        for polynomial in &polynomials {
            let coefficients = polynomial
                .field_coeffs()
                .get(..source_len)
                .ok_or(AkitaError::InvalidProof)?;
            let (digits, offset_bits) = source_digits(plan, coefficients)?;
            source
                .try_reserve_exact(digits.len())
                .map_err(|_| AkitaError::InvalidProof)?;
            bits.try_reserve_exact(offset_bits.len())
                .map_err(|_| AkitaError::InvalidProof)?;
            source.extend(digits);
            bits.extend(offset_bits);
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
        let mut segments = vec![source, t, f, h];
        if plan.source_width().is_some() {
            segments.push(bits);
        }
        Ok(segments)
    }
}

fn source_digits<F: Field + CanonicalEncoding>(
    plan: &OrdinaryNativeRootPlan,
    coefficients: &[F],
) -> Result<(Vec<i8>, Vec<i8>), AkitaError> {
    use jolt_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
    let profile = plan.profile();
    let degree = profile.inner.matrix.ring_dimension();
    let digits = profile.inner.digits.num_digits;
    let digit_count =
        checked::product([coefficients.len(), digits]).ok_or(AkitaError::InvalidProof)?;
    let bit_count = checked::product([
        coefficients.len(),
        plan.source_width().map_or(0, |width| width.bits() as usize),
    ])
    .ok_or(AkitaError::InvalidProof)?;
    if !coefficients.len().is_multiple_of(degree) {
        return Err(AkitaError::InvalidProof);
    }
    let mut source = Vec::new();
    let mut bits = Vec::new();
    source
        .try_reserve_exact(digit_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    bits.try_reserve_exact(bit_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    let modulus = akita_params::binary_setup::BINARY_SETUP_P;
    let offset = plan.source_width().map_or(4, |width| width.offset());
    let params = BalancedDecomposePow2Params::<F>::new(digits, 3);
    let ring_digits = checked::product([degree, digits]).ok_or(AkitaError::InvalidProof)?;
    for ring in coefficients.chunks_exact(degree) {
        let mut shifted = Vec::new();
        shifted
            .try_reserve_exact(degree)
            .map_err(|_| AkitaError::InvalidProof)?;
        for coefficient in ring {
            let value = coefficient
                .to_u128_checked()
                .ok_or(AkitaError::InvalidProof)?;
            let encoded = if value < offset {
                offset + value
            } else {
                let magnitude = modulus
                    .checked_sub(value)
                    .filter(|&x| x <= offset)
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "ordinary coefficient exceeds its signed width".into(),
                        )
                    })?;
                offset - magnitude
            };
            shifted.push(encoded);
        }
        let start = source.len();
        let end = checked::sum([start, ring_digits]).ok_or(AkitaError::InvalidProof)?;
        source.resize(end, 0);
        jolt_algebra::balanced_decompose_coefficients_pow2_i8_into(
            ring,
            source.get_mut(start..end).ok_or(AkitaError::InvalidProof)?,
            &params,
        );
        if let Some(width) = plan.source_width() {
            for bit in 0..width.bits() {
                bits.extend(shifted.iter().map(|value| -(((value >> bit) & 1) as i8)));
            }
        }
    }
    Ok((source, bits))
}

#[cfg(test)]
#[path = "binary_ordinary_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "binary_ordinary_slice_tests.rs"]
mod slice_tests;
