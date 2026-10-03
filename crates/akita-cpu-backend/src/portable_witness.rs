//! Private compression witness material for immediate transfer to opaque backends.

use crate::opaque::witness_build::compression_emission::{
    emit_packed_negative_binary, quotient_digits,
};
use crate::{CpuBackend, PortableCompressionState};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_error::AkitaError;
use akita_params::{
    field_modulus, r_decomp_levels, CommittedGroupParams, RelationQuotientPlan, RelationRhsLayout,
    RingRelationMode, WitnessLayout,
};
use jolt_field::{CanonicalEncoding, Field};

/// One nonempty canonical compression-owned range of final witness coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortableWitnessPatch {
    offset: usize,
    coefficients: Vec<i8>,
}

impl PortableWitnessPatch {
    /// Absolute coefficient offset in the validated witness layout.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Final i8 witness coefficients, including canonical compression digit padding.
    #[must_use]
    pub fn coefficients(&self) -> &[i8] {
        &self.coefficients
    }

    /// Consume the patch into its offset and coefficients.
    #[must_use]
    pub fn into_parts(self) -> (usize, Vec<i8>) {
        (self.offset, self.coefficients)
    }
}

/// Canonical F/H patches containing private witness material, sorted by offset.
/// Consumers must treat the witness length and offsets as one validated unit.
/// These CPU-emitted coefficients are ready for immediate transfer into an
/// opaque backend; protocol sequencing and transcript behavior remain with the prover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressionWitnessFragment {
    witness_len: usize,
    patches: Vec<PortableWitnessPatch>,
}

impl CompressionWitnessFragment {
    /// Canonical `WitnessLayout::live_coeff_len()` used to validate every patch.
    #[must_use]
    pub const fn witness_len(&self) -> usize {
        self.witness_len
    }

    /// Nonempty, disjoint patches in strictly increasing offset order.
    #[must_use]
    pub fn patches(&self) -> &[PortableWitnessPatch] {
        &self.patches
    }

    /// Consume the fragment into its validated length and patches.
    #[must_use]
    pub fn into_parts(self) -> (usize, Vec<PortableWitnessPatch>) {
        (self.witness_len, self.patches)
    }
}

impl<F: Field + CanonicalEncoding + 'static, E> CpuBackend<F, E> {
    /// Materialize only F/H digit and quotient ranges using canonical CPU emission.
    /// Requires no setup matrices and changes no backend caches or proof sessions.
    ///
    /// # Errors
    /// Rejects states in the wrong relation order, incompatible plans or modes,
    /// and layouts or quotient depths inconsistent with the level.
    #[allow(clippy::too_many_arguments)]
    pub fn materialize_compression_witness_fragment(
        &self,
        level: &CommittedGroupParams,
        relation_layout: &RelationRhsLayout,
        witness_layout: &WitnessLayout,
        relation_mode: RingRelationMode,
        outer_states: &[PortableCompressionState<F>],
        opening_state: &PortableCompressionState<F>,
    ) -> Result<CompressionWitnessFragment, AkitaError> {
        let log_basis = level.open().digits.log_basis;
        crate::validation::validate_i8_setup_log_basis(
            log_basis,
            "for compression witness patches",
        )?;
        if level.ring_relation_mode != relation_mode
            || outer_states.len() != relation_layout.groups.len()
        {
            return Err(AkitaError::InvalidProof);
        }
        let quotient_plan = RelationQuotientPlan::for_field_bits(level, F::MODULUS_BITS)?;
        witness_layout.validate_tail(level, relation_layout, quotient_plan)?;
        let opening_plan = relation_layout.opening_compression_plan()?;
        opening_state.validate(opening_plan, relation_mode)?;
        if witness_layout.compression_layers().len() != opening_plan.maps().len() {
            return Err(AkitaError::InvalidProof);
        }
        let modulus = field_modulus::<F>()?;
        for (relation_index, state) in outer_states.iter().enumerate() {
            let (_, plan) = relation_layout.group_compression_plan(relation_index)?;
            state.validate(plan, relation_mode)?;
            if !plan.modulus_profile().matches_modulus(modulus) {
                return Err(AkitaError::InvalidProof);
            }
        }
        if !opening_plan.modulus_profile().matches_modulus(modulus) {
            return Err(AkitaError::InvalidProof);
        }
        let levels = r_decomp_levels::<F>(log_basis);
        let decompose_params = match relation_mode {
            RingRelationMode::QuotientLift => {
                if witness_layout.quotient_depth() != Some(levels) {
                    return Err(AkitaError::InvalidProof);
                }
                Some(BalancedDecomposePow2Params::new(levels, log_basis))
            }
            RingRelationMode::ReducedEvaluation => None,
        };
        let mut patches = Vec::new();
        for layer in witness_layout.compression_layers() {
            let sources = outer_states
                .iter()
                .zip(layer.f_spans())
                .enumerate()
                .map(|(index, (state, (_, span)))| {
                    (
                        state,
                        span,
                        layer
                            .f_quotient_rows()
                            .and_then(|rows| rows.get(index).map(|(_, row)| *row)),
                    )
                })
                .chain(std::iter::once((
                    opening_state,
                    layer.h_span(),
                    layer.h_quotient_row(),
                )));
            for (state, span, row_index) in sources {
                let packed = state
                    .witness()
                    .stages()
                    .get(layer.map_index())
                    .ok_or(AkitaError::InvalidProof)?;
                let mut coefficients = Vec::with_capacity(span.range().len());
                emit_packed_negative_binary(
                    |_, digits| {
                        coefficients.extend_from_slice(digits);
                        Ok(())
                    },
                    span,
                    packed,
                )?;
                patches.push(PortableWitnessPatch {
                    offset: span.range().start,
                    coefficients,
                });
                match (row_index, decompose_params.as_ref()) {
                    (Some(index), Some(params)) => {
                        let row = witness_layout
                            .r_rows()
                            .get(index)
                            .ok_or(AkitaError::InvalidProof)?;
                        let quotient = state
                            .quotients()
                            .and_then(|rows| rows.get(layer.map_index()))
                            .ok_or(AkitaError::InvalidProof)?;
                        if quotient.ring_dim() != row.geometry().polynomial_modulus_dimension() {
                            return Err(AkitaError::InvalidProof);
                        }
                        patches.push(PortableWitnessPatch {
                            offset: row.range().start,
                            coefficients: quotient_digits(quotient.coeffs(), row, params)?,
                        });
                    }
                    (None, None) => {}
                    _ => return Err(AkitaError::InvalidProof),
                }
            }
        }
        patches.sort_unstable_by_key(PortableWitnessPatch::offset);
        let tail = witness_layout.tail_range();
        let mut end = tail.start;
        for patch in &patches {
            let next = patch
                .offset
                .checked_add(patch.coefficients.len())
                .ok_or(AkitaError::InvalidProof)?;
            if patch.coefficients.is_empty() || patch.offset < end || next > tail.end {
                return Err(AkitaError::InvalidProof);
            }
            end = next;
        }
        Ok(CompressionWitnessFragment {
            witness_len: witness_layout.live_coeff_len(),
            patches,
        })
    }
}
