//! Arithmetic over owned homogeneous sources, private to the CPU backend.
use crate::opaque::*;
use crate::opaque::{OpeningBatchKernel, OpeningFoldOutput};
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_params::*;
use akita_types::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};
type FoldedClaimEvals<F, const D: usize> =
    (Vec<CyclotomicRing<F, D>>, Vec<Vec<CyclotomicRing<F, D>>>);
pub(super) struct PreparedExtensionOpeningGroup<E: Field> {
    pub(super) proof_partials: Vec<E>,
    pub(super) row_partials_by_claim: Vec<Vec<E>>,
    pub(super) openings: Vec<E>,
}
fn evaluate_claims_at_prepared_point<F, E, Q, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    polys: &[&Q],
    prepared_point: &PreparedOpeningPoint<F, E>,
    num_positions_per_block: usize,
) -> Result<FoldedClaimEvals<F, D>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: Field,
    Q: RootOpeningSource<F, D>,
    B: ComputeBackendSetup<F> + for<'a> OpeningBatchKernel<Q::OpeningBatchView<'a>, F, D>,
{
    let _span = tracing::info_span!(
        "fold_evaluate_claims",
        num_claims = polys.len(),
        source = std::any::type_name::<Q>(),
        ring_dimension = D,
        positions_per_block = num_positions_per_block,
        live_blocks = prepared_point.ring_multiplier_point.fold_len(),
        compact_subfield = prepared_point.ring_multiplier_point.as_base().is_none(),
    )
    .entered();
    let point = &prepared_point.ring_multiplier_point;
    let plan = if let Some(base_point) = point.as_base() {
        OpeningFoldPlan::Base {
            live_block_weights: &base_point.live_block_weights,
            position_weights: &base_point.position_weights,
            num_positions_per_block,
        }
    } else {
        OpeningFoldPlan::Subfield {
            multipliers: point.as_subfield().ok_or_else(|| {
                AkitaError::Internal(
                    "source opening multiplier point is neither base nor subfield".into(),
                )
            })?,
            num_positions_per_block,
        }
    };
    let outputs = backend.evaluate_and_fold_batch(prepared, Q::opening_batch(polys)?, plan)?;
    if outputs.len() != polys.len() {
        return Err(AkitaError::InvalidSize {
            expected: polys.len(),
            actual: outputs.len(),
        });
    }
    for output in &outputs {
        if output.folded.len() != point.fold_len() {
            return Err(AkitaError::InvalidSize {
                expected: point.fold_len(),
                actual: output.folded.len(),
            });
        }
    }
    let (folded_rings, folded_blocks) = outputs
        .into_iter()
        .map(|OpeningFoldOutput { eval, folded }| (eval, folded))
        .unzip();
    Ok((folded_rings, folded_blocks))
}

/// Prepare one group's opening point and evaluate all of its claims.
///
/// Transcript ownership stays with the protocol layer so preparation-only
/// padding can never become public transcript state.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_and_evaluate_opening_group<F, E, Q, B, const D: usize>(
    backend: &B,
    prepared: Option<&B::PreparedSetup>,
    polys: &[&Q],
    protocol_point: &[E],
    basis: BasisMode,
    num_positions_per_block: usize,
    num_live_blocks: usize,
    alpha_bits: usize,
) -> Result<(PreparedOpeningPoint<F, E>, FoldedClaimEvals<F, D>), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: FpExtEncoding<F> + ExtField<F>,
    Q: RootOpeningSource<F, D>,
    B: ComputeBackendSetup<F> + for<'a> OpeningBatchKernel<Q::OpeningBatchView<'a>, F, D>,
{
    let prepared_point = prepare_opening_point::<F, E, D>(
        protocol_point,
        basis,
        num_positions_per_block,
        num_live_blocks,
        alpha_bits,
    )?;
    let folded = evaluate_claims_at_prepared_point(
        backend,
        prepared,
        polys,
        &prepared_point,
        num_positions_per_block,
    )?;
    Ok((prepared_point, folded))
}

pub(super) fn scalar_opening_from_folded_ring<F, E, const D: usize>(
    folded_ring: &CyclotomicRing<F, D>,
    prepared_point: &PreparedOpeningPoint<F, E>,
    inner_opening_point: &[E],
    basis: BasisMode,
) -> Result<E, AkitaError>
where
    F: Field + Ring,
    E: FpExtEncoding<F>,
{
    if <E as ExtField<F>>::DEGREE == 1 {
        return Ok(E::lift_base(folded_ring.coefficient_inner_product(
            prepared_point.packed_inner_trusted::<D>()?,
        )));
    }
    if !D.is_multiple_of(<E as ExtField<F>>::DEGREE)
        || !(D / <E as ExtField<F>>::DEGREE).is_power_of_two()
    {
        return Err(AkitaError::InvalidInput(
            "extension-field degree must divide the ring dimension into power-of-two slots"
                .to_string(),
        ));
    }
    let packed_slots = D / <E as ExtField<F>>::DEGREE;
    let packed_inner_bits = packed_slots.trailing_zeros() as usize;
    if inner_opening_point.len() > packed_inner_bits
        && inner_opening_point[packed_inner_bits..]
            .iter()
            .any(|coord| !coord.is_zero())
    {
        return Err(AkitaError::InvalidPointDimension {
            expected: packed_inner_bits,
            actual: inner_opening_point.len(),
        });
    }
    let mut point =
        inner_opening_point[..inner_opening_point.len().min(packed_inner_bits)].to_vec();
    point.resize(packed_inner_bits, E::zero());
    let weights = basis_weights(&point, basis)?;
    let packed_inner_point = embed_ring_subfield_vector::<F, E, D>(
        &weights,
        AkitaError::InvalidInput(
            "root opening point does not encode in the ring-subfield basis".to_string(),
        ),
    )?;
    recover_ring_subfield_inner_product::<F, E, D>(folded_ring, &packed_inner_point)
}
