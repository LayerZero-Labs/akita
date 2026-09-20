//! Ring-relation prover for the Akita PCS (§4.2).
//!
//! Coordinates public relation messages while private witness storage stays in the backend.
use crate::backend::OperationCtx;
use crate::ProverOpeningData;
#[cfg(test)]
use akita_challenges::Challenges;
use akita_error::AkitaError;
use akita_transcript::labels::ABSORB_OPENING_PAYLOAD;
use akita_types::dispatch_for_field;
#[cfg(test)]
use akita_types::RingRelationGroupOpening;
use akita_types::RingVec;
use akita_types::{
    CommittedGroupParams, OpeningFamily, RingRelationInstance, SignedDigitKernel, MAX_I8_LOG_BASIS,
};
use jolt_field::Unreduced;
use jolt_field::{CanonicalEncoding, Field, Ring};

use super::fold_grind;
use super::prove::PreparedGroupOpening;

use crate::backend::PreparedRelationGroupPublic;

#[inline]
fn validate_i8_setup_log_basis(log_basis: u32, context: &str) -> Result<(), AkitaError> {
    if SignedDigitKernel::for_log_basis(log_basis) == Some(SignedDigitKernel::I8) {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(format!(
            "log_basis must be in 1..={MAX_I8_LOG_BASIS} {context}"
        )))
    }
}

pub(crate) struct PreparedRingRelation<F: Field + CanonicalEncoding, E: Field, WitnessHandle> {
    pub(crate) instance: RingRelationInstance<F>,
    pub(crate) witness_handle: WitnessHandle,
    pub(crate) groups: Vec<PreparedRelationGroupPublic<F, E>>,
}

pub(in crate::protocol) struct PreparedRingRelationOutput<
    F: Field + CanonicalEncoding,
    E: Field,
    WitnessHandle,
> {
    pub(in crate::protocol) relation: PreparedRingRelation<F, E, WitnessHandle>,
    pub(in crate::protocol) opening_payload: RingVec<F>,
    pub(in crate::protocol) trace_claim: crate::protocol::prove::PreparedEvaluationTraceClaim<E>,
    pub(in crate::protocol) row_coefficients: Vec<E>,
}

pub fn validate_prepared_relation_groups<F, E>(
    groups: &[PreparedRelationGroupPublic<F, E>],
    level_params: &CommittedGroupParams,
    opening_batch: &akita_types::OpeningClaimsLayout,
    relation: &RingRelationInstance<F>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    E: akita_types::FpExtEncoding<F>,
{
    if groups.len() != opening_batch.num_groups()
        || relation.opening_batch() != opening_batch
        || relation.group_openings().len() != groups.len()
        || relation.extension_degree() != E::DEGREE
    {
        return Err(AkitaError::InvalidSetup(
            "prepared Stage 2 groups disagree with the relation batch".into(),
        ));
    }
    let geometry =
        akita_types::RelationWitnessGeometry::for_level(level_params, opening_batch, E::DEGREE)?;
    for (group_index, group) in groups.iter().enumerate() {
        let layout = opening_batch.group_layout(group_index)?;
        let group_params = level_params.group_params_geometry(opening_batch, group_index)?;
        if group.scalar_openings().len() != layout.num_polynomials() {
            return Err(AkitaError::InvalidSize {
                expected: layout.num_polynomials(),
                actual: group.scalar_openings().len(),
            });
        }
        match (
            geometry.group_opening_method(group_index)?,
            group.kind(),
            relation.group_openings()[group_index].coefficient_packing_geometry(),
        ) {
            (
                akita_types::OpeningMethod::EvaluationTrace,
                OpeningFamily::EvaluationTrace(point),
                None,
            ) if relation.group_ring_multiplier_point(group_index)?
                == &point.ring_multiplier_point => {}
            (
                akita_types::OpeningMethod::SubringCoefficientPacking { .. },
                OpeningFamily::SubringCoefficientPacking(point),
                Some(relation_geometry),
            ) if relation_geometry == point.geometry()
                && point.source_num_vars() == layout.num_vars()
                && point.num_live_positions()
                    == group_params.num_live_ring_elements_per_claim()
                && point.num_positions_per_block() == group_params.num_positions_per_block()
                && point.num_live_blocks() == group_params.num_live_blocks() => {}
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "prepared Stage 2 group method or point disagrees with its relation".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Validate the chunked-witness configuration at the prover boundary (no-panic
/// contract), before any witness math. Mirrors the planner entry guard and the
/// verifier layout resolution.
pub fn validate_chunked_witness_cfg(lp: &CommittedGroupParams) -> Result<(), AkitaError> {
    lp.witness_chunk.validate()
}

/// Prover-side builder for the ring relation $M(x) \cdot z = y(x) + (X^D + 1) \cdot r(x)$.
pub struct RingRelationProver;

impl RingRelationProver {
    /// Prepare the relation for one or more group-local opening points and
    /// polynomial slots, preserving payload-before-claim transcript order.
    ///
    /// Private arithmetic stays inside the backend; only scheduled public
    /// relation messages and opaque successor handles cross this boundary.
    ///
    /// # Errors
    ///
    /// Rejects inconsistent public geometry, malformed backend messages, and
    /// fold responses that fail the scheduled admission policy.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[allow(private_bounds)]
    #[tracing::instrument(skip_all, name = "RingRelationProver::prepare")]
    #[inline(never)]
    pub(in crate::protocol) fn prepare<'claims, 'source, F, PointF, T, B>(
        opening_ctx: &OperationCtx<'_, F, B>,
        prepared_group_openings: Vec<PreparedGroupOpening<PointF, B::PreparedOpeningHandle>>,
        commitment_material: Vec<B::CommitmentMaterialHandle>,
        block_claims: &ProverOpeningData<
            'claims,
            PointF,
            crate::backend::OpeningSource<'source, B::CommitmentHandle, B::WitnessHandle>,
            F,
        >,
        lp: CommittedGroupParams,
        transcript: &mut T,
        level: u32,
        reduction: &Option<crate::protocol::prove::ExtensionOpeningReduction<PointF>>,
        scalar_openings: &[PointF],
        trace_opening_batch: &akita_types::OpeningClaimsLayout,
        expected_witness_len: usize,
        commitment_ring_dimension: usize,
    ) -> Result<PreparedRingRelationOutput<F, PointF, B::WitnessHandle>, AkitaError>
    where
        F: Field
            + CanonicalEncoding
            + akita_serialization::AkitaSerialize
            + Ring
            + Unreduced
            + 'static,
        <F as Unreduced>::Wide: From<F>,
        PointF: akita_types::FpExtEncoding<F>
            + jolt_field::ExtField<F>
            + akita_serialization::AkitaSerialize,
        T: akita_types::ProverTranscriptGrinding<F>,
        B: crate::backend::ProverBackend<F, PointF>,
    {
        let consumer = opening_ctx.backend();
        let prepared_opening_handles = prepared_group_openings
            .into_iter()
            .map(PreparedGroupOpening::into_opening_handle)
            .collect::<Vec<_>>();
        let prepare_span = tracing::info_span!("ring_relation_prepare_inputs").entered();
        validate_i8_setup_log_basis(
            lp.open().digits.log_basis,
            "for i8 prover opening decomposition",
        )?;
        validate_chunked_witness_cfg(&lp)?;
        let dims = lp.role_dims();
        let opening_batch = block_claims.opening_layout().clone();
        let num_groups = block_claims.opening_claims().num_groups();
        if prepared_opening_handles.len() != num_groups {
            return Err(AkitaError::InvalidInput(
                "ring relation prover prepared group count mismatch".to_string(),
            ));
        }
        if commitment_material.len() != num_groups {
            return Err(AkitaError::InvalidInput(
                "prepared commitment material group count mismatch".into(),
            ));
        }
        let relation_geometry =
            akita_types::RelationWitnessGeometry::for_level(&lp, &opening_batch, PointF::DEGREE)?;
        let relation_rhs_layout = relation_geometry.rhs_layout();
        let group_commitments = (0..num_groups)
            .map(|group_index| {
                block_claims
                    .opening_claims()
                    .group_commitment(group_index)
                    .map(|commitment| commitment.rows().clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let num_claims = opening_batch.num_total_polynomials();
        if num_claims == 0 {
            return Err(AkitaError::InvalidInput(
                "batched prover requires at least one polynomial".to_string(),
            ));
        }
        drop(prepare_span);

        // Bind only the scheduled opening payload; uncompressed private rows
        // stay in the backend's build state.
        let opening_rows_span = tracing::info_span!(
            "ring_relation_opening_rows",
            groups = num_groups,
            claims = num_claims,
            d_d = dims.d_d(),
        )
        .entered();
        let build_start =
            crate::backend::OpaqueRecursiveWitnessBuildKernel::begin_recursive_witness(
                consumer,
                opening_ctx.proof_context(),
                &prepared_opening_handles,
                commitment_material,
                &lp,
                &opening_batch,
                relation_rhs_layout,
                &group_commitments,
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!("recursive witness assembly failed: {err:?}"))
            })?;
        let (
            prepared_relation_groups,
            opening_payload,
            opening_payload_ring_dimension,
            build_handle,
        ) = build_start.into_parts();
        let payload_geometry = relation_rhs_layout.opening_payload_geometry()?;
        if opening_payload.coeff_len() != payload_geometry.transmitted_coefficients()
            || opening_payload_ring_dimension != payload_geometry.transcript_ring_dimension()
            || prepared_relation_groups.len() != num_groups
            || prepared_relation_groups
                .iter()
                .flat_map(|group| group.scalar_openings())
                .copied()
                .collect::<Vec<_>>()
                != scalar_openings
        {
            return Err(AkitaError::InvalidProof);
        }
        opening_payload.append_flat_to_transcript(
            ABSORB_OPENING_PAYLOAD,
            opening_payload_ring_dimension,
            transcript,
        )?;
        drop(opening_rows_span);

        // Native public claim batching is intentionally delayed until every
        // opening digit has been bound through the complete D/H payload above.
        // Extension EOR supplies its already-bound coefficients because its
        // shared reduced point and final relation depend on that earlier batch.
        let (trace_claim, row_coefficients) =
            crate::protocol::prove::prepare_evaluation_trace_claim::<F, PointF, T>(
                reduction,
                scalar_openings,
                trace_opening_batch,
                transcript,
                level,
            )
            .map_err(|err| {
                AkitaError::InvalidInput(format!("prepare evaluation-trace claim failed: {err:?}"))
            })?;
        let row_coefficient_rings = dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            lp.role_dims().d_a(),
            |D| {
                let rings = crate::protocol::prove::row_coefficient_rings::<F, PointF, D>(
                    &row_coefficients,
                )
                .map_err(|err| {
                    AkitaError::InvalidInput(format!("row coefficient rings failed: {err:?}"))
                })?;
                Ok::<_, AkitaError>(RingVec::from_ring_elems(&rings))
            }
        )
        .map_err(|err| {
            AkitaError::InvalidInput(format!("root row-coefficient preparation failed: {err:?}"))
        })?;
        if !row_coefficient_rings.can_decode_vec(dims.d_a())
            || row_coefficient_rings.coeff_len() / dims.d_a() != num_claims
        {
            return Err(AkitaError::InvalidInput(
                "batched prover row coefficient length does not match claim count".to_string(),
            ));
        }
        let gamma = row_coefficient_rings
            .coeffs()
            .iter()
            .copied()
            .step_by(dims.d_a())
            .collect::<Vec<_>>();

        // Distributed-prover chunked layout: the grind emits one folded response
        // per block window (`z_i`), and the global response is their sum
        // (`Σ_i z_i = z`, exact coefficient-wise i32 accumulation).
        let fold_grind_span = tracing::info_span!(
            "ring_relation_fold_grind",
            groups = num_groups,
            claims = num_claims,
        )
        .entered();
        let grind_groups = (0..num_groups)
            .map(|group_index| {
                Ok(fold_grind::FoldGrindGroup {
                    group_index,
                    opening: prepared_opening_handles
                        .get(group_index)
                        .ok_or(AkitaError::InvalidProof)?,
                    num_polynomials: opening_batch.group_layout(group_index)?.num_polynomials(),
                    params: lp.group_params_geometry(&opening_batch, group_index)?,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let _grind_span = tracing::info_span!("fold_grind_sample").entered();
        let grind_outputs =
            fold_grind::sample_multi_group_fold_decompose_witnesses::<F, PointF, B, T>(
                opening_ctx,
                transcript,
                level,
                &lp,
                &opening_batch,
                &grind_groups,
                None,
            )
            .map_err(|err| AkitaError::InvalidInput(format!("fold grind failed: {err:?}")))?;
        drop(_grind_span);
        if grind_outputs.len() != num_groups {
            return Err(AkitaError::InvalidProof);
        }
        let folds = grind_outputs
            .into_iter()
            .enumerate()
            .map(|(group_index, output)| {
                let group = lp.group_params_geometry(&opening_batch, group_index)?;
                let expected_coefficients = group
                    .num_positions_per_block()
                    .checked_mul(group.num_digits_inner())
                    .and_then(|width| {
                        width.checked_mul(group.profile.inner.matrix.ring_dimension())
                    })
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("accepted-fold response width overflow".into())
                    })?;
                let metadata = crate::backend::AcceptedFoldHandle::metadata(&output.fold_handle);
                if metadata.ring_dimension() != group.profile.inner.matrix.ring_dimension()
                    || metadata.response_coordinate_count() != expected_coefficients
                    || metadata.num_chunks() != lp.witness_chunk.num_chunks
                {
                    return Err(AkitaError::InvalidInput(format!(
                        "accepted-fold metadata disagrees with planned group {group_index} geometry"
                    )));
                }
                Ok(crate::backend::RecursiveWitnessFoldInput::new(
                    output.fold_handle,
                    output.challenges,
                ))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let group_openings = folds
            .iter()
            .zip(&prepared_relation_groups)
            .map(|(fold, group)| match (fold.challenges(), group.kind()) {
                (
                    OpeningFamily::EvaluationTrace(challenges),
                    OpeningFamily::EvaluationTrace(point),
                ) => Ok(akita_types::RingRelationGroupOpening::evaluation_trace(
                    challenges.clone(),
                    point.ring_multiplier_point.clone(),
                )),
                (
                    OpeningFamily::SubringCoefficientPacking(challenges),
                    OpeningFamily::SubringCoefficientPacking(point),
                ) if challenges.geometry() == point.geometry() => Ok(
                    akita_types::RingRelationGroupOpening::coefficient_packing(challenges.clone()),
                ),
                _ => Err(AkitaError::InvalidProof),
            })
            .collect::<Result<Vec<_>, _>>()?;
        // The relation places the final group before the precommitted groups;
        // statement claims and backend opening handles retain their original order.
        let relation_commitments = relation_rhs_layout
            .groups
            .iter()
            .map(|group| {
                group_commitments
                    .get(group.group_index)
                    .ok_or(AkitaError::InvalidProof)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let relation_rhs = if lp.payload_mode.is_compressed() {
            let payloads = relation_commitments
                .iter()
                .map(|rows| rows.coeffs())
                .collect::<Vec<_>>();
            akita_types::assemble_compressed_relation_rhs(
                relation_rhs_layout,
                &payloads,
                opening_payload.coeffs(),
            )?
        } else {
            let rows = RingVec::from_coeffs(
                relation_commitments
                    .iter()
                    .flat_map(|row| row.coeffs())
                    .copied()
                    .collect(),
            );
            akita_types::assemble_relation_rhs(relation_rhs_layout, &opening_payload, &rows)?
        };
        let instance = RingRelationInstance::new(
            group_openings,
            PointF::DEGREE,
            opening_batch.clone(),
            gamma.clone(),
            row_coefficient_rings.clone(),
            relation_rhs,
            dims,
        )?;
        validate_prepared_relation_groups(
            &prepared_relation_groups,
            &lp,
            &opening_batch,
            &instance,
        )?;
        let witness_plan = crate::backend::ValidatedRecursiveWitnessPlan::new(
            expected_witness_len,
            commitment_ring_dimension,
        );
        let witness_handle =
            crate::backend::OpaqueRecursiveWitnessBuildKernel::finish_recursive_witness(
                consumer,
                build_handle,
                folds,
                crate::backend::RecursiveWitnessPublicInputs {
                    extension_degree: PointF::DEGREE,
                    gamma: &gamma,
                    row_coefficient_rings: &row_coefficient_rings,
                },
                &witness_plan,
            )?;
        drop(fold_grind_span);

        let manifest = crate::backend::RecursiveWitnessHandle::manifest(&witness_handle);
        if manifest.logical_len() != expected_witness_len
            || manifest.commitment_ring_dimension() != commitment_ring_dimension
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(PreparedRingRelationOutput {
            relation: PreparedRingRelation {
                instance,
                witness_handle,
                groups: prepared_relation_groups,
            },
            opening_payload,
            trace_claim,
            row_coefficients,
        })
    }
}

#[cfg(test)]
#[path = "ring_relation_tests.rs"]
mod prepared_group_tests;
