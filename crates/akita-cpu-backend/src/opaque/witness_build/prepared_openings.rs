use super::coefficient_packing;
use super::d_rows;
use super::finalize::{RelationDQuotientWitness, RingRelationGroupWitness};
use crate::opaque::OperationCtx;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_error::AkitaError;
use akita_types::{CommittedGroupParams, DigitBlocks, RingVec};
use jolt_field::{CanonicalEncoding, Field};

enum ConsumerOpeningKind<F: Field> {
    EvaluationTrace {
        e_folded: RingVec<F>,
    },
    CoefficientPacking {
        partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
    },
}

/// Opaque consumer state joining prepared opening rows to their E digits.
pub(crate) struct PreparedOpeningWitness<F: Field> {
    e_hat: DigitBlocks,
    kind: ConsumerOpeningKind<F>,
}

fn decompose_opening_rows<F: Field + CanonicalEncoding, const D: usize>(
    pre_folded_e: &[&[akita_algebra::CyclotomicRing<F, D>]],
    role_subcolumns: usize,
    depth_open: usize,
    log_basis: u32,
) -> Result<DigitBlocks, AkitaError> {
    let q = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let params = BalancedDecomposePow2Params::new(depth_open, log_basis, q);
    let total_rows: usize = pre_folded_e.iter().map(|rows| rows.len()).sum();
    if role_subcolumns == 0 || !total_rows.is_multiple_of(role_subcolumns) {
        return Err(AkitaError::InvalidSetup(
            "E rows do not form complete native-role subcolumn groups".into(),
        ));
    }
    let planes_per_semantic = role_subcolumns
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("E digit block width overflow".into()))?;
    let mut e_hat =
        DigitBlocks::zeroed(vec![planes_per_semantic; total_rows / role_subcolumns], D)?;
    let mut offset = 0usize;
    for folded_rows in pre_folded_e {
        for row in *folded_rows {
            row.balanced_decompose_pow2_i8_into_with_params(
                &mut e_hat.typed_planes_mut::<D>()?[offset..offset + depth_open],
                &params,
            );
            offset += depth_open;
        }
    }
    Ok(e_hat)
}

impl<F: Field + CanonicalEncoding> PreparedOpeningWitness<F> {
    pub(crate) fn evaluation_trace<const D: usize>(
        folded_by_claim: &[RingVec<F>],
        role_subcolumns: usize,
        depth_open: usize,
        log_basis: u32,
    ) -> Result<Self, AkitaError> {
        let typed = folded_by_claim
            .iter()
            .map(RingVec::as_ring_slice::<D>)
            .collect::<Result<Vec<_>, _>>()?;
        let e_hat = decompose_opening_rows::<F, D>(&typed, role_subcolumns, depth_open, log_basis)?;
        let e_folded = RingVec::from_coeffs(
            folded_by_claim
                .iter()
                .flat_map(|block| block.coeffs().iter().copied())
                .collect(),
        );
        Ok(Self {
            e_hat,
            kind: ConsumerOpeningKind::EvaluationTrace { e_folded },
        })
    }

    pub(crate) fn coefficient_packing<const D: usize>(
        level: &CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        geometry: &akita_types::RelationWitnessGeometry,
        group_index: usize,
        partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
    ) -> Result<Self, AkitaError> {
        let e_hat = coefficient_packing::materialize_coefficient_packing_d_input::<F, D>(
            level,
            opening_batch,
            geometry,
            group_index,
            &partials_by_claim,
        )?;
        Ok(Self {
            e_hat,
            kind: ConsumerOpeningKind::CoefficientPacking { partials_by_claim },
        })
    }

    fn e_hat(&self) -> &DigitBlocks {
        &self.e_hat
    }

    pub(crate) fn into_relation_witness(
        self,
        fold: crate::opaque::CpuAcceptedFold<F>,
        challenges: akita_types::GroupFoldChallenges,
        inner_relation: crate::opaque::OpaqueInnerRelationState<F>,
        role_dims: akita_types::CommitmentRingDims,
    ) -> Result<RingRelationGroupWitness<F>, AkitaError> {
        match (self.kind, challenges) {
            (
                ConsumerOpeningKind::EvaluationTrace { e_folded },
                akita_types::OpeningFamily::EvaluationTrace(_),
            ) => Ok(RingRelationGroupWitness::from_parts(
                fold,
                self.e_hat,
                e_folded,
                inner_relation,
                role_dims,
            )),
            (
                ConsumerOpeningKind::CoefficientPacking { partials_by_claim },
                akita_types::OpeningFamily::SubringCoefficientPacking(challenges),
            ) => {
                let product = coefficient_packing::fold_coefficient_packing_group(
                    challenges.geometry(),
                    &partials_by_claim,
                    challenges.canonical(),
                )?;
                Ok(RingRelationGroupWitness::from_coefficient_packing_parts(
                    fold,
                    self.e_hat,
                    product,
                    inner_relation,
                    role_dims,
                ))
            }
            _ => Err(AkitaError::InvalidSetup(
                "prepared opening material and fold challenges disagree".into(),
            )),
        }
    }
}

pub(crate) type PublicPreparedRelationOpening<F, E> = akita_types::OpeningFamily<
    akita_types::PreparedOpeningPoint<F, E>,
    akita_types::PreparedSubringCoefficientPackingPoint<E>,
>;

type PreparedGroupWitnessOutput<F, E> = (
    PreparedOpeningWitness<F>,
    PublicPreparedRelationOpening<F, E>,
    Vec<E>,
);

pub(super) fn prepare_group_opening_witness<F, E, Cfg, const D: usize>(
    handle: &crate::opaque::CpuPreparedOpeningHandle<F, E, Cfg>,
    level: &CommittedGroupParams,
    opening_batch: &akita_types::OpeningClaimsLayout,
    geometry: &akita_types::RelationWitnessGeometry,
    group_index: usize,
    group_dims: akita_types::CommitmentRingDims,
) -> Result<PreparedGroupWitnessOutput<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: jolt_field::ExtField<F>,
    Cfg: akita_config::CommitmentConfig<Field = F>,
{
    let (opening, public) =
        handle.relation_opening::<D>(level, opening_batch, geometry, group_index, group_dims)?;
    Ok((opening, public, handle.scalar_openings().to_vec()))
}

pub(super) fn prepare_opening_relation_rows<F, RB, const D: usize>(
    ring_switch_ctx: &OperationCtx<'_, F, RB>,
    opening_batch: &akita_types::OpeningClaimsLayout,
    openings: &[PreparedOpeningWitness<F>],
    has_preceding_groups: bool,
    d_row_len: usize,
    log_basis: u32,
    relation_mode: akita_types::RingRelationMode,
) -> Result<(RingVec<F>, RelationDQuotientWitness<F>), AkitaError>
where
    F: Field + CanonicalEncoding,
    RB: crate::opaque::RingSwitchProveBackend<F, D> + crate::opaque::DigitRowsComputeBackend<F>,
{
    let concatenated = has_preceding_groups
        .then(|| {
            coefficient_packing::concatenate_group_d_inputs(
                opening_batch,
                &openings
                    .iter()
                    .map(PreparedOpeningWitness::e_hat)
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()?;
    let e_hat = concatenated
        .as_ref()
        .or_else(|| openings.first().map(PreparedOpeningWitness::e_hat))
        .ok_or(AkitaError::InvalidProof)?;
    if d_row_len == 0 {
        let quotients = match relation_mode {
            akita_types::RingRelationMode::QuotientLift => {
                RelationDQuotientWitness::QuotientLift(RingVec::from_coeffs(Vec::new()))
            }
            akita_types::RingRelationMode::ReducedEvaluation => {
                RelationDQuotientWitness::ReducedEvaluation
            }
        };
        return Ok((RingVec::from_coeffs(Vec::new()), quotients));
    }
    match d_rows::compute_relation_d_rows::<F, RB, D>(
        ring_switch_ctx,
        d_row_len,
        log_basis,
        e_hat,
        relation_mode,
    )? {
        d_rows::RelationDRows::QuotientLift { reduced, quotients } => Ok((
            RingVec::from_ring_elems(&reduced),
            RelationDQuotientWitness::QuotientLift(RingVec::from_ring_elems(&quotients)),
        )),
        d_rows::RelationDRows::ReducedEvaluation { reduced } => Ok((
            RingVec::from_ring_elems(&reduced),
            RelationDQuotientWitness::ReducedEvaluation,
        )),
    }
}
