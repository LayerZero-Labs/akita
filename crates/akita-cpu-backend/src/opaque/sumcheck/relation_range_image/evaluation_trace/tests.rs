use super::*;

use akita_algebra::{poly::multilinear_eval, CyclotomicRing};
use akita_config::proof_optimized::fp128;
use akita_params::{
    basis_weights_prefix, r_decomp_levels, ring_opening_point_from_field, BasisMode,
    CommittedGroupParams, DigitRangePlan, OpeningClaimsLayout, RelationAddressGeometry,
    SisModulusProfileId, WitnessLayout,
};
use akita_types::{
    FpExtEncoding, PreparedOpeningPoint, RelationRangeImagePlan, RingMultiplierOpeningPoint,
};
use jolt_field::{Ext2, ExtField, One, Ring, Zero};

type F = fp128::Field;
const D: usize = 128;
const NUM_VARIABLES: usize = 16;

// Test-only constructors and readback for the prepared linear terms.
impl<E: Field> PreparedProverLinearTerms<E> {
    pub(crate) fn zero(live_lane_count: usize, coeff_count: usize) -> Self {
        Self {
            lane_weights: PreparedLaneWeights::Sparse(vec![Vec::new(); live_lane_count]),
            sources: Vec::new(),
            live_lane_count,
            coeff_count,
        }
    }

    pub(crate) fn from_dense(dense: Vec<E>, live_lane_count: usize, coeff_count: usize) -> Self {
        assert_eq!(dense.len(), live_lane_count * coeff_count);
        let mut lane_terms = vec![Vec::new(); live_lane_count];
        let sources = dense
            .chunks_exact(coeff_count)
            .enumerate()
            .map(|(lane, values)| {
                lane_terms[lane].push(PreparedLaneTerm {
                    factor: E::one(),
                    source_index: lane,
                    lane: 0,
                });
                PreparedTraceSource {
                    values: values.to_vec(),
                    lane_count: 1,
                }
            })
            .collect();
        Self {
            lane_weights: PreparedLaneWeights::Sparse(lane_terms),
            sources,
            live_lane_count,
            coeff_count,
        }
    }

    /// Compile arbitrary checked source segments into the shared Stage 2 engine.
    pub(crate) fn from_structured_weights(
        weights: &StructuredLinearWeights<E>,
        coeff_count: usize,
    ) -> Result<Self, AkitaError> {
        if coeff_count == 0
            || !coeff_count.is_power_of_two()
            || weights.physical_field_len == 0
            || !weights.physical_field_len.is_multiple_of(coeff_count)
            || weights.sources.is_empty()
            || weights.terms.is_empty()
        {
            return Err(AkitaError::InvalidSetup(
                "structured linear common-coordinate geometry is malformed".into(),
            ));
        }
        let live_lane_count = weights.physical_field_len / coeff_count;
        let sources = weights
            .sources
            .iter()
            .map(|source| {
                if source.is_empty() || !source.len().is_multiple_of(coeff_count) {
                    return Err(AkitaError::InvalidSetup(
                        "structured linear source geometry is malformed".into(),
                    ));
                }
                Ok(PreparedTraceSource {
                    values: source.as_ref().to_vec(),
                    lane_count: source.len() / coeff_count,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let mut lane_terms = vec![Vec::new(); live_lane_count];
        for term in &weights.terms {
            let source = weights
                .sources
                .get(term.source_index)
                .ok_or(AkitaError::InvalidProof)?;
            let segments = weights
                .segments
                .get(term.segment_range.clone())
                .ok_or(AkitaError::InvalidProof)?;
            if segments.is_empty() {
                return Err(AkitaError::InvalidSetup(
                    "structured linear source geometry is malformed".into(),
                ));
            }
            let source_lane_count = source.len() / coeff_count;
            for segment in segments {
                let target_end = segment
                    .physical_coefficient_start
                    .checked_add(segment.coefficient_count)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("structured linear target range overflow".into())
                    })?;
                let source_end = segment
                    .source_coefficient_start
                    .checked_add(segment.coefficient_count)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("structured linear source range overflow".into())
                    })?;
                if segment.coefficient_count == 0
                    || !segment.coefficient_count.is_multiple_of(coeff_count)
                    || !segment
                        .physical_coefficient_start
                        .is_multiple_of(coeff_count)
                    || !segment.source_coefficient_start.is_multiple_of(coeff_count)
                    || target_end > weights.physical_field_len
                    || source_end > source.len()
                {
                    return Err(AkitaError::InvalidSetup(
                        "structured linear segment is unaligned or out of bounds".into(),
                    ));
                }
                let target_lane_start = segment.physical_coefficient_start / coeff_count;
                let source_lane_start = segment.source_coefficient_start / coeff_count;
                let lane_count = segment.coefficient_count / coeff_count;
                for lane_offset in 0..lane_count {
                    let target_lane =
                        target_lane_start.checked_add(lane_offset).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "structured linear target lane overflow".into(),
                            )
                        })?;
                    let source_lane =
                        source_lane_start.checked_add(lane_offset).ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "structured linear source lane overflow".into(),
                            )
                        })?;
                    if source_lane >= source_lane_count {
                        return Err(AkitaError::InvalidProof);
                    }
                    lane_terms
                        .get_mut(target_lane)
                        .ok_or(AkitaError::InvalidProof)?
                        .push(PreparedLaneTerm {
                            factor: term.factor,
                            source_index: term.source_index,
                            lane: source_lane,
                        });
                }
            }
        }
        Ok(Self {
            lane_weights: PreparedLaneWeights::Sparse(lane_terms),
            sources,
            live_lane_count,
            coeff_count,
        })
    }

    pub(crate) fn materialize_dense(&self) -> Vec<E> {
        (0..self.live_lane_count)
            .flat_map(|lane| {
                (0..self.coeff_count)
                    .map(move |coefficient| self.get(lane, coefficient, self.coeff_count))
            })
            .collect()
    }
}

fn fold_prepared_trace_at_point<E: Field>(
    mut trace: PreparedProverLinearTerms<E>,
    live_len: usize,
    coeff_count: usize,
    point: &[E],
) -> E {
    let coefficient_bits = coeff_count.trailing_zeros() as usize;
    for &challenge in &point[..coefficient_bits] {
        trace.fold_coefficients(challenge);
    }
    let lane_point = &point[coefficient_bits..];
    let mut lanes = vec![E::zero(); 1 << lane_point.len()];
    trace.drain_into_lane_weights(&mut lanes[..live_len / coeff_count], E::one());
    multilinear_eval(&lanes, lane_point).unwrap()
}

fn materialize_semantic_trace_oracle<E: Field>(
    weights: &EvaluationTraceWeights<E>,
    output_scale: E,
) -> Vec<E> {
    let mut table = vec![E::zero(); weights.physical_field_len];
    for term in &weights.terms {
        let block_weights = basis_weights_prefix(
            &term.block_opening_point,
            term.basis,
            term.group_block_count,
        )
        .unwrap();
        let digit_count = term.opening_digit_weights.len();
        let block_stride = digit_count * term.source_ring_dimension;
        let role_subcolumns = term.source_ring_dimension / term.opening_ring_dimension;
        for segment in &term.segments {
            for local_block in 0..segment.block_count {
                let global_block = segment.global_block_start + local_block;
                let block_start = segment.physical_coefficient_start + local_block * block_stride;
                for role_subcolumn in 0..role_subcolumns {
                    let source_start = role_subcolumn * term.opening_ring_dimension;
                    for (digit, &digit_weight) in term.opening_digit_weights.iter().enumerate() {
                        let digit_start = block_start
                            + (role_subcolumn * digit_count + digit) * term.opening_ring_dimension;
                        let factor = output_scale
                            * term.coefficient
                            * block_weights[global_block]
                            * digit_weight;
                        for role_coefficient in 0..term.opening_ring_dimension {
                            table[digit_start + role_coefficient] +=
                                factor * term.inner_trace[source_start + role_coefficient];
                        }
                    }
                }
            }
        }
    }
    table
}

fn assert_prepared_opening_support_matches_semantic_trace<E>(basis: BasisMode)
where
    E: FpExtEncoding<F> + ExtField<F> + Ring,
{
    let opening_batch = OpeningClaimsLayout::new(NUM_VARIABLES, 2).unwrap();
    let level_params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        D,
        3,
        2,
        4,
        3,
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(D)
            .expect("D128 challenge"),
    )
    .with_decomp(64, (1usize << NUM_VARIABLES) / D, 2, 2, 2)
    .expect("local EvaluationTrace geometry");
    let relation_witness_geometry =
        akita_params::RelationWitnessGeometry::for_evaluation_trace_execution(
            &level_params,
            &opening_batch,
        )
        .unwrap();
    let witness_layout = WitnessLayout::new(
        &level_params,
        &opening_batch,
        &relation_witness_geometry,
        2,
        akita_params::RelationQuotientPlan::quotient_lift(r_decomp_levels::<F>(
            level_params.open().digits.log_basis,
        ))
        .unwrap(),
    )
    .unwrap();
    let live_len = witness_layout.live_coeff_len();
    let relation_address_geometry =
        RelationAddressGeometry::new(level_params.role_dims(), D, live_len).unwrap();
    let common_coefficient_count = relation_address_geometry.relation_coefficient_block_len();
    let plan = RelationRangeImagePlan::new(
        relation_witness_geometry,
        relation_address_geometry,
        DigitRangePlan::new(1usize << level_params.open().digits.log_basis).unwrap(),
        witness_layout,
        &opening_batch,
    )
    .unwrap();
    let digit_witness_domain = plan.digit_witness_domain();
    let group_params = level_params.group_params(&opening_batch, 0).unwrap();
    let base_outer_point =
        vec![F::zero(); group_params.position_index_bits() + group_params.block_index_bits()];
    let ring_opening_point = ring_opening_point_from_field(
        &base_outer_point,
        group_params.num_positions_per_block(),
        group_params.num_live_blocks(),
        basis,
    )
    .unwrap();
    let padded_point = (0..NUM_VARIABLES)
        .map(|index| E::from_u64(17 + 2 * index as u64))
        .collect();
    let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
    let prepared_point = akita_params::dispatch_for_field!(
        akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
        F,
        group_params.inner_commit_matrix_params().ring_dimension(),
        |D_G| {
            Ok::<_, akita_error::AkitaError>(PreparedOpeningPoint::from_parts(
                padded_point,
                ring_multiplier_point,
                CyclotomicRing::<F, D_G>::one(),
            ))
        }
    )
    .unwrap();
    let prepared_points = vec![prepared_point];
    let claim_coefficients = vec![E::from_u64(41), E::from_u64(43)];
    let semantic_trace = build_evaluation_trace_weights(
        crate::opaque::EvaluationTraceDescription::try_new::<F>(
            plan.digit_witness_domain(),
            plan.relation_address_geometry()
                .relation_coefficient_block_len(),
            plan.witness_layout(),
            &level_params,
            &opening_batch,
            &prepared_points,
            &claim_coefficients,
            basis,
        )
        .unwrap(),
    )
    .unwrap();
    let output_scale = E::from_u64(47);
    let point = (0..digit_witness_domain.num_vars())
        .map(|index| E::from_u64(53 + 2 * index as u64))
        .collect::<Vec<_>>();
    let expected_table = materialize_semantic_trace_oracle(&semantic_trace, output_scale);
    let mut padded_expected_table = expected_table.clone();
    padded_expected_table.resize(1usize << point.len(), E::zero());

    for coeff_count in [
        common_coefficient_count,
        common_coefficient_count / 2,
        common_coefficient_count / 4,
    ] {
        let prepared = PreparedProverLinearTerms::from_evaluation_trace(
            &semantic_trace,
            coeff_count,
            output_scale,
        )
        .unwrap();
        assert_eq!(prepared.materialize_dense(), expected_table,);
        let folded = fold_prepared_trace_at_point(prepared, live_len, coeff_count, &point);
        assert_eq!(
            folded,
            multilinear_eval(&padded_expected_table, &point).unwrap()
        );
    }
    for malformed_common_count in [0, 3, common_coefficient_count * 2] {
        assert!(PreparedProverLinearTerms::from_evaluation_trace(
            &semantic_trace,
            malformed_common_count,
            output_scale,
        )
        .is_err());
    }
}

#[test]
fn projected_semantic_trace_oracle_uses_role_native_subcolumns() {
    let weights = EvaluationTraceWeights {
        terms: vec![EvaluationTraceTerm {
            coefficient: F::from_u64(3),
            block_opening_point: vec![F::from_u64(5), F::from_u64(7)].into(),
            basis: BasisMode::Lagrange,
            group_block_count: 3,
            source_ring_dimension: 8,
            opening_ring_dimension: 4,
            coefficient_block_len: 2,
            opening_digit_weights: vec![F::from_u64(11), F::from_u64(13)].into(),
            inner_trace: (0..8)
                .map(|index| F::from_u64(17 + index as u64))
                .collect::<Vec<_>>()
                .into(),
            segments: vec![EvaluationTraceSegment {
                physical_coefficient_start: 8,
                global_block_start: 1,
                block_count: 2,
            }],
        }],
        physical_field_len: 64,
        num_vars: 6,
    };
    let output_scale = F::from_u64(29);
    let expected = materialize_semantic_trace_oracle(&weights, output_scale);
    let point = (0..weights.num_vars)
        .map(|index| F::from_u64(31 + index as u64))
        .collect::<Vec<_>>();

    for coeff_count in [2, 4] {
        let prepared =
            PreparedProverLinearTerms::from_evaluation_trace(&weights, coeff_count, output_scale)
                .expect("projected trace geometry");
        assert_eq!(prepared.materialize_dense(), expected);
        assert_eq!(
            fold_prepared_trace_at_point(prepared, expected.len(), coeff_count, &point),
            multilinear_eval(&expected, &point).unwrap()
        );
    }
}

#[test]
fn prepared_opening_support_matches_semantic_trace_across_bases_and_extension() {
    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        assert_prepared_opening_support_matches_semantic_trace::<F>(basis);
        assert_prepared_opening_support_matches_semantic_trace::<Ext2<F>>(basis);
    }
}

#[test]
fn coefficient_folds_reuse_prepared_source_buffers() {
    let coeff_count = 8;
    let live_lane_count = 2;
    let dense = (1..=live_lane_count * coeff_count)
        .map(|value| F::from_u64(value as u64))
        .collect::<Vec<_>>();
    let r0 = F::from_u64(37);
    let r1 = F::from_u64(41);

    let mut one_round =
        PreparedProverLinearTerms::from_dense(dense.clone(), live_lane_count, coeff_count);
    let one_round_allocations = one_round
        .sources
        .iter()
        .map(|source| (source.values.as_ptr(), source.values.capacity()))
        .collect::<Vec<_>>();
    one_round.fold_coefficients(r0);
    for (source, &(pointer, capacity)) in one_round.sources.iter().zip(&one_round_allocations) {
        assert_eq!(source.values.as_ptr(), pointer);
        assert_eq!(source.values.capacity(), capacity);
    }
    let expected_one_round = dense
        .chunks_exact(coeff_count)
        .flat_map(|lane| {
            lane.chunks_exact(2)
                .map(|pair| pair[0] + r0 * (pair[1] - pair[0]))
        })
        .collect::<Vec<_>>();
    assert_eq!(one_round.materialize_dense(), expected_one_round);

    let mut two_round =
        PreparedProverLinearTerms::from_dense(dense.clone(), live_lane_count, coeff_count);
    let two_round_allocations = two_round
        .sources
        .iter()
        .map(|source| (source.values.as_ptr(), source.values.capacity()))
        .collect::<Vec<_>>();
    two_round.fold_coefficients(r0);
    two_round.fold_coefficients(r1);
    for (source, &(pointer, capacity)) in two_round.sources.iter().zip(&two_round_allocations) {
        assert_eq!(source.values.as_ptr(), pointer);
        assert_eq!(source.values.capacity(), capacity);
    }
    let expected_two_round = expected_one_round
        .chunks_exact(2)
        .map(|pair| pair[0] + r1 * (pair[1] - pair[0]))
        .collect::<Vec<_>>();
    assert_eq!(two_round.materialize_dense(), expected_two_round);
}

#[test]
fn response_norm_linear_source_matches_dense_table_and_signed_claim() {
    let coeff_count = 4;
    let live_lane_count = 3;
    let coefficient_weights = (0..coeff_count)
        .map(|index| F::from_u64(43 + 11 * index as u64))
        .collect::<Vec<_>>();
    let lane_weights = (0..live_lane_count)
        .map(|index| F::from_u64(71 + 13 * index as u64))
        .collect::<Vec<_>>();
    let prepared = PreparedProverLinearTerms::from_response_norm_factors(
        coefficient_weights.clone(),
        lane_weights.clone(),
        live_lane_count,
        coeff_count,
    )
    .expect("valid rank-one source");
    let expected = (0..live_lane_count)
        .flat_map(|lane| {
            let factor = lane_weights[lane];
            coefficient_weights
                .iter()
                .map(move |&coefficient| factor * coefficient)
        })
        .collect::<Vec<_>>();
    let actual = prepared.materialize_dense();
    let digits = [i8::MIN, -7, 0, 4, 9, -3, 2, -1, 11, 5, -8, 6];
    let actual_claim = digits
        .iter()
        .zip(&actual)
        .map(|(&digit, &weight)| F::from_i64(i64::from(digit)) * weight)
        .sum::<F>();
    let expected_claim = digits
        .iter()
        .enumerate()
        .map(|(index, &digit)| {
            F::from_i64(i64::from(digit))
                * lane_weights[index / coeff_count]
                * coefficient_weights[index % coeff_count]
        })
        .sum::<F>();
    assert_eq!(actual, expected);
    assert_eq!(actual_claim, expected_claim);
}

#[test]
fn structured_linear_terms_reject_malformed_arena_and_incompatible_merge() {
    let valid = StructuredLinearWeights {
        sources: vec![(1..=8)
            .map(|value| F::from_u64(value as u64))
            .collect::<Vec<_>>()
            .into()],
        segments: vec![StructuredLinearSegment {
            physical_coefficient_start: 0,
            source_coefficient_start: 0,
            coefficient_count: 8,
        }],
        terms: vec![StructuredLinearTerm {
            factor: F::from_u64(11),
            source_index: 0,
            segment_range: 0..1,
        }],
        physical_field_len: 8,
    };
    assert!(PreparedProverLinearTerms::from_structured_weights(&valid, 4).is_ok());

    let mut malformed = valid.clone();
    malformed.sources.clear();
    assert!(PreparedProverLinearTerms::from_structured_weights(&malformed, 4).is_err());
    let mut malformed = valid.clone();
    malformed.terms.clear();
    assert!(PreparedProverLinearTerms::from_structured_weights(&malformed, 4).is_err());
    let mut malformed = valid.clone();
    malformed.terms[0].source_index = 1;
    assert!(PreparedProverLinearTerms::from_structured_weights(&malformed, 4).is_err());
    let mut malformed = valid.clone();
    malformed.terms[0].segment_range = 1..1;
    assert!(PreparedProverLinearTerms::from_structured_weights(&malformed, 4).is_err());
    let mut malformed = valid.clone();
    malformed.terms[0].segment_range = 0..2;
    assert!(PreparedProverLinearTerms::from_structured_weights(&malformed, 4).is_err());
    let mut malformed = valid.clone();
    malformed.segments[0].coefficient_count = 0;
    assert!(PreparedProverLinearTerms::from_structured_weights(&malformed, 4).is_err());
    let mut malformed = valid.clone();
    malformed.segments[0].physical_coefficient_start = 1;
    assert!(PreparedProverLinearTerms::from_structured_weights(&malformed, 4).is_err());
    let mut malformed = valid.clone();
    malformed.segments[0].source_coefficient_start = 4;
    assert!(PreparedProverLinearTerms::from_structured_weights(&malformed, 4).is_err());
    let mut malformed = valid.clone();
    malformed.sources[0] = vec![F::one(); 6].into();
    assert!(PreparedProverLinearTerms::from_structured_weights(&malformed, 4).is_err());

    let mut prepared = PreparedProverLinearTerms::from_structured_weights(&valid, 4).unwrap();
    let incompatible = PreparedProverLinearTerms::from_dense(vec![F::one(); 8], 4, 2);
    assert!(prepared.merge(incompatible).is_err());
}

/// A nonempty semantic trace source sharing every live response-norm lane.
pub(crate) fn response_norm_trace_fixture<E: Field>(
    live_lane_count: usize,
    coeff_count: usize,
) -> (PreparedProverLinearTerms<E>, Vec<E>) {
    let lane_bits = live_lane_count.next_power_of_two().trailing_zeros() as usize;
    let weights = EvaluationTraceWeights {
        terms: vec![EvaluationTraceTerm {
            coefficient: E::from_u64(3),
            block_opening_point: vec![E::from_u64(5); lane_bits].into(),
            basis: BasisMode::Lagrange,
            group_block_count: live_lane_count,
            source_ring_dimension: coeff_count,
            opening_ring_dimension: coeff_count,
            coefficient_block_len: coeff_count,
            opening_digit_weights: vec![E::from_u64(7)].into(),
            inner_trace: (0..coeff_count)
                .map(|i| E::from_u64(11 + i as u64))
                .collect::<Vec<_>>()
                .into(),
            segments: vec![EvaluationTraceSegment {
                physical_coefficient_start: 0,
                global_block_start: 0,
                block_count: live_lane_count,
            }],
        }],
        physical_field_len: live_lane_count * coeff_count,
        num_vars: lane_bits + coeff_count.trailing_zeros() as usize,
    };
    let dense = materialize_semantic_trace_oracle(&weights, E::one());
    let prepared =
        PreparedProverLinearTerms::from_evaluation_trace(&weights, coeff_count, E::one()).unwrap();
    assert!(!prepared.sources.is_empty());
    (prepared, dense)
}
