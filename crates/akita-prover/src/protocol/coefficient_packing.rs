//! Transcript-free construction of coefficient-packing opening material.

use crate::compute::SubringCoefficientPackingPartials;
use crate::validation::validate_i8_setup_log_basis;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::Challenges;
use akita_error::AkitaError;
use akita_types::{
    fold_coefficient_packing_partials, CoefficientPackingFoldProduct, CommittedGroupParams,
    DigitBlocks, OpeningClaimsLayout, OpeningMethod, RelationWitnessGeometry,
    SubringCoefficientPackingGeometry,
};
use jolt_field::{CanonicalEncoding, Field};

/// Fold one group's canonical partials with its single sampled subring challenge batch.
pub(super) fn fold_coefficient_packing_group<F: Field + jolt_field::Ring>(
    geometry: SubringCoefficientPackingGeometry,
    partials_by_claim: &[SubringCoefficientPackingPartials<F>],
    challenges: &Challenges,
) -> Result<CoefficientPackingFoldProduct<F>, AkitaError> {
    if partials_by_claim.len() != challenges.num_claims()
        || partials_by_claim.is_empty()
        || partials_by_claim.iter().any(|partials| {
            partials.geometry() != geometry
                || partials.num_live_blocks() != challenges.num_live_blocks_per_claim()
        })
    {
        return Err(AkitaError::InvalidInput(
            "coefficient-packing fold inputs disagree on claims, blocks, or geometry".into(),
        ));
    }
    let expected_challenges = partials_by_claim
        .len()
        .checked_mul(challenges.num_live_blocks_per_claim())
        .ok_or_else(|| AkitaError::InvalidInput("packing challenge count overflow".into()))?;
    if challenges.len() != expected_challenges {
        return Err(AkitaError::InvalidSize {
            expected: expected_challenges,
            actual: challenges.len(),
        });
    }
    let expected_coordinates = expected_challenges
        .checked_mul(geometry.partial_base_field_width())
        .ok_or_else(|| AkitaError::InvalidInput("packing partial length overflow".into()))?;
    let mut coordinates = Vec::new();
    coordinates
        .try_reserve_exact(expected_coordinates)
        .map_err(|_| AkitaError::InvalidInput("packing partial fold allocation failed".into()))?;
    for partials in partials_by_claim {
        coordinates.extend_from_slice(partials.coordinates());
    }
    fold_coefficient_packing_partials(geometry, challenges.as_slice(), &coordinates)
}

/// Concatenate group-local D inputs in canonical relation order.
pub(super) fn concatenate_group_d_inputs(
    opening_batch: &OpeningClaimsLayout,
    group_inputs: &[&DigitBlocks],
) -> Result<DigitBlocks, AkitaError> {
    if group_inputs.len() != opening_batch.num_groups() {
        return Err(AkitaError::InvalidSize {
            expected: opening_batch.num_groups(),
            actual: group_inputs.len(),
        });
    }
    let mut order = opening_batch.root_group_order()?.into_iter();
    let first_index = order.next().ok_or(AkitaError::InvalidProof)?;
    let first = *group_inputs
        .get(first_index)
        .ok_or(AkitaError::InvalidProof)?;
    let stride = first.digit_stride();
    let mut digits = Vec::new();
    let mut block_sizes = Vec::new();
    for group_index in std::iter::once(first_index).chain(order) {
        let group = *group_inputs
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if group.digit_stride() != stride {
            return Err(AkitaError::InvalidInput(
                "opening groups have mixed D dimensions".into(),
            ));
        }
        digits.extend_from_slice(group.digits());
        block_sizes.extend_from_slice(group.block_sizes());
    }
    DigitBlocks::new(digits, block_sizes, stride)
}

/// Decompose canonical coefficient-packing partials into the exact D input.
///
/// Logical blocks are ordered `[claim][partial block]`. Within each block,
/// coordinate planes are split into consecutive D-ring subcolumns and then
/// gadget digit planes.
pub(super) fn materialize_coefficient_packing_d_input<
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    const D_D: usize,
>(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    relation_geometry: &RelationWitnessGeometry,
    group_index: usize,
    partials_by_claim: &[SubringCoefficientPackingPartials<F>],
) -> Result<DigitBlocks, AkitaError> {
    let group_params = level_params.group_params_geometry(opening_batch, group_index)?;
    let group_layout = opening_batch.group_layout(group_index)?;
    let d_a = group_params.inner_commit_matrix_params().ring_dimension();
    let challenge_subring_dimension = match group_params.opening_method() {
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => challenge_subring_dimension,
        OpeningMethod::EvaluationTrace => {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing D input requires the coefficient-packing method".into(),
            ));
        }
    };
    let packing_geometry = SubringCoefficientPackingGeometry::try_new(
        relation_geometry.extension_degree(),
        d_a,
        challenge_subring_dimension,
    )?;
    let num_digits_open = group_params.num_digits_open();
    let log_basis_open = group_params.log_basis_open();
    validate_i8_setup_log_basis(log_basis_open, "for coefficient-packing D input")?;
    if num_digits_open == 0
        || partials_by_claim.len() != group_layout.num_polynomials()
        || D_D == 0
        || D_D != level_params.role_dims().d_d()
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing D input disagrees with scheduled claims, digits, or D dimension"
                .into(),
        ));
    }
    let expected_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: packing_geometry.challenge_subring_dimension(),
    };
    if relation_geometry.group_opening_method(group_index)? != expected_method
        || relation_geometry.extension_degree() != packing_geometry.extension_degree()
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing D input disagrees with relation method".into(),
        ));
    }
    let opening_geometry = relation_geometry.group_opening_geometry(group_index)?;
    if opening_geometry.polynomial_modulus_dimension()
        != packing_geometry.challenge_subring_dimension()
        || opening_geometry.coordinate_plane_count() != packing_geometry.extension_degree()
        || opening_geometry.physical_coefficient_width()
            != packing_geometry.partial_base_field_width()
        || !packing_geometry
            .partial_base_field_width()
            .is_multiple_of(D_D)
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing D input has incompatible physical geometry".into(),
        ));
    }

    let num_live_blocks = group_params.num_live_blocks();
    if partials_by_claim.iter().any(|partials| {
        partials.geometry() != packing_geometry || partials.num_live_blocks() != num_live_blocks
    }) {
        return Err(AkitaError::InvalidInput(
            "coefficient-packing claims have inconsistent partial geometry".into(),
        ));
    }
    let semantic_blocks = partials_by_claim
        .len()
        .checked_mul(num_live_blocks)
        .ok_or_else(|| {
            AkitaError::InvalidInput("coefficient-packing block count overflow".into())
        })?;
    let role_subcolumns = packing_geometry.partial_base_field_width() / D_D;
    let planes_per_block = role_subcolumns
        .checked_mul(num_digits_open)
        .ok_or_else(|| {
            AkitaError::InvalidInput("coefficient-packing digit plane count overflow".into())
        })?;
    let mut digits = DigitBlocks::zeroed(vec![planes_per_block; semantic_blocks], D_D)?;
    let q = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let params = BalancedDecomposePow2Params::new(num_digits_open, log_basis_open, q);
    let typed_planes = digits.typed_planes_mut::<D_D>()?;

    for (claim_index, partials) in partials_by_claim.iter().enumerate() {
        for block_index in 0..num_live_blocks {
            let semantic_index = claim_index
                .checked_mul(num_live_blocks)
                .and_then(|base| base.checked_add(block_index))
                .ok_or(AkitaError::InvalidProof)?;
            let source_start = block_index
                .checked_mul(packing_geometry.partial_base_field_width())
                .ok_or(AkitaError::InvalidProof)?;
            let source_end = source_start
                .checked_add(packing_geometry.partial_base_field_width())
                .ok_or(AkitaError::InvalidProof)?;
            let source = partials
                .coordinates()
                .get(source_start..source_end)
                .ok_or(AkitaError::InvalidProof)?;
            for subcolumn in 0..role_subcolumns {
                let ring_start = subcolumn.checked_mul(D_D).ok_or(AkitaError::InvalidProof)?;
                let ring_end = ring_start
                    .checked_add(D_D)
                    .ok_or(AkitaError::InvalidProof)?;
                let ring = CyclotomicRing::<F, D_D>::from_slice(
                    source
                        .get(ring_start..ring_end)
                        .ok_or(AkitaError::InvalidProof)?,
                );
                let plane_start = semantic_index
                    .checked_mul(planes_per_block)
                    .and_then(|base| base.checked_add(subcolumn * num_digits_open))
                    .ok_or(AkitaError::InvalidProof)?;
                let plane_end = plane_start
                    .checked_add(num_digits_open)
                    .ok_or(AkitaError::InvalidProof)?;
                ring.balanced_decompose_pow2_i8_into_with_params(
                    typed_planes
                        .get_mut(plane_start..plane_end)
                        .ok_or(AkitaError::InvalidProof)?,
                    &params,
                );
            }
        }
    }
    Ok(digits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::{SparseChallenge, SparseChallengeConfig};
    use akita_types::{
        gadget_row_scalars, CommittedGroupParams, OpenCommitMatrixParams, OpeningClaimsLayout,
        PolynomialGroupLayout, SisModulusProfileId,
    };
    use jolt_field::{Prime128OffsetA7F7, Ring, Zero};

    type F = Prime128OffsetA7F7;

    #[test]
    fn grouped_fold_preserves_claim_block_order_and_positive_high_half() {
        let geometry = SubringCoefficientPackingGeometry::try_new(2, 128, 64).unwrap();
        let partials = (0..2)
            .map(|claim| {
                SubringCoefficientPackingPartials::new(
                    geometry,
                    2,
                    (0..256)
                        .map(|index| F::from_i64(((claim * 256 + index) % 13) as i64 - 6))
                        .collect(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let sparse = vec![
            SparseChallenge {
                positions: vec![0, 63].into(),
                coeffs: vec![1, -1].into(),
            },
            SparseChallenge {
                positions: vec![1, 62].into(),
                coeffs: vec![2, 1].into(),
            },
            SparseChallenge {
                positions: vec![2].into(),
                coeffs: vec![-2].into(),
            },
            SparseChallenge {
                positions: vec![31, 63].into(),
                coeffs: vec![1, 2].into(),
            },
        ];
        let challenges = Challenges::from_sparse(sparse.clone(), 2, 2).unwrap();
        let got = fold_coefficient_packing_group(geometry, &partials, &challenges).unwrap();
        assert_eq!(got.geometry(), geometry);
        let flat = partials
            .iter()
            .flat_map(|partial| partial.coordinates().iter().copied())
            .collect::<Vec<_>>();
        let expected = fold_coefficient_packing_partials(geometry, &sparse, &flat).unwrap();
        assert_eq!(got, expected);
        assert!(got
            .quotient_high_half_base_field_coordinates()
            .iter()
            .any(|coefficient| !coefficient.is_zero()));
    }

    fn packing_fixture() -> (
        CommittedGroupParams,
        OpeningClaimsLayout,
        RelationWitnessGeometry,
    ) {
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            128,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::production_for_ring_dim(128).unwrap(),
        )
        .with_decomp(4, 4, 2, 2, 2)
        .unwrap();
        params.own_group_mut().opening.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
        params.own_group_mut().opening.fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let batch =
            OpeningClaimsLayout::from_groups(vec![PolynomialGroupLayout::new(9, 2)]).unwrap();
        let geometry = RelationWitnessGeometry::for_level(&params, &batch, 2).unwrap();
        (params, batch, geometry)
    }

    #[test]
    fn d_input_uses_physical_width_not_subring_modulus() {
        const D_D: usize = 128;
        let packing_geometry = SubringCoefficientPackingGeometry::try_new(2, 128, 64).unwrap();
        let (params, batch, relation_geometry) = packing_fixture();
        let partials = (0..2)
            .map(|claim| {
                let coordinates = (0..128)
                    .map(|index| F::from_i64(((claim * 128 + index) % 9) as i64 - 4))
                    .collect();
                SubringCoefficientPackingPartials::new(packing_geometry, 1, coordinates).unwrap()
            })
            .collect::<Vec<_>>();
        let digits = materialize_coefficient_packing_d_input::<F, D_D>(
            &params,
            &batch,
            &relation_geometry,
            0,
            &partials,
        )
        .unwrap();
        assert_eq!(digits.block_sizes(), &[2, 2]);
        assert_eq!(digits.digit_stride(), D_D);
        let scalars = gadget_row_scalars::<F>(2, 2);
        for (claim, block) in digits.iter_blocks().enumerate() {
            for coefficient in 0..D_D {
                let recomposed = (0..2).fold(F::zero(), |sum, digit| {
                    sum + F::from_i8(block[digit * D_D + coefficient]) * scalars[digit]
                });
                assert_eq!(recomposed, partials[claim].coordinates()[coefficient]);
            }
        }
    }

    #[test]
    fn d_input_rejects_method_and_width_mismatches() {
        let packing_geometry = SubringCoefficientPackingGeometry::try_new(2, 128, 64).unwrap();
        let (params, batch, relation_geometry) = packing_fixture();
        let partial =
            SubringCoefficientPackingPartials::new(packing_geometry, 1, vec![F::zero(); 128])
                .unwrap();
        assert!(materialize_coefficient_packing_d_input::<F, 128>(
            &params,
            &batch,
            &relation_geometry,
            0,
            &[partial.clone(), partial.clone()],
        )
        .is_ok());
        let wrong_extension = SubringCoefficientPackingPartials::new(
            SubringCoefficientPackingGeometry::try_new(1, 128, 64).unwrap(),
            1,
            vec![F::zero(); 64],
        )
        .unwrap();
        assert!(materialize_coefficient_packing_d_input::<F, 128>(
            &params,
            &batch,
            &relation_geometry,
            0,
            &[wrong_extension.clone(), wrong_extension],
        )
        .is_err());
        assert!(materialize_coefficient_packing_d_input::<F, 128>(
            &params,
            &batch,
            &relation_geometry,
            0,
            &[partial],
        )
        .is_err());
    }

    #[test]
    fn d_input_rejects_ambient_ring_and_block_count_aliases() {
        let (params, batch, relation_geometry) = packing_fixture();
        let aliased_width_geometry =
            SubringCoefficientPackingGeometry::try_new(2, 256, 64).unwrap();
        let wrong_ambient =
            SubringCoefficientPackingPartials::new(aliased_width_geometry, 1, vec![F::zero(); 128])
                .unwrap();
        assert!(materialize_coefficient_packing_d_input::<F, 128>(
            &params,
            &batch,
            &relation_geometry,
            0,
            &[wrong_ambient.clone(), wrong_ambient],
        )
        .is_err());

        let expected_geometry = SubringCoefficientPackingGeometry::try_new(2, 128, 64).unwrap();
        let wrong_blocks =
            SubringCoefficientPackingPartials::new(expected_geometry, 2, vec![F::zero(); 256])
                .unwrap();
        assert!(materialize_coefficient_packing_d_input::<F, 128>(
            &params,
            &batch,
            &relation_geometry,
            0,
            &[wrong_blocks.clone(), wrong_blocks],
        )
        .is_err());
    }

    #[test]
    fn d_input_preserves_claim_block_subcolumn_digit_order() {
        const D_D: usize = 64;
        let mut params = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            256,
            2,
            1,
            1,
            1,
            SparseChallengeConfig::production_for_ring_dim(256).unwrap(),
        )
        .with_decomp(4, 6, 2, 2, 2)
        .unwrap();
        params.own_group_mut().opening.opening_method = OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
        params.own_group_mut().opening.fold_challenge_config =
            SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        params.open_matrix = OpenCommitMatrixParams::new_unchecked(
            params.open().matrix.security_policy(),
            params.open().matrix.sis_table_key().table_digest,
            params.open().matrix.sis_modulus_profile(),
            params.open().matrix.output_rank(),
            8,
            params.open().matrix.coeff_linf_bound(),
            D_D,
        );
        let batch =
            OpeningClaimsLayout::from_groups(vec![PolynomialGroupLayout::new(11, 2)]).unwrap();
        let relation_geometry = RelationWitnessGeometry::for_level(&params, &batch, 2).unwrap();
        let packing_geometry = SubringCoefficientPackingGeometry::try_new(2, 256, 64).unwrap();
        let partials = (0..2)
            .map(|claim| {
                SubringCoefficientPackingPartials::new(
                    packing_geometry,
                    2,
                    (0..256)
                        .map(|index| F::from_i64(((claim * 256 + index) % 11) as i64 - 5))
                        .collect(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let digits = materialize_coefficient_packing_d_input::<F, D_D>(
            &params,
            &batch,
            &relation_geometry,
            0,
            &partials,
        )
        .unwrap();
        assert_eq!(digits.block_sizes(), &[4, 4, 4, 4]);
        let scalars = gadget_row_scalars::<F>(2, 2);
        for (semantic_index, block) in digits.iter_blocks().enumerate() {
            let claim = semantic_index / 2;
            let partial_block = semantic_index % 2;
            for subcolumn in 0..2 {
                for coefficient in 0..D_D {
                    let recomposed = (0..2).fold(F::zero(), |sum, digit| {
                        let plane = subcolumn * 2 + digit;
                        sum + F::from_i8(block[plane * D_D + coefficient]) * scalars[digit]
                    });
                    let source_index = partial_block * 128 + subcolumn * D_D + coefficient;
                    assert_eq!(recomposed, partials[claim].coordinates()[source_index]);
                }
            }
        }
    }

    #[test]
    fn grouped_d_input_uses_final_then_precommit_relation_order() {
        let precommitted = [
            PolynomialGroupLayout::new(8, 1),
            PolynomialGroupLayout::new(9, 1),
        ];
        let batch =
            OpeningClaimsLayout::from_root_groups(&precommitted, PolynomialGroupLayout::new(7, 1))
                .unwrap();
        let groups = [
            DigitBlocks::new(vec![1], vec![1], 1).unwrap(),
            DigitBlocks::new(vec![2], vec![1], 1).unwrap(),
            DigitBlocks::new(vec![3], vec![1], 1).unwrap(),
        ];
        let refs = groups.iter().collect::<Vec<_>>();
        let concatenated = concatenate_group_d_inputs(&batch, &refs).unwrap();
        assert_eq!(concatenated.digits(), &[3, 1, 2]);
        assert_eq!(concatenated.block_sizes(), &[1, 1, 1]);
    }
}
