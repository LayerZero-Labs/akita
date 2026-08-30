use super::*;
use akita_challenges::{SparseChallenge, PRODUCTION_FOLD_CHALLENGE_RING_DIMS};
use jolt_field::{
    Ext2, ExtField, FpExt4, One, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59, Zero,
};

fn field_value<T: Ring>(seed: usize) -> T {
    T::from_u64(((seed * 17 + 11) % 97 + 1) as u64)
}

fn signed_scale<T: Field + Ring>(value: T, coefficient: i8) -> T {
    value * T::from_i64(i64::from(coefficient))
}

fn boundary_challenge(s: usize) -> SparseChallenge {
    SparseChallenge {
        positions: vec![0, (s / 2) as u32, (s - 1) as u32].into(),
        coeffs: vec![2, -2, -1].into(),
    }
}

fn independent_subring_product<T: Field + Ring>(
    s: usize,
    challenge: &SparseChallenge,
    rhs: &[T],
) -> Vec<T> {
    let mut output = vec![T::zero(); s];
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        for (rhs_index, &rhs_coefficient) in rhs.iter().enumerate() {
            let ordinary_index = position as usize + rhs_index;
            let term = signed_scale(rhs_coefficient, coefficient);
            if ordinary_index >= s {
                output[ordinary_index - s] -= term;
            } else {
                output[ordinary_index] += term;
            }
        }
    }
    output
}

fn independent_ambient_product<T: Field + Ring>(
    geometry: SubringCoefficientPackingGeometry,
    challenge: &SparseChallenge,
    source: &[T],
) -> Vec<T> {
    let dimension = geometry.a_ring_dimension();
    let stride = geometry.subring_embedding_stride();
    let mut output = vec![T::zero(); dimension];
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        let challenge_index = position as usize * stride;
        for (source_index, &source_coefficient) in source.iter().enumerate() {
            let ordinary_index = challenge_index + source_index;
            let term = signed_scale(source_coefficient, coefficient);
            if ordinary_index >= dimension {
                output[ordinary_index - dimension] -= term;
            } else {
                output[ordinary_index] += term;
            }
        }
    }
    output
}

fn assert_s_linearity<F, E>(s: usize, h: usize)
where
    F: Field + Ring,
    E: ExtField<F>,
{
    let geometry = SubringCoefficientPackingGeometry::try_new(E::DEGREE, E::DEGREE * h * s, s)
        .expect("geometry");
    let source = (0..geometry.a_ring_dimension())
        .map(field_value::<F>)
        .collect::<Vec<_>>();
    let packing_weights = (0..geometry.subring_embedding_stride())
        .map(|index| field_value::<E>(index + 101))
        .collect::<Vec<_>>();
    let challenge = boundary_challenge(s);

    let ambient_product =
        multiply_a_ring_by_subring_challenge(geometry, &challenge, &source).expect("A product");
    assert_eq!(
        ambient_product,
        independent_ambient_product(geometry, &challenge, &source),
        "coefficientwise A product for k={} h={h} s={s}",
        E::DEGREE
    );
    let mapped_ambient =
        coefficient_packing_map::<F, E>(geometry, &ambient_product, &packing_weights)
            .expect("mapped A product");
    let mapped_source = coefficient_packing_map::<F, E>(geometry, &source, &packing_weights)
        .expect("mapped source");
    let subring_product = independent_subring_product(s, &challenge, &mapped_source);

    assert_eq!(
        mapped_ambient,
        subring_product,
        "k={} h={h} s={s}",
        E::DEGREE
    );

    let embedded = embed_subring_challenge_in_a_ring::<F>(geometry, &challenge).expect("embedding");
    for (index, &coefficient) in embedded.iter().enumerate() {
        let expected = challenge
            .positions
            .iter()
            .zip(&challenge.coeffs)
            .find_map(|(&position, &coefficient)| {
                (index == position as usize * geometry.subring_embedding_stride())
                    .then(|| F::from_i64(i64::from(coefficient)))
            })
            .unwrap_or_else(F::zero);
        assert_eq!(coefficient, expected);
    }
}

#[test]
fn subring_embedding_and_linearity_cover_production_field_tiers() {
    for &s in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
        for h in [1usize, 2, 4] {
            assert_s_linearity::<Prime128OffsetA7F7, Prime128OffsetA7F7>(s, h);
        }
    }
    for h in [1usize, 2, 4] {
        assert_s_linearity::<Prime64Offset59, Ext2<Prime64Offset59>>(64, h);
        assert_s_linearity::<Prime32Offset99, FpExt4<Prime32Offset99>>(64, h);
    }
}

fn direct_partial_oracle<F, E>(
    geometry: SubringCoefficientPackingGeometry,
    num_live_positions: usize,
    num_positions_per_block: usize,
    source: &[F],
    position_weights: &[E],
    packing_weights: &[E],
) -> Vec<F>
where
    F: Field,
    E: ExtField<F>,
{
    let s = geometry.challenge_subring_dimension();
    let num_blocks = num_live_positions.div_ceil(num_positions_per_block);
    let mut output = vec![F::zero(); num_blocks * E::DEGREE * s];
    let mut source_position = 0usize;
    for block in 0..num_blocks {
        let num_positions = (num_live_positions - source_position).min(num_positions_per_block);
        for subring_index in 0..s {
            let mut coefficient = E::zero();
            for (position, &position_weight) in
                position_weights.iter().take(num_positions).enumerate()
            {
                for (low_index, &packing_weight) in packing_weights.iter().enumerate() {
                    let ring_index = geometry
                        .a_ring_coefficient_index(low_index, subring_index)
                        .expect("ring index");
                    let flat_index =
                        (source_position + position) * geometry.a_ring_dimension() + ring_index;
                    coefficient += position_weight * packing_weight.mul_base(source[flat_index]);
                }
            }
            for (extension_coordinate, value) in coefficient.to_base_vec().into_iter().enumerate() {
                output[block * geometry.partial_base_field_width()
                    + extension_coordinate * s
                    + subring_index] = value;
            }
        }
        source_position += num_positions;
    }
    output
}

fn assert_partial_and_scalar_factorization<F, E>(h: usize)
where
    F: Field + Ring,
    E: ExtField<F>,
{
    let s = 64;
    let geometry = SubringCoefficientPackingGeometry::try_new(E::DEGREE, E::DEGREE * h * s, s)
        .expect("geometry");
    let num_live_positions = 6usize;
    let num_positions_per_block = 4usize;
    let num_blocks = num_live_positions.div_ceil(num_positions_per_block);
    let position_weights = (0..num_positions_per_block)
        .map(|index| field_value::<E>(index + 211))
        .collect::<Vec<_>>();
    let packing_weights = (0..geometry.subring_embedding_stride())
        .map(|index| field_value::<E>(index + 307))
        .collect::<Vec<_>>();
    let claim_weights = [field_value::<E>(401), field_value::<E>(402)];
    let block_weights = [field_value::<E>(501), field_value::<E>(502)];
    let tail_weights = (0..s)
        .map(|index| field_value::<E>(index + 601))
        .collect::<Vec<_>>();

    let source_len = num_live_positions * geometry.a_ring_dimension();
    let sources = (0..claim_weights.len())
        .map(|claim| {
            (0..source_len)
                .map(|index| field_value::<F>(claim * source_len + index + 701))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut all_partials = Vec::new();
    for source in &sources {
        let got = coefficient_packing_partials::<F, E>(
            geometry,
            num_live_positions,
            num_positions_per_block,
            source,
            &position_weights,
            &packing_weights,
        )
        .expect("partials");
        let expected = direct_partial_oracle(
            geometry,
            num_live_positions,
            num_positions_per_block,
            source,
            &position_weights,
            &packing_weights,
        );
        assert_eq!(got, expected);
        all_partials.push(got);
    }

    let got = coefficient_packing_scalar_opening::<F, E>(
        geometry,
        num_blocks,
        &all_partials,
        &claim_weights,
        &block_weights,
        &tail_weights,
    )
    .expect("scalar opening");

    let mut expected = E::zero();
    for (claim, source) in sources.iter().enumerate() {
        let mut source_position = 0usize;
        for &block_weight in &block_weights {
            let num_positions = (num_live_positions - source_position).min(num_positions_per_block);
            for (position, &position_weight) in
                position_weights.iter().take(num_positions).enumerate()
            {
                for (low_index, &packing_weight) in packing_weights.iter().enumerate() {
                    for (subring_index, &tail_weight) in tail_weights.iter().enumerate() {
                        let ring_index = geometry
                            .a_ring_coefficient_index(low_index, subring_index)
                            .expect("ring index");
                        let source_index =
                            (source_position + position) * geometry.a_ring_dimension() + ring_index;
                        expected += claim_weights[claim]
                            * block_weight
                            * position_weight
                            * packing_weight
                            * tail_weight.mul_base(source[source_index]);
                    }
                }
            }
            source_position += num_positions;
        }
    }
    assert_eq!(got, expected, "k={} h={h}", E::DEGREE);
}

#[test]
fn direct_partials_and_scalar_opening_match_flat_factorization() {
    assert_partial_and_scalar_factorization::<Prime128OffsetA7F7, Prime128OffsetA7F7>(1);
    assert_partial_and_scalar_factorization::<Prime64Offset59, Ext2<Prime64Offset59>>(2);
    assert_partial_and_scalar_factorization::<Prime32Offset99, FpExt4<Prime32Offset99>>(2);
}

#[test]
fn prepared_point_matches_direct_opening_in_both_bases() {
    type F = Prime64Offset59;
    type E = Ext2<F>;

    let geometry = SubringCoefficientPackingGeometry::try_new(2, 256, 64).unwrap();
    let num_live_positions = 6;
    let num_positions_per_block = 4;
    let source_num_vars = 11;
    let point = (0..source_num_vars)
        .map(|index| field_value::<E>(index + 1_001))
        .collect::<Vec<_>>();
    let source = (0..num_live_positions * geometry.a_ring_dimension())
        .map(|index| field_value::<F>(index + 2_001))
        .collect::<Vec<_>>();

    for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
        let prepared = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            basis,
            num_live_positions,
            num_positions_per_block,
            source_num_vars,
            &point,
        )
        .unwrap();
        let partials = coefficient_packing_partials::<F, E>(
            geometry,
            num_live_positions,
            num_positions_per_block,
            &source,
            prepared.position_weights(),
            prepared.packing_weights(),
        )
        .unwrap();
        let got = coefficient_packing_scalar_opening::<F, E>(
            geometry,
            prepared.num_live_blocks(),
            &[partials],
            &[E::one()],
            prepared.live_block_weights(),
            prepared.tail_weights(),
        )
        .unwrap();

        let direct_weights = basis_weights(&point, basis).unwrap();
        let expected = source
            .iter()
            .zip(&direct_weights)
            .fold(E::zero(), |sum, (&coefficient, &weight)| {
                sum + weight.mul_base(coefficient)
            });
        assert_eq!(got, expected, "basis={basis:?}");
    }
}

#[test]
fn single_partial_block_preserves_the_full_position_domain() {
    type F = Prime128OffsetA7F7;
    let geometry = SubringCoefficientPackingGeometry::try_new(1, 64, 64).expect("geometry");
    let num_live_positions = 2;
    let num_positions_per_block = 4;
    let source = (0..num_live_positions * geometry.a_ring_dimension())
        .map(field_value::<F>)
        .collect::<Vec<_>>();
    let position_weights = (0..num_positions_per_block)
        .map(|index| field_value::<F>(index + 1_101))
        .collect::<Vec<_>>();
    let packing_weights = [field_value::<F>(1_201)];

    let got = coefficient_packing_partials::<F, F>(
        geometry,
        num_live_positions,
        num_positions_per_block,
        &source,
        &position_weights,
        &packing_weights,
    )
    .expect("single partial block");
    let expected = direct_partial_oracle(
        geometry,
        num_live_positions,
        num_positions_per_block,
        &source,
        &position_weights,
        &packing_weights,
    );
    assert_eq!(got, expected);
    assert_eq!(got.len(), geometry.partial_base_field_width());
}

fn assert_fold_product_and_divisibility<F, E>()
where
    F: Field + Ring,
    E: ExtField<F>,
{
    let s = 64;
    let geometry =
        SubringCoefficientPackingGeometry::try_new(E::DEGREE, E::DEGREE * s, s).expect("geometry");
    let challenges = vec![
        boundary_challenge(s),
        SparseChallenge {
            positions: vec![1, (s - 1) as u32].into(),
            coeffs: vec![-1, 2].into(),
        },
    ];
    let partials = (0..challenges.len() * geometry.partial_base_field_width())
        .map(|index| field_value::<F>(index + 809))
        .collect::<Vec<_>>();
    let product =
        fold_coefficient_packing_partials(geometry, &challenges, &partials).expect("fold product");

    let mut packed_quotient = vec![E::zero(); s];
    for extension_coordinate in 0..E::DEGREE {
        let mut ordinary = vec![F::zero(); 2 * s - 1];
        for (term, challenge) in challenges.iter().enumerate() {
            let partial_offset =
                term * geometry.partial_base_field_width() + extension_coordinate * s;
            for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                for partial_index in 0..s {
                    ordinary[position as usize + partial_index] +=
                        signed_scale(partials[partial_offset + partial_index], coefficient);
                }
            }
        }
        for index in 0..s {
            let high = ordinary.get(index + s).copied().unwrap_or_else(F::zero);
            assert_eq!(
                product.reduced_base_field_coordinates()[extension_coordinate * s + index],
                ordinary[index] - high
            );
            assert_eq!(
                product.quotient_high_half_base_field_coordinates()
                    [extension_coordinate * s + index],
                high
            );
            assert_eq!(ordinary[index] - (ordinary[index] - high), high);
        }
    }

    for (subring_index, packed_coefficient) in packed_quotient.iter_mut().enumerate() {
        let coordinates = (0..E::DEGREE)
            .map(|extension_coordinate| {
                product.quotient_high_half_base_field_coordinates()
                    [extension_coordinate * s + subring_index]
            })
            .collect::<Vec<_>>();
        *packed_coefficient = E::from_base_slice(&coordinates);
    }
    let alpha = field_value::<E>(997);
    let packed_eval = packed_quotient
        .iter()
        .rev()
        .fold(E::zero(), |acc, &coefficient| acc * alpha + coefficient);
    let basis_combined_eval = (0..E::DEGREE).fold(E::zero(), |sum, coordinate| {
        let mut basis_coordinates = vec![F::zero(); E::DEGREE];
        basis_coordinates[coordinate] = F::one();
        let basis = E::from_base_slice(&basis_coordinates);
        let plane_eval = product.quotient_high_half_base_field_coordinates()
            [coordinate * s..(coordinate + 1) * s]
            .iter()
            .rev()
            .fold(E::zero(), |acc, &coefficient| {
                acc * alpha + E::lift_base(coefficient)
            });
        sum + basis * plane_eval
    });
    assert_eq!(packed_eval, basis_combined_eval);
}

#[test]
fn folded_product_returns_reduction_and_positive_divisibility_quotient() {
    assert_fold_product_and_divisibility::<Prime128OffsetA7F7, Prime128OffsetA7F7>();
    assert_fold_product_and_divisibility::<Prime64Offset59, Ext2<Prime64Offset59>>();
    assert_fold_product_and_divisibility::<Prime32Offset99, FpExt4<Prime32Offset99>>();
}

#[test]
fn malformed_reference_inputs_reject_without_panicking() {
    type F = Prime32Offset99;
    type E = FpExt4<F>;
    let geometry = SubringCoefficientPackingGeometry::try_new(4, 256, 64).expect("geometry");
    let source = vec![F::one(); 256];
    let packing_weights = vec![E::one(); 4];
    assert!(coefficient_packing_map::<F, F>(geometry, &source, &[F::one(); 4]).is_err());
    assert!(coefficient_packing_map::<F, E>(geometry, &source[..255], &packing_weights).is_err());
    assert!(coefficient_packing_map::<F, E>(geometry, &source, &packing_weights[..3]).is_err());

    let num_live_positions = 6;
    let num_positions_per_block = 4;
    let partial_source = vec![F::one(); num_live_positions * geometry.a_ring_dimension()];
    let position_weights = vec![E::one(); num_positions_per_block];
    assert!(coefficient_packing_partials::<F, E>(
        geometry,
        0,
        num_positions_per_block,
        &[],
        &position_weights,
        &packing_weights,
    )
    .is_err());
    for invalid_domain in [0, 3] {
        assert!(coefficient_packing_partials::<F, E>(
            geometry,
            num_live_positions,
            invalid_domain,
            &partial_source,
            &position_weights,
            &packing_weights,
        )
        .is_err());
    }
    for malformed_source in [
        &partial_source[..partial_source.len() - 1],
        &[partial_source.as_slice(), &[F::one()]].concat(),
    ] {
        assert!(coefficient_packing_partials::<F, E>(
            geometry,
            num_live_positions,
            num_positions_per_block,
            malformed_source,
            &position_weights,
            &packing_weights,
        )
        .is_err());
    }
    for malformed_weights in [
        &position_weights[..position_weights.len() - 1],
        &[position_weights.as_slice(), &[E::one()]].concat(),
    ] {
        assert!(coefficient_packing_partials::<F, E>(
            geometry,
            num_live_positions,
            num_positions_per_block,
            &partial_source,
            malformed_weights,
            &packing_weights,
        )
        .is_err());
    }
    for malformed_packing_weights in [
        &packing_weights[..packing_weights.len() - 1],
        &[packing_weights.as_slice(), &[E::one()]].concat(),
    ] {
        assert!(coefficient_packing_partials::<F, E>(
            geometry,
            num_live_positions,
            num_positions_per_block,
            &partial_source,
            &position_weights,
            malformed_packing_weights,
        )
        .is_err());
    }
    assert!(coefficient_packing_partials::<F, F>(
        geometry,
        num_live_positions,
        num_positions_per_block,
        &partial_source,
        &[F::one(); 4],
        &[F::one(); 4],
    )
    .is_err());

    let num_claims = 2;
    let num_blocks = 2;
    let scalar_partials =
        vec![vec![F::one(); num_blocks * geometry.partial_base_field_width()]; num_claims];
    let claim_weights = vec![E::one(); num_claims];
    let block_weights = vec![E::one(); num_blocks];
    let tail_weights = vec![E::one(); geometry.challenge_subring_dimension()];
    assert!(coefficient_packing_scalar_opening::<F, E>(
        geometry,
        num_blocks,
        &Vec::<Vec<F>>::new(),
        &[],
        &block_weights,
        &tail_weights,
    )
    .is_err());
    assert!(coefficient_packing_scalar_opening::<F, E>(
        geometry,
        0,
        &scalar_partials,
        &claim_weights,
        &[],
        &tail_weights,
    )
    .is_err());
    for malformed_claim in [
        scalar_partials[0][..scalar_partials[0].len() - 1].to_vec(),
        [scalar_partials[0].as_slice(), &[F::one()]].concat(),
    ] {
        let malformed_partials = [malformed_claim, scalar_partials[1].clone()];
        assert!(coefficient_packing_scalar_opening::<F, E>(
            geometry,
            num_blocks,
            &malformed_partials,
            &claim_weights,
            &block_weights,
            &tail_weights,
        )
        .is_err());
    }
    for malformed_claim_weights in [
        &claim_weights[..claim_weights.len() - 1],
        &[claim_weights.as_slice(), &[E::one()]].concat(),
    ] {
        assert!(coefficient_packing_scalar_opening::<F, E>(
            geometry,
            num_blocks,
            &scalar_partials,
            malformed_claim_weights,
            &block_weights,
            &tail_weights,
        )
        .is_err());
    }
    for malformed_block_weights in [
        &block_weights[..block_weights.len() - 1],
        &[block_weights.as_slice(), &[E::one()]].concat(),
    ] {
        assert!(coefficient_packing_scalar_opening::<F, E>(
            geometry,
            num_blocks,
            &scalar_partials,
            &claim_weights,
            malformed_block_weights,
            &tail_weights,
        )
        .is_err());
    }
    for malformed_tail_weights in [
        &tail_weights[..tail_weights.len() - 1],
        &[tail_weights.as_slice(), &[E::one()]].concat(),
    ] {
        assert!(coefficient_packing_scalar_opening::<F, E>(
            geometry,
            num_blocks,
            &scalar_partials,
            &claim_weights,
            &block_weights,
            malformed_tail_weights,
        )
        .is_err());
    }
    assert!(coefficient_packing_scalar_opening::<F, F>(
        geometry,
        num_blocks,
        &scalar_partials,
        &[F::one(); 2],
        &[F::one(); 2],
        &[F::one(); 64],
    )
    .is_err());

    let valid = SparseChallenge {
        positions: vec![0, 63].into(),
        coeffs: vec![1, -1].into(),
    };
    for malformed_source in [
        &source[..source.len() - 1],
        &[source.as_slice(), &[F::one()]].concat(),
    ] {
        assert!(multiply_a_ring_by_subring_challenge(geometry, &valid, malformed_source).is_err());
    }
    for malformed in [
        SparseChallenge {
            positions: vec![0, 1].into(),
            coeffs: vec![1].into(),
        },
        SparseChallenge {
            positions: vec![0, 0].into(),
            coeffs: vec![1, -1].into(),
        },
        SparseChallenge {
            positions: vec![0].into(),
            coeffs: vec![0].into(),
        },
        SparseChallenge {
            positions: vec![64].into(),
            coeffs: vec![1].into(),
        },
    ] {
        assert!(embed_subring_challenge_in_a_ring::<F>(geometry, &malformed).is_err());
        assert!(fold_coefficient_packing_partials(
            geometry,
            &[malformed],
            &vec![F::one(); geometry.partial_base_field_width()],
        )
        .is_err());
    }
    for malformed_partials in [
        vec![F::one(); geometry.partial_base_field_width() - 1],
        vec![F::one(); geometry.partial_base_field_width() + 1],
    ] {
        assert!(fold_coefficient_packing_partials(
            geometry,
            std::slice::from_ref(&valid),
            &malformed_partials,
        )
        .is_err());
    }
}
