use super::poly::DensePoly;
use super::prepared::{PreparedDenseStorage, PreparedDenseWitness};
use crate::compute::{
    CpuBackend, DecomposeFoldPlan, OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPlan,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::{CanonicalField, Prime128OffsetA7F7 as F};
use akita_types::{
    BasisMode, PreparedSubringCoefficientPackingPoint, SubringCoefficientPackingGeometry,
};

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

#[test]
fn ring_fold_matches_dense_multiplication_reference() {
    const D: usize = 8;
    let coeffs = (0..2).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_ring_coeffs(coeffs.clone());
    let scalars = vec![
        ring::<D>(100),
        ring::<D>(200),
        ring::<D>(300),
        ring::<D>(400),
    ];
    let got = poly.fold_blocks_ring(&scalars, 4);
    let expected = coeffs
        .chunks(4)
        .map(|block| {
            block
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (coeff, scalar)| {
                    acc + (*coeff * *scalar)
                })
        })
        .collect::<Vec<_>>();

    assert_eq!(got, expected);
}

#[test]
fn dense_constructor_reuses_owned_evaluation_buffer() {
    let evals = (0..2048).map(F::from_u64).collect::<Vec<_>>();
    let allocation = evals.as_ptr();
    let poly = DensePoly::<F>::from_field_evals(11, evals).unwrap();
    assert_eq!(poly.field_coeffs().as_ptr(), allocation);
}

#[test]
fn dense_source_has_exact_views_across_supported_ring_dimensions() {
    let evals = (1..=32).map(F::from_u64).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(5, evals.clone()).unwrap();

    fn assert_view<const D: usize>(poly: &DensePoly<F>, evals: &[F]) {
        let rings = poly.ring_coeffs::<D>().expect("supported dense view");
        let flat = rings
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(&flat[..evals.len()], evals);
        assert!(flat[evals.len()..].iter().all(|value| *value == F::zero()));
    }

    assert_view::<64>(&poly, &evals);
    assert_view::<128>(&poly, &evals);
    assert_view::<256>(&poly, &evals);
    assert_view::<512>(&poly, &evals);
    assert_view::<1024>(&poly, &evals);
}

#[test]
fn dense_digit_cache_is_exact_and_bit_packed() {
    const D: usize = 64;
    const NUM_DIGITS: usize = 32;
    const LOG_BASIS: u32 = 4;

    let evals = (0..128)
        .map(|index| F::from_u64((index * 17 + 9) as u64))
        .collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(7, evals).unwrap();
    let packed = poly
        .digit_planes_for::<D>(NUM_DIGITS, LOG_BASIS)
        .expect("i8 decomposition is packed");

    let q = (-F::one()).to_canonical_u128() + 1;
    let params = BalancedDecomposePow2Params::new(NUM_DIGITS, LOG_BASIS, q);
    let mut expected = vec![0i8; packed.len()];
    for (ring, planes) in poly
        .ring_coeffs::<D>()
        .unwrap()
        .iter()
        .zip(expected.chunks_exact_mut(NUM_DIGITS * D))
    {
        let (planes, remainder) = planes.as_chunks_mut::<D>();
        assert!(remainder.is_empty());
        ring.balanced_decompose_pow2_i8_into_with_params(planes, &params);
    }

    assert_eq!(packed.iter().collect::<Vec<_>>(), expected);
    let (encoded_bytes, bit_width) = poly.cached_digit_storage().unwrap();
    assert_eq!(bit_width, LOG_BASIS as u8);
    assert_eq!(encoded_bytes, packed.len() * LOG_BASIS as usize / 8);
}

#[test]
fn prepared_dense_witness_matches_canonical_opening_kernels() {
    const D: usize = 64;
    const NUM_DIGITS: usize = 32;
    const LOG_BASIS: u32 = 4;
    const POSITIONS_PER_BLOCK: usize = 4;

    let evals = (0..1024)
        .map(|index| match index % 4 {
            0 => F::from_i64((index % 29) as i64 - 14),
            1 => F::from_canonical_u128_reduced(u128::MAX - index as u128),
            2 => -F::from_u64((index + 1) as u64),
            _ => F::from_canonical_u128_reduced((1u128 << 127) + index as u128),
        })
        .collect::<Vec<_>>();
    let canonical = DensePoly::<F>::from_field_evals(10, evals.clone()).unwrap();
    let packed_source = DensePoly::<F>::from_field_evals(10, evals).unwrap();
    packed_source
        .digit_planes_for::<D>(NUM_DIGITS, LOG_BASIS)
        .expect("prepare packed digits");
    let prepared = packed_source.into_prepared_witness().unwrap();
    assert!(matches!(
        &prepared.storage,
        PreparedDenseStorage::Canonical(_)
    ));

    let num_rings = canonical.ring_coeffs::<D>().unwrap().len();
    let num_blocks = num_rings.div_ceil(POSITIONS_PER_BLOCK);
    let position_weights = (0..POSITIONS_PER_BLOCK)
        .map(|index| F::from_u64((index + 2) as u64))
        .collect::<Vec<_>>();
    let block_weights = (0..num_blocks)
        .map(|index| F::from_u64((index + 7) as u64))
        .collect::<Vec<_>>();
    let opening_plan = OpeningFoldPlan::Base {
        live_block_weights: &block_weights,
        position_weights: &position_weights,
        num_positions_per_block: POSITIONS_PER_BLOCK,
    };
    let canonical_opening = <CpuBackend as OpeningFoldKernel<_, F, D>>::evaluate_and_fold(
        &CpuBackend::DEFAULT,
        None,
        canonical.opening_view().unwrap(),
        opening_plan,
    )
    .unwrap();
    let prepared_opening = <CpuBackend as OpeningFoldKernel<_, F, D>>::evaluate_and_fold(
        &CpuBackend::DEFAULT,
        None,
        prepared.opening_view().unwrap(),
        opening_plan,
    )
    .unwrap();
    assert_eq!(prepared_opening, canonical_opening);

    let challenges = (0..num_blocks)
        .map(|block| SparseChallenge {
            positions: vec![1, 7, 19, 43].into(),
            coeffs: vec![1, -1, 2, if block.is_multiple_of(2) { -2 } else { 1 }].into(),
        })
        .collect::<Vec<_>>();
    let decompose_plan = DecomposeFoldPlan {
        challenges: &challenges,
        num_positions_per_block: POSITIONS_PER_BLOCK,
        num_digits: NUM_DIGITS,
        log_basis: LOG_BASIS,
    };
    let canonical_fold = <CpuBackend as OpeningFoldKernel<_, F, D>>::decompose_fold(
        &CpuBackend::DEFAULT,
        None,
        canonical.opening_view().unwrap(),
        decompose_plan,
    )
    .unwrap();
    let prepared_fold = <CpuBackend as OpeningFoldKernel<_, F, D>>::decompose_fold(
        &CpuBackend::DEFAULT,
        None,
        prepared.opening_view().unwrap(),
        decompose_plan,
    )
    .unwrap();
    assert_eq!(prepared_fold, canonical_fold);

    let geometry = SubringCoefficientPackingGeometry::try_new(1, D, D).unwrap();
    let public_point = (0..10)
        .map(|index| F::from_u64((index + 3) as u64))
        .collect::<Vec<_>>();
    let point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        num_rings,
        POSITIONS_PER_BLOCK,
        10,
        &public_point,
    )
    .unwrap();
    let canonical_refs = [&canonical];
    let prepared_refs: [&PreparedDenseWitness<F>; 1] = [&prepared];
    let canonical_batch =
        <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&canonical_refs).unwrap();
    let prepared_batch =
        <PreparedDenseWitness<F> as RootOpeningSource<F, D>>::opening_batch(&prepared_refs)
            .unwrap();
    let packing_plan = SubringCoefficientPackingPlan { point: &point };
    let canonical_partials =
        SubringCoefficientPackingBatchKernel::coefficient_packing_partials_batch(
            &CpuBackend::DEFAULT,
            None,
            canonical_batch,
            packing_plan,
        )
        .unwrap();
    let prepared_partials =
        SubringCoefficientPackingBatchKernel::coefficient_packing_partials_batch(
            &CpuBackend::DEFAULT,
            None,
            prepared_batch,
            packing_plan,
        )
        .unwrap();
    assert_eq!(prepared_partials, canonical_partials);
}

#[test]
fn prepared_dense_witness_packs_fast_reconstruction_spans() {
    const D: usize = 64;
    let evals = (0..1024)
        .map(|index| F::from_i64((index % 29) as i64 - 14))
        .collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(10, evals).unwrap();
    poly.digit_planes_for::<D>(8, 4).unwrap();
    let prepared = poly.into_prepared_witness().unwrap();
    assert!(matches!(prepared.storage, PreparedDenseStorage::Packed(_)));
}
