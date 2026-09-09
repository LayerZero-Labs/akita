use super::poly::DensePoly;
use crate::compute::{RootCommitSource, RootPolyMeta};
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use jolt_field::Prime128OffsetA7F7 as F;
use jolt_field::{CanonicalEncoding, Ring, Zero};

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

#[test]
fn ring_fold_matches_dense_multiplication_reference() {
    const D: usize = 8;
    let coeffs = (0..2).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_ring_coeffs(coeffs.clone()).unwrap();
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
fn dense_ring_constructor_rejects_empty_and_irregular_sources() {
    const D: usize = 512;
    // 96 rings previously exposed only the first 32 of the supplied rings.
    for num_rings in [0, 3, 96] {
        let result =
            DensePoly::<F>::from_ring_coeffs(vec![CyclotomicRing::<F, D>::zero(); num_rings]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }
    let zero_degree = DensePoly::<F>::from_ring_coeffs(vec![CyclotomicRing::<F, 0>::zero()]);
    assert!(matches!(zero_degree, Err(AkitaError::InvalidInput(_))));
    let odd_degree = DensePoly::<F>::from_ring_coeffs(vec![CyclotomicRing::<F, 3>::zero(); 2]);
    assert!(matches!(odd_degree, Err(AkitaError::InvalidInput(_))));
}

#[test]
fn dense_ring_constructor_preserves_the_entire_commitment_source_and_bound_scan() {
    const D: usize = 512;
    let modulus = (-F::from_u64(1)).to_u128_checked().unwrap() + 1;
    // Exercise both small-i8 and full-field storage, including physical padding.
    for num_rings in [1, 32, 64] {
        for reach in [127u64, 128] {
            let mut evals = (0..num_rings * D)
                .map(|idx| F::from_u64((idx % 32) as u64))
                .collect::<Vec<_>>();
            evals[num_rings * D - 2] = -F::from_u64(reach);
            evals[num_rings * D - 1] = F::from_u64(reach);
            let rings = evals
                .chunks_exact(D)
                .map(|chunk| CyclotomicRing::<F, D>::from_coefficients(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let poly = DensePoly::<F>::from_ring_coeffs(rings.clone()).unwrap();

            assert_eq!(1usize << RootPolyMeta::num_vars(&poly), evals.len());
            assert_eq!(&poly.field_coeffs()[..evals.len()], evals);
            assert_eq!(poly.ring_coeffs::<D>().unwrap(), rings);
            <DensePoly<F> as RootCommitSource<F, D>>::commit_view(&poly).unwrap();
            assert_eq!(
                <DensePoly<F> as RootCommitSource<F, D>>::committed_centered_reach(
                    &poly,
                    modulus,
                    modulus / 2,
                )
                .unwrap(),
                (u128::from(reach), u128::from(reach)),
            );
            assert_eq!(
                poly,
                DensePoly::from_field_evals(RootPolyMeta::num_vars(&poly), evals).unwrap(),
            );
        }
    }
}

#[test]
fn dense_field_constructor_rejects_unrepresentable_arities() {
    for num_vars in [usize::BITS as usize, usize::MAX] {
        let result = DensePoly::<F>::from_field_evals(num_vars, vec![F::zero()]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn dense_field_constructor_rejects_arity_that_truncates_to_a_valid_shift() {
    let result = DensePoly::<F>::from_field_evals((1usize << 32) + 14, vec![F::zero(); 1 << 14]);
    assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
}
