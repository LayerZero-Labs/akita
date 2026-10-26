use super::*;
use crate::commitment::{AcceptedReach, SignedBlockDecoder, SignedBlocks};
use crate::opaque::{
    CpuBackend, DecomposeFoldPlan, OpeningFoldKernel, OpeningFoldPlan, RootOpeningSource,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPlan,
};
use crate::sources::block_digits::{BlockDigits, DigitShape, I16Rows, I8Rows, RingValues};
use crate::sources::signed_blocks::{Decoded, I16Slice, I8Slice, SignedBlockDigits, SignedStorage};
use akita_challenges::SparseChallenge;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Decoder<'a> {
    values: &'a [i8],
    calls: AtomicUsize,
}
impl SignedBlockDecoder for Decoder<'_> {
    fn decode_i8(&self, start: usize, out: &mut [i8]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        out.copy_from_slice(&self.values[start..start + out.len()]);
    }
}

#[test]
fn signed_block_opening_and_packing_match_field_dense() {
    const D: usize = 64;
    const VARS: usize = 7;
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let geometry = akita_params::SubringCoefficientPackingGeometry::try_new(1, D, D).unwrap();
    let tables = akita_types::PreparedSubringCoefficientPackingPoint::new(
        geometry,
        akita_params::BasisMode::Lagrange,
        2,
        4,
        VARS,
        &(0..VARS)
            .map(|i| F::from_u64(i as u64 + 2))
            .collect::<Vec<_>>(),
    )
    .and_then(akita_types::SubringCoefficientPackingTables::new)
    .unwrap();
    for len in [128, 71] {
        let i8_values = (0..len).map(|i| (i % 13) as i8 - 6).collect::<Vec<_>>();
        let i16_values = i8_values.iter().copied().map(i16::from).collect::<Vec<_>>();
        let decoder = Decoder {
            values: &i8_values,
            calls: AtomicUsize::new(0),
        };
        let mut fields = i8_values
            .iter()
            .copied()
            .map(F::from_i8)
            .collect::<Vec<_>>();
        fields.resize(1 << VARS, F::zero());
        let poly = DensePoly::from_field_evals(VARS, fields).unwrap();
        let dense_view = <DensePoly<F> as RootOpeningSource<F, D>>::opening_view(&poly).unwrap();
        let challenges = [SparseChallenge {
            positions: vec![0, 3].into(),
            coeffs: vec![1, -1].into(),
        }];
        let blocks = [
            SignedBlocks::i8(&i8_values, VARS).unwrap(),
            SignedBlocks::i16(&i16_values, VARS).unwrap(),
            SignedBlocks::decoded_i8(&decoder, len, VARS).unwrap(),
        ];
        for block in blocks {
            for num_digits in [1, 3] {
                let plan = DecomposeFoldPlan {
                    challenges: &challenges,
                    num_positions_per_block: 2,
                    num_digits,
                    log_basis: 8,
                };
                let expected = <CpuBackend<F, F> as OpeningFoldKernel<_, F, D>>::decompose_fold(
                    &backend, None, dense_view, plan,
                )
                .unwrap();
                let actual = <CpuBackend<F, F> as OpeningFoldKernel<_, F, D>>::decompose_fold(
                    &backend, None, block, plan,
                )
                .unwrap();
                assert_eq!(
                    actual.centered_coeffs_flat(),
                    expected.centered_coeffs_flat()
                );
            }
            let block_weights = [F::from_u64(7)];
            let position_weights = [F::from_u64(2), F::from_u64(3)];
            let plan = OpeningFoldPlan::Base {
                live_block_weights: &block_weights,
                position_weights: &position_weights,
                num_positions_per_block: 2,
            };
            let expected = <CpuBackend<F, F> as OpeningFoldKernel<_, F, D>>::evaluate_and_fold(
                &backend, None, dense_view, plan,
            )
            .unwrap();
            let actual = <CpuBackend<F, F> as OpeningFoldKernel<_, F, D>>::evaluate_and_fold(
                &backend, None, block, plan,
            )
            .unwrap();
            assert_eq!(actual.eval, expected.eval);
            assert_eq!(actual.folded, expected.folded);
            let refs = [&poly];
            let dense_batch =
                <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(&refs).unwrap();
            let expected = <CpuBackend<F, F> as SubringCoefficientPackingBatchKernel<
                    _,
                    F,
                    F,
                    D,
                >>::coefficient_packing_partials_batch(
                    &backend,
                    None,
                    dense_batch,
                    SubringCoefficientPackingPlan { tables: &tables },
                )
                .unwrap();
            let actual = <CpuBackend<F, F> as SubringCoefficientPackingBatchKernel<
                    _,
                    F,
                    F,
                    D,
                >>::coefficient_packing_partials_batch(
                    &backend,
                    None,
                    vec![block],
                    SubringCoefficientPackingPlan { tables: &tables },
                )
                .unwrap();
            assert_eq!(actual[0].coordinates(), expected[0].coordinates());
        }
    }
}

fn rejected_fill<St: SignedStorage>(storage: St, len: usize, extrema: Option<(i64, i64)>)
where
    [St::Value; 8]: RingValues<F, 8>,
{
    let source = SignedBlockDigits::<F, _, 8>::new(
        storage,
        4,
        len,
        Some(AcceptedReach {
            modulus: (-F::from_u64(1)).to_u128_checked().unwrap() + 1,
            centering_threshold: akita_params::field_modulus::<F>().unwrap() / 2,
            negative_abs: Some(5),
            positive: Some(5),
        }),
        extrema,
    )
    .unwrap();
    for num_digits in [1, 3] {
        let shape = DigitShape {
            num_digits,
            log_basis: 8,
        };
        let mut scratch = <_ as BlockDigits<8>>::scratch(&source, shape, 3);
        let expected = AkitaError::InvalidInput(
            "committed source exceeds the scheduled coefficient bound".into(),
        );
        assert_eq!(
            <_ as I8Rows<8>>::i8_rows(&source, 0, 0, 0..2, shape, &mut scratch).unwrap_err(),
            expected
        );
        assert_eq!(
            <_ as I16Rows<8>>::i16_rows(&source, 0..2, shape, &mut scratch).unwrap_err(),
            expected
        );
        assert!(
            matches!(<_ as BlockDigits<8>>::digit_planes(&source, 0, shape, &mut scratch), Err(AkitaError::InvalidInput(message)) if message == "committed source exceeds the scheduled coefficient bound")
        );
    }
}

#[test]
fn signed_block_fill_rejects_each_storage_and_borrowed_rows() {
    for value in [-6, 6] {
        for len in [16, 9] {
            let mut values = vec![0i8; len];
            values[7] = value;
            let wide = values.iter().copied().map(i16::from).collect::<Vec<_>>();
            let decoder = Decoder {
                values: &values,
                calls: AtomicUsize::new(0),
            };
            for extrema in [
                None,
                Some((i64::from(value.min(0)), i64::from(value.max(0)))),
            ] {
                rejected_fill(I8Slice(&values), len, extrema);
                rejected_fill(I16Slice(&wide), len, extrema);
                rejected_fill(Decoded(&decoder), len, extrema);
            }
        }
    }
    for value in [-300i16, 300] {
        let mut values = [0; 16];
        values[7] = value;
        for extrema in [
            None,
            Some((i64::from(value.min(0)), i64::from(value.max(0)))),
        ] {
            rejected_fill(I16Slice(&values), values.len(), extrema);
        }
    }
}

#[test]
fn signed_block_rows_borrow_full_rings_copy_tail_and_pad_i16() {
    const D: usize = 8;
    let values = (0..13).map(|i| i as i8 - 6).collect::<Vec<_>>();
    let source =
        SignedBlockDigits::<F, _, D>::new(I8Slice(&values), 4, values.len(), None, None).unwrap();
    let shape = DigitShape {
        num_digits: 1,
        log_basis: 8,
    };
    let mut scratch = <_ as BlockDigits<D>>::scratch(&source, shape, 3);
    let full = <_ as I8Rows<D>>::i8_rows(&source, 0, 0, 0..1, shape, &mut scratch).unwrap();
    assert_eq!(full.as_flattened().as_ptr(), values.as_ptr());
    let tail = <_ as I8Rows<D>>::i8_rows(&source, 0, 0, 0..2, shape, &mut scratch).unwrap();
    assert_ne!(tail.as_flattened().as_ptr(), values.as_ptr());
    assert_eq!(&tail.as_flattened()[..13], values);
    assert_eq!(&tail.as_flattened()[13..], &[0; 3]);
    let wide = <_ as I16Rows<D>>::i16_rows(&source, 0..2, shape, &mut scratch).unwrap();
    assert_eq!(wide.len(), 3);
    assert_eq!(wide[2], [0; D]);
    let decoder = Decoder {
        values: &values,
        calls: AtomicUsize::new(0),
    };
    let source =
        SignedBlockDigits::<F, _, D>::new(Decoded(&decoder), 4, values.len(), None, None).unwrap();
    let mut scratch = <_ as BlockDigits<D>>::scratch(&source, shape, 3);
    <_ as I8Rows<D>>::i8_rows(&source, 0, 0, 0..2, shape, &mut scratch).unwrap();
    assert_eq!(decoder.calls.load(Ordering::Relaxed), 1);
    <_ as I16Rows<D>>::i16_rows(&source, 0..2, shape, &mut scratch).unwrap();
    assert_eq!(decoder.calls.load(Ordering::Relaxed), 2);
}

#[test]
fn signed_block_i16_carrier_boundaries_decompose_like_field_rows() {
    const D: usize = 8;
    let values = [i16::MIN, i16::MAX, -129, 128, -1, 0, 1, 255];
    let bounds = AcceptedReach {
        modulus: (-F::from_u64(1)).to_u128_checked().unwrap() + 1,
        centering_threshold: akita_params::field_modulus::<F>().unwrap() / 2,
        negative_abs: Some(32768),
        positive: Some(32767),
    };
    let source =
        SignedBlockDigits::<F, _, D>::new(I16Slice(&values), 3, D, Some(bounds), None).unwrap();
    let shape = DigitShape {
        num_digits: 3,
        log_basis: 8,
    };
    let mut scratch = <_ as BlockDigits<D>>::scratch(&source, shape, 2);
    let fields = CyclotomicRing::<F, D>::from_coefficients(
        values.map(|value| F::from_i64(i64::from(value))),
    );
    let params = akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params::new(
        shape.num_digits,
        shape.log_basis,
    );
    let mut expected_i8 = vec![[0i8; D]; shape.num_digits];
    fields.balanced_decompose_pow2_i8_into_with_params(&mut expected_i8, &params);
    assert_eq!(
        <_ as I8Rows<D>>::i8_rows(&source, 0, 0, 0..1, shape, &mut scratch).unwrap(),
        expected_i8
    );
    let mut expected_i16 = vec![[0i16; D]; shape.num_digits * 2];
    fields.balanced_decompose_pow2_i16_into(&mut expected_i16[..shape.num_digits], &params);
    assert_eq!(
        <_ as I16Rows<D>>::i16_rows(&source, 0..1, shape, &mut scratch).unwrap(),
        expected_i16
    );
}

#[test]
fn signed_block_wide_i16_fold_matches_field_dense() {
    const D: usize = 64;
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    for len in [128, 71] {
        let values = (0..len)
            .map(|i| [-300i16, 300, -256, 255, -1, 0, 1][i % 7])
            .collect::<Vec<_>>();
        let blocks = SignedBlocks::i16(&values, 7).unwrap();
        let mut fields = values
            .iter()
            .map(|&value| F::from_i64(i64::from(value)))
            .collect::<Vec<_>>();
        fields.resize(128, F::zero());
        let poly = DensePoly::from_field_evals(7, fields).unwrap();
        let dense_view = <DensePoly<F> as RootOpeningSource<F, D>>::opening_view(&poly).unwrap();
        let challenges = [SparseChallenge {
            positions: vec![0, 3].into(),
            coeffs: vec![1, -1].into(),
        }];
        for num_digits in [1, 3] {
            let plan = DecomposeFoldPlan {
                challenges: &challenges,
                num_positions_per_block: 2,
                num_digits,
                log_basis: 10,
            };
            let expected = <CpuBackend<F, F> as OpeningFoldKernel<_, F, D>>::decompose_fold(
                &backend, None, dense_view, plan,
            )
            .unwrap();
            let actual = <CpuBackend<F, F> as OpeningFoldKernel<_, F, D>>::decompose_fold(
                &backend, None, blocks, plan,
            )
            .unwrap();
            assert_eq!(
                actual.centered_coeffs_flat(),
                expected.centered_coeffs_flat()
            );
        }
    }
}

fn check_single_digit<St: SignedStorage>(storage: St, values: &[i16], log_basis: u32)
where
    [St::Value; 8]: RingValues<F, 8>,
{
    use crate::sources::block_digits::DigitPlanes;
    const D: usize = 8;
    let source = SignedBlockDigits::<F, _, D>::new(storage, 4, values.len(), None, None).unwrap();
    let shape = DigitShape {
        num_digits: 1,
        log_basis,
    };
    let mut scratch = <_ as BlockDigits<D>>::scratch(&source, shape, 2);
    let params =
        akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params::<F>::new(1, log_basis);
    let mut fields = [F::zero(); 16];
    for (dst, &value) in fields.iter_mut().zip(values) {
        *dst = F::from_i64(i64::from(value));
    }
    let mut expected8 = Vec::new();
    let mut expected16 = Vec::new();
    for ring in fields.as_chunks::<D>().0 {
        let ring = CyclotomicRing::from_coefficients(*ring);
        let mut row16 = [[0i16; D]];
        ring.balanced_decompose_pow2_i16_into(&mut row16, &params);
        expected16.extend(row16);
        if log_basis <= 8 {
            let mut row8 = [[0i8; D]];
            ring.balanced_decompose_pow2_i8_into_with_params(&mut row8, &params);
            expected8.extend(row8);
        }
    }
    if log_basis <= 8 {
        assert_eq!(
            <_ as I8Rows<D>>::i8_rows(&source, 0, 0, 0..2, shape, &mut scratch).unwrap(),
            expected8
        );
    }
    assert_eq!(
        <_ as I16Rows<D>>::i16_rows(&source, 0..2, shape, &mut scratch).unwrap(),
        expected16
    );
    for ring in 0..2 {
        match <_ as BlockDigits<D>>::digit_planes(&source, ring, shape, &mut scratch).unwrap() {
            DigitPlanes::I8(rows) => assert_eq!(rows, &expected8[ring..ring + 1]),
            DigitPlanes::I16(rows) => assert_eq!(rows, &expected16[ring..ring + 1]),
        }
    }
}

#[test]
fn signed_block_one_digit_reduces_outside_balanced_range() {
    for len in [16, 13] {
        let values = (0..len)
            .map(|i| [-128i8, -65, -64, -1, 0, 63, 64, 100, 127][i % 9])
            .collect::<Vec<_>>();
        let wide = values.iter().copied().map(i16::from).collect::<Vec<_>>();
        let decoder = Decoder {
            values: &values,
            calls: AtomicUsize::new(0),
        };
        check_single_digit(I8Slice(&values), &wide, 7);
        check_single_digit(I16Slice(&wide), &wide, 7);
        check_single_digit(Decoded(&decoder), &wide, 7);
        assert_eq!(decoder.calls.load(Ordering::Relaxed), 4);
        let wide = (0..len)
            .map(|i| [-32768i16, -257, -256, -1, 0, 255, 256, 300, 32767][i % 9])
            .collect::<Vec<_>>();
        check_single_digit(I16Slice(&wide), &wide, 9);
    }
}

#[test]
fn signed_block_centered_bounds_check_actual_values_across_threshold() {
    let modulus = (-F::from_u64(1)).to_u128_checked().unwrap() + 1;
    let values = [0i8, 100, 0, 100, 0, 100, 0, 100];
    let source = SignedBlockDigits::<F, _, 8>::new(
        I8Slice(&values),
        3,
        8,
        Some(AcceptedReach {
            modulus,
            centering_threshold: 50,
            negative_abs: Some(modulus - 100),
            positive: Some(0),
        }),
        None,
    )
    .unwrap();
    let shape = DigitShape {
        num_digits: 1,
        log_basis: 7,
    };
    let mut scratch = <_ as BlockDigits<8>>::scratch(&source, shape, 1);
    assert!(<_ as I8Rows<8>>::i8_rows(&source, 0, 0, 0..1, shape, &mut scratch).is_ok());
}

#[test]
fn signed_block_poly_metadata_and_centered_reach_match_dense() {
    use crate::custom_source::SignedBlockPoly;
    use crate::sources::poly::SourceCoefficients;
    use std::sync::Arc;

    struct OwnedDecoder(Vec<i8>);
    impl SignedBlockDecoder for OwnedDecoder {
        fn decode_i8(&self, start: usize, out: &mut [i8]) {
            out.copy_from_slice(&self.0[start..start + out.len()]);
        }
    }
    let modulus = (-F::from_u64(1)).to_u128_checked().unwrap() + 1;
    for num_vars in [0, 5, 7] {
        let domain = 1 << num_vars;
        for len in [0, domain / 2, domain] {
            let values = (0..len)
                .map(|i| [-128i8, -1, 0, 1, 100, 127][i % 6])
                .collect::<Vec<_>>();
            let mut fields = values.iter().copied().map(F::from_i8).collect::<Vec<_>>();
            fields.resize(domain, F::zero());
            let dense = DensePoly::from_field_evals(num_vars, fields.clone()).unwrap();
            for signed in [
                SignedBlockPoly::i8(values.clone(), num_vars).unwrap(),
                SignedBlockPoly::i16(values.iter().copied().map(i16::from).collect(), num_vars)
                    .unwrap(),
                SignedBlockPoly::decoded_i8(Arc::new(OwnedDecoder(values.clone())), len, num_vars)
                    .unwrap(),
            ] {
                let actual = <_ as CommitmentSource<F>>::descriptor(&signed).unwrap();
                let expected = dense.descriptor().unwrap();
                assert_eq!(actual.num_vars(), expected.num_vars());
                assert_eq!(
                    actual.total_coefficient_len(),
                    expected.total_coefficient_len()
                );
                assert_eq!(
                    actual.live_coefficient_len(),
                    expected.live_coefficient_len()
                );
                assert_eq!(actual.class(), expected.class());
                assert_eq!(
                    <_ as SourceCoefficients<F>>::source_coefficients(&signed)
                        .unwrap()
                        .as_ref(),
                    fields
                );
                for threshold in [modulus / 2, modulus / 2, 0, 50] {
                    assert_eq!(
                        <_ as CommitmentSource<F>>::committed_centered_reach(
                            &signed, modulus, threshold
                        )
                        .unwrap(),
                        dense.committed_centered_reach(modulus, threshold).unwrap(),
                    );
                }
            }
        }
    }
}
