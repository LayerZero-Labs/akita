use super::*;
use crate::compute::CompressionRowsItem;
use crate::kernels::crt_ntt::build_ntt_slot;
use akita_algebra::ntt::tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES};
use akita_field::{Fp64, Prime128Offset275};
use akita_types::layout::FlatMatrix;

#[test]
fn paired_i8_lane_matches_cyclic_negacyclic_quotient_identity() {
    type F = Fp64<4_294_967_197>;
    const D: usize = 64;
    let ProtocolCrtNttParams::Q32(params) =
        select_crt_ntt_params::<F, D>().expect("Q32 test parameters")
    else {
        panic!("test field must use Q32 parameters");
    };
    let lhs =
        CyclotomicRing::<F, D>::from_coefficients(from_fn(|i| F::from_i64((i % 7) as i64 - 3)));
    let rhs_coeffs = [[-1i8; D]];
    let rhs = CyclotomicRing::<F, D>::from_coefficients([F::from_i64(-1); D]);
    let neg = [CyclotomicCrtNtt::<i32, Q32_NUM_PRIMES, D>::from_ring_with_params(&lhs, &params)];
    let cyc = [CyclotomicCrtNtt::<i32, Q32_NUM_PRIMES, D>::from_ring_cyclic(&lhs, &params)];
    let neg_rows = [neg.as_slice()];
    let cyc_rows = [cyc.as_slice()];
    let requests = [PairedI8Request {
        cyclic_rows: &cyc_rows,
        negacyclic_rows: &neg_rows,
        num_rows: 1,
        coeffs: &rhs_coeffs,
        abs_bound: 1,
    }];

    let (cyclic_rows, negacyclic_rows, paired_rows, centered_rows) =
        fused_relation_rows_batch::<F, i32, Q32_NUM_PRIMES, D>(&[], &[], &requests, &[], &params)
            .expect("paired i8 request");
    assert!(cyclic_rows.is_empty() && negacyclic_rows.is_empty() && centered_rows.is_empty());
    let actual = &paired_rows[0];
    let mut cyclic = CyclotomicRing::zero();
    add_cyclic_product_into(&mut cyclic, &lhs, &rhs);
    let expected = quotient_from_cyclic_and_negacyclic(&cyclic, &(lhs * rhs));
    assert_eq!(actual.negacyclic, vec![lhs * rhs]);
    assert_eq!(actual.quotient, vec![expected]);
}

#[test]
fn centered_lane_derives_capacity_when_hint_is_underreported() {
    type F = Fp64<4_294_967_197>;
    const D: usize = 64;
    let ProtocolCrtNttParams::Q32(params) =
        select_crt_ntt_params::<F, D>().expect("Q32 test parameters")
    else {
        panic!("test field must use Q32 parameters");
    };
    let zero = CyclotomicRing::<F, D>::zero();
    let neg = [CyclotomicCrtNtt::<i32, Q32_NUM_PRIMES, D>::from_ring_with_params(&zero, &params)];
    let cyc = [CyclotomicCrtNtt::<i32, Q32_NUM_PRIMES, D>::from_ring_cyclic(&zero, &params)];
    let neg_rows = [neg.as_slice()];
    let cyc_rows = [cyc.as_slice()];
    let coeffs = [[1i32; D]];
    let requests = [PairedCenteredI32Request {
        cyclic_rows: &cyc_rows,
        negacyclic_rows: &neg_rows,
        num_rows: 1,
        coeffs: &coeffs,
        bounds: CenteredRhsBounds { capacity: 0 },
    }];
    let result =
        fused_relation_rows_batch::<F, i32, Q32_NUM_PRIMES, D>(&[], &[], &[], &requests, &params)
            .expect("observed centered bound augments capacity hint");
    assert_eq!(result.3[0], vec![CyclotomicRing::zero()]);
}

#[test]
fn q128_streaming_fallback_matches_oracle_for_heterogeneous_bounds() {
    type F = Prime128Offset275;
    const D: usize = 64;
    let ProtocolCrtNttParams::Q128(params) =
        select_crt_ntt_params::<F, D>().expect("Q128 test parameters")
    else {
        panic!("test field must use Q128 parameters");
    };
    let declared = 127;
    let chunk = safe_crt_chunk_width::<F, i32, Q128_NUM_PRIMES, D>(&params, 4096, declared)
        .expect("one declared-capacity term fits");
    let cols = chunk + 1;
    let lhs = CyclotomicRing::<F, D>::from_coefficients(from_fn(|coefficient| {
        F::from_i64((coefficient % 11) as i64 - 5)
    }));
    let entry = CyclotomicCrtNtt::from_ring_cyclic(&lhs, &params);
    let row = vec![entry; cols];
    let rows = [row.as_slice()];
    let wide_coeffs = vec![[127i8; D]; cols];
    let narrow_coeffs = vec![[-1i8; D]; cols];
    assert_eq!(
        safe_crt_chunk_width::<F, i32, Q128_NUM_PRIMES, D>(&params, cols, declared),
        Some(chunk),
        "wide item must cross its safe-width boundary"
    );
    assert_eq!(
        safe_crt_chunk_width::<F, i32, Q128_NUM_PRIMES, D>(&params, cols, 1),
        Some(cols),
        "narrow item should retain a distinct one-shot width"
    );
    let requests = [
        CyclicI8Request {
            cyclic_rows: &rows,
            num_rows: 1,
            coeffs: &wide_coeffs,
            abs_bound: declared,
            centered_digits: true,
        },
        CyclicI8Request {
            cyclic_rows: &rows,
            num_rows: 1,
            coeffs: &narrow_coeffs,
            abs_bound: 1,
            centered_digits: true,
        },
    ];
    let result =
        fused_relation_rows_batch::<F, i32, Q128_NUM_PRIMES, D>(&requests, &[], &[], &[], &params)
            .expect("heterogeneous streaming fallback");
    for (actual, digit) in result.0.iter().zip([127i64, -1]) {
        let rhs = CyclotomicRing::<F, D>::from_coefficients([F::from_i64(digit); D]);
        let product = {
            let mut out = CyclotomicRing::zero();
            add_cyclic_product_into(&mut out, &lhs, &rhs);
            out
        };
        let expected = (0..cols).fold(CyclotomicRing::zero(), |mut total, _| {
            total += product;
            total
        });
        assert_eq!(actual, &vec![expected]);
    }
}

#[test]
fn oversized_interleaved_compression_batch_preserves_mode_order_across_partitions() {
    type F = Prime128Offset275;
    const D: usize = 8;
    const ROWS: usize = 3;
    let matrix = (0..ROWS)
        .map(|row| {
            CyclotomicRing::<F, D>::from_coefficients(from_fn(|coefficient| {
                F::from_i64((row * D + coefficient + 1) as i64)
            }))
        })
        .collect::<Vec<_>>();
    let flat = FlatMatrix::from_ring_slice(&matrix);
    let slot = build_ntt_slot(flat.ring_view::<D>(ROWS, 1).expect("compression matrix"))
        .expect("Q128/D8 slot");
    let digits = [[-1i8; D]];
    let rhs = CyclotomicRing::<F, D>::from_coefficients([F::from_i64(-1); D]);
    let known_neg = matrix.iter().map(|lhs| *lhs * rhs).collect::<Vec<_>>();
    let expected_quotient = matrix
        .iter()
        .zip(&known_neg)
        .map(|(lhs, neg)| {
            let mut cyclic = CyclotomicRing::zero();
            add_cyclic_product_into(&mut cyclic, lhs, &rhs);
            quotient_from_cyclic_and_negacyclic(&cyclic, neg)
        })
        .collect::<Vec<_>>();

    let element_bytes = slot.cache_bytes() / (slot.total_elements() * 2);
    let max_lanes = FUSED_L2_CACHE_BYTES / (ROWS * element_bytes);
    let groups = max_lanes / 4 + 2;
    let mut items = Vec::with_capacity(groups * 3);
    for _ in 0..groups {
        items.push(CompressionRowsItem {
            digits: &digits,
            digit_abs_bound: 1,
            mode: CompressionRowsMode::NegacyclicOnly,
        });
        items.push(CompressionRowsItem {
            digits: &digits,
            digit_abs_bound: 1,
            mode: CompressionRowsMode::EagerPaired,
        });
        items.push(CompressionRowsItem {
            digits: &digits,
            digit_abs_bound: 1,
            mode: CompressionRowsMode::CyclicWithKnownNeg(&known_neg),
        });
    }
    let requested_lanes = groups * 4;
    assert!(requested_lanes > max_lanes, "test must force partitioning");

    let output = compression_rows_with_slot(
        &slot,
        CompressionRowsPlan {
            row_count: ROWS,
            column_count: 1,
            items: &items,
        },
    )
    .expect("oversized interleaved compression batch");
    assert_eq!(output.len(), items.len());
    for (index, actual) in output.iter().enumerate() {
        match index % 3 {
            0 => {
                assert_eq!(actual.u_neg.as_ref(), Some(&known_neg));
                assert!(actual.quotient.is_none());
            }
            1 => {
                assert_eq!(actual.u_neg.as_ref(), Some(&known_neg));
                assert_eq!(actual.quotient.as_ref(), Some(&expected_quotient));
            }
            2 => {
                assert!(actual.u_neg.is_none());
                assert_eq!(actual.quotient.as_ref(), Some(&expected_quotient));
            }
            _ => unreachable!(),
        }
    }

    let wrong_known = vec![CyclotomicRing::<F, D>::zero(); ROWS - 1];
    let invalid = [CompressionRowsItem {
        digits: &digits,
        digit_abs_bound: 1,
        mode: CompressionRowsMode::CyclicWithKnownNeg(&wrong_known),
    }];
    assert!(compression_rows_with_slot(
        &slot,
        CompressionRowsPlan {
            row_count: ROWS,
            column_count: 1,
            items: &invalid,
        }
    )
    .is_err());
}

#[test]
fn compression_dispatch_validates_only_the_domains_requested_by_modes() {
    type F = Prime128Offset275;
    const D: usize = 8;
    let flat = FlatMatrix::from_ring_slice(&[CyclotomicRing::<F, D>::one()]);
    let slot =
        build_ntt_slot(flat.ring_view::<D>(1, 1).expect("one cache entry")).expect("Q128/D8 slot");
    let digits = [[-1i8; D]];
    let known = [CyclotomicRing::<F, D>::zero()];
    let neg_only: [CompressionRowsItem<'_, F, D>; 1] = [CompressionRowsItem {
        digits: &digits,
        digit_abs_bound: 1,
        mode: CompressionRowsMode::NegacyclicOnly,
    }];
    let deferred = [CompressionRowsItem {
        digits: &digits,
        digit_abs_bound: 1,
        mode: CompressionRowsMode::CyclicWithKnownNeg(&known),
    }];
    let eager: [CompressionRowsItem<'_, F, D>; 1] = [CompressionRowsItem {
        digits: &digits,
        digit_abs_bound: 1,
        mode: CompressionRowsMode::EagerPaired,
    }];
    let plan = |items| CompressionRowsPlan {
        row_count: 1,
        column_count: 1,
        items,
    };

    let NttSlotCache::Q128 {
        neg,
        mut cyc,
        params,
    } = slot.clone()
    else {
        panic!("test field must use Q128 cache");
    };
    cyc.clear();
    let missing_cyclic = NttSlotCache::Q128 { neg, cyc, params };
    compression_rows_with_slot(&missing_cyclic, plan(&neg_only))
        .expect("neg-only mode must not inspect cyclic cache rows");
    assert!(compression_rows_with_slot(&missing_cyclic, plan(&deferred)).is_err());
    assert!(compression_rows_with_slot(&missing_cyclic, plan(&eager)).is_err());

    let NttSlotCache::Q128 {
        mut neg,
        cyc,
        params,
    } = slot
    else {
        panic!("test field must use Q128 cache");
    };
    neg.clear();
    let missing_negacyclic = NttSlotCache::Q128 { neg, cyc, params };
    compression_rows_with_slot(&missing_negacyclic, plan(&deferred))
        .expect("deferred mode must not inspect negacyclic cache rows");
    assert!(compression_rows_with_slot(&missing_negacyclic, plan(&neg_only)).is_err());
    assert!(compression_rows_with_slot(&missing_negacyclic, plan(&eager)).is_err());
}
