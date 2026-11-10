use super::*;
use crate::FlatMatrix;
use jolt_field::{Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use rand::{rngs::StdRng, SeedableRng};

fn oracle<F: Field + WithAccumulator, const D: usize>() {
    let mut rng = StdRng::seed_from_u64(0xc03);
    // Cross the 256-column accumulation boundary, including a partial chunk.
    let stride = 519;
    let entries: Vec<CyclotomicRing<F, D>> = (0..4 * stride)
        .map(|_| CyclotomicRing::random(&mut rng))
        .collect();
    let flat = FlatMatrix::from_ring_slice(&entries);
    let view = flat.ring_view::<D>(4, stride).unwrap();
    for columns in [0..1, 3..258, 2..518, 5..5] {
        let window = MatrixWindow {
            ring_d: D,
            stride,
            rows: 1..4,
            columns,
        };
        let mut scales: Vec<F> = (0..window.columns.len())
            .map(|_| F::random(&mut rng))
            .collect();
        for (index, scale) in scales.iter_mut().enumerate() {
            if index % 4 == 0 {
                *scale = F::zero();
            }
        }
        for zero in [false, true] {
            if zero {
                scales.fill(F::zero());
            }
            let expected: Vec<CyclotomicRing<F, D>> = window
                .rows
                .clone()
                .map(|row| {
                    let mut coefficients = [F::zero(); D];
                    for (col, scale) in window.columns.clone().zip(&scales) {
                        for (dst, coefficient) in coefficients
                            .iter_mut()
                            .zip(entries[row * stride + col].coefficients())
                        {
                            *dst += *coefficient * *scale;
                        }
                    }
                    CyclotomicRing::from_coefficients(coefficients)
                })
                .collect();
            assert_eq!(view.combine_columns(&window, &scales).unwrap(), expected);
        }
    }
    let window = MatrixWindow {
        ring_d: D,
        stride,
        rows: 0..4,
        columns: 1..3,
    };
    assert!(view.combine_columns(&window, &[F::one()]).is_err());
    for bad in [
        MatrixWindow {
            ring_d: D + 1,
            ..window.clone()
        },
        MatrixWindow {
            stride: stride + 1,
            ..window.clone()
        },
        MatrixWindow {
            rows: 0..5,
            ..window.clone()
        },
        MatrixWindow {
            columns: 0..stride + 1,
            ..window.clone()
        },
    ] {
        assert!(view.combine_columns(&bad, &[F::one(); 2]).is_err());
    }
}

#[test]
fn column_combinations_match_plain_field_sums() {
    oracle::<Prime32Offset99, 64>();
    oracle::<Prime64Offset59, 64>();
    oracle::<Prime128Offset275, 64>();
    oracle::<Prime128OffsetA7F7, 128>();
}
