use super::*;
use akita_algebra::offset_eq::eq_eval_at_index;
use jolt_field::{Ext2, One, Prime64Offset59, Ring, Zero};
use std::time::Instant;

type E = Ext2<Prime64Offset59>;

fn sample(width: usize, rows: usize, first: usize) -> PackedDirect<E> {
    PackedDirect {
        coefficient_bits: 0,
        role_bits: 0,
        digit_stride: 1,
        width,
        digits: (0..width)
            .map(|i| E::from_u64((i + 2) as u64))
            .collect::<Vec<_>>()
            .into(),
        plane_basis: vec![E::one()],
        scalar: E::from_u64(17),
        units: vec![Unit {
            first,
            rows,
            bases: vec![(11, E::from_u64(7)), (width * (rows + 2), E::from_u64(13))],
        }],
    }
}

fn point(bits: usize, seed: u64) -> Vec<E> {
    (0..bits)
        .map(|i| {
            E::from_base_fn(|coordinate| {
                Prime64Offset59::from_u64(seed + i as u64 + 3 * coordinate as u64)
            })
        })
        .collect()
}

fn streamed(p: &PackedDirect<E>, basis: BasisMode, left: &[E], right: &[E]) -> E {
    let blocks = basis_weights(left, basis).unwrap();
    let digits = PreparedDigits::<E, E>::new(right, &p.digits, &[])
        .unwrap()
        .unwrap();
    let mut result = E::zero();
    for unit in &p.units {
        for &(base, weight) in &unit.bases {
            let mut value = E::zero();
            digits
                .accumulate(&mut value, base, unit.first, unit.rows, &blocks)
                .unwrap();
            result += weight * value;
        }
    }
    p.scalar * result
}

#[test]
fn interval_contraction_matches_independent_expansion() {
    for width in [3, 8, 14, 20, 40, 56] {
        let p = sample(width, 259, 3);
        let left = point(9, 2);
        let right = point(
            (width * 600).next_power_of_two().trailing_zeros() as usize,
            7,
        );
        for basis in [BasisMode::Lagrange, BasisMode::Monomial] {
            let mut expected = E::zero();
            for unit in &p.units {
                for &(base, weight) in &unit.bases {
                    for i in 0..unit.rows {
                        let block = if basis == BasisMode::Lagrange {
                            eq_eval_at_index(&left, unit.first + i)
                        } else {
                            left.iter().enumerate().fold(E::one(), |v, (bit, &r)| {
                                if (unit.first + i) >> bit & 1 == 1 {
                                    v * r
                                } else {
                                    v
                                }
                            })
                        };
                        for (digit, &gadget) in p.digits.iter().enumerate() {
                            expected += weight
                                * block
                                * gadget
                                * eq_eval_at_index(&right, base + width * i + digit);
                        }
                    }
                }
            }
            assert_eq!(
                p.evaluate(basis, &left, &right).unwrap(),
                p.scalar * expected
            );
            assert_eq!(
                p.evaluate(basis, &left, &right).unwrap(),
                streamed(&p, basis, &left, &right)
            );
        }
    }
}

#[test]
#[ignore = "manual structural optimization benchmark"]
fn benchmark_large_packed_intervals() {
    for width in [8, 24, 40, 56] {
        let p = sample(width, (1usize << 16) - 3, 3);
        let left = point(16, 2);
        let right = point(
            (width * (1usize << 18))
                .next_power_of_two()
                .trailing_zeros() as usize,
            7,
        );
        let start = Instant::now();
        let old = streamed(&p, BasisMode::Lagrange, &left, &right);
        let old_time = start.elapsed();
        let start = Instant::now();
        let new = p.evaluate(BasisMode::Lagrange, &left, &right).unwrap();
        let new_time = start.elapsed();
        assert_eq!(old, new);
        eprintln!(
            "packed width={width} rows={} streamed={old_time:?} contracted={new_time:?}",
            p.units[0].rows
        );
    }
}

#[test]
fn interval_contraction_rejects_out_of_domain_windows() {
    let mut p = sample(8, 259, 3);
    let left = point(9, 2);
    let right = point(13, 7);
    p.units[0].first = 500;
    assert!(p.evaluate(BasisMode::Lagrange, &left, &right).is_err());
    p.units[0].first = 3;
    p.units[0].bases[0].0 = 8191;
    assert!(p.evaluate(BasisMode::Lagrange, &left, &right).is_err());
}
