use super::*;
use crate::binary_source::{BinaryCommitmentProfiles, BinarySourceDescriptor};
use akita_params::binary_setup::BinaryFieldProfile;
use jolt_field::{Ext2, ExtField, One, Prime64Offset59 as Base, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

type E = Ext2<Base>;

fn fixture(rows: usize, rank: usize, owners: usize) -> BinaryRootWitnessPlan {
    let descriptor = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 1, 4 * rows, 1).unwrap(),
        (0..owners).map(|owner| [owner as u8 + 1; 32]).collect(),
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap();
    BinaryRootWitnessPlan::for_profile(&descriptor, rank, 64, BinaryFieldProfile::P64, true, 0)
        .unwrap()
}

fn extension(rng: &mut StdRng) -> E {
    E::from_base_fn(|i| Base::from_u64(rng.gen_range(1..100) + i as u64))
}

// Independent signed polynomial long division, including the minus-trinomial
// sign. Contracting each reduced monomial with rho yields the response kernel.
fn long_division_weights(rho: &[E; DEGREE]) -> Vec<E> {
    (0..2 * DEGREE - 1)
        .map(|degree| {
            let mut monomial = vec![0i128; 2 * DEGREE - 1];
            monomial[degree] = 1;
            for exponent in (DEGREE..monomial.len()).rev() {
                let value = monomial[exponent];
                monomial[exponent - DEGREE / 2] += value;
                monomial[exponent - DEGREE] -= value;
                monomial[exponent] = 0;
            }
            monomial[..DEGREE]
                .iter()
                .zip(rho)
                .filter(|(value, _)| **value != 0)
                .fold(E::zero(), |sum, (&value, &weight)| {
                    sum + E::from_i128(value) * weight
                })
        })
        .collect()
}

fn dense_weights(
    plan: &BinaryRootWitnessPlan,
    rho: &[[E; DEGREE]],
    root_point: &[E],
    offset: usize,
    prefix_len: usize,
) -> Vec<E> {
    let rows = BinaryAPrefixLen::new(plan.descriptor().index_map(), plan.rank())
        .unwrap()
        .rows;
    let row_stride = plan.rank().next_power_of_two() * STRIDE;
    let mut dense = vec![E::zero(); prefix_len];
    for (owner, owner_rho) in rho.chunks_exact(plan.rank()).enumerate() {
        for row in 0..rows {
            let lane = (0..DEGREE)
                .map(|coefficient| {
                    (0..4).fold(E::zero(), |sum, digit| {
                        let address =
                            offset + ((owner * rows + row) * 4 + digit) * STRIDE + coefficient;
                        sum + E::from_u64(8u64.pow(digit as u32))
                            * eq_eval_at_index(root_point, address)
                    })
                })
                .collect::<Vec<_>>();
            for (rank, rho) in owner_rho.iter().enumerate() {
                let kernel = long_division_weights(rho);
                for setup_coefficient in 0..DEGREE {
                    dense[row * row_stride + rank * STRIDE + setup_coefficient] += lane
                        .iter()
                        .enumerate()
                        .fold(E::zero(), |sum, (coefficient, &value)| {
                            sum + value * kernel[setup_coefficient + coefficient]
                        });
                }
            }
        }
    }
    dense
}

#[test]
fn translated_a_matches_independent_e64_dense_weights_and_setup_dot_product() {
    let mut rng = StdRng::seed_from_u64(0x00a4_9931);
    // Long carry chains across the coefficient, digit, row, and owner axes.
    for (rows, rank, owners, offset) in [
        (1, 1, 1, 1023),
        (2, 3, 2, 4095),
        (4, 1, 3, 32767),
        (4, 1, 2, 21503),
    ] {
        let plan = fixture(rows, rank, owners);
        let prefix_len = rows * rank.next_power_of_two() * STRIDE * 2;
        let root_len = (offset + owners * rows * 4 * STRIDE).next_power_of_two();
        let root_point = (0..root_len.trailing_zeros())
            .map(|_| extension(&mut rng))
            .collect::<Vec<_>>();
        assert!(root_point
            .iter()
            .all(|x| x.base_coefficient(1) != Base::zero()));
        let rho = (0..owners * rank)
            .map(|_| core::array::from_fn(|_| extension(&mut rng)))
            .collect::<Vec<_>>();
        let product =
            BinaryAProduct::translated(&plan, &rho, &root_point, prefix_len, offset).unwrap();
        let terms = product.factor_terms().unwrap();
        assert!((1..=4).contains(&terms.len()));
        if offset == 21503 {
            assert_eq!(terms.len(), 4);
        }
        let expected = dense_weights(&plan, &rho, &root_point, offset, prefix_len);
        let mut actual = vec![E::zero(); prefix_len];
        for (row, coefficients) in terms {
            for (target, &row) in actual.chunks_exact_mut(coefficients.len()).zip(&row) {
                for (target, &coefficient) in target.iter_mut().zip(&coefficients) {
                    *target += row * coefficient;
                }
            }
        }
        assert_eq!(
            actual, expected,
            "rows={rows}, rank={rank}, offset={offset}"
        );
        // A dense independently sampled setup table exercises every local cell,
        // including padding that must have zero weight.
        let setup = (0..prefix_len)
            .map(|_| E::from_base(Base::from_u64(rng.gen_range(0..100))))
            .collect::<Vec<_>>();
        let dot = |weights: &[E]| {
            weights
                .iter()
                .zip(&setup)
                .fold(E::zero(), |sum, (&f, &s)| sum + f * s)
        };
        assert_eq!(dot(&actual), dot(&expected));
        for boolean_coordinates in [false, true] {
            let point = (0..prefix_len.trailing_zeros())
                .map(|i| {
                    if boolean_coordinates && i % 3 == 0 {
                        E::from_u64(u64::from(i % 2 != 0))
                    } else {
                        extension(&mut rng)
                    }
                })
                .collect::<Vec<_>>();
            let dense_mle = expected
                .iter()
                .enumerate()
                .fold(E::zero(), |sum, (index, &value)| {
                    sum + value * eq_eval_at_index(&point, index)
                });
            assert_eq!(product.evaluate(&point).unwrap(), dense_mle);
        }
        if offset == 4095 {
            assert!(product.factors().is_err());
        }
    }
}

#[test]
fn zero_offset_translation_preserves_aligned_factors_and_evaluation() {
    let mut rng = StdRng::seed_from_u64(0x0023_01a9);
    let plan = fixture(2, 3, 2);
    let prefix_len = 2 * 4 * STRIDE;
    let root_point = (0..plan.repacking().domain_len().trailing_zeros())
        .map(|_| extension(&mut rng))
        .collect::<Vec<_>>();
    let rho = (0..6)
        .map(|_| core::array::from_fn(|_| extension(&mut rng)))
        .collect::<Vec<_>>();
    let original = BinaryAProduct::new(&plan, &rho, &root_point, prefix_len).unwrap();
    let translated = BinaryAProduct::translated(&plan, &rho, &root_point, prefix_len, 0).unwrap();
    assert_eq!(original.factors().unwrap(), translated.factors().unwrap());
    assert_eq!(
        original.factor_terms().unwrap(),
        translated.factor_terms().unwrap()
    );
    let point = (0..prefix_len.trailing_zeros())
        .map(|_| extension(&mut rng))
        .collect::<Vec<_>>();
    assert_eq!(
        original.evaluate(&point).unwrap(),
        translated.evaluate(&point).unwrap()
    );
}

#[test]
fn translated_a_rejects_overflow_domain_caps_and_out_of_domain_response() {
    let plan = fixture(2, 3, 2);
    let rho = vec![[E::one(); DEGREE]; 6];
    let prefix_len = 2 * 4 * STRIDE;
    let point = vec![E::one(); 16];
    for offset in [usize::MAX, usize::MAX - 4096, (1 << 16) - 8192] {
        assert!(BinaryAProduct::translated(&plan, &rho, &point, prefix_len, offset).is_err());
    }
    assert!(BinaryAProduct::translated(&plan, &rho[..5], &point, prefix_len, 1).is_err());
    assert!(BinaryAProduct::translated(&plan, &rho, &point[..11], prefix_len, 1).is_err());
    assert!(BinaryAProduct::translated(&plan, &rho, &[E::one(); 33], prefix_len, 1).is_err());
    if let Some(oversized) = checked::pow2(33) {
        assert!(BinaryAProduct::translated(&plan, &rho, &point, oversized, 1).is_err());
    }
    assert!(BinaryAProduct::translated(&plan, &rho, &point, prefix_len - 1, 1).is_err());
}
