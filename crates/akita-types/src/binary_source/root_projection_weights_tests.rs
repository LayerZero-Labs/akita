use super::*;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, BinarySourceDescriptor};
use akita_algebra::ring::trinomial::signed::signed_trinomial_product;
use akita_challenges::SparseChallenge;
use jolt_field::{One, Prime128Offset275 as F, Ring, Zero};

#[test]
fn projection_direct_weights_match_integer_residual_for_arbitrary_coefficient_rows() {
    let descriptor = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
        vec![[1; 32], [9; 32]],
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap();
    let plan = BinaryRootWitnessPlan::new(&descriptor, 3, 256).unwrap();
    let challenges = Challenges::from_sparse(
        (0..4)
            .map(|column| SparseChallenge {
                positions: vec![0, 81, 161].into(),
                coeffs: vec![1, -1, if column == 0 { 1 } else { -1 }].into(),
            })
            .collect(),
        2,
        2,
    )
    .unwrap();
    let matrix: Vec<Vec<[i128; 648]>> = (0..3)
        .map(|i| {
            vec![std::array::from_fn(|t| {
                ((t * 7 + i * 13) % 101) as i128 - 50
            })]
        })
        .collect();
    let rho: Vec<[F; 648]> = (0..6)
        .map(|i| std::array::from_fn(|t| F::from_u64((t * 17 + i * 29 + 1) as u64)))
        .collect();
    let mut natural = plan
        .repacking()
        .segments()
        .iter()
        .map(|segment| {
            vec![
                0i8;
                segment.elements * segment.digits * segment.geometry.polynomial_modulus_dimension()
            ]
        })
        .collect::<Vec<_>>();
    for (segment, values) in natural.iter_mut().take(2).enumerate() {
        for (index, digit) in values.iter_mut().enumerate() {
            *digit = ((index * 3 + segment * 5) % 8) as i8 - 4;
        }
    }
    let recompose = |segment: usize, element: usize, digits: usize| {
        std::array::from_fn::<_, 648, _>(|t| {
            (0..digits)
                .map(|d| {
                    i128::from(natural[segment][(element * digits + d) * 648 + t])
                        * 8i128.pow(d as u32)
                })
                .sum::<i128>()
        })
    };
    let scalar = binary_scalar_challenges(&descriptor, &challenges).unwrap();
    let mut expected = F::zero();
    for owner in 0..2 {
        let response = recompose(0, owner, 4);
        for (i, row) in matrix.iter().enumerate() {
            let mut residual =
                signed_trinomial_product::<648, MinusTrinomial>(&row[0], &response).unwrap();
            for (column, scalar) in scalar[owner * 2..(owner + 1) * 2].iter().enumerate() {
                let challenge =
                    pack_signed_components(&[*scalar, [0; 162], [0; 162], [0; 162]]).unwrap();
                let image = recompose(1, (owner * 2 + column) * 3 + i, 8);
                let product =
                    signed_trinomial_product::<648, MinusTrinomial>(&challenge, &image).unwrap();
                for (value, term) in residual.iter_mut().zip(product) {
                    *value -= term;
                }
            }
            for (&weight, value) in rho[owner * 3 + i].iter().zip(residual) {
                expected += weight * F::from_i128(value);
            }
        }
    }
    let weights = binary_direct_a_weights(&plan, Some(&matrix), &challenges, &rho).unwrap();
    let table = plan
        .repacking()
        .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap()
        .into_iter()
        .map(|x| F::from_i64(i64::from(x)))
        .collect::<Vec<_>>();
    let terms = weights
        .materialize()
        .unwrap()
        .into_iter()
        .enumerate()
        .filter(|(_, weight)| !weight.is_zero())
        .collect();
    let relation = NativeRootWeights::new(table.len(), terms, expected).unwrap();
    relation.check_table(&table).unwrap();
    let mut changed = table.clone();
    let address = plan.repacking().coefficient_address(1, 0, 0, 647).unwrap();
    changed[address] += F::one();
    assert!(relation.check_table(&changed).is_err());
    assert!(binary_direct_a_weights(&plan, Some(&matrix), &challenges, &rho[..2]).is_err());
}
