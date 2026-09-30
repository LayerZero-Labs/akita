use super::*;
use akita_challenges::SparseChallenge;
use jolt_field::Prime128OffsetA7F7 as F;
use jolt_field::Zero;

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

fn sparse_challenge_as_ring<const D: usize>(challenge: &SparseChallenge) -> CyclotomicRing<F, D> {
    let mut coeffs = [F::zero(); D];
    for (&pos, &coeff) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        coeffs[pos as usize] += F::from_i64(i64::from(coeff));
    }
    CyclotomicRing::from_coefficients(coeffs)
}

fn add_ring_product_reference_high_half<const D: usize>(
    quotient: &mut [F],
    challenge: &CyclotomicRing<F, D>,
    ring: &CyclotomicRing<F, D>,
) {
    let rc = ring.coefficients();
    for (p, &c) in challenge.coefficients().iter().enumerate() {
        for s in (D - p)..D {
            quotient[p + s - D] += c * rc[s];
        }
    }
}

#[test]
fn fused_high_half_rows_match_independent_ring_products_for_non_power_of_two_ranks() {
    const D: usize = 8;
    for (row_rank, challenge_count) in [(1, 3), (3, 5), (5, 11)] {
        let sparse = (0..challenge_count)
            .map(|index| match index % 3 {
                0 => SparseChallenge {
                    positions: vec![0, 7].into(),
                    coeffs: vec![1, -1].into(),
                },
                1 => SparseChallenge {
                    positions: vec![2, 4, 6].into(),
                    coeffs: vec![1, 2, -1].into(),
                },
                _ => SparseChallenge {
                    positions: vec![1, 5].into(),
                    coeffs: vec![-1, 1].into(),
                },
            })
            .collect::<Vec<_>>();
        let challenges = Challenges::from_sparse(sparse.clone(), challenge_count, 1).unwrap();
        let relation_rows = (0..challenge_count * row_rank)
            .map(|index| ring::<D>(31 + 17 * index as u64))
            .collect::<Vec<_>>();
        let consistency_rows = (0..challenge_count)
            .map(|index| ring::<D>(901 + 19 * index as u64))
            .collect::<Vec<_>>();
        let got = parallel_high_half_accumulate_a_rows::<F, D>(
            &challenges,
            &relation_rows,
            row_rank,
            Some(&consistency_rows),
        )
        .unwrap();
        let got_without_consistency = parallel_high_half_accumulate_a_rows::<F, D>(
            &challenges,
            &relation_rows,
            row_rank,
            None,
        )
        .unwrap();
        let mut expected = vec![[F::zero(); D]; row_rank + 1];
        let mut expected_without_consistency = vec![[F::zero(); D]; row_rank];
        for (challenge_index, challenge) in sparse.iter().enumerate() {
            let expanded = sparse_challenge_as_ring::<D>(challenge);
            add_ring_product_reference_high_half(
                &mut expected[0],
                &expanded,
                &consistency_rows[challenge_index],
            );
            for row_index in 0..row_rank {
                let ring_index = challenge_index * row_rank + row_index;
                add_ring_product_reference_high_half(
                    &mut expected[row_index + 1],
                    &expanded,
                    &relation_rows[ring_index],
                );
                add_ring_product_reference_high_half(
                    &mut expected_without_consistency[row_index],
                    &expanded,
                    &relation_rows[ring_index],
                );
            }
        }
        assert_eq!(
            got, expected,
            "rank={row_rank}, challenges={challenge_count}"
        );
        assert_eq!(
            got_without_consistency, expected_without_consistency,
            "rank={row_rank}, challenges={challenge_count} without consistency"
        );
    }
}

#[test]
fn physical_quotient_row_preserves_packing_planes_and_rejects_bad_width() {
    let geometry = RelationRowGeometry::new(64, 2).unwrap();
    let coordinates = (0..128)
        .map(|index| F::from_u64(index as u64 + 1))
        .collect::<Vec<_>>();
    let row =
        RelationQuotientOutput::from_physical_coordinates(geometry, coordinates.clone()).unwrap();
    assert_eq!(row.geometry(), geometry);
    assert_eq!(row.coeffs(), coordinates);
    assert!(
        RelationQuotientOutput::from_physical_coordinates(geometry, vec![F::zero(); 64],).is_err()
    );
}

#[test]
fn fused_high_half_rows_reject_malformed_dimensions_and_challenges() {
    const D: usize = 8;
    let rows = [ring::<D>(7)];
    for (positions, coeffs) in [(vec![8], vec![1]), (vec![1, 2], vec![1])] {
        let challenges = Challenges::from_sparse(
            vec![SparseChallenge {
                positions: positions.into(),
                coeffs: coeffs.into(),
            }],
            1,
            1,
        )
        .unwrap();
        assert!(parallel_high_half_accumulate_a_rows::<F, D>(&challenges, &rows, 1, None).is_err());
    }
    let challenges = Challenges::from_sparse(
        vec![SparseChallenge {
            positions: vec![1].into(),
            coeffs: vec![1].into(),
        }],
        1,
        1,
    )
    .unwrap();
    assert!(parallel_high_half_accumulate_a_rows::<F, D>(&challenges, &rows, 0, None).is_err());
    assert!(parallel_high_half_accumulate_a_rows::<F, D>(&challenges, &rows, 2, None).is_err());
    assert!(
        parallel_high_half_accumulate_a_rows::<F, D>(&challenges, &rows, 1, Some(&[])).is_err()
    );
}

#[test]
fn sparse_high_half_streaming_matches_ring_multiplication_reference() {
    const D: usize = 8;
    let sparse = vec![
        SparseChallenge {
            positions: vec![0, 7].into(),
            coeffs: vec![1, -1].into(),
        },
        SparseChallenge {
            positions: vec![2, 4].into(),
            coeffs: vec![1, 2].into(),
        },
        SparseChallenge {
            positions: vec![1].into(),
            coeffs: vec![-1].into(),
        },
        SparseChallenge {
            positions: vec![3, 6].into(),
            coeffs: vec![1, 1].into(),
        },
    ];
    let rings = (0..sparse.len())
        .map(|idx| (idx != 3).then(|| ring::<D>(10 * idx as u64)))
        .collect::<Vec<_>>();
    let challenges = Challenges::from_sparse(sparse.clone(), sparse.len(), 1).unwrap();

    // A missing optional ring contributes exactly the zero ring.
    let rows = rings
        .iter()
        .map(|ring| ring.unwrap_or_else(|| CyclotomicRing::from_coefficients([F::zero(); D])))
        .collect::<Vec<_>>();
    let got = parallel_high_half_accumulate_a_rows::<F, D>(&challenges, &rows, 1, None).unwrap();
    let mut expected = vec![F::zero(); D];
    for (idx, ring) in rings.iter().enumerate() {
        if let Some(ring) = ring {
            let challenge = sparse_challenge_as_ring::<D>(&sparse[idx]);
            add_ring_product_reference_high_half::<D>(&mut expected, &challenge, ring);
        }
    }

    assert_eq!(got[0].as_slice(), expected);
}
