use super::*;

fn assert_onehot_decompose_fold_matches_dense<const D: usize>(
    onehot_k: usize,
    indices: Vec<Option<usize>>,
) {
    type F = Prime24Offset3;
    const POSITIONS_PER_BLOCK: usize = 8;

    let poly = OneHotPoly::<F>::new(onehot_k, indices).unwrap();
    let dense = materialize_onehot_as_dense::<F, D, _>(&poly);
    let num_blocks = poly.num_live_blocks_for(D, POSITIONS_PER_BLOCK).unwrap();
    let challenges = (0..num_blocks)
        .map(|block| SparseChallenge {
            positions: vec![
                (block * 7 % D) as u32,
                ((block * 7 + 5) % D) as u32,
                ((block * 7 + 19) % D) as u32,
            ]
            .into(),
            coeffs: vec![1, -1, 2].into(),
        })
        .collect::<Vec<_>>();

    let expected = dense.decompose_fold::<D>(&challenges, POSITIONS_PER_BLOCK, 3, 3);
    let got = poly.decompose_fold::<D>(&challenges, POSITIONS_PER_BLOCK, 3, 3);

    assert_eq!(got, expected);
}

#[test]
fn direct_indices_match_dense_for_all_chunk_relations() {
    let make_indices = |len: usize, modulus: usize| {
        (0..len)
            .map(|chunk| (chunk % 5 != 2).then_some((chunk * 29 + 7) % modulus))
            .collect::<Vec<_>>()
    };

    assert_onehot_decompose_fold_matches_dense::<64>(16, make_indices(128, 16));
    assert_onehot_decompose_fold_matches_dense::<64>(64, make_indices(32, 64));
    assert_onehot_decompose_fold_matches_dense::<64>(256, make_indices(8, 256));
    assert_onehot_decompose_fold_matches_dense::<256>(256, make_indices(8, 256));
}

#[cfg(feature = "parallel")]
#[test]
fn direct_indices_are_worker_count_invariant() {
    for workers in [1, 2, 4, 8, 16] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(direct_indices_match_dense_for_all_chunk_relations);
    }
}

#[test]
fn batched_direct_indices_match_dense_aggregation() {
    type F = Prime24Offset3;
    const D: usize = 64;
    const POSITIONS_PER_BLOCK: usize = 8;

    let polys = [
        OneHotPoly::<F>::new(
            16,
            (0..128)
                .map(|chunk| (chunk % 7 != 0).then_some((chunk * 11 + 3) % 16))
                .collect(),
        )
        .unwrap(),
        OneHotPoly::<F>::new(
            256,
            (0..8)
                .map(|chunk| (chunk % 3 != 1).then_some((chunk * 43 + 17) % 256))
                .collect(),
        )
        .unwrap(),
    ];
    let challenges = (0..8)
        .map(|block| SparseChallenge {
            positions: vec![(block * 3 % D) as u32, ((block * 3 + 13) % D) as u32].into(),
            coeffs: vec![1, -2].into(),
        })
        .collect::<Vec<_>>();
    let expected = aggregate_witnesses::<F, D>(
        &polys
            .iter()
            .zip(challenges.chunks_exact(4))
            .map(|(poly, challenges)| {
                materialize_onehot_as_dense::<F, D, _>(poly).decompose_fold::<D>(
                    challenges,
                    POSITIONS_PER_BLOCK,
                    2,
                    3,
                )
            })
            .collect::<Vec<_>>(),
    );
    let refs = polys.iter().collect::<Vec<_>>();
    let got =
        OneHotPoly::decompose_fold_batched::<D>(&refs, &challenges, POSITIONS_PER_BLOCK, 2, 3)
            .unwrap();

    assert_eq!(got, expected);
}
