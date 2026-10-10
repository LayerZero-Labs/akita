use super::{
    kernels, portable_fold_in_place, portable_round_product, LimbView, PackedBinary162,
    PARALLEL_CHUNK,
};
use crate::binary::BinaryField162 as F;

fn random_values(len: usize, mut state: u64) -> Vec<F> {
    (0..len)
        .map(|_| {
            let words: [u64; 3] = std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            });
            F([words[0], words[1], words[2] & F::TOP_MASK])
        })
        .collect()
}

#[test]
fn parallel_round_and_fold_match_sequential_dispatch_in_local_pools() {
    let pools: Vec<_> = [1, 3, 8]
        .into_iter()
        .map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        })
        .collect();
    let lhs = random_values(3 * PARALLEL_CHUNK + 3, 0x8371_623a_2b5c_d947);
    let rhs = random_values(lhs.len(), 0xa019_d843_49ce_728b);
    let challenge = F([0x93ab_2f41, u64::MAX, F::TOP_MASK]);
    // Deliberately unrelated to the table's dot product: each chunk must
    // contribute zero hint, and the public hint must be added exactly once.
    let claim = F([0x41eb_289a_63df_f572, 0x0194_793b_eeee_58ad, 3]);
    for len in [
        0,
        1,
        2,
        PARALLEL_CHUNK - 1,
        PARALLEL_CHUNK,
        PARALLEL_CHUNK + 1,
        2 * PARALLEL_CHUNK - 1,
        2 * PARALLEL_CHUNK,
        2 * PARALLEL_CHUNK + 1,
        3 * PARALLEL_CHUNK,
        3 * PARALLEL_CHUNK + 1,
        3 * PARALLEL_CHUNK + 3,
    ] {
        let lhs = PackedBinary162::from_scalars(&lhs[..len]);
        let rhs = PackedBinary162::from_scalars(&rhs[..len]);
        for pool in &pools {
            pool.install(|| {
                let mut folded_lhs = lhs.clone();
                let mut folded_rhs = rhs.clone();
                let mut expected_lhs = lhs.clone();
                let mut expected_rhs = rhs.clone();
                let pointers = [
                    folded_lhs.words.each_ref().map(|words| words.as_ptr()),
                    folded_rhs.words.each_ref().map(|words| words.as_ptr()),
                ];
                let capacities = [
                    folded_lhs.words.each_ref().map(|words| words.capacity()),
                    folded_rhs.words.each_ref().map(|words| words.capacity()),
                ];
                loop {
                    let n = expected_lhs.len();
                    let expected_round = (n >= 2).then(|| {
                        (kernels().round_product)(
                            &LimbView::new(expected_lhs.words.each_ref().map(|w| w.as_slice())),
                            &LimbView::new(expected_rhs.words.each_ref().map(|w| w.as_slice())),
                            claim,
                        )
                    });
                    assert_eq!(
                        folded_lhs.round_product(&folded_rhs, claim),
                        expected_round,
                        "initial length {len}, remaining {n}"
                    );
                    for (i, (folded, expected)) in [&mut folded_lhs, &mut folded_rhs]
                        .into_iter()
                        .zip([&mut expected_lhs, &mut expected_rhs])
                        .enumerate()
                    {
                        if n > 1 {
                            (kernels().fold_in_place)(
                                &mut LimbView::new(
                                    expected.words.each_mut().map(|w| w.as_mut_slice()),
                                ),
                                challenge,
                            );
                            expected.truncate(n.div_ceil(2));
                        }
                        folded.fold_in_place(challenge);
                        assert_eq!(folded.len(), n.div_ceil(2), "initial length {len}");
                        assert_eq!(
                            folded.words, expected.words,
                            "initial length {len}, remaining {n}"
                        );
                        assert_eq!(folded.words.each_ref().map(|w| w.as_ptr()), pointers[i]);
                        assert_eq!(folded.words.each_ref().map(|w| w.capacity()), capacities[i]);
                    }
                    if n <= 1 {
                        break;
                    }
                }
            });
        }
    }
}

#[test]
fn parallel_odd_chunk_matches_portable_sequence() {
    let len = PARALLEL_CHUNK + 3;
    let mut lhs = PackedBinary162::from_scalars(&random_values(len, 0x8371_623a_2b5c_d947));
    let mut rhs = PackedBinary162::from_scalars(&random_values(len, 0xa019_d843_49ce_728b));
    let mut portable_lhs = lhs.clone();
    let mut portable_rhs = rhs.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let r = F([0x93ab_2f41, u64::MAX, F::TOP_MASK]);
    let claim = F([41, 29, 3]);
    pool.install(|| {
        while lhs.len() > 1 {
            assert_eq!(
                lhs.round_product(&rhs, claim),
                Some(portable_round_product(
                    &LimbView::new(portable_lhs.words.each_ref().map(|w| w.as_slice())),
                    &LimbView::new(portable_rhs.words.each_ref().map(|w| w.as_slice())),
                    claim
                ))
            );
            let new_len = lhs.len().div_ceil(2);
            for portable in [&mut portable_lhs, &mut portable_rhs] {
                portable_fold_in_place(
                    &mut LimbView::new(portable.words.each_mut().map(|w| w.as_mut_slice())),
                    r,
                );
                portable.truncate(new_len);
            }
            lhs.fold_in_place(r);
            rhs.fold_in_place(r);
            assert_eq!(lhs.words, portable_lhs.words);
            assert_eq!(rhs.words, portable_rhs.words);
        }
    });
}
