use super::{kernels, PackedBinary162, PARALLEL_CHUNK};
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
    let pools: Vec<_> = [1, 3]
        .into_iter()
        .map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        })
        .collect();
    let lhs = random_values(3 * PARALLEL_CHUNK + 1, 0x8371_623a_2b5c_d947);
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
    ] {
        let lhs = PackedBinary162::from_scalars(&lhs[..len]);
        let rhs = PackedBinary162::from_scalars(&rhs[..len]);
        let expected_round = (len >= 2).then(|| (kernels().round_product)(&lhs, &rhs, claim));
        let mut expected_fold = lhs.clone();
        if len > 1 {
            (kernels().fold_in_place)(&mut expected_fold, challenge);
        }
        for pool in &pools {
            pool.install(|| {
                assert_eq!(
                    lhs.round_product(&rhs, claim),
                    expected_round,
                    "length {len}"
                );
                let mut folded = lhs.clone();
                let allocations = folded.words.each_ref().map(|words| words.as_ptr());
                folded.fold_in_place(challenge);
                assert_eq!(folded, expected_fold, "length {len}");
                assert_eq!(
                    folded.words.each_ref().map(|words| words.as_ptr()),
                    allocations
                );
            });
        }
    }
}
