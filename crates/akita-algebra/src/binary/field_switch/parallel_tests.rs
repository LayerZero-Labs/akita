use super::*;
use crate::binary::{BinaryField128 as H128, BinaryField192 as H192};

fn random_words(len: usize, mut state: u64) -> Vec<[u64; 3]> {
    (0..len)
        .map(|_| {
            std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            })
        })
        .collect()
}

fn compare_profile<H>(hosts: &[H], source: &[H::Source], point: &[H])
where
    H: SwitchField + std::fmt::Debug,
    H::Source: std::fmt::Debug,
{
    let chunk = kernels::PARALLEL_CHUNK;
    let batch: Vec<_> = random_words(H::BATCH_BITS, 0x79d0_487e_9123_5abc)
        .into_iter()
        .map(|words| F([words[0], words[1], words[2] & F::TOP_MASK]))
        .collect();
    let rows = row_weights::<H>(&batch).unwrap();
    let pools: Vec<_> = [1, 3]
        .into_iter()
        .map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        })
        .collect();
    for len in [1, 2, chunk - 1, chunk, chunk + 1, 2 * chunk, 3 * chunk + 1] {
        let hosts = &hosts[..len];
        let source = &source[..len];
        // Original scalar definition, independent of chunk shape and SIMD.
        let mut expected_partials = vec![H::Source::default(); 1 << H::BATCH_BITS];
        for (&source, &weight) in source.iter().zip(hosts) {
            for (word_index, mut word) in weight.coordinates().into_iter().enumerate() {
                while word != 0 {
                    expected_partials[word_index * 64 + word.trailing_zeros() as usize] ^= source;
                    word &= word - 1;
                }
            }
        }
        let expected_coefficients: Vec<_> = hosts
            .iter()
            .map(|weight| {
                let mut sum = F::ZERO;
                for (word_index, mut word) in weight.coordinates().into_iter().enumerate() {
                    while word != 0 {
                        sum += rows[word_index * 64 + word.trailing_zeros() as usize];
                        word &= word - 1;
                    }
                }
                sum
            })
            .collect();
        for pool in &pools {
            pool.install(|| {
                assert_eq!(
                    kernels::partials::<H>(source, hosts).unwrap(),
                    expected_partials
                );
                let mut packed = PackedBinary162::new();
                kernels::coefficients::<H>(hosts, &rows, packed.resize_words(len));
                assert_eq!(packed.to_scalars(), expected_coefficients);
            });
        }
    }
    for bits in [0, 1, 8, 14, 15, 16] {
        let source = &source[..1 << bits];
        let mut expected_weights = vec![H::ZERO; source.len()];
        // This is the original sequential host-field dispatch.
        H::equality_weights(&point[..bits], &mut expected_weights);
        let expected_partials = source
            .chunks(chunk)
            .zip(expected_weights.chunks(chunk))
            .map(|(source, weights)| kernels::partials::<H>(source, weights).unwrap())
            .fold(
                vec![H::Source::default(); 1 << H::BATCH_BITS],
                |mut sum, terms| {
                    for (sum, term) in sum.iter_mut().zip(terms) {
                        *sum ^= term;
                    }
                    sum
                },
            );
        let expected_coefficients: Vec<_> = expected_weights
            .chunks(chunk)
            .flat_map(|weights| {
                let mut block = PackedBinary162::new();
                kernels::coefficients::<H>(weights, &rows, block.resize_words(weights.len()));
                block.to_scalars()
            })
            .collect();
        for pool in &pools {
            pool.install(|| {
                let mut scratch = Vec::new();
                let partials =
                    partial_evaluations::<H>(source, &point[..bits], &mut scratch).unwrap();
                assert_eq!(scratch, expected_weights);
                assert_eq!(partials.values(), expected_partials);
                let mut packed = PackedBinary162::new();
                batched_weights(&scratch, &batch, &mut packed).unwrap();
                assert_eq!(packed.to_scalars(), expected_coefficients);
            });
        }
    }
}

#[test]
fn parallel_field_switch_matches_sequential_and_scalar_paths_in_local_pools() {
    let words = random_words(4 * kernels::PARALLEL_CHUNK, 0x19bc_efe4_56d2_07a3);
    let hosts128: Vec<_> = words
        .iter()
        .map(|w| H128::from_words([w[0], w[1]]))
        .collect();
    let source128: Vec<_> = words
        .iter()
        .map(|w| u128::from(w[1]) << 64 | u128::from(w[2]))
        .collect();
    let hosts192: Vec<_> = words.iter().copied().map(H192::from_words).collect();
    let source64: Vec<_> = words.iter().map(|w| w[1] ^ w[2]).collect();
    compare_profile::<H128>(&hosts128, &source128, &hosts128[..16]);
    compare_profile::<H192>(&hosts192, &source64, &hosts192[..16]);
}
