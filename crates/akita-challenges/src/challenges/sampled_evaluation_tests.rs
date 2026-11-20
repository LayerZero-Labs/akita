use super::*;
use jolt_field::{One, Prime128Offset6A23801F as F, Zero};
#[test]
fn bounded_sampled_accumulators_match_naive_at_extreme_and_embedded_points() {
    for (dimension, ones, twos) in [(64, 31, 10), (128, 65, 0)] {
        let challenge = SparseChallenge {
            positions: crate::SparseChallengePositions::from_slice(
                &(0..ones + twos).map(|i| i as u32).collect::<Vec<_>>(),
            ),
            coeffs: crate::SparseChallengeCoefficients::from_slice(
                &(0..ones + twos)
                    .map(|i| (if i < ones { 1i8 } else { 2i8 }) * if i % 3 == 0 { -1 } else { 1 })
                    .collect::<Vec<_>>(),
            ),
        };
        let batch = Challenges::from_sampled(
            vec![challenge.clone()],
            1,
            1,
            dimension,
            SparseChallengeConfig {
                count_pm1: ones,
                count_pm2: twos,
            },
        )
        .unwrap();
        for alpha in [F::zero(), F::one(), -F::one(), F::from_u64(0xdeadbeef)] {
            for stride in [1, 4] {
                let embedded = batch
                    .embed_subring_positions(dimension, stride, dimension * stride)
                    .unwrap();
                let powers = (0..dimension * stride)
                    .scan(F::one(), |v, _| {
                        let out = *v;
                        *v *= alpha;
                        Some(out)
                    })
                    .collect::<Vec<_>>();
                let expected = embedded.as_slice()[0]
                    .eval_at_pows::<F, F>(&powers)
                    .unwrap();
                assert_eq!(
                    embedded.evals_at_pows::<F, F>(&powers).unwrap(),
                    vec![expected]
                );
            }
        }
    }
}
