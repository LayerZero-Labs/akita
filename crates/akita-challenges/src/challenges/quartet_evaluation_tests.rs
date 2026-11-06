use super::*;
use jolt_field::{One, Prime128OffsetA7F7 as F, Zero};

#[test]
fn cached_quartets_match_sparse_evaluation_across_embeddings_and_extreme_points() {
    let config = SparseChallengeConfig {
        count_pm1: 31,
        count_pm2: 10,
    };
    let source = super::sampled_values::decode(0x145ab, 64, 1024, &config);
    let batch = Challenges::from_sampled(source, 1024, 1, 64, config).unwrap();
    for stride in [1, 4] {
        let embedded = batch
            .embed_subring_positions(64, stride, 64 * stride)
            .unwrap();
        for alpha in [F::zero(), F::one(), -F::one(), F::from_u64(239)] {
            let powers = (0..64 * stride)
                .scan(F::one(), |power, _| {
                    let value = *power;
                    *power *= alpha;
                    Some(value)
                })
                .collect::<Vec<_>>();
            let expected = embedded
                .as_slice()
                .iter()
                .map(|challenge| challenge.eval_at_pows::<F, F>(&powers).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(embedded.evals_at_pows::<F, F>(&powers).unwrap(), expected);
            assert_eq!(embedded.evals_at_pows::<F, F>(&powers).unwrap(), expected);
        }
    }
}
