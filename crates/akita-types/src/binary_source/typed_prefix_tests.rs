use super::*;
use akita_algebra::offset_eq::eq_eval_at_index;
use jolt_field::{Prime128Offset275 as F, Ring, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn paired_offset_equality_sums_match_direct_summation() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..300 {
        let a = (0..rng.gen_range(0..9))
            .map(|_| F::from_u64(rng.gen()))
            .collect::<Vec<_>>();
        let b = (0..rng.gen_range(0..9))
            .map(|_| F::from_u64(rng.gen()))
            .collect::<Vec<_>>();
        let (u, v, count) = (
            rng.gen_range(0..300),
            rng.gen_range(0..300),
            rng.gen_range(0..300),
        );
        let direct = (0..count).fold(F::zero(), |sum, q| {
            sum + eq_eval_at_index(&a, u + q) * eq_eval_at_index(&b, v + q)
        });
        assert_eq!(
            paired_offset_eq_sum(&a, u, &b, v, count).unwrap(),
            direct,
            "{u} {v} {count}"
        );
    }
}
