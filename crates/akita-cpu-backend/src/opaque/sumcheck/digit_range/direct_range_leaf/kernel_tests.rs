use super::*;
use jolt_field::Prime128Offset275 as F;

#[test]
fn kernel_sequences_and_lookahead_match_live_geometry() {
    use RoundKernel::{Dense as D, LivePrefix as L, OctetPrefix as O};
    for (live, cols, ring, expected) in [
        (1, 2, 1, vec![L, L, L]),
        (3, 2, 1, vec![L, L, D]),
        (4, 2, 1, vec![D, D, D]),
        (1, 3, 4, vec![O, O, O, O, L, L, L]),
        (3, 2, 5, vec![O, O, O, O, L, L, D]),
        (4, 2, 5, vec![O, O, O, O, D, D, D]),
        (1, 5, 1, vec![O, O, O, O, L, L]),
        (1, 0, 4, vec![O, O, O, O]),
    ] {
        let tau = vec![F::from_u64(7); cols + ring];
        let mut prover = LowBasisRangeCheckProver::new(
            PackedSignedDigits::from_i8_digits_auto(vec![1; live << ring]),
            &tau,
            DigitRangePlan::new(4).unwrap(),
            live,
            cols,
            ring,
        )
        .unwrap();
        if let LowBasisRangeImageStorage::OctetPrefix(prefix) = &prover.range_image {
            assert!(prefix.state.is_none());
        }
        for (round, &kernel) in expected.iter().enumerate() {
            assert_eq!(prover.round_kernel(round), kernel);
            let next = prover.round_kernel(round + 1);
            prover.compute_round_eq_factored(round);
            prover.ingest_challenge(round, F::from_u64(11));
            assert_eq!(prover.round_kernel(round + 1), next);
        }
        assert!(matches!(
            prover.range_image,
            LowBasisRangeImageStorage::Materialized(_)
        ));
    }
}
