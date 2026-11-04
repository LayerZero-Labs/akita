//! Stage 3 on typed prefix weight blocks against Stage 3 on the dense weight.

use super::product_table::{SetupProductSource, SetupProductSumcheck};
use akita_sumcheck::SumcheckKernel;
use akita_types::binary_source::{root_weight_fixtures, RootFixtureField as F};
use jolt_field::Ring;
use std::sync::Arc;

#[test]
fn structured_typed_prefix_stage3_matches_the_dense_weight_round_by_round() {
    for fixture in root_weight_fixtures() {
        let stage3 = fixture.stage3;
        let setup: Arc<[F]> = stage3.setup.into();
        let num_vars = setup.len().trailing_zeros() as usize;
        let mut expected = SetupProductSumcheck::<F, F>::new(
            SetupProductSource::Owned(Arc::clone(&setup)),
            setup.len(),
            stage3.weights,
            vec![F::from_u64(1)],
        )
        .unwrap();
        let mut actual = SetupProductSumcheck::<F, F>::structured(
            SetupProductSource::Owned(setup),
            Box::new(stage3.blocks),
        )
        .unwrap();
        assert_eq!(
            actual.input_claim(),
            expected.input_claim(),
            "{}",
            fixture.name
        );
        assert_eq!(actual.num_rounds(), num_vars, "{}", fixture.name);
        assert_eq!(expected.num_rounds(), num_vars, "{}", fixture.name);
        let mut claim = expected.input_claim();
        for round in 0..num_vars {
            let polynomial = expected.round_polynomial(round, claim).unwrap();
            assert_eq!(
                actual.round_polynomial(round, claim).unwrap(),
                polynomial,
                "{} round {round}",
                fixture.name
            );
            let challenge = F::from_u64(3_000_017 + 131 * round as u64);
            claim = polynomial.evaluate(challenge);
            expected.bind_challenge(round, challenge).unwrap();
            actual.bind_challenge(round, challenge).unwrap();
        }
        expected.finish().unwrap();
        actual.finish().unwrap();
        assert_eq!(
            actual.folded_table_value().unwrap(),
            expected.folded_table_value().unwrap(),
            "{}",
            fixture.name
        );
    }
}
