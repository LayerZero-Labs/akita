//! Stage 3 on typed prefix weight blocks against Stage 3 on the dense weight.

use super::product_table::{SetupProductSource, SetupProductSumcheck};
use akita_sumcheck::SumcheckKernel;
use akita_types::binary_source::{root_weight_scenarios, RootFixtureField as F};
use jolt_field::Ring;
use std::sync::Arc;

#[test]
fn structured_typed_prefix_stage3_matches_the_dense_weight_round_by_round() {
    for scenario in root_weight_scenarios() {
        let stage3 = scenario.compile_stage3();
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
            scenario.name
        );
        assert_eq!(actual.num_rounds(), num_vars, "{}", scenario.name);
        assert_eq!(expected.num_rounds(), num_vars, "{}", scenario.name);
        let mut claim = expected.input_claim();
        for round in 0..num_vars {
            let polynomial = expected.round_polynomial(round, claim).unwrap();
            assert_eq!(
                actual.round_polynomial(round, claim).unwrap(),
                polynomial,
                "{} round {round}",
                scenario.name
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
            scenario.name
        );
    }
}

#[test]
fn binary_root_p64_typed_stage3_matches_dense_extension_rounds() {
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_types::binary_source::{
        binary_word_manifest_fixture, BinaryAProduct, BinaryRootWitnessPlan, BinaryTypedPrefix,
    };
    use jolt_field::{Ext2, ExtField, One, Prime64Offset59 as Base, Zero};
    type E = Ext2<Base>;
    let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
    let plan = BinaryRootWitnessPlan::for_profile(
        manifest.descriptor(),
        3,
        64,
        BinaryFieldProfile::P64,
        true,
        0,
    )
    .unwrap();
    let typed = BinaryTypedPrefix::new(&manifest, true, 0, 64).unwrap();
    let extension = |i: usize| E::from_base_fn(|c| Base::from_u64((17 + i * 31 + c * 43) as u64));
    let point = (0..plan.repacking().domain_len().trailing_zeros())
        .map(|i| extension(i as usize))
        .collect::<Vec<_>>();
    let rho = (0..3)
        .map(|row| std::array::from_fn(|k| extension(row * 648 + k)))
        .collect::<Vec<_>>();
    let a = BinaryAProduct::new(&plan, &rho, &point, typed.domain_len).unwrap();
    let batches = [
        vec![extension(3000); manifest.b_rows()],
        vec![extension(4000); manifest.d_label(true, 0).unwrap().rank as usize],
    ];
    let alpha = extension(5000);
    let dense = typed
        .prover_weights(
            &a,
            &point,
            alpha,
            [&batches[0], &batches[1]],
            typed.domain_len,
        )
        .unwrap();
    let blocks = typed
        .prover_weight_blocks(
            &a,
            &point,
            alpha,
            [&batches[0], &batches[1]],
            typed.domain_len,
        )
        .unwrap();
    let setup: Arc<[Base]> = typed
        .setup_coefficients(&[41; 32], &manifest, typed.domain_len)
        .unwrap()
        .into();
    let mut expected = SetupProductSumcheck::<Base, E>::new(
        SetupProductSource::Owned(Arc::clone(&setup)),
        setup.len(),
        dense,
        vec![E::one()],
    )
    .unwrap();
    let mut actual = SetupProductSumcheck::<Base, E>::structured(
        SetupProductSource::Owned(setup),
        Box::new(blocks),
    )
    .unwrap();
    let mut claim = actual.input_claim();
    assert_eq!(claim, expected.input_claim());
    assert_ne!(claim.base_coefficient(1), Base::zero());
    let mut terminal_point = Vec::new();
    for round in 0..actual.num_rounds() {
        let polynomial = actual.round_polynomial(round, claim).unwrap();
        assert_eq!(polynomial, expected.round_polynomial(round, claim).unwrap());
        assert_eq!(
            polynomial.evaluate(E::zero()) + polynomial.evaluate(E::one()),
            claim
        );
        let challenge = extension(6000 + round);
        claim = polynomial.evaluate(challenge);
        actual.bind_challenge(round, challenge).unwrap();
        expected.bind_challenge(round, challenge).unwrap();
        terminal_point.push(challenge);
    }
    actual.finish().unwrap();
    expected.finish().unwrap();
    let value = actual.folded_table_value().unwrap();
    assert_eq!(value, expected.folded_table_value().unwrap());
    assert_ne!(value.base_coefficient(1), Base::zero());
    let weight = a.evaluate(&terminal_point).unwrap()
        + typed
            .digit_weight(&terminal_point, &point, alpha, [&batches[0], &batches[1]])
            .unwrap();
    assert_eq!(claim, value * weight);
}
