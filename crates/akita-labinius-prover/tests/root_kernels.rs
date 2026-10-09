#![cfg(feature = "labinius")]

mod root_reduction_support;

use akita_algebra::binary::BinaryField192;
use akita_challenges::BinaryChallengeSampler;
use akita_error::AkitaError;
use akita_labinius_prover::{
    lowered::{a_relation_quotients, encode_witness, parity_quotient_and_carry},
    prove_root_reduction_bytes,
    root_sumcheck::{
        prove_combined_rounds, prove_product_rounds, CombinedRootSumcheck, ProductSumcheck,
    },
    PreparedRootMatrices, TransparentRootProverOracle,
};
use akita_labinius_verifier::{
    channel::{
        finish_prover, new_root_prover, ClearChannel, RootChallengeChannel, RootFieldSite,
        RootSumcheckProverChannel,
    },
    codec::{exchange_binary, exchange_field},
    endpoint::{fold_integer, left_expansion},
    frontend::prove_frontend,
    lowered::{image_weights_dense, witness_weights_dense, LoweredChallenges, LoweredPublic},
    root::{bind_root_statement, exchange_root_auxiliary, RootEvaluationClaims, RootProverOracle},
};
use jolt_field::Zero;
use root_reduction_support::{admitted, common::TestHost, Case, BASES, F, H};

/// The reduction composed from the schoolbook fold, the long-division
/// quotients and the dense combined sumcheck, with no transform cache.
fn reference<T: TestHost>(case: &Case<T>) -> (Vec<u8>, RootEvaluationClaims<F>) {
    let setup = case.admitted.setup();
    let layout = &case.layout;
    let mut oracle = TransparentRootProverOracle::new(&case.image);
    let mut state = new_root_prover().unwrap();
    let mut ch = RootSumcheckProverChannel::new(&mut state);
    bind_root_statement(
        &case.admitted,
        case.base,
        &case.point,
        case.value,
        &mut ch,
        |layout, ch| oracle.bind_image(layout, ch),
    )
    .unwrap();
    let binary = prove_frontend::<T, _>(&case.source, &case.point, case.value, &mut ch).unwrap();
    let mut u = left_expansion::<T>(
        &case.source,
        &binary.point,
        setup.scalar_rows(),
        setup.columns(),
    )
    .unwrap();
    for element in &mut u {
        exchange_binary(&mut ch, element).unwrap();
    }
    let fold = ch
        .fold_challenges(
            &mut BinaryChallengeSampler::new(setup.profile().clone()),
            b"akita/labinius/root-fold/v1",
            setup.columns(),
        )
        .unwrap();
    let response = fold_integer::<T>(
        &case.source,
        setup.scalar_rows(),
        setup.columns(),
        &fold,
        setup.profile(),
    )
    .unwrap();
    let digits = encode_witness(layout, &response).unwrap();
    oracle.commit_response(layout, &digits, &mut ch).unwrap();
    let mut qa = a_relation_quotients(setup, &case.commitment, &fold, &response).unwrap();
    let (mut q, mut k) = parity_quotient_and_carry(setup, &binary, &u, &fold, &response).unwrap();
    exchange_root_auxiliary(layout, &mut ch, &mut qa, &mut q, &mut k).unwrap();
    let challenges = LoweredChallenges {
        alpha: ch.field_challenge(RootFieldSite::Alpha).unwrap(),
        xi: ch.field_challenge(RootFieldSite::Xi).unwrap(),
        gamma: ch.field_challenge(RootFieldSite::Gamma).unwrap(),
    };
    let public =
        LoweredPublic::new(layout, setup, &binary, &u, &fold, &qa, &q, &k, challenges).unwrap();
    let ky = image_weights_dense(layout, &public).unwrap();
    let mut y_y = case
        .image
        .iter()
        .zip(&ky)
        .fold(F::zero(), |sum, (&y, &weight)| sum + y * weight);
    exchange_field(&mut ch, &mut y_y).unwrap();
    let s = public.c_pub() - y_y;
    let tau: Vec<F> = (0..layout.witness_log_len())
        .map(|i| ch.field_challenge(RootFieldSite::Tau(i as u32)).unwrap())
        .collect();
    let beta = ch.field_challenge(RootFieldSite::Beta).unwrap();
    let kw = witness_weights_dense(layout, &public).unwrap();
    let mut combined = CombinedRootSumcheck::new(case.base, &digits, kw, &tau, beta, s).unwrap();
    let (response_point, _) = prove_combined_rounds(&mut combined, &mut ch, 0).unwrap();
    let (mut response_value, _) = combined.final_evaluations().unwrap();
    exchange_field(&mut ch, &mut response_value).unwrap();
    let mut product = ProductSumcheck::new(case.image.clone(), ky, y_y).unwrap();
    let (image_point, _) = prove_product_rounds(&mut product, &mut ch, 1).unwrap();
    let (mut image_value, _) = product.final_evaluations().unwrap();
    exchange_field(&mut ch, &mut image_value).unwrap();
    let claims = RootEvaluationClaims {
        response_point,
        response_value,
        image_point,
        image_value,
    };
    oracle.discharge(&claims, &mut ch).unwrap();
    (finish_prover(state), claims)
}

fn kernels_match_reference<T: TestHost>() {
    for base in BASES {
        for fold in [0, 1] {
            let case = Case::<T>::new(base, fold);
            let (expected, expected_claims) = reference(&case);
            let (proof, claims, _) = case.prove();
            assert_eq!(proof, expected, "{base:?} fold {fold}");
            assert_eq!(claims.response_point, expected_claims.response_point);
            assert_eq!(claims.response_value, expected_claims.response_value);
            assert_eq!(claims.image_point, expected_claims.image_point);
            assert_eq!(claims.image_value, expected_claims.image_value);
            case.verify(&proof).unwrap();
        }
    }
}

#[test]
fn kernel_reduction_emits_the_reference_proof_bytes_for_both_hosts() {
    kernels_match_reference::<H>();
    kernels_match_reference::<BinaryField192>();
}

#[test]
fn transform_cache_of_another_matrix_is_rejected_before_any_proof_byte() {
    for (fold, seed) in [(0, 0x32), (1, 0x32)] {
        let case = Case::<H>::new(BASES[1], 0);
        let foreign = admitted(fold, seed);
        let prepared = PreparedRootMatrices::prepare(foreign.setup()).unwrap();
        let mut oracle = TransparentRootProverOracle::new(&case.image);
        let result = prove_root_reduction_bytes(
            &case.admitted,
            &prepared,
            case.base,
            &case.source,
            &case.commitment,
            &case.point,
            case.value,
            &mut oracle,
        );
        assert!(matches!(result, Err(AkitaError::InvalidSetup(_))));
        assert!(oracle.response().is_empty());
    }
}
