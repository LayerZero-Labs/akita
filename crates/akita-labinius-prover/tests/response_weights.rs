#![cfg(feature = "labinius")]

mod lowered_support;

use akita_error::AkitaError;
use akita_labinius_prover::{
    response_weights::{coefficient_weights, image_weights},
    PreparedCommitMatrix,
};
use akita_labinius_verifier::lowered::{
    image_weight_mle, image_weights_dense, witness_weight_mle, witness_weights_dense,
    LoweredRootLayout,
};
use jolt_field::Zero;
use lowered_support::{Case, BASES, F};

#[test]
fn compact_coefficient_factor_tensor_digits_equals_dense_weights_for_each_base() {
    for base in BASES {
        let case = Case::new(base);
        let public = case.public();
        let prepared = PreparedCommitMatrix::prepare(&case.setup).unwrap();
        let compact = coefficient_weights(&prepared, &case.layout, &case.setup, &public).unwrap();
        assert_eq!(
            image_weights(&prepared, &case.layout, &public).unwrap(),
            image_weights_dense(&case.layout, &public).unwrap()
        );
        let dense = witness_weights_dense(&case.layout, &public, &case.setup).unwrap();
        let tensor = compact
            .iter()
            .flat_map(|&coefficient| {
                public
                    .digit_powers()
                    .iter()
                    .map(move |&digit| digit * coefficient)
            })
            .collect::<Vec<_>>();
        assert_eq!(tensor, dense, "{base:?}");
        assert_eq!(
            compact.len(),
            case.layout.witness_len() / public.digit_powers().len()
        );
        assert!(case.layout.padded_coefficients() > case.layout.degree());
        for row in compact.chunks_exact(case.layout.padded_coefficients()) {
            assert!(row
                .iter()
                .skip(case.layout.degree())
                .all(|&v| v == F::zero()));
        }
    }
}

#[test]
fn compact_weights_reject_a_public_digit_factor_from_a_different_base() {
    let case = Case::new(BASES[0]);
    let public = case.public();
    let foreign = LoweredRootLayout::new(&case.setup, &case.shape, BASES[1]).unwrap();
    assert!(matches!(
        coefficient_weights(
            &PreparedCommitMatrix::prepare(&case.setup).unwrap(),
            &foreign,
            &case.setup,
            &public
        ),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn rank_three_compact_and_image_tables_match_dense_definitions_with_padding() {
    use akita_algebra::{binary::BinaryField162 as B, MinusTrinomial};
    use akita_labinius_verifier::{
        lowered::{LoweredChallenges, LoweredPublic},
        AdmittedRootSetup, BinaryEvaluationClaim,
    };
    use akita_params::sis::labinius::LabiniusRootProfile;
    use akita_types::proof::AkitaSetupSeed;
    use jolt_field::Ring;
    use rand::SeedableRng;
    let admitted = AdmittedRootSetup::<F, 648, MinusTrinomial>::derive(
        LabiniusRootProfile::D648P128Q28BoundedW46Delta16,
        3,
        1,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .unwrap();
    let setup = admitted.setup();
    let prepared = PreparedCommitMatrix::prepare(setup).unwrap();
    let fold = akita_challenges::BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(
            &mut lowered_support::common::FixedDraw,
            b"weights",
            setup.columns(),
        )
        .unwrap();
    let claim = BinaryEvaluationClaim {
        point: vec![B::ZERO; setup.num_vars()],
        value: B::ZERO,
    };
    for base in BASES {
        let layout = LoweredRootLayout::new(setup, admitted.shape(), base).unwrap();
        let carry = vec![0; layout.encoding().a_carry_len()];
        let public = LoweredPublic::new(
            &layout,
            setup,
            &claim,
            &vec![B::ZERO; setup.columns()],
            &fold,
            &carry,
            &[0; 161],
            &[0; 162],
            LoweredChallenges {
                alpha: F::from_u64(7),
                xi: F::from_u64(11),
                gamma: F::from_u64(13),
            },
        )
        .unwrap();
        let dense = witness_weights_dense(&layout, &public, setup).unwrap();
        let image_dense = image_weights_dense(&layout, &public).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x51a);
        let response_point = lowered_support::random_point(&mut rng, layout.witness_log_len());
        let image_point = lowered_support::random_point(&mut rng, layout.image_log_len());
        let run = || {
            let compact = coefficient_weights(&prepared, &layout, setup, &public).unwrap();
            let tensor = compact
                .iter()
                .flat_map(|&coefficient| {
                    public
                        .digit_powers()
                        .iter()
                        .map(move |&digit| digit * coefficient)
                })
                .collect::<Vec<_>>();
            assert_eq!(tensor, dense);
            let image = image_weights(&prepared, &layout, &public).unwrap();
            assert_eq!(image, image_dense);
            assert_eq!(
                akita_algebra::poly::multilinear_eval(&tensor, &response_point).unwrap(),
                witness_weight_mle(&layout, &public, setup, &response_point).unwrap()
            );
            assert_eq!(
                akita_algebra::poly::multilinear_eval(&image, &image_point).unwrap(),
                image_weight_mle(&layout, &public, &image_point).unwrap()
            );
            let live = setup.columns() * setup.n_a() * layout.padded_coefficients();
            assert!(image[live..].iter().all(|&value| value == F::zero()));
        };
        run();
        #[cfg(feature = "parallel")]
        for threads in [1, 3] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(run);
        }
    }
}
