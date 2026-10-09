#![cfg(feature = "labinius")]

mod a_relation_support;
mod common;

use akita_algebra::{binary::BinaryField128 as H, MinusTrinomial, TrinomialRing};
use akita_challenges::BinaryChallengeSampler;
use akita_error::AkitaError;
use akita_labinius_prover::{
    a_carry_kernel::a_relation_carry, commit_binary_clear, PreparedCommitMatrix,
};
use akita_labinius_verifier::{endpoint::fold_integer, AdmittedRootSetup};
use akita_params::sis::labinius::{LabiniusDigitBase, LabiniusRootProfile};
use akita_types::proof::AkitaSetupSeed;
use jolt_field::{One, Prime128OffsetA7F7 as F, Ring};

const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128Q28BoundedW46Delta16;

#[test]
fn carry_kernel_matches_schoolbook_remainders_for_every_digit_base() {
    for fold in [0, 1] {
        let admitted = AdmittedRootSetup::<F, 648, MinusTrinomial>::derive(
            PROFILE,
            3,
            fold,
            128,
            AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
        )
        .unwrap();
        let setup = admitted.setup();
        let (source, _, _) = common::data::<H>(setup.source_len(), setup.num_vars());
        let commitment = commit_binary_clear::<H, F, 648, MinusTrinomial>(setup, &source).unwrap();
        let challenges = BinaryChallengeSampler::new(setup.profile().clone())
            .sample_challenges(&mut common::FixedDraw, b"carry", setup.columns())
            .unwrap();
        let response = fold_integer::<H>(
            &source,
            setup.scalar_rows(),
            setup.columns(),
            &challenges,
            setup.profile(),
        )
        .unwrap();
        let prepared = PreparedCommitMatrix::prepare(setup).unwrap();
        for base in [
            LabiniusDigitBase::Bits1,
            LabiniusDigitBase::Bits2,
            LabiniusDigitBase::Bits4,
        ] {
            let range = admitted
                .shape()
                .derive_encoding(base)
                .unwrap()
                .a_carry()
                .unwrap();
            let expected = a_relation_support::a_relation_carry(
                setup,
                &commitment,
                &challenges,
                &response,
                range,
            )
            .unwrap();
            let run = || {
                a_relation_carry(&prepared, setup, &commitment, &challenges, &response, range)
                    .unwrap()
            };
            assert_eq!(run(), expected);
            assert_eq!(expected.len(), setup.n_a() * 648);
            #[cfg(feature = "parallel")]
            for threads in [1, 3] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                assert_eq!(pool.install(run), expected);
            }
        }
    }
}

#[test]
fn carry_kernel_rejects_input_boundaries_and_nondivisible_remainders() {
    let admitted = AdmittedRootSetup::<F, 648, MinusTrinomial>::derive(
        PROFILE,
        2,
        0,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .unwrap();
    let setup = admitted.setup();
    let range = admitted
        .shape()
        .derive_encoding(LabiniusDigitBase::Bits1)
        .unwrap()
        .a_carry()
        .unwrap();
    let source = vec![0u128; setup.source_len()];
    let commitment = commit_binary_clear::<H, F, 648, MinusTrinomial>(setup, &source).unwrap();
    let challenges = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(&mut common::FixedDraw, b"carry", setup.columns())
        .unwrap();
    let response = vec![[0i64; 162]; setup.scalar_rows()];
    let prepared = PreparedCommitMatrix::prepare(setup).unwrap();
    let run = |image: &_, fold: &_, response: &_| {
        a_relation_carry(&prepared, setup, image, fold, response, range)
    };
    let mut changed = commitment.clone();
    let mut coefficients = *changed.images[0].coefficients();
    coefficients[0] += F::one();
    changed.images[0] = TrinomialRing::from_coefficients(coefficients).unwrap();
    assert_eq!(
        run(&changed, &challenges, &response),
        Err(AkitaError::InvalidProof)
    );
    let mut out_of_range = commitment.clone();
    let mut coefficients = *out_of_range.images[0].coefficients();
    coefficients[0] = F::from_u128(
        u128::from(setup.commitment_modulus().small_modulus().unwrap()) * (range.offset() + 1),
    );
    out_of_range.images[0] = TrinomialRing::from_coefficients(coefficients).unwrap();
    assert_eq!(
        run(&out_of_range, &challenges, &response),
        Err(AkitaError::InvalidProof)
    );
    assert_eq!(
        a_relation_support::a_relation_carry(setup, &out_of_range, &challenges, &response, range),
        Err(AkitaError::InvalidProof)
    );
    let foreign_profile = akita_challenges::BinaryChallengeProfile::fixed_weight(
        akita_challenges::BinaryScalarRing::Cyclotomic243,
        47,
    )
    .unwrap();
    let foreign_fold = BinaryChallengeSampler::new(foreign_profile)
        .sample_challenges(&mut common::FixedDraw, b"foreign-carry", setup.columns())
        .unwrap();
    assert_eq!(
        run(&commitment, &foreign_fold, &response),
        Err(AkitaError::InvalidProof)
    );
    for extra in [false, true] {
        let mut image = commitment.clone();
        let mut fold = challenges.clone();
        let mut rows = response.clone();
        if extra {
            image.images.push(image.images[0]);
            fold.push(fold[0].clone());
            rows.push(rows[0]);
        } else {
            image.images.pop();
            fold.pop();
            rows.pop();
        }
        assert_eq!(
            run(&image, &challenges, &response),
            Err(AkitaError::InvalidProof)
        );
        assert_eq!(
            run(&commitment, &fold, &response),
            Err(AkitaError::InvalidProof)
        );
        assert_eq!(
            run(&commitment, &challenges, &rows),
            Err(AkitaError::InvalidProof)
        );
    }
    for value in [setup.lower() - 1, setup.upper() + 1] {
        let mut rows = response.clone();
        rows[0][0] = value;
        assert_eq!(
            run(&commitment, &challenges, &rows),
            Err(AkitaError::InvalidProof)
        );
    }
    let other = AdmittedRootSetup::<F, 648, MinusTrinomial>::derive(
        PROFILE,
        2,
        0,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x32; 32]),
    )
    .unwrap();
    assert!(matches!(
        a_relation_carry(
            &PreparedCommitMatrix::prepare(other.setup()).unwrap(),
            setup,
            &commitment,
            &challenges,
            &response,
            range
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn carry_kernel_rejects_shared_prime_setups_with_a_typed_error() {
    let shared = AdmittedRootSetup::<F, 648, MinusTrinomial>::derive(
        LabiniusRootProfile::D648P128BoundedW46Delta16,
        2,
        0,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .unwrap();
    let setup = shared.setup();
    let lifted = AdmittedRootSetup::<F, 648, MinusTrinomial>::derive(
        PROFILE,
        2,
        0,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .unwrap();
    let range = lifted
        .shape()
        .derive_encoding(LabiniusDigitBase::Bits1)
        .unwrap()
        .a_carry()
        .unwrap();
    let source = vec![0u128; setup.source_len()];
    let commitment = commit_binary_clear::<H, F, 648, MinusTrinomial>(setup, &source).unwrap();
    let fold = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(&mut common::FixedDraw, b"carry", setup.columns())
        .unwrap();
    let response = vec![[0; 162]; setup.scalar_rows()];
    assert!(matches!(
        a_relation_carry(
            &PreparedCommitMatrix::prepare(setup).unwrap(),
            setup,
            &commitment,
            &fold,
            &response,
            range
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}
