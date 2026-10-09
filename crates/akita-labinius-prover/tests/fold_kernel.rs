#![cfg(feature = "labinius")]

#[path = "common/mod.rs"]
mod common;

use akita_algebra::binary::{BinaryField128 as F128, BinaryField192 as F192};
use akita_challenges::{
    BinaryChallenge, BinaryChallengeProfile, BinaryChallengeSampler, BinaryScalarRing,
};
use akita_error::AkitaError;
use akita_labinius_prover::fold_kernel::fold_integer as kernel;
use akita_labinius_verifier::endpoint::fold_integer as reference;
use akita_params::sis::labinius::LabiniusRootProfile;
use common::{FixedDraw, TestHost};
use rand::{rngs::StdRng, SeedableRng};

fn root_profile() -> BinaryChallengeProfile {
    LabiniusRootProfile::D648P128BoundedW46Delta16
        .challenge_profile()
        .unwrap()
}

fn challenges(profile: &BinaryChallengeProfile, columns: usize) -> Vec<BinaryChallenge> {
    BinaryChallengeSampler::new(profile.clone())
        .sample_challenges(&mut FixedDraw, b"fold-kernel-differential", columns)
        .unwrap()
}

fn random_cases<H: TestHost>() {
    let profiles = [
        root_profile(),
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 0).unwrap(),
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 1).unwrap(),
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 162).unwrap(),
    ];
    for seed in [0x1234, 0x4321, 0xcafe] {
        let mut rng = StdRng::seed_from_u64(seed);
        for (rows, columns) in [(1, 1), (1, 8), (8, 1), (2, 4), (8, 2), (16, 8), (4, 32)] {
            let source = (0..rows * columns)
                .map(|_| H::random_source(&mut rng))
                .collect::<Vec<_>>();
            for profile in &profiles {
                let fold = challenges(profile, columns);
                assert_eq!(
                    kernel::<H>(&source, rows, columns, &fold, profile),
                    reference::<H>(&source, rows, columns, &fold, profile),
                    "seed={seed}, rows={rows}, columns={columns}, cap={}",
                    profile.weight_cap()
                );
            }
        }
    }
}

#[test]
fn random_both_switch_fields_match_reference() {
    random_cases::<F128>();
    random_cases::<F192>();
}

#[test]
fn zero_and_all_ones_source_words_match_reference() {
    let profile = root_profile();
    let fold = challenges(&profile, 256);
    for word in [0, u128::MAX] {
        let source = vec![word; 4 * 256];
        assert_eq!(
            kernel::<F128>(&source, 4, 256, &fold, &profile),
            reference::<F128>(&source, 4, 256, &fold, &profile)
        );
    }
    for word in [0, u64::MAX] {
        let source = vec![word; 4 * 256];
        assert_eq!(
            kernel::<F192>(&source, 4, 256, &fold, &profile),
            reference::<F192>(&source, 4, 256, &fold, &profile)
        );
    }
}

#[test]
fn first_profile_reduced_rows_match_reference() {
    let profile = root_profile();
    let fold = challenges(&profile, 256);
    assert_eq!(profile.weight_cap(), 46);
    let mut rng = StdRng::seed_from_u64(0x64_256);
    let source = (0..64 * 256)
        .map(|_| F128::random_source(&mut rng))
        .collect::<Vec<_>>();
    assert_eq!(
        kernel::<F128>(&source, 64, 256, &fold, &profile),
        reference::<F128>(&source, 64, 256, &fold, &profile)
    );
}

#[test]
fn maximal_coefficient_and_maximum_weight_exercise_wide_lane_bound() {
    let profile = root_profile();
    let challenge = challenges(&profile, 512)
        .into_iter()
        .find(|c| c.weight() == 46)
        .unwrap();
    let fold = [challenge.clone()];
    // For a fixed challenge, every source bit contributes independently over Z.
    // Pick the coefficient and sign with largest possible magnitude, then set
    // exactly the source bits whose contribution has that sign. This maximises
    // that coefficient over every u128 source word, rather than just samples.
    let basis = (0..128)
        .map(|bit| reference::<F128>(&[1u128 << bit], 1, 1, &fold, &profile).unwrap()[0])
        .collect::<Vec<_>>();
    let (target, sign, _) = (0..162)
        .flat_map(|target| [-1i64, 1].map(move |sign| (target, sign)))
        .map(|(target, sign)| {
            (
                target,
                sign,
                basis
                    .iter()
                    .map(|row| (sign * row[target]).max(0))
                    .sum::<i64>(),
            )
        })
        .max_by_key(|&(_, _, magnitude)| magnitude)
        .unwrap();
    let word = basis.iter().enumerate().fold(0u128, |word, (bit, row)| {
        word | if sign * row[target] > 0 {
            1u128 << bit
        } else {
            0
        }
    });
    let columns = 2048;
    let fold = vec![challenge; columns];
    for word in [word, u128::MAX] {
        let source = vec![word; columns];
        let expected = reference::<F128>(&source, 1, columns, &fold, &profile).unwrap();
        assert_eq!(
            kernel::<F128>(&source, 1, columns, &fold, &profile),
            Ok(expected)
        );
    }
    let expected = reference::<F128>(&[word], 1, 1, &fold[..1], &profile).unwrap();
    assert!(expected[0][target].unsigned_abs() * columns as u64 > i16::MAX as u64);
}

#[test]
fn maximum_weight_one_sign_challenges_exceed_i16() {
    // Terms are private and deterministic sign replay is enforced. A cap-one
    // profile admits constructible maximum-weight challenges of one sign;
    // fabricating one-sign W46 challenges is unavailable through the public API.
    let profile = BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 1).unwrap();
    let challenge = challenges(&profile, 1).remove(0);
    assert_eq!(challenge.weight(), 1);
    let columns = 65_536;
    let fold = vec![challenge; columns];
    let source = vec![1u128; columns];
    let expected = reference::<F128>(&source, 1, columns, &fold, &profile).unwrap();
    assert_eq!(
        expected[0].iter().map(|x| x.unsigned_abs()).max(),
        Some(columns as u64)
    );
    assert_eq!(
        kernel::<F128>(&source, 1, columns, &fold, &profile),
        Ok(expected)
    );
}

#[test]
fn exact_i16_accumulation_boundary_matches_reference() {
    let profile =
        BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 1).unwrap();
    let sampled = challenges(&profile, 4096);
    let empty = sampled.iter().find(|c| c.weight() == 0).unwrap();
    let challenge = sampled.iter().find(|c| c.weight() == 1).unwrap();
    let columns = 32_768;
    let source = vec![1u128; columns];
    let mut fold = vec![challenge.clone(); columns];
    fold[columns - 1] = empty.clone();
    for magnitude in [32_767, 32_768] {
        let expected = reference::<F128>(&source, 1, columns, &fold, &profile).unwrap();
        assert_eq!(
            expected[0].iter().map(|x| x.unsigned_abs()).max(),
            Some(magnitude)
        );
        assert_eq!(
            kernel::<F128>(&source, 1, columns, &fold, &profile),
            Ok(expected)
        );
        fold[columns - 1] = challenge.clone();
    }
}

#[test]
fn every_geometry_rejection_matches_reference_and_precedes_challenges() {
    let profile = root_profile();
    let valid = challenges(&profile, 3);
    let invalid_profile =
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 162).unwrap();
    let invalid = challenges(&invalid_profile, 3);
    for fold in [&valid, &invalid] {
        for (rows, columns, source_len, challenge_len) in [
            (0, 2, 0, 2),
            (2, 0, 0, 0),
            (0, 0, 0, 0),
            (3, 2, 6, 2),
            (2, 3, 6, 3),
            (2, 2, 3, 2),
            (2, 2, 5, 2),
            (2, 2, 4, 1),
            (1, 1, 1, 2),
            (usize::MAX, 2, 0, 2),
            (1usize << (usize::BITS - 1), 2, 0, 2),
        ] {
            let source = vec![0u128; source_len];
            let fold = &fold[..challenge_len];
            let expected = reference::<F128>(&source, rows, columns, fold, &profile);
            assert!(matches!(expected, Err(AkitaError::InvalidInput(_))));
            assert_eq!(
                kernel::<F128>(&source, rows, columns, fold, &profile),
                expected
            );
        }
    }
}

#[test]
fn wrong_challenge_profiles_and_validation_order_match_reference() {
    let profile = root_profile();
    let valid = challenges(&profile, 2);
    let wrong_weight = challenges(
        &BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 162).unwrap(),
        1,
    )
    .remove(0);
    let wrong_signs = challenges(
        &BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 46).unwrap(),
        1,
    )
    .remove(0);
    let sign_error = reference::<F128>(&[0], 1, 1, std::slice::from_ref(&wrong_signs), &profile)
        .expect_err("different profile identity changes deterministic signs");
    assert!(
        matches!(&sign_error, AkitaError::InvalidInput(message) if message == "binary challenge signs do not match the profile rule")
    );
    let weight_error = reference::<F128>(&[0], 1, 1, std::slice::from_ref(&wrong_weight), &profile)
        .expect_err("maximum degree weight exceeds the admitted cap");
    assert_ne!(weight_error, sign_error);
    for (fold, first_error) in [
        (
            vec![wrong_weight.clone(), wrong_signs.clone()],
            weight_error,
        ),
        (vec![wrong_signs, wrong_weight.clone()], sign_error),
    ] {
        assert_eq!(
            reference::<F128>(&[0; 2], 1, 2, &fold, &profile),
            Err(first_error.clone())
        );
        assert_eq!(
            kernel::<F128>(&[0; 2], 1, 2, &fold, &profile),
            Err(first_error)
        );
    }
    for fold in [
        vec![wrong_weight.clone(), valid[1].clone()],
        vec![valid[0].clone(), wrong_weight],
    ] {
        let expected = reference::<F128>(&[0; 2], 1, 2, &fold, &profile);
        assert!(expected.is_err());
        assert_eq!(kernel::<F128>(&[0; 2], 1, 2, &fold, &profile), expected);
    }
    for wrong_profile in [
        BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic243, 0).unwrap(),
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 46).unwrap(),
        BinaryChallengeProfile::bounded_weight(BinaryScalarRing::Cyclotomic729, 46).unwrap(),
    ] {
        let expected = reference::<F128>(&[0; 2], 1, 2, &valid, &wrong_profile);
        assert!(expected.is_err());
        assert_eq!(
            kernel::<F128>(&[0; 2], 1, 2, &valid, &wrong_profile),
            expected
        );
    }
    // BinaryChallenge exposes only immutable terms and sampler construction, so
    // duplicate positions, bad signs and out-of-range positions are not public
    // input constructors. Wrong-profile challenges exercise its public errors.
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_schedules_match_serial_reference() {
    let profile = root_profile();
    let fold = challenges(&profile, 256);
    let mut rng = StdRng::seed_from_u64(0xface);
    let source = (0..64 * 256)
        .map(|_| F128::random_source(&mut rng))
        .collect::<Vec<_>>();
    let expected = reference::<F128>(&source, 64, 256, &fold, &profile);
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(
            pool.install(|| kernel::<F128>(&source, 64, 256, &fold, &profile)),
            expected
        );
    }
}
