#![cfg(feature = "labinius")]

mod common;
use common::{profile, setup};

use akita_algebra::{
    binary::BinaryField128, MinusTrinomial, PlusTrinomial, Prime64Offset23703 as F, TrinomialRing,
};
use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_labinius_prover::{commit_binary_clear, prove_binary_clear_bytes};
use akita_labinius_verifier::{
    codec::{decode_response_coefficient, encode_response_coefficient, response_width},
    BinaryClearCommitment, BinaryClearSetup,
};
use akita_params::sis::labinius::{LabiniusCoefficientPrime as P, LabiniusRingDegree as D};

#[test]
fn setup_admission_rejects_invalid_shapes_budget_interval_and_identities() {
    let good = setup::<F, 162, PlusTrinomial>(1, 4, 4);
    let matrix = good.matrix().to_vec();
    let check = |matrix, n_a, m, columns, lower, upper, lambda, profile, prime, degree| {
        BinaryClearSetup::<F, 162, PlusTrinomial>::new(
            matrix, n_a, m, columns, lower, upper, lambda, profile, prime, degree,
        )
    };
    for (n_a, m, columns) in [(0, 4, 4), (1, 0, 4), (1, 3, 4), (1, 4, 0), (1, 4, 3)] {
        assert!(check(
            matrix.clone(),
            n_a,
            m,
            columns,
            -1024,
            1024,
            128,
            profile(),
            P::P64Offset23703,
            D::D162
        )
        .is_err());
    }
    assert!(check(
        matrix[..3].to_vec(),
        1,
        4,
        4,
        -1024,
        1024,
        128,
        profile(),
        P::P64Offset23703,
        D::D162
    )
    .is_err());
    assert!(check(
        matrix.clone(),
        1,
        4,
        4,
        -1024,
        1024,
        10_000,
        profile(),
        P::P64Offset23703,
        D::D162
    )
    .is_err());
    let weak = BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 1).unwrap();
    assert!(check(
        matrix.clone(),
        1,
        4,
        4,
        -1024,
        1024,
        128,
        weak,
        P::P64Offset23703,
        D::D162
    )
    .is_err());
    assert!(check(
        matrix.clone(),
        1,
        4,
        4,
        i64::MIN,
        i64::MAX,
        128,
        profile(),
        P::P64Offset23703,
        D::D162
    )
    .is_err());
    for (lower, upper) in [(1, 2), (-2, -1), (1, -1)] {
        assert!(check(
            matrix.clone(),
            1,
            4,
            4,
            lower,
            upper,
            128,
            profile(),
            P::P64Offset23703,
            D::D162
        )
        .is_err());
    }
    assert!(check(
        matrix.clone(),
        1,
        4,
        4,
        -1024,
        1024,
        128,
        profile(),
        P::P128OffsetA7F7,
        D::D162
    )
    .is_err());
    for degree in [D::D324, D::D648, D::D486] {
        assert!(check(
            matrix.clone(),
            1,
            4,
            4,
            -1024,
            1024,
            128,
            profile(),
            P::P64Offset23703,
            degree
        )
        .is_err());
    }
    let wrong_scalar =
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic729, 47).unwrap();
    assert!(check(
        matrix.clone(),
        1,
        4,
        4,
        -1024,
        1024,
        128,
        wrong_scalar,
        P::P64Offset23703,
        D::D162
    )
    .is_err());
    assert!(check(
        Vec::new(),
        usize::MAX,
        4,
        4,
        -1024,
        1024,
        128,
        profile(),
        P::P64Offset23703,
        D::D162
    )
    .is_err());
    assert!(check(
        vec![matrix[0]],
        1,
        1,
        1usize << (usize::BITS - 1),
        -1024,
        1024,
        128,
        profile(),
        P::P64Offset23703,
        D::D162
    )
    .is_err());
    let wrong_sign = vec![TrinomialRing::<F, 162, MinusTrinomial>::zero().unwrap()];
    assert!(BinaryClearSetup::new(
        wrong_sign,
        1,
        1,
        1,
        -1024,
        1024,
        128,
        profile(),
        P::P64Offset23703,
        D::D162
    )
    .is_err());
    let wrong_tower_sign = vec![TrinomialRing::<F, 648, PlusTrinomial>::zero().unwrap()];
    assert!(BinaryClearSetup::new(
        wrong_tower_sign,
        1,
        1,
        1,
        -1024,
        1024,
        128,
        profile(),
        P::P64Offset23703,
        D::D648
    )
    .is_err());
}

#[test]
fn canonical_response_offsets_accept_exact_endpoints_and_reject_excess() {
    for (lower, upper) in [
        (-1024, 1024),
        (-128, 127),
        (-128, 128),
        (0, 0),
        (i64::MIN, i64::MAX),
    ] {
        for value in [lower, upper] {
            let bytes = encode_response_coefficient(value, lower, upper).unwrap();
            assert_eq!(bytes.len(), response_width(lower, upper).unwrap());
            assert_eq!(
                decode_response_coefficient(&bytes, lower, upper).unwrap(),
                value
            );
        }
        if upper < i64::MAX {
            assert!(encode_response_coefficient(upper + 1, lower, upper).is_err());
        }
    }
    assert_eq!(
        decode_response_coefficient(&2049u16.to_le_bytes(), -1024, 1024),
        Err(akita_error::AkitaError::InvalidProof)
    );
    assert_eq!(decode_response_coefficient(&[0], 0, 0).unwrap(), 0);
    assert!(decode_response_coefficient(&[1], 0, 0).is_err());
    assert!(decode_response_coefficient(&[0, 0], 0, 0).is_err());
    assert!(response_width(1, -1).is_err());
}

#[test]
fn prover_rejects_an_out_of_interval_response_without_retrying() {
    let good = setup::<F, 162, PlusTrinomial>(1, 1, 1);
    let tight = BinaryClearSetup::new(
        good.matrix().to_vec(),
        1,
        1,
        1,
        0,
        0,
        128,
        profile(),
        P::P64Offset23703,
        D::D162,
    )
    .unwrap();
    let source = [1u128];
    let commitment =
        commit_binary_clear::<BinaryField128, F, 162, PlusTrinomial>(&tight, &source).unwrap();
    assert!(
        prove_binary_clear_bytes(&tight, &source, &commitment, &[], BinaryField128::ONE).is_err()
    );
    // Caller-controlled shape errors return errors before source traversal.
    assert!(commit_binary_clear::<BinaryField128, F, 162, PlusTrinomial>(&good, &[]).is_err());
    assert!(prove_binary_clear_bytes(&good, &[], &commitment, &[], BinaryField128::ONE).is_err());
    assert!(prove_binary_clear_bytes(
        &good,
        &source,
        &commitment,
        &[BinaryField128::ONE],
        BinaryField128::ONE
    )
    .is_err());
    let malformed = BinaryClearCommitment::<F, 162, PlusTrinomial> { images: Vec::new() };
    assert!(
        prove_binary_clear_bytes(&good, &source, &malformed, &[], BinaryField128::ONE).is_err()
    );
    assert!(
        prove_binary_clear_bytes(&good, &source, &commitment, &[], BinaryField128::ZERO).is_err()
    );
}
