#![cfg(feature = "labinius")]

mod common;

use akita_algebra::{
    binary::{BinaryField128, BinaryField192},
    MinusTrinomial,
};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear, commit_binary_clear_prepared, prove_binary_clear_bytes,
    PreparedCommitMatrix,
};
use akita_labinius_verifier::{
    commitment::{canonical_coefficient, centered_coefficient},
    verify_binary_clear_bytes, AdmittedRootSetup,
};
use akita_params::sis::labinius::LabiniusRootProfile;
use akita_types::proof::AkitaSetupSeed;
use common::{data, TestHost};
use jolt_field::{One, Prime128OffsetA7F7 as F};

type Setup = AdmittedRootSetup<F, 648, MinusTrinomial>;

fn admitted(fold: u32) -> Setup {
    Setup::derive(
        LabiniusRootProfile::D648P128Q28BoundedW46Delta16,
        4,
        fold,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x28; 32]),
    )
    .unwrap()
}

fn clear<T: TestHost>()
where
    T::Source: Sync,
{
    for fold in [0, 1] {
        let admitted = admitted(fold);
        let setup = admitted.setup();
        assert_eq!(setup.n_a(), 3);
        let (source, point, claim) = data::<T>(setup.source_len(), setup.num_vars());
        let commitment = commit_binary_clear::<T, F, 648, MinusTrinomial>(setup, &source).unwrap();
        let q0 = setup.commitment_modulus().small_modulus().unwrap();
        assert!(commitment
            .images
            .iter()
            .flat_map(|image| image.coefficients())
            .all(|&value| canonical_coefficient(value).unwrap() < u128::from(q0)));
        let prepared = PreparedCommitMatrix::prepare(setup).unwrap();
        assert_eq!(
            commit_binary_clear_prepared::<T, F, 648, MinusTrinomial>(&prepared, setup, &source)
                .unwrap(),
            commitment
        );
        let proof = prove_binary_clear_bytes(setup, &source, &commitment, &point, claim).unwrap();
        verify_binary_clear_bytes(setup, &commitment, &point, claim, &proof).unwrap();
        let mut changed = commitment.clone();
        let mut coefficients = *changed.images[0].coefficients();
        coefficients[0] += F::one();
        changed.images[0] = akita_algebra::TrinomialRing::from_coefficients(coefficients).unwrap();
        assert!(matches!(
            verify_binary_clear_bytes(setup, &changed, &point, claim, &proof),
            Err(AkitaError::InvalidProof)
        ));
    }
}

#[test]
fn clear_opening_and_commitment_kernel_agree_for_both_hosts() {
    clear::<BinaryField128>();
    clear::<BinaryField192>();
}

#[test]
fn all_ones_commitments_match_the_reference_for_both_hosts() {
    let admitted = admitted(1);
    let setup = admitted.setup();
    let prepared = PreparedCommitMatrix::prepare(setup).unwrap();
    let u128_source = vec![u128::MAX; setup.source_len()];
    let u64_source = vec![u64::MAX; setup.source_len()];
    assert_eq!(
        commit_binary_clear::<BinaryField128, F, 648, MinusTrinomial>(setup, &u128_source).unwrap(),
        commit_binary_clear_prepared::<BinaryField128, F, 648, MinusTrinomial>(
            &prepared,
            setup,
            &u128_source
        )
        .unwrap()
    );
    assert_eq!(
        commit_binary_clear::<BinaryField192, F, 648, MinusTrinomial>(setup, &u64_source).unwrap(),
        commit_binary_clear_prepared::<BinaryField192, F, 648, MinusTrinomial>(
            &prepared,
            setup,
            &u64_source
        )
        .unwrap()
    );
}

#[test]
fn centred_lift_preserves_both_sides_of_zero() {
    assert_eq!(centered_coefficient(F::one()).unwrap(), 1);
    assert_eq!(centered_coefficient(-F::one()).unwrap(), -1);
}

#[test]
fn clear_endpoint_checks_integer_divisibility_and_carry_range_directly() {
    use akita_algebra::{binary::BinaryField162 as B, TrinomialRing};
    use akita_challenges::BinaryChallengeSampler;
    use akita_labinius_verifier::{
        endpoint::verify_endpoints, BinaryClearCommitment, BinaryEvaluationClaim,
    };
    use jolt_field::Ring;
    let admitted = admitted(0);
    let setup = admitted.setup();
    let q0 = setup.commitment_modulus().small_modulus().unwrap();
    let (lo, hi) = setup.a_carry().unwrap().interval();
    let response = vec![[0i64; 162]; setup.scalar_rows()];
    let claim = BinaryEvaluationClaim {
        point: vec![B::ZERO; setup.num_vars()],
        value: B::ZERO,
    };
    let fold = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(&mut common::FixedDraw, b"integer-check", 1)
        .unwrap();
    let mut commitment = BinaryClearCommitment {
        images: vec![TrinomialRing::<F, 648, MinusTrinomial>::zero().unwrap(); 3],
    };
    verify_endpoints(setup, &commitment, &claim, &[B::ZERO], &fold, &response).unwrap();
    for coefficient in [
        F::one(),
        F::from_u128(u128::from(q0) * (hi + 1) as u128),
        -F::from_u128(u128::from(q0) * (lo - 1).unsigned_abs()),
    ] {
        let mut coefficients = [F::from_u64(0); 648];
        coefficients[0] = coefficient;
        commitment.images[0] = TrinomialRing::from_coefficients(coefficients).unwrap();
        assert_eq!(
            verify_endpoints(setup, &commitment, &claim, &[B::ZERO], &fold, &response),
            Err(AkitaError::InvalidProof)
        );
    }
}
