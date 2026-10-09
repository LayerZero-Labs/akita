#![cfg(feature = "labinius")]

#[path = "quotient_support.rs"]
mod quotient_support;
use quotient_support::a_relation_quotients as reference;

#[path = "root_reduction_support.rs"]
mod support;

use akita_algebra::{
    binary::BinaryField192, embed_scalar, MinusTrinomial, PlusTrinomial, SmoothFftField,
    TrinomialModulus, TrinomialNttDomain, TrinomialRing,
};
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear,
    quotient_kernel::{a_relation_quotients as kernel, ConjugateModulus, PreparedQuotientMatrix},
    PreparedCommitMatrix,
};
use akita_labinius_verifier::{
    endpoint::{fold_integer, pack_response},
    source::challenge_scalar,
    BinaryClearCommitment, BinaryClearSetup,
};
use jolt_field::{One, WithPacking};
use support::{common::FixedDraw, common::TestHost, Case, BASES, F, H};

#[test]
fn p128_splits_both_degree_648_moduli() {
    assert_eq!((648 / 2) * MinusTrinomial::ROOT_ORDER_STRIDE, 1944);
    assert_eq!((648 / 2) * PlusTrinomial::ROOT_ORDER_STRIDE, 972);
    assert!(TrinomialNttDomain::<F, 648, MinusTrinomial>::new().is_ok());
    assert!(TrinomialNttDomain::<F, 648, PlusTrinomial>::new().is_ok());
}

struct OtherDraw;
impl akita_challenges::FoldDraw for OtherDraw {
    fn absorb_and_squeeze(&mut self, _payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        Ok([0x54; 32])
    }
}

fn folded<T: TestHost>(case: &Case<T>) -> (Vec<BinaryChallenge>, Vec<[i64; 162]>) {
    let setup = case.admitted.setup();
    let challenges = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(
            &mut FixedDraw,
            b"akita/labinius/root-fold/v1",
            setup.columns(),
        )
        .unwrap();
    let response = fold_integer::<T>(
        &case.source,
        setup.scalar_rows(),
        setup.columns(),
        &challenges,
        setup.profile(),
    )
    .unwrap();
    (challenges, response)
}

fn honest_root_geometries<T: TestHost>() {
    for fold in [0, 1] {
        for base in BASES {
            let case = Case::<T>::new(base, fold);
            let setup = case.admitted.setup();
            let (challenges, response) = folded(&case);
            let commit = PreparedCommitMatrix::prepare(setup).unwrap();
            let quotient = PreparedQuotientMatrix::prepare(setup).unwrap();
            let expected = reference(setup, &case.commitment, &challenges, &response).unwrap();
            let actual = kernel(
                &commit,
                &quotient,
                setup,
                &case.commitment,
                &challenges,
                &response,
                None,
            )
            .unwrap()
            .quotients;
            assert_eq!(actual, expected, "fold={fold}, base={base:?}");
            assert_eq!(actual.len(), setup.n_a());
            assert!(actual.iter().all(|row| row.len() == 647));
            #[cfg(feature = "parallel")]
            for threads in [1, 2, 3] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let parallel = pool.install(|| {
                    kernel(
                        &commit,
                        &quotient,
                        setup,
                        &case.commitment,
                        &challenges,
                        &response,
                        None,
                    )
                    .unwrap()
                    .quotients
                });
                assert_eq!(parallel, expected, "threads={threads}");
            }
        }
    }
}

#[test]
fn differential_every_root_geometry_and_both_hosts() {
    honest_root_geometries::<H>();
    honest_root_geometries::<BinaryField192>();
}

fn reject(
    setup: &BinaryClearSetup<F, 648, MinusTrinomial>,
    commit: &PreparedCommitMatrix<F, 648, MinusTrinomial>,
    quotient: &PreparedQuotientMatrix<F, 648, MinusTrinomial>,
    commitment: &BinaryClearCommitment<F, 648, MinusTrinomial>,
    challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
) {
    assert_eq!(
        reference(setup, commitment, challenges, response),
        Err(AkitaError::InvalidProof)
    );
    assert_eq!(
        kernel(commit, quotient, setup, commitment, challenges, response, None),
        Err(AkitaError::InvalidProof)
    );
}

#[test]
fn rejection_matches_reference_for_each_input_boundary_and_nonzero_remainders() {
    let case = Case::<H>::new(BASES[0], 1);
    let setup = case.admitted.setup();
    let (challenges, response) = folded(&case);
    let commit = PreparedCommitMatrix::prepare(setup).unwrap();
    let quotient = PreparedQuotientMatrix::prepare(setup).unwrap();
    let mut changed = response.clone();
    changed[0][0] += if changed[0][0] < setup.upper() { 1 } else { -1 };
    assert!(changed[0][0] >= setup.lower() && changed[0][0] <= setup.upper());
    reject(
        setup,
        &commit,
        &quotient,
        &case.commitment,
        &challenges,
        &changed,
    );
    for value in [setup.lower() - 1, setup.upper() + 1] {
        changed[0][0] = value;
        reject(
            setup,
            &commit,
            &quotient,
            &case.commitment,
            &challenges,
            &changed,
        );
    }
    let mut wrong_image = case.commitment.clone();
    let mut coefficients = *wrong_image.images[0].coefficients();
    coefficients[0] += F::one();
    wrong_image.images[0] = TrinomialRing::from_coefficients(coefficients).unwrap();
    reject(
        setup,
        &commit,
        &quotient,
        &wrong_image,
        &challenges,
        &response,
    );

    let alternate = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(&mut OtherDraw, b"different-fold", setup.columns())
        .unwrap();
    assert_ne!(alternate, challenges);
    reject(
        setup,
        &commit,
        &quotient,
        &case.commitment,
        &alternate,
        &response,
    );
    let invalid = BinaryChallengeSampler::new(
        akita_challenges::BinaryChallengeProfile::fixed_weight(
            akita_challenges::BinaryScalarRing::Cyclotomic243,
            47,
        )
        .unwrap(),
    )
    .sample_challenges(&mut FixedDraw, b"wrong-profile", setup.columns())
    .unwrap();
    reject(
        setup,
        &commit,
        &quotient,
        &case.commitment,
        &invalid,
        &response,
    );

    for extra in [false, true] {
        let mut wrong_count = case.commitment.clone();
        if extra {
            wrong_count.images.push(wrong_count.images[0]);
        } else {
            wrong_count.images.pop();
        }
        reject(
            setup,
            &commit,
            &quotient,
            &wrong_count,
            &challenges,
            &response,
        );
        let mut wrong_count = challenges.clone();
        if extra {
            wrong_count.push(wrong_count[0].clone());
        } else {
            wrong_count.pop();
        }
        reject(
            setup,
            &commit,
            &quotient,
            &case.commitment,
            &wrong_count,
            &response,
        );
        let mut wrong_count = response.clone();
        if extra {
            wrong_count.push(wrong_count[0]);
        } else {
            wrong_count.pop();
        }
        reject(
            setup,
            &commit,
            &quotient,
            &case.commitment,
            &challenges,
            &wrong_count,
        );
    }
}

#[test]
fn both_prepared_matrices_reject_different_seed_or_shape() {
    let case = Case::<H>::new(BASES[0], 1);
    let setup = case.admitted.setup();
    let (challenges, response) = folded(&case);
    let commit = PreparedCommitMatrix::prepare(setup).unwrap();
    let quotient = PreparedQuotientMatrix::prepare(setup).unwrap();
    for other in [support::admitted(1, 0x32), support::admitted(0, 0x31)] {
        let other_commit = PreparedCommitMatrix::prepare(other.setup()).unwrap();
        let other_quotient = PreparedQuotientMatrix::prepare(other.setup()).unwrap();
        for (prepared_commit, prepared_quotient) in [
            (&other_commit, &quotient),
            (&commit, &other_quotient),
            (&other_commit, &other_quotient),
        ] {
            assert!(matches!(
                kernel(
                    prepared_commit,
                    prepared_quotient,
                    setup,
                    &case.commitment,
                    &challenges,
                    &response,
                    None
                ),
                Err(AkitaError::InvalidSetup(_))
            ));
        }
    }
    // Reinterpret the identical matrix coefficient stream with a different row partition.
    let repartitioned = BinaryClearSetup::new(
        setup.matrix().to_vec(),
        setup.n_a() * 2,
        setup.m() / 2,
        setup.columns(),
        setup.lower(),
        setup.upper(),
        128,
        setup.profile().clone(),
        akita_params::sis::labinius::LabiniusCoefficientPrime::P128OffsetA7F7,
        akita_params::sis::labinius::LabiniusRingDegree::D648,
    )
    .unwrap();
    assert_ne!(
        setup.matrix_view_digest(),
        repartitioned.matrix_view_digest()
    );
    let repartitioned_commit = PreparedCommitMatrix::prepare(&repartitioned).unwrap();
    let repartitioned_quotient = PreparedQuotientMatrix::prepare(&repartitioned).unwrap();
    assert!(matches!(
        kernel(
            &repartitioned_commit,
            &quotient,
            setup,
            &case.commitment,
            &challenges,
            &response,
            None
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
    assert!(matches!(
        kernel(
            &commit,
            &repartitioned_quotient,
            setup,
            &case.commitment,
            &challenges,
            &response,
            None
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
    // Altering n_A separately exercises the rank boundary, with valid inputs for the original setup.
    let other = support::common::setup::<F, 648, MinusTrinomial>(
        setup.n_a() + 1,
        setup.m(),
        setup.columns(),
    );
    let other_commit = PreparedCommitMatrix::prepare(&other).unwrap();
    let other_quotient = PreparedQuotientMatrix::prepare(&other).unwrap();
    assert!(matches!(
        kernel(
            &other_commit,
            &quotient,
            setup,
            &case.commitment,
            &challenges,
            &response,
            None
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
    assert!(matches!(
        kernel(
            &commit,
            &other_quotient,
            setup,
            &case.commitment,
            &challenges,
            &response,
            None
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}

fn schoolbook<F: SmoothFftField>(lhs: &[F], rhs: &[F]) -> Vec<F> {
    let mut product = vec![F::zero(); lhs.len() + rhs.len() - 1];
    for (i, &left) in lhs.iter().enumerate() {
        for (j, &right) in rhs.iter().enumerate() {
            product[i + j] += left * right;
        }
    }
    product
}

fn check_identity<F, const D: usize, M>(setup: &BinaryClearSetup<F, D, M>)
where
    F: support::common::TestPrime + WithPacking,
    M: ConjugateModulus,
{
    let (source, _, _) = support::common::data::<H>(setup.source_len(), setup.num_vars());
    let commitment = commit_binary_clear::<H, F, D, M>(setup, &source).unwrap();
    let challenges = BinaryChallengeSampler::new(setup.profile().clone())
        .sample_challenges(&mut FixedDraw, b"identity", setup.columns())
        .unwrap();
    let response = fold_integer::<H>(
        &source,
        setup.scalar_rows(),
        setup.columns(),
        &challenges,
        setup.profile(),
    )
    .unwrap();
    let commit = PreparedCommitMatrix::prepare(setup).unwrap();
    let quotient = PreparedQuotientMatrix::prepare(setup).unwrap();
    let actual = kernel(
        &commit,
        &quotient,
        setup,
        &commitment,
        &challenges,
        &response,
        None,
    )
    .unwrap()
    .quotients;
    assert_eq!(
        actual,
        reference(setup, &commitment, &challenges, &response).unwrap()
    );
    let packed = pack_response(setup, &response).unwrap();
    for (row, coefficients) in actual.iter().enumerate() {
        let mut residual = vec![F::zero(); 2 * D - 1];
        for (matrix, value) in setup.matrix()[row * setup.m()..(row + 1) * setup.m()]
            .iter()
            .zip(&packed)
        {
            for (sum, term) in residual
                .iter_mut()
                .zip(schoolbook(matrix.coefficients(), value.coefficients()))
            {
                *sum += term;
            }
        }
        for (column, challenge) in challenges.iter().enumerate() {
            let embedded =
                embed_scalar::<F, 162, D, M>(&challenge_scalar(challenge).unwrap()).unwrap();
            for (sum, term) in residual.iter_mut().zip(schoolbook(
                embedded.coefficients(),
                commitment.images[column * setup.n_a() + row].coefficients(),
            )) {
                *sum -= term;
            }
        }
        let mut phi = vec![F::zero(); D + 1];
        phi[0] = F::one();
        phi[D] = F::one();
        phi[D / 2] = if M::MIDDLE_COEFFICIENT == 1 {
            F::one()
        } else {
            -F::one()
        };
        assert_eq!(residual, schoolbook(coefficients, &phi));
    }
}

#[test]
fn independent_polynomial_identity_for_both_signs_and_coefficient_fields() {
    use akita_algebra::Prime64Offset23703;
    check_identity(&support::common::setup::<F, 162, PlusTrinomial>(2, 2, 2));
    check_identity(&support::common::setup::<F, 324, MinusTrinomial>(2, 2, 2));
    check_identity(&support::common::setup::<F, 648, MinusTrinomial>(2, 2, 2));
    check_identity(&support::common::setup::<
        Prime64Offset23703,
        162,
        PlusTrinomial,
    >(2, 2, 2));
    check_identity(&support::common::setup::<
        Prime64Offset23703,
        324,
        MinusTrinomial,
    >(2, 2, 2));
    check_identity(&support::common::setup::<
        Prime64Offset23703,
        648,
        MinusTrinomial,
    >(2, 2, 2));
}
