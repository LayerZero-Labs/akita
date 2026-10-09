#![cfg(feature = "labinius")]

#[path = "lowered_support.rs"]
mod support;
use akita_algebra::{
    binary::BinaryField162 as B, MinusTrinomial, PlusTrinomial, Prime64Offset23703, TrinomialRing,
};
use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_error::AkitaError;
use akita_labinius_prover::lowered::{encode_witness, flatten_image, parity_quotient_and_carry};
use akita_labinius_verifier::{
    commitment::apply_matrix,
    endpoint::{pack_response, response_parity, verify_endpoints},
    lowered::{check_lowered_clear, LoweredRootLayout},
    BinaryClearSetup,
};
use akita_params::sis::labinius::{
    LabiniusCoefficientPrime as Prime, LabiniusRingDegree as Degree,
};
use jolt_field::Ring;
#[path = "a_relation_support.rs"]
mod a_relation_support;
use a_relation_support::a_relation_remainders;
use support::*;

#[test]
fn differential_response_image_and_public_expansion_mutations_reject() {
    for base in BASES {
        let case = Case::new(base);
        let public = case.public();
        let mut commitment = case.commitment.clone();
        commitment.images[0] += TrinomialRing::one().unwrap();
        assert!(verify_endpoints(
            &case.setup,
            &commitment,
            &case.claim,
            &case.u,
            &case.fold,
            &case.response
        )
        .is_err());
        let y = flatten_image(&case.layout, &commitment).unwrap();
        assert!(check_lowered_clear(&case.layout, &public, &case.setup, &case.w, &y).is_err());
        assert!(
            a_relation_remainders(&case.setup, &commitment, &case.fold, &case.response)
                .unwrap()
                .iter()
                .flatten()
                .any(|x| *x != F::from_u64(0))
        );
        for delta in [1, 2] {
            let mut response = case.response.clone();
            response[1][0] += delta;
            if delta == 2 {
                assert_eq!(
                    response_parity(&response[1]).unwrap(),
                    response_parity(&case.response[1]).unwrap()
                );
            }
            assert!(verify_endpoints(
                &case.setup,
                &case.commitment,
                &case.claim,
                &case.u,
                &case.fold,
                &response
            )
            .is_err());
            let w = encode_witness(&case.layout, &response).unwrap();
            assert!(check_lowered_clear(&case.layout, &public, &case.setup, &w, &case.y).is_err());
            assert!(
                a_relation_remainders(&case.setup, &case.commitment, &case.fold, &response)
                    .unwrap()
                    .iter()
                    .flatten()
                    .any(|x| *x != F::from_u64(0))
            );
        }
        // Keep the left-expansion check unchanged, isolating parity's U term.
        let mut claim = case.claim.clone();
        claim.point[case.setup.row_vars()] = B::ZERO;
        claim.value = case.u[0];
        let (q, k) =
            parity_quotient_and_carry(&case.setup, &claim, &case.u, &case.fold, &case.response)
                .unwrap();
        let mut u = case.u.clone();
        u[1] += B::ONE;
        assert!(verify_endpoints(
            &case.setup,
            &case.commitment,
            &claim,
            &u,
            &case.fold,
            &case.response
        )
        .is_err());
        let public = case.public_with(&claim, &u, &q, &k).unwrap();
        assert!(check_lowered_clear(&case.layout, &public, &case.setup, &case.w, &case.y).is_err());
        assert!(
            parity_quotient_and_carry(&case.setup, &claim, &u, &case.fold, &case.response).is_err()
        );
    }
}

#[test]
fn clear_quotient_carry_and_alphabet_mutations_reject() {
    for base in BASES {
        let case = Case::new(base);
        let check = |q: &[i128], k: &[i128]| {
            let public = case.public_with(&case.claim, &case.u, q, k).unwrap();
            assert!(
                check_lowered_clear(&case.layout, &public, &case.setup, &case.w, &case.y).is_err()
            );
        };
        let mut q = case.q.clone();
        q[0] += 1;
        check(&q, &case.k);
        let mut k = case.k.clone();
        k[0] += 1;
        check(&case.q, &k);
        for family in [0, 1] {
            let interval = if family == 0 {
                case.layout.encoding().quotient().interval()
            } else {
                case.layout.encoding().carry().interval()
            };
            for outside in [interval.0 - 1, interval.1 + 1] {
                let mut q = case.q.clone();
                let mut k = case.k.clone();
                if family == 0 {
                    q[0] = outside;
                } else {
                    k[0] = outside;
                }
                assert!(matches!(
                    case.public_with(&case.claim, &case.u, &q, &k),
                    Err(AkitaError::InvalidProof)
                ));
            }
        }
        let mut w = case.w.clone();
        w[0] = 1 << base.bits();
        assert!(
            check_lowered_clear(&case.layout, &case.public(), &case.setup, &w, &case.y).is_err()
        );
        assert!(case
            .public_with(&case.claim, &case.u, &case.q[..160], &case.k)
            .is_err());
        assert!(case
            .public_with(&case.claim, &case.u, &case.q, &case.k[..161])
            .is_err());
        let mut public_claim = case.claim.clone();
        public_claim.point.pop();
        assert!(case
            .public_with(&public_claim, &case.u, &case.q, &case.k)
            .is_err());
        assert!(case
            .public_with(&case.claim, &case.u[..1], &case.q, &case.k)
            .is_err());
    }
}

#[test]
fn packed_component_cancellation_keeps_prime_but_breaks_parity() {
    for base in BASES {
        let mut case = Case::new(base);
        let mut matrix = case.setup.matrix().to_vec();
        for row in matrix.chunks_exact_mut(2) {
            row[1] = row[0];
        }
        case.setup = Setup::new(
            matrix,
            case.setup.n_a(),
            2,
            2,
            -32768,
            32767,
            128,
            PROFILE.challenge_profile().unwrap(),
            PROFILE.coefficient_prime(),
            PROFILE.ring_degree(),
        )
        .unwrap();
        case.layout = LoweredRootLayout::new(&case.setup, &case.shape, base).unwrap();
        case.commitment.images.fill(TrinomialRing::zero().unwrap());
        case.response.fill([0; 162]);
        case.u.fill(B::ZERO);
        case.claim.point = vec![B::ONE, B::ZERO, B::ZERO, B::ZERO];
        case.claim.value = B::ZERO;
        (case.q, case.k) = parity_quotient_and_carry(
            &case.setup,
            &case.claim,
            &case.u,
            &case.fold,
            &case.response,
        )
        .unwrap();
        case.w = encode_witness(&case.layout, &case.response).unwrap();
        case.y = flatten_image(&case.layout, &case.commitment).unwrap();
        case.verify().unwrap();
        let public = case.public();
        case.response[1][0] = 1;
        case.response[5][0] = -1;
        let image = apply_matrix(
            &case.setup,
            &pack_response(&case.setup, &case.response).unwrap(),
        )
        .unwrap();
        assert!(image.iter().all(|x| *x == TrinomialRing::zero().unwrap()));
        assert_eq!(
            a_relation_remainders(&case.setup, &case.commitment, &case.fold, &case.response)
                .unwrap(),
            vec![vec![F::from_u64(0); 648]; case.setup.n_a()]
        );
        assert!(case.verify().is_err());
        assert!(parity_quotient_and_carry(
            &case.setup,
            &case.claim,
            &case.u,
            &case.fold,
            &case.response
        )
        .is_err());
        let w = encode_witness(&case.layout, &case.response).unwrap();
        assert!(check_lowered_clear(&case.layout, &public, &case.setup, &w, &case.y).is_err());
    }
}

#[test]
fn layout_rejects_each_constructible_setup_shape_mismatch() {
    let case = Case::new(BASES[0]);
    let n = case.setup.n_a();
    let build = |n, m, c, lo, hi, profile| {
        Setup::new(
            vec![TrinomialRing::one().unwrap(); n * m],
            n,
            m,
            c,
            lo,
            hi,
            128,
            profile,
            Prime::P128OffsetA7F7,
            Degree::D648,
        )
        .unwrap()
    };
    for setup in [
        build(n + 1, 2, 2, -32768, 32767, case.setup.profile().clone()),
        build(n, 4, 2, -32768, 32767, case.setup.profile().clone()),
        build(n, 2, 4, -32768, 32767, case.setup.profile().clone()),
        build(n, 2, 2, -32767, 32767, case.setup.profile().clone()),
        build(n, 2, 2, -32768, 32768, case.setup.profile().clone()),
        build(
            n,
            2,
            2,
            -32768,
            32767,
            BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 47).unwrap(),
        ),
    ] {
        assert!(matches!(
            LoweredRootLayout::new(&setup, &case.shape, BASES[0]),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
    let setup64 = BinaryClearSetup::<Prime64Offset23703, 648, MinusTrinomial>::new(
        vec![TrinomialRing::one().unwrap(); n * 2],
        n,
        2,
        2,
        -32768,
        32767,
        128,
        case.setup.profile().clone(),
        Prime::P64Offset23703,
        Degree::D648,
    )
    .unwrap();
    assert!(LoweredRootLayout::new(&setup64, &case.shape, BASES[0]).is_err());
    let setup324 = BinaryClearSetup::<F, 324, MinusTrinomial>::new(
        vec![TrinomialRing::one().unwrap(); n * 2],
        n,
        2,
        2,
        -32768,
        32767,
        128,
        case.setup.profile().clone(),
        Prime::P128OffsetA7F7,
        Degree::D324,
    )
    .unwrap();
    assert!(LoweredRootLayout::new(&setup324, &case.shape, BASES[0]).is_err());
    let setup162 = BinaryClearSetup::<F, 162, PlusTrinomial>::new(
        vec![TrinomialRing::one().unwrap(); n * 2],
        n,
        2,
        2,
        -32768,
        32767,
        128,
        case.setup.profile().clone(),
        Prime::P128OffsetA7F7,
        Degree::D162,
    )
    .unwrap();
    assert!(LoweredRootLayout::new(&setup162, &case.shape, BASES[0]).is_err());
    // Public constructors already prevent non-dyadic m, inconsistent scalar
    // rows, wrong field IDs and packing signs. The closed profile fixes dc.
    assert!(Setup::new(
        vec![TrinomialRing::one().unwrap(); n * 3],
        n,
        3,
        2,
        -32768,
        32767,
        128,
        case.setup.profile().clone(),
        Prime::P128OffsetA7F7,
        Degree::D648
    )
    .is_err());
    assert!(Setup::new(
        case.setup.matrix().to_vec(),
        n,
        2,
        2,
        -32768,
        32767,
        128,
        case.setup.profile().clone(),
        Prime::P64Offset23703,
        Degree::D648
    )
    .is_err());
    assert!(case.layout.sigma(648).is_err());
    assert!(case.layout.off(648).is_err());
    assert!(case
        .layout
        .image_address(case.layout.image_count(), 0)
        .is_err());
    assert!(case.layout.response_layout().address(2, 0, 0).is_err());
    assert!(case.layout.response_layout().address(0, 16, 0).is_err());
}
