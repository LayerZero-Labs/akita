#![cfg(feature = "labinius")]
#[path = "root_reduction_support.rs"]
mod old;
#[path = "common/lifted.rs"]
mod support;
use akita_algebra::{binary::BinaryField192, MinusTrinomial, TrinomialRing};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear, commit_binary_clear_prepared, prove_binary_clear_bytes,
    quotient_kernel::a_relation_quotients,
};
use akita_labinius_verifier::{
    lowered::{a_row_residual, check_lowered_clear},
    verify_binary_clear_bytes,
};
use common::TestHost;
use jolt_field::{One, Ring};
use old::common;
use support::{Case, RelationCase, BASES, F, H};

fn roundtrips<T: TestHost>()
where
    T::Source: Sync,
{
    for fold in [0, 1] {
        for base in BASES {
            let case = Case::<T>::new(base, fold);
            assert_eq!(case.layout.n_a(), 3);
            let setup = case.admitted.setup();
            let prepared = commit_binary_clear_prepared::<T, F, 648, MinusTrinomial>(
                case.prepared.commit(),
                setup,
                &case.source,
            )
            .unwrap();
            assert_eq!(prepared, case.commitment);
            let clear = prove_binary_clear_bytes(
                setup,
                &case.source,
                &case.commitment,
                &case.point,
                case.value,
            )
            .unwrap();
            verify_binary_clear_bytes(setup, &case.commitment, &case.point, case.value, &clear)
                .unwrap();
            let (proof, claims) = case.prove();
            assert_eq!(case.verify(&proof).unwrap(), claims);
            assert_eq!(
                proof.len() - case.layout.witness_len(),
                akita_labinius_verifier::root::root_reduction_wire_size::<T>(
                    case.admitted.shape(),
                    base
                )
                .unwrap()
            );
            eprintln!(
                "tag1 host {} base {base:?} geometry (4,{fold},128) m={} C={} proof={}",
                T::ROWS,
                setup.m(),
                setup.columns(),
                proof.len()
            );
        }
    }
}
#[test]
fn clear_and_root_all_bases_both_hosts_rank_three() {
    roundtrips::<H>();
    roundtrips::<BinaryField192>();
}
#[test]
fn prepared_commitments_agree_on_all_ones_sources() {
    for fold in [0, 1] {
        let case = Case::<H>::new(BASES[1], fold);
        let source = vec![u128::MAX; case.source.len()];
        let reference =
            commit_binary_clear::<H, F, 648, MinusTrinomial>(case.admitted.setup(), &source)
                .unwrap();
        let prepared = commit_binary_clear_prepared::<H, F, 648, MinusTrinomial>(
            case.prepared.commit(),
            case.admitted.setup(),
            &source,
        )
        .unwrap();
        assert_eq!(reference, prepared);
        let case = Case::<BinaryField192>::new(BASES[1], fold);
        let source = vec![u64::MAX; case.source.len()];
        assert_eq!(
            commit_binary_clear::<BinaryField192, F, 648, MinusTrinomial>(
                case.admitted.setup(),
                &source
            )
            .unwrap(),
            commit_binary_clear_prepared::<BinaryField192, F, 648, MinusTrinomial>(
                case.prepared.commit(),
                case.admitted.setup(),
                &source
            )
            .unwrap()
        );
    }
}
#[test]
fn lifted_relation_rejects_each_changed_witness_and_both_carry_range_ends() {
    for base in BASES {
        let r = RelationCase::new(base);
        let public = r.public(&r.a).unwrap();
        check_lowered_clear(&r.case.layout, &public, &r.w, &r.case.image).unwrap();
        for row in 0..3 {
            assert_eq!(
                a_row_residual(
                    &r.case.layout,
                    &public,
                    r.case.admitted.setup(),
                    &r.case.commitment,
                    &r.response,
                    row
                )
                .unwrap(),
                F::from_u64(0)
            );
        }
        let (lo, hi) = r.case.layout.encoding().a_carry().unwrap().interval();
        for value in [lo, hi] {
            let mut a = r.a.clone();
            a.carry[0] = value;
            assert!(r.public(&a).is_ok(), "inclusive carry endpoint");
        }
        for value in [lo - 1, hi + 1] {
            let mut a = r.a.clone();
            a.carry[0] = value;
            assert!(
                matches!(r.public(&a), Err(AkitaError::InvalidProof)),
                "carry interval {value}"
            );
        }
        for row in 0..3 {
            let mut a = r.a.clone();
            a.carry[648 * row] += 1;
            let p = r.public(&a).unwrap();
            assert_eq!(
                check_lowered_clear(&r.case.layout, &p, &r.w, &r.case.image),
                Err(AkitaError::InvalidProof)
            );
            let mut a = r.a.clone();
            a.quotients[row][0] += F::one();
            let p = r.public(&a).unwrap();
            assert_eq!(
                check_lowered_clear(&r.case.layout, &p, &r.w, &r.case.image),
                Err(AkitaError::InvalidProof)
            );
        }
        let mut image = r.case.image.clone();
        image[0] += F::one();
        assert_eq!(
            check_lowered_clear(&r.case.layout, &public, &r.w, &image),
            Err(AkitaError::InvalidProof)
        );
        let mut response = r.response.clone();
        response[0][0] += 1;
        let w = akita_labinius_prover::lowered::encode_witness(&r.case.layout, &response).unwrap();
        assert_eq!(
            check_lowered_clear(&r.case.layout, &public, &w, &r.case.image),
            Err(AkitaError::InvalidProof)
        );
    }
}
#[test]
fn arbitrary_image_lift_compensated_by_carry_is_accepted() {
    // small-modulus-root.md, "What is range-checked": T is unrestricted.
    // Binding uses cancellation across children, not a range check on T.
    for base in BASES {
        let mut r = RelationCase::new(base);
        let original = r.a.clone();
        let q0 = i128::from(268_433_353u32);
        let mut coefficients = *r.case.commitment.images[0].coefficients();
        coefficients[0] += F::from_u128(q0 as u128);
        r.case.commitment.images[0] = TrinomialRing::from_coefficients(coefficients).unwrap();
        r.case.refresh_image();
        r.a = a_relation_quotients(
            r.case.prepared.commit(),
            r.case.prepared.quotient(),
            r.case.admitted.setup(),
            &r.case.commitment,
            &r.fold,
            &r.response,
            r.case.layout.encoding().a_carry(),
        )
        .unwrap();
        assert_eq!(r.a.quotients, original.quotients);
        let mut expected = original.carry.clone();
        for term in r.fold[0].terms() {
            let s = usize::from(term.position);
            expected[4 * s] -= i128::from(term.coefficient) * if s % 2 == 0 { 1 } else { -1 };
        }
        assert_eq!(r.a.carry, expected);
        check_lowered_clear(
            &r.case.layout,
            &r.public(&r.a).unwrap(),
            &r.w,
            &r.case.image,
        )
        .unwrap();
        let (proof, claims) = r.case.prove();
        assert_eq!(r.case.verify(&proof).unwrap(), claims);
    }
}
#[test]
fn ka_wire_tampering_truncation_extension_and_unused_bits_reject() {
    let case = Case::<H>::new(BASES[0], 1);
    let (proof, _) = case.prove();
    let (qa, ka, width) = case.a_wire();
    let bits = case.layout.encoding().a_carry().unwrap().bits();
    for position in [qa, ka, ka + 648 * width, ka + 1296 * width] {
        let mut changed = proof.clone();
        changed[position] ^= 1;
        assert!(
            matches!(case.verify(&changed), Err(AkitaError::InvalidProof)),
            "wire {position}"
        );
    }
    assert_ne!(bits % 8, 0);
    let mut changed = proof.clone();
    changed[ka + width - 1] |= 1 << (bits % 8);
    assert!(matches!(
        case.verify(&changed),
        Err(AkitaError::InvalidProof)
    ));
    let mut changed = proof.clone();
    changed.remove(ka + width * 1944 - 1);
    assert!(matches!(
        case.verify(&changed),
        Err(AkitaError::InvalidProof)
    ));
    let mut changed = proof.clone();
    changed.insert(ka + width * 1944, 0);
    assert!(matches!(
        case.verify(&changed),
        Err(AkitaError::InvalidProof)
    ));
}
#[test]
fn profile_replay_in_both_directions_rejects() {
    let small = Case::<H>::new(BASES[1], 1);
    let old = old::Case::<H>::new(BASES[1], 1);
    assert!(matches!(
        old.verify(&small.prove().0),
        Err(AkitaError::InvalidProof)
    ));
    assert!(matches!(
        small.verify(&old.prove().0),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn explicit_shared_prime_setup_cannot_claim_a_lifted_shape() {
    use akita_algebra::binary::BinaryField162 as B;
    use akita_labinius_verifier::{
        lowered::{LoweredPublic, LoweredRootLayout},
        BinaryClearSetup, BinaryEvaluationClaim,
    };
    use akita_params::sis::labinius::{LabiniusDigitBase, LabiniusRootShape};
    let profile = support::SMALL;
    let shape = LabiniusRootShape::derive(profile, 2, 0, 128).unwrap();
    let zero = TrinomialRing::<F, 648, MinusTrinomial>::zero().unwrap();
    let explicit = BinaryClearSetup::new(
        vec![zero; 3],
        3,
        1,
        1,
        -32768,
        32767,
        128,
        profile.challenge_profile().unwrap(),
        profile.coefficient_prime(),
        profile.ring_degree(),
    )
    .unwrap();
    assert!(
        matches!(LoweredRootLayout::new(&explicit,&shape,LabiniusDigitBase::Bits2),Err(AkitaError::InvalidSetup(message)) if message=="lowered setup and root shape disagree")
    );
    let admitted = support::Setup::derive(
        profile,
        2,
        0,
        128,
        akita_types::proof::AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .unwrap();
    let layout =
        LoweredRootLayout::new(admitted.setup(), &shape, LabiniusDigitBase::Bits2).unwrap();
    let challenges = akita_labinius_verifier::lowered::LoweredChallenges {
        alpha: F::from_u64(7),
        xi: F::from_u64(11),
        gamma: F::from_u64(13),
    };
    let claim = BinaryEvaluationClaim {
        point: vec![B::ZERO; 2],
        value: B::ZERO,
    };
    let fold = akita_challenges::BinaryChallengeSampler::new(profile.challenge_profile().unwrap())
        .sample_challenges(&mut common::FixedDraw, b"probe", 1)
        .unwrap();
    assert!(matches!(
        LoweredPublic::new(
            &layout,
            admitted.setup(),
            &claim,
            &[B::ZERO],
            &fold,
            &vec![vec![F::from_u64(0); 647]; 3],
            &[0; 161],
            &[0; 162],
            challenges
        ),
        Err(AkitaError::InvalidProof)
    ));
    let source = vec![0u128; 4];
    let commitment =
        commit_binary_clear::<H, F, 648, MinusTrinomial>(admitted.setup(), &source).unwrap();
    let image = akita_labinius_prover::lowered::flatten_image(&layout, &commitment).unwrap();
    let prepared = akita_labinius_prover::PreparedRootMatrices::prepare(admitted.setup()).unwrap();
    let point = vec![H::ZERO; 2];
    let (proof, claims) = akita_labinius_prover::prove_root_reduction_bytes(
        &admitted,
        &prepared,
        LabiniusDigitBase::Bits2,
        &source,
        &commitment,
        &point,
        H::ZERO,
        &mut akita_labinius_prover::TransparentRootProverOracle::new(&image),
    )
    .unwrap();
    assert_eq!(
        proof.len() - layout.witness_len(),
        akita_labinius_verifier::root::root_reduction_wire_size::<H>(
            &shape,
            LabiniusDigitBase::Bits2
        )
        .unwrap()
    );
    assert_eq!(
        akita_labinius_verifier::verify_root_reduction_bytes(
            &admitted,
            LabiniusDigitBase::Bits2,
            &point,
            H::ZERO,
            &mut akita_labinius_verifier::root::TransparentRootVerifierOracle::new(&image),
            &proof
        )
        .unwrap(),
        claims
    );
    let clear =
        prove_binary_clear_bytes(admitted.setup(), &source, &commitment, &point, H::ZERO).unwrap();
    verify_binary_clear_bytes(admitted.setup(), &commitment, &point, H::ZERO, &clear).unwrap();
}
