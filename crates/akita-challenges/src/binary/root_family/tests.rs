use super::*;
use crate::{FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw};
use shake::digest::{ExtendableOutput, Update, XofReader};
use shake::Shake256;

#[derive(Default)]
struct Draw {
    payload: Vec<u8>,
    seed: [u8; 32],
}

impl FoldDraw for Draw {
    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        self.payload = payload.to_vec();
        let mut hash = Shake256::default();
        hash.update(payload);
        hash.finalize_xof().read(&mut self.seed);
        Ok(self.seed)
    }
}

#[test]
fn entropy_and_prime_unit_condition_are_exact() {
    let p = u128::MAX - 274;
    let order = |base: u128| {
        let mut power = 1;
        (1..=162)
            .find(|_| {
                power = power * (base % 243) % 243;
                power == 1
            })
            .unwrap()
    };
    assert_eq!(order(p), 6);
    assert_eq!(order(2), 162);
    let bits = usize::BITS - (6 * WEIGHT).leading_zeros();
    assert_eq!(bits, 9);
    let left = u32::try_from(order(p)).unwrap() * p.ilog2();
    let right = u32::try_from(BinaryScalarRing::Cyclotomic243.degree() / 2).unwrap() * bits;
    assert_eq!(left, 762);
    assert_eq!(right, 729);
    assert!(left >= right);
    let mut choose = num_bigint::BigUint::from(1u8);
    let mut support_count = choose.clone();
    for i in 1..=WEIGHT {
        choose = choose * (163 - i) / i;
        support_count += &choose;
    }
    assert_eq!(&support_count, profile().unwrap().cardinality());
    assert_eq!(
        profile().unwrap().cardinality().to_string(),
        "107268495235892251993000716928106195079952"
    );
    assert!((entropy_bits(162) - 136.3002783090589).abs() < 1e-11);
    assert!(validate(162, 136).is_ok());
    assert!(validate(162, 137).is_err());
}

#[test]
fn ordinary_group_root_selects_binary_family_and_indexed_columns() {
    let family = FoldChallengeFamily::BinaryRoot;
    let mut draw = Draw::default();
    let challenges = draw
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            162,
            7,
            8,
            1,
            &family,
            None,
        )
        .unwrap();
    assert!(draw.payload.ends_with(IDENTITY));
    let reference = BinaryChallengeSampler::new(profile().unwrap())
        .sample_from_seed(&draw.seed, 8)
        .unwrap();
    for (actual, expected) in challenges.as_slice().iter().zip(reference) {
        assert_eq!(actual.positions.len(), expected.weight());
        for ((position, coefficient), term) in actual
            .positions
            .iter()
            .zip(&actual.coeffs)
            .zip(expected.terms())
        {
            assert_eq!(*position, u32::from(term.position));
            assert_eq!(*coefficient, term.coefficient);
        }
    }
    let again = Draw::default()
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            162,
            7,
            8,
            1,
            &family,
            None,
        )
        .unwrap();
    assert_eq!(challenges, again);
    for (group, blocks, claims) in [(8, 8, 1), (7, 9, 1), (7, 8, 2)] {
        let mut changed = Draw::default();
        changed
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                162,
                group,
                blocks,
                claims,
                &family,
                None,
            )
            .unwrap();
        assert_ne!(changed.seed, draw.seed);
    }
    let mut ordinary = Draw::default();
    ordinary
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            162,
            7,
            8,
            1,
            &FoldChallengeFamily::pm1_only(46),
            None,
        )
        .unwrap();
    assert_ne!(ordinary.seed, draw.seed);
    assert!(Draw::default()
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            486,
            7,
            8,
            1,
            &family,
            None,
        )
        .is_err());
    assert!(sample(&[0; 32], (1 << 20) + 1).is_err());
}

#[test]
fn distinct_supports_have_distinct_binary_images_including_zero() {
    let profile = profile().unwrap();
    let mut seen = std::collections::HashSet::new();
    let mut sign_input = Vec::new();
    let mut signs = Vec::new();
    // Degree < 162 representatives are already reduced modulo Phi243, so
    // their support bitsets are exactly their images in B. Enumerate all
    // supports in twelve coordinates, including zero, and every basis vector.
    let supports = (0..4096u16)
        .map(|mask| {
            (0..12)
                .filter(|i| mask & (1 << i) != 0)
                .collect::<Vec<u16>>()
        })
        .chain((12..162).map(|i| vec![i]));
    for support in supports {
        let challenge =
            crate::BinaryChallenge::from_support(&profile, &support, &mut sign_input, &mut signs);
        let image = challenge.canonical_support_encoding(&profile).unwrap();
        assert!(seen.insert(image.clone()));
        assert_eq!(image.iter().all(|byte| *byte == 0), support.is_empty());
        let replay =
            crate::BinaryChallenge::from_support(&profile, &support, &mut sign_input, &mut signs);
        assert_eq!(challenge, replay);
    }
    assert_eq!(seen.len(), 4096 + 150);
}

#[test]
fn binary_family_replays_on_the_callers_live_session() {
    use crate::{ProverFoldDraw, VerifierFoldDraw};
    use akita_transcript::{new_prover_channel, new_verifier_channel, receive_bytes, send_bytes};
    let mut prover = new_prover_channel(b"root-family/session", b"root-family/instance").unwrap();
    send_bytes(&mut prover, &[11, 12]);
    let expected = ProverFoldDraw::new(&mut prover, 0, 3)
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            162,
            3,
            8,
            1,
            &FoldChallengeFamily::BinaryRoot,
            None,
        )
        .unwrap();
    send_bytes(&mut prover, &[13, 14]);
    let proof = prover.narg_string().to_vec();
    let mut verifier =
        new_verifier_channel(b"root-family/session", b"root-family/instance", &proof).unwrap();
    assert_eq!(receive_bytes(&mut verifier, 2).unwrap(), [11, 12]);
    let actual = VerifierFoldDraw::new(&mut verifier, 0, 3)
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            162,
            3,
            8,
            1,
            &FoldChallengeFamily::BinaryRoot,
            None,
        )
        .unwrap();
    assert_eq!(expected, actual);
    assert_eq!(receive_bytes(&mut verifier, 2).unwrap(), [13, 14]);
    verifier.check_eof().unwrap();
}

#[test]
fn binary_family_encoding_is_explicit_and_roundtrips() {
    let binary = FoldChallengeFamily::BinaryRoot;
    let encoded = postcard::to_allocvec(&binary).unwrap();
    assert_eq!(
        postcard::from_bytes::<FoldChallengeFamily>(&encoded).unwrap(),
        binary
    );
    let json = serde_json::to_string(&binary).unwrap();
    assert_eq!(json, r#"{"count_pm1":4294967295,"count_pm2":1}"#);
    assert_eq!(
        serde_json::from_str::<FoldChallengeFamily>(&json).unwrap(),
        binary
    );
    let alias = FoldChallengeFamily::SignedSparse(crate::SparseChallengeConfig {
        count_pm1: u32::MAX as usize,
        count_pm2: 1,
    });
    assert!(postcard::to_allocvec(&alias).is_err());
    assert_eq!(binary.domain_separator_bytes(), IDENTITY);
}
