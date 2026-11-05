use super::*;

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
fn binary_family_encoding_is_explicit_and_roundtrips() {
    let binary = FoldChallengeDistribution::BinaryRoot;
    let encoded = postcard::to_allocvec(&binary).unwrap();
    assert_eq!(
        postcard::from_bytes::<FoldChallengeDistribution>(&encoded).unwrap(),
        binary
    );
    let json = serde_json::to_string(&binary).unwrap();
    assert_eq!(json, r#"{"count_pm1":4294967295,"count_pm2":1}"#);
    assert_eq!(
        serde_json::from_str::<FoldChallengeDistribution>(&json).unwrap(),
        binary
    );
    let alias = FoldChallengeDistribution::SignedSparse(jolt_crypto::challenge::SparseChallengeConfig {
        count_pm1: u32::MAX as usize,
        count_pm2: 1,
    });
    assert!(postcard::to_allocvec(&alias).is_err());
    assert_eq!(binary.ordering_descriptor_bytes(), IDENTITY);
}
