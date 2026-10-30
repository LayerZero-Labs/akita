use super::*;

fn source() -> BinarySourceDescriptor {
    BinarySourceDescriptor::new(
        BinaryIndexMap::new(8, 19, 8, 2).unwrap(),
        vec![[1; 32], [2; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32],
        },
    )
    .unwrap()
}
#[test]
fn canonical_descriptor_and_all_fixed_fields() {
    let source = source();
    let bytes = source.canonical_bytes();
    assert_eq!(BinarySourceDescriptor::from_bytes(&bytes).unwrap(), source);
    // Every fixed byte, including all identity tags and every interval endpoint.
    for i in (0..24).chain(64..93) {
        let mut bad = bytes.clone();
        bad[i] ^= 1;
        assert!(
            BinarySourceDescriptor::from_bytes(&bad).is_err(),
            "offset {i}"
        );
    }
    for len in 0..bytes.len() {
        assert!(BinarySourceDescriptor::from_bytes(&bytes[..len]).is_err());
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(BinarySourceDescriptor::from_bytes(&bad).is_err());
    for offset in [24, 32, 40, 48, 56] {
        let mut bad = bytes.clone();
        bad[offset..offset + 8].copy_from_slice(&0u64.to_le_bytes());
        assert!(BinarySourceDescriptor::from_bytes(&bad).is_err());
    }
    let mut bad = bytes.clone();
    bad[24..32].copy_from_slice(&3u64.to_le_bytes());
    assert!(BinarySourceDescriptor::from_bytes(&bad).is_err());
    let mut bad = bytes.clone();
    bad[56..64].copy_from_slice(&3u64.to_le_bytes());
    assert!(BinarySourceDescriptor::from_bytes(&bad).is_err());
    let mut bad = bytes.clone();
    bad[189..221].copy_from_slice(&[1; 32]);
    assert!(BinarySourceDescriptor::from_bytes(&bad).is_err());
}
#[test]
fn windows_are_exact_and_off_by_one_is_rejected() {
    let s8 = (8i32.pow(8) - 1) / 7;
    assert_eq!(
        CANONICAL_IMAGE_WINDOW,
        (
            -((BINARY_SOURCE_Q as i32 - 1) / 2),
            (BINARY_SOURCE_Q as i32 - 1) / 2
        )
    );
    assert_eq!(ENFORCED_IMAGE_WINDOW, (-4 * s8, 3 * s8));
    assert_eq!(
        BINARY_RESPONSE_INTERVAL,
        (-4 * ((8i32.pow(4) - 1) / 7), 3 * ((8i32.pow(4) - 1) / 7))
    );
    for (offset, value) in [
        (68, CANONICAL_IMAGE_WINDOW.0),
        (72, CANONICAL_IMAGE_WINDOW.1),
        (76, ENFORCED_IMAGE_WINDOW.0),
        (80, ENFORCED_IMAGE_WINDOW.1),
    ] {
        for delta in [-1, 1] {
            let mut bytes = source().canonical_bytes();
            bytes[offset..offset + 4].copy_from_slice(&(value + delta).to_le_bytes());
            assert!(BinarySourceDescriptor::from_bytes(&bytes).is_err());
        }
    }
}
#[test]
fn prime_cap_counts_owners_and_components_and_reuses_duplicates() {
    let s = source();
    let mut claims: Vec<_> = (0..5)
        .map(|i| PrimePlaneClaim {
            polynomial: 0,
            point: vec![i; 11],
            tensor_component: 0,
        })
        .collect();
    assert_eq!(
        PrimePlaneManifest::new(&s, &claims).unwrap().planes().len(),
        5
    );
    claims.push(claims[2].clone());
    assert_eq!(
        PrimePlaneManifest::new(&s, &claims).unwrap().assignments(),
        &[0, 1, 2, 3, 4, 2]
    );
    claims.push(PrimePlaneClaim {
        polynomial: 0,
        point: vec![5; 11],
        tensor_component: 0,
    });
    assert!(PrimePlaneManifest::new(&s, &claims).is_err());
    claims.pop();
    claims[5].polynomial = 1;
    assert!(PrimePlaneManifest::new(&s, &claims).is_err());
    claims[5].polynomial = 0;
    claims[5].tensor_component = 1;
    assert!(PrimePlaneManifest::new(&s, &claims).is_err());
    claims.truncate(1);
    claims[0].point[0] = u128::MAX - 274;
    assert!(PrimePlaneManifest::new(&s, &claims).is_err());
    claims[0].point.pop();
    assert!(PrimePlaneManifest::new(&s, &claims).is_err());
}
struct Token(SourceAgreementBinding);
impl SourceAgreementProof for Token {
    fn binding(&self) -> &SourceAgreementBinding {
        &self.0
    }
}
#[test]
fn agreement_binds_every_context_component() {
    let binding = SourceAgreementBinding {
        session: [1; 32],
        group: [2; 32],
        source: source(),
        terminal_prime_claims: [3; 32],
    };
    let opening = BinaryOpening::from_authenticated_readings(1u8, 1u128, binding.clone());
    assert!(opening.agreement_requires(&Token(binding.clone())).is_ok());
    let mut wrong = binding.clone();
    wrong.session = [9; 32];
    assert!(opening.agreement_requires(&Token(wrong)).is_err());
    let mut wrong = binding.clone();
    wrong.group = [9; 32];
    assert!(opening.agreement_requires(&Token(wrong)).is_err());
    let mut wrong = binding.clone();
    wrong.terminal_prime_claims = [9; 32];
    assert!(opening.agreement_requires(&Token(wrong)).is_err());
    let mut wrong = binding;
    wrong.source = BinarySourceDescriptor::new(
        BinaryIndexMap::new(16, 19, 8, 2).unwrap(),
        vec![[8; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32],
        },
    )
    .unwrap();
    assert!(opening.agreement_requires(&Token(wrong)).is_err());
}

#[test]
fn descriptor_rejects_overflow_and_invalid_relations() {
    let good = source().canonical_bytes();
    for (offset, values) in [
        (24, vec![0, 3, 129, u64::MAX]),
        (32, vec![0, 129, u64::MAX]),
        (40, vec![0, 1, 7, u64::MAX]),
        (48, vec![0, 3, 4, u64::MAX]),
        (56, vec![0, 1, 3, 4097, u64::MAX]),
    ] {
        for value in values {
            let mut bytes = good.clone();
            bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            assert!(
                BinarySourceDescriptor::from_bytes(&bytes).is_err(),
                "offset {offset}, value {value}"
            );
        }
    }
}

#[test]
fn agreement_rejects_each_descriptor_binding_change() {
    let source = source();
    let binding = SourceAgreementBinding {
        session: [1; 32],
        group: [2; 32],
        source: source.clone(),
        terminal_prime_claims: [3; 32],
    };
    let opening = BinaryOpening::from_authenticated_readings(1u8, 1u128, binding.clone());
    let changes = [
        BinarySourceDescriptor::new(
            BinaryIndexMap::new(16, 19, 8, 2).unwrap(),
            source.polynomials().to_vec(),
            source.profiles().clone(),
        )
        .unwrap(),
        BinarySourceDescriptor::new(
            BinaryIndexMap::new(8, 20, 8, 2).unwrap(),
            source.polynomials().to_vec(),
            source.profiles().clone(),
        )
        .unwrap(),
        BinarySourceDescriptor::new(
            BinaryIndexMap::new(8, 19, 16, 2).unwrap(),
            source.polynomials().to_vec(),
            source.profiles().clone(),
        )
        .unwrap(),
        BinarySourceDescriptor::new(
            BinaryIndexMap::new(8, 19, 8, 1).unwrap(),
            source.polynomials().to_vec(),
            source.profiles().clone(),
        )
        .unwrap(),
        BinarySourceDescriptor::new(
            source.index_map().clone(),
            vec![[2; 32], [1; 32]],
            source.profiles().clone(),
        )
        .unwrap(),
        BinarySourceDescriptor::new(
            source.index_map().clone(),
            source.polynomials().to_vec(),
            BinaryCommitmentProfiles {
                a: [9; 32],
                b: [4; 32],
            },
        )
        .unwrap(),
    ];
    for source in changes {
        let mut wrong = binding.clone();
        wrong.source = source;
        assert!(opening.agreement_requires(&Token(wrong)).is_err());
    }
}

#[test]
fn group_dimensions_cannot_overflow_across_polynomials() {
    let map = BinaryIndexMap::new(128, 1, 1usize << (usize::BITS - 9), 1).unwrap();
    assert!(BinarySourceDescriptor::new(
        map,
        vec![[1; 32], [2; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32]
        }
    )
    .is_err());
}
