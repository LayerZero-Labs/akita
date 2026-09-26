use super::{portable_fold_in_place, portable_round_product, PackedBinary162};
use crate::binary::{tests::oracle, BinaryField162 as F};

fn sample_values(len: usize, mut state: u64) -> Vec<F> {
    (0..len)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            F::from_words([
                state ^ index as u64,
                state.rotate_left(23),
                state.rotate_right(11) & F::TOP_MASK,
            ])
            .unwrap()
        })
        .collect()
}

fn oracle_claim(lhs: &[F], rhs: &[F]) -> F {
    lhs.iter()
        .zip(rhs)
        .fold(F::ZERO, |sum, (&a, &b)| sum + oracle(a, b))
}

fn oracle_round(lhs: &[F], rhs: &[F]) -> [F; 3] {
    let mut coefficients = [F::ZERO; 3];
    let mut index = 0;
    while index < lhs.len() {
        let a0 = lhs[index];
        let b0 = rhs[index];
        let a1 = lhs.get(index + 1).copied().unwrap_or(F::ZERO);
        let b1 = rhs.get(index + 1).copied().unwrap_or(F::ZERO);
        let da = a0 + a1;
        let db = b0 + b1;
        coefficients[0] += oracle(a0, b0);
        coefficients[1] += oracle(a0, db) + oracle(da, b0);
        coefficients[2] += oracle(da, db);
        index += 2;
    }
    coefficients
}

fn oracle_fold(values: &[F], r: F) -> Vec<F> {
    values
        .chunks(2)
        .map(|pair| {
            let even = pair[0];
            let odd = pair.get(1).copied().unwrap_or(F::ZERO);
            even + oracle(even + odd, r)
        })
        .collect()
}

#[test]
fn packed_storage_refills_and_handles_terminal_boundaries() {
    let values = sample_values(9, 0x83a5_921c_44f0_7d19);
    let mut packed = PackedBinary162::new();
    assert!(packed.is_empty());
    assert_eq!(packed.get(0), None);
    assert_eq!(packed.round_product(&packed, F::ZERO), None);
    packed.fold_in_place(values[0]);
    assert!(packed.is_empty());

    packed.refill(&values[..1]);
    assert_eq!(packed.to_scalars(), values[..1]);
    assert_eq!(
        packed.round_product(&packed, oracle(values[0], values[0])),
        None
    );
    packed.fold_in_place(values[1]);
    assert_eq!(packed.to_scalars(), values[..1]);

    packed.refill(&values[..5]);
    packed.fold_in_place(values[5]);
    assert_eq!(packed.to_scalars(), oracle_fold(&values[..5], values[5]));
    assert_eq!(packed.len(), 3);

    packed.refill(&values);
    assert_eq!(packed.len(), values.len());
    assert_eq!(packed.get(8), Some(values[8]));
    assert_eq!(packed.get(9), None);
    let shorter = PackedBinary162::from_scalars(&values[..8]);
    assert_eq!(packed.round_product(&shorter, F::ZERO), None);
    let words = [0u128, u128::MAX, 1 << 127, 1 << 63];
    let mut directly_packed = PackedBinary162::new();
    directly_packed.refill_binary_words(&words);
    let expected: Vec<_> = words
        .into_iter()
        .map(|x| F([x as u64, (x >> 64) as u64, 0]))
        .collect();
    assert_eq!(directly_packed.to_scalars(), expected);
    directly_packed.refill_binary_words(&[0u64, u64::MAX]);
    assert_eq!(directly_packed.to_scalars(), [F::ZERO, F([u64::MAX, 0, 0])]);
    directly_packed.refill_binary_words::<u64>(&[]);
    assert!(directly_packed.is_empty());
    let allocations = packed.words.each_ref().map(|words| words.as_ptr());
    while packed.len() > 1 {
        packed.fold_in_place(values[0]);
    }
    packed.refill(&values);
    assert_eq!(
        packed.words.each_ref().map(|words| words.as_ptr()),
        allocations
    );
    assert_eq!(packed.to_scalars(), values);
}

#[test]
fn packed_round_coefficients_and_hardware_match_oracle() {
    let lhs = sample_values(65, 0x1234_5678_9abc_def0);
    let rhs = sample_values(65, 0x0fed_cba9_8765_4321);
    for len in [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65] {
        let packed_lhs = PackedBinary162::from_scalars(&lhs[..len]);
        let packed_rhs = PackedBinary162::from_scalars(&rhs[..len]);
        let claim = oracle_claim(&lhs[..len], &rhs[..len]);
        let expected = oracle_round(&lhs[..len], &rhs[..len]);
        assert_eq!(expected[1] + expected[2], claim);
        assert_eq!(packed_lhs.round_product(&packed_rhs, claim), Some(expected));
        assert_eq!(
            portable_round_product(&packed_lhs, &packed_rhs, claim),
            expected
        );

        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("aes")
            && std::arch::is_aarch64_feature_detected!("pmull")
        {
            // SAFETY: the feature checks establish PMULL support.
            assert_eq!(
                unsafe { super::arm::round_product(&packed_lhs, &packed_rhs, claim) },
                expected
            );
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("pclmulqdq") {
            // SAFETY: the feature check establishes PCLMUL support.
            assert_eq!(
                unsafe { super::x86::round_product(&packed_lhs, &packed_rhs, claim) },
                expected
            );
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("pclmulqdq")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("vpclmulqdq")
        {
            // SAFETY: the feature checks establish every vector requirement.
            assert_eq!(
                unsafe { super::x86::round_product_vec2(&packed_lhs, &packed_rhs, claim) },
                expected
            );
            let r = lhs[len / 2];
            let expected_fold = oracle_fold(&lhs[..len], r);
            let mut folded = packed_lhs.clone();
            // SAFETY: the feature checks establish every vector requirement.
            unsafe { super::x86::fold_in_place_vec2(&mut folded, r) };
            assert_eq!(folded.to_scalars(), expected_fold);
        }
    }
}

#[test]
fn packed_full_round_sequence_matches_scalar_oracle() {
    let challenges = sample_values(4, 0x4b61_7a2d_ee90_135c);
    let dense = F::from_words([u64::MAX, u64::MAX, F::TOP_MASK]).unwrap();
    for (len, fill) in [(7, None), (11, None), (7, Some(F::ZERO)), (7, Some(dense))] {
        let mut lhs = sample_values(len, 0x0a74_2c91_538d_66eb ^ len as u64);
        let mut rhs = sample_values(len, 0x97f4_e268_1b35_dca0 ^ len as u64);
        if let Some(fill) = fill {
            lhs.fill(fill);
            rhs.fill(fill);
        }
        let mut packed_lhs = PackedBinary162::from_scalars(&lhs);
        let mut packed_rhs = PackedBinary162::from_scalars(&rhs);
        let mut claim = oracle_claim(&lhs, &rhs);

        for &r in &challenges {
            if lhs.len() == 1 {
                break;
            }
            let expected = oracle_round(&lhs, &rhs);
            assert_eq!(packed_lhs.round_product(&packed_rhs, claim), Some(expected));
            let r_squared = oracle(r, r);
            claim = expected[0] + oracle(expected[1], r) + oracle(expected[2], r_squared);

            let mut portable_lhs = packed_lhs.clone();
            portable_fold_in_place(&mut portable_lhs, r);
            lhs = oracle_fold(&lhs, r);
            rhs = oracle_fold(&rhs, r);
            packed_lhs.fold_in_place(r);
            packed_rhs.fold_in_place(r);
            assert_eq!(packed_lhs.to_scalars(), lhs);
            assert_eq!(portable_lhs.to_scalars(), lhs);
            assert_eq!(packed_rhs.to_scalars(), rhs);
            assert_eq!(claim, oracle_claim(&lhs, &rhs));
        }

        assert_eq!(packed_lhs.len(), 1);
        assert_eq!(packed_rhs.len(), 1);
        assert_eq!(packed_lhs.round_product(&packed_rhs, claim), None);
    }
}
