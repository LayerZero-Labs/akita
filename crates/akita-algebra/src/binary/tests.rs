use super::{product, BinaryField162 as F};

// Deliberately independent bit convolution and polynomial long division.
fn oracle(a: F, b: F) -> F {
    let a = a.to_words();
    let b = b.to_words();
    let mut bits = [false; 323];
    for i in 0..162 {
        for j in 0..162 {
            bits[i + j] ^= ((a[i / 64] >> (i % 64)) & (b[j / 64] >> (j % 64)) & 1) != 0;
        }
    }
    for i in (162..323).rev() {
        if bits[i] {
            bits[i - 162] ^= true;
            bits[i - 81] ^= true;
        }
    }
    let mut words = [0; 3];
    for i in 0..162 {
        words[i / 64] |= u64::from(bits[i]) << (i % 64);
    }
    F::from_words(words).unwrap()
}

#[test]
fn products_match_independent_polynomial_oracle() {
    // Include both reduction folds, all word boundaries and dense inputs.
    let mut values = vec![
        F::ZERO,
        F::ONE,
        F::from_words([u64::MAX, u64::MAX, (1 << 34) - 1]).unwrap(),
    ];
    for bit in [63, 64, 80, 81, 127, 128, 160, 161] {
        let mut words = [0; 3];
        words[bit / 64] = 1 << (bit % 64);
        values.push(F::from_words(words).unwrap());
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(162);
    use rand::{RngCore, SeedableRng};
    for _ in 0..12 {
        values.push(
            F::from_words([
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64() & ((1 << 34) - 1),
            ])
            .unwrap(),
        );
    }
    for a in &values {
        assert_eq!(a.square(), oracle(*a, *a));
        for b in &values {
            let expected = oracle(*a, *b);
            assert_eq!(*a * *b, expected);
            assert_eq!(F::reduce(product::portable_product(a.0, b.0)), expected);
        }
    }
}

#[test]
fn field_order_and_inverse() {
    assert_eq!(F::ZERO.inverse(), None);
    for words in [[1, 0, 0], [2, 0, 0], [u64::MAX, 7, 13]] {
        let a = F::from_words(words).unwrap();
        assert_eq!(a * a.inverse().unwrap(), F::ONE);
        assert_eq!(a.square_n(162), a);
    }
    // Phi_243 is irreducible over F_2 exactly when ord_243(2)=phi(243).
    // Check that order independently of the field arithmetic implementation.
    let mut order = 1u32;
    for exponent in 1..=162 {
        order = (2 * order) % 243;
        assert_eq!(order == 1, exponent == 162);
    }
}

#[test]
fn canonical_encoding_rejects_high_bits_and_wrong_lengths() {
    let a = F::from_words([u64::MAX, u64::MAX, (1 << 34) - 1]).unwrap();
    assert_eq!(F::from_bytes(&a.to_bytes()), Some(a));
    assert_eq!(a.to_bytes(), [vec![255; 20], vec![3]].concat().as_slice());
    assert_eq!(F::from_words([0, 0, 1 << 34]), None);
    assert_eq!(F::from_bytes(&[0; 20]), None);
    assert_eq!(F::from_bytes(&[0; 22]), None);
    let mut bytes = [0; 21];
    bytes[20] = 4;
    assert_eq!(F::from_bytes(&bytes), None);
}
