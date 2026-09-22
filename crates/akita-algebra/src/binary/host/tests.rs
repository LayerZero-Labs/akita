use super::{BinaryField128 as F128, BinaryField192 as F192};

fn oracle128(a: F128, b: F128) -> F128 {
    let a = a.to_words();
    let b = b.to_words();
    let mut bits = [false; 255];
    for i in 0..128 {
        for j in 0..128 {
            bits[i + j] ^= ((a[i / 64] >> (i % 64)) & (b[j / 64] >> (j % 64)) & 1) != 0;
        }
    }
    for i in (128..255).rev() {
        if bits[i] {
            for offset in [0, 1, 2, 7] {
                bits[i - 128 + offset] ^= true;
            }
        }
    }
    F128::from_words(std::array::from_fn(|word| {
        (0..64).fold(0, |value, bit| {
            value | (u64::from(bits[64 * word + bit]) << bit)
        })
    }))
}

fn oracle64(a: u64, b: u64) -> u64 {
    let mut bits = [false; 127];
    for i in 0..64 {
        for j in 0..64 {
            bits[i + j] ^= ((a >> i) & (b >> j) & 1) != 0;
        }
    }
    for i in (64..127).rev() {
        if bits[i] {
            for offset in [0, 1, 3, 4] {
                bits[i - 64 + offset] ^= true;
            }
        }
    }
    (0..64).fold(0, |value, bit| value | (u64::from(bits[bit]) << bit))
}

fn oracle192(a: F192, b: F192) -> F192 {
    let a = a.to_words();
    let b = b.to_words();
    let mut coefficients = [0u64; 5];
    for i in 0..3 {
        for j in 0..3 {
            coefficients[i + j] ^= oracle64(a[i], b[j]);
        }
    }
    for degree in (3..=4).rev() {
        coefficients[degree - 3] ^= coefficients[degree];
        coefficients[degree - 2] ^= coefficients[degree];
    }
    F192::from_words([coefficients[0], coefficients[1], coefficients[2]])
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn polynomial_degree(polynomial: &[bool]) -> Option<usize> {
    polynomial.iter().rposition(|&coefficient| coefficient)
}

fn polynomial_remainder(mut dividend: Vec<bool>, divisor: &[bool]) -> Vec<bool> {
    let divisor_degree = polynomial_degree(divisor).unwrap();
    while let Some(dividend_degree) = polynomial_degree(&dividend) {
        if dividend_degree < divisor_degree {
            break;
        }
        let shift = dividend_degree - divisor_degree;
        for (i, &coefficient) in divisor.iter().take(divisor_degree + 1).enumerate() {
            dividend[shift + i] ^= coefficient;
        }
    }
    dividend.truncate(divisor_degree);
    dividend
}

fn polynomial_gcd(mut a: Vec<bool>, mut b: Vec<bool>) -> Vec<bool> {
    while polynomial_degree(&b).is_some() {
        let remainder = polynomial_remainder(a, &b);
        a = b;
        b = remainder;
    }
    a
}

fn polynomial_square_mod(value: &[bool], modulus: &[bool]) -> Vec<bool> {
    let mut square = vec![false; 2 * value.len() - 1];
    for (i, &coefficient) in value.iter().enumerate() {
        square[2 * i] = coefficient;
    }
    polynomial_remainder(square, modulus)
}

fn assert_rabin_irreducible(degree: usize, low_terms: &[usize]) {
    let mut modulus = vec![false; degree + 1];
    modulus[degree] = true;
    for &term in low_terms {
        modulus[term] = true;
    }
    let mut x = vec![false; degree];
    x[1] = true;
    let mut power = x.clone();
    for exponent in 1..=degree {
        power = polynomial_square_mod(&power, &modulus);
        if exponent == degree / 2 {
            let mut difference = power.clone();
            difference[1] ^= true;
            assert_eq!(
                polynomial_degree(&polynomial_gcd(difference, modulus.clone())),
                Some(0)
            );
        }
    }
    assert_eq!(power, x);
}

#[test]
fn binary128_matches_long_division_oracle() {
    let mut values = vec![
        F128::ZERO,
        F128::ONE,
        F128::from_words([u64::MAX, u64::MAX]),
    ];
    for bit in [1, 2, 7, 63, 64, 120, 126, 127] {
        let mut words = [0; 2];
        words[bit / 64] = 1 << (bit % 64);
        values.push(F128::from_words(words));
    }
    let mut state = 0x1280_5eed_cafe_babe;
    for _ in 0..12 {
        values.push(F128::from_words([next(&mut state), next(&mut state)]));
    }

    for &a in &values {
        assert_eq!(a + F128::ZERO, a);
        assert_eq!(a + a, F128::ZERO);
        assert_eq!(a * F128::ONE, a);
        for &b in &values {
            assert_eq!(a * b, oracle128(a, b));
        }
    }
}

#[test]
fn binary128_basis_and_frobenius() {
    assert_eq!(
        F128::from_words([0, 1 << 63]).mul_x(),
        F128::from_words([0x87, 0])
    );

    for mut value in [
        F128::from_words([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]),
        F128::from_words([u64::MAX, u64::MAX]),
    ] {
        let original = value;
        for _ in 0..128 {
            value = value * value;
        }
        assert_eq!(value, original);
    }
}

#[test]
fn base_moduli_are_irreducible() {
    // Rabin's criterion has one proper-prime-divisor check because 64 and 128
    // are powers of two. This uses polynomial long division independent of the
    // field implementations above.
    assert_rabin_irreducible(64, &[0, 1, 3, 4]);
    assert_rabin_irreducible(128, &[0, 1, 2, 7]);
}

#[test]
fn binary192_matches_long_division_oracle() {
    let mut values = vec![
        F192::ZERO,
        F192::ONE,
        F192::from_words([u64::MAX; 3]),
        F192::from_words([0, 1, 0]),
        F192::from_words([0, 0, 1]),
    ];
    for coefficient in 0..3 {
        for bit in [1, 3, 4, 62, 63] {
            let mut words = [0; 3];
            words[coefficient] = 1 << bit;
            values.push(F192::from_words(words));
        }
    }
    let mut state = 0x1920_5eed_dead_beef;
    for _ in 0..12 {
        values.push(F192::from_words([
            next(&mut state),
            next(&mut state),
            next(&mut state),
        ]));
    }

    for &a in &values {
        assert_eq!(a + F192::ZERO, a);
        assert_eq!(a + a, F192::ZERO);
        assert_eq!(a * F192::ONE, a);
        for &b in &values {
            assert_eq!(a * b, oracle192(a, b));
        }
    }
}

#[test]
fn binary192_embedding_and_basis_boundaries() {
    let a = 0x0123_4567_89ab_cdef;
    let b = 0xfedc_ba98_7654_3210;
    assert_eq!(
        F192::from_words([a, 0, 0]) * F192::from_words([b, 0, 0]),
        F192::from_words([oracle64(a, b), 0, 0])
    );
    assert_eq!(
        F192::from_words([1 << 63, 1 << 63, 1 << 63]).mul_x(),
        F192::from_words([0x1b; 3])
    );

    let y = F192::from_words([0, 1, 0]);
    let y_squared = F192::from_words([0, 0, 1]);
    assert_eq!(y_squared.mul_y(), F192::from_words([1, 1, 0]));
    assert_eq!(y_squared * y, F192::from_words([1, 1, 0]));
}

#[test]
fn binary192_frobenius_and_field_order() {
    for value in [
        F192::from_words([
            0x0123_4567_89ab_cdef,
            0xfedc_ba98_7654_3210,
            0x55aa_33cc_0ff0_f00f,
        ]),
        F192::from_words([u64::MAX; 3]),
    ] {
        let mut frobenius = value;
        for _ in 0..192 {
            frobenius = frobenius * frobenius;
        }
        assert_eq!(frobenius, value);

        let mut order = F192::ONE;
        for _ in 0..192 {
            order = order * order * value;
        }
        assert_eq!(order, F192::ONE);
    }
}
