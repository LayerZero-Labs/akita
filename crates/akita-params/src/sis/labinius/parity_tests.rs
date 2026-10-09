//! Independent coefficient oracles for the direct parity row, without production
//! polynomial arithmetic or the closed-form quotient coefficients.

use num_bigint::BigInt;
use rand::{rngs::StdRng, Rng, SeedableRng};

use super::{LabiniusRootProfile, LabiniusRootShape};

const D: usize = 162;
const WEIGHT: usize = 46;
const LOWER: i128 = -32_768;
const UPPER: i128 = 32_767;

struct Instance {
    b: Vec<Vec<i128>>,
    v: Vec<Vec<i128>>,
    u: Vec<Vec<i128>>,
    c: Vec<Vec<i128>>,
}

fn integer_product(lhs: &[BigInt], rhs: &[BigInt]) -> Vec<BigInt> {
    let mut product = vec![BigInt::from(0); lhs.len() + rhs.len() - 1];
    for (i, left) in lhs.iter().enumerate() {
        for (j, right) in rhs.iter().enumerate() {
            product[i + j] += left * right;
        }
    }
    product
}

fn residual(instance: &Instance) -> Vec<BigInt> {
    let mut result = vec![BigInt::from(0); 2 * D - 1];
    for (lhs, rhs, sign) in instance
        .b
        .iter()
        .zip(&instance.v)
        .map(|(b, v)| (b, v, 1))
        .chain(instance.u.iter().zip(&instance.c).map(|(u, c)| (u, c, -1)))
    {
        let lhs: Vec<_> = lhs.iter().map(|&x| BigInt::from(x)).collect();
        let rhs: Vec<_> = rhs.iter().map(|&x| BigInt::from(x)).collect();
        for (target, value) in result.iter_mut().zip(integer_product(&lhs, &rhs)) {
            *target += value * sign;
        }
    }
    result
}

// Ordinary schoolbook division by the monic X^162 + X^81 + 1.
fn integer_division(input: &[BigInt]) -> (Vec<BigInt>, Vec<BigInt>) {
    let mut remainder = input.to_vec();
    let mut quotient = vec![BigInt::from(0); D - 1];
    for degree in (D..input.len()).rev() {
        let coefficient = remainder[degree].clone();
        let shift = degree - D;
        quotient[shift] = coefficient.clone();
        for exponent in [0, D / 2, D] {
            remainder[shift + exponent] -= &coefficient;
        }
    }
    assert!(remainder[D..].iter().all(|x| *x == BigInt::from(0)));
    remainder.truncate(D);
    (quotient, remainder)
}

// A separate Boolean oracle computes each product directly in F2[X]/Phi162.
// It never reads the integer residual or quotient.
fn binary_product(lhs: &[i128], rhs: &[i128]) -> Vec<bool> {
    let mut product = vec![false; 2 * D - 1];
    for (i, left) in lhs.iter().enumerate() {
        for (j, right) in rhs.iter().enumerate() {
            product[i + j] ^= left % 2 != 0 && right % 2 != 0;
        }
    }
    for degree in (D..product.len()).rev() {
        if product[degree] {
            product[degree] = false;
            product[degree - D] ^= true;
            product[degree - D / 2] ^= true;
        }
    }
    product.truncate(D);
    product
}

fn binary_residual(instance: &Instance) -> Vec<bool> {
    let mut sum = vec![false; D];
    for (lhs, rhs) in instance
        .b
        .iter()
        .zip(&instance.v)
        .chain(instance.u.iter().zip(&instance.c))
    {
        for (target, bit) in sum.iter_mut().zip(binary_product(lhs, rhs)) {
            *target ^= bit;
        }
    }
    sum
}

fn make_valid(instance: &mut Instance) {
    // Reserve C_0=1: the remaining RHS can be arbitrary, while its missing
    // canonical U_0 is the binary difference from the LHS.
    instance.c[0].fill(0);
    instance.c[0][0] = 1;
    instance.u[0].fill(0);
    instance.u[0] = binary_residual(instance)
        .into_iter()
        .map(i128::from)
        .collect();
    assert!(binary_residual(instance).iter().all(|bit| !bit));
}

fn check_oracles(instance: &Instance, log_m: u32, log_c: u32) -> bool {
    let shape = LabiniusRootShape::derive(
        LabiniusRootProfile::D648P128BoundedW46Delta16,
        log_m + log_c,
        log_c,
        128,
    )
    .expect("small oracle geometry is admitted");
    assert_eq!(instance.b.len(), shape.scalars_per_column());
    assert_eq!(instance.v.len(), shape.scalars_per_column());
    assert_eq!(instance.u.len(), shape.fold_width());
    assert_eq!(instance.c.len(), shape.fold_width());
    assert!(instance
        .b
        .iter()
        .chain(&instance.u)
        .flatten()
        .all(|&x| x == 0 || x == 1));
    assert!(instance
        .v
        .iter()
        .flatten()
        .all(|x| (LOWER..=UPPER).contains(x)));
    assert!(instance
        .c
        .iter()
        .all(|c| c.iter().map(|x| x.unsigned_abs()).sum::<u128>() <= 46));

    let input = residual(instance);
    let h = BigInt::from(shape.parity_residual_bound());
    assert!(input.iter().all(|x| x >= &(-&h) && x <= &h));
    let (quotient, remainder) = integer_division(&input);
    let q_bound = BigInt::from(shape.honest_quotient_bound());
    assert_eq!(q_bound, &h * 2);
    assert!(quotient.iter().all(|x| x >= &(-&q_bound) && x <= &q_bound));

    // Check the full characteristic-zero identity as well as the coefficient
    // envelopes, so a defective oracle division cannot pass on bounds alone.
    let mut phi = vec![BigInt::from(0); D + 1];
    for exponent in [0, D / 2, D] {
        phi[exponent] = BigInt::from(1);
    }
    let mut reconstructed = integer_product(&quotient, &phi);
    for (coefficient, tail) in reconstructed.iter_mut().zip(&remainder) {
        *coefficient += tail;
    }
    assert_eq!(reconstructed, input);

    let all_even = remainder.iter().all(|x| x % 2 == BigInt::from(0));
    let binary = binary_residual(instance);
    assert_eq!(all_even, binary.iter().all(|bit| !bit));
    for (coefficient, bit) in remainder.iter().zip(binary) {
        assert_eq!(coefficient % 2 != BigInt::from(0), bit);
    }
    if all_even {
        let carry_bound = BigInt::from(shape.honest_carry_bound());
        assert_eq!(carry_bound, &h * 5 / 2);
        for coefficient in remainder {
            let carry = coefficient / 2;
            assert!(carry >= -&carry_bound && carry <= carry_bound);
        }
    }
    all_even
}

fn random_instance(rng: &mut StdRng, m: usize, columns: usize) -> Instance {
    let b = (0..m)
        .map(|_| (0..D).map(|_| rng.gen_range(0..=1)).collect())
        .collect();
    let v = (0..m)
        .map(|_| (0..D).map(|_| rng.gen_range(LOWER..=UPPER)).collect())
        .collect();
    let u = (0..columns)
        .map(|_| (0..D).map(|_| rng.gen_range(0..=1)).collect())
        .collect();
    let c = (0..columns)
        .map(|_| {
            let mut polynomial = vec![0; D];
            let mut populated = 0;
            while populated < WEIGHT {
                let position = rng.gen_range(0..D);
                if polynomial[position] == 0 {
                    polynomial[position] = if rng.gen_bool(0.5) { 1 } else { -1 };
                    populated += 1;
                }
            }
            polynomial
        })
        .collect();
    Instance { b, v, u, c }
}

#[test]
fn random_direct_parity_big_integer_and_binary_oracles_agree() {
    let mut rng = StdRng::seed_from_u64(0x0002_4316_2648);
    for (log_m, log_c) in [(2, 0), (2, 1), (3, 2)] {
        for _ in 0..4 {
            let mut instance = random_instance(&mut rng, 1 << log_m, 1 << log_c);
            check_oracles(&instance, log_m, log_c);
            make_valid(&mut instance);
            assert!(check_oracles(&instance, log_m, log_c));
            // C_0=1 makes this one-bit partial mutation a nonzero F162 error.
            instance.u[0][0] ^= 1;
            assert!(!check_oracles(&instance, log_m, log_c));
        }
    }
}

#[test]
fn extremal_direct_parity_endpoint_and_challenge_sign_patterns() {
    for endpoint in [LOWER, UPPER] {
        for sign in [-1, 1] {
            for (log_m, log_c) in [(2, 0), (2, 2), (3, 1)] {
                let m = 1 << log_m;
                let columns = 1 << log_c;
                let mut challenge = vec![0; D];
                challenge[..WEIGHT].fill(sign);
                let mut instance = Instance {
                    b: vec![vec![1; D]; m],
                    v: vec![vec![endpoint; D]; m],
                    u: vec![vec![1; D]; columns],
                    c: vec![challenge; columns],
                };
                // Negative challenges reinforce the positive endpoint; positive
                // challenges reinforce the negative endpoint in B*V - U*C.
                check_oracles(&instance, log_m, log_c);
                make_valid(&mut instance);
                assert!(check_oracles(&instance, log_m, log_c));
            }
        }
    }
}
