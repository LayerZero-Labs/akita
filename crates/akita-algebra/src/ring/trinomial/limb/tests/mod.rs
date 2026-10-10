use super::*;

mod accumulator;
mod bounds;
mod transform;
mod validation;

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn centered(value: i128, prime: u32) -> i32 {
    let p = i128::from(prime);
    let residue = value.rem_euclid(p);
    (if residue > p / 2 {
        residue - p
    } else {
        residue
    }) as i32
}

fn random_coefficients(prime: u32, seed: &mut u64) -> [i32; DEGREE] {
    std::array::from_fn(|_| centered(i128::from(next(seed)), prime))
}

fn random_bits(seed: &mut u64) -> [i32; DEGREE] {
    std::array::from_fn(|_| (next(seed) & 1) as i32)
}

// Independent signed convolution; descending reduction applies
// X^648 = X^324 - 1 before taking any residue modulo the prime.
fn schoolbook(lhs: &[i32; DEGREE], rhs: &[i32; DEGREE], prime: u32) -> [i32; DEGREE] {
    let mut convolution = [0i128; 2 * DEGREE - 1];
    for (i, &a) in lhs.iter().enumerate() {
        for (j, &b) in rhs.iter().enumerate() {
            convolution[i + j] += i128::from(a) * i128::from(b);
        }
    }
    for degree in (DEGREE..convolution.len()).rev() {
        let value = convolution[degree];
        convolution[degree - 324] += value;
        convolution[degree - DEGREE] -= value;
    }
    std::array::from_fn(|i| centered(convolution[i], prime))
}

fn transform(domain: &TrinomialLimbDomain, input: &[i32; DEGREE]) -> TrinomialLimbSlots {
    let mut slots = domain.zero_slots();
    domain.forward_centered(input, &mut slots).unwrap();
    slots
}

fn inverse(domain: &TrinomialLimbDomain, slots: &TrinomialLimbSlots) -> [i32; DEGREE] {
    let mut coefficients = [0; DEGREE];
    domain.inverse_centered(slots, &mut coefficients).unwrap();
    let half = (domain.prime() / 2) as i32;
    assert!(coefficients
        .iter()
        .all(|&value| (-half..=half).contains(&value)));
    coefficients
}
