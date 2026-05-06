//! Search helpers for explaining bounded-`L1` preset choices.
//!
//! This module is test-only on purpose: it is a small reproducible script for
//! checking how much Fiat-Shamir support a `(D, M, B)` bounded-`L1` norm gives.
//! Run the report with:
//!
//! ```text
//! cargo test -p akita-challenges bounded_l1_d32_m8_support_report -- --ignored --nocapture
//! ```

use crate::sampler::bounded_l1::{COEFFS_BOUND_32, D_32, MAX_L1_NORM_32};

#[derive(Clone, Copy, Debug)]
struct CappedCount {
    value: u128,
    ge_2_128: bool,
}

impl CappedCount {
    const ONE: Self = Self {
        value: 1,
        ge_2_128: false,
    };

    fn add(self, rhs: Self) -> Self {
        if self.ge_2_128 || rhs.ge_2_128 {
            return Self {
                value: u128::MAX,
                ge_2_128: true,
            };
        }
        match self.value.checked_add(rhs.value) {
            Some(value) => Self {
                value,
                ge_2_128: false,
            },
            None => Self {
                value: u128::MAX,
                ge_2_128: true,
            },
        }
    }

    fn double(self) -> Self {
        if self.ge_2_128 {
            return self;
        }
        match self.value.checked_mul(2) {
            Some(value) => Self {
                value,
                ge_2_128: false,
            },
            None => Self {
                value: u128::MAX,
                ge_2_128: true,
            },
        }
    }
}

fn bounded_l1_support_count_capped(d: usize, coeff_bound: usize, l1_bound: usize) -> CappedCount {
    let mut row = vec![CappedCount::ONE; l1_bound + 1];
    for _ in 0..d {
        let mut next = vec![
            CappedCount {
                value: 0,
                ge_2_128: false,
            };
            l1_bound + 1
        ];
        for budget in 0..=l1_bound {
            let mut acc = row[budget];
            for mag in 1..=coeff_bound.min(budget) {
                acc = acc.add(row[budget - mag].double());
            }
            next[budget] = acc;
        }
        row = next;
    }
    row[l1_bound]
}

fn bounded_l1_support_bits(d: usize, coeff_bound: usize, l1_bound: usize) -> f64 {
    let mut row = vec![1.0f64; l1_bound + 1];
    for _ in 0..d {
        let mut next = vec![0.0f64; l1_bound + 1];
        for budget in 0..=l1_bound {
            let mut acc = row[budget];
            for mag in 1..=coeff_bound.min(budget) {
                acc += 2.0 * row[budget - mag];
            }
            next[budget] = acc;
        }
        row = next;
    }
    row[l1_bound].log2()
}

fn bounded_l1_expected_hamming_weight(d: usize, coeff_bound: usize, l1_bound: usize) -> f64 {
    let mut counts = vec![1.0f64; l1_bound + 1];
    let mut hamming_sums = vec![0.0f64; l1_bound + 1];
    for _ in 0..d {
        let mut next_counts = vec![0.0f64; l1_bound + 1];
        let mut next_hamming_sums = vec![0.0f64; l1_bound + 1];
        for budget in 0..=l1_bound {
            let mut count = counts[budget];
            let mut hamming_sum = hamming_sums[budget];
            for mag in 1..=coeff_bound.min(budget) {
                count += 2.0 * counts[budget - mag];
                hamming_sum += 2.0 * (hamming_sums[budget - mag] + counts[budget - mag]);
            }
            next_counts[budget] = count;
            next_hamming_sums[budget] = hamming_sum;
        }
        counts = next_counts;
        hamming_sums = next_hamming_sums;
    }
    hamming_sums[l1_bound] / counts[l1_bound]
}

fn minimum_l1_bound_for_128_bits(d: usize, coeff_bound: usize) -> Option<usize> {
    (0..=d.saturating_mul(coeff_bound))
        .find(|&l1_bound| bounded_l1_support_count_capped(d, coeff_bound, l1_bound).ge_2_128)
}

#[test]
fn bounded_l1_d32_m8_l1_bound_is_minimal_for_128_bits() {
    let min_l1_bound =
        minimum_l1_bound_for_128_bits(D_32, COEFFS_BOUND_32).expect("support reaches 2^128");
    assert_eq!(min_l1_bound, MAX_L1_NORM_32);
}

#[test]
#[ignore = "prints the bounded-L1 support search table"]
fn bounded_l1_d32_m8_support_report() {
    let min_l1_bound =
        minimum_l1_bound_for_128_bits(D_32, COEFFS_BOUND_32).expect("support reaches 2^128");
    println!("bounded-L1 support search");
    println!("D = {D_32}, M = {COEFFS_BOUND_32}, target support = 2^128");
    println!("minimum B with at least 128 bits of support: {min_l1_bound}");
    println!(
        "expected hamming weight over the full bounded-L1 space at B = {min_l1_bound}: {:.3}",
        bounded_l1_expected_hamming_weight(D_32, COEFFS_BOUND_32, min_l1_bound),
    );
    println!();
    println!("neighboring bounds:");
    let start = min_l1_bound.saturating_sub(4);
    let end = (min_l1_bound + 4).min(D_32 * COEFFS_BOUND_32);
    for l1_bound in start..=end {
        let count = bounded_l1_support_count_capped(D_32, COEFFS_BOUND_32, l1_bound);
        let status = if count.ge_2_128 { "enough" } else { "short" };
        println!(
            "  B = {l1_bound:3}: log2(|space|) ~= {:9.6}  avg_hw ~= {:6.3}  {status}",
            bounded_l1_support_bits(D_32, COEFFS_BOUND_32, l1_bound),
            bounded_l1_expected_hamming_weight(D_32, COEFFS_BOUND_32, l1_bound),
        );
    }
}
