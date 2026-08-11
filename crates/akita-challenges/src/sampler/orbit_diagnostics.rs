//! Ignored empirical checks for fixed-point operator-norm orbit invariance.
//!
//! Run with, for example:
//!
//! ```text
//! AKITA_OP_NORM_PROPOSALS=1000000 \
//! AKITA_OP_NORM_COV_ACCEPTS=250000 \
//! AKITA_OP_NORM_RANDOM_ORBITS=1000000 \
//! AKITA_OP_NORM_FULL_ORBITS=8 \
//! cargo test -p akita-challenges --release \
//!   measure_operator_norm_orbit_invariance -- --ignored --nocapture
//! ```

use super::{op_norm::OpNormTable, SignedSparseScratch, XofCursor};
use crate::{
    SparseChallenge, SparseChallengeConfig, D128_SELECTIVE_L2_CHALLENGE_CONFIG,
    D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};
use std::{env, time::Instant};

const PREDICATE_SCALE: u32 = 48;

#[derive(Clone, Copy, Debug, Default)]
struct DecisionCounts {
    accept: u64,
    reject: u64,
    indeterminate: u64,
}

impl DecisionCounts {
    fn observe(&mut self, decision: super::op_norm::Decision) {
        match decision {
            super::op_norm::Decision::Accept => self.accept += 1,
            super::op_norm::Decision::Reject => self.reject += 1,
            super::op_norm::Decision::Indeterminate => self.indeterminate += 1,
        }
    }
}

#[derive(Debug)]
struct CovarianceStats {
    samples: usize,
    max_abs_mean: f64,
    diagonal_relative_spread: f64,
    max_off_diagonal_over_expected_diagonal: f64,
    off_diagonal_frobenius_over_trace: f64,
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn transform_challenge(
    challenge: &SparseChallenge,
    ring_dimension: usize,
    odd_automorphism: usize,
    shift: usize,
) -> SparseChallenge {
    debug_assert!(odd_automorphism < 2 * ring_dimension && odd_automorphism % 2 == 1);
    debug_assert!(shift < ring_dimension);
    let two_d = 2 * ring_dimension;
    let mut positions = Vec::with_capacity(challenge.positions.len());
    let mut coeffs = Vec::with_capacity(challenge.coeffs.len());
    for (&position, &coefficient) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        let exponent = (odd_automorphism * position as usize + shift) % two_d;
        if exponent < ring_dimension {
            positions.push(exponent as u32);
            coeffs.push(coefficient);
        } else {
            positions.push((exponent - ring_dimension) as u32);
            coeffs.push(-coefficient);
        }
    }
    SparseChallenge {
        positions: positions.into(),
        coeffs: coeffs.into(),
    }
}

fn accumulate_covariance(
    challenge: &SparseChallenge,
    ring_dimension: usize,
    sums: &mut [i64],
    products: &mut [i64],
) {
    for (&position, &coefficient) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
        sums[position as usize] += i64::from(coefficient);
    }
    for (&left_position, &left_coefficient) in
        challenge.positions.iter().zip(challenge.coeffs.iter())
    {
        for (&right_position, &right_coefficient) in
            challenge.positions.iter().zip(challenge.coeffs.iter())
        {
            products[left_position as usize * ring_dimension + right_position as usize] +=
                i64::from(left_coefficient) * i64::from(right_coefficient);
        }
    }
}

fn covariance_stats(
    samples: usize,
    challenge_l2_sq: u128,
    sums: &[i64],
    products: &[i64],
    ring_dimension: usize,
) -> CovarianceStats {
    let samples_f64 = samples as f64;
    let expected_diagonal = challenge_l2_sq as f64 / ring_dimension as f64;
    let means = sums
        .iter()
        .map(|&sum| sum as f64 / samples_f64)
        .collect::<Vec<_>>();
    let mut diagonal_min = f64::INFINITY;
    let mut diagonal_max = f64::NEG_INFINITY;
    let mut max_off_diagonal = 0.0f64;
    let mut off_diagonal_frobenius_sq = 0.0f64;
    for row in 0..ring_dimension {
        for column in 0..ring_dimension {
            let covariance = products[row * ring_dimension + column] as f64 / samples_f64
                - means[row] * means[column];
            if row == column {
                diagonal_min = diagonal_min.min(covariance);
                diagonal_max = diagonal_max.max(covariance);
            } else {
                max_off_diagonal = max_off_diagonal.max(covariance.abs());
                off_diagonal_frobenius_sq += covariance * covariance;
            }
        }
    }
    CovarianceStats {
        samples,
        max_abs_mean: means.iter().copied().map(f64::abs).fold(0.0, f64::max),
        diagonal_relative_spread: (diagonal_max - diagonal_min) / expected_diagonal,
        max_off_diagonal_over_expected_diagonal: max_off_diagonal / expected_diagonal,
        off_diagonal_frobenius_over_trace: off_diagonal_frobenius_sq.sqrt()
            / (ring_dimension as f64 * expected_diagonal),
    }
}

fn measure_family(
    label: &str,
    ring_dimension: usize,
    config: SparseChallengeConfig,
    threshold: u64,
) {
    let proposals = env_usize("AKITA_OP_NORM_PROPOSALS", 100_000);
    let covariance_accepts = env_usize("AKITA_OP_NORM_COV_ACCEPTS", 50_000);
    let random_orbits = env_usize("AKITA_OP_NORM_RANDOM_ORBITS", 100_000);
    let full_orbits = env_usize("AKITA_OP_NORM_FULL_ORBITS", 2);
    let table = OpNormTable::new(
        ring_dimension,
        PREDICATE_SCALE,
        config.l1_norm() as u64,
        threshold,
    )
    .expect("valid production operator-norm table");
    let mut proposal_cursor = XofCursor::from_seed(format!("{label}/proposals").as_bytes());
    let mut orbit_cursor = XofCursor::from_seed(format!("{label}/orbits").as_bytes());
    let mut scratch = SignedSparseScratch::new(config.count_pm1, config.count_pm2);
    let mut decisions = DecisionCounts::default();
    let start = Instant::now();
    for _ in 0..proposals {
        scratch
            .sample(
                &mut proposal_cursor,
                ring_dimension,
                config.count_pm1,
                config.count_pm2,
            )
            .expect("valid production proposal");
        decisions.observe(
            table
                .decide_parts(scratch.positions(), scratch.coeffs(), threshold)
                .expect("valid predicate input"),
        );
    }
    let predicate_elapsed = start.elapsed();

    let mut random_mismatches = 0usize;
    let mut baseline_counts = DecisionCounts::default();
    for _ in 0..random_orbits {
        scratch
            .sample(
                &mut proposal_cursor,
                ring_dimension,
                config.count_pm1,
                config.count_pm2,
            )
            .expect("valid production proposal");
        let challenge = scratch.take_challenge();
        let baseline = table
            .decide_parts(&challenge.positions, &challenge.coeffs, threshold)
            .expect("valid predicate input");
        baseline_counts.observe(baseline);
        let odd = 2 * orbit_cursor.next_usize_mod(ring_dimension) + 1;
        let shift = orbit_cursor.next_usize_mod(ring_dimension);
        let transformed = transform_challenge(&challenge, ring_dimension, odd, shift);
        let transformed_decision = table
            .decide_parts(&transformed.positions, &transformed.coeffs, threshold)
            .expect("valid transformed predicate input");
        random_mismatches += usize::from(transformed_decision != baseline);
    }

    let mut fully_checked = 0usize;
    let mut mixed_full_orbits = 0usize;
    while fully_checked < full_orbits {
        scratch
            .sample(
                &mut proposal_cursor,
                ring_dimension,
                config.count_pm1,
                config.count_pm2,
            )
            .expect("valid production proposal");
        if table
            .decide_parts(scratch.positions(), scratch.coeffs(), threshold)
            .expect("valid predicate input")
            != super::op_norm::Decision::Accept
        {
            continue;
        }
        let challenge = scratch.take_challenge();
        let mut orbit_is_mixed = false;
        'automorphisms: for automorphism_index in 0..ring_dimension {
            let odd = 2 * automorphism_index + 1;
            for shift in 0..ring_dimension {
                let transformed = transform_challenge(&challenge, ring_dimension, odd, shift);
                if table
                    .decide_parts(&transformed.positions, &transformed.coeffs, threshold)
                    .expect("valid transformed predicate input")
                    != super::op_norm::Decision::Accept
                {
                    orbit_is_mixed = true;
                    break 'automorphisms;
                }
            }
        }
        mixed_full_orbits += usize::from(orbit_is_mixed);
        fully_checked += 1;
    }

    let matrix_len = ring_dimension * ring_dimension;
    let mut raw_sums = vec![0i64; ring_dimension];
    let mut raw_products = vec![0i64; matrix_len];
    let mut symmetrized_sums = vec![0i64; ring_dimension];
    let mut symmetrized_products = vec![0i64; matrix_len];
    let covariance_start = Instant::now();
    let mut accepted = 0usize;
    while accepted < covariance_accepts {
        scratch
            .sample(
                &mut proposal_cursor,
                ring_dimension,
                config.count_pm1,
                config.count_pm2,
            )
            .expect("valid production proposal");
        if !table
            .accept_strict_parts(scratch.positions(), scratch.coeffs(), threshold)
            .expect("valid predicate input")
        {
            continue;
        }
        let challenge = scratch.take_challenge();
        accumulate_covariance(&challenge, ring_dimension, &mut raw_sums, &mut raw_products);
        let odd = 2 * orbit_cursor.next_usize_mod(ring_dimension) + 1;
        let shift = orbit_cursor.next_usize_mod(ring_dimension);
        let transformed = transform_challenge(&challenge, ring_dimension, odd, shift);
        accumulate_covariance(
            &transformed,
            ring_dimension,
            &mut symmetrized_sums,
            &mut symmetrized_products,
        );
        accepted += 1;
    }
    let covariance_elapsed = covariance_start.elapsed();
    let raw_covariance = covariance_stats(
        accepted,
        config.challenge_l2_sq_max(),
        &raw_sums,
        &raw_products,
        ring_dimension,
    );
    let symmetrized_covariance = covariance_stats(
        accepted,
        config.challenge_l2_sq_max(),
        &symmetrized_sums,
        &symmetrized_products,
        ring_dimension,
    );
    eprintln!(
        "operator_norm_orbit label={label} d={ring_dimension} proposals={proposals} decisions={decisions:?} predicate_elapsed={predicate_elapsed:?} random_orbits={random_orbits} random_baseline={baseline_counts:?} random_mismatches={random_mismatches} full_orbits={fully_checked} mixed_full_orbits={mixed_full_orbits} covariance_elapsed={covariance_elapsed:?} raw_covariance={raw_covariance:?} symmetrized_covariance={symmetrized_covariance:?}"
    );
    assert_eq!(raw_covariance.samples, covariance_accepts);
    assert_eq!(symmetrized_covariance.samples, covariance_accepts);
    assert!(raw_covariance.max_abs_mean.is_finite());
    assert!(raw_covariance.diagonal_relative_spread.is_finite());
    assert!(raw_covariance
        .max_off_diagonal_over_expected_diagonal
        .is_finite());
    assert!(raw_covariance.off_diagonal_frobenius_over_trace.is_finite());
}

#[test]
#[ignore = "empirical operator-norm orbit diagnostic"]
fn measure_operator_norm_orbit_invariance() {
    measure_family("d64_a31_b11_t19", 64, D64_SELECTIVE_L2_CHALLENGE_CONFIG, 19);
    measure_family(
        "d128_a31_b0_t14",
        128,
        D128_SELECTIVE_L2_CHALLENGE_CONFIG,
        14,
    );
}
