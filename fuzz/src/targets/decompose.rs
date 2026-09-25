//! Balanced power-of-two gadget decomposition.
//!
//! Oracle: the defining identity `x = Σ_l d_l · b^l` with every digit in the
//! balanced range `[-b/2, b/2)`, asserted only for coefficients the chosen
//! depth can represent (the full field when `levels · log_b` covers it). The
//! i8 entry point dispatches to AVX2/NEON (u32 fields), a u64 path (fp64),
//! or the generic path; all must agree with the generic i16 path.

use crate::gen::{self, modulus, Domain};
use crate::input::Reader;
use crate::stats;
use akita_algebra::ring::cyclotomic::{
    balanced_decompose_coefficients_pow2_i8_into, decompose_centering_threshold,
    BalancedDecomposePow2Params,
};
use akita_algebra::CyclotomicRing;
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::CommitmentConfig;
use jolt_field::{CanonicalEncoding, Field};

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    match reader.u8() % 3 {
        0 => case::<fp32::Field>(&mut reader, basis_range::<fp32::Dense>()),
        1 => case::<fp64::Field>(&mut reader, basis_range::<fp64::Dense>()),
        _ => case::<fp128::Field>(&mut reader, basis_range::<fp128::Dense>()),
    }
}

/// Union of the opening and source basis ranges the config lets the planner
/// select. Bases outside it (notably `log_basis = 1`) are not produced by any
/// shipped schedule; see `fuzz/FINDINGS.md`.
fn basis_range<Cfg: CommitmentConfig>() -> (u32, u32) {
    let (open_lo, open_hi) = Cfg::opening_basis_range();
    let (inner_lo, inner_hi) = Cfg::inner_basis_range();
    (open_lo.min(inner_lo), open_hi.max(inner_hi))
}

/// Largest magnitude exactly representable by `levels` balanced digits,
/// saturated at `u128::MAX`.
fn representable(levels: usize, log_basis: u32) -> (u128, u128) {
    // Digits lie in [-b/2, b/2 - 1]; the extremes are geometric sums.
    let b = 1u128 << log_basis;
    let mut sum = 0u128;
    let mut power = 1u128;
    for _ in 0..levels {
        sum = sum.saturating_add(power);
        power = power.saturating_mul(b);
    }
    let negative = (b / 2).saturating_mul(sum);
    let positive = (b / 2 - 1).saturating_mul(sum);
    (negative, positive)
}

fn case<F: Field + CanonicalEncoding>(reader: &mut Reader<'_>, (min_basis, max_basis): (u32, u32)) {
    let q = modulus::<F>();
    let field_bits = 128 - (q - 1).leading_zeros();
    let log_basis = min_basis + u32::from(reader.u8()) % (max_basis - min_basis + 1);
    let full_levels = field_bits.div_ceil(log_basis) as usize;
    let levels = match reader.u8() % 4 {
        0 => full_levels,
        1 => full_levels + 1,
        _ => 1 + usize::from(reader.u8()) % full_levels.max(1),
    };
    // `BalancedDecomposePow2Params` admits at most `128 + log_basis` bits.
    let levels = levels.min(((128 + log_basis) / log_basis) as usize);
    let (max_negative, max_positive) = representable(levels, log_basis);
    let bound = max_negative.min(max_positive);
    // At exactly the field width the centering threshold folds large values to
    // their negative representative, so every canonical value is covered.
    let exact_width = (levels as u32) * log_basis == field_bits;
    let domain = if exact_width || bound >= q / 2 {
        Domain::Full
    } else {
        Domain::Centered {
            negative: max_negative,
            positive: max_positive,
        }
    };

    // Multiples of 8 keep the SIMD bulk path reachable; the tail stays covered.
    let len = match reader.u8() % 4 {
        0 => 8 * (1 + usize::from(reader.u8() % 16)),
        _ => 1 + usize::from(reader.u8() % 64),
    };
    let coefficients: Vec<F> = gen::table(reader, len, domain);
    let params = BalancedDecomposePow2Params::new(levels, log_basis, q);
    let half_b = 1i128 << (log_basis - 1);
    let base = F::from_u128_reduced(1u128 << log_basis);

    let mut i16_planes = vec![0i16; len * levels];
    // The i16 path consumes whole rings; decompose each coefficient as the
    // constant term of a D=1 ring through the flat generic entry point below.
    for (index, &coefficient) in coefficients.iter().enumerate() {
        let ring = CyclotomicRing::<F, 1>::from_coefficients([coefficient]);
        let mut planes = vec![[0i16; 1]; levels];
        ring.balanced_decompose_pow2_i16_into(&mut planes, log_basis);
        for (level, plane) in planes.iter().enumerate() {
            i16_planes[level * len + index] = plane[0];
        }
    }
    for (index, &coefficient) in coefficients.iter().enumerate() {
        let mut recomposed = F::zero();
        let mut power = F::one();
        for level in 0..levels {
            let digit = i128::from(i16_planes[level * len + index]);
            assert!(
                (-half_b..half_b).contains(&digit),
                "digit {digit} outside balanced range for log_basis {log_basis}"
            );
            recomposed += F::from_i128(digit) * power;
            power *= base;
        }
        assert_eq!(
            recomposed, coefficient,
            "balanced decomposition does not recompose (levels {levels}, log_basis {log_basis}, threshold {})",
            decompose_centering_threshold(levels, log_basis, q)
        );
    }
    stats::count("decompose_i16");

    if log_basis <= 8 {
        let mut i8_planes = vec![0i8; len * levels];
        balanced_decompose_coefficients_pow2_i8_into(&coefficients, &mut i8_planes, &params);
        let widened: Vec<i16> = i8_planes.iter().map(|&digit| i16::from(digit)).collect();
        assert_eq!(
            widened, i16_planes,
            "i8 decomposition path disagrees with the generic i16 path"
        );
        stats::count("decompose_i8");
    }
}
