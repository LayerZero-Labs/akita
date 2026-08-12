//! Planner-only moment model for recursively folded response witnesses.
//!
//! The verifier never evaluates this module. The planner freezes its selected
//! integer response cap into the generated schedule, and the verifier enforces
//! that cap exactly.
//!
//! The model follows the witness construction rather than fitting one scale
//! factor to its final length. For a source vector `s` and a random negacyclic
//! challenge `c`, scalar challenge covariance gives
//! `E[||c * s||_2^2 | s] = E[||c||_2^2] ||s||_2^2`. The accepted challenge
//! sampler is not assumed to have perfect scalar covariance. Its measured
//! covariance defect and every approximation below are covered by a separate
//! source-model envelope. The response multiplier then has a distribution-free
//! Markov interpretation once that envelope bounds the conditional mean.

use akita_field::AkitaError;
use akita_types::sis::{compute_num_digits_field_width, HonestFoldPolicySpec};
use akita_types::{CommittedGroupParams, OpeningClaimsLayout, WitnessLayout};

/// Relative envelope for any underestimate by the typed moment model.
///
/// Current aggregate source measurements across fp32, fp64, and fp128 are 0.09
/// to 1.93 percent above the measured source energy. Separate typed-component
/// validation found up to 2.24 percent error in the unfavorable direction.
/// Three percent keeps model error separate from the response allowance. It
/// covers the rounded Gaussian, pseudo-Mersenne, challenge-covariance, and
/// finite-mixing approximations. Conservative overestimates do not consume
/// this envelope.
const SOURCE_MODEL_ENVELOPE_PPM: u128 = 1_030_000;

/// Per-attempt response cap relative to the conditional response-energy mean.
///
/// Markov's inequality gives `Pr[X <= 1.06 E[X]] >= 3/53` for every
/// nonnegative response energy `X`. Thus this is a distribution-free
/// completeness guarantee for grinding, not a Gaussian-tail assumption. With
/// 4096 independent transcript attempts, even this worst-case bound makes
/// exhaustion negligible.
const RESPONSE_MEAN_MULTIPLIER_PPM: u128 = 1_060_000;
const PPM: u128 = 1_000_000;
const MOMENT_PPM: u128 = 1_000_000;

/// Planner estimate of the squared Euclidean norm of one recursive witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SourceMomentEstimate {
    mean_l2_sq: u128,
    peak_second_moment_ppm: u128,
}

impl SourceMomentEstimate {
    /// Retain seven leading bits and round the remaining energy upward.
    ///
    /// This gives the suffix DP a bounded, reusable state domain while adding
    /// less than 1/64 relative error. The cap is conservative because the
    /// rounding direction is always upward.
    #[cfg(test)]
    pub(crate) const fn new(mean_l2_sq: u128) -> Option<Self> {
        Self::from_moments(mean_l2_sq, mean_l2_sq.saturating_mul(MOMENT_PPM))
    }

    /// Build a bounded DP state from total and largest-coordinate moments.
    pub(crate) const fn from_moments(
        mean_l2_sq: u128,
        peak_second_moment_ppm: u128,
    ) -> Option<Self> {
        if mean_l2_sq == 0 || peak_second_moment_ppm == 0 {
            None
        } else {
            let significant_bits = 7u32;
            let bit_len = u128::BITS - mean_l2_sq.leading_zeros();
            let discard = bit_len.saturating_sub(significant_bits);
            let quantum = 1u128 << discard;
            let rounded = match mean_l2_sq.checked_add(quantum - 1) {
                Some(value) => value & !(quantum - 1),
                None => u128::MAX,
            };
            let peak_bit_len = u128::BITS - peak_second_moment_ppm.leading_zeros();
            let peak_discard = peak_bit_len.saturating_sub(significant_bits);
            let peak_quantum = 1u128 << peak_discard;
            let rounded_peak = match peak_second_moment_ppm.checked_add(peak_quantum - 1) {
                Some(value) => value & !(peak_quantum - 1),
                None => u128::MAX,
            };
            Some(Self {
                mean_l2_sq: rounded,
                peak_second_moment_ppm: rounded_peak,
            })
        }
    }

    pub(crate) const fn mean_l2_sq(self) -> u128 {
        self.mean_l2_sq
    }

    pub(crate) const fn peak_second_moment_ppm(self) -> u128 {
        self.peak_second_moment_ppm
    }

    /// Freeze the planner's response-energy cap for one challenge family.
    pub(crate) fn response_l2_sq_cap(self, challenge_l2_sq: u128) -> Option<u128> {
        let numerator = self
            .mean_l2_sq
            .checked_mul(challenge_l2_sq)?
            .checked_mul(SOURCE_MODEL_ENVELOPE_PPM)?
            .checked_mul(RESPONSE_MEAN_MULTIPLIER_PPM)?;
        let scale = PPM.checked_mul(PPM)?;
        numerator
            .checked_add(scale - 1)
            .map(|rounded| rounded / scale)
    }

    /// Model a whole-response maximum at per-attempt acceptance probability 1/8.
    ///
    /// For a sub-Gaussian coordinate with variance proxy `v`, the selected
    /// threshold makes `2 N exp(-t^2/(2v)) <= 7/8`. The maximum of the average
    /// coordinate variance and the typed component peak prevents a large Z,
    /// E, T, R, or compression class from being hidden by aggregate energy.
    pub(crate) fn response_linf_cap(
        self,
        challenge_l2_sq: u128,
        num_live_blocks: usize,
        num_chunks: usize,
        num_fold_coeffs: usize,
    ) -> Option<u128> {
        if challenge_l2_sq == 0 || num_live_blocks == 0 || num_chunks == 0 || num_fold_coeffs == 0 {
            return None;
        }
        let average_variance =
            self.mean_l2_sq.checked_mul(challenge_l2_sq)? as f64 / num_fold_coeffs as f64;
        let blocks_per_chunk = num_live_blocks.div_ceil(num_chunks) as u128;
        let peak_variance = self
            .peak_second_moment_ppm
            .checked_mul(challenge_l2_sq)?
            .checked_mul(blocks_per_chunk)? as f64
            / MOMENT_PPM as f64;
        let variance =
            average_variance.max(peak_variance) * SOURCE_MODEL_ENVELOPE_PPM as f64 / PPM as f64;
        let union_log = (16.0 * num_fold_coeffs as f64 / 7.0).ln();
        let threshold = (2.0 * variance * union_log).sqrt();
        if !threshold.is_finite() || threshold <= 0.0 || threshold > u128::MAX as f64 {
            return None;
        }
        Some((threshold.ceil() as u128).max(1))
    }
}

fn checked_ceil_f64(value: f64, context: &str) -> Result<u128, AkitaError> {
    if !value.is_finite() || value < 0.0 || value > u128::MAX as f64 {
        return Err(AkitaError::InvalidSetup(format!(
            "{context} is outside the planner's numeric range"
        )));
    }
    Ok(value.ceil() as u128)
}

/// Exact second moment of a uniform centered digit in
/// `[-basis/2, basis/2)` for a power-of-two basis.
fn centered_uniform_digit_second_moment(basis: u128) -> Option<f64> {
    if basis < 2 || !basis.is_power_of_two() {
        return None;
    }
    Some((basis.checked_mul(basis)?.checked_add(2)? as f64) / 12.0)
}

fn moment_to_ppm(moment: f64, context: &str) -> Result<u128, AkitaError> {
    checked_ceil_f64(moment * MOMENT_PPM as f64, context)
}

fn field_digit_moments(
    scalar_count: usize,
    field_bits: u32,
    log_basis: u32,
    digit_count: usize,
) -> Result<(u128, u128), AkitaError> {
    if scalar_count == 0 || field_bits == 0 || log_basis == 0 || digit_count == 0 {
        return Err(AkitaError::InvalidSetup(
            "field digit moment requires positive geometry".into(),
        ));
    }
    let mut per_scalar = 0.0;
    let mut peak = 0.0f64;
    for plane in 0..digit_count {
        let consumed = (plane as u32)
            .checked_mul(log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("digit-plane width overflow".into()))?;
        if consumed >= field_bits {
            break;
        }
        let plane_bits = log_basis.min(field_bits - consumed);
        let basis = 1u128
            .checked_shl(plane_bits)
            .ok_or_else(|| AkitaError::InvalidSetup("digit-plane basis overflow".into()))?;
        let moment = centered_uniform_digit_second_moment(basis)
            .ok_or_else(|| AkitaError::InvalidSetup("digit-plane basis is not supported".into()))?;
        per_scalar += moment;
        peak = peak.max(moment);
    }
    Ok((
        checked_ceil_f64(
            per_scalar * scalar_count as f64,
            "finite-field digit energy",
        )?,
        moment_to_ppm(peak, "finite-field digit peak moment")?,
    ))
}

/// Modeled energy of a full-width finite-field balanced decomposition.
///
/// The model uses the uniform centered moment for each complete plane. This is
/// exact for a uniform power-of-two residue. The last plane uses the residual
/// field width rather than pretending that it is another full plane. The
/// supported pseudo-Mersenne moduli differ from `2^field_bits` by a negligible
/// fraction. Recursive E, T, and R values can also retain correlation instead
/// of being fully mixed. The explicit model envelope covers unfavorable error;
/// retained correlation usually makes this estimate conservative.
#[cfg(test)]
pub(crate) fn field_digit_energy(
    scalar_count: usize,
    field_bits: u32,
    log_basis: u32,
    digit_count: usize,
) -> Result<u128, AkitaError> {
    field_digit_moments(scalar_count, field_bits, log_basis, digit_count).map(|moments| moments.0)
}

/// Exact uniform-field source moments used by public setup prefixes.
pub(crate) fn uniform_field_source_moment(
    scalar_count: usize,
    field_bits: u32,
    log_basis: u32,
    digit_count: usize,
) -> Result<SourceMomentEstimate, AkitaError> {
    let (energy, peak) = field_digit_moments(scalar_count, field_bits, log_basis, digit_count)?;
    SourceMomentEstimate::from_moments(energy, peak)
        .ok_or_else(|| AkitaError::InvalidSetup("uniform field source is empty".into()))
}

fn bounded_field_source_moment(
    scalar_count: usize,
    field_bits: u32,
    log_basis: u32,
    digit_count: usize,
) -> Result<SourceMomentEstimate, AkitaError> {
    if scalar_count == 0 || field_bits == 0 || log_basis == 0 || digit_count == 0 {
        return Err(AkitaError::InvalidSetup(
            "bounded field source requires positive geometry".into(),
        ));
    }
    let mut per_scalar = 0u128;
    let mut peak = 0u128;
    for plane in 0..digit_count {
        let consumed = (plane as u32)
            .checked_mul(log_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("digit-plane width overflow".into()))?;
        if consumed >= field_bits {
            break;
        }
        let plane_bits = log_basis.min(field_bits - consumed);
        let half_basis = 1u128
            .checked_shl(plane_bits - 1)
            .ok_or_else(|| AkitaError::InvalidSetup("digit-plane bound overflow".into()))?;
        let square = half_basis
            .checked_mul(half_basis)
            .ok_or_else(|| AkitaError::InvalidSetup("digit-plane energy overflow".into()))?;
        per_scalar = per_scalar
            .checked_add(square)
            .ok_or_else(|| AkitaError::InvalidSetup("bounded source energy overflow".into()))?;
        peak = peak.max(square);
    }
    let energy = per_scalar
        .checked_mul(scalar_count as u128)
        .ok_or_else(|| AkitaError::InvalidSetup("bounded source energy overflow".into()))?;
    let peak_ppm = peak
        .checked_mul(MOMENT_PPM)
        .ok_or_else(|| AkitaError::InvalidSetup("bounded source peak overflow".into()))?;
    SourceMomentEstimate::from_moments(energy, peak_ppm)
        .ok_or_else(|| AkitaError::InvalidSetup("bounded field source is empty".into()))
}

fn centered_residue(value: i64, basis: i64) -> i64 {
    let residue = value.rem_euclid(basis);
    if residue >= basis / 2 {
        residue - basis
    } else {
        residue
    }
}

fn normal_cdf(value: f64) -> f64 {
    0.5 * (1.0 + libm::erf(value / core::f64::consts::SQRT_2))
}

/// Expected squared centered residue of a rounded normal integer.
fn rounded_normal_digit_second_moment(sigma: f64, basis: i64) -> f64 {
    if sigma <= f64::EPSILON {
        return 0.0;
    }

    // Once the standard deviation spans one residue period, the first
    // nonconstant Fourier coefficient of the wrapped normal is at most
    // exp(-2*pi^2), below 3e-9. Rounding contributes only still-smaller alias
    // terms for the supported bases. This is negligible beside the 3% source
    // envelope and bounds planner work at early, high-variance levels.
    if sigma >= basis as f64 {
        return centered_uniform_digit_second_moment(basis as u128).unwrap_or(0.0);
    }

    let radius = (8.0 * sigma + 0.5).ceil() as i64;
    let mut moment = 0.0;
    let mut lower_cdf = normal_cdf((-radius as f64 - 0.5) / sigma);
    for value in -radius..=radius {
        let upper = (value as f64 + 0.5) / sigma;
        let upper_cdf = normal_cdf(upper);
        let probability = upper_cdf - lower_cdf;
        let digit = centered_residue(value, basis) as f64;
        moment += probability * digit * digit;
        lower_cdf = upper_cdf;
    }
    moment
}

/// Expected energy after balanced-decomposing an approximately Gaussian
/// folded response.
///
/// `response_l2_sq` is the total response-energy mean before decomposition;
/// `response_coeff_count` is its physical scalar coefficient count.
pub(crate) fn gaussian_response_digit_energy(
    response_l2_sq: u128,
    response_coeff_count: usize,
    log_basis: u32,
    digit_count: usize,
) -> Result<u128, AkitaError> {
    if response_l2_sq == 0 || response_coeff_count == 0 || log_basis == 0 || digit_count == 0 {
        return Err(AkitaError::InvalidSetup(
            "Gaussian digit moment requires positive geometry".into(),
        ));
    }
    let basis = 1i64
        .checked_shl(log_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("Gaussian digit basis overflow".into()))?;
    let sigma = ((response_l2_sq as f64) / response_coeff_count as f64).sqrt();
    let mut per_response_coefficient = 0.0;
    let mut plane_sigma = sigma;
    for _ in 0..digit_count {
        per_response_coefficient += rounded_normal_digit_second_moment(plane_sigma, basis);
        plane_sigma /= basis as f64;
    }
    checked_ceil_f64(
        per_response_coefficient * response_coeff_count as f64,
        "Gaussian response digit energy",
    )
}

fn gaussian_response_digit_moments(
    response_l2_sq: u128,
    response_coeff_count: usize,
    peak_response_second_moment_ppm: u128,
    log_basis: u32,
    digit_count: usize,
) -> Result<(u128, u128), AkitaError> {
    let energy = gaussian_response_digit_energy(
        response_l2_sq,
        response_coeff_count,
        log_basis,
        digit_count,
    )?;
    let basis = 1i64
        .checked_shl(log_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("Gaussian digit basis overflow".into()))?;
    let mut sigma = ((peak_response_second_moment_ppm as f64) / MOMENT_PPM as f64).sqrt();
    let mut peak = 0.0f64;
    for _ in 0..digit_count {
        peak = peak.max(rounded_normal_digit_second_moment(sigma, basis));
        sigma /= basis as f64;
    }
    Ok((energy, moment_to_ppm(peak, "Gaussian digit peak moment")?))
}

/// Expected energy of negative-binary compression digits.
pub(crate) fn compression_digit_energy(coefficient_count: usize) -> u128 {
    coefficient_count.div_ceil(2) as u128
}

/// Apply the exact energy multiplicity of extension-field tensor packing under
/// exchangeable extension coordinates.
///
/// Coordinate zero appears once. Each of the other `K-1` coordinates appears
/// twice with opposite signs, so the expected multiplier is `(2K-1)/K`.
pub(crate) fn tensor_packed_energy(logical_energy: u128, extension_degree: usize) -> Option<u128> {
    if logical_energy == 0 || extension_degree == 0 {
        return None;
    }
    let numerator =
        logical_energy.checked_mul((extension_degree as u128).checked_mul(2)?.checked_sub(1)?)?;
    numerator
        .checked_add(extension_degree as u128 - 1)
        .map(|rounded| rounded / extension_degree as u128)
}

fn checked_logical_group_len(num_vars: usize, num_polynomials: usize) -> Result<usize, AkitaError> {
    1usize
        .checked_shl(num_vars as u32)
        .and_then(|len| len.checked_mul(num_polynomials))
        .ok_or_else(|| AkitaError::InvalidSetup("root source length overflow".into()))
}

/// Source moments of each root opening group before its first fold.
pub(crate) fn root_group_source_moments(
    params: &CommittedGroupParams,
    opening_layout: &OpeningClaimsLayout,
    final_policy: HonestFoldPolicySpec,
    precommitted_policies: &[HonestFoldPolicySpec],
    field_bits: u32,
) -> Result<Vec<SourceMomentEstimate>, AkitaError> {
    let final_group_index = opening_layout.root_final_group_index()?;
    if precommitted_policies.len() != final_group_index {
        return Err(AkitaError::InvalidSetup(
            "root response model requires one policy per precommitted group".into(),
        ));
    }
    let mut moments = Vec::with_capacity(opening_layout.num_groups());
    for group_index in 0..opening_layout.num_groups() {
        let group_layout = *opening_layout.group_layout(group_index)?;
        let group_params = params.group_params(opening_layout, group_index)?;
        let logical_len =
            checked_logical_group_len(group_layout.num_vars(), group_layout.num_polynomials())?;
        let policy = if group_index == final_group_index {
            final_policy
        } else {
            *precommitted_policies.get(group_index).ok_or_else(|| {
                AkitaError::InvalidSetup("precommitted response policy is missing".into())
            })?
        };
        let moment = match policy {
            HonestFoldPolicySpec::UnitOneHot(onehot) => {
                let chunk = onehot.source_chunk_size();
                if chunk == 0 || !logical_len.is_multiple_of(chunk) {
                    return Err(AkitaError::InvalidSetup(
                        "unit one-hot root length must be a multiple of its source chunk size"
                            .into(),
                    ));
                }
                SourceMomentEstimate::from_moments((logical_len / chunk) as u128, MOMENT_PPM)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("unit one-hot root source is empty".into())
                    })?
            }
            HonestFoldPolicySpec::BalancedSignedDigit(_) => bounded_field_source_moment(
                logical_len,
                field_bits,
                group_params.log_basis_inner(),
                group_params.num_digits_inner(),
            )?,
        };
        moments.push(moment);
    }
    Ok(moments)
}

fn checked_add_energy(total: &mut u128, value: u128) -> Result<(), AkitaError> {
    *total = total
        .checked_add(value)
        .ok_or_else(|| AkitaError::InvalidSetup("response-model energy overflow".into()))?;
    Ok(())
}

/// Predict the recursive witness produced by one ring-switch level from its
/// exact typed layout.
pub(crate) fn next_source_moment(
    params: &CommittedGroupParams,
    opening_layout: &OpeningClaimsLayout,
    source_groups: &[SourceMomentEstimate],
    field_bits: u32,
    extension_degree: usize,
) -> Result<SourceMomentEstimate, AkitaError> {
    if source_groups.len() != opening_layout.num_groups() {
        return Err(AkitaError::InvalidSetup(
            "response source moments disagree with the opening groups".into(),
        ));
    }
    let quotient_depth = compute_num_digits_field_width(field_bits, params.log_basis_open);
    let layout = WitnessLayout::new(
        params,
        opening_layout,
        params.witness_chunk.num_chunks,
        quotient_depth,
    )?;
    let mut logical_energy = 0u128;
    let mut logical_peak_ppm = 0u128;

    for unit in layout.units() {
        let group_index = unit.group_index();
        let group_params = params.group_params(opening_layout, group_index)?;
        let group_source = source_groups
            .get(group_index)
            .copied()
            .ok_or_else(|| AkitaError::InvalidSetup("response source group is missing".into()))?;
        let total_blocks = group_params.num_live_blocks();
        if total_blocks == 0 || group_params.num_digits_fold() == 0 {
            return Err(AkitaError::InvalidSetup(
                "response-model group geometry is empty".into(),
            ));
        }
        let chunk_source = group_source
            .mean_l2_sq()
            .checked_mul(unit.num_live_blocks() as u128)
            .and_then(|value| value.checked_add(total_blocks as u128 - 1))
            .map(|rounded| rounded / total_blocks as u128)
            .ok_or_else(|| AkitaError::InvalidSetup("chunk source energy overflow".into()))?;
        let response_energy = chunk_source
            .checked_mul(group_params.fold_challenge_config().challenge_l2_sq_max())
            .ok_or_else(|| AkitaError::InvalidSetup("fold response energy overflow".into()))?;
        let response_coeff_count = unit.z_range().len() / group_params.num_digits_fold();
        if response_energy != 0 && response_coeff_count != 0 {
            let peak_response_second_moment_ppm = group_source
                .peak_second_moment_ppm()
                .checked_mul(group_params.fold_challenge_config().challenge_l2_sq_max())
                .and_then(|value| value.checked_mul(unit.num_live_blocks() as u128))
                .ok_or_else(|| AkitaError::InvalidSetup("fold response peak overflow".into()))?;
            let (energy, peak) = gaussian_response_digit_moments(
                response_energy,
                response_coeff_count,
                peak_response_second_moment_ppm,
                group_params.log_basis_open(),
                group_params.num_digits_fold(),
            )?;
            checked_add_energy(&mut logical_energy, energy)?;
            logical_peak_ppm = logical_peak_ppm.max(peak);
        }

        let num_claims = opening_layout.group_layout(group_index)?.num_polynomials();
        let group_d_a = params.group_role_dims(opening_layout, group_index)?.d_a();
        let e_scalar_count = num_claims
            .checked_mul(unit.num_live_blocks())
            .and_then(|count| count.checked_mul(group_d_a))
            .ok_or_else(|| AkitaError::InvalidSetup("live E source length overflow".into()))?;
        let allocated_e_scalar_count = unit.e_range().len() / group_params.num_digits_open();
        if e_scalar_count > allocated_e_scalar_count {
            return Err(AkitaError::InvalidSetup(
                "live E source exceeds its witness span".into(),
            ));
        }
        if e_scalar_count != 0 {
            let (energy, peak) = field_digit_moments(
                e_scalar_count,
                field_bits,
                group_params.log_basis_open(),
                group_params.num_digits_open(),
            )?;
            checked_add_energy(&mut logical_energy, energy)?;
            logical_peak_ppm = logical_peak_ppm.max(peak);
        }
        let t_scalar_count = e_scalar_count
            .checked_mul(group_params.a_rows_len())
            .ok_or_else(|| AkitaError::InvalidSetup("live T source length overflow".into()))?;
        let allocated_t_scalar_count = unit.t_range().len() / group_params.num_digits_outer();
        if t_scalar_count > allocated_t_scalar_count {
            return Err(AkitaError::InvalidSetup(
                "live T source exceeds its witness span".into(),
            ));
        }
        if t_scalar_count != 0 {
            let (energy, peak) = field_digit_moments(
                t_scalar_count,
                field_bits,
                group_params.log_basis_outer(),
                group_params.num_digits_outer(),
            )?;
            checked_add_energy(&mut logical_energy, energy)?;
            logical_peak_ppm = logical_peak_ppm.max(peak);
        }
    }

    for row in layout.r_rows() {
        let scalar_count = row.range().len() / quotient_depth;
        if scalar_count != 0 {
            let (energy, peak) = field_digit_moments(
                scalar_count,
                field_bits,
                params.log_basis_open,
                quotient_depth,
            )?;
            checked_add_energy(&mut logical_energy, energy)?;
            logical_peak_ppm = logical_peak_ppm.max(peak);
        }
    }

    let compression_coefficients = layout
        .compression_layers()
        .iter()
        .try_fold(0usize, |total, layer| {
            let f = layer
                .f_spans()
                .iter()
                .try_fold(0usize, |sum, (_, span)| sum.checked_add(span.range().len()))?;
            total
                .checked_add(f)?
                .checked_add(layer.h_span().range().len())
        })
        .ok_or_else(|| AkitaError::InvalidSetup("compression model length overflow".into()))?;
    checked_add_energy(
        &mut logical_energy,
        compression_digit_energy(compression_coefficients),
    )?;
    if compression_coefficients != 0 {
        logical_peak_ppm = logical_peak_ppm.max(MOMENT_PPM.div_ceil(2));
    }

    let packed = tensor_packed_energy(logical_energy, extension_degree)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor-packed response energy overflow".into()))?;
    SourceMomentEstimate::from_moments(packed, logical_peak_ppm)
        .ok_or_else(|| AkitaError::InvalidSetup("modeled recursive witness is empty".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_plane_moments_include_the_residual_top_plane() {
        // fp64 at basis 64 has ten full planes and one four-bit top plane.
        let energy = field_digit_energy(1_000_000, 64, 6, 11).unwrap();
        let expected = 1_000_000.0 * (10.0 * 341.5 + 21.5);
        assert_eq!(energy, expected as u128);
    }

    #[test]
    fn tensor_pack_energy_matches_supported_extension_factors() {
        assert_eq!(tensor_packed_energy(400, 1), Some(400));
        assert_eq!(tensor_packed_energy(400, 2), Some(600));
        assert_eq!(tensor_packed_energy(400, 4), Some(700));
    }

    #[test]
    fn gaussian_z_model_matches_measured_cross_field_states() {
        // (response energy, coefficient count, log basis, digit count,
        //  observed decomposed-Z energy). These are independent honest-prover
        //  measurements from fp32, fp64, and fp128 schedules.
        let rows = [
            (21_319_133_492, 524_288, 3, 4, 8_570_345),
            (352_065_629, 65_536, 4, 3, 2_447_776),
            (3_847_283_483, 262_144, 3, 4, 3_767_203),
            (473_967_459, 65_536, 4, 3, 2_593_330),
            (234_370_171, 32_768, 5, 2, 3_041_573),
            (9_985_694_564, 262_144, 4, 3, 11_458_186),
            (483_233_512, 32_768, 6, 2, 11_379_250),
            (2_853_063_371, 16_384, 6, 2, 6_333_831),
        ];
        for (response, count, log_basis, digits, observed) in rows {
            let predicted =
                gaussian_response_digit_energy(response, count, log_basis, digits).unwrap();
            let relative_error = (predicted as f64 / observed as f64 - 1.0).abs();
            assert!(
                relative_error <= 0.02,
                "response={response} basis={log_basis}: predicted={predicted}, observed={observed}, error={relative_error}"
            );
        }
    }

    #[test]
    fn cap_multiplier_has_markov_grinding_budget() {
        let source = SourceMomentEstimate::new(1_048_576).unwrap();
        assert_eq!(source.response_l2_sq_cap(75), Some(85_862_646));
        // The 3% source envelope followed by the 1.06 grinding allowance.
        assert_eq!(
            85_862_646u128,
            (78_643_200u128 * 1_091_800u128).div_ceil(1_000_000u128)
        );
    }

    #[test]
    fn source_moment_bucketing_is_conservative_and_below_one_over_sixty_four() {
        for value in [1, 127, 128, 129, 1_000_000, u64::MAX as u128] {
            let bucketed = SourceMomentEstimate::new(value).unwrap().mean_l2_sq();
            assert!(bucketed >= value);
            assert!(bucketed - value < value.div_ceil(64).max(1));
        }
    }

    #[test]
    fn empirical_selected_l2_caps_cover_all_profile_families_with_twenty_percent_slack() {
        // (actual source energy, challenge energy, actual response energy,
        //  frozen cap, accepted nonce). These 44 observations come from fresh
        //  honest proofs after the heuristic-free catalog regeneration. They
        //  cover dense and one-hot fp32, fp64, and fp128, direct and recursive
        //  setup, multi-group, and W2/W4/W8 multi-chunk adapters. Every proof
        //  passed both verifier modes.
        let rows: [(u128, u128, u128, u128, u32); 44] = [
            // dense and one-hot fp32
            (10_015_544, 75, 768_874_990, 826_427_966, 0),
            (5_614_768, 75, 423_192_006, 461_511_721, 0),
            (4_613_516, 75, 340_208_986, 381_015_491, 0),
            (15_667_681, 31, 484_351_189, 532_348_404, 0),
            // dense and one-hot fp64
            (9_261_083, 75, 704_832_423, 772_763_812, 0),
            (5_720_601, 75, 429_348_945, 472_244_552, 0),
            (4_042_956, 75, 304_733_422, 332_717_753, 0),
            (3_703_226, 75, 279_849_868, 305_885_676, 0),
            (11_539_902, 75, 869_184_654, 955_221_935, 0),
            (6_234_942, 75, 468_948_420, 515_175_875, 0),
            (4_059_117, 75, 303_673_569, 332_717_753, 0),
            (3_706_588, 75, 271_853_746, 305_885_676, 0),
            // dense, direct one-hot, and recursive one-hot fp128
            (5_768_500, 75, 428_283_744, 477_610_968, 0),
            (3_432_873, 75, 254_190_731, 281_736_807, 0),
            (4_144_476, 75, 308_723_092, 340_767_376, 0),
            (1_859_475, 75, 142_560_067, 152_942_838, 0),
            (4_119_465, 75, 309_590_121, 338_084_168, 0),
            (3_032_670, 75, 226_764_894, 249_538_315, 0),
            (3_137_086, 75, 233_424_472, 257_587_938, 0),
            (4_466_256, 75, 335_391_510, 370_282_660, 0),
            (3_394_786, 75, 250_031_170, 279_053_599, 0),
            // direct and recursive multi-group
            (4_764_050, 75, 384_612_644, 391_748_322, 0),
            (1_940_240, 75, 143_777_206, 159_650_857, 0),
            (4_107_945, 75, 309_801_927, 340_767_376, 0),
            (3_026_122, 75, 224_172_600, 249_538_315, 0),
            (4_764_050, 75, 384_612_644, 391_748_322, 0),
            (1_940_240, 75, 143_777_206, 159_650_857, 0),
            (4_107_945, 75, 309_801_927, 340_767_376, 0),
            (3_026_122, 75, 224_172_600, 249_538_315, 0),
            // recursive multi-group W8R2
            (13_210_454, 75, 974_334_696, 1_084_015_903, 0),
            (3_602_972, 75, 270_721_392, 297_836_053, 0),
            (5_015_157, 75, 377_584_259, 418_580_399, 0),
            (3_421_179, 75, 259_583_015, 281_736_807, 0),
            // W2R2
            (3_231_280, 75, 236_704_824, 265_637_561, 0),
            (4_481_909, 75, 336_395_935, 370_282_660, 0),
            (3_395_692, 75, 258_847_366, 279_053_599, 0),
            // W4R2
            (7_895_398, 75, 582_254_430, 649_336_259, 0),
            (2_697_835, 75, 196_641_137, 222_706_238, 0),
            (4_199_089, 75, 316_426_985, 348_816_999, 0),
            (3_402_924, 75, 253_896_432, 279_053_599, 0),
            // W8R2
            (11_469_631, 75, 867_314_201, 944_489_104, 0),
            (3_502_750, 75, 257_308_170, 287_103_222, 0),
            (4_967_668, 75, 373_659_008, 413_213_983, 0),
            (3_416_981, 75, 255_935_881, 281_736_807, 0),
        ];

        for (source, challenge, response, cap, nonce) in rows {
            assert!(response <= cap, "honest response exceeded its frozen cap");
            assert!(
                cap * 100 <= response * 120,
                "frozen cap used more than twenty percent slack"
            );
            assert_eq!(nonce, 0, "empirical proof required grinding");

            // Recover the planner's frozen source estimate from its cap. The
            // current observations put aggregate source-model error between
            // +0.09 and +1.93 percent.
            let implied_source = cap as f64 * (PPM * PPM) as f64
                / (challenge * SOURCE_MODEL_ENVELOPE_PPM * RESPONSE_MEAN_MULTIPLIER_PPM) as f64;
            let source_ratio = implied_source / source as f64;
            assert!((1.000..=1.020).contains(&source_ratio));
        }
    }
}
