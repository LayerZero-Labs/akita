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
/// Markov's inequality gives `Pr[X <= (40/39) E[X]] >= 1/40` for every
/// nonnegative response energy `X`. Thus this is a distribution-free
/// completeness guarantee for grinding, not a Gaussian-tail assumption. With
/// 4096 independent transcript attempts, the exhaustion probability is below
/// `2^-149`.
const RESPONSE_MEAN_MULTIPLIER_NUMERATOR: u128 = 40;
const RESPONSE_MEAN_MULTIPLIER_DENOMINATOR: u128 = 39;
const PPM: u128 = 1_000_000;
const MOMENT_PPM: u128 = 1_000_000;

const SOURCE_COMPONENT_COUNT: usize = 5;
const Z_COMPONENT: usize = 0;
const E_COMPONENT: usize = 1;
const T_COMPONENT: usize = 2;
const R_COMPONENT: usize = 3;
const COMPRESSION_COMPONENT: usize = 4;

fn round_moment_up(value: u128) -> Option<u128> {
    if value == 0 {
        return Some(0);
    }
    let significant_bits = 7u32;
    let bit_len = u128::BITS - value.leading_zeros();
    let discard = bit_len.saturating_sub(significant_bits);
    let quantum = 1u128 << discard;
    value
        .checked_add(quantum - 1)
        .map(|rounded| rounded & !(quantum - 1))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct SourceMomentComponent {
    mean_l2_sq: u128,
    full_ring_peak_second_moment_ppm: u128,
    local_peak_second_moment_ppm: u128,
}

/// Planner estimate of the typed second moments of one recursive witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SourceMomentEstimate {
    mean_l2_sq: u128,
    components: [SourceMomentComponent; SOURCE_COMPONENT_COUNT],
    packing_ring_dimension: usize,
}

impl SourceMomentEstimate {
    /// Retain seven leading bits and round the remaining energy upward.
    ///
    /// This gives the suffix DP a bounded, reusable state domain while adding
    /// less than 1/64 relative error. The cap is conservative because the
    /// rounding direction is always upward.
    #[cfg(test)]
    pub(crate) fn new(mean_l2_sq: u128) -> Option<Self> {
        Self::from_moments(mean_l2_sq, mean_l2_sq.saturating_mul(MOMENT_PPM))
    }

    /// Build a bounded DP state for a source with one coordinate class.
    pub(crate) fn from_moments(mean_l2_sq: u128, peak_second_moment_ppm: u128) -> Option<Self> {
        let mut components = [SourceMomentComponent::default(); SOURCE_COMPONENT_COUNT];
        components[Z_COMPONENT] = SourceMomentComponent {
            mean_l2_sq,
            full_ring_peak_second_moment_ppm: peak_second_moment_ppm,
            local_peak_second_moment_ppm: peak_second_moment_ppm,
        };
        Self::from_components(components, 0)
    }

    fn from_components(
        mut components: [SourceMomentComponent; SOURCE_COMPONENT_COUNT],
        packing_ring_dimension: usize,
    ) -> Option<Self> {
        let mut mean_l2_sq = 0u128;
        for component in &mut components {
            if component.mean_l2_sq == 0 {
                *component = SourceMomentComponent::default();
                continue;
            }
            if component.full_ring_peak_second_moment_ppm == 0
                || component.local_peak_second_moment_ppm == 0
            {
                return None;
            }
            component.mean_l2_sq = round_moment_up(component.mean_l2_sq)?;
            component.full_ring_peak_second_moment_ppm =
                round_moment_up(component.full_ring_peak_second_moment_ppm)?;
            component.local_peak_second_moment_ppm =
                round_moment_up(component.local_peak_second_moment_ppm)?;
            mean_l2_sq = mean_l2_sq.checked_add(component.mean_l2_sq)?;
        }
        (mean_l2_sq != 0).then_some(Self {
            mean_l2_sq,
            components,
            packing_ring_dimension,
        })
    }

    pub(crate) const fn mean_l2_sq(self) -> u128 {
        self.mean_l2_sq
    }

    fn peak_column_second_moment_ppm(
        self,
        ring_dimension: usize,
        blocks_per_chunk: usize,
    ) -> Option<u128> {
        let mut remaining_coefficients =
            (ring_dimension as u128).checked_mul(blocks_per_chunk as u128)?;
        let mut buckets = [(0u128, 0u128); 2 * SOURCE_COMPONENT_COUNT];
        for (index, component) in self.components.iter().enumerate() {
            if component.mean_l2_sq == 0 {
                continue;
            }
            let peak = if self.packing_ring_dimension == 0
                || self.packing_ring_dimension == ring_dimension
            {
                component.full_ring_peak_second_moment_ppm
            } else {
                component.local_peak_second_moment_ppm
            };
            let total = component.mean_l2_sq.checked_mul(MOMENT_PPM)?;
            buckets[2 * index] = (peak, total / peak);
            if !total.is_multiple_of(peak) {
                buckets[2 * index + 1] = (total % peak, 1);
            }
        }
        buckets.sort_unstable_by_key(|&(value, _)| std::cmp::Reverse(value));
        let mut column_moment_ppm = 0u128;
        for (value, count) in buckets {
            let occupied_coefficients = remaining_coefficients.min(count);
            column_moment_ppm =
                column_moment_ppm.checked_add(value.checked_mul(occupied_coefficients)?)?;
            remaining_coefficients -= occupied_coefficients;
            if remaining_coefficients == 0 {
                break;
            }
        }
        Some(column_moment_ppm)
    }

    fn peak_response_second_moment_ppm(
        self,
        challenge_l2_sq: u128,
        ring_dimension: usize,
        blocks_per_chunk: usize,
    ) -> Option<u128> {
        self.peak_column_second_moment_ppm(ring_dimension, blocks_per_chunk)?
            .checked_mul(challenge_l2_sq)?
            .checked_add(ring_dimension as u128 - 1)
            .map(|rounded| rounded / ring_dimension as u128)
    }

    /// Freeze the planner's response-energy cap for one challenge family.
    pub(crate) fn response_l2_sq_cap(self, challenge_l2_sq: u128) -> Option<u128> {
        let numerator = self
            .mean_l2_sq
            .checked_mul(challenge_l2_sq)?
            .checked_mul(SOURCE_MODEL_ENVELOPE_PPM)?
            .checked_mul(RESPONSE_MEAN_MULTIPLIER_NUMERATOR)?;
        let scale = PPM.checked_mul(RESPONSE_MEAN_MULTIPLIER_DENOMINATOR)?;
        numerator
            .checked_add(scale - 1)
            .map(|rounded| rounded / scale)
    }

    /// Model a whole-response maximum at per-attempt acceptance probability 1/40.
    ///
    /// The selected threshold uses the two-sided normal quantile whose joint
    /// Gaussian slab probability is at least 1/40. The peak proxy fills the
    /// available source-column coordinates from the
    /// highest-moment Z, E, T, R, or compression class first. Each class is
    /// limited by its total energy, so disjoint classes share one column
    /// instead of each receiving the full block capacity.
    pub(crate) fn response_linf_cap(
        self,
        challenge_l2_sq: u128,
        num_live_blocks: usize,
        num_chunks: usize,
        num_fold_coeffs: usize,
        ring_dimension: usize,
    ) -> Option<u128> {
        if challenge_l2_sq == 0
            || num_live_blocks == 0
            || num_chunks == 0
            || num_fold_coeffs == 0
            || ring_dimension == 0
        {
            return None;
        }
        let average_variance =
            self.mean_l2_sq.checked_mul(challenge_l2_sq)? as f64 / num_fold_coeffs as f64;
        let blocks_per_chunk = num_live_blocks.div_ceil(num_chunks) as u128;
        let peak_variance = self.peak_response_second_moment_ppm(
            challenge_l2_sq,
            ring_dimension,
            blocks_per_chunk as usize,
        )? as f64
            / MOMENT_PPM as f64;
        let variance =
            average_variance.max(peak_variance) * SOURCE_MODEL_ENVELOPE_PPM as f64 / PPM as f64;
        let normal_quantile = whole_response_normal_quantile(num_fold_coeffs)?;
        let threshold = variance.sqrt() * normal_quantile;
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

/// Two-sided standard-normal quantile whose joint centered-Gaussian slab
/// probability is at least one fortieth over `count` coordinates.
///
/// The Gaussian correlation inequality lower-bounds the probability of the
/// intersection of symmetric coordinate slabs by the product of their
/// marginal probabilities, regardless of the covariance matrix. Thus each
/// marginal needs probability at least `(1/40)^(1/count)`; coordinate
/// independence is not assumed.
fn whole_response_normal_quantile(count: usize) -> Option<f64> {
    if count == 0 {
        return None;
    }
    let target_tail = -libm::expm1(libm::log(1.0 / 40.0) / count as f64);
    let mut lower = 0.0;
    let mut upper = 16.0;
    for _ in 0..64 {
        let midpoint = (lower + upper) * 0.5;
        let two_sided_tail = libm::erfc(midpoint / core::f64::consts::SQRT_2);
        if two_sided_tail > target_tail {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    Some(upper)
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

/// Apply the extension-field tensor-packing transform to source moments.
///
/// Coordinate zero appears once. Each of the other `K-1` coordinates appears
/// twice with opposite signs, so total energy has the exact multiplier
/// `(2K-1)/K` under exchangeable extension coordinates. A complete packed ring
/// has the same average peak-column multiplier. A strict subring can isolate
/// overlap positions, however, so it retains the local `2P` bound.
pub(crate) fn tensor_packed_moments(
    logical_energy: u128,
    logical_peak_second_moment_ppm: u128,
    extension_degree: usize,
) -> Option<(u128, u128, u128)> {
    if logical_energy == 0 || logical_peak_second_moment_ppm == 0 || extension_degree == 0 {
        return None;
    }
    let numerator =
        logical_energy.checked_mul((extension_degree as u128).checked_mul(2)?.checked_sub(1)?)?;
    let packed_energy = numerator
        .checked_add(extension_degree as u128 - 1)
        .map(|rounded| rounded / extension_degree as u128)?;
    let peak_numerator = logical_peak_second_moment_ppm
        .checked_mul((extension_degree as u128).checked_mul(2)?.checked_sub(1)?)?;
    let packed_peak = peak_numerator
        .checked_add(extension_degree as u128 - 1)
        .map(|rounded| rounded / extension_degree as u128)?;
    let local_peak =
        logical_peak_second_moment_ppm.checked_mul(if extension_degree == 1 { 1 } else { 2 })?;
    Some((packed_energy, packed_peak, local_peak))
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
                let chunk = group_layout.source().onehot_chunk_size().ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "unit one-hot response policy requires one-hot group metadata".into(),
                    )
                })?;
                if !logical_len.is_multiple_of(chunk) {
                    return Err(AkitaError::InvalidSetup(
                        "unit one-hot root length must be a multiple of its source chunk size"
                            .into(),
                    ));
                }
                let (energy, full_peak, local_peak) = tensor_packed_moments(
                    (logical_len / chunk) as u128,
                    MOMENT_PPM,
                    if akita_types::root_tensor_projection_enabled_for_width(
                        onehot.extension_degree(),
                        group_params.inner_commit_matrix_params().ring_dimension(),
                        group_layout.num_vars(),
                    ) {
                        onehot.extension_degree()
                    } else {
                        1
                    },
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("unit one-hot root source is empty".into())
                })?;
                let mut components = [SourceMomentComponent::default(); SOURCE_COMPONENT_COUNT];
                components[Z_COMPONENT] = SourceMomentComponent {
                    mean_l2_sq: energy,
                    full_ring_peak_second_moment_ppm: full_peak,
                    local_peak_second_moment_ppm: local_peak,
                };
                SourceMomentEstimate::from_components(
                    components,
                    group_params.inner_commit_matrix_params().ring_dimension(),
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("unit one-hot source moments overflow".into())
                })?
            }
            HonestFoldPolicySpec::BalancedSignedDigit(_) => {
                if group_layout.source() != akita_types::RootSourceProfile::Dense {
                    return Err(AkitaError::InvalidSetup(
                        "balanced response policy requires dense group metadata".into(),
                    ));
                }
                bounded_field_source_moment(
                    logical_len,
                    field_bits,
                    group_params.log_basis_inner(),
                    group_params.num_digits_inner(),
                )?
            }
        };
        moments.push(moment);
    }
    Ok(moments)
}

fn checked_add_component(
    components: &mut [SourceMomentComponent; SOURCE_COMPONENT_COUNT],
    index: usize,
    energy: u128,
    peak_second_moment_ppm: u128,
) -> Result<(), AkitaError> {
    let component = components
        .get_mut(index)
        .ok_or_else(|| AkitaError::InvalidSetup("response-model component is missing".into()))?;
    component.mean_l2_sq = component
        .mean_l2_sq
        .checked_add(energy)
        .ok_or_else(|| AkitaError::InvalidSetup("response-model energy overflow".into()))?;
    component.full_ring_peak_second_moment_ppm = component
        .full_ring_peak_second_moment_ppm
        .max(peak_second_moment_ppm);
    component.local_peak_second_moment_ppm = component.full_ring_peak_second_moment_ppm;
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
    let mut logical_components = [SourceMomentComponent::default(); SOURCE_COMPONENT_COUNT];

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
        let group_d_a = params.group_role_dims(opening_layout, group_index)?.d_a();
        let response_energy = chunk_source
            .checked_mul(group_params.fold_challenge_config().challenge_l2_sq_max())
            .ok_or_else(|| AkitaError::InvalidSetup("fold response energy overflow".into()))?;
        let response_coeff_count = unit.z_range().len() / group_params.num_digits_fold();
        if response_energy != 0 && response_coeff_count != 0 {
            let peak_response_second_moment_ppm = group_source
                .peak_response_second_moment_ppm(
                    group_params.fold_challenge_config().challenge_l2_sq_max(),
                    group_d_a,
                    unit.num_live_blocks(),
                )
                .ok_or_else(|| AkitaError::InvalidSetup("fold response peak overflow".into()))?;
            let (energy, peak) = gaussian_response_digit_moments(
                response_energy,
                response_coeff_count,
                peak_response_second_moment_ppm,
                group_params.log_basis_open(),
                group_params.num_digits_fold(),
            )?;
            checked_add_component(&mut logical_components, Z_COMPONENT, energy, peak)?;
        }

        let num_claims = opening_layout.group_layout(group_index)?.num_polynomials();
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
            checked_add_component(&mut logical_components, E_COMPONENT, energy, peak)?;
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
            checked_add_component(&mut logical_components, T_COMPONENT, energy, peak)?;
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
            checked_add_component(&mut logical_components, R_COMPONENT, energy, peak)?;
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
    if compression_coefficients != 0 {
        checked_add_component(
            &mut logical_components,
            COMPRESSION_COMPONENT,
            compression_digit_energy(compression_coefficients),
            MOMENT_PPM.div_ceil(2),
        )?;
    }

    let mut packed_components = [SourceMomentComponent::default(); SOURCE_COMPONENT_COUNT];
    for (packed, logical) in packed_components.iter_mut().zip(logical_components) {
        if logical.mean_l2_sq == 0 {
            continue;
        }
        let (energy, full_ring_peak, local_peak) = tensor_packed_moments(
            logical.mean_l2_sq,
            logical.full_ring_peak_second_moment_ppm,
            extension_degree,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("tensor-packed response energy overflow".into()))?;
        *packed = SourceMomentComponent {
            mean_l2_sq: energy,
            full_ring_peak_second_moment_ppm: full_ring_peak,
            local_peak_second_moment_ppm: local_peak,
        };
    }
    SourceMomentEstimate::from_components(packed_components, params.d_a())
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
    fn tensor_pack_moments_match_supported_extension_factors() {
        assert_eq!(tensor_packed_moments(400, 100, 1), Some((400, 100, 100)));
        assert_eq!(tensor_packed_moments(400, 100, 2), Some((600, 150, 200)));
        assert_eq!(tensor_packed_moments(400, 100, 4), Some((700, 175, 200)));
        assert_eq!(tensor_packed_moments(400, 100, 8), Some((750, 188, 200)));
    }

    #[test]
    fn peak_column_shares_capacity_across_disjoint_components() {
        const PEAK: u128 = 1 << 24;
        let component = SourceMomentComponent {
            mean_l2_sq: 1024,
            full_ring_peak_second_moment_ppm: PEAK,
            local_peak_second_moment_ppm: 2 * PEAK,
        };
        let source = SourceMomentEstimate::from_components(
            [
                component,
                component,
                component,
                component,
                Default::default(),
            ],
            8,
        )
        .unwrap();

        assert_eq!(
            source.peak_column_second_moment_ppm(8, 1),
            Some(8 * PEAK),
            "four disjoint component classes must share one eight-coefficient column"
        );
        assert_eq!(
            source.peak_column_second_moment_ppm(4, 2),
            Some(16 * PEAK),
            "a strict subring retains the local two-coordinate packing bound"
        );
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
        assert_eq!(source.response_l2_sq_cap(75), Some(83_079_484));
        // The 3% source envelope followed by the 40/39 grinding allowance.
        assert_eq!(
            83_079_484u128,
            (78_643_200u128 * 1_030_000u128 * 40).div_ceil(1_000_000u128 * 39)
        );
    }

    #[test]
    fn gaussian_slab_quantile_meets_joint_grinding_target() {
        let count = 16_384;
        let quantile = whole_response_normal_quantile(count).unwrap();
        let marginal = 1.0 - libm::erfc(quantile / core::f64::consts::SQRT_2);
        let joint_lower_bound = libm::exp(count as f64 * libm::log(marginal));
        assert!((joint_lower_bound * 40.0 - 1.0).abs() <= 1e-9);
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
        //  frozen cap, attempts). These 51 observations come from fresh honest
        //  proofs after the final catalog regeneration. They
        //  cover dense and one-hot fp32, fp64, and fp128, direct and recursive
        //  setup, multi-group, and W2/W4/W8 multi-chunk adapters. Every proof
        //  passed both verifier modes.
        let rows: [(u128, u128, u128, u128, u32); 51] = [
            // dense and one-hot fp64
            (36_377_951, 75, 2_741_456_911, 2_888_229_022, 1),
            (19_742_844, 75, 1_486_348_884, 1_596_602_684, 1),
            (39_353_267, 75, 2_904_628_561, 3_177_790_228, 1),
            (31_818_884, 75, 2_357_542_772, 2_544_309_170, 4),
            (4_773_827, 75, 359_247_329, 380_835_053, 1),
            (21_415_054, 75, 1_605_640_842, 1_716_029_440, 1),
            (39_779_231, 75, 3_051_107_723, 3_177_790_228, 1),
            (31_794_955, 75, 2_387_505_109, 2_544_309_170, 3),
            // dense and one-hot fp32
            (40_636_017, 31, 1_264_250_777, 1_345_730_230, 1),
            (42_533_515, 31, 1_318_821_345, 1_397_725_762, 1),
            (33_290_268, 31, 1_027_222_760, 1_098_864_630, 1),
            (40_718_457, 31, 1_269_963_959, 1_341_437_790, 1),
            (42_433_311, 31, 1_313_863_335, 1_397_725_762, 1),
            (33_295_963, 31, 1_006_319_953, 1_098_864_630, 3),
            // dense, direct one-hot, and recursive one-hot fp128
            (4_880_485, 75, 360_521_601, 387_650_167, 2),
            (1_933_979, 75, 141_335_441, 154_232_517, 1),
            (4_102_518, 75, 307_852_910, 328_910_376, 1),
            (8_333_578, 75, 623_432_126, 661_390_573, 1),
            (6_618_349, 75, 481_287_879, 527_197_736, 1),
            (6_802_195, 75, 501_184_457, 545_695_902, 1),
            (3_986_275, 75, 296_217_683, 319_174_499, 1),
            (8_253_947, 75, 611_007_933, 661_390_573, 4),
            (9_086_349, 75, 655_245_343, 723_862_450, 2),
            (8_480_847, 75, 624_559_207, 678_752_887, 1),
            (4_362_787, 75, 322_738_451, 347_084_013, 1),
            (8_302_641, 75, 618_622_149, 663_986_807, 1),
            // direct and recursive multi-group
            (4_361_032, 75, 347_763_142, 348_057_600, 1),
            (1_875_326, 75, 143_670_644, 149_202_314, 1),
            (4_105_775, 75, 309_862_343, 327_612_259, 1),
            (8_297_748, 75, 623_193_312, 661_390_573, 6),
            (27_365_041, 75, 2_032_457_103, 2_176_130_757, 1),
            (8_497_770, 75, 624_646_072, 678_428_357, 1),
            (4_349_937, 75, 320_226_849, 347_084_013, 1),
            (8_251_577, 75, 630_948_691, 663_986_807, 1),
            // recursive multi-group W8R2
            (83_730_610, 75, 6_095_056_218, 6_720_919_237, 1),
            (8_920_740, 75, 655_485_118, 714_207_705, 1),
            (8_491_858, 75, 642_392_086, 678_752_887, 1),
            (4_359_942, 75, 329_555_882, 347_084_013, 1),
            (8_331_832, 75, 625_512_838, 663_986_807, 5),
            // W2R2
            (7_729_843, 75, 582_150_045, 615_469_687, 1),
            (7_382_860, 75, 557_351_162, 590_480_936, 1),
            (4_191_220, 75, 324_120_482, 333_453_785, 1),
            (8_340_674, 75, 619_814_264, 661_390_573, 5),
            // W4R2
            (9_004_434, 75, 680_329_646, 712_179_397, 1),
            (8_485_411, 75, 625_088_747, 678_752_887, 1),
            (4_346_063, 75, 333_780_161, 347_084_013, 1),
            (8_358_015, 75, 627_831_795, 663_986_807, 4),
            // W8R2
            (25_391_642, 75, 1_890_084_424, 2_028_145_428, 1),
            (8_270_359, 75, 620_511_743, 658_956_604, 1),
            (4_365_132, 75, 323_097_542, 347_084_013, 1),
            (8_352_664, 75, 615_112_704, 663_986_807, 2),
        ];

        for (source, challenge, response, cap, attempts) in rows {
            assert!(response <= cap, "honest response exceeded its frozen cap");
            assert!(
                cap * 100 <= response * 120,
                "frozen cap used more than twenty percent slack"
            );
            assert!(
                attempts <= 6,
                "empirical proof used an unexpected attempt count"
            );

            // Recover the planner's frozen source estimate from its cap. The
            // current observations put aggregate source-model error between
            // -0.18 and +2.07 percent.
            let implied_source = cap as f64 * (PPM * RESPONSE_MEAN_MULTIPLIER_DENOMINATOR) as f64
                / (challenge * SOURCE_MODEL_ENVELOPE_PPM * RESPONSE_MEAN_MULTIPLIER_NUMERATOR)
                    as f64;
            let source_ratio = implied_source / source as f64;
            assert!((0.998..=1.021).contains(&source_ratio));
        }
    }
}
