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

use akita_error::{checked, AkitaError};
use akita_types::sis::HonestFoldPolicySpec;
use akita_types::{CommittedGroupParams, OpeningClaimsLayout, WitnessLayout};
use std::cell::RefCell;
use std::collections::HashMap;

/// Relative envelope for any underestimate by the typed moment model.
///
/// Historical cross-profile calibration found at most 2.24 percent error in
/// the unfavorable direction. Three percent keeps model error separate from
/// the response allowance. It covers the rounded Gaussian, pseudo-Mersenne,
/// challenge-covariance, and finite-mixing approximations. Current production
/// measurements are joined to their exact generated schedule by the profile
/// report pipeline. Conservative overestimates do not consume this envelope.
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
    let (per_scalar, peak) = field_digit_shape(field_bits, log_basis, digit_count)?;
    Ok((
        checked_ceil_f64(
            per_scalar * scalar_count as f64,
            "finite-field digit energy",
        )?,
        peak,
    ))
}

fn field_digit_shape(
    field_bits: u32,
    log_basis: u32,
    digit_count: usize,
) -> Result<(f64, u128), AkitaError> {
    let key = (field_bits, log_basis, digit_count);
    if let Some(shape) = FIELD_DIGIT_SHAPES.with(|cache| cache.borrow().get(&key).copied()) {
        return Ok(shape);
    }
    let mut per_scalar = 0.0;
    let mut peak_moment = 0.0f64;
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
        peak_moment = peak_moment.max(moment);
    }
    let shape = (
        per_scalar,
        moment_to_ppm(peak_moment, "finite-field digit peak moment")?,
    );
    FIELD_DIGIT_SHAPES.with(|cache| {
        cache.borrow_mut().insert(key, shape);
    });
    Ok(shape)
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

/// Deterministic maximum squared digit energy of a balanced signed-digit source
/// whose centered coefficients fit `source_log_bound` bits.
///
/// `source_log_bound` is the **declared committed-source bound**, not the field
/// width: a bounded source stops short of the field, so its final digit plane
/// only spans the bits the bound leaves. Charging that plane a full `log_basis`
/// of range would over-estimate its energy and inflate the L2 response cap the
/// A rank is priced against. A full-field source passes its own field width and
/// is unaffected.
///
/// Planes past the bound are **not** free. Balanced extraction carries:
/// `|c_p| <= |v| / b^p + b / (2·(b - 1))`, so the plane just past the bound can
/// still hold `±1`, and the canonical depth adds exactly one such plane whenever
/// `log_basis` divides `source_log_bound` (the `+1` correction in
/// `compute_num_digits`). Charging it `1` instead of dropping it keeps this a
/// true deterministic maximum. This never fires for a full-field source, whose
/// depth is `ceil(field_bits / log_basis)` and so never overshoots by a whole
/// plane.
fn bounded_field_source_moment(
    scalar_count: usize,
    source_log_bound: u32,
    log_basis: u32,
    digit_count: usize,
) -> Result<SourceMomentEstimate, AkitaError> {
    if scalar_count == 0 || source_log_bound == 0 || log_basis == 0 || digit_count == 0 {
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
        // `max(1)` is the carry plane described above; below the bound this is
        // just `min(log_basis, source_log_bound - consumed)`.
        let plane_bits = source_log_bound
            .saturating_sub(consumed)
            .min(log_basis)
            .max(1);
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

#[cfg(test)]
fn normal_cdf(value: f64) -> f64 {
    0.5 * (1.0 + libm::erf(value / core::f64::consts::SQRT_2))
}

type GaussianDigitMomentKey = (u128, usize, u128, u32, usize);
type GaussianDigitMoment = (u128, u128);
type FieldDigitShapeKey = (u32, u32, usize);
type FieldDigitShape = (f64, u128);

thread_local! {
    // The quantile depends only on the response coefficient count, while one
    // planner worker evaluates it for many source moments, slices, and DP
    // states. Keep the exact binary-search result local to that worker so the
    // offline search does not repeat 64 software `erfc` evaluations per
    // candidate or contend on a process-global cache.
    static WHOLE_RESPONSE_NORMAL_QUANTILES: RefCell<HashMap<usize, f64>> =
        RefCell::new(HashMap::new());
    // A finite-field decomposition shape depends only on its field width,
    // basis, and plane count. Candidate states change the scalar count but
    // otherwise repeat this exact shape, so cache the per-scalar and peak
    // moments rather than rerunning the digit-plane sweep.
    static FIELD_DIGIT_SHAPES: RefCell<HashMap<FieldDigitShapeKey, FieldDigitShape>> =
        RefCell::new(HashMap::new());
    // Recursive DP states often reproduce the same typed response moment and
    // decomposition geometry. The Gaussian digit calculation is deterministic
    // but may perform hundreds of software `erf` evaluations, so retain each
    // exact successful result within its planner worker.
    static GAUSSIAN_RESPONSE_DIGIT_MOMENTS:
        RefCell<HashMap<GaussianDigitMomentKey, GaussianDigitMoment>> =
        RefCell::new(HashMap::new());
}

/// Two-sided standard-normal quantile whose joint centered-Gaussian slab
/// probability is at least one fortieth over `count` coordinates.
///
/// The Gaussian correlation inequality lower-bounds the probability of the
/// intersection of symmetric coordinate slabs by the product of their
/// marginal probabilities, regardless of the covariance matrix. Thus each
/// marginal needs probability at least `(1/40)^(1/count)`; coordinate
/// independence is not assumed.
fn compute_whole_response_normal_quantile(count: usize) -> f64 {
    debug_assert!(count != 0);
    if count == 0 {
        return 0.0;
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
    upper
}

fn whole_response_normal_quantile(count: usize) -> Option<f64> {
    if count == 0 {
        return None;
    }
    Some(WHOLE_RESPONSE_NORMAL_QUANTILES.with(|cache| {
        if let Some(&quantile) = cache.borrow().get(&count) {
            return quantile;
        }
        let quantile = compute_whole_response_normal_quantile(count);
        cache.borrow_mut().insert(count, quantile);
        quantile
    }))
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
    // The rounded centered normal and squared balanced residue are both even.
    // Walk only positive buckets and evaluate their combined two-sided mass
    // with the upper tail. Besides halving the software-erf work, `erfc(a) -
    // erfc(b)` avoids subtracting two values near one in the positive tail.
    let tail_scale = sigma * core::f64::consts::SQRT_2;
    let mut lower_tail = libm::erfc(0.5 / tail_scale);
    for value in 1..=radius {
        let upper_tail = libm::erfc((value as f64 + 0.5) / tail_scale);
        let probability = lower_tail - upper_tail;
        let digit = centered_residue(value, basis) as f64;
        moment += probability * digit * digit;
        lower_tail = upper_tail;
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

fn compute_gaussian_response_digit_moments(
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

fn gaussian_response_digit_moments(
    response_l2_sq: u128,
    response_coeff_count: usize,
    peak_response_second_moment_ppm: u128,
    log_basis: u32,
    digit_count: usize,
) -> Result<(u128, u128), AkitaError> {
    let key = (
        response_l2_sq,
        response_coeff_count,
        peak_response_second_moment_ppm,
        log_basis,
        digit_count,
    );
    if let Some(moments) =
        GAUSSIAN_RESPONSE_DIGIT_MOMENTS.with(|cache| cache.borrow().get(&key).copied())
    {
        return Ok(moments);
    }
    let moments = compute_gaussian_response_digit_moments(
        response_l2_sq,
        response_coeff_count,
        peak_response_second_moment_ppm,
        log_basis,
        digit_count,
    )?;
    GAUSSIAN_RESPONSE_DIGIT_MOMENTS.with(|cache| {
        cache.borrow_mut().insert(key, moments);
    });
    Ok(moments)
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
    checked::pow2(num_vars)
        .and_then(|len| checked::product([len, num_polynomials]))
        .ok_or_else(|| AkitaError::InvalidSetup("root source length overflow".into()))
}

/// Source moments of each root opening group before its first fold.
///
/// `decomposition` supplies both the field width and the final group's declared
/// committed-source bound (`log_commit_bound`). Precommitted groups were frozen
/// by a possibly different producer whose bound is not carried in their params,
/// so they are priced at the shared field width — always a valid upper bound on
/// their source energy, and exactly the previous behavior.
pub(crate) fn root_group_source_moments(
    params: &CommittedGroupParams,
    opening_layout: &OpeningClaimsLayout,
    final_policy: HonestFoldPolicySpec,
    precommitted_policies: &[HonestFoldPolicySpec],
    decomposition: akita_types::DecompositionParams,
) -> Result<Vec<SourceMomentEstimate>, AkitaError> {
    let field_bits = decomposition.field_bits();
    let final_group_index = opening_layout.root_final_group_index()?;
    params.validate_opening_batch(opening_layout)?;
    if precommitted_policies.len() != final_group_index {
        return Err(AkitaError::InvalidSetup(
            "root response model requires one policy per precommitted group".into(),
        ));
    }
    let mut moments = Vec::with_capacity(opening_layout.num_groups());
    for group_index in 0..opening_layout.num_groups() {
        let group_layout = *opening_layout.group_layout(group_index)?;
        // Validate the grouped batch once above, then resolve each source view
        // directly. The public group accessor would revalidate every group for
        // every root candidate.
        let group_params = if group_index == final_group_index {
            params.final_group()
        } else {
            *params
                .preceding_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?
        };
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
                if !logical_len.is_multiple_of(chunk) {
                    return Err(AkitaError::InvalidSetup(
                        "unit one-hot root length must be a multiple of its source chunk size"
                            .into(),
                    ));
                }
                let (energy, coefficient_sq_max) = policy
                    .root_source_l2_sq(
                        logical_len,
                        group_params.inner_commit_matrix_params().ring_dimension(),
                    )
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "unit one-hot root source geometry is unsupported".into(),
                        )
                    })?;
                let peak = coefficient_sq_max.checked_mul(MOMENT_PPM).ok_or_else(|| {
                    AkitaError::InvalidSetup("unit one-hot root peak moment overflow".into())
                })?;
                let mut components = [SourceMomentComponent::default(); SOURCE_COMPONENT_COUNT];
                components[Z_COMPONENT] = SourceMomentComponent {
                    mean_l2_sq: energy,
                    full_ring_peak_second_moment_ppm: peak,
                    local_peak_second_moment_ppm: peak,
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
                let source_log_bound = if group_index == final_group_index {
                    decomposition.log_commit_bound
                } else {
                    field_bits
                };
                bounded_field_source_moment(
                    logical_len,
                    source_log_bound,
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
    let relation_geometry =
        akita_types::RelationWitnessGeometry::for_level(params, opening_layout, extension_degree)?;
    let layout = WitnessLayout::new(
        params,
        opening_layout,
        &relation_geometry,
        params.witness_chunk.num_chunks,
        akita_types::RelationQuotientPlan::for_field_bits(params, field_bits)?,
    )?;
    let mut logical_components = [SourceMomentComponent::default(); SOURCE_COMPONENT_COUNT];
    let final_group_index = opening_layout.root_final_group_index()?;

    for unit in layout.units() {
        let group_index = unit.group_index();
        // Relation and witness geometry construction already validated the
        // complete opening batch. Resolve the group directly here instead of
        // repeating that validation for every group-and-chunk unit.
        let group_params = if group_index == final_group_index {
            params.final_group()
        } else {
            *params
                .preceding_group_params(group_index)
                .ok_or(AkitaError::InvalidProof)?
        };
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
        let group_d_a = relation_geometry
            .rhs_layout()
            .groups
            .iter()
            .find_map(|group| (group.group_index == group_index).then_some(group.role_dims.d_a()))
            .ok_or_else(|| AkitaError::InvalidSetup("response relation group is missing".into()))?;
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
        let opening_width = relation_geometry
            .group_opening_geometry(group_index)?
            .physical_coefficient_width();
        let e_scalar_count = num_claims
            .checked_mul(unit.num_live_blocks())
            .and_then(|count| count.checked_mul(opening_width))
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
        let t_scalar_count = num_claims
            .checked_mul(unit.num_live_blocks())
            .and_then(|count| count.checked_mul(group_d_a))
            .and_then(|count| count.checked_mul(group_params.a_rows_len()))
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

    if let Some(quotient_depth) = layout.quotient_depth() {
        for row in layout.r_rows() {
            let scalar_count = row.range().len() / quotient_depth;
            if scalar_count != 0 {
                let (energy, peak) = field_digit_moments(
                    scalar_count,
                    field_bits,
                    params.open().digits.log_basis,
                    quotient_depth,
                )?;
                checked_add_component(&mut logical_components, R_COMPONENT, energy, peak)?;
            }
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
#[path = "response_model_tests.rs"]
mod tests;
