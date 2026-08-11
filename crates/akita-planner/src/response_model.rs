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
/// Aggregate source measurements across fp32, fp64, and fp128 currently put the
/// largest observed underestimate below 0.21 percent. Individual typed terms
/// can differ by up to 2.24 percent in the unfavorable direction, but their
/// errors did not align in the measured source vectors. Three percent keeps
/// model error separate from the response allowance. It covers the rounded
/// Gaussian, pseudo-Mersenne, challenge-covariance, and finite-mixing
/// approximations. Conservative overestimates do not consume this envelope.
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

/// Planner estimate of the squared Euclidean norm of one recursive witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SourceMomentEstimate {
    mean_l2_sq: u128,
}

impl SourceMomentEstimate {
    /// Retain seven leading bits and round the remaining energy upward.
    ///
    /// This gives the suffix DP a bounded, reusable state domain while adding
    /// less than 1/64 relative error. The cap is conservative because the
    /// rounding direction is always upward.
    pub(crate) const fn new(mean_l2_sq: u128) -> Option<Self> {
        if mean_l2_sq == 0 {
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
            Some(Self {
                mean_l2_sq: rounded,
            })
        }
    }

    pub(crate) const fn mean_l2_sq(self) -> u128 {
        self.mean_l2_sq
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

/// Modeled energy of a full-width finite-field balanced decomposition.
///
/// The model uses the uniform centered moment for each complete plane. This is
/// exact for a uniform power-of-two residue. The last plane uses the residual
/// field width rather than pretending that it is another full plane. The
/// supported pseudo-Mersenne moduli differ from `2^field_bits` by a negligible
/// fraction. Recursive E, T, and R values can also retain correlation instead
/// of being fully mixed. The explicit model envelope covers unfavorable error;
/// retained correlation usually makes this estimate conservative.
pub(crate) fn field_digit_energy(
    scalar_count: usize,
    field_bits: u32,
    log_basis: u32,
    digit_count: usize,
) -> Result<u128, AkitaError> {
    if scalar_count == 0 || field_bits == 0 || log_basis == 0 || digit_count == 0 {
        return Err(AkitaError::InvalidSetup(
            "field digit moment requires positive geometry".into(),
        ));
    }
    let mut per_scalar = 0.0;
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
        per_scalar += centered_uniform_digit_second_moment(basis)
            .ok_or_else(|| AkitaError::InvalidSetup("digit-plane basis is not supported".into()))?;
    }
    checked_ceil_f64(
        per_scalar * scalar_count as f64,
        "finite-field digit energy",
    )
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
        let mean_l2_sq = match policy {
            HonestFoldPolicySpec::UnitOneHot(onehot) => {
                let chunk = onehot.source_chunk_size();
                if chunk == 0 || !logical_len.is_multiple_of(chunk) {
                    return Err(AkitaError::InvalidSetup(
                        "unit one-hot root length must be a multiple of its source chunk size"
                            .into(),
                    ));
                }
                (logical_len / chunk) as u128
            }
            HonestFoldPolicySpec::BalancedSignedDigit(_) => field_digit_energy(
                logical_len,
                field_bits,
                group_params.log_basis_inner(),
                group_params.num_digits_inner(),
            )?,
        };
        moments.push(SourceMomentEstimate::new(mean_l2_sq).ok_or_else(|| {
            AkitaError::InvalidSetup("root response source has zero modeled energy".into())
        })?);
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
            checked_add_energy(
                &mut logical_energy,
                gaussian_response_digit_energy(
                    response_energy,
                    response_coeff_count,
                    group_params.log_basis_open(),
                    group_params.num_digits_fold(),
                )?,
            )?;
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
            checked_add_energy(
                &mut logical_energy,
                field_digit_energy(
                    e_scalar_count,
                    field_bits,
                    group_params.log_basis_open(),
                    group_params.num_digits_open(),
                )?,
            )?;
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
            checked_add_energy(
                &mut logical_energy,
                field_digit_energy(
                    t_scalar_count,
                    field_bits,
                    group_params.log_basis_outer(),
                    group_params.num_digits_outer(),
                )?,
            )?;
        }
    }

    for row in layout.r_rows() {
        let scalar_count = row.range().len() / quotient_depth;
        if scalar_count != 0 {
            checked_add_energy(
                &mut logical_energy,
                field_digit_energy(
                    scalar_count,
                    field_bits,
                    params.log_basis_open,
                    quotient_depth,
                )?,
            )?;
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

    let packed = tensor_packed_energy(logical_energy, extension_degree)
        .ok_or_else(|| AkitaError::InvalidSetup("tensor-packed response energy overflow".into()))?;
    SourceMomentEstimate::new(packed)
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
    fn empirical_selected_l2_caps_cover_every_profile_with_twenty_percent_slack() {
        // (actual source energy, challenge energy, actual response energy,
        //  frozen cap, accepted nonce). These 52 observations come from fresh
        //  honest proofs after regenerating every production catalog with the
        //  typed model. The profiles cover dense and one-hot fp32, fp64, and
        //  fp128, plus direct, recursive, multi-group, and W2/W4/W8 multi-chunk
        //  adapters. Every proof passed both verifier modes.
        let rows: [(u128, u128, u128, u128, u32); 52] = [
            // dense fp128, nv28
            (12_407_040, 75, 926_141_682, 1_019_618_919, 0),
            (5_796_378, 75, 445_552_700, 477_610_968, 0),
            (3_647_474, 75, 274_451_686, 300_519_261, 0),
            (7_743_642, 75, 586_433_408, 638_603_428, 0),
            // dense fp128 W8R2, nv16
            (35_107_808, 75, 2_626_361_028, 2_919_329_956, 0),
            // dense fp32, nv26
            (40_629_192, 31, 1_239_850_532, 1_384_105_850, 0),
            (57_099_436, 31, 1_768_110_122, 1_934_199_201, 0),
            (40_125_119, 31, 1_244_482_641, 1_366_360_903, 0),
            (33_318_735, 31, 1_049_730_223, 1_135_676_595, 0),
            // dense fp64, nv26
            (13_064_104, 75, 972_766_912, 1_073_283_072, 0),
            (6_572_335, 75, 484_480_549, 542_007_952, 0),
            (38_594_808, 75, 2_880_104_854, 3_219_849_216, 0),
            (32_046_576, 75, 2_358_924_124, 2_618_810_696, 0),
            // one-hot fp128 multi-group direct, nv32
            (13_113_613, 75, 990_397_381, 1_084_015_903, 0),
            (5_790_401, 75, 435_504_807, 477_610_968, 0),
            (3_629_916, 75, 274_360_018, 300_519_261, 0),
            (7_792_001, 75, 597_384_985, 638_603_428, 0),
            // one-hot fp128 multi-group recursive, nv32
            (5_238_923, 75, 389_262_109, 434_679_645, 0),
            (3_643_002, 75, 273_006_474, 300_519_261, 0),
            (7_690_060, 75, 576_160_576, 638_603_428, 0),
            // one-hot fp128 multi-group recursive W8R2, nv32
            (55_820_237, 75, 4_214_079_087, 4_980_033_455, 0),
            (12_646_637, 75, 945_965_213, 1_041_084_580, 0),
            (5_808_394, 75, 431_485_946, 477_610_968, 0),
            (3_633_833, 75, 269_471_043, 300_519_261, 0),
            (7_717_933, 75, 574_249_975, 638_603_428, 0),
            // one-hot fp128 direct, nv36
            (6_623_127, 75, 491_509_521, 547_374_367, 1),
            (6_806_508, 75, 503_824_058, 563_473_613, 0),
            (3_960_821, 75, 302_168_513, 327_351_337, 0),
            (7_780_509, 75, 582_947_255, 638_603_428, 0),
            // one-hot fp128 recursive, nv36
            (5_824_305, 75, 439_023_783, 477_610_968, 0),
            (3_646_029, 75, 271_086_421, 300_519_261, 0),
            (7_700_743, 75, 579_984_205, 638_603_428, 0),
            // one-hot fp128 W2R2, nv32
            (7_711_639, 75, 575_828_653, 633_237_013, 0),
            (7_395_059, 75, 560_656_609, 611_771_352, 0),
            (4_128_110, 75, 306_830_462, 340_767_376, 0),
            (7_767_427, 75, 578_967_153, 638_603_428, 0),
            // one-hot fp128 W4R2, nv32
            (6_962_796, 75, 523_336_068, 574_206_444, 0),
            (7_496_134, 75, 575_021_712, 622_504_182, 0),
            (4_151_889, 75, 308_980_429, 340_767_376, 0),
            (7_770_129, 75, 582_553_175, 638_603_428, 0),
            // one-hot fp128 W8R2, nv32
            (3_689_784, 75, 280_924_824, 303_202_468, 0),
            (5_084_626, 75, 379_634_330, 423_946_814, 0),
            (3_635_868, 75, 272_623_976, 300_519_261, 0),
            (7_648_184, 75, 575_044_430, 638_603_428, 0),
            // one-hot fp32, nv30
            (40_709_103, 31, 1_276_171_391, 1_384_105_850, 0),
            (57_194_754, 31, 1_755_752_816, 1_934_199_201, 0),
            (39_919_435, 31, 1_229_011_205, 1_366_360_903, 0),
            (33_184_890, 31, 1_030_585_156, 1_135_676_595, 0),
            // one-hot fp64, nv30
            (4_789_769, 75, 367_508_391, 397_114_737, 0),
            (21_440_544, 75, 1_607_978_058, 1_781_649_900, 0),
            (39_747_250, 75, 2_915_927_368, 3_262_780_539, 0),
            (31_824_799, 75, 2_357_368_597, 2_618_810_696, 0),
        ];

        for (source, challenge, response, cap, nonce) in rows {
            assert!(response <= cap, "honest response exceeded its frozen cap");
            assert!(
                cap * 100 <= response * 120,
                "frozen cap used more than twenty percent slack"
            );
            assert!(nonce <= 1, "empirical proof required unexpected grinding");

            // Recover the planner's frozen source estimate from its cap. The
            // rounding error is far below these bounds. This pins the observed
            // aggregate model error to about -0.21 to +8.96 percent.
            let implied_source = cap as f64 * (PPM * PPM) as f64
                / (challenge * SOURCE_MODEL_ENVELOPE_PPM * RESPONSE_MEAN_MULTIPLIER_PPM) as f64;
            let source_ratio = implied_source / source as f64;
            assert!((0.997..=1.09).contains(&source_ratio));
        }
    }
}
