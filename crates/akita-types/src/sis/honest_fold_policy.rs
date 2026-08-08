//! Offline honest-prover sizing policies for folded witnesses.
//!
//! These policies select an exact gadget depth for schedule generation. They
//! are not runtime protocol metadata and are never evaluated by the verifier.

use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;

use super::{
    decomposition_digits::balanced_digit_max, fold_witness_unsnapped_linf_cap,
    num_digits_for_bound, FoldChallengeNorms, FoldWitnessLinfCapConfig, FoldWitnessNorms,
    FOLD_LINF_FP32_SNAP_MIN_TSTAR_RETAIN_DEN, FOLD_LINF_FP32_SNAP_MIN_TSTAR_RETAIN_NUM,
    FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_DEN, FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_NUM,
};

/// Exact candidate geometry supplied to an offline honest-fold policy.
#[derive(Clone, Copy, Debug)]
pub struct HonestFoldSizingQuery<'a> {
    pub ring_dimension: usize,
    pub num_claims: usize,
    pub num_live_blocks: usize,
    /// Number of physical response windows emitted for this fold.
    pub num_chunks: usize,
    /// Total coefficients emitted across all physical response windows.
    pub num_fold_coeffs: usize,
    pub log_basis: u32,
    pub challenge_config: &'a SparseChallengeConfig,
}

/// One group-owned offline rule for selecting its folded-witness digit depth.
pub trait HonestFoldPolicy {
    /// Return the final digit depth, including any policy-local calibration.
    fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError>;
}

/// Explicit calibration for snapping an analytic cap to a smaller digit depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DigitSnapCalibration {
    pub retain_num: u32,
    pub retain_den: u32,
}

impl DigitSnapCalibration {
    pub const NONE: Self = Self {
        retain_num: 1,
        retain_den: 1,
    };

    /// Build a validated retained-cap ratio.
    pub fn new(retain_num: u32, retain_den: u32) -> Result<Self, AkitaError> {
        let calibration = Self {
            retain_num,
            retain_den,
        };
        calibration.validate()?;
        Ok(calibration)
    }

    fn validate(self) -> Result<(), AkitaError> {
        if self.retain_num == 0 || self.retain_den == 0 || self.retain_num > self.retain_den {
            return Err(AkitaError::InvalidSetup(
                "digit snap calibration requires 0 < numerator <= denominator".to_string(),
            ));
        }
        Ok(())
    }
}

/// Preserved average-case sizing rule for balanced signed-digit witnesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BalancedSignedDigitFoldPolicy {
    field_bits: u32,
    witness: FoldWitnessNorms,
    snap: DigitSnapCalibration,
}

impl BalancedSignedDigitFoldPolicy {
    /// Construct the historical policy, including its field-specific snap.
    #[must_use]
    pub const fn preserving_existing_behavior(field_bits: u32, witness: FoldWitnessNorms) -> Self {
        let snap = if field_bits == 32 {
            DigitSnapCalibration {
                retain_num: FOLD_LINF_FP32_SNAP_MIN_TSTAR_RETAIN_NUM,
                retain_den: FOLD_LINF_FP32_SNAP_MIN_TSTAR_RETAIN_DEN,
            }
        } else {
            DigitSnapCalibration {
                retain_num: FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_NUM,
                retain_den: FOLD_LINF_SNAP_MIN_TSTAR_RETAIN_DEN,
            }
        };
        Self {
            field_bits,
            witness,
            snap,
        }
    }

    fn unsnapped_cap(&self, query: HonestFoldSizingQuery<'_>) -> Result<(u128, u128), AkitaError> {
        self.witness.validate()?;
        self.snap.validate()?;
        validate_query(query)?;
        // This policy is the frozen pre-chunking baseline. Preserve its
        // logical single-fold geometry even though the query reports every
        // physical response coefficient.
        let logical_fold_coeffs = query.num_fold_coeffs / query.num_chunks;
        let cap_config =
            FoldWitnessLinfCapConfig::for_fold_coeffs(query.challenge_config, logical_fold_coeffs)?;
        let (cap, tail_cap) = fold_witness_unsnapped_linf_cap(
            query.num_live_blocks,
            query.num_claims,
            FoldChallengeNorms::new(query.challenge_config),
            self.witness,
            &cap_config,
        )?;
        Ok((cap, tail_cap))
    }

    fn digit_depth_for_cap(&self, cap: u128, log_basis: u32) -> usize {
        let log_cap = (128 - cap.leading_zeros()).saturating_add(1);
        num_digits_for_bound(log_cap, self.field_bits, log_basis)
    }
}

impl HonestFoldPolicy for BalancedSignedDigitFoldPolicy {
    fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError> {
        let (cap, tail_cap) = self.unsnapped_cap(query)?;
        let mut digits = self.digit_depth_for_cap(cap, query.log_basis);
        let floor = tail_cap.saturating_mul(u128::from(self.snap.retain_num))
            / u128::from(self.snap.retain_den);
        let floor = floor.max(1);
        while digits > 1 && balanced_digit_max(query.log_basis, digits - 1) >= floor {
            digits -= 1;
        }
        Ok(digits)
    }
}

/// Exact one-coordinate MGF policy for unit one-hot logical blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnitOneHotFoldPolicy {
    field_bits: u32,
    snap: DigitSnapCalibration,
    legacy_fallback: BalancedSignedDigitFoldPolicy,
}

impl UnitOneHotFoldPolicy {
    /// Construct the shipping no-snap policy with its historical dominance
    /// guard. Configurations call this only after establishing the unit
    /// one-hot source condition.
    #[must_use]
    pub const fn preserving_existing_behavior(
        field_bits: u32,
        legacy_witness: FoldWitnessNorms,
    ) -> Self {
        Self {
            field_bits,
            snap: DigitSnapCalibration::NONE,
            legacy_fallback: BalancedSignedDigitFoldPolicy::preserving_existing_behavior(
                field_bits,
                legacy_witness,
            ),
        }
    }

    /// Construct a one-hot policy with an explicit calibration and the exact
    /// pre-cutover policy used by its dominance guard.
    pub fn new(
        field_bits: u32,
        snap: DigitSnapCalibration,
        legacy_witness: FoldWitnessNorms,
    ) -> Result<Self, AkitaError> {
        snap.validate()?;
        legacy_witness.validate()?;
        Ok(Self {
            field_bits,
            snap,
            legacy_fallback: BalancedSignedDigitFoldPolicy::preserving_existing_behavior(
                field_bits,
                legacy_witness,
            ),
        })
    }

    fn exact_threshold(&self, query: HonestFoldSizingQuery<'_>) -> Option<u128> {
        let cfg = query.challenge_config;
        if cfg.weight() > query.ring_dimension || query.ring_dimension == 0 {
            return None;
        }
        let live_blocks_per_chunk = query.num_live_blocks.div_ceil(query.num_chunks);
        let contributions = query.num_claims.checked_mul(live_blocks_per_chunk)?;
        let worst_case = contributions.checked_mul(cfg.infinity_norm() as usize)? as u128;
        const MAX_EXACT_F64_INTEGER: usize = 1usize << 52;
        if contributions > MAX_EXACT_F64_INTEGER || query.num_fold_coeffs > MAX_EXACT_F64_INTEGER {
            return None;
        }
        let coordinates = query.num_fold_coeffs as f64;
        let contributions = contributions as f64;
        let dimension = query.ring_dimension as f64;

        // For each point in a fixed logarithmic grid, solve the Chernoff
        // inequality directly for the smallest admitted integer threshold:
        //
        //   2 N exp(m log M_X(lambda) - lambda t) <= 7/8.
        //
        // Every sampled lambda yields a valid upper bound. Missing the true
        // minimizer can therefore only select a larger threshold.
        let union_factor = round_up(round_up(16.0 * coordinates) / 7.0);
        let union_log = inflate_up(union_factor.ln());
        let mut best = worst_case;
        for exponent in -18i32..=18 {
            let lambda = 2f64.powf(f64::from(exponent) / 2.0);
            let mgf = unit_one_hot_mgf_upper(cfg, dimension, lambda);
            let log_mgf = inflate_up(mgf.ln());
            let numerator = round_up(round_up(contributions * log_mgf) + union_log);
            let required = round_up(numerator / round_down(lambda)).ceil();
            if required.is_finite() && required > 0.0 {
                best = best.min(required as u128);
            }
        }
        Some(best.max(1))
    }

    fn digit_depth_for_cap(&self, cap: u128, log_basis: u32) -> usize {
        let log_cap = (128 - cap.leading_zeros()).saturating_add(1);
        num_digits_for_bound(log_cap, self.field_bits, log_basis)
    }
}

impl HonestFoldPolicy for UnitOneHotFoldPolicy {
    fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError> {
        validate_query(query)?;
        self.snap.validate()?;
        let legacy = self.legacy_fallback.num_digits_fold(query)?;
        let Some(mut cap) = self.exact_threshold(query) else {
            return Ok(legacy);
        };
        let live_blocks_per_chunk = query.num_live_blocks.div_ceil(query.num_chunks);
        let challenge = FoldChallengeNorms::new(query.challenge_config);
        let worst_case = challenge
            .l1_norm
            .checked_mul(query.num_claims as u128)
            .and_then(|value| value.checked_mul(live_blocks_per_chunk as u128))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("unit one-hot worst-case cap overflow".into())
            })?;
        cap = cap.min(worst_case).max(1);
        let mut digits = self.digit_depth_for_cap(cap, query.log_basis);
        let floor =
            cap.saturating_mul(u128::from(self.snap.retain_num)) / u128::from(self.snap.retain_den);
        while digits > 1 && balanced_digit_max(query.log_basis, digits - 1) >= floor.max(1) {
            digits -= 1;
        }
        Ok(digits.min(legacy))
    }
}

/// Cloneable offline policy value used by generated-family enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HonestFoldPolicySpec {
    BalancedSignedDigit(BalancedSignedDigitFoldPolicy),
    UnitOneHot(UnitOneHotFoldPolicy),
}

impl HonestFoldPolicy for HonestFoldPolicySpec {
    fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError> {
        match self {
            Self::BalancedSignedDigit(policy) => policy.num_digits_fold(query),
            Self::UnitOneHot(policy) => policy.num_digits_fold(query),
        }
    }
}

fn validate_query(query: HonestFoldSizingQuery<'_>) -> Result<(), AkitaError> {
    if query.ring_dimension == 0
        || query.num_claims == 0
        || query.num_live_blocks == 0
        || query.num_chunks == 0
        || query.num_fold_coeffs == 0
        || query.log_basis == 0
    {
        return Err(AkitaError::InvalidSetup(
            "honest fold sizing requires positive geometry and basis".to_string(),
        ));
    }
    if !query.num_fold_coeffs.is_multiple_of(query.num_chunks) {
        return Err(AkitaError::InvalidSetup(
            "honest fold coefficient count must cover equally sized chunk responses".to_string(),
        ));
    }
    query
        .challenge_config
        .validate_for_ring_dim(query.ring_dimension)
        .map_err(|message| AkitaError::InvalidSetup(message.to_string()))
}

fn unit_one_hot_mgf_upper(challenge: &SparseChallengeConfig, dimension: f64, lambda: f64) -> f64 {
    let cosh_one = inflate_up(lambda.cosh());
    let cosh_two = inflate_up((2.0 * lambda).cosh());
    let pm1_term = round_up((challenge.count_pm1 as f64 / dimension) * round_up(cosh_one - 1.0));
    let pm2_term = round_up((challenge.count_pm2 as f64 / dimension) * round_up(cosh_two - 1.0));
    round_up(round_up(1.0 + pm1_term) + pm2_term)
}

fn round_up(value: f64) -> f64 {
    if !value.is_finite() || value == f64::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value >= 0.0 { bits + 1 } else { bits - 1 })
}

fn round_down(value: f64) -> f64 {
    if !value.is_finite() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

fn inflate_up(mut value: f64) -> f64 {
    // Leave ample room for platform libm error before composing the bound.
    // This affects only offline sizing, and the legacy dominance guard remains
    // the final ceiling on the selected digit depth.
    for _ in 0..16 {
        value = round_up(value);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::{D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT};

    fn d64_challenge() -> SparseChallengeConfig {
        SparseChallengeConfig::production_for_ring_dim(64).expect("D64 production challenge")
    }

    fn query<'a>(challenge: &'a SparseChallengeConfig) -> HonestFoldSizingQuery<'a> {
        HonestFoldSizingQuery {
            ring_dimension: 64,
            num_claims: 4,
            num_live_blocks: 16,
            num_chunks: 1,
            num_fold_coeffs: 4_096,
            log_basis: 3,
            challenge_config: challenge,
        }
    }

    #[test]
    fn d64_one_hot_mgf_uses_31_and_10_masses() {
        let challenge = d64_challenge();
        assert_eq!(challenge.count_pm1, D64_PRODUCTION_PM1_COUNT);
        assert_eq!(challenge.count_pm2, D64_PRODUCTION_PM2_COUNT);
        assert_eq!((challenge.count_pm1, challenge.count_pm2), (31, 10));

        let lambda = 0.75f64;
        let implemented = 1.0
            + (challenge.count_pm1 as f64 / 64.0) * (lambda.cosh() - 1.0)
            + (challenge.count_pm2 as f64 / 64.0) * ((2.0 * lambda).cosh() - 1.0);
        let specified = 1.0
            + (31.0 / 64.0) * (lambda.cosh() - 1.0)
            + (10.0 / 64.0) * ((2.0 * lambda).cosh() - 1.0);
        assert_eq!(implemented, specified);
    }

    fn directly_enumerated_one_coordinate_mgf(
        challenge: &SparseChallengeConfig,
        dimension: usize,
        lambda: f64,
    ) -> f64 {
        let zero_count = dimension - challenge.weight();
        let pm1_mass = challenge.count_pm1 as f64 * (lambda.exp() + (-lambda).exp()) / 2.0;
        let pm2_mass =
            challenge.count_pm2 as f64 * ((2.0 * lambda).exp() + (-2.0 * lambda).exp()) / 2.0;
        (zero_count as f64 + pm1_mass + pm2_mass) / dimension as f64
    }

    #[test]
    fn one_hot_mgf_matches_direct_law_and_uses_each_dimensions_counts() {
        let lambda = 0.625;
        for dimension in [64, 128, 256] {
            let challenge = SparseChallengeConfig::production_for_ring_dim(dimension)
                .expect("production challenge");
            let direct = directly_enumerated_one_coordinate_mgf(&challenge, dimension, lambda);
            let formula = 1.0
                + (challenge.count_pm1 as f64 / dimension as f64) * (lambda.cosh() - 1.0)
                + (challenge.count_pm2 as f64 / dimension as f64) * ((2.0 * lambda).cosh() - 1.0);
            assert!((formula - direct).abs() <= 8.0 * f64::EPSILON * direct);
            assert!(unit_one_hot_mgf_upper(&challenge, dimension as f64, lambda) >= direct);
        }

        let d128 = SparseChallengeConfig::production_for_ring_dim(128).unwrap();
        assert_eq!((d128.count_pm1, d128.count_pm2), (31, 0));
        assert_ne!(
            (d128.count_pm1, d128.count_pm2),
            (D64_PRODUCTION_PM1_COUNT, D64_PRODUCTION_PM2_COUNT)
        );
    }

    #[test]
    fn balanced_policy_preserves_legacy_digit_depth() {
        let challenge = d64_challenge();
        let query = query(&challenge);
        let witness = FoldWitnessNorms::bounded(3, 64);
        let policy = BalancedSignedDigitFoldPolicy::preserving_existing_behavior(128, witness);
        let actual = policy.num_digits_fold(query).expect("balanced policy");
        let cap_config =
            FoldWitnessLinfCapConfig::for_fold_coeffs(&challenge, query.num_fold_coeffs)
                .expect("cap config");
        let expected = super::super::fold_witness_digit_plan(
            query.num_live_blocks,
            query.num_claims,
            128,
            query.log_basis,
            FoldChallengeNorms::new(&challenge),
            witness,
            &cap_config,
        )
        .expect("legacy digit plan")
        .0;
        assert_eq!(actual, expected);
    }

    #[test]
    fn unit_one_hot_never_exceeds_legacy() {
        let challenge = d64_challenge();
        let legacy_witness = FoldWitnessNorms::new(1, 4);
        let one_hot = UnitOneHotFoldPolicy::preserving_existing_behavior(128, legacy_witness);
        let legacy =
            BalancedSignedDigitFoldPolicy::preserving_existing_behavior(128, legacy_witness);
        let flat = query(&challenge);
        let exact_digits = one_hot.num_digits_fold(flat).expect("one-hot policy");
        let legacy_digits = legacy.num_digits_fold(flat).expect("legacy policy");
        assert!(exact_digits <= legacy_digits);
        let deterministic_cap = flat
            .num_claims
            .checked_mul(flat.num_live_blocks)
            .and_then(|count| count.checked_mul(challenge.infinity_norm() as usize))
            .unwrap() as u128;
        assert!(one_hot.exact_threshold(flat).unwrap() <= deterministic_cap);
    }

    #[test]
    fn unit_one_hot_tightens_at_least_one_supported_geometry() {
        let challenge =
            SparseChallengeConfig::production_for_ring_dim(256).expect("D256 production challenge");
        let legacy_witness = FoldWitnessNorms::new(1, 4);
        let one_hot = UnitOneHotFoldPolicy::preserving_existing_behavior(128, legacy_witness);
        let legacy =
            BalancedSignedDigitFoldPolicy::preserving_existing_behavior(128, legacy_witness);
        let mut tightened = None;

        'geometry: for num_claims in [1, 2, 4] {
            for num_live_blocks in [4, 16, 64, 256, 1_024, 4_096] {
                for num_fold_coeffs in [256, 1_024, 4_096, 16_384] {
                    for log_basis in 1..=8 {
                        let query = HonestFoldSizingQuery {
                            ring_dimension: 256,
                            num_claims,
                            num_live_blocks,
                            num_chunks: 1,
                            num_fold_coeffs,
                            log_basis,
                            challenge_config: &challenge,
                        };
                        let exact_digits = one_hot.num_digits_fold(query).unwrap();
                        let legacy_digits = legacy.num_digits_fold(query).unwrap();
                        if exact_digits < legacy_digits {
                            tightened = Some((query, exact_digits, legacy_digits));
                            break 'geometry;
                        }
                    }
                }
            }
        }

        assert!(
            tightened.is_some(),
            "supported grid must include a tighter row"
        );
    }

    #[test]
    fn chunked_query_uses_physical_emitted_geometry() {
        let challenge = d64_challenge();
        let one_hot =
            UnitOneHotFoldPolicy::preserving_existing_behavior(128, FoldWitnessNorms::new(1, 4));
        for (
            _label,
            num_chunks,
            logical_num_live_blocks,
            logical_num_fold_coeffs,
            physical_num_fold_coeffs,
            expected_threshold,
        ) in [
            ("W2", 2, 16, 256, 512, 11),
            ("W4", 4, 32, 256, 1_024, 12),
            ("W8", 8, 64, 512, 4_096, 13),
        ] {
            let physical = HonestFoldSizingQuery {
                ring_dimension: 64,
                num_claims: 1,
                num_live_blocks: logical_num_live_blocks,
                num_chunks,
                num_fold_coeffs: physical_num_fold_coeffs,
                log_basis: 3,
                challenge_config: &challenge,
            };
            assert_eq!(
                physical.num_fold_coeffs,
                logical_num_fold_coeffs * num_chunks
            );
            assert_eq!(
                one_hot.exact_threshold(physical).expect("threshold"),
                expected_threshold
            );
            assert_eq!(one_hot.num_digits_fold(physical).expect("digit depth"), 2);
        }

        let old_w8_logical = HonestFoldSizingQuery {
            ring_dimension: 64,
            num_claims: 1,
            num_live_blocks: 64,
            num_chunks: 1,
            num_fold_coeffs: 512,
            log_basis: 3,
            challenge_config: &challenge,
        };
        assert_eq!(
            one_hot
                .exact_threshold(old_w8_logical)
                .expect("old threshold"),
            32
        );
        assert_eq!(
            one_hot
                .num_digits_fold(old_w8_logical)
                .expect("old digit depth"),
            3
        );
    }

    #[test]
    fn chunked_query_uses_largest_uneven_window() {
        let challenge = d64_challenge();
        let one_hot =
            UnitOneHotFoldPolicy::preserving_existing_behavior(128, FoldWitnessNorms::new(1, 4));
        let uneven = HonestFoldSizingQuery {
            ring_dimension: 64,
            num_claims: 2,
            num_live_blocks: 10,
            num_chunks: 4,
            num_fold_coeffs: 512,
            log_basis: 3,
            challenge_config: &challenge,
        };
        let largest_window = HonestFoldSizingQuery {
            num_live_blocks: 3,
            num_chunks: 1,
            ..uneven
        };

        assert_eq!(
            one_hot.exact_threshold(uneven),
            one_hot.exact_threshold(largest_window)
        );
    }

    #[test]
    fn chunked_query_accepts_empty_physical_windows() {
        let challenge = d64_challenge();
        let one_hot =
            UnitOneHotFoldPolicy::preserving_existing_behavior(128, FoldWitnessNorms::new(1, 4));
        let query = HonestFoldSizingQuery {
            ring_dimension: 64,
            num_claims: 1,
            num_live_blocks: 4,
            num_chunks: 8,
            num_fold_coeffs: 4_096,
            log_basis: 3,
            challenge_config: &challenge,
        };

        one_hot
            .num_digits_fold(query)
            .expect("empty physical windows have valid honest-fold sizing");
    }

    #[test]
    fn shipping_one_hot_policy_has_no_snap() {
        let policy =
            UnitOneHotFoldPolicy::preserving_existing_behavior(128, FoldWitnessNorms::new(1, 4));
        assert_eq!(policy.snap, DigitSnapCalibration::NONE);
    }
}
