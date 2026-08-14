//! Offline honest-prover sizing policies for folded witnesses.
//!
//! These policies select an exact gadget depth for schedule generation. They
//! are not runtime protocol metadata and are never evaluated by the verifier.

use crate::RootSourceProfile;
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;

#[cfg(test)]
use super::onehot_source::SourceClass;
use super::onehot_source::{
    deterministic_convolution_cap, ln_upper, max_log_mgf_upper, projected_source_classes, round_up,
};
use super::{
    fold_witness_linf_cap, num_digits_for_linf_cap, FoldChallengeNorms, FoldWitnessLinfCapConfig,
    FoldWitnessNorms,
};

/// Exact candidate geometry supplied to an offline honest-fold policy.
#[derive(Clone, Copy, Debug)]
pub struct HonestFoldSizingQuery<'a> {
    pub ring_dimension: usize,
    pub num_claims: usize,
    /// Exact live source rings per claim before block padding.
    pub num_live_ring_elements_per_claim: usize,
    pub num_live_blocks: usize,
    /// Source ring positions owned by one challenge block.
    pub num_positions_per_block: usize,
    /// Number of physical response windows emitted for this fold.
    pub num_chunks: usize,
    /// Total coefficients emitted across all physical response windows.
    pub num_fold_coeffs: usize,
    /// Exact root-source representation for this commitment group.
    pub source: RootSourceProfile,
    /// Exact source-plane norms after the selected inner decomposition.
    pub witness_norms: FoldWitnessNorms,
    /// Basis used to decompose the emitted folded response.
    pub log_basis_response: u32,
    pub challenge_config: &'a SparseChallengeConfig,
}

/// One group-owned offline rule for selecting its folded-witness digit depth.
pub trait HonestFoldPolicy {
    /// Return the final digit depth selected by the policy.
    fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError>;
}

/// Distribution-free sizing rule for balanced signed-digit witnesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BalancedSignedDigitFoldPolicy {
    field_bits: u32,
    witness: FoldWitnessNorms,
}

impl BalancedSignedDigitFoldPolicy {
    /// Construct the distribution-free policy.
    #[must_use]
    pub const fn universal(field_bits: u32, witness: FoldWitnessNorms) -> Self {
        Self {
            field_bits,
            witness,
        }
    }

    fn universal_cap(&self, query: HonestFoldSizingQuery<'_>) -> Result<u128, AkitaError> {
        self.witness.validate()?;
        validate_query(query)?;
        // This policy is the frozen pre-chunking baseline. Preserve its
        // logical single-fold geometry even though the query reports every
        // physical response coefficient.
        let logical_fold_coeffs = query.num_fold_coeffs / query.num_chunks;
        let cap_config =
            FoldWitnessLinfCapConfig::for_fold_coeffs(query.challenge_config, logical_fold_coeffs)?;
        let (cap, _) = fold_witness_linf_cap(
            query.num_live_blocks,
            query.num_claims,
            FoldChallengeNorms::new(query.challenge_config),
            query.witness_norms,
            &cap_config,
        )?;
        Ok(cap)
    }

    fn digit_depth_for_cap(&self, cap: u128, log_basis: u32) -> usize {
        num_digits_for_linf_cap(cap, self.field_bits, log_basis)
    }
}

impl HonestFoldPolicy for BalancedSignedDigitFoldPolicy {
    fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError> {
        if query.source != RootSourceProfile::Dense {
            return Err(AkitaError::InvalidSetup(
                "balanced fold policy requires dense source metadata".into(),
            ));
        }
        let cap = self.universal_cap(query)?;
        Ok(self.digit_depth_for_cap(cap, query.log_basis_response))
    }
}

/// Kernel-faithful physical-source policy for unit one-hot root folds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnitOneHotFoldPolicy {
    field_bits: u32,
    extension_degree: usize,
}

/// Canonical logical chunk size of the shipping unit one-hot representation.
pub const DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE: usize = 256;

impl UnitOneHotFoldPolicy {
    /// Construct a unit one-hot policy for one base/extension field profile.
    #[must_use]
    pub const fn new(field_bits: u32, extension_degree: usize) -> Self {
        Self {
            field_bits,
            extension_degree,
        }
    }

    /// Physical tensor-projection width used by this field profile.
    #[must_use]
    pub const fn extension_degree(self) -> usize {
        self.extension_degree
    }

    fn source_chunk_size(&self, query: HonestFoldSizingQuery<'_>) -> Option<usize> {
        query.source.onehot_chunk_size()
    }

    fn physical_extension_degree(&self, query: HonestFoldSizingQuery<'_>) -> Option<usize> {
        let source_len = query
            .num_live_ring_elements_per_claim
            .checked_mul(query.ring_dimension)?;
        if !source_len.is_power_of_two() {
            return None;
        }
        let num_vars = source_len.trailing_zeros() as usize;
        Some(
            if crate::root_tensor_projection_enabled_for_width(
                self.extension_degree,
                query.ring_dimension,
                num_vars,
            ) {
                self.extension_degree
            } else {
                1
            },
        )
    }

    fn independent_challenge_groups(&self, query: HonestFoldSizingQuery<'_>) -> Option<usize> {
        let chunk_size = self.source_chunk_size(query)?;
        let max_groups_per_claim = if chunk_size <= query.ring_dimension
            || query.num_positions_per_block >= chunk_size / query.ring_dimension
        {
            query.num_live_blocks.div_ceil(query.num_chunks)
        } else {
            let rings_per_chunk = chunk_size / query.ring_dimension;
            let mut max_groups = 0usize;
            for window in 0..query.num_chunks {
                let start = window.checked_mul(query.num_live_blocks)? / query.num_chunks;
                let end = (window + 1).checked_mul(query.num_live_blocks)? / query.num_chunks;
                if start == end {
                    continue;
                }
                let final_block_is_partial = end == query.num_live_blocks
                    && !query
                        .num_live_ring_elements_per_claim
                        .is_multiple_of(query.num_positions_per_block);
                if final_block_is_partial {
                    let final_block_start =
                        (query.num_live_blocks - 1).checked_mul(query.num_positions_per_block)?;
                    let final_width = query
                        .num_live_ring_elements_per_claim
                        .checked_sub(final_block_start)?;
                    max_groups = max_groups.max(distinct_source_chunks_for_range(
                        start,
                        end,
                        0,
                        final_width,
                        query.num_positions_per_block,
                        rings_per_chunk,
                    )?);
                    if start + 1 < end && final_width < query.num_positions_per_block {
                        max_groups = max_groups.max(distinct_source_chunks_for_range(
                            start,
                            end - 1,
                            final_width,
                            query.num_positions_per_block,
                            query.num_positions_per_block,
                            rings_per_chunk,
                        )?);
                    }
                } else {
                    max_groups = max_groups.max(distinct_source_chunks_for_range(
                        start,
                        end,
                        0,
                        query.num_positions_per_block,
                        query.num_positions_per_block,
                        rings_per_chunk,
                    )?);
                }
            }
            max_groups
        };
        query.num_claims.checked_mul(max_groups_per_claim)
    }

    #[cfg(test)]
    fn source_classes(&self, query: HonestFoldSizingQuery<'_>) -> Option<Vec<SourceClass>> {
        projected_source_classes(
            query.ring_dimension,
            self.source_chunk_size(query)?,
            self.physical_extension_degree(query)?,
        )
    }

    fn deterministic_cap(&self, query: HonestFoldSizingQuery<'_>) -> Option<u128> {
        let per_group = deterministic_convolution_cap(
            query.ring_dimension,
            self.source_chunk_size(query)?,
            self.physical_extension_degree(query)?,
            query.challenge_config,
        )?;
        per_group.checked_mul(self.independent_challenge_groups(query)? as u128)
    }

    fn exact_threshold(&self, query: HonestFoldSizingQuery<'_>) -> Option<u128> {
        let cfg = query.challenge_config;
        if cfg.weight() > query.ring_dimension || query.ring_dimension == 0 {
            return None;
        }
        let groups = self.independent_challenge_groups(query)?;
        let worst_case = self.deterministic_cap(query)?;
        const MAX_EXACT_F64_INTEGER: usize = 1usize << 52;
        if groups > MAX_EXACT_F64_INTEGER || query.num_fold_coeffs > MAX_EXACT_F64_INTEGER {
            return None;
        }
        // For each exact power-of-two point, solve the Chernoff
        // inequality directly for the smallest admitted integer threshold:
        //
        //   2 N exp(m log M_X(lambda) - lambda t) <= 7/8.
        //
        // Every sampled lambda yields a valid upper bound. Missing the true
        // minimizer can therefore only select a larger threshold.
        let union_factor = round_up(round_up(16.0 * query.num_fold_coeffs as f64) / 7.0);
        let union_log = ln_upper(union_factor)?;
        let mut best = worst_case;
        for exponent in -12i32..=2 {
            let lambda = 2f64.powi(exponent);
            let log_mgf = max_log_mgf_upper(
                query.ring_dimension,
                self.source_chunk_size(query)?,
                self.physical_extension_degree(query)?,
                cfg,
                exponent,
            )?;
            let numerator = round_up(round_up(groups as f64 * log_mgf) + union_log);
            let required = round_up(numerator / lambda).ceil();
            if required.is_finite() && required > 0.0 {
                best = best.min(required as u128);
            }
        }
        Some(best.max(1))
    }

    fn digit_depth_for_cap(&self, cap: u128, log_basis: u32) -> usize {
        num_digits_for_linf_cap(cap, self.field_bits, log_basis)
    }
}

/// Maximum number of distinct logical K-chunks reached by one response row.
///
/// This branch is used only for `P < K/D`, so as `q` varies over the supplied
/// interval the starting residue crosses at most one logical chunk boundary.
fn distinct_source_chunks_for_range(
    block_start: usize,
    block_end: usize,
    q_start: usize,
    q_end: usize,
    positions_per_block: usize,
    rings_per_chunk: usize,
) -> Option<usize> {
    if block_start >= block_end || q_start >= q_end {
        return Some(0);
    }
    let block_delta = (block_end - block_start - 1).checked_mul(positions_per_block)?;
    let quotient = block_delta / rings_per_chunk;
    let remainder = block_delta % rings_per_chunk;
    let crosses_extra_chunk = if remainder == 0 {
        false
    } else {
        let first_ring = block_start
            .checked_mul(positions_per_block)?
            .checked_add(q_start)?;
        let first_residue = first_ring % rings_per_chunk;
        let residue_interval_len = q_end - q_start;
        let residue_end = first_residue.checked_add(residue_interval_len)?;
        residue_end > rings_per_chunk - remainder
    };
    quotient
        .checked_add(1)?
        .checked_add(usize::from(crosses_extra_chunk))
}

impl HonestFoldPolicy for UnitOneHotFoldPolicy {
    fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError> {
        validate_query(query)?;
        if !matches!(query.source, RootSourceProfile::UnitOneHot { .. }) {
            return Err(AkitaError::InvalidSetup(
                "unit one-hot fold policy requires unit one-hot source metadata".into(),
            ));
        }
        let expected_fold_coeffs = query
            .num_chunks
            .checked_mul(query.num_positions_per_block)
            .and_then(|count| count.checked_mul(query.ring_dimension))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("unit one-hot response size overflow".into())
            })?;
        if query.num_fold_coeffs != expected_fold_coeffs {
            return Err(AkitaError::InvalidSetup(
                "unit one-hot response union must count each physical window, row, and ring coefficient exactly once"
                    .into(),
            ));
        }
        let cap = self
            .exact_threshold(query)
            .or_else(|| self.deterministic_cap(query))
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "unit one-hot source geometry is unsupported or overflows".into(),
                )
            })?
            .max(1);
        Ok(self.digit_depth_for_cap(cap, query.log_basis_response))
    }
}

/// Cloneable offline policy value used by generated-family enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HonestFoldPolicySpec {
    BalancedSignedDigit(BalancedSignedDigitFoldPolicy),
    UnitOneHot(UnitOneHotFoldPolicy),
}

impl HonestFoldPolicySpec {
    /// Source-plane norms for one selected A decomposition basis.
    ///
    /// Balanced sources follow the candidate basis. Unit one-hot sources keep
    /// their profile-owned sparse norm; the planner canonicalizes their
    /// already-single-digit representation without a basis sweep.
    #[must_use]
    pub fn witness_norms_for_inner_basis(
        self,
        log_basis_inner: u32,
        ring_dimension: usize,
        source: RootSourceProfile,
        num_vars: usize,
    ) -> FoldWitnessNorms {
        match self {
            Self::BalancedSignedDigit(_) => {
                FoldWitnessNorms::bounded(log_basis_inner, ring_dimension)
            }
            Self::UnitOneHot(policy) => {
                let classes = source
                    .onehot_chunk_size()
                    .and_then(|chunk_size| {
                        let extension_degree = if crate::root_tensor_projection_enabled_for_width(
                            policy.extension_degree,
                            ring_dimension,
                            num_vars,
                        ) {
                            policy.extension_degree
                        } else {
                            1
                        };
                        projected_source_classes(ring_dimension, chunk_size, extension_degree)
                    })
                    .unwrap_or_default();
                let infinity_norm = classes
                    .iter()
                    .map(|class| class.infinity_norm())
                    .max()
                    .unwrap_or(1);
                let l1_norm = classes
                    .iter()
                    .filter_map(|class| class.l1_norm())
                    .max()
                    .unwrap_or(infinity_norm);
                FoldWitnessNorms::new(infinity_norm, l1_norm)
            }
        }
    }
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
        || query.num_live_ring_elements_per_claim == 0
        || query.num_live_blocks == 0
        || query.num_positions_per_block == 0
        || query.num_chunks == 0
        || query.num_fold_coeffs == 0
        || query.log_basis_response == 0
    {
        return Err(AkitaError::InvalidSetup(
            "honest fold sizing requires positive geometry and basis".to_string(),
        ));
    }
    if query.num_live_blocks
        != query
            .num_live_ring_elements_per_claim
            .div_ceil(query.num_positions_per_block)
    {
        return Err(AkitaError::InvalidSetup(
            "honest fold block geometry is inconsistent with the live source length".into(),
        ));
    }
    query.witness_norms.validate()?;
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

#[cfg(any())]
mod legacy_tests {
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
            witness_norms: FoldWitnessNorms::bounded(3, 64),
            log_basis_response: 3,
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
    fn balanced_policy_uses_the_universal_digit_depth() {
        let challenge = d64_challenge();
        let query = query(&challenge);
        let witness = FoldWitnessNorms::bounded(3, 64);
        let policy = BalancedSignedDigitFoldPolicy::universal(128, witness);
        let actual = policy.num_digits_fold(query).expect("balanced policy");
        let cap_config =
            FoldWitnessLinfCapConfig::for_fold_coeffs(&challenge, query.num_fold_coeffs)
                .expect("cap config");
        let expected_cap = fold_witness_linf_cap(
            query.num_live_blocks,
            query.num_claims,
            FoldChallengeNorms::new(&challenge),
            witness,
            &cap_config,
        )
        .expect("universal cap")
        .0;
        let expected = num_digits_for_linf_cap(expected_cap, 128, query.log_basis_response);
        assert_eq!(actual, expected);
    }

    #[test]
    fn unit_one_hot_never_exceeds_the_universal_bound() {
        let challenge = d64_challenge();
        let legacy_witness = FoldWitnessNorms::new(1, 4);
        let one_hot = UnitOneHotFoldPolicy::new(128, legacy_witness);
        let legacy = BalancedSignedDigitFoldPolicy::universal(128, legacy_witness);
        let flat = query(&challenge);
        let exact_digits = one_hot.num_digits_fold(flat).expect("one-hot policy");
        let legacy_digits = legacy.num_digits_fold(flat).expect("legacy policy");
        assert!(exact_digits <= legacy_digits);
        let deterministic_cap = one_hot
            .contributions_per_emitted_coordinate(flat)
            .and_then(|count| count.checked_mul(challenge.infinity_norm() as usize))
            .unwrap() as u128;
        assert!(one_hot.exact_threshold(flat).unwrap() <= deterministic_cap);
    }

    #[test]
    fn wide_one_hot_blocks_price_every_packed_unit_entry() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(512).unwrap();
        let one_hot = UnitOneHotFoldPolicy::new(32, FoldWitnessNorms::new(1, 4));
        let query = HonestFoldSizingQuery {
            ring_dimension: 512,
            num_claims: 1,
            num_live_blocks: 512,
            num_chunks: 1,
            num_fold_coeffs: 4_096 * 512,
            witness_norms: FoldWitnessNorms::new(1, 4),
            log_basis_response: 3,
            challenge_config: &challenge,
        };

        assert_eq!(
            one_hot.contributions_per_emitted_coordinate(query),
            Some(8_192)
        );
        assert!(one_hot.exact_threshold(query).unwrap() > 31);
        assert!(one_hot.num_digits_fold(query).unwrap() >= 3);
    }

    #[test]
    fn unit_one_hot_tightens_at_least_one_supported_geometry() {
        let challenge =
            SparseChallengeConfig::production_for_ring_dim(256).expect("D256 production challenge");
        let legacy_witness = FoldWitnessNorms::new(1, 4);
        let one_hot = UnitOneHotFoldPolicy::new(128, legacy_witness);
        let legacy = BalancedSignedDigitFoldPolicy::universal(128, legacy_witness);
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
                            witness_norms: legacy_witness,
                            log_basis_response: log_basis,
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
        let one_hot = UnitOneHotFoldPolicy::new(128, FoldWitnessNorms::new(1, 4));
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
                witness_norms: FoldWitnessNorms::new(1, 4),
                log_basis_response: 3,
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
            witness_norms: FoldWitnessNorms::new(1, 4),
            log_basis_response: 3,
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
        let one_hot = UnitOneHotFoldPolicy::new(128, FoldWitnessNorms::new(1, 4));
        let uneven = HonestFoldSizingQuery {
            ring_dimension: 64,
            num_claims: 2,
            num_live_blocks: 10,
            num_chunks: 4,
            num_fold_coeffs: 512,
            witness_norms: FoldWitnessNorms::new(1, 4),
            log_basis_response: 3,
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
        let one_hot = UnitOneHotFoldPolicy::new(128, FoldWitnessNorms::new(1, 4));
        let query = HonestFoldSizingQuery {
            ring_dimension: 64,
            num_claims: 1,
            num_live_blocks: 4,
            num_chunks: 8,
            num_fold_coeffs: 4_096,
            witness_norms: FoldWitnessNorms::new(1, 4),
            log_basis_response: 3,
            challenge_config: &challenge,
        };

        one_hot
            .num_digits_fold(query)
            .expect("empty physical windows have valid honest-fold sizing");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query<'a>(
        challenge: &'a SparseChallengeConfig,
        source: RootSourceProfile,
        ring_dimension: usize,
        blocks: usize,
        positions_per_block: usize,
    ) -> HonestFoldSizingQuery<'a> {
        HonestFoldSizingQuery {
            ring_dimension,
            num_claims: 1,
            num_live_ring_elements_per_claim: blocks * positions_per_block,
            num_live_blocks: blocks,
            num_positions_per_block: positions_per_block,
            num_chunks: 1,
            num_fold_coeffs: positions_per_block * ring_dimension,
            source,
            witness_norms: FoldWitnessNorms::new(1, 1),
            log_basis_response: 3,
            challenge_config: challenge,
        }
    }

    #[test]
    fn fp128_nv36_drops_the_spurious_positions_per_block_multiplier() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(256).unwrap();
        let policy = UnitOneHotFoldPolicy::new(128, 1);
        let query = query(
            &challenge,
            RootSourceProfile::UnitOneHot { chunk_size: 256 },
            256,
            4_096,
            65_536,
        );
        assert!(policy.exact_threshold(query).unwrap() <= 219);
        assert_eq!(policy.num_digits_fold(query).unwrap(), 3);
    }

    #[test]
    fn fp64_psi_expansion_crosses_the_two_digit_positive_endpoint() {
        let d256_challenge = SparseChallengeConfig::production_for_ring_dim(256).unwrap();
        let d512_challenge = SparseChallengeConfig::production_for_ring_dim(512).unwrap();
        let policy = UnitOneHotFoldPolicy::new(64, 2);
        for query in [
            query(
                &d256_challenge,
                RootSourceProfile::UnitOneHot { chunk_size: 256 },
                256,
                256,
                4_096,
            ),
            query(
                &d512_challenge,
                RootSourceProfile::UnitOneHot { chunk_size: 256 },
                512,
                256,
                8_192,
            ),
        ] {
            assert!(policy.exact_threshold(query).unwrap() > 27);
            assert_eq!(policy.num_digits_fold(query).unwrap(), 3);
        }
    }

    #[test]
    fn every_k_vs_d_branch_uses_the_exact_group_source() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(256).unwrap();
        let direct = UnitOneHotFoldPolicy::new(128, 1);
        let projected = UnitOneHotFoldPolicy::new(64, 2);
        for chunk_size in [16, 256, 512] {
            let source = RootSourceProfile::UnitOneHot { chunk_size };
            let query = query(&challenge, source, 256, 8, 4);
            assert!(direct.exact_threshold(query).is_some());
            assert!(projected.exact_threshold(query).is_some());
        }
    }

    #[test]
    fn fp32_small_block_row_does_not_use_ceil_p_over_k() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(512).unwrap();
        let policy = UnitOneHotFoldPolicy::new(32, 4);
        let query = query(
            &challenge,
            RootSourceProfile::UnitOneHot { chunk_size: 256 },
            512,
            8,
            4,
        );
        let source_classes = policy.source_classes(query).unwrap();
        assert!(source_classes.iter().any(|class| class.magnitude_one == 4));
        assert_eq!(policy.independent_challenge_groups(query), Some(8));
    }

    #[test]
    fn k16_multiple_ones_per_ring_are_charged_after_psi() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(512).unwrap();
        let policy = UnitOneHotFoldPolicy::new(32, 4);
        let query = query(
            &challenge,
            RootSourceProfile::UnitOneHot { chunk_size: 16 },
            512,
            8,
            4,
        );
        let classes = policy.source_classes(query).unwrap();
        assert!(classes.iter().any(|class| class.magnitude_one == 64));
        assert!(classes.iter().any(|class| class.magnitude_two == 16));
        assert!(policy.deterministic_cap(query).unwrap() >= 8 * 32);
    }

    #[test]
    fn unit_policy_rejects_dense_metadata() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(256).unwrap();
        let policy = UnitOneHotFoldPolicy::new(128, 1);
        let dense = query(&challenge, RootSourceProfile::Dense, 256, 8, 4);
        assert!(policy.num_digits_fold(dense).is_err());
    }

    #[test]
    fn k_greater_than_d_counts_distinct_chunks_inside_each_window() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let policy = UnitOneHotFoldPolicy::new(128, 1);
        let query = HonestFoldSizingQuery {
            ring_dimension: 64,
            num_claims: 1,
            num_live_ring_elements_per_claim: 8,
            num_live_blocks: 8,
            num_positions_per_block: 1,
            num_chunks: 2,
            num_fold_coeffs: 2 * 64,
            source: RootSourceProfile::UnitOneHot { chunk_size: 256 },
            witness_norms: FoldWitnessNorms::new(1, 1),
            log_basis_response: 3,
            challenge_config: &challenge,
        };
        assert_eq!(policy.independent_challenge_groups(query), Some(1));
        assert!(policy.num_digits_fold(query).is_ok());
    }

    #[test]
    fn k_greater_than_d_handles_nondyadic_blocks_and_a_partial_final_block() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let policy = UnitOneHotFoldPolicy::new(128, 1);
        let query = HonestFoldSizingQuery {
            ring_dimension: 64,
            num_claims: 1,
            num_live_ring_elements_per_claim: 8,
            num_live_blocks: 3,
            num_positions_per_block: 3,
            num_chunks: 1,
            num_fold_coeffs: 3 * 64,
            source: RootSourceProfile::UnitOneHot { chunk_size: 256 },
            witness_norms: FoldWitnessNorms::new(1, 1),
            log_basis_response: 3,
            challenge_config: &challenge,
        };
        assert_eq!(policy.independent_challenge_groups(query), Some(2));
        assert!(policy.num_digits_fold(query).is_ok());
    }
}
