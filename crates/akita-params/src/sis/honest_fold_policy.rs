//! Offline honest-prover sizing policies for folded witnesses.
//!
//! These policies select an exact gadget depth for schedule generation. They
//! are not runtime protocol metadata and are never evaluated by the verifier.

use akita_error::AkitaError;
use jolt_crypto::challenge::SparseChallengeConfig;

#[cfg(test)]
use super::onehot_source::{canonical_source_classes, SourceClass};
use super::onehot_source::{deterministic_convolution_cap, ln_upper, max_log_mgf_upper, round_up};
use super::{
    fold_witness_linf_cap, num_digits_for_linf_cap, FoldChallengeNorms, FoldWitnessLinfCapConfig,
    FoldWitnessNorms,
};

/// Failure probability a fold's digit range must meet for an honest prover.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FoldFailureTarget {
    /// Sizing for a prover that retries: each policy's own per-attempt
    /// constant.
    PerAttempt,
    /// One shot: failure probability at most `2^-security_bits / masked_folds`
    /// for this fold, which the prover probes once.
    OneShot {
        security_bits: u32,
        masked_folds: u32,
    },
}

/// Exact candidate geometry supplied to an offline honest-fold policy.
#[derive(Clone, Copy, Debug)]
pub struct HonestFoldSizingQuery<'a> {
    pub ring_dimension: usize,
    /// Dimension in which the sparse fold challenge is sampled.
    pub challenge_dimension: usize,
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
    /// Exact source-plane norms after the selected inner decomposition.
    pub witness_norms: FoldWitnessNorms,
    /// Basis used to decompose the emitted folded response.
    pub log_basis_response: u32,
    pub challenge_config: &'a SparseChallengeConfig,
    pub failure_target: FoldFailureTarget,
}

/// One group-owned offline rule for selecting its folded-witness digit depth.
pub trait HonestFoldPolicy {
    /// Return the final digit depth selected by the policy.
    fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError>;
}

/// Distribution-free sizing rule for balanced signed-digit witnesses.
///
/// The producer contract derives the query's worst digit-plane norms from its
/// declared bound and selected decomposition. This policy combines those norms
/// with the fold challenge; it does not reinterpret the source declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BalancedSignedDigitFoldPolicy {
    field_bits: u32,
}

impl BalancedSignedDigitFoldPolicy {
    /// Construct the distribution-free policy for one field width.
    #[must_use]
    pub const fn universal(field_bits: u32) -> Self {
        Self { field_bits }
    }

    fn universal_cap(&self, query: HonestFoldSizingQuery<'_>) -> Result<u128, AkitaError> {
        validate_query(query)?;
        let cap_config = match query.failure_target {
            // This policy is the frozen pre-chunking baseline. Preserve its
            // logical single-fold geometry even though the query reports
            // every physical response coefficient.
            FoldFailureTarget::PerAttempt => FoldWitnessLinfCapConfig::for_fold_coeffs(
                query.challenge_config,
                query.num_fold_coeffs / query.num_chunks,
            )?,
            // A one-shot union bound covers every physical coefficient.
            FoldFailureTarget::OneShot {
                security_bits,
                masked_folds,
            } => FoldWitnessLinfCapConfig::one_shot_for_fold_coeffs(
                query.challenge_config,
                query.num_fold_coeffs,
                security_bits,
                masked_folds,
            )?,
        };
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
        let cap = self.universal_cap(query)?;
        Ok(self.digit_depth_for_cap(cap, query.log_basis_response))
    }
}

/// Kernel-faithful canonical-source policy for unit one-hot root folds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnitOneHotFoldPolicy {
    field_bits: u32,
    source_chunk_size: usize,
}

/// Canonical logical chunk size of the shipping unit one-hot representation.
pub const DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE: usize = 256;

impl UnitOneHotFoldPolicy {
    /// Construct a unit one-hot policy for one base/extension field profile.
    #[must_use]
    pub const fn canonical(field_bits: u32, source_chunk_size: usize) -> Self {
        Self {
            field_bits,
            source_chunk_size,
        }
    }

    /// Logical source chunk size covered by the unit one-hot contract.
    #[must_use]
    pub const fn source_chunk_size(self) -> usize {
        self.source_chunk_size
    }

    fn independent_challenge_groups(&self, query: HonestFoldSizingQuery<'_>) -> Option<usize> {
        let chunk_size = self.source_chunk_size;
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
        canonical_source_classes(query.ring_dimension, self.source_chunk_size)
    }

    fn deterministic_cap(&self, query: HonestFoldSizingQuery<'_>) -> Option<u128> {
        let per_group = deterministic_convolution_cap(
            query.ring_dimension,
            self.source_chunk_size,
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
        //   2 N exp(m log M_X(lambda) - lambda t) <= target,
        //
        // with target 7/8 per attempt, or 2^-security_bits / masked_folds for
        // one shot. Every sampled lambda yields a valid upper bound. Missing
        // the true minimizer can therefore only select a larger threshold,
        // and the deterministic cap, which never fails, bounds the result.
        let union_factor = match query.failure_target {
            FoldFailureTarget::PerAttempt => {
                round_up(round_up(16.0 * query.num_fold_coeffs as f64) / 7.0)
            }
            FoldFailureTarget::OneShot {
                security_bits,
                masked_folds,
            } => {
                // Scaling by a power of two is exact in binary floating point.
                round_up(round_up(2.0 * query.num_fold_coeffs as f64) * f64::from(masked_folds))
                    * 2f64.powi(i32::try_from(security_bits).ok()?)
            }
        };
        let union_log = ln_upper(union_factor)?;
        let mut best = worst_case;
        for exponent in -12i32..=2 {
            let lambda = 2f64.powi(exponent);
            let log_mgf = max_log_mgf_upper(
                query.ring_dimension,
                self.source_chunk_size,
                query.challenge_dimension,
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
        if self.source_chunk_size == 0 || !self.source_chunk_size.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "unit one-hot source chunk size must be a nonzero power of two".into(),
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
        || query.challenge_dimension == 0
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
    if !query
        .ring_dimension
        .is_multiple_of(query.challenge_dimension)
    {
        return Err(AkitaError::InvalidSetup(
            "honest fold challenge dimension must divide the ambient ring dimension".into(),
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
        .validate_for_ring_dim(query.challenge_dimension)
        .map_err(|message| AkitaError::InvalidSetup(message.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query<'a>(
        challenge: &'a SparseChallengeConfig,
        ring_dimension: usize,
        blocks: usize,
        positions_per_block: usize,
    ) -> HonestFoldSizingQuery<'a> {
        HonestFoldSizingQuery {
            ring_dimension,
            challenge_dimension: ring_dimension,
            num_claims: 1,

            num_live_ring_elements_per_claim: blocks * positions_per_block,
            num_positions_per_block: positions_per_block,
            num_live_blocks: blocks,

            num_chunks: 1,
            num_fold_coeffs: positions_per_block * ring_dimension,
            witness_norms: FoldWitnessNorms::new(1, 1),
            log_basis_response: 3,
            challenge_config: challenge,
            failure_target: FoldFailureTarget::PerAttempt,
        }
    }

    #[test]
    fn fp128_nv36_drops_the_spurious_positions_per_block_multiplier() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(256).unwrap();
        let policy = UnitOneHotFoldPolicy::canonical(128, 256);
        let query = query(&challenge, 256, 4_096, 65_536);
        assert!(policy.exact_threshold(query).unwrap() <= 219);
        assert_eq!(policy.num_digits_fold(query).unwrap(), 3);
    }

    #[test]
    fn fp64_canonical_source_uses_the_existing_degree_one_bound() {
        let d256_challenge = SparseChallengeConfig::production_for_ring_dim(256).unwrap();
        let d512_challenge = SparseChallengeConfig::production_for_ring_dim(512).unwrap();
        let policy = UnitOneHotFoldPolicy::canonical(64, 256);
        for query in [
            query(&d256_challenge, 256, 256, 4_096),
            query(&d512_challenge, 512, 256, 8_192),
        ] {
            assert!(policy.exact_threshold(query).is_some());
            assert!(policy.num_digits_fold(query).is_ok());
        }
    }

    #[test]
    fn reduced_challenge_subring_never_underprices_the_ambient_population() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let policy = UnitOneHotFoldPolicy::canonical(32, 256);
        let mut packing = query(&challenge, 1_024, 64, 16);
        packing.challenge_dimension = 64;
        let ambient = HonestFoldSizingQuery {
            challenge_dimension: 1_024,
            ..packing
        };
        assert!(
            policy.exact_threshold(packing).unwrap() >= policy.exact_threshold(ambient).unwrap()
        );
    }

    #[test]
    fn every_chunk_to_ring_branch_uses_the_canonical_group_source() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(256).unwrap();
        for chunk_size in [16, 256, 512] {
            let direct = UnitOneHotFoldPolicy::canonical(128, chunk_size);
            let query = query(&challenge, 256, 8, 4);
            assert!(direct.exact_threshold(query).is_some());
        }
    }

    #[test]
    fn every_field_tier_uses_the_same_canonical_root_energy() {
        for field_bits in [32, 64, 128] {
            let source = crate::sis::CommittedSourceContract::try_new(
                crate::sis::CommittedSourceClass::UnitOneHot {
                    source_chunk_size: 256,
                },
                crate::DecompositionParams {
                    log_basis: 3,
                    log_commit_bound: 1,
                    log_open_bound: Some(field_bits),
                },
            )
            .unwrap();
            let norms = source.source_norms(3, 1, 256, 4_096).unwrap();
            assert_eq!((norms.l2_sq, norms.coefficient_sq_max), (16, 1));
        }
    }

    #[test]
    fn root_energy_matches_exhaustive_canonical_onehot_tables() {
        const D: usize = 8;
        for chunk_size in [4, 8] {
            let source = crate::sis::CommittedSourceContract::try_new(
                crate::sis::CommittedSourceClass::UnitOneHot {
                    source_chunk_size: chunk_size,
                },
                crate::DecompositionParams {
                    log_basis: 3,
                    log_commit_bound: 1,
                    log_open_bound: Some(32),
                },
            )
            .unwrap();
            let modeled = source.source_norms(3, 1, D, D).unwrap().l2_sq;
            let chunk_count = D / chunk_size;
            let choices_per_chunk = chunk_size + 1;
            let mut observed_max = 0u128;
            for mut choice_code in 0..choices_per_chunk.pow(chunk_count as u32) {
                let mut source = [0i8; D];
                for chunk in 0..chunk_count {
                    let choice = choice_code % choices_per_chunk;
                    choice_code /= choices_per_chunk;
                    if choice != chunk_size {
                        source[chunk * chunk_size + choice] = 1;
                    }
                }
                let energy = source.iter().try_fold(0u128, |sum, value| {
                    sum.checked_add(u128::from(value.unsigned_abs()).pow(2))
                });
                observed_max = observed_max.max(energy.expect("small energy"));
            }
            assert_eq!(modeled, observed_max);
        }
    }

    #[test]
    fn fp32_small_block_row_uses_canonical_chunk_occupancy() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(512).unwrap();
        let policy = UnitOneHotFoldPolicy::canonical(32, 256);
        let query = query(&challenge, 512, 8, 4);
        let source_classes = policy.source_classes(query).unwrap();
        assert_eq!(source_classes[0].nonzero_count, 2);
        assert_eq!(policy.independent_challenge_groups(query), Some(8));
    }

    #[test]
    fn k16_multiple_ones_per_ring_are_charged_canonically() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(512).unwrap();
        let policy = UnitOneHotFoldPolicy::canonical(32, 16);
        let query = query(&challenge, 512, 8, 4);
        let classes = policy.source_classes(query).unwrap();
        assert_eq!(classes, vec![SourceClass { nonzero_count: 32 }]);
        assert!(policy.deterministic_cap(query).is_some());
    }

    #[test]
    fn k_greater_than_d_counts_distinct_chunks_inside_each_window() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let policy = UnitOneHotFoldPolicy::canonical(128, 256);
        let query = HonestFoldSizingQuery {
            ring_dimension: 64,
            challenge_dimension: 64,
            num_claims: 1,

            num_live_ring_elements_per_claim: 8,
            num_positions_per_block: 1,
            num_live_blocks: 8,

            num_chunks: 2,
            num_fold_coeffs: 2 * 64,
            witness_norms: FoldWitnessNorms::new(1, 1),
            log_basis_response: 3,
            challenge_config: &challenge,
            failure_target: FoldFailureTarget::PerAttempt,
        };
        assert_eq!(policy.independent_challenge_groups(query), Some(1));
        assert!(policy.num_digits_fold(query).is_ok());
    }

    #[test]
    fn k_greater_than_d_handles_nondyadic_blocks_and_a_partial_final_block() {
        let challenge = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let policy = UnitOneHotFoldPolicy::canonical(128, 256);
        let query = HonestFoldSizingQuery {
            ring_dimension: 64,
            challenge_dimension: 64,
            num_claims: 1,

            num_live_ring_elements_per_claim: 8,
            num_positions_per_block: 3,
            num_live_blocks: 3,

            num_chunks: 1,
            num_fold_coeffs: 3 * 64,
            witness_norms: FoldWitnessNorms::new(1, 1),
            log_basis_response: 3,
            challenge_config: &challenge,
            failure_target: FoldFailureTarget::PerAttempt,
        };
        assert_eq!(policy.independent_challenge_groups(query), Some(2));
        assert!(policy.num_digits_fold(query).is_ok());
    }

    /// One-shot target of each fold at or before the zero-knowledge cut.
    fn zk_target() -> FoldFailureTarget {
        FoldFailureTarget::OneShot {
            security_bits: crate::zk::ZK_FOLD_SECURITY_BITS,
            masked_folds: crate::zk::ZK_CUT_LEVEL + 1,
        }
    }

    /// Level 0 of `fp128_onehot_recursive final=36:1`: the one-hot root,
    /// folded with packing challenges in the 64-dimensional subring.
    fn headline_root(challenge: &SparseChallengeConfig) -> HonestFoldSizingQuery<'_> {
        HonestFoldSizingQuery {
            ring_dimension: 512,
            challenge_dimension: 64,
            num_claims: 1,
            num_live_ring_elements_per_claim: 1 << 27,
            num_positions_per_block: 1 << 15,
            num_live_blocks: 1 << 12,
            num_chunks: 1,
            num_fold_coeffs: (1 << 15) * 512,
            witness_norms: FoldWitnessNorms::new(1, 2),
            log_basis_response: 3,
            challenge_config: challenge,
            failure_target: zk_target(),
        }
    }

    /// Level 1 of `fp128_onehot_recursive final=36:1`: a recursive fold of
    /// balanced base-8 digits.
    fn headline_level1(challenge: &SparseChallengeConfig) -> HonestFoldSizingQuery<'_> {
        HonestFoldSizingQuery {
            ring_dimension: 512,
            challenge_dimension: 512,
            num_claims: 1,
            num_live_ring_elements_per_claim: 329_469,
            num_positions_per_block: 8_192,
            num_live_blocks: 41,
            num_chunks: 1,
            num_fold_coeffs: 8_192 * 512,
            witness_norms: FoldWitnessNorms::bounded(3, 512),
            log_basis_response: 3,
            challenge_config: challenge,
            failure_target: zk_target(),
        }
    }

    /// Smallest admitted magnitude of a response with `num_digits` balanced
    /// digits: the range covers `[-reach, reach]`.
    fn response_reach(log_basis: u32, num_digits: usize) -> u128 {
        let (negative, positive) =
            crate::sis::balanced_digit_representable_bounds(log_basis, num_digits);
        negative.min(positive)
    }

    /// `log2` of the one-shot failure bound of a balanced fold with digit
    /// reach `reach`, and whether `2N exp(-reach^2 / 2 sigma^2) <= 2^-137`
    /// holds in exact rational arithmetic (`ln 2 < 6932/10000`), or the reach
    /// covers the deterministic envelope, where the failure probability is 0.
    fn balanced_failure(query: HonestFoldSizingQuery<'_>, reach: u128) -> (f64, bool) {
        let challenge = FoldChallengeNorms::new(query.challenge_config);
        let envelope = crate::sis::fold_witness_beta_inf(
            query.num_live_blocks,
            query.num_claims,
            challenge,
            query.witness_norms,
        )
        .unwrap();
        if reach >= envelope {
            return (f64::NEG_INFINITY, true);
        }
        let linf = query.witness_norms.infinity_norm();
        let sigma_sq = (query.num_claims * query.num_live_blocks) as u128
            * query.challenge_config.challenge_l2_sq_max()
            * linf
            * linf;
        let two_n = 2 * query.num_fold_coeffs as u128;
        let log2_two_n = u128::from(128 - (two_n - 1).leading_zeros());
        let holds = reach * reach * 10_000 >= 2 * sigma_sq * (log2_two_n + 137) * 6_932;
        let log2_bound = (two_n as f64).log2()
            - (reach * reach) as f64 / (2.0 * sigma_sq as f64) * std::f64::consts::LOG2_E;
        (log2_bound, holds)
    }

    /// `log2` of the best Chernoff bound over the policy's lambda grid for a
    /// one-hot fold with threshold `reach`, and whether it is at most
    /// `2^-137` with every term rounded up; a reach that covers the
    /// deterministic cap has failure probability 0.
    fn onehot_failure(
        policy: UnitOneHotFoldPolicy,
        query: HonestFoldSizingQuery<'_>,
        reach: u128,
    ) -> (f64, bool) {
        if reach >= policy.deterministic_cap(query).unwrap() {
            return (f64::NEG_INFINITY, true);
        }
        let groups = policy.independent_challenge_groups(query).unwrap() as f64;
        let ln_two_n = ln_upper(2.0 * query.num_fold_coeffs as f64).unwrap();
        let ln_target = ln_upper(2f64.powi(137)).unwrap();
        let best = (-12i32..=2)
            .filter_map(|exponent| {
                let log_mgf = max_log_mgf_upper(
                    query.ring_dimension,
                    policy.source_chunk_size,
                    query.challenge_dimension,
                    query.challenge_config,
                    exponent,
                )?;
                let lambda = 2f64.powi(exponent);
                Some(round_up(groups * log_mgf) + ln_two_n - lambda * reach as f64)
            })
            .reduce(f64::min)
            .unwrap();
        (
            best * std::f64::consts::LOG2_E,
            round_up(best + ln_target) <= 0.0,
        )
    }

    /// W2-c: at the headline shapes each one-shot fold fails with probability
    /// at most `2^-136 / 2`, and the paper's zero-knowledge error stays below
    /// `2^-128`.
    #[test]
    fn one_shot_headline_folds_meet_the_per_fold_target() {
        let root_challenge = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let level1_challenge = SparseChallengeConfig::production_for_ring_dim(512).unwrap();
        assert_eq!(
            (root_challenge.count_pm1, root_challenge.count_pm2),
            (31, 10)
        );
        assert_eq!(
            (level1_challenge.count_pm1, level1_challenge.count_pm2),
            (19, 0)
        );

        let onehot = UnitOneHotFoldPolicy::canonical(128, DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE);
        let root = headline_root(&root_challenge);
        let root_digits = onehot.num_digits_fold(root).unwrap();
        let (root_log2, root_holds) = onehot_failure(onehot, root, response_reach(3, root_digits));
        assert!(root_holds, "root: log2 failure {root_log2}");

        let balanced = BalancedSignedDigitFoldPolicy::universal(128);
        let level1 = headline_level1(&level1_challenge);
        let level1_digits = balanced.num_digits_fold(level1).unwrap();
        let (level1_log2, level1_holds) =
            balanced_failure(level1, response_reach(3, level1_digits));
        assert!(level1_holds, "level 1: log2 failure {level1_log2}");

        // Theorem wo:thm:zk with the paper's parameters (section 6.2):
        // delta_LHL <= 2^-129.2, T_att = 163, epsilon = 2^-137, M = 2.37 and
        // a 256-bit nonce. Q_tot = 2^100 is this test's assumption; the paper
        // does not state it.
        let delta_fold = root_log2.exp2() + level1_log2.exp2();
        let t_att = 163f64;
        let error = 2.0 * delta_fold
            + (-129.2f64).exp2()
            + t_att * (-137f64).exp2() / 2.37
            + (t_att * 100f64.exp2() + t_att * t_att) / 256f64.exp2();
        assert!(
            error.log2() < -128.0,
            "zero-knowledge error 2^{}",
            error.log2()
        );
    }

    #[test]
    fn one_shot_targets_widen_both_policies() {
        let root_challenge = SparseChallengeConfig::production_for_ring_dim(64).unwrap();
        let level1_challenge = SparseChallengeConfig::production_for_ring_dim(512).unwrap();
        let onehot = UnitOneHotFoldPolicy::canonical(128, DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE);
        let root = headline_root(&root_challenge);
        let per_attempt = HonestFoldSizingQuery {
            failure_target: FoldFailureTarget::PerAttempt,
            ..root
        };
        assert!(
            onehot.exact_threshold(root).unwrap() > onehot.exact_threshold(per_attempt).unwrap()
        );

        let balanced = BalancedSignedDigitFoldPolicy::universal(128);
        let level1 = headline_level1(&level1_challenge);
        let per_attempt = HonestFoldSizingQuery {
            failure_target: FoldFailureTarget::PerAttempt,
            ..level1
        };
        assert!(
            balanced.universal_cap(level1).unwrap() > balanced.universal_cap(per_attempt).unwrap()
        );
    }
}
