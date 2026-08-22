//! Offline honest-prover sizing policies for folded witnesses.
//!
//! These policies select an exact gadget depth for schedule generation. They
//! are not runtime protocol metadata and are never evaluated by the verifier.

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;

#[cfg(test)]
use super::onehot_source::SourceClass;
use super::onehot_source::{
    canonical_source_classes, deterministic_convolution_cap, ln_upper, max_log_mgf_upper, round_up,
};
use super::{
    fold_witness_linf_cap, num_digits_for_linf_cap, FoldChallengeNorms, FoldWitnessLinfCapConfig,
    FoldWitnessNorms,
};

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
}

/// One group-owned offline rule for selecting its folded-witness digit depth.
pub trait HonestFoldPolicy {
    /// Return the final digit depth selected by the policy.
    fn num_digits_fold(&self, query: HonestFoldSizingQuery<'_>) -> Result<usize, AkitaError>;
}

/// Distribution-free sizing rule for balanced signed-digit witnesses.
///
/// This is the sizing rule for every source whose committed plane is balanced
/// base-`2^log_basis_inner` digits, which is the whole
/// `1 < log_commit_bound <= field_bits` range: a bounded source and a full-field
/// source decompose into the same digit alphabet and differ only in how many
/// digit planes they need. The declared source bound therefore does not appear
/// here — it is carried by
/// [`crate::DecompositionParams::log_commit_bound`] and consumed by the
/// A-role digit depth. The per-block source norms this policy sizes against
/// come from the query
/// ([`HonestFoldSizingQuery::witness_norms`], built by
/// [`HonestFoldPolicySpec::witness_norms_for_inner_basis`]), which describes one
/// balanced digit plane and is independent of the bound.
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

impl HonestFoldPolicySpec {
    /// Source-plane norms for one selected A decomposition basis.
    ///
    /// Balanced sources follow the candidate basis. Unit one-hot sources keep
    /// their profile-owned sparse norm; the planner canonicalizes their
    /// already-single-digit representation without a basis sweep.
    pub fn witness_norms_for_inner_basis(
        self,
        log_basis_inner: u32,
        ring_dimension: usize,
    ) -> Result<FoldWitnessNorms, AkitaError> {
        match self {
            Self::BalancedSignedDigit(_) => {
                Ok(FoldWitnessNorms::bounded(log_basis_inner, ring_dimension))
            }
            Self::UnitOneHot(policy) => {
                let classes = canonical_source_classes(ring_dimension, policy.source_chunk_size)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "unit one-hot source geometry is unsupported or overflows".into(),
                        )
                    })?;
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
                Ok(FoldWitnessNorms::new(infinity_norm, l1_norm))
            }
        }
    }

    /// Maximum physical squared coefficient norm of a valid canonical root source.
    ///
    /// This maximizes over every hot position allowed by the unit one-hot
    /// chunk contract. Distinct chunks occupy distinct canonical coefficients.
    pub fn root_source_l2_sq(
        self,
        logical_len: usize,
        ring_dimension: usize,
    ) -> Option<(u128, u128)> {
        match self {
            Self::BalancedSignedDigit(_) => None,
            Self::UnitOneHot(policy) => {
                if logical_len == 0
                    || ring_dimension == 0
                    || policy.source_chunk_size == 0
                    || !logical_len.is_multiple_of(policy.source_chunk_size)
                    || !logical_len.is_multiple_of(ring_dimension)
                {
                    return None;
                }
                let classes = canonical_source_classes(ring_dimension, policy.source_chunk_size)?;
                let per_group_energy = classes.iter().try_fold(0u128, |maximum, class| {
                    let energy = class.nonzero_count as u128;
                    Some(maximum.max(energy))
                })?;
                let group_count = if policy.source_chunk_size >= ring_dimension {
                    logical_len / policy.source_chunk_size
                } else {
                    logical_len / ring_dimension
                };
                let energy = per_group_energy.checked_mul(group_count as u128)?;
                let coefficient_sq_max =
                    usize::from(classes.iter().any(|class| class.nonzero_count > 0)) as u128;
                Some((energy, coefficient_sq_max))
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
            let policy =
                HonestFoldPolicySpec::UnitOneHot(UnitOneHotFoldPolicy::canonical(field_bits, 256));
            let (energy, coefficient_sq_max) = policy
                .root_source_l2_sq(4_096, 256)
                .expect("supported canonical root geometry");

            assert_eq!(energy, 16);
            assert_eq!(coefficient_sq_max, 1);
        }
    }

    #[test]
    fn root_energy_matches_exhaustive_canonical_onehot_tables() {
        const D: usize = 8;
        for chunk_size in [4, 8] {
            let policy =
                HonestFoldPolicySpec::UnitOneHot(UnitOneHotFoldPolicy::canonical(32, chunk_size));
            let modeled = policy
                .root_source_l2_sq(D, D)
                .expect("supported small root geometry")
                .0;
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
        };
        assert_eq!(policy.independent_challenge_groups(query), Some(2));
        assert!(policy.num_digits_fold(query).is_ok());
    }
}
