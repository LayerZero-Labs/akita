//! Deterministic standalone commitment-compression candidate selection.
//!
//! Selection tries ordered [`CompressionByteLadder`] values first. Each ladder
//! names a first-map alphabet and exact intermediate image sizes in bytes.
//! A ladder is kept only when every rung materializes as rank-one at
//! `bytes / field_bytes` and catalog replay succeeds. Exhaustive enumeration
//! remains an explicit completeness fallback, not the default search shape.

mod replay;

use akita_field::{AkitaError, CanonicalField};
use akita_types::{
    compression_digit_depth, field_bytes, protocol_dispatch_tier, slot_dims_for_tier,
    AjtaiKeyParams, CompressionAlphabet, CompressionCatalogContext, CompressionSourceId,
    LevelParams, ProtocolDispatchSlot, SetupMatrixEnvelope, ValidatedCompressionCatalog,
    STANDALONE_OPENING_BASE_LOG_BASIS,
};

use crate::PlannerPolicy;
use replay::{
    derive_compression_key, replay_compression_catalog, validate_replay_policy,
    CompressionChainDescriptor, CompressionMapDescriptor,
};

/// Protocol-fixed exhaustive search bound: two first-map alphabets, depths
/// 2/3, and at most four compression dispatch dimensions per tier
/// (`2 * (4^2 + 4^3) = 160`).
pub const MAX_EXHAUSTIVE_COMPRESSION_CANDIDATES: usize = 160;

/// First-map alphabet choice for a byte ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionFirstMapAlphabet {
    NegativeBinary,
    OpeningBase,
}

impl CompressionFirstMapAlphabet {
    #[must_use]
    const fn alphabet(self) -> CompressionAlphabet {
        match self {
            Self::NegativeBinary => CompressionAlphabet::NegativeBinary,
            Self::OpeningBase => CompressionAlphabet::OpeningBase {
                log_basis: STANDALONE_OPENING_BASE_LOG_BASIS,
            },
        }
    }
}

/// Ordered intermediate image sizes in bytes, including the terminal rung.
///
/// Each rung selects preferred ring dimension `bytes / field_bytes` and must
/// replay at rank one with that exact byte size. Misses skip the ladder; they
/// never inflate rank to force a hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressionByteLadder {
    first_alphabet: CompressionFirstMapAlphabet,
    image_bytes: &'static [usize],
}

impl CompressionByteLadder {
    #[must_use]
    pub const fn new(
        first_alphabet: CompressionFirstMapAlphabet,
        image_bytes: &'static [usize],
    ) -> Self {
        Self {
            first_alphabet,
            image_bytes,
        }
    }

    #[must_use]
    pub const fn first_alphabet(self) -> CompressionFirstMapAlphabet {
        self.first_alphabet
    }

    #[must_use]
    pub const fn image_bytes(self) -> &'static [usize] {
        self.image_bytes
    }
}

/// Negative-binary maps `256 → 128`.
pub const BINARY_256_128: CompressionByteLadder =
    CompressionByteLadder::new(CompressionFirstMapAlphabet::NegativeBinary, &[256, 128]);

/// Opening-base first map, then negative-binary `512 → 256 → 128`.
pub const OPENING_BASE_512_256_128: CompressionByteLadder =
    CompressionByteLadder::new(CompressionFirstMapAlphabet::OpeningBase, &[512, 256, 128]);

/// Opening-base first map, then negative-binary `1024 → 256 → 128`.
pub const OPENING_BASE_1024_256_128: CompressionByteLadder =
    CompressionByteLadder::new(CompressionFirstMapAlphabet::OpeningBase, &[1024, 256, 128]);

/// Shipped standalone ladders, tried in this order.
pub const DEFAULT_COMPRESSION_BYTE_LADDERS: &[CompressionByteLadder] = &[
    BINARY_256_128,
    OPENING_BASE_512_256_128,
    OPENING_BASE_1024_256_128,
];

/// Standalone compression objectives that do not participate in the generated
/// schedule catalog identity until the planner cutover integrates compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressionPlannerPolicy {
    target_payload_bytes: usize,
    ladders: &'static [CompressionByteLadder],
    allow_exhaustive_fallback: bool,
}

impl CompressionPlannerPolicy {
    #[must_use]
    pub const fn new(target_payload_bytes: usize) -> Self {
        Self {
            target_payload_bytes,
            ladders: DEFAULT_COMPRESSION_BYTE_LADDERS,
            allow_exhaustive_fallback: true,
        }
    }

    #[must_use]
    pub const fn with_ladders(
        target_payload_bytes: usize,
        ladders: &'static [CompressionByteLadder],
        allow_exhaustive_fallback: bool,
    ) -> Self {
        Self {
            target_payload_bytes,
            ladders,
            allow_exhaustive_fallback,
        }
    }

    #[must_use]
    pub const fn target_payload_bytes(self) -> usize {
        self.target_payload_bytes
    }

    #[must_use]
    pub const fn ladders(self) -> &'static [CompressionByteLadder] {
        self.ladders
    }

    #[must_use]
    pub const fn allow_exhaustive_fallback(self) -> bool {
        self.allow_exhaustive_fallback
    }
}

impl Default for CompressionPlannerPolicy {
    fn default() -> Self {
        Self::new(128)
    }
}

/// The best admissible standalone compression chain and its objective facts.
#[derive(Debug, Clone)]
pub struct CompressionSelection {
    catalog: ValidatedCompressionCatalog,
    payload_bytes: usize,
    target_met: bool,
    depth: usize,
    first_alphabet: CompressionAlphabet,
    map_ring_dimensions: Vec<usize>,
    candidates_considered: usize,
    used_exhaustive_fallback: bool,
    selected_ladder: Option<CompressionByteLadder>,
}

impl CompressionSelection {
    #[must_use]
    pub fn catalog(&self) -> &ValidatedCompressionCatalog {
        &self.catalog
    }

    #[must_use]
    pub fn into_catalog(self) -> ValidatedCompressionCatalog {
        self.catalog
    }

    #[must_use]
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    #[must_use]
    pub fn target_met(&self) -> bool {
        self.target_met
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    #[must_use]
    pub fn first_alphabet(&self) -> CompressionAlphabet {
        self.first_alphabet
    }

    #[must_use]
    pub fn map_ring_dimensions(&self) -> &[usize] {
        &self.map_ring_dimensions
    }

    #[must_use]
    pub fn candidates_considered(&self) -> usize {
        self.candidates_considered
    }

    #[must_use]
    pub fn used_exhaustive_fallback(&self) -> bool {
        self.used_exhaustive_fallback
    }

    #[must_use]
    pub fn selected_ladder(&self) -> Option<CompressionByteLadder> {
        self.selected_ladder
    }
}

#[derive(Debug)]
struct Candidate {
    catalog: ValidatedCompressionCatalog,
    payload_bytes: usize,
    global_setup_prefix_coeffs: usize,
    global_cache_field_coeffs: usize,
    logical_setup_coeffs: usize,
    descriptor_bytes: Vec<u8>,
    depth: usize,
    first_alphabet: CompressionAlphabet,
    map_ring_dimensions: Vec<usize>,
}

fn candidate_score(
    target_payload_bytes: usize,
    payload_bytes: usize,
    global_setup_prefix_coeffs: usize,
    global_cache_field_coeffs: usize,
    logical_setup_coeffs: usize,
    descriptor_bytes: &[u8],
) -> (bool, usize, usize, usize, usize, &[u8]) {
    (
        payload_bytes > target_payload_bytes,
        payload_bytes,
        global_setup_prefix_coeffs,
        global_cache_field_coeffs,
        logical_setup_coeffs,
        descriptor_bytes,
    )
}

/// Preferred rank-one ring dimension for an exact image size in bytes.
#[must_use]
pub fn rank_one_ring_dim_for_bytes(
    target_bytes: usize,
    field_element_bytes: usize,
) -> Option<usize> {
    if field_element_bytes == 0 || !target_bytes.is_multiple_of(field_element_bytes) {
        return None;
    }
    let d = target_bytes / field_element_bytes;
    (d > 0).then_some(d)
}

struct Materializer<'a, F> {
    policy: &'a PlannerPolicy,
    lp: &'a LevelParams,
    field_bits: u32,
    field_element_bytes: usize,
    base_setup_coeffs: usize,
    dimensions: &'a [usize],
    _field: core::marker::PhantomData<F>,
}

impl<F: CanonicalField> Materializer<'_, F> {
    fn key_for(
        &self,
        alphabet: CompressionAlphabet,
        d: usize,
        col_len: usize,
    ) -> Option<AjtaiKeyParams> {
        derive_compression_key(self.policy, self.policy.basis_range.1, alphabet, d, col_len)
    }

    fn finish_candidate(
        &self,
        first_alphabet: CompressionAlphabet,
        maps: &[CompressionMapDescriptor],
    ) -> Option<Candidate> {
        let descriptor = CompressionChainDescriptor {
            source: CompressionSourceId::CurrentOuter,
            max_opening_log_basis: self.policy.basis_range.1,
            maps: maps.to_vec(),
        };
        let catalog = replay_compression_catalog::<F>(
            self.policy,
            self.lp,
            CompressionCatalogContext::StandaloneCommitment,
            &[descriptor],
        )
        .ok()?;
        let projection = catalog.project_for_schedule().ok()?;
        let payload_coeffs = projection.payload_coeffs(CompressionSourceId::CurrentOuter)?;
        let payload_bytes = payload_coeffs.checked_mul(self.field_element_bytes)?;
        let (global_setup_prefix_coeffs, global_cache_field_coeffs) = global_setup_objectives(
            self.base_setup_coeffs,
            self.policy.ring_dimension,
            projection.max_flat_setup_prefix_coeffs(),
            projection.ntt_requirements(),
        )
        .ok()?;
        Some(Candidate {
            catalog,
            payload_bytes,
            global_setup_prefix_coeffs,
            global_cache_field_coeffs,
            logical_setup_coeffs: projection.logical_setup_coeffs(),
            descriptor_bytes: projection.descriptor_bytes().to_vec(),
            depth: maps.len(),
            first_alphabet,
            map_ring_dimensions: maps.iter().map(|map| map.ring_d).collect(),
        })
    }

    /// Materialize one ladder. Returns `None` on any miss (bad size, missing
    /// dispatch/SIS coverage, rank greater than one, or failed replay).
    fn materialize_ladder(
        &self,
        ladder: CompressionByteLadder,
        source_output_coeffs: usize,
    ) -> Option<Candidate> {
        let images = ladder.image_bytes();
        if images.len() < 2 || images.len() > 3 {
            return None;
        }
        if images.windows(2).any(|pair| pair[0] <= pair[1]) {
            return None;
        }
        let first_alphabet = ladder.first_alphabet().alphabet();
        let mut maps = Vec::with_capacity(images.len());
        let mut previous_output_coeffs = source_output_coeffs;
        for (index, &target_bytes) in images.iter().enumerate() {
            let alphabet = if index == 0 {
                first_alphabet
            } else {
                CompressionAlphabet::NegativeBinary
            };
            let d = rank_one_ring_dim_for_bytes(target_bytes, self.field_element_bytes)?;
            if !self.dimensions.contains(&d) || !self.policy.ring_dimension.is_multiple_of(d) {
                return None;
            }
            let digit_depth =
                compression_digit_depth(alphabet, self.field_bits, self.policy.basis_range.1)
                    .ok()?;
            let input_coeffs = previous_output_coeffs.checked_mul(digit_depth)?;
            if !input_coeffs.is_multiple_of(d) {
                return None;
            }
            let key = self.key_for(alphabet, d, input_coeffs / d)?;
            // Exact rank-one size: never inflate rank to hit the byte target.
            if key.row_len() != 1 {
                return None;
            }
            let output_coeffs = key.row_len().checked_mul(d)?;
            let output_bytes = output_coeffs.checked_mul(self.field_element_bytes)?;
            if output_bytes != target_bytes {
                return None;
            }
            maps.push(CompressionMapDescriptor {
                ring_d: d,
                alphabet,
            });
            previous_output_coeffs = output_coeffs;
        }
        self.finish_candidate(first_alphabet, &maps)
    }

    fn extend_chain(
        &self,
        target_depth: usize,
        first_alphabet: CompressionAlphabet,
        previous_output_coeffs: usize,
        maps: &mut Vec<CompressionMapDescriptor>,
        candidates: &mut Vec<Candidate>,
    ) {
        if maps.len() == target_depth {
            if let Some(candidate) = self.finish_candidate(first_alphabet, maps) {
                candidates.push(candidate);
            }
            return;
        }
        let alphabet = if maps.is_empty() {
            first_alphabet
        } else {
            CompressionAlphabet::NegativeBinary
        };
        let Ok(digit_depth) =
            compression_digit_depth(alphabet, self.field_bits, self.policy.basis_range.1)
        else {
            return;
        };
        let Some(input_coeffs) = previous_output_coeffs.checked_mul(digit_depth) else {
            return;
        };
        for &d in self.dimensions {
            if !self.policy.ring_dimension.is_multiple_of(d) || !input_coeffs.is_multiple_of(d) {
                continue;
            }
            let Some(key) = self.key_for(alphabet, d, input_coeffs / d) else {
                continue;
            };
            let Some(output_coeffs) = key.row_len().checked_mul(d) else {
                continue;
            };
            maps.push(CompressionMapDescriptor {
                ring_d: d,
                alphabet,
            });
            self.extend_chain(
                target_depth,
                first_alphabet,
                output_coeffs,
                maps,
                candidates,
            );
            maps.pop();
        }
    }

    fn enumerate_admitted_chains(&self, source_output_coeffs: usize) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        // Opening-base F1 uses the frozen standalone recomposition base
        // (`STANDALONE_OPENING_BASE_LOG_BASIS`), not `policy.basis_range.1`.
        // Digit depth follows that alphabet base; SIS pricing and
        // `max_opening_log_basis` still use the authenticated later-opening
        // envelope `policy.basis_range.1`.
        for depth in 2..=3 {
            for first_alphabet in [
                CompressionAlphabet::OpeningBase {
                    log_basis: STANDALONE_OPENING_BASE_LOG_BASIS,
                },
                CompressionAlphabet::NegativeBinary,
            ] {
                self.extend_chain(
                    depth,
                    first_alphabet,
                    source_output_coeffs,
                    &mut Vec::with_capacity(depth),
                    &mut candidates,
                );
            }
        }
        candidates
    }
}

fn global_setup_objectives(
    base_setup_coeffs: usize,
    gen_ring_dim: usize,
    compression_prefix_coeffs: usize,
    compression_ntt_requirements: &[(usize, usize)],
) -> Result<(usize, usize), AkitaError> {
    if gen_ring_dim == 0 {
        return Err(AkitaError::InvalidSetup(
            "global setup generation dimension must be non-zero".into(),
        ));
    }
    let compression_prefix_rings = compression_prefix_coeffs
        .checked_div(gen_ring_dim)
        .and_then(|rings| {
            rings.checked_add(usize::from(
                !compression_prefix_coeffs.is_multiple_of(gen_ring_dim),
            ))
        })
        .ok_or_else(|| {
            AkitaError::InvalidSetup("compression setup prefix ring count overflow".into())
        })?;
    let rounded_compression_prefix_coeffs = compression_prefix_rings
        .checked_mul(gen_ring_dim)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("compression setup prefix coefficient count overflow".into())
        })?;
    let global_prefix_coeffs = base_setup_coeffs.max(rounded_compression_prefix_coeffs);
    let global_cache_field_coeffs = compression_ntt_requirements.iter().try_fold(
        global_prefix_coeffs,
        |total, &(d, prefix_ring_elements)| {
            if d == gen_ring_dim {
                return Ok(total);
            }
            let compression_cache_coeffs =
                prefix_ring_elements.checked_mul(d).ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "compression NTT cache coefficient count overflow".into(),
                    )
                })?;
            total.checked_add(compression_cache_coeffs).ok_or_else(|| {
                AkitaError::InvalidSetup("global NTT cache coefficient count overflow".into())
            })
        },
    )?;
    Ok((global_prefix_coeffs, global_cache_field_coeffs))
}

fn selection_from_candidate(
    candidate: Candidate,
    target: usize,
    candidates_considered: usize,
    used_exhaustive_fallback: bool,
    selected_ladder: Option<CompressionByteLadder>,
) -> CompressionSelection {
    CompressionSelection {
        target_met: candidate.payload_bytes <= target,
        catalog: candidate.catalog,
        payload_bytes: candidate.payload_bytes,
        depth: candidate.depth,
        first_alphabet: candidate.first_alphabet,
        map_ring_dimensions: candidate.map_ring_dimensions,
        candidates_considered,
        used_exhaustive_fallback,
        selected_ladder,
    }
}

/// Select the deterministic best standalone compression chain under current
/// dispatch and generated SIS-table coverage.
///
/// Tries [`CompressionPlannerPolicy::ladders`] in order. The first ladder that
/// materializes with rank-one exact sizes and successful replay wins. If none
/// do and exhaustive fallback is allowed, enumerates admitted chains and picks
/// the best score. This selector checks the caller's setup-envelope coefficient
/// conversion but does not duplicate the configuration crate's per-level
/// envelope-coverage authority.
pub fn select_standalone_compression<F: CanonicalField>(
    policy: &PlannerPolicy,
    compression_policy: CompressionPlannerPolicy,
    lp: &LevelParams,
    setup_envelope: SetupMatrixEnvelope,
) -> Result<CompressionSelection, AkitaError> {
    validate_replay_policy::<F>(policy, lp)?;
    if setup_envelope.max_setup_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "compression planner setup envelope must be non-zero".into(),
        ));
    }
    let tier = protocol_dispatch_tier::<F>();
    let dimensions = slot_dims_for_tier(tier, ProtocolDispatchSlot::Compression);
    let source_d = lp.b_key.sis_table_key().ring_dimension as usize;
    let source_output_coeffs = lp.b_key.row_len().checked_mul(source_d).ok_or_else(|| {
        AkitaError::InvalidSetup("compression source output coefficient count overflow".into())
    })?;
    let base_setup_coeffs = setup_envelope
        .max_setup_len
        .checked_mul(policy.ring_dimension)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("base setup envelope coefficient count overflow".into())
        })?;
    let materializer = Materializer::<F> {
        policy,
        lp,
        field_bits: F::modulus_bits(),
        field_element_bytes: field_bytes(F::modulus_bits()),
        base_setup_coeffs,
        dimensions,
        _field: core::marker::PhantomData,
    };
    let target = compression_policy.target_payload_bytes();
    let mut ladders_considered = 0usize;
    for &ladder in compression_policy.ladders() {
        ladders_considered = ladders_considered.saturating_add(1);
        if let Some(candidate) = materializer.materialize_ladder(ladder, source_output_coeffs) {
            return Ok(selection_from_candidate(
                candidate,
                target,
                ladders_considered,
                false,
                Some(ladder),
            ));
        }
    }
    if !compression_policy.allow_exhaustive_fallback() {
        return Err(AkitaError::InvalidSetup(
            "no compression byte ladder materialized and exhaustive fallback is disabled".into(),
        ));
    }
    let candidates = materializer.enumerate_admitted_chains(source_output_coeffs);
    let candidates_considered = ladders_considered.saturating_add(candidates.len());
    if candidates.len() > MAX_EXHAUSTIVE_COMPRESSION_CANDIDATES {
        return Err(AkitaError::InvalidSetup(format!(
            "exhaustive compression enumeration produced {} candidates, exceeding ceiling {MAX_EXHAUSTIVE_COMPRESSION_CANDIDATES}",
            candidates.len()
        )));
    }
    let best = candidates
        .into_iter()
        .min_by(|left, right| {
            candidate_score(
                target,
                left.payload_bytes,
                left.global_setup_prefix_coeffs,
                left.global_cache_field_coeffs,
                left.logical_setup_coeffs,
                &left.descriptor_bytes,
            )
            .cmp(&candidate_score(
                target,
                right.payload_bytes,
                right.global_setup_prefix_coeffs,
                right.global_cache_field_coeffs,
                right.logical_setup_coeffs,
                &right.descriptor_bytes,
            ))
        })
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "no standalone compression chain is covered by current dispatch and SIS tables"
                    .into(),
            )
        })?;
    Ok(selection_from_candidate(
        best,
        target,
        candidates_considered,
        true,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::{Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
    use akita_types::sis::rounded_up_collision_inf_norm;
    use akita_types::{
        ChunkedWitnessCfg, DecompositionParams, SisModulusFamily, DEFAULT_SIS_SECURITY_BITS,
    };

    fn synthetic_envelope() -> SetupMatrixEnvelope {
        SetupMatrixEnvelope { max_setup_len: 1 }
    }

    fn fixture(
        family: SisModulusFamily,
        gen_d: usize,
        source_d: usize,
    ) -> (PlannerPolicy, LevelParams) {
        let policy = PlannerPolicy {
            ring_dimension: gen_d,
            decomposition: DecompositionParams {
                log_basis: 4,
                log_commit_bound: 1,
                log_open_bound: Some(4),
            },
            sis_family: family,
            min_sis_security_bits: DEFAULT_SIS_SECURITY_BITS,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            basis_range: (4, 4),
            onehot_chunk_size: 1,
            witness_chunk: ChunkedWitnessCfg::default(),
        };
        let source_bucket = rounded_up_collision_inf_norm(
            DEFAULT_SIS_SECURITY_BITS,
            family,
            source_d,
            policy.basis_range.1,
        )
        .expect("source collision bucket");
        let source_key = AjtaiKeyParams::try_new_with_min_rank(
            akita_types::SisTableKey {
                min_security_bits: DEFAULT_SIS_SECURITY_BITS,
                family,
                ring_dimension: source_d as u32,
                coeff_linf_bound: source_bucket,
            },
            8,
        )
        .expect("source key");
        let mut lp = LevelParams::params_only(
            family,
            gen_d,
            4,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(gen_d),
        );
        lp.b_key = source_key;
        (policy, lp)
    }

    fn assert_profile<F: CanonicalField>(
        family: SisModulusFamily,
        gen_d: usize,
        source_d: usize,
    ) -> CompressionSelection {
        let (policy, lp) = fixture(family, gen_d, source_d);
        let first = select_standalone_compression::<F>(
            &policy,
            CompressionPlannerPolicy::default(),
            &lp,
            synthetic_envelope(),
        )
        .expect("selection");
        let second = select_standalone_compression::<F>(
            &policy,
            CompressionPlannerPolicy::default(),
            &lp,
            synthetic_envelope(),
        )
        .expect("stable selection");
        assert_eq!(first.payload_bytes(), second.payload_bytes());
        assert_eq!(first.depth(), second.depth());
        assert_eq!(first.first_alphabet(), second.first_alphabet());
        assert_eq!(first.map_ring_dimensions(), second.map_ring_dimensions());
        assert_eq!(
            first.used_exhaustive_fallback(),
            second.used_exhaustive_fallback()
        );
        assert!(first.candidates_considered() > 0);
        assert!(first
            .map_ring_dimensions()
            .iter()
            .all(|&d| d.is_power_of_two()));
        first
    }

    #[test]
    fn rank_one_ring_dim_maps_bytes_by_field_element_size() {
        assert_eq!(rank_one_ring_dim_for_bytes(128, 16), Some(8));
        assert_eq!(rank_one_ring_dim_for_bytes(256, 16), Some(16));
        assert_eq!(rank_one_ring_dim_for_bytes(512, 8), Some(64));
        assert_eq!(rank_one_ring_dim_for_bytes(1024, 4), Some(256));
        assert_eq!(rank_one_ring_dim_for_bytes(130, 16), None);
        assert_eq!(rank_one_ring_dim_for_bytes(128, 0), None);
    }

    #[test]
    fn global_base_envelope_coalesces_prefix_before_later_tie_breaks() {
        let base_coeffs = 4096;
        let left = global_setup_objectives(base_coeffs, 64, 1024, &[(32, 16), (64, 8)])
            .expect("left objectives");
        let right = global_setup_objectives(base_coeffs, 64, 2048, &[(32, 16), (64, 32)])
            .expect("right objectives");
        assert_eq!(left, (4096, 4608));
        assert_eq!(right, left);

        let left_score = candidate_score(128, 128, left.0, left.1, 7, b"left");
        let right_score = candidate_score(128, 128, right.0, right.1, 9, b"right");
        assert!(left_score < right_score);
    }

    #[test]
    fn global_setup_prefix_rounds_to_whole_generation_rings() {
        assert_eq!(
            global_setup_objectives(64, 64, 96, &[]).expect("rounded prefix"),
            (128, 128)
        );
        assert!(global_setup_objectives(64, 0, 96, &[]).is_err());
        assert!(global_setup_objectives(64, 64, usize::MAX, &[]).is_err());
    }

    #[test]
    fn default_ladders_select_binary_256_128_across_field_tiers() {
        let q128 = assert_profile::<Prime128OffsetA7F7>(SisModulusFamily::Q128, 64, 32);
        let q64 = assert_profile::<Prime64Offset59>(SisModulusFamily::Q64, 128, 32);
        let q32 = assert_profile::<Prime32Offset99>(SisModulusFamily::Q32, 256, 64);
        assert_eq!(
            (
                q128.payload_bytes(),
                q128.target_met(),
                q128.depth(),
                q128.first_alphabet(),
                q128.map_ring_dimensions(),
                q128.selected_ladder(),
                q128.used_exhaustive_fallback(),
                q128.candidates_considered()
            ),
            (
                128,
                true,
                2,
                CompressionAlphabet::NegativeBinary,
                &[16, 8][..],
                Some(BINARY_256_128),
                false,
                1
            )
        );
        assert_eq!(
            (
                q64.payload_bytes(),
                q64.target_met(),
                q64.depth(),
                q64.first_alphabet(),
                q64.map_ring_dimensions(),
                q64.selected_ladder(),
                q64.used_exhaustive_fallback(),
                q64.candidates_considered()
            ),
            (
                128,
                true,
                2,
                CompressionAlphabet::NegativeBinary,
                &[32, 16][..],
                Some(BINARY_256_128),
                false,
                1
            )
        );
        assert_eq!(
            (
                q32.payload_bytes(),
                q32.target_met(),
                q32.depth(),
                q32.first_alphabet(),
                q32.map_ring_dimensions(),
                q32.selected_ladder(),
                q32.used_exhaustive_fallback(),
                q32.candidates_considered()
            ),
            (
                128,
                true,
                2,
                CompressionAlphabet::NegativeBinary,
                &[64, 32][..],
                Some(BINARY_256_128),
                false,
                1
            )
        );
        for selection in [&q128, &q64, &q32] {
            assert!(!selection.used_exhaustive_fallback());
            assert!(selection.candidates_considered() <= DEFAULT_COMPRESSION_BYTE_LADDERS.len());
        }
    }

    #[test]
    fn guided_candidate_ceiling_is_the_ladder_list_length() {
        let (policy, lp) = fixture(SisModulusFamily::Q128, 64, 32);
        let selection = select_standalone_compression::<Prime128OffsetA7F7>(
            &policy,
            CompressionPlannerPolicy::default(),
            &lp,
            synthetic_envelope(),
        )
        .expect("selection");
        assert!(!selection.used_exhaustive_fallback());
        assert!(selection.candidates_considered() <= DEFAULT_COMPRESSION_BYTE_LADDERS.len());
    }

    #[test]
    fn dead_end_ladder_advances_to_the_next_ladder() {
        let (policy, lp) = fixture(SisModulusFamily::Q128, 64, 32);
        // 64-byte terminal is not in compression dispatch as a preferred
        // rank-one image for this field (d=4 unsupported), so the first ladder
        // must miss and the second must win.
        const DEAD_THEN_BINARY: &[CompressionByteLadder] = &[
            CompressionByteLadder::new(CompressionFirstMapAlphabet::NegativeBinary, &[64, 32]),
            BINARY_256_128,
        ];
        let selection = select_standalone_compression::<Prime128OffsetA7F7>(
            &policy,
            CompressionPlannerPolicy::with_ladders(128, DEAD_THEN_BINARY, false),
            &lp,
            synthetic_envelope(),
        )
        .expect("second ladder");
        assert_eq!(selection.selected_ladder(), Some(BINARY_256_128));
        assert_eq!(selection.candidates_considered(), 2);
        assert!(!selection.used_exhaustive_fallback());
        assert_eq!(selection.map_ring_dimensions(), &[16, 8]);
    }

    fn wide_source_fixture() -> (PlannerPolicy, LevelParams) {
        let (policy, mut lp) = fixture(SisModulusFamily::Q128, 64, 32);
        // Widen the B image so an opening-base 512B first map needs rank > 1
        // at d=32 under b_range=4 (rank-one width ceiling 479).
        let source_bucket = rounded_up_collision_inf_norm(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q128,
            32,
            policy.basis_range.1,
        )
        .expect("source collision bucket");
        lp.b_key = AjtaiKeyParams::try_new(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q128,
            8,
            8,
            source_bucket,
            32,
        )
        .expect("wide source key");
        (policy, lp)
    }

    #[test]
    fn rank_greater_than_one_skips_ladder_instead_of_inflating() {
        let (policy, lp) = wide_source_fixture();
        let selection = select_standalone_compression::<Prime128OffsetA7F7>(
            &policy,
            CompressionPlannerPolicy::with_ladders(128, &[OPENING_BASE_512_256_128], true),
            &lp,
            synthetic_envelope(),
        )
        .expect("exhaustive after miss");
        assert!(selection.used_exhaustive_fallback());
        assert!(selection.selected_ladder().is_none());
    }

    #[test]
    fn disabling_exhaustive_fallback_errors_when_all_ladders_miss() {
        let (policy, lp) = wide_source_fixture();
        let err = select_standalone_compression::<Prime128OffsetA7F7>(
            &policy,
            CompressionPlannerPolicy::with_ladders(128, &[OPENING_BASE_512_256_128], false),
            &lp,
            synthetic_envelope(),
        )
        .expect_err("no silent inflation");
        assert!(format!("{err:?}").contains("exhaustive fallback is disabled"));
    }

    fn enumerated_candidates<F: CanonicalField>(
        policy: &PlannerPolicy,
        lp: &LevelParams,
    ) -> Vec<Candidate> {
        let tier = protocol_dispatch_tier::<F>();
        let dimensions = slot_dims_for_tier(tier, ProtocolDispatchSlot::Compression);
        let source_d = lp.b_key.sis_table_key().ring_dimension as usize;
        let source_output_coeffs = lp.b_key.row_len() * source_d;
        let materializer = Materializer::<F> {
            policy,
            lp,
            field_bits: F::modulus_bits(),
            field_element_bytes: field_bytes(F::modulus_bits()),
            base_setup_coeffs: policy.ring_dimension,
            dimensions,
            _field: core::marker::PhantomData,
        };
        materializer.enumerate_admitted_chains(source_output_coeffs)
    }

    #[test]
    fn enumerates_both_depths_and_first_alphabets_under_current_tables() {
        let (policy, lp) = fixture(SisModulusFamily::Q128, 64, 32);
        let candidates = enumerated_candidates::<Prime128OffsetA7F7>(&policy, &lp);
        assert!(candidates.len() <= MAX_EXHAUSTIVE_COMPRESSION_CANDIDATES);
        for depth in 2..=3 {
            for alphabet in [
                CompressionAlphabet::OpeningBase {
                    log_basis: STANDALONE_OPENING_BASE_LOG_BASIS,
                },
                CompressionAlphabet::NegativeBinary,
            ] {
                assert!(candidates.iter().any(|candidate| {
                    candidate.depth == depth && candidate.first_alphabet == alphabet
                }));
            }
        }
        assert!(candidates
            .iter()
            .flat_map(|candidate| &candidate.map_ring_dimensions)
            .all(|&d| d.is_power_of_two()));
    }

    #[test]
    fn opening_base_first_map_stays_frozen_when_later_opening_envelope_is_wider() {
        let (mut policy, mut lp) = fixture(SisModulusFamily::Q128, 64, 32);
        policy.basis_range = (2, 6);
        let source_bucket = rounded_up_collision_inf_norm(
            DEFAULT_SIS_SECURITY_BITS,
            SisModulusFamily::Q128,
            32,
            policy.basis_range.1,
        )
        .expect("wider envelope source bucket");
        lp.b_key = AjtaiKeyParams::try_new_with_min_rank(
            akita_types::SisTableKey {
                min_security_bits: DEFAULT_SIS_SECURITY_BITS,
                family: SisModulusFamily::Q128,
                ring_dimension: 32,
                coeff_linf_bound: source_bucket,
            },
            8,
        )
        .expect("wider envelope source key");
        let selection = select_standalone_compression::<Prime128OffsetA7F7>(
            &policy,
            CompressionPlannerPolicy::default(),
            &lp,
            synthetic_envelope(),
        )
        .expect("selection under wider later-opening envelope");
        let candidates = enumerated_candidates::<Prime128OffsetA7F7>(&policy, &lp);
        assert!(
            candidates.iter().any(|candidate| {
                candidate.first_alphabet
                    == CompressionAlphabet::OpeningBase {
                        log_basis: STANDALONE_OPENING_BASE_LOG_BASIS,
                    }
            }),
            "wider b_range must still admit frozen opening-base F1 (b_cmp=4), not retarget the alphabet"
        );
        assert!(matches!(
            selection.first_alphabet(),
            CompressionAlphabet::OpeningBase {
                log_basis: STANDALONE_OPENING_BASE_LOG_BASIS
            } | CompressionAlphabet::NegativeBinary
        ));
        assert!(!candidates.iter().any(|candidate| {
            matches!(
                candidate.first_alphabet,
                CompressionAlphabet::OpeningBase { log_basis }
                    if log_basis != STANDALONE_OPENING_BASE_LOG_BASIS
            )
        }));
    }

    #[test]
    fn rejects_non_shipped_security_floor_and_mismatched_authorities() {
        let (mut policy, lp) = fixture(SisModulusFamily::Q128, 64, 32);
        policy.min_sis_security_bits -= 1;
        assert!(select_standalone_compression::<Prime128OffsetA7F7>(
            &policy,
            CompressionPlannerPolicy::default(),
            &lp,
            synthetic_envelope()
        )
        .is_err());

        let (policy, mut lp) = fixture(SisModulusFamily::Q128, 64, 32);
        lp.ring_dimension = 128;
        assert!(select_standalone_compression::<Prime128OffsetA7F7>(
            &policy,
            CompressionPlannerPolicy::default(),
            &lp,
            synthetic_envelope()
        )
        .is_err());

        let (policy, lp) = fixture(SisModulusFamily::Q128, 64, 32);
        for max_setup_len in [0, usize::MAX] {
            assert!(select_standalone_compression::<Prime128OffsetA7F7>(
                &policy,
                CompressionPlannerPolicy::default(),
                &lp,
                SetupMatrixEnvelope { max_setup_len }
            )
            .is_err());
        }
    }

    #[test]
    fn reports_best_admissible_when_payload_target_is_unavailable() {
        let (policy, lp) = fixture(SisModulusFamily::Q128, 64, 32);
        let one_byte = select_standalone_compression::<Prime128OffsetA7F7>(
            &policy,
            CompressionPlannerPolicy::new(1),
            &lp,
            synthetic_envelope(),
        )
        .expect("best admissible");
        let zero_byte = select_standalone_compression::<Prime128OffsetA7F7>(
            &policy,
            CompressionPlannerPolicy::new(0),
            &lp,
            synthetic_envelope(),
        )
        .expect("zero target still selects the best admissible chain");
        assert!(!one_byte.target_met());
        assert!(!zero_byte.target_met());
        assert_eq!(zero_byte.payload_bytes(), one_byte.payload_bytes());
        assert_eq!(zero_byte.depth(), one_byte.depth());
        assert_eq!(
            zero_byte.map_ring_dimensions(),
            one_byte.map_ring_dimensions()
        );
        assert_eq!(one_byte.selected_ladder(), Some(BINARY_256_128));
    }
}
