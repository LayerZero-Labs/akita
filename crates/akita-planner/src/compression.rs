//! Deterministic standalone commitment-compression candidate selection.

mod replay;

use akita_field::{AkitaError, CanonicalField};
use akita_types::{
    compression_digit_depth, field_bytes, protocol_dispatch_tier, slot_dims_for_tier,
    AjtaiKeyParams, CompressionAlphabet, CompressionCatalogContext, CompressionSourceId,
    LevelParams, ProtocolDispatchSlot, SetupMatrixEnvelope, ValidatedCompressionCatalog,
};

use crate::PlannerPolicy;
use replay::{
    derive_compression_key, replay_compression_catalog, validate_replay_policy,
    CompressionChainDescriptor, CompressionMapDescriptor,
};

/// Standalone compression objectives that do not participate in the generated
/// schedule catalog identity until the planner cutover integrates compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressionPlannerPolicy {
    target_payload_bytes: usize,
}

impl CompressionPlannerPolicy {
    #[must_use]
    pub const fn new(target_payload_bytes: usize) -> Self {
        Self {
            target_payload_bytes,
        }
    }

    #[must_use]
    pub const fn target_payload_bytes(self) -> usize {
        self.target_payload_bytes
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

struct Enumeration<'a, F> {
    policy: &'a PlannerPolicy,
    lp: &'a LevelParams,
    field_bits: u32,
    field_element_bytes: usize,
    base_setup_coeffs: usize,
    dimensions: &'a [usize],
    candidates: Vec<Candidate>,
    _field: core::marker::PhantomData<F>,
}

impl<F: CanonicalField> Enumeration<'_, F> {
    fn key_for(
        &self,
        alphabet: CompressionAlphabet,
        d: usize,
        col_len: usize,
    ) -> Option<AjtaiKeyParams> {
        derive_compression_key(self.policy, self.policy.basis_range.1, alphabet, d, col_len)
    }

    fn extend_chain(
        &mut self,
        target_depth: usize,
        first_alphabet: CompressionAlphabet,
        previous_output_coeffs: usize,
        maps: &mut Vec<CompressionMapDescriptor>,
    ) {
        if maps.len() == target_depth {
            self.finish_candidate(first_alphabet, maps);
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
            self.extend_chain(target_depth, first_alphabet, output_coeffs, maps);
            maps.pop();
        }
    }

    fn finish_candidate(
        &mut self,
        first_alphabet: CompressionAlphabet,
        maps: &[CompressionMapDescriptor],
    ) {
        let descriptor = CompressionChainDescriptor {
            source: CompressionSourceId::CurrentOuter,
            max_opening_log_basis: self.policy.basis_range.1,
            maps: maps.to_vec(),
        };
        let Ok(catalog) = replay_compression_catalog::<F>(
            self.policy,
            self.lp,
            CompressionCatalogContext::StandaloneCommitment,
            &[descriptor],
        ) else {
            return;
        };
        let Ok(projection) = catalog.project_for_schedule() else {
            return;
        };
        let Some(payload_coeffs) = projection.payload_coeffs(CompressionSourceId::CurrentOuter)
        else {
            return;
        };
        let Some(payload_bytes) = payload_coeffs.checked_mul(self.field_element_bytes) else {
            return;
        };
        let Ok((global_setup_prefix_coeffs, global_cache_field_coeffs)) = global_setup_objectives(
            self.base_setup_coeffs,
            self.policy.ring_dimension,
            projection.max_flat_setup_prefix_coeffs(),
            projection.ntt_requirements(),
        ) else {
            return;
        };
        self.candidates.push(Candidate {
            catalog,
            payload_bytes,
            global_setup_prefix_coeffs,
            global_cache_field_coeffs,
            logical_setup_coeffs: projection.logical_setup_coeffs(),
            descriptor_bytes: projection.descriptor_bytes().to_vec(),
            depth: maps.len(),
            first_alphabet,
            map_ring_dimensions: maps.iter().map(|map| map.ring_d).collect(),
        });
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

/// Select the deterministic best standalone compression chain under current
/// dispatch and generated SIS-table coverage.
///
/// `setup_envelope` is the existing A/B/D envelope certified by the integrated
/// schedule/configuration caller. This selector checks its coefficient
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
    let mut enumeration = Enumeration::<F> {
        policy,
        lp,
        field_bits: F::modulus_bits(),
        field_element_bytes: field_bytes(F::modulus_bits()),
        base_setup_coeffs,
        dimensions,
        candidates: Vec::new(),
        _field: core::marker::PhantomData,
    };
    // The search space is protocol-fixed, never input-sized: two first-map
    // alphabets, depths 2/3, and at most four dispatch dimensions per tier
    // today (at most 2 * (4^2 + 4^3) = 160 raw dimension chains).
    for depth in 2..=3 {
        for first_alphabet in [
            CompressionAlphabet::OpeningBase { log_basis: 4 },
            CompressionAlphabet::NegativeBinary,
        ] {
            enumeration.extend_chain(
                depth,
                first_alphabet,
                source_output_coeffs,
                &mut Vec::with_capacity(depth),
            );
        }
    }
    let candidates_considered = enumeration.candidates.len();
    let target = compression_policy.target_payload_bytes();
    let best = enumeration
        .candidates
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
    Ok(CompressionSelection {
        target_met: best.payload_bytes <= target,
        catalog: best.catalog,
        payload_bytes: best.payload_bytes,
        depth: best.depth,
        first_alphabet: best.first_alphabet,
        map_ring_dimensions: best.map_ring_dimensions,
        candidates_considered,
    })
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
        assert!(first.candidates_considered() > 0);
        assert!(first.map_ring_dimensions().iter().all(|&d| d >= 32));
        first
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
    fn synthetic_sis_table_profiles_select_stable_q128_q64_q32_goldens() {
        // These isolate current dispatch/SIS-table behavior with synthetic
        // LevelParams. Real preset goldens belong to the schedule cutover that
        // supplies certified envelopes and integrated successor geometry.
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
                q128.candidates_considered()
            ),
            (
                512,
                false,
                2,
                CompressionAlphabet::OpeningBase { log_basis: 4 },
                &[32, 32][..],
                24
            )
        );
        assert_eq!(
            (
                q64.payload_bytes(),
                q64.target_met(),
                q64.depth(),
                q64.first_alphabet(),
                q64.map_ring_dimensions(),
                q64.candidates_considered()
            ),
            (
                256,
                false,
                2,
                CompressionAlphabet::NegativeBinary,
                &[32, 32][..],
                72
            )
        );
        assert_eq!(
            (
                q32.payload_bytes(),
                q32.target_met(),
                q32.depth(),
                q32.first_alphabet(),
                q32.map_ring_dimensions(),
                q32.candidates_considered()
            ),
            (
                256,
                false,
                2,
                CompressionAlphabet::OpeningBase { log_basis: 4 },
                &[32, 64][..],
                160
            )
        );
    }

    fn enumerated_candidates<F: CanonicalField>(
        policy: &PlannerPolicy,
        lp: &LevelParams,
    ) -> Vec<Candidate> {
        let tier = protocol_dispatch_tier::<F>();
        let dimensions = slot_dims_for_tier(tier, ProtocolDispatchSlot::Compression);
        let source_d = lp.b_key.sis_table_key().ring_dimension as usize;
        let source_output_coeffs = lp.b_key.row_len() * source_d;
        let mut enumeration = Enumeration::<F> {
            policy,
            lp,
            field_bits: F::modulus_bits(),
            field_element_bytes: field_bytes(F::modulus_bits()),
            base_setup_coeffs: policy.ring_dimension,
            dimensions,
            candidates: Vec::new(),
            _field: core::marker::PhantomData,
        };
        for depth in 2..=3 {
            for first_alphabet in [
                CompressionAlphabet::OpeningBase { log_basis: 4 },
                CompressionAlphabet::NegativeBinary,
            ] {
                enumeration.extend_chain(
                    depth,
                    first_alphabet,
                    source_output_coeffs,
                    &mut Vec::with_capacity(depth),
                );
            }
        }
        enumeration.candidates
    }

    #[test]
    fn enumerates_both_depths_and_first_alphabets_under_current_tables() {
        let (policy, lp) = fixture(SisModulusFamily::Q128, 64, 32);
        let candidates = enumerated_candidates::<Prime128OffsetA7F7>(&policy, &lp);
        for depth in 2..=3 {
            for alphabet in [
                CompressionAlphabet::OpeningBase { log_basis: 4 },
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
            .all(|&d| d >= 32));
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
    }
}
