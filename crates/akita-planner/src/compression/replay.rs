//! Checked replay of compact compression planner choices.

use akita_field::{AkitaError, CanonicalField};
use akita_types::sis::{rounded_up_collision_inf_norm, sis_table_key_for_linf_bound};
use akita_types::{
    compression_digit_depth, protocol_dispatch_tier, validate_compression_catalog, AjtaiKeyParams,
    CompressionAlphabet, CompressionCatalogContext, CompressionChainSpec, CompressionMapSpec,
    CompressionSourceId, LevelParams, ProtocolRingDispatchTierId, SisModulusFamily,
    ValidatedCompressionCatalog, DEFAULT_SIS_SECURITY_BITS,
};

use crate::PlannerPolicy;

/// Compact planner output replayed into the checked catalog before it can
/// affect protocol geometry. Generated schedules will store this shape at the
/// atomic schedule-schema cutover; until then it remains planner-internal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CompressionChainDescriptor {
    pub(super) source: CompressionSourceId,
    pub(super) maps: Vec<CompressionMapDescriptor>,
}

/// The only free choices in a compression map. Rank, width, collision bucket,
/// digit depth, and payload shape are derived during checked replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CompressionMapDescriptor {
    pub(super) ring_d: usize,
    pub(super) alphabet: CompressionAlphabet,
}

fn source_output_coeffs(
    lp: &LevelParams,
    source: CompressionSourceId,
) -> Result<usize, AkitaError> {
    let key = match source {
        CompressionSourceId::CurrentOuter => &lp.b_key,
        CompressionSourceId::PrecommittedOuter { index } => {
            &lp.precommitted_groups
                .get(index)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "compression descriptor precommitted source is out of range".into(),
                    )
                })?
                .b_key
        }
        CompressionSourceId::Opening => &lp.d_key,
    };
    key.row_len()
        .checked_mul(key.sis_table_key().ring_dimension as usize)
        .ok_or_else(|| AkitaError::InvalidSetup("compression source size overflow".into()))
}

pub(super) fn derive_compression_key(
    policy: &PlannerPolicy,
    range_log_basis: u32,
    alphabet: CompressionAlphabet,
    ring_d: usize,
    col_len: usize,
) -> Option<AjtaiKeyParams> {
    let ring_d = u32::try_from(ring_d).ok()?;
    let table_key = match alphabet {
        CompressionAlphabet::NegativeBinary => sis_table_key_for_linf_bound(
            policy.min_sis_security_bits,
            policy.sis_family,
            ring_d,
            1,
        )?,
        CompressionAlphabet::OpeningBase { .. } => {
            let coeff_linf_bound = rounded_up_collision_inf_norm(
                policy.min_sis_security_bits,
                policy.sis_family,
                ring_d as usize,
                range_log_basis,
            )?;
            akita_types::SisTableKey {
                min_security_bits: policy.min_sis_security_bits,
                family: policy.sis_family,
                ring_dimension: ring_d,
                coeff_linf_bound,
            }
        }
    };
    AjtaiKeyParams::try_new_with_min_rank(table_key, col_len).ok()
}

pub(super) fn validate_replay_policy<F: CanonicalField>(
    policy: &PlannerPolicy,
    lp: &LevelParams,
) -> Result<(), AkitaError> {
    if policy.min_sis_security_bits != DEFAULT_SIS_SECURITY_BITS {
        return Err(AkitaError::InvalidSetup(format!(
            "compression replay supports the shipped SIS security floor {DEFAULT_SIS_SECURITY_BITS}, got {}",
            policy.min_sis_security_bits
        )));
    }
    if policy.ring_dimension != lp.ring_dimension {
        return Err(AkitaError::InvalidSetup(format!(
            "compression replay generation dimension {} disagrees with level dimension {}",
            policy.ring_dimension, lp.ring_dimension
        )));
    }
    let field_family = match protocol_dispatch_tier::<F>() {
        ProtocolRingDispatchTierId::Fp128 => SisModulusFamily::Q128,
        ProtocolRingDispatchTierId::Fp64 => SisModulusFamily::Q64,
        ProtocolRingDispatchTierId::Fp32 => SisModulusFamily::Q32,
    };
    if policy.sis_family != field_family {
        return Err(AkitaError::InvalidSetup(format!(
            "compression replay SIS family {:?} disagrees with field family {field_family:?}",
            policy.sis_family
        )));
    }
    Ok(())
}

/// Replay compact planner choices through the existing SIS tables and the one
/// canonical catalog validator. This deliberately cannot invent an empty
/// catalog or accept stored ranks, widths, digit depths, or payload lengths.
pub(super) fn replay_compression_catalog<F: CanonicalField>(
    policy: &PlannerPolicy,
    lp: &LevelParams,
    context: CompressionCatalogContext<'_>,
    descriptors: &[CompressionChainDescriptor],
) -> Result<ValidatedCompressionCatalog, AkitaError> {
    validate_replay_policy::<F>(policy, lp)?;
    let range_log_basis = match context {
        CompressionCatalogContext::CoGeneratedLevel { .. } => lp.log_basis,
        CompressionCatalogContext::StandaloneCommitment {
            max_opening_log_basis,
        } => max_opening_log_basis,
    };
    let mut specs = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        let mut previous_output = source_output_coeffs(lp, descriptor.source)?;
        let mut maps = Vec::with_capacity(descriptor.maps.len());
        for map in &descriptor.maps {
            let digit_depth =
                compression_digit_depth(map.alphabet, F::modulus_bits(), range_log_basis)?;
            let input_coeffs = previous_output.checked_mul(digit_depth).ok_or_else(|| {
                AkitaError::InvalidSetup("compression replay input size overflow".into())
            })?;
            if map.ring_d == 0 || !input_coeffs.is_multiple_of(map.ring_d) {
                return Err(AkitaError::InvalidSetup(
                    "compression replay input is not divisible by its native dimension".into(),
                ));
            }
            let key = derive_compression_key(
                policy,
                range_log_basis,
                map.alphabet,
                map.ring_d,
                input_coeffs / map.ring_d,
            )
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "compression replay map is absent from the shipped SIS tables".into(),
                )
            })?;
            let audited_ring_d = key.sis_table_key().ring_dimension as usize;
            if audited_ring_d != map.ring_d {
                return Err(AkitaError::InvalidSetup(
                    "compression replay dimension disagrees with its audited SIS key".into(),
                ));
            }
            previous_output = key.row_len().checked_mul(audited_ring_d).ok_or_else(|| {
                AkitaError::InvalidSetup("compression replay output size overflow".into())
            })?;
            maps.push(CompressionMapSpec::new(key, map.alphabet));
        }
        specs.push(CompressionChainSpec::new(descriptor.source, maps));
    }
    validate_compression_catalog::<F>(lp, context, policy.ring_dimension, specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::sis::{rounded_up_collision_inf_norm, DEFAULT_SIS_SECURITY_BITS};
    use akita_types::{
        ChunkedWitnessCfg, DecompositionParams, OpeningClaimsLayout, SisModulusFamily,
    };

    fn fixture() -> (PlannerPolicy, LevelParams) {
        let policy = PlannerPolicy {
            ring_dimension: 64,
            decomposition: DecompositionParams {
                log_basis: 4,
                log_commit_bound: 1,
                log_open_bound: Some(4),
            },
            sis_family: SisModulusFamily::Q128,
            min_sis_security_bits: DEFAULT_SIS_SECURITY_BITS,
            ring_subfield_norm_bound: 1,
            claim_ext_degree: 1,
            chal_ext_degree: 1,
            basis_range: (4, 4),
            onehot_chunk_size: 1,
            witness_chunk: ChunkedWitnessCfg::default(),
        };
        let bucket =
            rounded_up_collision_inf_norm(DEFAULT_SIS_SECURITY_BITS, SisModulusFamily::Q128, 32, 4)
                .expect("source bucket");
        let source_key = AjtaiKeyParams::try_new_with_min_rank(
            akita_types::SisTableKey {
                min_security_bits: DEFAULT_SIS_SECURITY_BITS,
                family: SisModulusFamily::Q128,
                ring_dimension: 32,
                coeff_linf_bound: bucket,
            },
            8,
        )
        .expect("source key");
        let mut lp = LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            4,
            1,
            1,
            1,
            SparseChallengeConfig::pm1_only(64),
        )
        .with_decomp(1, 1, 1, 1, 0)
        .expect("relation-layout fixture");
        lp.b_key = source_key.clone();
        lp.d_key = source_key;
        (policy, lp)
    }

    fn depth_two(
        source: CompressionSourceId,
        first_alphabet: CompressionAlphabet,
    ) -> CompressionChainDescriptor {
        CompressionChainDescriptor {
            source,
            maps: vec![
                CompressionMapDescriptor {
                    ring_d: 32,
                    alphabet: first_alphabet,
                },
                CompressionMapDescriptor {
                    ring_d: 32,
                    alphabet: CompressionAlphabet::NegativeBinary,
                },
            ],
        }
    }

    #[test]
    fn requires_and_certifies_the_complete_co_generated_catalog() {
        let (policy, lp) = fixture();
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening layout");
        let descriptors = [
            depth_two(
                CompressionSourceId::CurrentOuter,
                CompressionAlphabet::OpeningBase {
                    log_basis: lp.log_basis,
                },
            ),
            depth_two(
                CompressionSourceId::Opening,
                CompressionAlphabet::OpeningBase {
                    log_basis: lp.log_basis,
                },
            ),
        ];
        let replay = || {
            replay_compression_catalog::<Prime128OffsetA7F7>(
                &policy,
                &lp,
                CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
                &descriptors,
            )
        };
        let first = replay().expect("complete checked catalog");
        let second = replay().expect("deterministic replay");
        assert_eq!(
            first
                .project_for_schedule()
                .expect("first projection")
                .descriptor_bytes(),
            second
                .project_for_schedule()
                .expect("second projection")
                .descriptor_bytes()
        );
        assert!(replay_compression_catalog::<Prime128OffsetA7F7>(
            &policy,
            &lp,
            CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
            &descriptors[..1],
        )
        .is_err());

        let mut unsupported = descriptors.clone();
        unsupported[0].maps[0].ring_d = 8;
        assert!(replay_compression_catalog::<Prime128OffsetA7F7>(
            &policy,
            &lp,
            CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
            &unsupported,
        )
        .is_err());

        #[cfg(target_pointer_width = "64")]
        {
            assert!(derive_compression_key(
                &policy,
                lp.log_basis,
                CompressionAlphabet::NegativeBinary,
                (u32::MAX as usize) + 1,
                1,
            )
            .is_none());

            let mut narrowing_alias = descriptors.clone();
            narrowing_alias[0].maps[0].ring_d = (u32::MAX as usize) + 1 + 32;
            assert!(replay_compression_catalog::<Prime128OffsetA7F7>(
                &policy,
                &lp,
                CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
                &narrowing_alias,
            )
            .is_err());
        }
    }

    #[test]
    fn direct_replay_validates_policy_field_security_and_generation_authorities() {
        let (policy, lp) = fixture();
        assert!(validate_replay_policy::<Prime128OffsetA7F7>(&policy, &lp).is_ok());

        let mut wrong_family = policy;
        wrong_family.sis_family = SisModulusFamily::Q64;
        assert!(validate_replay_policy::<Prime128OffsetA7F7>(&wrong_family, &lp).is_err());

        let mut wrong_security = policy;
        wrong_security.min_sis_security_bits += 1;
        assert!(validate_replay_policy::<Prime128OffsetA7F7>(&wrong_security, &lp).is_err());

        let mut wrong_dimension = policy;
        wrong_dimension.ring_dimension *= 2;
        assert!(validate_replay_policy::<Prime128OffsetA7F7>(&wrong_dimension, &lp).is_err());
    }
}
