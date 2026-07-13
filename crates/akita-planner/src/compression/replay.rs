//! Checked replay of compact compression planner choices.

use akita_field::{AkitaError, CanonicalField};
use akita_types::sis::{rounded_up_collision_inf_norm, sis_table_key_for_linf_bound};
use akita_types::{
    AjtaiKeyParams, CompressionAlphabet, CompressionCatalogContext, CompressionChainChoice,
    CompressionChoice, CompressionFChoice, CompressionMapChoice, CompressionSourceId,
    FrozenCompressionChainChoice, LevelParams, ValidatedCompressionCatalog,
    DEFAULT_SIS_SECURITY_BITS,
};

use crate::PlannerPolicy;

/// Compact planner output replayed into the checked catalog before it can
/// affect protocol geometry. Generated schedules will store this shape at the
/// atomic schedule-schema cutover; until then it remains planner-internal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CompressionChainDescriptor {
    pub(super) source: CompressionSourceId,
    pub(super) max_opening_log_basis: u32,
    pub(super) maps: Vec<CompressionMapDescriptor>,
}

/// The only free choices in a compression map. Rank, width, collision bucket,
/// digit depth, and payload shape are derived during checked replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CompressionMapDescriptor {
    pub(super) ring_d: usize,
    pub(super) alphabet: CompressionAlphabet,
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
    let field_family = akita_types::sis_family_for_field::<F>();
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
    let fixed = descriptors
        .iter()
        .map(|descriptor| {
            let maps = descriptor
                .maps
                .iter()
                .map(|map| {
                    Ok(CompressionMapChoice {
                        ring_d: u32::try_from(map.ring_d).map_err(|_| {
                            AkitaError::InvalidSetup(
                                "compression replay dimension exceeds u32".into(),
                            )
                        })?,
                        alphabet: map.alphabet,
                    })
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            match maps.as_slice() {
                [a, b] => Ok((
                    descriptor.source,
                    descriptor.max_opening_log_basis,
                    CompressionChainChoice::Two([*a, *b]),
                )),
                [a, b, c] => Ok((
                    descriptor.source,
                    descriptor.max_opening_log_basis,
                    CompressionChainChoice::Three([*a, *b, *c]),
                )),
                _ => Err(AkitaError::InvalidSetup(
                    "compression replay chain depth must be in 2..=3".into(),
                )),
            }
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let choice = match context {
        CompressionCatalogContext::CoGeneratedLevel { .. } => {
            let current = fixed
                .first()
                .filter(|(source, _, _)| *source == CompressionSourceId::CurrentOuter);
            let opening = fixed
                .last()
                .filter(|(source, _, _)| *source == CompressionSourceId::Opening);
            let (Some((_, current_max, current_chain)), Some((_, _, opening_chain))) =
                (current, opening)
            else {
                return Err(AkitaError::InvalidSetup(
                    "co-generated compression replay is missing current or opening source".into(),
                ));
            };
            let precommitted_outer = fixed[1..fixed.len() - 1]
                .iter()
                .enumerate()
                .map(|(index, (source, max, chain))| {
                    if *source != (CompressionSourceId::PrecommittedOuter { index }) {
                        return Err(AkitaError::InvalidSetup(
                            "compression replay precommitted sources are out of order".into(),
                        ));
                    }
                    freeze_f(lp, *source, *max, *chain)
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            CompressionChoice {
                f: CompressionFChoice {
                    current_outer: freeze_f(
                        lp,
                        CompressionSourceId::CurrentOuter,
                        *current_max,
                        *current_chain,
                    )?,
                    precommitted_outer: &precommitted_outer,
                },
                opening: Some(*opening_chain),
            }
            .replay::<F>(lp, context)
        }
        CompressionCatalogContext::StandaloneCommitment => {
            let [(CompressionSourceId::CurrentOuter, max, chain)] = fixed.as_slice() else {
                return Err(AkitaError::InvalidSetup(
                    "standalone compression replay requires exactly current outer".into(),
                ));
            };
            CompressionChoice {
                f: CompressionFChoice {
                    current_outer: freeze_f(lp, CompressionSourceId::CurrentOuter, *max, *chain)?,
                    precommitted_outer: &[],
                },
                opening: None,
            }
            .replay::<F>(lp, context)
        }
        CompressionCatalogContext::TerminalFold { .. } => {
            let Some((CompressionSourceId::CurrentOuter, current_max, current_chain)) =
                fixed.first()
            else {
                return Err(AkitaError::InvalidSetup(
                    "terminal-fold compression replay is missing current outer".into(),
                ));
            };
            let precommitted_outer = fixed[1..]
                .iter()
                .enumerate()
                .map(|(index, (source, max, chain))| {
                    if *source != (CompressionSourceId::PrecommittedOuter { index }) {
                        return Err(AkitaError::InvalidSetup(
                            "terminal-fold precommitted sources are out of order".into(),
                        ));
                    }
                    freeze_f(lp, *source, *max, *chain)
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            CompressionChoice {
                f: CompressionFChoice {
                    current_outer: freeze_f(
                        lp,
                        CompressionSourceId::CurrentOuter,
                        *current_max,
                        *current_chain,
                    )?,
                    precommitted_outer: &precommitted_outer,
                },
                opening: None,
            }
            .replay::<F>(lp, context)
        }
    }?;
    Ok(choice)
}

fn freeze_f(
    lp: &LevelParams,
    source: CompressionSourceId,
    max_opening_log_basis: u32,
    chain: CompressionChainChoice,
) -> Result<FrozenCompressionChainChoice, AkitaError> {
    let source_key = match source {
        CompressionSourceId::CurrentOuter => &lp.b_key,
        CompressionSourceId::PrecommittedOuter { index } => {
            &lp.precommitted_groups
                .get(index)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "compression precommitted source index is out of bounds".into(),
                    )
                })?
                .b_key
        }
        CompressionSourceId::Opening => {
            return Err(AkitaError::InvalidSetup(
                "opening compression is not a frozen F choice".into(),
            ));
        }
    };
    Ok(FrozenCompressionChainChoice::new(
        source_key,
        max_opening_log_basis,
        chain,
    ))
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
            max_opening_log_basis: 4,
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

    #[test]
    fn terminal_fold_adapter_preserves_current_source_and_frozen_envelope() {
        let (policy, lp) = fixture();
        let opening = OpeningClaimsLayout::new(4, 1).expect("opening layout");
        let descriptor = depth_two(
            CompressionSourceId::CurrentOuter,
            CompressionAlphabet::OpeningBase {
                log_basis: lp.log_basis,
            },
        );
        let context = CompressionCatalogContext::TerminalFold { opening: &opening };
        let catalog = replay_compression_catalog::<Prime128OffsetA7F7>(
            &policy,
            &lp,
            context,
            std::slice::from_ref(&descriptor),
        )
        .expect("terminal catalog");
        assert!(catalog.terminal_relation_layout().is_ok());
        assert!(catalog.co_generated_relation_layout().is_err());

        let mut wrong_source = descriptor;
        wrong_source.source = CompressionSourceId::Opening;
        assert!(replay_compression_catalog::<Prime128OffsetA7F7>(
            &policy,
            &lp,
            context,
            &[wrong_source],
        )
        .is_err());
    }
}
