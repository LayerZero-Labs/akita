//! Terminal matrix NTT caches for persistent and recursive verifiers.

use std::any::Any;
use std::collections::BTreeMap;

use akita_error::AkitaError;
use akita_types::{
    build_riscv64_scalar_q128_cache_artifact, decode_riscv64_scalar_q128_cache, dispatch_for_field,
    prepare_ntt_cache, setup_seed_digest, AkitaVerifierSetup, FoldSchedule, NttCacheMode,
    PreparedNttCache, PreparedVerifierNttCacheBinding, ScheduleRowDigest,
};
use jolt_field::{CanonicalEncoding, Field};

pub(crate) const TERMINAL_I16_LOG_BASIS: u32 = 16;
pub(crate) const TERMINAL_I16_ABS_BOUND: u64 = 1 << (TERMINAL_I16_LOG_BASIS - 1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TerminalNttCacheRequirement {
    pub(crate) ring_dimension: usize,
    pub(crate) prefix_len: usize,
    pub(crate) width: usize,
}

pub(crate) fn terminal_ntt_cache_requirement(
    schedule: &FoldSchedule,
) -> Result<TerminalNttCacheRequirement, AkitaError> {
    let terminal = &schedule.terminal;
    let width = terminal.inner_width();
    let prefix_len = terminal
        .inner
        .matrix
        .output_rank()
        .checked_mul(width)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal A cache prefix overflow".into()))?;
    if width == 0 || prefix_len == 0 {
        return Err(AkitaError::InvalidSetup(
            "terminal A cache requirement is empty".into(),
        ));
    }
    Ok(TerminalNttCacheRequirement {
        ring_dimension: terminal.d_a(),
        prefix_len,
        width,
    })
}

/// One prepared terminal `A` prefix, erased over its ring dimension.
struct TerminalNttEntry {
    requirement: TerminalNttCacheRequirement,
    cache_bytes: usize,
    cache: Box<dyn Any + Send + Sync>,
}

/// Exact negacyclic terminal `A` prefixes, one per ring dimension.
///
/// Each entry covers the longest prefix and the widest row among the
/// requirements at its dimension. A shorter or narrower product reads a prefix
/// of the same entry, and exact CRT capacity is monotone in the row width.
#[derive(Default)]
pub(crate) struct TerminalNttCache {
    entries: BTreeMap<usize, TerminalNttEntry>,
}

impl core::fmt::Debug for TerminalNttCache {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_map()
            .entries(
                self.entries
                    .iter()
                    .map(|(ring_d, entry)| (ring_d, (entry.requirement, entry.cache_bytes))),
            )
            .finish()
    }
}

impl TerminalNttCache {
    /// Join requirements into one covering requirement per ring dimension.
    fn join(
        requirements: impl IntoIterator<Item = TerminalNttCacheRequirement>,
    ) -> BTreeMap<usize, TerminalNttCacheRequirement> {
        let mut joined = BTreeMap::<usize, TerminalNttCacheRequirement>::new();
        for requirement in requirements {
            joined
                .entry(requirement.ring_dimension)
                .and_modify(|covering| {
                    covering.prefix_len = covering.prefix_len.max(requirement.prefix_len);
                    covering.width = covering.width.max(requirement.width);
                })
                .or_insert(requirement);
        }
        joined
    }

    /// Prepare every entry from the setup's public matrix.
    pub(crate) fn prepare<F: Field + CanonicalEncoding>(
        setup: &AkitaVerifierSetup<F>,
        requirements: impl IntoIterator<Item = TerminalNttCacheRequirement>,
    ) -> Result<Self, AkitaError> {
        let mut entries = BTreeMap::new();
        for (ring_d, requirement) in Self::join(requirements) {
            let entry = dispatch_for_field!(
                akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
                F,
                ring_d,
                |D| {
                    let matrix = setup
                        .expanded()
                        .shared_matrix()
                        .ring_view::<D>(1, requirement.prefix_len)?;
                    let prepared = prepare_ntt_cache(
                        matrix,
                        NttCacheMode::ExactNegacyclic {
                            width: requirement.width,
                            rhs_abs_bound: TERMINAL_I16_ABS_BOUND,
                        },
                    )?;
                    Ok::<_, AkitaError>(TerminalNttEntry {
                        requirement,
                        cache_bytes: prepared.cache_bytes(),
                        cache: Box::new(prepared),
                    })
                }
            )?;
            entries.insert(ring_d, entry);
        }
        Ok(Self { entries })
    }

    /// Install one trusted scalar Q128 artifact as the only entry.
    ///
    /// The artifact must carry the setup and schedule identities and exactly
    /// the geometry of `requirement`.
    pub(crate) fn install_trusted<F: Field + CanonicalEncoding>(
        setup: &AkitaVerifierSetup<F>,
        requirement: TerminalNttCacheRequirement,
        schedule_row_digest: ScheduleRowDigest,
        artifact: &[u8],
    ) -> Result<Self, AkitaError> {
        let expected_binding = PreparedVerifierNttCacheBinding {
            setup_seed_digest: setup_seed_digest(&setup.expanded().descriptor.setup_seed).map_err(
                |error| AkitaError::InvalidSetup(format!("setup seed identity: {error}")),
            )?,
            schedule_row_digest,
            setup_field_elements: setup.expanded().descriptor.num_field_elements,
        };
        let entry = dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            requirement.ring_dimension,
            |D| {
                let (metadata, prepared) =
                    decode_riscv64_scalar_q128_cache::<F, D>(artifact, expected_binding)?;
                if metadata.base_prefix_len != requirement.prefix_len
                    || metadata.width != requirement.width
                    || metadata.rhs_abs_bound != TERMINAL_I16_ABS_BOUND
                {
                    return Err(AkitaError::InvalidSetup(
                        "trusted prepared terminal cache geometry does not match its schedule"
                            .into(),
                    ));
                }
                Ok::<_, AkitaError>(TerminalNttEntry {
                    requirement,
                    cache_bytes: prepared.cache_bytes(),
                    cache: Box::new(prepared),
                })
            }
        )?;
        Ok(Self {
            entries: BTreeMap::from([(requirement.ring_dimension, entry)]),
        })
    }

    /// Borrow the prepared entry for ring dimension `D`.
    pub(crate) fn get<const D: usize>(&self) -> Result<&PreparedNttCache<D>, AkitaError> {
        self.entries
            .get(&D)
            .ok_or_else(|| {
                AkitaError::InvalidSetup(format!(
                    "verifier has no prepared terminal matrix at ring dimension {D}"
                ))
            })?
            .cache
            .downcast_ref::<PreparedNttCache<D>>()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("prepared terminal matrix type mismatch".into())
            })
    }

    /// In-memory byte footprint of every prepared entry.
    pub(crate) fn cache_bytes(&self) -> usize {
        self.entries.values().map(|entry| entry.cache_bytes).sum()
    }
}

/// Build the scalar Q128 prepared terminal cache consumed by a RISC V verifier.
///
/// The returned bytes are derived performance state. They are not canonical
/// setup bytes and must be bound to the verifier program or another trusted
/// setup installation boundary before use.
pub fn build_riscv64_terminal_ntt_cache<F: Field + CanonicalEncoding>(
    setup: &AkitaVerifierSetup<F>,
    schedule: &FoldSchedule,
    schedule_row_digest: ScheduleRowDigest,
) -> Result<Vec<u8>, AkitaError> {
    let requirement = terminal_ntt_cache_requirement(schedule)?;
    let setup_seed_digest = setup_seed_digest(&setup.expanded().descriptor.setup_seed)
        .map_err(|error| AkitaError::InvalidSetup(format!("setup seed identity: {error}")))?;
    let binding = PreparedVerifierNttCacheBinding {
        setup_seed_digest,
        schedule_row_digest,
        setup_field_elements: setup.expanded().descriptor.num_field_elements,
    };
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        requirement.ring_dimension,
        |D| {
            let matrix = setup
                .expanded()
                .shared_matrix()
                .ring_view::<D>(1, requirement.prefix_len)?;
            build_riscv64_scalar_q128_cache_artifact(
                matrix,
                requirement.width,
                TERMINAL_I16_ABS_BOUND,
                binding,
            )
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::fp128::OneHot;
    use akita_types::{
        prepared_verifier_ntt_cache_metadata, AkitaExpandedSetup, AkitaScheduleLookupKey,
        AkitaSetupDescriptor, FlatMatrix, PolynomialGroupLayout, SetupPrefixVerifierRegistry,
    };
    use jolt_field::{Prime128Offset275 as F, Ring};
    use std::sync::Arc;

    #[test]
    fn terminal_builder_binds_the_resolved_schedule_and_installs() {
        std::thread::Builder::new()
            .name("terminal-cache-builder-test".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(terminal_builder_binds_the_resolved_schedule_and_installs_inner)
            .expect("spawn terminal cache builder test")
            .join()
            .expect("terminal cache builder test thread");
    }

    fn terminal_builder_binds_the_resolved_schedule_and_installs_inner() {
        let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>()
            .expect("workspace schedule catalog");
        let row = catalog
            .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                15, 1,
            )))
            .expect("workspace fp128 schedule");
        let selection = row.selection();
        let schedule = row.schedule();
        let requirement = terminal_ntt_cache_requirement(schedule).expect("terminal requirement");
        let setup_field_elements = requirement
            .prefix_len
            .checked_mul(requirement.ring_dimension)
            .expect("setup field count");
        let seed: akita_types::AkitaSetupSeed = [6; 32].into();
        let setup = AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupDescriptor {
                        max_num_vars: 15,
                        max_num_batched_polys: 1,
                        num_field_elements: setup_field_elements,
                        setup_seed: seed.clone(),
                    },
                    FlatMatrix::from_flat_data(vec![F::from_i64(1); setup_field_elements]),
                ),
            ),
            SetupPrefixVerifierRegistry::new(seed),
        )
        .expect("verifier setup");

        let artifact = build_riscv64_terminal_ntt_cache(&setup, schedule, selection.row_digest)
            .expect("terminal cache artifact");
        let metadata = prepared_verifier_ntt_cache_metadata(&artifact).expect("metadata");
        assert_eq!(metadata.ring_dimension, requirement.ring_dimension);
        assert_eq!(metadata.base_prefix_len, requirement.prefix_len);
        assert_eq!(metadata.width, requirement.width);
        assert_eq!(metadata.binding.schedule_row_digest, selection.row_digest);

        let installed =
            TerminalNttCache::install_trusted(&setup, requirement, selection.row_digest, &artifact)
                .expect("install terminal cache");
        assert!(installed.cache_bytes() > 0);

        let other_row = ScheduleRowDigest::from_bytes([0xa5; 32]);
        assert!(matches!(
            TerminalNttCache::install_trusted(&setup, requirement, other_row, &artifact),
            Err(AkitaError::InvalidSetup(_))
        ));
        let narrower = TerminalNttCacheRequirement {
            width: requirement.width - 1,
            ..requirement
        };
        assert!(matches!(
            TerminalNttCache::install_trusted(&setup, narrower, selection.row_digest, &artifact),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}
