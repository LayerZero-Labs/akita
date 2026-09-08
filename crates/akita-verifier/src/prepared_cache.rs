//! Target cache construction for persistent and recursive verifiers.

use akita_error::AkitaError;
use akita_types::{
    build_riscv64_scalar_q128_cache_artifact, dispatch_for_field, setup_seed_digest,
    AkitaVerifierSetup, FoldSchedule, PreparedVerifierNttCacheBinding, ScheduleRowDigest,
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
    let setup_seed_digest = setup_seed_digest(&setup.expanded.descriptor.setup_seed)
        .map_err(|error| AkitaError::InvalidSetup(format!("setup seed identity: {error}")))?;
    let binding = PreparedVerifierNttCacheBinding {
        setup_seed_digest,
        schedule_row_digest,
        setup_field_elements: setup.expanded.descriptor.num_field_elements,
    };
    dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        requirement.ring_dimension,
        |D| {
            let matrix = setup
                .expanded
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

        setup
            .install_trusted_prepared_verifier_ntt_cache(&artifact, selection.row_digest)
            .expect("install terminal cache");
        assert!(setup.verifier_ntt_cache_bytes().expect("cache bytes") > 0);
    }
}
