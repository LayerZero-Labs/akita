use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_prover::{
    prewarm_ntt_requirements, CpuBackend, NttExecutionRequirements, UniformProverStack,
};
use akita_types::FoldSchedule;

/// Prewarm the exact cache union for the profile's shared CPU owner.
///
/// `from_prove_schedule` includes recursive setup-prefix commitment work as
/// well as root A-negacyclic work for relation proving. Adding the root
/// commitment routes therefore contributes only any larger A prefix and the
/// commit-only B-negacyclic domain after physical-owner max-joining. This
/// helper is intentionally limited to the uniform profile stack; heterogeneous
/// stacks require phase- and owner-specific preparation.
pub(crate) fn prewarm_uniform_profile_execution<F>(
    stack: &UniformProverStack<'_, F, CpuBackend>,
    schedule: &FoldSchedule,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let requirements = NttExecutionRequirements::from_commit_and_prove_schedule(schedule)?;
    prewarm_ntt_requirements::<F, _>(stack, &requirements)
}
