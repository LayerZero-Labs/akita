use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_prover::{
    prewarm_ntt_requirements, CpuBackend, NttExecutionRequirements, NttOperationCluster,
    UniformProverStack,
};
use akita_types::{FoldSchedule, NttTransformDomain};

/// Prewarm the exact cache union for the profile's shared CPU owner.
///
/// `from_prove_schedule` already includes root A-negacyclic work for relation
/// proving. Adding the root commitment routes therefore contributes only any
/// larger A prefix and the commit-only B-negacyclic domain after physical-owner
/// max-joining. This helper is intentionally limited to the uniform profile
/// stack; heterogeneous stacks require phase- and owner-specific preparation.
pub(crate) fn prewarm_uniform_profile_execution<F>(
    stack: &UniformProverStack<'_, F, CpuBackend>,
    schedule: &FoldSchedule,
) -> Result<(), AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let mut requirements = NttExecutionRequirements::from_prove_schedule(schedule)?;
    let root = &schedule.root.params;
    let final_layout = &root.final_group.commitment;
    requirements.add_matrix(
        0,
        NttOperationCluster::Commit,
        final_layout.inner_commit_matrix.ring_dimension(),
        final_layout.inner_commit_matrix.output_rank(),
        final_layout.inner_commit_matrix.input_width(),
        NttTransformDomain::Negacyclic,
    )?;
    requirements.add_matrix(
        0,
        NttOperationCluster::Commit,
        final_layout.outer_commit_matrix.ring_dimension(),
        final_layout.outer_commit_matrix.output_rank(),
        final_layout.outer_commit_matrix.input_width(),
        NttTransformDomain::Negacyclic,
    )?;
    for precommitted in &root.precommitted_groups {
        let layout = &precommitted.commitment.layout;
        requirements.add_matrix(
            0,
            NttOperationCluster::Commit,
            layout.inner_commit_matrix.ring_dimension(),
            layout.inner_commit_matrix.output_rank(),
            layout.inner_commit_matrix.input_width(),
            NttTransformDomain::Negacyclic,
        )?;
        requirements.add_matrix(
            0,
            NttOperationCluster::Commit,
            layout.outer_commit_matrix.ring_dimension(),
            layout.outer_commit_matrix.output_rank(),
            layout.outer_commit_matrix.input_width(),
            NttTransformDomain::Negacyclic,
        )?;
    }
    prewarm_ntt_requirements::<F, _>(stack, &requirements)
}
