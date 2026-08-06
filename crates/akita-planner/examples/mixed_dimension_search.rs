use akita_config::{
    policy_of,
    proof_optimized::fp128::{D64OneHot, MixedDimFp128OneHot},
    CommitmentConfig, RecursiveCommitmentConfig,
};
use akita_planner::{find_group_batch_schedule, find_schedule};
use akita_types::{AkitaScheduleLookupKey, CommittedGroupProfile, PolynomialGroupLayout};

fn print_schedule(
    label: &str,
    setup_generation_dimension: usize,
    planned: &akita_types::PlannedFoldSchedule,
) {
    let schedule = &planned.schedule;
    let physical_setup = akita_types::setup_matrix_field_elements_for_schedule(schedule)
        .expect("physical setup envelope");
    println!("{label}");
    println!(
        "  objective: setup={} generated D{} ring elements ({} physical field elements), proof={} bytes",
        planned.estimate.estimated_setup_envelope_ring_elements,
        setup_generation_dimension,
        physical_setup,
        planned
            .estimate
            .estimated_proof_payload_bytes()
            .expect("proof estimate"),
    );
    println!(
        "  L0: {:?}, ranks={}/{}/{}, input={}, output={}",
        schedule.root.params.final_group.commitment.role_dims(),
        schedule
            .root
            .params
            .final_group
            .commitment
            .inner_commit_matrix
            .output_rank(),
        schedule
            .root
            .params
            .final_group
            .commitment
            .outer_commit_matrix
            .output_rank(),
        schedule
            .root
            .params
            .final_group
            .commitment
            .open_commit_matrix
            .output_rank(),
        schedule.root.input_witness_len,
        schedule.root.output_witness_len,
    );
    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        println!(
            "  L{}: {:?}, ranks={}/{}/{}, input={}, output={}",
            index + 1,
            fold.params.witness.role_dims(),
            fold.params.witness.inner_commit_matrix.output_rank(),
            fold.params.witness.outer_commit_matrix.output_rank(),
            fold.params.witness.open_commit_matrix.output_rank(),
            fold.input_witness_len,
            fold.output_witness_len,
        );
    }
    println!(
        "  L{} terminal: D{}, rank={}, input={}",
        schedule.recursive_folds.len() + 1,
        schedule.terminal.params.witness.d_a(),
        schedule
            .terminal
            .params
            .witness
            .inner_commit_matrix
            .output_rank(),
        schedule.terminal.input_witness_len,
    );
}

fn main() -> Result<(), akita_field::AkitaError> {
    let num_vars = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(36);
    let direct_policy = policy_of::<MixedDimFp128OneHot>();
    let direct_key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
    let direct = find_schedule(
        direct_key.final_group,
        &direct_policy,
        MixedDimFp128OneHot::root_honest_fold_policy(),
        MixedDimFp128OneHot::ring_challenge_config,
        MixedDimFp128OneHot::fold_challenge_shape_at_level,
    )?;
    print_schedule(
        &format!("direct scalar mixed-D search (nv={num_vars})"),
        direct_policy.ring_dimension,
        &direct,
    );

    type MixedRecursive = RecursiveCommitmentConfig<MixedDimFp128OneHot>;
    let mixed_recursive_policy = policy_of::<MixedRecursive>();
    let mixed_recursive_key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(32, 2));
    let mixed_recursive = find_group_batch_schedule(
        &mixed_recursive_key,
        MixedRecursive::root_honest_fold_policy(),
        &[],
        &mixed_recursive_policy,
        MixedRecursive::ring_challenge_config,
        MixedRecursive::fold_challenge_shape_at_level,
    )?;
    println!("recursive policy scalar dispatch through grouped planner:");
    println!(
        "  setup={} D{} ring elements, proof={} bytes",
        mixed_recursive
            .estimate
            .estimated_setup_envelope_ring_elements,
        mixed_recursive_policy.ring_dimension,
        mixed_recursive.estimate.estimated_proof_payload_bytes()?,
    );

    let precommit_layout = PolynomialGroupLayout::singleton(16);
    let precommit_key = AkitaScheduleLookupKey::single(precommit_layout);
    let precommit_policy = policy_of::<D64OneHot>();
    let precommit = find_schedule(
        precommit_key.final_group,
        &precommit_policy,
        D64OneHot::root_honest_fold_policy(),
        D64OneHot::ring_challenge_config,
        D64OneHot::fold_challenge_shape_at_level,
    )?;
    let descriptor = CommittedGroupProfile::from_params(
        precommit_layout,
        &precommit.schedule.root.params.final_group.commitment,
    );
    let recursive_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![descriptor, descriptor],
    };
    let precommitted_honest_fold_policies = vec![
        D64OneHot::root_honest_fold_policy(),
        D64OneHot::root_honest_fold_policy(),
    ];
    type Recursive = RecursiveCommitmentConfig<D64OneHot>;
    let recursive_policy = policy_of::<Recursive>();
    let preserved = find_group_batch_schedule(
        &recursive_key,
        Recursive::root_honest_fold_policy(),
        &precommitted_honest_fold_policies,
        &recursive_policy,
        Recursive::ring_challenge_config,
        Recursive::fold_challenge_shape_at_level,
    )?;
    println!("preserved recursive grouped planner:");
    println!(
        "  setup={} D64 ring elements, proof={} bytes, levels={}, offload_edges={}",
        preserved.estimate.estimated_setup_envelope_ring_elements,
        preserved.estimate.estimated_proof_payload_bytes()?,
        preserved.schedule.recursive_folds.len() + 2,
        preserved.estimate.selected_offload_edges,
    );
    Ok(())
}
