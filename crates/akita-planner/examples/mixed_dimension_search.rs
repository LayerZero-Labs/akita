use akita_config::{
    policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
};
use akita_planner::find_schedule;
use akita_types::{AkitaScheduleLookupKey, CommittedGroupProfile, PolynomialGroupLayout};

fn print_schedule(label: &str, planned: &akita_types::PlannedFoldSchedule) {
    let schedule = &planned.schedule;
    let physical_setup = akita_types::setup_matrix_field_elements_for_schedule(schedule)
        .expect("physical setup capacity");
    println!("{label}");
    println!(
        "  objective: setup={} field elements ({} recomputed field elements), proof={} bytes",
        planned.estimate.estimated_num_setup_field_elements,
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
    let direct_policy = policy_of::<OneHot>();
    let direct_key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
    let direct = find_schedule(
        &direct_key,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &[],
        &direct_policy,
        OneHot::ring_challenge_config,
    )?;
    print_schedule(
        &format!("direct scalar mixed-D search (nv={num_vars})"),
        &direct,
    );

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let recursive_policy = policy_of::<Recursive>();
    let scalar_recursive_key =
        AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
    let scalar_recursive = find_schedule(
        &scalar_recursive_key,
        akita_config::honest_fold_policy_of::<Recursive>(),
        &[],
        &recursive_policy,
        Recursive::ring_challenge_config,
    )?;
    print_schedule("adaptive recursive scalar planner", &scalar_recursive);
    println!(
        "  offload_edges={}",
        scalar_recursive.estimate.selected_offload_edges,
    );

    let precommit_layout = PolynomialGroupLayout::singleton(16);
    let independent = find_schedule(
        &AkitaScheduleLookupKey::single(precommit_layout),
        akita_config::honest_fold_policy_of::<OneHot>(),
        &[],
        &direct_policy,
        OneHot::ring_challenge_config,
    )?;
    let descriptor = CommittedGroupProfile::try_from_params(
        precommit_layout,
        &independent.schedule.root.params.final_group.commitment,
    )?;
    let recursive_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![descriptor, descriptor],
    };
    let precommitted_honest_fold_policies = vec![
        akita_config::honest_fold_policy_of::<OneHot>(),
        akita_config::honest_fold_policy_of::<OneHot>(),
    ];
    let adaptive_recursive = find_schedule(
        &recursive_key,
        akita_config::honest_fold_policy_of::<Recursive>(),
        &precommitted_honest_fold_policies,
        &recursive_policy,
        Recursive::ring_challenge_config,
    )?;
    print_schedule("adaptive recursive grouped planner", &adaptive_recursive);
    println!(
        "  offload_edges={}",
        adaptive_recursive.estimate.selected_offload_edges,
    );
    Ok(())
}
