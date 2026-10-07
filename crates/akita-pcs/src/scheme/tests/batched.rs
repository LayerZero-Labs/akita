use super::*;

#[test]
fn batched_commit_matches_individual_commits() {
    let scheme = workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout(&scheme, 16);
    let num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 1) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 7) as u64)).collect();
    let poly_a = DensePoly::<F>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = scheme.setup_prover(num_vars, 2).unwrap();
    let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
    let poly_groups = [std::slice::from_ref(&poly_a), std::slice::from_ref(&poly_b)];

    let (batched_commitments, _batched_handles): (Vec<_>, Vec<_>) = poly_groups
        .iter()
        .map(|group| {
            stack.commit(
                scheme.schedules(),
                &stack.import_source(group.to_vec()).expect("source"),
                akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .map(|output| (output.committed_group, output.private_handle))
        .unzip();
    let akita_cpu_backend::CommitOutput {
        committed_group: commitment_a,
        private_handle: _hint_a,
    } = stack
        .commit(
            scheme.schedules(),
            &stack.import_source(vec![poly_a.clone()]).expect("source"),
            akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let akita_cpu_backend::CommitOutput {
        committed_group: commitment_b,
        private_handle: _hint_b,
    } = stack
        .commit(
            scheme.schedules(),
            &stack.import_source(vec![poly_b.clone()]).expect("source"),
            akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();

    assert_eq!(batched_commitments, vec![commitment_a, commitment_b]);
}

#[test]
fn commit_rejects_mixed_group_arity() {
    let scheme = workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    let layout = singleton_layout(&scheme, 16);
    let num_vars =
        layout.position_index_bits() + layout.block_index_bits() + D.trailing_zeros() as usize;
    let evals = vec![F::one(); 1usize << num_vars];
    let smaller_evals = vec![F::one(); 1usize << (num_vars - 1)];
    let poly = DensePoly::<F>::from_field_evals(num_vars, &evals).unwrap();
    let smaller = DensePoly::<F>::from_field_evals(num_vars - 1, &smaller_evals).unwrap();
    let setup = scheme.setup_prover(num_vars, 2).unwrap();
    let stack = CpuBackend::<F, F>::new(setup.expanded.clone()).expect("backend");

    // An empty precommitted group prefix is unrepresentable, so no grouped context
    // can carry one. `PrecommittedGroupProfiles` owns that rejection; see
    // `precommitted_group_profiles_reject_an_empty_prefix`.
    let error = stack
        .import_source(vec![poly, smaller])
        .expect_err("one imported group must be homogeneous");
    assert!(matches!(error, AkitaError::InvalidInput(_)));
}
