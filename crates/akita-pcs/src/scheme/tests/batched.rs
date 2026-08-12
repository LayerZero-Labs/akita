use super::*;

#[test]
fn batched_commit_matches_individual_commits() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 1) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 7) as u64)).collect();
    let poly_a = DensePoly::<F>::from_field_evals(num_vars, D, &evals_a).unwrap();
    let poly_b = DensePoly::<F>::from_field_evals(num_vars, D, &evals_b).unwrap();
    let setup = Scheme::setup_prover(num_vars, 2).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let poly_groups = [std::slice::from_ref(&poly_a), std::slice::from_ref(&poly_b)];

    let (batched_commitments, batched_hints): (Vec<_>, Vec<_>) = poly_groups
        .iter()
        .map(|group| {
            Scheme::commit::<_, _>(
                &setup,
                group,
                &stack,
                akita_prover::GroupContext::scheduler_without_prior_groups(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .map(akita_prover::CommitOutput::into_parts)
        .unzip();
    let akita_prover::CommitOutput {
        committed_group: commitment_a,
        hint: hint_a,
    } = Scheme::commit::<_, _>(
        &setup,
        std::slice::from_ref(&poly_a),
        &stack,
        akita_prover::GroupContext::scheduler_without_prior_groups(),
    )
    .unwrap();
    let akita_prover::CommitOutput {
        committed_group: commitment_b,
        hint: hint_b,
    } = Scheme::commit::<_, _>(
        &setup,
        std::slice::from_ref(&poly_b),
        &stack,
        akita_prover::GroupContext::scheduler_without_prior_groups(),
    )
    .unwrap();

    assert_eq!(batched_commitments, vec![commitment_a, commitment_b]);
    assert_eq!(batched_hints, vec![hint_a, hint_b]);
}

#[test]
fn commit_rejects_mixed_group_arity() {
    let layout = singleton_layout::<Cfg>(16);
    let num_vars =
        layout.position_index_bits() + layout.block_index_bits() + D.trailing_zeros() as usize;
    let evals = vec![F::one(); 1usize << num_vars];
    let smaller_evals = vec![F::one(); 1usize << (num_vars - 1)];
    let poly = DensePoly::<F>::from_field_evals(num_vars, D, &evals).unwrap();
    let smaller = DensePoly::<F>::from_field_evals(num_vars - 1, D, &smaller_evals).unwrap();
    let setup = Scheme::setup_prover(num_vars, 2).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");

    // An empty prior-group prefix is unrepresentable, so no grouped context
    // can carry one. `PriorGroupProfiles` owns that rejection; see
    // `prior_group_profiles_reject_an_empty_prefix`.
    let error = Scheme::commit(
        &setup,
        &[poly, smaller],
        &stack,
        akita_prover::GroupContext::scheduler_without_prior_groups(),
    )
    .expect_err("one committed group must be homogeneous");
    assert!(matches!(error, AkitaError::InvalidInput(_)));
}
