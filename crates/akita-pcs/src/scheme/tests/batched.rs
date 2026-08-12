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
        .map(|group| Scheme::commit::<_, _>(&setup, group, &stack))
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .unzip();
    let (commitment_a, hint_a) =
        Scheme::commit::<_, _>(&setup, std::slice::from_ref(&poly_a), &stack).unwrap();
    let (commitment_b, hint_b) =
        Scheme::commit::<_, _>(&setup, std::slice::from_ref(&poly_b), &stack).unwrap();

    assert_eq!(batched_commitments, vec![commitment_a, commitment_b]);
    assert_eq!(batched_hints, vec![hint_a, hint_b]);
}
