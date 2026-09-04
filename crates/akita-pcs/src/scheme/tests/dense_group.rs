use super::*;

type DenseGroupCfg = Cfg;

#[test]
fn dense_group_commit_freezes_scalar_s_profile() {
    const NUM_VARS: usize = 16;

    let scheme = workspace_scheme::<DenseGroupCfg>().expect("workspace schedule artifact");
    let setup = scheme.setup_prover(NUM_VARS, 1).expect("dense group setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared dense group setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("dense group stack");

    let evals = (0..1usize << NUM_VARS)
        .map(|index| F::from_u64((3 * index + 7) as u64))
        .collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(NUM_VARS, &evals).expect("dense polynomial");

    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint: _hint,
    } = scheme
        .commit(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("dense group commit");

    assert_eq!(
        commitment.profile.group,
        akita_types::PolynomialGroupLayout::new(NUM_VARS, 1)
    );
    assert_eq!(
        commitment.profile,
        catalog_profile(
            &scheme,
            akita_types::PolynomialGroupLayout::new(NUM_VARS, 1)
        )
    );
    assert_eq!(
        commitment.rows().count(),
        commitment.profile.outer.matrix.output_rank()
    );
}
