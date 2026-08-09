use super::*;

type DenseGroupCfg = Cfg;
type DenseGroupScheme = AkitaCommitmentScheme<DenseGroupCfg>;

#[test]
fn dense_group_commit_freezes_uniform_precommit_profile() {
    const NUM_VARS: usize = 16;

    let setup = DenseGroupScheme::setup_prover(NUM_VARS, 1).expect("dense group setup");
    let prepared = CpuBackend
        .prepare_setup(&setup)
        .expect("prepared dense group setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("dense group stack");

    let evals = (0..1usize << NUM_VARS)
        .map(|index| F::from_u64((3 * index + 7) as u64))
        .collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(NUM_VARS, D, &evals).expect("dense polynomial");

    let (commitment, _hint) =
        DenseGroupScheme::commit_group(&setup, std::slice::from_ref(&poly), &stack)
            .expect("dense group commit");

    assert_eq!(
        commitment.profile.group,
        akita_types::PolynomialGroupLayout::new(NUM_VARS, 1)
    );
    assert_eq!(commitment.profile.inner_commit_matrix.ring_dimension(), 64);
    assert_eq!(commitment.profile.outer_commit_matrix.ring_dimension(), 64);
    assert_eq!(
        commitment.rows().count(),
        commitment.profile.outer_commit_matrix.output_rank()
    );
}
