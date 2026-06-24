//! End-to-end batched prove/verify with four distinct delegating cluster backends.

use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{
    batched_prove, CommitCluster, CommitmentProver, ComputeBackendSetup, CpuBackend, DensePoly,
    OpeningCluster, ProverCommitmentGroup, ProverComputeStack, ProverOpeningBatch,
    RingSwitchCluster, TensorCluster, UniformProverStack,
};
use akita_transcript::AkitaTranscript;
use akita_types::{
    lagrange_weights, BasisMode, CommitmentGroup, OpeningBatchShape, PointVariableSelection,
    VerifierOpeningBatch,
};
use akita_verifier::CommitmentVerifier;
use std::any::TypeId;

type Cfg = fp128::D64Full;
type F = fp128::Field;
const D: usize = Cfg::D;
type Scheme = AkitaCommitmentScheme<D, Cfg>;

fn assert_distinct_cluster_types() {
    let types = [
        TypeId::of::<CommitCluster>(),
        TypeId::of::<OpeningCluster>(),
        TypeId::of::<TensorCluster>(),
        TypeId::of::<RingSwitchCluster>(),
    ];
    for i in 0..types.len() {
        for j in (i + 1)..types.len() {
            assert_ne!(types[i], types[j], "cluster marker types must be distinct");
        }
    }
}

#[test]
fn heterogeneous_delegating_clusters_batched_prove_and_verify() {
    assert_distinct_cluster_types();

    const NUM_VARS: usize = 16;
    let opening_batch = OpeningBatchShape::new(NUM_VARS, 1).expect("opening batch");
    let layout = Cfg::get_params_for_batched_commitment(&opening_batch).expect("layout");
    let alpha = D.trailing_zeros() as usize;
    let full_num_vars = layout.m_vars + layout.r_vars + alpha;

    let len = 1usize << full_num_vars;
    let evals: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F, D>::from_field_evals(full_num_vars, &evals).unwrap();

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(full_num_vars, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");

    let commit_backend = CommitCluster;
    let opening = OpeningCluster;
    let tensor = TensorCluster;
    let ring = RingSwitchCluster;
    let stack = ProverComputeStack::new(
        (&commit_backend, &prepared),
        (&opening, &prepared),
        (&tensor, &prepared),
        (&ring, &prepared),
        setup.expanded.as_ref(),
    )
    .expect("heterogeneous stack");

    assert_eq!(
        stack.commit().backend() as *const _,
        &commit_backend as *const _
    );
    assert_eq!(stack.opening().backend() as *const _, &opening as *const _);
    assert_eq!(stack.tensor().backend() as *const _, &tensor as *const _);
    assert_eq!(stack.ring_switch().backend() as *const _, &ring as *const _);

    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let commit_stack = UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
        .expect("commit stack");
    let (commitment, hint) = akita_prover::commit::<Cfg, D, DensePoly<F, D>, CpuBackend>(
        std::slice::from_ref(&poly),
        setup.expanded.as_ref(),
        &commit_stack,
    )
    .expect("commit");

    let opening_point: Vec<F> = (0..full_num_vars)
        .map(|i| F::from_u64((i + 2) as u64))
        .collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |acc, (&coeff, &weight)| acc + coeff * weight);

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/heterogeneous-batched-prove");
    let proof = batched_prove::<Cfg, _, DensePoly<F, D>, _, _, _, _, D>(
        &setup.expanded,
        &setup.prefix_slots,
        &stack,
        ProverOpeningBatch {
            point: opening_point[..].into(),
            groups: vec![ProverCommitmentGroup {
                point_vars: PointVariableSelection::prefix(
                    opening_point.len(),
                    opening_point.len(),
                )
                .expect("full-point prover group"),
                polynomials: &poly_refs[..],
                commitment: (commitments[0].clone(), hint),
            }],
        },
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("heterogeneous batched prove");

    assert!(
        !proof.is_root_direct(),
        "fixture must exercise folded recursive prove, not root-direct"
    );

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/heterogeneous-batched-prove");
    <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        VerifierOpeningBatch::from_groups(
            opening_point.clone(),
            vec![CommitmentGroup {
                claims: vec![opening],
                commitment: &commitments[0],
            }],
        )
        .expect("valid verifier claims"),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("heterogeneous batched verify");
}
