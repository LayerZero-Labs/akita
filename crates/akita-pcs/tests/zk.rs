#![allow(missing_docs)]
#![cfg(all(feature = "zk", feature = "planner"))]

mod common;

use akita_algebra::CyclotomicRing;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::kernels::linear::mat_vec_mul_ntt_single_i8;
use akita_prover::{AkitaProverSetup, CommitmentProver, QuadraticEquation};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
use akita_transcript::{Blake2bTranscript, Transcript};
use akita_types::{
    AkitaBatchedProof, AkitaCommitmentHint, AkitaScheduleInputs, AkitaScheduleLookupKey,
    AkitaSchedulePlan, AkitaVerifierSetup, AppendToTranscript, ClaimIncidenceSummary,
    CommitmentEnvelope, DecompositionParams, RingCommitment, RingMultiplierOpeningPoint,
    ScheduleProvider, SisModulusFamily,
};
use akita_verifier::CommitmentVerifier;
use common::*;
use std::marker::PhantomData;

type Scheme<const D: usize, Cfg> = AkitaCommitmentScheme<D, Cfg>;

#[derive(Clone, Copy, Debug)]
struct RuntimePlanned<Cfg>(PhantomData<Cfg>);

impl<Cfg: CommitmentConfig> ScheduleProvider for RuntimePlanned<Cfg> {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("zk-runtime-planned/{key:?}")
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }
}

impl<Cfg: CommitmentConfig> akita_planner::PlannerConfig for RuntimePlanned<Cfg> {
    type PlannerField = Cfg::Field;

    const PLANNER_D: usize = Cfg::D;

    fn planner_field_bits() -> u32 {
        Cfg::decomposition().field_bits()
    }

    fn planner_sis_modulus_family() -> SisModulusFamily {
        Cfg::sis_modulus_family()
    }

    fn planner_stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
        <Self as CommitmentConfig>::stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        <Self as ScheduleProvider>::schedule_plan(key)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, akita_field::AkitaError> {
        <Self as CommitmentConfig>::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, akita_field::AkitaError> {
        akita_config::current_level_layout_with_log_basis::<Self>(inputs, log_basis)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, akita_field::AkitaError> {
        <Self as CommitmentConfig>::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        <Self as CommitmentConfig>::log_basis_search_range(inputs)
    }
}

impl<Cfg: CommitmentConfig> CommitmentConfig for RuntimePlanned<Cfg> {
    type Field = Cfg::Field;
    type ClaimField = Cfg::ClaimField;
    type ChallengeField = Cfg::ChallengeField;

    const D: usize = Cfg::D;

    fn decomposition() -> DecompositionParams {
        Cfg::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
        Cfg::stage1_challenge_config(d)
    }

    fn sis_modulus_family() -> SisModulusFamily {
        Cfg::sis_modulus_family()
    }

    fn audited_root_rank(role: akita_types::AjtaiRole, max_num_vars: usize) -> usize {
        Cfg::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Cfg::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        _max_num_batched_polys: usize,
        _max_num_points: usize,
    ) -> Result<(usize, usize), akita_field::AkitaError> {
        let envelope = Cfg::envelope(max_num_vars);
        let rows = envelope
            .max_n_a
            .max(envelope.max_n_b)
            .max(envelope.max_n_d)
            .max(4);
        Ok((rows, 16_384))
    }

    fn level_params_with_log_basis(inputs: AkitaScheduleInputs, log_basis: u32) -> LevelParams {
        Cfg::level_params_with_log_basis(inputs, log_basis)
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, akita_field::AkitaError> {
        Cfg::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, akita_field::AkitaError> {
        Cfg::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
        Cfg::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Cfg::log_basis_search_range(inputs)
    }
}

fn single_point_group_incidence(num_vars: usize) -> ClaimIncidenceSummary {
    ClaimIncidenceSummary {
        num_vars,
        num_points: 1,
        num_groups: 1,
        num_claims: 1,
        claim_to_point: vec![0],
        claim_to_group: vec![0],
        claim_poly_indices: vec![0],
        group_poly_counts: vec![1],
        group_claim_counts: vec![1],
        point_claim_counts: vec![1],
        point_group_counts: vec![1],
    }
}

fn plain_root_d_image<const D: usize>(
    setup: &AkitaProverSetup<F, D>,
    poly: &DensePoly<F, D>,
    point: &[F],
    layout: &LevelParams,
    commitment: &RingCommitment<F, D>,
    hint: AkitaCommitmentHint<F, D>,
    label: &'static [u8],
) -> Vec<CyclotomicRing<F, D>> {
    let alpha_bits = D.trailing_zeros() as usize;
    let outer_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        outer_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("ring opening point");
    let ring_multiplier_point = RingMultiplierOpeningPoint::from_base(&ring_opening_point);
    let (y_ring, w_folded) = poly.evaluate_and_fold_ring(
        &ring_multiplier_point.b,
        &ring_multiplier_point.a,
        layout.block_len,
    );

    let mut transcript = Blake2bTranscript::<F>::new(label);
    commitment.append_to_transcript(ABSORB_COMMITMENT, &mut transcript);
    for coord in point {
        transcript.append_field(ABSORB_EVALUATION_CLAIMS, coord);
    }
    transcript.append_serde(ABSORB_EVALUATION_CLAIMS, &y_ring);

    let quad_eq = QuadraticEquation::<F, D>::new_prover(
        &setup.ntt_shared,
        vec![ring_opening_point],
        vec![ring_multiplier_point],
        vec![0usize],
        &[poly],
        vec![w_folded],
        &single_point_group_incidence(point.len()),
        layout.clone(),
        vec![hint],
        &mut transcript,
        std::slice::from_ref(commitment),
        std::slice::from_ref(&y_ring),
        vec![F::one()],
        setup.expanded.seed.max_stride,
    )
    .expect("debug quadratic equation");

    assert!(
        quad_eq.d_blinding_digits().is_some(),
        "zk quadratic equation should sample D-blinding digits"
    );
    let plain_v = mat_vec_mul_ntt_single_i8(
        &setup.ntt_shared,
        layout.d_key.row_len(),
        setup.expanded.seed.max_stride,
        quad_eq.w_hat_flat().expect("debug w_hat"),
    );
    assert_ne!(
        quad_eq.v, plain_v,
        "debug zk v should include fresh D-blinding"
    );
    plain_v
}

fn assert_folded_v_hiding<const D: usize>(
    nv: usize,
    proof: &AkitaBatchedProof<F, F>,
    second_proof: &AkitaBatchedProof<F, F>,
    plain_root_v: &[CyclotomicRing<F, D>],
) {
    let root = proof
        .root
        .as_fold()
        .expect("fixture should use folded root");
    let second_root = second_proof
        .root
        .as_fold()
        .expect("second fixture should use folded root");
    assert_ne!(
        root.v, second_root.v,
        "zk root v should re-randomize for the same folded witness at D={D}, nv={nv}"
    );
    assert_ne!(
        root.v.to_vec::<D>(),
        plain_root_v,
        "zk root v should not expose the plain D * w_hat image at D={D}, nv={nv}"
    );

    let recursive_levels: Vec<_> = proof.fold_levels().collect();
    let second_recursive_levels: Vec<_> = second_proof.fold_levels().collect();
    assert!(
        !recursive_levels.is_empty(),
        "fixture should include recursive folded v coverage at D={D}, nv={nv}"
    );
    assert_eq!(
        recursive_levels.len(),
        second_recursive_levels.len(),
        "same fixture should produce the same number of recursive fold levels"
    );
    for (level_idx, (level, second_level)) in recursive_levels
        .iter()
        .zip(second_recursive_levels.iter())
        .enumerate()
    {
        assert_ne!(
            level.v, second_level.v,
            "zk recursive v should re-randomize at recursive level {level_idx} for D={D}, nv={nv}"
        );
    }
}

fn run_zk_dense_commitment_hiding<const D: usize, BaseCfg>(nv: usize, label: &'static [u8])
where
    BaseCfg: CommitmentConfig<Field = F, ClaimField = F>,
    RuntimePlanned<BaseCfg>: CommitmentConfig<Field = F, ClaimField = F>,
    Scheme<D, RuntimePlanned<BaseCfg>>: CommitmentProver<
            F,
            D,
            ProverSetup = AkitaProverSetup<F, D>,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            CommitHint = AkitaCommitmentHint<F, D>,
            ClaimField = F,
            BatchedProof = AkitaBatchedProof<F, F>,
        > + CommitmentVerifier<
            F,
            D,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            ClaimField = F,
            BatchedProof = AkitaBatchedProof<F, F>,
        >,
{
    type Cfg<Base> = RuntimePlanned<Base>;

    assert_eq!(BaseCfg::D, D);
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = Cfg::<BaseCfg>::commitment_layout(nv).expect("zk layout");
        let mut rng = StdRng::seed_from_u64(0x5eed_5eed_0000 + D as u64 + nv as u64);
        let evals: Vec<F> = (0..1usize << nv)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).expect("dense poly");
        let point = random_point(nv, 0x0bad_f00d_0000 + D as u64 + nv as u64);
        let expected_opening = opening_from_poly(&poly, &point, &layout);

        let setup = <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
        let verifier_setup =
            <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::setup_verifier(&setup);

        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) =
            <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::commit(commit_input, &setup)
                .expect("first zk commit");
        let (rerandomized_commitment, _) =
            <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::commit(commit_input, &setup)
                .expect("second zk commit");
        assert_ne!(
            commitment, rerandomized_commitment,
            "zk commitment should re-randomize for the same polynomial at D={D}, nv={nv}"
        );

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(label);
        let proof = <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("zk prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize zk proof");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize zk proof");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(label);
        <Scheme<D, Cfg<BaseCfg>> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("zk verify");
    });
}

fn run_zk_dense_v_hiding<const D: usize, BaseCfg>(nv: usize, label: &'static [u8])
where
    BaseCfg: CommitmentConfig<Field = F, ClaimField = F>,
    RuntimePlanned<BaseCfg>: CommitmentConfig<Field = F, ClaimField = F>,
    Scheme<D, RuntimePlanned<BaseCfg>>: CommitmentProver<
            F,
            D,
            ProverSetup = AkitaProverSetup<F, D>,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            CommitHint = AkitaCommitmentHint<F, D>,
            ClaimField = F,
            BatchedProof = AkitaBatchedProof<F, F>,
        > + CommitmentVerifier<
            F,
            D,
            VerifierSetup = AkitaVerifierSetup<F>,
            Commitment = RingCommitment<F, D>,
            ClaimField = F,
            BatchedProof = AkitaBatchedProof<F, F>,
        >,
{
    type Cfg<Base> = RuntimePlanned<Base>;

    assert_eq!(BaseCfg::D, D);
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = Cfg::<BaseCfg>::commitment_layout(nv).expect("zk layout");
        let mut rng = StdRng::seed_from_u64(0x5eed_5eed_0000 + D as u64 + nv as u64);
        let evals: Vec<F> = (0..1usize << nv)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).expect("dense poly");
        let point = random_point(nv, 0x0bad_f00d_0000 + D as u64 + nv as u64);
        let expected_opening = opening_from_poly(&poly, &point, &layout);

        let setup = <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
        let verifier_setup =
            <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::setup_verifier(&setup);

        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) =
            <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::commit(commit_input, &setup)
                .expect("first zk commit");

        let plain_root_v = plain_root_d_image::<D>(
            &setup,
            &poly,
            &point,
            &layout,
            &commitment,
            hint.clone(),
            b"zk-debug-plain-root-v",
        );

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];

        let mut prover_transcript = Blake2bTranscript::<F>::new(label);
        let proof = <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitments[0], hint.clone()),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("zk prove");

        let mut second_prover_transcript = Blake2bTranscript::<F>::new(label);
        let second_proof = <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitments[0], hint),
            &mut second_prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("second zk prove");
        assert_folded_v_hiding::<D>(nv, &proof, &second_proof, &plain_root_v);

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize zk proof");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize zk proof");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(label);
        <Scheme<D, Cfg<BaseCfg>> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("zk verify");
    });
}

#[test]
fn zk_dense_d32_commitments_rerandomize_and_verify() {
    run_zk_dense_commitment_hiding::<32, fp128::D32Full>(14, b"zk_commitment_dense_d32");
}

#[test]
fn zk_dense_d64_commitments_rerandomize_and_verify() {
    run_zk_dense_commitment_hiding::<64, fp128::D64Full>(15, b"zk_commitment_dense_d64");
}

#[test]
fn zk_dense_d128_commitments_rerandomize_and_verify() {
    run_zk_dense_commitment_hiding::<128, fp128::D128Full>(16, b"zk_commitment_dense_d128");
}

#[test]
fn zk_dense_d32_hides_folded_v_and_verifies() {
    run_zk_dense_v_hiding::<32, fp128::D32Full>(14, b"zk_v_dense_d32");
}

#[test]
fn zk_dense_d64_hides_folded_v_and_verifies() {
    run_zk_dense_v_hiding::<64, fp128::D64Full>(15, b"zk_v_dense_d64");
}

#[test]
fn zk_dense_d128_hides_folded_v_and_verifies() {
    run_zk_dense_v_hiding::<128, fp128::D128Full>(16, b"zk_v_dense_d128");
}
