#![allow(missing_docs)]
#![cfg(feature = "zk")]

mod common;

use akita_algebra::CyclotomicRing;
use akita_config::proof_optimized::fp32;
use akita_config::CommitmentConfig;
use akita_field::{ExtField, LiftBase};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::kernels::linear::mat_vec_mul_ntt_single_i8;
use akita_prover::{AkitaProverSetup, CommitmentProver, QuadraticEquation};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::labels::{ABSORB_COMMITMENT, ABSORB_EVALUATION_CLAIMS};
use akita_transcript::{AkitaTranscript, Transcript};
use akita_types::{
    lagrange_weights, AkitaBatchedProof, AkitaBatchedRootProof, AkitaCommitmentHint,
    AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, AkitaVerifierSetup,
    AppendToTranscript, ClaimIncidenceSummary, CommitmentEnvelope, DecompositionParams,
    FlatRingVec, MRowLayout, RingCommitment, RingMultiplierOpeningPoint, SetupMatrixEnvelope,
    SisModulusFamily,
};
use akita_verifier::CommitmentVerifier;
use common::*;
use std::marker::PhantomData;

type Scheme<const D: usize, Cfg> = AkitaCommitmentScheme<D, Cfg>;

#[derive(Clone, Copy, Debug)]
struct RuntimePlanned<Cfg>(PhantomData<Cfg>);

impl<Cfg: CommitmentConfig> CommitmentConfig for RuntimePlanned<Cfg> {
    type Field = Cfg::Field;
    type ClaimField = Cfg::ClaimField;
    type ChallengeField = Cfg::ChallengeField;

    const D: usize = Cfg::D;

    fn decomposition() -> DecompositionParams {
        Cfg::decomposition()
    }

    fn stage1_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, akita_field::AkitaError> {
        Cfg::stage1_challenge_config(d)
    }

    fn sis_modulus_family() -> SisModulusFamily {
        Cfg::sis_modulus_family()
    }

    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
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
    ) -> Result<SetupMatrixEnvelope, akita_field::AkitaError> {
        let envelope = Cfg::envelope(max_num_vars);
        let rows = envelope
            .max_n_a
            .max(envelope.max_n_b)
            .max(envelope.max_n_d)
            .max(4);
        SetupMatrixEnvelope::from_rows_stride(rows, 16_384)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Cfg::log_basis_search_range(inputs)
    }
}

fn single_point_group_incidence(num_vars: usize) -> ClaimIncidenceSummary {
    ClaimIncidenceSummary::same_point(num_vars, 1).expect("valid single-point incidence")
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
    let (y_ring, w_folded) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );

    let mut transcript = AkitaTranscript::<F>::new(label);
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
        vec![CyclotomicRing::<F, D>::one()],
        setup.expanded.seed.max_stride,
        MRowLayout::Intermediate,
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

fn bumped_flat_ring_vec(flat: &FlatRingVec<F>) -> FlatRingVec<F> {
    let mut coeffs = flat.coeffs().to_vec();
    assert!(
        !coeffs.is_empty(),
        "fixture should expose a non-empty flat ring vector"
    );
    coeffs[0] += F::one();
    FlatRingVec::from_coeffs(coeffs)
}

fn tamper_first_stage1_child_claim(proof: &mut AkitaBatchedProof<F, F>) {
    if let Some(root) = proof.root.as_fold_mut() {
        for stage in &mut root.stage1.stages {
            if let Some(claim) = stage.child_claims.first_mut() {
                *claim += F::one();
                return;
            }
        }
    }
    for step in &mut proof.steps {
        if let Some(level) = step.as_intermediate_mut() {
            for stage in &mut level.stage1.stages {
                if let Some(claim) = stage.child_claims.first_mut() {
                    *claim += F::one();
                    return;
                }
            }
        }
    }
    panic!("fixture should expose at least one stage-1 child claim");
}

fn tamper_first_stage1_round_coeff(proof: &mut AkitaBatchedProof<F, F>) {
    if let Some(root) = proof.root.as_fold_mut() {
        for stage in &mut root.stage1.stages {
            for round in &mut stage.sumcheck_proof_masked.masked_round_polys {
                if let Some(coeff) = round.coeffs_except_linear_term.first_mut() {
                    *coeff += F::one();
                    return;
                }
            }
        }
    }
    for step in &mut proof.steps {
        if let Some(level) = step.as_intermediate_mut() {
            for stage in &mut level.stage1.stages {
                for round in &mut stage.sumcheck_proof_masked.masked_round_polys {
                    if let Some(coeff) = round.coeffs_except_linear_term.first_mut() {
                        *coeff += F::one();
                        return;
                    }
                }
            }
        }
    }
    panic!("fixture should expose a masked stage-1 round coefficient");
}

fn tamper_first_stage2_round_coeff(proof: &mut AkitaBatchedProof<F, F>) {
    let root = proof
        .root
        .as_fold_mut()
        .expect("fixture should use a folded root");
    let round = root
        .stage2
        .sumcheck_proof_masked
        .masked_round_polys
        .iter_mut()
        .find(|round| !round.coeffs_except_linear_term.is_empty())
        .expect("fixture should expose a masked stage-2 round coefficient");
    round.coeffs_except_linear_term[0] += F::one();
}

fn random_fp32_extension_point(nv: usize, seed: u64) -> Vec<fp32::ExtensionField> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv)
        .map(|_| {
            let limbs = (0..<fp32::ExtensionField as ExtField<fp32::Field>>::EXT_DEGREE)
                .map(|_| fp32::Field::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            <fp32::ExtensionField as ExtField<fp32::Field>>::from_base_slice(&limbs)
        })
        .collect()
}

fn dense_fp32_extension_opening(
    evals: &[fp32::Field],
    point: &[fp32::ExtensionField],
) -> fp32::ExtensionField {
    let weights = lagrange_weights(point).expect("valid extension opening point");
    evals
        .iter()
        .zip(weights.iter())
        .fold(fp32::ExtensionField::zero(), |acc, (&coeff, &weight)| {
            acc + weight * fp32::ExtensionField::lift_base(coeff)
        })
}

#[derive(Clone, Copy)]
enum ExpectedRoot {
    Terminal,
    Fold,
}

fn run_zk_fp32_extension_opening_reduction<const NV: usize>(
    label: &'static [u8],
    expected_root: ExpectedRoot,
) {
    type Cfg = fp32::D32Full;
    const D: usize = Cfg::D;

    init_rayon_pool();
    run_on_large_stack(move || {
        let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
        let evals = (0..1usize << NV)
            .map(|_| fp32::Field::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect::<Vec<_>>();
        let poly =
            DensePoly::<fp32::Field, D>::from_field_evals(NV, &evals).expect("dense fp32 poly");
        let point = random_fp32_extension_point(NV, 0xcafe_babe);
        let expected_opening = dense_fp32_extension_opening(&evals, &point);

        let setup = <Scheme<D, Cfg> as CommitmentProver<fp32::Field, D>>::setup_prover(NV, 1, 1);
        let verifier_setup =
            <Scheme<D, Cfg> as CommitmentProver<fp32::Field, D>>::setup_verifier(&setup);
        let (commitment, hint) = <Scheme<D, Cfg> as CommitmentProver<fp32::Field, D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .expect("zk fp32 commit");

        let mut prover_transcript = AkitaTranscript::<fp32::Field>::new(label);
        let proof = <Scheme<D, Cfg> as CommitmentProver<fp32::Field, D>>::batched_prove(
            &setup,
            prove_input(&point, std::slice::from_ref(&poly), &commitment, hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("zk fp32 prove");

        match (expected_root, &proof.root) {
            (ExpectedRoot::Terminal, AkitaBatchedRootProof::Terminal(root)) => {
                assert!(
                    root.extension_opening_reduction.is_some(),
                    "fixture must exercise root extension-opening reduction"
                );
            }
            (ExpectedRoot::Fold, AkitaBatchedRootProof::Fold(root)) => {
                assert!(
                    root.extension_opening_reduction.is_some(),
                    "fixture must exercise folded-root extension-opening reduction"
                );
            }
            (ExpectedRoot::Terminal, other) => {
                panic!("expected terminal root extension-reduction proof, got {other:?}");
            }
            (ExpectedRoot::Fold, other) => {
                panic!("expected folded root extension-reduction proof, got {other:?}");
            }
        }

        let openings = [expected_opening];
        let mut verifier_transcript = AkitaTranscript::<fp32::Field>::new(label);
        <Scheme<D, Cfg> as CommitmentVerifier<fp32::Field, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitment),
            BasisMode::Lagrange,
        )
        .expect("zk fp32 extension-opening reduction verify");

        let mut tampered = proof.clone();
        let reduction = match &mut tampered.root {
            AkitaBatchedRootProof::Terminal(root) => root.extension_opening_reduction.as_mut(),
            AkitaBatchedRootProof::Fold(root) => root.extension_opening_reduction.as_mut(),
            AkitaBatchedRootProof::Direct { .. } => None,
        }
        .expect("fixture should carry extension-opening reduction partials");
        let partial = reduction
            .partials
            .first_mut()
            .expect("extension reduction should expose partials");
        *partial += fp32::ExtensionField::one();
        let mut verifier_transcript = AkitaTranscript::<fp32::Field>::new(label);
        assert!(
            <Scheme<D, Cfg> as CommitmentVerifier<fp32::Field, D>>::batched_verify(
                &tampered,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&point, &openings, &commitment),
                BasisMode::Lagrange,
            )
            .is_err(),
            "zk verifier should reject tampered extension-opening partials"
        );
    });
}

#[test]
fn zk_fp32_extension_opening_reduction_terminal_root_verifies() {
    run_zk_fp32_extension_opening_reduction::<13>(
        b"zk/fp32-extension-root-terminal",
        ExpectedRoot::Terminal,
    );
}

#[test]
fn zk_fp32_extension_opening_reduction_folded_root_verifies() {
    run_zk_fp32_extension_opening_reduction::<14>(
        b"zk/fp32-extension-root-fold",
        ExpectedRoot::Fold,
    );
}

fn run_zk_dense_commitment_hiding<const D: usize, BaseCfg>(nv: usize, label: &'static [u8])
where
    BaseCfg: CommitmentConfig<Field = F, ClaimField = F>,
    akita_planner::test_utils::PlannerCfg<RuntimePlanned<BaseCfg>>:
        CommitmentConfig<Field = F, ClaimField = F>,
    Scheme<D, akita_planner::test_utils::PlannerCfg<RuntimePlanned<BaseCfg>>>: CommitmentProver<
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
    type Cfg<Base> = akita_planner::test_utils::PlannerCfg<RuntimePlanned<Base>>;

    assert_eq!(BaseCfg::D, D);
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = Cfg::<BaseCfg>::get_params_for_batched_commitment(
            &akita_types::ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence"),
        )
        .expect("zk layout");
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

        let mut prover_transcript = AkitaTranscript::<F>::new(label);
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

        let mut verifier_transcript = AkitaTranscript::<F>::new(label);
        <Scheme<D, Cfg<BaseCfg>> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("zk verify");

        assert!(
            !decoded.zk_hiding.hiding_witness.is_empty(),
            "fixture should carry deferred ZK hiding witness material"
        );

        let mut trailing_hiding_witness = decoded.clone();
        trailing_hiding_witness
            .zk_hiding
            .hiding_witness
            .push(F::one());
        let mut verifier_transcript = AkitaTranscript::<F>::new(label);
        assert!(
            <Scheme<D, Cfg<BaseCfg>> as CommitmentVerifier<F, D>>::batched_verify(
                &trailing_hiding_witness,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&point, &openings, &commitments[0]),
                BasisMode::Lagrange,
            )
            .is_err(),
            "zk verifier should reject unreferenced trailing hiding witness slots"
        );
    });
}

fn run_zk_dense_cursor_binding_negatives() {
    type Cfg = akita_planner::test_utils::PlannerCfg<RuntimePlanned<fp128::D32Full>>;
    const D: usize = fp128::D32Full::D;
    const NV: usize = 14;
    const LABEL: &[u8] = b"zk_cursor_binding_negatives";

    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = Cfg::get_params_for_batched_commitment(
            &ClaimIncidenceSummary::same_point(NV, 1).expect("singleton incidence"),
        )
        .expect("zk layout");
        let mut rng = StdRng::seed_from_u64(0x5eed_c0de_0001);
        let evals: Vec<F> = (0..1usize << NV)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly");
        let point = random_point(NV, 0x0bad_cafe_0001);
        let expected_opening = opening_from_poly(&poly, &point, &layout);

        let setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme<D, Cfg> as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("zk commit");

        let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let mut prover_transcript = AkitaTranscript::<F>::new(LABEL);
        let proof = <Scheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("zk prove");
        assert!(
            proof.root.as_fold().is_some(),
            "fixture should exercise folded-root cursor checks"
        );

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

        let mut verifier_transcript = AkitaTranscript::<F>::new(LABEL);
        <Scheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("zk verify");

        assert!(
            !decoded.zk_hiding.u_blind.is_empty(),
            "fixture should carry a ZK hiding commitment"
        );
        assert!(
            !decoded.zk_hiding.hiding_witness.is_empty(),
            "fixture should carry consumed ZK hiding witness slots"
        );
        assert!(
            !decoded.zk_hiding.b_blinding_digits.is_empty(),
            "fixture should carry revealed ZK hiding commitment blinding digits"
        );

        let assert_rejects = |case: &str, tamper: &dyn Fn(&mut AkitaBatchedProof<F, F>)| {
            let mut tampered = decoded.clone();
            tamper(&mut tampered);
            let mut verifier_transcript = AkitaTranscript::<F>::new(LABEL);
            assert!(
                <Scheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                    &tampered,
                    &verifier_setup,
                    &mut verifier_transcript,
                    verify_input(&point, &openings, &commitments[0]),
                    BasisMode::Lagrange,
                )
                .is_err(),
                "zk verifier should reject tampered {case}"
            );
        };

        assert_rejects("u_blind", &|proof| {
            proof.zk_hiding.u_blind[0] += F::one();
        });
        assert_rejects("b_blinding_digits", &|proof| {
            proof.zk_hiding.b_blinding_digits[0] =
                proof.zk_hiding.b_blinding_digits[0].wrapping_add(1);
        });
        assert_rejects("first consumed hiding_witness slot", &|proof| {
            proof.zk_hiding.hiding_witness[0] += F::one();
        });
        assert_rejects("last consumed hiding_witness slot", &|proof| {
            let last = proof.zk_hiding.hiding_witness.len() - 1;
            proof.zk_hiding.hiding_witness[last] += F::one();
        });
        assert_rejects("root y_rings", &|proof| {
            let root = proof
                .root
                .as_fold_mut()
                .expect("fixture should use a folded root");
            root.y_rings = bumped_flat_ring_vec(&root.y_rings);
        });
        assert_rejects("stage1 s_claim", &|proof| {
            let root = proof
                .root
                .as_fold_mut()
                .expect("fixture should use a folded root");
            root.stage1.s_claim += F::one();
        });
        assert_rejects("stage1 child claim", &tamper_first_stage1_child_claim);
        assert_rejects(
            "masked stage1 round coefficient",
            &tamper_first_stage1_round_coeff,
        );
        assert_rejects(
            "masked stage2 round coefficient",
            &tamper_first_stage2_round_coeff,
        );
        assert_rejects("next_w_eval_masked", &|proof| {
            let root = proof
                .root
                .as_fold_mut()
                .expect("fixture should use a folded root");
            root.stage2.next_w_eval_masked += F::one();
        });
    });
}

fn run_zk_dense_v_hiding<const D: usize, BaseCfg>(nv: usize, label: &'static [u8])
where
    BaseCfg: CommitmentConfig<Field = F, ClaimField = F>,
    akita_planner::test_utils::PlannerCfg<RuntimePlanned<BaseCfg>>:
        CommitmentConfig<Field = F, ClaimField = F>,
    Scheme<D, akita_planner::test_utils::PlannerCfg<RuntimePlanned<BaseCfg>>>: CommitmentProver<
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
    type Cfg<Base> = akita_planner::test_utils::PlannerCfg<RuntimePlanned<Base>>;

    assert_eq!(BaseCfg::D, D);
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = Cfg::<BaseCfg>::get_params_for_batched_commitment(
            &akita_types::ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence"),
        )
        .expect("zk layout");
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

        let mut prover_transcript = AkitaTranscript::<F>::new(label);
        let proof = <Scheme<D, Cfg<BaseCfg>> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitments[0], hint.clone()),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("zk prove");

        let mut second_prover_transcript = AkitaTranscript::<F>::new(label);
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

        let mut second_serialized = Vec::new();
        let second_proof_shape = second_proof.shape();
        second_proof
            .serialize_compressed(&mut second_serialized)
            .expect("serialize second zk proof");
        let second_decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(second_serialized),
            &second_proof_shape,
        )
        .expect("deserialize second zk proof");

        let mut verifier_transcript = AkitaTranscript::<F>::new(label);
        <Scheme<D, Cfg<BaseCfg>> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("zk verify");
        let mut second_verifier_transcript = AkitaTranscript::<F>::new(label);
        <Scheme<D, Cfg<BaseCfg>> as CommitmentVerifier<F, D>>::batched_verify(
            &second_decoded,
            &verifier_setup,
            &mut second_verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("second zk verify");
    });
}

fn run_zk_dense_batched_shape_cases() {
    type Cfg = akita_planner::test_utils::PlannerCfg<RuntimePlanned<fp128::D32Full>>;
    const D: usize = fp128::D32Full::D;
    const NV: usize = 14;

    init_rayon_pool();
    run_on_large_stack(|| {
        let make_poly = |seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            let evals: Vec<F> = (0..1usize << NV)
                .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
                .collect();
            DensePoly::<F, D>::from_field_evals(NV, &evals).expect("dense poly")
        };

        const SAME_POINT_POLYS: usize = 3;
        let same_point_incidence =
            ClaimIncidenceSummary::same_point(NV, SAME_POINT_POLYS).expect("valid incidence");
        let same_point_layout = Cfg::get_params_for_batched_commitment(&same_point_incidence)
            .expect("same-point batched layout");
        let same_point_polys: Vec<DensePoly<F, D>> = (0..SAME_POINT_POLYS)
            .map(|idx| make_poly(0xd3e5_8000 + idx as u64))
            .collect();
        let same_point = random_point(NV, 0xaaaa_8000);
        let same_point_openings: Vec<F> = same_point_polys
            .iter()
            .map(|poly| opening_from_poly(poly, &same_point, &same_point_layout))
            .collect();
        let setup =
            <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, SAME_POINT_POLYS, 1);
        let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme<D, Cfg> as CommitmentProver<F, D>>::commit(&same_point_polys, &setup)
                .expect("same-point zk batched commit");
        let same_point_poly_refs: Vec<&DensePoly<F, D>> = same_point_polys.iter().collect();
        let mut prover_transcript = AkitaTranscript::<F>::new(b"zk/batched-shape/same-point");
        let proof = <Scheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&same_point, &same_point_poly_refs, &commitment, hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("same-point zk batched prove");
        assert!(
            !proof.zk_hiding.hiding_witness.is_empty(),
            "same-point batched ZK proof should consume hiding masks"
        );
        match &proof.root {
            AkitaBatchedRootProof::Fold(root) => {
                assert_eq!(root.y_rings.coeff_len() / D, 1);
            }
            AkitaBatchedRootProof::Terminal(root) => {
                assert_eq!(root.y_rings.coeff_len() / D, 1);
            }
            AkitaBatchedRootProof::Direct { .. } => {
                panic!("same-point fixture should use a folded or terminal ZK proof")
            }
        }
        let proof_shape = proof.shape();
        let mut serialized = Vec::new();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize same-point zk proof");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize same-point zk proof");
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"zk/batched-shape/same-point");
        <Scheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&same_point, &same_point_openings, &commitment),
            BasisMode::Lagrange,
        )
        .expect("same-point zk batched verify");

        const NUM_POINTS: usize = 2;
        let num_polys_per_point = [1usize; NUM_POINTS];
        let total_claims: usize = num_polys_per_point.iter().sum();
        let multipoint_incidence =
            ClaimIncidenceSummary::from_point_polys(NV, num_polys_per_point.to_vec())
                .expect("valid multipoint incidence");
        let multipoint_layout = Cfg::get_params_for_batched_commitment(&multipoint_incidence)
            .expect("multipoint batched layout");
        let polys_per_point: Vec<Vec<DensePoly<F, D>>> = (0..NUM_POINTS)
            .map(|point_idx| vec![make_poly(0xd3e5_9000 + point_idx as u64)])
            .collect();
        let opening_points_owned: Vec<Vec<F>> = (0..NUM_POINTS)
            .map(|point_idx| random_point(NV, 0xaaaa_9000 + point_idx as u64))
            .collect();
        let openings_per_point: Vec<Vec<F>> = polys_per_point
            .iter()
            .zip(opening_points_owned.iter())
            .map(|(polys, point)| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly(poly, point, &multipoint_layout))
                    .collect()
            })
            .collect();
        let polys_per_point_refs: Vec<&[DensePoly<F, D>]> =
            polys_per_point.iter().map(Vec::as_slice).collect();
        let openings_per_point_refs: Vec<&[F]> =
            openings_per_point.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();
        let setup =
            <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(NV, total_claims, NUM_POINTS);
        let verifier_setup = <Scheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
        let commit_outputs = <Scheme<D, Cfg> as CommitmentProver<F, D>>::batched_commit(
            &polys_per_point_refs,
            &setup,
        )
        .expect("multipoint zk batched commit");
        let (commitments, hints): (Vec<_>, Vec<_>) = commit_outputs.into_iter().unzip();
        let mut prover_transcript = AkitaTranscript::<F>::new(b"zk/batched-shape/multipoint");
        let proof = <Scheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_inputs_from_groups(&opening_points, &polys_per_point_refs, &commitments, hints),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("multipoint zk batched prove");
        match &proof.root {
            AkitaBatchedRootProof::Fold(root) => {
                assert_eq!(root.y_rings.coeff_len() / D, NUM_POINTS);
            }
            AkitaBatchedRootProof::Terminal(root) => {
                assert_eq!(root.y_rings.coeff_len() / D, NUM_POINTS);
            }
            AkitaBatchedRootProof::Direct { .. } => {
                panic!("multipoint fixture should use a folded or terminal ZK proof")
            }
        }
        let proof_shape = proof.shape();
        let mut serialized = Vec::new();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize multipoint zk proof");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize multipoint zk proof");
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"zk/batched-shape/multipoint");
        <Scheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_inputs_from_groups(&opening_points, &openings_per_point_refs, &commitments),
            BasisMode::Lagrange,
        )
        .expect("multipoint zk batched verify");
    });
}

#[test]
fn zk_dense_d32_rejects_cursor_binding_tampering() {
    run_zk_dense_cursor_binding_negatives();
}

#[test]
fn zk_dense_d32_batched_shape_cases_verify() {
    run_zk_dense_batched_shape_cases();
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
