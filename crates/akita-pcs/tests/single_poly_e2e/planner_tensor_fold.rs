use super::*;
use akita_challenges::TensorChallengeShape;
use akita_config::proof_optimized::fp128::D64OneHotTensor;
use akita_config::CommitmentConfig;
use akita_types::{AkitaScheduleInputs, ClaimIncidenceSummary, Step};

type TensorPresetScheme = AkitaCommitmentScheme<ONEHOT_D, D64OneHotTensor>;

fn planned_root_uses_tensor_shape(nv: usize) {
    let incidence = ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence");
    let schedule = D64OneHotTensor::get_params_for_prove(&incidence).expect("prove schedule");
    let Some(Step::Fold(root)) = schedule.steps.first() else {
        panic!("D64OneHotTensor schedule must start with a fold step");
    };
    assert_eq!(
        root.params.fold_challenge_shape,
        TensorChallengeShape::Tensor,
        "planner must bake the tensor shape into the root fold step"
    );
    assert!(
        root.params.num_blocks.is_power_of_two(),
        "tensor sampler requires a power-of-two num_blocks at the root"
    );

    let inputs = AkitaScheduleInputs {
        num_vars: nv,
        level: 0,
        current_w_len: 1usize << nv,
    };
    let log_basis = D64OneHotTensor::log_basis_at_level(inputs);
    let flat_baseline =
        <OneHotCfg as CommitmentConfig>::root_level_layout_with_log_basis(inputs, log_basis)
            .expect("flat baseline layout");
    assert!(
        root.params.num_digits_fold >= flat_baseline.num_digits_fold,
        "tensor envelope must require at least as many fold digits as the flat envelope"
    );
}

#[test]
fn planner_routes_tensor_shape_through_root_fold_step() {
    for nv in [12, 20] {
        planned_root_uses_tensor_shape(nv);
    }
}

fn run_prove_verify(nv: usize, label: Vec<u8>) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = D64OneHotTensor::commitment_layout(nv).expect("layout");
        assert_eq!(
            layout.fold_challenge_shape,
            TensorChallengeShape::Tensor,
            "commitment_layout must surface the tensor root shape"
        );
        let poly = make_onehot_poly(&layout, 0x1357_0000 + nv as u64);
        let pt = random_point(nv, 0x1357_f00d + nv as u64);
        let expected_opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);

        let setup = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(nv, 1, 1);
        let verifier_setup =
            <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .expect("commit");

        let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(&label);
        let proof = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        let root = proof.root.as_fold().expect("preset must fold root");
        assert!(
            !root.stage1.stages.is_empty(),
            "preset must exercise the stage-1 fold"
        );

        let mut verifier_transcript = Blake2bTranscript::<F>::new(&label);
        <TensorPresetScheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("verify");
    });
}

#[test]
fn d64_onehot_tensor_prove_verify_nv20() {
    run_prove_verify(20, b"single_poly_e2e/d64_onehot_tensor_nv20".to_vec());
}

#[test]
fn d64_onehot_tensor_rejects_tampered_proof() {
    const NV: usize = 12;
    init_rayon_pool();
    run_on_large_stack(|| {
        let layout = D64OneHotTensor::commitment_layout(NV).expect("layout");
        let poly = make_onehot_poly(&layout, 0xb1a5_0000 + NV as u64);
        let pt = random_point(NV, 0xb1a5_f00d);
        let expected_opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);

        let setup = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(NV, 1, 1);
        let verifier_setup =
            <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .expect("commit");

        let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"single_poly_e2e/d64_onehot_tensor_tampered");
        let mut proof = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        // Tampering with the absorbed stage-1 `s_claim` after the proof
        // is built breaks the verifier's stage-2 reconstruction. The new
        // tensor sampling labels are bound through the same transcript,
        // so the same negative guard exercises label propagation.
        let root = proof.root.as_fold_mut().expect("preset must fold root");
        root.stage1.s_claim += F::one();

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"single_poly_e2e/d64_onehot_tensor_tampered");
        let result = <TensorPresetScheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "verifier must reject tampered tensor stage-1 v on the production preset"
        );
    });
}
