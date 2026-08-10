use super::*;
use akita_config::proof_optimized::fp32;
use akita_field::ExtField;
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

type SmallCfg = fp32::OneHot;
type SmallF = fp32::Field;
type SmallE = fp32::ExtensionField;
type SmallScheme = AkitaCommitmentScheme<SmallCfg>;

const SMALL_D: usize = SmallCfg::D;
const SMALL_NV: usize = 16;
const SMALL_BATCH: usize = 2;
const TRANSCRIPT_LABEL: &[u8] = b"test/fp32-ext4-folded-only";

fn small_verifier_statement<'a>(
    point: &[SmallE],
    openings: &[SmallE],
    commitment: &'a CommittedGroup<SmallF>,
) -> GroupBatchStatement<'a, SmallE, SmallF> {
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.to_vec(),
        openings.to_vec(),
        commitment,
    )
    .expect("valid fp32 verifier group")])
    .expect("valid fp32 verifier claims");
    selected_statement::<SmallCfg>(claims).expect("selected fp32 verifier statement")
}

fn onehot_poly(seed: usize) -> OneHotPoly<SmallF, u8> {
    onehot_poly_for_num_vars(SMALL_NV, seed)
}

fn onehot_poly_for_num_vars(num_vars: usize, seed: usize) -> OneHotPoly<SmallF, u8> {
    let onehot_k = 256;
    assert!(onehot_k <= 1usize << u8::BITS);
    let num_chunks = (1usize << num_vars) / onehot_k;
    let indices = (0..num_chunks)
        .map(|chunk| Some(((chunk * 29 + seed * 41 + 7) % onehot_k) as u8))
        .collect();
    OneHotPoly::new(onehot_k, SMALL_D, indices).expect("valid fp32 one-hot polynomial")
}

fn extension_point(num_vars: usize) -> Vec<SmallE> {
    (0..num_vars)
        .map(|coordinate| {
            SmallE::from_base_slice(&[
                SmallF::from_u64((coordinate * 5 + 1) as u64),
                SmallF::from_u64((coordinate * 5 + 2) as u64),
                SmallF::from_u64((coordinate * 5 + 3) as u64),
                SmallF::from_u64((coordinate * 5 + 4) as u64),
            ])
        })
        .collect()
}

fn onehot_opening(poly: &OneHotPoly<SmallF, u8>, weights: &[SmallE]) -> SmallE {
    let onehot_k = poly.onehot_k();
    poly.indices()
        .iter()
        .enumerate()
        .filter_map(|(chunk, hot)| hot.map(|index| weights[chunk * onehot_k + usize::from(index)]))
        .fold(SmallE::zero(), |sum, weight| sum + weight)
}

fn grouped_onehot_poly(params: &CommittedGroupParams, seed: usize) -> OneHotPoly<SmallF, u8> {
    // Keep the tensor-partial fixture sparse at large arity. `K = 256` is
    // divisible by the supported native ring dimensions and leaves enough low
    // variables available for the Ext4 tensor head, so the one-hot kernel
    // never materializes the full 2^num_vars equality table.
    let onehot_k = 256;
    let total_field = params
        .num_live_blocks
        .checked_mul(params.num_positions_per_block)
        .and_then(|count| count.checked_mul(params.d_a()))
        .expect("grouped one-hot field length");
    assert_eq!(total_field % onehot_k, 0);
    let indices = (0..total_field / onehot_k)
        .map(|chunk| Some(((chunk * 29 + seed * 41 + 7) % onehot_k) as u8))
        .collect();
    OneHotPoly::new(onehot_k, params.d_a(), indices).expect("grouped one-hot polynomial")
}

fn onehot_opening_at_point(poly: &OneHotPoly<SmallF, u8>, point: &[SmallE]) -> SmallE {
    let onehot_k = poly.onehot_k();
    poly.indices()
        .iter()
        .enumerate()
        .filter_map(|(chunk, hot)| {
            hot.map(|index| {
                let evaluation_index = chunk * onehot_k + usize::from(index);
                point
                    .iter()
                    .enumerate()
                    .fold(SmallE::one(), |weight, (variable, &coordinate)| {
                        if (evaluation_index >> variable) & 1 == 0 {
                            weight * (SmallE::one() - coordinate)
                        } else {
                            weight * coordinate
                        }
                    })
            })
        })
        .fold(SmallE::zero(), |sum, weight| sum + weight)
}

#[test]
fn fp32_ext4_folded_eor_batched_roundtrip_and_rejections() {
    let opening_batch =
        OpeningClaimsLayout::new(SMALL_NV, SMALL_BATCH).expect("fp32 opening layout");
    let schedule = SmallCfg::get_params_for_prove(&opening_batch).expect("supported fp32 schedule");
    assert!(
        schedule.num_fold_levels() >= 2,
        "fixture must exercise the folded-only root/suffix topology"
    );

    let polys = [onehot_poly(0), onehot_poly(1)];
    let poly_refs: Vec<_> = polys.iter().collect();
    let point = extension_point(SMALL_NV);
    let weights = lagrange_weights(&point).expect("extension-field Lagrange weights");
    let openings: Vec<_> = polys
        .iter()
        .map(|poly| onehot_opening(poly, &weights))
        .collect();

    let setup = SmallScheme::setup_prover(SMALL_NV, SMALL_BATCH).expect("fp32 prover setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared fp32 setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("fp32 prover stack");
    let verifier_setup = SmallScheme::setup_verifier(&setup).expect("fp32 verifier setup");
    let (commitment, hint) =
        SmallScheme::commit(&setup, &polys, &stack).expect("fp32 batched commitment");

    let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.clone(),
        vec![SmallE::zero(); poly_refs.len()],
        commitment.clone(),
    )
    .expect("valid fp32 prover group")])
    .expect("valid fp32 prover claims");
    let mut prover_transcript = AkitaTranscript::<SmallF>::new(TRANSCRIPT_LABEL);
    let proof = SmallScheme::batched_prove::<_, _, _>(
        &setup,
        selected_prover_data::<SmallCfg, _>(prover_claims, vec![hint], vec![&poly_refs[..]])
            .expect("selected fp32 prover data"),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("fp32 extension proof");
    assert!(
        proof.root.extension_opening_reduction.is_some(),
        "non-base fp32 claims must use root extension-opening reduction"
    );
    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize fp32 extension proof");
    let proof = AkitaBatchedProof::<SmallF, SmallE>::deserialize_uncompressed(&bytes[..], &shape)
        .expect("deserialize fp32 extension proof");

    let mut verifier_transcript = AkitaTranscript::<SmallF>::new(TRANSCRIPT_LABEL);
    SmallScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        small_verifier_statement(&point, &openings, &commitment),
        BasisMode::Lagrange,
    )
    .expect("verify fp32 extension proof");

    let mut wrong_openings = openings.clone();
    wrong_openings[1] += SmallE::one();
    let mut verifier_transcript = AkitaTranscript::<SmallF>::new(TRANSCRIPT_LABEL);
    SmallScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        small_verifier_statement(&point, &wrong_openings, &commitment),
        BasisMode::Lagrange,
    )
    .expect_err("wrong batched extension opening must reject");

    let mut tampered = proof.clone();
    let reduction = tampered
        .root
        .extension_opening_reduction
        .as_mut()
        .expect("root EOR payload");
    let partial = reduction
        .partials
        .first_mut()
        .expect("root EOR must carry a partial evaluation");
    *partial += SmallE::one();
    let mut verifier_transcript = AkitaTranscript::<SmallF>::new(TRANSCRIPT_LABEL);
    SmallScheme::batched_verify(
        &tampered,
        &verifier_setup,
        &mut verifier_transcript,
        small_verifier_statement(&point, &openings, &commitment),
        BasisMode::Lagrange,
    )
    .expect_err("tampered extension-opening reduction partial must reject");

    let mut stripped = proof.clone();
    stripped.root.extension_opening_reduction = None;
    let mut verifier_transcript = AkitaTranscript::<SmallF>::new(TRANSCRIPT_LABEL);
    SmallScheme::batched_verify(
        &stripped,
        &verifier_setup,
        &mut verifier_transcript,
        small_verifier_statement(&point, &openings, &commitment),
        BasisMode::Lagrange,
    )
    .expect_err("omitting the required root extension-opening reduction must reject");
}

#[test]
fn fp32_ext4_multiblock_l2_pcs_roundtrip_and_stage2_rejections() {
    const NUM_VARS: usize = 20;
    const LABEL: &[u8] = b"test/fp32-ext4-multiblock-l2-pcs";
    type L2Cfg = crate::test_support::ForcedSmallFieldL2Config<SmallCfg>;
    type L2Scheme = AkitaCommitmentScheme<L2Cfg>;

    let opening_layout = OpeningClaimsLayout::new(NUM_VARS, 1).expect("L2 opening layout");
    let schedule = L2Cfg::get_params_for_prove(&opening_layout).expect("forced L2 schedule");
    let l2_step = schedule
        .recursive_folds
        .iter()
        .find(|step| {
            matches!(
                step.params.witness.inner_commit_matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            )
        })
        .expect("schedule-selected small-field L2 fold");
    let akita_types::InnerCommitSecurityRoute::L2 {
        norm_proof_shape, ..
    } = l2_step.params.witness.inner_commit_matrix.security_route()
    else {
        unreachable!("selected route checked above")
    };
    assert!(
        norm_proof_shape
            .limb_gram_layout()
            .expect("checked LimbGram shape")
            .expect("small-field fixture must use LimbGram")
            .block_count()
            > 1
    );

    let poly = onehot_poly_for_num_vars(NUM_VARS, 3);
    let point = extension_point(NUM_VARS);
    let opening = onehot_opening_at_point(&poly, &point);

    // Prove the same polynomial opening through the ordinary L-infinity
    // schedule before exercising the synthetic L2 schedule below.
    {
        const LINF_LABEL: &[u8] = b"test/fp32-ext4-same-witness-linf";
        let setup = SmallScheme::setup_prover(NUM_VARS, 1).expect("Linf prover setup");
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepared Linf setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("Linf prover stack");
        let verifier_setup = SmallScheme::setup_verifier(&setup).expect("Linf verifier setup");
        let (commitment, hint) = SmallScheme::commit(&setup, std::slice::from_ref(&poly), &stack)
            .expect("Linf commitment");
        let poly_refs = [&poly];
        let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![SmallE::zero()],
            commitment.clone(),
        )
        .expect("Linf prover group")])
        .expect("Linf prover claims");
        let mut prover_transcript = AkitaTranscript::<SmallF>::new(LINF_LABEL);
        let proof = SmallScheme::batched_prove(
            &setup,
            selected_prover_data::<SmallCfg, _>(prover_claims, vec![hint], vec![&poly_refs])
                .expect("Linf prover data"),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("same-witness Linf proof");
        assert!(proof
            .recursive_folds
            .iter()
            .all(|fold| fold.stage1.norm_proof.is_none()));
        let mut verifier_transcript = AkitaTranscript::<SmallF>::new(LINF_LABEL);
        SmallScheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            small_verifier_statement(&point, &[opening], &commitment),
            BasisMode::Lagrange,
        )
        .expect("verify same-witness Linf proof");
    }

    let setup = L2Scheme::setup_prover(NUM_VARS, 1).expect("L2 prover setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared L2 setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("L2 prover stack");
    let verifier_setup = L2Scheme::setup_verifier(&setup).expect("L2 verifier setup");
    let (commitment, hint) =
        L2Scheme::commit(&setup, std::slice::from_ref(&poly), &stack).expect("L2 commitment");
    let poly_refs = [&poly];
    let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.clone(),
        vec![SmallE::zero()],
        commitment.clone(),
    )
    .expect("L2 prover group")])
    .expect("L2 prover claims");
    let mut prover_transcript = AkitaTranscript::<SmallF>::new(LABEL);
    let proof = L2Scheme::batched_prove(
        &setup,
        selected_prover_data::<L2Cfg, _>(prover_claims, vec![hint], vec![&poly_refs])
            .expect("L2 prover data"),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("small-field L2 proof");
    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize small-field L2 PCS proof");
    let proof = AkitaBatchedProof::<SmallF, SmallE>::deserialize_uncompressed(&bytes[..], &shape)
        .expect("deserialize small-field L2 PCS proof");
    let l2_index = proof
        .recursive_folds
        .iter()
        .position(|fold| fold.stage1.norm_proof.is_some())
        .expect("proof must carry the selected L2 norm");
    assert!(
        proof.recursive_folds[l2_index]
            .stage1
            .norm_proof
            .as_ref()
            .expect("L2 norm proof")
            .subclaims
            .len()
            > 1
    );

    let verify = |candidate: &AkitaBatchedProof<SmallF, SmallE>| {
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![opening],
            &commitment,
        )
        .expect("L2 verifier group")])
        .expect("L2 verifier claims");
        let mut transcript = AkitaTranscript::<SmallF>::new(LABEL);
        L2Scheme::batched_verify(
            candidate,
            &verifier_setup,
            &mut transcript,
            selected_statement::<L2Cfg>(claims).expect("L2 verifier statement"),
            BasisMode::Lagrange,
        )
    };
    verify(&proof).expect("verify serialized small-field L2 PCS proof");

    let mut bad_subclaim = proof.clone();
    bad_subclaim.recursive_folds[l2_index]
        .stage1
        .norm_proof
        .as_mut()
        .expect("L2 norm proof")
        .subclaims[0] += SmallE::one();
    assert!(verify(&bad_subclaim).is_err());

    let mut bad_virtual = proof.clone();
    bad_virtual.recursive_folds[l2_index]
        .stage1
        .norm_proof
        .as_mut()
        .expect("L2 norm proof")
        .virtual_evaluations[0] += SmallE::one();
    assert!(verify(&bad_virtual).is_err());

    let mut bad_stage2 = proof;
    bad_stage2.recursive_folds[l2_index]
        .stage2
        .sumcheck_proof
        .round_polys[0]
        .coeffs_except_linear_term[0] += SmallE::one();
    assert!(verify(&bad_stage2).is_err());
}

#[test]
fn fp32_ext4_multi_group_uses_one_batched_eor_sumcheck() {
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;
    type ProtocolCfg = fp32::OneHot;
    type ProtocolScheme = AkitaCommitmentScheme<ProtocolCfg>;
    let pre_group = PolynomialGroupLayout::new(PRE_NV, 1);
    let pre_params = ProtocolCfg::runtime_schedule(AkitaScheduleLookupKey::single(pre_group))
        .expect("precommit schedule")
        .root
        .params
        .final_group
        .commitment;
    let pre_poly = grouped_onehot_poly(&pre_params, 1);
    let pre_setup = ProtocolScheme::setup_prover(PRE_NV, 1).expect("precommit setup");
    let pre_prepared = CpuBackend::DEFAULT
        .prepare_setup(&pre_setup)
        .expect("prepared precommit setup");
    let pre_stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &pre_prepared,
        pre_setup.expanded.as_ref(),
    )
    .expect("precommit stack");
    let (pre_commitment, pre_hint) =
        ProtocolScheme::commit_group(&pre_setup, std::slice::from_ref(&pre_poly), &pre_stack)
            .expect("precommit");

    let grouped_layout = OpeningClaimsLayout::from_groups(vec![
        akita_types::PolynomialGroupLayout::new(PRE_NV, 1),
        akita_types::PolynomialGroupLayout::new(FINAL_NV, 1),
    ])
    .expect("grouped layout");
    let schedule = ProtocolCfg::runtime_schedule(AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(FINAL_NV, 1),
        precommitteds: vec![pre_commitment.profile],
    })
    .expect("grouped schedule");
    let root_params = &schedule.root.params.final_group.commitment;
    assert_eq!(
        root_params
            .group_role_dims(&grouped_layout, 0)
            .expect("pre group dims")
            .d_a(),
        128
    );
    assert_eq!(
        root_params
            .group_role_dims(&grouped_layout, 1)
            .expect("final group dims")
            .d_a(),
        256
    );
    let final_poly = grouped_onehot_poly(root_params, 2);

    let setup = ProtocolScheme::setup_prover(FINAL_NV, 2).expect("protocol setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared protocol setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("protocol stack");
    let (final_commitment, final_hint, _selection) = ProtocolScheme::commit_final_group(
        &setup,
        std::slice::from_ref(&final_poly),
        &stack,
        vec![pre_commitment.profile],
    )
    .expect("final commitment");

    let mut pre_point = extension_point(PRE_NV);
    pre_point[0] += SmallE::one();
    let final_point = extension_point(FINAL_NV);
    let pre_opening = onehot_opening_at_point(&pre_poly, &pre_point);
    let final_opening = onehot_opening_at_point(&final_poly, &final_point);
    let pre_refs = [&pre_poly];
    let final_refs = [&final_poly];
    let prover_claims = selected_prover_data::<ProtocolCfg, _>(
        OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                pre_point.clone(),
                vec![pre_opening],
                pre_commitment.clone(),
            )
            .expect("pre prover claims"),
            PolynomialGroupClaims::new(
                final_point.clone(),
                vec![final_opening],
                final_commitment.clone(),
            )
            .expect("final prover claims"),
        ])
        .expect("grouped prover claims"),
        vec![pre_hint, final_hint],
        vec![&pre_refs, &final_refs],
    )
    .expect("grouped prover data");
    let selection = prover_claims.0;

    let mut prover_transcript = AkitaTranscript::<SmallF>::new(b"test/fp32-ext4-multi-group-eor");
    let proof = ProtocolScheme::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("grouped extension proof");
    let reduction = proof
        .root
        .extension_opening_reduction
        .as_ref()
        .expect("grouped root EOR");
    assert_eq!(
        reduction.sumcheck.round_polys.len(),
        FINAL_NV
            - akita_types::tensor_opening_split::<SmallF, SmallE>()
                .expect("tensor split")
                .0,
        "all groups must share one max-arity sumcheck"
    );

    let verifier_setup = ProtocolScheme::setup_verifier(&setup).expect("verifier setup");
    let verify_claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(pre_point.clone(), vec![pre_opening], &pre_commitment)
            .expect("pre verifier claims"),
        PolynomialGroupClaims::new(final_point.clone(), vec![final_opening], &final_commitment)
            .expect("final verifier claims"),
    ])
    .expect("grouped verifier claims");
    let mut verifier_transcript = AkitaTranscript::<SmallF>::new(b"test/fp32-ext4-multi-group-eor");
    ProtocolScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        GroupBatchStatement::new(selection, verify_claims.clone())
            .expect("grouped verifier statement"),
        BasisMode::Lagrange,
    )
    .expect("verify grouped extension proof");

    let mut stripped = proof.clone();
    stripped.root.extension_opening_reduction = None;
    let mut stripped_transcript = AkitaTranscript::<SmallF>::new(b"test/fp32-ext4-multi-group-eor");
    ProtocolScheme::batched_verify(
        &stripped,
        &verifier_setup,
        &mut stripped_transcript,
        GroupBatchStatement::new(selection, verify_claims)
            .expect("stripped-proof verifier statement"),
        BasisMode::Lagrange,
    )
    .expect_err("omitting the required multi-group root extension-opening reduction must reject");

    let tampered_claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(
            pre_point,
            vec![pre_opening + SmallE::one()],
            &pre_commitment,
        )
        .expect("tampered pre claims"),
        PolynomialGroupClaims::new(final_point, vec![final_opening], &final_commitment)
            .expect("final verifier claims"),
    ])
    .expect("tampered grouped claims");
    let mut tampered_transcript = AkitaTranscript::<SmallF>::new(b"test/fp32-ext4-multi-group-eor");
    ProtocolScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut tampered_transcript,
        GroupBatchStatement::new(selection, tampered_claims)
            .expect("tampered-claims verifier statement"),
        BasisMode::Lagrange,
    )
    .expect_err("tampered smaller-group opening must reject");
}
