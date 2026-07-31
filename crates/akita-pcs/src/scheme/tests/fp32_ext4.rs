use super::*;
use akita_config::proof_optimized::fp32;
use akita_field::ExtField;

type SmallCfg = fp32::D128OneHot;
type SmallF = fp32::Field;
type SmallE = fp32::ExtensionField;
type SmallScheme = AkitaCommitmentScheme<SmallCfg>;

const SMALL_D: usize = SmallCfg::D;
const SMALL_NV: usize = 16;
const SMALL_BATCH: usize = 2;
const TRANSCRIPT_LABEL: &[u8] = b"test/fp32-ext4-folded-only";

fn onehot_poly(seed: usize) -> OneHotPoly<SmallF, u8> {
    let onehot_k = SmallCfg::onehot_chunk_size();
    assert!(onehot_k <= 1usize << u8::BITS);
    let num_chunks = (1usize << SMALL_NV) / onehot_k;
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
    let prepared = CpuBackend
        .prepare_setup(&setup)
        .expect("prepared fp32 setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("fp32 prover stack");
    let verifier_setup = SmallScheme::setup_verifier(&setup).expect("fp32 verifier setup");
    let (commitment, hint) =
        SmallScheme::commit(&setup, &polys, &stack).expect("fp32 batched commitment");

    let mut prover_transcript = AkitaTranscript::<SmallF>::new(TRANSCRIPT_LABEL);
    let proof = SmallScheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&point, &poly_refs, &commitment, hint),
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
        verifier_claims(&point, &openings, &commitment),
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
        verifier_claims(&point, &wrong_openings, &commitment),
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
        verifier_claims(&point, &openings, &commitment),
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
        verifier_claims(&point, &openings, &commitment),
        BasisMode::Lagrange,
    )
    .expect_err("omitting the required root extension-opening reduction must reject");
}

#[test]
fn fp32_ext4_multi_group_uses_one_batched_eor_sumcheck() {
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;
    type ProtocolCfg =
        crate::test_support::EnvelopeFinalGroupConfig<fp32::D256OneHot, fp32::D128OneHot>;
    type PreNativeCfg = fp32::D256OneHot;
    type PreNativeScheme = AkitaCommitmentScheme<PreNativeCfg>;
    type ProtocolScheme = AkitaCommitmentScheme<ProtocolCfg>;

    let pre_layout = OpeningClaimsLayout::new(PRE_NV, 1).expect("precommit layout");
    let pre_params =
        <PrecommittedCommitmentConfig<PreNativeCfg> as CommitmentConfig>::
            get_params_for_batched_commitment(&pre_layout)
                .expect("precommit params");
    let pre_poly = grouped_onehot_poly(&pre_params, 1);
    let pre_setup = PreNativeScheme::setup_prover(PRE_NV, 1).expect("precommit setup");
    let pre_prepared = CpuBackend
        .prepare_setup(&pre_setup)
        .expect("prepared precommit setup");
    let pre_stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend,
        &pre_prepared,
        pre_setup.expanded.as_ref(),
    )
    .expect("precommit stack");
    let (_pre_frozen, pre_commitment, pre_hint) =
        PreNativeScheme::commit_group(&pre_setup, std::slice::from_ref(&pre_poly), &pre_stack)
            .expect("precommit");

    let grouped_layout = OpeningClaimsLayout::from_groups(vec![
        akita_types::PolynomialGroupLayout::new(PRE_NV, 1),
        akita_types::PolynomialGroupLayout::new(FINAL_NV, 1),
    ])
    .expect("grouped layout");
    let schedule = ProtocolCfg::get_params_for_prove(&grouped_layout).expect("grouped schedule");
    let root_params = &schedule.root.params.final_group.commitment;
    assert_eq!(
        root_params
            .group_role_dims(&grouped_layout, 0)
            .expect("pre group dims")
            .d_a(),
        256
    );
    assert_eq!(
        root_params
            .group_role_dims(&grouped_layout, 1)
            .expect("final group dims")
            .d_a(),
        128
    );
    let final_poly = grouped_onehot_poly(root_params, 2);

    let setup = ProtocolScheme::setup_prover(FINAL_NV, 2).expect("protocol setup");
    let prepared = CpuBackend
        .prepare_setup(&setup)
        .expect("prepared protocol setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("protocol stack");
    let (final_commitment, final_hint) = ProtocolScheme::commit_final_group(
        &setup,
        std::slice::from_ref(&final_poly),
        &stack,
        vec![akita_types::PolynomialGroupLayout::new(PRE_NV, 1)],
    )
    .expect("final commitment");

    let mut pre_point = extension_point(PRE_NV);
    pre_point[0] += SmallE::one();
    let final_point = extension_point(FINAL_NV);
    let pre_opening = onehot_opening_at_point(&pre_poly, &pre_point);
    let final_opening = onehot_opening_at_point(&final_poly, &final_point);
    let pre_refs = [&pre_poly];
    let final_refs = [&final_poly];
    let prover_claims = ProverOpeningData::new(
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
        verify_claims.clone(),
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
        verify_claims,
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
        tampered_claims,
        BasisMode::Lagrange,
    )
    .expect_err("tampered smaller-group opening must reject");
}
