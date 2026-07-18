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

fn extension_point() -> Vec<SmallE> {
    (0..SMALL_NV)
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
    let point = extension_point();
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
        akita_types::SetupContributionMode::Direct,
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
}
