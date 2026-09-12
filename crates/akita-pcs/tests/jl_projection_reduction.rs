use akita_algebra::jl::eval_block_tensor_mle;
use akita_prover::prove_jl_projection_chain;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{labels, AkitaTranscript, Transcript};
use akita_types::{
    JlBlockLayerPlan, JlCertificateId, JlMatrixDomain, JlMatrixLawId, JlProjectionChainPlan,
    JlProjectionProof, JlProjectionProofShape, JlProjectionStemId,
};
use akita_verifier::verify_jl_projection_chain;
use jolt_field::{
    CanonicalEncoding, Ext2, ExtField, Field, FpExt4, One, Prime128OffsetA7F7, Prime32Offset99,
    Prime64Offset59,
};

fn plan(schedule: [u8; 32], fold_level: u32, retries: u32) -> JlProjectionChainPlan {
    let first_domain = JlMatrixDomain::new(
        schedule,
        fold_level,
        JlCertificateId::ProjZ,
        JlProjectionStemId::Z,
        0,
        4,
        8,
        JlMatrixLawId::BalancedTernaryRepeatedBlock,
    )
    .unwrap();
    let second_domain = JlMatrixDomain::new(
        schedule,
        fold_level,
        JlCertificateId::ProjZ,
        JlProjectionStemId::Z,
        1,
        4,
        8,
        JlMatrixLawId::BalancedTernaryRepeatedBlock,
    )
    .unwrap();
    let first = JlBlockLayerPlan::new(first_domain, 2).unwrap();
    let second = JlBlockLayerPlan::new(second_domain, 1).unwrap();
    JlProjectionChainPlan::new(vec![first, second], retries, u128::MAX).unwrap()
}

fn bind<F>(transcript: &mut AkitaTranscript<F>)
where
    F: Field + CanonicalEncoding,
{
    transcript.absorb_and_record_bytes(
        labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        b"bound outgoing witness",
    );
}

fn source() -> Vec<i128> {
    (-8i128..8).map(|value| (value * 7) % 11).collect()
}

fn prove_fixture<F, E>(
    public_plan: &JlProjectionChainPlan,
    retry: u32,
) -> (JlProjectionProof<E>, AkitaTranscript<F>)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
{
    let mut transcript = AkitaTranscript::<F>::prover(b"jl/reduction/test", b"instance");
    bind(&mut transcript);
    let proof =
        prove_jl_projection_chain::<F, E, _>(&mut transcript, public_plan, retry, &source())
            .unwrap();
    (proof, transcript)
}

fn verify_fixture<F, E>(public_plan: &JlProjectionChainPlan, proof: &JlProjectionProof<E>)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
{
    let mut transcript = AkitaTranscript::<F>::verifier(b"jl/reduction/test", b"instance");
    bind(&mut transcript);
    let source_claim =
        verify_jl_projection_chain::<F, E, _>(&mut transcript, public_plan, proof).unwrap();
    let source_field = source().into_iter().map(E::from_i128).collect::<Vec<_>>();
    let first = public_plan.layers()[0];
    let direct = eval_block_tensor_mle(
        &source_field,
        first.blocks(),
        first.matrix_context(0).shape().unwrap().cols(),
        &source_claim.point,
    )
    .unwrap();
    assert_eq!(source_claim.evaluation, direct);
}

fn roundtrip<F, E>()
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
{
    let public_plan = plan([7; 32], 2, 1);
    let (proof, _) = prove_fixture::<F, E>(&public_plan, 0);
    assert_eq!(proof.reverse_layers.len(), 2);
    verify_fixture::<F, E>(&public_plan, &proof);
}

#[test]
fn reverse_two_layer_chain_roundtrips_all_shipped_fields() {
    roundtrip::<Prime32Offset99, FpExt4<Prime32Offset99>>();
    roundtrip::<Prime64Offset59, Ext2<Prime64Offset59>>();
    roundtrip::<Prime128OffsetA7F7, Prime128OffsetA7F7>();
}

#[test]
fn repeated_block_identity_tensor_matrix_roundtrips() {
    type F = Prime128OffsetA7F7;
    let domain = JlMatrixDomain::new(
        [3; 32],
        5,
        JlCertificateId::ProjZ,
        JlProjectionStemId::Z,
        0,
        2,
        4,
        JlMatrixLawId::BalancedTernaryRepeatedBlock,
    )
    .unwrap();
    let public_plan = JlProjectionChainPlan::new(
        vec![JlBlockLayerPlan::new(domain, 4).unwrap()],
        1,
        u128::MAX,
    )
    .unwrap();
    let (proof, _) = prove_fixture::<F, F>(&public_plan, 0);
    verify_fixture::<F, F>(&public_plan, &proof);
}

fn assert_rejects(
    public_plan: &JlProjectionChainPlan,
    proof: &JlProjectionProof<Prime128OffsetA7F7>,
) {
    type F = Prime128OffsetA7F7;
    let mut transcript = AkitaTranscript::<F>::verifier(b"jl/reduction/test", b"instance");
    bind(&mut transcript);
    assert!(verify_jl_projection_chain::<F, F, _>(&mut transcript, public_plan, proof).is_err());
}

#[test]
fn mutations_of_seed_image_layer_terminal_retry_and_context_reject() {
    type F = Prime128OffsetA7F7;
    let public_plan = plan([7; 32], 2, 2);
    let (proof, _) = prove_fixture::<F, F>(&public_plan, 0);

    let mut seed_transcript = AkitaTranscript::<F>::verifier(b"jl/reduction/test", b"instance");
    bind(&mut seed_transcript);
    seed_transcript.append_bytes(labels::ABSORB_COMMITMENT, b"seed mutation");
    assert!(
        verify_jl_projection_chain::<F, F, _>(&mut seed_transcript, &public_plan, &proof).is_err()
    );

    let mut image = proof.clone();
    image.clear_image[0] += 1;
    assert_rejects(&public_plan, &image);

    let invalid_layer = JlBlockLayerPlan::new(
        JlMatrixDomain::new(
            [7; 32],
            2,
            JlCertificateId::ProjZ,
            JlProjectionStemId::Z,
            9,
            4,
            8,
            JlMatrixLawId::BalancedTernaryRepeatedBlock,
        )
        .unwrap(),
        1,
    )
    .unwrap();
    assert!(
        JlProjectionChainPlan::new(vec![public_plan.layers()[0], invalid_layer], 2, u128::MAX)
            .is_err()
    );

    let mut terminal = proof.clone();
    terminal.reverse_layers[0].input_evaluation += F::one();
    assert_rejects(&public_plan, &terminal);

    let mut retry = proof.clone();
    retry.retry_index = Some(1);
    assert_rejects(&public_plan, &retry);

    let context_mutation = plan([8; 32], 2, 2);
    assert_rejects(&context_mutation, &proof);
    let level_mutation = plan([7; 32], 3, 2);
    assert_rejects(&level_mutation, &proof);
}

#[test]
fn headerless_shape_rejects_missing_extra_and_malformed_elements() {
    type F = Prime128OffsetA7F7;
    let public_plan = plan([5; 32], 1, 2);
    let (proof, _) = prove_fixture::<F, F>(&public_plan, 1);
    let shape = JlProjectionProofShape::from_plan(&public_plan).unwrap();
    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes).unwrap();
    let decoded = JlProjectionProof::<F>::deserialize_compressed_exact(&bytes, &shape).unwrap();
    assert_eq!(decoded, proof);

    let mut extra = bytes.clone();
    extra.push(0);
    assert!(JlProjectionProof::<F>::deserialize_compressed_exact(&extra, &shape).is_err());
    assert!(JlProjectionProof::<F>::deserialize_compressed_exact(
        &bytes[..bytes.len() - 1],
        &shape
    )
    .is_err());

    let mut missing_layer = proof.clone();
    missing_layer.reverse_layers.pop();
    assert_rejects(&public_plan, &missing_layer);
    let mut wrong_image = proof;
    wrong_image.clear_image.pop();
    assert_rejects(&public_plan, &wrong_image);
}
