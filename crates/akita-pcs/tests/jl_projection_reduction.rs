use akita_algebra::jl::{base_field_modulus, eval_block_tensor_mle};
use akita_prover::{
    absorb_jl_projection_batch_images, prepare_jl_projection_batch,
    prove_jl_projection_reduction_batch, JlProjectionProverInput,
};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{labels, AkitaTranscript, Transcript};
use akita_types::{
    JlBlockLayerPlan, JlCertificateId, JlMatrixEnvelopeDomain, JlMatrixLawId, JlMatrixMember,
    JlProjectionBatchPlan, JlProjectionBatchProof, JlProjectionBatchProofShape,
    JlProjectionChainPlan, JlProjectionStemId,
};
use akita_verifier::{
    absorb_jl_projection_verification_images, prepare_jl_projection_verification,
    verify_jl_projection_reduction_batch,
};
use jolt_field::{
    CanonicalEncoding, Ext2, ExtField, Field, FpExt4, One, Prime128OffsetA7F7, Prime32Offset99,
    Prime64Offset59,
};

const TEST_ENERGY_BOUND: u128 = 10_000_000;

fn envelope(schedule: [u8; 32], fold_level: u32, depth: u16) -> JlMatrixEnvelopeDomain {
    JlMatrixEnvelopeDomain::new(
        schedule,
        fold_level,
        depth,
        4,
        8,
        JlMatrixLawId::BalancedTernaryRepeatedBlock,
    )
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn layer(
    schedule: [u8; 32],
    fold_level: u32,
    certificate: JlCertificateId,
    stem: JlProjectionStemId,
    layer: u16,
    depth: u16,
    rows: usize,
    cols: usize,
    blocks: usize,
) -> JlBlockLayerPlan {
    let member = JlMatrixMember::new(
        envelope(schedule, fold_level, depth),
        certificate,
        stem,
        layer,
        rows,
        cols,
    )
    .unwrap();
    JlBlockLayerPlan::new(member, blocks).unwrap()
}

fn two_certificate_plan(
    schedule: [u8; 32],
    fold_level: u32,
    retries: u32,
) -> JlProjectionBatchPlan {
    let z = JlProjectionChainPlan::new(
        vec![
            layer(
                schedule,
                fold_level,
                JlCertificateId::ProjZ,
                JlProjectionStemId::Z,
                0,
                0,
                4,
                8,
                2,
            ),
            layer(
                schedule,
                fold_level,
                JlCertificateId::ProjZ,
                JlProjectionStemId::Z,
                1,
                1,
                4,
                8,
                1,
            ),
        ],
        TEST_ENERGY_BOUND,
    )
    .unwrap();
    let et = JlProjectionChainPlan::new(
        vec![
            layer(
                schedule,
                fold_level,
                JlCertificateId::ProjEt,
                JlProjectionStemId::EtTail,
                0,
                0,
                2,
                8,
                2,
            ),
            layer(
                schedule,
                fold_level,
                JlCertificateId::ProjEt,
                JlProjectionStemId::EtTail,
                1,
                1,
                2,
                4,
                1,
            ),
        ],
        TEST_ENERGY_BOUND,
    )
    .unwrap();
    JlProjectionBatchPlan::new(vec![z, et], retries).unwrap()
}

fn single_certificate_plan(retries: u32) -> JlProjectionBatchPlan {
    let envelope = JlMatrixEnvelopeDomain::new(
        [3; 32],
        5,
        0,
        2,
        4,
        JlMatrixLawId::BalancedTernaryRepeatedBlock,
    )
    .unwrap();
    let member = JlMatrixMember::new(
        envelope,
        JlCertificateId::ProjZ,
        JlProjectionStemId::Z,
        0,
        2,
        4,
    )
    .unwrap();
    let chain = JlProjectionChainPlan::new(
        vec![JlBlockLayerPlan::new(member, 4).unwrap()],
        TEST_ENERGY_BOUND,
    )
    .unwrap();
    JlProjectionBatchPlan::new(vec![chain], retries).unwrap()
}

fn bind<F: Field + CanonicalEncoding>(transcript: &mut AkitaTranscript<F>) {
    transcript.absorb_and_record_bytes(
        labels::ABSORB_NEXT_LEVEL_WITNESS_BINDING,
        b"bound outgoing witness",
    );
}

fn z_source() -> Vec<i128> {
    (-8i128..8).map(|value| (value * 7) % 11).collect()
}

fn et_source() -> Vec<i128> {
    z_source().into_iter().rev().collect()
}

fn prove<F, E>(
    plan: &JlProjectionBatchPlan,
    retry: u32,
    sources: &[Vec<i128>],
) -> JlProjectionBatchProof<E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
{
    let mut transcript = AkitaTranscript::<F>::prover(b"jl/reduction/test", b"instance");
    bind(&mut transcript);
    let inputs = sources
        .iter()
        .map(|source| JlProjectionProverInput { source })
        .collect::<Vec<_>>();
    let prepared =
        prepare_jl_projection_batch::<F, _>(&mut transcript, plan, retry, &inputs).unwrap();
    let ready = absorb_jl_projection_batch_images::<F, _>(&mut transcript, prepared).unwrap();
    prove_jl_projection_reduction_batch::<F, E, _>(&mut transcript, ready).unwrap()
}

fn verify<F, E>(
    plan: &JlProjectionBatchPlan,
    proof: &JlProjectionBatchProof<E>,
    sources: &[Vec<i128>],
) where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
{
    let mut transcript = AkitaTranscript::<F>::verifier(b"jl/reduction/test", b"instance");
    bind(&mut transcript);
    let prepared =
        prepare_jl_projection_verification::<F, E, _>(&mut transcript, plan, proof).unwrap();
    let ready =
        absorb_jl_projection_verification_images::<F, E, _>(&mut transcript, prepared).unwrap();
    let claims = verify_jl_projection_reduction_batch::<F, E, _>(&mut transcript, ready).unwrap();
    assert_eq!(claims.len(), sources.len());
    for ((claim, source), chain) in claims.iter().zip(sources).zip(plan.chains()) {
        let source_field = source.iter().copied().map(E::from_i128).collect::<Vec<_>>();
        let first = chain.layers()[0];
        let direct = eval_block_tensor_mle(
            &source_field,
            first.blocks(),
            first.matrix_member().shape().unwrap().cols(),
            &claim.point,
        )
        .unwrap();
        assert_eq!(claim.evaluation, direct);
    }
}

fn assert_rejects<F, E>(plan: &JlProjectionBatchPlan, proof: &JlProjectionBatchProof<E>)
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
{
    let mut transcript = AkitaTranscript::<F>::verifier(b"jl/reduction/test", b"instance");
    bind(&mut transcript);
    let rejected = prepare_jl_projection_verification::<F, E, _>(&mut transcript, plan, proof)
        .and_then(|prepared| {
            absorb_jl_projection_verification_images::<F, E, _>(&mut transcript, prepared)
        })
        .and_then(|ready| verify_jl_projection_reduction_batch::<F, E, _>(&mut transcript, ready))
        .is_err();
    assert!(rejected);
}

fn roundtrip<F, E>()
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
{
    let plan = two_certificate_plan([7; 32], 2, 1);
    let sources = [z_source(), et_source()];
    let proof = prove::<F, E>(&plan, 0, &sources);
    assert_eq!(proof.chains.len(), 2);
    assert!(proof.retry_index.is_none());
    verify::<F, E>(&plan, &proof, &sources);
}

#[test]
fn two_certificate_forest_roundtrips_all_shipped_fields() {
    roundtrip::<Prime32Offset99, FpExt4<Prime32Offset99>>();
    roundtrip::<Prime64Offset59, Ext2<Prime64Offset59>>();
    roundtrip::<Prime128OffsetA7F7, Prime128OffsetA7F7>();
}

#[test]
fn repeated_block_identity_tensor_matrix_roundtrips() {
    type F = Prime128OffsetA7F7;
    let plan = single_certificate_plan(1);
    let sources = [z_source()];
    let proof = prove::<F, F>(&plan, 0, &sources);
    verify::<F, F>(&plan, &proof, &sources);
}

#[test]
fn proof_mutations_and_public_domain_mutations_reject() {
    type F = Prime128OffsetA7F7;
    let plan = two_certificate_plan([7; 32], 2, 2);
    let sources = [z_source(), et_source()];
    let proof = prove::<F, F>(&plan, 0, &sources);

    let mut image = proof.clone();
    image.chains[0].clear_image[0] += 1;
    assert_rejects::<F, F>(&plan, &image);

    let mut terminal = proof.clone();
    terminal.chains[0].reverse_layers[0].input_evaluation += F::one();
    assert_rejects::<F, F>(&plan, &terminal);

    let mut retry = proof.clone();
    retry.retry_index = Some(1);
    assert_rejects::<F, F>(&plan, &retry);

    let context_mutation = two_certificate_plan([8; 32], 2, 2);
    assert_rejects::<F, F>(&context_mutation, &proof);
    let level_mutation = two_certificate_plan([7; 32], 3, 2);
    assert_rejects::<F, F>(&level_mutation, &proof);

    let mut order = proof.clone();
    order.chains.swap(0, 1);
    assert_rejects::<F, F>(&plan, &order);
}

fn modulus_alias_rejects<F>()
where
    F: Field + CanonicalEncoding + ExtField<F> + AkitaSerialize,
{
    let plan = two_certificate_plan([11; 32], 4, 1);
    let sources = [z_source(), et_source()];
    let proof = prove::<F, F>(&plan, 0, &sources);
    let modulus = i128::try_from(base_field_modulus::<F>().unwrap()).unwrap();
    for alias in [modulus, -modulus] {
        let mut malformed = proof.clone();
        malformed.chains[0].clear_image[0] = alias;
        assert_rejects::<F, F>(&plan, &malformed);
    }
}

#[test]
fn noncanonical_clear_image_aliases_reject_for_all_base_fields() {
    modulus_alias_rejects::<Prime32Offset99>();
    modulus_alias_rejects::<Prime64Offset59>();
    type F = Prime128OffsetA7F7;
    let plan = two_certificate_plan([12; 32], 4, 1);
    let sources = [z_source(), et_source()];
    let mut proof = prove::<F, F>(&plan, 0, &sources);
    proof.chains[0].clear_image[0] =
        i128::try_from(base_field_modulus::<F>().unwrap() / 2).unwrap() + 1;
    assert_rejects::<F, F>(&plan, &proof);
}

#[test]
fn all_images_are_absorbed_before_any_reduction() {
    type F = Prime128OffsetA7F7;
    let plan = two_certificate_plan([13; 32], 2, 1);
    let sources = [z_source(), et_source()];
    let proof = prove::<F, F>(&plan, 0, &sources);

    let mut verifier = AkitaTranscript::<F>::verifier(b"jl/reduction/test", b"instance");
    bind(&mut verifier);
    let prepared =
        prepare_jl_projection_verification::<F, F, _>(&mut verifier, &plan, &proof).unwrap();
    let ready =
        absorb_jl_projection_verification_images::<F, F, _>(&mut verifier, prepared).unwrap();
    verifier.append_bytes(labels::ABSORB_COMMITMENT, b"between images and reductions");
    assert!(verify_jl_projection_reduction_batch::<F, F, _>(&mut verifier, ready).is_err());
}

#[test]
fn headerless_batch_shape_rejects_missing_extra_and_malformed_elements() {
    type F = Prime128OffsetA7F7;
    let plan = two_certificate_plan([5; 32], 1, 2);
    let sources = [z_source(), et_source()];
    let proof = prove::<F, F>(&plan, 1, &sources);
    let shape = JlProjectionBatchProofShape::from_plan(&plan).unwrap();
    let mut bytes = Vec::new();
    proof.serialize_compressed(&mut bytes).unwrap();
    let mut expected_prelude = Vec::new();
    proof
        .retry_index
        .unwrap()
        .serialize_compressed(&mut expected_prelude)
        .unwrap();
    for chain in &proof.chains {
        for coordinate in &chain.clear_image {
            coordinate
                .serialize_compressed(&mut expected_prelude)
                .unwrap();
        }
    }
    assert_eq!(&bytes[..expected_prelude.len()], expected_prelude);
    let decoded =
        JlProjectionBatchProof::<F>::deserialize_compressed_exact(&bytes, &shape).unwrap();
    assert_eq!(decoded, proof);

    let mut extra = bytes.clone();
    extra.push(0);
    assert!(JlProjectionBatchProof::<F>::deserialize_compressed_exact(&extra, &shape).is_err());
    assert!(JlProjectionBatchProof::<F>::deserialize_compressed_exact(
        &bytes[..bytes.len() - 1],
        &shape,
    )
    .is_err());

    let mut missing_chain = proof.clone();
    missing_chain.chains.pop();
    assert_rejects::<F, F>(&plan, &missing_chain);
    let mut wrong_image = proof;
    wrong_image.chains[0].clear_image.pop();
    assert_rejects::<F, F>(&plan, &wrong_image);
}
