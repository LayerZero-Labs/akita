#![allow(missing_docs)]

mod common;

use blake2::{Blake2b512, Digest};
use common::*;
use hachi_pcs::algebra::{CyclotomicRing, Fp64};
use hachi_pcs::protocol::commitment::RingCommitment;
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::proof::{FlatRingVec, HachiBatchedProof, PackedDigits};
use hachi_pcs::protocol::transcript::{labels, Blake2bTranscript, KeccakTranscript};
use hachi_pcs::{
    CanonicalField, CommitmentScheme, FromSmallInt, HachiDeserialize, HachiSerialize, Transcript,
};

type FixtureField = Fp64<4294967197>;

fn digest_hex(bytes: &[u8]) -> String {
    let digest = Blake2b512::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn compressed_bytes(value: &impl HachiSerialize) -> Vec<u8> {
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .expect("fixture serialization");
    assert_eq!(bytes.len(), value.compressed_size());
    bytes
}

fn assert_fixture(name: &str, bytes: &[u8], expected_len: usize, expected_digest: &str) {
    let actual_digest = digest_hex(bytes);
    if expected_digest == "UPDATE_ME" {
        panic!("{name}: len={} digest={actual_digest}", bytes.len());
    }
    assert_eq!(bytes.len(), expected_len, "{name} length changed");
    assert_eq!(actual_digest, expected_digest, "{name} digest changed");
}

fn sample_transcript_schedule<T: Transcript<FixtureField>>(transcript: &mut T) -> FixtureField {
    transcript.append_bytes(labels::ABSORB_COMMITMENT, b"commitment-a");
    transcript.append_bytes(labels::ABSORB_COMMITMENT, b"commitment-b");
    transcript.append_serde(labels::ABSORB_EVALUATION_CLAIMS, &42u64);
    let rho = transcript.challenge_scalar(labels::CHALLENGE_LINEAR_RELATION);

    transcript.append_bytes(labels::ABSORB_RING_SWITCH_MESSAGE, b"ring-switch");
    let zeta = transcript.challenge_scalar(labels::CHALLENGE_RING_SWITCH);

    transcript.append_field(labels::ABSORB_SUMCHECK_ROUND, &(rho + zeta));
    let r = transcript.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);

    transcript.append_field(labels::ABSORB_STOP_CONDITION, &r);
    transcript.challenge_scalar(labels::CHALLENGE_STOP_CONDITION)
}

#[test]
fn transcript_challenge_regression_vectors() {
    let mut blake = Blake2bTranscript::<FixtureField>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let mut keccak = KeccakTranscript::<FixtureField>::new(labels::DOMAIN_HACHI_PROTOCOL);

    assert_eq!(
        sample_transcript_schedule(&mut blake).to_canonical_u128(),
        359576514,
        "Blake2b transcript challenge changed"
    );
    assert_eq!(
        sample_transcript_schedule(&mut keccak).to_canonical_u128(),
        1860762462,
        "Keccak transcript challenge changed"
    );
}

#[test]
fn serialization_regression_vectors() {
    type R = CyclotomicRing<FixtureField, 4>;
    let rings = vec![
        R::from_coefficients([1, 2, 3, 4].map(FixtureField::from_u64)),
        R::from_coefficients([5, 8, 13, 21].map(FixtureField::from_u64)),
    ];
    let commitment = RingCommitment { u: rings.clone() };
    let commitment_bytes = compressed_bytes(&commitment);
    assert_fixture(
        "ring_commitment",
        &commitment_bytes,
        72,
        "977ca358868575f0c78cd11cea9c44c1eba04393df83750ead62d78173a17e506eb472376c20310fd5fc21f0f8733133603fd810784c30738d93b497cb6ac6f1",
    );
    let decoded_commitment = RingCommitment::<FixtureField, 4>::deserialize_compressed(
        &mut std::io::Cursor::new(&commitment_bytes),
        &(),
    )
    .expect("commitment deserialize");
    assert_eq!(decoded_commitment, commitment);

    let flat_rings = FlatRingVec::from_ring_elems(&rings);
    let flat_ring_bytes = compressed_bytes(&flat_rings);
    assert_fixture(
        "flat_ring_vec",
        &flat_ring_bytes,
        64,
        "d118e1ba6ab51b4fd51722ddf78f2f0904abcc281a6e2a2ab7744431f3395694663ee05d574c1db959fbc34a82055cf0263ecf83b6e77f9c3823098e43b5a9b9",
    );
    let decoded_flat = FlatRingVec::<FixtureField>::deserialize_compressed(
        &mut std::io::Cursor::new(&flat_ring_bytes),
        &flat_rings.coeff_len(),
    )
    .expect("flat ring vec deserialize");
    assert_eq!(decoded_flat, flat_rings.clone().into_compact());

    let packed_digits = PackedDigits::from_i8_digits(&[-4, -1, 0, 1, 2, 3, -2, 0, 1], 4);
    let packed_bytes = compressed_bytes(&packed_digits);
    assert_fixture(
        "packed_digits",
        &packed_bytes,
        5,
        "135ebf8d615aa8b85286f34c3a6573ce68acbb4b4f87a8ff88b022ec8355af76eb63c3d0ef93ee34d905dd57896cdaccaf4ac9f27162bd293fdafd1df05b9ff8",
    );
    let decoded_digits = PackedDigits::deserialize_compressed(
        &mut std::io::Cursor::new(&packed_bytes),
        &(packed_digits.num_elems, packed_digits.bits_per_elem),
    )
    .expect("packed digits deserialize");
    assert_eq!(decoded_digits, packed_digits);
}

#[test]
fn production_onehot_nv32_proof_regression_vector() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let nv = 32;
        let layout = OneHotCfg::commitment_layout(nv).expect("layout");
        let poly = make_onehot_poly(&layout, 0xabad_1dea);
        let point = random_point(nv, 0xfeed_face);
        let expected_opening = opening_from_poly(&poly, &point, &layout);

        let setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_prover(nv, 1, 1);
        let verifier_setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::commit(commit_input, &setup)
        .expect("commit");

        let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"protocol_regression_vectors/onehot_nv32");
        let proof = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        let proof_shape = proof.shape();
        let proof_bytes = compressed_bytes(&proof);
        assert_fixture(
            "production_onehot_nv32_proof",
            &proof_bytes,
            77216,
            "6d12c3dd34ad293437104331ce1d2ecaa23810b3f9a39fbb0ed0936e9a24e7944049571584f72fa5b31ad98248423c41b0d95490107d1a6b95761b0749eda23f",
        );

        let decoded = HachiBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(&proof_bytes),
            &proof_shape,
        )
        .expect("proof deserialize");
        assert_eq!(decoded, proof);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"protocol_regression_vectors/onehot_nv32");
        <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("verify");
    });
}

#[test]
fn production_dense_nv26_proof_regression_vector() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let nv = 26;
        let layout = DenseCfg::commitment_layout(nv).expect("layout");
        let poly = make_dense_poly(nv, 0xd00d_f00d);
        let point = random_point(nv, 0xdecaf_bad);
        let expected_opening = opening_from_poly(&poly, &point, &layout);

        let setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_prover(nv, 1, 1);
        let verifier_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::commit(commit_input, &setup)
        .expect("commit");

        let poly_refs: [&DensePoly<F, DENSE_D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"protocol_regression_vectors/dense_nv26");
        let proof = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::batched_prove(
            &setup,
            prove_input(&point, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        let proof_shape = proof.shape();
        let proof_bytes = compressed_bytes(&proof);
        assert_fixture(
            "production_dense_nv26_proof",
            &proof_bytes,
            132672,
            "db6043b1e883688e3c63f69c2a89e45caae38f36198a9f1223c689985ba6647d05b132d50fe1f3f54cec214244e7bc10462dcf31a56078e7e9590d7eb6f5bfd5",
        );

        let decoded = HachiBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(&proof_bytes),
            &proof_shape,
        )
        .expect("proof deserialize");
        assert_eq!(decoded, proof);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"protocol_regression_vectors/dense_nv26");
        <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&point, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("verify");
    });
}
