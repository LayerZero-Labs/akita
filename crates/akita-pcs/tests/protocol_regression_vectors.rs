#![allow(missing_docs)]

mod common;

use akita_algebra::{CyclotomicRing, Fp64};
use akita_field::{CanonicalField, FromSmallInt};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_prover::MultilinearPolynomial;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::Transcript;
use akita_transcript::{labels, Blake2bTranscript, KeccakTranscript};
use akita_types::Step;
use akita_types::{AkitaBatchedProof, FlatRingVec, PackedDigits, RingCommitment};
use akita_types::{AkitaRootBatchSummary, AkitaScheduleLookupKey, ScheduleProvider};
use akita_verifier::CommitmentVerifier;
use blake2::{Blake2b512, Digest};
use common::*;

type FixtureField = Fp64<4294967197>;

// Temporary cutover guard for the Akita crate decomposition.
//
// These vectors deliberately pin exact bytes for the current monolithic
// implementation so crate extraction can prove it preserved the protocol
// surface. They are useful ONLY for this crate-decomposition cutover and are
// not intended to become permanent protocol test vectors:
// new protocol features, schedule-table updates, proof layout changes, or
// planner changes will make them stale quickly. Retire these byte-for-byte
// vectors after the crate decomposition cutover is complete.

fn digest_hex(bytes: &[u8]) -> String {
    let digest = Blake2b512::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn compressed_bytes(value: &impl AkitaSerialize) -> Vec<u8> {
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

fn best_full_d(nv: usize) -> usize {
    best_full_d_for_key(AkitaScheduleLookupKey::singleton(nv, nv, 1))
}

fn best_full_d_for_key(key: AkitaScheduleLookupKey) -> usize {
    [
        (
            32,
            fp128::D32Full::schedule_plan(key).expect("D32 full schedule lookup"),
        ),
        (
            64,
            fp128::D64Full::schedule_plan(key).expect("D64 full schedule lookup"),
        ),
        (
            128,
            fp128::D128Full::schedule_plan(key).expect("D128 full schedule lookup"),
        ),
    ]
    .into_iter()
    .filter_map(|(d, plan)| plan.map(|plan| (d, plan.exact_proof_bytes)))
    .min_by_key(|&(_, bytes)| bytes)
    .map(|(d, _)| d)
    .expect("at least one full schedule should exist")
}

fn make_dense_poly_for<const D: usize>(nv: usize, seed: u64) -> DensePoly<F, D> {
    let mut rng = StdRng::seed_from_u64(seed);
    let evals: Vec<F> = (0..1usize << nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    DensePoly::<F, D>::from_field_evals(nv, &evals).expect("dense poly")
}

fn make_onehot_poly_for<const D: usize>(
    layout: &LevelParams,
    seed: u64,
) -> (OneHotPoly<F, D, u8>, Vec<Option<u8>>) {
    let total_ring = layout.num_blocks * layout.block_len;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..D) as u8))
        .collect();
    let poly = OneHotPoly::<F, D, u8>::new(D, indices.clone()).expect("onehot poly");
    (poly, indices)
}

fn planned_batch_root_layout<Cfg: CommitmentConfig>(
    nv: usize,
    total_claims: usize,
    batch: AkitaRootBatchSummary,
) -> LevelParams {
    let schedule =
        Cfg::get_params_for_prove(nv, nv, total_claims, batch).expect("planned batch schedule");
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => root_step.params.clone(),
        Some(Step::Direct(_)) => Cfg::get_params_for_commitment(nv, total_claims)
            .expect("direct batch commitment params"),
        None => panic!("planned batch schedule is empty"),
    }
}

fn onehot_lagrange_opening_for<const D: usize>(indices: &[Option<u8>], point: &[F]) -> F {
    assert_eq!(indices.len() * D, 1usize << point.len());
    indices
        .iter()
        .enumerate()
        .filter_map(|(chunk_idx, hot_idx)| hot_idx.map(|hot_idx| chunk_idx * D + hot_idx as usize))
        .fold(F::zero(), |acc, field_pos| {
            acc + point
                .iter()
                .enumerate()
                .fold(F::one(), |weight, (bit, &r)| {
                    if ((field_pos >> bit) & 1) == 1 {
                        weight * r
                    } else {
                        weight * (F::one() - r)
                    }
                })
        })
}

#[test]
fn transcript_challenge_regression_vectors() {
    let mut blake = Blake2bTranscript::<FixtureField>::new(labels::DOMAIN_AKITA_PROTOCOL);
    let mut keccak = KeccakTranscript::<FixtureField>::new(labels::DOMAIN_AKITA_PROTOCOL);

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
fn onehot_proof_regression_vector() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let nv = if cfg!(debug_assertions) { 26 } else { 32 };
        let layout = OneHotCfg::commitment_layout(nv).expect("layout");
        let poly = make_onehot_poly(&layout, 0xabad_1dea);
        let point = random_point(nv, 0xfeed_face);
        let expected_opening = opening_from_poly(&poly, &point, &layout);

        let setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_prover(nv, 1, 1);
        let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::commit(commit_input, &setup)
        .expect("commit");

        let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];

        let transcript_domain = if cfg!(debug_assertions) {
            b"protocol_regression_vectors/onehot_nv26".as_slice()
        } else {
            b"protocol_regression_vectors/onehot_nv32".as_slice()
        };
        let mut prover_transcript = Blake2bTranscript::<F>::new(transcript_domain);
        let proof = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
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
        if cfg!(debug_assertions) {
            assert_fixture(
                "debug_onehot_nv26_proof",
                &proof_bytes,
                72936,
                "261053c5c3b6b8e44a2b3b61bfbe688e4f60f5ba82220f01174697dc0aedd98d6cac8dac4703936841523c23a8dbeb1b60ef7f818346389532a291502d7d6c94",
            );
        } else {
            assert_fixture(
                "production_onehot_nv32_proof",
                &proof_bytes,
                77216,
                "6d12c3dd34ad293437104331ce1d2ecaa23810b3f9a39fbb0ed0936e9a24e7944049571584f72fa5b31ad98248423c41b0d95490107d1a6b95761b0749eda23f",
            );
        }

        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(&proof_bytes),
            &proof_shape,
        )
        .expect("proof deserialize");
        assert_eq!(decoded, proof);

        let mut verifier_transcript = Blake2bTranscript::<F>::new(transcript_domain);
        <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
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
fn dense_proof_regression_vector() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let nv = if cfg!(debug_assertions) { 20 } else { 26 };
        match best_full_d(nv) {
            32 => run_dense_proof_regression_vector::<32, fp128::D32Full>(nv),
            64 => run_dense_proof_regression_vector::<64, fp128::D64Full>(nv),
            128 => run_dense_proof_regression_vector::<128, fp128::D128Full>(nv),
            d => panic!("unsupported dense planner ring dimension {d}"),
        }
    });
}

#[test]
fn mixed_aggregated_batch_regression_vector() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let nv = if cfg!(debug_assertions) { 20 } else { 26 };
        // Mixed dense/one-hot batching is currently exercised under the dense
        // D128 config in the main e2e suite. Keep this cutover vector aligned
        // with that supported surface.
        run_mixed_aggregated_batch_regression_vector::<128, fp128::D128Full>(nv);
    });
}

#[test]
fn dense_multipoint_batch_regression_vector() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let nv = if cfg!(debug_assertions) { 20 } else { 26 };
        // Multipoint batching coverage follows the existing dense D128 e2e
        // path. A future planner-selector API can decide whether D32 should
        // become the default for this shape.
        run_dense_multipoint_batch_regression_vector::<128, fp128::D128Full>(nv);
    });
}

fn run_dense_proof_regression_vector<const D: usize, Cfg: CommitmentConfig<Field = F>>(nv: usize) {
    let layout = Cfg::commitment_layout(nv).expect("layout");
    let poly = make_dense_poly_for::<D>(nv, 0xd00d_f00d);
    let point = random_point(nv, 0xdeca_fbad);
    let expected_opening = opening_from_poly(&poly, &point, &layout);

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);

    let commit_input = std::slice::from_ref(&poly);
    let (commitment, hint) =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(commit_input, &setup)
            .expect("commit");

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [expected_opening];

    let transcript_domain = match (cfg!(debug_assertions), D) {
        (true, 32) => b"protocol_regression_vectors/dense_d32_nv20".as_slice(),
        (true, 64) => b"protocol_regression_vectors/dense_d64_nv20".as_slice(),
        (true, 128) => b"protocol_regression_vectors/dense_d128_nv20".as_slice(),
        (false, 32) => b"protocol_regression_vectors/dense_d32_nv26".as_slice(),
        (false, 64) => b"protocol_regression_vectors/dense_d64_nv26".as_slice(),
        (false, 128) => b"protocol_regression_vectors/dense_d128_nv26".as_slice(),
        _ => panic!("unsupported dense vector ring dimension {D}"),
    };
    let mut prover_transcript = Blake2bTranscript::<F>::new(transcript_domain);
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        prove_input(&point, &poly_refs, &commitments[0], hint),
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("prove");

    let proof_shape = proof.shape();
    let proof_bytes = compressed_bytes(&proof);
    match (cfg!(debug_assertions), D) {
        (true, 32) => assert_fixture(
            "debug_dense_d32_nv20_proof",
            &proof_bytes,
            69312,
            "9acc96855030350f481fc49b241f696d4f66746c45687933e942d614a90d247cb6325b927e5e8ad8e93aa77d4a80427c59a46ea55c4c296825fa220ee3de76f9",
        ),
        (true, 64) => assert_fixture(
            "debug_dense_d64_nv20_proof",
            &proof_bytes,
            0,
            "UPDATE_ME",
        ),
        (true, 128) => assert_fixture(
            "debug_dense_d128_nv20_proof",
            &proof_bytes,
            124496,
            "5d982b1e8f5295a9847a1ea42095d8251374b79e14bce10833ceab6d076d7dbdeedec5f4e6a3cc2ce74df7894944788eaef41713eae88d60b93cf30ca20b7848",
        ),
        (false, 32) => assert_fixture(
            "production_dense_d32_nv26_proof",
            &proof_bytes,
            73712,
            "620e925d02e5b323c9cb96ce8b0ba56e305f3ca2898b41829db658a1a73c2c16b9b989f2dda071b407734464edd5a3965ada8c40bd15f52c55fa6d17d0131625",
        ),
        (false, 64) => assert_fixture(
            "production_dense_d64_nv26_proof",
            &proof_bytes,
            0,
            "UPDATE_ME",
        ),
        (false, 128) => assert_fixture(
            "production_dense_d128_nv26_proof",
            &proof_bytes,
            132672,
            "db6043b1e883688e3c63f69c2a89e45caae38f36198a9f1223c689985ba6647d05b132d50fe1f3f54cec214244e7bc10462dcf31a56078e7e9590d7eb6f5bfd5",
        ),
        _ => panic!("unsupported dense vector ring dimension {D}"),
    }

    let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
        &mut std::io::Cursor::new(&proof_bytes),
        &proof_shape,
    )
    .expect("proof deserialize");
    assert_eq!(decoded, proof);

    let mut verifier_transcript = Blake2bTranscript::<F>::new(transcript_domain);
    <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verify_input(&point, &openings, &commitments[0]),
        BasisMode::Lagrange,
    )
    .expect("verify");
}

fn run_mixed_aggregated_batch_regression_vector<
    const D: usize,
    Cfg: CommitmentConfig<Field = F>,
>(
    nv: usize,
) {
    let point_group_counts = [1usize];
    let total_claims = 4usize;
    let batch = AkitaRootBatchSummary::from_claim_group_sizes(&[4], 1).expect("batch summary");
    let layout = planned_batch_root_layout::<Cfg>(nv, total_claims, batch);

    let dense_a = make_dense_poly_for::<D>(nv, 0xba7c_0001);
    let dense_b = make_dense_poly_for::<D>(nv, 0xba7c_0002);
    let (onehot_a, onehot_a_indices) = make_onehot_poly_for::<D>(&layout, 0xba7c_1001);
    let (onehot_b, onehot_b_indices) = make_onehot_poly_for::<D>(&layout, 0xba7c_1002);

    let group = [
        MultilinearPolynomial::dense(&dense_a),
        MultilinearPolynomial::onehot(&onehot_a),
        MultilinearPolynomial::dense(&dense_b),
        MultilinearPolynomial::onehot(&onehot_b),
    ];
    let poly_groups: [&[MultilinearPolynomial<'_, F, D, u8>]; 1] = [&group];

    let point = random_point(nv, 0xba7c_f00d);
    let openings = vec![
        opening_from_poly(&dense_a, &point, &layout),
        onehot_lagrange_opening_for::<D>(&onehot_a_indices, &point),
        opening_from_poly(&dense_b, &point, &layout),
        onehot_lagrange_opening_for::<D>(&onehot_b_indices, &point),
    ];

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        nv,
        total_claims,
        1,
    );
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitments, hints) =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_commit(
            &poly_groups,
            &point_group_counts,
            &setup,
        )
        .expect("mixed batched commit");

    let mut hints = hints.into_iter();
    let prover_claims = vec![(
        point.as_slice(),
        vec![CommittedPolynomials {
            polynomials: group.as_slice(),
            commitment: &commitments[0],
            hint: hints.next().unwrap(),
        }],
    )];

    let transcript_domain = match (cfg!(debug_assertions), D) {
        (true, 32) => b"protocol_regression_vectors/mixed_aggregate_d32_nv20".as_slice(),
        (true, 128) => b"protocol_regression_vectors/mixed_aggregate_d128_nv20".as_slice(),
        (false, 32) => b"protocol_regression_vectors/mixed_aggregate_d32_nv26".as_slice(),
        (false, 128) => b"protocol_regression_vectors/mixed_aggregate_d128_nv26".as_slice(),
        _ => panic!("unsupported mixed aggregate vector ring dimension {D}"),
    };
    let mut prover_transcript = Blake2bTranscript::<F>::new(transcript_domain);
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        prover_claims,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("mixed aggregate prove");

    let proof_shape = proof.shape();
    let proof_bytes = compressed_bytes(&proof);
    match (cfg!(debug_assertions), D) {
        (true, 32) => assert_fixture(
            "debug_mixed_aggregate_d32_nv20_proof",
            &proof_bytes,
            71072,
            "fdccf5d50933cdec970120ad8bb72809156e41f207e4ae0709e9b25b89092c09bd20ed381c9c537215a71041f5cd85dd5ca7e5c81dbc26af1d044049746ae431",
        ),
        (true, 128) => assert_fixture(
            "debug_mixed_aggregate_d128_nv20_proof",
            &proof_bytes,
            128080,
            "0849010272137b3e889908330d45034c4f99d0392f9559b64b1dfe1fd07d0e759b98b0076e3aa5088624766cbd8c0ca86bab0c40852e3bfe86391385f878cb19",
        ),
        (false, 32) => assert_fixture(
            "production_mixed_aggregate_d32_nv26_proof",
            &proof_bytes,
            74816,
            "52237be9829d11b8d718d491607d131e0425d6f33feca2cb95ab6e951e9060bf243cf1fb6fc5a23267858214938acb9dd8d9ba4c1ce5daa6ba5a9a92fe24d43e",
        ),
        (false, 128) => assert_fixture(
            "production_mixed_aggregate_d128_nv26_proof",
            &proof_bytes,
            134832,
            "968c415029588e0ee7767ba048415882c1af7e0fb1229816bf4bacc1e2a0b894fbd95449690747a5e620cb199b31ce649522bffbe553e83fc438d0d54d9b89e6",
        ),
        _ => panic!("unsupported mixed aggregate vector ring dimension {D}"),
    }

    let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
        &mut std::io::Cursor::new(&proof_bytes),
        &proof_shape,
    )
    .expect("mixed aggregate proof deserialize");
    assert_eq!(decoded, proof);

    let verifier_claims = vec![(
        point.as_slice(),
        vec![CommittedOpenings {
            openings: openings.as_slice(),
            commitment: &commitments[0],
        }],
    )];
    let mut verifier_transcript = Blake2bTranscript::<F>::new(transcript_domain);
    <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims,
        BasisMode::Lagrange,
    )
    .expect("mixed aggregate verify");
}

fn run_dense_multipoint_batch_regression_vector<
    const D: usize,
    Cfg: CommitmentConfig<Field = F>,
>(
    nv: usize,
) {
    let point_group_counts = [2usize, 1usize];
    let total_claims = 5usize;
    let batch =
        AkitaRootBatchSummary::from_claim_group_sizes(&[2, 1, 2], 2).expect("batch summary");
    let layout = planned_batch_root_layout::<Cfg>(nv, total_claims, batch);

    let dense_a = make_dense_poly_for::<D>(nv, 0xded5_0001);
    let dense_b = make_dense_poly_for::<D>(nv, 0xded5_0002);
    let dense_c = make_dense_poly_for::<D>(nv, 0xded5_0003);
    let dense_d = make_dense_poly_for::<D>(nv, 0xded5_0004);
    let dense_e = make_dense_poly_for::<D>(nv, 0xded5_0005);

    let group_a = [dense_a, dense_b];
    let group_b = [dense_c];
    let group_c = [dense_d, dense_e];
    let poly_groups: [&[DensePoly<F, D>]; 3] = [&group_a, &group_b, &group_c];

    let point_a = random_point(nv, 0xded5_f00d);
    let point_b = random_point(nv, 0xded5_f11d);
    let opening_a0 = group_a
        .iter()
        .map(|poly| opening_from_poly(poly, &point_a, &layout))
        .collect::<Vec<_>>();
    let opening_a1 = group_b
        .iter()
        .map(|poly| opening_from_poly(poly, &point_a, &layout))
        .collect::<Vec<_>>();
    let opening_b0 = group_c
        .iter()
        .map(|poly| opening_from_poly(poly, &point_b, &layout))
        .collect::<Vec<_>>();

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
        nv,
        total_claims,
        2,
    );
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitments, hints) =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_commit(
            &poly_groups,
            &point_group_counts,
            &setup,
        )
        .expect("dense multipoint commit");

    let mut hints = hints.into_iter();
    let prover_claims = vec![
        (
            point_a.as_slice(),
            vec![
                CommittedPolynomials {
                    polynomials: group_a.as_slice(),
                    commitment: &commitments[0],
                    hint: hints.next().unwrap(),
                },
                CommittedPolynomials {
                    polynomials: group_b.as_slice(),
                    commitment: &commitments[1],
                    hint: hints.next().unwrap(),
                },
            ],
        ),
        (
            point_b.as_slice(),
            vec![CommittedPolynomials {
                polynomials: group_c.as_slice(),
                commitment: &commitments[2],
                hint: hints.next().unwrap(),
            }],
        ),
    ];

    let transcript_domain = match (cfg!(debug_assertions), D) {
        (true, 32) => b"protocol_regression_vectors/dense_multipoint_d32_nv20".as_slice(),
        (true, 128) => b"protocol_regression_vectors/dense_multipoint_d128_nv20".as_slice(),
        (false, 32) => b"protocol_regression_vectors/dense_multipoint_d32_nv26".as_slice(),
        (false, 128) => b"protocol_regression_vectors/dense_multipoint_d128_nv26".as_slice(),
        _ => panic!("unsupported dense multipoint vector ring dimension {D}"),
    };
    let mut prover_transcript = Blake2bTranscript::<F>::new(transcript_domain);
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        prover_claims,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("dense multipoint prove");

    let proof_shape = proof.shape();
    let proof_bytes = compressed_bytes(&proof);
    match (cfg!(debug_assertions), D) {
        (true, 32) => assert_fixture(
            "debug_dense_multipoint_d32_nv20_proof",
            &proof_bytes,
            72768,
            "32761c0e091df22578a5c4611a0268b64ec5d27b7f1e53ac3970cee3d20f0bead2c4b587f5d6966f541990b4c2287acc3da23d66041da607602320df664d9abf",
        ),
        (true, 128) => assert_fixture(
            "debug_dense_multipoint_d128_nv20_proof",
            &proof_bytes,
            132224,
            "d3ccfc8203cc060ecdff049cb5a1611e36bc8abf16ef4ad8bb1b2869ab0704c059967bcc8900e9f0d6cfd95f7fdc60fc97c0686ea3636539d19302be9053181a",
        ),
        (false, 32) => assert_fixture(
            "production_dense_multipoint_d32_nv26_proof",
            &proof_bytes,
            76000,
            "eff2d0144637697330fa066f20e2217dce37f81d06033ca08b176a9b33de7932ce32f0c2405c433e3380952c9a511d420cd0b49f1a1f5cbc7f97619df397f403",
        ),
        (false, 128) => assert_fixture(
            "production_dense_multipoint_d128_nv26_proof",
            &proof_bytes,
            138848,
            "a5886be6c9d083ab0127537781075bf8c49a25023c103d68eb99193dc58a8c56363ad2d282e28feb9f4fdaf8a9553c9fc5af4c2b1e0dd13003f6e0772bf2b5ee",
        ),
        _ => panic!("unsupported dense multipoint vector ring dimension {D}"),
    }

    let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
        &mut std::io::Cursor::new(&proof_bytes),
        &proof_shape,
    )
    .expect("dense multipoint proof deserialize");
    assert_eq!(decoded, proof);

    let verifier_claims = vec![
        (
            point_a.as_slice(),
            vec![
                CommittedOpenings {
                    openings: opening_a0.as_slice(),
                    commitment: &commitments[0],
                },
                CommittedOpenings {
                    openings: opening_a1.as_slice(),
                    commitment: &commitments[1],
                },
            ],
        ),
        (
            point_b.as_slice(),
            vec![CommittedOpenings {
                openings: opening_b0.as_slice(),
                commitment: &commitments[2],
            }],
        ),
    ];
    let mut verifier_transcript = Blake2bTranscript::<F>::new(transcript_domain);
    <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims,
        BasisMode::Lagrange,
    )
    .expect("dense multipoint verify");
}
