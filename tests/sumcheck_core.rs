#![allow(missing_docs)]

use hachi_pcs::algebra::Fp64;
use hachi_pcs::protocol::transcript::labels;
use hachi_pcs::protocol::{Blake2bTranscript, CompressedUniPoly, SumcheckProof, Transcript, UniPoly};
use hachi_pcs::{CanonicalField, FieldCore, FieldSampling};
use rand::RngCore;
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Fp64<4294967197>;

#[test]
fn compressed_unipoly_round_trip_and_eval() {
    let mut rng = StdRng::seed_from_u64(123);

    for degree in 0..8usize {
        let coeffs: Vec<F> = (0..=degree).map(|_| F::sample(&mut rng)).collect();
        let poly = UniPoly::from_coeffs(coeffs);

        // Hint is g(0) + g(1).
        let hint = poly.evaluate(&F::zero()) + poly.evaluate(&F::one());

        let compressed = poly.compress();
        let decompressed = compressed.decompress(&hint);

        // Decompression should be functionally equivalent (it may materialize
        // a trailing zero linear term for constant polynomials).
        for x_u64 in [0u64, 1, 2, 3, 17] {
            let x = F::from_u64(x_u64);
            let direct = poly.evaluate(&x);
            let decompressed_direct = decompressed.evaluate(&x);
            let via_hint = compressed.eval_from_hint(&hint, &x);
            assert_eq!(direct, decompressed_direct);
            assert_eq!(direct, via_hint);
        }
    }
}

#[test]
fn sumcheck_proof_verifier_driver_is_transcript_deterministic() {
    // This test checks that the verifier driver absorbs messages and samples challenges
    // consistently, and that the returned (final_claim, r_vec) matches a manual replay.
    let mut rng = StdRng::seed_from_u64(999);

    let num_rounds = 5usize;
    let degree_bound = 7usize;

    // Build random per-round univariates (degree <= degree_bound), compress them.
    let round_polys: Vec<CompressedUniPoly<F>> = (0..num_rounds)
        .map(|_| {
            let deg = (rng.next_u32() as usize) % (degree_bound + 1);
            let coeffs: Vec<F> = (0..=deg).map(|_| F::sample(&mut rng)).collect();
            UniPoly::from_coeffs(coeffs).compress()
        })
        .collect();

    let proof = SumcheckProof { round_polys };
    let claim0 = F::sample(&mut rng);

    // Verifier run.
    let mut t1 = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let (final_claim_1, r_1) = proof
        .verify::<F, _, _>(
            claim0,
            num_rounds,
            degree_bound,
            &mut t1,
            |tr| tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND),
        )
        .unwrap();

    // Manual replay with a fresh transcript (must match).
    let mut t2 = Blake2bTranscript::<F>::new(labels::DOMAIN_HACHI_PROTOCOL);
    let mut claim = claim0;
    let mut r_manual = Vec::with_capacity(num_rounds);
    for poly in &proof.round_polys {
        t2.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
        let r_i = t2.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND);
        r_manual.push(r_i);
        claim = poly.eval_from_hint(&claim, &r_i);
    }

    assert_eq!(r_1, r_manual);
    assert_eq!(final_claim_1, claim);
}

