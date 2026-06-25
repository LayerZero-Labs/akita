//! Byte-equality gate for the stage-2 clear sumcheck pilot.

use super::stage2_pilot::{prove_clear, prove_clear_via_registry};
use super::{new_stage2_test_prover, Stage2Params, F};
use akita_serialization::{AkitaSerialize, Compress};
use akita_sumcheck::{SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::labels as tr_labels;
use akita_transcript::{AkitaTranscript, Transcript};

fn new_transcript() -> AkitaTranscript<F> {
    <AkitaTranscript<F> as Transcript<F>>::new(tr_labels::DOMAIN_AKITA_PROTOCOL)
}

fn sample_round(tr: &mut AkitaTranscript<F>) -> F {
    tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
}

fn proof_wire_bytes(proof: &SumcheckProof<F>) -> Vec<u8> {
    let mut buf = Vec::new();
    proof
        .serialize_with_mode(&mut buf, Compress::No)
        .expect("sumcheck proof serializes");
    buf
}

fn assert_proof_bytes_match(legacy: &SumcheckProof<F>, pilot: &SumcheckProof<F>, label: &str) {
    assert_eq!(legacy, pilot, "{label}: decoded proof mismatch");
    assert_eq!(
        proof_wire_bytes(legacy),
        proof_wire_bytes(pilot),
        "{label}: serialized proof bytes mismatch"
    );
}

#[test]
fn stage2_pilot_clear_matches_legacy_driver() {
    let col_bits = 3usize;
    let ring_bits = 1usize;
    let y_len = 1usize << ring_bits;
    let x_len = 1usize << col_bits;
    let n = x_len * y_len;
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((i as u64) + 2))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((3 * i as u64) + 5))
        .collect();
    let m_evals_x: Vec<F> = (0..x_len)
        .map(|i| F::from_u64((7 * i as u64) + 11))
        .collect();

    for b in [4usize, 8usize, 16usize] {
        let half = (b / 2) as i8;
        let w_compact: Vec<i8> = (0..n).map(|i| ((i * 5 + 3) % b) as i8 - half).collect();
        let params = Stage2Params {
            stage1_point: &stage1_point,
            b,
            live_x_cols: x_len,
            col_bits,
            ring_bits,
        };

        let mut legacy = new_stage2_test_prover(
            F::from_u64(13),
            w_compact.clone(),
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            params,
        );
        let mut pilot = new_stage2_test_prover(
            F::from_u64(13),
            w_compact,
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            params,
        );

        let mut tr_legacy = new_transcript();
        let (proof_legacy, r_legacy, claim_legacy) = legacy
            .prove::<F, _, _>(&mut tr_legacy, sample_round)
            .unwrap();

        let mut tr_pilot = new_transcript();
        let (proof_pilot, r_pilot, claim_pilot) =
            prove_clear(&mut pilot, &mut tr_pilot, sample_round).unwrap();

        assert_proof_bytes_match(&proof_legacy, &proof_pilot, "prove_clear");
        assert_eq!(r_legacy, r_pilot);
        assert_eq!(claim_legacy, claim_pilot);
    }
}

#[test]
fn stage2_pilot_registry_matches_legacy_driver() {
    let ring_bits = 1usize;
    let live_x_cols = 5usize;
    let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
    let y_len = 1usize << ring_bits;
    let b = 8usize;
    let half = (b / 2) as i8;
    let w_prefix: Vec<i8> = (0..(live_x_cols * y_len))
        .map(|i| ((i * 7 + 5) % b) as i8 - half)
        .collect();
    let stage1_point: Vec<F> = (0..(col_bits + ring_bits))
        .map(|i| F::from_u64((i as u64) + 31))
        .collect();
    let alpha_evals_y: Vec<F> = (0..y_len)
        .map(|i| F::from_u64((5 * i as u64) + 7))
        .collect();
    let m_evals_x: Vec<F> = (0..(1usize << col_bits))
        .map(|i| F::from_u64((11 * i as u64) + 13))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b,
        live_x_cols,
        col_bits,
        ring_bits,
    };

    let legacy = new_stage2_test_prover(
        F::from_u64(17),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        params,
    );
    let pilot_prover =
        new_stage2_test_prover(F::from_u64(17), w_prefix, alpha_evals_y, m_evals_x, params);

    let mut legacy = legacy;
    let mut tr_legacy = new_transcript();
    let (proof_legacy, r_legacy, claim_legacy) = legacy
        .prove::<F, _, _>(&mut tr_legacy, sample_round)
        .unwrap();

    let mut tr_registry = new_transcript();
    let (proof_registry, r_registry, claim_registry) = prove_clear_via_registry(
        pilot_prover,
        col_bits + ring_bits,
        &mut tr_registry,
        sample_round,
    )
    .unwrap();

    assert_proof_bytes_match(&proof_legacy, &proof_registry, "prove_clear_via_registry");
    assert_eq!(r_legacy, r_registry);
    assert_eq!(claim_legacy, claim_registry);
}
