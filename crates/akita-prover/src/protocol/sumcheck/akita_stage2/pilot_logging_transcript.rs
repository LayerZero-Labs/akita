//! Logging-transcript event equality for the stage-2 clear sumcheck pilot.
//!
//! Locks that [`prove_clear`] and legacy [`SumcheckInstanceProverExt::prove`]
//! emit identical absorb/squeeze schedules, and that a matching toy verifier
//! replays the same schedule on the produced proof.

#![cfg(all(feature = "logging-transcript", not(feature = "zk")))]

use super::stage2_pilot::prove_clear;
use super::{new_stage2_test_prover, Stage2Params, F};
use crate::protocol::sumcheck::akita_stage1::pad_compact_witness;
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::AkitaError;
use akita_sumcheck::{
    multilinear_eval, SumcheckInstanceProverExt, SumcheckInstanceVerifier,
    SumcheckInstanceVerifierExt,
};
use akita_transcript::labels as tr_labels;
use akita_transcript::{AkitaTranscript, LoggingTranscript, Transcript};

fn new_logging_transcript() -> LoggingTranscript<AkitaTranscript<F>> {
    LoggingTranscript::wrap(<AkitaTranscript<F> as Transcript<F>>::new(
        tr_labels::DOMAIN_AKITA_PROTOCOL,
    ))
}

fn sample_round(tr: &mut LoggingTranscript<AkitaTranscript<F>>) -> F {
    tr.challenge_scalar(tr_labels::CHALLENGE_SUMCHECK_ROUND)
}

fn witness_eval_table(
    w_compact: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Vec<F> {
    let padded = if live_x_cols == (1usize << col_bits) {
        w_compact.to_vec()
    } else {
        pad_compact_witness(w_compact, live_x_cols, col_bits, ring_bits)
    };
    padded.iter().map(|&w| F::from_i64(w as i64)).collect()
}

struct Stage2ToyVerifier {
    batching_coeff: F,
    s_claim: F,
    relation_claim: F,
    stage1_point: Vec<F>,
    w_evals: Vec<F>,
    alpha_evals_y: Vec<F>,
    m_evals_x: Vec<F>,
    col_bits: usize,
    ring_bits: usize,
}

impl Stage2ToyVerifier {
    fn from_fixture(
        batching_coeff: F,
        w_compact: &[i8],
        stage1_point: &[F],
        s_claim: F,
        alpha_evals_y: &[F],
        m_evals_x: &[F],
        params: &Stage2Params<'_>,
        relation_claim: F,
    ) -> Self {
        Self {
            batching_coeff,
            s_claim,
            relation_claim,
            stage1_point: stage1_point.to_vec(),
            w_evals: witness_eval_table(
                w_compact,
                params.live_x_cols,
                params.col_bits,
                params.ring_bits,
            ),
            alpha_evals_y: alpha_evals_y.to_vec(),
            m_evals_x: m_evals_x.to_vec(),
            col_bits: params.col_bits,
            ring_bits: params.ring_bits,
        }
    }
}

impl SumcheckInstanceVerifier<F> for Stage2ToyVerifier {
    fn num_rounds(&self) -> usize {
        self.col_bits + self.ring_bits
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn input_claim(&self) -> F {
        self.batching_coeff * self.s_claim + self.relation_claim
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, AkitaError> {
        let eq = EqPolynomial::mle(&self.stage1_point, challenges)?;
        let w = multilinear_eval(&self.w_evals, challenges)?;
        let (y_challenges, x_challenges) = challenges.split_at(self.ring_bits);
        let alpha = multilinear_eval(&self.alpha_evals_y, y_challenges)?;
        let row = multilinear_eval(&self.m_evals_x, x_challenges)?;
        Ok(self.batching_coeff * eq * w * (w + F::one()) + w * alpha * row)
    }
}

fn assert_transcript_events_match(
    left: &LoggingTranscript<AkitaTranscript<F>>,
    right: &LoggingTranscript<AkitaTranscript<F>>,
    label: &str,
) {
    // Isolated stage-2 sumcheck has no descriptor preamble; compare schedules only.
    assert_eq!(
        left.events(),
        right.events(),
        "{label}: transcript event streams differ"
    );
}

#[test]
fn stage2_pilot_legacy_and_clear_transcript_events_match() {
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

    for b in [4usize, 8usize] {
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

        let mut tr_legacy = new_logging_transcript();
        legacy
            .prove::<F, _, _>(&mut tr_legacy, sample_round)
            .unwrap();

        let mut tr_pilot = new_logging_transcript();
        prove_clear(&mut pilot, &mut tr_pilot, sample_round).unwrap();

        assert_transcript_events_match(&tr_legacy, &tr_pilot, "legacy vs prove_clear");
    }
}

#[test]
fn stage2_pilot_prover_verifier_transcript_events_match() {
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

    let batching_coeff = F::from_u64(17);
    let s_claim = super::s_claim_from_compact_rows(&w_prefix, &params);
    let relation_claim =
        super::relation_claim_from_compact_rows(&w_prefix, &alpha_evals_y, &m_evals_x, &params);

    let mut prover = new_stage2_test_prover(
        batching_coeff,
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        params,
    );
    let verifier = Stage2ToyVerifier::from_fixture(
        batching_coeff,
        &w_prefix,
        &stage1_point,
        s_claim,
        &alpha_evals_y,
        &m_evals_x,
        &params,
        relation_claim,
    );

    let mut prover_tr = new_logging_transcript();
    let (proof, _, _) = prove_clear(&mut prover, &mut prover_tr, sample_round).unwrap();

    let mut verifier_tr = new_logging_transcript();
    verifier
        .verify::<F, _, _>(&proof, &mut verifier_tr, sample_round)
        .unwrap();

    assert_transcript_events_match(&prover_tr, &verifier_tr, "prove_clear vs verify");
}
