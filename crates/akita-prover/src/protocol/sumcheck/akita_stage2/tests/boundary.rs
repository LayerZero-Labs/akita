use super::*;
use crate::protocol::sumcheck::akita_stage1::pad_compact_witness;
use akita_field::AkitaError;
use akita_sumcheck::multilinear_eval;

fn try_new_stage2_prover(
    w_compact: Vec<i8>,
    relation_weight_evals: Vec<F>,
    live_segments: usize,
    segment_bits: usize,
    coeff_bits: usize,
    gamma: F,
) -> Result<AkitaStage2Prover<F>, AkitaError> {
    let coeff_len = 1usize << coeff_bits;
    let stage1_point: Vec<F> = (0..(segment_bits + coeff_bits))
        .map(|i| F::from_u64((i as u64) + 2))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b: 8,
        live_segments,
        segment_bits,
        coeff_bits,
    };
    let alpha_evals_coeff: Vec<F> = (0..coeff_len).map(|i| F::from_u64(i as u64 + 1)).collect();
    let m_evals_segment: Vec<F> = (0..(1usize << segment_bits))
        .map(|i| F::from_u64(i as u64 + 3))
        .collect();
    let relation_weight_claim = relation_weight_claim_from_split(
        &w_compact,
        &alpha_evals_coeff,
        &m_evals_segment,
        None,
        &params,
    );
    let s_claim = s_claim_from_compact_rows(&w_compact, &params);
    AkitaStage2Prover::new(
        gamma,
        w_compact,
        &stage1_point,
        s_claim,
        params.b,
        relation_weight_evals,
        relation_weight_claim,
        live_segments,
        segment_bits,
        coeff_bits,
    )
}

#[test]
fn stage2_constructor_rejects_oversized_witness() {
    let live_segments = 5usize;
    let segment_bits = 3usize;
    let coeff_bits = 2usize;
    let coeff_len = 1usize << coeff_bits;
    let witness_len = live_segments * coeff_len;
    let mut w_compact: Vec<i8> = (0..witness_len).map(|i| (i % 5) as i8).collect();
    let relation_weight_evals = vec![F::zero(); witness_len];
    w_compact.push(1);

    let err = match try_new_stage2_prover(
        w_compact,
        relation_weight_evals,
        live_segments,
        segment_bits,
        coeff_bits,
        F::from_u64(13),
    ) {
        Err(err) => err,
        Ok(_) => panic!("oversized witness must be rejected"),
    };
    assert!(matches!(
        err,
        AkitaError::InvalidSize {
            expected: 20,
            actual: 21
        }
    ));
}

#[test]
fn stage2_constructor_rejects_undersized_witness() {
    let live_segments = 5usize;
    let segment_bits = 3usize;
    let coeff_bits = 2usize;
    let coeff_len = 1usize << coeff_bits;
    let witness_len = live_segments * coeff_len;
    let w_compact: Vec<i8> = (0..witness_len).map(|i| (i % 5) as i8).collect();
    let relation_weight_evals = vec![F::zero(); witness_len];
    let stage1_point: Vec<F> = (0..(segment_bits + coeff_bits))
        .map(|i| F::from_u64((i as u64) + 2))
        .collect();

    let err = match AkitaStage2Prover::new(
        F::from_u64(13),
        w_compact[..19].to_vec(),
        &stage1_point,
        F::zero(),
        8,
        relation_weight_evals,
        F::zero(),
        live_segments,
        segment_bits,
        coeff_bits,
    ) {
        Err(err) => err,
        Ok(_) => panic!("undersized witness must be rejected"),
    };
    assert!(matches!(
        err,
        AkitaError::InvalidSize {
            expected: 20,
            actual: 19
        }
    ));
}

#[test]
fn stage2_constructor_rejects_relation_weight_length_mismatch() {
    let live_segments = 5usize;
    let segment_bits = 3usize;
    let coeff_bits = 2usize;
    let coeff_len = 1usize << coeff_bits;
    let witness_len = live_segments * coeff_len;
    let w_compact: Vec<i8> = (0..witness_len).map(|i| (i % 5) as i8).collect();
    let relation_weight_evals = vec![F::zero(); witness_len - 1];

    let err = match try_new_stage2_prover(
        w_compact,
        relation_weight_evals,
        live_segments,
        segment_bits,
        coeff_bits,
        F::from_u64(13),
    ) {
        Err(err) => err,
        Ok(_) => panic!("relation weight length mismatch must be rejected"),
    };
    assert!(matches!(
        err,
        AkitaError::InvalidSize {
            expected: 20,
            actual: 19
        }
    ));
}

#[test]
fn gamma_nonzero_range_term_changes_with_arbitrary_padded_witness_advice() {
    let live_segments = 5usize;
    let segment_bits = 3usize;
    let coeff_bits = 2usize;
    let coeff_len = 1usize << coeff_bits;
    let segment_capacity = 1usize << segment_bits;
    let stage1_point: Vec<F> = (0..(segment_bits + coeff_bits))
        .map(|i| F::from_u64((i as u64) + 17))
        .collect();
    let params = Stage2Params {
        stage1_point: &stage1_point,
        b: 8,
        live_segments,
        segment_bits,
        coeff_bits,
    };

    let half = 4i8;
    let w_live: Vec<i8> = (0..(live_segments * coeff_len))
        .map(|i| ((i * 3 + 1) % 8) as i8 - half)
        .collect();

    let s_claim_live = s_claim_from_compact_rows(&w_live, &params);

    let mut w_advice = pad_compact_witness(&w_live, live_segments, segment_bits, coeff_bits);
    for x in live_segments..segment_capacity {
        for y in 0..coeff_len {
            w_advice[x * coeff_len + y] = 7i8;
        }
    }
    let s_evals_advice: Vec<F> = w_advice
        .iter()
        .map(|&w| {
            let w = F::from_i64(w as i64);
            w * (w + F::one())
        })
        .collect();
    let s_claim_advice =
        multilinear_eval(&s_evals_advice, &stage1_point).expect("valid padded witness shape");

    assert_ne!(
        s_claim_live, s_claim_advice,
        "arbitrary padded witness advice must change the virtual range term"
    );

    let gamma = F::from_u64(29);
    let input_claim_live = gamma * s_claim_live;
    let input_claim_advice = gamma * s_claim_advice;
    assert_ne!(
        input_claim_live, input_claim_advice,
        "gamma != 0 couples the range term into the fused claim"
    );

    let err = match try_new_stage2_prover(
        w_advice,
        vec![F::zero(); live_segments * coeff_len],
        live_segments,
        segment_bits,
        coeff_bits,
        gamma,
    ) {
        Err(err) => err,
        Ok(_) => panic!("padded full hypercube witness must be rejected at stage-2 boundary"),
    };
    assert!(matches!(
        err,
        AkitaError::InvalidSize {
            expected: 20,
            actual: 32
        }
    ));
}
