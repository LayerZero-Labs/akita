//! Tier-A vs `AkitaStage2Prover` round-equivalence gate.
//!
//! The optimized stage-2 prover is registered verbatim as a fast path; these
//! tests assert its per-round polynomials match the descriptor-driven Tier-A
//! engine on the same witness layout.

use super::{new_stage2_test_prover, Stage2Params, F};
use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{AkitaError, FieldCore, FromPrimitiveInt};
use akita_protocol::ids::{AkitaChallengeId, AkitaOpeningId, AkitaPublicId};
use akita_protocol::{matches_stage2_intermediate_descriptor, stage2_descriptor, LevelRole};
use akita_sumcheck::{
    assert_round_polynomial_equivalence, InstanceProverFastPath, PublicBinding, SumcheckEngine,
    SumcheckInstanceProver,
};
use akita_witness::PolynomialView;

fn hypercube_len(num_vars: usize) -> usize {
    1usize << num_vars
}

/// Witness hypercube table: index `(x << ring_bits) | y` holds `w[x][y]`.
fn build_w_hypercube<E: FieldCore + FromPrimitiveInt>(
    w_compact: &[i8],
    live_x_cols: usize,
    ring_bits: usize,
) -> Vec<E> {
    let y_len = hypercube_len(ring_bits);
    let num_vars = ring_bits + (live_x_cols.next_power_of_two().trailing_zeros() as usize);
    let mut table = vec![E::zero(); hypercube_len(num_vars)];
    for x in 0..live_x_cols {
        for y in 0..y_len {
            let idx = (x << ring_bits) | y;
            table[idx] = E::from_i64(w_compact[x * y_len + y] as i64);
        }
    }
    table
}

/// `alpha(y)` extended to the full hypercube (constant along each x column).
fn build_alpha_hypercube<E: FieldCore>(
    alpha_evals_y: &[E],
    col_bits: usize,
    ring_bits: usize,
) -> Vec<E> {
    let num_vars = col_bits + ring_bits;
    let y_mask = hypercube_len(ring_bits) - 1;
    let mut table = Vec::with_capacity(hypercube_len(num_vars));
    for idx in 0..hypercube_len(num_vars) {
        let y = idx & y_mask;
        table.push(alpha_evals_y[y]);
    }
    table
}

/// `m(x)` extended to the full hypercube (constant along each y row).
fn build_m_hypercube<E: FieldCore>(m_evals_x: &[E], col_bits: usize, ring_bits: usize) -> Vec<E> {
    let num_vars = col_bits + ring_bits;
    let mut table = Vec::with_capacity(hypercube_len(num_vars));
    for idx in 0..hypercube_len(num_vars) {
        let x = idx >> ring_bits;
        table.push(m_evals_x[x]);
    }
    table
}

#[allow(clippy::too_many_arguments)]
fn build_stage2_tier_a_engine<E: FieldCore + FromPrimitiveInt>(
    batching_coeff: E,
    w_compact: &[i8],
    stage1_point: &[E],
    alpha_evals_y: &[E],
    m_evals_x: &[E],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
    input_claim: E,
) -> Result<SumcheckEngine<E>, AkitaError> {
    let num_rounds = col_bits + ring_bits;
    let descriptor = stage2_descriptor(num_rounds, LevelRole::Intermediate);
    debug_assert!(matches_stage2_intermediate_descriptor(&descriptor));

    let eq_table = EqPolynomial::evals(stage1_point)?;
    let w_table = build_w_hypercube::<E>(w_compact, live_x_cols, ring_bits);
    let alpha_table = build_alpha_hypercube(alpha_evals_y, col_bits, ring_bits);
    let m_table = build_m_hypercube(m_evals_x, col_bits, ring_bits);

    SumcheckEngine::new(
        &descriptor,
        input_claim,
        |opening| match opening {
            AkitaOpeningId::Witness => PolynomialView::new(num_rounds, &w_table),
        },
        |public| match public {
            AkitaPublicId::EqStage1Point => Ok(PublicBinding::Multilinear(PolynomialView::new(
                num_rounds, &eq_table,
            )?)),
            AkitaPublicId::Alpha => Ok(PublicBinding::Multilinear(PolynomialView::new(
                num_rounds,
                &alpha_table,
            )?)),
            AkitaPublicId::RelationRow => Ok(PublicBinding::Multilinear(PolynomialView::new(
                num_rounds, &m_table,
            )?)),
        },
        |challenge| match challenge {
            AkitaChallengeId::BatchingCoeff => Ok(batching_coeff),
        },
    )
}

#[test]
fn stage2_fast_path_matches_tier_a_round_by_round() {
    // ring_bits = 1 keeps two_round_prefix off (requires ring_bits >= 2).
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
        let stage2 = new_stage2_test_prover(
            F::from_u64(13),
            w_compact.clone(),
            alpha_evals_y.clone(),
            m_evals_x.clone(),
            params,
        );
        let input_claim = stage2.input_claim();
        let tier_a = build_stage2_tier_a_engine(
            F::from_u64(13),
            &w_compact,
            &stage1_point,
            &alpha_evals_y,
            &m_evals_x,
            x_len,
            col_bits,
            ring_bits,
            input_claim,
        )
        .expect("tier-A engine builds");

        let mut fast = InstanceProverFastPath::new(stage2);
        let mut reference = tier_a;

        assert_round_polynomial_equivalence(&mut reference, &mut fast, |round| {
            F::from_u64((round as u64) + 37)
        })
        .unwrap_or_else(|err| panic!("stage-2 fast path must match Tier A for b={b}: {err:?}"));
    }
}

#[test]
fn stage2_fast_path_matches_tier_a_with_live_column_padding() {
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
    let stage2 = new_stage2_test_prover(
        F::from_u64(17),
        w_prefix.clone(),
        alpha_evals_y.clone(),
        m_evals_x.clone(),
        params,
    );
    let input_claim = stage2.input_claim();
    let tier_a = build_stage2_tier_a_engine(
        F::from_u64(17),
        &w_prefix,
        &stage1_point,
        &alpha_evals_y,
        &m_evals_x,
        live_x_cols,
        col_bits,
        ring_bits,
        input_claim,
    )
    .expect("tier-A engine builds");

    let mut fast = InstanceProverFastPath::new(stage2);
    let mut reference = tier_a;

    assert_round_polynomial_equivalence(&mut reference, &mut fast, |round| {
        F::from_u64((round as u64) + 53)
    })
    .expect("live-column padding must not break fast-path equivalence");
}
