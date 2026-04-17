//! Fused Stage 1 prover: range-check + relation in a single non-eq-factored
//! sumcheck.
//!
//! The fused polynomial is:
//!
//! `f(z) = eq(tau0, z) * Q(S(z)) + gamma_rel * w(z) * alpha(y_z) * M(x_z)`
//!
//! The range-check half reuses the existing `HachiStage1Prover` for computing
//! the inner polynomial `q(X)`, then multiplies by the eq linear factor.
//! The relation half scans the same witness pairs against `alpha_compact` and
//! `m_compact` tables.
//!
//! The combined polynomial has degree `max(leaf_deg + 1, 2) = leaf_deg + 1`
//! per round, where `leaf_deg = b/2` for b <= 8 and 4 for b > 8.

use super::hachi_stage1 as single_stage;
use super::hachi_stage2::accumulate_relation_coeffs;
use super::{
    fold_evals_in_place, EqFactoredSumcheckInstanceProver, SumcheckInstanceProver, UniPoly,
};
use crate::algebra::fields::HasUnreducedOps;
use crate::{CanonicalField, FieldCore, FromSmallInt};

enum RelationWTable<E: FieldCore> {
    Compact(Vec<i8>),
    Full(Vec<E>),
}

/// Fused Stage 1 prover combining the range-check sumcheck with the relation
/// `w * alpha * M` in a single non-eq-factored sumcheck.
pub struct FusedStage1Prover<E: FieldCore> {
    range_prover: single_stage::HachiStage1Prover<E>,
    range_claim: E,

    gamma_rel: E,
    w_table: RelationWTable<E>,
    alpha_compact: Vec<E>,
    m_compact: Vec<E>,
    relation_claim: E,

    col_bits: usize,
    ring_bits: usize,
    num_vars: usize,
    rounds_completed: usize,

    cached_range_poly: Option<UniPoly<E>>,
    cached_rel_poly: Option<UniPoly<E>>,
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> FusedStage1Prover<E> {
    /// Create a fused Stage 1 prover.
    ///
    /// `gamma_rel` batches the relation into the range-check sumcheck.
    /// The compact witness is cloned internally for the relation scan.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w_evals_compact: &[i8],
        tau0: &[E],
        b: usize,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
        gamma_rel: E,
        alpha_evals_y: Vec<E>,
        m_evals_x: Vec<E>,
        relation_claim: E,
    ) -> Self {
        let num_vars = col_bits + ring_bits;
        let y_len = 1usize << ring_bits;

        debug_assert_eq!(w_evals_compact.len(), live_x_cols * y_len);
        debug_assert_eq!(alpha_evals_y.len(), y_len);
        debug_assert_eq!(m_evals_x.len(), 1 << col_bits);

        let range_prover = single_stage::HachiStage1Prover::new(
            w_evals_compact,
            tau0,
            b,
            live_x_cols,
            col_bits,
            ring_bits,
        );

        let w_padded = pad_compact_witness(w_evals_compact, live_x_cols, col_bits, ring_bits);

        Self {
            range_prover,
            range_claim: E::zero(),
            gamma_rel,
            w_table: RelationWTable::Compact(w_padded),
            alpha_compact: alpha_evals_y,
            m_compact: m_evals_x,
            relation_claim,
            col_bits,
            ring_bits,
            num_vars,
            rounds_completed: 0,
            cached_range_poly: None,
            cached_rel_poly: None,
        }
    }

    /// Return `S(r1) = w(r1) * (w(r1) + 1)` after all rounds.
    pub fn final_s_claim(&self) -> E {
        self.range_prover.final_s_claim()
    }

    /// Return `w(r1)` after all rounds.
    ///
    /// # Panics
    ///
    /// Panics if the witness table was not fully folded (i.e., not all rounds
    /// were completed).
    pub fn final_w_eval(&self) -> E {
        match &self.w_table {
            RelationWTable::Full(w_full) => {
                assert_eq!(w_full.len(), 1, "w_table not fully folded");
                w_full[0]
            }
            RelationWTable::Compact(_) => panic!("w_table remained compact after final fold"),
        }
    }

    fn in_y_round(&self) -> bool {
        self.rounds_completed < self.ring_bits
    }

    fn current_y_width(&self) -> usize {
        self.ring_bits
            .saturating_sub(self.rounds_completed.min(self.ring_bits))
    }

    fn current_x_width(&self) -> usize {
        self.col_bits
            .saturating_sub(self.rounds_completed.saturating_sub(self.ring_bits))
    }

    fn compute_relation_poly(&self) -> UniPoly<E> {
        let folding_y = self.in_y_round();
        let current_y_width = self.current_y_width();
        let current_x_width = self.current_x_width();
        let current_y_mask = (1usize << current_y_width).wrapping_sub(1);
        let current_x_mask = (1usize << current_x_width).wrapping_sub(1);

        let alpha = &self.alpha_compact;
        let m = &self.m_compact;

        let mut rel = [E::zero(); 3];

        match &self.w_table {
            RelationWTable::Compact(w_compact) => {
                let half = w_compact.len() / 2;
                for j in 0..half {
                    let w0 = E::from_i64(w_compact[2 * j] as i64);
                    let w1 = E::from_i64(w_compact[2 * j + 1] as i64);
                    let dw = w1 - w0;
                    let (a0, a1, m0, m1) = if folding_y {
                        (
                            alpha[(2 * j) & current_y_mask],
                            alpha[(2 * j + 1) & current_y_mask],
                            m[(2 * j) >> current_y_width],
                            m[(2 * j + 1) >> current_y_width],
                        )
                    } else {
                        (
                            alpha[(2 * j) >> current_x_width],
                            alpha[(2 * j + 1) >> current_x_width],
                            m[(2 * j) & current_x_mask],
                            m[(2 * j + 1) & current_x_mask],
                        )
                    };
                    let p0 = a0 * m0;
                    let p1 = a1 * m1;
                    accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
                }
            }
            RelationWTable::Full(w_full) => {
                let half = w_full.len() / 2;
                for j in 0..half {
                    let w0 = w_full[2 * j];
                    let w1 = w_full[2 * j + 1];
                    let dw = w1 - w0;
                    let (a0, a1, m0, m1) = if folding_y {
                        (
                            alpha[(2 * j) & current_y_mask],
                            alpha[(2 * j + 1) & current_y_mask],
                            m[(2 * j) >> current_y_width],
                            m[(2 * j + 1) >> current_y_width],
                        )
                    } else {
                        (
                            alpha[(2 * j) >> current_x_width],
                            alpha[(2 * j + 1) >> current_x_width],
                            m[(2 * j) & current_x_mask],
                            m[(2 * j + 1) & current_x_mask],
                        )
                    };
                    let p0 = a0 * m0;
                    let p1 = a1 * m1;
                    accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
                }
            }
        }

        UniPoly::from_coeffs(rel.to_vec())
    }

    fn fold_relation_tables(&mut self, r: E) {
        self.w_table = match std::mem::replace(&mut self.w_table, RelationWTable::Full(Vec::new()))
        {
            RelationWTable::Compact(w_compact) => {
                let mut w_full: Vec<E> = w_compact.iter().map(|&w| E::from_i64(w as i64)).collect();
                fold_evals_in_place(&mut w_full, r);
                RelationWTable::Full(w_full)
            }
            RelationWTable::Full(mut w_full) => {
                fold_evals_in_place(&mut w_full, r);
                RelationWTable::Full(w_full)
            }
        };

        if self.in_y_round() {
            fold_evals_in_place(&mut self.alpha_compact, r);
        } else {
            fold_evals_in_place(&mut self.m_compact, r);
        }
    }
}

impl<E: FieldCore + FromSmallInt + CanonicalField + HasUnreducedOps> SumcheckInstanceProver<E>
    for FusedStage1Prover<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        self.range_prover.degree_bound_inner() + 1
    }

    fn input_claim(&self) -> E {
        self.gamma_rel * self.relation_claim
    }

    fn compute_round_univariate(&mut self, round: usize, _previous_claim: E) -> UniPoly<E> {
        let range_poly = self
            .range_prover
            .compute_full_range_check_round(round, self.range_claim);

        let rel_poly = self.compute_relation_poly();

        self.cached_range_poly = Some(range_poly.clone());
        self.cached_rel_poly = Some(rel_poly.clone());

        let range_deg = range_poly.degree();
        let rel_deg = rel_poly.degree();
        let max_deg = range_deg.max(rel_deg);
        let mut combined_coeffs = vec![E::zero(); max_deg + 1];
        for (i, &c) in range_poly.coeffs.iter().enumerate() {
            combined_coeffs[i] += c;
        }
        for (i, &c) in rel_poly.coeffs.iter().enumerate() {
            combined_coeffs[i] += self.gamma_rel * c;
        }

        UniPoly::from_coeffs(combined_coeffs)
    }

    fn ingest_challenge(&mut self, round: usize, r: E) {
        if let Some(rp) = self.cached_range_poly.take() {
            self.range_claim = rp.evaluate(&r);
        }
        if let Some(rp) = self.cached_rel_poly.take() {
            self.relation_claim = rp.evaluate(&r);
        }

        self.range_prover.ingest_challenge(round, r);
        self.fold_relation_tables(r);

        self.rounds_completed += 1;
    }

    fn finalize(&mut self) {
        self.range_prover.finalize();
    }
}

fn pad_compact_witness(
    w_compact: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Vec<i8> {
    let y_len = 1usize << ring_bits;
    let x_len = 1usize << col_bits;
    let full_len = x_len * y_len;
    if live_x_cols == x_len {
        return w_compact.to_vec();
    }
    let mut padded = vec![0i8; full_len];
    for x in 0..live_x_cols {
        let src_start = x * y_len;
        let dst_start = x * y_len;
        padded[dst_start..dst_start + y_len]
            .copy_from_slice(&w_compact[src_start..src_start + y_len]);
    }
    padded
}

/// Pad compact witness (`i8`) to full field-element table for claim reduction.
pub fn pad_compact_witness_field<E: FieldCore + FromSmallInt>(
    w_compact: &[i8],
    live_x_cols: usize,
    col_bits: usize,
    ring_bits: usize,
) -> Vec<E> {
    let y_len = 1usize << ring_bits;
    let x_len = 1usize << col_bits;
    let full_len = x_len * y_len;
    let mut padded = vec![E::zero(); full_len];
    for x in 0..live_x_cols {
        for y in 0..y_len {
            padded[x * y_len + y] = E::from_i64(w_compact[x * y_len + y] as i64);
        }
    }
    padded
}
