//! Flat witness and relation-weight fold for stage 2.
//!
//! Every round binds one Boolean variable by folding adjacent live pairs
//! `(2j, 2j+1)` with zero extension outside the current live range.

use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    pub(super) fn fold_witness_compact_to_field(
        w_compact: &[i8],
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        cfg_into_iter!(0..w_compact.len().div_ceil(2))
            .map(|j| {
                let left = 2 * j;
                let w0 = i16::from(w_compact[left]);
                let w1 = w_compact.get(left + 1).copied().map(i16::from).unwrap_or(0);
                fold_lut.fold(w0, w1)
            })
            .collect()
    }

    #[inline]
    pub(super) fn build_compact_w_fold_lut(w_compact: &[i8], r: E) -> CompactPairFoldLut<E> {
        let min_w = w_compact
            .iter()
            .copied()
            .map(i32::from)
            .min()
            .unwrap_or(0)
            .min(0);
        let max_w = w_compact
            .iter()
            .copied()
            .map(i32::from)
            .max()
            .unwrap_or(0)
            .max(0);
        CompactPairFoldLut::from_contiguous_range(min_w as i16, max_w as i16, r)
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold> AkitaStage2Prover<E> {
    pub(super) fn fold_witness_field_flat(evals: Vec<E>, challenge: E) -> Vec<E> {
        if evals.len() <= 1 {
            return evals;
        }
        fold_live_evals_zero_padded(&evals, challenge)
    }

    pub(super) fn fold_witness_through_two_challenges(
        w_compact: &[i8],
        r0: E,
        r1: E,
    ) -> Vec<E> {
        let lut0 = Self::build_compact_w_fold_lut(w_compact, r0);
        let after_r0 = Self::fold_witness_compact_to_field(w_compact, &lut0);
        Self::fold_witness_field_flat(after_r0, r1)
    }

    pub(super) fn fold_relation_weight_flat(&mut self, challenge: E) {
        let folded = fold_live_evals_zero_padded(self.relation_weight.evals(), challenge);
        let live_len = folded.len();
        self.relation_weight = RelationWeightPolynomial::from_live_evals(folded, live_len)
            .expect("relation weight flat fold preserves shape");
        self.relation_coeff_len = live_len;
        self.live_segments = 1;
    }

    pub(super) fn fold_relation_field_flat(evals: &[E], challenge: E) -> Vec<E> {
        fold_live_evals_zero_padded(evals, challenge)
    }

    pub(super) fn fold_relation_weight_through_two_challenges(
        evals: &[E],
        r0: E,
        r1: E,
    ) -> Vec<E> {
        let after_r0 = fold_live_evals_zero_padded(evals, r0);
        fold_live_evals_zero_padded(&after_r0, r1)
    }
}
