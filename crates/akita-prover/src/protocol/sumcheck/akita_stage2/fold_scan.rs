//! Unified witness/relation fold schedule and fused fold-scan entry points.

use super::*;

/// Fused fold-and-scan paths that preserve hot-cache round messages.
pub(crate) enum FusedFoldScan<'a, E: FieldCore> {
    InitialRound2 {
        w_compact: &'a [i8],
        relation_round2: &'a [E],
        r0: E,
        r1: E,
    },
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    pub(super) fn fold_witness_flat_compact(
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
        fold_live_evals_zero_padded(&evals, challenge)
    }

    pub(super) fn fold_witness_through_two_challenges(w_compact: &[i8], r0: E, r1: E) -> Vec<E> {
        let lut0 = Self::build_compact_w_fold_lut(w_compact, r0);
        let after_r0 = Self::fold_witness_flat_compact(w_compact, &lut0);
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

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn fold_relation_weight_through_two_challenges(evals: &[E], r0: E, r1: E) -> Vec<E> {
        let after_r0 = fold_live_evals_zero_padded(evals, r0);
        fold_live_evals_zero_padded(&after_r0, r1)
    }

    pub(super) fn fold_relation_weight_initial_batch(
        evals: &[E],
        live_segments: usize,
        coeff_len: usize,
        r0: E,
        r1: E,
    ) -> Vec<E> {
        debug_assert!(coeff_len.is_power_of_two());
        debug_assert!(coeff_len >= 4);
        let next_coeff_len = coeff_len >> 2;
        let mut out = vec![E::zero(); live_segments * next_coeff_len];
        for segment in 0..live_segments {
            let src_start = segment * coeff_len;
            let dst_start = segment * next_coeff_len;
            let column = &evals[src_start..src_start + coeff_len];
            for (quad, dst) in out[dst_start..dst_start + next_coeff_len]
                .iter_mut()
                .enumerate()
            {
                let base = 4 * quad;
                *dst = Self::direct_fold_e_quad_to_round2(
                    column[base],
                    column[base + 1],
                    column[base + 2],
                    column[base + 3],
                    r0,
                    r1,
                );
            }
        }
        out
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn fold_witness_compact_to_field(
        w_compact: &[i8],
        fold_lut: &CompactPairFoldLut<E>,
    ) -> Vec<E> {
        Self::fold_witness_flat_compact(w_compact, fold_lut)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn fold_relation_field_flat(evals: &[E], challenge: E) -> Vec<E> {
        fold_live_evals_zero_padded(evals, challenge)
    }

    pub(super) fn run_fused_fold_scan(
        &self,
        fused: FusedFoldScan<'_, E>,
    ) -> (Vec<E>, NormRoundTerms<E>, [E; 3]) {
        match fused {
            FusedFoldScan::InitialRound2 {
                w_compact,
                relation_round2,
                r0,
                r1,
            } => self.fused_fold_scan_initial_round2(w_compact, relation_round2, r0, r1),
        }
    }
}
