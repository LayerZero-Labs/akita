//! Flat pair-scan kernel for stage-2 dense rounds.
//!
//! Dense stage-2 rounds now use the same flat-address pair identity as the
//! target kernel: each step emits only `(idx0, idx1)`, and witness/relation
//! reads use the live flat zero-extension convention.

use super::*;

/// One active fold pair in the current round.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PairStep {
    pub idx0: usize,
    pub idx1: usize,
}

/// Visitation order for flat fold pairs in one round.
pub(crate) trait FlatPairStream {
    fn next_pair(&mut self) -> Option<PairStep>;
}

/// Gruen split-eq blocked pair order used by the dense scan path.
pub(crate) struct BlockedFlatPairs {
    num_first: usize,
    num_second: usize,
    j_high: usize,
    j_low: usize,
}

impl BlockedFlatPairs {
    pub(crate) fn for_block(num_first: usize, num_second: usize, j_high: usize) -> Self {
        Self {
            num_first,
            num_second,
            j_high,
            j_low: 0,
        }
    }
}

impl FlatPairStream for BlockedFlatPairs {
    fn next_pair(&mut self) -> Option<PairStep> {
        if self.j_high >= self.num_second || self.j_low >= self.num_first {
            return None;
        }
        let j = self.j_high * self.num_first + self.j_low;
        self.j_low += 1;
        Some(PairStep {
            idx0: 2 * j,
            idx1: 2 * j + 1,
        })
    }
}

#[inline]
fn accumulate_compact_virt_pair<E: FieldCore + HasUnreducedOps>(
    inner_virt: &mut [E::MulU64Accum; 4],
    eq_in: E,
    w0: i64,
    dw: i64,
) {
    let q0 = w0 * (w0 + 1);
    if q0 != 0 {
        inner_virt[0] += eq_in.mul_u64_unreduced(q0 as u64);
    }
    let q1 = dw * (2 * w0 + 1);
    accum_small_signed::<E>(inner_virt, 1, eq_in, q1);
    let q2 = dw * dw;
    if q2 != 0 {
        inner_virt[3] += eq_in.mul_u64_unreduced(q2 as u64);
    }
}

#[inline]
fn accumulate_compact_virt_skip_linear_pair<E: FieldCore + HasUnreducedOps>(
    inner_virt: &mut [E::MulU64Accum; 2],
    eq_in: E,
    w0: i64,
    dw: i64,
) {
    let q0 = w0 * (w0 + 1);
    if q0 != 0 {
        inner_virt[0] += eq_in.mul_u64_unreduced(q0 as u64);
    }
    let q2 = dw * dw;
    if q2 != 0 {
        inner_virt[1] += eq_in.mul_u64_unreduced(q2 as u64);
    }
}

#[inline]
fn accumulate_full_virt_pair<E: FieldCore>(inner_virt: &mut [E; 3], eq_in: E, w0: E, dw: E) {
    inner_virt[0] += eq_in * (w0 * (w0 + E::one()));
    inner_virt[1] += eq_in * (dw * (w0 + w0 + E::one()));
    inner_virt[2] += eq_in * (dw * dw);
}

#[inline]
fn accumulate_full_virt_skip_linear_pair<E: FieldCore>(
    inner_virt: &mut [E; 2],
    eq_in: E,
    w0: E,
    dw: E,
) {
    inner_virt[0] += eq_in * (w0 * (w0 + E::one()));
    inner_virt[1] += eq_in * (dw * dw);
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> AkitaStage2Prover<E> {
    #[tracing::instrument(skip_all, name = "AkitaStage2Prover::scan_round_compact_blocked")]
    pub(super) fn scan_round_compact_blocked(
        &self,
        w_compact: &[i8],
    ) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert_eq!(w_compact.len() / 2, num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 2], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::MulU64Accum::zero(); 2];
                    let mut pairs = BlockedFlatPairs::for_block(num_first, num_second, j_high);
                    while let Some(step) = pairs.next_pair() {
                        let w0 = w_compact.get(step.idx0).copied().unwrap_or(0) as i64;
                        let w1 = w_compact.get(step.idx1).copied().unwrap_or(0) as i64;
                        let dw = w1 - w0;
                        let j_low = (step.idx0 / 2) - j_high * num_first;
                        let eq_in = e_first[j_low];
                        accumulate_compact_virt_skip_linear_pair::<E>(
                            &mut inner_virt,
                            eq_in,
                            w0,
                            dw,
                        );
                        let (p0, p1) = self.relation_weight.pair_flat(step.idx0, step.idx1);
                        accumulate_relation_coeffs_signed(&mut rel, w0, dw, p0, p1);
                    }
                    let reduced_inner: [E; 2] = reduce_compact_virt_skip_linear(inner_virt);
                    let e_out = e_second[j_high];
                    virt[0] += e_out * reduced_inner[0];
                    virt[1] += e_out * reduced_inner[1];
                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );
            (
                NormRoundTerms::SkipLinear(virt_coeffs),
                reduce_compact_rel(rel_accum),
            )
        } else {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 3], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::MulU64Accum::zero(); 4];
                    let mut pairs = BlockedFlatPairs::for_block(num_first, num_second, j_high);
                    while let Some(step) = pairs.next_pair() {
                        let w0 = w_compact.get(step.idx0).copied().unwrap_or(0) as i64;
                        let w1 = w_compact.get(step.idx1).copied().unwrap_or(0) as i64;
                        let dw = w1 - w0;
                        let j_low = (step.idx0 / 2) - j_high * num_first;
                        let eq_in = e_first[j_low];
                        accumulate_compact_virt_pair::<E>(&mut inner_virt, eq_in, w0, dw);
                        let (p0, p1) = self.relation_weight.pair_flat(step.idx0, step.idx1);
                        accumulate_relation_coeffs_signed(&mut rel, w0, dw, p0, p1);
                    }
                    let reduced_inner: [E; 3] = reduce_compact_virt(inner_virt);
                    let e_out = e_second[j_high];
                    virt[0] += e_out * reduced_inner[0];
                    virt[1] += e_out * reduced_inner[1];
                    virt[2] += e_out * reduced_inner[2];
                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );
            (
                NormRoundTerms::Full(virt_coeffs),
                reduce_compact_rel(rel_accum),
            )
        }
    }

    #[tracing::instrument(skip_all, name = "AkitaStage2Prover::scan_round_full_blocked")]
    pub(super) fn scan_round_full_blocked(&self, w_full: &[E]) -> (NormRoundTerms<E>, [E; 3]) {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert_eq!(w_full.len() / 2, num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::zero(); 2];
                    let mut pairs = BlockedFlatPairs::for_block(num_first, num_second, j_high);
                    while let Some(step) = pairs.next_pair() {
                        let w0 = w_full.get(step.idx0).copied().unwrap_or_else(E::zero);
                        let w1 = w_full.get(step.idx1).copied().unwrap_or_else(E::zero);
                        let dw = w1 - w0;
                        let j_low = (step.idx0 / 2) - j_high * num_first;
                        let eq_in = e_first[j_low];
                        accumulate_full_virt_skip_linear_pair::<E>(&mut inner_virt, eq_in, w0, dw);
                        let (p0, p1) = self.relation_weight.pair_flat(step.idx0, step.idx1);
                        accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
                    }
                    let e_out = e_second[j_high];
                    virt[0] += e_out * inner_virt[0];
                    virt[1] += e_out * inner_virt[1];
                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );
            (NormRoundTerms::SkipLinear(virt_coeffs), rel_coeffs)
        } else {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 3], [E::zero(); 3]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::zero(); 3];
                    let mut pairs = BlockedFlatPairs::for_block(num_first, num_second, j_high);
                    while let Some(step) = pairs.next_pair() {
                        let w0 = w_full.get(step.idx0).copied().unwrap_or_else(E::zero);
                        let w1 = w_full.get(step.idx1).copied().unwrap_or_else(E::zero);
                        let dw = w1 - w0;
                        let j_low = (step.idx0 / 2) - j_high * num_first;
                        let eq_in = e_first[j_low];
                        accumulate_full_virt_pair::<E>(&mut inner_virt, eq_in, w0, dw);
                        let (p0, p1) = self.relation_weight.pair_flat(step.idx0, step.idx1);
                        accumulate_relation_coeffs(&mut rel, w0, dw, p0, p1);
                    }
                    let e_out = e_second[j_high];
                    virt[0] += e_out * inner_virt[0];
                    virt[1] += e_out * inner_virt[1];
                    virt[2] += e_out * inner_virt[2];
                    (virt, rel)
                },
                |(mut va, mut ra), (vb, rb)| {
                    for (ai, bi) in va.iter_mut().zip(vb.iter()) {
                        *ai += *bi;
                    }
                    for (ai, bi) in ra.iter_mut().zip(rb.iter()) {
                        *ai += *bi;
                    }
                    (va, ra)
                }
            );
            (NormRoundTerms::Full(virt_coeffs), rel_coeffs)
        }
    }

    #[cfg(test)]
    pub(super) fn compute_round_compact_dense_polys(
        &self,
        w_compact: &[i8],
    ) -> (UniPoly<E>, UniPoly<E>) {
        let (virt_q_coeffs, rel_coeffs) = self.scan_round_compact_blocked(w_compact);
        self.polys_from_terms(virt_q_coeffs, rel_coeffs)
    }
}
