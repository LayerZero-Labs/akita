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

/// Borrowed storage view for the Stage-2 witness polynomial.
///
/// Both arms are live flat evaluations. Reads outside the live range return
/// implicit zero, matching the relation-weight zero-extension convention.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WitnessPolynomial<'a, E: FieldCore> {
    CompactDigits(&'a [i8]),
    FieldEvals(&'a [E]),
}

impl<'a, E: FieldCore> WitnessPolynomial<'a, E> {
    #[inline]
    pub(crate) fn live_len(&self) -> usize {
        match self {
            Self::CompactDigits(evals) => evals.len(),
            Self::FieldEvals(evals) => evals.len(),
        }
    }

    #[inline]
    pub(crate) fn compact_pair(&self, idx0: usize, idx1: usize) -> Option<(i64, i64)> {
        match self {
            Self::CompactDigits(evals) => {
                let w0 = evals.get(idx0).copied().unwrap_or(0) as i64;
                let w1 = evals.get(idx1).copied().unwrap_or(0) as i64;
                Some((w0, w1))
            }
            Self::FieldEvals(_) => None,
        }
    }

    #[inline]
    pub(crate) fn field_pair(&self, idx0: usize, idx1: usize) -> Option<(E, E)> {
        match self {
            Self::CompactDigits(_) => None,
            Self::FieldEvals(evals) => {
                let w0 = evals.get(idx0).copied().unwrap_or_else(E::zero);
                let w1 = evals.get(idx1).copied().unwrap_or_else(E::zero);
                Some((w0, w1))
            }
        }
    }
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
        let witness = WitnessPolynomial::<E>::CompactDigits(w_compact);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert!(witness.live_len() <= 2 * num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_accum) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 2], [E::MulU64Accum::zero(); 6]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::MulU64Accum::zero(); 2];
                    let mut pairs = BlockedFlatPairs::for_block(num_first, num_second, j_high);
                    while let Some(step) = pairs.next_pair() {
                        let (w0, w1) = witness
                            .compact_pair(step.idx0, step.idx1)
                            .expect("compact scan uses compact witness storage");
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
                        let (w0, w1) = witness
                            .compact_pair(step.idx0, step.idx1)
                            .expect("compact scan uses compact witness storage");
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
        let witness = WitnessPolynomial::FieldEvals(w_full);
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let num_first = e_first.len();
        let num_second = e_second.len();
        debug_assert!(witness.live_len() <= 2 * num_first * num_second);

        if self.can_skip_norm_linear_coeff() {
            let (virt_coeffs, rel_coeffs) = cfg_fold_reduce!(
                0..num_second,
                || ([E::zero(); 2], [E::zero(); 3]),
                |(mut virt, mut rel), j_high| {
                    let mut inner_virt = [E::zero(); 2];
                    let mut pairs = BlockedFlatPairs::for_block(num_first, num_second, j_high);
                    while let Some(step) = pairs.next_pair() {
                        let (w0, w1) = witness
                            .field_pair(step.idx0, step.idx1)
                            .expect("full scan uses field witness storage");
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
                        let (w0, w1) = witness
                            .field_pair(step.idx0, step.idx1)
                            .expect("full scan uses field witness storage");
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

    #[tracing::instrument(skip_all, name = "AkitaStage2Prover::scan_round")]
    pub(super) fn scan_round(
        &self,
        witness: WitnessPolynomial<'_, E>,
    ) -> (NormRoundTerms<E>, [E; 3]) {
        match witness {
            WitnessPolynomial::CompactDigits(w) => self.scan_round_compact_blocked(w),
            WitnessPolynomial::FieldEvals(w) => self.scan_round_full_blocked(w),
        }
    }

    #[cfg(test)]
    pub(super) fn compute_round_compact_dense_polys(
        &self,
        w_compact: &[i8],
    ) -> (UniPoly<E>, UniPoly<E>) {
        let (virt_q_coeffs, rel_coeffs) =
            self.scan_round(WitnessPolynomial::CompactDigits(w_compact));
        self.polys_from_terms(virt_q_coeffs, rel_coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn witness_polynomial_compact_reads_use_flat_zero_extension() {
        let witness = WitnessPolynomial::<F>::CompactDigits(&[-2, 3, 5]);

        assert_eq!(witness.live_len(), 3);
        assert_eq!(witness.compact_pair(0, 2), Some((-2, 5)));
        assert_eq!(witness.compact_pair(2, 3), Some((5, 0)));
        assert_eq!(witness.compact_pair(4, 5), Some((0, 0)));
        assert_eq!(witness.field_pair(0, 1), None);
    }

    #[test]
    fn witness_polynomial_field_reads_use_flat_zero_extension() {
        let evals = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
        let witness = WitnessPolynomial::FieldEvals(&evals);

        assert_eq!(witness.live_len(), 3);
        assert_eq!(
            witness.field_pair(1, 2),
            Some((F::from_u64(11), F::from_u64(13)))
        );
        assert_eq!(witness.field_pair(2, 3), Some((F::from_u64(13), F::zero())));
        assert_eq!(witness.field_pair(4, 5), Some((F::zero(), F::zero())));
        assert_eq!(witness.compact_pair(0, 1), None);
    }
}
