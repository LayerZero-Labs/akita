use super::*;

#[allow(clippy::too_many_arguments)]
fn fold_lane_and_compute_next_round<
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps,
    const SKIP_LINEAR: bool,
>(
    prover: &RelationRangeImageProver<E>,
    source: &[E],
    target: &mut [E],
    next_alpha_factor: &[E],
    lane: usize,
    lane_weight: E,
    challenge: E,
    e_first: &[E],
    e_second: &[E],
    first_bits: usize,
    block_size: usize,
) -> ([E; 3], [E; 3]) {
    let next_coeff_count = target.len();
    let next_coefficient_half = next_coeff_count / 2;
    let equality_address_base = lane * next_coefficient_half;
    let mut virt = [E::zero(); 3];
    let mut rel = [E::zero(); 3];
    let mut blk = 0usize;

    while blk < next_coefficient_half {
        let (j_high, blk_end) = stage2_eq_block(
            equality_address_base,
            blk,
            e_first.len(),
            first_bits,
            block_size,
            next_coefficient_half,
        );
        let mut inner_virt = [E::zero(); 3];

        for coefficient_pair in blk..blk_end {
            let left = 2 * coefficient_pair;
            let source_base = 2 * left;
            let w0 =
                source[source_base] + challenge * (source[source_base + 1] - source[source_base]);
            let w1 = source[source_base + 2]
                + challenge * (source[source_base + 3] - source[source_base + 2]);
            target[left] = w0;
            target[left + 1] = w1;
            let dw = w1 - w0;

            let j_low = (equality_address_base + coefficient_pair) & (e_first.len() - 1);
            let e_in = e_first[j_low];
            inner_virt[0] += e_in * (w0 * (w0 + E::one()));
            if !SKIP_LINEAR {
                inner_virt[1] += e_in * (dw * (w0 + w0 + E::one()));
            }
            inner_virt[2] += e_in * (dw * dw);

            let p0 = next_alpha_factor[left] * lane_weight;
            let p1 = next_alpha_factor[left + 1] * lane_weight;
            let trace_index = lane * next_coeff_count + left;
            let (t0, t1) = prover
                .evaluation_trace
                .pair_from_flat_index(trace_index, next_coeff_count);
            accumulate_relation_coeffs(&mut rel, w0, dw, p0 + t0, p1 + t1);
        }

        let e_out = e_second[j_high];
        virt[0] += e_out * inner_virt[0];
        if !SKIP_LINEAR {
            virt[1] += e_out * inner_virt[1];
        }
        virt[2] += e_out * inner_virt[2];
        blk = blk_end;
    }

    (virt, rel)
}

fn add_round_terms<E: FieldCore>(left: &mut ([E; 3], [E; 3]), right: ([E; 3], [E; 3])) {
    for (left_term, right_term) in left.0.iter_mut().zip(right.0) {
        *left_term += right_term;
    }
    for (left_term, right_term) in left.1.iter_mut().zip(right.1) {
        *left_term += right_term;
    }
}

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps> RelationRangeImageProver<E> {
    #[tracing::instrument(
        skip_all,
        name = "RelationRangeImageProver::fuse_folded_coefficients_and_compute_next_round"
    )]
    pub(super) fn fuse_folded_coefficients_and_compute_next_round(
        &self,
        folded_witness: &[E],
        next_alpha_factor: &[E],
        challenge: E,
    ) -> (Vec<E>, NormRoundTerms<E>, [E; 3]) {
        debug_assert!(self.in_coefficient_round());
        debug_assert!(self.current_coefficient_width() >= 2);
        let old_coeff_count = self.common_alpha_factor.len();
        let next_coeff_count = old_coeff_count / 2;
        debug_assert_eq!(next_alpha_factor.len(), next_coeff_count);
        debug_assert_eq!(folded_witness.len(), self.live_lane_count * old_coeff_count);

        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let first_bits = e_first.len().trailing_zeros() as usize;
        let next_coefficient_half = next_coeff_count / 2;
        let block_size = e_first.len().min(next_coefficient_half);
        let mut output = vec![E::zero(); self.live_lane_count * next_coeff_count];
        let skip_linear = self.can_skip_norm_linear_coeff();

        #[cfg(feature = "parallel")]
        let totals = output
            .par_chunks_mut(next_coeff_count)
            .enumerate()
            .map(|(lane, target)| {
                let source_start = lane * old_coeff_count;
                let source = &folded_witness[source_start..source_start + old_coeff_count];
                let lane_weight = self.relation_lane_weights[lane];
                if skip_linear {
                    fold_lane_and_compute_next_round::<E, true>(
                        self,
                        source,
                        target,
                        next_alpha_factor,
                        lane,
                        lane_weight,
                        challenge,
                        e_first,
                        e_second,
                        first_bits,
                        block_size,
                    )
                } else {
                    fold_lane_and_compute_next_round::<E, false>(
                        self,
                        source,
                        target,
                        next_alpha_factor,
                        lane,
                        lane_weight,
                        challenge,
                        e_first,
                        e_second,
                        first_bits,
                        block_size,
                    )
                }
            })
            .reduce(
                || ([E::zero(); 3], [E::zero(); 3]),
                |mut left, right| {
                    add_round_terms(&mut left, right);
                    left
                },
            );

        #[cfg(not(feature = "parallel"))]
        let totals = {
            let mut totals = ([E::zero(); 3], [E::zero(); 3]);
            for (lane, target) in output.chunks_mut(next_coeff_count).enumerate() {
                let source_start = lane * old_coeff_count;
                let source = &folded_witness[source_start..source_start + old_coeff_count];
                let lane_weight = self.relation_lane_weights[lane];
                let round_terms = if skip_linear {
                    fold_lane_and_compute_next_round::<E, true>(
                        self,
                        source,
                        target,
                        next_alpha_factor,
                        lane,
                        lane_weight,
                        challenge,
                        e_first,
                        e_second,
                        first_bits,
                        block_size,
                    )
                } else {
                    fold_lane_and_compute_next_round::<E, false>(
                        self,
                        source,
                        target,
                        next_alpha_factor,
                        lane,
                        lane_weight,
                        challenge,
                        e_first,
                        e_second,
                        first_bits,
                        block_size,
                    )
                };
                add_round_terms(&mut totals, round_terms);
            }
            totals
        };

        let virt_terms = if skip_linear {
            NormRoundTerms::SkipLinear([totals.0[0], totals.0[2]])
        } else {
            NormRoundTerms::Full(totals.0)
        };
        (output, virt_terms, totals.1)
    }
}
