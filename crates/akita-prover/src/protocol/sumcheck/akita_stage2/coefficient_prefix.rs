use super::*;

impl<E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold> AkitaStage2Prover<E> {
    pub(super) fn fold_witness_embedded_coefficient_full(
        w_full: &[E],
        live_segments: usize,
        coeff_len: usize,
        r: E,
    ) -> Vec<E> {
        debug_assert!(coeff_len.is_power_of_two());
        debug_assert!(coeff_len >= 2);
        let next_coeff_len = coeff_len >> 1;
        let mut out = vec![E::zero(); live_segments * next_coeff_len];

        #[cfg(feature = "parallel")]
        out.par_chunks_mut(next_coeff_len)
            .enumerate()
            .for_each(|(x, column_out)| {
                let column_start = x * coeff_len;
                let column = &w_full[column_start..column_start + coeff_len];
                for (pair_coeff, dst) in column_out.iter_mut().enumerate() {
                    let left = 2 * pair_coeff;
                    let w0 = column[left];
                    let w1 = column[left + 1];
                    *dst = w0 + r * (w1 - w0);
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (x, column_out) in out.chunks_mut(next_coeff_len).enumerate() {
            let column_start = x * coeff_len;
            let column = &w_full[column_start..column_start + coeff_len];
            for (pair_coeff, dst) in column_out.iter_mut().enumerate() {
                let left = 2 * pair_coeff;
                let w0 = column[left];
                let w1 = column[left + 1];
                *dst = w0 + r * (w1 - w0);
            }
        }

        out
    }
}
