use super::*;
use jolt_poly::OmittedConstantPoly;

impl<E: Field + Ring + Unreduced> LowBasisRangeCheckProver<E> {
    #[inline]
    pub(super) fn use_sparse_x_y_round(&self) -> bool {
        !self.in_x_phase() && self.live_x_cols < (1usize << self.col_bits)
    }

    #[tracing::instrument(
        skip_all,
        name = "LowBasisRangeCheckProver::compute_round_compact_sparse_x_y"
    )]
    pub(super) fn compute_round_compact_sparse_x_y<S: CompactRangeImageSource + ?Sized>(
        &self,
        compact_range_image: &S,
    ) -> OmittedConstantPoly<E> {
        debug_assert!(self.use_sparse_x_y_round());
        let y_len = compact_range_image.len() / self.live_x_cols;
        let y_pairs = y_len / 2;
        compute_range_round_polynomial_from_compact_image_pairs(
            &self.split_eq,
            &self.polynomial_precomputation,
            |j| {
                let x = j / y_pairs;
                if x >= self.live_x_cols {
                    return (0, 0);
                }
                let y_pair = j % y_pairs;
                let top = x * y_len + 2 * y_pair;
                (
                    compact_range_image.range_image_value(top),
                    compact_range_image.range_image_value(top + 1),
                )
            },
        )
    }
}
