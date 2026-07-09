//! Relation-matrix row layout with EvaluationTrace as the last row.
//!
//! Quotient-bearing rows keep today's physical indices
//! (`FoldEvaluation | FoldConsistency | OuterConsistency | OpeningConsistency`).
//! One field-level EvaluationTrace row is appended after them and weighted by
//! `eq(row_index, evaluation_trace_row)`.

use super::{LevelParams, RelationMatrixRowLayout};
use crate::proof::OpeningClaimsLayout;
use akita_field::AkitaError;

/// Logical relation-matrix rows: quotient-bearing rows plus trailing EvaluationTrace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelationRowLayout {
    /// Today's `relation_matrix_row_count_for` (no EvaluationTrace).
    pub quotient_rows: usize,
}

impl RelationRowLayout {
    /// Build from level params and opening batch (scalar or multi-group root).
    pub fn for_level(
        lp: &LevelParams,
        m_row_layout: RelationMatrixRowLayout,
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<Self, AkitaError> {
        opening_batch.check()?;
        if lp.has_precommitted_groups() {
            lp.reject_multi_group_multi_chunk("RelationRowLayout::for_level")?;
            if opening_batch.num_groups() != lp.root_group_count() {
                return Err(AkitaError::InvalidSetup(
                    "multi-group RelationRowLayout requires opening_batch.num_groups() == root_group_count()"
                        .to_string(),
                ));
            }
        } else {
            lp.require_scalar_level("RelationRowLayout::for_level")?;
        }
        let quotient_rows =
            lp.relation_matrix_row_count_for(opening_batch.num_groups(), m_row_layout)?;
        Ok(Self { quotient_rows })
    }

    /// Logical index of the shared EvaluationTrace row (last row).
    #[inline]
    #[must_use]
    pub fn evaluation_trace_row(self) -> usize {
        self.quotient_rows
    }

    /// Total logical rows including EvaluationTrace.
    #[inline]
    #[must_use]
    pub fn total_row_count(self) -> usize {
        self.quotient_rows.saturating_add(1)
    }

    /// Boolean variables needed to index the padded row space
    /// (`next_power_of_two(total_row_count).trailing_zeros()`).
    pub fn row_index_num_vars(self) -> Result<usize, AkitaError> {
        let padded = self
            .total_row_count()
            .checked_next_power_of_two()
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation-row index width overflow".to_string())
            })?;
        Ok(padded.trailing_zeros() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{AjtaiKeyParams, LevelParams, PrecommittedLevelParams, SisModulusFamily};
    use crate::schedule::PrecommittedGroupParams;
    use crate::PolynomialGroupLayout;
    use akita_challenges::SparseChallengeConfig;

    fn sample_params_only() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            64,
            3,
            2,
            4,
            3,
            SparseChallengeConfig::pm1_only(3),
        )
    }

    fn sample_layout_lp() -> LevelParams {
        sample_params_only().with_decomp(4, 2, 2, 2, 0).unwrap()
    }

    #[test]
    fn scalar_evaluation_trace_is_last() {
        let lp = sample_params_only()
            .with_layout(&sample_layout_lp(), 128)
            .unwrap();
        let batch = OpeningClaimsLayout::new(4, 1).expect("batch");
        let layout = RelationRowLayout::for_level(&lp, RelationMatrixRowLayout::WithDBlock, &batch)
            .expect("layout");
        let quotient = lp
            .relation_matrix_row_count_for(1, RelationMatrixRowLayout::WithDBlock)
            .unwrap();
        assert_eq!(layout.quotient_rows, quotient);
        assert_eq!(layout.evaluation_trace_row(), quotient);
        assert_eq!(layout.total_row_count(), quotient + 1);
        assert_eq!(
            layout.row_index_num_vars().unwrap(),
            (quotient + 1).next_power_of_two().trailing_zeros() as usize
        );
    }

    #[test]
    fn multi_group_matches_legacy_quotient_count() {
        let lp = sample_params_only()
            .with_layout(&sample_layout_lp(), 128)
            .unwrap();
        let precommit_lp = sample_params_only()
            .with_layout(&sample_layout_lp(), 128)
            .unwrap();
        let precommit = PrecommittedLevelParams {
            layout: PrecommittedGroupParams::from_params(
                PolynomialGroupLayout::new(4, 1),
                &precommit_lp,
            ),
            a_key: precommit_lp.a_key.clone(),
            b_key: AjtaiKeyParams::new_unchecked(
                precommit_lp.b_key.min_security_bits(),
                precommit_lp.b_key.sis_family(),
                5,
                precommit_lp.b_key.col_len(),
                precommit_lp.b_key.coeff_linf_bound(),
                precommit_lp.ring_dimension,
            ),
            num_blocks: precommit_lp.num_blocks,
            block_len: precommit_lp.block_len,
            num_digits_commit: precommit_lp.num_digits_commit,
            num_digits_open: precommit_lp.num_digits_open,
            num_digits_fold_one: precommit_lp.num_digits_fold_one,
        };
        let mut grouped = lp;
        grouped.precommitted_groups = vec![precommit];
        let batch = OpeningClaimsLayout::from_group_sizes(4, &[1, 1]).expect("batch");
        let layout =
            RelationRowLayout::for_level(&grouped, RelationMatrixRowLayout::WithDBlock, &batch)
                .expect("layout");
        let quotient = grouped
            .relation_matrix_row_count_for(2, RelationMatrixRowLayout::WithDBlock)
            .unwrap();
        assert_eq!(layout.quotient_rows, quotient);
        assert_eq!(layout.evaluation_trace_row(), quotient);
    }
}
