use std::sync::Arc;

use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{AkitaError, FieldCore};

use crate::layout::relation::RelationLayout;
use crate::layout::{LevelParams, RelationMatrixRowLayout};

/// Minimal setup-contribution data needed to derive setup-index weights.
#[derive(Clone)]
pub struct SetupContributionPlanInputs<E: FieldCore> {
    pub relation_matrix_row_layout: RelationMatrixRowLayout,
    pub rows: usize,
    pub n_a: usize,
    pub n_b: usize,
    pub n_d: usize,
    pub num_groups: usize,
    pub num_polys_per_group: Vec<usize>,
    pub num_t_vectors: usize,
    pub num_claims: usize,
    pub num_blocks: usize,
    pub block_len: usize,
    pub depth_open: usize,
    pub depth_commit: usize,
    pub depth_fold: usize,
    pub inner_width: usize,
    /// Expanded τ₁ eq table, shared by reference so plan inputs can be built
    /// (and cloned) without copying the full `2^|τ₁|` expansion.
    pub eq_tau1: Arc<[E]>,
}

impl<E: FieldCore> SetupContributionPlanInputs<E> {
    /// Build challenge-free setup-contribution inputs from per-level params.
    ///
    /// Mirrors the prover's `create_setup_contribution_inputs` field derivation
    /// without materializing `eq_tau1`.
    ///
    /// # Errors
    ///
    /// Returns an error when level layout parameters are inconsistent.
    pub fn from_level_params(
        lp: &LevelParams,
        relation_layout: &RelationLayout,
        num_polys_per_group: &[usize],
        depth_fold: usize,
    ) -> Result<Self, AkitaError> {
        let num_polynomials: usize = num_polys_per_group.iter().copied().sum();
        let num_groups = num_polys_per_group.len().max(1);
        let depth_commit = lp.num_digits_commit;
        let depth_open = lp.num_digits_open;
        if lp.num_blocks == 0 || !lp.num_blocks.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "num_blocks must be a non-zero power of two".into(),
            ));
        }
        if lp.block_len == 0 || depth_commit == 0 || depth_open == 0 || depth_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator layout has zero width".into(),
            ));
        }
        let inner_width = lp
            .block_len
            .checked_mul(depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".into()))?;
        if lp.a_key.col_len() < inner_width {
            return Err(AkitaError::InvalidSetup(
                "A-key column width is too small for setup contribution layout".into(),
            ));
        }
        let expected_b_width = num_polynomials
            .checked_mul(lp.a_key.row_len())
            .and_then(|width| width.checked_mul(depth_open))
            .and_then(|width| width.checked_mul(lp.num_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".into()))?;
        if lp.b_key.col_len() < expected_b_width {
            return Err(AkitaError::InvalidSetup(
                "B-key column width is too small for setup contribution layout".into(),
            ));
        }
        let row_plan = relation_layout.row_plan();
        let rows = row_plan.trace_row();
        let relation_matrix_row_layout = if row_plan
            .families()
            .iter()
            .any(|family| matches!(family.id(), crate::RelationRowId::D))
        {
            RelationMatrixRowLayout::WithDBlock
        } else {
            RelationMatrixRowLayout::WithoutDBlock
        };
        let a_provider = relation_layout.family_provider(crate::RelationRowId::A {
            group: crate::RelationGroupId::Current,
        })?;
        let b_provider = relation_layout.family_provider(crate::RelationRowId::B {
            group: crate::RelationGroupId::Current,
        })?;
        let d_provider = if row_plan
            .families()
            .iter()
            .any(|family| matches!(family.id(), crate::RelationRowId::D))
        {
            Some(relation_layout.family_provider(crate::RelationRowId::D)?)
        } else {
            None
        };
        let a_family = a_provider.family();
        let b_family = b_provider.family();
        let d_family = d_provider.as_ref().map(|provider| provider.family());
        // Constructing these providers resolves and validates the quotient
        // edges through the same authority used by heterogeneous compression
        // families, before setup geometry is prepared.
        let n_a = a_family.rows().len();
        let n_b = b_family.rows().len();
        let n_d = d_family.map_or(0, |family| family.rows().len());
        if n_a != lp.a_key.row_len()
            || n_b != lp.b_key.row_len()
            || (n_d != 0 && n_d != lp.d_key.row_len())
            || a_family.native_ring_dim() != lp.a_key.sis_table_key().ring_dimension as usize
            || b_family.native_ring_dim() != lp.b_key.sis_table_key().ring_dimension as usize
            || d_family.is_some_and(|family| {
                family.native_ring_dim() != lp.d_key.sis_table_key().ring_dimension as usize
            })
        {
            return Err(AkitaError::InvalidSetup(
                "relation plan matrix rows disagree with level parameters".into(),
            ));
        }
        Ok(Self {
            relation_matrix_row_layout,
            rows,
            n_a,
            n_b,
            n_d,
            num_groups,
            num_polys_per_group: num_polys_per_group.to_vec(),
            num_t_vectors: num_polynomials,
            num_claims: num_polynomials,
            num_blocks: lp.num_blocks,
            block_len: lp.block_len,
            depth_open,
            depth_commit,
            depth_fold,
            inner_width,
            eq_tau1: Vec::new().into(),
        })
    }

    /// Attach the τ₁ eq-polynomial expansion after [`Self::from_level_params`].
    ///
    /// # Errors
    ///
    /// Returns an error when `tau1` cannot be expanded or is shorter than `min_rows`.
    pub fn with_eq_tau1_from_tau(
        mut self,
        tau1: &[E],
        min_rows: usize,
    ) -> Result<Self, AkitaError> {
        self.eq_tau1 = EqPolynomial::evals(tau1)?.into();
        if self.eq_tau1.len() < min_rows {
            return Err(AkitaError::InvalidSize {
                expected: min_rows,
                actual: self.eq_tau1.len(),
            });
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{CanonicalField, Prime128OffsetA7F7 as F};

    #[test]
    fn from_level_params_rejects_non_pow2_num_blocks() {
        let valid = LevelParams::params_only(
            crate::SisModulusFamily::Q128,
            64,
            3,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::pm1_only(64),
        )
        .with_decomp(1, 1, 2, 3, 0)
        .unwrap();
        let opening = crate::OpeningClaimsLayout::new(1, 2).unwrap();
        let relation_layout = crate::RelationLayout::from_authenticated_statement(
            &valid,
            &opening,
            RelationMatrixRowLayout::WithoutDBlock,
            F::modulus_bits(),
        )
        .unwrap();
        let mut lp = valid;
        lp.num_blocks = 3;
        lp.block_len = 8;
        lp.num_digits_commit = 2;
        lp.num_digits_open = 3;
        assert!(SetupContributionPlanInputs::<F>::from_level_params(
            &lp,
            &relation_layout,
            &[2],
            2,
        )
        .is_err());
    }
}
