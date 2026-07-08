use akita_algebra::eq_poly::EqPolynomial;
use akita_field::{AkitaError, FieldCore};

use crate::layout::{LevelParams, RelationMatrixRowLayout};

/// Minimal setup-contribution data needed to derive `bar_omega`.
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
    pub eq_tau1: Vec<E>,
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
        num_polys_per_group: &[usize],
        relation_matrix_row_layout: RelationMatrixRowLayout,
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
        let rows = lp.relation_matrix_row_count_for(num_groups, relation_matrix_row_layout)?;
        Ok(Self {
            relation_matrix_row_layout,
            rows,
            n_a: lp.a_key.row_len(),
            n_b: lp.b_key.row_len(),
            n_d: lp.d_key.row_len(),
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
            eq_tau1: Vec::new(),
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
        self.eq_tau1 = EqPolynomial::evals(tau1)?;
        if self.eq_tau1.len() < min_rows {
            return Err(AkitaError::InvalidSize {
                expected: min_rows,
                actual: self.eq_tau1.len(),
            });
        }
        Ok(self)
    }
}
