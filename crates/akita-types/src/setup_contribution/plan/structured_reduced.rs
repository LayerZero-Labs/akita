use super::*;
use akita_algebra::ring::terminal_residue_kernel;
use akita_challenges::{Challenges, SparseChallenge};

struct ValidatedSparseChallenge<'a> {
    challenge: &'a SparseChallenge,
}

impl<'a> ValidatedSparseChallenge<'a> {
    fn new(challenge: &'a SparseChallenge, dimension: usize) -> Result<Self, AkitaError> {
        challenge.validate_dyn(dimension)?;
        Ok(Self { challenge })
    }

    fn evaluate<E: Field>(&self, functional: &[E]) -> Result<E, AkitaError> {
        self.challenge
            .positions
            .iter()
            .zip(&self.challenge.coeffs)
            .try_fold(E::zero(), |sum, (&position, &coefficient)| {
                let weight = *functional
                    .get(position as usize)
                    .ok_or(AkitaError::InvalidProof)?;
                Ok(sum + weight * E::from_i64(i64::from(coefficient)))
            })
    }
}

fn embedded_terminal_functionals<E: Field>(
    native_equality: &[E],
    ambient_dimension: usize,
    subcolumns: usize,
    alpha: E,
) -> Result<Vec<Vec<E>>, AkitaError> {
    (0..subcolumns)
        .map(|subcolumn| {
            let start = subcolumn
                .checked_mul(native_equality.len())
                .ok_or(AkitaError::InvalidProof)?;
            let end = start
                .checked_add(native_equality.len())
                .filter(|&end| end <= ambient_dimension)
                .ok_or(AkitaError::InvalidProof)?;
            let mut embedded = vec![E::zero(); ambient_dimension];
            embedded[start..end].copy_from_slice(native_equality);
            terminal_residue_kernel(&embedded, alpha)
        })
        .collect()
}

impl<E: Field> SetupContributionPlan<E> {
    /// Contract one reduced-evaluation group's structured E/T/Z terms with
    /// their genuine public ring multipliers.
    pub fn evaluate_reduced_structured_group<F>(
        &self,
        group_id: usize,
        challenges: &Challenges,
        opening_multiplier: &crate::PreparedRingMultiplier<E>,
    ) -> Result<E, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let group_index = self
            .groups
            .iter()
            .position(|group| group.group_id == group_id)
            .ok_or(AkitaError::InvalidProof)?;
        let group = &self.groups[group_index];
        let (alpha, weights) = match &self.direct_scan_state {
            DirectScanState::Reduced { alpha, groups, .. } => (
                *alpha,
                groups.get(group_index).ok_or_else(|| {
                    AkitaError::InvalidSetup("reduced direct-scan group is missing".into())
                })?,
            ),
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "reduced structured contraction requires prepared reduced state".into(),
                ));
            }
        };
        if !matches!(group.opening_method, crate::OpeningMethod::EvaluationTrace) {
            return Err(AkitaError::InvalidSetup(
                "reduced structured contraction disagrees with its prepared mode".into(),
            ));
        }
        let block_claims = group
            .num_claims
            .checked_mul(group.num_live_blocks)
            .ok_or_else(|| AkitaError::InvalidSetup("structured block count overflow".into()))?;
        if challenges.len() != block_claims
            || challenges.num_claims() != group.num_claims
            || challenges.num_live_blocks_per_claim() != group.num_live_blocks
        {
            return Err(AkitaError::InvalidProof);
        }
        let [a_role, b_role, d_role] = &weights.roles;
        let weights = &weights.weights;
        let (a_functional, a_equality) = (&a_role.functional, &a_role.equality);
        let (b_functional, b_equality) = (&b_role.functional, &b_role.equality);
        let (d_functional, d_equality) = (&d_role.functional, &d_role.equality);
        let d_a = group.role_dims.d_a();
        if a_functional.len() != d_a
            || a_equality.len() != d_a
            || b_functional.len() != group.role_dims.d_b()
            || b_equality.len() != group.role_dims.d_b()
            || d_functional.len() != group.role_dims.d_d()
            || d_equality.len() != group.role_dims.d_d()
        {
            return Err(AkitaError::InvalidSetup(
                "reduced structured coefficient state has malformed dimensions".into(),
            ));
        }

        let opening_gadget = extension_gadget::<F, E>(group.depth_open, group.log_basis_open);
        let commitment_gadget = extension_gadget::<F, E>(group.depth_commit, group.log_basis_outer);
        let witness_gadget = extension_gadget::<F, E>(group.depth_witness, group.log_basis_inner);
        let (outer_subcolumns, _) =
            SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims)?;
        let opening_subcolumns = group.opening_subcolumns;
        let opening_functionals =
            embedded_terminal_functionals(d_equality, d_a, opening_subcolumns, alpha)?;
        let commitment_functionals =
            embedded_terminal_functionals(b_equality, d_a, outer_subcolumns, alpha)?;
        let e_stride = checked::product([opening_subcolumns, group.depth_open])
            .ok_or_else(|| AkitaError::InvalidSetup("structured E stride overflow".into()))?;
        let t_row_stride = checked::product([outer_subcolumns, group.depth_commit])
            .ok_or_else(|| AkitaError::InvalidSetup("structured T row stride overflow".into()))?;
        let t_stride = group
            .n_a
            .checked_mul(t_row_stride)
            .ok_or_else(|| AkitaError::InvalidSetup("structured T stride overflow".into()))?;
        let expected_e = block_claims
            .checked_mul(e_stride)
            .ok_or(AkitaError::InvalidProof)?;
        let expected_t = block_claims
            .checked_mul(t_stride)
            .ok_or(AkitaError::InvalidProof)?;
        let expected_z = group
            .num_positions_per_block
            .checked_mul(group.depth_witness)
            .ok_or(AkitaError::InvalidProof)?;
        if weights.e.len() != expected_e
            || weights.t.len() != expected_t
            || weights.z.len() != expected_z
            || group.a_row_weights.len() != group.n_a
        {
            return Err(AkitaError::InvalidProof);
        }

        let mut commitment_multipliers = vec![E::zero(); commitment_functionals.len()];
        let mut et = E::zero();
        for (block_claim, challenge) in challenges.as_slice().iter().enumerate() {
            let challenge = ValidatedSparseChallenge::new(challenge, d_a)?;
            let e_start = block_claim
                .checked_mul(e_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let e_weights = checked_slice(&weights.e, e_start, e_stride, "reduced structured E")?;
            let mut e = E::zero();
            for (subcolumn, functional) in opening_functionals.iter().enumerate() {
                let multiplier = challenge.evaluate(functional)?;
                let digit_start = subcolumn
                    .checked_mul(group.depth_open)
                    .ok_or(AkitaError::InvalidProof)?;
                let digit_weights = checked_slice(
                    e_weights,
                    digit_start,
                    group.depth_open,
                    "reduced structured E digits",
                )?;
                e += multiplier
                    * digit_weights
                        .iter()
                        .zip(&opening_gadget)
                        .fold(E::zero(), |sum, (&eq, &gadget)| sum + eq * gadget);
            }

            let t_start = block_claim
                .checked_mul(t_stride)
                .ok_or(AkitaError::InvalidProof)?;
            let t_weights = checked_slice(&weights.t, t_start, t_stride, "reduced structured T")?;
            for (multiplier, functional) in commitment_multipliers
                .iter_mut()
                .zip(&commitment_functionals)
            {
                *multiplier = challenge.evaluate(functional)?;
            }
            let mut t = E::zero();
            for (row, &row_weight) in t_weights
                .chunks_exact(t_row_stride)
                .zip(group.a_row_weights.iter())
            {
                let mut row_evaluation = E::zero();
                for (subcolumn, &multiplier) in commitment_multipliers.iter().enumerate() {
                    let digit_start = subcolumn
                        .checked_mul(group.depth_commit)
                        .ok_or(AkitaError::InvalidProof)?;
                    let digit_weights = checked_slice(
                        row,
                        digit_start,
                        group.depth_commit,
                        "reduced structured T digits",
                    )?;
                    row_evaluation += multiplier
                        * digit_weights
                            .iter()
                            .zip(&commitment_gadget)
                            .fold(E::zero(), |sum, (&eq, &gadget)| sum + eq * gadget);
                }
                t += row_weight * row_evaluation;
            }
            et += group.consistency_weight * e + t;
        }

        let mut z = E::zero();
        for position in 0..group.num_positions_per_block {
            let start = position
                .checked_mul(group.depth_witness)
                .ok_or(AkitaError::InvalidProof)?;
            let eq = checked_slice(
                &weights.z,
                start,
                group.depth_witness,
                "reduced structured Z",
            )?;
            let digit_evaluation = eq
                .iter()
                .zip(&witness_gadget)
                .fold(E::zero(), |sum, (&eq, &gadget)| sum + eq * gadget);
            z += opening_multiplier.evaluate_position_functional(position, a_functional)?
                * digit_evaluation;
        }
        Ok(et + group.consistency_weight * z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{One, Prime128Offset275 as F, Ring, Zero};

    #[test]
    fn ambient_sparse_multiplier_is_not_scalarized_at_alpha() {
        let challenge = SparseChallenge {
            positions: vec![3].into(),
            coeffs: vec![1].into(),
        };
        let native_equality = [F::zero(), F::one(), F::zero(), F::zero()];
        let alpha = F::from_u64(2);

        let functional = embedded_terminal_functionals(&native_equality, 4, 1, alpha).unwrap();
        let reduced = ValidatedSparseChallenge::new(&challenge, functional[0].len())
            .unwrap()
            .evaluate(&functional[0])
            .unwrap();

        // In F[X]/(X^4 + 1), X^3 * X = -1. Evaluating the two
        // polynomials independently first would incorrectly produce 16.
        assert_eq!(reduced, -F::one());
        assert_ne!(reduced, alpha * alpha * alpha * alpha);
    }
}
