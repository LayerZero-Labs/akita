//! Deferred R1CS relations used by Akita ZK plain-opening checks.

use akita_field::{AkitaError, ExtField, FieldCore};

/// A witness cursor referenced by a deferred R1CS row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkR1csVariable {
    /// Index into the plain-opened hiding witness.
    HiddenWitness(usize),
    /// Index into the verifier-local auxiliary witness.
    AuxiliaryWitness(usize),
}

/// One linear term inside an R1CS linear combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZkR1csTerm<E: FieldCore> {
    /// Variable referenced by this term.
    pub variable: ZkR1csVariable,
    /// Public coefficient multiplying the variable.
    pub coeff: E,
}

/// A linear combination used by a deferred R1CS row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZkR1csLinearCombination<E: FieldCore> {
    /// Public constant term.
    pub constant: E,
    /// Variable terms.
    pub terms: Vec<ZkR1csTerm<E>>,
}

impl<E: FieldCore> ZkR1csLinearCombination<E> {
    /// Construct a constant linear combination.
    pub fn constant(constant: E) -> Self {
        Self {
            constant,
            terms: Vec::new(),
        }
    }

    /// Construct the zero linear combination.
    pub fn zero() -> Self {
        Self::constant(E::zero())
    }

    /// Construct the one linear combination.
    pub fn one() -> Self {
        Self::constant(E::one())
    }

    /// Construct a single-variable linear combination.
    pub fn variable(variable: ZkR1csVariable, coeff: E) -> Self {
        Self {
            constant: E::zero(),
            terms: vec![ZkR1csTerm { variable, coeff }],
        }
    }

    /// Add `scale * source` into this linear combination.
    pub fn add_scaled(&mut self, scale: E, source: &Self) {
        self.constant += scale * source.constant;
        self.terms
            .extend(source.terms.iter().cloned().map(|term| ZkR1csTerm {
                variable: term.variable,
                coeff: scale * term.coeff,
            }));
    }

    fn evaluate(&self, witness: &ZkR1csWitness<'_, E>) -> Option<E> {
        let mut value = self.constant;
        for term in &self.terms {
            let term_value = witness.get(term.variable)?;
            value += term.coeff * term_value;
        }
        Some(value)
    }
}

/// A deferred relation over linear combinations in the R1CS witness.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ZkR1csRelation<E: FieldCore> {
    /// Ordinary R1CS row enforcing `<A, X> * <B, X> = <C, X>`.
    R1cs {
        /// Human-readable relation category used only for diagnostics.
        description: &'static str,
        /// Left linear combination.
        a: ZkR1csLinearCombination<E>,
        /// Right linear combination.
        b: ZkR1csLinearCombination<E>,
        /// Output linear combination.
        c: ZkR1csLinearCombination<E>,
    },
    /// Auxiliary-generation row assigning `auxiliary = <A, X> * <B, X>`.
    GenerateAuxiliary {
        /// Human-readable relation category used only for diagnostics.
        description: &'static str,
        /// Left linear combination.
        a: ZkR1csLinearCombination<E>,
        /// Right linear combination.
        b: ZkR1csLinearCombination<E>,
        /// Auxiliary witness variable assigned by this row.
        auxiliary: ZkR1csVariable,
    },
}

struct ZkR1csWitness<'a, E: FieldCore> {
    hiding_witness: &'a [E],
    aux_witness: Vec<Option<E>>,
}

impl<'a, E: FieldCore> ZkR1csWitness<'a, E> {
    fn new(hiding_witness: &'a [E], aux_count: usize) -> Self {
        Self {
            hiding_witness,
            aux_witness: vec![None; aux_count],
        }
    }

    fn get(&self, variable: ZkR1csVariable) -> Option<E> {
        match variable {
            ZkR1csVariable::HiddenWitness(cursor) => self.hiding_witness.get(cursor).copied(),
            ZkR1csVariable::AuxiliaryWitness(cursor) => {
                self.aux_witness.get(cursor).copied().flatten()
            }
        }
    }

    fn set_aux(&mut self, variable: ZkR1csVariable, value: E) -> Option<()> {
        let ZkR1csVariable::AuxiliaryWitness(cursor) = variable else {
            return None;
        };
        *self.aux_witness.get_mut(cursor)? = Some(value);
        Some(())
    }
}

fn add_scaled_lc<E: FieldCore>(
    target: &mut ZkR1csLinearCombination<E>,
    scale: E,
    source: &ZkR1csLinearCombination<E>,
) {
    target.add_scaled(scale, source);
}

/// Consume one hiding-witness slot as a linear combination over `E`.
pub fn zk_base_mask_lc<E: FieldCore>(hiding_cursor: &mut usize) -> ZkR1csLinearCombination<E> {
    let variable = ZkR1csVariable::HiddenWitness(*hiding_cursor);
    *hiding_cursor += 1;
    ZkR1csLinearCombination::variable(variable, E::one())
}

/// Consume `E::EXT_DEGREE` hiding-witness slots as one extension-field mask.
pub fn zk_ext_mask_lc<F, E>(hiding_cursor: &mut usize) -> ZkR1csLinearCombination<E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let mask = zk_ext_mask_lc_at::<F, E>(*hiding_cursor);
    *hiding_cursor += <E as ExtField<F>>::EXT_DEGREE;
    mask
}

/// Build an extension-field mask from hiding-witness slots starting at `start`.
pub fn zk_ext_mask_lc_at<F, E>(start: usize) -> ZkR1csLinearCombination<E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let mut out = ZkR1csLinearCombination::zero();
    for idx in 0..<E as ExtField<F>>::EXT_DEGREE {
        let mut coeffs = vec![F::zero(); <E as ExtField<F>>::EXT_DEGREE];
        coeffs[idx] = F::one();
        out.add_scaled(
            E::from_base_slice(&coeffs),
            &ZkR1csLinearCombination::variable(
                ZkR1csVariable::HiddenWitness(start + idx),
                E::one(),
            ),
        );
    }
    out
}

/// Consume `count` base-field hiding-witness slots as linear combinations.
pub fn zk_base_mask_lcs<E: FieldCore>(
    count: usize,
    hiding_cursor: &mut usize,
) -> Vec<ZkR1csLinearCombination<E>> {
    (0..count)
        .map(|_| zk_base_mask_lc::<E>(hiding_cursor))
        .collect()
}

/// Lift a base-field hiding witness into the verifier's relation field.
pub fn lift_hiding_witness<F, E>(hiding_witness: &[F]) -> Vec<E>
where
    F: FieldCore,
    E: ExtField<F>,
{
    hiding_witness.iter().copied().map(E::lift_base).collect()
}

/// Transpose extension-field column masks into row-basis masks.
///
/// If `column_masks[v]` masks an extension scalar decomposed in the fixed
/// `F`-basis of `E`, the returned row `u` is
/// `sum_v coeff_{u,v} * column_masks[v]`.
///
/// # Errors
///
/// Returns an error if `[E:F]` is zero, not a power of two, or the column count
/// does not match `[E:F]`.
pub fn zk_row_masks_from_column_masks<F, E>(
    column_masks: &[ZkR1csLinearCombination<E>],
) -> Result<Vec<ZkR1csLinearCombination<E>>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let width = <E as ExtField<F>>::EXT_DEGREE;
    if width == 0 || !width.is_power_of_two() {
        return Err(AkitaError::InvalidInput(format!(
            "extension-opening tensor reduction requires power-of-two extension degree, got {width}"
        )));
    }
    if column_masks.len() != width {
        return Err(AkitaError::InvalidSize {
            expected: width,
            actual: column_masks.len(),
        });
    }

    let mut row_masks = vec![ZkR1csLinearCombination::zero(); width];
    for (col_idx, col_mask) in column_masks.iter().enumerate() {
        let mut basis = vec![E::zero(); width];
        basis[col_idx] = E::one();
        let row_coeffs = transpose_extension_columns::<F, E>(&basis)?;
        for (row_mask, coeff) in row_masks.iter_mut().zip(row_coeffs) {
            row_mask.add_scaled(coeff, col_mask);
        }
    }
    Ok(row_masks)
}

fn transpose_extension_columns<F, E>(column_partials: &[E]) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let width = <E as ExtField<F>>::EXT_DEGREE;
    if column_partials.len() != width {
        return Err(AkitaError::InvalidSize {
            expected: width,
            actual: column_partials.len(),
        });
    }

    let mut rows = vec![vec![F::zero(); width]; width];
    for (column, partial) in column_partials.iter().enumerate() {
        let coords = partial.to_base_vec();
        if coords.len() != width {
            return Err(AkitaError::InvalidSize {
                expected: width,
                actual: coords.len(),
            });
        }
        for (row, coord) in coords.into_iter().enumerate() {
            rows[row][column] = coord;
        }
    }
    Ok(rows
        .into_iter()
        .map(|coords| E::from_base_slice(&coords))
        .collect())
}

/// Return a masked linear value with base masks subtracted symbolically.
///
/// This represents `masked_value - sum_i mask_coeffs[i] * masks[i]`.
///
/// # Errors
///
/// Returns an error if the mask and coefficient slices have different lengths.
pub fn zk_masked_linear_value_lc<E>(
    masked_value: E,
    masks: &[ZkR1csLinearCombination<E>],
    mask_coeffs: &[E],
) -> Result<ZkR1csLinearCombination<E>, AkitaError>
where
    E: FieldCore,
{
    if masks.len() != mask_coeffs.len() {
        return Err(AkitaError::InvalidSize {
            expected: mask_coeffs.len(),
            actual: masks.len(),
        });
    }
    let mut value = ZkR1csLinearCombination::constant(masked_value);
    for (mask, &mask_coeff) in masks.iter().zip(mask_coeffs) {
        value.add_scaled(-mask_coeff, mask);
    }
    Ok(value)
}

fn eq_evals<E: FieldCore>(point: &[E]) -> Result<Vec<E>, AkitaError> {
    let size = 1usize
        .checked_shl(point.len() as u32)
        .ok_or_else(|| AkitaError::InvalidInput("eq table length overflow".to_string()))?;
    let mut evals = vec![E::zero(); size];
    evals[0] = E::one();
    let mut len = 1usize;
    for &t in point.iter().rev() {
        let one_minus_t = E::one() - t;
        for j in (0..len).rev() {
            evals[2 * j + 1] = evals[j] * t;
            evals[2 * j] = evals[j] * one_minus_t;
        }
        len *= 2;
    }
    Ok(evals)
}

/// Combine per-coefficient `y` masks into the stage-2 relation-claim mask.
///
/// # Errors
///
/// Returns an error if the equality table or mask indexing would overflow, or
/// if `y_masks` does not contain `y_count * D` masks.
pub fn zk_relation_claim_mask_from_y_masks<E, const D: usize>(
    tau1: &[E],
    alpha: E,
    y_count: usize,
    y_masks: &[ZkR1csLinearCombination<E>],
) -> Result<ZkR1csLinearCombination<E>, AkitaError>
where
    E: FieldCore,
{
    let expected_masks = y_count.checked_mul(D).ok_or(AkitaError::InvalidProof)?;
    if y_masks.len() != expected_masks {
        return Err(AkitaError::InvalidSize {
            expected: expected_masks,
            actual: y_masks.len(),
        });
    }

    let eq_tau1 = eq_evals(tau1)?;
    let mut alpha_pows = Vec::with_capacity(D);
    let mut alpha_power = E::one();
    for _ in 0..D {
        alpha_pows.push(alpha_power);
        alpha_power *= alpha;
    }

    let mut out = ZkR1csLinearCombination::zero();
    for y_idx in 0..y_count {
        let row_coeff = eq_tau1.get(1 + y_idx).copied().unwrap_or_else(E::zero);
        for coeff_idx in 0..D {
            let coeff = row_coeff * alpha_pows[coeff_idx];
            out.add_scaled(coeff, &y_masks[y_idx * D + coeff_idx]);
        }
    }
    Ok(out)
}

/// Push a linear zero relation `residual = 0`.
pub fn zk_push_linear_zero<E: FieldCore>(
    relations: &mut ZkRelationAccumulator<E>,
    description: &'static str,
    residual: ZkR1csLinearCombination<E>,
) {
    relations.push_r1cs(
        description,
        residual,
        ZkR1csLinearCombination::one(),
        ZkR1csLinearCombination::zero(),
    );
}

/// Accumulates ZK plain-opening R1CS rows for one verifier level.
///
/// Ordinary rows have the form `<A, X> * <B, X> = <C, X>`, while
/// auxiliary-generation rows assign verifier-local auxiliary variables for
/// later rows. Future Spartan or tail-sigma integration can replace the final
/// `verify_all` pass with a proof verification over the same row inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZkRelationAccumulator<E: FieldCore> {
    relations: Vec<ZkR1csRelation<E>>,
    aux_count: usize,
}

impl<E: FieldCore> Default for ZkRelationAccumulator<E> {
    fn default() -> Self {
        Self {
            relations: Vec::new(),
            aux_count: 0,
        }
    }
}

impl<E: FieldCore> ZkRelationAccumulator<E> {
    /// Construct an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an R1CS row.
    pub fn push_r1cs(
        &mut self,
        description: &'static str,
        a: ZkR1csLinearCombination<E>,
        b: ZkR1csLinearCombination<E>,
        c: ZkR1csLinearCombination<E>,
    ) {
        self.relations.push(ZkR1csRelation::R1cs {
            description,
            a,
            b,
            c,
        });
    }

    /// Push an auxiliary-generation row and return its synthesized output.
    ///
    /// The returned linear combination is a relation-local auxiliary wire. During
    /// [`Self::verify_all`], the row assigns that wire to `<A, X> * <B, X>`.
    ///
    /// # Errors
    ///
    /// Returns an error if auxiliary cursor allocation overflows.
    pub fn new_auxilary(
        &mut self,
        description: &'static str,
        a: ZkR1csLinearCombination<E>,
        b: ZkR1csLinearCombination<E>,
    ) -> Result<ZkR1csLinearCombination<E>, AkitaError> {
        let aux_index = self.aux_count;
        self.aux_count = self
            .aux_count
            .checked_add(1)
            .ok_or(AkitaError::InvalidProof)?;
        let auxiliary = ZkR1csVariable::AuxiliaryWitness(aux_index);
        let c = ZkR1csLinearCombination::variable(auxiliary, E::one());
        self.relations.push(ZkR1csRelation::GenerateAuxiliary {
            description,
            a,
            b,
            auxiliary,
        });
        Ok(c)
    }

    /// Return the true value expression for `masked_value = true_value + mask`.
    pub fn unmask_lc(
        masked_value: E,
        mask: &ZkR1csLinearCombination<E>,
    ) -> ZkR1csLinearCombination<E> {
        let mut true_value = ZkR1csLinearCombination::constant(masked_value);
        add_scaled_lc(&mut true_value, -E::one(), mask);
        true_value
    }

    /// Return the true claim expression for a masked sumcheck claim.
    #[doc(hidden)]
    pub fn push_masked_claim_relation(
        &mut self,
        _description: &'static str,
        masked_claim: E,
        mask: &ZkR1csLinearCombination<E>,
    ) -> ZkR1csLinearCombination<E> {
        Self::unmask_lc(masked_claim, mask)
    }

    /// Record one masked standard sumcheck round relation.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn push_masked_full_round_relation<F>(
        &mut self,
        description: &'static str,
        previous_masked_claim: E,
        previous_mask: &ZkR1csLinearCombination<E>,
        public_coeffs: &[E],
        r_round: E,
        hiding_cursor: &mut usize,
    ) -> (ZkR1csLinearCombination<E>, ZkR1csLinearCombination<E>)
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        // Current round message:
        //
        //   G~_i(X) = G_i(X) + rho_i(X)
        //
        // `public_coeffs` are the transcript-visible coefficients of G~_i.
        // The corresponding rho_i coefficients are already present in
        // `hiding_witness`, and this verifier cursor points at those slots.
        let mut mask_coeffs = Vec::with_capacity(public_coeffs.len());
        for &_public_coeff in public_coeffs {
            let mask_coeff = zk_ext_mask_lc::<F, E>(hiding_cursor);
            mask_coeffs.push(mask_coeff);
        }

        let public_eval_at_zero = public_coeffs.first().copied().unwrap_or_else(E::zero);
        let public_eval_at_one = public_coeffs
            .iter()
            .copied()
            .fold(E::zero(), |acc, coeff| acc + coeff);
        let public_round_sum = public_eval_at_zero + public_eval_at_one;

        // The ordinary sumcheck chain is C_{i-1} = G_i(0) + G_i(1).
        // With public masked values C~_{i-1} = C_{i-1} + eta_{i-1}
        // and G~_i = G_i + rho_i, the verifier records:
        //
        //   G~_i(0) + G~_i(1) - C~_{i-1}
        //     + eta_{i-1} - (rho_i(0) + rho_i(1)) = 0.
        //
        // For rho_i(X) = rho_0 + rho_1 X + ...,
        // rho_i(0) + rho_i(1) = 2 * rho_0 + rho_1 + rho_2 + ...
        let mut round_sum_mask = ZkR1csLinearCombination::zero();
        for (idx, mask_coeff) in mask_coeffs.iter().enumerate() {
            let weight = if idx == 0 {
                E::one() + E::one()
            } else {
                E::one()
            };
            add_scaled_lc(&mut round_sum_mask, weight, mask_coeff);
        }

        let mut chain_residual =
            ZkR1csLinearCombination::constant(public_round_sum - previous_masked_claim);
        add_scaled_lc(&mut chain_residual, E::one(), previous_mask);
        add_scaled_lc(&mut chain_residual, -E::one(), &round_sum_mask);
        self.push_r1cs(
            description,
            chain_residual,
            ZkR1csLinearCombination::one(),
            ZkR1csLinearCombination::zero(),
        );

        // The next public claim is C~_i = G~_i(r_i), so its mask is
        // eta_i = rho_i(r_i).
        let mut next_mask = ZkR1csLinearCombination::zero();
        let mut r_power = E::one();
        for mask_coeff in &mask_coeffs {
            add_scaled_lc(&mut next_mask, r_power, mask_coeff);
            r_power *= r_round;
        }

        (next_mask, round_sum_mask)
    }

    /// Record one masked eq-factored sumcheck round relation.
    ///
    /// Eq-factored rounds do not send the full round polynomial. They send the
    /// inner polynomial coefficients except the linear term:
    ///
    /// `q~(X) = q~_0 + q~_1 X + q~_2 X^2 + ...`, with `[q~_0, q~_2, q~_3, ...]`
    /// on the transcript.
    ///
    /// The omitted linear term is represented indirectly by the incoming claim,
    /// so the verifier advances the scaled claim with a linear transition:
    ///
    /// `C~_i = previous_coeff * C~_{i-1} + sum_j coeff_j * q~_j`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn push_masked_eq_factored_round_relation<F>(
        &mut self,
        previous_masked_claim: E,
        previous_mask: &ZkR1csLinearCombination<E>,
        previous_coeff: E,
        next_masked_claim: E,
        r_round: E,
        public_coeffs_except_linear: &[E],
        transition_coeffs: &[E],
        hiding_cursor: &mut usize,
    ) -> Result<(ZkR1csLinearCombination<E>, ZkR1csLinearCombination<E>), AkitaError>
    where
        F: FieldCore,
        E: ExtField<F>,
    {
        // The mask part follows the identical transition:
        //
        //   eta_i = previous_coeff * eta_{i-1} + sum_j coeff_j * rho_j.
        let mut next_mask_transition = ZkR1csLinearCombination::zero();
        add_scaled_lc(&mut next_mask_transition, previous_coeff, previous_mask);

        // Current eq-factored round message:
        //
        //   q~_j = q_j + rho_j
        //
        // for every stored coefficient j in [0, 2, 3, ...]. Each stored
        // coefficient contributes to both the true transition and the mask
        // transition using the verifier-derived `transition_coeff`.
        debug_assert_eq!(public_coeffs_except_linear.len(), transition_coeffs.len());
        let mut known_terms_mask = ZkR1csLinearCombination::zero();
        let mut next_known_power = r_round * r_round;
        for (idx, (&_public_coeff, &transition_coeff)) in public_coeffs_except_linear
            .iter()
            .zip(transition_coeffs.iter())
            .enumerate()
        {
            let mask_coeff = zk_ext_mask_lc::<F, E>(hiding_cursor);
            add_scaled_lc(&mut next_mask_transition, transition_coeff, &mask_coeff);
            let known_weight = if idx == 0 {
                E::one()
            } else {
                let weight = next_known_power;
                next_known_power *= r_round;
                weight
            };
            add_scaled_lc(&mut known_terms_mask, known_weight, &mask_coeff);
        }
        let mut public_transition = previous_coeff * previous_masked_claim;
        for (&public_coeff, &transition_coeff) in public_coeffs_except_linear
            .iter()
            .zip(transition_coeffs.iter())
        {
            public_transition += transition_coeff * public_coeff;
        }
        let next_mask = self.new_auxilary(
            "masked eq-factored sumcheck next mask",
            next_mask_transition.clone(),
            ZkR1csLinearCombination::one(),
        )?;
        let mut transition_residual =
            ZkR1csLinearCombination::constant(next_masked_claim - public_transition);
        add_scaled_lc(&mut transition_residual, E::one(), &next_mask_transition);
        add_scaled_lc(&mut transition_residual, -E::one(), &next_mask);
        self.push_r1cs(
            "masked eq-factored sumcheck round transition",
            transition_residual,
            ZkR1csLinearCombination::one(),
            ZkR1csLinearCombination::zero(),
        );
        Ok((next_mask, known_terms_mask))
    }

    /// Check every deferred relation against the revealed plain-opening payload.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if any deferred equality fails.
    pub fn verify_all(&self, hiding_witness: &[E]) -> Result<(), AkitaError> {
        let mut witness = ZkR1csWitness::new(hiding_witness, self.aux_count);
        for relation in &self.relations {
            match relation {
                ZkR1csRelation::R1cs {
                    description,
                    a,
                    b,
                    c,
                } => {
                    let (Some(a_value), Some(b_value)) =
                        (a.evaluate(&witness), b.evaluate(&witness))
                    else {
                        tracing::error!(
                            description,
                            "deferred ZK plain-opening relation missing witness variable"
                        );
                        return Err(AkitaError::InvalidProof);
                    };
                    let Some(c_value) = c.evaluate(&witness) else {
                        tracing::error!(
                            description,
                            "deferred ZK plain-opening relation missing witness variable"
                        );
                        return Err(AkitaError::InvalidProof);
                    };
                    if a_value * b_value != c_value {
                        tracing::error!(description, "deferred ZK plain-opening relation failed");
                        return Err(AkitaError::InvalidInput(format!(
                            "deferred ZK relation failed: {description}"
                        )));
                    }
                }
                ZkR1csRelation::GenerateAuxiliary {
                    description,
                    a,
                    b,
                    auxiliary,
                } => {
                    let (Some(a_value), Some(b_value)) =
                        (a.evaluate(&witness), b.evaluate(&witness))
                    else {
                        tracing::error!(
                            description,
                            "deferred ZK auxiliary relation missing witness variable"
                        );
                        return Err(AkitaError::InvalidProof);
                    };
                    if witness.set_aux(*auxiliary, a_value * b_value).is_none() {
                        tracing::error!(description, "deferred ZK R1CS auxiliary cursor missing");
                        return Err(AkitaError::InvalidProof);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ZkR1csLinearCombination, ZkR1csVariable, ZkRelationAccumulator};
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn r1cs_relation_checks_hiding_witness_terms() {
        let mut relations = ZkRelationAccumulator::<F>::new();
        relations.push_r1cs(
            "test product",
            ZkR1csLinearCombination::variable(ZkR1csVariable::HiddenWitness(0), F::one()),
            ZkR1csLinearCombination::variable(ZkR1csVariable::HiddenWitness(1), F::one()),
            ZkR1csLinearCombination::constant(F::from_u64(12)),
        );

        relations
            .verify_all(&[F::from_u64(3), F::from_u64(4)])
            .expect("valid hiding-backed R1CS row");
    }
}
