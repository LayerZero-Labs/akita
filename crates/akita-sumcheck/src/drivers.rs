//! Sumcheck proof driver functions.
//!
//! Contains the generic prove/verify loops for standard and eq-factored
//! sumchecks, including prefix-round omission variants used by the
//! bivariate-skip optimization.

use super::traits::{
    EqFactoredSumcheckInstanceProver, EqFactoredSumcheckInstanceVerifier,
    EqFactoredSumcheckRoundState, SumcheckInstanceProver, SumcheckInstanceVerifier,
};
use super::types::{
    EqFactoredSumcheckProof, EqFactoredSumcheckProofMasked, EqFactoredUniPoly, FullUniPoly,
    SumcheckProof, SumcheckProofMasked,
};
use akita_algebra::uni_poly::{CompressedUniPoly, UniPoly};
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;

/// Prover output for an eq-factored sumcheck with plain-opening round masks.
pub type EqFactoredMaskedProveOutput<E> = (EqFactoredSumcheckProofMasked<E>, Vec<E>);

/// Prover output for a standard sumcheck with plain-opening round masks.
pub type MaskedProveOutput<E> = (SumcheckProofMasked<E>, Vec<E>);

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

fn zk_add_scaled_lc<E: FieldCore>(
    target: &mut ZkR1csLinearCombination<E>,
    scale: E,
    source: &ZkR1csLinearCombination<E>,
) {
    target.add_scaled(scale, source);
}

fn zk_hiding_lc<E: FieldCore>(hiding_cursor: &mut usize) -> ZkR1csLinearCombination<E> {
    let variable = ZkR1csVariable::HiddenWitness(*hiding_cursor);
    *hiding_cursor += 1;
    ZkR1csLinearCombination::variable(variable, E::one())
}

/// Per-instance ZK final-relation emitter for standard sumchecks.
pub trait ZkSumcheckFinalRelation<E: FieldCore>: SumcheckInstanceVerifier<E> {
    /// Return the mask inherited by this sumcheck's input claim.
    ///
    /// Standard masked sumchecks use this as `eta_{-1}` for the first
    /// round relation. Implementations may record any handoff rows needed to
    /// synthesize the returned mask.
    ///
    /// # Errors
    ///
    /// Returns an error if the implementation cannot record its handoff rows.
    fn initial_claim_mask(
        &self,
        _relations: &mut ZkRelationAccumulator<E>,
    ) -> Result<ZkR1csLinearCombination<E>, AkitaError> {
        Ok(ZkR1csLinearCombination::zero())
    }

    /// Record the instance-specific final check as deferred relations.
    ///
    /// # Errors
    ///
    /// Returns an error if the instance cannot evaluate the relation data at
    /// the sampled challenge point.
    fn record_final_relation(
        &self,
        challenges: &[E],
        final_claim: ZkR1csLinearCombination<E>,
        relations: &mut ZkRelationAccumulator<E>,
    ) -> Result<(), AkitaError>;
}

/// Per-instance ZK final-relation emitter for eq-factored sumchecks.
pub trait ZkEqFactoredFinalRelation<E: FieldCore>: EqFactoredSumcheckInstanceVerifier<E> {
    /// Return the mask inherited by this sumcheck's input claim.
    fn initial_claim_mask(&self) -> ZkR1csLinearCombination<E> {
        ZkR1csLinearCombination::zero()
    }

    /// Record the instance-specific final check as deferred relations.
    ///
    /// # Errors
    ///
    /// Returns an error if the instance cannot evaluate the relation data at
    /// the sampled challenge point.
    fn record_final_relation(
        &self,
        round_state: &Self::RoundState,
        challenges: &[E],
        scaled_claim: ZkR1csLinearCombination<E>,
        claim_scale: E,
        handoff_mask: ZkR1csLinearCombination<E>,
        relations: &mut ZkRelationAccumulator<E>,
    ) -> Result<(), AkitaError>;
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
        zk_add_scaled_lc(&mut true_value, -E::one(), mask);
        true_value
    }

    fn push_masked_claim_relation(
        &mut self,
        _description: &'static str,
        masked_claim: E,
        mask: &ZkR1csLinearCombination<E>,
    ) -> ZkR1csLinearCombination<E> {
        Self::unmask_lc(masked_claim, mask)
    }

    /// Record one standard masked sumcheck round.
    fn push_masked_full_round_relation(
        &mut self,
        description: &'static str,
        previous_masked_claim: E,
        previous_mask: &ZkR1csLinearCombination<E>,
        public_coeffs: &[E],
        r_round: E,
        hiding_cursor: &mut usize,
    ) -> ZkR1csLinearCombination<E> {
        // Current round message:
        //
        //   G~_i(X) = G_i(X) + rho_i(X)
        //
        // `public_coeffs` are the transcript-visible coefficients of G~_i.
        // The corresponding rho_i coefficients are already present in
        // `hiding_witness`, and this verifier cursor points at those slots.
        let mut mask_coeffs = Vec::with_capacity(public_coeffs.len());
        for &_public_coeff in public_coeffs {
            let mask_coeff = zk_hiding_lc(hiding_cursor);
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
            zk_add_scaled_lc(&mut round_sum_mask, weight, mask_coeff);
        }

        let mut chain_residual =
            ZkR1csLinearCombination::constant(public_round_sum - previous_masked_claim);
        zk_add_scaled_lc(&mut chain_residual, E::one(), previous_mask);
        zk_add_scaled_lc(&mut chain_residual, -E::one(), &round_sum_mask);
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
            zk_add_scaled_lc(&mut next_mask, r_power, mask_coeff);
            r_power *= r_round;
        }

        next_mask
    }

    /// Record one eq-factored masked sumcheck round.
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
    #[allow(clippy::too_many_arguments)]
    fn push_masked_eq_factored_round_relation(
        &mut self,
        previous_masked_claim: E,
        previous_mask: &ZkR1csLinearCombination<E>,
        previous_coeff: E,
        next_masked_claim: E,
        r_round: E,
        public_coeffs_except_linear: &[E],
        transition_coeffs: &[E],
        hiding_cursor: &mut usize,
    ) -> Result<(ZkR1csLinearCombination<E>, ZkR1csLinearCombination<E>), AkitaError> {
        // The mask part follows the identical transition:
        //
        //   eta_i = previous_coeff * eta_{i-1} + sum_j coeff_j * rho_j.
        let mut next_mask_transition = ZkR1csLinearCombination::zero();
        zk_add_scaled_lc(&mut next_mask_transition, previous_coeff, previous_mask);

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
            let mask_coeff = zk_hiding_lc(hiding_cursor);
            zk_add_scaled_lc(&mut next_mask_transition, transition_coeff, &mask_coeff);
            let known_weight = if idx == 0 {
                E::one()
            } else {
                let weight = next_known_power;
                next_known_power *= r_round;
                weight
            };
            zk_add_scaled_lc(&mut known_terms_mask, known_weight, &mask_coeff);
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
        zk_add_scaled_lc(&mut transition_residual, E::one(), &next_mask_transition);
        zk_add_scaled_lc(&mut transition_residual, -E::one(), &next_mask);
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
                        return Err(AkitaError::InvalidProof);
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

/// Advance the scaled claim state for one eq-factored sumcheck round.
#[doc(hidden)]
#[inline]
pub fn advance_eq_factored_claim<E: FieldCore>(
    scaled_claim: E,
    claim_scale: E,
    l_at_0: E,
    l_at_1: E,
    poly: &EqFactoredUniPoly<E>,
    r_round: E,
) -> (E, E) {
    let q_0 = poly.constant_term();
    let q_higher_sum = poly.higher_term_sum_at_one();
    let q_known_at_r = poly.eval_known_terms(&r_round);
    let current_scalar = l_at_0 + l_at_1;
    let scaled_linear_term =
        scaled_claim - claim_scale * current_scalar * q_0 - claim_scale * l_at_1 * q_higher_sum;
    let l_at_r = l_at_0 + (l_at_1 - l_at_0) * r_round;
    let next_claim_scale = claim_scale * l_at_1;
    let next_scaled_claim =
        next_claim_scale * l_at_r * q_known_at_r + l_at_r * r_round * scaled_linear_term;
    (next_scaled_claim, next_claim_scale)
}

fn eq_factored_claim_transition_coeffs<E: FieldCore>(
    claim_scale: E,
    l_at_0: E,
    l_at_1: E,
    r_round: E,
    stored_coeff_count: usize,
) -> (E, Vec<E>) {
    let current_scalar = l_at_0 + l_at_1;
    let l_at_r = l_at_0 + (l_at_1 - l_at_0) * r_round;
    let previous_coeff = l_at_r * r_round;
    let mut coeffs = Vec::with_capacity(stored_coeff_count);
    if stored_coeff_count == 0 {
        return (previous_coeff, coeffs);
    }

    coeffs.push(claim_scale * l_at_r * (l_at_1 - r_round * current_scalar));
    let higher_coeff_base = claim_scale * l_at_1 * l_at_r;
    let mut r_power = r_round * r_round;
    for _ in 1..stored_coeff_count {
        coeffs.push(higher_coeff_base * (r_power - r_round));
        r_power *= r_round;
    }
    (previous_coeff, coeffs)
}

/// Produce an eq-factored sumcheck proof.
///
/// The prover sends the inner polynomial `q(X)` with its linear coefficient
/// omitted in every round, while the driver maintains the verifier-equivalent
/// scaled claim update.
///
/// # Errors
///
/// Returns an error if any generated round polynomial exceeds the instance's
/// degree bound.
#[tracing::instrument(skip_all, name = "prove_eq_factored_sumcheck")]
#[inline(never)]
pub fn prove_eq_factored_sumcheck<F, T, E, S, Inst>(
    instance: &mut Inst,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(EqFactoredSumcheckProof<E>, Vec<E>, E), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
    Inst: EqFactoredSumcheckInstanceProver<E>,
{
    let num_rounds = instance.num_rounds();
    let degree_bound = instance.degree_bound();
    let mut scaled_claim = instance.input_claim();
    let mut claim_scale = E::one();
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut challenges = Vec::with_capacity(num_rounds);

    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &scaled_claim);

    for round in 0..num_rounds {
        let poly = instance.compute_round_eq_factored(round);
        if poly.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "eq-factored sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
        let r_i = sample_challenge(transcript);
        let (l_at_0, l_at_1) = instance.current_linear_factor_evals();
        (scaled_claim, claim_scale) =
            advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, &poly, r_i);
        challenges.push(r_i);
        instance.ingest_challenge(round, r_i);
        round_polys.push(poly);
    }

    instance.finalize();
    Ok((
        EqFactoredSumcheckProof { round_polys },
        challenges,
        scaled_claim,
    ))
}

fn mask_eq_factored_poly<E>(
    poly: &EqFactoredUniPoly<E>,
    pad_poly: EqFactoredUniPoly<E>,
    degree_bound: usize,
) -> Result<EqFactoredUniPoly<E>, AkitaError>
where
    E: FieldCore,
{
    let stored_coeffs = EqFactoredUniPoly::<E>::stored_coeff_count_for_degree(degree_bound);
    if pad_poly.coeffs_except_linear_term.len() != stored_coeffs {
        return Err(AkitaError::InvalidProof);
    }
    let mut masked_coeffs = Vec::with_capacity(stored_coeffs);
    for idx in 0..stored_coeffs {
        let true_coeff = poly
            .coeffs_except_linear_term
            .get(idx)
            .copied()
            .unwrap_or_else(E::zero);
        let pad = pad_poly.coeffs_except_linear_term[idx];
        masked_coeffs.push(true_coeff + pad);
    }
    Ok(EqFactoredUniPoly {
        coeffs_except_linear_term: masked_coeffs,
    })
}

/// ZK extension for eq-factored sumcheck provers.
///
/// This mirrors the ordinary high-level sumcheck driver, but the transcript and
/// returned proof payload carry masked round messages only.
pub trait ZkEqFactoredSumcheckInstanceProverExt<E>: EqFactoredSumcheckInstanceProver<E>
where
    E: FieldCore,
{
    /// Prove with precommitted pad polynomials from the plain-opening hiding
    /// witness.
    ///
    /// # Errors
    ///
    /// Returns an error if pad shape is invalid or a round exceeds the degree
    /// bound.
    #[tracing::instrument(skip_all, name = "prove_zk_eq_factored_sumcheck")]
    #[inline(never)]
    fn prove_zk<F, T, S>(
        &mut self,
        transcript: &mut T,
        mut sample_challenge: S,
        pre_sampled_pads: Vec<EqFactoredUniPoly<E>>,
    ) -> Result<EqFactoredMaskedProveOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        if pre_sampled_pads.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: pre_sampled_pads.len(),
            });
        }
        let degree_bound = self.degree_bound();
        let input_claim = self.input_claim();
        let mut scaled_claim = input_claim;
        let mut claim_scale = E::one();
        let mut masked_round_polys = Vec::with_capacity(num_rounds);
        let mut challenges = Vec::with_capacity(num_rounds);

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &input_claim);

        for (round, pad_poly) in pre_sampled_pads.into_iter().enumerate() {
            let poly = self.compute_round_eq_factored(round);
            if poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "eq-factored sumcheck round poly degree {} exceeds bound {}",
                    poly.degree(),
                    degree_bound
                )));
            }
            // Eq-factored messages store q_i(X) without its linear coefficient:
            // [q_0, q_2, q_3, ...]. The ZK proof sends the masked stored part
            // q~_j = q_j + rho_j. The omitted q_1 is still determined by the
            // incoming true claim, so the prover advances its private
            // `scaled_claim` with the unmasked `poly`.
            let masked_poly = mask_eq_factored_poly(&poly, pad_poly, degree_bound)?;

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &masked_poly);
            let r_i = sample_challenge(transcript);
            let (l_at_0, l_at_1) = self.current_linear_factor_evals();
            (scaled_claim, claim_scale) =
                advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, &poly, r_i);
            challenges.push(r_i);
            self.ingest_challenge(round, r_i);
            masked_round_polys.push(masked_poly);
        }

        self.finalize();
        Ok((
            EqFactoredSumcheckProofMasked { masked_round_polys },
            challenges,
        ))
    }
}

impl<E, Inst> ZkEqFactoredSumcheckInstanceProverExt<E> for Inst
where
    E: FieldCore,
    Inst: EqFactoredSumcheckInstanceProver<E>,
{
}

/// ZK extension for eq-factored sumcheck verifiers.
pub trait ZkEqFactoredSumcheckInstanceVerifierExt<E>:
    EqFactoredSumcheckInstanceVerifier<E> + ZkEqFactoredFinalRelation<E>
where
    E: FieldCore,
{
    /// Verify masked round messages and record deferred round residuals.
    ///
    /// # Errors
    ///
    /// Returns an error if the masked round count is invalid or a round exceeds
    /// the degree bound.
    #[tracing::instrument(skip_all, name = "verify_zk_eq_factored_sumcheck")]
    #[inline(never)]
    fn verify_zk<F, T, S>(
        &self,
        masks: &EqFactoredSumcheckProofMasked<E>,
        transcript: &mut T,
        relations: &mut ZkRelationAccumulator<E>,
        hiding_cursor: &mut usize,
        mut sample_challenge: S,
    ) -> Result<(Vec<E>, ZkR1csLinearCombination<E>), AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        if masks.masked_round_polys.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: masks.masked_round_polys.len(),
            });
        }

        let degree_bound = self.degree_bound();
        let scaled_claim = self.input_claim();
        let mut scaled_claim_handle = scaled_claim;
        let mut scaled_claim_mask = self.initial_claim_mask();
        let mut masked_scaled_claim = scaled_claim;
        let mut masked_claim_scale = E::one();
        let mut challenges = Vec::with_capacity(num_rounds);
        let mut round_state = self.start_round_state();
        let mut handoff_mask = ZkR1csLinearCombination::zero();

        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &scaled_claim);

        for round in 0..num_rounds {
            let masked_poly = &masks.masked_round_polys[round];
            if masked_poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "eq-factored sumcheck round poly exceeds degree bound {degree_bound}"
                )));
            }

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, masked_poly);
            let r_i = sample_challenge(transcript);
            let (l_at_0, l_at_1) = round_state.current_linear_factor_evals();
            let previous_masked_claim = masked_scaled_claim;
            // The eq-factored verifier never receives q_1. Instead the claim
            // transition is linear in the previous claim and the stored
            // coefficients:
            //
            //   C_i = a_prev * C_{i-1} + sum_j a_j * q_j,
            //
            // where j ranges over stored coefficients [0, 2, 3, ...]. The same
            // coefficients apply to masks, giving
            //
            //   eta_i = a_prev * eta_{i-1} + sum_j a_j * rho_j.
            //
            // `coeffs_except_linear` are the a_j values; `previous_coeff` is
            // a_prev.
            let (previous_coeff, coeffs_except_linear) = eq_factored_claim_transition_coeffs(
                masked_claim_scale,
                l_at_0,
                l_at_1,
                r_i,
                masked_poly.coeffs_except_linear_term.len(),
            );
            // Advance the public masked claim using q~. The R1CS relation below
            // proves this public transition is consistent with the hidden mask
            // transition above, so the unmasked chain follows the true q.
            (masked_scaled_claim, masked_claim_scale) = advance_eq_factored_claim(
                masked_scaled_claim,
                masked_claim_scale,
                l_at_0,
                l_at_1,
                masked_poly,
                r_i,
            );
            scaled_claim_handle = masked_scaled_claim;
            // `next_claim_mask` is eta_i for the scaled running claim. The
            // `round_handoff_mask` is rho_0 + rho_2 r_i^2 + ...: the masked
            // known part of q_i(r_i), used by stage-specific final relations
            // when this round's folded witness value is handed off.
            let (next_claim_mask, round_handoff_mask) = relations
                .push_masked_eq_factored_round_relation(
                    previous_masked_claim,
                    &scaled_claim_mask,
                    previous_coeff,
                    masked_scaled_claim,
                    r_i,
                    &masked_poly.coeffs_except_linear_term,
                    &coeffs_except_linear,
                    hiding_cursor,
                )?;
            scaled_claim_mask = next_claim_mask;
            handoff_mask = round_handoff_mask;
            challenges.push(r_i);
            round_state.ingest_challenge(round, r_i);
        }

        let final_claim_lc = relations.push_masked_claim_relation(
            "eq-factored sumcheck final claim",
            scaled_claim_handle,
            &scaled_claim_mask,
        );
        self.record_final_relation(
            &round_state,
            &challenges,
            final_claim_lc,
            masked_claim_scale,
            handoff_mask.clone(),
            relations,
        )?;
        Ok((challenges, handoff_mask))
    }
}

impl<E, Inst> ZkEqFactoredSumcheckInstanceVerifierExt<E> for Inst
where
    E: FieldCore,
    Inst: EqFactoredSumcheckInstanceVerifier<E> + ZkEqFactoredFinalRelation<E>,
{
}

/// Verify an eq-factored sumcheck proof.
///
/// The verifier absorbs each round message, samples the corresponding
/// challenge, updates the scaled running claim from the current eq-factor
/// evaluations and the transmitted `q(X)` data, and finally checks the
/// expected folded oracle value at the full challenge point.
///
/// This creates and owns the mutable eq-factored round state locally, while
/// keeping `verifier` itself immutable.
///
/// # Errors
///
/// Returns an error if the proof length is invalid, a round polynomial exceeds
/// the verifier degree bound, or the final folded oracle value does not match.
#[tracing::instrument(skip_all, name = "verify_eq_factored_sumcheck")]
#[inline(never)]
pub fn verify_eq_factored_sumcheck<F, T, E, S, V>(
    proof: &EqFactoredSumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
    V: EqFactoredSumcheckInstanceVerifier<E>,
{
    let num_rounds = verifier.num_rounds();
    if proof.round_polys.len() != num_rounds {
        return Err(AkitaError::InvalidSize {
            expected: num_rounds,
            actual: proof.round_polys.len(),
        });
    }

    let degree_bound = verifier.degree_bound();
    let mut scaled_claim = verifier.input_claim();
    let mut claim_scale = E::one();
    let mut challenges = Vec::with_capacity(num_rounds);
    let mut round_state = verifier.start_round_state();

    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &scaled_claim);

    for (round, poly) in proof.round_polys.iter().enumerate() {
        if poly.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "eq-factored sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, poly);
        let r_i = sample_challenge(transcript);
        let (l_at_0, l_at_1) = round_state.current_linear_factor_evals();
        (scaled_claim, claim_scale) =
            advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, poly, r_i);
        challenges.push(r_i);
        round_state.ingest_challenge(round, r_i);
    }

    let expected = verifier.expected_output_claim(&round_state, &challenges)?;
    if scaled_claim != claim_scale * expected {
        return Err(AkitaError::InvalidProof);
    }
    Ok(challenges)
}

/// Produce a sumcheck proof while omitting the first `omitted_prefix_rounds`
/// transcript rounds from the stored proof.
///
/// This still drives the prover in the ordinary strict pipeline
/// `compute message -> absorb challenge -> ingest challenge -> ...`; it only
/// changes which compressed univariates are retained in the returned
/// [`SumcheckProof`]. Callers can use this to serialize early rounds via a
/// stage-local bivariate-skip proof instead of directly in the sumcheck proof.
///
/// # Errors
///
/// Returns an error if `omitted_prefix_rounds` exceeds the instance round
/// count, or if any per-round polynomial exceeds the instance's degree bound.
#[tracing::instrument(skip_all, name = "prove_sumcheck")]
#[inline(never)]
pub fn prove_sumcheck_with_omitted_prefix_rounds<F, T, E, S, Inst, A>(
    instance: &mut Inst,
    transcript: &mut T,
    mut sample_challenge: S,
    omitted_prefix_rounds: usize,
    mut absorb_after_compute: A,
) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
    Inst: SumcheckInstanceProver<E>,
    A: FnMut(usize, &Inst, &mut T) -> Result<(), AkitaError>,
{
    let num_rounds = instance.num_rounds();
    if omitted_prefix_rounds > num_rounds {
        return Err(AkitaError::InvalidInput(format!(
            "sumcheck omitted_prefix_rounds {omitted_prefix_rounds} exceeds num_rounds {num_rounds}"
        )));
    }

    let mut claim = instance.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        omitted_prefix_rounds,
        "prove_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = instance.degree_bound();
    let mut round_polys = Vec::with_capacity(num_rounds - omitted_prefix_rounds);
    let mut r = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let g = instance.compute_round_univariate(round, claim);
        let round_sum = g.evaluate(&E::zero()) + g.evaluate(&E::one());
        debug_assert!(
            round_sum == claim,
            "sumcheck round {round} univariate does not match previous claim hint"
        );

        let compressed = g.compress();
        if compressed.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                compressed.degree(),
                degree_bound
            )));
        }

        absorb_after_compute(round, instance, transcript)?;
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let r_i = sample_challenge(transcript);
        r.push(r_i);

        claim = compressed.eval_from_hint(&claim, &r_i);
        instance.ingest_challenge(round, r_i);
        if round >= omitted_prefix_rounds {
            round_polys.push(compressed);
        }
    }

    instance.finalize();
    Ok((SumcheckProof { round_polys }, r, claim))
}

/// Verify a sumcheck proof whose first `prefix_rounds` rounds are reconstructed by
/// a caller-supplied generator instead of being stored in `proof`.
///
/// The verifier still follows the ordinary transcript pipeline, sampling each
/// challenge only after absorbing that round's compressed univariate. For
/// rounds `round < prefix_rounds`, the compressed univariate is provided by
/// `prefix_round_poly`; later rounds are read from `proof`.
///
/// Returns the full challenge point `r` on success.
///
/// # Errors
///
/// Returns an error if `prefix_rounds` exceeds the verifier round count, if the
/// suffix proof length is inconsistent, if a generated/stored round polynomial
/// exceeds the degree bound, or if the final oracle check fails.
#[tracing::instrument(skip_all, name = "verify_sumcheck")]
#[inline(never)]
pub fn verify_sumcheck_with_prefix_rounds<F, T, E, S, V, A, P>(
    proof: &SumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    mut sample_challenge: S,
    prefix_rounds: usize,
    mut absorb_before_round: A,
    mut prefix_round_poly: P,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
    V: SumcheckInstanceVerifier<E>,
    A: FnMut(usize, &mut T) -> Result<(), AkitaError>,
    P: FnMut(usize, E, &[E]) -> CompressedUniPoly<E>,
{
    let num_rounds = verifier.num_rounds();
    if prefix_rounds > num_rounds {
        return Err(AkitaError::InvalidInput(format!(
            "sumcheck prefix_rounds {prefix_rounds} exceeds num_rounds {num_rounds}"
        )));
    }
    let expected_suffix_rounds = num_rounds - prefix_rounds;
    if proof.round_polys.len() != expected_suffix_rounds {
        return Err(AkitaError::InvalidSize {
            expected: expected_suffix_rounds,
            actual: proof.round_polys.len(),
        });
    }

    let mut claim = verifier.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        prefix_rounds,
        "verify_sumcheck input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = verifier.degree_bound();
    let mut challenges = Vec::with_capacity(num_rounds);
    let mut suffix_iter = proof.round_polys.iter();

    for round in 0..num_rounds {
        absorb_before_round(round, transcript)?;
        let poly = if round < prefix_rounds {
            prefix_round_poly(round, claim, &challenges)
        } else {
            suffix_iter.next().cloned().ok_or(AkitaError::InvalidSize {
                expected: expected_suffix_rounds,
                actual: proof.round_polys.len(),
            })?
        };
        if poly.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
        let r_i = sample_challenge(transcript);
        challenges.push(r_i);
        claim = poly.eval_from_hint(&claim, &r_i);
    }
    debug_assert!(suffix_iter.next().is_none());

    check_sumcheck_output_claim(claim, verifier, &challenges)?;
    Ok(challenges)
}

/// Enforce the final sumcheck oracle equality for the provided challenge point.
///
/// This is useful when some prefix rounds are reconstructed outside the generic
/// verifier driver and the caller needs to check the final oracle value against
/// the full concatenated challenge vector.
///
/// # Errors
///
/// Returns any error produced by `verifier.expected_output_claim`, or
/// [`AkitaError::InvalidProof`] if the final claim does not match the oracle
/// evaluation at `challenges`.
pub fn check_sumcheck_output_claim<E, V>(
    final_claim: E,
    verifier: &V,
    challenges: &[E],
) -> Result<(), AkitaError>
where
    E: FieldCore + AkitaSerialize,
    V: SumcheckInstanceVerifier<E>,
{
    let expected = verifier.expected_output_claim(challenges)?;
    if final_claim != expected {
        tracing::error!(
            rounds = verifier.num_rounds(),
            degree_bound = verifier.degree_bound(),
            diff_is_zero = (final_claim - expected).is_zero(),
            "verify_sumcheck MISMATCH"
        );
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

fn pad_coeffs_to_degree<E: FieldCore>(coeffs: &[E], degree_bound: usize) -> Vec<E> {
    let mut out = coeffs.to_vec();
    out.resize(degree_bound.saturating_add(1), E::zero());
    out
}

fn mask_full_poly<E>(
    poly: &UniPoly<E>,
    pad_poly: FullUniPoly<E>,
    degree_bound: usize,
) -> Result<FullUniPoly<E>, AkitaError>
where
    E: FieldCore,
{
    let true_coeffs = pad_coeffs_to_degree(&poly.coeffs, degree_bound);
    if pad_poly.coeffs().len() != true_coeffs.len() {
        return Err(AkitaError::InvalidProof);
    }
    let mut masked_coeffs = Vec::with_capacity(true_coeffs.len());
    for (idx, true_coeff) in true_coeffs.into_iter().enumerate() {
        let pad = pad_poly.coeffs()[idx];
        masked_coeffs.push(true_coeff + pad);
    }
    Ok(FullUniPoly::from_coeffs(masked_coeffs))
}

/// Produce a sumcheck proof for a single instance, driving the Fiat-Shamir transcript.
///
/// This method:
/// - does **not** absorb the initial claim into the transcript (callers should do so),
/// - appends each round message under `labels::ABSORB_SUMCHECK_ROUND`,
/// - samples one challenge per round via `sample_challenge`,
/// - updates the running claim using the per-round hint (`g(0)+g(1)`).
///
/// It returns the proof, the derived point `r`, and the final claimed value at `r`.
///
/// # Errors
///
/// Returns an error if any per-round polynomial exceeds the instance's degree bound.
pub fn prove_sumcheck<F, T, E, S, Inst>(
    instance: &mut Inst,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
    Inst: SumcheckInstanceProver<E>,
{
    prove_sumcheck_with_omitted_prefix_rounds::<F, T, E, S, Inst, _>(
        instance,
        transcript,
        sample_challenge,
        0,
        |_, _, _| Ok(()),
    )
}

/// ZK extension for standard sumcheck provers.
pub trait ZkSumcheckInstanceProverExt<E>: SumcheckInstanceProver<E>
where
    E: FieldCore,
{
    /// Prove with precommitted pad polynomials from the plain-opening hiding
    /// witness.
    ///
    /// # Errors
    ///
    /// Returns an error if pad shape is invalid or a round exceeds the degree
    /// bound.
    #[tracing::instrument(skip_all, name = "prove_zk_sumcheck")]
    #[inline(never)]
    fn prove_zk<F, T, S>(
        &mut self,
        transcript: &mut T,
        mut sample_challenge: S,
        pre_sampled_pads: Vec<FullUniPoly<E>>,
    ) -> Result<MaskedProveOutput<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        if pre_sampled_pads.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: pre_sampled_pads.len(),
            });
        }
        let mut claim = self.input_claim();
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

        let degree_bound = self.degree_bound();
        let mut masked_round_polys = Vec::with_capacity(num_rounds);
        let mut r = Vec::with_capacity(num_rounds);

        for (round, pad_poly) in pre_sampled_pads.into_iter().enumerate() {
            let g = self.compute_round_univariate(round, claim);

            let compressed = g.compress();
            if compressed.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "sumcheck round poly degree {} exceeds bound {}",
                    compressed.degree(),
                    degree_bound
                )));
            }
            let masked_poly = mask_full_poly(&g, pad_poly, degree_bound)?;

            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &masked_poly);
            let r_i = sample_challenge(transcript);
            r.push(r_i);

            claim = g.evaluate(&r_i);
            self.ingest_challenge(round, r_i);
            masked_round_polys.push(masked_poly);
        }

        self.finalize();
        Ok((SumcheckProofMasked { masked_round_polys }, r))
    }
}

impl<E, Inst> ZkSumcheckInstanceProverExt<E> for Inst
where
    E: FieldCore,
    Inst: SumcheckInstanceProver<E>,
{
}

/// Verify a single-instance sumcheck proof.
///
/// This function:
/// - absorbs the initial claim into the transcript,
/// - delegates round-by-round verification to [`SumcheckProof::verify`],
/// - performs the final oracle check: `final_claim == verifier.expected_output_claim(r)`.
///
/// Returns the challenge point `r` on success.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if the final sumcheck claim does not
/// match the oracle evaluation, or propagates any error from the per-round
/// verification (e.g. degree-bound violation, round-count mismatch).
pub fn verify_sumcheck<F, T, E, S, V>(
    proof: &SumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
    V: SumcheckInstanceVerifier<E>,
{
    verify_sumcheck_with_prefix_rounds::<F, T, E, S, V, _, _>(
        proof,
        verifier,
        transcript,
        sample_challenge,
        0,
        |_, _| Ok(()),
        |_, _, _| unreachable!("no prefix rounds requested"),
    )
}

/// ZK extension for standard sumcheck verifiers.
pub trait ZkSumcheckInstanceVerifierExt<E>:
    SumcheckInstanceVerifier<E> + ZkSumcheckFinalRelation<E>
where
    E: FieldCore,
{
    /// Verify masked round messages and record deferred round residuals.
    ///
    /// # Errors
    ///
    /// Returns an error if the masked round count is invalid or a round exceeds
    /// the degree bound.
    #[tracing::instrument(skip_all, name = "verify_zk_sumcheck")]
    #[inline(never)]
    fn verify_zk<F, T, S>(
        &self,
        masks: &SumcheckProofMasked<E>,
        transcript: &mut T,
        relations: &mut ZkRelationAccumulator<E>,
        hiding_cursor: &mut usize,
        mut sample_challenge: S,
    ) -> Result<Vec<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        T: Transcript<F>,
        E: AkitaSerialize,
        S: FnMut(&mut T) -> E,
    {
        let num_rounds = self.num_rounds();
        if masks.masked_round_polys.len() != num_rounds {
            return Err(AkitaError::InvalidSize {
                expected: num_rounds,
                actual: masks.masked_round_polys.len(),
            });
        }

        let initial_claim = self.input_claim();
        let mut masked_claim_handle = initial_claim;
        let mut claim_mask = self.initial_claim_mask(relations)?;
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &initial_claim);

        let degree_bound = self.degree_bound();
        let mut challenges = Vec::with_capacity(num_rounds);
        for round in 0..num_rounds {
            let masked_poly = &masks.masked_round_polys[round];
            if masked_poly.degree() > degree_bound {
                return Err(AkitaError::InvalidInput(format!(
                    "sumcheck round poly exceeds degree bound {degree_bound}"
                )));
            }
            transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, masked_poly);
            let r_i = sample_challenge(transcript);
            challenges.push(r_i);
            let description = if round == 0 {
                "masked sumcheck input chain"
            } else {
                "masked sumcheck round chain"
            };
            let next_claim_mask = relations.push_masked_full_round_relation(
                description,
                masked_claim_handle,
                &claim_mask,
                masked_poly.coeffs(),
                r_i,
                hiding_cursor,
            );
            masked_claim_handle = masked_poly.evaluate(&r_i);
            claim_mask = next_claim_mask;
        }

        let final_claim_lc = relations.push_masked_claim_relation(
            "sumcheck final claim",
            masked_claim_handle,
            &claim_mask,
        );
        self.record_final_relation(&challenges, final_claim_lc, relations)?;
        Ok(challenges)
    }
}

impl<E, Inst> ZkSumcheckInstanceVerifierExt<E> for Inst
where
    E: FieldCore,
    Inst: SumcheckInstanceVerifier<E> + ZkSumcheckFinalRelation<E>,
{
}

#[cfg(test)]
mod tests {
    use super::{ZkR1csLinearCombination, ZkR1csVariable, ZkRelationAccumulator};
    use akita_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn r1cs_equality_relation_accepts_matching_scalars() {
        let mut relations = ZkRelationAccumulator::<F>::new();
        relations.push_r1cs(
            "test equality",
            ZkR1csLinearCombination::constant(F::zero()),
            ZkR1csLinearCombination::one(),
            ZkR1csLinearCombination::zero(),
        );

        relations.verify_all(&[]).expect("matching R1CS row");
    }

    #[test]
    fn r1cs_relation_checks_hiding_witness_terms() {
        let mut relations = ZkRelationAccumulator::<F>::new();
        relations.push_r1cs(
            "test square",
            ZkR1csLinearCombination::variable(ZkR1csVariable::HiddenWitness(0), F::one()),
            ZkR1csLinearCombination::variable(ZkR1csVariable::HiddenWitness(0), F::one()),
            ZkR1csLinearCombination::constant(F::from_u64(9)),
        );

        relations
            .verify_all(&[F::from_u64(3)])
            .expect("valid hiding-backed R1CS row");
    }
}
