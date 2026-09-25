use super::super::physical_l2_norm::{exact_claims, prepare_norm_term, FusedRangeNormProver};
use super::*;
use crate::opaque::{
    Stage1FinalClaims, Stage1PublicTransition, Stage1RoundPolynomial, Stage1Step, Stage1Transition,
};
use akita_sumcheck::{EqFactoredSumcheckInstanceProver, SumcheckInstanceProver};
use jolt_field::{Fold, Unreduced};
use jolt_poly::UnivariatePolynomial;

enum ActiveStage<E: Field> {
    Low(LowBasisRangeCheckProver<E>),
    Product2(ClassIndexedProductSubcheckProver<E, 2>),
    Product4(ClassIndexedProductSubcheckProver<E, 4>),
    Product8(ClassIndexedProductSubcheckProver<E, 8>),
    Leaf(ClassIndexedRangeLeafProver<E>),
    Fused(FusedRangeNormProver<E>),
}

impl<E: Field + Ring + Fold + Unreduced> ActiveStage<E> {
    fn step(&self, product_index: usize) -> Stage1Step {
        match self {
            Self::Low(_) | Self::Leaf(_) => Stage1Step::RangeLeaf,
            Self::Product2(_) | Self::Product4(_) | Self::Product8(_) => {
                Stage1Step::Product(product_index)
            }
            Self::Fused(_) => Stage1Step::FusedRangeNorm,
        }
    }

    fn num_rounds(&self) -> usize {
        match self {
            Self::Low(p) => p.num_rounds(),
            Self::Product2(p) => p.num_rounds(),
            Self::Product4(p) => p.num_rounds(),
            Self::Product8(p) => p.num_rounds(),
            Self::Leaf(p) => p.num_rounds(),
            Self::Fused(p) => p.num_rounds(),
        }
    }

    fn degree_bound(&self) -> usize {
        match self {
            Self::Low(p) => p.degree_bound(),
            Self::Product2(p) => p.degree_bound(),
            Self::Product4(p) => p.degree_bound(),
            Self::Product8(p) => p.degree_bound(),
            Self::Leaf(p) => p.degree_bound(),
            Self::Fused(p) => p.degree_bound(),
        }
    }

    fn round(&mut self, round: usize, claim: E) -> (Stage1RoundPolynomial<E>, Option<E>) {
        match self {
            Self::Low(p) => {
                let tau = p.current_tau();
                (
                    Stage1RoundPolynomial::EqFactored(p.compute_round_eq_factored(round)),
                    Some(tau),
                )
            }
            Self::Product2(p) => {
                let tau = p.current_tau();
                (
                    Stage1RoundPolynomial::EqFactored(p.compute_round_eq_factored(round)),
                    Some(tau),
                )
            }
            Self::Product4(p) => {
                let tau = p.current_tau();
                (
                    Stage1RoundPolynomial::EqFactored(p.compute_round_eq_factored(round)),
                    Some(tau),
                )
            }
            Self::Product8(p) => {
                let tau = p.current_tau();
                (
                    Stage1RoundPolynomial::EqFactored(p.compute_round_eq_factored(round)),
                    Some(tau),
                )
            }
            Self::Leaf(p) => {
                let tau = p.current_tau();
                (
                    Stage1RoundPolynomial::EqFactored(p.compute_round_eq_factored(round)),
                    Some(tau),
                )
            }
            Self::Fused(p) => (
                Stage1RoundPolynomial::Standard(p.compute_round_univariate(round, claim)),
                None,
            ),
        }
    }

    fn bind(&mut self, round: usize, challenge: E) {
        match self {
            Self::Low(p) => p.ingest_challenge(round, challenge),
            Self::Product2(p) => p.ingest_challenge(round, challenge),
            Self::Product4(p) => p.ingest_challenge(round, challenge),
            Self::Product8(p) => p.ingest_challenge(round, challenge),
            Self::Leaf(p) => p.ingest_challenge(round, challenge),
            Self::Fused(p) => p.ingest_challenge(round, challenge),
        }
    }

    fn finalize(&mut self) {
        match self {
            Self::Low(p) => p.finalize(),
            Self::Product2(p) => p.finalize(),
            Self::Product4(p) => p.finalize(),
            Self::Product8(p) => p.finalize(),
            Self::Leaf(p) => p.finalize(),
            Self::Fused(p) => p.finalize(),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Phase {
    Running,
    ProductPublic,
    ProductBatch,
    L2Public,
    L2Batch,
    Merge,
    FinalPublic,
    Ready,
}

struct PendingRound<E: Field> {
    step: Stage1Step,
    round: usize,
    polynomial: Stage1RoundPolynomial<E>,
    tau: Option<E>,
}

struct PhysicalState<E: Field> {
    plan: akita_types::PhysicalResponsePlan,
    integers: Vec<Vec<i128>>,
    response_l2_sq: u128,
    subclaims: Vec<E>,
    subclaim_weights: Vec<E>,
}

pub(crate) struct DigitRangeSession<E: Field> {
    source: CompactDigitSource,
    plan: DigitRangePlan,
    leaf_coeffs: Vec<Vec<E>>,
    equality_point: Vec<E>,
    weights: Vec<E>,
    claim: E,
    product_index: usize,
    active: Option<ActiveStage<E>>,
    phase: Phase,
    next_round: usize,
    pending: Option<PendingRound<E>>,
    pending_product_claims: Option<Vec<E>>,
    point: Vec<E>,
    physical: Option<PhysicalState<E>>,
    final_range_image: Option<E>,
    final_virtual_evaluations: Option<Vec<E>>,
}

impl<E: Field + Ring + Fold + Unreduced> DigitRangeSession<E> {
    pub(crate) fn new(
        prover: DigitRangeProver<E>,
        physical: Option<akita_types::PhysicalResponsePlan>,
    ) -> Result<Self, AkitaError> {
        let DigitRangeProver {
            mut digit_source,
            equality_point,
            plan,
            live_block_count,
            high_variable_count,
            low_variable_count,
        } = prover;
        if let Some(physical) = &physical {
            if physical.domain().num_vars() != equality_point.len()
                || physical.domain().live_len() != digit_source.live_len()
            {
                return Err(AkitaError::InvalidSetup(
                    "physical response and digit-range domains disagree".into(),
                ));
            }
        }
        if physical.is_none() && plan.basis() <= 8 {
            let active = LowBasisRangeCheckProver::new(
                digit_source.digits(),
                &equality_point,
                plan,
                live_block_count,
                high_variable_count,
                low_variable_count,
            )?;
            return Ok(Self {
                source: digit_source,
                plan,
                leaf_coeffs: Vec::new(),
                equality_point,
                weights: vec![E::one()],
                claim: E::zero(),
                product_index: 0,
                active: Some(ActiveStage::Low(active)),
                phase: Phase::Running,
                next_round: 0,
                pending: None,
                pending_product_claims: None,
                point: Vec::new(),
                physical: None,
                final_range_image: None,
                final_virtual_evaluations: None,
            });
        }
        if physical.is_some() {
            digit_source.prepare_class_indexed_leaf();
        }
        let mut session = Self {
            source: digit_source,
            plan,
            leaf_coeffs: plan.leaf_coeffs::<E>(),
            equality_point,
            weights: vec![E::one()],
            claim: E::zero(),
            product_index: 0,
            active: None,
            phase: Phase::Running,
            next_round: 0,
            pending: None,
            pending_product_claims: None,
            point: Vec::new(),
            physical: None,
            final_range_image: None,
            final_virtual_evaluations: None,
        };
        if let Some(physical) = physical {
            let compact = session.source.digits();
            let integers =
                physical.materialize_virtual_integers(compact.len(), |start, output| {
                    compact.view().decode_range(start, output)?;
                    Ok(())
                })?;
            let (response_l2_sq, subclaims) = exact_claims::<E>(&physical, &integers)?;
            session.physical = Some(PhysicalState {
                plan: physical,
                integers,
                response_l2_sq,
                subclaims,
                subclaim_weights: Vec::new(),
            });
        }
        session.start_product_or_leaf()?;
        Ok(session)
    }

    fn start_product_or_leaf(&mut self) -> Result<(), AkitaError> {
        if self.product_index < self.plan.product_stage_arities().len() {
            let stage = self.product_index;
            let input = ProductSubcheckInput {
                source: self.source.clone(),
                plan: self.plan,
                leaf_polynomials: &self.leaf_coeffs,
                stage_index: stage,
                parent_weights: self.weights.clone(),
                equality_point: &self.equality_point,
                input_claim: self.claim,
            };
            self.active = Some(
                match self
                    .plan
                    .product_stage_lane_count(stage)
                    .ok_or(AkitaError::InvalidProof)?
                {
                    2 => ActiveStage::Product2(ClassIndexedProductSubcheckProver::new(
                        input.source,
                        input.plan,
                        input.leaf_polynomials,
                        input.stage_index,
                        input.parent_weights,
                        input.equality_point,
                        input.input_claim,
                    )?),
                    4 => ActiveStage::Product4(ClassIndexedProductSubcheckProver::new(
                        input.source,
                        input.plan,
                        input.leaf_polynomials,
                        input.stage_index,
                        input.parent_weights,
                        input.equality_point,
                        input.input_claim,
                    )?),
                    8 => ActiveStage::Product8(ClassIndexedProductSubcheckProver::new(
                        input.source,
                        input.plan,
                        input.leaf_polynomials,
                        input.stage_index,
                        input.parent_weights,
                        input.equality_point,
                        input.input_claim,
                    )?),
                    _ => return Err(AkitaError::InvalidProof),
                },
            );
            self.begin_active();
        } else if self.physical.is_some() {
            self.phase = Phase::L2Public;
        } else {
            let coefficients = self
                .plan
                .batch_leaf_polynomials(&self.weights, &self.leaf_coeffs)?;
            self.active = Some(ActiveStage::Leaf(ClassIndexedRangeLeafProver::new(
                self.source.clone(),
                &self.equality_point,
                self.claim,
                coefficients,
            )?));
            self.begin_active();
        }
        Ok(())
    }

    fn begin_active(&mut self) {
        self.phase = Phase::Running;
        self.next_round = 0;
        self.pending = None;
        self.point.clear();
    }

    pub(crate) fn round_polynomial(
        &mut self,
        step: Stage1Step,
        round: usize,
        previous_claim: E,
    ) -> Result<Stage1RoundPolynomial<E>, AkitaError> {
        let active = self.active.as_mut().ok_or_else(|| {
            AkitaError::InvalidInput("Stage 1 has no active sumcheck step".into())
        })?;
        if self.phase != Phase::Running
            || active.step(self.product_index) != step
            || round != self.next_round
            || self.pending.is_some()
            || previous_claim != self.claim
        {
            return Err(AkitaError::InvalidInput(
                "Stage 1 step, round, or previous claim mismatch".into(),
            ));
        }
        let degree_bound = active.degree_bound();
        let (polynomial, tau) = active.round(round, self.claim);
        let valid = match &polynomial {
            Stage1RoundPolynomial::EqFactored(poly) => poly.degree() <= degree_bound,
            Stage1RoundPolynomial::Standard(poly) => {
                poly.degree() <= degree_bound
                    && poly.evaluate(E::zero()) + poly.evaluate(E::one()) == self.claim
            }
        };
        if !valid {
            return Err(AkitaError::InvalidInput(
                "Stage 1 consumer returned an invalid round polynomial".into(),
            ));
        }
        self.pending = Some(PendingRound {
            step,
            round,
            polynomial: polynomial.clone(),
            tau,
        });
        Ok(polynomial)
    }

    pub(crate) fn bind_challenge(
        &mut self,
        step: Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        let pending = self.pending.take().ok_or_else(|| {
            AkitaError::InvalidInput("Stage 1 challenge arrived before its polynomial".into())
        })?;
        if pending.step != step || pending.round != round {
            return Err(AkitaError::InvalidInput(
                "Stage 1 challenge step or round mismatch".into(),
            ));
        }
        self.claim = match &pending.polynomial {
            Stage1RoundPolynomial::EqFactored(poly) => akita_sumcheck::advance_eq_factored_claim(
                self.claim,
                pending.tau.ok_or(AkitaError::InvalidProof)?,
                poly,
                challenge,
            ),
            Stage1RoundPolynomial::Standard(poly) => poly.evaluate(challenge),
        };
        let active = self.active.as_mut().ok_or(AkitaError::InvalidProof)?;
        active.bind(round, challenge);
        self.point.push(challenge);
        self.next_round += 1;
        if self.next_round == active.num_rounds() {
            active.finalize();
            self.phase = match active {
                ActiveStage::Product2(_) | ActiveStage::Product4(_) | ActiveStage::Product8(_) => {
                    Phase::ProductPublic
                }
                _ => Phase::FinalPublic,
            };
        }
        Ok(())
    }

    pub(crate) fn public_transition(
        &mut self,
        step: Stage1Step,
    ) -> Result<Stage1PublicTransition<E>, AkitaError> {
        match self.phase {
            Phase::ProductPublic if step == Stage1Step::Product(self.product_index) => {
                let claims = match self.active.take().ok_or(AkitaError::InvalidProof)? {
                    ActiveStage::Product2(p) => p.final_child_claims(),
                    ActiveStage::Product4(p) => p.final_child_claims(),
                    ActiveStage::Product8(p) => p.final_child_claims(),
                    _ => return Err(AkitaError::InvalidProof),
                };
                self.pending_product_claims = Some(claims.clone());
                self.phase = Phase::ProductBatch;
                Ok(Stage1PublicTransition::ProductChildClaims(claims))
            }
            Phase::L2Public if step == Stage1Step::FusedRangeNorm => {
                let physical = self.physical.as_ref().ok_or(AkitaError::InvalidProof)?;
                self.phase = if physical.subclaims.is_empty() {
                    Phase::Merge
                } else {
                    Phase::L2Batch
                };
                Ok(Stage1PublicTransition::PhysicalL2Claims {
                    response_l2_sq: physical.response_l2_sq,
                    subclaims: physical.subclaims.clone(),
                })
            }
            Phase::FinalPublic => {
                let active = self.active.take().ok_or(AkitaError::InvalidProof)?;
                if active.step(self.product_index) != step {
                    return Err(AkitaError::InvalidInput(
                        "Stage 1 final transition step mismatch".into(),
                    ));
                }
                let (range_image_evaluation, virtual_evaluations) = match active {
                    ActiveStage::Low(p) => (p.final_range_image_eval(), Vec::new()),
                    ActiveStage::Leaf(p) => (p.final_range_image_eval(), Vec::new()),
                    ActiveStage::Fused(p) => {
                        let expected =
                            p.range.final_range_claim() + p.norm_merge * p.norm.final_claim()?;
                        if self.claim != expected {
                            return Err(AkitaError::InvalidInput(
                                "fused Stage 1 final claim mismatch".into(),
                            ));
                        }
                        (
                            p.range.final_range_image_eval(),
                            p.norm.virtual_evaluations()?,
                        )
                    }
                    _ => return Err(AkitaError::InvalidProof),
                };
                self.final_range_image = Some(range_image_evaluation);
                self.final_virtual_evaluations = Some(virtual_evaluations.clone());
                self.phase = Phase::Ready;
                Ok(Stage1PublicTransition::Final {
                    range_image_evaluation,
                    virtual_evaluations,
                })
            }
            _ => Err(AkitaError::InvalidInput(
                "unexpected Stage 1 public transition".into(),
            )),
        }
    }

    pub(crate) fn bind_batch_challenge(
        &mut self,
        transition: Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError> {
        match (self.phase, transition) {
            (Phase::ProductBatch, Stage1Transition::ProductBatch(index))
                if index == self.product_index =>
            {
                let claims = match self.plan.product_stage_lane_count(index) {
                    Some(count) => count,
                    None => return Err(AkitaError::InvalidProof),
                };
                let child_claims = self
                    .pending_product_claims
                    .take()
                    .ok_or(AkitaError::InvalidProof)?;
                if child_claims.len() != claims {
                    return Err(AkitaError::InvalidProof);
                }
                self.weights = self.plan.interstage_batch_weights(challenge, claims);
                self.claim = self.plan.batch_claims(&self.weights, &child_claims)?;
                self.equality_point.clone_from(&self.point);
                self.product_index += 1;
                self.start_product_or_leaf()
            }
            (Phase::L2Batch, Stage1Transition::L2SubclaimBatch) => {
                let physical = self.physical.as_mut().ok_or(AkitaError::InvalidProof)?;
                let mut power = E::one();
                physical.subclaim_weights = physical
                    .subclaims
                    .iter()
                    .map(|_| {
                        let weight = power;
                        power *= challenge;
                        weight
                    })
                    .collect();
                self.phase = Phase::Merge;
                Ok(())
            }
            (Phase::Merge, Stage1Transition::RangeNormMerge) => self.start_fused(challenge),
            _ => Err(AkitaError::InvalidInput(
                "unexpected Stage 1 batching challenge".into(),
            )),
        }
    }

    fn start_fused(&mut self, merge: E) -> Result<(), AkitaError> {
        let coefficients = self
            .plan
            .batch_leaf_polynomials(&self.weights, &self.leaf_coeffs)?;
        let range = ClassIndexedRangeLeafProver::new(
            self.source.clone(),
            &self.equality_point,
            self.claim,
            coefficients,
        )?;
        let physical = self.physical.as_mut().ok_or(AkitaError::InvalidProof)?;
        let norm_input_claim = if physical.subclaims.is_empty() {
            E::from_u128(physical.response_l2_sq)
        } else {
            physical
                .subclaims
                .iter()
                .zip(&physical.subclaim_weights)
                .fold(E::zero(), |sum, (&claim, &weight)| sum + claim * weight)
        };
        let norm = prepare_norm_term(
            &physical.plan,
            core::mem::take(&mut physical.integers),
            &physical.subclaim_weights,
        )?;
        let range_input_claim = range.input_claim();
        self.claim = range_input_claim + merge * norm_input_claim;
        self.active = Some(ActiveStage::Fused(FusedRangeNormProver {
            range,
            norm,
            norm_merge: merge,
            input_claim: self.claim,
            rounds_completed: 0,
        }));
        self.begin_active();
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<Stage1FinalClaims<E>, AkitaError> {
        if self.phase != Phase::Ready || self.pending.is_some() {
            return Err(AkitaError::InvalidInput(
                "Stage 1 finalized before its public claims were consumed".into(),
            ));
        }
        Ok(Stage1FinalClaims::new(self.point, self.claim))
    }
}
