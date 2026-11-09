//! Ordered ordinary-group compiler for the admitted grouped root.

use akita_error::{checked, AkitaError};
use akita_params::ordinary_source::OrdinarySourceWidth;
use akita_params::GroupCommitPhaseParams;
use jolt_field::Field;

use super::OrdinaryNativeRootPlan;
use crate::native_root_relation::NativeRootWeights;
use crate::root_repacking::RootCoefficientRepacking;
use crate::AkitaSetupDescriptor;

/// Exact ordered ordinary identities fixed before root randomness.
///
/// The grouped drivers bind each width and complete profile, commitment, point
/// and slot evaluation before drawing row batches and consuming these weights.
#[derive(Clone, Debug)]
pub struct OrdinaryNativeRootBatch {
    plans: Vec<OrdinaryNativeRootPlan>,
    linear_batch_count: usize,
}

/// One group's complete public claim in the batch's frozen order.
pub struct OrdinaryNativeRootClaim<'a, E> {
    pub profile: &'a GroupCommitPhaseParams,
    pub width: OrdinarySourceWidth,
    pub point: &'a [E],
    pub evaluations: &'a [E],
    pub payload: &'a [E],
}

impl OrdinaryNativeRootBatch {
    pub fn new(
        sources: &[(GroupCommitPhaseParams, OrdinarySourceWidth)],
        first_segment: usize,
    ) -> Result<Self, AkitaError> {
        if sources.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "ordinary source batch is empty".into(),
            ));
        }
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(sources.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        let mut next = first_segment;
        let mut linear_batch_count = 0;
        for &(profile, width) in sources {
            let plan = OrdinaryNativeRootPlan::new_signed(profile, next, width)?;
            next = checked::sum([next, plan.segments().len()]).ok_or(AkitaError::InvalidProof)?;
            linear_batch_count = checked::sum([linear_batch_count, plan.linear_batch_count()])
                .ok_or(AkitaError::InvalidProof)?;
            plans.push(plan);
        }
        Ok(Self {
            plans,
            linear_batch_count,
        })
    }

    pub fn plans(&self) -> &[OrdinaryNativeRootPlan] {
        &self.plans
    }

    pub fn linear_batch_count(&self) -> usize {
        self.linear_batch_count
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_weights<E: Field>(
        &self,
        repacking: &RootCoefficientRepacking,
        setup: &AkitaSetupDescriptor,
        claims: &[OrdinaryNativeRootClaim<'_, E>],
        alpha: E,
        batches: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        if claims.len() != self.plans.len() || batches.len() != self.linear_batch_count {
            return Err(AkitaError::InvalidProof);
        }
        let mut components = Vec::new();
        components
            .try_reserve_exact(self.plans.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        let mut offset = 0;
        for (plan, public) in self.plans.iter().zip(claims) {
            if public.profile != plan.profile() || Some(public.width) != plan.source_width() {
                return Err(AkitaError::InvalidProof);
            }
            let range = checked::range(offset, plan.linear_batch_count())
                .ok_or(AkitaError::InvalidProof)?;
            offset = range.end;
            components.push(plan.linear_weights(
                repacking,
                setup,
                public.point,
                public.evaluations,
                public.payload,
                alpha,
                batches.get(range).ok_or(AkitaError::InvalidProof)?,
            )?);
        }
        NativeRootWeights::merge(repacking.domain_len(), components, E::zero())
    }
}
