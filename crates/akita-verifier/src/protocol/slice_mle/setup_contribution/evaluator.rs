#[cfg(test)]
use akita_algebra::ring::eval_ring_at_pows;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, MulBaseUnreduced};
use akita_types::{
    AkitaExpandedSetup, GroupedSetupContributionPlan, SetupContributionPlan,
    SetupContributionPlanInputs, WitnessLayout,
};

use crate::protocol::ring_switch::RingSwitchDeferredRowEval;

pub(crate) enum SetupEvaluatorMode<'a, F: FieldCore, E: FieldCore> {
    Direct {
        setup: &'a AkitaExpandedSetup<F>,
        prepared: &'a RingSwitchDeferredRowEval<E>,
        alpha_pows_b: &'a [E],
        alpha_pows_d: &'a [E],
    },
    #[cfg(test)]
    Recursive { setup: &'a AkitaExpandedSetup<F> },
}

pub(crate) enum SetupEvaluation<E> {
    Direct(E),
    #[cfg(test)]
    Recursive(E),
}

pub(crate) struct SetupEvaluator<'a, F: FieldCore, E: FieldCore> {
    inputs: &'a SetupContributionPlanInputs<E>,
    full_vec_randomness: &'a [E],
    eq_low: Option<&'a [E]>,
    z_block_low_eq: Option<&'a [E]>,
    alpha_pows: &'a [E],
    fold_gadget: &'a [F],
    chunk_layout: &'a WitnessLayout,
}

impl<'a, F, E> SetupEvaluator<'a, F, E>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        inputs: &'a SetupContributionPlanInputs<E>,
        full_vec_randomness: &'a [E],
        eq_low: Option<&'a [E]>,
        z_block_low_eq: Option<&'a [E]>,
        alpha_pows: &'a [E],
        fold_gadget: &'a [F],
        chunk_layout: &'a WitnessLayout,
    ) -> Self {
        Self {
            inputs,
            full_vec_randomness,
            eq_low,
            z_block_low_eq,
            alpha_pows,
            fold_gadget,
            chunk_layout,
        }
    }

    pub(crate) fn evaluate<const D: usize>(
        &self,
        mode: SetupEvaluatorMode<'_, F, E>,
    ) -> Result<SetupEvaluation<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        if self.alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.alpha_pows.len(),
            });
        }
        match mode {
            SetupEvaluatorMode::Direct {
                setup,
                prepared,
                alpha_pows_b,
                alpha_pows_d,
            } => {
                validate_grouped_role_alpha_pows(
                    prepared,
                    self.alpha_pows,
                    alpha_pows_b,
                    alpha_pows_d,
                )?;
                let plan = self.prepare_grouped(prepared)?;
                let value =
                    plan.evaluate_direct::<F>(setup, self.alpha_pows, alpha_pows_b, alpha_pows_d)?;
                Ok(SetupEvaluation::Direct(value))
            }
            #[cfg(test)]
            SetupEvaluatorMode::Recursive { setup } => {
                let plan = self.prepare_flat()?;
                let value = recursive_inner_product::<F, E, D>(&plan, setup, self.alpha_pows)?;
                Ok(SetupEvaluation::Recursive(value))
            }
        }
    }

    pub(crate) fn prepare_flat(&self) -> Result<SetupContributionPlan<E>, AkitaError> {
        SetupContributionPlan::prepare(
            self.inputs,
            self.full_vec_randomness,
            self.eq_low,
            self.z_block_low_eq,
            self.fold_gadget,
            self.chunk_layout,
        )
    }

    pub(crate) fn prepare_grouped(
        &self,
        prepared: &RingSwitchDeferredRowEval<E>,
    ) -> Result<GroupedSetupContributionPlan<E>, AkitaError> {
        SetupContributionPlan::finish_grouped_plan::<F>(
            &prepared.setup_contribution_static,
            self.full_vec_randomness,
            self.eq_low,
            self.z_block_low_eq,
            (!self.fold_gadget.is_empty()).then_some(self.fold_gadget),
            &prepared.setup_contribution_groups,
        )
    }
}

fn validate_grouped_role_alpha_pows<E: FieldCore>(
    prepared: &RingSwitchDeferredRowEval<E>,
    alpha_pows_a: &[E],
    alpha_pows_b: &[E],
    alpha_pows_d: &[E],
) -> Result<(), AkitaError> {
    let d_a = prepared.role_dims.d_a();
    if alpha_pows_a.len() != d_a {
        return Err(AkitaError::InvalidSize {
            expected: d_a,
            actual: alpha_pows_a.len(),
        });
    }
    let d_d = prepared.role_dims.d_d();
    if alpha_pows_d.len() != d_d {
        return Err(AkitaError::InvalidSize {
            expected: d_d,
            actual: alpha_pows_d.len(),
        });
    }
    let d_b = prepared.role_dims.d_b();
    if alpha_pows_b.len() != d_b {
        return Err(AkitaError::InvalidSize {
            expected: d_b,
            actual: alpha_pows_b.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
fn recursive_inner_product<F, E, const D: usize>(
    plan: &SetupContributionPlan<E>,
    setup: &AkitaExpandedSetup<F>,
    alpha_pows: &[E],
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: ExtField<F>,
{
    let bar_omega = plan.materialize_bar_omega();
    let setup_len = setup.shared_matrix().total_ring_elements_at::<D>()?;
    if setup_len < bar_omega.len() {
        return Err(AkitaError::InvalidSize {
            expected: bar_omega.len(),
            actual: setup_len,
        });
    }
    let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_len)?;
    Ok(setup_view
        .as_slice()
        .iter()
        .zip(bar_omega)
        .map(|(ring, weight)| eval_ring_at_pows(ring, alpha_pows) * weight)
        .sum())
}
