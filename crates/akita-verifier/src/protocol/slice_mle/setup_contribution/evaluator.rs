#[cfg(test)]
use akita_algebra::ring::eval_ring_at_pows;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_types::{AkitaExpandedSetup, SetupContributionPlan, SetupContributionPlanInputs};

pub(crate) enum SetupEvaluatorMode<'a, F: FieldCore> {
    Direct {
        setup: &'a AkitaExpandedSetup<F>,
    },
    #[cfg(test)]
    Recursive {
        setup: &'a AkitaExpandedSetup<F>,
    },
}

pub(crate) enum SetupEvaluation<E> {
    Direct(E),
    #[cfg(test)]
    Recursive(E),
}

pub struct SetupEvaluator<'a, F: FieldCore, E: FieldCore> {
    inputs: &'a SetupContributionPlanInputs<E>,
    full_vec_randomness: &'a [E],
    eq_low: Option<&'a [E]>,
    z_block_low_eq: Option<&'a [E]>,
    alpha_pows: &'a [E],
    fold_gadget: &'a [F],
    offset_e: usize,
    offset_t: usize,
    offset_z: usize,
    offset_u: usize,
}

impl<'a, F, E> SetupEvaluator<'a, F, E>
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inputs: &'a SetupContributionPlanInputs<E>,
        full_vec_randomness: &'a [E],
        eq_low: Option<&'a [E]>,
        z_block_low_eq: Option<&'a [E]>,
        alpha_pows: &'a [E],
        fold_gadget: &'a [F],
        offset_e: usize,
        offset_t: usize,
        offset_z: usize,
        offset_u: usize,
    ) -> Self {
        Self {
            inputs,
            full_vec_randomness,
            eq_low,
            z_block_low_eq,
            alpha_pows,
            fold_gadget,
            offset_e,
            offset_t,
            offset_z,
            offset_u,
        }
    }

    pub(crate) fn evaluate<const D: usize>(
        &self,
        mode: SetupEvaluatorMode<'_, F>,
    ) -> Result<SetupEvaluation<E>, AkitaError> {
        if self.alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.alpha_pows.len(),
            });
        }
        let plan = self.prepare()?;
        match mode {
            SetupEvaluatorMode::Direct { setup } => {
                let value = plan.evaluate_direct::<F, D>(setup, self.alpha_pows)?;
                Ok(SetupEvaluation::Direct(value))
            }
            #[cfg(test)]
            SetupEvaluatorMode::Recursive { setup } => {
                let value = recursive_inner_product::<F, E, D>(&plan, setup, self.alpha_pows)?;
                Ok(SetupEvaluation::Recursive(value))
            }
        }
    }

    pub fn prepare(&self) -> Result<SetupContributionPlan<E>, AkitaError> {
        SetupContributionPlan::prepare(
            self.inputs,
            self.full_vec_randomness,
            self.eq_low,
            self.z_block_low_eq,
            self.fold_gadget,
            self.offset_e,
            self.offset_t,
            self.offset_z,
            self.offset_u,
        )
    }
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
