mod group;

use super::*;
use crate::verifier_work::{record_verifier_work, VerifierWorkEvent};

impl<E: FieldCore> SetupContributionPlan<E> {
    pub fn evaluate_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        record_verifier_work(VerifierWorkEvent::DirectSetupEval);
        self.evaluate_role_dims_direct(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d)
    }

    fn evaluate_role_dims_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let d_a = alpha_pows_a.len();
        let d_b = alpha_pows_b.len();
        let d_d = alpha_pows_d.len();
        let base_d = role_alpha_base_len(d_a, d_b, d_d)?;
        let base_pows = alpha_pows_a.get(..base_d).ok_or(AkitaError::InvalidProof)?;
        let a_projection = if d_a == base_d {
            RoleProjection::identity()
        } else {
            role_projection(alpha_pows_a, base_pows).ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "A alpha powers do not decompose over base dimension".into(),
                )
            })?
        };
        let b_projection = role_projection(alpha_pows_b, base_pows).ok_or_else(|| {
            AkitaError::InvalidSetup("B alpha powers do not decompose over base dimension".into())
        })?;
        let d_projection = role_projection(alpha_pows_d, base_pows).ok_or_else(|| {
            AkitaError::InvalidSetup("D alpha powers do not decompose over base dimension".into())
        })?;

        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            base_d,
            |BASE_D| {
                self.evaluate_role_dims_direct_typed::<F, BASE_D>(
                    setup,
                    base_pows,
                    &a_projection,
                    &b_projection,
                    &d_projection,
                )
            }
        )
    }

    fn evaluate_role_dims_direct_typed<F, const BASE_D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        base_pows: &[E],
        a_projection: &RoleProjection<E>,
        b_projection: &RoleProjection<E>,
        d_projection: &RoleProjection<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        if base_pows.len() != BASE_D {
            return Err(AkitaError::InvalidSize {
                expected: BASE_D,
                actual: base_pows.len(),
            });
        }
        let required = self.required_base_ring_rows(
            a_projection.ratio(),
            b_projection.ratio(),
            d_projection.ratio(),
        )?;
        let setup_len = setup.shared_matrix().total_ring_elements_at::<BASE_D>()?;
        if required > setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }
        let setup_view = setup.shared_matrix().ring_view::<BASE_D>(1, setup_len)?;
        let mut acc = E::zero();
        for group in &self.groups {
            acc += group.evaluate_base_ring_direct::<F, BASE_D>(
                &setup_view,
                base_pows,
                a_projection,
                b_projection,
                d_projection,
                self.d_rows,
                self.d_physical_cols,
            )?;
        }
        Ok(acc)
    }

    fn required_base_ring_rows(
        &self,
        a_ratio: usize,
        b_ratio: usize,
        d_ratio: usize,
    ) -> Result<usize, AkitaError> {
        let mut required = self
            .d_rows
            .checked_mul(self.d_physical_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?
            .checked_mul(d_ratio)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup D base-ring footprint overflow".into())
            })?;
        for group in &self.groups {
            let b_required = group
                .n_b
                .checked_mul(group.t_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
            let a_required = group
                .n_a
                .checked_mul(group.z_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
            let b_base_required = b_required.checked_mul(b_ratio).ok_or_else(|| {
                AkitaError::InvalidSetup("setup B base-ring footprint overflow".into())
            })?;
            let a_base_required = a_required.checked_mul(a_ratio).ok_or_else(|| {
                AkitaError::InvalidSetup("setup A base-ring footprint overflow".into())
            })?;
            required = required.max(b_base_required).max(a_base_required);
        }
        Ok(required)
    }
}

fn role_alpha_base_len(d_a: usize, d_b: usize, d_d: usize) -> Result<usize, AkitaError> {
    for (role, dim) in [("A", d_a), ("B", d_b), ("D", d_d)] {
        if dim == 0 || !dim.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(format!(
                "{role} setup contribution ring dimension must be a non-zero power of two"
            )));
        }
    }
    Ok(d_a.min(d_b).min(d_d))
}
