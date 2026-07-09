mod group;

use super::*;

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
        let d_a = alpha_pows_a.len();
        let d_b = alpha_pows_b.len();
        let d_d = alpha_pows_d.len();
        if d_a == 0 || d_b == 0 || d_d == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup contribution role alpha powers must be non-empty".into(),
            ));
        }
        if d_a == d_b && d_b == d_d && alpha_pows_a == alpha_pows_b && alpha_pows_a == alpha_pows_d
        {
            return self.evaluate_uniform_direct(setup, alpha_pows_a, d_a);
        }
        if let Some(value) =
            self.evaluate_divisible_direct(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d)?
        {
            return Ok(value);
        }

        self.evaluate_packed_direct(setup, alpha_pows_a, alpha_pows_b, alpha_pows_d)
    }

    fn evaluate_divisible_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<Option<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let base_d = alpha_pows_a
            .len()
            .min(alpha_pows_b.len())
            .min(alpha_pows_d.len());
        if base_d == 0
            || !alpha_pows_a.len().is_multiple_of(base_d)
            || !alpha_pows_b.len().is_multiple_of(base_d)
            || !alpha_pows_d.len().is_multiple_of(base_d)
        {
            return Ok(None);
        }
        let base_pows = if alpha_pows_a.len() == base_d {
            alpha_pows_a
        } else if alpha_pows_b.len() == base_d {
            alpha_pows_b
        } else {
            alpha_pows_d
        };
        let Some(a_scales) = alpha_chunk_scales(alpha_pows_a, base_pows) else {
            return Ok(None);
        };
        let Some(b_scales) = alpha_chunk_scales(alpha_pows_b, base_pows) else {
            return Ok(None);
        };
        let Some(d_scales) = alpha_chunk_scales(alpha_pows_d, base_pows) else {
            return Ok(None);
        };

        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            base_d,
            |BASE_D| {
                self.evaluate_divisible_direct_typed::<F, BASE_D>(
                    setup, base_pows, &a_scales, &b_scales, &d_scales,
                )
            }
        )
        .map(Some)
    }

    fn evaluate_divisible_direct_typed<F, const BASE_D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        base_pows: &[E],
        a_scales: &AlphaChunkScales<E>,
        b_scales: &AlphaChunkScales<E>,
        d_scales: &AlphaChunkScales<E>,
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
        let required = self.required_divisible_base(
            a_scales.scales.len(),
            b_scales.scales.len(),
            d_scales.scales.len(),
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
            acc += group.evaluate_divisible_packed_direct::<F, BASE_D>(
                &setup_view,
                base_pows,
                a_scales,
                b_scales,
                d_scales,
                self.d_rows,
                self.d_physical_cols,
            )?;
        }
        Ok(acc)
    }

    fn required_divisible_base(
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
            .ok_or_else(|| AkitaError::InvalidSetup("setup D base footprint overflow".into()))?;
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
                AkitaError::InvalidSetup("setup B base footprint overflow".into())
            })?;
            let a_base_required = a_required.checked_mul(a_ratio).ok_or_else(|| {
                AkitaError::InvalidSetup("setup A base footprint overflow".into())
            })?;
            required = required.max(b_base_required).max(a_base_required);
        }
        Ok(required)
    }

    fn evaluate_uniform_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows: &[E],
        ring_d: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_d,
            |D| self.evaluate_uniform_direct_typed::<F, D>(setup, alpha_pows)
        )
    }

    fn evaluate_uniform_direct_typed<F, const D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        if alpha_pows.len() != D {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: alpha_pows.len(),
            });
        }
        let required = self.required()?;
        let setup_len = setup.shared_matrix().total_ring_elements_at::<D>()?;
        if required > setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }
        let setup_view = setup.shared_matrix().ring_view::<D>(1, setup_len)?;
        let mut acc = E::zero();
        for group in &self.groups {
            acc += group.evaluate_uniform_packed_direct_typed(
                &setup_view,
                alpha_pows,
                self.d_rows,
                self.d_physical_cols,
            )?;
        }
        Ok(acc)
    }

    fn evaluate_packed_direct<F>(
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
        dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Inner), F, d_a, |D_A| {
            dispatch_for_field!(ProtocolDispatchSlot::Role(RingRole::Outer), F, d_b, |D_B| {
                dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Opening),
                    F,
                    d_d,
                    |D_D| {
                        self.evaluate_packed_direct_typed::<F, D_A, D_B, D_D>(
                            setup,
                            alpha_pows_a,
                            alpha_pows_b,
                            alpha_pows_d,
                        )
                    }
                )
            })
        })
    }

    fn evaluate_packed_direct_typed<F, const D_A: usize, const D_B: usize, const D_D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        if alpha_pows_a.len() != D_A {
            return Err(AkitaError::InvalidSize {
                expected: D_A,
                actual: alpha_pows_a.len(),
            });
        }
        if alpha_pows_b.len() != D_B {
            return Err(AkitaError::InvalidSize {
                expected: D_B,
                actual: alpha_pows_b.len(),
            });
        }
        if alpha_pows_d.len() != D_D {
            return Err(AkitaError::InvalidSize {
                expected: D_D,
                actual: alpha_pows_d.len(),
            });
        }
        let mut acc = E::zero();
        for group in &self.groups {
            acc += group.evaluate_packed_direct_typed::<F, D_A, D_B, D_D>(
                setup,
                alpha_pows_a,
                alpha_pows_b,
                alpha_pows_d,
                self.d_rows,
                self.d_physical_cols,
            )?;
        }
        Ok(acc)
    }
}
