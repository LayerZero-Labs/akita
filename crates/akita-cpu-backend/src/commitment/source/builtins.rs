use super::compiler::{checked_logical_len, validate_plan_extents};
use super::*;
use akita_params::SUPPORTED_COMMITMENT_RING_DIMS;

impl<F: Field> DenseCoefficientSource<F> for crate::DensePoly<F> {
    fn coefficients(&self) -> &[F] {
        self.field_coeffs()
    }

    fn nonzero_coefficient_len(&self) -> usize {
        self.nonzero_coeff_len()
    }
}

impl<F: Field> CommitmentSource<F> for crate::DensePoly<F> {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        let num_vars = crate::opaque::RootPolyMeta::<F>::num_vars(self);
        CommitSourceDescriptor::new(
            num_vars,
            checked_logical_len(num_vars)?
                .max(SUPPORTED_COMMITMENT_RING_DIMS[SUPPORTED_COMMITMENT_RING_DIMS.len() - 1]),
            checked_logical_len(num_vars)?,
            CommitSourceClass::Dense,
            "akita_dense",
        )
    }

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding,
    {
        Ok(crate::opaque::centered_reach_of_field_coeffs(
            &self.field_coeffs()[..self.nonzero_coeff_len()],
            modulus,
            centering_threshold,
        ))
    }

    fn available_polynomial_types(
        &self,
        plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        validate_plan_extents(&<Self as CommitmentSource<F>>::descriptor(self)?, plan)?;
        AvailablePolynomialTypes::new(vec![PolynomialType::Dense(DenseType::Coefficients)])
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        validate_plan_extents(&self.descriptor()?, plan)?;
        match selected.polynomial_type() {
            PolynomialType::Dense(DenseType::Coefficients) => Ok(PolynomialRepresentation::Dense(
                DenseRepresentation::Coefficients(self),
            )),
            _ => Err(AkitaError::InvalidInput(
                "dense source received an unadvertised representation selection".into(),
            )),
        }
    }
}

macro_rules! impl_onehot_commitment_source {
    ($index:ty, $width:ident, $variant:ident) => {
        impl<F: Field> CommitmentSource<F> for crate::OneHotPoly<F, $index> {
            fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
                let num_vars = crate::opaque::RootPolyMeta::<F>::num_vars(self);
                let logical_len = checked_logical_len(num_vars)?;
                CommitSourceDescriptor::new(
                    num_vars,
                    logical_len,
                    logical_len,
                    CommitSourceClass::OneHot {
                        chunk_size: self.onehot_k(),
                    },
                    "akita_onehot",
                )
            }

            fn committed_centered_reach(
                &self,
                _modulus: u128,
                _centering_threshold: u128,
            ) -> Result<(u128, u128), AkitaError>
            where
                F: CanonicalEncoding,
            {
                Ok((0, 1))
            }

            fn available_polynomial_types(
                &self,
                plan: &CommitInnerPlan,
            ) -> Result<AvailablePolynomialTypes, AkitaError> {
                validate_plan_extents(&self.descriptor()?, plan)?;
                self.validate_ring_dimension(plan.ring_dimension)?;
                AvailablePolynomialTypes::new(vec![PolynomialType::OneHot(OneHotType::new(
                    self.onehot_k(),
                    OneHotIndexWidth::$width,
                )?)])
            }

            fn represent_as(
                &self,
                selected: PolynomialTypeSelection,
                plan: &CommitInnerPlan,
            ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
                let available = self.available_polynomial_types(plan)?;
                if available.as_slice().first().copied() != Some(selected.polynomial_type()) {
                    return Err(AkitaError::InvalidInput(
                        "one-hot source received an unadvertised representation selection".into(),
                    ));
                }
                Ok(PolynomialRepresentation::OneHot(OneHotRepresentation {
                    positions: UnitPositionSlice::$variant(self.indices()),
                    chunk_size: self.onehot_k(),
                    num_vars: crate::opaque::RootPolyMeta::<F>::num_vars(self),
                }))
            }
        }
    };
}

impl_onehot_commitment_source!(u8, U8, U8);
impl_onehot_commitment_source!(u16, U16, U16);
impl_onehot_commitment_source!(u32, U32, U32);
impl_onehot_commitment_source!(usize, Usize, Usize);

impl<F: Field> CommitmentSource<F> for crate::opaque::RecursiveWitnessFlat {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        let logical_len = self
            .live_coeff_len()
            .max(1)
            .checked_next_power_of_two()
            .ok_or_else(|| {
                AkitaError::InvalidInput("recursive witness logical extent overflows usize".into())
            })?;
        let num_vars = logical_len.trailing_zeros() as usize;
        CommitSourceDescriptor::new(
            num_vars,
            self.commitment_physical_len()?,
            self.live_coeff_len(),
            CommitSourceClass::ShortNorm,
            "akita_recursive_packed",
        )
    }

    fn committed_centered_reach(
        &self,
        _modulus: u128,
        _centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding,
    {
        let (_, _, negative_abs_max, positive_max) = self.packed_representation_parts();
        Ok((u128::from(negative_abs_max), u128::from(positive_max)))
    }

    fn available_polynomial_types(
        &self,
        plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        validate_plan_extents(&<Self as CommitmentSource<F>>::descriptor(self)?, plan)?;
        let (_, signed_bit_width, _, _) = self.packed_representation_parts();
        AvailablePolynomialTypes::new(vec![PolynomialType::ShortNorm(ShortNormType::new(
            signed_bit_width,
        )?)])
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        let available = <Self as CommitmentSource<F>>::available_polynomial_types(self, plan)?;
        if available.as_slice().first().copied() != Some(selected.polynomial_type()) {
            return Err(AkitaError::InvalidInput(
                "short-norm source received an unadvertised representation selection".into(),
            ));
        }
        let physical_coefficient_len =
            <Self as CommitmentSource<F>>::descriptor(self)?.total_coefficient_len();
        let representation = ShortNormRepresentation::from_validated_packed(
            self.packed_digits(),
            self.live_coeff_len(),
            physical_coefficient_len,
        )?;
        Ok(PolynomialRepresentation::ShortNorm(representation))
    }
}
