//! Ordinary point records over the same committed polynomial slots.

use super::PolynomialGroupClaims;

/// One point and all polynomial evaluations, in commitment slot order.
#[derive(Clone, Copy, Debug)]
pub struct OrdinaryPointEvaluation<'a, E> {
    /// Complete multilinear point in the source's axis order.
    pub point: &'a [E],
    /// Exactly one value per committed polynomial.
    pub evaluations: &'a [E],
}

impl<F: Clone, C> PolynomialGroupClaims<'_, F, C> {
    /// Number of points authenticated for each polynomial slot.
    pub fn num_points(&self) -> usize {
        1 + usize::from(self.second_point.is_some())
    }

    /// Ordered ordinary readings; binary claims retain their carrier record.
    pub fn point_evaluations(&self) -> impl Iterator<Item = OrdinaryPointEvaluation<'_, F>> {
        std::iter::once(OrdinaryPointEvaluation {
            point: self.point(),
            evaluations: self.evaluations(),
        })
        .chain(self.second_point.iter().map(|(point, evaluations)| {
            OrdinaryPointEvaluation {
                point: point.as_ref(),
                evaluations,
            }
        }))
    }
}

#[cfg(feature = "labinius-trinomial")]
impl<'a, E, F> PolynomialGroupClaims<'a, E, crate::CommittedGroup<F>>
where
    F: jolt_field::Field + jolt_field::CanonicalEncoding,
    E: jolt_field::ExtField<F>,
{
    /// Authenticate every ordinary polynomial at two distinct E64 points.
    ///
    /// This shape requires a P64 binary root in the same opening. The first
    /// point remains the layout carrier; both records are transcript-bound.
    /// Use `map_commitment` to borrow the frozen claim for verification.
    pub fn new_two_points(
        points: [Vec<E>; 2],
        evaluations: [Vec<E>; 2],
        commitment: crate::CommittedGroup<F>,
    ) -> Result<Self, akita_error::AkitaError> {
        use akita_error::AkitaError;
        use akita_params::binary_setup::BinaryFieldProfile;

        BinaryFieldProfile::P64.check_field_pair(akita_params::field_modulus::<F>()?, E::DEGREE)?;
        let layout = commitment.profile().group;
        if commitment.kind() != &crate::proof::commitment::CommittedGroupKind::Ordinary
            || points[0] == points[1]
            || points.iter().any(|point| point.len() != layout.num_vars())
            || evaluations
                .iter()
                .any(|values| values.len() != layout.num_polynomials() || values.is_empty())
        {
            return Err(AkitaError::InvalidInput(
                "two-point ordinary claim requires distinct points and every polynomial value"
                    .into(),
            ));
        }
        let [first, second] = points;
        let [first_values, second_values] = evaluations;
        let mut claims = Self::new(first, first_values, commitment)?;
        claims.second_point = Some((second.into(), second_values));
        Ok(claims)
    }
}
