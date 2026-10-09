//! Trace-form multiplication adjoints and the independent monomial reference.
use akita_algebra::{
    fft::SmoothFftField,
    ring::{TrinomialModulus, TrinomialNttDomain, TrinomialRing},
};
use akita_error::{checked, AkitaError};
use akita_types::{RelationPolynomial, RelationPolynomialKind, TrinomialSign};
use jolt_field::Field;

use super::{powers, zero_vec};

pub(crate) fn trace_inverse_scale<F: Field>(degree: usize) -> Result<F, AkitaError> {
    if degree == 0 || !degree.is_multiple_of(2) {
        return Err(AkitaError::InvalidSetup("invalid trace-form degree".into()));
    }
    let denominator = checked::product([3, degree / 2])
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| AkitaError::InvalidSetup("trace-form denominator overflow".into()))?;
    F::from_u64(denominator).inverse().ok_or_else(|| {
        AkitaError::InvalidSetup("trinomial trace form is singular in this field".into())
    })
}

/// Apply `G[s,t] = Tr(Y^(s+t))` to a coefficient vector, for either trinomial.
/// Both slices must have the same positive even length; output is overwritten.
pub fn trace_gram_into<F: Field, M: TrinomialModulus>(
    input: &[F],
    output: &mut [F],
) -> Result<(), AkitaError> {
    let degree = input.len();
    if output.len() != degree || degree == 0 || !degree.is_multiple_of(2) {
        return Err(AkitaError::InvalidInput(
            "trace-form vector length mismatch".into(),
        ));
    }
    let h = degree / 2;
    let hf = F::from_u64(u64::try_from(h).map_err(|_| AkitaError::InvalidProof)?);
    let c = if M::MIDDLE_COEFFICIENT == -1 {
        F::one()
    } else {
        -F::one()
    };
    let get = |index| input.get(index).copied().ok_or(AkitaError::InvalidProof);
    *output.get_mut(0).ok_or(AkitaError::InvalidProof)? =
        hf * (F::from_u64(2) * get(0)? + c * get(h)?);
    *output.get_mut(h).ok_or(AkitaError::InvalidProof)? = hf * (c * get(0)? - get(h)?);
    for r in 1..h {
        let a = get(h - r)?;
        let b = get(degree - r)?;
        *output.get_mut(r).ok_or(AkitaError::InvalidProof)? = hf * (c * a - b);
        *output.get_mut(h + r).ok_or(AkitaError::InvalidProof)? =
            -hf * (a + F::from_u64(2) * c * b);
    }
    Ok(())
}

/// Apply `G^-1` in linear work. A noninvertible `3*(D/2)` is `InvalidSetup`.
pub fn trace_gram_inverse<F: Field, M: TrinomialModulus>(
    input: &[F],
) -> Result<Vec<F>, AkitaError> {
    let degree = input.len();
    let z = trace_inverse_scale::<F>(degree)?;
    let h = degree / 2;
    let c = if M::MIDDLE_COEFFICIENT == -1 {
        F::one()
    } else {
        -F::one()
    };
    let mut output = zero_vec(degree)?;
    let get = |index| input.get(index).copied().ok_or(AkitaError::InvalidProof);
    *output.get_mut(0).ok_or(AkitaError::InvalidProof)? = (get(0)? + c * get(h)?) * z;
    *output.get_mut(h).ok_or(AkitaError::InvalidProof)? =
        (c * get(0)? - F::from_u64(2) * get(h)?) * z;
    for r in 1..h {
        *output.get_mut(h - r).ok_or(AkitaError::InvalidProof)? =
            (F::from_u64(2) * c * get(r)? - get(h + r)?) * z;
        *output.get_mut(degree - r).ok_or(AkitaError::InvalidProof)? =
            -(get(r)? + c * get(h + r)?) * z;
    }
    Ok(output)
}

/// Compute the coefficient-dot-product adjoint `G(a * G^-1 u)` by transforms.
/// The input functional must have exactly `D` coefficients.
pub fn multiplication_adjoint<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    a: &TrinomialRing<F, D, M>,
    u: &[F],
) -> Result<Vec<F>, AkitaError> {
    if u.len() != D {
        return Err(AkitaError::InvalidInput(
            "adjoint functional degree mismatch".into(),
        ));
    }
    let inverse = trace_gram_inverse::<F, M>(u)?;
    let v = TrinomialRing::from_coefficients(
        inverse
            .as_slice()
            .try_into()
            .map_err(|_| AkitaError::InvalidProof)?,
    )
    .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
    let domain = TrinomialNttDomain::<F, D, M>::new()
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
    let product = domain.multiply(a, &v);
    let mut output = zero_vec(D)?;
    trace_gram_into::<F, M>(product.coefficients(), &mut output)?;
    Ok(output)
}

/// Independent definition: `(Y^n mod Phi)(alpha)` for `0 <= n <= 2D-2`.
/// The recurrence reduces monomials directly, without products or transforms.
pub fn remainder_evaluations<F: Field>(
    polynomial: RelationPolynomial,
    alpha: F,
) -> Result<Vec<F>, AkitaError> {
    let c = match polynomial.kind() {
        RelationPolynomialKind::Trinomial(TrinomialSign::Minus) => F::one(),
        RelationPolynomialKind::Trinomial(TrinomialSign::Plus) => -F::one(),
        _ => {
            return Err(AkitaError::InvalidInput(
                "remainder weights require a trinomial".into(),
            ))
        }
    };
    let degree = polynomial.degree();
    let len = checked::mul_add(2, degree, 0)
        .and_then(|value| value.checked_sub(1))
        .ok_or(AkitaError::InvalidProof)?;
    let mut rho = powers(alpha, len)?;
    for n in degree..len {
        let value = c * *rho.get(n - degree / 2).ok_or(AkitaError::InvalidProof)?
            - *rho.get(n - degree).ok_or(AkitaError::InvalidProof)?;
        *rho.get_mut(n).ok_or(AkitaError::InvalidProof)? = value;
    }
    Ok(rho)
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use super::*;
    use akita_algebra::ring::{MinusTrinomial, PlusTrinomial};
    use jolt_field::{One, Prime128OffsetA7F7, Prime31Offset19, Ring, Zero};
    type F = Prime128OffsetA7F7;

    fn random(state: &mut u64) -> F {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        F::from_u64(*state)
    }
    fn check<const D: usize, M: TrinomialModulus>() {
        let mut state = 0x59ec_3f19_18ab_031d;
        for _ in 0..2 {
            let a = TrinomialRing::<F, D, M>::from_coefficients(std::array::from_fn(|_| {
                random(&mut state)
            }))
            .unwrap();
            let y = TrinomialRing::<F, D, M>::from_coefficients(std::array::from_fn(|_| {
                random(&mut state)
            }))
            .unwrap();
            let u: Vec<F> = (0..D).map(|_| random(&mut state)).collect();
            let mut restored = vec![F::zero(); D];
            trace_gram_into::<F, M>(&trace_gram_inverse::<F, M>(&u).unwrap(), &mut restored)
                .unwrap();
            assert_eq!(restored, u);
            trace_gram_into::<F, M>(&u, &mut restored).unwrap();
            assert_eq!(trace_gram_inverse::<F, M>(&restored).unwrap(), u);
            let product = a.schoolbook_mul(&y).unwrap();
            let weights = multiplication_adjoint(&a, &u).unwrap();
            let dot =
                |x: &[F], z: &[F]| x.iter().zip(z).fold(F::zero(), |sum, (&a, &b)| sum + a * b);
            assert_eq!(
                dot(&u, product.coefficients()),
                dot(&weights, y.coefficients())
            );
            let alpha = random(&mut state);
            let polynomial = if M::MIDDLE_COEFFICIENT == -1 {
                RelationPolynomial::minus_trinomial(D).unwrap()
            } else {
                RelationPolynomial::plus_trinomial(D).unwrap()
            };
            let rho = remainder_evaluations(polynomial, alpha).unwrap();
            let point_weights = multiplication_adjoint(&a, &powers(alpha, D).unwrap()).unwrap();
            for t in 0..D {
                // Shift and descending reduction are independent of the trace maps.
                let mut shifted = vec![F::zero(); 2 * D - 1];
                shifted[t..t + D].copy_from_slice(a.coefficients());
                for n in (D..shifted.len()).rev() {
                    let leading = shifted[n];
                    shifted[n] = F::zero();
                    shifted[n - D] -= leading;
                    if M::MIDDLE_COEFFICIENT == -1 {
                        shifted[n - D / 2] += leading;
                    } else {
                        shifted[n - D / 2] -= leading;
                    }
                }
                let reduced = &shifted[..D];
                assert_eq!(weights[t], dot(&u, reduced));
                let direct = super::super::horner(reduced, alpha);
                let recurrence = a
                    .coefficients()
                    .iter()
                    .enumerate()
                    .fold(F::zero(), |sum, (s, &coefficient)| {
                        sum + coefficient * rho[s + t]
                    });
                assert_eq!(point_weights[t], direct);
                assert_eq!(recurrence, direct);
            }
        }
    }
    #[test]
    fn trace_maps_and_schoolbook_adjoint_agree_plus_degree_6() {
        check::<6, PlusTrinomial>();
    }
    #[test]
    fn trace_maps_and_schoolbook_adjoint_agree_minus_degree_6() {
        check::<6, MinusTrinomial>();
    }
    #[test]
    fn trace_maps_and_schoolbook_adjoint_agree_plus_degree_648() {
        check::<648, PlusTrinomial>();
    }
    #[test]
    fn trace_maps_and_schoolbook_adjoint_agree_minus_degree_648() {
        check::<648, MinusTrinomial>();
    }
    #[test]
    fn malformed_trace_and_adjoint_vectors_reject_with_typed_errors() {
        let mut output = [F::zero(); 6];
        assert!(matches!(
            trace_gram_into::<F, MinusTrinomial>(&[], &mut []),
            Err(AkitaError::InvalidInput(_))
        ));
        assert!(trace_gram_into::<F, MinusTrinomial>(&[F::one(); 5], &mut output).is_err());
        assert!(trace_gram_into::<F, MinusTrinomial>(&[F::one(); 6], &mut output[..5]).is_err());
        assert!(matches!(
            trace_gram_inverse::<F, MinusTrinomial>(&[F::one(); 5]),
            Err(AkitaError::InvalidSetup(_))
        ));
        assert!(matches!(
            trace_inverse_scale::<F>(0),
            Err(AkitaError::InvalidSetup(_))
        ));
        assert!(matches!(
            trace_inverse_scale::<F>(usize::MAX - 1),
            Err(AkitaError::InvalidSetup(_))
        ));
        // Validate singularity without allocating a degree-sized vector: h=P
        // makes 3h zero in this already-available small prime field.
        let singular_degree = usize::try_from(2 * ((1u64 << 31) - 19)).unwrap();
        assert!(matches!(
            trace_inverse_scale::<Prime31Offset19>(singular_degree),
            Err(AkitaError::InvalidSetup(_))
        ));
        let a = TrinomialRing::<F, 6, MinusTrinomial>::one().unwrap();
        assert!(multiplication_adjoint(&a, &[F::one(); 5]).is_err());
    }
}
