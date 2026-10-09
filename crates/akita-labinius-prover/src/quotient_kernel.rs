//! Conjugate-ring transforms for the unreduced A-relation quotient.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

use akita_algebra::{
    embed_scalar, MinusTrinomial, PlusTrinomial, SmoothFftField, TrinomialModulus, TrinomialNtt,
    TrinomialNttDomain, TrinomialRing,
};
use akita_challenges::BinaryChallenge;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    commitment::centered_coefficient, endpoint::pack_response, lowered::ARelationAuxiliary,
    profile::BinaryClearSetup, source::challenge_scalar, BinaryClearCommitment,
};
use akita_params::sis::labinius::LabiniusSignedDigitRange;
use jolt_field::WithPacking;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::PreparedCommitMatrix;

mod sealed {
    pub trait Sealed {}
    impl Sealed for akita_algebra::MinusTrinomial {}
    impl Sealed for akita_algebra::PlusTrinomial {}
}

/// The two supported trinomials, paired with their conjugates.
pub trait ConjugateModulus: TrinomialModulus + sealed::Sealed + Send + Sync {
    /// Modulus with the opposite middle coefficient.
    type Conjugate: TrinomialModulus + Send + Sync;
}
impl ConjugateModulus for MinusTrinomial {
    type Conjugate = PlusTrinomial;
}
impl ConjugateModulus for PlusTrinomial {
    type Conjugate = MinusTrinomial;
}

type NttPair<F, const D: usize, M> = (
    TrinomialNtt<F, D, M>,
    TrinomialNtt<F, D, <M as ConjugateModulus>::Conjugate>,
);
type ColumnWork<'a, F, const D: usize, M> = (
    &'a mut [NttPair<F, D, M>],
    (&'a TrinomialRing<F, D, M>, &'a [TrinomialRing<F, D, M>]),
);

/// Reusable conjugate-ring transform of A, bound to the matrix it came from.
pub struct PreparedQuotientMatrix<F, const D: usize, M: ConjugateModulus> {
    domain: TrinomialNttDomain<F, D, M::Conjugate>,
    matrix: Vec<TrinomialNtt<F, D, M::Conjugate>>,
    n_a: usize,
    m: usize,
    digest: [u8; 32],
}

impl<F: SmoothFftField, const D: usize, M: ConjugateModulus> PreparedQuotientMatrix<F, D, M> {
    /// Transform every matrix entry once in the conjugate ring.
    pub fn prepare(setup: &BinaryClearSetup<F, D, M>) -> Result<Self, AkitaError> {
        let domain = TrinomialNttDomain::new()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let mut matrix = Vec::new();
        matrix
            .try_reserve_exact(setup.matrix().len())
            .map_err(|_| AkitaError::InvalidSetup("conjugate matrix allocation failed".into()))?;
        let mut workspace = domain.workspace();
        for element in setup.matrix() {
            let conjugate = TrinomialRing::from_coefficients(*element.coefficients())
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
            matrix.push(domain.forward_with_workspace(&conjugate, &mut workspace));
        }
        Ok(Self {
            domain,
            matrix,
            n_a: setup.n_a(),
            m: setup.m(),
            digest: *setup.matrix_view_digest(),
        })
    }

    /// Bytes of transformed matrix field coefficients. Excludes plans and Vec
    /// metadata; the transform API does not expose plan or workspace heap sizes.
    pub fn prepared_bytes(&self) -> Result<usize, AkitaError> {
        checked::product([self.matrix.len(), D, size_of::<F>()])
            .ok_or_else(|| AkitaError::InvalidSetup("prepared payload size overflow".into()))
    }

    /// Reject a setup whose matrix shape or digest differs from the cached one.
    pub(crate) fn check_setup(&self, setup: &BinaryClearSetup<F, D, M>) -> Result<(), AkitaError> {
        if self.n_a != setup.n_a()
            || self.m != setup.m()
            || self.digest != *setup.matrix_view_digest()
        {
            return Err(AkitaError::InvalidSetup(
                "prepared quotient matrix does not match setup".into(),
            ));
        }
        Ok(())
    }
}

/// Recover QA and optional integer KA using conjugate-ring transforms.
///
/// Count, interval, packing, and challenge validation follow the reference's
/// order. Both cache bindings are checked immediately afterwards, before any
/// remainder check, and mismatches return `InvalidSetup`. Column transforms
/// use per-thread workspaces under `parallel`; reduction follows column order.
/// A zero slot residual is exact because each domain transform is a bijection.
pub fn a_relation_quotients<F, const D: usize, M>(
    prepared_commit: &PreparedCommitMatrix<F, D, M>,
    prepared_quotient: &PreparedQuotientMatrix<F, D, M>,
    setup: &BinaryClearSetup<F, D, M>,
    commitment: &BinaryClearCommitment<F, D, M>,
    fold_challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
    a_carry_range: Option<LabiniusSignedDigitRange>,
) -> Result<ARelationAuxiliary<F>, AkitaError>
where
    F: SmoothFftField + WithPacking,
    M: ConjugateModulus,
{
    let image_count =
        checked::product([setup.columns(), setup.n_a()]).ok_or(AkitaError::InvalidProof)?;
    if commitment.images.len() != image_count
        || fold_challenges.len() != setup.columns()
        || response.len() != setup.scalar_rows()
        || response
            .iter()
            .flatten()
            .any(|&v| v < setup.lower() || v > setup.upper())
    {
        return Err(AkitaError::InvalidProof);
    }
    let packed = pack_response(setup, response)?;
    let mut embedded = Vec::new();
    embedded
        .try_reserve_exact(setup.columns())
        .map_err(|_| AkitaError::InvalidProof)?;
    for challenge in fold_challenges {
        challenge
            .validate(setup.profile())
            .map_err(|_| AkitaError::InvalidProof)?;
        embedded.push(
            embed_scalar::<F, 162, D, M>(&challenge_scalar(challenge)?)
                .map_err(|_| AkitaError::InvalidProof)?,
        );
    }
    // Preserve the reference's degree-extent validation before binding checks.
    checked::product([2, D])
        .and_then(|length| length.checked_sub(1))
        .ok_or(AkitaError::InvalidProof)?;
    prepared_quotient.check_setup(setup)?;
    prepared_commit.check_setup(setup)?;
    let small_modulus = setup.commitment_modulus().small_modulus();
    if small_modulus.is_some() != a_carry_range.is_some() {
        return Err(AkitaError::InvalidSetup(
            "A-carry range does not match commitment modulus".into(),
        ));
    }
    let domain = prepared_commit.domain();
    let conjugate = &prepared_quotient.domain;
    let mut lhs = Vec::new();
    lhs.try_reserve_exact(setup.n_a())
        .map_err(|_| AkitaError::InvalidProof)?;
    lhs.resize_with(setup.n_a(), || (domain.zero_ntt(), conjugate.zero_ntt()));
    let mut workspace = domain.workspace();
    let mut conjugate_workspace = conjugate.workspace();
    let mut transformed = domain.zero_ntt();
    let mut conjugate_transformed = conjugate.zero_ntt();
    for (element, value) in packed.iter().enumerate() {
        domain.forward_into_with_workspace(value, &mut transformed, &mut workspace);
        let value_conjugate = TrinomialRing::from_coefficients(*value.coefficients())
            .map_err(|_| AkitaError::InvalidProof)?;
        conjugate.forward_into_with_workspace(
            &value_conjugate,
            &mut conjugate_transformed,
            &mut conjugate_workspace,
        );
        for ((accumulator, conjugate_accumulator), (row, conjugate_row)) in lhs.iter_mut().zip(
            prepared_commit
                .matrix_ntt()
                .chunks_exact(setup.m())
                .zip(prepared_quotient.matrix.chunks_exact(setup.m())),
        ) {
            accumulator.add_assign_pointwise_mul(
                row.get(element).ok_or(AkitaError::InvalidProof)?,
                &transformed,
            );
            conjugate_accumulator.add_assign_pointwise_mul(
                conjugate_row.get(element).ok_or(AkitaError::InvalidProof)?,
                &conjugate_transformed,
            );
        }
    }
    let mut rhs = Vec::new();
    rhs.try_reserve_exact(image_count)
        .map_err(|_| AkitaError::InvalidProof)?;
    rhs.resize_with(image_count, || (domain.zero_ntt(), conjugate.zero_ntt()));
    let column_work = |workspaces: &mut (_, _),
                       (output, (challenge, images)): ColumnWork<'_, F, D, M>|
     -> Result<(), AkitaError> {
        let challenge_ntt = domain.forward_with_workspace(challenge, &mut workspaces.0);
        let challenge_conjugate = TrinomialRing::from_coefficients(*challenge.coefficients())
            .map_err(|_| AkitaError::InvalidProof)?;
        let challenge_conjugate_ntt =
            conjugate.forward_with_workspace(&challenge_conjugate, &mut workspaces.1);
        for ((result, conjugate_result), image) in output.iter_mut().zip(images) {
            let image_ntt = domain.forward_with_workspace(image, &mut workspaces.0);
            *result = challenge_ntt.pointwise_mul(&image_ntt);
            let image_conjugate = TrinomialRing::from_coefficients(*image.coefficients())
                .map_err(|_| AkitaError::InvalidProof)?;
            let image_conjugate_ntt =
                conjugate.forward_with_workspace(&image_conjugate, &mut workspaces.1);
            *conjugate_result = challenge_conjugate_ntt.pointwise_mul(&image_conjugate_ntt);
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    rhs.par_chunks_mut(setup.n_a())
        .zip(
            embedded
                .par_iter()
                .zip(commitment.images.par_chunks(setup.n_a())),
        )
        .try_for_each_init(|| (domain.workspace(), conjugate.workspace()), column_work)?;
    #[cfg(not(feature = "parallel"))]
    {
        let mut column_workspaces = (domain.workspace(), conjugate.workspace());
        for item in rhs
            .chunks_mut(setup.n_a())
            .zip(embedded.iter().zip(commitment.images.chunks(setup.n_a())))
        {
            column_work(&mut column_workspaces, item)?;
        }
    }
    let half = checked::exact_div(D, 2).ok_or(AkitaError::InvalidProof)?;
    let quotient_len = D.checked_sub(1).ok_or(AkitaError::InvalidProof)?;
    let two_inverse = F::from_u64(2).inverse().ok_or(AkitaError::InvalidProof)?;
    let mut quotients = Vec::new();
    quotients
        .try_reserve_exact(setup.n_a())
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut carry = Vec::new();
    if small_modulus.is_some() {
        carry
            .try_reserve_exact(checked::product([setup.n_a(), D]).ok_or(AkitaError::InvalidProof)?)
            .map_err(|_| AkitaError::InvalidProof)?;
    }
    for (row, (accumulator, conjugate_accumulator)) in lhs.iter().enumerate() {
        let mut remainder_slots = *accumulator.slots();
        let mut remainder = small_modulus.map(|_| {
            domain
                .inverse_with_workspace(accumulator, &mut workspace)
                .into_coefficients()
        });
        let mut residual = conjugate
            .inverse_with_workspace(conjugate_accumulator, &mut conjugate_workspace)
            .into_coefficients();
        for column in rhs.chunks_exact(setup.n_a()) {
            let (term, conjugate_term) = column.get(row).ok_or(AkitaError::InvalidProof)?;
            for (destination, &value) in remainder_slots.iter_mut().zip(term.slots()) {
                *destination -= value;
            }
            if let Some(remainder) = &mut remainder {
                let coefficients = domain
                    .inverse_with_workspace(term, &mut workspace)
                    .into_coefficients();
                for (destination, value) in remainder.iter_mut().zip(coefficients) {
                    *destination -= value;
                }
            }
            let coefficients = conjugate
                .inverse_with_workspace(conjugate_term, &mut conjugate_workspace)
                .into_coefficients();
            for (destination, value) in residual.iter_mut().zip(coefficients) {
                *destination -= value;
            }
        }
        if let (Some(q0), Some(range), Some(remainder)) = (small_modulus, a_carry_range, remainder)
        {
            // Admission ensures 6*H_A < P, so the honest remainder bounded by
            // 3*H_A has a unique centered lift. Division here is over Z, not F_P.
            let (lower, upper) = range.interval();
            let q0 = i128::from(q0);
            for (&coefficient, conjugate_residual) in remainder.iter().zip(&mut residual) {
                let lifted = centered_coefficient(coefficient)?;
                if lifted % q0 != 0 {
                    return Err(AkitaError::InvalidProof);
                }
                let value = lifted / q0;
                if value < lower || value > upper {
                    return Err(AkitaError::InvalidProof);
                }
                carry.push(value);
                // The remainder has degree below D, hence the same coefficients
                // in both rings. Subtract it before recovering the field quotient.
                *conjugate_residual -= coefficient;
            }
        } else if remainder_slots.iter().any(|v| !v.is_zero()) {
            return Err(AkitaError::InvalidProof);
        }
        let mut quotient = Vec::new();
        quotient
            .try_reserve_exact(D)
            .map_err(|_| AkitaError::InvalidProof)?;
        let low = residual.get(..half).ok_or(AkitaError::InvalidProof)?;
        let high = residual.get(half..).ok_or(AkitaError::InvalidProof)?;
        // 2 Q = (L + mu H) - mu X^(D/2) L in the conjugate ring.
        for (&l, &h) in low.iter().zip(high) {
            quotient.push(
                (if M::MIDDLE_COEFFICIENT > 0 {
                    l + h
                } else {
                    l - h
                }) * two_inverse,
            );
        }
        for &l in low {
            quotient.push((if M::MIDDLE_COEFFICIENT > 0 { -l } else { l }) * two_inverse);
        }
        if quotient.last().is_none_or(|v| !v.is_zero()) {
            return Err(AkitaError::InvalidProof);
        }
        quotient.truncate(quotient_len);
        quotients.push(quotient);
    }
    Ok(ARelationAuxiliary { quotients, carry })
}
