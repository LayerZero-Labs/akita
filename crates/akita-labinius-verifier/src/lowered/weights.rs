use super::{
    horner, multiplication_adjoint, signed_field, zero_vec, LoweredPublic, LoweredRootLayout,
};
use crate::BinaryClearSetup;
use akita_algebra::{
    fft::SmoothFftField,
    offset_eq::eq_eval_at_index,
    ring::{MinusTrinomial, PlusTrinomial, TrinomialModulus, TrinomialNttDomain, TrinomialRing},
};
use akita_error::{checked, AkitaError};
use akita_types::{RelationPolynomialKind, TrinomialSign};

/// Dense reference response weights, using direct reduced-monomial evaluation.
/// Coefficient tails have zero weight. No trace map or transform is used.
pub fn witness_weights_dense<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    setup: &BinaryClearSetup<F, D, M>,
) -> Result<Vec<F>, AkitaError> {
    public.validate_layout(layout)?;
    layout.validate_setup(setup)?;
    let mut weights = zero_vec(layout.witness_len())?;
    for j in 0..layout.m() {
        for t in 0..D {
            let mut coefficient = public.parity_coefficient_weight(j, t)?;
            for (i, &gamma) in public.gamma_powers.iter().take(layout.n_a()).enumerate() {
                let index = checked::mul_add(i, layout.m(), j).ok_or(AkitaError::InvalidProof)?;
                let a = setup.matrix().get(index).ok_or(AkitaError::InvalidProof)?;
                coefficient +=
                    gamma * shifted_evaluation(a.coefficients(), t, &public.remainder_evaluations)?;
            }
            for (l, &digit) in public.digit_powers.iter().enumerate() {
                let address = layout.response_layout().address(j, l, t)?;
                *weights.get_mut(address).ok_or(AkitaError::InvalidProof)? = digit * coefficient;
            }
        }
    }
    Ok(weights)
}

fn shifted_evaluation<F: SmoothFftField>(a: &[F], t: usize, rho: &[F]) -> Result<F, AkitaError> {
    let mut value = F::zero();
    for (s, &coefficient) in a.iter().enumerate() {
        if !coefficient.is_zero() {
            let index = checked::sum([s, t]).ok_or(AkitaError::InvalidProof)?;
            value += coefficient * *rho.get(index).ok_or(AkitaError::InvalidProof)?;
        }
    }
    Ok(value)
}

/// Dense reference image weights from the same monomial remainder definition.
/// The challenge's zero coefficients are skipped; tails and entry padding are zero.
pub fn image_weights_dense<F: SmoothFftField>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
) -> Result<Vec<F>, AkitaError> {
    public.validate_layout(layout)?;
    let mut weights = zero_vec(layout.image_len())?;
    for e in 0..layout.image_count() {
        let gamma = *public
            .gamma_powers
            .get(e % layout.n_a())
            .ok_or(AkitaError::InvalidProof)?;
        let challenge = public
            .embedded_challenges
            .get(e / layout.n_a())
            .ok_or(AkitaError::InvalidProof)?;
        for t in 0..layout.degree() {
            let address = layout.image_address(e, t)?;
            *weights.get_mut(address).ok_or(AkitaError::InvalidProof)? =
                -gamma * shifted_evaluation(challenge, t, &public.remainder_evaluations)?;
        }
    }
    Ok(weights)
}

/// Structured response terminal, with one canonical setup-owner contraction.
/// Preparation uses `z_i[s] = <u_i, rem(Y^s * e)>`. The contraction makes one
/// matrix pass into `n_A * D` accumulators; it never materializes setup weights.
pub fn witness_weight_mle<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    setup: &BinaryClearSetup<F, D, M>,
    rho: &[F],
) -> Result<F, AkitaError> {
    public.validate_layout(layout)?;
    layout.validate_setup(setup)?;
    if rho.len() != layout.witness_log_len() {
        return Err(AkitaError::InvalidInput(
            "response MLE point dimension mismatch".into(),
        ));
    }
    let digit_bits = layout.response_layout().digit_depth().trailing_zeros() as usize;
    let coefficient_bits = layout.padded_coefficients().trailing_zeros() as usize;
    let component_bits = layout.k().trailing_zeros() as usize;
    let (rho_l, rest) = rho
        .split_at_checked(digit_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let (rho_t, rho_j) = rest
        .split_at_checked(coefficient_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let (rho_c, rho_s) = rho_t
        .split_at_checked(component_bits)
        .ok_or(AkitaError::InvalidProof)?;
    let gadget = public
        .digit_powers
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (l, &digit)| {
            acc + eq_eval_at_index(rho_l, l) * digit
        });
    let e = TrinomialRing::<F, D, M>::from_coefficients(std::array::from_fn(|t| {
        eq_eval_at_index(rho_t, t)
    }))
    .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
    let z = multiplication_adjoint(&e, &public.alpha_powers)?;
    let row_len = checked::product([layout.n_a(), D]).ok_or(AkitaError::InvalidProof)?;
    let mut dense_rows = zero_vec(row_len)?;
    for (row, &gamma) in dense_rows.chunks_exact_mut(D).zip(&public.gamma_powers) {
        for (destination, &coefficient) in row.iter_mut().zip(&z) {
            *destination = gamma * coefficient;
        }
    }
    let prepared = layout.setup_view().prepare(rho_j, &dense_rows, gadget)?;
    let a_part =
        prepared.contract_matrix(setup.matrix().iter().map(|a| a.coefficients().as_slice()))?;
    let mut scalar = F::zero();
    for (s, &power) in public.xi_powers.iter().enumerate() {
        let t = checked::product([s, layout.k()]).ok_or(AkitaError::InvalidProof)?;
        scalar += eq_eval_at_index(rho_s, s) * signed_field::<F>(layout.sigma(t)?) * power;
    }
    let mut binary = F::zero();
    for (r, &value) in public.binary_rows.iter().enumerate() {
        binary += eq_eval_at_index(rho_c, r % layout.k())
            * eq_eval_at_index(rho_j, r / layout.k())
            * value;
    }
    Ok(a_part + public.g()? * gadget * scalar * binary)
}

/// Structured image terminal in coefficient-low, image-entry-high order.
/// It contracts the challenges by entry equality, then uses one ring product
/// and evaluation per matrix row. This is independent of the dense reference.
pub fn image_weight_mle<F: SmoothFftField>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    rho_y: &[F],
) -> Result<F, AkitaError> {
    public.validate_layout(layout)?;
    if rho_y.len() != layout.image_log_len() {
        return Err(AkitaError::InvalidInput(
            "image MLE point dimension mismatch".into(),
        ));
    }
    let (rho_t, rho_e) = rho_y
        .split_at_checked(layout.padded_coefficients().trailing_zeros() as usize)
        .ok_or(AkitaError::InvalidProof)?;
    match (layout.degree(), layout.polynomial().kind()) {
        (162, RelationPolynomialKind::Trinomial(TrinomialSign::Plus)) => {
            image_terminal::<F, 162, PlusTrinomial>(layout, public, rho_t, rho_e)
        }
        (324, RelationPolynomialKind::Trinomial(TrinomialSign::Minus)) => {
            image_terminal::<F, 324, MinusTrinomial>(layout, public, rho_t, rho_e)
        }
        (648, RelationPolynomialKind::Trinomial(TrinomialSign::Minus)) => {
            image_terminal::<F, 648, MinusTrinomial>(layout, public, rho_t, rho_e)
        }
        _ => Err(AkitaError::InvalidSetup(
            "unsupported image terminal ring".into(),
        )),
    }
}

fn image_terminal<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
    rho_t: &[F],
    rho_e: &[F],
) -> Result<F, AkitaError> {
    let domain = TrinomialNttDomain::<F, D, M>::new()
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
    let e = TrinomialRing::<F, D, M>::from_coefficients(std::array::from_fn(|t| {
        eq_eval_at_index(rho_t, t)
    }))
    .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
    let mut workspace = domain.workspace();
    let mut result = F::zero();
    for (i, &gamma) in public.gamma_powers.iter().take(layout.n_a()).enumerate() {
        let mut coefficients = zero_vec(D)?;
        for (col, challenge) in public.embedded_challenges.iter().enumerate() {
            let entry = checked::mul_add(col, layout.n_a(), i).ok_or(AkitaError::InvalidProof)?;
            let weight = eq_eval_at_index(rho_e, entry);
            for (destination, &coefficient) in coefficients.iter_mut().zip(challenge) {
                if !coefficient.is_zero() {
                    *destination += weight * coefficient;
                }
            }
        }
        let cbar = TrinomialRing::from_coefficients(
            coefficients
                .as_slice()
                .try_into()
                .map_err(|_| AkitaError::InvalidProof)?,
        )
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        let product = domain.multiply_with_workspace(&cbar, &e, &mut workspace);
        result -= gamma * horner(product.coefficients(), public.challenges.alpha);
    }
    Ok(result)
}
