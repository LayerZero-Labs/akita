//! Clear lowered root relation with explicit coefficient-field challenges.
//!
//! This defines public linear weights and checks a clear digit table. It does
//! not bind challenges, commit tables, or implement a probabilistic proof.

pub(crate) mod adjoint;
mod check;
mod layout;
mod public;
mod weights;

pub use adjoint::{
    multiplication_adjoint, remainder_evaluations, trace_gram_into, trace_gram_inverse,
};
pub use check::{a_row_residual, check_lowered_clear, parity_row_residual};
pub use layout::LoweredRootLayout;
pub use public::{LoweredChallenges, LoweredPublic};
pub use weights::{
    image_weight_mle, image_weights_dense, witness_weight_mle, witness_weights_dense,
};

use akita_error::AkitaError;
use jolt_field::Field;

/// Signed integer reduction; even i128::MIN has a representable magnitude.
pub(crate) fn signed_field<F: Field>(value: i128) -> F {
    let magnitude = F::from_u128(value.unsigned_abs());
    if value < 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub(crate) fn horner<F: Field>(coefficients: &[F], point: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::zero(), |acc, &c| acc * point + c)
}

pub(crate) fn powers<F: Field>(point: F, len: usize) -> Result<Vec<F>, AkitaError> {
    let mut result = zero_vec(len)?;
    let mut power = F::one();
    for entry in &mut result {
        *entry = power;
        power *= point;
    }
    Ok(result)
}

pub(crate) fn zero_vec<F: Field>(len: usize) -> Result<Vec<F>, AkitaError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidInput("lowered table allocation failed".into()))?;
    result.resize(len, F::zero());
    Ok(result)
}
