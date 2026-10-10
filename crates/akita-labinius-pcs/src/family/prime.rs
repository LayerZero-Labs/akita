//! One base-field polynomial for the prime left opening and its exact claim.

use super::FieldFamily;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::lowered::LoweredRootLayout;
use jolt_field::{ExtField, Field, One, Zero};

/// Flatten the prime left opening into one base-field table, coordinate
/// innermost: entry `degree * i + k` is canonical base coefficient `k` of
/// extension element `i`. The first coordinate is the constant coefficient.
///
/// # Errors
///
/// Returns `InvalidSetup` for an unsupported extension degree or a failed
/// allocation, and `InvalidInput` for a table length that disagrees with the
/// admitted reduction layout.
pub fn flatten_prime_table<P: FieldFamily>(
    layout: &LoweredRootLayout,
    table: &[P::Challenge],
) -> Result<Vec<P::Base>, AkitaError> {
    let degree = P::Challenge::DEGREE;
    if !matches!(degree, 1 | 2) {
        return Err(AkitaError::InvalidSetup(
            "prime opening supports extension degrees one and two".into(),
        ));
    }
    if table.len() != layout.prime_len() {
        return Err(AkitaError::InvalidInput(
            "prime opening table length mismatch".into(),
        ));
    }
    let len = checked::product([layout.prime_len(), degree])
        .ok_or_else(|| AkitaError::InvalidSetup("prime element table length overflow".into()))?;
    let mut flattened = Vec::new();
    flattened
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidSetup("prime element table allocation failed".into()))?;
    for element in table {
        for coordinate in 0..degree {
            flattened.push(element.base_coefficient(coordinate));
        }
    }
    Ok(flattened)
}

/// Convert the reduction's extension-valued prime evaluation into the claim
/// on [`flatten_prime_table`], with the lowest variable first.
///
/// In degree one, the point and value are unchanged. In degree two, the
/// canonical basis is `(1, omega)`, so the new first coordinate is
/// `omega / (1 + omega)` and the value is divided by `1 + omega`.
/// The shipped `Ext2` uses coefficients `c0 + c1 * omega` in exactly this order.
///
/// # Errors
///
/// Returns `InvalidProof` for a point length that disagrees with the admitted
/// layout. Unsupported extension degrees, a noninvertible basis denominator,
/// overflow and allocation failures return `InvalidSetup`.
pub fn prime_opening_claim<P: FieldFamily>(
    layout: &LoweredRootLayout,
    point: &[P::Challenge],
    value: P::Challenge,
) -> Result<(Vec<P::Challenge>, P::Challenge), AkitaError> {
    let degree = P::Challenge::DEGREE;
    if !matches!(degree, 1 | 2) {
        return Err(AkitaError::InvalidSetup(
            "prime opening supports extension degrees one and two".into(),
        ));
    }
    if point.len() != layout.prime_log_len() {
        return Err(AkitaError::InvalidProof);
    }
    let len = checked::sum([point.len(), usize::from(degree == 2)])
        .ok_or_else(|| AkitaError::InvalidSetup("prime evaluation point length overflow".into()))?;
    let mut converted = Vec::new();
    converted
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidSetup("prime evaluation point allocation failed".into()))?;
    let value = if degree == 2 {
        // ExtField's generator enumerates the canonical base coordinates in
        // ascending order. This constructs [0, 1] without a slice precondition.
        let omega = P::Challenge::from_base_fn(|coordinate| {
            if coordinate == 1 {
                P::Base::one()
            } else {
                P::Base::zero()
            }
        });
        let inverse = (P::Challenge::one() + omega).inverse().ok_or_else(|| {
            AkitaError::InvalidSetup("prime coordinate denominator is not invertible".into())
        })?;
        converted.push(omega * inverse);
        value * inverse
    } else {
        value
    };
    converted.extend_from_slice(point);
    Ok((converted, value))
}
