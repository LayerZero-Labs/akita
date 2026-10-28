//! Honest construction of the U view and both committed binary carries.

use akita_algebra::binary::BinaryField162;
use akita_algebra::ring::trinomial::{
    module_map::{scalar_to_binary, unpack_signed_components},
    signed::signed_trinomial_product,
};
use akita_algebra::PlusTrinomial;
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_types::binary_source::{
    binary_scalar_challenges, binary_scalar_lift, BinaryEvaluationBounds, BinarySourceDescriptor,
    BinaryTerminalClaim, BINARY_PARITY_CARRY_DIGITS, BINARY_VALUE_CARRY_DIGITS,
};

/// Natural coefficient-order digit segments fixed by the outgoing commitment.
/// No carry table is a clear proof message.
pub struct BinaryEvaluationWitness {
    pub parity_digits: Vec<i8>,
    pub value_digits: Vec<i8>,
    pub bounds: BinaryEvaluationBounds,
}

/// Construct U before the fold challenge, from the caller's terminal row point.
///
/// Sources use polynomial/column/ring-row order. The returned one-digit scalar
/// elements use polynomial/column order. Only the claimed polynomial is live;
/// unclaimed polynomial views are zero and are not assigned evaluation claims.
pub fn binary_row_partials(
    descriptor: &BinarySourceDescriptor,
    claim: &BinaryTerminalClaim,
    sources: &[[i128; 648]],
) -> Result<Vec<i8>, AkitaError> {
    let (weights, _) = claim.equality_weights(descriptor)?;
    let columns = descriptor.index_map().columns();
    let ring_rows = checked::exact_div(weights.len(), 4).ok_or(AkitaError::InvalidProof)?;
    let count = checked::product([descriptor.polynomials().len(), columns, ring_rows])
        .ok_or(AkitaError::InvalidProof)?;
    if sources.len() != count {
        return Err(AkitaError::InvalidSize {
            expected: count,
            actual: sources.len(),
        });
    }
    let owner = descriptor
        .polynomials()
        .iter()
        .position(|id| id == claim.polynomial())
        .ok_or(AkitaError::InvalidProof)?;
    let output_len = checked::product([descriptor.polynomials().len(), columns, 162])
        .ok_or(AkitaError::InvalidProof)?;
    let mut partials = Vec::new();
    partials
        .try_reserve_exact(output_len)
        .map_err(|_| AkitaError::InvalidProof)?;
    partials.resize(output_len, 0);
    for column in 0..columns {
        let mut value = BinaryField162::ZERO;
        for (row, &weight) in weights.iter().enumerate() {
            let element = checked::mul_add(owner, columns, column)
                .and_then(|x| checked::mul_add(x, ring_rows, row / 4))
                .ok_or(AkitaError::InvalidProof)?;
            let components =
                unpack_signed_components(sources.get(element).ok_or(AkitaError::InvalidProof)?)?;
            value += weight
                * scalar_to_binary(components.get(row % 4).ok_or(AkitaError::InvalidProof)?)?;
        }
        let start = checked::mul_add(owner, columns, column)
            .and_then(|x| checked::product([x, 162]))
            .ok_or(AkitaError::InvalidProof)?;
        let range = checked::range(start, 162).ok_or(AkitaError::InvalidProof)?;
        for (target, bit) in partials
            .get_mut(range)
            .ok_or(AkitaError::InvalidProof)?
            .iter_mut()
            .zip(binary_scalar_lift(value))
        {
            *target = bit as i8;
        }
    }
    Ok(partials)
}

impl BinaryEvaluationWitness {
    /// Construct K and K_val after the response, before Stage-2 batching.
    /// U is the already committed order-three view; every supplied digit may
    /// occupy the ordinary balanced alphabet, without a booleanity predicate.
    pub fn build(
        descriptor: &BinarySourceDescriptor,
        claim: &BinaryTerminalClaim,
        partials: &[i8],
        response: &[[i128; 648]],
        challenges: &Challenges,
        proof_modulus: u128,
    ) -> Result<Self, AkitaError> {
        let (row_weights, column_weights) = claim.equality_weights(descriptor)?;
        let columns = descriptor.index_map().columns();
        let polynomials = descriptor.polynomials().len();
        let bounds = BinaryEvaluationBounds::new(row_weights.len(), columns, proof_modulus)?;
        let ring_rows = checked::exact_div(row_weights.len(), 4).ok_or(AkitaError::InvalidProof)?;
        let expected_response =
            checked::product([polynomials, ring_rows]).ok_or(AkitaError::InvalidProof)?;
        let expected_partials =
            checked::product([polynomials, columns, 162]).ok_or(AkitaError::InvalidProof)?;
        if response.len() != expected_response
            || partials.len() != expected_partials
            || partials.iter().any(|digit| !(-4..=3).contains(digit))
            || response
                .iter()
                .flatten()
                .any(|value| !(-2340..=1755).contains(value))
        {
            return Err(AkitaError::InvalidProof);
        }
        let owner = descriptor
            .polynomials()
            .iter()
            .position(|id| id == claim.polynomial())
            .ok_or(AkitaError::InvalidProof)?;
        let scalar_challenges = binary_scalar_challenges(descriptor, challenges)?;
        let mut parity = [0i128; 162];
        let mut value = [0i128; 162];
        for (row, weight) in row_weights.into_iter().enumerate() {
            let element =
                checked::mul_add(owner, ring_rows, row / 4).ok_or(AkitaError::InvalidProof)?;
            let low = response
                .get(element)
                .ok_or(AkitaError::InvalidProof)?
                .map(|coefficient| (coefficient + 2340) % 8 - 4);
            let components = unpack_signed_components(&low)?;
            let product = signed_trinomial_product::<162, PlusTrinomial>(
                &binary_scalar_lift(weight),
                components.get(row % 4).ok_or(AkitaError::InvalidProof)?,
            )?;
            accumulate(&mut parity, &product, false)?;
        }
        for (column, weight) in column_weights.into_iter().enumerate() {
            let element =
                checked::mul_add(owner, columns, column).ok_or(AkitaError::InvalidProof)?;
            let start = checked::product([element, 162]).ok_or(AkitaError::InvalidProof)?;
            let range = checked::range(start, 162).ok_or(AkitaError::InvalidProof)?;
            let partial = partials.get(range).ok_or(AkitaError::InvalidProof)?;
            let product = signed_trinomial_product::<162, PlusTrinomial>(
                partial,
                scalar_challenges
                    .get(element)
                    .ok_or(AkitaError::InvalidProof)?,
            )?;
            accumulate(&mut parity, &product, true)?;
            let product = signed_trinomial_product::<162, PlusTrinomial>(
                partial,
                &binary_scalar_lift(weight),
            )?;
            accumulate(&mut value, &product, false)?;
        }
        let terminal = BinaryField162::from_bytes(claim.value()).ok_or(AkitaError::InvalidProof)?;
        accumulate(&mut value, &binary_scalar_lift(terminal), true)?;
        let mut parity_digits =
            vec![
                0;
                checked::product([polynomials, BINARY_PARITY_CARRY_DIGITS, 162])
                    .ok_or(AkitaError::InvalidProof)?
            ];
        let mut value_digits = vec![
            0;
            checked::product([polynomials, BINARY_VALUE_CARRY_DIGITS, 162])
                .ok_or(AkitaError::InvalidProof)?
        ];
        for (residual, digits, count, bound) in [
            (
                &parity,
                &mut parity_digits,
                BINARY_PARITY_CARRY_DIGITS,
                bounds.parity_carry_bound,
            ),
            (
                &value,
                &mut value_digits,
                BINARY_VALUE_CARRY_DIGITS,
                bounds.value_carry_bound,
            ),
        ] {
            for (coefficient, &integer) in residual.iter().enumerate() {
                if integer % 2 != 0 {
                    return Err(AkitaError::InvalidProof);
                }
                let word = (integer / 2)
                    .checked_add(i128::try_from(bound).map_err(|_| AkitaError::InvalidProof)?)
                    .and_then(|x| u128::try_from(x).ok())
                    .ok_or(AkitaError::InvalidProof)?;
                let width = checked::product([3, count])
                    .and_then(|x| u32::try_from(x).ok())
                    .ok_or(AkitaError::InvalidProof)?;
                if word >= 1u128.checked_shl(width).ok_or(AkitaError::InvalidProof)? {
                    return Err(AkitaError::InvalidProof);
                }
                for digit in 0..count {
                    let address = checked::mul_add(owner, count, digit)
                        .and_then(|x| checked::mul_add(x, 162, coefficient))
                        .ok_or(AkitaError::InvalidProof)?;
                    let shift = checked::product([3, digit]).ok_or(AkitaError::InvalidProof)?;
                    *digits.get_mut(address).ok_or(AkitaError::InvalidProof)? =
                        ((word >> shift) & 7) as i8 - 4;
                }
            }
        }
        Ok(Self {
            parity_digits,
            value_digits,
            bounds,
        })
    }
}

fn accumulate(
    target: &mut [i128; 162],
    source: &[i128; 162],
    subtract: bool,
) -> Result<(), AkitaError> {
    for (target, &source) in target.iter_mut().zip(source) {
        *target = if subtract {
            target.checked_sub(source)
        } else {
            target.checked_add(source)
        }
        .ok_or(AkitaError::InvalidProof)?;
    }
    Ok(())
}
