//! Honest construction of the U view and both committed binary carries.

use akita_algebra::binary::BinaryField162;
use akita_algebra::ring::trinomial::{
    module_map::{scalar_to_binary, unpack_signed_components},
    signed::{signed_trinomial_product, signed_trinomial_remainder},
};
use akita_algebra::PlusTrinomial;
use akita_challenges::Challenges;
use akita_error::{checked, AkitaError};
use akita_types::binary_source::{
    binary_scalar_challenges, binary_scalar_lift, BinaryEvaluationBounds, BinarySourceDescriptor,
    BinaryTerminalClaim, BINARY_PARITY_CARRY_DIGITS, BINARY_VALUE_CARRY_DIGITS,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Natural coefficient-order digit segments fixed by the outgoing commitment.
/// No carry table is a clear proof message.
pub struct BinaryEvaluationWitness {
    pub parity_digits: Vec<i8>,
    pub value_digits: Vec<i8>,
    pub bounds: BinaryEvaluationBounds,
}

/// Construct U before the fold challenge, from the field-switch terminal row point.
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

/// Construct the binary row partials directly from packed source carriers.
/// This uses one F162 product per carrier and no signed-ring expansion.
pub(crate) fn packed_row_partials(
    descriptor: &BinarySourceDescriptor,
    claim: &BinaryTerminalClaim,
    sources: &[u128],
) -> Result<Vec<i8>, AkitaError> {
    let (weights, _) = claim.equality_weights(descriptor)?;
    let index = descriptor.index_map();
    let columns = index.columns();
    let rows =
        checked::exact_div(index.padded_carriers(), columns).ok_or(AkitaError::InvalidProof)?;
    let expected = checked::product([descriptor.polynomials().len(), index.padded_carriers()])
        .ok_or(AkitaError::InvalidProof)?;
    if sources.len() != expected || weights.len() != rows {
        return Err(AkitaError::InvalidInput(
            "packed binary row geometry mismatch".into(),
        ));
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
    let owner_start =
        checked::product([owner, index.padded_carriers()]).ok_or(AkitaError::InvalidProof)?;
    let range =
        checked::range(owner_start, index.padded_carriers()).ok_or(AkitaError::InvalidProof)?;
    let sources = sources.get(range).ok_or(AkitaError::InvalidProof)?;
    let owner_len = checked::product([columns, 162]).ok_or(AkitaError::InvalidProof)?;
    let range = checked::product([owner, owner_len])
        .and_then(|start| checked::range(start, owner_len))
        .ok_or(AkitaError::InvalidProof)?;
    let targets = partials.get_mut(range).ok_or(AkitaError::InvalidProof)?;
    let column = |(words, targets): (&[u128], &mut [i8])| -> Result<(), AkitaError> {
        let mut value = BinaryField162::ZERO;
        for (&word, &weight) in words.iter().zip(&weights) {
            let embedded = BinaryField162::from_words([word as u64, (word >> 64) as u64, 0])
                .ok_or(AkitaError::InvalidProof)?;
            value += weight * embedded;
        }
        for (target, bit) in targets.iter_mut().zip(binary_scalar_lift(value)) {
            *target = bit as i8;
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    sources
        .par_chunks_exact(rows)
        .zip(targets.par_chunks_exact_mut(162))
        .try_for_each(column)?;
    #[cfg(not(feature = "parallel"))]
    sources
        .chunks_exact(rows)
        .zip(targets.chunks_exact_mut(162))
        .try_for_each(column)?;
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
        let mut parity = tracing::info_span!("binary_parity_row_convolution").in_scope(|| {
            let mut scalar_rows = Vec::new();
            scalar_rows
                .try_reserve_exact(row_weights.len())
                .map_err(|_| AkitaError::InvalidProof)?;
            let start = checked::product([owner, ring_rows]).ok_or(AkitaError::InvalidProof)?;
            let range = checked::range(start, ring_rows).ok_or(AkitaError::InvalidProof)?;
            for element in response.get(range).ok_or(AkitaError::InvalidProof)? {
                let low = element.map(|coefficient| (coefficient + 2340) % 8 - 4);
                for component in unpack_signed_components(&low)? {
                    let mut values = [0i8; 162];
                    for (destination, value) in values.iter_mut().zip(component) {
                        *destination = i8::try_from(value).map_err(|_| AkitaError::InvalidProof)?;
                    }
                    scalar_rows.push(values);
                }
            }
            let product = crate::kernels::binary_source::weighted_integer_convolution::binary_weighted_integer_convolution(
                &row_weights,
                &scalar_rows,
            )?;
            signed_trinomial_remainder::<162, PlusTrinomial>(&product)
        })?;
        // Each column adds `partial * challenge` to the parity sum and
        // `partial * lift(weight)` to the value sum. The sums are exact
        // integers far inside i128, so columns run in parallel and reduce.
        let column = |column: usize| -> Result<[[i128; 162]; 2], AkitaError> {
            let element =
                checked::mul_add(owner, columns, column).ok_or(AkitaError::InvalidProof)?;
            let start = checked::product([element, 162]).ok_or(AkitaError::InvalidProof)?;
            let range = checked::range(start, 162).ok_or(AkitaError::InvalidProof)?;
            let partial = partials.get(range).ok_or(AkitaError::InvalidProof)?;
            let weight = *column_weights.get(column).ok_or(AkitaError::InvalidProof)?;
            Ok([
                signed_trinomial_product::<162, PlusTrinomial>(
                    partial,
                    scalar_challenges
                        .get(element)
                        .ok_or(AkitaError::InvalidProof)?,
                )?,
                signed_trinomial_product::<162, PlusTrinomial>(
                    partial,
                    &binary_scalar_lift(weight),
                )?,
            ])
        };
        let add = |mut sums: [[i128; 162]; 2], other: [[i128; 162]; 2]| {
            for (sum, other) in sums.iter_mut().zip(&other) {
                accumulate(sum, other, false)?;
            }
            Ok(sums)
        };
        #[cfg(feature = "parallel")]
        let [parity_sum, mut value] = (0..column_weights.len())
            .into_par_iter()
            .map(column)
            .try_reduce(|| [[0; 162]; 2], add)?;
        #[cfg(not(feature = "parallel"))]
        let [parity_sum, mut value] = (0..column_weights.len())
            .map(column)
            .try_fold([[0; 162]; 2], |sums, products| add(sums, products?))?;
        accumulate(&mut parity, &parity_sum, true)?;
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
