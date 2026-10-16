//! Characteristic-zero response folding with bit-sliced carry-save counters.
//!
//! Each sign counts shifted source words in six-word bit vectors. A streaming
//! tree combines pairs at each weight: sixteen inputs emit one weight-sixteen
//! vector, and counting uses fewer carry-save additions than inputs. The
//! schedule depends only on that count, not on source bits or carries.
//!
//! The checked sum B of challenge weights bounds the inputs to either sign.
//! For B <= i16::MAX, fifteen unsigned bit planes hold every count, including
//! pending vectors; the input counter fits u16. Final normalization cannot emit
//! a nonzero weight-32768 carry. Word shifts retain their spill and all shift
//! distances are below the word width. The signed product has magnitude <= B.
//! Reduction has magnitude <= B at degrees 243..322, <= 2B at 162..242, and
//! <= 4B in the response, so i64 arithmetic cannot overflow. Larger or overflowing
//! weight sums use the reference's checked arithmetic and overflow errors.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

use akita_algebra::binary::field_switch::SwitchField;
use akita_challenges::{BinaryChallenge, BinaryChallengeProfile};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{endpoint, source::challenge_binary};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Fold source columns with the reference endpoint's validation and integer semantics.
///
/// Count the shifted source words separately for positive and negative terms,
/// subtract their integer counts, and reduce modulo `Y^162 + Y^81 + 1`.
/// The checked weight sum selects the bounded counter or the reference endpoint.
/// With `parallel`, independent rows write disjoint response slots; without it,
/// the same row kernel runs serially.
pub fn fold_integer<H: SwitchField>(
    source: &[H::Source],
    scalar_rows: usize,
    columns: usize,
    challenges: &[BinaryChallenge],
    profile: &BinaryChallengeProfile,
) -> Result<Vec<[i64; 162]>, AkitaError> {
    let length = checked::product([scalar_rows, columns])
        .ok_or_else(|| AkitaError::InvalidInput("integer fold size overflow".into()))?;
    if !scalar_rows.is_power_of_two()
        || !columns.is_power_of_two()
        || source.len() != length
        || challenges.len() != columns
    {
        return Err(AkitaError::InvalidInput(
            "integer fold geometry mismatch".into(),
        ));
    }
    for challenge in challenges {
        challenge_binary(challenge, profile)?;
    }
    let bound = checked::sum(challenges.iter().map(BinaryChallenge::weight));
    if bound.is_none_or(|bound| bound > i16::MAX as usize) {
        return endpoint::fold_integer::<H>(source, scalar_rows, columns, challenges, profile);
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(scalar_rows)
        .map_err(|_| AkitaError::InvalidInput("integer response allocation failed".into()))?;
    result.resize(scalar_rows, [0i64; 162]);

    let fold = |(row, response): (usize, &mut [i64; 162])| {
        fold_row(challenges, response, |column| {
            let index =
                checked::mul_add(column, scalar_rows, row).ok_or(AkitaError::InvalidProof)?;
            let word = *source.get(index).ok_or(AkitaError::InvalidProof)?;
            Ok(word.into())
        })
    };
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && scalar_rows > 1 {
        result.par_iter_mut().enumerate().try_for_each(fold)?;
        return Ok(result);
    }
    result.iter_mut().enumerate().try_for_each(fold)?;
    Ok(result)
}

fn fold_row(
    challenges: &[BinaryChallenge],
    response: &mut [i64; 162],
    mut word_at: impl FnMut(usize) -> Result<u128, AkitaError>,
) -> Result<(), AkitaError> {
    let mut positive = BitCounter::new();
    let mut negative = BitCounter::new();
    for (column, challenge) in challenges.iter().enumerate() {
        // Source.into() is the low 128 coordinates of embed_source, with zeros
        // in the remaining 34 coordinates, for both sealed SwitchField types.
        let word = word_at(column)?;
        for term in challenge.terms() {
            let position = usize::from(term.position);
            let shift = position % 64;
            let low = word << shift;
            // Splitting the right shift keeps the zero-shift case below 128.
            let spill = (word >> (127 - shift)) >> 1;
            let lo = low as u64;
            let hi = (low >> 64) as u64;
            let spill = spill as u64;
            // Challenge validation bounds position below 162, hence the word
            // offset is 0, 1 or 2 and all three shifted limbs fit the vector.
            let shifted = match position / 64 {
                0 => [lo, hi, spill, 0, 0, 0],
                1 => [0, lo, hi, spill, 0, 0],
                _ => [0, 0, lo, hi, spill, 0],
            };
            if term.coefficient == 1 {
                positive.add(shifted);
            } else {
                negative.add(shifted);
            }
        }
    }
    let positive = positive.planes();
    let negative = negative.planes();
    let mut product = [0i64; 323];
    for (weight, (positive, negative)) in positive.iter().zip(&negative).enumerate() {
        for ((chunk, &positive), &negative) in product.chunks_mut(64).zip(positive).zip(negative) {
            for (bit, coefficient) in chunk.iter_mut().enumerate() {
                let difference = ((positive >> bit) & 1) as i64 - ((negative >> bit) & 1) as i64;
                *coefficient += difference * (1i64 << weight);
            }
        }
    }
    for degree in (162..323).rev() {
        let leading = *product.get(degree).ok_or(AkitaError::InvalidProof)?;
        *product.get_mut(degree).ok_or(AkitaError::InvalidProof)? = 0;
        for position in [degree - 162, degree - 81] {
            *product.get_mut(position).ok_or(AkitaError::InvalidProof)? -= leading;
        }
    }
    for (destination, &coefficient) in response.iter_mut().zip(&product) {
        *destination = coefficient;
    }
    Ok(())
}

/// A streaming carry-save tree with weights 1 through 16384.
///
/// At each weight, `sum` holds a residue and `pending` holds one input exactly
/// when that bit of `inputs` is set. Each pair of inputs joins the residue in
/// one carry-save addition and sends its carry to the next weight. After n
/// inputs, the weighted sum of residues and active pending vectors is the
/// exact count at every source position.
struct BitCounter {
    sum: [[u64; 6]; 15],
    pending: [[u64; 6]; 15],
    inputs: u16,
}

impl BitCounter {
    fn new() -> Self {
        Self {
            sum: [[0; 6]; 15],
            pending: [[0; 6]; 15],
            inputs: 0,
        }
    }

    #[inline]
    fn add(&mut self, mut input: [u64; 6]) {
        let mut occupied = self.inputs;
        self.inputs += 1;
        for (sum, pending) in self.sum.iter_mut().zip(&mut self.pending) {
            if occupied & 1 == 0 {
                *pending = input;
                return;
            }
            (input, *sum) = carry_save(*sum, *pending, input);
            occupied >>= 1;
        }
    }

    fn planes(mut self) -> [[u64; 6]; 15] {
        let mut carry = [0; 6];
        for (sum, pending) in self.sum.iter_mut().zip(self.pending) {
            let input = if self.inputs & 1 == 0 {
                [0; 6]
            } else {
                pending
            };
            (carry, *sum) = carry_save(*sum, input, carry);
            self.inputs >>= 1;
        }
        self.sum
    }
}

/// Return the weight-two carry and weight-one sum of three bit vectors.
#[inline]
fn carry_save(a: [u64; 6], b: [u64; 6], c: [u64; 6]) -> ([u64; 6], [u64; 6]) {
    let mut carry = [0; 6];
    let mut sum = [0; 6];
    for ((((carry, sum), a), b), c) in carry.iter_mut().zip(&mut sum).zip(a).zip(b).zip(c) {
        let pair = a ^ b;
        *sum = pair ^ c;
        *carry = (a & b) | (pair & c);
    }
    (carry, sum)
}
