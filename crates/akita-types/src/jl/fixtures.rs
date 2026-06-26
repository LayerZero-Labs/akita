//! Shared JL consistency harness helpers for downstream crate tests.

use akita_algebra::EqPolynomial;
use akita_challenges::JlProjectionMatrix;
use akita_field::{
    field_modulus, AkitaError, CanonicalBytes, CanonicalField, FieldCore, TranscriptChallenge,
};
use akita_transcript::{AkitaTranscript, Transcript};

use super::{embed_signed_i32, jl_l2_norm_sq_checked, padded_live_table, JlWitnessLayout};

/// Embed a balanced witness digit for sumcheck table tests.
pub fn signed_witness_digit<F: FieldCore + CanonicalField>(value: i32) -> F {
    embed_signed_i32::<F>(value, field_modulus::<F>() / 2).expect("test digit in signed window")
}

/// Map witness digits to embedded field evals for padded consistency tables.
pub fn witness_evals_from_digits<F: FieldCore + CanonicalField>(digits: &[i32]) -> Vec<F> {
    digits.iter().copied().map(signed_witness_digit).collect()
}

/// Evaluate a padded witness table at a multilinear point via `eq(·)`.
pub fn eval_padded_table_at<F: FieldCore>(table: &[F], point: &[F]) -> Result<F, AkitaError> {
    let eq = EqPolynomial::evals(point)?;
    if eq.len() != table.len() {
        return Err(AkitaError::InvalidSize {
            expected: table.len(),
            actual: eq.len(),
        });
    }
    Ok(table
        .iter()
        .zip(eq.iter())
        .fold(F::zero(), |acc, (&value, &weight)| acc + value * weight))
}

/// Shared setup for JL consistency prove/verify roundtrip tests.
pub struct JlConsistencyFixture<F> {
    pub matrix: JlProjectionMatrix,
    pub layout: JlWitnessLayout,
    pub witness: Vec<F>,
    pub padded_witness: Vec<F>,
    pub image_coords: Vec<i32>,
    pub norm_bound: u128,
}

/// Build a reproducible JL consistency fixture from a transcript seed.
pub fn consistency_fixture<F>(
    seed: &[u8],
    n_rows: usize,
    live_x_cols: usize,
    ring_bits: usize,
) -> JlConsistencyFixture<F>
where
    F: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge,
{
    let ring_len = 1usize << ring_bits;
    let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
    let mut transcript = AkitaTranscript::<F>::new(seed);
    let matrix =
        JlProjectionMatrix::sample::<F, _>(&mut transcript, n_rows, live_x_cols * ring_len)
            .expect("fixture matrix sample");
    let layout = JlWitnessLayout::new(matrix.cols(), live_x_cols, col_bits, ring_bits).unwrap();
    let witness_digits: Vec<i32> = (0..layout.live_len()).map(|i| (i as i32 % 5) - 2).collect();
    let witness = witness_evals_from_digits::<F>(&witness_digits);
    let padded_witness = padded_live_table(layout, &witness).unwrap();
    let image = matrix.project_digits(&witness_digits).unwrap();
    let norm_bound = jl_l2_norm_sq_checked(image.coords()).unwrap();

    JlConsistencyFixture {
        matrix,
        layout,
        witness,
        padded_witness,
        image_coords: image.coords().to_vec(),
        norm_bound,
    }
}
