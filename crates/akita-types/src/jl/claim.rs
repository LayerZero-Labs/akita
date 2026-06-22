//! JL consistency input-claim formation.

use akita_algebra::EqPolynomial;
use akita_field::{AkitaError, CanonicalField, FieldCore};

use super::wire::embed_jl_image_coords;

/// Batched JL image claim `sum_j eq(r_J, j) * embed(p[j])`.
pub fn jl_image_claim<F>(
    image_coords: &[i32],
    n_rows: usize,
    bound_sq: Option<u128>,
    r_j: &[F],
) -> Result<F, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let image = embed_jl_image_coords::<F>(image_coords, n_rows, bound_sq)?;
    let eq_j = EqPolynomial::evals(r_j)?;
    if eq_j.len() < n_rows {
        return Err(AkitaError::InvalidSize {
            expected: n_rows,
            actual: eq_j.len(),
        });
    }
    Ok(image
        .iter()
        .zip(eq_j.iter())
        .fold(F::zero(), |acc, (&coord, &weight)| acc + weight * coord))
}
