//! Standalone JL consistency-sumcheck prover.

use akita_algebra::UniPoly;
use akita_challenges::jl::mle::build_jl_row_weights;
use akita_challenges::jl::JlProjectionMatrix;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckProof};
use akita_transcript::{labels, Transcript};
use akita_types::jl::{
    absorb_jl_image, jl_image_claim, padded_live_table, sample_jl_row_point,
    validate_layout_for_matrix_mle, JlWitnessLayout, JL_CONSISTENCY_DEGREE,
};

/// Prove JL consistency for a compact flat witness table.
///
/// The witness table must use `w[x * 2^ring_bits + y]` order and contain only
/// the live entries. Padding to the sumcheck hypercube is handled internally.
pub fn prove_jl_consistency<F, T>(
    transcript: &mut T,
    matrix: &JlProjectionMatrix,
    layout: JlWitnessLayout,
    witness_evals: &[F],
    image_coords: &[i32],
    image_norm_bound_sq: Option<u128>,
) -> Result<(SumcheckProof<F>, Vec<F>, F), AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
    T: Transcript<F>,
{
    validate_layout_for_matrix_mle(matrix.cols(), layout)?;
    if witness_evals.len() != layout.live_len() {
        return Err(AkitaError::InvalidSize {
            expected: layout.live_len(),
            actual: witness_evals.len(),
        });
    }
    absorb_jl_image::<F, T>(transcript, image_coords);
    let r_j = sample_jl_row_point(transcript, matrix.n_rows());
    let image_claim =
        jl_image_claim::<F>(image_coords, matrix.n_rows(), image_norm_bound_sq, &r_j)?;
    let weight_table = padded_row_weight_table(matrix, layout, &r_j)?;
    let witness_table = padded_live_table(layout, witness_evals)?;
    let mut prover = JlConsistencyProver::new(layout, witness_table, weight_table, image_claim)?;
    prover.prove::<F, T, _>(transcript, |tr| {
        tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
    })
}

/// Prover instance for `Σ_i w(i) g(i) = claim`.
#[derive(Debug, Clone)]
pub struct JlConsistencyProver<F: FieldCore> {
    layout: JlWitnessLayout,
    input_claim: F,
    w_table: Vec<F>,
    weight_table: Vec<F>,
}

impl<F: FieldCore> JlConsistencyProver<F> {
    /// Construct a JL product-sumcheck prover over two padded tables.
    pub fn new(
        layout: JlWitnessLayout,
        w_table: Vec<F>,
        weight_table: Vec<F>,
        input_claim: F,
    ) -> Result<Self, AkitaError> {
        if w_table.len() != layout.padded_len() {
            return Err(AkitaError::InvalidSize {
                expected: layout.padded_len(),
                actual: w_table.len(),
            });
        }
        if weight_table.len() != layout.padded_len() {
            return Err(AkitaError::InvalidSize {
                expected: layout.padded_len(),
                actual: weight_table.len(),
            });
        }
        if !layout.padded_len().is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "JL consistency table length must be a power of two".to_string(),
            ));
        }
        Ok(Self {
            layout,
            input_claim,
            w_table,
            weight_table,
        })
    }
}

impl<F: FieldCore> SumcheckInstanceProver<F> for JlConsistencyProver<F> {
    fn num_rounds(&self) -> usize {
        self.layout.num_vars()
    }

    fn degree_bound(&self) -> usize {
        JL_CONSISTENCY_DEGREE
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        let (constant, linear, quadratic) =
            accumulate_product_round(&self.w_table, &self.weight_table);
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: F) {
        fold_table(&mut self.w_table, r_round);
        fold_table(&mut self.weight_table, r_round);
    }
}

fn padded_row_weight_table<F>(
    matrix: &JlProjectionMatrix,
    layout: JlWitnessLayout,
    r_j: &[F],
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore,
{
    let weights = build_jl_row_weights(matrix, r_j)?;
    padded_live_table(layout, &weights[..layout.live_len()])
}

fn accumulate_product_round<F: FieldCore>(lhs: &[F], rhs: &[F]) -> (F, F, F) {
    let half = lhs.len() / 2;
    let mut constant = F::zero();
    let mut linear = F::zero();
    let mut quadratic = F::zero();
    for pair_idx in 0..half {
        let l0 = lhs[2 * pair_idx];
        let l1 = lhs[2 * pair_idx + 1];
        let r0 = rhs[2 * pair_idx];
        let r1 = rhs[2 * pair_idx + 1];
        let dl = l1 - l0;
        let dr = r1 - r0;
        constant += l0 * r0;
        linear += l0 * dr + dl * r0;
        quadratic += dl * dr;
    }
    (constant, linear, quadratic)
}

fn fold_table<F: FieldCore>(table: &mut Vec<F>, r: F) {
    let half = table.len() / 2;
    for idx in 0..half {
        let left = table[2 * idx];
        let right = table[2 * idx + 1];
        table[idx] = left + r * (right - left);
    }
    table.truncate(half);
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::EqPolynomial;
    use akita_challenges::jl::mle::build_jl_row_weights;
    use akita_field::Fp64;
    use akita_transcript::AkitaTranscript;
    use akita_types::jl::{embed_jl_image_coords, embed_signed_i32, field_modulus};

    type F = Fp64<4294967197>;

    fn sample_matrix(n_rows: usize, cols: usize) -> JlProjectionMatrix {
        let mut transcript = AkitaTranscript::<F>::new(b"jl-prover-layout-test");
        JlProjectionMatrix::sample::<F, _>(&mut transcript, n_rows, cols).unwrap()
    }

    fn signed_field(value: i32) -> F {
        embed_signed_i32::<F>(value, field_modulus::<F>() / 2).unwrap()
    }

    fn witness_evals(digits: &[i32]) -> Vec<F> {
        digits.iter().copied().map(signed_field).collect()
    }

    #[test]
    fn row_weights_match_direct_integer_projection_for_flat_layout() {
        let live_x_cols = 3;
        let ring_bits = 2;
        let ring_len = 1usize << ring_bits;
        let matrix = sample_matrix(8, live_x_cols * ring_len);
        let layout = JlWitnessLayout::new(matrix.cols(), live_x_cols, 2, ring_bits).unwrap();
        let witness: Vec<i32> = (0..layout.live_len()).map(|i| (i as i32 % 5) - 2).collect();
        let image = matrix.project_digits(&witness).unwrap();
        let row_bits = matrix.n_rows().next_power_of_two().trailing_zeros() as usize;
        let r_j: Vec<F> = (0..row_bits).map(|i| F::from_u64(7 + i as u64)).collect();
        let eq_j = EqPolynomial::evals(&r_j).unwrap();
        let image_claim =
            image
                .coords()
                .iter()
                .zip(eq_j.iter())
                .fold(F::zero(), |acc, (&coord, &weight)| {
                    acc + weight * embed_signed_i32::<F>(coord, field_modulus::<F>() / 2).unwrap()
                });
        let g = build_jl_row_weights(&matrix, &r_j).unwrap();
        let flat_claim = witness
            .iter()
            .zip(g.iter())
            .fold(F::zero(), |acc, (&w, &weight)| {
                acc + weight * embed_signed_i32::<F>(w, field_modulus::<F>() / 2).unwrap()
            });

        assert_eq!(image_claim, flat_claim);
    }

    #[test]
    fn image_embedding_checks_shape_norm_and_signed_window() {
        let ok = embed_jl_image_coords::<F>(&[-3, 4], 2, Some(25)).unwrap();
        assert_eq!(ok.len(), 2);
        assert!(matches!(
            embed_jl_image_coords::<F>(&[-3, 4], 3, Some(25)),
            Err(AkitaError::InvalidSize { .. })
        ));
        assert!(embed_jl_image_coords::<F>(&[-3, 4], 2, Some(24)).is_err());
        assert!(embed_jl_image_coords::<F>(&[i32::MAX], 1, None).is_err());
    }

    #[test]
    fn prove_rejects_image_norm_bound() {
        let live_x_cols = 3;
        let ring_bits = 2;
        let ring_len = 1usize << ring_bits;
        let mut transcript = AkitaTranscript::<F>::new(b"jl-prover-norm-bound");
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut transcript, 8, live_x_cols * ring_len).unwrap();
        let layout = JlWitnessLayout::new(matrix.cols(), live_x_cols, 2, ring_bits).unwrap();
        let witness_digits: Vec<i32> = (0..layout.live_len()).map(|i| (i as i32 % 5) - 2).collect();
        let witness = witness_evals(&witness_digits);
        let image = matrix.project_digits(&witness_digits).unwrap();
        let norm_bound = image.l2_norm_sq_checked().unwrap();
        assert!(norm_bound > 0);

        assert!(prove_jl_consistency(
            &mut transcript,
            &matrix,
            layout,
            &witness,
            image.coords(),
            Some(norm_bound - 1),
        )
        .is_err());
    }

    #[test]
    fn prove_rejects_nonminimal_layout_for_matrix_mle() {
        let live_x_cols = 2;
        let ring_bits = 2;
        let ring_len = 1usize << ring_bits;
        let mut transcript = AkitaTranscript::<F>::new(b"jl-prover-malformed-layout");
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut transcript, 8, live_x_cols * ring_len).unwrap();
        let layout = JlWitnessLayout::new(matrix.cols(), live_x_cols, 2, ring_bits).unwrap();
        let witness_digits = vec![1; layout.live_len()];
        let witness = witness_evals(&witness_digits);
        let image = matrix.project_digits(&witness_digits).unwrap();

        assert!(prove_jl_consistency(
            &mut transcript,
            &matrix,
            layout,
            &witness,
            image.coords(),
            None,
        )
        .is_err());
    }
}
