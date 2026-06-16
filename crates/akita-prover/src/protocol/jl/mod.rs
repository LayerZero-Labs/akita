//! Standalone JL consistency-sumcheck helpers.

use akita_algebra::UniPoly;
use akita_challenges::jl::mle::{build_jl_row_weights, eval_jl_mle_at};
use akita_challenges::jl::JlProjectionMatrix;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{
    SumcheckInstanceProver, SumcheckInstanceProverExt, SumcheckInstanceVerifier,
    SumcheckInstanceVerifierExt, SumcheckProof,
};
use akita_transcript::{labels, Transcript};

/// Degree bound for the JL product sumcheck.
pub const JL_CONSISTENCY_DEGREE: usize = 2;

/// Witness layout for the flattened JL consistency table.
///
/// The compact witness order is `w[x * 2^ring_bits + y]`, with `x` as the
/// outer column index and `y` as the ring-slot index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JlWitnessLayout {
    /// Number of live outer columns before power-of-two padding.
    pub live_x_cols: usize,
    /// Number of bits in the padded outer-column hypercube.
    pub col_bits: usize,
    /// Number of bits in the ring-slot hypercube.
    pub ring_bits: usize,
    ring_len: usize,
    padded_len: usize,
}

impl JlWitnessLayout {
    /// Construct and validate the flat JL witness layout for `matrix`.
    ///
    /// # Errors
    ///
    /// Returns an error if the live shape does not match the matrix column count
    /// or if any power-of-two layout size overflows.
    pub fn new(
        matrix: &JlProjectionMatrix,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
    ) -> Result<Self, AkitaError> {
        if live_x_cols == 0 {
            return Err(AkitaError::InvalidInput(
                "JL witness layout requires a non-zero live column count".to_string(),
            ));
        }
        let ring_len = pow2(ring_bits, "JL witness ring dimension")?;
        let padded_x_cols = pow2(col_bits, "JL witness padded column dimension")?;
        if live_x_cols > padded_x_cols {
            return Err(AkitaError::InvalidInput(format!(
                "JL witness live columns {live_x_cols} exceed padded column capacity {padded_x_cols}"
            )));
        }
        let live_len = live_x_cols.checked_mul(ring_len).ok_or_else(|| {
            AkitaError::InvalidInput("JL witness live length overflow".to_string())
        })?;
        if matrix.cols() != live_len {
            return Err(AkitaError::InvalidSize {
                expected: live_len,
                actual: matrix.cols(),
            });
        }
        let padded_len = padded_x_cols.checked_mul(ring_len).ok_or_else(|| {
            AkitaError::InvalidInput("JL witness padded length overflow".to_string())
        })?;
        Ok(Self {
            live_x_cols,
            col_bits,
            ring_bits,
            ring_len,
            padded_len,
        })
    }

    /// Number of live flat witness entries, equal to the JL matrix column count.
    pub fn live_len(&self) -> usize {
        self.live_x_cols * self.ring_len
    }

    /// Number of padded flat witness entries in the sumcheck hypercube.
    pub fn padded_len(&self) -> usize {
        self.padded_len
    }

    /// Number of variables in the flat witness hypercube.
    pub fn num_vars(&self) -> usize {
        self.col_bits + self.ring_bits
    }

    /// Flat index for compact witness order `w[x * 2^ring_bits + y]`.
    pub fn flat_index(&self, x: usize, y: usize) -> Result<usize, AkitaError> {
        if x >= self.live_x_cols || y >= self.ring_len {
            return Err(AkitaError::InvalidInput(
                "JL witness flat index out of range".to_string(),
            ));
        }
        Ok(x * self.ring_len + y)
    }
}

/// Absorb verifier-wire JL image coordinates before sampling `r_J`.
pub fn absorb_jl_image<F, T>(transcript: &mut T, image_coords: &[i32])
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.absorb_and_record_bytes(
        labels::ABSORB_JL_IMAGE,
        &image_coords_to_bytes(image_coords),
    );
}

/// Embed and optionally norm-check verifier-wire JL image coordinates.
///
/// # Errors
///
/// Returns an error if the coordinate count does not match the matrix row count,
/// if the checked integer L2 norm exceeds `bound_sq`, or if any signed
/// coordinate lies outside the field's injective signed window.
pub fn embed_jl_image_coords<F>(
    image_coords: &[i32],
    n_rows: usize,
    bound_sq: Option<u128>,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if image_coords.len() != n_rows {
        return Err(AkitaError::InvalidSize {
            expected: n_rows,
            actual: image_coords.len(),
        });
    }
    if let Some(bound_sq) = bound_sq {
        check_l2_norm(image_coords, bound_sq)?;
    }
    let q = field_modulus::<F>();
    let half_q = q / 2;
    image_coords
        .iter()
        .map(|&coord| embed_signed_i32::<F>(coord, half_q))
        .collect()
}

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
    validate_layout_for_matrix_mle(matrix, layout)?;
    if witness_evals.len() != layout.live_len() {
        return Err(AkitaError::InvalidSize {
            expected: layout.live_len(),
            actual: witness_evals.len(),
        });
    }
    absorb_jl_image::<F, T>(transcript, image_coords);
    let r_j = sample_jl_row_point(transcript, matrix);
    let image_claim =
        jl_image_claim::<F>(image_coords, matrix.n_rows(), image_norm_bound_sq, &r_j)?;
    let weight_table = padded_row_weight_table(matrix, layout, &r_j)?;
    let witness_table = padded_live_table(layout, witness_evals)?;
    let mut prover = JlConsistencyProver::new(layout, witness_table, weight_table, image_claim)?;
    prover.prove::<F, T, _>(transcript, |tr| {
        tr.challenge_scalar(labels::CHALLENGE_SUMCHECK_ROUND)
    })
}

/// Verify a standalone JL consistency sumcheck proof.
///
/// The verifier receives `w_tilde(r_w)` through `w_eval_hook`; this standalone
/// helper does not perform a commitment opening.
pub fn verify_jl_consistency<F, T, W>(
    transcript: &mut T,
    matrix: &JlProjectionMatrix,
    layout: JlWitnessLayout,
    image_coords: &[i32],
    image_norm_bound_sq: Option<u128>,
    proof: &SumcheckProof<F>,
    w_eval_hook: W,
) -> Result<Vec<F>, AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
    T: Transcript<F>,
    W: Fn(&[F]) -> Result<F, AkitaError> + Send + Sync,
{
    validate_layout_for_matrix_mle(matrix, layout)?;
    absorb_jl_image::<F, T>(transcript, image_coords);
    let r_j = sample_jl_row_point(transcript, matrix);
    let image_claim =
        jl_image_claim::<F>(image_coords, matrix.n_rows(), image_norm_bound_sq, &r_j)?;
    let verifier = JlConsistencyVerifier {
        matrix,
        layout,
        r_j,
        input_claim: image_claim,
        w_eval_hook,
    };
    verifier.verify::<F, T, _>(proof, transcript, |tr| {
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

/// Verifier instance for the JL product sumcheck.
pub struct JlConsistencyVerifier<'a, F, W>
where
    F: FieldCore,
{
    matrix: &'a JlProjectionMatrix,
    layout: JlWitnessLayout,
    r_j: Vec<F>,
    input_claim: F,
    w_eval_hook: W,
}

impl<F, W> SumcheckInstanceVerifier<F> for JlConsistencyVerifier<'_, F, W>
where
    F: FieldCore + CanonicalField,
    W: Fn(&[F]) -> Result<F, AkitaError> + Send + Sync,
{
    fn num_rounds(&self) -> usize {
        self.layout.num_vars()
    }

    fn degree_bound(&self) -> usize {
        JL_CONSISTENCY_DEGREE
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn expected_output_claim(&self, challenges: &[F]) -> Result<F, AkitaError> {
        if challenges.len() != self.layout.num_vars() {
            return Err(AkitaError::InvalidSize {
                expected: self.layout.num_vars(),
                actual: challenges.len(),
            });
        }
        let w_eval = (self.w_eval_hook)(challenges)?;
        let jl_eval = eval_jl_mle_at(self.matrix, &self.r_j, challenges)?;
        Ok(w_eval * jl_eval)
    }
}

fn sample_jl_row_point<F, T>(transcript: &mut T, matrix: &JlProjectionMatrix) -> Vec<F>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let row_bits = matrix.n_rows().next_power_of_two().trailing_zeros() as usize;
    (0..row_bits)
        .map(|_| transcript.challenge_scalar(labels::CHALLENGE_JL_ROW))
        .collect()
}

fn jl_image_claim<F>(
    image_coords: &[i32],
    n_rows: usize,
    bound_sq: Option<u128>,
    r_j: &[F],
) -> Result<F, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let image = embed_jl_image_coords::<F>(image_coords, n_rows, bound_sq)?;
    let eq_j = akita_algebra::EqPolynomial::evals(r_j)?;
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

fn padded_live_table<F: FieldCore>(
    layout: JlWitnessLayout,
    live_evals: &[F],
) -> Result<Vec<F>, AkitaError> {
    if live_evals.len() != layout.live_len() {
        return Err(AkitaError::InvalidSize {
            expected: layout.live_len(),
            actual: live_evals.len(),
        });
    }
    let mut table = vec![F::zero(); layout.padded_len()];
    table[..live_evals.len()].copy_from_slice(live_evals);
    Ok(table)
}

fn validate_layout_for_matrix_mle(
    matrix: &JlProjectionMatrix,
    layout: JlWitnessLayout,
) -> Result<(), AkitaError> {
    let matrix_hyper = matrix.cols().next_power_of_two();
    if layout.padded_len() != matrix_hyper {
        return Err(AkitaError::InvalidInput(format!(
            "JL layout padded length {} does not match matrix MLE hypercube {matrix_hyper}",
            layout.padded_len()
        )));
    }
    Ok(())
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

fn check_l2_norm(coords: &[i32], bound_sq: u128) -> Result<(), AkitaError> {
    let mut norm_sq = 0u128;
    for &coord in coords {
        let mag = u128::from(coord.unsigned_abs());
        let sq = mag * mag;
        norm_sq = norm_sq.checked_add(sq).ok_or_else(|| {
            AkitaError::InvalidInput("JL image squared norm exceeds u128".to_string())
        })?;
    }
    if norm_sq > bound_sq {
        return Err(AkitaError::InvalidInput(format!(
            "JL image squared L2 norm {norm_sq} exceeds bound {bound_sq}"
        )));
    }
    Ok(())
}

#[inline]
fn embed_signed_i32<F>(coord: i32, half_q: u128) -> Result<F, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let mag = u128::from(coord.unsigned_abs());
    if mag > half_q {
        return Err(AkitaError::InvalidInput(format!(
            "JL image coordinate {coord} outside injective signed window (|c| <= {half_q})"
        )));
    }
    let elem = F::from_canonical_u128_reduced(mag);
    Ok(if coord < 0 { -elem } else { elem })
}

#[inline]
fn field_modulus<F>() -> u128
where
    F: FieldCore + CanonicalField,
{
    (-F::one()).to_canonical_u128() + 1
}

fn image_coords_to_bytes(image_coords: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(image_coords));
    for &coord in image_coords {
        bytes.extend_from_slice(&coord.to_le_bytes());
    }
    bytes
}

fn pow2(bits: usize, name: &str) -> Result<usize, AkitaError> {
    1usize
        .checked_shl(bits as u32)
        .ok_or_else(|| AkitaError::InvalidInput(format!("{name} overflows usize")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::EqPolynomial;
    use akita_challenges::jl::mle::build_jl_row_weights;
    use akita_field::Fp64;
    use akita_transcript::AkitaTranscript;

    type F = Fp64<4294967197>;

    fn sample_matrix(n_rows: usize, cols: usize) -> JlProjectionMatrix {
        let mut transcript = AkitaTranscript::<F>::new(b"jl-pr2-layout-test");
        JlProjectionMatrix::sample::<F, _>(&mut transcript, n_rows, cols).unwrap()
    }

    #[test]
    fn layout_pins_flat_x_outer_y_inner_order() {
        let live_x_cols = 3;
        let ring_bits = 2;
        let ring_len = 1usize << ring_bits;
        let col_bits = 2;
        let matrix = sample_matrix(8, live_x_cols * ring_len);
        let layout = JlWitnessLayout::new(&matrix, live_x_cols, col_bits, ring_bits).unwrap();

        assert_eq!(layout.live_len(), 12);
        assert_eq!(layout.padded_len(), 16);
        assert_eq!(layout.num_vars(), 4);
        assert_eq!(layout.flat_index(0, 0).unwrap(), 0);
        assert_eq!(layout.flat_index(0, 3).unwrap(), 3);
        assert_eq!(layout.flat_index(1, 0).unwrap(), 4);
        assert_eq!(layout.flat_index(2, 3).unwrap(), 11);
    }

    #[test]
    fn row_weights_match_direct_integer_projection_for_flat_layout() {
        let live_x_cols = 3;
        let ring_bits = 2;
        let ring_len = 1usize << ring_bits;
        let matrix = sample_matrix(8, live_x_cols * ring_len);
        let layout = JlWitnessLayout::new(&matrix, live_x_cols, 2, ring_bits).unwrap();
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
}
