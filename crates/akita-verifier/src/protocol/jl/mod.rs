//! Standalone JL consistency-sumcheck verifier.

use akita_challenges::jl::mle::eval_jl_mle_at;
use akita_challenges::jl::JlProjectionMatrix;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceVerifier, SumcheckInstanceVerifierExt, SumcheckProof};
use akita_transcript::{labels, Transcript};
use akita_types::jl::{
    absorb_jl_image, jl_image_claim, sample_jl_row_point, validate_layout_for_matrix_mle,
    JlWitnessLayout, JL_CONSISTENCY_DEGREE,
};

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
    validate_layout_for_matrix_mle(matrix.cols(), layout)?;
    absorb_jl_image::<F, T>(transcript, image_coords);
    let r_j = sample_jl_row_point(transcript, matrix.n_rows());
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

/// Verifier instance for the JL product sumcheck.
pub(crate) struct JlConsistencyVerifier<'a, F, W>
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

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::{
        CanonicalBytes, CanonicalField, FieldCore, FromPrimitiveInt, Prime128OffsetA7F7,
        Prime32Offset99, Prime64Offset59, TranscriptChallenge,
    };
    use akita_prover::protocol::jl::prove_jl_consistency;
    use akita_transcript::AkitaTranscript;
    use akita_types::jl::fixtures::{eval_padded_table_at, witness_evals_from_digits};
    use akita_types::jl::padded_live_table;

    fn roundtrip_fixture<F>(seed: &[u8], n_rows: usize, live_x_cols: usize, ring_bits: usize)
    where
        F: FieldCore
            + CanonicalField
            + CanonicalBytes
            + TranscriptChallenge
            + AkitaSerialize
            + FromPrimitiveInt,
    {
        let ring_len = 1usize << ring_bits;
        let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
        let mut prover_transcript = AkitaTranscript::<F>::new(seed);
        let matrix = JlProjectionMatrix::sample::<F, _>(
            &mut prover_transcript,
            n_rows,
            live_x_cols * ring_len,
        )
        .unwrap();
        let mut verifier_transcript = AkitaTranscript::<F>::new(seed);
        let verifier_matrix = JlProjectionMatrix::sample::<F, _>(
            &mut verifier_transcript,
            n_rows,
            live_x_cols * ring_len,
        )
        .unwrap();
        let layout = JlWitnessLayout::new(matrix.cols(), live_x_cols, col_bits, ring_bits).unwrap();
        let witness_digits: Vec<i32> = (0..layout.live_len()).map(|i| (i as i32 % 5) - 2).collect();
        let witness = witness_evals_from_digits::<F>(&witness_digits);
        let padded_witness = padded_live_table(layout, &witness).unwrap();
        let image = matrix.project_digits(&witness_digits).unwrap();
        let norm_bound = image.l2_norm_sq_checked().unwrap();

        let (proof, r_w, _final_claim) = prove_jl_consistency(
            &mut prover_transcript,
            &matrix,
            layout,
            &witness,
            image.coords(),
            Some(norm_bound),
        )
        .unwrap();

        let challenges = verify_jl_consistency(
            &mut verifier_transcript,
            &verifier_matrix,
            layout,
            image.coords(),
            Some(norm_bound),
            &proof,
            |point| eval_padded_table_at(&padded_witness, point),
        )
        .unwrap();

        assert_eq!(challenges, r_w);
    }

    fn tampered_image_rejects<F>(seed: &[u8], n_rows: usize, live_x_cols: usize, ring_bits: usize)
    where
        F: FieldCore
            + CanonicalField
            + CanonicalBytes
            + TranscriptChallenge
            + AkitaSerialize
            + FromPrimitiveInt,
    {
        let ring_len = 1usize << ring_bits;
        let col_bits = live_x_cols.next_power_of_two().trailing_zeros() as usize;
        let mut prover_transcript = AkitaTranscript::<F>::new(seed);
        let matrix = JlProjectionMatrix::sample::<F, _>(
            &mut prover_transcript,
            n_rows,
            live_x_cols * ring_len,
        )
        .unwrap();
        let mut verifier_transcript = AkitaTranscript::<F>::new(seed);
        let verifier_matrix = JlProjectionMatrix::sample::<F, _>(
            &mut verifier_transcript,
            n_rows,
            live_x_cols * ring_len,
        )
        .unwrap();
        let layout = JlWitnessLayout::new(matrix.cols(), live_x_cols, col_bits, ring_bits).unwrap();
        let witness_digits: Vec<i32> = (0..layout.live_len()).map(|i| (i as i32 % 5) - 2).collect();
        let witness = witness_evals_from_digits::<F>(&witness_digits);
        let padded_witness = padded_live_table(layout, &witness).unwrap();
        let image = matrix.project_digits(&witness_digits).unwrap();
        let norm_bound = image.l2_norm_sq_checked().unwrap() + 1;
        let (proof, _, _) = prove_jl_consistency(
            &mut prover_transcript,
            &matrix,
            layout,
            &witness,
            image.coords(),
            Some(norm_bound),
        )
        .unwrap();
        let mut tampered = image.coords().to_vec();
        tampered[0] += 1;

        assert!(verify_jl_consistency(
            &mut verifier_transcript,
            &verifier_matrix,
            layout,
            &tampered,
            Some(norm_bound + 100),
            &proof,
            |point| eval_padded_table_at(&padded_witness, point),
        )
        .is_err());
    }

    #[test]
    fn jl_consistency_roundtrip_fp64() {
        type F = Prime64Offset59;
        roundtrip_fixture::<F>(b"jl-verifier-roundtrip-fp64", 8, 3, 2);
    }

    #[test]
    fn jl_consistency_roundtrip_fp32() {
        type F = Prime32Offset99;
        roundtrip_fixture::<F>(b"jl-verifier-roundtrip-fp32", 8, 3, 2);
    }

    #[test]
    fn jl_consistency_roundtrip_fp128() {
        type F = Prime128OffsetA7F7;
        roundtrip_fixture::<F>(b"jl-verifier-roundtrip-fp128", 8, 3, 2);
    }

    #[test]
    fn jl_consistency_rejects_tampered_image_fp64() {
        type F = Prime64Offset59;
        tampered_image_rejects::<F>(b"jl-verifier-tampered-fp64", 8, 3, 2);
    }

    #[test]
    fn jl_consistency_rejects_tampered_image_fp128() {
        type F = Prime128OffsetA7F7;
        tampered_image_rejects::<F>(b"jl-verifier-tampered-fp128", 8, 3, 2);
    }

    #[test]
    fn verify_rejects_nonminimal_layout_for_matrix_mle() {
        type F = Prime64Offset59;
        let live_x_cols = 2;
        let ring_bits = 2;
        let ring_len = 1usize << ring_bits;
        let mut transcript = AkitaTranscript::<F>::new(b"jl-verifier-malformed-layout");
        let matrix =
            JlProjectionMatrix::sample::<F, _>(&mut transcript, 8, live_x_cols * ring_len).unwrap();
        let layout = JlWitnessLayout::new(matrix.cols(), live_x_cols, 2, ring_bits).unwrap();
        let witness_digits = vec![1i32; layout.live_len()];
        let witness = witness_evals_from_digits::<F>(&witness_digits);
        let image = matrix.project_digits(&witness_digits).unwrap();
        let padded = padded_live_table(layout, &witness).unwrap();
        let empty_proof = SumcheckProof {
            round_polys: Vec::new(),
        };

        let mut verify_transcript = AkitaTranscript::<F>::new(b"jl-verifier-malformed-layout");
        let verify_matrix =
            JlProjectionMatrix::sample::<F, _>(&mut verify_transcript, 8, live_x_cols * ring_len)
                .unwrap();

        assert!(verify_jl_consistency(
            &mut verify_transcript,
            &verify_matrix,
            layout,
            image.coords(),
            None,
            &empty_proof,
            |point| eval_padded_table_at(&padded, point),
        )
        .is_err());
    }
}
