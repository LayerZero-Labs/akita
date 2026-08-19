use super::*;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{
    CanonicalBytes, CanonicalField, ExtField, FpExt2, FpExt4, Prime32Offset99, Prime64Offset59,
    TranscriptChallenge, TwoNr,
};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::EqFactoredSumcheckInstanceProverExt;
use akita_transcript::{
    labels as transcript_labels, sample_ext_challenge, AkitaTranscript, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn scalar_range_entry_coefficients<E: FieldCore + FromPrimitiveInt>(
    basis: usize,
    left: E,
    right: E,
) -> Vec<E> {
    let delta = right - left;
    match basis {
        4 => {
            let twice_left = left + left;
            vec![
                left * (left - E::from_u64(2)),
                delta * (twice_left - E::from_u64(2)),
                delta * delta,
            ]
        }
        8 => {
            let twice_left = left + left;
            let sixteen_times_left = twice_left + twice_left;
            let sixteen_times_left = sixteen_times_left + sixteen_times_left;
            let sixteen_times_left = sixteen_times_left + sixteen_times_left;
            let left_squared = left * left;
            let first_quadratic = left_squared - twice_left;
            let second_quadratic =
                left_squared - (sixteen_times_left + twice_left) + E::from_u64(72);
            let delta_squared = delta * delta;
            let first_linear = delta * (twice_left - E::from_u64(2));
            let second_linear = delta * (twice_left - E::from_u64(18));
            vec![
                first_quadratic * second_quadratic,
                first_quadratic * second_linear + first_linear * second_quadratic,
                first_quadratic * delta_squared
                    + first_linear * second_linear
                    + delta_squared * second_quadratic,
                delta_squared * (first_linear + second_linear),
                delta_squared * delta_squared,
            ]
        }
        _ => unreachable!("scalar Stage 1 reference only supports basis 4 or 8"),
    }
}

struct ScalarStage1Reference<E: FieldCore> {
    range_image: Vec<E>,
    split_eq: GruenSplitEq<E>,
    basis: usize,
    num_vars: usize,
    rounds_completed: usize,
}

impl<E: FieldCore + FromPrimitiveInt> ScalarStage1Reference<E> {
    fn new(
        digit_witness: &[i8],
        tau: &[E],
        basis: usize,
        live_x_cols: usize,
        col_bits: usize,
        ring_bits: usize,
    ) -> Self {
        let padded = pad_compact_witness(digit_witness, live_x_cols, col_bits, ring_bits);
        Self {
            range_image: build_compact_range_image(&padded)
                .into_iter()
                .map(|value| E::from_i64(i64::from(value)))
                .collect(),
            split_eq: GruenSplitEq::new(tau).unwrap(),
            basis,
            num_vars: col_bits + ring_bits,
            rounds_completed: 0,
        }
    }

    fn round_polynomial(&self) -> EqFactoredUniPoly<E> {
        let (e_first, e_second) = self.split_eq.remaining_eq_tables();
        let first_bits = e_first.len().trailing_zeros();
        let mut coefficients = vec![E::zero(); self.basis / 2 + 1];
        for (pair_index, pair) in self.range_image.chunks_exact(2).enumerate() {
            let inner_index = pair_index & (e_first.len() - 1);
            let outer_index = pair_index >> first_bits;
            let weight = e_first[inner_index] * e_second[outer_index];
            let entry = scalar_range_entry_coefficients(self.basis, pair[0], pair[1]);
            for (coefficient, entry_coefficient) in coefficients.iter_mut().zip(entry) {
                *coefficient += weight * entry_coefficient;
            }
        }
        EqFactoredUniPoly::from_q_coeffs(coefficients)
    }

    fn fold(&mut self, challenge: E) {
        let mut next = Vec::with_capacity(self.range_image.len() / 2);
        for pair in self.range_image.chunks_exact(2) {
            next.push(pair[0] + challenge * (pair[1] - pair[0]));
        }
        self.range_image = next;
        self.split_eq.bind(challenge);
        self.rounds_completed += 1;
    }

    fn final_range_image_eval(&self) -> E {
        assert_eq!(self.range_image.len(), 1);
        self.range_image[0]
    }
}

impl<E: FieldCore + FromPrimitiveInt> EqFactoredSumcheckInstanceProver<E>
    for ScalarStage1Reference<E>
{
    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn degree_bound(&self) -> usize {
        self.basis / 2
    }

    fn input_claim(&self) -> E {
        E::zero()
    }

    fn current_linear_factor_evals(&self) -> (E, E) {
        self.split_eq.linear_factor_evals()
    }

    fn compute_round_eq_factored(&mut self, round: usize) -> EqFactoredUniPoly<E> {
        assert_eq!(round, self.rounds_completed);
        self.round_polynomial()
    }

    fn ingest_challenge(&mut self, round: usize, challenge: E) {
        assert_eq!(round, self.rounds_completed);
        self.fold(challenge);
    }
}

fn scalar_fold_octet_class<E: FieldCore + FromPrimitiveInt>(
    basis: usize,
    class: usize,
    challenges: [E; 3],
) -> E {
    let mut values = match basis {
        4 => (0..8)
            .map(|index| E::from_u64(((class >> index) & 1) as u64 * 2))
            .collect::<Vec<_>>(),
        8 => {
            let range_values = [0u64, 2, 6, 12];
            let left = class >> 8;
            let right = class & 0xff;
            (0..4)
                .map(|index| E::from_u64(range_values[(left >> (2 * index)) & 3]))
                .chain((0..4).map(|index| E::from_u64(range_values[(right >> (2 * index)) & 3])))
                .collect()
        }
        _ => unreachable!("folded octets only support basis 4 or 8"),
    };
    for challenge in challenges {
        values = values
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect();
    }
    values[0]
}

fn expected_octet_class_code(basis: usize, octet: &[i16]) -> u16 {
    match basis {
        4 => octet.iter().enumerate().fold(0u16, |code, (index, value)| {
            code | (u16::from(*value == 2) << index)
        }),
        8 => {
            let digit_code = |values: &[i16]| {
                values
                    .iter()
                    .enumerate()
                    .fold(0u16, |code, (index, value)| {
                        let digit = match value {
                            0 => 0,
                            2 => 1,
                            6 => 2,
                            12 => 3,
                            _ => unreachable!("basis-8 range image"),
                        };
                        code | (digit << (2 * index))
                    })
            };
            (digit_code(&octet[..4]) << 8) | digit_code(&octet[4..])
        }
        _ => unreachable!("folded octets only support basis 4 or 8"),
    }
}

fn assert_folded_octets_match_scalar<E: FieldCore + FromPrimitiveInt + HasUnreducedOps>(
    folded: &FoldedOctetRangeImage<E>,
    scalar_range_image: &[E],
    compact_range_image: &[i16],
    basis: usize,
    live_x_cols: usize,
    y_len: usize,
    challenges: [E; 3],
) {
    let next_y_len = y_len / 8;
    let expected_codes = compact_range_image
        .chunks_exact(y_len)
        .flat_map(|column| {
            column
                .chunks_exact(8)
                .map(|octet| expected_octet_class_code(basis, octet))
        })
        .collect::<Vec<_>>();
    assert_eq!(expected_codes.len(), live_x_cols * next_y_len);
    assert_eq!(folded.class_codes, expected_codes);

    for (class, (&value, taylor)) in folded
        .class_values
        .iter()
        .zip(&folded.class_taylor_coefficients)
        .enumerate()
    {
        let expected_value = scalar_fold_octet_class(basis, class, challenges);
        assert_eq!(
            value, expected_value,
            "class value mismatch for class={class}"
        );
        let expected_taylor =
            scalar_range_entry_coefficients(basis, expected_value, expected_value + E::one());
        let expected_taylor = match basis {
            4 => [expected_taylor[0], expected_taylor[1], E::zero(), E::zero()],
            8 => [
                expected_taylor[0],
                expected_taylor[1],
                expected_taylor[2],
                expected_taylor[3],
            ],
            _ => unreachable!("folded octets only support basis 4 or 8"),
        };
        assert_eq!(
            *taylor, expected_taylor,
            "Taylor row mismatch for class={class}"
        );
    }

    assert_eq!(
        (0..folded.len())
            .map(|index| folded.value(index))
            .collect::<Vec<_>>(),
        scalar_range_image
    );
}

fn new_stage1_transcript() -> AkitaTranscript<F> {
    AkitaTranscript::new(transcript_labels::DOMAIN_AKITA_PROTOCOL)
}

fn sample_stage1_challenge(transcript: &mut AkitaTranscript<F>) -> F {
    transcript.challenge_scalar(transcript_labels::CHALLENGE_SUMCHECK_ROUND)
}

#[test]
fn stage1_compact_pipeline_randomized_scalar_transcript_and_bytes_differential() {
    let shapes = [
        (6usize, 3usize, 4usize),
        (1, 3, 4),
        (7, 3, 4),
        (8, 3, 4),
        (5, 3, 5),
    ];
    for basis in [4usize, 8] {
        for (case, &(live_x_cols, col_bits, ring_bits)) in shapes.iter().enumerate() {
            let seed = 0x5a17_0000_u64 | ((basis as u64) << 8) | case as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let y_len = 1usize << ring_bits;
            let half = (basis / 2) as i8;
            let digit_witness = (0..live_x_cols * y_len)
                .map(|_| rng.gen_range(0..basis) as i8 - half)
                .collect::<Vec<_>>();
            let compact_range_image = build_compact_range_image(&digit_witness);
            let column_then_ring_tau = (0..col_bits + ring_bits)
                .map(|_| F::from_u64(rng.gen_range(2..u64::MAX)))
                .collect::<Vec<_>>();
            let tau = ordered_equality_point(&column_then_ring_tau, col_bits, ring_bits);
            let manual_challenges = (0..col_bits + ring_bits)
                .map(|_| F::from_u64(rng.gen_range(2..u64::MAX)))
                .collect::<Vec<_>>();

            let mut optimized = LowBasisRangeCheckProver::new(
                std::sync::Arc::from(digit_witness.as_slice()),
                &tau,
                DigitRangePlan::new(basis).unwrap(),
                live_x_cols,
                col_bits,
                ring_bits,
            )
            .unwrap();
            let mut scalar = ScalarStage1Reference::new(
                &digit_witness,
                &tau,
                basis,
                live_x_cols,
                col_bits,
                ring_bits,
            );

            for (round, &challenge) in manual_challenges.iter().enumerate() {
                let optimized_poly = optimized.compute_round_eq_factored(round);
                let scalar_poly = scalar.compute_round_eq_factored(round);
                assert_eq!(
                    optimized_poly, scalar_poly,
                    "round polynomial mismatch basis={basis} case={case} round={round}"
                );
                assert_eq!(
                    optimized.current_linear_factor_evals(),
                    scalar.current_linear_factor_evals(),
                    "linear factor mismatch basis={basis} case={case} round={round}"
                );

                optimized.ingest_challenge(round, challenge);
                scalar.ingest_challenge(round, challenge);
                if round == 2 {
                    match &optimized.range_image {
                        LowBasisRangeImageStorage::FoldedOctets(folded)
                            if live_x_cols < 1usize << col_bits =>
                        {
                            assert_folded_octets_match_scalar(
                                folded,
                                &scalar.range_image[..live_x_cols * (y_len / 8)],
                                &compact_range_image,
                                basis,
                                live_x_cols,
                                y_len,
                                [
                                    manual_challenges[0],
                                    manual_challenges[1],
                                    manual_challenges[2],
                                ],
                            );
                        }
                        LowBasisRangeImageStorage::Compact(_) => {
                            panic!("round three must leave compact digit storage")
                        }
                        LowBasisRangeImageStorage::FoldedOctets(_) => {
                            panic!("full-width round must not use the sparse folded state")
                        }
                        LowBasisRangeImageStorage::Materialized(actual)
                            if live_x_cols == 1usize << col_bits =>
                        {
                            assert_eq!(actual, &scalar.range_image);
                        }
                        LowBasisRangeImageStorage::Materialized(_) => {
                            panic!("round three must retain folded octet classes")
                        }
                    }
                } else if round >= 3 {
                    match &optimized.range_image {
                        LowBasisRangeImageStorage::Materialized(actual) => {
                            let live_len = actual.len();
                            assert_eq!(actual, &scalar.range_image[..live_len]);
                        }
                        LowBasisRangeImageStorage::Compact(_)
                        | LowBasisRangeImageStorage::FoldedOctets(_) => {
                            panic!("folded octets must materialize after the next fold")
                        }
                    }
                }
            }
            assert_eq!(
                optimized.final_range_image_eval(),
                scalar.final_range_image_eval(),
                "final evaluation mismatch basis={basis} case={case}"
            );

            let mut optimized_for_proof = LowBasisRangeCheckProver::new(
                std::sync::Arc::from(digit_witness.as_slice()),
                &tau,
                DigitRangePlan::new(basis).unwrap(),
                live_x_cols,
                col_bits,
                ring_bits,
            )
            .unwrap();
            let mut scalar_for_proof = ScalarStage1Reference::new(
                &digit_witness,
                &tau,
                basis,
                live_x_cols,
                col_bits,
                ring_bits,
            );
            let mut optimized_transcript = new_stage1_transcript();
            let mut scalar_transcript = new_stage1_transcript();
            let (optimized_proof, optimized_challenges, optimized_claim) = optimized_for_proof
                .prove::<F, _, _>(&mut optimized_transcript, sample_stage1_challenge)
                .unwrap();
            let (scalar_proof, scalar_challenges, scalar_claim) = scalar_for_proof
                .prove::<F, _, _>(&mut scalar_transcript, sample_stage1_challenge)
                .unwrap();
            assert_eq!(optimized_proof, scalar_proof);
            assert_eq!(optimized_challenges, scalar_challenges);
            assert_eq!(optimized_claim, scalar_claim);
            assert_eq!(
                optimized_for_proof.final_range_image_eval(),
                scalar_for_proof.final_range_image_eval()
            );

            let mut optimized_bytes = Vec::new();
            optimized_proof
                .serialize_compressed(&mut optimized_bytes)
                .unwrap();
            let mut scalar_bytes = Vec::new();
            scalar_proof
                .serialize_compressed(&mut scalar_bytes)
                .unwrap();
            assert_eq!(optimized_bytes, scalar_bytes);
            assert_eq!(
                optimized_transcript.challenge_scalar(b"stage1-differential-after-proof"),
                scalar_transcript.challenge_scalar(b"stage1-differential-after-proof")
            );
        }
    }
}

fn assert_basis8_exact_histogram_differential<B, E>()
where
    B: FieldCore + CanonicalField + CanonicalBytes + TranscriptChallenge + 'static,
    E: ExtField<B> + FromPrimitiveInt + HasOptimizedFold + HasUnreducedOps + AkitaSerialize,
{
    assert!(E::DELAYED_PRODUCT_SUM_IS_EXACT);
    #[cfg(feature = "parallel")]
    assert_eq!(rayon::current_num_threads(), 1);

    let basis = 8usize;
    let live_x_cols = 6usize;
    let col_bits = 3usize;
    let ring_bits = 4usize;
    let y_len = 1usize << ring_bits;
    let digit_witness = (0..live_x_cols * y_len)
        .map(|index| ((index * 13 + 5) % basis) as i8 - 4)
        .collect::<Vec<_>>();
    let column_then_ring_tau = (0..col_bits + ring_bits)
        .map(|index| E::from_u64(index as u64 * 17 + 3))
        .collect::<Vec<_>>();
    let tau = DigitRangeEqualityPoint::from_column_then_ring_challenges(
        &column_then_ring_tau,
        col_bits,
        ring_bits,
    )
    .unwrap()
    .into_coordinates();
    let manual_challenges = (0..col_bits + ring_bits)
        .map(|index| E::from_u64(index as u64 * 19 + 7))
        .collect::<Vec<_>>();

    let mut optimized = LowBasisRangeCheckProver::new(
        std::sync::Arc::from(digit_witness.as_slice()),
        &tau,
        DigitRangePlan::new(basis).unwrap(),
        live_x_cols,
        col_bits,
        ring_bits,
    )
    .unwrap();
    let mut scalar = ScalarStage1Reference::new(
        &digit_witness,
        &tau,
        basis,
        live_x_cols,
        col_bits,
        ring_bits,
    );

    for (round, &challenge) in manual_challenges.iter().enumerate() {
        assert_eq!(
            optimized.compute_round_eq_factored(round),
            scalar.compute_round_eq_factored(round),
            "exact-histogram round polynomial mismatch at round {round}"
        );
        assert_eq!(
            optimized.current_linear_factor_evals(),
            scalar.current_linear_factor_evals(),
            "exact-histogram linear factor mismatch at round {round}"
        );
        optimized.ingest_challenge(round, challenge);
        scalar.ingest_challenge(round, challenge);
    }
    assert_eq!(
        optimized.final_range_image_eval(),
        scalar.final_range_image_eval()
    );

    let mut optimized_for_proof = LowBasisRangeCheckProver::new(
        std::sync::Arc::from(digit_witness.as_slice()),
        &tau,
        DigitRangePlan::new(basis).unwrap(),
        live_x_cols,
        col_bits,
        ring_bits,
    )
    .unwrap();
    let mut scalar_for_proof = ScalarStage1Reference::new(
        &digit_witness,
        &tau,
        basis,
        live_x_cols,
        col_bits,
        ring_bits,
    );
    let mut optimized_transcript =
        AkitaTranscript::<B>::new(transcript_labels::DOMAIN_AKITA_PROTOCOL);
    let mut scalar_transcript = AkitaTranscript::<B>::new(transcript_labels::DOMAIN_AKITA_PROTOCOL);
    let (optimized_proof, optimized_challenges, optimized_claim) = optimized_for_proof
        .prove::<B, _, _>(&mut optimized_transcript, |transcript| {
            sample_ext_challenge::<B, E, _>(transcript, transcript_labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();
    let (scalar_proof, scalar_challenges, scalar_claim) = scalar_for_proof
        .prove::<B, _, _>(&mut scalar_transcript, |transcript| {
            sample_ext_challenge::<B, E, _>(transcript, transcript_labels::CHALLENGE_SUMCHECK_ROUND)
        })
        .unwrap();
    assert_eq!(optimized_proof, scalar_proof);
    assert_eq!(optimized_challenges, scalar_challenges);
    assert_eq!(optimized_claim, scalar_claim);
    assert_eq!(
        optimized_for_proof.final_range_image_eval(),
        scalar_for_proof.final_range_image_eval()
    );

    let mut optimized_bytes = Vec::new();
    optimized_proof
        .serialize_compressed(&mut optimized_bytes)
        .unwrap();
    let mut scalar_bytes = Vec::new();
    scalar_proof
        .serialize_compressed(&mut scalar_bytes)
        .unwrap();
    assert_eq!(optimized_bytes, scalar_bytes);
    assert_eq!(
        optimized_transcript.challenge_scalar(b"stage1-exact-differential-after-proof"),
        scalar_transcript.challenge_scalar(b"stage1-exact-differential-after-proof")
    );
}

fn assert_production_extension_histograms() {
    assert_basis8_exact_histogram_differential::<Prime32Offset99, FpExt4<Prime32Offset99>>();
    assert_basis8_exact_histogram_differential::<Prime64Offset59, FpExt2<Prime64Offset59, TwoNr>>();
}

#[test]
fn stage1_basis8_exact_histogram_matches_scalar_transcript_and_bytes() {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(assert_production_extension_histograms);

    #[cfg(not(feature = "parallel"))]
    assert_production_extension_histograms();
}
