#![no_main]

use akita_sumcheck::{CompressedUniPoly, SumcheckProof};
use akita_transcript::{AkitaTranscript, Transcript};
use jolt_field::{Prime128Offset275 as F, Ring};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let [rounds, degree, claim, data @ ..] = data else {
        return;
    };
    let num_rounds = usize::from(rounds % 5);
    let degree_bound = usize::from(degree % 5);
    let proof = SumcheckProof {
        // Bound both the number and width of typed round messages. A chunk's
        // first byte selects its stored coefficient count, including zero.
        round_polys: data
            .chunks_exact(6)
            .take(5)
            .map(|chunk| CompressedUniPoly {
                coeffs_except_linear_term: chunk[1..]
                    .iter()
                    .take(usize::from(chunk[0] % 6))
                    .map(|&coefficient| F::from_u64(u64::from(coefficient)))
                    .collect(),
            })
            .collect(),
    };
    let valid_shape = proof.round_polys.len() == num_rounds
        && proof
            .round_polys
            .iter()
            .all(|poly| (1..=degree_bound).contains(&poly.coeffs_except_linear_term.len()));
    let mut transcript = AkitaTranscript::<F>::new(b"fuzz/sumcheck-rounds");
    let mut samples = 0;
    let result = proof.verify::<F, _, _>(
        F::from_u64(u64::from(*claim)),
        num_rounds,
        degree_bound,
        &mut transcript,
        |tr| {
            samples += 1;
            Ok(tr.challenge_scalar(b"round"))
        },
    );
    assert_eq!(result.is_ok(), valid_shape);
    if valid_shape {
        assert_eq!(samples, num_rounds);
    } else {
        assert_eq!(samples, 0);
        assert_eq!(
            transcript.challenge_bytes(b"state", 32),
            AkitaTranscript::<F>::new(b"fuzz/sumcheck-rounds").challenge_bytes(b"state", 32)
        );
    }
});
