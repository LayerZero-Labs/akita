//! The binary root family's draw through [`FoldChallengeDraw`].

use super::*;
use akita_challenges::{root_family, BinaryChallengeSampler};
use akita_params::FoldChallengeDistribution;

#[derive(Default)]
struct Draw {
    message_lens: Vec<usize>,
}

const MESSAGE: [u8; 32] = [0x5c; 32];

impl FoldDraw for Draw {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        self.message_lens.push(len);
        decode(&MESSAGE[..len.min(MESSAGE.len())])
    }
}

fn binary_root_family(ring_d: usize) -> Result<FoldChallengeFamily, AkitaError> {
    FoldChallengeFamily::new(
        FoldChallengeDrawDomain::EvaluationTrace,
        ring_d,
        FoldChallengeDistribution::BinaryRoot,
        None,
    )
}

#[test]
fn ordinary_group_root_selects_binary_family_and_indexed_columns() {
    let family = binary_root_family(162).unwrap();
    assert_eq!(family.distribution(), FoldChallengeDistribution::BinaryRoot);
    assert_eq!(family.message_len(8).unwrap(), 32);
    assert_eq!(family.set_bits(), 136);
    let mut draw = Draw::default();
    let challenges = draw.draw_fold_challenges(&family, 8, 1).unwrap().unwrap();
    assert_eq!(draw.message_lens, vec![32]);
    let reference = BinaryChallengeSampler::new(root_family::profile().unwrap())
        .sample_from_seed(&MESSAGE, 8)
        .unwrap();
    for (actual, expected) in challenges.as_slice().iter().zip(reference) {
        assert_eq!(actual.positions.len(), expected.weight());
        for ((position, coefficient), term) in actual
            .positions
            .iter()
            .zip(&actual.coeffs)
            .zip(expected.terms())
        {
            assert_eq!(*position, u32::from(term.position));
            assert_eq!(*coefficient, term.coefficient);
        }
    }
    let again = Draw::default()
        .draw_fold_challenges(&family, 8, 1)
        .unwrap()
        .unwrap();
    assert_eq!(challenges, again);
    assert!(binary_root_family(486).is_err());
    assert!(FoldChallengeFamily::new(
        FoldChallengeDrawDomain::EvaluationTrace,
        162,
        FoldChallengeDistribution::BinaryRoot,
        Some(jolt_crypto::challenge::OperatorNormRejection::D64_SELECTIVE_L2),
    )
    .is_err());
    assert!(family.stream(8).is_err());
    assert!(root_family::sample(&[0; 32], (1 << 20) + 1).is_err());
}


#[test]
fn binary_family_replays_on_the_callers_live_session() {
    use akita_transcript::{new_prover_channel, new_verifier_channel};
    let family = binary_root_family(162).unwrap();
    let mut prover = new_prover_channel(b"root-family/session");
    let expected = ProverFoldDraw::new(&mut prover, 0, 3)
        .draw_fold_challenges(&family, 8, 1)
        .unwrap();
    let proof = prover.narg_string().to_vec();
    let mut verifier = new_verifier_channel(b"root-family/session", &proof);
    let actual = VerifierFoldDraw::new(&mut verifier, 0, 3)
        .draw_fold_challenges(&family, 8, 1)
        .unwrap();
    assert_eq!(expected, actual);
    verifier.check_eof().unwrap();
}

