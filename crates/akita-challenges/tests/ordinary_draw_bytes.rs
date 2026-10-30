use akita_challenges::{FoldChallengeDrawDomain, FoldDraw, SparseChallengeConfig};
use akita_error::AkitaError;

#[derive(Default)]
struct CapturingDraw(Vec<u8>);

impl FoldDraw for CapturingDraw {
    fn absorb_and_squeeze(&mut self, payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        self.0.extend_from_slice(payload);
        Ok([7; 32])
    }
}

#[test]
fn ordinary_draw_matches_pre_seam_bytes() {
    let mut draw = CapturingDraw::default();
    let challenges = draw
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            64,
            2,
            2,
            1,
            &SparseChallengeConfig {
                count_pm1: 2,
                count_pm2: 1,
            },
            None,
        )
        .unwrap();
    // Fixed legacy context, independent of the context-building implementation.
    let expected = concat!(
        "616b6974612f666f6c642d6368616c6c656e67652d726f756e642f7631",
        "020000000000000002000000000000000100000000000000616b2f632f7766",
        "020000000000000040000000000000000002000000000000000100000000000000"
    );
    let actual: String = draw.0.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(actual, expected);
    // SHAKE256([7; 32] || index_le64), partial Fisher–Yates, then sign bytes.
    let sampled = challenges.as_slice();
    assert_eq!(sampled[0].positions.as_slice(), &[24, 2, 54]);
    assert_eq!(sampled[0].coeffs.as_slice(), &[1, -1, 2]);
    assert_eq!(sampled[1].positions.as_slice(), &[0, 35, 42]);
    assert_eq!(sampled[1].coeffs.as_slice(), &[1, -1, -2]);
}
