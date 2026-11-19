use super::*;
use crate::{
    FoldChallengeDrawDomain, FoldChallengeFamily, FoldDraw, OperatorNormRejection,
    SparseChallengeConfig, D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};

struct Message(Vec<u8>);
impl FoldDraw for Message {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        assert_eq!(len, self.0.len());
        decode(&self.0)
    }
}

fn family(
    d: usize,
    config: SparseChallengeConfig,
    rejection: Option<OperatorNormRejection>,
) -> FoldChallengeFamily {
    FoldChallengeFamily::new(
        FoldChallengeDrawDomain::EvaluationTrace,
        d,
        config,
        rejection,
    )
    .unwrap()
}

#[test]
fn recorded_advice_replays_every_route_in_draw_order() {
    let production = |d| SparseChallengeConfig::production_for_ring_dim(d).unwrap();
    // Quartet, ternary, unfiltered position and operator-norm position routes.
    let draws = [
        (family(64, production(64), None), 1024),
        (family(128, production(128), None), 1024),
        (family(128, production(128), None), 600),
        (
            family(
                64,
                D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                Some(OperatorNormRejection::D64_SELECTIVE_L2),
            ),
            600,
        ),
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let messages = draws
        .iter()
        .map(|(family, count)| {
            (0..family.stream(*count).unwrap().message_len())
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state as u8
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let draw_all = || {
        draws
            .iter()
            .zip(&messages)
            .map(|((family, count), message)| {
                Message(message.clone())
                    .draw_fold_challenges(family, *count, 1)
                    .unwrap()
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let unadvised = draw_all();

    let (recorded, frame) = record_checked_hints(draw_all).unwrap();
    assert_eq!(recorded, unadvised);
    assert_eq!(mode(), Advice::Off);
    let tape_len =
        |offset: usize| u64::from_le_bytes(frame[offset..offset + 8].try_into().unwrap());
    assert_eq!(tape_len(0), 1024 * 32);
    assert_eq!(tape_len(8), 1024 * 32);
    assert!(frame.len() > 16 + 2 * 1024 * 32);

    install_checked_hints(&frame, frame.len()).unwrap();
    assert!(record_checked_hints(|| ()).is_err());
    assert_eq!(draw_all(), unadvised);
    finish_checked_hints().unwrap();
}
