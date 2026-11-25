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
    let draws = [
        (family(128, production(128), None), 600),
        (family(64, production(64), None), 1024),
        (family(128, production(128), None), 1024),
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

    let (recorded, tape) = record_checked_hints(|| Ok(draw_all())).unwrap();
    assert_eq!(recorded, unadvised);
    assert_eq!(mode(), Advice::Off);
    // The first draw has 600 position records of eight words each.
    let start = 600 * 8 * size_of::<u64>();
    let quartet_bytes = 1024 * 4 * size_of::<u64>();
    let mut draw = Message(messages[1].clone());
    install_checked_hints(&tape[start..start + quartet_bytes], quartet_bytes).unwrap();
    assert_eq!(
        draw.draw_fold_challenges(&draws[1].0, 1024, 1).unwrap(),
        Some(unadvised[1].clone())
    );
    finish_checked_hints().unwrap();

    install_checked_hints(&tape, tape.len()).unwrap();
    assert!(record_checked_hints(|| Ok(())).is_err());
    assert_eq!(draw_all(), unadvised);
    finish_checked_hints().unwrap();
}

#[test]
fn recording_is_scoped_to_the_verification_call() {
    let cause = AkitaError::InvalidProof;
    let result = record_checked_hints::<()>(|| {
        assert!(install_checked_hints(&[], 0).is_err());
        assert!(finish_checked_hints().is_err());
        assert!(abort_checked_hints().is_err());
        assert!(record_checked_hints(|| Ok(())).is_err());
        record([1].into_iter())?;
        Err(cause.clone())
    });
    assert_eq!(result.unwrap_err(), cause);
    assert_eq!(mode(), Advice::Off);

    let panicked =
        std::panic::catch_unwind(|| record_checked_hints::<()>(|| panic!("test unwind")));
    assert!(panicked.is_err());
    let ((), tape) = record_checked_hints(|| Ok(())).unwrap();
    assert!(tape.is_empty());
}
