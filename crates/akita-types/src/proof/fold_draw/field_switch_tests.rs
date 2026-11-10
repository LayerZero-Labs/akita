//! Frozen main #441 field-switch evaluations through preview’s shared Jolt decoder.
use super::*;
use akita_error::AkitaError;
use jolt_algebra::{Prime128Offset6A23801F as F, Ring};
use jolt_crypto::challenge::SparseChallengeConfig;
struct Message(Vec<u8>);
impl FoldDraw for Message {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        assert_eq!(len, self.0.len() - 1);
        decode(&self.0[1..])
    }
}
fn family() -> FoldChallengeFamily {
    FoldChallengeFamily::new(
        FoldChallengeDrawDomain::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        },
        64,
        SparseChallengeConfig {
            count_pm1: 31,
            count_pm2: 10,
        },
        None,
    )
    .unwrap()
}
fn message() -> Message {
    let mut bytes = vec![0];
    bytes.extend_from_slice(MESSAGE);
    Message(bytes)
}
#[test]
fn shared_sampler_matches_main_field_switch_fixture() {
    let sampled = message()
        .draw_fold_challenges(&family(), 1024, 1)
        .unwrap()
        .unwrap();
    for (i, stride) in [1, 4].into_iter().enumerate() {
        let batch = sampled
            .embed_subring_positions(64, stride, 64 * stride)
            .unwrap();
        let mut p = F::from_u64(1);
        let powers = (0..64 * stride)
            .map(|_| {
                let v = p;
                p *= F::from_u64(37);
                v
            })
            .collect::<Vec<_>>();
        let values = batch.evals_at_pows::<F, F>(&powers).unwrap();
        assert_eq!(values[0], F::from_u128(EXPECTED[i][0]));
        assert_eq!(values[1023], F::from_u128(EXPECTED[i][1]));
        assert_eq!(
            values.iter().copied().fold(F::from_u64(0), |a, b| a + b),
            F::from_u128(EXPECTED[i][2])
        );
    }
}

const MESSAGE: &[u8] = include_bytes!("field_switch_message.bin");
const EXPECTED: [[u128; 3]; 2] = [
    [
        40544065902344454397037691921406379642u128,
        148436690822564892961175858852520944655u128,
        270681290819792749220569730658856053299u128,
    ],
    [
        91275897945596252000852350436102186229u128,
        92391489860930062131740839803393956855u128,
        161059714558888528037951495051275981897u128,
    ],
];
