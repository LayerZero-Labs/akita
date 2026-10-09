#![cfg(feature = "labinius")]
#![allow(dead_code)]

use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField128, BinaryField162, BinaryField192},
    Prime64Offset23703, SmoothFftField, TrinomialModulus, TrinomialRing,
};
use akita_challenges::{
    BinaryChallenge, BinaryChallengeProfile, BinaryChallengeSampler, BinaryScalarRing, FoldDraw,
};
use akita_error::AkitaError;
use akita_labinius_verifier::{channel::ClearChannel, profile::BinaryClearSetup};
use akita_params::sis::labinius::{LabiniusCoefficientPrime, LabiniusRingDegree};
use jolt_field::Prime128OffsetA7F7;
use rand::{rngs::StdRng, RngCore, SeedableRng};

pub(crate) trait TestHost: SwitchField {
    fn random(rng: &mut StdRng) -> Self;
    fn random_source(rng: &mut StdRng) -> Self::Source;
}
impl TestHost for BinaryField128 {
    fn random(rng: &mut StdRng) -> Self {
        Self::from_words([rng.next_u64(), rng.next_u64()])
    }
    fn random_source(rng: &mut StdRng) -> Self::Source {
        u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64)
    }
}
impl TestHost for BinaryField192 {
    fn random(rng: &mut StdRng) -> Self {
        Self::from_words([rng.next_u64(), rng.next_u64(), rng.next_u64()])
    }
    fn random_source(rng: &mut StdRng) -> Self::Source {
        rng.next_u64()
    }
}
pub(crate) trait TestPrime: SmoothFftField {
    const ID: LabiniusCoefficientPrime;
}
impl TestPrime for Prime64Offset23703 {
    const ID: LabiniusCoefficientPrime = LabiniusCoefficientPrime::P64Offset23703;
}
impl TestPrime for Prime128OffsetA7F7 {
    const ID: LabiniusCoefficientPrime = LabiniusCoefficientPrime::P128OffsetA7F7;
}

pub(crate) fn profile() -> BinaryChallengeProfile {
    BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 47).unwrap()
}
pub(crate) fn setup<F: TestPrime, const D: usize, M: TrinomialModulus>(
    n_a: usize,
    m: usize,
    columns: usize,
) -> BinaryClearSetup<F, D, M> {
    let mut rng = StdRng::seed_from_u64(0x215_197);
    let matrix = (0..n_a * m)
        .map(|_| {
            TrinomialRing::from_coefficients(std::array::from_fn(|_| F::from_u64(rng.next_u64())))
                .unwrap()
        })
        .collect();
    BinaryClearSetup::new(
        matrix,
        n_a,
        m,
        columns,
        -1024,
        1024,
        128,
        profile(),
        F::ID,
        match D {
            162 => LabiniusRingDegree::D162,
            324 => LabiniusRingDegree::D324,
            648 => LabiniusRingDegree::D648,
            _ => panic!("test degree"),
        },
    )
    .unwrap()
}
pub(crate) fn data<H: TestHost>(len: usize, variables: usize) -> (Vec<H::Source>, Vec<H>, H) {
    let mut rng = StdRng::seed_from_u64(0xabc_571);
    let source = (0..len)
        .map(|_| H::random_source(&mut rng))
        .collect::<Vec<_>>();
    let point = (0..variables)
        .map(|_| H::random(&mut rng))
        .collect::<Vec<_>>();
    let claim = host_mle::<H>(&source, &point);
    (source, point, claim)
}
pub(crate) fn host_mle<H: SwitchField>(source: &[H::Source], point: &[H]) -> H {
    source
        .iter()
        .enumerate()
        .fold(H::ZERO, |sum, (index, &word)| {
            let weight = point.iter().enumerate().fold(H::ONE, |acc, (axis, &r)| {
                acc * if index >> axis & 1 == 1 {
                    r
                } else {
                    H::ONE + r
                }
            });
            sum + weight * H::embed_source(word)
        })
}
pub(crate) fn binary_mle(values: &[BinaryField162], point: &[BinaryField162]) -> BinaryField162 {
    values
        .iter()
        .enumerate()
        .fold(BinaryField162::ZERO, |sum, (index, &value)| {
            let weight = point
                .iter()
                .enumerate()
                .fold(BinaryField162::ONE, |acc, (axis, &r)| {
                    acc * if index >> axis & 1 == 1 {
                        r
                    } else {
                        BinaryField162::ONE + r
                    }
                });
            sum + weight * value
        })
}

pub(crate) struct FixedDraw;
impl FoldDraw for FixedDraw {
    fn absorb_and_squeeze(&mut self, _payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        Ok([0x53; 32])
    }
}

/// A deterministic channel for arithmetic oracles and isolated endpoint tests.
/// Its message replay does not stand in for Fiat-Shamir transcript tests.
#[derive(Default)]
pub(crate) struct Tape {
    pub(crate) messages: Vec<Vec<u8>>,
    pub(crate) replay: bool,
    pub(crate) message_cursor: usize,
    pub(crate) draws: usize,
    pub(crate) zero_batch: usize,
}
impl Tape {
    pub(crate) fn verifier(messages: Vec<Vec<u8>>, zero_batch: usize) -> Self {
        Self {
            messages,
            replay: true,
            zero_batch,
            ..Self::default()
        }
    }
    pub(crate) fn draw_value(index: usize, zero_batch: usize) -> BinaryField162 {
        if index < zero_batch {
            BinaryField162::ONE
        } else {
            BinaryField162::from_words([
                0x1234_5678 ^ index as u64,
                0xabc0_4321 ^ (index as u64 * 17),
                0x2f12 ^ index as u64,
            ])
            .unwrap()
        }
    }
}
impl ClearChannel for Tape {
    fn public(&mut self, _bytes: &[u8]) -> Result<(), AkitaError> {
        Ok(())
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        if self.replay {
            let message = self
                .messages
                .get(self.message_cursor)
                .ok_or(AkitaError::InvalidProof)?;
            if message.len() != bytes.len() {
                return Err(AkitaError::InvalidProof);
            }
            bytes.copy_from_slice(message);
            self.message_cursor += 1;
        } else {
            self.messages.push(bytes.to_vec());
        }
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        let value = Self::draw_value(self.draws, self.zero_batch);
        self.draws += 1;
        let mut bytes = [0u8; 32];
        bytes[..21].copy_from_slice(&value.to_bytes());
        Ok(bytes)
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        sampler.sample_challenges(&mut FixedDraw, label, count)
    }
}
