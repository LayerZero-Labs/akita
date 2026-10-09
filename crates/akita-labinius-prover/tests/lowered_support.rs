#![cfg(feature = "labinius")]
#![allow(dead_code)]

#[path = "common/mod.rs"]
pub(crate) mod common;

use akita_algebra::{
    binary::{BinaryField128 as H, BinaryField162 as B},
    MinusTrinomial, TrinomialRing,
};
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear,
    lowered::{encode_witness, flatten_image, parity_quotient_and_carry},
};
use akita_labinius_verifier::{
    endpoint::{fold_integer, left_expansion, verify_endpoints},
    lowered::{LoweredChallenges, LoweredPublic, LoweredRootLayout},
    BinaryClearCommitment, BinaryClearSetup, BinaryEvaluationClaim,
};
use akita_params::sis::labinius::{LabiniusDigitBase, LabiniusRootProfile, LabiniusRootShape};
use common::{data, FixedDraw, Tape};
use jolt_field::{Ring, Zero};
use rand::{rngs::StdRng, RngCore, SeedableRng};

pub(crate) type F = jolt_field::Prime128OffsetA7F7;
pub(crate) type Setup = BinaryClearSetup<F, 648, MinusTrinomial>;
pub(crate) type Commitment = BinaryClearCommitment<F, 648, MinusTrinomial>;
pub(crate) const BASES: [LabiniusDigitBase; 3] = [
    LabiniusDigitBase::Bits1,
    LabiniusDigitBase::Bits2,
    LabiniusDigitBase::Bits4,
];
pub(crate) const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128BoundedW46Delta16;

pub(crate) fn setup(shape: &LabiniusRootShape) -> Setup {
    let mut rng = StdRng::seed_from_u64(0x215_197);
    let n = shape.rank_a() as usize;
    let m = shape.ring_elements_per_column();
    let matrix = (0..n * m)
        .map(|_| {
            TrinomialRing::from_coefficients(std::array::from_fn(|_| random_field(&mut rng)))
                .unwrap()
        })
        .collect();
    Setup::new(
        matrix,
        n,
        m,
        shape.fold_width(),
        -32768,
        32767,
        128,
        PROFILE.challenge_profile().unwrap(),
        PROFILE.coefficient_prime(),
        PROFILE.ring_degree(),
    )
    .unwrap()
}
pub(crate) fn random_field(rng: &mut StdRng) -> F {
    F::from_u128(u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64))
}
pub(crate) fn random_point(rng: &mut StdRng, len: usize) -> Vec<F> {
    (0..len).map(|_| random_field(rng)).collect()
}
pub(crate) fn boolean_point(index: usize, len: usize) -> Vec<F> {
    (0..len)
        .map(|bit| F::from_u64(((index >> bit) & 1) as u64))
        .collect()
}
pub(crate) fn signed(x: i128) -> F {
    if x < 0 {
        -F::from_u128(x.unsigned_abs())
    } else {
        F::from_u128(x as u128)
    }
}
pub(crate) fn dot(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::zero(), |sum, (&a, &b)| sum + a * b)
}
pub(crate) fn dot_digits(a: &[u8], b: &[F]) -> F {
    a.iter().zip(b).fold(F::zero(), |sum, (&a, &b)| {
        sum + F::from_u64(u64::from(a)) * b
    })
}

pub(crate) struct Case {
    pub(crate) shape: LabiniusRootShape,
    pub(crate) setup: Setup,
    pub(crate) layout: LoweredRootLayout,
    pub(crate) commitment: Commitment,
    pub(crate) claim: BinaryEvaluationClaim,
    pub(crate) u: Vec<B>,
    pub(crate) fold: Vec<BinaryChallenge>,
    pub(crate) response: Vec<[i64; 162]>,
    pub(crate) q: Vec<i128>,
    pub(crate) k: Vec<i128>,
    pub(crate) w: Vec<u8>,
    pub(crate) y: Vec<F>,
    pub(crate) challenges: LoweredChallenges<F>,
}
impl Case {
    pub(crate) fn new(base: LabiniusDigitBase) -> Self {
        let shape = LabiniusRootShape::derive(PROFILE, 4, 1, 128).unwrap();
        let setup = setup(&shape);
        Self::with_setup(base, shape, setup)
    }
    /// The same honest transcript over a caller-chosen admitted setup.
    pub(crate) fn with_setup(
        base: LabiniusDigitBase,
        shape: LabiniusRootShape,
        setup: Setup,
    ) -> Self {
        let (source, _, _) = data::<H>(setup.source_len(), setup.num_vars());
        let commitment = commit_binary_clear::<H, F, 648, MinusTrinomial>(&setup, &source).unwrap();
        let point = (0..setup.num_vars())
            .map(|i| Tape::draw_value(i + 8, 0))
            .collect::<Vec<_>>();
        let u = left_expansion::<H>(&source, &point, setup.scalar_rows(), setup.columns()).unwrap();
        let value = common::binary_mle(&u, &point[setup.row_vars()..]);
        let claim = BinaryEvaluationClaim { point, value };
        let fold = BinaryChallengeSampler::new(setup.profile().clone())
            .sample_challenges(&mut FixedDraw, b"lowered-oracle", setup.columns())
            .unwrap();
        let response = fold_integer::<H>(
            &source,
            setup.scalar_rows(),
            setup.columns(),
            &fold,
            setup.profile(),
        )
        .unwrap();
        let layout = LoweredRootLayout::new(&setup, &shape, base).unwrap();
        let (q, k) = parity_quotient_and_carry(&setup, &claim, &u, &fold, &response).unwrap();
        let w = encode_witness(&layout, &response).unwrap();
        let y = flatten_image(&layout, &commitment).unwrap();
        // All witness messages are fixed before drawing the three field challenges.
        let mut rng = StdRng::seed_from_u64(0xdec0de);
        let challenges = LoweredChallenges {
            alpha: random_field(&mut rng),
            xi: random_field(&mut rng),
            gamma: random_field(&mut rng),
        };
        Self {
            shape,
            setup,
            layout,
            commitment,
            claim,
            u,
            fold,
            response,
            q,
            k,
            w,
            y,
            challenges,
        }
    }
    pub(crate) fn verify(&self) -> Result<(), AkitaError> {
        verify_endpoints(
            &self.setup,
            &self.commitment,
            &self.claim,
            &self.u,
            &self.fold,
            &self.response,
        )
    }
    pub(crate) fn public(&self) -> LoweredPublic<F> {
        self.public_with(&self.claim, &self.u, &self.q, &self.k)
            .unwrap()
    }
    pub(crate) fn public_with(
        &self,
        claim: &BinaryEvaluationClaim,
        u: &[B],
        q: &[i128],
        k: &[i128],
    ) -> Result<LoweredPublic<F>, AkitaError> {
        LoweredPublic::new(
            &self.layout,
            &self.setup,
            claim,
            u,
            &self.fold,
            &[],
            q,
            k,
            self.challenges,
        )
    }
}
