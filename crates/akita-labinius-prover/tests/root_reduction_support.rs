#![cfg(feature = "labinius")]
#![allow(dead_code)]

#[path = "common/mod.rs"]
pub(crate) mod common;

use akita_algebra::{binary::BinaryField128, MinusTrinomial};
use akita_labinius_prover::{
    commit_binary_clear, lowered::flatten_image, prove_root_reduction_bytes,
    TransparentRootProverOracle,
};
use akita_labinius_verifier::{
    lowered::LoweredRootLayout,
    root::{verify_root_reduction_bytes, RootEvaluationClaims, TransparentRootVerifierOracle},
    AdmittedRootSetup, BinaryClearCommitment,
};
use akita_params::sis::labinius::{LabiniusDigitBase, LabiniusRootProfile};
use akita_types::proof::AkitaSetupSeed;
use common::{data, TestHost};

pub(crate) type F = jolt_field::Prime128OffsetA7F7;
pub(crate) type H = BinaryField128;
pub(crate) type Setup = AdmittedRootSetup<F, 648, MinusTrinomial>;
pub(crate) type Commitment = BinaryClearCommitment<F, 648, MinusTrinomial>;
pub(crate) const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128BoundedW46Delta16;
pub(crate) const BASES: [LabiniusDigitBase; 3] = [
    LabiniusDigitBase::Bits1,
    LabiniusDigitBase::Bits2,
    LabiniusDigitBase::Bits4,
];

pub(crate) fn admitted(fold: u32, seed: u8) -> Setup {
    Setup::derive(
        PROFILE,
        4,
        fold,
        128,
        AkitaSetupSeed::shake256_paged_v1([seed; 32]),
    )
    .unwrap()
}

pub(crate) struct Case<T: TestHost = H> {
    pub(crate) admitted: Setup,
    pub(crate) base: LabiniusDigitBase,
    pub(crate) source: Vec<T::Source>,
    pub(crate) point: Vec<T>,
    pub(crate) value: T,
    pub(crate) commitment: Commitment,
    pub(crate) layout: LoweredRootLayout,
    pub(crate) image: Vec<F>,
}

impl<T: TestHost> Case<T> {
    pub(crate) fn new(base: LabiniusDigitBase, fold: u32) -> Self {
        let admitted = admitted(fold, 0x31);
        let setup = admitted.setup();
        let (source, point, value) = data::<T>(setup.source_len(), setup.num_vars());
        let commitment = commit_binary_clear::<T, F, 648, MinusTrinomial>(setup, &source).unwrap();
        let layout = LoweredRootLayout::new(setup, admitted.shape(), base).unwrap();
        let image = flatten_image(&layout, &commitment).unwrap();
        Self {
            admitted,
            base,
            source,
            point,
            value,
            commitment,
            layout,
            image,
        }
    }

    pub(crate) fn prove(&self) -> (Vec<u8>, RootEvaluationClaims<F>, Vec<u8>) {
        let mut oracle = TransparentRootProverOracle::new(&self.image);
        let (proof, claims) = prove_root_reduction_bytes(
            &self.admitted,
            self.base,
            &self.source,
            &self.commitment,
            &self.point,
            self.value,
            &mut oracle,
        )
        .unwrap();
        (proof, claims, oracle.response().to_vec())
    }

    pub(crate) fn verify(
        &self,
        proof: &[u8],
    ) -> Result<RootEvaluationClaims<F>, akita_error::AkitaError> {
        let mut oracle = TransparentRootVerifierOracle::new(&self.image);
        verify_root_reduction_bytes(
            &self.admitted,
            self.base,
            &self.point,
            self.value,
            &mut oracle,
            proof,
        )
    }

    /// Message regions in the normative wire order; oracle public inputs are absent.
    pub(crate) fn regions(&self) -> Vec<(&'static str, usize, usize)> {
        let encoding = self.layout.encoding();
        let lengths = [
            (
                "frontend",
                T::ROWS * core::mem::size_of::<T::Source>() + (2 * self.point.len() + 1) * 21,
            ),
            ("U", self.layout.columns() * 21),
            ("W", self.layout.witness_len()),
            ("QA", self.layout.n_a() * 647 * 16),
            (
                "Q",
                encoding.parity_quotient_len() * (encoding.quotient().bits() as usize).div_ceil(8),
            ),
            (
                "K",
                encoding.parity_carry_len() * (encoding.carry().bits() as usize).div_ceil(8),
            ),
            ("y_Y", 16),
            (
                "combined",
                self.layout.witness_log_len() * ((1usize << self.base.bits()) + 1) * 16,
            ),
            ("w_eval", 16),
            ("product", self.layout.image_log_len() * 2 * 16),
            ("y_eval", 16),
        ];
        let mut start = 0;
        lengths
            .into_iter()
            .map(|(name, len)| {
                let region = (name, start, len);
                start += len;
                region
            })
            .collect()
    }
}
