#![allow(dead_code)]

use akita_algebra::binary::{field_switch::SwitchField, BinaryField128, BinaryField192};
use akita_config::{
    policy_of, CommitmentConfig, SetupRequirements, TrustedScheduleCatalog,
    ValidatedScheduleCatalog,
};
use akita_cpu_backend::AkitaProverSetup;
use akita_error::AkitaError;
use akita_labinius_pcs::{
    config::{DigitConfig, Digits1, Digits2, Digits4},
    ImageCommitOutput, ImageConfig, RootPcsProver, RootPcsVerifier, RootSetup, F,
};
use akita_labinius_verifier::lowered::LoweredRootLayout;
use akita_params::{
    CommittedGroupBatchProfile, GroupCommitPhaseParams, PlannedFoldSchedule, PolynomialGroupLayout,
    ScheduleLookupKey,
};
use akita_planner::emit::{GroupedGenerationRequest, PrecommittedProducer};
use akita_types::AkitaVerifierSetup;
use std::sync::OnceLock;

pub(crate) trait Host: SwitchField<Source: Sync> {
    fn source(len: usize) -> Vec<Self::Source>;
    fn point(len: usize) -> Vec<Self>;
}
impl Host for BinaryField128 {
    fn source(len: usize) -> Vec<u128> {
        (0..len)
            .map(|i| (i as u128 + 1).wrapping_mul(0x1234_5678_9abc_def0_2151))
            .collect()
    }
    fn point(len: usize) -> Vec<Self> {
        (0..len)
            .map(|i| Self::from_words([i as u64 + 17, 0x514a_89ff_7231]))
            .collect()
    }
}
impl Host for BinaryField192 {
    fn source(len: usize) -> Vec<u64> {
        (0..len)
            .map(|i| (i as u64 + 1).wrapping_mul(0x2151_978a_5713_2ab1))
            .collect()
    }
    fn point(len: usize) -> Vec<Self> {
        (0..len)
            .map(|i| Self::from_words([i as u64 + 17, 0x514a_89ff_7231, 0x00a1_376a]))
            .collect()
    }
}

pub(crate) fn evaluate<H: SwitchField>(source: &[H::Source], point: &[H]) -> H {
    let mut table = source
        .iter()
        .copied()
        .map(H::embed_source)
        .collect::<Vec<_>>();
    for &coordinate in point {
        for index in 0..table.len() / 2 {
            table[index] =
                table[2 * index] * (H::ONE + coordinate) + table[2 * index + 1] * coordinate;
        }
        table.truncate(table.len() / 2);
    }
    assert_eq!(table.len(), 1);
    table[0]
}

fn admit<C: CommitmentConfig>(
    key: &ScheduleLookupKey,
    planned: PlannedFoldSchedule,
) -> TrustedScheduleCatalog<C> {
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &planned.schedule.root.params,
        )
        .unwrap(),
        precommitteds: key.precommitteds.clone(),
    };
    TrustedScheduleCatalog::new(
        ValidatedScheduleCatalog::try_new(
            C::schedule_family_name(),
            [(profiles, planned.schedule)],
            &policy_of::<C>(),
            C::ring_challenge_config,
        )
        .unwrap(),
    )
    .unwrap()
}

pub(crate) struct Fixture<C: DigitConfig> {
    pub(crate) images: TrustedScheduleCatalog<ImageConfig>,
    pub(crate) digits: TrustedScheduleCatalog<C>,
    pub(crate) prover_setup: AkitaProverSetup<F>,
    pub(crate) verifier_setup: AkitaVerifierSetup<F>,
}
impl<C: DigitConfig> Fixture<C> {
    pub(crate) fn catalogs(
        root: &RootSetup,
    ) -> (
        TrustedScheduleCatalog<ImageConfig>,
        TrustedScheduleCatalog<C>,
    ) {
        let layout = LoweredRootLayout::new(root.setup(), root.shape(), C::BASE).unwrap();
        let image_log = layout.image_log_len();
        static IMAGE_SMALL: OnceLock<TrustedScheduleCatalog<ImageConfig>> = OnceLock::new();
        static IMAGE_LARGE: OnceLock<TrustedScheduleCatalog<ImageConfig>> = OnceLock::new();
        let images = match image_log {
            10 => IMAGE_SMALL
                .get_or_init(|| crate::common::catalog(image_log))
                .clone(),
            11 => IMAGE_LARGE
                .get_or_init(|| crate::common::catalog(image_log))
                .clone(),
            _ => crate::common::catalog(image_log),
        };
        let image_key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(image_log));
        let producer = PrecommittedProducer::try_new(
            images
                .resolve_key(&image_key)
                .unwrap()
                .profiles()
                .final_group,
            ImageConfig::committed_source_contract().unwrap(),
        )
        .unwrap();
        let scalar_key =
            ScheduleLookupKey::single(PolynomialGroupLayout::singleton(layout.witness_log_len()));
        let scalar = akita_planner::find_schedule(
            &scalar_key,
            C::committed_source_contract().unwrap(),
            &[],
            &policy_of::<C>(),
            C::ring_challenge_config,
        )
        .unwrap();
        let scalar_catalog = admit::<C>(&scalar_key, scalar);
        let request = GroupedGenerationRequest::new(scalar_key.final_group, vec![producer]);
        let grouped = akita_planner::find_adapted_schedule(
            scalar_catalog.resolve_key(&scalar_key).unwrap(),
            &request,
            C::committed_source_contract().unwrap(),
            &policy_of::<C>(),
            C::ring_challenge_config,
        )
        .unwrap();
        (images, admit::<C>(&request.key(), grouped))
    }
    pub(crate) fn new(root: &RootSetup) -> Self {
        let (images, digits) = Self::catalogs(root);
        Self::from_catalogs(root, images, digits)
    }
    pub(crate) fn from_catalogs(
        root: &RootSetup,
        images: TrustedScheduleCatalog<ImageConfig>,
        digits: TrustedScheduleCatalog<C>,
    ) -> Self {
        let layout = LoweredRootLayout::new(root.setup(), root.shape(), C::BASE).unwrap();
        let max = layout.image_log_len().max(layout.witness_log_len());
        let requirements = SetupRequirements::from_catalog(&images, max, 2)
            .unwrap()
            .union(SetupRequirements::from_catalog(&digits, max, 2).unwrap())
            .unwrap();
        let prover_setup = akita_pcs::new_prover_setup(&requirements).unwrap();
        let verifier_setup = prover_setup
            .to_verifier_setup(requirements.matrix_capacity())
            .unwrap();
        Self {
            images,
            digits,
            prover_setup,
            verifier_setup,
        }
    }
    pub(crate) fn prover(&self, root: RootSetup) -> RootPcsProver<C> {
        RootPcsProver::new(
            root,
            self.images.clone(),
            self.digits.clone(),
            self.prover_setup.clone(),
        )
        .unwrap()
    }
    pub(crate) fn verifier(&self, root: RootSetup) -> RootPcsVerifier<C> {
        RootPcsVerifier::new(
            root,
            self.images.clone(),
            self.digits.clone(),
            self.verifier_setup.clone(),
        )
        .unwrap()
    }
}

pub(crate) trait TestDigits: DigitConfig {
    fn fixture(fold: u32) -> &'static Fixture<Self>;
}
macro_rules! cached {
    ($config:ty) => {
        impl TestDigits for $config {
            fn fixture(fold: u32) -> &'static Fixture<Self> {
                static ZERO: OnceLock<Fixture<$config>> = OnceLock::new();
                static ONE: OnceLock<Fixture<$config>> = OnceLock::new();
                match fold {
                    0 => ZERO.get_or_init(|| Fixture::new(&crate::common::admitted(0, 0x31))),
                    1 => ONE.get_or_init(|| Fixture::new(&crate::common::admitted(1, 0x31))),
                    _ => panic!("unexpected fold {fold}"),
                }
            }
        }
    };
}
cached!(Digits1);
cached!(Digits2);
cached!(Digits4);

pub(crate) struct Case<C: TestDigits, H: Host = BinaryField128> {
    pub(crate) root: RootSetup,
    pub(crate) source: Vec<H::Source>,
    pub(crate) point: Vec<H>,
    pub(crate) value: H,
    pub(crate) layout: LoweredRootLayout,
    pub(crate) output: ImageCommitOutput,
    pub(crate) prover: RootPcsProver<C>,
    pub(crate) verifier: RootPcsVerifier<C>,
}
impl<C: TestDigits, H: Host> Case<C, H> {
    pub(crate) fn new(fold: u32) -> Self {
        let root = crate::common::admitted(fold, 0x31);
        let fixture = C::fixture(fold);
        let prover = fixture.prover(root.clone());
        let verifier = fixture.verifier(root.clone());
        let source = H::source(root.setup().source_len());
        let point = H::point(root.setup().num_vars());
        let value = evaluate::<H>(&source, &point);
        let output = prover.commit::<H>(&source).unwrap();
        let layout = LoweredRootLayout::new(root.setup(), root.shape(), C::BASE).unwrap();
        Self {
            root,
            source,
            point,
            value,
            layout,
            output,
            prover,
            verifier,
        }
    }
    pub(crate) fn prove(&self) -> Vec<u8> {
        self.prover
            .open::<H>(&self.source, &self.output, &self.point, self.value)
            .unwrap()
    }
    pub(crate) fn verify(&self, proof: &[u8]) -> Result<(), AkitaError> {
        self.verifier
            .verify::<H>(&self.output.committed_group, &self.point, self.value, proof)
    }
    pub(crate) fn regions(&self, proof: &[u8]) -> Vec<(&'static str, usize, usize)> {
        let frontend =
            H::ROWS * core::mem::size_of::<H::Source>() + (2 * self.point.len() + 1) * 21;
        let response_length = frontend + self.layout.columns() * 21;
        let commitment_len = u64::from_le_bytes(
            proof[response_length..response_length + 8]
                .try_into()
                .unwrap(),
        ) as usize;
        let encoding = self.layout.encoding();
        let sizes = [
            (
                "frontend partials",
                H::ROWS * core::mem::size_of::<H::Source>(),
            ),
            ("frontend rounds", 2 * self.point.len() * 21),
            ("frontend terminal", 21),
            ("U", self.layout.columns() * 21),
            ("W length", 8),
            ("W commitment", commitment_len),
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
                self.layout.witness_log_len() * ((1usize << C::BASE.bits()) + 1) * 16,
            ),
            ("w_eval", 16),
            ("product", self.layout.image_log_len() * 2 * 16),
            ("y_eval", 16),
            ("inner length", 8),
        ];
        let mut start = 0;
        let mut regions = sizes
            .into_iter()
            .map(|(name, len)| {
                let region = (name, start, len);
                start += len;
                region
            })
            .collect::<Vec<_>>();
        let inner = u64::from_le_bytes(proof[start - 8..start].try_into().unwrap()) as usize;
        regions.push(("inner proof", start, inner));
        assert_eq!(start + inner, proof.len());
        regions
    }
}
