#![allow(dead_code)]

use akita_algebra::binary::{field_switch::SwitchField, BinaryField128, BinaryField192};
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog, ValidatedScheduleCatalog};
use akita_cpu_backend::AkitaProverSetup;
use akita_error::AkitaError;
use akita_labinius_pcs::{
    ImageCommitOutput, ImageConfig, ImageEvaluation, ImageProver, ImageVerifier, PreparedMatrix,
    RootSetup, F,
};
use akita_labinius_prover::lowered::flatten_image;
use akita_labinius_verifier::{channel::ClearChannel, lowered::LoweredRootLayout};
use akita_params::{
    sis::labinius::{LabiniusDigitBase, LabiniusRootProfile},
    CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout, ScheduleLookupKey,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_types::{proof::AkitaSetupSeed, AkitaVerifierSetup};
use jolt_field::Ring;
use std::sync::OnceLock;

pub(crate) const PREFIX: &[u8] = b"image-pcs-test-parent-prefix";
pub(crate) const INNER_DOMAIN: &[u8] = b"akita/labinius/image-akita-opening-session/v1";
pub(crate) const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128BoundedW46Delta16;
pub(crate) const BASES: [LabiniusDigitBase; 3] = [
    LabiniusDigitBase::Bits1,
    LabiniusDigitBase::Bits2,
    LabiniusDigitBase::Bits4,
];

pub(crate) trait Host: SwitchField<Source: Sync> {
    fn source(len: usize) -> Vec<Self::Source>;
}
impl Host for BinaryField128 {
    fn source(len: usize) -> Vec<u128> {
        (0..len)
            .map(|i| (i as u128 + 1) * 0x1234_5678_9abc_def0_2151)
            .collect()
    }
}
impl Host for BinaryField192 {
    fn source(len: usize) -> Vec<u64> {
        (0..len)
            .map(|i| (i as u64 + 1).wrapping_mul(0x2151_978a_5713_2ab1))
            .collect()
    }
}

pub(crate) fn admitted(fold: u32, seed: u8) -> RootSetup {
    RootSetup::derive(
        PROFILE,
        4,
        fold,
        128,
        AkitaSetupSeed::shake256_paged_v1([seed; 32]),
    )
    .unwrap()
}

pub(crate) fn catalog(log: usize) -> TrustedScheduleCatalog<ImageConfig> {
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(log));
    let policy = policy_of::<ImageConfig>();
    let planned = akita_planner::find_schedule(
        &key,
        ImageConfig::committed_source_contract().unwrap(),
        &[],
        &policy,
        ImageConfig::ring_challenge_config,
    )
    .unwrap();
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &planned.schedule.root.params,
        )
        .unwrap(),
        precommitteds: key.precommitteds.clone(),
    };
    let validated = ValidatedScheduleCatalog::try_new(
        ImageConfig::schedule_family_name(),
        [(profiles, planned.schedule)],
        &policy,
        ImageConfig::ring_challenge_config,
    )
    .unwrap();
    TrustedScheduleCatalog::new(validated).unwrap()
}

pub(crate) struct Fixture {
    pub(crate) scheme: AkitaCommitmentScheme<ImageConfig>,
    pub(crate) prover_setup: AkitaProverSetup<F>,
    pub(crate) verifier_setup: AkitaVerifierSetup<F>,
}
impl Fixture {
    fn new(log: usize) -> Self {
        let scheme = AkitaCommitmentScheme::new(catalog(log));
        let prover_setup = scheme.setup_prover(log, 1).unwrap();
        let verifier_setup = scheme.setup_verifier(&prover_setup).unwrap();
        Self {
            scheme,
            prover_setup,
            verifier_setup,
        }
    }
}
pub(crate) fn fixture(log: usize) -> &'static Fixture {
    static SMALL: OnceLock<Fixture> = OnceLock::new();
    static LARGE: OnceLock<Fixture> = OnceLock::new();
    match log {
        10 => SMALL.get_or_init(|| Fixture::new(log)),
        11 => LARGE.get_or_init(|| Fixture::new(log)),
        _ => panic!("unexpected test image log {log}"),
    }
}

pub(crate) struct Case<H: Host = BinaryField128> {
    pub(crate) root: RootSetup,
    pub(crate) source: Vec<H::Source>,
    pub(crate) layout: LoweredRootLayout,
    pub(crate) output: ImageCommitOutput,
    pub(crate) table: Vec<F>,
    pub(crate) point: Vec<F>,
    pub(crate) value: F,
    pub(crate) prover: ImageProver,
    pub(crate) verifier: ImageVerifier,
}
impl<H: Host> Case<H> {
    pub(crate) fn new(fold: u32) -> Self {
        let root = admitted(fold, 0x31);
        let layout = LoweredRootLayout::new(root.setup(), root.shape(), BASES[0]).unwrap();
        let f = fixture(layout.image_log_len());
        let prover =
            ImageProver::new(f.scheme.schedules().clone(), f.prover_setup.clone()).unwrap();
        let verifier =
            ImageVerifier::new(f.scheme.schedules().clone(), f.verifier_setup.clone()).unwrap();
        let source = H::source(root.setup().source_len());
        let prepared = PreparedMatrix::prepare(root.setup()).unwrap();
        let output = prover.commit::<H>(&root, &prepared, &source).unwrap();
        let table = flatten_image(&layout, &output.image).unwrap();
        let point = (0..layout.image_log_len())
            .map(|i| F::from_u64(i as u64 + 2))
            .collect::<Vec<_>>();
        let value = akita_algebra::poly::multilinear_eval(&table, &point).unwrap();
        Self {
            root,
            source,
            layout,
            output,
            table,
            point,
            value,
            prover,
            verifier,
        }
    }
    pub(crate) fn evaluation(&self) -> ImageEvaluation<'_> {
        ImageEvaluation {
            point: &self.point,
            value: self.value,
        }
    }
    pub(crate) fn prove(&self, prefix: &[u8]) -> (Vec<u8>, [u8; 32], [u8; 32]) {
        let state =
            akita_transcript::new_prover_channel(b"image-tests-parent/v1", b"context").unwrap();
        let mut record = Recording::new(state);
        record.public(prefix).unwrap();
        let started = std::time::Instant::now();
        self.prover
            .open_on_channel::<H, _>(&self.root, &self.output, self.evaluation(), &mut record)
            .unwrap();
        eprintln!(
            "image log {}: inner proof {} bytes, opening {:?}",
            self.layout.image_log_len(),
            record.messages[1].len(),
            started.elapsed()
        );
        assert_eq!(record.draws.len(), 1);
        assert_eq!(record.messages.len(), 2);
        let seed = record.draws[0];
        let after = record.challenge_block().unwrap();
        (record.inner.narg_string().to_vec(), seed, after)
    }
    pub(crate) fn verify(&self, proof: &[u8], prefix: &[u8]) -> Result<[u8; 32], AkitaError> {
        let mut state =
            akita_transcript::new_verifier_channel(b"image-tests-parent/v1", b"context", proof)?;
        state.public(prefix)?;
        self.verifier.verify_on_channel::<H, _>(
            &self.root,
            &self.output.committed_group,
            self.evaluation(),
            &mut state,
        )?;
        let after = ClearChannel::challenge_block(&mut state)?;
        state.check_eof()?;
        Ok(after)
    }
}

pub(crate) struct Recording<S> {
    pub(crate) inner: S,
    pub(crate) messages: Vec<Vec<u8>>,
    pub(crate) message_calls: usize,
    pub(crate) public: Vec<Vec<u8>>,
    pub(crate) draws: Vec<[u8; 32]>,
}
impl<S> Recording<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self {
            inner,
            messages: Vec::new(),
            message_calls: 0,
            public: Vec::new(),
            draws: Vec::new(),
        }
    }
}
impl<S: ClearChannel> ClearChannel for Recording<S> {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.public.push(bytes.to_vec());
        self.inner.public(bytes)
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        self.message_calls += 1;
        self.inner.message(bytes)?;
        self.messages.push(bytes.to_vec());
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        let seed = self.inner.challenge_block()?;
        self.draws.push(seed);
        Ok(seed)
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        self.inner.fold_challenges(sampler, label, count)
    }
}
