//! W5-e manual measurement (`specs/whiteout-akita.md` §5.6): proof bytes,
//! prover time and verifier time of `AkitaResponseProver` on jolt-zk's
//! synthetic Σ-protocol instance, scaled until `e` reaches the target size.
//!
//! ```text
//! cargo run --release -p akita-pcs --no-default-features \
//!     --features parallel,zk-prover --example zk_response -- <scale>
//! ```
//!
//! The synthetic instance's witness grows with `scale`; its pad and
//! commitment shapes stay small, so only `|e|` matches the headline.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use akita_config::proof_optimized::fp128::ResponseDigits;
use akita_config::recursive_commitment::RecursiveCommitmentConfig;
use akita_config::{policy_of, CommitmentConfig};
use akita_params::{
    ChunkedWitnessCfg, CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout,
    ScheduleLookupKey,
};
use akita_pcs::zk::{AkitaResponseProver, CompressionShape};
use akita_pcs::AkitaCommitmentScheme;
use akita_schedules::ValidatedScheduleCatalog;
use akita_transcript::{ProverChannel, TranscriptSponge, VerifierChannel};
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::test_support::{SyntheticSigma, F};
use jolt_zk::whiteout::{ImageCompression, ResponseProver, ResponseStatement, ResponseWitness};

type Cfg = RecursiveCommitmentConfig<ResponseDigits>;
const D: usize = 256;

/// Times the inner prover and counts the bytes it writes.
struct Measured<'a> {
    inner: &'a AkitaResponseProver<Cfg>,
    prove: Mutex<Option<(Duration, usize)>>,
    verify: Mutex<Option<Duration>>,
}

impl ResponseProver<F, D> for Measured<'_> {
    type Sponge = TranscriptSponge;

    fn compression(&self) -> &dyn ImageCompression<F> {
        self.inner
    }

    fn prove_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverChannel,
    ) -> Result<(), ZkError> {
        let before = transcript.narg_string().len();
        let started = Instant::now();
        self.inner.prove_response(statement, witness, transcript)?;
        *self.prove.lock().unwrap() = Some((started.elapsed(), transcript.narg_string().len() - before));
        Ok(())
    }

    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierChannel<'_>,
    ) -> Result<(), ZkError> {
        let started = Instant::now();
        self.inner.verify_response(statement, transcript)?;
        *self.verify.lock().unwrap() = Some(started.elapsed());
        Ok(())
    }
}

fn main() {
    let scale: usize = std::env::args()
        .nth(1)
        .map_or(1, |arg| arg.parse().expect("scale"));
    std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || run(scale))
        .unwrap()
        .join()
        .unwrap();
}

fn run(scale: usize) {
    let sigma = SyntheticSigma::<D>::new(1, scale);
    let len = sigma.response_len(&CompressionShape::<Cfg>::default());
    let num_vars = len.next_power_of_two().trailing_zeros() as usize;
    println!("scale {scale}: |e| = {len}, nu = {num_vars}");

    let started = Instant::now();
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
    let policy = policy_of::<Cfg>(chunking);
    let schedule = akita_planner::find_schedule(
        &key,
        Cfg::committed_source_contract().unwrap(),
        &[],
        &policy,
        Cfg::ring_challenge_config,
    )
    .unwrap()
    .schedule;
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )
        .unwrap(),
        precommitteds: Vec::new(),
    };
    let catalog = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        [(profiles, schedule)],
        &policy,
        Cfg::ring_challenge_config,
    )
    .unwrap();
    let scheme = AkitaCommitmentScheme::<Cfg>::from_schedule_artifact(
        &catalog.to_artifact_bytes().unwrap(),
        chunking,
    )
    .unwrap();
    let setup = AkitaResponseProver::setup(&scheme, num_vars).unwrap();
    let akita = AkitaResponseProver::new(scheme, setup).unwrap();
    println!("planning and setup: {:.2?}", started.elapsed());

    let measured = Measured {
        inner: &akita,
        prove: Mutex::new(None),
        verify: Mutex::new(None),
    };
    let proof = sigma.prove(&measured, 2).unwrap();
    sigma.verify(&proof, &measured).unwrap();
    let (prove, bytes) = measured.prove.lock().unwrap().unwrap();
    let verify = measured.verify.lock().unwrap().unwrap();
    println!("response proof: {bytes} bytes");
    println!("response prover: {prove:.2?}");
    println!("response verifier: {verify:.2?}");
}
