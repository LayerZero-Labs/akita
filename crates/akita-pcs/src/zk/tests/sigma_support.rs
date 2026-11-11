//! New-field instance of Jolt's synthetic Whiteout fixture.
//!
//! Adapted from jolt-zk at 790c3fe260792e84107b18640b2093786fbc36e1. The upstream
//! test-support fixture fixes its field; this instance preserves its geometry,
//! relation and witnesses while using Akita's new protocol field. It invokes
//! the public Sigma prover/verifier, including the unchecked adversarial route.

use jolt_algebra::One;
use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, Blake2b512, BytesMsg, ProverState,
    UniformBytes, VerifierState, VerifierTranscript, TAG_BYTES,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use spongefish::DuplexSpongeInit;

use super::sigma_fixture::Instance;
pub(super) use super::sigma_fixture::RawImages;
pub(super) use super::sigma_fixture::F;
use super::sigma_transport::{receive_bytes, receive_values, send_bytes, send_values};
use jolt_zk::masking::{ResidualRelation, ZkError};
use jolt_zk::whiteout::ImageCompression;
use jolt_zk::whiteout::{prove_sigma, prove_sigma_unchecked};
use jolt_zk::whiteout::{verify_sigma, BlockTarget, LiftedRelation, SigmaSetup};
use jolt_zk::whiteout::{ResponseProver, ResponseStatement, ResponseVerifier, ResponseWitness};

pub(super) const PROTOCOL: [u8; TAG_BYTES] = protocol_tag(b"jolt-zk/whiteout-sigma");
pub(super) const LABEL: &[u8] = b"whiteout-sigma";

/// Sends `e` in the clear; the verifier checks the statement directly.
pub(super) struct CheckingResponseProver;

impl<const D: usize> ResponseProver<F, D> for CheckingResponseProver {
    type Sponge = Blake2b512;

    fn compression(&self) -> &dyn ImageCompression<F> {
        &RawImages
    }

    fn prove_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverState<Blake2b512>,
    ) -> Result<(), ZkError> {
        statement.check_witness(witness)?;
        SendingResponseProver.prove_response(statement, witness, transcript)
    }

    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierState<'_, Blake2b512>,
    ) -> Result<(), ZkError> {
        let bytes = receive_bytes(transcript, statement.layout()?.len())?;
        let witness = ResponseWitness {
            digits: bytes.iter().map(|&byte| byte as i8).collect(),
        };
        Ok(statement.check_witness(&witness)?)
    }
}

/// Sends `e` without checking it, so the verifier's check is what rejects.
pub(super) struct SendingResponseProver;

impl<const D: usize> ResponseProver<F, D> for SendingResponseProver {
    type Sponge = Blake2b512;

    fn compression(&self) -> &dyn ImageCompression<F> {
        &RawImages
    }

    fn prove_response(
        &self,
        _statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut ProverState<Blake2b512>,
    ) -> Result<(), ZkError> {
        let bytes: Vec<u8> = witness.digits.iter().map(|&digit| digit as u8).collect();
        send_bytes(transcript, &bytes);
        Ok(())
    }

    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut VerifierState<'_, Blake2b512>,
    ) -> Result<(), ZkError> {
        ResponseProver::<F, D>::verify_response(&CheckingResponseProver, statement, transcript)
    }
}

/// One synthetic Σ-protocol instance over `R_D`: a residual witness under a
/// two-level commitment, five masked messages and one masked product.
pub(super) struct SyntheticSigma<const D: usize> {
    instance: Instance<D>,
    checks: bool,
}

impl<const D: usize> SyntheticSigma<D> {
    /// An honest instance, proved with every prover self-check.
    pub(super) fn new(seed: u64, scale: usize) -> Self {
        Self {
            instance: Instance::scaled(seed, scale),
            checks: true,
        }
    }

    /// An instance whose first affine row is off by one, proved with the
    /// prover's self-checks skipped: its response statement fails Item 5,
    /// `ct(h) = 0`, and nothing else.
    pub(super) fn violating_item_five(seed: u64, scale: usize) -> Self {
        let mut instance = Instance::scaled(seed, scale);
        if let Some(row) = instance.relation.affine.first_mut() {
            row.constant += F::one();
        }
        Self {
            instance,
            checks: false,
        }
    }

    /// The length of `e`, the response statement's committed vector, under
    /// `compression`: only its witness lengths are read.
    pub(super) fn response_len(&self, compression: &dyn ImageCompression<F>) -> usize {
        let unknown = || BlockTarget::Compressed {
            payload: Vec::new(),
            image: None,
        };
        self.instance
            .setup_with([unknown(), unknown()])
            .response_layout(compression)
            .map_or(0, |layout| layout.len())
    }

    /// The images, then the Σ-protocol with `response` proving the
    /// response statement; `seed` drives the prover's randomness.
    pub(super) fn prove<P>(&self, response: &P, seed: u64) -> Result<Vec<u8>, ZkError>
    where
        P: ResponseProver<F, D> + Sync,
        P::Sponge: UniformBytes + DuplexSpongeInit<U = u8> + Send,
    {
        let setup = self
            .instance
            .setup_with(self.instance.targets(response.compression()));
        let lifted = setup.lift(&self.instance.relation, &self.instance.claims)?;
        prove_instance(&self.instance, &setup, &lifted, response, self.checks, seed)
            .map(|(proof, _)| proof)
    }

    /// Verifies `proof`, reading the images from it, with `response`
    /// verifying the response statement.
    pub(super) fn verify<P>(&self, proof: &[u8], response: &P) -> Result<(), ZkError>
    where
        P: ResponseVerifier<F, D>,
        P::Sponge: UniformBytes + DuplexSpongeInit<U = u8>,
    {
        verify_instance(&self.instance, &self.instance.relation, proof, response)
    }
}

/// Proves `instance` under `setup`, sending the images first; returns the
/// proof and the attempt count.
pub(super) fn prove_instance<const D: usize, P>(
    instance: &Instance<D>,
    setup: &SigmaSetup<F, D>,
    lifted: &LiftedRelation<F>,
    response: &P,
    checks: bool,
    seed: u64,
) -> Result<(Vec<u8>, usize), ZkError>
where
    P: ResponseProver<F, D> + Sync,
    P::Sponge: UniformBytes + DuplexSpongeInit<U = u8> + Send,
{
    let mut transcript = prover_transcript::<P::Sponge>(&PROTOCOL, &BytesMsg(LABEL));
    for target in instance.targets(response.compression()) {
        if let BlockTarget::Compressed { payload, .. } = target {
            send_values(&mut transcript, &payload);
        }
    }
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let prove = if checks {
        prove_sigma
    } else {
        prove_sigma_unchecked
    };
    let attempts = prove(
        setup,
        lifted,
        &instance.s,
        &instance.pads,
        &mut transcript,
        &mut rng,
        response,
    )?;
    Ok((transcript.narg_string().to_vec(), attempts))
}

/// Verifies a proof of `relation`, reading the images from the proof.
pub(super) fn verify_instance<const D: usize, P>(
    instance: &Instance<D>,
    relation: &ResidualRelation<F>,
    proof: &[u8],
    response: &P,
) -> Result<(), ZkError>
where
    P: ResponseVerifier<F, D>,
    P::Sponge: UniformBytes + DuplexSpongeInit<U = u8>,
{
    let mut transcript = verifier_transcript::<P::Sponge>(&PROTOCOL, &BytesMsg(LABEL), proof);
    let compression = response.compression();
    let mut receive = |rings: usize| -> Result<BlockTarget<F, D>, ZkError> {
        Ok(BlockTarget::Compressed {
            payload: receive_values::<F, _>(&mut transcript, compression.payload_len(rings * D)?)?,
            image: None,
        })
    };
    let targets = [receive(instance.u.len())?, receive(instance.t_p.len())?];
    let setup = instance.setup_with(targets);
    let lifted = setup.lift(relation, &instance.claims)?;
    verify_sigma(&setup, &lifted, &mut transcript, response)?;
    Ok(VerifierTranscript::check_eof(transcript)?)
}
