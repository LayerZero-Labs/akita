//! The zero-knowledge end-to-end harness (`specs/whiteout-akita.md` §5.5),
//! shared by `zk_end_to_end.rs` and `examples/zk_distinguisher.rs`. See the
//! test file's header for the transcript it runs.

#![allow(dead_code)]

use crate::instance::*;
use akita_config::proof_optimized::fp128;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig};
use akita_error::AkitaError;
use akita_params::ChunkedWitnessCfg;
use akita_pcs::zk::{
    AkitaResponseProver, MaskedPrefix, ResidualOpening, ResidualShape, UncheckedResponseProver,
};
use akita_pcs::AkitaCommitmentScheme;
use akita_types::{AkitaSponge, PROOF_STREAM_PROTOCOL};
use jolt_algebra::{CyclotomicRing, Field, One, Ring, Zero};
use jolt_crypto::lattice::{derive_public_matrix_prefix, sample_box_digits, PublicMatrixSeed};
use jolt_transcript::{Channel, ProverTranscript, SiteId, TranscriptEvent, VerifierTranscript};
use jolt_zk::masking::{
    CommittedPads, CommittedPadsView, PadLayout, ResidualRelation, ResidualRows, ZkError,
};
use jolt_zk::whiteout::{
    prove_sigma, prove_sigma_unchecked, sample_zero_constant_term, verify_sigma, BlockTarget,
    ImageCompression, LatticePadCommitment, PadPlacement, SigmaSetup, WhiteoutError,
};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

/// The response proof's schedule family.
pub(crate) type ResponseCfg = RecursiveCommitmentConfig<fp128::ResponseDigits>;

/// Rank of the pad commitment `A_p` and of `A_D`, the Σ-protocol's binding
/// commitment to `V`. Harness parameters: W6 sizes both for Jolt.
pub(crate) const PAD_RANK: usize = 2;
pub(crate) const V_RANK: usize = 1;
/// The pad commitment's randomness alphabet (jolt-zk's radix 16).
pub(crate) const PAD_RANDOMNESS_LOG_BASIS: u32 = 4;

/// Diagnostic sites of the harness's own messages.
pub(crate) const PAD_SITE: SiteId = SiteId::label("harness/pad-commitment");
pub(crate) const SIGMA_SITE: SiteId = SiteId::label("whiteout/sigma");

/// Why a run failed, by layer.
#[derive(Debug)]
pub(crate) enum Failure {
    Akita(#[expect(dead_code, reason = "read through Debug in failure messages")] AkitaError),
    Zk(ZkError),
}

impl From<AkitaError> for Failure {
    fn from(error: AkitaError) -> Self {
        Self::Akita(error)
    }
}

impl From<ZkError> for Failure {
    fn from(error: ZkError) -> Self {
        Self::Zk(error)
    }
}

impl From<WhiteoutError> for Failure {
    fn from(error: WhiteoutError) -> Self {
        Self::Zk(error.into())
    }
}

pub(crate) fn public_stream(len: usize, seed: u8) -> Vec<F> {
    derive_public_matrix_prefix(len, &PublicMatrixSeed::shake256_paged_v1([seed; 32])).unwrap()
}

/// One synthetic statement's public setup for both roles.
pub(crate) struct Harness<C: CommitmentConfig<Field = F, ExtField = F>> {
    pub(crate) instance: Instance<C>,
    pub(crate) layout: PadLayout,
    pub(crate) shape: ResidualShape,
    /// The public streams `A_p` and `A_D` are cut from.
    pub(crate) pad_stream: Vec<F>,
    pub(crate) v_stream: Vec<F>,
    pub(crate) pad_randomness_columns: usize,
    /// Set once e's length is known, from the setup above.
    pub(crate) response: Option<AkitaResponseProver<ResponseCfg>>,
}

/// What a proof's prover drew and computed, kept for the freshness checks.
pub(crate) struct ZkProof {
    pub(crate) proof: Vec<u8>,
    pub(crate) pads: Vec<F>,
    pub(crate) residual: ResidualOpening<F, F>,
    pub(crate) pad_randomness: Vec<i8>,
    pub(crate) pad_image: Vec<F>,
}

/// What the prover changes to cheat consistently (HT4).
#[derive(Clone, Copy, Default)]
pub(crate) struct Cheat {
    /// The pad slot whose committed pad is one less than the pad that masks
    /// its message: the decoded message is one more than the true one, and
    /// the auxiliary pads are recomputed so the products follow it.
    pub(crate) shifted_slot: Option<usize>,
    /// Skip the Σ prover's per-row checks.
    pub(crate) unchecked_sigma: bool,
    /// Skip the response prover's statement check too.
    pub(crate) unchecked_response: bool,
    /// An injected leak in the first pad (the first statement claim's), for
    /// the distinguisher's positive controls (HT2).
    pub(crate) leak: Option<PadLeak>,
}

/// A weak pad, which an honest prover never draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PadLeak {
    /// The pad is zero: the message is sent in the clear.
    Zero,
    /// The pad has only 64 random bits.
    Bits64,
}

/// What the verifier replays before the Σ-protocol.
pub(crate) struct Replayed {
    pub(crate) pad_payload: Vec<F>,
    pub(crate) prefix: MaskedPrefix<F, F>,
    pub(crate) masked_auxiliary: Vec<F>,
    pub(crate) relation: ResidualRelation<F>,
}

impl<C: CommitmentConfig<Field = F, ExtField = F>> Harness<C> {
    pub(crate) fn new(instance: Instance<C>) -> Self {
        let layout = instance.layout(&vec![0; COUNTING_BYTES]);
        let resolved = instance
            .scheme
            .schedules()
            .resolve_selection(instance.selection())
            .unwrap();
        let shape = ResidualShape::new(resolved.schedule()).unwrap();
        let pad_randomness_columns = akita_params::zk::hiding_randomness_columns(
            akita_params::zk::ImageShape {
                rows: PAD_RANK,
                ring_dim: shape.ring_dim(),
                modulus: C::sis_modulus_profile(),
            },
            akita_params::zk::DigitAlphabet {
                log_basis: PAD_RANDOMNESS_LOG_BASIS,
            },
        )
        .unwrap();
        let mut harness = Self {
            instance,
            layout,
            shape,
            pad_stream: public_stream(1 << 21, 21),
            v_stream: public_stream(1 << 21, 31),
            pad_randomness_columns,
            response: None,
        };
        let num_vars = with_ring_dim!(
            harness.shape.ring_dim(),
            D,
            harness.response_num_vars::<D>()
        );
        harness.response = Some(response_prover(num_vars));
        harness
    }

    pub(crate) fn response(&self) -> &AkitaResponseProver<ResponseCfg> {
        self.response.as_ref().unwrap()
    }

    pub(crate) fn pad_commitment<const D: usize>(&self) -> LatticePadCommitment<F, D> {
        LatticePadCommitment::new(
            PadPlacement::new(&self.layout).unwrap(),
            &self.pad_stream,
            PAD_RANK,
            self.pad_randomness_columns,
        )
        .unwrap()
    }

    /// The Σ setup over `blocks` of the residual and the pad image.
    pub(crate) fn sigma<const D: usize>(
        &self,
        u_a: &[F],
        stream: &[F],
        pad_image: BlockTarget<F, D>,
    ) -> Result<SigmaSetup<F, D>, Failure> {
        Ok(SigmaSetup::new(
            self.shape.segments(),
            self.shape.blocks::<D>(stream, u_a)?,
            &self.pad_commitment::<D>(),
            pad_image,
            V_RANK,
            &self.v_stream,
        )?)
    }

    /// e's variable count, from the public Σ setup alone.
    pub(crate) fn response_num_vars<const D: usize>(&self) -> usize {
        let profile = self.shape.geometry().profile;
        let u_a = vec![F::zero(); profile.outer.matrix.output_rank() * D];
        let stream = self.instance.setup.expanded.shared_matrix.as_field_slice();
        let sigma = self
            .sigma::<D>(
                &u_a,
                stream,
                BlockTarget::Compressed {
                    payload: Vec::new(),
                    image: None,
                },
            )
            .unwrap();
        let layout = sigma
            .response_layout(&akita_pcs::zk::CompressionShape::<ResponseCfg>::default())
            .unwrap();
        AkitaResponseProver::<ResponseCfg>::num_vars(&layout).unwrap()
    }

    /// The verifier's replay of the pad image, the masked claims, the prefix
    /// and the auxiliary block, and the residual rows they lower to.
    pub(crate) fn replay(
        &self,
        transcript: &mut VerifierTranscript<'_, AkitaSponge>,
        pad_image_len: usize,
    ) -> Result<Replayed, Failure> {
        transcript.site(PAD_SITE);
        let pad_payload = transcript
            .receive_n::<F>(
                self.response()
                    .payload_len(pad_image_len)
                    .map_err(ZkError::from)?,
            )
            .map_err(AkitaError::from)?;
        let mut view = CommittedPadsView::counting();
        let (prefix, masked_auxiliary) = self.instance.verify_on(
            transcript,
            &mut view,
            &self.layout,
            self.instance.statement(),
        )?;
        if view.used() != self.layout.num_pads {
            return Err(Failure::Zk(ZkError::PadCountMismatch {
                scheduled: self.layout.num_pads,
                used: view.used(),
            }));
        }
        let mut relation = ResidualRelation::new(self.layout.clone(), 1);
        prefix.rows(&masked_auxiliary).append_rows(&mut relation)?;
        Ok(Replayed {
            pad_payload,
            prefix,
            masked_auxiliary,
            relation,
        })
    }

    /// Proves the statement with the prover's randomness from `seed`.
    pub(crate) fn prove(&self, seed: u64, cheat: Cheat) -> Result<ZkProof, Failure> {
        with_ring_dim!(self.shape.ring_dim(), D, self.prove_at::<D>(seed, cheat))
    }

    pub(crate) fn prove_at<const D: usize>(
        &self,
        seed: u64,
        cheat: Cheat,
    ) -> Result<ZkProof, Failure> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pads: Vec<F> = (0..self.layout.num_pads)
            .map(|_| F::random(&mut rng))
            .collect();
        match cheat.leak {
            None => {}
            Some(PadLeak::Zero) => pads[0] = F::zero(),
            Some(PadLeak::Bits64) => pads[0] = F::from_u64(rng.next_u64()),
        }
        let g = sample_zero_constant_term::<F, 64, _>(&mut rng);
        let pad_randomness = sample_box_digits(
            &mut rng,
            self.pad_randomness_columns * D,
            PAD_RANDOMNESS_LOG_BASIS,
        )
        .unwrap()
        .to_vec();
        // The committed pads differ from the masking pads only in a cheat:
        // the shifted slot's committed pad is one less, so its message
        // decodes to one more than the prover's.
        let mut committed = pads.clone();
        if let Some(slot) = cheat.shifted_slot {
            assert!(slot < self.layout.auxiliary.start, "shift a message slot");
            committed[slot] -= F::one();
        }
        let (image, pad_opening) =
            self.pad_commitment::<D>()
                .commit(&committed, g, &pad_randomness)?;
        let pad_image = flatten(&image);
        let compressed = self
            .response()
            .compress(&pad_image)
            .map_err(ZkError::from)?;

        let mut transcript = ProverTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION);
        transcript.site(PAD_SITE);
        transcript.send_all(&compressed.payload);
        let mut masking = CommittedPads::uncommitted(self.layout.clone(), pads)?;
        let (residual, auxiliary) =
            self.instance
                .prove_prefix_on(&mut masking, own_pads, &mut transcript)?;
        // A consistent cheat sends the products of its shifted messages.
        let auxiliary = match cheat.shifted_slot {
            None => auxiliary,
            Some(_) => {
                let bytes = transcript.narg().to_vec();
                let mut replay =
                    VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION, &bytes);
                replay.site(PAD_SITE);
                replay
                    .receive_n::<F>(compressed.payload.len())
                    .map_err(AkitaError::from)?;
                let prefix = self.instance.verify_prefix_on(
                    &mut replay,
                    &mut CommittedPadsView::counting(),
                    self.instance.statement(),
                )?;
                implied_auxiliary(&prefix, &committed)
            }
        };
        self.instance
            .send_auxiliary(&mut masking, &auxiliary, &mut transcript)?;

        // The prover's residual rows: the verifier's own lowering of the
        // prefix, replayed over the prover's bytes.
        let bytes = transcript.narg().to_vec();
        let mut replay =
            VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION, &bytes);
        let Replayed { relation, .. } = self.replay(&mut replay, pad_image.len())?;
        let sigma = self.sigma::<D>(
            &residual.residual_commitment,
            self.instance.setup.expanded.shared_matrix.as_field_slice(),
            BlockTarget::Compressed {
                payload: compressed.payload.clone(),
                image: Some(image),
            },
        )?;
        let lifted = sigma.lift(&relation, &self.shape.claims(&residual.point)?)?;
        let s = self.shape.residual(&residual)?;
        transcript.site(SIGMA_SITE);
        if !cheat.unchecked_sigma {
            prove_sigma(
                &sigma,
                &lifted,
                &s,
                &pad_opening,
                &mut transcript,
                &mut rng,
                self.response(),
            )?;
        } else if !cheat.unchecked_response {
            prove_sigma_unchecked(
                &sigma,
                &lifted,
                &s,
                &pad_opening,
                &mut transcript,
                &mut rng,
                self.response(),
            )?;
        } else {
            prove_sigma_unchecked(
                &sigma,
                &lifted,
                &s,
                &pad_opening,
                &mut transcript,
                &mut rng,
                &UncheckedResponseProver(self.response()),
            )?;
        }
        Ok(ZkProof {
            proof: transcript.finish(),
            pads: committed,
            residual,
            pad_randomness,
            pad_image,
        })
    }

    /// Verifies `proof`, returning the verifier's event log.
    pub(crate) fn verify(&self, proof: &[u8]) -> Result<Vec<TranscriptEvent>, Failure> {
        with_ring_dim!(self.shape.ring_dim(), D, self.verify_at::<D>(proof))
    }

    pub(crate) fn verify_at<const D: usize>(
        &self,
        proof: &[u8],
    ) -> Result<Vec<TranscriptEvent>, Failure> {
        let mut transcript =
            VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION, proof);
        let Replayed {
            pad_payload,
            prefix,
            relation,
            ..
        } = self.replay(&mut transcript, PAD_RANK * D)?;
        let claim = prefix.residual_claim();
        let sigma = self.sigma::<D>(
            claim.commitment(),
            self.instance
                .verifier
                .setup()
                .expanded()
                .shared_matrix()
                .as_field_slice(),
            BlockTarget::Compressed {
                payload: pad_payload,
                image: None,
            },
        )?;
        let lifted = sigma.lift(&relation, &self.shape.claims(claim.point())?)?;
        transcript.site(SIGMA_SITE);
        verify_sigma(&sigma, &lifted, &mut transcript, self.response())?;
        let events = transcript.events().to_vec();
        transcript.finish().map_err(AkitaError::from)?;
        Ok(events)
    }

    /// The verifier's event log with the statement claims routed through
    /// the clear transcript instead of the masked channel, through the
    /// masked prefix: HT1's positive control.
    pub(crate) fn verify_with_clear_claims(&self, proof: &[u8]) -> Vec<TranscriptEvent> {
        let mut transcript =
            VerifierTranscript::<AkitaSponge>::new(&PROOF_STREAM_PROTOCOL, SESSION, proof);
        transcript.site(PAD_SITE);
        let payload_len = self
            .response()
            .payload_len(PAD_RANK * self.shape.ring_dim())
            .unwrap();
        transcript.receive_n::<F>(payload_len).unwrap();
        transcript.site(CLAIMS_SITE);
        let claims: Vec<F> = transcript.receive_n(self.instance.groups.len()).unwrap();
        // The replay's outcome does not matter, only the sites it reads.
        let _ = self.instance.verifier.verify_masked_prefix(
            &mut transcript,
            &mut CommittedPadsView::counting(),
            self.instance.statement(),
            own_pads(claims),
            BASIS,
        );
        transcript.events().to_vec()
    }
}

/// The coefficients of `rings`, ring by ring.
pub(crate) fn flatten<const D: usize>(rings: &[CyclotomicRing<F, D>]) -> Vec<F> {
    rings
        .iter()
        .flat_map(|ring| ring.coefficients().iter().copied())
        .collect()
}

/// Runs `$body` with `$D` the const ring dimension `$d`.
macro_rules! with_ring_dim {
    ($d:expr, $D:ident, $body:expr) => {
        match $d {
            64 => {
                const $D: usize = 64;
                $body
            }
            128 => {
                const $D: usize = 128;
                $body
            }
            256 => {
                const $D: usize = 256;
                $body
            }
            other => panic!("residual ring dimension {other}"),
        }
    };
}
use with_ring_dim;

/// The response prover over a catalog holding the response schedule of
/// `e` with `num_vars` variables.
pub(crate) fn response_prover(num_vars: usize) -> AkitaResponseProver<ResponseCfg> {
    let scheme = response_scheme();
    let setup = AkitaResponseProver::setup(&scheme, num_vars).unwrap();
    AkitaResponseProver::new(scheme, setup).unwrap()
}

/// The checked-in response-schedule catalog, which holds a row for every
/// end-to-end statement's `e`.
pub(crate) fn response_scheme() -> AkitaCommitmentScheme<ResponseCfg> {
    AkitaCommitmentScheme::new(
        akita_config::test_support::workspace_schedule_catalog::<ResponseCfg>(
            ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap(),
    )
}

pub(crate) fn single() -> Harness<Cfg> {
    single_seeded(7)
}

pub(crate) fn single_seeded(seed: u64) -> Harness<Cfg> {
    Harness::new(Instance::seeded(
        AkitaCommitmentScheme::new(zk_catalog()),
        seed,
    ))
}

pub(crate) fn multi() -> Harness<RecCfg> {
    Harness::new(Instance::multi_group_seeded(7))
}
