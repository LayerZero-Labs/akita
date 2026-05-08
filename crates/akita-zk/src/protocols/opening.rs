//! Ring-native Ajtai opening proof with exact box rejection.

use crate::compact::CompactRingVec;
use crate::error::ZkResult;
use crate::norm::{centered_i128, ring_vec_within_infinity_bound, sample_ring_vec_box};
use crate::rejection::gaertner::{
    gaertner_acceptance, gaertner_roll, GaertnerOutcome, GaertnerRejectionParams,
};
use crate::rejection::gaussian::{
    gaussian_rejection_acceptance, sample_ring_vec_discrete_gaussian,
};
use crate::rejection::{BoxRejectionParams, GaussianRejectionParams};
use crate::relations::AjtaiRelation;
use crate::ring_ext::{add_ring_vecs, mul_sparse_challenge_vec, sub_ring_vecs};
use akita_algebra::CyclotomicRing;
use akita_challenges::{sample_sparse_challenges, SparseChallenge, SparseChallengeConfig};
use akita_field::{AkitaError, CanonicalField, FieldCore, PseudoMersenneField};
use akita_serialization::{AkitaSerialize, Compress};
use akita_transcript::Transcript;
#[cfg(feature = "parallel")]
use rand_chacha::rand_core::Rng as ChaChaRng;
#[cfg(feature = "parallel")]
use rand_chacha::rand_core::SeedableRng as ChaChaSeedableRng;
#[cfg(feature = "parallel")]
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const DOMAIN_AKITA_ZK_OPENING: &[u8] = b"ak/zk/open";
const ABSORB_SHAPE: &[u8] = b"ak/zk/a/sh";
const ABSORB_REJECTION: &[u8] = b"ak/zk/a/rj";
const ABSORB_MATRIX: &[u8] = b"ak/zk/a/m";
const ABSORB_COMMITMENT: &[u8] = b"ak/zk/a/t";
const ABSORB_ANNOUNCEMENT: &[u8] = b"ak/zk/a/a";
const CHALLENGE_OPENING: &[u8] = b"ak/zk/c";

#[cfg(feature = "parallel")]
struct CompatChaCha20Rng(ChaCha20Rng);

#[cfg(feature = "parallel")]
impl RngCore for CompatChaCha20Rng {
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// Non-interactive Ajtai opening proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AjtaiOpeningProof<F: FieldCore, const D: usize> {
    /// First Sigma-protocol message `a = A y`.
    pub announcement: Vec<CyclotomicRing<F, D>>,
    /// Accepted response `z = y + c s`.
    pub response: Vec<CyclotomicRing<F, D>>,
}

/// Non-interactive Ajtai opening proof with compact response encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactAjtaiOpeningProof<F: FieldCore, const D: usize> {
    /// First Sigma-protocol message `a = A y`.
    pub announcement: Vec<CyclotomicRing<F, D>>,
    /// Accepted response packed as centered two's-complement coefficients.
    pub response: CompactRingVec,
}

impl<F: FieldCore, const D: usize> CompactAjtaiOpeningProof<F, D> {
    /// Serialized payload size when shape metadata is known externally.
    ///
    /// This counts full-field announcement rings plus packed response bytes.
    pub fn serialized_size(&self) -> usize
    where
        F: AkitaSerialize,
    {
        self.announcement
            .iter()
            .map(|ring| ring.serialized_size(Compress::No))
            .sum::<usize>()
            + self.response.packed_byte_len()
    }

    /// Expand into the full-field proof representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the compact response shape is invalid.
    pub fn expand(self) -> ZkResult<AjtaiOpeningProof<F, D>>
    where
        F: CanonicalField,
    {
        Ok(AjtaiOpeningProof {
            announcement: self.announcement,
            response: self.response.unpack()?,
        })
    }
}

impl<F: FieldCore + AkitaSerialize, const D: usize> AkitaSerialize
    for CompactAjtaiOpeningProof<F, D>
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), akita_serialization::SerializationError> {
        for ring in &self.announcement {
            ring.serialize_with_mode(&mut writer, compress)?;
        }
        self.response.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        self.serialized_size()
    }
}

/// Interactive Ajtai opening transcript with an explicit challenge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AjtaiOpeningTranscript<F: FieldCore, const D: usize> {
    /// First Sigma-protocol message `a = A y`.
    pub announcement: Vec<CyclotomicRing<F, D>>,
    /// Verifier challenge `c`.
    pub challenge: SparseChallenge,
    /// Accepted response `z = y + c s`.
    pub response: Vec<CyclotomicRing<F, D>>,
}

/// Prove knowledge of a short Ajtai opening.
///
/// # Errors
///
/// Returns an error if the relation, witness, challenge config, or rejection
/// parameters are invalid, or if no non-aborting proof is found within
/// `max_attempts`.
pub fn prove_ajtai_opening<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &BoxRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<AjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField + Send + Sync,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    if max_attempts == 0 {
        return Err(AkitaError::InvalidInput(
            "max_attempts must be non-zero".to_string(),
        ));
    }
    validate_opening_inputs(relation, witness, challenge_cfg, params)?;
    if !relation.check_short_opening(witness, params.witness_bound)? {
        return Err(AkitaError::InvalidInput(
            "witness is not a short opening of the relation".to_string(),
        ));
    }

    for _ in 0..max_attempts {
        let mask = sample_ring_vec_box::<F, R, D>(rng, witness.len(), params.gamma)?;
        let announcement = relation.commit(&mask)?;
        let challenge =
            fiat_shamir_challenge::<F, T, D>(relation, challenge_cfg, params, &announcement)?;
        let shift = mul_sparse_challenge_vec(&challenge, witness)?;
        let response = add_ring_vecs(&mask, &shift)?;
        if ring_vec_within_infinity_bound(&response, params.response_bound)? {
            return Ok(AjtaiOpeningProof {
                announcement,
                response,
            });
        }
    }

    Err(AkitaError::InvalidInput(format!(
        "failed to sample non-aborting proof after {max_attempts} attempts"
    )))
}

/// Prove knowledge of a short Ajtai opening and pack the response.
///
/// # Errors
///
/// Returns an error if proving fails or if the accepted response does not fit
/// the configured compact response bound.
pub fn prove_compact_ajtai_opening<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &BoxRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<CompactAjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    let proof = prove_ajtai_opening::<F, T, R, D>(
        relation,
        witness,
        challenge_cfg,
        params,
        rng,
        max_attempts,
    )?;
    compact_ajtai_opening_proof(&proof, params)
}

/// Prove knowledge of a short Ajtai opening with the experimental Gaussian
/// rejection policy.
///
/// # Errors
///
/// Returns an error if inputs are invalid, if the witness is not a short
/// opening, or if no non-aborting proof is found within `max_attempts`.
pub fn prove_gaussian_heuristic_ajtai_opening<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<AjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    if max_attempts == 0 {
        return Err(AkitaError::InvalidInput(
            "max_attempts must be non-zero".to_string(),
        ));
    }
    validate_gaussian_opening_inputs(relation, witness, challenge_cfg, params)?;
    if !relation.check_short_opening(witness, params.witness_bound)? {
        return Err(AkitaError::InvalidInput(
            "witness is not a short opening of the relation".to_string(),
        ));
    }

    #[cfg(feature = "parallel")]
    {
        prove_gaussian_heuristic_ajtai_opening_parallel::<F, T, R, D>(
            relation,
            witness,
            challenge_cfg,
            params,
            rng,
            max_attempts,
        )
    }
    #[cfg(not(feature = "parallel"))]
    {
        prove_gaussian_heuristic_ajtai_opening_sequential::<F, T, R, D>(
            relation,
            witness,
            challenge_cfg,
            params,
            rng,
            max_attempts,
        )
    }
}

#[cfg(feature = "parallel")]
fn prove_gaussian_heuristic_ajtai_opening_parallel<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<AjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField + Send + Sync,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    let mut seeds = Vec::with_capacity(max_attempts);
    for _ in 0..max_attempts {
        let mut seed = <ChaCha20Rng as ChaChaSeedableRng>::Seed::default();
        rng.fill_bytes(seed.as_mut());
        seeds.push(seed);
    }

    seeds
        .into_par_iter()
        .find_map_any(|seed| {
            let mut attempt_rng =
                CompatChaCha20Rng(<ChaCha20Rng as ChaChaSeedableRng>::from_seed(seed));
            match try_gaussian_heuristic_attempt::<F, T, _, D>(
                relation,
                witness,
                challenge_cfg,
                params,
                &mut attempt_rng,
            ) {
                Ok(Some(proof)) => Some(Ok(proof)),
                Ok(None) => None,
                Err(err) => Some(Err(err)),
            }
        })
        .unwrap_or_else(|| {
            Err(AkitaError::InvalidInput(format!(
                "failed to sample non-aborting gaussian proof after {max_attempts} attempts"
            )))
        })
}

#[cfg(not(feature = "parallel"))]
fn prove_gaussian_heuristic_ajtai_opening_sequential<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<AjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    for _ in 0..max_attempts {
        if let Some(proof) = try_gaussian_heuristic_attempt::<F, T, R, D>(
            relation,
            witness,
            challenge_cfg,
            params,
            rng,
        )? {
            return Ok(proof);
        }
    }

    Err(AkitaError::InvalidInput(format!(
        "failed to sample non-aborting gaussian proof after {max_attempts} attempts"
    )))
}

fn try_gaussian_heuristic_attempt<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
    rng: &mut R,
) -> ZkResult<Option<AjtaiOpeningProof<F, D>>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    let mask = sample_ring_vec_discrete_gaussian::<F, R, D>(
        rng,
        witness.len(),
        params.sigma,
        params.mask_bound,
    )?;
    let announcement = relation.commit(&mask)?;
    let challenge =
        fiat_shamir_gaussian_challenge::<F, T, D>(relation, challenge_cfg, params, &announcement)?;
    let shift = mul_sparse_challenge_vec(&challenge, witness)?;
    let response = add_ring_vecs(&mask, &shift)?;
    if !ring_vec_within_infinity_bound(&response, params.response_bound)? {
        return Ok(None);
    }

    let accept_probability = gaussian_rejection_acceptance(
        ring_vec_l2_squared(&mask)?,
        ring_vec_l2_squared(&response)?,
        params.rejection_m,
        params.sigma,
    );
    if uniform_f64(rng) <= accept_probability {
        return Ok(Some(AjtaiOpeningProof {
            announcement,
            response,
        }));
    }
    Ok(None)
}

/// Prove with experimental Gaussian rejection and pack the response.
///
/// # Errors
///
/// Returns an error if proving fails or if the accepted response does not fit
/// the configured compact response bound.
pub fn prove_compact_gaussian_heuristic_ajtai_opening<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<CompactAjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField + Send + Sync,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    let proof = prove_gaussian_heuristic_ajtai_opening::<F, T, R, D>(
        relation,
        witness,
        challenge_cfg,
        params,
        rng,
        max_attempts,
    )?;
    Ok(CompactAjtaiOpeningProof {
        announcement: proof.announcement,
        response: CompactRingVec::pack_with_bound(&proof.response, params.response_bound)?,
    })
}

/// Pack an already generated full-field proof response.
///
/// # Errors
///
/// Returns an error if the response does not fit the configured response bound.
pub fn compact_ajtai_opening_proof<F, const D: usize>(
    proof: &AjtaiOpeningProof<F, D>,
    params: &BoxRejectionParams,
) -> ZkResult<CompactAjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    Ok(CompactAjtaiOpeningProof {
        announcement: proof.announcement.clone(),
        response: CompactRingVec::pack_with_bound(&proof.response, params.response_bound)?,
    })
}

/// Public-sign Ajtai opening proof using the Gärtner rejection rule.
///
/// **Non-ZK measurement only.** The Gärtner rule selects between `y - v` and
/// `y + v` according to `f_v / g_v`; without BLISS-style structure on
/// `(A, s, t)` the sign cannot be hidden, so this proof carries it
/// explicitly. Use only as a rejection-policy benchmark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaertnerAjtaiOpeningProof<F: FieldCore, const D: usize> {
    /// First Sigma-protocol message `a = A y`.
    pub announcement: Vec<CyclotomicRing<F, D>>,
    /// Accepted response `z = y + sign * c * w` for the recorded `sign`.
    pub response: Vec<CyclotomicRing<F, D>>,
    /// Public sign bit selected by the rejection rule. Either `+1` or `-1`.
    pub sign: i8,
}

/// Public-sign Ajtai opening proof using the Gärtner rejection rule with
/// compact response encoding. Non-ZK; measurement only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactGaertnerAjtaiOpeningProof<F: FieldCore, const D: usize> {
    /// First Sigma-protocol message `a = A y`.
    pub announcement: Vec<CyclotomicRing<F, D>>,
    /// Accepted response packed as centered two's-complement coefficients.
    pub response: CompactRingVec,
    /// Public sign bit selected by the rejection rule. Either `+1` or `-1`.
    pub sign: i8,
}

impl<F: FieldCore, const D: usize> CompactGaertnerAjtaiOpeningProof<F, D> {
    /// Serialized payload size when shape metadata is known externally.
    ///
    /// Counts full-field announcement rings, packed response bytes, and the
    /// one-byte public sign.
    pub fn serialized_size(&self) -> usize
    where
        F: AkitaSerialize,
    {
        self.announcement
            .iter()
            .map(|ring| ring.serialized_size(Compress::No))
            .sum::<usize>()
            + self.response.packed_byte_len()
            + 1
    }

    /// Expand into the full-field proof representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the compact response shape is invalid.
    pub fn expand(self) -> ZkResult<GaertnerAjtaiOpeningProof<F, D>>
    where
        F: CanonicalField,
    {
        Ok(GaertnerAjtaiOpeningProof {
            announcement: self.announcement,
            response: self.response.unpack()?,
            sign: self.sign,
        })
    }
}

/// Prove knowledge of a short Ajtai opening with the Gärtner rejection
/// policy. **Non-ZK** because the sign bit is sent in the clear.
///
/// # Errors
///
/// Returns an error if inputs are invalid, if the witness is not a short
/// opening, or if no non-aborting proof is found within `max_attempts`.
pub fn prove_gaertner_ajtai_opening<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<GaertnerAjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    if max_attempts == 0 {
        return Err(AkitaError::InvalidInput(
            "max_attempts must be non-zero".to_string(),
        ));
    }
    validate_gaertner_opening_inputs(relation, witness, challenge_cfg, params)?;
    if !relation.check_short_opening(witness, params.witness_bound)? {
        return Err(AkitaError::InvalidInput(
            "witness is not a short opening of the relation".to_string(),
        ));
    }

    #[cfg(feature = "parallel")]
    {
        prove_gaertner_ajtai_opening_parallel::<F, T, R, D>(
            relation,
            witness,
            challenge_cfg,
            params,
            rng,
            max_attempts,
        )
    }
    #[cfg(not(feature = "parallel"))]
    {
        prove_gaertner_ajtai_opening_sequential::<F, T, R, D>(
            relation,
            witness,
            challenge_cfg,
            params,
            rng,
            max_attempts,
        )
    }
}

#[cfg(feature = "parallel")]
fn prove_gaertner_ajtai_opening_parallel<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<GaertnerAjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    let mut seeds = Vec::with_capacity(max_attempts);
    for _ in 0..max_attempts {
        let mut seed = <ChaCha20Rng as ChaChaSeedableRng>::Seed::default();
        rng.fill_bytes(seed.as_mut());
        seeds.push(seed);
    }

    seeds
        .into_par_iter()
        .find_map_any(|seed| {
            let mut attempt_rng =
                CompatChaCha20Rng(<ChaCha20Rng as ChaChaSeedableRng>::from_seed(seed));
            match try_gaertner_attempt::<F, T, _, D>(
                relation,
                witness,
                challenge_cfg,
                params,
                &mut attempt_rng,
            ) {
                Ok(Some(proof)) => Some(Ok(proof)),
                Ok(None) => None,
                Err(err) => Some(Err(err)),
            }
        })
        .unwrap_or_else(|| {
            Err(AkitaError::InvalidInput(format!(
                "failed to sample non-aborting gaertner proof after {max_attempts} attempts"
            )))
        })
}

#[cfg(not(feature = "parallel"))]
fn prove_gaertner_ajtai_opening_sequential<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<GaertnerAjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    for _ in 0..max_attempts {
        if let Some(proof) =
            try_gaertner_attempt::<F, T, R, D>(relation, witness, challenge_cfg, params, rng)?
        {
            return Ok(proof);
        }
    }

    Err(AkitaError::InvalidInput(format!(
        "failed to sample non-aborting gaertner proof after {max_attempts} attempts"
    )))
}

fn try_gaertner_attempt<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
    rng: &mut R,
) -> ZkResult<Option<GaertnerAjtaiOpeningProof<F, D>>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    let mask = sample_ring_vec_discrete_gaussian::<F, R, D>(
        rng,
        witness.len(),
        params.sigma,
        params.mask_bound,
    )?;
    let announcement = relation.commit(&mask)?;
    let challenge =
        fiat_shamir_gaertner_challenge::<F, T, D>(relation, challenge_cfg, params, &announcement)?;
    let shift = mul_sparse_challenge_vec(&challenge, witness)?;

    let inner_y_v = ring_vec_inner_product_centered(&mask, &shift)? as f64;
    let v_l2_sq = ring_vec_l2_squared(&shift)?;
    let (f_v, g_v) = gaertner_acceptance(inner_y_v, v_l2_sq, params.sigma, params.rejection_m);
    let outcome = gaertner_roll(uniform_f64(rng), f_v, g_v);
    let (response, sign) = match outcome {
        GaertnerOutcome::SignNegative => (sub_ring_vecs(&mask, &shift)?, -1_i8),
        GaertnerOutcome::SignPositive => (add_ring_vecs(&mask, &shift)?, 1_i8),
        GaertnerOutcome::Abort => return Ok(None),
    };
    if !ring_vec_within_infinity_bound(&response, params.response_bound)? {
        return Ok(None);
    }

    Ok(Some(GaertnerAjtaiOpeningProof {
        announcement,
        response,
        sign,
    }))
}

/// Prove with the Gärtner rejection policy and pack the response.
///
/// # Errors
///
/// Returns an error if proving fails or if the accepted response does not fit
/// the configured compact response bound.
pub fn prove_compact_gaertner_ajtai_opening<F, T, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
    rng: &mut R,
    max_attempts: usize,
) -> ZkResult<CompactGaertnerAjtaiOpeningProof<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
    R: RngCore + ?Sized,
{
    let proof = prove_gaertner_ajtai_opening::<F, T, R, D>(
        relation,
        witness,
        challenge_cfg,
        params,
        rng,
        max_attempts,
    )?;
    Ok(CompactGaertnerAjtaiOpeningProof {
        announcement: proof.announcement,
        response: CompactRingVec::pack_with_bound(&proof.response, params.response_bound)?,
        sign: proof.sign,
    })
}

/// Verify a public-sign Gärtner Ajtai opening proof.
///
/// # Errors
///
/// Returns an error if public inputs or rejection parameters are invalid, or
/// if the recorded sign is not in `{-1, +1}`.
pub fn verify_gaertner_ajtai_opening<F, T, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
    proof: &GaertnerAjtaiOpeningProof<F, D>,
) -> ZkResult<bool>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
{
    validate_gaertner_public_inputs(relation, challenge_cfg, params)?;
    if proof.announcement.len() != relation.row_count() {
        return Err(AkitaError::InvalidInput(format!(
            "announcement length {} does not match relation row count {}",
            proof.announcement.len(),
            relation.row_count()
        )));
    }
    if proof.response.len() != relation.col_count() {
        return Err(AkitaError::InvalidInput(format!(
            "response length {} does not match relation column count {}",
            proof.response.len(),
            relation.col_count()
        )));
    }
    if proof.sign != 1 && proof.sign != -1 {
        return Err(AkitaError::InvalidInput(format!(
            "sign must be +1 or -1, got {}",
            proof.sign
        )));
    }
    if !ring_vec_within_infinity_bound(&proof.response, params.response_bound)? {
        return Ok(false);
    }

    let challenge = fiat_shamir_gaertner_challenge::<F, T, D>(
        relation,
        challenge_cfg,
        params,
        &proof.announcement,
    )?;
    let lhs = relation.commit(&proof.response)?;
    let ct = mul_sparse_challenge_vec(&challenge, &relation.commitment)?;
    let rhs = if proof.sign == 1 {
        add_ring_vecs(&proof.announcement, &ct)?
    } else {
        sub_ring_vecs(&proof.announcement, &ct)?
    };
    Ok(lhs == rhs)
}

/// Verify a compact public-sign Gärtner Ajtai opening proof.
///
/// # Errors
///
/// Returns an error if public inputs, rejection parameters, or compact
/// response encoding are invalid.
pub fn verify_compact_gaertner_ajtai_opening<F, T, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
    proof: &CompactGaertnerAjtaiOpeningProof<F, D>,
) -> ZkResult<bool>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
{
    let expanded = proof.clone().expand()?;
    verify_gaertner_ajtai_opening::<F, T, D>(relation, challenge_cfg, params, &expanded)
}

/// Verify a non-interactive Ajtai opening proof.
///
/// # Errors
///
/// Returns an error if public inputs or rejection parameters are invalid.
pub fn verify_ajtai_opening<F, T, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &BoxRejectionParams,
    proof: &AjtaiOpeningProof<F, D>,
) -> ZkResult<bool>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
{
    validate_public_inputs(relation, challenge_cfg, params)?;
    if proof.announcement.len() != relation.row_count() {
        return Err(AkitaError::InvalidInput(format!(
            "announcement length {} does not match relation row count {}",
            proof.announcement.len(),
            relation.row_count()
        )));
    }
    if proof.response.len() != relation.col_count() {
        return Err(AkitaError::InvalidInput(format!(
            "response length {} does not match relation column count {}",
            proof.response.len(),
            relation.col_count()
        )));
    }

    let challenge =
        fiat_shamir_challenge::<F, T, D>(relation, challenge_cfg, params, &proof.announcement)?;
    verify_ajtai_opening_transcript(
        relation,
        params,
        &AjtaiOpeningTranscript {
            announcement: proof.announcement.clone(),
            challenge,
            response: proof.response.clone(),
        },
    )
}

/// Verify a non-interactive compact Ajtai opening proof.
///
/// # Errors
///
/// Returns an error if public inputs, rejection parameters, or compact response
/// encoding are invalid.
pub fn verify_compact_ajtai_opening<F, T, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &BoxRejectionParams,
    proof: &CompactAjtaiOpeningProof<F, D>,
) -> ZkResult<bool>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
{
    let expanded = proof.clone().expand()?;
    verify_ajtai_opening::<F, T, D>(relation, challenge_cfg, params, &expanded)
}

/// Verify a non-interactive proof using the experimental Gaussian rejection
/// policy.
///
/// # Errors
///
/// Returns an error if public inputs or rejection parameters are invalid.
pub fn verify_gaussian_heuristic_ajtai_opening<F, T, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
    proof: &AjtaiOpeningProof<F, D>,
) -> ZkResult<bool>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
{
    validate_gaussian_public_inputs(relation, challenge_cfg, params)?;
    if proof.announcement.len() != relation.row_count() {
        return Err(AkitaError::InvalidInput(format!(
            "announcement length {} does not match relation row count {}",
            proof.announcement.len(),
            relation.row_count()
        )));
    }
    if proof.response.len() != relation.col_count() {
        return Err(AkitaError::InvalidInput(format!(
            "response length {} does not match relation column count {}",
            proof.response.len(),
            relation.col_count()
        )));
    }
    if !ring_vec_within_infinity_bound(&proof.response, params.response_bound)? {
        return Ok(false);
    }

    let challenge = fiat_shamir_gaussian_challenge::<F, T, D>(
        relation,
        challenge_cfg,
        params,
        &proof.announcement,
    )?;
    let lhs = relation.commit(&proof.response)?;
    let ct = mul_sparse_challenge_vec(&challenge, &relation.commitment)?;
    let rhs = add_ring_vecs(&proof.announcement, &ct)?;
    Ok(lhs == rhs)
}

/// Verify a compact proof using the experimental Gaussian rejection policy.
///
/// # Errors
///
/// Returns an error if public inputs, rejection parameters, or compact response
/// encoding are invalid.
pub fn verify_compact_gaussian_heuristic_ajtai_opening<F, T, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
    proof: &CompactAjtaiOpeningProof<F, D>,
) -> ZkResult<bool>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    T: Transcript<F>,
{
    let expanded = proof.clone().expand()?;
    verify_gaussian_heuristic_ajtai_opening::<F, T, D>(relation, challenge_cfg, params, &expanded)
}

/// Simulate an honest-verifier Ajtai opening transcript for a fixed challenge.
///
/// This is the interactive simulator, not a Fiat-Shamir random-oracle
/// programming simulator. It samples `z` from the accepted response box and
/// sets `a = A z - c t`.
///
/// # Errors
///
/// Returns an error if public inputs or parameters are invalid.
pub fn simulate_ajtai_opening_transcript<F, R, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    params: &BoxRejectionParams,
    challenge: SparseChallenge,
    rng: &mut R,
) -> ZkResult<AjtaiOpeningTranscript<F, D>>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
    R: RngCore + ?Sized,
{
    validate_rejection_params::<F, D>(relation.col_count(), params)?;
    let response =
        sample_ring_vec_box::<F, R, D>(rng, relation.col_count(), params.response_bound)?;
    let az = relation.commit(&response)?;
    let ct = mul_sparse_challenge_vec(&challenge, &relation.commitment)?;
    let announcement = sub_ring_vecs(&az, &ct)?;
    Ok(AjtaiOpeningTranscript {
        announcement,
        challenge,
        response,
    })
}

/// Verify an interactive Ajtai opening transcript with an explicit challenge.
///
/// # Errors
///
/// Returns an error if the transcript shape or rejection parameters are invalid.
pub fn verify_ajtai_opening_transcript<F, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    params: &BoxRejectionParams,
    transcript: &AjtaiOpeningTranscript<F, D>,
) -> ZkResult<bool>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    validate_rejection_params::<F, D>(relation.col_count(), params)?;
    if transcript.announcement.len() != relation.row_count() {
        return Err(AkitaError::InvalidInput(format!(
            "announcement length {} does not match relation row count {}",
            transcript.announcement.len(),
            relation.row_count()
        )));
    }
    if transcript.response.len() != relation.col_count() {
        return Err(AkitaError::InvalidInput(format!(
            "response length {} does not match relation column count {}",
            transcript.response.len(),
            relation.col_count()
        )));
    }
    if !ring_vec_within_infinity_bound(&transcript.response, params.response_bound)? {
        return Ok(false);
    }

    let lhs = relation.commit(&transcript.response)?;
    let ct = mul_sparse_challenge_vec(&transcript.challenge, &relation.commitment)?;
    let rhs = add_ring_vecs(&transcript.announcement, &ct)?;
    Ok(lhs == rhs)
}

fn validate_opening_inputs<F, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &BoxRejectionParams,
) -> ZkResult<()>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    validate_public_inputs(relation, challenge_cfg, params)?;
    if witness.len() != relation.col_count() {
        return Err(AkitaError::InvalidInput(format!(
            "witness length {} does not match relation column count {}",
            witness.len(),
            relation.col_count()
        )));
    }
    Ok(())
}

fn validate_public_inputs<F, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &BoxRejectionParams,
) -> ZkResult<()>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    challenge_cfg
        .validate::<D>()
        .map_err(|e| AkitaError::InvalidInput(format!("invalid challenge config: {e}")))?;
    if params.challenge_l1_bound < challenge_cfg.l1_norm() {
        return Err(AkitaError::InvalidInput(format!(
            "rejection challenge L1 bound {} is smaller than config bound {}",
            params.challenge_l1_bound,
            challenge_cfg.l1_norm()
        )));
    }
    validate_rejection_params::<F, D>(relation.col_count(), params)
}

fn validate_rejection_params<F, const D: usize>(
    witness_len: usize,
    params: &BoxRejectionParams,
) -> ZkResult<()>
where
    F: PseudoMersenneField,
{
    let expected_revealed = witness_len
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidInput("revealed coefficient overflow".to_string()))?;
    if params.revealed_coefficients != expected_revealed {
        return Err(AkitaError::InvalidInput(format!(
            "params reveal {} coefficients, but relation reveals {expected_revealed}",
            params.revealed_coefficients
        )));
    }
    if params.gamma <= params.beta {
        return Err(AkitaError::InvalidInput(
            "gamma must be larger than beta".to_string(),
        ));
    }
    if params.response_bound != params.gamma - params.beta {
        return Err(AkitaError::InvalidInput(
            "response_bound must equal gamma - beta".to_string(),
        ));
    }
    params.validate_no_modular_wrap::<F>()
}

fn validate_gaussian_opening_inputs<F, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
) -> ZkResult<()>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    validate_gaussian_public_inputs(relation, challenge_cfg, params)?;
    if witness.len() != relation.col_count() {
        return Err(AkitaError::InvalidInput(format!(
            "witness length {} does not match relation column count {}",
            witness.len(),
            relation.col_count()
        )));
    }
    Ok(())
}

fn validate_gaussian_public_inputs<F, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
) -> ZkResult<()>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    challenge_cfg
        .validate::<D>()
        .map_err(|e| AkitaError::InvalidInput(format!("invalid challenge config: {e}")))?;
    if params.challenge_l1_bound < challenge_cfg.l1_norm() {
        return Err(AkitaError::InvalidInput(format!(
            "rejection challenge L1 bound {} is smaller than config bound {}",
            params.challenge_l1_bound,
            challenge_cfg.l1_norm()
        )));
    }
    validate_gaussian_rejection_params::<F, D>(relation.col_count(), params)
}

fn validate_gaussian_rejection_params<F, const D: usize>(
    witness_len: usize,
    params: &GaussianRejectionParams,
) -> ZkResult<()>
where
    F: PseudoMersenneField,
{
    let expected_revealed = witness_len
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidInput("revealed coefficient overflow".to_string()))?;
    if params.revealed_coefficients != expected_revealed {
        return Err(AkitaError::InvalidInput(format!(
            "params reveal {} coefficients, but relation reveals {expected_revealed}",
            params.revealed_coefficients
        )));
    }
    if params.response_bound == 0 {
        return Err(AkitaError::InvalidInput(
            "response_bound must be non-zero".to_string(),
        ));
    }
    let required_mask_bound = params
        .response_bound
        .checked_add(params.beta)
        .ok_or_else(|| AkitaError::InvalidInput("mask bound overflow".to_string()))?;
    if params.mask_bound < required_mask_bound {
        return Err(AkitaError::InvalidInput(
            "mask_bound must cover response_bound + beta".to_string(),
        ));
    }
    if !params.sigma.is_finite() || params.sigma <= 0.0 {
        return Err(AkitaError::InvalidInput(
            "sigma must be finite and positive".to_string(),
        ));
    }
    if !params.rejection_m.is_finite() || params.rejection_m < 1.0 {
        return Err(AkitaError::InvalidInput(
            "rejection_m must be finite and at least one".to_string(),
        ));
    }
    params.validate_no_modular_wrap::<F>()
}

fn fiat_shamir_challenge<F, T, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &BoxRejectionParams,
    announcement: &[CyclotomicRing<F, D>],
) -> ZkResult<SparseChallenge>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let mut transcript = T::new(DOMAIN_AKITA_ZK_OPENING);
    absorb_public_inputs(&mut transcript, relation, challenge_cfg, params);
    append_ring_vec(&mut transcript, ABSORB_ANNOUNCEMENT, announcement);
    let mut challenges =
        sample_sparse_challenges::<F, T, D>(&mut transcript, CHALLENGE_OPENING, 1, challenge_cfg)?;
    Ok(challenges.remove(0))
}

fn fiat_shamir_gaussian_challenge<F, T, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
    announcement: &[CyclotomicRing<F, D>],
) -> ZkResult<SparseChallenge>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let mut transcript = T::new(DOMAIN_AKITA_ZK_OPENING);
    absorb_gaussian_public_inputs(&mut transcript, relation, challenge_cfg, params);
    append_ring_vec(&mut transcript, ABSORB_ANNOUNCEMENT, announcement);
    let mut challenges =
        sample_sparse_challenges::<F, T, D>(&mut transcript, CHALLENGE_OPENING, 1, challenge_cfg)?;
    Ok(challenges.remove(0))
}

fn fiat_shamir_gaertner_challenge<F, T, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
    announcement: &[CyclotomicRing<F, D>],
) -> ZkResult<SparseChallenge>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let mut transcript = T::new(DOMAIN_AKITA_ZK_OPENING);
    absorb_gaertner_public_inputs(&mut transcript, relation, challenge_cfg, params);
    append_ring_vec(&mut transcript, ABSORB_ANNOUNCEMENT, announcement);
    let mut challenges =
        sample_sparse_challenges::<F, T, D>(&mut transcript, CHALLENGE_OPENING, 1, challenge_cfg)?;
    Ok(challenges.remove(0))
}

fn validate_gaertner_opening_inputs<F, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    witness: &[CyclotomicRing<F, D>],
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
) -> ZkResult<()>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    validate_gaertner_public_inputs(relation, challenge_cfg, params)?;
    if witness.len() != relation.col_count() {
        return Err(AkitaError::InvalidInput(format!(
            "witness length {} does not match relation column count {}",
            witness.len(),
            relation.col_count()
        )));
    }
    Ok(())
}

fn validate_gaertner_public_inputs<F, const D: usize>(
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
) -> ZkResult<()>
where
    F: FieldCore + CanonicalField + PseudoMersenneField,
{
    challenge_cfg
        .validate::<D>()
        .map_err(|e| AkitaError::InvalidInput(format!("invalid challenge config: {e}")))?;
    if params.challenge_l1_bound < challenge_cfg.l1_norm() {
        return Err(AkitaError::InvalidInput(format!(
            "rejection challenge L1 bound {} is smaller than config bound {}",
            params.challenge_l1_bound,
            challenge_cfg.l1_norm()
        )));
    }
    validate_gaertner_rejection_params::<F, D>(relation.col_count(), params)
}

fn validate_gaertner_rejection_params<F, const D: usize>(
    witness_len: usize,
    params: &GaertnerRejectionParams,
) -> ZkResult<()>
where
    F: PseudoMersenneField,
{
    let expected_revealed = witness_len
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidInput("revealed coefficient overflow".to_string()))?;
    if params.revealed_coefficients != expected_revealed {
        return Err(AkitaError::InvalidInput(format!(
            "params reveal {} coefficients, but relation reveals {expected_revealed}",
            params.revealed_coefficients
        )));
    }
    if params.response_bound == 0 {
        return Err(AkitaError::InvalidInput(
            "response_bound must be non-zero".to_string(),
        ));
    }
    let required_mask_bound = params
        .response_bound
        .checked_add(params.beta)
        .ok_or_else(|| AkitaError::InvalidInput("mask bound overflow".to_string()))?;
    if params.mask_bound < required_mask_bound {
        return Err(AkitaError::InvalidInput(
            "mask_bound must cover response_bound + beta".to_string(),
        ));
    }
    if !params.sigma.is_finite() || params.sigma <= 0.0 {
        return Err(AkitaError::InvalidInput(
            "sigma must be finite and positive".to_string(),
        ));
    }
    if !params.rejection_m.is_finite() || params.rejection_m < 1.0 {
        return Err(AkitaError::InvalidInput(
            "rejection_m must be finite and at least one".to_string(),
        ));
    }
    params.validate_no_modular_wrap::<F>()
}

fn absorb_gaertner_public_inputs<F, T, const D: usize>(
    transcript: &mut T,
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaertnerRejectionParams,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(ABSORB_SHAPE, &(D as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SHAPE, &(relation.row_count() as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SHAPE, &(relation.col_count() as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SHAPE, &challenge_cfg.domain_separator_bytes());
    transcript.append_bytes(ABSORB_REJECTION, b"gaertner-public-sign-v1");
    transcript.append_bytes(ABSORB_REJECTION, &params.witness_bound.to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &(params.challenge_l1_bound as u64).to_le_bytes(),
    );
    transcript.append_bytes(ABSORB_REJECTION, &params.beta.to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &(params.revealed_coefficients as u64).to_le_bytes(),
    );
    transcript.append_bytes(ABSORB_REJECTION, &params.response_bound.to_le_bytes());
    transcript.append_bytes(ABSORB_REJECTION, &params.mask_bound.to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &params.width_factor.to_bits().to_le_bytes(),
    );
    transcript.append_bytes(ABSORB_REJECTION, &params.sigma.to_bits().to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &params.rejection_m.to_bits().to_le_bytes(),
    );
    transcript.append_bytes(ABSORB_REJECTION, &params.zk_error_bits.to_le_bytes());
    transcript.append_bytes(ABSORB_REJECTION, &params.tail_error_bits.to_le_bytes());
    for row in &relation.matrix {
        append_ring_vec(transcript, ABSORB_MATRIX, row);
    }
    append_ring_vec(transcript, ABSORB_COMMITMENT, &relation.commitment);
}

fn absorb_public_inputs<F, T, const D: usize>(
    transcript: &mut T,
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &BoxRejectionParams,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(ABSORB_SHAPE, &(D as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SHAPE, &(relation.row_count() as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SHAPE, &(relation.col_count() as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SHAPE, &challenge_cfg.domain_separator_bytes());
    transcript.append_bytes(ABSORB_REJECTION, &params.witness_bound.to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &(params.challenge_l1_bound as u64).to_le_bytes(),
    );
    transcript.append_bytes(ABSORB_REJECTION, &params.beta.to_le_bytes());
    transcript.append_bytes(ABSORB_REJECTION, &params.gamma.to_le_bytes());
    transcript.append_bytes(ABSORB_REJECTION, &params.response_bound.to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &(params.revealed_coefficients as u64).to_le_bytes(),
    );
    for row in &relation.matrix {
        append_ring_vec(transcript, ABSORB_MATRIX, row);
    }
    append_ring_vec(transcript, ABSORB_COMMITMENT, &relation.commitment);
}

fn absorb_gaussian_public_inputs<F, T, const D: usize>(
    transcript: &mut T,
    relation: &AjtaiRelation<F, D>,
    challenge_cfg: &SparseChallengeConfig,
    params: &GaussianRejectionParams,
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(ABSORB_SHAPE, &(D as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SHAPE, &(relation.row_count() as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SHAPE, &(relation.col_count() as u64).to_le_bytes());
    transcript.append_bytes(ABSORB_SHAPE, &challenge_cfg.domain_separator_bytes());
    transcript.append_bytes(ABSORB_REJECTION, b"gaussian-heuristic-v1");
    transcript.append_bytes(ABSORB_REJECTION, &params.witness_bound.to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &(params.challenge_l1_bound as u64).to_le_bytes(),
    );
    transcript.append_bytes(ABSORB_REJECTION, &params.beta.to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &(params.revealed_coefficients as u64).to_le_bytes(),
    );
    transcript.append_bytes(ABSORB_REJECTION, &params.response_bound.to_le_bytes());
    transcript.append_bytes(ABSORB_REJECTION, &params.mask_bound.to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &params.width_factor.to_bits().to_le_bytes(),
    );
    transcript.append_bytes(ABSORB_REJECTION, &params.sigma.to_bits().to_le_bytes());
    transcript.append_bytes(
        ABSORB_REJECTION,
        &params.rejection_m.to_bits().to_le_bytes(),
    );
    transcript.append_bytes(ABSORB_REJECTION, &params.zk_error_bits.to_le_bytes());
    transcript.append_bytes(ABSORB_REJECTION, &params.tail_error_bits.to_le_bytes());
    for row in &relation.matrix {
        append_ring_vec(transcript, ABSORB_MATRIX, row);
    }
    append_ring_vec(transcript, ABSORB_COMMITMENT, &relation.commitment);
}

fn append_ring_vec<F, T, const D: usize>(
    transcript: &mut T,
    label: &[u8],
    values: &[CyclotomicRing<F, D>],
) where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.append_bytes(label, &(values.len() as u64).to_le_bytes());
    for value in values {
        transcript.append_serde(label, value);
    }
}

fn ring_vec_l2_squared<F, const D: usize>(values: &[CyclotomicRing<F, D>]) -> ZkResult<f64>
where
    F: CanonicalField + PseudoMersenneField,
{
    let mut sum = 0.0;
    for ring in values {
        for &coeff in ring.coefficients() {
            let value = centered_i128(coeff)? as f64;
            sum += value * value;
        }
    }
    Ok(sum)
}

fn ring_vec_inner_product_centered<F, const D: usize>(
    a: &[CyclotomicRing<F, D>],
    b: &[CyclotomicRing<F, D>],
) -> ZkResult<i128>
where
    F: CanonicalField + PseudoMersenneField,
{
    if a.len() != b.len() {
        return Err(AkitaError::InvalidInput(format!(
            "ring inner product length mismatch: {} != {}",
            a.len(),
            b.len()
        )));
    }
    let mut sum: i128 = 0;
    for (a_i, b_i) in a.iter().zip(b.iter()) {
        for j in 0..D {
            let av = centered_i128(a_i.coefficients()[j])?;
            let bv = centered_i128(b_i.coefficients()[j])?;
            let product = av.checked_mul(bv).ok_or_else(|| {
                AkitaError::InvalidInput("ring inner product term overflow".to_string())
            })?;
            sum = sum.checked_add(product).ok_or_else(|| {
                AkitaError::InvalidInput("ring inner product accumulator overflow".to_string())
            })?;
        }
    }
    Ok(sum)
}

fn uniform_f64<R>(rng: &mut R) -> f64
where
    R: RngCore + ?Sized,
{
    const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
    (((rng.next_u64() >> 11) as f64) + 0.5) * SCALE
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;
    use akita_field::Prime128OffsetA7F7;
    use akita_transcript::Blake2bTranscript;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Prime128OffsetA7F7;
    type Tr = Blake2bTranscript<F>;
    const D: usize = 64;

    fn ring_from_terms(terms: &[(usize, i128)]) -> CyclotomicRing<F, D> {
        let mut coeffs = [F::zero(); D];
        for &(idx, value) in terms {
            coeffs[idx] = crate::norm::field_from_centered_i128(value).unwrap();
        }
        CyclotomicRing::from_coefficients(coeffs)
    }

    fn challenge_cfg() -> SparseChallengeConfig {
        SparseChallengeConfig::ExactShell {
            count_mag1: 30,
            count_mag2: 12,
        }
    }

    fn test_relation() -> (AjtaiRelation<F, D>, Vec<CyclotomicRing<F, D>>) {
        let a = ring_from_terms(&[(0, 3), (1, 1), (7, -2)]);
        let witness = vec![ring_from_terms(&[(0, 5), (2, -3), (9, 1)])];
        let commitment = vec![a * witness[0]];
        let relation = AjtaiRelation::new(vec![vec![a]], commitment).unwrap();
        (relation, witness)
    }

    #[test]
    fn honest_box_rejected_opening_proof_verifies() {
        let (relation, witness) = test_relation();
        let cfg = challenge_cfg();
        let params = BoxRejectionParams::for_half_acceptance(1, D, &cfg, 16).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let proof =
            prove_ajtai_opening::<F, Tr, _, D>(&relation, &witness, &cfg, &params, &mut rng, 128)
                .unwrap();

        assert!(verify_ajtai_opening::<F, Tr, D>(&relation, &cfg, &params, &proof).unwrap());
    }

    #[test]
    fn tampered_response_fails() {
        let (relation, witness) = test_relation();
        let cfg = challenge_cfg();
        let params = BoxRejectionParams::for_half_acceptance(1, D, &cfg, 16).unwrap();
        let mut rng = StdRng::seed_from_u64(8);
        let mut proof =
            prove_ajtai_opening::<F, Tr, _, D>(&relation, &witness, &cfg, &params, &mut rng, 128)
                .unwrap();

        proof.response[0].coefficients_mut()[0] += F::one();

        assert!(!verify_ajtai_opening::<F, Tr, D>(&relation, &cfg, &params, &proof).unwrap());
    }

    #[test]
    fn interactive_simulated_transcript_verifies() {
        let (relation, _) = test_relation();
        let cfg = challenge_cfg();
        let params = BoxRejectionParams::for_half_acceptance(1, D, &cfg, 16).unwrap();
        let challenge = SparseChallenge {
            positions: vec![0, 5, 11],
            coeffs: vec![1, -1, 2],
        };
        let mut rng = StdRng::seed_from_u64(9);
        let transcript =
            simulate_ajtai_opening_transcript(&relation, &params, challenge, &mut rng).unwrap();

        assert!(verify_ajtai_opening_transcript(&relation, &params, &transcript).unwrap());
    }

    #[test]
    fn out_of_bound_witness_is_rejected() {
        let (relation, mut witness) = test_relation();
        witness[0].coefficients_mut()[3] = crate::norm::field_from_centered_i128(17).unwrap();
        let cfg = challenge_cfg();
        let params = BoxRejectionParams::for_half_acceptance(1, D, &cfg, 16).unwrap();
        let mut rng = StdRng::seed_from_u64(10);

        assert!(prove_ajtai_opening::<F, Tr, _, D>(
            &relation, &witness, &cfg, &params, &mut rng, 128,
        )
        .is_err());
    }

    #[test]
    fn gaussian_heuristic_opening_proof_verifies() {
        let (relation, witness) = test_relation();
        let cfg = challenge_cfg();
        let params = GaussianRejectionParams::for_l2_bound(1, D, &cfg, 16, 16.0, 128, 128).unwrap();
        let mut rng = StdRng::seed_from_u64(11);
        let proof = prove_gaussian_heuristic_ajtai_opening::<F, Tr, _, D>(
            &relation, &witness, &cfg, &params, &mut rng, 512,
        )
        .unwrap();

        assert!(verify_gaussian_heuristic_ajtai_opening::<F, Tr, D>(
            &relation, &cfg, &params, &proof
        )
        .unwrap());
    }

    #[test]
    fn gaertner_opening_proof_verifies() {
        let (relation, witness) = test_relation();
        let cfg = challenge_cfg();
        let params = GaertnerRejectionParams::for_l2_bound(1, D, &cfg, 16, 16.0, 128, 128).unwrap();
        let mut rng = StdRng::seed_from_u64(13);
        let proof = prove_gaertner_ajtai_opening::<F, Tr, _, D>(
            &relation, &witness, &cfg, &params, &mut rng, 512,
        )
        .unwrap();

        assert!(proof.sign == 1 || proof.sign == -1);
        assert!(
            verify_gaertner_ajtai_opening::<F, Tr, D>(&relation, &cfg, &params, &proof).unwrap()
        );
    }

    #[test]
    fn gaertner_tampered_sign_fails() {
        let (relation, witness) = test_relation();
        let cfg = challenge_cfg();
        let params = GaertnerRejectionParams::for_l2_bound(1, D, &cfg, 16, 16.0, 128, 128).unwrap();
        let mut rng = StdRng::seed_from_u64(14);
        let mut proof = prove_gaertner_ajtai_opening::<F, Tr, _, D>(
            &relation, &witness, &cfg, &params, &mut rng, 512,
        )
        .unwrap();
        proof.sign = -proof.sign;

        assert!(
            !verify_gaertner_ajtai_opening::<F, Tr, D>(&relation, &cfg, &params, &proof).unwrap()
        );
    }
}
