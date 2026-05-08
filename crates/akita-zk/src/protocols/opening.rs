//! Ring-native Ajtai opening proof with exact box rejection.

use crate::compact::CompactRingVec;
use crate::error::ZkResult;
use crate::norm::{ring_vec_within_infinity_bound, sample_ring_vec_box};
use crate::rejection::BoxRejectionParams;
use crate::relations::AjtaiRelation;
use crate::ring_ext::{add_ring_vecs, mul_sparse_challenge_vec, sub_ring_vecs};
use akita_algebra::CyclotomicRing;
use akita_challenges::{sample_sparse_challenges, SparseChallenge, SparseChallengeConfig};
use akita_field::{AkitaError, CanonicalField, FieldCore, PseudoMersenneField};
use akita_serialization::{AkitaSerialize, Compress};
use akita_transcript::Transcript;
use rand_core::RngCore;

const DOMAIN_AKITA_ZK_OPENING: &[u8] = b"ak/zk/open";
const ABSORB_SHAPE: &[u8] = b"ak/zk/a/sh";
const ABSORB_REJECTION: &[u8] = b"ak/zk/a/rj";
const ABSORB_MATRIX: &[u8] = b"ak/zk/a/m";
const ABSORB_COMMITMENT: &[u8] = b"ak/zk/a/t";
const ABSORB_ANNOUNCEMENT: &[u8] = b"ak/zk/a/a";
const CHALLENGE_OPENING: &[u8] = b"ak/zk/c";

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
    F: FieldCore + CanonicalField + PseudoMersenneField,
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
}
