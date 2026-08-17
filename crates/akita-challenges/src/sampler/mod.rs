//! Sparse ring fold challenge sampling via Fiat-Shamir with PRG expansion.
//!
//! After the prover's folded witness message `v` is absorbed, the protocol
//! samples sparse ring elements `c` used to fold the witness toward the next
//! commitment. Every [`SparseChallengeConfig`] uses the signed-sparse sampler:
//! `count_pm1` coefficients at ±1 and `count_pm2` at ±2.

mod op_norm;
mod op_norm_accumulate;
mod position_sample;
mod signed_sparse;
mod xof;

pub(crate) use position_sample::MAX_STACK_RING_DIM;
pub(crate) use signed_sparse::SignedSparseScratch;
pub(crate) use xof::XofCursor;

#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::Transcript;
use std::sync::{Arc, LazyLock};

use crate::{OperatorNormRejection, SparseChallenge, SparseChallengeConfig};

use op_norm::OpNormTable;

const OP_NORM_PREDICATE_SCALE: u32 = 48;
const MAX_OP_NORM_ATTEMPTS: usize = 4096;
const PACKING_CHALLENGE_BATCH_SIZE: usize = 128;
static D64_SELECTIVE_L2_OP_NORM_TABLE: LazyLock<Result<Arc<OpNormTable>, &'static str>> =
    LazyLock::new(|| {
        let config = crate::D64_SELECTIVE_L2_CHALLENGE_CONFIG;
        let policy = OperatorNormRejection::D64_SELECTIVE_L2;
        let table = OpNormTable::new(
            64,
            OP_NORM_PREDICATE_SCALE,
            config.l1_norm() as u64,
            u64::from(policy.threshold),
        )
        .map_err(|_| "failed to initialize the D64 selective-L2 operator-norm table")?;
        if !table.strict_threshold_contains_shrunken_subset(
            policy.fractional_bits,
            policy.root_coordinate_error_units,
            u64::from(policy.threshold),
            policy.rounding_margin_units,
        ) {
            return Err("the D64 operator-norm table does not contain its certified subset");
        }
        Ok(Arc::new(table))
    });
static D128_SELECTIVE_L2_OP_NORM_TABLE: LazyLock<Result<Arc<OpNormTable>, &'static str>> =
    LazyLock::new(|| {
        let config = crate::D128_SELECTIVE_L2_CHALLENGE_CONFIG;
        let policy = OperatorNormRejection::D128_SELECTIVE_L2;
        let table = OpNormTable::new(
            128,
            OP_NORM_PREDICATE_SCALE,
            config.l1_norm() as u64,
            u64::from(policy.threshold),
        )
        .map_err(|_| "failed to initialize the D128 selective-L2 operator-norm table")?;
        if !table.strict_threshold_contains_shrunken_subset(
            policy.fractional_bits,
            policy.root_coordinate_error_units,
            u64::from(policy.threshold),
            policy.rounding_margin_units,
        ) {
            return Err("the D128 operator-norm table does not contain its certified subset");
        }
        Ok(Arc::new(table))
    });

pub(crate) fn sample_challenges_from_xof_cursor(
    cursor: &mut XofCursor,
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
    rejection: Option<OperatorNormRejection>,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let Some(rejection) = rejection else {
        return SignedSparseScratch::sample_challenges(cursor, ring_d, n, cfg);
    };
    rejection
        .validate(ring_d, cfg)
        .map_err(|error| AkitaError::InvalidSetup(error.into()))?;
    let table = match rejection {
        OperatorNormRejection::D64_SELECTIVE_L2 => &*D64_SELECTIVE_L2_OP_NORM_TABLE,
        OperatorNormRejection::D128_SELECTIVE_L2 => &*D128_SELECTIVE_L2_OP_NORM_TABLE,
        _ => {
            return Err(AkitaError::InvalidSetup(
                "unsupported operator-norm rejection policy".into(),
            ));
        }
    };
    let table = Arc::clone(
        table
            .as_ref()
            .map_err(|message| AkitaError::InvalidSetup((*message).into()))?,
    );
    let mut scratch = SignedSparseScratch::new(cfg.count_pm1, cfg.count_pm2);
    let mut challenges = Vec::with_capacity(n);
    for _ in 0..n {
        let mut accepted = None;
        for _ in 0..MAX_OP_NORM_ATTEMPTS {
            scratch.sample(cursor, ring_d, cfg.count_pm1, cfg.count_pm2)?;
            if table.accept_strict_parts(
                scratch.positions(),
                scratch.coeffs(),
                u64::from(rejection.threshold),
            )? {
                accepted = Some(scratch.take_challenge());
                break;
            }
        }
        challenges.push(accepted.ok_or_else(|| {
            AkitaError::InvalidInput(format!(
                "operator-norm rejection exceeded {MAX_OP_NORM_ATTEMPTS} attempts"
            ))
        })?);
    }
    Ok(challenges)
}

pub(crate) fn sample_batched_challenges_from_seed(
    seed: &[u8],
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let num_batches = n.div_ceil(PACKING_CHALLENGE_BATCH_SIZE);
    let sample_batch = |batch_index: usize| {
        let canonical_batch_index = u64::try_from(batch_index).map_err(|_| {
            AkitaError::InvalidSetup("sparse challenge batch index exceeds u64".into())
        })?;
        let mut cursor = XofCursor::from_batched_seed(seed, canonical_batch_index);
        let mut scratch = SignedSparseScratch::new(cfg.count_pm1, cfg.count_pm2);
        let start = batch_index
            .checked_mul(PACKING_CHALLENGE_BATCH_SIZE)
            .ok_or_else(|| AkitaError::InvalidSetup("sparse challenge batch overflow".into()))?;
        let batch_len = n.saturating_sub(start).min(PACKING_CHALLENGE_BATCH_SIZE);
        let mut batch = Vec::with_capacity(batch_len);
        for _ in 0..batch_len {
            scratch.sample(&mut cursor, ring_d, cfg.count_pm1, cfg.count_pm2)?;
            batch.push(scratch.take_challenge());
        }
        Ok::<_, AkitaError>(batch)
    };
    #[cfg(feature = "parallel")]
    {
        const PARALLEL_THRESHOLD: usize = 1 << 14;
        let work = n
            .checked_mul(cfg.count_pm1.saturating_add(cfg.count_pm2))
            .ok_or_else(|| AkitaError::InvalidSetup("sparse challenge work overflow".into()))?;
        if num_batches > 1 && work >= PARALLEL_THRESHOLD {
            return (0..num_batches)
                .into_par_iter()
                .map(sample_batch)
                .collect::<Result<Vec<_>, _>>()
                .map(|batches| batches.into_iter().flatten().collect());
        }
    }
    (0..num_batches)
        .map(sample_batch)
        .collect::<Result<Vec<_>, _>>()
        .map(|batches| batches.into_iter().flatten().collect())
}

/// Sample `n` sparse ring fold challenges from a transcript.
///
/// # Errors
///
/// Returns an error if challenge sampling fails.
#[tracing::instrument(skip_all, name = "sample_sparse_challenges")]
pub fn sample_sparse_challenges<F, T>(
    transcript: &mut T,
    label: &[u8],
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
    grind_nonce: u32,
) -> Result<Vec<SparseChallenge>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    if ring_d > MAX_STACK_RING_DIM {
        return Err(AkitaError::InvalidInput(format!(
            "ring dimension {ring_d} exceeds supported stack sampler limit ({MAX_STACK_RING_DIM})"
        )));
    }
    cfg.validate_dyn(ring_d)
        .map_err(|e| AkitaError::InvalidInput(format!("invalid sparse challenge config: {e}")))?;

    let domain_sep = cfg.domain_separator_bytes();
    let mut absorb_buf = Vec::with_capacity(label.len() + 8 + 8 + domain_sep.len() + 4);
    absorb_buf.extend_from_slice(label);
    absorb_buf.extend_from_slice(&(n as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&(ring_d as u64).to_le_bytes());
    absorb_buf.extend_from_slice(&domain_sep);
    absorb_buf.extend_from_slice(&grind_nonce.to_le_bytes());

    transcript.append_bytes(ABSORB_SPARSE_CHALLENGE, &absorb_buf);
    let seed = transcript.challenge_bytes(CHALLENGE_SPARSE_CHALLENGE, 32);
    let mut cursor = XofCursor::from_seed(&seed);
    sample_challenges_from_xof_cursor(&mut cursor, ring_d, n, cfg, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::xof::XofCursor;

    #[test]
    fn pm1_only_matches_pm2_zero_sampler() {
        let ring_d = 128;
        let cfg = SparseChallengeConfig::pm1_only(31);
        let seed = [7u8; 32];
        let legacy = {
            let mut cursor = XofCursor::from_seed(&seed);
            let mut scratch = SignedSparseScratch::new(31, 0);
            scratch.sample(&mut cursor, ring_d, 31, 0).unwrap();
            scratch.take_challenge()
        };
        let unified = {
            let mut cursor = XofCursor::from_seed(&seed);
            SignedSparseScratch::sample_challenges(&mut cursor, ring_d, 1, &cfg)
                .unwrap()
                .pop()
                .expect("one challenge")
        };
        assert_eq!(legacy.positions, unified.positions);
        assert_eq!(legacy.coeffs, unified.coeffs);
    }

    #[test]
    fn batched_draw_matches_canonical_substreams() {
        let ring_d = 64;
        let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d).unwrap();
        let seed = [11u8; 32];
        let challenge_count = 2 * PACKING_CHALLENGE_BATCH_SIZE;
        let batch =
            sample_batched_challenges_from_seed(&seed, ring_d, challenge_count, &cfg).unwrap();
        let expected = (0..2)
            .flat_map(|batch_index| {
                let mut cursor = XofCursor::from_batched_seed(&seed, batch_index);
                let mut scratch = SignedSparseScratch::new(cfg.count_pm1, cfg.count_pm2);
                (0..PACKING_CHALLENGE_BATCH_SIZE)
                    .map(|_| {
                        scratch
                            .sample(&mut cursor, ring_d, cfg.count_pm1, cfg.count_pm2)
                            .unwrap();
                        scratch.take_challenge()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(batch, expected);
        assert_ne!(batch[0], batch[1]);
    }
}
