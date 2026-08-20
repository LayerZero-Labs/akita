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
pub(crate) use xof::{IndexedXofPrefix, XofCursor};

use akita_error::AkitaError;
use akita_transcript::labels::{ABSORB_SPARSE_CHALLENGE, CHALLENGE_SPARSE_CHALLENGE};
use akita_transcript::Transcript;
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::sync::{Arc, LazyLock};

use crate::{OperatorNormRejection, SparseChallenge, SparseChallengeConfig};

use op_norm::OpNormTable;

const OP_NORM_PREDICATE_SCALE: u32 = 48;
const MAX_OP_NORM_ATTEMPTS: usize = 4096;
#[cfg(feature = "parallel")]
const PARALLEL_SAMPLING_WORK_THRESHOLD: usize = 1 << 14;
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

#[derive(Clone)]
struct IndexedSamplingPolicy {
    rejection: Option<OperatorNormRejection>,
    op_norm_table: Option<Arc<OpNormTable>>,
}

impl IndexedSamplingPolicy {
    fn new(
        ring_d: usize,
        cfg: &SparseChallengeConfig,
        rejection: Option<OperatorNormRejection>,
    ) -> Result<Self, AkitaError> {
        let Some(rejection) = rejection else {
            return Ok(Self {
                rejection: None,
                op_norm_table: None,
            });
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
        let op_norm_table = Arc::clone(
            table
                .as_ref()
                .map_err(|message| AkitaError::InvalidSetup((*message).into()))?,
        );
        Ok(Self {
            rejection: Some(rejection),
            op_norm_table: Some(op_norm_table),
        })
    }
}

struct IndexedChallengeWorker {
    prefix: IndexedXofPrefix,
    cursor: XofCursor,
    scratch: SignedSparseScratch,
    policy: IndexedSamplingPolicy,
}

impl IndexedChallengeWorker {
    fn new(seed: &[u8], cfg: &SparseChallengeConfig, policy: &IndexedSamplingPolicy) -> Self {
        let prefix = IndexedXofPrefix::new(seed);
        let cursor = XofCursor::from_indexed_prefix(&prefix, 0);
        Self {
            prefix,
            cursor,
            scratch: SignedSparseScratch::new(cfg.count_pm1, cfg.count_pm2),
            policy: policy.clone(),
        }
    }

    fn sample(
        &mut self,
        coordinate_index: u64,
        ring_d: usize,
        cfg: &SparseChallengeConfig,
    ) -> Result<SparseChallenge, AkitaError> {
        self.cursor
            .reset_indexed_prefix(&self.prefix, coordinate_index);
        let Some(rejection) = self.policy.rejection else {
            self.scratch
                .sample(&mut self.cursor, ring_d, cfg.count_pm1, cfg.count_pm2)?;
            return Ok(self.scratch.take_challenge());
        };
        let table = self.policy.op_norm_table.as_ref().ok_or_else(|| {
            AkitaError::InvalidSetup("operator-norm rejection table is unavailable".into())
        })?;
        for _ in 0..MAX_OP_NORM_ATTEMPTS {
            self.scratch
                .sample(&mut self.cursor, ring_d, cfg.count_pm1, cfg.count_pm2)?;
            if table.accept_strict_parts(
                self.scratch.positions(),
                self.scratch.coeffs(),
                u64::from(rejection.threshold),
            )? {
                return Ok(self.scratch.take_challenge());
            }
        }
        Err(AkitaError::InvalidInput(format!(
            "operator-norm rejection exceeded {MAX_OP_NORM_ATTEMPTS} attempts at coordinate {coordinate_index}"
        )))
    }
}

pub(crate) fn sample_indexed_challenges_from_seed(
    seed: &[u8],
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
    rejection: Option<OperatorNormRejection>,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    u64::try_from(n - 1).map_err(|_| {
        AkitaError::InvalidSetup("sparse challenge coordinate index exceeds u64".into())
    })?;
    let policy = IndexedSamplingPolicy::new(ring_d, cfg, rejection)?;
    #[cfg(feature = "parallel")]
    {
        let work = n
            .checked_mul(cfg.weight())
            .ok_or_else(|| AkitaError::InvalidSetup("sparse challenge work overflow".into()))?;
        if work >= PARALLEL_SAMPLING_WORK_THRESHOLD {
            return (0..n)
                .into_par_iter()
                .map_init(
                    || IndexedChallengeWorker::new(seed, cfg, &policy),
                    |worker, index| {
                        let coordinate_index = u64::try_from(index).map_err(|_| {
                            AkitaError::InvalidSetup(
                                "sparse challenge coordinate index exceeds u64".into(),
                            )
                        })?;
                        worker.sample(coordinate_index, ring_d, cfg)
                    },
                )
                .collect();
        }
    }
    sample_indexed_challenges_sequential(seed, ring_d, n, cfg, &policy)
}

fn sample_indexed_challenges_sequential(
    seed: &[u8],
    ring_d: usize,
    n: usize,
    cfg: &SparseChallengeConfig,
    policy: &IndexedSamplingPolicy,
) -> Result<Vec<SparseChallenge>, AkitaError> {
    let mut worker = IndexedChallengeWorker::new(seed, cfg, policy);
    (0..n)
        .map(|index| {
            let coordinate_index = u64::try_from(index).map_err(|_| {
                AkitaError::InvalidSetup("sparse challenge coordinate index exceeds u64".into())
            })?;
            worker.sample(coordinate_index, ring_d, cfg)
        })
        .collect()
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
    F: Field + CanonicalEncoding,
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
    sample_indexed_challenges_from_seed(&seed, ring_d, n, cfg, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::xof::XofCursor;

    #[test]
    fn pm1_only_matches_pm2_zero_sampler() {
        let ring_d = 128;
        let seed = [7u8; 32];
        let legacy = {
            let mut cursor = XofCursor::from_seed(&seed);
            let mut scratch = SignedSparseScratch::new(31, 0);
            scratch.sample(&mut cursor, ring_d, 31, 0).unwrap();
            scratch.take_challenge()
        };
        let unified = {
            let mut cursor = XofCursor::from_seed(&seed);
            let mut scratch = SignedSparseScratch::new(31, 0);
            scratch.sample(&mut cursor, ring_d, 31, 0).unwrap();
            scratch.take_challenge()
        };
        assert_eq!(legacy.positions, unified.positions);
        assert_eq!(legacy.coeffs, unified.coeffs);
    }

    #[test]
    fn indexed_draw_matches_individual_coordinate_streams() {
        let ring_d = 64;
        let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d).unwrap();
        let seed = [11u8; 32];
        let challenge_count = 257;
        let indexed =
            sample_indexed_challenges_from_seed(&seed, ring_d, challenge_count, &cfg, None)
                .unwrap();
        let policy = IndexedSamplingPolicy::new(ring_d, &cfg, None).unwrap();
        let expected = (0..challenge_count)
            .map(|index| {
                IndexedChallengeWorker::new(&seed, &cfg, &policy)
                    .sample(index as u64, ring_d, &cfg)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(indexed, expected);
        assert_ne!(indexed[0], indexed[1]);
    }

    #[test]
    fn reprogramming_one_coordinate_leaves_every_other_coordinate_unchanged() {
        let ring_d = 64;
        let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d).unwrap();
        let seed = [23u8; 32];
        let alternate_seed = [29u8; 32];
        let mut challenges =
            sample_indexed_challenges_from_seed(&seed, ring_d, 12, &cfg, None).unwrap();
        let before = challenges.clone();
        let policy = IndexedSamplingPolicy::new(ring_d, &cfg, None).unwrap();
        let mut worker = IndexedChallengeWorker::new(&alternate_seed, &cfg, &policy);
        challenges[7] = worker.sample(7, ring_d, &cfg).unwrap();
        assert_ne!(challenges[7], before[7]);
        for index in (0..challenges.len()).filter(|&index| index != 7) {
            assert_eq!(challenges[index], before[index]);
        }
    }

    #[test]
    fn indexed_operator_rejection_uses_independent_coordinate_streams() {
        let seed = [31u8; 32];
        for (ring_d, cfg, rejection) in [
            (
                64,
                crate::D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                OperatorNormRejection::D64_SELECTIVE_L2,
            ),
            (
                128,
                crate::D128_SELECTIVE_L2_CHALLENGE_CONFIG,
                OperatorNormRejection::D128_SELECTIVE_L2,
            ),
        ] {
            let indexed =
                sample_indexed_challenges_from_seed(&seed, ring_d, 8, &cfg, Some(rejection))
                    .unwrap();
            let policy = IndexedSamplingPolicy::new(ring_d, &cfg, Some(rejection)).unwrap();
            let individual = (0..8)
                .map(|index| {
                    IndexedChallengeWorker::new(&seed, &cfg, &policy)
                        .sample(index, ring_d, &cfg)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(indexed, individual);
            assert!(indexed.iter().all(|challenge| {
                challenge.positions.len() == cfg.weight()
                    && challenge.coeffs.iter().filter(|&&c| c.abs() == 1).count() == cfg.count_pm1
                    && challenge.coeffs.iter().filter(|&&c| c.abs() == 2).count() == cfg.count_pm2
            }));
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn indexed_parallel_and_sequential_sampling_match() {
        let ring_d = 64;
        let cfg = SparseChallengeConfig::production_for_ring_dim(ring_d).unwrap();
        let seed = [37u8; 32];
        let challenge_count = 400;
        let parallel =
            sample_indexed_challenges_from_seed(&seed, ring_d, challenge_count, &cfg, None)
                .unwrap();
        let policy = IndexedSamplingPolicy::new(ring_d, &cfg, None).unwrap();
        let sequential =
            sample_indexed_challenges_sequential(&seed, ring_d, challenge_count, &cfg, &policy)
                .unwrap();
        assert_eq!(parallel, sequential);
    }
}
