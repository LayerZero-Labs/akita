//! Sampled sparse challenges for one folding round.

use crate::{SparseChallenge, SparseChallengeConfig};
use akita_error::AkitaError;
#[cfg(feature = "parallel")]
use jolt_field::solinas::parallel::*;
use jolt_field::{ExtField, Field, Ring};
use std::sync::{Arc, OnceLock};

#[cfg(feature = "parallel")]
// Tuned with `benches/sparse_challenge.rs::bench_sparse_evaluation` on an
// Apple M4 Max (16 cores, 64 GiB). Near the crossover, two to four coarse
// leaves amortize Rayon dispatch.
const PARALLEL_COARSE_LEAF_TERMS: usize = 1 << 15;
#[cfg(feature = "parallel")]
// Larger batches expose enough work to keep finer leaves busy.
const PARALLEL_FINE_LEAF_TERMS: usize = 1 << 13;
#[cfg(feature = "parallel")]
// The fine tier starts above the largest measured coarse-leaf sweet spot.
const PARALLEL_FINE_TOTAL_TERMS: usize = 1 << 17;

#[derive(Debug, Clone)]
enum ChallengeStorage {
    Sparse(Arc<[SparseChallenge]>),
    // Retain the decoded buffer rather than copying it into another allocation.
    Quartets(Arc<Vec<[u64; 4]>>),
}

/// Stage-1 fold challenges in claim-major block order.
#[derive(Debug, Clone)]
pub struct Challenges {
    /// Per-(claim, block) challenges, with the representation explicit.
    storage: ChallengeStorage,
    /// Exact number of live blocks packed into one claim.
    num_live_blocks_per_claim: usize,
    /// Number of claims represented by this vector.
    num_claims: usize,
    // Only the private sampler or an injective embedding may set this cache.
    sampled_family: Option<(usize, SparseChallengeConfig)>,
    embedding_stride: usize,
    materialized: Arc<OnceLock<Vec<SparseChallenge>>>,
}

impl PartialEq for Challenges {
    fn eq(&self, other: &Self) -> bool {
        self.num_live_blocks_per_claim == other.num_live_blocks_per_claim
            && self.num_claims == other.num_claims
            && ((self.embedding_stride == other.embedding_stride
                && match (&self.storage, &other.storage) {
                    (ChallengeStorage::Sparse(a), ChallengeStorage::Sparse(b)) => Arc::ptr_eq(a, b),
                    (ChallengeStorage::Quartets(a), ChallengeStorage::Quartets(b)) => {
                        Arc::ptr_eq(a, b)
                    }
                    _ => false,
                })
                || self.as_slice() == other.as_slice())
    }
}
impl Eq for Challenges {}
impl Challenges {
    pub(crate) fn from_sampled(
        challenges: Vec<SparseChallenge>,
        blocks: usize,
        claims: usize,
        ring_d: usize,
        config: SparseChallengeConfig,
    ) -> Result<Self, AkitaError> {
        let mut result = Self::from_sparse(challenges, blocks, claims)?;
        result.sampled_family = Some((ring_d, config));
        Ok(result)
    }

    pub(crate) fn from_checked_quartets(
        packed: Vec<[u64; 4]>,
        blocks: usize,
        claims: usize,
    ) -> Result<Self, AkitaError> {
        if blocks.checked_mul(claims) != Some(packed.len()) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(Self {
            storage: ChallengeStorage::Quartets(Arc::new(packed)),
            num_live_blocks_per_claim: blocks,
            num_claims: claims,
            // The authenticated quartet format is exactly D64/31/10.
            sampled_family: Some((
                64,
                SparseChallengeConfig {
                    count_pm1: 31,
                    count_pm2: 10,
                },
            )),
            embedding_stride: 1,
            materialized: Arc::new(OnceLock::new()),
        })
    }
    /// Whether immutable challenge storage was generated in this checked family.
    /// Values supplied through `from_sparse` always require full validation.
    #[must_use]
    pub fn is_sampled_family(&self, ring_d: usize, config: &SparseChallengeConfig) -> bool {
        self.sampled_family
            .as_ref()
            .is_some_and(|(d, c)| *d == ring_d && c == config)
    }

    /// Construct challenges from a sampled vector and its claim/block layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector length does not match the layout.
    pub fn from_sparse(
        challenges: Vec<SparseChallenge>,
        num_live_blocks_per_claim: usize,
        num_claims: usize,
    ) -> Result<Self, AkitaError> {
        let expected = num_live_blocks_per_claim
            .checked_mul(num_claims)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge count overflow".to_string()))?;
        if challenges.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: challenges.len(),
            });
        }
        Ok(Self {
            storage: ChallengeStorage::Sparse(challenges.into()),
            num_live_blocks_per_claim,
            num_claims,
            sampled_family: None,
            embedding_stride: 1,
            materialized: Arc::new(OnceLock::new()),
        })
    }

    /// Return the sparse challenges in claim-major block order.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[SparseChallenge] {
        match &self.storage {
            ChallengeStorage::Quartets(codes) => self
                .materialized
                .get_or_init(|| {
                    codes
                        .iter()
                        .map(|words| {
                            let mut p1 = crate::SparseChallengePositions::new();
                            let mut p2 = crate::SparseChallengePositions::new();
                            let mut c1 = crate::SparseChallengeCoefficients::new();
                            let mut c2 = crate::SparseChallengeCoefficients::new();
                            for q in 0..16 {
                                let mut code = (words[q / 4] >> (10 * (q % 4))) & 1023;
                                for i in 0..4 {
                                    let c = (code % 5) as i8 - 2;
                                    code /= 5;
                                    let p = ((4 * q + i) * self.embedding_stride) as u32;
                                    if c.unsigned_abs() == 1 {
                                        p1.push(p);
                                        c1.push(c);
                                    }
                                    if c.unsigned_abs() == 2 {
                                        p2.push(p);
                                        c2.push(c);
                                    }
                                }
                            }
                            p1.extend(p2);
                            c1.extend(c2);
                            SparseChallenge {
                                positions: p1,
                                coeffs: c1,
                            }
                        })
                        .collect()
                })
                .as_slice(),
            ChallengeStorage::Sparse(challenges) if self.embedding_stride == 1 => challenges,
            ChallengeStorage::Sparse(challenges) => {
                self.materialized
                    .get_or_init(|| {
                        challenges
                            .iter()
                            .map(|challenge| {
                                // The checked embedding constructor bounds every product by u32::MAX.
                                SparseChallenge {
                                    positions: challenge
                                        .positions
                                        .iter()
                                        .map(|&p| ((p as usize) * self.embedding_stride) as u32)
                                        .collect(),
                                    coeffs: crate::SparseChallengeCoefficients::from_slice(
                                        &challenge.coeffs,
                                    ),
                                }
                            })
                            .collect()
                    })
                    .as_slice()
            }
        }
    }

    /// Number of block challenges represented by this value.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.storage {
            ChallengeStorage::Sparse(challenges) => challenges.len(),
            ChallengeStorage::Quartets(codes) => codes.len(),
        }
    }

    /// Whether this challenge set is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of claims represented by this challenge set.
    #[inline]
    #[must_use]
    pub fn num_claims(&self) -> usize {
        self.num_claims
    }

    /// Number of logical block challenges per claim.
    #[inline]
    #[must_use]
    pub fn num_live_blocks_per_claim(&self) -> usize {
        self.num_live_blocks_per_claim
    }

    /// Evaluate one challenge at the precomputed `alpha` powers.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range or the challenge is invalid.
    pub fn eval_at_pows<F, E>(&self, index: usize, alpha_pows: &[E]) -> Result<E, AkitaError>
    where
        F: Field + Ring,
        E: Field + ExtField<F>,
    {
        self.as_slice()
            .get(index)
            .ok_or_else(|| {
                AkitaError::InvalidInput(format!(
                    "challenge index {index} out of range for {} challenges",
                    self.len()
                ))
            })?
            .eval_at_pows::<F, E>(alpha_pows)
    }

    /// Evaluate every challenge at the precomputed `alpha` powers.
    ///
    /// # Errors
    ///
    /// Returns an error if the powers or any challenge are invalid.
    pub fn evals_at_pows<F, E>(&self, alpha_pows: &[E]) -> Result<Vec<E>, AkitaError>
    where
        F: Field + Ring,
        E: Field + ExtField<F>,
    {
        match (&self.storage, self.sampled_family) {
            (ChallengeStorage::Quartets(codes), Some((d, _))) if d == alpha_pows.len() => {
                return self.evaluate_quartets::<F, E>(codes, alpha_pows);
            }
            _ => {}
        }
        let sampled = self
            .sampled_family
            .as_ref()
            .filter(|(d, _)| *d == alpha_pows.len());
        let doubles = if sampled.is_some_and(|(_, config)| config.count_pm2 != 0) {
            alpha_pows.iter().map(|&v| v + v).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let source = match (&self.storage, sampled) {
            (ChallengeStorage::Sparse(challenges), Some(_)) => challenges.as_ref(),
            _ => self.as_slice(),
        };
        let evaluate = |challenge: &SparseChallenge| {
            if sampled.is_some() {
                // Sampler provenance guarantees the shape, range and +/-1/2 family.
                // Bounds-checked indexing is retained at this immutable boundary.
                Ok(E::signed_sum(
                    challenge.positions.iter().zip(&challenge.coeffs).map(
                        |(&position, &coefficient)| {
                            let powers = if coefficient == 2 || coefficient == -2 {
                                &doubles[..]
                            } else {
                                alpha_pows
                            };
                            (
                                &powers[(position as usize) * self.embedding_stride],
                                coefficient < 0,
                            )
                        },
                    ),
                ))
            } else {
                challenge.eval_at_pows::<F, E>(alpha_pows)
            }
        };
        #[cfg(feature = "parallel")]
        let parallel_candidate = {
            // Same M4 Max benchmark above: smaller batches never amortized
            // the initial Rayon dispatch, independent of sparse weight.
            const MIN_PARALLEL_CHALLENGES: usize = 1 << 9;
            source.len() >= MIN_PARALLEL_CHALLENGES
        };
        #[cfg(feature = "parallel")]
        if parallel_candidate {
            let sparse_terms = source.iter().try_fold(0usize, |sum, challenge| {
                sum.checked_add(challenge.positions.len()).ok_or_else(|| {
                    AkitaError::InvalidInput("sparse challenge term count overflow".into())
                })
            })?;
            if sparse_terms <= PARALLEL_COARSE_LEAF_TERMS {
                return source.iter().map(evaluate).collect();
            }
            let leaf_terms = if sparse_terms > PARALLEL_FINE_TOTAL_TERMS {
                PARALLEL_FINE_LEAF_TERMS
            } else {
                PARALLEL_COARSE_LEAF_TERMS
            };
            // Production batches have a uniform sparse weight. Pricing leaves
            // from the exact average also keeps arbitrary valid batches linear
            // without rescanning their terms at every recursive fork.
            let average_terms = sparse_terms.div_ceil(source.len());
            let leaf_challenges = (leaf_terms / average_terms).max(1);
            let mut evaluations = Vec::new();
            evaluations.try_reserve_exact(source.len()).map_err(|_| {
                AkitaError::InvalidInput("challenge evaluation allocation failed".into())
            })?;
            evaluations.resize(source.len(), E::zero());
            evaluations
                .par_chunks_mut(leaf_challenges)
                .zip(source.par_chunks(leaf_challenges))
                .try_for_each(|(evaluation_chunk, challenge_chunk)| {
                    for (evaluation, challenge) in evaluation_chunk.iter_mut().zip(challenge_chunk)
                    {
                        *evaluation = evaluate(challenge)?;
                    }
                    Ok::<(), AkitaError>(())
                })?;
            return Ok(evaluations);
        }

        source.iter().map(evaluate).collect()
    }

    fn evaluate_quartets<F: Field + Ring, E: Field + ExtField<F>>(
        &self,
        codes: &[[u64; 4]],
        powers: &[E],
    ) -> Result<Vec<E>, AkitaError> {
        let mut tables = Vec::with_capacity(16 * 625);
        {
            for q in 0..16 {
                let mut p = [E::zero(); 4];
                for (i, value) in p.iter_mut().enumerate() {
                    *value = *powers
                        .get((q * 4 + i) * self.embedding_stride)
                        .ok_or(AkitaError::InvalidProof)?;
                }
                let sum = p.iter().copied().fold(E::zero(), |a, b| a + b);
                let mut v3 = -(sum + sum);
                for _ in 0..5 {
                    let mut v2 = v3;
                    for _ in 0..5 {
                        let mut v1 = v2;
                        for _ in 0..5 {
                            let mut v0 = v1;
                            for _ in 0..5 {
                                tables.push(v0);
                                v0 += p[0];
                            }
                            v1 += p[1];
                        }
                        v2 += p[2];
                    }
                    v3 += p[3];
                }
            }
        }
        let mut output = Vec::with_capacity(codes.len());
        for words in codes {
            let mut sum = E::zero();
            macro_rules! add_quartet {
                ($q:literal) => {{
                    let code = ((words[$q / 4] >> (10 * ($q % 4))) & 1023) as usize;
                    sum += *tables
                        .get($q * 625 + code)
                        .ok_or(AkitaError::InvalidProof)?;
                }};
            }
            add_quartet!(0);
            add_quartet!(1);
            add_quartet!(2);
            add_quartet!(3);
            add_quartet!(4);
            add_quartet!(5);
            add_quartet!(6);
            add_quartet!(7);
            add_quartet!(8);
            add_quartet!(9);
            add_quartet!(10);
            add_quartet!(11);
            add_quartet!(12);
            add_quartet!(13);
            add_quartet!(14);
            add_quartet!(15);
            output.push(sum);
        }
        Ok(output)
    }

    /// Select complete claims in the requested order.
    ///
    /// # Errors
    ///
    /// Returns an error if any claim index is out of range.
    pub fn select_claims(&self, claim_indices: &[usize]) -> Result<Self, AkitaError> {
        let capacity = claim_indices
            .len()
            .checked_mul(self.num_live_blocks_per_claim)
            .ok_or_else(|| AkitaError::InvalidSetup("challenge count overflow".to_string()))?;
        let mut selected = Vec::with_capacity(capacity);
        for &claim_index in claim_indices {
            let start = claim_index
                .checked_mul(self.num_live_blocks_per_claim)
                .ok_or_else(|| AkitaError::InvalidSetup("challenge offset overflow".to_string()))?;
            let end = start
                .checked_add(self.num_live_blocks_per_claim)
                .ok_or_else(|| AkitaError::InvalidSetup("challenge offset overflow".to_string()))?;
            selected.extend_from_slice(self.as_slice().get(start..end).ok_or(
                AkitaError::InvalidSize {
                    expected: end,
                    actual: self.len(),
                },
            )?);
        }
        Self::from_sparse(
            selected,
            self.num_live_blocks_per_claim,
            claim_indices.len(),
        )
    }

    /// Embed subring challenge positions into an ambient A ring.
    ///
    /// Coefficients and claim/block order are preserved exactly. Only each
    /// position `j` changes to `embedding_stride * j`.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent, a source challenge is
    /// malformed, or an embedded position overflows or leaves the A ring.
    pub fn embed_subring_positions(
        &self,
        challenge_subring_dimension: usize,
        embedding_stride: usize,
        ambient_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if challenge_subring_dimension == 0
            || embedding_stride == 0
            || challenge_subring_dimension
                .checked_mul(embedding_stride)
                .filter(|&dimension| dimension == ambient_ring_dimension)
                .is_none()
        {
            return Err(AkitaError::InvalidInput(
                "subring challenge embedding dimensions are inconsistent".into(),
            ));
        }
        let sampled = self
            .sampled_family
            .as_ref()
            .is_some_and(|(d, _)| *d == challenge_subring_dimension);
        if embedding_stride == 1 {
            if !sampled {
                for challenge in self.as_slice() {
                    challenge.validate_dyn(challenge_subring_dimension)?;
                }
            }
            return Ok(self.clone());
        }
        if sampled
            && ambient_ring_dimension
                .checked_sub(1)
                .is_some_and(|maximum| maximum <= u32::MAX as usize)
        {
            let stride = self
                .embedding_stride
                .checked_mul(embedding_stride)
                .ok_or_else(|| {
                    AkitaError::InvalidInput("subring embedding stride overflow".into())
                })?;
            return Ok(Self {
                storage: self.storage.clone(),
                num_live_blocks_per_claim: self.num_live_blocks_per_claim,
                num_claims: self.num_claims,
                sampled_family: self
                    .sampled_family
                    .as_ref()
                    .map(|(_, config)| (ambient_ring_dimension, *config)),
                embedding_stride: stride,
                materialized: Arc::new(OnceLock::new()),
            });
        }
        let source = self.as_slice();
        let embed = |challenge: &SparseChallenge| {
            if !sampled {
                challenge.validate_dyn(challenge_subring_dimension)?;
            }
            let mut positions = crate::SparseChallengePositions::from_slice(&challenge.positions);
            for position in positions.iter_mut() {
                let embedded_position = usize::try_from(*position)
                    .ok()
                    .and_then(|position| position.checked_mul(embedding_stride))
                    .filter(|&position| position < ambient_ring_dimension)
                    .and_then(|position| u32::try_from(position).ok())
                    .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "subring challenge embedded position overflow".into(),
                        )
                    })?;
                *position = embedded_position;
            }
            Ok::<_, AkitaError>(SparseChallenge {
                positions,
                coeffs: crate::SparseChallengeCoefficients::from_slice(&challenge.coeffs),
            })
        };
        #[cfg(feature = "parallel")]
        let embedded = {
            let work = source
                .len()
                .checked_mul(
                    source
                        .first()
                        .map_or(0, |challenge| challenge.positions.len()),
                )
                .ok_or_else(|| {
                    AkitaError::InvalidInput("subring challenge embedding work overflow".into())
                })?;
            const PARALLEL_THRESHOLD: usize = 1 << 14;
            if work >= PARALLEL_THRESHOLD {
                source
                    .par_iter()
                    .map(embed)
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                source.iter().map(embed).collect::<Result<Vec<_>, _>>()?
            }
        };
        #[cfg(not(feature = "parallel"))]
        let embedded = source.iter().map(embed).collect::<Result<Vec<_>, _>>()?;
        let mut result =
            Self::from_sparse(embedded, self.num_live_blocks_per_claim, self.num_claims)?;
        if sampled {
            result.sampled_family = self
                .sampled_family
                .as_ref()
                .map(|(_, c)| (ambient_ring_dimension, *c));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime128Offset275;

    type F = Prime128Offset275;

    #[test]
    fn batch_evaluation_preserves_challenge_order() {
        let challenges = (0..4096)
            .map(|index| SparseChallenge {
                positions: (0..32).collect::<Vec<_>>().into(),
                coeffs: (0..32)
                    .map(|position| if (index + position) % 2 == 0 { 1 } else { -1 })
                    .collect::<Vec<_>>()
                    .into(),
            })
            .collect::<Vec<_>>();
        let batch = Challenges::from_sparse(challenges, 4096, 1).unwrap();
        let powers = (0..32)
            .map(|index| F::from_u64(index + 1))
            .collect::<Vec<_>>();
        let actual = batch.evals_at_pows::<F, F>(&powers).unwrap();
        let expected = batch
            .as_slice()
            .iter()
            .map(|challenge| challenge.eval_at_pows::<F, F>(&powers).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn subring_embedding_preserves_coefficients_and_batch_order() {
        let source = Challenges::from_sparse(
            vec![
                SparseChallenge {
                    positions: vec![0, 3, 7].into(),
                    coeffs: vec![1, -2, 1].into(),
                },
                SparseChallenge {
                    positions: vec![1, 6].into(),
                    coeffs: vec![-1, 2].into(),
                },
            ],
            1,
            2,
        )
        .unwrap();
        let embedded = source.embed_subring_positions(8, 4, 32).unwrap();
        assert_eq!(embedded.num_claims(), 2);
        assert_eq!(embedded.num_live_blocks_per_claim(), 1);
        assert_eq!(embedded.as_slice()[0].positions.as_slice(), &[0, 12, 28]);
        assert_eq!(embedded.as_slice()[0].coeffs, source.as_slice()[0].coeffs);
        assert_eq!(embedded.as_slice()[1].positions.as_slice(), &[4, 24]);
        assert_eq!(embedded.as_slice()[1].coeffs, source.as_slice()[1].coeffs);
        assert!(source.embed_subring_positions(8, 0, 32).is_err());
        assert!(source.embed_subring_positions(8, 2, 32).is_err());
    }
}

/// Challenges decoded from a deterministic message through the fold-draw
/// decoder, standing in for a squeezed transcript message.
#[cfg(test)]
mod sampled_values {
    use crate::{SparseChallenge, SparseChallengeConfig};

    pub(super) fn decode(
        seed: u64,
        ring_d: usize,
        count: usize,
        config: &SparseChallengeConfig,
    ) -> Vec<SparseChallenge> {
        let stream = crate::sampler::SparseChallengeShell::new(ring_d, config, None)
            .unwrap()
            .stream(count)
            .unwrap();
        let mut state = seed;
        let message: Vec<u8> = (0..stream.message_len())
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        stream.decode(&message).unwrap().unwrap()
    }
}

#[cfg(test)]
mod sampled_family_tests {
    use super::*;
    #[test]
    fn only_sampled_immutable_storage_carries_provenance() {
        let config = SparseChallengeConfig {
            count_pm1: 19,
            count_pm2: 0,
        };
        let values = super::sampled_values::decode(83, 512, 2, &config);
        let plain = Challenges::from_sparse(values.clone(), 1, 2).unwrap();
        let sampled = Challenges::from_sampled(values, 1, 2, 512, config).unwrap();
        assert_eq!(plain, sampled);
        assert!(!plain.is_sampled_family(512, &config));
        assert!(sampled.is_sampled_family(512, &config));
        assert!(!sampled.is_sampled_family(256, &config));
        for stride in [1, 4] {
            assert_eq!(
                plain
                    .embed_subring_positions(512, stride, 512 * stride)
                    .unwrap(),
                sampled
                    .embed_subring_positions(512, stride, 512 * stride)
                    .unwrap()
            );
        }
        assert!(core::ptr::eq(
            sampled.as_slice(),
            sampled
                .embed_subring_positions(512, 1, 512)
                .unwrap()
                .as_slice()
        ));
        let invalid = Challenges::from_sparse(
            vec![SparseChallenge {
                positions: vec![0, 0].into(),
                coeffs: vec![1, 1].into(),
            }],
            1,
            1,
        )
        .unwrap();
        assert!(invalid.embed_subring_positions(512, 1, 512).is_err());
        assert!(invalid.embed_subring_positions(512, 4, 2048).is_err());
    }
}

#[cfg(test)]
mod signed_evaluation_tests {
    use super::*;
    use jolt_field::Prime128OffsetA7F7 as F;
    #[test]
    fn sampled_signed_batches_match_the_sparse_polynomial() {
        let config = SparseChallengeConfig {
            count_pm1: 31,
            count_pm2: 10,
        };
        let values = super::sampled_values::decode(42, 64, 16, &config);
        let sampled = Challenges::from_sampled(values, 1, 16, 64, config).unwrap();
        for stride in [1, 4] {
            let batch = sampled
                .embed_subring_positions(64, stride, 64 * stride)
                .unwrap();
            let mut power = F::from_u64(1);
            let powers: Vec<F> = (0..64 * stride)
                .map(|_| {
                    let v = power;
                    power *= F::from_u64(37);
                    v
                })
                .collect();
            let expected: Vec<F> = batch
                .as_slice()
                .iter()
                .map(|c| c.eval_at_pows::<F, F>(&powers).unwrap())
                .collect();
            assert_eq!(batch.evals_at_pows::<F, F>(&powers).unwrap(), expected);
        }
    }
}

#[cfg(test)]
mod embedding_view_tests {
    use super::*;
    use jolt_field::Prime128OffsetA7F7 as F;
    #[test]
    fn checked_embeddings_are_lazy_composable_and_preserve_public_values() {
        let config = SparseChallengeConfig {
            count_pm1: 31,
            count_pm2: 10,
        };
        let values = super::sampled_values::decode(18, 64, 1024, &config);
        let source = Challenges::from_sampled(values, 1, 1024, 64, config).unwrap();
        let once = source.embed_subring_positions(64, 4, 256).unwrap();
        assert!(once.materialized.get().is_none());
        let (ChallengeStorage::Sparse(a), ChallengeStorage::Sparse(b)) =
            (&source.storage, &once.storage)
        else {
            panic!("sampled sparse storage expected");
        };
        assert!(Arc::ptr_eq(a, b));
        let powers: Vec<F> = (0..256).map(|i| F::from_u64(i as u64 * 7 + 11)).collect();
        let result = once.evals_at_pows::<F, F>(&powers).unwrap();
        assert!(once.materialized.get().is_none());
        for (i, c) in source.as_slice().iter().enumerate() {
            let expected = c.positions.iter().zip(&c.coeffs).fold(
                F::from_u64(0),
                |sum, (&p, &coefficient)| {
                    sum + powers[p as usize * 4] * F::from_i64(coefficient as i64)
                },
            );
            assert_eq!(result[i], expected);
        }
        let twice = once.embed_subring_positions(256, 2, 512).unwrap();
        let direct = source.embed_subring_positions(64, 8, 512).unwrap();
        assert_eq!(twice, direct);
        let subset = once.select_claims(&[7, 1]).unwrap();
        assert_eq!(subset.as_slice()[0], once.as_slice()[7]);
        let eager = Challenges::from_sparse(once.as_slice().to_vec(), 1, 1024).unwrap();
        assert_eq!(once, eager);
    }
}

#[cfg(test)]
mod sampled_evaluation_tests;

#[cfg(test)]
mod quartet_storage_tests;
