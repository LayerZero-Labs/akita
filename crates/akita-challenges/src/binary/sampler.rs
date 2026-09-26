use crate::FoldDraw;
use akita_error::{checked, AkitaError};
use num_bigint::BigUint;

use crate::sampler::{
    sample_distinct_positions_into, DistinctPositionScratch, IndexedXofPrefix, XofCursor,
};

use super::{BinaryChallenge, BinaryChallengeFamily, BinaryChallengeProfile, INLINE_BINARY_WEIGHT};

const TRANSCRIPT_DOMAIN: &[u8] = b"akita/labinius/binary-challenge/v2";

/// Reusable exact sampler for one binary challenge profile.
pub struct BinaryChallengeSampler {
    profile: BinaryChallengeProfile,
    bounded_rank_bits: usize,
    scratch: BinarySamplingScratch,
}

impl BinaryChallengeSampler {
    /// Construct a sampler with precomputed exact counts for its scalar ring.
    #[inline]
    #[must_use]
    pub fn new(profile: BinaryChallengeProfile) -> Self {
        let bounded_rank_bits = match profile.family() {
            BinaryChallengeFamily::FixedWeight => 0,
            BinaryChallengeFamily::BoundedWeight
                if profile.cardinality() == &BigUint::from(1u8) =>
            {
                0
            }
            BinaryChallengeFamily::BoundedWeight => {
                (profile.cardinality() - BigUint::from(1u8)).bits() as usize
            }
        };
        Self {
            profile,
            bounded_rank_bits,
            scratch: BinarySamplingScratch::new(),
        }
    }

    /// Return the sampled family identity and certified bounds.
    #[inline]
    #[must_use]
    pub const fn profile(&self) -> &BinaryChallengeProfile {
        &self.profile
    }

    /// Sample `count` challenges from one domain-separated transcript draw.
    ///
    /// Every challenge uses an independently indexed SHAKE256 substream. The
    /// bounded-weight sampler rejects out-of-range integers without a retry cap,
    /// so it has no sampler-failure probability to add to the fold budget.
    pub fn sample_challenges<D: FoldDraw>(
        &mut self,
        draw: &mut D,
        label: &[u8],
        count: usize,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        let count_u64 = u64::try_from(count)
            .map_err(|_| AkitaError::InvalidInput("binary challenge count exceeds u64".into()))?;
        let label_len = u64::try_from(label.len())
            .map_err(|_| AkitaError::InvalidInput("binary challenge label exceeds u64".into()))?;
        let context_len = checked::sum([
            TRANSCRIPT_DOMAIN.len(),
            8,
            label.len(),
            8,
            self.profile.identity_bytes().len(),
        ])
        .ok_or_else(|| {
            AkitaError::InvalidInput("binary challenge transcript context is too large".into())
        })?;
        let mut context = Vec::new();
        context.try_reserve_exact(context_len).map_err(|_| {
            AkitaError::InvalidInput("binary challenge transcript context is too large".into())
        })?;
        context.extend_from_slice(TRANSCRIPT_DOMAIN);
        context.extend_from_slice(&label_len.to_le_bytes());
        context.extend_from_slice(label);
        context.extend_from_slice(&count_u64.to_le_bytes());
        context.extend_from_slice(self.profile.identity_bytes());
        let seed = draw.absorb_and_squeeze(&context)?;
        self.sample_from_seed(&seed, count)
    }

    fn sample_from_seed(
        &mut self,
        seed: &[u8; 32],
        count: usize,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        let prefix = IndexedXofPrefix::new(seed)
            .map_err(|message| AkitaError::InvalidInput(message.into()))?;
        let mut challenges = Vec::new();
        challenges.try_reserve_exact(count).map_err(|_| {
            AkitaError::InvalidInput("binary challenge batch allocation failed".into())
        })?;
        for index in 0..count {
            let coordinate = u64::try_from(index).map_err(|_| {
                AkitaError::InvalidInput("binary challenge coordinate exceeds u64".into())
            })?;
            self.scratch
                .cursor
                .reset_indexed_prefix(&prefix, coordinate);
            self.scratch
                .sample_support(&self.profile, self.bounded_rank_bits)?;
            challenges.push(BinaryChallenge::from_support(
                &self.profile,
                &self.scratch.support,
                &mut self.scratch.sign_input,
                &mut self.scratch.sign_bytes,
            ));
        }
        Ok(challenges)
    }
}

struct BinarySamplingScratch {
    cursor: XofCursor,
    distinct_positions: DistinctPositionScratch,
    fixed_positions: Vec<u32>,
    support: Vec<u16>,
    rank_bytes: Vec<u8>,
    rank_digits: Vec<u32>,
    rank: BigUint,
    sign_input: Vec<u8>,
    sign_bytes: Vec<u8>,
}

impl BinarySamplingScratch {
    fn new() -> Self {
        Self {
            cursor: XofCursor::new(),
            distinct_positions: DistinctPositionScratch::new(),
            fixed_positions: Vec::new(),
            support: Vec::with_capacity(INLINE_BINARY_WEIGHT),
            rank_bytes: Vec::new(),
            rank_digits: Vec::new(),
            rank: BigUint::from(0u8),
            sign_input: Vec::new(),
            sign_bytes: Vec::new(),
        }
    }

    fn sample_support(
        &mut self,
        profile: &BinaryChallengeProfile,
        bounded_rank_bits: usize,
    ) -> Result<(), AkitaError> {
        match profile.family() {
            BinaryChallengeFamily::FixedWeight => {
                self.sample_weight(profile.scalar_ring().degree(), profile.weight_cap())
            }
            BinaryChallengeFamily::BoundedWeight => {
                uniform_biguint_below(
                    &mut self.cursor,
                    profile.cardinality(),
                    bounded_rank_bits,
                    &mut self.rank_bytes,
                    &mut self.rank_digits,
                    &mut self.rank,
                );
                let weight = profile
                    .scalar_ring()
                    .counts()
                    .weight_for_ball_rank(&self.rank, profile.weight_cap());
                self.sample_weight(profile.scalar_ring().degree(), weight)
            }
        }
    }

    fn sample_weight(&mut self, degree: usize, weight: usize) -> Result<(), AkitaError> {
        self.fixed_positions.resize(weight, 0);
        sample_distinct_positions_into(
            &mut self.cursor,
            degree,
            &mut self.fixed_positions,
            &mut self.distinct_positions,
        )?;
        self.fixed_positions.sort_unstable();
        self.support.clear();
        self.support.try_reserve(weight).map_err(|_| {
            AkitaError::InvalidInput("binary challenge support allocation failed".into())
        })?;
        for &position in &self.fixed_positions {
            self.support.push(u16::try_from(position).map_err(|_| {
                AkitaError::InvalidInput("binary challenge position exceeds u16".into())
            })?);
        }
        Ok(())
    }
}

fn uniform_biguint_below(
    cursor: &mut XofCursor,
    upper: &BigUint,
    bits: usize,
    bytes: &mut Vec<u8>,
    digits: &mut Vec<u32>,
    rank: &mut BigUint,
) {
    debug_assert!(*upper > BigUint::from(0u8));
    if bits == 0 {
        rank.assign_from_slice(&[]);
        return;
    }
    let byte_len = bits.div_ceil(8);
    bytes.resize(byte_len, 0);
    let high_bits = bits % 8;
    loop {
        cursor.fill_bytes(bytes);
        if high_bits != 0 {
            bytes[byte_len - 1] &= (1u8 << high_bits) - 1;
        }
        digits.clear();
        digits.extend(bytes.chunks(4).map(|chunk| {
            let mut digit = [0u8; 4];
            digit[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(digit)
        }));
        rank.assign_from_slice(digits);
        if *rank < *upper {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shake::digest::{ExtendableOutput, Update, XofReader};
    use shake::Shake256;

    struct TestDraw;

    impl FoldDraw for TestDraw {
        fn absorb_and_squeeze(&mut self, payload: &[u8]) -> Result<[u8; 32], AkitaError> {
            let mut xof = Shake256::default();
            xof.update(payload);
            let mut seed = [0u8; 32];
            xof.finalize_xof().read(&mut seed);
            Ok(seed)
        }
    }

    #[test]
    fn bounded_integer_rejection_matches_an_independent_xof_mapping() {
        let seed = [0x3cu8; 32];
        let prefix = IndexedXofPrefix::new(&seed).unwrap();
        let upper = BigUint::from(5u8);
        let mut rank_bytes = Vec::new();
        let mut rank_digits = Vec::new();
        let mut actual = BigUint::from(0u8);
        let mut saw_rejection = false;
        for coordinate in 0..64u64 {
            let mut production = XofCursor::from_indexed_prefix(&prefix, coordinate);
            uniform_biguint_below(
                &mut production,
                &upper,
                3,
                &mut rank_bytes,
                &mut rank_digits,
                &mut actual,
            );

            let mut independent = Shake256::default();
            independent.update(&seed);
            independent.update(&coordinate.to_le_bytes());
            let mut reader = independent.finalize_xof();
            let mut attempts = 0;
            let expected = loop {
                let mut byte = [0u8; 1];
                reader.read(&mut byte);
                attempts += 1;
                let candidate = byte[0] & 7;
                if candidate < 5 {
                    break candidate;
                }
            };
            saw_rejection |= attempts > 1;
            assert_eq!(actual, BigUint::from(expected));

            let mut actual_suffix = [0u8; 16];
            let mut expected_suffix = [0u8; 16];
            production.fill_bytes(&mut actual_suffix);
            reader.read(&mut expected_suffix);
            assert_eq!(actual_suffix, expected_suffix);
        }
        assert!(saw_rejection);
    }

    #[test]
    fn transcript_replay_is_deterministic_and_profile_separated() {
        let bounded_profile = BinaryChallengeProfile::bounded_weight(
            super::super::BinaryScalarRing::Cyclotomic243,
            46,
        )
        .unwrap();
        let fixed_profile =
            BinaryChallengeProfile::fixed_weight(super::super::BinaryScalarRing::Cyclotomic243, 46)
                .unwrap();
        let sample = |profile: BinaryChallengeProfile| {
            let mut draw = TestDraw;
            let mut sampler = BinaryChallengeSampler::new(profile);
            sampler
                .sample_challenges(&mut draw, b"binary-fold", 4)
                .unwrap()
        };
        let first = sample(bounded_profile.clone());
        assert_eq!(first, sample(bounded_profile));
        assert_ne!(first, sample(fixed_profile));
    }

    #[test]
    fn sampled_challenges_replay_the_unique_profile_sign_pattern() {
        for profile in [
            BinaryChallengeProfile::fixed_weight(super::super::BinaryScalarRing::Cyclotomic243, 47)
                .unwrap(),
            BinaryChallengeProfile::bounded_weight(
                super::super::BinaryScalarRing::Cyclotomic243,
                46,
            )
            .unwrap(),
            BinaryChallengeProfile::fixed_weight(super::super::BinaryScalarRing::Cyclotomic729, 25)
                .unwrap(),
        ] {
            let mut sampler = BinaryChallengeSampler::new(profile.clone());
            let challenges = sampler.sample_from_seed(&[0x5au8; 32], 8).unwrap();
            for challenge in challenges {
                challenge.validate(&profile).unwrap();
                assert_eq!(
                    challenge.coefficient_l1_norm(),
                    challenge.coefficient_l2_squared_norm()
                );
            }
        }
    }

    #[test]
    fn zero_support_is_sampled_when_it_is_the_counted_family() {
        let profile = BinaryChallengeProfile::bounded_weight(
            super::super::BinaryScalarRing::Cyclotomic243,
            0,
        )
        .unwrap();
        let mut sampler = BinaryChallengeSampler::new(profile.clone());
        let challenge = sampler
            .sample_from_seed(&[7u8; 32], 1)
            .unwrap()
            .pop()
            .unwrap();
        assert!(challenge.terms().is_empty());
        assert_eq!(challenge.coefficient_linf_norm(), 0);
        assert_eq!(
            challenge.canonical_support_encoding(&profile).unwrap(),
            vec![0; 21]
        );
    }
}
