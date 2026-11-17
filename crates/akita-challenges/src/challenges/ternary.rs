use super::*;
use crate::checked_hint::ternary::{CODES, GROUPS, ZERO_CODE};

pub(super) fn materialize(codes: &[[u64; 4]], stride: usize) -> Vec<SparseChallenge> {
    codes
        .iter()
        .map(|words| {
            let mut positions = crate::SparseChallengePositions::new();
            let mut coeffs = crate::SparseChallengeCoefficients::new();
            for group in 0..GROUPS {
                let mut code = (words[group / 6] >> (10 * (group % 6))) & 1023;
                for i in 0..6 {
                    let coefficient = (code % 3) as i8 - 1;
                    code /= 3;
                    let position = group * 6 + i;
                    if position < 128 && coefficient != 0 {
                        // Only sampler-authenticated storage reaches this path;
                        // subring embedding already checked the u32 extent.
                        positions.push((position * stride) as u32);
                        coeffs.push(coefficient);
                    }
                }
            }
            SparseChallenge { positions, coeffs }
        })
        .collect()
}

impl Challenges {
    pub(crate) fn from_checked_ternary(
        packed: Vec<[u64; 4]>,
        blocks: usize,
        claims: usize,
    ) -> Result<Self, AkitaError> {
        if blocks.checked_mul(claims) != Some(packed.len()) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(Self {
            storage: ChallengeStorage::Ternary(Arc::new(packed)),
            num_live_blocks_per_claim: blocks,
            num_claims: claims,
            sampled_family: Some((
                128,
                SparseChallengeConfig {
                    count_pm1: 31,
                    count_pm2: 0,
                },
            )),
            embedding_stride: 1,
            materialized: Arc::new(OnceLock::new()),
        })
    }

    pub(super) fn evaluate_ternary<F: Field + Ring, E: Field + ExtField<F>>(
        &self,
        codes: &[[u64; 4]],
        powers: &[E],
    ) -> Result<Vec<E>, AkitaError> {
        let mut tables = Vec::new();
        tables
            .try_reserve_exact(GROUPS * CODES)
            .map_err(|_| AkitaError::InvalidProof)?;
        tables.resize(GROUPS * CODES, E::zero());
        for (group, table) in tables.chunks_exact_mut(CODES).enumerate() {
            let mut p = [E::zero(); 6];
            for (i, value) in p.iter_mut().enumerate() {
                let position = group * 6 + i;
                if position < 128 {
                    *value = position
                        .checked_mul(self.embedding_stride)
                        .and_then(|at| powers.get(at))
                        .copied()
                        .ok_or(AkitaError::InvalidProof)?;
                }
            }
            table[0] = -p.iter().copied().fold(E::zero(), |a, b| a + b);
            let mut filled = 1;
            for power in p {
                for i in 0..filled {
                    let one = table[i] + power;
                    table[filled + i] = one;
                    table[2 * filled + i] = one + power;
                }
                filled *= 3;
            }
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(codes.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        for words in codes {
            let mut sum = E::zero();
            macro_rules! add {
                ($group:literal) => {{
                    let code = ((words[$group / 6] >> (10 * ($group % 6))) & 1023) as usize;
                    if code != ZERO_CODE {
                        sum += *tables[$group * CODES..($group + 1) * CODES]
                            .get(code)
                            .ok_or(AkitaError::InvalidProof)?;
                    }
                }};
            }
            add!(0);
            add!(1);
            add!(2);
            add!(3);
            add!(4);
            add!(5);
            add!(6);
            add!(7);
            add!(8);
            add!(9);
            add!(10);
            add!(11);
            add!(12);
            add!(13);
            add!(14);
            add!(15);
            add!(16);
            add!(17);
            add!(18);
            add!(19);
            add!(20);
            add!(21);
            output.push(sum);
        }
        Ok(output)
    }
}
