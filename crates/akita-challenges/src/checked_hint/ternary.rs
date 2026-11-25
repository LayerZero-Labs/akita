//! Checked six-coefficient advice for the production D128/31/0 shell.
use akita_error::AkitaError;

pub(crate) const GROUPS: usize = 22;
pub(crate) const CODES: usize = 729;
pub(crate) const ZERO_CODE: usize = 364;

const fn binomials() -> [[u128; 32]; 128] {
    let mut out = [[0; 32]; 128];
    let mut n = 0;
    while n < 128 {
        out[n][0] = 1;
        let mut k = 1;
        while k <= n && k < 32 {
            // C(127,31) is below 2^99; none of these additions overflow.
            out[n][k] = out[n - 1][k - 1] + out[n - 1][k];
            k += 1;
        }
        n += 1;
    }
    out
}
static BINOMIALS: [[u128; 32]; 128] = binomials();

const fn metadata() -> [u16; CODES] {
    let mut out = [0; CODES];
    let mut code = 0;
    while code < CODES {
        let (mut rest, mut count, mut mask, mut signs) = (code, 0, 0, 0);
        let mut i = 0;
        while i < 6 {
            let digit = rest % 3;
            rest /= 3;
            if digit != 1 {
                mask |= 1 << i;
                if digit == 0 {
                    signs |= 1 << count;
                }
                count += 1;
            }
            i += 1;
        }
        out[code] = count | (mask << 3) | (signs << 9);
        code += 1;
    }
    out
}
const META: [u16; CODES] = metadata();

const fn rank_block(group: usize) -> [[u128; 64]; 32] {
    let mut out = [[0; 64]; 32];
    let mut prefix = 0;
    while prefix < 32 {
        let mut mask = 1usize;
        while mask < 64 {
            let high = usize::BITS as usize - 1 - mask.leading_zeros() as usize;
            let previous = mask ^ (1 << high);
            let rank_index = prefix + previous.count_ones() as usize + 1;
            let position = 6 * group + high;
            if rank_index < 32 && position < 128 {
                out[prefix][mask] = out[prefix][previous] + BINOMIALS[position][rank_index];
            }
            mask += 1;
        }
        prefix += 1;
    }
    out
}

// Separate constants keep each compile-time evaluation small. These are exact
// combinatorial contributions, independent of the proof, setup and field.
macro_rules! blocks {
    ($( $name:ident : $index:literal ),+ $(,)?) => {
        $(static $name: [[u128; 64]; 32] = rank_block($index);)+
        static RANKS: [&[[u128; 64]; 32]; GROUPS] = [$(&$name),+];
    };
}
blocks!(
    B0:0, B1:1, B2:2, B3:3, B4:4, B5:5, B6:6, B7:7,
    B8:8, B9:9, B10:10, B11:11, B12:12, B13:13, B14:14,
    B15:15, B16:16, B17:17, B18:18, B19:19, B20:20, B21:21,
);

/// Pack a D128/31/0 challenge into the four ternary-code words [`validate`] checks.
pub(crate) fn encode(challenge: &crate::SparseChallenge) -> Result<[u64; 4], AkitaError> {
    // Group 21 covers positions 126..132; positions from 128 on stay zero.
    let mut digits = [1u64; 6 * GROUPS];
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        if position >= 128 {
            return Err(AkitaError::Internal("ternary position out of range".into()));
        }
        digits[position as usize] = (i64::from(coefficient) + 1) as u64;
    }
    let mut words = [0u64; 4];
    for (group, digits) in digits.chunks_exact(6).enumerate() {
        let code = digits.iter().rev().fold(0, |code, &digit| code * 3 + digit);
        words[group / 6] |= code << (10 * (group % 6));
    }
    Ok(words)
}

pub(crate) fn validate(words: &[u64; 4], expected: u128, signs: u64) -> Result<(), AkitaError> {
    if words[..3].iter().any(|word| word >> 60 != 0) || words[3] >> 40 != 0 {
        return Err(AkitaError::InvalidProof);
    }
    let (mut count, mut rank, mut actual_signs) = (0usize, 0u128, 0u64);
    for group in 0..GROUPS {
        let code = ((words[group / 6] >> (10 * (group % 6))) & 1023) as usize;
        let meta = *META.get(code).ok_or(AkitaError::InvalidProof)?;
        let add = (meta & 7) as usize;
        let mask = ((meta >> 3) & 63) as usize;
        if count + add > 31 || (group == GROUPS - 1 && mask & !3 != 0) {
            return Err(AkitaError::InvalidProof);
        }
        if add != 0 {
            // group<22, mask<64 and count<=31 were established above. The
            // supports are disjoint and ordered, so their colex sum fits u128.
            rank += RANKS[group][count][mask];
            actual_signs |= u64::from(meta >> 9) << count;
            count += add;
        }
    }
    if count != 31 || rank != expected || signs != actual_signs {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
