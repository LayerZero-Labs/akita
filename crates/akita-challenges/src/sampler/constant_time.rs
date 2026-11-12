//! Prover-side constant-time sampling of one selective-L2 challenge.
//!
//! A successful Fiat-Shamir challenge is public. During a prover's rejection
//! loop, however, a challenge from an unpublished attempt can become correlated
//! with the witness when the response is rejected. Its positions and signs must
//! not leak through branches or memory addresses. The indexed sampler is for
//! public challenges; this implementation uses masked full scans for the shuffle
//! and dense scatter, arithmetic signs, and every operator-norm frequency.
//!
//! This is constant time with respect to the accepted challenge, conditional on
//! the rejection trace. It is NOT constant time with respect to an arbitrary
//! secret seed: uniform-index rejection counts and norm-filter acceptance bits
//! are exposed by control flow and XOF buffer offsets. In the random-oracle/XOF
//! model with a fresh uniform root, those rejection counts are independent of
//! the accepted positions and signs. Callers must not encode secrets in the root
//! or otherwise assume that its full byte stream is hidden. No witness is an
//! input to this API. The guarantee concerns source-level control flow and
//! addresses; compiler and target instruction timing still need platform audit.
//!
//! [`sample_indexed_challenges_from_seed`]: super::sample_indexed_challenges_from_seed

use super::xof::{IndexedXofPrefix, XofCursor};
use super::{D64_SELECTIVE_L2_OP_NORM_TABLE, MAX_OP_NORM_ATTEMPTS};
use crate::{
    OperatorNormRejection, SparseChallenge, D64_L2_OP_NORM_PM1_COUNT, D64_L2_OP_NORM_PM2_COUNT,
    D64_SELECTIVE_L2_CHALLENGE_CONFIG, INLINE_SPARSE_WEIGHT,
};
use akita_error::AkitaError;

const CHALLENGE_RING_DIM: usize = 64;

/// Nonzero coefficients of every attempt challenge.
const WEIGHT: usize = D64_L2_OP_NORM_PM1_COUNT + D64_L2_OP_NORM_PM2_COUNT;
// The returned challenge's storage must not spill to the heap.
const _: () = assert!(WEIGHT <= INLINE_SPARSE_WEIGHT);

/// The D = 64 selective-L2 challenge that
/// `sample_indexed_challenges_from_seed(seed, 64, 1, &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
/// Some(OperatorNormRejection::D64_SELECTIVE_L2))` returns, sampled in
/// constant time with respect to the challenge.
///
/// The positions and signs of an unpublished attempt may be sensitive after
/// witness-dependent response rejection. Shuffle and scatter use full masked
/// scans, signs use arithmetic, and the norm predicate scans every frequency.
/// Successful protocol challenges are public; this API takes no witness.
///
/// The guarantee is conditional on the rejection trace: control flow exposes
/// uniform-index rejection counts and norm-filter acceptance bits. Under the
/// XOF model with a fresh uniform root those are independent of the accepted
/// positions and signs. This is not constant time in an arbitrary secret seed;
/// callers must not encode secrets in the root or require the entire XOF stream
/// to stay hidden. Compiler and instruction timing require platform auditing.
///
/// # Errors
///
/// Rejects roots whose length is not 32 bytes, unavailable certified tables,
/// and exhaustion of the 4096-candidate norm-filter budget.
pub fn sample_challenge_ct(seed: &[u8]) -> Result<SparseChallenge, AkitaError> {
    sample_challenge_ct_counted(seed).map(|(challenge, _)| challenge)
}

/// [`sample_challenge_ct`] and the number of candidates it drew. The count
/// is independent of the challenge returned under the XOF model with a fresh
/// uniform root. This exposes the same rejection trace and has the same errors
/// and timing contract as [`sample_challenge_ct`].
pub fn sample_challenge_ct_counted(seed: &[u8]) -> Result<(SparseChallenge, usize), AkitaError> {
    let config = D64_SELECTIVE_L2_CHALLENGE_CONFIG;
    let threshold = OperatorNormRejection::D64_SELECTIVE_L2.threshold;
    let table = D64_SELECTIVE_L2_OP_NORM_TABLE
        .as_ref()
        .map_err(|message| AkitaError::InvalidSetup((*message).into()))?;
    let prefix =
        IndexedXofPrefix::new(seed).map_err(|message| AkitaError::InvalidSetup(message.into()))?;
    let mut cursor = XofCursor::new();
    cursor.reset_indexed_prefix(&prefix, 0);
    // Fixed-size stack buffers, and the returned challenge fits its inline
    // storage: no allocation depends on the draw.
    let mut positions = [0u32; WEIGHT];
    let mut coeffs = [0i8; WEIGHT];
    let mut sign_bytes = [0u8; WEIGHT];
    for attempt in 1..=MAX_OP_NORM_ATTEMPTS {
        draw_positions(&mut cursor, &mut positions);
        cursor.fill_bytes(&mut sign_bytes);
        for (index, (coeff, &byte)) in coeffs.iter_mut().zip(&sign_bytes).enumerate() {
            let magnitude = if index < config.count_pm1 { 1 } else { 2 };
            // `(1 − 2·(byte & 1)) · magnitude`: the sign bit selects by arithmetic.
            *coeff = (1 - 2 * (byte & 1) as i8) * magnitude;
        }
        let dense = scatter_ct(&positions, &coeffs);
        if table.accept_strict_dense_ct(&dense, config.l1_norm() as u64, u64::from(threshold))? {
            let challenge = SparseChallenge {
                positions: positions.iter().copied().collect(),
                coeffs: coeffs.iter().copied().collect(),
            };
            return Ok((challenge, attempt));
        }
    }
    Err(AkitaError::InvalidInput(format!(
        "operator-norm rejection exceeded {MAX_OP_NORM_ATTEMPTS} attempts"
    )))
}

/// The verifier's partial Fisher–Yates shuffle over `0..64`, reading and
/// writing the permutation through masked scans of all 64 entries.
fn draw_positions(cursor: &mut XofCursor, out: &mut [u32]) {
    let mut permutation: [u8; CHALLENGE_RING_DIM] = std::array::from_fn(|index| index as u8);
    for (i, dst) in out.iter_mut().enumerate() {
        let j = i + cursor.next_usize_mod(CHALLENGE_RING_DIM - i);
        let at_j = select_ct(&permutation, j);
        let at_i = select_ct(&permutation, i);
        for (index, slot) in permutation.iter_mut().enumerate() {
            let is_i = eq_mask(index, i);
            let is_j = eq_mask(index, j);
            // Position i takes the value at j; position j (when j ≠ i) takes
            // the value at i. When j = i both writes leave the slot unchanged.
            *slot = (*slot & !(is_i | is_j)) | (at_j & is_i) | (at_i & is_j & !is_i);
        }
        *dst = u32::from(at_j);
    }
}

/// `values[index]`, read through a masked scan of every entry.
fn select_ct(values: &[u8; CHALLENGE_RING_DIM], index: usize) -> u8 {
    values
        .iter()
        .enumerate()
        .fold(0, |acc, (position, &value)| {
            acc | (value & eq_mask(position, index))
        })
}

/// `0xFF` if `a == b`, else `0`, without a branch.
fn eq_mask(a: usize, b: usize) -> u8 {
    let difference = (a ^ b) as u64;
    // The top bit of `d | −d` is set exactly when `d ≠ 0`.
    (((difference | difference.wrapping_neg()) >> 63) as u8).wrapping_sub(1)
}

/// The dense coefficients of `(positions, coeffs)`, built by a masked scan:
/// every output coefficient reads every support entry, and entries at a
/// repeated position add.
fn scatter_ct(positions: &[u32], coeffs: &[i8]) -> [i8; CHALLENGE_RING_DIM] {
    std::array::from_fn(|index| {
        positions
            .iter()
            .zip(coeffs)
            .fold(0i8, |acc, (&position, &coeff)| {
                acc.wrapping_add(coeff & eq_mask(position as usize, index) as i8)
            })
    })
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test code")]
mod tests {
    use super::*;
    use crate::sample_indexed_challenges_from_seed;

    /// Byte identity with the verifier's sampler, over seeds that include
    /// norm-filter rejections.
    #[test]
    fn matches_the_verifier_sampler() {
        let mut rejected = 0;
        for index in 0u64..100_000 {
            let mut seed = [0u8; 32];
            seed[..8].copy_from_slice(&index.to_le_bytes());
            seed[8..16].copy_from_slice(&index.wrapping_mul(0x9e37_79b9_7f4a_7c15).to_le_bytes());
            let expected = sample_indexed_challenges_from_seed(
                &seed,
                CHALLENGE_RING_DIM,
                1,
                &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                Some(OperatorNormRejection::D64_SELECTIVE_L2),
            )
            .unwrap();
            let (actual, attempts) = sample_challenge_ct_counted(&seed).unwrap();
            assert_eq!([actual], expected.as_slice(), "seed {index}");
            rejected += usize::from(attempts > 1);
        }
        assert!(
            rejected > 10_000,
            "only {rejected} seeds hit a norm rejection"
        );
    }
    #[test]
    fn fixed_shape_and_masked_scatter() {
        assert_eq!(eq_mask(0, 0), 255);
        assert_eq!(eq_mask(usize::MAX, usize::MAX), 255);
        assert_eq!(eq_mask(0, usize::MAX), 0);
        let dense = scatter_ct(&[3, 0, 63], &[-2, 1, -1]);
        assert_eq!(dense[0], 1);
        assert_eq!(dense[3], -2);
        assert_eq!(dense[63], -1);
        assert_eq!(dense.iter().filter(|&&c| c != 0).count(), 3);
        let challenge = sample_challenge_ct(&[0; 32]).unwrap();
        challenge.validate::<64>().unwrap();
        assert!(!challenge.positions.spilled());
        assert!(!challenge.coeffs.spilled());
    }

    #[test]
    fn rejects_noncanonical_root_lengths() {
        for seed in [&[][..], &[0; 31][..], &[0; 33][..]] {
            assert!(matches!(
                sample_challenge_ct(seed),
                Err(AkitaError::InvalidSetup(_))
            ));
        }
    }
}
