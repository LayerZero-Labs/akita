//! Dense evaluation of the certified predicate without support-dependent lookups.

use super::OpNormTable;
use akita_error::AkitaError;

impl OpNormTable {
    /// The strict predicate of [`Self::accept_strict_parts`] on a dense
    /// coefficient vector with `‖c‖₁ = l1`, evaluated in constant time.
    ///
    /// Every frequency is accumulated from every coefficient position, so no
    /// table row is chosen by a secret position, and the per-frequency
    /// comparisons are combined by masks with no early exit. The integer
    /// accumulators equal the sparse path's exactly. Acceptance requires every
    /// frequency's upper enclosure to be at most the threshold.
    /// The caller guarantees that the dense vector's L1 norm is at most `l1`.
    /// Shape and public bounds are checked before coefficient arithmetic.
    pub(crate) fn accept_strict_dense_ct(
        &self,
        dense: &[i8],
        l1: u64,
        t: u64,
    ) -> Result<bool, AkitaError> {
        if dense.len() != self.d {
            return Err(AkitaError::InvalidSize {
                expected: self.d,
                actual: dense.len(),
            });
        }
        let l1 = i128::from(l1);
        if l1 > self.max_l1 || i128::from(t) > self.max_t {
            return Err(AkitaError::InvalidInput(
                "operator-norm predicate: ||c||_1 or threshold outside the validated range"
                    .to_string(),
            ));
        }
        let half_d = self.d / 2;
        let r = l1 * self.eps_root;
        let threshold = (i128::from(t) * i128::from(t)) << (2 * self.q);
        let mut rejected = 0i128;
        for k in 0..half_d {
            let (mut re, mut im) = (0i64, 0i64);
            for (pos, &coeff) in dense.iter().enumerate() {
                let (cos_row, sin_row) = self.freq_row(pos, half_d);
                let coeff = i64::from(coeff);
                re += coeff * cos_row.get(k).copied().unwrap_or_default();
                im += coeff * sin_row.get(k).copied().unwrap_or_default();
            }
            let (re, im) = (i128::from(re), i128::from(im));
            let upper = re * re + im * im + 2 * r * (ct_abs(re) + ct_abs(im)) + 2 * r * r;
            rejected |= threshold - upper;
        }
        Ok(rejected >= 0)
    }
}

/// `|x|` without a branch.
#[inline]
fn ct_abs(x: i128) -> i128 {
    let sign = x >> 127;
    (x ^ sign) - sign
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_predicate_matches_sparse_on_both_sides_of_threshold() {
        let table = OpNormTable::new(64, 48, 128, 18).unwrap();
        for dense in [
            [0i8; 64],
            [1; 64],
            [-1; 64],
            std::array::from_fn(|i| (i % 5) as i8 - 2),
        ] {
            let positions: Vec<_> = (0..64).collect();
            let l1 = dense.iter().map(|c| u64::from(c.unsigned_abs())).sum();
            assert_eq!(
                table.accept_strict_dense_ct(&dense, l1, 18).unwrap(),
                table.accept_strict_parts(&positions, &dense, 18).unwrap(),
            );
        }
        assert!(table.accept_strict_dense_ct(&[0; 63], 0, 18).is_err());
        assert!(table.accept_strict_dense_ct(&[0; 64], 129, 18).is_err());
        assert!(table.accept_strict_dense_ct(&[0; 64], 0, 19).is_err());
    }
}
