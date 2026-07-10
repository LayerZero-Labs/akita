//! Utilities for the equality polynomial `eq(x, y) = Πᵢ (xᵢ yᵢ + (1 − xᵢ)(1 − yᵢ))`.
//!
//! The equality polynomial evaluates to 1 when `x = y` (over the boolean hypercube)
//! and 0 otherwise. Its multilinear extension (MLE) is used throughout sumcheck
//! protocols.
//!
//! Adapted from Jolt's `EqPolynomial` implementation.
//!
//! ## Bit / index order: Little-endian
//!
//! The evaluation tables produced by this module use **little-endian** bit order:
//! entry `b` (as an integer index) corresponds to the boolean vector where
//! bit `k` of `b` equals `x[k]`. In other words, `r[0]` corresponds to the
//! **least-significant bit** (bit 0) and `r[n-1]` to the MSB.

use crate::{AkitaError, FieldCore};
use std::marker::PhantomData;
use std::mem;

/// Maximum memory budget for one materialized equality-table allocation family.
///
/// This is deliberately separate from serialization's generic sequence cap:
/// equality tables may be larger than serialized proof vectors, but verifier-
/// reachable code still needs an explicit allocation ceiling.
pub const MAX_MATERIALIZED_EQ_TABLE_BYTES: usize = 1 << 30;

/// Utilities for the equality polynomial `eq(x, y) = Πᵢ (xᵢ yᵢ + (1 − xᵢ)(1 − yᵢ))`.
pub struct EqPolynomial<E: FieldCore>(PhantomData<E>);

impl<E: FieldCore> EqPolynomial<E> {
    fn table_len(num_vars: usize) -> Result<usize, AkitaError> {
        let shift = u32::try_from(num_vars).map_err(|_| AkitaError::InvalidSize {
            expected: usize::BITS as usize,
            actual: num_vars,
        })?;
        let len = 1usize
            .checked_shl(shift)
            .ok_or_else(|| AkitaError::InvalidInput("eq table dimension overflow".to_string()))?;
        Ok(len)
    }

    fn check_element_budget(label: &str, len: usize) -> Result<(), AkitaError> {
        let elem_size = mem::size_of::<E>().max(1);
        let bytes = len.checked_mul(elem_size).ok_or_else(|| {
            AkitaError::InvalidInput(format!("{label} byte-size overflow for {len} elements"))
        })?;
        if bytes > MAX_MATERIALIZED_EQ_TABLE_BYTES {
            return Err(AkitaError::InvalidInput(format!(
                "{label} requires {bytes} bytes, exceeding equality-table budget of {MAX_MATERIALIZED_EQ_TABLE_BYTES} bytes"
            )));
        }
        Ok(())
    }

    fn checked_table_len(label: &str, num_vars: usize) -> Result<usize, AkitaError> {
        let len = Self::table_len(num_vars)?;
        Self::check_element_budget(label, len)?;
        Ok(len)
    }

    fn zero_vec(label: &str, len: usize) -> Result<Vec<E>, AkitaError> {
        let mut out = Vec::new();
        out.try_reserve_exact(len).map_err(|_| {
            AkitaError::InvalidInput(format!("{label} allocation failed for {len} elements"))
        })?;
        out.resize(len, E::zero());
        Ok(out)
    }

    /// Compute the MLE of the equality polynomial at two points:
    /// `eq(x, y) = Πᵢ (xᵢ yᵢ + (1 − xᵢ)(1 − yᵢ))`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x.len() != y.len()`.
    pub fn mle(x: &[E], y: &[E]) -> Result<E, AkitaError> {
        if x.len() != y.len() {
            return Err(AkitaError::InvalidSize {
                expected: x.len(),
                actual: y.len(),
            });
        }
        Ok(x.iter()
            .zip(y.iter())
            .map(|(&x_i, &y_i)| x_i * y_i + (E::one() - x_i) * (E::one() - y_i))
            .fold(E::one(), |acc, v| acc * v))
    }

    /// Compute the zero selector: `eq(r, 0) = Πᵢ (1 − rᵢ)`.
    pub fn zero_selector(r: &[E]) -> E {
        r.iter().fold(E::one(), |acc, &r_i| acc * (E::one() - r_i))
    }

    /// Compute the full evaluation table `{ eq(r, x) : x ∈ {0,1}^n }`.
    ///
    /// Uses **little-endian** bit order: entry `b` has bit `k` of `b`
    /// corresponding to `r[k]`.
    ///
    /// For a scaled table, use [`Self::evals_with_scaling`].
    pub fn evals(r: &[E]) -> Result<Vec<E>, AkitaError> {
        Self::evals_with_scaling(r, None)
    }

    /// Compute the first `len` entries of the little-endian equality table.
    ///
    /// The split representation keeps this bounded by the requested prefix
    /// instead of allocating the full `2^|r|` table. This is useful when a
    /// protocol has a padded row domain but only materializes its live rows.
    pub fn evals_prefix(r: &[E], len: usize) -> Result<Vec<E>, AkitaError> {
        let split = SplitEqEvals::new(r)?;
        if len > split.len() {
            return Err(AkitaError::InvalidSize {
                expected: split.len(),
                actual: len,
            });
        }
        (0..len).map(|index| split.eval_at(index)).collect()
    }

    /// Compute the full evaluation table with optional scaling:
    /// `scaling_factor · eq(r, x)` for all `x ∈ {0,1}^n`.
    ///
    /// Uses the same **little-endian** index order as [`Self::evals`].
    /// If `scaling_factor` is `None`, defaults to 1 (no scaling).
    pub fn evals_with_scaling(r: &[E], scaling_factor: Option<E>) -> Result<Vec<E>, AkitaError> {
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 16;
            if r.len() > PARALLEL_THRESHOLD {
                return Self::evals_parallel(r, scaling_factor);
            }
        }
        Self::evals_serial(r, scaling_factor)
    }

    /// Serial (single-threaded) version of [`Self::evals_with_scaling`].
    ///
    /// Uses **little-endian** index order.
    pub fn evals_serial(r: &[E], scaling_factor: Option<E>) -> Result<Vec<E>, AkitaError> {
        let size = Self::checked_table_len("eq evaluation table", r.len())?;
        let mut evals = Self::zero_vec("eq evaluation table", size)?;
        evals[0] = scaling_factor.unwrap_or(E::one());
        let mut len = 1usize;
        for &t in r.iter().rev() {
            let one_minus_t = E::one() - t;
            for j in (0..len).rev() {
                evals[2 * j + 1] = evals[j] * t;
                evals[2 * j] = evals[j] * one_minus_t;
            }
            len *= 2;
        }
        Ok(evals)
    }

    /// Compute eq evaluations and cache intermediate tables.
    ///
    /// Returns `result` where `result[j]` contains evaluations for the prefix
    /// `r[..j]`: `result[j][x] = eq(r[..j], x)` for `x ∈ {0,1}^j`.
    ///
    /// So `result[0] = [1]`, `result[1]` has 2 entries, ..., and `result[n]`
    /// equals [`Self::evals`] called on `r`.
    pub fn evals_cached(r: &[E]) -> Result<Vec<Vec<E>>, AkitaError> {
        Self::evals_cached_with_scaling(r, None)
    }

    /// Like [`Self::evals_cached`], but with optional scaling.
    pub fn evals_cached_with_scaling(
        r: &[E],
        scaling_factor: Option<E>,
    ) -> Result<Vec<Vec<E>>, AkitaError> {
        let final_len = Self::table_len(r.len())?;
        let total_len = final_len
            .checked_mul(2)
            .and_then(|len| len.checked_sub(1))
            .ok_or_else(|| {
                AkitaError::InvalidInput("cached eq table total length overflow".to_string())
            })?;
        Self::check_element_budget("cached eq tables", total_len)?;
        let mut result = Vec::with_capacity(r.len() + 1);
        let mut layer_len = 1usize;
        for _ in 0..=r.len() {
            result.push(Self::zero_vec("cached eq table layer", layer_len)?);
            layer_len = layer_len.saturating_mul(2);
        }
        result[0][0] = scaling_factor.unwrap_or(E::one());
        for j in 0..r.len() {
            let idx = r.len() - 1 - j;
            let t = r[idx];
            let one_minus_t = E::one() - t;
            let prev_len = 1 << j;
            for i in (0..prev_len).rev() {
                result[j + 1][2 * i + 1] = result[j][i] * t;
                result[j + 1][2 * i] = result[j][i] * one_minus_t;
            }
        }
        Ok(result)
    }

    /// Parallel version of [`Self::evals_with_scaling`].
    ///
    /// Uses rayon to compute the largest layers of the DP tree in parallel.
    /// Uses the same **little-endian** index order as [`Self::evals`].
    #[cfg(feature = "parallel")]
    pub fn evals_parallel(r: &[E], scaling_factor: Option<E>) -> Result<Vec<E>, AkitaError> {
        use rayon::prelude::*;

        let final_size = Self::checked_table_len("eq evaluation table", r.len())?;
        let mut evals = Self::zero_vec("eq evaluation table", final_size)?;
        evals[0] = scaling_factor.unwrap_or(E::one());
        let mut size = 1;

        // Forward iteration (r[0] first) produces little-endian ordering.
        for &r_i in r.iter() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_i;
                    *x -= *y;
                });

            size *= 2;
        }

        Ok(evals)
    }
}

/// Dao-Thaler / Gruen split of the equality table (eprint 2024/1210).
///
/// Instead of materializing the full `2^n` table `eq(point, ·)`, store the two
/// half-tables `e_in = eq(point[..m], ·)` (low / inner bits) and
/// `e_out = eq(point[m..], ·)` (high / outer bits) with `m = n / 2`. By the
/// product structure of `eq`, for an index `x = x_out * in_len + x_in`
/// (little-endian, so `x_in` is the low `m` bits):
///
/// ```text
/// eq(point, x) = e_out[x_out] * e_in[x_in].
/// ```
///
/// This cuts the equality allocation from `2^n` to `2^{n-m} + 2^m` and lets a
/// contraction `Σ_x eq(point, x) · src(x)` run as an outer loop over `x_out`
/// (parallelizable) wrapping an inner loop over `x_in` that can defer reduction
/// via [`akita_field::MulBaseUnreduced`].
#[derive(Debug, Clone)]
pub struct SplitEqEvals<E: FieldCore> {
    /// Equality table over the high (outer) `n - m` variables, size `2^{n-m}`.
    pub e_out: Vec<E>,
    /// Equality table over the low (inner) `m` variables, size `2^m`.
    pub e_in: Vec<E>,
}

impl<E: FieldCore> SplitEqEvals<E> {
    /// Build the split tables for `eq(point, ·)`. The low `point.len() / 2`
    /// coordinates form the inner table; the rest form the outer table.
    ///
    /// # Errors
    ///
    /// Propagates [`EqPolynomial::evals`] allocation / overflow errors.
    pub fn new(point: &[E]) -> Result<Self, AkitaError> {
        let m = point.len() / 2;
        let e_in = EqPolynomial::evals(&point[..m])?;
        let e_out = EqPolynomial::evals(&point[m..])?;
        Ok(Self { e_out, e_in })
    }

    /// Number of inner (low-bit) indices, `2^m`.
    pub fn in_len(&self) -> usize {
        self.e_in.len()
    }

    /// Number of outer (high-bit) indices, `2^{n-m}`.
    pub fn out_len(&self) -> usize {
        self.e_out.len()
    }

    /// Total number of Boolean-hypercube entries represented by the split
    /// tables.
    #[inline]
    pub fn len(&self) -> usize {
        self.e_in.len() * self.e_out.len()
    }

    /// Whether either split factor is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.e_in.is_empty() || self.e_out.is_empty()
    }

    /// Evaluate the split equality table at one little-endian index.
    ///
    /// This keeps callers that need a sparse or strided subset of the table
    /// from materializing the full `2^n` vector.
    pub fn eval_at(&self, index: usize) -> Result<E, AkitaError> {
        if index >= self.len() {
            return Err(AkitaError::InvalidInput(format!(
                "split equality index {index} is outside table length {}",
                self.len()
            )));
        }
        let in_len = self.in_len();
        Ok(self.e_out[index / in_len] * self.e_in[index % in_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RandomSampling;
    use akita_field::Fp64;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;

    #[test]
    fn evals_matches_mle_pointwise() {
        let mut rng = StdRng::seed_from_u64(0xEE);
        for n in 1..8 {
            let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let table = EqPolynomial::evals(&r).unwrap();
            assert_eq!(table.len(), 1 << n);
            for (idx, &val) in table.iter().enumerate() {
                let bits: Vec<F> = (0..n)
                    .map(|k| {
                        if (idx >> k) & 1 == 1 {
                            F::one()
                        } else {
                            F::zero()
                        }
                    })
                    .collect();
                let expected = EqPolynomial::mle(&r, &bits).unwrap();
                assert_eq!(val, expected, "n={n} idx={idx}");
            }
        }
    }

    #[test]
    fn evals_with_scaling_scales_uniformly() {
        let mut rng = StdRng::seed_from_u64(0xAB);
        let r: Vec<F> = (0..5).map(|_| F::random(&mut rng)).collect();
        let scale = F::from_u64(7);
        let unscaled = EqPolynomial::evals(&r).unwrap();
        let scaled = EqPolynomial::evals_with_scaling(&r, Some(scale)).unwrap();
        for (u, s) in unscaled.iter().zip(scaled.iter()) {
            assert_eq!(*s, *u * scale);
        }
    }

    #[test]
    fn split_eq_evals_factorizes_full_table() {
        let mut rng = StdRng::seed_from_u64(0x5917);
        for n in 0..9 {
            let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let full = EqPolynomial::evals(&r).unwrap();
            let split = SplitEqEvals::new(&r).unwrap();
            assert_eq!(split.in_len() * split.out_len(), full.len(), "n={n}");
            let in_len = split.in_len();
            for x_out in 0..split.out_len() {
                for x_in in 0..in_len {
                    let idx = x_out * in_len + x_in;
                    assert_eq!(
                        split.e_out[x_out] * split.e_in[x_in],
                        full[idx],
                        "n={n} x_out={x_out} x_in={x_in}"
                    );
                }
            }
        }
    }

    #[test]
    fn split_eq_evals_supports_sparse_lookup() {
        let mut rng = StdRng::seed_from_u64(0x5EED);
        let point: Vec<F> = (0..17).map(|_| F::random(&mut rng)).collect();
        let split = SplitEqEvals::new(&point).unwrap();
        assert_eq!(split.len(), 1 << point.len());
        for index in [0, 1, 127, 1 << 16, (1 << 17) - 1] {
            let bits: Vec<F> = (0..point.len())
                .map(|bit| F::from_u64(((index >> bit) & 1) as u64))
                .collect();
            assert_eq!(
                split.eval_at(index).unwrap(),
                EqPolynomial::mle(&point, &bits).unwrap()
            );
        }
        assert!(split.eval_at(split.len()).is_err());
    }

    #[test]
    fn evals_cached_last_matches_evals() {
        let mut rng = StdRng::seed_from_u64(0xCD);
        for n in 1..8 {
            let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let table = EqPolynomial::evals(&r).unwrap();
            let cached = EqPolynomial::evals_cached(&r).unwrap();
            assert_eq!(cached.len(), n + 1);
            assert_eq!(cached[0], vec![F::one()]);
            assert_eq!(*cached.last().unwrap(), table);
        }
    }

    #[test]
    fn materialized_budget_is_not_the_serialization_sequence_cap() {
        let entries = akita_serialization::DEFAULT_MAX_SEQUENCE_LEN
            .checked_mul(2)
            .unwrap();
        EqPolynomial::<F>::check_element_budget("test eq table", entries).unwrap();
    }

    #[test]
    fn evals_rejects_tables_over_materialized_budget() {
        let max_entries = MAX_MATERIALIZED_EQ_TABLE_BYTES / mem::size_of::<F>().max(1);
        EqPolynomial::<F>::check_element_budget("test eq table", max_entries).unwrap();
        assert!(EqPolynomial::<F>::check_element_budget("test eq table", max_entries + 1).is_err());
    }

    #[test]
    fn evals_cached_rejects_total_layer_budget_overflow() {
        let max_entries = MAX_MATERIALIZED_EQ_TABLE_BYTES / mem::size_of::<F>().max(1);
        let mut final_len = 1usize;
        let mut vars = 0usize;
        loop {
            let total_len = final_len
                .checked_mul(2)
                .and_then(|len| len.checked_sub(1))
                .unwrap();
            if total_len > max_entries {
                break;
            }
            final_len = final_len.checked_mul(2).unwrap();
            vars += 1;
        }
        let r = vec![F::one(); vars];
        assert!(EqPolynomial::<F>::evals_cached(&r).is_err());
    }

    #[test]
    fn zero_selector_matches_mle_at_origin() {
        let mut rng = StdRng::seed_from_u64(0x00);
        for n in 1..8 {
            let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let zeros = vec![F::zero(); n];
            let expected = EqPolynomial::mle(&r, &zeros).unwrap();
            let actual = EqPolynomial::zero_selector(&r);
            assert_eq!(actual, expected, "n={n}");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn evals_parallel_matches_serial() {
        let mut rng = StdRng::seed_from_u64(0xFF);
        for n in 1..20 {
            let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let serial = EqPolynomial::evals_serial(&r, None).unwrap();
            let parallel = EqPolynomial::evals_parallel(&r, None).unwrap();
            assert_eq!(serial, parallel, "n={n}");
        }
    }
}
