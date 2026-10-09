//! Dense coefficient-field sumchecks for the succinct LaBinius root.
//!
//! Round `i` binds table-index bit `i`. Caller code must bind the statement
//! and obtain the terminal polynomial openings separately.

use akita_algebra::SmoothFftField;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::root_sumcheck::{
    alphabet_polynomial, combined_input_claim, combined_shape, product_shape,
};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::SumcheckInstanceProver;
use jolt_poly::UnivariatePoly;

/// Combined alphabet-and-linear-relation instance over the coefficient field.
///
/// For `N = 2^nu`, owned peak table storage (excluding caller-held originals)
/// is `(2*N + max(1,N/2) + nu) * size_of::<F>() + N` bytes, plus
/// `2^(2*b) * (2^b + 2)` field elements for the round-zero digit-pair table
/// and `O(2^b)` round interpolation workspace. The equality factor uses a
/// prefix scalar and one suffix table. At `nu=26`, a dense table occupies
/// 1 GiB when field elements occupy 16 bytes. Capacities of the folded tables
/// remain allocated; the digit copy is released after the first challenge.
#[derive(Debug)]
pub struct CombinedRootSumcheck<F: SmoothFftField> {
    base: LabiniusDigitBase,
    w: Vec<F>,
    kw: Vec<F>,
    tau: Vec<F>,
    equality_suffix: Vec<F>,
    equality_prefix: F,
    beta: F,
    claim: F,
    degree: usize,
    next_round: usize,
    digits: Option<Vec<u8>>,
    digit_pair_values: Vec<F>,
    alphabet_size: usize,
}

impl<F: SmoothFftField> CombinedRootSumcheck<F> {
    /// Construct an honest digit instance, rejecting malformed statement data.
    /// The supplied linear claim need not be true; sumcheck checks it.
    pub fn new(
        base: LabiniusDigitBase,
        w: &[u8],
        kw: Vec<F>,
        tau: &[F],
        beta: F,
        s: F,
    ) -> Result<Self, AkitaError> {
        let invalid = || AkitaError::InvalidInput("invalid combined root sumcheck geometry".into());
        let expected = checked::pow2(tau.len()).ok_or_else(invalid)?;
        if !w.len().is_power_of_two() || w.len() != expected || kw.len() != expected {
            return Err(invalid());
        }
        let shape = combined_shape(tau.len(), base)?;
        let alphabet_size = checked::pow2(base.bits() as usize).ok_or_else(invalid)?;
        if w.iter().any(|&digit| usize::from(digit) >= alphabet_size) {
            return Err(AkitaError::InvalidInput(
                "root digit leaves its alphabet".into(),
            ));
        }
        let mut field_w = Vec::new();
        field_w.try_reserve_exact(expected).map_err(|_| invalid())?;
        field_w.extend(w.iter().map(|&digit| F::from_u64(u64::from(digit))));
        let mut digits = Vec::new();
        digits.try_reserve_exact(expected).map_err(|_| invalid())?;
        digits.extend_from_slice(w);
        let mut equality_point = Vec::new();
        equality_point
            .try_reserve_exact(tau.len())
            .map_err(|_| invalid())?;
        equality_point.extend_from_slice(tau);

        let suffix_len = (expected / 2).max(1);
        let mut equality_suffix = Vec::new();
        equality_suffix
            .try_reserve_exact(suffix_len)
            .map_err(|_| invalid())?;
        equality_suffix.push(F::one());
        for &coordinate in tau.iter().skip(1) {
            let live_len = equality_suffix.len();
            for index in 0..live_len {
                let value = equality_suffix[index];
                equality_suffix[index] = value * (F::one() - coordinate);
                equality_suffix.push(value * coordinate);
            }
        }

        let width = checked::sum([shape.degree_bound(), 1]).ok_or_else(invalid)?;
        let lut_len =
            checked::product([alphabet_size, alphabet_size, width]).ok_or_else(invalid)?;
        let mut digit_pair_values = Vec::new();
        digit_pair_values
            .try_reserve_exact(lut_len)
            .map_err(|_| invalid())?;
        for left in 0..alphabet_size {
            for right in 0..alphabet_size {
                let w0 = F::from_u64(left as u64);
                let delta = F::from_u64(right as u64) - w0;
                for node in 0..width {
                    digit_pair_values.push(alphabet_polynomial(
                        base,
                        w0 + F::from_u64(node as u64) * delta,
                    ));
                }
            }
        }
        Ok(Self {
            base,
            w: field_w,
            kw,
            tau: equality_point,
            equality_suffix,
            equality_prefix: F::one(),
            beta,
            claim: combined_input_claim(beta, s),
            degree: shape.degree_bound(),
            next_round: 0,
            digits: Some(digits),
            digit_pair_values,
            alphabet_size,
        })
    }

    /// Return `(W~(rho), K_W~(rho))` once every challenge has been bound.
    pub fn final_evaluations(&self) -> Option<(F, F)> {
        if self.next_round != self.tau.len() {
            return None;
        }
        Some((*self.w.first()?, *self.kw.first()?))
    }
}

impl<F: SmoothFftField> SumcheckInstanceProver<F> for CombinedRootSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.tau.len()
    }
    fn degree_bound(&self) -> usize {
        self.degree
    }
    fn input_claim(&self) -> F {
        self.claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        let Some(&tau) = self.tau.get(self.next_round) else {
            return UnivariatePoly::zero();
        };
        // The public closed base enum limits this array to degree 17.
        let mut values = [F::zero(); 18];
        let width = self.degree + 1;
        for (pair_index, ((w_pair, kw_pair), &suffix)) in self
            .w
            .chunks_exact(2)
            .zip(self.kw.chunks_exact(2))
            .zip(&self.equality_suffix)
            .enumerate()
        {
            let w0 = w_pair[0];
            let dw = w_pair[1] - w0;
            let kw0 = kw_pair[0];
            let dkw = kw_pair[1] - kw0;
            let lut_row = self
                .digits
                .as_ref()
                .and_then(|digits| digits.chunks_exact(2).nth(pair_index))
                .and_then(|pair| {
                    checked::mul_add(
                        usize::from(pair[0]),
                        self.alphabet_size,
                        usize::from(pair[1]),
                    )
                })
                .and_then(|row| checked::product([row, width]))
                .and_then(|start| checked::range(start, width))
                .and_then(|range| self.digit_pair_values.get(range));
            for (node, value) in values.iter_mut().take(width).enumerate() {
                let t = F::from_u64(node as u64);
                let w = w0 + t * dw;
                let kw = kw0 + t * dkw;
                let p = lut_row
                    .and_then(|row| row.get(node))
                    .copied()
                    .unwrap_or_else(|| alphabet_polynomial(self.base, w));
                let eq =
                    self.equality_prefix * suffix * ((F::one() - tau) * (F::one() - t) + tau * t);
                *value += eq * p + self.beta * w * kw;
            }
        }
        UnivariatePoly::from_evals(&values[..width])
    }

    fn ingest_challenge(&mut self, _round: usize, challenge: F) {
        let Some(&tau) = self.tau.get(self.next_round) else {
            return;
        };
        // SmoothFftField does not require jolt_field::Fold: use field arithmetic
        // directly, without the optimized Fold-only algebra helper.
        for table in [&mut self.w, &mut self.kw] {
            let half = table.len() / 2;
            for index in 0..half {
                table[index] =
                    table[2 * index] + challenge * (table[2 * index + 1] - table[2 * index]);
            }
            table.truncate(half);
        }
        self.equality_prefix *= (F::one() - tau) * (F::one() - challenge) + tau * challenge;
        let half = self.equality_suffix.len() / 2;
        if half != 0 {
            for index in 0..half {
                self.equality_suffix[index] =
                    self.equality_suffix[2 * index] + self.equality_suffix[2 * index + 1];
            }
            self.equality_suffix.truncate(half);
        }
        self.digits = None;
        self.next_round += 1;
    }
}

/// Product instance for `sum_x Y(x) * K_Y(x)` over the coefficient field.
///
/// For `N = 2^mu`, peak owned storage is `2*N` field elements plus constant
/// interpolation workspace. In-place folding retains the initial capacities.
#[derive(Debug)]
pub struct ProductSumcheck<F: SmoothFftField> {
    y: Vec<F>,
    ky: Vec<F>,
    rounds: usize,
    next_round: usize,
    claim: F,
}

impl<F: SmoothFftField> ProductSumcheck<F> {
    /// Construct a product instance from equally sized nonempty Boolean tables.
    pub fn new(y: Vec<F>, ky: Vec<F>, claimed_sum: F) -> Result<Self, AkitaError> {
        if !y.len().is_power_of_two() || y.len() != ky.len() {
            return Err(AkitaError::InvalidInput(
                "invalid product sumcheck geometry".into(),
            ));
        }
        let rounds = y.len().trailing_zeros() as usize;
        product_shape(rounds)?;
        Ok(Self {
            y,
            ky,
            rounds,
            next_round: 0,
            claim: claimed_sum,
        })
    }

    /// Return `(Y~(rho'), K_Y~(rho'))` once every challenge has been bound.
    pub fn final_evaluations(&self) -> Option<(F, F)> {
        if self.next_round != self.rounds {
            return None;
        }
        Some((*self.y.first()?, *self.ky.first()?))
    }
}

impl<F: SmoothFftField> SumcheckInstanceProver<F> for ProductSumcheck<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        2
    }
    fn input_claim(&self) -> F {
        self.claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: F) -> UnivariatePoly<F> {
        let mut values = [F::zero(); 3];
        for (y, ky) in self.y.chunks_exact(2).zip(self.ky.chunks_exact(2)) {
            for (node, value) in values.iter_mut().enumerate() {
                let t = F::from_u64(node as u64);
                *value += (y[0] + t * (y[1] - y[0])) * (ky[0] + t * (ky[1] - ky[0]));
            }
        }
        UnivariatePoly::from_evals(&values)
    }

    fn ingest_challenge(&mut self, _round: usize, challenge: F) {
        if self.next_round >= self.rounds {
            return;
        }
        for table in [&mut self.y, &mut self.ky] {
            let half = table.len() / 2;
            for index in 0..half {
                table[index] =
                    table[2 * index] + challenge * (table[2 * index + 1] - table[2 * index]);
            }
            table.truncate(half);
        }
        self.next_round += 1;
    }
}
