//! Independent dense combined sumcheck for tests and benchmark baselines.

use akita_error::{checked, AkitaError};
use akita_labinius_verifier::channel::ClearChannel;
use akita_labinius_verifier::root_sumcheck::{
    alphabet_polynomial, bind_root_sumcheck_instance, combined_input_claim, combined_shape,
    RootSumcheckInstance,
};
use akita_sumcheck::{
    prove_sumcheck, InfallibleSumcheck, SumcheckInstanceProver, SumcheckProverChannel,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use jolt_poly::UnivariatePoly;

/// Combined alphabet-and-linear-relation instance over the coefficient field.
///
/// For `N = 2^nu`, owned peak table storage (excluding caller-held originals)
/// is `(2*N + max(1,N/2) + nu) * size_of::<F>() + N` bytes, plus
/// `16^2 * 18` field elements for the round-zero digit-pair table
/// and a constant round interpolation workspace. The equality factor uses a
/// prefix scalar and one suffix table. At `nu=26`, a dense table occupies
/// 1 GiB when field elements occupy 16 bytes. Capacities of the folded tables
/// remain allocated; the digit copy is released after the first challenge.
#[derive(Debug)]
pub(crate) struct CombinedRootSumcheck<F: Field> {
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

impl<F: Field> CombinedRootSumcheck<F> {
    /// Construct an honest digit instance, rejecting malformed statement data.
    /// The supplied linear claim need not be true; sumcheck checks it.
    pub(crate) fn new(w: &[u8], kw: Vec<F>, tau: &[F], beta: F, s: F) -> Result<Self, AkitaError> {
        let invalid = || AkitaError::InvalidInput("invalid combined root sumcheck geometry".into());
        let expected = checked::pow2(tau.len()).ok_or_else(invalid)?;
        if !w.len().is_power_of_two() || w.len() != expected || kw.len() != expected {
            return Err(invalid());
        }
        let shape = combined_shape(tau.len())?;
        let alphabet_size = 16;
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
                    digit_pair_values
                        .push(alphabet_polynomial(w0 + F::from_u64(node as u64) * delta));
                }
            }
        }
        Ok(Self {
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
    pub(crate) fn final_evaluations(&self) -> Option<(F, F)> {
        if self.next_round != self.tau.len() {
            return None;
        }
        Some((*self.w.first()?, *self.kw.first()?))
    }
}

impl<F: Field> SumcheckInstanceProver<F> for CombinedRootSumcheck<F> {
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
        // Round degree 17: the alphabet polynomial times the equality line.
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
                    .unwrap_or_else(|| alphabet_polynomial(w));
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
        // `Field` does not require jolt_field::Fold: use field arithmetic
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

/// Bind the combined instance identity and prove its rounds on the channel.
/// The returned claim still requires the terminal check and table opening.
pub(crate) fn prove_combined_rounds<F, E, C>(
    instance: &mut CombinedRootSumcheck<E>,
    channel: &mut C,
    invocation: u32,
) -> Result<(Vec<E>, E), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    C: SumcheckProverChannel<E> + ClearChannel,
{
    let num_vars = instance.num_rounds();
    let shape = combined_shape(num_vars)?;
    bind_root_sumcheck_instance(
        channel,
        RootSumcheckInstance::Combined,
        invocation,
        num_vars,
    )?;
    prove_sumcheck::<F, E, C, _>(
        &mut InfallibleSumcheck(instance),
        channel,
        shape,
        invocation,
    )
}
