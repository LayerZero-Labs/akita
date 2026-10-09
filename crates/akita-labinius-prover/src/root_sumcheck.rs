//! Dense product sumcheck for the succinct LaBinius root.
//!
//! Round `i` binds table-index bit `i`. Caller code must bind the statement
//! and obtain the terminal polynomial openings separately.

use akita_algebra::SmoothFftField;
use akita_error::AkitaError;
use akita_labinius_verifier::channel::ClearChannel;
use akita_labinius_verifier::root_sumcheck::{
    bind_root_sumcheck_instance, product_shape, RootSumcheckInstance,
};
use akita_sumcheck::{
    prove_sumcheck, InfallibleSumcheck, SumcheckInstanceProver, SumcheckProverChannel,
};
use jolt_field::ExtField;
use jolt_poly::UnivariatePoly;

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

/// Bind the product instance identity and prove its rounds on the channel.
/// The returned claim still requires the terminal check and table opening.
pub fn prove_product_rounds<F, C>(
    instance: &mut ProductSumcheck<F>,
    channel: &mut C,
    invocation: u32,
) -> Result<(Vec<F>, F), AkitaError>
where
    F: SmoothFftField + ExtField<F>,
    C: SumcheckProverChannel<F> + ClearChannel,
{
    let num_vars = instance.num_rounds();
    let shape = product_shape(num_vars)?;
    bind_root_sumcheck_instance(channel, RootSumcheckInstance::Product, invocation, num_vars)?;
    prove_sumcheck::<F, F, C, _>(
        &mut InfallibleSumcheck(instance),
        channel,
        shape,
        invocation,
    )
}
