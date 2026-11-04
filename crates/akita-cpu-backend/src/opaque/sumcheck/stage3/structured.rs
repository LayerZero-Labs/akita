//! Stage 3 rounds on a setup source that is zero outside aligned live blocks.
//!
//! The weights stay in the caller's block form, where each live block is a
//! scaled sum of at most two pool rows. Each round contracts the setup values
//! of a block against its rows; binding folds the compact setup values and the
//! pool. Once a block is one cell wide, both tables scatter to the dense
//! index state, which the ordinary rounds continue on.

use super::product_table::SetupProductSource;
use super::utils::fold_pair;
use akita_error::{checked, AkitaError};
use jolt_field::solinas::parallel::*;
use jolt_field::{Field, MulBaseUnreduced, Ring, Unreduced, Zero};

/// Stage 3 weights of a setup source that is zero outside their live blocks.
///
/// Live block `k` covers cells `blocks()[k] * W .. (blocks()[k] + 1) * W`,
/// where `W` is the width before any binding. Every weight outside the live
/// blocks is zero. Binding folds the lowest unbound address bit, which lies
/// inside a block while the width is at least two.
pub(crate) trait StructuredSetupWeights<E: Field>: Send + Sync {
    /// Cells of the unbound domain.
    fn len(&self) -> usize;
    /// Current cells per block.
    fn width(&self) -> usize;
    /// Index of every live block.
    fn blocks(&self) -> &[usize];
    /// The `(scale, row)` terms of live block `live`: its weights are the sum
    /// of `scale * row`. A zero scale marks an absent term.
    fn block_terms(&self, live: usize) -> Result<[(E, &[E]); 2], AkitaError>;
    /// Bind the lowest unbound address bit to `challenge`.
    fn bind(&mut self, challenge: E) -> Result<(), AkitaError>;
}

/// Block phase of a structured setup product.
pub(super) struct StructuredSetupProduct<E: Field> {
    weights: Box<dyn StructuredSetupWeights<E>>,
    /// Cells per block before any binding.
    block_len: usize,
    /// Setup values of every live block at the current width, block-major.
    /// Empty before the first bind, when the rounds read the source.
    setup: Vec<E>,
}

/// Dense index state handed to the ordinary rounds.
pub(super) struct ScatteredIndexState<E> {
    pub(super) table: Vec<E>,
    pub(super) factor: Vec<E>,
}

impl<E> StructuredSetupProduct<E>
where
    E: Field + Ring + Unreduced,
{
    pub(super) fn new(weights: Box<dyn StructuredSetupWeights<E>>) -> Result<Self, AkitaError> {
        let width = weights.width();
        if !weights.len().is_power_of_two() || !width.is_power_of_two() || width > weights.len() {
            return Err(AkitaError::InvalidInput(
                "structured setup weights have an invalid block geometry".into(),
            ));
        }
        Ok(Self {
            weights,
            block_len: width,
            setup: Vec::new(),
        })
    }

    pub(super) fn len(&self) -> usize {
        self.weights.len()
    }

    /// `sum_x S(x) W(x)` over the live blocks of the unbound source.
    pub(super) fn input_claim<F: Field>(&self, source: &[F]) -> Result<E, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        cfg_try_fold_reduce!(
            0..self.weights.blocks().len(),
            E::zero,
            |mut claim, live| -> Result<E, AkitaError> {
                let cells = self.source_block(source, live)?;
                for (scale, row) in self.weights.block_terms(live)? {
                    if !scale.is_zero() {
                        let dot = row.iter().zip(cells).fold(
                            <E as Unreduced>::Product::zero(),
                            |sum, (&weight, &setup)| sum + weight.mul_base_unreduced(setup),
                        );
                        claim += scale * E::reduce_product(dot);
                    }
                }
                Ok(claim)
            },
            |left: E, right: E| Ok(left + right)
        )
    }

    /// Round polynomial `(constant, linear, quadratic)` of the current round.
    pub(super) fn round<F: Field>(
        &self,
        source: Option<&SetupProductSource<F>>,
    ) -> Result<(E, E, E), AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let width = self.weights.width();
        let blocks = self.weights.blocks().len();
        let (constant, one, quadratic) = if self.setup.is_empty() {
            let source = source
                .ok_or_else(|| AkitaError::Internal("setup source already released".into()))?
                .coefficients();
            cfg_try_fold_reduce!(
                0..blocks,
                zero3,
                |sums, live| -> Result<_, AkitaError> {
                    let cells = self.source_block(source, live)?;
                    self.block_round(sums, live, |row| {
                        pair_products(row, cells, |weight, setup| weight.mul_base_unreduced(setup))
                    })
                },
                |left, right| Ok(add3(left, right))
            )?
        } else {
            cfg_try_fold_reduce!(
                0..blocks,
                zero3,
                |sums, live| -> Result<_, AkitaError> {
                    let cells = self
                        .setup
                        .get(live * width..(live + 1) * width)
                        .ok_or(AkitaError::InvalidProof)?;
                    self.block_round(sums, live, |row| {
                        pair_products(row, cells, |weight, setup| weight.mul_unreduced(setup))
                    })
                },
                |left, right| Ok(add3(left, right))
            )?
        };
        Ok((constant, one - constant - quadratic, quadratic))
    }

    /// Add one block's `(sum s0 w0, sum s1 w1, sum ds dw)` to `sums`, term
    /// by term.
    fn block_round(
        &self,
        mut sums: (E, E, E),
        live: usize,
        products: impl Fn(&[E]) -> (E, E, E),
    ) -> Result<(E, E, E), AkitaError> {
        for (scale, row) in self.weights.block_terms(live)? {
            if !scale.is_zero() {
                let (constant, one, quadratic) = products(row);
                sums = add3(sums, (scale * constant, scale * one, scale * quadratic));
            }
        }
        Ok(sums)
    }

    /// Unbound source cells of live block `live`.
    fn source_block<'a, F>(&self, source: &'a [F], live: usize) -> Result<&'a [F], AkitaError> {
        let block = *self
            .weights
            .blocks()
            .get(live)
            .ok_or(AkitaError::InvalidProof)?;
        let start = checked::product([block, self.block_len]).ok_or(AkitaError::InvalidProof)?;
        source
            .get(start..)
            .and_then(|cells| cells.get(..self.block_len))
            .ok_or(AkitaError::InvalidProof)
    }

    /// Bind the lowest unbound bit. Returns the dense index state once every
    /// block is a single cell.
    pub(super) fn bind<F: Field>(
        &mut self,
        source: Option<&SetupProductSource<F>>,
        challenge: E,
    ) -> Result<Option<ScatteredIndexState<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let width = self.weights.width();
        let half = width / 2;
        if half == 0 {
            return Err(AkitaError::InvalidInput(
                "structured setup blocks are fully bound".into(),
            ));
        }
        let mut folded = vec![E::zero(); self.weights.blocks().len() * half];
        if self.setup.is_empty() {
            let source = source
                .ok_or_else(|| AkitaError::Internal("setup source already released".into()))?
                .coefficients();
            let this = &*self;
            cfg_chunks_mut!(&mut folded, half)
                .enumerate()
                .try_for_each(|(live, output)| -> Result<(), AkitaError> {
                    let cells = this.source_block(source, live)?;
                    for (value, pair) in output.iter_mut().zip(cells.chunks_exact(2)) {
                        *value = fold_pair(E::lift_base(pair[0]), E::lift_base(pair[1]), challenge);
                    }
                    Ok(())
                })?;
        } else {
            cfg_chunks_mut!(&mut folded, half)
                .zip(cfg_chunks!(&self.setup, width))
                .for_each(|(output, cells)| {
                    for (value, pair) in output.iter_mut().zip(cells.chunks_exact(2)) {
                        *value = fold_pair(pair[0], pair[1], challenge);
                    }
                });
        }
        self.setup = folded;
        self.weights.bind(challenge)?;
        if half > 1 {
            return Ok(None);
        }
        let rows = self.weights.len() / self.block_len;
        let mut table = vec![E::zero(); rows];
        let mut factor = vec![E::zero(); rows];
        for (live, (&block, &setup)) in self.weights.blocks().iter().zip(&self.setup).enumerate() {
            let weight = self
                .weights
                .block_terms(live)?
                .into_iter()
                .fold(E::zero(), |sum, (scale, row)| sum + scale * row[0]);
            *table.get_mut(block).ok_or(AkitaError::InvalidProof)? = setup;
            *factor.get_mut(block).ok_or(AkitaError::InvalidProof)? = weight;
        }
        Ok(Some(ScatteredIndexState { table, factor }))
    }
}

fn zero3<E: Field>() -> (E, E, E) {
    (E::zero(), E::zero(), E::zero())
}

fn add3<E: Field>(left: (E, E, E), right: (E, E, E)) -> (E, E, E) {
    (left.0 + right.0, left.1 + right.1, left.2 + right.2)
}

/// `(sum s0 r0, sum s1 r1, sum (s1 - s0)(r1 - r0))` over the cell pairs of
/// one block row `r` and its setup values `s`, each reduced once.
fn pair_products<E, S>(row: &[E], cells: &[S], product: impl Fn(E, S) -> E::Product) -> (E, E, E)
where
    E: Field + Unreduced,
    S: Copy + core::ops::Sub<Output = S>,
{
    let mut sums: [E::Product; 3] = core::array::from_fn(|_| Zero::zero());
    for (weights, setup) in row.chunks_exact(2).zip(cells.chunks_exact(2)) {
        sums[0] += product(weights[0], setup[0]);
        sums[1] += product(weights[1], setup[1]);
        sums[2] += product(weights[1] - weights[0], setup[1] - setup[0]);
    }
    sums.map(E::reduce_product).into()
}
