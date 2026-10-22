//! The response proof's row batch (Whiteout §4.3, "One opening for the
//! response"): the affine rows of Items 2, 3 and 5, the three product
//! claims and the compression chains, weighted by `eq(η, ·)`, plus the
//! padding term `Σ_{j ≥ len} eq(τ_pad, j)·e_j`, into one equation
//! `⟨A, e⟩ = b_η`. The prover builds `A`; the verifier evaluates `A(r)`
//! from the factored rows (§5.8, P1). The row maps and their transposes are
//! jolt-zk's ([`ResponseStatement::rows`], [`ResponseStatement::weigh`]);
//! this module owns the row order.

use akita_error::AkitaError;
use akita_types::NegativeBinarySupport;
use jolt_algebra::{JoltField, Zero};
use jolt_poly::EqPolynomial;
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::{ResponseRows, ResponseStatement};

use super::compression::Chains;
use super::F;

/// The padded size of `e`: `2^ν` entries, the first `len` of them `e`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResponseShape {
    pub(crate) len: usize,
    pub(crate) num_vars: usize,
}

impl ResponseShape {
    pub(crate) fn new(len: usize) -> Result<Self, AkitaError> {
        if len == 0 {
            return Err(AkitaError::InvalidInput("empty response vector".into()));
        }
        Ok(Self {
            len,
            num_vars: ceil_log2(len),
        })
    }

    pub(crate) fn padded_len(self) -> usize {
        1 << self.num_vars
    }
}

/// `eq(point, j)` for every `j`, the point little-endian (`point[0]` is the
/// least significant index bit), as Akita's tables and sumchecks bind it.
/// Unlike Akita's budgeted tables, it materializes the full `2^27` domain
/// of the response vector.
pub(crate) fn eq_table<F: JoltField>(point: &[F]) -> Vec<F> {
    EqPolynomial::new(point.iter().rev().copied().collect()).evaluations()
}

/// `⌈log₂ n⌉` for `n ≥ 1`.
pub(crate) fn ceil_log2(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

/// The values the product reduction sends: `t_×`, `u_×` and `w_×`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProductClaims<F> {
    pub(crate) target: F,
    pub(crate) x: F,
    pub(crate) y: F,
}

/// `⟨A, e⟩ = b`, with `A` over the padded `e`.
pub(crate) struct BatchedRows<F> {
    pub(crate) a: Vec<F>,
    pub(crate) b: F,
}

/// Row indices: Items 2, 3 and 5 in [`ResponseRows`] order, the product
/// target, the `U` and `W` claims, then the compression chains' rows.
struct RowOrder {
    v_image: usize,
    linear: usize,
    chains: usize,
}

impl RowOrder {
    fn new<const D: usize>(
        statement: &ResponseStatement<'_, F, D>,
        chains: &Chains<'_>,
    ) -> Result<Self, AkitaError> {
        let rows = statement.row_shape();
        Ok(Self {
            v_image: rows.v_image,
            linear: rows.linear,
            chains: chains.rows()?,
        })
    }

    fn chains(&self) -> std::ops::Range<usize> {
        let start = self.product_target() + 3;
        start..start + self.chains
    }

    fn ct_h(&self) -> usize {
        self.v_image + self.linear
    }

    fn product_target(&self) -> usize {
        self.ct_h() + 1
    }

    fn len(&self) -> usize {
        self.chains().end
    }
}

/// The number of coordinates of `η`.
pub(crate) fn row_variables<const D: usize>(
    statement: &ResponseStatement<'_, F, D>,
    chains: &Chains<'_>,
) -> Result<usize, AkitaError> {
    Ok(ceil_log2(RowOrder::new(statement, chains)?.len()))
}

/// The batch weights `eq(η, l)`: on [`ResponseRows`], on the chains' rows,
/// and on the three product claims.
pub(crate) struct RowWeights<F> {
    pub(crate) rows: ResponseRows<F>,
    pub(crate) chains: Vec<F>,
    pub(crate) claims: ProductClaims<F>,
}

/// The padding term `Σ_{j ∈ [len, 2^ν)} eq(τ_pad, j)·e_j`. If `e` is
/// nonzero past `len`, the term is a nonzero multilinear polynomial in
/// `τ_pad`, and with the η batch one nonzero polynomial in disjoint
/// variables, so a cheat survives with probability at most `ν/q` beyond the
/// batch's own; no separate batching scalar is needed. It hides nothing.
pub(crate) struct PaddingTerm<F> {
    pub(crate) point: Vec<F>,
}

impl PaddingTerm<F> {
    /// The padding entries of `e` and their weights.
    fn support(&self, shape: ResponseShape) -> Result<Option<NegativeBinarySupport>, AkitaError> {
        if shape.len == shape.padded_len() {
            return Ok(None);
        }
        NegativeBinarySupport::from_intervals(
            std::iter::once(shape.len..shape.padded_len()).collect(),
            shape.padded_len(),
        )
        .map(Some)
    }

    /// The term's weights on `e`'s padding entries.
    pub(crate) fn weights(&self, shape: ResponseShape) -> Result<Vec<F>, AkitaError> {
        if self.point.len() != shape.num_vars {
            return Err(AkitaError::InvalidInput("padding point width".into()));
        }
        let half = shape.num_vars / 2;
        let (lo, hi) = (eq_table(&self.point[..half]), eq_table(&self.point[half..]));
        let mask = (1 << half) - 1;
        Ok((shape.len..shape.padded_len())
            .map(|j| lo[j & mask] * hi[j >> half])
            .collect())
    }

    /// The term's weight at `point`: `Σ_{j ≥ len} eq(τ_pad, j)·eq(point, j)`.
    pub(crate) fn at(&self, shape: ResponseShape, point: &[F]) -> Result<F, AkitaError> {
        Ok(match self.support(shape)? {
            Some(support) => support.evaluate_restricted_equality_at_point(&self.point, point)?,
            None => F::zero(),
        })
    }
}

impl RowWeights<F> {
    /// `eq(η, l)` on every row; `product_point` is `r_×`, at which the `U`
    /// and `W` claims hold.
    pub(crate) fn new<const D: usize>(
        statement: &ResponseStatement<'_, F, D>,
        eta: &[F],
        product_point: &[F],
        chains: &Chains<'_>,
    ) -> Result<Self, ZkError> {
        let order = RowOrder::new(statement, chains).map_err(zk_error)?;
        if eta.len() != ceil_log2(order.len()) {
            return Err(zk_error(AkitaError::InvalidInput(
                "row batching point width".into(),
            )));
        }
        let weights = eq_table(eta);
        let slice = |range: std::ops::Range<usize>| {
            weights
                .get(range)
                .map(<[F]>::to_vec)
                .ok_or_else(|| zk_error(AkitaError::Internal("rows outside the batch".into())))
        };
        let target = order.product_target();
        let [target_weight, x_weight, y_weight] = <[F; 3]>::try_from(slice(target..target + 3)?)
            .map_err(|_| zk_error(AkitaError::Internal("product rows".into())))?;
        let products = statement.row_shape().products;
        let kappa = eq_table(product_point);
        let kappa = kappa
            .get(..products)
            .ok_or_else(|| zk_error(AkitaError::InvalidInput("product point too short".into())))?;
        Ok(Self {
            rows: ResponseRows {
                v_image: slice(0..order.v_image)?,
                linear: slice(order.v_image..order.ct_h())?,
                ct_h: weights.get(order.ct_h()).copied().ok_or_else(|| {
                    zk_error(AkitaError::Internal("rows outside the batch".into()))
                })?,
                product_target: target_weight,
                product_x: kappa.iter().map(|k| *k * x_weight).collect(),
                product_y: kappa.iter().map(|k| *k * y_weight).collect(),
            },
            chains: slice(order.chains())?,
            claims: ProductClaims {
                target: target_weight,
                x: x_weight,
                y: y_weight,
            },
        })
    }

    /// `Σ_l eq(η, l)·rhs_l`: only the product rows have nonzero targets.
    pub(crate) fn target(&self, claims: ProductClaims<F>) -> F {
        self.claims.target * claims.target + self.claims.x * claims.x + self.claims.y * claims.y
    }
}

/// `b_η = Σ_l eq(η, l)·b_l`: the product claims' targets less the rows'
/// constants, from the public targets and payloads alone.
pub(crate) fn batch_target<const D: usize>(
    statement: &ResponseStatement<'_, F, D>,
    alpha: F,
    weights: &RowWeights<F>,
    claims: ProductClaims<F>,
    chains: &Chains<'_>,
) -> Result<F, ZkError> {
    let constant = statement.constant(&weights.rows, alpha)?;
    let chain_constant = chains.constant(alpha, &weights.chains).map_err(zk_error)?;
    Ok(weights.target(claims) - constant - chain_constant)
}

/// Batches every row at `η` for the prover: `A_j = Σ_l eq(η, l) H_{l,j}`
/// through the rows' transpose, plus the padding term, and `b_η`.
pub(crate) fn batch_rows<const D: usize>(
    statement: &ResponseStatement<'_, F, D>,
    shape: ResponseShape,
    alpha: F,
    weights: &RowWeights<F>,
    padding: &PaddingTerm<F>,
    claims: ProductClaims<F>,
    chains: &Chains<'_>,
) -> Result<BatchedRows<F>, ZkError> {
    let (mut a, _) = statement.weigh(&weights.rows, alpha)?;
    if a.len() != shape.len {
        return Err(zk_error(AkitaError::Internal("row weights length".into())));
    }
    a.extend(padding.weights(shape).map_err(zk_error)?);
    chains
        .accumulate(alpha, &weights.chains, &mut a)
        .map_err(zk_error)?;
    Ok(BatchedRows {
        a,
        b: batch_target(statement, alpha, weights, claims, chains)?,
    })
}

/// `A(r)` for the verifier, without `A`: jolt-zk's factored rows, the
/// chains' relation and the padding term, each at `point`.
pub(crate) fn batch_weight_at<const D: usize>(
    statement: &ResponseStatement<'_, F, D>,
    shape: ResponseShape,
    alpha: F,
    weights: &RowWeights<F>,
    padding: &PaddingTerm<F>,
    chains: &Chains<'_>,
    point: &[F],
) -> Result<F, ZkError> {
    Ok(statement.weight_at(&weights.rows, alpha, point)?
        + chains
            .evaluate(alpha, &weights.chains, point)
            .map_err(zk_error)?
        + padding.at(shape, point).map_err(zk_error)?)
}

/// An Akita failure, carried as a [`ZkError`].
pub(crate) fn zk_error(error: AkitaError) -> ZkError {
    ZkError::Whiteout(Box::new(error))
}
