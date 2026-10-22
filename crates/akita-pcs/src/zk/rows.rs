//! The response proof's row batch (Whiteout §4.3, "One opening for the
//! response"): the affine rows of Items 2, 3 and 5, the three product
//! claims, and the zero-padding rows, weighted by `eq(η, ·)` into one
//! equation `⟨A, e⟩ = b_η`. The row maps and their transposes are
//! jolt-zk's ([`ResponseStatement::rows`], [`ResponseStatement::weigh`]);
//! this module owns the row order.

use akita_error::AkitaError;
use jolt_algebra::JoltField;
use jolt_poly::EqPolynomial;
use jolt_zk::masking::ZkError;
use jolt_zk::whiteout::{ResponseRows, ResponseStatement};

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
/// target, the `U` and `W` claims, then the padding rows `e_j = 0` for
/// `j ∈ [len, 2^ν)`.
struct RowOrder {
    v_image: usize,
    linear: usize,
    padding: usize,
}

impl RowOrder {
    fn new<F: JoltField, const D: usize>(
        statement: &ResponseStatement<'_, F, D>,
        shape: ResponseShape,
    ) -> Self {
        let rows = statement.row_shape();
        Self {
            v_image: rows.v_image,
            linear: rows.linear,
            padding: shape.padded_len() - shape.len,
        }
    }

    fn ct_h(&self) -> usize {
        self.v_image + self.linear
    }

    fn product_target(&self) -> usize {
        self.ct_h() + 1
    }

    fn len(&self) -> usize {
        self.product_target() + 3 + self.padding
    }
}

/// The number of coordinates of `η`.
pub(crate) fn row_variables<F: JoltField, const D: usize>(
    statement: &ResponseStatement<'_, F, D>,
    shape: ResponseShape,
) -> usize {
    ceil_log2(RowOrder::new(statement, shape).len())
}

/// The batch weights `eq(η, l)`: on [`ResponseRows`], on the padding
/// entries of `e`, and on the three product claims.
pub(crate) struct RowWeights<F> {
    pub(crate) rows: ResponseRows<F>,
    pub(crate) padding: Vec<F>,
    pub(crate) claims: ProductClaims<F>,
}

impl<F: JoltField> RowWeights<F> {
    /// `eq(η, l)` on every row; `product_point` is `r_×`, at which the `U`
    /// and `W` claims hold.
    pub(crate) fn new<const D: usize>(
        statement: &ResponseStatement<'_, F, D>,
        shape: ResponseShape,
        eta: &[F],
        product_point: &[F],
    ) -> Result<Self, ZkError> {
        let order = RowOrder::new(statement, shape);
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
            padding: slice(target + 3..order.len())?,
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

/// Batches every row at `eta`: `A_j = Σ_l eq(η, l) H_{l,j}` and
/// `b_η = Σ_l eq(η, l) b_l`, through the rows' transpose.
pub(crate) fn batch_rows<F: JoltField, const D: usize>(
    statement: &ResponseStatement<'_, F, D>,
    shape: ResponseShape,
    alpha: F,
    weights: RowWeights<F>,
    claims: ProductClaims<F>,
) -> Result<BatchedRows<F>, ZkError> {
    let (mut a, constant) = statement.weigh(&weights.rows, alpha)?;
    if a.len() != shape.len {
        return Err(zk_error(AkitaError::Internal("row weights length".into())));
    }
    let b = weights.target(claims) - constant;
    a.extend(weights.padding);
    Ok(BatchedRows { a, b })
}

/// An Akita failure, carried as a [`ZkError`].
pub(crate) fn zk_error(error: AkitaError) -> ZkError {
    ZkError::Whiteout(Box::new(error))
}
