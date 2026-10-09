//! Transform adjoints for compact response and image coefficient weights.

use akita_algebra::{SmoothFftField, TrinomialModulus, TrinomialNtt, TrinomialRing};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    lowered::{trace_gram_into, trace_gram_inverse, LoweredPublic, LoweredRootLayout},
    BinaryClearSetup,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::info_span;

use crate::PreparedCommitMatrix;

/// Build the response-weight factor in coefficient-low, ring-element-high order.
///
/// The matrix cache must belong to `setup`, and `public` must use `layout`.
/// Multiplying this table by `public.digit_powers()` in digit-innermost order
/// gives the response weights. Each column is formed directly in its padded
/// destination by the transform adjoint `G(sum_i A_ij * G^-1 u_i)`, followed
/// by the unchanged parity weight. Coefficient padding remains zero. Parallel
/// tasks own independent transform workspaces; no dense m-by-D copy is built.
pub fn coefficient_weights<F, const D: usize, M>(
    prepared: &PreparedCommitMatrix<F, D, M>,
    layout: &LoweredRootLayout,
    setup: &BinaryClearSetup<F, D, M>,
    public: &LoweredPublic<F>,
) -> Result<Vec<F>, AkitaError>
where
    F: SmoothFftField,
    M: TrinomialModulus + Send + Sync,
{
    public.validate_layout(layout)?;
    prepared.check_setup(setup)?;
    let len = checked::product([layout.m(), layout.padded_coefficients()])
        .ok_or(AkitaError::InvalidProof)?;
    if layout.degree() != D
        || layout.m() != setup.m()
        || layout.n_a() != setup.n_a()
        || checked::product([len, public.digit_powers().len()]) != Some(layout.witness_len())
    {
        return Err(AkitaError::InvalidInput(
            "invalid compact response-weight geometry".into(),
        ));
    }
    let mut weights = zero_weights(len)?;
    let span = info_span!("root_a_weights").entered();
    let row_tests = transformed_row_tests(prepared, layout, public)?;
    let domain = prepared.domain();
    let fill_matrix = |workspace: &mut _,
                       (j, coefficients): (usize, &mut [F])|
     -> Result<(), AkitaError> {
        let mut accumulator = domain.zero_ntt();
        for (matrix_row, test) in prepared
            .matrix_ntt()
            .chunks_exact(layout.m())
            .zip(&row_tests)
        {
            accumulator
                .add_assign_pointwise_mul(matrix_row.get(j).ok_or(AkitaError::InvalidProof)?, test);
        }
        let product = domain.inverse_with_workspace(&accumulator, workspace);
        trace_gram_into::<F, M>(
            product.coefficients(),
            coefficients.get_mut(..D).ok_or(AkitaError::InvalidProof)?,
        )
    };
    #[cfg(feature = "parallel")]
    weights
        .par_chunks_mut(layout.padded_coefficients())
        .enumerate()
        .try_for_each_init(|| domain.workspace(), fill_matrix)?;
    #[cfg(not(feature = "parallel"))]
    {
        let mut workspace = domain.workspace();
        for item in weights.chunks_mut(layout.padded_coefficients()).enumerate() {
            fill_matrix(&mut workspace, item)?;
        }
    }
    drop(span);
    let add_parity = |(j, coefficients): (usize, &mut [F])| -> Result<(), AkitaError> {
        for (t, weight) in coefficients.iter_mut().take(D).enumerate() {
            *weight += public.parity_coefficient_weight(j, t)?;
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    weights
        .par_chunks_mut(layout.padded_coefficients())
        .enumerate()
        .try_for_each(add_parity)?;
    #[cfg(not(feature = "parallel"))]
    weights
        .chunks_mut(layout.padded_coefficients())
        .enumerate()
        .try_for_each(add_parity)?;
    Ok(weights)
}

/// Build image weights by own-ring products followed by the shared trace map.
///
/// `public` must use `layout`. Each challenge is transformed once per column;
/// each image row receives `-G(iota(Ch_col) * G^-1 u_i)`. Coefficient tails and
/// entry padding stay zero. This is the optimized counterpart of the verifier's
/// independent remainder-power dense definition.
pub fn image_weights<F, const D: usize, M>(
    prepared: &PreparedCommitMatrix<F, D, M>,
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
) -> Result<Vec<F>, AkitaError>
where
    F: SmoothFftField,
    M: TrinomialModulus + Send + Sync,
{
    public.validate_layout(layout)?;
    if layout.degree() != D || public.embedded_challenge_coefficients().len() != layout.columns() {
        return Err(AkitaError::InvalidInput(
            "invalid image-weight geometry".into(),
        ));
    }
    let mut weights = zero_weights(layout.image_len())?;
    let row_tests = transformed_row_tests(prepared, layout, public)?;
    let domain = prepared.domain();
    let column_len = checked::product([layout.n_a(), layout.padded_coefficients()])
        .ok_or(AkitaError::InvalidProof)?;
    let live_len =
        checked::product([layout.columns(), column_len]).ok_or(AkitaError::InvalidProof)?;
    let live = weights
        .get_mut(..live_len)
        .ok_or(AkitaError::InvalidProof)?;
    let fill_column =
        |workspace: &mut _, (column, output): (usize, &mut [F])| -> Result<(), AkitaError> {
            let coefficients: [F; D] = public
                .embedded_challenge_coefficients()
                .get(column)
                .ok_or(AkitaError::InvalidProof)?
                .as_slice()
                .try_into()
                .map_err(|_| AkitaError::InvalidProof)?;
            let challenge = TrinomialRing::from_coefficients(coefficients)
                .map_err(|_| AkitaError::InvalidProof)?;
            let challenge = domain.forward_with_workspace(&challenge, workspace);
            for (entry, row_test) in output
                .chunks_exact_mut(layout.padded_coefficients())
                .zip(&row_tests)
            {
                let product =
                    domain.inverse_with_workspace(&challenge.pointwise_mul(row_test), workspace);
                let live = entry.get_mut(..D).ok_or(AkitaError::InvalidProof)?;
                trace_gram_into::<F, M>(product.coefficients(), live)?;
                for coefficient in live {
                    *coefficient = -*coefficient;
                }
            }
            Ok(())
        };
    #[cfg(feature = "parallel")]
    live.par_chunks_mut(column_len)
        .enumerate()
        .try_for_each_init(|| domain.workspace(), fill_column)?;
    #[cfg(not(feature = "parallel"))]
    {
        let mut workspace = domain.workspace();
        for item in live.chunks_mut(column_len).enumerate() {
            fill_column(&mut workspace, item)?;
        }
    }
    Ok(weights)
}

fn transformed_row_tests<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    prepared: &PreparedCommitMatrix<F, D, M>,
    layout: &LoweredRootLayout,
    public: &LoweredPublic<F>,
) -> Result<Vec<TrinomialNtt<F, D, M>>, AkitaError> {
    let inverse = trace_gram_inverse::<F, M>(public.alpha_powers())?;
    let mut transformed = Vec::new();
    transformed
        .try_reserve_exact(layout.n_a())
        .map_err(|_| AkitaError::InvalidProof)?;
    let mut workspace = prepared.domain().workspace();
    for &gamma in public
        .gamma_powers()
        .get(..layout.n_a())
        .ok_or(AkitaError::InvalidProof)?
    {
        let mut coefficients: [F; D] = inverse
            .as_slice()
            .try_into()
            .map_err(|_| AkitaError::InvalidProof)?;
        for coefficient in &mut coefficients {
            *coefficient *= gamma;
        }
        let test =
            TrinomialRing::from_coefficients(coefficients).map_err(|_| AkitaError::InvalidProof)?;
        transformed.push(
            prepared
                .domain()
                .forward_with_workspace(&test, &mut workspace),
        );
    }
    Ok(transformed)
}

fn zero_weights<F: SmoothFftField>(len: usize) -> Result<Vec<F>, AkitaError> {
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidInput("coefficient-weight allocation failed".into()))?;
    weights.resize(len, F::zero());
    Ok(weights)
}
