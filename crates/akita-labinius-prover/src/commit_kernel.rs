//! Prepared transform-domain commitment to packed binary source columns.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable,
    clippy::unwrap_used
)]

use akita_algebra::{
    binary::field_switch::{embed_source, SwitchField},
    SmoothFftField, TrinomialI8Lut, TrinomialModulus, TrinomialNtt, TrinomialNttDomain,
    TrinomialRing,
};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{
    commitment::reduce_commitment_image, BinaryClearCommitment, BinaryClearSetup,
};
use jolt_field::WithPacking;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A reusable matrix transform and signed-digit table for an explicit setup.
///
/// Matrix entries retain row-major order. Preparation is independent of the
/// source, host field, and number of columns; the matrix digest prevents use
/// with a different matrix. The reference commitment path remains unchanged.
pub struct PreparedCommitMatrix<F, const D: usize, M: TrinomialModulus> {
    domain: TrinomialNttDomain<F, D, M>,
    matrix: Vec<TrinomialNtt<F, D, M>>,
    lut: TrinomialI8Lut<F, D, M>,
    n_a: usize,
    m: usize,
    digest: [u8; 32],
    matrix_bytes: usize,
    prepared_bytes: usize,
}

impl<F: SmoothFftField, const D: usize, M: TrinomialModulus> PreparedCommitMatrix<F, D, M> {
    /// Transform every matrix entry once and prepare the smallest signed table.
    pub fn prepare(setup: &BinaryClearSetup<F, D, M>) -> Result<Self, AkitaError> {
        let domain = TrinomialNttDomain::new()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        // log_basis=1 covers [-1, 1), excluding +1. The next basis covers
        // [-2, 2), including all three possible binary packing coefficients.
        let lut = domain
            .prepare_i8_lut(2)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let matrix_bytes = checked::product([setup.matrix().len(), D, size_of::<F>()])
            .ok_or_else(|| AkitaError::InvalidSetup("prepared matrix size overflow".into()))?;
        let prepared_bytes = checked::sum([matrix_bytes, lut.table_bytes()])
            .ok_or_else(|| AkitaError::InvalidSetup("prepared payload size overflow".into()))?;
        let mut matrix = Vec::new();
        matrix
            .try_reserve_exact(setup.matrix().len())
            .map_err(|_| AkitaError::InvalidSetup("prepared matrix allocation failed".into()))?;
        let mut workspace = domain.workspace();
        for element in setup.matrix() {
            matrix.push(domain.forward_with_workspace(element, &mut workspace));
        }
        Ok(Self {
            domain,
            matrix,
            lut,
            n_a: setup.n_a(),
            m: setup.m(),
            digest: *setup.matrix_view_digest(),
            matrix_bytes,
            prepared_bytes,
        })
    }

    /// Bytes of transformed matrix field coefficients, excluding Vec metadata.
    pub fn matrix_bytes(&self) -> usize {
        self.matrix_bytes
    }

    /// Bytes of the two position-aware signed-digit tables.
    pub fn table_bytes(&self) -> usize {
        self.lut.table_bytes()
    }

    /// Matrix plus table payload bytes, excluding transform-plan and metadata.
    ///
    /// The algebra domain does not expose its own heap storage size. This count
    /// excludes that fixed-degree plan and temporary per-column workspaces.
    pub fn prepared_bytes(&self) -> usize {
        self.prepared_bytes
    }

    /// Borrow the reusable canonical transform plan.
    pub fn domain(&self) -> &TrinomialNttDomain<F, D, M> {
        &self.domain
    }

    /// Borrow transformed matrix entries in row-major order.
    pub fn matrix_ntt(&self) -> &[TrinomialNtt<F, D, M>] {
        &self.matrix
    }

    /// Borrow the signed-digit table covering `[-2, 2)`.
    pub fn i8_lut(&self) -> &TrinomialI8Lut<F, D, M> {
        &self.lut
    }

    /// Reject a setup whose matrix shape or digest differs from the cached one.
    pub(crate) fn check_setup(&self, setup: &BinaryClearSetup<F, D, M>) -> Result<(), AkitaError> {
        if self.n_a != setup.n_a()
            || self.m != setup.m()
            || self.digest != *setup.matrix_view_digest()
        {
            return Err(AkitaError::InvalidSetup(
                "prepared commitment matrix does not match setup".into(),
            ));
        }
        Ok(())
    }
}

/// Check the common packed-source length contract before allocation.
pub(crate) fn check_source_len(expected: usize, actual: usize) -> Result<(), AkitaError> {
    if actual != expected {
        return Err(AkitaError::InvalidSize { expected, actual });
    }
    Ok(())
}

/// Admit the common signed-interleaving geometry for one source element.
pub(crate) fn source_element_rank<const D: usize>(words: usize) -> Result<usize, AkitaError> {
    checked::exact_div(D, 162)
        .filter(|k| matches!(k, 1 | 2 | 4) && words == *k)
        .ok_or_else(|| AkitaError::InvalidInput("source element geometry mismatch".into()))
}

/// Pack exactly one group of `k` source words directly into signed coefficients.
///
/// `D = 162 * k`, with `k` in `{1, 2, 4}`. Coefficient `s * k + c` is
/// coordinate `s` of `embed_source::<H>(words[c])`, negated for odd `s` when
/// `k > 1`, matching `pack_scalar_components`' signed interleaving.
pub fn pack_binary_element_i8<H: SwitchField, const D: usize>(
    words: &[H::Source],
) -> Result<[i8; D], AkitaError> {
    let k = source_element_rank::<D>(words.len())?;
    let mut coefficients = [0i8; D];
    for (component, &word) in words.iter().enumerate() {
        let bytes = embed_source::<H>(word).to_bytes();
        for (scalar, row) in coefficients.chunks_exact_mut(k).enumerate() {
            let byte = bytes.get(scalar / 8).ok_or(AkitaError::InvalidProof)?;
            let bit = ((byte >> (scalar % 8)) & 1) as i8;
            *row.get_mut(component).ok_or(AkitaError::InvalidProof)? =
                if k == 1 || scalar % 2 == 0 { bit } else { -bit };
        }
    }
    Ok(coefficients)
}

/// Commit binary columns with a cached matrix, one source transform per element,
/// and one inverse transform per output row.
///
/// Output order and source-length errors match `commit_binary_clear`. A cache
/// from another matrix returns `InvalidSetup`. With `parallel`, columns use
/// independent workspaces and write disjoint output slices in canonical order.
pub fn commit_binary_clear_prepared<H, F, const D: usize, M>(
    prepared: &PreparedCommitMatrix<F, D, M>,
    setup: &BinaryClearSetup<F, D, M>,
    source: &[H::Source],
) -> Result<BinaryClearCommitment<F, D, M>, AkitaError>
where
    H: SwitchField,
    H::Source: Sync,
    F: SmoothFftField + WithPacking,
    M: TrinomialModulus + Send + Sync,
{
    check_source_len(setup.source_len(), source.len())?;
    prepared.check_setup(setup)?;
    let image_count = checked::product([setup.columns(), setup.n_a()])
        .ok_or_else(|| AkitaError::InvalidSetup("commitment size overflow".into()))?;
    let mut images = Vec::new();
    images
        .try_reserve_exact(image_count)
        .map_err(|_| AkitaError::InvalidInput("matrix image allocation failed".into()))?;
    let zero =
        TrinomialRing::zero().map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
    images.resize(image_count, zero);

    #[cfg(feature = "parallel")]
    images
        .par_chunks_mut(setup.n_a())
        .zip(source.par_chunks(setup.scalar_rows()))
        .try_for_each(|(output, words)| commit_column::<H, F, D, M>(prepared, words, output))?;
    #[cfg(not(feature = "parallel"))]
    for (output, words) in images
        .chunks_mut(setup.n_a())
        .zip(source.chunks(setup.scalar_rows()))
    {
        commit_column::<H, F, D, M>(prepared, words, output)?;
    }
    reduce_commitment_image(setup, &mut images)?;
    Ok(BinaryClearCommitment { images })
}

fn commit_column<H, F, const D: usize, M>(
    prepared: &PreparedCommitMatrix<F, D, M>,
    words: &[H::Source],
    output: &mut [TrinomialRing<F, D, M>],
) -> Result<(), AkitaError>
where
    H: SwitchField,
    F: SmoothFftField + WithPacking,
    M: TrinomialModulus,
{
    let mut workspace = prepared.domain.workspace();
    let mut transformed = prepared.domain.zero_ntt();
    let mut accumulators = Vec::new();
    accumulators
        .try_reserve_exact(prepared.n_a)
        .map_err(|_| AkitaError::InvalidInput("matrix accumulator allocation failed".into()))?;
    accumulators.resize(prepared.n_a, prepared.domain.zero_ntt());
    let k = checked::exact_div(D, 162).ok_or(AkitaError::InvalidProof)?;
    for (element, chunk) in words.chunks_exact(k).enumerate() {
        let digits = pack_binary_element_i8::<H, D>(chunk)?;
        prepared
            .domain
            .forward_i8_with_lut_into_workspace(
                &digits,
                &prepared.lut,
                &mut transformed,
                &mut workspace,
            )
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        for (accumulator, row) in accumulators
            .iter_mut()
            .zip(prepared.matrix.chunks_exact(prepared.m))
        {
            let matrix_element = row.get(element).ok_or(AkitaError::InvalidProof)?;
            accumulator.add_assign_pointwise_mul_packed(matrix_element, &transformed);
        }
    }
    for (image, accumulator) in output.iter_mut().zip(&accumulators) {
        *image = prepared
            .domain
            .inverse_with_workspace(accumulator, &mut workspace);
    }
    Ok(())
}
