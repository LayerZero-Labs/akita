//! Shared inner commit: `t = A·s ; t̂ = decompose(t)` for one witness.
//!
//! This replaces every per-representation `commit_inner` impl. The
//! representation only provides its [`AjtaiOpeningView`]; this helper owns the
//! `A` commit, the opening decomposition, and the shape validation.

use akita_algebra::CyclotomicRing;
use akita_field::parallel::*;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore};
use akita_types::{FlatDigitBlocks, LevelParams};

use crate::commit::ajtai::backend::CommitBackend;
use crate::commit::ajtai::spec::{MatrixRole, MatrixSpec, RingDomain};
use crate::commit::decompose::decompose_rows;
use crate::commit::opening_view::AjtaiOpeningView;
use crate::commit::pipeline::{
    checked_commit_b_input_len, commit_inner_flat_digit_count, validate_commit_inner_shape,
};
use crate::CommitInnerWitness;

/// Shared `t = A·s ; t̂ = decompose(t)` for one witness. `num_blocks == 1` is
/// the recursive/single case.
///
/// # Errors
///
/// Returns an error if the `A` width overflows, the inner commit fails, or the
/// resulting witness shape is invalid.
pub(crate) fn commit_inner_one<F, const D: usize, P, B>(
    poly: &P,
    backend: &B,
    commitment_key: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<CommitInnerWitness<F, D>, AkitaError>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    P: AjtaiOpeningView<F, D>,
    B: CommitBackend<F>,
{
    let a_cols = params
        .block_len
        .checked_mul(params.num_digits_commit)
        .ok_or_else(|| AkitaError::InvalidSetup("inner A commit width overflow".to_string()))?;
    let a_matrix = MatrixSpec {
        role: MatrixRole::AInner,
        rows: params.a_key.row_len(),
        cols: a_cols,
        domain: RingDomain::Negacyclic,
    };
    let opening = poly.to_ajtai_opening(
        params.block_len,
        params.num_blocks,
        params.num_digits_commit,
        params.log_basis,
    )?;
    let t = backend.ajtai_commit::<D>(commitment_key, a_matrix, opening)?;
    let decomposed_inner_rows = decompose_rows(&t, params.num_digits_open, params.log_basis)?;
    let inner = CommitInnerWitness {
        recomposed_inner_rows: t,
        decomposed_inner_rows,
    };
    validate_commit_inner_shape(
        &inner,
        params.num_blocks,
        params.a_key.row_len(),
        params.num_digits_open,
        params.log_basis,
    )?;
    Ok(inner)
}

/// Commit a group of witnesses, concatenating each `t̂_i` into the flat outer
/// `B` input. Returns `(per-poly decomposed rows, per-poly recomposed rows,
/// concatenated B input digits)`.
///
/// # Errors
///
/// Returns an error if a length overflows or any inner commit fails.
#[allow(clippy::type_complexity)]
pub(crate) fn commit_inner_group<F, const D: usize, P, B>(
    polys: &[P],
    backend: &B,
    commitment_key: &B::PreparedSetup<D>,
    params: &LevelParams,
) -> Result<
    (
        Vec<FlatDigitBlocks<D>>,
        Vec<Vec<Vec<CyclotomicRing<F, D>>>>,
        Vec<[i8; D]>,
    ),
    AkitaError,
>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    P: AjtaiOpeningView<F, D> + Sync,
    B: CommitBackend<F>,
{
    let b_input_len_per_poly = commit_inner_flat_digit_count(
        params.num_blocks,
        params.a_key.row_len(),
        params.num_digits_open,
    )?;
    let total_b_input_len = checked_commit_b_input_len(polys.len(), b_input_len_per_poly)?;
    let mut b_input_digits = vec![[0i8; D]; total_b_input_len];
    let mut decomposed_inner_rows: Vec<FlatDigitBlocks<D>> = (0..polys.len())
        .map(|_| FlatDigitBlocks::new(Vec::new(), Vec::new()))
        .collect::<Result<_, _>>()?;
    let mut recomposed_inner_rows: Vec<Vec<Vec<CyclotomicRing<F, D>>>> =
        vec![Vec::new(); polys.len()];
    cfg_chunks_mut!(b_input_digits, b_input_len_per_poly)
        .zip(cfg_iter!(polys))
        .zip(cfg_iter_mut!(decomposed_inner_rows))
        .zip(cfg_iter_mut!(recomposed_inner_rows))
        .try_for_each(
            |(((dst, poly), decomposed), recomposed)| -> Result<(), AkitaError> {
                let inner = commit_inner_one::<F, D, P, B>(poly, backend, commitment_key, params)?;
                dst.copy_from_slice(inner.decomposed_inner_rows.flat_digits());
                *decomposed = inner.decomposed_inner_rows;
                *recomposed = inner.recomposed_inner_rows;
                Ok(())
            },
        )?;
    Ok((decomposed_inner_rows, recomposed_inner_rows, b_input_digits))
}
