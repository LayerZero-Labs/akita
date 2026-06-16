//! The outer `B` / tiered `B'`+`F` commit — the single definition shared by
//! root single-tier, root tiered, and recursive `commit_w`.

use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore};
use akita_types::LevelParams;

use crate::commit::ajtai::backend::CommitBackend;
use crate::commit::ajtai::opening::AjtaiOpeningType;
use crate::commit::ajtai::spec::{MatrixRole, MatrixSpec, RingDomain};
use crate::commit::decompose::decompose_rows;

/// Outer commit: `u = B · t̂` (single-tier) or
/// `u = F · decompose(blockdiag(B') · t̂)` (tiered).
///
/// Returns the unblinded sent commitment image. ZK blinding of the single-tier
/// `B` image is added by the caller (the pipeline owns the blinding hint).
///
/// # Errors
///
/// Returns an error if a window is malformed or any matvec fails.
pub(crate) fn outer_commit<F, const D: usize, B>(
    backend: &B,
    commitment_key: &B::PreparedSetup<D>,
    params: &LevelParams,
    t_hat: &[[i8; D]],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    B: CommitBackend<F>,
{
    match &params.f_key {
        None => {
            let matrix = MatrixSpec {
                role: MatrixRole::BOuter,
                rows: params.b_key.row_len(),
                cols: t_hat.len(),
                domain: RingDomain::Negacyclic,
            };
            let u = backend
                .ajtai_commit::<D>(
                    commitment_key,
                    matrix,
                    AjtaiOpeningType::DigitVector {
                        digits: t_hat,
                        log_basis: params.log_basis,
                    },
                )?
                .into_iter()
                .next()
                .unwrap_or_default();
            if u.len() != params.b_key.row_len() {
                return Err(AkitaError::InvalidSetup(format!(
                    "backend returned {} B commitment rows, expected {}",
                    u.len(),
                    params.b_key.row_len()
                )));
            }
            Ok(u)
        }
        Some(_) => tiered_outer_commit(backend, commitment_key, params, t_hat),
    }
}

/// Tiered second-tier commitment: `u_final = F · decompose(blockdiag(B') · t̂)`.
fn tiered_outer_commit<F, const D: usize, B>(
    backend: &B,
    commitment_key: &B::PreparedSetup<D>,
    params: &LevelParams,
    t_hat: &[[i8; D]],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
    B: CommitBackend<F>,
{
    let f_key = params.f_key.as_ref().ok_or_else(|| {
        AkitaError::InvalidSetup("tiered_outer_commit requires a second-tier F key".to_string())
    })?;
    let width_small = params.b_key.col_len();
    if width_small == 0 || !t_hat.len().is_multiple_of(width_small) {
        return Err(AkitaError::InvalidSetup(
            "tiered commit: first-tier B' width does not divide the opening input".to_string(),
        ));
    }
    let bp = MatrixSpec {
        role: MatrixRole::BOuterTierSlice,
        rows: params.b_key.row_len(),
        cols: width_small,
        domain: RingDomain::Negacyclic,
    };
    // u_concat = (B'·t̂_slice_0 ‖ … ‖ B'·t̂_slice_{f-1}), negacyclic.
    let mut u_concat: Vec<CyclotomicRing<F, D>> = Vec::new();
    for chunk in t_hat.chunks(width_small) {
        u_concat.extend(
            backend
                .ajtai_commit::<D>(
                    commitment_key,
                    bp,
                    AjtaiOpeningType::DigitVector {
                        digits: chunk,
                        log_basis: params.log_basis,
                    },
                )?
                .into_iter()
                .flatten(),
        );
    }
    // û_concat = decompose(u_concat) at the opening digit depth, ordered
    // [slice][b'_row][digit].
    let u_hat = decompose_rows(
        std::slice::from_ref(&u_concat),
        params.num_digits_open,
        params.log_basis,
    )?;
    let f = MatrixSpec {
        role: MatrixRole::FOuterTier,
        rows: f_key.row_len(),
        cols: u_hat.flat_digits().len(),
        domain: RingDomain::Negacyclic,
    };
    let u_final = backend
        .ajtai_commit::<D>(
            commitment_key,
            f,
            AjtaiOpeningType::DigitVector {
                digits: u_hat.flat_digits(),
                log_basis: params.log_basis,
            },
        )?
        .into_iter()
        .next()
        .unwrap_or_default();
    if u_final.len() != f_key.row_len() {
        return Err(AkitaError::InvalidProof);
    }
    Ok(u_final)
}
