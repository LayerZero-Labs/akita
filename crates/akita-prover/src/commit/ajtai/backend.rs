//! The Ajtai commit primitive trait.
//!
//! `ajtai_commit(commitment_key, spec, opening) -> commitment` is one Ajtai
//! commitment (one matrix multiply). Every `A`/`B`/`B'`/`F` commit in the
//! scheme flows through it. It extends [`DigitRowsComputeBackend`] so the
//! commitment-key contract (`PreparedSetup<D>`, `prepare_expanded`,
//! `validate_prepared_setup`) and the dedicated ZK blinding mat-vecs
//! (`zk_b_digit_rows` / `zk_d_digit_rows`) are reused as-is.

use crate::commit::ajtai::opening::AjtaiOpeningType;
use crate::commit::ajtai::spec::MatrixSpec;
use crate::compute::DigitRowsComputeBackend;
use akita_algebra::CyclotomicRing;
use akita_field::unreduced::{HasWide, ReduceTo};
use akita_field::{AdditiveGroup, AkitaError, CanonicalField, FieldCore};

/// The single Ajtai commit primitive: `commitment = commitment_key · opening`.
pub trait CommitBackend<F>: DigitRowsComputeBackend<F>
where
    F: FieldCore + CanonicalField,
{
    /// `commitment = commitment_key · opening`, under matrix window `spec`.
    ///
    /// `out.len()` is the number of opening blocks and `out[b].len() ==
    /// spec.rows`. The opening is matched once, validated against `spec`, and
    /// dispatched to a concrete kernel. No per-element dispatch.
    ///
    /// # Errors
    ///
    /// Returns an error if the window is malformed (zero `rows`/`cols`,
    /// footprint exceeds the commitment key, per-block width mismatch, or
    /// out-of-range `log_basis`) or if a kernel fails. Never panics on
    /// verifier-style malformed input.
    fn ajtai_commit<const D: usize>(
        &self,
        commitment_key: &Self::PreparedSetup<D>,
        spec: MatrixSpec,
        opening: AjtaiOpeningType<'_, F, D>,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
    where
        F: HasWide,
        F::Wide: AdditiveGroup + From<F> + ReduceTo<F>;
}
