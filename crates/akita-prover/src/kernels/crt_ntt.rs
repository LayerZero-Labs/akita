//! Protocol-facing CRT+NTT parameter dispatch and matrix caching.

pub use akita_algebra::ring::{
    build_ntt_slot as build_ntt_slot_flat, select_crt_ntt_params, NttSlotCache,
    ProtocolCrtNttParams,
};

use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::RingMatrixView;

/// Build an NTT slot cache for a matrix view (flat 1D storage).
///
/// # Errors
///
/// Returns an error if no CRT+NTT parameter set matches the field modulus and ring degree.
#[tracing::instrument(skip_all, name = "build_ntt_slot")]
pub fn build_ntt_slot<F: FieldCore + CanonicalField, const D: usize>(
    mat: RingMatrixView<'_, F, D>,
) -> Result<NttSlotCache<D>, AkitaError> {
    build_ntt_slot_flat(mat.coefficients(), mat.num_rows(), mat.num_cols())
}
