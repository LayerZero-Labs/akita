use akita_algebra::binary::field_switch::SwitchField;
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::{LabiniusDigitBase, LabiniusRootShape};

/// Exact root proof bytes, excluding bytes emitted by the commitment oracle.
/// Derivation needs only the admitted shape, digit base, and sealed host field.
pub fn root_reduction_wire_size<H: SwitchField>(
    shape: &LabiniusRootShape,
    base: LabiniusDigitBase,
) -> Result<usize, AkitaError> {
    let invalid = || AkitaError::InvalidSetup("root wire size overflow".into());
    let encoding = shape.derive_encoding(base)?;
    let n = checked::ceil_log2(shape.num_cells()).ok_or_else(invalid)?;
    let coefficient_bits = checked::sum([
        shape.profile().coefficient_prime().modulus().ilog2() as usize,
        1,
    ])
    .ok_or_else(invalid)?;
    let coefficient_bytes = checked::div_ceil(coefficient_bits, 8).ok_or_else(invalid)?;
    let quotient_degree = shape
        .commitment_degree()
        .checked_sub(1)
        .ok_or_else(invalid)?;
    let rank = usize::try_from(shape.rank_a()).map_err(|_| invalid())?;
    let frontend =
        checked::product([H::ROWS, core::mem::size_of::<H::Source>()]).ok_or_else(invalid)?;
    let frontend_rounds = checked::mul_add(n, 2, 1)
        .and_then(|count| checked::product([count, 21]))
        .ok_or_else(invalid)?;
    let u = checked::product([21, shape.fold_width()]).ok_or_else(invalid)?;
    let qa = checked::product([rank, quotient_degree, coefficient_bytes]).ok_or_else(invalid)?;
    let q = checked::div_ceil(encoding.quotient().bits() as usize, 8)
        .and_then(|width| checked::product([encoding.parity_quotient_len(), width]))
        .ok_or_else(invalid)?;
    let k = checked::div_ceil(encoding.carry().bits() as usize, 8)
        .and_then(|width| checked::product([encoding.parity_carry_len(), width]))
        .ok_or_else(invalid)?;
    let degree = checked::pow2(base.bits() as usize)
        .and_then(|size| checked::sum([size, 1]))
        .ok_or_else(invalid)?;
    let combined = checked::product([encoding.response_table_log_len(), degree, coefficient_bytes])
        .ok_or_else(invalid)?;
    let product = checked::product([encoding.image_table_log_len(), 2, coefficient_bytes])
        .ok_or_else(invalid)?;
    let evaluations = checked::product([3, coefficient_bytes]).ok_or_else(invalid)?;
    checked::sum([
        frontend,
        frontend_rounds,
        u,
        qa,
        q,
        k,
        evaluations,
        combined,
        product,
    ])
    .ok_or_else(invalid)
}
