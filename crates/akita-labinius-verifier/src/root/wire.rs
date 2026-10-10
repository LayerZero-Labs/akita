use akita_algebra::binary::field_switch::SwitchField;
use akita_error::{checked, AkitaError};
use akita_params::{sis::labinius::LabiniusRootShape, FOLD_RESPONSE_NONCE_BITS};
use akita_transcript::nonce_max_bytes;
use jolt_field::{CanonicalEncoding, ExtField, Field};

use crate::{
    grinding::RootGrindingPlan,
    root_sumcheck::{combined_shape, product_shape},
    statement::RootOpeningMode,
};

/// Largest root proof in `mode` over the field pair `(F, E)` in bytes,
/// excluding bytes emitted by the commitment oracle.
///
/// Every message has a fixed width except the nonces: the fold-response nonce
/// and one proof-of-work nonce per site of the grinding plan with a nonzero
/// target. Each is counted here at its canonical LEB128 maximum; a proof is
/// shorter by the bytes its nonces do not use.
/// A challenge-field element takes `E::DEGREE` canonical elements of `F`.
///
/// The frontend, the binary left opening and the parity integers `Q`, `K` are
/// present exactly with a binary claim. The row claim `y_P`, the product
/// instance and its table evaluation are present exactly with a prime claim;
/// the prime left opening itself is the oracle's.
/// Derivation needs only the admitted shape, the field pair and the sealed
/// host field. A proof prime the shape does not admit is an error.
pub fn root_reduction_wire_size<H, F, E>(
    shape: &LabiniusRootShape,
    mode: RootOpeningMode,
) -> Result<usize, AkitaError>
where
    H: SwitchField,
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let invalid = || AkitaError::InvalidSetup("root wire size overflow".into());
    let element = checked::product([E::DEGREE, F::NUM_BYTES]).ok_or_else(invalid)?;
    let encoding = shape.derive_encoding(crate::admitted::field_characteristic::<F>()?)?;
    let clear_integers = |len: usize, bits: u32| {
        checked::div_ceil(bits as usize, 8).and_then(|width| checked::product([len, width]))
    };
    let binary = if mode.has_binary() {
        let n = checked::ceil_log2(shape.num_cells()).ok_or_else(invalid)?;
        let frontend =
            checked::product([H::ROWS, core::mem::size_of::<H::Source>()]).ok_or_else(invalid)?;
        let frontend_rounds = checked::mul_add(n, 2, 1)
            .and_then(|count| checked::product([count, 21]))
            .ok_or_else(invalid)?;
        let u = checked::product([21, shape.fold_width()]).ok_or_else(invalid)?;
        let q = clear_integers(encoding.parity_quotient_len(), encoding.quotient().bits())
            .ok_or_else(invalid)?;
        let k = clear_integers(encoding.parity_carry_len(), encoding.carry().bits())
            .ok_or_else(invalid)?;
        checked::sum([frontend, frontend_rounds, u, q, k]).ok_or_else(invalid)?
    } else {
        0
    };
    let prime = if mode.has_prime() {
        let num_vars = encoding.prime_table_log_len();
        // `y_P`, the round messages and the table evaluation.
        checked::product([num_vars, product_shape(num_vars)?.degree_bound()])
            .and_then(|messages| checked::sum([messages, 2]))
            .and_then(|elements| checked::product([elements, element]))
            .ok_or_else(invalid)?
    } else {
        0
    };
    let ka =
        clear_integers(encoding.a_carry_len(), encoding.a_carry().bits()).ok_or_else(invalid)?;
    let combined_degree = combined_shape(encoding.response_table_log_len())?.degree_bound();
    let combined = checked::product([encoding.response_table_log_len(), combined_degree, element])
        .ok_or_else(invalid)?;
    let image_degree = combined_shape(encoding.image_table_log_len())?.degree_bound();
    let image = checked::product([encoding.image_table_log_len(), image_degree, element])
        .ok_or_else(invalid)?;
    // `y_Y` and the two digit-table evaluations.
    let evaluations = checked::product([3, element]).ok_or_else(invalid)?;
    let grinding = RootGrindingPlan::new::<H, F, E>(shape, &encoding, mode)?.nonce_max_bytes()?;
    checked::sum([
        binary,
        nonce_max_bytes(FOLD_RESPONSE_NONCE_BITS),
        grinding,
        ka,
        evaluations,
        combined,
        image,
        prime,
    ])
    .ok_or_else(invalid)
}
