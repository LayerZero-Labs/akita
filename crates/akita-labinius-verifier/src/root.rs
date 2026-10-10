//! Root reduction to authenticated evaluations of the committed tables.

mod auxiliary;
mod oracle;
mod verify;
mod wire;

pub use auxiliary::exchange_root_auxiliary;
pub use oracle::{
    bind_transparent_image, check_transparent_evaluations, exchange_transparent_prime_opening,
    RootEvaluationClaims, RootProverOracle, RootVerifierOracle, TransparentRootVerifierOracle,
};
pub use verify::{verify_root_reduction, verify_root_reduction_bytes};
pub use wire::root_reduction_wire_size;

use crate::{
    channel::RootChallengeChannel, codec, grinding::RootGrindingPlan, lowered::LoweredRootLayout,
    statement::RootStatement, AdmittedRootSetup,
};
use akita_algebra::{binary::field_switch::SwitchField, TrinomialModulus};
use akita_error::{checked, AkitaError};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// The bytes that name the field pair `(F, E)` in the root statement.
///
/// 1. The characteristic of `F` as sixteen little-endian bytes.
/// 2. The extension degree `d = E::DEGREE` as four little-endian bytes.
/// 3. The multiplication table of the coordinate basis `b_0, ..., b_(d-1)` of
///    `E` over `F`, where `b_i` has base coordinates `e_i`: for each pair
///    `i <= j` in lexicographic order, the `d` base coordinates of `b_i * b_j`,
///    each a canonical little-endian element of `F`.
///
/// The table determines `E` as an `F`-algebra together with the basis in which
/// challenges are sampled and messages are encoded, so two extensions of one
/// degree that differ in their defining polynomial have different bytes. For
/// `E = F` it is the single element one.
fn field_pair_identity<F, E>() -> Result<Vec<u8>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let invalid = || AkitaError::InvalidSetup("root field pair identity overflow".into());
    let degree = E::DEGREE;
    let products = checked::sum([degree, 1])
        .and_then(|next| checked::product([degree, next]))
        .ok_or_else(invalid)?
        / 2;
    let len = checked::product([products, degree, F::NUM_BYTES])
        .and_then(|table| checked::sum([16, 4, table]))
        .ok_or_else(invalid)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|_| invalid())?;
    bytes.extend_from_slice(&crate::admitted::field_characteristic::<F>()?.to_le_bytes());
    bytes.extend_from_slice(&u32::try_from(degree).map_err(|_| invalid())?.to_le_bytes());
    let mut basis = Vec::new();
    basis.try_reserve_exact(degree).map_err(|_| invalid())?;
    let mut coordinates = zero_vec::<F>(degree)?;
    for index in 0..degree {
        coordinates.fill(F::zero());
        *coordinates.get_mut(index).ok_or_else(invalid)? = F::one();
        basis.push(E::from_base_slice(&coordinates));
    }
    let mut encoded = zero_vec::<u8>(F::NUM_BYTES)?;
    for (index, &left) in basis.iter().enumerate() {
        for &right in basis.iter().skip(index) {
            for coordinate in (left * right).to_base_vec() {
                coordinate.to_bytes_le(&mut encoded);
                bytes.extend_from_slice(&encoded);
            }
        }
    }
    if bytes.len() != len {
        return Err(invalid());
    }
    Ok(bytes)
}

/// Admit the field pair, then bind the admitted setup, the field pair, the
/// grinding plan, the image owner and the claims before any proof message.
///
/// `F` is the base field of the committed tables and `E` the challenge field.
/// The admitted setup's identity names neither, so the bytes documented on
/// `field_pair_identity` are absorbed here, followed by
/// [`RootGrindingPlan::canonical_bytes`]: the plan is fixed before any proof
/// message, and the channel replays it from here on.
///
/// The plan is the plan of the statement's opening mode, and the three modes
/// have three different plans, so no separate mode byte is absorbed. After
/// the image binding come the host point and value when the statement has a
/// binary claim, then the bytes documented on `PrimeClaim::statement_bytes`
/// when it has a prime claim.
/// The image binding callback is invoked exactly once after layout validation.
pub fn bind_root_statement<H, F, E, const D: usize, M, S>(
    admitted: &AdmittedRootSetup<D, M>,
    statement: &RootStatement<'_, H, E>,
    channel: &mut S,
    bind_image: impl FnOnce(&LoweredRootLayout, &mut S) -> Result<(), AkitaError>,
) -> Result<LoweredRootLayout, AkitaError>
where
    H: SwitchField,
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    M: TrinomialModulus,
    S: RootChallengeChannel<E>,
{
    let setup = admitted.setup();
    let mode = statement.mode()?;
    if let Some((point, _)) = statement.binary {
        if point.len() != setup.num_vars() {
            return Err(AkitaError::InvalidPointDimension {
                expected: setup.num_vars(),
                actual: point.len(),
            });
        }
    }
    let layout = LoweredRootLayout::new::<F, E, D, M>(setup, admitted.shape())?;
    if let Some(prime) = statement.prime {
        prime.validate(&layout)?;
    }
    let plan = RootGrindingPlan::new::<H, F, E>(admitted.shape(), layout.encoding(), mode)?;
    let mut domain = Vec::new();
    codec::length_prefixed(&mut domain, b"akita/labinius/root-reduction/v1")?;
    channel.public(&domain)?;
    channel.public(&admitted.identity_bytes::<H>()?)?;
    channel.public(&field_pair_identity::<F, E>()?)?;
    channel.public(&plan.canonical_bytes()?)?;
    bind_image(&layout, channel)?;
    if let Some((point, value)) = statement.binary {
        for host in point.iter().copied().chain(std::iter::once(value)) {
            for word in host.coordinates().iter().take(H::ROWS / 64) {
                channel.public(&word.to_le_bytes())?;
            }
        }
    }
    if let Some(prime) = statement.prime {
        channel.public(&prime.statement_bytes::<F>()?)?;
    }
    channel.schedule(plan)?;
    Ok(layout)
}

/// Fallibly initialize statement-sized storage.
pub(crate) fn zero_vec<T: Clone + Default>(len: usize) -> Result<Vec<T>, AkitaError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidInput("root allocation failed".into()))?;
    values.resize(len, T::default());
    Ok(values)
}
