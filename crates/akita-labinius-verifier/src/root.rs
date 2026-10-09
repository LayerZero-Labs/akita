//! Root reduction to authenticated response and image table evaluations.

mod auxiliary;
mod oracle;
mod verify;
mod wire;

pub use auxiliary::exchange_root_auxiliary;
pub use oracle::{
    bind_transparent_image, check_transparent_evaluations, RootEvaluationClaims, RootProverOracle,
    RootVerifierOracle, TransparentRootVerifierOracle,
};
pub use verify::{verify_root_reduction, verify_root_reduction_bytes};
pub use wire::root_reduction_wire_size;

use crate::{channel::ClearChannel, codec, lowered::LoweredRootLayout, AdmittedRootSetup};
use akita_algebra::{binary::field_switch::SwitchField, SmoothFftField, TrinomialModulus};
use akita_error::AkitaError;
use akita_params::sis::labinius::LabiniusDigitBase;

/// Bind the admitted setup, base, image owner and host claim before the frontend.
/// The image binding callback is invoked exactly once after layout validation.
pub fn bind_root_statement<H, F, const D: usize, M, S>(
    admitted: &AdmittedRootSetup<F, D, M>,
    base: LabiniusDigitBase,
    point: &[H],
    value: H,
    channel: &mut S,
    bind_image: impl FnOnce(&LoweredRootLayout, &mut S) -> Result<(), AkitaError>,
) -> Result<LoweredRootLayout, AkitaError>
where
    H: SwitchField,
    F: SmoothFftField,
    M: TrinomialModulus,
    S: ClearChannel,
{
    let setup = admitted.setup();
    if point.len() != setup.num_vars() {
        return Err(AkitaError::InvalidPointDimension {
            expected: setup.num_vars(),
            actual: point.len(),
        });
    }
    let layout = LoweredRootLayout::new(setup, admitted.shape(), base)?;
    let mut domain = Vec::new();
    codec::length_prefixed(&mut domain, b"akita/labinius/root-reduction/v2")?;
    channel.public(&domain)?;
    channel.public(&admitted.identity_bytes::<H>()?)?;
    channel.public(&[base.tag()])?;
    bind_image(&layout, channel)?;
    for host in point.iter().copied().chain(std::iter::once(value)) {
        for word in host.coordinates().iter().take(H::ROWS / 64) {
            channel.public(&word.to_le_bytes())?;
        }
    }
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
