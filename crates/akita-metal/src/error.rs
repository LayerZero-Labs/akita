use jolt_metal::{ErrorClass, MetalError};

/// A failure of an Akita Metal operation.
///
/// [`AkitaMetalError::class`] tells the caller whether it may fall back to
/// the CPU backend, retry, or must stop.
#[derive(Debug, thiserror::Error)]
pub enum AkitaMetalError {
    /// The device runtime failed; see [`MetalError::class`].
    #[error(transparent)]
    Metal(#[from] MetalError),
    /// The operation does not support the requested shape. Checked before
    /// anything is encoded, so the CPU path can take over.
    #[error("unsupported shape: {0}")]
    Shape(String),
}

impl AkitaMetalError {
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Metal(error) => error.class(),
            Self::Shape(_) => ErrorClass::Setup,
        }
    }
}
