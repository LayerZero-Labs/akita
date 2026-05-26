//! Error types for Metal setup, transfer, and dispatch.

use thiserror::Error;

/// Result alias for the Metal accelerator crate.
pub type MetalResult<T> = Result<T, MetalError>;

/// Errors raised before or during Metal execution.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MetalError {
    /// Metal is unavailable on the current target platform.
    #[error("Metal backend is only available on macOS")]
    UnsupportedPlatform,
    /// The host platform supports Metal, but no default device was found.
    #[error("no default Metal device is available")]
    NoSystemDevice,
    /// The crate has not yet wired the requested runtime facility.
    #[error("Metal device/runtime support is not wired yet: {0}")]
    RuntimeUnavailable(&'static str),
    /// Metal failed to compile an embedded kernel library.
    #[error("failed to compile Metal kernel library: {0}")]
    KernelLibrary(String),
    /// Metal failed to look up an embedded kernel function.
    #[error("failed to load Metal kernel `{name}`: {message}")]
    KernelFunction {
        /// Kernel function name.
        name: &'static str,
        /// Metal-reported failure message.
        message: String,
    },
    /// Metal failed to build a compute pipeline.
    #[error("failed to build Metal pipeline `{name}`: {message}")]
    Pipeline {
        /// Kernel function name.
        name: &'static str,
        /// Metal-reported failure message.
        message: String,
    },
    /// Metal command execution did not complete successfully.
    #[error("Metal command buffer failed while running {0}")]
    CommandFailed(&'static str),
    /// The host attempted to build an invalid kernel dispatch.
    #[error("invalid Metal kernel input: {0}")]
    InvalidInput(&'static str),
}
