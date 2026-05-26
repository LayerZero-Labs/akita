//! Apple Metal kernels and runtime helpers for Akita compute backends.
//!
//! This crate is intentionally kept outside the foundational algebra, prover,
//! and verifier crates. Device code should remain an optional accelerator layer,
//! with explicit host-side transfer and dispatch boundaries.

pub mod device;
pub mod error;
pub mod field;
pub mod kernels;

pub use device::{
    Fp128BufferOptions, Fp128BufferStorageMode, Fp128DispatchOptions, Fp128DispatchProfile,
    Fp128PipelineInfo, Fp128TransferProfile, Fp128VectorBuffers, MetalBackend, MetalDeviceInfo,
};
pub use error::{MetalError, MetalResult};
