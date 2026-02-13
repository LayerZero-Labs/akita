//! Backend contracts and concrete backend implementations.

pub mod scalar;
pub mod traits;

pub use scalar::ScalarBackend;
pub use traits::{CrtReconstruct, NttPrimeOps, NttTransform, RingBackend};
