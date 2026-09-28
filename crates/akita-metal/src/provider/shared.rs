//! Thread-shareable handles to the device objects the provider keeps.
//!
//! The commitment executor requires `Send + Sync` stage operations and
//! resident state, while `jolt-metal`'s device, library and buffer types do
//! not declare either. [`Shared`] asserts both for exactly the objects the
//! provider keeps across calls.

use jolt_metal::runtime::DeviceBuffer;

use crate::library::AkitaMetal;
use crate::matvec::DeviceNttMatrix;
use crate::onehot::DeviceFlatMatrix;

/// A device object the provider keeps across calls and threads.
pub(super) struct Shared<T>(T);

impl<T: DeviceObject> Shared<T> {
    pub(super) fn new(value: T) -> Self {
        Self(value)
    }

    pub(super) fn get(&self) -> &T {
        &self.0
    }
}

/// Objects [`Shared`] may hold: a device with its compiled library, and
/// buffers (or matrices of buffers) that GPU work only reads once they are
/// shared.
pub(super) trait DeviceObject {}

impl DeviceObject for AkitaMetal {}
impl<T: Send + Sync> DeviceObject for DeviceBuffer<T> {}
impl<const K: usize, const D: usize> DeviceObject for DeviceNttMatrix<K, D> {}
impl<F: Send + Sync> DeviceObject for DeviceFlatMatrix<F> {}

// SAFETY: every `DeviceObject` holds only
// - retained `MTLDevice`, `MTLCommandQueue`, `MTLComputePipelineState` and
//   `MTLBuffer` objects, which Metal documents as thread-safe and whose
//   retain counts objc2 updates atomically;
// - a buffer's cached `contents` pointer, which `jolt-metal` dereferences only
//   through `DeviceBuffer::read(&mut self)`; a `Shared` hands out `&T` only, so
//   no host view of a shared buffer exists;
// - plain host data (names, limits, reflection, prime tables).
// Command buffers and encoders are created per `Batch` on the calling thread
// and never stored. The provider binds shared buffers only as kernel inputs
// (the setup matrices after their preparation, and the inner rows after the
// inner stage), so concurrent batches on several threads never write one.
unsafe impl<T: DeviceObject> Send for Shared<T> {}
// SAFETY: as for `Send` above.
unsafe impl<T: DeviceObject> Sync for Shared<T> {}
