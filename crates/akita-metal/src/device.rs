//! Minimal device facade for the Metal backend.

use core::marker::PhantomData;

use akita_field::fields::Fp128;

use crate::error::{MetalError, MetalResult};
#[cfg(target_os = "macos")]
use crate::field::fp128::{Fp128KernelParams, Fp128Limb};
use crate::field::fp128::{Fp128MetalParams, Fp128VectorOp, Fp128VectorPlan};

/// Host-visible information about the selected Metal device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// Whether the device uses unified CPU/GPU memory.
    pub supports_unified_memory: bool,
    /// Device-reported maximum resident working set, in bytes.
    pub recommended_max_working_set_size: u64,
    /// Device-reported maximum host/device transfer rate, in bytes per second.
    pub max_transfer_rate: u64,
    /// Maximum x dimension for a threadgroup on this device.
    pub max_threads_per_threadgroup_width: u64,
    /// Whether timestamp counter sampling is available at compute stage boundaries.
    pub supports_stage_boundary_counters: bool,
}

/// Static facts about a compiled `Fp128` Metal pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp128PipelineInfo {
    /// Arithmetic operation the pipeline implements.
    pub op: Fp128VectorOp,
    /// SIMD-group width reported by Metal for this pipeline.
    pub thread_execution_width: u64,
    /// Maximum number of threads this pipeline accepts per threadgroup.
    pub max_total_threads_per_threadgroup: u64,
    /// Default x dimension selected by this crate for each threadgroup.
    pub default_threadgroup_width: u64,
}

/// Storage layout used by reusable `Fp128` vector buffers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Fp128BufferStorageMode {
    /// CPU-visible buffers are used directly by the Metal kernels.
    #[default]
    Shared,
    /// Kernels use private GPU buffers, with shared staging buffers for
    /// host/device transfers.
    PrivateStaged,
}

impl Fp128BufferStorageMode {
    /// Stable string used by probes and benchmark output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::PrivateStaged => "private_staged",
        }
    }
}

/// Allocation controls for reusable `Fp128` vector buffers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fp128BufferOptions {
    /// Storage layout for the GPU-facing buffers.
    pub storage_mode: Fp128BufferStorageMode,
}

impl Fp128BufferOptions {
    /// Set the reusable buffer storage layout.
    #[must_use]
    pub const fn with_storage_mode(mut self, storage_mode: Fp128BufferStorageMode) -> Self {
        self.storage_mode = storage_mode;
        self
    }
}

/// Optional dispatch controls for one `Fp128` vector operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Fp128DispatchOptions {
    /// Override the x dimension used for each threadgroup.
    ///
    /// `None` uses the crate default, currently `4 * thread_execution_width`
    /// capped by Metal's per-pipeline maximum.
    pub threadgroup_width: Option<u64>,
    /// Resolve Metal timestamp counters around the compute pass.
    ///
    /// This is useful telemetry but adds overhead; leave it off for
    /// accept/reject dispatch timing.
    pub sample_counters: bool,
}

impl Fp128DispatchOptions {
    /// Set the x dimension used for each threadgroup.
    #[must_use]
    pub const fn with_threadgroup_width(mut self, threadgroup_width: Option<u64>) -> Self {
        self.threadgroup_width = threadgroup_width;
        self
    }

    /// Toggle Metal timestamp-counter sampling.
    #[must_use]
    pub const fn with_sample_counters(mut self, sample_counters: bool) -> Self {
        self.sample_counters = sample_counters;
        self
    }
}

/// Low-level timing data for one host/device transfer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fp128TransferProfile {
    /// Buffer storage layout used for the transfer.
    pub storage_mode: Fp128BufferStorageMode,
    /// CPU time spent copying between Rust slices and CPU-visible Metal buffers.
    pub host_copy_ns: u128,
    /// CPU wall-clock time spent on the GPU blit command buffer, if any.
    pub gpu_blit_ns: u128,
    /// End-to-end transfer time.
    pub total_ns: u128,
}

/// Low-level timing and dispatch-shape data for one vector dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fp128DispatchProfile {
    /// Arithmetic operation that was dispatched.
    pub op: Fp128VectorOp,
    /// Number of field elements processed.
    pub len: usize,
    /// SIMD-group width reported by Metal for this pipeline.
    pub thread_execution_width: u64,
    /// Maximum number of threads this pipeline accepts per threadgroup.
    pub max_total_threads_per_threadgroup: u64,
    /// x dimension used for each threadgroup.
    pub threadgroup_width: u64,
    /// Number of threadgroups dispatched.
    pub threadgroup_count: u64,
    /// Number of compute dispatches encoded with this shape.
    pub dispatch_count: u32,
    /// CPU wall-clock time spent from command-buffer creation through completion.
    pub cpu_wall_ns: u128,
    /// GPU elapsed time from Metal timestamp counters, when available.
    pub gpu_elapsed_us: Option<f64>,
    /// Raw timestamp-counter sample at compute encoder start, when available.
    pub counter_sample_start: Option<u64>,
    /// Raw timestamp-counter sample at compute encoder end, when available.
    pub counter_sample_end: Option<u64>,
    /// Raw timestamp-counter sample delta, when available.
    pub counter_sample_delta: Option<u64>,
    /// CPU timestamp calibration span from `sample_timestamps`, when available.
    pub counter_cpu_timestamp_delta: Option<u64>,
    /// GPU timestamp calibration span from `sample_timestamps`, when available.
    pub counter_gpu_timestamp_delta: Option<u64>,
}

/// Reusable shared buffers for one `Fp128<P>` vector shape.
pub struct Fp128VectorBuffers<const P: u128> {
    len: usize,
    params: Fp128MetalParams,
    _field: PhantomData<Fp128<P>>,
    #[cfg(target_os = "macos")]
    raw: platform::RawFp128VectorBuffers,
}

impl<const P: u128> Fp128VectorBuffers<P> {
    /// Number of field elements in the buffers.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector buffers are empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Field parameters associated with the buffers.
    #[must_use]
    pub const fn params(&self) -> Fp128MetalParams {
        self.params
    }

    /// Storage layout used by the reusable buffers.
    #[must_use]
    pub const fn storage_mode(&self) -> Fp128BufferStorageMode {
        self.raw_storage_mode()
    }

    #[cfg(target_os = "macos")]
    const fn raw_storage_mode(&self) -> Fp128BufferStorageMode {
        self.raw.storage_mode
    }

    #[cfg(not(target_os = "macos"))]
    const fn raw_storage_mode(&self) -> Fp128BufferStorageMode {
        Fp128BufferStorageMode::Shared
    }
}

/// Entry point for Metal-backed Akita compute.
pub struct MetalBackend {
    #[cfg(target_os = "macos")]
    raw: platform::RawMetalBackend,
}

impl MetalBackend {
    /// Construct a Metal backend handle.
    pub fn new() -> MetalResult<Self> {
        platform_new_backend()
    }

    /// Return information about the selected default Metal device.
    pub fn default_device_info(&self) -> MetalResult<MetalDeviceInfo> {
        platform_device_info(self)
    }

    /// Return static pipeline facts for one `Fp128` vector operation.
    pub fn fp128_pipeline_info(&self, op: Fp128VectorOp) -> MetalResult<Fp128PipelineInfo> {
        platform_fp128_pipeline_info(self, op)
    }

    /// Allocate reusable buffers for `Fp128<P>` vector operations.
    pub fn create_fp128_vector_buffers<const P: u128>(
        &self,
        len: usize,
    ) -> MetalResult<Fp128VectorBuffers<P>> {
        self.create_fp128_vector_buffers_with_options(len, Fp128BufferOptions::default())
    }

    /// Allocate reusable buffers for `Fp128<P>` vector operations with explicit
    /// allocation options.
    pub fn create_fp128_vector_buffers_with_options<const P: u128>(
        &self,
        len: usize,
        options: Fp128BufferOptions,
    ) -> MetalResult<Fp128VectorBuffers<P>> {
        Fp128VectorPlan::validate_len(len)?;
        platform_create_fp128_vector_buffers::<P>(self, len, options)
    }

    /// Copy CPU input vectors into reusable Metal buffers.
    pub fn upload_fp128_vector_inputs<const P: u128>(
        &self,
        buffers: &mut Fp128VectorBuffers<P>,
        lhs: &[Fp128<P>],
        rhs: &[Fp128<P>],
    ) -> MetalResult<()> {
        self.upload_fp128_vector_inputs_profiled(buffers, lhs, rhs)
            .map(|_| ())
    }

    /// Copy CPU input vectors into reusable Metal buffers and return transfer
    /// timing data.
    pub fn upload_fp128_vector_inputs_profiled<const P: u128>(
        &self,
        buffers: &mut Fp128VectorBuffers<P>,
        lhs: &[Fp128<P>],
        rhs: &[Fp128<P>],
    ) -> MetalResult<Fp128TransferProfile> {
        validate_binary_lengths(buffers.len, lhs.len(), rhs.len())?;
        platform_upload_fp128_vector_inputs(self, buffers, lhs, rhs)
    }

    /// Dispatch one `Fp128<P>` vector operation using reusable Metal buffers.
    pub fn dispatch_fp128_vector<const P: u128>(
        &self,
        op: Fp128VectorOp,
        buffers: &Fp128VectorBuffers<P>,
    ) -> MetalResult<()> {
        self.dispatch_fp128_vector_with_options(op, buffers, Fp128DispatchOptions::default())
    }

    /// Dispatch one `Fp128<P>` vector operation using explicit dispatch options.
    pub fn dispatch_fp128_vector_with_options<const P: u128>(
        &self,
        op: Fp128VectorOp,
        buffers: &Fp128VectorBuffers<P>,
        options: Fp128DispatchOptions,
    ) -> MetalResult<()> {
        platform_dispatch_fp128_vector(self, op, buffers, options)
    }

    /// Dispatch one `Fp128<P>` vector operation and return low-level timing data.
    pub fn dispatch_fp128_vector_profiled<const P: u128>(
        &self,
        op: Fp128VectorOp,
        buffers: &Fp128VectorBuffers<P>,
    ) -> MetalResult<Fp128DispatchProfile> {
        self.dispatch_fp128_vector_profiled_with_options(
            op,
            buffers,
            Fp128DispatchOptions::default(),
        )
    }

    /// Dispatch one `Fp128<P>` vector operation with explicit options and return
    /// low-level timing data.
    pub fn dispatch_fp128_vector_profiled_with_options<const P: u128>(
        &self,
        op: Fp128VectorOp,
        buffers: &Fp128VectorBuffers<P>,
        options: Fp128DispatchOptions,
    ) -> MetalResult<Fp128DispatchProfile> {
        platform_dispatch_fp128_vector_profiled(self, op, buffers, options)
    }

    /// Copy a reusable Metal output buffer into an existing CPU vector.
    pub fn read_fp128_vector_output_into<const P: u128>(
        &self,
        buffers: &Fp128VectorBuffers<P>,
        out: &mut [Fp128<P>],
    ) -> MetalResult<()> {
        self.read_fp128_vector_output_into_profiled(buffers, out)
            .map(|_| ())
    }

    /// Copy a reusable Metal output buffer into an existing CPU vector and
    /// return transfer timing data.
    pub fn read_fp128_vector_output_into_profiled<const P: u128>(
        &self,
        buffers: &Fp128VectorBuffers<P>,
        out: &mut [Fp128<P>],
    ) -> MetalResult<Fp128TransferProfile> {
        if out.len() != buffers.len {
            return Err(MetalError::InvalidInput(
                "fp128 vector output length must match the reusable buffers",
            ));
        }
        platform_read_fp128_vector_output_into(self, buffers, out)
    }

    /// Copy a reusable Metal output buffer into a new CPU vector.
    pub fn read_fp128_vector_output<const P: u128>(
        &self,
        buffers: &Fp128VectorBuffers<P>,
    ) -> MetalResult<Vec<Fp128<P>>> {
        let mut out = vec![Fp128::<P>::zero(); buffers.len];
        self.read_fp128_vector_output_into(buffers, &mut out)?;
        Ok(out)
    }

    /// Run a complete `Fp128<P>` vector operation, including buffer allocation
    /// and host/device copies.
    pub fn fp128_vector<const P: u128>(
        &self,
        op: Fp128VectorOp,
        lhs: &[Fp128<P>],
        rhs: &[Fp128<P>],
    ) -> MetalResult<Vec<Fp128<P>>> {
        validate_binary_lengths(lhs.len(), lhs.len(), rhs.len())?;
        let mut buffers = self.create_fp128_vector_buffers::<P>(lhs.len())?;
        self.upload_fp128_vector_inputs(&mut buffers, lhs, rhs)?;
        self.dispatch_fp128_vector(op, &buffers)?;
        self.read_fp128_vector_output(&buffers)
    }
}

fn validate_binary_lengths(buffer_len: usize, lhs_len: usize, rhs_len: usize) -> MetalResult<()> {
    if lhs_len != rhs_len {
        return Err(MetalError::InvalidInput(
            "fp128 vector kernels require equal-length inputs",
        ));
    }
    if lhs_len != buffer_len {
        return Err(MetalError::InvalidInput(
            "fp128 vector input length must match the reusable buffers",
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn platform_new_backend() -> MetalResult<MetalBackend> {
    Ok(MetalBackend {
        raw: platform::RawMetalBackend::new()?,
    })
}

#[cfg(not(target_os = "macos"))]
fn platform_new_backend() -> MetalResult<MetalBackend> {
    Err(MetalError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
fn platform_device_info(backend: &MetalBackend) -> MetalResult<MetalDeviceInfo> {
    Ok(backend.raw.device_info())
}

#[cfg(not(target_os = "macos"))]
fn platform_device_info(_backend: &MetalBackend) -> MetalResult<MetalDeviceInfo> {
    Err(MetalError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
fn platform_fp128_pipeline_info(
    backend: &MetalBackend,
    op: Fp128VectorOp,
) -> MetalResult<Fp128PipelineInfo> {
    Ok(backend.raw.fp128_pipeline_info(op))
}

#[cfg(not(target_os = "macos"))]
fn platform_fp128_pipeline_info(
    _backend: &MetalBackend,
    _op: Fp128VectorOp,
) -> MetalResult<Fp128PipelineInfo> {
    Err(MetalError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
fn platform_create_fp128_vector_buffers<const P: u128>(
    backend: &MetalBackend,
    len: usize,
    options: Fp128BufferOptions,
) -> MetalResult<Fp128VectorBuffers<P>> {
    Ok(Fp128VectorBuffers {
        len,
        params: Fp128MetalParams::for_modulus::<P>(),
        _field: PhantomData,
        raw: backend.raw.create_fp128_vector_buffers::<P>(len, options)?,
    })
}

#[cfg(not(target_os = "macos"))]
fn platform_create_fp128_vector_buffers<const P: u128>(
    _backend: &MetalBackend,
    _len: usize,
    _options: Fp128BufferOptions,
) -> MetalResult<Fp128VectorBuffers<P>> {
    Err(MetalError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
fn platform_upload_fp128_vector_inputs<const P: u128>(
    backend: &MetalBackend,
    buffers: &mut Fp128VectorBuffers<P>,
    lhs: &[Fp128<P>],
    rhs: &[Fp128<P>],
) -> MetalResult<Fp128TransferProfile> {
    backend
        .raw
        .upload_fp128_vector_inputs(&mut buffers.raw, lhs, rhs)
}

#[cfg(not(target_os = "macos"))]
fn platform_upload_fp128_vector_inputs<const P: u128>(
    _backend: &MetalBackend,
    _buffers: &mut Fp128VectorBuffers<P>,
    _lhs: &[Fp128<P>],
    _rhs: &[Fp128<P>],
) -> MetalResult<Fp128TransferProfile> {
    Err(MetalError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
fn platform_dispatch_fp128_vector<const P: u128>(
    backend: &MetalBackend,
    op: Fp128VectorOp,
    buffers: &Fp128VectorBuffers<P>,
    options: Fp128DispatchOptions,
) -> MetalResult<()> {
    backend.raw.dispatch_fp128_vector(op, &buffers.raw, options)
}

#[cfg(not(target_os = "macos"))]
fn platform_dispatch_fp128_vector<const P: u128>(
    _backend: &MetalBackend,
    _op: Fp128VectorOp,
    _buffers: &Fp128VectorBuffers<P>,
    _options: Fp128DispatchOptions,
) -> MetalResult<()> {
    Err(MetalError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
fn platform_dispatch_fp128_vector_profiled<const P: u128>(
    backend: &MetalBackend,
    op: Fp128VectorOp,
    buffers: &Fp128VectorBuffers<P>,
    options: Fp128DispatchOptions,
) -> MetalResult<Fp128DispatchProfile> {
    backend
        .raw
        .dispatch_fp128_vector_profiled(op, &buffers.raw, options)
}

#[cfg(not(target_os = "macos"))]
fn platform_dispatch_fp128_vector_profiled<const P: u128>(
    _backend: &MetalBackend,
    _op: Fp128VectorOp,
    _buffers: &Fp128VectorBuffers<P>,
    _options: Fp128DispatchOptions,
) -> MetalResult<Fp128DispatchProfile> {
    Err(MetalError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
fn platform_read_fp128_vector_output_into<const P: u128>(
    backend: &MetalBackend,
    buffers: &Fp128VectorBuffers<P>,
    out: &mut [Fp128<P>],
) -> MetalResult<Fp128TransferProfile> {
    backend.raw.read_fp128_vector_output_into(&buffers.raw, out)
}

#[cfg(not(target_os = "macos"))]
fn platform_read_fp128_vector_output_into<const P: u128>(
    _backend: &MetalBackend,
    _buffers: &Fp128VectorBuffers<P>,
    _out: &mut [Fp128<P>],
) -> MetalResult<Fp128TransferProfile> {
    Err(MetalError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
mod platform;

#[cfg(test)]
mod tests {
    use akita_field::fields::Fp128;
    use akita_field::CanonicalField;

    use super::*;

    const P_A7F7: u128 = 0xffffffffffffffffffffffff00005809;
    const P_2355: u128 = 0xfffffffffffffffffffffffffffff6cd;

    fn sample_inputs<const P: u128>(len: usize) -> (Vec<Fp128<P>>, Vec<Fp128<P>>) {
        let mut lhs = Vec::with_capacity(len);
        let mut rhs = Vec::with_capacity(len);
        let mut state = 0x9e37_79b9_7f4a_7c15_d1b5_4a32_d192_ed03u128;

        for i in 0..len {
            state = state
                .wrapping_mul(0xda94_2042_e4dd_58b5_1331_11eb_a1f0_a2c3)
                .wrapping_add(i as u128 + 1);
            lhs.push(Fp128::<P>::from_canonical_u128_reduced(state));
            rhs.push(Fp128::<P>::from_canonical_u128_reduced(
                state.rotate_left(37) ^ (i as u128),
            ));
        }

        lhs[0] = Fp128::<P>::zero();
        rhs[0] = Fp128::<P>::zero();
        lhs[1] = Fp128::<P>::one();
        rhs[1] = Fp128::<P>::one();
        (lhs, rhs)
    }

    fn assert_ops_match<const P: u128>() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let (lhs, rhs) = sample_inputs::<P>(2051);

        for storage_mode in [
            Fp128BufferStorageMode::Shared,
            Fp128BufferStorageMode::PrivateStaged,
        ] {
            for op in [Fp128VectorOp::Add, Fp128VectorOp::Sub, Fp128VectorOp::Mul] {
                let mut buffers = backend
                    .create_fp128_vector_buffers_with_options::<P>(
                        lhs.len(),
                        Fp128BufferOptions::default().with_storage_mode(storage_mode),
                    )
                    .unwrap();
                backend
                    .upload_fp128_vector_inputs(&mut buffers, &lhs, &rhs)
                    .unwrap();
                backend.dispatch_fp128_vector(op, &buffers).unwrap();
                let gpu = backend.read_fp128_vector_output(&buffers).unwrap();
                let expected: Vec<_> = lhs
                    .iter()
                    .zip(&rhs)
                    .map(|(&lhs, &rhs)| match op {
                        Fp128VectorOp::Add => lhs + rhs,
                        Fp128VectorOp::Sub => lhs - rhs,
                        Fp128VectorOp::Mul => lhs * rhs,
                    })
                    .collect();
                assert_eq!(gpu, expected, "storage_mode={storage_mode:?}, op={op:?}");
            }
        }
    }

    #[test]
    fn backend_construction_reports_platform_state() {
        match MetalBackend::new() {
            Ok(backend) => {
                let info = backend.default_device_info().unwrap();
                assert!(!info.name.is_empty());
            }
            #[cfg(target_os = "macos")]
            Err(MetalError::NoSystemDevice | MetalError::KernelLibrary(_)) => {}
            #[cfg(not(target_os = "macos"))]
            Err(MetalError::UnsupportedPlatform) => {}
            Err(err) => panic!("unexpected Metal backend construction error: {err:?}"),
        }
    }

    #[test]
    fn metal_fp128_vector_ops_match_cpu_prime_a7f7() {
        assert_ops_match::<P_A7F7>();
    }

    #[test]
    fn metal_fp128_vector_ops_match_cpu_prime_2355() {
        assert_ops_match::<P_2355>();
    }
}
