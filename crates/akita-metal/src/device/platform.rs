use std::{mem, ptr, slice, time::Instant};

use akita_field::fields::Fp128;
use metal::{
    Buffer, BufferRef, CommandBufferRef, CommandQueue, CompileOptions, ComputePassDescriptor,
    ComputePipelineState, CounterSampleBuffer, CounterSampleBufferRef, Device,
    MTLCommandBufferStatus, MTLCounterSamplingPoint, MTLResourceOptions, MTLSize, MTLStorageMode,
    NSRange,
};
use objc::rc::autoreleasepool;

use super::{
    Fp128BufferOptions, Fp128BufferStorageMode, Fp128DispatchOptions, Fp128DispatchProfile,
    Fp128KernelParams, Fp128Limb, Fp128PipelineInfo, Fp128TransferProfile, Fp128VectorOp,
    MetalDeviceInfo, MetalError, MetalResult,
};
use crate::kernels::FP128_VECTOR_METAL;

pub(super) struct RawMetalBackend {
    device: Device,
    queue: CommandQueue,
    add_pipeline: ComputePipelineState,
    sub_pipeline: ComputePipelineState,
    mul_pipeline: ComputePipelineState,
}

pub(super) struct RawFp128VectorBuffers {
    lhs: Buffer,
    rhs: Buffer,
    out: Buffer,
    lhs_staging: Option<Buffer>,
    rhs_staging: Option<Buffer>,
    out_staging: Option<Buffer>,
    pub(super) storage_mode: Fp128BufferStorageMode,
    params: Fp128KernelParams,
}

impl RawMetalBackend {
    pub(super) fn new() -> MetalResult<Self> {
        autoreleasepool(|| {
            let device = Device::system_default().ok_or(MetalError::NoSystemDevice)?;
            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(FP128_VECTOR_METAL, &options)
                .map_err(MetalError::KernelLibrary)?;

            Ok(Self {
                add_pipeline: pipeline(
                    &device,
                    &library,
                    Fp128VectorOp::Add.metal_function_name(),
                )?,
                sub_pipeline: pipeline(
                    &device,
                    &library,
                    Fp128VectorOp::Sub.metal_function_name(),
                )?,
                mul_pipeline: pipeline(
                    &device,
                    &library,
                    Fp128VectorOp::Mul.metal_function_name(),
                )?,
                queue: device.new_command_queue(),
                device,
            })
        })
    }

    pub(super) fn device_info(&self) -> MetalDeviceInfo {
        let max_threads_per_threadgroup = self.device.max_threads_per_threadgroup();
        MetalDeviceInfo {
            name: self.device.name().to_owned(),
            supports_unified_memory: self.device.has_unified_memory(),
            recommended_max_working_set_size: self.device.recommended_max_working_set_size(),
            max_transfer_rate: self.device.max_transfer_rate(),
            max_threads_per_threadgroup_width: max_threads_per_threadgroup.width,
            supports_stage_boundary_counters: self
                .device
                .supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary),
        }
    }

    pub(super) fn create_fp128_vector_buffers<const P: u128>(
        &self,
        len: usize,
        options: Fp128BufferOptions,
    ) -> MetalResult<RawFp128VectorBuffers> {
        let byte_len = len
            .checked_mul(mem::size_of::<Fp128Limb>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(MetalError::InvalidInput(
                "fp128 vector buffer byte length overflowed",
            ))?;
        let shared_options =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache;
        let private_options = MTLResourceOptions::StorageModePrivate;

        let shared_buffer = || self.device.new_buffer(byte_len, shared_options);
        let private_buffer = || self.device.new_buffer(byte_len, private_options);

        let buffers = match options.storage_mode {
            Fp128BufferStorageMode::Shared => RawFp128VectorBuffers {
                lhs: shared_buffer(),
                rhs: shared_buffer(),
                out: shared_buffer(),
                lhs_staging: None,
                rhs_staging: None,
                out_staging: None,
                storage_mode: Fp128BufferStorageMode::Shared,
                params: Fp128KernelParams::new::<P>(len)?,
            },
            Fp128BufferStorageMode::PrivateStaged => RawFp128VectorBuffers {
                lhs: private_buffer(),
                rhs: private_buffer(),
                out: private_buffer(),
                lhs_staging: Some(shared_buffer()),
                rhs_staging: Some(shared_buffer()),
                out_staging: Some(shared_buffer()),
                storage_mode: Fp128BufferStorageMode::PrivateStaged,
                params: Fp128KernelParams::new::<P>(len)?,
            },
        };

        Ok(buffers)
    }

    pub(super) fn upload_fp128_vector_inputs<const P: u128>(
        &self,
        buffers: &mut RawFp128VectorBuffers,
        lhs: &[Fp128<P>],
        rhs: &[Fp128<P>],
    ) -> MetalResult<Fp128TransferProfile> {
        let total_start = Instant::now();
        let host_copy_start = Instant::now();
        let gpu_blit_ns = match buffers.storage_mode {
            Fp128BufferStorageMode::Shared => {
                copy_fields_to_buffer(lhs, &buffers.lhs);
                copy_fields_to_buffer(rhs, &buffers.rhs);
                0
            }
            Fp128BufferStorageMode::PrivateStaged => {
                let lhs_staging = buffers
                    .lhs_staging
                    .as_ref()
                    .ok_or(MetalError::InvalidInput(
                        "private fp128 buffers missing lhs staging",
                    ))?;
                let rhs_staging = buffers
                    .rhs_staging
                    .as_ref()
                    .ok_or(MetalError::InvalidInput(
                        "private fp128 buffers missing rhs staging",
                    ))?;
                copy_fields_to_buffer(lhs, lhs_staging);
                copy_fields_to_buffer(rhs, rhs_staging);
                let host_copy_ns = host_copy_start.elapsed().as_nanos();
                let blit_start = Instant::now();
                self.blit_copy_to_private(lhs_staging, &buffers.lhs, rhs_staging, &buffers.rhs)?;
                return Ok(Fp128TransferProfile {
                    storage_mode: buffers.storage_mode,
                    host_copy_ns,
                    gpu_blit_ns: blit_start.elapsed().as_nanos(),
                    total_ns: total_start.elapsed().as_nanos(),
                });
            }
        };

        Ok(Fp128TransferProfile {
            storage_mode: buffers.storage_mode,
            host_copy_ns: host_copy_start.elapsed().as_nanos(),
            gpu_blit_ns,
            total_ns: total_start.elapsed().as_nanos(),
        })
    }

    pub(super) fn read_fp128_vector_output_into<const P: u128>(
        &self,
        buffers: &RawFp128VectorBuffers,
        out: &mut [Fp128<P>],
    ) -> MetalResult<Fp128TransferProfile> {
        let total_start = Instant::now();
        let gpu_blit_ns = if buffers.storage_mode == Fp128BufferStorageMode::PrivateStaged {
            let out_staging = buffers
                .out_staging
                .as_ref()
                .ok_or(MetalError::InvalidInput(
                    "private fp128 buffers missing output staging",
                ))?;
            let blit_start = Instant::now();
            self.blit_copy_from_private(&buffers.out, out_staging)?;
            blit_start.elapsed().as_nanos()
        } else {
            0
        };

        let host_copy_start = Instant::now();
        let source = buffers.out_staging.as_ref().unwrap_or(&buffers.out);
        copy_fields_from_buffer(source, out);
        Ok(Fp128TransferProfile {
            storage_mode: buffers.storage_mode,
            host_copy_ns: host_copy_start.elapsed().as_nanos(),
            gpu_blit_ns,
            total_ns: total_start.elapsed().as_nanos(),
        })
    }

    pub(super) fn dispatch_fp128_vector(
        &self,
        op: Fp128VectorOp,
        buffers: &RawFp128VectorBuffers,
        options: Fp128DispatchOptions,
    ) -> MetalResult<()> {
        self.dispatch_fp128_vector_inner(op, buffers, options, false)
            .map(|_| ())
    }

    pub(super) fn dispatch_fp128_vector_profiled(
        &self,
        op: Fp128VectorOp,
        buffers: &RawFp128VectorBuffers,
        options: Fp128DispatchOptions,
    ) -> MetalResult<Fp128DispatchProfile> {
        self.dispatch_fp128_vector_inner(op, buffers, options, true)
    }

    pub(super) fn fp128_pipeline_info(&self, op: Fp128VectorOp) -> Fp128PipelineInfo {
        let pipeline = match op {
            Fp128VectorOp::Add => &self.add_pipeline,
            Fp128VectorOp::Sub => &self.sub_pipeline,
            Fp128VectorOp::Mul => &self.mul_pipeline,
        };

        Fp128PipelineInfo {
            op,
            thread_execution_width: pipeline.thread_execution_width(),
            max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup(),
            default_threadgroup_width: default_threadgroup_width(pipeline),
        }
    }

    fn dispatch_fp128_vector_inner(
        &self,
        op: Fp128VectorOp,
        buffers: &RawFp128VectorBuffers,
        options: Fp128DispatchOptions,
        profile: bool,
    ) -> MetalResult<Fp128DispatchProfile> {
        autoreleasepool(|| {
            let pipeline = self.pipeline_for(op);
            let len = buffers.params.len as u64;
            let shape = dispatch_shape(pipeline, len, options)?;

            let sample_counters = profile && options.sample_counters;
            let counter_sample_buffer = sample_counters
                .then(|| create_counter_sample_buffer(&self.device))
                .flatten();
            let destination_buffer = counter_sample_buffer.as_ref().map(|_| {
                self.device.new_buffer(
                    (mem::size_of::<u64>() * NUM_COUNTER_SAMPLES as usize) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            });

            let mut cpu_timestamp_start = 0;
            let mut gpu_timestamp_start = 0;
            if counter_sample_buffer.is_some() {
                self.device
                    .sample_timestamps(&mut cpu_timestamp_start, &mut gpu_timestamp_start);
            }

            let command_buffer = self.queue.new_command_buffer();
            let wall_start = Instant::now();
            let encoder = match counter_sample_buffer.as_ref() {
                Some(sample_buffer) => {
                    let descriptor = ComputePassDescriptor::new();
                    attach_counter_sample_buffer(descriptor, sample_buffer);
                    command_buffer.compute_command_encoder_with_descriptor(descriptor)
                }
                None => command_buffer.new_compute_command_encoder(),
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(1, Some(&buffers.rhs), 0);
            encoder.set_bytes(
                3,
                mem::size_of::<Fp128KernelParams>() as u64,
                ptr::from_ref(&buffers.params).cast(),
            );

            let threadgroups = MTLSize {
                width: shape.threadgroup_count,
                height: 1,
                depth: 1,
            };
            let threads_per_threadgroup = MTLSize {
                width: shape.threadgroup_width,
                height: 1,
                depth: 1,
            };
            encoder.set_buffer(0, Some(&buffers.lhs), 0);
            encoder.set_buffer(2, Some(&buffers.out), 0);
            encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
            encoder.end_encoding();

            if let (Some(sample_buffer), Some(destination_buffer)) =
                (counter_sample_buffer.as_ref(), destination_buffer.as_ref())
            {
                resolve_counter_samples(command_buffer, sample_buffer, destination_buffer);
            }

            command_buffer.commit();
            command_buffer.wait_until_completed();
            let cpu_wall_ns = wall_start.elapsed().as_nanos();

            if command_buffer.status() == MTLCommandBufferStatus::Completed {
                let mut cpu_timestamp_end = 0;
                let mut gpu_timestamp_end = 0;
                if counter_sample_buffer.is_some() {
                    self.device
                        .sample_timestamps(&mut cpu_timestamp_end, &mut gpu_timestamp_end);
                }

                let counter_measurement = destination_buffer.as_ref().and_then(|buffer| {
                    gpu_counter_measurement(
                        buffer,
                        cpu_timestamp_start,
                        cpu_timestamp_end,
                        gpu_timestamp_start,
                        gpu_timestamp_end,
                    )
                });

                Ok(Fp128DispatchProfile {
                    op,
                    len: buffers.params.len as usize,
                    thread_execution_width: pipeline.thread_execution_width(),
                    max_total_threads_per_threadgroup: pipeline.max_total_threads_per_threadgroup(),
                    threadgroup_width: shape.threadgroup_width,
                    threadgroup_count: shape.threadgroup_count,
                    dispatch_count: 1,
                    cpu_wall_ns,
                    gpu_elapsed_us: counter_measurement
                        .as_ref()
                        .and_then(|measurement| measurement.elapsed_us),
                    counter_sample_start: counter_measurement
                        .as_ref()
                        .map(|measurement| measurement.sample_start),
                    counter_sample_end: counter_measurement
                        .as_ref()
                        .map(|measurement| measurement.sample_end),
                    counter_sample_delta: counter_measurement
                        .as_ref()
                        .map(|measurement| measurement.sample_delta),
                    counter_cpu_timestamp_delta: counter_measurement
                        .as_ref()
                        .map(|measurement| measurement.cpu_timestamp_delta),
                    counter_gpu_timestamp_delta: counter_measurement
                        .as_ref()
                        .map(|measurement| measurement.gpu_timestamp_delta),
                })
            } else {
                Err(MetalError::CommandFailed(op.metal_function_name()))
            }
        })
    }

    fn pipeline_for(&self, op: Fp128VectorOp) -> &ComputePipelineState {
        match op {
            Fp128VectorOp::Add => &self.add_pipeline,
            Fp128VectorOp::Sub => &self.sub_pipeline,
            Fp128VectorOp::Mul => &self.mul_pipeline,
        }
    }

    fn blit_copy_to_private(
        &self,
        lhs_staging: &Buffer,
        lhs_private: &Buffer,
        rhs_staging: &Buffer,
        rhs_private: &Buffer,
    ) -> MetalResult<()> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_blit_command_encoder();
            encoder.copy_from_buffer(lhs_staging, 0, lhs_private, 0, lhs_staging.length());
            encoder.copy_from_buffer(rhs_staging, 0, rhs_private, 0, rhs_staging.length());
            encoder.end_encoding();
            wait_for_command(command_buffer, "fp128 buffer upload")
        })
    }

    fn blit_copy_from_private(
        &self,
        out_private: &Buffer,
        out_staging: &Buffer,
    ) -> MetalResult<()> {
        autoreleasepool(|| {
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_blit_command_encoder();
            encoder.copy_from_buffer(out_private, 0, out_staging, 0, out_private.length());
            encoder.end_encoding();
            wait_for_command(command_buffer, "fp128 buffer readback")
        })
    }
}

const NUM_COUNTER_SAMPLES: u64 = 2;

struct Fp128DispatchShape {
    threadgroup_width: u64,
    threadgroup_count: u64,
}

struct GpuCounterMeasurement {
    elapsed_us: Option<f64>,
    sample_start: u64,
    sample_end: u64,
    sample_delta: u64,
    cpu_timestamp_delta: u64,
    gpu_timestamp_delta: u64,
}

fn dispatch_shape(
    pipeline: &ComputePipelineState,
    len: u64,
    options: Fp128DispatchOptions,
) -> MetalResult<Fp128DispatchShape> {
    let threadgroup_width = options
        .threadgroup_width
        .unwrap_or_else(|| default_threadgroup_width(pipeline));

    if threadgroup_width == 0 {
        return Err(MetalError::InvalidInput(
            "fp128 vector threadgroup width must be nonzero",
        ));
    }
    if threadgroup_width > pipeline.max_total_threads_per_threadgroup() {
        return Err(MetalError::InvalidInput(
            "fp128 vector threadgroup width exceeds the Metal pipeline limit",
        ));
    }

    Ok(Fp128DispatchShape {
        threadgroup_width,
        threadgroup_count: len.div_ceil(threadgroup_width),
    })
}

fn default_threadgroup_width(pipeline: &ComputePipelineState) -> u64 {
    pipeline
        .thread_execution_width()
        .saturating_mul(4)
        .min(pipeline.max_total_threads_per_threadgroup())
        .max(1)
}

fn create_counter_sample_buffer(device: &Device) -> Option<CounterSampleBuffer> {
    if !device.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary) {
        return None;
    }

    let descriptor = metal::CounterSampleBufferDescriptor::new();
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_sample_count(NUM_COUNTER_SAMPLES);

    let counter_sets = device.counter_sets();
    let timestamp_counter = counter_sets
        .iter()
        .find(|counter_set| counter_set.name() == "timestamp")?;
    descriptor.set_counter_set(timestamp_counter);

    device
        .new_counter_sample_buffer_with_descriptor(&descriptor)
        .ok()
}

fn attach_counter_sample_buffer(
    descriptor: &metal::ComputePassDescriptorRef,
    sample_buffer: &CounterSampleBufferRef,
) {
    let Some(attachment) = descriptor.sample_buffer_attachments().object_at(0) else {
        return;
    };

    attachment.set_sample_buffer(sample_buffer);
    attachment.set_start_of_encoder_sample_index(0);
    attachment.set_end_of_encoder_sample_index(1);
}

fn resolve_counter_samples(
    command_buffer: &CommandBufferRef,
    sample_buffer: &CounterSampleBufferRef,
    destination_buffer: &BufferRef,
) {
    let blit_encoder = command_buffer.new_blit_command_encoder();
    blit_encoder.resolve_counters(
        sample_buffer,
        NSRange::new(0, NUM_COUNTER_SAMPLES),
        destination_buffer,
        0,
    );
    blit_encoder.end_encoding();
}

fn gpu_counter_measurement(
    resolved_sample_buffer: &BufferRef,
    cpu_timestamp_start: u64,
    cpu_timestamp_end: u64,
    gpu_timestamp_start: u64,
    gpu_timestamp_end: u64,
) -> Option<GpuCounterMeasurement> {
    let samples = unsafe {
        slice::from_raw_parts(
            resolved_sample_buffer.contents().cast::<u64>(),
            NUM_COUNTER_SAMPLES as usize,
        )
    };
    let sample_start = samples[0];
    let sample_end = samples[1];
    let cpu_timestamp_delta = cpu_timestamp_end.checked_sub(cpu_timestamp_start)?;
    let gpu_timestamp_delta = gpu_timestamp_end.checked_sub(gpu_timestamp_start)?;
    let sample_delta = sample_end.checked_sub(sample_start)?;

    let elapsed_us = if cpu_timestamp_delta == 0 || gpu_timestamp_delta == 0 || sample_delta == 0 {
        None
    } else {
        let nanoseconds =
            (sample_delta as f64) / (gpu_timestamp_delta as f64) * (cpu_timestamp_delta as f64);
        Some(nanoseconds / 1000.0)
    };

    Some(GpuCounterMeasurement {
        elapsed_us,
        sample_start,
        sample_end,
        sample_delta,
        cpu_timestamp_delta,
        gpu_timestamp_delta,
    })
}

fn pipeline(
    device: &Device,
    library: &metal::Library,
    name: &'static str,
) -> MetalResult<ComputePipelineState> {
    let function = library
        .get_function(name, None)
        .map_err(|message| MetalError::KernelFunction { name, message })?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalError::Pipeline { name, message })
}

fn copy_fields_from_buffer<const P: u128>(buffer: &Buffer, out: &mut [Fp128<P>]) {
    assert_fp128_limb_layout::<P>();
    unsafe {
        ptr::copy_nonoverlapping(
            buffer.contents().cast::<Fp128<P>>(),
            out.as_mut_ptr(),
            out.len(),
        );
    }
}

fn copy_fields_to_buffer<const P: u128>(fields: &[Fp128<P>], buffer: &Buffer) {
    assert_fp128_limb_layout::<P>();
    unsafe {
        ptr::copy_nonoverlapping(
            fields.as_ptr(),
            buffer.contents().cast::<Fp128<P>>(),
            fields.len(),
        );
    }
}

fn assert_fp128_limb_layout<const P: u128>() {
    assert_eq!(mem::size_of::<Fp128<P>>(), mem::size_of::<Fp128Limb>());
    assert_eq!(mem::align_of::<Fp128<P>>(), mem::align_of::<Fp128Limb>());
}

fn wait_for_command(command_buffer: &CommandBufferRef, context: &'static str) -> MetalResult<()> {
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() == MTLCommandBufferStatus::Completed {
        Ok(())
    } else {
        Err(MetalError::CommandFailed(context))
    }
}
