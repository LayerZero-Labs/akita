//! Commitment stages on the device, contributed to the CPU backend.
//!
//! [`MetalCommitmentProvider`] implements `akita-cpu-backend`'s
//! [`CommitmentStageProvider`]: installed with
//! `CpuBackend::with_commitment_stage_provider`, it runs the inner (A) and
//! outer (B) commitment stages of every request it accepts on the GPU, and
//! the CPU backend keeps everything else (compression, retained state,
//! proving). Each stage's output equals the CPU stage's word for word, so
//! setup, commitment and proof bytes do not depend on the route.
//!
//! - **Inner**: dense coefficient sources are decomposed into balanced digit
//!   planes (`i8` for `log_basis <= 8`, `i16` up to 16) and multiplied by A;
//!   one-hot sources run the one-hot kernel. The rows `t` are read back once,
//!   so the retained state (and a CPU outer stage) get host rows.
//! - **Outer**: `t` is arranged into the dyadic slices (zero padding), each
//!   `D_A` ring decomposed as `D_A / D_B` subrings of base-`2^log_basis_outer`
//!   digits, and multiplied by B; `u` comes back in `[slice][row][coeff]`
//!   order.
//!
//! Shapes without kernels are declined by `supports_plan` before any source
//! is materialized, and the CPU stages take the request. Setup matrices are
//! prepared on the device from the setup's coefficient form on first use and
//! cached per shape.

/// Runs `$body` with `$d` bound to the runtime ring degree `$degree` as a
/// constant, for every degree with transform kernels.
macro_rules! with_ring_degree {
    ($degree:expr, |$d:ident| $body:expr) => {
        match $degree {
            64 => {
                const $d: usize = 64;
                $body
            }
            128 => {
                const $d: usize = 128;
                $body
            }
            256 => {
                const $d: usize = 256;
                $body
            }
            512 => {
                const $d: usize = 512;
                $body
            }
            1024 => {
                const $d: usize = 1024;
                $body
            }
            other => Err(AkitaError::InvalidSetup(format!(
                "ring degree {other} has no Metal commitment kernel"
            ))),
        }
    };
}

mod inner;
mod matrix;
mod outer;
mod shared;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use akita_algebra::tables::{q128_primes, Q64_PRIMES};
use akita_algebra::NttPrime;
use akita_cpu_backend::commitment_backend::{
    BackendInstanceId, BackendKindId, CommitmentOperationContext, CommitmentRequestCapabilities,
    CommitmentStageProvider, CommitmentStages, DenseType, OneHotIndexWidth, OneHotType,
    PolynomialType, PreparedInnerCommitment, PreparedOuterCommitment, StageDimensionCapabilities,
    StageResources, StateOwnerCapability,
};
use akita_error::AkitaError;
use akita_types::AkitaExpandedSetup;
use jolt_field::{
    CanonicalEncoding, Field, Prime128OffsetA7F7, Prime64Offset59, Unreduced, WithCommitAccumulator,
};
use jolt_metal::MetalField;

use crate::error::AkitaMetalError;
use crate::library::{AkitaMetal, RING_DEGREES};
use inner::{MetalInnerCommit, MetalInnerExporter};
use matrix::{single_column_fits, MatrixCache};
use outer::MetalOuterCommit;
use shared::Shared;

mod sealed {
    pub trait Sealed {}
    impl Sealed for jolt_field::Prime128OffsetA7F7 {}
    impl Sealed for jolt_field::Prime64Offset59 {}
}

/// A field with device commitment stages: the base fields of Akita's fp128
/// and fp64 presets.
pub trait CommitmentField:
    MetalField
    + Field
    + CanonicalEncoding
    + Unreduced
    + WithCommitAccumulator
    + Send
    + Sync
    + 'static
    + sealed::Sealed
{
    /// The CRT primes device matrices use, as prefixes chosen by
    /// [`plan_matvec`](crate::matvec::plan_matvec).
    fn crt_primes() -> Vec<NttPrime<i32>>;
}

impl CommitmentField for Prime128OffsetA7F7 {
    fn crt_primes() -> Vec<NttPrime<i32>> {
        q128_primes().to_vec()
    }
}

impl CommitmentField for Prime64Offset59 {
    fn crt_primes() -> Vec<NttPrime<i32>> {
        Q64_PRIMES.to_vec()
    }
}

/// Backend kind of the Metal commitment stages.
struct MetalCommitKind;
/// Execution context external source operations would target.
struct MetalCommitContext;

/// One-hot chunk sizes the inner stage advertises: every power of two up to
/// `2^20` entries.
const ONEHOT_LOG_CHUNKS: usize = 20;

/// How often each stage ran, for tests and benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageCalls {
    /// Inner stages run on the device.
    pub inner: usize,
    /// Outer stages run on the device.
    pub outer: usize,
    /// One-hot inner stages at a ring degree without a one-hot kernel for
    /// the field (fp128 at `D = 1024`), committed with the CPU column sweep.
    pub onehot_host: usize,
    /// Wall time in the inner stages: staging, device work and read-back.
    pub inner_time: Duration,
    /// Wall time in the outer stages.
    pub outer_time: Duration,
}

#[derive(Default)]
struct Counters {
    inner: AtomicUsize,
    outer: AtomicUsize,
    onehot_host: AtomicUsize,
    inner_nanos: AtomicU64,
    outer_nanos: AtomicU64,
}

fn bump(counter: &AtomicUsize) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Counts one completed stage that started at `start`.
fn record(calls: &AtomicUsize, nanos: &AtomicU64, start: Instant) {
    bump(calls);
    let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    nanos.fetch_add(elapsed, Ordering::Relaxed);
}

/// Inner and outer commitment stages on an Apple GPU for one expanded setup.
///
/// Install it with `CpuBackend::with_commitment_stage_provider`. Device setup
/// matrices are prepared on first use per `(ring degree, rows, cols, plan)`
/// from the setup's coefficient form and kept until the provider is dropped.
pub struct MetalCommitmentProvider<F: CommitmentField> {
    metal: Shared<AkitaMetal>,
    expanded: Arc<AkitaExpandedSetup<F>>,
    matrices: MatrixCache<F>,
    counters: Counters,
    /// Digit-plane bytes per dense inner chunk.
    dense_chunk_bytes: usize,
}

/// Default digit-plane bytes per dense inner chunk: the fp128 dense nv26
/// root (1 GiB of `i16` planes) runs as one chunk, larger sources in several.
pub const DEFAULT_DENSE_CHUNK_BYTES: usize = 1 << 30;

impl<F: CommitmentField> MetalCommitmentProvider<F> {
    /// Opens the system GPU and compiles the kernels.
    ///
    /// Without a usable device this returns an error of class
    /// [`ErrorClass::Unavailable`](crate::ErrorClass::Unavailable), and the
    /// caller keeps the plain CPU backend.
    pub fn new(expanded: Arc<AkitaExpandedSetup<F>>) -> Result<Self, AkitaMetalError> {
        Ok(Self::with_device(AkitaMetal::new()?, expanded))
    }

    /// A provider on an already opened device.
    pub fn with_device(metal: AkitaMetal, expanded: Arc<AkitaExpandedSetup<F>>) -> Self {
        Self {
            metal: Shared::new(metal),
            expanded,
            matrices: MatrixCache::new(),
            counters: Counters::default(),
            dense_chunk_bytes: DEFAULT_DENSE_CHUNK_BYTES,
        }
    }

    /// Caps the digit planes of one dense inner chunk at `bytes` (at least
    /// one block per chunk). Any value gives the same commitment; smaller
    /// chunks use less device memory.
    pub fn with_dense_chunk_bytes(mut self, bytes: usize) -> Self {
        self.dense_chunk_bytes = bytes;
        self
    }

    /// How often each stage has run on this provider.
    pub fn stage_calls(&self) -> StageCalls {
        StageCalls {
            inner: self.counters.inner.load(Ordering::Relaxed),
            outer: self.counters.outer.load(Ordering::Relaxed),
            onehot_host: self.counters.onehot_host.load(Ordering::Relaxed),
            inner_time: Duration::from_nanos(self.counters.inner_nanos.load(Ordering::Relaxed)),
            outer_time: Duration::from_nanos(self.counters.outer_nanos.load(Ordering::Relaxed)),
        }
    }

    /// Setup matrices currently prepared on the device.
    pub fn prepared_matrices(&self) -> usize {
        self.matrices.len()
    }
}

/// Whether the inner stage implements a plan of this ring degree and digit
/// width: a degree with transform kernels, `i8` or `i16` digits, and a single
/// column that fits the CRT profile.
fn inner_supported<F: CommitmentField>(ring_degree: usize, log_basis: u32) -> bool {
    RING_DEGREES.contains(&ring_degree)
        && (1..=16).contains(&log_basis)
        && single_column_fits::<F>(ring_degree, log_basis)
}

/// Whether the outer stage implements a plan with these A and B ring degrees
/// and outer digit width: `i8` digits, a B degree with transform kernels
/// dividing the A degree, and a single column that fits the CRT profile.
fn outer_supported<F: CommitmentField>(d_a: usize, d_b: usize, log_basis: u32) -> bool {
    RING_DEGREES.contains(&d_b)
        && d_a.is_power_of_two()
        && d_a >= d_b
        && (1..=8).contains(&log_basis)
        && single_column_fits::<F>(d_b, log_basis)
}

/// Every standard type the inner stage commits: one-hot first, so one-hot
/// sources keep their structure, then dense coefficients.
fn standard_types() -> Result<Vec<PolynomialType>, AkitaError> {
    let mut types = Vec::new();
    for log_chunk in 0..=ONEHOT_LOG_CHUNKS {
        for width in [
            OneHotIndexWidth::U8,
            OneHotIndexWidth::U16,
            OneHotIndexWidth::U32,
            OneHotIndexWidth::Usize,
        ] {
            let chunk_size = akita_error::checked::pow2(log_chunk)
                .ok_or_else(|| AkitaError::InvalidSetup("one-hot chunk size overflows".into()))?;
            types.push(PolynomialType::OneHot(OneHotType::new(chunk_size, width)?));
        }
    }
    types.push(PolynomialType::Dense(DenseType::Coefficients));
    Ok(types)
}

impl<F: CommitmentField> CommitmentStageProvider<F> for MetalCommitmentProvider<F> {
    fn stages<'a>(
        &'a self,
        expanded: &AkitaExpandedSetup<F>,
    ) -> Result<CommitmentStages<'a, F>, AkitaError> {
        if expanded.descriptor() != self.expanded.descriptor() {
            return Err(AkitaError::InvalidSetup(
                "the Metal commitment provider was prepared for a different setup".into(),
            ));
        }
        // Both stages share one owner, so the outer stage takes the device
        // image directly.
        let owner = StateOwnerCapability::new();
        let instance = BackendInstanceId::issue();
        let dimensions = StageDimensionCapabilities::new(RING_DEGREES.to_vec())?;
        let context = |name| {
            CommitmentOperationContext::new(
                expanded.descriptor(),
                instance,
                name,
                StageResources::none(),
            )
        };
        let inner = PreparedInnerCommitment::new(
            Arc::new(MetalInnerCommit::new(self, owner.clone())),
            owner.clone(),
            context("metal-inner")?,
            CommitmentRequestCapabilities::split::<MetalCommitContext>(
                BackendKindId::of::<MetalCommitKind>("metal")?,
                standard_types()?,
            ),
            dimensions.clone(),
            Some(Arc::new(MetalInnerExporter::<F>::new(owner.clone()))),
        )?;
        let outer = PreparedOuterCommitment::new(
            Arc::new(MetalOuterCommit::new(self, owner.clone())),
            owner,
            context("metal-outer")?,
            dimensions,
        );
        Ok(CommitmentStages {
            inner: Some(inner),
            outer: Some(outer),
        })
    }
}

impl From<AkitaMetalError> for AkitaError {
    fn from(error: AkitaMetalError) -> Self {
        AkitaError::InvalidSetup(format!("Metal commitment stage: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{inner_supported, outer_supported};
    use jolt_field::{Prime128OffsetA7F7, Prime64Offset59};

    type Fp128 = Prime128OffsetA7F7;
    type Fp64 = Prime64Offset59;

    #[test]
    fn production_plans_are_supported() {
        // fp128 dense nv26 and one-hot roots, fp64 dense nv16.
        assert!(inner_supported::<Fp128>(1024, 16));
        assert!(inner_supported::<Fp128>(256, 3));
        assert!(inner_supported::<Fp64>(1024, 13));
        assert!(outer_supported::<Fp128>(1024, 64, 3));
        assert!(outer_supported::<Fp64>(512, 128, 3));
    }

    #[test]
    fn plans_without_kernels_are_declined() {
        // Ring degrees without transform kernels.
        assert!(!inner_supported::<Fp64>(2048, 8));
        assert!(!inner_supported::<Fp128>(32, 8));
        assert!(!outer_supported::<Fp128>(2048, 2048, 3));
        // Digits wider than i16 (inner) or i8 (outer).
        assert!(!inner_supported::<Fp128>(512, 17));
        assert!(!outer_supported::<Fp128>(512, 64, 9));
        // A B degree above the A degree.
        assert!(!outer_supported::<Fp128>(64, 128, 3));
    }
}
