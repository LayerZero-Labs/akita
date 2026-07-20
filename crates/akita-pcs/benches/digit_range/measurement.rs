use super::cases::{BenchmarkCase, BenchmarkField as F, DigitDistribution};
use akita_transcript::{labels, AkitaTranscript};
use akita_verifier::AkitaStage1Verifier;
use criterion::black_box;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

struct CountingAllocator;

static MEASURE_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: this forwards the allocation request unchanged to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && MEASURE_ALLOCATIONS.load(Ordering::Relaxed) {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: this forwards the allocation request unchanged to the system allocator.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && MEASURE_ALLOCATIONS.load(Ordering::Relaxed) {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` are the pair supplied to this allocator by the caller.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: this forwards the reallocation request unchanged to the system allocator.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() && MEASURE_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        new_pointer
    }
}

fn record_allocation(bytes: usize) {
    ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

#[derive(Clone, Copy)]
struct AllocationMetrics {
    allocation_count: u64,
    allocated_bytes: u64,
}

fn start_allocation_measurement() {
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    MEASURE_ALLOCATIONS.store(true, Ordering::Relaxed);
}

fn finish_allocation_measurement() -> AllocationMetrics {
    MEASURE_ALLOCATIONS.store(false, Ordering::Relaxed);
    AllocationMetrics {
        allocation_count: ALLOCATION_COUNT.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

impl DigitDistribution {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "uniform" => Some(Self::Uniform),
            "zero-heavy" => Some(Self::ZeroHeavy),
            "alternating-endpoints" => Some(Self::AlternatingEndpoints),
            "seeded-high-entropy" => Some(Self::SeededHighEntropy),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
enum MeasurementPhase {
    Construct,
    Prove,
    ProveTotal,
    Verify,
}

impl MeasurementPhase {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "construct" => Some(Self::Construct),
            "prove" => Some(Self::Prove),
            "prove-total" => Some(Self::ProveTotal),
            "verify" => Some(Self::Verify),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Construct => "construct",
            Self::Prove => "prove",
            Self::ProveTotal => "prove-total",
            Self::Verify => "verify",
        }
    }
}

struct MeasurementCase {
    phase: MeasurementPhase,
    basis: usize,
    live_numerator: usize,
    live_name: &'static str,
    distribution: DigitDistribution,
}

impl MeasurementCase {
    fn parse(case: &str) -> Result<Self, String> {
        let parts = case.split('/').collect::<Vec<_>>();
        if parts.len() != 4 {
            return Err(format!(
                "measurement case must be <phase>/b<basis>/<live-prefix>/<distribution>; got {case}"
            ));
        }
        let phase = MeasurementPhase::parse(parts[0])
            .ok_or_else(|| format!("unsupported measurement phase: {}", parts[0]))?;
        let basis = parts[1]
            .strip_prefix('b')
            .and_then(|basis| basis.parse::<usize>().ok())
            .filter(|basis| matches!(basis, 4 | 8 | 16 | 32 | 64))
            .ok_or_else(|| format!("unsupported measurement basis: {}", parts[1]))?;
        let (live_numerator, live_name) = match parts[2] {
            "full" => (4, "full"),
            "three-quarters" => (3, "three-quarters"),
            _ => return Err(format!("unsupported live prefix: {}", parts[2])),
        };
        let distribution = DigitDistribution::parse(parts[3])
            .ok_or_else(|| format!("unsupported digit distribution: {}", parts[3]))?;
        Ok(Self {
            phase,
            basis,
            live_numerator,
            live_name,
            distribution,
        })
    }

    fn name(&self) -> String {
        format!(
            "{}/b{}/{}/{}",
            self.phase.name(),
            self.basis,
            self.live_name,
            self.distribution.name()
        )
    }
}

fn run(case: MeasurementCase) {
    if cfg!(feature = "parallel") {
        eprintln!(
            "allocation measurement requires --no-default-features so every allocation is observed on one thread"
        );
        std::process::exit(2);
    }

    let benchmark_case = BenchmarkCase::new(case.basis, case.live_numerator, case.distribution);
    let (elapsed_ns, metrics) = match case.phase {
        MeasurementPhase::Construct => {
            let prover_input = benchmark_case.prover_input();
            start_allocation_measurement();
            let started = Instant::now();
            let prover = prover_input.build();
            let elapsed = started.elapsed();
            let metrics = finish_allocation_measurement();
            black_box(prover);
            (elapsed.as_nanos(), metrics)
        }
        MeasurementPhase::Prove => {
            let prover = benchmark_case.prover_input().build();
            let mut transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
            start_allocation_measurement();
            let started = Instant::now();
            let output = prover
                .prove(&mut transcript)
                .expect("measurement proof succeeds");
            let elapsed = started.elapsed();
            let metrics = finish_allocation_measurement();
            black_box(output);
            (elapsed.as_nanos(), metrics)
        }
        MeasurementPhase::ProveTotal => {
            let prover_input = benchmark_case.prover_input();
            let mut transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
            start_allocation_measurement();
            let started = Instant::now();
            let output = prover_input
                .build()
                .prove(&mut transcript)
                .expect("measurement proof succeeds");
            let elapsed = started.elapsed();
            let metrics = finish_allocation_measurement();
            black_box(output);
            (elapsed.as_nanos(), metrics)
        }
        MeasurementPhase::Verify => {
            let mut prover_transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
            let (proof, _) = benchmark_case
                .prover_input()
                .build()
                .prove(&mut prover_transcript)
                .expect("measurement reference proof succeeds");
            let verifier = AkitaStage1Verifier::new(
                benchmark_case.equality_point.clone(),
                benchmark_case.plan,
            );
            let mut verifier_transcript = AkitaTranscript::<F>::new(labels::DOMAIN_AKITA_PROTOCOL);
            start_allocation_measurement();
            let started = Instant::now();
            let output = verifier
                .verify(&proof, &mut verifier_transcript)
                .expect("measurement verification succeeds");
            let elapsed = started.elapsed();
            let metrics = finish_allocation_measurement();
            black_box(output);
            (elapsed.as_nanos(), metrics)
        }
    };

    println!("case,live_elements,elapsed_ns,allocation_count,allocated_bytes");
    println!(
        "{},{},{elapsed_ns},{},{}",
        case.name(),
        benchmark_case.domain.live_len(),
        metrics.allocation_count,
        metrics.allocated_bytes,
    );
}

pub(super) fn run_requested() -> bool {
    let arguments = std::env::args().collect::<Vec<_>>();
    let Some(argument_index) = arguments
        .iter()
        .position(|argument| argument == "--measure-case")
    else {
        return false;
    };
    let case = arguments
        .get(argument_index + 1)
        .unwrap_or_else(|| {
            eprintln!("--measure-case requires a case name");
            std::process::exit(2);
        })
        .as_str();
    run(MeasurementCase::parse(case).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    }));
    true
}
