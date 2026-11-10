//! Clear eq-factored driver: native timing, scoped allocation/work counters,
//! and fixed symbols for external Linux Callgrind collection.
//!
//! Run `--counts` separately from Criterion. Allocation bytes count successful
//! allocation requests (including the full requested size of reallocations).
//! Coefficient exports/imports count copies through the canonical field codec,
//! including the public input claim. They exclude implicit Rust Copy assignments.
//! Callgrind: --collect-atstart=no --toggle-collect=akita_driver_eq_factored_prove
//! or --toggle-collect=akita_driver_clear_verify, with `--instruction-count`.

// The exported Jolt operator macros refer to num_traits; Jolt re-exports the
// exact Zero and One traits, avoiding another dependency just for this bench.
extern crate jolt_field as num_traits;

#[path = "sumcheck_driver/counts.rs"]
mod counts;
#[path = "sumcheck_driver/field.rs"]
mod field;
#[path = "sumcheck_driver/instance.rs"]
mod instance;

use akita_sumcheck::{prove_eq_factored_sumcheck, verify_eq_factored_sumcheck, SumcheckShape};
use akita_transcript::{new_prover_channel, new_verifier_channel};
use criterion::{BatchSize, Criterion};
use field::Counted;
use instance::{Instance, Prover, Verifier, ROUNDS, SESSION, WIDTH};
use jolt_field::{ExtField, Prime128OffsetA7F7 as F};
use std::hint::black_box;

#[global_allocator]
static ALLOCATOR: counts::CountingAllocator = counts::CountingAllocator;

fn prove<E: ExtField<F>>(instance: &mut Instance<E>, channel: &mut Prover) -> (Vec<E>, E) {
    prove_eq_factored_sumcheck::<F, E, _, _>(
        instance,
        channel,
        SumcheckShape::new(ROUNDS, WIDTH).unwrap(),
        0,
    )
    .unwrap()
}

fn verify<E: ExtField<F>>(instance: &Instance<E>, channel: &mut Verifier<'_>) -> Vec<E> {
    verify_eq_factored_sumcheck::<F, E, _, _>(
        &instance.tau,
        instance.claim(),
        SumcheckShape::new(ROUNDS, WIDTH).unwrap(),
        channel,
        0,
        |point| Ok(instance.expected(point)),
    )
    .unwrap()
}

// These functions measure native arithmetic, without the counting-field wrapper.
#[no_mangle]
#[inline(never)]
fn akita_driver_eq_factored_prove(instance: &mut Instance<F>, channel: &mut Prover) -> (Vec<F>, F) {
    prove(instance, channel)
}

#[no_mangle]
#[inline(never)]
fn akita_driver_clear_verify(instance: &Instance<F>, channel: &mut Verifier<'_>) -> Vec<F> {
    verify(instance, channel)
}

fn native_fixture() -> (Instance<F>, Vec<u8>, Vec<F>) {
    let instance = Instance::new();
    let mut prover = Prover(new_prover_channel(SESSION));
    let (point, output) = prove(&mut instance.clone(), &mut prover);
    assert_eq!(output, instance.expected(&point));
    let proof = prover.0.narg_string().to_vec();
    let mut verifier = Verifier(new_verifier_channel(SESSION, &proof));
    assert_eq!(verify(&instance, &mut verifier), point);
    verifier.0.check_eof().unwrap();
    (instance, proof, point)
}

fn work_counts() -> (counts::Counts, counts::Counts) {
    let instance = Instance::<Counted>::new();
    let mut prover = Prover(new_prover_channel(SESSION));
    let mut kernel = instance.clone();
    let ((point, output), mut proving) = counts::measure(|| prove(&mut kernel, &mut prover));
    assert_eq!(output, instance.expected(&point));
    let proof = prover.0.narg_string().to_vec();
    let mut verifier = Verifier(new_verifier_channel(SESSION, &proof));
    let (verified_point, mut verification) = counts::measure(|| verify(&instance, &mut verifier));
    assert_eq!(verified_point, point);
    verifier.0.check_eof().unwrap();
    // Count allocations with the native field separately: instrumentation must
    // not determine which allocations the optimizer can eliminate.
    let native = Instance::<F>::new();
    let mut native_kernel = native.clone();
    let mut native_prover = Prover(new_prover_channel(SESSION));
    let ((native_point, native_output), native_proving) =
        counts::measure(|| prove(&mut native_kernel, &mut native_prover));
    assert_eq!(native_output, native.expected(&native_point));
    let native_proof = native_prover.0.narg_string().to_vec();
    let mut native_verifier = Verifier(new_verifier_channel(SESSION, &native_proof));
    let (native_verified, native_verification) =
        counts::measure(|| verify(&native, &mut native_verifier));
    assert_eq!(native_verified, native_point);
    native_verifier.0.check_eof().unwrap();
    proving.allocations = native_proving.allocations;
    proving.allocated_bytes = native_proving.allocated_bytes;
    verification.allocations = native_verification.allocations;
    verification.allocated_bytes = native_verification.allocated_bytes;
    assert_eq!(proof, native_proof);
    assert_eq!(point.iter().map(|x| x.0).collect::<Vec<_>>(), native_point);
    assert!(proving.allocations > 0 && verification.allocations > 0);
    assert!(proving.mul > 0 && verification.mul > 0);
    // Each of 60 wire coefficients is exported once. The public claim adds one.
    assert_eq!(proving.coefficient_exports, (ROUNDS * WIDTH + 1) as u64);
    assert_eq!(verification.coefficient_imports, (ROUNDS * WIDTH) as u64);
    (proving, verification)
}

fn main() {
    if std::env::args().any(|arg| arg == "--counts") {
        let (proving, verification) = work_counts();
        let repeated = work_counts();
        assert_eq!(
            (&proving, &verification),
            (&repeated.0, &repeated.1),
            "work counts must repeat exactly"
        );
        println!("eq_factored_prove: {proving}");
        println!("clear_verify: {verification}");
        return;
    }
    let (instance, proof, point) = native_fixture();
    if std::env::args().any(|arg| arg == "--instruction-count") {
        let mut prover = Prover(new_prover_channel(SESSION));
        let (prover_point, claim) =
            akita_driver_eq_factored_prove(&mut instance.clone(), &mut prover);
        assert_eq!(claim, instance.expected(&prover_point));
        let mut verifier = Verifier(new_verifier_channel(SESSION, &proof));
        assert_eq!(akita_driver_clear_verify(&instance, &mut verifier), point);
        verifier.0.check_eof().unwrap();
        return;
    }
    let mut c = Criterion::default().configure_from_args();
    let mut group = c.benchmark_group("sumcheck_driver");
    group.bench_function("eq_factored_prove/20", |b| {
        b.iter_batched(
            || (instance.clone(), Prover(new_prover_channel(SESSION))),
            |(mut kernel, mut channel)| {
                let output = akita_driver_eq_factored_prove(&mut kernel, &mut channel);
                (kernel, channel, black_box(output))
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("clear_verify/20", |b| {
        b.iter_batched(
            || Verifier(new_verifier_channel(SESSION, &proof)),
            |mut channel| {
                let point = akita_driver_clear_verify(&instance, &mut channel);
                (channel, black_box(point))
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
    c.final_summary();
}
