//! Combined root sumcheck kernel at the two shapes of the sample geometry
//! `(22, 8, 128)`, on both proof-field pairs.
//!
//! | Instance | Variables | Digit slots | Weighted digits |
//! | --- | --- | --- | --- |
//! | response | 24 | 4 | 3 |
//! | image | 22 | 8 | 7 |
//!
//! Both tables put the digit slot innermost, then 1024 padded coefficients of
//! which 648 are live, then the entry. Padding holds digit zero and weight
//! zero. Live digits are uniform, which occupies every digit class; the
//! response of a real proof is more concentrated.
//!
//! Each run constructs the kernel and proves every round, with the inputs
//! cloned outside the timing. Every round is checked against its claim and the
//! last claim against `combined_terminal`, so a run is also a proof that the
//! verifier accepts. The harness prints the minimum and the median.
//!
//! ```text
//! cargo bench -p akita-labinius-prover --no-default-features \
//!     --features labinius,parallel,transcript-blake2b --bench combined_kernel
//! ```
//!
//! `RAYON_NUM_THREADS` selects the worker count. `COMBINED_KERNEL_RUNS`
//! (default 7) sets the timed runs after one warm-up, `COMBINED_KERNEL_ONLY`
//! keeps the cases whose name contains it, and `COMBINED_KERNEL_ROUNDS=1`
//! adds the median time of the constructor, of each round's message and of
//! each fold.

#![cfg(feature = "labinius")]

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use akita_labinius_prover::combined_kernel::CombinedRootKernel;
use akita_labinius_verifier::root_sumcheck::combined_terminal;
use akita_sumcheck::SumcheckInstanceProver;
use jolt_field::{Ext2, Field, Prime128Offset275, Prime64Offset59};
use rand::{rngs::StdRng, RngCore, SeedableRng};

/// Padded and live coefficients of one ring element.
const PADDED_COEFFICIENTS: usize = 1024;
const LIVE_COEFFICIENTS: usize = 648;

struct Shape {
    name: &'static str,
    num_vars: usize,
    digit_slots: usize,
    weighted_digits: usize,
}

const SHAPES: [Shape; 2] = [
    Shape {
        name: "response",
        num_vars: 24,
        digit_slots: 4,
        weighted_digits: 3,
    },
    Shape {
        name: "image",
        num_vars: 22,
        digit_slots: 8,
        weighted_digits: 7,
    },
];

struct Inputs<E> {
    digits: Vec<u8>,
    digit_factor: Vec<E>,
    compact: Vec<E>,
    tau: Vec<E>,
    beta: E,
    s: E,
    challenges: Vec<E>,
}

fn inputs<E: Field>(shape: &Shape, rng: &mut StdRng) -> Inputs<E> {
    let len = 1usize << shape.num_vars;
    let compact_len = len / shape.digit_slots;
    let mut digits = vec![0u8; len];
    let mut compact = vec![E::zero(); compact_len];
    let mut s = E::zero();
    for (index, (slots, weight)) in digits
        .chunks_exact_mut(shape.digit_slots)
        .zip(&mut compact)
        .enumerate()
    {
        if index % PADDED_COEFFICIENTS >= LIVE_COEFFICIENTS {
            continue;
        }
        let mut stored = 0u64;
        for (slot, digit) in slots.iter_mut().take(shape.weighted_digits).enumerate() {
            *digit = (rng.next_u32() % 16) as u8;
            stored |= u64::from(*digit) << (4 * slot);
        }
        *weight = E::random(rng);
        s += *weight * E::from_u64(stored);
    }
    let digit_factor = (0..shape.digit_slots)
        .map(|slot| {
            if slot < shape.weighted_digits {
                E::from_u64(1 << (4 * slot))
            } else {
                E::zero()
            }
        })
        .collect();
    Inputs {
        digits,
        digit_factor,
        compact,
        tau: (0..shape.num_vars).map(|_| E::random(rng)).collect(),
        beta: E::random(rng),
        s,
        challenges: (0..shape.num_vars).map(|_| E::random(rng)).collect(),
    }
}

/// One complete proof: total time, constructor time and, per round, the
/// message and fold times.
fn prove<E: Field>(inputs: &Inputs<E>) -> (Duration, Duration, Vec<[Duration; 2]>) {
    let digit_factor = inputs.digit_factor.clone();
    let compact = inputs.compact.clone();
    let start = Instant::now();
    let mut kernel = CombinedRootKernel::new(
        black_box(&inputs.digits),
        digit_factor,
        compact,
        &inputs.tau,
        inputs.beta,
        inputs.s,
    )
    .expect("benchmark inputs have valid geometry and digits");
    let setup = start.elapsed();
    let mut claim = kernel.input_claim();
    let mut rounds = Vec::with_capacity(inputs.challenges.len());
    for (round, &challenge) in inputs.challenges.iter().enumerate() {
        let before = Instant::now();
        let polynomial = kernel.compute_round_univariate(round, claim);
        let message = before.elapsed();
        assert!(
            polynomial.evaluate(E::zero()) + polynomial.evaluate(E::one()) == claim,
            "round {round} does not match its claim"
        );
        claim = polynomial.evaluate(challenge);
        let before = Instant::now();
        kernel.ingest_challenge(round, challenge);
        rounds.push([message, before.elapsed()]);
    }
    let evaluations = kernel.final_evaluations();
    let total = start.elapsed();
    let (w, k) = evaluations.expect("every challenge was bound");
    let terminal = combined_terminal(&inputs.tau, &inputs.challenges, inputs.beta, w, k)
        .expect("the point has the instance dimension");
    assert!(claim == terminal, "final claim fails the terminal check");
    black_box(kernel);
    (total, setup, rounds)
}

fn median(values: &mut [Duration]) -> Duration {
    values.sort();
    values.get(values.len() / 2).copied().unwrap_or_default()
}

fn case<E: Field>(pair: &str, shape: &Shape, runs: usize, per_round: bool) {
    let name = format!("{}/{pair}", shape.name);
    if std::env::var("COMBINED_KERNEL_ONLY").is_ok_and(|only| !name.contains(&only)) {
        return;
    }
    let mut rng = StdRng::seed_from_u64(0xc0_6b_1e + shape.num_vars as u64);
    let inputs = inputs::<E>(shape, &mut rng);
    prove(&inputs);
    let mut totals = Vec::with_capacity(runs);
    let mut setups = Vec::with_capacity(runs);
    let mut rounds = vec![[Vec::new(), Vec::new()]; shape.num_vars];
    for _ in 0..runs {
        let (total, setup, round_times) = prove(&inputs);
        totals.push(total);
        setups.push(setup);
        for (samples, times) in rounds.iter_mut().zip(round_times) {
            for (sample, time) in samples.iter_mut().zip(times) {
                sample.push(time);
            }
        }
    }
    let middle = median(&mut totals);
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let threads = 1;
    println!(
        "combined_kernel/{name} nu={} slots={} threads={threads} runs={runs} \
         min={:.3?} median={middle:.3?} max={:.3?}",
        shape.num_vars,
        shape.digit_slots,
        totals.first().copied().unwrap_or_default(),
        totals.last().copied().unwrap_or_default(),
    );
    if per_round {
        println!(
            "combined_kernel/{name} constructor={:>12.3?}",
            median(&mut setups)
        );
        for (round, [message, fold]) in rounds.iter_mut().enumerate() {
            println!(
                "combined_kernel/{name} round={round:>2} message={:>12.3?} fold={:>12.3?}",
                median(message),
                median(fold),
            );
        }
    }
}

fn main() {
    let runs = std::env::var("COMBINED_KERNEL_RUNS")
        .ok()
        .and_then(|runs| runs.parse().ok())
        .unwrap_or(7);
    let per_round = std::env::var("COMBINED_KERNEL_ROUNDS").is_ok_and(|rounds| rounds == "1");
    for shape in &SHAPES {
        case::<Prime128Offset275>("p128", shape, runs, per_round);
        case::<Ext2<Prime64Offset59>>("p64x2", shape, runs, per_round);
    }
}
