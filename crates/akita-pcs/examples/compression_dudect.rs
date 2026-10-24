//! dudect-style timing check of the compression a Σ attempt runs on its
//! secret `t_D` before the accept bit (`specs/whiteout-akita.md` A36; an
//! intentional diagnostic, not a CI test).
//!
//! ```text
//! cargo run --release -p akita-pcs --no-default-features \
//!     --features parallel,zk-diagnostics --example compression_dudect -- \
//!     [measurements per test, default 10^5] [seed, default 0]
//! ```
//!
//! It times `ImageCompression::compress` of `AkitaResponseProver`, the call
//! the attempt makes (Akita's packing, the two maps' products and the
//! witness extraction), on images of `t_D`'s length at the W5-e geometry
//! (320 coefficients, five rings of 64) and of one full chunk (512). Each
//! test is a Welch t-test on raw timings, a fixed class against uniform
//! images, interleaved at random, every input written into slots allocated
//! once before its chunk is timed; the statistic is also computed on
//! measurements cropped at dudect's ten percentiles (from the first chunk),
//! and a test passes if every `|t| < 4.5`. Fixed classes: zero (every digit
//! of the first map zero), `−1` (one nonzero bit plane), `+1` (every bit
//! plane nonzero) and one fixed uniform image (the A/A control).
//!
//! As in the prover, every measurement runs on jolt-zk's DIT pool (A11).
//! Mirrors jolt-zk's `examples/sigma_dudect.rs`, whose statistics and slot
//! discipline this copies.

#![allow(clippy::print_stdout, clippy::unwrap_used)]

use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use akita_config::proof_optimized::fp128::ResponseDigits;
use akita_config::recursive_commitment::RecursiveCommitmentConfig;
use akita_config::{policy_of, CommitmentConfig};
use akita_params::{
    ChunkedWitnessCfg, CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout,
    ScheduleLookupKey,
};
use akita_pcs::zk::{AkitaResponseProver, CompressionShape};
use akita_pcs::AkitaCommitmentScheme;
use akita_schedules::ValidatedScheduleCatalog;
use jolt_algebra::{Field, One, Zero};
use jolt_zk::whiteout::diagnostics::with_dit;
use jolt_zk::whiteout::test_support::{SyntheticSigma, F};
use jolt_zk::whiteout::ImageCompression;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

type Cfg = RecursiveCommitmentConfig<ResponseDigits>;
const CHUNK: usize = 1_000;
const THRESHOLD: f64 = 4.5;
const CROPS: usize = 10;

/// Welch's t over two samples of timings, by running moments.
#[derive(Clone, Default)]
struct Welch {
    n: [f64; 2],
    mean: [f64; 2],
    m2: [f64; 2],
}

impl Welch {
    fn push(&mut self, class: usize, x: f64) {
        self.n[class] += 1.0;
        let delta = x - self.mean[class];
        self.mean[class] += delta / self.n[class];
        self.m2[class] += delta * (x - self.mean[class]);
    }

    fn t(&self) -> f64 {
        let var = |c: usize| self.m2[c] / (self.n[c] - 1.0) / self.n[c];
        (self.mean[0] - self.mean[1]) / (var(0) + var(1)).sqrt()
    }
}

/// The raw test and tests cropped at dudect's percentiles
/// `1 − 0.5^{10(k+1)/CROPS}`, thresholds from the first chunk.
#[derive(Default)]
struct Cropped {
    raw: Welch,
    thresholds: Vec<f64>,
    crops: Vec<Welch>,
}

impl Cropped {
    fn push(&mut self, class: usize, x: f64) {
        self.raw.push(class, x);
        for (threshold, test) in self.thresholds.iter().zip(&mut self.crops) {
            if x < *threshold {
                test.push(class, x);
            }
        }
    }

    fn set_thresholds(&mut self, first: &mut [f64]) {
        first.sort_by(f64::total_cmp);
        self.thresholds = (0..CROPS)
            .map(|k| {
                let q = 1.0 - 0.5f64.powf(10.0 * (k as f64 + 1.0) / CROPS as f64);
                first[((first.len() - 1) as f64 * q) as usize]
            })
            .collect();
        self.crops = vec![Welch::default(); CROPS];
    }

    /// The largest `|t|` over the raw and cropped tests, with its test
    /// (`None` for the raw one, else the crop index).
    fn max_t(&self) -> (f64, Option<usize>) {
        std::iter::once((None, &self.raw))
            .chain(
                self.crops
                    .iter()
                    .enumerate()
                    .map(|(k, test)| (Some(k), test)),
            )
            .filter(|(_, test)| test.n[0] > 100.0 && test.n[1] > 100.0)
            .map(|(k, test)| (test.t().abs(), k))
            .fold(
                (0.0, None),
                |best, next| if next.0 > best.0 { next } else { best },
            )
    }
}

fn report(name: &str, test: &Cropped, failures: &mut usize) {
    let (t, crop) = test.max_t();
    let pass = t < THRESHOLD;
    if !pass {
        *failures += 1;
    }
    println!(
        "  {name}: n = {:.0} / {:.0}, mean {:.0} / {:.0} ns, raw |t| = {:.2}, max |t| = {t:.2} ({}) {}",
        test.raw.n[0],
        test.raw.n[1],
        test.raw.mean[0],
        test.raw.mean[1],
        test.raw.t().abs(),
        crop.map_or("raw".to_owned(), |k| format!("crop {k}")),
        if pass { "PASS" } else { "LEAK?" }
    );
}

/// Times `compress` on `fixed` vs uniform images of its length. The images
/// live in `CHUNK` slots allocated once, so both classes are read from the
/// same addresses, and a uniform image is drawn for either class, so both
/// leave the generator in the same state (jolt-zk's slot discipline, A11).
fn fixed_vs_random(
    measurements: usize,
    rng: &mut StdRng,
    fixed: &[F],
    compress: impl Fn(&[F]),
) -> Cropped {
    let mut test = Cropped::default();
    let mut first = Vec::new();
    let mut slots: Vec<(usize, Vec<F>)> = (0..CHUNK)
        .map(|_| (0, vec![F::zero(); fixed.len()]))
        .collect();
    for chunk in 0..measurements.div_ceil(CHUNK) {
        for (class, image) in &mut slots {
            *class = (rng.next_u32() & 1) as usize;
            for (slot, fixed) in image.iter_mut().zip(fixed) {
                let random = F::random(rng);
                *slot = if *class == 0 { *fixed } else { random };
            }
        }
        for (class, image) in &slots {
            let start = Instant::now();
            compress(black_box(image));
            let elapsed = start.elapsed().as_nanos() as f64;
            if chunk == 0 {
                first.push(elapsed);
            }
            test.push(*class, elapsed);
        }
        if chunk == 0 {
            test.set_thresholds(&mut first);
        }
    }
    test
}

/// A response prover whose catalog holds the planner's response schedule
/// for `e` of the D = 64 synthetic Σ instance, as `examples/zk_response.rs`
/// builds it. Only its compression backend is used.
fn response_prover() -> AkitaResponseProver<Cfg> {
    let len = SyntheticSigma::<64>::new(1, 1).response_len(&CompressionShape::<Cfg>::default());
    let num_vars = len.next_power_of_two().trailing_zeros() as usize;
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
    let policy = policy_of::<Cfg>(chunking);
    let schedule = akita_planner::find_schedule(
        &key,
        Cfg::committed_source_contract().unwrap(),
        &[],
        &policy,
        Cfg::ring_challenge_config,
    )
    .unwrap()
    .schedule;
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )
        .unwrap(),
        precommitteds: Vec::new(),
    };
    let catalog = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        [(profiles, schedule)],
        &policy,
        Cfg::ring_challenge_config,
    )
    .unwrap();
    let scheme = AkitaCommitmentScheme::<Cfg>::from_schedule_artifact(
        &catalog.to_artifact_bytes().unwrap(),
        chunking,
    )
    .unwrap();
    let setup = AkitaResponseProver::setup(&scheme, num_vars).unwrap();
    AkitaResponseProver::new(scheme, setup).unwrap()
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let measurements: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(100_000);
    let seed: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(0);
    let prover = response_prover();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut failures = 0;
    with_dit(|| {
        println!("seed {seed}, {measurements} measurements per test:");
        for len in [320, 512] {
            println!(" image of {len} coefficients:");
            let uniform: Vec<F> = (0..len).map(|_| F::random(&mut rng)).collect();
            for (label, fixed) in [
                ("0", vec![F::zero(); len]),
                ("−1", vec![-F::one(); len]),
                ("+1", vec![F::one(); len]),
                ("uniform (A/A)", uniform),
            ] {
                let test = fixed_vs_random(measurements, &mut rng, &fixed, |image| {
                    let _ = black_box(prover.compress(image).unwrap());
                });
                report(&format!("compress, fixed {label}"), &test, &mut failures);
            }
        }
    })
    .unwrap();
    println!("{failures} test(s) above |t| = {THRESHOLD}");
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
