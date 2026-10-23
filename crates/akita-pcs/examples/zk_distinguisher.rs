//! HT2, the two-witness distinguisher (`specs/whiteout-akita.md` §5.5): a
//! manual diagnostic of zero knowledge, not a CI test.
//!
//! It proves the single-group end-to-end statement (`tests/zk_end_to_end.rs`)
//! `N` times for each of two witnesses with the same public shape and point
//! (the claims are masked, so they need not agree), and runs a per-byte χ²
//! homogeneity test over every proof byte: for each byte position, a 2 × 256
//! contingency table of the two witnesses' byte values. A position is flagged
//! when its p-value is below `α / positions` (Bonferroni, `α = 0.01`).
//!
//! Two injected leaks calibrate its power at the same `N`: the first claim's
//! pad set to zero (the claim goes out in the clear) and that pad drawn with
//! only 64 random bits (its high bytes go out in the clear). An honest run
//! must flag nothing; each control must be flagged. Each configuration runs
//! `trials` times, and the report gives the detection count per
//! configuration.
//!
//! ```text
//! cargo run --release -p akita-pcs --no-default-features \
//!     --features parallel,test-support,logging --example zk_distinguisher -- [N] [trials]
//! ```

#![allow(missing_docs, dead_code)]

#[path = "../tests/common/mod.rs"]
mod common;
#[path = "../tests/zk_end_to_end/harness.rs"]
mod harness;
#[path = "../tests/masked_prefix/instance.rs"]
mod instance;

use akita_pcs::AkitaCommitmentScheme;
use harness::{Cheat, Harness, PadLeak};
use instance::*;
use jolt_algebra::Ring;

const ALPHA: f64 = 0.01;

/// `ln Γ(x)` for `x > 0` (Lanczos, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let t = x + 7.5;
    let series = G[1..]
        .iter()
        .enumerate()
        .fold(G[0], |sum, (i, g)| sum + g / (x + i as f64 + 1.0));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
}

/// The regularized upper incomplete gamma function `Q(a, x)`.
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        // Series for P(a, x).
        let (mut term, mut sum, mut n) = (1.0 / a, 1.0 / a, a);
        for _ in 0..10_000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - (sum.ln() - x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Continued fraction for Q(a, x) (modified Lentz).
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

/// The χ² homogeneity p-value of one byte position over two samples.
fn homogeneity_p_value(a: &[u32; 256], b: &[u32; 256]) -> f64 {
    let (na, nb) = (a.iter().sum::<u32>() as f64, b.iter().sum::<u32>() as f64);
    let total = na + nb;
    let mut statistic = 0.0;
    let mut categories = 0;
    for (&x, &y) in a.iter().zip(b) {
        let column = f64::from(x + y);
        if column == 0.0 {
            continue;
        }
        categories += 1;
        let (ea, eb) = (na * column / total, nb * column / total);
        statistic += (f64::from(x) - ea).powi(2) / ea + (f64::from(y) - eb).powi(2) / eb;
    }
    if categories < 2 {
        return 1.0;
    }
    gamma_q(f64::from(categories - 1) / 2.0, statistic / 2.0)
}

/// The flagged positions and the smallest p-value.
fn distinguish(first: &[Vec<u8>], second: &[Vec<u8>]) -> (usize, f64) {
    let positions = first.iter().chain(second).map(Vec::len).min().unwrap_or(0);
    let threshold = ALPHA / positions as f64;
    let counts = |proofs: &[Vec<u8>], position: usize| {
        let mut counts = [0u32; 256];
        for proof in proofs {
            counts[usize::from(proof[position])] += 1;
        }
        counts
    };
    let mut flagged = 0;
    let mut smallest = 1.0f64;
    for position in 0..positions {
        let p = homogeneity_p_value(&counts(first, position), &counts(second, position));
        smallest = smallest.min(p);
        if p < threshold {
            flagged += 1;
        }
    }
    (flagged, smallest)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: u64 = args.next().map_or(32, |arg| arg.parse().expect("N"));
    let trials: u64 = args.next().map_or(1, |arg| arg.parse().expect("trials"));
    let report = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(move || run(n, trials))
        .unwrap()
        .join()
        .unwrap();
    println!("{report}");
}

fn run(n: u64, trials: u64) -> String {
    // Two witnesses of one shape: the dense polynomial and an affine image
    // of it, whose claim differs in every byte (a shift alone would leave
    // the claim's high bytes equal, and hide the 64-bit control).
    let (evals, _) = dense_poly(NV);
    let other: Vec<F> = evals
        .iter()
        .map(|value| *value * F::from_u64(0x9e37_79b9_7f4a_7c15) + F::from_u64(17))
        .collect();
    let other_poly = akita_cpu_backend::DensePoly::<F>::from_field_evals(NV, &other).unwrap();
    let witnesses = [
        Harness::new(Instance::seeded(
            AkitaCommitmentScheme::new(zk_catalog()),
            1,
        )),
        Harness::new(Instance::with_polynomial(
            AkitaCommitmentScheme::new(zk_catalog()),
            2,
            (other, other_poly),
        )),
    ];
    let mut lines = vec![format!(
        "N = {n} proofs per witness, {trials} trial(s), α = {ALPHA}"
    )];
    for (label, leak) in [
        ("honest", None),
        ("pad 0", Some(PadLeak::Zero)),
        ("64-bit pad", Some(PadLeak::Bits64)),
    ] {
        let mut detections = 0;
        let mut smallest = 1.0f64;
        let started = std::time::Instant::now();
        for trial in 0..trials {
            let proofs = |witness: usize| {
                (0..n)
                    .map(|index| {
                        let seed = (trial << 40) | ((witness as u64) << 32) | index;
                        let cheat = Cheat {
                            leak,
                            ..Cheat::default()
                        };
                        witnesses[witness].prove(seed, cheat).unwrap().proof
                    })
                    .collect::<Vec<_>>()
            };
            let (flagged, p) = distinguish(&proofs(0), &proofs(1));
            smallest = smallest.min(p);
            if flagged > 0 {
                detections += 1;
            }
        }
        lines.push(format!(
            "{label}: detected in {detections}/{trials} trials, smallest p = {smallest:.3e}, {:.1?}",
            started.elapsed()
        ));
    }
    lines.join("\n")
}
