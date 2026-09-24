//! Single-threaded generic negacyclic NTT benchmark, excluding setup.
//!
//! Optional arguments specify transforms per sample (default 16384) and
//! samples (default 7).
use std::hint::black_box;
use std::time::Instant;

use akita_algebra::ntt::butterfly::forward_ntt;
use akita_algebra::ntt::tables::Q64_PRIMES;
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut, MontCoeff};

fn measure<const D: usize>(iterations: usize, samples: usize) {
    let params = CrtNttParamSet::<_, 3, D>::new(Q64_PRIMES);
    let lut = DigitMontLut::new_with_digit_bound(&params, 128);
    let mut state = 0x39128fa024ce7u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let signed: Vec<[i8; D]> = (0..128)
        .map(|_| std::array::from_fn(|_| next() as i8))
        .collect();
    let general: Vec<[[MontCoeff<i32>; D]; 3]> = (0..128)
        .map(|_| {
            std::array::from_fn(|k| {
                let p = i64::from(params.primes[k].p);
                std::array::from_fn(|_| {
                    MontCoeff::from_raw(((next() % (2 * p - 1) as u64) as i64 - p + 1) as i32)
                })
            })
        })
        .collect();
    let run = |index: usize, kind: usize| {
        if kind == 0 {
            black_box(CyclotomicCrtNtt::from_i8_with_lut(
                black_box(&signed[index % 128]),
                &params,
                &lut,
            ));
        } else {
            let mut data = *black_box(&general[index % 128]);
            for (k, limb) in data.iter_mut().enumerate() {
                forward_ntt(
                    limb,
                    params.primes[k],
                    &params.twiddles[k],
                    params.kernel_plan(),
                );
            }
            black_box(data);
        }
    };
    for kind in 0..2 {
        for index in 0..128 {
            run(index, kind);
        }
    }
    for sample in 0..samples {
        for offset in 0..2 {
            let kind = (sample + offset) % 2;
            let start = Instant::now();
            for index in 0..iterations {
                run(index, kind);
            }
            let ns = start.elapsed().as_nanos() as f64 / iterations as f64;
            let name = if kind == 0 {
                "signed_i8"
            } else {
                "general_montgomery"
            };
            println!("{D},{name},{sample},{iterations},{ns:.3}");
        }
    }
}

fn main() {
    if !cfg!(target_arch = "aarch64") || std::env::var("AKITA_SCALAR_NTT").as_deref() == Ok("1") {
        eprintln!("This benchmark requires AArch64 NEON with AKITA_SCALAR_NTT unset.");
        std::process::exit(1);
    }
    let args: Vec<_> = std::env::args().collect();
    let iterations = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16384);
    let samples = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    println!("degree,input,sample,iterations,ns_per_ring_three_primes");
    measure::<128>(iterations, samples);
    measure::<256>(iterations, samples);
    measure::<512>(iterations, samples);
    measure::<1024>(iterations, samples);
}
