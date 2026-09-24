//! Single-threaded binary forward-transform comparison, excluding table setup.
//! `cargo run -p akita-algebra --release --example binary_ntt_startup -- 16384 5`
use std::hint::black_box;
use std::time::Instant;

use akita_algebra::ntt::{tables::Q64_PRIMES, BinaryNttStrategy};
use akita_algebra::{CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut};

fn measure<const D: usize>(iterations: usize, samples: usize) {
    let params = CrtNttParamSet::new(Q64_PRIMES);
    let lut = DigitMontLut::new_with_digit_bound(&params, 2);
    let mut state = 0x929084723123abefu64;
    let input: Vec<[i8; D]> = (0..512)
        .map(|_| {
            std::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 1) as i8
            })
        })
        .collect();
    let modes = [
        ("generic", None),
        ("mask", Some(BinaryNttStrategy::Mask)),
        ("pairs", Some(BinaryNttStrategy::SelectedPairs)),
        ("selected4", Some(BinaryNttStrategy::Selected4)),
        ("split4", Some(BinaryNttStrategy::Split4)),
        ("split8", Some(BinaryNttStrategy::Split8)),
        ("compact4", Some(BinaryNttStrategy::Compact4)),
        ("positional4", Some(BinaryNttStrategy::Positional4)),
    ];
    let run = |index: usize, mode| {
        let digits = black_box(&input[index % input.len()]);
        black_box(match mode {
            None => CyclotomicCrtNtt::from_i8_with_lut(digits, &params, &lut),
            Some(strategy) => {
                CyclotomicCrtNtt::from_binary_with_lut(digits, &params, &lut, strategy)
                    .expect("binary benchmark input")
            }
        })
    };
    for (_, mode) in modes {
        for index in 0..512 {
            run(index, mode);
        }
    }
    for sample in 0..samples {
        for offset in 0..modes.len() {
            let (name, mode) = modes[(sample + offset) % modes.len()];
            let start = Instant::now();
            for index in 0..iterations {
                run(index, mode);
            }
            let ns = start.elapsed().as_nanos() as f64 / iterations as f64;
            println!("{D},{name},{sample},{iterations},{ns:.3}");
        }
    }
}

fn main() {
    // Keep the generic control free of binary auto-detection. Explicit modes
    // are selected through the public API in the same executable.
    std::env::set_var("AKITA_BINARY_NTT", "off");
    let args: Vec<_> = std::env::args().collect();
    let iterations = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16384);
    let samples = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    println!("degree,strategy,sample,iterations,ns_per_ring_three_primes");
    measure::<128>(iterations, samples);
    measure::<256>(iterations, samples);
    measure::<512>(iterations, samples);
    measure::<1024>(iterations, samples);
}
