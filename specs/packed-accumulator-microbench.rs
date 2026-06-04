// D1 microbench for specs/packed-sumcheck.md (packed ProductAccum lane/reduce choice).
//
// Run:
//   rustc -C opt-level=3 -C target-cpu=native -C codegen-units=1 \
//     specs/packed-accumulator-microbench.rs -o /tmp/accbench && /tmp/accbench
//
// Observed (Apple Silicon / NEON, autovectorized scalar — directional, not hand-tuned
// intrinsics; in-cache and 64MB working sets gave identical numbers ⇒ compute-bound):
//   full-width:  eager 1.19  chunked-K3 0.74  u128-once 0.29   (ns/product)
//   small (a<2^8): eager 1.54  chunked-K3 0.80  u64-deferred 0.32  u128-once 0.29
// Conclusion: the cost is in-loop modular-reduction *frequency*, not lane width.
// Single-reduce wins (u128 full-width 0.29; u64 deferred for small operands 0.32).
// See packed-sumcheck.md D1. Re-run with real packed types/intrinsics on AVX2/AVX-512
// before locking the full-width-round lane choice.
//
// Directional microbench: packed-style lane accumulation strategies for a 31-bit
// pseudo-Mersenne field (Mersenne31, p = 2^31 - 1). Mimics W SIMD lanes laid out
// lane-major (matches the packed transpose layout). Autovectorized scalar; on
// aarch64 the u64-lane loops become NEON (2x uint64x2), u128 stays scalar+carry.
// Not hand-tuned intrinsics — meant to settle the *relative* question:
//   (1) is u128-per-lane accumulation much worse than u64-lane?
//   (2) do small operands (small-balanced digits / onehot) let us defer far longer
//       in u64 lanes, beating full-width chunked reduce?
use std::hint::black_box;
use std::time::Instant;

const P: u64 = (1u64 << 31) - 1;
const W: usize = 4; // NEON: 4x u64 = 2x uint64x2
const T: usize = 1 << 20; // products per lane
const N: usize = W * T;
const R: usize = 50;

#[inline(always)]
fn red(x: u64) -> u64 {
    let x = (x & P) + (x >> 31);
    let x = (x & P) + (x >> 31);
    let x = (x & P) + (x >> 31);
    if x >= P {
        x - P
    } else {
        x
    }
}

fn gen(small: bool) -> (Vec<u64>, Vec<u64>) {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut nxt = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let a: Vec<u64> = (0..N)
        .map(|_| if small { nxt() % 256 } else { nxt() % P })
        .collect();
    let b: Vec<u64> = (0..N).map(|_| nxt() % P).collect();
    (a, b)
}

// u64 lanes, reduce every k products
fn chunked(a: &[u64], b: &[u64], k: usize) -> u64 {
    let mut acc = [0u64; W];
    let mut part = [0u64; W];
    let mut cnt = 0usize;
    for j in 0..T {
        let base = j * W;
        for l in 0..W {
            part[l] += a[base + l] * b[base + l];
        }
        cnt += 1;
        if cnt == k {
            for l in 0..W {
                acc[l] = red(acc[l] + red(part[l]));
                part[l] = 0;
            }
            cnt = 0;
        }
    }
    let mut tot = 0u64;
    for l in 0..W {
        tot = red(tot + red(acc[l] + red(part[l])));
    }
    tot
}

// u64 lanes, single reduce at end (only valid when sum cannot overflow u64)
fn deferred(a: &[u64], b: &[u64]) -> u64 {
    let mut acc = [0u64; W];
    for j in 0..T {
        let base = j * W;
        for l in 0..W {
            acc[l] += a[base + l] * b[base + l];
        }
    }
    let mut tot = 0u64;
    for l in 0..W {
        tot = red(tot + red(acc[l]));
    }
    tot
}

// u128 lanes, single reduce at end
fn u128acc(a: &[u64], b: &[u64]) -> u64 {
    let mut acc = [0u128; W];
    for j in 0..T {
        let base = j * W;
        for l in 0..W {
            acc[l] += (a[base + l] as u128) * (b[base + l] as u128);
        }
    }
    let mut tot = 0u64;
    for l in 0..W {
        tot = red(tot + (acc[l] % P as u128) as u64);
    }
    tot
}

// reduce every product (no deferral)
fn eager(a: &[u64], b: &[u64]) -> u64 {
    let mut acc = [0u64; W];
    for j in 0..T {
        let base = j * W;
        for l in 0..W {
            acc[l] = red(acc[l] + red(a[base + l] * b[base + l]));
        }
    }
    let mut tot = 0u64;
    for l in 0..W {
        tot = red(tot + red(acc[l]));
    }
    tot
}

fn timeit<F: Fn(&[u64], &[u64]) -> u64>(name: &str, a: &[u64], b: &[u64], f: F) {
    let mut s = 0u64;
    for _ in 0..3 {
        s = s.wrapping_add(f(black_box(a), black_box(b)));
    } // warmup
    let t = Instant::now();
    for _ in 0..R {
        s = s.wrapping_add(f(black_box(a), black_box(b)));
    }
    let el = t.elapsed();
    let per = el.as_nanos() as f64 / (R as f64 * N as f64);
    println!(
        "{:<24} {:>8.4} ns/prod   (chk {})",
        name,
        per,
        black_box(s) & 0xff
    );
}

fn main() {
    let (af, bf) = gen(false);
    let (asm, bsm) = gen(true);
    println!("N={} per-lane T={} W={} R={}  p=2^31-1", N, T, W, R);
    println!("-- full-width operands (a,b < 2^31) --");
    timeit("full eager", &af, &bf, |a, b| eager(a, b));
    timeit("full chunked K=3", &af, &bf, |a, b| chunked(a, b, 3));
    timeit("full u128 once", &af, &bf, |a, b| u128acc(a, b));
    println!("-- small operand a<2^8 (balanced digit / onehot-like) --");
    timeit("small eager", &asm, &bsm, |a, b| eager(a, b));
    timeit("small chunked K=3", &asm, &bsm, |a, b| chunked(a, b, 3));
    timeit("small deferred once", &asm, &bsm, |a, b| deferred(a, b));
    timeit("small u128 once", &asm, &bsm, |a, b| u128acc(a, b));
}
