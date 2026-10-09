#![cfg(all(feature = "labinius-binary", feature = "parallel"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use akita_algebra::binary::{BinaryField162 as F, PackedBinary162};

struct CountingAllocator;

static MEASURING: AtomicBool = AtomicBool::new(false);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);

fn count(bytes: usize) {
    if MEASURING.load(Ordering::SeqCst) {
        REQUESTED.fetch_add(bytes, Ordering::SeqCst);
    }
}

// SAFETY: every operation forwards its unchanged pointer and layout to System;
// the counters neither access allocated memory nor allocate themselves.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: the caller supplies a valid allocation layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: the caller supplies a valid allocation layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies a live System allocation and its layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: the caller supplies a live System allocation, its layout,
        // and a valid nonzero new size; all arguments are forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    REQUESTED.store(0, Ordering::SeqCst);
    MEASURING.store(true, Ordering::SeqCst);
    let result = operation();
    MEASURING.store(false, Ordering::SeqCst);
    (result, REQUESTED.load(Ordering::SeqCst))
}

#[test]
fn packed_parallel_kernels_allocate_less_than_64_kib() {
    const LEN: usize = 1 << 18;
    const LIMIT: usize = 64 * 1024;
    // Copying three limbs per fold requests 24*LEN = 6,291,456 bytes;
    // copying both operands per round requests 48*LEN = 12,582,912 bytes.
    // Either table-copying implementation exceeds this fixed bound.
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        // Wait for every worker's startup before measuring job bookkeeping.
        pool.broadcast(|_| ());
        let scalars: Vec<_> = (0..LEN)
            .map(|i| {
                let word = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                F::from_words([word, word.rotate_left(23), word >> 30]).unwrap()
            })
            .collect();
        let mut folded = PackedBinary162::from_scalars(&scalars);
        let lhs = folded.clone();
        let rhs = folded.clone();
        let r = scalars[17];
        // Initialize runtime dispatch outside the measurement as well.
        let mut warm = PackedBinary162::from_scalars(&scalars[..2]);
        let _ = warm.round_product(&warm, F::ZERO);
        warm.fold_in_place(r);

        let (_, fold_bytes) = measured(|| pool.install(|| folded.fold_in_place(r)));
        let (round, round_bytes) = measured(|| pool.install(|| lhs.round_product(&rhs, F::ZERO)));
        // Write directly so the allocation evidence remains visible with the
        // default test harness capture enabled; this occurs after measurement.
        writeln!(
            std::io::stderr(),
            "packed allocation: threads={threads}, fold={fold_bytes} bytes, round={round_bytes} bytes, limit={LIMIT} bytes"
        )
        .unwrap();
        assert_eq!(folded.len(), LEN / 2);
        assert!(round.is_some());
        assert!(fold_bytes < LIMIT, "fold requested {fold_bytes} bytes");
        assert!(round_bytes < LIMIT, "round requested {round_bytes} bytes");
    }
}
