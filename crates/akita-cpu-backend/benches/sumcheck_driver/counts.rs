//! Process-local counters, enabled only around the operation under measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static ADD: AtomicU64 = AtomicU64::new(0);
static SUB: AtomicU64 = AtomicU64::new(0);
static MUL: AtomicU64 = AtomicU64::new(0);
static NEG: AtomicU64 = AtomicU64::new(0);
static INV: AtomicU64 = AtomicU64::new(0);
static EXPORTS: AtomicU64 = AtomicU64::new(0);
static IMPORTS: AtomicU64 = AtomicU64::new(0);

pub(super) struct CountingAllocator;

fn allocation(bytes: usize) {
    if ACTIVE.load(Relaxed) {
        ALLOCATIONS.fetch_add(1, Relaxed);
        BYTES.fetch_add(bytes as u64, Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocation(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocation(layout.size());
        }
        ptr
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() {
            // Count each successful realloc as one request for its full new size.
            allocation(size);
        }
        new
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Counts {
    pub(super) allocations: u64,
    pub(super) allocated_bytes: u64,
    pub(super) add: u64,
    pub(super) sub: u64,
    pub(super) mul: u64,
    pub(super) neg: u64,
    pub(super) inverse: u64,
    pub(super) coefficient_exports: u64,
    pub(super) coefficient_imports: u64,
    pub(super) coefficient_codec_copies: u64,
}

impl std::fmt::Display for Counts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "allocations={} allocated_bytes={} add={} sub={} mul={} neg={} inverse={} coefficient_exports={} coefficient_imports={} coefficient_codec_copies={}",
            self.allocations, self.allocated_bytes, self.add, self.sub, self.mul,
            self.neg, self.inverse, self.coefficient_exports, self.coefficient_imports,
            self.coefficient_codec_copies,
        )
    }
}

pub(super) fn measure<T>(operation: impl FnOnce() -> T) -> (T, Counts) {
    let counters = [
        &ALLOCATIONS,
        &BYTES,
        &ADD,
        &SUB,
        &MUL,
        &NEG,
        &INV,
        &EXPORTS,
        &IMPORTS,
    ];
    for counter in counters {
        counter.store(0, Relaxed);
    }
    assert!(!ACTIVE.swap(true, Relaxed));
    let output = operation();
    ACTIVE.store(false, Relaxed);
    let [allocations, allocated_bytes, add, sub, mul, neg, inverse, coefficient_exports, coefficient_imports] =
        counters.map(|counter| counter.load(Relaxed));
    (
        output,
        Counts {
            allocations,
            allocated_bytes,
            add,
            sub,
            mul,
            neg,
            inverse,
            coefficient_exports,
            coefficient_imports,
            coefficient_codec_copies: coefficient_exports + coefficient_imports,
        },
    )
}

macro_rules! counter {
    ($name:ident, $counter:ident) => {
        pub(super) fn $name() {
            if ACTIVE.load(Relaxed) {
                $counter.fetch_add(1, Relaxed);
            }
        }
    };
}
counter!(add, ADD);
counter!(sub, SUB);
counter!(mul, MUL);
counter!(neg, NEG);
counter!(inverse, INV);
counter!(export, EXPORTS);
counter!(import, IMPORTS);
