use super::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static REMAINING: Cell<Option<usize>> = const { Cell::new(None) };
    static ATTEMPTS: Cell<usize> = const { Cell::new(0) };
}

struct TestAllocator;
#[global_allocator]
static ALLOCATOR: TestAllocator = TestAllocator;

fn reject_allocation() -> bool {
    REMAINING
        .try_with(|remaining| match remaining.get() {
            None => false,
            Some(n) => {
                ATTEMPTS.with(|attempts| attempts.set(attempts.get() + 1));
                if n == 0 {
                    true
                } else {
                    remaining.set(Some(n - 1));
                    false
                }
            }
        })
        .unwrap_or(false)
}

// SAFETY: all successful allocations and all deallocations use System with the
// caller's original pointer/layout. Returning null is the allocator failure API;
// realloc failure leaves the original allocation untouched. Fault injection is
// confined to the current test thread and the explicitly guarded operation.
unsafe impl GlobalAlloc for TestAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if reject_allocation() {
            std::ptr::null_mut()
        } else {
            unsafe { System.alloc(layout) }
        }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if reject_allocation() {
            std::ptr::null_mut()
        } else {
            unsafe { System.alloc_zeroed(layout) }
        }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if reject_allocation() {
            std::ptr::null_mut()
        } else {
            unsafe { System.realloc(ptr, layout, size) }
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn with_allocation_budget<T>(allowed: usize, f: impl FnOnce() -> T) -> (T, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            REMAINING.with(|remaining| remaining.set(None));
        }
    }
    ATTEMPTS.with(|attempts| attempts.set(0));
    REMAINING.with(|remaining| remaining.set(Some(allowed)));
    let reset = Reset;
    let result = f();
    drop(reset);
    (result, ATTEMPTS.with(Cell::get))
}

#[test]
fn oversize_and_occupied_sessions_reject_without_allocating() {
    assert!(!active());
    let bytes = frame(&[7; 64], &[], &[]);
    let (result, attempts) =
        with_allocation_budget(0, || install_checked_hints(&bytes, bytes.len() - 1));
    assert!(result.is_err());
    assert_eq!(attempts, 0);
    assert!(!active());

    // The exact trusted bound is accepted; a rejected second install must not
    // allocate, replace the tape, or change its consumption cursor.
    install_checked_hints(&bytes, bytes.len()).unwrap();
    let (result, attempts) =
        with_allocation_budget(0, || install_checked_hints(&bytes, bytes.len()));
    assert!(result.is_err());
    assert_eq!(attempts, 0);
    take(HintTape::Quartets, 2, |words| {
        assert_eq!(words, &[u64::from_le_bytes([7; 8]); 8]);
        Ok(())
    })
    .unwrap();
    finish_checked_hints().unwrap();
}

#[test]
fn malformed_position_section_rejects_before_copying_quartets() {
    assert!(!active());
    let frame = frame(&[0; 8], &[], &[0; 1]);
    let (result, attempts) =
        with_allocation_budget(0, || install_checked_hints(&frame, frame.len()));
    assert!(result.is_err());
    assert_eq!(attempts, 0);
    assert!(!active());
}

#[test]
fn tape_allocation_failures_leave_no_partial_session() {
    #[repr(align(8))]
    struct Storage([u8; 104]);
    let words =
        |range: std::ops::RangeInclusive<u64>| range.flat_map(u64::to_le_bytes).collect::<Vec<_>>();
    let frame = frame(&words(1..=4), &words(5..=8), &words(9..=9));
    for offset in [0, 1] {
        let mut storage = Storage([0; 104]);
        storage.0[offset..offset + frame.len()].copy_from_slice(&frame);
        let bytes = &storage.0[offset..offset + frame.len()];
        for allowed in 0..3 {
            let (result, attempts) =
                with_allocation_budget(allowed, || install_checked_hints(bytes, bytes.len()));
            assert!(result.is_err());
            assert_eq!(attempts, allowed + 1);
            assert!(!active());
            install_checked_hints(bytes, bytes.len()).unwrap();
            for (tape, expected) in [
                (HintTape::Quartets, &[1, 2, 3, 4][..]),
                (HintTape::Ternary, &[5, 6, 7, 8][..]),
                (HintTape::Positions, &[9][..]),
            ] {
                take(tape, 1, |words| {
                    assert_eq!(words, expected);
                    Ok(())
                })
                .unwrap();
            }
            finish_checked_hints().unwrap();
        }
    }
}
