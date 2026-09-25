#![no_main]

//! Every target in one instrumented binary. `AKITA_FUZZ_TARGET` selects one
//! (the campaign always sets it); without it the first input byte picks the
//! target, so `cargo fuzz run fuzz_all` fuzzes all of them together. The
//! campaign ships this binary instead of one ~1.4 GiB sanitizer build per target.

use akita_fuzz::targets::{by_name, ALL};
use std::sync::OnceLock;

static TARGET: OnceLock<Option<fn(&[u8])>> = OnceLock::new();

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let selected = TARGET.get_or_init(|| {
        std::env::var("AKITA_FUZZ_TARGET").ok().map(|name| {
            by_name(&name).unwrap_or_else(|| panic!("unknown AKITA_FUZZ_TARGET {name}"))
        })
    });
    match (selected, data.split_first()) {
        (Some(run), _) => run(data),
        (None, Some((&index, rest))) => (ALL[usize::from(index) % ALL.len()].1)(rest),
        (None, None) => {}
    }
});
