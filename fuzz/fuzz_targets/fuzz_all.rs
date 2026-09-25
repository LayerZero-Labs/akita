#![no_main]

//! Every target in one instrumented binary, selected by `AKITA_FUZZ_TARGET`.
//! The campaign ships this instead of one ~1.4 GiB sanitizer binary per target.

use std::sync::OnceLock;

static TARGET: OnceLock<fn(&[u8])> = OnceLock::new();

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let run = TARGET.get_or_init(|| {
        let name =
            std::env::var("AKITA_FUZZ_TARGET").expect("AKITA_FUZZ_TARGET must name a target");
        akita_fuzz::targets::by_name(&name)
            .unwrap_or_else(|| panic!("unknown AKITA_FUZZ_TARGET {name}"))
    });
    run(data);
});
