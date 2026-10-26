#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| akita_fuzz::targets::pcs::onehot(data));
