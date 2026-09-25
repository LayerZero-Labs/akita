//! Domain logic for Akita's fuzz targets, independent of the fuzzing engine.
//!
//! Each target is a `pub fn run(data: &[u8])` in [`targets`]. The libFuzzer
//! entry points in `fuzz_targets/` only forward to these functions, so another
//! engine or a plain replay driver can reuse the same generators and oracles.

pub mod env;
pub mod gen;
pub mod input;
pub mod oracle;
pub mod pcs;
pub mod stats;
pub mod targets;
