//! Experimental binary measurement inside one caller transcript.

// `p128` and `p64` each compile the same driver source with their field's types.
#![allow(clippy::duplicate_mod)]

#[path = "binary_full_size/config.rs"]
mod config;
#[path = "binary_full_size/metrics.rs"]
mod metrics;
#[path = "binary_full_size/p128.rs"]
mod p128;
#[path = "binary_full_size/p64.rs"]
mod p64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut field = "p128".to_owned();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--field" {
            field = args.next().ok_or("missing --field value")?;
        }
    }
    let run = match field.as_str() {
        "p128" => p128::run::main,
        "p64" => p64::run::main,
        _ => return Err("field must be p128 or p64".into()),
    };
    std::thread::Builder::new()
        .name("binary-full-size".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || run().map_err(|error| error.to_string()))?
        .join()
        .map_err(|_| "measurement thread panicked")?
        .map_err(Into::into)
}
