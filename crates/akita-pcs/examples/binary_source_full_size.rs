//! Experimental binary-source measurement inside one caller transcript.

#[cfg(feature = "binary-source-measurements")]
#[path = "binary_source_full_size/run.rs"]
mod run;

#[cfg(feature = "binary-source-measurements")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::thread::Builder::new()
        .name("binary-full-size".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| run::main().map_err(|error| error.to_string()))?
        .join()
        .map_err(|_| "measurement thread panicked")?
        .map_err(Into::into)
}

#[cfg(not(feature = "binary-source-measurements"))]
fn main() {
    eprintln!("binary_source_full_size requires --features binary-source-measurements");
    std::process::exit(2);
}
