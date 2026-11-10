//! Experimental binary measurement inside one caller transcript.

#[path = "binary_full_size/run.rs"]
mod run;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::thread::Builder::new()
        .name("binary-full-size".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| run::main().map_err(|error| error.to_string()))?
        .join()
        .map_err(|_| "measurement thread panicked")?
        .map_err(Into::into)
}
