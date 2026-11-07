//! Experimental binary-source measurement inside one caller transcript.

#[cfg(feature = "labinius-trinomial")]
#[path = "binary_source_full_size/run.rs"]
mod run;

#[cfg(feature = "labinius-trinomial")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::thread::Builder::new()
        .name("binary-full-size".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| run::main().map_err(|error| error.to_string()))?
        .join()
        .map_err(|_| "measurement thread panicked")?
        .map_err(Into::into)
}

#[cfg(not(feature = "labinius-trinomial"))]
fn main() {
    eprintln!("binary_source_full_size requires --features labinius-trinomial");
    std::process::exit(2);
}
