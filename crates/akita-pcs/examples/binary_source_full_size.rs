//! Experimental binary-source measurement inside one caller transcript.

#[cfg(feature = "binary-source-measurements")]
#[path = "binary_source_full_size/metrics.rs"]
mod metrics;

#[cfg(feature = "binary-source-measurements")]
type F = jolt_field::Prime128Offset275;
#[cfg(feature = "binary-source-measurements")]
type E = F;
#[cfg(feature = "binary-source-measurements")]
const PROFILE: akita_params::binary_setup::BinaryFieldProfile =
    akita_params::binary_setup::BinaryFieldProfile::P128;
#[cfg(feature = "binary-source-measurements")]
#[path = "binary_source_full_size/p64.rs"]
mod p64;

#[cfg(feature = "binary-source-measurements")]
#[path = "binary_source_full_size/run.rs"]
mod run;

#[cfg(feature = "binary-source-measurements")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::thread::Builder::new()
        .name("binary-full-size".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let p64 = std::env::args().collect::<Vec<_>>()
                .windows(2).any(|pair| pair == ["--field", "p64"]);
            let result = if p64 {
                p64::run::main()
            } else {
                run::main()
            };
            result.map_err(|error| error.to_string())
        })?
        .join()
        .map_err(|_| "measurement thread panicked")?
        .map_err(Into::into)
}

#[cfg(not(feature = "binary-source-measurements"))]
fn main() {
    eprintln!("binary_source_full_size requires --features binary-source-measurements");
    std::process::exit(2);
}
