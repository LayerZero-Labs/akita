//! Experimental binary-source measurement inside one caller transcript.

#[cfg(feature = "binary-source-measurements")]
#[path = "binary_source_full_size/metrics.rs"]
mod metrics;

#[cfg(feature = "binary-source-measurements")]
#[path = "binary_source_full_size/p64.rs"]
mod p64;

#[cfg(feature = "binary-source-measurements")]
#[path = "binary_source_full_size/run.rs"]
mod run;

#[cfg(feature = "binary-source-measurements")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use akita_params::binary_setup::BinaryFieldProfile;

    std::thread::Builder::new()
        .name("binary-full-size".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let mut field = BinaryFieldProfile::P128;
            let mut arguments = Vec::new();
            let mut args = std::env::args().skip(1);
            while let Some(arg) = args.next() {
                let value = args.next().ok_or("missing argument value")?;
                if arg == "--field" {
                    field = match value.as_str() {
                        "p64" => BinaryFieldProfile::P64,
                        "p128" => BinaryFieldProfile::P128,
                        _ => return Err(format!("unknown field {value}")),
                    };
                } else {
                    arguments.extend([arg, value]);
                }
            }
            let result = match field {
                BinaryFieldProfile::P64 => run::main::<run::config::P64>(arguments),
                BinaryFieldProfile::P128 => run::main::<run::config::P128>(arguments),
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
