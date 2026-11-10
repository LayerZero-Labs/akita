//! Experimental binary measurement inside one caller transcript.

#[path = "binary_full_size/catalog.rs"]
mod catalog;
#[path = "binary_full_size/claims.rs"]
mod claims;
#[path = "binary_full_size/config.rs"]
mod config;
#[path = "binary_full_size/metrics.rs"]
mod metrics;
#[path = "binary_full_size/native_catalog.rs"]
mod native_catalog;
#[path = "binary_full_size/run.rs"]
mod run;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut field = "p128".to_owned();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--field" {
            field = args.next().ok_or("missing --field value")?;
        }
    }
    let run: fn() -> run::Result<()> = match field.as_str() {
        "p128" => run::main::<config::p128::FullSize>,
        "p64" => run::main::<config::p64::FullSize>,
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
