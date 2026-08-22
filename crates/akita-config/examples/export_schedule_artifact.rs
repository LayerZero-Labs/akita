//! Export one compiled migration catalog as a trusted schedule artifact.

use akita_config::{trusted_schedule_catalog_from_embedded, TrustedScheduleCatalog};
use std::path::PathBuf;

fn usage() -> &'static str {
    "usage: cargo run -p akita-config --example export_schedule_artifact -- <family> <output.aks>"
}

fn compiled_catalog(family: &str) -> Result<TrustedScheduleCatalog, String> {
    match family {
        #[cfg(feature = "schedules-fp128-dense")]
        "fp128_dense" => {
            trusted_schedule_catalog_from_embedded::<akita_config::proof_optimized::fp128::Dense>()
                .map_err(|error| error.to_string())
        }
        #[cfg(feature = "schedules-fp128-onehot")]
        "fp128_onehot" => {
            trusted_schedule_catalog_from_embedded::<akita_config::proof_optimized::fp128::OneHot>()
                .map_err(|error| error.to_string())
        }
        #[cfg(feature = "schedules-fp32-dense")]
        "fp32_dense" => {
            trusted_schedule_catalog_from_embedded::<akita_config::proof_optimized::fp32::Dense>()
                .map_err(|error| error.to_string())
        }
        #[cfg(feature = "schedules-fp32-onehot")]
        "fp32_onehot" => {
            trusted_schedule_catalog_from_embedded::<akita_config::proof_optimized::fp32::OneHot>()
                .map_err(|error| error.to_string())
        }
        #[cfg(feature = "schedules-fp64-dense")]
        "fp64_dense" => {
            trusted_schedule_catalog_from_embedded::<akita_config::proof_optimized::fp64::Dense>()
                .map_err(|error| error.to_string())
        }
        #[cfg(feature = "schedules-fp64-onehot")]
        "fp64_onehot" => {
            trusted_schedule_catalog_from_embedded::<akita_config::proof_optimized::fp64::OneHot>()
                .map_err(|error| error.to_string())
        }
        _ => Err(format!(
            "schedule family {family:?} is unknown or its Cargo feature is disabled"
        )),
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let family = args.next().ok_or_else(|| usage().to_string())?;
    let output = PathBuf::from(args.next().ok_or_else(|| usage().to_string())?);
    if args.next().is_some() {
        return Err(usage().to_string());
    }
    let catalog = compiled_catalog(&family)?;
    let bytes = catalog
        .to_artifact_bytes()
        .map_err(|error| error.to_string())?;
    std::fs::write(&output, &bytes)
        .map_err(|error| format!("write {}: {error}", output.display()))?;
    println!(
        "wrote {} rows for {} to {} ({} bytes)",
        catalog.len(),
        catalog.family_name(),
        output.display(),
        bytes.len()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
