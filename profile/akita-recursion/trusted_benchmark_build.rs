// Shared build-script bridge for host-selected benchmark inputs.

const TRUSTED_ARTIFACT_ENV: &str = "AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT";
const PREPARED_CACHE_ENV: &str = "AKITA_RECURSION_PREPARED_VERIFIER_CACHE";
const PREPARED_CACHE_SOURCE: &str = "prepared_verifier_cache.rs";
const MAX_PREPARED_CACHE_BYTES: u64 = 64 * 1024 * 1024;

fn write_prepared_cache_source() {
    println!("cargo:rerun-if-env-changed={PREPARED_CACHE_ENV}");
    let out_dir = std::path::PathBuf::from(
        std::env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR to the build script"),
    );
    let source_path = out_dir.join(PREPARED_CACHE_SOURCE);
    let source = match std::env::var_os(PREPARED_CACHE_ENV) {
        None => "pub static PROGRAM_BOUND_VERIFIER_CACHE: Option<&'static [u8]> = None;\n"
            .to_string(),
        Some(path) => {
            let path = std::path::PathBuf::from(path);
            let canonical = std::fs::canonicalize(&path).unwrap_or_else(|error| {
                panic!(
                    "failed to resolve prepared verifier cache `{}`: {error}",
                    path.display()
                )
            });
            let metadata = std::fs::metadata(&canonical).unwrap_or_else(|error| {
                panic!(
                    "failed to stat prepared verifier cache `{}`: {error}",
                    canonical.display()
                )
            });
            assert!(
                metadata.is_file() && metadata.len() <= MAX_PREPARED_CACHE_BYTES,
                "prepared verifier cache must be a regular file no larger than {MAX_PREPARED_CACHE_BYTES} bytes"
            );
            println!("cargo:rerun-if-changed={}", canonical.display());
            format!(
                "pub static PROGRAM_BOUND_VERIFIER_CACHE: Option<&'static [u8]> = Some(include_bytes!({:?}));\n",
                canonical.to_string_lossy()
            )
        }
    };
    std::fs::write(&source_path, source).unwrap_or_else(|error| {
        panic!(
            "failed to write generated cache source `{}`: {error}",
            source_path.display()
        )
    });
}

fn main() {
    println!("cargo:rerun-if-env-changed={TRUSTED_ARTIFACT_ENV}");
    println!("cargo:rustc-check-cfg=cfg(akita_trusted_benchmark_artifact)");

    match std::env::var(TRUSTED_ARTIFACT_ENV) {
        Ok(value) if value == "1" => println!("cargo:rustc-cfg=akita_trusted_benchmark_artifact"),
        Ok(value) => {
            panic!(
                "AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT must be exactly `1` when set, got `{value}`"
            );
        }
        Err(std::env::VarError::NotPresent) => {}
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT must be valid Unicode");
        }
    }
    write_prepared_cache_source();
}
