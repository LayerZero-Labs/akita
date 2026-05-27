// Shared build-script bridge for the host-driven trusted benchmark cfg.

fn main() {
    println!("cargo:rerun-if-env-changed=AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT");
    println!("cargo:rustc-check-cfg=cfg(akita_trusted_benchmark_artifact)");

    match std::env::var("AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT") {
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
}
