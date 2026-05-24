//! Build-script bridge for the host-driven trusted benchmark cfg.

fn main() {
    println!("cargo:rerun-if-env-changed=AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT");
    println!("cargo:rustc-check-cfg=cfg(akita_trusted_benchmark_artifact)");

    if std::env::var_os("AKITA_RECURSION_TRUSTED_BENCHMARK_ARTIFACT").is_some() {
        println!("cargo:rustc-cfg=akita_trusted_benchmark_artifact");
    }
}
