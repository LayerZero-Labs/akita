//! `akita-preprocess`: generate and check pinned preprocessing releases of
//! Akita's own configurations.

fn main() -> std::process::ExitCode {
    akita_preprocess::cli_main(&akita_preprocess::AkitaConfigs)
}
