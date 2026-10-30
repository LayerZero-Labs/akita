//! `akita-preprocess`: generate and check pinned preprocessing releases.

use akita_error::AkitaError;
use akita_params::sis::SisModulusProfileId;
use akita_preprocess::{
    check_release, generate, known_answer, release_dirs, write_release, ReleaseDescription, Role,
    DESCRIPTION_FILE, MANIFEST_FILE,
};
use akita_types::preprocessing::{hex, ReleaseManifest};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
usage:
  akita-preprocess generate <params/RELEASE> <OUT_DIR> [--role all|prover|verifier|TARGET] [--write-manifest]
  akita-preprocess check [<params/RELEASE>...]
  akita-preprocess anchor <params/RELEASE>
  akita-preprocess known-answer

Paths in a description are relative to the repository root, the current
directory unless --repo-root DIR is given.";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(mut args: Vec<String>) -> Result<(), AkitaError> {
    let repo_root =
        take_option(&mut args, "--repo-root")?.map_or_else(|| PathBuf::from("."), PathBuf::from);
    let usage = || AkitaError::InvalidInput(USAGE.into());
    let command = if args.is_empty() {
        return Err(usage());
    } else {
        args.remove(0)
    };
    match command.as_str() {
        "generate" => {
            let role =
                take_option(&mut args, "--role")?.map_or(Role::All, |role| Role::parse(&role));
            let write_manifest = take_flag(&mut args, "--write-manifest");
            let [release, out] = args.as_slice() else {
                return Err(usage());
            };
            let release = Path::new(release);
            let description = ReleaseDescription::read(&release.join(DESCRIPTION_FILE))?;
            let generated = generate(&repo_root, &description)?;
            write_release(Path::new(out), &generated, &role)?;
            if write_manifest {
                std::fs::write(release.join(MANIFEST_FILE), &generated.manifest).map_err(
                    |error| AkitaError::InvalidInput(format!("cannot write manifest: {error}")),
                )?;
            }
            println!("{} {}", description.name, generated.bundle_id);
            Ok(())
        }
        "check" => {
            let releases = if args.is_empty() {
                release_dirs(&repo_root.join("params"))?
            } else {
                args.iter().map(PathBuf::from).collect()
            };
            if releases.is_empty() {
                return Err(AkitaError::InvalidInput("no release to check".into()));
            }
            for release in releases {
                let bundle = check_release(&repo_root, &release)?;
                println!("ok {} {bundle}", release.display());
            }
            Ok(())
        }
        "anchor" => {
            let [release] = args.as_slice() else {
                return Err(usage());
            };
            let bytes = std::fs::read(Path::new(release).join(MANIFEST_FILE)).map_err(|error| {
                AkitaError::InvalidInput(format!("cannot read manifest: {error}"))
            })?;
            let bundle = akita_types::preprocessing::bundle_id(&bytes);
            for set in ReleaseManifest::decode(&bytes)?.sets {
                println!("AnchorEntry::new(\"{bundle}\", \"{}\"),", set.name);
            }
            Ok(())
        }
        "known-answer" => {
            for field in [
                SisModulusProfileId::Q32Offset99,
                SisModulusProfileId::Q64Offset59,
                SisModulusProfileId::Q128OffsetA7F7,
            ] {
                println!(
                    "{field:?} {}",
                    hex(&known_answer::compute_known_answer_digest(field)?)
                );
            }
            Ok(())
        }
        _ => Err(usage()),
    }
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    let before = args.len();
    args.retain(|arg| arg != flag);
    args.len() != before
}

fn take_option(args: &mut Vec<String>, option: &str) -> Result<Option<String>, AkitaError> {
    let Some(index) = args.iter().position(|arg| arg == option) else {
        return Ok(None);
    };
    if index + 1 >= args.len() {
        return Err(AkitaError::InvalidInput(format!("{option} needs a value")));
    }
    let value = args.remove(index + 1);
    args.remove(index);
    Ok(Some(value))
}
