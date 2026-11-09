//! Generate the q24 commitment-prime width certificates and fixed-width cell
//! certificates as CSV files.
//!
//! Output goes to `crates/akita-sis-estimator/data` unless `--output-dir PATH`
//! names a comparison directory. The q25 files and runtime table are untouched.

use akita_sis_estimator::labinius_width_table::q24;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process,
    time::Instant,
};

fn main() {
    let mut args = env::args().skip(1);
    let output = match args.next().as_deref() {
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"),
        Some("--output-dir") => PathBuf::from(
            args.next()
                .unwrap_or_else(|| fatal("missing output directory")),
        ),
        Some(_) => fatal("usage: q24_infinity_width_table [--output-dir PATH]"),
    };
    if args.next().is_some() {
        fatal("unexpected argument");
    }
    let started = Instant::now();
    let widths = q24::generate_width_csv().unwrap_or_else(|error| fatal(&error.to_string()));
    eprintln!(
        "q24 width certificates completed in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    write_file(
        &output.join("q24_commitment_prime_infinity_width.csv"),
        &widths,
    );
    let started = Instant::now();
    let cells = q24::generate_cells_csv().unwrap_or_else(|error| fatal(&error.to_string()));
    write_file(&output.join("q24_infinity_cells.csv"), &cells);
    eprintln!(
        "q24 fixed-width certificates completed in {:.3}s",
        started.elapsed().as_secs_f64()
    );
}

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fatal(&format!("create output directory failed: {error}")));
    }
    fs::write(path, contents)
        .unwrap_or_else(|error| fatal(&format!("write {} failed: {error}", path.display())));
}

fn fatal(message: &str) -> ! {
    eprintln!("error: {message}");
    process::exit(2);
}
