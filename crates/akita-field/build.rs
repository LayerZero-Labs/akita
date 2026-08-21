use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(tool: &str, args: &[&str]) {
    let status = Command::new(tool)
        .args(args)
        .status()
        .unwrap_or_else(|error| panic!("failed to run {tool}: {error}"));
    assert!(status.success(), "{tool} failed with {status}");
}

fn compile_asm(compiler: &str, source: &Path, object: &Path, asm_dir: &Path, apple: bool) {
    let source = source.to_str().expect("UTF-8 assembly source path");
    let object = object.to_str().expect("UTF-8 object path");
    let include = asm_dir.to_str().expect("UTF-8 assembly include path");

    let mut args = vec!["-c", "-I", include];
    if apple {
        args.extend(["-arch", "arm64"]);
    }
    args.extend([source, "-o", object]);
    run(compiler, &args);
}

fn main() {
    let asm_dir = PathBuf::from("asm/aarch64");
    for file in [
        "fp128_add.S",
        "fp128_add_body.inc",
        "fp128_sub.S",
        "fp128_sub_body.inc",
        "fp128_mul.S",
        "fp128_mul_body.inc",
    ] {
        println!("cargo:rerun-if-changed={}", asm_dir.join(file).display());
    }

    if env::var_os("CARGO_FEATURE_FP128_ASM_EXPERIMENT").is_none() {
        return;
    }

    assert_eq!(
        env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
        Ok("aarch64"),
        "fp128-asm-experiment currently supports only AArch64"
    );

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".to_owned());
    let archiver = env::var("AR").unwrap_or_else(|_| "ar".to_owned());
    let apple = env::var("CARGO_CFG_TARGET_VENDOR").as_deref() == Ok("apple");

    let mut objects = Vec::new();
    for stem in ["fp128_add", "fp128_sub", "fp128_mul"] {
        let source = asm_dir.join(format!("{stem}.S"));
        let object = out_dir.join(format!("{stem}.o"));
        compile_asm(&compiler, &source, &object, &asm_dir, apple);
        objects.push(object);
    }

    let archive = out_dir.join("libakita_fp128_linkage.a");
    let archive_string = archive.to_str().expect("UTF-8 archive path");
    let mut archive_args = vec!["crs", archive_string];
    for object in &objects {
        archive_args.push(object.to_str().expect("UTF-8 object path"));
    }
    run(&archiver, &archive_args);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=akita_fp128_linkage");
}
