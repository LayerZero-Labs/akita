//! Uninstrumented companion to the Akita fuzz targets.
//!
//! ```text
//! akita-fuzz-tool list                      target names known to the library
//! akita-fuzz-tool cases [LOG2_COST]         planned and excluded catalog cases
//! akita-fuzz-tool seeds OUT_DIR             deterministic seed corpora
//! akita-fuzz-tool smoke TARGET N [SEED]     N pseudo-random inputs, no libFuzzer
//! akita-fuzz-tool replay TARGET FILE...     run inputs once, report timings
//! ```

use akita_fuzz::input::SplitMix64;
use akita_fuzz::pcs::{Limits, Selector};
use akita_fuzz::targets;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("help");
    match command {
        "list" => {
            for (name, _) in targets::ALL {
                println!("{name}");
            }
        }
        "cases" => {
            let log2 = args
                .get(1)
                .and_then(|value| value.parse().ok())
                .unwrap_or(20u32);
            print!("{}", targets::pcs::describe_cases(log2));
        }
        "seeds" => seeds(Path::new(args.get(1).expect("seeds OUT_DIR"))),
        "smoke" => {
            let name = args.get(1).expect("smoke TARGET N [SEED]");
            let count: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(100);
            let seed: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
            let run = targets::by_name(name).unwrap_or_else(|| panic!("unknown target {name}"));
            let mut rng = SplitMix64::new(seed);
            let started = Instant::now();
            for index in 0..count {
                let len = (rng.next_u64() % 4096) as usize;
                let data: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
                let one = Instant::now();
                run(&data);
                if index < 3 || one.elapsed().as_secs_f64() > 5.0 {
                    eprintln!("input {index}: {:.3}s", one.elapsed().as_secs_f64());
                }
            }
            eprintln!(
                "{name}: {count} inputs in {:.2}s",
                started.elapsed().as_secs_f64()
            );
        }
        "replay" => {
            let name = args.get(1).expect("replay TARGET FILE...");
            let run = targets::by_name(name).unwrap_or_else(|| panic!("unknown target {name}"));
            for path in &args[2..] {
                let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
                let started = Instant::now();
                run(&data);
                eprintln!("{path}: ok in {:.3}s", started.elapsed().as_secs_f64());
            }
        }
        _ => {
            eprintln!("usage: akita-fuzz-tool list|cases|seeds|smoke|replay ...");
            std::process::exit(2);
        }
    }
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    std::fs::create_dir_all(dir).expect("create seed dir");
    std::fs::write(dir.join(name), bytes).expect("write seed");
}

fn random_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    (0..len).map(|_| rng.next_u64() as u8).collect()
}

fn seeds(out: &Path) {
    // Primitive targets: zeros plus deterministic random inputs of mixed sizes.
    for (name, _) in targets::ALL {
        let dir = out.join(name);
        write(&dir, "zeros", &[0u8; 512]);
        for index in 0..32u64 {
            let len = [64, 256, 1024, 4096][index as usize % 4];
            write(
                &dir,
                &format!("random-{index:02}"),
                &random_bytes(0x5eed_0000 + index, len),
            );
        }
    }

    // End-to-end targets: one seed family per planned case, selected by its
    // leading u16. The case list depends on the process cost limit, so these
    // mirror each target's default limit.
    let pcs: [(&str, Selector, u32); 6] = [
        ("pcs_dense", Selector::DenseSingle, 18),
        ("pcs_onehot", Selector::OneHotSingle, 20),
        ("pcs_batch", Selector::Batch, 20),
        ("pcs_recursive", Selector::Recursive, 21),
        ("pcs_reject", Selector::AnyDirect, 17),
        ("pcs_parallel", Selector::AnyDirect, 17),
    ];
    for (name, selector, log2) in pcs {
        let limits = Limits {
            max_cost: 1 << log2,
        };
        let cases = registry_for(limits).select(selector);
        for (index, _) in cases.iter().enumerate() {
            for variant in 0..3u64 {
                let mut bytes = (index as u16).to_le_bytes().to_vec();
                bytes.extend(random_bytes(
                    0xca5e_0000 ^ ((index as u64) << 8) ^ variant,
                    2048,
                ));
                if variant == 0 {
                    bytes.truncate(2 + 64);
                }
                write(
                    &out.join(name),
                    &format!("case-{index:03}-{variant}"),
                    &bytes,
                );
            }
        }
    }

    // Boundary targets run with a 2^16 cost limit; seed the verifier with
    // honest proofs and the deserializer with honest public objects.
    let registry = registry_for(Limits { max_cost: 1 << 16 });
    for (index, (family, case)) in registry.select(Selector::AnyDirect).iter().enumerate() {
        let family = &registry.families()[*family];
        let mut bytes = (index as u16).to_le_bytes().to_vec();
        bytes.extend([0u8, 1u8]);
        bytes.extend(family.fixture_proof(*case));
        write(
            &out.join("verifier_boundary"),
            &format!("honest-{index:03}"),
            &bytes,
        );
        for (object, encoded) in family.fixture_public_objects(*case).iter().enumerate() {
            write(
                &out.join("public_deserialize"),
                &format!("object-{index:03}-{object}"),
                encoded,
            );
        }
        let mut prover = (index as u16).to_le_bytes().to_vec();
        prover.extend(random_bytes(0xb0da_0000 ^ index as u64, 2048));
        write(
            &out.join("prover_boundary"),
            &format!("case-{index:03}"),
            &prover,
        );
    }

    let artifacts = akita_fuzz::env::artifacts_dir();
    for (selector, family) in targets::artifact::FAMILIES.iter().enumerate() {
        let path: PathBuf = artifacts.join(format!("{family}.aks"));
        let mut bytes = vec![selector as u8];
        bytes.extend(
            std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
        );
        write(&out.join("schedule_artifact"), family, &bytes);
    }
}

fn registry_for(limits: Limits) -> akita_fuzz::pcs::Registry {
    akita_fuzz::pcs::Registry::load(limits)
}
