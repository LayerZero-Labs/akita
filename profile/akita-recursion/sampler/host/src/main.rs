//! Bounded trace-only sampler benchmark; deliberately excludes jolt-sdk/host.
use akita_challenges::{SparseChallengeConfig, PRODUCTION_FOLD_CHALLENGE_RING_DIMS};
use common::{constants::RAM_START_ADDRESS, jolt_device::MemoryConfig};
use jolt_host::{GuestElf, Program};
use jolt_inlines_keccak256 as _;

struct Case {
    d: u32,
    pm1: u32,
    pm2: u32,
    rejection: bool,
    production: bool,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 4, "usage: host TARGET_DIR matrix SEEDS [production|stress|all], or TARGET_DIR SEED D PM1 PM2 DRAWS REJECTION");
    let mut program = Program::new("akita-sampler-guest");
    program.set_func("sampler_bench");
    program.set_profile(
        &std::env::var("SAMPLER_GUEST_PROFILE").unwrap_or_else(|_| "sampler-guest".into()),
    );
    program.set_std(true);
    program.set_stack_size(1048576);
    program.set_heap_size(16777216);
    program.set_max_input_size(256);
    program.set_max_output_size(64);
    program.build(&args[1]);
    let profile = program.instruction_profile();
    let elf = program.elf.expect("guest ELF");
    let image = jolt_program::image::decode_elf(elf.bytes(), profile).unwrap();
    let memory = MemoryConfig {
        max_input_size: 256,
        max_output_size: 64,
        stack_size: 1048576,
        heap_size: 16777216,
        program_size: Some(image.program_end - RAM_START_ADDRESS),
        ..MemoryConfig::default()
    };
    drop(image);
    if args[2] == "matrix" {
        let seeds: u64 = args[3].parse().unwrap();
        assert!((1..=256).contains(&seeds));
        let selection = args.get(4).map(String::as_str).unwrap_or("all");
        assert!(["production", "stress", "all"].contains(&selection));
        let production_draws: u32 = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(32);
        assert!((1..=1024).contains(&production_draws));
        for case in matrix() {
            if (selection == "production" && !case.production)
                || (selection == "stress" && case.production)
            {
                continue;
            }
            let weight = case.pm1 + case.pm2;
            // Bound sampled output size; use fewer accepted draws for the
            // expensive norm predicate. Execute-only mode retains no rows.
            let draws = if case.rejection {
                4
            } else if case.production {
                production_draws
            } else {
                (2048 / weight.max(1)).clamp(1, 16)
            };
            for index in 0..seeds {
                let seed = index.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(0x5eed);
                trace_case(&elf, &memory, seed, &case, draws);
            }
        }
    } else {
        assert_eq!(args.len(), 8);
        let seed = args[2].parse().unwrap();
        let case = Case {
            d: args[3].parse().unwrap(),
            pm1: args[4].parse().unwrap(),
            pm2: args[5].parse().unwrap(),
            rejection: args[7].parse().unwrap(),
            production: false,
        };
        trace_case(&elf, &memory, seed, &case, args[6].parse().unwrap());
    }
}

fn matrix() -> Vec<Case> {
    let mut cases = Vec::new();
    for &d in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
        let config = SparseChallengeConfig::production_for_ring_dim(d).unwrap();
        cases.push(Case {
            d: d as u32,
            pm1: config.count_pm1 as u32,
            pm2: config.count_pm2 as u32,
            rejection: false,
            production: true,
        });
    }
    cases.push(Case {
        d: 64,
        pm1: 31,
        pm2: 11,
        rejection: true,
        production: true,
    });
    cases.push(Case {
        d: 128,
        pm1: 31,
        pm2: 0,
        rejection: true,
        production: true,
    });
    for &d in PRODUCTION_FOLD_CHALLENGE_RING_DIMS {
        let mut weights = vec![
            0,
            1,
            d / 64,
            d / 64 + 1,
            16,
            31,
            32,
            41,
            42,
            48,
            49,
            63,
            64,
            65,
            d / 2,
            d,
        ];
        weights.retain(|&weight| weight <= d);
        weights.sort_unstable();
        weights.dedup();
        for weight in weights {
            let mut splits = vec![0, weight / 2, weight];
            splits.sort_unstable();
            splits.dedup();
            for pm1 in splits {
                let pm2 = weight - pm1;
                if cases.iter().any(|c| {
                    c.d == d as u32 && c.pm1 == pm1 as u32 && c.pm2 == pm2 as u32 && !c.rejection
                }) {
                    continue;
                }
                cases.push(Case {
                    d: d as u32,
                    pm1: pm1 as u32,
                    pm2: pm2 as u32,
                    rejection: false,
                    production: false,
                });
            }
        }
    }
    cases
}

fn trace_case(elf: &GuestElf, memory: &MemoryConfig, seed: u64, case: &Case, draws: u32) {
    let Case {
        d,
        pm1,
        pm2,
        rejection,
        production,
    } = *case;
    let expected = guest::sampler_bench(seed, d, pm1, pm2, draws, rejection);
    eprintln!("CASE seed={seed} D={d} pm1={pm1} pm2={pm2} draws={draws} rejection={rejection} production={production}");
    let input = postcard::to_stdvec(&(seed, d, pm1, pm2, draws, rejection)).unwrap();
    // Execute-only mode retains cycle accounting and public output without
    // allocating trace rows, bytecode expansions, or any prover state.
    let (trace_len, device, _) = tracer::execute(
        elf.bytes(),
        Some(elf.path()),
        &input,
        &[],
        &[],
        memory,
        None,
    );
    assert!(!device.panic, "guest panicked");
    let actual: [u64; 2] = postcard::from_bytes(&device.outputs).unwrap();
    assert_eq!(actual, expected, "guest/native mismatch");
    println!("seed={seed} D={d} pm1={pm1} pm2={pm2} draws={draws} rejection={rejection} production={production} trace={} digest={:016x}{:016x}", trace_len, actual[0], actual[1]);
}
