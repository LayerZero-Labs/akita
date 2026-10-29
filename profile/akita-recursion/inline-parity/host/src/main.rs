//! Traces the inline-parity guest under Jolt's emulator and checks it against
//! a native run of the same function on Akita's portable kernels.
//!
//! The run fails unless every guest inline Akita selects executes in the
//! trace, the guest selects the Jolt NTT backend at `D = 64` and the generic
//! path at `D = 128`, and every check digest matches the native reference.

use std::collections::BTreeMap;
use std::process::ExitCode;

use akita_algebra::ntt::I32_LAZY_DOT_BATCH;
use guest::{ParityReport, CHECKS};
use jolt_program::image::decode_elf;
use jolt_sdk::postcard;
// Jolt expands inline instructions through link-time `inventory`
// registrations; naming the crates keeps them in the binary.
use jolt_inlines_blake2 as _;
use jolt_inlines_keccak256 as _;
use jolt_inlines_ntt as _;

const SEED: u64 = 0x5eed_a417_a000_0001;
const CUSTOM_0: u8 = 0x0b;

/// Inline instructions the guest must execute, as `(name, funct3, funct7)`.
const EXPECTED_INLINES: [(&str, u32, u32); 4] = [
    (
        "blake2b compression",
        jolt_inlines_blake2::BLAKE2_FUNCT3,
        jolt_inlines_blake2::BLAKE2_FUNCT7,
    ),
    (
        "keccak absorb-permute",
        jolt_inlines_keccak256::KECCAK256_ABSORB_PERMUTE_FUNCT3,
        jolt_inlines_keccak256::KECCAK256_FUNCT7,
    ),
    (
        "ntt forward 64",
        jolt_inlines_ntt::FUNCT3,
        jolt_inlines_ntt::FUNCT7,
    ),
    // `jolt-inlines-ntt` keeps the dot's funct7 (9) private.
    ("ntt pointwise dot 64", jolt_inlines_ntt::FUNCT3, 9),
];

fn main() -> ExitCode {
    match run() {
        Ok(()) => {
            println!("inline parity OK");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("inline parity FAILED: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let target_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/akita-inline-parity-targets".to_string());

    // The reference must come from the portable kernels, not this host's SIMD.
    std::env::set_var("AKITA_SCALAR_NTT", "1");
    let native = guest::inline_parity(SEED);

    let program = guest::compile_inline_parity(&target_dir);
    let elf = program
        .get_elf_contents()
        .ok_or("guest build produced no ELF")?;
    let image = decode_elf(&elf, program.instruction_profile())
        .map_err(|err| format!("decode guest ELF: {err:?}"))?;
    let inline_sites: BTreeMap<usize, (u32, u32)> = image
        .instructions
        .iter()
        .filter_map(|instruction| {
            let row = instruction.row();
            let key = row.inline?;
            (key.opcode == CUSTOM_0)
                .then_some((row.address, (u32::from(key.funct3), u32::from(key.funct7))))
        })
        .collect();

    let input = postcard::to_stdvec(&SEED).map_err(|err| format!("encode input: {err}"))?;
    let summary = program.trace_analyze(&input, &[], &[]);
    if summary.io_device.panic {
        return Err("guest panicked".to_string());
    }
    let traced: ParityReport = postcard::from_bytes(&summary.io_device.outputs)
        .map_err(|err| format!("decode guest output: {err}"))?;
    println!("trace length: {}", summary.trace_len());

    let mut executions = BTreeMap::<(u32, u32), usize>::new();
    for row in summary.trace.rows() {
        let row = row.instruction();
        if row.is_first_in_sequence {
            if let Some(&key) = inline_sites.get(&row.address) {
                *executions.entry(key).or_default() += 1;
            }
        }
    }
    let mut failures = Vec::new();
    for (name, funct3, funct7) in EXPECTED_INLINES {
        let count = executions.get(&(funct3, funct7)).copied().unwrap_or(0);
        println!("inline {name}: {count} executions");
        if count == 0 {
            failures.push(format!("guest never executed the {name} inline"));
        }
    }

    println!(
        "dot batch sizes (D64, D128): guest {:?}, native {:?}",
        traced.dot_batch_sizes, native.dot_batch_sizes
    );
    if traced.dot_batch_sizes != [I32_LAZY_DOT_BATCH, 1] {
        failures.push("guest did not select the Jolt NTT backend for D64 only".to_string());
    }
    if native.dot_batch_sizes != [1, 1] {
        failures.push("native reference did not use the portable kernels".to_string());
    }

    if traced.digests.len() != CHECKS.len() || native.digests.len() != CHECKS.len() {
        return Err("report length does not match the check list".to_string());
    }
    for ((name, guest_digest), native_digest) in
        CHECKS.iter().zip(&traced.digests).zip(&native.digests)
    {
        let verdict = if guest_digest == native_digest {
            "ok"
        } else {
            failures.push(format!(
                "{name}: guest {guest_digest:016x} != native {native_digest:016x}"
            ));
            "MISMATCH"
        };
        println!("{verdict:>8}  {name}  {guest_digest:016x}");
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}
