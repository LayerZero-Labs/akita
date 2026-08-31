//! Mint a deployment setup identity once and report it for recording.
//!
//! Run this at deployment time, record the printed seed, and configure both the
//! prover and the verifier with it.
//!
//! ```text
//! cargo run -p akita-pcs --release --features disk-persistence --example generate_setup
//! ```
#![allow(missing_docs)]

use akita_config::proof_optimized::fp128;
use akita_pcs::AkitaCommitmentScheme;
use akita_types::{setup_seed_digest, AkitaSetupSeed};
use rand::rngs::OsRng;

type Config = fp128::Dense;

const MAX_NUM_VARS: usize = 14;
const MAX_NUM_BATCHED_POLYS: usize = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let setup_seed = AkitaSetupSeed::from_rng(&mut OsRng);

    println!("setup seed entropy: {}", hex(&setup_seed.seed));
    println!("derivation:         {:?}", setup_seed.derivation);
    println!(
        "setup seed digest:  {}",
        hex(&setup_seed_digest(&setup_seed)?)
    );

    let setup = AkitaCommitmentScheme::<Config>::setup_prover(
        MAX_NUM_VARS,
        MAX_NUM_BATCHED_POLYS,
        setup_seed,
    )?;

    println!(
        "materialized {} public field elements for max_num_vars={MAX_NUM_VARS}, max_num_batched_polys={MAX_NUM_BATCHED_POLYS}",
        setup.expanded.shared_matrix().num_field_elements()
    );
    println!();
    println!("Record the seed entropy above and configure both the prover and the");
    println!("verifier with it out of band; never read a setup seed from a proof.");

    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
