//! Labrador prover loop.

use crate::error::HachiError;
use crate::protocol::labrador::comkey::LabradorComKeySeed;
use crate::protocol::labrador::config::logq_bits;
use crate::protocol::labrador::fold::prove_level;
use crate::protocol::labrador::guardrails::LABRADOR_MAX_LEVELS;
use crate::protocol::labrador::setup::LabradorSetup;
use crate::protocol::labrador::types::{LabradorProof, LabradorStatement, LabradorWitness};
use crate::protocol::labrador::LabradorReductionConfig;
use crate::protocol::labrador::{select_config, select_config_with_mode};
use crate::protocol::prg::MatrixPrgBackendChoice;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};

/// Build a recursive Labrador proof with optional tail acceptance.
///
/// Standard levels are applied while witness size decreases. Tail mode is then
/// attempted once and accepted only if total `(proof + witness)` size improves.
///
/// # Errors
///
/// Returns an error if folding fails or if recursion limits are exceeded.
pub fn prove<F, T, const D: usize>(
    initial_witness: LabradorWitness<F, D>,
    initial_statement: &LabradorStatement<F, D>,
    comkey_seed: &LabradorComKeySeed,
    jl_seed: &[u8; 16],
    backend: MatrixPrgBackendChoice,
    transcript: &mut T,
) -> Result<LabradorProof<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
    T: Transcript<F>,
{
    if initial_witness.rows().is_empty() {
        return Err(HachiError::InvalidInput(
            "cannot prove with empty Labrador witness".to_string(),
        ));
    }

    let mut levels = Vec::new();
    let mut witness = initial_witness;
    let mut _statement = initial_statement.clone();
    let mut level_idx = 0usize;

    while level_idx + 1 < LABRADOR_MAX_LEVELS {
        let before_size = witness_size_bits::<F, D>(&witness);
        if before_size == 0 || witness.rows().len() <= 1 {
            break;
        }

        let cfg = select_config(&witness)?;
        let r = witness.rows().len();
        let max_len = witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let setup = LabradorSetup::new(&cfg, r, max_len, comkey_seed, backend);
        let fold = prove_level(
            &witness,
            &_statement,
            &cfg,
            &setup,
            comkey_seed,
            jl_seed,
            backend,
            level_idx,
            transcript,
        )?;
        let after_size = witness_size_bits::<F, D>(&fold.next_witness);
        if after_size >= before_size {
            break;
        }
        levels.push(fold.level_proof);
        _statement = fold.statement;
        witness = fold.next_witness;
        level_idx += 1;
    }

    if level_idx + 1 < LABRADOR_MAX_LEVELS {
        let tail_cfg = select_config_with_mode(&witness, true)?;

        let baseline_bits = witness_size_bits::<F, D>(&witness)
            + levels
                .iter()
                .map(level_payload_size_bits::<F, D>)
                .sum::<usize>();

        // Clone transcript so we can roll back if tail doesn't help.
        let r = witness.rows().len();
        let max_len = witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let tail_setup = LabradorSetup::new(&tail_cfg, r, max_len, comkey_seed, backend);
        let mut tail_transcript = transcript.clone();
        if let Ok(tail) = prove_level(
            &witness,
            &_statement,
            &tail_cfg,
            &tail_setup,
            comkey_seed,
            jl_seed,
            backend,
            level_idx,
            &mut tail_transcript,
        ) {
            let candidate_bits = witness_size_bits::<F, D>(&tail.next_witness)
                + levels
                    .iter()
                    .map(level_payload_size_bits::<F, D>)
                    .sum::<usize>()
                + level_payload_size_bits::<F, D>(&tail.level_proof);
            if candidate_bits < baseline_bits {
                levels.push(tail.level_proof);
                _statement = tail.statement;
                witness = tail.next_witness;
                *transcript = tail_transcript;
            }
        }
    }

    Ok(LabradorProof {
        levels,
        final_opening_witness: witness,
    })
}

/// Build a recursive Labrador proof using a caller-supplied initial config.
///
/// Falls back to the provided config if `select_config` fails for a level.
///
/// # Errors
///
/// Returns [`HachiError`] if any fold level fails (e.g. empty witness,
/// invalid config, or transcript errors).
pub fn prove_with_config<F, T, const D: usize>(
    initial_witness: LabradorWitness<F, D>,
    initial_statement: &LabradorStatement<F, D>,
    initial_config: &LabradorReductionConfig,
    comkey_seed: &LabradorComKeySeed,
    jl_seed: &[u8; 16],
    backend: MatrixPrgBackendChoice,
    transcript: &mut T,
) -> Result<LabradorProof<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
    T: Transcript<F>,
{
    if initial_witness.rows().is_empty() {
        return Err(HachiError::InvalidInput(
            "cannot prove with empty Labrador witness".to_string(),
        ));
    }

    let mut levels = Vec::new();
    let mut witness = initial_witness;
    let mut statement = initial_statement.clone();
    let mut level_idx = 0usize;
    let mut fallback_cfg = *initial_config;
    let mut force_first_level = true;

    while level_idx + 1 < LABRADOR_MAX_LEVELS {
        let before_size = witness_size_bits::<F, D>(&witness);
        if before_size == 0 || witness.rows().len() <= 1 {
            break;
        }

        let cfg = select_config(&witness).unwrap_or(fallback_cfg);
        let r = witness.rows().len();
        let max_len = witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let setup = LabradorSetup::new(&cfg, r, max_len, comkey_seed, backend);
        let fold = prove_level(
            &witness,
            &statement,
            &cfg,
            &setup,
            comkey_seed,
            jl_seed,
            backend,
            level_idx,
            transcript,
        )?;
        let after_size = witness_size_bits::<F, D>(&fold.next_witness);
        if after_size >= before_size && !force_first_level {
            break;
        }

        levels.push(fold.level_proof);
        statement = fold.statement;
        witness = fold.next_witness;
        fallback_cfg = cfg;
        level_idx += 1;
        force_first_level = false;
    }

    if level_idx + 1 < LABRADOR_MAX_LEVELS {
        let tail_cfg =
            select_config_with_mode(&witness, true).unwrap_or_else(|_| LabradorReductionConfig {
                tail: true,
                kappa1: 0,
                fu: 1,
                bu: logq_bits::<F>(),
                ..fallback_cfg
            });

        let baseline_bits = witness_size_bits::<F, D>(&witness)
            + levels
                .iter()
                .map(level_payload_size_bits::<F, D>)
                .sum::<usize>();

        let r = witness.rows().len();
        let max_len = witness
            .rows()
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        let tail_setup = LabradorSetup::new(&tail_cfg, r, max_len, comkey_seed, backend);
        let mut tail_transcript = transcript.clone();
        if let Ok(tail) = prove_level(
            &witness,
            &statement,
            &tail_cfg,
            &tail_setup,
            comkey_seed,
            jl_seed,
            backend,
            level_idx,
            &mut tail_transcript,
        ) {
            let candidate_bits = witness_size_bits::<F, D>(&tail.next_witness)
                + levels
                    .iter()
                    .map(level_payload_size_bits::<F, D>)
                    .sum::<usize>()
                + level_payload_size_bits::<F, D>(&tail.level_proof);
            if candidate_bits < baseline_bits {
                levels.push(tail.level_proof);
                witness = tail.next_witness;
                *transcript = tail_transcript;
            }
        }
    }

    Ok(LabradorProof {
        levels,
        final_opening_witness: witness,
    })
}

fn witness_size_bits<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> usize {
    let logq_bits = logq_bits::<F>();
    witness
        .rows()
        .iter()
        .map(|row| row.len() * D * logq_bits)
        .sum()
}

fn level_payload_size_bits<F: FieldCore + CanonicalField, const D: usize>(
    level: &crate::protocol::labrador::LabradorLevelProof<F, D>,
) -> usize {
    let logq_bits = logq_bits::<F>();
    let ring_elems = level.u1.len() + level.u2.len() + level.bb.len();
    let ring_bits = ring_elems * D * logq_bits;
    let jl_bits = level.jl_projection.len() * 32;
    ring_bits + jl_bits + 64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::algebra::ring::CyclotomicRing;
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn sample_witness() -> LabradorWitness<F, D> {
        let row = |len: usize| -> Vec<CyclotomicRing<F, D>> {
            (0..len)
                .map(|i| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                        F::from_i64(((i + j) as i64 % 7) - 3)
                    }))
                })
                .collect()
        };
        LabradorWitness::new(vec![row(6), row(6), row(6)])
    }

    #[test]
    fn prover_loop_returns_final_opening_witness() {
        let statement = crate::protocol::labrador::types::LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: Vec::new(),
            beta_sq: 1024,
            hash: [0u8; 16],
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let proof = prove(
            sample_witness(),
            &statement,
            &[1u8; 32],
            &[2u8; 16],
            MatrixPrgBackendChoice::Shake256,
            &mut transcript,
        )
        .unwrap();
        assert!(!proof.final_opening_witness.rows().is_empty());
        assert!(proof.levels.len() <= LABRADOR_MAX_LEVELS);
    }

    #[test]
    fn prover_proof_verifies() {
        let statement = crate::protocol::labrador::types::LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: Vec::new(),
            beta_sq: 1 << 30,
            hash: [0u8; 16],
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let proof = prove(
            sample_witness(),
            &statement,
            &[1u8; 32],
            &[2u8; 16],
            MatrixPrgBackendChoice::Shake256,
            &mut transcript,
        )
        .unwrap();

        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        crate::protocol::labrador::verify(
            &statement,
            &proof,
            &[1u8; 32],
            &[2u8; 16],
            MatrixPrgBackendChoice::Shake256,
            &mut verify_transcript,
        )
        .unwrap();
    }
}
