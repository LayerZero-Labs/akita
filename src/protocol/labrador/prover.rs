//! Labrador prover loop.

use crate::error::HachiError;
use crate::primitives::serialization::Compress;
use crate::protocol::labrador::comkey::LabradorComKeySeed;
use crate::protocol::labrador::config::{logq_bits, plan_fold, trivial_plan};
use crate::protocol::labrador::fold::prove_level;
use crate::protocol::labrador::guardrails::LABRADOR_MAX_LEVELS;
use crate::protocol::labrador::setup::LabradorSetup;
use crate::protocol::labrador::types::{LabradorProof, LabradorStatement, LabradorWitness};
use crate::protocol::labrador::LabradorReductionConfig;
use crate::protocol::proof::FlatLabradorWitness;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt, HachiSerialize};
use std::sync::Arc;

/// Build a recursive Labrador proof with optional tail acceptance.
///
/// Standard levels are applied while witness size decreases. Tail mode is then
/// attempted once and accepted only if total `(proof + witness)` size improves.
///
/// # Errors
///
/// Returns an error if folding fails or if recursion limits are exceeded.
#[tracing::instrument(skip_all, name = "labrador::prove")]
pub fn prove<F, T, const D: usize>(
    initial_witness: LabradorWitness<F, D>,
    initial_statement: &LabradorStatement<F, D>,
    comkey_seed: &LabradorComKeySeed,
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

        let plan = plan_fold::<F, D>(&witness, false)?;
        let cfg = plan.config;
        let rr: usize = plan.nu.iter().sum();
        let setup = Arc::new(LabradorSetup::new(&cfg, rr, plan.nn, comkey_seed));
        let fold = prove_level(
            &witness,
            &_statement,
            &cfg,
            &plan,
            &setup,
            level_idx,
            transcript,
        )?;
        let candidate_bits =
            transition_candidate_bits::<F, D>(&fold.level_proof, &fold.next_witness);
        if candidate_bits >= before_size {
            break;
        }
        levels.push(fold.level_proof);
        _statement = fold.statement;
        witness = fold.next_witness;
        level_idx += 1;
    }

    if level_idx + 1 < LABRADOR_MAX_LEVELS {
        let tail_plan = plan_fold::<F, D>(&witness, true)?;
        let tail_cfg = tail_plan.config;
        let baseline_bits = witness_size_bits::<F, D>(&witness);

        let rr: usize = tail_plan.nu.iter().sum();
        let tail_setup = Arc::new(LabradorSetup::new(&tail_cfg, rr, tail_plan.nn, comkey_seed));
        let mut tail_transcript = transcript.clone();
        if let Ok(tail) = prove_level(
            &witness,
            &_statement,
            &tail_cfg,
            &tail_plan,
            &tail_setup,
            level_idx,
            &mut tail_transcript,
        ) {
            let candidate_bits =
                transition_candidate_bits::<F, D>(&tail.level_proof, &tail.next_witness);
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
#[tracing::instrument(skip_all, name = "labrador::prove_with_config")]
pub fn prove_with_config<F, T, const D: usize>(
    initial_witness: LabradorWitness<F, D>,
    initial_statement: &LabradorStatement<F, D>,
    initial_config: &LabradorReductionConfig,
    comkey_seed: &LabradorComKeySeed,
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
    let initial_row_lengths: Vec<usize> = witness.rows().iter().map(|row| row.len()).collect();
    let initial_ring_elems: usize = initial_row_lengths.iter().sum();
    let initial_witness_bits = witness_size_bits::<F, D>(&witness);
    let initial_witness_bytes =
        FlatLabradorWitness::from_typed(&witness).serialized_size(Compress::No);
    tracing::debug!(
        ?initial_row_lengths,
        total_ring_elems = initial_ring_elems,
        witness_bits = initial_witness_bits,
        serialized_bytes = initial_witness_bytes,
        f = initial_config.f,
        b = initial_config.b,
        fu = initial_config.fu,
        bu = initial_config.bu,
        kappa = initial_config.kappa,
        kappa1 = initial_config.kappa1,
        tail = initial_config.tail,
        "labrador initial witness"
    );

    while level_idx + 1 < LABRADOR_MAX_LEVELS {
        let before_size = witness_size_bits::<F, D>(&witness);
        if before_size == 0 || witness.rows().len() <= 1 {
            break;
        }

        let plan = plan_fold::<F, D>(&witness, false).unwrap_or_else(|_| {
            let row_lengths: Vec<usize> = witness.rows().iter().map(|r| r.len()).collect();
            trivial_plan(fallback_cfg, &row_lengths)
        });
        let cfg = plan.config;
        let rr: usize = plan.nu.iter().sum();
        let setup = Arc::new(LabradorSetup::new(&cfg, rr, plan.nn, comkey_seed));

        let mut attempt_transcript = transcript.clone();
        let fold = prove_level(
            &witness,
            &statement,
            &cfg,
            &plan,
            &setup,
            level_idx,
            &mut attempt_transcript,
        )?;
        let next_witness_bits = witness_size_bits::<F, D>(&fold.next_witness);
        let level_bits = level_payload_size_bits::<F, D>(&fold.level_proof);
        let candidate_bits = level_bits + next_witness_bits;
        let rr: usize = plan.nu.iter().sum();
        tracing::debug!(
            current_bits = before_size,
            level_bits,
            next_witness_bits,
            candidate_bits,
            accept = candidate_bits < before_size,
            nn = plan.nn,
            rr,
            nu = ?plan.nu,
            f = cfg.f,
            b = cfg.b,
            fu = cfg.fu,
            bu = cfg.bu,
            kappa = cfg.kappa,
            kappa1 = cfg.kappa1,
            tail = cfg.tail,
            "labrador non-tail candidate"
        );
        if candidate_bits >= before_size {
            break;
        }

        *transcript = attempt_transcript;
        levels.push(fold.level_proof);
        statement = fold.statement;
        witness = fold.next_witness;
        fallback_cfg = cfg;
        level_idx += 1;
    }

    if level_idx + 1 < LABRADOR_MAX_LEVELS {
        let tail_plan = plan_fold::<F, D>(&witness, true).unwrap_or_else(|_| {
            let row_lengths: Vec<usize> = witness.rows().iter().map(|r| r.len()).collect();
            trivial_plan(
                LabradorReductionConfig {
                    tail: true,
                    kappa1: 0,
                    fu: 1,
                    bu: logq_bits::<F>(),
                    ..fallback_cfg
                },
                &row_lengths,
            )
        });
        let tail_cfg = tail_plan.config;
        let baseline_bits = witness_size_bits::<F, D>(&witness);

        let rr: usize = tail_plan.nu.iter().sum();
        let tail_setup = Arc::new(LabradorSetup::new(&tail_cfg, rr, tail_plan.nn, comkey_seed));
        let mut tail_transcript = transcript.clone();
        if let Ok(tail) = prove_level(
            &witness,
            &statement,
            &tail_cfg,
            &tail_plan,
            &tail_setup,
            level_idx,
            &mut tail_transcript,
        ) {
            let next_witness_bits = witness_size_bits::<F, D>(&tail.next_witness);
            let level_bits = level_payload_size_bits::<F, D>(&tail.level_proof);
            let candidate_bits = level_bits + next_witness_bits;
            tracing::debug!(
                baseline_bits,
                level_bits,
                next_witness_bits,
                candidate_bits,
                accept = candidate_bits < baseline_bits,
                nn = tail_plan.nn,
                rr,
                nu = ?tail_plan.nu,
                f = tail_cfg.f,
                b = tail_cfg.b,
                fu = tail_cfg.fu,
                bu = tail_cfg.bu,
                kappa = tail_cfg.kappa,
                kappa1 = tail_cfg.kappa1,
                tail = tail_cfg.tail,
                "labrador final tail compare"
            );
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
    let jl_bits = level.jl_projection.len() * 64;
    ring_bits + jl_bits + 64
}

fn transition_candidate_bits<F: FieldCore + CanonicalField, const D: usize>(
    level: &crate::protocol::labrador::LabradorLevelProof<F, D>,
    next_witness: &LabradorWitness<F, D>,
) -> usize {
    level_payload_size_bits::<F, D>(level) + witness_size_bits::<F, D>(next_witness)
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
            reduced_constraints: None,
            beta_sq: 1024,
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let proof = prove(sample_witness(), &statement, &[1u8; 32], &mut transcript).unwrap();
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
            reduced_constraints: None,
            beta_sq: 1 << 30,
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let proof = prove(sample_witness(), &statement, &[1u8; 32], &mut transcript).unwrap();

        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        crate::protocol::labrador::verify(&statement, &proof, &[1u8; 32], &mut verify_transcript)
            .unwrap();
    }
}
