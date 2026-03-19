//! Labrador prover loop.

use crate::error::HachiError;
use crate::primitives::serialization::Compress;
use crate::protocol::labrador::comkey::LabradorComKeySeed;
use crate::protocol::labrador::config::{
    estimate_fold_step, estimate_selected_fold_step, logq_bits, plan_fold, trivial_plan,
    LabradorFoldPlan,
};
use crate::protocol::labrador::fold::prove_level;
use crate::protocol::labrador::guardrails::LABRADOR_MAX_LEVELS;
use crate::protocol::labrador::setup::LabradorSetup;
use crate::protocol::labrador::types::{
    LabradorLevelProof, LabradorProof, LabradorStatement, LabradorWitness,
};
use crate::protocol::labrador::LabradorReductionConfig;
use crate::protocol::proof::{FlatLabradorLevelProof, FlatLabradorWitness};
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
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + HachiSerialize,
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
        let before_bytes = witness_size_bytes::<F, D>(&witness);
        if before_bytes == 0 || witness.rows().len() <= 1 {
            break;
        }

        let estimate = estimate_fold_step::<F, D>(&witness, false)?;
        if estimate.transition_bytes >= before_bytes {
            break;
        }
        let plan = estimate.plan;
        let cfg = plan.config;
        let virtual_row_count: usize = plan.row_split_counts.iter().sum();
        let setup = Arc::new(LabradorSetup::new(
            &cfg,
            virtual_row_count,
            plan.virtual_row_len,
            comkey_seed,
        ));
        let mut attempt_transcript = transcript.clone();
        let fold = prove_level(
            &witness,
            &_statement,
            &cfg,
            &plan,
            &setup,
            level_idx,
            &mut attempt_transcript,
        )?;
        let actual_level_bytes = level_size_bytes::<F, D>(&fold.level_proof);
        let actual_next_witness_bytes = witness_size_bytes::<F, D>(&fold.next_witness);
        let actual_candidate_bytes = actual_level_bytes + actual_next_witness_bytes;
        tracing::debug!(
            current_bytes = before_bytes,
            estimated_level_bytes = estimate.level_payload_bytes,
            estimated_next_witness_bytes = estimate.next_witness_bytes,
            estimated_candidate_bytes = estimate.transition_bytes,
            actual_level_bytes,
            actual_next_witness_bytes,
            actual_candidate_bytes,
            accept = actual_candidate_bytes < before_bytes,
            virtual_row_len = plan.virtual_row_len,
            virtual_row_count,
            row_split_counts = ?plan.row_split_counts,
            witness_digit_parts = cfg.witness_digit_parts,
            witness_digit_bits = cfg.witness_digit_bits,
            aux_digit_parts = cfg.aux_digit_parts,
            aux_digit_bits = cfg.aux_digit_bits,
            inner_commit_rank = cfg.inner_commit_rank,
            outer_commit_rank = cfg.outer_commit_rank,
            tail = cfg.tail,
            "labrador non-tail candidate"
        );
        if actual_candidate_bytes >= before_bytes {
            break;
        }
        levels.push(fold.level_proof);
        _statement = fold.statement;
        witness = fold.next_witness;
        *transcript = attempt_transcript;
        level_idx += 1;
    }

    if level_idx + 1 < LABRADOR_MAX_LEVELS {
        let baseline_bytes = witness_size_bytes::<F, D>(&witness);
        let tail_estimate = estimate_fold_step::<F, D>(&witness, true)?;
        if tail_estimate.transition_bytes >= baseline_bytes {
            return Ok(LabradorProof {
                levels,
                final_opening_witness: witness,
            });
        }
        let tail_plan = tail_estimate.plan;
        let tail_cfg = tail_plan.config;

        let virtual_row_count: usize = tail_plan.row_split_counts.iter().sum();
        let tail_setup = Arc::new(LabradorSetup::new(
            &tail_cfg,
            virtual_row_count,
            tail_plan.virtual_row_len,
            comkey_seed,
        ));
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
            let actual_level_bytes = level_size_bytes::<F, D>(&tail.level_proof);
            let actual_next_witness_bytes = witness_size_bytes::<F, D>(&tail.next_witness);
            let actual_candidate_bytes = actual_level_bytes + actual_next_witness_bytes;
            tracing::debug!(
                baseline_bytes,
                estimated_level_bytes = tail_estimate.level_payload_bytes,
                estimated_next_witness_bytes = tail_estimate.next_witness_bytes,
                estimated_candidate_bytes = tail_estimate.transition_bytes,
                actual_level_bytes,
                actual_next_witness_bytes,
                actual_candidate_bytes,
                accept = actual_candidate_bytes < baseline_bytes,
                virtual_row_len = tail_plan.virtual_row_len,
                virtual_row_count,
                row_split_counts = ?tail_plan.row_split_counts,
                witness_digit_parts = tail_cfg.witness_digit_parts,
                witness_digit_bits = tail_cfg.witness_digit_bits,
                aux_digit_parts = tail_cfg.aux_digit_parts,
                aux_digit_bits = tail_cfg.aux_digit_bits,
                inner_commit_rank = tail_cfg.inner_commit_rank,
                outer_commit_rank = tail_cfg.outer_commit_rank,
                tail = tail_cfg.tail,
                "labrador final tail compare"
            );
            if actual_candidate_bytes < baseline_bytes {
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

/// Build a recursive Labrador proof using a caller-supplied initial plan.
///
/// The initial plan is used for the first fold level. Later levels fall back to
/// the last accepted config if `plan_fold` fails.
///
/// # Errors
///
/// Returns [`HachiError`] if any fold level fails (e.g. empty witness,
/// invalid config, or transcript errors).
///
/// # Panics
///
/// Panics if estimating a trivial follow-on fold unexpectedly fails while
/// proving a previously accepted recursive step.
#[tracing::instrument(skip_all, name = "labrador::prove_with_plan")]
pub fn prove_with_plan<F, T, const D: usize>(
    initial_witness: LabradorWitness<F, D>,
    initial_statement: &LabradorStatement<F, D>,
    initial_plan: &LabradorFoldPlan,
    comkey_seed: &LabradorComKeySeed,
    transcript: &mut T,
) -> Result<LabradorProof<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt + HachiSerialize,
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
    let mut fallback_cfg = initial_plan.config;
    let initial_row_lengths: Vec<usize> = witness.rows().iter().map(|row| row.len()).collect();
    let initial_ring_elems: usize = initial_row_lengths.iter().sum();
    let initial_witness_bytes = witness_size_bytes::<F, D>(&witness);
    tracing::debug!(
        ?initial_row_lengths,
        total_ring_elems = initial_ring_elems,
        witness_bytes = initial_witness_bytes,
        serialized_bytes = initial_witness_bytes,
        virtual_row_len = initial_plan.virtual_row_len,
        row_split_counts = ?initial_plan.row_split_counts,
        witness_digit_parts = initial_plan.config.witness_digit_parts,
        witness_digit_bits = initial_plan.config.witness_digit_bits,
        aux_digit_parts = initial_plan.config.aux_digit_parts,
        aux_digit_bits = initial_plan.config.aux_digit_bits,
        inner_commit_rank = initial_plan.config.inner_commit_rank,
        outer_commit_rank = initial_plan.config.outer_commit_rank,
        tail = initial_plan.config.tail,
        "labrador initial witness"
    );

    while level_idx + 1 < LABRADOR_MAX_LEVELS {
        let before_bytes = witness_size_bytes::<F, D>(&witness);
        if before_bytes == 0 || witness.rows().len() <= 1 {
            break;
        }

        let estimate = if level_idx == 0 {
            estimate_selected_fold_step::<F, D>(&witness, initial_plan)?
        } else {
            estimate_fold_step::<F, D>(&witness, false).unwrap_or_else(|_| {
                let row_lengths: Vec<usize> = witness.rows().iter().map(|r| r.len()).collect();
                let plan = trivial_plan(fallback_cfg, &row_lengths);
                estimate_selected_fold_step::<F, D>(&witness, &plan)
                    .expect("trivial fold estimate must succeed")
            })
        };
        if estimate.transition_bytes >= before_bytes {
            break;
        }
        let plan = estimate.plan;
        let cfg = plan.config;
        let virtual_row_count: usize = plan.row_split_counts.iter().sum();
        let setup = Arc::new(LabradorSetup::new(
            &cfg,
            virtual_row_count,
            plan.virtual_row_len,
            comkey_seed,
        ));

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
        let actual_level_bytes = level_size_bytes::<F, D>(&fold.level_proof);
        let actual_next_witness_bytes = witness_size_bytes::<F, D>(&fold.next_witness);
        let actual_candidate_bytes = actual_level_bytes + actual_next_witness_bytes;
        tracing::debug!(
            current_bytes = before_bytes,
            estimated_level_bytes = estimate.level_payload_bytes,
            estimated_next_witness_bytes = estimate.next_witness_bytes,
            estimated_candidate_bytes = estimate.transition_bytes,
            actual_level_bytes,
            actual_next_witness_bytes,
            actual_candidate_bytes,
            accept = actual_candidate_bytes < before_bytes,
            virtual_row_len = plan.virtual_row_len,
            virtual_row_count,
            row_split_counts = ?plan.row_split_counts,
            witness_digit_parts = cfg.witness_digit_parts,
            witness_digit_bits = cfg.witness_digit_bits,
            aux_digit_parts = cfg.aux_digit_parts,
            aux_digit_bits = cfg.aux_digit_bits,
            inner_commit_rank = cfg.inner_commit_rank,
            outer_commit_rank = cfg.outer_commit_rank,
            tail = cfg.tail,
            "labrador non-tail candidate"
        );
        if actual_candidate_bytes >= before_bytes {
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
                    outer_commit_rank: 0,
                    aux_digit_parts: 1,
                    aux_digit_bits: logq_bits::<F>(),
                    ..fallback_cfg
                },
                &row_lengths,
            )
        });
        let baseline_bytes = witness_size_bytes::<F, D>(&witness);
        let tail_estimate = estimate_fold_step::<F, D>(&witness, true).unwrap_or_else(|_| {
            estimate_selected_fold_step::<F, D>(&witness, &tail_plan)
                .expect("tail trivial estimate must succeed")
        });
        if tail_estimate.transition_bytes >= baseline_bytes {
            return Ok(LabradorProof {
                levels,
                final_opening_witness: witness,
            });
        }
        let tail_cfg = tail_plan.config;

        let virtual_row_count: usize = tail_plan.row_split_counts.iter().sum();
        let tail_setup = Arc::new(LabradorSetup::new(
            &tail_cfg,
            virtual_row_count,
            tail_plan.virtual_row_len,
            comkey_seed,
        ));
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
            let actual_level_bytes = level_size_bytes::<F, D>(&tail.level_proof);
            let actual_next_witness_bytes = witness_size_bytes::<F, D>(&tail.next_witness);
            let actual_candidate_bytes = actual_level_bytes + actual_next_witness_bytes;
            tracing::debug!(
                baseline_bytes,
                estimated_level_bytes = tail_estimate.level_payload_bytes,
                estimated_next_witness_bytes = tail_estimate.next_witness_bytes,
                estimated_candidate_bytes = tail_estimate.transition_bytes,
                actual_level_bytes,
                actual_next_witness_bytes,
                actual_candidate_bytes,
                accept = actual_candidate_bytes < baseline_bytes,
                virtual_row_len = tail_plan.virtual_row_len,
                virtual_row_count,
                row_split_counts = ?tail_plan.row_split_counts,
                witness_digit_parts = tail_cfg.witness_digit_parts,
                witness_digit_bits = tail_cfg.witness_digit_bits,
                aux_digit_parts = tail_cfg.aux_digit_parts,
                aux_digit_bits = tail_cfg.aux_digit_bits,
                inner_commit_rank = tail_cfg.inner_commit_rank,
                outer_commit_rank = tail_cfg.outer_commit_rank,
                tail = tail_cfg.tail,
                "labrador final tail compare"
            );
            if actual_candidate_bytes < baseline_bytes {
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

fn witness_size_bytes<F: FieldCore + CanonicalField + HachiSerialize, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> usize {
    FlatLabradorWitness::from_typed(witness).serialized_size(Compress::No)
}

fn level_size_bytes<F: FieldCore + HachiSerialize, const D: usize>(
    level: &LabradorLevelProof<F, D>,
) -> usize {
    FlatLabradorLevelProof::from_typed(level).serialized_size(Compress::No)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::algebra::ring::CyclotomicRing;
    use crate::primitives::serialization::Compress;
    use crate::protocol::labrador::{verify, LabradorStatement};
    use crate::protocol::proof::FlatLabradorProof;
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_RECURSION;
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

    fn large_mixed_witness() -> LabradorWitness<F, D> {
        let row = |len: usize| -> Vec<CyclotomicRing<F, D>> {
            (0..len)
                .map(|i| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                        F::from_i64(((3 * i + 5 * j) as i64 % 9) - 4)
                    }))
                })
                .collect()
        };
        LabradorWitness::new_unchecked(vec![row(1376), row(1376), row(1280)])
    }

    #[test]
    fn prover_loop_returns_final_opening_witness() {
        let statement = LabradorStatement {
            inner_opening_payload: Vec::new(),
            linear_garbage_payload: Vec::new(),
            challenges: Vec::new(),
            constraints: Vec::new(),
            reduced_constraints: None,
            witness_norm_bound_sq: 1024,
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_RECURSION);
        let proof = prove(sample_witness(), &statement, &[1u8; 32], &mut transcript).unwrap();
        assert!(!proof.final_opening_witness.rows().is_empty());
        assert!(proof.levels.len() <= LABRADOR_MAX_LEVELS);
    }

    #[test]
    fn prover_proof_verifies() {
        let statement = LabradorStatement {
            inner_opening_payload: Vec::new(),
            linear_garbage_payload: Vec::new(),
            challenges: Vec::new(),
            constraints: Vec::new(),
            reduced_constraints: None,
            witness_norm_bound_sq: 1 << 30,
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_RECURSION);
        let proof = prove(sample_witness(), &statement, &[1u8; 32], &mut transcript).unwrap();

        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_RECURSION);
        verify(&statement, &proof, &[1u8; 32], &mut verify_transcript).unwrap();
    }

    #[test]
    fn prover_never_exceeds_no_recursion_baseline() {
        let witness = large_mixed_witness();
        let baseline_bytes = 4 + witness_size_bytes::<F, D>(&witness);
        let statement = LabradorStatement {
            inner_opening_payload: Vec::new(),
            linear_garbage_payload: Vec::new(),
            challenges: Vec::new(),
            constraints: Vec::new(),
            reduced_constraints: None,
            witness_norm_bound_sq: witness.norm().saturating_mul(256),
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_RECURSION);
        let proof = prove(witness, &statement, &[1u8; 32], &mut transcript).unwrap();
        let proof_bytes = FlatLabradorProof::from_typed(&proof).serialized_size(Compress::No);
        assert!(
            proof_bytes <= baseline_bytes,
            "proof bytes {proof_bytes} must stay below no-recursion baseline {baseline_bytes}"
        );
    }
}
