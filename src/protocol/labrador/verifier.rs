//! Labrador verifier/reducer loop.

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::labrador::fold::replay_level_transcript;
use crate::protocol::labrador::guardrails::LABRADOR_MAX_LEVELS;
use crate::protocol::labrador::types::{
    LabradorConstraint, LabradorProof, LabradorStatement, LabradorWitness,
};
use crate::protocol::prg::MatrixPrgBackendChoice;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore, FieldSampling};

/// Output of verifier-side Labrador reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorVerifyResult<F: FieldCore, const D: usize> {
    /// Statement after replaying all reduction levels.
    pub terminal_statement: LabradorStatement<F, D>,
    /// Final clear opening witness from the proof payload.
    pub final_opening_witness: LabradorWitness<F, D>,
}

/// Verify Labrador recursive levels and return terminal reduction state.
///
/// Replays the prover's transcript schedule for each level to keep Fiat-Shamir
/// in sync, then checks structural consistency, norm bounds, and constraints.
///
/// # Errors
///
/// Returns [`HachiError::InvalidProof`] on structural inconsistencies,
/// norm bound violations, or constraint failures.
pub fn verify<F, T, const D: usize>(
    initial_statement: &LabradorStatement<F, D>,
    proof: &LabradorProof<F, D>,
    backend: MatrixPrgBackendChoice,
    transcript: &mut T,
) -> Result<LabradorVerifyResult<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    if proof.levels.len() > LABRADOR_MAX_LEVELS {
        return Err(HachiError::InvalidProof);
    }
    if proof.final_opening_witness.rows.is_empty() {
        return Err(HachiError::InvalidProof);
    }

    let mut current_statement = initial_statement.clone();
    let mut seen_tail = false;
    for (idx, level) in proof.levels.iter().enumerate() {
        if idx == 0 && (level.u1 != current_statement.u1 || level.u2 != current_statement.u2) {
            return Err(HachiError::InvalidProof);
        }
        if level.input_row_lengths.is_empty()
            || level.input_row_lengths.len() != level.input_row_chunks.len()
        {
            return Err(HachiError::InvalidProof);
        }
        if level.tail {
            if seen_tail || idx + 1 != proof.levels.len() {
                return Err(HachiError::InvalidProof);
            }
            seen_tail = true;
        }

        replay_level_transcript::<F, T, D>(level, idx, backend, transcript)?;

        current_statement = LabradorStatement {
            u1: level.u1.clone(),
            u2: level.u2.clone(),
            constraints: current_statement.constraints.clone(),
            beta_sq: level.norm_sq,
            hash: [0u8; 16],
        };
    }

    let final_norm = recompute_witness_norm(&proof.final_opening_witness);
    if final_norm > current_statement.beta_sq {
        return Err(HachiError::InvalidProof);
    }

    verify_constraints(&current_statement.constraints, &proof.final_opening_witness)?;

    Ok(LabradorVerifyResult {
        terminal_statement: current_statement,
        final_opening_witness: proof.final_opening_witness.clone(),
    })
}

fn recompute_witness_norm<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
) -> u128 {
    witness
        .rows
        .iter()
        .flat_map(|row| row.s.iter())
        .map(|ring| ring.coeff_norm_sq())
        .fold(0u128, |acc, v| acc.saturating_add(v))
}

fn verify_constraints<F: FieldCore + CanonicalField, const D: usize>(
    constraints: &[LabradorConstraint<F, D>],
    witness: &LabradorWitness<F, D>,
) -> Result<(), HachiError> {
    for (idx, cnst) in constraints.iter().enumerate() {
        let mut lhs = CyclotomicRing::<F, D>::zero();
        for (entry_idx, entry) in cnst.entries.iter().enumerate() {
            if entry.row >= witness.rows.len() {
                return Err(HachiError::InvalidProof);
            }
            let row = &witness.rows[entry.row].s;
            let mult = cnst.multiplicities.get(entry_idx).copied().unwrap_or(1);
            let coeffs = cnst
                .coefficients
                .get(entry_idx)
                .ok_or(HachiError::InvalidProof)?;
            let mut inner = CyclotomicRing::<F, D>::zero();
            for (j, coeff) in coeffs.iter().enumerate() {
                let witness_idx = entry.offset + j;
                let w_elem = row
                    .get(witness_idx)
                    .copied()
                    .unwrap_or_else(CyclotomicRing::<F, D>::zero);
                inner += *coeff * w_elem;
            }
            let mult_ring = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|k| {
                if k == 0 {
                    F::from_u64(mult as u64)
                } else {
                    F::zero()
                }
            }));
            lhs += mult_ring * inner;
        }

        let target = cnst
            .target
            .first()
            .copied()
            .unwrap_or_else(CyclotomicRing::<F, D>::zero);
        if lhs != target {
            return Err(HachiError::InvalidInput(format!(
                "Labrador constraint {idx} not satisfied"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::algebra::ring::CyclotomicRing;
    use crate::protocol::labrador::types::{
        LabradorConstraint, LabradorLevelProof, LabradorReductionConfig, LabradorWitnessRow,
    };
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
    use crate::protocol::transcript::Blake2bTranscript;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn sample_level() -> LabradorLevelProof<F, D> {
        LabradorLevelProof {
            tail: false,
            input_row_lengths: vec![4, 2],
            input_row_chunks: vec![1, 1],
            config: LabradorReductionConfig {
                f: 1,
                b: 8,
                fu: 2,
                bu: 10,
                kappa: 3,
                kappa1: 2,
                tail: false,
            },
            u1: vec![CyclotomicRing::one(), CyclotomicRing::one()],
            u2: vec![CyclotomicRing::one(), CyclotomicRing::one()],
            jl_projection: [0; 256],
            jl_nonce: 1,
            bb: vec![CyclotomicRing::zero()],
            norm_sq: 128,
        }
    }

    #[test]
    fn verify_accepts_structurally_valid_proof() {
        let level = sample_level();
        let statement = LabradorStatement {
            u1: level.u1.clone(),
            u2: level.u2.clone(),
            constraints: Vec::<LabradorConstraint<F, D>>::new(),
            beta_sq: level.norm_sq,
            hash: [0u8; 16],
        };
        let proof = LabradorProof {
            levels: vec![level],
            final_opening_witness: LabradorWitness {
                rows: vec![LabradorWitnessRow {
                    s: vec![CyclotomicRing::one()],
                    norm_sq: 64,
                }],
            },
        };
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let out = verify(
            &statement,
            &proof,
            MatrixPrgBackendChoice::Shake256,
            &mut transcript,
        )
        .unwrap();
        assert!(!out.final_opening_witness.rows.is_empty());
    }
}
