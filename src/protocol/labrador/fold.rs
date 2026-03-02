//! Labrador amortization transitions (standard and tail levels).

use crate::algebra::ring::CyclotomicRing;
use crate::error::HachiError;
use crate::protocol::commitment::utils::linear::decompose_rows;
use crate::protocol::labrador::comkey::LabradorComKeySeed;
use crate::protocol::labrador::commit::commit_linear_only;
use crate::protocol::labrador::johnson_lindenstrauss::{
    collapse, project, zero_constant_term_for_proof,
};
use crate::protocol::labrador::transcript::{
    absorb_labrador_jl_nonce, absorb_labrador_jl_projection, absorb_labrador_level_context,
    LabradorLevelTranscriptContext,
};
use crate::protocol::labrador::types::{
    LabradorLevelProof, LabradorReductionConfig, LabradorStatement, LabradorWitness,
    LabradorWitnessRow,
};
use crate::protocol::prg::MatrixPrgBackendChoice;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{challenge_ring_element, Transcript};
use crate::{CanonicalField, FieldCore, FieldSampling, HachiSerialize};

/// Output of one Labrador fold transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorFoldResult<F: FieldCore, const D: usize> {
    /// Next witness after amortization.
    pub next_witness: LabradorWitness<F, D>,
    /// Replay-complete level proof record.
    pub level_proof: LabradorLevelProof<F, D>,
    /// Reduced statement consumed by the next verifier step.
    pub statement: LabradorStatement<F, D>,
}

/// Perform one standard (non-tail) Labrador fold.
///
/// # Errors
///
/// Returns `HachiError::InvalidInput` if `config.tail` is true, or propagates
/// errors from commitment, projection, or hashing.
pub fn standard_fold<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    config: &LabradorReductionConfig,
    comkey_seed: &LabradorComKeySeed,
    jl_seed: &[u8; 16],
    backend: MatrixPrgBackendChoice,
    level_index: usize,
    transcript: &mut T,
) -> Result<LabradorFoldResult<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    if config.tail {
        return Err(HachiError::InvalidInput(
            "standard_fold requires non-tail config".to_string(),
        ));
    }
    fold_impl(
        witness,
        config,
        comkey_seed,
        jl_seed,
        backend,
        level_index,
        false,
        transcript,
    )
}

/// Perform one tail Labrador fold.
///
/// # Errors
///
/// Returns `HachiError::InvalidInput` if `config.tail` is false, or propagates
/// errors from commitment, projection, or hashing.
pub fn tail_fold<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    config: &LabradorReductionConfig,
    comkey_seed: &LabradorComKeySeed,
    jl_seed: &[u8; 16],
    backend: MatrixPrgBackendChoice,
    level_index: usize,
    transcript: &mut T,
) -> Result<LabradorFoldResult<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    if !config.tail {
        return Err(HachiError::InvalidInput(
            "tail_fold requires tail config".to_string(),
        ));
    }
    fold_impl(
        witness,
        config,
        comkey_seed,
        jl_seed,
        backend,
        level_index,
        true,
        transcript,
    )
}

/// Core fold implementation following the C Labrador protocol phases:
///   1. Commit: inner + outer Ajtai commitment → u1
///   2. Project: JL projection → p[256], nonce
///   3. LIFTS × (collapse + lift): build linear constraints from JL
///   4. Amortize: absorb into transcript, sample ring-element challenges,
///      fold z = sum_i c_i * s_i, decompose z → output witness
#[allow(clippy::too_many_arguments)]
fn fold_impl<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    config: &LabradorReductionConfig,
    comkey_seed: &LabradorComKeySeed,
    jl_seed: &[u8; 16],
    backend: MatrixPrgBackendChoice,
    level_index: usize,
    tail_mode: bool,
    transcript: &mut T,
) -> Result<LabradorFoldResult<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    // Phase 1: Commit
    let artifacts = commit_linear_only(witness, config, comkey_seed, backend)?;

    // Phase 2: JL Projection
    let (jl_projection, jl_nonce) = project(witness, jl_seed, backend)?;

    // Phase 3: Lift (collapse JL → linear constraint, zero constant term for proof)
    let alpha = std::array::from_fn(|i| ((jl_projection[i] as i64) & 3) - 2);
    let target = collapse(&jl_projection, &alpha);
    let b_poly = scalar_to_constant_poly::<F, D>(target);
    let (b_transmitted, _b_constant) = zero_constant_term_for_proof(b_poly);

    let norm_sq: u128 = witness.rows.iter().map(|r| r.norm_sq).sum();

    // Transcript: absorb level context, commitments, JL, lift, norm.
    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index,
            tail: tail_mode,
            input_row_lengths: witness.rows.iter().map(|r| r.s.len()).collect(),
            input_row_chunks: witness.rows.iter().map(|_| 1usize).collect(),
            f: config.f,
            b: config.b,
            fu: config.fu,
            bu: config.bu,
            kappa: config.kappa,
            kappa1: config.kappa1,
            prg_backend_id: backend as u8,
        },
    )?;
    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &artifacts.u1);
    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &artifacts.u2);
    absorb_labrador_jl_projection(transcript, &jl_projection);
    absorb_labrador_jl_nonce(transcript, jl_nonce);

    let mut bb_bytes = Vec::new();
    vec![b_transmitted]
        .serialize_compressed(&mut bb_bytes)
        .map_err(|e| HachiError::InvalidInput(format!("serialize bb: {e}")))?;
    transcript.append_bytes(labels::ABSORB_LABRADOR_BB, &bb_bytes);
    transcript.append_bytes(
        labels::ABSORB_LABRADOR_NORM,
        &(norm_sq as u64).to_le_bytes(),
    );

    // Phase 4: Amortize — sample r challenge ring-elements from transcript, fold
    let r = witness.rows.len();
    let max_len = witness
        .rows
        .iter()
        .map(|row| row.s.len())
        .max()
        .unwrap_or(0);

    let challenges: Vec<CyclotomicRing<F, D>> = (0..r)
        .map(|_| challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AMORTIZE))
        .collect();

    let z = amortize_witness(witness, &challenges, max_len);
    let decomposed_z = decompose_rows(&z, config.f, config.b as u32);

    let mut output_rows = Vec::new();
    let chunk_size = if config.f > 0 {
        decomposed_z.len() / config.f
    } else {
        decomposed_z.len()
    };

    if config.f > 0 && chunk_size > 0 {
        for i in 0..config.f {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(decomposed_z.len());
            let row_data = decomposed_z[start..end].to_vec();
            let row_norm = row_data.iter().map(|x| x.coeff_norm_sq()).sum();
            output_rows.push(LabradorWitnessRow {
                s: row_data,
                norm_sq: row_norm,
            });
        }
    } else {
        let row_norm = decomposed_z.iter().map(|x| x.coeff_norm_sq()).sum();
        output_rows.push(LabradorWitnessRow {
            s: decomposed_z,
            norm_sq: row_norm,
        });
    }

    let next_witness = LabradorWitness { rows: output_rows };
    let out_norm_sq: u128 = next_witness.rows.iter().map(|r| r.norm_sq).sum();

    let level_proof = LabradorLevelProof {
        tail: tail_mode,
        input_row_lengths: witness.rows.iter().map(|r| r.s.len()).collect(),
        input_row_chunks: witness.rows.iter().map(|_| 1usize).collect(),
        config: *config,
        u1: artifacts.u1.clone(),
        u2: artifacts.u2.clone(),
        jl_projection,
        jl_nonce,
        bb: vec![b_transmitted],
        norm_sq: out_norm_sq,
    };

    let statement = LabradorStatement {
        u1: artifacts.u1,
        u2: artifacts.u2,
        constraints: Vec::new(),
        beta_sq: out_norm_sq,
        hash: [0u8; 16],
    };

    Ok(LabradorFoldResult {
        next_witness,
        level_proof,
        statement,
    })
}

/// Compute z = sum_i c_i * s_i (all-row linear combination).
fn amortize_witness<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
    challenges: &[CyclotomicRing<F, D>],
    max_len: usize,
) -> Vec<CyclotomicRing<F, D>> {
    let mut z = vec![CyclotomicRing::<F, D>::zero(); max_len];
    for (row, challenge) in witness.rows.iter().zip(challenges.iter()) {
        for (j, elem) in row.s.iter().enumerate() {
            z[j] += *challenge * *elem;
        }
    }
    z
}

fn scalar_to_constant_poly<F: FieldCore + CanonicalField, const D: usize>(
    value: i64,
) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
        if i == 0 {
            F::from_i64(value)
        } else {
            F::zero()
        }
    }))
}

/// Replay the transcript schedule for one Labrador level (verifier side).
///
/// Absorbs the same data the prover absorbed and squeezes the same challenges,
/// advancing the transcript state to stay in sync.
///
/// # Errors
///
/// Returns an error if the level context cannot be encoded.
pub fn replay_level_transcript<F, T, const D: usize>(
    level: &LabradorLevelProof<F, D>,
    level_index: usize,
    backend: MatrixPrgBackendChoice,
    transcript: &mut T,
) -> Result<(), HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling,
    T: Transcript<F>,
{
    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index,
            tail: level.tail,
            input_row_lengths: level.input_row_lengths.clone(),
            input_row_chunks: level.input_row_chunks.clone(),
            f: level.config.f,
            b: level.config.b,
            fu: level.config.fu,
            bu: level.config.bu,
            kappa: level.config.kappa,
            kappa1: level.config.kappa1,
            prg_backend_id: backend as u8,
        },
    )?;
    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &level.u1);
    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &level.u2);
    absorb_labrador_jl_projection(transcript, &level.jl_projection);
    absorb_labrador_jl_nonce(transcript, level.jl_nonce);

    let mut bb_bytes = Vec::new();
    level
        .bb
        .clone()
        .serialize_compressed(&mut bb_bytes)
        .map_err(|e| HachiError::InvalidInput(format!("serialize bb: {e}")))?;
    transcript.append_bytes(labels::ABSORB_LABRADOR_BB, &bb_bytes);
    transcript.append_bytes(
        labels::ABSORB_LABRADOR_NORM,
        &(level.norm_sq as u64).to_le_bytes(),
    );

    let r = level.input_row_lengths.len();
    for _ in 0..r {
        let _: CyclotomicRing<F, D> =
            challenge_ring_element(transcript, labels::CHALLENGE_LABRADOR_AMORTIZE);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::types::LabradorReductionConfig;
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn sample_witness() -> LabradorWitness<F, D> {
        let row = |len: usize| LabradorWitnessRow {
            s: (0..len)
                .map(|i| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|j| {
                        F::from_i64(((i + j) as i64 % 5) - 2)
                    }))
                })
                .collect(),
            norm_sq: 64,
        };
        LabradorWitness {
            rows: vec![row(4), row(4), row(2)],
        }
    }

    #[test]
    fn standard_fold_produces_decomposed_output() {
        let witness = sample_witness();
        let cfg = LabradorReductionConfig {
            f: 1,
            b: 8,
            fu: 2,
            bu: 10,
            kappa: 3,
            kappa1: 2,
            tail: false,
        };
        let seed = [1u8; 32];
        let jl_seed = [2u8; 16];
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let out = standard_fold(
            &witness,
            &cfg,
            &seed,
            &jl_seed,
            MatrixPrgBackendChoice::Shake256,
            0,
            &mut transcript,
        )
        .unwrap();
        assert!(
            !out.next_witness.rows.is_empty(),
            "fold must produce output witness"
        );
        assert!(!out.level_proof.u2.is_empty());
    }

    #[test]
    fn tail_fold_produces_decomposed_output() {
        let witness = sample_witness();
        let cfg = LabradorReductionConfig {
            f: 1,
            b: 8,
            fu: 1,
            bu: 32,
            kappa: 2,
            kappa1: 0,
            tail: true,
        };
        let seed = [3u8; 32];
        let jl_seed = [4u8; 16];
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let out = tail_fold(
            &witness,
            &cfg,
            &seed,
            &jl_seed,
            MatrixPrgBackendChoice::Shake256,
            1,
            &mut transcript,
        )
        .unwrap();
        assert!(
            !out.next_witness.rows.is_empty(),
            "tail fold must produce output"
        );
        assert!(out.level_proof.tail);
    }

    #[test]
    fn amortize_is_linear_combination() {
        let witness = sample_witness();
        let one = CyclotomicRing::<F, D>::one();
        let challenges = vec![one; witness.rows.len()];
        let max_len = witness.rows.iter().map(|r| r.s.len()).max().unwrap();
        let z = amortize_witness(&witness, &challenges, max_len);

        for j in 0..max_len {
            let expected = witness
                .rows
                .iter()
                .map(|row| {
                    row.s
                        .get(j)
                        .copied()
                        .unwrap_or_else(CyclotomicRing::<F, D>::zero)
                })
                .fold(CyclotomicRing::<F, D>::zero(), |a, b| a + b);
            assert_eq!(z[j], expected);
        }
    }
}
