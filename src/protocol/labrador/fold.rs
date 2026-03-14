//! Labrador amortization transitions (standard and tail levels).

use crate::algebra::ring::CyclotomicRing;
use crate::algebra::SparseChallenge;
use crate::error::HachiError;
#[cfg(feature = "parallel")]
use crate::parallel::*;
use crate::protocol::commitment::utils::linear::{
    decompose_rows_with_carry, mat_vec_mul_crt_ntt_i8_many, try_centered_i8_cache_from_ring_coeffs,
};
use crate::protocol::labrador::aggregation::{
    add_phi_flat_in_place, aggregate_jl_constraints_prover, aggregate_statement,
};
use crate::protocol::labrador::config::LabradorFoldPlan;
use crate::protocol::labrador::constraints::{build_next_constraint_plan, pair_index};
use crate::protocol::labrador::johnson_lindenstrauss::project;
use crate::protocol::labrador::setup::LabradorSetup;
use crate::protocol::labrador::transcript::{
    absorb_labrador_jl_projection, absorb_labrador_level_context, LabradorLevelTranscriptContext,
};
use crate::protocol::labrador::types::{
    LabradorLevelProof, LabradorReductionConfig, LabradorStatement, LabradorWitness,
};
use crate::protocol::labrador::utils::mat_vec_mul;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::{challenge_sparse_ring_elements_rejection_sampled, Transcript};
use crate::{CanonicalField, FieldCore, FieldSampling, FromSmallInt};
use std::sync::Arc;

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

use crate::protocol::labrador::commit::ntt_two_tier_commit;

/// Perform one Labrador fold level (standard or tail, determined by `config.tail`).
///
/// Follows the C Labrador protocol phases:
///   1. Reshape witness according to `plan.nu` into virtual rows of length `plan.nn`
///   2. Commit: inner + outer two-tier commitment → u1
///   3. Project: JL projection → p\[256\], nonce
///   4. LIFTS × (collapse + lift): build linear constraints from JL
///   5. Amortize: absorb into transcript, sample ring-element challenges,
///      fold z = sum_i c_i * s_i, decompose z → output witness
///
/// # Errors
///
/// Returns `HachiError::InvalidInput` if the witness is empty or `config.f` is zero.
/// Propagates errors from commitment, projection, or hashing.
#[tracing::instrument(
    skip_all,
    name = "labrador::prove_level",
    fields(level_index, tail = config.tail)
)]
pub fn prove_level<F, T, const D: usize>(
    witness: &LabradorWitness<F, D>,
    statement: &LabradorStatement<F, D>,
    config: &LabradorReductionConfig,
    plan: &LabradorFoldPlan,
    setup: &Arc<LabradorSetup<F, D>>,
    level_index: usize,
    transcript: &mut T,
) -> Result<LabradorFoldResult<F, D>, HachiError>
where
    F: FieldCore + CanonicalField + FieldSampling + FromSmallInt,
    T: Transcript<F>,
{
    if witness.rows().is_empty() {
        return Err(HachiError::InvalidInput(
            "cannot fold empty Labrador witness".to_string(),
        ));
    }
    if config.f == 0 {
        return Err(HachiError::InvalidInput(
            "Labrador fold requires f > 0".to_string(),
        ));
    }

    let orig_row_lengths: Vec<usize> = witness.rows().iter().map(|row| row.len()).collect();

    // Phase 0: Reshape witness according to nu-partition.
    let reshaped = reshape_rows(witness.rows(), &plan.nu, plan.nn);
    let virtual_witness = LabradorWitness::new_unchecked(reshaped);
    let virt_row_lengths: Vec<usize> = virtual_witness.rows().iter().map(|r| r.len()).collect();
    let rr = virt_row_lengths.len();
    let nn = plan.nn;

    // Phase 1: Inner commitments (t_i) and outer commitment u1.
    let (t_hat, u1) = ntt_two_tier_commit(
        &setup.a_ntt,
        &setup.b_ntt,
        virtual_witness.rows(),
        config.fu,
        config.bu as u32,
        config.kappa1,
    )?;

    // Absorb level context and u1 before deriving JL seed.
    absorb_labrador_level_context(
        transcript,
        &LabradorLevelTranscriptContext {
            level_index,
            tail: config.tail,
            input_row_lengths: orig_row_lengths.clone(),
            f: config.f,
            b: config.b,
            fu: config.fu,
            bu: config.bu,
            kappa: config.kappa,
            kappa1: config.kappa1,
        },
    )?;

    transcript.append_serde(labels::ABSORB_LABRADOR_U1, &u1);

    // Phase 2: JL Projection — nonce + matrix squeezed from transcript.
    let (jl_projection, jl_nonce, jl_matrix) = project(&virtual_witness, transcript)?;

    absorb_labrador_jl_projection(transcript, &jl_projection);

    // Phase 3: JL lift constraints and aggregation (on virtual rows).
    let (phi_jl_flat, b_jl, bb) =
        aggregate_jl_constraints_prover(&virtual_witness, &jl_matrix, transcript)?;

    // Aggregate statement constraints on ORIGINAL rows, then reshape phi.
    let (phi_stmt_orig, b_stmt) = aggregate_statement(statement, &orig_row_lengths, transcript)?;
    let phi_stmt = reshape_phi::<F, D>(&phi_stmt_orig, &plan.nu, nn);

    let mut phi_total = phi_stmt;
    add_phi_flat_in_place(&mut phi_total, &phi_jl_flat)?;
    let b_total = b_stmt + b_jl;

    // Linear garbage h_ij from aggregated phi and virtual witness.
    let h = compute_linear_garbage(&phi_total, &virtual_witness)?;

    let h_hat = tracing::info_span!("labrador::decompose_linear_garbage")
        .in_scope(|| decompose_rows_with_carry(&h, config.fu, config.bu as u32));

    let u2 = build_u2(setup, &h_hat);

    // Absorb u2 before amortization challenges.
    transcript.append_serde(labels::ABSORB_LABRADOR_U2, &u2);

    // Phase 4: Amortize — sample rr challenge ring-elements from transcript, fold.
    let challenges = sample_amortize_challenges::<F, T, D>(transcript, rr)?;
    let z = amortize_witness(&virtual_witness, &challenges, nn);

    let decomposed_z = tracing::info_span!("labrador::decompose_amortized_witness")
        .in_scope(|| decompose_rows_with_carry(&z, config.f, config.b as u32));
    let z_rows = split_decomposed_rows(&decomposed_z, config.f, z.len())?;
    let next_witness = assemble_output_witness(z_rows, &t_hat, &h_hat, config.tail);

    let out_norm_sq: u128 =
        tracing::info_span!("labrador::next_witness_norm").in_scope(|| next_witness.norm());

    let reduced_constraints = if config.tail {
        None
    } else {
        Some(Box::new(build_next_constraint_plan(
            &phi_total,
            &b_total,
            &challenges,
            &virt_row_lengths,
            nn,
            config,
            setup.verifier_setup(),
        )?))
    };

    let level_proof = LabradorLevelProof {
        tail: config.tail,
        input_row_lengths: orig_row_lengths,
        config: *config,
        nn,
        nu: plan.nu.clone(),
        u1: u1.clone(),
        u2: u2.clone(),
        jl_projection,
        jl_nonce,
        bb,
        norm_sq: out_norm_sq,
    };

    let statement = LabradorStatement {
        u1,
        u2,
        challenges: challenges.clone(),
        constraints: Vec::new(),
        reduced_constraints,
        beta_sq: out_norm_sq,
    };

    Ok(LabradorFoldResult {
        next_witness,
        level_proof,
        statement,
    })
}

/// Reshape witness rows according to nu-partition into virtual rows of length `nn`.
#[tracing::instrument(skip_all, name = "labrador::reshape_rows")]
fn reshape_rows<F: FieldCore, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
    nu: &[usize],
    nn: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let mut result = Vec::with_capacity(nu.iter().copied().sum());
    let mut group: Vec<CyclotomicRing<F, D>> = Vec::new();

    for (i, row) in rows.iter().enumerate() {
        group.extend_from_slice(row);
        let splits = if i < nu.len() { nu[i] } else { 0 };
        if splits > 0 {
            for chunk_idx in 0..splits {
                let start = chunk_idx * nn;
                if start + nn <= group.len() {
                    result.push(group[start..start + nn].to_vec());
                } else {
                    let mut virtual_row = vec![CyclotomicRing::<F, D>::zero(); nn];
                    let available = group.len().saturating_sub(start).min(nn);
                    if available > 0 {
                        virtual_row[..available].copy_from_slice(&group[start..start + available]);
                    }
                    result.push(virtual_row);
                }
            }
            group.clear();
        }
    }
    result
}

/// Reshape phi vectors (same layout as witness reshaping).
fn reshape_phi<F: FieldCore, const D: usize>(
    phi: &[Vec<CyclotomicRing<F, D>>],
    nu: &[usize],
    nn: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    reshape_rows(phi, nu, nn)
}

#[tracing::instrument(skip_all, name = "labrador::split_decomposed_rows")]
fn split_decomposed_rows<F: FieldCore, const D: usize>(
    flat: &[CyclotomicRing<F, D>],
    parts: usize,
    len: usize,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError> {
    if parts == 0 {
        return Err(HachiError::InvalidInput(
            "cannot split decomposition with zero parts".to_string(),
        ));
    }
    if flat.len() != len * parts {
        return Err(HachiError::InvalidInput(format!(
            "decomposition length mismatch: got {}, expected {}",
            flat.len(),
            len * parts
        )));
    }
    let rows: Vec<Vec<CyclotomicRing<F, D>>> = cfg_into_iter!(0..parts)
        .map(|part| {
            let mut row = Vec::with_capacity(len);
            for idx in 0..len {
                row.push(flat[idx * parts + part]);
            }
            row
        })
        .collect();
    Ok(rows)
}

fn try_centered_i8_rows<F: CanonicalField, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
) -> Option<Vec<Vec<[i8; D]>>> {
    rows.iter()
        .map(|row| try_centered_i8_cache_from_ring_coeffs(row))
        .collect()
}

#[tracing::instrument(skip_all, name = "labrador::compute_linear_garbage")]
fn compute_linear_garbage<F: FieldCore + CanonicalField + FromSmallInt, const D: usize>(
    phi: &[Vec<CyclotomicRing<F, D>>],
    witness: &LabradorWitness<F, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
    let r = witness.rows().len();
    if phi.len() != r {
        return Err(HachiError::InvalidInput(
            "phi row count mismatch".to_string(),
        ));
    }
    for (phi_row, witness_row) in phi.iter().zip(witness.rows().iter()) {
        if phi_row.len() != witness_row.len() {
            return Err(HachiError::InvalidInput(
                "phi row length mismatch".to_string(),
            ));
        }
    }
    let rows = witness.rows();
    let nn = phi.first().map_or(0, Vec::len);
    let pair_count = r * (r + 1) / 2;

    if let Some(rows_i8) = try_centered_i8_rows(rows) {
        if let Ok(cross) = mat_vec_mul_crt_ntt_i8_many(phi, &rows_i8) {
            let mut out = vec![CyclotomicRing::<F, D>::zero(); pair_count];
            for i in 0..r {
                out[pair_index(i, i, r)] = cross[i][i];
                for j in i + 1..r {
                    let pair = pair_index(i, j, r);
                    out[pair] = cross[i][j] + cross[j][i];
                }
            }
            return Ok(out);
        }
    }

    const LINEAR_GARBAGE_COL_BLOCK: usize = 32;
    let out = cfg_fold_reduce!(
        (0..nn.div_ceil(LINEAR_GARBAGE_COL_BLOCK)),
        || vec![CyclotomicRing::<F, D>::zero(); pair_count],
        |mut acc, block_idx| {
            let start = block_idx * LINEAR_GARBAGE_COL_BLOCK;
            let end = (start + LINEAR_GARBAGE_COL_BLOCK).min(nn);
            for col in start..end {
                for i in 0..r {
                    phi[i][col].mul_accumulate_into(&rows[i][col], &mut acc[pair_index(i, i, r)]);
                    for j in i + 1..r {
                        let pair = pair_index(i, j, r);
                        phi[i][col].mul_accumulate_into(&rows[j][col], &mut acc[pair]);
                        phi[j][col].mul_accumulate_into(&rows[i][col], &mut acc[pair]);
                    }
                }
            }
            acc
        },
        |mut acc, partial| {
            for (dst, src) in acc.iter_mut().zip(partial.into_iter()) {
                *dst += src;
            }
            acc
        }
    );
    Ok(out)
}

/// Compute z = sum_i c_i * s_i (all-row linear combination).
#[tracing::instrument(skip_all, name = "labrador::amortize_witness")]
fn amortize_witness<F: FieldCore + CanonicalField, const D: usize>(
    witness: &LabradorWitness<F, D>,
    challenges: &[SparseChallenge],
    max_len: usize,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..max_len)
        .map(|j| {
            let mut acc = CyclotomicRing::<F, D>::zero();
            for (row, challenge) in witness.rows().iter().zip(challenges.iter()) {
                if let Some(elem) = row.get(j) {
                    elem.mul_by_sparse_into(challenge, &mut acc);
                }
            }
            acc
        })
        .collect()
}

#[tracing::instrument(skip_all, name = "labrador::build_u2")]
fn build_u2<F: FieldCore, const D: usize>(
    setup: &LabradorSetup<F, D>,
    h_hat: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    if !setup.matrices.d_mat.is_empty() {
        mat_vec_mul(&setup.matrices.d_mat, h_hat)
    } else {
        h_hat.to_vec()
    }
}

#[tracing::instrument(skip_all, name = "labrador::sample_amortize_challenges")]
fn sample_amortize_challenges<F, T, const D: usize>(
    transcript: &mut T,
    rows: usize,
) -> Result<Vec<SparseChallenge>, HachiError>
where
    F: FieldCore + CanonicalField + FromSmallInt,
    T: Transcript<F>,
{
    challenge_sparse_ring_elements_rejection_sampled::<F, T, D>(
        transcript,
        labels::CHALLENGE_LABRADOR_AMORTIZE,
        rows,
    )
}

#[tracing::instrument(skip_all, name = "labrador::assemble_output_witness")]
fn assemble_output_witness<F: FieldCore, const D: usize>(
    mut z_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    t_hat: &[CyclotomicRing<F, D>],
    h_hat: &[CyclotomicRing<F, D>],
    tail: bool,
) -> LabradorWitness<F, D> {
    if !tail {
        let mut aux = Vec::with_capacity(t_hat.len() + h_hat.len());
        aux.extend_from_slice(t_hat);
        aux.extend_from_slice(h_hat);
        z_rows.push(aux);
    }
    LabradorWitness::new_unchecked(z_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::aggregation::aggregate_jl_constraints_verifier;
    use crate::protocol::labrador::config::trivial_plan;
    use crate::protocol::labrador::constraints::{LabradorConstraint, LabradorConstraintTerm};
    use crate::protocol::labrador::johnson_lindenstrauss::LabradorJlMatrix;
    use crate::protocol::labrador::types::LabradorReductionConfig;
    use crate::protocol::labrador::{verify, LabradorProof};
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
                        F::from_i64(((i + j) as i64 % 5) - 2)
                    }))
                })
                .collect()
        };
        LabradorWitness::new(vec![row(4), row(4), row(4)])
    }

    fn make_plan(
        cfg: &LabradorReductionConfig,
        witness: &LabradorWitness<F, D>,
    ) -> LabradorFoldPlan {
        let row_lengths: Vec<usize> = witness.rows().iter().map(|r| r.len()).collect();
        trivial_plan(*cfg, &row_lengths)
    }

    fn replay_amortize_challenges_for_level(
        statement: &LabradorStatement<F, D>,
        level: &LabradorLevelProof<F, D>,
    ) -> Vec<SparseChallenge> {
        let rr = level.nu.iter().sum::<usize>();
        let virt_row_lengths = vec![level.nn; rr];
        let jl_cols = rr * level.nn * D;
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        absorb_labrador_level_context(
            &mut transcript,
            &LabradorLevelTranscriptContext {
                level_index: 0,
                tail: level.tail,
                input_row_lengths: level.input_row_lengths.clone(),
                f: level.config.f,
                b: level.config.b,
                fu: level.config.fu,
                bu: level.config.bu,
                kappa: level.config.kappa,
                kappa1: level.config.kappa1,
            },
        )
        .unwrap();
        transcript.append_serde(labels::ABSORB_LABRADOR_U1, &level.u1);
        let jl_matrix = LabradorJlMatrix::replay_nonce_search::<F, Blake2bTranscript<F>>(
            &mut transcript,
            level.jl_nonce,
            jl_cols,
        )
        .unwrap();
        absorb_labrador_jl_projection(&mut transcript, &level.jl_projection);
        aggregate_jl_constraints_verifier(
            &virt_row_lengths,
            &level.jl_projection,
            &jl_matrix,
            &level.bb,
            &mut transcript,
        )
        .unwrap();
        aggregate_statement(statement, &level.input_row_lengths, &mut transcript).unwrap();
        transcript.append_serde(labels::ABSORB_LABRADOR_U2, &level.u2);
        sample_amortize_challenges::<F, Blake2bTranscript<F>, D>(&mut transcript, rr).unwrap()
    }

    #[test]
    fn standard_fold_produces_decomposed_output() {
        let witness = sample_witness();
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: Vec::new(),
            reduced_constraints: None,
            beta_sq: 1 << 20,
        };
        let cfg = LabradorReductionConfig {
            f: 1,
            b: 8,
            fu: 2,
            bu: 10,
            kappa: 3,
            kappa1: 2,
            tail: false,
        };
        let plan = make_plan(&cfg, &witness);
        let seed = [1u8; 32];
        let rr = plan.nu.iter().sum::<usize>();
        let setup = std::sync::Arc::new(LabradorSetup::new(&cfg, rr, plan.nn, &seed));
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let out = prove_level(
            &witness,
            &statement,
            &cfg,
            &plan,
            &setup,
            0,
            &mut transcript,
        )
        .unwrap();
        assert!(
            !out.next_witness.rows().is_empty(),
            "fold must produce output witness"
        );
        assert_eq!(out.next_witness.rows().len(), cfg.f + 1);
        assert!(!out.level_proof.u2.is_empty());
    }

    #[test]
    fn tail_fold_produces_decomposed_output() {
        let witness = sample_witness();
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: Vec::new(),
            reduced_constraints: None,
            beta_sq: 1 << 20,
        };
        let cfg = LabradorReductionConfig {
            f: 1,
            b: 8,
            fu: 1,
            bu: 32,
            kappa: 2,
            kappa1: 0,
            tail: true,
        };
        let plan = make_plan(&cfg, &witness);
        let seed = [3u8; 32];
        let rr = plan.nu.iter().sum::<usize>();
        let setup = std::sync::Arc::new(LabradorSetup::new(&cfg, rr, plan.nn, &seed));
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let out = prove_level(
            &witness,
            &statement,
            &cfg,
            &plan,
            &setup,
            1,
            &mut transcript,
        )
        .unwrap();
        assert!(
            !out.next_witness.rows().is_empty(),
            "tail fold must produce output"
        );
        assert_eq!(out.next_witness.rows().len(), cfg.f);
        assert!(out.level_proof.tail);
    }

    #[test]
    fn amortize_challenges_replay_and_bind_transcript_inputs() {
        let mk_ring = |c: i64| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|i| {
                if i == 0 {
                    F::from_i64(c)
                } else {
                    F::zero()
                }
            }))
        };
        let witness = LabradorWitness::new(vec![
            vec![mk_ring(1), mk_ring(2)],
            vec![mk_ring(3), mk_ring(-1)],
        ]);
        let target = witness.rows()[0][0] + witness.rows()[1][1];
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: vec![LabradorConstraint::new(
                vec![
                    LabradorConstraintTerm::new(0, 0, vec![mk_ring(1), mk_ring(0)]),
                    LabradorConstraintTerm::new(1, 0, vec![mk_ring(0), mk_ring(1)]),
                ],
                target,
            )],
            reduced_constraints: None,
            beta_sq: 1 << 40,
        };
        let cfg = LabradorReductionConfig {
            f: 4,
            b: 8,
            fu: 4,
            bu: 8,
            kappa: 2,
            kappa1: 2,
            tail: false,
        };
        let plan = make_plan(&cfg, &witness);
        let comkey_seed = [9u8; 32];
        let rr = plan.nu.iter().sum::<usize>();
        let setup = std::sync::Arc::new(LabradorSetup::new(&cfg, rr, plan.nn, &comkey_seed));
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let fold = prove_level(
            &witness,
            &statement,
            &cfg,
            &plan,
            &setup,
            0,
            &mut transcript,
        )
        .unwrap();

        let replayed = replay_amortize_challenges_for_level(&statement, &fold.level_proof);
        assert_eq!(replayed, fold.statement.challenges);

        let mut mutated_u2 = fold.level_proof.clone();
        mutated_u2.u2[0] += mk_ring(1);
        let replayed_u2 = replay_amortize_challenges_for_level(&statement, &mutated_u2);
        assert_ne!(replayed_u2, fold.statement.challenges);

        let mut mutated_nonce = fold.level_proof.clone();
        mutated_nonce.jl_nonce = if mutated_nonce.jl_nonce == 1 { 2 } else { 1 };
        let replayed_nonce = replay_amortize_challenges_for_level(&statement, &mutated_nonce);
        assert_ne!(replayed_nonce, fold.statement.challenges);
    }

    #[test]
    fn amortize_is_linear_combination() {
        let witness = sample_witness();
        let one = SparseChallenge {
            positions: vec![0],
            coeffs: vec![1],
        };
        let challenges = vec![one; witness.rows().len()];
        let max_len = witness.rows().iter().map(|r| r.len()).max().unwrap();
        let z = amortize_witness(&witness, &challenges, max_len);

        for (j, z_elem) in z.iter().enumerate().take(max_len) {
            let expected = witness
                .rows()
                .iter()
                .map(|row| {
                    row.get(j)
                        .copied()
                        .unwrap_or_else(CyclotomicRing::<F, D>::zero)
                })
                .fold(CyclotomicRing::<F, D>::zero(), |a, b| a + b);
            assert_eq!(*z_elem, expected);
        }
    }

    #[test]
    fn standard_fold_roundtrip_verifies() {
        let mk_ring = |c: i64| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|i| {
                if i == 0 {
                    F::from_i64(c)
                } else {
                    F::zero()
                }
            }))
        };
        let witness = LabradorWitness::new(vec![
            vec![mk_ring(1), mk_ring(2)],
            vec![mk_ring(3), mk_ring(-1)],
        ]);
        let target = witness.rows()[0][0] + witness.rows()[1][1];
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: vec![LabradorConstraint::new(
                vec![
                    LabradorConstraintTerm::new(0, 0, vec![mk_ring(1), mk_ring(0)]),
                    LabradorConstraintTerm::new(1, 0, vec![mk_ring(0), mk_ring(1)]),
                ],
                target,
            )],
            reduced_constraints: None,
            beta_sq: 1 << 40,
        };
        let cfg = LabradorReductionConfig {
            f: 4,
            b: 8,
            fu: 4,
            bu: 8,
            kappa: 2,
            kappa1: 2,
            tail: false,
        };
        let plan = make_plan(&cfg, &witness);
        let comkey_seed = [9u8; 32];
        let rr = plan.nu.iter().sum::<usize>();
        let setup = std::sync::Arc::new(LabradorSetup::new(&cfg, rr, plan.nn, &comkey_seed));
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let fold = prove_level(
            &witness,
            &statement,
            &cfg,
            &plan,
            &setup,
            0,
            &mut transcript,
        )
        .unwrap();

        let proof = LabradorProof {
            levels: vec![fold.level_proof],
            final_opening_witness: fold.next_witness,
        };
        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        verify(&statement, &proof, &comkey_seed, &mut verify_transcript).unwrap();

        let base_proof = LabradorProof {
            levels: Vec::new(),
            final_opening_witness: proof.final_opening_witness.clone(),
        };
        let mut base_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        verify(
            &fold.statement,
            &base_proof,
            &comkey_seed,
            &mut base_transcript,
        )
        .unwrap();
    }

    #[test]
    fn two_level_fold_roundtrip_verifies() {
        let mk_ring = |c: i64| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|i| {
                if i == 0 {
                    F::from_i64(c)
                } else {
                    F::zero()
                }
            }))
        };
        let witness = LabradorWitness::new(vec![
            vec![mk_ring(1), mk_ring(2)],
            vec![mk_ring(3), mk_ring(-1)],
        ]);
        let target = witness.rows()[0][0] + witness.rows()[1][1];
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: vec![LabradorConstraint::new(
                vec![
                    LabradorConstraintTerm::new(0, 0, vec![mk_ring(1), mk_ring(0)]),
                    LabradorConstraintTerm::new(1, 0, vec![mk_ring(0), mk_ring(1)]),
                ],
                target,
            )],
            reduced_constraints: None,
            beta_sq: 1 << 40,
        };
        let cfg = LabradorReductionConfig {
            f: 4,
            b: 8,
            fu: 4,
            bu: 8,
            kappa: 2,
            kappa1: 2,
            tail: false,
        };
        let comkey_seed = [9u8; 32];
        let plan1 = make_plan(&cfg, &witness);
        let rr1 = plan1.nu.iter().sum::<usize>();
        let setup1 = std::sync::Arc::new(LabradorSetup::new(&cfg, rr1, plan1.nn, &comkey_seed));
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let fold1 = prove_level(
            &witness,
            &statement,
            &cfg,
            &plan1,
            &setup1,
            0,
            &mut transcript,
        )
        .unwrap();
        let plan2 = make_plan(&cfg, &fold1.next_witness);
        let rr2 = plan2.nu.iter().sum::<usize>();
        let setup2 = std::sync::Arc::new(LabradorSetup::new(&cfg, rr2, plan2.nn, &comkey_seed));
        let fold2 = prove_level(
            &fold1.next_witness,
            &fold1.statement,
            &cfg,
            &plan2,
            &setup2,
            1,
            &mut transcript,
        )
        .unwrap();

        let r = fold2.level_proof.input_row_lengths.len();
        let challenges = &fold2.statement.challenges;
        let aux_row = &fold2.next_witness.rows()[cfg.f];
        let t_hat_len = r * cfg.kappa * cfg.fu;
        let t_hat = &aux_row[..t_hat_len];
        let mut t_flat = Vec::with_capacity(r * cfg.kappa);
        for chunk in t_hat.chunks(cfg.fu) {
            t_flat.push(CyclotomicRing::gadget_recompose_pow2(chunk, cfg.bu as u32));
        }
        let z_parts: Vec<Vec<CyclotomicRing<F, D>>> = fold2.next_witness.rows()[..cfg.f].to_vec();
        let mut z = Vec::with_capacity(z_parts[0].len());
        for idx in 0..z_parts[0].len() {
            let mut slice = Vec::with_capacity(cfg.f);
            for part in &z_parts {
                slice.push(part[idx]);
            }
            z.push(CyclotomicRing::gadget_recompose_pow2(&slice, cfg.b as u32));
        }
        let az = mat_vec_mul(&setup2.matrices.a_mat, &z);
        let mut rhs = vec![CyclotomicRing::<F, D>::zero(); cfg.kappa];
        for (row_idx, t_row) in t_flat.chunks(cfg.kappa).enumerate() {
            for k in 0..cfg.kappa {
                t_row[k].mul_by_sparse_into(&challenges[row_idx], &mut rhs[k]);
            }
        }
        assert_eq!(az, rhs);

        let proof = LabradorProof {
            levels: vec![fold1.level_proof, fold2.level_proof],
            final_opening_witness: fold2.next_witness,
        };
        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        verify(&statement, &proof, &comkey_seed, &mut verify_transcript).unwrap();
    }
}

#[cfg(test)]
mod malicious_prover {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::protocol::labrador::config::trivial_plan;
    use crate::protocol::labrador::constraints::{LabradorConstraint, LabradorConstraintTerm};
    use crate::protocol::labrador::types::LabradorReductionConfig;
    use crate::protocol::labrador::{verify, LabradorProof};
    use crate::protocol::transcript::labels::DOMAIN_LABRADOR_PROTOCOL;
    use crate::protocol::transcript::Blake2bTranscript;
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    fn mk_ring(c: i64) -> CyclotomicRing<F, D> {
        CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|i| {
            if i == 0 {
                F::from_i64(c)
            } else {
                F::zero()
            }
        }))
    }

    fn valid_single_level_proof() -> (LabradorStatement<F, D>, LabradorProof<F, D>, [u8; 32]) {
        let witness = LabradorWitness::new(vec![
            vec![mk_ring(1), mk_ring(2)],
            vec![mk_ring(3), mk_ring(-1)],
        ]);
        let target = witness.rows()[0][0] + witness.rows()[1][1];
        let statement = LabradorStatement {
            u1: Vec::new(),
            u2: Vec::new(),
            challenges: Vec::new(),
            constraints: vec![LabradorConstraint::new(
                vec![
                    LabradorConstraintTerm::new(0, 0, vec![mk_ring(1), mk_ring(0)]),
                    LabradorConstraintTerm::new(1, 0, vec![mk_ring(0), mk_ring(1)]),
                ],
                target,
            )],
            reduced_constraints: None,
            beta_sq: 1 << 40,
        };
        let cfg = LabradorReductionConfig {
            f: 4,
            b: 8,
            fu: 4,
            bu: 8,
            kappa: 2,
            kappa1: 2,
            tail: false,
        };
        let comkey_seed = [9u8; 32];
        let row_lengths: Vec<usize> = witness.rows().iter().map(|r| r.len()).collect();
        let plan = trivial_plan(cfg, &row_lengths);
        let rr = plan.nu.iter().sum::<usize>();
        let setup = std::sync::Arc::new(LabradorSetup::new(&cfg, rr, plan.nn, &comkey_seed));
        let mut transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        let fold = prove_level(
            &witness,
            &statement,
            &cfg,
            &plan,
            &setup,
            0,
            &mut transcript,
        )
        .unwrap();
        let proof = LabradorProof {
            levels: vec![fold.level_proof],
            final_opening_witness: fold.next_witness,
        };
        (statement, proof, comkey_seed)
    }

    fn assert_verification_fails(
        statement: &LabradorStatement<F, D>,
        proof: &LabradorProof<F, D>,
        comkey_seed: &[u8; 32],
    ) {
        let mut verify_transcript = Blake2bTranscript::<F>::new(DOMAIN_LABRADOR_PROTOCOL);
        assert!(
            verify(statement, proof, comkey_seed, &mut verify_transcript).is_err(),
            "maliciously altered proof should fail verification"
        );
    }

    #[test]
    fn malicious_u1_fails_verification() {
        let (statement, mut proof, comkey_seed) = valid_single_level_proof();
        proof.levels[0].u1[0].coefficients_mut()[0] += F::one();
        assert_verification_fails(&statement, &proof, &comkey_seed);
    }

    #[test]
    fn malicious_u2_fails_verification() {
        let (statement, mut proof, comkey_seed) = valid_single_level_proof();
        proof.levels[0].u2[0].coefficients_mut()[0] += F::one();
        assert_verification_fails(&statement, &proof, &comkey_seed);
    }

    #[test]
    fn malicious_jl_projection_fails_verification() {
        let (statement, mut proof, comkey_seed) = valid_single_level_proof();
        proof.levels[0].jl_projection[0] = i64::MAX;
        assert_verification_fails(&statement, &proof, &comkey_seed);
    }

    #[test]
    fn malicious_jl_nonce_fails_verification() {
        let (statement, mut proof, comkey_seed) = valid_single_level_proof();
        proof.levels[0].jl_nonce += 1;
        assert_verification_fails(&statement, &proof, &comkey_seed);
    }

    #[test]
    fn malicious_bb_fails_verification() {
        let (statement, mut proof, comkey_seed) = valid_single_level_proof();
        proof.levels[0].bb[0].coefficients_mut()[0] += F::one();
        assert_verification_fails(&statement, &proof, &comkey_seed);
    }
}
