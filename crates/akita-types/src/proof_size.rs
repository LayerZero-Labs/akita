//! Header-stripped proof-byte formulas shared by the offline planner DP,
//! the schedule selector, and profiling tooling.
//!
//! This is the single source of truth for level/scheduling proof-byte
//! accounting. [`level_proof_bytes`] scores one fold level;
//! [`estimate_proof_bytes`] walks a compact [`GeneratedScheduleTableEntry`]
//! and sums the whole proof. Neither is on the prover/verifier replay path.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;

use crate::generated::{GeneratedScheduleTableEntry, GeneratedStep, SisModulusFamily};
use crate::layout::{field_bytes, proof_ring_vec_bytes, sumcheck_rounds};
use crate::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, root_extension_opening_partials,
    stage1_tree_stage_shapes, w_ring_element_count_with_counts_bits,
    w_ring_element_count_with_counts_for_layout_bits, AkitaScheduleInputs, AkitaScheduleLookupKey,
    DecompositionParams, DirectStep, DirectWitnessShape, FoldStep, LevelParams, MRowLayout,
    Schedule, Step,
};

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

#[cfg(feature = "zk")]
fn eq_factored_round_mask_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    sumcheck_bytes(rounds, degree, elem_bytes)
}

fn stage1_proof_bytes(rounds: usize, b: usize, elem_bytes: usize) -> usize {
    stage1_tree_stage_shapes(rounds, b)
        .into_iter()
        .map(|stage| {
            ({
                #[cfg(feature = "zk")]
                {
                    eq_factored_round_mask_bytes(rounds, stage.sumcheck_proof.1, elem_bytes)
                }
                #[cfg(not(feature = "zk"))]
                {
                    sumcheck_bytes(rounds, stage.sumcheck_proof.1, elem_bytes)
                }
            }) + stage.child_claims * elem_bytes
        })
        .sum::<usize>()
        + elem_bytes
}

/// Header-stripped byte size of one folded proof level, parametrized by
/// [`MRowLayout`].
///
/// Ring-valued objects (`y`, `v`, next-witness commitment) serialize over
/// the base SIS field. Sumcheck objects and scalar evaluations serialize
/// over the challenge field, which may be a non-trivial extension of the
/// base field for small-prime configurations.
///
/// `next_lp` is required on the `Intermediate` arm (it sizes the next-level
/// witness commitment shipped on the wire) and unused on the `Terminal` arm;
/// terminal callers pass `None`.
///
/// # Panics
///
/// Panics if `layout == Intermediate` and `next_lp` is `None`. This helper
/// is offline (planner / selector / profiling) and is not on the verifier
/// path, so the no-panic boundary does not apply.
pub fn level_proof_bytes(
    base_field_bits: u32,
    challenge_field_bits: u32,
    lp: &LevelParams,
    next_lp: Option<&LevelParams>,
    next_w_len: usize,
    num_claims: usize,
    layout: MRowLayout,
) -> usize {
    let base_elem_bytes = field_bytes(base_field_bits);
    let challenge_elem_bytes = field_bytes(challenge_field_bits);
    let y_bytes = proof_ring_vec_bytes(num_claims, lp.ring_dimension, base_elem_bytes);
    let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
    let sumcheck = sumcheck_bytes(rounds, 3, challenge_elem_bytes);
    match layout {
        MRowLayout::Terminal => y_bytes + sumcheck,
        MRowLayout::Intermediate => {
            let next_lp = next_lp
                .expect("level_proof_bytes(Intermediate) requires next_lp; caller must pass Some");
            let v_bytes =
                proof_ring_vec_bytes(lp.d_key.row_len(), lp.ring_dimension, base_elem_bytes);
            let next_commit_bytes = proof_ring_vec_bytes(
                next_lp.b_key.row_len(),
                next_lp.ring_dimension,
                base_elem_bytes,
            );
            let next_eval_bytes = challenge_elem_bytes;
            let b = 1usize << lp.log_basis;
            let stage1_bytes = stage1_proof_bytes(rounds, b, challenge_elem_bytes);
            y_bytes + v_bytes + stage1_bytes + sumcheck + next_commit_bytes + next_eval_bytes
        }
    }
}

fn padded_boolean_vars(len: usize) -> Result<usize, AkitaError> {
    let padded = len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening witness length overflow".to_string()))?;
    Ok(padded.trailing_zeros() as usize)
}

fn extension_opening_reduction_level_bytes(
    challenge_field_bits: u32,
    extension_opening_width: usize,
    fold_level: usize,
    key: AkitaScheduleLookupKey,
    current_w_len: usize,
) -> Result<usize, AkitaError> {
    if extension_opening_width <= 1 {
        return Ok(0);
    }
    let (partials, opening_vars) = if fold_level == 0 {
        (
            root_extension_opening_partials(extension_opening_width, key.num_w_vectors),
            key.num_vars,
        )
    } else {
        (extension_opening_width, padded_boolean_vars(current_w_len)?)
    };
    extension_opening_reduction_proof_bytes(
        challenge_field_bits,
        partials,
        opening_vars,
        extension_opening_width,
    )
}

/// Build the runtime [`Schedule`] for a compact generated entry, expanding
/// each fold level via
/// [`crate::generated::GeneratedFoldStep::expand_to_level_params`] and
/// computing each step's witness lengths and proof bytes.
///
/// This is the single canonical entry walker: it replaces the former
/// `akita-derive` materializer (`schedule_plan_from_table` +
/// `schedule_from_plan`). [`estimate_proof_bytes`] is a thin reader of the
/// resulting `total_bytes`, and `akita-config` wraps this with the
/// per-config policy values to drive prove/verify.
///
/// The policy hooks are threaded as values/closures so this stays free of a
/// `<Cfg>` parameter:
/// - `stage1(ring_d)` resolves the sparse-challenge config (≡
///   `Cfg::stage1_challenge_config`).
/// - `fold_shape(inputs)` resolves the fold-round tensor shape (≡
///   `Cfg::fold_challenge_shape_at_level`).
///
/// # Errors
///
/// Returns an error when the entry is structurally invalid, a fold step
/// names an unsupported ring dimension, layout expansion fails, or a
/// witness length overflows.
#[allow(clippy::too_many_arguments)]
pub fn schedule_from_entry_bits<Stage1, FoldShape>(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    sis_family: SisModulusFamily,
    root_decomp: DecompositionParams,
    challenge_field_bits: u32,
    extension_opening_width: usize,
    ring_subfield_norm_bound: u32,
    stage1: Stage1,
    fold_shape: FoldShape,
) -> Result<Schedule, AkitaError>
where
    Stage1: Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    FoldShape: Fn(AkitaScheduleInputs) -> TensorChallengeShape,
{
    entry.validate()?;
    let field_bits = root_decomp.field_bits();
    let expected_root_w_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;
    let root_is_batched = key.num_points != 1
        || key.num_t_vectors != 1
        || key.num_w_vectors != 1
        || key.num_z_vectors != 1;
    let batched_root_dims = root_is_batched.then_some((key.num_t_vectors, field_bits));

    let mut steps = Vec::with_capacity(entry.steps.len());
    let mut total = 0usize;
    let mut fold_level = 0usize;
    let mut current_w_len = expected_root_w_len;
    let mut current_log_basis = root_decomp.log_basis;
    let mut terminal_witness_field_len: Option<usize> = None;

    for (idx, step) in entry.steps.iter().enumerate() {
        match step {
            GeneratedStep::Fold(level) => {
                let next = entry.steps.get(idx + 1).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "generated schedule ended with a fold step at level {fold_level}"
                    ))
                })?;
                let is_terminal = matches!(next, GeneratedStep::Direct(_));
                let inputs = AkitaScheduleInputs {
                    num_vars: key.num_vars,
                    level: fold_level,
                    current_w_len,
                };
                let batched = if fold_level == 0 {
                    batched_root_dims
                } else {
                    None
                };
                let lp = level.expand_to_level_params(
                    sis_family,
                    fold_level,
                    current_w_len,
                    root_decomp,
                    stage1(level.ring_d as usize)?,
                    fold_shape(inputs),
                    ring_subfield_norm_bound,
                    batched,
                )?;
                let (np, nt, nw, nz) = if fold_level == 0 {
                    (
                        key.num_points,
                        key.num_t_vectors,
                        key.num_w_vectors,
                        key.num_z_vectors,
                    )
                } else {
                    (1, 1, 1, 1)
                };
                let mul_d = |ring: usize| -> Result<usize, AkitaError> {
                    ring.checked_mul(lp.ring_dimension).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "generated next witness length overflow".to_string(),
                        )
                    })
                };
                let (next_w_len, next_lp, layout) = if is_terminal {
                    let ring = w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &lp,
                        np,
                        nt,
                        nw,
                        nz,
                        MRowLayout::Terminal,
                    )?;
                    let len = mul_d(ring)?;
                    terminal_witness_field_len = Some(len);
                    (len, None, MRowLayout::Terminal)
                } else {
                    let ring =
                        w_ring_element_count_with_counts_bits(field_bits, &lp, np, nt, nw, nz)?;
                    let len = mul_d(ring)?;
                    let GeneratedStep::Fold(next_level) = next else {
                        unreachable!("non-terminal fold successor is a fold step");
                    };
                    let next_inputs = AkitaScheduleInputs {
                        num_vars: key.num_vars,
                        level: fold_level + 1,
                        current_w_len: len,
                    };
                    let next_lp = next_level.expand_to_level_params(
                        sis_family,
                        fold_level + 1,
                        len,
                        root_decomp,
                        stage1(next_level.ring_d as usize)?,
                        fold_shape(next_inputs),
                        ring_subfield_norm_bound,
                        None,
                    )?;
                    (len, Some(next_lp), MRowLayout::Intermediate)
                };
                let num_claims_here = if fold_level == 0 {
                    key.num_z_vectors
                } else {
                    1
                };
                let level_bytes = level_proof_bytes(
                    field_bits,
                    challenge_field_bits,
                    &lp,
                    next_lp.as_ref(),
                    next_w_len,
                    num_claims_here,
                    layout,
                ) + extension_opening_reduction_level_bytes(
                    challenge_field_bits,
                    extension_opening_width,
                    fold_level,
                    key,
                    current_w_len,
                )?;
                total = total.checked_add(level_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("proof byte total overflow".to_string())
                })?;
                steps.push(Step::Fold(FoldStep {
                    params: lp,
                    current_w_len,
                    next_w_len,
                    level_bytes,
                }));
                fold_level += 1;
                current_w_len = next_w_len;
                current_log_basis = match next {
                    GeneratedStep::Fold(next_level) => next_level.log_basis,
                    GeneratedStep::Direct(_) => level.log_basis,
                };
            }
            GeneratedStep::Direct(direct) => {
                let (witness_shape, direct_current_w_len, params) = if fold_level == 0 {
                    // Root-direct: ship the cleartext field-element witness;
                    // carry the expanded root commit layout. The per-claim
                    // layout is expanded unscaled, then scaled to the batched
                    // widths. A strict batched-scaling audit failure (the
                    // large-`num_vars` edge) yields the *uncommittable*
                    // `params: None` rather than propagating — matching the
                    // former materializer's graceful fallback.
                    let params = match direct.commit {
                        Some(commit) => {
                            let per_claim = commit.expand_to_level_params(
                                sis_family,
                                0,
                                expected_root_w_len,
                                root_decomp,
                                stage1(commit.ring_d as usize)?,
                                fold_shape(AkitaScheduleInputs {
                                    num_vars: key.num_vars,
                                    level: 0,
                                    current_w_len: expected_root_w_len,
                                }),
                                ring_subfield_norm_bound,
                                None,
                            )?;
                            match batched_root_dims {
                                Some((num_t_vectors, fb)) => {
                                    crate::scale_batched_root_layout(&per_claim, num_t_vectors, fb)
                                        .ok()
                                }
                                None => Some(per_claim),
                            }
                        }
                        None => None,
                    };
                    (
                        DirectWitnessShape::FieldElements(expected_root_w_len),
                        expected_root_w_len,
                        params,
                    )
                } else {
                    let len = terminal_witness_field_len.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing precomputed witness length".to_string(),
                        )
                    })?;
                    (
                        DirectWitnessShape::PackedDigits((len, current_log_basis)),
                        len,
                        None,
                    )
                };
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
                total = total.checked_add(direct_bytes).ok_or_else(|| {
                    AkitaError::InvalidSetup("proof byte total overflow".to_string())
                })?;
                steps.push(Step::Direct(DirectStep {
                    current_w_len: direct_current_w_len,
                    witness_shape,
                    direct_bytes,
                    params,
                }));
            }
        }
    }

    Ok(Schedule {
        steps,
        total_bytes: total,
    })
}

/// Total header-stripped proof bytes for a compact generated schedule entry.
///
/// Thin reader of [`schedule_from_entry_bits`]; the result equals the former
/// materializer's `exact_proof_bytes`, so the schedule selector can compare
/// entries without materializing a plan.
///
/// # Errors
///
/// Propagates [`schedule_from_entry_bits`].
#[allow(clippy::too_many_arguments)]
pub fn estimate_proof_bytes<Stage1, FoldShape>(
    entry: &GeneratedScheduleTableEntry,
    key: AkitaScheduleLookupKey,
    sis_family: SisModulusFamily,
    root_decomp: DecompositionParams,
    challenge_field_bits: u32,
    extension_opening_width: usize,
    ring_subfield_norm_bound: u32,
    stage1: Stage1,
    fold_shape: FoldShape,
) -> Result<usize, AkitaError>
where
    Stage1: Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    FoldShape: Fn(AkitaScheduleInputs) -> TensorChallengeShape,
{
    Ok(schedule_from_entry_bits(
        entry,
        key,
        sis_family,
        root_decomp,
        challenge_field_bits,
        extension_opening_width,
        ring_subfield_norm_bound,
        stage1,
        fold_shape,
    )?
    .total_bytes)
}

#[cfg(test)]
mod tests {
    //! End-to-end byte-formula tests: build a synthetic proof body via the
    //! runtime serializer and compare its size against the
    //! [`level_proof_bytes`] formula at every supported log_basis.

    use super::*;

    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::AkitaError;
    use akita_field::{FieldCore, Prime128OffsetA7F7};
    use akita_serialization::{AkitaSerialize, Compress};
    use akita_sumcheck::EqFactoredUniPoly;
    #[cfg(not(feature = "zk"))]
    use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProof, SumcheckProof};
    #[cfg(feature = "zk")]
    use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProofMasked, SumcheckProofMasked};

    use crate::{
        direct_witness_bytes, AkitaLevelProof, AkitaStage1Proof, AkitaStage1StageProof,
        AkitaStage2Proof, DirectWitnessProof, DirectWitnessShape, FlatRingVec, PackedDigits,
        SisModulusFamily, TerminalLevelProof,
    };

    type F = Prime128OffsetA7F7;

    #[cfg(not(feature = "zk"))]
    fn dummy_sumcheck<F: FieldCore>(rounds: usize, degree: usize) -> SumcheckProof<F> {
        SumcheckProof {
            round_polys: (0..rounds)
                .map(|_| CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(); degree],
                })
                .collect(),
        }
    }

    #[cfg(feature = "zk")]
    fn dummy_sumcheck_proof_masked<F: FieldCore>(
        rounds: usize,
        degree: usize,
    ) -> SumcheckProofMasked<F> {
        let compressed_rounds = || {
            (0..rounds)
                .map(|_| CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(); degree],
                })
                .collect()
        };
        SumcheckProofMasked {
            masked_round_polys: compressed_rounds(),
        }
    }

    #[cfg(not(feature = "zk"))]
    fn dummy_eq_factored_sumcheck<F: FieldCore>(
        rounds: usize,
        degree: usize,
    ) -> EqFactoredSumcheckProof<F> {
        EqFactoredSumcheckProof {
            round_polys: (0..rounds)
                .map(|_| EqFactoredUniPoly {
                    coeffs_except_linear_term: vec![
                        F::zero();
                        EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(degree)
                    ],
                })
                .collect(),
        }
    }

    #[cfg(feature = "zk")]
    fn dummy_eq_factored_sumcheck_proof_masked<F: FieldCore>(
        rounds: usize,
        degree: usize,
    ) -> EqFactoredSumcheckProofMasked<F> {
        let rounds_for = || {
            (0..rounds)
                .map(|_| EqFactoredUniPoly {
                    coeffs_except_linear_term: vec![
                        F::zero();
                        EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(degree)
                    ],
                })
                .collect()
        };
        EqFactoredSumcheckProofMasked {
            masked_round_polys: rounds_for(),
        }
    }

    fn dummy_stage1_proof<F: FieldCore>(rounds: usize, b: usize) -> AkitaStage1Proof<F> {
        AkitaStage1Proof {
            stages: stage1_tree_stage_shapes(rounds, b)
                .into_iter()
                .map(|shape| AkitaStage1StageProof {
                    #[cfg(not(feature = "zk"))]
                    sumcheck_proof: dummy_eq_factored_sumcheck(rounds, shape.sumcheck_proof.1),
                    #[cfg(feature = "zk")]
                    sumcheck_proof_masked: dummy_eq_factored_sumcheck_proof_masked(
                        rounds,
                        shape.sumcheck_proof.1,
                    ),
                    child_claims: vec![F::zero(); shape.child_claims],
                })
                .collect(),
            s_claim: F::zero(),
        }
    }

    fn exact_level_proof_bytes<F: FieldCore + AkitaSerialize>(
        lp: &LevelParams,
        next_lp: &LevelParams,
        next_w_len: usize,
    ) -> Result<usize, AkitaError> {
        let current_coeffs = lp
            .d_key
            .row_len()
            .checked_mul(lp.ring_dimension)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive proof sizing overflow".to_string())
            })?;
        let next_commit_coeffs = next_lp
            .b_key
            .row_len()
            .checked_mul(next_lp.ring_dimension)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive proof sizing overflow".to_string())
            })?;
        let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
        let b = 1usize << lp.log_basis;

        let proof = AkitaLevelProof {
            y_ring: FlatRingVec::from_coeffs(vec![F::zero(); lp.ring_dimension]),
            extension_opening_reduction: None,
            v: FlatRingVec::from_coeffs(vec![F::zero(); current_coeffs]),
            stage1: dummy_stage1_proof(rounds, b),
            stage2: AkitaStage2Proof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: dummy_sumcheck(rounds, 3),
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: dummy_sumcheck_proof_masked(rounds, 3),
                next_w_commitment: FlatRingVec::from_coeffs(vec![F::zero(); next_commit_coeffs]),
                #[cfg(not(feature = "zk"))]
                next_w_eval: F::zero(),
                #[cfg(feature = "zk")]
                next_w_eval_masked: F::zero(),
            },
        };
        Ok(proof.serialized_size(Compress::No))
    }

    #[test]
    fn planned_level_bytes_match_two_stage_payload_at_all_bases() {
        const D: usize = 64;
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        };
        let next_lp =
            LevelParams::params_only(SisModulusFamily::Q128, D, 2, 2, 3, 2, stage1_config.clone());
        let next_w_len = D * 8;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(
                SisModulusFamily::Q128,
                D,
                log_basis,
                2,
                2,
                2,
                stage1_config.clone(),
            )
            .with_decomp(0, 0, 1, 1, 0)
            .unwrap();
            assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    next_w_len,
                    1,
                    MRowLayout::Intermediate,
                ),
                exact_level_proof_bytes::<F>(&lp, &next_lp, next_w_len).unwrap(),
                "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_terminal_level_bytes_match_terminal_payload_at_all_bases() {
        const D: usize = 64;
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        };
        let next_w_len = D * 8;
        let num_claims = 3;
        let final_w_num_elems = 1024;
        let final_w_bits = 5;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(
                SisModulusFamily::Q128,
                D,
                log_basis,
                2,
                2,
                2,
                stage1_config.clone(),
            )
            .with_decomp(0, 0, 1, 1, 0)
            .unwrap();
            let rounds = sumcheck_rounds(D, next_w_len);

            let final_witness = DirectWitnessProof::PackedDigits(PackedDigits::from_i8_digits(
                &vec![0i8; final_w_num_elems],
                final_w_bits,
            ));
            let final_witness_bytes_runtime = final_witness.serialized_size(Compress::No);
            let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
                vec![CyclotomicRing::<F, D>::zero(); num_claims],
                None,
                #[cfg(not(feature = "zk"))]
                dummy_sumcheck(rounds, 3),
                #[cfg(feature = "zk")]
                dummy_sumcheck_proof_masked(rounds, 3),
                final_witness,
            );

            let serialized_without_witness =
                terminal_proof.serialized_size(Compress::No) - final_witness_bytes_runtime;

            assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    None,
                    next_w_len,
                    num_claims,
                    MRowLayout::Terminal,
                ),
                serialized_without_witness,
                "planned terminal-level bytes should match the serialized terminal body \
                 (less final_witness) at log_basis={log_basis}"
            );

            assert_eq!(
                direct_witness_bytes(
                    128,
                    &DirectWitnessShape::PackedDigits((final_w_num_elems, final_w_bits))
                ),
                final_witness_bytes_runtime,
                "direct_witness_bytes should match the serialized packed-digit \
                 final witness at log_basis={log_basis}"
            );
        }
    }
}
