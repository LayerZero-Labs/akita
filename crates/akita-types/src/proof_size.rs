//! Header-stripped proof-byte formula for one fold level, shared by the
//! offline planner DP, the schedule selector, and profiling tooling.
//!
//! This is the single source of truth for direct-mode per-level proof-byte accounting:
//! [`level_proof_bytes`] scores one fold level. It is not on the
//! prover/verifier replay path. The compact-entry walker that sums a whole
//! proof (`schedule_from_entry`) lives in `akita-planner`, next to the
//! schedule-table representation it consumes.

use crate::layout::{field_bytes, proof_ring_vec_bytes, sumcheck_rounds};
use crate::{CommittedGroupParams, DigitRangePlan};
use akita_field::AkitaError;

/// Fixed wire size of `fold_grind_nonce` on every fold level proof.
pub const FOLD_GRIND_NONCE_BYTES: usize = 4;

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

fn stage1_proof_bytes(rounds: usize, b: usize, elem_bytes: usize) -> usize {
    DigitRangePlan::new(b)
        .expect("scheduled range basis must be certified")
        .stage_shapes(rounds)
        .into_iter()
        .map(|stage| {
            ({ sumcheck_bytes(rounds, stage.sumcheck_proof.1, elem_bytes) })
                + stage.child_claims * elem_bytes
        })
        .sum::<usize>()
        + elem_bytes
}

/// Header-stripped byte size of one non-terminal folded proof level.
///
/// Ring-valued objects (`y`, `v`, next-witness commitment) serialize over
/// the base SIS field. Sumcheck objects and scalar evaluations serialize
/// over the challenge field, which may be a non-trivial extension of the
/// base field for small-prime configurations.
///
/// This prices the **direct-mode** folded payload only
/// (`SetupContributionMode::Direct`): the y/v ring blocks, the stage-1
/// range-check tree, the fused stage-2 sumcheck, and the next-level witness
/// binding plus its evaluation. An ordinary recursive edge ships the outer
/// commitment; an edge into the suffix terminal reuses that terminal proof's
/// inner `t` state and ships no duplicate commitment bytes. It deliberately
/// **excludes** the optional
/// recursive stage-3 setup-product sumcheck
/// (`SetupContributionMode::Recursive`), whose per-level overhead is priced
/// separately by [`stage3_setup_product_bytes`]. The planner adds that exact
/// payload when the selected successor consumes an incoming setup prefix.
///
/// `next_lp` is required only for an intermediate outer-commitment binding
/// (it sizes the next-level witness commitment shipped on the wire). It is
/// unused for a terminal-inner binding.
///
/// # Errors
///
/// Returns an error if an outer-commitment binding has no `next_lp`, or if an
/// intermediate layout has no outgoing witness binding.
pub fn level_proof_bytes(
    base_field_bits: u32,
    challenge_field_bits: u32,
    lp: &CommittedGroupParams,
    next_lp: Option<&CommittedGroupParams>,
    output_witness_len: usize,
    next_witness_binding: Option<crate::NextWitnessBindingPolicy>,
) -> Result<usize, AkitaError> {
    let base_elem_bytes = field_bytes(base_field_bits);
    let challenge_elem_bytes = field_bytes(challenge_field_bits);
    let rounds = sumcheck_rounds(lp.d_a(), output_witness_len);
    let sumcheck = sumcheck_bytes(rounds, 3, challenge_elem_bytes);
    let v_bytes = proof_ring_vec_bytes(
        lp.open_commit_matrix.output_rank(),
        lp.role_dims().d_d(),
        base_elem_bytes,
    );
    let next_commit_bytes = match next_witness_binding {
        Some(crate::NextWitnessBindingPolicy::OuterCommitment) => {
            let next_lp = next_lp.ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "outer-commitment level proof is missing successor params".to_string(),
                )
            })?;
            proof_ring_vec_bytes(
                next_lp.outer_commit_matrix.output_rank(),
                next_lp.role_dims().d_b(),
                base_elem_bytes,
            )
        }
        Some(crate::NextWitnessBindingPolicy::TerminalInnerState) => 0,
        None => {
            return Err(AkitaError::InvalidSetup(
                "intermediate level is missing an outgoing witness binding".to_string(),
            ))
        }
    };
    let next_eval_bytes = challenge_elem_bytes;
    let b = 1usize << lp.log_basis_open;
    let stage1_bytes = stage1_proof_bytes(rounds, b, challenge_elem_bytes);
    Ok(v_bytes
        + FOLD_GRIND_NONCE_BYTES
        + stage1_bytes
        + sumcheck
        + next_commit_bytes
        + next_eval_bytes)
}

/// Header-stripped byte size of the recursive-mode stage-3 setup-product
/// sumcheck payload (`SetupSumcheckProof`) for one non-terminal fold level.
///
/// This is the proof-size overhead that `SetupContributionMode::Recursive`
/// adds on top of the direct-mode payload priced by [`level_proof_bytes`]. It
/// is added to the direct fold payload before the planner compares direct and
/// offloaded successor edges.
///
/// The payload is the setup claim and the carried next-witness opening (two
/// challenge-field elements), followed by a degree-[`crate::SETUP_SUMCHECK_DEGREE`]
/// product sumcheck. Stage 3 fuses the setup-product term with the carried
/// witness term, so the serialized round count is the max of the setup domain
/// rounds (`log2(D) + log2(next_pow2(setup_ring_len))`) and the witness domain
/// rounds (`sumcheck_rounds(D, output_witness_len)`).
///
/// `ring_dimension` and the next-power-of-two of `setup_ring_len` must be
/// powers of two; this offline helper is not on the verifier path.
pub fn stage3_setup_product_bytes(
    challenge_field_bits: u32,
    ring_dimension: usize,
    setup_ring_len: usize,
    output_witness_len: usize,
) -> usize {
    let challenge_elem_bytes = field_bytes(challenge_field_bits);
    let ring_bits = ring_dimension.trailing_zeros() as usize;
    let lambda_bits = setup_ring_len.next_power_of_two().trailing_zeros() as usize;
    let setup_rounds = ring_bits + lambda_bits;
    let witness_rounds = sumcheck_rounds(ring_dimension, output_witness_len);
    let rounds = setup_rounds.max(witness_rounds);
    // Claimed setup contribution + carried setup-prefix opening + carried
    // next-witness opening + degree-2 fused setup/carry sumcheck rounds.
    3 * challenge_elem_bytes
        + sumcheck_bytes(rounds, crate::SETUP_SUMCHECK_DEGREE, challenge_elem_bytes)
}

#[cfg(test)]
mod tests {
    //! End-to-end byte-formula tests: build a synthetic proof body via the
    //! runtime serializer and compare its size against the
    //! [`level_proof_bytes`] formula at every supported log_basis.

    use super::*;

    use akita_challenges::SparseChallengeConfig;
    use akita_field::AkitaError;
    use akita_field::{CanonicalField, FieldCore, Prime128OffsetA7F7};
    use akita_serialization::{AkitaSerialize, Compress};
    use akita_sumcheck::EqFactoredUniPoly;
    use akita_sumcheck::{CompressedUniPoly, EqFactoredSumcheckProof, SumcheckProof};

    use crate::golomb_rice::golomb_rice_encode_vec;
    use crate::tail_golomb_rice_z_params;
    use crate::{
        terminal_response_bytes, AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof,
        FoldLevelProof, RingVec, SetupSumcheckProof, SisModulusProfileId, TerminalLevelProof,
        TerminalResponse, TerminalResponseShape, SETUP_SUMCHECK_DEGREE,
    };

    type F = Prime128OffsetA7F7;

    fn terminal_response_fixture(
        lp: &CommittedGroupParams,
        num_claims: usize,
    ) -> (TerminalResponse<F>, TerminalResponseShape) {
        let field_bits = F::modulus_bits();
        let shape = TerminalResponseShape::from_groups(
            lp,
            field_bits,
            [(lp as &dyn crate::LevelParamsLike, num_claims, num_claims, 1)],
        )
        .expect("terminal response shape");
        let layout = shape.layout.clone();
        let group = layout.groups[0];
        let (rice_low_bits, zigzag_w) =
            tail_golomb_rice_z_params(lp, num_claims).expect("golomb z params");
        let z_payload =
            golomb_rice_encode_vec(&vec![0i64; group.z_coords], rice_low_bits, zigzag_w)
                .expect("encode zero z segment");
        let witness = TerminalResponse {
            layout: layout.clone(),
            z_payloads: vec![z_payload],
            e_fields: RingVec::from_coeffs(vec![F::zero(); group.e_field_elems]),
            t_fields: RingVec::from_coeffs(vec![F::zero(); group.t_field_elems]),
        };
        (witness, shape)
    }

    fn dummy_sumcheck<F: FieldCore>(rounds: usize, degree: usize) -> SumcheckProof<F> {
        SumcheckProof {
            round_polys: (0..rounds)
                .map(|_| CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(); degree],
                })
                .collect(),
        }
    }

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

    fn dummy_stage1_proof<F: FieldCore>(rounds: usize, b: usize) -> AkitaStage1Proof<F> {
        AkitaStage1Proof {
            stages: DigitRangePlan::new(b)
                .expect("test range basis")
                .stage_shapes(rounds)
                .into_iter()
                .map(|shape| AkitaStage1StageProof {
                    sumcheck_proof: dummy_eq_factored_sumcheck(rounds, shape.sumcheck_proof.1),
                    child_claims: vec![F::zero(); shape.child_claims],
                })
                .collect(),
            range_image_evaluation: F::zero(),
        }
    }

    /// Build a degree-[`SETUP_SUMCHECK_DEGREE`] stage-3 setup-product proof
    /// whose round count matches the fused setup/carry verifier rounds.
    fn dummy_stage3_proof<F: FieldCore>(
        d: usize,
        setup_ring_len: usize,
        output_witness_len: usize,
    ) -> SetupSumcheckProof<F> {
        let ring_bits = d.trailing_zeros() as usize;
        let lambda_bits = setup_ring_len.next_power_of_two().trailing_zeros() as usize;
        let setup_rounds = ring_bits + lambda_bits;
        let witness_rounds = sumcheck_rounds(d, output_witness_len);
        let rounds = setup_rounds.max(witness_rounds);
        SetupSumcheckProof {
            claim: F::zero(),
            setup_prefix_eval: F::zero(),
            next_w_eval: F::zero(),
            sumcheck: akita_sumcheck::SumcheckProof {
                round_polys: (0..rounds)
                    .map(|_| CompressedUniPoly {
                        coeffs_except_linear_term: vec![F::zero(); SETUP_SUMCHECK_DEGREE],
                    })
                    .collect(),
            },
        }
    }

    fn exact_level_proof_bytes<F: FieldCore + CanonicalField + AkitaSerialize>(
        lp: &CommittedGroupParams,
        next_lp: &CommittedGroupParams,
        output_witness_len: usize,
        stage3_setup_ring_len: Option<usize>,
        next_witness_binding: crate::NextWitnessBindingPolicy,
    ) -> Result<usize, AkitaError> {
        let current_coeffs = lp
            .open_commit_matrix
            .output_rank()
            .checked_mul(lp.role_dims().d_d())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive proof sizing overflow".to_string())
            })?;
        let next_commit_coeffs = next_lp
            .outer_commit_matrix
            .output_rank()
            .checked_mul(next_lp.role_dims().d_b())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive proof sizing overflow".to_string())
            })?;
        let rounds = sumcheck_rounds(lp.d_a(), output_witness_len);
        let b = 1usize << lp.log_basis_open;

        let proof = FoldLevelProof {
            extension_opening_reduction: None,
            v: RingVec::from_coeffs(vec![F::zero(); current_coeffs]),
            fold_grind_nonce: 0,
            stage1: dummy_stage1_proof(rounds, b),
            stage2: AkitaStage2Proof {
                sumcheck_proof: dummy_sumcheck(rounds, 3),
                next_witness_binding: match next_witness_binding {
                    crate::NextWitnessBindingPolicy::OuterCommitment => {
                        crate::NextWitnessBinding::OuterCommitment(RingVec::from_coeffs(vec![
                            F::zero();
                            next_commit_coeffs
                        ]))
                    }
                    crate::NextWitnessBindingPolicy::TerminalInnerState => {
                        crate::NextWitnessBinding::TerminalInnerState
                    }
                },
                next_w_eval: F::zero(),
            },
            stage3_sumcheck_proof: stage3_setup_ring_len.map(|setup_ring_len| {
                dummy_stage3_proof::<F>(lp.d_a(), setup_ring_len, output_witness_len)
            }),
        };
        Ok(proof.serialized_size(Compress::No))
    }

    #[test]
    fn planned_level_bytes_match_non_offloaded_payload_at_all_bases() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let next_lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            2,
            2,
            3,
            2,
            fold_challenge_config,
        );
        let output_witness_len = D * 8;

        for log_basis in 2..=6 {
            let lp = CommittedGroupParams::params_only(
                SisModulusProfileId::Q128OffsetA7F7,
                D,
                log_basis,
                2,
                2,
                2,
                fold_challenge_config,
            )
            .with_decomp(1, 1, 1, 1, 1)
            .unwrap();
            assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    output_witness_len,
                    Some(crate::NextWitnessBindingPolicy::OuterCommitment),
                )
                .unwrap(),
                exact_level_proof_bytes::<F>(
                    &lp,
                    &next_lp,
                    output_witness_len,
                    None,
                    crate::NextWitnessBindingPolicy::OuterCommitment,
                )
                .unwrap(),
                "planned level bytes should match the serialized non-offloaded body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_level_bytes_use_native_d_and_successor_b_dimensions() {
        const D_A: usize = 128;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let mut lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D_A,
            4,
            2,
            3,
            2,
            fold_challenge_config,
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();
        lp.outer_commit_matrix = crate::OuterCommitMatrixParams::new_unchecked(
            lp.outer_commit_matrix.security_policy(),
            lp.outer_commit_matrix.sis_table_key().table_digest,
            lp.outer_commit_matrix.sis_modulus_profile(),
            lp.outer_commit_matrix.output_rank(),
            lp.outer_commit_matrix.input_width() * 2,
            lp.outer_commit_matrix.coeff_linf_bound(),
            64,
        );
        lp.open_commit_matrix = crate::OpenCommitMatrixParams::new_unchecked(
            lp.open_commit_matrix.security_policy(),
            lp.open_commit_matrix.sis_table_key().table_digest,
            lp.open_commit_matrix.sis_modulus_profile(),
            lp.open_commit_matrix.output_rank(),
            lp.open_commit_matrix.input_width() * 4,
            lp.open_commit_matrix.coeff_linf_bound(),
            32,
        );

        let mut next_lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D_A,
            2,
            2,
            3,
            2,
            fold_challenge_config,
        );
        next_lp.outer_commit_matrix = crate::OuterCommitMatrixParams::new_unchecked(
            next_lp.outer_commit_matrix.security_policy(),
            next_lp.outer_commit_matrix.sis_table_key().table_digest,
            next_lp.outer_commit_matrix.sis_modulus_profile(),
            next_lp.outer_commit_matrix.output_rank(),
            next_lp.outer_commit_matrix.input_width() * 2,
            next_lp.outer_commit_matrix.coeff_linf_bound(),
            64,
        );
        next_lp.open_commit_matrix = crate::OpenCommitMatrixParams::new_unchecked(
            next_lp.open_commit_matrix.security_policy(),
            next_lp.open_commit_matrix.sis_table_key().table_digest,
            next_lp.open_commit_matrix.sis_modulus_profile(),
            next_lp.open_commit_matrix.output_rank(),
            next_lp.open_commit_matrix.input_width() * 2,
            next_lp.open_commit_matrix.coeff_linf_bound(),
            64,
        );

        let output_witness_len = D_A * 8;
        let planned = level_proof_bytes(
            128,
            128,
            &lp,
            Some(&next_lp),
            output_witness_len,
            Some(crate::NextWitnessBindingPolicy::OuterCommitment),
        )
        .unwrap();
        let serialized = exact_level_proof_bytes::<F>(
            &lp,
            &next_lp,
            output_witness_len,
            None,
            crate::NextWitnessBindingPolicy::OuterCommitment,
        )
        .unwrap();
        assert_eq!(planned, serialized);
    }

    #[test]
    fn terminal_inner_binding_removes_exactly_the_outer_commitment_bytes() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            4,
            2,
            2,
            2,
            fold_challenge_config,
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();
        let next_lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            2,
            2,
            3,
            2,
            fold_challenge_config,
        );
        let output_witness_len = D * 8;

        let outer = level_proof_bytes(
            128,
            128,
            &lp,
            Some(&next_lp),
            output_witness_len,
            Some(crate::NextWitnessBindingPolicy::OuterCommitment),
        )
        .unwrap();
        let terminal_inner = level_proof_bytes(
            128,
            128,
            &lp,
            None,
            output_witness_len,
            Some(crate::NextWitnessBindingPolicy::TerminalInnerState),
        )
        .unwrap();
        let expected_outer_commitment = proof_ring_vec_bytes(
            next_lp.outer_commit_matrix.output_rank(),
            D,
            field_bytes(128),
        );
        assert_eq!(outer - terminal_inner, expected_outer_commitment);
        assert_eq!(
            terminal_inner,
            exact_level_proof_bytes::<F>(
                &lp,
                &next_lp,
                output_witness_len,
                None,
                crate::NextWitnessBindingPolicy::TerminalInnerState,
            )
            .unwrap()
        );
    }

    #[test]
    fn level_proof_bytes_rejects_incomplete_intermediate_schedule() {
        const D: usize = 64;
        let lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            4,
            2,
            2,
            2,
            SparseChallengeConfig::pm1_only(3),
        )
        .with_decomp(1, 1, 1, 1, 1)
        .unwrap();

        let missing_successor = level_proof_bytes(
            128,
            128,
            &lp,
            None,
            D * 8,
            Some(crate::NextWitnessBindingPolicy::OuterCommitment),
        );
        assert!(matches!(
            missing_successor,
            Err(AkitaError::InvalidSetup(_))
        ));

        let missing_binding = level_proof_bytes(128, 128, &lp, None, D * 8, None);
        assert!(matches!(missing_binding, Err(AkitaError::InvalidSetup(_))));
    }

    #[test]
    fn stage3_setup_product_bytes_match_serialized_payload() {
        // The recursive stage-3 payload is priced separately from the direct
        // planner bytes. Check the formula against the real serialized
        // SetupSumcheckProof across representative (D, setup_ring_len, output_witness_len)
        // shapes, including non-power-of-two setup lengths that exercise lambda
        // padding and witness-longer shapes that exercise the fused round count.
        const CHALLENGE_BITS: u32 = 128;
        for &d in &[32usize, 64, 128] {
            for &setup_ring_len in &[1usize, 3, 8, 17, 64, 100] {
                for &output_witness_len in &[d, d * 8, d * 256] {
                    let proof = dummy_stage3_proof::<F>(d, setup_ring_len, output_witness_len);
                    let serialized = proof.claim.serialized_size(Compress::No)
                        + proof.setup_prefix_eval.serialized_size(Compress::No)
                        + proof.next_w_eval.serialized_size(Compress::No)
                        + proof.sumcheck.serialized_size(Compress::No);
                    assert_eq!(
                        stage3_setup_product_bytes(CHALLENGE_BITS, d, setup_ring_len, output_witness_len),
                        serialized,
                        "stage3 formula must match the serialized SetupSumcheckProof \
                         at D={d}, setup_ring_len={setup_ring_len}, output_witness_len={output_witness_len}"
                    );
                }
            }
        }
    }

    #[test]
    fn stage3_payload_is_additive_over_direct_level_bytes() {
        // The recursive stage-3 setup-product proof is pure overhead layered on
        // top of the direct-mode payload: a level proof carrying it must
        // serialize to exactly the direct `level_proof_bytes` plus
        // `stage3_setup_product_bytes`, with no other field affected.
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let next_lp = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            2,
            2,
            3,
            2,
            fold_challenge_config,
        );
        let output_witness_len = D * 8;
        // 100 is not a power of two, so the verifier pads lambda to 128.
        let setup_ring_len = 100usize;

        for log_basis in 2..=6 {
            let lp = CommittedGroupParams::params_only(
                SisModulusProfileId::Q128OffsetA7F7,
                D,
                log_basis,
                2,
                2,
                2,
                fold_challenge_config,
            )
            .with_decomp(1, 1, 1, 1, 1)
            .unwrap();

            let terminal_bytes = exact_level_proof_bytes::<F>(
                &lp,
                &next_lp,
                output_witness_len,
                None,
                crate::NextWitnessBindingPolicy::OuterCommitment,
            )
            .unwrap();
            let recursive_bytes = exact_level_proof_bytes::<F>(
                &lp,
                &next_lp,
                output_witness_len,
                Some(setup_ring_len),
                crate::NextWitnessBindingPolicy::OuterCommitment,
            )
            .unwrap();

            assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    output_witness_len,
                    Some(crate::NextWitnessBindingPolicy::OuterCommitment),
                )
                .unwrap(),
                terminal_bytes,
                "direct planner bytes must exclude the stage-3 payload at log_basis={log_basis}"
            );
            assert_eq!(
                recursive_bytes - terminal_bytes,
                stage3_setup_product_bytes(128, D, setup_ring_len, output_witness_len),
                "stage-3 payload must be additive over the direct level bytes at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_terminal_level_bytes_match_terminal_payload_at_all_bases() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let num_claims = 3;

        for log_basis in 2..=6 {
            let lp = CommittedGroupParams::params_only(
                SisModulusProfileId::Q128OffsetA7F7,
                D,
                log_basis,
                2,
                2,
                2,
                fold_challenge_config,
            )
            .with_decomp(1, 1, 1, 1, 1)
            .unwrap();

            let (terminal_response, witness_shape) = terminal_response_fixture(&lp, num_claims);
            let terminal_response_bytes_runtime = terminal_response.serialized_size(Compress::No);
            let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
                None,
                terminal_response,
                0,
            );

            let serialized_without_witness =
                terminal_proof.serialized_size(Compress::No) - terminal_response_bytes_runtime;

            // A direct terminal level carries only the `fold_grind_nonce`
            // (plus any extension-opening reduction, absent from this
            // fixture); this mirrors the planner's terminal-direct accounting.
            assert_eq!(
                FOLD_GRIND_NONCE_BYTES, serialized_without_witness,
                "planned terminal-level bytes should match the serialized terminal body \
                 (less terminal_response) at log_basis={log_basis}"
            );

            let scheduled_bytes = terminal_response_bytes(128, &witness_shape);
            assert!(
                scheduled_bytes >= terminal_response_bytes_runtime,
                "scheduled direct witness budget must cover serialized terminal response \
                 at log_basis={log_basis}"
            );
        }
    }
}
