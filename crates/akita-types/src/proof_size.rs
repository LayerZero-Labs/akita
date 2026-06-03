//! Header-stripped proof-byte formula for one fold level, shared by the
//! offline planner DP, the schedule selector, and profiling tooling.
//!
//! This is the single source of truth for per-level proof-byte accounting:
//! [`level_proof_bytes`] scores one fold level. It is not on the
//! prover/verifier replay path. The compact-entry walker that sums a whole
//! proof (`schedule_from_entry`) lives in `akita-planner`, next to the
//! schedule-table representation it consumes.

use crate::layout::{field_bytes, proof_ring_vec_bytes, sumcheck_rounds};
use crate::{stage1_tree_stage_shapes, LevelParams, MRowLayout};

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
        MRowLayout::WithoutDBlock => y_bytes + sumcheck,
        MRowLayout::WithDBlock => {
            let next_lp = next_lp
                .expect("level_proof_bytes(WithDBlock) requires next_lp; caller must pass Some");
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
        AkitaStage2Proof, CleartextWitnessProof, CleartextWitnessShape, FlatRingVec, PackedDigits,
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
            stage3_sumcheck_proof: None,
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
                    MRowLayout::WithDBlock,
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

            let final_witness = CleartextWitnessProof::PackedDigits(PackedDigits::from_i8_digits(
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
                    MRowLayout::WithoutDBlock,
                ),
                serialized_without_witness,
                "planned terminal-level bytes should match the serialized terminal body \
                 (less final_witness) at log_basis={log_basis}"
            );

            assert_eq!(
                direct_witness_bytes(
                    128,
                    &CleartextWitnessShape::PackedDigits((final_w_num_elems, final_w_bits))
                ),
                final_witness_bytes_runtime,
                "direct_witness_bytes should match the serialized packed-digit \
                 final witness at log_basis={log_basis}"
            );
        }
    }
}
