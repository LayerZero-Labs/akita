//! Header-stripped proof-byte formula for one fold level, shared by the
//! offline planner DP, the schedule selector, and profiling tooling.
//!
//! This is the single source of truth for per-level proof-byte accounting:
//! [`level_proof_bytes`] scores one fold level. It is not on the
//! prover/verifier replay path. The compact-entry walker that sums a whole
//! proof (`schedule_from_entry`) lives in `akita-planner`, next to the
//! schedule-table representation it consumes.

use crate::layout::{field_bytes, proof_ring_vec_bytes, sumcheck_rounds};
use crate::{stage1_tree_stage_shapes, LevelParams, RelationMatrixRowLayout};

/// Fixed wire size of `fold_grind_nonce` on every fold level proof.
pub const FOLD_GRIND_NONCE_BYTES: usize = 4;

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

fn stage1_proof_bytes(rounds: usize, b: usize, elem_bytes: usize) -> usize {
    stage1_tree_stage_shapes(rounds, b)
        .into_iter()
        .map(|stage| {
            ({ sumcheck_bytes(rounds, stage.sumcheck_proof.1, elem_bytes) })
                + stage.child_claims * elem_bytes
        })
        .sum::<usize>()
        + elem_bytes
}

/// Header-stripped byte size of one folded proof level, parametrized by
/// [`RelationMatrixRowLayout`].
///
/// Ring-valued objects (`y`, `v`, next-witness commitment) serialize over
/// the base SIS field. Sumcheck objects and scalar evaluations serialize
/// over the challenge field, which may be a non-trivial extension of the
/// base field for small-prime configurations.
///
/// This prices the **direct-mode** two-stage fold payload only
/// (`SetupContributionMode::Direct`): the y/v ring blocks, the stage-1
/// range-check tree, the fused stage-2 sumcheck, and the next-level witness
/// commitment plus its evaluation. It deliberately **excludes** the optional
/// recursive stage-3 setup-product sumcheck
/// (`SetupContributionMode::Recursive`), whose per-level overhead is priced
/// separately by [`stage3_setup_product_bytes`] and is not fed into the
/// planner DP. The shipped schedules and the planner score the direct-mode
/// proof; recursive observed sizes are reported on top of that baseline.
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
    _num_claims: usize,
    layout: RelationMatrixRowLayout,
) -> usize {
    let base_elem_bytes = field_bytes(base_field_bits);
    let challenge_elem_bytes = field_bytes(challenge_field_bits);
    let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
    let sumcheck = sumcheck_bytes(rounds, 3, challenge_elem_bytes);
    match layout {
        RelationMatrixRowLayout::WithoutDBlock => FOLD_GRIND_NONCE_BYTES + sumcheck,
        RelationMatrixRowLayout::WithDBlock => {
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
            v_bytes
                + FOLD_GRIND_NONCE_BYTES
                + stage1_bytes
                + sumcheck
                + next_commit_bytes
                + next_eval_bytes
        }
    }
}

/// Header-stripped byte size of the recursive-mode stage-3 setup-product
/// sumcheck payload (`SetupSumcheckProof`) for one non-terminal fold level.
///
/// This is the proof-size overhead that `SetupContributionMode::Recursive`
/// adds on top of the direct-mode payload priced by [`level_proof_bytes`]. It
/// is reporting/assertion-only and is intentionally **not** fed into the
/// planner DP: the shipped schedules price the direct-mode fold, and recursive
/// observed sizes are reported separately.
///
/// The payload is the setup claim and the carried next-witness opening (two
/// challenge-field elements), followed by a degree-[`crate::SETUP_SUMCHECK_DEGREE`]
/// product sumcheck. Stage 3 fuses the setup-product term with the carried
/// witness term, so the serialized round count is the max of the setup domain
/// rounds (`log2(D) + log2(next_pow2(setup_ring_len))`) and the witness domain
/// rounds (`sumcheck_rounds(D, next_w_len)`).
///
/// `ring_dimension` and the next-power-of-two of `setup_ring_len` must be
/// powers of two; this offline helper is not on the verifier path.
pub fn stage3_setup_product_bytes(
    challenge_field_bits: u32,
    ring_dimension: usize,
    setup_ring_len: usize,
    next_w_len: usize,
) -> usize {
    let challenge_elem_bytes = field_bytes(challenge_field_bits);
    let ring_bits = ring_dimension.trailing_zeros() as usize;
    let lambda_bits = setup_ring_len.next_power_of_two().trailing_zeros() as usize;
    let setup_rounds = ring_bits + lambda_bits;
    let witness_rounds = sumcheck_rounds(ring_dimension, next_w_len);
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
    use crate::proof::{segment_typed_witness_shape_from_groups, SegmentTypedWitness};
    use crate::tail_golomb_rice_z_params;
    use crate::{
        direct_witness_bytes, AkitaIntermediateStage2Proof, AkitaLevelProof, AkitaStage1Proof,
        AkitaStage1StageProof, AkitaStage2Proof, CleartextWitnessProof, CleartextWitnessShape,
        RingVec, SetupSumcheckProof, SisModulusProfileId, TerminalLevelProof,
        SETUP_SUMCHECK_DEGREE,
    };

    type F = Prime128OffsetA7F7;

    fn segment_typed_final_witness(
        lp: &LevelParams,
        num_claims: usize,
    ) -> (CleartextWitnessProof<F>, CleartextWitnessShape) {
        let field_bits = F::modulus_bits();
        let shape = segment_typed_witness_shape_from_groups(
            lp,
            field_bits,
            [(lp as &dyn crate::LevelParamsLike, num_claims, num_claims, 1)],
            1,
            crate::TerminalQuotientMode::Include,
        )
        .expect("segment-typed witness shape");
        let CleartextWitnessShape::SegmentTyped(ref segment_shape) = shape else {
            panic!("expected segment-typed witness shape");
        };
        let layout = segment_shape.layout.clone();
        let group = layout.groups[0];
        let (rice_low_bits, zigzag_w) =
            tail_golomb_rice_z_params(lp, num_claims).expect("golomb z params");
        let z_payload =
            golomb_rice_encode_vec(&vec![0i64; group.z_coords], rice_low_bits, zigzag_w)
                .expect("encode zero z segment");
        let witness = SegmentTypedWitness {
            layout: layout.clone(),
            z_payloads: vec![z_payload],
            e_fields: RingVec::from_coeffs(vec![F::zero(); group.e_field_elems]),
            t_fields: RingVec::from_coeffs(vec![F::zero(); group.t_field_elems]),
            r_fields: RingVec::from_coeffs(vec![F::zero(); layout.r_field_elems]),
        };
        (CleartextWitnessProof::SegmentTyped(witness), shape)
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
            stages: stage1_tree_stage_shapes(rounds, b)
                .into_iter()
                .map(|shape| AkitaStage1StageProof {
                    sumcheck_proof: dummy_eq_factored_sumcheck(rounds, shape.sumcheck_proof.1),
                    child_claims: vec![F::zero(); shape.child_claims],
                })
                .collect(),
            s_claim: F::zero(),
        }
    }

    /// Build a degree-[`SETUP_SUMCHECK_DEGREE`] stage-3 setup-product proof
    /// whose round count matches the fused setup/carry verifier rounds.
    fn dummy_stage3_proof<F: FieldCore>(
        d: usize,
        setup_ring_len: usize,
        next_w_len: usize,
    ) -> SetupSumcheckProof<F> {
        let ring_bits = d.trailing_zeros() as usize;
        let lambda_bits = setup_ring_len.next_power_of_two().trailing_zeros() as usize;
        let setup_rounds = ring_bits + lambda_bits;
        let witness_rounds = sumcheck_rounds(d, next_w_len);
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
        lp: &LevelParams,
        next_lp: &LevelParams,
        next_w_len: usize,
        stage3_setup_ring_len: Option<usize>,
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

        let proof = AkitaLevelProof::Intermediate {
            extension_opening_reduction: None,
            v: RingVec::from_coeffs(vec![F::zero(); current_coeffs]),
            fold_grind_nonce: 0,
            stage1: dummy_stage1_proof(rounds, b),
            stage2: AkitaStage2Proof::Intermediate(AkitaIntermediateStage2Proof {
                sumcheck_proof: dummy_sumcheck(rounds, 3),
                next_w_commitment: RingVec::from_coeffs(vec![F::zero(); next_commit_coeffs]),
                next_w_eval: F::zero(),
            }),
            stage3_sumcheck_proof: stage3_setup_ring_len.map(|setup_ring_len| {
                dummy_stage3_proof::<F>(lp.ring_dimension, setup_ring_len, next_w_len)
            }),
        };
        Ok(proof.serialized_size(Compress::No))
    }

    #[test]
    fn planned_level_bytes_match_two_stage_payload_at_all_bases() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let next_lp = LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            2,
            2,
            3,
            2,
            fold_challenge_config,
        );
        let next_w_len = D * 8;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(
                SisModulusProfileId::Q128OffsetA7F7,
                D,
                log_basis,
                2,
                2,
                2,
                fold_challenge_config,
            )
            .with_decomp(1, 1, 1, 1)
            .unwrap();
            assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    next_w_len,
                    1,
                    RelationMatrixRowLayout::WithDBlock,
                ),
                exact_level_proof_bytes::<F>(&lp, &next_lp, next_w_len, None).unwrap(),
                "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn stage3_setup_product_bytes_match_serialized_payload() {
        // The recursive stage-3 payload is priced separately from the direct
        // planner bytes. Check the formula against the real serialized
        // SetupSumcheckProof across representative (D, setup_ring_len, next_w_len)
        // shapes, including non-power-of-two setup lengths that exercise lambda
        // padding and witness-longer shapes that exercise the fused round count.
        const CHALLENGE_BITS: u32 = 128;
        for &d in &[32usize, 64, 128] {
            for &setup_ring_len in &[1usize, 3, 8, 17, 64, 100] {
                for &next_w_len in &[d, d * 8, d * 256] {
                    let proof = dummy_stage3_proof::<F>(d, setup_ring_len, next_w_len);
                    let serialized = proof.claim.serialized_size(Compress::No)
                        + proof.setup_prefix_eval.serialized_size(Compress::No)
                        + proof.next_w_eval.serialized_size(Compress::No)
                        + proof.sumcheck.serialized_size(Compress::No);
                    assert_eq!(
                        stage3_setup_product_bytes(CHALLENGE_BITS, d, setup_ring_len, next_w_len),
                        serialized,
                        "stage3 formula must match the serialized SetupSumcheckProof \
                         at D={d}, setup_ring_len={setup_ring_len}, next_w_len={next_w_len}"
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
        let next_lp = LevelParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            D,
            2,
            2,
            3,
            2,
            fold_challenge_config,
        );
        let next_w_len = D * 8;
        // 100 is not a power of two, so the verifier pads lambda to 128.
        let setup_ring_len = 100usize;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(
                SisModulusProfileId::Q128OffsetA7F7,
                D,
                log_basis,
                2,
                2,
                2,
                fold_challenge_config,
            )
            .with_decomp(1, 1, 1, 1)
            .unwrap();

            let direct_bytes =
                exact_level_proof_bytes::<F>(&lp, &next_lp, next_w_len, None).unwrap();
            let recursive_bytes =
                exact_level_proof_bytes::<F>(&lp, &next_lp, next_w_len, Some(setup_ring_len))
                    .unwrap();

            assert_eq!(
                level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    next_w_len,
                    1,
                    RelationMatrixRowLayout::WithDBlock,
                ),
                direct_bytes,
                "direct planner bytes must exclude the stage-3 payload at log_basis={log_basis}"
            );
            assert_eq!(
                recursive_bytes - direct_bytes,
                stage3_setup_product_bytes(128, D, setup_ring_len, next_w_len),
                "stage-3 payload must be additive over the direct level bytes at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_terminal_level_bytes_match_terminal_payload_at_all_bases() {
        const D: usize = 64;
        let fold_challenge_config = SparseChallengeConfig::pm1_only(3);
        let next_w_len = D * 8;
        let num_claims = 3;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(
                SisModulusProfileId::Q128OffsetA7F7,
                D,
                log_basis,
                2,
                2,
                2,
                fold_challenge_config,
            )
            .with_decomp(1, 1, 1, 1)
            .unwrap();
            let rounds = sumcheck_rounds(D, next_w_len);

            let (final_witness, witness_shape) = segment_typed_final_witness(&lp, num_claims);
            let final_witness_bytes_runtime = final_witness.serialized_size(Compress::No);
            let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
                None,
                dummy_sumcheck(rounds, 3),
                final_witness,
                0,
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
                    RelationMatrixRowLayout::WithoutDBlock,
                ),
                serialized_without_witness,
                "planned terminal-level bytes should match the serialized terminal body \
                 (less final_witness) at log_basis={log_basis}"
            );

            let scheduled_bytes = direct_witness_bytes(128, &witness_shape);
            assert!(
                scheduled_bytes >= final_witness_bytes_runtime,
                "scheduled direct witness budget must cover serialized segment-typed witness \
                 at log_basis={log_basis}"
            );
        }
    }
}
