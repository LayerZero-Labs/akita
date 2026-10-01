#[cfg(test)]
mod tests {
    //! Legacy structured-fixture cross-checks for the fixed-message
    //! layout. End-to-end emission and parser-bound reconciliation live
    //! in the PCS transcript-hardening and protocol-soundness suites.

    use akita_params::proof_size::*;
    use akita_params::{CommittedGroupParams, DigitRangePlan};

    use akita_challenges::SparseChallengeConfig;
    use akita_error::AkitaError;
    use akita_serialization::{AkitaSerialize, Compress};
    use akita_sumcheck::{EqFactoredSumcheckProof, SumcheckProof};
    use jolt_field::{
        CanonicalEncoding, Ext2, ExtField, Field, FpExt4, Prime128OffsetA7F7, Prime32Offset99,
        Prime64Offset59, Zero,
    };
    use jolt_poly::{CompressedPoly, OmittedConstantPoly};

    use crate::{
        AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof, FoldLevelProof,
        PhysicalL2NormProof, RingVec, SetupSumcheckProof, TerminalLevelProof, TerminalResponse,
    };
    use akita_params::golomb_rice::golomb_rice_encode_vec;
    use akita_params::sis::sis_l2_table_key_for_collision_sq;
    use akita_params::InnerCommitSecurityRoute;
    use akita_params::{
        terminal_response_bytes, OpeningClaimsLayout, PhysicalL2NormProofShape, SisL2TableDigest,
        SisModulusProfileId, TerminalResponseShape, DEFAULT_SIS_SECURITY_POLICY,
        SETUP_SUMCHECK_DEGREE,
    };

    type F = Prime128OffsetA7F7;
    // `pm1_only(3)` prices the fixtures' response cap 127 below A bucket 4095.
    const TEST_TERMINAL_A_BUCKET: u128 = 4_095;

    fn test_opening_layout(
        lp: &CommittedGroupParams,
        output_witness_len: usize,
    ) -> OpeningClaimsLayout {
        OpeningClaimsLayout::new(
            akita_params::sumcheck_rounds(lp.d_a(), output_witness_len),
            1,
        )
        .expect("test opening layout")
    }

    fn planned_level_proof_bytes(
        base_field_bits: u32,
        challenge_field_bits: u32,
        lp: &CommittedGroupParams,
        next_lp: Option<&CommittedGroupParams>,
        output_witness_len: usize,
        next_witness_binding: Option<akita_params::NextWitnessBindingPolicy>,
    ) -> Result<usize, AkitaError> {
        let opening_layout = test_opening_layout(lp, output_witness_len);
        let successor_ring_dimension = next_lp.map_or(lp.d_a(), CommittedGroupParams::d_a);
        let next_outer_payload = match next_witness_binding {
            Some(akita_params::NextWitnessBindingPolicy::OuterPayload) => next_lp,
            Some(akita_params::NextWitnessBindingPolicy::TerminalInnerState) => None,
            None => {
                return Err(AkitaError::InvalidSetup(
                    "test level is missing an outgoing witness binding".into(),
                ));
            }
        };
        let extension_degree = usize::try_from(challenge_field_bits / base_field_bits)
            .map_err(|_| AkitaError::InvalidSetup("test extension degree overflow".into()))?;
        let relation_geometry = lp.relation_address_geometry(
            &opening_layout,
            extension_degree,
            successor_ring_dimension,
            output_witness_len,
        )?;
        nonterminal_level_layout(
            base_field_bits,
            challenge_field_bits,
            lp,
            relation_geometry,
            next_outer_payload,
        )
        .and_then(NonterminalLevelLayout::encoded_len)
    }

    fn terminal_response_fixture(
        lp: &CommittedGroupParams,
        num_claims: usize,
    ) -> (TerminalResponse<F>, TerminalResponseShape) {
        let field_bits = F::MODULUS_BITS;
        let shape = TerminalResponseShape::from_groups(
            lp,
            field_bits,
            [(
                lp.final_group_scalar().expect("scalar final group"),
                num_claims,
                num_claims,
                1,
                127,
            )],
        )
        .expect("terminal response shape");
        let layout = shape.layout.clone();
        let group = layout.groups[0];
        let rice_low_bits = group.z_rice_low_bits;
        let zigzag_w = akita_params::golomb_rice::golomb_rice_zigzag_width(
            group.z_linf_cap.unwrap_or(i16::MAX as u128),
        );
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

    fn dummy_sumcheck<F: Field>(rounds: usize, degree: usize) -> SumcheckProof<F> {
        SumcheckProof {
            round_polys: (0..rounds)
                .map(|_| CompressedPoly::new(vec![F::zero(); degree]))
                .collect(),
        }
    }

    fn dummy_eq_factored_sumcheck<F: Field>(
        rounds: usize,
        degree: usize,
    ) -> EqFactoredSumcheckProof<F> {
        EqFactoredSumcheckProof {
            round_polys: (0..rounds)
                .map(|_| OmittedConstantPoly::new(vec![F::zero(); degree]))
                .collect(),
        }
    }

    fn dummy_stage1_proof<F: Field>(
        rounds: usize,
        b: usize,
        route: InnerCommitSecurityRoute,
    ) -> AkitaStage1Proof<F> {
        let plan = DigitRangePlan::new(b).expect("test range basis");
        let (stage_count, norm_proof) = match route {
            InnerCommitSecurityRoute::Linf(_) => (plan.stage_count(), None),
            InnerCommitSecurityRoute::L2 {
                norm_proof_shape, ..
            } => {
                let (subclaims, virtual_evaluations) = match norm_proof_shape {
                    PhysicalL2NormProofShape::Direct { .. } => (0, 1),
                    PhysicalL2NormProofShape::LimbGram { limb_count, .. } => (
                        norm_proof_shape.subclaim_count().expect("test subclaims"),
                        limb_count,
                    ),
                };
                (
                    plan.product_stage_arities().len(),
                    Some(PhysicalL2NormProof {
                        response_l2_sq: 0,
                        subclaims: vec![F::zero(); subclaims],
                        virtual_evaluations: vec![F::zero(); virtual_evaluations],
                        sumcheck: dummy_sumcheck(rounds, plan.leaf_degree() + 1),
                    }),
                )
            }
        };
        AkitaStage1Proof {
            stages: plan
                .stage_shapes(rounds)
                .into_iter()
                .take(stage_count)
                .map(|shape| AkitaStage1StageProof {
                    sumcheck_proof: dummy_eq_factored_sumcheck(rounds, shape.sumcheck_proof.1),
                    child_claims: vec![F::zero(); shape.child_claims],
                })
                .collect(),
            range_image_evaluation: F::zero(),
            norm_proof,
        }
    }

    /// Build a degree-[`SETUP_SUMCHECK_DEGREE`] stage-3 setup-product proof
    /// whose round count matches the setup verifier rounds.
    fn dummy_stage3_proof<F: Field>(d: usize, setup_ring_len: usize) -> SetupSumcheckProof<F> {
        let ring_bits = d.trailing_zeros() as usize;
        let lambda_bits = setup_ring_len.next_power_of_two().trailing_zeros() as usize;
        let rounds = ring_bits + lambda_bits;
        SetupSumcheckProof {
            claim: F::zero(),
            setup_prefix_eval: F::zero(),
            sumcheck: akita_sumcheck::SumcheckProof {
                round_polys: (0..rounds)
                    .map(|_| CompressedPoly::new(vec![F::zero(); SETUP_SUMCHECK_DEGREE]))
                    .collect(),
            },
        }
    }

    fn exact_level_proof_bytes<
        B: Field + CanonicalEncoding + AkitaSerialize,
        E: Field + ExtField<B> + AkitaSerialize,
    >(
        lp: &CommittedGroupParams,
        next_lp: &CommittedGroupParams,
        output_witness_len: usize,
        stage3_setup_ring_len: Option<usize>,
        next_witness_binding: akita_params::NextWitnessBindingPolicy,
    ) -> Result<usize, AkitaError> {
        let current_coeffs = lp.opening_payload_geometry()?.transmitted_coefficients();
        let next_commit_coeffs = next_lp.outer_payload_geometry()?.transmitted_coefficients();
        let opening_layout = test_opening_layout(lp, output_witness_len);
        let extension_degree = E::DEGREE;
        let rounds = lp
            .relation_address_geometry(
                &opening_layout,
                extension_degree,
                next_lp.d_a(),
                output_witness_len,
            )?
            .relation_point_variable_count();
        let b = 1usize << lp.open().digits.log_basis;

        let proof = FoldLevelProof {
            extension_opening_reduction: None,
            opening_payload: RingVec::from_coeffs(vec![B::zero(); current_coeffs]),
            stage1: dummy_stage1_proof::<E>(rounds, b, lp.inner().matrix.security_route()),
            stage2: AkitaStage2Proof {
                sumcheck_proof: dummy_sumcheck::<E>(rounds, 3),
                next_witness_binding: match next_witness_binding {
                    akita_params::NextWitnessBindingPolicy::OuterPayload => {
                        crate::NextWitnessBinding::OuterPayload(RingVec::from_coeffs(vec![
                            B::zero();
                            next_commit_coeffs
                        ]))
                    }
                    akita_params::NextWitnessBindingPolicy::TerminalInnerState => {
                        crate::NextWitnessBinding::TerminalInnerState
                    }
                },
                next_w_eval: E::zero(),
            },
            stage3_sumcheck_proof: stage3_setup_ring_len
                .map(|setup_ring_len| dummy_stage3_proof::<E>(lp.d_a(), setup_ring_len)),
        };
        Ok(proof.serialized_size(Compress::No))
    }

    fn assert_l2_bytes_match_for_profile<B, E>(
        profile: SisModulusProfileId,
        field_bits: u32,
        log_basis: u32,
        fold_digit_count: usize,
    ) where
        B: Field + CanonicalEncoding + AkitaSerialize,
        E: Field + ExtField<B> + AkitaSerialize,
    {
        const D: usize = 128;
        let challenge = SparseChallengeConfig::pm1_only(3);
        let mut lp = CommittedGroupParams::params_only(profile, D, log_basis, 4, 2, 2, challenge)
            .with_decomp(256, 8, 1, 1, 1)
            .unwrap();
        lp.own_group_mut().opening.num_digits_fold = fold_digit_count;
        let shape = PhysicalL2NormProofShape::derive(
            profile,
            lp.inner().matrix.input_width() * D,
            1usize << log_basis,
            fold_digit_count,
        )
        .expect("profile norm shape");
        let table_key = sis_l2_table_key_for_collision_sq(
            DEFAULT_SIS_SECURITY_POLICY,
            SisL2TableDigest::CURRENT,
            profile,
            D as u32,
            1u128 << 50,
        )
        .expect("profile L2 key");
        lp.own_group_mut().profile.inner.matrix =
            akita_params::InnerCommitMatrixParams::try_new_l2_with_min_rank(
                table_key,
                lp.inner().matrix.input_width(),
                1u128 << 32,
                shape,
            )
            .expect("profile L2 matrix");
        let next_lp = CommittedGroupParams::params_only(profile, D, 3, 2, 2, 2, challenge)
            .with_decomp(8, 8, 1, 1, 1)
            .unwrap();
        let output_witness_len = D * 8;
        assert_eq!(
            planned_level_proof_bytes(
                field_bits,
                128,
                &lp,
                Some(&next_lp),
                output_witness_len,
                Some(akita_params::NextWitnessBindingPolicy::OuterPayload),
            )
            .unwrap(),
            exact_level_proof_bytes::<B, E>(
                &lp,
                &next_lp,
                output_witness_len,
                None,
                akita_params::NextWitnessBindingPolicy::OuterPayload,
            )
            .unwrap(),
            "planned L2 bytes must match {profile:?} serialization for {shape:?}"
        );
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
                planned_level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    output_witness_len,
                    Some(akita_params::NextWitnessBindingPolicy::OuterPayload),
                )
                .unwrap(),
                exact_level_proof_bytes::<F, F>(
                    &lp,
                    &next_lp,
                    output_witness_len,
                    None,
                    akita_params::NextWitnessBindingPolicy::OuterPayload,
                )
                .unwrap(),
                "planned level bytes should match the serialized non-offloaded body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_level_bytes_follow_successor_padded_relation_domain() {
        let challenge = SparseChallengeConfig::pm1_only(3);
        let current = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            3,
            2,
            2,
            2,
            challenge,
        )
        .with_decomp(1, 1, 2, 2, 2)
        .unwrap();
        let successor = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            128,
            3,
            2,
            2,
            2,
            challenge,
        )
        .with_decomp(1, 1, 2, 2, 2)
        .unwrap();
        let output_witness_len = 64;
        let opening_layout = OpeningClaimsLayout::new(6, 1).unwrap();
        let rounds = current
            .relation_address_geometry(&opening_layout, 1, successor.d_a(), output_witness_len)
            .unwrap()
            .relation_point_variable_count();
        assert_eq!(rounds, 7);
        let successor_padded_terminal_eor = akita_params::extension_opening_reduction_level_bytes(
            128,
            2,
            akita_params::PolynomialGroupLayout::singleton(rounds),
        )
        .unwrap();
        let stale_terminal_eor = akita_params::extension_opening_reduction_level_bytes(
            128,
            2,
            akita_params::PolynomialGroupLayout::singleton(akita_params::sumcheck_rounds(
                current.d_a(),
                output_witness_len,
            )),
        )
        .unwrap();
        assert!(successor_padded_terminal_eor > stale_terminal_eor);

        assert_eq!(
            nonterminal_level_layout(
                128,
                128,
                &current,
                current
                    .relation_address_geometry(
                        &opening_layout,
                        1,
                        successor.d_a(),
                        output_witness_len,
                    )
                    .unwrap(),
                Some(&successor),
            )
            .and_then(NonterminalLevelLayout::encoded_len)
            .unwrap(),
            exact_level_proof_bytes::<F, F>(
                &current,
                &successor,
                output_witness_len,
                None,
                akita_params::NextWitnessBindingPolicy::OuterPayload,
            )
            .unwrap()
        );
    }

    #[test]
    fn planned_level_bytes_match_serialized_l2_payloads() {
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
        let table_key = sis_l2_table_key_for_collision_sq(
            DEFAULT_SIS_SECURITY_POLICY,
            SisL2TableDigest::CURRENT,
            SisModulusProfileId::Q128OffsetA7F7,
            D as u32,
            1u128 << 50,
        )
        .expect("generated L2 key");
        let shapes = [
            PhysicalL2NormProofShape::Direct {
                physical_response_len: 512,
            },
            PhysicalL2NormProofShape::LimbGram {
                physical_response_len: 512,
                block_len: 32,
                limb_count: 3,
            },
        ];

        for shape in shapes {
            let mut lp = CommittedGroupParams::params_only(
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
            lp.own_group_mut().profile.inner.matrix =
                akita_params::InnerCommitMatrixParams::try_new_l2_with_min_rank(
                    table_key,
                    shape.physical_response_len() / D,
                    1u128 << 30,
                    shape,
                )
                .expect("audited L2 matrix");
            let output_witness_len = D * 8;
            assert_eq!(
                planned_level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    output_witness_len,
                    Some(akita_params::NextWitnessBindingPolicy::OuterPayload),
                )
                .unwrap(),
                exact_level_proof_bytes::<F, F>(
                    &lp,
                    &next_lp,
                    output_witness_len,
                    None,
                    akita_params::NextWitnessBindingPolicy::OuterPayload,
                )
                .unwrap(),
                "planned L2 bytes should match serialized body for {shape:?}"
            );
        }
    }

    #[test]
    fn planned_l2_bytes_match_small_field_extension_serialization() {
        assert_l2_bytes_match_for_profile::<Prime32Offset99, FpExt4<Prime32Offset99>>(
            SisModulusProfileId::Q32Offset99,
            32,
            4,
            3,
        );
        assert_l2_bytes_match_for_profile::<Prime64Offset59, Ext2<Prime64Offset59>>(
            SisModulusProfileId::Q64Offset59,
            64,
            6,
            2,
        );
    }

    #[test]
    fn planned_level_bytes_use_fixed_compressed_d_and_successor_b_payloads() {
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
        lp.own_group_mut().profile.outer.matrix =
            akita_params::OuterCommitMatrixParams::new_unchecked(
                lp.outer().matrix.security_policy(),
                lp.outer().matrix.sis_table_key().table_digest,
                lp.outer().matrix.sis_modulus_profile(),
                lp.outer().matrix.output_rank(),
                lp.outer().matrix.input_width() * 2,
                lp.outer().matrix.coeff_linf_bound(),
                64,
            );
        lp.open_matrix = akita_params::OpenCommitMatrixParams::new_unchecked(
            lp.open().matrix.security_policy(),
            lp.open().matrix.sis_table_key().table_digest,
            lp.open().matrix.sis_modulus_profile(),
            lp.open().matrix.output_rank(),
            lp.open().matrix.input_width() * 4,
            lp.open().matrix.coeff_linf_bound(),
            64,
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
        next_lp.own_group_mut().profile.outer.matrix =
            akita_params::OuterCommitMatrixParams::new_unchecked(
                next_lp.outer().matrix.security_policy(),
                next_lp.outer().matrix.sis_table_key().table_digest,
                next_lp.outer().matrix.sis_modulus_profile(),
                next_lp.outer().matrix.output_rank(),
                next_lp.outer().matrix.input_width() * 2,
                next_lp.outer().matrix.coeff_linf_bound(),
                64,
            );
        next_lp.open_matrix = akita_params::OpenCommitMatrixParams::new_unchecked(
            next_lp.open().matrix.security_policy(),
            next_lp.open().matrix.sis_table_key().table_digest,
            next_lp.open().matrix.sis_modulus_profile(),
            next_lp.open().matrix.output_rank(),
            next_lp.open().matrix.input_width() * 2,
            next_lp.open().matrix.coeff_linf_bound(),
            64,
        );

        let output_witness_len = D_A * 8;
        let planned = planned_level_proof_bytes(
            128,
            128,
            &lp,
            Some(&next_lp),
            output_witness_len,
            Some(akita_params::NextWitnessBindingPolicy::OuterPayload),
        )
        .unwrap();
        let serialized = exact_level_proof_bytes::<F, F>(
            &lp,
            &next_lp,
            output_witness_len,
            None,
            akita_params::NextWitnessBindingPolicy::OuterPayload,
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

        let outer = planned_level_proof_bytes(
            128,
            128,
            &lp,
            Some(&next_lp),
            output_witness_len,
            Some(akita_params::NextWitnessBindingPolicy::OuterPayload),
        )
        .unwrap();
        let terminal_inner = planned_level_proof_bytes(
            128,
            128,
            &lp,
            None,
            output_witness_len,
            Some(akita_params::NextWitnessBindingPolicy::TerminalInnerState),
        )
        .unwrap();
        assert_eq!(
            outer - terminal_inner,
            akita_params::COMPRESSION_TARGET_BYTES
        );
        assert_eq!(
            terminal_inner,
            exact_level_proof_bytes::<F, F>(
                &lp,
                &next_lp,
                output_witness_len,
                None,
                akita_params::NextWitnessBindingPolicy::TerminalInnerState,
            )
            .unwrap()
        );
    }

    #[test]
    fn test_pricing_helper_rejects_missing_binding_policy() {
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

        let missing_binding = planned_level_proof_bytes(128, 128, &lp, None, D * 8, None);
        assert!(matches!(missing_binding, Err(AkitaError::InvalidSetup(_))));
    }

    #[test]
    fn stage3_setup_product_bytes_match_serialized_payload() {
        // The recursive stage-3 payload is priced separately from the direct
        // planner bytes. Check the formula against the real serialized
        // SetupSumcheckProof across representative (D, setup_ring_len) shapes,
        // including non-power-of-two setup lengths that exercise lambda padding.
        const CHALLENGE_BITS: u32 = 128;
        for &d in &[32usize, 64, 128] {
            for &setup_ring_len in &[1usize, 3, 8, 17, 64, 100] {
                let proof = dummy_stage3_proof::<F>(d, setup_ring_len);
                let serialized = proof.claim.serialized_size(Compress::No)
                    + proof.setup_prefix_eval.serialized_size(Compress::No)
                    + proof.sumcheck.serialized_size(Compress::No);
                assert_eq!(
                    stage3_setup_product_bytes(CHALLENGE_BITS, d, setup_ring_len),
                    serialized,
                    "stage3 formula must match the serialized SetupSumcheckProof \
                     at D={d}, setup_ring_len={setup_ring_len}"
                );
            }
        }
    }

    #[test]
    fn stage3_payload_is_additive_over_direct_level_bytes() {
        // The recursive stage-3 setup-product proof is pure overhead layered on
        // top of the direct-mode payload: a level proof carrying it must
        // serialize to exactly the direct level layout plus
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

            let terminal_bytes = exact_level_proof_bytes::<F, F>(
                &lp,
                &next_lp,
                output_witness_len,
                None,
                akita_params::NextWitnessBindingPolicy::OuterPayload,
            )
            .unwrap();
            let recursive_bytes = exact_level_proof_bytes::<F, F>(
                &lp,
                &next_lp,
                output_witness_len,
                Some(setup_ring_len),
                akita_params::NextWitnessBindingPolicy::OuterPayload,
            )
            .unwrap();

            assert_eq!(
                planned_level_proof_bytes(
                    128,
                    128,
                    &lp,
                    Some(&next_lp),
                    output_witness_len,
                    Some(akita_params::NextWitnessBindingPolicy::OuterPayload),
                )
                .unwrap(),
                terminal_bytes,
                "direct planner bytes must exclude the stage-3 payload at log_basis={log_basis}"
            );
            assert_eq!(
                recursive_bytes - terminal_bytes,
                stage3_setup_product_bytes(128, D, setup_ring_len),
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
            let mut lp = CommittedGroupParams::params_only(
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
            let inner = lp.inner().matrix;
            lp.own_group_mut().profile.inner.matrix =
                akita_params::InnerCommitMatrixParams::new_unchecked(
                    inner.security_policy(),
                    inner
                        .sis_table_key()
                        .expect("L infinity test matrix")
                        .table_digest,
                    inner.sis_modulus_profile(),
                    inner.output_rank(),
                    inner.input_width(),
                    TEST_TERMINAL_A_BUCKET,
                    inner.ring_dimension(),
                );

            let (terminal_response, witness_shape) = terminal_response_fixture(&lp, num_claims);
            let terminal_response_bytes_runtime = terminal_response.serialized_size(Compress::No);
            let terminal_proof = TerminalLevelProof::<F, F>::new_with_extension_opening_reduction(
                None,
                terminal_response,
            );

            let serialized_without_witness =
                terminal_proof.serialized_size(Compress::No) - terminal_response_bytes_runtime;

            // Nonce messages are accounted separately.
            assert_eq!(
                0, serialized_without_witness,
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
