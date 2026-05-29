use super::*;

#[derive(Clone)]
struct Fp32RingSubfieldRootFoldCfg;
#[derive(Clone)]
struct Fp32RingSubfieldOuterFallbackCfg;

impl Fp32RingSubfieldRootFoldCfg {
    fn root_lp() -> LevelParams {
        LevelParams::params_only(
            akita_types::SisModulusFamily::Q32,
            Self::D,
            3,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(0, 0, 12, 12, 12, 0)
        .unwrap()
    }
}

fn fp32_ring_subfield_setup_matrix_size<F>(
    lp: &LevelParams,
    max_num_claims: usize,
) -> Result<(usize, usize), AkitaError>
where
    F: akita_field::CanonicalField,
{
    let _field_marker = core::marker::PhantomData::<F>;
    let outer_width = lp.outer_width();
    #[cfg(feature = "zk")]
    let outer_width = {
        outer_width
            .checked_add(akita_types::zk::blinding_column_count::<F>(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("ZK outer width overflow".to_string()))?
    };

    Ok((
        lp.a_key
            .row_len()
            .max(lp.b_key.row_len())
            .max(lp.d_key.row_len()),
        lp.inner_width().max(outer_width).max(
            lp.d_matrix_width()
                .checked_mul(max_num_claims.max(1))
                .ok_or_else(|| AkitaError::InvalidSetup("D matrix width overflow".to_string()))?,
        ),
    ))
}

impl CommitmentConfig for Fp32RingSubfieldRootFoldCfg {
    type Field = akita_field::Prime32Offset99;
    type ClaimField = akita_field::RingSubfieldFp4<Self::Field>;
    type ChallengeField = Self::ClaimField;

    const D: usize = 16;

    fn decomposition() -> akita_types::DecompositionParams {
        akita_types::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        }
    }

    fn stage1_challenge_config(
        _d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Ok(akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        })
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }

    fn audited_root_rank(_role: akita_types::AjtaiRole, _max_num_vars: usize) -> usize {
        1
    }

    fn envelope(_max_num_vars: usize) -> akita_types::CommitmentEnvelope {
        akita_types::CommitmentEnvelope {
            max_n_a: 1,
            max_n_b: 1,
            max_n_d: 1,
        }
    }

    fn max_setup_matrix_size(
        _max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        let max_num_claims = max_num_batched_polys
            .checked_mul(max_num_points)
            .ok_or_else(|| AkitaError::InvalidSetup("claim count overflow".to_string()))?;
        let lp = Self::root_lp();
        fp32_ring_subfield_setup_matrix_size::<Self::Field>(&lp, max_num_claims)
    }

    fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = akita_types::scale_batched_root_layout(
            &Self::root_lp(),
            incidence.num_claims(),
            Self::stage1_challenge_config(Self::D)
                .expect("stage1 challenge config")
                .l1_norm(),
            Self::decomposition().field_bits(),
        )?;
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            incidence.num_points(),
            incidence.num_polynomials(),
            incidence.num_claims(),
            incidence.num_public_rows(),
            akita_types::MRowLayout::Terminal,
        )?;
        let compact_w_len = w_ring * Self::D;
        Ok(akita_types::Schedule {
            steps: vec![
                Step::Fold(akita_types::FoldStep {
                    params: lp.clone(),
                    current_w_len: akita_types::root_current_w_len(&lp),
                    delta_fold_per_poly: lp.num_digits_fold,
                    w_ring,
                    next_w_len: compact_w_len,
                    level_bytes: 0,
                }),
                Step::Direct(akita_types::DirectStep {
                    current_w_len: compact_w_len,
                    witness_shape: akita_types::DirectWitnessShape::PackedDigits((
                        compact_w_len,
                        3,
                    )),
                    direct_bytes: compact_w_len,
                    commit_params: None,
                    // Stub fixture: terminal-direct level params equal the
                    // fold's `lp` (matches the deleted
                    // `Cfg::level_params_with_log_basis` override that
                    // returned `Self::root_lp()`).
                    level_params: Some(lp.clone()),
                }),
            ],
            total_bytes: 0,
        })
    }
}

impl Fp32RingSubfieldOuterFallbackCfg {
    fn root_lp() -> LevelParams {
        LevelParams::params_only(
            akita_types::SisModulusFamily::Q32,
            Self::D,
            3,
            1,
            1,
            1,
            akita_challenges::SparseChallengeConfig::Uniform {
                weight: 1,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(1, 0, 12, 12, 12, 0)
        .unwrap()
    }
}

impl CommitmentConfig for Fp32RingSubfieldOuterFallbackCfg {
    type Field = akita_field::Prime32Offset99;
    type ClaimField = akita_field::RingSubfieldFp4<Self::Field>;
    type ChallengeField = Self::ClaimField;

    const D: usize = 16;

    fn decomposition() -> akita_types::DecompositionParams {
        akita_types::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        }
    }

    fn stage1_challenge_config(
        _d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Ok(akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        })
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        akita_types::SisModulusFamily::Q32
    }

    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, AkitaError> {
        Ok(None)
    }

    fn audited_root_rank(_role: akita_types::AjtaiRole, _max_num_vars: usize) -> usize {
        1
    }

    fn envelope(_max_num_vars: usize) -> akita_types::CommitmentEnvelope {
        akita_types::CommitmentEnvelope {
            max_n_a: 1,
            max_n_b: 1,
            max_n_d: 1,
        }
    }

    fn max_setup_matrix_size(
        _max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), AkitaError> {
        let max_num_claims = max_num_batched_polys
            .checked_mul(max_num_points)
            .ok_or_else(|| AkitaError::InvalidSetup("claim count overflow".to_string()))?;
        let lp = Self::root_lp();
        fp32_ring_subfield_setup_matrix_size::<Self::Field>(&lp, max_num_claims)
    }

    fn log_basis_search_range(_inputs: AkitaScheduleInputs) -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = akita_types::scale_batched_root_layout(
            &Self::root_lp(),
            incidence.num_claims(),
            Self::stage1_challenge_config(Self::D)
                .expect("stage1 challenge config")
                .l1_norm(),
            Self::decomposition().field_bits(),
        )?;
        // Single-fold schedule: the root IS the terminal fold, so its
        // shipped `w` is built under MRowLayout::Terminal (no D-block in
        // the per-row `r` quotients). The schedule's `next_w_len` and the
        // following Direct step's witness shape must match that reduced
        // length.
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            incidence.num_points(),
            incidence.num_polynomials(),
            incidence.num_claims(),
            incidence.num_public_rows(),
            akita_types::MRowLayout::Terminal,
        )?;
        let next_w_len = w_ring * Self::D;
        Ok(akita_types::Schedule {
            steps: vec![
                Step::Fold(akita_types::FoldStep {
                    params: lp.clone(),
                    current_w_len: akita_types::root_current_w_len(&lp),
                    delta_fold_per_poly: lp.num_digits_fold,
                    w_ring,
                    next_w_len,
                    level_bytes: 0,
                }),
                Step::Direct(akita_types::DirectStep {
                    current_w_len: next_w_len,
                    witness_shape: akita_types::DirectWitnessShape::PackedDigits((next_w_len, 3)),
                    direct_bytes: next_w_len,
                    commit_params: None,
                    // Stub fixture: terminal-direct level params equal the
                    // fold's `lp` (matches the deleted
                    // `Cfg::level_params_with_log_basis` override that
                    // returned `Self::root_lp()`).
                    level_params: Some(lp.clone()),
                }),
            ],
            total_bytes: 0,
        })
    }
}

#[test]
fn fp32_ring_subfield_root_fold_roundtrip_uses_extension_gamma() {
    type SmallCfg = Fp32RingSubfieldRootFoldCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ClaimField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 1;
    type SmallScheme = AkitaCommitmentScheme<SMALL_D, SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals = (0..len)
        .map(|idx| SmallF::from_u64((3 * idx as u64) + 9))
        .collect::<Vec<_>>();
    let poly = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals).unwrap();
    let point = (0..NUM_VARS)
        .map(|idx| {
            SmallE::new([
                SmallF::from_u64((idx + 5) as u64),
                SmallF::from_u64((idx + 7) as u64),
                SmallF::from_u64((idx + 11) as u64),
                SmallF::from_u64((idx + 13) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let weights = lagrange_weights(&point).unwrap();
    let opening = evals
        .iter()
        .zip(weights.iter())
        .fold(SmallE::zero(), |acc, (&coeff, &weight)| {
            acc + weight * SmallE::lift_base(coeff)
        });

    let setup =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let (commitment, hint) = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

    let poly_refs = [&poly];
    let commitments = [commitment];
    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    // After Phase 1, a tiny `NUM_VARS=1` schedule has a single fold level so
    // the root is the `Terminal` variant (not `Fold`). Both shapes carry an
    // optional extension-opening reduction payload; this test asserts the
    // payload is absent at the root in the degree-1 extension case.
    let root_extension_opening_reduction = match &proof.root {
        akita_types::AkitaBatchedRootProof::Fold(fold) => fold.extension_opening_reduction.as_ref(),
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            terminal.extension_opening_reduction.as_ref()
        }
        akita_types::AkitaBatchedRootProof::Direct { .. } => {
            panic!("root-direct proof has no folded root extension-opening reduction")
        }
    };
    assert!(
        root_extension_opening_reduction.is_none(),
        "root fold must not carry an unchecked extension-opening reduction payload"
    );

    let openings = [opening];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .unwrap();

    let wrong_openings = [opening + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &wrong_openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );
    assert!(result.is_err());

    let wrong_point = [point[0] + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &wrong_point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );
    assert!(result.is_err());
}

#[test]
fn fp32_ring_subfield_outer_extension_uses_root_tensor_projection() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ClaimField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 5;
    type SmallScheme = AkitaCommitmentScheme<SMALL_D, SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals_a = (0..len)
        .map(|idx| SmallF::from_u64((idx as u64) + 3))
        .collect::<Vec<_>>();
    let evals_b = (0..len)
        .map(|idx| SmallF::from_u64((2 * idx as u64) + 7))
        .collect::<Vec<_>>();
    let poly_a = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals_a).unwrap();
    let poly_b = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals_b).unwrap();
    let point = (0..NUM_VARS)
        .map(|idx| {
            SmallE::new([
                SmallF::from_u64((idx + 2) as u64),
                SmallF::from_u64((idx + 4) as u64),
                SmallF::from_u64((idx + 6) as u64),
                SmallF::from_u64((idx + 8) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let weights = lagrange_weights(&point).unwrap();
    let opening_a = evals_a
        .iter()
        .zip(weights.iter())
        .fold(SmallE::zero(), |acc, (&coeff, &weight)| {
            acc + weight * SmallE::lift_base(coeff)
        });
    let opening_b = evals_b
        .iter()
        .zip(weights.iter())
        .fold(SmallE::zero(), |acc, (&coeff, &weight)| {
            acc + weight * SmallE::lift_base(coeff)
        });

    let setup =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 2, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let poly_refs = [&poly_a, &poly_b];
    let (commitment, hint) = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &poly_refs,
    )
    .unwrap();
    let commitments = [commitment];
    let openings = [opening_a, opening_b];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-outer-direct");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    // After Phase 1, the root variant depends on the schedule: multi-fold
    // produces `Fold`, single-fold produces `Terminal`. Both carry the
    // extension-opening reduction payload as `Option`.
    let root_extension_opening_reduction = match &proof.root {
        akita_types::AkitaBatchedRootProof::Fold(fold) => fold.extension_opening_reduction.as_ref(),
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            terminal.extension_opening_reduction.as_ref()
        }
        akita_types::AkitaBatchedRootProof::Direct { .. } => {
            panic!("root-direct proof has no folded root extension-opening reduction")
        }
    };
    assert!(
        root_extension_opening_reduction.is_some(),
        "root tensor projection must prove the extension-opening reduction"
    );

    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-outer-direct");
    <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .unwrap();

    let wrong_openings = [opening_a, opening_b + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-outer-direct");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: &wrong_openings[..],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    );
    assert!(result.is_err());
}

#[test]
fn fp32_ring_subfield_multipoint_extension_uses_root_tensor_projection() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ClaimField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 5;
    type SmallScheme = AkitaCommitmentScheme<SMALL_D, SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals = (0..len)
        .map(|idx| SmallF::from_u64((3 * idx as u64) + 5))
        .collect::<Vec<_>>();
    let poly = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals).unwrap();
    let point_a = (0..NUM_VARS)
        .map(|idx| {
            SmallE::new([
                SmallF::from_u64((idx + 3) as u64),
                SmallF::from_u64((idx + 5) as u64),
                SmallF::from_u64((idx + 7) as u64),
                SmallF::from_u64((idx + 9) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let point_b = (0..NUM_VARS)
        .map(|idx| {
            SmallE::new([
                SmallF::from_u64((idx + 11) as u64),
                SmallF::from_u64((idx + 13) as u64),
                SmallF::from_u64((idx + 17) as u64),
                SmallF::from_u64((idx + 19) as u64),
            ])
        })
        .collect::<Vec<_>>();
    let opening_at = |point: &[SmallE]| {
        let weights = lagrange_weights(point).unwrap();
        evals
            .iter()
            .zip(weights.iter())
            .fold(SmallE::zero(), |acc, (&coeff, &weight)| {
                acc + weight * SmallE::lift_base(coeff)
            })
    };
    let opening_a = opening_at(&point_a);
    let opening_b = opening_at(&point_b);

    let setup =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 2, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let poly_refs = [&poly];
    let (commitment, hint) = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &poly_refs,
    )
    .unwrap();
    let commitments = [commitment];
    let openings_a = [opening_a];
    let openings_b = [opening_b];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-multipoint-direct");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![
            (
                &point_a[..],
                CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint: hint.clone(),
                },
            ),
            (
                &point_b[..],
                CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint,
                },
            ),
        ],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    // After Phase 1, the root variant depends on the schedule: multi-fold
    // produces `Fold`, single-fold produces `Terminal`. Both carry the
    // extension-opening reduction payload as `Option`.
    let root_extension_opening_reduction = match &proof.root {
        akita_types::AkitaBatchedRootProof::Fold(fold) => fold.extension_opening_reduction.as_ref(),
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            terminal.extension_opening_reduction.as_ref()
        }
        akita_types::AkitaBatchedRootProof::Direct { .. } => {
            panic!("root-direct proof has no folded root extension-opening reduction")
        }
    };
    assert!(
        root_extension_opening_reduction.is_some(),
        "root tensor projection must prove the extension-opening reduction"
    );

    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-multipoint-direct");
    <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![
            (
                &point_a[..],
                CommittedOpenings {
                    openings: &openings_a[..],
                    commitment: &commitments[0],
                },
            ),
            (
                &point_b[..],
                CommittedOpenings {
                    openings: &openings_b[..],
                    commitment: &commitments[0],
                },
            ),
        ],
        BasisMode::Lagrange,
    )
    .unwrap();

    let wrong_openings_b = [opening_b + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-multipoint-direct");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![
            (
                &point_a[..],
                CommittedOpenings {
                    openings: &openings_a[..],
                    commitment: &commitments[0],
                },
            ),
            (
                &point_b[..],
                CommittedOpenings {
                    openings: &wrong_openings_b[..],
                    commitment: &commitments[0],
                },
            ),
        ],
        BasisMode::Lagrange,
    );
    assert!(
        result.is_err(),
        "root tensor projection must reject a wrong claim at any point"
    );
}
