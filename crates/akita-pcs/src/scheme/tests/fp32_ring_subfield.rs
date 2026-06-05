use super::*;

/// Scale a per-polynomial root layout to a batched root layout without
/// SIS-floor audit on the scaled B/D keys (synthetic fixture only).
fn scale_batched_root_layout_unchecked(
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    if num_claims == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    let d = root_lp.ring_dimension;
    let b_col_len = root_lp
        .b_key
        .col_len()
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched outer width overflow".to_string()))?;
    let d_col_len = root_lp
        .d_key
        .col_len()
        .checked_mul(num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("batched D width overflow".to_string()))?;
    let mut scaled = root_lp.clone();
    scaled.b_key = akita_types::AjtaiKeyParams::new_unchecked(
        scaled.b_key.sis_family(),
        scaled.b_key.row_len(),
        b_col_len,
        scaled.b_key.collision_inf(),
        d,
    );
    scaled.d_key = akita_types::AjtaiKeyParams::new_unchecked(
        scaled.d_key.sis_family(),
        scaled.d_key.row_len(),
        d_col_len,
        scaled.d_key.collision_inf(),
        d,
    );
    Ok(scaled)
}

#[derive(Clone)]
struct Fp32RingSubfieldRootFoldCfg;
#[derive(Clone)]
struct Fp32RingSubfieldDirectRootFoldCfg;
#[derive(Clone)]
struct Fp32RingSubfieldOuterFallbackCfg;

/// Synthetic root `LevelParams` for the two fp32 ring-subfield test
/// fixtures.
///
/// Both fixtures share the same `(family, D, log_basis, log_commit_bound,
/// stage1, ring_subfield)` setup and only differ in their
/// `with_decomp` arguments. Constructs the placeholder via
/// `params_only` and stamps the A and B/D collision buckets via
/// `new_unchecked` so the layout carries real buckets instead of the
/// `0` `params_only` default. This `(family, D)` is intentionally
/// outside the audited SIS-floor tables, so the fixture scales via
/// [`scale_batched_root_layout_unchecked`] rather than the strict,
/// table-audited expansion path.
fn fp32_ring_subfield_root_lp(m_vars: usize) -> LevelParams {
    use akita_types::AjtaiKeyParams;
    let sis_family = akita_types::SisModulusFamily::Q32;
    // Commit ring dimension must equal the static `D` the scheme dispatches
    // (`DensePoly::<SmallF, D>` / `validate_commit_level_params::<F, D>`); both
    // fixtures pin `D = 32`. `ring_subfield = 2` below is `RingSubfieldFp4`'s
    // embedding norm bound (a claim-field property), not `D / d`, so the
    // collision buckets are independent of this dimension.
    let d: usize = 32;
    let stage1 = akita_challenges::SparseChallengeConfig::Uniform {
        weight: 1,
        nonzero_coeffs: vec![-1, 1],
    };
    // Match the verifier-reachable derivation for this fixture's
    // `(log_basis=3, log_commit_bound=32, stage1.inf_norm=1, ring_subfield=2)`:
    // `bd_raw = 7`, `a_collision_raw = 7 * 1 * 2 = 14` → bucket `15`,
    // `bd_collision_raw = 7` → bucket `7`.
    let a_bucket: u32 = 15;
    let bd_bucket: u32 = 7;
    let mut params = LevelParams::params_only(sis_family, d, 3, 1, 1, 1, stage1);
    params.a_key = AjtaiKeyParams::new_unchecked(sis_family, 1, 0, a_bucket, d);
    params.b_key = AjtaiKeyParams::new_unchecked(sis_family, 1, 0, bd_bucket, d);
    params.d_key = AjtaiKeyParams::new_unchecked(sis_family, 1, 0, bd_bucket, d);
    params.with_decomp(m_vars, 0, 12, 12, 0).unwrap()
}

impl Fp32RingSubfieldRootFoldCfg {
    fn root_lp() -> LevelParams {
        fp32_ring_subfield_root_lp(0)
    }
}

fn fp32_ring_subfield_setup_matrix_size<F>(
    lp: &LevelParams,
    max_num_claims: usize,
) -> Result<akita_types::SetupMatrixEnvelope, AkitaError>
where
    F: akita_field::CanonicalField,
{
    let _field_marker = core::marker::PhantomData::<F>;
    let outer_width = lp
        .outer_width()
        .checked_mul(max_num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("B matrix width overflow".to_string()))?;
    #[cfg(feature = "zk")]
    let max_zk_b_len = lp
        .b_key
        .row_len()
        .checked_mul(akita_types::zk::blinding_digit_plane_count::<F>(
            lp.b_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("ZK B setup footprint overflow".to_string()))?;

    let d_width = lp
        .d_matrix_width()
        .checked_mul(max_num_claims)
        .ok_or_else(|| AkitaError::InvalidSetup("D matrix width overflow".to_string()))?;
    #[cfg(feature = "zk")]
    let max_zk_d_len = lp
        .d_key
        .row_len()
        .checked_mul(akita_types::zk::blinding_digit_plane_count::<F>(
            lp.d_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("ZK D setup footprint overflow".to_string()))?;
    let max_setup_len =
        lp.a_key
            .row_len()
            .checked_mul(lp.inner_width())
            .ok_or_else(|| AkitaError::InvalidSetup("A setup footprint overflow".to_string()))?
            .max(lp.b_key.row_len().checked_mul(outer_width).ok_or_else(|| {
                AkitaError::InvalidSetup("B setup footprint overflow".to_string())
            })?)
            .max(lp.d_key.row_len().checked_mul(d_width).ok_or_else(|| {
                AkitaError::InvalidSetup("D setup footprint overflow".to_string())
            })?);
    Ok(akita_types::SetupMatrixEnvelope {
        max_setup_len,
        #[cfg(feature = "zk")]
        max_zk_b_len,
        #[cfg(feature = "zk")]
        max_zk_d_len,
    })
}

/// Total-claim limit shared by both fp32 ring-subfield fixtures.
///
/// Setup sizing is driven by the maximum number of claims a single batched
/// opening can carry, which is bounded by `max_num_batched_polys` (each
/// point opens at most every committed polynomial). `max_num_points` may
/// not exceed `max_num_batched_polys` for these fixtures.
fn fp32_ring_subfield_max_claims(
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<usize, AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    if max_num_points == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_points must be at least 1".to_string(),
        ));
    }
    if max_num_points > max_num_batched_polys {
        return Err(AkitaError::InvalidSetup(format!(
            "max_num_points ({max_num_points}) cannot exceed max_num_batched_polys ({max_num_batched_polys})"
        )));
    }
    Ok(max_num_batched_polys)
}

/// Single source of truth for the hand-built terminal-root schedule shared by
/// the sumcheck and direct fp32 fixtures. The `mode` argument drives the
/// terminal witness quotient (`IncludeRHat` vs `OmitRHat`), the terminal level
/// bytes, and the direct step's recorded mode, so the two fixtures differ only
/// by the mode they pass in.
fn fp32_root_terminal_schedule(
    incidence: &ClaimIncidenceSummary,
    mode: akita_types::TerminalProofMode,
) -> Result<akita_types::Schedule, AkitaError> {
    let field_bits = Fp32RingSubfieldRootFoldCfg::decomposition().field_bits();
    let lp = scale_batched_root_layout_unchecked(
        &Fp32RingSubfieldRootFoldCfg::root_lp(),
        incidence.num_claims(),
    )?;
    let quotient = mode.terminal_witness_quotient();
    let w_ring = akita_types::w_ring_element_count_with_counts_for_layout_bits_and_quotient(
        field_bits,
        &lp,
        incidence.num_points(),
        incidence.num_polynomials(),
        incidence.num_claims(),
        incidence.num_public_rows(),
        akita_types::MRowLayout::WithoutDBlock,
        quotient,
    )?;
    let compact_w_len = w_ring * Fp32RingSubfieldRootFoldCfg::D;
    let witness_shape = akita_types::CleartextWitnessShape::PackedDigits((compact_w_len, 3));
    let direct_bytes = akita_types::direct_witness_bytes(field_bits, &witness_shape);
    let challenge_field_bits = field_bits
        .checked_mul(Fp32RingSubfieldRootFoldCfg::CHAL_EXT_DEGREE as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("challenge field bits overflow".to_string()))?;
    let level_bytes = akita_types::terminal_level_proof_bytes_for_mode(
        field_bits,
        challenge_field_bits,
        &lp,
        compact_w_len,
        incidence.num_claims(),
        mode,
    );
    Ok(akita_types::Schedule {
        steps: vec![
            Step::Fold(akita_types::FoldStep {
                params: lp.clone(),
                current_w_len: akita_types::root_current_w_len(&lp),
                next_w_len: compact_w_len,
                level_bytes,
            }),
            Step::Direct(akita_types::DirectStep {
                current_w_len: compact_w_len,
                witness_shape,
                direct_bytes,
                terminal_proof_mode: mode,
                params: Some(lp.clone()),
            }),
        ],
        total_bytes: level_bytes + direct_bytes,
    })
}

impl CommitmentConfig for Fp32RingSubfieldRootFoldCfg {
    type Field = akita_field::Prime32Offset99;
    type ClaimField = akita_field::RingSubfieldFp4<Self::Field>;
    type ChallengeField = Self::ClaimField;

    const D: usize = 32;

    fn decomposition() -> akita_types::DecompositionParams {
        akita_types::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        }
    }

    fn ring_challenge_config(
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

    fn max_setup_matrix_size(
        _max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        let lp = Self::root_lp();
        let max_num_claims = fp32_ring_subfield_max_claims(max_num_batched_polys, max_num_points)?;
        fp32_ring_subfield_setup_matrix_size::<Self::Field>(&lp, max_num_claims)
    }

    fn basis_range() -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        fp32_root_terminal_schedule(incidence, Self::terminal_proof_mode())
    }
}

impl CommitmentConfig for Fp32RingSubfieldDirectRootFoldCfg {
    type Field = <Fp32RingSubfieldRootFoldCfg as CommitmentConfig>::Field;
    type ClaimField = <Fp32RingSubfieldRootFoldCfg as CommitmentConfig>::ClaimField;
    type ChallengeField = <Fp32RingSubfieldRootFoldCfg as CommitmentConfig>::ChallengeField;

    const D: usize = Fp32RingSubfieldRootFoldCfg::D;

    fn decomposition() -> akita_types::DecompositionParams {
        Fp32RingSubfieldRootFoldCfg::decomposition()
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Fp32RingSubfieldRootFoldCfg::ring_challenge_config(d)
    }

    fn sis_modulus_family() -> akita_types::SisModulusFamily {
        Fp32RingSubfieldRootFoldCfg::sis_modulus_family()
    }

    fn terminal_proof_mode() -> akita_types::TerminalProofMode {
        akita_types::TerminalProofMode::DirectRingRelations
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        Fp32RingSubfieldRootFoldCfg::max_setup_matrix_size(
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
        )
    }

    fn basis_range() -> (u32, u32) {
        Fp32RingSubfieldRootFoldCfg::basis_range()
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        fp32_root_terminal_schedule(incidence, Self::terminal_proof_mode())
    }
}

impl Fp32RingSubfieldOuterFallbackCfg {
    fn root_lp() -> LevelParams {
        fp32_ring_subfield_root_lp(1)
    }
}

impl CommitmentConfig for Fp32RingSubfieldOuterFallbackCfg {
    type Field = akita_field::Prime32Offset99;
    type ClaimField = akita_field::RingSubfieldFp4<Self::Field>;
    type ChallengeField = Self::ClaimField;

    const D: usize = 32;

    fn decomposition() -> akita_types::DecompositionParams {
        akita_types::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        }
    }

    fn ring_challenge_config(
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

    fn max_setup_matrix_size(
        _max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        let lp = Self::root_lp();
        let max_num_claims = fp32_ring_subfield_max_claims(max_num_batched_polys, max_num_points)?;
        fp32_ring_subfield_setup_matrix_size::<Self::Field>(&lp, max_num_claims)
    }

    fn basis_range() -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        incidence: &ClaimIncidenceSummary,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = scale_batched_root_layout_unchecked(&Self::root_lp(), incidence.num_claims())?;
        // Single-fold schedule: the root IS the terminal fold, so its
        // shipped `w` is built under MRowLayout::WithoutDBlock (no D-block in
        // the per-row `r` quotients). The schedule's `next_w_len` and the
        // following Direct step's witness shape must match that reduced
        // length.
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            incidence.num_points(),
            incidence.num_polynomials(),
            incidence.num_claims(),
            incidence.num_public_rows(),
            akita_types::MRowLayout::WithoutDBlock,
        )?;
        let next_w_len = w_ring * Self::D;
        Ok(akita_types::Schedule {
            steps: vec![
                Step::Fold(akita_types::FoldStep {
                    params: lp.clone(),
                    current_w_len: akita_types::root_current_w_len(&lp),
                    next_w_len,
                    level_bytes: 0,
                }),
                Step::Direct(akita_types::DirectStep {
                    current_w_len: next_w_len,
                    witness_shape: akita_types::CleartextWitnessShape::PackedDigits((
                        next_w_len, 3,
                    )),
                    direct_bytes: next_w_len,
                    terminal_proof_mode: akita_types::TerminalProofMode::RingSwitchSumcheck,
                    params: Some(lp.clone()),
                }),
            ],
            total_bytes: 0,
        })
    }
}

#[test]
fn fp32_ring_subfield_setup_sizing_uses_total_claim_limit() {
    let one_point = Fp32RingSubfieldOuterFallbackCfg::max_setup_matrix_size(5, 2, 1).unwrap();
    let two_points = Fp32RingSubfieldOuterFallbackCfg::max_setup_matrix_size(5, 2, 2).unwrap();
    assert_eq!(one_point.max_setup_len, two_points.max_setup_len);
}

#[test]
fn fp32_ring_subfield_setup_rejects_more_points_than_claims() {
    assert!(Fp32RingSubfieldRootFoldCfg::max_setup_matrix_size(5, 1, 2).is_err());
    assert!(Fp32RingSubfieldOuterFallbackCfg::max_setup_matrix_size(5, 1, 2).is_err());
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
        akita_types::SetupContributionMode::Direct,
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
        akita_types::AkitaBatchedRootProof::ZeroFold { .. } => {
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
        akita_types::SetupContributionMode::Direct,
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
        akita_types::SetupContributionMode::Direct,
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
        akita_types::SetupContributionMode::Direct,
    );
    assert!(result.is_err());
}

#[test]
fn fp32_ring_subfield_direct_terminal_root_roundtrip_checks_rows() {
    type SmallCfg = Fp32RingSubfieldDirectRootFoldCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ClaimField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 1;
    type SmallScheme = AkitaCommitmentScheme<SMALL_D, SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals_a = (0..len)
        .map(|idx| SmallF::from_u64((5 * idx as u64) + 17))
        .collect::<Vec<_>>();
    let evals_b = (0..len)
        .map(|idx| SmallF::from_u64((7 * idx as u64) + 29))
        .collect::<Vec<_>>();
    let poly_a = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals_a).unwrap();
    let poly_b = DensePoly::<SmallF, SMALL_D>::from_field_evals(NUM_VARS, &evals_b).unwrap();
    let point_a = [SmallE::new([
        SmallF::from_u64(3),
        SmallF::from_u64(5),
        SmallF::from_u64(7),
        SmallF::from_u64(11),
    ])];
    let point_b = [SmallE::new([
        SmallF::from_u64(13),
        SmallF::from_u64(17),
        SmallF::from_u64(19),
        SmallF::from_u64(23),
    ])];
    let opening_at = |evals: &[SmallF], point: &[SmallE]| {
        let weights = lagrange_weights(point).unwrap();
        evals
            .iter()
            .zip(weights.iter())
            .fold(SmallE::zero(), |acc, (&coeff, &weight)| {
                acc + weight * SmallE::lift_base(coeff)
            })
    };
    let opening_a0 = opening_at(&evals_a, &point_a);
    let opening_a1 = opening_at(&evals_b, &point_a);
    let opening_b0 = opening_at(&evals_a, &point_b);
    let opening_b1 = opening_at(&evals_b, &point_b);

    let setup =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 4, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let (commitment, hint) = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &[poly_a.clone(), poly_b.clone()],
    )
    .unwrap();

    let poly_refs = [&poly_a, &poly_b];
    let commitments = [commitment];
    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-direct-terminal-root");
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
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let terminal = match &proof.root {
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => terminal,
        other => panic!("direct terminal-root fixture produced unexpected root: {other:?}"),
    };
    assert!(matches!(
        terminal.relation,
        akita_types::TerminalRelationProof::DirectRingRelations
    ));
    assert!(terminal.stage2_sumcheck().is_none());

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    assert_eq!(bytes.len(), proof.size());
    let decoded =
        AkitaBatchedProof::<SmallF, SmallE>::deserialize_uncompressed(&*bytes, &shape).unwrap();
    assert_eq!(decoded, proof);

    let openings_a = [opening_a0, opening_a1];
    let openings_b = [opening_b0, opening_b1];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-direct-terminal-root");
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
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let wrong_openings_b = [opening_b0, opening_b1 + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-direct-terminal-root");
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
        akita_types::SetupContributionMode::Direct,
    );
    assert!(
        result.is_err(),
        "direct terminal rows must reject wrong claims"
    );

    let mut tampered_y = proof.clone();
    let akita_types::AkitaBatchedRootProof::Terminal(terminal) = &mut tampered_y.root else {
        panic!("expected terminal root proof");
    };
    let mut y_coeffs = terminal.y_rings.coeffs().to_vec();
    let first_coeff = y_coeffs.first_mut().expect("non-empty y rows");
    *first_coeff += SmallF::one();
    terminal.y_rings = FlatRingVec::from_coeffs(y_coeffs);

    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-direct-terminal-root");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &tampered_y,
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
        akita_types::SetupContributionMode::Direct,
    );
    assert!(
        result.is_err(),
        "direct terminal rows must reject tampered y rows"
    );

    let mut tampered = proof.clone();
    let akita_types::AkitaBatchedRootProof::Terminal(terminal) = &mut tampered.root else {
        panic!("expected terminal root proof");
    };
    let akita_types::CleartextWitnessProof::PackedDigits(packed) = &mut terminal.final_witness
    else {
        panic!("expected packed terminal witness");
    };
    let first_byte = packed.data.first_mut().expect("non-empty packed witness");
    *first_byte ^= 1;

    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-direct-terminal-root");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &tampered,
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
        akita_types::SetupContributionMode::Direct,
    );
    assert!(
        result.is_err(),
        "direct terminal rows must reject tampered cleartext witnesses"
    );
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
        akita_types::SetupContributionMode::Direct,
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
        akita_types::AkitaBatchedRootProof::ZeroFold { .. } => {
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
        akita_types::SetupContributionMode::Direct,
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
        akita_types::SetupContributionMode::Direct,
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
        akita_types::SetupContributionMode::Direct,
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
        akita_types::AkitaBatchedRootProof::ZeroFold { .. } => {
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
        akita_types::SetupContributionMode::Direct,
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
        akita_types::SetupContributionMode::Direct,
    );
    assert!(
        result.is_err(),
        "root tensor projection must reject a wrong claim at any point"
    );
}
