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
        scaled.b_key.collision_l2_sq(),
        d,
    );
    scaled.d_key = akita_types::AjtaiKeyParams::new_unchecked(
        scaled.d_key.sis_family(),
        scaled.d_key.row_len(),
        d_col_len,
        scaled.d_key.collision_l2_sq(),
        d,
    );
    Ok(scaled)
}

#[derive(Clone)]
struct Fp32RingSubfieldRootFoldCfg;
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
/// `0` `params_only` default. The fixture scales batched B/D widths
/// via [`scale_batched_root_layout_unchecked`] because it is synthetic.
fn fp32_ring_subfield_root_lp(m_vars: usize) -> LevelParams {
    use akita_types::AjtaiKeyParams;
    let sis_family = akita_types::SisModulusFamily::Q32;
    // Commit ring dimension must equal the static `D` the scheme dispatches
    // (`DensePoly::<SmallF, D>` / `validate_commit_level_params::<F, D>`); both
    // fixtures pin `D = 32`. `ring_subfield = 2` below is `RingSubfieldFpExt4`'s
    // embedding norm bound (a claim-field property), not `D / d`, so the
    // collision buckets are independent of this dimension.
    let d: usize = 32;
    let stage1 = akita_challenges::SparseChallengeConfig::Uniform {
        weight: 1,
        nonzero_coeffs: vec![-1, 1],
    };
    // Match the verifier-reachable derivation for this fixture's
    // `(log_basis=3, log_commit_bound=32, stage1.inf_norm=1, ring_subfield=2)`:
    // B/D use the exact derived key `32 * 7^2`; A rounds `linf=14` up to
    // coefficient bucket 15, giving `32 * 15^2`.
    let a_bucket: u128 = 7200;
    let bd_bucket: u128 = 1568;
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
/// Setup sizing is driven by the maximum number of claims a single shared
/// opening can carry, bounded by `max_num_batched_polys`.
fn fp32_ring_subfield_max_claims(max_num_batched_polys: usize) -> Result<usize, AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    Ok(max_num_batched_polys)
}

impl CommitmentConfig for Fp32RingSubfieldRootFoldCfg {
    type Field = akita_field::Prime32Offset99;
    type ExtField = akita_field::RingSubfieldFpExt4<Self::Field>;

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
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        let lp = Self::root_lp();
        let max_num_claims = fp32_ring_subfield_max_claims(max_num_batched_polys)?;
        fp32_ring_subfield_setup_matrix_size::<Self::Field>(&lp, max_num_claims)
    }

    fn basis_range() -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        opening_batch: &OpeningBatch,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = scale_batched_root_layout_unchecked(&Self::root_lp(), opening_batch.num_claims())?;
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            1,
            opening_batch.num_polynomials(),
            opening_batch.num_claims(),
            1,
            akita_types::MRowLayout::WithoutDBlock,
        )?;
        let compact_w_len = w_ring * Self::D;
        Ok(akita_types::Schedule {
            steps: vec![
                Step::Fold(akita_types::FoldStep {
                    params: lp.clone(),
                    current_w_len: akita_types::root_current_w_len(&lp),
                    next_w_len: compact_w_len,
                    level_bytes: 0,
                }),
                Step::Direct(akita_types::DirectStep {
                    current_w_len: compact_w_len,
                    witness_shape: akita_types::CleartextWitnessShape::PackedDigits((
                        compact_w_len,
                        3,
                    )),
                    direct_bytes: compact_w_len,
                    // Stub fixture: terminal-direct level params equal the
                    // fold's `lp`.
                    params: Some(lp.clone()),
                }),
            ],
            total_bytes: 0,
        })
    }
}

impl Fp32RingSubfieldOuterFallbackCfg {
    fn root_lp() -> LevelParams {
        fp32_ring_subfield_root_lp(1)
    }
}

impl CommitmentConfig for Fp32RingSubfieldOuterFallbackCfg {
    type Field = akita_field::Prime32Offset99;
    type ExtField = akita_field::RingSubfieldFpExt4<Self::Field>;

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
    ) -> Result<akita_types::SetupMatrixEnvelope, AkitaError> {
        let lp = Self::root_lp();
        let max_num_claims = fp32_ring_subfield_max_claims(max_num_batched_polys)?;
        fp32_ring_subfield_setup_matrix_size::<Self::Field>(&lp, max_num_claims)
    }

    fn basis_range() -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        opening_batch: &OpeningBatch,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = scale_batched_root_layout_unchecked(&Self::root_lp(), opening_batch.num_claims())?;
        // Single-fold schedule: the root IS the terminal fold, so its
        // shipped `w` is built under MRowLayout::WithoutDBlock (no D-block in
        // the per-row `r` quotients). The schedule's `next_w_len` and the
        // following Direct step's witness shape must match that reduced
        // length.
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            1,
            opening_batch.num_polynomials(),
            opening_batch.num_claims(),
            1,
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
                    // Stub fixture: terminal-direct level params equal the
                    // fold's `lp`.
                    params: Some(lp.clone()),
                }),
            ],
            total_bytes: 0,
        })
    }
}

#[test]
fn fp32_ring_subfield_setup_sizing_uses_total_claim_limit() {
    let one_point = Fp32RingSubfieldOuterFallbackCfg::max_setup_matrix_size(5, 2).unwrap();
    let two_points = Fp32RingSubfieldOuterFallbackCfg::max_setup_matrix_size(5, 2).unwrap();
    assert_eq!(one_point.max_setup_len, two_points.max_setup_len);
}

#[test]
fn fp32_ring_subfield_root_fold_roundtrip_uses_extension_gamma() {
    type SmallCfg = Fp32RingSubfieldRootFoldCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ExtField;
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
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 1).unwrap();
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
        (
            &point[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        ),
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
        (
            &point[..],
            vec![CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            }],
        ),
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
        (
            &point[..],
            vec![CommittedOpenings {
                openings: &wrong_openings[..],
                commitment: &commitments[0],
            }],
        ),
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
        (
            &wrong_point[..],
            vec![CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            }],
        ),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );
    assert!(result.is_err());
}

#[test]
fn fp32_ring_subfield_outer_extension_uses_root_tensor_projection() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ExtField;
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
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 2).unwrap();
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
        (
            &point[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        ),
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
        (
            &point[..],
            vec![CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            }],
        ),
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
        (
            &point[..],
            vec![CommittedOpenings {
                openings: &wrong_openings[..],
                commitment: &commitments[0],
            }],
        ),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );
    assert!(result.is_err());
}

#[test]
fn fp32_ring_subfield_extension_rejects_tampered_reduction_partial() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ExtField;
    type SmallL = <SmallCfg as CommitmentConfig>::ExtField;
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
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 2).unwrap();
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
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-eor-partial-tamper");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        (
            &point[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        ),
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let mut tampered = proof.clone();
    let reduction = match &mut tampered.root {
        akita_types::AkitaBatchedRootProof::Fold(fold) => fold.extension_opening_reduction.as_mut(),
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => {
            terminal.extension_opening_reduction.as_mut()
        }
        akita_types::AkitaBatchedRootProof::ZeroFold { .. } => {
            panic!("root-direct proof has no folded root extension-opening reduction")
        }
    }
    .expect("fixture should carry extension-opening reduction");
    *reduction
        .partials
        .first_mut()
        .expect("reduction partials must be nonempty") += SmallL::one();

    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-eor-partial-tamper");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &tampered,
        &verifier_setup,
        &mut verifier_transcript,
        (
            &point[..],
            vec![CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            }],
        ),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );
    assert!(
        matches!(result, Err(AkitaError::InvalidProof)),
        "tampered EOR partial must reject with InvalidProof, got {result:?}"
    );
}

#[test]
fn fp32_ring_subfield_batched_extension_uses_root_tensor_projection() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ExtField;
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

    let setup =
        <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_prover(NUM_VARS, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::setup_verifier(&setup);
    let polys = [poly.clone(), poly];
    let poly_refs = [&polys[0], &polys[1]];
    let (commitment, hint) = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &poly_refs,
    )
    .unwrap();
    let commitments = [commitment];
    let openings = [opening_a, opening_a];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-batched-direct");
    let proof = <SmallScheme as CommitmentProver<SmallF, SMALL_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        (
            &point_a[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        ),
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
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-batched-direct");
    <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        (
            &point_a[..],
            vec![CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            }],
        ),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let wrong_openings = [opening_a, opening_a + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-batched-direct");
    let result = <SmallScheme as CommitmentVerifier<SmallF, SMALL_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        (
            &point_a[..],
            vec![CommittedOpenings {
                openings: &wrong_openings[..],
                commitment: &commitments[0],
            }],
        ),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );
    assert!(
        result.is_err(),
        "root tensor projection must reject a wrong batched claim"
    );
}
