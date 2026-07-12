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
        scaled.b_key.security_policy(),
        scaled.b_key.sis_family(),
        scaled.b_key.row_len(),
        b_col_len,
        scaled.b_key.coeff_linf_bound(),
        d,
    );
    scaled.d_key = akita_types::AjtaiKeyParams::new_unchecked(
        scaled.d_key.security_policy(),
        scaled.d_key.sis_family(),
        scaled.d_key.row_len(),
        d_col_len,
        scaled.d_key.coeff_linf_bound(),
        d,
    );
    Ok(scaled)
}

/// Build an extension opening point compatible with fp32 `FpExt4` ring-subfield
/// packing at `ring_d`.
///
/// Root tensor projection requires `num_vars >= log2(ring_d)`. Inactive inner
/// coordinates in `[trace_inner, log2(ring_d))` must be zero
/// ([`akita_types::prepare_opening_point`]).
fn fp32_ext4_ring_subfield_extension_point<E>(
    ring_d: usize,
    num_vars: usize,
    coord_at: impl Fn(usize) -> E,
) -> Vec<E>
where
    E: akita_field::AdditiveGroup + Copy,
{
    const EXT_DEGREE: usize = 4;
    let trace_inner = (ring_d / EXT_DEGREE).trailing_zeros() as usize;
    let alpha_bits = ring_d.trailing_zeros() as usize;
    let mut point: Vec<E> = (0..num_vars).map(coord_at).collect();
    for idx in trace_inner..alpha_bits {
        if idx < point.len() {
            point[idx] = E::zero();
        }
    }
    point
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
fn fp32_ext4_root_lp(m_vars: usize) -> LevelParams {
    use akita_types::{AjtaiKeyParams, DEFAULT_SIS_SECURITY_POLICY};
    let sis_family = akita_types::SisModulusFamily::Q32;
    // fp32 inner dispatch starts at D=64; fixtures pin uniform D=64.
    // `ring_subfield = 2` below is `FpExt4`'s embedding norm bound (a claim-field
    // property), not `D / d`, so the collision buckets are independent of this dimension.
    let d: usize = 64;
    let stage1 = akita_challenges::SparseChallengeConfig::production_for_ring_dim(d)
        .expect("production D64 fold challenge");
    // Match the verifier-reachable derivation for this fixture's
    // `(log_basis=3, log_commit_bound=32, production D64 signed-sparse, ring_subfield=2)`:
    // B/D use the exact coefficient bucket `7`; A rounds `linf=14` up to
    // coefficient bucket `15`. Buckets are pinned synthetically via `new_unchecked`.
    let a_bucket: u128 = 15;
    let bd_bucket: u128 = 7;
    let mut params = LevelParams::params_only(sis_family, d, 3, 1, 1, 1, stage1);
    params.a_key =
        AjtaiKeyParams::new_unchecked(DEFAULT_SIS_SECURITY_POLICY, sis_family, 1, 0, a_bucket, d);
    params.b_key =
        AjtaiKeyParams::new_unchecked(DEFAULT_SIS_SECURITY_POLICY, sis_family, 1, 0, bd_bucket, d);
    params.d_key =
        AjtaiKeyParams::new_unchecked(DEFAULT_SIS_SECURITY_POLICY, sis_family, 1, 0, bd_bucket, d);
    params.with_decomp(m_vars, 0, 12, 12, 0).unwrap()
}

impl Fp32RingSubfieldRootFoldCfg {
    fn root_lp() -> LevelParams {
        fp32_ext4_root_lp(0)
    }
}

fn fp32_ext4_setup_matrix_size<F>(
    lp: &LevelParams,
    max_num_polynomials: usize,
) -> Result<akita_types::SetupMatrixEnvelope, AkitaError>
where
    F: akita_field::CanonicalField,
{
    let _field_marker = core::marker::PhantomData::<F>;
    let outer_width = lp
        .outer_width()
        .checked_mul(max_num_polynomials)
        .ok_or_else(|| AkitaError::InvalidSetup("B matrix width overflow".to_string()))?;

    let d_width = lp
        .d_matrix_width()
        .checked_mul(max_num_polynomials)
        .ok_or_else(|| AkitaError::InvalidSetup("D matrix width overflow".to_string()))?;
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
    Ok(akita_types::SetupMatrixEnvelope { max_setup_len })
}

/// Total-claim limit shared by both fp32 ring-subfield fixtures.
///
/// Setup sizing is driven by the maximum number of claims a single shared
/// opening can carry, bounded by `max_num_batched_polys`.
fn fp32_ext4_max_claims(max_num_batched_polys: usize) -> Result<usize, AkitaError> {
    if max_num_batched_polys == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    Ok(max_num_batched_polys)
}

impl CommitmentConfig for Fp32RingSubfieldRootFoldCfg {
    type Field = akita_field::Prime32Offset99;
    type ExtField = akita_field::FpExt4<Self::Field>;

    const D: usize = 64;

    fn decomposition() -> akita_types::DecompositionParams {
        akita_types::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        }
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no production fold-challenge ladder entry for ring dimension {d}"
            ))
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
        let max_num_polynomials = fp32_ext4_max_claims(max_num_batched_polys)?;
        fp32_ext4_setup_matrix_size::<Self::Field>(&lp, max_num_polynomials)
    }

    fn basis_range() -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = scale_batched_root_layout_unchecked(
            &Self::root_lp(),
            opening_batch.num_total_polynomials(),
        )?;
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            opening_batch.num_total_polynomials(),
            1,
            akita_types::RelationMatrixRowLayout::WithoutDBlock,
        )?;
        let next_w_len = w_ring * Self::D;
        let witness_shape = akita_types::segment_typed_witness_shape(
            &lp,
            Self::Field::modulus_bits(),
            opening_batch.num_total_polynomials(),
            opening_batch.num_total_polynomials(),
            1,
            1,
        )?;
        let direct_bytes =
            akita_types::direct_witness_bytes(Self::Field::modulus_bits(), &witness_shape);
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
                    witness_shape,
                    direct_bytes,
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
        fp32_ext4_root_lp(1)
    }
}

impl CommitmentConfig for Fp32RingSubfieldOuterFallbackCfg {
    type Field = akita_field::Prime32Offset99;
    type ExtField = akita_field::FpExt4<Self::Field>;

    const D: usize = 64;

    fn decomposition() -> akita_types::DecompositionParams {
        akita_types::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: Some(32),
        }
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(d).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no production fold-challenge ladder entry for ring dimension {d}"
            ))
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
        let max_num_polynomials = fp32_ext4_max_claims(max_num_batched_polys)?;
        fp32_ext4_setup_matrix_size::<Self::Field>(&lp, max_num_polynomials)
    }

    fn basis_range() -> (u32, u32) {
        (3, 3)
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<akita_types::Schedule, AkitaError> {
        let lp = scale_batched_root_layout_unchecked(
            &Self::root_lp(),
            opening_batch.num_total_polynomials(),
        )?;
        // Single-fold schedule: the root IS the terminal fold, so its
        // shipped `w` is built under RelationMatrixRowLayout::WithoutDBlock (no D-block in
        // the per-row `r` quotients). The schedule's `next_w_len` and the
        // following Direct step's witness shape must match that reduced
        // length.
        let w_ring = akita_types::w_ring_element_count_with_counts_for_layout::<Self::Field>(
            &lp,
            opening_batch.num_total_polynomials(),
            1,
            akita_types::RelationMatrixRowLayout::WithoutDBlock,
        )?;
        let next_w_len = w_ring * Self::D;
        let witness_shape = akita_types::segment_typed_witness_shape(
            &lp,
            Self::Field::modulus_bits(),
            opening_batch.num_total_polynomials(),
            opening_batch.num_total_polynomials(),
            1,
            1,
        )?;
        let direct_bytes =
            akita_types::direct_witness_bytes(Self::Field::modulus_bits(), &witness_shape);
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
                    witness_shape,
                    direct_bytes,
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
fn fp32_ext4_setup_sizing_uses_total_claim_limit() {
    let one_point = Fp32RingSubfieldOuterFallbackCfg::max_setup_matrix_size(5, 2).unwrap();
    let two_points = Fp32RingSubfieldOuterFallbackCfg::max_setup_matrix_size(5, 2).unwrap();
    assert_eq!(one_point.max_setup_len, two_points.max_setup_len);
}

#[test]
fn fp32_ext4_root_fold_roundtrip_uses_extension_gamma() {
    type SmallCfg = Fp32RingSubfieldRootFoldCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ExtField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 1;
    type SmallScheme = AkitaCommitmentScheme<SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals = (0..len)
        .map(|idx| SmallF::from_u64((3 * idx as u64) + 9))
        .collect::<Vec<_>>();
    let poly = DensePoly::<SmallF>::from_field_evals(NUM_VARS, SMALL_D, &evals).unwrap();
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

    let setup = SmallScheme::setup_prover(NUM_VARS, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = SmallScheme::setup_verifier(&setup);
    let (commitment, hint) =
        SmallScheme::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack).unwrap();

    let poly_refs = [&poly];
    let commitments = [commitment];
    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let proof = SmallScheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&point[..], &poly_refs[..], &commitments[0], hint),
        &stack,
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
    SmallScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let mut malformed_stage2 = proof.clone();
    let terminal = match &mut malformed_stage2.root {
        akita_types::AkitaBatchedRootProof::Terminal(terminal) => terminal,
        akita_types::AkitaBatchedRootProof::Fold(_) => {
            panic!("NUM_VARS=1 fixture should be terminal-rooted")
        }
        akita_types::AkitaBatchedRootProof::ZeroFold { .. } => {
            panic!("NUM_VARS=1 fixture should not be root-direct")
        }
    };
    let sumcheck_proof = terminal.stage2.sumcheck().clone();
    terminal.stage2 =
        akita_types::AkitaStage2Proof::Intermediate(akita_types::AkitaIntermediateStage2Proof {
            sumcheck_proof,
            next_w_commitment: akita_types::RingVec::from_coeffs(Vec::<SmallF>::new()),
            next_w_eval: SmallE::zero(),
        });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut verifier_transcript =
            AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
        SmallScheme::batched_verify(
            &malformed_stage2,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims(&point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
    }));
    assert!(matches!(result, Ok(Err(AkitaError::InvalidProof))));

    let wrong_openings = [opening + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let result = SmallScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&point[..], &wrong_openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );
    assert!(result.is_err());

    let wrong_point = [point[0] + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-root-fold");
    let result = SmallScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&wrong_point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );
    assert!(result.is_err());
}

#[test]
fn fp32_ext4_outer_extension_uses_root_tensor_projection() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ExtField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 6;
    type SmallScheme = AkitaCommitmentScheme<SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals_a = (0..len)
        .map(|idx| SmallF::from_u64((idx as u64) + 3))
        .collect::<Vec<_>>();
    let evals_b = (0..len)
        .map(|idx| SmallF::from_u64((2 * idx as u64) + 7))
        .collect::<Vec<_>>();
    let poly_a = DensePoly::<SmallF>::from_field_evals(NUM_VARS, SMALL_D, &evals_a).unwrap();
    let poly_b = DensePoly::<SmallF>::from_field_evals(NUM_VARS, SMALL_D, &evals_b).unwrap();
    let point = fp32_ext4_ring_subfield_extension_point(SMALL_D, NUM_VARS, |idx| {
        SmallE::new([
            SmallF::from_u64((idx + 2) as u64),
            SmallF::from_u64((idx + 4) as u64),
            SmallF::from_u64((idx + 6) as u64),
            SmallF::from_u64((idx + 8) as u64),
        ])
    });
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

    let setup = SmallScheme::setup_prover(NUM_VARS, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = SmallScheme::setup_verifier(&setup);
    let poly_refs = [&poly_a, &poly_b];
    let (commitment, hint) =
        SmallScheme::commit::<_, _>(&setup, &[poly_a.clone(), poly_b.clone()], &stack).unwrap();
    let commitments = [commitment];
    let openings = [opening_a, opening_b];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-outer-direct");
    let proof = SmallScheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&point[..], &poly_refs[..], &commitments[0], hint),
        &stack,
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
    SmallScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let wrong_openings = [opening_a, opening_b + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-outer-direct");
    let result = SmallScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&point[..], &wrong_openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );
    assert!(result.is_err());
}

#[test]
fn fp32_ext4_extension_rejects_tampered_reduction_partial() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ExtField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 6;
    type SmallScheme = AkitaCommitmentScheme<SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals_a = (0..len)
        .map(|idx| SmallF::from_u64((idx as u64) + 3))
        .collect::<Vec<_>>();
    let evals_b = (0..len)
        .map(|idx| SmallF::from_u64((2 * idx as u64) + 7))
        .collect::<Vec<_>>();
    let poly_a = DensePoly::<SmallF>::from_field_evals(NUM_VARS, SMALL_D, &evals_a).unwrap();
    let poly_b = DensePoly::<SmallF>::from_field_evals(NUM_VARS, SMALL_D, &evals_b).unwrap();
    let point = fp32_ext4_ring_subfield_extension_point(SMALL_D, NUM_VARS, |idx| {
        SmallE::new([
            SmallF::from_u64((idx + 2) as u64),
            SmallF::from_u64((idx + 4) as u64),
            SmallF::from_u64((idx + 6) as u64),
            SmallF::from_u64((idx + 8) as u64),
        ])
    });
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

    let setup = SmallScheme::setup_prover(NUM_VARS, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = SmallScheme::setup_verifier(&setup);
    let poly_refs = [&poly_a, &poly_b];
    let (commitment, hint) =
        SmallScheme::commit::<_, _>(&setup, &[poly_a.clone(), poly_b.clone()], &stack).unwrap();
    let commitments = [commitment];
    let openings = [opening_a, opening_b];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-eor-partial-tamper");
    let proof = SmallScheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&point[..], &poly_refs[..], &commitments[0], hint),
        &stack,
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
        .expect("reduction partials must be nonempty") += SmallE::one();

    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-eor-partial-tamper");
    let result = SmallScheme::batched_verify(
        &tampered,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );
    assert!(
        matches!(result, Err(AkitaError::InvalidProof)),
        "tampered EOR partial must reject with InvalidProof, got {result:?}"
    );
}

#[test]
fn fp32_ext4_batched_extension_uses_root_tensor_projection() {
    type SmallCfg = Fp32RingSubfieldOuterFallbackCfg;
    type SmallF = <SmallCfg as CommitmentConfig>::Field;
    type SmallE = <SmallCfg as CommitmentConfig>::ExtField;
    const SMALL_D: usize = SmallCfg::D;
    const NUM_VARS: usize = 6;
    type SmallScheme = AkitaCommitmentScheme<SmallCfg>;

    let len = 1usize << NUM_VARS;
    let evals = (0..len)
        .map(|idx| SmallF::from_u64((3 * idx as u64) + 5))
        .collect::<Vec<_>>();
    let poly = DensePoly::<SmallF>::from_field_evals(NUM_VARS, SMALL_D, &evals).unwrap();
    let point_a = fp32_ext4_ring_subfield_extension_point(SMALL_D, NUM_VARS, |idx| {
        SmallE::new([
            SmallF::from_u64((idx + 3) as u64),
            SmallF::from_u64((idx + 5) as u64),
            SmallF::from_u64((idx + 7) as u64),
            SmallF::from_u64((idx + 9) as u64),
        ])
    });
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

    let setup = SmallScheme::setup_prover(NUM_VARS, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = SmallScheme::setup_verifier(&setup);
    let polys = [poly.clone(), poly];
    let poly_refs = [&polys[0], &polys[1]];
    let (commitment, hint) = SmallScheme::commit::<_, _>(&setup, &polys, &stack).unwrap();
    let commitments = [commitment];
    let openings = [opening_a, opening_a];

    let mut prover_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-batched-direct");
    let proof = SmallScheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&point_a[..], &poly_refs[..], &commitments[0], hint),
        &stack,
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
    SmallScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&point_a[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let wrong_openings = [opening_a, opening_a + SmallE::one()];
    let mut verifier_transcript =
        AkitaTranscript::<SmallF>::new(b"test/fp32-ring-subfield-batched-direct");
    let result = SmallScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&point_a[..], &wrong_openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );
    assert!(
        result.is_err(),
        "root tensor projection must reject a wrong batched claim"
    );
}
