//! The 64-bit binary root through the public grouped prove and verify entry points.
use super::*;
use akita_config::proof_optimized::fp64::Dense as Fp64Dense;
use akita_prover::backend::SetupBatchBackend;
use akita_serialization::Valid;
use akita_types::binary::{PrimeClaimValue, PrimeGroupClaim, PrimePlaneClaim};
use akita_types::setup_claims::{
    SetupBatchMemory, SetupBatchStatement, SetupClaim, SetupPolynomialId, SetupQuery,
};
use akita_types::FpExtEncoding;
use akita_verifier::SetupClaimAccumulator;
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, PseudoMersenne,
    Unreduced, WithCommitAccumulator,
};
use std::sync::LazyLock;

/// The fp64 dense suffix with the binary root's five-bit outgoing digits.
/// The small binary source's ranks carry no security claim.
#[derive(Clone)]
pub(super) struct BinaryTestSuffix64;

impl CommitmentConfig for BinaryTestSuffix64 {
    type Field = jolt_field::Prime64Offset59;
    type ExtField = jolt_field::Ext2<jolt_field::Prime64Offset59>;

    // Narrowed to the dimensions a mixed root admits.
    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        akita_config::proof_optimized::fp64::BINARY_MIXED_ROOT_RING_DIMENSION_SCHEDULE;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: Fp64Dense::decomposition().log_basis,
            // The outgoing root witness holds base-eight digits in [-4, 3].
            log_commit_bound: 5,
            // Openings carry full base-field elements below p < 2^64.
            log_open_bound: Some(64),
        }
    }

    fn ring_challenge_config(dimension: usize) -> Result<FoldChallengeDistribution, AkitaError> {
        Fp64Dense::ring_challenge_config(dimension)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        SisModulusProfileId::Q64Offset59
    }

    fn opening_basis_range() -> (u32, u32) {
        Fp64Dense::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Fp64Dense::inner_basis_range()
    }

    fn committed_source_class() -> CommittedSourceClass {
        Fp64Dense::committed_source_class()
    }

    fn recursive_setup_planning() -> bool {
        true
    }

    fn schedule_family_name() -> &'static str {
        "binary_source_p64_fixture"
    }
}

fn p64_catalog() -> TrustedScheduleCatalog<BinaryTestSuffix64> {
    static CATALOG: LazyLock<TrustedScheduleCatalog<BinaryTestSuffix64>> =
        LazyLock::new(stage3_catalog::plan::<BinaryTestSuffix64>);
    CATALOG.clone()
}

type Verify = Box<dyn Fn(&[u8], Option<PrimeGroupClaim>, bool) -> Result<(), AkitaError>>;

/// One honest proof and a verifier for its statement, with an optional
/// replacement prime claim.
pub(super) struct RoundTrip {
    pub(super) proof: Vec<u8>,
    pub(super) prime: Option<PrimeGroupClaim>,
    verify: Verify,
}

impl RoundTrip {
    pub(super) fn verify(&self, proof: &[u8], prime: Option<PrimeGroupClaim>) {
        (self.verify)(proof, prime, false).unwrap();
    }

    /// The verifier's error on `proof`, which it must reject without panicking.
    pub(super) fn rejects(&self, proof: &[u8], prime: Option<PrimeGroupClaim>) -> AkitaError {
        let outcome = catch_unwind(AssertUnwindSafe(|| (self.verify)(proof, prime, false)));
        outcome
            .expect("verifier panicked")
            .expect_err("verifier accepted")
    }
}

/// Base-field bounds of the CPU backend, the grouped prover and the verifier.
pub(super) trait RootBase:
    Field
    + CanonicalEncoding
    + Unreduced<Wide: From<Self> + AdditiveGroup>
    + PseudoMersenne
    + Ring
    + WithCommitAccumulator
    + Valid
    + AkitaSerialize
    + AkitaDeserialize<Context = ()>
    + 'static
{
}

impl<T> RootBase for T where
    T: Field
        + CanonicalEncoding
        + Unreduced<Wide: From<T> + AdditiveGroup>
        + PseudoMersenne
        + Ring
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static
{
}

/// Challenge-field bounds of the CPU backend, the grouped prover and the verifier.
pub(super) trait RootExt<F: Field>:
    ExtField<F>
    + FpExtEncoding<F>
    + MulBaseUnreduced<F>
    + Unreduced
    + Fold
    + Ring
    + Valid
    + AkitaSerialize
    + AkitaDeserialize<Context = ()>
    + 'static
{
}

impl<F: Field, T> RootExt<F> for T where
    T: ExtField<F>
        + FpExtEncoding<F>
        + MulBaseUnreduced<F>
        + Unreduced
        + Fold
        + Ring
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static
{
}

/// An ordinary suffix configuration that a binary root opens through.
pub(super) trait RootConfig:
    CommitmentConfig<Field: RootBase, ExtField: RootExt<<Self as CommitmentConfig>::Field>>
    + Clone
    + 'static
{
}

impl<C> RootConfig for C where
    C: CommitmentConfig<Field: RootBase, ExtField: RootExt<<C as CommitmentConfig>::Field>>
        + Clone
        + 'static
{
}

/// The word claim; `mixed` adds an E-valued prime claim and an
/// 18-variable ordinary group under the 23-variable carrier.
pub(super) fn round_trip<Cfg: RootConfig>(
    field: BinaryFieldProfile,
    catalog: TrustedScheduleCatalog<Cfg>,
    mixed: bool,
    setup_batch_level: Option<u32>,
) -> RoundTrip {
    let scheme = AkitaCommitmentScheme::<Cfg>::new(catalog);
    let layout = PolynomialGroupLayout::singleton(if mixed { 23 } else { 18 });
    let carrier = scheme
        .schedules()
        .resolve_key(&ScheduleLookupKey::single(layout))
        .unwrap()
        .profiles()
        .final_group;
    let key = if mixed {
        ScheduleLookupKey {
            final_group: PolynomialGroupLayout::singleton(18),
            precommitteds: vec![carrier],
            root_accumulated_setup_prefix_len: 0,
        }
    } else {
        ScheduleLookupKey::single(layout)
    };
    let resolved = scheme.schedules().resolve_key(&key).unwrap().clone();
    let setup = scheme
        .setup_prover(layout.num_vars(), 1 + usize::from(mixed))
        .unwrap();
    let backend = Arc::new(CpuBackend::new(setup.expanded.clone()).unwrap());
    let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
    let descriptor = BinarySourceDescriptor::new(
        index.clone(),
        vec![[29; 32]],
        BinaryCommittedGroupManifest::profiles_for_field(&index, 1, 0, field).unwrap(),
    )
    .unwrap();
    let manifest =
        BinaryCommittedGroupManifest::with_field_profile(descriptor.clone(), 0, &carrier, field)
            .unwrap();
    let source = backend
        .import_binary_source(manifest.clone(), carrier, vec![prime_plane::WORDS.to_vec()])
        .unwrap();
    let committed = backend
        .commit(
            scheme.schedules(),
            &source,
            GroupContext::explicit(&carrier),
        )
        .unwrap();
    let commitment = committed.committed_group;
    let word = honest_claim(&descriptor, &prime_plane::WORDS);

    // The prime reading at an E point whose coordinates all have a
    // nonzero u part over P64, from the committed words directly.
    let coordinates = |value: Cfg::ExtField| -> PrimeClaimValue {
        let c = (0..<Cfg::ExtField as ExtField<Cfg::Field>>::DEGREE)
            .map(|i| value.base_coefficient(i).to_u128_checked().unwrap())
            .collect::<Vec<_>>();
        match *c.as_slice() {
            [base] => base.into(),
            [c0, c1] => [c0, c1].into(),
            _ => unreachable!("binary roots use degree one or two"),
        }
    };
    let prime = mixed.then(|| {
        let point = (0..11u128)
            .map(|a| {
                Cfg::ExtField::from_base_fn(|c| Cfg::Field::from_u128([19 + 7 * a, 5 + 3 * a][c]))
            })
            .collect::<Vec<_>>();
        let eq = |point: &[Cfg::ExtField], index: usize| {
            point
                .iter()
                .enumerate()
                .fold(Cfg::ExtField::one(), |product, (axis, &x)| {
                    product
                        * if (index >> axis) & 1 == 0 {
                            Cfg::ExtField::one() - x
                        } else {
                            x
                        }
                })
        };
        let value = prime_plane::WORDS.iter().enumerate().fold(
            Cfg::ExtField::zero(),
            |sum, (row, word)| {
                (0..128).fold(sum, |sum, bit| {
                    if (word >> bit) & 1 == 0 {
                        sum
                    } else {
                        sum + eq(&point[8..], row) * eq(&point[..8], bit)
                    }
                })
            },
        );
        if field == BinaryFieldProfile::P64 {
            assert_ne!(value.base_coefficient(1), Cfg::Field::zero());
        }
        PrimeGroupClaim {
            point: point.into_iter().map(coordinates).collect(),
            evaluations: vec![PrimePlaneClaim {
                polynomial: 0,
                value: coordinates(value),
            }],
        }
    });

    let ordinary_point = (0..18)
        .map(|_| Cfg::ExtField::from_base_fn(|c| Cfg::Field::from_u64([7, 3][c])))
        .collect::<Vec<_>>();
    // The 18-variable source is the first variable, so it reads point[0].
    let ordinary_value = ordinary_point[0];
    let ordinary = mixed.then(|| {
        let coefficients = (0..1usize << 18)
            .map(|i| Cfg::Field::from_u64((i & 1) as u64))
            .collect::<Vec<_>>();
        let source = backend
            .import_source(vec![DensePoly::from_field_evals(18, &coefficients).unwrap()])
            .unwrap();
        backend
            .commit(
                scheme.schedules(),
                &source,
                GroupContext::explicit(&resolved.profiles().final_group),
            )
            .unwrap()
    });

    let first_prefix = resolved.schedule().recursive_folds[0]
        .params
        .setup_prefix()
        .unwrap()
        .slot_id()
        .unwrap();
    let later_prefixes = resolved
        .schedule()
        .recursive_folds
        .iter()
        .skip(1)
        .filter_map(|fold| fold.params.setup_prefix().and_then(|p| p.slot_id()))
        .collect::<Vec<_>>();
    let typed = BinaryTypedPrefix::new(
        &manifest,
        usize::from(mixed) * field.extension_degree(),
        resolved.schedule().recursive_folds[0].params.d_a(),
    )
    .unwrap();
    let prefixes = backend
        .prepare_binary_typed_prefixes(&manifest, &typed, &first_prefix, &later_prefixes)
        .unwrap();
    let verifier_setup = backend
        .prepare_binary_typed_verifier_setup(
            &scheme.setup_verifier(&setup).unwrap(),
            &manifest,
            &typed,
            &first_prefix,
            &prefixes.get(&first_prefix).unwrap().public,
        )
        .unwrap();

    let mut groups = vec![
        PolynomialGroupClaims::<Cfg::ExtField, _>::new_binary_planes(
            commitment.clone(),
            word.clone(),
            prime.clone(),
        )
        .unwrap(),
    ];
    let mut handles = vec![committed.private_handle];
    let ordinary = ordinary.map(|ordinary| {
        groups.push(
            PolynomialGroupClaims::new(
                ordinary_point.clone(),
                vec![ordinary_value],
                ordinary.committed_group.clone(),
            )
            .unwrap(),
        );
        handles.push(ordinary.private_handle);
        ordinary.committed_group
    });
    let data = selected_prover_data::<Cfg>(
        OpeningClaims::from_groups(groups).unwrap(),
        handles,
        scheme.schedules(),
    );
    let inherited = setup_batch_level.map(|level| {
        let domain = resolved
            .schedule()
            .recursive_folds
            .get(level as usize)
            .and_then(|fold| fold.params.setup_prefix())
            .map_or(32, |prefix| prefix.n_prefix().unwrap());
        let polynomial =
            SetupPolynomialId::new::<Cfg::Field, Cfg::ExtField>(&setup.expanded, domain).unwrap();
        let query = SetupQuery::evaluation(vec![Cfg::ExtField::zero(); polynomial.num_variables()])
            .unwrap();
        let value = if level == 0 {
            // The binary root has no consumer for this ordinary setup claim.
            Cfg::ExtField::zero()
        } else {
            backend
                .prepare_setup_query(&polynomial, &query, 32, SetupBatchMemory::Regenerate)
                .unwrap()
                .0
        };
        let claim = SetupClaim::new(polynomial.clone(), query, value).unwrap();
        SetupBatchStatement::new(polynomial, 32, vec![claim]).unwrap()
    });
    let data = match (setup_batch_level, inherited.as_ref()) {
        (Some(level), Some(batch)) => data
            .with_setup_batch(level, batch, scheme.schedules())
            .unwrap(),
        _ => data,
    };
    let mut prover = new_prover_channel(b"binary-root-field-e2e");
    prover.public_message(&BytesMsg(b"application-prefix"));
    let mut registry = akita_prover::BackendRegistry::<Cfg>::new().unwrap();
    let root = registry.register(backend.as_ref(), &prefixes).unwrap();
    let mut route =
        akita_prover::FixedFoldRoute::new(vec![root; resolved.schedule().num_fold_levels()]);
    akita_prover::batched_prove_embedded::<Cfg, _>(
        setup.expanded.descriptor(),
        scheme.schedules(),
        &registry,
        data,
        &mut prover,
        BasisMode::Lagrange,
        &mut route,
    )
    .unwrap();
    let proof = prover.narg_string().to_vec();
    eprintln!("binary root {field:?} mixed={mixed}: {} bytes", proof.len());

    let verify = move |bytes: &[u8], prime: Option<PrimeGroupClaim>, defer: bool| {
        let binary = PolynomialGroupClaims::<Cfg::ExtField, _>::new_binary_planes(
            commitment.clone(),
            word.clone(),
            prime,
        )?
        .map_commitment(|_| &commitment);
        let mut groups = vec![binary];
        if let Some(ordinary) = &ordinary {
            groups.push(PolynomialGroupClaims::new(
                ordinary_point.clone(),
                vec![ordinary_value],
                ordinary,
            )?);
        }
        let statement =
            selected_statement::<Cfg>(OpeningClaims::from_groups(groups)?, scheme.schedules());
        let verifier = scheme.verifier(verifier_setup.clone())?;
        let mut replay = new_verifier_channel(b"binary-root-field-e2e", bytes);
        replay.public_message(&BytesMsg(b"application-prefix"));
        let mut accumulator = setup_batch_level
            .zip(inherited.clone())
            .map(|(level, batch)| SetupClaimAccumulator::from_batch(level, batch));
        let policy = if defer {
            SetupPolicy::Defer(accumulator.as_mut().unwrap())
        } else {
            SetupPolicy::Check(accumulator.as_mut())
        };
        let result =
            verifier.batched_verify_embedded(&mut replay, statement, BasisMode::Lagrange, policy);
        if let Some(accumulator) = accumulator {
            assert_eq!(
                accumulator.into_pending().len(),
                usize::from(result.is_err())
            );
        }
        result
            .and_then(|pending| pending.check_eof(replay))
            .map(|_| ())
    };
    RoundTrip {
        proof,
        prime,
        verify: Box::new(verify),
    }
}

pub(super) fn p64_round_trip(mixed: bool) -> RoundTrip {
    round_trip(BinaryFieldProfile::P64, p64_catalog(), mixed, None)
}

#[test]
fn p64_binary_claim_round_trip() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opened = p64_round_trip(false);
        opened.verify(&opened.proof, None);
    });
}

#[test]
fn p64_binary_prime_and_ordinary_round_trip() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opened = p64_round_trip(true);
        opened.verify(&opened.proof, opened.prime.clone());
    });
}

#[test]
fn proofs_do_not_verify_under_the_other_field() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let p64 = p64_round_trip(false);
        let p128 = round_trip(BinaryFieldProfile::P128, test_catalog(), false, None);
        p64.verify(&p64.proof, None);
        p128.verify(&p128.proof, None);
        for error in [
            p128.rejects(&p64.proof, None),
            p64.rejects(&p128.proof, None),
        ] {
            assert!(matches!(error, AkitaError::InvalidProof), "{error:?}");
        }
    });
}

#[test]
fn binary_root_rejects_inherited_setup_batch_at_level_zero() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opened = round_trip(BinaryFieldProfile::P128, test_catalog(), false, Some(0));
        for defer in [false, true] {
            let result = (opened.verify)(&opened.proof, None, defer);
            assert!(
                matches!(
                    &result,
                    Err(AkitaError::InvalidInput(message))
                        if message == "setup batch target cannot be a binary root"
                ),
                "{result:?}"
            );
        }
    });
}

#[test]
fn binary_root_accepts_inherited_setup_batch_at_level_one() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opened = round_trip(BinaryFieldProfile::P128, test_catalog(), false, Some(1));
        opened.verify(&opened.proof, None);
    });
}
