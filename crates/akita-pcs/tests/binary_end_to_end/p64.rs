//! The 64-bit binary root through the public grouped prove and verify entry points.
use super::*;
use akita_config::proof_optimized::fp64::Dense as Fp64Dense;
use akita_types::binary::{PrimeClaimValue, PrimeGroupClaim, PrimePlaneClaim};
use jolt_field::{CanonicalEncoding, ExtField};
use std::sync::LazyLock;

/// The fp64 dense suffix with the binary root's five-bit outgoing digits.
/// The small binary source's ranks carry no security claim.
#[derive(Clone)]
pub(super) struct BinaryTestSuffix64;

impl CommitmentConfig for BinaryTestSuffix64 {
    type Field = jolt_field::Prime64Offset59;
    type ExtField = jolt_field::Ext2<jolt_field::Prime64Offset59>;

    // The mixed root admits native degrees up to 648 and a 64-dimensional
    // ordinary B image, which the suffix dimensions must also fit.
    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        akita_schedules::RingDimensionSchedule {
            potential_a_dimensions: &[64, 128, 256, 512],
            potential_b_dimensions: &[64],
            suffix_dimensions: &[64],
            ..Fp64Dense::RING_DIMENSION_SCHEDULE
        };

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

type Verify = Box<dyn Fn(&[u8], Option<PrimeGroupClaim>) -> Result<(), AkitaError>>;

/// One honest proof and a verifier for its statement, with an optional
/// replacement prime claim.
pub(super) struct RoundTrip {
    pub(super) proof: Vec<u8>,
    pub(super) prime: Option<PrimeGroupClaim>,
    verify: Verify,
}

impl RoundTrip {
    pub(super) fn verify(&self, proof: &[u8], prime: Option<PrimeGroupClaim>) {
        (self.verify)(proof, prime).unwrap();
    }

    pub(super) fn rejects(&self, proof: &[u8], prime: Option<PrimeGroupClaim>) {
        let outcome = catch_unwind(AssertUnwindSafe(|| (self.verify)(proof, prime)));
        assert!(outcome.is_ok(), "verifier panicked");
        assert!(outcome.unwrap().is_err(), "verifier accepted");
    }
}

/// One fixture driver per configuration. The bodies differ only in their
/// field types, so a macro instantiates the same source for both.
macro_rules! round_trip {
    ($name:ident, $cfg:ty, $field:expr, $catalog:expr) => {
        /// The word claim; `mixed` adds an E-valued prime claim and an
        /// 18-variable ordinary group under the 23-variable carrier.
        pub(super) fn $name(mixed: bool) -> RoundTrip {
            type Cfg = $cfg;
            type Base = <Cfg as CommitmentConfig>::Field;
            type Ext = <Cfg as CommitmentConfig>::ExtField;
            let field: BinaryFieldProfile = $field;
            let scheme = AkitaCommitmentScheme::<Cfg>::new($catalog);
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
            let manifest = BinaryCommittedGroupManifest::with_field_profile(
                descriptor.clone(),
                0,
                &carrier,
                field,
            )
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
            let coordinates = |value: Ext| -> PrimeClaimValue {
                let c = (0..<Ext as ExtField<Base>>::DEGREE)
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
                    .map(|a| Ext::from_base_fn(|c| Base::from_u128([19 + 7 * a, 5 + 3 * a][c])))
                    .collect::<Vec<_>>();
                let eq = |point: &[Ext], index: usize| {
                    point
                        .iter()
                        .enumerate()
                        .fold(Ext::one(), |product, (axis, &x)| {
                            product
                                * if (index >> axis) & 1 == 0 {
                                    Ext::one() - x
                                } else {
                                    x
                                }
                        })
                };
                let value =
                    prime_plane::WORDS
                        .iter()
                        .enumerate()
                        .fold(Ext::zero(), |sum, (row, word)| {
                            (0..128).fold(sum, |sum, bit| {
                                if (word >> bit) & 1 == 0 {
                                    sum
                                } else {
                                    sum + eq(&point[8..], row) * eq(&point[..8], bit)
                                }
                            })
                        });
                if field == BinaryFieldProfile::P64 {
                    assert_ne!(value.base_coefficient(1), Base::zero());
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
                .map(|_| Ext::from_base_fn(|c| Base::from_u64([7, 3][c])))
                .collect::<Vec<_>>();
            // The 18-variable source is the first variable, so it reads point[0].
            let ordinary_value = ordinary_point[0];
            let ordinary = mixed.then(|| {
                let coefficients = (0..1usize << 18)
                    .map(|i| Base::from_u64((i & 1) as u64))
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
                true,
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

            let mut groups = vec![PolynomialGroupClaims::<Ext, _>::new_binary_planes(
                commitment.clone(),
                Some(word.clone()),
                prime.clone(),
            )
            .unwrap()];
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
            let mut prover = new_prover_channel(b"binary-root-field-e2e");
            prover.public_message(&BytesMsg(b"application-prefix"));
            let mut registry = akita_prover::BackendRegistry::<Cfg>::new().unwrap();
            let root = registry.register(backend.as_ref(), &prefixes).unwrap();
            let mut route =
                akita_prover::FixedFoldRoute::new(vec![
                    root;
                    resolved.schedule().num_fold_levels()
                ]);
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

            let verify = move |bytes: &[u8], prime: Option<PrimeGroupClaim>| {
                let binary = PolynomialGroupClaims::<Ext, _>::new_binary_planes(
                    commitment.clone(),
                    Some(word.clone()),
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
                let statement = selected_statement::<Cfg>(
                    OpeningClaims::from_groups(groups)?,
                    scheme.schedules(),
                );
                let verifier = scheme.verifier(verifier_setup.clone())?;
                let mut replay = new_verifier_channel(b"binary-root-field-e2e", bytes);
                replay.public_message(&BytesMsg(b"application-prefix"));
                verifier
                    .batched_verify_embedded(
                        &mut replay,
                        statement,
                        BasisMode::Lagrange,
                        SetupPolicy::Check(None),
                    )?
                    .check_eof(replay)
                    .map(|_| ())
            };
            RoundTrip {
                proof,
                prime,
                verify: Box::new(verify),
            }
        }
    };
}

round_trip!(
    p64_round_trip,
    BinaryTestSuffix64,
    BinaryFieldProfile::P64,
    p64_catalog()
);
round_trip!(
    p128_round_trip,
    BinaryTestSuffix,
    BinaryFieldProfile::P128,
    test_catalog()
);

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
        let p128 = p128_round_trip(false);
        p128.rejects(&p64.proof, None);
        p64.rejects(&p128.proof, None);
    });
}
