//! Prime readings are computed independently from the committed bit tables.
use super::*;
use akita_challenges::{FoldDraw, ProverFoldDraw, VerifierFoldDraw};
use akita_types::binary_source::{
    binary_root_challenges, binary_root_statement, PrimePlaneClaim, PrimePlaneManifest,
};
use jolt_field::CanonicalEncoding;

const WORDS: [u128; 8] = [
    0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
    0x8000_0000_0000_0000_0000_0000_0000_0001,
    0xaaaa_5555_aaaa_5555_aaaa_5555_aaaa_5555,
    0x3333_cccc_3333_cccc_3333_cccc_3333_cccc,
    0x1212_3434_5656_7878_9090_abab_cdcd_efef,
    0xffff_0000_ffff_0000_ffff_0000_ffff_0000,
    0x0101_0202_0303_0404_0505_0606_0707_0808,
    0x1357_9bdf_2468_ace0_1357_9bdf_2468_ace0,
];

fn descriptor(owners: usize) -> BinarySourceDescriptor {
    let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
    BinarySourceDescriptor::new(
        index.clone(),
        (0..owners).map(|i| [29 + i as u8; 32]).collect(),
        BinaryCommittedGroupManifest::commitment_profiles(&index, owners, 0).unwrap(),
    )
    .unwrap()
}

fn equality(point: &[u128], index: usize) -> F {
    point
        .iter()
        .enumerate()
        .fold(F::one(), |product, (axis, &x)| {
            let x = F::from_u128(x);
            product
                * if (index >> axis) & 1 == 0 {
                    F::one() - x
                } else {
                    x
                }
        })
}

fn claim(words: &[u128], polynomial: usize, component: u8, point_id: usize) -> PrimePlaneClaim {
    let point = (0..11)
        .map(|axis| 19 + axis as u128 * 7 + point_id as u128 * 13)
        .collect::<Vec<_>>();
    // Only 128 coefficients are live. Coefficients 128..162 and 162..256
    // are zero. Every scalar row participates; the component is an identity tag.
    let value = words.iter().enumerate().fold(F::zero(), |sum, (i, word)| {
        let carrier_weight = equality(&point[8..], i);
        (0..128).fold(sum, |sum, bit| {
            if (word >> bit) & 1 == 0 {
                sum
            } else {
                sum + carrier_weight * equality(&point[..8], bit)
            }
        })
    });
    PrimePlaneClaim {
        polynomial,
        point,
        tensor_component: component,
        value: value.to_u128_checked().unwrap(),
    }
}

#[test]
fn prime_plane_prime_only() {
    init_rayon_pool();
    run_on_large_stack(|| {
        fixture(false, false, 1, 1, true);
        fixture(false, false, 5, 1, true);
    });
}

#[test]
fn prime_plane_word_and_one_two_five_planes() {
    init_rayon_pool();
    run_on_large_stack(|| {
        for planes in [1, 2, 5] {
            fixture(false, true, planes, 1, false);
        }
    });
}

#[test]
fn prime_plane_mixed_ordinary_group() {
    init_rayon_pool();
    run_on_large_stack(|| {
        for planes in [1, 2, 5] {
            fixture(true, true, planes, 1, false);
        }
    });
}

#[test]
fn prime_plane_several_polynomials_do_not_share() {
    init_rayon_pool();
    run_on_large_stack(|| {
        fixture(false, true, 2, 2, false);
        fixture(false, false, 2, 2, false);
    });
}

fn fixture(mixed: bool, word: bool, planes: usize, owners: usize, mutations: bool) {
    let layout = PolynomialGroupLayout::new(if mixed { 23 } else { 18 }, owners);
    let catalog = if owners == 1 {
        test_catalog()
    } else {
        multiple_owner_catalog(layout)
    };
    let scheme = AkitaCommitmentScheme::<BinaryTestSuffix>::new(catalog);
    let carrier = scheme
        .schedules()
        .resolve_key(&ScheduleLookupKey::single(layout))
        .unwrap()
        .profiles()
        .final_group;
    let profiles = if mixed {
        scheme
            .schedules()
            .resolve_key(&ScheduleLookupKey {
                final_group: PolynomialGroupLayout::singleton(18),
                precommitteds: vec![carrier],
            })
            .unwrap()
            .profiles()
            .clone()
    } else {
        scheme
            .schedules()
            .resolve_key(&ScheduleLookupKey::single(layout))
            .unwrap()
            .profiles()
            .clone()
    };
    let setup = scheme
        .setup_prover(if mixed { 23 } else { 18 }, owners + usize::from(mixed))
        .unwrap();
    let backend = Arc::new(CpuBackend::new(setup.expanded.clone()).unwrap());
    let descriptor = descriptor(owners);
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier).unwrap();
    let words = (0..owners)
        .map(|owner| WORDS.iter().map(|x| x ^ owner as u128).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let source = backend
        .import_binary_source(manifest, carrier, words.clone())
        .unwrap();
    let committed = backend
        .commit(
            scheme.schedules(),
            &source,
            GroupContext::explicit(&carrier),
        )
        .unwrap();
    let commitment = committed.committed_group;
    let word_claim = word.then(|| honest_claim(&descriptor, &words[0]));
    let primes = (0..planes)
        .map(|plane| {
            let owner = plane % owners;
            claim(&words[owner], owner, (plane % 4) as u8, plane / 4)
        })
        .collect::<Vec<_>>();
    let checked = PrimePlaneManifest::new(&descriptor, &primes).unwrap();
    assert_eq!(checked.planes().len(), planes);
    let (ordinary, ordinary_handle, ordinary_point, ordinary_value) = if mixed {
        let coefficients = (0..1usize << 18)
            .map(|i| F::from_u64((i & 1) as u64))
            .collect::<Vec<_>>();
        let point = vec![F::from_u64(7); 18];
        let value = point[0];
        let source = backend
            .import_source(vec![DensePoly::from_field_evals(18, &coefficients).unwrap()])
            .unwrap();
        let ordinary = backend
            .commit(
                scheme.schedules(),
                &source,
                GroupContext::explicit(&profiles.final_group),
            )
            .unwrap();
        (
            Some(ordinary.committed_group),
            Some(ordinary.private_handle),
            point,
            value,
        )
    } else {
        (None, None, Vec::new(), F::zero())
    };
    let prove = |primes: Vec<PrimePlaneClaim>| {
        let binary = PolynomialGroupClaims::<F, _>::new_binary_planes(
            commitment.clone(),
            word_claim.clone(),
            primes,
        )
        .unwrap();
        let mut groups = vec![binary];
        let mut handles = vec![committed.private_handle.clone()];
        if let Some(ordinary) = &ordinary {
            groups.push(
                PolynomialGroupClaims::new(
                    ordinary_point.clone(),
                    vec![ordinary_value],
                    ordinary.clone(),
                )
                .unwrap(),
            );
            handles.push(ordinary_handle.clone().unwrap());
        }
        let data = selected_prover_data::<BinaryTestSuffix>(
            OpeningClaims::from_groups(groups).unwrap(),
            handles,
            scheme.schedules(),
        );
        let mut prover = new_prover_channel(b"prime-plane-e2e", b"caller").unwrap();
        prover.public_message(b"application-prefix");
        scheme
            .batched_prove_in_session(
                &setup,
                data,
                backend.as_ref(),
                &mut prover,
                BasisMode::Lagrange,
            )
            .unwrap();
        let bytes = prover.narg_string().to_vec();
        prover.public_message(b"application-suffix");
        (bytes, prover.verifier_message::<[u8; 32]>())
    };
    let (proof, continuation) = prove(primes.clone());
    eprintln!(
        "prime_plane mixed={mixed} word={word} owners={owners} planes={planes}: {} bytes",
        proof.len()
    );
    let verifier = scheme
        .verifier(scheme.setup_verifier(&setup).unwrap())
        .unwrap();
    let verify = |bytes: &[u8], commitment: &CommittedGroup<F>, primes: Vec<PrimePlaneClaim>| {
        let binary = PolynomialGroupClaims::<F, _>::new_binary_planes(
            commitment.clone(),
            word_claim.clone(),
            primes.clone(),
        )?
        .map_commitment(|_| commitment);
        let mut groups = vec![binary];
        if let Some(ordinary) = &ordinary {
            groups.push(PolynomialGroupClaims::new(
                ordinary_point.clone(),
                vec![ordinary_value],
                ordinary,
            )?);
        }
        let statement = selected_statement::<BinaryTestSuffix>(
            OpeningClaims::from_groups(groups)?,
            scheme.schedules(),
        );
        let mut replay = new_verifier_channel(b"prime-plane-e2e", b"caller", bytes)?;
        replay.public_message(b"application-prefix");
        let opening = verifier.batched_verify_readings_in_session(
            &mut replay,
            statement.clone(),
            BasisMode::Lagrange,
            [43; 32],
            [47; 32],
        )?;
        assert_eq!(opening.binary_reading(), &word_claim);
        assert_eq!(opening.prime_reading(), &primes);
        assert_eq!(opening.binding().session, [43; 32]);
        assert_eq!(opening.binding().group, [47; 32]);
        assert_eq!(opening.binding().source, descriptor);
        replay.public_message(b"application-suffix");
        assert_eq!(replay.verifier_message::<[u8; 32]>()?, continuation);
        replay.check_eof()?;
        let mut old_replay = new_verifier_channel(b"prime-plane-e2e", b"caller", bytes)?;
        old_replay.public_message(b"application-prefix");
        verifier.batched_verify_in_session(&mut old_replay, statement, BasisMode::Lagrange)?;
        old_replay.public_message(b"application-suffix");
        assert_eq!(old_replay.verifier_message::<[u8; 32]>()?, continuation);
        old_replay.check_eof()
    };
    verify(&proof, &commitment, primes.clone()).unwrap();
    if planes == 1 {
        let mut duplicated = primes.clone();
        duplicated.push(primes[0].clone());
        let duplicate = prove(duplicated);
        assert_eq!(
            proof, duplicate.0,
            "identical requests must share one plane and canonical statement"
        );
        assert_eq!(continuation, duplicate.1);
    }
    if !mutations {
        return;
    }
    let rejects = |label: &str,
                   bytes: &[u8],
                   commitment: &CommittedGroup<F>,
                   primes: Vec<PrimePlaneClaim>| {
        let result = catch_unwind(AssertUnwindSafe(|| verify(bytes, commitment, primes)));
        assert!(result.is_ok(), "{label} caused verifier panic");
        assert!(result.unwrap().is_err(), "{label} accepted");
    };
    for index in 0..primes.len() {
        let mut altered = primes.clone();
        altered[index].value ^= 1;
        rejects("prime value", &proof, &commitment, altered);
        for coordinate in 0..primes[index].point.len() {
            let mut altered = primes.clone();
            altered[index].point[coordinate] ^= 1;
            rejects("prime coordinate", &proof, &commitment, altered);
        }
        let mut altered = primes.clone();
        altered[index].tensor_component = (altered[index].tensor_component + 1) % 4;
        rejects("tensor component", &proof, &commitment, altered);
    }
    // The prime-only D payload is the first prover message. Alter one field
    // coefficient of its EP commitment without disturbing its framing.
    let mut altered = proof.clone();
    altered[0] ^= 1;
    rejects(
        "EP D payload coefficient",
        &altered,
        &commitment,
        primes.clone(),
    );
    let mut foreign_words = words.clone();
    foreign_words[0][0] ^= 1;
    assert_ne!(claim(&foreign_words[0], 0, 0, 0).value, primes[0].value);
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier).unwrap();
    let source = backend
        .import_binary_source(manifest, carrier, foreign_words)
        .unwrap();
    let foreign = backend
        .commit(
            scheme.schedules(),
            &source,
            GroupContext::explicit(&carrier),
        )
        .unwrap();
    if planes == 1 && !word && !mixed {
        for (fault, public) in [
            (Fault::PrimePlane, &commitment),
            (Fault::ForeignCommitment, &foreign.committed_group),
        ] {
            let attacker = FaultBackend::new(
                backend.clone(),
                setup.expanded.descriptor().setup_seed.seed,
                words[0].clone(),
                fault,
            );
            let claims =
                OpeningClaims::from_groups(vec![PolynomialGroupClaims::<F, _>::new_binary_planes(
                    public.clone(),
                    None,
                    primes.clone(),
                )
                .unwrap()])
                .unwrap();
            let data = selected_prover_data::<BinaryTestSuffix>(
                claims,
                vec![committed.private_handle.clone()],
                scheme.schedules(),
            );
            let mut prover = new_prover_channel(b"prime-plane-e2e", b"caller").unwrap();
            prover.public_message(b"application-prefix");
            attacker
                .prove_prime_fault(&setup, scheme.schedules(), data, &mut prover)
                .unwrap();
            let malformed = prover.narg_string().to_vec();
            eprintln!(
                "completed adversarial {fault:?} prime proof: {} bytes",
                malformed.len()
            );
            rejects(
                "completed false prime witness",
                &malformed,
                public,
                primes.clone(),
            );
        }
    }
    rejects(
        "different prime source",
        &proof,
        &foreign.committed_group,
        primes,
    );
}

#[test]
fn prime_plane_caps_and_owner_sharing() {
    let descriptor = descriptor(2);
    let one = claim(&WORDS, 0, 0, 0);
    let duplicate = PrimePlaneManifest::new(&descriptor, &[one.clone(), one.clone()]).unwrap();
    assert_eq!(duplicate.assignments(), &[0, 0]);
    let mut other_owner = one.clone();
    other_owner.polynomial = 1;
    assert_eq!(
        PrimePlaneManifest::new(&descriptor, &[one, other_owner])
            .unwrap()
            .planes()
            .len(),
        2
    );
    let distinct = (0..6).map(|i| claim(&WORDS, 0, 0, i)).collect::<Vec<_>>();
    assert!(matches!(
        PrimePlaneManifest::new(&descriptor, &distinct),
        Err(AkitaError::InvalidInput(_))
    ));
    let components = (0..6)
        .map(|i| claim(&WORDS, i / 4, (i % 4) as u8, 0))
        .collect::<Vec<_>>();
    assert!(matches!(
        PrimePlaneManifest::new(&descriptor, &components),
        Err(AkitaError::InvalidInput(_))
    ));
}

struct FirstSqueeze<D> {
    inner: D,
    first: Option<[u8; akita_transcript::FOLD_CHALLENGE_SEED_LEN]>,
}
impl<D: FoldDraw> FoldDraw for FirstSqueeze<D> {
    fn absorb_and_squeeze(
        &mut self,
        payload: &[u8],
    ) -> Result<[u8; akita_transcript::FOLD_CHALLENGE_SEED_LEN], AkitaError> {
        let seed = self.inner.absorb_and_squeeze(payload)?;
        self.first.get_or_insert(seed);
        Ok(seed)
    }
}

#[test]
fn prime_plane_every_claim_field_binds_first_challenge_on_both_sides() {
    let descriptor = descriptor(2);
    let carrier = test_catalog()
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(18),
        ))
        .unwrap()
        .profiles()
        .final_group;
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier).unwrap();
    let claim = claim(&WORDS, 0, 0, 0);
    let first = |claim: PrimePlaneClaim| {
        let planes = PrimePlaneManifest::new(&descriptor, &[claim]).unwrap();
        let mut prover = new_prover_channel(b"prime-plane-binding", b"caller").unwrap();
        let mut verifier = new_verifier_channel(b"prime-plane-binding", b"caller", &[]).unwrap();
        binary_root_statement::<F, F, _>(
            &mut prover,
            &manifest,
            None,
            &planes,
            &[F::one(); 128],
            None,
        )
        .unwrap();
        binary_root_statement::<F, F, _>(
            &mut verifier,
            &manifest,
            None,
            &planes,
            &[F::one(); 128],
            None,
        )
        .unwrap();
        let mut sent = FirstSqueeze {
            inner: ProverFoldDraw::new(&mut prover, 0, 0),
            first: None,
        };
        let mut received = FirstSqueeze {
            inner: VerifierFoldDraw::new(&mut verifier, 0, 0),
            first: None,
        };
        assert_eq!(
            binary_root_challenges(&mut sent, 2, 2).unwrap(),
            binary_root_challenges(&mut received, 2, 2).unwrap()
        );
        assert_eq!(sent.first, received.first);
        sent.first.unwrap()
    };
    let baseline = first(claim.clone());
    let mut altered = claim.clone();
    altered.polynomial = 1;
    assert_ne!(baseline, first(altered), "polynomial did not bind");
    let mut altered = claim.clone();
    altered.tensor_component = 1;
    assert_ne!(baseline, first(altered), "component did not bind");
    let mut altered = claim.clone();
    altered.value ^= 1;
    assert_ne!(baseline, first(altered), "value did not bind");
    for coordinate in 0..claim.point.len() {
        let mut altered = claim.clone();
        altered.point[coordinate] ^= 1;
        assert_ne!(
            baseline,
            first(altered),
            "coordinate {coordinate} did not bind"
        );
    }
}

fn multiple_owner_catalog(
    layout: PolynomialGroupLayout,
) -> TrustedScheduleCatalog<BinaryTestSuffix> {
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let policy = akita_config::policy_of::<BinaryTestSuffix>(chunking);
    let schedule = akita_planner::find_schedule(
        &ScheduleLookupKey::single(layout),
        BinaryTestSuffix::committed_source_contract().unwrap(),
        &[],
        &policy,
        BinaryTestSuffix::ring_challenge_config,
    )
    .unwrap()
    .schedule;
    let profiles = akita_params::CommittedGroupBatchProfile {
        final_group: akita_params::GroupCommitPhaseParams::try_from_params(
            layout,
            &schedule.root.params,
        )
        .unwrap(),
        precommitteds: Vec::new(),
    };
    let catalog = akita_schedules::ValidatedScheduleCatalog::try_new(
        BinaryTestSuffix::schedule_family_name(),
        [(profiles, schedule)],
        &policy,
        BinaryTestSuffix::ring_challenge_config,
    )
    .unwrap();
    TrustedScheduleCatalog::new(catalog, chunking).unwrap()
}
