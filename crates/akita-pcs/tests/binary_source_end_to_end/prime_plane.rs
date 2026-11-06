//! Prime readings are computed independently from the committed bit tables.
use super::*;
use akita_challenges::{FoldDraw, ProverFoldDraw, VerifierFoldDraw};
use akita_types::binary_source::{
    binary_root_challenges, binary_root_statement, PrimeGroupClaim, PrimePlaneClaim,
    PrimePlaneManifest,
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

fn prime_point() -> Vec<u128> {
    (0..11).map(|axis| 19 + axis as u128 * 7).collect()
}

fn claim(words: &[u128], polynomial: usize) -> PrimePlaneClaim {
    let point = prime_point();
    // Only 128 coefficients are live. Coefficients 128..162 and 162..256
    // are zero. Every scalar row participates at the shared point.
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
        value: value.to_u128_checked().unwrap(),
    }
}

#[test]
fn prime_plane_prime_only() {
    init_rayon_pool();
    run_on_large_stack(|| {
        fixture(false, false, 1, 1, true);
        fixture(false, false, 2, 2, false);
        fixture(false, false, 5, 5, true);
    });
}

#[test]
fn prime_plane_word_and_several_polynomials() {
    init_rayon_pool();
    run_on_large_stack(|| {
        for planes in [1, 2, 5] {
            fixture(false, true, planes, planes, false);
        }
    });
}

#[test]
fn prime_plane_mixed_ordinary_group() {
    init_rayon_pool();
    run_on_large_stack(|| {
        for planes in [1, 2, 5] {
            fixture(true, true, planes, planes, false);
            fixture(true, false, planes, planes, false);
        }
    });
}

#[test]
fn prime_plane_several_polynomials_at_one_point() {
    init_rayon_pool();
    run_on_large_stack(|| {
        fixture(false, true, 0, 2, false);
        fixture(false, true, 2, 2, false);
        fixture(false, false, 2, 2, false);
    });
}

#[test]
fn projection_two_polynomials_and_prime_plane_rejects_y_mutations() {
    init_rayon_pool();
    run_on_large_stack(|| fixture(false, false, 1, 2, true));
}

fn fixture(mixed: bool, word: bool, planes: usize, owners: usize, mutations: bool) {
    let (layout, catalog) = if owners == 1 {
        (
            PolynomialGroupLayout::new(if mixed { 23 } else { 18 }, owners),
            test_catalog(),
        )
    } else {
        multiple_owner_catalog(owners, mixed)
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
                root_accumulated_setup_prefix_len: 0,
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
        .setup_prover(layout.num_vars().max(18), owners + usize::from(mixed))
        .unwrap();
    let backend = Arc::new(CpuBackend::new(setup.expanded.clone()).unwrap());
    let descriptor = descriptor(owners);
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier).unwrap();
    let words = (0..owners)
        .map(|owner| WORDS.iter().map(|x| x ^ owner as u128).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let source = backend
        .import_binary_source(manifest.clone(), carrier, words.clone())
        .unwrap();
    let committed = backend
        .commit(
            scheme.schedules(),
            &source,
            GroupContext::explicit(&carrier),
        )
        .unwrap();
    let commitment = committed.committed_group;
    // Width 128 stores one logical word per carrier, including every owner.
    let packed_source = backend
        .import_packed_binary_source(manifest.clone(), carrier, 8, words.clone())
        .unwrap();
    let packed = backend
        .commit(
            scheme.schedules(),
            &packed_source,
            GroupContext::explicit(&carrier),
        )
        .unwrap();
    let mut logical_bytes = Vec::new();
    let mut packed_bytes = Vec::new();
    commitment.serialize_compressed(&mut logical_bytes).unwrap();
    packed
        .committed_group
        .serialize_compressed(&mut packed_bytes)
        .unwrap();
    assert_eq!(packed_bytes, logical_bytes);
    let word_claim = word.then(|| honest_claim(&descriptor, &words[0]));
    let primes = (planes > 0).then(|| PrimeGroupClaim {
        point: prime_point(),
        evaluations: (0..planes)
            .map(|owner| claim(&words[owner], owner))
            .collect(),
    });
    let checked = PrimePlaneManifest::new(&descriptor, primes.clone()).unwrap();
    assert_eq!(checked.planes().len(), planes);
    if word && planes == 1 && owners == 1 && !mixed {
        assert!(
            PolynomialGroupClaims::<F, _>::new_binary_planes(commitment.clone(), None, None)
                .is_err()
        );
        for mutation in 0..7 {
            let mut bad = primes.clone().unwrap();
            match mutation {
                0 => bad.evaluations.clear(),
                1 => bad.evaluations.push(bad.evaluations[0].clone()),
                2 => bad.evaluations[0].polynomial = owners,
                3 => bad.evaluations[0].value = akita_params::binary_setup::BINARY_SETUP_P,
                4 => bad.point[0] = akita_params::binary_setup::BINARY_SETUP_P,
                5 => {
                    bad.point.pop();
                }
                _ => {
                    let mut extra = bad.evaluations[0].clone();
                    extra.value ^= 1;
                    bad.evaluations.push(extra);
                }
            }
            assert!(
                PolynomialGroupClaims::<F, _>::new_binary_planes(
                    commitment.clone(),
                    word_claim.clone(),
                    Some(bad)
                )
                .is_err(),
                "mutation {mutation}"
            );
        }
    }
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
    let root_key = if mixed {
        ScheduleLookupKey {
            final_group: PolynomialGroupLayout::singleton(18),
            precommitteds: vec![carrier],
            root_accumulated_setup_prefix_len: 0,
        }
    } else {
        ScheduleLookupKey::single(layout)
    };
    let resolved = scheme.schedules().resolve_key(&root_key).unwrap();
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
        .filter_map(|fold| {
            fold.params
                .setup_prefix()
                .and_then(|prefix| prefix.slot_id())
        })
        .collect::<Vec<_>>();
    let typed = BinaryTypedPrefix::new(
        &manifest,
        word,
        planes,
        resolved.schedule().recursive_folds[0].params.d_a(),
    )
    .unwrap();
    let prefixes = backend
        .prepare_binary_a_prefixes(&manifest, &typed, &first_prefix, &later_prefixes)
        .unwrap();
    let verifier_setup = backend
        .prepare_binary_a_verifier_setup(
            &scheme.setup_verifier(&setup).unwrap(),
            &manifest,
            &typed,
            &first_prefix,
            &prefixes.get(&first_prefix).unwrap().public,
        )
        .unwrap();
    let prove = |primes: Option<PrimeGroupClaim>, handle| {
        let binary = PolynomialGroupClaims::<F, _>::new_binary_planes(
            commitment.clone(),
            word_claim.clone(),
            primes,
        )
        .unwrap();
        let mut groups = vec![binary];
        let mut handles = vec![handle];
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
        let successor_coefficients = scheme
            .schedules()
            .resolve_selection(data.selection())
            .unwrap()
            .schedule()
            .recursive_folds[0]
            .params
            .outer_payload_geometry()
            .unwrap()
            .transmitted_coefficients();
        let mut prover = new_prover_channel(b"prime-plane-e2e");
        prover.public_message(&BytesMsg(b"application-prefix"));
        akita_transcript::clear_thread_events();
        let mut registry = akita_prover::BackendRegistry::<BinaryTestSuffix>::new().unwrap();
        let root = registry.register(backend.as_ref(), &prefixes).unwrap();
        let mut route =
            akita_prover::FixedFoldRoute::new(vec![root; resolved.schedule().num_fold_levels()]);
        akita_prover::batched_prove_embedded::<BinaryTestSuffix, _>(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            data,
            &mut prover,
            BasisMode::Lagrange,
            &mut route,
        )
        .unwrap();
        akita_transcript::finish_proof_ranges(&prover);
        let ranges = akita_transcript::thread_proof_ranges();
        let _ = stage3::message_offsets(
            &ranges,
            first_prefix.n_prefix().unwrap().trailing_zeros() as usize,
        );
        let bytes = prover.narg_string().to_vec();
        prover.public_message(&BytesMsg(b"application-suffix"));
        (
            bytes,
            prover.verifier_message::<[u8; 32]>(),
            successor_coefficients,
            ranges,
        )
    };
    let (proof, continuation, successor_coefficients, ranges) =
        prove(primes.clone(), committed.private_handle.clone());
    let packed_proof = prove(primes.clone(), packed.private_handle);
    assert_eq!(packed_proof.0, proof, "packed prime/projection proof drift");
    assert_eq!(packed_proof.1, continuation, "packed continuation drift");
    byte_contract::compare_baseline(
        &proof,
        &format!("binary-prime-{mixed}-{word}-{planes}-{owners}.bin"),
    );
    eprintln!(
        "prime_plane dev={} mixed={mixed} word={word} owners={owners} planes={planes}: {} bytes",
        akita_params::DEV_PROTOCOL,
        proof.len()
    );
    let verifier = scheme.verifier(verifier_setup).unwrap();
    let verify = |bytes: &[u8], commitment: &CommittedGroup<F>, primes: Option<PrimeGroupClaim>| {
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
        let mut replay = new_verifier_channel(b"prime-plane-e2e", bytes);
        replay.public_message(&BytesMsg(b"application-prefix"));
        let (pending, opening) = verifier.batched_verify_readings_embedded(
            &mut replay,
            statement.clone(),
            BasisMode::Lagrange,
            SetupPolicy::Check(None),
            [43; 32],
            [47; 32],
        )?;
        assert_eq!(opening.binary_reading(), &word_claim);
        assert_eq!(opening.prime_reading(), &primes);
        assert_eq!(opening.binding().session, [43; 32]);
        assert_eq!(opening.binding().group, [47; 32]);
        assert_eq!(opening.binding().source, descriptor);
        replay.public_message(&BytesMsg(b"application-suffix"));
        assert_eq!(replay.verifier_message::<[u8; 32]>(), continuation);
        pending.check_eof(replay)?;
        let mut old_replay = new_verifier_channel(b"prime-plane-e2e", bytes);
        old_replay.public_message(&BytesMsg(b"application-prefix"));
        let pending = verifier.batched_verify_embedded(
            &mut old_replay,
            statement,
            BasisMode::Lagrange,
            SetupPolicy::Check(None),
        )?;
        old_replay.public_message(&BytesMsg(b"application-suffix"));
        assert_eq!(old_replay.verifier_message::<[u8; 32]>(), continuation);
        pending.check_eof(old_replay).map(|_| ())
    };
    let no_a_reads = akita_types::binary_source::BinaryAReadGuard::forbid();
    verify(&proof, &commitment, primes.clone()).unwrap();
    drop(no_a_reads);
    if planes > 1 {
        let mut reordered = primes.clone().unwrap();
        reordered.evaluations.reverse();
        let reordered_proof = prove(Some(reordered), committed.private_handle.clone());
        assert_eq!(
            proof, reordered_proof.0,
            "polynomial ordering changed proof bytes"
        );
        assert_eq!(continuation, reordered_proof.1);
    }
    if !mutations {
        return;
    }
    let rejects = |label: &str,
                   bytes: &[u8],
                   commitment: &CommittedGroup<F>,
                   primes: Option<PrimeGroupClaim>| {
        let result = catch_unwind(AssertUnwindSafe(|| verify(bytes, commitment, primes)));
        assert!(result.is_ok(), "{label} caused verifier panic");
        assert!(result.unwrap().is_err(), "{label} accepted");
    };
    stage3::reject_each_message(
        &proof,
        &ranges,
        first_prefix.n_prefix().unwrap().trailing_zeros() as usize,
        |bytes| rejects("Stage 3 message", bytes, &commitment, primes.clone()),
    );
    projection::mutate_values(&proof, successor_coefficients, word, owners, |bytes| {
        rejects("projection output", bytes, &commitment, primes.clone());
    });
    let prime = primes.as_ref().unwrap();
    for index in 0..prime.evaluations.len() {
        let mut altered = prime.clone();
        altered.evaluations[index].value ^= 1;
        rejects("prime value", &proof, &commitment, Some(altered));
    }
    for coordinate in 0..prime.point.len() {
        let mut altered = prime.clone();
        altered.point[coordinate] ^= 1;
        rejects("prime coordinate", &proof, &commitment, Some(altered));
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
    assert_ne!(
        claim(&foreign_words[0], 0).value,
        prime.evaluations[0].value
    );
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
    if planes == 1 && !word && !mixed && owners == 1 {
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
            let mut prover = new_prover_channel(b"prime-plane-e2e");
            prover.public_message(&BytesMsg(b"application-prefix"));
            attacker
                .prove_prime_fault(&setup, scheme.schedules(), &prefixes, data, &mut prover)
                .unwrap();
            let malformed = prover.narg_string().to_vec();
            eprintln!(
                "completed adversarial {fault:?} prime proof: {} bytes",
                malformed.len()
            );
            assert_eq!(
                verify(&malformed, public, primes.clone()),
                Err(AkitaError::InvalidProof),
                "completed {fault:?} prime forgery must fail proof verification"
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
fn prime_plane_rejects_a_second_value_for_the_same_polynomial() {
    let descriptor = descriptor(2);
    let one = claim(&WORDS, 0);
    for value in [one.value, one.value ^ 1] {
        let primes = PrimeGroupClaim {
            point: prime_point(),
            evaluations: vec![
                one.clone(),
                PrimePlaneClaim {
                    polynomial: 0,
                    value,
                },
            ],
        };
        assert!(matches!(
            PrimePlaneManifest::new(&descriptor, Some(primes)),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    let primes = PrimeGroupClaim {
        point: prime_point(),
        evaluations: vec![one, claim(&WORDS, 1)],
    };
    assert_eq!(
        PrimePlaneManifest::new(&descriptor, Some(primes))
            .unwrap()
            .planes()
            .len(),
        2
    );
}

struct FirstSqueeze<D> {
    inner: D,
    first: Option<Vec<u8>>,
}
impl<D: FoldDraw> FoldDraw for FirstSqueeze<D> {
    fn draw_message<T>(
        &mut self,
        len: usize,
        decode: impl FnOnce(&[u8]) -> Result<Option<T>, AkitaError>,
    ) -> Result<Option<T>, AkitaError> {
        let first = &mut self.first;
        self.inner.draw_message(len, |message| {
            first.get_or_insert_with(|| message.to_vec());
            decode(message)
        })
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
    let claim = PrimeGroupClaim {
        point: prime_point(),
        evaluations: vec![claim(&WORDS, 0), claim(&WORDS, 1)],
    };
    let first = |claim: PrimeGroupClaim| {
        let planes = PrimePlaneManifest::new(&descriptor, Some(claim)).unwrap();
        let mut prover = new_prover_channel(b"prime-plane-binding");
        let mut verifier = new_verifier_channel(b"prime-plane-binding", &[]);
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
    altered.evaluations.remove(0);
    assert_ne!(
        baseline,
        first(altered.clone()),
        "polynomial list did not bind"
    );
    let one_owner = first(altered.clone());
    altered.evaluations[0].polynomial = 0;
    assert_ne!(one_owner, first(altered), "polynomial did not bind");
    for index in 0..claim.evaluations.len() {
        let mut altered = claim.clone();
        altered.evaluations[index].value ^= 1;
        assert_ne!(baseline, first(altered), "value {index} did not bind");
    }
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

/// Select a real schedule with capacity for every claimed polynomial's plane.
fn multiple_owner_catalog(
    owners: usize,
    mixed: bool,
) -> (
    PolynomialGroupLayout,
    TrustedScheduleCatalog<BinaryTestSuffix>,
) {
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let policy = akita_config::policy_of::<BinaryTestSuffix>(chunking);
    let contract = BinaryTestSuffix::committed_source_contract().unwrap();
    // Reuse the admitted ordinary fixture whose native degrees fit this root.
    let ordinary = mixed.then(|| {
        let catalog = test_catalog();
        let carrier = catalog
            .resolve_key(&ScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(23),
            ))
            .unwrap()
            .profiles()
            .final_group;
        let row = catalog
            .resolve_key(&ScheduleLookupKey {
                final_group: PolynomialGroupLayout::singleton(18),
                precommitteds: vec![carrier],
                root_accumulated_setup_prefix_len: 0,
            })
            .unwrap();
        (
            row.profiles().final_group,
            row.schedule().root.params.clone(),
        )
    });
    for vars in 18..=26 {
        let layout = PolynomialGroupLayout::new(vars, owners);
        let producer = akita_planner::find_schedule_with_root_setup_mode(
            &ScheduleLookupKey::single(layout),
            contract,
            &[],
            &policy,
            BinaryTestSuffix::ring_challenge_config,
            akita_params::SetupContributionMode::Recursive,
            None,
        )
        .unwrap()
        .schedule;
        let producer_profile = producer.root.params.own_group().profile;
        let key = if let Some((ordinary, _)) = &ordinary {
            ScheduleLookupKey {
                final_group: ordinary.group,
                precommitteds: vec![producer_profile],
                root_accumulated_setup_prefix_len: 0,
            }
        } else {
            ScheduleLookupKey::single(layout)
        };
        let root = if let Some((_, ordinary)) = &ordinary {
            akita_planner::find_schedule_with_root_setup_mode(
                &key,
                contract,
                &[contract],
                &policy,
                BinaryTestSuffix::ring_challenge_config,
                akita_params::SetupContributionMode::Recursive,
                Some(ordinary),
            )
            .unwrap()
            .schedule
        } else {
            producer.clone()
        };
        let first = &root.recursive_folds[0].params;
        let Some(prefix) = first.setup_prefix() else {
            continue;
        };
        let manifest =
            BinaryCommittedGroupManifest::new(descriptor(owners), 0, &producer_profile).unwrap();
        let typed = BinaryTypedPrefix::new(&manifest, true, owners, first.d_a()).unwrap();
        let plan = akita_types::binary_source::BinaryRootWitnessPlan::new(
            manifest.descriptor(),
            manifest.a_rank(),
            first.d_a(),
        )
        .unwrap()
        .with_prime_planes(true, owners)
        .unwrap();
        let plan = if let Some((ordinary, _)) = &ordinary {
            let ordinary = akita_types::binary_source::OrdinaryNativeRootPlan::new(
                *ordinary,
                plan.repacking().segments().len(),
            )
            .unwrap();
            plan.with_ordinary(&ordinary).unwrap()
        } else {
            plan
        };
        if prefix.n_prefix().unwrap() < typed.domain_len
            || root.root.output_witness_len < plan.repacking().natural_end().unwrap()
            || akita_params::witness_commitment_domain_len(
                root.root.output_witness_len,
                first.d_a(),
            )
            .unwrap()
                < plan.repacking().domain_len()
        {
            continue;
        }
        let producer_profiles = akita_params::CommittedGroupBatchProfile {
            final_group: producer_profile,
            precommitteds: Vec::new(),
        };
        let mut rows = vec![(producer_profiles, producer)];
        if ordinary.is_some() {
            rows.push((
                akita_params::CommittedGroupBatchProfile {
                    final_group: root.root.params.own_group().profile,
                    precommitteds: key.precommitteds,
                },
                root,
            ));
        }
        let catalog = akita_schedules::ValidatedScheduleCatalog::try_new(
            BinaryTestSuffix::schedule_family_name(),
            rows,
            &policy,
            BinaryTestSuffix::ring_challenge_config,
        )
        .unwrap();
        return (
            layout,
            TrustedScheduleCatalog::new(catalog, chunking).unwrap(),
        );
    }
    panic!("no fixture schedule holds the multi-polynomial binary root");
}

#[test]
fn prime_plane_p64_base_claims_and_extension_typed_prefix_endpoint() {
    use akita_algebra::eq_poly::EqPolynomial;
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_types::binary_source::{
        binary_word_manifest_fixture, BinaryAProduct, BinaryAReadGuard, BinaryRootWitnessPlan,
    };
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as Base};
    type E = Ext2<Base>;
    let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
    for word in [false, true] {
        let plan = BinaryRootWitnessPlan::for_profile(
            manifest.descriptor(),
            3,
            64,
            BinaryFieldProfile::P64,
            word,
            1,
        )
        .unwrap();
        let typed = BinaryTypedPrefix::from_plan(&manifest, &plan).unwrap();
        let extension =
            |i: usize| E::from_base_fn(|c| Base::from_u64((17 + i * 31 + c * 43) as u64));
        let stage2 = (0..plan.repacking().domain_len().trailing_zeros())
            .map(|i| extension(i as usize))
            .collect::<Vec<_>>();
        let rho = (0..3)
            .map(|r| std::array::from_fn(|k| extension(r * 648 + k)))
            .collect::<Vec<_>>();
        let a = BinaryAProduct::new(&plan, &rho, &stage2, typed.domain_len).unwrap();
        let batches = [
            vec![extension(3000); manifest.b_rows()],
            vec![extension(4000); manifest.d_label(word, 1).unwrap().rank as usize],
        ];
        let refs = [&batches[0][..], &batches[1][..]];
        let weights = typed
            .prover_weights(&a, &stage2, extension(5000), refs, typed.domain_len)
            .unwrap();
        let setup = typed
            .setup_coefficients::<Base>(&[41; 32], &manifest, typed.domain_len)
            .unwrap();
        let point = (0..typed.domain_len.trailing_zeros())
            .map(|i| extension(6000 + i as usize))
            .collect::<Vec<_>>();
        let eq = EqPolynomial::evals(&point).unwrap();
        let value = setup
            .iter()
            .zip(&eq)
            .fold(E::zero(), |sum, (&v, &eq)| sum + E::lift_base(v) * eq);
        assert_ne!(value.base_coefficient(1), Base::zero());
        let expected = weights
            .iter()
            .zip(eq)
            .fold(E::zero(), |sum, (&v, eq)| sum + v * eq);
        let guard = BinaryAReadGuard::forbid();
        let actual = a.evaluate(&point).unwrap()
            + typed
                .digit_weight(&point, &stage2, extension(5000), refs)
                .unwrap();
        assert_eq!(actual, expected);
        drop(guard);
    }
    let base_claim = PrimeGroupClaim {
        point: vec![2; 11],
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: 3,
        }],
    };
    let prime = PrimePlaneManifest::with_field_profile(
        manifest.descriptor(),
        Some(base_claim),
        BinaryFieldProfile::P64,
    )
    .unwrap();
    let mut channel = new_prover_channel(b"p64-prime-public-admission");
    binary_root_statement::<Base, E, _>(
        &mut channel,
        &manifest,
        None,
        &prime,
        &vec![Base::zero(); manifest.payload_coefficients().unwrap()],
        None,
    )
    .unwrap();
}
