//! Caller terminal claims, dense widths, padding and polynomial ownership.
use super::*;
use akita_types::binary_source::{BinaryOpening, PrimePlaneClaim, PrimePlaneManifest};

#[path = "grouped_surface_agreement.rs"]
mod agreement;
#[path = "grouped_surface_catalog.rs"]
mod catalog;
#[path = "grouped_surface_claims.rs"]
mod claims;
#[path = "grouped_surface_ordinary.rs"]
mod ordinary;

#[test]
fn grouped_surface_every_dense_width_word_and_prime_claims() {
    init_rayon_pool();
    run_on_large_stack(|| {
        for width in [1, 2, 4, 8, 16, 32, 64, 128] {
            for word in [false, true] {
                fixture(width, 8 * (128 / width), 8, 1, word, usize::from(!word));
            }
        }
    });
}

#[test]
fn grouped_surface_small_odd_and_partial_carriers() {
    init_rayon_pool();
    run_on_large_stack(|| {
        // Log zero, odd logs, fewer than one carrier, non-power-of-two live
        // carrier counts, and a partial final ring element in column order.
        for (width, length, carriers) in [
            (1, 1, 8),
            (8, 8, 8),
            (32, 13, 8),
            (64, 9, 8),
            (128, 5, 8),
            (128, 9, 16),
        ] {
            fixture(width, length, carriers, 1, true, 1);
        }
    });
}

#[test]
fn grouped_surface_each_polynomial_owns_its_reduced_terminals() {
    init_rayon_pool();
    run_on_large_stack(|| {
        for (word, planes) in [(true, 0), (false, 5), (true, 5)] {
            fixture(32, 23, 8, 2, word, planes);
        }
    });
}

fn fixture(width: usize, length: usize, carriers: usize, owners: usize, word: bool, planes: usize) {
    let index = BinaryIndexMap::new(width, length, carriers, 2).unwrap();
    let words = (0..owners)
        .map(|owner| claims::words(&index, owner))
        .collect::<Vec<_>>();
    let planes = (0..planes)
        .map(|plane| (plane % owners, plane, (plane % 4) as u8))
        .collect::<Vec<_>>();
    open(&index, words, word, &planes, true);
}

/// Prove and verify one opening of `words`, one source per owner, returning the
/// verifier's authenticated readings. Each plane is (owner, point seed,
/// tensor component). With `mutate`, every claim and context change rejects.
pub(super) fn open(
    index: &BinaryIndexMap,
    words: Vec<Vec<u128>>,
    word: bool,
    planes: &[(usize, usize, u8)],
    mutate: bool,
) -> BinaryOpening<Vec<BinaryWordClaim>, Vec<PrimePlaneClaim>> {
    let owners = words.len();
    let (width, length, carriers) = (
        index.word_width(),
        index.logical_len(),
        index.padded_carriers(),
    );
    let scheme = AkitaCommitmentScheme::<BinaryTestSuffix>::new(catalog::catalog(owners));
    let resolved = scheme
        .schedules()
        .resolve_key(&ScheduleLookupKey::single(PolynomialGroupLayout::new(
            18, owners,
        )))
        .unwrap();
    let carrier = resolved.profiles().final_group;
    let setup = scheme.setup_prover(18, owners).unwrap();
    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
    let descriptor = BinarySourceDescriptor::new(
        index.clone(),
        (0..owners).map(|owner| [29 + owner as u8; 32]).collect(),
        BinaryCommittedGroupManifest::commitment_profiles(index, owners, 0).unwrap(),
    )
    .unwrap();
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier).unwrap();
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
    let terminals = if word {
        words
            .iter()
            .enumerate()
            .map(|(owner, words)| claims::word_claim(&descriptor, owner, words))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let primes = planes
        .iter()
        .map(|&(owner, seed, component)| {
            claims::prime_claim(&descriptor, owner, &words[owner], seed, component)
        })
        .collect::<Vec<_>>();
    let prime_manifest = PrimePlaneManifest::new(&descriptor, &primes).unwrap();
    assert_eq!(prime_manifest.planes().len(), planes.len());
    if mutate && planes.len() == 5 {
        let mut excessive = primes.clone();
        excessive.push(claims::prime_claim(&descriptor, 0, &words[0], 5, 1));
        assert!(matches!(
            PrimePlaneManifest::new(&descriptor, &excessive),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    if let (true, Some(first)) = (mutate, terminals.first()) {
        let mut duplicate = terminals.clone();
        duplicate.push(first.clone());
        assert!(matches!(
            PolynomialGroupClaims::<F, _>::new_binary_planes(
                commitment.clone(),
                duplicate,
                primes.clone()
            ),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    let first_prefix = resolved.schedule().recursive_folds[0]
        .params
        .setup_prefix()
        .unwrap()
        .slot_id()
        .unwrap();
    let later = resolved
        .schedule()
        .recursive_folds
        .iter()
        .skip(1)
        .filter_map(|fold| fold.params.setup_prefix().and_then(|p| p.slot_id()))
        .collect::<Vec<_>>();
    let prefixes = backend
        .prepare_binary_a_prefixes(&manifest, &first_prefix, &later)
        .unwrap();
    let verifier_setup = backend
        .prepare_binary_a_verifier_setup(
            &scheme.setup_verifier(&setup).unwrap(),
            &manifest,
            &first_prefix,
            &prefixes.get(&first_prefix).unwrap().public,
        )
        .unwrap();
    let binary = PolynomialGroupClaims::<F, _>::new_binary_planes(
        commitment.clone(),
        terminals.clone(),
        primes.clone(),
    )
    .unwrap();
    let data = selected_prover_data::<BinaryTestSuffix>(
        OpeningClaims::from_groups(vec![binary]).unwrap(),
        vec![committed.private_handle],
        scheme.schedules(),
    );
    let mut prover = new_prover_channel(b"grouped-surface", b"caller").unwrap();
    prover.public_message(&[43u8; 32]);
    prover.public_message(&[47u8; 32]);
    // The caller runs its stages and supplies exactly their reduced terminal
    // claims. These messages belong to the same proof stream as the opening.
    let stage_message = [11u8; 32];
    prover.prover_message(&stage_message);
    let stage_challenge: [u8; 32] = prover.verifier_message();
    let mut registry = akita_prover::BackendRegistry::<BinaryTestSuffix>::new().unwrap();
    let root = registry.register(&backend, &prefixes).unwrap();
    let mut route =
        akita_prover::FixedFoldRoute::new(vec![root; resolved.schedule().num_fold_levels()]);
    akita_prover::batched_prove_in_session::<BinaryTestSuffix, _>(
        setup.expanded.descriptor(),
        scheme.schedules(),
        &registry,
        data,
        &mut prover,
        BasisMode::Lagrange,
        &mut route,
    )
    .unwrap();
    let opening_bytes = prover.narg_string().len() - 32;
    prover.public_message(b"caller-after-opening");
    let continuation: [u8; 32] = prover.verifier_message();
    prover.prover_message(&continuation);
    let proof = prover.narg_string().to_vec();
    eprintln!("grouped_surface dev={} width={width} length={length} carriers={carriers} owners={owners} word={word} planes={}: {opening_bytes} bytes",
        akita_params::DEV_PROTOCOL, planes.len());
    let verifier = scheme.verifier(verifier_setup).unwrap();
    let verify =
        |bytes: &[u8],
         words: Vec<BinaryWordClaim>,
         primes: Vec<PrimePlaneClaim>,
         session: [u8; 32],
         group: [u8; 32]|
         -> Result<BinaryOpening<Vec<BinaryWordClaim>, Vec<PrimePlaneClaim>>, AkitaError> {
            let binary = PolynomialGroupClaims::<F, _>::new_binary_planes(
                commitment.clone(),
                words.clone(),
                primes.clone(),
            )?
            .map_commitment(|_| &commitment);
            let statement = selected_statement::<BinaryTestSuffix>(
                OpeningClaims::from_groups(vec![binary])?,
                scheme.schedules(),
            );
            let mut replay = new_verifier_channel(b"grouped-surface", b"caller", bytes)?;
            replay.public_message(&session);
            replay.public_message(&group);
            assert_eq!(replay.prover_message::<[u8; 32]>()?, stage_message);
            let challenge = replay.verifier_message::<[u8; 32]>()?;
            let reading = verifier.batched_verify_readings_in_session(
                &mut replay,
                statement,
                BasisMode::Lagrange,
                session,
                group,
            )?;
            assert_eq!(challenge, stage_challenge);
            assert_eq!(reading.binary_reading(), &words);
            assert_eq!(reading.prime_reading(), &primes);
            assert_eq!(&reading.binding().source, &descriptor);
            agreement::check(&reading);
            // The opening leaves the caller's final message unread.
            replay.public_message(b"caller-after-opening");
            assert_eq!(replay.verifier_message::<[u8; 32]>()?, continuation);
            assert_eq!(replay.prover_message::<[u8; 32]>()?, continuation);
            replay.check_eof()?;
            Ok(reading)
        };
    let reading = verify(
        &proof,
        terminals.clone(),
        primes.clone(),
        [43; 32],
        [47; 32],
    )
    .unwrap();
    if !mutate {
        return reading;
    }
    let rejects = |words: Vec<BinaryWordClaim>, primes: Vec<PrimePlaneClaim>, session, group| {
        let result = catch_unwind(AssertUnwindSafe(|| {
            verify(&proof, words, primes, session, group)
        }));
        assert!(result.is_ok(), "mutated terminal caused a panic");
        assert!(result.unwrap().is_err(), "mutated terminal accepted");
    };
    for (owner, terminal) in terminals.iter().enumerate() {
        let mut value = *terminal.value();
        value[0] ^= 1;
        let mut changed = terminals.clone();
        changed[owner] = BinaryWordClaim::new(
            &descriptor,
            *terminal.polynomial(),
            terminal.point().to_vec(),
            value,
        )
        .unwrap();
        rejects(changed, primes.clone(), [43; 32], [47; 32]);
    }
    for plane in 0..primes.len() {
        let mut changed = primes.clone();
        changed[plane].value ^= 1;
        rejects(terminals.clone(), changed, [43; 32], [47; 32]);
    }
    if owners > 1 {
        if terminals.len() > 1 {
            let swapped = terminals
                .iter()
                .enumerate()
                .map(|(i, claim)| {
                    BinaryWordClaim::new(
                        &descriptor,
                        descriptor.polynomials()[1 - i],
                        claim.point().to_vec(),
                        *claim.value(),
                    )
                    .unwrap()
                })
                .collect();
            rejects(swapped, primes.clone(), [43; 32], [47; 32]);
        }
        if !primes.is_empty() {
            let mut changed = primes.clone();
            changed[0].polynomial ^= 1;
            rejects(terminals.clone(), changed, [43; 32], [47; 32]);
        }
    }
    rejects(terminals.clone(), primes.clone(), [44; 32], [47; 32]);
    rejects(terminals, primes, [43; 32], [48; 32]);
    reading
}
