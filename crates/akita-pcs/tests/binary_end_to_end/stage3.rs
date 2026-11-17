//! Typed-prefix admission and Stage 3 transcript mutation scenarios.

use super::*;
use akita_params::{GroupCommitPhaseParams, PrecommittedGroupProfiles};
use akita_schedules::ResolvedScheduleRow;

fn with_binary_commit_fixture(
    test: impl FnOnce(
            &AkitaCommitmentScheme<BinaryTestSuffix>,
            &akita_pcs::AkitaProverSetup<F>,
            &CpuBackend<F, F>,
            &ResolvedScheduleRow,
            GroupCommitPhaseParams,
            &dyn Fn(u64) -> BinaryCommittedGroupManifest,
        ) + Send
        + 'static,
) {
    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = AkitaCommitmentScheme::<BinaryTestSuffix>::new(test_catalog());
        let row = scheme
            .schedules()
            .resolve_key(&ScheduleLookupKey::single(
                PolynomialGroupLayout::singleton(18),
            ))
            .unwrap();
        let carrier = row.profiles().final_group;
        let setup = scheme.setup_prover(18, 1).unwrap();
        let backend = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
        let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
        let make_manifest = |setup_group| {
            let profiles = BinaryCommittedGroupManifest::profiles_for_field(
                &index,
                1,
                setup_group,
                BinaryFieldProfile::P128,
            )
            .unwrap();
            let descriptor =
                BinarySourceDescriptor::new(index.clone(), vec![[29; 32]], profiles).unwrap();
            BinaryCommittedGroupManifest::with_field_profile(
                descriptor,
                setup_group,
                &carrier,
                BinaryFieldProfile::P128,
            )
            .unwrap()
        };
        test(&scheme, &setup, &backend, row, carrier, &make_manifest);
    });
}

#[test]
fn stage3_prefix_context_rejects_manifest_swap_and_missing_context_before_proof() {
    with_binary_commit_fixture(|scheme, setup, backend, row, carrier, make_manifest| {
        let first = row.schedule().recursive_folds[0]
            .params
            .setup_prefix()
            .unwrap()
            .slot_id()
            .unwrap();
        let installed_manifest = make_manifest(0);
        let typed = BinaryTypedPrefix::new(
            &installed_manifest,
            0,
            row.schedule().recursive_folds[0].params.d_a(),
        )
        .unwrap();
        let swapped_manifest = make_manifest(1);
        assert_eq!(
            installed_manifest.descriptor().index_map(),
            swapped_manifest.descriptor().index_map()
        );
        let prefixes = backend
            .prepare_binary_typed_prefixes(&installed_manifest, &typed, &first, &[])
            .unwrap();
        let ordinary = scheme.setup_verifier(setup).unwrap();
        let prepared = backend
            .prepare_binary_typed_verifier_setup(
                &ordinary,
                &installed_manifest,
                &typed,
                &first,
                &prefixes.get(&first).unwrap().public,
            )
            .unwrap();
        for (name, manifest, verifier_setup) in [
            ("missing typed context", installed_manifest, ordinary),
            ("same-geometry manifest swap", swapped_manifest, prepared),
        ] {
            // A structurally valid public payload suffices: context admission
            // must reject the statement before consuming any proof atom.
            let commitment = CommittedGroup::new_binary(
                carrier,
                akita_types::Commitment::new(akita_types::RingVec::from_coeffs(vec![
                    F::zero();
                    manifest
                        .payload_coefficients()
                        .unwrap()
                ])),
                manifest.clone(),
            )
            .unwrap();
            let claim = honest_claim(manifest.descriptor(), &[0; 8]);
            let claims = PolynomialGroupClaims::<F, _>::new_binary(commitment.clone(), claim)
                .unwrap()
                .map_commitment(|_| &commitment);
            let statement = selected_statement::<BinaryTestSuffix>(
                OpeningClaims::from_groups(vec![claims]).unwrap(),
                scheme.schedules(),
            );
            let verifier = scheme.verifier(verifier_setup).unwrap();
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut replay = new_verifier_channel(b"binary-source-end-to-end-v1", &[]);
                replay.public_message(&BytesMsg(b"application-prefix"));
                verifier.batched_verify_embedded(
                    &mut replay,
                    statement,
                    BasisMode::Lagrange,
                    SetupPolicy::Check(None),
                )
            }));
            assert!(result.is_ok(), "{name} caused a verifier panic");
            assert!(
                matches!(
                    result.unwrap(),
                    Err(AkitaError::InvalidInput(message))
                        if message == "binary statement differs from its prepared typed prefix"
                ),
                "{name} did not reject at prefix-context admission"
            );
        }
    });
}

#[test]
fn binary_commit_context_checks_profile_and_index_but_not_roster_contents() {
    with_binary_commit_fixture(|scheme, _, backend, _, carrier, make_manifest| {
        for setup_group in [0, 1, 7] {
            let manifest = make_manifest(setup_group);
            let source = backend
                .import_binary_source(manifest, carrier, vec![vec![1; 8]])
                .unwrap();
            let precommitteds = (setup_group > 0).then(|| {
                PrecommittedGroupProfiles::from_profiles(vec![carrier; setup_group as usize])
                    .unwrap()
            });
            let context = |profile| match &precommitteds {
                Some(roster) => GroupContext::explicit_with_precommitted_groups(profile, roster),
                None => GroupContext::explicit(profile),
            };
            let commitment = backend
                .commit(scheme.schedules(), &source, context(&carrier))
                .unwrap()
                .committed_group;
            assert_eq!(commitment.profile(), &carrier);
            let mut wrong_profile = carrier;
            wrong_profile.group = PolynomialGroupLayout::singleton(17);
            assert!(matches!(
                backend.commit(scheme.schedules(), &source, context(&wrong_profile)),
                Err(AkitaError::InvalidInput(message))
                    if message == "binary source differs from its explicit carrier profile"
            ));
            let wrong_roster = PrecommittedGroupProfiles::from_profiles(vec![carrier; 2]).unwrap();
            let mut wrong_contexts = vec![GroupContext::explicit_with_precommitted_groups(
                &carrier,
                &wrong_roster,
            )];
            if setup_group > 0 {
                wrong_contexts.push(GroupContext::explicit(&carrier));
            }
            for wrong_context in wrong_contexts {
                assert!(matches!(
                    backend.commit(scheme.schedules(), &source, wrong_context),
                    Err(AkitaError::InvalidInput(message)) if message
                        == "binary commitment group index differs from the source setup group"
                ));
            }
            if setup_group > 0 {
                let foreign_roster = PrecommittedGroupProfiles::from_profiles(vec![
                    wrong_profile;
                    setup_group
                        as usize
                ])
                .unwrap();
                let foreign = backend
                    .commit(
                        scheme.schedules(),
                        &source,
                        GroupContext::explicit_with_precommitted_groups(&carrier, &foreign_roster),
                    )
                    .unwrap()
                    .committed_group;
                let mut expected_bytes = Vec::new();
                commitment
                    .serialize_compressed(&mut expected_bytes)
                    .unwrap();
                let mut foreign_bytes = Vec::new();
                foreign.serialize_compressed(&mut foreign_bytes).unwrap();
                assert_eq!(foreign_bytes, expected_bytes);
            }
        }
    });
}

pub(super) fn reject_each_message(
    proof: &[u8],
    ranges: &[akita_transcript::ProofMessageRange],
    prefix_rounds: usize,
    mut reject: impl FnMut(&[u8]),
) {
    for offset in message_offsets(ranges, prefix_rounds) {
        let mut altered = proof.to_vec();
        altered[offset] ^= 1;
        reject(&altered);
    }
}

/// Every binary group contributes one product, one quadratic sumcheck and one
/// prefix evaluation, including a group containing several polynomials.
pub(super) fn message_offsets(
    ranges: &[akita_transcript::ProofMessageRange],
    prefix_rounds: usize,
) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut round_messages = 0;
    let mut terminal_messages = 0;
    for range in ranges {
        let site = akita_transcript::ProtocolSiteId::from_bytes(range.context.site_id);
        if site.level != 0
            || range.context.kind != akita_transcript::ProtocolMessageKind::ProofAtoms as u32
        {
            continue;
        }
        let round = site.family == akita_transcript::SITE_FAMILY_SUMCHECK
            && site.invocation == akita_params::SumcheckProtocol::Stage3.tag();
        let terminal = site.family == akita_transcript::SITE_FAMILY_STAGE3;
        if round || terminal {
            assert!(range.len.is_multiple_of(16));
            for offset in (range.start..range.start + range.len).step_by(16) {
                offsets.push(offset);
                if round {
                    round_messages += 1;
                } else {
                    terminal_messages += 1;
                }
            }
        }
    }
    assert_eq!(round_messages, 2 * prefix_rounds);
    assert_eq!(
        terminal_messages, 2,
        "one product claim and one prefix evaluation"
    );
    offsets
}

pub(super) fn reject_foreign_prefixes(
    backend: &CpuBackend<F, F>,
    ordinary_verifier_setup: &akita_types::AkitaVerifierSetup<F>,
    manifest: &BinaryCommittedGroupManifest,
    typed: &BinaryTypedPrefix,
    first_prefix: &akita_params::SetupPrefixSlotId,
) {
    let ordinary = backend.prepare_setup_prefix(first_prefix).unwrap();
    assert!(backend
        .prepare_binary_typed_verifier_setup(
            ordinary_verifier_setup,
            manifest,
            typed,
            first_prefix,
            &ordinary.public
        )
        .is_err());
    let mut foreign = typed
        .setup_coefficients::<F>(
            &ordinary_verifier_setup
                .expanded()
                .descriptor()
                .setup_seed
                .seed,
            manifest,
            first_prefix.n_prefix().unwrap(),
        )
        .unwrap();
    foreign[0] += F::one();
    let foreign = backend
        .prepare_setup_prefix_from_coefficients(first_prefix, foreign)
        .unwrap();
    assert!(backend
        .prepare_binary_typed_verifier_setup(
            ordinary_verifier_setup,
            manifest,
            typed,
            first_prefix,
            &foreign.public
        )
        .is_err());
}

#[test]
fn stage3_p64_extension_endpoint_matches_the_committed_base_prefix_oracle() {
    p64_typed_prefix_endpoint(0);
}

pub(super) fn p64_typed_prefix_endpoint(planes: usize) -> BinaryCommittedGroupManifest {
    use akita_algebra::eq_poly::EqPolynomial;
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_types::binary::{
        binary_word_manifest_fixture, BinaryAProduct, BinaryAReadGuard, BinaryRootWitnessPlan,
    };
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as Base};
    type E = Ext2<Base>;
    let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
    let plan = BinaryRootWitnessPlan::for_profile(
        manifest.descriptor(),
        3,
        64,
        BinaryFieldProfile::P64,
        planes,
    )
    .unwrap();
    let typed = BinaryTypedPrefix::from_plan(&manifest, &plan).unwrap();
    let extension = |i: usize| E::from_base_fn(|c| Base::from_u64((17 + i * 31 + c * 43) as u64));
    let stage2 = (0..plan.repacking().domain_len().trailing_zeros())
        .map(|i| extension(i as usize))
        .collect::<Vec<_>>();
    let rho = (0..3)
        .map(|row| std::array::from_fn(|k| extension(row * 648 + k)))
        .collect::<Vec<_>>();
    let a = BinaryAProduct::new(&plan, &rho, &stage2, typed.domain_len).unwrap();
    let batches = [
        vec![extension(3000); manifest.b_rows()],
        vec![extension(4000); manifest.d_label(planes).unwrap().rank as usize],
    ];
    let weights = typed
        .prover_weights(
            &a,
            &stage2,
            extension(5000),
            [&batches[0], &batches[1]],
            typed.domain_len,
        )
        .unwrap();
    let setup = typed
        .setup_coefficients::<Base>(&[41; 32], &manifest, typed.domain_len)
        .unwrap();
    let point = (0..typed.domain_len.trailing_zeros())
        .map(|i| extension(6000 + i as usize))
        .collect::<Vec<_>>();
    let equality = EqPolynomial::evals(&point).unwrap();
    let eval = setup
        .iter()
        .zip(&equality)
        .fold(E::zero(), |sum, (&v, &eq)| sum + E::lift_base(v) * eq);
    assert_ne!(eval.base_coefficient(1), Base::zero());
    let dense = weights
        .iter()
        .zip(equality)
        .fold(E::zero(), |sum, (&v, eq)| sum + v * eq);
    let _guard = BinaryAReadGuard::forbid();
    let online = a.evaluate(&point).unwrap()
        + typed
            .digit_weight(&point, &stage2, extension(5000), [&batches[0], &batches[1]])
            .unwrap();
    assert_eq!(online, dense);
    for coordinate in 0..2 {
        let altered = eval
            + E::from_base_fn(|c| {
                if c == coordinate {
                    Base::one()
                } else {
                    Base::zero()
                }
            });
        assert_ne!(online * altered, online * eval);
    }
    manifest
}
