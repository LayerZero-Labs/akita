//! Typed-prefix admission and Stage 3 transcript mutation scenarios.

use super::*;

#[test]
fn stage3_prefix_context_rejects_manifest_swap_and_missing_context_before_proof() {
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
        let first = row.schedule().recursive_folds[0]
            .params
            .setup_prefix()
            .unwrap()
            .slot_id()
            .unwrap();
        let setup = scheme.setup_prover(18, 1).unwrap();
        let backend = CpuBackend::<F, F>::new(setup.expanded.clone()).unwrap();
        let index = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
        let make_manifest = |setup_group| {
            let profiles =
                BinaryCommittedGroupManifest::commitment_profiles(&index, 1, setup_group).unwrap();
            let descriptor =
                BinarySourceDescriptor::new(index.clone(), vec![[29; 32]], profiles).unwrap();
            BinaryCommittedGroupManifest::new(descriptor, setup_group, &carrier).unwrap()
        };
        let installed_manifest = make_manifest(0);
        let typed = BinaryTypedPrefix::new(
            &installed_manifest,
            true,
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
            .prepare_binary_a_prefixes(&installed_manifest, &typed, &first, &[])
            .unwrap();
        let ordinary = scheme.setup_verifier(&setup).unwrap();
        let prepared = backend
            .prepare_binary_a_verifier_setup(
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
                    akita_types::binary_source::BINARY_COMMITTED_PAYLOAD_COEFFICIENTS
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
                let mut replay =
                    new_verifier_channel(b"binary-source-end-to-end-v1", b"caller", &[]).unwrap();
                replay.public_message(b"application-prefix");
                verifier.batched_verify_in_session(&mut replay, statement, BasisMode::Lagrange)
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
        .prepare_binary_a_verifier_setup(
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
        .prepare_binary_a_verifier_setup(
            ordinary_verifier_setup,
            manifest,
            typed,
            first_prefix,
            &foreign.public
        )
        .is_err());
}

/// Time the root's A arithmetic and actual Stage 3 transcript replay.
///
/// Capture is active only on the calling thread for the duration of `verify`,
/// which keeps unrelated fixture spans out and requires exactly one completed
/// root A span per verification. The subscriber itself is the binary's global
/// default, installed once, for the reason documented at
/// `common::proof_size::install_capture_layer`: with a scoped subscriber,
/// another thread that reaches the root A callsite first without one can cache
/// it as never interested, and this capture then records nothing.
pub(super) fn measure_root_a<T>(verify: impl FnOnce() -> T) -> (T, std::time::Duration) {
    use std::cell::RefCell;
    use std::sync::Once;
    use std::time::{Duration, Instant};
    use tracing::subscriber::Interest;
    use tracing::{span::Id, Metadata, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::Layer;

    const ROOT_A: &str = "binary_root_a_stage3";
    thread_local! {
        /// The completed root A duration, while this thread's capture is active.
        static CAPTURE: RefCell<Option<Option<Duration>>> = const { RefCell::new(None) };
    }
    fn captures(metadata: &Metadata<'_>) -> bool {
        metadata.is_span() && metadata.name() == ROOT_A
    }

    struct Started(Instant);
    struct Capture;
    impl<S> Layer<S> for Capture
    where
        S: Subscriber + for<'lookup> LookupSpan<'lookup>,
    {
        fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
            if captures(metadata) {
                Interest::always()
            } else {
                Interest::never()
            }
        }

        fn enabled(&self, metadata: &Metadata<'_>, _context: Context<'_, S>) -> bool {
            captures(metadata)
        }

        fn on_enter(&self, id: &Id, context: Context<'_, S>) {
            let span = context.span(id).expect("entered span is registered");
            let mut extensions = span.extensions_mut();
            if extensions.get_mut::<Started>().is_none() {
                extensions.insert(Started(Instant::now()));
            }
        }

        fn on_close(&self, id: Id, context: Context<'_, S>) {
            let span = context.span(&id).expect("closing span is registered");
            let elapsed = span
                .extensions()
                .get::<Started>()
                .expect("root A span was entered")
                .0
                .elapsed();
            CAPTURE.with(|capture| {
                if let Some(slot) = capture.borrow_mut().as_mut() {
                    assert!(
                        slot.replace(elapsed).is_none(),
                        "verification emitted more than one root A span"
                    );
                }
            });
        }
    }

    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(Capture))
            .expect("the root A capture is the test binary's only global subscriber");
    });
    CAPTURE.with(|capture| *capture.borrow_mut() = Some(None));
    let output = verify();
    let elapsed = CAPTURE
        .with(|capture| capture.borrow_mut().take())
        .expect("the capture stays active for the whole verification")
        .expect("verification emitted exactly one root A span");
    (output, elapsed)
}
