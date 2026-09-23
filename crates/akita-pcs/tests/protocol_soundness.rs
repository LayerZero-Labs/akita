#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend, DensePoly, SelectedProverOpeningData};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::{
    lagrange_weights, AkitaCommitmentHint, AkitaScheduleLookupKey, AkitaVerifierSetup, BasisMode,
    CommittedGroup, CommittedGroupBatchProfile, FpExtEncoding, GroupBatchStatement, OpeningClaims,
    OpeningScheduleSelection, PolynomialGroupClaims, PolynomialGroupLayout,
};
use jolt_field::{
    CanonicalBytes, CanonicalEncoding, ExtField, Field, Fold, One, PseudoMersenne, Ring, Unreduced,
    Zero,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod common;
use common::load_workspace_scheme;
#[cfg(feature = "logging-transcript")]
use common::native_mutations::{
    assert_native_ranges_match_context, representative_native_mutation_ranges,
    selected_sumcheck_protocols,
};

const STACK_SIZE: usize = 256 * 1024 * 1024;

fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn soundness test thread")
        .join()
        .expect("soundness test thread panicked");
}

fn selection_for<Cfg: CommitmentConfig>(
    commitment: &CommittedGroup<Cfg::Field>,
    schedules: &TrustedScheduleCatalog<Cfg>,
) -> OpeningScheduleSelection {
    schedules
        .resolve_profiles(&CommittedGroupBatchProfile {
            final_group: *commitment.profile(),
            precommitteds: Vec::new(),
        })
        .expect("select schedule")
        .selection()
}

fn prove_input<'a, Cfg, P>(
    selection: OpeningScheduleSelection,
    point: &'a [Cfg::ExtField],
    polynomials: &'a [&'a P],
    commitment: &'a CommittedGroup<Cfg::Field>,
    hint: AkitaCommitmentHint<Cfg::Field>,
    schedules: &TrustedScheduleCatalog<Cfg>,
) -> SelectedProverOpeningData<
    'a,
    Cfg::ExtField,
    akita_prover::PreparedProverGroup<'a, P>,
    Cfg::Field,
>
where
    Cfg: CommitmentConfig,
    P: akita_prover::RootPolyMeta<Cfg::Field>,
{
    let group = PolynomialGroupClaims::new(
        point.to_vec(),
        vec![Cfg::ExtField::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover group");
    let selected = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        OpeningClaims::from_groups(vec![group]).expect("valid prover claims"),
        vec![hint],
        vec![polynomials],
        schedules,
    )
    .expect("valid prover opening data");
    assert_eq!(selected.selection(), selection);
    selected
}

fn verify_input<'a, Cfg: CommitmentConfig>(
    selection: OpeningScheduleSelection,
    point: &[Cfg::ExtField],
    opening: Cfg::ExtField,
    commitment: &'a CommittedGroup<Cfg::Field>,
) -> GroupBatchStatement<'a, Cfg::ExtField, Cfg::Field> {
    let group = PolynomialGroupClaims::new(point.to_vec(), vec![opening], commitment)
        .expect("valid verifier group");
    GroupBatchStatement::new(
        selection,
        OpeningClaims::from_groups(vec![group]).expect("valid verifier claims"),
    )
    .expect("valid verifier statement")
}

type NativeFixture<F, E> = (
    AkitaVerifierSetup<F>,
    CommittedGroup<F>,
    Vec<u8>,
    Vec<E>,
    E,
    OpeningScheduleSelection,
);

fn make_dense_fixture<F, Cfg>(
    scheme: &AkitaCommitmentScheme<Cfg>,
    num_vars: usize,
    label: &'static [u8],
) -> NativeFixture<F, Cfg::ExtField>
where
    F: CanonicalBytes
        + CanonicalEncoding
        + Unreduced
        + Field
        + Ring
        + PseudoMersenne
        + Valid
        + AkitaDeserialize<Context = ()>
        + AkitaSerialize
        + jolt_field::WithCommitAccumulator
        + 'static,
    <F as Unreduced>::Wide: From<F>,
    Cfg: CommitmentConfig<Field = F>,
    Cfg::ExtField: ExtField<F> + FpExtEncoding<F> + Unreduced + Fold + AkitaSerialize,
{
    let mut rng = StdRng::seed_from_u64(0x0ddc_0ffe_e123_4567);
    let evaluations = (0..1usize << num_vars)
        .map(|_| F::from_u128_reduced(rng.gen::<u128>()))
        .collect::<Vec<_>>();
    let polynomial = DensePoly::from_field_evals(num_vars, &evaluations).expect("dense fixture");
    let point = (0..num_vars)
        .map(|_| {
            let coordinates = (0..Cfg::ExtField::DEGREE)
                .map(|_| F::from_u128_reduced(rng.gen::<u128>()))
                .collect::<Vec<_>>();
            Cfg::ExtField::from_base_slice(&coordinates)
        })
        .collect::<Vec<_>>();
    let weights = lagrange_weights(&point).expect("Lagrange weights");
    let opening = evaluations
        .iter()
        .zip(weights)
        .fold(Cfg::ExtField::zero(), |sum, (&value, weight)| {
            sum + Cfg::ExtField::lift_base(value) * weight
        });

    let setup = scheme.setup_prover(num_vars, 1).expect("prover setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("prover stack");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        prover_state: hint,
    } = scheme
        .commit(
            &setup,
            std::slice::from_ref(&polynomial),
            stack.commitment(),
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commitment");
    let selection = selection_for::<Cfg>(&commitment, scheme.schedules());
    let polynomial_refs = [&polynomial];
    let proof = scheme
        .batched_prove(
            &setup,
            prove_input::<Cfg, _>(
                selection,
                &point,
                &polynomial_refs,
                &commitment,
                hint,
                scheme.schedules(),
            ),
            &stack,
            label,
            BasisMode::Lagrange,
        )
        .expect("native proof");
    (verifier_setup, commitment, proof, point, opening, selection)
}

fn assert_native_soundness_boundaries<F, Cfg>(num_vars: usize, label: &'static [u8])
where
    F: CanonicalBytes
        + CanonicalEncoding
        + Unreduced
        + Field
        + Ring
        + PseudoMersenne
        + Valid
        + AkitaDeserialize<Context = ()>
        + AkitaSerialize
        + jolt_field::WithCommitAccumulator
        + 'static,
    <F as Unreduced>::Wide: From<F>,
    Cfg: CommitmentConfig<Field = F> + 'static,
    Cfg::ExtField: ExtField<F> + FpExtEncoding<F> + Unreduced + Fold + AkitaSerialize,
{
    let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");
    #[cfg(feature = "logging-transcript")]
    akita_transcript::clear_thread_events();
    let (setup, commitment, proof, point, opening, selection) =
        make_dense_fixture::<F, Cfg>(&scheme, num_vars, label);
    #[cfg(feature = "logging-transcript")]
    let proof_ranges = akita_transcript::thread_proof_ranges();
    #[cfg(feature = "logging-transcript")]
    assert_native_ranges_match_context(&proof_ranges);
    let resolved = scheme
        .schedules()
        .resolve_selection(selection)
        .expect("selected schedule");
    let native_bound = akita_schedules::expanded_schedule_native_proof_bound(
        &AkitaScheduleLookupKey {
            final_group: resolved.profiles().final_group.group,
            precommitteds: resolved.profiles().precommitteds.clone(),
        },
        resolved.schedule(),
        &akita_config::policy_of::<Cfg>(),
    )
    .expect("native proof bound");
    assert!(
        proof.len() <= native_bound,
        "valid native proof exceeds its schedule-derived parser bound"
    );
    let verify = |candidate: &[u8], claimed: Cfg::ExtField, session: &[u8]| {
        scheme.batched_verify(
            candidate,
            &setup,
            session,
            verify_input::<Cfg>(selection, &point, claimed, &commitment),
            BasisMode::Lagrange,
        )
    };

    verify(&proof, opening, label).expect("honest native proof must verify");
    verify(&proof, opening + Cfg::ExtField::one(), label)
        .expect_err("claimed opening must be bound");
    verify(&proof, opening, b"soundness/wrong-session").expect_err("session must be bound");

    let mut trailing = proof.clone();
    trailing.push(0);
    assert!(matches!(
        verify(&trailing, opening, label),
        Err(AkitaError::InvalidProof)
    ));
    assert!(matches!(
        verify(&proof[..proof.len() - 1], opening, label),
        Err(AkitaError::InvalidProof)
    ));

    #[cfg(feature = "logging-transcript")]
    let mutation_offsets = {
        let by_role = representative_native_mutation_ranges(proof_ranges);
        assert!(
            !by_role.is_empty(),
            "native proof must expose fixed-shape semantic-role ranges"
        );
        let sumcheck_protocols = selected_sumcheck_protocols(&by_role);
        for protocol in [
            akita_types::SumcheckProtocol::Stage1,
            akita_types::SumcheckProtocol::Stage2,
        ] {
            assert!(
                sumcheck_protocols.contains(&protocol),
                "native fixture must exercise {protocol:?} sumcheck messages"
            );
        }
        for (family, protocol) in [
            (
                akita_transcript::SITE_FAMILY_PHYSICAL_L2,
                akita_types::SumcheckProtocol::PhysicalL2,
            ),
            (
                akita_transcript::SITE_FAMILY_STAGE3,
                akita_types::SumcheckProtocol::Stage3,
            ),
        ] {
            if by_role.iter().any(|(bucket, _)| bucket.family == family) {
                assert!(
                    sumcheck_protocols.contains(&protocol),
                    "fixture with {protocol:?} messages must mutate its sumcheck"
                );
            }
        }
        if Cfg::ExtField::DEGREE > 1 {
            assert!(
                by_role.iter().any(|(bucket, _)| bucket.family
                    == akita_transcript::SITE_FAMILY_EXTENSION_OPENING_REDUCTION),
                "extension-field workload must exercise native EOR messages"
            );
            assert!(
                sumcheck_protocols
                    .contains(&akita_types::SumcheckProtocol::ExtensionOpeningReduction),
                "extension-field workload must exercise EOR sumcheck messages"
            );
        }
        by_role
            .into_iter()
            .map(|(_, range)| range.start)
            .collect::<Vec<_>>()
    };
    #[cfg(not(feature = "logging-transcript"))]
    let mutation_offsets = vec![
        0,
        proof.len() / 4,
        proof.len() / 2,
        proof.len() * 3 / 4,
        proof.len() - 1,
    ];
    for offset in mutation_offsets {
        let mut malformed = proof.clone();
        malformed[offset] ^= 1;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify(&malformed, opening, label)
        }));
        assert!(
            matches!(outcome, Ok(Err(AkitaError::InvalidProof))),
            "mutated native proof at offset {offset} must be classified as InvalidProof: {outcome:?}"
        );
    }
}

#[test]
fn fp128_native_stream_rejects_statement_session_and_proof_mutations() {
    run_on_large_stack(|| {
        assert_native_soundness_boundaries::<fp128::Field, fp128::Dense>(
            14,
            b"soundness/fp128-native",
        );
    });
}

#[test]
fn fp32_extension_native_stream_rejects_statement_session_and_proof_mutations() {
    run_on_large_stack(|| {
        assert_native_soundness_boundaries::<fp32::Field, fp32::Dense>(
            20,
            b"soundness/fp32-extension-native",
        );
    });
}

#[test]
fn small_field_dense_uncataloged_roots_fail_fast() {
    let fp32_catalog = akita_config::test_support::workspace_schedule_catalog::<fp32::Dense>()
        .expect("fp32 dense catalog");
    let fp64_catalog = akita_config::test_support::workspace_schedule_catalog::<fp64::Dense>()
        .expect("fp64 dense catalog");
    for result in [
        fp32_catalog.resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(8),
        )),
        fp64_catalog.resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(9),
        )),
    ] {
        assert!(matches!(
            result,
            Err(akita_error::AkitaError::UnsupportedSchedule(_))
        ));
    }
}

#[test]
fn tiny_roots_and_setup_capacities_are_rejected() {
    let scheme = load_workspace_scheme::<fp128::Dense>().expect("workspace schedule catalog");
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(4));
    assert!(matches!(
        scheme.schedules().resolve_key(&key),
        Err(akita_error::AkitaError::UnsupportedSchedule(_))
    ));
    let error = scheme
        .setup_prover(4, 1)
        .expect_err("tiny setup capacity must reject");
    assert!(matches!(error, akita_error::AkitaError::InvalidSetup(_)));
}
