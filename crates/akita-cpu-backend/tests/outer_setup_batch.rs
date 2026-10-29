#![cfg(feature = "external-setup-accumulation")]

use akita_config::{test_support::workspace_schedule_catalog, CommitmentConfig, SetupRequirements};
use akita_cpu_backend::{AkitaProverSetup, CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::{BasisMode, ChunkedWitnessCfg, PolynomialGroupLayout, ScheduleLookupKey};
use akita_prover::{
    backend::SetupBatchBackend, BackendRegistry, FixedFoldRoute, SelectedProverOpeningData,
    SetupPrefixProverRegistry,
};
use akita_serialization::{AkitaSerialize, Valid};
use akita_types::{
    setup_claims::{
        SetupBatchMemory, SetupBatchStatement, SetupClaim, SetupPolynomialId, SetupQuery,
    },
    AkitaVerifierSetup, FpExtEncoding, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
    SetupPrefixVerifierRegistry,
};
use akita_verifier::{AkitaVerifier, SetupClaimAccumulator, SetupClaimAdvice, SetupMode};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, One, PseudoMersenne, Ring};

struct Advice<'a, F: Field, E: Field> {
    backend: &'a CpuBackend<F, E>,
    polynomial: &'a SetupPolynomialId,
    calls: usize,
}

impl<F, E> SetupClaimAdvice<E> for Advice<'_, F, E>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F>,
{
    fn answer(&mut self, query: &SetupQuery<E>) -> Result<E, AkitaError> {
        self.calls += 1;
        self.backend
            .prepare_setup_query(self.polynomial, query, 32, SetupBatchMemory::Regenerate)
            .map(|(value, _)| value)
    }
}

#[test]
fn direct_outer_closes_collected_claims_without_an_extra_sumcheck() {
    run::<akita_config::proof_optimized::fp128::Dense>(14, false);
}

#[test]
fn offloaded_outer_batches_original_direct_and_stage3_claims() {
    run::<akita_config::RecursiveCommitmentConfig<akita_config::proof_optimized::fp32::Dense>>(
        24, true,
    );
}

fn run<Cfg>(nv: usize, offloaded: bool)
where
    Cfg: CommitmentConfig,
    Cfg::Field: Ring
        + PseudoMersenne
        + AkitaSerialize
        + Valid
        + jolt_field::Unreduced
        + jolt_field::WithCommitAccumulator,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + Ring + AkitaSerialize + jolt_field::Fold,
    <Cfg::Field as jolt_field::Unreduced>::Wide: From<Cfg::Field> + jolt_field::AdditiveGroup,
{
    const SESSION: &[u8] = b"integrated-outer-setup";
    let catalog =
        workspace_schedule_catalog::<Cfg>(ChunkedWitnessCfg::default_non_chunked()).unwrap();
    let row = catalog
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(nv),
        ))
        .unwrap();
    let levels = row.schedule().num_fold_levels();
    let direct_level = (0..levels - 1)
        .find(|&level| {
            row.schedule()
                .recursive_folds
                .get(level)
                .is_none_or(|next| next.params.setup_prefix().is_none())
        })
        .unwrap() as u32;
    let selected = row
        .schedule()
        .recursive_folds
        .iter()
        .enumerate()
        .find_map(|(level, fold)| {
            fold.params
                .setup_prefix()
                .map(|p| (level as u32, p.n_prefix().unwrap()))
        });
    assert_eq!(selected.is_some(), offloaded);
    let capacity = SetupRequirements::from_catalog(&catalog, nv, 1)
        .unwrap()
        .matrix_capacity();
    let capacity = akita_params::SetupMatrixCapacity {
        num_field_elements: capacity.num_field_elements.next_power_of_two(),
    };
    let setup = AkitaProverSetup::<Cfg::Field>::generate_with_capacity(nv, 1, capacity).unwrap();
    let backend = CpuBackend::<Cfg::Field, Cfg::ExtField>::new(setup.expanded.clone()).unwrap();
    let source = backend
        .import_source(vec![DensePoly::from_field_evals(
            nv,
            vec![Cfg::Field::one(); 1 << nv],
        )
        .unwrap()])
        .unwrap();
    let committed = backend
        .commit(
            &catalog,
            &source,
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();
    let claims = || {
        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            vec![Cfg::ExtField::from_u64(2); nv],
            vec![Cfg::ExtField::one()],
            &committed.committed_group,
        )
        .unwrap()])
        .unwrap()
    };
    let opening = || {
        SelectedProverOpeningData::from_committed_claims::<Cfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                vec![Cfg::ExtField::from_u64(2); nv],
                vec![Cfg::ExtField::one()],
                committed.committed_group.clone(),
            )
            .unwrap()])
            .unwrap(),
            vec![committed.private_handle.clone()],
            &catalog,
        )
        .unwrap()
    };
    let selection = opening().selection();
    let ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
        row.schedule(),
        opening().opening_layout(),
    )
    .unwrap();
    let mut prefixes = SetupPrefixProverRegistry::default();
    let mut public =
        SetupPrefixVerifierRegistry::new(setup.expanded.descriptor().setup_seed.clone());
    for id in ids {
        let slot = backend.prepare_setup_prefix(&id).unwrap();
        public.insert(slot.public.clone()).unwrap();
        prefixes.insert(slot).unwrap();
    }
    let mut registry = BackendRegistry::<Cfg>::new().unwrap();
    let owner = registry.register(&backend, &prefixes).unwrap();
    let verifier = AkitaVerifier::<Cfg>::for_selection(
        AkitaVerifierSetup::from_parts(setup.expanded.clone(), public).unwrap(),
        catalog.clone(),
        selection,
        None,
    )
    .unwrap();
    let prove = |batch: Option<(u32, &SetupBatchStatement<Cfg::ExtField>)>| {
        #[cfg(feature = "logging-transcript")]
        akita_transcript::clear_thread_events();
        let opening = match batch {
            Some((level, batch)) => opening().with_setup_batch(level, batch),
            None => opening(),
        };
        akita_prover::batched_prove::<Cfg, _>(
            setup.expanded.descriptor(),
            &catalog,
            &registry,
            opening,
            SESSION,
            BasisMode::Lagrange,
            &mut FixedFoldRoute::new(vec![owner; levels]),
        )
    };
    let ordinary = prove(None).unwrap();
    #[cfg(feature = "logging-transcript")]
    let ordinary_rounds = sumcheck_rounds();
    verifier
        .batched_verify(
            &ordinary,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            #[cfg(feature = "external-setup-accumulation")]
            None,
            #[cfg(feature = "external-setup-accumulation")]
            &mut akita_verifier::SetupMode::Check,
        )
        .unwrap();
    let (level, len) = selected.unwrap_or((0, capacity.num_field_elements));
    let polynomial =
        SetupPolynomialId::new::<Cfg::Field, Cfg::ExtField>(&setup.expanded, len).unwrap();
    let mut advice = Advice {
        backend: &backend,
        polynomial: &polynomial,
        calls: 0,
    };
    assert!(matches!(
        verifier.batched_verify(
            &ordinary,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            None,
            &mut SetupMode::Defer(&mut advice),
        ),
        Err(AkitaError::InvalidInput(_))
    ));
    assert_eq!(advice.calls, 0);
    let mut checked = SetupClaimAccumulator::new(polynomial.clone());
    verifier
        .batched_verify(
            &ordinary,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            Some(&mut checked),
            &mut SetupMode::Check,
        )
        .unwrap();
    assert!(checked.into_pending().is_empty());
    let mut collected = SetupClaimAccumulator::new(polynomial.clone());
    verifier
        .batched_verify(
            &ordinary,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            Some(&mut collected),
            &mut SetupMode::Defer(&mut advice),
        )
        .unwrap();
    let pending = collected.into_pending();
    assert_eq!(pending.len(), levels - 1);
    if offloaded {
        assert!(
            advice.calls < pending.len(),
            "Stage 3 input comes from the proof, not advice"
        );
        assert!(
            advice.calls > 0,
            "the collection also contains direct folds"
        );
    } else {
        assert_eq!(advice.calls, pending.len());
    }
    // Every collected obligation is the original exact functional and answer.
    for claim in &pending {
        let (value, _) = backend
            .prepare_setup_query(&polynomial, claim.query(), 32, SetupBatchMemory::Regenerate)
            .unwrap();
        assert_eq!(value, claim.value());
    }
    let ordinary_advice_calls = advice.calls;
    let inherited = SetupBatchStatement::new(polynomial.clone(), 32, pending).unwrap();
    let mut invalid = SetupClaimAccumulator::from_batch((levels - 1) as u32, inherited.clone());
    assert!(verifier
        .batched_verify(
            &ordinary,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            Some(&mut invalid),
            &mut SetupMode::Check,
        )
        .is_err());
    let outer = prove(Some((level, &inherited))).unwrap();
    #[cfg(feature = "logging-transcript")]
    assert_eq!(sumcheck_rounds(), ordinary_rounds);
    let mut checked = SetupClaimAccumulator::from_batch(level, inherited.clone());
    verifier
        .batched_verify(
            &outer,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            Some(&mut checked),
            &mut SetupMode::Check,
        )
        .unwrap();
    assert!(checked.into_pending().is_empty());
    assert!(verifier
        .batched_verify(
            &outer,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            None,
            &mut SetupMode::Check,
        )
        .is_err());
    let advice_calls_before_outer = advice.calls;
    let mut carried = SetupClaimAccumulator::from_batch(level, inherited.clone());
    verifier
        .batched_verify(
            &outer,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            Some(&mut carried),
            &mut SetupMode::Defer(&mut advice),
        )
        .unwrap();
    let carried = carried.into_pending();
    assert_eq!(carried.len(), inherited.claims().len() + levels - 1);
    assert_eq!(
        advice.calls - advice_calls_before_outer,
        ordinary_advice_calls - usize::from(!offloaded),
        "the selected batch's outer answer comes from the transcript, not advice"
    );
    for claim in &carried {
        let (value, _) = backend
            .prepare_setup_query(&polynomial, claim.query(), 32, SetupBatchMemory::Regenerate)
            .unwrap();
        assert_eq!(value, claim.value());
    }
    if offloaded {
        // A direct suffix fold can close the same mixed collection.
        let direct = prove(Some((direct_level, &inherited))).unwrap();
        #[cfg(feature = "logging-transcript")]
        assert_eq!(sumcheck_rounds(), ordinary_rounds);
        let mut checked = SetupClaimAccumulator::from_batch(direct_level, inherited.clone());
        verifier
            .batched_verify(
                &direct,
                SESSION,
                GroupBatchStatement::new(selection, claims()).unwrap(),
                BasisMode::Lagrange,
                Some(&mut checked),
                &mut SetupMode::Check,
            )
            .unwrap();
        assert!(checked.into_pending().is_empty());
        let advice_calls_before_direct = advice.calls;
        let mut carried = SetupClaimAccumulator::from_batch(direct_level, inherited.clone());
        verifier
            .batched_verify(
                &direct,
                SESSION,
                GroupBatchStatement::new(selection, claims()).unwrap(),
                BasisMode::Lagrange,
                Some(&mut carried),
                &mut SetupMode::Defer(&mut advice),
            )
            .unwrap();
        assert_eq!(
            advice.calls - advice_calls_before_direct,
            ordinary_advice_calls - 1,
            "the direct outer Stage 2 collects its transcript answer without advice"
        );
        // Pending claims must not disappear when the caller changes the mode.
        assert!(matches!(
            verifier.batched_verify(
                &ordinary,
                SESSION,
                GroupBatchStatement::new(selection, claims()).unwrap(),
                BasisMode::Lagrange,
                Some(&mut carried),
                &mut SetupMode::Check,
            ),
            Err(AkitaError::InvalidInput(_))
        ));
        assert_eq!(
            carried.into_pending().len(),
            inherited.claims().len() + levels - 1
        );
    }
    let mut changed = inherited.claims().to_vec();
    changed[0] = SetupClaim::new(
        polynomial.clone(),
        changed[0].query().clone(),
        changed[0].value() + Cfg::ExtField::one(),
    )
    .unwrap();
    let changed = SetupBatchStatement::new(polynomial, 32, changed).unwrap();
    let mut invalid = SetupClaimAccumulator::from_batch(level, changed.clone());
    assert!(verifier
        .batched_verify(
            &outer,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            Some(&mut invalid),
            &mut SetupMode::Check,
        )
        .is_err());
    assert!(prove(Some((level, &changed))).is_err());
}

#[cfg(feature = "logging-transcript")]
fn sumcheck_rounds() -> Vec<akita_transcript::ProtocolContextRecord> {
    let rounds = akita_transcript::thread_events()
        .into_iter()
        .filter_map(|event| {
            let akita_transcript::TranscriptEvent::Context(record) = event;
            (record.site_id[..4] == akita_transcript::SITE_FAMILY_SUMCHECK.to_le_bytes()
                && record.kind == akita_transcript::ProtocolMessageKind::ProofAtoms as u32)
                .then_some(record)
        })
        .collect::<Vec<_>>();
    assert!(!rounds.is_empty());
    rounds
}
