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
fn offloaded_outer_consumes_inherited_claims_in_both_modes() {
    run::<akita_config::RecursiveCommitmentConfig<akita_config::proof_optimized::fp128::Dense>>(
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
        let proof = akita_prover::batched_prove::<Cfg, _>(
            setup.expanded.descriptor(),
            &catalog,
            &registry,
            opening,
            SESSION,
            BasisMode::Lagrange,
            &mut FixedFoldRoute::new(vec![owner; levels]),
        );
        #[cfg(feature = "logging-transcript")]
        if proof.is_ok() {
            for event in akita_transcript::thread_events() {
                let akita_transcript::TranscriptEvent::Context(record) = event;
                let site = akita_transcript::ProtocolSiteId::from_bytes(record.site_id);
                if site.family == akita_transcript::SITE_FAMILY_STAGE3
                    || (site.family == akita_transcript::SITE_FAMILY_SUMCHECK
                        && site.invocation == akita_params::SumcheckProtocol::Stage3.tag())
                {
                    assert!(
                        row.schedule()
                            .recursive_folds
                            .get(site.level as usize)
                            .is_some_and(|fold| fold.params.setup_prefix().is_some()),
                        "direct folds must emit no Stage 3 transcript events"
                    );
                }
            }
        }
        proof
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
            None,
            akita_verifier::SetupMode::Check,
        )
        .unwrap();
    let (level, len) = selected.unwrap_or((0, capacity.num_field_elements));
    let polynomial =
        SetupPolynomialId::new::<Cfg::Field, Cfg::ExtField>(&setup.expanded, len).unwrap();
    let empty = SetupBatchStatement::new(polynomial.clone(), 32, Vec::new()).unwrap();
    assert_eq!(prove(Some((level, &empty))).unwrap(), ordinary);
    let mut empty_accumulator = SetupClaimAccumulator::from_batch(level, empty);
    verifier
        .batched_verify(
            &ordinary,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            Some(&mut empty_accumulator),
            SetupMode::Check,
        )
        .unwrap();
    assert!(empty_accumulator.into_pending().is_empty());
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
            SetupMode::Defer,
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
            SetupMode::Check,
        )
        .unwrap();
    assert!(checked.into_pending().is_empty());
    let mut collected = SetupClaimAccumulator::new(polynomial.clone()).with_advice(&mut advice);
    verifier
        .batched_verify(
            &ordinary,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            Some(&mut collected),
            SetupMode::Defer,
        )
        .unwrap();
    let pending = collected.into_pending();
    let direct_folds = (0..levels - 1)
        .filter(|&level| {
            row.schedule()
                .recursive_folds
                .get(level)
                .is_none_or(|next| next.params.setup_prefix().is_none())
        })
        .count();
    assert_eq!(pending.len(), direct_folds);
    assert_eq!(
        advice.calls,
        pending.len(),
        "only direct setup evaluations are exported"
    );
    if offloaded {
        assert!(
            pending.len() < levels - 1,
            "checked Stage 3 inputs must not be exported"
        );
        assert!(
            advice.calls > 0,
            "the collection also contains direct folds"
        );
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
            SetupMode::Check,
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
            SetupMode::Check,
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
            SetupMode::Check,
        )
        .is_err());
    let advice_calls_before_outer = advice.calls;
    let mut carried =
        SetupClaimAccumulator::from_batch(level, inherited.clone()).with_advice(&mut advice);
    verifier
        .batched_verify(
            &outer,
            SESSION,
            GroupBatchStatement::new(selection, claims()).unwrap(),
            BasisMode::Lagrange,
            Some(&mut carried),
            SetupMode::Defer,
        )
        .unwrap();
    let carried = carried.into_pending();
    assert_eq!(
        carried.len(),
        direct_folds
            + if offloaded {
                0
            } else {
                inherited.claims().len()
            },
        "a checked recursive batch consumes all inherited inputs"
    );
    assert_eq!(
        advice.calls - advice_calls_before_outer,
        ordinary_advice_calls,
        "every direct fold obtains its setup answer from advice"
    );
    for claim in &carried {
        let (value, _) = backend
            .prepare_setup_query(&polynomial, claim.query(), 32, SetupBatchMemory::Regenerate)
            .unwrap();
        assert_eq!(value, claim.value());
    }
    if offloaded {
        let next_input = SetupBatchStatement::new(polynomial.clone(), 32, carried).unwrap();
        let next_proof = prove(Some((level, &next_input))).unwrap();
        let mut checked = SetupClaimAccumulator::from_batch(level, next_input.clone());
        verifier
            .batched_verify(
                &next_proof,
                SESSION,
                GroupBatchStatement::new(selection, claims()).unwrap(),
                BasisMode::Lagrange,
                Some(&mut checked),
                SetupMode::Check,
            )
            .unwrap();
        assert!(checked.into_pending().is_empty());
        let mut deferred =
            SetupClaimAccumulator::from_batch(level, next_input).with_advice(&mut advice);
        verifier
            .batched_verify(
                &next_proof,
                SESSION,
                GroupBatchStatement::new(selection, claims()).unwrap(),
                BasisMode::Lagrange,
                Some(&mut deferred),
                SetupMode::Defer,
            )
            .unwrap();
        assert_eq!(
            deferred.into_pending().len(),
            direct_folds,
            "recursive generations must not retain their inherited input lists"
        );
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
                SetupMode::Check,
            )
            .unwrap();
        assert!(checked.into_pending().is_empty());
        let advice_calls_before_direct = advice.calls;
        let mut carried = SetupClaimAccumulator::from_batch(direct_level, inherited.clone())
            .with_advice(&mut advice);
        verifier
            .batched_verify(
                &direct,
                SESSION,
                GroupBatchStatement::new(selection, claims()).unwrap(),
                BasisMode::Lagrange,
                Some(&mut carried),
                SetupMode::Defer,
            )
            .unwrap();
        // Pending claims must not disappear when the caller changes the mode.
        assert!(matches!(
            verifier.batched_verify(
                &ordinary,
                SESSION,
                GroupBatchStatement::new(selection, claims()).unwrap(),
                BasisMode::Lagrange,
                Some(&mut carried),
                SetupMode::Check,
            ),
            Err(AkitaError::InvalidInput(_))
        ));
        assert_eq!(
            carried.into_pending().len(),
            inherited.claims().len() + direct_folds
        );
        assert_eq!(
            advice.calls - advice_calls_before_direct,
            ordinary_advice_calls,
            "a direct batch target uses ordinary Stage 2 advice, without a Stage 3 claim"
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
            SetupMode::Check,
        )
        .is_err());
    if offloaded {
        assert!(prove(Some((level, &changed))).is_err());
    } else {
        // A direct prover only binds inherited inputs; the verifier checks them.
        let invalid_proof = prove(Some((level, &changed))).unwrap();
        let mut invalid = SetupClaimAccumulator::from_batch(level, changed);
        assert!(verifier
            .batched_verify(
                &invalid_proof,
                SESSION,
                GroupBatchStatement::new(selection, claims()).unwrap(),
                BasisMode::Lagrange,
                Some(&mut invalid),
                SetupMode::Check,
            )
            .is_err());
    }
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
