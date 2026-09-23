use akita_config::CommitmentConfig;
use akita_cpu_backend::CommitmentHandle;
use akita_prover::SelectedProverOpeningData;
use akita_types::{
    CommittedGroup, GroupBatchStatement, OpeningClaims, OpeningScheduleSelection,
    PolynomialGroupClaims,
};
use jolt_field::Field;

#[allow(clippy::type_complexity)]
pub(super) fn prover_claims<'a, Cfg>(
    schedules: &akita_config::TrustedScheduleCatalog<Cfg>,
    selection: OpeningScheduleSelection,
    point: &'a [Cfg::ExtField],
    evaluations: &[Cfg::ExtField],
    commitment: &'a CommittedGroup<Cfg::Field>,
    handle: CommitmentHandle<Cfg::Field, Cfg::ExtField, Cfg>,
) -> SelectedProverOpeningData<
    'a,
    Cfg::ExtField,
    CommitmentHandle<Cfg::Field, Cfg::ExtField, Cfg>,
    Cfg::Field,
>
where
    Cfg: CommitmentConfig,
{
    let group =
        PolynomialGroupClaims::new(point.to_vec(), evaluations.to_vec(), commitment.clone())
            .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    let selected = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        opening_claims,
        vec![handle],
        schedules,
    )
    .expect("valid prover opening data");
    assert_eq!(selected.selection(), selection);
    selected
}

pub(super) fn verifier_claims<'a, E: Field, F: Field>(
    selection: OpeningScheduleSelection,
    point: &[E],
    openings: &[E],
    commitment: &'a CommittedGroup<F>,
) -> GroupBatchStatement<'a, E, F> {
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.to_vec(),
        openings.to_vec(),
        commitment,
    )
    .expect("valid verifier claims group")])
    .expect("valid verifier claims");
    GroupBatchStatement::new(selection, claims).expect("valid verifier statement")
}
