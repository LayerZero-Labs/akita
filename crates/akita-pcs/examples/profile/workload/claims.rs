use akita_config::CommitmentConfig;
use akita_prover::SelectedProverOpeningData;
use akita_types::{
    AkitaCommitmentHint, CommittedGroup, GroupBatchStatement, OpeningClaims,
    OpeningScheduleSelection, PolynomialGroupClaims,
};
use jolt_field::{Field, Zero};

pub(super) fn prover_claims<'a, Cfg, P>(
    schedules: &akita_config::TrustedScheduleCatalog,
    selection: OpeningScheduleSelection,
    point: &'a [Cfg::ExtField],
    polynomials: &'a [&'a P],
    commitment: &'a CommittedGroup<Cfg::Field>,
    hint: AkitaCommitmentHint<Cfg::Field>,
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
    .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    let selected = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        opening_claims,
        vec![hint],
        vec![polynomials],
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
