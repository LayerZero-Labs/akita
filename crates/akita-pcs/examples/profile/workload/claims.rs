use akita_field::FieldCore;
use akita_prover::{ProverOpeningData, SelectedProverOpeningData};
use akita_types::{
    AkitaCommitmentHint, CommittedGroup, GroupBatchStatement, OpeningClaims,
    OpeningScheduleSelection, PolynomialGroupClaims,
};

pub(super) fn prover_claims<'a, E: FieldCore, P, CommitF: FieldCore>(
    selection: OpeningScheduleSelection,
    point: &'a [E],
    polynomials: &'a [&'a P],
    commitment: &'a CommittedGroup<CommitF>,
    hint: AkitaCommitmentHint<CommitF>,
) -> SelectedProverOpeningData<'a, E, akita_prover::PreparedProverGroup<'a, P>, CommitF>
where
    P: akita_prover::RootPolyMeta<CommitF>,
{
    let group = PolynomialGroupClaims::new(
        point.to_vec(),
        vec![E::zero(); polynomials.len()],
        commitment.clone(),
    )
    .expect("valid prover claims group");
    let opening_claims = OpeningClaims::from_groups(vec![group]).expect("valid prover claims");
    (
        selection,
        ProverOpeningData::new(opening_claims, vec![hint], vec![polynomials])
            .expect("valid prover opening data"),
    )
}

pub(super) fn verifier_claims<'a, E: FieldCore, F: FieldCore>(
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
