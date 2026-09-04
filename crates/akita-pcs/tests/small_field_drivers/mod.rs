//! Typed prove→verify drivers for small-field configs (`ExtField != Field`).
//!
//! The four small-field matrix cells (dense/one-hot × direct/precommitted)
//! previously each inlined a complete setup → commit → prove → serialize →
//! verify flow inside `small_field_test!`, so the macro carried four near-copies
//! of the same ~80 lines.
//!
//! Everything that is actually shared lives here as generic functions over
//! `Cfg`. What genuinely differs between cells — how the polynomial is built
//! and how its opening is computed from an independent oracle — stays at the
//! call site, which is the only part worth reading per cell.

use crate::common::load_workspace_scheme;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend, MultilinearPolynomial, UniformProverStack};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::{
    AkitaBatchedProof, BasisMode, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
};

use akita_prover::SelectedProverOpeningData;
use akita_serialization::Valid;
use akita_types::FpExtEncoding;
use jolt_field::{CanonicalBytes, CanonicalEncoding, ExtField, Field, PseudoMersenne, Ring, Zero};
use jolt_field::{Fold, Unreduced, WithCommitAccumulator};

/// Artifacts retained from a successful single-group roundtrip so focused
/// protocol tests can mutate the exact proof that was already produced.
pub(super) struct SingleGroupRoundtrip<Cfg: CommitmentConfig> {
    pub(super) scheme: AkitaCommitmentScheme<Cfg>,
    pub(super) proof: AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    pub(super) verifier_setup: akita_types::AkitaVerifierSetup<Cfg::Field>,
    pub(super) selection: akita_types::OpeningScheduleSelection,
    pub(super) commitment: akita_types::CommittedGroup<Cfg::Field>,
    pub(super) point: Vec<Cfg::ExtField>,
    pub(super) expected: Cfg::ExtField,
}

/// Single committed group, no precommits: commit `poly`, prove its opening at
/// `point`, round-trip the proof through serialization, and verify `expected`.
pub(super) fn single_group_roundtrip<Cfg>(
    nv: usize,
    poly: &MultilinearPolynomial<Cfg::Field, u8>,
    point: Vec<Cfg::ExtField>,
    expected: Cfg::ExtField,
    label: &[u8],
    what: &str,
) -> SingleGroupRoundtrip<Cfg>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding
        + CanonicalBytes
        + CanonicalEncoding
        + Unreduced
        + Field
        + Ring
        + Field
        + PseudoMersenne
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static,
    Cfg::ExtField: ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + AkitaSerialize,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field>,
{
    let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");
    let setup = scheme.setup_prover(nv, 1).expect("setup");
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
    let stack =
        UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = scheme
        .commit::<_, _>(
            &setup,
            std::slice::from_ref(poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");
    let poly_refs = [poly];

    let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.clone(),
        vec![Cfg::ExtField::zero()],
        commitment.clone(),
    )
    .expect("prover group")])
    .expect("prover claims");
    let prover_data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        prover_claims,
        vec![hint],
        vec![&poly_refs[..]],
        scheme.schedules(),
    )
    .expect("prover data");
    let selection = prover_data.selection();

    let mut pt = AkitaTranscript::<Cfg::Field>::new(label);
    let proof = scheme
        .batched_prove::<_, _, _>(&setup, prover_data, &stack, &mut pt, BasisMode::Lagrange)
        .expect("prove");

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).expect("serialize");
    let decoded = AkitaBatchedProof::<Cfg::Field, Cfg::ExtField>::deserialize_uncompressed(
        &bytes[..],
        &shape,
    )
    .expect("deserialize");

    let verify_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.clone(),
        vec![expected],
        &commitment,
    )
    .expect("verifier group")])
    .expect("verifier claims");
    let mut vt = AkitaTranscript::<Cfg::Field>::new(label);
    scheme
        .batched_verify(
            &decoded,
            &verifier_setup,
            &mut vt,
            GroupBatchStatement::new(selection, verify_claims).expect("statement"),
            BasisMode::Lagrange,
        )
        .unwrap_or_else(|e| panic!("{what} nv={nv}: {e:?}"));

    SingleGroupRoundtrip {
        scheme,
        proof: decoded,
        verifier_setup,
        selection,
        commitment,
        point,
        expected,
    }
}

/// Shared tail of the two-group (precommit + final) cells: serialize the
/// proof, round-trip it, and verify both group openings.
///
/// The *head* of those cells stays at the call site on purpose. Committing the
/// pre-group, resolving the combined schedule, and deriving the final group's
/// ring dimension are interleaved — the final polynomial cannot be built until
/// the pre-commitment exists — so folding that into a driver would mean
/// threading polynomial-construction closures through it. This split keeps the
/// genuinely shared part shared without inventing that indirection.
#[allow(clippy::too_many_arguments)]
pub(super) fn two_group_verify_roundtrip<Cfg>(
    scheme: &AkitaCommitmentScheme<Cfg>,
    proof: &AkitaBatchedProof<Cfg::Field, Cfg::ExtField>,
    verifier_setup: &akita_types::AkitaVerifierSetup<Cfg::Field>,
    selection: akita_types::OpeningScheduleSelection,
    pre: (
        &akita_types::CommittedGroup<Cfg::Field>,
        &[Cfg::ExtField],
        Cfg::ExtField,
    ),
    fin: (
        &akita_types::CommittedGroup<Cfg::Field>,
        &[Cfg::ExtField],
        Cfg::ExtField,
    ),
    label: &[u8],
    what: &str,
) where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding
        + CanonicalBytes
        + CanonicalEncoding
        + Unreduced
        + Field
        + Ring
        + Field
        + PseudoMersenne
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static,
    Cfg::ExtField: ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Valid
        + AkitaDeserialize<Context = ()>
        + AkitaSerialize,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field>,
{
    let (pre_commitment, pre_point, pre_opening) = pre;
    let (final_commitment, final_point, final_opening) = fin;

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).expect("serialize");
    let decoded = AkitaBatchedProof::<Cfg::Field, Cfg::ExtField>::deserialize_uncompressed(
        &bytes[..],
        &shape,
    )
    .expect("deserialize");

    let verify_claims = OpeningClaims::from_groups(vec![
        PolynomialGroupClaims::new(pre_point.to_vec(), vec![pre_opening], pre_commitment)
            .expect("pre verifier group"),
        PolynomialGroupClaims::new(final_point.to_vec(), vec![final_opening], final_commitment)
            .expect("final verifier group"),
    ])
    .expect("verifier claims");

    let mut vt = AkitaTranscript::<Cfg::Field>::new(label);
    scheme
        .batched_verify(
            &decoded,
            verifier_setup,
            &mut vt,
            GroupBatchStatement::new(selection, verify_claims).expect("statement"),
            BasisMode::Lagrange,
        )
        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
}
