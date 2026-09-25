//! The only place that spells out Akita's end-to-end trait bounds.
//!
//! Every bounded public operation used by the harness is a method of
//! [`PcsOps`], blanket-implemented for each config that satisfies the bounds
//! of the production APIs. Generic harness code then needs only `Cfg: PcsOps`.

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::{
    AkitaProverSetup, CommitOutput, CommitmentHandle, CpuBackend, CpuSource, DensePoly,
    GroupContext, OneHotPoly,
};
use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ProverBackend, SelectedProverOpeningData};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::{
    AkitaVerifierSetup, BasisMode, CommittedGroup, FoldSchedule, FpExtEncoding,
    GroupBatchStatement, OpeningClaims, OpeningClaimsLayout, OpeningScheduleSelection,
};
use jolt_field::{
    AdditiveGroup, CanonicalBytes, ExtField, Field, Fold, MulBaseUnreduced, PseudoMersenne, Ring,
    Unreduced, WithCommitAccumulator,
};
use std::sync::Arc;

pub type Handle<Cfg> =
    CommitmentHandle<<Cfg as CommitmentConfig>::Field, <Cfg as CommitmentConfig>::ExtField, Cfg>;
pub type Output<Cfg> =
    CommitOutput<<Cfg as CommitmentConfig>::Field, <Cfg as CommitmentConfig>::ExtField, Cfg>;
pub type Claims<'a, Cfg> = OpeningClaims<
    'a,
    <Cfg as CommitmentConfig>::ExtField,
    CommittedGroup<<Cfg as CommitmentConfig>::Field>,
>;
/// One group's `(point, evaluations, commitment)` in a verifier statement.
pub type ClaimRow<'a, Cfg> = (
    Vec<<Cfg as CommitmentConfig>::ExtField>,
    Vec<<Cfg as CommitmentConfig>::ExtField>,
    &'a CommittedGroup<<Cfg as CommitmentConfig>::Field>,
);
pub type Opening<'a, Cfg> = SelectedProverOpeningData<
    'a,
    <Cfg as CommitmentConfig>::ExtField,
    Handle<Cfg>,
    <Cfg as CommitmentConfig>::Field,
>;
pub type Statement<'a, Cfg> =
    GroupBatchStatement<'a, <Cfg as CommitmentConfig>::ExtField, <Cfg as CommitmentConfig>::Field>;

/// A prepared, reusable proof input: selection and layout are public.
#[derive(Clone)]
pub struct Proved {
    pub proof: Vec<u8>,
    pub selection: OpeningScheduleSelection,
    pub layout: OpeningClaimsLayout,
}

pub trait PcsOps: CommitmentConfig {
    fn load_scheme(bytes: &[u8]) -> Result<AkitaCommitmentScheme<Self>, AkitaError>;

    fn schedules(scheme: &AkitaCommitmentScheme<Self>) -> &TrustedScheduleCatalog<Self>;

    fn encode_commitment(commitment: &CommittedGroup<Self::Field>) -> Vec<u8>;

    fn setup(
        scheme: &AkitaCommitmentScheme<Self>,
        max_num_vars: usize,
        max_num_polys: usize,
    ) -> Result<AkitaProverSetup<Self::Field>, AkitaError>;

    fn backend(
        scheme: &AkitaCommitmentScheme<Self>,
        setup: &AkitaProverSetup<Self::Field>,
    ) -> Result<CpuBackend<Self>, AkitaError>;

    fn verifier_setup(
        scheme: &AkitaCommitmentScheme<Self>,
        setup: &AkitaProverSetup<Self::Field>,
    ) -> Result<AkitaVerifierSetup<Self::Field>, AkitaError>;

    fn narrowed_verifier_setup(
        scheme: &AkitaCommitmentScheme<Self>,
        setup: &AkitaProverSetup<Self::Field>,
        schedule: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<AkitaVerifierSetup<Self::Field>, AkitaError>;

    fn commit_dense(
        backend: &CpuBackend<Self>,
        polys: Vec<DensePoly<Self::Field>>,
        context: GroupContext<'_>,
    ) -> Result<Output<Self>, AkitaError>;

    fn commit_onehot(
        backend: &CpuBackend<Self>,
        polys: Vec<OneHotPoly<Self::Field, u8>>,
        context: GroupContext<'_>,
    ) -> Result<Output<Self>, AkitaError>;

    fn select<'a>(
        claims: Claims<'a, Self>,
        handles: Vec<Handle<Self>>,
        scheme: &AkitaCommitmentScheme<Self>,
    ) -> Result<Opening<'a, Self>, AkitaError>;

    fn prove(
        scheme: &AkitaCommitmentScheme<Self>,
        setup: &AkitaProverSetup<Self::Field>,
        opening: Opening<'_, Self>,
        backend: &CpuBackend<Self>,
        session: &[u8],
        basis: BasisMode,
    ) -> Result<Proved, AkitaError>;

    fn verify(
        scheme: &AkitaCommitmentScheme<Self>,
        proof: &[u8],
        setup: &AkitaVerifierSetup<Self::Field>,
        session: &[u8],
        statement: Statement<'_, Self>,
        basis: BasisMode,
    ) -> Result<(), AkitaError>;
}

impl<Cfg> PcsOps for Cfg
where
    Cfg: CommitmentConfig + 'static,
    Cfg::Field: Field
        + CanonicalBytes
        + Unreduced
        + PseudoMersenne
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + WithCommitAccumulator
        + Ring
        + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Ring
        + Unreduced
        + Fold
        + AkitaSerialize
        + MulBaseUnreduced<Cfg::Field>
        + 'static,
    CpuBackend<Cfg>: ProverBackend<Cfg::Field, Cfg::ExtField, CommitmentHandle = Handle<Cfg>>,
    DensePoly<Cfg::Field>: CpuSource<Cfg::Field, Cfg::ExtField, Cfg>,
    OneHotPoly<Cfg::Field, u8>: CpuSource<Cfg::Field, Cfg::ExtField, Cfg>,
{
    fn load_scheme(bytes: &[u8]) -> Result<AkitaCommitmentScheme<Self>, AkitaError> {
        AkitaCommitmentScheme::<Self>::from_schedule_artifact(bytes)
    }

    fn schedules(scheme: &AkitaCommitmentScheme<Self>) -> &TrustedScheduleCatalog<Self> {
        scheme.schedules()
    }

    fn encode_commitment(commitment: &CommittedGroup<Self::Field>) -> Vec<u8> {
        let mut bytes = Vec::new();
        commitment
            .serialize_compressed(&mut bytes)
            .expect("commitments encode");
        bytes
    }

    fn setup(
        scheme: &AkitaCommitmentScheme<Self>,
        max_num_vars: usize,
        max_num_polys: usize,
    ) -> Result<AkitaProverSetup<Self::Field>, AkitaError> {
        scheme.setup_prover(max_num_vars, max_num_polys)
    }

    fn backend(
        scheme: &AkitaCommitmentScheme<Self>,
        setup: &AkitaProverSetup<Self::Field>,
    ) -> Result<CpuBackend<Self>, AkitaError> {
        CpuBackend::<Self>::new(Arc::clone(&setup.expanded), scheme.schedules())
    }

    fn verifier_setup(
        scheme: &AkitaCommitmentScheme<Self>,
        setup: &AkitaProverSetup<Self::Field>,
    ) -> Result<AkitaVerifierSetup<Self::Field>, AkitaError> {
        scheme.setup_verifier(setup)
    }

    fn narrowed_verifier_setup(
        scheme: &AkitaCommitmentScheme<Self>,
        setup: &AkitaProverSetup<Self::Field>,
        schedule: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<AkitaVerifierSetup<Self::Field>, AkitaError> {
        scheme.setup_verifier_for_schedule(setup, schedule, layout)
    }

    fn commit_dense(
        backend: &CpuBackend<Self>,
        polys: Vec<DensePoly<Self::Field>>,
        context: GroupContext<'_>,
    ) -> Result<Output<Self>, AkitaError> {
        let source = backend.import_source(polys)?;
        backend.commit(&source, context)
    }

    fn commit_onehot(
        backend: &CpuBackend<Self>,
        polys: Vec<OneHotPoly<Self::Field, u8>>,
        context: GroupContext<'_>,
    ) -> Result<Output<Self>, AkitaError> {
        let source = backend.import_source(polys)?;
        backend.commit(&source, context)
    }

    fn select<'a>(
        claims: Claims<'a, Self>,
        handles: Vec<Handle<Self>>,
        scheme: &AkitaCommitmentScheme<Self>,
    ) -> Result<Opening<'a, Self>, AkitaError> {
        SelectedProverOpeningData::from_committed_claims::<Self>(
            claims,
            handles,
            scheme.schedules(),
        )
    }

    fn prove(
        scheme: &AkitaCommitmentScheme<Self>,
        setup: &AkitaProverSetup<Self::Field>,
        opening: Opening<'_, Self>,
        backend: &CpuBackend<Self>,
        session: &[u8],
        basis: BasisMode,
    ) -> Result<Proved, AkitaError> {
        let selection = opening.selection();
        let layout = opening.opening_layout().clone();
        let proof = scheme.batched_prove(setup, opening, backend, session, basis)?;
        Ok(Proved {
            proof,
            selection,
            layout,
        })
    }

    fn verify(
        scheme: &AkitaCommitmentScheme<Self>,
        proof: &[u8],
        setup: &AkitaVerifierSetup<Self::Field>,
        session: &[u8],
        statement: Statement<'_, Self>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        scheme.batched_verify(proof, setup, session, statement, basis)
    }
}
