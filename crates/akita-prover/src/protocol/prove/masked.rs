//! The zero-knowledge opening's prover side up to the cut
//! (`specs/whiteout-akita.md` §4.5–4.6).
//!
//! Levels `0..=cut_level` run the ordinary fold prover over jolt-zk's masked
//! channel: every witness-dependent message goes out as m̂ = m + p with the
//! next committed pads, and the statement claims and derived claims bind
//! nothing. The opening stops after the cut level's Stage 2
//! (`SuffixEnd::Cut`) and hands back the [`ResidualOpening`] and Akita's
//! auxiliary values.

use super::registry::SuccessorHandoff;
use super::*;
use crate::backend::{
    FoldExecutionRequirements, ProofScope, ResidualExportKernel, ResidualWitness,
    ValidatedRecursiveWitnessCommitPlan,
};
use crate::SelectedProverOpeningData;
use akita_config::TrustedScheduleCatalog;
use akita_params::zk::ZkOpeningSchedule;
use akita_types::{check_range_leaf, AkitaSetupDescriptor, MaskedOpeningClaims, ProverGrinding};
use jolt_algebra::AdditiveGroup;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_transcript::{Channel, ProverTranscript};
use jolt_zk::masking::{CommittedPads, ValuesDomain};
use zeroize::Zeroizing;

/// What the masked prefix hands the residual back end at the cut: the
/// opening of the residual commitment u_a, `[B | B′](t̂_a, r_a) = u_a` with
/// `A·G⁻¹(w_k) = G·t̂_a`, and the evaluation claim `w̃_k(point) = witness_eval`
/// that the residual relation's `Witness(0)` stands for
/// (`specs/whiteout-akita.md` §4.6, §4.8.1). The digit orders are
/// [`ResidualWitness`]'s.
///
/// `public_segments` and the matrix `blocks` of §4.6 are not built yet.
pub struct ResidualOpening<F, E> {
    /// The last masked level.
    pub cut_level: u32,
    /// The evaluation point ([`akita_types::residual_point`]).
    pub point: Point<HIGH_TO_LOW, E>,
    /// w̃_k(point), the cut level's witness evaluation.
    pub witness_eval: E,
    /// u_a, the raw outer commitment.
    pub residual_commitment: Vec<F>,
    /// w_k's digits.
    pub w_k: Zeroizing<Vec<i8>>,
    /// Base-two logarithm of w_k's balanced digit alphabet: the cut level's
    /// open basis, which its range check proves.
    pub w_k_log_basis: u32,
    /// The inner digits t̂_a.
    pub t_hat_a: Zeroizing<Vec<i8>>,
    /// Base-two logarithm of t̂_a's gadget basis.
    pub t_hat_log_basis: u32,
    /// The hiding randomness r_a.
    pub r_a: Zeroizing<Vec<i8>>,
    /// Base-two logarithm of r_a's alphabet
    /// (`akita_params::zk::RESIDUAL_RANDOMNESS_LOG_BASIS`).
    pub r_a_log_basis: u32,
}

/// One backend proves every masked level, so a committed successor stays
/// where it is.
struct OwnBackend;

impl<Cfg, B> SuccessorHandoff<Cfg, B> for OwnBackend
where
    Cfg: CommitmentConfig,
    B: ProverBackend<Cfg::Field, Cfg::ExtField>,
{
    fn handoff_successor(
        &mut self,
        _: &B,
        _: &B::ProofSessionHandle,
        _: usize,
        _: &CommittedGroupParams,
        _: &RingRelationInstance<Cfg::Field>,
        _: &ValidatedRecursiveWitnessCommitPlan,
        _: &NextWitnessState<Cfg::Field>,
        _: &B::WitnessHandle,
        _: &B::CommitmentMaterialHandle,
    ) -> Result<(), AkitaError> {
        Ok(())
    }
}

/// Proves levels `0..=cut_level` of a zero-knowledge opening on `backend`,
/// sending every witness-dependent message through `pads`' masked channel on
/// `transcript`, and stops after the cut level's Stage 2.
///
/// `openings` are the statement claims as affine forms over the pads, in
/// group order; each must decode under `pads` to the claim in `opening`. The
/// result is the cut's [`ResidualOpening`] and Akita's auxiliary values, in
/// slot order: the Stage-1 leaf check's one auxiliary slot per masked level.
///
/// # Errors
///
/// [`AkitaError::InvalidSetup`] or [`AkitaError::UnsupportedSchedule`] if the
/// selected schedule is not an admitted zero-knowledge schedule,
/// [`AkitaError::InvalidInput`] if `openings` do not decode to the claims,
/// [`AkitaError::FoldOverflow`] if a one-shot fold response overflows, and
/// the fold prover's errors otherwise. A panic is returned as
/// [`AkitaError::Internal`].
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn prove_masked_prefix<Cfg, B, S, O>(
    expanded: &AkitaSetupDescriptor,
    schedules: &TrustedScheduleCatalog<Cfg>,
    backend: &B,
    prefixes: &SetupPrefixProverRegistry<Cfg::Field, B::CommitmentHandle>,
    opening: SelectedProverOpeningData<'_, Cfg::ExtField, B::CommitmentHandle, Cfg::Field>,
    openings: &MaskedOpeningClaims<Cfg::ExtField>,
    pads: &mut CommittedPads<Cfg::ExtField, O>,
    transcript: &mut ProverTranscript<S>,
    basis: BasisMode,
) -> Result<
    (
        ResidualOpening<Cfg::Field, Cfg::ExtField>,
        Vec<Cfg::ExtField>,
    ),
    AkitaError,
>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding + AkitaSerialize + Unreduced + PseudoMersenne + Ring + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field> + AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + ExtField<Cfg::Field>
        + Unreduced
        + Fold
        + Ring
        + MulBaseUnreduced<Cfg::Field>
        + AkitaSerialize
        + CanonicalEncoding
        + 'static,
    B: ProverBackend<Cfg::Field, Cfg::ExtField>
        + ResidualExportKernel<Cfg::Field, Cfg::ExtField>
        + 'static,
    B::CommitmentHandle: crate::CommitmentHandleMetadata,
    S: Sponge,
{
    catch_prover_panic(move || {
        let (resolved, claims, grinding_plan, descriptor_bytes) =
            root::resolve_root::<Cfg, B::CommitmentHandle>(expanded, schedules, opening, basis)?;
        let schedule = resolved.schedule();
        let layout = claims.opening_layout();
        let cut_level =
            ZkOpeningSchedule::admit(schedule, Cfg::committed_source_contract()?, Cfg::EXT_DEGREE)?
                .cut_level();
        let decoded = openings
            .iter()
            .map(|form| form.evaluate(pads.pads(), &[]))
            .collect::<Result<Vec<_>, _>>()?;
        if decoded != claims.opening_claims().flat_evaluations() {
            return Err(AkitaError::InvalidInput(
                "masked statement claims do not decode to the opening's claims".into(),
            ));
        }

        let scope = ProofScope::admitted(
            backend,
            backend.prepare_executor(expanded, schedules, schedule, layout)?,
        );
        for level in 0..=cut_level as usize {
            backend.begin_fold(
                scope.session(),
                &FoldExecutionRequirements::new(schedule, layout, level)?,
            )?;
        }
        let mut channel = pads.channel(transcript);
        channel.site(
            akita_params::ProtocolSiteId {
                family: akita_params::transcript_site::SITE_FAMILY_ROOT_STATEMENT,
                ..akita_params::ProtocolSiteId::default()
            }
            .into(),
        );
        channel.public_bytes(&descriptor_bytes);
        let mut grinding = ProverGrinding::new(&mut channel, &grinding_plan);
        let mut level = root::prove_root::<Cfg, B, S, _>(
            expanded,
            prefixes,
            backend,
            claims.opening_claims(),
            layout,
            &claims.groups,
            schedule,
            scope.session(),
            &mut grinding,
            basis,
            &mut OwnBackend,
        )?;
        let mut auxiliary = ValuesDomain::default();
        for index in 0..cut_level as usize {
            level.lower_range_leaf(&mut auxiliary)?;
            level = suffix::prove_recursive_step::<Cfg, B, S, _>(
                expanded,
                prefixes,
                backend,
                &mut grinding,
                level.next_state,
                schedule,
                scope.session(),
                index,
                &mut OwnBackend,
            )?;
        }
        level.lower_range_leaf(&mut auxiliary)?;
        grinding.finish_through(cut_level)?;

        let SuffixProverState {
            public,
            witness_handle,
            commitment_material,
        } = level.next_state;
        let cut_params = match (cut_level as usize).checked_sub(1) {
            None => &schedule.root.params,
            Some(index) => {
                &schedule
                    .recursive_folds
                    .get(index)
                    .ok_or_else(|| AkitaError::Internal("the cut level is not scheduled".into()))?
                    .params
            }
        };
        let successor = schedule
            .recursive_folds
            .get(cut_level as usize)
            .ok_or_else(|| AkitaError::Internal("the cut has no successor level".into()))?;
        let ResidualWitness { w_k, t_hat, r_a } =
            backend.export_residual(&witness_handle, &commitment_material, &successor.params)?;
        backend.release_witness_handle(witness_handle)?;
        drop(commitment_material);
        scope.finish()?;
        let NextWitnessState::OuterPayload(residual_commitment) = public.binding else {
            return Err(AkitaError::Internal(
                "the cut level's successor is not a recursive commitment".into(),
            ));
        };
        let residual = &successor.params.own_group().profile;
        Ok((
            ResidualOpening {
                cut_level,
                point: akita_types::residual_point(public.sumcheck_challenges, w_k.len()),
                witness_eval: public.opening,
                residual_commitment: residual_commitment.coeffs().to_vec(),
                w_k,
                w_k_log_basis: cut_params.open().digits.log_basis,
                t_hat_a: t_hat,
                t_hat_log_basis: residual.outer.digits.log_basis,
                r_a,
                r_a_log_basis: successor.params.own_group().opening.log_basis_open,
            },
            auxiliary.auxiliary,
        ))
    })
}

impl<F: Field, E: Field, M, W> ProveLevelOutput<F, E, M, W> {
    /// Runs this level's Stage-1 leaf check in `values`, which records its
    /// auxiliary value.
    fn lower_range_leaf(&self, values: &mut ValuesDomain<E>) -> Result<(), AkitaError> {
        let RangeLeafClaim {
            leaf,
            range_image,
            claim,
        } = self.range_leaf.as_ref().ok_or_else(|| {
            AkitaError::Internal("a masked level has no Stage-1 leaf check".into())
        })?;
        check_range_leaf(values, leaf, *range_image, *claim)?;
        Ok(())
    }
}
