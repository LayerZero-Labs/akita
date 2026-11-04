//! The zero-knowledge opening's verifier side up to the cut
//! (`specs/whiteout-akita.md` §4.5–4.6): the fold verifier replayed over a
//! masked channel through levels `0..=cut_level`, stopping after the cut
//! level's Stage 2.

use super::root::verify_root;
use super::suffix::{verify_suffix_level, SuffixVerifierState};
use super::verify::OpeningPlan;
use super::NextWitness;
use crate::masked::{CompressedCut, MaskedPrefix, MaskedReplay, ResidualClaim};
use crate::AkitaVerifier;
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::zk::{AdmittedZkSchedule, ResidualGeometry};
use akita_params::BasisMode;
use akita_serialization::{AkitaSerialize, Valid};
use akita_transcript::{PrivateProofChannel, VerifierChannel};
use akita_types::{
    CommittedGroup, CountingGrinding, FoldResponseReplay, FpExtEncoding, GroupBatchStatement,
    MaskedOpeningClaims, MaskedVerifierChannel, OpeningClaims, VerifierGrinding,
    VerifierGrindingChannel,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field, JoltField, PseudoMersenne, Ring};
use jolt_zk::masking::{CommittedPadsView, ZkError};
use jolt_zk::whiteout::ImageCompression;

impl<Cfg> AkitaVerifier<Cfg>
where
    Cfg: CommitmentConfig,
    Cfg::Field: Field
        + CanonicalEncoding
        + JoltField
        + akita_serialization::AkitaSerialize
        + PseudoMersenne
        + Valid,
    Cfg::ExtField: FpExtEncoding<Cfg::Field> + ExtField<Cfg::Field> + Ring + AkitaSerialize + Valid,
{
    /// Replays levels `0..=cut_level` of a zero-knowledge opening over the
    /// masked channel of `pads` on `state`, and returns the masked
    /// prefix: its stage checks, to lower into residual rows once the
    /// auxiliary block has arrived, and the [`ResidualClaim`] at the cut.
    ///
    /// `statement` supplies the commitments, points and schedule selection;
    /// its evaluations are not read. The claims are `openings`, affine forms
    /// over the pads in group order. The caller must already have absorbed
    /// the masked statement claims those forms are built from into
    /// `state`, before this call: the replay binds none of them.
    ///
    /// The cut level sends u_a as its compression under `compression`, the
    /// response proof's chain (`specs/whiteout-akita.md` A47); the
    /// [`ResidualClaim`] holds that payload.
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidSetup`] or [`AkitaError::UnsupportedSchedule`] if
    /// the selected schedule is not an admitted zero-knowledge schedule
    /// (`ZkOpeningSchedule::admit`), and the fold verifier's errors otherwise.
    #[tracing::instrument(skip_all, name = "AkitaVerifier::verify_masked_prefix")]
    pub fn verify_masked_prefix<C>(
        &self,
        state: &mut VerifierChannel<'_>,
        pads: &mut CommittedPadsView<C>,
        statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
        openings: MaskedOpeningClaims<Cfg::ExtField>,
        compression: &dyn ImageCompression<Cfg::Field>,
        basis: BasisMode,
    ) -> Result<MaskedPrefix<Cfg::ExtField, Cfg::Field>, AkitaError> {
        let selection = statement.selection();
        let claims = statement.into_claims();
        let (opening, cut) =
            self.plan_masked_opening(selection, &claims, &openings, compression, basis)?;
        let result = (|| {
            akita_transcript::bind_instance(state, &opening.descriptor_bytes)?;
            let mut channel = MaskedVerifierChannel::new(state, pads);
            let mut grinding = VerifierGrinding::new(&mut channel, &opening.grinding_plan);
            let prefix =
                self.replay_masked_levels(&mut grinding, &opening, &claims, openings, basis, cut)?;
            grinding.finish_through(cut.level)?;
            Ok(prefix)
        })();
        if result.is_err() {
            let _ = akita_transcript::reject(state);
        }
        result
    }

    /// The pad-layout replay of [`verify_masked_prefix`](Self::verify_masked_prefix)
    /// over `view`, a counting view of the pad schedule
    /// (`CommittedPadsView::counting`). Afterwards `view.used()` counts the
    /// masked messages' pads, and [`MaskedPrefix::shape`] the auxiliary slots
    /// and operands.
    ///
    /// `state` must read zero bytes, long enough for the prefix; never a
    /// proof's bytes. The layout is
    /// a function of the public schedule, and a proof can end early or fail to
    /// decode.
    ///
    /// # Errors
    ///
    /// As [`verify_masked_prefix`](Self::verify_masked_prefix), except that
    /// no proof-of-work nonce is read or checked.
    #[tracing::instrument(skip_all, name = "AkitaVerifier::count_masked_prefix")]
    pub fn count_masked_prefix(
        &self,
        state: &mut VerifierChannel<'_>,
        view: &mut CommittedPadsView<()>,
        statement: GroupBatchStatement<'_, Cfg::ExtField, Cfg::Field>,
        openings: MaskedOpeningClaims<Cfg::ExtField>,
        compression: &dyn ImageCompression<Cfg::Field>,
        basis: BasisMode,
    ) -> Result<MaskedPrefix<Cfg::ExtField, Cfg::Field>, AkitaError> {
        let selection = statement.selection();
        let claims = statement.into_claims();
        let (opening, cut) =
            self.plan_masked_opening(selection, &claims, &openings, compression, basis)?;
        akita_transcript::bind_instance(state, &opening.descriptor_bytes)?;
        let mut channel = MaskedVerifierChannel::new(state, view);
        let mut grinding = CountingGrinding::new(&mut channel, &opening.grinding_plan);
        let prefix =
            self.replay_masked_levels(&mut grinding, &opening, &claims, openings, basis, cut)?;
        grinding.finish_through(cut.level)?;
        Ok(prefix)
    }

    fn plan_masked_opening(
        &self,
        selection: akita_params::OpeningScheduleSelection,
        claims: &OpeningClaims<'_, Cfg::ExtField, &CommittedGroup<Cfg::Field>>,
        openings: &MaskedOpeningClaims<Cfg::ExtField>,
        compression: &dyn ImageCompression<Cfg::Field>,
        basis: BasisMode,
    ) -> Result<(OpeningPlan<'_>, CompressedCut), AkitaError> {
        let opening = self.plan_opening(selection, claims, basis, None)?;
        let admitted = AdmittedZkSchedule::new(
            opening.schedule,
            Cfg::committed_source_contract()?,
            Cfg::EXT_DEGREE,
        )?;
        if openings.len() != opening.opening_batch.num_total_polynomials() {
            return Err(AkitaError::InvalidInput(
                "masked statement claims differ from the opening layout".into(),
            ));
        }
        let image_len =
            ResidualGeometry::new(admitted, Cfg::Field::MODULUS_BITS)?.commitment_len()?;
        let cut = CompressedCut {
            level: admitted.zk().cut_level(),
            image_len,
            payload_len: compression.payload_len(image_len).map_err(ZkError::from)?,
        };
        Ok((opening, cut))
    }

    fn replay_masked_levels<'proof, G>(
        &self,
        grinding: &mut G,
        opening: &OpeningPlan<'_>,
        claims: &OpeningClaims<'_, Cfg::ExtField, &CommittedGroup<Cfg::Field>>,
        openings: MaskedOpeningClaims<Cfg::ExtField>,
        basis: BasisMode,
        cut: CompressedCut,
    ) -> Result<MaskedPrefix<Cfg::ExtField, Cfg::Field>, AkitaError>
    where
        G: FoldResponseReplay<
            State: PrivateProofChannel<Cfg::Field> + VerifierGrindingChannel<'proof>,
        >,
    {
        let setup = &self.setup;
        let schedule = opening.schedule;
        let raw_claims = opening.raw_claims(claims)?;
        let mut replay = MaskedReplay::new(grinding, cut);
        // A masked opening checks its setup contribution in full at every level.
        let mut setup_policy = crate::SetupPolicy::Check(None);
        let root_span = tracing::info_span!("masked_prefix_level", level = 0).entered();
        let root = verify_root::<Cfg::Field, Cfg::ExtField, _>(
            setup,
            &mut replay,
            &raw_claims,
            openings,
            &opening.opening_batch,
            basis,
            &schedule.root_fold().params,
            schedule.recursive_folds.first(),
            &schedule.terminal,
            &mut setup_policy,
        )?;
        drop(root_span);
        let mut state =
            SuffixVerifierState::from_root(root, schedule.root_fold().output_witness_len);
        for offset in 0..cut.level as usize {
            let _span = tracing::info_span!("masked_prefix_level", level = offset + 1).entered();
            state = verify_suffix_level(
                setup,
                &mut replay,
                schedule,
                offset,
                state,
                &mut setup_policy,
            )?;
        }
        // Admission puts a recursive fold after the cut, so its successor
        // message is the compressed u_a payload, never a terminal state.
        let NextWitness::OuterPayload(payload) = state.witness else {
            return Err(AkitaError::UnsupportedSchedule(
                "the zero-knowledge cut hands its witness to the terminal".into(),
            ));
        };
        let claim = ResidualClaim::new(
            cut.level,
            state.opening_point,
            state.witness_len,
            state.opening,
            payload.coeffs().to_vec(),
        );
        Ok(MaskedPrefix::new(replay, claim))
    }
}
