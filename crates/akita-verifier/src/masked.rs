//! The masked prefix of a zero-knowledge opening (`specs/whiteout-akita.md`
//! §4.4–4.6).
//!
//! Levels `0..=cut_level` run the ordinary fold verifier over a masked
//! channel. Every witness-dependent message arrives as m̂ = m + p, so the
//! replay holds each claim as an affine form over the pads and records the
//! two stage checks of each level ([`check_range_leaf`],
//! [`check_stage2_oracle`]) instead of evaluating them. The checks lower into
//! residual rows only once the auxiliary block has arrived, after the cut
//! (W6's transcript order), and the same lowering serves the pad layout
//! ([`OperandShapeDomain`]) and the prover's auxiliary values
//! ([`ValuesDomain`]).

use crate::claims::{clear_sumcheck, ClaimReplay, SumcheckFormat, SumcheckInstance};
use akita_error::AkitaError;
use akita_transcript::{ChallengeCodec, PrivateProofChannel, ProtocolSiteId};
use akita_types::{
    check_range_leaf, check_stage2_oracle, grinding_sumcheck_body_site, FoldResponseReplay,
    GrindingReplay, RangeLeaf, Stage2OracleWeights, VerifierGrindingChannel, FIELD_DIGEST_BYTES,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use jolt_poly::{Point, HIGH_TO_LOW, LOW_TO_HIGH};
use jolt_sumcheck::{DegreeSchedule, MaskedRound};
use jolt_transcript::{ClaimDomain, PadSlots};
use jolt_zk::masking::{
    sumcheck_rows, AffineRow, OperandShapeDomain, PadIndex, PadLayout, ResidualRelation,
    ResidualRows, ResidualVar, RoundFormat, RowsDomain, ValuesDomain, ZkError,
};

/// One stage check of a masked level, on claims held as affine forms over
/// the pads.
#[derive(Clone, Debug, PartialEq, Eq)]
enum MaskedCheck<F> {
    /// [`check_range_leaf`] on the range image and Stage 1's running claim.
    RangeLeaf {
        leaf: RangeLeaf<F>,
        range_image: AffineRow<F>,
        claim: AffineRow<F>,
    },
    /// [`check_stage2_oracle`] on the witness evaluation and Stage 2's
    /// running claim.
    Stage2Oracle {
        weights: Stage2OracleWeights<F>,
        witness_eval: AffineRow<F>,
        claim: AffineRow<F>,
    },
}

/// The fold verifier's replay at masked levels: a grinding replay over a
/// masked channel (the verifier's, or the counting view's), and the stage
/// checks recorded so far.
pub(crate) struct MaskedReplay<'g, G, F> {
    grinding: &'g mut G,
    checks: Vec<MaskedCheck<F>>,
    cut: CompressedCut,
    #[cfg(feature = "test-support")]
    batch_coefficients: Vec<F>,
}

impl<'g, G, F> MaskedReplay<'g, G, F> {
    pub(crate) fn new(grinding: &'g mut G, cut: CompressedCut) -> Self {
        Self {
            grinding,
            checks: Vec::new(),
            cut,
            #[cfg(feature = "test-support")]
            batch_coefficients: Vec::new(),
        }
    }
}

/// The cut level's successor message: u_a, an image of `image_len`
/// coefficients, sent as its `payload_len`-coefficient compression
/// (`specs/whiteout-akita.md` A47).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompressedCut {
    pub(crate) level: u32,
    pub(crate) image_len: usize,
    pub(crate) payload_len: usize,
}

/// The masked value received in `slots`: one value of the pad field, one pad.
fn masked_claim<F: Field>(masked: F, slots: PadSlots) -> Result<AffineRow<F>, AkitaError> {
    if slots.len != 1 {
        return Err(AkitaError::InvalidSetup(
            "a masked claim takes exactly one pad".into(),
        ));
    }
    Ok(AffineRow::masked(masked, PadIndex(slots.start)))
}

fn clear_only<T>() -> Result<T, AkitaError> {
    Err(AkitaError::UnsupportedSchedule(
        "a masked level runs only the base-field, basis-8, ℓ∞ stage checks".into(),
    ))
}

impl<G: GrindingReplay, F> GrindingReplay for MaskedReplay<'_, G, F> {
    type State = G::State;

    fn state_mut(&mut self) -> &mut G::State {
        self.grinding.state_mut()
    }

    fn poison(&mut self) {
        self.grinding.poison();
    }

    fn check_public(&mut self, holds: bool) -> Result<(), AkitaError> {
        self.grinding.check_public(holds)
    }

    fn grind_query(&mut self, site: akita_params::GrindingSite) -> Result<(), AkitaError> {
        self.grinding.grind_query(site)
    }
}

impl<G: FoldResponseReplay, F> FoldResponseReplay for MaskedReplay<'_, G, F> {
    fn read_fold_response(&mut self, site: akita_params::GrindingSite) -> Result<u32, AkitaError> {
        self.grinding.read_fold_response(site)
    }

    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
    ) -> Result<(), AkitaError> {
        self.grinding
            .record_fold_challenges(level, group, coordinate_count)
    }
}

/// The masked replay: claims are affine forms over the pads, and the stage
/// checks are recorded for [`MaskedPrefix`]. Claims live in the base field
/// (`E::DEGREE = 1`, which admission requires), so each takes one pad.
impl<'proof, F, E, G> ClaimReplay<F, E> for MaskedReplay<'_, G, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + ChallengeCodec,
    G: FoldResponseReplay<State: PrivateProofChannel<F> + VerifierGrindingChannel<'proof>>,
{
    type Claim = AffineRow<E>;

    fn constant(value: E) -> AffineRow<E> {
        AffineRow::constant(value)
    }

    fn add(lhs: &AffineRow<E>, rhs: &AffineRow<E>) -> AffineRow<E> {
        lhs.add_scaled(rhs, E::one())
    }

    fn scale(value: &AffineRow<E>, scale: E) -> AffineRow<E> {
        value.scale(scale)
    }

    fn clear_path(_: E) -> Result<AffineRow<E>, AkitaError> {
        clear_only()
    }

    fn clear_values(_: &[AffineRow<E>]) -> Result<Vec<E>, AkitaError> {
        clear_only()
    }

    fn successor_payload_len(&self, level: u32, image_len: usize) -> Result<usize, AkitaError> {
        if level != self.cut.level {
            return Ok(image_len);
        }
        if image_len != self.cut.image_len {
            return Err(AkitaError::InvalidSetup(
                "the cut's successor image differs from the residual commitment".into(),
            ));
        }
        Ok(self.cut.payload_len)
    }

    #[cfg(feature = "labinius-trinomial")]
    fn clear_replay<R>(
        &mut self,
        _: impl FnOnce(&mut akita_types::VerifierGrinding<'_, '_>) -> Result<R, AkitaError>,
    ) -> Result<R, AkitaError> {
        clear_only()
    }

    fn receive_field_digest(
        &mut self,
        _: ProtocolSiteId,
    ) -> Result<[u8; FIELD_DIGEST_BYTES], AkitaError> {
        clear_only()
    }

    #[cfg(feature = "test-support")]
    fn record_batch_coefficients(&mut self, level: u32, coefficients: &[E]) {
        if level == 0 {
            self.batch_coefficients = coefficients.to_vec();
        }
    }

    /// The openings bind nothing on a masked channel: the masked messages
    /// they derive from already do. The setup-prefix evaluation stays
    /// public.
    fn bind_openings(
        &mut self,
        level: u32,
        prefix: Option<&E>,
        _openings: &[AffineRow<E>],
    ) -> Result<(), AkitaError> {
        akita_types::bind_fold_openings::<F, E, _>(self.grinding, level, prefix, &[])
    }

    /// Receives the masked rounds and lowers them with jolt-zk's
    /// `sumcheck_rows`: the omitted coefficient of each round is solved from
    /// the running claim, so the rounds add no row and the running claim is an
    /// affine form over the pads.
    fn single_sumcheck(
        &mut self,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
        instance: SumcheckInstance<E>,
        input: AffineRow<E>,
    ) -> Result<(Vec<E>, AffineRow<E>), AkitaError> {
        let mut rounds = Vec::with_capacity(instance.rounds);
        for round in 0..instance.rounds {
            let round_id = u32::try_from(round).map_err(|_| AkitaError::InvalidProof)?;
            let mut wire = vec![E::zero(); instance.degree];
            let slots = self.grinding.state_mut().exchange_private_extension_group(
                grinding_sumcheck_body_site(protocol, level, stage, instance.invocation, round_id),
                &mut wire,
            )?;
            let challenge = self.grinding.grinded_ext_challenge::<E>(
                akita_params::GrindingSite::SumcheckRound {
                    protocol,
                    level,
                    stage,
                    round: round_id,
                },
            )?;
            rounds.push(MaskedRound {
                wire,
                slots,
                challenge,
            });
        }
        let format = match instance.format {
            SumcheckFormat::Compressed => RoundFormat::Compressed,
            SumcheckFormat::EqFactored { equality_point } => RoundFormat::EqFactored {
                equality_point: Point::<LOW_TO_HIGH, E>::low_to_high(equality_point),
            },
        };
        let mut unused = ResidualRelation::new(PadLayout::new(0, 0)?, 0);
        let output = sumcheck_rows(
            &mut unused,
            input,
            &rounds,
            &DegreeSchedule::uniform(instance.rounds, instance.degree),
            &format,
        )?;
        if unused.rows().next().is_some() {
            return Err(AkitaError::Internal(
                "a one-omitted round format lowered into a row".into(),
            ));
        }
        Ok((rounds.iter().map(|round| round.challenge).collect(), output))
    }

    fn public_sumcheck(
        &mut self,
        protocol: akita_params::SumcheckProtocol,
        level: u32,
        stage: u32,
        instance: SumcheckInstance<E>,
        input: E,
    ) -> Result<(Vec<E>, E), AkitaError> {
        clear_sumcheck::<F, E, _>(self.grinding, protocol, level, stage, instance, input)
    }

    fn range_image(&mut self, level: u32, stage: u32) -> Result<AffineRow<E>, AkitaError> {
        let (masked, slots) =
            akita_types::stage1_range_image::<F, E, _>(self.grinding, level, stage, E::zero())?;
        masked_claim(masked, slots)
    }

    fn witness_eval(&mut self, level: u32) -> Result<AffineRow<E>, AkitaError> {
        let (masked, slots) =
            akita_types::stage2_w_eval::<F, E, _>(self.grinding, level, E::zero())?;
        masked_claim(masked, slots)
    }

    /// Both claims are formed from the same openings with public
    /// coefficients, so the check is on those coefficients: it needs no row.
    fn check_equal(&mut self, lhs: &AffineRow<E>, rhs: &AffineRow<E>) -> Result<(), AkitaError> {
        let difference = lhs.add_scaled(rhs, -E::one());
        if difference.constant.is_zero()
            && difference
                .terms
                .iter()
                .all(|(_, coefficient)| coefficient.is_zero())
        {
            Ok(())
        } else {
            Err(AkitaError::InvalidProof)
        }
    }

    fn check_range_leaf(
        &mut self,
        leaf: RangeLeaf<E>,
        range_image: AffineRow<E>,
        claim: AffineRow<E>,
    ) -> Result<(), AkitaError> {
        self.checks.push(MaskedCheck::RangeLeaf {
            leaf,
            range_image,
            claim,
        });
        Ok(())
    }

    fn check_stage2_oracle(
        &mut self,
        weights: Stage2OracleWeights<E>,
        witness_eval: AffineRow<E>,
        claim: AffineRow<E>,
    ) -> Result<(), AkitaError> {
        self.checks.push(MaskedCheck::Stage2Oracle {
            weights,
            witness_eval,
            claim,
        });
        Ok(())
    }
}

/// The hand-off at the cut, as one value: the residual relation's
/// `Witness(0)` is the evaluation of the w_k that `commitment` (`cm(u_a)`, as this
/// verifier received it) commits to, at `point`, and the cut row equates it
/// with the cut level's masked Stage-2 witness evaluation. The residual back
/// end opens `commitment` at `point` and nothing else, so neither can drift
/// from the row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidualClaim<F, B> {
    cut_level: u32,
    point: Point<HIGH_TO_LOW, F>,
    witness_eval: AffineRow<F>,
    commitment: Vec<B>,
}

impl<F: Field, B> ResidualClaim<F, B> {
    /// The residual relation variable this claim defines.
    pub const WITNESS: ResidualVar = ResidualVar::Witness(0);

    /// The claim of a replay that ended at `cut_level` with Stage-2
    /// challenges `challenges` (round order) on a w_k of `witness_len`
    /// digits, and received `commitment`.
    pub(crate) fn new(
        cut_level: u32,
        challenges: Vec<F>,
        witness_len: usize,
        witness_eval: AffineRow<F>,
        commitment: Vec<B>,
    ) -> Self {
        Self {
            cut_level,
            point: akita_types::residual_point(challenges, witness_len),
            witness_eval,
            commitment,
        }
    }

    /// The last masked level.
    pub fn cut_level(&self) -> u32 {
        self.cut_level
    }

    /// The evaluation point of w_k: r₂ as `HIGH_TO_LOW`, padded with leading
    /// zeros to w_k's variable count ([`akita_types::residual_point`]).
    pub fn point(&self) -> &Point<HIGH_TO_LOW, F> {
        &self.point
    }

    /// `cm(u_a)`, the compressed outer commitment to w_k this verifier
    /// received (A47); the Σ-protocol's response proof opens it to u_a.
    pub fn commitment(&self) -> &[B] {
        &self.commitment
    }

    /// The cut level's masked witness evaluation.
    pub fn witness_eval(&self) -> &AffineRow<F> {
        &self.witness_eval
    }

    /// The cut row, `Witness(0) − w̃_k(r₂) = 0`.
    fn row(&self) -> AffineRow<F> {
        AffineRow::variable(Self::WITNESS).add_scaled(&self.witness_eval, -F::one())
    }
}

/// The masked prefix of a zero-knowledge opening, after the replay of levels
/// `0..=cut_level`: the stage checks of every masked level, in transcript
/// order, and the [`ResidualClaim`] at the cut.
///
/// Each check lowers through the one formula in `akita-types`, in any
/// [`ClaimDomain`]: residual rows ([`rows`](Self::rows)), the pad layout's
/// auxiliary slots and operands ([`shape`](Self::shape)), or the prover's
/// auxiliary values ([`auxiliary_values`](Self::auxiliary_values)). Each
/// masked level lowers into three products and one auxiliary slot
/// (`specs/whiteout-akita.md` §4.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedPrefix<F, B> {
    checks: Vec<MaskedCheck<F>>,
    claim: ResidualClaim<F, B>,
    #[cfg(feature = "test-support")]
    batch_coefficients: Vec<F>,
}

impl<F: Field, B> MaskedPrefix<F, B> {
    pub(crate) fn new<G>(replay: MaskedReplay<'_, G, F>, claim: ResidualClaim<F, B>) -> Self {
        Self {
            checks: replay.checks,
            claim,
            #[cfg(feature = "test-support")]
            batch_coefficients: replay.batch_coefficients,
        }
    }

    /// The root fold's EvaluationBatch coefficients, one per opening claim,
    /// in claim order: the challenges that batch the masked openings.
    #[cfg(feature = "test-support")]
    pub fn batch_coefficients(&self) -> &[F] {
        &self.batch_coefficients
    }

    /// The hand-off at the cut.
    pub fn residual_claim(&self) -> &ResidualClaim<F, B> {
        &self.claim
    }

    /// Lowers every stage check, in transcript order, into `domain`, with
    /// each recorded claim mapped into the domain by `lift`.
    ///
    /// # Errors
    ///
    /// The domain's error.
    pub fn lower<D: ClaimDomain<F>>(
        &self,
        domain: &mut D,
        mut lift: impl FnMut(&AffineRow<F>) -> Result<D::Value, D::Error>,
    ) -> Result<(), D::Error> {
        for check in &self.checks {
            match check {
                MaskedCheck::RangeLeaf {
                    leaf,
                    range_image,
                    claim,
                } => check_range_leaf(domain, leaf, lift(range_image)?, lift(claim)?)?,
                MaskedCheck::Stage2Oracle {
                    weights,
                    witness_eval,
                    claim,
                } => check_stage2_oracle(domain, weights, lift(witness_eval)?, lift(claim)?)?,
            }
        }
        Ok(())
    }

    /// Counts the auxiliary slots and records the product operands of the
    /// stage checks in the pad layout's shape routine.
    ///
    /// # Errors
    ///
    /// As [`OperandShapeDomain`].
    pub fn shape(&self, shape: &mut OperandShapeDomain) -> Result<(), ZkError> {
        self.lower(shape, |form| {
            Ok(form.as_masked_operand().map(|operand| operand.pad))
        })
    }

    /// The auxiliary values of the stage checks, in slot order, under the
    /// committed `pads`. The prover sends them, masked, in the auxiliary
    /// block.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if a claim names a pad outside `pads` or a decoded check
    /// fails.
    pub fn auxiliary_values(&self, pads: &[F]) -> Result<Vec<F>, ZkError> {
        let mut values = ValuesDomain::default();
        self.lower(&mut values, |form| form.evaluate(pads, &[]))?;
        Ok(values.auxiliary)
    }

    /// The residual rows over `masked_auxiliary`, the received auxiliary
    /// block of the masked prefix.
    pub fn rows<'r>(&'r self, masked_auxiliary: &'r [F]) -> MaskedPrefixRows<'r, F, B> {
        MaskedPrefixRows {
            prefix: self,
            masked_auxiliary,
        }
    }
}

/// [`MaskedPrefix`]'s rows over a received auxiliary block.
pub struct MaskedPrefixRows<'r, F, B> {
    prefix: &'r MaskedPrefix<F, B>,
    masked_auxiliary: &'r [F],
}

impl<F: Field, B> ResidualRows<F> for MaskedPrefixRows<'_, F, B> {
    type Error = ZkError;

    /// Every stage check's rows, then the cut row
    /// `Witness(0) − w̃_k(r₂) = 0`, which hands the cut level's witness
    /// evaluation to the residual commitment's evaluation claim at r₂.
    fn append_rows(&self, relation: &mut ResidualRelation<F>) -> Result<(), ZkError> {
        let mut rows = RowsDomain::new(relation, self.masked_auxiliary);
        self.prefix.lower(&mut rows, |form| Ok(form.clone()))?;
        rows.finish()?;
        relation.affine.push(self.prefix.claim.row());
        Ok(())
    }
}
