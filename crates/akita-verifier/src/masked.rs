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

use crate::claims::ClaimReplay;
use akita_error::AkitaError;
use akita_types::{
    check_range_leaf, check_stage2_oracle, FoldResponseReplay, GrindingReplay,
    GrindingSumcheckVerifier, RangeLeaf, Stage2OracleWeights, FIELD_DIGEST_BYTES,
};
use jolt_algebra::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_poly::{Point, HIGH_TO_LOW, LOW_TO_HIGH};
use jolt_sumcheck::{
    receive_masked_single, DegreeSchedule, SingleFormat, SingleInstance, SumcheckError,
};
use jolt_transcript::{ClaimDomain, Fork, PadSlots, PrivateChannel, SiteId};
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
}

impl<'g, G, F> MaskedReplay<'g, G, F> {
    pub(crate) fn new(grinding: &'g mut G) -> Self {
        Self {
            grinding,
            checks: Vec::new(),
        }
    }
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

    fn grind_query(&mut self, site: akita_params::GrindingSite) -> Result<(), AkitaError> {
        self.grinding.grind_query(site)
    }
}

impl<G: FoldResponseReplay, F> FoldResponseReplay for MaskedReplay<'_, G, F> {
    type Sponge = G::Sponge;

    fn read_fold_response(
        &mut self,
        site: akita_params::GrindingSite,
    ) -> Result<Fork<G::Sponge>, AkitaError> {
        self.grinding.read_fold_response(site)
    }

    fn record_fold_challenges(
        &mut self,
        level: u32,
        group: u32,
        coordinate_count: usize,
        draws: &akita_types::FoldDrawRecord,
    ) -> Result<(), AkitaError> {
        self.grinding
            .record_fold_challenges(level, group, coordinate_count, draws)
    }
}

/// The masked replay: claims are affine forms over the pads, and the stage
/// checks are recorded for [`MaskedPrefix`]. Claims live in the base field
/// (`E::DEGREE = 1`, which admission requires), so each takes one pad.
impl<F, E, G> ClaimReplay<F, E> for MaskedReplay<'_, G, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    G: FoldResponseReplay<State: PrivateChannel<Error: Into<SumcheckError<E>>>>,
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

    fn receive_field_digest(&mut self, _: SiteId) -> Result<[u8; FIELD_DIGEST_BYTES], AkitaError> {
        clear_only()
    }

    /// The openings bind nothing on a masked channel: the masked messages
    /// they derive from already do. The setup-prefix evaluation stays
    /// public.
    fn bind_openings(
        &mut self,
        level: u32,
        prefix: Option<&E>,
        openings: &[AffineRow<E>],
    ) -> Result<(), AkitaError> {
        let unbound = akita_types::extension_slots::<E>(openings.len())?;
        akita_types::bind_fold_openings::<F, E, _>(self.grinding, level, prefix, &unbound)
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
        instance: SingleInstance<E>,
        input: AffineRow<E>,
    ) -> Result<(Vec<E>, AffineRow<E>), AkitaError> {
        let mut context =
            GrindingSumcheckVerifier::<F, E, _>::new(self.grinding, protocol, level, stage);
        let rounds = receive_masked_single::<F, E, AkitaError, _>(&instance, &mut context)?;
        let format = match instance.format {
            SingleFormat::Compressed => RoundFormat::Compressed,
            SingleFormat::EqFactored { equality_point } => {
                RoundFormat::EqFactored { equality_point }
            }
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

/// Where the masked prefix stops: the last masked level's Stage-2 output, and
/// the residual commitment the Σ-protocol opens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutState<F, B> {
    /// The last masked level.
    pub cut_level: u32,
    /// r₂, the cut level's Stage-2 point, as the residual witness's
    /// evaluation point: the first coordinate is the most significant index
    /// bit (`specs/whiteout-sigma.md` A18).
    pub point: Point<HIGH_TO_LOW, F>,
    /// The cut level's witness evaluation w̃_k(r₂), masked.
    pub witness_eval: AffineRow<F>,
    /// u_a, the raw outer commitment of the cut level's output witness w_k,
    /// over the base field `B`.
    pub residual_commitment: Vec<B>,
}

/// The masked prefix of a zero-knowledge opening, after the replay of levels
/// `0..=cut_level`: the stage checks of every masked level, in transcript
/// order, and the cut.
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
    cut: CutState<F, B>,
}

impl<F: Field, B> MaskedPrefix<F, B> {
    pub(crate) fn new<G>(replay: MaskedReplay<'_, G, F>, cut: CutState<F, B>) -> Self {
        Self {
            checks: replay.checks,
            cut,
        }
    }

    /// The cut.
    pub fn cut(&self) -> &CutState<F, B> {
        &self.cut
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
        relation.affine.push(
            AffineRow::variable(ResidualVar::Witness(0))
                .add_scaled(&self.prefix.cut.witness_eval, -F::one()),
        );
        Ok(())
    }
}

impl<F, B> CutState<F, B> {
    /// The cut of a replay that ended at `cut_level` with Stage-2 point
    /// `challenges` (in round order, low variable first).
    pub(crate) fn new(
        cut_level: u32,
        challenges: Vec<F>,
        witness_eval: AffineRow<F>,
        residual_commitment: Vec<B>,
    ) -> Self
    where
        F: Copy,
    {
        Self {
            cut_level,
            point: Point::<LOW_TO_HIGH, F>::low_to_high(challenges)
                .match_endianness::<HIGH_TO_LOW>(),
            witness_eval,
            residual_commitment,
        }
    }
}
