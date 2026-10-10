//! Proof-of-work plans for root field challenges.
//!
//! Akita prices each field challenge to a `2^-128` per-query rate
//! (`specs/transcript-grinding.md`): a site whose conditional bad fraction is
//! `L / |E|` carries the least `g` with `L * 2^128 <= |E| * 2^g` bits of proof
//! of work, paid by an inline nonce directly before the protected challenge.
//! This module names the root reduction's sites and their loss factors. The
//! target arithmetic, the cap, the predicate transition and the nonce codec
//! are Akita's and are called from here and from `channel`, not restated.

use akita_algebra::binary::{field_switch::SwitchField, BinaryField162};
use akita_error::{checked, AkitaError};
use akita_params::{
    grind_bits_for_loss, multilinear_point_loss_factor, polynomial_identity_loss_factor,
    powers_batch_loss_factor,
    sis::labinius::{LabiniusRootEncoding, LabiniusRootShape},
    ChallengeFieldOrder, GRINDING_NONCE_SLACK_BITS, MAX_GRINDING_BITS, TRANSCRIPT_SECURITY_BITS,
};
use akita_sumcheck::SumcheckShape;
use akita_transcript::nonce_max_bytes;
use jolt_field::{CanonicalEncoding, ExtField, Field};

use crate::{
    root_sumcheck::{combined_shape, product_shape},
    statement::RootOpeningMode,
};

const PLAN_DOMAIN: &[u8] = b"akita/labinius/root-grinding-plan/v1";
/// Site tag, invocation, loss factor, target, nonce width and multiplicity.
const RUN_BYTES: usize = 1 + 4 + 8 + 1 + 1 + 8;
/// A frontend challenge is uniform over the `2^162` elements of `B`.
const BINARY_CHALLENGE_BITS: u32 = BinaryField162::DEGREE as u32;

/// One field-challenge site of the root reduction.
///
/// The two frontend sites draw from the binary field `B`; every other site
/// draws from the challenge field `E`. Sumcheck instances `0` and `1` are the
/// combined instances of the response and image tables; instance `2` is the
/// product instance of the prime left opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootGrindingSite {
    /// The frontend's batching point, `H::BATCH_BITS` coordinates.
    FrontendBatch,
    /// One round challenge of the frontend sumcheck.
    FrontendRound,
    /// Evaluation point of the commitment rows and their carries.
    Alpha,
    /// Evaluation point of the parity row.
    Xi,
    /// Powers batching the commitment rows and, with a binary claim, the
    /// parity row.
    Gamma,
    /// Equality point of the combined instance `invocation`.
    EqualityPoint { invocation: u32 },
    /// Scalar batching the two terms of instance `invocation`: the alphabet
    /// term with the linear term of a combined instance, the value claim with
    /// the row claim of the product instance.
    Batch { invocation: u32 },
    /// One round challenge of sumcheck instance `invocation`.
    Round { invocation: u32 },
    /// Scalar batching the prime row with the gamma-batched rows.
    PrimeRow,
}

impl RootGrindingSite {
    /// Canonical discriminator and invocation; the discriminator is also the
    /// site's diagnostic detail.
    pub(crate) const fn key(self) -> (u8, u32) {
        match self {
            Self::FrontendBatch => (0, 0),
            Self::FrontendRound => (1, 0),
            Self::Alpha => (2, 0),
            Self::Xi => (3, 0),
            Self::Gamma => (4, 0),
            Self::EqualityPoint { invocation } => (5, invocation),
            Self::Batch { invocation } => (6, invocation),
            Self::Round { invocation } => (7, invocation),
            Self::PrimeRow => (8, 0),
        }
    }

    const fn in_frontend(self) -> bool {
        matches!(self, Self::FrontendBatch | Self::FrontendRound)
    }
}

/// Consecutive visits of one site, all with the same loss factor and target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootGrindingRun {
    site: RootGrindingSite,
    loss_factor: u64,
    grind_bits: u8,
    multiplicity: usize,
}

impl RootGrindingRun {
    pub const fn site(&self) -> RootGrindingSite {
        self.site
    }

    /// Numerator `L` of the site's conditional bad fraction `L / |E|`.
    pub const fn loss_factor(&self) -> u64 {
        self.loss_factor
    }

    /// Low predicate bits a nonce must clear; zero for a site without work.
    pub const fn grind_bits(&self) -> u8 {
        self.grind_bits
    }

    /// Exclusive bit width of the nonce search range: zero for a site without
    /// work, otherwise the target plus Akita's slack.
    pub const fn nonce_bits(&self) -> u8 {
        if self.grind_bits == 0 {
            0
        } else {
            self.grind_bits.saturating_add(GRINDING_NONCE_SLACK_BITS)
        }
    }

    pub const fn multiplicity(&self) -> usize {
        self.multiplicity
    }
}

fn overflow() -> AkitaError {
    AkitaError::InvalidSetup("root grinding plan overflow".into())
}

/// Append `multiplicity` visits of `site`, priced against the exact
/// cardinality of its challenge set. A site that would need more than Akita's
/// cap is an error; a site that is never visited is not recorded.
fn push_run(
    runs: &mut Vec<RootGrindingRun>,
    site: RootGrindingSite,
    loss_factor: u64,
    order: ChallengeFieldOrder,
    multiplicity: usize,
) -> Result<(), AkitaError> {
    let grind_bits = grind_bits_for_loss(loss_factor, order).map_err(|_| {
        AkitaError::InvalidSetup(
            "LaBinius challenge field is too small for the root reduction's challenges".into(),
        )
    })?;
    if multiplicity != 0 {
        runs.try_reserve(1).map_err(|_| overflow())?;
        runs.push(RootGrindingRun {
            site,
            loss_factor,
            grind_bits,
            multiplicity,
        });
    }
    Ok(())
}

/// Append the sites of an opening in `mode` whose challenges are elements of
/// `E`, in replay order.
///
/// This is the admission of the challenge field: `E` over `F` is admitted
/// exactly when every one of these sites has a target within Akita's cap.
pub(crate) fn field_runs<F, E>(
    shape: &LabiniusRootShape,
    encoding: &LabiniusRootEncoding,
    mode: RootOpeningMode,
    runs: &mut Vec<RootGrindingRun>,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let order = ChallengeFieldOrder::from_field(
        F::MODULUS_BITS,
        E::DEGREE,
        crate::admitted::field_characteristic::<F>()?,
    )?;
    // A commitment row and the prime row are tested after reduction modulo
    // the degree-`D` modulus; the parity row is tested unreduced, with its
    // quotient term.
    let row_degree = shape
        .commitment_degree()
        .checked_sub(1)
        .ok_or_else(overflow)?;
    let parity_degree = checked::product([shape.scalar_degree(), 2])
        .and_then(|degree| degree.checked_sub(2))
        .ok_or_else(overflow)?;
    let rows = usize::try_from(shape.rank_a())
        .ok()
        .and_then(|rank| checked::sum([rank, usize::from(mode.has_binary())]))
        .ok_or_else(overflow)?;
    push_run(
        runs,
        RootGrindingSite::Alpha,
        polynomial_identity_loss_factor(row_degree)?,
        order,
        1,
    )?;
    if mode.has_binary() {
        push_run(
            runs,
            RootGrindingSite::Xi,
            polynomial_identity_loss_factor(parity_degree)?,
            order,
            1,
        )?;
    }
    push_run(
        runs,
        RootGrindingSite::Gamma,
        powers_batch_loss_factor(rows)?,
        order,
        1,
    )?;
    if mode.has_prime() {
        push_run(
            runs,
            RootGrindingSite::PrimeRow,
            powers_batch_loss_factor(2)?,
            order,
            1,
        )?;
    }
    for (invocation, num_vars) in [
        (0, encoding.response_table_log_len()),
        (1, encoding.image_table_log_len()),
    ] {
        push_run(
            runs,
            RootGrindingSite::EqualityPoint { invocation },
            multilinear_point_loss_factor(num_vars)?,
            order,
            1,
        )?;
        push_run(
            runs,
            RootGrindingSite::Batch { invocation },
            powers_batch_loss_factor(2)?,
            order,
            1,
        )?;
        push_run(
            runs,
            RootGrindingSite::Round { invocation },
            polynomial_identity_loss_factor(combined_shape(num_vars)?.degree_bound())?,
            order,
            num_vars,
        )?;
    }
    if mode.has_prime() {
        let num_vars = encoding.prime_table_log_len();
        push_run(
            runs,
            RootGrindingSite::Batch { invocation: 2 },
            powers_batch_loss_factor(2)?,
            order,
            1,
        )?;
        push_run(
            runs,
            RootGrindingSite::Round { invocation: 2 },
            polynomial_identity_loss_factor(product_shape(num_vars)?.degree_bound())?,
            order,
            num_vars,
        )?;
    }
    Ok(())
}

/// Field-challenge sites of a root reduction or standalone sumchecks, in replay order.
///
/// The reduction's plan is a function of the admitted shape, the field pair, the host
/// field and the opening mode. Its canonical bytes are absorbed before any
/// claim or proof message, so both roles replay the same sites with the same
/// targets. The three modes have three different plans: the frontend sites and
/// `Xi` appear exactly with a binary claim, `PrimeRow` and instance `2`
/// exactly with a prime claim. Binding the plan therefore binds the mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootGrindingPlan {
    runs: Vec<RootGrindingRun>,
}

impl RootGrindingPlan {
    /// Derive the plan of a reduction of `shape` in `mode` over the field
    /// pair `(F, E)` with host field `H`; `encoding` is the shape's encoding
    /// for the characteristic of `F`.
    ///
    /// The frontend runs only with a binary claim. Its challenges are uniform
    /// in `B`, so its sites are priced against `2^162` and need no work. The
    /// frontend has no nonce slot: a frontend site with a nonzero target is an
    /// error.
    pub fn new<H, F, E>(
        shape: &LabiniusRootShape,
        encoding: &LabiniusRootEncoding,
        mode: RootOpeningMode,
    ) -> Result<Self, AkitaError>
    where
        H: SwitchField,
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let binary = ChallengeFieldOrder::from_full_capacity(BINARY_CHALLENGE_BITS)?;
        let frontend_rounds = checked::ceil_log2(shape.num_cells()).ok_or_else(overflow)?;
        let mut runs = Vec::new();
        if mode.has_binary() {
            push_run(
                &mut runs,
                RootGrindingSite::FrontendBatch,
                multilinear_point_loss_factor(H::BATCH_BITS)?,
                binary,
                1,
            )?;
            push_run(
                &mut runs,
                RootGrindingSite::FrontendRound,
                polynomial_identity_loss_factor(2)?,
                binary,
                frontend_rounds,
            )?;
        }
        if runs.iter().any(|run| run.grind_bits != 0) {
            return Err(AkitaError::InvalidSetup(
                "a frontend challenge has no proof-of-work slot".into(),
            ));
        }
        field_runs::<F, E>(shape, encoding, mode, &mut runs)?;
        Ok(Self { runs })
    }

    /// Derive a standalone sumcheck plan containing only the round sites of
    /// `instances`, in replay order. Each invocation's checked shape fixes
    /// its round count and degree bound; both combined and product instances
    /// use the reduction's round loss function and exact challenge-field order.
    pub fn sumcheck_rounds<F, E>(instances: &[(u32, SumcheckShape)]) -> Result<Self, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let order = ChallengeFieldOrder::from_field(
            F::MODULUS_BITS,
            E::DEGREE,
            crate::admitted::field_characteristic::<F>()?,
        )?;
        let mut runs = Vec::new();
        for &(invocation, shape) in instances {
            push_run(
                &mut runs,
                RootGrindingSite::Round { invocation },
                polynomial_identity_loss_factor(shape.degree_bound())?,
                order,
                shape.num_rounds(),
            )?;
        }
        Ok(Self { runs })
    }

    pub fn runs(&self) -> &[RootGrindingRun] {
        &self.runs
    }

    /// Largest number of proof bytes the plan's nonces occupy: every nonce at
    /// its canonical LEB128 maximum.
    pub fn nonce_max_bytes(&self) -> Result<usize, AkitaError> {
        self.runs
            .iter()
            .filter(|run| run.grind_bits != 0)
            .try_fold(0, |total, run| {
                checked::product([nonce_max_bytes(run.nonce_bits()), run.multiplicity])
                    .and_then(|bytes| checked::sum([total, bytes]))
            })
            .ok_or_else(overflow)
    }

    /// The statement bytes that fix the plan and the policy it was derived
    /// under.
    ///
    /// 1. The domain `akita/labinius/root-grinding-plan/v1`.
    /// 2. The security target in bits as two little-endian bytes, the nonce
    ///    slack in bits and the target cap, one byte each.
    /// 3. The run count as four little-endian bytes.
    /// 4. Per run: the site discriminator, the invocation as four
    ///    little-endian bytes, the loss factor as eight, the target and the
    ///    nonce width as one byte each, and the multiplicity as eight.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, AkitaError> {
        let count = u32::try_from(self.runs.len()).map_err(|_| overflow())?;
        let len = checked::product([self.runs.len(), RUN_BYTES])
            .and_then(|runs| checked::sum([PLAN_DOMAIN.len(), 2 + 1 + 1 + 4, runs]))
            .ok_or_else(overflow)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len).map_err(|_| overflow())?;
        bytes.extend_from_slice(PLAN_DOMAIN);
        bytes.extend_from_slice(&TRANSCRIPT_SECURITY_BITS.to_le_bytes());
        bytes.push(GRINDING_NONCE_SLACK_BITS);
        bytes.push(MAX_GRINDING_BITS);
        bytes.extend_from_slice(&count.to_le_bytes());
        for run in &self.runs {
            let (tag, invocation) = run.site.key();
            let multiplicity = u64::try_from(run.multiplicity).map_err(|_| overflow())?;
            bytes.push(tag);
            bytes.extend_from_slice(&invocation.to_le_bytes());
            bytes.extend_from_slice(&run.loss_factor.to_le_bytes());
            bytes.push(run.grind_bits);
            bytes.push(run.nonce_bits());
            bytes.extend_from_slice(&multiplicity.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// Monotone replay position over a plan's challenge-field sites.
///
/// The frontend draws its `B` challenges itself and its sites carry no work,
/// so replay starts at the first challenge-field site. An unscheduled cursor
/// holds no plan and rejects every visit.
#[derive(Debug, Default)]
pub(crate) struct RootGrindingCursor {
    plan: Option<RootGrindingPlan>,
    run: usize,
    offset: usize,
}

impl RootGrindingCursor {
    /// Start replaying `plan`. A cursor is scheduled at most once.
    pub(crate) fn schedule(&mut self, plan: RootGrindingPlan) -> Result<(), AkitaError> {
        if self.plan.is_some() {
            return Err(AkitaError::InvalidInput(
                "root channel already replays a grinding plan".into(),
            ));
        }
        self.run = plan
            .runs
            .iter()
            .take_while(|run| run.site.in_frontend())
            .count();
        self.offset = 0;
        self.plan = Some(plan);
        Ok(())
    }

    /// Consume one visit of `site`, which must be the next planned visit.
    pub(crate) fn next(&mut self, site: RootGrindingSite) -> Result<RootGrindingRun, AkitaError> {
        let mismatch = || {
            AkitaError::InvalidInput("root challenge site differs from the next plan entry".into())
        };
        let run = *self
            .plan
            .as_ref()
            .and_then(|plan| plan.runs.get(self.run))
            .filter(|run| run.site == site)
            .ok_or_else(mismatch)?;
        let offset = checked::sum([self.offset, 1]).ok_or_else(mismatch)?;
        if offset == run.multiplicity {
            self.run = checked::sum([self.run, 1]).ok_or_else(mismatch)?;
            self.offset = 0;
        } else {
            self.offset = offset;
        }
        Ok(run)
    }

    /// Require that a plan was scheduled and every visit was consumed.
    pub(crate) fn finish(&self) -> Result<(), AkitaError> {
        match &self.plan {
            Some(plan) if self.run == plan.runs.len() && self.offset == 0 => Ok(()),
            _ => Err(AkitaError::InvalidInput(
                "the root grinding plan has unconsumed sites".into(),
            )),
        }
    }
}
