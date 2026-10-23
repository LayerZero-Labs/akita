//! Zero-knowledge opening schedules and their hiding randomness.
//!
//! A zero-knowledge schedule carries a [`ZkOpeningSchedule`] whose cut masks
//! levels `0..=cut_level`. At each masked level:
//!
//! - the fold response of every group is one-shot: the grinding plan
//!   schedules it with `nonce_bits = 0`, so no nonce precedes the fold
//!   challenges, and an out-of-range response aborts the proof instead of
//!   being retried;
//! - each group's fold digit count meets
//!   [`ZkOpeningSchedule::fold_failure_target`] for the level's witness
//!   groups;
//! - the B slices of every group that carries witness data, and the level's
//!   D image, carry fresh hiding randomness columns, sized by
//!   [`hiding_randomness_columns`]; the randomness hides each image by the
//!   leftover hash lemma once its entropy exceeds the image's bit length by
//!   `2λ` plus a fixed slack. A setup prefix is public and carries none.
//!
//! [`ZkOpeningSchedule::admit`] checks all of this in release builds. Catalog
//! admission and the prover both run it. A clear catalog rejects every
//! zero-knowledge schedule and a zero-knowledge catalog every clear one, so
//! only a zero-knowledge configuration selects one.

use crate::sis::{
    BalancedSignedDigitFoldPolicy, CommittedSourceContract, FoldFailureTarget, FoldWitnessNorms,
    HonestFoldPolicy, HonestFoldPolicySpec, HonestFoldSizingQuery, SisModulusProfileId,
};
use crate::{
    CommitmentPayloadMode, CommitmentSliceCount, CommittedGroupParams, FoldParams, FoldSchedule,
    GroupCommitPhaseParams, GroupOpenPhaseParams, InnerCommitSecurityRoute, OpeningMethod,
    RelationQuotientPlan, RelationWitnessGeometry, WitnessLayout,
};
use akita_error::AkitaError;
use core::ops::Range;
use jolt_algebra::{Prime128Offset275, Prime128OffsetA7F7, Prime32Offset99, Prime64Offset59};
use jolt_crypto::challenge::SparseChallengeConfig;
use jolt_crypto::lattice::required_randomness_columns;

/// The only supported cut: levels 0 and 1 are masked (`k = 2`).
pub const ZK_CUT_LEVEL: u32 = 1;

/// Every one-shot response of a witness group fails with probability at most
/// `2^-ZK_FOLD_SECURITY_BITS / masked_folds`, so all of them together fail
/// with probability at most `2^-ZK_FOLD_SECURITY_BITS`.
pub const ZK_FOLD_SECURITY_BITS: u32 = 136;

/// Failure target of each one-shot group response at a level of an opening
/// masked through `cut_level` that folds `witness_groups` groups carrying
/// witness data.
///
/// Each of the `cut_level + 1` masked levels gets an equal share of the
/// budget, split equally among its witness groups, so the union over every
/// witness response is `2^-ZK_FOLD_SECURITY_BITS`. A setup prefix takes its
/// level's target too, but its response is a function of public data, so its
/// failure is a completeness term outside that budget.
pub(crate) fn one_shot_fold_target(
    cut_level: u32,
    witness_groups: usize,
) -> Result<FoldFailureTarget, AkitaError> {
    let masked_folds = u32::try_from(witness_groups)
        .ok()
        .filter(|&groups| groups != 0)
        .and_then(|groups| groups.checked_mul(cut_level.checked_add(1)?))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "a masked level cannot fold {witness_groups} witness groups"
            ))
        })?;
    Ok(FoldFailureTarget::OneShot {
        security_bits: ZK_FOLD_SECURITY_BITS,
        masked_folds,
    })
}

/// The zero-knowledge part of a fold schedule: the cut that masks levels
/// `0..=cut_level`.
///
/// A [`FoldSchedule`] stores it, its descriptor bytes bind it, and its
/// grinding plan makes the masked fold responses one-shot. Deserializing one
/// validates the cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ZkCutEncoding")]
pub struct ZkOpeningSchedule {
    cut_level: u32,
}

/// The serialized form of a [`ZkOpeningSchedule`], before its cut is checked.
#[derive(serde::Deserialize)]
struct ZkCutEncoding {
    cut_level: u32,
}

impl TryFrom<ZkCutEncoding> for ZkOpeningSchedule {
    type Error = AkitaError;

    fn try_from(encoding: ZkCutEncoding) -> Result<Self, AkitaError> {
        let zk = Self {
            cut_level: encoding.cut_level,
        };
        zk.validate_cut()?;
        Ok(zk)
    }
}

impl ZkOpeningSchedule {
    /// The cut masking levels `0..=cut_level`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] for a cut other than
    /// [`ZK_CUT_LEVEL`].
    pub fn new(cut_level: u32) -> Result<Self, AkitaError> {
        Self::try_from(ZkCutEncoding { cut_level })
    }

    fn validate_cut(self) -> Result<(), AkitaError> {
        if self.cut_level != ZK_CUT_LEVEL {
            return Err(AkitaError::UnsupportedSchedule(format!(
                "zero-knowledge cut level {} is unsupported; only {ZK_CUT_LEVEL} is",
                self.cut_level
            )));
        }
        Ok(())
    }

    /// The last masked level.
    #[must_use]
    pub const fn cut_level(self) -> u32 {
        self.cut_level
    }

    /// Whether fold `level` is masked.
    #[must_use]
    pub const fn masks(self, level: usize) -> bool {
        level <= self.cut_level as usize
    }

    /// Failure target of each one-shot group response at a masked level that
    /// folds `witness_groups` groups carrying witness data (every group but
    /// a setup prefix).
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] for zero witness groups or an
    /// overflowing response count.
    pub fn fold_failure_target(
        self,
        witness_groups: usize,
    ) -> Result<FoldFailureTarget, AkitaError> {
        one_shot_fold_target(self.cut_level, witness_groups)
    }

    /// The one admission check of a zero-knowledge schedule, in release
    /// builds.
    ///
    /// `root_source` is the contract the root's own group is produced under,
    /// and `claim_extension_degree` the degree of the claim field over the
    /// base field. The check requires:
    ///
    /// - `schedule.zk` is a supported cut that precedes the last nonterminal
    ///   fold, whose output is the terminal witness, sent in the clear;
    /// - claims in the base field (`E::DEGREE = 1`): the masked stage checks
    ///   have no extension-field products;
    /// - no setup prefix at the level after the cut, so the cut level runs no
    ///   Stage 3;
    /// - the residual commitment, which the level after the cut opens, has
    ///   d_A = d_B;
    /// - every masked level uses the ℓ∞ route (an ℓ2 route reveals the
    ///   response energy and carries a per-attempt ℓ2 cap and an
    ///   operator-norm challenge family, none of which is one-shot sized),
    ///   the quotient relation (not reduced evaluation), the open basis b = 8
    ///   for every group, and one witness chunk, and carries exactly
    ///   [`d_randomness_columns`] for its D image;
    /// - every group a masked level folds has at least the fold digits its
    ///   [`MaskedFoldSizing`] derives for the level's
    ///   [`fold_failure_target`](Self::fold_failure_target), and its B slices
    ///   carry exactly [`b_randomness_columns`] if it carries witness data
    ///   (the level's own group, a precommitted group) and none if it is a
    ///   setup prefix, whose commitment is public setup data;
    /// - the level after the cut, whose own group is the residual commitment
    ///   to w_k, opens at [`RESIDUAL_RANDOMNESS_LOG_BASIS`] in one outer
    ///   slice with a raw payload, carries exactly [`b_randomness_columns`] B
    ///   columns (r_a), no D randomness and no preceding group;
    /// - every group of every later level carries no randomness.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] for a missing or
    /// unsupported cut and [`AkitaError::InvalidSetup`] naming the first
    /// violating condition otherwise.
    pub fn admit(
        schedule: &FoldSchedule,
        root_source: CommittedSourceContract,
        claim_extension_degree: usize,
    ) -> Result<Self, AkitaError> {
        let zk = schedule.zk.ok_or_else(|| {
            AkitaError::UnsupportedSchedule("schedule has no zero-knowledge cut".into())
        })?;
        zk.validate_cut()?;
        let cut = zk.cut_level as usize;
        let after_cut = schedule.recursive_folds.get(cut).ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "zero-knowledge cut level {cut} reaches the terminal witness"
            ))
        })?;
        if claim_extension_degree != 1 {
            return Err(AkitaError::InvalidSetup(
                "zero knowledge needs claims in the base field".into(),
            ));
        }
        if after_cut.params.setup_prefix().is_some() {
            return Err(AkitaError::InvalidSetup(
                "the cut level must not run Stage 3".into(),
            ));
        }
        let residual = after_cut.params.role_dims();
        if residual.d_a() != residual.d_b() {
            return Err(AkitaError::InvalidSetup(
                "the residual commitment needs d_A = d_B".into(),
            ));
        }
        let levels = std::iter::once(&schedule.root.params)
            .chain(schedule.recursive_folds.iter().map(|fold| &fold.params));
        for (level, params) in levels.enumerate() {
            let invalid = |what: &str| AkitaError::InvalidSetup(format!("level {level} {what}"));
            let own_index = params.groups().len() - 1;
            let expected_d = if zk.masks(level) {
                if !matches!(
                    params.inner().matrix.security_route(),
                    InnerCommitSecurityRoute::Linf(_)
                ) {
                    return Err(invalid("is masked on an L2 route"));
                }
                if params.ring_relation_mode.is_reduced_evaluation() {
                    return Err(invalid("is masked under reduced evaluation"));
                }
                if params.open().digits.log_basis != MASKED_LOG_BASIS
                    || params
                        .groups()
                        .iter()
                        .any(|group| group.opening.log_basis_open != MASKED_LOG_BASIS)
                {
                    return Err(invalid("is masked with an open basis other than 8"));
                }
                if params.witness_chunk.num_chunks != 1 {
                    return Err(invalid("is masked across several witness chunks"));
                }
                let target = zk.fold_failure_target(witness_group_count(params))?;
                for (index, group) in params.groups().iter().enumerate() {
                    let sizing = MaskedFoldSizing::for_group(schedule, level, index, root_source)?;
                    let required = sizing.num_digits_fold(target)?;
                    if group.opening.num_digits_fold < required {
                        return Err(invalid(&format!(
                            "has {} fold digits in group {index}; its one-shot target needs \
                             {required}",
                            group.opening.num_digits_fold
                        )));
                    }
                    let expected_b = if group.setup_natural_len.is_some() {
                        0
                    } else {
                        b_randomness_columns(group)?
                    };
                    if group.profile.randomness_columns != expected_b {
                        return Err(invalid(if index == own_index {
                            "has the wrong B hiding randomness"
                        } else {
                            "gives a preceding group the wrong B hiding randomness"
                        }));
                    }
                }
                d_randomness_columns(params)?
            } else if level == cut + 1 {
                let own = params.own_group();
                // The cut successor: its own group is the residual commitment
                // to w_k, hidden by r_a, which the Σ-protocol consumes at
                // the residual alphabet; the level itself is never folded.
                // It consumes no preceding group: the Stage-3 check above.
                if own.opening.log_basis_open != RESIDUAL_RANDOMNESS_LOG_BASIS {
                    return Err(invalid(
                        "opens the residual commitment at a basis other than 16",
                    ));
                }
                if own.profile.outer_slice_count != CommitmentSliceCount::ONE {
                    return Err(invalid("commits the residual in more than one outer slice"));
                }
                if params.open_randomness_columns != 0 {
                    return Err(invalid("carries D hiding randomness after the cut"));
                }
                if params.payload_mode != CommitmentPayloadMode::Raw {
                    return Err(invalid("sends the residual commitment compressed"));
                }
                if own.profile.randomness_columns != b_randomness_columns(own)? {
                    return Err(invalid("has the wrong residual randomness r_a"));
                }
                continue;
            } else {
                if params.own_group().profile.randomness_columns != 0 {
                    return Err(invalid("has the wrong B hiding randomness"));
                }
                if params
                    .preceding_group_iter()
                    .any(|group| group.profile.randomness_columns != 0)
                {
                    return Err(invalid(
                        "gives a preceding group the wrong B hiding randomness",
                    ));
                }
                0
            };
            if params.open_randomness_columns != expected_d {
                return Err(invalid("has the wrong D hiding randomness"));
            }
        }
        Ok(zk)
    }
}

/// Groups of a fold that carry witness data: every group but a setup prefix,
/// whose source is public setup data.
#[must_use]
pub fn witness_group_count(params: &CommittedGroupParams) -> usize {
    params
        .groups()
        .iter()
        .filter(|group| group.setup_natural_len.is_none())
        .count()
}

/// Base-two logarithm of the open basis the masked stage checks assume
/// (b = 8: no Stage-1 product stages).
pub const MASKED_LOG_BASIS: u32 = 3;

/// Base-two logarithm of the alphabet of r_a, the residual commitment's
/// hiding randomness (radix 16). The level after the cut opens at this basis,
/// so [`b_randomness_columns`] sizes r_a, and the backend samples it, at the
/// alphabet the Σ-protocol's `ResidualRandomness` segment declares
/// (`specs/whiteout-akita.md` §4.8.1).
pub const RESIDUAL_RANDOMNESS_LOG_BASIS: u32 = 4;

/// The honest-fold sizing inputs of one group a masked level folds, read
/// from its scheduled parameters.
///
/// - The root's own group is sized by the root's source contract.
/// - A recursive level's own group holds the balanced digits of the previous
///   witness.
/// - A preceding group (a setup prefix or a precommitted group) is sized by
///   the universal balanced policy on its own inner digit alphabet. Every
///   committed digit lies in that alphabet whatever its producer, so the
///   bound holds without the producer contract, which the schedule does not
///   record.
pub struct MaskedFoldSizing<'a> {
    params: &'a CommittedGroupParams,
    group: &'a GroupOpenPhaseParams,
    policy: HonestFoldPolicySpec,
    challenge: SparseChallengeConfig,
    num_claims: usize,
    witness_norms: FoldWitnessNorms,
}

impl<'a> MaskedFoldSizing<'a> {
    /// The sizing inputs of the own group of fold `level` of `schedule`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] for a level that is not a
    /// nonterminal fold.
    pub fn new(
        schedule: &'a FoldSchedule,
        level: usize,
        root_source: CommittedSourceContract,
    ) -> Result<Self, AkitaError> {
        let own_index = level_params(schedule, level)?.groups().len() - 1;
        Self::for_group(schedule, level, own_index, root_source)
    }

    /// The sizing inputs of group `group_index`, in canonical group order,
    /// of fold `level` of `schedule`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::UnsupportedSchedule`] for a level that is not a
    /// nonterminal fold or a missing group.
    pub fn for_group(
        schedule: &'a FoldSchedule,
        level: usize,
        group_index: usize,
        root_source: CommittedSourceContract,
    ) -> Result<Self, AkitaError> {
        let field_bits = root_source.decomposition().field_bits();
        let params = level_params(schedule, level)?;
        let group = params.groups().get(group_index).ok_or_else(|| {
            AkitaError::UnsupportedSchedule(format!("level {level} has no group {group_index}"))
        })?;
        let inner = group.profile.inner.digits;
        let d_a = group.profile.inner.matrix.ring_dimension();
        let universal = HonestFoldPolicySpec::BalancedSignedDigit(
            BalancedSignedDigitFoldPolicy::universal(field_bits),
        );
        let (policy, num_claims, witness_norms) = if group_index + 1 != params.groups().len() {
            (
                universal,
                group.profile.group.num_polynomials(),
                FoldWitnessNorms::bounded(inner.log_basis, d_a),
            )
        } else if level == 0 {
            let source = group.profile.group;
            let source_len = jolt_algebra::checked::pow2(source.num_vars())
                .and_then(|len| len.checked_mul(source.num_polynomials()))
                .ok_or_else(|| AkitaError::InvalidSetup("source length overflow".into()))?;
            (
                root_source.class().honest_fold_policy(field_bits),
                source.num_polynomials(),
                root_source
                    .source_norms(inner.log_basis, inner.num_digits, d_a, source_len)?
                    .fold_witness,
            )
        } else {
            (
                universal,
                1,
                FoldWitnessNorms::bounded(inner.log_basis, d_a),
            )
        };
        Ok(Self {
            params,
            group,
            policy,
            challenge: group.opening.fold_challenge_config,
            num_claims,
            witness_norms,
        })
    }

    /// The level's scheduled parameters.
    #[must_use]
    pub const fn params(&self) -> &'a CommittedGroupParams {
        self.params
    }

    /// The group's honest-fold policy.
    #[must_use]
    pub const fn policy(&self) -> &HonestFoldPolicySpec {
        &self.policy
    }

    /// The group's scheduled parameters.
    #[must_use]
    pub const fn group(&self) -> &'a GroupOpenPhaseParams {
        self.group
    }

    /// The sizing query for `failure_target`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when the response size overflows.
    pub fn query(
        &self,
        failure_target: FoldFailureTarget,
    ) -> Result<HonestFoldSizingQuery<'_>, AkitaError> {
        let group = self.group;
        let d_a = group.profile.inner.matrix.ring_dimension();
        HonestFoldSizingQuery::new(
            d_a,
            match group.opening.opening_method {
                OpeningMethod::EvaluationTrace => d_a,
                OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension,
                } => challenge_subring_dimension,
            },
            self.num_claims,
            group.profile.blocks,
            self.params.witness_chunk.num_chunks,
            group.profile.inner.matrix.input_width(),
            self.witness_norms,
            group.opening.log_basis_open,
            &self.challenge,
            failure_target,
        )
    }

    /// The fold digit count the group's policy derives for `failure_target`.
    pub fn num_digits_fold(&self, failure_target: FoldFailureTarget) -> Result<usize, AkitaError> {
        self.policy.num_digits_fold(self.query(failure_target)?)
    }
}

/// The parameters of nonterminal fold `level` of `schedule`.
fn level_params(
    schedule: &FoldSchedule,
    level: usize,
) -> Result<&CommittedGroupParams, AkitaError> {
    match level.checked_sub(1) {
        None => Ok(&schedule.root.params),
        Some(index) => schedule
            .recursive_folds
            .get(index)
            .map(|fold| &fold.params)
            .ok_or_else(|| {
                AkitaError::UnsupportedSchedule("a masked level must be a nonterminal fold".into())
            }),
    }
}

/// Slack bits added on top of `2λ` when sizing hiding randomness.
const HIDING_SLACK_BITS: usize = 8;

/// Shape of one outer-map image hidden by a randomness vector: one B slice or
/// one level's D image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageShape {
    /// Image rank (output ring elements).
    pub rows: usize,
    /// Ring dimension of the image and of each randomness column.
    pub ring_dim: usize,
    /// Field modulus profile of the image coefficients.
    pub modulus: SisModulusProfileId,
}

/// Balanced digit alphabet `[-2^(log_basis-1), 2^(log_basis-1) - 1]` of the
/// witness that holds the randomness. It is that witness's open basis, the one
/// alphabet the next level's range check proves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DigitAlphabet {
    /// Base-two logarithm of the digit base.
    pub log_basis: u32,
}

/// Number of randomness ring columns that hide one image of `image` shape
/// with digits drawn uniformly from `alphabet`.
///
/// # Errors
///
/// Returns an error for a zero ring dimension, an alphabet outside `1..=8`,
/// or an overflowing size.
pub fn hiding_randomness_columns(
    image: ImageShape,
    alphabet: DigitAlphabet,
) -> Result<usize, AkitaError> {
    let image_coeffs = image
        .rows
        .checked_mul(image.ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("hidden image size overflows usize".into()))?;
    let columns = match image.modulus {
        SisModulusProfileId::Q32Offset99 => required_randomness_columns::<Prime32Offset99>(
            image_coeffs,
            0,
            image.ring_dim,
            alphabet.log_basis,
            HIDING_SLACK_BITS,
        ),
        SisModulusProfileId::Q64Offset59 => required_randomness_columns::<Prime64Offset59>(
            image_coeffs,
            0,
            image.ring_dim,
            alphabet.log_basis,
            HIDING_SLACK_BITS,
        ),
        SisModulusProfileId::Q128OffsetA7F7 => required_randomness_columns::<Prime128OffsetA7F7>(
            image_coeffs,
            0,
            image.ring_dim,
            alphabet.log_basis,
            HIDING_SLACK_BITS,
        ),
        SisModulusProfileId::Q128Offset275 => required_randomness_columns::<Prime128Offset275>(
            image_coeffs,
            0,
            image.ring_dim,
            alphabet.log_basis,
            HIDING_SLACK_BITS,
        ),
    }?;
    Ok(columns)
}

/// Hiding-randomness columns of every B slice of `group`'s commitment.
///
/// The randomness hides one slice image and is drawn from the open alphabet
/// of the opening that consumes the commitment, the alphabet that level's
/// range check proves.
///
/// # Errors
///
/// Returns the sizing errors of [`hiding_randomness_columns`].
pub fn b_randomness_columns(group: &GroupOpenPhaseParams) -> Result<usize, AkitaError> {
    let outer = &group.profile.outer.matrix;
    hiding_randomness_columns(
        ImageShape {
            rows: outer.output_rank(),
            ring_dim: outer.ring_dimension(),
            modulus: outer.sis_modulus_profile(),
        },
        DigitAlphabet {
            log_basis: group.opening.log_basis_open,
        },
    )
}

/// Hiding-randomness columns of the D image `params`' level makes, drawn from
/// that level's open alphabet.
///
/// # Errors
///
/// Returns the sizing errors of [`hiding_randomness_columns`].
pub fn d_randomness_columns(params: &CommittedGroupParams) -> Result<usize, AkitaError> {
    let open = params.open();
    hiding_randomness_columns(
        ImageShape {
            rows: open.matrix.output_rank(),
            ring_dim: open.matrix.ring_dimension(),
            modulus: open.matrix.sis_modulus_profile(),
        },
        DigitAlphabet {
            log_basis: open.digits.log_basis,
        },
    )
}

/// The residual commitment at a zero-knowledge schedule's cut, as public
/// data: the one source of the Σ-protocol's residual segments, their
/// alphabets and the opening rows (`specs/whiteout-akita.md` §4.6, §4.8.1).
/// The prover's hand-off and both roles' Σ setup read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidualGeometry {
    /// The last masked level.
    pub cut_level: u32,
    /// Base-two logarithm of w_k's balanced alphabet: the cut level's open
    /// basis, which its range check proves.
    pub witness_log_basis: u32,
    /// The residual commitment to w_k: the cut successor's own group. Its
    /// inner digits t̂_a are `profile.outer.digits`.
    pub profile: GroupCommitPhaseParams,
    /// Base-two logarithm of r_a's alphabet: the cut successor's open basis
    /// ([`RESIDUAL_RANDOMNESS_LOG_BASIS`] after admission).
    pub randomness_log_basis: u32,
    /// Coefficient ranges of w_k that hold only public data, ascending and
    /// disjoint: the response and inner digits of the cut level's setup
    /// prefix (§3.5), a function of the setup and of public challenges.
    /// Empty when the cut level consumes no setup prefix.
    pub public_spans: Vec<Range<usize>>,
}

impl ResidualGeometry {
    /// The residual geometry of `schedule` over a `field_bits`-bit base
    /// field.
    ///
    /// # Errors
    ///
    /// [`AkitaError::UnsupportedSchedule`] without a zero-knowledge cut,
    /// [`AkitaError::InvalidSetup`] if the cut has no successor level, and
    /// the witness-layout errors of the cut level.
    pub fn new(schedule: &FoldSchedule, field_bits: u32) -> Result<Self, AkitaError> {
        let zk = schedule.zk.ok_or_else(|| {
            AkitaError::UnsupportedSchedule("schedule has no zero-knowledge cut".into())
        })?;
        let cut = zk.cut_level() as usize;
        let successor = schedule.recursive_folds.get(cut).ok_or_else(|| {
            AkitaError::InvalidSetup("the zero-knowledge cut has no successor level".into())
        })?;
        let (cut_params, public_spans) = match cut.checked_sub(1) {
            // The root consumes no setup prefix.
            None => (&schedule.root.params, Vec::new()),
            Some(index) => {
                let fold = schedule.recursive_folds.get(index).ok_or_else(|| {
                    AkitaError::InvalidSetup("the cut level is not scheduled".into())
                })?;
                (&fold.params, setup_prefix_spans(fold, field_bits)?)
            }
        };
        let own = successor.params.own_group();
        Ok(Self {
            cut_level: zk.cut_level(),
            witness_log_basis: cut_params.open().digits.log_basis,
            profile: own.profile,
            randomness_log_basis: own.opening.log_basis_open,
            public_spans,
        })
    }
}

/// The witness ranges of `fold`'s output that its setup prefix fills: the
/// prefix group's response and inner digits.
fn setup_prefix_spans(fold: &FoldParams, field_bits: u32) -> Result<Vec<Range<usize>>, AkitaError> {
    let params = &fold.params;
    let Some(prefix) = params.setup_prefix() else {
        return Ok(Vec::new());
    };
    let layout = crate::suffix_opening_layout(fold.input_witness_len, prefix.setup_natural_len)?;
    // Extension degree 1: a zero-knowledge schedule's claims are in the base
    // field (`ZkOpeningSchedule::admit` rejects any other degree).
    let geometry = RelationWitnessGeometry::for_level(params, &layout, 1)?;
    let witness = WitnessLayout::new(
        params,
        &layout,
        &geometry,
        params.witness_chunk.num_chunks,
        RelationQuotientPlan::for_field_bits(params, field_bits)?,
    )?;
    // The setup prefix is the fold's group 0 (`suffix_opening_layout`).
    let mut spans = witness
        .units()
        .iter()
        .filter(|unit| unit.group_index() == 0)
        .flat_map(|unit| [unit.z_range(), unit.t_range()])
        .filter(|range| !range.is_empty())
        .collect::<Vec<_>>();
    spans.sort_by_key(|range| range.start);
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headline_slice_needs_forty_five_columns() {
        // d_b = 64, rank 1, b = 8 over the 128-bit field: 64 * 128 + 256 + 8
        // bits at 192 bits per column.
        let image = ImageShape {
            rows: 1,
            ring_dim: 64,
            modulus: SisModulusProfileId::Q128OffsetA7F7,
        };
        assert_eq!(
            hiding_randomness_columns(image, DigitAlphabet { log_basis: 3 }).unwrap(),
            45
        );
    }

    #[test]
    fn residual_commitment_needs_ninety_seven_columns() {
        // Three R_256 rows at b = 16: 768 * 128 + 264 bits at 1024 bits per
        // column.
        let image = ImageShape {
            rows: 3,
            ring_dim: 256,
            modulus: SisModulusProfileId::Q128OffsetA7F7,
        };
        assert_eq!(
            hiding_randomness_columns(image, DigitAlphabet { log_basis: 4 }).unwrap(),
            97
        );
    }

    #[test]
    fn witness_groups_split_each_masked_level_share() {
        let target = |groups| one_shot_fold_target(ZK_CUT_LEVEL, groups);
        let one_shot = |masked_folds| FoldFailureTarget::OneShot {
            security_bits: ZK_FOLD_SECURITY_BITS,
            masked_folds,
        };
        assert_eq!(target(1), Ok(one_shot(2)));
        assert_eq!(target(3), Ok(one_shot(6)));
        assert!(target(0).is_err());
    }

    #[test]
    fn rejects_unsupported_alphabets() {
        let image = ImageShape {
            rows: 1,
            ring_dim: 64,
            modulus: SisModulusProfileId::Q64Offset59,
        };
        assert!(hiding_randomness_columns(image, DigitAlphabet { log_basis: 0 }).is_err());
        assert!(hiding_randomness_columns(image, DigitAlphabet { log_basis: 9 }).is_err());
    }
}
