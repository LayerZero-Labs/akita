use akita_challenges::{Challenges, SparseChallenge};
use akita_error::AkitaError;
pub(super) use akita_prover::backend::*;
use akita_types::{
    PreparedSubringCoefficientPackingPoint, RingMultiplierOpeningPoint,
    SubfieldMultiplierOpeningPoint, SubringCoefficientPackingGeometry,
};
use jolt_field::Field;
use std::ops::Range;
#[derive(Debug, Clone, Copy)]
pub(crate) enum OpeningFoldPlan<'a, F: Field> {
    /// Base multiplier point: scalar fold weights.
    Base {
        /// Outer evaluation scalars applied to the folded blocks.
        live_block_weights: &'a [F],
        /// Per-block fold scalars.
        position_weights: &'a [F],
        /// Number of ring-element positions in each block.
        num_positions_per_block: usize,
    },
    /// Proper-extension multiplier point in compact subfield coordinates.
    Subfield {
        /// Position and outer-fold multipliers for this opening.
        multipliers: &'a SubfieldMultiplierOpeningPoint<F>,
        /// Number of ring-element positions in each block.
        num_positions_per_block: usize,
    },
}

impl<F: Field> OpeningFoldPlan<'_, F> {
    pub(crate) fn num_positions_per_block(self) -> usize {
        match self {
            Self::Base {
                num_positions_per_block,
                ..
            }
            | Self::Subfield {
                num_positions_per_block,
                ..
            } => num_positions_per_block,
        }
    }

    /// Validate exact position and live-fold weight lengths at a kernel boundary.
    pub(crate) fn validate<const D: usize>(self, num_live_blocks: usize) -> Result<(), AkitaError> {
        let (fold_len, position_len, num_positions_per_block) = match self {
            Self::Base {
                live_block_weights,
                position_weights,
                num_positions_per_block,
            } => (
                live_block_weights.len(),
                position_weights.len(),
                num_positions_per_block,
            ),
            Self::Subfield {
                multipliers,
                num_positions_per_block,
            } => {
                multipliers.ensure_ring_dim::<D>()?;
                (
                    multipliers.fold_len(),
                    multipliers.position_len(),
                    num_positions_per_block,
                )
            }
        };
        if !num_positions_per_block.is_power_of_two()
            || num_live_blocks == 0
            || position_len != num_positions_per_block
            || fold_len != num_live_blocks
        {
            return Err(AkitaError::InvalidInput(
                "opening fold weights do not match exact L/F geometry".to_string(),
            ));
        }
        Ok(())
    }
}

/// Checked scalar inputs for one coefficient-packing projection batch.
///
/// Source data stays in the representation-specific borrowed batch view. The
/// output layout is fixed by `geometry` as
/// `[claim][block][extension coordinate][subring coefficient]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SubringCoefficientPackingPlan<'a, E: Field> {
    /// Canonically split public opening point.
    pub point: &'a PreparedSubringCoefficientPackingPoint<E>,
}

impl<E: Field> SubringCoefficientPackingPlan<'_, E> {
    /// Validate the plan at a source kernel boundary.
    pub(crate) fn validate<const D: usize>(
        &self,
        source_num_vars: usize,
    ) -> Result<(), AkitaError> {
        if self.point.geometry().a_ring_dimension() != D
            || self.point.source_num_vars() != source_num_vars
        {
            return Err(AkitaError::InvalidInput(
                "coefficient-packing plan disagrees with source geometry or arity".into(),
            ));
        }
        Ok(())
    }
}

/// Canonical base-field coordinates for one claim's packed partials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubringCoefficientPackingPartials<F: Field> {
    geometry: SubringCoefficientPackingGeometry,
    num_live_blocks: usize,
    coordinates: Vec<F>,
}

impl<F: Field> SubringCoefficientPackingPartials<F> {
    /// Build a typed partial buffer after checking its exact physical width.
    pub(crate) fn new(
        geometry: SubringCoefficientPackingGeometry,
        num_live_blocks: usize,
        coordinates: Vec<F>,
    ) -> Result<Self, AkitaError> {
        let expected = num_live_blocks
            .checked_mul(geometry.partial_base_field_width())
            .ok_or_else(|| {
                AkitaError::InvalidInput("coefficient-packing partial length overflow".into())
            })?;
        if num_live_blocks == 0 || coordinates.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: coordinates.len(),
            });
        }
        Ok(Self {
            geometry,
            num_live_blocks,
            coordinates,
        })
    }

    /// Checked packing geometry.
    pub(crate) const fn geometry(&self) -> SubringCoefficientPackingGeometry {
        self.geometry
    }

    /// Number of live blocks represented by this claim.
    pub(crate) const fn num_live_blocks(&self) -> usize {
        self.num_live_blocks
    }

    /// Canonical `[block][extension coordinate][subring coefficient]` data.
    pub(crate) fn coordinates(&self) -> &[F] {
        &self.coordinates
    }
}

impl<F: Field> AsRef<[F]> for SubringCoefficientPackingPartials<F> {
    fn as_ref(&self) -> &[F] {
        self.coordinates()
    }
}

/// Decompose + challenge-fold parameters for one opening.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DecomposeFoldPlan<'a> {
    /// Sparse fold challenges, outermost first.
    pub challenges: &'a [SparseChallenge],
    /// Number of ring-element positions in each block.
    pub num_positions_per_block: usize,
    /// Number of balanced digits.
    pub num_digits: usize,
    /// Logarithm of the gadget basis.
    pub log_basis: u32,
}

/// Batched decompose + fold parameters at one opening point.
///
/// A representation may keep a fast batched kernel rather than folding each
/// polynomial independently and aggregating later.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DecomposeFoldBatchPlan<'a> {
    /// Sparse-challenge batched fold.
    Sparse {
        /// Sparse fold challenges, outermost first.
        challenges: &'a [SparseChallenge],
        /// Number of ring-element positions in each block.
        num_positions_per_block: usize,
        /// Number of balanced digits.
        num_digits: usize,
        /// Logarithm of the gadget basis.
        log_basis: u32,
    },
    /// Chunk-aware sparse fold with full challenges and canonical ranges.
    SparseChunked {
        /// Full claim-major challenge carrier.
        challenges: &'a Challenges,
        /// Exact canonical ranges for one claim.
        chunk_ranges: &'a [Range<usize>],
        /// Number of ring-element positions in each block.
        num_positions_per_block: usize,
        /// Number of balanced digits.
        num_digits: usize,
        /// Logarithm of the gadget basis.
        log_basis: u32,
    },
}

impl DecomposeFoldBatchPlan<'_> {
    pub(crate) fn scalar_params(self) -> (usize, usize, u32) {
        match self {
            Self::Sparse {
                num_positions_per_block,
                num_digits,
                log_basis,
                ..
            }
            | Self::SparseChunked {
                num_positions_per_block,
                num_digits,
                log_basis,
                ..
            } => (num_positions_per_block, num_digits, log_basis),
        }
    }

    /// Validate the challenge layout against every source's live-block extent.
    pub(crate) fn validate_uniform_batch(
        self,
        live_blocks: impl ExactSizeIterator<Item = usize>,
    ) -> Result<usize, AkitaError> {
        let num_polys = live_blocks.len();
        if num_polys == 0 {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold requires at least one polynomial".to_string(),
            ));
        }
        let (challenges, declared_challenges_per_poly, num_positions_per_block) = match self {
            Self::Sparse {
                challenges,
                num_positions_per_block,
                ..
            } => (challenges, None, num_positions_per_block),
            Self::SparseChunked {
                challenges,
                num_positions_per_block,
                ..
            } => (
                challenges.as_slice(),
                Some(challenges.num_live_blocks_per_claim()),
                num_positions_per_block,
            ),
        };
        if challenges.is_empty() || num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold requires positive block geometry".to_string(),
            ));
        }
        if !challenges.len().is_multiple_of(num_polys) {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold challenge count is not divisible by polynomial count"
                    .to_string(),
            ));
        }
        let challenges_per_poly =
            declared_challenges_per_poly.unwrap_or_else(|| challenges.len() / num_polys);
        let expected =
            akita_error::checked::product([num_polys, challenges_per_poly]).ok_or_else(|| {
                AkitaError::InvalidInput("batched decompose_fold challenge count overflow".into())
            })?;
        if challenges.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: challenges.len(),
            });
        }
        if live_blocks
            .into_iter()
            .any(|count| count != challenges_per_poly)
        {
            return Err(AkitaError::InvalidInput(
                "batched decompose_fold sources have different live-block extents".into(),
            ));
        }
        Ok(challenges_per_poly)
    }
}

/// Canonical response geometry for one validated fold probe.

#[derive(Clone, Copy, Debug)]
pub(crate) struct RingSwitchRelationPlan {
    /// Number of D-side cyclic rows to produce.
    pub n_d: usize,
    /// Number of B-side cyclic rows to produce.
    pub n_b: usize,
    /// Number of A-side quotient rows to produce.
    pub n_a: usize,
    /// Logarithm of the D/opening gadget basis used to produce `e_hat`.
    pub log_basis_open: u32,
    /// Logarithm of the B/outer gadget basis used to produce `t_hat`.
    pub log_basis_outer: u32,
}

/// Scalar operation parameters for an inner Ajtai commit.
///
/// Polynomial data lives in a request-compiled commitment representation; this
/// plan carries only the shape parameters the selected operation needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommitInnerPlan {
    /// Runtime ring dimension used by the A-side commitment.
    pub ring_dimension: usize,
    /// Number of live source blocks committed by this request.
    pub num_live_blocks: usize,
    /// Number of A rows to produce.
    pub n_a: usize,
    /// Number of ring-element positions in each root block.
    pub num_positions_per_block: usize,
    /// Number of balanced digits used for the A-side commit.
    pub num_digits_inner: usize,
    /// Logarithm of the committed source-witness gadget basis.
    pub log_basis_inner: u32,
}

impl CommitInnerPlan {
    /// Build inner-commit parameters from a frozen standalone precommit profile.
    pub(crate) fn from_profile(profile: &akita_types::GroupCommitPhaseParams) -> Self {
        Self {
            ring_dimension: profile.inner.matrix.ring_dimension(),
            num_live_blocks: profile.blocks.live_blocks,
            n_a: profile.inner.matrix.output_rank(),
            num_positions_per_block: profile.blocks.positions_per_block,
            num_digits_inner: profile.inner.digits.num_digits,
            log_basis_inner: profile.inner.digits.log_basis,
        }
    }
}

/// Evaluation-trace inputs that are not fixed by the accepted-fold manifest.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FoldEvaluationTracePlan<'a, F: Field> {
    multiplier_point: &'a RingMultiplierOpeningPoint<F>,
    num_positions_per_block: usize,
    depth_commit: usize,
    log_basis_inner: u32,
}

impl<F: Field> FoldEvaluationTracePlan<'_, F> {
    pub(crate) const fn multiplier_point(&self) -> &RingMultiplierOpeningPoint<F> {
        self.multiplier_point
    }
    pub(crate) const fn num_positions_per_block(&self) -> usize {
        self.num_positions_per_block
    }
    pub(crate) const fn depth_commit(&self) -> usize {
        self.depth_commit
    }
    pub(crate) const fn log_basis_inner(&self) -> u32 {
        self.log_basis_inner
    }
}

/// Akita-validated inputs for relation work over an accepted fold.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ValidatedFoldRelationPlan<'a, F: Field> {
    n_a: usize,
    log_basis_open: u32,
    log_basis_outer: u32,
    evaluation_trace: Option<FoldEvaluationTracePlan<'a, F>>,
}

impl<'a, F: Field> ValidatedFoldRelationPlan<'a, F> {
    pub(crate) fn new(
        opening_method: akita_types::OpeningMethod,
        n_a: usize,
        log_basis_open: u32,
        log_basis_outer: u32,
        evaluation_trace: Option<(&'a RingMultiplierOpeningPoint<F>, usize, usize, u32)>,
    ) -> Result<Self, AkitaError> {
        let expects_trace = matches!(opening_method, akita_types::OpeningMethod::EvaluationTrace);
        if n_a == 0 || expects_trace != evaluation_trace.is_some() {
            return Err(AkitaError::InvalidInput(
                "fold relation plan disagrees with its opening family".into(),
            ));
        }
        let evaluation_trace = evaluation_trace
            .map(
                |(multiplier_point, num_positions_per_block, depth_commit, log_basis_inner)| {
                    if num_positions_per_block == 0 || depth_commit == 0 {
                        return Err(AkitaError::InvalidInput(
                            "fold relation plan has empty evaluation-trace geometry".into(),
                        ));
                    }
                    Ok(FoldEvaluationTracePlan {
                        multiplier_point,
                        num_positions_per_block,
                        depth_commit,
                        log_basis_inner,
                    })
                },
            )
            .transpose()?;
        Ok(Self {
            n_a,
            log_basis_open,
            log_basis_outer,
            evaluation_trace,
        })
    }

    pub(crate) const fn n_a(&self) -> usize {
        self.n_a
    }
    pub(crate) const fn log_basis_open(&self) -> u32 {
        self.log_basis_open
    }
    pub(crate) const fn log_basis_outer(&self) -> u32 {
        self.log_basis_outer
    }
    pub(crate) const fn evaluation_trace(&self) -> Option<FoldEvaluationTracePlan<'a, F>> {
        self.evaluation_trace
    }
}
