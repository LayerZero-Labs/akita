use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::{AkitaError, FieldCore};
use akita_types::{
    CommittedGroupParams, CommittedGroupProfile, PreparedSubringCoefficientPackingPoint,
    SubfieldMultiplierOpeningPoint, SubringCoefficientPackingGeometry,
};

// ===========================================================================
// Open, source-typed operation boundary
//
// The prover compute boundary is open by source type `S` instead of closed
// over Akita's built-in representation plan shapes:
//
// - operation kernels (`RootCommitKernel`, `OpeningFoldKernel`, ...) take the
//   borrowed representation view as a generic type parameter `S`, so a
//   downstream crate can define its own local view type and implement the
//   relevant kernel for `CpuBackend` without modifying an Akita-owned enum;
// - root polynomials expose those views through capability traits
//   (`RootCommitSource`, `RootOpeningSource`, ...) whose associated view types
//   become the `S` a kernel runs over;
// - a prover run threads operation *contexts* (`OperationCtx`) bundled into a
//   `ProverComputeStack`, each carrying a backend plus its validated prepared
//   setup, so commitment / opening / tensor / ring-switch work can run on
//   independent backends while the protocol still sees canonical Akita outputs.
//
// Built-in representations implement these kernels in their backend modules.
// Shared arithmetic may remain as private CPU helpers, but protocol and public
// extension boundaries use the source-typed kernels directly.
// ===========================================================================

/// Scalar operation parameters for an inner Ajtai commit.
///
/// The polynomial data lives in the borrowed commit source view (`S`); this
/// plan carries only the shape parameters the kernel needs to size its work.
#[derive(Debug, Clone, Copy)]
pub struct CommitInnerPlan {
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
    /// Build inner-commit parameters from a validated commitment layout.
    pub fn from_level(params: &CommittedGroupParams) -> Self {
        Self {
            n_a: params.inner_commit_matrix.output_rank(),
            num_positions_per_block: params.num_positions_per_block,
            num_digits_inner: params.num_digits_inner,
            log_basis_inner: params.log_basis_inner,
        }
    }

    /// Build inner-commit parameters from a frozen standalone precommit profile.
    pub fn from_profile(profile: &CommittedGroupProfile) -> Self {
        Self {
            n_a: profile.inner_commit_matrix.output_rank(),
            num_positions_per_block: profile.num_positions_per_block,
            num_digits_inner: profile.num_digits_inner,
            log_basis_inner: profile.log_basis_inner,
        }
    }
}

/// Fold parameters for a fused evaluate-and-fold opening.
///
/// For source rings `p[b, j]`, position multipliers `a[j]`, and outer
/// multipliers `s[b]`, every backend returns
/// `e[b] = sum_j p[b, j] * a[j]` and `y = sum_b e[b] * s[b]`.
/// The folded `e` rows become the next recursive relation witness.
///
/// The base/subfield split keeps degree-one scalar folds direct while proper
/// extension folds retain their compact subfield coordinates.
#[derive(Debug, Clone, Copy)]
pub enum OpeningFoldPlan<'a, F: FieldCore> {
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

impl<F: FieldCore> OpeningFoldPlan<'_, F> {
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

/// Fused evaluate-and-fold output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningFoldOutput<F: FieldCore, const D: usize> {
    /// Evaluation of the polynomial at the opening point.
    pub eval: CyclotomicRing<F, D>,
    /// Folded witness rows in ring form.
    pub folded: Vec<CyclotomicRing<F, D>>,
}

/// Checked scalar inputs for one coefficient-packing projection batch.
///
/// Source data stays in the representation-specific borrowed batch view. The
/// output layout is fixed by `geometry` as
/// `[claim][block][extension coordinate][subring coefficient]`.
#[derive(Debug, Clone, Copy)]
pub struct SubringCoefficientPackingPlan<'a, E: FieldCore> {
    /// Canonically split public opening point.
    pub point: &'a PreparedSubringCoefficientPackingPoint<E>,
}

impl<E: FieldCore> SubringCoefficientPackingPlan<'_, E> {
    /// Validate the plan at a source kernel boundary.
    pub fn validate<const D: usize>(&self, source_num_vars: usize) -> Result<(), AkitaError> {
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
pub struct SubringCoefficientPackingPartials<F: FieldCore> {
    geometry: SubringCoefficientPackingGeometry,
    num_live_blocks: usize,
    coordinates: Vec<F>,
}

impl<F: FieldCore> SubringCoefficientPackingPartials<F> {
    /// Build a typed partial buffer after checking its exact physical width.
    pub fn new(
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
    pub const fn geometry(&self) -> SubringCoefficientPackingGeometry {
        self.geometry
    }

    /// Number of live blocks represented by this claim.
    pub const fn num_live_blocks(&self) -> usize {
        self.num_live_blocks
    }

    /// Canonical `[block][extension coordinate][subring coefficient]` data.
    pub fn coordinates(&self) -> &[F] {
        &self.coordinates
    }
}

/// Decompose + challenge-fold parameters for one opening.
#[derive(Debug, Clone, Copy)]
pub struct DecomposeFoldPlan<'a> {
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
pub enum DecomposeFoldBatchPlan<'a> {
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
}

/// Scalar operation parameters for the fused ring-switch relation rows.
///
/// The decomposed witness data (`e_hat`, `t_hat`, centered `z` segment) and the
/// centered infinity norm live in the borrowed relation source view (`S`).
#[derive(Debug, Clone, Copy)]
pub struct RingSwitchRelationPlan {
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
