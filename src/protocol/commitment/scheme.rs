//! Prover-side commitment-scheme trait surface for Hachi protocol code.

use crate::protocol::hachi_poly_ops::HachiPolyOps;
use crate::{CanonicalField, FieldCore};
use akita_field::HachiError;
use akita_transcript::Transcript;
use akita_types::BasisMode;
use akita_verifier::{CommitmentVerifier, OpeningPoints};

/// One committed polynomial group opened at an opening point.
///
/// The `polynomials` slice is the exact group committed together by
/// `CommitmentProver::commit`; `commitment` and `hint` are the corresponding
/// outputs for that group.
#[derive(Debug, Clone)]
pub struct CommittedPolynomials<'a, P, C, H> {
    /// Polynomials that were committed together as one group.
    pub polynomials: &'a [P],
    /// Commitment for `polynomials`.
    pub commitment: &'a C,
    /// Prover-side hint for `commitment`.
    pub hint: H,
}

/// Batched prover input grouped by opening point.
pub type ProverClaims<'a, F, P, C, H> =
    Vec<(OpeningPoints<'a, F>, Vec<CommittedPolynomials<'a, P, C, H>>)>;

/// Prover-side commitment-scheme interface used by Hachi protocol code.
///
/// Generic over field `F` and cyclotomic ring degree `D`.
/// Caller-provided root polynomials are provided as `impl HachiPolyOps<F, D>`.
/// Recursive `w` witnesses are internal to the protocol and no longer modelled
/// through this trait.
pub trait CommitmentProver<F, const D: usize>: CommitmentVerifier<F, D>
where
    F: FieldCore + CanonicalField,
{
    /// Prover setup parameters.
    type ProverSetup: Clone + Send + Sync;
    /// Prover-side hint produced for one commitment group.
    type CommitHint: Clone + Send + Sync;
    /// Build prover setup for maximum polynomial dimension, batch capacity,
    /// and distinct opening-point count.
    ///
    /// # Panics
    ///
    /// Panics if internal setup fails (programming error, not adversarial input).
    fn setup_prover(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Self::ProverSetup;

    /// Derive verifier setup from prover setup.
    fn setup_verifier(
        setup: &Self::ProverSetup,
    ) -> <Self as CommitmentVerifier<F, D>>::VerifierSetup;

    /// Commit to polynomials.
    ///
    /// The root layout is derived automatically from the polynomial dimension.
    /// All polynomials in `polys` are aggregated into one commitment. Callers
    /// that need multiple commitments should call this method repeatedly, once
    /// per commitment group.
    ///
    /// # Errors
    ///
    /// Returns an error when setup/parameter constraints are not satisfied.
    fn commit<P: HachiPolyOps<F, D>>(
        polys: &[P],
        setup: &Self::ProverSetup,
    ) -> Result<(Self::Commitment, Self::CommitHint), HachiError>;

    /// Commit several polynomial groups using one shared batched shape.
    ///
    /// The outer `poly_groups` slice indexes commitment groups. The
    /// `point_group_sizes` slice describes how those commitment groups will be
    /// distributed across opening points in a later batched proof.
    ///
    /// Implementations may override this to choose a root layout from the full
    /// grouped batch shape. The default preserves the primitive per-group
    /// behavior by calling [`commit`](Self::commit) once per group.
    ///
    /// # Errors
    ///
    /// Returns an error when the group shape is malformed or when any
    /// per-group commitment fails.
    #[allow(clippy::type_complexity)]
    fn batched_commit<P: HachiPolyOps<F, D>>(
        poly_groups: &[&[P]],
        point_group_sizes: &[usize],
        setup: &Self::ProverSetup,
    ) -> Result<(Vec<Self::Commitment>, Vec<Self::CommitHint>), HachiError> {
        if poly_groups.is_empty() {
            return Err(HachiError::InvalidInput(
                "batched_commit requires at least one commitment group".to_string(),
            ));
        }
        if point_group_sizes.is_empty() || point_group_sizes.contains(&0) {
            return Err(HachiError::InvalidInput(
                "batched_commit requires nonempty point group sizes".to_string(),
            ));
        }
        let total_groups = point_group_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                HachiError::InvalidInput("batched_commit group count overflow".to_string())
            })
        })?;
        if total_groups != poly_groups.len() {
            return Err(HachiError::InvalidInput(
                "batched_commit point group sizes do not match commitment groups".to_string(),
            ));
        }

        let mut commitments = Vec::with_capacity(poly_groups.len());
        let mut hints = Vec::with_capacity(poly_groups.len());
        for group in poly_groups {
            let (commitment, hint) = Self::commit(group, setup)?;
            commitments.push(commitment);
            hints.push(hint);
        }
        Ok((commitments, hints))
    }

    /// Produce a fused batched opening proof for one or more opening points.
    ///
    /// The outer vector indexes opening points. Each point carries the
    /// committed polynomial groups opened at that point.
    ///
    /// A singleton opening is the 1x1 special case (one polynomial, one
    /// commitment group, one opening point). Same-point batching is the
    /// special case `opening_points.len() == 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if any opening point is invalid or proof generation
    /// fails.
    #[allow(clippy::too_many_arguments)]
    fn batched_prove<'a, T: Transcript<F>, P: HachiPolyOps<F, D>>(
        setup: &Self::ProverSetup,
        claims: ProverClaims<'a, F, P, Self::Commitment, Self::CommitHint>,
        transcript: &mut T,
        basis: BasisMode,
    ) -> Result<Self::BatchedProof, HachiError>;
}
