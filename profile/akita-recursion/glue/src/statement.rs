//! Verifier-facing statement and proof inputs carried by the recursion blob.

use crate::AkitaJoltCase;
use akita_error::AkitaError;
use akita_types::{
    AkitaBatchedProof, AkitaBatchedProofShape, AkitaVerifierSetup, CommittedGroup,
    GroupBatchStatement, OpeningClaims, OpeningScheduleSelection, PolynomialGroupClaims,
};
use jolt_field::Field;

/// One ordered commitment group carried in a multi-group verifier statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaJoltOpeningGroup<F: Field, E: Field = F> {
    /// Opening point for this group.
    pub opening_point: Vec<E>,
    /// Claimed evaluations, one per polynomial in this group.
    pub openings: Vec<E>,
    /// Commitment and its frozen algebraic profile.
    pub commitment: CommittedGroup<F>,
}

/// Bundled verifier inputs that travel from the host to the Jolt guest.
///
/// `D` is the cyclotomic root-envelope dimension pinned by the host config.
/// The guest must use the same value to reject blobs built for a different
/// verifier monomorphization; per-level dimensions remain schedule-owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaJoltInputs<F: Field, const D: usize, E: Field = F> {
    /// Exact CI case and verifier monomorphization represented by this blob.
    pub case: AkitaJoltCase,
    /// Domain label both prover and verifier transcripts were initialized with.
    pub transcript_domain: Vec<u8>,
    /// Number of variables of the public polynomial (informational; sanity).
    pub num_vars: u64,
    /// Opening point in the multilinear basis.
    pub opening_point: Vec<E>,
    /// Claimed opening values for the final/new group.
    pub openings: Vec<E>,
    /// Earlier commitment groups in transcript order.
    pub precommitted_groups: Vec<AkitaJoltOpeningGroup<F, E>>,
    /// Exact generated schedule row accepted for this opening batch.
    pub schedule_selection: OpeningScheduleSelection,
    /// Final/new committed-poly group.
    pub commitment: CommittedGroup<F>,
    /// Expanded verifier setup (matrix prefix usable by the verifier kernel).
    pub verifier_setup: AkitaVerifierSetup<F>,
    /// Proof shape used to decode `proof` after schedule-shape admission.
    pub proof_shape: AkitaBatchedProofShape,
    /// The Akita batched proof itself.
    pub proof: AkitaBatchedProof<F, E>,
}

impl<F: Field, const D: usize, E: Field> AkitaJoltInputs<F, D, E> {
    /// Build the ordered verifier claim represented by this blob.
    pub fn verifier_statement<'a>(&'a self) -> Result<GroupBatchStatement<'a, E, F>, AkitaError> {
        let num_vars = usize::try_from(self.num_vars).map_err(|_| {
            AkitaError::InvalidInput("recursion blob num_vars does not fit usize".to_string())
        })?;
        if num_vars != self.opening_point.len() {
            return Err(AkitaError::InvalidInput(
                "final recursion opening point does not cover all variables".to_string(),
            ));
        }
        let mut groups = Vec::with_capacity(self.precommitted_groups.len() + 1);
        for group in &self.precommitted_groups {
            groups.push(PolynomialGroupClaims::new(
                group.opening_point.as_slice(),
                group.openings.clone(),
                &group.commitment,
            )?);
        }
        groups.push(PolynomialGroupClaims::new(
            self.opening_point.as_slice(),
            self.openings.clone(),
            &self.commitment,
        )?);
        let claims = OpeningClaims::from_groups(groups)?;
        GroupBatchStatement::new(self.schedule_selection, claims)
    }
}
