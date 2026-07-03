use crate::api::CommitmentWithHint;
use crate::compute::RootPolyMeta;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;
use akita_types::{
    padded_scalar_batch_num_vars, validate_scalar_point_matches_poly_arity, AkitaCommitmentHint,
    Commitment, OpeningClaims, OpeningClaimsLayout, PointVariableSelection, PolynomialGroupClaims,
    PolynomialGroupLayout, RingVec,
};

/// Prover opening input: public claims plus prover-only hints and polynomials.
#[derive(Debug, Clone)]
pub struct ProverOpeningData<'a, PointF: Clone, P, CommitF: FieldCore> {
    opening_claims: OpeningClaims<'a, PointF, Commitment<CommitF>>,
    hints: Vec<AkitaCommitmentHint<CommitF>>,
    polynomials: Vec<&'a [&'a P]>,
}

impl<'a, PointF: Clone, P, CommitF: FieldCore> ProverOpeningData<'a, PointF, P, CommitF> {
    /// Bundle public claims with matching prover hints and polynomial groups.
    pub fn new(
        opening_claims: OpeningClaims<'a, PointF, Commitment<CommitF>>,
        hints: Vec<AkitaCommitmentHint<CommitF>>,
        polynomials: Vec<&'a [&'a P]>,
    ) -> Result<Self, AkitaError> {
        let data = Self {
            opening_claims,
            hints,
            polynomials,
        };
        data.check_alignment()?;
        Ok(data)
    }

    fn check_alignment(&self) -> Result<(), AkitaError> {
        if self.opening_claims.num_groups() != self.hints.len()
            || self.opening_claims.num_groups() != self.polynomials.len()
        {
            return Err(AkitaError::InvalidInput(
                "prover opening data group counts are misaligned".to_string(),
            ));
        }
        for group_index in 0..self.opening_claims.num_groups() {
            let expected = self.opening_claims.group_evaluations(group_index)?.len();
            let actual = self
                .polynomials
                .get(group_index)
                .ok_or_else(|| AkitaError::InvalidInput("missing polynomial group".to_string()))?
                .len();
            if actual != expected {
                return Err(AkitaError::InvalidInput(
                    "prover opening data polynomial/evaluation counts are misaligned".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Validate alignment and root polynomial shape.
    pub fn validate<PolyF>(&self) -> Result<(), AkitaError>
    where
        PolyF: FieldCore,
        P: RootPolyMeta<PolyF>,
    {
        self.check_alignment()?;
        let num_vars = self.num_vars::<PolyF>()?;
        if self.opening_claims.num_vars() != num_vars {
            return Err(AkitaError::InvalidInput(format!(
                "opening point length {} does not match padded batch domain {num_vars}",
                self.opening_claims.num_vars()
            )));
        }
        Ok(())
    }

    /// Largest natural root arity across all polynomial groups.
    pub fn num_vars<PolyF>(&self) -> Result<usize, AkitaError>
    where
        PolyF: FieldCore,
        P: RootPolyMeta<PolyF>,
    {
        self.polynomials
            .iter()
            .flat_map(|group| group.iter().map(|poly| poly.num_vars()))
            .max()
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "prover opening data requires at least one polynomial".to_string(),
                )
            })
    }

    /// Shared opening point.
    pub fn point(&self) -> &[PointF] {
        self.opening_claims.point()
    }

    /// Public claims carried by this prover input.
    pub fn opening_claims(&self) -> &OpeningClaims<'a, PointF, Commitment<CommitF>> {
        &self.opening_claims
    }

    /// Layout-only opening geometry derived from prover polynomials.
    pub fn opening_layout<PolyF>(&self) -> Result<OpeningClaimsLayout, AkitaError>
    where
        PolyF: FieldCore,
        P: RootPolyMeta<PolyF>,
    {
        let padded_num_vars = padded_scalar_batch_num_vars(
            self.polynomials
                .iter()
                .flat_map(|group| group.iter().map(|poly| poly.num_vars())),
        )?;
        validate_scalar_point_matches_poly_arity(self.point().len(), padded_num_vars)?;
        let group_sizes = self
            .polynomials
            .iter()
            .map(|group| group.len())
            .collect::<Vec<_>>();
        OpeningClaimsLayout::from_group_sizes(padded_num_vars, &group_sizes)
    }

    /// Prover-only hints, one per polynomial group.
    pub fn hints(&self) -> &[AkitaCommitmentHint<CommitF>] {
        &self.hints
    }

    /// Borrow one prover hint.
    pub fn group_hint(&self, index: usize) -> Result<&AkitaCommitmentHint<CommitF>, AkitaError> {
        self.hints.get(index).ok_or(AkitaError::InvalidProof)
    }

    /// Borrow one polynomial group.
    pub fn group_polys(&self, index: usize) -> Result<&'a [&'a P], AkitaError> {
        self.polynomials
            .get(index)
            .copied()
            .ok_or(AkitaError::InvalidProof)
    }

    /// Polynomials flattened in canonical claim order.
    pub fn flat_polys(&self) -> Vec<&'a P> {
        self.polynomials
            .iter()
            .flat_map(|group| group.iter().copied())
            .collect()
    }

    /// Commitments in commitment-group order.
    pub fn commitments(&self) -> Vec<&Commitment<CommitF>> {
        self.opening_claims
            .groups()
            .iter()
            .map(PolynomialGroupClaims::commitment)
            .collect()
    }

    /// Absorb the normalized batch shape, commitments, and shared point.
    pub fn append_to_transcript<T>(
        &self,
        ring_dim: usize,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        CommitF: CanonicalField,
        PointF: ExtField<CommitF>,
        P: RootPolyMeta<CommitF>,
        T: Transcript<CommitF>,
    {
        let layout = self.opening_layout::<CommitF>()?;
        layout.append_to_transcript::<CommitF, T>(transcript)?;
        for commitment in self.commitments() {
            commitment.append_to_transcript(
                akita_transcript::labels::ABSORB_COMMITMENT,
                ring_dim,
                transcript,
            )?;
        }
        for coord in self.point() {
            akita_transcript::append_ext_field::<CommitF, PointF, T>(
                transcript,
                akita_transcript::labels::ABSORB_EVALUATION_CLAIMS,
                coord,
            );
        }
        Ok(())
    }

    /// Return the only group when the current single-group path applies.
    pub fn single_group_polys(&self) -> Option<&'a [&'a P]> {
        self.polynomials
            .first()
            .copied()
            .filter(|_| self.polynomials.len() == 1)
    }

    /// Borrow the current single-group fold batch's commitment rows as flat proof storage.
    pub(crate) fn single_fold_commitment(&self) -> Result<RingVec<CommitF>, AkitaError> {
        let commitment = self
            .opening_claims
            .single_group_commitment()
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "multi-group fold proving is not supported yet".to_string(),
                )
            })?;
        Ok(commitment.rows().clone())
    }

    /// Preserve grouping metadata while replacing the flat polynomial stream.
    pub(crate) fn regroup_polynomial_refs<'b, Q>(
        self,
        polynomials: &'b [&'b Q],
    ) -> Result<ProverOpeningData<'b, PointF, Q, CommitF>, AkitaError>
    where
        'a: 'b,
    {
        let mut input_offset = 0usize;
        let mut regrouped = Vec::with_capacity(self.polynomials.len());
        for group in self.polynomials {
            let group_len = group.len();
            let input_end = input_offset.checked_add(group_len).ok_or_else(|| {
                AkitaError::InvalidInput("fold input group offset overflow".to_string())
            })?;
            let replacement_polynomials =
                polynomials.get(input_offset..input_end).ok_or_else(|| {
                    AkitaError::InvalidInput("fold input group shape mismatch".to_string())
                })?;
            regrouped.push(replacement_polynomials);
            input_offset = input_end;
        }
        if input_offset != polynomials.len() {
            return Err(AkitaError::InvalidInput(
                "fold input group coverage mismatch".to_string(),
            ));
        }
        ProverOpeningData::new(self.opening_claims, self.hints, regrouped)
    }

    /// Build the single-claim batch used by recursive suffix fold levels.
    pub(crate) fn new_suffix(
        opening_point: &[PointF],
        recursive_num_vars: usize,
        polynomials: &'a [&'a P],
        commitment: CommitmentWithHint<CommitF>,
    ) -> Result<Self, AkitaError>
    where
        PointF: FieldCore,
    {
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(recursive_num_vars, PointF::zero());
        let point_vars = PointVariableSelection::prefix(recursive_num_vars, recursive_num_vars)?;
        let claims = PolynomialGroupClaims::new(point_vars, vec![PointF::zero()], commitment.0)?;
        ProverOpeningData::new(
            OpeningClaims::from_groups(padded_point, vec![claims])?,
            vec![commitment.1],
            vec![polynomials],
        )
    }
}
