use crate::api::CommitmentWithHint;
use crate::compute::{RootOpeningSource, RootPolyShape};
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;
use akita_types::{
    AkitaCommitmentHint, FlatRingVec, LevelParams, MRowLayout, OpeningClaims, RingCommitment,
};

/// Prover opening input: public claims plus prover-only hints and polynomials.
#[derive(Debug, Clone)]
pub struct ProverOpeningData<'a, PointF: Clone, P, CommitF: FieldCore, const D: usize> {
    opening_claims: OpeningClaims<'a, PointF, RingCommitment<CommitF, D>>,
    hints: Vec<AkitaCommitmentHint<CommitF, D>>,
    polynomials: Vec<&'a [&'a P]>,
}

impl<'a, PointF: Clone, P, CommitF: FieldCore, const D: usize>
    ProverOpeningData<'a, PointF, P, CommitF, D>
{
    /// Bundle public claims with matching prover hints and polynomial groups.
    pub fn new(
        opening_claims: OpeningClaims<'a, PointF, RingCommitment<CommitF, D>>,
        hints: Vec<AkitaCommitmentHint<CommitF, D>>,
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
        P: RootPolyShape<PolyF, D> + RootOpeningSource<PolyF, D>,
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
        P: RootPolyShape<PolyF, D>,
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

    /// Public claims carried by this prover input.
    pub fn opening_claims(&self) -> &OpeningClaims<'a, PointF, RingCommitment<CommitF, D>> {
        &self.opening_claims
    }

    /// Prover-only hints, one per polynomial group.
    pub fn hints(&self) -> &[AkitaCommitmentHint<CommitF, D>] {
        &self.hints
    }

    /// Borrow one prover hint.
    pub fn group_hint(&self, index: usize) -> Result<&AkitaCommitmentHint<CommitF, D>, AkitaError> {
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
    pub fn commitments(&self) -> Vec<&RingCommitment<CommitF, D>> {
        self.opening_claims
            .groups()
            .iter()
            .map(|group| group.commitment())
            .collect()
    }

    /// Absorb the normalized batch shape, commitments, and shared point.
    pub fn append_to_transcript<T>(&self, transcript: &mut T) -> Result<(), AkitaError>
    where
        CommitF: CanonicalField,
        PointF: ExtField<CommitF>,
        T: Transcript<CommitF>,
    {
        self.opening_claims
            .append_to_transcript::<CommitF, T>(transcript)
    }

    /// Return the only group when the current single-group path applies.
    pub fn single_group_polys(&self) -> Option<&'a [&'a P]> {
        self.polynomials
            .first()
            .copied()
            .filter(|_| self.polynomials.len() == 1)
    }

    /// Borrow root fold commitment rows in the scheduled M-row commitment order.
    pub(crate) fn fold_commitment(
        &self,
        params: &LevelParams,
    ) -> Result<FlatRingVec<CommitF>, AkitaError> {
        let opening_batch = self.opening_claims.layout()?;
        if self.opening_claims.num_groups() != opening_batch.num_groups() {
            return Err(AkitaError::InvalidInput(
                "fold commitment group count mismatch".to_string(),
            ));
        }
        let mut group_order = (0..opening_batch.num_groups())
            .map(|group_index| {
                let range = params.root_commitment_row_range(
                    &opening_batch,
                    group_index,
                    MRowLayout::WithDBlock,
                )?;
                Ok((range.start, range.len(), group_index))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        group_order.sort_by_key(|(start, _, _)| *start);

        let mut rows = Vec::new();
        for (_, expected, group_index) in group_order {
            let commitment = self.opening_claims.group_commitment(group_index)?;
            if commitment.u.len() != expected {
                return Err(AkitaError::InvalidInput(
                    "fold commitment row count mismatch".to_string(),
                ));
            }
            rows.extend_from_slice(&commitment.u);
        }
        Ok(FlatRingVec::from_ring_elems(&rows))
    }

    /// Preserve grouping metadata while replacing the flat polynomial stream.
    pub(crate) fn regroup_polynomial_refs<'b, Q>(
        self,
        polynomials: &'b [&'b Q],
    ) -> Result<ProverOpeningData<'b, PointF, Q, CommitF, D>, AkitaError>
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
        commitment: CommitmentWithHint<CommitF, D>,
    ) -> Result<Self, AkitaError>
    where
        PointF: FieldCore,
    {
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(recursive_num_vars, PointF::zero());
        let point_vars =
            akita_types::PointVariableSelection::prefix(recursive_num_vars, recursive_num_vars)?;
        let claims = akita_types::PolynomialGroupClaims::new(
            point_vars,
            vec![PointF::zero()],
            commitment.0,
        )?;
        ProverOpeningData::new(
            OpeningClaims::from_groups(padded_point, vec![claims])?,
            vec![commitment.1],
            vec![polynomials],
        )
    }
}
