use crate::api::CommitmentWithHint;
use crate::backend::RecursiveFoldSource;
use crate::compute::RootPolyMeta;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;
use akita_types::{
    AkitaCommitmentHint, Commitment, LevelParams, OpeningClaims, OpeningClaimsLayout,
    PointVariableSelection, PolynomialGroupClaims, PolynomialGroupLayout, RelationMatrixRowLayout,
    RingVec, SetupPrefixSlot,
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
        let layout = self.opening_layout::<PolyF>()?;
        let max_num_vars = layout.max_num_vars();
        if self.opening_claims.num_vars() != max_num_vars {
            return Err(AkitaError::InvalidInput(format!(
                "opening point length {} does not match max group arity {max_num_vars}",
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
        let mut groups = Vec::with_capacity(self.polynomials.len());
        for (group_index, group) in self.polynomials.iter().enumerate() {
            let first_poly = group.first().ok_or_else(|| {
                AkitaError::InvalidInput("opening polynomial groups must be nonempty".to_string())
            })?;
            let group_num_vars = first_poly.num_vars();
            if group.iter().any(|poly| poly.num_vars() != group_num_vars) {
                return Err(AkitaError::InvalidInput(
                    "opening polynomial groups must have uniform arity".to_string(),
                ));
            }
            let point_vars = self.opening_claims.group_point_vars(group_index)?;
            if point_vars.num_vars() != group_num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: group_num_vars,
                    actual: point_vars.num_vars(),
                });
            }
            groups.push(PolynomialGroupLayout::new(group_num_vars, group.len()));
        }
        OpeningClaimsLayout::from_groups(groups)
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
        // Bind each group's active arity rather than collapsing every group to
        // the shared padded domain. `opening_layout` also validates that the
        // public point-variable selection matches the prover's polynomial shape,
        // which keeps this byte-identical to the verifier's `claims.layout()`
        // absorb for well-formed inputs.
        let layout = self.opening_layout::<CommitF>()?;
        layout.append_batch_shape_to_transcript::<CommitF, T>(transcript)?;
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

    /// Borrow root fold commitment rows in the scheduled M-row commitment order.
    pub(crate) fn fold_commitment(
        &self,
        params: &LevelParams,
    ) -> Result<RingVec<CommitF>, AkitaError> {
        let opening_batch = self.opening_claims.layout()?;
        if self.opening_claims.num_groups() != opening_batch.num_groups() {
            return Err(AkitaError::InvalidInput(
                "fold commitment group count mismatch".to_string(),
            ));
        }

        let mut group_order = (0..opening_batch.num_groups())
            .map(|group_index| {
                let range = params.commitment_row_range(
                    &opening_batch,
                    group_index,
                    RelationMatrixRowLayout::WithDBlock,
                )?;
                Ok((range.start, range.len(), group_index))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        group_order.sort_by_key(|(start, _, _)| *start);

        let mut coeffs = Vec::new();
        let commitment_ring_dim = params.role_dims().d_a();
        for (_, expected_rows, group_index) in group_order {
            let commitment = self.opening_claims.group_commitment(group_index)?;
            let rows = commitment.rows();
            if !rows.can_decode_vec(commitment_ring_dim) {
                return Err(AkitaError::InvalidInput(format!(
                    "fold commitment row shape mismatch for group {group_index}: \
                     coeff_len {} is not divisible by d_a {commitment_ring_dim}",
                    rows.coeff_len()
                )));
            }
            let actual_rows = rows.coeff_len() / commitment_ring_dim;
            if actual_rows != expected_rows {
                return Err(AkitaError::InvalidInput(format!(
                    "fold commitment row count mismatch for group {group_index}: \
                     expected {expected_rows}, actual {actual_rows}"
                )));
            }
            coeffs.extend_from_slice(rows.coeffs());
        }
        Ok(RingVec::from_coeffs(coeffs))
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
}

impl<'a, PointF, CommitF> ProverOpeningData<'a, PointF, RecursiveFoldSource<CommitF>, CommitF>
where
    PointF: FieldCore,
    CommitF: FieldCore,
{
    fn setup_prefix_column_major_indices(
        setup_prefix_point_len: usize,
        setup_slot: &SetupPrefixSlot<CommitF>,
        offset: usize,
        shared_point_len: usize,
    ) -> Result<PointVariableSelection, AkitaError> {
        let ring_bits = setup_slot.id.d_setup.trailing_zeros() as usize;
        let params = &setup_slot.id.commitment_params;
        let expected = ring_bits
            .checked_add(params.layout.r_vars)
            .and_then(|n| n.checked_add(params.layout.m_vars))
            .ok_or_else(|| AkitaError::InvalidSetup("setup-prefix point length overflow".into()))?;
        if setup_prefix_point_len != expected {
            return Err(AkitaError::InvalidPointDimension {
                expected,
                actual: setup_prefix_point_len,
            });
        }
        let mut indices = Vec::with_capacity(expected);
        indices.extend(offset..offset + ring_bits);
        indices.extend(
            offset + ring_bits + params.layout.m_vars
                ..offset + ring_bits + params.layout.m_vars + params.layout.r_vars,
        );
        indices.extend(offset + ring_bits..offset + ring_bits + params.layout.m_vars);
        PointVariableSelection::new(indices, shared_point_len)
    }

    fn shared_stage3_point(
        setup_prefix_point: &[PointF],
        witness_point: &[PointF],
    ) -> Result<(Vec<PointF>, usize), AkitaError> {
        if setup_prefix_point.len() >= witness_point.len() {
            if &setup_prefix_point[setup_prefix_point.len() - witness_point.len()..]
                != witness_point
            {
                return Err(AkitaError::InvalidInput(
                    "stage-3 suffix opening points are inconsistent".to_string(),
                ));
            }
            Ok((setup_prefix_point.to_vec(), 0))
        } else {
            if &witness_point[witness_point.len() - setup_prefix_point.len()..]
                != setup_prefix_point
            {
                return Err(AkitaError::InvalidInput(
                    "stage-3 suffix opening points are inconsistent".to_string(),
                ));
            }
            Ok((
                witness_point.to_vec(),
                witness_point.len() - setup_prefix_point.len(),
            ))
        }
    }

    pub(crate) fn recursive_suffix_eor_claims(
        shared_point: Vec<PointF>,
        setup_prefix_point_vars: Option<PointVariableSelection>,
        witness_point_len: usize,
    ) -> Result<OpeningClaims<'a, PointF>, AkitaError> {
        let mut groups = Vec::with_capacity(usize::from(setup_prefix_point_vars.is_some()) + 1);
        if let Some(setup_prefix_point_vars) = setup_prefix_point_vars {
            groups.push(PolynomialGroupClaims::new(
                setup_prefix_point_vars,
                vec![PointF::zero()],
                (),
            )?);
        }
        groups.push(PolynomialGroupClaims::new(
            PointVariableSelection::suffix(witness_point_len, shared_point.len())?,
            vec![PointF::zero()],
            (),
        )?);
        OpeningClaims::from_groups_allow_custom_routing(shared_point, groups)
    }

    /// Build the single-group recursive suffix batch using the mixed-source type.
    pub(crate) fn new_recursive_suffix_source(
        opening_point: &[PointF],
        recursive_num_vars: usize,
        witness_polys: &'a [&'a RecursiveFoldSource<CommitF>],
        commitment: CommitmentWithHint<CommitF>,
    ) -> Result<Self, AkitaError> {
        let mut padded_point = opening_point.to_vec();
        padded_point.resize(recursive_num_vars, PointF::zero());
        let point_vars = PointVariableSelection::prefix(recursive_num_vars, recursive_num_vars)?;
        let claims = PolynomialGroupClaims::new(point_vars, vec![PointF::zero()], commitment.0)?;
        ProverOpeningData::new(
            OpeningClaims::from_groups(padded_point, vec![claims])?,
            vec![commitment.1],
            vec![witness_polys],
        )
    }

    /// Build recursive suffix opening data, with an optional setup-prefix group.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_recursive_suffix_fold(
        opening_point: &[PointF],
        recursive_num_vars: usize,
        setup_prefix_opening: Option<(Vec<PointF>, PointF)>,
        setup_slot: Option<&'a SetupPrefixSlot<CommitF>>,
        setup_polys: Option<&'a [&'a RecursiveFoldSource<CommitF>]>,
        witness_eval: PointF,
        witness_polys: &'a [&'a RecursiveFoldSource<CommitF>],
        witness_commitment: CommitmentWithHint<CommitF>,
    ) -> Result<(Self, OpeningClaims<'a, PointF>, Vec<PointF>), AkitaError> {
        match (setup_prefix_opening, setup_slot, setup_polys) {
            (
                Some((setup_prefix_point, setup_prefix_eval)),
                Some(setup_slot),
                Some(setup_polys),
            ) => {
                let (shared_point, setup_offset) =
                    Self::shared_stage3_point(&setup_prefix_point, opening_point)?;
                let setup_point_vars = Self::setup_prefix_column_major_indices(
                    setup_prefix_point.len(),
                    setup_slot,
                    setup_offset,
                    shared_point.len(),
                )?;
                let fold_claims = Self::new_recursive_suffix_with_setup_prefix(
                    shared_point.clone(),
                    setup_point_vars.clone(),
                    opening_point.len(),
                    setup_prefix_eval,
                    witness_eval,
                    setup_slot,
                    setup_polys,
                    witness_polys,
                    setup_slot.hint.clone(),
                    witness_commitment.1,
                    witness_commitment.0,
                )?;
                let eor_claims = Self::recursive_suffix_eor_claims(
                    shared_point.clone(),
                    Some(setup_point_vars),
                    opening_point.len(),
                )?;
                Ok((fold_claims, eor_claims, shared_point))
            }
            (None, None, None) => {
                let fold_claims = Self::new_recursive_suffix_source(
                    opening_point,
                    recursive_num_vars,
                    witness_polys,
                    witness_commitment,
                )?;
                let eor_claims = Self::recursive_suffix_eor_claims(
                    opening_point.to_vec(),
                    None,
                    opening_point.len(),
                )?;
                Ok((fold_claims, eor_claims, opening_point.to_vec()))
            }
            _ => Err(AkitaError::InvalidInput(
                "setup-prefix suffix inputs are incomplete".to_string(),
            )),
        }
    }

    /// Build the two-group recursive suffix batch used when the previous fold
    /// offloaded setup-prefix evaluation into this fold.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_recursive_suffix_with_setup_prefix(
        shared_point: Vec<PointF>,
        setup_prefix_point_vars: PointVariableSelection,
        witness_point_len: usize,
        setup_prefix_eval: PointF,
        witness_eval: PointF,
        setup_slot: &'a SetupPrefixSlot<CommitF>,
        setup_polys: &'a [&'a RecursiveFoldSource<CommitF>],
        witness_polys: &'a [&'a RecursiveFoldSource<CommitF>],
        setup_hint: AkitaCommitmentHint<CommitF>,
        witness_hint: AkitaCommitmentHint<CommitF>,
        witness_commitment: Commitment<CommitF>,
    ) -> Result<Self, AkitaError> {
        let setup_commitment_rows =
            setup_slot.commitment.rows.first().cloned().ok_or_else(|| {
                AkitaError::InvalidSetup("setup-prefix slot has no commitment rows".into())
            })?;
        let setup_group = PolynomialGroupClaims::new(
            setup_prefix_point_vars,
            vec![setup_prefix_eval],
            Commitment::new(setup_commitment_rows),
        )?;
        let witness_group = PolynomialGroupClaims::new(
            PointVariableSelection::suffix(witness_point_len, shared_point.len())?,
            vec![witness_eval],
            witness_commitment,
        )?;
        ProverOpeningData::new(
            OpeningClaims::from_groups_allow_custom_routing(
                shared_point,
                vec![setup_group, witness_group],
            )?,
            vec![setup_hint, witness_hint],
            vec![setup_polys, witness_polys],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp32;
    use akita_transcript::labels::ABSORB_COMMITMENT;
    use akita_transcript::AkitaTranscript;

    type F = Fp32<251>;

    #[derive(Clone)]
    struct MockPoly {
        num_vars: usize,
    }

    impl RootPolyMeta<F> for MockPoly {
        fn num_ring_elems(&self) -> usize {
            0
        }

        fn num_vars(&self) -> usize {
            self.num_vars
        }
    }

    fn empty_hint() -> AkitaCommitmentHint<F> {
        AkitaCommitmentHint::new(Vec::new())
    }

    fn commitment() -> Commitment<F> {
        Commitment::new(RingVec::from_coeffs(vec![F::zero()]))
    }

    fn multi_group_data<'a>(
        pre_refs: &'a [&'a MockPoly],
        final_refs: &'a [&'a MockPoly],
    ) -> ProverOpeningData<'a, F, MockPoly, F> {
        let claims = OpeningClaims::from_groups(
            vec![F::zero(); 4],
            vec![
                PolynomialGroupClaims::new(
                    PointVariableSelection::prefix(2, 4).expect("pre point vars"),
                    vec![F::zero()],
                    commitment(),
                )
                .expect("pre group"),
                PolynomialGroupClaims::new(
                    PointVariableSelection::prefix(4, 4).expect("final point vars"),
                    vec![F::zero(), F::zero()],
                    commitment(),
                )
                .expect("final group"),
            ],
        )
        .expect("claims");
        ProverOpeningData::new(
            claims,
            vec![empty_hint(), empty_hint()],
            vec![pre_refs, final_refs],
        )
        .expect("prover data")
    }

    #[test]
    fn opening_layout_preserves_precise_group_arities() {
        let pre_poly = MockPoly { num_vars: 2 };
        let final_a = MockPoly { num_vars: 4 };
        let final_b = MockPoly { num_vars: 4 };
        let pre_refs = [&pre_poly];
        let final_refs = [&final_a, &final_b];
        let data = multi_group_data(&pre_refs, &final_refs);

        let layout = data.opening_layout::<F>().expect("precise layout");

        assert_eq!(
            layout.groups(),
            &[
                PolynomialGroupLayout::new(2, 1),
                PolynomialGroupLayout::new(4, 2)
            ]
        );
    }

    #[test]
    fn opening_layout_rejects_group_arity_mismatch() {
        let pre_poly = MockPoly { num_vars: 3 };
        let final_a = MockPoly { num_vars: 4 };
        let final_b = MockPoly { num_vars: 4 };
        let pre_refs = [&pre_poly];
        let final_refs = [&final_a, &final_b];
        let data = multi_group_data(&pre_refs, &final_refs);

        let err = data
            .opening_layout::<F>()
            .expect_err("pre group point vars claim two variables");

        assert!(matches!(
            err,
            AkitaError::InvalidPointDimension {
                expected: 3,
                actual: 2
            }
        ));
    }

    #[test]
    fn append_to_transcript_binds_precise_group_shape_not_padded_max() {
        let pre_poly = MockPoly { num_vars: 2 };
        let final_a = MockPoly { num_vars: 4 };
        let final_b = MockPoly { num_vars: 4 };
        let pre_refs = [&pre_poly];
        let final_refs = [&final_a, &final_b];
        let data = multi_group_data(&pre_refs, &final_refs);

        let mut precise = AkitaTranscript::<F>::new(b"test/precise-group-shape");
        data.append_to_transcript(1, &mut precise)
            .expect("precise transcript absorb");
        let precise_challenge = precise.challenge_scalar(b"after-shape");

        let padded_layout =
            OpeningClaimsLayout::from_group_sizes(4, &[1, 2]).expect("old padded layout");
        let mut padded = AkitaTranscript::<F>::new(b"test/precise-group-shape");
        padded_layout
            .append_batch_shape_to_transcript::<F, _>(&mut padded)
            .expect("padded shape absorb");
        for commitment in data.commitments() {
            commitment
                .append_to_transcript(ABSORB_COMMITMENT, 1, &mut padded)
                .expect("commitment absorb");
        }
        for coord in data.point() {
            akita_transcript::append_ext_field::<F, F, _>(
                &mut padded,
                akita_transcript::labels::ABSORB_EVALUATION_CLAIMS,
                coord,
            );
        }
        let padded_challenge = padded.challenge_scalar(b"after-shape");

        assert_ne!(precise_challenge, padded_challenge);
    }
}
