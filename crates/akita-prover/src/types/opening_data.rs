use crate::api::CommitmentWithHint;
use crate::backend::RecursiveFoldSource;
use crate::compute::RootPolyMeta;
use akita_error::AkitaError;
use akita_transcript::Transcript;
use akita_types::{
    AkitaCommitmentHint, Commitment, CommittedGroupParams, OpeningClaims, OpeningClaimsLayout,
    PolynomialGroupClaims, PolynomialGroupLayout, RingVec, SetupPrefixSlot,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Prover opening input: public claims plus prover-only hints and polynomials.
#[derive(Debug, Clone)]
pub struct ProverOpeningData<'a, PointF: Clone, P, CommitF: Field> {
    opening_claims: OpeningClaims<'a, PointF, Commitment<CommitF>>,
    hints: Vec<AkitaCommitmentHint<CommitF>>,
    polynomials: Vec<&'a [&'a P]>,
}

impl<'a, PointF: Clone, P, CommitF> ProverOpeningData<'a, PointF, P, CommitF>
where
    CommitF: Field,
{
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
        PolyF: Field,
        P: RootPolyMeta<PolyF>,
    {
        self.check_alignment()?;
        self.opening_layout::<PolyF>()?;
        Ok(())
    }

    /// Largest natural root arity across all polynomial groups.
    pub fn num_vars<PolyF>(&self) -> Result<usize, AkitaError>
    where
        PolyF: Field,
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

    /// Public claims carried by this prover input.
    pub fn opening_claims(&self) -> &OpeningClaims<'a, PointF, Commitment<CommitF>> {
        &self.opening_claims
    }

    /// Layout-only opening geometry derived from prover polynomials.
    pub fn opening_layout<PolyF>(&self) -> Result<OpeningClaimsLayout, AkitaError>
    where
        PolyF: Field,
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
            let group_point = self.opening_claims.group_point(group_index)?;
            if group_point.len() != group_num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: group_num_vars,
                    actual: group_point.len(),
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

    /// Absorb the normalized batch shape, commitments, and group points.
    pub fn append_to_transcript<T>(
        &self,
        root_params: &CommittedGroupParams,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        CommitF: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
        PointF: ExtField<CommitF>,
        P: RootPolyMeta<CommitF>,
        T: Transcript<CommitF>,
    {
        // `opening_layout` validates that each public point matches its
        // polynomial group's shape, keeping this byte-identical to verifier
        // replay for well-formed inputs.
        let layout = self.opening_layout::<CommitF>()?;
        layout.append_batch_shape_to_transcript::<CommitF, T>(transcript)?;
        for (group_index, commitment) in self.commitments().into_iter().enumerate() {
            let ring_dim = root_params.group_role_dims(&layout, group_index)?.d_b();
            commitment.append_to_transcript(
                akita_transcript::labels::ABSORB_COMMITMENT,
                ring_dim,
                transcript,
            )?;
        }
        for group in self.opening_claims.groups() {
            for coord in group.point() {
                akita_transcript::append_ext_field::<CommitF, PointF, T>(
                    transcript,
                    akita_transcript::labels::ABSORB_EVALUATION_CLAIMS,
                    coord,
                );
            }
        }
        Ok(())
    }

    /// Borrow root fold commitment rows in the scheduled M-row commitment order.
    pub(crate) fn fold_commitment(
        &self,
        params: &CommittedGroupParams,
    ) -> Result<RingVec<CommitF>, AkitaError> {
        let opening_batch = self.opening_claims.layout()?;
        if self.opening_claims.num_groups() != opening_batch.num_groups() {
            return Err(AkitaError::InvalidInput(
                "fold commitment group count mismatch".to_string(),
            ));
        }

        let mut group_order = (0..opening_batch.num_groups())
            .map(|group_index| {
                let range = params.commitment_row_range(&opening_batch, group_index)?;
                Ok((range.start, range.len(), group_index))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        group_order.sort_by_key(|(start, _, _)| *start);

        let mut coeffs = Vec::new();
        for (_, expected_rows, group_index) in group_order {
            let commitment = self.opening_claims.group_commitment(group_index)?;
            let rows = commitment.rows();
            let commitment_ring_dim = params.group_role_dims(&opening_batch, group_index)?.d_b();
            if !rows.can_decode_vec(commitment_ring_dim) {
                return Err(AkitaError::InvalidInput(format!(
                    "fold commitment row shape mismatch for group {group_index}: \
                     coeff_len {} is not divisible by d_b {commitment_ring_dim}",
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
    PointF: Field,
    CommitF: Field,
{
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
    ) -> Result<Self, AkitaError> {
        if opening_point.len() > recursive_num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: recursive_num_vars,
                actual: opening_point.len(),
            });
        }
        let witness_group = PolynomialGroupClaims::new(
            opening_point.to_vec(),
            vec![witness_eval],
            witness_commitment.0,
        )?;

        match (setup_prefix_opening, setup_slot, setup_polys) {
            (
                Some((setup_prefix_point, setup_prefix_eval)),
                Some(setup_slot),
                Some(setup_polys),
            ) => {
                let setup_commitment_rows =
                    setup_slot.commitment.rows.first().cloned().ok_or_else(|| {
                        AkitaError::InvalidSetup("setup-prefix slot has no commitment rows".into())
                    })?;
                let setup_group = PolynomialGroupClaims::new(
                    setup_prefix_point,
                    vec![setup_prefix_eval],
                    Commitment::new(setup_commitment_rows),
                )?;
                ProverOpeningData::new(
                    OpeningClaims::from_groups(vec![setup_group, witness_group])?,
                    vec![setup_slot.hint.clone(), witness_commitment.1],
                    vec![setup_polys, witness_polys],
                )
            }
            (None, None, None) => ProverOpeningData::new(
                OpeningClaims::from_groups(vec![witness_group])?,
                vec![witness_commitment.1],
                vec![witness_polys],
            ),
            _ => Err(AkitaError::InvalidInput(
                "setup-prefix suffix inputs are incomplete".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_transcript::labels::ABSORB_COMMITMENT;
    use akita_transcript::AkitaTranscript;
    use akita_types::{PrecommittedGroupDescriptor, PrecommittedLevelParams, SisModulusProfileId};
    use jolt_field::Fp32;
    use jolt_field::Zero;

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
        AkitaCommitmentHint::new(1, Vec::new()).expect("empty test hint")
    }

    fn commitment() -> Commitment<F> {
        Commitment::new(RingVec::from_coeffs(vec![F::zero(); 64]))
    }

    fn multi_group_params() -> CommittedGroupParams {
        let pre_layout = PolynomialGroupLayout::new(2, 1);
        let mut pre = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            1,
            1,
            1,
            1,
            SparseChallengeConfig::production_for_ring_dim(64)
                .expect("test ring has a production challenge"),
        )
        .with_decomp(1, 1, 1, 1, 1)
        .expect("precommitted params");
        let inner = &pre.inner_commit_matrix;
        pre.inner_commit_matrix = akita_types::InnerCommitMatrixParams::new_unchecked(
            inner.security_policy(),
            inner.sis_table_key().table_digest,
            inner.sis_modulus_profile(),
            inner.output_rank(),
            inner.input_width(),
            1,
            inner.ring_dimension(),
        );
        let outer = &pre.outer_commit_matrix;
        pre.outer_commit_matrix = akita_types::OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            outer.input_width(),
            1,
            outer.ring_dimension(),
        );
        let mut root = CommittedGroupParams::params_only(
            SisModulusProfileId::Q128OffsetA7F7,
            64,
            1,
            1,
            1,
            1,
            SparseChallengeConfig::production_for_ring_dim(64)
                .expect("test ring has a production challenge"),
        )
        .with_decomp(1, 1, 1, 1, 1)
        .expect("root params");
        root.precommitted_groups.push(PrecommittedLevelParams {
            layout: PrecommittedGroupDescriptor::from_params(pre_layout, &pre),
            inner_commit_matrix: pre.inner_commit_matrix,
            outer_commit_matrix: pre.outer_commit_matrix,
            log_basis_open: pre.log_basis_open,
            fold_challenge_config: pre.fold_challenge_config,
            num_digits_inner: pre.num_digits_inner,
            num_digits_outer: pre.num_digits_outer,
            num_digits_open: pre.num_digits_open,
            num_digits_fold_one: pre.num_digits_fold_one,
        });
        root
    }

    fn multi_group_data<'a>(
        pre_refs: &'a [&'a MockPoly],
        final_refs: &'a [&'a MockPoly],
    ) -> ProverOpeningData<'a, F, MockPoly, F> {
        let claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(vec![F::zero(); 2], vec![F::zero()], commitment())
                .expect("pre group"),
            PolynomialGroupClaims::new(
                vec![F::zero(); 4],
                vec![F::zero(), F::zero()],
                commitment(),
            )
            .expect("final group"),
        ])
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
        let root_params = multi_group_params();

        let mut precise = AkitaTranscript::<F>::new(b"test/precise-group-shape");
        data.append_to_transcript(&root_params, &mut precise)
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
                .append_to_transcript(ABSORB_COMMITMENT, 64, &mut padded)
                .expect("commitment absorb");
        }
        for group in data.opening_claims().groups() {
            for coord in group.point() {
                akita_transcript::append_ext_field::<F, F, _>(
                    &mut padded,
                    akita_transcript::labels::ABSORB_EVALUATION_CLAIMS,
                    coord,
                );
            }
        }
        let padded_challenge = padded.challenge_scalar(b"after-shape");

        assert_ne!(precise_challenge, padded_challenge);
    }
}
