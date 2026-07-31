use crate::api::CommitmentWithHint;
use crate::backend::RecursiveFoldSource;
use crate::compute::RootPolyMeta;
use crate::protocol::core::RootProverGroupMeta;
use crate::PreparedProverGroup;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore};
use akita_transcript::Transcript;
use akita_types::{
    normalize_recursive_opening_point, AkitaCommitmentHint, Commitment, CommittedGroup,
    CommittedGroupParams, OpeningClaims, OpeningClaimsLayout, OpeningScheduleSelection,
    PolynomialGroupClaims, PolynomialGroupLayout, RingVec, SetupPrefixSlot,
};

/// Exact top-level row selection paired with its prover opening material.
pub type SelectedProverOpeningData<'a, PointF, G, CommitF> = (
    OpeningScheduleSelection,
    ProverOpeningData<'a, PointF, G, CommitF>,
);

#[derive(Debug, Clone)]
struct ProverGroupInput<G, CommitF: FieldCore> {
    hint: AkitaCommitmentHint<CommitF>,
    group: G,
}

impl<G, CommitF: FieldCore> ProverGroupInput<G, CommitF> {
    fn new(hint: AkitaCommitmentHint<CommitF>, group: G) -> Self {
        Self { hint, group }
    }

    fn hint(&self) -> &AkitaCommitmentHint<CommitF> {
        &self.hint
    }

    fn group(&self) -> &G {
        &self.group
    }
}

fn bind_group_inputs<G, CommitF: FieldCore>(
    hints: Vec<AkitaCommitmentHint<CommitF>>,
    groups: Vec<G>,
) -> Result<Vec<ProverGroupInput<G, CommitF>>, AkitaError> {
    if hints.len() != groups.len() {
        return Err(AkitaError::InvalidInput(
            "prover hint and prepared-source counts are misaligned".to_string(),
        ));
    }
    Ok(hints
        .into_iter()
        .zip(groups)
        .map(|(hint, group)| ProverGroupInput::new(hint, group))
        .collect())
}

/// Prover opening input: public claims plus ordered group-local prover material.
#[derive(Debug, Clone)]
pub struct ProverOpeningData<'a, PointF: Clone, G, CommitF: FieldCore> {
    opening_claims: OpeningClaims<'a, PointF, Commitment<CommitF>>,
    group_inputs: Vec<ProverGroupInput<G, CommitF>>,
}

impl<'a, PointF: Clone, P, CommitF: FieldCore>
    ProverOpeningData<'a, PointF, PreparedProverGroup<'a, P>, CommitF>
where
    P: RootPolyMeta<CommitF>,
{
    fn new_internal(
        opening_claims: OpeningClaims<'a, PointF, Commitment<CommitF>>,
        hints: Vec<AkitaCommitmentHint<CommitF>>,
        polynomials: Vec<&'a [&'a P]>,
    ) -> Result<Self, AkitaError> {
        let groups = polynomials
            .into_iter()
            .map(PreparedProverGroup::from_refs)
            .collect::<Result<Vec<_>, _>>()?;
        let group_inputs = bind_group_inputs(hints, groups)?;
        let data = Self {
            opening_claims,
            group_inputs,
        };
        data.check_alignment()?;
        Ok(data)
    }

    /// Bundle public claims with matching prover hints and polynomial groups.
    pub fn new(
        opening_claims: OpeningClaims<'a, PointF, CommittedGroup<CommitF>>,
        hints: Vec<AkitaCommitmentHint<CommitF>>,
        polynomials: Vec<&'a [&'a P]>,
    ) -> Result<Self, AkitaError> {
        let groups = polynomials
            .into_iter()
            .map(PreparedProverGroup::from_refs)
            .collect::<Result<Vec<_>, _>>()?;
        let raw_groups = opening_claims
            .groups()
            .iter()
            .map(|group| {
                PolynomialGroupClaims::new(
                    group.point().to_vec(),
                    group.evaluations().to_vec(),
                    group.commitment().commitment().clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let group_inputs = bind_group_inputs(hints, groups)?;
        let data = Self {
            opening_claims: OpeningClaims::from_groups(raw_groups)?,
            group_inputs,
        };
        data.check_alignment()?;
        Ok(data)
    }
}

#[allow(private_bounds)]
impl<'a, PointF: Clone, G, CommitF: FieldCore> ProverOpeningData<'a, PointF, G, CommitF>
where
    G: RootProverGroupMeta<CommitF>,
{
    fn check_alignment(&self) -> Result<(), AkitaError> {
        if self.opening_claims.num_groups() != self.group_inputs.len() {
            return Err(AkitaError::InvalidInput(
                "prover opening data group counts are misaligned".to_string(),
            ));
        }
        for group_index in 0..self.opening_claims.num_groups() {
            let expected = self.opening_claims.group_evaluations(group_index)?.len();
            let actual = self
                .group_inputs
                .get(group_index)
                .ok_or_else(|| AkitaError::InvalidInput("missing polynomial group".to_string()))?
                .group
                .num_polynomials();
            if actual != expected {
                return Err(AkitaError::InvalidInput(
                    "prover opening data polynomial/evaluation counts are misaligned".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Largest natural root arity across all polynomial groups.
    pub fn num_vars(&self) -> Result<usize, AkitaError> {
        self.group_inputs
            .iter()
            .map(|input| input.group.num_vars())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
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
    pub fn opening_layout(&self) -> Result<OpeningClaimsLayout, AkitaError> {
        let mut groups = Vec::with_capacity(self.group_inputs.len());
        for (group_index, input) in self.group_inputs.iter().enumerate() {
            let group = &input.group;
            let group_num_vars = group.num_vars()?;
            let group_point = self.opening_claims.group_point(group_index)?;
            if group_point.len() != group_num_vars {
                return Err(AkitaError::InvalidPointDimension {
                    expected: group_num_vars,
                    actual: group_point.len(),
                });
            }
            groups.push(PolynomialGroupLayout::new(
                group_num_vars,
                group.num_polynomials(),
            ));
        }
        OpeningClaimsLayout::from_groups(groups)
    }

    /// Borrow one prover hint.
    pub fn group_hint(&self, index: usize) -> Result<&AkitaCommitmentHint<CommitF>, AkitaError> {
        self.group_inputs
            .get(index)
            .map(ProverGroupInput::hint)
            .ok_or(AkitaError::InvalidProof)
    }

    /// Borrow one polynomial group.
    pub(crate) fn group(&self, index: usize) -> Result<&G, AkitaError> {
        self.group_inputs
            .get(index)
            .map(ProverGroupInput::group)
            .ok_or(AkitaError::InvalidProof)
    }

    pub(crate) fn groups(&self) -> impl ExactSizeIterator<Item = &G> {
        self.group_inputs.iter().map(ProverGroupInput::group)
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
        CommitF: CanonicalField,
        PointF: ExtField<CommitF>,
        T: Transcript<CommitF>,
    {
        // `opening_layout` validates that each public point matches its
        // polynomial group's shape, keeping this byte-identical to verifier
        // replay for well-formed inputs.
        let layout = self.opening_layout()?;
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
    ) -> Result<ProverOpeningData<'b, PointF, PreparedProverGroup<'b, Q>, CommitF>, AkitaError>
    where
        'a: 'b,
        Q: RootPolyMeta<CommitF>,
    {
        let mut input_offset = 0usize;
        let mut regrouped = Vec::with_capacity(self.group_inputs.len());
        for input in self.group_inputs {
            let group_len = input.group.num_polynomials();
            let input_end = input_offset.checked_add(group_len).ok_or_else(|| {
                AkitaError::InvalidInput("fold input group offset overflow".to_string())
            })?;
            let replacement_polynomials =
                polynomials.get(input_offset..input_end).ok_or_else(|| {
                    AkitaError::InvalidInput("fold input group shape mismatch".to_string())
                })?;
            regrouped.push(ProverGroupInput::new(
                input.hint,
                PreparedProverGroup::from_refs(replacement_polynomials)?,
            ));
            input_offset = input_end;
        }
        if input_offset != polynomials.len() {
            return Err(AkitaError::InvalidInput(
                "fold input group coverage mismatch".to_string(),
            ));
        }
        let data = ProverOpeningData {
            opening_claims: self.opening_claims,
            group_inputs: regrouped,
        };
        data.check_alignment()?;
        Ok(data)
    }
}

impl<'a, PointF, CommitF>
    ProverOpeningData<'a, PointF, PreparedProverGroup<'a, RecursiveFoldSource<CommitF>>, CommitF>
where
    PointF: FieldCore,
    CommitF: FieldCore,
{
    /// Build the single-group recursive suffix batch using the mixed-source type.
    pub(crate) fn new_recursive_suffix_source(
        opening_point: &[PointF],
        recursive_num_vars: usize,
        witness_polys: &'a [&'a RecursiveFoldSource<CommitF>],
        commitment: CommitmentWithHint<CommitF>,
    ) -> Result<Self, AkitaError> {
        let padded_point = normalize_recursive_opening_point(opening_point, recursive_num_vars)?;
        let claims = PolynomialGroupClaims::new(padded_point, vec![PointF::zero()], commitment.0)?;
        ProverOpeningData::new_internal(
            OpeningClaims::from_groups(vec![claims])?,
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
    ) -> Result<Self, AkitaError> {
        let padded_witness_point =
            normalize_recursive_opening_point(opening_point, recursive_num_vars)?;

        match (setup_prefix_opening, setup_slot, setup_polys) {
            (
                Some((setup_prefix_point, setup_prefix_eval)),
                Some(setup_slot),
                Some(setup_polys),
            ) => {
                let block_claims = Self::new_recursive_suffix_with_setup_prefix(
                    setup_prefix_point.clone(),
                    padded_witness_point.clone(),
                    setup_prefix_eval,
                    witness_eval,
                    setup_slot,
                    setup_polys,
                    witness_polys,
                    setup_slot.hint.clone(),
                    witness_commitment.1,
                    witness_commitment.0,
                )?;
                Ok(block_claims)
            }
            (None, None, None) => {
                let block_claims = Self::new_recursive_suffix_source(
                    &padded_witness_point,
                    recursive_num_vars,
                    witness_polys,
                    witness_commitment,
                )?;
                Ok(block_claims)
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
        setup_prefix_point: Vec<PointF>,
        witness_point: Vec<PointF>,
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
            setup_prefix_point,
            vec![setup_prefix_eval],
            Commitment::new(setup_commitment_rows),
        )?;
        let witness_group =
            PolynomialGroupClaims::new(witness_point, vec![witness_eval], witness_commitment)?;
        ProverOpeningData::new_internal(
            OpeningClaims::from_groups(vec![setup_group, witness_group])?,
            vec![setup_hint, witness_hint],
            vec![setup_polys, witness_polys],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp32;
    use akita_transcript::labels::ABSORB_COMMITMENT;
    use akita_transcript::AkitaTranscript;
    use akita_types::{CommittedGroupProfile, PrecommittedLevelParams, SisModulusProfileId};

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
            2,
            inner.ring_dimension(),
        );
        let outer = &pre.outer_commit_matrix;
        pre.outer_commit_matrix = akita_types::OuterCommitMatrixParams::new_unchecked(
            outer.security_policy(),
            outer.sis_table_key().table_digest,
            outer.sis_modulus_profile(),
            outer.output_rank(),
            outer.input_width(),
            3,
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
            layout: CommittedGroupProfile::from_params(pre_layout, &pre),
            log_basis_open: pre.log_basis_open,
            fold_challenge_config: pre.fold_challenge_config,
            num_digits_open: pre.num_digits_open,
            num_digits_fold: pre.num_digits_fold,
            fold_witness_linf_cap: pre.fold_witness_linf_cap,
        });
        root
    }

    fn multi_group_data<'a>(
        pre_refs: &'a [&'a MockPoly],
        final_refs: &'a [&'a MockPoly],
    ) -> ProverOpeningData<'a, F, PreparedProverGroup<'a, MockPoly>, F> {
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
        ProverOpeningData::new_internal(
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

        let layout = data.opening_layout().expect("precise layout");

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
            .opening_layout()
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
