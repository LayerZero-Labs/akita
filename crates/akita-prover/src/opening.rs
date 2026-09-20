//! Public claims paired with immutable reusable commitment handles.
use crate::backend::CommitmentHandleMetadata;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_transcript::Transcript;
use akita_types::{
    Commitment, CommittedGroup, CommittedGroupBatchProfile, CommittedGroupParams, OpeningClaims,
    OpeningClaimsLayout, OpeningScheduleSelection, PolynomialGroupClaims,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

pub struct SelectedProverOpeningData<'a, E: Clone, H, F: Field> {
    selection: OpeningScheduleSelection,
    opening_data: ProverOpeningData<'a, E, H, F>,
}
impl<'a, E: Clone, H: CommitmentHandleMetadata, F: Field> SelectedProverOpeningData<'a, E, H, F> {
    pub fn from_committed_claims<Cfg: CommitmentConfig<Field = F, ExtField = E>>(
        claims: OpeningClaims<'a, E, CommittedGroup<F>>,
        handles: Vec<H>,
        schedules: &TrustedScheduleCatalog<Cfg>,
    ) -> Result<Self, AkitaError> {
        if claims.num_groups() != handles.len() {
            return Err(AkitaError::InvalidProof);
        }
        let profile = CommittedGroupBatchProfile::from_ordered_groups(
            claims
                .groups()
                .iter()
                .map(PolynomialGroupClaims::commitment),
        )?;
        let selection = schedules.resolve_profiles(&profile)?.selection();
        let opening_layout = claims.committed_layout()?;
        for ((layout, handle), claim) in opening_layout
            .groups()
            .iter()
            .zip(&handles)
            .zip(claims.groups())
        {
            if claim.point().len() != layout.num_vars() {
                return Err(AkitaError::InvalidProof);
            }
            let meta = handle.metadata();
            if meta.num_vars() != layout.num_vars()
                || meta.num_polynomials() != layout.num_polynomials()
            {
                return Err(AkitaError::InvalidInput(
                    "commitment handle shape differs from statement".into(),
                ));
            }
        }
        let groups = claims
            .groups()
            .iter()
            .map(|g| {
                PolynomialGroupClaims::new(
                    g.point().to_vec(),
                    g.evaluations().to_vec(),
                    g.commitment().commitment().clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            selection,
            opening_data: ProverOpeningData::from_parts(
                OpeningClaims::from_groups(groups)?,
                opening_layout,
                handles,
            )?,
        })
    }
    pub fn selection(&self) -> OpeningScheduleSelection {
        self.selection
    }
    pub fn opening_layout(&self) -> &OpeningClaimsLayout {
        self.opening_data.opening_layout()
    }
    pub(crate) fn into_low_level_parts(
        self,
    ) -> (OpeningScheduleSelection, ProverOpeningData<'a, E, H, F>) {
        (self.selection, self.opening_data)
    }
}

pub struct ProverOpeningData<'a, E: Clone, G, F: Field> {
    opening_claims: OpeningClaims<'a, E, Commitment<F>>,
    opening_layout: OpeningClaimsLayout,
    groups: Vec<G>,
}
impl<'a, PointF: Clone, G, CommitF: Field> ProverOpeningData<'a, PointF, G, CommitF> {
    pub(crate) fn from_parts(
        opening_claims: OpeningClaims<'a, PointF, Commitment<CommitF>>,
        opening_layout: OpeningClaimsLayout,
        groups: Vec<G>,
    ) -> Result<Self, AkitaError> {
        if opening_claims.num_groups() != groups.len()
            || groups.len() != opening_layout.num_groups()
        {
            return Err(AkitaError::InvalidProof);
        }
        for (claims, layout) in opening_claims.groups().iter().zip(opening_layout.groups()) {
            if claims.point().len() > layout.num_vars()
                || claims.evaluations().len() != layout.num_polynomials()
            {
                return Err(AkitaError::InvalidProof);
            }
        }
        Ok(Self {
            opening_claims,
            opening_layout,
            groups,
        })
    }
    pub fn opening_claims(&self) -> &OpeningClaims<'a, PointF, Commitment<CommitF>> {
        &self.opening_claims
    }
    pub fn opening_layout(&self) -> &OpeningClaimsLayout {
        &self.opening_layout
    }
    pub(crate) fn group(&self, index: usize) -> Result<&G, AkitaError> {
        self.groups.get(index).ok_or(AkitaError::InvalidProof)
    }
    pub fn commitments(&self) -> Vec<&Commitment<CommitF>> {
        self.opening_claims
            .groups()
            .iter()
            .map(PolynomialGroupClaims::commitment)
            .collect()
    }
    pub(crate) fn map_groups<'b, Q>(
        &'b self,
        mut map: impl FnMut(&'b G) -> Q,
    ) -> Result<ProverOpeningData<'a, PointF, Q, CommitF>, AkitaError> {
        ProverOpeningData::from_parts(
            self.opening_claims.clone(),
            self.opening_layout.clone(),
            self.groups.iter().map(&mut map).collect(),
        )
    }
    /// Absorb the normalized batch shape, commitments, and group points.
    pub fn append_to_transcript<T>(
        &self,
        root_params: &CommittedGroupParams,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        CommitF: CanonicalEncoding + AkitaSerialize,
        PointF: ExtField<CommitF>,
        T: Transcript<CommitF>,
    {
        let layout = self.opening_layout();
        let relation_geometry =
            akita_types::RelationWitnessGeometry::for_level(root_params, layout, PointF::DEGREE)?;
        let relation_layout = relation_geometry.rhs_layout();
        layout.append_batch_shape_to_transcript::<CommitF, T>(transcript)?;
        for (group_index, commitment) in self.commitments().into_iter().enumerate() {
            let compression = relation_layout.compression_plan_for_group(group_index)?;
            if commitment.rows().coeff_len() != compression.terminal_coefficients() {
                return Err(AkitaError::InvalidInput(
                    "root compressed commitment does not match scheduled root params".into(),
                ));
            }
            let ring_dim = compression
                .maps()
                .last()
                .ok_or(AkitaError::InvalidProof)?
                .ring_dimension();
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
}
