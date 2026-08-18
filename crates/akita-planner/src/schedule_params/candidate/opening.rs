use super::*;

/// Checked opening geometry considered by one planner candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PlannerOpeningCandidate {
    EvaluationTrace {
        challenge_config: SparseChallengeConfig,
    },
    SubringCoefficientPacking {
        geometry: akita_types::SubringCoefficientPackingGeometry,
    },
}

impl PlannerOpeningCandidate {
    /// Preserve the historical full-ring EvaluationTrace candidate.
    pub(crate) fn evaluation_trace(challenge_config: SparseChallengeConfig) -> Self {
        Self::EvaluationTrace { challenge_config }
    }

    /// Enumerate the fixed production packing domain for one level/role tuple.
    pub(crate) fn coefficient_packing_domain(
        absolute_level: usize,
        extension_degree: usize,
        dimensions: CommitmentRingDims,
    ) -> Result<Vec<Self>, AkitaError> {
        dimensions.validate_role_projection()?;
        if absolute_level > 1 {
            return Ok(Vec::new());
        }
        Ok(akita_challenges::PRODUCTION_FOLD_CHALLENGE_RING_DIMS
            .iter()
            .copied()
            .filter_map(|s| {
                Self::coefficient_packing(absolute_level, extension_degree, dimensions, s).ok()
            })
            .collect())
    }

    /// Admit one coefficient-packing candidate before SIS lookup or sizing.
    pub(crate) fn coefficient_packing(
        absolute_level: usize,
        extension_degree: usize,
        dimensions: CommitmentRingDims,
        challenge_subring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if absolute_level > 1 {
            return Err(AkitaError::InvalidSetup(
                "coefficient packing is restricted to absolute fold levels zero and one".into(),
            ));
        }
        dimensions.validate_role_projection()?;
        let geometry = akita_types::SubringCoefficientPackingGeometry::try_new(
            extension_degree,
            dimensions.d_a(),
            challenge_subring_dimension,
        )?;
        if !geometry
            .partial_base_field_width()
            .is_multiple_of(dimensions.d_d())
        {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing width must decompose into D-native subcolumns".into(),
            ));
        }
        Ok(Self::SubringCoefficientPacking { geometry })
    }

    pub(crate) const fn method(self) -> akita_types::OpeningMethod {
        match self {
            Self::EvaluationTrace { .. } => akita_types::OpeningMethod::EvaluationTrace,
            Self::SubringCoefficientPacking { geometry } => {
                akita_types::OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension: geometry.challenge_subring_dimension(),
                }
            }
        }
    }

    pub(crate) const fn challenge_config(self) -> SparseChallengeConfig {
        match self {
            Self::EvaluationTrace { challenge_config } => challenge_config,
            Self::SubringCoefficientPacking { geometry } => geometry.fold_challenge_config(),
        }
    }

    pub(crate) const fn challenge_dimension(self, ambient_dimension: usize) -> usize {
        match self {
            Self::EvaluationTrace { .. } => ambient_dimension,
            Self::SubringCoefficientPacking { geometry } => geometry.challenge_subring_dimension(),
        }
    }

    pub(crate) const fn is_coefficient_packing(self) -> bool {
        matches!(self, Self::SubringCoefficientPacking { .. })
    }

    pub(crate) const fn physical_coefficient_width(self, ambient_dimension: usize) -> usize {
        match self {
            Self::EvaluationTrace { .. } => ambient_dimension,
            Self::SubringCoefficientPacking { geometry } => geometry.partial_base_field_width(),
        }
    }

    pub(crate) fn validate_for(
        self,
        absolute_level: usize,
        extension_degree: usize,
        dimensions: CommitmentRingDims,
    ) -> Result<(), AkitaError> {
        let Self::SubringCoefficientPacking { geometry } = self else {
            self.challenge_config()
                .validate_for_ring_dim(dimensions.d_a())
                .map_err(|reason| AkitaError::InvalidSetup(reason.to_string()))?;
            return Ok(());
        };
        let expected = Self::coefficient_packing(
            absolute_level,
            extension_degree,
            dimensions,
            geometry.challenge_subring_dimension(),
        )?;
        if self != expected {
            return Err(AkitaError::InvalidSetup(
                "planner opening candidate disagrees with its checked geometry".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_candidate_admission_uses_registered_geometry() {
        let admitted = PlannerOpeningCandidate::coefficient_packing(
            1,
            2,
            CommitmentRingDims {
                inner: 256,
                outer: 128,
                opening: 64,
            },
            64,
        )
        .expect("registered two-subcolumn packing geometry");
        assert_eq!(
            admitted.method(),
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64
            }
        );
        assert_eq!(
            admitted.challenge_config(),
            SparseChallengeConfig::production_for_ring_dim(64).unwrap()
        );
        assert!(admitted.is_coefficient_packing());

        assert!(PlannerOpeningCandidate::coefficient_packing(
            2,
            2,
            CommitmentRingDims::uniform(128),
            64
        )
        .is_err());
        assert!(PlannerOpeningCandidate::coefficient_packing(
            1,
            2,
            CommitmentRingDims::uniform(256),
            32
        )
        .is_err());
        assert!(PlannerOpeningCandidate::coefficient_packing(
            1,
            4,
            CommitmentRingDims::uniform(128),
            64
        )
        .is_err());
        assert!(PlannerOpeningCandidate::coefficient_packing(
            1,
            2,
            CommitmentRingDims {
                inner: 256,
                outer: 128,
                opening: 256,
            },
            64
        )
        .is_err());
    }

    #[test]
    fn packing_domain_is_exact_for_each_extension_degree() {
        let dimensions = CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 64,
        };
        for (extension_degree, expected) in
            [(1, vec![64, 128, 256]), (2, vec![64, 128]), (4, vec![64])]
        {
            let actual = PlannerOpeningCandidate::coefficient_packing_domain(
                0,
                extension_degree,
                dimensions,
            )
            .unwrap()
            .into_iter()
            .map(|candidate| match candidate.method() {
                akita_types::OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension,
                } => challenge_subring_dimension,
                akita_types::OpeningMethod::EvaluationTrace => unreachable!(),
            })
            .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
        assert!(
            PlannerOpeningCandidate::coefficient_packing_domain(2, 2, dimensions)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn evaluation_trace_rejects_an_unaudited_challenge_family() {
        let candidate =
            PlannerOpeningCandidate::evaluation_trace(SparseChallengeConfig::pm1_only(0));
        assert!(candidate
            .validate_for(0, 1, CommitmentRingDims::uniform(256))
            .is_err());
    }
}
