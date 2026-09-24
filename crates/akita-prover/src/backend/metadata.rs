use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};
use jolt_poly::{NormalizedPoly, UnivariatePoly};

/// Public, coefficient-free description of an accepted fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcceptedFoldMetadata {
    pub(crate) ring_dimension: usize,
    pub(crate) response_coordinate_count: usize,
    pub(crate) num_chunks: usize,
}

impl AcceptedFoldMetadata {
    /// Construct accepted-fold metadata after checking the public geometry.
    pub fn try_new(
        ring_dimension: usize,
        response_coordinate_count: usize,
        num_chunks: usize,
    ) -> Result<Self, AkitaError> {
        if ring_dimension == 0 || !ring_dimension.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "accepted-fold ring dimension must be a non-zero power of two".into(),
            ));
        }
        if response_coordinate_count == 0
            || !response_coordinate_count.is_multiple_of(ring_dimension)
        {
            return Err(AkitaError::InvalidInput(
                "accepted-fold response width must be a non-zero multiple of the ring dimension"
                    .into(),
            ));
        }
        if num_chunks == 0 || !num_chunks.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "accepted-fold chunk count must be a non-zero power of two".into(),
            ));
        }
        Ok(Self {
            ring_dimension,
            response_coordinate_count,
            num_chunks,
        })
    }

    pub const fn ring_dimension(self) -> usize {
        self.ring_dimension
    }

    pub const fn response_coordinate_count(self) -> usize {
        self.response_coordinate_count
    }

    pub const fn num_chunks(self) -> usize {
        self.num_chunks
    }
}

/// Public scalar output of a completed consumer-owned relation session.
pub struct RelationWitnessFinalClaims<E: Field> {
    witness_evaluation: E,
    final_claim: E,
}

impl<E: Field> RelationWitnessFinalClaims<E> {
    pub fn new(witness_evaluation: E, final_claim: E) -> Self {
        Self {
            witness_evaluation,
            final_claim,
        }
    }
    pub fn witness_evaluation(&self) -> E {
        self.witness_evaluation
    }
    pub fn final_claim(&self) -> E {
        self.final_claim
    }
}

/// Public, witness-free opening metadata retained for Stage 2.
#[derive(Clone)]
pub struct PreparedRelationGroupPublic<F: Field, E: Field> {
    kind: akita_types::OpeningFamily<
        akita_types::PreparedOpeningPoint<F, E>,
        akita_types::PreparedSubringCoefficientPackingPoint<E>,
    >,
    scalar_openings: Vec<E>,
}

impl<F: Field, E: Field> PreparedRelationGroupPublic<F, E> {
    pub fn new(
        kind: akita_types::OpeningFamily<
            akita_types::PreparedOpeningPoint<F, E>,
            akita_types::PreparedSubringCoefficientPackingPoint<E>,
        >,
        scalar_openings: Vec<E>,
    ) -> Self {
        Self {
            kind,
            scalar_openings,
        }
    }

    pub const fn kind(
        &self,
    ) -> &akita_types::OpeningFamily<
        akita_types::PreparedOpeningPoint<F, E>,
        akita_types::PreparedSubringCoefficientPackingPoint<E>,
    > {
        &self.kind
    }

    pub fn scalar_openings(&self) -> &[E] {
        &self.scalar_openings
    }

    #[cfg(test)]
    pub(crate) fn coefficient_packing_point(
        &self,
    ) -> Option<&akita_types::PreparedSubringCoefficientPackingPoint<E>> {
        match &self.kind {
            akita_types::OpeningFamily::EvaluationTrace(_) => None,
            akita_types::OpeningFamily::SubringCoefficientPacking(point) => Some(point),
        }
    }
}

/// Public geometry attached to consumer-owned commitment relation material.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommitmentMaterialMetadata {
    pub(crate) ring_dimension: usize,
    pub(crate) source_count: usize,
    pub(crate) has_compression: bool,
}

impl CommitmentMaterialMetadata {
    pub fn try_new(
        ring_dimension: usize,
        source_count: usize,
        has_compression: bool,
    ) -> Result<Self, AkitaError> {
        if ring_dimension == 0 || !ring_dimension.is_power_of_two() || source_count == 0 {
            return Err(AkitaError::InvalidInput(
                "commitment material requires a power-of-two ring and at least one source".into(),
            ));
        }
        Ok(Self {
            ring_dimension,
            source_count,
            has_compression,
        })
    }

    pub const fn ring_dimension(self) -> usize {
        self.ring_dimension
    }

    pub const fn source_count(self) -> usize {
        self.source_count
    }

    pub const fn has_compression(self) -> bool {
        self.has_compression
    }
}

/// Consumer-owned commitment material transported opaquely by Akita.
pub trait CommitmentRelationMaterial<F>: Send + 'static
where
    F: Field + CanonicalEncoding,
{
    fn metadata(&self) -> CommitmentMaterialMetadata;
}

/// Narrow terminal-only declassification of commitment material.
pub trait TerminalCommitmentMaterialKernel<F, M>
where
    F: Field + CanonicalEncoding,
    M: CommitmentRelationMaterial<F>,
{
    fn terminal_message(
        &self,
        material: &M,
    ) -> Result<crate::backend::TerminalTFieldsMessage, AkitaError>;

    fn consume_terminal_row(&self, material: M) -> Result<akita_types::RingVec<F>, AkitaError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage1Step {
    Product(usize),
    RangeLeaf,
    FusedRangeNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage1Transition {
    ProductBatch(usize),
    L2SubclaimBatch,
    RangeNormMerge,
}

#[derive(Clone)]
pub enum Stage1RoundPolynomial<E: Field> {
    EqFactored(NormalizedPoly<E>),
    Standard(UnivariatePoly<E>),
}

pub enum Stage1PublicTransition<E: Field> {
    ProductChildClaims(Vec<E>),
    PhysicalL2Claims {
        response_l2_sq: u128,
        subclaims: Vec<E>,
    },
    Final {
        range_image_evaluation: E,
        virtual_evaluations: Vec<E>,
    },
}

pub struct Stage1FinalClaims<E: Field> {
    point: Vec<E>,
    final_claim: E,
}

impl<E: Field> Stage1FinalClaims<E> {
    pub fn new(point: Vec<E>, final_claim: E) -> Self {
        Self { point, final_claim }
    }
    pub fn point(&self) -> &[E] {
        &self.point
    }
    pub fn final_claim(&self) -> E {
        self.final_claim
    }
}
