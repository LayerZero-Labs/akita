use akita_error::AkitaError;

/// Public shape of one retained source group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceMetadata {
    num_polynomials: usize,
    num_vars: usize,
}

impl SourceMetadata {
    /// Validate and construct source metadata.
    pub fn try_new(num_polynomials: usize, num_vars: usize) -> Result<Self, AkitaError> {
        if num_polynomials == 0 {
            return Err(AkitaError::InvalidInput(
                "source group must contain at least one polynomial".into(),
            ));
        }
        Ok(Self {
            num_polynomials,
            num_vars,
        })
    }

    /// Number of polynomials in the homogeneous group.
    pub const fn num_polynomials(self) -> usize {
        self.num_polynomials
    }

    /// Number of variables in each polynomial.
    pub const fn num_vars(self) -> usize {
        self.num_vars
    }
}

/// Immutable planned geometry attached to a recursive-witness handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecursiveWitnessManifest {
    logical_len: usize,
    commitment_domain_len: usize,
    commitment_ring_dimension: usize,
}

impl RecursiveWitnessManifest {
    /// Validate and construct a witness manifest.
    pub fn try_new(
        logical_len: usize,
        commitment_domain_len: usize,
        commitment_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if logical_len == 0
            || commitment_domain_len < logical_len
            || commitment_ring_dimension == 0
            || !commitment_ring_dimension.is_power_of_two()
        {
            return Err(AkitaError::InvalidInput(
                "invalid recursive-witness manifest".into(),
            ));
        }
        Ok(Self {
            logical_len,
            commitment_domain_len,
            commitment_ring_dimension,
        })
    }

    /// Logical coefficient count before commitment padding.
    pub const fn logical_len(self) -> usize {
        self.logical_len
    }

    /// Planned coefficient count in the commitment domain.
    pub const fn commitment_domain_len(self) -> usize {
        self.commitment_domain_len
    }

    /// Planned commitment ring dimension.
    pub const fn commitment_ring_dimension(self) -> usize {
        self.commitment_ring_dimension
    }

    /// Boolean-hypercube arity of the padded logical witness.
    pub fn num_vars(self) -> Result<usize, AkitaError> {
        akita_error::checked::ceil_log2(self.logical_len)
            .ok_or_else(|| AkitaError::InvalidInput("recursive witness length overflow".into()))
    }
}

/// Public geometry of a consumer-owned relation witness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelationWitnessMetadata {
    witness_len: usize,
    column_bits: usize,
    coefficient_bits: usize,
}

impl RelationWitnessMetadata {
    /// Construct validated relation metadata.
    pub fn try_new(
        witness_len: usize,
        column_bits: usize,
        coefficient_bits: usize,
    ) -> Result<Self, AkitaError> {
        if witness_len == 0 {
            return Err(AkitaError::InvalidInput(
                "invalid relation-witness metadata".into(),
            ));
        }
        Ok(Self {
            witness_len,
            column_bits,
            coefficient_bits,
        })
    }

    /// Packed witness length.
    pub const fn witness_len(self) -> usize {
        self.witness_len
    }

    /// Number of bits selecting a packed column.
    pub const fn column_bits(self) -> usize {
        self.column_bits
    }

    /// Number of coefficient bits represented by the relation table.
    pub const fn coefficient_bits(self) -> usize {
        self.coefficient_bits
    }
}

/// Representation-free accepted fold.
pub trait AcceptedFoldHandle: Send + Sync + 'static {
    /// Public accepted-fold shape.
    fn metadata(&self) -> crate::backend::AcceptedFoldMetadata;
}

/// Representation-free complete recursive-witness handle.
pub trait RecursiveWitnessHandle: Send + Sync + 'static {
    /// Immutable public geometry planned for this witness.
    fn manifest(&self) -> RecursiveWitnessManifest;
}

/// Public dimensions and producer provenance of reusable, immutable committed
/// source storage.
pub trait CommitmentHandleMetadata {
    fn metadata(&self) -> SourceMetadata;
    /// The committed-source contract that admitted this group at commit time.
    ///
    /// Proof bytes do not bind it. Trusted planning code compares it with the
    /// contract a schedule row was planned under.
    fn producer_contract(&self) -> akita_types::sis::CommittedSourceContract;
}

/// A protocol input selected from the backend's opaque handle family.
pub enum OpeningSource<'a, C, W> {
    Commitment(&'a C),
    Witness(&'a W),
}
impl<C, W> Copy for OpeningSource<'_, C, W> {}
impl<C, W> Clone for OpeningSource<'_, C, W> {
    fn clone(&self) -> Self {
        *self
    }
}
