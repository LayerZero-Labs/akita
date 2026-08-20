//! Committed group profiles and runtime schedule lookup identity.

use crate::descriptor_bytes::push_usize;
use crate::{
    CommitmentSliceCount, CommitmentSliceGeometry, CommittedGroup, CommittedGroupParams,
    InnerCommitMatrixParams, OpeningClaimsLayout, OuterCommitMatrixParams, PolynomialGroupLayout,
};
use akita_field::{AkitaError, FieldCore};

/// Physical coefficient representation authenticated by a commitment.
///
/// This is commitment identity, not an opening policy. In particular, changing
/// [`crate::OpeningMethod`] cannot reinterpret an existing commitment between
/// these encodings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommittedSourceEncoding {
    /// The source's canonical base-field coefficient table.
    CanonicalCoefficientTable,
    /// The existing tensor/subfield projection used by extension-field EOR.
    TensorSubfieldProjection {
        /// Degree of the extension whose coordinates are packed by the projection.
        extension_degree: usize,
    },
}

impl CommittedSourceEncoding {
    /// Select the physical source representation when a producer creates a
    /// commitment for one scheduled consumer.
    #[must_use]
    pub fn for_producer(
        opening_method: crate::OpeningMethod,
        extension_degree: usize,
        _ring_dimension: usize,
        _source_num_vars: usize,
        is_root: bool,
    ) -> Self {
        if !is_root
            && matches!(opening_method, crate::OpeningMethod::EvaluationTrace)
            && extension_degree > 1
        {
            Self::TensorSubfieldProjection { extension_degree }
        } else {
            Self::CanonicalCoefficientTable
        }
    }

    pub(crate) fn append_descriptor_bytes(self, bytes: &mut Vec<u8>) {
        match self {
            Self::CanonicalCoefficientTable => bytes.push(0),
            Self::TensorSubfieldProjection { extension_degree } => {
                bytes.push(1);
                push_usize(bytes, extension_degree);
            }
        }
    }

    /// Validate that this encoding can represent coefficients at `ring_dimension`.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor projection's extension degree does not
    /// divide the usable half-ring coefficient capacity.
    pub fn validate(self, ring_dimension: usize) -> Result<(), AkitaError> {
        if let Self::TensorSubfieldProjection { extension_degree } = self {
            let tensor_capacity = ring_dimension / 2;
            if extension_degree <= 1
                || !extension_degree.is_power_of_two()
                || extension_degree > tensor_capacity
                || !tensor_capacity.is_multiple_of(extension_degree)
            {
                return Err(AkitaError::InvalidSetup(
                    "tensor source encoding requires a power-of-two extension degree greater than one dividing half the A ring dimension"
                        .into(),
                ));
            }
        }
        Ok(())
    }
}

/// Root layout metadata frozen when a standalone commitment group is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CommittedGroupProfile {
    /// Version of the canonical committed-profile encoding.
    pub version: u8,
    /// Per-group root schedule entry shape.
    pub group: PolynomialGroupLayout,
    /// Exact number of live source ring elements per claim (`N`).
    pub num_live_ring_elements_per_claim: usize,
    /// Number of positions per block (`M`), power-of-two in the current Boolean layout.
    pub num_positions_per_block: usize,
    /// Exact number of live blocks (`B = ceil(N / M)`).
    pub num_live_blocks: usize,
    /// Number of logical B inputs committed through one physical B matrix.
    pub outer_slice_count: CommitmentSliceCount,
    /// Gadget basis selected for the standalone A/source digits.
    pub log_basis_inner: u32,
    /// Exact gadget depth used by the standalone A/source relation.
    pub num_digits_inner: usize,
    /// Complete audited A/source matrix identity.
    pub inner_commit_matrix: InnerCommitMatrixParams,
    /// Gadget basis selected for the standalone B/`t_hat` digits.
    pub log_basis_outer: u32,
    /// Exact gadget depth used by the standalone B/`t_hat` relation.
    pub num_digits_outer: usize,
    /// Complete audited B/commitment matrix identity.
    pub outer_commit_matrix: OuterCommitMatrixParams,
}

impl CommittedGroupProfile {
    /// Current committed-profile format.
    pub const VERSION: u8 = 4;

    /// Build and validate frozen group metadata from concrete root commitment parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when the matrices, decomposition, slicing, or exact root
    /// geometry cannot describe a standalone commitment for `group`.
    pub fn try_from_params(
        group: PolynomialGroupLayout,
        params: &CommittedGroupParams,
    ) -> Result<Self, AkitaError> {
        if params.source_encoding != CommittedSourceEncoding::CanonicalCoefficientTable {
            return Err(AkitaError::InvalidSetup(
                "standalone commitment profiles require canonical coefficient sources".into(),
            ));
        }
        let profile = Self::from_params_fields(group, params);
        profile.validate_frozen_precommit(
            profile
                .inner_commit_matrix
                .sis_modulus_profile()
                .field_bits(),
        )?;
        Ok(profile)
    }

    fn from_params_fields(group: PolynomialGroupLayout, params: &CommittedGroupParams) -> Self {
        Self {
            version: Self::VERSION,
            group,
            num_live_ring_elements_per_claim: params.num_live_ring_elements_per_claim,
            num_positions_per_block: params.num_positions_per_block,
            num_live_blocks: params.num_live_blocks,
            outer_slice_count: params.outer_slice_count,
            log_basis_inner: params.log_basis_inner,
            num_digits_inner: params.num_digits_inner,
            inner_commit_matrix: params.inner_commit_matrix,
            log_basis_outer: params.log_basis_outer,
            num_digits_outer: params.num_digits_outer,
            outer_commit_matrix: params.outer_commit_matrix,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_params_unchecked_for_test(
        group: PolynomialGroupLayout,
        params: &CommittedGroupParams,
    ) -> Self {
        Self::from_params_fields(group, params)
    }

    /// This group's block triple.
    ///
    /// Step 4 makes this a stored field; until then it is assembled on demand
    /// from the three flat fields, which is free.
    #[inline]
    #[must_use]
    pub fn blocks(&self) -> crate::BlockGeometry {
        crate::BlockGeometry::new(
            self.num_live_ring_elements_per_claim,
            self.num_positions_per_block,
            self.num_live_blocks,
        )
    }

    /// The A-role gadget decomposition.
    #[inline]
    #[must_use]
    pub fn inner_digits(&self) -> crate::GadgetDigits {
        crate::GadgetDigits::new(self.log_basis_inner, self.num_digits_inner)
    }

    /// The B-role gadget decomposition.
    #[inline]
    #[must_use]
    pub fn outer_digits(&self) -> crate::GadgetDigits {
        crate::GadgetDigits::new(self.log_basis_outer, self.num_digits_outer)
    }

    /// Canonical versioned bytes used for catalog and schedule-key identity.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        self.append_descriptor_bytes(&mut bytes);
        bytes
    }

    /// The block triple and both `(basis, depth)` pairs are already contiguous
    /// here, so the atomic leaf encoders are byte-neutral. This encoder is
    /// therefore already role-atomic: geometry, then `basis, depth, matrix`
    /// twice, which is the shape step 4 makes structural.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.push(self.version);
        push_usize(bytes, self.group.num_vars());
        push_usize(bytes, self.group.num_polynomials());
        self.blocks().append_descriptor_bytes(bytes);
        self.outer_slice_count.append_descriptor_bytes(bytes);
        self.inner_digits().append_descriptor_bytes(bytes);
        self.inner_commit_matrix.append_descriptor_bytes(bytes);
        self.outer_digits().append_descriptor_bytes(bytes);
        self.outer_commit_matrix.append_descriptor_bytes(bytes);
    }

    /// Validate that this layout is a well-formed standalone commitment group.
    pub fn validate(&self, field_bits: u32) -> Result<(), AkitaError> {
        self.group.validate()?;
        if self.version != Self::VERSION {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported committed-group profile version {}",
                self.version
            )));
        }
        self.inner_commit_matrix.validate()?;
        self.outer_commit_matrix.validate()?;
        let inner_ring_dimension = self.inner_commit_matrix.ring_dimension();
        let outer_ring_dimension = self.outer_commit_matrix.ring_dimension();
        if !inner_ring_dimension.is_power_of_two()
            || !outer_ring_dimension.is_power_of_two()
            || !inner_ring_dimension.is_multiple_of(outer_ring_dimension)
        {
            return Err(AkitaError::InvalidSetup(
                "precommitted group requires power-of-two A/B dimensions with A divisible by B"
                    .to_string(),
            ));
        }
        self.inner_digits().validate(field_bits)?;
        if self.log_basis_outer == 0 || self.num_digits_outer == 0 {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout requires nonzero outer basis and digit depth".to_string(),
            ));
        }
        self.outer_slice_count.validate_for_commitment(
            0,
            crate::CommitmentPayloadMode::Compressed,
            self.num_live_blocks,
        )?;
        if self.inner_commit_matrix.sis_modulus_profile().field_bits() != field_bits
            || self.outer_commit_matrix.sis_modulus_profile().field_bits() != field_bits
        {
            return Err(AkitaError::InvalidSetup(
                "committed-group matrix modulus profile does not match the field".to_string(),
            ));
        }
        let expected_a_width = self
            .num_positions_per_block
            .checked_mul(self.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("committed-group A width overflow".into()))?;
        let expected_b_width = CommitmentSliceGeometry::try_new(
            self.outer_slice_count,
            self.num_live_blocks,
            self.group.num_polynomials(),
            self.inner_commit_matrix.output_rank(),
            self.num_digits_outer,
            inner_ring_dimension,
            outer_ring_dimension,
        )?
        .physical_input_width();
        if self.inner_commit_matrix.input_width() != expected_a_width
            || self.outer_commit_matrix.input_width() != expected_b_width
        {
            return Err(AkitaError::InvalidSetup(
                "committed-group A/B matrix widths do not match frozen geometry".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate that frozen exact block geometry matches `group.num_vars`.
    pub fn validate_root_geometry(&self) -> Result<(), AkitaError> {
        let inner_ring_dimension = self.inner_commit_matrix.ring_dimension();
        let alpha = inner_ring_dimension.trailing_zeros() as usize;
        let Some(source_field_len) = self
            .num_live_ring_elements_per_claim
            .checked_mul(inner_ring_dimension)
        else {
            return Err(AkitaError::InvalidSetup(
                "commitment group layout geometry overflow".to_string(),
            ));
        };
        let num_vars = u32::try_from(self.group.num_vars()).map_err(|_| {
            AkitaError::InvalidSetup("commitment group variable count exceeds u32".to_string())
        })?;
        let expected_field_len = 1usize.checked_shl(num_vars).ok_or_else(|| {
            AkitaError::InvalidSetup("commitment group field length overflow".to_string())
        })?;
        if source_field_len != expected_field_len
            || self.num_positions_per_block == 0
            || !self.num_positions_per_block.is_power_of_two()
            || self.num_live_blocks
                != self
                    .num_live_ring_elements_per_claim
                    .div_ceil(self.num_positions_per_block)
        {
            return Err(AkitaError::InvalidSetup(format!(
                "precommitted group geometry does not match group.num_vars: \
                 N={} L={} F={} alpha={} group.num_vars={}",
                self.num_live_ring_elements_per_claim,
                self.num_positions_per_block,
                self.num_live_blocks,
                alpha,
                self.group.num_vars()
            )));
        }
        Ok(())
    }

    /// Validate metadata frozen by a precommitted group at precommit time.
    pub fn validate_frozen_precommit(&self, field_bits: u32) -> Result<(), AkitaError> {
        self.validate(field_bits)?;
        self.validate_root_geometry()?;
        Ok(())
    }
}

/// Canonical runtime schedule lookup key.
///
/// Openings without precommitted groups use an empty `precommitteds` vector and
/// store their only group in `final_group`. Multi-group roots list earlier
/// groups in `precommitteds` and the final group in `final_group`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AkitaScheduleLookupKey {
    /// Final group shape for the multi-group root commitment.
    pub final_group: PolynomialGroupLayout,
    /// Previously committed groups in caller-supplied transcript order.
    pub precommitteds: Vec<CommittedGroupProfile>,
}

/// A non-empty ordered prefix of groups committed before a final group.
///
/// The type is non-empty by construction, so a grouped commitment context
/// cannot describe "no precommitted groups" — that state has its own spelling in
/// `PrecommittedGroupContext::NoPrecommittedGroups`. Both constructors reject an empty
/// input, which is why building a grouped context is infallible.
#[derive(Debug, Clone)]
pub struct PrecommittedGroupProfiles {
    profiles: Vec<CommittedGroupProfile>,
}

impl PrecommittedGroupProfiles {
    /// Take ownership of an already ordered profile vector without cloning it.
    ///
    /// # Errors
    ///
    /// Returns an error when `profiles` is empty.
    pub fn from_profiles(profiles: Vec<CommittedGroupProfile>) -> Result<Self, AkitaError> {
        if profiles.is_empty() {
            return Err(AkitaError::InvalidInput(
                "precommitted group profiles must describe at least one group".to_string(),
            ));
        }
        Ok(Self { profiles })
    }

    /// Extract profiles from committed groups in caller-supplied order.
    ///
    /// # Errors
    ///
    /// Returns an error when `groups` is empty.
    pub fn from_ordered_groups<'a, F, I>(groups: I) -> Result<Self, AkitaError>
    where
        F: FieldCore + 'a,
        I: IntoIterator<Item = &'a CommittedGroup<F>>,
        I::IntoIter: ExactSizeIterator,
    {
        let groups = groups.into_iter();
        let mut profiles = Vec::with_capacity(groups.len());
        profiles.extend(groups.map(|group| *group.profile()));
        Self::from_profiles(profiles)
    }

    /// Borrow the exact ordered profiles.
    pub fn as_slice(&self) -> &[CommittedGroupProfile] {
        &self.profiles
    }
}

impl AkitaScheduleLookupKey {
    /// Scalar root-opening context with no precommitted groups.
    pub fn single(final_group: PolynomialGroupLayout) -> Self {
        Self {
            final_group,
            precommitteds: Vec::new(),
        }
    }

    /// Canonical ordered bytes used for schedule-key identity tests and caches.
    pub fn canonical_descriptor_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_usize(&mut bytes, self.final_group.num_vars());
        push_usize(&mut bytes, self.final_group.num_polynomials());
        push_usize(&mut bytes, self.precommitteds.len());
        for descriptor in &self.precommitteds {
            descriptor.append_descriptor_bytes(&mut bytes);
        }
        bytes
    }

    /// Build a multi-group opening layout from this schedule lookup key.
    pub fn opening_layout(&self) -> Result<OpeningClaimsLayout, AkitaError> {
        let mut groups: Vec<PolynomialGroupLayout> = self
            .precommitteds
            .iter()
            .map(|layout| layout.group)
            .collect();
        groups.push(self.final_group);
        OpeningClaimsLayout::from_groups(groups)
    }

    /// Number of commitment groups in this schedule key.
    pub fn num_commitment_groups(&self) -> usize {
        self.precommitteds.len() + 1
    }

    /// Maximum opening arity across the final and precommitted groups.
    ///
    /// This is the maximum group-local opening/EOR domain. It is intentionally
    /// distinct from `final_group.num_vars()`, which remains the source arity
    /// used to size the final commitment and root witness.
    pub fn max_num_vars(&self) -> usize {
        self.precommitteds
            .iter()
            .map(|descriptor| descriptor.group.num_vars())
            .fold(self.final_group.num_vars(), usize::max)
    }

    /// Total number of polynomials across the final and precommitted groups.
    pub fn num_polynomials(&self) -> Result<usize, AkitaError> {
        let mut total = self.final_group.num_polynomials();
        for layout in &self.precommitteds {
            total = total
                .checked_add(layout.group.num_polynomials())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "multi-group root polynomial count overflow".to_string(),
                    )
                })?;
        }
        Ok(total)
    }

    /// Whether the complete opening key fits a setup's public capacity.
    pub fn fits_setup_capacity(
        &self,
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<bool, AkitaError> {
        Ok(self.max_num_vars() <= max_num_vars && self.num_polynomials()? <= max_num_batched_polys)
    }

    /// Validate per-group metadata.
    pub fn validate(&self, field_bits: u32) -> Result<(), AkitaError> {
        self.final_group.validate()?;
        if self.final_group.num_vars() == 0 {
            return Err(AkitaError::InvalidSetup(
                "schedule lookup key dimensions must be at least 1".to_string(),
            ));
        }
        for layout in &self.precommitteds {
            layout.group.validate()?;
            layout.validate(field_bits)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod source_encoding_tests {
    use super::CommittedSourceEncoding;
    use crate::OpeningMethod;

    #[test]
    fn producer_encoding_is_canonical_at_root_and_tensor_only_for_recursive_et() {
        assert_eq!(
            CommittedSourceEncoding::for_producer(OpeningMethod::EvaluationTrace, 4, 512, 20, true,),
            CommittedSourceEncoding::CanonicalCoefficientTable,
        );
        assert_eq!(
            CommittedSourceEncoding::for_producer(OpeningMethod::EvaluationTrace, 4, 512, 8, true,),
            CommittedSourceEncoding::CanonicalCoefficientTable,
        );
        assert_eq!(
            CommittedSourceEncoding::for_producer(
                OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension: 64,
                },
                4,
                512,
                20,
                true,
            ),
            CommittedSourceEncoding::CanonicalCoefficientTable,
        );
        assert_eq!(
            CommittedSourceEncoding::for_producer(OpeningMethod::EvaluationTrace, 4, 64, 1, false,),
            CommittedSourceEncoding::TensorSubfieldProjection {
                extension_degree: 4,
            },
        );

        let valid_recursive_boundary = CommittedSourceEncoding::for_producer(
            OpeningMethod::EvaluationTrace,
            32,
            64,
            16,
            false,
        );
        assert_eq!(
            valid_recursive_boundary,
            CommittedSourceEncoding::TensorSubfieldProjection {
                extension_degree: 32,
            },
        );
        valid_recursive_boundary
            .validate(64)
            .expect("k=32 fits the half-ring capacity at d_A=64");

        let invalid_recursive_boundary = CommittedSourceEncoding::for_producer(
            OpeningMethod::EvaluationTrace,
            64,
            64,
            16,
            false,
        );
        assert_eq!(
            invalid_recursive_boundary,
            CommittedSourceEncoding::TensorSubfieldProjection {
                extension_degree: 64,
            },
            "recursive EvaluationTrace must not silently change the authenticated encoding",
        );
        assert!(invalid_recursive_boundary.validate(64).is_err());
    }
}

/// Exact ordered committed profiles used to resolve a verifier-approved row.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CommittedGroupBatchProfile {
    /// Final/new commitment group.
    pub final_group: CommittedGroupProfile,
    /// Earlier commitments in transcript order.
    pub precommitteds: Vec<CommittedGroupProfile>,
}

impl CommittedGroupBatchProfile {
    /// Assemble an exact batch profile from ordered committed groups.
    ///
    /// Each group carries its own profile, so the prefix is derived here
    /// rather than supplied and cross-checked.
    ///
    /// # Errors
    ///
    /// Returns an error when `groups` is empty.
    pub fn from_ordered_groups<'a, F, I>(groups: I) -> Result<Self, AkitaError>
    where
        F: FieldCore + 'a,
        I: IntoIterator<Item = &'a CommittedGroup<F>>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut profiles = groups
            .into_iter()
            .map(|group| *group.profile())
            .collect::<Vec<_>>();
        let final_group = profiles.pop().ok_or_else(|| {
            AkitaError::InvalidInput(
                "committed group batch profile requires at least one group".to_string(),
            )
        })?;
        Ok(Self {
            final_group,
            precommitteds: profiles,
        })
    }

    /// Build the corresponding public opening layout.
    pub fn opening_layout(&self) -> Result<OpeningClaimsLayout, AkitaError> {
        let mut groups = self
            .precommitteds
            .iter()
            .map(|profile| profile.group)
            .collect::<Vec<_>>();
        groups.push(self.final_group.group);
        OpeningClaimsLayout::from_groups(groups)
    }

    /// Validate all frozen profiles.
    pub fn validate(&self, field_bits: u32) -> Result<(), AkitaError> {
        self.final_group.validate_frozen_precommit(field_bits)?;
        for profile in &self.precommitteds {
            profile.validate_frozen_precommit(field_bits)?;
        }
        Ok(())
    }
}
