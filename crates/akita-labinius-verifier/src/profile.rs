//! Explicit setup admission and transcript identity.
//!
//! Admission here checks geometry, fold entropy, and integer no-wrap only.
//! `BinaryClearSetup::new` takes the matrix from its caller and performs no SIS
//! or width-table security admission; `crate::admitted::AdmittedRootSetup` is
//! the seed-derived construction that does.

use akita_algebra::binary::field_switch::SwitchField;
use akita_algebra::fft::SmoothFftField;
use akita_algebra::ring::{TrinomialModulus, TrinomialNttDomain, TrinomialRing};
use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::{
    checked_source_comparison_class_bound, LabiniusCoefficientPrime, LabiniusRingDegree,
    LabiniusSourceComparisonId, SourceOccurrenceBound,
};

/// Admitted explicit setup for the standalone clear differential oracle.
#[derive(Clone, Debug)]
pub struct BinaryClearSetup<F, const D: usize, M: TrinomialModulus> {
    matrix: Vec<TrinomialRing<F, D, M>>,
    n_a: usize,
    m: usize,
    columns: usize,
    scalar_rows: usize,
    source_len: usize,
    k: usize,
    lower: i64,
    upper: i64,
    lambda_fold: u32,
    profile: BinaryChallengeProfile,
    coefficient_prime: LabiniusCoefficientPrime,
    ring_degree: LabiniusRingDegree,
    matrix_view_digest: [u8; 32],
}

impl<F: SmoothFftField, const D: usize, M: TrinomialModulus> BinaryClearSetup<F, D, M> {
    /// Check geometry, challenge budget, and accepted-interval no-wrap.
    ///
    /// The matrix is row-major with `n_a * m` elements. No SIS lookup is made.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        matrix: Vec<TrinomialRing<F, D, M>>,
        n_a: usize,
        m: usize,
        columns: usize,
        lower: i64,
        upper: i64,
        lambda_fold: u32,
        profile: BinaryChallengeProfile,
        coefficient_prime: LabiniusCoefficientPrime,
        ring_degree: LabiniusRingDegree,
    ) -> Result<Self, AkitaError> {
        if n_a == 0
            || !m.is_power_of_two()
            || !columns.is_power_of_two()
            || !matches!(
                ring_degree,
                LabiniusRingDegree::D162 | LabiniusRingDegree::D324 | LabiniusRingDegree::D648
            )
            || usize::try_from(ring_degree.degree()).ok() != Some(D)
            || profile.scalar_ring() != BinaryScalarRing::Cyclotomic243
            || lower > 0
            || upper < 0
            || lower > upper
        {
            return Err(AkitaError::InvalidSetup(
                "invalid clear binary setup geometry or identity".into(),
            ));
        }
        let k = usize::try_from(ring_degree.packing_degree())
            .map_err(|_| AkitaError::InvalidSetup("packing rank conversion overflow".into()))?;
        let expected_sign = if k == 1 { 1 } else { -1 };
        if M::MIDDLE_COEFFICIENT != expected_sign {
            return Err(AkitaError::InvalidSetup(
                "trinomial sign does not match packing rank".into(),
            ));
        }
        let actual_modulus = match F::MODULUS_BITS {
            64 => (1u128 << 64).checked_sub(F::OFFSET),
            128 => u128::MAX
                .checked_sub(F::OFFSET)
                .and_then(|value| value.checked_add(1)),
            _ => None,
        };
        if actual_modulus != Some(coefficient_prime.modulus()) {
            return Err(AkitaError::InvalidSetup(
                "coefficient prime does not match field".into(),
            ));
        }
        let matrix_len = checked::product([n_a, m])
            .ok_or_else(|| AkitaError::InvalidSetup("matrix size overflow".into()))?;
        if matrix.len() != matrix_len {
            return Err(AkitaError::InvalidSetup("matrix length mismatch".into()));
        }
        let scalar_rows = checked::product([k, m])
            .ok_or_else(|| AkitaError::InvalidSetup("scalar row count overflow".into()))?;
        let source_len = checked::product([scalar_rows, columns])
            .ok_or_else(|| AkitaError::InvalidSetup("source table size overflow".into()))?;
        // Validate every fixed-size allocation and proof-message extent before replay.
        for extent in [
            checked::product([matrix_len, D, F::NUM_BYTES]),
            checked::product([n_a, columns, D, F::NUM_BYTES]),
            checked::product([scalar_rows, 162, 8]),
            checked::product([source_len, core::mem::size_of::<u128>()]),
            checked::product([columns, 21]),
        ] {
            if extent.is_none_or(|bytes| bytes > isize::MAX as usize) {
                return Err(AkitaError::InvalidSetup(
                    "clear binary allocation extent overflow".into(),
                ));
            }
        }
        let fold_width = u64::try_from(columns)
            .map_err(|_| AkitaError::InvalidSetup("fold width conversion overflow".into()))?;
        if !profile.meets_budget(fold_width, lambda_fold) {
            return Err(AkitaError::InvalidSetup(
                "binary challenge profile fails fold budget".into(),
            ));
        }
        TrinomialNttDomain::<F, D, M>::new()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let mut header = Vec::new();
        crate::codec::length_prefixed(&mut header, b"akita/labinius/clear-matrix-view/v1")?;
        for dimension in [n_a, m, D] {
            append_size(&mut header, dimension)?;
        }
        header.extend_from_slice(&M::MIDDLE_COEFFICIENT.to_le_bytes());
        let coefficient_count = checked::product([matrix_len, D])
            .ok_or_else(|| AkitaError::InvalidSetup("matrix coefficient count overflow".into()))?;
        let mut coefficients = Vec::new();
        coefficients
            .try_reserve_exact(coefficient_count)
            .map_err(|_| AkitaError::InvalidSetup("matrix digest allocation failed".into()))?;
        for element in &matrix {
            coefficients.extend_from_slice(element.coefficients());
        }
        let matrix_view_digest = crate::channel::matrix_digest(&header, &coefficients)?;
        let diameter = u128::try_from(i128::from(upper) - i128::from(lower))
            .map_err(|_| AkitaError::InvalidSetup("response interval diameter overflow".into()))?;
        let occurrence = SourceOccurrenceBound::binary_extracted(
            u128::from(profile.multiplication_linf_operator_bound()),
            diameter,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("source comparison bound overflow".into()))?;
        checked_source_comparison_class_bound(
            LabiniusSourceComparisonId {
                coefficient_prime,
                ring_degree,
                matrix_view_digest,
            },
            &[occurrence],
        )?;
        Ok(Self {
            matrix,
            n_a,
            m,
            columns,
            scalar_rows,
            source_len,
            k,
            lower,
            upper,
            lambda_fold,
            profile,
            coefficient_prime,
            ring_degree,
            matrix_view_digest,
        })
    }

    /// Explicit row-major Ajtai matrix.
    pub fn matrix(&self) -> &[TrinomialRing<F, D, M>] {
        &self.matrix
    }
    /// Number of Ajtai matrix rows.
    pub fn n_a(&self) -> usize {
        self.n_a
    }
    /// Number of packed ring elements in each source column.
    pub fn m(&self) -> usize {
        self.m
    }
    /// Number of source columns and fold challenges.
    pub fn columns(&self) -> usize {
        self.columns
    }
    /// Number of scalar components per packed ring element.
    pub fn k(&self) -> usize {
        self.k
    }
    /// Number of scalar rows per source column.
    pub fn scalar_rows(&self) -> usize {
        self.scalar_rows
    }
    /// Exactly required host source length.
    pub fn source_len(&self) -> usize {
        self.source_len
    }
    /// Total number of multilinear coordinates.
    pub fn num_vars(&self) -> usize {
        self.source_len.trailing_zeros() as usize
    }
    /// Low-order coordinates selecting scalar rows.
    pub fn row_vars(&self) -> usize {
        self.scalar_rows.trailing_zeros() as usize
    }
    /// Inclusive accepted integer lower endpoint.
    pub fn lower(&self) -> i64 {
        self.lower
    }
    /// Inclusive accepted integer upper endpoint.
    pub fn upper(&self) -> i64 {
        self.upper
    }
    /// Admitted folding entropy budget.
    pub fn lambda_fold(&self) -> u32 {
        self.lambda_fold
    }
    /// Complete sampled-challenge profile.
    pub fn profile(&self) -> &BinaryChallengeProfile {
        &self.profile
    }
    /// Versioned matrix-view digest.
    pub fn matrix_view_digest(&self) -> &[u8; 32] {
        &self.matrix_view_digest
    }

    /// Canonical public setup identity, including the sealed host profile.
    pub fn identity_bytes<H: SwitchField>(&self) -> Result<Vec<u8>, AkitaError> {
        let mut bytes = Vec::new();
        let host_tag: &[u8] = if H::ROWS == 128 {
            b"f128-source-u128"
        } else {
            b"f192-source-u64"
        };
        crate::codec::length_prefixed(&mut bytes, host_tag)?;
        append_size(&mut bytes, 162)?;
        append_size(&mut bytes, D)?;
        bytes.extend_from_slice(&M::MIDDLE_COEFFICIENT.to_le_bytes());
        crate::codec::length_prefixed(&mut bytes, self.coefficient_prime.label().as_bytes())?;
        for dimension in [self.k, self.m, self.columns, self.n_a] {
            append_size(&mut bytes, dimension)?;
        }
        bytes.extend_from_slice(&self.lower.to_le_bytes());
        bytes.extend_from_slice(&self.upper.to_le_bytes());
        bytes.extend_from_slice(&self.lambda_fold.to_le_bytes());
        crate::codec::length_prefixed(&mut bytes, self.profile.identity_bytes())?;
        bytes.extend_from_slice(&self.matrix_view_digest);
        // Ring degree is already bound by D, scalar degree, and modulus sign.
        if usize::try_from(self.ring_degree.degree()).ok() != Some(D) {
            return Err(AkitaError::InvalidSetup(
                "setup ring identity mismatch".into(),
            ));
        }
        Ok(bytes)
    }
}

fn append_size(bytes: &mut Vec<u8>, value: usize) -> Result<(), AkitaError> {
    let value = u64::try_from(value)
        .map_err(|_| AkitaError::InvalidSetup("identity size conversion overflow".into()))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}
