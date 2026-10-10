//! Explicit setup admission and transcript identity.
//!
//! Admission here checks geometry, fold entropy, and integer no-wrap only.
//! `BinaryClearSetup::new` takes the matrix from its caller and performs no SIS
//! or width-table security admission; `crate::admitted::AdmittedRootSetup` is
//! the seed-derived construction that does.
//!
//! The setup is arithmetic modulo the commitment prime `q` only: it carries no
//! proof-field type. The proof field enters at `crate::lowered`.

use core::marker::PhantomData;

use akita_algebra::binary::field_switch::SwitchField;
use akita_algebra::ring::TrinomialModulus;
use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::{
    checked_source_comparison_class_bound, LabiniusCommitmentModulus, LabiniusRingDegree,
    LabiniusSourceComparisonId, SourceOccurrenceBound,
};

/// Admitted explicit setup for the standalone clear differential oracle.
#[derive(Clone, Debug)]
pub struct BinaryClearSetup<const D: usize, M: TrinomialModulus> {
    /// Canonical residues below `q`, ordered by row, element, then coefficient.
    matrix: Vec<u32>,
    /// Exact integer `H_i`, ordered by row, then coefficient.
    a_offset_remainders: Vec<i128>,
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
    ring_degree: LabiniusRingDegree,
    matrix_view_digest: [u8; 32],
    commitment_modulus: LabiniusCommitmentModulus,
    trinomial: PhantomData<fn() -> M>,
}

impl<const D: usize, M: TrinomialModulus> BinaryClearSetup<D, M> {
    /// Check geometry, challenge budget, and accepted-interval no-wrap.
    ///
    /// The matrix is row-major with `n_a * m` elements of `D` canonical
    /// residues below the commitment prime. No SIS lookup is made. Every
    /// accepted response coefficient lies in `[lower, upper]`; the interval
    /// need not be symmetric.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        matrix: Vec<u32>,
        n_a: usize,
        m: usize,
        columns: usize,
        lower: i64,
        upper: i64,
        lambda_fold: u32,
        profile: BinaryChallengeProfile,
        ring_degree: LabiniusRingDegree,
        commitment_modulus: LabiniusCommitmentModulus,
    ) -> Result<Self, AkitaError> {
        let modulus = commitment_modulus.modulus();
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
        let matrix_len = checked::product([n_a, m, D])
            .ok_or_else(|| AkitaError::InvalidSetup("matrix size overflow".into()))?;
        if matrix.len() != matrix_len {
            return Err(AkitaError::InvalidSetup("matrix length mismatch".into()));
        }
        if matrix.iter().any(|&coefficient| coefficient >= modulus) {
            return Err(AkitaError::InvalidSetup(
                "matrix coefficient is not reduced modulo the commitment prime".into(),
            ));
        }
        let scalar_rows = checked::product([k, m])
            .ok_or_else(|| AkitaError::InvalidSetup("scalar row count overflow".into()))?;
        let source_len = checked::product([scalar_rows, columns])
            .ok_or_else(|| AkitaError::InvalidSetup("source table size overflow".into()))?;
        // Validate every fixed-size allocation and proof-message extent before replay.
        for extent in [
            checked::product([matrix_len, size_of::<u32>()]),
            checked::product([n_a, columns, D, size_of::<u32>()]),
            checked::product([n_a, D, size_of::<i128>()]),
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
        let a_offset_remainders = a_offset_remainders::<D, M>(&matrix, n_a, k, lower, upper)?;
        let matrix_view_digest = matrix_view_digest::<D, M>(&matrix, n_a, m, commitment_modulus)?;
        let diameter = u128::try_from(i128::from(upper) - i128::from(lower))
            .map_err(|_| AkitaError::InvalidSetup("response interval diameter overflow".into()))?;
        let occurrence = SourceOccurrenceBound::binary_extracted(
            u128::from(profile.multiplication_linf_operator_bound()),
            diameter,
        )
        .ok_or_else(|| AkitaError::InvalidSetup("source comparison bound overflow".into()))?;
        checked_source_comparison_class_bound(
            LabiniusSourceComparisonId {
                commitment_modulus,
                ring_degree,
                matrix_view_digest,
            },
            &[occurrence],
        )?;
        Ok(Self {
            matrix,
            a_offset_remainders,
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
            ring_degree,
            matrix_view_digest,
            commitment_modulus,
            trinomial: PhantomData,
        })
    }

    /// Closed commitment modulus identity.
    pub fn commitment_modulus(&self) -> LabiniusCommitmentModulus {
        self.commitment_modulus
    }
    /// The commitment prime `q`.
    pub fn modulus(&self) -> u32 {
        self.commitment_modulus.modulus()
    }
    /// Row-major Ajtai matrix: canonical residues of element `(i, j)` are at
    /// `(i * m + j) * D .. + D`.
    pub fn matrix(&self) -> &[u32] {
        &self.matrix
    }
    /// Matrix-bound exact integers `H_i = rem_Phi((sum_j A_ij) * O)`, ordered by
    /// row then coefficient. `O[t]` is the signed-packing response offset:
    /// `-lower` at a positively packed coefficient and `upper` at a negated one.
    pub fn a_offset_remainders(&self) -> &[i128] {
        &self.a_offset_remainders
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

/// Exact `H_i = rem_Phi((sum_j A_ij) * O)` over the integers, one row at a time.
fn a_offset_remainders<const D: usize, M: TrinomialModulus>(
    matrix: &[u32],
    n_a: usize,
    k: usize,
    lower: i64,
    upper: i64,
) -> Result<Vec<i128>, AkitaError> {
    let overflow = || AkitaError::InvalidSetup("response offset remainder overflow".into());
    // A scalar coefficient in `[lower, upper]` is stored as `v - lower` where
    // it is packed with sign `+1` and as `upper - v` where it is negated.
    let (positive_offset, negated_offset) = (-i128::from(lower), i128::from(upper));
    let row_len = matrix
        .len()
        .checked_div(n_a)
        .filter(|&len| len != 0 && D != 0)
        .ok_or_else(overflow)?;
    let unreduced = checked::product([2, D])
        .and_then(|len| len.checked_sub(1))
        .ok_or_else(overflow)?;
    let mut remainders = Vec::new();
    remainders
        .try_reserve_exact(checked::product([n_a, D]).ok_or_else(overflow)?)
        .map_err(|_| AkitaError::InvalidSetup("offset remainder allocation failed".into()))?;
    let mut column_sum = Vec::new();
    column_sum
        .try_reserve_exact(D)
        .map_err(|_| AkitaError::InvalidSetup("offset remainder allocation failed".into()))?;
    let mut product = Vec::new();
    product
        .try_reserve_exact(unreduced)
        .map_err(|_| AkitaError::InvalidSetup("offset remainder allocation failed".into()))?;
    for row in matrix.chunks_exact(row_len) {
        column_sum.clear();
        column_sum.resize(D, 0i128);
        for element in row.chunks_exact(D) {
            for (sum, &coefficient) in column_sum.iter_mut().zip(element) {
                *sum = sum
                    .checked_add(i128::from(coefficient))
                    .ok_or_else(overflow)?;
            }
        }
        product.clear();
        product.resize(unreduced, 0i128);
        for (s, &sum) in column_sum.iter().enumerate() {
            let end = checked::sum([s, D]).ok_or_else(overflow)?;
            for (t, destination) in product
                .get_mut(s..end)
                .ok_or_else(overflow)?
                .iter_mut()
                .enumerate()
            {
                let positive = k == 1 || (t / k).is_multiple_of(2);
                let factor = if positive {
                    positive_offset
                } else {
                    negated_offset
                };
                *destination = sum
                    .checked_mul(factor)
                    .and_then(|term| destination.checked_add(term))
                    .ok_or_else(overflow)?;
            }
        }
        crate::commitment::reduce_trinomial::<M>(&mut product, D)?;
        remainders.extend_from_slice(product.get(..D).ok_or_else(overflow)?);
    }
    Ok(remainders)
}

fn append_size(bytes: &mut Vec<u8>, value: usize) -> Result<(), AkitaError> {
    let value = u64::try_from(value)
        .map_err(|_| AkitaError::InvalidSetup("identity size conversion overflow".into()))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn matrix_view_digest<const D: usize, M: TrinomialModulus>(
    matrix: &[u32],
    n_a: usize,
    m: usize,
    modulus: LabiniusCommitmentModulus,
) -> Result<[u8; 32], AkitaError> {
    let mut header = Vec::new();
    crate::codec::length_prefixed(&mut header, b"akita/labinius/clear-matrix-view/v1")?;
    header.extend_from_slice(&modulus.modulus().to_le_bytes());
    for dimension in [n_a, m, D] {
        append_size(&mut header, dimension)?;
    }
    header.extend_from_slice(&M::MIDDLE_COEFFICIENT.to_le_bytes());
    crate::channel::matrix_digest(&header, matrix)
}
