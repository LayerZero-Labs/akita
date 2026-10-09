//! Seed-derived, SIS-admitted binary-root setup and its public identity.

use akita_algebra::{
    binary::field_switch::SwitchField,
    fft::SmoothFftField,
    ring::{TrinomialModulus, TrinomialRing},
};
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::{
    LabiniusCommitmentModulus, LabiniusRootProfile, LabiniusRootShape,
};
use akita_types::proof::{
    derive_public_matrix_prefix, AkitaSetupSeed, PublicMatrixDerivation,
    MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
};
use jolt_field::{CanonicalEncoding, Field};

use crate::BinaryClearSetup;

// Resource limits for this materialized coefficient-form API, not stream limits.
const MAX_MATRIX_BYTES: usize = 1 << 26;
const MAX_RING_BYTES: usize = 1 << 16;

/// Derive the tight row-major trinomial view of the seed's public field prefix.
///
/// Coefficient `t` of element `(i, j)` is at `(i * columns + j) * D + t`.
/// The seed is passed unchanged to the canonical expander. Empty dimensions,
/// invalid degrees, unsupported field descriptors, and excessive allocations
/// return `InvalidSetup` before expansion.
pub fn derive_trinomial_matrix<
    F: Field + CanonicalEncoding,
    const D: usize,
    M: TrinomialModulus,
>(
    seed: &AkitaSetupSeed,
    rows: usize,
    columns: usize,
) -> Result<Vec<TrinomialRing<F, D, M>>, AkitaError> {
    if rows == 0 || columns == 0 || D == 0 || !D.is_multiple_of(2) {
        return Err(AkitaError::InvalidSetup(
            "invalid trinomial matrix dimensions or degree".into(),
        ));
    }
    // The expander expects this descriptor to succeed internally.
    akita_params::field_modulus_be_bytes::<F>()?;
    let matrix_len = checked::product([rows, columns])
        .ok_or_else(|| AkitaError::InvalidSetup("trinomial matrix size overflow".into()))?;
    let coefficient_count = checked::product([rows, columns, D])
        .ok_or_else(|| AkitaError::InvalidSetup("trinomial coefficient count overflow".into()))?;
    let field_bytes = checked::product([coefficient_count, core::mem::size_of::<F>()]);
    let matrix_bytes =
        checked::product([matrix_len, core::mem::size_of::<TrinomialRing<F, D, M>>()]);
    if coefficient_count > MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
        || core::mem::size_of::<TrinomialRing<F, D, M>>() > MAX_RING_BYTES
        || [field_bytes, matrix_bytes]
            .into_iter()
            .any(|extent| extent.is_none_or(|bytes| bytes > MAX_MATRIX_BYTES))
    {
        return Err(AkitaError::InvalidSetup(
            "trinomial matrix exceeds materialization budget".into(),
        ));
    }
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(matrix_len)
        .map_err(|_| AkitaError::InvalidSetup("trinomial matrix allocation failed".into()))?;
    // The frozen expander allocates infallibly. Probe its bounded allocation
    // fallibly first, then release the probe before the canonical expansion.
    // As with other allocations, concurrent/global allocator exhaustion remains
    // possible; this check cannot change the expander's allocation API.
    let mut allocation_probe = Vec::<F>::new();
    allocation_probe
        .try_reserve_exact(coefficient_count)
        .map_err(|_| AkitaError::InvalidSetup("public prefix allocation failed".into()))?;
    drop(allocation_probe);
    let prefix = derive_public_matrix_prefix::<F>(coefficient_count, seed);
    materialize_trinomial_matrix(prefix.as_field_slice(), matrix)
}

// Keep fixed-size ring temporaries out of the public entry point's stack frame.
// In particular, debug builds must reject a huge const D before reserving a
// frame containing [F; D]. Only the resource-checked path above calls here.
#[inline(never)]
fn materialize_trinomial_matrix<F: Field, const D: usize, M: TrinomialModulus>(
    coefficients: &[F],
    mut matrix: Vec<TrinomialRing<F, D, M>>,
) -> Result<Vec<TrinomialRing<F, D, M>>, AkitaError> {
    for chunk in coefficients.chunks_exact(D) {
        let coefficients: [F; D] = chunk.try_into().map_err(|_| {
            AkitaError::InvalidSetup("trinomial coefficient length mismatch".into())
        })?;
        matrix.push(
            TrinomialRing::from_coefficients(coefficients)
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?,
        );
    }
    Ok(matrix)
}

/// A root shape admitted by the certified table, with its seed-derived matrix.
///
/// The only constructor is [`Self::derive`]; a caller-supplied matrix cannot
/// acquire this type's admission identity. The clear opening API consumes the
/// immutable [`Self::setup`] view.
#[derive(Clone, Debug)]
pub struct AdmittedRootSetup<F, const D: usize, M: TrinomialModulus> {
    setup: BinaryClearSetup<F, D, M>,
    shape: LabiniusRootShape,
    seed: AkitaSetupSeed,
    log_num_cells: u32,
    log_fold_width: u32,
    lambda_fold: u32,
}

impl<F: SmoothFftField, const D: usize, M: TrinomialModulus> AdmittedRootSetup<F, D, M> {
    /// Admit the geometry, derive its public matrix, and validate the clear setup.
    pub fn derive(
        profile: LabiniusRootProfile,
        log_num_cells: u32,
        log_fold_width: u32,
        lambda_fold: u32,
        seed: AkitaSetupSeed,
    ) -> Result<Self, AkitaError> {
        if profile.commitment_modulus() != LabiniusCommitmentModulus::CoefficientPrime {
            return Err(AkitaError::InvalidSetup(
                "LaBinius foreign commitment modulus protocol is not implemented".into(),
            ));
        }
        let shape = LabiniusRootShape::derive(profile, log_num_cells, log_fold_width, lambda_fold)?;
        let n_a = usize::try_from(shape.rank_a())
            .map_err(|_| AkitaError::InvalidSetup("root rank conversion overflow".into()))?;
        let (lower, upper) = profile.response_interval();
        let lower = i64::try_from(lower)
            .map_err(|_| AkitaError::InvalidSetup("root lower endpoint exceeds i64".into()))?;
        let upper = i64::try_from(upper)
            .map_err(|_| AkitaError::InvalidSetup("root upper endpoint exceeds i64".into()))?;
        let matrix = derive_trinomial_matrix(&seed, n_a, shape.ring_elements_per_column())?;
        // This canonical constructor already checks F's prime, D, and M's sign.
        let setup = BinaryClearSetup::new(
            matrix,
            n_a,
            shape.ring_elements_per_column(),
            shape.fold_width(),
            lower,
            upper,
            lambda_fold,
            profile.challenge_profile()?,
            profile.coefficient_prime(),
            profile.ring_degree(),
        )
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        Ok(Self {
            setup,
            shape,
            seed,
            log_num_cells,
            log_fold_width,
            lambda_fold,
        })
    }

    /// Immutable clear-protocol setup, including the derived matrix.
    pub fn setup(&self) -> &BinaryClearSetup<F, D, M> {
        &self.setup
    }
    /// Certified root geometry.
    pub fn shape(&self) -> &LabiniusRootShape {
        &self.shape
    }
    /// Unmodified public field-stream identity.
    pub fn seed(&self) -> &AkitaSetupSeed {
        &self.seed
    }
    /// Original base-two source-cell count.
    pub fn log_num_cells(&self) -> u32 {
        self.log_num_cells
    }
    /// Original base-two fold width.
    pub fn log_fold_width(&self) -> u32 {
        self.log_fold_width
    }
    /// Original fold entropy budget.
    pub fn lambda_fold(&self) -> u32 {
        self.lambda_fold
    }

    /// Bind the profile, derivation inputs, admitted rank, seed, and clear setup.
    ///
    /// Layout is specified in `specs/labinius-setup-contract.md`. The nested
    /// clear identity binds the host field and the existing matrix-view digest.
    pub fn identity_bytes<H: SwitchField>(&self) -> Result<Vec<u8>, AkitaError> {
        let domain = b"akita/labinius/admitted-root-setup/v1";
        let profile = self.shape.profile().identity_bytes()?;
        let clear = self
            .setup
            .identity_bytes::<H>()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        let capacity = checked::sum([8, domain.len(), 8, profile.len(), 16, 1, 32, 8, clear.len()])
            .ok_or_else(|| AkitaError::InvalidSetup("root identity size overflow".into()))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| AkitaError::InvalidSetup("root identity allocation failed".into()))?;
        crate::codec::length_prefixed(&mut bytes, domain)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        crate::codec::length_prefixed(&mut bytes, &profile)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        for value in [
            self.log_num_cells,
            self.log_fold_width,
            self.lambda_fold,
            self.shape.rank_a(),
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.push(match self.seed.derivation {
            PublicMatrixDerivation::Shake256PagedV1 => 1,
        });
        bytes.extend_from_slice(&self.seed.seed);
        crate::codec::length_prefixed(&mut bytes, &clear)
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::MinusTrinomial;
    use jolt_field::Prime128OffsetA7F7;

    #[test]
    fn small_commitment_modulus_rejects_before_protocol_admission() {
        let result = AdmittedRootSetup::<Prime128OffsetA7F7, 648, MinusTrinomial>::derive(
            LabiniusRootProfile::D648P128Q28BoundedW46Delta16,
            22,
            8,
            128,
            AkitaSetupSeed::shake256_paged_v1([0x28; 32]),
        );
        assert!(matches!(
            result,
            Err(AkitaError::InvalidSetup(message))
                if message.contains("foreign commitment modulus protocol is not implemented")
        ));
    }
}
