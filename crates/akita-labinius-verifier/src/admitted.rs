//! Seed-derived, SIS-admitted binary-root setup and its public identity.

use akita_algebra::{binary::field_switch::SwitchField, ring::TrinomialModulus};
use akita_error::{checked, AkitaError};
use akita_params::sis::labinius::{LabiniusRootProfile, LabiniusRootShape};
use akita_types::proof::{
    derive_public_matrix_prefix, AkitaSetupSeed, PublicMatrixDerivation,
    MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
};
use jolt_field::{CanonicalEncoding, Field};

use crate::BinaryClearSetup;

// Resource limits for this materialized coefficient-form API, not stream limits.
// The transient stream prefix is held in its field representation.
const MAX_MATRIX_BYTES: usize = 1 << 26;
const MAX_PREFIX_BYTES: usize = 1 << 27;

/// Derive the tight row-major matrix of residues from the seed's public field
/// prefix over the stream field `P`.
///
/// Coefficient `t` of element `(i, j)` is at `(i * columns + j) * degree + t`.
/// The seed is passed unchanged to the canonical expander; each canonical
/// prefix coefficient is then reduced modulo `q`. `P` only names the stream
/// being reduced and never enters the result's type. Empty dimensions, invalid
/// degrees, unsupported field descriptors, and excessive allocations return
/// `InvalidSetup` before expansion.
pub fn derive_trinomial_matrix<P: Field + CanonicalEncoding>(
    seed: &AkitaSetupSeed,
    rows: usize,
    columns: usize,
    degree: usize,
    q: u32,
) -> Result<Vec<u32>, AkitaError> {
    if rows == 0 || columns == 0 || degree == 0 || !degree.is_multiple_of(2) || q == 0 {
        return Err(AkitaError::InvalidSetup(
            "invalid trinomial matrix dimensions or degree".into(),
        ));
    }
    // The expander expects this descriptor to succeed internally.
    akita_params::field_modulus_be_bytes::<P>()?;
    let coefficient_count = checked::product([rows, columns, degree])
        .ok_or_else(|| AkitaError::InvalidSetup("trinomial coefficient count overflow".into()))?;
    let prefix_bytes = checked::product([coefficient_count, size_of::<P>()]);
    let matrix_bytes = checked::product([coefficient_count, size_of::<u32>()]);
    if coefficient_count > MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
        || P::NUM_BYTES > size_of::<u128>()
        || prefix_bytes.is_none_or(|bytes| bytes > MAX_PREFIX_BYTES)
        || matrix_bytes.is_none_or(|bytes| bytes > MAX_MATRIX_BYTES)
    {
        return Err(AkitaError::InvalidSetup(
            "trinomial matrix exceeds materialization budget".into(),
        ));
    }
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(coefficient_count)
        .map_err(|_| AkitaError::InvalidSetup("trinomial matrix allocation failed".into()))?;
    // The frozen expander allocates infallibly. Probe its bounded allocation
    // fallibly first, then release the probe before the canonical expansion.
    // As with other allocations, concurrent/global allocator exhaustion remains
    // possible; this check cannot change the expander's allocation API.
    let mut allocation_probe = Vec::<P>::new();
    allocation_probe
        .try_reserve_exact(coefficient_count)
        .map_err(|_| AkitaError::InvalidSetup("public prefix allocation failed".into()))?;
    drop(allocation_probe);
    let prefix = derive_public_matrix_prefix::<P>(coefficient_count, seed);
    for coefficient in prefix.as_field_slice() {
        let mut bytes = [0u8; size_of::<u128>()];
        coefficient.to_bytes_le(
            bytes
                .get_mut(..P::NUM_BYTES)
                .ok_or_else(|| AkitaError::InvalidSetup("stream field exceeds 128 bits".into()))?,
        );
        let residue = u128::from_le_bytes(bytes) % u128::from(q);
        matrix.push(u32::try_from(residue).map_err(|_| {
            AkitaError::InvalidSetup("reduced coefficient conversion overflow".into())
        })?);
    }
    Ok(matrix)
}

/// Characteristic of the prime field `F`, which must fit in 128 bits.
///
/// This is the prime that proof-field admission and the derivation-bias check
/// take.
pub fn field_characteristic<F: Field + CanonicalEncoding>() -> Result<u128, AkitaError> {
    let too_wide = || AkitaError::InvalidSetup("field characteristic exceeds 128 bits".into());
    let modulus_bytes = akita_params::field_modulus_be_bytes::<F>()?;
    let (high, low) = modulus_bytes.split_at_checked(16).ok_or_else(too_wide)?;
    <[u8; 16]>::try_from(low)
        .ok()
        .map(u128::from_be_bytes)
        .filter(|_| high.iter().all(|&byte| byte == 0))
        .ok_or_else(too_wide)
}

/// A root shape admitted by the certified table, with its seed-derived matrix.
///
/// The only constructor is [`Self::derive`]; a caller-supplied matrix cannot
/// acquire this type's admission identity. The clear opening API consumes the
/// immutable [`Self::setup`] view.
#[derive(Clone, Debug)]
pub struct AdmittedRootSetup<const D: usize, M: TrinomialModulus> {
    setup: BinaryClearSetup<D, M>,
    shape: LabiniusRootShape,
    seed: AkitaSetupSeed,
    log_num_cells: u32,
    log_fold_width: u32,
    lambda_fold: u32,
}

impl<const D: usize, M: TrinomialModulus> AdmittedRootSetup<D, M> {
    /// Admit the geometry, derive its public matrix from the stream field `P`,
    /// and validate the clear setup.
    ///
    /// The matrix is the stream reduced modulo `q`, so `P` must be large
    /// enough for the reduction bias to stay below the policy floor
    /// ([`LabiniusRootShape::check_derivation_bias`]). The accepted response
    /// interval is the shape's balanced base-16 range.
    pub fn derive<P: Field + CanonicalEncoding>(
        profile: LabiniusRootProfile,
        log_num_cells: u32,
        log_fold_width: u32,
        lambda_fold: u32,
        seed: AkitaSetupSeed,
    ) -> Result<Self, AkitaError> {
        let shape = LabiniusRootShape::derive(profile, log_num_cells, log_fold_width, lambda_fold)?;
        let n_a = usize::try_from(shape.rank_a())
            .map_err(|_| AkitaError::InvalidSetup("root rank conversion overflow".into()))?;
        shape.check_derivation_bias(field_characteristic::<P>()?)?;
        let (lower, upper) = shape.response().interval();
        let lower = i64::try_from(lower)
            .map_err(|_| AkitaError::InvalidSetup("root lower endpoint exceeds i64".into()))?;
        let upper = i64::try_from(upper)
            .map_err(|_| AkitaError::InvalidSetup("root upper endpoint exceeds i64".into()))?;
        let matrix = derive_trinomial_matrix::<P>(
            &seed,
            n_a,
            shape.ring_elements_per_column(),
            D,
            profile.commitment_modulus().modulus(),
        )?;
        // This canonical constructor already checks D, M's sign and reduction.
        let setup = BinaryClearSetup::new(
            matrix,
            n_a,
            shape.ring_elements_per_column(),
            shape.fold_width(),
            lower,
            upper,
            lambda_fold,
            profile.challenge_profile()?,
            profile.ring_degree(),
            profile.commitment_modulus(),
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
    pub fn setup(&self) -> &BinaryClearSetup<D, M> {
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
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use akita_algebra::MinusTrinomial;
    use jolt_field::{CanonicalBytes, Prime128Offset275};

    #[test]
    fn derived_matrix_is_the_reduced_canonical_prefix() {
        let profile = LabiniusRootProfile::D648Q25BoundedW46;
        let seed = AkitaSetupSeed::shake256_paged_v1([0x28; 32]);
        let admitted = AdmittedRootSetup::<648, MinusTrinomial>::derive::<Prime128Offset275>(
            profile,
            4,
            1,
            128,
            seed.clone(),
        )
        .unwrap();
        let setup = admitted.setup();
        let q = setup.modulus();
        let prefix = derive_public_matrix_prefix::<Prime128Offset275>(setup.matrix().len(), &seed);
        for (&residue, original) in setup.matrix().iter().zip(prefix.as_field_slice()) {
            let mut bytes = [0u8; 16];
            original.to_bytes_le(&mut bytes);
            assert_eq!(
                u128::from(residue),
                u128::from_le_bytes(bytes) % u128::from(q)
            );
        }
        // A caller-supplied matrix must already be reduced.
        let mut unreduced = setup.matrix().to_vec();
        unreduced[0] = q;
        assert!(matches!(
            BinaryClearSetup::<648, MinusTrinomial>::new(
                unreduced,
                setup.n_a(),
                setup.m(),
                setup.columns(),
                setup.lower(),
                setup.upper(),
                128,
                setup.profile().clone(),
                profile.ring_degree(),
                profile.commitment_modulus(),
            ),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}
