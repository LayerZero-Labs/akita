//! Seed-derived, SIS-admitted binary-root setup and its public identity.

use akita_algebra::{binary::field_switch::SwitchField, ring::TrinomialModulus};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::{
    derive_centered_q_matrix, SetupStreamLabel, SetupStreamModulus, SetupStreamRing,
    SetupStreamRole,
};
use akita_params::sis::labinius::{LabiniusRootProfile, LabiniusRootShape};
use akita_types::proof::{AkitaSetupSeed, PublicMatrixDerivation};
use jolt_field::{CanonicalEncoding, Field};

use crate::BinaryClearSetup;

/// Derive the tight row-major q-uniform trinomial matrix for a binary group.
///
/// Coefficient `t` of element `(i, j)` is at `(i * columns + j) * degree + t`.
/// Each entry uses its own typed q stream; P does not supply q candidates.
/// Returned values are canonical residues for the existing clear setup API;
/// the canonical stream first samples their centered integer representatives.
/// `P` remains a validated proof-field descriptor; it does not seed the q stream.
///
/// # Errors
/// Rejects invalid shapes, unsupported fields, or excessive allocation.
pub fn derive_trinomial_matrix<P: Field + CanonicalEncoding, M: TrinomialModulus>(
    seed: &AkitaSetupSeed,
    rows: usize,
    columns: usize,
    degree: usize,
    q: u32,
) -> Result<Vec<u32>, AkitaError> {
    akita_params::field_modulus_be_bytes::<P>()?;
    let rank = u32::try_from(rows)
        .map_err(|_| AkitaError::InvalidSetup("root rank exceeds u32".into()))?;
    let degree = u32::try_from(degree)
        .map_err(|_| AkitaError::InvalidSetup("root degree exceeds u32".into()))?;
    let label = SetupStreamLabel {
        role: SetupStreamRole::Commitment,
        group: 0,
        level: 0,
        rank,
        row: 0,
        column: 0,
        ring: if M::MIDDLE_COEFFICIENT > 0 {
            SetupStreamRing::PlusTrinomial
        } else {
            SetupStreamRing::MinusTrinomial
        },
        degree,
        modulus: SetupStreamModulus::Q(q),
    };
    let centered = derive_centered_q_matrix(&seed.seed, label, columns)?;
    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(centered.len())
        .map_err(|_| AkitaError::InvalidSetup("trinomial matrix allocation failed".into()))?;
    for value in centered {
        matrix.push(
            u32::try_from(value.rem_euclid(i64::from(q)))
                .map_err(|_| AkitaError::InvalidSetup("q residue exceeds u32".into()))?,
        );
    }
    Ok(matrix)
}

/// Characteristic of the prime field `F`, which must fit in 128 bits.
///
/// This is the prime used by proof-field admission.
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
    /// Admit the geometry, derive its public matrix by exact q sampling,
    /// and validate the clear setup.
    ///
    /// The typed q stream is independent of `P`; no P-to-q reduction bias
    /// occurs. The accepted response interval is the shape's balanced
    /// base-16 range.
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
        let (lower, upper) = shape.response().interval();
        let lower = i64::try_from(lower)
            .map_err(|_| AkitaError::InvalidSetup("root lower endpoint exceeds i64".into()))?;
        let upper = i64::try_from(upper)
            .map_err(|_| AkitaError::InvalidSetup("root upper endpoint exceeds i64".into()))?;
        let matrix = derive_trinomial_matrix::<P, M>(
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
    use jolt_field::Prime128Offset275;

    #[test]
    fn derived_matrix_is_the_exact_typed_q_stream() {
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
        let label = SetupStreamLabel {
            role: SetupStreamRole::Commitment,
            group: 0,
            level: 0,
            rank: u32::try_from(setup.n_a()).unwrap(),
            row: 0,
            column: 0,
            ring: SetupStreamRing::MinusTrinomial,
            degree: 648,
            modulus: SetupStreamModulus::Q(q),
        };
        let expected = derive_centered_q_matrix(&seed.seed, label, setup.m()).unwrap();
        for (&residue, centered) in setup.matrix().iter().zip(expected) {
            assert_eq!(i64::from(residue), centered.rem_euclid(i64::from(q)));
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
