//! Root setup derivation for a nested Akita setup.

use akita_error::AkitaError;
use akita_labinius_verifier::codec::length_prefixed;
use akita_types::proof::AkitaSetupSeed;
use jolt_field::Prime128Offset275;

use crate::{session::canonical_bytes, shipped::SupportedGeometry, RootSetup};

/// Domain label of the root matrix seed.
const ROOT_MATRIX_SEED_DOMAIN: &[u8] = b"akita/labinius/root-matrix-seed/v1";

/// Admit `geometry` and derive its public root matrix for the nested Akita
/// setup with seed `nested_seed`.
///
/// The matrix seed is `Blake2b-256(LP(domain) || LP(nested_seed))`, with `LP`
/// the u64 little-endian length prefix and the seed in its canonical
/// compressed encoding, and it keeps the nested seed's derivation algorithm.
/// The matrix is the `Prime128Offset275` stream of that seed reduced modulo
/// `q`, for every proof-field family: the stream field fixes the reduction
/// bias, and the label keeps the matrix from being a reduced prefix of the
/// nested Akita setup matrix.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the seed cannot be encoded, or
/// the admission and derivation errors of `AdmittedRootSetup::derive`.
pub fn derive_root_setup(
    geometry: SupportedGeometry,
    nested_seed: &AkitaSetupSeed,
) -> Result<RootSetup, AkitaError> {
    let invalid =
        |error: AkitaError| AkitaError::InvalidSetup(format!("root matrix seed: {error}"));
    let mut preimage = Vec::new();
    length_prefixed(&mut preimage, ROOT_MATRIX_SEED_DOMAIN).map_err(invalid)?;
    length_prefixed(
        &mut preimage,
        &canonical_bytes(nested_seed).map_err(invalid)?,
    )
    .map_err(invalid)?;
    RootSetup::derive::<Prime128Offset275>(
        geometry.profile,
        geometry.log_num_cells,
        geometry.log_fold_width,
        geometry.lambda_fold,
        AkitaSetupSeed {
            derivation: nested_seed.derivation,
            seed: akita_params::digest_descriptor_bytes(&preimage),
        },
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use akita_params::sis::labinius::LabiniusRootProfile;

    /// The matrix seed is the digest of the spelled-out byte layout, and the
    /// nested seed never reaches the matrix stream unchanged.
    #[test]
    fn root_matrix_seed_is_the_labelled_digest_of_the_nested_seed() {
        let geometry = SupportedGeometry {
            profile: LabiniusRootProfile::D648Q25BoundedW46,
            log_num_cells: 4,
            log_fold_width: 1,
            lambda_fold: 128,
        };
        let nested = AkitaSetupSeed::shake256_paged_v1([0x5a; 32]);
        let mut preimage = Vec::new();
        preimage.extend_from_slice(&34u64.to_le_bytes());
        preimage.extend_from_slice(b"akita/labinius/root-matrix-seed/v1");
        preimage.extend_from_slice(&33u64.to_le_bytes());
        preimage.push(1);
        preimage.extend_from_slice(&nested.seed);
        let expected =
            AkitaSetupSeed::shake256_paged_v1(akita_params::digest_descriptor_bytes(&preimage));

        let admitted = derive_root_setup(geometry, &nested).unwrap();
        assert_eq!(admitted.seed(), &expected);
        assert_ne!(admitted.seed(), &nested);
        let direct = RootSetup::derive::<Prime128Offset275>(
            geometry.profile,
            geometry.log_num_cells,
            geometry.log_fold_width,
            geometry.lambda_fold,
            nested,
        )
        .unwrap();
        assert_ne!(admitted.setup().matrix(), direct.setup().matrix());
    }
}
