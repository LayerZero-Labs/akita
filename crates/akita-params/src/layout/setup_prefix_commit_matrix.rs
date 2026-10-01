//! Setup-prefix wire encoding for SIS commitment matrices.

use super::*;

pub(super) trait SetupPrefixCommitMatrixParams: Sized {
    const ROLE: SisMatrixRole;

    fn sis_modulus_profile(&self) -> SisModulusProfileId;
    fn security_policy(&self) -> SisSecurityPolicyId;
    fn sis_table_key(&self) -> Option<crate::SisTableKey>;
    fn output_rank(&self) -> usize;
    fn input_width(&self) -> usize;
    fn coeff_linf_bound(&self) -> Option<u128>;
    fn ring_dimension(&self) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn new_unchecked(
        policy: SisSecurityPolicyId,
        table_digest: SisTableDigest,
        sis_modulus_profile: SisModulusProfileId,
        output_rank: usize,
        input_width: usize,
        coeff_linf_bound: u128,
        ring_dimension: usize,
    ) -> Self;
}

macro_rules! impl_setup_prefix_commit_matrix_params {
    ($ty:ty, $role:expr) => {
        impl SetupPrefixCommitMatrixParams for $ty {
            const ROLE: SisMatrixRole = $role;

            fn sis_modulus_profile(&self) -> SisModulusProfileId {
                self.sis_modulus_profile()
            }
            fn security_policy(&self) -> SisSecurityPolicyId {
                self.security_policy()
            }
            fn sis_table_key(&self) -> Option<crate::SisTableKey> {
                Some(self.sis_table_key())
            }
            fn output_rank(&self) -> usize {
                self.output_rank()
            }
            fn input_width(&self) -> usize {
                self.input_width()
            }
            fn coeff_linf_bound(&self) -> Option<u128> {
                Some(self.coeff_linf_bound())
            }
            fn ring_dimension(&self) -> usize {
                self.ring_dimension()
            }
            fn new_unchecked(
                policy: SisSecurityPolicyId,
                table_digest: SisTableDigest,
                sis_modulus_profile: SisModulusProfileId,
                output_rank: usize,
                input_width: usize,
                coeff_linf_bound: u128,
                ring_dimension: usize,
            ) -> Self {
                Self::new_unchecked(
                    policy,
                    table_digest,
                    sis_modulus_profile,
                    output_rank,
                    input_width,
                    coeff_linf_bound,
                    ring_dimension,
                )
            }
        }
    };
}

impl_setup_prefix_commit_matrix_params!(OuterCommitMatrixParams, SisMatrixRole::Outer);

impl SetupPrefixCommitMatrixParams for InnerCommitMatrixParams {
    const ROLE: SisMatrixRole = SisMatrixRole::Inner;

    fn sis_modulus_profile(&self) -> SisModulusProfileId {
        self.sis_modulus_profile()
    }
    fn security_policy(&self) -> SisSecurityPolicyId {
        self.security_policy()
    }
    fn sis_table_key(&self) -> Option<crate::SisTableKey> {
        self.sis_table_key()
    }
    fn output_rank(&self) -> usize {
        self.output_rank()
    }
    fn input_width(&self) -> usize {
        self.input_width()
    }
    fn coeff_linf_bound(&self) -> Option<u128> {
        self.coeff_linf_bound()
    }
    fn ring_dimension(&self) -> usize {
        self.ring_dimension()
    }
    fn new_unchecked(
        policy: SisSecurityPolicyId,
        table_digest: SisTableDigest,
        sis_modulus_profile: SisModulusProfileId,
        output_rank: usize,
        input_width: usize,
        coeff_linf_bound: u128,
        ring_dimension: usize,
    ) -> Self {
        Self::new_unchecked(
            policy,
            table_digest,
            sis_modulus_profile,
            output_rank,
            input_width,
            coeff_linf_bound,
            ring_dimension,
        )
    }
}

/// Wire layout mirrors the commit-matrix descriptor bytes:
/// profile tag, policy tag, role tag, table digest, ring dim, row, col, linf.
pub(super) fn serialize_commit_matrix<K: SetupPrefixCommitMatrixParams, W: Write>(
    key: &K,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    let table_key = key.sis_table_key().ok_or_else(|| {
        SerializationError::InvalidData("setup prefix cannot use an L2 A security route".into())
    })?;
    serialize_sis_modulus_profile(key.sis_modulus_profile(), &mut writer)?;
    serialize_sis_security_policy(key.security_policy(), &mut writer)?;
    serialize_sis_matrix_role(table_key.role, &mut writer)?;
    serialize_sis_table_digest(table_key.table_digest, &mut writer)?;
    (table_key.ring_dimension as usize).serialize_with_mode(&mut writer, compress)?;
    key.output_rank()
        .serialize_with_mode(&mut writer, compress)?;
    key.input_width()
        .serialize_with_mode(&mut writer, compress)?;
    key.coeff_linf_bound()
        .ok_or_else(|| {
            SerializationError::InvalidData("setup prefix cannot use an L2 A security route".into())
        })?
        .serialize_with_mode(&mut writer, compress)?;
    Ok(())
}

pub(super) fn deserialize_commit_matrix<K: SetupPrefixCommitMatrixParams, R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<K, SerializationError> {
    let sis_modulus_profile = deserialize_sis_modulus_profile(&mut reader)?;
    let policy = deserialize_sis_security_policy(&mut reader)?;
    let role = deserialize_sis_matrix_role(&mut reader)?;
    if role != K::ROLE {
        return Err(SerializationError::InvalidData(
            "setup-prefix commitment matrix has the wrong role".to_string(),
        ));
    }
    let table_digest = deserialize_sis_table_digest(&mut reader)?;
    let ring_dimension = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let row_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let col_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
    let coeff_linf_bound = u128::deserialize_with_mode(&mut reader, compress, validate, &())?;
    Ok(K::new_unchecked(
        policy,
        table_digest,
        sis_modulus_profile,
        row_len,
        col_len,
        coeff_linf_bound,
        ring_dimension,
    ))
}

pub(super) fn commit_matrix_serialized_size<K: SetupPrefixCommitMatrixParams>(
    key: &K,
    compress: Compress,
) -> usize {
    1 + 1
        + 1
        + 32
        + key.ring_dimension().serialized_size(compress)
        + key.output_rank().serialized_size(compress)
        + key.input_width().serialized_size(compress)
        + 0u128.serialized_size(compress)
}
