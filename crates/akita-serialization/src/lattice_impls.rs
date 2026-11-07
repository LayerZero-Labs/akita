//! Akita's wire format for Jolt's public-matrix seed: the one-byte
//! derivation tag, then the 32 seed bytes (`PublicMatrixSeed`'s canonical
//! bytes). Setup digests hash exactly these bytes.

use std::io::{Read, Write};

use jolt_crypto::lattice::PublicMatrixSeed;

use crate::{AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate};

impl Valid for PublicMatrixSeed {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl AkitaSerialize for PublicMatrixSeed {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        writer.write_all(&self.to_canonical_bytes())?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        Self::CANONICAL_LEN
    }
}

impl AkitaDeserialize for PublicMatrixSeed {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let mut bytes = [0u8; Self::CANONICAL_LEN];
        reader.read_exact(&mut bytes)?;
        Self::from_canonical_bytes(&bytes)
            .map_err(|error| SerializationError::InvalidData(error.to_string()))
    }
}
