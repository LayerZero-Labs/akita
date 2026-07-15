//! Serialization primitives for Akita types

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::io::{Read, Write};

/// Default maximum number of elements accepted by self-described validated
/// vector decoding.
///
/// Protocol shapes should normally provide tighter bounds before vector
/// allocation. This cap protects generic verifier-facing decoders from
/// allocating directly from attacker-controlled lengths. Recursive setup-prefix
/// sidecars currently store prover hint digit streams that can exceed `2^24`
/// entries, so keep the temporary global cap high enough for that local cache.
pub const DEFAULT_MAX_SEQUENCE_LEN: usize = 1 << 25;

/// Compression mode for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compress {
    /// Enable compression
    Yes,
    /// Disable compression
    No,
}

/// Validation mode for deserialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Validate {
    /// Enable validation
    Yes,
    /// Disable validation
    No,
}

/// Serialization error types
#[derive(Debug, thiserror::Error)]
pub enum SerializationError {
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid data
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Unexpected data
    #[error("Unexpected data")]
    UnexpectedData,

    /// Encoded sequence length exceeds the configured decoder limit.
    #[error("Sequence length {len} exceeds maximum {max}")]
    LengthLimitExceeded {
        /// Encoded length.
        len: u64,
        /// Maximum accepted length.
        max: usize,
    },
}

/// Trait for validating deserialized data.
/// This is checked after deserialization when `Validate::Yes` is used.
pub trait Valid {
    /// Check that the current value is valid
    fn check(&self) -> Result<(), SerializationError>;

    /// Batch check for efficiency when validating multiple elements.
    fn batch_check<'a>(batch: impl Iterator<Item = &'a Self>) -> Result<(), SerializationError>
    where
        Self: 'a,
    {
        for item in batch {
            item.check()?;
        }
        Ok(())
    }
}

/// Serializer in little endian format.
pub trait AkitaSerialize {
    /// Serialize with customization flags.
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError>;

    /// Returns the serialized size in bytes for the given compression mode.
    fn serialized_size(&self, compress: Compress) -> usize;

    /// Serialize in compressed form.
    fn serialize_compressed<W: Write>(&self, writer: W) -> Result<(), SerializationError> {
        self.serialize_with_mode(writer, Compress::Yes)
    }

    /// Returns the compressed size in bytes.
    fn compressed_size(&self) -> usize {
        self.serialized_size(Compress::Yes)
    }

    /// Serialize in uncompressed form.
    fn serialize_uncompressed<W: Write>(&self, writer: W) -> Result<(), SerializationError> {
        self.serialize_with_mode(writer, Compress::No)
    }

    /// Returns the uncompressed size in bytes.
    fn uncompressed_size(&self) -> usize {
        self.serialized_size(Compress::No)
    }
}

/// Deserializer in little endian format.
///
/// The [`Context`](Self::Context) associated type carries any shape information
/// that the deserializer cannot recover from the byte stream itself. Fixed-size
/// types (primitives, field elements, rings) use `Context = ()`. Proof types
/// whose headers have been stripped use a schedule-derived shape descriptor.
pub trait AkitaDeserialize: Sized {
    /// External shape context needed for deserialization. `()` for
    /// self-describing types.
    type Context;

    /// Deserialize with customization flags and external context.
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &Self::Context,
    ) -> Result<Self, SerializationError>;

    /// Deserialize from compressed form with validation.
    fn deserialize_compressed<R: Read>(
        reader: R,
        ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::Yes, ctx)
    }

    /// Deserialize from compressed form without validation.
    ///
    /// This is for trusted internal buffers whose producer and shape have
    /// already been checked in the same trust domain. Use
    /// [`Self::deserialize_compressed`] for verifier-facing bytes.
    fn deserialize_compressed_unchecked<R: Read>(
        reader: R,
        ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::No, ctx)
    }

    /// Deserialize from uncompressed form with validation.
    fn deserialize_uncompressed<R: Read>(
        reader: R,
        ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::Yes, ctx)
    }

    /// Deserialize from uncompressed form without validation.
    ///
    /// This is for trusted internal buffers whose producer and shape have
    /// already been checked in the same trust domain. Use
    /// [`Self::deserialize_uncompressed`] for verifier-facing bytes.
    fn deserialize_uncompressed_unchecked<R: Read>(
        reader: R,
        ctx: &Self::Context,
    ) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No, ctx)
    }
}

mod primitive_impls {
    use super::*;

    fn checked_vec_len(len: u64, validate: Validate) -> Result<usize, SerializationError> {
        let len_usize =
            usize::try_from(len).map_err(|_| SerializationError::LengthLimitExceeded {
                len,
                max: usize::MAX,
            })?;

        if validate == Validate::Yes && len_usize > DEFAULT_MAX_SEQUENCE_LEN {
            return Err(SerializationError::LengthLimitExceeded {
                len,
                max: DEFAULT_MAX_SEQUENCE_LEN,
            });
        }

        Ok(len_usize)
    }

    macro_rules! impl_primitive_serialization {
        ($t:ty, $size:expr) => {
            impl Valid for $t {
                fn check(&self) -> Result<(), SerializationError> {
                    Ok(())
                }
            }

            impl AkitaSerialize for $t {
                fn serialize_with_mode<W: Write>(
                    &self,
                    mut writer: W,
                    _compress: Compress,
                ) -> Result<(), SerializationError> {
                    writer.write_all(&self.to_le_bytes())?;
                    Ok(())
                }

                fn serialized_size(&self, _compress: Compress) -> usize {
                    $size
                }
            }

            impl AkitaDeserialize for $t {
                type Context = ();
                fn deserialize_with_mode<R: Read>(
                    mut reader: R,
                    _compress: Compress,
                    _validate: Validate,
                    _ctx: &(),
                ) -> Result<Self, SerializationError> {
                    let mut bytes = [0u8; $size];
                    reader.read_exact(&mut bytes)?;
                    Ok(<$t>::from_le_bytes(bytes))
                }
            }
        };
    }

    impl_primitive_serialization!(u8, 1);
    impl_primitive_serialization!(u16, 2);
    impl_primitive_serialization!(u32, 4);
    impl_primitive_serialization!(u64, 8);
    impl_primitive_serialization!(u128, 16);
    impl_primitive_serialization!(i8, 1);
    impl_primitive_serialization!(i16, 2);
    impl_primitive_serialization!(i32, 4);
    impl_primitive_serialization!(i64, 8);
    impl_primitive_serialization!(i128, 16);

    impl Valid for usize {
        fn check(&self) -> Result<(), SerializationError> {
            Ok(())
        }
    }

    impl AkitaSerialize for usize {
        fn serialize_with_mode<W: Write>(
            &self,
            writer: W,
            compress: Compress,
        ) -> Result<(), SerializationError> {
            (*self as u64).serialize_with_mode(writer, compress)
        }

        fn serialized_size(&self, _compress: Compress) -> usize {
            8
        }
    }

    impl AkitaDeserialize for usize {
        type Context = ();
        fn deserialize_with_mode<R: Read>(
            reader: R,
            compress: Compress,
            validate: Validate,
            _ctx: &(),
        ) -> Result<Self, SerializationError> {
            let val = u64::deserialize_with_mode(reader, compress, validate, &())?;
            usize::try_from(val).map_err(|_| SerializationError::LengthLimitExceeded {
                len: val,
                max: usize::MAX,
            })
        }
    }

    impl Valid for bool {
        fn check(&self) -> Result<(), SerializationError> {
            Ok(())
        }
    }

    impl AkitaSerialize for bool {
        fn serialize_with_mode<W: Write>(
            &self,
            mut writer: W,
            _compress: Compress,
        ) -> Result<(), SerializationError> {
            writer.write_all(&[*self as u8])?;
            Ok(())
        }

        fn serialized_size(&self, _compress: Compress) -> usize {
            1
        }
    }

    impl AkitaDeserialize for bool {
        type Context = ();
        fn deserialize_with_mode<R: Read>(
            mut reader: R,
            _compress: Compress,
            _validate: Validate,
            _ctx: &(),
        ) -> Result<Self, SerializationError> {
            let mut byte = [0u8; 1];
            reader.read_exact(&mut byte)?;
            match byte[0] {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(SerializationError::InvalidData(
                    "Invalid bool value".to_string(),
                )),
            }
        }
    }

    impl<T: Valid> Valid for Vec<T> {
        fn check(&self) -> Result<(), SerializationError> {
            for item in self {
                item.check()?;
            }
            Ok(())
        }
    }

    impl<T: AkitaSerialize> AkitaSerialize for Vec<T> {
        fn serialize_with_mode<W: Write>(
            &self,
            mut writer: W,
            compress: Compress,
        ) -> Result<(), SerializationError> {
            (self.len() as u64).serialize_with_mode(&mut writer, compress)?;
            for item in self {
                item.serialize_with_mode(&mut writer, compress)?;
            }
            Ok(())
        }

        fn serialized_size(&self, compress: Compress) -> usize {
            let len_size = 8;
            let items_size: usize = self.iter().map(|item| item.serialized_size(compress)).sum();
            len_size + items_size
        }
    }

    impl<T: AkitaDeserialize<Context = ()>> AkitaDeserialize for Vec<T> {
        type Context = ();
        fn deserialize_with_mode<R: Read>(
            mut reader: R,
            compress: Compress,
            validate: Validate,
            _ctx: &(),
        ) -> Result<Self, SerializationError> {
            let encoded_len = u64::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let len = checked_vec_len(encoded_len, validate)?;
            let mut vec = Vec::with_capacity(len);
            for _ in 0..len {
                vec.push(T::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            Ok(vec)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn validated_vec_rejects_default_limit_exhaustion() {
        let mut bytes = Vec::new();
        ((DEFAULT_MAX_SEQUENCE_LEN as u64) + 1)
            .serialize_compressed(&mut bytes)
            .unwrap();

        let err = Vec::<u8>::deserialize_compressed(&bytes[..], &()).unwrap_err();
        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { .. }
        ));
    }

    #[test]
    fn unchecked_vec_is_reserved_for_trusted_internal_buffers() {
        let mut bytes = Vec::new();
        3u64.serialize_compressed(&mut bytes).unwrap();
        bytes.extend_from_slice(&[1, 2, 3]);

        let decoded = Vec::<u8>::deserialize_compressed_unchecked(&bytes[..], &()).unwrap();
        assert_eq!(decoded, vec![1, 2, 3]);
    }

    proptest! {
        #[test]
        fn vec_u8_round_trips(values in proptest::collection::vec(any::<u8>(), 0..1024)) {
            let mut encoded = Vec::new();
            prop_assert!(values.serialize_compressed(&mut encoded).is_ok());

            match Vec::<u8>::deserialize_compressed(&encoded[..], &()) {
                Ok(decoded) => prop_assert_eq!(decoded, values),
                Err(err) => prop_assert!(false, "round trip failed: {err}"),
            }
        }

        #[test]
        fn bool_rejects_non_canonical_bytes(byte in 2u8..) {
            let err = bool::deserialize_compressed(&[byte][..], &()).unwrap_err();
            prop_assert!(matches!(err, SerializationError::InvalidData(_)));
        }
    }
}
