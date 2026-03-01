//! Serialization primitives for Hachi types

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::io::{Read, Write};

// Re-export derive macros
pub use hachi_derive::hachi_macro;

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
pub trait HachiSerialize {
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
pub trait HachiDeserialize {
    /// Deserialize with customization flags.
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError>
    where
        Self: Sized;

    /// Deserialize from compressed form with validation.
    fn deserialize_compressed<R: Read>(reader: R) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::Yes)
    }

    /// Deserialize from compressed form without validation.
    fn deserialize_compressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::No)
    }

    /// Deserialize from uncompressed form with validation.
    fn deserialize_uncompressed<R: Read>(reader: R) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        Self::deserialize_with_mode(reader, Compress::No, Validate::Yes)
    }

    /// Deserialize from uncompressed form without validation.
    fn deserialize_uncompressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError>
    where
        Self: Sized,
    {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
    }
}

mod primitive_impls {
    use super::*;

    macro_rules! impl_primitive_serialization {
        ($t:ty, $size:expr) => {
            impl Valid for $t {
                fn check(&self) -> Result<(), SerializationError> {
                    Ok(())
                }
            }

            impl HachiSerialize for $t {
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

            impl HachiDeserialize for $t {
                fn deserialize_with_mode<R: Read>(
                    mut reader: R,
                    _compress: Compress,
                    _validate: Validate,
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

    impl HachiSerialize for usize {
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

    impl HachiDeserialize for usize {
        fn deserialize_with_mode<R: Read>(
            reader: R,
            compress: Compress,
            validate: Validate,
        ) -> Result<Self, SerializationError> {
            let val = u64::deserialize_with_mode(reader, compress, validate)?;
            Ok(val as usize)
        }
    }

    impl Valid for bool {
        fn check(&self) -> Result<(), SerializationError> {
            Ok(())
        }
    }

    impl HachiSerialize for bool {
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

    impl HachiDeserialize for bool {
        fn deserialize_with_mode<R: Read>(
            mut reader: R,
            _compress: Compress,
            _validate: Validate,
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

    impl<T: HachiSerialize> HachiSerialize for Vec<T> {
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

    impl<T: HachiDeserialize> HachiDeserialize for Vec<T> {
        fn deserialize_with_mode<R: Read>(
            mut reader: R,
            compress: Compress,
            validate: Validate,
        ) -> Result<Self, SerializationError> {
            let len = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
            let mut vec = Vec::with_capacity(len);
            for _ in 0..len {
                vec.push(T::deserialize_with_mode(&mut reader, compress, validate)?);
            }
            Ok(vec)
        }
    }
}
