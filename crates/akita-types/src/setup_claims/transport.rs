//! Canonical, bounded transport for a statement authenticated by a recursive guest's public output.
use super::{query::NativeTerm, SetupBatchStatement, SetupClaim, SetupPolynomialId, SetupQuery};
use akita_algebra::offset_eq::{EqPairTensorAxis, EqPairTensorFamily, EqPairTensorWeights};
use akita_error::AkitaError;
use akita_serialization::AkitaDeserialize;
use jolt_field::{CanonicalEncoding, ExtField, Field};

fn invalid() -> AkitaError {
    AkitaError::InvalidInput("invalid canonical setup statement".into())
}
struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn integer(&mut self) -> Result<usize, AkitaError> {
        usize::deserialize_compressed(&mut self.0, &()).map_err(|_| invalid())
    }
    fn field<F: Field + CanonicalEncoding>(&mut self) -> Result<F, AkitaError> {
        let bytes = self.0.get(..F::NUM_BYTES).ok_or_else(invalid)?;
        let value = F::from_bytes_le_checked(bytes).ok_or_else(invalid)?;
        self.0 = &self.0[F::NUM_BYTES..];
        Ok(value)
    }
    fn fields<F: Field + CanonicalEncoding>(&mut self) -> Result<Vec<F>, AkitaError> {
        let n = self.integer()?;
        if n > self.0.len() / F::NUM_BYTES {
            return Err(invalid());
        }
        (0..n).map(|_| self.field()).collect()
    }
    fn query<F: Field + CanonicalEncoding>(
        &mut self,
        fields: usize,
    ) -> Result<SetupQuery<F>, AkitaError> {
        match self.integer()? {
            0 => SetupQuery::evaluation(self.fields()?),
            1 => {
                let n = self.integer()?;
                if n == 0 || n > 4096 || n > self.0.len() / 8 {
                    return Err(invalid());
                }
                let mut terms = Vec::with_capacity(n);
                for _ in 0..n {
                    let coefficient: Vec<F> = self.fields()?;
                    let d = coefficient.len();
                    if !(2..=1024).contains(&d) || !d.is_power_of_two() || d > fields {
                        return Err(invalid());
                    }
                    let point: Vec<F> = self.fields()?;
                    let right_limit = 1usize
                        .checked_shl(point.len().try_into().map_err(|_| invalid())?)
                        .ok_or_else(invalid)?;
                    let count = self.integer()?;
                    if count > 4096 || count > self.0.len() / 8 {
                        return Err(invalid());
                    }
                    let mut tensors = Vec::with_capacity(count);
                    for _ in 0..count {
                        let left_offset = self.integer()?;
                        let right_offset = self.integer()?;
                        let scalar: Vec<F> = self.fields()?;
                        if scalar.len() != 1 {
                            return Err(invalid());
                        }
                        let count = self.integer()?;
                        if count > 8 {
                            return Err(invalid());
                        }
                        let mut axes = Vec::with_capacity(count);
                        let (mut left, mut right) = (left_offset, right_offset);
                        for _ in 0..count {
                            let len = self.integer()?;
                            let left_stride = self.integer()?;
                            let right_stride = self.integer()?;
                            if len == 0 {
                                return Err(invalid());
                            }
                            left = left
                                .checked_add(
                                    (len - 1).checked_mul(left_stride).ok_or_else(invalid)?,
                                )
                                .ok_or_else(invalid)?;
                            right = right
                                .checked_add(
                                    (len - 1).checked_mul(right_stride).ok_or_else(invalid)?,
                                )
                                .ok_or_else(invalid)?;
                            let weights = match self.integer()? {
                                0 => EqPairTensorWeights::Unit,
                                1 => {
                                    let v: Vec<F> = self.fields()?;
                                    if v.len() != len {
                                        return Err(invalid());
                                    }
                                    EqPairTensorWeights::Dense(v.into())
                                }
                                2 => {
                                    let bits = self.integer()?;
                                    if bits >= usize::BITS as usize || (1usize << bits) != len {
                                        return Err(invalid());
                                    }
                                    let mut v = Vec::with_capacity(bits);
                                    for _ in 0..bits {
                                        v.push(
                                            self.fields::<F>()?
                                                .try_into()
                                                .map_err(|_| invalid())?,
                                        );
                                    }
                                    EqPairTensorWeights::BitProduct(v.into())
                                }
                                _ => return Err(invalid()),
                            };
                            axes.push(EqPairTensorAxis {
                                len,
                                left_stride,
                                right_stride,
                                weights,
                            });
                        }
                        if left >= fields / d || right >= right_limit {
                            return Err(invalid());
                        }
                        tensors.push(EqPairTensorFamily {
                            left_offset,
                            right_offset,
                            scalar: scalar[0],
                            axes,
                        });
                    }
                    terms.push(NativeTerm {
                        coefficient,
                        point,
                        tensors,
                    });
                }
                Ok(SetupQuery::from_native_terms(terms))
            }
            _ => Err(invalid()),
        }
    }
}
impl<F: Field + CanonicalEncoding + ExtField<F>> SetupBatchStatement<F> {
    /// Decode the exact canonical descriptor, bound to the caller's expected public setup identity.
    /// Acceptance still requires binding these bytes to the verified program's public output.
    pub fn from_descriptor_bytes(
        bytes: &[u8],
        expected: &SetupPolynomialId,
    ) -> Result<Self, AkitaError> {
        let mut identity = Vec::new();
        expected.encode(&mut identity)?;
        let rest = bytes
            .strip_prefix(identity.as_slice())
            .ok_or_else(invalid)?;
        let mut input = Reader(rest);
        let coefficient_len = input.integer()?;
        let n = input.integer()?;
        if n > 65536 || n > input.0.len() / 8 {
            return Err(invalid());
        }
        let mut claims = Vec::with_capacity(n);
        for _ in 0..n {
            let query = input.query(expected.field_len())?;
            let value = input.field()?;
            claims.push(SetupClaim::new(expected.clone(), query, value)?);
        }
        if !input.0.is_empty() {
            return Err(invalid());
        }
        let statement = Self::new(expected.clone(), coefficient_len, claims)?;
        if statement.descriptor_bytes::<F>()? != bytes {
            return Err(invalid());
        }
        Ok(statement)
    }
}

#[cfg(test)]
mod tests;
