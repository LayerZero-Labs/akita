use super::commitment::MatrixCommitmentKey;
use super::proof::{QuadraticMask, ZkSigmaProof};
use super::relation::{LinearExpression, LinearRelation, QuadraticRelation};
use super::statement::{ZkSigmaStatement, ZkSigmaWitness};
use crate::primitives::serialization::{
    Compress, HachiDeserialize, HachiSerialize, SerializationError, Valid, Validate,
};
use crate::FieldCore;
use std::io::{Read, Write};

impl<F: Valid + FieldCore> Valid for MatrixCommitmentKey<F> {
    fn check(&self) -> Result<(), SerializationError> {
        let expected = self
            .rows
            .checked_mul(self.cols)
            .ok_or_else(|| SerializationError::InvalidData("matrix shape overflow".to_string()))?;
        if self.entries.len() != expected {
            return Err(SerializationError::InvalidData(format!(
                "matrix entries length {}, expected {}",
                self.entries.len(),
                expected
            )));
        }
        self.entries.check()
    }
}

impl<F: FieldCore> HachiSerialize for MatrixCommitmentKey<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.rows.serialize_with_mode(&mut writer, compress)?;
        self.cols.serialize_with_mode(&mut writer, compress)?;
        self.entries.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.rows.serialized_size(compress)
            + self.cols.serialized_size(compress)
            + self.entries.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for MatrixCommitmentKey<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let rows = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let cols = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let entries = Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            rows,
            cols,
            entries,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Valid + FieldCore> Valid for LinearExpression<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs.check()?;
        self.constant.check()
    }
}

impl<F: FieldCore> HachiSerialize for LinearExpression<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.coeffs.serialize_with_mode(&mut writer, compress)?;
        self.constant.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs.serialized_size(compress) + self.constant.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for LinearExpression<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let coeffs = Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let constant = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self { coeffs, constant };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Valid + FieldCore> Valid for LinearRelation<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.expression.check()?;
        self.target.check()
    }
}

impl<F: FieldCore> HachiSerialize for LinearRelation<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.expression.serialize_with_mode(&mut writer, compress)?;
        self.target.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.expression.serialized_size(compress) + self.target.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for LinearRelation<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let expression =
            LinearExpression::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let target = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self { expression, target };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Valid + FieldCore> Valid for QuadraticRelation<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.left.check()?;
        self.right.check()?;
        self.output.check()?;
        self.target.check()
    }
}

impl<F: FieldCore> HachiSerialize for QuadraticRelation<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.left.serialize_with_mode(&mut writer, compress)?;
        self.right.serialize_with_mode(&mut writer, compress)?;
        self.output.serialize_with_mode(&mut writer, compress)?;
        self.target.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.left.serialized_size(compress)
            + self.right.serialized_size(compress)
            + self.output.serialized_size(compress)
            + self.target.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for QuadraticRelation<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let left = LinearExpression::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let right = LinearExpression::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let output = LinearExpression::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let target = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            left,
            right,
            output,
            target,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Valid + FieldCore> Valid for QuadraticMask<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.left.check()?;
        self.right.check()?;
        self.output.check()
    }
}

impl<F: FieldCore> HachiSerialize for QuadraticMask<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.left.serialize_with_mode(&mut writer, compress)?;
        self.right.serialize_with_mode(&mut writer, compress)?;
        self.output.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.left.serialized_size(compress)
            + self.right.serialized_size(compress)
            + self.output.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for QuadraticMask<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let left = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let right = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let output = F::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            left,
            right,
            output,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Valid + FieldCore> Valid for ZkSigmaStatement<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.commitment_key.check()?;
        self.commitment.check()?;
        self.linear_relations.check()?;
        self.quadratic_relations.check()
    }
}

impl<F: FieldCore> HachiSerialize for ZkSigmaStatement<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.commitment_key
            .serialize_with_mode(&mut writer, compress)?;
        self.commitment.serialize_with_mode(&mut writer, compress)?;
        self.linear_relations
            .serialize_with_mode(&mut writer, compress)?;
        self.quadratic_relations
            .serialize_with_mode(&mut writer, compress)?;
        self.response_linf_bound
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(bound) = self.response_linf_bound {
            bound.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.commitment_key.serialized_size(compress)
            + self.commitment.serialized_size(compress)
            + self.linear_relations.serialized_size(compress)
            + self.quadratic_relations.serialized_size(compress)
            + self.response_linf_bound.is_some().serialized_size(compress)
            + self
                .response_linf_bound
                .map_or(0, |bound| bound.serialized_size(compress))
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for ZkSigmaStatement<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let commitment_key =
            MatrixCommitmentKey::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let commitment = Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let linear_relations =
            Vec::<LinearRelation<F>>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let quadratic_relations = Vec::<QuadraticRelation<F>>::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?;
        let has_bound = bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let response_linf_bound = if has_bound {
            Some(u128::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?)
        } else {
            None
        };
        let out = Self {
            commitment_key,
            commitment,
            linear_relations,
            quadratic_relations,
            response_linf_bound,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Valid + FieldCore> Valid for ZkSigmaWitness<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.values.check()
    }
}

impl<F: FieldCore> HachiSerialize for ZkSigmaWitness<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.values.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.values.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for ZkSigmaWitness<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let values = Vec::<F>::deserialize_with_mode(reader, compress, validate, &())?;
        let out = Self { values };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: Valid + FieldCore> Valid for ZkSigmaProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.attempt.check()?;
        self.mask_commitment.check()?;
        self.linear_masks.check()?;
        self.quadratic_masks.check()?;
        self.response.check()
    }
}

impl<F: FieldCore> HachiSerialize for ZkSigmaProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.attempt.serialize_with_mode(&mut writer, compress)?;
        self.mask_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.linear_masks
            .serialize_with_mode(&mut writer, compress)?;
        self.quadratic_masks
            .serialize_with_mode(&mut writer, compress)?;
        self.response.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.attempt.serialized_size(compress)
            + self.mask_commitment.serialized_size(compress)
            + self.linear_masks.serialized_size(compress)
            + self.quadratic_masks.serialized_size(compress)
            + self.response.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for ZkSigmaProof<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let attempt = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let mask_commitment =
            Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let linear_masks = Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let quadratic_masks =
            Vec::<QuadraticMask<F>>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let response = Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            attempt,
            mask_commitment,
            linear_masks,
            quadratic_masks,
            response,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
