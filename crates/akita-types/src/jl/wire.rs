//! Headerless projection-reduction proof containers.

use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use akita_sumcheck::{uniform_sumcheck_shape, SumcheckProof, SumcheckProofShape};
use jolt_field::Field;
use std::io::{Read, Write};

use super::JlProjectionChainPlan;

/// Degree bound of the bilinear `X * (eq * J)` reduction summand.
pub const JL_PROJECTION_REDUCTION_DEGREE: usize = 2;

/// One reverse layer proof and its transcript-fixed input evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlLayerReductionProof<E: Field> {
    /// Degree-two consistency sumcheck.
    pub sumcheck: SumcheckProof<E>,
    /// Evaluation of this layer's input at the sumcheck terminal point.
    pub input_evaluation: E,
}

/// Complete proof for one chain. Layer proofs are serialized in reverse order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlProjectionProof<E: Field> {
    /// Present exactly when the public plan permits more than one attempt.
    pub retry_index: Option<u32>,
    /// Clear final image in canonical signed-wide coordinates.
    pub clear_image: Vec<i128>,
    /// Last layer first, source layer last.
    pub reverse_layers: Vec<JlLayerReductionProof<E>>,
}

/// Schedule-derived, non-wire shape for decoding a headerless chain proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlProjectionProofShape {
    retry_is_encoded: bool,
    clear_image_len: usize,
    reverse_sumchecks: Vec<SumcheckProofShape>,
}

impl JlProjectionProofShape {
    /// Derive the only accepted wire shape from a checked public plan.
    pub fn from_plan(plan: &JlProjectionChainPlan) -> Result<Self, akita_error::AkitaError> {
        let mut reverse_sumchecks = Vec::new();
        reverse_sumchecks
            .try_reserve_exact(plan.layers().len())
            .map_err(|_| {
                akita_error::AkitaError::InvalidInput("JL proof shape allocation failed".into())
            })?;
        for layer in plan.layers().iter().rev() {
            reverse_sumchecks.push(uniform_sumcheck_shape(
                layer.reduction_num_vars()?,
                JL_PROJECTION_REDUCTION_DEGREE,
            ));
        }
        Ok(Self {
            retry_is_encoded: plan.max_retries() > 1,
            clear_image_len: plan.final_image_len(),
            reverse_sumchecks,
        })
    }
}

impl<E: Field + Valid> Valid for JlLayerReductionProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.sumcheck.check()?;
        self.input_evaluation.check()
    }
}

impl<E: Field + Valid> Valid for JlProjectionProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.clear_image.check()?;
        self.reverse_layers.check()
    }
}

impl<E: Field + AkitaSerialize> AkitaSerialize for JlProjectionProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        if let Some(retry) = self.retry_index {
            retry.serialize_with_mode(&mut writer, compress)?;
        }
        for coordinate in &self.clear_image {
            coordinate.serialize_with_mode(&mut writer, compress)?;
        }
        for layer in &self.reverse_layers {
            layer.sumcheck.serialize_with_mode(&mut writer, compress)?;
            layer
                .input_evaluation
                .serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.retry_index
            .map_or(0, |retry| retry.serialized_size(compress))
            + self
                .clear_image
                .iter()
                .map(|coordinate| coordinate.serialized_size(compress))
                .sum::<usize>()
            + self
                .reverse_layers
                .iter()
                .map(|layer| {
                    layer.sumcheck.serialized_size(compress)
                        + layer.input_evaluation.serialized_size(compress)
                })
                .sum::<usize>()
    }
}

impl<E> AkitaDeserialize for JlProjectionProof<E>
where
    E: Field + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = JlProjectionProofShape;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        shape: &Self::Context,
    ) -> Result<Self, SerializationError> {
        let retry_index = shape
            .retry_is_encoded
            .then(|| u32::deserialize_with_mode(&mut reader, compress, validate, &()))
            .transpose()?;
        let mut clear_image = Vec::new();
        clear_image
            .try_reserve_exact(shape.clear_image_len)
            .map_err(|_| {
                SerializationError::InvalidData("JL clear-image allocation failed".into())
            })?;
        for _ in 0..shape.clear_image_len {
            clear_image.push(i128::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let mut reverse_layers = Vec::new();
        reverse_layers
            .try_reserve_exact(shape.reverse_sumchecks.len())
            .map_err(|_| SerializationError::InvalidData("JL layer allocation failed".into()))?;
        for sumcheck_shape in &shape.reverse_sumchecks {
            reverse_layers.push(JlLayerReductionProof {
                sumcheck: SumcheckProof::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    sumcheck_shape,
                )?,
                input_evaluation: E::deserialize_with_mode(&mut reader, compress, validate, &())?,
            });
        }
        let proof = Self {
            retry_index,
            clear_image,
            reverse_layers,
        };
        if matches!(validate, Validate::Yes) {
            proof.check()?;
        }
        Ok(proof)
    }
}

/// Deferred source evaluation left after a complete reverse chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlSourceClaim<E: Field> {
    /// Source tensor point, local-column variables followed by block variables.
    pub point: Vec<E>,
    /// Claimed source MLE evaluation at `point`.
    pub evaluation: E,
}
