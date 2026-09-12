//! Headerless projection-reduction proof containers.

use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
    DEFAULT_MAX_SEQUENCE_LEN,
};
use akita_sumcheck::{uniform_sumcheck_shape, SumcheckProof, SumcheckProofShape};
use jolt_field::Field;
use std::io::{Read, Write};

use super::{JlAlignedEtProjectionPlan, JlProjectionBatchPlan, JlProjectionChainPlan};

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
    /// Clear final image in canonical signed-wide coordinates.
    pub clear_image: Vec<i128>,
    /// Last layer first, source layer last.
    pub reverse_layers: Vec<JlLayerReductionProof<E>>,
}

/// Schedule-derived, non-wire shape for decoding a headerless chain proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlProjectionProofShape {
    clear_image_len: usize,
    reverse_sumchecks: Vec<SumcheckProofShape>,
}

impl JlProjectionProofShape {
    /// Derive the only accepted wire shape from a checked public plan.
    pub fn from_plan(plan: &JlProjectionChainPlan) -> Result<Self, akita_error::AkitaError> {
        validate_clear_image_len(plan.final_image_len())?;
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
            clear_image_len: plan.final_image_len(),
            reverse_sumchecks,
        })
    }
}

/// Complete headerless proof for one level projection forest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlProjectionBatchProof<E: Field> {
    /// Present exactly when the public plan permits multiple whole-forest candidates.
    pub retry_index: Option<u32>,
    /// Certificate/stem chains in the public plan's canonical order.
    pub chains: Vec<JlProjectionProof<E>>,
}

/// Schedule-derived decoder shape for one level projection forest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlProjectionBatchProofShape {
    retry_is_encoded: bool,
    chains: Vec<JlProjectionProofShape>,
}

/// Headerless proof for `Z` and an aligned private-E/private-T selector join.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlAlignedEtProjectionProof<E: Field> {
    /// Present exactly when multiple whole-forest candidates are scheduled.
    pub retry_index: Option<u32>,
    /// Aggregate semantic-Z clear image and reverse reduction.
    pub z: JlProjectionProof<E>,
    /// Final E/T-tail clear image and reverse reduction to the joined table.
    pub et_tail: JlProjectionProof<E>,
    /// Private E-stem image evaluation at the tail's terminal inner point.
    pub e_stem_image_evaluation: E,
    /// Private T-stem image evaluation at the same inner point.
    pub t_stem_image_evaluation: E,
    /// Private E-stem reverse reduction.
    pub e_stem_reverse_layers: Vec<JlLayerReductionProof<E>>,
    /// Private T-stem reverse reduction.
    pub t_stem_reverse_layers: Vec<JlLayerReductionProof<E>>,
}

/// Schedule-derived decoder shape for an aligned E/T selector-join proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlAlignedEtProjectionProofShape {
    retry_is_encoded: bool,
    z: JlProjectionProofShape,
    e_stem: JlProjectionProofShape,
    t_stem: JlProjectionProofShape,
    et_tail: JlProjectionProofShape,
}

impl JlAlignedEtProjectionProofShape {
    /// Derive the exact headerless shape from the aligned graph plan.
    pub fn from_plan(plan: &JlAlignedEtProjectionPlan) -> Result<Self, akita_error::AkitaError> {
        Ok(Self {
            retry_is_encoded: plan.batch().retry_is_encoded(),
            z: JlProjectionProofShape::from_plan(plan.z()?)?,
            e_stem: JlProjectionProofShape::from_plan(plan.e_stem()?)?,
            t_stem: JlProjectionProofShape::from_plan(plan.t_stem()?)?,
            et_tail: JlProjectionProofShape::from_plan(plan.et_tail()?)?,
        })
    }
}

impl JlProjectionBatchProofShape {
    /// Derive the only accepted wire shape from a checked public batch plan.
    pub fn from_plan(plan: &JlProjectionBatchPlan) -> Result<Self, akita_error::AkitaError> {
        let mut chains = Vec::new();
        chains.try_reserve_exact(plan.chains().len()).map_err(|_| {
            akita_error::AkitaError::InvalidInput("JL batch proof-shape allocation failed".into())
        })?;
        for chain in plan.chains() {
            chains.push(JlProjectionProofShape::from_plan(chain)?);
        }
        Ok(Self {
            retry_is_encoded: plan.retry_is_encoded(),
            chains,
        })
    }
}

fn validate_clear_image_len(len: usize) -> Result<(), akita_error::AkitaError> {
    if len > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(akita_error::AkitaError::InvalidInput(format!(
            "JL proof image length {len} exceeds proof-sequence bound {DEFAULT_MAX_SEQUENCE_LEN}"
        )));
    }
    Ok(())
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

impl<E: Field + Valid> Valid for JlProjectionBatchProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.chains.check()
    }
}

impl<E: Field + Valid> Valid for JlAlignedEtProjectionProof<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.z.check()?;
        self.et_tail.check()?;
        self.e_stem_image_evaluation.check()?;
        self.t_stem_image_evaluation.check()?;
        self.e_stem_reverse_layers.check()?;
        self.t_stem_reverse_layers.check()
    }
}

impl<E: Field + AkitaSerialize> AkitaSerialize for JlProjectionProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
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
        self.clear_image
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
            clear_image,
            reverse_layers,
        };
        if matches!(validate, Validate::Yes) {
            proof.check()?;
        }
        Ok(proof)
    }
}

impl<E: Field + AkitaSerialize> AkitaSerialize for JlProjectionBatchProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        if let Some(retry) = self.retry_index {
            retry.serialize_with_mode(&mut writer, compress)?;
        }
        for chain in &self.chains {
            for coordinate in &chain.clear_image {
                coordinate.serialize_with_mode(&mut writer, compress)?;
            }
        }
        for chain in &self.chains {
            for layer in &chain.reverse_layers {
                layer.sumcheck.serialize_with_mode(&mut writer, compress)?;
                layer
                    .input_evaluation
                    .serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.retry_index
            .map_or(0, |retry| retry.serialized_size(compress))
            + self
                .chains
                .iter()
                .map(|chain| chain.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<E> AkitaDeserialize for JlProjectionBatchProof<E>
where
    E: Field + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = JlProjectionBatchProofShape;

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
        let mut chains = Vec::new();
        chains.try_reserve_exact(shape.chains.len()).map_err(|_| {
            SerializationError::InvalidData("JL batch proof allocation failed".into())
        })?;
        for chain_shape in &shape.chains {
            let mut clear_image = Vec::new();
            clear_image
                .try_reserve_exact(chain_shape.clear_image_len)
                .map_err(|_| {
                    SerializationError::InvalidData("JL clear-image allocation failed".into())
                })?;
            for _ in 0..chain_shape.clear_image_len {
                clear_image.push(i128::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            chains.push(JlProjectionProof {
                clear_image,
                reverse_layers: Vec::new(),
            });
        }
        for (chain, chain_shape) in chains.iter_mut().zip(&shape.chains) {
            chain
                .reverse_layers
                .try_reserve_exact(chain_shape.reverse_sumchecks.len())
                .map_err(|_| {
                    SerializationError::InvalidData("JL layer allocation failed".into())
                })?;
            for sumcheck_shape in &chain_shape.reverse_sumchecks {
                chain.reverse_layers.push(JlLayerReductionProof {
                    sumcheck: SumcheckProof::deserialize_with_mode(
                        &mut reader,
                        compress,
                        validate,
                        sumcheck_shape,
                    )?,
                    input_evaluation: E::deserialize_with_mode(
                        &mut reader,
                        compress,
                        validate,
                        &(),
                    )?,
                });
            }
        }
        let proof = Self {
            retry_index,
            chains,
        };
        if matches!(validate, Validate::Yes) {
            proof.check()?;
        }
        Ok(proof)
    }
}

impl<E: Field + AkitaSerialize> AkitaSerialize for JlAlignedEtProjectionProof<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        if let Some(retry) = self.retry_index {
            retry.serialize_with_mode(&mut writer, compress)?;
        }
        serialize_image(&self.z.clear_image, &mut writer, compress)?;
        serialize_image(&self.et_tail.clear_image, &mut writer, compress)?;
        serialize_layers(&self.z.reverse_layers, &mut writer, compress)?;
        serialize_layers(&self.et_tail.reverse_layers, &mut writer, compress)?;
        self.e_stem_image_evaluation
            .serialize_with_mode(&mut writer, compress)?;
        self.t_stem_image_evaluation
            .serialize_with_mode(&mut writer, compress)?;
        serialize_layers(&self.e_stem_reverse_layers, &mut writer, compress)?;
        serialize_layers(&self.t_stem_reverse_layers, &mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.retry_index
            .map_or(0, |retry| retry.serialized_size(compress))
            + serialized_image_size(&self.z.clear_image, compress)
            + serialized_image_size(&self.et_tail.clear_image, compress)
            + serialized_layers_size(&self.z.reverse_layers, compress)
            + serialized_layers_size(&self.et_tail.reverse_layers, compress)
            + self.e_stem_image_evaluation.serialized_size(compress)
            + self.t_stem_image_evaluation.serialized_size(compress)
            + serialized_layers_size(&self.e_stem_reverse_layers, compress)
            + serialized_layers_size(&self.t_stem_reverse_layers, compress)
    }
}

impl<E> AkitaDeserialize for JlAlignedEtProjectionProof<E>
where
    E: Field + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = JlAlignedEtProjectionProofShape;

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
        let z_image = deserialize_image(&mut reader, compress, validate, shape.z.clear_image_len)?;
        let et_image = deserialize_image(
            &mut reader,
            compress,
            validate,
            shape.et_tail.clear_image_len,
        )?;
        let z_layers =
            deserialize_layers(&mut reader, compress, validate, &shape.z.reverse_sumchecks)?;
        let et_layers = deserialize_layers(
            &mut reader,
            compress,
            validate,
            &shape.et_tail.reverse_sumchecks,
        )?;
        let e_stem_image_evaluation =
            E::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let t_stem_image_evaluation =
            E::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let e_stem_reverse_layers = deserialize_layers(
            &mut reader,
            compress,
            validate,
            &shape.e_stem.reverse_sumchecks,
        )?;
        let t_stem_reverse_layers = deserialize_layers(
            &mut reader,
            compress,
            validate,
            &shape.t_stem.reverse_sumchecks,
        )?;
        let proof = Self {
            retry_index,
            z: JlProjectionProof {
                clear_image: z_image,
                reverse_layers: z_layers,
            },
            et_tail: JlProjectionProof {
                clear_image: et_image,
                reverse_layers: et_layers,
            },
            e_stem_image_evaluation,
            t_stem_image_evaluation,
            e_stem_reverse_layers,
            t_stem_reverse_layers,
        };
        if matches!(validate, Validate::Yes) {
            proof.check()?;
        }
        Ok(proof)
    }
}

fn serialize_image<W: Write>(
    image: &[i128],
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    for coordinate in image {
        coordinate.serialize_with_mode(&mut writer, compress)?;
    }
    Ok(())
}

fn serialize_layers<E: Field + AkitaSerialize, W: Write>(
    layers: &[JlLayerReductionProof<E>],
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    for layer in layers {
        layer.sumcheck.serialize_with_mode(&mut writer, compress)?;
        layer
            .input_evaluation
            .serialize_with_mode(&mut writer, compress)?;
    }
    Ok(())
}

fn serialized_image_size(image: &[i128], compress: Compress) -> usize {
    image
        .iter()
        .map(|coordinate| coordinate.serialized_size(compress))
        .sum()
}

fn serialized_layers_size<E: Field + AkitaSerialize>(
    layers: &[JlLayerReductionProof<E>],
    compress: Compress,
) -> usize {
    layers
        .iter()
        .map(|layer| {
            layer.sumcheck.serialized_size(compress)
                + layer.input_evaluation.serialized_size(compress)
        })
        .sum()
}

fn deserialize_image<R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    len: usize,
) -> Result<Vec<i128>, SerializationError> {
    let mut image = Vec::new();
    image
        .try_reserve_exact(len)
        .map_err(|_| SerializationError::InvalidData("JL clear-image allocation failed".into()))?;
    for _ in 0..len {
        image.push(i128::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?);
    }
    Ok(image)
}

fn deserialize_layers<E, R>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    shapes: &[SumcheckProofShape],
) -> Result<Vec<JlLayerReductionProof<E>>, SerializationError>
where
    E: Field + Valid + AkitaDeserialize<Context = ()>,
    R: Read,
{
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(shapes.len())
        .map_err(|_| SerializationError::InvalidData("JL layer allocation failed".into()))?;
    for shape in shapes {
        layers.push(JlLayerReductionProof {
            sumcheck: SumcheckProof::deserialize_with_mode(&mut reader, compress, validate, shape)?,
            input_evaluation: E::deserialize_with_mode(&mut reader, compress, validate, &())?,
        });
    }
    Ok(layers)
}

/// Deferred source evaluation left after a complete reverse chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JlSourceClaim<E: Field> {
    /// Source tensor point, local-column variables followed by block variables.
    pub point: Vec<E>,
    /// Claimed source MLE evaluation at `point`.
    pub evaluation: E,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_shape_defensively_rejects_oversized_clear_image() {
        assert!(validate_clear_image_len(DEFAULT_MAX_SEQUENCE_LEN + 1).is_err());
    }
}
