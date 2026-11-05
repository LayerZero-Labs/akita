//! CPU compilation of mixed native root rows into the ordinary Stage 2 domain.

use akita_error::AkitaError;
use akita_prover::backend::{NativeRootRelationRequest, RelationCompiler};
use akita_types::native_root_relation::NativeRootWeights;
use jolt_algebra::Field;

use crate::CpuBackend;

/// Dense public coefficients and the scalar claim consumed by Stage 2.
/// Coefficient addresses are exactly those of the ordinary successor witness.
pub struct CompiledNativeRoot<E: Field> {
    pub weights: NativeRootWeights<E>,
    pub evaluations: Vec<E>,
}

impl<F: Field, E: Field> RelationCompiler<NativeRootRelationRequest<'_, E>> for CpuBackend<F, E> {
    type Output = CompiledNativeRoot<E>;

    fn compile_relation(
        &self,
        request: NativeRootRelationRequest<'_, E>,
    ) -> Result<Self::Output, AkitaError> {
        let weights = request.relation.ring_switch(
            request.alpha,
            request.row_weights,
            request.padding_weights,
        )?;
        let evaluations = weights.materialize()?;
        Ok(CompiledNativeRoot {
            weights,
            evaluations,
        })
    }
}

impl<E: Field + jolt_algebra::Unreduced + jolt_algebra::Fold + 'static> CompiledNativeRoot<E> {
    /// Consume these public weights in the ordinary fused relation/range-image
    /// sumcheck. Stage 1 authenticates `range_image_evaluation`; the successor
    /// opening authenticates the returned session's final witness evaluation.
    pub fn begin_stage2(
        self,
        witness: Vec<i8>,
        stage1_point: &[E],
        range_image_evaluation: E,
        batching: E,
    ) -> Result<crate::CpuStage2SessionHandle<E>, AkitaError> {
        crate::CpuStage2SessionHandle::from_dense_relation(
            witness,
            self.evaluations,
            self.weights.claim(),
            stage1_point,
            range_image_evaluation,
            batching,
        )
    }
}
