//! Exports the residual witness at a zero-knowledge opening's cut
//! (`specs/whiteout-akita.md` §4.8.1).

use crate::kernels::linear::decompose_commit_blocks_into;
use akita_error::AkitaError;
use akita_params::{dispatch_for_field, CommitmentSliceCount, CommittedGroupParams};
use akita_prover::backend::{ResidualExportKernel, ResidualWitness};
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use zeroize::Zeroizing;

impl<F, E> ResidualExportKernel<F, E> for crate::opaque::CpuBackend<F, E>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + 'static,
    E: ExtField<F> + 'static,
{
    fn export_residual(
        &self,
        witness: &Self::WitnessHandle,
        material: &Self::CommitmentMaterialHandle,
        params: &CommittedGroupParams,
    ) -> Result<ResidualWitness, AkitaError> {
        self.validate_binding(&witness.operation_binding())?;
        self.validate_binding(&material.binding())?;
        let profile = &params.own_group().profile;
        let ring_dim = profile.inner.matrix.ring_dimension();
        let n_a = profile.inner.matrix.output_rank();
        let live_blocks = profile.blocks.live_blocks;
        if profile.outer_slice_count != CommitmentSliceCount::ONE
            || profile.outer.matrix.ring_dimension() != ring_dim
        {
            return Err(AkitaError::InvalidInput(
                "a residual commitment has one slice and d_A = d_B".into(),
            ));
        }
        let [rows] = material.inner_rows() else {
            return Err(AkitaError::InvalidInput(
                "a residual commitment commits one source".into(),
            ));
        };
        let randomness = material.outer_randomness();
        if rows.ring_dim() != ring_dim
            || rows.count() != live_blocks * n_a
            || randomness.len() != profile.randomness_columns * ring_dim
        {
            return Err(AkitaError::InvalidInput(
                "residual material differs from the cut successor's parameters".into(),
            ));
        }
        let t_hat = dispatch_for_field!(
            akita_params::ProtocolDispatchSlot::Role(akita_params::RingRole::Inner),
            F,
            ring_dim,
            |D| {
                let rings = rows.as_ring_slice::<D>().map_err(|_| {
                    AkitaError::Internal("residual inner rows are not whole rings".into())
                })?;
                let blocks = rings.chunks_exact(n_a).collect::<Vec<_>>();
                let digits = decompose_commit_blocks_into::<F, D, D>(
                    &blocks,
                    profile.outer.digits.num_digits,
                    profile.outer.digits.log_basis,
                )?;
                Ok::<_, AkitaError>(Zeroizing::new(
                    digits.typed_planes::<D>()?.as_flattened().to_vec(),
                ))
            }
        )?;
        Ok(ResidualWitness {
            w_k: Zeroizing::new(witness.logical.digits().collect()),
            t_hat,
            r_a: Zeroizing::new(randomness.to_vec()),
        })
    }
}
