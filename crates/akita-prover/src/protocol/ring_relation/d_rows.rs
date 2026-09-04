//! D-role relation rows for ring-relation witness construction.

use crate::backend::RingSwitchRelationView;
use crate::compute::{
    OperationCtx, RingSwitchProveBackend, RingSwitchRelationKernel, RingSwitchRelationPlan,
};
use crate::DigitRowsComputeBackend;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{DigitBlocks, RingRelationMode};
use jolt_field::{CanonicalEncoding, Field};

use super::relation_quotient::quotient_from_cyclic_and_reduced;

pub(super) enum RelationDRows<F: Field, const D: usize> {
    QuotientLift {
        reduced: Vec<CyclotomicRing<F, D>>,
        quotients: Vec<CyclotomicRing<F, D>>,
    },
    ReducedEvaluation {
        reduced: Vec<CyclotomicRing<F, D>>,
    },
}

/// Compute the private D-block rows `v = D * e_hat` and their relation quotients.
///
/// D-role kernel: `d_row_len` is the D-matrix row count and `e_hat` carries
/// the opening digits at the D-role ring dimension. Callers extract both from
/// the schedule; this function must not read schedule types.
pub(super) fn compute_relation_d_rows<F, RB, const D: usize>(
    ring_switch_ctx: &OperationCtx<'_, F, RB>,
    d_row_len: usize,
    log_basis: u32,
    e_hat: &DigitBlocks,
    relation_mode: RingRelationMode,
) -> Result<RelationDRows<F, D>, AkitaError>
where
    F: Field + CanonicalEncoding,
    RB: RingSwitchProveBackend<F, D> + DigitRowsComputeBackend<F>,
{
    let backend = ring_switch_ctx.backend();
    let prepared = ring_switch_ctx.prepared();
    let _span = tracing::info_span!(
        "compute_relation_v",
        e_hat_planes = e_hat.typed_planes::<D>()?.len()
    )
    .entered();
    match relation_mode {
        RingRelationMode::ReducedEvaluation => {
            let rows = backend.digit_rows(
                prepared,
                d_row_len,
                &[e_hat.typed_planes::<D>()?],
                log_basis,
            )?;
            let [reduced] = rows
                .try_into()
                .map_err(|_: Vec<_>| AkitaError::InvalidProof)?;
            if reduced.len() != d_row_len {
                return Err(AkitaError::InvalidProof);
            }
            return Ok(RelationDRows::ReducedEvaluation { reduced });
        }
        RingRelationMode::QuotientLift => {}
    }
    let rows = RingSwitchRelationKernel::relation_rows(
        backend,
        prepared,
        RingSwitchRelationView {
            e_hat: e_hat.typed_planes::<D>()?,
            t_hat: &[],
            z_segment: &[],
            z_folded_centered_inf_norm: 0,
        },
        RingSwitchRelationPlan {
            n_d: d_row_len,
            n_b: 0,
            n_a: 0,
            log_basis_open: log_basis,
            log_basis_outer: log_basis,
        },
    )?;
    if rows.d_negacyclic.len() != d_row_len
        || rows.d_cyclic.len() != d_row_len
        || !rows.b_cyclic.is_empty()
        || !rows.a_quotients.is_empty()
    {
        return Err(AkitaError::InvalidProof);
    }
    let quotients = rows
        .d_cyclic
        .iter()
        .zip(&rows.d_negacyclic)
        .map(|(cyclic, reduced)| quotient_from_cyclic_and_reduced(cyclic, reduced))
        .collect();
    Ok(RelationDRows::QuotientLift {
        reduced: rows.d_negacyclic,
        quotients,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{ComputeBackendSetup, CpuBackend};
    use crate::AkitaProverSetup;
    use jolt_field::Prime64Offset59;

    #[test]
    fn reduced_d_rows_prepare_only_the_negacyclic_product() {
        type F = Prime64Offset59;
        const D: usize = 64;
        let setup = AkitaProverSetup::<F>::generate_with_capacity(
            8,
            1,
            akita_types::SetupMatrixCapacity {
                num_field_elements: 4 * D,
            },
        )
        .unwrap();
        let prepared = CpuBackend::DEFAULT
            .prepare_expanded(setup.expanded.clone())
            .unwrap();
        let ctx =
            OperationCtx::new(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref()).unwrap();
        let e_hat = DigitBlocks::new(vec![-1; 2 * D], vec![2], D).unwrap();

        let rows = compute_relation_d_rows::<F, CpuBackend, D>(
            &ctx,
            2,
            1,
            &e_hat,
            RingRelationMode::ReducedEvaluation,
        )
        .unwrap();

        assert!(matches!(
            rows,
            RelationDRows::ReducedEvaluation { ref reduced } if reduced.len() == 2
        ));
        let key = akita_types::NttCacheKey::from_matrix_shape(
            D,
            2,
            2,
            akita_types::NttTransformDomain::Negacyclic,
        )
        .unwrap();
        assert_eq!(
            prepared.ntt_cache_bytes().unwrap(),
            CpuBackend::DEFAULT
                .planned_ntt_cache_entry_bytes(&prepared, key)
                .unwrap()
        );
        assert_eq!(prepared.compression_ntt_cache_bytes(), 0);
    }
}
