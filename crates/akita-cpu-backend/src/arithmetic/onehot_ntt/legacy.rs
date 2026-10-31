//! Experimental one-hot A commitment using the existing prepared CRT NTT.
use super::super::{CpuBackend, CpuPreparedSetup};
use crate::kernels::linear::mat_vec_mul_ntt_digits_i8;
use crate::opaque::{CommitInnerPlan, OneHotSource};
use crate::OneHotIndex;
use akita_error::{checked, AkitaError};
use akita_types::{NttCacheKey, NttTransformDomain, RingVec};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};

impl<F: Field + CanonicalEncoding, E> CpuBackend<F, E> {
    pub(crate) fn onehot_ntt_legacy_rows<const D: usize, I: OneHotIndex>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        sources: &[OneHotSource<'_, I>],
        plan: CommitInnerPlan,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        let width = checked::product([plan.num_positions_per_block, plan.num_digits_inner])
            .ok_or_else(|| AkitaError::InvalidSetup("one-hot NTT width overflow".into()))?;
        let key =
            NttCacheKey::from_matrix_shape(D, plan.n_a, width, NttTransformDomain::Negacyclic)?;
        prepared.with_shared_ntt::<D, _>(key, |ntt| {
            sources
                .iter()
                .map(|source| {
                    let count = source.num_live_blocks_for(D, plan.num_positions_per_block)?;
                    if count != plan.num_live_blocks {
                        return Err(AkitaError::InvalidInput(
                            "one-hot NTT block count mismatch".into(),
                        ));
                    }
                    let decode = |block: usize| {
                        // Expand only this block, directly into the low digit plane.
                        // Every coefficient is 0 or 1: no digit decomposition is needed.
                        let entries = source.materialize_block_range(
                            D,
                            plan.num_positions_per_block,
                            block..block + 1,
                        )?;
                        let mut digits = vec![[0i8; D]; width];
                        for entry in entries.block(0) {
                            digits[entry.pos_in_block() * plan.num_digits_inner]
                                [entry.coeff_idx()] = 1;
                        }
                        Ok(digits)
                    };
                    // Binary coefficients fit [-2, 1]. This selects the LUT and
                    // batched-dot NTT lift without decomposing the witness or
                    // changing the schedule's actual gadget basis.
                    let rows = cfg_into_iter!(0..count)
                        .map(|block| {
                            let digits = decode(block)?;
                            mat_vec_mul_ntt_digits_i8::<F, D>(ntt, plan.n_a, width, &[&digits], 2)?
                                .pop()
                                .ok_or_else(|| {
                                    AkitaError::InvalidInput("missing one-hot NTT block".into())
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    crate::typed_inner_rows(rows)
                })
                .collect()
        })
    }
}
