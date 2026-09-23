#![allow(missing_docs)]

use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use akita_prover::compute::{
    aggregate_decompose_fold_witnesses, CpuBackend, DecomposeFoldBatchPlan, OpeningBatchKernel,
};
use akita_prover::DecomposeFoldWitness;
use jolt_field::{Prime64Offset59, Ring};

type F = Prime64Offset59;
const D: usize = 4;

struct ExternalBatch<'a>(&'a [DecomposeFoldWitness<F>]);

impl OpeningBatchKernel<ExternalBatch<'_>, F, D> for CpuBackend {
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: ExternalBatch<'_>,
        _plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<Vec<DecomposeFoldWitness<F>>, AkitaError> {
        Ok(vec![aggregate_decompose_fold_witnesses::<F, D>(
            source.0.iter().cloned().map(Ok),
        )?])
    }
}

#[test]
fn downstream_batch_kernel_reuses_checked_aggregation() {
    let witnesses = [
        DecomposeFoldWitness::from_coefficient_parts(vec![[F::from_u64(1); D]], vec![[1; D]]),
        DecomposeFoldWitness::from_coefficient_parts(vec![[F::from_u64(1); D]], vec![[2; D]]),
    ];
    let challenges = [SparseChallenge {
        positions: vec![0].into(),
        coeffs: vec![1].into(),
    }];
    let got = <CpuBackend as OpeningBatchKernel<ExternalBatch<'_>, F, D>>::decompose_fold_batch(
        &CpuBackend::DEFAULT,
        None,
        ExternalBatch(&witnesses),
        DecomposeFoldBatchPlan::Sparse {
            challenges: &challenges,
            challenges_per_poly: 1,
            num_chunks: 1,
            num_positions_per_block: 1,
            num_digits: 1,
            log_basis: 1,
        },
    )
    .unwrap();

    assert_eq!(got[0].centered_coeffs_flat(), &[3; D]);
    assert_eq!(got[0].z_folded_rings.coeffs(), &[F::from_u64(2); D]);
}
