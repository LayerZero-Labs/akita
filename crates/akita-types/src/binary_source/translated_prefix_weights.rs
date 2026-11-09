//! Public typed weights translated into an assembled root coefficient domain.

use super::*;
use akita_algebra::offset_eq::OffsetEqWindow;
use akita_algebra::ring::trinomial::adjoint::TrinomialAdjoint;
use akita_algebra::ring::trinomial::MinusTrinomial;

impl BinaryTypedPrefix {
    /// Compile the existing A adjoint and B/D residue kernels at translated addresses.
    /// This reads no setup coefficients and returns only the natural prefix stream.
    #[allow(clippy::too_many_arguments)]
    pub fn translated_weights<E: Field>(
        &self,
        plan: &BinaryRootWitnessPlan,
        rho: &[[E; A_DEGREE]],
        root_point: &[E],
        source_offset: usize,
        alpha: E,
        batches: [&[E]; 2],
    ) -> Result<Vec<E>, AkitaError> {
        self.validate_batches(batches)?;
        let owners = plan.descriptor().polynomials().len();
        if rho.len() != checked::product([owners, plan.rank()]).ok_or(AkitaError::InvalidProof)? {
            return Err(AkitaError::InvalidProof);
        }
        let domain = checked::pow2(root_point.len()).ok_or(AkitaError::InvalidProof)?;
        if checked::sum([source_offset, plan.repacking().natural_end()?])
            .is_none_or(|end| end > domain)
        {
            return Err(AkitaError::InvalidProof);
        }
        let equality = OffsetEqWindow::new(root_point)?;
        let mut weights = Vec::new();
        weights.try_reserve_exact(self.natural_len).map_err(|_| AkitaError::InvalidProof)?;
        weights.resize(self.natural_len, E::zero());
        let adjoint = TrinomialAdjoint::<A_DEGREE, MinusTrinomial>::new()?;
        for row in 0..self.a.rows {
            for owner in 0..owners {
                let element = checked::mul_add(owner, self.a.rows, row)
                    .ok_or(AkitaError::InvalidProof)?;
                let mut response = [E::zero(); A_DEGREE];
                for (coefficient, weight) in response.iter_mut().enumerate() {
                    for (address, power) in plan.repacking()
                        .coefficient_weights(0, element, coefficient, E::one())?
                    {
                        let address = checked::sum([source_offset, address])
                            .ok_or(AkitaError::InvalidProof)?;
                        *weight += power * equality.eval(address);
                    }
                }
                for rank in 0..plan.rank() {
                    let index = checked::mul_add(owner, plan.rank(), rank)
                        .ok_or(AkitaError::InvalidProof)?;
                    let coefficients = adjoint.apply(
                        rho.get(index).ok_or(AkitaError::InvalidProof)?, &response,
                    )?;
                    let start = checked::product([row, self.a.row_stride])
                        .and_then(|start| checked::mul_add(rank, A_STRIDE, start))
                        .ok_or(AkitaError::InvalidProof)?;
                    let range = checked::range(start, A_DEGREE).ok_or(AkitaError::InvalidProof)?;
                    for (target, value) in weights.get_mut(range).ok_or(AkitaError::InvalidProof)?
                        .iter_mut().zip(coefficients)
                    {
                        *target += value;
                    }
                }
            }
        }
        let kernel = ResidueKernelPoint::new(alpha, SLOTS)?;
        for (chunk, cells) in weights.chunks_exact_mut(SLOTS).enumerate() {
            let cell = checked::product([chunk, SLOTS]).ok_or(AkitaError::InvalidProof)?;
            let Some((segment, row, q, j)) = self.locate(cell)? else {
                continue;
            };
            let mut values = [E::zero(); SLOTS];
            for (slot, value) in values.iter_mut().enumerate() {
                if let Some(address) = segment.address(q, j, slot) {
                    let address = checked::sum([source_offset, address])
                        .filter(|&address| address < domain).ok_or(AkitaError::InvalidProof)?;
                    *value = equality.eval(address);
                }
            }
            kernel.field_kernel_into(&values, cells)?;
            let batch = *batches.get(segment.role).and_then(|rows| rows.get(row))
                .ok_or(AkitaError::InvalidProof)?;
            for value in cells {
                *value *= batch;
            }
        }
        Ok(weights)
    }
}
