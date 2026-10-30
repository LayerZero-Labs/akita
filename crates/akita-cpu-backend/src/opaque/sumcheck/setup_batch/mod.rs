//! Rectangular external accumulation: shared coefficient challenges, one setup contraction, one index phase.
use super::stage3::product_table::{
    contract_setup_coefficients, CoefficientTerm, SetupProductSource, SetupProductSumcheck,
    SetupProductWeights,
};
use crate::opaque::CpuBackend;
use akita_error::{checked, AkitaError};
use akita_prover::backend::{SetupBatchBackend, SetupBatchKernel};
use akita_sumcheck::SumcheckKernel;
use akita_types::{
    setup_claims::{
        SetupBatchMemory, SetupBatchStatement, SetupPolynomialId, SetupQuery, SetupWeightTerm,
    },
    FpExtEncoding,
};
#[cfg(feature = "parallel")]
use jolt_field::{solinas::parallel::*, Zero};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};
use std::sync::Arc;

/// Native advice preparation, bound to its owner, exact query and rectangular split.
pub struct CpuPreparedSetupQuery<E: Field> {
    owner: u64,
    polynomial: SetupPolynomialId,
    coefficient_len: usize,
    descriptor: Vec<u8>,
    terms: Vec<CoefficientTerm<E>>,
    weights: SetupProductWeights<E>,
    value: E,
}

impl<E: Field> CpuPreparedSetupQuery<E> {
    pub(in crate::opaque::sumcheck) fn value(&self) -> E {
        self.value
    }
}

impl<F, E> SetupBatchBackend<F, E> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F>,
{
    type PreparedQuery = CpuPreparedSetupQuery<E>;
    type BatchKernel = SetupProductSumcheck<F, E>;

    fn prepare_setup_query(
        &self,
        polynomial: &SetupPolynomialId,
        query: &SetupQuery<E>,
        coefficient_len: usize,
        memory: SetupBatchMemory,
    ) -> Result<(E, Self::PreparedQuery), AkitaError> {
        let prepared = self
            .prepare_setup_queries(polynomial, std::iter::once(query), coefficient_len, memory)?
            .pop()
            .ok_or(AkitaError::InvalidProof)?;
        Ok((prepared.value, prepared))
    }

    fn begin_setup_batch(
        &self,
        statement: &SetupBatchStatement<E>,
        mut prepared: Vec<Self::PreparedQuery>,
        coefficients: &[E],
    ) -> Result<Self::BatchKernel, AkitaError> {
        let setup = Arc::clone(&self.prepared()?.expanded);
        statement
            .polynomial()
            .validate_materialized::<F, E>(&setup)?;
        if prepared.len() != statement.claims().len() || coefficients.len() != prepared.len() {
            return Err(AkitaError::InvalidInput(
                "setup preparation count mismatch".into(),
            ));
        }
        for ((item, claim), &coefficient) in prepared
            .iter_mut()
            .zip(statement.claims())
            .zip(coefficients)
        {
            let mut descriptor = Vec::new();
            claim
                .query()
                .append_descriptor_bytes::<F>(&mut descriptor)?;
            if item.owner != self.owner_id()
                || item.polynomial != *statement.polynomial()
                || item.coefficient_len != statement.coefficient_len()
                || item.descriptor != descriptor
                || item.value != claim.value()
            {
                return Err(AkitaError::InvalidInput(
                    "setup preparation does not match the bound claim".into(),
                ));
            }
            // Every term of one query receives the same batching coefficient.
            for term in &mut item.terms {
                for value in &mut term.factor {
                    *value *= coefficient;
                }
            }
        }
        let mut terms = Vec::new();
        let mut weights = Vec::with_capacity(prepared.len());
        for query in prepared {
            terms.extend(query.terms);
            weights.push(query.weights);
        }
        let kernel = SetupProductSumcheck::from_prepared(
            SetupProductSource::Expanded(setup),
            statement.polynomial().field_len() / statement.coefficient_len(),
            statement.coefficient_len(),
            terms,
            weights,
        )?;
        if kernel.input_claim() != statement.input_claim(coefficients)? {
            return Err(AkitaError::InvalidProof);
        }
        Ok(kernel)
    }
}

impl<F, E> CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F>,
{
    /// Compile every available query before traversing the setup once for all terms.
    pub(in crate::opaque::sumcheck) fn prepare_setup_queries<'a>(
        &self,
        polynomial: &SetupPolynomialId,
        queries: impl IntoIterator<Item = &'a SetupQuery<E>>,
        coefficient_len: usize,
        memory: SetupBatchMemory,
    ) -> Result<Vec<CpuPreparedSetupQuery<E>>, AkitaError>
    where
        E: 'a,
    {
        let setup = &self.prepared()?.expanded;
        polynomial.validate_materialized::<F, E>(setup)?;
        let compiled = queries
            .into_iter()
            .map(|query| Ok((query, query.compile(polynomial, coefficient_len)?)))
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let source = setup
            .shared_matrix()
            .as_field_slice()
            .get(..polynomial.field_len())
            .ok_or(AkitaError::InvalidProof)?;
        let factors = compiled
            .iter()
            .flat_map(|(_, terms)| terms)
            .collect::<Vec<_>>();
        let mut tables = coefficient_contractions(source, coefficient_len, &factors)?.into_iter();
        compiled
            .into_iter()
            .map(|(query, factors)| {
                let terms = factors
                    .iter()
                    .map(|term| {
                        Ok(CoefficientTerm {
                            table: tables.next().ok_or(AkitaError::InvalidProof)?,
                            factor: term.coefficient().to_vec(),
                        })
                    })
                    .collect::<Result<Vec<_>, AkitaError>>()?;
                let value = terms
                    .iter()
                    .flat_map(|term| term.table.iter().zip(&term.factor))
                    .fold(E::zero(), |sum, (&s, &w)| sum + s * w);
                let mut descriptor = Vec::new();
                query.append_descriptor_bytes::<F>(&mut descriptor)?;
                let weights = match memory {
                    SetupBatchMemory::Retain => SetupProductWeights::Retained(factors),
                    SetupBatchMemory::Regenerate => SetupProductWeights::Regenerate {
                        polynomial: polynomial.clone(),
                        query: query.clone(),
                    },
                };
                Ok(CpuPreparedSetupQuery {
                    owner: self.owner_id(),
                    polynomial: polynomial.clone(),
                    coefficient_len,
                    descriptor,
                    terms,
                    weights,
                    value,
                })
            })
            .collect()
    }
}

impl<F, E> SetupBatchKernel<E> for SetupProductSumcheck<F, E>
where
    F: Field + CanonicalEncoding,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F>,
{
    fn setup_evaluation(&self) -> Result<E, AkitaError> {
        self.folded_table_value()
    }
}

#[cfg(test)]
std::thread_local! {
    static PREPARATION_SWEEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Fuse all query terms in one traversal; a singleton keeps the existing kernel.
fn coefficient_contractions<F, E>(
    source: &[F],
    dim: usize,
    terms: &[&SetupWeightTerm<E>],
) -> Result<Vec<Vec<E>>, AkitaError>
where
    F: Field,
    E: Field + Ring + MulBaseUnreduced<F>,
{
    #[cfg(test)]
    PREPARATION_SWEEPS.with(|sweeps| sweeps.set(sweeps.get() + 1));
    let rows = checked::exact_div(source.len(), dim).ok_or(AkitaError::InvalidProof)?;
    if terms
        .iter()
        .any(|term| term.coefficient().len() != dim || term.index_len() != rows)
    {
        return Err(AkitaError::InvalidProof);
    }
    if let [term] = terms {
        let active = term.active_index_len();
        let table = match term.dense_index() {
            Some(index) => contract_setup_coefficients(source, dim, active, |row| index[row]),
            None => contract_setup_coefficients(source, dim, active, |row| term.index_weight(row)),
        };
        return Ok(vec![table]);
    }
    #[cfg(not(feature = "parallel"))]
    {
        const SETUP_BLOCK_BYTES: usize = 2 << 20;
        let active = terms
            .iter()
            .map(|term| term.active_index_len())
            .max()
            .unwrap_or(0);
        let source_len = checked::product([active, dim]).ok_or(AkitaError::InvalidProof)?;
        let source = source.get(..source_len).ok_or(AkitaError::InvalidProof)?;
        // Reuse a small setup block across every query while keeping the singleton
        // contraction's small accumulator and optimized arithmetic loop.
        let row_bytes = checked::product([dim, std::mem::size_of::<F>()])
            .ok_or(AkitaError::InvalidProof)?
            .max(1);
        let block_rows = (SETUP_BLOCK_BYTES / row_bytes).max(1);
        let block_len = checked::product([block_rows, dim]).ok_or(AkitaError::InvalidProof)?;
        let mut tables = (0..terms.len())
            .map(|_| vec![E::zero(); dim])
            .collect::<Vec<_>>();
        for (block, source) in source.chunks(block_len).enumerate() {
            let start = block * block_rows;
            for (table, term) in tables.iter_mut().zip(terms) {
                let required = term
                    .active_index_len()
                    .saturating_sub(start)
                    .min(source.len() / dim);
                if required == 0 {
                    continue;
                }
                let values = match term.dense_index() {
                    Some(index) => {
                        contract_setup_coefficients(source, dim, required, |row| index[start + row])
                    }
                    None => contract_setup_coefficients(source, dim, required, |row| {
                        term.index_weight(start + row)
                    }),
                };
                for (value, addend) in table.iter_mut().zip(values) {
                    *value += addend;
                }
            }
        }
        Ok(tables)
    }
    #[cfg(feature = "parallel")]
    {
        let count = terms.len();
        if count == 0 {
            return Ok(Vec::new());
        }
        let slots = checked::product([count, dim]).ok_or(AkitaError::InvalidProof)?;
        let active = terms
            .iter()
            .map(|term| term.active_index_len())
            .max()
            .unwrap_or(0);
        let (products, _) = cfg_fold_reduce!(
            0..active,
            || (
                (0..slots)
                    .map(|_| <E as jolt_field::Unreduced>::Product::zero())
                    .collect::<Vec<_>>(),
                vec![E::zero(); count],
            ),
            |(mut products, mut weights), index| {
                for (weight, term) in weights.iter_mut().zip(terms) {
                    *weight = term.index_weight(index);
                }
                let row = &source[index * dim..(index + 1) * dim];
                // Load each setup coefficient once and apply all query weights to it.
                for (&base, products) in row.iter().zip(products.chunks_exact_mut(count)) {
                    for (product, &weight) in products.iter_mut().zip(&weights) {
                        if !weight.is_zero() {
                            *product += weight.mul_base_unreduced(base);
                        }
                    }
                }
                (products, weights)
            },
            |(mut left, weights), (right, _)| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                (left, weights)
            }
        );
        let mut tables = (0..count)
            .map(|_| Vec::with_capacity(dim))
            .collect::<Vec<_>>();
        let mut products = products.into_iter();
        for _ in 0..dim {
            for table in &mut tables {
                table.push(E::reduce_product(
                    products.next().ok_or(AkitaError::InvalidProof)?,
                ));
            }
        }
        Ok(tables)
    }
}

#[cfg(test)]
mod tests;
