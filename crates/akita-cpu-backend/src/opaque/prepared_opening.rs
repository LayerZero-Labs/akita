//! Construction of fully bound CPU prepared-opening handles.

use crate::opaque::openings::PreparedOpeningSource;
use crate::opaque::{CpuPreparedOpeningHandle, OperationBinding};
use akita_prover::backend::PreparedGroupOpening;
use jolt_field::{CanonicalEncoding, Field};

pub(in crate::opaque) fn evaluation_trace<F, E, Cfg>(
    binding: OperationBinding,
    source: PreparedOpeningSource<F, E, Cfg>,
    point: akita_types::PreparedOpeningPoint<F, E>,
    folded_by_claim: Vec<akita_types::RingVec<F>>,
    scalar_openings: Vec<E>,
) -> PreparedGroupOpening<E, CpuPreparedOpeningHandle<F, E, Cfg>>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F>,
{
    let handle = CpuPreparedOpeningHandle::evaluation_trace(
        binding,
        source,
        point,
        folded_by_claim,
        scalar_openings.clone(),
    );
    PreparedGroupOpening::new(scalar_openings, handle)
}

pub(in crate::opaque) fn coefficient_packing<F, E, Cfg>(
    binding: OperationBinding,
    source: PreparedOpeningSource<F, E, Cfg>,
    point: akita_types::PreparedSubringCoefficientPackingPoint<E>,
    partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
    scalar_openings: Vec<E>,
) -> PreparedGroupOpening<E, CpuPreparedOpeningHandle<F, E, Cfg>>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F>,
{
    let handle = CpuPreparedOpeningHandle::coefficient_packing(
        binding,
        source,
        point,
        partials_by_claim,
        scalar_openings.clone(),
    );
    PreparedGroupOpening::new(scalar_openings, handle)
}
