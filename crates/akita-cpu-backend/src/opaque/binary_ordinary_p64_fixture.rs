//! The actual P64 ordinary commitment preimage used by the mixed root fixture.
use crate::commitment::{CommitmentExecutor, PortableStatePolicy};
use crate::opaque::owned::{CommitmentHandle, CommittedSource};
use crate::{CpuBackend, DensePoly};
use akita_algebra::eq_poly::EqPolynomial;
use akita_params::GroupCommitPhaseParams;
use akita_types::binary::OrdinaryNativeRootPlan;
use jolt_field::{Ext2, ExtField, Prime64Offset59 as F, Ring, Zero};
use std::sync::Arc;
type E = Ext2<F>;

pub(crate) fn ordinary_preimage(
    profile: GroupCommitPhaseParams,
    plan: &OrdinaryNativeRootPlan,
) -> (Vec<Vec<i8>>, Vec<E>, E, Vec<E>) {
    let setup = akita_types::AkitaSetupDescriptor {
        max_num_vars: 10,
        max_num_batched_polys: 1,
        num_field_elements: 1 << 16,
        setup_seed: [41; 32].into(),
    };
    let matrix =
        akita_types::derive_public_matrix_prefix::<F>(setup.num_field_elements, &setup.setup_seed);
    let expanded = akita_types::AkitaExpandedSetup::from_verified_parts(setup, matrix).unwrap();
    let backend = CpuBackend::<F, E>::new(Arc::new(expanded)).unwrap();
    // In particular, -1 is stored as p-1; the signed decoder must use p.
    let values = (0..1024)
        .map(|i| F::from_i64((i % 8) as i64 - 4))
        .collect::<Vec<_>>();
    let source = backend
        .import_source(vec![DensePoly::from_field_evals(10, &values).unwrap()])
        .unwrap();
    let prepared = backend.prepared().unwrap();
    let executor = CommitmentExecutor::cpu(
        &backend,
        prepared,
        &prepared.expanded,
        Vec::new(),
        PortableStatePolicy,
    )
    .unwrap();
    let (public, retained) = executor
        .execute_full_via_outer_image(
            &backend,
            profile,
            &source.storage.commitment_sources(),
            None,
        )
        .unwrap();
    drop(executor);
    let payload = public
        .rows()
        .coeffs()
        .iter()
        .copied()
        .map(E::lift_base)
        .collect();
    let handle = CommitmentHandle {
        owner: backend.owner_id(),
        committed: Arc::new(CommittedSource {
            source: source.storage,
            metadata: source.metadata,
            commitment_id: backend.owner().next_operation_id().unwrap(),
            parameters: profile,
            producer_contract: akita_params::ordinary_source::OrdinarySourceWidth::new(3)
                .unwrap()
                .producer_contract()
                .unwrap(),
            public: public.commitment().clone(),
            retained: Some(retained),
        }),
    };
    let preimage = backend.ordinary_root_preimage(&handle, plan).unwrap();
    assert!(preimage[0].contains(&-1));
    let point = (0..10)
        .map(|i| E::from_base_fn(|c| F::from_u64(17 + i * 3 + c as u64 * 43)))
        .collect::<Vec<_>>();
    let evaluation = values
        .iter()
        .zip(EqPolynomial::evals(&point).unwrap())
        .fold(E::zero(), |sum, (&v, w)| sum + E::lift_base(v) * w);
    (preimage, point, evaluation, payload)
}
