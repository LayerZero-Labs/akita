use super::*;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig, SetupRequirements};

type F = fp32::Field;
type E = fp32::ExtensionField;
type RecursiveCfg = RecursiveCommitmentConfig<fp32::Dense>;

/// Commit `poly` under `scheme`'s family on `backend`, prove its single-group
/// opening against the shared `setup`, and verify the proof.
fn commit_prove_verify<Cfg, P>(
    scheme: &akita_pcs::AkitaCommitmentScheme<Cfg>,
    setup: &akita_cpu_backend::AkitaProverSetup<F>,
    backend: &CpuBackend<F, E>,
    poly: P,
    point: Vec<E>,
    expected: E,
    label: &[u8],
) where
    Cfg: CommitmentConfig<Field = F, ExtField = E>,
    P: akita_cpu_backend::CpuSource<F, E>,
{
    let akita_cpu_backend::CommitOutput {
        committed_group: commitment,
        private_handle,
    } = backend
        .commit(
            scheme.schedules(),
            &backend.import_source(vec![poly]).expect("source"),
            akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");
    let prover_data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![expected],
            commitment.clone(),
        )
        .expect("prover group")])
        .expect("prover claims"),
        vec![private_handle],
        scheme.schedules(),
    )
    .expect("prover data");
    let selection = prover_data.selection();
    let proof = scheme
        .batched_prove(setup, prover_data, backend, label, BasisMode::Lagrange)
        .expect("prove");
    let verify_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point,
        vec![expected],
        &commitment,
    )
    .expect("verifier group")])
    .expect("verifier claims");
    scheme
        .batched_verify(
            &proof,
            &scheme.setup_verifier(setup).expect("verifier setup"),
            label,
            GroupBatchStatement::new(selection, verify_claims).expect("statement"),
            BasisMode::Lagrange,
        )
        .expect("verify");
}

// One setup built from the union of a recursive family's requirements and a
// scalar family's must prove under both catalogs on one backend. `nv = 24` is
// the smallest fp32 recursive row that opens a setup prefix, so the recursive
// proof imports that slot from the combined registry rather than from a
// registry built for its family alone.
//
// This is the bounded smoke for combined recursive setups. Two recursive
// families with distinct prefix slot sets first share a field at fp128
// `(32, 4)`; `combined_recursive_setup_proves_both_onehot_families` in
// `recursive_setup_e2e.rs` proves there and is production-sized.
#[test]
fn fp32_combined_setup_proves_recursive_and_onehot_families() {
    const MAX_NV: usize = 24;
    const ONEHOT_NV: usize = 16;

    init_rayon_pool();
    run_on_large_stack(|| {
        let recursive_scheme =
            load_workspace_scheme::<RecursiveCfg>().expect("workspace recursive catalog");
        let onehot_scheme =
            load_workspace_scheme::<fp32::OneHot>().expect("workspace one-hot catalog");

        let recursive_requirements = SetupRequirements::from_catalog::<RecursiveCfg>(
            recursive_scheme.schedules(),
            MAX_NV,
            1,
        )
        .expect("recursive requirements");
        let recursive_slots = recursive_requirements.prefix_slot_ids().to_vec();
        assert!(
            !recursive_slots.is_empty(),
            "the recursive row at nv={MAX_NV} must open a setup prefix"
        );
        let onehot_requirements =
            SetupRequirements::from_catalog::<fp32::OneHot>(onehot_scheme.schedules(), MAX_NV, 1)
                .expect("one-hot requirements");
        assert_ne!(
            recursive_requirements.matrix_capacity(),
            onehot_requirements.matrix_capacity(),
            "the families must need different matrix capacities for the union to bind"
        );
        let combined = recursive_requirements
            .union(onehot_requirements)
            .expect("combined requirements");
        assert_eq!(combined.prefix_slot_ids(), recursive_slots.as_slice());

        let setup = akita_pcs::new_prover_setup::<F>(&combined).expect("combined setup");
        let backend = CpuBackend::new(setup.expanded.clone()).expect("backend");

        let n = 1usize << MAX_NV;
        let evals = (0..n)
            .map(|i| F::from_u64((i as u64).wrapping_mul(7).wrapping_add(13)))
            .collect::<Vec<_>>();
        let point = (0..MAX_NV)
            .map(|i| E::from_u64((i as u64).wrapping_mul(3).wrapping_add(1)))
            .collect::<Vec<_>>();
        let weights = lagrange_weights::<E>(&point).expect("weights");
        let expected = (0..n)
            .map(|i| weights[i] * E::lift_base(evals[i]))
            .fold(E::from_u64(0), |a, b| a + b);
        commit_prove_verify(
            &recursive_scheme,
            &setup,
            &backend,
            akita_cpu_backend::DensePoly::<F>::from_field_evals(MAX_NV, &evals)
                .expect("dense poly"),
            point,
            expected,
            b"completeness/fp32_combined_setup/recursive",
        );

        let onehot_k = akita_config::unit_onehot_source_chunk_size::<fp32::OneHot>()
            .expect("unit one-hot config");
        let onehot = akita_cpu_backend::OneHotPoly::<F, u8>::new(
            onehot_k,
            (0..(1usize << ONEHOT_NV) / onehot_k)
                .map(|chunk| Some(((chunk * 29 + 7) % onehot_k) as u8))
                .collect(),
        )
        .expect("one-hot poly");
        let point = (0..ONEHOT_NV)
            .map(|i| E::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
            .collect::<Vec<_>>();
        let expected = onehot_opening_lagrange(&onehot, &point);
        commit_prove_verify(
            &onehot_scheme,
            &setup,
            &backend,
            onehot,
            point,
            expected,
            b"completeness/fp32_combined_setup/onehot",
        );
    });
}
