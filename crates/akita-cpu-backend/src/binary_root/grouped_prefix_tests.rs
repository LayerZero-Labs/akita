use super::{Ext2, F, Ring, Zero};
use crate::CpuBackend;
use akita_types::binary_source::{
    grouped_binding_fixture, grouped_sources_fixture, GroupedNativeRootPlan, GroupedTypedPrefix,
};
use akita_types::{
    derive_public_matrix_prefix, AkitaExpandedSetup, AkitaVerifierSetup,
    SetupPrefixVerifierRegistry,
};
use std::sync::Arc;

type E = Ext2<F>;

fn setup(typed: &GroupedTypedPrefix) -> (CpuBackend<F, E>, AkitaVerifierSetup<F>) {
    let descriptor = typed.setup().clone();
    let matrix = derive_public_matrix_prefix(descriptor.num_field_elements, &descriptor.setup_seed);
    let expanded = Arc::new(
        AkitaExpandedSetup::from_verified_parts(descriptor.clone(), matrix).unwrap(),
    );
    let backend = CpuBackend::new(expanded.clone()).unwrap();
    let verifier = AkitaVerifierSetup::from_parts(
        expanded,
        SetupPrefixVerifierRegistry::new(descriptor.setup_seed),
    )
    .unwrap();
    (backend, verifier)
}

#[test]
fn grouped_prefix_preparation_authenticates_all_streams() {
    let (binary, ordinary) = grouped_sources_fixture();
    let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let typed = grouped_binding_fixture(&plan);
    let coefficients = typed.setup_coefficients::<F>().unwrap();
    for ((range, local), group) in typed.parts().iter().zip(plan.binary()) {
        let expected = local
            .setup_coefficients::<F>(
                &typed.setup().setup_seed.seed,
                &group.source().manifest,
                local.domain_len,
            )
            .unwrap();
        assert_eq!(&coefficients[range.clone()], &expected[..local.natural_len]);
    }
    assert!(coefficients[typed.natural_len()..].iter().all(|x| x.is_zero()));
    let (backend, ordinary_setup) = setup(&typed);
    let registry = backend.prepare_grouped_typed_prefixes(&typed, &[]).unwrap();
    let public = &registry.get(typed.first()).unwrap().public;
    let verifier = backend
        .prepare_grouped_typed_verifier_setup(&ordinary_setup, &typed, public)
        .unwrap();
    typed.validate_verifier_setup(&verifier).unwrap();
    assert_eq!(verifier.prefix_context(typed.first()), Some(typed.identity()));
    assert!(ordinary_setup.prefix_context(typed.first()).is_none());
    assert!(verifier.ensure_no_prefix_contexts().is_err());
    assert!(backend
        .prepare_grouped_typed_prefixes(&typed, std::slice::from_ref(typed.first()))
        .is_err());

    let mut changed = public.clone();
    let mut coefficients = changed.commitment.rows[0].coeffs().to_vec();
    coefficients[0] += F::from_u64(1);
    changed.commitment.rows[0] = akita_types::RingVec::from_coeffs(coefficients);
    assert!(backend
        .prepare_grouped_typed_verifier_setup(&ordinary_setup, &typed, &changed)
        .is_err());
}

#[test]
fn grouped_rejects_foreign_prefix_context() {
    let (mut binary, ordinary) = grouped_sources_fixture();
    let plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let typed = grouped_binding_fixture(&plan);
    let (backend, ordinary_setup) = setup(&typed);
    assert!(typed.validate_verifier_setup(&ordinary_setup).is_err());
    let registry = backend.prepare_grouped_typed_prefixes(&typed, &[]).unwrap();
    let public = &registry.get(typed.first()).unwrap().public;
    let verifier = backend
        .prepare_grouped_typed_verifier_setup(&ordinary_setup, &typed, public)
        .unwrap();
    typed.validate_verifier_setup(&verifier).unwrap();
    binary.swap(0, 1);
    let foreign_plan = GroupedNativeRootPlan::new(&binary, &ordinary, 128).unwrap();
    let foreign = grouped_binding_fixture(&foreign_plan);
    assert_eq!(typed.first(), foreign.first());
    assert!(foreign.validate_verifier_setup(&verifier).is_err());
    assert!(backend
        .prepare_grouped_typed_verifier_setup(&ordinary_setup, &foreign, public)
        .is_err());
    let mut changed_setup = typed.setup().clone();
    changed_setup.setup_seed = [42; 32].into();
    let foreign = GroupedTypedPrefix::new(
        &plan,
        *typed.successor(),
        typed.first().clone(),
        changed_setup,
    )
    .unwrap();
    assert!(backend
        .prepare_grouped_typed_prefixes(&foreign, &[])
        .is_err());
}
