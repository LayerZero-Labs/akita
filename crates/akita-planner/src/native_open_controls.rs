//! Binding refusals and evaluation-equation controls for the complete opening.

use super::*;
use akita_algebra::{One, Zero};
use akita_config::proof_optimized::fp64::ExtensionField as E;
use akita_types::binary_source::PrimePlaneManifest;
use akita_types::{AkitaVerifierSetup, OpeningClaims, PolynomialGroupClaims};
use native_open_fixture::{ext, opening, Claims};

pub(super) fn reject_foreign_context(tamper: bool) {
    let opening = opening();
    let mut identity = opening.binding.identity().to_vec();
    if tamper {
        identity[0] ^= 1;
    }
    let ordinary = AkitaVerifierSetup::from_parts(
        opening.setup.expanded().clone(),
        opening.setup.prefix_slots().clone(),
    )
    .unwrap();
    let setup = if tamper {
        ordinary
            .with_prefix_context(opening.binding.first().clone(), identity)
            .unwrap()
    } else {
        let (mut binary, signed) = actual_profiles::sources();
        let mut claims = binary[0].prime.claims().to_vec();
        claims[0].point[0] = [5, 7].into();
        binary[0].prime = PrimePlaneManifest::with_field_profile(
            binary[0].manifest.descriptor(),
            claims,
            binary[0].manifest.field_profile(),
        )
        .unwrap();
        let plan = GroupedNativeRootPlan::new(
            &binary,
            &signed,
            opening.binding.outgoing().destination_ring_dimension(),
        )
        .unwrap();
        let foreign = GroupedTypedPrefix::new(
            &plan,
            *opening.binding.successor(),
            opening.binding.first().clone(),
            opening.binding.setup().clone(),
        )
        .unwrap();
        let backend =
            akita_cpu_backend::CpuBackend::<Field, E>::new(opening.setup.expanded().clone())
                .unwrap();
        let prepared = backend
            .prepare_grouped_typed_prefixes(&foreign, &[])
            .unwrap();
        let setup = backend
            .prepare_grouped_typed_verifier_setup(
                &ordinary,
                &foreign,
                &prepared.get(foreign.first()).unwrap().public,
            )
            .unwrap();
        foreign.validate_verifier_setup(&setup).unwrap();
        assert_eq!(foreign.outgoing(), opening.binding.outgoing());
        setup
    };
    assert!(opening.binding.validate_verifier_setup(&setup).is_err());
    assert!(opening
        .verify(&opening.claims, &setup, &opening.proof)
        .is_err());
}

pub(super) fn reject_substituted_plan() {
    let opening = opening();
    let (binary, mut signed) = actual_profiles::sources();
    signed[3].1 = akita_params::ordinary_source::OrdinarySourceWidth::new(34).unwrap();
    let changed = GroupedNativeRootPlan::new(
        &binary,
        &signed,
        opening.binding.outgoing().destination_ring_dimension(),
    )
    .unwrap();
    assert!(opening.binding.validate_plan(&changed).is_err());
}

pub(super) fn reject_source_order() {
    let opening = opening();
    let mut groups = opening.claims.groups().to_vec();
    groups.swap(0, 1);
    let claims = OpeningClaims::from_groups(groups).unwrap();
    let row = opening
        .fixture
        .catalog
        .resolve_selection(opening.fixture.bindings[0].0)
        .unwrap();
    assert!(akita_config::bind_grouped_root(row, opening.binding.setup(), &claims).is_err());
    assert!(opening
        .verify(&claims, &opening.setup, &opening.proof)
        .is_err());
}

pub(super) fn reject_wrong_field() {
    let opening = opening();
    type Other = akita_config::proof_optimized::fp128::Field;
    let descriptor = opening.binding.setup().clone();
    let matrix = akita_types::derive_public_matrix_prefix::<Other>(
        descriptor.num_field_elements,
        &descriptor.setup_seed,
    );
    let public = akita_types::SetupPrefixVerifierRegistry::new(descriptor.setup_seed.clone());
    let expanded =
        akita_types::AkitaExpandedSetup::from_verified_parts(descriptor, matrix).unwrap();
    let foreign = AkitaVerifierSetup::from_parts(std::sync::Arc::new(expanded), public).unwrap();
    assert!(opening.binding.validate_verifier_setup(&foreign).is_err());
    assert!(opening.binding.setup_coefficients::<Other>().is_err());
    assert!(akita_params::binary_setup::BinaryFieldProfile::P64
        .check_field_pair(
            akita_params::SisModulusProfileId::Q128Offset275.modulus(),
            1,
        )
        .is_err());
}

pub(super) fn reject_short_capacity() {
    let opening = opening();
    let mut setup = opening.binding.setup().clone();
    setup.num_field_elements = opening.fixture.bindings[0]
        .1
        .matrix_capacity()
        .num_field_elements
        - 1;
    let row = opening
        .fixture
        .catalog
        .resolve_selection(opening.fixture.bindings[0].0)
        .unwrap();
    assert!(akita_config::bind_grouped_root(row, &setup, &opening.claims).is_err());
}

pub(super) fn reject_values(point: usize) {
    let opening = opening();
    let row = opening
        .fixture
        .catalog
        .resolve_selection(opening.fixture.bindings[0].0)
        .unwrap();
    for group in 0..8 {
        let original = &opening.claims.groups()[group];
        let slots = if group < 4 {
            1
        } else {
            original.evaluations().len()
        };
        for slot in 0..slots {
            for coordinate in 0..2 {
                let mut groups = opening.claims.groups().to_vec();
                if group < 4 {
                    let akita_types::opening_claims::GroupClaimKind::Binary {
                        manifest,
                        claim,
                        prime,
                    } = original.kind()
                    else {
                        unreachable!()
                    };
                    let mut readings = prime.claims().to_vec();
                    let mut value: [u128; 2] = readings[point].evaluations[slot]
                        .value
                        .coordinates()
                        .try_into()
                        .unwrap();
                    value[coordinate] =
                        (value[coordinate] + 1) % manifest.field_profile().modulus();
                    readings[point].evaluations[slot].value = value.into();
                    groups[group] = PolynomialGroupClaims::new_binary_planes(
                        original.commitment().clone(),
                        claim.clone(),
                        readings.clone(),
                    )
                    .unwrap();
                } else {
                    let readings = original.point_evaluations().collect::<Vec<_>>();
                    let points = [readings[0].point.to_vec(), readings[1].point.to_vec()];
                    let mut values = [
                        readings[0].evaluations.to_vec(),
                        readings[1].evaluations.to_vec(),
                    ];
                    let mut delta = [0; 2];
                    delta[coordinate] = 1;
                    values[point][slot] += ext::<E>(delta);
                    let signed = &opening.binding.plan().ordinary().unwrap().plans()[group - 4];
                    let mut batches = vec![E::zero(); slots];
                    batches[slot] = E::one();
                    let weights = |values: &[E]| {
                        signed
                            .evaluation_weights(
                                opening.binding.outgoing(),
                                &points[point],
                                values,
                                &batches,
                            )
                            .unwrap()
                    };
                    let honest = weights(readings[point].evaluations);
                    let changed = weights(&values[point]);
                    // The coefficient oracle depends on the point, not the claimed
                    // value. An isolated slot changes only the equation's RHS.
                    assert_eq!(honest.claim(), readings[point].evaluations[slot]);
                    assert_eq!(changed.claim() - honest.claim(), ext::<E>(delta));
                    let variables = honest.table_len().trailing_zeros() as usize;
                    let probe = vec![ext::<E>([11, 13]); variables];
                    assert_eq!(
                        honest.evaluate(&probe).unwrap(),
                        changed.evaluate(&probe).unwrap()
                    );
                    groups[group] = PolynomialGroupClaims::new_two_points(
                        points,
                        values,
                        original.commitment().clone(),
                    )
                    .unwrap();
                }
                let changed: Claims = OpeningClaims::from_groups(groups).unwrap();
                // Binary values are part of the immutable plan identity; ordinary
                // values are bound by the statement and tested by its equations.
                assert_eq!(
                    akita_config::bind_grouped_root(row, opening.binding.setup(), &changed,)
                        .is_err(),
                    group < 4
                );
                assert!(
                    opening
                        .verify(&changed, &opening.setup, &opening.proof)
                        .is_err(),
                    "accepted group {group}, slot {slot}, point {point}, coordinate {coordinate}"
                );
            }
        }
    }
}
