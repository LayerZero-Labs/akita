//! Observe order-zero binding before any witness-dependent message or challenge.
use super::*;
use akita_challenges::{ProverFoldDraw, VerifierFoldDraw};
use akita_types::binary::{binary_root_challenges, binary_root_statement};

#[test]
fn field_switch_every_order_zero_field_changes_the_first_challenge_on_both_sides() {
    let catalog = test_catalog();
    let carrier = catalog
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(18),
        ))
        .unwrap()
        .profiles()
        .final_group;
    let index = BinaryIndexMap::new(8, 128, 8, 2).unwrap();
    // Two valid identifiers let the claim's identifier vary independently of
    // its manifest. The hook tests the same canonical binder as both drivers.
    let descriptor = BinarySourceDescriptor::new(
        index.clone(),
        vec![[29; 32], [30; 32]],
        BinaryCommittedGroupManifest::profiles_for_field(&index, 2, 0, BinaryFieldProfile::P128)
            .unwrap(),
    )
    .unwrap();
    let manifest = BinaryCommittedGroupManifest::with_field_profile(
        descriptor.clone(),
        0,
        &carrier,
        BinaryFieldProfile::P128,
    )
    .unwrap();
    let claim = BinaryWordClaim::new(
        &descriptor,
        [29; 32],
        (0..7)
            .map(|axis| (31u128 + axis * 17).to_le_bytes())
            .collect(),
        19u128.to_le_bytes(),
    )
    .unwrap();
    let binary = vec![F::one(); 128];
    let ordinary = vec![F::from_u64(2); 128];
    let point = vec![F::from_u64(3); 18];
    let values = vec![F::from_u64(4)];
    let first = |manifest: &BinaryCommittedGroupManifest,
                 claim: &BinaryWordClaim,
                 binary: &[F],
                 ordinary: &[F],
                 point: &[F],
                 values: &[F]| {
        let mut prover = new_prover_channel(b"binary-root-binding-test");
        let mut verifier = new_verifier_channel(b"binary-root-binding-test", &[]);
        binary_root_statement::<F, F, _>(
            &mut prover,
            manifest,
            claim,
            &akita_types::binary::PrimePlaneManifest::with_field_profile(
                manifest.descriptor(),
                None,
                BinaryFieldProfile::P128,
            )
            .unwrap(),
            binary,
            Some((64, ordinary, point, values)),
        )
        .unwrap();
        binary_root_statement::<F, F, _>(
            &mut verifier,
            manifest,
            claim,
            &akita_types::binary::PrimePlaneManifest::with_field_profile(
                manifest.descriptor(),
                None,
                BinaryFieldProfile::P128,
            )
            .unwrap(),
            binary,
            Some((64, ordinary, point, values)),
        )
        .unwrap();
        let mut sent_draw = FirstSqueeze {
            inner: ProverFoldDraw::new(&mut prover, 0, 0),
            first: None,
        };
        let mut replayed_draw = FirstSqueeze {
            inner: VerifierFoldDraw::new(&mut verifier, 0, 0),
            first: None,
        };
        let sent = binary_root_challenges(&mut sent_draw, 2, 1).unwrap();
        let replayed = binary_root_challenges(&mut replayed_draw, 2, 1).unwrap();
        assert_eq!(sent, replayed, "prover/verifier sampled challenges differ");
        assert_eq!(
            sent_draw.first, replayed_draw.first,
            "prover/verifier first squeeze differs"
        );
        sent_draw
            .first
            .expect("binary root must squeeze its first challenge")
    };
    let baseline = first(&manifest, &claim, &binary, &ordinary, &point, &values);
    let changed = |label,
                   manifest: &BinaryCommittedGroupManifest,
                   claim: &BinaryWordClaim,
                   binary: &[F],
                   ordinary: &[F],
                   point: &[F],
                   values: &[F]| {
        assert_ne!(
            baseline,
            first(manifest, claim, binary, ordinary, point, values),
            "{label} did not change the first challenge"
        );
    };
    let other_descriptor = BinarySourceDescriptor::new(
        index.clone(),
        vec![[29; 32], [30; 32]],
        BinaryCommittedGroupManifest::profiles_for_field(&index, 2, 1, BinaryFieldProfile::P128)
            .unwrap(),
    )
    .unwrap();
    let other_manifest = BinaryCommittedGroupManifest::with_field_profile(
        other_descriptor,
        1,
        &carrier,
        BinaryFieldProfile::P128,
    )
    .unwrap();
    changed(
        "manifest",
        &other_manifest,
        &claim,
        &binary,
        &ordinary,
        &point,
        &values,
    );
    let other_claim = |polynomial, point, value| {
        BinaryWordClaim::new(&descriptor, polynomial, point, value).unwrap()
    };
    let altered = other_claim([30; 32], claim.point().to_vec(), *claim.value());
    changed(
        "polynomial",
        &manifest,
        &altered,
        &binary,
        &ordinary,
        &point,
        &values,
    );
    for coordinate in 0..claim.point().len() {
        let mut host_point = claim.point().to_vec();
        host_point[coordinate][0] ^= 1;
        let altered = other_claim(*claim.polynomial(), host_point, *claim.value());
        changed(
            "word point coordinate",
            &manifest,
            &altered,
            &binary,
            &ordinary,
            &point,
            &values,
        );
    }
    let mut value = *claim.value();
    value[0] ^= 1;
    let altered = other_claim(*claim.polynomial(), claim.point().to_vec(), value);
    changed(
        "value", &manifest, &altered, &binary, &ordinary, &point, &values,
    );
    let mut other_binary = binary.clone();
    other_binary[0] += F::one();
    changed(
        "binary commitment",
        &manifest,
        &claim,
        &other_binary,
        &ordinary,
        &point,
        &values,
    );
    let mut other_ordinary = ordinary.clone();
    other_ordinary[0] += F::one();
    changed(
        "ordinary commitment",
        &manifest,
        &claim,
        &binary,
        &other_ordinary,
        &point,
        &values,
    );
    for coordinate in 0..point.len() {
        let mut other_point = point.clone();
        other_point[coordinate] += F::one();
        changed(
            "ordinary point coordinate",
            &manifest,
            &claim,
            &binary,
            &ordinary,
            &other_point,
            &values,
        );
    }
    for index in 0..values.len() {
        let mut other_values = values.clone();
        other_values[index] += F::one();
        changed(
            "ordinary evaluation",
            &manifest,
            &claim,
            &binary,
            &ordinary,
            &point,
            &other_values,
        );
    }
}

#[test]
fn p64_prime_claim_u_coordinates_bind_the_opening() {
    init_rayon_pool();
    run_on_large_stack(|| {
        let opened = super::p64::p64_round_trip(true);
        let prime = opened.prime.clone().unwrap();
        for point in [false, true] {
            let mut changed = prime.clone();
            let value = if point {
                &mut changed.point[3]
            } else {
                &mut changed.evaluations[0].value
            };
            let [base, u]: [u128; 2] = value.coordinates().try_into().unwrap();
            *value = [base, (u + 1) % BinaryFieldProfile::P64.modulus()].into();
            opened.rejects(&opened.proof, Some(changed));
        }
    });
}
