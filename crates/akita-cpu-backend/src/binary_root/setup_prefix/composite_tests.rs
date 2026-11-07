use super::*;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_types::binary_source::binary_word_manifest_fixture;
use jolt_field::{Ext2, Prime64Offset59 as F, Zero};
type E = Ext2<F>;

fn slot(domain: usize) -> SetupPrefixSlotId {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<
        akita_config::proof_optimized::fp64::Dense,
    >(akita_params::ChunkedWitnessCfg::default_non_chunked())
    .unwrap();
    let profile = catalog
        .catalog()
        .rows()
        .find_map(|row| {
            std::iter::once(&row.schedule().root)
                .chain(&row.schedule().recursive_folds)
                .find_map(|fold| {
                    akita_params::setup_prefix_precommitted_params(&fold.params, domain).ok()
                })
        })
        .expect("fixture successor prefix")
        .profile;
    SetupPrefixSlotId {
        natural_len: domain,
        commitment_profile: profile,
    }
}

fn encoded(slot: &SetupPrefixVerifierSlot<F>) -> Vec<u8> {
    let mut bytes = Vec::new();
    slot.serialize_compressed(&mut bytes).unwrap();
    bytes
}

#[test]
fn trusted_composite_preparation_preserves_singleton_and_authenticates_distinct_seeds() {
    let manifests =
        [1, 2].map(|owners| binary_word_manifest_fixture(BinaryFieldProfile::P64, owners));
    let typed = manifests
        .each_ref()
        .map(|manifest| BinaryTypedPrefix::new(manifest, true, 0, 64).unwrap());
    let slots = typed.each_ref().map(|prefix| slot(prefix.domain_len));
    let combined_domain = (typed[0].domain_len.div_ceil(typed[1].domain_len) * typed[1].domain_len
        + typed[1].domain_len)
        .next_power_of_two();
    let combined_slot = slot(combined_domain);
    let capacity = akita_params::SetupMatrixCapacity {
        num_field_elements: slots
            .iter()
            .chain(std::iter::once(&combined_slot))
            .map(|id| akita_params::setup_prefix_slot_field_elements(id).unwrap())
            .max()
            .unwrap(),
    };
    let setup = crate::AkitaProverSetup::<F>::generate_with_capacity(20, 1, capacity).unwrap();
    let backend = CpuBackend::<F, E>::new(setup.expanded.clone()).unwrap();
    let seeds = [setup.expanded.descriptor().setup_seed.seed, [93; 32]];
    let blocks = (0..2)
        .map(|index| {
            CompositePrefixBlock::binary(
                index,
                &seeds[index],
                &manifests[index],
                &typed[index],
                &slots[index],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let definitions = (0..2)
        .map(|index| TrustedCompositePrefixDefinition::Binary {
            seed: &seeds[index],
            manifest: &manifests[index],
            typed: &typed[index],
            local_slot: &slots[index],
        })
        .collect::<Vec<_>>();
    let singleton = CompositePrefixLayout::new(&blocks[..1], b"singleton root", &slots[0]).unwrap();
    let original = backend
        .prepare_binary_a_prefixes(&manifests[0], &typed[0], &slots[0], &[])
        .unwrap();
    let prepared = backend
        .prepare_composite_prefix(&singleton, &definitions[..1], typed[0].domain_len)
        .unwrap();
    assert_eq!(
        encoded(&prepared.prepared().public),
        encoded(&original.get(&slots[0]).unwrap().public)
    );
    assert_eq!(
        prepared.cost,
        CompositePrefixPreparationCost {
            table_len: typed[0].domain_len,
            local_scans: 1
        }
    );

    let layout =
        CompositePrefixLayout::new(&blocks, b"two translated binary roots", &combined_slot)
            .unwrap();
    assert!(backend
        .prepare_composite_prefix(&layout, &definitions, combined_domain - 1)
        .is_err());
    assert!(backend
        .prepare_composite_prefix(&layout, &definitions[..1], combined_domain)
        .is_err());
    let wrong_seed = [94; 32];
    let wrong = [TrustedCompositePrefixDefinition::Binary {
        seed: &wrong_seed,
        manifest: &manifests[0],
        typed: &typed[0],
        local_slot: &slots[0],
    }];
    assert!(backend
        .prepare_composite_prefix(&singleton, &wrong, typed[0].domain_len)
        .is_err());
    let prepared = backend
        .prepare_composite_prefix(&layout, &definitions, combined_domain)
        .unwrap();
    assert_eq!(prepared.identity(), layout.identity());
    assert_eq!(
        prepared.cost,
        CompositePrefixPreparationCost {
            table_len: combined_domain,
            local_scans: 2
        }
    );
    let local = (0..2)
        .map(|i| {
            typed[i]
                .setup_coefficients::<F>(&seeds[i], &manifests[i], typed[i].domain_len)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut dense = vec![F::zero(); combined_domain];
    for (local, placement) in local.iter().zip(layout.placements()) {
        dense[placement.offset..placement.offset + local.len()].copy_from_slice(local);
    }
    let expected = backend
        .prepare_setup_prefix_from_coefficients(&combined_slot, dense)
        .unwrap();
    assert_eq!(
        encoded(&prepared.prepared().public),
        encoded(&expected.public)
    );
    let ordinary = setup.to_verifier_setup(capacity).unwrap();
    let verifier = backend
        .prepare_composite_verifier_setup(&ordinary, &layout, &prepared)
        .unwrap();
    assert_eq!(
        verifier.prefix_context(&combined_slot),
        Some(layout.identity())
    );
    assert_eq!(
        encoded(verifier.prefix_slots().get(&combined_slot).unwrap()),
        encoded(&prepared.prepared().public)
    );
    let changed =
        CompositePrefixLayout::new(&blocks, b"different root addresses", &combined_slot).unwrap();
    assert!(backend
        .prepare_composite_verifier_setup(&ordinary, &changed, &prepared)
        .is_err());
}

#[test]
fn opaque_ordinary_local_table_is_copied_without_reinterpreting_its_roles() {
    let manifest = binary_word_manifest_fixture(BinaryFieldProfile::P64, 1);
    let typed = BinaryTypedPrefix::new(&manifest, true, 0, 64).unwrap();
    let local_slot = slot(typed.domain_len);
    let combined_slot = slot(typed.domain_len * 2);
    let capacity = akita_params::SetupMatrixCapacity {
        num_field_elements: akita_params::setup_prefix_slot_field_elements(&combined_slot).unwrap(),
    };
    let setup = crate::AkitaProverSetup::<F>::generate_with_capacity(20, 1, capacity).unwrap();
    let backend = CpuBackend::<F, E>::new(setup.expanded).unwrap();
    let seed = [71; 32];
    let binary = CompositePrefixBlock::binary(0, &seed, &manifest, &typed, &local_slot).unwrap();
    let ordinary = CompositePrefixBlock {
        kind: CompositePrefixKind::Ordinary,
        index: 1,
        setup_identity: [9; 32],
        local_identity: b"authenticated ordinary local definition".to_vec(),
        domain_len: 256,
    };
    let layout = CompositePrefixLayout::new(
        &[binary, ordinary.clone()],
        b"binary and opaque ordinary root",
        &combined_slot,
    )
    .unwrap();
    let coefficients = vec![F::from_u64(17); 256];
    let definitions = [
        TrustedCompositePrefixDefinition::Binary {
            seed: &seed,
            manifest: &manifest,
            typed: &typed,
            local_slot: &local_slot,
        },
        TrustedCompositePrefixDefinition::Ordinary {
            block: &ordinary,
            coefficients: &coefficients,
        },
    ];
    let prepared = backend
        .prepare_composite_prefix(&layout, &definitions, layout.domain_len())
        .unwrap();
    let local = typed
        .setup_coefficients::<F>(&seed, &manifest, typed.domain_len)
        .unwrap();
    let mut dense = vec![F::zero(); layout.domain_len()];
    dense[..local.len()].copy_from_slice(&local);
    dense[local.len()..local.len() + 256].copy_from_slice(&coefficients);
    let expected = backend
        .prepare_setup_prefix_from_coefficients(&combined_slot, dense)
        .unwrap();
    assert_eq!(
        encoded(&prepared.prepared().public),
        encoded(&expected.public)
    );
    let invalid = [
        TrustedCompositePrefixDefinition::Binary {
            seed: &seed,
            manifest: &manifest,
            typed: &typed,
            local_slot: &local_slot,
        },
        TrustedCompositePrefixDefinition::Ordinary {
            block: &ordinary,
            coefficients: &coefficients[..255],
        },
    ];
    assert!(backend
        .prepare_composite_prefix(&layout, &invalid, layout.domain_len())
        .is_err());
}
