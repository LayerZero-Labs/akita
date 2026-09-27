use super::*;
use crate::proof_optimized::fp128;
use akita_types::{
    derive_public_matrix_prefix, AkitaExpandedSetup, AkitaSetupDescriptor,
    SetupPrefixVerifierRegistry,
};
use jolt_field::Ring;
use std::sync::Arc;

fn setup(descriptor: AkitaSetupDescriptor) -> AkitaVerifierSetup<fp128::Field> {
    let registry = SetupPrefixVerifierRegistry::new(descriptor.setup_seed.clone());
    let matrix = derive_public_matrix_prefix(descriptor.num_field_elements, &descriptor.setup_seed);
    AkitaVerifierSetup::from_parts(
        Arc::new(AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(descriptor, matrix)),
        registry,
    )
    .unwrap()
}

#[test]
fn public_setup_identity_is_stable_and_binds_all_descriptor_fields() {
    let descriptor = AkitaSetupDescriptor {
        max_num_vars: 8,
        max_num_batched_polys: 1,
        num_field_elements: 8,
        setup_seed: [7; 32].into(),
    };
    let expected = public_setup_identity::<fp128::OneHot>(&setup(descriptor.clone())).unwrap();
    assert_eq!(
        expected,
        public_setup_identity::<fp128::OneHot>(&setup(descriptor.clone())).unwrap()
    );
    for field in 0..4 {
        let mut changed = descriptor.clone();
        match field {
            0 => changed.max_num_vars += 1,
            1 => changed.max_num_batched_polys += 1,
            2 => changed.num_field_elements += 1,
            _ => changed.setup_seed.seed[0] ^= 1,
        }
        assert_ne!(
            expected,
            public_setup_identity::<fp128::OneHot>(&setup(changed)).unwrap()
        );
    }
    let original = setup(descriptor.clone());
    let mut matrix = original.expanded().shared_matrix.as_field_slice().to_vec();
    matrix[0] += fp128::Field::from_u64(1);
    let changed = AkitaVerifierSetup::from_parts(
        Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                descriptor,
                akita_types::FlatMatrix::from_flat_data(matrix),
            ),
        ),
        original.prefix_slots().clone(),
    )
    .unwrap();
    assert_ne!(
        expected,
        public_setup_identity::<fp128::OneHot>(&changed).unwrap()
    );
    // This fixed vector pins the documented encoding across independent runs.
    assert_eq!(
        expected,
        [
            184, 123, 24, 205, 255, 249, 162, 241, 92, 244, 194, 10, 136, 147, 61, 75, 114, 210,
            59, 213, 47, 58, 106, 127, 94, 188, 116, 253, 146, 16, 169, 255
        ]
    );
}

#[test]
fn public_schedule_identity_is_stable_and_binds_selection() {
    let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>().unwrap();
    let row = catalog.rows().next().unwrap();
    let layout = row.profiles().opening_layout().unwrap();
    let identity =
        public_schedule_identity::<fp128::OneHot>(row.selection(), row.schedule(), &layout)
            .unwrap();
    let reconstructed: FoldSchedule =
        serde_json::from_value(serde_json::to_value(row.schedule()).unwrap()).unwrap();
    assert_eq!(
        identity,
        public_schedule_identity::<fp128::OneHot>(row.selection(), &reconstructed, &layout,)
            .unwrap()
    );
    let mut selection = row.selection();
    selection.row_digest = akita_types::ScheduleRowDigest::from_bytes([42; 32]);
    assert_ne!(
        identity,
        public_schedule_identity::<fp128::OneHot>(selection, row.schedule(), &layout,).unwrap()
    );
    assert_eq!(
        identity,
        [
            83, 108, 104, 26, 87, 189, 141, 92, 189, 217, 166, 112, 62, 243, 38, 133, 26, 112, 52,
            97, 29, 220, 56, 181, 95, 199, 39, 71, 163, 59, 142, 63
        ]
    );
}

#[test]
fn schedule_encoding_binds_parameter_mutations() {
    fn mutations(value: &serde_json::Value) -> Vec<serde_json::Value> {
        use serde_json::Value;
        match value {
            Value::Number(number) => number
                .as_u64()
                .and_then(|n| n.checked_add(1))
                .map(|n| vec![Value::from(n)])
                .unwrap_or_default(),
            Value::Bool(value) => vec![Value::Bool(!value)],
            Value::String(value) => [
                "Raw",
                "Compressed",
                "QuotientLift",
                "ReducedEvaluation",
                "CoefficientPacking",
                "EvaluationTrace",
            ]
            .into_iter()
            .filter(|candidate| *candidate != value)
            .map(|candidate| Value::String(candidate.into()))
            .collect(),
            Value::Object(fields) => fields
                .iter()
                .flat_map(|(key, child)| {
                    mutations(child).into_iter().map(|changed| {
                        let mut out = fields.clone();
                        out.insert(key.clone(), changed);
                        Value::Object(out)
                    })
                })
                .collect(),
            Value::Array(values) => values
                .iter()
                .enumerate()
                .flat_map(|(index, child)| {
                    mutations(child).into_iter().map(move |changed| {
                        let mut out = values.clone();
                        out[index] = changed;
                        Value::Array(out)
                    })
                })
                .collect(),
            _ => Vec::new(),
        }
    }
    fn difference(a: &serde_json::Value, b: &serde_json::Value) -> String {
        match (a, b) {
            (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
                let (key, child) = a
                    .iter()
                    .find(|(key, child)| b.get(*key) != Some(*child))
                    .unwrap();
                format!("/{key}{}", difference(child, &b[key]))
            }
            (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
                let index = a.iter().zip(b).position(|(a, b)| a != b).unwrap();
                format!("/{index}{}", difference(&a[index], &b[index]))
            }
            _ => format!(": {a} -> {b}"),
        }
    }
    let catalog = crate::test_support::workspace_schedule_catalog::<fp128::OneHot>().unwrap();
    let recursive = crate::test_support::workspace_schedule_catalog::<
        crate::RecursiveCommitmentConfig<fp128::OneHot>,
    >()
    .unwrap();
    let schedules = catalog.rows().chain(recursive.rows());
    let mut covered = 0;
    for row in schedules {
        let layout = row.profiles().opening_layout().unwrap();
        let original =
            public_schedule_identity::<fp128::OneHot>(row.selection(), row.schedule(), &layout)
                .unwrap();
        let encoded = serde_json::to_value(row.schedule()).unwrap();
        for changed in mutations(&encoded) {
            if let Ok(schedule) = serde_json::from_value::<FoldSchedule>(changed.clone()) {
                if let Ok(identity) =
                    public_schedule_identity::<fp128::OneHot>(row.selection(), &schedule, &layout)
                {
                    assert_ne!(
                        original,
                        identity,
                        "unbound parameter {}",
                        difference(&encoded, &changed)
                    );
                }
                covered += 1;
            }
        }
    }
    assert!(covered > 50, "exercise the complete expanded schedule");
}
