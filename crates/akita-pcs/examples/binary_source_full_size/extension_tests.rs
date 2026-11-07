//! Extension claims through the public PCS, including the ordinary successor.
use crate::metrics;
use crate::run::{self, config::P64};
use akita_error::AkitaError;
use serde_json::{json, Value};

fn fixture(shape: &'static str, points: usize) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let directory = std::env::temp_dir().join(format!(
                "akita-p64-extension-{}-{shape}",
                std::process::id()
            ));
            std::fs::create_dir_all(&directory).unwrap();
            let catalog = super::fixture_directory().join(format!("3-{shape}.json"));
            std::fs::copy(&catalog, directory.join("native-catalog.json")).unwrap_or_else(
                |error| panic!("Linux catalog pending at {}: {error}", catalog.display()),
            );
            run::prove::<P64>(3, shape, &directory).unwrap();
            let timings = metrics::Timings::default();
            run::verify::<P64>(3, shape, &directory, &timings).unwrap();
            let path = directory.join("claims.json");
            let bytes = std::fs::read(&path).unwrap();
            let original: Value = serde_json::from_slice(&bytes).unwrap();
            let readings = original["claims"]["extension"].as_array().unwrap();
            assert_eq!(readings.len(), points);
            for reading in readings {
                assert!(reading["point"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|x| x[1] != 0));
                assert_ne!(reading["value"][1], 0);
                assert_ne!(reading["value"][0], reading["value"][1]);
            }
            let reject = |changed: &Value, expected: AkitaError| {
                std::fs::write(&path, serde_json::to_vec(changed).unwrap()).unwrap();
                let error = run::verify::<P64>(3, shape, &directory, &timings).unwrap_err();
                assert_eq!(
                    error.downcast_ref::<AkitaError>(),
                    Some(&expected),
                    "{error}"
                );
            };
            // Keep the honest proof fixed. Each well-formed mutation reaches the
            // verifier with the same plane count and prepared setup identity.
            for point in 0..points {
                for coordinate in 0..2 {
                    let mut changed = original.clone();
                    let value = &mut changed["claims"]["extension"][point]["value"][coordinate];
                    *value = json!(value.as_u64().unwrap() ^ 1);
                    reject(&changed, AkitaError::InvalidProof);
                    let mut changed = original.clone();
                    let value = &mut changed["claims"]["extension"][point]["point"][0][coordinate];
                    *value = json!(value.as_u64().unwrap() ^ 1);
                    reject(&changed, AkitaError::InvalidProof);
                }
                let mut changed = original.clone();
                changed["claims"]["extension"][point]["value"]
                    .as_array_mut()
                    .unwrap()
                    .swap(0, 1);
                reject(&changed, AkitaError::InvalidProof);
            }
            if points == 2 {
                let mut changed = original.clone();
                changed["claims"]["extension"]
                    .as_array_mut()
                    .unwrap()
                    .swap(0, 1);
                reject(&changed, AkitaError::InvalidProof);
                let mut changed = original.clone();
                changed["claims"]["extension"][0]["point"] = readings[1]["point"].clone();
                changed["claims"]["extension"][1]["point"] = readings[0]["point"].clone();
                reject(&changed, AkitaError::InvalidProof);
                let mut changed = original.clone();
                changed["claims"]["extension"][1]["point"] = readings[0]["point"].clone();
                reject(
                    &changed,
                    AkitaError::InvalidInput("invalid or duplicate extension prime point".into()),
                );
            }
            let mut changed = original.clone();
            changed["claims"]["extension"][0]["point"][0][1] = json!(u64::MAX - 58);
            reject(
                &changed,
                AkitaError::InvalidInput("invalid or duplicate extension prime point".into()),
            );
            // Replace the whole extension manifest by a valid base claim. This
            // must fail the typed-prefix binding before any proof is accepted.
            let mut changed = original.clone();
            changed["claims"]["extension"] = Value::Null;
            let base_point = readings[0]["point"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x[0].clone())
                .collect::<Vec<_>>();
            changed["claims"]["prime"] = json!([base_point, readings[0]["value"][0]]);
            reject(
                &changed,
                AkitaError::InvalidInput(
                    "binary statement differs from its prepared typed prefix".into(),
                ),
            );
            std::fs::write(&path, bytes).unwrap();
            run::verify::<P64>(3, shape, &directory, &timings).unwrap();
            std::fs::remove_dir_all(directory).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn p64_extension_one_point_public_pcs() {
    fixture("prime-extension-one", 1);
}

#[test]
fn p64_extension_two_points_public_pcs() {
    fixture("prime-extension-two", 2);
}

#[test]
fn p64_word_and_extension_two_points_public_pcs() {
    fixture("binary-and-prime-extension-two", 2);
}

#[test]
fn p64_mixed_extension_two_points_public_pcs() {
    fixture("binary-and-prime-extension-two-plus-ordinary", 2);
}
