use super::*;
use crate::binary_setup::{SetupStreamRing, SetupStreamRole, BINARY_SETUP_P64};

fn labels(column: usize, role: SetupStreamRole) -> [SetupStreamLabel; 4] {
    std::array::from_fn(|lane| SetupStreamLabel {
        role,
        group: 19,
        level: 0,
        rank: 2,
        row: (lane % 2) as u32,
        column: (column + lane / 2) as u64,
        ring: SetupStreamRing::Negacyclic,
        degree: 64,
        modulus: SetupStreamModulus::Prime64Offset59,
    })
}

fn compare() {
    for seed in [[0; 32], [51; 32], [255; 32]] {
        for role in [
            SetupStreamRole::ImageCommitment,
            SetupStreamRole::WitnessCommitment,
        ] {
            for columns in [1usize, 2, 3, 4, 5, 7] {
                for column in (0..columns).step_by(2) {
                    let labels = labels(column, role);
                    let mut streams = new_four(&seed, labels).unwrap();
                    let mut scalar =
                        labels.map(|label| TypedSetupStream::new(&seed, label).unwrap());
                    // Every lane, also for the padded quartet in an odd-column case.
                    for length in [0, 1, 7, 8, 135, 136, 137, 512, 4096] {
                        let mut actual = std::array::from_fn(|_| vec![0; length]);
                        read_four(&mut streams, actual.each_mut().map(Vec::as_mut_slice));
                        for (lane, expected) in scalar.iter_mut().enumerate() {
                            let mut bytes = vec![0; length];
                            expected.squeeze.read(&mut bytes);
                            assert_eq!(actual[lane], bytes, "column {column}, lane {lane}");
                        }
                    }
                    // Divergent cursors and unequal lengths must not replay any bytes.
                    let mut actual = std::array::from_fn(|lane| vec![0; 137 * lane]);
                    read_four(&mut streams, actual.each_mut().map(Vec::as_mut_slice));
                    for (lane, expected) in scalar.iter_mut().enumerate() {
                        let mut bytes = vec![0; 137 * lane];
                        expected.squeeze.read(&mut bytes);
                        assert_eq!(actual[lane], bytes);
                    }
                    let mut tail = [[0; 512]; 4];
                    read_four(&mut streams, tail.each_mut().map(|lane| &mut lane[..]));
                    for (lane, expected) in scalar.iter_mut().enumerate() {
                        let mut bytes = [0; 512];
                        expected.squeeze.read(&mut bytes);
                        assert_eq!(tail[lane], bytes);
                    }
                    let actual = TypedSetupStream::p64_four_entries(&seed, labels).unwrap();
                    for (lane, label) in labels.into_iter().enumerate() {
                        let mut expected = TypedSetupStream::new(&seed, label).unwrap();
                        for value in actual[lane] {
                            assert_eq!(value, u128::from(expected.sample_p64().unwrap()));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn four_way_typed_bytes_and_entries() {
    compare();
}

#[test]
fn four_way_typed_forced_scalar() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            FORCE_SCALAR.with(|flag| flag.set(false));
        }
    }
    let _reset = Reset;
    FORCE_SCALAR.with(|flag| flag.set(true));
    assert!(!available());
    compare();
}

#[test]
fn four_way_typed_forced_rejection_and_continuation() {
    let labels = labels(3, SetupStreamRole::WitnessCommitment);
    let mut streams = new_four(&[11; 32], labels).unwrap();
    let mut scalar = labels.map(|label| TypedSetupStream::new(&[11; 32], label).unwrap());
    for (lane, (actual, expected)) in streams.iter_mut().zip(&mut scalar).enumerate() {
        // Force distinct rejection patterns in the initial block. This shifts
        // each reader across the 512-byte prepared prefix at a different draw.
        for word in actual.squeeze.block.chunks_exact_mut(8).take(lane + 1) {
            word.copy_from_slice(&BINARY_SETUP_P64.to_le_bytes());
        }
        expected.squeeze = actual.squeeze.clone();
    }
    let mut bytes = [[0; 512]; 4];
    read_four(&mut streams, bytes.each_mut().map(|lane| &mut lane[..]));
    for (lane, (actual, expected)) in streams.iter_mut().zip(&mut scalar).enumerate() {
        let mut values = [0; 64];
        fill_from_prefix(actual, &bytes[lane], &mut values).unwrap();
        for value in values {
            assert_eq!(value, u128::from(expected.sample_p64().unwrap()));
        }
        for _ in 0..100 {
            assert_eq!(actual.sample_p64().unwrap(), expected.sample_p64().unwrap());
        }
    }
}

#[test]
fn four_way_typed_rejection_budget_and_errors() {
    let label = labels(0, SetupStreamRole::ImageCommitment)[0];
    let mut stream = TypedSetupStream::new(&[0; 32], label).unwrap();
    let mut prefix = vec![255; 8 * 256];
    let mut value = [0];
    let error = fill_from_prefix(&mut stream, &prefix, &mut value).unwrap_err();
    assert!(matches!(error, AkitaError::InvalidSetup(message)
        if message == "p64 setup rejection budget exhausted"));
    prefix[8 * 255..].copy_from_slice(&7u64.to_le_bytes());
    fill_from_prefix(&mut stream, &prefix, &mut value).unwrap();
    assert_eq!(value, [7]);
    let mut two = [0; 2];
    let repeated = prefix.repeat(2);
    fill_from_prefix(&mut stream, &repeated, &mut two).unwrap();
    assert_eq!(two, [7, 7]);
    let mut attempts = 0;
    let error = sample_p64_from(|bytes| {
        attempts += 1;
        bytes.fill(255);
    })
    .unwrap_err();
    assert_eq!(attempts, 256);
    assert!(matches!(error, AkitaError::InvalidSetup(message)
        if message == "p64 setup rejection budget exhausted"));
    for invalid in [
        SetupStreamLabel { rank: 0, ..label },
        SetupStreamLabel { row: 2, ..label },
        SetupStreamLabel { degree: 0, ..label },
    ] {
        assert_eq!(
            TypedSetupStream::p64_four_entries(&[0; 32], [invalid; 4])
                .unwrap_err()
                .to_string(),
            TypedSetupStream::new(&[0; 32], invalid)
                .err()
                .unwrap()
                .to_string()
        );
    }
    assert!(TypedSetupStream::p64_four_entries(
        &[0; 32],
        [SetupStreamLabel {
            modulus: SetupStreamModulus::Prime128Offset275,
            ..label
        }; 4]
    )
    .is_err());
}
