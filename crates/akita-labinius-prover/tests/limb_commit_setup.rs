#![cfg(feature = "labinius")]

mod common;

use akita_algebra::{
    binary::{BinaryField128, BinaryField192},
    MinusTrinomial,
};
use akita_error::{checked, AkitaError};
use akita_labinius_prover::{
    commit_binary_clear, commit_binary_clear_prepared, commit_binary_clear_small_modulus_prepared,
    PreparedCommitMatrix, PreparedLimbCommitMatrix,
};
use akita_labinius_verifier::{commitment::canonical_coefficient, AdmittedRootSetup};
use akita_params::sis::labinius::LabiniusRootProfile;
use akita_types::proof::AkitaSetupSeed;
use common::TestHost;
use jolt_field::Prime128OffsetA7F7 as F;
use rand::{rngs::StdRng, SeedableRng};

const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648P128Q28BoundedW46Delta16;
const Q0: u32 = 268_433_353;
type Setup = AdmittedRootSetup<F, 648, MinusTrinomial>;

fn admitted(cells: u32, fold: u32, seed: u8) -> Setup {
    Setup::derive(
        PROFILE,
        cells,
        fold,
        128,
        AkitaSetupSeed::shake256_paged_v1([seed; 32]),
    )
    .unwrap()
}

fn compare_sources<H: TestHost>() {
    let mut rng = StdRng::seed_from_u64(0x0028_0648);
    let ones = if size_of::<H::Source>() == size_of::<u128>() {
        u128::MAX
    } else {
        u128::from(u64::MAX)
    };
    let ones = H::Source::try_from(ones).ok().unwrap();
    // Includes both q28_clear geometries and four other admitted shapes.
    for cells in [4, 5] {
        for fold in [0, 1, 2] {
            let admitted = admitted(cells, fold, 0x28);
            let setup = admitted.setup();
            assert_eq!(setup.commitment_modulus().small_modulus(), Some(Q0));
            let limb = PreparedLimbCommitMatrix::prepare_for_setup(setup).unwrap();
            let p128 = PreparedCommitMatrix::prepare(setup).unwrap();
            let random = (0..setup.source_len())
                .map(|_| H::random_source(&mut rng))
                .collect::<Vec<_>>();
            for source in [
                vec![H::Source::default(); setup.source_len()],
                vec![ones; setup.source_len()],
                random,
            ] {
                let actual =
                    commit_binary_clear_small_modulus_prepared::<H, F>(&limb, setup, &source)
                        .unwrap();
                assert_eq!(
                    actual,
                    commit_binary_clear::<H, F, 648, MinusTrinomial>(setup, &source).unwrap(),
                    "reference mismatch: cells={cells}, fold={fold}"
                );
                assert_eq!(
                    actual,
                    commit_binary_clear_prepared::<H, F, 648, MinusTrinomial>(
                        &p128, setup, &source
                    )
                    .unwrap(),
                    "p128 mismatch: cells={cells}, fold={fold}"
                );
                assert!(actual
                    .images
                    .iter()
                    .flat_map(|image| image.coefficients())
                    .all(|&coefficient| canonical_coefficient(coefficient)
                        .is_ok_and(|value| value < u128::from(Q0))));
            }
        }
    }
}

#[test]
fn admitted_small_geometries_match_both_reference_paths_for_both_hosts() {
    compare_sources::<BinaryField128>();
    compare_sources::<BinaryField192>();
}

#[test]
fn shared_prime_setup_is_rejected() {
    let admitted = Setup::derive(
        LabiniusRootProfile::D648P128BoundedW46Delta16,
        4,
        1,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x28; 32]),
    )
    .unwrap();
    assert!(matches!(
        PreparedLimbCommitMatrix::prepare_for_setup(admitted.setup()),
        Err(AkitaError::InvalidSetup(_))
    ));
}

fn reject_inputs<H: TestHost>() {
    let first = admitted(4, 1, 0x28);
    let second = admitted(4, 1, 0x29);
    let setup = first.setup();
    let other = second.setup();
    assert_eq!(setup.n_a(), other.n_a());
    assert_eq!(setup.m(), other.m());
    assert_eq!(setup.columns(), other.columns());
    assert_ne!(setup.matrix_view_digest(), other.matrix_view_digest());
    let prepared = PreparedLimbCommitMatrix::prepare_for_setup(setup).unwrap();
    let source = vec![H::Source::default(); setup.source_len()];
    assert!(matches!(
        commit_binary_clear_small_modulus_prepared::<H, F>(&prepared, other, &source),
        Err(AkitaError::InvalidSetup(_))
    ));
    let coefficients: Vec<_> = setup
        .matrix()
        .iter()
        .flat_map(|element| element.coefficients())
        .map(|&coefficient| u32::try_from(canonical_coefficient(coefficient).unwrap()).unwrap())
        .collect();
    let raw =
        PreparedLimbCommitMatrix::prepare(Q0, 648, setup.n_a(), setup.m(), &coefficients).unwrap();
    assert!(matches!(
        commit_binary_clear_small_modulus_prepared::<H, F>(&raw, setup, &source),
        Err(AkitaError::InvalidSetup(_))
    ));
    for actual in [source.len() - 1, source.len() + 1] {
        let malformed = vec![H::Source::default(); actual];
        assert_eq!(
            commit_binary_clear_small_modulus_prepared::<H, F>(&prepared, setup, &malformed)
                .unwrap_err(),
            AkitaError::InvalidSize {
                expected: source.len(),
                actual,
            }
        );
        // Source validation also takes precedence over a mismatched matrix.
        assert_eq!(
            commit_binary_clear_small_modulus_prepared::<H, F>(&prepared, other, &malformed)
                .unwrap_err(),
            AkitaError::InvalidSize {
                expected: source.len(),
                actual,
            }
        );
    }
}

#[test]
fn mismatched_unbound_and_wrong_length_inputs_are_rejected_for_both_hosts() {
    reject_inputs::<BinaryField128>();
    reject_inputs::<BinaryField192>();
}

#[cfg(feature = "parallel")]
fn compare_threads<H: TestHost>() {
    let admitted = admitted(5, 2, 0x28);
    let setup = admitted.setup();
    let prepared = PreparedLimbCommitMatrix::prepare_for_setup(setup).unwrap();
    let mut rng = StdRng::seed_from_u64(0x0031_0648);
    let source = (0..setup.source_len())
        .map(|_| H::random_source(&mut rng))
        .collect::<Vec<_>>();
    let results = [1, 3].map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                commit_binary_clear_small_modulus_prepared::<H, F>(&prepared, setup, &source)
                    .unwrap()
            })
    });
    let [single, multiple] = results;
    assert_eq!(single, multiple);
}

#[cfg(feature = "parallel")]
#[test]
fn one_and_three_thread_pools_agree_for_both_hosts() {
    compare_threads::<BinaryField128>();
    compare_threads::<BinaryField192>();
}

fn print_load(when: &str) {
    if let Ok(output) = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
    {
        if output.status.success() {
            if let Some(load) = String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .find_map(|word| word.parse::<f64>().ok())
            {
                println!("load {when}: one_minute={load:.2}");
            }
        }
    }
}

#[test]
#[ignore = "full-size commitment timing; run in release with parallel enabled"]
fn full_size_small_modulus_commit() {
    use std::time::Instant;

    if cfg!(debug_assertions) {
        panic!("measurement requires --release");
    }
    print_load("before");
    let start = Instant::now();
    let admitted = admitted(22, 8, 0x28);
    println!("derive setup: {:.6}s", start.elapsed().as_secs_f64());
    let shape = admitted.shape();
    assert_eq!(shape.rank_a(), 3);
    assert_eq!(shape.ring_elements_per_column(), 4096);
    assert_eq!(shape.fold_width(), 256);
    let setup = admitted.setup();
    assert_eq!((setup.n_a(), setup.m(), setup.columns()), (3, 4096, 256));
    let committed_source_bits = checked::product([setup.source_len(), 128]).unwrap();
    let ring_coefficient_slots = checked::product([setup.m(), setup.columns(), 648]).unwrap();
    println!(
        "shape: rank={} width={} columns={} committed_source_bits={} ring_coefficient_slots={}",
        setup.n_a(),
        setup.m(),
        setup.columns(),
        committed_source_bits,
        ring_coefficient_slots
    );
    let mut rng = StdRng::seed_from_u64(0x0256_4096_0648);
    let source = (0..setup.source_len())
        .map(|_| BinaryField128::random_source(&mut rng))
        .collect::<Vec<_>>();
    let start = Instant::now();
    let limb = PreparedLimbCommitMatrix::prepare_for_setup(setup).unwrap();
    println!("prepare limb: {:.6}s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let p128 = PreparedCommitMatrix::prepare(setup).unwrap();
    println!("prepare p128: {:.6}s", start.elapsed().as_secs_f64());
    println!(
        "prepared_bytes: limb={} p128={}",
        limb.prepared_bytes(),
        p128.prepared_bytes()
    );
    let mut expected = None;
    for repetition in 1..=2 {
        let start = Instant::now();
        let reference = commit_binary_clear_prepared::<BinaryField128, F, 648, MinusTrinomial>(
            &p128, setup, &source,
        )
        .unwrap();
        println!(
            "commit p128 {repetition}: {:.6}s",
            start.elapsed().as_secs_f64()
        );
        if let Some(previous) = expected.as_ref() {
            assert_eq!(&reference, previous);
        }
        let start = Instant::now();
        let actual =
            commit_binary_clear_small_modulus_prepared::<BinaryField128, F>(&limb, setup, &source)
                .unwrap();
        println!(
            "commit limb {repetition}: {:.6}s",
            start.elapsed().as_secs_f64()
        );
        assert_eq!(actual, reference);
        expected = Some(reference);
    }
    print_load("after");
}
