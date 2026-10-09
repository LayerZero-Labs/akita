#![cfg(feature = "labinius")]

mod root_reduction_support;
mod serial_support;

use akita_algebra::{binary::BinaryField162 as B, MinusTrinomial};
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_labinius_prover::lowered::{encode_witness, flatten_image, parity_quotient_and_carry};
use akita_labinius_verifier::{
    frontend::BinaryEvaluationClaim, lowered::LoweredRootLayout, profile::BinaryClearSetup,
    source::equality_weights,
};
use akita_params::sis::labinius::{LabiniusCoefficientPrime, LabiniusRingDegree};
use rand::{rngs::StdRng, Rng, SeedableRng};
use root_reduction_support::{admitted, common::FixedDraw, Case, BASES, F, H};

fn random_binary(rng: &mut StdRng) -> B {
    B::from_words([rng.gen(), rng.gen(), rng.gen::<u64>() & ((1 << 34) - 1)]).unwrap()
}

fn scalar_parity(values: &[i64; 162]) -> B {
    let mut words = [0u64; 3];
    for (bit, &value) in values.iter().enumerate() {
        words[bit / 64] |= ((value & 1) as u64) << (bit % 64);
    }
    B::from_words(words).unwrap()
}

fn maximal_challenges(setup: &BinaryClearSetup<F, 648, MinusTrinomial>) -> Vec<BinaryChallenge> {
    let mut sampler = BinaryChallengeSampler::new(setup.profile().clone());
    let challenges = sampler
        .sample_challenges(&mut FixedDraw, b"serial-phase-tests", setup.columns() * 32)
        .unwrap()
        .into_iter()
        .filter(|challenge| challenge.weight() == setup.profile().weight_cap())
        .take(setup.columns())
        .collect::<Vec<_>>();
    assert_eq!(challenges.len(), setup.columns());
    challenges
}

fn matching_expansion(
    claim: &BinaryEvaluationClaim,
    setup: &BinaryClearSetup<F, 648, MinusTrinomial>,
    challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
) -> Vec<B> {
    let residual = equality_weights(&claim.point[..setup.row_vars()])
        .unwrap()
        .iter()
        .zip(response)
        .fold(B::ZERO, |sum, (&weight, values)| {
            sum + weight * scalar_parity(values)
        });
    let mut words = [0u64; 3];
    for term in challenges[0].terms() {
        let bit = usize::from(term.position);
        words[bit / 64] |= 1 << (bit % 64);
    }
    let mut u = vec![B::ZERO; setup.columns()];
    u[0] = residual * B::from_words(words).unwrap().inverse().unwrap();
    u
}

fn compare_witness(layout: &LoweredRootLayout, response: &[[i64; 162]]) {
    assert_eq!(
        encode_witness(layout, response),
        serial_support::encode_witness(layout, response)
    );
}

fn compare_parity(
    setup: &BinaryClearSetup<F, 648, MinusTrinomial>,
    claim: &BinaryEvaluationClaim,
    challenges: &[BinaryChallenge],
    response: &[[i64; 162]],
) {
    let u = matching_expansion(claim, setup, challenges, response);
    let expected =
        serial_support::parity_quotient_and_carry(setup, claim, &u, challenges, response).unwrap();
    assert_eq!(
        parity_quotient_and_carry(setup, claim, &u, challenges, response).unwrap(),
        expected
    );
}

fn differential_outputs() {
    let mut rng = StdRng::seed_from_u64(0x0531_7162);
    for fold in [0, 1] {
        let admitted = admitted(fold, 0x6a);
        let setup = admitted.setup();
        let challenges = maximal_challenges(setup);
        let claim = BinaryEvaluationClaim {
            point: (0..setup.num_vars())
                .map(|_| random_binary(&mut rng))
                .collect(),
            value: B::ZERO,
        };
        let mut responses = vec![
            vec![[setup.lower(); 162]; setup.scalar_rows()],
            vec![[setup.upper(); 162]; setup.scalar_rows()],
        ];
        for _ in 0..12 {
            responses.push(
                (0..setup.scalar_rows())
                    .map(|_| std::array::from_fn(|_| rng.gen_range(setup.lower()..=setup.upper())))
                    .collect(),
            );
        }
        for response in &responses {
            compare_parity(setup, &claim, &challenges, response);
            for base in BASES {
                let layout = LoweredRootLayout::new(setup, admitted.shape(), base).unwrap();
                assert!(encode_witness(&layout, response).is_ok());
                compare_witness(&layout, response);
            }
        }
    }
}

#[test]
fn scalar_phases_match_frozen_arithmetic_for_random_and_extremal_inputs() {
    differential_outputs();
}

#[cfg(feature = "parallel")]
#[test]
fn scalar_phases_match_in_one_and_three_thread_pools() {
    for threads in [1, 3] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(differential_outputs);
    }
}

#[test]
fn rejected_witness_coefficients_and_odd_parity_match_frozen_errors() {
    let admitted = admitted(1, 0x6a);
    let setup = admitted.setup();
    let challenges = maximal_challenges(setup);
    let claim = BinaryEvaluationClaim {
        point: vec![B::ZERO; setup.num_vars()],
        value: B::ZERO,
    };
    let u = vec![B::ZERO; setup.columns()];
    let mut below = vec![[0; 162]; setup.scalar_rows()];
    below[0][0] = setup.lower() - 1;
    let mut above = vec![[0; 162]; setup.scalar_rows()];
    above[setup.scalar_rows() - 1][161] = setup.upper() + 1;
    for response in [below, above, vec![[0; 162]; setup.scalar_rows() - 1]] {
        for base in BASES {
            let layout = LoweredRootLayout::new(setup, admitted.shape(), base).unwrap();
            compare_witness(&layout, &response);
            assert_eq!(
                encode_witness(&layout, &response),
                Err(AkitaError::InvalidProof)
            );
        }
        assert_eq!(
            parity_quotient_and_carry(setup, &claim, &u, &challenges, &response),
            serial_support::parity_quotient_and_carry(setup, &claim, &u, &challenges, &response)
        );
        assert_eq!(
            parity_quotient_and_carry(setup, &claim, &u, &challenges, &response),
            Err(AkitaError::InvalidProof)
        );
    }
    let mut odd = vec![[0; 162]; setup.scalar_rows()];
    odd[0][0] = 1;
    assert_eq!(
        parity_quotient_and_carry(setup, &claim, &u, &challenges, &odd),
        serial_support::parity_quotient_and_carry(setup, &claim, &u, &challenges, &odd)
    );
    assert_eq!(
        parity_quotient_and_carry(setup, &claim, &u, &challenges, &odd),
        Err(AkitaError::InvalidProof)
    );
}

#[test]
fn wide_interval_uses_checked_parity_arithmetic_with_exact_outputs() {
    let admitted = admitted(0, 0x6a);
    let original = admitted.setup();
    let bound = 1i64 << 59;
    let setup = BinaryClearSetup::new(
        original.matrix().to_vec(),
        original.n_a(),
        original.m(),
        original.columns(),
        -bound,
        bound,
        original.lambda_fold(),
        original.profile().clone(),
        LabiniusCoefficientPrime::P128OffsetA7F7,
        LabiniusRingDegree::D648,
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(0x0912_2162);
    let claim = BinaryEvaluationClaim {
        point: (0..setup.num_vars())
            .map(|_| random_binary(&mut rng))
            .collect(),
        value: B::ZERO,
    };
    let challenges = maximal_challenges(&setup);
    for response in [
        vec![[-bound; 162]; setup.scalar_rows()],
        vec![[bound; 162]; setup.scalar_rows()],
        (0..setup.scalar_rows())
            .map(|_| std::array::from_fn(|_| rng.gen_range(-bound..=bound)))
            .collect(),
    ] {
        compare_parity(&setup, &claim, &challenges, &response);
    }
}

#[test]
fn image_blocks_preserve_flattened_values_and_zero_padding() {
    for base in BASES {
        let case = Case::<H>::new(base, 1);
        assert_eq!(
            flatten_image(&case.layout, &case.commitment),
            serial_support::flatten_image(&case.layout, &case.commitment)
        );
    }
}
