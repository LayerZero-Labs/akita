#![cfg(feature = "labinius")]

use akita_algebra::{Prime64Offset23703, SmoothFftField};
use akita_labinius_prover::root_sumcheck::{
    prove_combined_rounds, prove_product_rounds, CombinedRootSumcheck, ProductSumcheck,
};
use akita_labinius_verifier::{
    channel::{self, ClearChannel, RootSumcheckProverChannel, RootSumcheckVerifierChannel},
    root_sumcheck::{
        bind_root_sumcheck_instance, verify_combined_rounds, verify_product_rounds,
        RootSumcheckInstance,
    },
};
use akita_params::sis::labinius::LabiniusDigitBase;
use akita_sumcheck::SumcheckInstanceProver;
use jolt_field::{Field, One, Prime128OffsetA7F7, Ring, Zero};

const BASES: [LabiniusDigitBase; 3] = [
    LabiniusDigitBase::Bits1,
    LabiniusDigitBase::Bits2,
    LabiniusDigitBase::Bits4,
];

#[test]
fn every_header_field_changes_the_transcript_even_without_rounds() {
    let draw = |kind, invocation, dimension| {
        let mut prover = channel::new_prover().unwrap();
        bind_root_sumcheck_instance(&mut prover, kind, invocation, dimension).unwrap();
        let challenge = prover.challenge_block().unwrap();
        let proof = channel::finish_prover(prover);
        let mut verifier = channel::new_verifier(&proof).unwrap();
        bind_root_sumcheck_instance(&mut verifier, kind, invocation, dimension).unwrap();
        assert_eq!(challenge, verifier.challenge_block().unwrap());
        channel::finish_verifier(verifier).unwrap();
        challenge
    };
    let baseline = draw(RootSumcheckInstance::Combined(BASES[0]), 0, 0);
    for (kind, invocation, dimension) in [
        (RootSumcheckInstance::Product, 0, 0),
        (RootSumcheckInstance::Combined(BASES[0]), 1, 0),
        (RootSumcheckInstance::Combined(BASES[0]), 0, 1),
        (RootSumcheckInstance::Combined(BASES[1]), 0, 0),
        (RootSumcheckInstance::Combined(BASES[2]), 0, 0),
    ] {
        assert_ne!(baseline, draw(kind, invocation, dimension));
    }
}

#[test]
fn instance_replay_binds_invocation_base_and_common_prefix_dimension() {
    type F = Prime128OffsetA7F7;
    let mut state = channel::new_prover().unwrap();
    let mut combined = CombinedRootSumcheck::new(
        BASES[0],
        &[0; 4],
        vec![F::zero(); 4],
        &[F::from_u64(3), F::from_u64(5)],
        F::from_u64(7),
        F::zero(),
    )
    .unwrap();
    let (rho, claim) = prove_combined_rounds(
        &mut combined,
        &mut RootSumcheckProverChannel::new(&mut state),
        11,
    )
    .unwrap();
    let proof = channel::finish_prover(state);
    for (invocation, dimension, base) in [(12, 2, BASES[0]), (11, 1, BASES[0]), (11, 2, BASES[1])] {
        let mut state = channel::new_verifier(&proof).unwrap();
        let replay = verify_combined_rounds(
            &mut RootSumcheckVerifierChannel::new(&mut state),
            invocation,
            dimension,
            base,
            F::from_u64(7),
            F::zero(),
        );
        if let Ok(result) = replay {
            assert_ne!(result.challenges.first(), rho.first());
        }
    }
    let mut state = channel::new_verifier(&proof).unwrap();
    let result = verify_combined_rounds(
        &mut RootSumcheckVerifierChannel::new(&mut state),
        11,
        2,
        BASES[0],
        F::from_u64(7),
        F::zero(),
    )
    .unwrap();
    assert_eq!((result.challenges, result.output_claim), (rho, claim));
    channel::finish_verifier(state).unwrap();

    let mut state = channel::new_prover().unwrap();
    let mut product =
        ProductSumcheck::new(vec![F::zero(); 4], vec![F::one(); 4], F::zero()).unwrap();
    let (rho, _) = prove_product_rounds(
        &mut product,
        &mut RootSumcheckProverChannel::new(&mut state),
        11,
    )
    .unwrap();
    let proof = channel::finish_prover(state);
    for (invocation, dimension) in [(12, 2), (11, 1)] {
        let mut state = channel::new_verifier(&proof).unwrap();
        let result = verify_product_rounds(
            &mut RootSumcheckVerifierChannel::new(&mut state),
            invocation,
            dimension,
            F::zero(),
        )
        .unwrap();
        assert_ne!(result.challenges.first(), rho.first());
    }
}

// Independent Boolean-basis expansion, rather than in-place table folding.
fn boolean_expansion<F: Field>(table: &[F], point: &[F]) -> F {
    table
        .iter()
        .enumerate()
        .fold(F::zero(), |sum, (index, &value)| {
            let basis = point.iter().enumerate().fold(F::one(), |basis, (bit, &r)| {
                basis
                    * if index >> bit & 1 == 0 {
                        F::one() - r
                    } else {
                        r
                    }
            });
            sum + basis * value
        })
}

fn exhaustive_rounds<F: SmoothFftField>() {
    for base in BASES {
        let alphabet_size = 1u8 << base.bits();
        for left in 0..alphabet_size {
            for right in 0..alphabet_size {
                let digits = [left, right];
                let w = [F::from_u64(u64::from(left)), F::from_u64(u64::from(right))];
                let kw = [F::from_u64(7), F::from_u64(11)];
                let tau = [F::from_u64(13)];
                let beta = F::from_u64(19);
                let s = w[0] * kw[0] + w[1] * kw[1];
                let mut instance =
                    CombinedRootSumcheck::new(base, &digits, kw.to_vec(), &tau, beta, s).unwrap();
                let poly = instance.compute_round_univariate(0, beta * s);
                for node in 0..=u64::from(alphabet_size) + 1 {
                    let point = [F::from_u64(node)];
                    let w_eval = boolean_expansion(&w, &point);
                    let kw_eval = boolean_expansion(&kw, &point);
                    let eq_eval = boolean_expansion(&[F::one() - tau[0], tau[0]], &point);
                    let alphabet = (0..alphabet_size).fold(F::one(), |product, digit| {
                        product * (w_eval - F::from_u64(u64::from(digit)))
                    });
                    assert_eq!(
                        poly.evaluate(point[0]),
                        eq_eval * alphabet + beta * w_eval * kw_eval
                    );
                }
            }
        }
    }
}

#[test]
fn every_one_variable_digit_table_matches_every_interpolation_node() {
    exhaustive_rounds::<Prime64Offset23703>();
    exhaustive_rounds::<Prime128OffsetA7F7>();
}

fn controlled_cancellation<F: SmoothFftField>() {
    for base in BASES {
        // W, tau, weights and s are fixed before enumerating beta.
        let w = [F::zero(), F::from_u64(255)];
        let kw = [F::from_u64(2), F::from_u64(3)];
        let tau = F::from_u64(5);
        let linear = w[0] * kw[0] + w[1] * kw[1];
        let s = linear - F::one();
        let p = (0..1u64 << base.bits()).fold(F::one(), |v, digit| v * (w[1] - F::from_u64(digit)));
        let z = tau * p;
        assert_ne!(z, F::zero());
        let cancelling_beta = -z;
        for beta in [F::zero(), cancelling_beta, cancelling_beta + F::one()] {
            assert_eq!(
                z + beta * (linear - s) == F::zero(),
                beta == cancelling_beta
            );
            let honest_s = linear;
            assert_ne!(z + beta * (linear - honest_s), F::zero());
        }
        // Changing s after seeing beta invalidates the conditional argument.
        let beta = F::from_u64(7);
        let adaptive_s = linear + z * beta.inverse().unwrap();
        assert_eq!(z + beta * (linear - adaptive_s), F::zero());
    }
}

#[test]
fn fixed_claim_has_at_most_one_cancelling_beta() {
    controlled_cancellation::<Prime64Offset23703>();
    controlled_cancellation::<Prime128OffsetA7F7>();
}
