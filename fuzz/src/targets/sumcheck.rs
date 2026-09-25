//! Standard sumcheck through the production native prover/verifier drivers.
//!
//! The instance is a product of `k` multilinear tables, so each round
//! polynomial is computed exactly as a product of linear factors, and the
//! verifier oracle is the product of independent multilinear evaluations at
//! the Fiat-Shamir point. This exercises round compression, native encoding,
//! challenge derivation, and replay, not the Akita-specific relation kernels
//! (those are reached through the end-to-end targets).

use crate::input::Reader;
use crate::{gen, stats};
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_error::AkitaError;
use akita_sumcheck::{
    prove_sumcheck_native, verify_sumcheck_native, InfallibleSumcheck, NativeSumcheckProverChannel,
    NativeSumcheckRole, NativeSumcheckShape, NativeSumcheckVerifierChannel, SumcheckInstanceProver,
    SumcheckInstanceVerifier,
};
use akita_transcript::{
    native_prover_ext_challenge, native_verifier_ext_challenge, new_native_prover,
    new_native_verifier, NativeProverState, NativeVerifierState, ProtocolSiteId,
    SITE_FAMILY_SUMCHECK,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};
use jolt_poly::UnivariatePoly;
use std::marker::PhantomData;

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    match reader.u8() % 3 {
        0 => case::<fp128::Field, fp128::Field>(&mut reader),
        1 => case::<fp32::Field, fp32::ExtensionField>(&mut reader),
        _ => case::<fp64::Field, fp64::ExtensionField>(&mut reader),
    }
}

fn site(invocation: u32, round: u32, role: NativeSumcheckRole) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_SUMCHECK,
        invocation,
        round,
        detail: role as u32,
        ..ProtocolSiteId::default()
    }
}

struct ProverChannel<F, E> {
    state: NativeProverState,
    _marker: PhantomData<(F, E)>,
}

impl<F: Field + CanonicalEncoding, E: ExtField<F>> NativeSumcheckProverChannel<E>
    for ProverChannel<F, E>
{
    fn state_mut(&mut self) -> &mut NativeProverState {
        &mut self.state
    }

    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: NativeSumcheckRole,
    ) -> ProtocolSiteId {
        site(invocation, round, role)
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        native_prover_ext_challenge::<F, E>(
            &mut self.state,
            site(invocation, round, NativeSumcheckRole::Challenge),
        )
        .map_err(|_| AkitaError::InvalidProof)
    }
}

struct VerifierChannel<'proof, F, E> {
    state: NativeVerifierState<'proof>,
    _marker: PhantomData<(F, E)>,
}

impl<'proof, F: Field + CanonicalEncoding, E: ExtField<F>> NativeSumcheckVerifierChannel<'proof, E>
    for VerifierChannel<'proof, F, E>
{
    fn state_mut(&mut self) -> &mut NativeVerifierState<'proof> {
        &mut self.state
    }

    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: NativeSumcheckRole,
    ) -> ProtocolSiteId {
        site(invocation, round, role)
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        native_verifier_ext_challenge::<F, E>(
            &mut self.state,
            site(invocation, round, NativeSumcheckRole::Challenge),
        )
        .map_err(|_| AkitaError::InvalidProof)
    }
}

/// `Σ_x Π_i p_i(x)` over little-endian Boolean `x`.
struct ProductInstance<E> {
    tables: Vec<Vec<E>>,
    rounds: usize,
    claim: E,
}

fn multiply<E: Field>(lhs: &[E], rhs: &[E]) -> Vec<E> {
    let mut out = vec![E::zero(); lhs.len() + rhs.len() - 1];
    for (i, &a) in lhs.iter().enumerate() {
        for (j, &b) in rhs.iter().enumerate() {
            out[i + j] += a * b;
        }
    }
    out
}

impl<E: Field> SumcheckInstanceProver<E> for ProductInstance<E> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        self.tables.len()
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _claim: E) -> UnivariatePoly<E> {
        let half = self.tables[0].len() / 2;
        let mut sum = vec![E::zero(); self.tables.len() + 1];
        for pair in 0..half {
            let mut product = vec![E::one()];
            for table in &self.tables {
                let low = table[2 * pair];
                product = multiply(&product, &[low, table[2 * pair + 1] - low]);
            }
            for (acc, coefficient) in sum.iter_mut().zip(product) {
                *acc += coefficient;
            }
        }
        UnivariatePoly::new(sum)
    }

    fn ingest_challenge(&mut self, _round: usize, r: E) {
        for table in &mut self.tables {
            let half = table.len() / 2;
            for index in 0..half {
                let low = table[2 * index];
                table[index] = low + r * (table[2 * index + 1] - low);
            }
            table.truncate(half);
        }
    }
}

struct ProductVerifier<E> {
    tables: Vec<Vec<E>>,
    rounds: usize,
    claim: E,
}

fn evaluate<E: Field>(table: &[E], point: &[E]) -> E {
    let mut layer = table.to_vec();
    for &x in point {
        let half = layer.len() / 2;
        for index in 0..half {
            layer[index] = layer[2 * index] + x * (layer[2 * index + 1] - layer[2 * index]);
        }
        layer.truncate(half);
    }
    layer[0]
}

impl<E: Field> SumcheckInstanceVerifier<E> for ProductVerifier<E> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        self.tables.len()
    }

    fn input_claim(&self) -> E {
        self.claim
    }

    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, AkitaError> {
        Ok(self
            .tables
            .iter()
            .fold(E::one(), |acc, table| acc * evaluate(table, challenges)))
    }
}

fn case<F: Field + CanonicalEncoding, E: ExtField<F>>(reader: &mut Reader<'_>) {
    let rounds = usize::from(reader.u8() % 11);
    let degree = 1 + usize::from(reader.u8() % 4);
    let invocation = reader.u32();
    let session_len = usize::from(reader.u8() % 16);
    let session = reader.take(session_len).to_vec();
    let size = 1usize << rounds;
    let tables: Vec<Vec<E>> = (0..degree)
        .map(|_| {
            let coordinates: Vec<Vec<F>> = (0..E::DEGREE)
                .map(|_| gen::table(reader, size, gen::Domain::Full))
                .collect();
            (0..size)
                .map(|index| {
                    E::from_base_slice(&coordinates.iter().map(|c| c[index]).collect::<Vec<_>>())
                })
                .collect()
        })
        .collect();
    let claim = (0..size).fold(E::zero(), |acc, index| {
        acc + tables
            .iter()
            .fold(E::one(), |product, table| product * table[index])
    });
    let shape = NativeSumcheckShape::new(rounds, degree).expect("valid shape");

    let mut instance = ProductInstance {
        tables: tables.clone(),
        rounds,
        claim,
    };
    let mut prover = ProverChannel::<F, E> {
        state: new_native_prover(&session, b"akita-fuzz/sumcheck").expect("bounded session"),
        _marker: PhantomData,
    };
    let (challenges, final_claim) = prove_sumcheck_native::<F, E, _, _>(
        &mut InfallibleSumcheck(&mut instance),
        &mut prover,
        shape,
        invocation,
    )
    .expect("honest sumcheck must prove");
    assert_eq!(challenges.len(), rounds, "one challenge per round");
    let expected_final = tables
        .iter()
        .fold(E::one(), |acc, table| acc * evaluate(table, &challenges));
    assert_eq!(
        final_claim, expected_final,
        "prover final claim equals the product at the challenge point"
    );
    let proof = prover.state.narg_string().to_vec();

    let verify = |proof: &[u8], claim: E| -> Result<Vec<E>, AkitaError> {
        let verifier = ProductVerifier {
            tables: tables.clone(),
            rounds,
            claim,
        };
        let mut channel = VerifierChannel::<F, E> {
            state: new_native_verifier(&session, b"akita-fuzz/sumcheck", proof)
                .expect("bounded session"),
            _marker: PhantomData,
        };
        let replayed =
            verify_sumcheck_native::<F, E, _, _>(&verifier, &mut channel, shape, invocation)?;
        channel
            .state
            .check_eof()
            .map_err(|_| AkitaError::InvalidProof)?;
        Ok(replayed)
    };
    assert_eq!(
        verify(&proof, claim).expect("honest sumcheck must verify"),
        challenges,
        "replayed challenges"
    );
    stats::count("sumcheck_honest");

    match reader.u8() % 4 {
        0 => {
            let mut delta = gen::ext_scalar::<F, E>(reader);
            if delta == E::zero() {
                delta = E::one();
            }
            assert!(
                matches!(verify(&proof, claim + delta), Err(AkitaError::InvalidProof)),
                "a false sumcheck claim must be rejected"
            );
            stats::count("sumcheck_false_claim");
        }
        1 if !proof.is_empty() => {
            let mut tampered = proof.clone();
            let offset = reader.u32() as usize % tampered.len();
            tampered[offset] ^= reader.u8().max(1);
            assert!(
                matches!(verify(&tampered, claim), Err(AkitaError::InvalidProof)),
                "a tampered sumcheck proof must be rejected"
            );
            stats::count("sumcheck_tampered");
        }
        2 if !proof.is_empty() => {
            let len = reader.u32() as usize % proof.len();
            assert!(
                matches!(verify(&proof[..len], claim), Err(AkitaError::InvalidProof)),
                "truncated proof"
            );
        }
        _ => {}
    }
}
