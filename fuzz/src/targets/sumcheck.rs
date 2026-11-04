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
    prove_sumcheck, verify_sumcheck, InfallibleSumcheck, SumcheckInstanceProver,
    SumcheckInstanceVerifier, SumcheckProverChannel, SumcheckRole, SumcheckShape,
    SumcheckVerifierChannel,
};
use akita_transcript::{
    ext_challenge, new_prover_channel, new_verifier_channel, ChallengeCodec, ProtocolSiteId,
    ProverChannel as ProverState, VerifierChannel as VerifierState, SITE_FAMILY_SUMCHECK,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    prove_single, verify_single, ProveRounds, ProverRoundContext, SingleFormat, SingleInstance,
    SumcheckError, SumcheckSite, VerifierRoundContext,
};
use jolt_transcript::{
    Blake2b512, Channel, ProtocolId, ProverTranscript, SiteId, VerifierTranscript,
};

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("akita-fuzz/sumcheck");

pub fn run(data: &[u8]) {
    let mut reader = Reader::new(data);
    match reader.u8() % 3 {
        0 => case::<fp128::Field, fp128::Field>(&mut reader),
        1 => case::<fp32::Field, fp32::ExtensionField>(&mut reader),
        _ => case::<fp64::Field, fp64::ExtensionField>(&mut reader),
    }
}

fn site(site: SumcheckSite) -> SiteId {
    let mut bytes = [0u8; 32];
    bytes[..4].copy_from_slice(&site.invocation.to_le_bytes());
    bytes[4..8].copy_from_slice(&site.round.to_le_bytes());
    bytes[8..12].copy_from_slice(&(site.role as u32).to_le_bytes());
    SiteId(bytes)
}

struct ProverContext {
    state: ProverTranscript<Blake2b512>,
}

impl<F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec> SumcheckProverChannel<E>
    for ProverChannel<F, E>
{
    fn state_mut(&mut self) -> &mut ProverState {
        &mut self.state
    }

    fn site(&self, sumcheck_site: SumcheckSite) -> Option<SiteId> {
        Some(site(sumcheck_site))
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        ext_challenge::<E, _>(
            &mut self.state,
            site(invocation, round, SumcheckRole::Challenge),
        )
        .map_err(|_| AkitaError::InvalidProof)
    }
}

struct VerifierContext<'proof> {
    state: VerifierTranscript<'proof, Blake2b512>,
}

impl<'proof, F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec>
    SumcheckVerifierChannel<'proof, E> for VerifierChannel<'proof, F, E>
{
    fn state_mut(&mut self) -> &mut VerifierState<'proof> {
        &mut self.state
    }

    fn site(&self, sumcheck_site: SumcheckSite) -> Option<SiteId> {
        Some(site(sumcheck_site))
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<E, AkitaError> {
        ext_challenge::<E, _>(
            &mut self.state,
            site(invocation, round, SumcheckRole::Challenge),
        )
        .map_err(|_| AkitaError::InvalidProof)
    }
}

/// `Σ_x Π_i p_i(x)` over little-endian Boolean `x`.
struct ProductInstance<E> {
    tables: Vec<Vec<E>>,
    rounds: usize,
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

impl<E: Field> ProductInstance<E> {
    fn bind(&mut self, r: E) {
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

impl<E: Field> ProveRounds<E> for ProductInstance<E> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<E>,
        _round: usize,
        _previous_claim: E,
    ) -> Result<UnivariatePoly<E>, SumcheckError<E>> {
        if let Some(r) = bind {
            self.bind(r);
        }
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
        Ok(UnivariatePoly::new(sum))
    }

    fn finish_rounds(&mut self, bind: E) -> Result<(), SumcheckError<E>> {
        self.bind(bind);
        Ok(())
    }
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

fn case<F: Field + CanonicalEncoding, E: ExtField<F> + ChallengeCodec>(reader: &mut Reader<'_>) {
    // The negative-check choice leads the input so large tables cannot starve it.
    let negative = reader.u8() % 4;
    let tamper_offset = reader.u32() as usize;
    let tamper_mask = reader.u8().max(1);
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
    let instance = |claim: E| SingleInstance {
        input_claim: claim,
        rounds,
        degree,
        format: SingleFormat::Compressed,
        invocation,
    };

    let mut member = ProductInstance {
        tables: tables.clone(),
        rounds,
    };
    let mut prover = ProverChannel::<F, E> {
        state: new_prover_channel(&session),
        _marker: PhantomData,
    };
    let reduction =
        prove_single::<F, E, SumcheckError<E>, _>(&mut member, &instance(claim), &mut prover)
            .expect("honest sumcheck must prove");
    let challenges = reduction.challenges;
    assert_eq!(challenges.len(), rounds, "one challenge per round");
    let expected_final = tables
        .iter()
        .fold(E::one(), |acc, table| acc * evaluate(table, &challenges));
    assert_eq!(
        reduction.output_claim, expected_final,
        "prover final claim equals the product at the challenge point"
    );
    let proof = prover.state.finish();

    // `Err(())` is any rejection: transcript, replay, EOF, or the oracle.
    let verify = |proof: &[u8], claim: E| -> Result<Vec<E>, ()> {
        let mut channel = VerifierContext {
            state: VerifierTranscript::new(&PROTOCOL, &session, proof),
        };
        let mut channel = VerifierChannel::<F, E> {
            state: new_verifier_channel(&session, proof),
            _marker: PhantomData,
        };
        let replayed = verify_sumcheck::<F, E, _, _>(&verifier, &mut channel, shape, invocation)?;
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

    match negative {
        0 => {
            let mut delta = gen::ext_scalar::<F, E>(reader);
            if delta == E::zero() {
                delta = E::one();
            }
            assert!(
                verify(&proof, claim + delta).is_err(),
                "a false sumcheck claim must be rejected"
            );
            stats::count("sumcheck_false_claim");
        }
        1 if !proof.is_empty() => {
            let mut tampered = proof.clone();
            let offset = tamper_offset % tampered.len();
            tampered[offset] ^= tamper_mask;
            assert!(
                verify(&tampered, claim).is_err(),
                "a tampered sumcheck proof must be rejected"
            );
            stats::count("sumcheck_tampered");
        }
        2 if !proof.is_empty() => {
            let len = tamper_offset % proof.len();
            assert!(verify(&proof[..len], claim).is_err(), "truncated proof");
        }
        _ => {}
    }
}
