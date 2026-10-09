#![cfg(feature = "labinius")]

mod combined_support;
mod quotient_support;
mod root_reduction_support;

use akita_algebra::binary::BinaryField192;
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_labinius_prover::{
    lowered::{encode_witness, flatten_image, parity_quotient_and_carry},
    prove_root_reduction,
    root_sumcheck::{prove_product_rounds, ProductSumcheck},
    PreparedRootMatrices, TransparentRootProverOracle,
};
use akita_labinius_verifier::{
    channel::{
        finish_prover, new_root_prover, ClearChannel, RootChallengeChannel, RootFieldSite,
        RootSumcheckProverChannel,
    },
    codec::{exchange_binary, exchange_field},
    endpoint::{fold_integer, left_expansion, verify_left_expansion},
    frontend::prove_frontend,
    lowered::{
        image_weights_dense, witness_weights_dense, LoweredChallenges, LoweredPublic,
        LoweredRootLayout,
    },
    root::{bind_root_statement, exchange_root_auxiliary, RootEvaluationClaims, RootProverOracle},
};
use akita_sumcheck::{SumcheckProverChannel, SumcheckRole};
use combined_support::{prove_combined_rounds, CombinedRootSumcheck};
use jolt_field::Zero;
use quotient_support::a_relation_quotients;
use root_reduction_support::{admitted, common::TestHost, Case, BASES, F, H};
use std::cell::Cell;

/// The reduction composed from the schoolbook fold, the long-division
/// quotients and the dense combined sumcheck, with no transform cache.
fn reference<T: TestHost>(
    case: &Case<T>,
) -> Result<(Vec<u8>, RootEvaluationClaims<F>), AkitaError> {
    let setup = case.admitted.setup();
    if case.source.len() != setup.source_len() {
        return Err(AkitaError::InvalidSize {
            expected: setup.source_len(),
            actual: case.source.len(),
        });
    }
    let mut oracle = TransparentRootProverOracle::new(&case.image);
    let mut state = new_root_prover()?;
    let mut ch = RootSumcheckProverChannel::new(&mut state);
    let layout = bind_root_statement(
        &case.admitted,
        case.base,
        &case.point,
        case.value,
        &mut ch,
        |layout, ch| oracle.bind_image(layout, ch),
    )?;
    let binary = prove_frontend::<T, _>(&case.source, &case.point, case.value, &mut ch)?;
    let mut u = left_expansion::<T>(
        &case.source,
        &binary.point,
        setup.scalar_rows(),
        setup.columns(),
    )?;
    for element in &mut u {
        exchange_binary(&mut ch, element)?;
    }
    verify_left_expansion(setup, &binary, &u)?;
    let fold = ch.fold_challenges(
        &mut BinaryChallengeSampler::new(setup.profile().clone()),
        b"akita/labinius/root-fold/v1",
        setup.columns(),
    )?;
    let response = fold_integer::<T>(
        &case.source,
        setup.scalar_rows(),
        setup.columns(),
        &fold,
        setup.profile(),
    )?;
    if response
        .iter()
        .flatten()
        .any(|&coefficient| coefficient < setup.lower() || coefficient > setup.upper())
    {
        return Err(AkitaError::InvalidInput(
            "root response leaves the admitted interval".into(),
        ));
    }
    let digits = encode_witness(&layout, &response)?;
    oracle.commit_response(&layout, &digits, &mut ch)?;
    let mut qa = a_relation_quotients(setup, &case.commitment, &fold, &response)?;
    let (mut q, mut k) = parity_quotient_and_carry(setup, &binary, &u, &fold, &response)?;
    exchange_root_auxiliary(&layout, &mut ch, &mut qa, &mut q, &mut k)?;
    let challenges = LoweredChallenges {
        alpha: ch.field_challenge(RootFieldSite::Alpha)?,
        xi: ch.field_challenge(RootFieldSite::Xi)?,
        gamma: ch.field_challenge(RootFieldSite::Gamma)?,
    };
    let public = LoweredPublic::new(&layout, setup, &binary, &u, &fold, &qa, &q, &k, challenges)?;
    let image = flatten_image(&layout, &case.commitment)?;
    let ky = image_weights_dense(&layout, &public)?;
    let mut y_y = image
        .iter()
        .zip(&ky)
        .fold(F::zero(), |sum, (&y, &weight)| sum + y * weight);
    exchange_field(&mut ch, &mut y_y)?;
    let s = public.c_pub() - y_y;
    let mut tau = Vec::new();
    tau.try_reserve_exact(layout.witness_log_len())
        .map_err(|_| AkitaError::InvalidInput("root equality-point allocation failed".into()))?;
    for coordinate in 0..layout.witness_log_len() {
        let site = u32::try_from(coordinate).map_err(|_| AkitaError::InvalidProof)?;
        tau.push(ch.field_challenge(RootFieldSite::Tau(site))?);
    }
    let beta = ch.field_challenge(RootFieldSite::Beta)?;
    let kw = witness_weights_dense(&layout, &public)?;
    let mut combined = CombinedRootSumcheck::new(case.base, &digits, kw, &tau, beta, s)?;
    let (response_point, _) = prove_combined_rounds(&mut combined, &mut ch, 0)?;
    let (mut response_value, _) = combined
        .final_evaluations()
        .ok_or(AkitaError::InvalidProof)?;
    drop(combined);
    exchange_field(&mut ch, &mut response_value)?;
    let mut product = ProductSumcheck::new(image, ky, y_y)?;
    let (image_point, _) = prove_product_rounds(&mut product, &mut ch, 1)?;
    let (mut image_value, _) = product
        .final_evaluations()
        .ok_or(AkitaError::InvalidProof)?;
    exchange_field(&mut ch, &mut image_value)?;
    let claims = RootEvaluationClaims {
        response_point,
        response_value,
        image_point,
        image_value,
    };
    oracle.discharge(&claims, &mut ch)?;
    Ok((finish_prover(state), claims))
}

fn kernels_match_reference<T: TestHost>() {
    for base in BASES {
        for fold in [0, 1] {
            let case = Case::<T>::new(base, fold);
            let (expected, expected_claims) = reference(&case).unwrap();
            let (proof, claims, _) = case.prove();
            assert_eq!(proof, expected, "{base:?} fold {fold}");
            assert_eq!(claims.response_point, expected_claims.response_point);
            assert_eq!(claims.response_value, expected_claims.response_value);
            assert_eq!(claims.image_point, expected_claims.image_point);
            assert_eq!(claims.image_value, expected_claims.image_value);
            case.verify(&proof).unwrap();
        }
    }
}

#[test]
fn kernel_reduction_emits_the_reference_proof_bytes_for_both_hosts() {
    kernels_match_reference::<H>();
    kernels_match_reference::<BinaryField192>();
}

#[test]
fn transform_cache_of_another_matrix_is_rejected_before_any_proof_byte() {
    for (fold, seed) in [(0, 0x32), (1, 0x32)] {
        let case = Case::<H>::new(BASES[1], 0);
        let foreign = admitted(fold, seed);
        let prepared = PreparedRootMatrices::prepare(foreign.setup()).unwrap();
        let mut oracle = CountingOracle {
            inner: TransparentRootProverOracle::new(&case.image),
            calls: 0,
        };
        let mut state = new_root_prover().unwrap();
        let mut channel = CountingChannel {
            inner: RootSumcheckProverChannel::new(&mut state),
            operations: Cell::new(0),
        };
        let result = prove_root_reduction(
            &case.admitted,
            &prepared,
            case.base,
            &case.source,
            &case.commitment,
            &case.point,
            case.value,
            &mut oracle,
            &mut channel,
        );
        assert_eq!(channel.operations.get(), 0);
        assert_eq!(oracle.calls, 0);
        assert!(matches!(result, Err(AkitaError::InvalidSetup(_))));
        assert!(oracle.inner.response().is_empty());
        assert!(finish_prover(state).is_empty());
    }
}

struct CountingChannel<'state> {
    inner: RootSumcheckProverChannel<'state>,
    operations: Cell<usize>,
}

impl ClearChannel for CountingChannel<'_> {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.operations.set(self.operations.get() + 1);
        self.inner.public(bytes)
    }

    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        self.operations.set(self.operations.get() + 1);
        self.inner.message(bytes)
    }

    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        self.operations.set(self.operations.get() + 1);
        self.inner.challenge_block()
    }

    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        self.operations.set(self.operations.get() + 1);
        self.inner.fold_challenges(sampler, label, count)
    }
}

impl RootChallengeChannel<F> for CountingChannel<'_> {
    fn field_challenge(&mut self, site: RootFieldSite) -> Result<F, AkitaError> {
        self.operations.set(self.operations.get() + 1);
        self.inner.field_challenge(site)
    }
}

impl SumcheckProverChannel<F> for CountingChannel<'_> {
    fn state_mut(&mut self) -> &mut akita_transcript::ProverChannel {
        self.operations.set(self.operations.get() + 1);
        <RootSumcheckProverChannel<'_> as SumcheckProverChannel<F>>::state_mut(&mut self.inner)
    }

    fn sumcheck_site(
        &self,
        invocation: u32,
        round: u32,
        role: SumcheckRole,
    ) -> akita_transcript::ProtocolSiteId {
        self.operations.set(self.operations.get() + 1);
        <RootSumcheckProverChannel<'_> as SumcheckProverChannel<F>>::sumcheck_site(
            &self.inner,
            invocation,
            round,
            role,
        )
    }

    fn round_challenge(&mut self, invocation: u32, round: u32) -> Result<F, AkitaError> {
        self.operations.set(self.operations.get() + 1);
        self.inner.round_challenge(invocation, round)
    }
}

struct CountingOracle<'image> {
    inner: TransparentRootProverOracle<'image, F>,
    calls: usize,
}

impl RootProverOracle<F> for CountingOracle<'_> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.calls += 1;
        self.inner.bind_image(layout, channel)
    }

    fn commit_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        digits: &[u8],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.calls += 1;
        self.inner.commit_response(layout, digits, channel)
    }

    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<F>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.calls += 1;
        self.inner.discharge(claims, channel)
    }
}
