//! Shared end-to-end fixtures and forwarding transcript instrumentation.

use akita_algebra::binary::{field_switch::SwitchField, BinaryField128};
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_labinius_pcs::{
    family::{
        tables::{shipped_catalog, ShippedCatalogs},
        FieldFamily,
    },
    shipped::{SupportedGeometry, SUPPORTED_GEOMETRIES},
    Committed, Prover, Verifier,
};
use akita_labinius_prover::prove_root_reduction_bytes;
use akita_labinius_verifier::{
    channel::ClearChannel,
    lowered::LoweredRootLayout,
    root::{
        verify_root_reduction_bytes, RootEvaluationClaims, RootProverOracle, RootVerifierOracle,
    },
    PrimeClaim, RootOpeningMode, RootStatement,
};
use akita_serialization::AkitaSerialize;
use jolt_field::{ExtField, Field, One, Ring, Zero};
use std::{ops::Range, path::PathBuf, time::Instant};

/// The host field of every message in this file.
pub(super) type H = BinaryField128;

pub(super) fn artifacts() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts")
}

/// The shipped geometry with `2^log_num_cells` cells.
pub(super) fn geometry(log_num_cells: u32) -> SupportedGeometry {
    *SUPPORTED_GEOMETRIES
        .iter()
        .find(|geometry| geometry.log_num_cells == log_num_cells)
        .unwrap()
}

fn splitmix64(mut state: u64) -> impl FnMut() -> u64 {
    move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

pub(super) fn message(len: usize, seed: u64) -> Vec<u128> {
    let mut next = splitmix64(seed);
    (0..len)
        .map(|_| u128::from(next()) << 64 | u128::from(next()))
        .collect()
}

/// The multilinear extension of `source` at `point`, first variable lowest.
fn evaluate(source: &[u128], point: &[H]) -> H {
    let mut table = source
        .iter()
        .map(|&word| H::embed_source(word))
        .collect::<Vec<_>>();
    for &coordinate in point {
        for index in 0..table.len() / 2 {
            table[index] =
                table[2 * index] * (H::ONE + coordinate) + table[2 * index + 1] * coordinate;
        }
        table.truncate(table.len() / 2);
    }
    assert_eq!(table.len(), 1);
    table[0]
}

/// A prover and a verifier of one shipped geometry, and one true statement.
pub(super) struct Case<P: FieldFamily> {
    pub(super) prover: Prover<P>,
    pub(super) verifier: Verifier<P>,
    pub(super) source: Vec<u128>,
    pub(super) point: Vec<H>,
    pub(super) value: H,
    pub(super) setup_seconds: f64,
    mode: RootOpeningMode,
    pub(super) prime: Option<PrimeClaim<P::Challenge>>,
}

impl<P: FieldFamily> Case<P> {
    pub(super) fn new(log_num_cells: u32, mode: RootOpeningMode) -> Self {
        let start = Instant::now();
        let geometry = geometry(log_num_cells);
        let ShippedCatalogs {
            tables,
            digits: catalog,
            elements,
        } = shipped_catalog::<P>(geometry, &artifacts()).unwrap();
        let elements = mode.has_prime().then_some(elements);
        let requirements = tables
            .setup_requirements::<P>(&catalog, elements.as_ref())
            .unwrap();
        let setup = akita_pcs::new_prover_setup(&requirements).unwrap();
        let verifier_setup = setup
            .to_verifier_setup(requirements.matrix_capacity())
            .unwrap();
        let prover = Prover::<P>::new(geometry, catalog.clone(), elements.clone(), setup).unwrap();
        let verifier = Verifier::<P>::new(geometry, catalog, elements, verifier_setup).unwrap();
        let setup_seconds = start.elapsed().as_secs_f64();
        let source = message(prover.root().setup().source_len(), 1);
        let mut next = splitmix64(2);
        let point = (0..prover.root().setup().num_vars())
            .map(|_| H::from_words([next(), next()]))
            .collect::<Vec<_>>();
        let value = evaluate(&source, &point);
        let prime = mode.has_prime().then(|| prime_claim::<P>(&prover, &source));
        Self {
            mode,
            prime,
            prover,
            verifier,
            source,
            point,
            value,
            setup_seconds,
        }
    }

    pub(super) fn statement(&self) -> RootStatement<'_, H, P::Challenge> {
        RootStatement {
            binary: self.mode.has_binary().then_some((&self.point, self.value)),
            prime: self.prime.as_ref(),
        }
    }

    /// Prove the statement through `oracle` wrapped in a [`Recording`].
    pub(super) fn open_recorded(&self, committed: &Committed<P>) -> (Vec<u8>, Sections) {
        let mut oracle = Recording::new(self.prover.oracle(committed, self.mode).unwrap());
        let (proof, _) = prove_root_reduction_bytes::<H, P::Base, P::Challenge, _, _, _>(
            self.prover.root(),
            &self.source,
            &committed.clear,
            &self.statement(),
            &mut oracle,
        )
        .unwrap();
        (proof, oracle.sections)
    }
}

/// Build a true cell claim from random field coordinates and bit weights.
fn prime_claim<P: FieldFamily>(prover: &Prover<P>, source: &[u128]) -> PrimeClaim<P::Challenge> {
    let layout = LoweredRootLayout::new::<P::Base, P::Challenge, 648, _>(
        prover.root().setup(),
        prover.root().shape(),
    )
    .unwrap();
    let mut next = splitmix64(4);
    let mut random = || {
        P::Challenge::from_base_fn(|_| {
            P::Base::from_u128(u128::from(next()) | (u128::from(next()) << 64))
        })
    };
    let point = (0..prover.root().setup().num_vars())
        .map(|_| random())
        .collect::<Vec<_>>();
    let weights = (0..layout.degree() / layout.k())
        .map(|_| random())
        .collect::<Vec<_>>();
    // Each byte lookup sums only the set source bits; bits >=128 are zero.
    let byte_weights = (0..16)
        .map(|byte| {
            (0u16..256)
                .map(|value| {
                    (0..8)
                        .filter(|bit| value & (1 << bit) != 0)
                        .fold(P::Challenge::zero(), |sum, bit| {
                            sum + weights[8 * byte + bit]
                        })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut table = source
        .iter()
        .map(|word| {
            word.to_le_bytes()
                .iter()
                .enumerate()
                .fold(P::Challenge::zero(), |sum, (byte, &value)| {
                    sum + byte_weights[byte][usize::from(value)]
                })
        })
        .collect::<Vec<_>>();
    for &coordinate in &point {
        for index in 0..table.len() / 2 {
            table[index] = table[2 * index] * (P::Challenge::one() - coordinate)
                + table[2 * index + 1] * coordinate;
        }
        table.truncate(table.len() / 2);
    }
    assert_eq!(table.len(), 1);
    PrimeClaim::cell_evaluation(&layout, &point, &weights, table[0]).unwrap()
}

/// What the nested scheme sent in the reduction's two proving calls, and how
/// long each call took.
#[derive(Default)]
pub(super) struct Sections {
    pub(super) prime: Vec<u8>,
    pub(super) prime_seconds: f64,
    pub(super) response: Vec<u8>,
    pub(super) response_seconds: f64,
    pub(super) nested: Vec<u8>,
    pub(super) nested_seconds: f64,
}

/// An oracle that forwards every call and records the two nested sections.
///
/// The reduction's channel reports no position, so the sections of a proof are
/// located by the bytes the oracle exchanged, never by the reduction's wire
/// grammar.
pub(super) struct Recording<O> {
    inner: O,
    pub(super) sections: Sections,
}

impl<O> Recording<O> {
    pub(super) fn new(inner: O) -> Self {
        Self {
            inner,
            sections: Sections::default(),
        }
    }
}

impl<E: Field, O: RootProverOracle<E>> RootProverOracle<E> for Recording<O> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.inner.bind_image(layout, channel)
    }
    fn commit_prime_opening<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        table: &[E],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        let start = Instant::now();
        let mut tap = Tap(channel, &mut self.sections.prime);
        let result = self.inner.commit_prime_opening(layout, table, &mut tap);
        self.sections.prime_seconds = start.elapsed().as_secs_f64();
        result
    }
    fn commit_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        digits: &[u8],
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        let start = Instant::now();
        let mut tap = Tap(channel, &mut self.sections.response);
        let result = self.inner.commit_response(layout, digits, &mut tap);
        self.sections.response_seconds = start.elapsed().as_secs_f64();
        result
    }
    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<E>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        let start = Instant::now();
        let mut tap = Tap(channel, &mut self.sections.nested);
        let result = self.inner.discharge(claims, &mut tap);
        self.sections.nested_seconds = start.elapsed().as_secs_f64();
        result
    }
}

impl<E: Field, O: RootVerifierOracle<E>> RootVerifierOracle<E> for Recording<O> {
    fn bind_image<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        self.inner.bind_image(layout, channel)
    }
    fn bind_prime_opening<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        let start = Instant::now();
        let mut tap = Tap(channel, &mut self.sections.prime);
        let result = self.inner.bind_prime_opening(layout, &mut tap);
        self.sections.prime_seconds = start.elapsed().as_secs_f64();
        result
    }
    fn bind_response<S: ClearChannel>(
        &mut self,
        layout: &LoweredRootLayout,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        let start = Instant::now();
        let mut tap = Tap(channel, &mut self.sections.response);
        let result = self.inner.bind_response(layout, &mut tap);
        self.sections.response_seconds = start.elapsed().as_secs_f64();
        result
    }
    fn discharge<S: ClearChannel>(
        &mut self,
        claims: &RootEvaluationClaims<E>,
        channel: &mut S,
    ) -> Result<(), AkitaError> {
        let start = Instant::now();
        let mut tap = Tap(channel, &mut self.sections.nested);
        let result = self.inner.discharge(claims, &mut tap);
        self.sections.nested_seconds = start.elapsed().as_secs_f64();
        result
    }
}

/// A channel that forwards to another and keeps a copy of every message.
struct Tap<'a, S>(&'a mut S, &'a mut Vec<u8>);

impl<S: ClearChannel> ClearChannel for Tap<'_, S> {
    fn public(&mut self, bytes: &[u8]) -> Result<(), AkitaError> {
        self.0.public(bytes)
    }
    fn message(&mut self, bytes: &mut [u8]) -> Result<(), AkitaError> {
        self.0.message(bytes)?;
        self.1.extend_from_slice(bytes);
        Ok(())
    }
    fn challenge_block(&mut self) -> Result<[u8; 32], AkitaError> {
        self.0.challenge_block()
    }
    fn fold_challenges(
        &mut self,
        sampler: &mut BinaryChallengeSampler,
        label: &[u8],
        count: usize,
        nonce: &mut u32,
    ) -> Result<Vec<BinaryChallenge>, AkitaError> {
        self.0.fold_challenges(sampler, label, count, nonce)
    }
}

/// The one place `section` occurs in `proof`.
pub(super) fn locate(proof: &[u8], section: &[u8]) -> Range<usize> {
    let mut starts = proof
        .windows(section.len())
        .enumerate()
        .filter(|(_, window)| *window == section)
        .map(|(start, _)| start);
    let start = starts.next().unwrap();
    assert!(starts.next().is_none(), "a proof section must occur once");
    start..start + section.len()
}

/// One opening at the sample geometry, with its timings and sizes on stderr.
///
/// The commit split comes from the two spans of `Prover::commit`, printed as
/// they close. The prove and verify splits are the time inside the nested
/// scheme's two calls against the rest of the reduction.
pub(super) fn sample_geometry_opening<P: FieldFamily>(name: &str, mode: RootOpeningMode) {
    let case = Case::<P>::new(22, mode);
    let spans = tracing_subscriber::fmt()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .finish();
    let start = Instant::now();
    let committed =
        tracing::subscriber::with_default(spans, || case.prover.commit::<H>(&case.source).unwrap());
    let commit_seconds = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let (proof, open) = case.open_recorded(&committed);
    let open_seconds = start.elapsed().as_secs_f64();

    let mut oracle = Recording::new(case.verifier.oracle(&committed.commitment, mode).unwrap());
    let start = Instant::now();
    verify_root_reduction_bytes::<H, P::Base, P::Challenge, _, _, _>(
        case.verifier.root(),
        &case.statement(),
        &mut oracle,
        &proof,
    )
    .unwrap();
    let verify_seconds = start.elapsed().as_secs_f64();
    let verify = oracle.sections;

    eprintln!(
        "{mode:?} opening {name} (22, 8): setup_seconds={:.3} commit_seconds={commit_seconds:.3} \
         open_seconds={open_seconds:.3} open_reduction_seconds={:.3} \
         open_response_commitment_seconds={:.3} open_nested_proof_seconds={:.3} \
         verify_seconds={verify_seconds:.4} verify_reduction_seconds={:.4} \
         verify_response_commitment_seconds={:.4} verify_nested_proof_seconds={:.4} \
         commitment_bytes={} proof_bytes={} reduction_bytes={} response_commitment_bytes={} \
         nested_proof_bytes={}",
        case.setup_seconds,
        open_seconds - open.prime_seconds - open.response_seconds - open.nested_seconds,
        open.response_seconds,
        open.nested_seconds,
        verify_seconds - verify.prime_seconds - verify.response_seconds - verify.nested_seconds,
        verify.response_seconds,
        verify.nested_seconds,
        committed.commitment.compressed_size(),
        proof.len(),
        proof.len() - open.prime.len() - open.response.len() - open.nested.len(),
        open.response.len(),
        open.nested.len(),
    );
    if mode.has_prime() {
        eprintln!("{mode:?} opening {name}: prime_left_commitment_bytes={} open_prime_left_commitment_seconds={:.3} verify_prime_left_commitment_seconds={:.4}", open.prime.len(), open.prime_seconds, verify.prime_seconds);
        eprintln!("{mode:?} opening {name}: three_group_nested_proof_bytes={} framing_bytes=8 open_three_group_nested_proof_seconds={:.3} verify_three_group_nested_proof_seconds={:.4}", open.nested.len() - 8, open.nested_seconds, verify.nested_seconds);
    }
}
