#![cfg(feature = "labinius")]
#![allow(dead_code)]

pub(crate) mod combined;

use akita_algebra::{
    binary::{field_switch::SwitchField, BinaryField128, BinaryField192},
    MinusTrinomial, TrinomialModulus,
};
use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing, FoldDraw};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear, lowered::encode_image, prove_root_reduction_bytes,
    TransparentRootProverOracle,
};
use akita_labinius_verifier::{
    lowered::LoweredRootLayout,
    profile::BinaryClearSetup,
    root::{verify_root_reduction_bytes, RootEvaluationClaims, TransparentRootVerifierOracle},
    AdmittedRootSetup, BinaryClearCommitment, PrimeClaim, RootOpeningMode, RootStatement,
};
use akita_params::sis::labinius::{
    LabiniusCommitmentModulus, LabiniusRingDegree, LabiniusRootProfile,
};
use akita_types::proof::AkitaSetupSeed;
use jolt_field::{CanonicalEncoding, ExtField, Field, Prime128Offset275};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::marker::PhantomData;

pub(crate) const PROFILE: LabiniusRootProfile = LabiniusRootProfile::D648Q25BoundedW46;
pub(crate) const Q: u32 = LabiniusCommitmentModulus::Q25Plus14561.modulus();
pub(crate) type Admitted = AdmittedRootSetup<648, MinusTrinomial>;

pub(crate) trait TestHost: SwitchField {
    fn random(rng: &mut StdRng) -> Self;
    fn random_source(rng: &mut StdRng) -> Self::Source;
    fn flip(word: &mut Self::Source);
}
impl TestHost for BinaryField128 {
    fn random(rng: &mut StdRng) -> Self {
        Self::from_words([rng.next_u64(), rng.next_u64()])
    }
    fn random_source(rng: &mut StdRng) -> Self::Source {
        u128::from(rng.next_u64()) | (u128::from(rng.next_u64()) << 64)
    }
    fn flip(word: &mut Self::Source) {
        *word ^= 1;
    }
}
impl TestHost for BinaryField192 {
    fn random(rng: &mut StdRng) -> Self {
        Self::from_words([rng.next_u64(), rng.next_u64(), rng.next_u64()])
    }
    fn random_source(rng: &mut StdRng) -> Self::Source {
        rng.next_u64()
    }
    fn flip(word: &mut Self::Source) {
        *word ^= 1;
    }
}

pub(crate) struct FixedDraw;
impl FoldDraw for FixedDraw {
    fn absorb_and_squeeze(&mut self, _payload: &[u8]) -> Result<[u8; 32], AkitaError> {
        Ok([0x53; 32])
    }
}

/// An explicit setup with a seeded matrix of residues below the commitment prime.
pub(crate) fn clear_setup<const D: usize, M: TrinomialModulus>(
    n_a: usize,
    m: usize,
    columns: usize,
) -> BinaryClearSetup<D, M> {
    let mut rng = StdRng::seed_from_u64(0x215_197);
    let matrix = (0..n_a * m * D)
        .map(|_| (rng.next_u64() % u64::from(Q)) as u32)
        .collect();
    clear_setup_with(matrix, n_a, m, columns).unwrap()
}

/// The same explicit setup over a caller-chosen matrix.
pub(crate) fn clear_setup_with<const D: usize, M: TrinomialModulus>(
    matrix: Vec<u32>,
    n_a: usize,
    m: usize,
    columns: usize,
) -> Result<BinaryClearSetup<D, M>, AkitaError> {
    BinaryClearSetup::new(
        matrix,
        n_a,
        m,
        columns,
        -1024,
        1024,
        128,
        BinaryChallengeProfile::fixed_weight(BinaryScalarRing::Cyclotomic243, 47).unwrap(),
        match D {
            162 => LabiniusRingDegree::D162,
            324 => LabiniusRingDegree::D324,
            648 => LabiniusRingDegree::D648,
            _ => panic!("test degree"),
        },
        LabiniusCommitmentModulus::Q25Plus14561,
    )
}

pub(crate) fn data<H: TestHost>(len: usize, variables: usize) -> (Vec<H::Source>, Vec<H>, H) {
    let mut rng = StdRng::seed_from_u64(0xabc_571);
    let source = (0..len)
        .map(|_| H::random_source(&mut rng))
        .collect::<Vec<_>>();
    let point = (0..variables)
        .map(|_| H::random(&mut rng))
        .collect::<Vec<_>>();
    let claim = host_mle::<H>(&source, &point);
    (source, point, claim)
}

pub(crate) fn host_mle<H: SwitchField>(source: &[H::Source], point: &[H]) -> H {
    source
        .iter()
        .enumerate()
        .fold(H::ZERO, |sum, (index, &word)| {
            let weight = point.iter().enumerate().fold(H::ONE, |acc, (axis, &r)| {
                acc * if index >> axis & 1 == 1 {
                    r
                } else {
                    H::ONE + r
                }
            });
            sum + weight * H::embed_source(word)
        })
}

/// The shipped profile at sixteen cells. The matrix is the reduced public
/// prefix of the Akita setup stream over `Prime128Offset275` for every proof
/// field pair: a 64-bit stream fails the derivation-bias check.
pub(crate) fn admitted(log_fold_width: u32, seed: u8) -> Admitted {
    Admitted::derive::<Prime128Offset275>(
        PROFILE,
        4,
        log_fold_width,
        128,
        AkitaSetupSeed::shake256_paged_v1([seed; 32]),
    )
    .unwrap()
}

/// `sum_cell eq(point, cell) * sum_s bit_weights[s] * bit_s(cell)`, read
/// straight from the source words as integers zero and one.
pub(crate) fn cell_functional<T: TestHost, E: Field>(
    source: &[T::Source],
    point: &[E],
    bit_weights: &[E],
) -> E {
    source
        .iter()
        .enumerate()
        .fold(E::zero(), |sum, (index, &word)| {
            let weight = point.iter().enumerate().fold(E::one(), |acc, (axis, &r)| {
                acc * if index >> axis & 1 == 1 {
                    r
                } else {
                    E::one() - r
                }
            });
            let bits: u128 = word.into();
            let cell = (0..128)
                .filter(|s| bits >> s & 1 == 1)
                .fold(E::zero(), |acc, s| acc + bit_weights[s]);
            sum + weight * cell
        })
}

/// One root reduction case over the base field `F` and the challenge field
/// `E`: a committed source with a binary claim and a prime cell claim about
/// it, the opening mode that selects which of them the statement carries, and
/// the transparent image digit table.
pub(crate) struct RootCase<T: TestHost, F, E> {
    pub(crate) admitted: Admitted,
    pub(crate) mode: RootOpeningMode,
    pub(crate) source: Vec<T::Source>,
    pub(crate) point: Vec<T>,
    pub(crate) value: T,
    pub(crate) cell_point: Vec<E>,
    pub(crate) bit_weights: Vec<E>,
    pub(crate) prime: PrimeClaim<E>,
    pub(crate) commitment: BinaryClearCommitment,
    pub(crate) layout: LoweredRootLayout,
    pub(crate) image: Vec<u8>,
    base: PhantomData<F>,
}

impl<T: TestHost, F: Field + CanonicalEncoding, E: ExtField<F>> RootCase<T, F, E> {
    pub(crate) fn new(admitted: Admitted, mode: RootOpeningMode) -> Self {
        let setup = admitted.setup();
        let (source, point, value) = data::<T>(setup.source_len(), setup.num_vars());
        let commitment = commit_binary_clear::<T, 648, MinusTrinomial>(setup, &source).unwrap();
        let layout = LoweredRootLayout::new::<F, E, _, _>(setup, admitted.shape()).unwrap();
        let image = encode_image(&layout, &commitment).unwrap();
        let mut rng = StdRng::seed_from_u64(0x0009_a1e5);
        let mut elements = |count: usize| -> Vec<E> {
            (0..count)
                .map(|_| {
                    let coordinates: Vec<F> = (0..E::DEGREE)
                        .map(|_| F::from_u64(rng.next_u64()))
                        .collect();
                    E::from_base_slice(&coordinates)
                })
                .collect()
        };
        let cell_point = elements(setup.num_vars());
        let bit_weights = elements(162);
        let prime = PrimeClaim::cell_evaluation(
            &layout,
            &cell_point,
            &bit_weights,
            cell_functional::<T, E>(&source, &cell_point, &bit_weights),
        )
        .unwrap();
        Self {
            admitted,
            mode,
            source,
            point,
            value,
            cell_point,
            bit_weights,
            prime,
            commitment,
            layout,
            image,
            base: PhantomData,
        }
    }

    /// The statement of `mode` about this case's claims.
    pub(crate) fn statement(&self, mode: RootOpeningMode) -> RootStatement<'_, T, E> {
        RootStatement {
            binary: mode
                .has_binary()
                .then_some((self.point.as_slice(), self.value)),
            prime: mode.has_prime().then_some(&self.prime),
        }
    }

    pub(crate) fn prove(&self) -> Result<(Vec<u8>, RootEvaluationClaims<E>), AkitaError> {
        prove_root_reduction_bytes::<T, F, E, _, _, _>(
            &self.admitted,
            &self.source,
            &self.commitment,
            &self.statement(self.mode),
            &mut TransparentRootProverOracle::<F, E>::new(&self.image),
        )
    }

    pub(crate) fn verify(&self, proof: &[u8]) -> Result<RootEvaluationClaims<E>, AkitaError> {
        verify_root_reduction_bytes::<T, F, E, _, _, _>(
            &self.admitted,
            &self.statement(self.mode),
            &mut TransparentRootVerifierOracle::<F, E>::new(&self.image),
            proof,
        )
    }
}
