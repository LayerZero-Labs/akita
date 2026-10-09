#![cfg(feature = "labinius")]
#![allow(dead_code)]
use crate::common;
use akita_algebra::{
    binary::{BinaryField128, BinaryField162 as B},
    MinusTrinomial,
};
use akita_challenges::{BinaryChallenge, BinaryChallengeSampler};
use akita_error::AkitaError;
use akita_labinius_prover::{
    commit_binary_clear,
    lowered::{encode_witness, flatten_image, parity_quotient_and_carry},
    prove_root_reduction_bytes,
    quotient_kernel::a_relation_quotients,
    PreparedRootMatrices, TransparentRootProverOracle,
};
use akita_labinius_verifier::{
    endpoint::{fold_integer, left_expansion},
    lowered::{ARelationAuxiliary, LoweredChallenges, LoweredPublic, LoweredRootLayout},
    root::{verify_root_reduction_bytes, RootEvaluationClaims, TransparentRootVerifierOracle},
    AdmittedRootSetup, BinaryClearCommitment, BinaryEvaluationClaim,
};
use akita_params::sis::labinius::{LabiniusDigitBase, LabiniusRootProfile};
use akita_types::proof::AkitaSetupSeed;
use common::{data, FixedDraw, Tape, TestHost};
use jolt_field::Ring;
pub(crate) type F = jolt_field::Prime128OffsetA7F7;
pub(crate) type H = BinaryField128;
pub(crate) type Setup = AdmittedRootSetup<F, 648, MinusTrinomial>;
pub(crate) type Commitment = BinaryClearCommitment<F, 648, MinusTrinomial>;
pub(crate) const SMALL: LabiniusRootProfile = LabiniusRootProfile::D648P128Q28BoundedW46Delta16;
pub(crate) const BASES: [LabiniusDigitBase; 3] = [
    LabiniusDigitBase::Bits1,
    LabiniusDigitBase::Bits2,
    LabiniusDigitBase::Bits4,
];
pub(crate) fn admitted(profile: LabiniusRootProfile, fold: u32) -> Setup {
    Setup::derive(
        profile,
        4,
        fold,
        128,
        AkitaSetupSeed::shake256_paged_v1([0x31; 32]),
    )
    .unwrap()
}
pub(crate) struct Case<T: TestHost = H> {
    pub(crate) admitted: Setup,
    pub(crate) prepared: PreparedRootMatrices<F, 648, MinusTrinomial>,
    pub(crate) base: LabiniusDigitBase,
    pub(crate) source: Vec<T::Source>,
    pub(crate) point: Vec<T>,
    pub(crate) value: T,
    pub(crate) commitment: Commitment,
    pub(crate) layout: LoweredRootLayout,
    pub(crate) image: Vec<F>,
}
impl<T: TestHost> Case<T> {
    pub(crate) fn new(base: LabiniusDigitBase, fold: u32) -> Self {
        let admitted = admitted(SMALL, fold);
        let setup = admitted.setup();
        let (source, point, value) = data::<T>(setup.source_len(), setup.num_vars());
        let commitment = commit_binary_clear::<T, F, 648, MinusTrinomial>(setup, &source).unwrap();
        let prepared = PreparedRootMatrices::prepare(setup).unwrap();
        let layout = LoweredRootLayout::new(setup, admitted.shape(), base).unwrap();
        let image = flatten_image(&layout, &commitment).unwrap();
        Self {
            admitted,
            prepared,
            base,
            source,
            point,
            value,
            commitment,
            layout,
            image,
        }
    }
    pub(crate) fn refresh_image(&mut self) {
        self.image = flatten_image(&self.layout, &self.commitment).unwrap();
    }
    pub(crate) fn prove(&self) -> (Vec<u8>, RootEvaluationClaims<F>) {
        prove_root_reduction_bytes(
            &self.admitted,
            &self.prepared,
            self.base,
            &self.source,
            &self.commitment,
            &self.point,
            self.value,
            &mut TransparentRootProverOracle::new(&self.image),
        )
        .unwrap()
    }
    pub(crate) fn verify(&self, proof: &[u8]) -> Result<RootEvaluationClaims<F>, AkitaError> {
        verify_root_reduction_bytes(
            &self.admitted,
            self.base,
            &self.point,
            self.value,
            &mut TransparentRootVerifierOracle::new(&self.image),
            proof,
        )
    }
    pub(crate) fn a_wire(&self) -> (usize, usize, usize) {
        let start = T::ROWS * size_of::<T::Source>()
            + (2 * self.point.len() + 1) * 21
            + self.layout.columns() * 21
            + self.layout.witness_len();
        let qa_bytes = self.layout.n_a() * 647 * 16;
        let ka_width = self.layout.encoding().a_carry().unwrap().bits().div_ceil(8) as usize;
        (start, start + qa_bytes, ka_width)
    }
}
pub(crate) struct RelationCase {
    pub(crate) case: Case,
    pub(crate) claim: BinaryEvaluationClaim,
    pub(crate) u: Vec<B>,
    pub(crate) fold: Vec<BinaryChallenge>,
    pub(crate) response: Vec<[i64; 162]>,
    pub(crate) a: ARelationAuxiliary<F>,
    pub(crate) q: Vec<i128>,
    pub(crate) k: Vec<i128>,
    pub(crate) w: Vec<u8>,
    pub(crate) challenges: LoweredChallenges<F>,
}
impl RelationCase {
    pub(crate) fn new(base: LabiniusDigitBase) -> Self {
        let case = Case::new(base, 1);
        let setup = case.admitted.setup();
        let point = (0..setup.num_vars())
            .map(|i| Tape::draw_value(i + 8, 0))
            .collect::<Vec<_>>();
        let u = left_expansion::<H>(&case.source, &point, setup.scalar_rows(), setup.columns())
            .unwrap();
        let value = common::binary_mle(&u, &point[setup.row_vars()..]);
        let claim = BinaryEvaluationClaim { point, value };
        let fold = BinaryChallengeSampler::new(setup.profile().clone())
            .sample_challenges(&mut FixedDraw, b"lifted-oracle", setup.columns())
            .unwrap();
        let response = fold_integer::<H>(
            &case.source,
            setup.scalar_rows(),
            setup.columns(),
            &fold,
            setup.profile(),
        )
        .unwrap();
        let a = a_relation_quotients(
            case.prepared.commit(),
            case.prepared.quotient(),
            setup,
            &case.commitment,
            &fold,
            &response,
            case.layout.encoding().a_carry(),
        )
        .unwrap();
        let (q, k) = parity_quotient_and_carry(setup, &claim, &u, &fold, &response).unwrap();
        let w = encode_witness(&case.layout, &response).unwrap();
        let challenges = LoweredChallenges {
            alpha: F::from_u64(781),
            xi: F::from_u64(911),
            gamma: F::from_u64(1237),
        };
        Self {
            case,
            claim,
            u,
            fold,
            response,
            a,
            q,
            k,
            w,
            challenges,
        }
    }
    pub(crate) fn public(&self, a: &ARelationAuxiliary<F>) -> Result<LoweredPublic<F>, AkitaError> {
        LoweredPublic::new(
            &self.case.layout,
            self.case.admitted.setup(),
            &self.claim,
            &self.u,
            &self.fold,
            a,
            &self.q,
            &self.k,
            self.challenges,
        )
    }
}

/// Replay through the auxiliary messages and independently decode signed packing
/// from the transparent response digits. The oracle receives transmitted KA.
pub(crate) fn transmitted_relation(
    case: &Case,
) -> (Vec<Vec<i128>>, Vec<BinaryChallenge>, ARelationAuxiliary<F>) {
    use akita_labinius_verifier::{
        channel::{self, ClearChannel, RootSumcheckVerifierChannel},
        codec::exchange_binary,
        frontend::verify_frontend,
        root::{bind_root_statement, exchange_root_auxiliary, RootVerifierOracle},
    };
    let proof = case.prove().0;
    case.verify(&proof).unwrap();
    let mut state = channel::new_root_verifier(&proof).unwrap();
    let mut ch = RootSumcheckVerifierChannel::new(&mut state);
    let mut oracle = TransparentRootVerifierOracle::new(&case.image);
    bind_root_statement(
        &case.admitted,
        case.base,
        &case.point,
        case.value,
        &mut ch,
        |l, c| oracle.bind_image(l, c),
    )
    .unwrap();
    verify_frontend(&case.point, case.value, &mut ch).unwrap();
    for _ in 0..case.layout.columns() {
        let mut u = B::ZERO;
        exchange_binary(&mut ch, &mut u).unwrap();
    }
    let fold = ch
        .fold_challenges(
            &mut BinaryChallengeSampler::new(case.admitted.setup().profile().clone()),
            b"akita/labinius/root-fold/v1",
            case.layout.columns(),
        )
        .unwrap();
    oracle.bind_response(&case.layout, &mut ch).unwrap();
    let dc = case.layout.encoding().response().digit_count();
    let packed = (0..case.layout.m())
        .map(|j| {
            (0..648)
                .map(|t| {
                    let unsigned = (0..dc).fold(0i128, |v, l| {
                        v + (i128::from(oracle.response()[l + dc * (t + 1024 * j)])
                            << (case.base.bits() * l as u32))
                    });
                    unsigned - if (t / 4) % 2 == 0 { 32768 } else { 32767 }
                })
                .collect()
        })
        .collect();
    let mut a = ARelationAuxiliary {
        quotients: vec![vec![F::from_u64(0); 647]; 3],
        carry: vec![0; 1944],
    };
    exchange_root_auxiliary(
        &case.layout,
        &mut ch,
        &mut a,
        &mut vec![0; 161],
        &mut vec![0; 162],
    )
    .unwrap();
    (packed, fold, a)
}
