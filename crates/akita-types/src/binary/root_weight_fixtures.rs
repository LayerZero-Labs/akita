//! Shared binary root inputs and stage-specific prover fixtures.

use akita_challenges::{Challenges, FoldChallengeDistribution, SparseChallenge};
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::sis::{
    sis_table_key_for_linf_bound, InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole,
    SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
};
use akita_params::{
    BlockGeometry, CommitmentSliceCount, CommitmentSliceGeometry, GadgetDigits,
    GroupCommitPhaseParams, PolynomialGroupLayout, RoleParams, SisModulusProfileId,
};
use jolt_field::{Prime128Offset275, Ring};

use super::{
    BinaryAOperand, BinaryAProduct, BinaryCommittedGroupManifest, BinaryIndexMap, BinaryRootPublic,
    BinarySourceDescriptor, BinaryTerminalClaim, BinaryTypedPrefix, PrimeGroupClaim,
    PrimePlaneClaim, PrimePlaneManifest, TypedPrefixWeights, BINARY_PROJECTION_COUNT,
};
use crate::native_root_relation::NativeRootWeights;
use crate::AkitaSetupDescriptor;

/// The proof field of the fixtures, matching the mixed ordinary group.
pub type RootFixtureField = Prime128Offset275;

/// One compiled root relation with A, as the prover sees it.
pub struct RootWeightFixture {
    pub name: String,
    pub weights: NativeRootWeights<RootFixtureField>,
    pub negative_binary_weights: Vec<(usize, RootFixtureField)>,
    /// Live witness prefix: the natural end of the repacked digits plus any
    /// extra zero rings.
    pub logical_len: usize,
}

/// The typed setup table, its dense Stage 3 weight and the same weight in
/// live blocks.
pub struct Stage3Fixture {
    pub setup: Vec<RootFixtureField>,
    pub weights: Vec<RootFixtureField>,
    pub blocks: TypedPrefixWeights<RootFixtureField>,
}

/// Root shape parameters, as in the structured-weight differential tests.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub(crate) word_width: usize,
    pub(crate) carriers: usize,
    pub(crate) columns: usize,
    pub(crate) owners: usize,
    pub(crate) prime_planes: usize,
    pub(crate) extra_rings: usize,
    pub(crate) mixed: bool,
}

/// SplitMix64: deterministic public inputs without a test-only RNG dependency.
#[derive(Clone)]
pub(crate) struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }

    fn u127(&mut self) -> u128 {
        ((u128::from(self.next()) << 64) | u128::from(self.next())) >> 1
    }

    pub(crate) fn field(&mut self) -> RootFixtureField {
        RootFixtureField::from_u128(self.u127())
    }

    fn binary_field_bytes(&mut self) -> [u8; 21] {
        let mut bytes = [0u8; 21];
        for byte in &mut bytes {
            *byte = self.next() as u8;
        }
        bytes[20] &= 0x03;
        bytes
    }
}

/// One small ordinary group with real audited table cells.
pub fn binary_ordinary_profile_fixture(field: BinaryFieldProfile) -> GroupCommitPhaseParams {
    let (modulus, digits) = match field {
        BinaryFieldProfile::P128 => (SisModulusProfileId::Q128Offset275, 43),
        BinaryFieldProfile::P64 => (SisModulusProfileId::Q64Offset59, 22),
    };
    let key = |role, degree| {
        sis_table_key_for_linf_bound(
            DEFAULT_SIS_SECURITY_POLICY,
            SisTableDigest::CURRENT,
            modulus,
            role,
            degree,
            8,
        )
        .expect("audited ordinary table cell")
    };
    let inner = InnerCommitMatrixParams::try_new_with_min_rank(key(SisMatrixRole::Inner, 128), 2)
        .expect("inner fixture params");
    let geometry = CommitmentSliceGeometry::try_new(
        CommitmentSliceCount::ONE,
        4,
        1,
        inner.output_rank(),
        digits,
        128,
        64,
    )
    .expect("fixture slice geometry");
    let outer = OuterCommitMatrixParams::try_new_with_min_rank(
        key(SisMatrixRole::Outer, 64),
        geometry.physical_input_width(),
    )
    .expect("outer fixture params");
    let profile = GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group: PolynomialGroupLayout::singleton(10),
        blocks: BlockGeometry::new(8, 2, 4),
        outer_slice_count: CommitmentSliceCount::ONE,
        inner: RoleParams::new(GadgetDigits::new(3, 1), inner),
        outer: RoleParams::new(GadgetDigits::new(3, digits), outer),
    };
    profile
        .validate_frozen_precommit(if field == BinaryFieldProfile::P128 {
            128
        } else {
            64
        })
        .expect("frozen fixture profile");
    profile
}

/// Setup seed of every fixture.
const SETUP_SEED: [u8; 32] = [41; 32];

/// The root public inputs, its manifest, the live witness length and the B/D
/// payload length.
fn public(
    shape: Shape,
    stream: &mut Stream,
) -> (BinaryRootPublic, BinaryCommittedGroupManifest, usize, usize) {
    let index = BinaryIndexMap::new(
        shape.word_width,
        shape.carriers * (128 / shape.word_width),
        shape.carriers,
        shape.columns,
    )
    .expect("fixture index map");
    let profiles = BinaryCommittedGroupManifest::profiles_for_field(
        &index,
        shape.owners,
        0,
        BinaryFieldProfile::P128,
    )
    .expect("fixture commitment profiles");
    let polynomials = (0..shape.owners)
        .map(|owner| [29 + owner as u8; 32])
        .collect::<Vec<_>>();
    let descriptor = BinarySourceDescriptor::new(index, polynomials.clone(), profiles)
        .expect("fixture descriptor");
    let manifest = BinaryCommittedGroupManifest::with_field_profile(
        descriptor.clone(),
        0,
        &binary_ordinary_profile_fixture(BinaryFieldProfile::P128),
        BinaryFieldProfile::P128,
    )
    .expect("fixture manifest");
    let rows = descriptor.index_map().padded_carriers() / descriptor.index_map().columns();
    let claim = BinaryTerminalClaim::new(
        &descriptor,
        polynomials[shape.owners - 1],
        (0..rows.trailing_zeros())
            .map(|_| stream.binary_field_bytes())
            .collect(),
        (0..descriptor.index_map().columns().trailing_zeros())
            .map(|_| stream.binary_field_bytes())
            .collect(),
        stream.binary_field_bytes(),
    )
    .expect("fixture word claim");
    let arity = descriptor.index_map().padded_carriers().trailing_zeros() as usize + 8;
    let claims = (shape.prime_planes > 0).then(|| PrimeGroupClaim {
        point: (0..arity).map(|_| stream.u127().into()).collect(),
        evaluations: (0..shape.prime_planes)
            .map(|polynomial| PrimePlaneClaim {
                polynomial,
                value: stream.u127().into(),
            })
            .collect(),
    });
    let prime =
        PrimePlaneManifest::with_field_profile(&descriptor, claims, BinaryFieldProfile::P128)
            .expect("fixture prime planes");
    let setup = AkitaSetupDescriptor {
        max_num_vars: 0,
        max_num_batched_polys: 0,
        num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        setup_seed: SETUP_SEED.into(),
    };
    let ordinary = shape
        .mixed
        .then(|| binary_ordinary_profile_fixture(BinaryFieldProfile::P128));
    let build = |logical_len| {
        BinaryRootPublic::new(&manifest, &claim, &prime, &setup, logical_len, 64, ordinary)
    };
    let natural_end = build(1 << 22)
        .and_then(|public| public.plan().repacking().natural_end())
        .expect("fixture natural end");
    assert!(build(natural_end - 1).is_err());
    let logical_len = natural_end + shape.extra_rings * 64;
    let payload_len = manifest.b_rows() * manifest.b_ring_dimension();
    (
        build(logical_len).expect("fixture root"),
        manifest,
        logical_len,
        payload_len,
    )
}

/// Raw public inputs shared by the independent root compilers and fixtures.
/// Creating a scenario compiles neither Stage 2 nor Stage 3 weights.
pub struct RootWeightScenario {
    pub name: String,
    pub(crate) public: BinaryRootPublic,
    pub(crate) manifest: BinaryCommittedGroupManifest,
    pub(crate) logical_len: usize,
    pub(crate) challenges: Challenges,
    pub(crate) alpha: RootFixtureField,
    pub(crate) batches: Vec<RootFixtureField>,
    pub(crate) rho: Vec<[RootFixtureField; 648]>,
    pub(crate) projected: [i128; BINARY_PROJECTION_COUNT],
    pub(crate) commitment: Vec<RootFixtureField>,
    pub(crate) d_payload: Vec<RootFixtureField>,
    pub(crate) ordinary_claim: Option<(
        Vec<RootFixtureField>,
        RootFixtureField,
        Vec<RootFixtureField>,
    )>,
    pub(crate) matrix: Vec<Vec<[i128; 648]>>,
    pub(crate) stream: Stream,
}

impl RootWeightScenario {
    pub(crate) fn new(shape: Shape, seed: u64) -> Self {
        let mut stream = Stream(seed);
        let (public, manifest, logical_len, payload_len) = public(shape, &mut stream);
        let descriptor = public.plan().descriptor().clone();
        let index = descriptor.index_map();
        let owners = descriptor.polynomials().len();
        let columns = index.columns();
        let weight = FoldChallengeDistribution::BinaryRoot.weight();
        let challenges = Challenges::from_sparse(
            (0..columns * owners)
                .map(|_| {
                    let mut positions = Vec::new();
                    while positions.len() < weight {
                        let position = stream.below(162) as u32;
                        if !positions.contains(&position) {
                            positions.push(position);
                        }
                    }
                    positions.sort_unstable();
                    SparseChallenge {
                        positions: positions.into(),
                        coeffs: (0..weight)
                            .map(|_| if stream.next() & 1 == 0 { 1i8 } else { -1 })
                            .collect::<Vec<_>>()
                            .into(),
                    }
                })
                .collect(),
            columns,
            owners,
        )
        .expect("fixture challenges");
        let alpha = stream.field();
        let batches = (0..public.batch_count())
            .map(|_| stream.field())
            .collect::<Vec<_>>();
        let rho = (0..owners * public.plan().rank())
            .map(|_| std::array::from_fn(|_| stream.field()))
            .collect::<Vec<[RootFixtureField; 648]>>();
        let projected: [i128; BINARY_PROJECTION_COUNT] =
            std::array::from_fn(|_| i128::from(stream.next() as i64));
        let commitment = (0..payload_len).map(|_| stream.field()).collect::<Vec<_>>();
        let d_payload = (0..payload_len).map(|_| stream.field()).collect::<Vec<_>>();
        let ordinary = binary_ordinary_profile_fixture(BinaryFieldProfile::P128);
        let ordinary_point = (0..ordinary.group.num_vars())
            .map(|_| stream.field())
            .collect::<Vec<_>>();
        let ordinary_payload = (0..ordinary
            .outer_compression_plan()
            .expect("fixture compression plan")
            .terminal_coefficients())
            .map(|_| stream.field())
            .collect::<Vec<_>>();
        let ordinary_claim = shape
            .mixed
            .then(|| (ordinary_point, stream.field(), ordinary_payload));
        let a_rows = index.padded_carriers() / (4 * columns);
        let matrix = (0..public.plan().rank())
            .map(|_| {
                (0..a_rows)
                    .map(|_| {
                        std::array::from_fn(|_| {
                            i128::from(stream.below(2 * 7_126_704 + 1)) - 7_126_704
                        })
                    })
                    .collect()
            })
            .collect::<Vec<Vec<[i128; 648]>>>();
        Self {
            name: format!("{shape:?}"),
            public,
            manifest,
            logical_len,
            challenges,
            alpha,
            batches,
            rho,
            projected,
            commitment,
            d_payload,
            ordinary_claim,
            matrix,
            stream,
        }
    }

    /// Compile only the Stage 2 root relation, including its A operand.
    pub fn compile_stage2(&self) -> RootWeightFixture {
        let (weights, negative_binary_weights) = self
            .public
            .relation(
                &self.challenges,
                self.alpha,
                &self.batches,
                &self.rho,
                &self.projected,
                &self.commitment,
                &self.d_payload,
                self.ordinary_claim
                    .as_ref()
                    .map(|(point, value, payload)| (point.as_slice(), *value, payload.as_slice())),
                Some(BinaryAOperand::new::<RootFixtureField>(&self.matrix)),
            )
            .expect("fixture relation");
        RootWeightFixture {
            name: self.name.clone(),
            weights,
            negative_binary_weights,
            logical_len: self.logical_len,
        }
    }

    /// Compile only the typed-prefix Stage 3 setup and weights.
    pub fn compile_stage3(&self) -> Stage3Fixture {
        let mut stream = self.stream.clone();
        let plan = self.public.plan();
        let manifest = &self.manifest;
        let typed = BinaryTypedPrefix::from_plan(manifest, plan).expect("fixture typed prefix");
        let prefix_len = typed.domain_len;
        let root_point = (0..plan.repacking().domain_len().trailing_zeros())
            .map(|_| stream.field())
            .collect::<Vec<_>>();
        let alpha = stream.field();
        let batches = [0, 1].map(|_| {
            (0..manifest.b_rows())
                .map(|_| stream.field())
                .collect::<Vec<_>>()
        });
        let batches = [batches[0].as_slice(), batches[1].as_slice()];
        let a = BinaryAProduct::new(plan, &self.rho, &root_point, prefix_len)
            .expect("fixture A product");
        Stage3Fixture {
            setup: typed
                .setup_coefficients(&SETUP_SEED, manifest, prefix_len)
                .expect("fixture typed setup"),
            weights: typed
                .prover_weights(&a, &root_point, alpha, batches, prefix_len)
                .expect("fixture dense weights"),
            blocks: typed
                .prover_weight_blocks(&a, &root_point, alpha, batches, prefix_len)
                .expect("fixture weight blocks"),
        }
    }
}

/// Raw input scenarios covering up to five owners, prime planes, extra zero rings
/// and the mixed ordinary group.
pub fn root_weight_scenarios() -> impl Iterator<Item = RootWeightScenario> {
    let base = Shape {
        word_width: 128,
        carriers: 8,
        columns: 2,
        owners: 1,
        prime_planes: 0,
        extra_rings: 0,
        mixed: false,
    };
    let mixed = Shape {
        mixed: true,
        ..base
    };
    [
        base,
        Shape {
            owners: 2,
            prime_planes: 5,
            ..base
        },
        Shape {
            prime_planes: 3,
            extra_rings: 5,
            ..base
        },
        Shape {
            prime_planes: 2,
            ..mixed
        },
        Shape {
            carriers: 64,
            columns: 4,
            owners: 2,
            prime_planes: 2,
            ..base
        },
        Shape {
            word_width: 8,
            carriers: 32,
            prime_planes: 5,
            extra_rings: 2,
            ..mixed
        },
    ]
    .into_iter()
    .zip(1u64..)
    .map(|(mut shape, seed)| {
        shape.owners = shape.owners.max(shape.prime_planes);
        RootWeightScenario::new(shape, seed)
    })
}

/// Small word-root manifest with the selected base profile. Larger owner counts
/// are useful for layout arithmetic only, without allocating their witnesses.
pub fn binary_word_manifest_fixture(
    field: BinaryFieldProfile,
    owners: usize,
) -> BinaryCommittedGroupManifest {
    let index = BinaryIndexMap::new(128, 8, 8, 2).expect("fixture index");
    let profiles = BinaryCommittedGroupManifest::profiles_for_field(&index, owners, 0, field)
        .expect("fixture profiles");
    let descriptor = BinarySourceDescriptor::new(
        index,
        (0..owners)
            .map(|owner| {
                let mut id = [1; 32];
                id[..8].copy_from_slice(&(owner as u64).to_le_bytes());
                id
            })
            .collect(),
        profiles,
    )
    .expect("fixture descriptor");
    BinaryCommittedGroupManifest::with_field_profile(
        descriptor,
        0,
        &binary_ordinary_profile_fixture(field),
        field,
    )
    .expect("fixture manifest")
}
