//! Real commitments and one complete staged four-group p64 session.

use super::*;
use akita_algebra::{CanonicalEncoding, EqPolynomial, Ring, Zero};
use akita_config::proof_optimized::fp64::ExtensionField as E;
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_params::{BasisMode, DecompositionParams, SisModulusProfileId};
use akita_prover::{BackendRegistry, FixedFoldRoute, SelectedProverOpeningData};
use akita_types::binary_source::{BinaryWordClaim};
use akita_types::{
    AkitaVerifierSetup, CommittedGroup, FpExtEncoding, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims,
};
use std::sync::Arc;

pub(super) type Claims = OpeningClaims<'static, E, CommittedGroup<Field>>;

pub(super) struct Opening {
    pub(super) fixture: &'static Fixture,
    pub(super) binding: GroupedTypedPrefix,
    pub(super) setup: AkitaVerifierSetup<Field>,
    pub(super) claims: Claims,
    pub(super) proof: Vec<u8>,
}

pub(super) fn ext<T: FpExtEncoding<Field>>(coordinates: [u64; 2]) -> T {
    T::from_base_slice(&coordinates.map(Field::from_u64))
}

// Explicit commitment contexts use only this source contract. The retained
// catalog and policy remain those of the receiving session; no row is planned
// or opened under these producer-only adapters.
#[derive(Clone)]
struct Producer<const BITS: u32>;

impl<const BITS: u32> CommitmentConfig for Producer<BITS> {
    type Field = Field;
    type ExtField = E;
    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        Config::RING_DIMENSION_SCHEDULE;

    fn decomposition() -> DecompositionParams {
        Config::decomposition()
    }

    fn ring_challenge_config(
        dimension: usize,
    ) -> Result<akita_challenges::FoldChallengeDistribution, AkitaError> {
        Config::ring_challenge_config(dimension)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Config::sis_modulus_profile()
    }

    fn opening_basis_range() -> (u32, u32) {
        Config::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Config::inner_basis_range()
    }

    fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
        Config::committed_source_class()
    }

    fn committed_source_contract() -> Result<akita_params::sis::CommittedSourceContract, AkitaError>
    {
        akita_params::ordinary_source::OrdinarySourceWidth::new(BITS)?
            .producer_contract(Self::sis_modulus_profile())
    }

    fn recursive_setup_planning() -> bool {
        Config::recursive_setup_planning()
    }

    fn schedule_family_name() -> &'static str {
        Config::schedule_family_name()
    }
}

pub(super) fn opening() -> &'static Opening {
    static OPENING: OnceLock<Opening> = OnceLock::new();
    OPENING.get_or_init(|| {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(build)
            .unwrap()
            .join()
            .unwrap()
    })
}

fn build() -> Opening {
    let fixture = actual_profiles::actual_fixture();
    let row = fixture.catalog.resolve_selection(fixture.bindings[0].0).unwrap();
    let requirements = NativeSetupRequirements::<Field>::from_catalog(
        &fixture.catalog, 22, 16, &fixture.bindings,
    ).unwrap();
    let setup = akita_cpu_backend::AkitaProverSetup::generate_with_capacity(
        22, 16, requirements.requirements().matrix_capacity(),
    ).unwrap();
    let binding = requirements.bind_setup(
        setup.expanded.descriptor().clone(), requirements.requirements().prefix_slot_ids(),
    ).unwrap().remove(0).1;
    assert_eq!(binding.parts().iter().map(|(range, _)| range.start).collect::<Vec<_>>(),
        [0, 140_416, 280_832, 421_248]);
    assert_eq!(binding.natural_len(), 561_664);
    assert_eq!(binding.first().n_prefix().unwrap(), 1_048_576);
    assert_eq!(binding.plan().d_payload_ranges().unwrap().last().unwrap().end, 512);
    assert_eq!(binding.plan().binary().iter().map(|binary| {
        let descriptor = binary.source().manifest.descriptor();
        descriptor.index_map().columns() * descriptor.polynomials().len()
    }).collect::<Vec<_>>(), [2, 2, 2, 2]);
    let backend = Arc::new(CpuBackend::<Field, E>::new(setup.expanded.clone()).unwrap());
    let later = requirements.requirements().prefix_slot_ids().iter()
        .filter(|id| *id != binding.first()).cloned().collect::<Vec<_>>();
    let prefixes = backend.prepare_grouped_typed_prefixes(&binding, &later).unwrap();
    let mut public = akita_types::SetupPrefixVerifierRegistry::new(
        setup.expanded.descriptor().setup_seed.clone(),
    );
    for slot in prefixes.public_slots() {
        public.insert(slot.clone()).unwrap();
    }
    let ordinary = AkitaVerifierSetup::from_parts(setup.expanded.clone(), public).unwrap();
    let verifier_setup = backend.prepare_grouped_typed_verifier_setup(
        &ordinary, &binding, &prefixes.get(binding.first()).unwrap().public,
    ).unwrap();
    binding.validate_verifier_setup(&verifier_setup).unwrap();
    let mut groups = Vec::new();
    let mut handles = Vec::new();
    let mut preceding = Vec::new();
    for binary in binding.plan().binary() {
        let source = binary.source();
        let descriptor = source.manifest.descriptor();
        let imported = backend.import_binary_source(
            source.manifest.clone(), source.carrier, vec![vec![0; 8]],
        ).unwrap();
        let frozen = (!preceding.is_empty()).then(|| {
            akita_params::PrecommittedGroupProfiles::from_profiles(preceding.clone()).unwrap()
        });
        let context = match &frozen {
            Some(frozen) => {
                GroupContext::explicit_with_precommitted_groups(&source.carrier, frozen)
            }
            None => GroupContext::explicit(&source.carrier),
        };
        let committed = backend.commit(&fixture.catalog, &imported, context).unwrap();
        let split = descriptor.index_map().claim_variable_split();
        let word = BinaryWordClaim::new(descriptor, descriptor.polynomials()[0],
            vec![[7; 16]; split.intra_carrier + split.carrier_address], [0; 16],
        ).unwrap();
        groups.push(PolynomialGroupClaims::new_binary_planes(
            committed.committed_group, Some(word), source.prime.claims().to_vec(),
        ).unwrap());
        handles.push(committed.private_handle);
        preceding.push(source.carrier);
    }
    for (index, signed) in binding.plan().ordinary().unwrap().plans().iter().enumerate() {
        let profile = signed.profile();
        let offset = signed.source_width().unwrap().offset() as i128;
        let values = (0..profile.group.num_polynomials()).map(|slot| {
            (0..1usize << profile.group.num_vars()).map(|i| {
                Field::from_i128(match (i + slot) % 4 {
                    0 => -offset, 1 => offset - 1, 2 => -1, _ => 0,
                })
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let source = backend.import_source(values.iter().map(|values| {
            DensePoly::from_field_evals(profile.group.num_vars(), values).unwrap()
        }).collect()).unwrap();
        macro_rules! commit {
            ($width:literal) => {{
                let producer = TrustedScheduleCatalog::<Producer<$width>>::new(
                    fixture.catalog.catalog().clone(), ChunkedWitnessCfg::default_non_chunked(),
                ).unwrap();
                let frozen = akita_params::PrecommittedGroupProfiles::from_profiles(
                    preceding.clone(),
                ).unwrap();
                backend.commit(&producer, &source,
                    GroupContext::explicit_with_precommitted_groups(profile, &frozen),
                ).unwrap()
            }};
        }
        let committed = match index {
            0 => commit!(2), 1 => commit!(9), 2 => commit!(18), 3 => commit!(33),
            _ => unreachable!(),
        };
        let points = [vec![ext::<E>([2, 3]); 13], vec![ext::<E>([2, 4]); 13]];
        let readings = points.each_ref().map(|point| {
            let eq = EqPolynomial::evals(point).unwrap();
            values.iter().map(|values| values.iter().zip(&eq)
                .fold(E::zero(), |sum, (&v, &weight)| {
                    sum + ext::<E>([v.to_u128_checked().unwrap() as u64, 0]) * weight
                })).collect()
        });
        groups.push(PolynomialGroupClaims::new_two_points(
            points, readings, committed.committed_group,
        ).unwrap());
        handles.push(committed.private_handle);
        preceding.push(*profile);
    }
    let claims = OpeningClaims::from_groups(groups).unwrap();
    let data = SelectedProverOpeningData::from_committed_claims::<Config>(
        claims.clone(), handles, &fixture.catalog,
    ).unwrap();
    let mut registry = BackendRegistry::<Config>::new().unwrap();
    let id = registry.register(backend.as_ref(), &prefixes).unwrap();
    let mut route = FixedFoldRoute::new(vec![id; row.schedule().num_fold_levels()]);
    let mut channel = akita_transcript::new_prover_channel(b"staged-four-group-p64");
    akita_prover::batched_prove_embedded::<Config, _>(
        setup.expanded.descriptor(), &fixture.catalog, &registry, data, &mut channel,
        BasisMode::Lagrange, &mut route,
    ).unwrap();
    let opening = Opening {
        fixture, binding, setup: verifier_setup, claims, proof: channel.narg_string().to_vec(),
    };
    opening.verify(&opening.claims, &opening.setup, &opening.proof).unwrap();
    opening
}

impl Opening {
    pub(super) fn verify(
        &self, claims: &Claims, setup: &AkitaVerifierSetup<Field>, proof: &[u8],
    ) -> Result<(), AkitaError> {
        let claims = OpeningClaims::from_groups(claims.groups().iter()
            .map(|group| group.map_commitment(|commitment| commitment)).collect())?;
        let statement = GroupBatchStatement::new(self.fixture.bindings[0].0, claims)?;
        let verifier = akita_verifier::AkitaVerifier::<Config>::for_selection(
            setup.clone(), self.fixture.catalog.clone(), self.fixture.bindings[0].0, None,
        )?;
        let _no_a_reads = akita_types::binary_source::BinaryAReadGuard::forbid();
        let mut channel = akita_transcript::new_verifier_channel(b"staged-four-group-p64", proof);
        verifier.batched_verify_embedded(&mut channel, statement, BasisMode::Lagrange,
            akita_verifier::SetupPolicy::Check(None),
        )?.check_eof(channel).map(|_| ())
    }
}
