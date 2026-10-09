use super::*;
use akita_config::proof_optimized::fp128;
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_params::SetupMatrixCapacity;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
#[cfg(feature = "disk-persistence")]
use jolt_field::Zero;

type Cfg = fp128::Dense;
type TestF = fp128::Field;

fn schedules() -> TrustedScheduleCatalog<Cfg> {
    akita_config::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("workspace schedule catalog")
}

fn requirements_at(max_num_vars: usize, max_num_batched_polys: usize) -> SetupRequirements<TestF> {
    SetupRequirements::from_catalog::<Cfg>(&schedules(), max_num_vars, max_num_batched_polys)
        .expect("workspace setup requirements")
}

#[derive(Clone)]
struct WrongModulusProfileConfig;

impl CommitmentConfig for WrongModulusProfileConfig {
    type Field = TestF;
    type ExtField = <Cfg as CommitmentConfig>::ExtField;

    fn schedule_family_name() -> &'static str {
        "test_wrong_modulus_profile"
    }

    const RING_DIMENSION_SCHEDULE: akita_config::RingDimensionSchedule =
        Cfg::RING_DIMENSION_SCHEDULE;

    fn decomposition() -> akita_params::DecompositionParams {
        Cfg::decomposition()
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        Cfg::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> akita_params::SisModulusProfileId {
        akita_params::SisModulusProfileId::Q64Offset59
    }

    fn opening_basis_range() -> (u32, u32) {
        Cfg::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Cfg::inner_basis_range()
    }

    fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
        Cfg::committed_source_class()
    }
}

#[test]
fn expanded_setup_roundtrips_and_derives_same_verifier() {
    let prover_setup = new_prover_setup::<TestF>(&requirements_at(14, 3)).unwrap();
    let capacity = SetupMatrixCapacity {
        num_field_elements: prover_setup.expanded.shared_matrix().num_field_elements() / 2,
    };
    let verifier_setup = prover_setup.to_verifier_setup(capacity).unwrap();

    let mut bytes = Vec::new();
    verifier_setup
        .expanded()
        .serialize_compressed(&mut bytes)
        .unwrap();
    let decoded = AkitaExpandedSetup::<TestF>::deserialize_compressed(&bytes[..], &()).unwrap();

    assert_eq!(decoded, verifier_setup.expanded().as_ref().clone());
    assert_eq!(decoded.descriptor().max_num_batched_polys, 3);

    decoded.check().unwrap();
    let decoded_prover = AkitaProverSetup {
        prefix_slots: akita_cpu_backend::SetupPrefixProverRegistry::new(
            decoded.descriptor().setup_seed.clone(),
        ),
        expanded: std::sync::Arc::new(decoded.clone()),
    };
    let derived_verifier = decoded_prover.to_verifier_setup(capacity).unwrap();
    assert_eq!(derived_verifier, verifier_setup);
    assert_eq!(
        verifier_setup
            .expanded()
            .shared_matrix()
            .num_field_elements(),
        capacity.num_field_elements
    );
}

#[test]
fn setup_accepts_field_coupled_presets() {
    // The D64 catalog begins at nv=14, the first singleton shape with the
    // required root and suffix folds.
    new_prover_setup::<fp128::Field>(&requirements_at(14, 1))
        .expect("fp128 dense preset should accept the default field");
}

#[test]
fn setup_rejects_a_mismatched_field_profile_before_materialization() {
    let error =
        TrustedScheduleCatalog::<WrongModulusProfileConfig>::new(schedules().catalog().clone())
            .expect_err("field modulus and SIS profile must agree before setup materialization");
    assert!(error.to_string().contains("does not match field modulus"));
}

#[test]
fn verifier_setup_prefix_registry_is_authenticated_only_by_recomputation() {
    use akita_types::{AkitaVerifierSetup, RingVec, SetupPrefixVerifierRegistry};
    use jolt_field::One;
    type Recursive = akita_config::RecursiveCommitmentConfig<fp128::Dense>;

    // Prefix commitments run ring-dispatch arithmetic that needs the
    // backend's enlarged fixture stack in debug builds.
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let catalog = akita_config::test_support::workspace_schedule_catalog::<Recursive>()
                .expect("recursive dense catalog");
            let requirements = SetupRequirements::from_catalog::<Recursive>(&catalog, 26, 1)
                .expect("recursive dense requirements");
            assert!(!requirements.prefix_slot_ids().is_empty());
            let prover = new_prover_setup::<TestF>(&requirements).expect("prover setup");
            let capacity = SetupMatrixCapacity {
                num_field_elements: prover.expanded.shared_matrix().num_field_elements(),
            };
            let derived = prover.to_verifier_setup(capacity).expect("verifier setup");
            assert!(derived.prefix_registry_authenticated());

            let mut bytes = Vec::new();
            derived.serialize_compressed(&mut bytes).unwrap();
            let decoded = AkitaVerifierSetup::<TestF>::deserialize_compressed_exact(&bytes, &())
                .expect("strict decode");
            assert!(!decoded.prefix_registry_authenticated());
            let authenticated =
                authenticate_verifier_setup_prefixes(decoded.clone()).expect("honest registry");
            assert!(authenticated.prefix_registry_authenticated());
            assert_eq!(authenticated, derived);

            let mut registry = SetupPrefixVerifierRegistry::new(
                decoded.expanded().descriptor().setup_seed.clone(),
            );
            for (index, (_, slot)) in decoded.prefix_slots().iter().enumerate() {
                let mut slot = slot.clone();
                if index == 0 {
                    let mut coeffs = slot.commitment.rows[0].coeffs().to_vec();
                    coeffs[0] += TestF::one();
                    slot.commitment.rows[0] = RingVec::from_coeffs(coeffs);
                }
                registry.insert(slot).expect("structurally valid slot");
            }
            let altered = AkitaVerifierSetup::from_parts(decoded.expanded().clone(), registry)
                .expect("structurally valid setup");
            assert!(matches!(
                authenticate_verifier_setup_prefixes(altered),
                Err(AkitaError::InvalidSetup(_))
            ));
        })
        .unwrap()
        .join()
        .unwrap();
}

#[cfg(feature = "disk-persistence")]
mod disk_persistence;
