use akita_challenges::FoldChallengeDistribution;
use akita_config::{proof_optimized::fp128::DenseBounded, CommitmentConfig};
use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::{sis::CommittedSourceClass, DecompositionParams, SisModulusProfileId};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_types::FpExtEncoding;
use jolt_field::{AdditiveGroup, Fold, PseudoMersenne, Ring, Unreduced, WithCommitAccumulator};

#[derive(Clone)]
pub(crate) struct P128;
#[derive(Clone)]
pub(crate) struct P64;

pub(crate) trait MeasurementConfig:
    CommitmentConfig<
    Field: AkitaDeserialize<Context = ()>
               + Unreduced<Wide: From<Self::Field> + AdditiveGroup>
               + PseudoMersenne
               + WithCommitAccumulator
               + 'static,
    ExtField: FpExtEncoding<Self::Field> + Unreduced + Fold + Ring + AkitaSerialize + 'static,
>
{
    const PROFILE: BinaryFieldProfile;

    #[cfg(test)]
    type Attacker: akita_prover::backend::ProverBackend<
            Self::Field,
            Self::ExtField,
            CommitmentHandle = akita_cpu_backend::CommitmentHandle<Self::Field, Self::ExtField>,
        > + 'static;

    #[cfg(test)]
    fn attacker(
        backend: std::sync::Arc<akita_cpu_backend::CpuBackend<Self::Field, Self::ExtField>>,
        seed: [u8; 32],
        words: Vec<u128>,
    ) -> Option<Self::Attacker>;
}

impl MeasurementConfig for P128 {
    const PROFILE: BinaryFieldProfile = BinaryFieldProfile::P128;

    #[cfg(test)]
    type Attacker = akita_cpu_backend::CpuBackend<Self::Field, Self::ExtField>;

    #[cfg(test)]
    fn attacker(
        _backend: std::sync::Arc<akita_cpu_backend::CpuBackend<Self::Field, Self::ExtField>>,
        _seed: [u8; 32],
        _words: Vec<u128>,
    ) -> Option<Self::Attacker> {
        None
    }
}

impl MeasurementConfig for P64 {
    const PROFILE: BinaryFieldProfile = BinaryFieldProfile::P64;

    #[cfg(test)]
    type Attacker = crate::p64::fault_backend::FaultBackend;

    #[cfg(test)]
    fn attacker(
        backend: std::sync::Arc<akita_cpu_backend::CpuBackend<Self::Field, Self::ExtField>>,
        seed: [u8; 32],
        words: Vec<u128>,
    ) -> Option<Self::Attacker> {
        crate::p64::FAULT
            .with(|selected| selected.get())
            .map(|fault| crate::p64::fault_backend::FaultBackend::new(backend, seed, words, fault))
    }
}

/// Ordinary suffix for an explicitly selected experimental binary fixture.
/// Both fields deliberately share the P128 search dimensions and basis ranges
/// to compare the root migration with a fixed degree-64 suffix search space.
/// P64 still uses Q64Offset59 and 64-bit opening bounds in every admission audit.
/// This is a controlled research preset, not the canonical fp64::Dense benchmark.
macro_rules! impl_config {
    ($name:ident, $base:ty, $extension:ty) => {
        impl CommitmentConfig for $name {
            type Field = $base;
            type ExtField = $extension;
            const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
                DenseBounded::RING_DIMENSION_SCHEDULE;

            fn decomposition() -> DecompositionParams {
                DecompositionParams {
                    log_basis: DenseBounded::decomposition().log_basis,
                    log_commit_bound: 5,
                    log_open_bound: Some(match Self::PROFILE {
                        BinaryFieldProfile::P128 => 128,
                        BinaryFieldProfile::P64 => 64,
                    }),
                }
            }
            fn ring_challenge_config(
                dimension: usize,
            ) -> Result<FoldChallengeDistribution, AkitaError> {
                DenseBounded::ring_challenge_config(dimension)
            }
            fn sis_modulus_profile() -> SisModulusProfileId {
                match Self::PROFILE {
                    BinaryFieldProfile::P128 => SisModulusProfileId::Q128Offset275,
                    BinaryFieldProfile::P64 => SisModulusProfileId::Q64Offset59,
                }
            }
            fn opening_basis_range() -> (u32, u32) {
                DenseBounded::opening_basis_range()
            }
            fn inner_basis_range() -> (u32, u32) {
                DenseBounded::inner_basis_range()
            }
            fn committed_source_class() -> CommittedSourceClass {
                DenseBounded::committed_source_class()
            }
            fn recursive_setup_planning() -> bool {
                true
            }
            fn schedule_family_name() -> &'static str {
                match Self::PROFILE {
                    BinaryFieldProfile::P128 => "binary_source_full_size_fixture",
                    BinaryFieldProfile::P64 => "binary_source_p64_research",
                }
            }
        }
    };
}

impl_config!(
    P128,
    jolt_field::Prime128Offset275,
    jolt_field::Prime128Offset275
);
impl_config!(
    P64,
    jolt_field::Prime64Offset59,
    jolt_field::Ext2<jolt_field::Prime64Offset59>
);
