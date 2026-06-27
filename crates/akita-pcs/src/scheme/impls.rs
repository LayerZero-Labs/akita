//! Per-preset D-free PCS methods plus typed dispatch impls for [`super::AkitaCommitmentScheme`].
//!
//! Rust cannot use `Cfg::D` in const-generic positions on a single blanket impl, so each
//! shipped `CommitmentConfig` preset gets a macro-expanded impl with a literal root `D`.

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::tensor_verifier;
macro_rules! impl_akita_commitment_scheme {
    ($cfg:ty, $field:ty, $ext_field:ty, $d:expr) => {
        impl $crate::scheme::AkitaCommitmentScheme<$cfg>
        where
            $cfg: akita_config::CommitmentConfig<Field = $field, ExtField = $ext_field>,
            $field: akita_field::FieldCore
                + akita_field::CanonicalField
                + akita_field::RandomSampling
                + akita_field::unreduced::HasWide
                + akita_field::HalvingField
                + akita_field::FromPrimitiveInt
                + akita_field::PseudoMersenneField
                + akita_serialization::Valid
                + akita_serialization::AkitaSerialize,
            $ext_field: akita_types::FpExtEncoding<$field>,
            $ext_field: akita_field::FrobeniusExtField<$field>
                + akita_field::FromPrimitiveInt
                + akita_field::unreduced::HasUnreducedOps
                + akita_field::unreduced::HasOptimizedFold
                + akita_serialization::AkitaSerialize,
        {
            /// Build prover setup without exposing the root ring dimension as a public type parameter.
            ///
            /// # Errors
            ///
            /// Returns an error if the requested capacity, field tower, or generated setup is invalid.
            pub fn setup_prover(
                max_num_vars: usize,
                max_num_polys_per_commitment_group: usize,
            ) -> Result<akita_prover::AkitaProverSetup<$field>, akita_field::AkitaError> {
                <Self as akita_prover::TypedCommitmentProver<$field, $d>>::setup_prover(
                    max_num_vars,
                    max_num_polys_per_commitment_group,
                )
            }

            /// Build recursive prover setup without exposing the root ring dimension as a public type parameter.
            ///
            /// # Errors
            ///
            /// Returns an error if base setup construction or recursive setup-prefix population fails.
            pub fn setup_prover_recursion(
                max_num_vars: usize,
                max_num_polys_per_commitment_group: usize,
            ) -> Result<akita_prover::AkitaProverSetup<$field>, akita_field::AkitaError> {
                <Self as akita_prover::TypedCommitmentProver<$field, $d>>::setup_prover_recursion(
                    max_num_vars,
                    max_num_polys_per_commitment_group,
                )
            }

            /// Derive verifier setup from prover setup.
            #[must_use]
            pub fn setup_verifier(
                setup: &akita_prover::AkitaProverSetup<$field>,
            ) -> akita_types::AkitaVerifierSetup<$field> {
                <Self as akita_prover::TypedCommitmentProver<$field, $d>>::setup_verifier(setup)
            }

            /// Commit a single opening-point bundle without caller-visible root `D`.
            ///
            /// # Errors
            ///
            /// Returns an error when setup/parameter constraints are not satisfied.
            pub fn commit<P, B>(
                setup: &akita_prover::AkitaProverSetup<$field>,
                polys: &[P],
                stack: &akita_prover::compute::UniformProverStack<'_, $field, B>,
            ) -> Result<
                (
                    akita_types::FlatRingVec<$field>,
                    akita_types::AkitaCommitmentHint<$field>,
                ),
                akita_field::AkitaError,
            >
            where
                $field: akita_field::FromPrimitiveInt
                    + akita_field::unreduced::HasWide
                    + akita_field::RandomSampling
                    + 'static,
                <$field as akita_field::unreduced::HasWide>::Wide:
                    From<$field> + akita_field::unreduced::ReduceTo<$field>,
                B: akita_prover::compute::ComputeBackendSetup<$field>,
                P: akita_prover::compute::RootCommitPoly<$field, $d>,
                B: akita_prover::compute::RootCommitBackend<$field, P, $ext_field, $d>,
            {
                let (commitment, hint) = <Self as akita_prover::TypedCommitmentProver<
                    $field,
                    $d,
                >>::commit(setup, polys, stack)?;
                Ok((akita_types::FlatRingVec::from_commitment(&commitment), hint))
            }

            /// Commit the polynomial bundle used by a batched prove without caller-visible root `D`.
            ///
            /// # Errors
            ///
            /// Returns an error if input validation, layout selection, or any per-point commitment fails.
            pub fn batched_commit<P, B>(
                setup: &akita_prover::AkitaProverSetup<$field>,
                polys: &[P],
                stack: &akita_prover::compute::UniformProverStack<'_, $field, B>,
            ) -> Result<
                (
                    akita_types::FlatRingVec<$field>,
                    akita_types::AkitaCommitmentHint<$field>,
                ),
                akita_field::AkitaError,
            >
            where
                $field: akita_field::FromPrimitiveInt
                    + akita_field::unreduced::HasWide
                    + akita_field::RandomSampling
                    + 'static,
                <$field as akita_field::unreduced::HasWide>::Wide:
                    From<$field> + akita_field::unreduced::ReduceTo<$field>,
                B: akita_prover::compute::ComputeBackendSetup<$field>,
                P: akita_prover::compute::RootCommitPoly<$field, $d>,
                B: akita_prover::compute::RootCommitBackend<$field, P, $ext_field, $d>,
            {
                let (commitment, hint) = <Self as akita_prover::TypedCommitmentProver<
                    $field,
                    $d,
                >>::batched_commit(setup, polys, stack)?;
                Ok((akita_types::FlatRingVec::from_commitment(&commitment), hint))
            }

            /// Commit one standalone one-hot commitment group without caller-visible root `D`.
            ///
            /// # Errors
            ///
            /// Returns an error if the group is empty, dense, exceeds setup capacity, or cannot be planned.
            pub fn commit_group<P, B>(
                setup: &akita_prover::AkitaProverSetup<$field>,
                polys: &[P],
                stack: &akita_prover::compute::UniformProverStack<'_, $field, B>,
            ) -> Result<
                akita_prover::CommittedGroupHandle<
                    akita_types::FlatRingVec<$field>,
                    akita_types::AkitaCommitmentHint<$field>,
                >,
                akita_field::AkitaError,
            >
            where
                $field: akita_field::FromPrimitiveInt
                    + akita_field::unreduced::HasWide
                    + akita_field::RandomSampling
                    + 'static,
                <$field as akita_field::unreduced::HasWide>::Wide:
                    From<$field> + akita_field::unreduced::ReduceTo<$field>,
                P: akita_prover::compute::RootCommitPoly<$field, $d>,
                B: akita_prover::compute::RootCommitBackend<$field, P, $ext_field, $d>,
            {
                let group =
                    <Self as akita_prover::TypedCommitmentProver<$field, $d>>::commit_group(
                        setup, polys, stack,
                    )?;
                Ok(akita_prover::CommittedGroupHandle {
                    schedule: group.schedule,
                    commitment: akita_types::FlatRingVec::from_commitment(&group.commitment),
                    hint: group.hint,
                })
            }

            /// Produce a fused batched opening proof without caller-visible root `D`.
            ///
            /// # Errors
            ///
            /// Returns an error if any opening point is invalid or proof generation fails.
            #[allow(clippy::too_many_arguments)]
            pub fn batched_prove<'a, T, P, B>(
                setup: &akita_prover::AkitaProverSetup<$field>,
                claims: akita_prover::ProverOpeningBatch<'a, $ext_field, P, $field>,
                stacks: &'a impl akita_prover::compute::LevelProveStacks<
                    'a,
                    $field,
                    Commit = B,
                    Opening = B,
                    Tensor = B,
                    RingSwitch = B,
                >,
                transcript: &mut T,
                basis: akita_types::BasisMode,
                setup_contribution_mode: akita_types::SetupContributionMode,
            ) -> Result<akita_types::AkitaBatchedProof<$field, $ext_field>, akita_field::AkitaError>
            where
                T: akita_transcript::Transcript<$field>
                    + akita_prover::ProverTranscriptGrind<$field>,
                $field: akita_field::FromPrimitiveInt
                    + akita_field::unreduced::HasWide
                    + akita_field::RandomSampling
                    + 'static,
                <$field as akita_field::unreduced::HasWide>::Wide: From<$field>
                    + akita_field::unreduced::ReduceTo<$field>
                    + akita_field::AdditiveGroup,
                B: akita_prover::compute::ComputeBackendSetup<$field>,
                P: akita_prover::compute::RootProvePoly<$field, $d>,
                B: akita_prover::compute::RootProveFlowBackend<$field, P, $ext_field, $d>
                    + akita_prover::compute::RecursiveWitnessProveFlowBackend<$field, $ext_field>
                    + 'a,
                <B as akita_prover::compute::ComputeBackendSetup<$field>>::PreparedSetup: 'a,
            {
                let claims = claims.into_typed::<$d>()?;
                <Self as akita_prover::TypedCommitmentProver<$field, $d>>::batched_prove(
                    setup,
                    claims,
                    stacks,
                    transcript,
                    basis,
                    setup_contribution_mode,
                )
            }

            /// Verify a fused batched opening proof without caller-visible root `D`.
            ///
            /// # Errors
            ///
            /// Returns an error when verification fails.
            pub fn batched_verify<T: akita_transcript::Transcript<$field>>(
                proof: &akita_types::AkitaBatchedProof<$field, $ext_field>,
                setup: &akita_types::AkitaVerifierSetup<$field>,
                transcript: &mut T,
                claims: akita_types::VerifierOpeningBatch<
                    '_,
                    $ext_field,
                    &akita_types::FlatRingVec<$field>,
                >,
                basis: akita_types::BasisMode,
                setup_contribution_mode: akita_types::SetupContributionMode,
            ) -> Result<(), akita_field::AkitaError> {
                let typed_commitments = claims
                    .groups
                    .iter()
                    .map(|group| group.commitment.try_to_ring_commitment::<$d>())
                    .collect::<Result<Vec<_>, akita_field::AkitaError>>()?;
                let typed_groups = claims
                    .groups
                    .iter()
                    .zip(typed_commitments.iter())
                    .map(|(group, commitment)| akita_types::CommitmentGroup {
                        claims: group.claims.clone(),
                        commitment,
                    })
                    .collect::<Vec<_>>();
                let typed_claims = akita_types::VerifierOpeningBatch::from_shape_and_groups(
                    claims.point.clone(),
                    claims.shape.clone(),
                    typed_groups,
                )?;
                <Self as akita_verifier::TypedCommitmentVerifier<$field, $d>>::batched_verify(
                    proof,
                    setup,
                    transcript,
                    typed_claims,
                    basis,
                    setup_contribution_mode,
                )
            }

            /// Protocol identifier.
            #[must_use]
            pub fn protocol_name() -> &'static [u8] {
                <Self as akita_verifier::TypedCommitmentVerifier<$field, $d>>::protocol_name()
            }
        }

        impl akita_prover::TypedCommitmentProver<$field, $d>
            for $crate::scheme::AkitaCommitmentScheme<$cfg>
        where
            $cfg: akita_config::CommitmentConfig<Field = $field, ExtField = $ext_field>,
            $field: akita_field::FieldCore
                + akita_field::CanonicalField
                + akita_field::RandomSampling
                + akita_field::unreduced::HasWide
                + akita_field::HalvingField
                + akita_field::FromPrimitiveInt
                + akita_field::PseudoMersenneField
                + akita_serialization::Valid
                + akita_serialization::AkitaSerialize,
            $ext_field: akita_types::FpExtEncoding<$field>,
            $ext_field: akita_field::FrobeniusExtField<$field>
                + akita_field::FromPrimitiveInt
                + akita_field::unreduced::HasUnreducedOps
                + akita_field::unreduced::HasOptimizedFold
                + akita_serialization::AkitaSerialize,
        {
            type ProverSetup = akita_prover::AkitaProverSetup<$field>;
            type VerifierSetup = akita_types::AkitaVerifierSetup<$field>;
            type Commitment = akita_types::RingCommitment<$field, $d>;
            type ExtField = $ext_field;
            type CommitHint = akita_types::AkitaCommitmentHint<$field>;
            type BatchedProof = akita_types::AkitaBatchedProof<$field, $ext_field>;

            fn setup_prover(
                max_num_vars: usize,
                max_num_polys_per_commitment_group: usize,
            ) -> Result<Self::ProverSetup, akita_field::AkitaError> {
                akita_types::validate_ring_subfield_role::<$field, $ext_field, $d>(
                    "extension field",
                )?;
                akita_setup::new_prover_setup::<$field, $cfg>(
                    max_num_vars,
                    max_num_polys_per_commitment_group,
                )
            }

            fn setup_prover_recursion(
                max_num_vars: usize,
                max_num_polys_per_commitment_group: usize,
            ) -> Result<Self::ProverSetup, akita_field::AkitaError> {
                akita_types::validate_ring_subfield_role::<$field, $ext_field, $d>(
                    "extension field",
                )?;
                akita_setup::new_prover_setup_recursion::<$field, $cfg>(
                    max_num_vars,
                    max_num_polys_per_commitment_group,
                )
            }

            fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
                setup
                    .verifier_setup()
                    .expect("prover setup must convert to verifier setup")
            }

            #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit")]
            fn commit<P, B>(
                setup: &Self::ProverSetup,
                polys: &[P],
                stack: &akita_prover::compute::UniformProverStack<'_, $field, B>,
            ) -> Result<(Self::Commitment, Self::CommitHint), akita_field::AkitaError>
            where
                $field: akita_field::FromPrimitiveInt
                    + akita_field::unreduced::HasWide
                    + akita_field::RandomSampling
                    + 'static,
                <$field as akita_field::unreduced::HasWide>::Wide:
                    From<$field> + akita_field::unreduced::ReduceTo<$field>,
                B: akita_prover::compute::ComputeBackendSetup<$field>,
                P: akita_prover::compute::RootCommitPoly<$field, $d>,
                B: akita_prover::compute::RootCommitBackend<$field, P, $ext_field, $d>,
            {
                setup.ensure_compile_time_ring_dim::<$d>()?;
                akita_prover::commit::<$cfg, $d, P, B>(polys, setup.expanded.as_ref(), stack)
            }

            #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_commit")]
            fn batched_commit<P, B>(
                setup: &Self::ProverSetup,
                polys: &[P],
                stack: &akita_prover::compute::UniformProverStack<'_, $field, B>,
            ) -> Result<(Self::Commitment, Self::CommitHint), akita_field::AkitaError>
            where
                $field: akita_field::FromPrimitiveInt
                    + akita_field::unreduced::HasWide
                    + akita_field::RandomSampling
                    + 'static,
                <$field as akita_field::unreduced::HasWide>::Wide:
                    From<$field> + akita_field::unreduced::ReduceTo<$field>,
                B: akita_prover::compute::ComputeBackendSetup<$field>,
                P: akita_prover::compute::RootCommitPoly<$field, $d>,
                B: akita_prover::compute::RootCommitBackend<$field, P, $ext_field, $d>,
            {
                setup.ensure_compile_time_ring_dim::<$d>()?;
                akita_prover::batched_commit::<$cfg, $d, P, B>(
                    polys,
                    setup.expanded.as_ref(),
                    stack,
                )
            }

            #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::commit_group")]
            fn commit_group<P, B>(
                setup: &Self::ProverSetup,
                polys: &[P],
                stack: &akita_prover::compute::UniformProverStack<'_, $field, B>,
            ) -> Result<
                akita_prover::CommittedGroupHandle<Self::Commitment, Self::CommitHint>,
                akita_field::AkitaError,
            >
            where
                $field: akita_field::FromPrimitiveInt
                    + akita_field::unreduced::HasWide
                    + akita_field::RandomSampling
                    + 'static,
                <$field as akita_field::unreduced::HasWide>::Wide:
                    From<$field> + akita_field::unreduced::ReduceTo<$field>,
                P: akita_prover::compute::RootCommitPoly<$field, $d>,
                B: akita_prover::compute::RootCommitBackend<$field, P, $ext_field, $d>,
            {
                akita_prover::commit_group::<$cfg, $d, P, B>(polys, setup.expanded.as_ref(), stack)
            }

            #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_prove")]
            fn batched_prove<'a, T, P, B>(
                setup: &Self::ProverSetup,
                claims: akita_prover::TypedProverOpeningBatch<'a, Self::ExtField, P, $field, $d>,
                stacks: &'a impl akita_prover::compute::LevelProveStacks<
                    'a,
                    $field,
                    Commit = B,
                    Opening = B,
                    Tensor = B,
                    RingSwitch = B,
                >,
                transcript: &mut T,
                basis: akita_types::BasisMode,
                setup_contribution_mode: akita_types::SetupContributionMode,
            ) -> Result<Self::BatchedProof, akita_field::AkitaError>
            where
                T: akita_transcript::Transcript<$field>
                    + akita_prover::ProverTranscriptGrind<$field>,
                $field: akita_field::FromPrimitiveInt
                    + akita_field::unreduced::HasWide
                    + akita_field::RandomSampling
                    + 'static,
                <$field as akita_field::unreduced::HasWide>::Wide: From<$field>
                    + akita_field::unreduced::ReduceTo<$field>
                    + akita_field::AdditiveGroup,
                B: akita_prover::compute::ComputeBackendSetup<$field>,
                P: akita_prover::compute::RootProvePoly<$field, $d>,
                B: akita_prover::compute::RootProveFlowBackend<$field, P, $ext_field, $d>
                    + akita_prover::compute::RecursiveWitnessProveFlowBackend<$field, $ext_field>
                    + 'a,
                <B as akita_prover::compute::ComputeBackendSetup<$field>>::PreparedSetup: 'a,
            {
                let t_prove_total = std::time::Instant::now();
                akita_types::validate_ring_subfield_role::<$field, $ext_field, $d>(
                    "extension field",
                )?;
                setup.ensure_compile_time_ring_dim::<$d>()?;
                let proof = akita_prover::batched_prove::<$cfg, T, P, B, B, B, B, $d>(
                    &setup.expanded,
                    &setup.prefix_slots,
                    stacks,
                    claims,
                    transcript,
                    basis,
                    setup_contribution_mode,
                )?;

                tracing::info!(
                    levels = proof.num_fold_levels() + usize::from(proof.root.as_fold().is_some()),
                    elapsed_s = t_prove_total.elapsed().as_secs_f64(),
                    "akita batched prove complete"
                );

                Ok(proof)
            }
        }

        impl akita_verifier::CommitmentVerifier<$field>
            for $crate::scheme::AkitaCommitmentScheme<$cfg>
        where
            $cfg: akita_config::CommitmentConfig<Field = $field, ExtField = $ext_field>,
            $field: akita_field::FieldCore
                + akita_field::CanonicalField
                + akita_field::RandomSampling
                + akita_field::unreduced::HasWide
                + akita_field::HalvingField
                + akita_field::FromPrimitiveInt
                + akita_field::PseudoMersenneField
                + akita_serialization::Valid
                + akita_serialization::AkitaSerialize,
            $ext_field: akita_types::FpExtEncoding<$field>,
            $ext_field: akita_field::FrobeniusExtField<$field>
                + akita_field::FromPrimitiveInt
                + akita_serialization::AkitaSerialize,
        {
            type VerifierSetup = akita_types::AkitaVerifierSetup<$field>;
            type Commitment = akita_types::FlatRingVec<$field>;
            type ExtField = $ext_field;
            type BatchedProof = akita_types::AkitaBatchedProof<$field, $ext_field>;

            #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_verify")]
            fn batched_verify<T: akita_transcript::Transcript<$field>>(
                proof: &Self::BatchedProof,
                setup: &Self::VerifierSetup,
                transcript: &mut T,
                claims: akita_types::VerifierOpeningBatch<'_, Self::ExtField, &Self::Commitment>,
                basis: akita_types::BasisMode,
                setup_contribution_mode: akita_types::SetupContributionMode,
            ) -> Result<(), akita_field::AkitaError> {
                $crate::scheme::AkitaCommitmentScheme::<$cfg>::batched_verify(
                    proof,
                    setup,
                    transcript,
                    claims,
                    basis,
                    setup_contribution_mode,
                )
            }

            fn protocol_name() -> &'static [u8] {
                b"Akita"
            }
        }

        impl akita_verifier::TypedCommitmentVerifier<$field, $d>
            for $crate::scheme::AkitaCommitmentScheme<$cfg>
        where
            $cfg: akita_config::CommitmentConfig<Field = $field, ExtField = $ext_field>,
            $field: akita_field::FieldCore
                + akita_field::CanonicalField
                + akita_field::RandomSampling
                + akita_field::unreduced::HasWide
                + akita_field::HalvingField
                + akita_field::FromPrimitiveInt
                + akita_field::PseudoMersenneField
                + akita_serialization::Valid
                + akita_serialization::AkitaSerialize,
            $ext_field: akita_types::FpExtEncoding<$field>,
            $ext_field: akita_field::FrobeniusExtField<$field>
                + akita_field::FromPrimitiveInt
                + akita_serialization::AkitaSerialize,
        {
            type VerifierSetup = akita_types::AkitaVerifierSetup<$field>;
            type Commitment = akita_types::RingCommitment<$field, $d>;
            type ExtField = $ext_field;
            type BatchedProof = akita_types::AkitaBatchedProof<$field, $ext_field>;

            #[tracing::instrument(skip_all, name = "AkitaCommitmentScheme::batched_verify")]
            fn batched_verify<T: akita_transcript::Transcript<$field>>(
                proof: &Self::BatchedProof,
                setup: &Self::VerifierSetup,
                transcript: &mut T,
                claims: akita_types::VerifierOpeningBatch<'_, Self::ExtField, &Self::Commitment>,
                basis: akita_types::BasisMode,
                setup_contribution_mode: akita_types::SetupContributionMode,
            ) -> Result<(), akita_field::AkitaError> {
                let t_verify_akita = std::time::Instant::now();
                akita_types::validate_ring_subfield_role::<$field, $ext_field, $d>(
                    "extension field",
                )?;
                setup.ensure_root_ring_dim($d)?;
                akita_verifier::batched_verify::<$cfg, T, $d>(
                    proof,
                    setup,
                    transcript,
                    claims,
                    basis,
                    setup_contribution_mode,
                )?;

                tracing::info!(
                    levels = proof.num_fold_levels() + 1,
                    elapsed_s = t_verify_akita.elapsed().as_secs_f64(),
                    "akita batched verify complete"
                );

                Ok(())
            }

            fn protocol_name() -> &'static [u8] {
                b"Akita"
            }
        }
    };
}

impl_akita_commitment_scheme!(fp128::D128Full, fp128::Field, fp128::Field, 128);
impl_akita_commitment_scheme!(fp128::D128OneHot, fp128::Field, fp128::Field, 128);
impl_akita_commitment_scheme!(fp128::D64Full, fp128::Field, fp128::Field, 64);
impl_akita_commitment_scheme!(fp128::D64OneHot, fp128::Field, fp128::Field, 64);
impl_akita_commitment_scheme!(fp128::D64OneHotTiered, fp128::Field, fp128::Field, 64);
impl_akita_commitment_scheme!(fp128::D32Full, fp128::Field, fp128::Field, 32);
impl_akita_commitment_scheme!(fp128::D32OneHot, fp128::Field, fp128::Field, 32);

impl_akita_commitment_scheme!(fp32::D64Full, fp32::Field, fp32::ExtensionField, 64);
impl_akita_commitment_scheme!(fp32::D64OneHot, fp32::Field, fp32::ExtensionField, 64);
impl_akita_commitment_scheme!(fp32::D128Full, fp32::Field, fp32::ExtensionField, 128);
impl_akita_commitment_scheme!(fp32::D128OneHot, fp32::Field, fp32::ExtensionField, 128);
impl_akita_commitment_scheme!(fp32::D256Full, fp32::Field, fp32::ExtensionField, 256);
impl_akita_commitment_scheme!(fp32::D256OneHot, fp32::Field, fp32::ExtensionField, 256);

impl_akita_commitment_scheme!(fp64::D128Full, fp64::Field, fp64::ExtensionField, 128);
impl_akita_commitment_scheme!(fp64::D128OneHot, fp64::Field, fp64::ExtensionField, 128);
impl_akita_commitment_scheme!(fp64::D256Full, fp64::Field, fp64::ExtensionField, 256);
impl_akita_commitment_scheme!(fp64::D256OneHot, fp64::Field, fp64::ExtensionField, 256);
impl_akita_commitment_scheme!(fp64::D64Full, fp64::Field, fp64::ExtensionField, 64);
impl_akita_commitment_scheme!(fp64::D64OneHot, fp64::Field, fp64::ExtensionField, 64);
impl_akita_commitment_scheme!(fp64::D32Full, fp64::Field, fp64::ExtensionField, 32);
impl_akita_commitment_scheme!(fp64::D32OneHot, fp64::Field, fp64::ExtensionField, 32);

impl_akita_commitment_scheme!(
    tensor_verifier::fp128::D64OneHotTensor,
    fp128::Field,
    fp128::Field,
    64
);
